"""
错误监控工具
提供错误统计、告警和分析功能
"""
import time
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .error_codes import ErrorCode, ErrorCategory
from .exceptions import BaseBusinessException

logger = logging.getLogger(__name__)

@dataclass
class ErrorMetric:
    """错误指标"""
    error_code: int
    category: str
    count: int = 0
    last_occurrence: Optional[datetime] = None
    first_occurrence: Optional[datetime] = None
    recent_traces: deque = field(default_factory=lambda: deque(maxlen=10))

@dataclass
class ErrorAlert:
    """错误告警"""
    error_code: int
    category: str
    threshold: int
    time_window: int  # 秒
    message: str
    enabled: bool = True

class ErrorMonitor:
    """错误监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.error_metrics: Dict[int, ErrorMetric] = {}
        self.error_history: deque = deque(maxlen=max_history)
        self.alerts: List[ErrorAlert] = []
        self.alert_cooldown: Dict[int, datetime] = {}
        
        # 设置默认告警规则
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """设置默认告警规则"""
        self.alerts = [
            ErrorAlert(1000, "SYSTEM", 5, 300, "系统错误频发"),
            ErrorAlert(5001, "RATE_LIMIT", 10, 60, "限流异常频发"),
            ErrorAlert(2000, "BUSINESS", 10, 300, "业务异常频发"),
            ErrorAlert(6000, "EXTERNAL", 5, 180, "外部服务异常频发"),
        ]
    
    def record_error(self, exception: BaseBusinessException, extra_context: Optional[Dict[str, Any]] = None):
        """记录错误"""
        error_code = exception.error_code.code
        category = exception.error_code.category.value
        current_time = datetime.now()
        
        # 更新错误指标
        if error_code not in self.error_metrics:
            self.error_metrics[error_code] = ErrorMetric(
                error_code=error_code,
                category=category,
                first_occurrence=current_time
            )
        
        metric = self.error_metrics[error_code]
        metric.count += 1
        metric.last_occurrence = current_time
        metric.recent_traces.append(exception.trace_id)
        
        # 添加到历史记录
        error_record = {
            'timestamp': current_time,
            'error_code': error_code,
            'category': category,
            'trace_id': exception.trace_id,
            'message': exception.message,
            'user_message': exception.user_message,
            'context': exception.context.__dict__ if exception.context else {},
            'extra_context': extra_context or {}
        }
        self.error_history.append(error_record)
        
        # 检查告警
        self._check_alerts(error_code, category)
        
        # 记录监控日志
        logger.info(
            f"错误监控记录: {error_code} - {exception.message}",
            extra={
                'error_code': error_code,
                'category': category,
                'trace_id': exception.trace_id,
                'error_count': metric.count,
                'monitoring': True
            }
        )
    
    def _check_alerts(self, error_code: int, category: str):
        """检查告警条件"""
        current_time = datetime.now()
        
        for alert in self.alerts:
            if not alert.enabled:
                continue
                
            # 检查是否匹配告警规则
            if alert.error_code != error_code and alert.category != category:
                continue
            
            # 检查冷却时间
            cooldown_key = f"{alert.error_code}_{alert.category}"
            if cooldown_key in self.alert_cooldown:
                if current_time - self.alert_cooldown[cooldown_key] < timedelta(minutes=5):
                    continue
            
            # 计算时间窗口内的错误数量
            window_start = current_time - timedelta(seconds=alert.time_window)
            error_count = sum(
                1 for record in self.error_history
                if (record['timestamp'] >= window_start and 
                    (record['error_code'] == alert.error_code or 
                     record['category'] == alert.category))
            )
            
            # 触发告警
            if error_count >= alert.threshold:
                self._trigger_alert(alert, error_count, current_time)
                self.alert_cooldown[cooldown_key] = current_time
    
    def _trigger_alert(self, alert: ErrorAlert, error_count: int, current_time: datetime):
        """触发告警"""
        alert_msg = (f"告警: {alert.message} - "
                    f"在{alert.time_window}秒内发生{error_count}次错误 "
                    f"(阈值: {alert.threshold})")
        
        logger.warning(
            alert_msg,
            extra={
                'alert': True,
                'error_code': alert.error_code,
                'category': alert.category,
                'error_count': error_count,
                'threshold': alert.threshold,
                'time_window': alert.time_window
            }
        )
        
        # 这里可以集成外部告警系统，如邮件、钉钉、Slack等
        # self._send_external_alert(alert_msg)
    
    def get_error_statistics(self, time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """获取错误统计信息"""
        if time_range:
            cutoff_time = datetime.now() - time_range
            filtered_history = [
                record for record in self.error_history
                if record['timestamp'] >= cutoff_time
            ]
        else:
            filtered_history = list(self.error_history)
        
        # 按类别统计
        category_stats = defaultdict(int)
        error_code_stats = defaultdict(int)
        
        for record in filtered_history:
            category_stats[record['category']] += 1
            error_code_stats[record['error_code']] += 1
        
        # 计算错误率
        total_errors = len(filtered_history)
        error_rate_by_category = {
            category: (count / total_errors * 100) if total_errors > 0 else 0
            for category, count in category_stats.items()
        }
        
        return {
            'total_errors': total_errors,
            'time_range': str(time_range) if time_range else 'all_time',
            'category_stats': dict(category_stats),
            'error_code_stats': dict(error_code_stats),
            'error_rate_by_category': error_rate_by_category,
            'top_errors': sorted(
                error_code_stats.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            'recent_errors': list(self.error_history)[-10:] if self.error_history else []
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        recent_errors = self.get_error_statistics(timedelta(minutes=5))
        hourly_errors = self.get_error_statistics(timedelta(hours=1))
        
        # 计算健康评分 (0-100)
        health_score = 100
        
        # 系统错误严重扣分
        system_errors = recent_errors['category_stats'].get('SYSTEM', 0)
        health_score -= system_errors * 10
        
        # 限流错误适度扣分
        rate_limit_errors = recent_errors['category_stats'].get('RATE_LIMIT', 0)
        health_score -= rate_limit_errors * 2
        
        # 业务错误轻微扣分
        business_errors = recent_errors['category_stats'].get('BUSINESS', 0)
        health_score -= business_errors * 1
        
        health_score = max(0, min(100, health_score))
        
        # 确定健康状态
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "warning"
        else:
            status = "critical"
        
        return {
            'status': status,
            'health_score': health_score,
            'recent_errors': recent_errors['total_errors'],
            'hourly_errors': hourly_errors['total_errors'],
            'last_check': datetime.now(),
            'active_alerts': len([a for a in self.alerts if a.enabled])
        }
    
    def add_custom_alert(self, error_code: int, category: str, threshold: int, 
                        time_window: int, message: str) -> bool:
        """添加自定义告警规则"""
        try:
            alert = ErrorAlert(error_code, category, threshold, time_window, message)
            self.alerts.append(alert)
            logger.info(f"添加自定义告警规则: {message}")
            return True
        except Exception as e:
            logger.error(f"添加告警规则失败: {e}")
            return False
    
    def clear_history(self, before_time: Optional[datetime] = None):
        """清理历史记录"""
        if before_time is None:
            before_time = datetime.now() - timedelta(days=7)  # 默认保留7天
        
        original_count = len(self.error_history)
        self.error_history = deque(
            [record for record in self.error_history if record['timestamp'] > before_time],
            maxlen=self.max_history
        )
        
        cleared_count = original_count - len(self.error_history)
        if cleared_count > 0:
            logger.info(f"清理历史错误记录: {cleared_count}条")

# 全局错误监控实例
error_monitor = ErrorMonitor() 