"""
请求追踪器
记录请求ID、模型使用、成本和失败原因
"""
import uuid
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RequestInfo:
    """请求信息"""
    request_id: str
    user_id: Optional[str]
    thread_id: Optional[str]
    question: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "processing"  # processing, success, failed
    selected_model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_cost: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        data = asdict(self)
        data['start_time_iso'] = datetime.fromtimestamp(self.start_time).isoformat()
        if self.end_time:
            data['end_time_iso'] = datetime.fromtimestamp(self.end_time).isoformat()
            data['duration_seconds'] = round(self.end_time - self.start_time, 3)
        return data

class RequestTracker:
    """请求追踪器"""
    
    def __init__(self):
        self.active_requests: Dict[str, RequestInfo] = {}
        self.completed_requests: Dict[str, RequestInfo] = {}
        self.max_history = 1000  # 最多保留1000条历史记录
    
    def start_request(self, question: str, user_id: Optional[str] = None, 
                     thread_id: Optional[str] = None) -> str:
        """开始新请求追踪"""
        request_id = str(uuid.uuid4())
        request_info = RequestInfo(
            request_id=request_id,
            user_id=user_id,
            thread_id=thread_id,
            question=question,
            start_time=time.time()
        )
        
        self.active_requests[request_id] = request_info
        logger.info(f"开始追踪请求: {request_id}")
        return request_id
    
    def update_model_selection(self, request_id: str, model_name: str):
        """更新选择的模型"""
        if request_id in self.active_requests:
            self.active_requests[request_id].selected_model = model_name
            logger.info(f"请求 {request_id} 选择模型: {model_name}")
    
    def update_token_usage(self, request_id: str, input_tokens: int, 
                          output_tokens: int, cached_tokens: int = 0):
        """更新token使用量"""
        if request_id in self.active_requests:
            req = self.active_requests[request_id]
            req.input_tokens = input_tokens
            req.output_tokens = output_tokens
            req.cached_tokens = cached_tokens
            logger.debug(f"请求 {request_id} Token使用: 输入={input_tokens}, 输出={output_tokens}")
    
    def update_cost(self, request_id: str, total_cost: float):
        """更新总成本"""
        if request_id in self.active_requests:
            self.active_requests[request_id].total_cost = total_cost
            logger.debug(f"请求 {request_id} 总成本: ${total_cost}")
    
    def increment_retry(self, request_id: str):
        """增加重试次数"""
        if request_id in self.active_requests:
            self.active_requests[request_id].retry_count += 1
            logger.info(f"请求 {request_id} 重试次数: {self.active_requests[request_id].retry_count}")
    
    def complete_request(self, request_id: str, success: bool = True, 
                        error_message: Optional[str] = None):
        """完成请求追踪"""
        if request_id not in self.active_requests:
            logger.warning(f"未找到活跃请求: {request_id}")
            return
        
        request_info = self.active_requests[request_id]
        request_info.end_time = time.time()
        request_info.status = "success" if success else "failed"
        if error_message:
            request_info.error_message = error_message
        
        # 移动到完成列表
        self.completed_requests[request_id] = request_info
        del self.active_requests[request_id]
        
        # 保持历史记录大小
        if len(self.completed_requests) > self.max_history:
            oldest_key = min(self.completed_requests.keys(), 
                           key=lambda k: self.completed_requests[k].start_time)
            del self.completed_requests[oldest_key]
        
        # 记录详细日志
        duration = request_info.end_time - request_info.start_time
        if success:
            logger.info(f"请求完成: {request_id}, 模型: {request_info.selected_model}, "
                       f"耗时: {duration:.2f}s, 成本: ${request_info.total_cost}")
        else:
            logger.error(f"请求失败: {request_id}, 模型: {request_info.selected_model}, "
                        f"错误: {error_message}, 重试: {request_info.retry_count}次")
    
    def get_request_info(self, request_id: str) -> Optional[RequestInfo]:
        """获取请求信息"""
        if request_id in self.active_requests:
            return self.active_requests[request_id]
        return self.completed_requests.get(request_id)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total_requests = len(self.active_requests) + len(self.completed_requests)
        success_count = sum(1 for req in self.completed_requests.values() 
                          if req.status == "success")
        failed_count = sum(1 for req in self.completed_requests.values() 
                         if req.status == "failed")
        
        total_cost = sum(req.total_cost for req in self.completed_requests.values())
        
        model_usage = {}
        for req in self.completed_requests.values():
            if req.selected_model:
                model_usage[req.selected_model] = model_usage.get(req.selected_model, 0) + 1
        
        return {
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.completed_requests),
            "total_requests": total_requests,
            "success_rate": round(success_count / len(self.completed_requests) * 100, 2) 
                          if self.completed_requests else 0,
            "total_cost": round(total_cost, 4),
            "model_usage": model_usage
        }
    
    def get_failed_requests(self, limit: int = 50) -> list:
        """获取失败请求列表"""
        failed_requests = [
            req.to_dict() for req in self.completed_requests.values() 
            if req.status == "failed"
        ]
        # 按时间倒序排列
        failed_requests.sort(key=lambda x: x['start_time'], reverse=True)
        return failed_requests[:limit]

# 全局请求追踪器实例
request_tracker = RequestTracker() 