"""
重试管理器
实现智能重试机制和退避算法
"""
import asyncio
import random
import time
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    """重试策略"""
    FIXED = "fixed"           # 固定间隔
    LINEAR = "linear"         # 线性增长
    EXPONENTIAL = "exponential"  # 指数退避
    JITTERED_EXPONENTIAL = "jittered_exponential"  # 带抖动的指数退避

@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    base_delay: float = 1.0  # 基础延迟（秒）
    max_delay: float = 60.0  # 最大延迟（秒）
    strategy: RetryStrategy = RetryStrategy.JITTERED_EXPONENTIAL
    backoff_multiplier: float = 2.0  # 退避倍数
    jitter_range: float = 0.1  # 抖动范围（0-1）

class RetryManager:
    """重试管理器"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.retry_stats = {}  # 重试统计
    
    def calculate_delay(self, attempt: int) -> float:
        """计算重试延迟时间"""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
            
        elif self.config.strategy == RetryStrategy.JITTERED_EXPONENTIAL:
            base_delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
            jitter = base_delay * self.config.jitter_range * (random.random() * 2 - 1)
            delay = base_delay + jitter
            
        else:
            delay = self.config.base_delay
        
        # 确保延迟在合理范围内
        delay = max(0.1, min(delay, self.config.max_delay))
        
        logger.debug(f"计算重试延迟: 第{attempt}次重试, 延迟{delay:.2f}秒")
        return delay
    
    async def retry_with_backoff(self, 
                               operation: Callable[[], Awaitable[Any]], 
                               operation_name: str = "operation",
                               should_retry: Callable[[Exception], bool] = None) -> Any:
        """
        带退避的重试操作
        
        Args:
            operation: 要重试的异步操作
            operation_name: 操作名称（用于日志）
            should_retry: 判断是否应该重试的函数
            
        Returns:
            操作结果
            
        Raises:
            最后一次尝试的异常
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.info(f"执行{operation_name}: 第{attempt}次尝试")
                result = await operation()
                
                # 成功时记录统计
                self._record_success(operation_name, attempt)
                
                if attempt > 1:
                    logger.info(f"{operation_name}重试成功: 第{attempt}次尝试")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # 判断是否应该重试
                if should_retry and not should_retry(e):
                    logger.info(f"{operation_name}不应重试的错误: {e}")
                    break
                
                if attempt == self.config.max_retries:
                    logger.error(f"{operation_name}重试失败: 已达最大重试次数{self.config.max_retries}")
                    break
                
                # 计算并等待重试延迟
                delay = self.calculate_delay(attempt)
                logger.warning(f"{operation_name}第{attempt}次尝试失败: {e}, "
                             f"{delay:.2f}秒后进行第{attempt + 1}次重试")
                
                await asyncio.sleep(delay)
        
        # 记录失败统计
        self._record_failure(operation_name, self.config.max_retries)
        
        # 抛出最后一次异常
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"{operation_name}重试失败，原因未知")
    
    def _record_success(self, operation_name: str, attempts: int):
        """记录成功统计"""
        if operation_name not in self.retry_stats:
            self.retry_stats[operation_name] = {
                "total_attempts": 0,
                "total_successes": 0,
                "total_failures": 0,
                "attempts_distribution": {}
            }
        
        stats = self.retry_stats[operation_name]
        stats["total_attempts"] += attempts
        stats["total_successes"] += 1
        stats["attempts_distribution"][attempts] = stats["attempts_distribution"].get(attempts, 0) + 1
    
    def _record_failure(self, operation_name: str, attempts: int):
        """记录失败统计"""
        if operation_name not in self.retry_stats:
            self.retry_stats[operation_name] = {
                "total_attempts": 0,
                "total_successes": 0,
                "total_failures": 0,
                "attempts_distribution": {}
            }
        
        stats = self.retry_stats[operation_name]
        stats["total_attempts"] += attempts
        stats["total_failures"] += 1
        stats["attempts_distribution"][attempts] = stats["attempts_distribution"].get(attempts, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取重试统计信息"""
        summary = {}
        
        for operation_name, stats in self.retry_stats.items():
            total_operations = stats["total_successes"] + stats["total_failures"]
            success_rate = stats["total_successes"] / total_operations * 100 if total_operations > 0 else 0
            avg_attempts = stats["total_attempts"] / total_operations if total_operations > 0 else 0
            
            summary[operation_name] = {
                "total_operations": total_operations,
                "success_rate": round(success_rate, 2),
                "average_attempts": round(avg_attempts, 2),
                "total_successes": stats["total_successes"],
                "total_failures": stats["total_failures"],
                "attempts_distribution": stats["attempts_distribution"]
            }
        
        return summary
    
    def reset_stats(self):
        """重置统计信息"""
        self.retry_stats.clear()
        logger.info("重试统计信息已重置")

class RateLimitRetryManager(RetryManager):
    """专门针对限流的重试管理器"""
    
    def __init__(self):
        # 针对限流的特殊配置
        config = RetryConfig(
            max_retries=5,  # 限流重试次数可以多一些
            base_delay=5.0,  # 限流基础延迟较长
            max_delay=300.0,  # 最大延迟5分钟
            strategy=RetryStrategy.JITTERED_EXPONENTIAL,
            backoff_multiplier=1.5,  # 较温和的退避倍数
            jitter_range=0.2  # 更大的抖动范围避免雷群效应
        )
        super().__init__(config)
    
    def should_retry_rate_limit_error(self, exception: Exception) -> bool:
        """判断限流错误是否应该重试"""
        error_msg = str(exception).lower()
        
        # 可重试的限流错误类型
        retryable_errors = [
            "ratelimitexceededexception",
            "rate limit",
            "quota exceeded",
            "请求高峰",
            "达到限制",
            "too many requests"
        ]
        
        # 不可重试的错误类型
        non_retryable_errors = [
            "authentication",
            "authorization", 
            "invalid api key",
            "permission denied",
            "模型不存在",
            "invalid model"
        ]
        
        # 检查是否为不可重试的错误
        for error_pattern in non_retryable_errors:
            if error_pattern in error_msg:
                return False
        
        # 检查是否为可重试的限流错误
        for error_pattern in retryable_errors:
            if error_pattern in error_msg:
                return True
        
        # 默认不重试未知错误
        return False
    
    async def retry_llm_selection(self, select_llm_func: Callable[[], Awaitable[Any]]) -> Any:
        """重试LLM选择操作"""
        return await self.retry_with_backoff(
            operation=select_llm_func,
            operation_name="LLM选择",
            should_retry=self.should_retry_rate_limit_error
        )

# 全局重试管理器实例
retry_manager = RetryManager()
rate_limit_retry_manager = RateLimitRetryManager() 