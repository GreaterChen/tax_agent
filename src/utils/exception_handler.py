"""
异常处理器
负责处理各种异常类型和生成用户友好的错误信息
"""
import logging
from typing import List

logger = logging.getLogger(__name__)

class ExceptionHandler:
    """异常处理器"""
    
    @staticmethod
    def is_rate_limit_exception(exception: Exception) -> bool:
        """判断是否为限流相关异常"""
        error_msg = str(exception).lower()
        exception_name = exception.__class__.__name__
        
        # 检查异常类型和错误信息
        rate_limit_indicators = [
            "ratelimitexceededexception",
            "rate limit",
            "quota",
            "限流",
            "请求高峰",
            "达到限制"
        ]
        
        return (exception_name.lower() == "ratelimitexceededexception" or 
                any(indicator in error_msg for indicator in rate_limit_indicators))
    
    @staticmethod
    def handle_rate_limit_exception(exception: Exception) -> List[str]:
        """处理限流异常，返回友好的用户提示"""
        try:
            # 尝试获取异常的详细信息
            if hasattr(exception, 'retry_after'):
                retry_after = exception.retry_after
            else:
                retry_after = 60  # 默认重试时间
                
            if hasattr(exception, 'available_models'):
                models_info = f"涉及模型: {', '.join(exception.available_models)}"
            else:
                models_info = ""
            
            # 生成友好的错误消息
            base_message = "🚫 系统目前处于请求高峰期，所有AI模型都已达到使用限制。"
            
            retry_message = f"⏰ 建议 {retry_after} 秒后重试，或选择非高峰时段使用。"
            
            tips = [
                "💡 使用建议:",
                "• 尝试简化您的问题以减少处理时间",
                "• 避免在高峰时段(工作日9-18点)发起复杂查询", 
                "• 如有紧急需求，请联系系统管理员"
            ]
            
            result = [base_message, retry_message]
            result.extend(tips)
            
            if models_info:
                result.append(f"📊 {models_info}")
                
            # 记录限流日志用于监控
            logger.warning(f"用户请求被限流拒绝: {exception}, 建议重试时间: {retry_after}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"处理限流异常时出错: {e}")
            return [
                "🚫 系统目前处于请求高峰期，请稍后再试。",
                "⏰ 建议1分钟后重试，感谢您的理解。"
            ]
    
    @staticmethod
    def handle_general_exception(exception: Exception, context: str = "") -> List[str]:
        """处理一般异常"""
        error_msg = str(exception)
        
        # 记录详细错误日志
        logger.error(f"处理请求时发生错误 {context}: {error_msg}")
        
        # 返回用户友好的错误信息
        return [f"抱歉，处理您的请求时发生错误: {error_msg}"]

# 全局实例
exception_handler = ExceptionHandler() 