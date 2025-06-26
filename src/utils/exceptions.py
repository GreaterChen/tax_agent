"""
标准化异常类体系
提供结构化异常信息、错误追踪和上下文管理
"""
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

from .error_codes import ErrorCode, ErrorCategory

@dataclass
class ErrorContext:
    """错误上下文信息"""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class BaseBusinessException(Exception):
    """业务异常基类"""
    
    def __init__(
        self, 
        error_code: ErrorCode,
        message: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional[ErrorContext] = None,
        **kwargs
    ):
        self.error_code = error_code
        self.message = message or error_code.message
        self.user_message = error_code.user_message
        self.cause = cause
        self.context = context or ErrorContext()
        self.trace_id = str(uuid.uuid4())
        self.extra_data = kwargs
        
        # 构造异常消息
        exception_message = f"[{error_code.code}] {self.message}"
        if cause:
            exception_message += f" (caused by: {str(cause)})"
            
        super().__init__(exception_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code.code,
            "error_message": self.message,
            "user_message": self.user_message,
            "category": self.error_code.category.value,
            "retryable": self.error_code.retryable,
            "trace_id": self.trace_id,
            "timestamp": self.context.timestamp.isoformat(),
            "context": {
                "request_id": self.context.request_id,
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "operation": self.context.operation,
                "component": self.context.component,
                "extra_data": self.context.extra_data,
            },
            "extra_data": self.extra_data,
            "cause": str(self.cause) if self.cause else None,
            "stack_trace": traceback.format_exc() if self.cause else None,
        }

class SystemException(BaseBusinessException):
    """系统异常"""
    
    def __init__(self, error_code: ErrorCode = ErrorCode.SYSTEM_ERROR, **kwargs):
        if error_code.category != ErrorCategory.SYSTEM:
            raise ValueError("SystemException must use SYSTEM category error code")
        super().__init__(error_code, **kwargs)

class BusinessException(BaseBusinessException):
    """业务异常"""
    
    def __init__(self, error_code: ErrorCode = ErrorCode.AGENT_QUERY_FAILED, **kwargs):
        if error_code.category != ErrorCategory.BUSINESS:
            raise ValueError("BusinessException must use BUSINESS category error code")
        super().__init__(error_code, **kwargs)

class ValidationException(BaseBusinessException):
    """参数验证异常"""
    
    def __init__(self, error_code: ErrorCode = ErrorCode.INVALID_REQUEST, **kwargs):
        if error_code.category != ErrorCategory.VALIDATION:
            raise ValueError("ValidationException must use VALIDATION category error code")
        super().__init__(error_code, **kwargs)

class AuthException(BaseBusinessException):
    """认证授权异常"""
    
    def __init__(self, error_code: ErrorCode = ErrorCode.UNAUTHORIZED, **kwargs):
        if error_code.category != ErrorCategory.AUTH:
            raise ValueError("AuthException must use AUTH category error code")
        super().__init__(error_code, **kwargs)

class RateLimitException(BaseBusinessException):
    """限流异常"""
    
    def __init__(
        self, 
        error_code: ErrorCode = ErrorCode.RATE_LIMIT_EXCEEDED,
        retry_after: int = 60,
        available_models: Optional[List[str]] = None,
        **kwargs
    ):
        if error_code.category != ErrorCategory.RATE_LIMIT:
            raise ValueError("RateLimitException must use RATE_LIMIT category error code")
        
        self.retry_after = retry_after
        self.available_models = available_models or []
        
        # 添加到额外数据中
        kwargs['retry_after'] = retry_after
        kwargs['available_models'] = available_models
        
        super().__init__(error_code, **kwargs)

class ExternalServiceException(BaseBusinessException):
    """外部服务异常"""
    
    def __init__(
        self, 
        error_code: ErrorCode = ErrorCode.EXTERNAL_API_ERROR,
        service_name: Optional[str] = None,
        **kwargs
    ):
        if error_code.category != ErrorCategory.EXTERNAL:
            raise ValueError("ExternalServiceException must use EXTERNAL category error code")
        
        self.service_name = service_name
        kwargs['service_name'] = service_name
        
        super().__init__(error_code, **kwargs)

# 向后兼容的异常类
class RateLimitExceededException(RateLimitException):
    """限流超限异常 (向后兼容)"""
    
    def __init__(self, message: str, available_models: list = None, retry_after: int = 60):
        super().__init__(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message=message,
            retry_after=retry_after,
            available_models=available_models
        )

class ExceptionFactory:
    """异常工厂类"""
    
    @staticmethod
    def create_system_exception(
        error_code: ErrorCode,
        cause: Optional[Exception] = None,
        context: Optional[ErrorContext] = None,
        **kwargs
    ) -> SystemException:
        """创建系统异常"""
        return SystemException(error_code, cause=cause, context=context, **kwargs)
    
    @staticmethod
    def create_business_exception(
        error_code: ErrorCode,
        cause: Optional[Exception] = None,
        context: Optional[ErrorContext] = None,
        **kwargs
    ) -> BusinessException:
        """创建业务异常"""
        return BusinessException(error_code, cause=cause, context=context, **kwargs)
    
    @staticmethod
    def create_rate_limit_exception(
        error_code: ErrorCode = ErrorCode.RATE_LIMIT_EXCEEDED,
        retry_after: int = 60,
        available_models: Optional[List[str]] = None,
        context: Optional[ErrorContext] = None,
        **kwargs
    ) -> RateLimitException:
        """创建限流异常"""
        return RateLimitException(
            error_code=error_code,
            retry_after=retry_after,
            available_models=available_models,
            context=context,
            **kwargs
        )
    
    @staticmethod
    def create_validation_exception(
        error_code: ErrorCode = ErrorCode.INVALID_REQUEST,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        context: Optional[ErrorContext] = None,
        **kwargs
    ) -> ValidationException:
        """创建参数验证异常"""
        if field_name:
            kwargs['field_name'] = field_name
        if field_value is not None:
            kwargs['field_value'] = field_value
        
        return ValidationException(error_code, context=context, **kwargs)
    
    @staticmethod
    def wrap_exception(
        original_exception: Exception,
        error_code: ErrorCode,
        context: Optional[ErrorContext] = None,
        **kwargs
    ) -> BaseBusinessException:
        """包装原始异常为业务异常"""
        exception_class_map = {
            ErrorCategory.SYSTEM: SystemException,
            ErrorCategory.BUSINESS: BusinessException,
            ErrorCategory.VALIDATION: ValidationException,
            ErrorCategory.AUTH: AuthException,
            ErrorCategory.RATE_LIMIT: RateLimitException,
            ErrorCategory.EXTERNAL: ExternalServiceException,
        }
        
        exception_class = exception_class_map.get(error_code.category, BaseBusinessException)
        return exception_class(error_code, cause=original_exception, context=context, **kwargs) 