"""
标准化错误码体系
提供标准化的错误码定义和多语言错误信息
"""
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass

class ErrorCategory(Enum):
    """错误类别"""
    SYSTEM = "SYSTEM"           # 系统错误 1000-1999
    BUSINESS = "BUSINESS"       # 业务错误 2000-2999  
    VALIDATION = "VALIDATION"   # 参数验证错误 3000-3999
    AUTH = "AUTH"              # 认证授权错误 4000-4999
    RATE_LIMIT = "RATE_LIMIT"  # 限流错误 5000-5999
    EXTERNAL = "EXTERNAL"      # 外部服务错误 6000-6999

@dataclass
class ErrorInfo:
    """错误信息"""
    code: int
    message: str
    user_message: str  # 用户友好的错误信息
    category: ErrorCategory
    retryable: bool = False
    log_level: str = "ERROR"

class ErrorCode(Enum):
    """标准化错误码"""
    
    # 系统错误 1000-1999
    SYSTEM_ERROR = ErrorInfo(1000, "System internal error", "系统内部错误，请稍后重试", ErrorCategory.SYSTEM)
    CONFIG_ERROR = ErrorInfo(1001, "Configuration error", "系统配置错误", ErrorCategory.SYSTEM)
    DATABASE_ERROR = ErrorInfo(1002, "Database connection error", "数据库连接异常", ErrorCategory.SYSTEM, retryable=True)
    REDIS_ERROR = ErrorInfo(1003, "Redis connection error", "缓存服务异常", ErrorCategory.SYSTEM, retryable=True)
    
    # 业务错误 2000-2999
    AGENT_QUERY_FAILED = ErrorInfo(2000, "Agent query execution failed", "查询执行失败", ErrorCategory.BUSINESS)
    WORKFLOW_ERROR = ErrorInfo(2001, "Workflow execution error", "工作流执行异常", ErrorCategory.BUSINESS)
    LLM_ERROR = ErrorInfo(2002, "LLM service error", "AI模型服务异常", ErrorCategory.BUSINESS, retryable=True)
    TOOL_EXECUTION_ERROR = ErrorInfo(2003, "Tool execution failed", "工具执行失败", ErrorCategory.BUSINESS)
    
    # 参数验证错误 3000-3999
    INVALID_REQUEST = ErrorInfo(3000, "Invalid request parameters", "请求参数无效", ErrorCategory.VALIDATION)
    MISSING_REQUIRED_FIELD = ErrorInfo(3001, "Missing required field", "缺少必填字段", ErrorCategory.VALIDATION)
    INVALID_FORMAT = ErrorInfo(3002, "Invalid data format", "数据格式不正确", ErrorCategory.VALIDATION)
    REQUEST_TOO_LARGE = ErrorInfo(3003, "Request payload too large", "请求内容过大", ErrorCategory.VALIDATION)
    
    # 认证授权错误 4000-4999
    UNAUTHORIZED = ErrorInfo(4001, "Authentication required", "需要身份认证", ErrorCategory.AUTH)
    FORBIDDEN = ErrorInfo(4003, "Access forbidden", "访问被拒绝", ErrorCategory.AUTH)
    TOKEN_EXPIRED = ErrorInfo(4004, "Token expired", "访问令牌已过期", ErrorCategory.AUTH)
    INSUFFICIENT_PRIVILEGES = ErrorInfo(4005, "Insufficient privileges", "权限不足", ErrorCategory.AUTH)
    
    # 限流错误 5000-5999
    RATE_LIMIT_EXCEEDED = ErrorInfo(5001, "Rate limit exceeded", "请求频率超限，请稍后重试", ErrorCategory.RATE_LIMIT, retryable=True, log_level="WARN")
    QPM_LIMIT_EXCEEDED = ErrorInfo(5002, "QPM limit exceeded", "每分钟请求数超限", ErrorCategory.RATE_LIMIT, retryable=True, log_level="WARN")
    TPM_LIMIT_EXCEEDED = ErrorInfo(5003, "TPM limit exceeded", "每分钟Token数超限", ErrorCategory.RATE_LIMIT, retryable=True, log_level="WARN")
    CONCURRENT_LIMIT_EXCEEDED = ErrorInfo(5004, "Concurrent request limit exceeded", "并发请求数超限", ErrorCategory.RATE_LIMIT, retryable=True, log_level="WARN")
    
    # 外部服务错误 6000-6999
    EXTERNAL_API_ERROR = ErrorInfo(6000, "External API error", "外部API调用失败", ErrorCategory.EXTERNAL, retryable=True)
    NETWORK_TIMEOUT = ErrorInfo(6001, "Network timeout", "网络超时", ErrorCategory.EXTERNAL, retryable=True)
    SERVICE_UNAVAILABLE = ErrorInfo(6002, "Service unavailable", "服务暂时不可用", ErrorCategory.EXTERNAL, retryable=True)
    
    @property
    def code(self) -> int:
        return self.value.code
    
    @property
    def message(self) -> str:
        return self.value.message
    
    @property
    def user_message(self) -> str:
        return self.value.user_message
    
    @property
    def category(self) -> ErrorCategory:
        return self.value.category
    
    @property
    def retryable(self) -> bool:
        return self.value.retryable
    
    @property
    def log_level(self) -> str:
        return self.value.log_level

class ErrorCodeManager:
    """错误码管理器"""
    
    @staticmethod
    def get_error_by_code(code: int) -> Optional[ErrorCode]:
        """根据错误码获取错误信息"""
        for error_code in ErrorCode:
            if error_code.code == code:
                return error_code
        return None
    
    @staticmethod
    def get_errors_by_category(category: ErrorCategory) -> list[ErrorCode]:
        """根据类别获取错误码列表"""
        return [error_code for error_code in ErrorCode if error_code.category == category]
    
    @staticmethod
    def is_retryable_error(code: int) -> bool:
        """判断错误是否可重试"""
        error_code = ErrorCodeManager.get_error_by_code(code)
        return error_code.retryable if error_code else False

# 错误码到HTTP状态码的映射
ERROR_CODE_TO_HTTP_STATUS = {
    ErrorCategory.SYSTEM: 500,
    ErrorCategory.BUSINESS: 500,
    ErrorCategory.VALIDATION: 400,
    ErrorCategory.AUTH: 401,
    ErrorCategory.RATE_LIMIT: 429,
    ErrorCategory.EXTERNAL: 502,
} 