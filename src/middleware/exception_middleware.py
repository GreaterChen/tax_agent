"""
全局异常处理中间件
提供统一的异常捕获、处理和监控
"""
import traceback
import uuid
import time
import logging
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import ValidationError

from ..utils.exceptions import (
    BaseBusinessException, 
    ExceptionFactory,
    ErrorContext,
    RateLimitExceededException  # 向后兼容
)
from ..utils.error_codes import ErrorCode, ErrorCategory
from utils.response_util import ResponseUtil
from ..utils.error_monitor import error_monitor

logger = logging.getLogger(__name__)

class ExceptionHandlingMiddleware(BaseHTTPMiddleware):
    """全局异常处理中间件"""
    
    def __init__(self, app: FastAPI, enable_debug: bool = False):
        super().__init__(app)
        self.enable_debug = enable_debug
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求并捕获异常"""
        # 生成请求追踪ID
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        try:
            # 执行请求
            response = await call_next(request)
            
            # 记录成功请求
            process_time = time.time() - start_time
            logger.info(
                f"请求成功: {request.method} {request.url.path}",
                extra={
                    'trace_id': trace_id,
                    'method': request.method,
                    'path': request.url.path,
                    'status_code': response.status_code,
                    'process_time': process_time,
                    'client_ip': self._get_client_ip(request),
                }
            )
            
            # 添加追踪头
            response.headers["X-Trace-ID"] = trace_id
            return response
            
        except BaseBusinessException as e:
            # 处理业务异常
            return await self._handle_business_exception(e, request, trace_id, start_time)
            
        except RateLimitExceededException as e:
            # 向后兼容的限流异常处理
            return await self._handle_rate_limit_exception(e, request, trace_id, start_time)
            
        except ValidationError as e:
            # 处理Pydantic验证异常
            return await self._handle_validation_exception(e, request, trace_id, start_time)
            
        except Exception as e:
            # 处理未知异常
            return await self._handle_unknown_exception(e, request, trace_id, start_time)
    
    async def _handle_business_exception(
        self, 
        exception: BaseBusinessException, 
        request: Request, 
        trace_id: str,
        start_time: float
    ) -> Response:
        """处理业务异常"""
        # 补充错误上下文
        if not exception.context.request_id:
            exception.context.request_id = trace_id
        
        process_time = time.time() - start_time
        
        # 记录错误监控
        error_monitor.record_error(exception, extra_context={
            'method': request.method,
            'path': request.url.path,
            'process_time': process_time,
            'client_ip': self._get_client_ip(request),
        })
        
        # 记录异常日志
        log_level = getattr(logging, exception.error_code.log_level, logging.ERROR)
        logger.log(
            log_level,
            f"业务异常: {exception.error_code.code} - {exception.message}",
            extra={
                'trace_id': trace_id,
                'error_code': exception.error_code.code,
                'error_category': exception.error_code.category.value,
                'method': request.method,
                'path': request.url.path,
                'process_time': process_time,
                'client_ip': self._get_client_ip(request),
                'exception_dict': exception.to_dict(),
            }
        )
        
        # 创建响应
        response = ResponseUtil.error_from_exception(exception)
        response.headers["X-Trace-ID"] = trace_id
        return response
    
    async def _handle_rate_limit_exception(
        self, 
        exception: RateLimitExceededException, 
        request: Request, 
        trace_id: str,
        start_time: float
    ) -> Response:
        """处理向后兼容的限流异常"""
        # 转换为新的异常类型
        context = ErrorContext(
            request_id=trace_id,
            operation=f"{request.method} {request.url.path}",
            component="rate_limiter"
        )
        
        business_exception = ExceptionFactory.create_rate_limit_exception(
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            retry_after=getattr(exception, 'retry_after', 60),
            available_models=getattr(exception, 'available_models', []),
            context=context,
            message=str(exception)
        )
        
        return await self._handle_business_exception(business_exception, request, trace_id, start_time)
    
    async def _handle_validation_exception(
        self, 
        exception: ValidationError, 
        request: Request, 
        trace_id: str,
        start_time: float
    ) -> Response:
        """处理参数验证异常"""
        # 提取验证错误详情
        errors = []
        for error in exception.errors():
            field_name = '.'.join(str(x) for x in error.get('loc', []))
            error_msg = error.get('msg', '未知验证错误')
            errors.append(f"{field_name}: {error_msg}")
        
        error_message = "请求参数验证失败: " + "; ".join(errors)
        
        context = ErrorContext(
            request_id=trace_id,
            operation=f"{request.method} {request.url.path}",
            component="validation",
            extra_data={'validation_errors': exception.errors()}
        )
        
        business_exception = ExceptionFactory.create_validation_exception(
            error_code=ErrorCode.INVALID_REQUEST,
            message=error_message,
            context=context,
            cause=exception
        )
        
        return await self._handle_business_exception(business_exception, request, trace_id, start_time)
    
    async def _handle_unknown_exception(
        self, 
        exception: Exception, 
        request: Request, 
        trace_id: str,
        start_time: float
    ) -> Response:
        """处理未知异常"""
        process_time = time.time() - start_time
        
        # 记录详细的异常信息
        logger.error(
            f"未处理异常: {type(exception).__name__} - {str(exception)}",
            extra={
                'trace_id': trace_id,
                'method': request.method,
                'path': request.url.path,
                'process_time': process_time,
                'client_ip': self._get_client_ip(request),
                'exception_type': type(exception).__name__,
                'stack_trace': traceback.format_exc(),
            }
        )
        
        # 创建系统异常
        context = ErrorContext(
            request_id=trace_id,
            operation=f"{request.method} {request.url.path}",
            component="unknown"
        )
        
        # 判断是否为常见的系统异常
        error_code = self._classify_system_exception(exception)
        
        business_exception = ExceptionFactory.create_system_exception(
            error_code=error_code,
            cause=exception,
            context=context
        )
        
        return await self._handle_business_exception(business_exception, request, trace_id, start_time)
    
    def _classify_system_exception(self, exception: Exception) -> ErrorCode:
        """分类系统异常"""
        exception_name = type(exception).__name__.lower()
        exception_msg = str(exception).lower()
        
        # 数据库相关异常
        if any(keyword in exception_name for keyword in ['database', 'connection', 'mysql', 'postgres']):
            return ErrorCode.DATABASE_ERROR
            
        # Redis相关异常
        if any(keyword in exception_name for keyword in ['redis', 'cache']):
            return ErrorCode.REDIS_ERROR
            
        # 配置相关异常
        if any(keyword in exception_name for keyword in ['config', 'environment', 'setting']):
            return ErrorCode.CONFIG_ERROR
            
        # 默认系统异常
        return ErrorCode.SYSTEM_ERROR
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP地址"""
        # 尝试从各种头部获取真实IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
            
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
            
        return request.client.host if request.client else "unknown"

def setup_exception_handling(app: FastAPI, enable_debug: bool = False):
    """设置全局异常处理"""
    
    # 添加异常处理中间件
    app.add_middleware(ExceptionHandlingMiddleware, enable_debug=enable_debug)
    
    # 添加特定异常处理器（作为补充）
    @app.exception_handler(BaseBusinessException)
    async def business_exception_handler(request: Request, exc: BaseBusinessException):
        """业务异常处理器"""
        trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4()))
        
        if not exc.context.request_id:
            exc.context.request_id = trace_id
            
        response = ResponseUtil.error_from_exception(exc)
        response.headers["X-Trace-ID"] = trace_id
        return response
    
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        """404错误处理器"""
        trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4()))
        
        context = ErrorContext(
            request_id=trace_id,
            operation=f"{request.method} {request.url.path}",
            component="routing"
        )
        
        business_exception = ExceptionFactory.create_validation_exception(
            error_code=ErrorCode.INVALID_REQUEST,
            message=f"请求路径不存在: {request.url.path}",
            context=context
        )
        
        response = ResponseUtil.error_from_exception(business_exception)
        response.headers["X-Trace-ID"] = trace_id
        return response
    
    logger.info("全局异常处理中间件已启用") 