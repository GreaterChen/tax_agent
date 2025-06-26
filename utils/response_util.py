# -*- coding: utf-8 -*-
"""
统一响应工具类
用于构建API响应的标准格式，支持结构化错误信息和链路跟踪
"""
from datetime import datetime
from fastapi import status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from typing import Any, Dict, Mapping, Optional, Union
import logging

# 导入标准异常类
from src.utils.exceptions import BaseBusinessException
from src.utils.error_codes import ErrorCode, ErrorCategory, ERROR_CODE_TO_HTTP_STATUS


class HttpStatusConstant:
    """
    返回状态码常量
    """
    SUCCESS = 200           # 操作成功
    CREATED = 201          # 对象创建成功
    ACCEPTED = 202         # 请求已经被接受
    NO_CONTENT = 204       # 操作已经执行成功，但是没有返回数据
    MOVED_PERM = 301       # 资源已被移除
    SEE_OTHER = 303        # 重定向
    NOT_MODIFIED = 304     # 资源没有被修改
    BAD_REQUEST = 400      # 参数列表错误（缺少，格式不匹配）
    UNAUTHORIZED = 401     # 未授权
    FORBIDDEN = 403        # 访问受限，授权过期
    NOT_FOUND = 404        # 资源，服务未找到
    BAD_METHOD = 405       # 不允许的http方法
    CONFLICT = 409         # 资源冲突，或者资源被锁
    UNSUPPORTED_TYPE = 415 # 不支持的数据，媒体类型
    ERROR = 500            # 系统内部错误
    NOT_IMPLEMENTED = 501  # 接口未实现
    WARN = 300             # 系统警告消息


logger = logging.getLogger(__name__)

class ResponseUtil:
    """
    统一响应工具类
    用于构建API响应的标准格式，支持结构化错误信息和链路跟踪
    """

    @classmethod
    def success(
        cls,
        msg: str = '操作成功',
        data: Optional[Any] = None,
        rows: Optional[Any] = None,
        dict_content: Optional[Dict] = None,
        model_content: Optional[BaseModel] = None,
        headers: Optional[Mapping[str, str]] = None,
        media_type: Optional[str] = None,
        background: Optional[BackgroundTask] = None,
    ) -> Response:
        """
        成功响应方法

        :param msg: 可选，自定义成功响应信息
        :param data: 可选，成功响应结果中属性为data的值
        :param rows: 可选，成功响应结果中属性为rows的值
        :param dict_content: 可选，dict类型，成功响应结果中自定义属性的值
        :param model_content: 可选，BaseModel类型，成功响应结果中自定义属性的值
        :param headers: 可选，响应头信息
        :param media_type: 可选，响应结果媒体类型
        :param background: 可选，响应返回后执行的后台任务
        :return: 成功响应结果
        """
        result = {'code': HttpStatusConstant.SUCCESS, 'msg': msg}

        if data is not None:
            result['data'] = data
        if rows is not None:
            result['rows'] = rows
        if dict_content is not None:
            result.update(dict_content)
        if model_content is not None:
            result.update(model_content.model_dump(by_alias=True))

        result.update({'success': True, 'time': datetime.now()})

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder(result),
            headers=headers,
            media_type=media_type,
            background=background,
        )

    @classmethod
    def failure(
        cls,
        msg: str = '操作失败',
        data: Optional[Any] = None,
        rows: Optional[Any] = None,
        dict_content: Optional[Dict] = None,
        model_content: Optional[BaseModel] = None,
        headers: Optional[Mapping[str, str]] = None,
        media_type: Optional[str] = None,
        background: Optional[BackgroundTask] = None,
    ) -> Response:
        """
        失败响应方法

        :param msg: 可选，自定义失败响应信息
        :param data: 可选，失败响应结果中属性为data的值
        :param rows: 可选，失败响应结果中属性为rows的值
        :param dict_content: 可选，dict类型，失败响应结果中自定义属性的值
        :param model_content: 可选，BaseModel类型，失败响应结果中自定义属性的值
        :param headers: 可选，响应头信息
        :param media_type: 可选，响应结果媒体类型
        :param background: 可选，响应返回后执行的后台任务
        :return: 失败响应结果
        """
        result = {'code': HttpStatusConstant.WARN, 'msg': msg}

        # 始终包含data字段，如果没有提供则为空字典
        result['data'] = data if data is not None else {}
        if rows is not None:
            result['rows'] = rows
        if dict_content is not None:
            result.update(dict_content)
        if model_content is not None:
            result.update(model_content.model_dump(by_alias=True))

        result.update({'success': False, 'time': datetime.now()})

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder(result),
            headers=headers,
            media_type=media_type,
            background=background,
        )

    @classmethod
    def error(
        cls,
        msg: str = '系统内部错误',
        data: Optional[Any] = None,
        rows: Optional[Any] = None,
        dict_content: Optional[Dict] = None,
        model_content: Optional[BaseModel] = None,
        headers: Optional[Mapping[str, str]] = None,
        media_type: Optional[str] = None,
        background: Optional[BackgroundTask] = None,
    ) -> Response:
        """
        错误响应方法

        :param msg: 可选，自定义错误响应信息
        :param data: 可选，错误响应结果中属性为data的值
        :param rows: 可选，错误响应结果中属性为rows的值
        :param dict_content: 可选，dict类型，错误响应结果中自定义属性的值
        :param model_content: 可选，BaseModel类型，错误响应结果中自定义属性的值
        :param headers: 可选，响应头信息
        :param media_type: 可选，响应结果媒体类型
        :param background: 可选，响应返回后执行的后台任务
        :return: 错误响应结果
        """
        result = {'code': HttpStatusConstant.ERROR, 'msg': msg}

        # 始终包含data字段，如果没有提供则为空字典
        result['data'] = data if data is not None else {}
        if rows is not None:
            result['rows'] = rows
        if dict_content is not None:
            result.update(dict_content)
        if model_content is not None:
            result.update(model_content.model_dump(by_alias=True))

        result.update({'success': False, 'time': datetime.now()})

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder(result),
            headers=headers,
            media_type=media_type,
            background=background,
        )

    @classmethod
    def unauthorized(
        cls,
        msg: str = '登录信息已过期，访问系统资源失败',
        data: Optional[Any] = None,
        rows: Optional[Any] = None,
        dict_content: Optional[Dict] = None,
        model_content: Optional[BaseModel] = None,
        headers: Optional[Mapping[str, str]] = None,
        media_type: Optional[str] = None,
        background: Optional[BackgroundTask] = None,
    ) -> Response:
        """
        未认证响应方法

        :param msg: 可选，自定义未认证响应信息
        :param data: 可选，未认证响应结果中属性为data的值
        :param rows: 可选，未认证响应结果中属性为rows的值
        :param dict_content: 可选，dict类型，未认证响应结果中自定义属性的值
        :param model_content: 可选，BaseModel类型，未认证响应结果中自定义属性的值
        :param headers: 可选，响应头信息
        :param media_type: 可选，响应结果媒体类型
        :param background: 可选，响应返回后执行的后台任务
        :return: 未认证响应结果
        """
        result = {'code': HttpStatusConstant.UNAUTHORIZED, 'msg': msg}

        # 始终包含data字段，如果没有提供则为空字典
        result['data'] = data if data is not None else {}
        if rows is not None:
            result['rows'] = rows
        if dict_content is not None:
            result.update(dict_content)
        if model_content is not None:
            result.update(model_content.model_dump(by_alias=True))

        result.update({'success': False, 'time': datetime.now()})

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder(result),
            headers=headers,
            media_type=media_type,
            background=background,
        )

    @classmethod
    def forbidden(
        cls,
        msg: str = '该用户无此接口权限',
        data: Optional[Any] = None,
        rows: Optional[Any] = None,
        dict_content: Optional[Dict] = None,
        model_content: Optional[BaseModel] = None,
        headers: Optional[Mapping[str, str]] = None,
        media_type: Optional[str] = None,
        background: Optional[BackgroundTask] = None,
    ) -> Response:
        """
        未授权响应方法

        :param msg: 可选，自定义未授权响应信息
        :param data: 可选，未授权响应结果中属性为data的值
        :param rows: 可选，未授权响应结果中属性为rows的值
        :param dict_content: 可选，dict类型，未授权响应结果中自定义属性的值
        :param model_content: 可选，BaseModel类型，未授权响应结果中自定义属性的值
        :param headers: 可选，响应头信息
        :param media_type: 可选，响应结果媒体类型
        :param background: 可选，响应返回后执行的后台任务
        :return: 未授权响应结果
        """
        result = {'code': HttpStatusConstant.FORBIDDEN, 'msg': msg}

        # 始终包含data字段，如果没有提供则为空字典
        result['data'] = data if data is not None else {}
        if rows is not None:
            result['rows'] = rows
        if dict_content is not None:
            result.update(dict_content)
        if model_content is not None:
            result.update(model_content.model_dump(by_alias=True))

        result.update({'success': False, 'time': datetime.now()})

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder(result),
            headers=headers,
            media_type=media_type,
            background=background,
        )

    @classmethod
    def streaming(
        cls,
        *,
        data: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        media_type: Optional[str] = None,
        background: Optional[BackgroundTask] = None,
    ) -> Response:
        """
        流式响应方法

        :param data: 流式传输的内容
        :param headers: 可选，响应头信息
        :param media_type: 可选，响应结果媒体类型
        :param background: 可选，响应返回后执行的后台任务
        :return: 流式响应结果
        """
        return StreamingResponse(
            status_code=status.HTTP_200_OK, 
            content=data, 
            headers=headers, 
            media_type=media_type, 
            background=background
        )

    @classmethod
    def error_from_exception(
        cls,
        exception: BaseBusinessException,
        headers: Optional[Mapping[str, str]] = None,
        media_type: Optional[str] = None,
        background: Optional[BackgroundTask] = None,
    ) -> Response:
        """
        从业务异常创建错误响应
        
        :param exception: 业务异常对象
        :param headers: 可选，响应头信息
        :param media_type: 可选，响应结果媒体类型
        :param background: 可选，响应返回后执行的后台任务
        :return: 错误响应结果
        """
        error_dict = exception.to_dict()
        
        # 根据异常类别确定HTTP状态码
        http_status = ERROR_CODE_TO_HTTP_STATUS.get(
            exception.error_code.category, 
            status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
        # 构建响应内容
        result = {
            'code': exception.error_code.code,
            'msg': exception.user_message,  # 用户友好的错误信息
            'data': {},  # 始终包含data字段，异常情况下为空字典
            'success': False,
            'time': datetime.now(),
            'trace_id': exception.trace_id,
            'retryable': exception.error_code.retryable,
            'category': exception.error_code.category.value,
        }
        
        # 添加调试信息（仅在开发环境）
        if logger.isEnabledFor(logging.DEBUG):
            result['debug'] = {
                'error_message': exception.message,
                'context': error_dict['context'],
                'extra_data': error_dict['extra_data'],
                'cause': error_dict['cause'],
            }
        
        # 特殊处理限流异常
        if exception.error_code.category == ErrorCategory.RATE_LIMIT:
            result['retry_after'] = exception.extra_data.get('retry_after', 60)
            result['available_models'] = exception.extra_data.get('available_models', [])
        
        # 记录错误日志
        log_level = getattr(logging, exception.error_code.log_level, logging.ERROR)
        logger.log(log_level, f"业务异常: {exception.error_code.code} - {exception.message}", 
                  extra={'trace_id': exception.trace_id, 'error_dict': error_dict})
        
        return JSONResponse(
            status_code=http_status,
            content=jsonable_encoder(result),
            headers=headers,
            media_type=media_type,
            background=background,
        )
    
    @classmethod
    def build_dict_response(
        cls,
        code: int = HttpStatusConstant.SUCCESS,
        msg: str = '操作成功',
        data: Optional[Any] = None,
        success: bool = True,
        trace_id: Optional[str] = None
    ) -> Dict:
        """
        构建字典格式响应（用于兼容旧接口）
        
        :param code: 响应状态码
        :param msg: 响应消息
        :param data: 响应数据
        :param success: 是否成功
        :param trace_id: 链路追踪ID
        :return: 字典格式响应
        """
        result = {
            "code": code,
            "msg": msg,
            "data": data,  
            "success": success,
            "time": datetime.now()
        }
        
        if trace_id:
            result["trace_id"] = trace_id
            
        return result 