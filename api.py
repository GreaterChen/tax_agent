"""FastAPI接口实现"""
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import uuid
import logging
import shutil
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 首先配置日志系统，确保UTF-8编码
from src.utils.logging_config import setup_root_logging
setup_root_logging()

from src.agent import async_tax_agent
from src.scheduler.news_crawler import NewsCrawler
from utils.response_util import ResponseUtil
from src.middleware.exception_middleware import setup_exception_handling
from src.utils.exceptions import (
    ExceptionFactory, 
    ErrorContext, 
    ValidationException,
    BaseBusinessException
)
from src.utils.error_codes import ErrorCode
from src.utils.error_monitor import error_monitor

logger = logging.getLogger(__name__)

# 创建爬虫实例
crawler = NewsCrawler(os.getenv("DATABASE_URL"))

# 文件上传临时目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    crawler.start()
    yield
    # 关闭时
    crawler.stop()

# 创建FastAPI应用
app = FastAPI(
    title="税务问答系统",
    description="基于LangChain和FastAPI的税务问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# 设置全局异常处理
setup_exception_handling(app, enable_debug=os.getenv("DEBUG", "false").lower() == "true")

@app.post("/query")
async def query(
    request: Request,
    text: str = Form(...),
    thread_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[])
):
    """处理问答请求（支持可选文件上传） - 完全异步版本
    
    Args:
        request: FastAPI请求对象
        text: 问题文本
        thread_id: 线程ID（可选）
        files: 上传的文件列表（可选）
        
    Returns:
        统一格式的响应
    """
    # 获取追踪ID
    trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4()))
    
    # 参数验证
    if not text or not text.strip():
        context = ErrorContext(
            request_id=trace_id,
            operation="POST /query",
            component="validation"
        )
        raise ExceptionFactory.create_validation_exception(
            error_code=ErrorCode.MISSING_REQUIRED_FIELD,
            field_name="text",
            context=context
        )
    
    # 使用提供的thread_id或生成新的uuid
    thread_id = thread_id or f"async_thread_{uuid.uuid4().hex}"
    
    # 处理上传文件
    temp_file_paths = []
    
    try:
        # 保存上传的文件到临时目录
        for file in files:
            if file.filename:
                # 生成唯一的文件名
                file_id = str(uuid.uuid4())
                file_extension = os.path.splitext(file.filename)[1]
                temp_filename = f"{file_id}_{file.filename}"
                temp_file_path = os.path.join(UPLOAD_DIR, temp_filename)
                
                # 保存文件
                with open(temp_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                temp_file_paths.append(temp_file_path)
                logger.info(f"保存上传文件: {file.filename} -> {temp_file_path}")
        
        # 记录请求日志
        file_count = len(temp_file_paths)
        request_type = "带文件的查询" if file_count > 0 else "纯文本查询"
        logger.info(f"处理异步{request_type}请求: {text[:100]}...", extra={
            'trace_id': trace_id,
            'thread_id': thread_id,
            'file_count': file_count,
            'execution_mode': 'async_only'
        })
        
        # 使用完全异步的agent
        result = await async_tax_agent.query(
            question=text,
            thread_id=thread_id,
            session_files=temp_file_paths if temp_file_paths else None,
            user_id=trace_id
        )
        
        # 构建响应数据
        response_data = {
            "answers": result.get("result", []),
            "thread_id": thread_id,
            "request_id": result.get("request_id"),
            "model_used": result.get("model_used"),
            "provider": result.get("provider"),
            "total_cost": result.get("total_cost", 0),
            "currency": result.get("currency", "CNY"),
            "token_usage": result.get("token_usage", {}),
            "cost_breakdown": result.get("cost_breakdown", {}),
            "execution_mode": "async_unified"
        }
        
        # 添加文件信息
        if result.get("file_info"):
            response_data["file_info"] = result["file_info"]
            
        # 添加错误信息（如果有）
        if result.get("error"):
            response_data["error"] = result["error"]
            response_data["fallback_used"] = result.get("fallback_used", False)
        
        return ResponseUtil.success(
            msg="异步查询成功",
            data=response_data
        )
        
    except BaseBusinessException:
        # 业务异常直接重新抛出，由全局异常处理器处理
        raise
        
    except Exception as e:
        # 包装未知异常为业务异常
        context = ErrorContext(
            request_id=trace_id,
            operation="POST /query",
            component="async_tax_agent"
        )
        
        logger.error(f"异步查询执行失败: {str(e)}", extra={'trace_id': trace_id})
        
        raise ExceptionFactory.create_business_exception(
            error_code=ErrorCode.AGENT_QUERY_FAILED,
            cause=e,
            context=context
        )
    
    finally:
        # 清理临时文件
        for temp_file_path in temp_file_paths:
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    logger.info(f"清理临时文件: {temp_file_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {temp_file_path}, {e}")



@app.get("/health")
async def health_check():
    """系统健康检查"""
    health_status = error_monitor.get_health_status()
    
    return ResponseUtil.success(
        msg="健康检查完成",
        data=health_status
    )

@app.get("/metrics/errors")
async def error_metrics(hours: int = 1):
    """获取错误统计指标"""
    from datetime import timedelta
    
    time_range = timedelta(hours=hours) if hours > 0 else None
    error_stats = error_monitor.get_error_statistics(time_range)
    
    return ResponseUtil.success(
        msg="错误统计获取成功", 
        data=error_stats
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)