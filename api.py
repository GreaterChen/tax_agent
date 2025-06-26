"""FastAPI接口实现"""
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import uuid
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 首先配置日志系统，确保UTF-8编码
from src.utils.logging_config import setup_root_logging
setup_root_logging()

from src.agent import tax_agent
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

# 使用全局Agent实例

class Question(BaseModel):
    """问题请求模型"""
    text: str
    thread_id: Optional[str] = None
    web_search: Optional[bool] = True
    session_files: Optional[List[str]] = []
    enable_rag: Optional[bool] = True
    
class Answer(BaseModel):
    """回答响应模型"""
    answers: List[str]
    thread_id: str

@app.post("/query")
async def query(question: Question, request: Request):
    """处理问答请求
    
    Args:
        question: 包含问题文本和线程ID的请求体
        request: FastAPI请求对象
        
    Returns:
        统一格式的响应
    """
    # 获取追踪ID
    trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4()))
    
    # 参数验证
    if not question.text or not question.text.strip():
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
    thread_id = question.thread_id or f"thread_{uuid.uuid4().hex}"
    
    # 记录请求日志
    logger.info(f"处理查询请求: {question.text[:100]}...", extra={
        'trace_id': trace_id,
        'thread_id': thread_id,
        'web_search': question.web_search,
        'enable_rag': question.enable_rag
    })
    
    try:
        # 调用代理执行查询
        result = await tax_agent.query(
            question.text, 
            thread_id, 
            question.web_search, 
            question.session_files, 
            question.enable_rag
        )
        
        # 构建完整的响应数据，包含所有token和cost信息
        response_data = {
            "answers": result.get("result", []),
            "thread_id": thread_id,
            "request_id": result.get("request_id"),
            "model_used": result.get("model_used"),
            "provider": result.get("provider"),
            "total_cost": result.get("total_cost", 0),
            "currency": result.get("currency", "CNY"),
            
            # 详细的token使用量信息
            "token_usage": result.get("token_usage", {}),
            
            # 详细的成本分解信息
            "cost_breakdown": result.get("cost_breakdown", {}),
        }
        
        return ResponseUtil.success(
            msg="查询成功",
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
            component="tax_agent"
        )
        
        logger.error(f"查询执行失败: {str(e)}", extra={'trace_id': trace_id})
        
        raise ExceptionFactory.create_business_exception(
            error_code=ErrorCode.AGENT_QUERY_FAILED,
            cause=e,
            context=context
        )

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