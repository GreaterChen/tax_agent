"""FastAPI接口实现"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import uuid
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from src.agent import tax_agent
from src.scheduler.news_crawler import NewsCrawler
from utils.response_util import ResponseUtil

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
async def query(question: Question):
    """处理问答请求
    
    Args:
        question: 包含问题文本和线程ID的请求体
        
    Returns:
        统一格式的响应
    """
    try:
        # 使用提供的thread_id或生成新的uuid
        thread_id = question.thread_id or f"thread_{uuid.uuid4().hex}"
        
        answers = await tax_agent.query(question.text, thread_id, question.web_search, 
                                  question.session_files, question.enable_rag)
        
        # 构建响应数据
        response_data = {
            "answers": answers,
            "thread_id": thread_id
        }
        
        return ResponseUtil.success(
            msg="查询成功",
            data=response_data
        )
    except Exception as e:
        return ResponseUtil.error(
            msg=f"处理请求时发生错误: {str(e)}",
            data=None
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 