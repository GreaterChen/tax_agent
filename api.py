"""FastAPI接口实现"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import uuid

from src.agent import TaxAgent
from src.scheduler.news_crawler import NewsCrawler

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

# 创建Agent实例
agent = TaxAgent()

class Question(BaseModel):
    """问题请求模型"""
    text: str
    thread_id: Optional[str] = None
    
class Answer(BaseModel):
    """回答响应模型"""
    answers: List[str]
    thread_id: str

@app.post("/query", response_model=Answer)
async def query(question: Question):
    """处理问答请求
    
    Args:
        question: 包含问题文本和线程ID的请求体
        
    Returns:
        Answer: 包含回答列表和线程ID的响应
    """
    try:
        # 使用提供的thread_id或生成新的uuid
        thread_id = question.thread_id or f"thread_{uuid.uuid4().hex}"
        
        answers = agent.query(question.text, thread_id)
        return Answer(answers=answers, thread_id=thread_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理请求时发生错误: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 