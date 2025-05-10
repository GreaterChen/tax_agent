"""FastAPI接口实现 (精简版本)"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

from src.agent import TaxAgent

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    yield
    # 关闭时
    pass

# 创建FastAPI应用
app = FastAPI(
    title="税务问答系统",
    description="基于LangChain和FastAPI的税务问答系统 (精简版)",
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
        # 使用提供的thread_id或生成新的
        thread_id = question.thread_id or f"thread_{len(question.text)}"
        
        answers = agent.query(question.text, thread_id)
        return Answer(answers=answers, thread_id=thread_id)
    except Exception as e:
        import traceback
        error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # 控制台输出
        with open("error.log", "a") as f:
            f.write(f"{error_msg}\n")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 