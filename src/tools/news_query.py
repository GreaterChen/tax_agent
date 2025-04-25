import os
from typing import Union
from datetime import datetime
from langchain_core.tools import StructuredTool
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field

# 输入模型
class NewsQueryInput(BaseModel):
    """新闻查询输入"""
    language: str = Field(..., description="语言代码，取值范围：zh_sim / zh_hk / eng")
    number: int = Field(..., description="需要返回的新闻数量", ge=1, le=10)

# 输出模型
class NewsQueryOutput(BaseModel):
    """新闻查询结果"""
    content: str
    url: str
    source: str
    date: str

class NewsQueryError(BaseModel):
    """错误结果"""
    error: str

class NewsQueryTool:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def query(self, language: str, number: int) -> str:
        """根据语言和指定数量从新闻数据库中查询最新的新闻内容和链接"""
        try:
            query = text("""
                SELECT content, url, source, date 
                FROM news 
                WHERE language = :language 
                ORDER BY date DESC, id DESC 
                LIMIT :number
            """)

            with self.engine.connect() as conn:
                results = conn.execute(query, {
                    "language": language,
                    "number": number
                }).fetchall()

            if results:
                news_list = [
                    f"[{i+1}] 来源：{row[2]} 日期：{row[3]}\n内容：{row[0]}\n链接：{row[1]}"
                    for i, row in enumerate(results)
                ]
                return "\n\n".join(news_list)
            else:
                return "未找到符合条件的新闻记录。"

        except Exception as e:
            return f"查询失败: {str(e)}"

# 工具封装为 StructuredTool
query_tool_instance = NewsQueryTool(os.getenv("DATABASE_URL"))
news_query_tool = StructuredTool.from_function(
    func=query_tool_instance.query,
    name="news_query",
    description="根据语言代码和数量查询新闻内容",
    args_schema=NewsQueryInput
)
