import os
import time
from typing import Union, Dict, Any
from datetime import datetime
from langchain_core.tools import StructuredTool
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field

# 输入模型
class NewsQueryInput(BaseModel):
    """新闻查询输入"""
    language: str = Field(..., description="语言代码，取值范围：zh-cn / zh-hk / en")
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

    def query(self, language: str, number: int) -> Dict[str, Any]:
        """根据语言和指定数量从新闻数据库中查询最新的新闻内容和链接（支持使用统计）"""
        start_time = time.time()
        
        try:
            # 转换语言格式以兼容旧格式
            lang_mapping = {
                "zh-cn": "zh-cn",
                "zh-hk": "zh-hk", 
                "en": "en",
                # 兼容旧格式
                "zh_sim": "zh-cn",
                "zh_hk": "zh-hk",
                "eng": "en",
                "Sim": "zh-cn",
                "Trad": "zh-hk",
                "Eng": "en"
            }
            db_language = lang_mapping.get(language, language)
            
            query = text("""
                SELECT content, url, source, date 
                FROM news 
                WHERE language = :language 
                ORDER BY date DESC, id DESC 
                LIMIT :number
            """)

            with self.engine.connect() as conn:
                results = conn.execute(query, {
                    "language": db_language,
                    "number": number
                }).fetchall()

            processing_time = time.time() - start_time
            
            if results:
                news_list = [
                    f"[{i+1}] 来源：{row[2]} 日期：{row[3]}\n内容：{row[0]}\n链接：{row[1]}"
                    for i, row in enumerate(results)
                ]
                response = "\n\n".join(news_list)
            else:
                response = "未找到符合条件的新闻记录。"

            # 构建包含使用统计的完整结果
            result = {
                "response": response,
                "usage_info": {
                    "request_id": f"news_query_{int(time.time())}",
                    "model_used": "database_query",
                    "provider": "local",
                    "total_cost": 0.0,
                    "currency": "CNY",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "cost_breakdown": {"input_cost": 0, "output_cost": 0, "total_cost": 0},
                    "processing_time": processing_time
                }
            }
            
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_response = f"查询失败: {str(e)}"
            
            return {
                "response": error_response,
                "usage_info": {
                    "request_id": f"news_query_error_{int(time.time())}",
                    "model_used": "database_query",
                    "provider": "local",
                    "total_cost": 0.0,
                    "currency": "CNY",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "cost_breakdown": {"input_cost": 0, "output_cost": 0, "total_cost": 0},
                    "processing_time": processing_time
                }
            }

# 工具封装为 StructuredTool
query_tool_instance = NewsQueryTool(os.getenv("DATABASE_URL"))
news_query_tool = StructuredTool.from_function(
    func=query_tool_instance.query,
    name="news_query",
    description="根据语言代码和数量查询新闻内容",
    args_schema=NewsQueryInput
)
