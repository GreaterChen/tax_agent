"""新闻查询工具实现"""
from typing import Dict, Optional
from datetime import datetime
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from pydantic import BaseModel

class NewsQueryInput(BaseModel):
    """新闻查询输入模型"""
    language: str
    source: str
    date: str
    
class NewsQueryOutput(BaseModel):
    """新闻查询输出模型"""
    content: str
    url: str

class NewsQueryTool:
    """新闻查询工具类"""
    
    def __init__(self, db_url: str):
        """初始化数据库连接
        
        Args:
            db_url: 数据库连接URL
        """
        self.engine = create_engine(db_url)
        
    @tool("news_query")
    def query(self, language: str, source: str, date: str) -> Dict[str, str]:
        """从数据库查询新闻
        
        Args:
            language: 语言代码 (zh_sim/zh_hk/eng)
            source: 新闻来源
            date: 日期 (格式: YYYYMMDD)
            
        Returns:
            Dict[str, str]: 包含新闻内容和URL的字典
        """
        try:
            # 验证日期格式
            datetime.strptime(date, "%Y%m%d")
            
            # 构建SQL查询
            query = text("""
                SELECT content, url 
                FROM news 
                WHERE language = :language 
                AND source = :source 
                AND date = :date
                ORDER BY id DESC 
                LIMIT 1
            """)
            
            # 执行查询
            with self.engine.connect() as conn:
                result = conn.execute(
                    query,
                    {"language": language, "source": source, "date": date}
                ).first()
                
            if result:
                return {
                    "content": result[0],
                    "url": result[1]
                }
            else:
                return {
                    "error": f"未找到符合条件的新闻: language={language}, source={source}, date={date}"
                }
                
        except ValueError:
            return {"error": "日期格式错误,应为YYYYMMDD"}
        except Exception as e:
            return {"error": f"查询错误: {str(e)}"} 