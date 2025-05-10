"""简化版Web搜索工具，使用Google搜索API实现，无本地处理"""
import os
import requests
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.tools import StructuredTool

# 输入模型
class WebSearchInput(BaseModel):
    """Web搜索输入"""
    query: str = Field(..., description="搜索查询词")
    max_results: int = Field(5, description="返回的最大结果数")

# API配置
CSE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyDFYC1uxFUkjjxQg-DjMmECJCTu2JsE85I')
CSE_ENGINE_ID = os.getenv('GOOGLE_CSE_ID', '45f253b7863e94f3f')

class WebSearchTool:
    def __init__(self):
        """初始化搜索工具"""
        pass
        
    def search(self, query: str, max_results: int = 5) -> str:
        """进行网络搜索
        
        Args:
            query: 搜索查询词
            max_results: 返回的最大结果数
            
        Returns:
            搜索结果文本
        """
        try:
            # 使用Google自定义搜索API
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': CSE_API_KEY,
                'cx': CSE_ENGINE_ID,
                'q': query,
                'num': min(max_results, 10)  # Google API最多支持10个结果
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                return f"搜索请求失败: HTTP {response.status_code}, {response.text}"
                
            data = response.json()
            
            if 'items' not in data or not data['items']:
                return "未找到相关搜索结果。"
                
            results = []
            for i, item in enumerate(data['items'], 1):
                title = item.get('title', '无标题')
                link = item.get('link', '#')
                snippet = item.get('snippet', '无摘要内容')
                
                results.append(f"[{i}] {title}\n链接: {link}\n摘要: {snippet}\n")
                
            return "\n".join(results)
        except Exception as e:
            return f"搜索过程中发生错误: {str(e)}"

# 实例化工具
search_tool = WebSearchTool()
web_search_tool = StructuredTool.from_function(
    func=search_tool.search,
    name="web_search",
    description="通过互联网搜索查找信息，特别适合查询最新的税务政策、法规和解释",
    args_schema=WebSearchInput
) 