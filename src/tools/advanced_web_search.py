"""高级网页搜索工具

此工具实现阿里云联网搜索功能，包括：
1. 查询改写与多组检索词生成
2. 阿里云搜索API调用
3. 结果汇总与优化
"""

import os
import time
import json
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_community.chat_models import ChatTongyi
import langdetect

# API配置
from alibabacloud_tea_openapi.models import Config
from alibabacloud_searchplat20240529.client import Client
from alibabacloud_searchplat20240529.models import GetWebSearchRequest

# API配置
ALIYUN_API_KEY = 'OS-69n85n4c6v27922e'
ALIYUN_ENDPOINT = 'default-0jqr.platform-cn-shanghai.opensearch.aliyuncs.com'
ALIYUN_WORKSPACE = 'default'
ALIYUN_SERVICE_ID = 'ops-web-search-001'

# 默认搜索网站列表
DEFAULT_SEARCH_SITES = {
    "ACCA": {
        "domain": "accaglobal.com/hk",
        "description": "ACCA考试程序性通知与会员资格、会费等事项。仅英文",
        "languages": ["en"]
    },
    "HKICPA": {
        "domain": "hkicpa.org.hk",
        "description": "HKICPA考试程序性通知与会员资格、会费等事项。三语",
        "languages": ["en", "zh-cn", "zh-hk"]
    },
    "IRD": {
        "domain": "ird.gov.hk/eng",
        "description": "香港的税收政策法规新闻/行政机构介绍等。三语",
        "languages": ["en", "zh-cn", "zh-hk"]
    },
    "CHINATAX": {
        "domain": "chinatax.gov.cn/chinatax",
        "description": "和中国大陆税收相关的新闻/行政机构介绍等。",
        "languages": ["zh-cn"]
    }
}

# 输入模型
class AdvancedWebSearchInput(BaseModel):
    """高级网络搜索查询输入"""
    query: str = Field(..., description="用户的原始问题")

class AdvancedWebSearchTool:
    def __init__(self):
        """初始化搜索工具及LLM"""
        self.llm = ChatTongyi(
            model="qwen-long",
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        # 初始化结果缓存
        self.result_cache = {}

    def _aliyun_search(self, query: str) -> List[Dict]:
        """执行阿里云联网搜索
        
        Args:
            query: 查询词
            
        Returns:
            搜索结果列表
        """
        # 检查缓存
        cache_key = query
        if cache_key in self.result_cache:
            print(f"使用缓存的搜索结果: {query}")
            return self.result_cache[cache_key]
        
        try:
            # 配置阿里云客户端
            config = Config(
                bearer_token=ALIYUN_API_KEY,
                endpoint=ALIYUN_ENDPOINT,
                protocol="http"
            )
            client = Client(config=config)
            
            # 构建网站限制条件
            # site_filter = " OR ".join([f"site:{site['domain']}" for site in DEFAULT_SEARCH_SITES.values()])
            # filtered_query = f"({query}) AND ({site_filter})"
            filtered_query = f"{query}"
            
            # 创建请求
            request = GetWebSearchRequest(
                query=filtered_query,
                way="full",
                top_k=10  # 增加返回结果数量，因为有网站过滤可能会减少有效结果
            )
            
            print(f"执行阿里云联网搜索: {filtered_query}")
            
            # 发送请求
            response = client.get_web_search(ALIYUN_WORKSPACE, ALIYUN_SERVICE_ID, request)
            search_results = response.body.to_map()
            
            if "result" not in search_results or "search_result" not in search_results["result"]:
                print("未找到搜索结果")
                return []
            
            # 过滤和处理结果
            results = []
            for item in search_results["result"]["search_result"]:
                url = item.get("link", "").lower()
                # 添加来源网站信息
                source_site = next(
                    (name for name, site in DEFAULT_SEARCH_SITES.items() 
                        if site["domain"].lower() in url),
                    "Unknown"
                )
                
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "content": item.get("content", ""),
                    "source": source_site,
                    "query": query
                })
            
            # 保存到缓存
            self.result_cache[cache_key] = results
            
            print(f"找到 {len(results)} 个来自指定网站的搜索结果")
            return results
        except Exception as e:
            print(f"搜索失败: {str(e)}")
            return []

    def search(self, query: str) -> str:
        """执行搜索
        
        Args:
            query: 用户原始查询
            
        Returns:
            搜索结果分析
        """
        try:
            print(f"处理查询: {query}")
            
            # 执行搜索
            results = self._aliyun_search(query)
            
            if not results:
                return "未找到相关搜索结果。您可以尝试重新表述问题或使用不同的关键词。"
            
            # 构建提示内容
            context_str = ""
            for i, result in enumerate(results):
                context_str += f"[Source {i+1}]\n"
                context_str += f"URL: {result.get('link', '')}\n"
                context_str += f"Title: {result.get('title', '')}\n"
                context_str += f"Summary: {result.get('snippet', '')}\n"
                context_str += f"Content: {result.get('content', '')}\n\n"
            
            # 检测查询语言
            is_chinese = langdetect.detect(query).startswith('zh')
            
            # 根据语言构建提示
            if is_chinese:
                prompt_template = """
                请根据以下搜索结果，用中文为用户提供一个全面而准确的回答。必须使用中文回答！

                用户查询: {question}

                搜索结果:
                {context}

                要求:
                1. 必须使用中文回答
                2. 直接回答用户的问题
                3. 综合所有来源中的相关信息
                4. 引用信息来源，标明信息来自哪个网页，必须包含完整的URL链接
                5. 每个关键信息点后都应该添加对应的URL引用，格式为 [来源: URL]
                6. 如果不同来源有冲突信息，请指出这些不一致
                7. 如果内容中没有足够的信息，请诚实地说明
                8. 不要生成未在提供的内容中明确提到的信息

                请以专业、清晰的方式组织回答，避免不必要的重复。回答应基于事实，不要过度解释或猜测。每个重要陈述都需要有相应的URL引用支持。
                """
            else:
                prompt_template = """
                Based on the following search results, provide a comprehensive and accurate answer in English. You MUST answer in English!

                User Query: {question}

                Search Results:
                {context}

                Requirements:
                1. MUST answer in English
                2. Answer the user's question directly
                3. Integrate relevant information from all sources
                4. Cite your sources by including the complete URL for each webpage referenced
                5. Add URL citations after each key point in the format [Source: URL]
                6. If there are inconsistencies between sources, point them out
                7. If there is insufficient information, honestly acknowledge it
                8. Do not generate information not explicitly mentioned in the provided content

                Organize your answer in a professional and clear manner, avoiding unnecessary repetition. The answer should be fact-based without excessive explanation or speculation. Each significant statement should be supported by a URL citation.
                """
            
            # 替换占位符
            prompt = prompt_template.format(
                question=query,
                context=context_str
            )
            
            try:
                print("生成回答...")
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as e:
                print(f"回答生成失败: {str(e)}")
                return "分析内容时出现错误"
            
        except Exception as e:
            print(f"搜索过程中发生未预期的错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"搜索过程中发生错误: {str(e)}"

# 创建工具实例
advanced_web_search_instance = AdvancedWebSearchTool()

# 封装为StructuredTool
advanced_web_search_tool = StructuredTool.from_function(
    func=advanced_web_search_instance.search,
    name="advanced_web_search",
    description="使用阿里云联网搜索API搜索信息并生成答案。回答的结果是最终版，不要再尝试重复查询类似问题",
    args_schema=AdvancedWebSearchInput
) 