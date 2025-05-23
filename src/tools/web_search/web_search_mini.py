"""高级网页搜索工具

此工具实现高级搜索功能，包括：
1. 查询改写与多组检索词生成
2. 分步骤搜索
3. 对搜索结果进行RAG处理
4. 结果汇总与优化
"""

import os
import time
import json
import requests
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_community.chat_models import ChatZhipuAI, ChatTongyi
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import PyPDF2 
import spacy
from spacy.language import Language
from spacy.tokens import Doc
import langdetect
import numpy as np

# 导入LangChain的文档加载器
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.documents import Document

# 配置日志
def setup_logging():
    """配置日志系统"""
    # 创建logs目录（如果不存在）
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 生成日志文件名，包含日期
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = logs_dir / f"search_log_{current_date}.log"
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger("advanced_web_search")

# 初始化日志记录器
logger = setup_logging()

# 默认新闻来源
DEFAULT_NEWS_SOURCES = [
    "accaglobal.com/hk",       # ACCA官网
    "hkicpa.org.hk",                    # HKICPA官网
    "ird.gov.hk/eng",       # 香港税务局官网
    "chinatax.gov.cn/chinatax/"  # 国家税务总局官网
]

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
        self.content_cache = {}
        
        # 初始化spacy模型缓存
        self.nlp_models = {}
        
        # 设置默认新闻来源
        self.default_news_sources = DEFAULT_NEWS_SOURCES

    def _detect_language(self, text: str) -> str:
        """检测文本语言
        
        Args:
            text: 输入文本
            
        Returns:
            语言代码 (en/zh-cn/zh-hk)
        """
        try:
            lang = langdetect.detect(text)
            # 针对简体中文、繁体中文（香港）和英文进行优化
            lang_map = {
                'zh-cn': 'zh-cn',  # 简体中文
                'zh-tw': 'zh-hk',  # 台湾繁体中文映射到香港繁体中文
                'zh-hk': 'zh-hk',  # 香港繁体中文
                'zh': 'zh-cn',     # 未指定的中文默认为简体
                'en': 'en'         # 英文
            }
            return lang_map.get(lang, 'en')  # 默认返回英语
        except:
            return 'en'  # 检测失败时默认使用英语


    def _generate_search_queries(self, original_query: str) -> List[Dict]:
        """生成搜索查询
        
        Args:
            original_query: 用户原始查询
            
        Returns:
            包含原始查询和生成的查询列表
        """
        target_sites = self.default_news_sources
        
        # 检测查询语言
        query_lang = self._detect_language(original_query)
        
        # 根据语言选择提示模板
        if query_lang.startswith('zh'):
            prompt_text = """
            请基于以下原始问题生成2组不同的中文搜索查询词，以便获得更全面的搜索结果。
            
            原始问题: {0}
            
            {1}
            
            生成的查询词要求：
            1. 必须使用中文
            2. 提取原始问题中的核心概念和关键词
            3. 使用同义词或相关术语进行扩展
            4. 针对不同角度生成特定查询
            5. 简洁明了，去除不必要的词语
            6. 适合网络搜索引擎使用
            
            请以JSON格式返回结果，格式如下:
            ```json
            [
              {{
                "query": "中文查询词1",
                "explanation": "此查询词的目的解释"
              }},
              {{
                "query": "中文查询词2",
                "explanation": "此查询词的目的解释"
              }}
            ]
            ```
            """
        else:
            prompt_text = """
            Please generate 2 different English search queries based on the original question below to get more comprehensive search results.
            
            Original Question: {0}
            
            {1}
            
            Requirements for generated queries:
            1. Must be in English
            2. Extract core concepts and keywords from the original question
            3. Use synonyms or related terms for expansion
            4. Generate specific queries from different angles
            5. Be concise and remove unnecessary words
            6. Suitable for search engines
            
            Please return the result in JSON format as follows:
            ```json
            [
              {{
                "query": "English query 1",
                "explanation": "Purpose of this query"
              }},
              {{
                "query": "English query 2",
                "explanation": "Purpose of this query"
              }}
            ]
            ```
            """
        
        sites_text = f"目标网站: {', '.join(target_sites)}" if query_lang.startswith('zh') else f"Target sites: {', '.join(target_sites)}"
        prompt = prompt_text.format(original_query, sites_text)
        
        print(f"生成{'中文' if query_lang.startswith('zh') else '英文'}搜索查询词...")
        response = self.llm.invoke(prompt)
        
        try:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = re.search(r'\[\s*\{.*\}\s*\]', response.content, re.DOTALL).group(0)
            
            search_queries = json.loads(json_str)
            
            # 限制只返回2个生成的查询
            search_queries = search_queries[:2]
            
            # 为每个查询添加目标网站
            for query in search_queries:
                query["target_sites"] = target_sites
            
            # 添加原始查询到查询列表的开头
            original_query_dict = {
                "query": original_query,
                "explanation": "原始查询" if query_lang.startswith('zh') else "Original query",
                "target_sites": target_sites
            }
            search_queries.insert(0, original_query_dict)
            
            print(f"生成了 {len(search_queries)} 组搜索查询（包括原始查询）" if query_lang.startswith('zh') else 
                  f"Generated {len(search_queries)} search queries (including original query)")
            return search_queries
            
        except Exception as e:
            error_msg = f"解析搜索查询词失败: {str(e)}" if query_lang.startswith('zh') else f"Failed to parse search queries: {str(e)}"
            print(error_msg)
            # 回退方案
            fallback_queries = [
                {
                    "query": original_query, 
                    "explanation": "原始查询" if query_lang.startswith('zh') else "Original query", 
                    "target_sites": target_sites
                }
            ]
            return fallback_queries

    def _process_pdf_content(self, pdf_content: bytes, query: str) -> List[str]:
        """处理PDF内容并返回所有文本片段
        
        Args:
            pdf_content: PDF文件内容
            query: 搜索查询
            
        Returns:
            所有文本片段列表
        """
        try:
            # 创建唯一的临时文件
            import uuid
            import tempfile
            
            temp_dir = tempfile.gettempdir()
            unique_filename = f"pdf_{uuid.uuid4().hex}.pdf"
            temp_pdf_path = os.path.join(temp_dir, unique_filename)
            
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_content)
            
            # 使用LangChain的PyPDFLoader处理PDF
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()
            
            # 清理临时文件
            try:
                os.remove(temp_pdf_path)
            except Exception as e:
                logger.warning(f"清理临时PDF文件失败: {str(e)}")
            
            # 如果没有文档，返回空列表
            if not documents:
                logger.warning("PDF中未提取到文本内容")
                return []
            
            logger.info(f"从PDF中提取了 {len(documents)} 页内容")
            
            # 直接提取文本内容，不进行分割和检索
            all_chunks = [doc.page_content for doc in documents]
            
            logger.info(f"返回所有 {len(all_chunks)} 个PDF文本片段，不进行检索筛选")
            
            return all_chunks
            
        except Exception as e:
            logger.error(f"处理PDF内容时出错: {str(e)}", exc_info=True)
            return []

    def _process_html_content(self, html_content: str, query: str) -> str:
        """处理HTML内容并返回文本内容
        
        Args:
            html_content: HTML内容
            query: 搜索查询
            
        Returns:
            处理后的文本内容
        """
        try:
            # 使用BeautifulSoup处理HTML内容
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 只移除明确不包含有用内容的元素
            for tag in ['script', 'style', 'footer', 'header', 'nav']:
                for element in soup.find_all(tag):
                    if element:
                        element.extract()
            
            # 提取文本，保留更多的结构信息
            text = soup.get_text(separator='\n')
            
            # 清理文本（去除多余的空行但保留基本结构）
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line:  # 只过滤完全空的行
                    lines.append(line)
            
            text = '\n'.join(lines)
            
            # 限制文本长度以避免上下文过长，但提高上限
            if len(text) > 100000:
                logger.warning(f"HTML内容过长 ({len(text)} 字符)，截断为 100000 字符")
                text = text[:100000] + "..."
            
            if len(text.strip()) < 20:  # 降低无效页面的判断阈值
                return "内容太少，可能是无效页面"
            
            return text
            
        except Exception as e:
            logger.error(f"处理HTML内容时出错: {str(e)}", exc_info=True)
            return f"处理HTML内容时出错: {str(e)}"

    def _fetch_page_content(self, url: str, query: str) -> str:
        """获取网页内容，包括PDF处理
        
        Args:
            url: 网页URL
            query: 搜索查询
            
        Returns:
            处理后的内容
        """
        # 记录开始获取内容
        logger.info(f"开始获取页面内容: {url}")
        
        # 检查缓存
        cache_key = f"{url}_{query}"
        if cache_key in self.content_cache:
            logger.info(f"使用缓存的页面内容: {url}")
            return self.content_cache[cache_key]
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            logger.info(f"发送HTTP请求: {url}")
            
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            logger.info(f"页面内容类型: {content_type}")
            
            # 处理PDF文件
            if 'application/pdf' in content_type:
                logger.info(f"处理PDF文件: {url}")
                relevant_chunks = self._process_pdf_content(response.content, query)
                if relevant_chunks:
                    content = "\n\n".join(relevant_chunks)
                    # 记录提取的内容长度
                    logger.info(f"从PDF提取了 {len(relevant_chunks)} 个相关片段，总长度: {len(content)} 字符")
                    # 记录内容摘要
                    content_summary = content[:200] + "..." if len(content) > 200 else content
                    logger.debug(f"PDF内容摘要: {content_summary}")
                    self.content_cache[cache_key] = content
                    return content
                logger.warning(f"无法从PDF中提取相关内容: {url}")
                return "无法从PDF中提取相关内容"
            
            # 处理HTML内容 - 直接处理不使用RAG
            if 'text/html' in content_type:
                logger.info(f"处理HTML内容: {url}")
                content = self._process_html_content(response.text, query)
                # 记录提取的内容长度
                logger.info(f"从HTML提取了内容，长度: {len(content)} 字符")
                # 记录内容摘要
                content_summary = content[:200] + "..." if len(content) > 200 else content
                logger.debug(f"HTML内容摘要: {content_summary}")
                self.content_cache[cache_key] = content
                return content
            
            logger.warning(f"不支持的内容类型: {content_type}")
            return f"不支持的内容类型: {content_type}"
            
        except Exception as e:
            logger.error(f"获取页面内容失败 ({url}): {str(e)}", exc_info=True)
            return f"获取页面内容失败: {str(e)}"

    def _fetch_contents_parallel(self, search_results: List[Dict], query: str, max_workers: int = 3) -> List[Dict]:
        """并行获取多个网页内容
        
        Args:
            search_results: 搜索结果列表
            query: 搜索查询
            max_workers: 最大线程数
            
        Returns:
            包含内容的搜索结果列表
        """
        unique_links = set()
        unique_results = []
        
        # 去重
        for result in search_results:
            if result["link"] not in unique_links:
                unique_links.add(result["link"])
                unique_results.append(result)
        
        logger.info(f"开始并行获取 {len(unique_results)} 个唯一网页内容 (去重后)")
        
        results_with_content = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_result = {
                executor.submit(self._fetch_page_content, result["link"], query): result 
                for result in unique_results
            }
            
            for future in as_completed(future_to_result):
                result = future_to_result[future]
                try:
                    content = future.result()
                    if not content.startswith(("获取页面内容失败", "不支持的内容类型", "内容太少")):
                        result_with_content = result.copy()
                        result_with_content["content"] = content
                        results_with_content.append(result_with_content)
                        logger.info(f"成功获取内容: {result['link']}")
                    else:
                        logger.warning(f"获取内容失败: {result['link']} - {content[:100]}")
                except Exception as e:
                    logger.error(f"获取内容出错: {result['link']} - {str(e)}", exc_info=True)
        
        logger.info(f"并行获取完成，成功获取 {len(results_with_content)}/{len(unique_results)} 个网页内容")
        return results_with_content

    def _analyze_content(self, query: str, results_with_content: List[Dict]) -> str:
        """使用LangChain风格分析内容并生成答案
        
        Args:
            query: 原始查询
            results_with_content: 包含内容的搜索结果列表
            
        Returns:
            分析结果
        """
        if not results_with_content:
            logger.warning("没有可分析的内容")
            return "未找到相关内容。" if self._detect_language(query).startswith('zh') else "No relevant content found."
        
        logger.info(f"开始分析 {len(results_with_content)} 个网页内容")
        
        # 将结果转换为文档格式
        documents = []
        for result in results_with_content:
            # 提取元数据
            metadata = {
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "query": result.get("query", "")
            }
            
            # 创建文档
            doc = Document(
                page_content=result.get("content", ""),
                metadata=metadata
            )
            documents.append(doc)
        
        # 直接使用所有文档，不进行检索筛选
        logger.info(f"使用所有 {len(documents)} 个文档进行分析，不进行检索筛选")
        
        # 构建提示内容
        context_str = ""
        for i, doc in enumerate(documents):
            domain = urlparse(doc.metadata["url"]).netloc
            context_str += f"[Source {i+1}] {doc.metadata['title']} ({domain})\n"
            context_str += f"URL: {doc.metadata['url']}\n"
            context_str += f"Summary: {doc.metadata['snippet']}\n"
            
            # 如果内容太长，截取摘要
            content = doc.page_content
            # if len(content) > 3000:
            #     logger.info(f"文档 {i+1} 内容过长 ({len(content)} 字符)，截取前 3000 字符作为摘要")
            #     content = content[:3000] + "..."
            
            context_str += f"Content: {content}\n\n"
            
            logger.info(f"文档 {i+1}: {doc.metadata['title']} - {doc.metadata['url']}")
        
        # 检测查询语言
        query_lang = self._detect_language(query)
        logger.info(f"检测到查询语言: {query_lang}")
        
        # 根据语言构建提示
        if query_lang.startswith('zh'):
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
        
        logger.info("生成提示完成，开始调用LLM生成回答")
        
        try:
            logger.info("调用LLM生成回答")
            response = self.llm.invoke(prompt)
            logger.info("LLM回答生成成功")
            return response.content
        except Exception as e:
            logger.error(f"回答生成失败: {str(e)}", exc_info=True)
            error_msg = "分析内容时出现错误，以下是找到的信息摘要：\n\n" if query_lang.startswith('zh') else "Error occurred during analysis. Here is a summary of the information found:\n\n"
            for i, doc in enumerate(documents[:3]):
                error_msg += f"- {doc.metadata['title']}\n"
                error_msg += f"  Source: {doc.metadata['url']}\n"
                error_msg += f"  Summary: {doc.metadata['snippet']}\n\n"
            return error_msg

    def _bing_search(self, query: str) -> List[Dict]:
        """执行搜索
        
        Args:
            query: 查询词
            
        Returns:
            搜索结果列表
        """
        # 记录搜索查询
        logger.info(f"搜索查询: {query}")
        
        # 检查缓存
        cache_key = f"{query}"
        if cache_key in self.result_cache:
            logger.info(f"使用缓存的搜索结果: {query}")
            return self.result_cache[cache_key]
        
        # 使用新的API端点
        url = f"http://8.216.81.217:8000/search"
        params = {
            "keyword": query
        }
        
        logger.info(f"执行搜索请求: {url}?keyword={quote_plus(query)}")
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            search_data = response.json()
            
            # 记录API响应
            logger.debug(f"搜索API响应: {json.dumps(search_data, ensure_ascii=False)}")
            
            if "results" not in search_data or not search_data["results"]:
                logger.warning("未找到搜索结果")
                return []
                
            results = []
            for item in search_data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("abstract", ""),
                    "query": query
                })
            
            # 保存到缓存
            self.result_cache[cache_key] = results
            
            logger.info(f"找到 {len(results)} 个搜索结果")
            
            # 记录搜索结果的URL
            for i, result in enumerate(results):
                logger.info(f"结果 {i+1}: {result['title']} - {result['link']}")
                
            return results
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}", exc_info=True)
            return []

    def search(self, query: str) -> str:
        """执行高级搜索
        
        Args:
            query: 用户原始查询
            
        Returns:
            搜索结果分析
        """
        try:
            # 记录搜索开始
            logger.info(f"===== 开始新的搜索任务 =====")
            logger.info(f"用户查询: {query}")
            
            # 1. 直接使用原始查询执行搜索
            search_results = self._bing_search(query)
            
            if not search_results:
                logger.warning(f"未找到相关搜索结果: {query}")
                return "未找到相关搜索结果。您可以尝试重新表述问题或使用不同的关键词。"
            
            # 2. 获取网页内容
            logger.info(f"开始获取 {len(search_results)} 个网页的内容")
            results_with_content = self._fetch_contents_parallel(search_results, query)
            
            if not results_with_content:
                logger.warning("未能获取任何网页的内容")
                links_list = "\n".join([f"- {result['title']}: {result['link']}" for result in search_results[:5]])
                return f"找到了以下搜索结果，但无法获取详细内容：\n{links_list}\n\n您可以直接访问这些链接查看内容。"
            
            logger.info(f"成功获取了 {len(results_with_content)} 个网页的内容")
            
            # 3. 分析内容并生成答案
            logger.info("开始分析内容并生成回答")
            final_answer = self._analyze_content(query, results_with_content)
            
            # 记录生成的回答
            answer_summary = final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
            logger.info(f"生成的回答摘要: {answer_summary}")
            
            logger.info("===== 搜索任务完成 =====")
            return final_answer
            
        except Exception as e:
            logger.error(f"搜索过程中发生未预期的错误: {str(e)}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            return f"搜索过程中发生错误: {str(e)}"

# 创建工具实例
advanced_web_search_instance = AdvancedWebSearchTool()

# 封装为StructuredTool
advanced_web_search_tool = StructuredTool.from_function(
    func=advanced_web_search_instance.search,
    name="advanced_web_search",
    description="使用高级技术在互联网上搜索信息，包括查询改写、多组检索词生成和内容分析",
    args_schema=AdvancedWebSearchInput
) 