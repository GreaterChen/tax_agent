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
from typing import Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_community.chat_models import ChatZhipuAI, ChatTongyi
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import PyPDF2  # 使用PyPDF2替代PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import spacy
from spacy.language import Language
from spacy.tokens import Doc
import langdetect

# 导入LangChain的文档加载器
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_core.documents import Document

# API配置
CSE_API_KEY = 'AIzaSyDFYC1uxFUkjjxQg-DjMmECJCTu2JsE85I'
CSE_ENGINE_ID = '45f253b7863e94f3f'

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
    target_sites: Optional[List[str]] = Field(DEFAULT_NEWS_SOURCES, description="限制搜索的目标网站列表")
    max_results_per_query: int = Field(3, description="每个检索词返回的最大结果数")

class AdvancedWebSearchTool:
    def __init__(self):
        """初始化搜索工具及LLM"""
        self.llm = ChatTongyi(
            model="qwen-max",
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        
        # 初始化结果缓存
        self.result_cache = {}
        self.content_cache = {}
        
        # 初始化向量存储 - 使用主流的多语言嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large"  # 广泛使用的多语言嵌入模型，MTEB排行榜第一
        )
        
        # 初始化spacy模型缓存
        self.nlp_models = {}
        
        # HTML处理配置
        self.html_headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
        ]
        
        # 初始化HTML分割器
        self.html_splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=self.html_headers_to_split_on,
            max_chunk_size=1000,
            separators=["\n\n", "\n", ". ", " ", ""],
            elements_to_preserve=["table", "ul", "ol", "code"]
        )
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        # 设置默认新闻来源
        self.default_news_sources = DEFAULT_NEWS_SOURCES

    def _get_nlp_model(self, lang: str) -> Language:
        """获取对应语言的spacy模型
        
        Args:
            lang: 语言代码 (en/zh-cn/zh-hk)
            
        Returns:
            spacy语言模型
        """
        if lang not in self.nlp_models:
            try:
                # 根据语言加载对应的模型
                if lang in ['zh-cn', 'zh-hk']:
                    # 对于中文（简体和香港繁体），使用中文模型
                    self.nlp_models[lang] = spacy.load("zh_core_web_sm")
                elif lang == 'en':
                    # 英文模型
                    self.nlp_models[lang] = spacy.load("en_core_web_sm")
                else:
                    # 默认使用英文模型
                    self.nlp_models[lang] = spacy.load("en_core_web_sm")
            except OSError:
                # 如果模型未安装，下载并安装
                if lang in ['zh-cn', 'zh-hk']:
                    spacy.cli.download("zh_core_web_sm")
                    self.nlp_models[lang] = spacy.load("zh_core_web_sm")
                else:
                    spacy.cli.download("en_core_web_sm")
                    self.nlp_models[lang] = spacy.load("en_core_web_sm")
                
        return self.nlp_models[lang]

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

    def _extract_keywords_spacy(self, text: str) -> List[str]:
        """使用spacy提取关键词，优化支持简体中文、繁体中文（香港）和英文
        
        Args:
            text: 输入文本
            
        Returns:
            关键词列表
        """
        # 检测语言
        lang = self._detect_language(text)
        
        # 获取对应的spacy模型
        nlp = self._get_nlp_model(lang)
        
        # 针对文本长度进行安全处理
        if len(text) > 10000:
            text = text[:10000]  # 避免处理过长文本
        
        # 处理文本
        doc = nlp(text)
        
        keywords = set()
        
        # 提取名词短语
        for chunk in doc.noun_chunks:
            if len(chunk.text.strip()) > 1:  # 过滤单字符
                keywords.add(chunk.text.strip().lower())
        
        # 提取命名实体
        for ent in doc.ents:
            if len(ent.text.strip()) > 1:
                keywords.add(ent.text.strip().lower())
        
        # 针对不同语言使用不同的提取策略
        if lang == 'en':
            # 英文：提取名词、动词、形容词
            for token in doc:
                if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop and len(token.text.strip()) > 1:
                    keywords.add(token.text.strip().lower())
        else:
            # 中文（简体和繁体）：保留所有非停用词的名词、动词、形容词和副词
            for token in doc:
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop and len(token.text.strip()) > 0:
                    keywords.add(token.text.strip().lower())
        
        return list(keywords)

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的关键词相似度
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            
        Returns:
            相似度分数 (0-1)
        """
        # 使用spacy提取关键词
        keywords1 = set(self._extract_keywords_spacy(text1))
        keywords2 = set(self._extract_keywords_spacy(text2))
        
        if not keywords1 or not keywords2:
            return 0.0
            
        # 计算Jaccard相似度
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0

    def _preprocess_query_for_language(self, query: str, lang: str) -> str:
        """针对不同语言对查询进行预处理
        
        Args:
            query: 原始查询
            lang: 语言代码
            
        Returns:
            预处理后的查询
        """
        processed_query = query
        
        # 对特定语言的查询进行优化
        if lang.startswith('zh'):
            # 中文查询优化：保留完整的中文词组
            pass  # 实际上不需要特殊处理，保持原样
        elif lang == 'en':
            # 英文查询优化：转为小写
            processed_query = query.lower()
        
        return processed_query

    def _hybrid_retrieval(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """使用LangChain实现混合检索（向量检索+关键词检索）
        
        Args:
            query: 搜索查询
            documents: 文档列表
            top_k: 返回结果数量
            
        Returns:
            最相关的文档列表
        """
        # 检测查询语言
        query_lang = self._detect_language(query)
        
        # 对查询预处理
        processed_query = self._preprocess_query_for_language(query, query_lang)
        
        # 创建向量存储
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # 向量检索
        vector_results = vectorstore.similarity_search(processed_query, k=top_k*2)
        
        # 关键词检索
        keyword_scores = {}
        for doc in documents:
            # 计算关键词相似度
            score = self._keyword_similarity(processed_query, doc.page_content)
            keyword_scores[doc.page_content] = score
        
        # 混合排序
        hybrid_scores = {}
        for doc in documents:
            # 检查是否在向量检索结果中
            vector_score = 0.0
            for i, vector_doc in enumerate(vector_results):
                if doc.page_content == vector_doc.page_content:
                    # 计算归一化的向量分数 (越小的索引，分数越高)
                    vector_score = 1.0 - (i / len(vector_results))
                    break
            
            # 获取关键词分数
            keyword_score = keyword_scores.get(doc.page_content, 0.0)
            
            # 根据查询语言调整权重
            if query_lang.startswith('zh'):
                # 中文：更重视关键词匹配
                hybrid_scores[doc.page_content] = 0.5 * vector_score + 0.5 * keyword_score
            else:
                # 英文：均衡权重
                hybrid_scores[doc.page_content] = 0.6 * vector_score + 0.4 * keyword_score
        
        # 排序
        sorted_docs = sorted(
            [(doc, hybrid_scores.get(doc.page_content, 0.0)) for doc in documents],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 返回top_k结果
        return [doc for doc, score in sorted_docs[:top_k]]

    def _generate_search_queries(self, original_query: str, target_sites: Optional[List[str]] = None) -> List[Dict]:
        """生成3组搜索查询
        
        Args:
            original_query: 用户原始查询
            target_sites: 目标网站列表
            
        Returns:
            3组搜索查询列表
        """
        if target_sites is None:
            target_sites = self.default_news_sources
            
        prompt_text = """
        请基于以下原始问题生成3组不同的搜索查询词，以便获得更全面的搜索结果。
        
        原始问题: {0}
        
        {1}
        
        生成的查询词应该：
        1. 提取原始问题中的核心概念和关键词
        2. 使用同义词或相关术语进行扩展
        3. 针对不同角度生成特定查询
        4. 简洁明了，去除不必要的词语
        5. 适合网络搜索引擎使用
        
        请以JSON格式返回结果，格式如下:
        ```json
        [
          {{
            "query": "查询词1",
            "explanation": "此查询词的目的解释"
          }},
          {{
            "query": "查询词2",
            "explanation": "此查询词的目的解释"
          }},
          {{
            "query": "查询词3",
            "explanation": "此查询词的目的解释"
          }}
        ]
        ```
        """
        
        sites_text = f"目标网站: {', '.join(target_sites)}" if target_sites else ""
        prompt = prompt_text.format(original_query, sites_text)
        
        print("生成搜索查询词...")
        response = self.llm.invoke(prompt)
        
        try:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = re.search(r'\[\s*\{.*\}\s*\]', response.content, re.DOTALL).group(0)
            
            search_queries = json.loads(json_str)
            
            # 限制只返回3个查询
            search_queries = search_queries[:3]
            
            # 为每个查询添加目标网站
            if target_sites:
                for query in search_queries:
                    query["target_sites"] = target_sites
            
            print(f"生成了 {len(search_queries)} 组搜索查询")
            return search_queries
            
        except Exception as e:
            print(f"解析搜索查询词失败: {str(e)}")
            # 回退方案
            fallback_queries = [
                {"query": original_query, "explanation": "原始查询", "target_sites": target_sites if target_sites else self.default_news_sources}
            ]
            return fallback_queries

    def _extract_keywords(self, text: str) -> List[str]:
        """提取文本中的关键词
        
        Args:
            text: 输入文本
            
        Returns:
            关键词列表
        """
        # 使用LLM提取关键词
        prompt = f"""
        请从以下文本中提取关键词（包括专业术语、实体名词等），以JSON数组格式返回。
        注意：
        1. 关键词应该是独立的词或短语
        2. 包括专业术语、实体名词、重要动词等
        3. 去除停用词和无意义词语
        
        文本：{text}
        
        请按如下格式返回：
        ```json
        ["关键词1", "关键词2", "关键词3", ...]
        ```
        """
        
        try:
            response = self.llm.invoke(prompt)
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content)
            if json_match:
                keywords = json.loads(json_match.group(1))
                return keywords
            return []
        except Exception as e:
            print(f"提取关键词失败: {str(e)}")
            return []

    def _process_pdf_content(self, pdf_content: bytes, query: str) -> List[str]:
        """处理PDF内容并返回相关片段
        
        Args:
            pdf_content: PDF文件内容
            query: 搜索查询
            
        Returns:
            相关文本片段列表
        """
        try:
            # 使用uuid创建唯一的临时文件名避免并发冲突
            import uuid
            import tempfile
            
            # 创建唯一的临时文件
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
                print(f"清理临时PDF文件失败: {str(e)}")
            
            # 如果没有文档，返回空列表
            if not documents:
                print("PDF中未提取到文本内容")
                return []
            
            # 分割文档
            split_docs = self.text_splitter.split_documents(documents)
            
            # 创建向量存储
            vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            # 检索相关文档
            retrieved_docs = vectorstore.similarity_search(
                query, 
                k=3, 
                fetch_k=5
            )
            
            # 提取文本内容
            relevant_chunks = [doc.page_content for doc in retrieved_docs]
            
            return relevant_chunks
            
        except Exception as e:
            print(f"处理PDF内容时出错: {str(e)}")
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
            
            # 移除不必要的元素
            for tag in ['script', 'style', 'nav', 'footer', 'header', 'iframe', 'meta', 'noscript']:
                for element in soup.find_all(tag):
                    if element:
                        element.extract()
            
            # 提取文本
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # 限制文本长度以避免上下文过长
            if len(text) > 50000:
                text = text[:50000] + "..."
            
            if len(text.strip()) < 50:
                return "内容太少，可能是无效页面"
            
            return text
            
        except Exception as e:
            print(f"处理HTML内容时出错: {str(e)}")
            return f"处理HTML内容时出错: {str(e)}"

    def _fetch_page_content(self, url: str, query: str) -> str:
        """获取网页内容，包括PDF处理
        
        Args:
            url: 网页URL
            query: 搜索查询
            
        Returns:
            处理后的内容
        """
        # 检查缓存
        cache_key = f"{url}_{query}"
        if cache_key in self.content_cache:
            print(f"使用缓存的页面内容: {url}")
            return self.content_cache[cache_key]
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            print(f"获取页面内容: {url}")
            
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            # 处理PDF文件
            if 'application/pdf' in content_type:
                print(f"处理PDF文件: {url}")
                relevant_chunks = self._process_pdf_content(response.content, query)
                if relevant_chunks:
                    content = "\n\n".join(relevant_chunks)
                    self.content_cache[cache_key] = content
                    return content
                return "无法从PDF中提取相关内容"
            
            # 处理HTML内容 - 直接处理不使用RAG
            if 'text/html' in content_type:
                print(f"处理HTML内容: {url}")
                content = self._process_html_content(response.text, query)
                self.content_cache[cache_key] = content
                return content
                
            return f"不支持的内容类型: {content_type}"
            
        except Exception as e:
            print(f"获取页面内容失败 ({url}): {str(e)}")
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
                except Exception as e:
                    print(f"获取内容出错: {str(e)}")
        
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
            return "未找到相关内容。"
        
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
                page_content=result.get("content", ""),  # 限制长度
                metadata=metadata
            )
            documents.append(doc)
        
        # 使用混合检索获取最相关的文档
        relevant_docs = self._hybrid_retrieval(query, documents, top_k=3)
        
        # 构建提示内容
        context_str = ""
        for i, doc in enumerate(relevant_docs):
            domain = urlparse(doc.metadata["url"]).netloc
            context_str += f"[来源 {i+1}] {doc.metadata['title']} ({domain})\n"
            context_str += f"URL: {doc.metadata['url']}\n"
            context_str += f"查询词: {doc.metadata['query']}\n"
            context_str += f"内容: {doc.page_content}...\n\n"
        
        # 检测查询语言
        query_lang = self._detect_language(query)
        
        # 根据语言构建提示
        if query_lang.startswith('zh'):
            prompt_template = """
            请根据以下搜索结果，为用户提供一个全面而准确的回答。

            用户查询: {question}

            搜索结果:
            {context}

            要求:
            1. 直接回答用户的问题
            2. 综合所有来源中的相关信息
            3. 引用信息来源，标明信息来自哪个网页
            4. 如果不同来源有冲突信息，请指出这些不一致
            5. 如果内容中没有足够的信息，请诚实地说明
            6. 不要生成未在提供的内容中明确提到的信息

            请以专业、清晰的方式组织回答，避免不必要的重复。回答应基于事实，不要过度解释或猜测。
            """
        else:
            prompt_template = """
            Based on the following search results, provide a comprehensive and accurate answer to the user's question.

            User Query: {question}

            Search Results:
            {context}

            Requirements:
            1. Answer the user's question directly
            2. Integrate relevant information from all sources
            3. Cite your sources, indicating which webpage the information comes from
            4. If there are inconsistencies between sources, point them out
            5. If there is insufficient information, honestly acknowledge it
            6. Do not generate information not explicitly mentioned in the provided content

            Organize your answer in a professional and clear manner, avoiding unnecessary repetition. The answer should be fact-based without excessive explanation or speculation.
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
            fallback_response = "分析内容时出现错误，以下是找到的信息摘要：\n\n"
            for i, doc in enumerate(relevant_docs[:3]):
                fallback_response += f"- {doc.metadata['title']}\n"
                fallback_response += f"  来源: {doc.metadata['url']}\n"
                fallback_response += f"  摘要: {doc.metadata['snippet']}\n\n"
            return fallback_response

    def _google_search(self, query: str, target_sites: Optional[List[str]] = None, num: int = 3) -> List[Dict]:
        """执行Google搜索
        
        Args:
            query: 查询词
            target_sites: 限制搜索的目标网站
            num: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        # 检查缓存
        cache_key = f"{query}_{'-'.join(target_sites) if target_sites else 'no_sites'}_{num}"
        if cache_key in self.result_cache:
            print(f"使用缓存的搜索结果: {query}")
            return self.result_cache[cache_key]
        
        search_query = query
        if target_sites and len(target_sites) > 0:
            site_query = " OR ".join([f"site:{site}" for site in target_sites])
            search_query = f"{query} ({site_query})"
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": CSE_API_KEY,
            "cx": CSE_ENGINE_ID,
            "q": search_query,
            "num": num
        }
        
        print(f"执行Google搜索: {search_query}")
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            search_results = response.json()
            
            if "items" not in search_results:
                print("未找到搜索结果")
                return []
                
            results = []
            for item in search_results.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "query": query
                })
            
            # 保存到缓存
            self.result_cache[cache_key] = results
            
            print(f"找到 {len(results)} 个搜索结果")
            return results
        except Exception as e:
            print(f"搜索失败: {str(e)}")
            return []

    def search(self, query: str, 
               target_sites: Optional[List[str]] = None, 
               max_results_per_query: int = 3) -> str:
        """执行高级搜索
        
        Args:
            query: 用户原始查询
            target_sites: 限制搜索的目标网站列表
            max_results_per_query: 每个检索词返回的最大结果数
            
        Returns:
            搜索结果分析
        """
        try:
            print(f"处理查询: {query}")
            
            # 如果未指定目标网站，使用默认新闻来源
            if target_sites is None:
                target_sites = self.default_news_sources
                
            print(f"目标网站: {', '.join(target_sites)}")
            
            # 1. 生成多组搜索查询
            search_queries = self._generate_search_queries(query, target_sites)
            
            # 2. 执行多组搜索
            all_search_results = []
            for search_query in search_queries:
                query_text = search_query["query"]
                query_sites = search_query.get("target_sites", target_sites)
                
                # 添加延迟以避免API限制
                time.sleep(1)
                
                results = self._google_search(query_text, query_sites, max_results_per_query)
                all_search_results.extend(results)
            
            if not all_search_results:
                return "未找到相关搜索结果。您可以尝试重新表述问题或使用不同的关键词。"
            
            # 3. 获取网页内容
            results_with_content = self._fetch_contents_parallel(all_search_results, query)
            
            if not results_with_content:
                links_list = "\n".join([f"- {result['title']}: {result['link']}" for result in all_search_results[:5]])
                return f"找到了以下搜索结果，但无法获取详细内容：\n{links_list}\n\n您可以直接访问这些链接查看内容。"
            
            # 4. 分析内容并生成答案
            final_answer = self._analyze_content(query, results_with_content)
            
            print("搜索和分析完成")
            return final_answer
            
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
    description="使用高级技术在互联网上搜索信息，包括查询改写、多组检索词生成和内容分析",
    args_schema=AdvancedWebSearchInput
) 