"""高级网页搜索工具

此工具实现高级搜索功能，包括：
1. 查询改写与多组检索词生成
2. 分步骤搜索
3. 对搜索结果进行聚类分析
4. 结果汇总与优化
"""

import os
import time
import json
import requests
from typing import Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_community.chat_models import ChatZhipuAI
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# API配置
CSE_API_KEY = 'AIzaSyDFYC1uxFUkjjxQg-DjMmECJCTu2JsE85I'
CSE_ENGINE_ID = '45f253b7863e94f3f'

# 输入模型
class AdvancedWebSearchInput(BaseModel):
    """高级网络搜索查询输入"""
    query: str = Field(..., description="用户的原始问题")
    target_sites: Optional[List[str]] = Field(None, description="限制搜索的目标网站列表")
    max_results_per_query: int = Field(5, description="每个检索词返回的最大结果数")
    enable_clustering: bool = Field(True, description="是否对搜索结果进行聚类")

class AdvancedWebSearchTool:
    def __init__(self):
        """初始化搜索工具及LLM"""
        # 初始化LLM，用于查询改写和内容分析
        self.llm = ChatZhipuAI(
            model="glm-4-flash",
            temperature=0.3,
            zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY")
        )
        
        # 初始化结果缓存
        self.result_cache = {}
        self.content_cache = {}
    
    def _generate_search_queries(self, original_query: str, target_sites: Optional[List[str]] = None) -> List[Dict]:
        """生成多组搜索查询
        
        Args:
            original_query: 用户原始查询
            target_sites: 目标网站列表
            
        Returns:
            多组搜索查询列表，每组包含查询词和目标网站
        """
        prompt_text = """
        请基于以下原始问题生成5组不同的搜索查询词，以便获得更全面的搜索结果。
        
        原始问题: {0}
        
        {1}
        
        生成的查询词应该：
        1. 提取原始问题中的核心概念和关键词
        2. 使用同义词或相关术语进行扩展
        3. 针对不同角度或子问题生成特定查询
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
          ...
        ]
        ```
        """
        
        sites_text = f"目标网站: {', '.join(target_sites)}" if target_sites else ""
        prompt = prompt_text.format(original_query, sites_text)
        
        print("生成搜索查询词...")
        response = self.llm.invoke(prompt)
        
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = re.search(r'\[\s*\{.*\}\s*\]', response.content, re.DOTALL).group(0)
            
            search_queries = json.loads(json_str)
            
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
                {"query": original_query, "explanation": "原始查询", "target_sites": target_sites if target_sites else []}
            ]
            return fallback_queries
    
    def _is_suspicious_url(self, url: str) -> bool:
        """检测URL是否可疑（垃圾链接、恶意链接等）
        
        Args:
            url: 要检查的URL
            
        Returns:
            True如果URL可疑，否则False
        """
        suspicious_patterns = [
            r'娱乐', r'博彩', r'赌博', r'棋牌', r'彩票', r'菠菜', r'威尼斯人',
            r'ddos', r'cc攻击', r'黑客', r'暗网', r'黑产',
            r'SEO', r'SEO优化', r'SEO Consulting',
            r'\d+q\.com', r'447q\.com', r'd9828\.com', r'91NA\.COM',
            r'红包', r'股东招商', r'代理', r'注册登录',
            r'催情', r'迷幻水', r'伟哥', r'壮阳',
            r'外汇交易', r'MT5', r'WeChat', r'QQ:',
            r'赚钱', r'赢利', r'投资'
        ]
        
        # 检查URL中是否包含可疑模式
        url_lower = url.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, url_lower, re.IGNORECASE):
                return True
                
        # 检查URL是否包含大量无意义参数
        parsed_url = urlparse(url)
        if len(parsed_url.query) > 100:
            return True
            
        # 不是HTTP或HTTPS链接
        if not url_lower.startswith('http'):
            return True
            
        return False
        
    def _filter_search_results(self, search_results: List[Dict]) -> List[Dict]:
        """过滤搜索结果，移除垃圾链接和无关内容
        
        Args:
            search_results: 原始搜索结果
            
        Returns:
            过滤后的搜索结果
        """
        filtered_results = []
        
        for result in search_results:
            # 如果链接可疑，跳过
            if self._is_suspicious_url(result["link"]):
                print(f"跳过可疑链接: {result['link']}")
                continue
                
            # 过滤包含可疑关键词的标题或摘要
            title_and_snippet = f"{result.get('title', '')} {result.get('snippet', '')}"
            if re.search(r'(博彩|赌博|棋牌|彩票|ddos|攻击|黑客|暗网|红包|股东招商|代理)', 
                         title_and_snippet, re.IGNORECASE):
                print(f"跳过可疑内容: {result['title']}")
                continue
                
            # 添加到过滤后的结果
            filtered_results.append(result)
            
        return filtered_results
    
    def _google_search(self, query: str, target_sites: Optional[List[str]] = None, num: int = 5) -> List[Dict]:
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
            "num": min(10, num + 5)  # 多获取几个结果以备过滤
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
            
            # 过滤结果
            filtered_results = self._filter_search_results(results)
            
            # 限制结果数量
            if len(filtered_results) > num:
                filtered_results = filtered_results[:num]
            
            # 保存到缓存
            self.result_cache[cache_key] = filtered_results
            
            print(f"找到 {len(filtered_results)} 个有效搜索结果")
            return filtered_results
        except Exception as e:
            print(f"搜索失败: {str(e)}")
            return []
    
    def _fetch_page_content(self, url: str) -> str:
        """获取网页内容"""
        # 检查缓存
        if url in self.content_cache:
            print(f"使用缓存的页面内容: {url}")
            return self.content_cache[url]
        
        # 检查URL是否可疑
        if self._is_suspicious_url(url):
            print(f"跳过可疑链接: {url}")
            return "获取页面内容失败: 可疑链接"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            print(f"获取页面内容: {url}")
            
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type and 'text/plain' not in content_type:
                print(f"跳过非文本内容: {content_type}")
                return "获取页面内容失败: 非文本内容"
            
            # 安全地处理编码
            if hasattr(response, 'encoding') and response.encoding:
                if response.encoding.lower() == 'iso-8859-1':
                    if hasattr(response, 'apparent_encoding') and response.apparent_encoding:
                        response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除脚本、样式和导航元素
            if soup:
                for tag in ['script', 'style', 'nav', 'footer', 'header', 'iframe']:
                    for element in soup.find_all(tag):
                        if element:
                            element.extract()
                
                # 获取文本内容
                text = soup.get_text()
                
                # 清理文本（移除多余空白）
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                # 检查文本是否包含可疑内容
                suspicious_patterns = [
                    r'博彩', r'赌博', r'棋牌', r'彩票', r'菠菜', r'威尼斯人',
                    r'ddos', r'cc攻击', r'黑客', r'暗网', r'黑产',
                    r'催情', r'迷幻水', r'伟哥', r'壮阳',
                    r'外汇交易', r'贵金属', r'期货',
                    r'股东招商', r'代理', r'注册登录'
                ]
                
                for pattern in suspicious_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        print(f"跳过可疑内容页面: {url}")
                        return "获取页面内容失败: 可疑内容"
                
                # 限制文本长度
                content = text[:10000]
                
                # 过滤空内容
                if len(content.strip()) < 50:
                    print(f"内容太少，可能是垃圾页面: {url}")
                    return "获取页面内容失败: 内容太少"
                
                # 保存到缓存
                self.content_cache[url] = content
                
                print(f"成功获取内容: {len(content)} 字符")
                return content
            else:
                print(f"无法解析网页内容: {url}")
                return "无法解析网页内容"
        except Exception as e:
            print(f"获取页面内容失败 ({url}): {str(e)}")
            return f"获取页面内容失败: {str(e)}"
    
    def _fetch_contents_parallel(self, search_results: List[Dict], max_workers: int = 5) -> List[Dict]:
        """并行获取多个网页内容
        
        Args:
            search_results: 搜索结果列表
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
        
        # 创建任务列表
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_result = {executor.submit(self._fetch_page_content, result["link"]): result for result in unique_results}
            
            # 获取结果
            for future in as_completed(future_to_result):
                result = future_to_result[future]
                try:
                    content = future.result()
                    if not content.startswith("获取页面内容失败"):
                        result_with_content = result.copy()
                        result_with_content["content"] = content
                        results_with_content.append(result_with_content)
                except Exception as e:
                    print(f"获取内容出错: {str(e)}")
        
        return results_with_content
    
    def _cluster_results(self, results_with_content: List[Dict]) -> List[Dict]:
        """对结果进行聚类分析
        
        Args:
            results_with_content: 包含内容的搜索结果列表
            
        Returns:
            聚类后的结果列表
        """
        if not results_with_content or len(results_with_content) <= 1:
            return results_with_content
            
        prompt = """
        请分析以下搜索结果，并将它们聚类成不同的主题组。结果应该是一个JSON数组，每个元素包含一个主题和属于该主题的结果索引。

        搜索结果:
        """
        
        for i, result in enumerate(results_with_content):
            prompt += f"\n[{i}] 标题: {result['title']}\n摘要: {result['snippet'][:200]}...\n"
        
        prompt += """
        请按照以下格式返回聚类结果:
        ```json
        [
          {
            "topic": "主题1描述",
            "indices": [0, 2, 5]
          },
          {
            "topic": "主题2描述",
            "indices": [1, 3, 4]
          }
        ]
        ```
        请确保每个搜索结果只出现在一个聚类中，且所有结果都被聚类。
        """
        
        try:
            print("对搜索结果进行聚类分析...")
            response = self.llm.invoke(prompt)
            
            # 提取JSON部分
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = re.search(r'\[\s*\{.*\}\s*\]', response.content, re.DOTALL).group(0)
                
            clusters = json.loads(json_str)
            
            # 重组结果
            clustered_results = []
            for cluster in clusters:
                topic = cluster["topic"]
                for idx in cluster["indices"]:
                    if 0 <= idx < len(results_with_content):
                        result = results_with_content[idx].copy()
                        result["cluster"] = topic
                        clustered_results.append(result)
            
            # 确保所有结果都包含在内
            included_indices = set()
            for cluster in clusters:
                included_indices.update(cluster["indices"])
            
            # 添加未聚类的结果
            for i, result in enumerate(results_with_content):
                if i not in included_indices:
                    result_copy = result.copy()
                    result_copy["cluster"] = "其他"
                    clustered_results.append(result_copy)
            
            print(f"聚类完成，共 {len(clusters)} 个主题")
            return clustered_results
            
        except Exception as e:
            print(f"聚类分析失败: {str(e)}")
            # 如果聚类失败，返回原始结果
            for result in results_with_content:
                result["cluster"] = "默认"
            return results_with_content
    
    def _analyze_content(self, query: str, results_with_content: List[Dict]) -> str:
        """使用LLM分析页面内容，提取有用信息
        
        Args:
            query: 原始查询
            results_with_content: 包含内容的搜索结果列表
            
        Returns:
            分析结果
        """
        if not results_with_content:
            return "未找到相关内容。"
        
        # 将内容预处理为合适的长度，避免超出LLM上下文窗口
        processed_contents = []
        for result in results_with_content:
            # 为每个内容分配约3000字符的空间
            max_length = 3000
            content = result.get("content", "")
            
            if len(content) > max_length:
                content_preview = content[:max_length] + "..."
            else:
                content_preview = content
                
            processed_contents.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "content": content_preview,
                "cluster": result.get("cluster", "默认"),
                "query": result.get("query", "")
            })
        
        # 按主题组织内容
        clusters = {}
        for content in processed_contents:
            cluster = content["cluster"]
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(content)
        
        combined_content = f"用户查询: {query}\n\n"
        
        # 将聚类内容组织到一起
        for cluster, contents in clusters.items():
            combined_content += f"## 主题: {cluster}\n\n"
            for i, content in enumerate(contents):
                domain = urlparse(content["link"]).netloc
                combined_content += f"来源 {i+1} [{domain}] (查询词: {content['query']})\n"
                combined_content += f"标题: {content['title']}\n"
                combined_content += f"网址: {content['link']}\n"
                combined_content += f"内容摘要: {content['content'][:500]}...\n\n"
        
        prompt = f"""
        我需要你分析以下搜索结果的内容，找出与用户查询最相关的信息，并给出一个全面而准确的回答。
        
        用户查询: {query}
        
        搜索结果内容:
        {combined_content}
        
        请根据以上内容，为用户提供一个全面而准确的回答。你的回答应该：
        1. 直接回答用户的问题
        2. 综合所有来源中的相关信息
        3. 按照主题组织信息，确保回答结构清晰
        4. 引用信息来源，标明信息来自哪个网页
        5. 如果不同来源有冲突信息，请指出这些不一致
        6. 如果内容中没有足够的信息，请诚实地说明
        7. 不要生成未在提供的内容中明确提到的信息
        8. 如果提供的内容中包含垃圾信息，请忽略这些内容
        
        回答格式应该专业、清晰，避免不必要的重复。确保你的回答是基于事实的，不要过度解释或猜测。
        """
        
        try:
            print("分析搜索结果内容...")
            response = self.llm.invoke(prompt)
            return self._validate_and_improve_answer(query, response.content, processed_contents)
        except Exception as e:
            print(f"内容分析失败: {str(e)}")
            # 简单的后备选项
            fallback_response = "分析内容时出现错误，以下是找到的信息摘要：\n\n"
            for cluster, contents in clusters.items():
                fallback_response += f"## {cluster}\n\n"
                for content in contents[:2]:  # 每个主题最多显示2条结果
                    fallback_response += f"- {content['title']}\n"
                    fallback_response += f"  来源: {content['link']}\n"
                    fallback_response += f"  摘要: {content['snippet']}\n\n"
            return fallback_response
            
    def _validate_and_improve_answer(self, query: str, answer: str, contents: List[Dict]) -> str:
        """验证和改进生成的答案，确保准确性和完整性
        
        Args:
            query: 用户查询
            answer: 生成的回答
            contents: 处理过的内容列表
            
        Returns:
            验证和改进后的回答
        """
        # 将所有内容组合在一起，用于验证
        all_content = "\n\n".join([
            f"{content['title']}\n{content['snippet']}\n{content['content'][:1000]}"
            for content in contents
        ])
        
        validate_prompt = f"""
        请验证以下回答，确保其中的信息全部来自提供的内容，且没有添加未在内容中提到的任何信息。

        用户查询: {query}
        
        生成的回答:
        {answer}
        
        内容来源:
        {all_content}
        
        请指出回答中任何未在内容中明确提到的信息。如果发现任何不准确或未被支持的信息，请提供修正。
        返回修正后的回答，或者如果原回答完全准确，则返回原回答。
        """
        
        try:
            validation_response = self.llm.invoke(validate_prompt)
            
            # 简单检查是否提到了不准确信息
            validation_text = validation_response.content.lower()
            if "不准确" in validation_text or "未提及" in validation_text or "未在内容中" in validation_text:
                # 如果发现不准确信息，使用验证后的修正回答
                print("发现不准确信息，使用验证后的回答")
                
                # 提取验证后的回答
                validated_answer = validation_response.content
                
                # 如果验证回答太长，可能包含了分析过程，尝试提取修正后的部分
                if len(validated_answer) > len(answer) * 1.5:
                    sections = validated_answer.split("\n\n")
                    for i, section in enumerate(sections):
                        if "修正后的回答" in section or "以下是修正的回答" in section or "正确的回答" in section:
                            if i + 1 < len(sections):
                                return sections[i + 1]
                
                return validated_answer
            
            # 如果没有发现不准确信息，使用原始回答
            return answer
        except Exception as e:
            print(f"验证答案失败: {str(e)}")
            return answer  # 如果验证失败，返回原始回答
    
    def search(self, query: str, 
               target_sites: Optional[List[str]] = None, 
               max_results_per_query: int = 5,
               enable_clustering: bool = True) -> str:
        """执行高级搜索
        
        Args:
            query: 用户原始查询
            target_sites: 限制搜索的目标网站列表
            max_results_per_query: 每个检索词返回的最大结果数
            enable_clustering: 是否对搜索结果进行聚类
            
        Returns:
            搜索结果分析
        """
        try:
            print(f"处理查询: {query}")
            if target_sites:
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
            results_with_content = self._fetch_contents_parallel(all_search_results)
            
            if not results_with_content:
                links_list = "\n".join([f"- {result['title']}: {result['link']}" for result in all_search_results[:5]])
                return f"找到了以下搜索结果，但无法获取详细内容：\n{links_list}\n\n您可以直接访问这些链接查看内容。"
            
            # 4. 对结果进行聚类分析（可选）
            if enable_clustering and len(results_with_content) > 3:
                clustered_results = self._cluster_results(results_with_content)
            else:
                # 不进行聚类，使用默认分组
                for result in results_with_content:
                    result["cluster"] = "默认"
                clustered_results = results_with_content
            
            # 5. 分析内容并生成答案
            final_answer = self._analyze_content(query, clustered_results)
            
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
    description="使用高级技术在互联网上搜索信息，包括查询改写、多组检索词生成、结果聚类和内容分析",
    args_schema=AdvancedWebSearchInput
) 