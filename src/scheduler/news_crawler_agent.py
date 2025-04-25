"""基于 LangChain 的智能新闻爬虫 Agent"""
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sqlalchemy import create_engine, text
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
import urllib3
# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class NewsURLs(BaseModel):
    """新闻URL列表输出模型"""
    urls: List[str] = Field(description="新闻文章的URL列表")

class NewsContent(BaseModel):
    """新闻内容输出模型"""
    title: str = Field(description="新闻标题")
    content: str = Field(description="新闻正文内容")
    publish_date: str = Field(description="发布日期")

class NewsCrawlerAgent:
    """智能新闻爬虫 Agent"""
    
    def __init__(self, db_url: str, zhipuai_api_key: str):
        """初始化爬虫 Agent
        
        Args:
            db_url: 数据库连接URL
            zhipuai_api_key: 智谱AI API密钥
        """
        self.engine = create_engine(db_url)
        self.llm = ChatZhipuAI(
            model="glm-4-flash",
            temperature=0,
            zhipuai_api_key=zhipuai_api_key,
            max_tokens=2048
        )
        
        # 初始化URL提取的提示模板
        self.url_extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的网页分析助手。你的任务是从给定的HTML内容中提取所有新闻文章的URL。
            请分析HTML结构，找出所有新闻链接。通常这些链接会在新闻列表、文章卡片或类似的容器中。
            只返回新闻文章的URL，不要包含其他类型的链接（如导航、广告等）。
            请只返回以下格式的 JSON：
            {{
            "urls": ["url1", "url2", "url3", ..., "url10"]
            }}
            """),
            ("human", "{html_content}")
        ])
        
        # 初始化内容提取的提示模板
        self.content_extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的新闻内容提取助手。你的任务是从给定的HTML内容中提取新闻的关键信息。
            请提取以下字段：
            1. 标题：新闻的主标题
            2. 正文内容：新闻的主要内容，去除广告和无关信息
            3. 发布日期：新闻的发布时间，必须转换为标准的 ISO 8601 格式 (YYYY-MM-DD)
               - 如果原文日期包含具体时间，只保留日期部分
               - 如果原文日期使用中文或其他格式，需要转换为此标准格式
               - 示例：2024-03-15
            
            请确保提取的内容准确完整，并去除HTML标签和多余的空白字符。
            请只返回以下格式的 JSON：
            {{
            "title": "新闻标题",
            "content": "新闻正文",
            "publish_date": "YYYY-MM-DD"
            }}
            """),
            ("human", "{html_content}")
        ])
        
        # 设置输出解析器
        self.url_parser = JsonOutputParser(pydantic_schema=NewsURLs)
        self.content_parser = JsonOutputParser(pydantic_schema=NewsContent)
        
        # 构建处理链
        self.url_chain = (
            {"html_content": RunnablePassthrough()} 
            | self.url_extract_prompt 
            | self.llm 
            | self.url_parser
        )
        
        self.content_chain = (
            {"html_content": RunnablePassthrough()} 
            | self.content_extract_prompt 
            | self.llm 
            | self.content_parser
        )
            
    def fetch_html(self, url: str) -> str:
        """同步获取网页HTML内容"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        try:
            response = requests.get(url, timeout=30, headers=headers, verify=False)
            if response.status_code == 200:
                return response.text
            else:
                print(f"获取页面失败 {url}: HTTP {response.status_code}")
                print(f"Response headers: {response.headers}")
                print(f"Response content: {response.text[:500]}...")  # 只打印前500个字符
                return ""
        except Exception as e:
            print(f"获取页面失败 {url}: {str(e)}")
            return ""
            
    def extract_news_urls(self, html_content: str) -> List[str]:
        """使用LLM提取新闻URL列表"""
        try:
            result = self.url_chain.invoke(html_content)
            return result['urls']
        except Exception as e:
            print(f"提取URL失败: {str(e)}")
            return []
            
    def extract_news_content(self, html_content: str) -> Dict:
        """使用LLM提取新闻内容"""
        try:
            result = self.content_chain.invoke(html_content)
            return {
                "title": result['title'],
                "content": result['content'],
                "publish_date": result['publish_date']
            }
        except Exception as e:
            print(f"提取内容失败: {str(e)}")
            return None
            
    def save_to_db(self, news_data: Dict, source: Dict):
        """保存新闻到数据库"""
        if not news_data:
            return
            
        data = {
            "language": source["language"],
            "source": source["source"],
            "date": news_data["publish_date"],
            "content": f"{news_data['title']}\n\n{news_data['content']}",
            "url": source["url"]
        }
        
        # 检查是否已存在
        check_query = text("""
            SELECT id FROM news 
            WHERE url = :url AND date = :date
        """)
        
        # 插入数据
        insert_query = text("""
            INSERT INTO news (language, source, date, content, url)
            VALUES (:language, :source, :date, :content, :url)
        """)
        
        with self.engine.connect() as conn:
            existing = conn.execute(
                check_query,
                {"url": data["url"], "date": data["date"]}
            ).first()
            
            if not existing:
                conn.execute(insert_query, data)
                conn.commit()
                print(f"成功保存新闻: {data['url']}")
            else:
                print(f"新闻已存在,跳过: {data['url']}")
                
    def process_news_url(self, url: str, source: Dict):
        """处理单个新闻URL"""
        html_content = self.fetch_html(url)
        if html_content:
            news_data = self.extract_news_content(html_content)
            if news_data:   # TODO 出错处理
                source_with_url = dict(source)
                source_with_url["url"] = url
                self.save_to_db(news_data, source_with_url)
                
    def crawl_news(self, source: Dict):
        """爬取单个来源的新闻
        
        Args:
            source: 包含url、language等信息的字典
        """
        try:
            # 获取主页HTML
            html_content = self.fetch_html(source["url"])
            if not html_content:    # TODO 出错处理
                return
                
            # 提取新闻URL列表
            news_urls = self.extract_news_urls(html_content)
            if not news_urls:
                return
                
            # 串行处理所有新闻URL
            for url in news_urls:
                self.process_news_url(url, source)
                
        except Exception as e:
            print(f"爬取失败 {source['source']}: {str(e)}")
            
    def get_source_list(self) -> List[Dict]:
        """从数据库获取需要爬取的网站列表"""
        query = text("""
            SELECT url, language, source_name, info 
            FROM news_sources 
            WHERE is_active = true
        """)
        
        with self.engine.connect() as conn:
            results = conn.execute(query).fetchall()
            
        return [
            {
                "url": row[0],
                "language": row[1],
                "source": row[2],
                "info": row[3]
            }
            for row in results
        ]
        
    def crawl_all(self):
        """爬取所有来源的新闻"""
        sources = self.get_source_list()
        for source in sources:
            self.crawl_news(source) 