"""新闻爬虫实现"""
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Dict, List

class NewsCrawler:
    """新闻爬虫类"""
    
    def __init__(self, db_url: str):
        """初始化数据库连接
        
        Args:
            db_url: 数据库连接URL
        """
        self.engine = create_engine(db_url)
        self.scheduler = BackgroundScheduler()
        
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
        
    def crawl_news(self, source: Dict):
        """爬取单个来源的新闻
        
        Args:
            source: 包含url、language等信息的字典
        """
        try:
            # 获取网页内容
            response = requests.get(source["url"], timeout=30)
            response.raise_for_status()
            
            # 解析网页
            soup = BeautifulSoup(response.text, "html.parser")
            
            # 提取新闻内容(这里需要根据具体网站结构调整)
            content = soup.find("article").get_text(strip=True)
            
            # 准备数据
            today = datetime.now().strftime("%Y%m%d")
            data = {
                "language": source["language"],
                "source": source["source"],
                "date": today,
                "content": content,
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
                # 检查是否已存在
                existing = conn.execute(
                    check_query,
                    {"url": data["url"], "date": data["date"]}
                ).first()
                
                if not existing:
                    conn.execute(insert_query, data)
                    conn.commit()
                    print(f"成功爬取并保存新闻: {source['source']}")
                else:
                    print(f"新闻已存在,跳过: {source['source']}")
                    
        except Exception as e:
            print(f"爬取失败 {source['source']}: {str(e)}")
            
    def crawl_all(self):
        """爬取所有来源的新闻"""
        sources = self.get_source_list()
        for source in sources:
            self.crawl_news(source)
            
    def start(self):
        """启动定时任务"""
        # 每天凌晨2点执行
        self.scheduler.add_job(
            self.crawl_all,
            'cron',
            hour=2,
            minute=0
        )
        self.scheduler.start()
        
    def stop(self):
        """停止定时任务"""
        self.scheduler.shutdown() 