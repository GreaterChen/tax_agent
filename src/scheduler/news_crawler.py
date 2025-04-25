"""新闻爬虫实现"""
import os
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import asyncio
from .news_crawler_agent import NewsCrawlerAgent

class NewsCrawler:
    """新闻爬虫类"""
    
    def __init__(self, db_url: str):
        """初始化爬虫
        
        Args:
            db_url: 数据库连接URL
        """
        self.agent = NewsCrawlerAgent(
            db_url=db_url,
            zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY"),
        )
        self.scheduler = BackgroundScheduler()
        
    def crawl_all(self):
        """爬取所有来源的新闻"""
        asyncio.run(self.agent.crawl_all())
            
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

