"""
工具管理器模块
负责管理和配置各种Agent工具
"""
import os
import logging
from typing import List, Optional
from langchain_core.tools import BaseTool

from src.tools.latex_calc.latex_calc import latex_calc_tool
from src.tools.news_query import news_query_tool
from src.tools.web_search.web_search import advanced_web_search_tool
from src.tools.vector_search.vector_search import vector_search_tool

logger = logging.getLogger(__name__)

class ToolsManager:
    """工具管理器"""
    
    def __init__(self):
        # 基础工具（始终可用）
        self.base_tools = [
            latex_calc_tool,
            vector_search_tool
        ]
        
        # 可选工具
        self.web_search_tool = advanced_web_search_tool
        self.news_query_tool = news_query_tool
        
        logger.info("工具管理器初始化完成")
    
    def get_tools(self, web_search: bool = True, news_query: bool = False, 
                  session_vector_tool: Optional[BaseTool] = None) -> List[BaseTool]:
        """
        根据配置获取工具列表
        
        Args:
            web_search: 是否启用网络搜索
            news_query: 是否启用新闻查询
            session_vector_tool: 会话级向量搜索工具
            
        Returns:
            工具列表
        """
        tools = []
        
        # 添加会话级向量搜索工具（优先级最高）
        if session_vector_tool:
            tools.append(session_vector_tool)
            logger.info("添加会话级向量搜索工具")
        
        # 添加基础工具
        tools.extend(self.base_tools)
        logger.info(f"添加基础工具: {[tool.name for tool in self.base_tools]}")
        
        # 添加可选工具
        if web_search:
            tools.append(self.web_search_tool)
            logger.info("添加网络搜索工具")
        
        if news_query:
            tools.append(self.news_query_tool)
            logger.info("添加新闻查询工具")
        
        return tools
    
    def create_session_vector_tool(self, session_files: List[str], thread_id: str) -> Optional[BaseTool]:
        """
        创建会话级向量搜索工具
        
        Args:
            session_files: 会话文件列表
            thread_id: 线程ID
            
        Returns:
            会话级向量搜索工具或None
        """
        try:
            from src.tools.session_vector_search.session_vector_search import create_session_vector_tool
            tool = create_session_vector_tool(session_files, thread_id)
            if tool:
                logger.info(f"成功创建会话级向量搜索工具，文件数: {len(session_files)}")
            return tool
        except Exception as e:
            logger.error(f"创建会话级向量搜索工具失败: {str(e)}")
            return None
    
    def get_available_tools_info(self) -> dict:
        """获取可用工具信息"""
        return {
            "base_tools": [
                {"name": tool.name, "description": tool.description} 
                for tool in self.base_tools
            ],
            "optional_tools": [
                {"name": self.web_search_tool.name, "description": self.web_search_tool.description, "type": "web_search"},
                {"name": self.news_query_tool.name, "description": self.news_query_tool.description, "type": "news_query"}
            ]
        }

# 全局工具管理器实例
tools_manager = ToolsManager() 