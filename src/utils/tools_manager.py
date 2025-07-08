"""
工具管理器模块
负责管理和配置各种Agent工具
"""
import logging
from typing import List
from langchain_core.tools import BaseTool

from src.tools.web_search.web_search_mini import advanced_web_search_tool
from src.tools.examist.examist_tool import examist_tool

logger = logging.getLogger(__name__)

class ToolsManager:
    """工具管理器"""
    
    def __init__(self):
        # 固定工具列表 - 总是启用web搜索和examist
        self.tools = [
            advanced_web_search_tool,
            examist_tool
        ]
        logger.info(f"工具管理器初始化完成，固定工具: {[tool.name for tool in self.tools]}")
    
    def get_tools(self) -> List[BaseTool]:
        """
        获取固定的工具列表
        
        Returns:
            工具列表
        """
        logger.info(f"返回固定工具列表: {[tool.name for tool in self.tools]}")
        return self.tools
    
    def get_available_tools_info(self) -> dict:
        """获取可用工具信息"""
        return {
            "tools": [
                {"name": tool.name, "description": tool.description} 
                for tool in self.tools
            ]
        }

# 全局工具管理器实例
tools_manager = ToolsManager() 