"""
完全异步的工具管理器
负责异步调用各种工具，支持并发执行
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AsyncToolsManager:
    """完全异步的工具管理器"""
    
    def __init__(self):
        # 导入所有异步工具
        from src.tools.intention_recognition import intention_recognition_tool
        from src.tools.general_response import general_response_tool  
        from src.tools.self_introduction import self_introduction_tool
        from src.tools.plans_pricing_tool import plans_pricing_tool
        from src.tools.final_summary import final_summary_tool
        from src.tools.examist.examist_tool import examist_tool
        from src.tools.web_search.web_search_mini import advanced_web_search_tool
        
        self.intention_tool = intention_recognition_tool
        self.general_tool = general_response_tool
        self.self_intro_tool = self_introduction_tool
        self.pricing_tool = plans_pricing_tool
        self.summary_tool = final_summary_tool
        self.examist_tool = examist_tool
        self.web_search_tool = advanced_web_search_tool
        
        logger.info("异步工具管理器初始化完成")
    
    async def intention_recognition(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """异步意图识别"""
        try:
            # 由于工具本身可能还是同步的，我们在线程中运行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.intention_tool.func,
                messages
            )
            logger.info("异步意图识别完成")
            return result
        except Exception as e:
            logger.error(f"异步意图识别失败: {e}")
            return {"Intentions": [], "Lang": "en"}
    
    async def general_response(self, query: str, lang: str = "en") -> str:
        """异步通用回答"""
        try:
            from src.tools.general_response import general_response_tool_instance
            
            # 调用异步方法
            result = await general_response_tool_instance.generate_response(query, lang)
            return result.get("response", "抱歉，无法生成回答。")
        except Exception as e:
            logger.error(f"异步通用回答失败: {e}")
            return "抱歉，处理您的请求时遇到了问题。"
    
    async def examist_analysis(self, query: str) -> str:
        """异步香港税务分析"""
        try:
            from src.tools.examist.examist_tool import examist_tool_instance
            
            # 调用异步方法
            result = await examist_tool_instance.analyze_tax_query(query)
            return result
        except Exception as e:
            logger.error(f"异步税务分析失败: {e}")
            return f"税务分析遇到技术问题: {str(e)}"
    
    async def web_search(self, query: str) -> str:
        """异步网络搜索"""
        try:
            # 使用线程池执行同步工具
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.web_search_tool.func,
                query
            )
            return result
        except Exception as e:
            logger.error(f"异步网络搜索失败: {e}")
            return f"网络搜索遇到问题: {str(e)}"
    
    async def plans_pricing(self, query: str, lang: str = "en") -> str:
        """异步产品推荐"""
        try:
            from src.tools.plans_pricing_tool import plans_pricing_tool_instance
            
            # 调用异步方法
            result = await plans_pricing_tool_instance.recommend_products(query, lang)
            return result.get("response", "抱歉，无法生成产品推荐。")
        except Exception as e:
            logger.error(f"异步产品推荐失败: {e}")
            return f"产品推荐遇到问题: {str(e)}"
    
    async def self_introduction(self, query: str, lang: str = "en") -> str:
        """异步自我介绍"""
        try:
            from src.tools.self_introduction import self_introduction_tool_instance
            
            # 调用异步方法
            result = await self_introduction_tool_instance.introduce_self(query, lang)
            return result.get("response", "抱歉，无法生成自我介绍。")
        except Exception as e:
            logger.error(f"异步自我介绍失败: {e}")
            return f"自我介绍遇到问题: {str(e)}"
    
    async def final_summary(self, query: str, tool_results: List[Dict[str, Any]]) -> str:
        """异步最终总结"""
        try:
            from src.tools.final_summary import final_summary_tool_instance
            
            # 构建消息格式
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=query)]
            
            # 创建空的意图结果
            intention_result = {"Intentions": [], "Lang": "zh-cn"}
            
            # 调用异步方法
            result = await final_summary_tool_instance.generate_final_summary(
                messages=messages,
                intention_result=intention_result,
                tool_result=tool_results,
                original_query=query
            )
            return result.get("response", "抱歉，无法生成最终总结。")
        except Exception as e:
            logger.error(f"异步最终总结失败: {e}")
            return f"最终总结遇到问题: {str(e)}"
    
    async def execute_tools_concurrently(self, intentions: List[Dict[str, Any]], query: str) -> List[str]:
        """并发执行多个工具"""
        tasks = []
        
        for intention in intentions:
            code = intention.get("Code", "")
            content = intention.get("Content", query)
            lang = intention.get("Lang", "en")  # 假设意图中包含语言信息
            
            if code == "A":  # 香港税务分析
                task = self.examist_analysis(content)
            elif code == "B":  # 网络搜索
                task = self.web_search(content)
            elif code == "C":  # 产品推荐
                task = self.plans_pricing(content, lang)
            elif code == "D":  # 自我介绍
                task = self.self_introduction(content, lang)
            elif code == "E":  # 通用回答
                task = self.general_response(content, lang)
            else:
                # 未知意图，使用通用回答
                task = self.general_response(content, lang)
            
            tasks.append(task)
        
        # 并发执行所有任务
        try:
            logger.info(f"开始并发执行{len(tasks)}个工具任务")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果，将异常转换为错误消息
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"工具{i+1}执行失败: {str(result)}"
                    logger.error(error_msg)
                    processed_results.append(error_msg)
                else:
                    processed_results.append(str(result))
            
            logger.info(f"并发工具执行完成，{len(processed_results)}个结果")
            return processed_results
            
        except Exception as e:
            logger.error(f"并发工具执行失败: {e}")
            return [f"并发执行失败: {str(e)}"]

# 全局异步工具管理器实例
tools_manager = AsyncToolsManager() 