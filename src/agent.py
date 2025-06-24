"""税务问答Agent实现 - 核心调用模块"""
import asyncio
import time
import logging
from typing import List, Dict, Optional

from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver

from config.llm_config import llm_config
from src.utils.llm_selector import llm_selector
from src.utils.tools_manager import tools_manager
from src.utils.prompts import SYSTEM_PROMPT, create_enhanced_question, create_non_rag_question

logger = logging.getLogger(__name__)

class TaxAgent:
    """税务问答Agent - 核心调用类"""
    
    def __init__(self):
        self.memory = MemorySaver()
        logger.info("TaxAgent初始化完成")
    
    def _create_graph(self, tools: List, llm_config_item: Dict) -> StateGraph:
        """根据工具列表和LLM配置创建图"""
        # 动态绑定工具到选定的LLM
        llm_with_tools = llm_config_item["llm"].bind_tools(tools)
        tool_node = ToolNode(tools)
        
        # 创建代理
        agent = create_react_agent(
            model=llm_with_tools,
            tools=tools,
            prompt=SYSTEM_PROMPT
        )
        
        builder = StateGraph(MessagesState)
        builder.add_node("agent", agent)
        builder.add_node("tools", tool_node)

        builder.set_entry_point("agent")
        builder.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                "__end__": END,
            }
        )
        builder.add_edge("tools", "agent")
        return builder.compile(checkpointer=self.memory)

    async def query(self, question: str, thread_id: Optional[str] = None, 
              web_search: bool = True, session_files: Optional[List[str]] = None, 
              enable_rag: bool = True) -> List[str]:
        """执行查询 - 主要入口方法"""
        try:
            # 处理会话文档和问题增强
            enhanced_question, session_vector_tool = self._process_session_files(
                question, session_files, enable_rag, thread_id
            )
            
            # 获取工具列表
            tools = tools_manager.get_tools(
                web_search=web_search,
                session_vector_tool=session_vector_tool
            )
            
            # 智能选择LLM
            selected_llm = await self._select_llm_with_retry(enhanced_question)
            
            # 创建工作流并执行
            workflow = self._create_graph(tools, selected_llm)
            result = await self._execute_workflow(workflow, enhanced_question, thread_id)
            
            return result if result else ["抱歉，未能获取到有效回答"]
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return [f"抱歉，处理您的请求时发生错误: {str(e)}"]
    
    def _process_session_files(self, question: str, session_files: Optional[List[str]], 
                             enable_rag: bool, thread_id: str) -> tuple:
        """处理会话文件和问题增强"""
        session_vector_tool = None
        enhanced_question = question
        
        if session_files and len(session_files) > 0:
            if enable_rag:
                # RAG模式：创建会话级向量搜索工具
                session_vector_tool = tools_manager.create_session_vector_tool(session_files, thread_id)
                enhanced_question = create_enhanced_question(question, session_files)
            else:
                # 非RAG模式：直接读取文件内容
                try:
                    from src.utils.file_utils import read_session_files_content
                    file_contents = read_session_files_content(session_files)
                    if file_contents:
                        enhanced_question = create_non_rag_question(question, file_contents)
                except ImportError:
                    logger.warning("file_utils模块不可用，跳过文件内容读取")
        
        return enhanced_question, session_vector_tool
    
    async def _select_llm_with_retry(self, enhanced_question: str) -> Dict:
        """选择LLM（异步处理）"""
        try:
            # 检查是否已经在事件循环中
            loop = asyncio.get_running_loop()
            return await llm_selector.select_best_llm(enhanced_question)
        except RuntimeError:
            # 如果没有运行中的事件循环，创建新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(llm_selector.select_best_llm(enhanced_question))
            finally:
                loop.close()
    
    async def _execute_workflow(self, workflow, enhanced_question: str, thread_id: str) -> List[str]:
        """执行工作流，支持重试机制"""
        messages = [HumanMessage(content=enhanced_question)]
        config = {
            "configurable": {
                "thread_id": thread_id or "default"
            },
            "recursion_limit": 10
        }
        
        result = []
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                for step in workflow.stream({"messages": messages}, config=config, stream_mode="updates"):
                    if "agent" in step and "messages" in step['agent']:
                        last_msg = step['agent']['messages'][-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            result.append(last_msg.content)
                
                # 如果成功，跳出重试循环
                if result:
                    break
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "quota" in error_msg:
                    # 限流错误处理
                    self._handle_rate_limit_error(enhanced_question, retry, max_retries)
                    if retry < max_retries - 1:
                        # 重新创建工作流
                        selected_llm = await self._select_llm_with_retry(enhanced_question)
                        tools = tools_manager.get_tools(web_search=True)  # 简化重试时的工具配置
                        workflow = self._create_graph(tools, selected_llm)
                        time.sleep(2 ** retry)  # 指数退避
                    else:
                        raise e
                else:
                    # 非限流错误，直接抛出
                    raise e
        
        return result
    
    def _handle_rate_limit_error(self, enhanced_question: str, retry: int, max_retries: int):
        """处理限流错误"""
        logger.warning(f"触发限流错误，第 {retry + 1}/{max_retries} 次重试")
        # 这里可以添加更多的错误处理逻辑，比如通知相关模块等
    
    async def get_status(self) -> Dict:
        """获取Agent状态"""
        try:
            # 检查是否已经在事件循环中
            loop = asyncio.get_running_loop()
            llm_status = await llm_selector.get_usage_status()
        except RuntimeError:
            # 如果没有运行中的事件循环，创建新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                llm_status = loop.run_until_complete(llm_selector.get_usage_status())
            finally:
                loop.close()
        
        config_status = llm_config.get_status()
        tools_info = tools_manager.get_available_tools_info()
        
        return {
            "agent_status": "running",
            "llm_config": config_status,
            "llm_usage": llm_status,
            "tools": tools_info
        }

# 创建全局实例
tax_agent = TaxAgent()
