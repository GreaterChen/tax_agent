"""
工作流管理器
负责创建和执行LangGraph工作流
"""
import logging
from typing import List, Dict, Any, Tuple
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langmem.short_term import SummarizationNode

from src.utils.prompts import SYSTEM_PROMPT
from src.utils.exceptions import ExceptionFactory, ErrorContext
from src.utils.error_codes import ErrorCode

logger = logging.getLogger(__name__)

class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self):
        self.memory = MemorySaver()
        self.file_summary_cache = {}  # 缓存文件总结信息
    
    def create_graph_with_summary(self, tools: List, llm_config_item: Dict) -> StateGraph:
        """创建带有消息总结功能的图"""
        llm = llm_config_item["llm"]
        
        # 从配置中获取阈值
        max_context_tokens = llm_config_item.get("max_context_tokens", 8000)
        summary_trigger_tokens = llm_config_item.get("summary_trigger_tokens", 6000)
        max_summary_tokens = llm_config_item.get("max_summary_tokens", 500)
        
        # 使用较小的模型进行总结以节省成本
        summary_model = llm.bind(max_tokens=max_summary_tokens)
        
        # 创建更安全的总结节点配置
        try:
            summarization_node = SummarizationNode(
                model=summary_model,
                token_counter=count_tokens_approximately,
                max_tokens=max_context_tokens,
                max_tokens_before_summary=summary_trigger_tokens,
                max_summary_tokens=max_summary_tokens,
                input_messages_key="messages",
                output_messages_key="summarized_messages",
                name="summarization"
            )
        except Exception as e:
            logger.error(f"创建总结节点失败: {e}")
            # 如果总结节点创建失败，创建一个简单的直通节点
            def simple_passthrough(state):
                """简单的直通节点，不进行总结"""
                return {"summarized_messages": state.get("messages", [])}
            summarization_node = simple_passthrough
        
        # 动态绑定工具到LLM
        llm_with_tools = llm.bind_tools(tools)
        tool_node = ToolNode(tools)
        
        def call_model(state):
            """调用LLM模型"""
            try:
                # 确保state不为None
                if state is None:
                    state = {}
                
                # 如果有总结后的消息，使用总结后的；否则使用原始消息
                messages = state.get("summarized_messages", state.get("messages", []))
                
                # 确保messages是列表
                if not isinstance(messages, list):
                    messages = []
                
                # 确保系统提示词在消息列表的开头
                if not any(isinstance(msg, SystemMessage) for msg in messages):
                    messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
                
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
                
            except Exception as e:
                logger.error(f"调用LLM模型时出错: {e}")
                # 创建业务异常
                context = ErrorContext(
                    operation="call_model",
                    component="workflow_manager"
                )
                raise ExceptionFactory.create_business_exception(
                    error_code=ErrorCode.LLM_ERROR,
                    cause=e,
                    context=context
                )
        
        # 构建图
        from src.agent import AgentState  # 避免循环导入
        builder = StateGraph(AgentState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", tool_node)
        
        # 添加总结节点到工作流
        builder.add_node("summarize", summarization_node)
        # builder.add_edge(START, "summarize")
        # builder.add_edge("summarize", "call_model")
        builder.add_edge(START, "call_model")
        
        # 添加工具调用的条件边
        builder.add_conditional_edges(
            "call_model",
            tools_condition,
            {
                "tools": "tools",
                "__end__": END,
            }
        )
        builder.add_edge("tools", "call_model")
        
        return builder.compile(checkpointer=self.memory)
    
    async def replace_file_messages_with_summaries(self, messages: List[AnyMessage], thread_id: str) -> List[AnyMessage]:
        """
        在对话历史中替换文件消息为总结内容
        
        Args:
            messages: 原始消息列表
            thread_id: 线程ID
            
        Returns:
            替换后的消息列表
        """
        from src.utils.session_processor import session_processor
        
        # 获取已处理的文件消息
        processed_file_messages = session_processor.get_processed_file_messages(thread_id)
        
        if not processed_file_messages:
            return messages
        
        # 创建文件ID到总结内容的映射
        file_summaries = {}
        for file_msg in processed_file_messages:
            file_info = file_msg.get("file_info", {})
            if file_info.get("is_summary", False):
                # 如果已经是总结，创建映射
                filename = file_info.get("filename", "unknown")
                content_hash = file_info.get("content_hash", "")
                file_id = f"file_{filename}_{content_hash[:8]}" if content_hash else f"file_{filename}"
                file_summaries[file_id] = file_msg.get("content", "")
        
        # 替换消息中的文件内容
        updated_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage) and hasattr(msg, 'additional_kwargs'):
                additional_kwargs = msg.additional_kwargs or {}
                if additional_kwargs.get("is_file_message", False):
                    file_id = additional_kwargs.get("file_id", "")
                    if file_id in file_summaries:
                        # 替换为总结内容
                        updated_msg = HumanMessage(
                            content=file_summaries[file_id],
                            additional_kwargs=additional_kwargs
                        )
                        updated_messages.append(updated_msg)
                        logger.info(f"替换文件消息 {file_id} 为总结内容")
                    else:
                        updated_messages.append(msg)
                else:
                    updated_messages.append(msg)
            else:
                updated_messages.append(msg)
        
        return updated_messages
    
    async def execute_workflow_with_tracking(self, workflow, user_question: str, 
                                           thread_id: str, file_messages: List[Dict] = None) -> Tuple[List[str], List[AIMessage]]:
        """
        执行工作流，返回结果和AI响应对象
        
        Args:
            workflow: 工作流对象
            user_question: 用户问题
            thread_id: 线程ID
            file_messages: 文件消息列表
        
        Returns:
            Tuple[List[str], List[AIMessage]]: (响应文本列表, AI响应对象列表)
        """
        config = {
            "configurable": {
                "thread_id": thread_id or "default"
            },
            "recursion_limit": 10
        }
        
        # 准备初始状态 - 系统提示词 + 文件消息 + 用户问题
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        
        # 添加文件消息（如果有）
        if file_messages:
            for file_msg in file_messages:
                # 为每个文件创建独立的消息，添加文件标识
                file_info = file_msg.get("file_info", {})
                filename = file_info.get("filename", "unknown")
                content_hash = file_info.get("content_hash", "")
                
                # 创建带有文件标识的消息
                file_message = HumanMessage(
                    content=file_msg.get("content", ""),
                    additional_kwargs={
                        "file_info": file_info,
                        "is_file_message": True,
                        "file_id": f"file_{filename}_{content_hash[:8]}" if content_hash else f"file_{filename}"
                    }
                )
                messages.append(file_message)
        
        # 添加用户问题
        messages.append(HumanMessage(content=user_question))
        
        initial_state = {
            "messages": messages,
            "context": {}
        }
        
        # 执行工作流
        result = []
        ai_responses = []
        
        try:
            for step in workflow.stream(initial_state, config=config, stream_mode="updates"):
                # 检查 step 是否为有效字典
                if not isinstance(step, dict):
                    logger.warning(f"工作流步骤不是字典格式: {type(step)}")
                    continue
                
                # 从不同节点收集响应
                for node_name, node_state in step.items():
                    try:
                        # 检查 node_state 是否为 None 或空
                        if node_state is None:
                            logger.debug(f"节点 {node_name} 返回空状态")
                            continue
                        
                        # 确保 node_state 是字典类型
                        if not isinstance(node_state, dict):
                            logger.debug(f"节点 {node_name} 状态不是字典格式: {type(node_state)}")
                            continue
                        
                        # 检查是否有 messages
                        if "messages" in node_state:
                            messages = node_state["messages"]
                            if messages and isinstance(messages, list):  # 确保 messages 不为空且是列表
                                for msg in messages:
                                    if isinstance(msg, AIMessage) and msg.content:
                                        result.append(msg.content)
                                        ai_responses.append(msg)
                                        
                        # 如果有总结信息，记录到日志
                        if ("context" in node_state and 
                            isinstance(node_state["context"], dict) and
                            "running_summary" in node_state["context"]):
                            summary_info = node_state["context"]["running_summary"]
                            if summary_info:
                                logger.info(f"消息已总结，节省token数量")
                    except Exception as e:
                        logger.error(f"处理节点 {node_name} 状态时出错: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"工作流执行出错: {e}")
            # 创建工作流异常
            context = ErrorContext(
                operation="execute_workflow",
                component="workflow_manager",
                extra_data={"thread_id": thread_id}
            )
            raise ExceptionFactory.create_business_exception(
                error_code=ErrorCode.WORKFLOW_ERROR,
                cause=e,
                context=context
            )
        
        return result, ai_responses

# 全局实例
workflow_manager = WorkflowManager() 