"""
工作流管理器
负责创建和执行基于意图识别的LangGraph工作流
"""
import logging
import json
from typing import List, Dict, Any, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from src.utils.prompts import SYSTEM_PROMPT
from src.utils.exceptions import ExceptionFactory, ErrorContext
from src.utils.error_codes import ErrorCode
from src.tools.intention_recognition import intention_recognition_tool
from src.tools.web_search.web_search_mini import advanced_web_search_tool
from src.tools.self_introduction import self_introduction_tool
from src.tools.general_response import general_response_tool
from src.tools.plans_pricing_tool import plans_pricing_tool
from src.tools.final_summary import final_summary_tool

logger = logging.getLogger(__name__)

from src.tools.examist.examist_tool import examist_tool

class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self):
        self.memory = MemorySaver()
        self.file_summary_cache = {}  # 缓存文件总结信息
    
    def create_intention_based_workflow(self, llm_config_item: Dict) -> StateGraph:
        """创建基于意图识别的工作流"""
        llm = llm_config_item["llm"]
        
        # 注释：移除了总结功能以简化实现
        # 基于意图识别的工作流不需要复杂的消息总结
        
        def intention_recognition_node(state):
            """意图识别节点 - 必须执行的第一步，包含容错处理"""
            logger.info("开始执行意图识别")
            
            try:
                # 获取消息历史 - 如果没有messages就报错
                messages = state["messages"]
                
                # 提取用户的原始查询（最后一条用户消息）
                original_query = ""
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        original_query = msg.content
                        break
                
                # 构建对话历史用于意图识别
                message_history = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        message_history.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        message_history.append({"role": "assistant", "content": msg.content})
                
                # 调用意图识别工具
                intention_result = intention_recognition_tool.func(messages=message_history)
                
                # 验证意图识别结果的格式
                if not isinstance(intention_result, dict):
                    raise ValueError("意图识别结果不是字典格式")
                
                if "Intentions" not in intention_result:
                    raise ValueError("意图识别结果缺少Intentions字段")
                
                intentions = intention_result["Intentions"]
                if not isinstance(intentions, list) or len(intentions) == 0:
                    # 意图为空的情况 - 直接跳过工具执行
                    logger.warning("意图识别结果为空，将跳过工具执行直接进入总结阶段")
                    context = state.get("context", {})
                    context["skip_tools"] = True  # 标记跳过工具执行
                    context["original_query"] = original_query
                    context["empty_intentions"] = True  # 标记意图为空
                    
                    return {
                        "context": context,
                        "messages": messages
                    }
                
                # 验证每个意图的格式
                for i, intention in enumerate(intentions):
                    if not isinstance(intention, dict):
                        raise ValueError(f"第{i+1}个意图不是字典格式")
                    if "Code" not in intention or "Content" not in intention:
                        raise ValueError(f"第{i+1}个意图缺少必要字段Code或Content")
                
                # 将意图识别结果和原始查询存储到状态中
                context = state.get("context", {})
                context["intention_result"] = intention_result
                context["original_query"] = original_query
                context["skip_tools"] = False  # 正常执行工具
                
                logger.info(f"意图识别完成: {intention_result}")
                
                return {
                    "context": context,
                    "messages": messages  # 保持原有消息不变，不添加中间结果
                }
                
            except Exception as e:
                logger.error(f"意图识别失败: {e}")
                
                # 提取用户的原始查询
                original_query = ""
                for msg in reversed(state["messages"]):
                    if isinstance(msg, HumanMessage):
                        original_query = msg.content
                        break
                
                # 容错处理：标记为直接总结模式，跳过工具执行
                context = state.get("context", {})
                context["skip_tools"] = True  # 标记跳过工具执行
                context["original_query"] = original_query
                context["intention_recognition_error"] = str(e)  # 记录错误信息
                
                logger.warning(f"意图识别失败，将跳过工具执行直接进入总结阶段: {e}")
                
                return {
                    "context": context,
                    "messages": state["messages"]
                }
        
        def intention_router(state):
            """意图路由器 - 决定是否跳过工具执行"""
            context = state["context"]
            skip_tools = context.get("skip_tools", False)
            
            if skip_tools:
                logger.info("跳过工具执行，直接进入最终总结")
                return "final_summary"
            else:
                logger.info("继续执行工具")
                return "multi_tool_executor"
        
        def multi_tool_executor_node(state):
            """多工具执行器节点 - 根据多个意图顺序执行相应的工具"""
            try:
                logger.info("开始执行多工具处理")
                
                context = state["context"]
                intention_result = context["intention_result"]
                intentions = intention_result["Intentions"]
                lang = intention_result.get("Lang", "en")
                
                # 存储所有工具的执行结果
                tool_results = []
                
                # 按顺序处理每个意图，但先去重
                unique_intentions = []
                seen_codes = set()
                for intention in intentions:
                    code = intention["Code"]
                    # 对于相同的意图代码，只保留第一个
                    if code not in seen_codes:
                        unique_intentions.append(intention)
                        seen_codes.add(code)
                    else:
                        logger.warning(f"检测到重复的意图代码 {code}，已跳过")
                
                logger.info(f"去重后处理{len(unique_intentions)}个意图: {[intent['Code'] for intent in unique_intentions]}")
                for i, intention in enumerate(unique_intentions):
                    code = intention["Code"]
                    content = intention["Content"]
                    
                    logger.info(f"处理第{i+1}个意图: Code={code}, Content={content[:100]}...")
                    
                    try:
                        if code == "A":  # 做题家 -> Examist工具
                            logger.info("调用Examist工具")
                            result = examist_tool.func(query=content)
                            tool_results.append({
                                "tool": "examist",
                                "intention": intention,
                                "result": result,
                                "success": True
                            })
                            
                        elif code == "B":  # 一般询问 -> websearch工具
                            logger.info("调用网络搜索工具")
                            result = advanced_web_search_tool.func(query=content)
                            tool_results.append({
                                "tool": "web_search",
                                "intention": intention,
                                "result": result,
                                "success": True
                            })
                            
                        elif code == "C":  # 推荐课程 -> Plans and Pricing
                            logger.info("调用课程计划和定价工具")
                            result = plans_pricing_tool.func(query=content, lang=lang)
                            tool_results.append({
                                "tool": "plans_pricing",
                                "intention": intention,
                                "result": result,
                                "success": True
                            })
                            
                        elif code == "D":  # 你是谁 -> Self introduction
                            logger.info("调用自我介绍工具")
                            result = self_introduction_tool.func(query=content, lang=lang)
                            tool_results.append({
                                "tool": "self_introduction",
                                "intention": intention,
                                "result": result,
                                "success": True
                            })
                            
                        elif code == "E":  # 没用的问题 -> qwen plus一轮游
                            logger.info("调用通用回复工具")
                            result = general_response_tool.func(query=content, lang=lang)
                            tool_results.append({
                                "tool": "general_response",
                                "intention": intention,
                                "result": result,
                                "success": True
                            })
                            
                        else:
                            # 未知意图代码，使用通用回复工具
                            logger.warning(f"未知的意图代码: {code}，使用通用回复工具")
                            result = general_response_tool.func(query=content, lang=lang)
                            tool_results.append({
                                "tool": "general_response_fallback",
                                "intention": intention,
                                "result": result,
                                "success": True
                            })
                            
                    except Exception as tool_error:
                        logger.error(f"执行意图{i+1}的工具时失败: {tool_error}")
                        # 对于A类意图（examist），不进行容错处理，直接抛出异常
                        if code == "A":
                            raise tool_error
                        else:
                            # 对于其他工具，记录错误但继续执行
                            tool_results.append({
                                "tool": f"error_{code}",
                                "intention": intention,
                                "result": f"工具执行失败: {str(tool_error)}",
                                "success": False,
                                "error": str(tool_error)
                            })
                
                logger.info(f"多工具执行完成，原始意图{len(intentions)}个，去重后{len(unique_intentions)}个，共执行了{len(tool_results)}个工具")
                
                # 将所有工具结果保存到context中
                context["tool_results"] = tool_results
                
                return {
                    "messages": state["messages"],
                    "context": context
                }
                    
            except Exception as e:
                logger.error(f"多工具执行器失败: {e}")
                # 如果是A类意图的错误，直接传播
                if any(intention.get("Code") == "A" for intention in context.get("intention_result", {}).get("Intentions", [])):
                    raise e
                
                # 否则创建错误结果
                context["tool_results"] = [{
                    "tool": "multi_tool_executor_error",
                    "intention": {"Code": "ERROR", "Content": "多工具执行器失败"},
                    "result": f"多工具执行器执行失败: {str(e)}",
                    "success": False,
                    "error": str(e)
                }]
                
                return {
                    "messages": state["messages"],
                    "context": context
                }
        
        def final_summary_node(state):
            """最终汇总节点 - 生成最终回答"""
            try:
                logger.info("执行最终汇总")
                
                context = state["context"]
                original_query = context["original_query"]
                messages = state["messages"]
                skip_tools = context.get("skip_tools", False)
                
                if skip_tools:
                    # 如果跳过了工具执行，让大模型直接基于问题回答
                    logger.info("意图识别失败或为空，让大模型直接回答用户问题")
                    
                    # 构建直接回答的prompt
                    from langchain_core.messages import SystemMessage, HumanMessage
                    
                    # 检测语言
                    is_chinese = any(char in original_query for char in "的了吗是中国香港台湾")
                    
                    if is_chinese:
                        direct_prompt = f"""你是HKCA Learning Media的专业AI助手，专门提供香港税务和会计服务咨询。

用户问题：{original_query}

请直接基于你的知识为用户提供专业、准确、有帮助的回答。如果涉及具体的税务问题，请提供相关的香港税法条例和实务指导。如果需要更详细的专业建议，请建议用户联系我们的专业顾问团队。

请用中文回答："""
                    else:
                        direct_prompt = f"""You are a professional AI assistant at HKCA Learning Media, specializing in Hong Kong taxation and accounting services.

User Question: {original_query}

Please provide a professional, accurate, and helpful response based on your knowledge. If this involves specific tax matters, please provide relevant Hong Kong tax ordinances and practical guidance. If detailed professional advice is needed, please suggest contacting our professional advisory team.

Please respond in English:"""
                    
                    # 使用配置的LLM进行直接回答
                    try:
                        from config.llm_config import llm_config
                        available_llms = llm_config.get_available_llms()
                        if available_llms:
                            llm = available_llms[0]["llm"]
                        else:
                            # 备用LLM
                            from langchain_openai import ChatOpenAI
                            import os
                            llm = ChatOpenAI(
                                model="gpt-4o-mini",
                                api_key=os.getenv("OPENAI_API_KEY"),
                                temperature=0.2
                            )
                        
                        response = llm.invoke([HumanMessage(content=direct_prompt)])
                        final_response = response.content
                        
                    except Exception as llm_error:
                        logger.error(f"直接回答生成失败: {llm_error}")
                        if is_chinese:
                            final_response = f"抱歉，我在处理您的问题时遇到了技术问题。您的问题是：{original_query}\n\n请稍后重试或联系我们的客服团队获得帮助。"
                        else:
                            final_response = f"Sorry, I encountered a technical issue while processing your question: {original_query}\n\nPlease try again later or contact our customer service team for assistance."
                    
                else:
                    # 正常情况：有意图识别结果和工具结果
                    intention_result = context["intention_result"]
                    tool_results = context["tool_results"]
                    
                    # 调用最终汇总工具，现在需要传递多个工具结果
                    final_response = final_summary_tool.func(
                        messages=messages,
                        intention_result=intention_result,
                        tool_result=tool_results,  # 现在传递的是工具结果列表
                        original_query=original_query
                    )
                
                logger.info("最终汇总完成")
                
                # 将最终回答添加到消息历史中
                final_messages = list(messages)
                final_messages.append(AIMessage(content=final_response))
                
                return {
                    "messages": final_messages,
                    "context": context
                }
                    
            except Exception as e:
                logger.error(f"最终汇总失败: {e}")
                
                # 根据情况生成fallback回答
                context = state["context"]
                original_query = context.get("original_query", "")
                skip_tools = context.get("skip_tools", False)
                
                if skip_tools:
                    # 如果是跳过工具的情况
                    is_chinese = any(char in original_query for char in "的了吗是中国香港台湾")
                    if is_chinese:
                        fallback_response = f"抱歉，在处理您的问题时遇到了技术问题。您的问题：{original_query}\n\n建议您重新提问或联系我们的专业顾问团队。"
                    else:
                        fallback_response = f"Sorry, there was a technical issue processing your question: {original_query}\n\nPlease try asking again or contact our professional advisory team."
                else:
                    # 如果是工具执行后的汇总失败
                    tool_results = context.get("tool_results", [])
                    
                    if tool_results:
                        # 汇总所有成功的工具结果
                        successful_results = [tr for tr in tool_results if tr.get("success", False)]
                        if successful_results:
                            combined_result = "\n\n".join([
                                f"**{tr['tool']}工具结果:**\n{tr['result']}" 
                                for tr in successful_results
                            ])
                            fallback_response = f"根据您的查询，我为您提供以下信息：\n\n{combined_result}"
                        else:
                            # 如果所有工具都失败了
                            error_info = "\n".join([
                                f"- {tr['tool']}: {tr.get('error', '未知错误')}"
                                for tr in tool_results
                            ])
                            fallback_response = f"抱歉，在处理您的查询时遇到了一些问题：\n{error_info}"
                    else:
                        fallback_response = f"抱歉，在生成最终回答时遇到了问题：{str(e)}"
                
                final_messages = list(state["messages"])
                final_messages.append(AIMessage(content=fallback_response))
                
                return {
                    "messages": final_messages,
                    "context": context
                }
        
        # 构建新的工作流图
        from src.agent import AgentState  # 避免循环导入
        builder = StateGraph(AgentState)
        
        # 添加节点
        builder.add_node("intention_recognition", intention_recognition_node)
        builder.add_node("multi_tool_executor", multi_tool_executor_node)
        builder.add_node("final_summary", final_summary_node)
        
        # 构建工作流路径
        builder.add_edge(START, "intention_recognition")
        
        # 添加条件边：根据意图识别结果决定是否跳过工具执行
        builder.add_conditional_edges(
            "intention_recognition",
            intention_router,
            {
                "multi_tool_executor": "multi_tool_executor",  # 正常执行工具
                "final_summary": "final_summary"              # 跳过工具执行
            }
        )
        
        builder.add_edge("multi_tool_executor", "final_summary")
        builder.add_edge("final_summary", END)
        
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
        final_result = None
        final_ai_response = None
        
        try:
            for step in workflow.stream(initial_state, config=config, stream_mode="updates"):
                # 检查 step 是否为有效字典
                if not isinstance(step, dict):
                    logger.warning(f"工作流步骤不是字典格式: {type(step)}")
                    continue
                
                # 从不同节点收集响应，只保留最新的AI消息
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
                        
                        # 检查是否有 messages，只关注final_summary节点的最终结果
                        if "messages" in node_state and node_name == "final_summary":
                            messages = node_state["messages"]
                            if messages and isinstance(messages, list):  # 确保 messages 不为空且是列表
                                # 找到最后一个AI消息作为最终结果
                                for msg in reversed(messages):  # 从后往前找
                                    if isinstance(msg, AIMessage) and msg.content:
                                        final_result = msg.content
                                        final_ai_response = msg
                                        logger.info(f"获取到最终AI回答: {msg.content[:100]}...")
                                        break
                                        
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
        
        # 返回结果，现在只返回最终的单个回答
        if final_result and final_ai_response:
            logger.info("工作流执行完成，返回最终结果")
            return [final_result], [final_ai_response]  # 保持原有的返回格式，但只包含最终结果
        else:
            logger.warning("工作流执行完成，但未找到最终AI回答")
            return [], []

# 全局实例
workflow_manager = WorkflowManager() 