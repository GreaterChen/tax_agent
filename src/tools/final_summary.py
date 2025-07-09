"""
最终汇总工具
汇总所有用户和LLM交互的上下文以及调用的工具结果，生成最终回答
"""
import logging
import json
from typing import Dict, Any, List, Union
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# 最终汇总的System Prompt
FINAL_SUMMARY_PROMPT = """You are a professional AI assistant at HKCA Learning Media, specializing in Hong Kong taxation and accounting services.

Your task is to provide a comprehensive, helpful, and professional final response to the user based on:
1. The conversation history and context
2. The user's original intention(s) and query
3. The results from one or more specialized tools that were executed

## Context Information:
**User's Intention(s):** {intention_info}
**Original Query:** {original_query}
**Conversation History:** {conversation_history}
**Tool Execution Results:** {tool_results}

## Instructions:
1. Synthesize all the information above to provide a complete and accurate answer
2. Maintain a professional, helpful, and friendly tone
3. Ensure the response directly addresses the user's original query
4. If multiple tools were executed for different aspects of the query, integrate all relevant information coherently
5. If the tools provided specific information (like search results, course recommendations, tax analysis, etc.), incorporate it naturally
6. If there were any limitations or issues with tool execution, handle them gracefully
7. Respond in the language that best matches the user's preference: {response_language}
8. Structure your response clearly with appropriate formatting when needed
9. Always prioritize accuracy and relevance over length
10. When multiple intentions were detected and addressed, organize the response to cover each aspect clearly

Please provide your final response:"""

class FinalSummaryInput(BaseModel):
    """最终汇总输入模型"""
    messages: List[BaseMessage] = Field(..., description="完整的对话消息历史")
    intention_result: Dict = Field(..., description="意图识别的结果")
    tool_result: Union[str, List[Dict]] = Field(..., description="工具执行的结果，可以是单个结果字符串或多个工具结果的列表")
    original_query: str = Field(..., description="用户的原始查询")

class FinalSummaryTool:
    """最终汇总工具"""
    
    def __init__(self):
        """初始化最终汇总工具"""
        # 使用异步LLM客户端进行最终汇总
        from src.utils.llm_client import llm_client
        self.llm_client = llm_client
        logger.info("最终汇总工具初始化完成")
    
    def format_conversation_history(self, messages: List[BaseMessage]) -> str:
        """格式化对话历史"""
        if not messages:
            return "无对话历史"
        
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                # 过滤掉意图识别的中间结果，只保留真正的回复
                if not msg.content.startswith("意图识别完成:"):
                    formatted_messages.append(f"Assistant: {msg.content}")
        
        return "\n".join(formatted_messages[-10:])  # 只保留最近10条消息
    
    def format_intention_info(self, intention_result: Dict) -> str:
        """格式化意图识别信息"""
        try:
            intentions = intention_result.get("Intentions", [])
            if intentions:
                code_mapping = {
                    "A": "税务问题求解",
                    "B": "一般事实查询", 
                    "C": "课程推荐",
                    "D": "身份询问",
                    "E": "无关问题"
                }
                
                if len(intentions) == 1:
                    # 单个意图
                    intention = intentions[0]
                    code = intention.get("Code", "Unknown")
                    content = intention.get("Content", "")
                    code_desc = code_mapping.get(code, f"未知类型({code})")
                    return f"意图类型: {code_desc}, 具体内容: {content}"
                else:
                    # 多个意图
                    intention_details = []
                    for i, intention in enumerate(intentions):
                        code = intention.get("Code", "Unknown")
                        content = intention.get("Content", "")
                        code_desc = code_mapping.get(code, f"未知类型({code})")
                        intention_details.append(f"意图{i+1}: {code_desc} - {content}")
                    
                    return f"多重意图识别 (共{len(intentions)}个):\n" + "\n".join(intention_details)
            else:
                return "无法解析用户意图"
        except Exception as e:
            logger.error(f"格式化意图信息失败: {e}")
            return "意图信息解析错误"
    
    def determine_response_language(self, intention_result: Dict, original_query: str) -> str:
        """确定回复语言"""
        try:
            # 首先从意图识别结果中获取语言偏好
            lang = intention_result.get("Lang", "en")
            
            if lang in ["zh-cn", "Sim"]:
                return "Simplified Chinese (简体中文)"
            elif lang in ["zh-hk", "Trad"]:
                return "Traditional Chinese (繁體中文)"
            else:
                return "English"
                
        except Exception:
            # 如果无法确定，通过查询内容简单判断
            if any(char in original_query for char in "的了吗是"):
                return "Simplified Chinese (简体中文)"
            else:
                return "English"
    
    def format_tool_results(self, tool_result: Union[str, List[Dict]]) -> str:
        """格式化工具执行结果"""
        try:
            if isinstance(tool_result, str):
                # 兼容旧格式，单个工具结果
                return tool_result
            elif isinstance(tool_result, list):
                # 新格式，多个工具结果
                if not tool_result:
                    return "无工具执行结果"
                
                formatted_results = []
                for i, result in enumerate(tool_result):
                    tool_name = result.get("tool", f"工具{i+1}")
                    success = result.get("success", False)
                    result_content = result.get("result", "")
                    intention = result.get("intention", {})
                    
                    # 工具名称映射
                    tool_name_mapping = {
                        "examist": "税务专家分析",
                        "web_search": "网络搜索",
                        "plans_pricing": "课程推荐",
                        "self_introduction": "自我介绍",
                        "general_response": "通用回复"
                    }
                    
                    display_name = tool_name_mapping.get(tool_name, tool_name)
                    
                    if success:
                        formatted_results.append(f"**{display_name}结果:**\n{result_content}")
                    else:
                        error_msg = result.get("error", "未知错误")
                        formatted_results.append(f"**{display_name}执行失败:** {error_msg}")
                
                return "\n\n".join(formatted_results)
            else:
                return str(tool_result)
                
        except Exception as e:
            logger.error(f"格式化工具结果失败: {e}")
            return f"工具结果格式化错误: {str(e)}"
    
    async def generate_final_summary(self, messages: List[BaseMessage], intention_result: Dict, 
                             tool_result: Union[str, List[Dict]], original_query: str) -> Dict[str, Any]:
        """
        生成最终汇总回答（异步版本，支持token和成本统计）
        
        Args:
            messages: 完整的对话消息历史
            intention_result: 意图识别的结果
            tool_result: 工具执行的结果，可以是单个字符串或多个工具结果的列表
            original_query: 用户的原始查询
            
        Returns:
            Dict: 包含汇总回答和使用统计的字典
        """
        try:
            # 格式化各种信息
            conversation_history = self.format_conversation_history(messages)
            intention_info = self.format_intention_info(intention_result)
            response_language = self.determine_response_language(intention_result, original_query)
            formatted_tool_results = self.format_tool_results(tool_result)
            
            # 构建最终的prompt
            final_prompt = FINAL_SUMMARY_PROMPT.format(
                intention_info=intention_info,
                original_query=original_query,
                conversation_history=conversation_history,
                tool_results=formatted_tool_results,
                response_language=response_language
            )
            
            # 调用异步LLM生成最终回答，使用第一个可用的模型
            response_content, usage_info = await self.llm_client.simple_chat(
                user_message="请基于以上信息生成最终回答",
                system_message=final_prompt,
                model_name=None  # 让服务自动选择模型
            )
            
            logger.info("最终汇总回答生成成功")
            
            # 构建包含使用统计的完整结果
            result = {
                "response": response_content,
                "usage_info": {
                    "request_id": usage_info.request_id,
                    "model_used": usage_info.model_used,
                    "provider": usage_info.provider,
                    "total_cost": usage_info.total_cost,
                    "currency": usage_info.currency,
                    "token_usage": usage_info.token_usage,
                    "cost_breakdown": usage_info.cost_breakdown,
                    "processing_time": usage_info.processing_time
                }
            }
            
            logger.info(f"最终汇总完成 - 模型: {usage_info.model_used}, "
                       f"Token: {usage_info.token_usage.get('total_tokens', 0)}, "
                       f"成本: {usage_info.total_cost}{usage_info.currency}")
            
            return result
                
        except Exception as e:
            logger.error(f"最终汇总回答生成失败: {e}")
            # 返回一个备用回答
            formatted_fallback_results = self.format_tool_results(tool_result)
            
            fallback_response = ""
            if "简体中文" in self.determine_response_language(intention_result, original_query):
                fallback_response = f"抱歉，在生成最终回答时遇到了问题。不过根据您的查询，这里是工具执行的结果：\n\n{formatted_fallback_results}\n\n如需进一步帮助，请重新提问或联系我们的客服团队。"
            else:
                fallback_response = f"Sorry, there was an issue generating the final response. However, based on your query, here's the tool execution result:\n\n{formatted_fallback_results}\n\nFor further assistance, please ask again or contact our customer service team."
            
            return {
                "response": fallback_response,
                "usage_info": {
                    "request_id": "error",
                    "model_used": "fallback",
                    "provider": "local",
                    "total_cost": 0.0,
                    "currency": "CNY",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "cost_breakdown": {"input_cost": 0, "output_cost": 0, "total_cost": 0},
                    "processing_time": 0.0
                }
            }
    
    def generate_final_summary_sync(self, messages: List[BaseMessage], intention_result: Dict, 
                                   tool_result: Union[str, List[Dict]], original_query: str) -> str:
        """
        生成最终汇总回答（同步版本，向后兼容），使用线程池执行异步版本
        
        Args:
            messages: 完整的对话消息历史
            intention_result: 意图识别的结果
            tool_result: 工具执行的结果，可以是单个字符串或多个工具结果的列表
            original_query: 用户的原始查询
            
        Returns:
            str: 最终汇总回答内容
        """
        try:
            # 使用线程池执行异步版本
            import asyncio
            
            # 获取当前事件循环或创建新的
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 在线程池中执行异步方法
            async def _async_generate():
                result = await self.generate_final_summary(messages, intention_result, tool_result, original_query)
                return result.get("response", "")
            
            response_content = loop.run_until_complete(_async_generate())
            logger.info("同步最终汇总回答生成成功")
            
            return response_content
                
        except Exception as e:
            logger.error(f"同步最终汇总回答生成失败: {e}")
            return self._get_fallback_summary(intention_result, tool_result, original_query)
    
    def _get_fallback_summary(self, intention_result: Dict, tool_result: Union[str, List[Dict]], original_query: str) -> str:
        """获取默认最终汇总"""
        formatted_fallback_results = self.format_tool_results(tool_result)
        response_language = self.determine_response_language(intention_result, original_query)
        
        if "简体中文" in response_language:
            return f"抱歉，在生成最终回答时遇到了问题。不过根据您的查询，这里是工具执行的结果：\n\n{formatted_fallback_results}\n\n如需进一步帮助，请重新提问或联系我们的客服团队。"
        elif "繁體中文" in response_language:
            return f"抱歉，在生成最終回答時遇到了問題。不過根據您的查詢，這裡是工具執行的結果：\n\n{formatted_fallback_results}\n\n如需進一步幫助，請重新提問或聯繫我們的客服團隊。"
        else:
            return f"Sorry, there was an issue generating the final response. However, based on your query, here's the tool execution result:\n\n{formatted_fallback_results}\n\nFor further assistance, please ask again or contact our customer service team."

# 创建工具实例
final_summary_tool_instance = FinalSummaryTool()

# 封装为StructuredTool（同步版本，向后兼容）
final_summary_tool = StructuredTool.from_function(
    func=final_summary_tool_instance.generate_final_summary_sync,
    name="final_summary",
    description="汇总所有上下文信息和工具执行结果，生成用户的最终回答。",
    args_schema=FinalSummaryInput
) 