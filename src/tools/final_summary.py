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
        try:
            from config.llm_config import llm_config
            
            # 使用LLMConfig对象获取最优的模型
            available_llms = llm_config.get_available_llms()
            
            # 优先使用第一个可用的模型
            if available_llms:
                first_llm = available_llms[0]
                self.llm = first_llm.get("llm")
                logger.info(f"最终汇总工具初始化完成，使用模型: {first_llm.get('name')}")
            else:
                # 如果没有配置的模型，使用默认的ChatOpenAI
                from langchain_openai import ChatOpenAI
                import os
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini", 
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    temperature=0.2  # 较低的temperature确保回答准确性
                )
                logger.warning("未找到任何配置的模型，使用默认模型")
                
        except Exception as e:
            logger.error(f"最终汇总工具初始化失败: {e}")
            # 回退到默认配置
            from langchain_openai import ChatOpenAI
            import os
            self.llm = ChatOpenAI(
                model="gpt-4o-mini", 
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                temperature=0.2
            )
    
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
    
    def generate_final_summary(self, messages: List[BaseMessage], intention_result: Dict, 
                             tool_result: Union[str, List[Dict]], original_query: str) -> str:
        """
        生成最终汇总回答
        
        Args:
            messages: 完整的对话消息历史
            intention_result: 意图识别的结果
            tool_result: 工具执行的结果，可以是单个字符串或多个工具结果的列表
            original_query: 用户的原始查询
            
        Returns:
            str: 最终的汇总回答
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
            
            # 调用LLM生成最终回答
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages_for_llm = [
                SystemMessage(content=final_prompt),
                HumanMessage(content="请基于以上信息生成最终回答")
            ]
            
            response = self.llm.invoke(messages_for_llm)
            
            logger.info("最终汇总回答生成成功")
            return response.content
                
        except Exception as e:
            logger.error(f"最终汇总回答生成失败: {e}")
            # 返回一个备用回答
            formatted_fallback_results = self.format_tool_results(tool_result)
            
            if "简体中文" in self.determine_response_language(intention_result, original_query):
                return f"抱歉，在生成最终回答时遇到了问题。不过根据您的查询，这里是工具执行的结果：\n\n{formatted_fallback_results}\n\n如需进一步帮助，请重新提问或联系我们的客服团队。"
            else:
                return f"Sorry, there was an issue generating the final response. However, based on your query, here's the tool execution result:\n\n{formatted_fallback_results}\n\nFor further assistance, please ask again or contact our customer service team."

# 创建工具实例
final_summary_tool_instance = FinalSummaryTool()

# 封装为StructuredTool
final_summary_tool = StructuredTool.from_function(
    func=final_summary_tool_instance.generate_final_summary,
    name="final_summary",
    description="汇总所有上下文信息和工具执行结果，生成用户的最终回答。",
    args_schema=FinalSummaryInput
) 