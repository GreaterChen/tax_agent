"""
通用回复工具
用于处理E类意图（没用的问题），使用远程LLM服务
"""
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

# 通用回复的System Prompt - 严格按照用户要求，不做任何改动
GENERAL_RESPONSE_PROMPT = """** System Prompt **
    Role:
    You are a helpful professional HK taxation receptionist.
    Task: 
    Answer briefly with professional tongue and friendly attitude.
    If you don't know and the question is unrelated to taxation, just say you don't know.
    If you know, you can answer, but briefly."""

class GeneralResponseTool:
    """通用回复工具"""
    
    def __init__(self):
        # 使用异步LLM客户端进行通用回复
        from src.utils.llm_client import llm_client
        self.llm_client = llm_client
        logger.info("通用回复工具初始化完成")
    
    async def generate_response(self, query: str, lang: str = "en") -> Dict[str, Any]:
        """
        生成通用回复（异步版本，支持token和成本统计）
        
        Args:
            query: 用户的一般询问
            lang: 用户期望的回复语言 (zh-cn, zh-hk, en)
            
        Returns:
            Dict: 包含回复内容和使用统计的字典
        """
        try:
            # 构建消息，考虑语言偏好
            language_instruction = ""
            if lang in ["zh-cn", "Sim"]:
                language_instruction = "\n请用简体中文回复。"
            elif lang in ["zh-hk", "Trad"]:
                language_instruction = "\n請用繁體中文回覆。"
            else:
                language_instruction = "\nPlease reply in English."
            
            system_prompt = GENERAL_RESPONSE_PROMPT + language_instruction
            
            # 调用异步LLM生成回复，使用qwen-plus模型
            response_content, usage_info = await self.llm_client.simple_chat(
                user_message=query,
                system_message=system_prompt,
                model_name="qwen-plus"
            )
            
            logger.info("通用回复生成成功")
            
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
            
            logger.info(f"通用回复完成 - 模型: {usage_info.model_used}, "
                       f"Token: {usage_info.token_usage.get('total_tokens', 0)}, "
                       f"成本: {usage_info.total_cost}{usage_info.currency}")
            
            return result
                
        except Exception as e:
            logger.error(f"通用回复生成失败: {e}")
            # 根据语言返回默认回复
            default_response = ""
            if lang in ["zh-cn", "Sim"]:
                default_response = "抱歉，我不太了解这个问题。如果您有香港税务相关的问题，我很乐意为您解答。"
            elif lang in ["zh-hk", "Trad"]:
                default_response = "抱歉，我不太了解這個問題。如果您有香港稅務相關的問題，我很樂意為您解答。"
            else:
                default_response = "I'm sorry, but I don't know about that. If you have any Hong Kong taxation related questions, I'd be happy to help."
            
            return {
                "response": default_response,
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

# 创建工具实例
general_response_tool_instance = GeneralResponseTool() 