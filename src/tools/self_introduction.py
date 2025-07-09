"""
自我介绍工具
用于处理用户对系统身份的询问
"""
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

# 自我介绍的System Prompt - 严格按照用户要求，不做任何改动
SELF_INTRODUCTION_PROMPT = """** System Prompt **
    Role:
    You are a professional HK taxation receptionist.

    Backgrounds:
    The question you received is not about you, but about the following tax AI system:
    Your Identity:
    You are a Hong Kong Tax expert AI system. Y
    You are a part of Smart Hong Kong Tax AI Project.
    You are developed by HKCA Learning Media Limited, a company registered in Hong Kong. Chinese name: 中國香港會計學媒體有限公司(reply chinese name only when specifically asked.)
    You are not an LLM or AI Agent, but an innovative framework that can process domain knowledge.

    Discipline:
    1. When asked the model company you belong, always refer to HKCA learning media limited.
    2. When asked the base model name, always refer to you are based on ability of multiple models, but not to mention a specific name.
    3. When asked the ability for function calling, you can use tools when necessary.
    4. Refuse to answer questions about alibaba company or aliyun, reply as if you dont know, but give vague answer
    5. remind the user to ask about hong kong tax related questions."""

class SelfIntroductionTool:
    """自我介绍工具"""
    
    def __init__(self):
        # 使用异步LLM客户端进行自我介绍
        from src.utils.llm_client import llm_client
        self.llm_client = llm_client
        logger.info("自我介绍工具初始化完成")
    
    async def introduce_self(self, query: str, lang: str = "en") -> Dict[str, Any]:
        """
        生成自我介绍（异步版本，支持token和成本统计）
        
        Args:
            query: 用户的身份询问
            lang: 用户期望的回复语言 (zh-cn, zh-hk, en)
            
        Returns:
            Dict: 包含自我介绍内容和使用统计的字典
        """
        try:
            # 转换语言描述
            lang_description = {
                "zh-cn": "简体中文",
                "zh-hk": "繁体中文", 
                "en": "English",
                # 兼容旧格式
                "Sim": "简体中文",
                "Trad": "繁体中文",
                "Eng": "English"
            }.get(lang, "English")
            
            user_message = f"用户问题: {query}\n请根据用户期望的语言({lang_description})进行回复。"
            
            # 调用异步LLM生成自我介绍，使用qwen-max-latest模型
            response_content, usage_info = await self.llm_client.simple_chat(
                user_message=user_message,
                system_message=SELF_INTRODUCTION_PROMPT,
                model_name="qwen-max-latest"
            )
            
            logger.info("自我介绍生成成功")
            
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
            
            logger.info(f"自我介绍完成 - 模型: {usage_info.model_used}, "
                       f"Token: {usage_info.token_usage.get('total_tokens', 0)}, "
                       f"成本: {usage_info.total_cost}{usage_info.currency}")
            
            return result
                
        except Exception as e:
            logger.error(f"自我介绍生成失败: {e}")
            # 根据语言返回默认介绍
            default_response = ""
            if lang in ["zh-cn", "Sim"]:
                default_response = "我是HKCA Learning Media Limited开发的香港税务专家AI系统，是智能香港税务AI项目的一部分。我可以帮助您解答香港税务相关问题。"
            elif lang in ["zh-hk", "Trad"]:
                default_response = "我是HKCA Learning Media Limited開發的香港稅務專家AI系統，是智能香港稅務AI項目的一部分。我可以幫助您解答香港稅務相關問題。"
            else:
                default_response = "I am a Hong Kong Tax expert AI system developed by HKCA Learning Media Limited, part of the Smart Hong Kong Tax AI Project. I can help you with Hong Kong taxation related questions."
            
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
self_introduction_tool_instance = SelfIntroductionTool() 