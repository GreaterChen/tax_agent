"""
通用回复工具
用于处理E类意图（没用的问题），使用qwen plus一轮游
"""
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
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

class GeneralResponseInput(BaseModel):
    """通用回复输入模型"""
    query: str = Field(..., description="用户的一般询问")
    lang: str = Field(default="en", description="用户期望的回复语言 (zh-cn, zh-hk, en)")

class GeneralResponseTool:
    """通用回复工具"""
    
    def __init__(self):
        # 使用qwen plus进行通用回复
        try:
            from config.llm_config import llm_config
            
            # 使用LLMConfig对象的正确方法获取配置
            available_llms = llm_config.get_available_llms()
            
            # 查找qwen plus或qwen相关的配置
            qwen_llm = None
            for llm_info in available_llms:
                model_name = llm_info.get("name", "").lower()
                if "qwen" in model_name:  # 查找任何包含qwen的模型
                    qwen_llm = llm_info.get("llm")
                    logger.info(f"通用回复工具初始化完成，使用模型: {llm_info.get('name')}")
                    break
            
            if qwen_llm:
                self.llm = qwen_llm
            else:
                # 如果没有找到qwen相关模型，使用第一个可用的模型
                if available_llms:
                    first_llm = available_llms[0]
                    self.llm = first_llm.get("llm")
                    logger.info(f"未找到qwen模型，使用第一个可用模型: {first_llm.get('name')}")
                else:
                    # 如果都没有，使用默认的ChatOpenAI
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(
                        model="gpt-4o-mini", 
                        api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                        temperature=0.1  # 保持简洁一致的回复
                    )
                    logger.warning("未找到任何配置的模型，使用默认模型")
                
        except Exception as e:
            logger.error(f"通用回复工具初始化失败: {e}")
            # 回退到默认配置
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="gpt-4o-mini", 
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                temperature=0.1
            )
    
    def generate_response(self, query: str, lang: str = "en") -> str:
        """
        生成通用回复
        
        Args:
            query: 用户的一般询问
            lang: 用户期望的回复语言 (zh-cn, zh-hk, en)
            
        Returns:
            str: 通用回复内容
        """
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # 构建消息，考虑语言偏好
            language_instruction = ""
            if lang in ["zh-cn", "Sim"]:
                language_instruction = "\n请用简体中文回复。"
            elif lang in ["zh-hk", "Trad"]:
                language_instruction = "\n請用繁體中文回覆。"
            else:
                language_instruction = "\nPlease reply in English."
            
            messages = [
                SystemMessage(content=GENERAL_RESPONSE_PROMPT + language_instruction),
                HumanMessage(content=query)
            ]
            
            # 调用LLM生成回复
            response = self.llm.invoke(messages)
            
            logger.info("通用回复生成成功")
            return response.content
                
        except Exception as e:
            logger.error(f"通用回复生成失败: {e}")
            # 根据语言返回默认回复
            if lang in ["zh-cn", "Sim"]:
                return "抱歉，我不太了解这个问题。如果您有香港税务相关的问题，我很乐意为您解答。"
            elif lang in ["zh-hk", "Trad"]:
                return "抱歉，我不太了解這個問題。如果您有香港稅務相關的問題，我很樂意為您解答。"
            else:
                return "I'm sorry, but I don't know about that. If you have any Hong Kong taxation related questions, I'd be happy to help."

# 创建工具实例
general_response_tool_instance = GeneralResponseTool()

# 封装为StructuredTool
general_response_tool = StructuredTool.from_function(
    func=general_response_tool_instance.generate_response,
    name="general_response",
    description="处理一般性询问，提供简洁专业的回复。主要用于处理与税务无关的问题。",
    args_schema=GeneralResponseInput
) 