"""
自我介绍工具
用于处理用户对系统身份的询问
"""
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
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

class SelfIntroductionInput(BaseModel):
    """自我介绍输入模型"""
    query: str = Field(..., description="用户关于身份的询问")
    lang: str = Field(default="en", description="用户期望的回复语言 (zh-cn, zh-hk, en)")

class SelfIntroductionTool:
    """自我介绍工具"""
    
    def __init__(self):
        # 使用配置中的LLM进行自我介绍
        self.llm = ChatOpenAI(
            model="qwen-max-latest", 
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0  # 设置为0以获得更稳定的结果
        )
        logger.info("自我介绍工具初始化完成")
    
    def introduce_self(self, query: str, lang: str = "en") -> str:
        """
        生成自我介绍
        
        Args:
            query: 用户的身份询问
            lang: 用户期望的回复语言 (zh-cn, zh-hk, en)
            
        Returns:
            str: 自我介绍内容
        """
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
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
            
            # 构建消息
            messages = [
                SystemMessage(content=SELF_INTRODUCTION_PROMPT),
                HumanMessage(content=f"用户问题: {query}\n请根据用户期望的语言({lang_description})进行回复。")
            ]
            
            # 调用LLM生成自我介绍
            response = self.llm.invoke(messages)
            
            logger.info("自我介绍生成成功")
            return response.content
                
        except Exception as e:
            logger.error(f"自我介绍生成失败: {e}")
            # 根据语言返回默认介绍
            if lang in ["zh-cn", "Sim"]:
                return "我是HKCA Learning Media Limited开发的香港税务专家AI系统，是智能香港税务AI项目的一部分。我可以帮助您解答香港税务相关问题。"
            elif lang in ["zh-hk", "Trad"]:
                return "我是HKCA Learning Media Limited開發的香港稅務專家AI系統，是智能香港稅務AI項目的一部分。我可以幫助您解答香港稅務相關問題。"
            else:
                return "I am a Hong Kong Tax expert AI system developed by HKCA Learning Media Limited, part of the Smart Hong Kong Tax AI Project. I can help you with Hong Kong taxation related questions."

# 创建工具实例
self_introduction_tool_instance = SelfIntroductionTool()

# 封装为StructuredTool
self_introduction_tool = StructuredTool.from_function(
    func=self_introduction_tool_instance.introduce_self,
    name="self_introduction",
    description="处理用户对系统身份的询问，提供专业的自我介绍。",
    args_schema=SelfIntroductionInput
) 