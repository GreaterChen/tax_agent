"""
意图识别工具
分析用户查询的意图，将其分类到不同的处理流程
"""
import json
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# 意图识别的System Prompt
INTENTION_RECOGNITION_PROMPT = """System Prompt**
Role:
You are a professional Exceptionist at HKCA Learning Media, a taxation consultancy and educational institution.
Your Responsibilities:
Your primary task is to identify the user's intent and direct their request to the appropriate associate.
Input:
The user's original message.
Tasks:
Categorize the user's requirements and match them to the corresponding intention codes.
Determine the appropriate associate(s) to handle the request.
Forward the relevant details to the designated associate(s), ensuring all information is accurately conveyed without omissions or additions.
If the input is in a language other than English, translate it into English.
Reference: Intention Code Table
Code A: The user has a specific Hong Kong or Chinese taxation problem involving facts and a question. For example, inquiries about tax treatment or procedures under certain circumstances.
    e.g. Explain the xxx principal or regime. / The fact is xxx, Identify the tax liability. / The fact is xxx, Calculate the Stamp Duty, Profit Tax, Salary Tax, Property Tax...
Code B: The user is asking about general facts related to Hong Kong taxation, including exam enrollment (HKICPA, ACCA), or information about Hong Kong government entities, especially the IRD, or news updates.
    e.g. Who is the head of IRD? / What is the exam fee for HKICPA, ACCA? / Where can I download Inland Revenue Ordinance? / What is the First Schedule of Stamp Duty Ordinance(SDO) ?
Code C: The user is seeking recommendations for classes, consulting plans, or pricing information.
    e.g. Which course is suitbale for me? /  Where can I find HK tax consultant?
Code D: The user is inquiring about your identity.
    e.g. Who are you? / Are you Chatgpt? / Are you deepseek?
Code E: The user requests other types of information or fails to provide a coherent request.
    e.g. How's the weather in HK?
Output:
Output in JSON format:
{
    "Mode":"Hybrid/Single" #If multiple intentions are found -> Hybrid. Otherwise -> Single.
    "Lang":"zh-cn/zh-hk/en" #The user's expected response language, which is based on user's original message: zh-cn for Simplified Chinese, zh-hk for Traditional Chinese, en for English.
    "Intentions":[
        {"Code":"A/B/C/D/E",
        "Content":"The full input that is needed to satisfy the intention."
        }
    ]
}
Examples for Inention field.
{
    "Code":"A" , # A Specific China & HK Taxation Question.
    "Content":"The FULL content of user's input, including fact and quesiton. DO not omit any information or change any information. IF multiple separate questions are asked, only pass along the first FACT-QUESTION. OUTPUT IN ENGLISH."
}

{
    "Code":"B" , # General Inquiry
    "Content":"Rephrase the user's question clearly."
}

{
    "Code":"E" , # General Inquiry
    "Content":"Keep the user's original input. use **** to cover inappropriate content."
}

Format:
Strict JSON format.

Rules:
1. Must have at least 1 intention. multiple intentions are common and acceptible.
2. Do not omit any information. Pass along exactly what you received. 
3. Always translate to english. 
4. You must pay attention to chat history, the full task might be located partly in the chat history!"""

class IntentionRecognitionTool:
    """意图识别工具"""
    
    def __init__(self):
        # 使用异步LLM客户端进行意图识别
        from src.utils.llm_client import llm_client
        self.llm_client = llm_client
        logger.info("意图识别工具初始化完成")
    
    def _extract_json_from_markdown(self, content: str) -> str:
        """
        从markdown格式的响应中提取JSON内容
        
        Args:
            content: 原始响应内容，可能包含markdown代码块
            
        Returns:
            str: 清理后的JSON字符串
        """
        # 移除markdown代码块标记
        # 匹配 ```json...``` 或 ```...``` 格式
        json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_pattern, content, re.DOTALL)
        
        if match:
            # 提取代码块内的内容
            extracted = match.group(1).strip()
            logger.debug(f"从markdown中提取JSON: {extracted}")
            return extracted
        
        # 如果没有找到markdown代码块，尝试直接寻找JSON对象
        json_object_pattern = r'\{.*\}'
        match = re.search(json_object_pattern, content, re.DOTALL)
        
        if match:
            extracted = match.group(0).strip()
            logger.debug(f"直接提取JSON对象: {extracted}")
            return extracted
        
        # 如果都没找到，返回原始内容（让后续的json.loads抛出异常）
        logger.warning(f"无法从响应中提取JSON，返回原始内容: {content}")
        return content.strip()
    
    async def recognize_intention(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        识别用户意图（异步版本，支持token和成本统计）
        
        Args:
            messages: 对话消息历史
            
        Returns:
            Dict: 包含意图识别结果和使用统计的字典
        """
        try:
            # 构建用于意图识别的完整消息
            formatted_messages = [{"role": "system", "content": INTENTION_RECOGNITION_PROMPT}]
            
            # 添加对话历史
            for msg in messages:
                if msg.get("role") in ["user", "assistant", "system"]:
                    formatted_messages.append({
                        "role": msg.get("role"),
                        "content": msg.get("content", "")
                    })
            
            # 调用异步LLM进行意图识别，使用qwen-max-latest模型
            response_content, usage_info = await self.llm_client.chat_completion(
                messages=formatted_messages,
                model_name="qwen-max-latest"
            )
            
            # 清理响应内容，提取JSON
            cleaned_content = self._extract_json_from_markdown(response_content)
            
            # 解析JSON响应 - 如果失败直接抛出异常
            intention_result = json.loads(cleaned_content)
            logger.info(f"意图识别成功: {intention_result}")
            
            # 构建包含使用统计的完整结果
            result = {
                "intention_result": intention_result,
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
            
            logger.info(f"意图识别完成 - 模型: {usage_info.model_used}, "
                       f"Token: {usage_info.token_usage.get('total_tokens', 0)}, "
                       f"成本: {usage_info.total_cost}{usage_info.currency}")
            
            return result
                
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            # 直接抛出异常
            raise e

# 创建工具实例
intention_recognition_tool_instance = IntentionRecognitionTool() 