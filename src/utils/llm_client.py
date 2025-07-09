"""
异步LLM服务调用客户端
用于调用远程LLM服务，替代直接调用LangChain LLM，并提供统一的token和成本追踪
"""
import json
import logging
import time
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

@dataclass
class LLMUsageInfo:
    """LLM调用使用信息"""
    request_id: str
    model_used: str
    provider: str
    total_cost: float
    currency: str
    token_usage: Dict[str, Any]
    cost_breakdown: Dict[str, Any]
    processing_time: float

class AsyncLLMClient:
    """异步LLM服务调用客户端"""
    
    def __init__(self, base_url: str = "http://8.216.81.217:8002"):
        """
        初始化LLM客户端
        
        Args:
            base_url: LLM服务的基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.query_endpoint = f"{self.base_url}/query"
        self.timeout = aiohttp.ClientTimeout(total=120)  # 2分钟超时
        logger.info(f"LLM客户端初始化完成，服务地址: {self.query_endpoint}")
    
    def _normalize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """
        将不同格式的消息转换为标准OpenAI格式
        
        Args:
            messages: 消息列表，可以是字典或LangChain消息对象
            
        Returns:
            标准化的消息列表（OpenAI格式）
        """
        normalized = []
        
        for msg in messages:
            if isinstance(msg, dict):
                # 已经是字典格式，直接使用
                if "role" in msg and "content" in msg:
                    normalized.append(msg)
                else:
                    logger.warning(f"字典消息格式不正确，跳过: {msg}")
            elif isinstance(msg, BaseMessage):
                # LangChain消息对象 -> OpenAI格式转换
                if isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                else:
                    role = "user"  # 默认为用户消息
                
                normalized.append({
                    "role": role,
                    "content": msg.content
                })
            else:
                # 其他格式，尝试转换为用户消息
                content = str(msg) if msg else ""
                if content:  # 只有非空内容才添加
                    normalized.append({
                        "role": "user",
                        "content": content
                    })
        
        return normalized
    
    async def _make_request(self, messages: List[Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        发送HTTP请求到LLM服务
        
        Args:
            messages: 消息列表
            model_name: 指定的模型名称，可选
            
        Returns:
            API响应数据
        """
        # 标准化消息格式
        normalized_messages = self._normalize_messages(messages)
        
        # 构建请求数据
        request_data = {
            "messages": normalized_messages
        }
        
        if model_name:
            request_data["model_name"] = model_name
        
        # 发送请求
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                start_time = time.time()
                async with session.post(
                    self.query_endpoint,
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    processing_time = time.time() - start_time
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"LLM服务请求失败，状态码: {response.status}, 响应: {error_text}")
                    
                    response_data = await response.json()
                    response_data['_processing_time'] = processing_time
                    
                    logger.info(f"LLM服务调用成功，耗时: {processing_time:.2f}秒")
                    return response_data
                    
            except asyncio.TimeoutError:
                raise Exception("LLM服务请求超时")
            except aiohttp.ClientError as e:
                raise Exception(f"LLM服务网络错误: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"LLM服务响应格式错误: {str(e)}")
    
    async def chat_completion(self, messages: List[Any], model_name: Optional[str] = None) -> Tuple[str, LLMUsageInfo]:
        """
        执行聊天补全请求
        
        Args:
            messages: 消息列表
            model_name: 指定的模型名称，可选
            
        Returns:
            Tuple[回答内容, 使用信息]
        """
        try:
            # 发送请求
            response_data = await self._make_request(messages, model_name)
            
            # 检查响应格式
            if response_data.get("code") != 200:
                raise Exception(f"LLM服务业务错误: {response_data.get('msg', '未知错误')}")
            
            data = response_data.get("data", {})
            if not data:
                raise Exception("LLM服务返回空数据")
            
            # 提取回答内容
            answers = data.get("answers", [])
            if not answers:
                raise Exception("LLM服务未返回回答内容")
            
            answer = answers[0] if isinstance(answers, list) else str(answers)
            
            # 构建使用信息
            usage_info = LLMUsageInfo(
                request_id=data.get("request_id", ""),
                model_used=data.get("model_used", "unknown"),
                provider=data.get("provider", "unknown"),
                total_cost=data.get("total_cost", 0.0),
                currency=data.get("currency", "CNY"),
                token_usage=data.get("token_usage", {}),
                cost_breakdown=data.get("cost_breakdown", {}),
                processing_time=response_data.get('_processing_time', 0.0)
            )
            
            logger.info(f"LLM调用完成 - 模型: {usage_info.model_used}, "
                       f"Token: {usage_info.token_usage.get('total_tokens', 0)}, "
                       f"成本: {usage_info.total_cost}{usage_info.currency}")
            
            return answer, usage_info
            
        except Exception as e:
            logger.error(f"LLM服务调用失败: {e}")
            raise e
    
    async def simple_chat(self, user_message: str, system_message: Optional[str] = None, 
                         model_name: Optional[str] = None) -> Tuple[str, LLMUsageInfo]:
        """
        简单的聊天接口，用于单轮对话
        
        Args:
            user_message: 用户消息
            system_message: 系统消息，可选
            model_name: 指定的模型名称，可选
            
        Returns:
            Tuple[回答内容, 使用信息]
        """
        messages = []
        
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })
        
        messages.append({
            "role": "user", 
            "content": user_message
        })
        
        return await self.chat_completion(messages, model_name)

# 创建全局客户端实例
llm_client = AsyncLLMClient() 