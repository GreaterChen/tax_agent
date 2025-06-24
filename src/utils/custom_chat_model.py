"""
自定义LangChain ChatModel
整合LLM轮询管理功能，保持LangChain原生接口体验
"""
import asyncio
import logging
from typing import Any, Dict, Iterator, List, Optional, Union, AsyncGenerator
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.runnables import Runnable

from src.utils.llm_manager import llm_manager

logger = logging.getLogger(__name__)

class ManagedChatModel(BaseChatModel):
    """
    自定义ChatModel，使用LLM管理器实现轮询和限流
    完全兼容LangChain的ChatModel接口
    """
    
    # 模型参数
    model_name: str = Field(default="qwen-max-latest", alias="model")
    temperature: float = Field(default=0.1)
    max_tokens: Optional[int] = Field(default=None)
    streaming: bool = Field(default=False)
    
    # 自定义参数
    preferred_provider: Optional[str] = Field(default=None)
    enable_fallback: bool = Field(default=True)
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识"""
        return "managed_chat_model"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回标识参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "preferred_provider": self.preferred_provider,
        }
    
    def _convert_messages_to_dict(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """将LangChain消息转换为标准格式"""
        converted_messages = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                role = "user"  # 默认为user
            
            converted_messages.append({
                "role": role,
                "content": message.content
            })
        
        return converted_messages
    
    def bind_tools(
        self,
        tools: List[BaseTool],
        **kwargs: Any,
    ) -> Runnable:
        """绑定工具到模型"""
        # 创建一个新的实例，包含工具信息
        return self.bind(tools=tools, **kwargs)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步生成方法 - 内部调用异步方法"""
        # 在同步环境中运行异步代码
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # 如果已经在事件循环中，使用run_until_complete可能会死锁
            # 创建新的事件循环
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self._agenerate(messages, stop, None, **kwargs))
                )
                return future.result()
        else:
            return loop.run_until_complete(
                self._agenerate(messages, stop, None, **kwargs)
            )
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成方法 - 核心实现"""
        try:
            # 转换消息格式
            converted_messages = self._convert_messages_to_dict(messages)
            
            # 合并参数，过滤掉LLM管理器不支持的参数
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['tools', 'tool_choice', 'functions', 'function_call']}
            
            call_kwargs = {
                "messages": converted_messages,
                "preferred_model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **filtered_kwargs
            }
            
            # 调用LLM管理器
            if run_manager:
                await run_manager.on_llm_start(
                    serialized={"name": self._llm_type},
                    prompts=[str(messages)],
                )
            
            # 使用LLM管理器进行调用
            llm_response = await llm_manager.chat_completion(**call_kwargs)
            
            # 构建LangChain格式的响应
            ai_message = AIMessage(content=llm_response.content)
            generation = ChatGeneration(
                message=ai_message,
                generation_info={
                    "model": llm_response.model,
                    "provider": llm_response.provider,
                    "api_key_id": llm_response.api_key_id,
                    "usage": llm_response.usage,
                    "response_time": llm_response.response_time,
                }
            )
            
            result = ChatResult(generations=[generation])
            
            if run_manager:
                await run_manager.on_llm_end(LLMResult(generations=[[generation]]))
            
            return result
            
        except Exception as e:
            logger.error(f"ManagedChatModel调用失败: {e}")
            if run_manager:
                await run_manager.on_llm_error(e)
            raise e
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """流式生成 - 暂时不支持，降级到普通生成"""
        logger.warning("流式生成暂未实现，降级到普通生成")
        result = self._generate(messages, stop, run_manager, **kwargs)
        yield result.generations[0]
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatGeneration, None]:
        """异步流式生成 - 暂时不支持，降级到普通生成"""
        logger.warning("异步流式生成暂未实现，降级到普通生成")
        result = await self._agenerate(messages, stop, run_manager, **kwargs)
        yield result.generations[0]


class QwenManagedChatModel(ManagedChatModel):
    """通义千问专用ChatModel"""
    
    def __init__(self, **kwargs):
        super().__init__(
            model_name=kwargs.get("model", "qwen-max-latest"),
            preferred_provider="qwen",
            **kwargs
        )


class OpenAIManagedChatModel(ManagedChatModel):
    """OpenAI专用ChatModel"""
    
    def __init__(self, **kwargs):
        super().__init__(
            model_name=kwargs.get("model", "gpt-4o-mini"),
            preferred_provider="openai",
            **kwargs
        )


# 便捷的工厂函数
def create_managed_chat_model(
    provider: str = "auto",
    model: str = None,
    **kwargs
) -> ManagedChatModel:
    """
    创建管理的ChatModel
    
    Args:
        provider: 提供商 ("qwen", "openai", "auto")
        model: 模型名称
        **kwargs: 其他参数
    
    Returns:
        ManagedChatModel实例
    """
    if provider == "qwen":
        return QwenManagedChatModel(model=model or "qwen-max-latest", **kwargs)
    elif provider == "openai":
        return OpenAIManagedChatModel(model=model or "gpt-4o-mini", **kwargs)
    else:  # auto
        return ManagedChatModel(
            model_name=model or "qwen-max-latest",
            **kwargs
        )


# 兼容性别名 - 可以直接替换原有的ChatModel
ChatTongyiManaged = QwenManagedChatModel
ChatOpenAIManaged = OpenAIManagedChatModel