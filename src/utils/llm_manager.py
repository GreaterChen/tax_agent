"""
LLM管理器
实现API key轮询、限流控制和统一调用接口
"""
import os
import yaml
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import time
from datetime import datetime

from src.utils.token_counter import token_counter
from src.utils.rate_limiter import RateLimiter, ApiKeyRateLimiter

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    """供应商类型"""
    QWEN = "qwen"
    OPENAI = "openai"

@dataclass
class ApiKeyConfig:
    """API Key配置"""
    key: str
    priority: int
    qpm_limit: int
    tpm_limit: int
    enabled: bool
    current_qpm: int = 0
    current_tpm: int = 0
    last_used: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    max_tokens: int
    temperature: float
    enabled: bool

@dataclass
class ProviderConfig:
    """供应商配置"""
    base_url: str
    api_keys: List[ApiKeyConfig]
    models: List[ModelConfig]

@dataclass
class LLMRequest:
    """LLM请求"""
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False

@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    provider: str
    api_key_id: str
    usage: Dict[str, int]
    response_time: float

class LLMManager:
    """LLM管理器"""
    
    def __init__(self, config_path: str = "config/llm_config.yaml"):
        self.config_path = config_path
        self.config = None
        self.providers = {}
        self.rate_limiter = None
        self.api_key_limiter = None
        self._last_config_reload = 0
        self._config_reload_interval = 300  # 5分钟重载一次配置
        
        # 初始化
        self._load_config()
        self._init_rate_limiter()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # 解析环境变量
            self._resolve_env_variables()
            
            # 构建provider配置
            self._build_provider_configs()
            
            self._last_config_reload = time.time()
            logger.info("LLM配置加载成功")
            
        except Exception as e:
            logger.error(f"加载LLM配置失败: {e}")
            raise Exception(f"LLM配置加载失败: {e}")
    
    def _resolve_env_variables(self):
        """解析环境变量"""
        def resolve_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, "")
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value
        
        self.config = resolve_value(self.config)
    
    def _build_provider_configs(self):
        """构建provider配置"""
        self.providers = {}
        
        for provider_name, provider_data in self.config["llm_config"]["providers"].items():
            # 构建API key配置
            api_keys = []
            for key_data in provider_data["api_keys"]:
                if key_data["key"]:  # 只添加非empty的API key
                    api_keys.append(ApiKeyConfig(
                        key=key_data["key"],
                        priority=key_data["priority"],
                        qpm_limit=key_data["qpm_limit"],
                        tpm_limit=key_data["tpm_limit"],
                        enabled=key_data["enabled"]
                    ))
            
            # 构建模型配置
            models = []
            for model_data in provider_data["models"]:
                if model_data["enabled"]:
                    models.append(ModelConfig(
                        name=model_data["name"],
                        max_tokens=model_data["max_tokens"],
                        temperature=model_data["temperature"],
                        enabled=model_data["enabled"]
                    ))
            
            if api_keys and models:  # 只添加有有效API key和模型的provider
                self.providers[provider_name] = ProviderConfig(
                    base_url=provider_data["base_url"],
                    api_keys=sorted(api_keys, key=lambda x: x.priority),  # 按优先级排序
                    models=models
                )
    
    def _init_rate_limiter(self):
        """初始化限流器"""
        try:
            rate_limit_config = self.config["llm_config"]["rate_limit"]
            self.rate_limiter = RateLimiter(
                key_prefix=rate_limit_config["key_prefix"],
                window_size=rate_limit_config["window_size"],
                precision=rate_limit_config["precision"]
            )
            self.api_key_limiter = ApiKeyRateLimiter(self.rate_limiter)
            logger.info("限流器初始化成功")
        except Exception as e:
            logger.error(f"限流器初始化失败: {e}")
            raise Exception(f"限流器初始化失败: {e}")
    
    async def _check_config_reload(self):
        """检查是否需要重载配置"""
        if time.time() - self._last_config_reload > self._config_reload_interval:
            try:
                self._load_config()
                logger.info("配置重载成功")
            except Exception as e:
                logger.warning(f"配置重载失败: {e}")
    
    async def _select_api_key(self, provider_name: str, request_tokens: int) -> Optional[ApiKeyConfig]:
        """选择可用的API key"""
        provider = self.providers.get(provider_name)
        if not provider:
            return None
        
        # 按优先级顺序尝试API key
        for api_key in provider.api_keys:
            if not api_key.enabled:
                continue
            
            # 检查限流
            allowed, status = await self.api_key_limiter.check_api_key_limits(
                api_key.key,
                request_tokens,
                api_key.qpm_limit,
                api_key.tpm_limit
            )
            
            if allowed:
                # 更新使用统计
                api_key.current_qpm = status.get("current_qpm", 0)
                api_key.current_tpm = status.get("current_tpm", 0)
                api_key.last_used = datetime.now()
                return api_key
            else:
                logger.debug(f"API key {api_key.key[-8:]} 超出限制: {status.get('reason', '')}")
        
        return None
    
    async def _call_llm_api(self, provider_name: str, api_key: str, base_url: str,
                           request: LLMRequest) -> Dict[str, Any]:
        """调用LLM API"""
        headers = {
            "Content-Type": "application/json",
        }
        
        # 根据provider设置认证header
        if provider_name == "qwen":
            headers["Authorization"] = f"Bearer {api_key}"
        elif provider_name == "openai":
            headers["Authorization"] = f"Bearer {api_key}"
        
        # 构建请求体 - 安全地提取值
        def safe_extract(value, convert_func=None):
            """安全地提取值，处理FieldInfo等特殊对象"""
            if hasattr(value, 'default'):
                # 这是一个Field对象，获取默认值
                value = value.default
            if value is None:
                return None
            if convert_func:
                return convert_func(value)
            return value
        
        data = {
            "model": safe_extract(request.model, str),
            "messages": safe_extract(request.messages),
            "temperature": safe_extract(request.temperature, float),
            "max_tokens": safe_extract(request.max_tokens, int),
            "stream": safe_extract(request.stream, bool)
        }
        
        # 移除None值
        data = {k: v for k, v in data.items() if v is not None}
        
        # API端点
        url = f"{base_url.rstrip('/')}/chat/completions"
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                timeout = aiohttp.ClientTimeout(total=self.config["llm_config"]["global"]["request_timeout"])
                async with session.post(url, json=data, headers=headers, timeout=timeout) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "data": result,
                            "response_time": response_time
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}",
                            "response_time": response_time
                        }
                        
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": "请求超时",
                    "response_time": time.time() - start_time
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time
                }
    
    async def chat_completion(self, messages: List[Dict[str, Any]], 
                            preferred_model: Optional[str] = None,
                            temperature: Optional[float] = None,
                            max_tokens: Optional[int] = None,
                            max_retries: Optional[int] = None) -> LLMResponse:
        """
        聊天完成API
        
        Args:
            messages: 消息列表
            preferred_model: 首选模型
            temperature: 温度参数
            max_tokens: 最大token数
            max_retries: 最大重试次数
            
        Returns:
            LLM响应
        """
        await self._check_config_reload()
        
        # 获取全局配置
        global_config = self.config["llm_config"]["global"]
        max_retries = max_retries or global_config["max_retries"]
        
        # 计算请求token数
        model_for_counting = preferred_model or "gpt-4o-mini"
        request_tokens = token_counter.count_messages_tokens(messages, model_for_counting)
        
        # 检查token限制
        max_token_window = global_config["max_token_window"]
        if request_tokens > max_token_window:
            # 尝试截断消息
            truncated_messages = token_counter.truncate_messages(
                messages, model_for_counting, max_token_window
            )
            if token_counter.count_messages_tokens(truncated_messages, model_for_counting) > max_token_window:
                raise Exception(f"请求token数 ({request_tokens}) 超出限制 ({max_token_window})")
            messages = truncated_messages
            request_tokens = token_counter.count_messages_tokens(messages, model_for_counting)
        
        # 确定模型优先级顺序
        model_priorities = self.config["llm_config"]["model_priority"]
        if preferred_model:
            # 如果指定了首选模型，将其放在最前面
            model_priorities = [p for p in model_priorities if p["model"] == preferred_model] + \
                             [p for p in model_priorities if p["model"] != preferred_model]
        
        last_error = None
        
        # 按模型优先级尝试
        for model_priority in model_priorities:
            provider_name = model_priority["provider"]
            model_name = model_priority["model"]
            
            if provider_name not in self.providers:
                continue
            
            provider = self.providers[provider_name]
            
            # 检查模型是否可用
            model_config = next((m for m in provider.models if m.name == model_name and m.enabled), None)
            if not model_config:
                continue
            
            # 选择API key
            api_key = await self._select_api_key(provider_name, request_tokens)
            if not api_key:
                last_error = f"Provider {provider_name} 无可用API key"
                logger.warning(last_error)
                continue
            
            # 构建请求
            request = LLMRequest(
                messages=messages,
                model=model_name,
                temperature=temperature or model_config.temperature,
                max_tokens=max_tokens or model_config.max_tokens
            )
            
            # 重试逻辑
            for retry in range(max_retries + 1):
                try:
                    result = await self._call_llm_api(
                        provider_name, api_key.key, provider.base_url, request
                    )
                    
                    if result["success"]:
                        # 解析响应
                        data = result["data"]
                        content = data["choices"][0]["message"]["content"]
                        usage = data.get("usage", {})
                        
                        # 重置错误计数
                        api_key.error_count = 0
                        api_key.last_error = None
                        
                        return LLMResponse(
                            content=content,
                            model=model_name,
                            provider=provider_name,
                            api_key_id=api_key.key[-8:],
                            usage=usage,
                            response_time=result["response_time"]
                        )
                    else:
                        # API调用失败
                        error_msg = result["error"]
                        api_key.error_count += 1
                        api_key.last_error = error_msg
                        
                        # 如果是严重错误（如API key无效），标记为禁用
                        if "unauthorized" in error_msg.lower() or "invalid" in error_msg.lower():
                            api_key.enabled = False
                            logger.error(f"API key {api_key.key[-8:]} 被禁用: {error_msg}")
                            break
                        
                        if retry < max_retries:
                            retry_delay = global_config["retry_delay"] * (2 ** retry)  # 指数退避
                            logger.warning(f"重试 {retry + 1}/{max_retries}: {error_msg}, 等待 {retry_delay}s")
                            await asyncio.sleep(retry_delay)
                        else:
                            last_error = error_msg
                            
                except Exception as e:
                    error_msg = str(e)
                    api_key.error_count += 1
                    api_key.last_error = error_msg
                    
                    if retry < max_retries:
                        retry_delay = global_config["retry_delay"] * (2 ** retry)
                        logger.warning(f"重试 {retry + 1}/{max_retries}: {error_msg}, 等待 {retry_delay}s")
                        await asyncio.sleep(retry_delay)
                    else:
                        last_error = error_msg
        
        # 所有模型都失败了
        error_msg = f"所有LLM服务不可用，最后错误: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    async def get_status(self) -> Dict[str, Any]:
        """获取LLM管理器状态"""
        status = {
            "providers": {},
            "global_config": self.config["llm_config"]["global"],
            "last_config_reload": datetime.fromtimestamp(self._last_config_reload).isoformat()
        }
        
        for provider_name, provider in self.providers.items():
            api_keys_status = []
            
            for api_key in provider.api_keys:
                usage = await self.api_key_limiter.get_api_key_usage(api_key.key)
                api_keys_status.append({
                    "id": api_key.key[-8:],
                    "priority": api_key.priority,
                    "enabled": api_key.enabled,
                    "qpm_limit": api_key.qpm_limit,
                    "tpm_limit": api_key.tpm_limit,
                    "current_qpm": usage.get("requests", 0),
                    "current_tpm": usage.get("tokens", 0),
                    "error_count": api_key.error_count,
                    "last_error": api_key.last_error,
                    "last_used": api_key.last_used.isoformat() if api_key.last_used else None
                })
            
            status["providers"][provider_name] = {
                "base_url": provider.base_url,
                "models": [{"name": m.name, "enabled": m.enabled} for m in provider.models],
                "api_keys": api_keys_status
            }
        
        return status


# 全局实例
llm_manager = LLMManager()