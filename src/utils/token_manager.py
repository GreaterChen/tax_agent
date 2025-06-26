"""
统一Token管理器
整合所有token计算、提取、成本计算和统计逻辑，消除冗余
"""
import logging
import tiktoken
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 支持ChatTongyi的导入
try:
    from langchain_community.chat_models import ChatTongyi
    CHATGONGYI_AVAILABLE = True
except ImportError:
    CHATGONGYI_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Token使用量数据类"""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    source: str = "unknown"  # api_response, manual_calculation, estimation
    provider: str = "unknown"
    model_used: str = ""
    
    def __post_init__(self):
        """自动计算总token数"""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens

@dataclass 
class CostInfo:
    """成本信息数据类"""
    input_cost: float = 0.0
    output_cost: float = 0.0
    cached_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    llm_name: str = ""
    token_usage: TokenUsage = None
    
    def __post_init__(self):
        """自动计算总成本"""
        if self.total_cost == 0:
            self.total_cost = self.input_cost + self.output_cost + self.cached_cost

class TokenManager:
    """统一Token管理器"""
    
    # 模型编码器和限制配置
    MODEL_CONFIG = {
        "gpt-4o-mini": {"encoding": "cl100k_base", "limit": 128000},
        "gpt-4o": {"encoding": "cl100k_base", "limit": 128000},
        "gpt-4": {"encoding": "cl100k_base", "limit": 8000},
        "gpt-3.5-turbo": {"encoding": "cl100k_base", "limit": 4000},
        "qwen-max-latest": {"encoding": "cl100k_base", "limit": 32000},
        "qwen-max": {"encoding": "cl100k_base", "limit": 32000},
        "qwen-plus": {"encoding": "cl100k_base", "limit": 32000},
        "qwen-plus-latest": {"encoding": "cl100k_base", "limit": 32000},
        "qwen-turbo": {"encoding": "cl100k_base", "limit": 8000},
        "qwen-long": {"encoding": "cl100k_base", "limit": 10000000},
    }
    
    def __init__(self):
        self._encoders = {}
        self._load_encoders()
    
    def _load_encoders(self):
        """预加载编码器"""
        try:
            unique_encodings = set(config["encoding"] for config in self.MODEL_CONFIG.values())
            for encoding_name in unique_encodings:
                self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
            logger.info("Token编码器加载成功")
        except Exception as e:
            logger.error(f"Token编码器加载失败: {e}")
            self._encoders["cl100k_base"] = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        """
        统一的token计算方法
        
        Args:
            text: 输入文本
            model: 模型名称
            
        Returns:
            token数量
        """
        if not text:
            return 0
            
        try:
            config = self.MODEL_CONFIG.get(model, self.MODEL_CONFIG["gpt-4o-mini"])
            encoder = self._encoders.get(config["encoding"])
            
            if encoder:
                return len(encoder.encode(text))
            else:
                return self._estimate_tokens(text)
                
        except Exception as e:
            logger.warning(f"Token计算失败，使用估算: {e}")
            return self._estimate_tokens(text)
    
    def count_messages_tokens(self, messages: Union[List[Dict], List[BaseMessage]], model: str = "gpt-4o-mini") -> int:
        """
        计算消息列表的token数
        
        Args:
            messages: 消息列表（支持字典格式或LangChain格式）
            model: 模型名称
            
        Returns:
            总token数
        """
        if not messages:
            return 0
            
        try:
            # 统一转换为字典格式
            normalized_messages = self._normalize_messages(messages)
            
            total_tokens = 3  # 对话格式的基础开销
            
            for message in normalized_messages:
                total_tokens += 3  # 每条消息的固定开销
                
                # 计算content的token
                content = message.get("content", "")
                if content:
                    total_tokens += self.count_tokens(str(content), model)
                
                # 计算role的token  
                role = message.get("role", "")
                if role:
                    total_tokens += self.count_tokens(role, model)
                
                # name字段的开销
                if "name" in message:
                    total_tokens += 1 + self.count_tokens(message["name"], model)
            
            return total_tokens
            
        except Exception as e:
            logger.error(f"消息token计算失败: {e}")
            # 降级处理
            total_text = ""
            for msg in messages:
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                total_text += str(content) + " "
            return self._estimate_tokens(total_text)
    
    def extract_api_token_usage(self, response: AIMessage, llm_instance: Any) -> Optional[TokenUsage]:
        """
        从API响应中提取token使用量
        
        Args:
            response: LLM响应对象
            llm_instance: LLM实例
            
        Returns:
            TokenUsage对象，如果提取失败返回None
        """
        try:
            # 检查response_metadata
            if not hasattr(response, 'response_metadata') or not response.response_metadata:
                return None
            
            metadata = response.response_metadata
            
            # 根据LLM类型选择提取策略
            if isinstance(llm_instance, ChatOpenAI):
                return self._extract_openai_tokens(metadata, response, llm_instance)
            elif CHATGONGYI_AVAILABLE and isinstance(llm_instance, ChatTongyi):
                return self._extract_tongyi_tokens(metadata)
            else:
                return self._extract_generic_tokens(metadata)
                
        except Exception as e:
            logger.error(f"API token提取失败: {e}")
            return None
    
    def calculate_token_usage(self, request_text: str, response_text: str, 
                            llm_instance: Any, api_response: Optional[AIMessage] = None) -> TokenUsage:
        """
        计算完整的token使用量（优先使用API数据）
        
        Args:
            request_text: 请求文本
            response_text: 响应文本
            llm_instance: LLM实例
            api_response: API响应对象（可选）
            
        Returns:
            TokenUsage对象
        """
        # 1. 优先尝试从API响应提取
        if api_response:
            api_usage = self.extract_api_token_usage(api_response, llm_instance)
            if api_usage:
                logger.info(f"使用API token数据: {api_usage}")
                return api_usage
        
        # 2. Fallback到手动计算
        model_name = self._get_model_name(llm_instance)
        input_tokens = self.count_tokens(request_text, model_name)
        output_tokens = self.count_tokens(response_text, model_name)
        
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            source="manual_calculation",
            provider=self._get_provider_name(llm_instance),
            model_used=model_name
        )
        
        logger.info(f"使用手动计算token数据: {usage}")
        return usage
    
    def calculate_cost(self, token_usage: TokenUsage, llm_config: Dict) -> CostInfo:
        """
        根据token使用量计算成本
        
        Args:
            token_usage: Token使用量
            llm_config: LLM配置信息
            
        Returns:
            CostInfo对象
        """
        input_cost = (token_usage.input_tokens / 1000) * llm_config.get("input_price", 0)
        output_cost = (token_usage.output_tokens / 1000) * llm_config.get("output_price", 0)
        cached_cost = (token_usage.cached_tokens / 1000) * llm_config.get("cached_price", 0)
        
        return CostInfo(
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            cached_cost=round(cached_cost, 6),
            currency=llm_config.get("currency", "USD"),
            llm_name=llm_config.get("name", "unknown"),
            token_usage=token_usage
        )
    
    def estimate_response_tokens(self, request_text: str, model: str = "gpt-4o-mini", 
                               multiplier: float = 3.0) -> int:
        """预估响应token数"""
        request_tokens = self.count_tokens(request_text, model)
        return int(request_tokens * multiplier)
    
    def validate_token_limit(self, text_or_messages: Union[str, List], model: str) -> Tuple[bool, int]:
        """
        验证token是否超限
        
        Returns:
            (是否符合限制, 实际token数)
        """
        if isinstance(text_or_messages, str):
            tokens = self.count_tokens(text_or_messages, model)
        else:
            tokens = self.count_messages_tokens(text_or_messages, model)
        
        limit = self.MODEL_CONFIG.get(model, self.MODEL_CONFIG["gpt-4o-mini"])["limit"]
        return tokens <= limit, tokens
    
    # ====== 私有方法 ======
    
    def _normalize_messages(self, messages: Union[List[Dict], List[BaseMessage]]) -> List[Dict]:
        """统一消息格式"""
        normalized = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized.append(msg)
            elif isinstance(msg, BaseMessage):
                role = "user"
                if isinstance(msg, AIMessage):
                    role = "assistant"
                elif isinstance(msg, SystemMessage):
                    role = "system"
                elif isinstance(msg, HumanMessage):
                    role = "user"
                
                normalized.append({
                    "role": role,
                    "content": msg.content
                })
        return normalized
    
    def _extract_openai_tokens(self, metadata: Dict, response: AIMessage, llm_instance: ChatOpenAI) -> Optional[TokenUsage]:
        """提取OpenAI格式的token信息"""
        # 检查usage_metadata
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            
            # 验证token数据有效性
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            cached_tokens = usage.get('cached_tokens', 0)
            
            if input_tokens > 0 or output_tokens > 0:
                return TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    source="usage_metadata",
                    provider=self._get_provider_name(llm_instance),
                    model_used=self._get_model_name(llm_instance)
                )
        
        return None
    
    def _extract_tongyi_tokens(self, metadata: Dict) -> Optional[TokenUsage]:
        """提取通义千问的token信息"""
        if 'token_usage' not in metadata:
            return None
        
        token_usage = metadata['token_usage']
        
        # 验证必需字段
        required_fields = ['input_tokens', 'output_tokens', 'total_tokens']
        if not all(field in token_usage for field in required_fields):
            return None
        
        return TokenUsage(
            input_tokens=token_usage['input_tokens'],
            output_tokens=token_usage['output_tokens'],
            total_tokens=token_usage['total_tokens'],
            cached_tokens=0,  # 通义千问暂不支持缓存
            source="api_response",
            provider="qwen",
            model_used="qwen"
        )
    
    def _extract_generic_tokens(self, metadata: Dict) -> Optional[TokenUsage]:
        """通用token提取"""
        # 尝试常见字段
        token_fields = {
            'input_tokens': ['input_tokens', 'prompt_tokens', 'input_token_count'],
            'output_tokens': ['output_tokens', 'completion_tokens', 'output_token_count'],
            'total_tokens': ['total_tokens', 'total_token_count'],
            'cached_tokens': ['cached_tokens', 'cache_hit_tokens']
        }
        
        result = {}
        for target_field, candidates in token_fields.items():
            for candidate in candidates:
                if candidate in metadata:
                    result[target_field] = metadata[candidate]
                    break
                elif 'usage' in metadata and candidate in metadata['usage']:
                    result[target_field] = metadata['usage'][candidate]
                    break
                elif 'token_usage' in metadata and candidate in metadata['token_usage']:
                    result[target_field] = metadata['token_usage'][candidate]
                    break
        
        if result.get('input_tokens', 0) > 0 or result.get('output_tokens', 0) > 0:
            return TokenUsage(
                input_tokens=result.get('input_tokens', 0),
                output_tokens=result.get('output_tokens', 0),
                cached_tokens=result.get('cached_tokens', 0),
                total_tokens=result.get('total_tokens', 0),
                source="generic_extraction",
                provider="generic"
            )
        
        return None
    
    def _get_model_name(self, llm_instance: Any) -> str:
        """获取模型名称"""
        if hasattr(llm_instance, 'model_name'):
            return llm_instance.model_name
        elif hasattr(llm_instance, 'model'):
            return llm_instance.model
        elif isinstance(llm_instance, ChatOpenAI):
            # 根据base_url判断
            base_url = getattr(llm_instance, 'openai_api_base', '') or getattr(llm_instance, 'base_url', '')
            if 'dashscope.aliyuncs.com' in str(base_url):
                return "qwen-max-latest"
            else:
                return "gpt-4o-mini"
        else:
            return "gpt-4o-mini"
    
    def _get_provider_name(self, llm_instance: Any) -> str:
        """获取provider名称"""
        if isinstance(llm_instance, ChatOpenAI):
            base_url = getattr(llm_instance, 'openai_api_base', '') or getattr(llm_instance, 'base_url', '')
            if 'dashscope.aliyuncs.com' in str(base_url):
                return "qwen_compatible"
            else:
                return "openai"
        elif CHATGONGYI_AVAILABLE and isinstance(llm_instance, ChatTongyi):
            return "qwen"
        else:
            return "unknown"
    
    def _estimate_tokens(self, text: str) -> int:
        """简单token估算"""
        if not text:
            return 0
        
        # 中文1字符=1token，英文4字符=1token
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        other_chars = len(text) - chinese_chars
        return chinese_chars + max(other_chars // 4, 1)

# 全局实例
token_manager = TokenManager() 