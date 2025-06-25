"""
Token计算工具
支持不同模型的token统计
"""
import tiktoken
from typing import List, Dict, Any, Optional
import logging
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import time

logger = logging.getLogger(__name__)

class TokenCounter:
    """Token计算器"""
    
    # 不同模型的编码器映射
    MODEL_ENCODINGS = {
        "gpt-4o-mini": "cl100k_base",
        "gpt-4o": "cl100k_base", 
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "qwen-max-latest": "cl100k_base",  # 通义千问使用相同编码
        "qwen-max": "cl100k_base",
        "qwen-plus": "cl100k_base",
    }
    
    # 模型的token限制
    MODEL_LIMITS = {
        "gpt-4o-mini": 128000,
        "gpt-4o": 128000,
        "qwen-max-latest": 32000,
        "qwen-max": 30000,
        "qwen-plus": 32000,
    }
    
    def __init__(self):
        self._encoders = {}
        self._load_encoders()
    
    def _load_encoders(self):
        """预加载编码器"""
        try:
            for encoding_name in set(self.MODEL_ENCODINGS.values()):
                self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
            logger.info("Token编码器加载成功")
        except Exception as e:
            logger.error(f"Token编码器加载失败: {e}")
            # 降级到默认编码器
            self._encoders["cl100k_base"] = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        """
        计算文本的token数量
        
        Args:
            text: 输入文本
            model: 模型名称
            
        Returns:
            token数量
        """
        try:
            encoding_name = self.MODEL_ENCODINGS.get(model, "cl100k_base")
            encoder = self._encoders.get(encoding_name)
            
            if not encoder:
                # 如果编码器不存在，使用默认估算
                return self._estimate_tokens(text)
                
            return len(encoder.encode(text))
            
        except Exception as e:
            logger.warning(f"Token计算失败，使用估算方法: {e}")
            return self._estimate_tokens(text)
    
    def count_messages_tokens(self, messages: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> int:
        """
        计算消息列表的总token数量
        
        Args:
            messages: 消息列表
            model: 模型名称
            
        Returns:
            总token数量
        """
        total_tokens = 0
        
        try:
            # 消息格式的固定开销
            tokens_per_message = 3  # 每条消息的固定token开销
            tokens_per_name = 1     # 如果消息有name字段的开销
            
            for message in messages:
                total_tokens += tokens_per_message
                
                # 计算content的token
                content = message.get("content", "")
                if content:
                    total_tokens += self.count_tokens(str(content), model)
                
                # 计算role的token
                role = message.get("role", "")
                if role:
                    total_tokens += self.count_tokens(role, model)
                
                # 如果有name字段
                if "name" in message:
                    total_tokens += tokens_per_name
                    total_tokens += self.count_tokens(message["name"], model)
            
            # 对话格式的额外开销
            total_tokens += 3  # 每次对话的固定开销
            
            return total_tokens
            
        except Exception as e:
            logger.error(f"消息token计算失败: {e}")
            # 降级估算
            total_text = ""
            for msg in messages:
                total_text += str(msg.get("content", "")) + " "
            return self._estimate_tokens(total_text)
    
    def count_langchain_messages_tokens(self, messages: List[BaseMessage], model: str = "gpt-4o-mini") -> int:
        """
        计算LangChain消息的token数量
        
        Args:
            messages: LangChain消息列表
            model: 模型名称
            
        Returns:
            总token数量
        """
        try:
            # 转换为标准格式
            standard_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = "user"
                elif isinstance(msg, AIMessage):
                    role = "assistant"
                elif isinstance(msg, SystemMessage):
                    role = "system"
                else:
                    role = "user"  # 默认为user
                
                standard_messages.append({
                    "role": role,
                    "content": msg.content
                })
            
            return self.count_messages_tokens(standard_messages, model)
            
        except Exception as e:
            logger.error(f"LangChain消息token计算失败: {e}")
            # 降级估算
            total_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
            return self._estimate_tokens(total_text)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算token数量（降级方法）
        大致按照1 token ≈ 0.75个单词或4个字符计算
        """
        if not text:
            return 0
        
        # 中文字符按1个字符=1个token计算
        # 英文按4个字符=1个token计算
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        other_chars = len(text) - chinese_chars
        
        estimated_tokens = chinese_chars + (other_chars // 4)
        return max(estimated_tokens, 1)
    
    def validate_token_limit(self, messages: List[Dict[str, Any]], model: str, 
                           max_tokens: Optional[int] = None) -> tuple[bool, int]:
        """
        验证消息是否超过模型token限制
        
        Args:
            messages: 消息列表
            model: 模型名称
            max_tokens: 自定义最大token数(可选)
            
        Returns:
            (是否符合限制, 实际token数)
        """
        try:
            actual_tokens = self.count_messages_tokens(messages, model)
            
            # 确定token限制
            if max_tokens:
                limit = max_tokens
            else:
                limit = self.MODEL_LIMITS.get(model, 100000)  # 默认100k
            
            return actual_tokens <= limit, actual_tokens
            
        except Exception as e:
            logger.error(f"Token限制验证失败: {e}")
            return False, 0
    
    def truncate_messages(self, messages: List[Dict[str, Any]], model: str, 
                         max_tokens: Optional[int] = None, 
                         preserve_system: bool = True) -> List[Dict[str, Any]]:
        """
        截断消息以符合token限制
        
        Args:
            messages: 消息列表
            model: 模型名称
            max_tokens: 最大token数
            preserve_system: 是否保留系统消息
            
        Returns:
            截断后的消息列表
        """
        try:
            if not messages:
                return messages
            
            # 确定token限制
            if max_tokens:
                limit = max_tokens
            else:
                limit = self.MODEL_LIMITS.get(model, 100000)
            
            # 预留一些空间给响应
            target_tokens = int(limit * 0.85)  # 使用85%的限制
            
            # 分离系统消息和其他消息
            system_messages = []
            other_messages = []
            
            for msg in messages:
                if msg.get("role") == "system" and preserve_system:
                    system_messages.append(msg)
                else:
                    other_messages.append(msg)
            
            # 计算系统消息的token
            system_tokens = self.count_messages_tokens(system_messages, model) if system_messages else 0
            remaining_tokens = target_tokens - system_tokens
            
            if remaining_tokens <= 0:
                logger.warning("系统消息已超出token限制")
                return system_messages
            
            # 从最新的消息开始，逐步添加直到达到token限制
            selected_messages = []
            current_tokens = 0
            
            # 倒序遍历其他消息（保留最新的对话）
            for msg in reversed(other_messages):
                msg_tokens = self.count_messages_tokens([msg], model)
                if current_tokens + msg_tokens <= remaining_tokens:
                    selected_messages.insert(0, msg)  # 插入到开头保持顺序
                    current_tokens += msg_tokens
                else:
                    break
            
            # 合并系统消息和选中的消息
            result = system_messages + selected_messages
            
            logger.info(f"消息截断完成: {len(messages)} -> {len(result)}, tokens: {current_tokens + system_tokens}")
            return result
            
        except Exception as e:
            logger.error(f"消息截断失败: {e}")
            return messages[:10]  # 降级策略：只保留前10条消息

    def estimate_response_tokens(self, request_text: str, model: str = "gpt-4o-mini", 
                               multiplier: float = 3.0) -> int:
        """
        估算回复token数量
        
        Args:
            request_text: 请求文本
            model: 模型名称
            multiplier: 回复长度倍数（通常回复比请求长2-4倍）
            
        Returns:
            预估的回复token数
        """
        request_tokens = self.count_tokens(request_text, model)
        return int(request_tokens * multiplier)

    def analyze_conversation_tokens(self, request: str, response: str, 
                                  model: str = "gpt-4o-mini") -> Dict[str, int]:
        """
        分析完整对话的token使用情况
        
        Args:
            request: 请求文本
            response: 回复文本
            model: 模型名称
            
        Returns:
            详细的token分析结果
        """
        request_tokens = self.count_tokens(request, model)
        response_tokens = self.count_tokens(response, model)
        total_tokens = request_tokens + response_tokens
        
        # 计算效率指标
        efficiency_ratio = response_tokens / request_tokens if request_tokens > 0 else 0
        
        return {
            "request_tokens": request_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
            "efficiency_ratio": round(efficiency_ratio, 2),
            "model": model,
            "timestamp": int(time.time())
        }

    def batch_count_tokens(self, texts: List[str], model: str = "gpt-4o-mini") -> List[int]:
        """
        批量计算多个文本的token数
        
        Args:
            texts: 文本列表
            model: 模型名称
            
        Returns:
            每个文本的token数列表
        """
        return [self.count_tokens(text, model) for text in texts]

# 全局实例
token_counter = TokenCounter()