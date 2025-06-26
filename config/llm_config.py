"""
LLM配置管理模块
"""
import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class LLMConfig:
    """LLM配置管理器"""
    
    def __init__(self):
        self.llm_configs = []
        self._build_configs()
    
    def _build_configs(self):
        """构建LLM配置"""
        # qwen-plus配置 (使用ChatOpenAI兼容模式)
        if os.getenv("DASHSCOPE_API_KEY"):
            self.llm_configs.append({
                "name": "qwen_max",
                "llm": ChatOpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    model="qwen-max"
                ),
                "provider": "qwen",
                "priority": 1,
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": os.getenv("DASHSCOPE_API_KEY"),
                "model_name": "qwen-max",
                "input_price": 0.0024,  # 每1K tokens价格
                "output_price": 0.0096,  # 每1K tokens价格
                "cached_price": 0.00096,  # 命中缓存的价格
                "currency": "CNY",  # 货币类型
                "qpm_limit": 1200,
                "tpm_limit": 1000000,
                "max_context_tokens": 10000,  # 最大上下文token数
                "summary_trigger_tokens": 8000,  # 触发总结的token阈值
                "max_summary_tokens": 600,  # 总结最大token数
                "enabled": True
            })
        
        # gpt-4o-mini配置
        if os.getenv("OPENAI_API_KEY"):
            self.llm_configs.append({
                "name": "gpt_4o_mini",
                "llm": ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                ),
                "provider": "openai", 
                "priority": 2,
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model_name": "gpt-4o-mini",
                "input_price": 0.00015,  # 每1K tokens价格(USD)
                "output_price": 0.0006,  # 每1K tokens价格(USD)
                "cached_price": 0.000075,  # 命中缓存的价格(USD)
                "currency": "USD",  # 货币类型
                "qpm_limit": 50,
                "tpm_limit": 150000,
                "max_context_tokens": 8000,  # 最大上下文token数
                "summary_trigger_tokens": 6000,  # 触发总结的token阈值
                "max_summary_tokens": 500,  # 总结最大token数
                "enabled": True
            })
        
        # 过滤有效的模型
        self.available_llms = [
            config for config in self.llm_configs 
            if config["enabled"]
        ]
        
        if not self.available_llms:
            raise Exception("没有可用的LLM配置，请检查环境变量设置")
            
        logger.info(f"LLM配置加载完成，可用模型: {[llm['name'] for llm in self.available_llms]}")
    
    def get_available_llms(self) -> List[Dict[str, Any]]:
        """获取可用的LLM配置列表"""
        return self.available_llms
    
    def get_llm_by_name(self, name: str) -> Dict[str, Any]:
        """根据名称获取LLM配置"""
        for llm_config in self.available_llms:
            if llm_config["name"] == name:
                return llm_config
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """获取配置状态"""
        return {
            "total_models": len(self.llm_configs),
            "available_models": len(self.available_llms),
            "models": {
                llm_config["name"]: {
                    "provider": llm_config["provider"],
                    "priority": llm_config["priority"],
                    "enabled": llm_config["enabled"],
                    "qpm_limit": llm_config["qpm_limit"],
                    "tpm_limit": llm_config["tpm_limit"],
                    "max_context_tokens": llm_config["max_context_tokens"],
                    "summary_trigger_tokens": llm_config["summary_trigger_tokens"],
                    "max_summary_tokens": llm_config["max_summary_tokens"]
                }
                for llm_config in self.available_llms
            }
        }

    def calculate_cost(self, llm_name: str, input_tokens: int, output_tokens: int, 
                      cached_tokens: int = 0) -> dict:
        """
        计算LLM调用成本（保留向后兼容性）
        
        Args:
            llm_name: LLM名称
            input_tokens: 输入token数
            output_tokens: 输出token数
            cached_tokens: 缓存命中token数
            
        Returns:
            成本详情字典
        """
        llm_config = self.get_llm_by_name(llm_name)
        if not llm_config:
            return {"error": f"未找到LLM配置: {llm_name}"}
        
        # 使用统一Token管理器计算成本
        from src.utils.token_manager import token_manager, TokenUsage
        
        token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            source="config_calculation"
        )
        
        cost_info = token_manager.calculate_cost(token_usage, llm_config)
        
        # 转换为传统格式
        return {
            "llm_name": llm_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "input_cost": cost_info.input_cost,
            "output_cost": cost_info.output_cost,
            "cached_cost": cost_info.cached_cost,
            "total_cost": cost_info.total_cost,
            "currency": cost_info.currency
        }

    def get_pricing_info(self, llm_name: str) -> dict:
        """获取LLM定价信息"""
        llm_config = self.get_llm_by_name(llm_name)
        if not llm_config:
            return {"error": f"未找到LLM配置: {llm_name}"}
            
        return {
            "llm_name": llm_name,
            "provider": llm_config["provider"],
            "input_price": llm_config["input_price"],
            "output_price": llm_config["output_price"],
            "cached_price": llm_config["cached_price"],
            "currency": llm_config["currency"],  # 使用模型配置中的货币
            "unit": "per 1K tokens"
        }

# 全局配置实例
llm_config = LLMConfig() 