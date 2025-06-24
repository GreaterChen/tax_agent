"""
LLM配置管理模块
"""
import os
from typing import List, Dict, Any
from langchain_community.chat_models import ChatTongyi
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
        # qwen-max-latest配置
        if os.getenv("DASHSCOPE_API_KEY"):
            self.llm_configs.append({
                "name": "qwen_max_latest",
                "llm": ChatTongyi(
                    model="qwen-max-latest",
                    api_key=os.getenv("DASHSCOPE_API_KEY")
                ),
                "provider": "qwen",
                "priority": 1,
                "qpm_limit": 1200,
                "tpm_limit": 1000000,
                "enabled": True
            })
        
        # gpt-4o-mini配置
        if os.getenv("OPENAI_API_KEY"):
            self.llm_configs.append({
                "name": "gpt_4o_mini",
                "llm": ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=os.getenv("OPENAI_API_KEY")
                ),
                "provider": "openai", 
                "priority": 2,
                "qpm_limit": 50,
                "tpm_limit": 150000,
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
                    "tpm_limit": llm_config["tpm_limit"]
                }
                for llm_config in self.available_llms
            }
        }

# 全局配置实例
llm_config = LLMConfig() 