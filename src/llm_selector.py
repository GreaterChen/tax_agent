"""
LLM选择器模块
负责根据限流状态选择最佳可用的LLM
"""
import asyncio
import logging
from typing import Dict, Any
from config.llm_config import llm_config
from src.utils.rate_limiter import RateLimiter
from src.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)

class LLMSelector:
    """LLM选择器 - 负责智能选择可用的LLM"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.token_counter = TokenCounter()
        logger.info("LLM选择器初始化完成，限流功能已启用")
    
    async def select_best_llm(self, question: str) -> Dict[str, Any]:
        """
        选择最佳可用的LLM
        
        Args:
            question: 用户问题，用于计算token数
            
        Returns:
            选中的LLM配置
        """
        # 获取可用的LLM列表
        available_llms = llm_config.get_available_llms()
        
        # 使用第一个可用模型来计算token（因为大部分模型使用相同的编码器）
        reference_model = self._get_reference_model_name(available_llms[0])
        request_tokens = self.token_counter.count_tokens(question, reference_model)
        
        # 按优先级尝试每个LLM
        
        for llm_config_item in available_llms:
            if not llm_config_item["enabled"]:
                continue
                
            llm_name = llm_config_item["name"]
            
            try:
                # 检查限流状态
                allowed, status = await self.rate_limiter.check_and_increment(
                    llm_name, 
                    request_count=1, 
                    token_count=request_tokens,
                    qpm_limit=llm_config_item["qpm_limit"],
                    tpm_limit=llm_config_item["tpm_limit"]
                )
                
                if allowed:
                    logger.info(f"选择LLM: {llm_name}")
                    return llm_config_item
                else:
                    logger.warning(f"LLM {llm_name} 限流中: {status.get('reason', 'Unknown')}")
                    continue
                    
            except Exception as e:
                logger.error(f"检查LLM {llm_name} 限流状态失败: {e}")
                # 继续尝试下一个LLM
                continue
        
        # 如果所有LLM都限流，使用第一个可用的（降级策略）
        logger.warning("所有LLM都达到限流，使用优先级最高的LLM（降级策略）")
        return available_llms[0]
    
    def _get_reference_model_name(self, llm_config_item: Dict[str, Any]) -> str:
        """
        从LLM配置中提取模型名称用于token计算
        
        Args:
            llm_config_item: LLM配置项
            
        Returns:
            模型名称字符串
        """
        # 尝试从配置中获取模型名称
        llm = llm_config_item.get("llm")
        if hasattr(llm, "model_name"):
            return llm.model_name
        elif hasattr(llm, "model"):
            return llm.model
        else:
            # 回退到默认模型（大部分模型使用相同的编码器）
            return "gpt-4o-mini"
    
    def disable_llm(self, llm_name: str):
        """临时禁用指定的LLM"""
        available_llms = llm_config.get_available_llms()
        for llm_config_item in available_llms:
            if llm_config_item["name"] == llm_name:
                llm_config_item["enabled"] = False
                logger.warning(f"LLM {llm_name} 已被临时禁用")
                break
    
    def enable_llm(self, llm_name: str):
        """重新启用指定的LLM"""
        available_llms = llm_config.get_available_llms()
        for llm_config_item in available_llms:
            if llm_config_item["name"] == llm_name:
                llm_config_item["enabled"] = True
                logger.info(f"LLM {llm_name} 已重新启用")
                break
    
    async def get_usage_status(self) -> Dict[str, Any]:
        """获取所有LLM的使用状态"""
        status = {"llms": {}}
        
        available_llms = llm_config.get_available_llms()
        for llm_config_item in available_llms:
            llm_name = llm_config_item["name"]
            try:
                usage = await self.rate_limiter.get_current_usage(llm_name)
                status["llms"][llm_name] = {
                    "provider": llm_config_item["provider"],
                    "priority": llm_config_item["priority"],
                    "enabled": llm_config_item["enabled"],
                    "current_qpm": usage.get("requests", 0),
                    "current_tpm": usage.get("tokens", 0),
                    "qpm_limit": llm_config_item["qpm_limit"],
                    "tpm_limit": llm_config_item["tpm_limit"],
                    "qpm_usage_percent": round(usage.get("requests", 0) / llm_config_item["qpm_limit"] * 100, 1),
                    "tpm_usage_percent": round(usage.get("tokens", 0) / llm_config_item["tpm_limit"] * 100, 1)
                }
            except Exception as e:
                logger.error(f"获取LLM {llm_name} 使用状态失败: {e}")
                status["llms"][llm_name] = {
                    "provider": llm_config_item["provider"],
                    "priority": llm_config_item["priority"],
                    "enabled": llm_config_item["enabled"],
                    "error": str(e)
                }
        
        return status

# 全局选择器实例
llm_selector = LLMSelector() 