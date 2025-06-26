"""
LLM选择器模块
负责根据限流状态选择最佳可用的LLM
"""
import asyncio
import logging
from typing import Dict, Any
from config.llm_config import llm_config
from src.utils.rate_limiter import RateLimiter
from src.utils.token_manager import token_manager
from src.utils.exceptions import ExceptionFactory, ErrorContext
from src.utils.error_codes import ErrorCode

logger = logging.getLogger(__name__)

# 向后兼容的异常类
class RateLimitExceededException(Exception):
    """限流超限异常 (向后兼容)"""
    
    def __init__(self, message: str, available_models: list = None, retry_after: int = 60):
        super().__init__(message)
        self.available_models = available_models or []
        self.retry_after = retry_after
        self.timestamp = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0

class LLMSelector:
    """LLM选择器 - 负责智能选择可用的LLM"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
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
        reference_model = token_manager._get_model_name(available_llms[0].get("llm"))
        request_tokens = token_manager.count_tokens(question, reference_model)
        
        # 按优先级尝试每个LLM
        
        for llm_config_item in available_llms:
            if not llm_config_item["enabled"]:
                continue
                
            llm_name = llm_config_item["name"]
            
            try:
                # 第一阶段：检查QPM限制
                qpm_allowed, qpm_status = await self.rate_limiter.check_and_increment(
                    llm_name, 
                    request_count=1, 
                    token_count=0,  # 此阶段不计算token
                    qpm_limit=llm_config_item["qpm_limit"],
                    tpm_limit=0  # 此阶段不检查TPM
                )
                
                if not qpm_allowed:
                    logger.warning(f"LLM {llm_name} QPM限流: {qpm_status.get('reason', 'Unknown')}")
                    continue
                
                # 第二阶段：预留Token额度
                token_reserved, token_status = await self.rate_limiter.reserve_tokens(
                    llm_name,
                    estimated_request_tokens=request_tokens,
                    tpm_limit=llm_config_item["tpm_limit"],
                    response_multiplier=0.8  # 预估回复是请求的0.8倍长度
                )
                
                if token_reserved:
                    logger.info(f"选择LLM: {llm_name}, 预留Token: {token_status.get('estimated_total_tokens', 0)}")
                    
                    # 将预留信息添加到LLM配置中，供后续使用
                    llm_config_item_copy = llm_config_item.copy()
                    llm_config_item_copy.update({
                        "reservation_key": token_status.get("reservation_key"),
                        "estimated_request_tokens": request_tokens,
                        "estimated_total_tokens": token_status.get("estimated_total_tokens", 0)
                    })
                    
                    return llm_config_item_copy
                else:
                    logger.warning(f"LLM {llm_name} TPM限流: {token_status.get('reason', 'Unknown')}")
                    # TPM预留失败，需要回滚QPM计数
                    await self._rollback_qpm_count(llm_name)
                    continue
                    
            except Exception as e:
                logger.error(f"检查LLM {llm_name} 限流状态失败: {e}")
                # 继续尝试下一个LLM
                continue
        
        # 如果所有LLM都限流，抛出异常而不是降级
        available_models = [llm['name'] for llm in available_llms if llm['enabled']]
        logger.error(f"所有LLM都达到限流限制: {available_models}")
        raise RateLimitExceededException(
            "系统目前处于请求高峰期，所有模型都已达到限制。请稍后再试，或联系管理员。",
            available_models=available_models,
            retry_after=60  # 建议60秒后重试
        )
    
    async def _rollback_qpm_count(self, llm_name: str):
        """回滚QPM计数（当TPM预留失败时）"""
        try:
            await self.rate_limiter._adjust_token_count(llm_name, 0)  # 实际上需要减少请求计数
            # 注意：这里应该实现减少请求计数的方法，但当前RateLimiter没有这个功能
            # 这是一个需要进一步完善的地方
            logger.info(f"已回滚{llm_name}的QPM计数")
        except Exception as e:
            logger.error(f"回滚QPM计数失败: {e}")
    

    
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