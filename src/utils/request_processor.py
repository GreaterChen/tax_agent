"""
请求处理器
负责LLM选择、token管理和请求清理等逻辑
"""
import logging
from typing import Dict, List
from langchain_core.messages import AIMessage

from src.utils.llm_selector import llm_selector, RateLimitExceededException
from src.utils.request_tracker import request_tracker
from src.utils.retry_manager import rate_limit_retry_manager
from src.utils.unified_token_manager import unified_token_manager

logger = logging.getLogger(__name__)

class RequestProcessor:
    """请求处理器"""
    
    @staticmethod
    async def select_llm_with_retry_mechanism(enhanced_question: str, request_id: str) -> Dict:
        """使用重试机制选择LLM"""
        async def select_llm_operation():
            try:
                return await llm_selector.select_best_llm(enhanced_question)
            except RateLimitExceededException as e:
                # 增加重试计数
                request_tracker.increment_retry(request_id)
                raise e
        
        # 使用专门的限流重试管理器
        return await rate_limit_retry_manager.retry_llm_selection(select_llm_operation)
    
    @staticmethod
    async def calculate_costs(llm_info: Dict, request_text: str, 
                            responses: List[str], ai_responses: List[AIMessage], 
                            request_id: str) -> Dict:
        """计算完整的成本信息，使用统一Token管理器"""
        try:
            llm_name = llm_info["name"]
            llm_instance = llm_info["llm"]
            
            # 合并所有响应文本
            combined_response = " ".join(responses)
            
            # 使用统一Token管理器计算token使用量
            # 优先使用API响应中的token信息
            api_response = ai_responses[-1] if ai_responses else None
            
            token_usage = unified_token_manager.calculate_token_usage(
                request_text=request_text,
                response_text=combined_response,
                llm_instance=llm_instance,
                api_response=api_response
            )
            
            # 更新请求追踪中的token信息
            request_tracker.update_token_usage(
                request_id, 
                token_usage.input_tokens, 
                token_usage.output_tokens, 
                token_usage.cached_tokens
            )
            
            # 计算成本
            cost_info = unified_token_manager.calculate_cost(token_usage, llm_info)
            
            # 转换为传统格式以保持兼容性
            result = {
                "llm_name": cost_info.llm_name,
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "cached_tokens": token_usage.cached_tokens,
                "total_tokens": token_usage.total_tokens,
                "input_cost": cost_info.input_cost,
                "output_cost": cost_info.output_cost,
                "cached_cost": cost_info.cached_cost,
                "total_cost": cost_info.total_cost,
                "currency": cost_info.currency,
                "token_source": token_usage.source,
                "provider": token_usage.provider,
                "model_used": token_usage.model_used
            }
            
            # 完成TPM最终统计
            await RequestProcessor.finalize_token_usage_with_actual_data(
                llm_info, 
                token_usage.input_tokens, 
                token_usage.output_tokens
            )
            
            logger.info(f"成本计算完成 - {llm_name}: "
                       f"输入{token_usage.input_tokens}tokens, "
                       f"输出{token_usage.output_tokens}tokens, "
                       f"缓存{token_usage.cached_tokens}tokens, "
                       f"成本{cost_info.total_cost}{cost_info.currency}, "
                       f"来源: {token_usage.source}")
            
            return result
            
        except Exception as e:
            logger.error(f"成本计算失败: {e}")
            return {
                "error": str(e),
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "total_cost": 0.0,
                "token_source": "error"
            }
    
    @staticmethod
    async def finalize_token_usage_with_actual_data(llm_info: Dict, actual_input_tokens: int, actual_output_tokens: int):
        """使用实际token数据完成TPM统计"""
        try:
            llm_name = llm_info["name"]
            reservation_key = llm_info.get("reservation_key")
            
            if not reservation_key:
                logger.warning(f"未找到{llm_name}的token预留信息，跳过TPM最终确定")
                return
            
            # 使用实际的token数据（来自API响应或fallback计算）
            finalize_result = await llm_selector.rate_limiter.finalize_token_usage(
                llm_name,
                reservation_key,
                actual_input_tokens,
                actual_output_tokens
            )
            
            # 记录详细的使用统计
            total_actual_tokens = actual_input_tokens + actual_output_tokens
            estimated_tokens = llm_info.get("estimated_total_tokens", 0)
            
            logger.info(f"TPM统计完成 - {llm_name}: "
                       f"输入Token={actual_input_tokens}, "
                       f"输出Token={actual_output_tokens}, "
                       f"实际总计={total_actual_tokens}, "
                       f"预估总计={estimated_tokens}, "
                       f"效率={finalize_result.get('efficiency', 'N/A')}")
            
        except Exception as e:
            logger.error(f"TPM最终统计失败: {e}")
    
    @staticmethod
    async def cleanup_failed_request(llm_info: Dict):
        """清理失败请求的token预留"""
        try:
            llm_name = llm_info["name"]
            reservation_key = llm_info.get("reservation_key")
            
            if reservation_key:
                # 清理预留的token（设置实际使用为0）
                await llm_selector.rate_limiter.finalize_token_usage(
                    llm_name, reservation_key, 0, 0
                )
                logger.info(f"已清理{llm_name}的失败请求token预留")
                
        except Exception as e:
            logger.error(f"清理失败请求token预留时出错: {e}")

# 全局实例
request_processor = RequestProcessor() 