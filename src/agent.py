"""
完全异步的税务问答Agent
绕过LangChain的同步限制，直接使用异步LLM客户端
"""
import asyncio
import logging
import uuid
from typing import List, Dict, Optional, Any
from src.utils.tools_manager import tools_manager
from src.utils.llm_client import llm_client, LLMUsageInfo

logger = logging.getLogger(__name__)

class AsyncTaxAgent:
    """完全异步的税务问答Agent"""
    
    def __init__(self):
        self.tools_manager = tools_manager
        self.llm_client = llm_client
        logger.info("AsyncTaxAgent初始化完成")
    
    async def query(self, question: str, thread_id: Optional[str] = None, 
                   session_files: Optional[List[str]] = None, 
                   user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        异步查询处理
        
        Args:
            question: 用户问题
            thread_id: 会话ID
            session_files: 会话文件列表
            user_id: 用户ID
            
        Returns:
            包含回答和使用统计的字典
        """
        request_id = str(uuid.uuid4())
        logger.info(f"开始处理异步查询: {question[:100]}...", extra={
            'request_id': request_id,
            'thread_id': thread_id,
            'user_id': user_id
        })
        
        try:
            # 1. 异步意图识别
            messages = [{"role": "user", "content": question}]
            intention_result = await self.tools_manager.intention_recognition(messages)
            
            intentions = intention_result.get("Intentions", [])
            if not intentions:
                # 没有识别到意图，直接返回通用回答
                answer = await self.tools_manager.general_response(question)
                return self._build_response(request_id, [answer], intention_result.get("_usage_info"))
            
            # 2. 并发执行工具
            logger.info(f"并发执行{len(intentions)}个意图")
            tool_results = await self.tools_manager.execute_tools_concurrently(intentions, question)
            
            # 3. 最终总结
            final_answer = await self.tools_manager.final_summary(
                query=question, 
                tool_results=[
                    {"tool": f"intention_{i}", "result": result} 
                    for i, result in enumerate(tool_results)
                ],
                intention_result=intention_result
            )
            
            # 只返回最终总结结果
            return self._build_response(request_id, [final_answer], intention_result.get("_usage_info"))
            
        except Exception as e:
            logger.error(f"异步查询失败: {e}", extra={'request_id': request_id})
            
            # 降级处理：直接返回通用回答
            try:
                fallback_answer = await self.tools_manager.general_response(question)
                return self._build_response(request_id, [fallback_answer], None, error=str(e))
            except Exception as fallback_error:
                logger.error(f"降级处理也失败: {fallback_error}")
                return self._build_error_response(request_id, str(e))
    
    def _build_response(self, request_id: str, results: List[str], 
                       usage_info: Optional[Dict] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """构建响应结果"""
        response = {
            "result": results,
            "request_id": request_id,
            "model_used": usage_info.get("model_used", "unknown") if usage_info else "unknown",
            "provider": "remote_llm_service",
            "total_cost": usage_info.get("total_cost", 0.0) if usage_info else 0.0,
            "currency": "CNY",
            "token_usage": usage_info.get("token_usage", {}) if usage_info else {},
            "cost_breakdown": {
                "total_cost": usage_info.get("total_cost", 0.0) if usage_info else 0.0,
                "currency": "CNY"
            }
        }
        
        if error:
            response["error"] = error
            response["fallback_used"] = True
        
        return response
    
    def _build_error_response(self, request_id: str, error: str) -> Dict[str, Any]:
        """构建错误响应"""
        return {
            "result": [f"抱歉，处理您的请求时发生错误: {error}"],
            "request_id": request_id,
            "model_used": "unknown",
            "provider": "error",
            "total_cost": 0.0,
            "currency": "CNY",
            "token_usage": {},
            "cost_breakdown": {"total_cost": 0.0, "currency": "CNY"},
            "error": error
        }

# 创建全局异步agent实例
async_tax_agent = AsyncTaxAgent() 