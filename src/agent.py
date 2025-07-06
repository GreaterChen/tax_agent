"""税务问答Agent实现 - 核心调用模块"""
import asyncio
import logging
from typing import List, Dict, Optional, Any, TypedDict
from .utils.logging_config import get_logger

from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import AnyMessage

# 修补 typing 模块以支持 Python < 3.11
import typing
if not hasattr(typing, 'NotRequired'):
    from typing_extensions import NotRequired
    typing.NotRequired = NotRequired

from config.llm_config import llm_config
from src.utils.llm_selector import RateLimitExceededException
from src.utils.tools_manager import tools_manager
from src.utils.request_tracker import request_tracker

# 导入专门的管理器
from src.utils.workflow_manager import workflow_manager
from src.utils.session_processor import session_processor
from src.utils.request_processor import request_processor

# 导入新的异常系统
from src.utils.exceptions import (
    ExceptionFactory,
    ErrorContext,
    RateLimitException,
    BaseBusinessException
)
from src.utils.error_codes import ErrorCode

logger = get_logger(__name__)

class AgentState(MessagesState):
    """Agent状态，包含消息和上下文"""
    context: dict[str, Any]

class LLMInputState(TypedDict):
    """LLM输入状态"""
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]

class TaxAgent:
    """税务问答Agent - 核心调用类"""
    
    def __init__(self):
        logger.info("TaxAgent初始化完成")

    async def query(self, question: str, thread_id: Optional[str] = None, 
              session_files: Optional[List[str]] = None, 
              user_id: Optional[str] = None) -> Dict[str, any]:
        """执行查询 - 主要入口方法，返回包含结果和成本信息的字典"""
        
        # 开始请求追踪
        request_id = request_tracker.start_request(question, user_id, thread_id)
        
        try:
            # 1. 处理会话文档和问题增强
            user_question, session_vector_tool, file_messages = await session_processor.process_session_files(
                question, session_files, thread_id
            )
            
            # 2. 获取工具列表
            tools = tools_manager.get_tools(
                web_search=True,
                session_vector_tool=session_vector_tool
            )
            
            # 3. 使用重试机制选择LLM
            selected_llm = await request_processor.select_llm_with_retry_mechanism(
                user_question, request_id
            )
            
            # 4. 更新请求追踪中的模型信息
            request_tracker.update_model_selection(request_id, selected_llm["name"])
            
            # 5. 创建工作流并执行
            workflow = workflow_manager.create_graph_with_summary(tools, selected_llm)
            result, ai_responses = await workflow_manager.execute_workflow_with_tracking(
                workflow, user_question, thread_id, file_messages
            )
            
            # 6. 完成会话的文件总结任务（在对话结束后）
            if session_files:
                try:
                    await session_processor.finalize_session_summaries(thread_id)
                    logger.info(f"完成会话 {thread_id} 的文件总结任务")
                except Exception as e:
                    logger.error(f"完成文件总结失败: {e}")
            
            # 7. 计算成本
            cost_info = await request_processor.calculate_costs(
                selected_llm, user_question, result, ai_responses, request_id
            )
            
            # 8. 更新成本信息
            request_tracker.update_cost(request_id, cost_info.get("total_cost", 0))
            
            # 9. 完成请求追踪
            request_tracker.complete_request(request_id, success=True)
            
            # 10. 构建响应结果
            response_result = {
                "result": result if result else ["抱歉，未能获取到有效回答"],
                "request_id": request_id,
                "model_used": selected_llm["name"],
                "provider": cost_info.get("provider", "unknown"),
                "total_cost": cost_info.get("total_cost", 0),
                "currency": cost_info.get("currency", "CNY"),
                
                # 详细的token使用量信息
                "token_usage": {
                    "input_tokens": cost_info.get("input_tokens", 0),
                    "output_tokens": cost_info.get("output_tokens", 0),
                    "cached_tokens": cost_info.get("cached_tokens", 0),
                    "total_tokens": cost_info.get("total_tokens", 0),
                    "token_source": cost_info.get("token_source", "unknown")
                },
                
                # 详细的成本分解信息
                "cost_breakdown": {
                    "input_cost": cost_info.get("input_cost", 0),
                    "output_cost": cost_info.get("output_cost", 0),
                    "cached_cost": cost_info.get("cached_cost", 0),
                    "total_cost": cost_info.get("total_cost", 0),
                    "currency": cost_info.get("currency", "CNY"),
                },
            }
            
            # 11. 添加文件处理信息
            if file_messages:
                response_result["file_info"] = {
                    "file_count": len(file_messages),
                    "files": [msg.get("file_info", {}) for msg in file_messages]
                }
            
            return response_result
            
        except RateLimitExceededException as e:
            # 限流异常的特殊处理
            request_tracker.complete_request(request_id, success=False, error_message=str(e))
            
            # 转换为新的异常系统
            context = ErrorContext(
                request_id=request_id,
                user_id=user_id,
                session_id=thread_id,
                operation="tax_agent_query",
                component="llm_selector"
            )
            
            raise ExceptionFactory.create_rate_limit_exception(
                error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                retry_after=getattr(e, 'retry_after', 60),
                available_models=getattr(e, 'available_models', []),
                context=context,
                message=str(e)
            )
            
        except BaseBusinessException:
            # 业务异常直接重新抛出
            request_tracker.complete_request(request_id, success=False, error_message="业务异常")
            raise
            
        except Exception as e:
            # 清理失败请求的token预留
            if 'selected_llm' in locals() and selected_llm.get("reservation_key"):
                await request_processor.cleanup_failed_request(selected_llm)
            
            request_tracker.complete_request(request_id, success=False, error_message=str(e))
            
            # 转换为业务异常
            context = ErrorContext(
                request_id=request_id,
                user_id=user_id,
                session_id=thread_id,
                operation="tax_agent_query",
                component="tax_agent"
            )
            
            raise ExceptionFactory.create_business_exception(
                error_code=ErrorCode.AGENT_QUERY_FAILED,
                cause=e,
                context=context
            )

# 创建全局实例
tax_agent = TaxAgent()
