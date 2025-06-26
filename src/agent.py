"""税务问答Agent实现 - 核心调用模块"""
import asyncio
import logging
from typing import List, Dict, Optional, Any, TypedDict

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
from src.utils.exception_handler import exception_handler

logger = logging.getLogger(__name__)

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
              web_search: bool = True, session_files: Optional[List[str]] = None, 
              enable_rag: bool = True, user_id: Optional[str] = None) -> Dict[str, any]:
        """执行查询 - 主要入口方法，返回包含结果和成本信息的字典"""
        
        # 开始请求追踪
        request_id = request_tracker.start_request(question, user_id, thread_id)
        
        try:
            # 1. 处理会话文档和问题增强
            enhanced_question, session_vector_tool = session_processor.process_session_files(
                question, session_files, enable_rag, thread_id
            )
            
            # 2. 获取工具列表
            tools = tools_manager.get_tools(
                web_search=web_search,
                session_vector_tool=session_vector_tool
            )
            
            # 3. 使用重试机制选择LLM
            selected_llm = await request_processor.select_llm_with_retry_mechanism(
                enhanced_question, request_id
            )
            
            # 4. 更新请求追踪中的模型信息
            request_tracker.update_model_selection(request_id, selected_llm["name"])
            
            # 5. 创建工作流并执行
            workflow = workflow_manager.create_graph_with_summary(tools, selected_llm)
            result, ai_responses = await workflow_manager.execute_workflow_with_tracking(
                workflow, enhanced_question, thread_id
            )
            
            # 6. 计算成本
            cost_info = await request_processor.calculate_costs(
                selected_llm, enhanced_question, result, ai_responses, request_id
            )
            
            # 7. 更新成本信息
            request_tracker.update_cost(request_id, cost_info.get("total_cost", 0))
            
            # 8. 完成请求追踪
            request_tracker.complete_request(request_id, success=True)
            
            # 9. 返回完整结果
            return {
                "result": result if result else ["抱歉，未能获取到有效回答"],
                "request_id": request_id,
                "model_used": selected_llm["name"],
                "total_cost": cost_info.get("total_cost", 0),
                "cost_breakdown": cost_info,
                "token_usage": {
                    "input_tokens": cost_info.get("input_tokens", 0),
                    "output_tokens": cost_info.get("output_tokens", 0),
                    "total_tokens": cost_info.get("input_tokens", 0) + cost_info.get("output_tokens", 0)
                },
            }
            
        except RateLimitExceededException as e:
            # 限流异常的特殊处理
            error_response = exception_handler.handle_rate_limit_exception(e)
            request_tracker.complete_request(request_id, success=False, error_message=str(e))
            
            return {
                "result": error_response,
                "request_id": request_id,
                "model_used": None,
                "total_cost": 0,
                "error_type": "rate_limit",
                "retry_after": getattr(e, 'retry_after', 60)
            }
            
        except Exception as e:
            # 清理失败请求的token预留
            if 'selected_llm' in locals() and selected_llm.get("reservation_key"):
                await request_processor.cleanup_failed_request(selected_llm)
            
            error_response = exception_handler.handle_general_exception(e, "处理查询请求")
            request_tracker.complete_request(request_id, success=False, error_message=str(e))
            
            return {
                "result": error_response,
                "request_id": request_id,
                "model_used": None,
                "total_cost": 0,
                "error_type": "general"
            }

# 创建全局实例
tax_agent = TaxAgent()
