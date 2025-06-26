"""
会话处理器
负责处理会话文件和问题增强逻辑
"""
import logging
from typing import List, Optional, Tuple

from src.utils.tools_manager import tools_manager
from src.utils.prompts import create_enhanced_question, create_non_rag_question

logger = logging.getLogger(__name__)

class SessionProcessor:
    """会话处理器"""
    
    @staticmethod
    def process_session_files(question: str, session_files: Optional[List[str]], 
                            enable_rag: bool, thread_id: str) -> Tuple[str, Optional[object]]:
        """
        处理会话文件和问题增强
        
        Args:
            question: 原始问题
            session_files: 会话文件列表
            enable_rag: 是否启用RAG
            thread_id: 线程ID
            
        Returns:
            Tuple[str, Optional[object]]: (增强后的问题, 会话向量工具)
        """
        session_vector_tool = None
        enhanced_question = question
        
        if session_files and len(session_files) > 0:
            if enable_rag:
                # RAG模式：创建会话级向量搜索工具
                session_vector_tool = tools_manager.create_session_vector_tool(session_files, thread_id)
                enhanced_question = create_enhanced_question(question, session_files)
            else:
                # 非RAG模式：直接读取文件内容
                try:
                    from src.utils.file_utils import read_session_files_content
                    file_contents = read_session_files_content(session_files)
                    if file_contents:
                        enhanced_question = create_non_rag_question(question, file_contents)
                except ImportError:
                    logger.warning("file_utils模块不可用，跳过文件内容读取")
        
        return enhanced_question, session_vector_tool

# 全局实例
session_processor = SessionProcessor() 