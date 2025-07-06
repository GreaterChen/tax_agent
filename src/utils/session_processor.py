"""
会话处理器
负责处理会话文件和问题增强逻辑 - 完全采用新的文件处理系统
"""
import logging
import asyncio
from typing import List, Optional, Tuple, Dict, Any

from src.utils.prompts import create_non_rag_question
from src.utils.file_utils import message_manager

logger = logging.getLogger(__name__)

class SessionProcessor:
    """会话处理器"""
    
    def __init__(self):
        self.pending_file_summaries = {}  # 存储待处理的总结任务
    
    async def process_session_files(self, question: str, session_files: Optional[List[str]], 
                                  thread_id: str) -> Tuple[str, Optional[object], List[Dict[str, Any]]]:
        """
        处理会话文件和问题增强
        
        Args:
            question: 原始问题
            session_files: 会话文件列表
            thread_id: 线程ID
            
        Returns:
            Tuple[str, Optional[object], List[Dict[str, Any]]]: (增强后的问题, 会话向量工具, 文件消息列表)
        """
        session_vector_tool = None
        enhanced_question = question
        file_messages = []
        
        if session_files and len(session_files) > 0:
            # 使用新的文件处理逻辑
            try:
                # 处理文件消息
                file_messages = await message_manager.process_file_messages(session_files)
                
                # 将文件消息存储到pending tasks中
                self.pending_file_summaries[thread_id] = file_messages
                
                # 构建增强问题
                if file_messages:
                    # 提取文件内容用于问题增强
                    file_contents = []
                    for msg in file_messages:
                        file_contents.append(msg["content"])
                    
                    combined_content = "\n".join(file_contents)
                    enhanced_question = create_non_rag_question(question, combined_content)
                
            except Exception as e:
                logger.error(f"新文件处理系统失败: {e}")
                # 不进行降级，直接使用原始问题
                enhanced_question = question
        
        return enhanced_question, session_vector_tool, file_messages
    
    async def finalize_session_summaries(self, thread_id: str) -> bool:
        """
        完成会话的文件总结任务
        
        Args:
            thread_id: 线程ID
            
        Returns:
            bool: 是否成功完成总结
        """
        if thread_id not in self.pending_file_summaries:
            return True
        
        try:
            # 获取待处理的文件消息
            file_messages = self.pending_file_summaries[thread_id]
            
            # 完成总结任务
            updated_messages = await message_manager.finalize_summaries(file_messages)
            
            # 更新存储的消息
            self.pending_file_summaries[thread_id] = updated_messages
            
            logger.info(f"完成会话 {thread_id} 的文件总结任务")
            return True
            
        except Exception as e:
            logger.error(f"完成会话总结失败 {thread_id}: {e}")
            return False
    
    def get_processed_file_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        获取已处理的文件消息
        
        Args:
            thread_id: 线程ID
            
        Returns:
            List[Dict[str, Any]]: 已处理的文件消息列表
        """
        return self.pending_file_summaries.get(thread_id, [])
    
    def cleanup_session(self, thread_id: str):
        """
        清理会话相关数据
        
        Args:
            thread_id: 线程ID
        """
        if thread_id in self.pending_file_summaries:
            del self.pending_file_summaries[thread_id]
            logger.info(f"清理会话 {thread_id} 的数据")

# 全局实例
session_processor = SessionProcessor() 