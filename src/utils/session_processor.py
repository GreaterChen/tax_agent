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
                                  thread_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        处理会话文件和问题增强
        
        Args:
            question: 原始问题
            session_files: 会话文件列表
            thread_id: 线程ID
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: (原始问题, 文件消息列表)
        """
        file_messages = []
        
        if session_files and len(session_files) > 0:
            # 使用新的文件处理逻辑
            try:
                # 处理文件消息
                file_messages = await message_manager.process_file_messages(session_files)
                
                # 将文件消息存储到pending tasks中
                self.pending_file_summaries[thread_id] = file_messages
                
                logger.info(f"成功处理 {len(file_messages)} 个文件消息")
                
            except Exception as e:
                logger.error(f"新文件处理系统失败: {e}")
                file_messages = []
        
        # 返回原始问题和文件消息列表
        return question, file_messages
    
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