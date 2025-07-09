"""Hong Kong Tax Expert Tool - Examist1

This tool implements professional Hong Kong tax consultation functions, including:
1. Tax question filtering and analysis
2. Textbook content retrieval
3. Legal statute queries
4. IRAC structure answer generation
"""

import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Optional

# 添加examist模块路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 导入examist1核心模块
from .core import core

# 配置日志
logger = logging.getLogger("examist_tool")

class ExamistTool:
    """Hong Kong tax expert tool class"""
    
    def __init__(self):
        """Initialize Hong Kong tax expert tool"""
        self.result_cache = {}
        self.cache_ttl = 3600  # 1小时缓存
        logger.info("ExamistTool初始化完成")

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key"""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache is valid"""
        return time.time() - timestamp < self.cache_ttl

    async def analyze_tax_query(self, query: str) -> str:
        """Analyze Hong Kong tax questions and generate professional answers
        
        Args:
            query: User's Hong Kong tax question
            
        Returns:
            Professional tax consultation answer
        """
        # 记录查询开始
        logger.info(f"===== 开始香港税务问题分析 =====")
        logger.info(f"用户查询: {query[:100]}...")
        
        # 检查缓存
        cache_key = self._get_cache_key(query)
        if cache_key in self.result_cache:
            cached_result, timestamp = self.result_cache[cache_key]
            if self._is_cache_valid(timestamp):
                logger.info("使用缓存的分析结果")
                return cached_result
        
        try:
            # 调用examist1核心处理函数
            logger.info("调用Examist1核心处理流程")
            start_time = time.time()
            
            result = await core(query)
            
            processing_time = time.time() - start_time
            logger.info(f"Examist1处理完成，耗时: {processing_time:.2f}秒")
            
            # 处理结果
            if result.get('status') == '1':
                # 成功处理
                content = result.get('content', '')
                logger.info(f"税务分析成功，回答长度: {len(content)} 字符")
                
                # 缓存结果 - 直接缓存原始内容
                self.result_cache[cache_key] = (content, time.time())
                
                logger.info("===== 香港税务问题分析完成 =====")
                return content
                
            else:
                # Processing failed - 直接返回原始错误内容
                error_content = result.get('content', 'Processing failed')
                logger.warning(f"税务分析失败: {error_content}")
                
                return error_content
                
        except Exception as e:
            logger.error(f"香港税务分析过程中发生错误: {str(e)}", exc_info=True)
            
            return f"Technical error occurred: {str(e)}"

# 创建工具实例
examist_tool_instance = ExamistTool() 