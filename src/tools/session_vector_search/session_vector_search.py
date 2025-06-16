"""会话级向量搜索工具

此工具实现基于会话文档的向量搜索功能，支持上传文件的RAG功能。
每个会话有独立的文档数据库，使用混合检索技术（BM25 + 余弦相似度）。
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 添加Dependencies目录到Python路径
current_dir = Path(__file__).parent
dependencies_dir = current_dir / "Dependencies"
if str(dependencies_dir) not in sys.path:
    sys.path.append(str(dependencies_dir))

# 导入本工具独立的依赖模块
from embedding import (
    process_query_embedding, 
    process_session_files,
    save_session_database,
    load_session_database
)
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("session_vector_search")

# 输入模型
class SessionVectorSearchInput(BaseModel):
    """会话级向量搜索输入"""
    query: str = Field(..., description="搜索查询字符串")

class SessionVectorSearchTool:
    def __init__(self, session_files: List[str], thread_id: str):
        """初始化会话级向量搜索工具
        
        Args:
            session_files: 会话文档文件路径列表
            thread_id: 会话ID
        """
        self.session_files = session_files
        self.thread_id = thread_id
        self.db_path = current_dir / "Dependencies" / "database" / thread_id
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_path = self.db_path / "embeddings.json"
        self.content_path = self.db_path / "content.json"
        
        # 处理会话文档并加载数据库
        self._initialize_database()

    def _initialize_database(self) -> None:
        """初始化会话数据库"""
        try:
            # 检查是否需要重新处理文件
            if self._should_rebuild_database():
                logger.info(f"为会话 {self.thread_id} 构建向量数据库")
                
                # 处理会话文件
                content_data, embeddings_data = process_session_files(self.session_files)
                
                # 保存数据库
                if content_data and embeddings_data:
                    save_session_database(
                        content_data, embeddings_data,
                        self.content_path, self.embeddings_path
                    )
            
            # 加载数据库
            self.content_data, self.embeddings_data = load_session_database(
                self.content_path, self.embeddings_path
            )
            
        except Exception as e:
            logger.error(f"初始化会话数据库时出错: {str(e)}")
            self.content_data = []
            self.embeddings_data = []

    def _should_rebuild_database(self) -> bool:
        """判断是否需要重建数据库
        
        Returns:
            是否需要重建
        """
        # 如果数据库文件不存在，需要重建
        if not (self.content_path.exists() and self.embeddings_path.exists()):
            return True
        
        # 检查文件时间戳（简化版本，实际可以更复杂）
        try:
            db_modified = min(
                self.content_path.stat().st_mtime,
                self.embeddings_path.stat().st_mtime
            )
            
            for file_path in self.session_files:
                if os.path.exists(file_path):
                    file_modified = os.path.getmtime(file_path)
                    if file_modified > db_modified:
                        return True
                        
        except Exception as e:
            logger.warning(f"检查文件时间戳失败: {str(e)}")
            return True
            
        return False

    def _hybrid_retrieval(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """混合检索（余弦相似度 + BM25）
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            最相关的文档列表
        """
        if not self.content_data or not self.embeddings_data:
            logger.warning("会话数据库为空，无法执行检索")
            return []
            
        try:
            # 获取查询的嵌入向量
            query_embedding = process_query_embedding(query)
            
            # 提取所有文档的嵌入向量
            doc_embeddings = [item['vector'] for item in self.embeddings_data]
            
            # 1. 余弦相似度计算
            cosine_scores = cosine_similarity(
                [query_embedding], 
                doc_embeddings
            )[0]
            
            # 2. BM25检索
            # 准备文档内容
            texts = [item['content'] for item in self.content_data]
            
            # 对文档进行分词（支持中英文）
            tokenized_texts = []
            for text in texts:
                # 简单的中英文分词
                tokens = []
                current_word = ""
                for char in text:
                    if char.isalnum() or char in "一二三四五六七八九十":
                        current_word += char
                    else:
                        if current_word:
                            tokens.append(current_word)
                            current_word = ""
                        if char.strip():  # 非空白字符
                            tokens.append(char)
                if current_word:
                    tokens.append(current_word)
                tokenized_texts.append(tokens)
            
            bm25 = BM25Okapi(tokenized_texts)
            
            # 对查询进行分词并计算BM25分数
            query_tokens = []
            current_word = ""
            for char in query:
                if char.isalnum() or char in "一二三四五六七八九十":
                    current_word += char
                else:
                    if current_word:
                        query_tokens.append(current_word)
                        current_word = ""
                    if char.strip():
                        query_tokens.append(char)
            if current_word:
                query_tokens.append(current_word)
                
            bm25_scores = np.array(bm25.get_scores(query_tokens))
            
            # 归一化分数
            if len(cosine_scores) > 1:
                cosine_min, cosine_max = cosine_scores.min(), cosine_scores.max()
                if cosine_max > cosine_min:
                    cosine_scores = (cosine_scores - cosine_min) / (cosine_max - cosine_min)
                    
            if len(bm25_scores) > 1:
                bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
                if bm25_max > bm25_min:
                    bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
            
            # 3. 混合排序（加权：余弦相似度60%，BM25 40%）
            final_scores = 0.6 * cosine_scores + 0.4 * bm25_scores
            
            # 获取top_k文档的索引
            top_indices = np.argsort(final_scores)[-top_k:][::-1]
            
            # 返回排序后的文档
            results = []
            for idx in top_indices:
                if final_scores[idx] > 0.1:  # 设置最低相关性阈值
                    results.append({
                        "metadata": self.content_data[idx].get("metadata", {}),
                        "content": self.content_data[idx].get("content", ""),
                        "score": float(final_scores[idx])
                    })
            
            logger.info(f"会话查询 '{query}' 返回了 {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"执行会话混合检索时出错: {str(e)}")
            return []

    def search(self, query: str) -> List[Dict[str, Any]]:
        """执行会话级向量搜索
        
        Args:
            query: 搜索查询字符串
            
        Returns:
            包含元数据和内容的搜索结果列表
        """
        try:
            # 执行混合检索
            results = self._hybrid_retrieval(query)
            
            # 添加来源信息到结果中
            for result in results:
                result['source_type'] = 'session_document'
                result['thread_id'] = self.thread_id
            
            return results
        except Exception as e:
            logger.error(f"会话向量搜索时出错: {str(e)}")
            return []

def create_session_vector_tool(session_files: List[str], thread_id: str):
    """创建会话级向量搜索工具
    
    Args:
        session_files: 会话文档文件路径列表
        thread_id: 会话ID
        
    Returns:
        StructuredTool: 会话级向量搜索工具
    """
    try:
        if not session_files:
            logger.warning("没有提供会话文件")
            return None
            
        # 创建搜索工具实例
        search_tool = SessionVectorSearchTool(session_files, thread_id)
        
        # 检查是否成功加载数据
        if not search_tool.content_data:
            logger.warning("会话文档处理失败，无法创建搜索工具")
            return None
        
        # 封装为StructuredTool
        return StructuredTool.from_function(
            func=search_tool.search,
            name="session_vector_search",
            description=f"【优先使用】搜索用户上传的文档内容，这些文档与用户问题强相关。基于向量相似度和BM25混合检索。当前会话包含 {len(session_files)} 个文档，共 {len(search_tool.content_data)} 个文档块。用户上传文档说明这些内容对回答问题至关重要，必须优先检索这些文档！",
            args_schema=SessionVectorSearchInput
        )
        
    except Exception as e:
        logger.error(f"创建会话级向量搜索工具失败: {str(e)}")
        return None

# 测试代码
if __name__ == "__main__":
    # 测试功能
    test_files = ["test_document.txt"]  # 替换为实际测试文件
    test_thread_id = "test_thread_123"
    
    tool = create_session_vector_tool(test_files, test_thread_id)
    if tool:
        results = tool.invoke({"query": "测试查询"})
        print(json.dumps(results, ensure_ascii=False, indent=2)) 