"""向量搜索工具

此工具实现基于向量的搜索功能，结合BM25和余弦相似度进行混合检索。
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
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入依赖模块
from Dependencies.embedding import process_query_embedding

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vector_search")

# 输入模型
class VectorSearchInput(BaseModel):
    """向量搜索输入"""
    query: str = Field(..., description="搜索查询字符串")

class VectorSearchTool:
    def __init__(self, lightweight_mode: bool = True, cache_size: int = 100):
        """初始化向量搜索工具
        
        Args:
            lightweight_mode: 是否启用轻量级模式（跳过ColBERT计算）
            cache_size: 查询缓存大小
        """
        self.db_path = Path(__file__).parent / "Dependencies" / "database"
        self.embeddings_path = self.db_path / "embeddings.json"
        self.content_path = self.db_path / "content.json"
        self.lightweight_mode = lightweight_mode
        
        # 查询缓存
        from functools import lru_cache
        self.query_cache = {}
        self.cache_size = cache_size
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        # 加载数据库内容
        self._load_database()
        
        # 预计算BM25（避免每次查询重新计算）
        self._precompute_bm25()

    def _load_database(self) -> None:
        """加载数据库内容和嵌入向量"""
        try:
            # 加载内容数据
            if self.content_path.exists():
                with open(self.content_path, 'r', encoding='utf-8') as f:
                    self.content_data = json.load(f)
                logger.info(f"已加载 {len(self.content_data)} 条内容记录")
            else:
                logger.warning(f"内容数据文件不存在: {self.content_path}")
                self.content_data = []
            
            # 加载嵌入向量
            if self.embeddings_path.exists():
                with open(self.embeddings_path, 'r', encoding='utf-8') as f:
                    self.embeddings_data = json.load(f)
                logger.info(f"已加载 {len(self.embeddings_data)} 条嵌入向量")
            else:
                logger.warning(f"嵌入向量文件不存在: {self.embeddings_path}")
                self.embeddings_data = []
                
        except Exception as e:
            logger.error(f"加载数据库时出错: {str(e)}")
            self.content_data = []
            self.embeddings_data = []

    def _precompute_bm25(self) -> None:
        """预计算BM25索引，避免检索时重复计算"""
        try:
            if self.content_data:
                texts = [item['content'] for item in self.content_data]
                self.tokenized_texts = [text.split() for text in texts]
                self.bm25 = BM25Okapi(self.tokenized_texts)
                logger.info("BM25索引预计算完成")
            else:
                self.bm25 = None
        except Exception as e:
            logger.error(f"预计算BM25时出错: {str(e)}")
            self.bm25 = None

    def _get_cached_query_embedding(self, query: str) -> Dict[str, Any]:
        """获取缓存的查询嵌入向量"""
        if query in self.query_cache:
            logger.info(f"使用缓存的查询嵌入: {query}")
            return self.query_cache[query]
        
        # 计算新的嵌入向量
        query_embeddings = process_query_embedding(query)
        
        # 管理缓存大小
        if len(self.query_cache) >= self.cache_size:
            # 删除最旧的缓存项
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[query] = query_embeddings
        return query_embeddings

    def _lightweight_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """轻量级检索（仅Dense + Sparse + BM25，跳过ColBERT）
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            最相关的文档列表
        """
        if not self.content_data or not self.embeddings_data:
            logger.warning("数据库为空，无法执行检索")
            return []
            
        try:
            # 获取缓存的查询嵌入向量
            query_embeddings = self._get_cached_query_embedding(query)
            
            # 1. Dense向量余弦相似度计算（使用numpy优化）
            doc_dense_embeddings = np.array([
                item.get('dense_vector', item.get('vector', [0.0] * 1024))
                for item in self.embeddings_data
            ])
            
            if query_embeddings['dense'] and len(doc_dense_embeddings) > 0:
                query_vec = np.array(query_embeddings['dense']).reshape(1, -1)
                # 使用更快的numpy计算替代sklearn
                dot_product = np.dot(query_vec, doc_dense_embeddings.T)[0]
                query_norm = np.linalg.norm(query_vec)
                doc_norms = np.linalg.norm(doc_dense_embeddings, axis=1)
                cosine_scores = dot_product / (query_norm * doc_norms + 1e-8)
            else:
                cosine_scores = np.zeros(len(self.embeddings_data))
            
            # 2. Sparse向量检索（优化循环）
            sparse_scores = np.zeros(len(self.embeddings_data))
            if query_embeddings['sparse']:
                query_tokens = set(query_embeddings['sparse'].keys())
                for i, item in enumerate(self.embeddings_data):
                    doc_sparse = item.get('sparse_vector', {})
                    if doc_sparse:
                        # 只计算有交集的token
                        common_tokens = query_tokens & set(doc_sparse.keys())
                        if common_tokens:
                            score = sum(
                                query_embeddings['sparse'][token] * doc_sparse[token]
                                for token in common_tokens
                            )
                            sparse_scores[i] = score
            
            # 3. 预计算的BM25
            bm25_scores = np.zeros(len(self.embeddings_data))
            if self.bm25:
                tokenized_query = query.split()
                bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
            
            # 快速归一化
            def fast_normalize(scores):
                min_score, max_score = scores.min(), scores.max()
                if max_score > min_score:
                    return (scores - min_score) / (max_score - min_score + 1e-8)
                return scores
            
            cosine_scores = fast_normalize(cosine_scores)
            sparse_scores = fast_normalize(sparse_scores)
            bm25_scores = fast_normalize(bm25_scores)
            
            # 轻量级加权（去除ColBERT）
            final_scores = (
                0.5 * cosine_scores +      # Dense检索
                0.3 * sparse_scores +      # Sparse检索
                0.2 * bm25_scores          # BM25
            )
            
            # 使用numpy的argpartition优化top-k选择
            if len(final_scores) <= top_k:
                top_indices = np.argsort(final_scores)[::-1]
            else:
                top_indices = np.argpartition(final_scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]
            
            # 返回结果
            results = []
            for idx in top_indices:
                results.append({
                    "metadata": self.content_data[idx].get("metadata", {}),
                    "content": self.content_data[idx].get("content", ""),
                    "scores": {
                        "final": float(final_scores[idx]),
                        "dense": float(cosine_scores[idx]),
                        "sparse": float(sparse_scores[idx]),
                        "bm25": float(bm25_scores[idx]),
                        "mode": "lightweight"
                    }
                })
            
            logger.info(f"轻量级查询 '{query}' 返回了 {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"执行轻量级检索时出错: {str(e)}")
            return []

    def _hybrid_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """混合检索（余弦相似度 + BM25 + BGE-M3多向量）
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            最相关的文档列表
        """
        if not self.content_data or not self.embeddings_data:
            logger.warning("数据库为空，无法执行检索")
            return []
            
        try:
            # 获取查询的多种嵌入向量
            query_embeddings = process_query_embedding(query)
            
            # 1. Dense向量余弦相似度计算
            doc_dense_embeddings = [
                item.get('dense_vector', item.get('vector', []))  # 兼容旧格式
                for item in self.embeddings_data
            ]
            
            if query_embeddings['dense'] and doc_dense_embeddings:
                cosine_scores = cosine_similarity(
                    [query_embeddings['dense']], 
                    doc_dense_embeddings
                )[0]
            else:
                cosine_scores = np.zeros(len(self.embeddings_data))
            
            # 2. Sparse向量检索（类似BM25但基于学习到的权重）
            sparse_scores = np.zeros(len(self.embeddings_data))
            if query_embeddings['sparse']:
                for i, item in enumerate(self.embeddings_data):
                    doc_sparse = item.get('sparse_vector', {})
                    if doc_sparse:
                        # 计算稀疏向量匹配分数
                        score = 0.0
                        for token, weight in query_embeddings['sparse'].items():
                            if token in doc_sparse:
                                score += weight * doc_sparse[token]
                        sparse_scores[i] = score
            
            # 3. ColBERT多向量交互分数
            colbert_scores = np.zeros(len(self.embeddings_data))
            if query_embeddings['colbert'] is not None:
                for i, item in enumerate(self.embeddings_data):
                    doc_colbert = item.get('colbert_vector')
                    if doc_colbert is not None:
                        # 使用BGE-M3的colbert_score方法
                        try:
                            from Dependencies.embedding import embeddings_model
                            colbert_scores[i] = embeddings_model.colbert_score(
                                query_embeddings['colbert'], 
                                doc_colbert
                            )
                        except:
                            colbert_scores[i] = 0.0
            
            # 4. 传统BM25作为补充
            texts = [item['content'] for item in self.content_data]
            tokenized_texts = [text.split() for text in texts]
            bm25 = BM25Okapi(tokenized_texts)
            tokenized_query = query.split()
            bm25_scores = np.array(bm25.get_scores(tokenized_query))
            
            # 归一化所有分数
            def normalize_scores(scores):
                if len(scores) > 1 and scores.max() > scores.min():
                    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                return scores
            
            cosine_scores = normalize_scores(cosine_scores)
            sparse_scores = normalize_scores(sparse_scores)
            colbert_scores = normalize_scores(colbert_scores)
            bm25_scores = normalize_scores(bm25_scores)
            
            # 5. 加权混合排序（BGE-M3官方推荐权重）
            final_scores = (
                0.4 * cosine_scores +      # Dense检索
                0.2 * sparse_scores +      # Sparse检索
                0.4 * colbert_scores +     # ColBERT检索
                0.1 * bm25_scores          # 传统BM25补充
            ) / 1.1  # 归一化权重总和
            
            # 获取top_k文档的索引
            top_indices = np.argsort(final_scores)[-top_k:][::-1]
            
            # 返回排序后的文档
            results = []
            for idx in top_indices:
                results.append({
                    "metadata": self.content_data[idx].get("metadata", {}),
                    "content": self.content_data[idx].get("content", ""),
                    "scores": {
                        "final": float(final_scores[idx]),
                        "dense": float(cosine_scores[idx]),
                        "sparse": float(sparse_scores[idx]),
                        "colbert": float(colbert_scores[idx]),
                        "bm25": float(bm25_scores[idx])
                    }
                })
            
            logger.info(f"查询 '{query}' 返回了 {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"执行混合检索时出错: {str(e)}")
            return []

    def search(self, query: str) -> List[Dict[str, Any]]:
        """执行向量搜索
        
        Args:
            query: 搜索查询字符串
            
        Returns:
            包含元数据和内容的搜索结果列表
        """
        try:
            # 根据模式选择检索方法
            if self.lightweight_mode:
                results = self._lightweight_retrieval(query)
            else:
                results = self._hybrid_retrieval(query)
            
            return results
        except Exception as e:
            logger.error(f"向量搜索时出错: {str(e)}")
            return []

def vector_search(query: str) -> List[Dict[str, Any]]:
    """执行向量搜索
    
    Args:
        query: 搜索查询字符串
        
    Returns:
        包含元数据和内容的搜索结果列表
    """
    try:
        # 初始化搜索工具
        search_tool = VectorSearchTool()
        
        # 执行混合检索
        results = search_tool._hybrid_retrieval(query)
        
        # 返回结果
        return results
    except Exception as e:
        logger.error(f"向量搜索时出错: {str(e)}")
        return []

# 创建工具实例（默认轻量级模式，适合CPU部署）
vector_search_instance = VectorSearchTool(lightweight_mode=True, cache_size=100)

# 封装为StructuredTool
vector_search_tool = StructuredTool.from_function(
    func=vector_search_instance.search,
    name="vector_search",
    description="基于向量的搜索功能，支持轻量级CPU模式和完整混合检索模式",
    args_schema=VectorSearchInput
)

# 测试代码
if __name__ == "__main__":
    results = vector_search("something about Profession")
    print(json.dumps(results, ensure_ascii=False, indent=2)) 