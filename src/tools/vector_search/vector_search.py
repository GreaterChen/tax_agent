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

class VectorSearchTool:
    def __init__(self):
        """初始化向量搜索工具"""
        self.db_path = Path(__file__).parent / "Dependencies" / "database"
        self.embeddings_path = self.db_path / "embeddings.json"
        self.content_path = self.db_path / "content.json"
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        # 加载数据库内容
        self._load_database()

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

    def _hybrid_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """混合检索（余弦相似度 + BM25）
        
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
            
            # 对文档进行分词
            tokenized_texts = [text.split() for text in texts]
            bm25 = BM25Okapi(tokenized_texts)
            
            # 对查询进行分词并计算BM25分数
            tokenized_query = query.split()
            bm25_scores = np.array(bm25.get_scores(tokenized_query))
            
            # 归一化分数
            if len(cosine_scores) > 1:
                cosine_scores = (cosine_scores - cosine_scores.min()) / (cosine_scores.max() - cosine_scores.min() + 1e-8)
            if len(bm25_scores) > 1:
                bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
            
            # 3. 混合排序（加权）
            final_scores = 0.6 * cosine_scores + 0.4 * bm25_scores
            
            # 获取top_k文档的索引
            top_indices = np.argsort(final_scores)[-top_k:][::-1]
            
            # 返回排序后的文档
            results = []
            for idx in top_indices:
                results.append({
                    "metadata": self.content_data[idx].get("metadata", {}),
                    "content": self.content_data[idx].get("content", "")
                })
            
            logger.info(f"查询 '{query}' 返回了 {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"执行混合检索时出错: {str(e)}")
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

# 测试代码
if __name__ == "__main__":
    results = vector_search("something about Profession")
    print(json.dumps(results, ensure_ascii=False, indent=2)) 