"""会话级嵌入处理模块

此模块负责会话文档的嵌入向量生成，支持处理多种文本文件格式。
专门为会话级文档检索而设计，独立于其他工具模块。
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import hashlib

# 导入文档处理相关库
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredRTFLoader
)
from langchain_core.documents import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("session_embedding")

# 初始化嵌入模型
embeddings_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"  # 使用多语言嵌入模型
)

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
)

def process_query_embedding(query: str) -> np.ndarray:
    """处理查询的嵌入向量
    
    Args:
        query: 查询字符串
        
    Returns:
        嵌入向量（numpy数组）
    """
    try:
        # 使用嵌入模型生成向量
        query_embedding = embeddings_model.embed_query(query)
        return np.array(query_embedding)
    except Exception as e:
        logger.error(f"生成查询嵌入向量时出错: {str(e)}")
        # 返回空向量作为后备
        return np.zeros(1024)  # 假设向量维度为1024

def process_text_embedding(text: str) -> np.ndarray:
    """处理文本的嵌入向量
    
    Args:
        text: 文本字符串
        
    Returns:
        嵌入向量（numpy数组）
    """
    try:
        # 使用嵌入模型生成向量
        text_embedding = embeddings_model.embed_query(text)
        return np.array(text_embedding)
    except Exception as e:
        logger.error(f"生成文本嵌入向量时出错: {str(e)}")
        # 返回空向量作为后备
        return np.zeros(1024)

def split_text(content: str) -> List[str]:
    """分割文本为块
    
    Args:
        content: 文本内容
        
    Returns:
        文本块列表
    """
    try:
        chunks = text_splitter.split_text(content)
        return chunks
    except Exception as e:
        logger.error(f"分割文本时出错: {str(e)}")
        return [content]  # 如果分割失败，返回原文本

def read_file_content(file_path: str) -> str:
    """读取文件内容，支持多种文档格式
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件内容字符串
    """
    try:
        file_ext = Path(file_path).suffix.lower()
        
        # 根据文件扩展名选择合适的加载器
        if file_ext == '.txt':
            # 文本文件，尝试不同编码
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"无法读取文本文件 {file_path}: {str(e)}")
                    return ""
                    
        elif file_ext == '.pdf':
            # PDF文件
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
            
        elif file_ext in ['.doc', '.docx']:
            # Word文档
            loader = UnstructuredWordDocumentLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
            
        elif file_ext == '.rtf':
            # RTF文档
            loader = UnstructuredRTFLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
            
        else:
            # 未知格式，尝试作为文本文件读取
            logger.warning(f"未知文件格式 {file_ext}，尝试作为文本文件读取: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"无法读取文件 {file_path}: {str(e)}")
                    return ""
                    
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {str(e)}")
        return ""

def generate_chunk_metadata(file_path: str, chunk_index: int) -> Dict[str, Any]:
    """生成文档块的元数据
    
    Args:
        file_path: 文件路径
        chunk_index: 块索引
        
    Returns:
        元数据字典
    """
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    
    metadata = {
        "source": file_path,
        "chunk_index": chunk_index,
        "file_name": file_name,
        "file_size": file_size,
        "chunk_id": hashlib.md5(f"{file_path}_{chunk_index}".encode()).hexdigest()
    }
    
    return metadata

def process_session_files(file_paths: List[str]) -> tuple[List[Dict], List[Dict]]:
    """处理会话文件列表，生成内容和嵌入数据
    
    Args:
        file_paths: 文件路径列表
        
    Returns:
        (content_data, embeddings_data) 元组
    """
    content_data = []
    embeddings_data = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            continue
            
        # 读取文件内容
        content = read_file_content(file_path)
        if not content:
            continue
            
        # 分割文本
        chunks = split_text(content)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 10:  # 忽略太短的chunk
                continue
                
            # 生成元数据
            metadata = generate_chunk_metadata(file_path, i)
            chunk_id = metadata["chunk_id"]
            
            # 生成embedding
            embedding = process_text_embedding(chunk)
            
            # 添加到内容数据
            content_data.append({
                "id": chunk_id,
                "content": chunk,
                "metadata": metadata
            })
            
            # 添加到embedding数据
            embeddings_data.append({
                "id": chunk_id,
                "vector": embedding.tolist()
            })
    
    logger.info(f"处理完成，生成了 {len(content_data)} 个文档块")
    return content_data, embeddings_data

def save_session_database(content_data: List[Dict], embeddings_data: List[Dict], 
                         content_path: Path, embeddings_path: Path) -> bool:
    """保存会话数据库
    
    Args:
        content_data: 内容数据
        embeddings_data: 嵌入数据
        content_path: 内容文件路径
        embeddings_path: 嵌入文件路径
        
    Returns:
        是否保存成功
    """
    try:
        # 保存内容数据
        with open(content_path, 'w', encoding='utf-8') as f:
            json.dump(content_data, f, ensure_ascii=False, indent=2)
            
        # 保存嵌入数据
        with open(embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"会话数据库已保存到 {content_path.parent}")
        return True
    except Exception as e:
        logger.error(f"保存会话数据库时出错: {str(e)}")
        return False

def load_session_database(content_path: Path, embeddings_path: Path) -> tuple[List[Dict], List[Dict]]:
    """加载会话数据库
    
    Args:
        content_path: 内容文件路径
        embeddings_path: 嵌入文件路径
        
    Returns:
        (content_data, embeddings_data) 元组
    """
    content_data = []
    embeddings_data = []
    
    try:
        # 加载内容数据
        if content_path.exists():
            with open(content_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            logger.info(f"已加载 {len(content_data)} 条内容记录")
        
        # 加载嵌入数据
        if embeddings_path.exists():
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            logger.info(f"已加载 {len(embeddings_data)} 条嵌入向量")
                
    except Exception as e:
        logger.error(f"加载会话数据库时出错: {str(e)}")
        
    return content_data, embeddings_data 