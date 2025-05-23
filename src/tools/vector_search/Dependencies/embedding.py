"""嵌入处理模块

此模块负责文档的嵌入向量生成，支持处理.txt、.doc、.pdf和.rtf文件。
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import shutil
import uuid
import tempfile
from tqdm import tqdm

# 导入文档处理相关库
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredRTFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("embedding")

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

def process_query_embedding(query: str) -> List[float]:
    """处理查询的嵌入向量
    
    Args:
        query: 查询字符串
        
    Returns:
        嵌入向量
    """
    try:
        # 使用嵌入模型生成向量
        query_embedding = embeddings_model.embed_query(query)
        return query_embedding
    except Exception as e:
        logger.error(f"生成查询嵌入向量时出错: {str(e)}")
        # 返回空向量作为后备
        return [0.0] * 1024  # 假设向量维度为1024

def load_document(file_path: str) -> List[Document]:
    """根据文件类型加载文档
    
    Args:
        file_path: 文件路径
        
    Returns:
        文档对象列表
    """
    try:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.doc' or file_ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_ext == '.rtf':
            loader = UnstructuredRTFLoader(file_path)
        else:
            logger.warning(f"不支持的文件类型: {file_ext}")
            return []
            
        documents = loader.load()
        logger.info(f"已加载文档 {file_path}，共 {len(documents)} 个部分")
        return documents
    except Exception as e:
        logger.error(f"加载文档 {file_path} 时出错: {str(e)}")
        return []

def extract_metadata(file_path: str, doc_id: str) -> Dict[str, Any]:
    """从文件中提取元数据
    
    Args:
        file_path: 文件路径
        doc_id: 文档ID
        
    Returns:
        元数据字典
    """
    file_name = Path(file_path).name
    file_size = Path(file_path).stat().st_size
    file_ext = Path(file_path).suffix.lower()
    
    # 根据文件名生成标题
    title = Path(file_path).stem
    
    # 默认元数据
    metadata = {
        "Title": title,
        "Credibility": "5",  # 默认可信度
        "ID": doc_id,
        "Copyright": "",
        "Location": f"Body>Section,1,Document>Chapter,1,{file_name}"
    }
    
    return metadata

def process_documents(input_dir: str = None, output_dir: str = None, batch_size: int = 32) -> None:
    """处理文档并生成嵌入向量
    
    Args:
        input_dir: 输入文档目录
        output_dir: 输出目录
        batch_size: 批处理大小
    """
    # 设置默认目录
    if input_dir is None:
        input_dir = str(Path(__file__).parent / "input")
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "database")
    
    # 确保目录存在
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    # 初始化存储列表
    content_data = []
    embeddings_data = []
    
    # 加载现有数据（如果存在）
    content_file = output_path / "content.json"
    embeddings_file = output_path / "embeddings.json"
    
    if content_file.exists():
        try:
            with open(content_file, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            logger.info(f"已加载现有内容数据，共 {len(content_data)} 条记录")
        except Exception as e:
            logger.error(f"加载内容数据时出错: {str(e)}")
            content_data = []
    
    if embeddings_file.exists():
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            logger.info(f"已加载现有嵌入向量，共 {len(embeddings_data)} 条记录")
        except Exception as e:
            logger.error(f"加载嵌入向量时出错: {str(e)}")
            embeddings_data = []
    
    # 收集所有支持的文件
    supported_extensions = ['.txt', '.pdf', '.doc', '.docx', '.rtf']
    files_to_process = []
    
    for file in input_path.glob('**/*'):
        if file.is_file() and file.suffix.lower() in supported_extensions:
            files_to_process.append(file)
    
    logger.info(f"找到 {len(files_to_process)} 个待处理文件")
    
    # 处理每个文件
    for file_path in tqdm(files_to_process, desc="处理文件"):
        try:
            # 生成文档ID
            doc_id = str(uuid.uuid4())
            
            # 加载文档
            documents = load_document(str(file_path))
            
            if not documents:
                continue
                
            # 提取元数据
            metadata = extract_metadata(str(file_path), doc_id)
            
            # 分割文档
            chunks = text_splitter.split_documents(documents)
            
            logger.info(f"文档 {file_path.name} 分割为 {len(chunks)} 个块")
            
            # 批量处理文档块
            for i in tqdm(range(0, len(chunks), batch_size), desc="处理文档块"):
                batch = chunks[i:i + batch_size]
                
                # 批量生成嵌入向量
                texts = [chunk.page_content for chunk in batch]
                embedding_vectors = embeddings_model.embed_documents(texts)
                
                # 处理每个块的结果
                for j, (chunk, embedding_vector) in enumerate(zip(batch, embedding_vectors)):
                    chunk_idx = i + j
                    
                    # 更新块的位置信息
                    chunk_metadata = metadata.copy()
                    chunk_metadata["Location"] = f"{metadata['Location']}>Chunk,{chunk_idx+1}"
                    
                    # 添加到内容数据
                    content_item = {
                        "metadata": chunk_metadata,
                        "content": chunk.page_content
                    }
                    content_data.append(content_item)
                    
                    # 添加到嵌入数据
                    embedding_item = {
                        "id": f"{doc_id}_{chunk_idx}",
                        "doc_id": doc_id,
                        "chunk_id": chunk_idx,
                        "vector": embedding_vector
                    }
                    embeddings_data.append(embedding_item)
            
            logger.info(f"成功处理文档 {file_path.name}")
            
        except Exception as e:
            logger.error(f"处理文档 {file_path} 时出错: {str(e)}")
    
    # 保存数据
    try:
        with open(content_file, 'w', encoding='utf-8') as f:
            json.dump(content_data, f, ensure_ascii=False, indent=2)
        
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存 {len(content_data)} 条内容记录和 {len(embeddings_data)} 条嵌入向量")
    except Exception as e:
        logger.error(f"保存数据时出错: {str(e)}")

# 测试代码
if __name__ == "__main__":
    # 处理文档
    process_documents()
    
    # 测试查询嵌入
    query_embedding = process_query_embedding("税务相关法规")
    print(f"查询嵌入向量维度: {len(query_embedding)}") 