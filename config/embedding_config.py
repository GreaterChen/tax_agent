"""Embedding模型配置模块

统一管理embedding模型的配置，确保仅使用本地文件，不访问外网
"""

import os
from typing import Dict, Any

# 设置环境变量以启用离线模式
def set_offline_mode():
    """设置环境变量以启用离线模式，禁止访问外网"""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

# Embedding模型配置
EMBEDDING_MODEL_CONFIG = {
    "model_name": "intfloat/multilingual-e5-large",
    "model_kwargs": {
        'device': 'cpu',
        'trust_remote_code': False
    },
    "encode_kwargs": {
        'normalize_embeddings': True
    },
    # 使用默认缓存位置
    "cache_folder": None
}

# 文本分割器配置
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "length_function": len,
    "separators": ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
}

def get_embedding_model_config() -> Dict[str, Any]:
    """获取embedding模型配置
    
    Returns:
        embedding模型配置字典
    """
    # 自动设置离线模式
    set_offline_mode()
    return EMBEDDING_MODEL_CONFIG.copy()

def get_text_splitter_config() -> Dict[str, Any]:
    """获取文本分割器配置
    
    Returns:
        文本分割器配置字典
    """
    return TEXT_SPLITTER_CONFIG.copy()

def is_model_available_locally(model_name: str = None) -> bool:
    """检查模型是否在本地可用
    
    Args:
        model_name: 模型名称，默认使用配置中的模型
        
    Returns:
        bool: 模型是否在本地可用
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL_CONFIG["model_name"]
    
    # 检查缓存目录
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = f"models--{model_name.replace('/', '--')}"
    model_path = os.path.join(cache_dir, model_dir)
    
    return os.path.exists(model_path)

def get_local_model_path(model_name: str = None) -> str:
    """获取本地模型路径
    
    Args:
        model_name: 模型名称，默认使用配置中的模型
        
    Returns:
        本地模型路径，如果不存在返回模型名称
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL_CONFIG["model_name"]
    
    # 检查缓存目录
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = f"models--{model_name.replace('/', '--')}"
    model_path = os.path.join(cache_dir, model_dir)
    
    if os.path.exists(model_path):
        # 寻找snapshots目录下的实际模型路径
        snapshots_path = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_path):
            # 返回第一个找到的snapshot路径
            for snapshot_dir in os.listdir(snapshots_path):
                snapshot_path = os.path.join(snapshots_path, snapshot_dir)
                if os.path.isdir(snapshot_path):
                    return snapshot_path
    
    # 如果找不到本地路径，返回模型名称（将由HuggingFace缓存处理）
    return model_name

if __name__ == "__main__":
    # 测试配置
    print("Embedding模型配置:")
    print(get_embedding_model_config())
    print("\n文本分割器配置:")
    print(get_text_splitter_config())
    print(f"\n模型本地可用性: {is_model_available_locally()}")
    print(f"本地模型路径: {get_local_model_path()}") 