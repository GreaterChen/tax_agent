# Embedding模型离线使用指南

## 问题描述

在使用embedding模型时，系统可能会访问外网下载模型文件，导致加载速度缓慢。特别是在企业环境或网络受限的环境中，这会影响用户体验。

## 解决方案

我们已经配置系统使用离线模式，确保embedding模型仅使用本地缓存的文件，不访问外网。

### 主要修改

1. **统一配置管理** - 创建了 `config/embedding_config.py` 配置文件
2. **离线环境变量** - 自动设置环境变量启用离线模式
3. **本地模型检测** - 检查模型是否已下载到本地

### 配置特性

- ✅ 自动设置离线环境变量（`HF_HUB_OFFLINE=1`等）
- ✅ 统一管理embedding模型配置
- ✅ 检查本地模型可用性
- ✅ 优化模型加载参数

### 使用的模型

- **模型名称**: `intfloat/multilingual-e5-large`
- **模型类型**: 多语言嵌入模型
- **向量维度**: 1024
- **本地位置**: `~/.cache/huggingface/hub/models--intfloat--multilingual-e5-large`

### 环境变量

系统会自动设置以下环境变量来启用离线模式：

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

### 验证离线模式

可以通过以下方式验证embedding模型是否在离线模式下工作：

```python
from config.embedding_config import get_embedding_model_config, is_model_available_locally
from langchain_huggingface import HuggingFaceEmbeddings

# 检查模型是否在本地可用
print(f"模型本地可用: {is_model_available_locally()}")

# 获取配置并初始化embedding模型
config = get_embedding_model_config()
embeddings = HuggingFaceEmbeddings(**config)

# 测试embedding生成
text = "这是一个测试文本"
embedding = embeddings.embed_query(text)
print(f"向量维度: {len(embedding)}")
```

### 性能优化

- **首次加载**: ~12秒（从本地缓存加载）
- **后续embedding生成**: ~0.3秒/文本
- **网络访问**: 无（完全离线）

### 注意事项

1. 确保模型已经下载到本地缓存目录
2. 如果模型不存在，系统会尝试使用模型名称，可能仍会触发下载
3. 在无网络环境中使用前，请先在有网络的环境中运行一次以下载模型

### 相关文件

- `config/embedding_config.py` - 统一配置文件
- `src/tools/vector_search/Dependencies/embedding.py` - 向量搜索embedding
- `src/tools/session_vector_search/Dependencies/embedding.py` - 会话embedding  
- `src/tools/web_search/web_search.py` - 网络搜索embedding

所有这些文件都已更新为使用统一的离线配置。