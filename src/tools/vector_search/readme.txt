# 向量搜索工具 (Vector Search Tool)

## 简介

向量搜索工具是一个基于混合检索技术（BM25 + 余弦相似度）的文档搜索系统，能够处理并检索多种格式的文档内容。该工具支持中英文混合检索，并针对税务领域文档进行了优化。

## 功能特点

1. **多格式文档支持**：处理 `.txt`、`.doc`、`.pdf` 和 `.rtf` 格式的文档
2. **混合检索算法**：结合 BM25 和余弦相似度进行混合排序，提高检索准确性
3. **多语言支持**：使用多语言嵌入模型，支持中英文文档检索
4. **元数据管理**：为每个文档块生成丰富的元数据信息
5. **高效向量存储**：使用JSON格式存储文档内容和向量数据

## 目录结构

```
vector_search/
├── vector_search.py       # 主搜索功能模块
├── requirements.txt       # 项目依赖
├── readme.txt            # 项目说明文档
└── Dependencies/
    ├── embedding.py      # 嵌入向量生成模块
    ├── input/            # 文档输入目录
    └── database/         # 向量数据库存储目录
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 添加文档

将需要处理的文档（`.txt`、`.doc`、`.pdf`、`.rtf`）放入 `Dependencies/input/` 目录中。

### 3. 处理文档生成向量

```python
from Dependencies.embedding import process_documents

# 处理文档并生成向量
process_documents()
```

### 4. 执行向量搜索

```python
from vector_search import vector_search

# 执行搜索
results = vector_search("税务相关法规")
print(results)
```

## 搜索结果格式

搜索结果以列表形式返回，每个结果包含元数据和内容：

```json
[
  {
    "metadata": {
      "Title": "文档标题",
      "Credibility": "可信度评分",
      "ID": "唯一标识符",
      "Copyright": "版权信息",
      "Location": "文档位置信息"
    },
    "content": "文档内容片段"
  },
  ...
]
```

## 注意事项

1. 首次运行时需要下载嵌入模型，请确保网络连接正常
2. 处理大型文档可能需要较多内存和计算资源
3. 建议定期清理 `Dependencies/input/` 目录中已处理的文档，避免重复处理 