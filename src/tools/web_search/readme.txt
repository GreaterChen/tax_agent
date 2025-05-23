# Web搜索工具 (Web Search Tool)

## 简介

Web搜索工具是一个基于多种搜索引擎API的高级网络信息检索系统。该工具采用LangChain框架，集成了智能查询改写、多语言处理、RAG检索增强和结果智能分析等功能，能够提供高质量的搜索结果。

## 功能特点

1. **智能查询处理**
   - 查询改写与多组检索词生成
   - 多语言支持（中文简体、香港繁体、英文）
   - 基于spaCy的自然语言处理

2. **高级搜索功能**
   - 多源搜索引擎集成
   - 并行内容获取
   - PDF和HTML文档智能处理
   - 缓存机制优化性能

3. **智能结果处理**
   - 基于向量检索的相关性排序
   - 混合检索策略（BM25 + 向量相似度）
   - 智能结果分析与汇总
   - 自动元数据提取

4. **专业领域优化**
   - 预配置专业网站源（如ACCA、HKICPA等）
   - 特定领域内容识别
   - 结果可信度评估

## 技术栈

- LangChain框架
- Tongyi（通义）大语言模型
- HuggingFace多语言嵌入模型
- spaCy自然语言处理
- FAISS向量数据库
- BeautifulSoup网页解析
- 多种文档处理库（PyPDF2等）

## 目录结构

```
web_search/
├── web_search.py         # 主搜索功能模块
├── web_search_mini.py    # 精简版搜索模块
├── requirements.txt      # 项目依赖
└── readme.txt           # 项目说明文档
```

## 环境要求

- Python 3.8+
- 相关依赖包（见requirements.txt）
- 必要的API密钥配置

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

需要在环境变量或者.env中配置以下环境变量：
- DASHSCOPE_API_KEY（通义千问API密钥）

### 3. 基本使用

```python
from web_search import AdvancedWebSearchTool

# 初始化搜索工具
search_tool = AdvancedWebSearchTool()

# 执行搜索
results = search_tool.search("最新税收政策变化")
print(results)
```