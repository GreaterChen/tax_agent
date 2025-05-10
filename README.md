# 税务问答系统 (精简版)

基于LangChain、LangGraph和FastAPI构建的轻量级税务领域问答系统，专为低配置服务器(2核2G)设计。

## 特性

- 🤖 基于智谱AI GLM-4模型的问答功能
- 🧮 支持数学计算功能
- 🔍 通过Google搜索API实现互联网搜索
- 🧠 支持对话记忆功能
- 🚀 轻量级设计，适合低配置服务器部署

## 系统架构

本系统采用轻量级设计，将计算密集型任务委托给外部API处理，主要包括以下组件：

1. **FastAPI服务**: 提供HTTP接口，处理用户请求
2. **TaxAgent**: 核心问答代理，基于LangGraph实现的交互式代理
3. **工具集**:
   - 计算器工具: 处理数学计算需求
   - Web搜索工具: 调用Google搜索API获取互联网信息

## 技术栈

- 🐍 Python 3.9+
- ⚡ FastAPI: 高性能API框架
- 🔗 LangChain: 大模型应用开发框架
- 📊 LangGraph: 多代理交互框架

## 快速开始

### 环境准备

1. 克隆项目并进入项目目录
```bash
git clone https://github.com/yourusername/tax-agent-lite.git
cd tax-agent-lite
```

2. 创建并激活虚拟环境
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

### 环境变量配置

创建.env文件，添加以下内容:
```
# 必须配置
ZHIPUAI_API_KEY=你的智谱AI API密钥

# 可选配置(使用默认值时可不配置)
GOOGLE_API_KEY=你的Google API密钥
GOOGLE_CSE_ID=你的Google自定义搜索引擎ID
```

### 启动服务

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

访问 http://your_server_ip:8000/docs 查看API文档

## API使用

### 问答接口

**端点**: `/query`

**方法**: POST

**请求体**:
```json
{
  "text": "增值税的税率是多少?",
  "thread_id": "optional_thread_id"
}
```

**响应**:
```json
{
  "answers": ["增值税税率分为三档：13%（基本税率）、9%（低税率）和6%（低税率）..."],
  "thread_id": "thread_12345"
}
```

## 部署说明

### Docker部署

1. 构建Docker镜像
```bash
docker build -t tax-agent-lite .
```

2. 运行容器
```bash
docker run -d -p 8000:8000 --env-file .env --name tax-agent-lite tax-agent-lite
```

### 服务器要求

- CPU: 2核
- 内存: 2GB
- 操作系统: Linux (推荐)
- 网络: 可访问外部API的网络环境

## 注意事项

1. 智谱AI API密钥需要自行申请
2. Google搜索API有免费额度限制，超出需付费
3. 系统依赖外部API，需确保网络环境稳定
