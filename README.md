# 税务问答Agent系统

基于LangChain和FastAPI开发的智能税务问答系统，支持税务计算和新闻查询。

## 功能特点

- 基于智谱AI GLM-4-flash模型的智能问答
- 支持数学计算(四则运算、括号优先级)
- 自动爬取和查询税务新闻
- RESTful API接口
- 定时更新新闻数据
- 支持多轮对话记忆功能
- 使用LangSmith进行对话追踪和监控

## 系统架构

- 后端框架: FastAPI 0.110.0
- LLM框架: 
  - LangChain 0.1.12
  - LangGraph 0.0.26
- 大语言模型: 智谱AI GLM-4-flash
- 数据库: MySQL (SQLAlchemy 2.0.28)
- 定时任务: APScheduler 3.10.4
- 向量检索: FAISS

## 目录结构

```
.
├── src/                   # 源代码目录
│   ├── tools/            # 工具实现
│   │   ├── calculator.py # 计算器工具
│   │   └── news_query.py # 新闻查询工具
│   ├── scheduler/        # 定时任务
│   │   ├── news_crawler.py # 新闻爬虫
│   │   └── news_crawler_agent.py # 爬虫具体实现
│   └── agent.py         # Agent核心实现
├── .env                 # 环境变量配置
├── api.py              # FastAPI接口
├── requirements.txt    # 依赖包列表
└── README.md          # 项目说明
```

## 快速开始

1. 安装依赖:

```bash
pip install -r requirements.txt
```

2. 配置环境变量:

在`.env`文件中配置以下必要参数:

```
# LangSmith配置
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_api_key
LANGSMITH_PROJECT=your_project_name

# API密钥
ZHIPUAI_API_KEY=your_zhipuai_api_key

# 数据库配置
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/database_name

# 应用配置
APP_ENV=development
DEBUG=true
```

3. 初始化数据库:

```sql
CREATE DATABASE tax_news;

CREATE TABLE news (
    id INT AUTO_INCREMENT PRIMARY KEY,
    language VARCHAR(10),
    source VARCHAR(100),
    date VARCHAR(10),
    content TEXT,
    url VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE news_sources (
    id INT AUTO_INCREMENT PRIMARY KEY,
    url VARCHAR(500),
    language VARCHAR(10),
    source_name VARCHAR(100),
    info TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

4. 启动服务:

```bash
uvicorn api:app --reload
```

## API文档

启动服务后访问 http://localhost:8000/docs 查看API文档。

### 示例请求

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "我的本金500元, 年利率3%, 计算三年后我能拿到多少钱"}'
```

## 技术特点

1. 基于LangGraph的ReAct架构实现Agent
2. 支持工具的热插拔，便于扩展新功能
3. 使用LangSmith进行对话追踪和性能监控
4. 支持多轮对话上下文记忆
5. 内置税务计算器和新闻查询工具
6. 定时任务自动更新税务新闻数据
7. 所有交互均使用中文

## 注意事项

1. 确保所有必要的环境变量都已正确配置
2. 数据库需要提前创建并配置正确的访问权限
3. 建议在开发环境下启用DEBUG模式
4. 使用LangSmith追踪功能需要配置相应的API密钥
