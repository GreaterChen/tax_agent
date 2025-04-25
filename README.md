# 税务问答Agent系统

基于LangChain v0.3和FastAPI开发的税务问答系统,支持税务计算和新闻查询。

## 功能特点

- 基于Claude 3模型的智能问答
- 支持数学计算(四则运算、括号优先级)
- 自动爬取和查询税务新闻
- RESTful API接口
- 定时更新新闻数据

## 系统架构

- 后端框架: FastAPI
- LLM框架: LangChain v0.3
- 大语言模型: Claude 3
- 数据库: MySQL
- 定时任务: APScheduler

## 目录结构

```
.
├── src/
│   ├── tools/              # 工具实现
│   │   ├── calculator.py   # 计算器工具
│   │   └── news_query.py   # 新闻查询工具
│   ├── scheduler/          # 定时任务
│   │   └── news_crawler.py # 新闻爬虫
│   ├── agent.py           # Agent实现
│   └── api.py             # FastAPI接口
├── .env                   # 环境变量配置
├── requirements.txt       # 依赖包列表
└── README.md             # 项目说明
```

## 快速开始

1. 安装依赖:

```bash
pip install -r requirements.txt
```

2. 配置环境变量:

复制`.env.example`为`.env`,并填写相关配置:

```
ANTHROPIC_API_KEY=your_api_key
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/tax_news
```

3. 初始化数据库:

```sql
CREATE DATABASE tax_news;

CREATE TABLE news (
    id INT AUTO_INCREMENT PRIMARY KEY,
    language VARCHAR(10),
    source VARCHAR(100),
    date VARCHAR(8),
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
     -d '{"text": "请帮我计算个人所得税,月收入15000元"}'
```

## 开发说明

1. Agent实现基于LangGraph,使用ReAct架构
2. 工具系统支持热插拔,便于扩展
3. 新闻爬虫每天凌晨2点自动运行
4. 所有响应使用中文
