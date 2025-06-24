# 税务Agent系统技术实现指南

## 🎯 系统架构

### 核心设计理念
- **模块化设计**: 将配置、选择器、工具管理等功能分离到独立模块
- **保持LangChain原生体验**: 使用原生的`ChatTongyi`、`ChatOpenAI`等ChatModel
- **强制限流功能**: 限流功能为必选，确保生产环境稳定性
- **清晰的职责分离**: 每个模块只负责自己的核心功能

### 架构组件

```
langchain-agent/
├── config/
│   └── llm_config.py          # LLM配置管理
├── src/
│   ├── agent.py               # 核心Agent调用逻辑
│   ├── llm_selector.py        # LLM选择器（含限流）
│   ├── tools_manager.py       # 工具管理器
│   ├── prompts.py            # 提示词管理
│   └── utils/
│       ├── rate_limiter.py    # Redis限流器
│       └── token_counter.py   # Token计算器
```

## 🔧 核心模块详解

### 1. LLM配置管理 (`config/llm_config.py`)

**职责**: 管理所有LLM的配置信息

```python
class LLMConfig:
    def _build_configs(self):
        # 通义千问配置
        if os.getenv("QWEN_API_KEY_1"):
            self.llm_configs.append({
                "name": "qwen_key1",
                "llm": ChatTongyi(...),
                "priority": 1,
                "qpm_limit": 100,
                "tpm_limit": 200000,
            })
        # ...其他配置
```

**特点**:
- 动态读取环境变量
- 按优先级排序
- 支持多个API key配置
- 自动过滤无效配置

### 2. LLM选择器 (`src/llm_selector.py`)

**职责**: 智能选择可用的LLM，处理限流逻辑

```python
class LLMSelector:
    async def select_best_llm(self, question: str) -> Dict[str, Any]:
        # 计算token数
        request_tokens = self.token_counter.count_tokens(question)
        
        # 按优先级检查限流
        for llm_config_item in available_llms:
            allowed, status = await self.rate_limiter.check_and_increment(...)
            if allowed:
                return llm_config_item
```

**特点**:
- 强制启用限流功能
- 按优先级智能选择
- 实时限流检查
- 提供使用状态查询

### 3. 工具管理器 (`src/tools_manager.py`)

**职责**: 管理各种Agent工具的加载和配置

```python
class ToolsManager:
    def get_tools(self, web_search=True, session_vector_tool=None):
        tools = []
        if session_vector_tool:
            tools.append(session_vector_tool)  # 最高优先级
        tools.extend(self.base_tools)
        if web_search:
            tools.append(self.web_search_tool)
        return tools
```

**特点**:
- 基础工具始终可用
- 会话工具优先级最高
- 灵活的工具组合
- 工具信息查询接口

### 4. 提示词管理 (`src/prompts.py`)

**职责**: 统一管理系统提示词和问题增强逻辑

```python
SYSTEM_PROMPT = """你是一个专业的税务顾问助手..."""

def create_enhanced_question(question: str, session_files: list = None) -> str:
    if session_files:
        return f"""用户已上传相关文档：{file_names}
        
用户问题：{question}"""
    return question
```

### 5. 核心Agent (`src/agent.py`)

**职责**: 调用协调，业务流程控制

```python
class TaxAgent:
    def query(self, question: str, ...):
        # 1. 处理会话文件和问题增强
        enhanced_question, session_vector_tool = self._process_session_files(...)
        
        # 2. 获取工具列表
        tools = tools_manager.get_tools(...)
        
        # 3. 智能选择LLM
        selected_llm = self._select_llm_with_retry(enhanced_question)
        
        # 4. 创建工作流并执行
        workflow = self._create_graph(tools, selected_llm)
        return self._execute_workflow(workflow, ...)
```

## 🚀 部署配置

### 必需环境变量

```bash
# Redis配置（限流功能必需）
REDIS_URL=redis://localhost:6379/2

# 至少配置一个LLM API Key
QWEN_API_KEY_1=sk-xxx
QWEN_API_KEY_2=sk-yyy
OPENAI_API_KEY_1=sk-zzz
OPENAI_API_KEY_2=sk-www

# 或使用默认配置
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-xxx
```

### 启动检查

系统启动时会自动检查：
1. Redis连接是否正常
2. 至少有一个可用的LLM配置
3. 限流器和Token计算器初始化

## 📊 功能特性

### ✅ 模块化特性

1. **配置管理**:
   - 独立的LLM配置模块
   - 环境变量动态加载
   - 配置状态查询

2. **智能选择**:
   - 强制限流检查
   - 按优先级轮询
   - 实时使用量监控

3. **工具管理**:
   - 灵活的工具组合
   - 会话级RAG支持
   - 工具信息查询

4. **提示词管理**:
   - 统一的系统提示词
   - 智能问题增强
   - RAG/非RAG模式支持

### 🔄 调用流程

```
用户请求 
    ↓
TaxAgent.query() 
    ↓
处理会话文件 → 获取工具列表 → 选择LLM → 创建工作流 → 执行查询
    ↓           ↓             ↓         ↓         ↓
prompts.py  tools_manager  llm_selector  LangGraph  结果返回
```

## 🛠 维护和监控

### 状态查询

```python
# 完整的系统状态
status = tax_agent.get_status()
print(status)
# 输出:
# {
#   "agent_status": "running",
#   "llm_config": {...},      # LLM配置状态
#   "llm_usage": {...},       # 实时使用量
#   "tools": {...}           # 工具信息
# }

# LLM使用状态
usage_status = await llm_selector.get_usage_status()

# 配置状态
config_status = llm_config.get_status()

# 工具信息
tools_info = tools_manager.get_available_tools_info()
```

### 日志监控

每个模块都有详细的日志记录：

```python
# config/llm_config.py
logger.info(f"LLM配置加载完成，可用模型: {model_names}")

# src/llm_selector.py  
logger.info(f"选择LLM: {llm_name}")
logger.warning(f"LLM {llm_name} 限流中: {reason}")

# src/tools_manager.py
logger.info(f"添加基础工具: {tool_names}")

# src/agent.py
logger.info("TaxAgent初始化完成")
logger.warning(f"触发限流错误，第 {retry + 1}/{max_retries} 次重试")
```

## 🔒 生产就绪特性

1. **强制限流**: 必须配置Redis，确保生产环境稳定
2. **模块隔离**: 各模块职责清晰，便于维护和测试
3. **错误处理**: 完整的异常捕获和重试机制
4. **状态监控**: 多层级的状态查询接口
5. **配置验证**: 启动时自动验证配置有效性

## 💡 开发指南

### 添加新的LLM提供商

1. 在`config/llm_config.py`中添加配置：
```python
if os.getenv("NEW_PROVIDER_API_KEY"):
    self.llm_configs.append({
        "name": "new_provider",
        "llm": NewProviderChatModel(...),
        "provider": "new_provider",
        "priority": 5,
        "qpm_limit": 60,
        "tpm_limit": 100000,
    })
```

### 添加新的工具

1. 在`src/tools_manager.py`中注册：
```python
from src.tools.new_tool import new_tool

class ToolsManager:
    def __init__(self):
        self.new_tool = new_tool  # 添加新工具
    
    def get_tools(self, use_new_tool=False, ...):
        if use_new_tool:
            tools.append(self.new_tool)
```

这个模块化架构既保持了代码的清晰度，又提供了完整的生产级功能！ 