# 文件处理功能指南

## 概述

本系统完全重写了文件处理功能，基于token数量实现分级处理策略，提供更智能的文件内容管理和处理能力。

## 功能特性

### 1. 基于Token的分级处理

系统根据文件内容的token数量自动选择最优的处理策略：

- **< 1000 tokens**: 直接放入消息上下文
- **1000-10000 tokens**: 当前全部放入消息，异步总结后替换
- **> 10000 tokens**: 截取前10000 tokens，按照中等文件流程处理

### 2. 支持的文件格式

- **PDF文件** (.pdf) - 使用PyPDFLoader
- **Word文档** (.doc, .docx) - 使用UnstructuredWordDocumentLoader  
- **RTF文档** (.rtf) - 使用UnstructuredRTFLoader
- **纯文本** (.txt) - 支持UTF-8、GBK、Latin-1编码

### 3. 智能错误处理

- 文件读取失败时，在消息中添加错误提示
- 只存储文件名和文件大小，不存储无法读取的内容
- 编码错误时自动尝试多种编码方式

## 核心架构

### 1. 文件处理器 (NewFileProcessor)

```python
from src.utils.file_utils import message_manager

# 处理文件列表
file_messages = await message_manager.process_file_messages(file_paths)

# 完成总结任务
updated_messages = await message_manager.finalize_summaries(file_messages)
```

### 2. 消息管理器 (MessageManager)

- 管理文件消息的生命周期
- 协调异步总结任务
- 提供标准消息格式转换

### 3. 文件存储系统 (FileStorage)

- 全文内容存储（用于后续意图识别）
- 总结内容缓存
- 基于内容哈希的去重机制

### 4. 异步总结器 (AsyncSummarizer)

- 使用LLM对大文件进行总结
- 避免重复总结相同内容
- 异步执行不阻塞主流程

## API接口

### 统一查询接口

```http
POST /query
Content-Type: multipart/form-data

text: 问题内容
thread_id: 可选的线程ID  
web_search: true
enable_rag: false
files: 可选的文件列表（支持多文件上传）
```

#### 使用示例

**纯文本查询**：
```bash
curl -X POST "http://localhost:8000/query" \
  -F "text=什么是增值税？" \
  -F "web_search=true"
```

**带文件查询**：
```bash
curl -X POST "http://localhost:8000/query" \
  -F "text=这些文档说了什么？" \
  -F "enable_rag=false" \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx"
```

## 处理流程示例

### 小文件处理 (< 1000 tokens)

```
1. 读取文件内容
2. 计算token数量: 500 tokens
3. 策略选择: DIRECT
4. 直接放入消息上下文
5. 立即可用于问答
```

### 中等文件处理 (1000-10000 tokens)

```
1. 读取文件内容
2. 计算token数量: 5000 tokens
3. 策略选择: SUMMARIZE
4. 当前完整内容放入消息
5. 启动异步总结任务
6. 对话结束后替换为总结内容
7. 保存全文用于后续召回
```

### 大文件处理 (> 10000 tokens)

```
1. 读取文件内容  
2. 计算token数量: 25000 tokens
3. 策略选择: TRUNCATE
4. 截取前10000 tokens
5. 按中等文件流程处理
6. 保存完整原文用于后续召回
```

### 错误处理

```
1. 尝试读取文件内容
2. 读取失败
3. 生成错误消息: "File 'xxx.x' uploaded but failed to open"
4. 只存储文件名和大小
5. 继续处理其他文件
```

## 配置说明

### Token限制配置

```python
token_limits = {
    "direct": 1000,      # 直接处理的token上限
    "summarize": 10000   # 总结处理的token上限
}
```

### 存储配置

```python
storage_dir = "file_storage"  # 存储根目录
full_content_dir = "file_storage/full_content"  # 全文存储
summaries_dir = "file_storage/summaries"        # 总结存储
```

## 测试和验证

运行测试脚本验证功能：

```bash
python test_file_processing.py
```

测试内容包括：
- Token计算准确性
- 分级处理策略
- 异步总结功能
- 会话处理器集成
- 错误处理机制

## 扩展性设计

### 1. 意图识别支持

系统保存了所有文件的全文内容，为后续的意图识别和智能召回功能预留了接口：

```python
# 获取全文内容（预留接口）
full_content = file_storage.load_full_content(content_hash)

# 意图识别和召回（待实现）
relevant_content = intent_recognizer.recall_relevant_content(query, full_content)
```

### 2. 缓存优化

基于内容哈希的缓存机制，避免重复处理相同文件：

```python
content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
cached_summary = file_storage.load_summary(content_hash)
```

### 3. 多模态支持

架构支持扩展到图片、音频等多媒体文件：

```python
# 预留多媒体处理接口
async def _read_image_content(self, file_path: str) -> Optional[str]:
    # OCR或图像描述生成
    pass

async def _read_audio_content(self, file_path: str) -> Optional[str]:
    # 语音转文字
    pass
```

## 性能优化

1. **异步处理**: 总结任务不阻塞主流程
2. **内容截取**: 使用二分搜索精确控制token数量
3. **编码预加载**: 一次性加载所有需要的tiktoken编码器
4. **临时文件管理**: 自动清理上传的临时文件
5. **错误恢复**: 降级机制确保系统稳定性

## 监控和日志

系统提供详细的日志记录：

```
INFO: 保存上传文件: document.pdf -> /uploads/uuid_document.pdf
INFO: 文件处理策略: SUMMARIZE, Token数量: 5000
INFO: 完成会话 thread_123 的文件总结任务
INFO: 清理临时文件: /uploads/uuid_document.pdf
```

## 注意事项

1. **内存使用**: 大文件处理时注意内存占用
2. **异步任务**: 确保在对话结束前完成总结任务
3. **文件清理**: 及时清理临时文件和缓存
4. **编码支持**: 特殊编码文件可能需要手动指定
5. **依赖库**: 确保安装了所需的文档加载器依赖 