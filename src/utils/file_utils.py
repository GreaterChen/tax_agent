"""
文件处理工具模块 - 完全重写版本

基于token数量的分级处理策略：
- < 1000 tokens: 直接放在消息列表中
- 1000-10000 tokens: 当前全部放在message中，异步总结后替换
- > 10000 tokens: 截取前10000 tokens，按照上面的流程处理
"""

import os
import logging
import asyncio
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .token_manager import token_manager
from config.file_config import FILE_PROCESSING_CONFIG

# 配置日志
logger = logging.getLogger(__name__)

class ProcessingStrategy(Enum):
    """文件处理策略枚举"""
    DIRECT = "direct"           # 直接放入消息 < 1000 tokens
    SUMMARIZE = "summarize"     # 需要总结 1000-10000 tokens
    TRUNCATE = "truncate"       # 截取处理 > 10000 tokens

@dataclass
class FileProcessingResult:
    """文件处理结果"""
    success: bool
    filename: str
    file_size: int
    strategy: ProcessingStrategy
    content: Optional[str] = None
    token_count: int = 0
    error_message: Optional[str] = None
    content_hash: Optional[str] = None
    truncated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

@dataclass
class FileMessage:
    """文件消息数据结构"""
    role: str = "user"
    content: str = ""
    file_info: Optional[Dict[str, Any]] = None
    is_summary: bool = False
    original_hash: Optional[str] = None

class FileStorage:
    """文件存储管理器"""
    
    def __init__(self, storage_dir: str = "file_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.full_content_dir = self.storage_dir / "full_content"
        self.summaries_dir = self.storage_dir / "summaries"
        self.full_content_dir.mkdir(exist_ok=True)
        self.summaries_dir.mkdir(exist_ok=True)
    
    def save_full_content(self, content_hash: str, content: str, metadata: Dict[str, Any]) -> str:
        """保存全文内容"""
        try:
            file_path = self.full_content_dir / f"{content_hash}.json"
            data = {
                "content": content,
                "metadata": metadata,
                "timestamp": time.time()
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return str(file_path)
        except Exception as e:
            logger.error(f"保存全文内容失败: {e}")
            return ""
    
    def load_full_content(self, content_hash: str) -> Optional[str]:
        """加载全文内容"""
        try:
            file_path = self.full_content_dir / f"{content_hash}.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("content")
            return None
        except Exception as e:
            logger.error(f"加载全文内容失败: {e}")
            return None
    
    def save_summary(self, content_hash: str, summary: str, metadata: Dict[str, Any]) -> str:
        """保存总结内容"""
        try:
            file_path = self.summaries_dir / f"{content_hash}_summary.json"
            data = {
                "summary": summary,
                "metadata": metadata,
                "timestamp": time.time()
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return str(file_path)
        except Exception as e:
            logger.error(f"保存总结内容失败: {e}")
            return ""
    
    def load_summary(self, content_hash: str) -> Optional[str]:
        """加载总结内容"""
        try:
            file_path = self.summaries_dir / f"{content_hash}_summary.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("summary")
            return None
        except Exception as e:
            logger.error(f"加载总结内容失败: {e}")
            return None

class AsyncSummarizer:
    """异步总结器"""
    
    def __init__(self, storage: FileStorage):
        self.storage = storage
        self.active_tasks = {}
    
    async def summarize_content(self, content: str, filename: str, content_hash: str) -> Optional[str]:
        """异步总结文件内容"""
        try:
            # 避免重复任务
            if content_hash in self.active_tasks:
                return await self.active_tasks[content_hash]
            
            # 创建总结任务
            task = asyncio.create_task(self._create_summary(content, filename))
            self.active_tasks[content_hash] = task
            
            try:
                summary = await task
                if summary:
                    # 保存总结
                    self.storage.save_summary(content_hash, summary, {
                        "filename": filename,
                        "original_length": len(content),
                        "summary_length": len(summary)
                    })
                return summary
            finally:
                # 清理任务
                self.active_tasks.pop(content_hash, None)
                
        except Exception as e:
            logger.error(f"总结文件内容失败 {filename}: {e}")
            return None
    
    async def _create_summary(self, content: str, filename: str) -> Optional[str]:
        """使用LLM创建总结"""
        try:
            # 导入LLM配置
            from config.llm_config import llm_config
            
            # 获取指定的总结LLM
            summary_llm_name = FILE_PROCESSING_CONFIG["summary"]["llm_name"]
            suitable_llm = llm_config.get_llm_by_name(summary_llm_name)
            if not suitable_llm:
                logger.error(f"未找到总结LLM: {summary_llm_name}")
                return None
            
            llm_instance = suitable_llm["llm"]
            
            # 构建总结提示
            summary_prompt = FILE_PROCESSING_CONFIG["summary"]["prompt_template"].format(
                filename=filename,
                content=content
            )

            # 调用LLM
            response = await llm_instance.ainvoke(summary_prompt)
            
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"LLM总结失败: {e}")
            return None

class FileProcessor:
    """文件处理器"""
    
    def __init__(self):
        self.storage = FileStorage(FILE_PROCESSING_CONFIG["file_storage"]["storage_dir"])
        self.summarizer = AsyncSummarizer(self.storage)
        self.token_limits = FILE_PROCESSING_CONFIG["token_limits"]
    
    async def process_uploaded_files(self, file_paths: List[str]) -> Tuple[List[FileMessage], List[asyncio.Task]]:
        """处理上传的文件列表
            
        Returns:
            (消息列表, 异步任务列表)
        """
        messages = []
        summary_tasks = []
        
        for file_path in file_paths:
            try:
                result = await self.process_single_file(file_path)
                message, task = self._create_file_message(result)
                messages.append(message)
                
                if task:
                    summary_tasks.append(task)
                    
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
                # 添加错误消息
                error_message = FileMessage(
                    role="user",
                    content=f'File "{os.path.basename(file_path)}" uploaded but failed to process: {str(e)}',
                    file_info={"error": True, "filename": os.path.basename(file_path)}
                )
                messages.append(error_message)
        
        return messages, summary_tasks
    
    async def process_single_file(self, file_path: str) -> FileProcessingResult:
        """处理单个文件"""
        filename = os.path.basename(file_path)
        
        try:
            # 获取文件大小
            file_size = os.path.getsize(file_path)
            
            # 尝试读取文件内容
            content = await self._read_file_content(file_path)
            
            if content is None:
                return FileProcessingResult(
                    success=False,
                    filename=filename,
                    file_size=file_size,
                    strategy=ProcessingStrategy.DIRECT,
                    error_message="无法读取文件内容"
                )
            
            # 计算token数量
            token_count = token_manager.count_tokens(content)
            
            # 生成内容哈希
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # 确定处理策略
            strategy = self._determine_strategy(token_count)
            
            # 根据策略处理内容
            processed_content = content
            truncated = False
            
            if strategy == ProcessingStrategy.TRUNCATE:
                # 截取前10000 tokens
                processed_content = self._truncate_content(content, self.token_limits["summarize"])
                truncated = True
                token_count = self.token_limits["summarize"]
            
            # 保存全文内容（用于后续意图识别）
            if strategy in [ProcessingStrategy.SUMMARIZE, ProcessingStrategy.TRUNCATE]:
                self.storage.save_full_content(content_hash, content, {
                    "filename": filename,
                    "file_size": file_size,
                    "original_token_count": token_manager.count_tokens(content),
                    "strategy": strategy.value
                })
            
            return FileProcessingResult(
                success=True,
                filename=filename,
                file_size=file_size,
                strategy=strategy,
                content=processed_content,
                token_count=token_count,
                content_hash=content_hash,
                truncated=truncated
            )
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return FileProcessingResult(
                success=False,
                filename=filename,
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                strategy=ProcessingStrategy.DIRECT,
                error_message=str(e)
            )
    
    def _determine_strategy(self, token_count: int) -> ProcessingStrategy:
        """确定处理策略"""
        if token_count < self.token_limits["direct"]:
            return ProcessingStrategy.DIRECT
        elif token_count <= self.token_limits["summarize"]:
            return ProcessingStrategy.SUMMARIZE
        else:
            return ProcessingStrategy.TRUNCATE
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """截取内容到指定token数量"""
        try:
            # 使用二分搜索找到最佳截取点
            left, right = 0, len(content)
            result = content
            
            while left < right:
                mid = (left + right + 1) // 2
                truncated = content[:mid]
                
                if token_manager.count_tokens(truncated) <= max_tokens:
                    result = truncated
                    left = mid
                else:
                    right = mid - 1
            
            return result
        except Exception as e:
            logger.error(f"内容截取失败: {e}")
            # 简单截取
            return content[:max_tokens * 4]  # 粗略估算4字符=1token
    
    def _create_file_message(self, result: FileProcessingResult) -> Tuple[FileMessage, Optional[asyncio.Task]]:
        """创建文件消息"""
        if not result.success:
            # 失败情况
            return FileMessage(
                role="user",
                content=f'File "{result.filename}" uploaded but failed to open',
                file_info={
                    "filename": result.filename,
                    "file_size": result.file_size,
                    "error": True
                }
            ), None
        
        # 成功情况
        if result.strategy == ProcessingStrategy.DIRECT:
            # 直接放入消息
            return FileMessage(
                role="user",
                content=f"=== 文档：{result.filename} ===\n{result.content}",
                file_info={
                    "filename": result.filename,
                    "file_size": result.file_size,
                    "token_count": result.token_count,
                    "strategy": result.strategy.value
                }
            ), None
        
        else:
            # 需要总结的情况
            truncate_note = " (content truncated)" if result.truncated else ""
            
            message = FileMessage(
                role="user",
                content=f"=== 文档：{result.filename}{truncate_note} ===\n{result.content}",
                file_info={
                    "filename": result.filename,
                    "file_size": result.file_size,
                    "token_count": result.token_count,
                    "strategy": result.strategy.value,
                    "content_hash": result.content_hash,
                    "truncated": result.truncated
                },
                original_hash=result.content_hash
            )
            
            # 创建异步总结任务
            task = asyncio.create_task(
                self.summarizer.summarize_content(
                    result.content, 
                    result.filename,
                    result.content_hash
                )
            )
            
            return message, task
    
    async def _read_file_content(self, file_path: str) -> Optional[str]:
        """读取文件内容"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                return await self._read_pdf_content(file_path)
            elif file_extension in ['.docx', '.doc']:
                return await self._read_word_content(file_path)
            elif file_extension == '.rtf':
                return await self._read_rtf_content(file_path)
            else:
                return await self._read_text_content(file_path)
                
        except Exception as e:
            logger.error(f"读取文件内容失败: {file_path}, 错误: {str(e)}")
            return None
    
    async def _read_pdf_content(self, file_path: str) -> Optional[str]:
        """读取PDF文件内容"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            return "\n".join([doc.page_content for doc in docs])
        except ImportError:
            logger.error("PyPDFLoader未安装，无法读取PDF文件")
            return None
        except Exception as e:
            logger.error(f"读取PDF文件失败: {str(e)}")
            return None
    
    async def _read_word_content(self, file_path: str) -> Optional[str]:
        """读取Word文档内容"""
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            return "\n".join([doc.page_content for doc in docs])
        except ImportError:
            logger.error("UnstructuredWordDocumentLoader未安装，无法读取Word文件")
            return None
        except Exception as e:
            logger.error(f"读取Word文件失败: {str(e)}")
            return None
    
    async def _read_rtf_content(self, file_path: str) -> Optional[str]:
        """读取RTF文件内容"""
        try:
            from langchain_community.document_loaders import UnstructuredRTFLoader
            loader = UnstructuredRTFLoader(file_path)
            docs = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            return "\n".join([doc.page_content for doc in docs])
        except ImportError:
            logger.error("UnstructuredRTFLoader未安装，无法读取RTF文件")
            return None
        except Exception as e:
            logger.error(f"读取RTF文件失败: {str(e)}")
            return None
    
    async def _read_text_content(self, file_path: str) -> Optional[str]:
        """读取纯文本文件内容"""
        try:
            def read_with_encoding(path: str, encoding: str) -> str:
                with open(path, 'r', encoding=encoding) as f:
                    return f.read()
            
            # 尝试不同编码
            encodings = ['utf-8', 'gbk', 'latin-1']
            for encoding in encodings:
                try:
                    return await asyncio.get_event_loop().run_in_executor(
                        None, read_with_encoding, file_path, encoding
                    )
                except UnicodeDecodeError:
                    continue
            
            logger.error(f"无法识别文件编码: {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"读取文本文件失败: {str(e)}")
            return None

class MessageManager:
    """消息管理器"""
    
    def __init__(self):
        self.file_processor = FileProcessor()
        self.pending_summary_tasks = {}
    
    async def process_file_messages(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """处理文件消息并返回标准消息格式"""
        if not file_paths:
            return []
        
        messages, summary_tasks = await self.file_processor.process_uploaded_files(file_paths)
        
        # 存储待处理的总结任务
        for i, task in enumerate(summary_tasks):
            message = messages[i]
            if message.original_hash:
                self.pending_summary_tasks[message.original_hash] = task
        
        # 转换为标准消息格式
        result = []
        for message in messages:
            result.append({
                "role": message.role,
                "content": message.content,
                "file_info": message.file_info
            })
        
        return result
    
    async def finalize_summaries(self, processed_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """完成总结任务并替换消息内容"""
        updated_messages = []
        
        for message in processed_messages:
            file_info = message.get("file_info", {})
            
            # 检查是否需要总结替换
            if file_info.get("strategy") in ["summarize", "truncate"] and file_info.get("content_hash"):
                content_hash = file_info["content_hash"]
                
                # 等待总结任务完成
                if content_hash in self.pending_summary_tasks:
                    try:
                        summary = await self.pending_summary_tasks[content_hash]
                        
                        if summary:
                            # 替换为总结内容
                            filename = file_info["filename"]
                            limit_note = f"File {filename} length limit exceeded, summary: "
                            
                            updated_message = message.copy()
                            updated_message["content"] = limit_note + summary
                            updated_message["file_info"]["is_summary"] = True
                            updated_messages.append(updated_message)
                        else:
                            # 总结失败，保持原内容
                            updated_messages.append(message)
                            
                    except Exception as e:
                        logger.error(f"总结任务失败: {e}")
                        updated_messages.append(message)
                    finally:
                        # 清理任务
                        self.pending_summary_tasks.pop(content_hash, None)
                else:
                    updated_messages.append(message)
            else:
                updated_messages.append(message)
        
        return updated_messages

# 创建全局实例
message_manager = MessageManager()

# 向后兼容的便捷函数
async def process_file_messages(file_paths: List[str]) -> List[Dict[str, Any]]:
    """处理文件消息的便捷函数"""
    return await message_manager.process_file_messages(file_paths)

async def finalize_file_summaries(processed_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """完成文件总结的便捷函数"""
    return await message_manager.finalize_summaries(processed_messages) 