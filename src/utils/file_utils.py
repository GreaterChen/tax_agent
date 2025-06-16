"""
文件处理工具模块

提供各种文件读取和处理功能，支持多种文件格式。
"""

import os
import logging
from typing import List
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)

class FileProcessor:
    """文件处理器类"""
    
    @staticmethod
    def read_session_files_content(session_files: List[str]) -> str:
        """读取会话文件的内容（非RAG模式）
        
        Args:
            session_files: 文件路径列表
            
        Returns:
            合并后的文件内容字符串
        """
        contents = []
        for file_path in session_files:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"文件不存在: {file_path}")
                    continue
                
                file_name = os.path.basename(file_path)
                file_content = FileProcessor.read_single_file_content(file_path)
                
                if file_content:
                    # 限制单个文件内容长度，避免prompt过长
                    max_content_length = 50000  # 约10K字符
                    if len(file_content) > max_content_length:
                        file_content = file_content[:max_content_length] + "\n...(内容已截断)..."
                    
                    contents.append(f"=== 文档：{file_name} ===\n{file_content}\n")
                    
            except Exception as e:
                logger.error(f"读取文件失败: {file_path}, 错误: {str(e)}")
                continue
        
        # 限制总内容长度
        combined_content = "\n".join(contents)
        max_total_length = 30000  # 约30K字符
        if len(combined_content) > max_total_length:
            combined_content = combined_content[:max_total_length] + "\n...(总内容已截断)..."
        
        return combined_content
    
    @staticmethod
    def read_single_file_content(file_path: str) -> str:
        """读取单个文件的内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容字符串
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                return FileProcessor._read_pdf_content(file_path)
                
            elif file_extension in ['.docx', '.doc']:
                return FileProcessor._read_word_content(file_path)
                
            elif file_extension == '.rtf':
                return FileProcessor._read_rtf_content(file_path)
                
            else:  # 默认按文本文件处理
                return FileProcessor._read_text_content(file_path)
                
        except Exception as e:
            logger.error(f"读取文件内容失败: {file_path}, 错误: {str(e)}")
            return f"[读取文件失败: {os.path.basename(file_path)}]"
    
    @staticmethod
    def _read_pdf_content(file_path: str) -> str:
        """读取PDF文件内容"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        except ImportError:
            logger.error("PyPDFLoader未安装，无法读取PDF文件")
            return f"[PDF文件读取失败: 缺少PyPDFLoader依赖]"
        except Exception as e:
            logger.error(f"读取PDF文件失败: {str(e)}")
            return f"[PDF文件读取失败: {str(e)}]"
    
    @staticmethod
    def _read_word_content(file_path: str) -> str:
        """读取Word文档内容"""
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        except ImportError:
            logger.error("UnstructuredWordDocumentLoader未安装，无法读取Word文件")
            return f"[Word文件读取失败: 缺少依赖]"
        except Exception as e:
            logger.error(f"读取Word文件失败: {str(e)}")
            return f"[Word文件读取失败: {str(e)}]"
    
    @staticmethod
    def _read_rtf_content(file_path: str) -> str:
        """读取RTF文件内容"""
        try:
            from langchain_community.document_loaders import UnstructuredRTFLoader
            loader = UnstructuredRTFLoader(file_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        except ImportError:
            logger.error("UnstructuredRTFLoader未安装，无法读取RTF文件")
            return f"[RTF文件读取失败: 缺少依赖]"
        except Exception as e:
            logger.error(f"读取RTF文件失败: {str(e)}")
            return f"[RTF文件读取失败: {str(e)}]"
    
    @staticmethod
    def _read_text_content(file_path: str) -> str:
        """读取纯文本文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"无法识别文件编码: {str(e)}")
                    return f"[文本文件读取失败: 编码问题]"
        except Exception as e:
            logger.error(f"读取文本文件失败: {str(e)}")
            return f"[文本文件读取失败: {str(e)}]"

# 提供便捷的函数接口
def read_session_files_content(session_files: List[str]) -> str:
    """读取会话文件内容的便捷函数"""
    return FileProcessor.read_session_files_content(session_files)

def read_single_file_content(file_path: str) -> str:
    """读取单个文件内容的便捷函数"""
    return FileProcessor.read_single_file_content(file_path) 