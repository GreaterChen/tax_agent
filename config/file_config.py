"""
文件处理配置
定义文件处理相关的配置参数
"""

# 文件处理相关配置
FILE_PROCESSING_CONFIG = {
    # Token限制配置
    "token_limits": {
        "direct": 1000,      # 直接放入消息的token限制
        "summarize": 10000   # 需要总结处理的token限制
    },
    
    # 文件处理策略配置
    "processing_strategies": {
        "DIRECT": "direct",           # 直接放入消息 < 1000 tokens
        "SUMMARIZE": "summarize",     # 需要总结 1000-10000 tokens  
        "TRUNCATE": "truncate"        # 截取处理 > 10000 tokens
    },
    
    # 支持的文件格式
    "supported_file_formats": {
        'pdf': 'PDF文档',
        'docx': 'Word文档',
        'doc': 'Word文档(旧版)',
        'rtf': 'RTF文档',
        'txt': '文本文件',
        'md': 'Markdown文件'
    },
    
    # 文件存储配置
    "file_storage": {
        "storage_dir": "file_storage",
        "max_file_size": 100 * 1024 * 1024,  # 100MB
        "cleanup_after_days": 30  # 30天后清理文件
    },
    
    # 总结配置
    "summary": {
        "max_summary_length": 2000,  # 总结最大长度（字符数）
        "llm_name": "qwen-max",      # 总结使用的LLM
        "prompt_template": """请为以下文档内容生成一个简洁的总结，突出关键信息和要点：

文档名称：{filename}
文档内容：
{content}

请提供一个结构化的总结，包括：
1. 文档主要内容概述
2. 关键信息和数据
3. 重要结论或建议

总结："""
    }
} 