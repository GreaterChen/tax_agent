"""
统一日志配置模块
解决Windows系统下中文日志乱码问题
"""
import os
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(
    module_name: str = "langchain_agent",
    log_level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    设置统一的日志配置，确保UTF-8编码
    
    Args:
        module_name: 模块名称
        log_level: 日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        log_dir: 日志目录
        
    Returns:
        配置好的logger实例
    """
    
    # 创建logger
    logger = logging.getLogger(module_name)
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
    
    logger.setLevel(log_level)
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handlers = []
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # 文件输出
    if file_output:
        # 确保日志目录存在
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # 生成日志文件名
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file = log_path / f"{module_name}_{current_date}.log"
        
        # 创建文件处理器，明确指定UTF-8编码
        file_handler = logging.FileHandler(
            log_file, 
            mode='a', 
            encoding='utf-8'  # 关键：明确指定UTF-8编码
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # 添加所有处理器
    for handler in handlers:
        logger.addHandler(handler)
    
    # 防止日志传播到根logger
    logger.propagate = False
    
    logger.info(f"日志系统已初始化: {module_name}, 级别: {logging.getLevelName(log_level)}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    获取logger实例，如果不存在则创建
    
    Args:
        name: logger名称
        
    Returns:
        logger实例
    """
    return setup_logging(name)

def setup_root_logging():
    """
    设置根日志配置，确保所有模块的日志都使用UTF-8编码
    """
    # 获取根logger
    root_logger = logging.getLogger()
    
    # 如果已经配置过，先清除现有handlers
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    root_logger.setLevel(logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = log_path / f"app_{current_date}.log"
    
    file_handler = logging.FileHandler(
        log_file, 
        mode='a', 
        encoding='utf-8'  # 关键：明确指定UTF-8编码
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    root_logger.info("根日志系统已配置，使用UTF-8编码")

# 在模块导入时自动配置根日志
setup_root_logging() 