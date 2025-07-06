#!/usr/bin/env python3
"""
文件处理功能测试脚本
测试新的基于token的分级文件处理功能
"""
import asyncio
import tempfile
import os
from pathlib import Path

# 测试文件处理功能
async def test_file_processing():
    """测试文件处理功能"""
    from src.utils.file_utils import message_manager
    
    # 创建测试文件
    test_files = []
    
    # 1. 创建小文件 (< 1000 tokens)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("这是一个小文件测试。\n" * 20)  # 约100个字符
        test_files.append(f.name)
    
    # 2. 创建中等文件 (1000-10000 tokens)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("这是一个中等大小的文件测试内容。包含更多的信息和详细的描述。\n" * 100)  # 约5000个字符
        test_files.append(f.name)
    
    # 3. 创建大文件 (> 10000 tokens)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("这是一个大文件测试内容。包含大量的信息和详细的描述。用于测试文件截取功能。\n" * 500)  # 约25000个字符
        test_files.append(f.name)
    
    # 4. 创建无法读取的文件
    invalid_file = tempfile.NamedTemporaryFile(suffix='.invalid', delete=False)
    invalid_file.close()
    test_files.append(invalid_file.name)
    
    try:
        print("开始测试文件处理功能...")
        
        # 测试文件处理
        file_messages = await message_manager.process_file_messages(test_files)
        
        print(f"\n处理了 {len(file_messages)} 个文件消息:")
        for i, msg in enumerate(file_messages):
            file_info = msg.get("file_info", {})
            print(f"\n文件 {i+1}:")
            print(f"  文件名: {file_info.get('filename', 'Unknown')}")
            print(f"  文件大小: {file_info.get('file_size', 0)} bytes")
            print(f"  处理策略: {file_info.get('strategy', 'Unknown')}")
            print(f"  Token数量: {file_info.get('token_count', 0)}")
            print(f"  是否有错误: {file_info.get('error', False)}")
            print(f"  消息内容长度: {len(msg.get('content', ''))}")
        
        # 测试总结功能
        print("\n\n开始测试总结功能...")
        updated_messages = await message_manager.finalize_summaries(file_messages)
        
        print(f"\n总结完成后的消息:")
        for i, msg in enumerate(updated_messages):
            file_info = msg.get("file_info", {})
            print(f"\n文件 {i+1}:")
            print(f"  文件名: {file_info.get('filename', 'Unknown')}")
            print(f"  是否总结: {file_info.get('is_summary', False)}")
            print(f"  消息内容长度: {len(msg.get('content', ''))}")
            if file_info.get('is_summary'):
                print(f"  总结内容预览: {msg.get('content', '')[:100]}...")
        
        print("\n文件处理功能测试完成！")
        
    finally:
        # 清理测试文件
        for file_path in test_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"清理测试文件: {file_path}")
            except Exception as e:
                print(f"清理文件失败: {file_path}, {e}")

async def test_token_calculation():
    """测试token计算功能"""
    from src.utils.token_manager import token_manager
    
    print("\n开始测试token计算功能...")
    
    test_texts = [
        "这是一个简单的测试",
        "这是一个更长的测试文本，包含更多的内容和信息。" * 10,
        "这是一个非常长的测试文本，用于测试token计算的准确性。" * 100,
    ]
    
    for i, text in enumerate(test_texts):
        token_count = token_manager.count_tokens(text)
        print(f"文本 {i+1}: {len(text)} 字符, {token_count} tokens")
    
    print("token计算功能测试完成！")

async def test_session_processor():
    """测试会话处理器功能"""
    from src.utils.session_processor import session_processor
    
    print("\n开始测试会话处理器功能...")
    
    # 创建测试文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("这是一个测试文档，用于测试会话处理器的功能。\n包含一些测试内容。")
        test_file = f.name
    
    try:
        question = "这个文档说了什么？"
        thread_id = "test_thread_123"
        
        # 测试非RAG模式
        enhanced_question, session_vector_tool, file_messages = await session_processor.process_session_files(
            question, [test_file], False, thread_id
        )
        
        print(f"原始问题: {question}")
        print(f"增强后问题长度: {len(enhanced_question)}")
        print(f"文件消息数量: {len(file_messages)}")
        print(f"会话向量工具: {session_vector_tool}")
        
        # 测试总结完成
        success = await session_processor.finalize_session_summaries(thread_id)
        print(f"总结完成状态: {success}")
        
        # 获取处理后的消息
        processed_messages = session_processor.get_processed_file_messages(thread_id)
        print(f"处理后消息数量: {len(processed_messages)}")
        
        # 清理会话
        session_processor.cleanup_session(thread_id)
        
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"清理测试文件: {test_file}")
    
    print("会话处理器功能测试完成！")

async def main():
    """主测试函数"""
    print("========== 新文件处理功能测试 ==========")
    
    try:
        await test_token_calculation()
        await test_file_processing()
        await test_session_processor()
        
        print("\n========== 所有测试完成 ==========")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 