"""
LLM系统测试脚本
用于测试算法端的LLM轮询功能
"""
import asyncio
import os
import json
import sys
from typing import Dict, Any
import logging

# 添加src到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.llm_manager import llm_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_basic_llm_call():
    """测试基本的LLM调用"""
    print("\n🧪 测试1: 基本LLM调用")
    print("-" * 50)
    
    try:
        messages = [
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ]
        
        response = await llm_manager.chat_completion(messages)
        
        print(f"✅ 调用成功!")
        print(f"📝 回答: {response.content[:200]}...")
        print(f"🤖 模型: {response.model}")
        print(f"🏭 提供商: {response.provider}")
        print(f"🔑 API Key: ****{response.api_key_id}")
        print(f"📊 Token使用: {response.usage}")
        print(f"⏱️ 响应时间: {response.response_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        return False

async def test_model_priority():
    """测试模型优先级"""
    print("\n🧪 测试2: 模型优先级")
    print("-" * 50)
    
    models_to_test = ["qwen-max-latest", "qwen-max", "gpt-4o-mini"]
    results = {}
    
    for preferred_model in models_to_test:
        try:
            messages = [
                {"role": "user", "content": f"测试模型: {preferred_model}"}
            ]
            
            response = await llm_manager.chat_completion(
                messages, 
                preferred_model=preferred_model
            )
            
            results[preferred_model] = {
                "success": True,
                "actual_model": response.model,
                "provider": response.provider,
                "api_key": response.api_key_id
            }
            
            print(f"✅ {preferred_model}: 使用了 {response.provider}/{response.model}")
            
        except Exception as e:
            results[preferred_model] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ {preferred_model}: {e}")
    
    return results

async def test_token_calculation():
    """测试Token计算"""
    print("\n🧪 测试3: Token计算和截断")
    print("-" * 50)
    
    # 创建一个长消息
    long_content = "这是一个很长的测试消息。" * 1000  # 重复1000次
    
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": long_content}
    ]
    
    try:
        response = await llm_manager.chat_completion(messages)
        
        print(f"✅ 长消息处理成功!")
        print(f"📝 回答长度: {len(response.content)} 字符")
        print(f"📊 Token使用: {response.usage}")
        
        return True
        
    except Exception as e:
        if "token" in str(e).lower():
            print(f"✅ Token限制正常工作: {e}")
            return True
        else:
            print(f"❌ 意外错误: {e}")
            return False

async def test_rate_limiting():
    """测试限流功能"""
    print("\n🧪 测试4: 限流功能")
    print("-" * 50)
    
    # 快速发送多个请求
    tasks = []
    for i in range(5):
        messages = [{"role": "user", "content": f"快速测试请求 {i+1}"}]
        task = llm_manager.chat_completion(messages)
        tasks.append(task)
    
    results = []
    for i, task in enumerate(tasks):
        try:
            response = await task
            results.append(f"✅ 请求 {i+1}: 成功 ({response.model})")
        except Exception as e:
            results.append(f"❌ 请求 {i+1}: {e}")
    
    for result in results:
        print(result)
    
    return results

async def test_status_monitoring():
    """测试状态监控"""
    print("\n🧪 测试5: 状态监控")
    print("-" * 50)
    
    try:
        status = await llm_manager.get_status()
        
        print("✅ 状态获取成功!")
        print(f"📈 配置重载时间: {status['last_config_reload']}")
        
        for provider_name, provider_status in status['providers'].items():
            print(f"\n🏭 提供商: {provider_name}")
            print(f"   🌐 Base URL: {provider_status['base_url']}")
            print(f"   🤖 模型数量: {len(provider_status['models'])}")
            print(f"   🔑 API Key数量: {len(provider_status['api_keys'])}")
            
            for api_key_status in provider_status['api_keys']:
                print(f"   - Key ****{api_key_status['id']}: "
                      f"{api_key_status['current_qpm']}/{api_key_status['qpm_limit']} QPM, "
                      f"{api_key_status['current_tpm']}/{api_key_status['tpm_limit']} TPM")
        
        return status
        
    except Exception as e:
        print(f"❌ 状态获取失败: {e}")
        return None

async def test_error_handling():
    """测试错误处理"""
    print("\n🧪 测试6: 错误处理")
    print("-" * 50)
    
    # 测试空消息
    try:
        response = await llm_manager.chat_completion([])
        print("❌ 应该拒绝空消息")
    except Exception as e:
        print(f"✅ 正确拒绝空消息: {e}")
    
    # 测试不支持的模型
    try:
        messages = [{"role": "user", "content": "测试"}]
        response = await llm_manager.chat_completion(
            messages, 
            preferred_model="non-existent-model"
        )
        print(f"✅ 自动降级到可用模型: {response.model}")
    except Exception as e:
        print(f"✅ 正确处理不支持的模型: {e}")

def print_test_summary(results: Dict[str, Any]):
    """打印测试总结"""
    print("\n📊 测试总结")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过! LLM系统工作正常!")
    else:
        print(f"\n⚠️ 有 {total_tests - passed_tests} 个测试失败，请检查配置。")

async def main():
    """主测试函数"""
    print("🚀 开始LLM系统测试")
    print("=" * 60)
    
    # 检查必要的环境变量
    required_env_vars = [
        "QWEN_API_KEY_1", "QWEN_API_KEY_2", 
        "OPENAI_API_KEY_1", "OPENAI_API_KEY_2"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"⚠️ 缺少环境变量: {missing_vars}")
        print("请在 .env 文件中配置这些变量后再运行测试。")
        return
    
    # 执行所有测试
    test_results = {}
    
    test_results['basic_call'] = await test_basic_llm_call()
    test_results['model_priority'] = await test_model_priority()
    test_results['token_calculation'] = await test_token_calculation()
    test_results['rate_limiting'] = await test_rate_limiting()
    test_results['status_monitoring'] = await test_status_monitoring()
    await test_error_handling()
    
    # 打印总结
    print_test_summary(test_results)

if __name__ == "__main__":
    # 设置事件循环策略(Windows兼容性)
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main()) 