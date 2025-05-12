import requests
import json
from typing import List, Dict, Any
import os
import time

# API端点
API_URL = "http://127.0.0.1:8000/query"

# 问题列表
questions = [
    {
        "en": "How much time the Module 14 Test of HKICPA QP exam takes? (Search and Answer)",
        "zh": "HKICPA考试的Module 14是多长时间？（搜索之后回答）"
    },
    {
        "en": "Is the Module 14 of HKICPA QP Exam Open Book or not? (Search and Answer)",
        "zh": "HKICPA考试的Module 14是开卷还是闭卷？（搜索之后回答）"
    },
    {
        "en": "I want to enroll in June 2025 session of HKICPA QP exam, what is the DDL of enrolment? (Search and Answer)",
        "zh": "我想参加2025年6月份的HKICPA考试，报名的截止日期是？（搜索之后回答）"
    },
    {
        "en": "When should I pay my first HKICPA annual fee? (Search and Answer)",
        "zh": "我第一次的HKICPA年费应该什么时候交？（搜索之后回答）"
    },
    {
        "en": "Do I need working experience to apply for ACCA, or I only have to pass the QP exam? (Search and Answer)",
        "zh": "申请ACCA需要工作经历吗，还是过了资格考试就行？（搜索之后回答）"
    },
    {
        "en": "I want to enroll in Dec 2025 session of ACCA QP exam, what is the DDL of enrolment? (Search and Answer)",
        "zh": "我想参加2025年12月份的ACCA考试，报名的截止日期是？（搜索之后回答）"
    },
    {
        "en": "Is Hongkong's DTA with Armenia already in force or not? (Search and Answer)",
        "zh": "香港和亚美尼亚的DTA生效了吗？（搜索之后回答）"
    },
    {
        "en": "Who is the Commissioner of IRD? (Search and Answer)",
        "zh": "香港税务局的局长是谁？（搜索之后回答）"
    },
    {
        "en": "How to enroll in HKICPA exam? (Search and Answer)",
        "zh": "如何报名HKICPA考试？（搜索之后回答）"
    },
    {
        "en": "What's new on China's Announcement on the Revision of the 'Administrative Measures for Departure Tax Refunds for Overseas Travelers? (Search and Answer)",
        "zh": "〈境外旅客购物离境退税管理办法（试行）〉有什么新改动？（搜索之后回答）"
    }
]

class RetryableError(Exception):
    """可重试的错误"""
    pass

def check_answer_validity(result: Dict) -> bool:
    """检查答案是否有效
    
    Args:
        result: 问答结果字典
        
    Returns:
        bool: 答案是否有效
    """
    # 检查英文答案
    if "answer_en" in result and isinstance(result["answer_en"], list):
        for answer in result["answer_en"]:
            if isinstance(answer, str) and "error" in answer.lower():
                return False
                
    # 检查中文答案
    if "answer_zh" in result and isinstance(result["answer_zh"], list):
        for answer in result["answer_zh"]:
            if isinstance(answer, str) and ("error" in answer.lower() or "错误" in answer):
                return False
                
    return True

def query_tax_agent(text: str, max_retries: int = 3) -> Dict[str, Any]:
    """向税务问答API发送请求，支持重试机制
    
    Args:
        text: 问题文本
        max_retries: 最大重试次数
        
    Returns:
        Dict: API响应的JSON数据
    """
    payload = {
        "text": text
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # 如果请求失败则抛出异常
            result = response.json()
            
            # 检查答案是否有效
            if not check_answer_validity(result):
                raise RetryableError("获取到无效答案")
                
            return result
            
        except (requests.exceptions.RequestException, RetryableError) as e:
            if attempt < max_retries - 1:  # 如果还有重试机会
                wait_time = (attempt + 1) * 2  # 指数退避
                print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                print(f"达到最大重试次数 ({max_retries})")
                return {
                    "answers": [f"请求错误: {str(e)}"]
                }

def main():
    """主函数，处理所有问题并打印结果"""
    print("开始查询税务问答系统...\n")
    
    results = []
    results_file = "tax_qa_results.json"
    
    # 如果文件已存在，尝试加载之前的结果
    try:
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"已加载{len(results)}个已有结果\n")
            
            # 检查已有结果中的错误
            retry_indices = []
            for i, result in enumerate(results):
                if not check_answer_validity(result):
                    retry_indices.append(i)
            
            if retry_indices:
                print(f"发现{len(retry_indices)}个需要重试的问题")
                
                for i in retry_indices:
                    question = results[i]
                    print(f"\n重试问题 {i+1}:")
                    
                    # 重试英文问题
                    if not check_answer_validity({"answer_en": question["answer_en"]}):
                        print(f"重试英文: {question['question_en']}")
                        en_result = query_tax_agent(question['question_en'])
                        if check_answer_validity({"answer_en": en_result["answers"]}):
                            results[i]["answer_en"] = en_result["answers"]
                            print("英文答案更新成功")
                    
                    # 重试中文问题
                    # if not check_answer_validity({"answer_zh": question["answer_zh"]}):
                    #     print(f"重试中文: {question['question_zh']}")
                    #     zh_result = query_tax_agent(question['question_zh'])
                    #     if check_answer_validity({"answer_zh": zh_result["answers"]}):
                    #         results[i]["answer_zh"] = zh_result["answers"]
                    #         print("中文答案更新成功")
                    
                    # 保存更新后的结果
                    with open(results_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"已保存更新后的结果")
                    
    except Exception as e:
        print(f"加载已有结果失败: {e}\n")
    
    # 处理每个问题（中英文版本）
    for i, question in enumerate(questions, 1):
        # 如果这个问题已经处理过且答案有效，跳过
        if i <= len(results) and check_answer_validity(results[i-1]):
            print(f"问题 {i:02d} 已处理且有效，跳过\n")
            continue
            
        print(f"问题 {i:02d}:")
        
        try:
            # 英文问题
            print(f"英文: {question['en']}")
            en_result = query_tax_agent(question['en'])
            print("回答:")
            for answer in en_result["answers"]:
                print(f"- {answer}")
            
            # 中文问题
            # print(f"中文: {question['zh']}")
            # zh_result = query_tax_agent(question['zh'])
            # print("回答:")
            # for answer in zh_result["answers"]:
            #     print(f"- {answer}")
            
            # 保存当前问题的结果
            current_result = {
                "question_en": question["en"],
                "answer_en": en_result["answers"],
                # "question_zh": question["zh"],
                # "answer_zh": zh_result["answers"]
            }
            
            # 如果是更新已有结果
            if i <= len(results):
                results[i-1] = current_result
            else:
                results.append(current_result)
            
            # 立即保存到文件
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n已保存第 {i} 个问题的结果")
            
        except Exception as e:
            print(f"\n处理问题 {i} 时发生错误: {e}")
            print("已保存之前的结果，程序将继续处理下一个问题")
        
        print("\n" + "-"*50 + "\n")
    
    print(f"查询完成，共处理 {len(results)} 个问题，结果已保存到 {results_file}")

if __name__ == "__main__":
    main() 