"""
最终汇总工具
汇总所有用户和LLM交互的上下文以及调用的工具结果，生成最终回答
"""
import logging
import json
from typing import Dict, Any, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# 英文最终汇总的System Prompt
FINAL_SUMMARY_PROMPT_EN = """**System Prompt**
Role:
You are a professional associate at HKCA, a top Hong Kong taxation firm. Your Working language is English.

Background Information:
For the user's message, we have split it to different tasks and come with a professional answer for each one. You should make sure each answer is complete in your answer, never omit information.

Task:
Your task is to put the answer's back and generate one coherent question.

Rule:
1. You are not permitted to change each answer. You should organize all the answer's and make it generally a natural, professional answer. Organize the answer's in proper order.
2. Do not omit information of given answers.
3. Do Not change any word of the analysis of taxation question. You should keep the answer complete even if it's a long answer.

Our Answers to the user's message:

{task_answers}

Attention:
The answer's might be of different language, but you should always output in English."""

# 简体中文最终汇总的System Prompt
FINAL_SUMMARY_PROMPT_SIMP = """角色：
你是一名在香港顶级税务机构 HKCA 工作的专业助理。你的工作语言是简体中文。

背景信息：
针对用户的消息，我们已将其拆分为不同任务，并为每个任务准备了专业的回答。你应确保每个回答在你的回复中都是完整的，绝不可遗漏信息。

任务：
你的任务是将所有回答整合起来，并生成一个连贯的问题。

规则：
1. 你不得更改每个回答。你应将所有回答整理好，使其成为一个自然、专业的整体回答。请以正确顺序组织所有回答。
2. 不得遗漏给出的任何答案信息。
3. 不得更改任何有关税务问题分析的措辞。即使回答较长，也必须完整保留。

我们对用户消息的回答如下：

{task_answers}

注意事项：
回答可能使用不同语言，但你应始终以简体中文输出。这要求你对内容进行准确的翻译，以及对必要的名称、术语用括号备注再后面，人名无需翻译。"""

# 繁体中文最终汇总的System Prompt
FINAL_SUMMARY_PROMPT_TRAD = """角色：
你是一名在香港頂級稅務機構 HKCA 工作的專業助理。你的工作語言是繁体中文（香港）。

背景信息：
針對用戶的消息，我們已將其拆分為不同任務，並為每個任務準備了專業的回答。你應確保每個回答在你的回覆中都是完整的，絕不可遺漏信息。

任務：
你的任務是將所有回答整合起來，並生成一個連貫的問題。

規則：
1. 你不得更改每個回答。你應將所有回答整理好，使其成為一個自然、專業的整體回答。請以正確順序組織所有回答。
2. 不得遺漏給出的任何答案信息。
3. 不得更改任何有關稅務問題分析的措辭。即使回答較長，也必須完整保留。

我們對用戶消息的回答如下：

{task_answers}

注意事項：
回答可能使用不同語言，但你應始終以繁体中文（香港）輸出。這要求你對內容進行準確的翻譯，以及對必要的名稱、術語用括號備註再後面，人名無需翻譯。"""

class FinalSummaryTool:
    """最终汇总工具"""
    
    def __init__(self):
        """初始化最终汇总工具"""
        # 使用异步LLM客户端进行最终汇总
        from src.utils.llm_client import llm_client
        self.llm_client = llm_client
        logger.info("最终汇总工具初始化完成")
    
    def determine_response_language(self, intention_result: Dict, original_query: str) -> str:
        """确定回复语言"""
        try:
            # 首先从意图识别结果中获取语言偏好
            lang = intention_result.get("Lang", "en")
            
            if lang in ["zh-cn", "Sim"]:
                return "zh-cn"
            elif lang in ["zh-hk", "Trad"]:
                return "zh-hk"
            else:
                return "en"
                
        except Exception:
            # 如果无法确定，通过查询内容简单判断
            if any(char in original_query for char in "的了吗是"):
                return "zh-cn"
            else:
                return "en"
    
    def format_task_answers(self, tool_result: Union[str, List[Dict]], intention_result: Dict) -> str:
        """格式化任务和回答"""
        try:
            if isinstance(tool_result, str):
                # 兼容旧格式，单个工具结果
                return f"**Task 1**\n{intention_result.get('Intentions', [{}])[0].get('Content', '未知任务')}\n\n**Answer 1**\n{tool_result}"
            elif isinstance(tool_result, list):
                # 新格式，多个工具结果
                if not tool_result:
                    return "**Task 1**\n无任务\n\n**Answer 1**\n无工具执行结果"
                
                formatted_results = []
                intentions = intention_result.get("Intentions", [])
                
                for i, result in enumerate(tool_result):
                    task_num = i + 1
                    result_content = result.get("result", "")
                    
                    # 获取对应的意图内容作为任务描述
                    task_description = "未知任务"
                    if i < len(intentions):
                        task_description = intentions[i].get("Content", "未知任务")
                    
                    formatted_results.append(f"**Task {task_num}**\n{task_description}\n\n**Answer {task_num}**\n{result_content}")
                
                return "\n\n".join(formatted_results)
            else:
                return f"**Task 1**\n{intention_result.get('Intentions', [{}])[0].get('Content', '未知任务')}\n\n**Answer 1**\n{str(tool_result)}"
                
        except Exception as e:
            logger.error(f"格式化任务和回答失败: {e}")
            return f"**Task 1**\n任务格式化错误\n\n**Answer 1**\n{str(tool_result)}"

    async def generate_final_summary(self, messages: List[BaseMessage], intention_result: Dict, 
                             tool_result: Union[str, List[Dict]], original_query: str) -> Dict[str, Any]:
        """
        生成最终汇总回答（异步版本，支持token和成本统计）
        
        Args:
            messages: 完整的对话消息历史
            intention_result: 意图识别的结果
            tool_result: 工具执行的结果，可以是单个字符串或多个工具结果的列表
            original_query: 用户的原始查询
            
        Returns:
            Dict: 包含汇总回答和使用统计的字典
        """
        try:
            # 确定语言并选择对应的prompt
            response_language = self.determine_response_language(intention_result, original_query)
            
            if response_language == "en":
                prompt_template = FINAL_SUMMARY_PROMPT_EN
            elif response_language == "zh-cn":
                prompt_template = FINAL_SUMMARY_PROMPT_SIMP
            else:  # zh-hk
                prompt_template = FINAL_SUMMARY_PROMPT_TRAD
            
            # 格式化任务和回答
            task_answers = self.format_task_answers(tool_result, intention_result)
            
            # 构建最终的prompt
            final_prompt = prompt_template.format(task_answers=task_answers)
            
            # 调用异步LLM生成最终回答，使用第一个可用的模型
            response_content, usage_info = await self.llm_client.simple_chat(
                user_message="请基于以上信息生成最终回答",
                system_message=final_prompt,
                model_name=None  # 让服务自动选择模型
            )
            
            logger.info("最终汇总回答生成成功")
            
            # 构建包含使用统计的完整结果
            result = {
                "response": response_content,
                "usage_info": {
                    "request_id": usage_info.request_id,
                    "model_used": usage_info.model_used,
                    "provider": usage_info.provider,
                    "total_cost": usage_info.total_cost,
                    "currency": usage_info.currency,
                    "token_usage": usage_info.token_usage,
                    "cost_breakdown": usage_info.cost_breakdown,
                    "processing_time": usage_info.processing_time
                }
            }
            
            logger.info(f"最终汇总完成 - 模型: {usage_info.model_used}, "
                       f"Token: {usage_info.token_usage.get('total_tokens', 0)}, "
                       f"成本: {usage_info.total_cost}{usage_info.currency}")
            
            return result
                
        except Exception as e:
            logger.error(f"最终汇总回答生成失败: {e}")
            # 返回一个备用回答
            task_answers = self.format_task_answers(tool_result, intention_result)
            response_language = self.determine_response_language(intention_result, original_query)
            
            if response_language == "zh-cn":
                fallback_response = f"抱歉，在生成最终回答时遇到了问题。不过根据您的查询，这里是工具执行的结果：\n\n{task_answers}\n\n如需进一步帮助，请重新提问或联系我们的客服团队。"
            elif response_language == "zh-hk":
                fallback_response = f"抱歉，在生成最終回答時遇到了問題。不過根據您的查詢，這裡是工具執行的結果：\n\n{task_answers}\n\n如需進一步幫助，請重新提問或聯繫我們的客服團隊。"
            else:  # en
                fallback_response = f"Sorry, there was an issue generating the final response. However, based on your query, here's the tool execution result:\n\n{task_answers}\n\nFor further assistance, please ask again or contact our customer service team."
            
            return {
                "response": fallback_response,
                "usage_info": {
                    "request_id": "error",
                    "model_used": "fallback",
                    "provider": "local",
                    "total_cost": 0.0,
                    "currency": "CNY",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "cost_breakdown": {"input_cost": 0, "output_cost": 0, "total_cost": 0},
                    "processing_time": 0.0
                }
            }

# 创建工具实例
final_summary_tool_instance = FinalSummaryTool() 