import re
import json
import asyncio
import logging
from typing import Any, Dict

# 处理相对导入和绝对导入
try:
    from .ALLM import ALLM
    from .text import Law_List
except ImportError:
    from ALLM import ALLM
    from text import Law_List

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SYS_PROMPT = '''
Role:
You are a Hong Kong Taxation Expert.
Input:
You will receive:
    1. the Problem
    2. the Textbook content that are relevant
    3. the Rule notes -- a reminder from our paralegal, only 80% trustworthy, must verify
    4. the law texts -- the full texts of relevant laws.

Task:
You should first in the Reasoning field.
1. Point out all the key facts in this problem. ALL means ALL, you should analyse sentence by sentence. Fot unrelevant facts, you should also point out.
1.1 Any fact that might have a legal meaning.
1.2 Any circumstances that can cause different opinions and discussions.
1.3 The corresponding law.
2. Try to Understand the Question, beware of following circumsances:
    2.1 The Question might be asking for an amount, which is added of multiple amount -> You should compute each amount and sum up.
    2.2 The Question might be asking for possible tax treatment -> You should list all the possible tax treatments. List them according to certain methodologies. Analyze one by one. 
Then, as your most important deliverable,  give your "Final_Answer" in IRAC structure. 


Use markdown tables when calculation.
The final answer must be complete, as it is the only content our client can see.
Rule:
1. Output in JSON format:{"Reasoning":"","Final_Answer":"The final complete answer in IRAC structure, a whole str. "} ...... No "confidence" key needed.
2. Always think twice, think harder and deeper. Seek for all the corner cases, other duties, regimes or rights that might be triggered at the same time, e.g. other fix duties or duplicates...
3. Mention all the rules(ordinances, practice notes, cases...)  clearly in your answer.
4. Give exact answers.
'''

system_msg = {
    "role": "system",
    "content": SYS_PROMPT
}

def clean_markdown(text: str) -> str:
    """
    清理 Markdown ```json ... ``` 包裹，以及常见前后缀。
    """
    text = re.sub(r"```(?:json)?\n([\s\S]*?)\n```", r"\1", text)
    prefixes = [
        r"^Sure! Here's the result:\s*",
        r"^以下是.*?:\s*",
        r"^Here is the structured output:\s*"
    ]
    for pre in prefixes:
        text = re.sub(pre, '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_json_with_keys(text: str, keys: set) -> Dict[str, Any]:
    """
    从文本提取 JSON 对象并确保包含所有目标 keys。若缺失 key 则自动补齐为空值。
    """
    cleaned = clean_markdown(text)
    json_candidates = []

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            json_candidates.append(data)
    except json.JSONDecodeError:
        logger.debug("整体 JSON 解析失败，尝试正则提取: %s", cleaned)

    for match in re.finditer(r"\{[\s\S]*?\}", cleaned):
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                json_candidates.append(data)
        except json.JSONDecodeError:
            continue

    for candidate in json_candidates:
        matched_keys = keys.intersection(candidate.keys())
        if matched_keys:
            for key in keys:
                candidate.setdefault(key, "")
            return candidate

    raise ValueError(f"无法提取包含 keys={keys} 的 JSON 对象，原始响应：{text}")


async def Final(
    Question: str,
    Pages: str,
    Notes: str,
    Laws: str ,
    model_name: str = "o4-mini"
) -> Dict[str, Any]:

    user_message = f'''The Presented Question is: \n{Question}\n\nThe Relevant Textbook Content is: \n{Pages}\n\nThe Rule Notes: \n{Notes}\n\nThe Law Texts: \n{Laws}'''
    user_msg = {"role": "user", "content": user_message}
    messages = [system_msg, user_msg]
    target_keys = {"Reasoning", "Final_Answer"}

    for attempt in range(3):
        reply = await ALLM(model_name, messages)
        try:
            data = extract_json_with_keys(reply, target_keys)
            logger.info("成功提取 JSON：%s", data)
            return data
        except ValueError as e:
            logger.warning("第 %d 次提取失败：%s", attempt + 1, e)
            messages.append({
                "role": "user",
                "content": (
                    "Previous output failed. Please make sure the output contains keys (Reasoning, Final_Answer) in this JSON format:\n"
                    '''{
  "Reasoning": "Your reasoning",
  "Final_Answer": "Full IRAC answer string"
}'''
                    f"\nYour previous reply:\n{reply}"
                )
            })

    logger.error("多次尝试后仍无法提取 JSON，返回空结构")
    return {"Reasoning": "", "Final_Answer": ""}

# 示例同步调用入口
'''
Law=""
if __name__ == "__main__":
    async def main():
        Laws  =""
        Question = "Mama's Bakery ...（略）"
        Pages = "Page 658-661 ...（略）"
        Notes = "Rule notes from paralegal..."
        result = await Final(Question, Pages, Notes, Laws)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(main())

'''