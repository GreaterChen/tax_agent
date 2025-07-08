import re
import json
import asyncio
import logging
from typing import Any, Dict
from .ALLM import ALLM

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

with open('src/tools/examist/Dependencies/Books/Book_Index.txt','r', encoding='utf-8') as f:
    Book_Index = f.read()

SYS_PROMPT = '''
Role:
You are a Hong Kong Taxation Expert.
Input:
You will be given a Hong Kong Taxation Question plus an index for Hong Kong Taxation Text Book.
Task:
Your Task is to mark the scope of titles of the textbook, that is relevant, necessary and helpful to solving the question. Output Reasoning, Titles, and Request of Pages.
In Reasoning, output your analysis.
In Titles, output the Titles you think should select.
In Request of Pages,  output the corresponding pages of the titles you select. Strictly follow the format rule.
Rules:
1. Always Analysis deeply to the Question FIRST. Your Analysis should be put into the Reasoning field.
2. Always select the levels in the index as low as possible. Reason: Select unnecessary high pages when not needed, e.g. select 1.2 when should select 1.2.1, will result in time wasted in researching 1.2.2, 1.2.3..., THIS SHOULD BE AVOIDED.
3. Thinking of exceptions of regimes, or corner cases, key points, avoid missing important sections.
4. Output the pages as format. In the corresponding pages of the titles you select, the format must be:  "Request": ["12-14", "15-20", "18", "13"]. Must strictly follow the format. Otherwise cannot be recognized.
5. General Output in JSON:
{"Reasoning":"","Titles":"","","Request";["1","3-5"...]}
'''

system_msg = {
    "role": "system",
    "content": SYS_PROMPT
}

def clean_markdown(text: str) -> str:
    """
    清理 Markdown ```json ... ``` 包裹，以及常见前后缀。
    """
    text = re.sub(r"```json[\s\S]*?```", lambda m: m.group(0).strip('`'), text)
    text = re.sub(r"```([\s\S]*?)```", lambda m: m.group(1), text)
    prefixes = [r"^Sure!\s*Here's the result:\s*", r"^以下是.*?:"]
    for pre in prefixes:
        text = re.sub(pre, '', text)
    return text.strip()

def extract_json_with_keys(text: str, keys: set) -> Dict[str, Any]:
    """
    从文本提取 JSON 对象并检查包含指定 keys。失败时抛出 ValueError。
    """
    cleaned = clean_markdown(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and keys.issubset(data.keys()):
            return data
    except json.JSONDecodeError:
        logger.debug("整体 JSON 解析失败，尝试正则提取: %s", cleaned)
    match = re.search(r"\{[\s\S]*?\}", cleaned)
    if match:
        snippet = match.group(0)
        try:
            data = json.loads(snippet)
            if isinstance(data, dict) and keys.issubset(data.keys()):
                return data
        except json.JSONDecodeError:
            logger.debug("正则提取后 JSON 解析失败: %s", snippet)
    raise ValueError(f"无法提取包含 keys={keys} 的 JSON 对象，原始响应：{text}")

async def Select_Page(
    text: str,
    model_name: str = "o4-mini"
) -> Dict[str, Any]:

    user_message = f'''Question: \n  {text}\n TextBook Index \n {Book_Index}'''
    user_msg = {"role": "user", "content": user_message}
    messages = [system_msg, user_msg]
    target_keys = {"Reasoning", "Titles", "Request"}

    while True:
        reply = await ALLM(model_name, messages)
        try:
            data = extract_json_with_keys(reply, target_keys)
            return data
        except ValueError as e:
            logger.warning("提取失败：%s", e)
            messages = [{
                "role": "user",
                "content": (
                    "先前输出无法提取到有效 JSON，请确保返回的 JSON 对象格式如下，且字段齐全：\n"
                    '''{
  "Reasoning": "...",
  "Titles": "...",
  "Request": ["1", "3-5"]
}'''
                    f"\n原始返回如下：\n{reply}"
                )
            }]

# 同步调用示例
if __name__ == "__main__":
    async def main():

        text = '''
Mama's Bakery ("the Bakery") is a partnership business run by Hazel, Iris and MB Limited ("MBL"). Apart from the manufacture of bakery products, the Bakery was also engaged in retail sale at a shop in Yuen Long.
The Bakery's profits before tax for the year ended 31 December 2021, after adding back the accounting depreciation, was HK$432,850, which had reflected the following events that happened during the year:
(1) As part of the Bakery's expansion plan, it rented an additional shop premises ("the Premises") adjacent to its existing shop. The tenancy agreement of the Premises (with two copies) was for 3 years starting from 1 February 2021 at monthly rent of HK$40,000 for February 2021 to July 2022 (i.e. the first 18 months), and HK$42,000 for August 2022 to January 2024 (i.e. the remaining 18 months).
The tenancy agreement also provided for a 1-month rent-free period in August 2021.
The Bakery incurred HK$150,000 on the renovation of the Premises.
Hazel and Iris were each entitled to 45% of the Bakery's profits, and the remaining 10% was for MBL. Also, Hazel and Iris drew a monthly salary of HK$20,000 and HK$18,000 respectively. Hazel, who was a major baker of the Bakery, was diagnosed with a hand condition that required prolonged treatments. It was agreed that the Bakery would reimburse Hazel's medical expenses, totaling HK$75,294 for the year ("the Medical Expenses"). To facilitate the delivery of products, the Bakery acquired, on hire purchase terms, a motor vehicle which cost HK$250,000. Deposit of HK$25,000 was made. During the year ended 31 December 2021, total repayment made under the hire purchase contract was HK$37,294, of which HK$6,029 was interest payments.

Required:

Compute the stamp duty payable on the tenancy agreement in respect of the Premises.



'''
        result = await Select_Page(text)
        print(result)

    asyncio.run(main())
