import re
import json
import asyncio
import logging
from typing import Any, Dict

# 处理相对导入和绝对导入
try:
    from .ALLM import ALLM
except ImportError:
    from ALLM import ALLM

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 系统提示词（Prompt）
SYS_PROMPT = '''
Role:
You are a Hong Kong Taxation Expert.
Input: 
You will receive a question concerning Hong Kong Taxation.

Tasks:
Task 1 (put in reasoning field)
You need to analyse and determine whether the answer meets the following criteria:
1. Clarity Standard
The Question must be a specific Taxation Question that contains a specific problem to solve.
Assess whether the given question is clear enough to begin investigating and answering, and that it contains no obvious flaws or contradictions that would make a definitive answer impossible—for example, if the question indicates that a response must be based on certain facts but those facts are not provided, are incomplete or inconsistent, or if the question is self-contradictory.
If the case and question are clearly stated questions, then it meet the clarity standard, regardless of specific taxation questions it might contain.
2. One-Question Standard
Only Contains 1 Question.

Task 2: Follow The Steps (put in result & content field)
1. Determine if the input meets the Clarity Standard:
   * The Clarity Standard means the input must contain at least one clear, complete, and understandable question, with sufficient context.

2. If the input does NOT meet the Clarity Standard:
   * Set the "result" field to "fail".
   * Point out the reason why the input does not meet the Clarity Standard.
   * Output format:
     {"result":"fail","content":"reason why the input does not meet the Clarity Standard"}

3. If the input DOES meet the Clarity Standard:

   * Set the "result" field to "pass".

4. If the input contains multiple questions:

   * Extract only the **first** question.
   * Include all complete case facts, background, and context related to that first question.
   * Do NOT omit any information or change any terminology.
   * Output format:
     {"result":"pass","content":"the first question, with complete case fact, context and related information"}

5. If the input contains only one question:

   * Copy the full question into the "content" field as-is.
   * Output format:
     {"result":"pass","content":"the question"}

Task 3 Determine the field the question belongs to.
Cross-field questions are common, you should consider multiple fields.
The field information might be in the question, or in the fact. Even the question point to one field, but the fact might point to another field.
You should decide which fields this question belongs to, you can always select multiple fields.

Put only corresponding codes in.
    Code and Field
    A Hong Kong Tax System (Principle of Taxation, Basic Law, Sources and Interpretation) and Tax Admin (IRD, Assessments and Payment， Objection, Holdover, Appeal and Error or Omission Claim, Offences and Penalties, Field Audit and Investigation)
    B Hong Kong Profit Tax
    C Hong Kong Salary Tax
    D Hong Kong Property Tax
    E Personal Assessment
    F Hong Kong Stamp Duty
    G Hong Kong Profits Tax Liabilities for Cross-border Transactions
    H China Tax (Tax System and Administration in Mainland China)
    I Tax Planning and Anti-avoidance
    J Transfer Pricing

    If you are not 100% sure about the field, e.g., you are not sure between profit tax and property tax, You are encouraged to put both in the fields,e.g. ["B","D"] it is the safest way.
    
OUTPUT:
{"reasoning":"task1","result":"fail/pass","content":"","fields":[""]}
'''

# 系统消息结构
system_msg = {
    "role": "system",
    "content": SYS_PROMPT
}

# 清理 Markdown 内容（去除 ```json ``` 等包裹）
def clean_markdown(text: str) -> str:
    text = re.sub(r"```json[\s\S]*?```", lambda m: m.group(0).strip('`'), text)
    text = re.sub(r"```([\s\S]*?)```", lambda m: m.group(1), text)
    prefixes = [r"^Sure!\s*Here's the result:\s*", r"^以下是.*?:"]
    for pre in prefixes:
        text = re.sub(pre, '', text)
    return text.strip()

# 提取 JSON 并校验包含字段
def extract_json_with_keys(text: str, keys: set) -> Dict[str, Any]:
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

# 主处理函数
async def Filter(text: str, model_name: str = "o4-mini") -> Dict[str, Any]:
    user_message = f"The presented question is:\n  {text}"
    user_msg = {"role": "user", "content": user_message}
    messages = [system_msg, user_msg]
    target_keys = {"result", "content", "fields"}

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
                    f"先前输出无法提取到有效 JSON，请仅返回包含 'result', 'content', 'fields' 的 JSON 对象：\n{reply}"
                )
            }]


if __name__ == "__main__":
    async def main():
        text = '''
        Betty is a fresh university graduate majoring in visual arts. In late 2020, her father passed
away and she inherited a residential property located in Cheung Chau
("Cheung Chau Property"). The Cheung Chau Property was let out at a monthly rent of
HK$9,800 inclusive of rates and government rent throughout the year of assessment 2021/22.
On 30 May 2021, Betty pledged the Cheung Chau Property to a bank and obtained a
mortgage loan to acquire a car parking space in Shatin ("Shatin CPS") for investment
purpose. She could only let out the Shatin CPS from 1 February 2022 at a monthly rent of
HK$3,800.
In August 2021, Betty started her sole-proprietorship business, namely B's Art House, as a
painting instructor. She rented a studio in Kwun Tong from the landlord to conduct
the painting classes. Up to 31 March 2022, she made profits of HK$180,000 (after all
necessary tax adjustments) from the painting classes. Moreover, she found that there was
surplus space in her studio, thus she entered into a lease agreement to let part of the studio
to her friend at a monthly rent of HK$2,700 since 1 November 2021.
During the year of assessment 2021/22, Betty paid interest of HK$10,000 for the mortgage
loan (i.e. HK$1,000 × 10 months). She also paid government rent of HK$2,000 and HK$800
for the Cheung Chau Property and the Shatin CPS respectively (rates of both properties were
fully waived).

Required:

Betty wonders if B's Art House qualifies for the two-tiered profits tax rates. Identify and analyse the relevant considerations for Betty.
        '''
        result = await Filter(text)
        print(json.dumps(result, indent=2))

    asyncio.run(main())
