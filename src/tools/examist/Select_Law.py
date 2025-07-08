import re
import json
import asyncio
import logging
from typing import Any, Dict
from .ALLM import ALLM
from .Get_Law_Indices import Get_Law_Indices

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Law Index 示例

SYS_PROMPT = '''
**Input:**
You will be given a Hong Kong Taxation Question, the corresponding Textbook Content, and a list of law titles and its selectible section paths.
**Aim:**
Analyze the laws relevant to the question, Explain the laws and their importance to the question, Convert the laws to JSON according to format rules.
**Task:**

1. **Analyze.** (generate in "Analyze" field)
   You should Analyze the Question and Read the Textbook Content Carefully, and then determine all Rules that are relevant to the question.
   You should focus mainly on the laws mentioned in the textbook, but also consider other laws that are related to the question.

2. **Notes.** (generate in "Notes" field)
   Then, as a major result of your work, you should edit Notes to explain the laws you selected: why they are important to the question, explain one by one in detail according to the textbook.
   You should ONLY explain the relationship, You are prohibited from answer the question or making the calculation!
3. **Laws** (a JSON format list of laws you selected)
   At last, you should make a JSON data to summarize the law list you selected, according to rules below:

**RULE 1 : The Structure of "Laws"**
For multiple laws, generate a list of dicts. Each dict contains `Title` and `Path`.
**Example:**

```json
"Laws":[{"Title":"The Title, same as provided","Path":"Level-Order,Level-Order"},{...More}]
```

The `Title` must be the same as in the provided law indices.
The `Path` must be one of the selectable path types provided in the law indices.
For example, if the law indices specify only Section or Section+Subsection, then you must not select Section+Subsection+Paragraph.

**RULE 2 : The Structure of "Paths"**
The `Path` is in a Level-Order format. You must not generate paths freely. Instead, you can only select levels from the given index.
E.g., when given 4 choices of levels, e.g. Section, Section+Subsection, Schedule, Schedule+Head,
You should only choose from the 4 based on your need, and add the orders accordingly.
**Example:**
`"Path":"Section-2,Subsection-1"` is valid; `"Path":"Schedule-First,Head-1"` is also valid.

**Output Format:**

```json
{"Analyze":"","Notes":"","Laws":[{"Title":"","Path":"Level-Order,Level-Order"},{...}]}
```

**Attention:**

1. Think harder.
2. Consider carefully all the corner cases, exceptions, and other minor consequences that most people might overlook.
3. The legal granularity must be selected as precisely as possible. If more precise levels are available, do not select higher-level sections unnecessarily.
   **Reason:** Selecting unnecessarily high-level titles when not needed (e.g., selecting Part 1 when you should select s.1) will result in time wasted researching unrelated sections (e.g., s.2, s.3, s.4) that are also under Part 1. **THIS SHOULD BE AVOIDED.** Cite `subsection` (ordinance) / `subparagraph` (notes) when appropriate.
4. Do not give final answers in the Notes.
**Common Mistakes**
Mistake 1: In law path, put lower levels in the level part.
e.g.: "Path":"Section-2(1)". WRONG! Invalid, because put (1) in the level of order, result in error. the level should purely be the order of the level.

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

    # 尝试直接解析
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            json_candidates.append(data)
    except json.JSONDecodeError:
        logger.debug("整体 JSON 解析失败，尝试正则提取: %s", cleaned)

    # 使用正则提取所有 JSON 对象候选
    for match in re.finditer(r"\{[\s\S]*?\}", cleaned):
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                json_candidates.append(data)
        except json.JSONDecodeError:
            continue

    # 选择包含至少一个目标 key 的最完整 JSON
    for candidate in json_candidates:
        matched_keys = keys.intersection(candidate.keys())
        if matched_keys:
            # 补齐缺失字段
            for key in keys:
                candidate.setdefault(key, "" if key in ("Analyze", "Notes") else [])
            return candidate

    raise ValueError(f"无法提取包含 keys={keys} 的 JSON 对象，原始响应：{text}")

async def Select_Law(
    Question: str,
    Pages: str,
    Law_Indices: str,
    model_name: str = "o4-mini"
) -> Dict[str, Any]:

    user_message = f'''The Presented Question is: \n{Question}\n\nThe Relevant Textbook Content is: \n{Pages}\n\nThe Title and designatable path \n{Law_Indices}'''
    user_msg = {"role": "user", "content": user_message}
    messages = [system_msg, user_msg]
    target_keys = {"Analyze", "Notes", "Laws"}

    for attempt in range(3):  # 最多尝试 3 次
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
                    "Previous output failed. Please make sure the output contains（Analyze、Notes and Laws）：\n"
                    '''{
  "Analyze": "...",
  "Notes": "...",
  "Laws": [{"Title": "IRO", "Path": "Section-14,Subsection-1"}]
}'''
                    f"\nYour original reply:\n{reply}"
                )
            })

    # 如果三次都失败，返回空结构，防止崩溃
    logger.error("多次尝试后仍无法提取 JSON，返回空结构")
    return {"Analyze": "", "Notes": "", "Laws": []}

# 示例同步调用入口
'''
Law_Indices = Get_Law_Indices()
if __name__ == "__main__":
    async def main():
        Question = "Mama's Bakery Stamp duty rates lease"
        Pages = "this question is connected with stamp duty ordinance first Sch. head1(a)（i）"
        result = await Select_Law(Question, Pages, Law_Indices)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(main())
 '''
