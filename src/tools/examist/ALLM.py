import asyncio
from typing import List, Dict, Any

import aiohttp


async def ALLM(
    model_name: str,
    messages: List[Dict[str, Any]],
) -> str:
    """
    异步调用 /query 接口并返回答案。

    Parameters
    ----------
    model_name : str
        指定使用的 LLM 模型名称，例如 'qwen-max-latest'。
    messages : List[Dict[str, Any]]
        OpenAI 兼容格式的消息数组。
    url : str, optional
        接口地址，默认文档给出的 URL。
    timeout : int, optional
        请求超时时间（秒）。

    Returns
    -------
    str
        多条答案用换行符拼接后返回。

    Raises
    ------
    RuntimeError
        当接口 code ≠ 200 时抛出。
    aiohttp.ClientError
        网络请求相关异常。
    """
    url = "http://8.216.81.217:8002/query"
    timeout: int = 100


    payload = {
        "messages": messages,
        "model_name": model_name,
    }

    timeout_cfg = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            res = await resp.json()

    if res.get("code") != 200:
        raise RuntimeError(f"LLM request failed: {res.get('msg', 'Unknown error')}")

    answers = res.get("data", {}).get("answers", [])
    return "\n".join(answers)


# ------------------- 示例 -------------------
if __name__ == "__main__":
    async def main():
        msgs = [
            {"role": "system", "content": "你是一个专业的税务顾问，请提供准确的税务建议。"},
            {"role": "user", "content": "什么是增值税？"}
        ]
        reply = await ALLM("o4-mini", msgs)
        print(reply)

    asyncio.run(main())
