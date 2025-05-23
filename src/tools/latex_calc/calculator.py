from typing import TypedDict, Union
from langchain_core.tools import tool

class CalcResult(TypedDict):
    result: float

class CalcError(TypedDict):
    error: str

@tool
def calculate(operation: str) -> Union[CalcResult, CalcError]:
    """计算数学表达式的结果

    Args:
        operation: 数学表达式字符串，支持Python标准运算符：+ (加法), - (减法), * (乘法), / (除法), ** (幂运算), 例如 "2 + 3 * 5"

    Returns:
        result: 如果计算成功，返回结果值。
        error: 如果计算失败，返回错误信息。
    """
    try:
        result = eval(operation)
        return {"result": float(result)}
    except Exception as e:
        return {"error": f"计算错误: {str(e)}"}
