"""计算器工具实现"""
from typing import Dict
from langchain_core.tools import tool
import ast
import operator

# 支持的运算符
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

class Calculator(ast.NodeVisitor):
    """计算器类,用于解析和计算数学表达式"""
    
    def visit_BinOp(self, node):
        """处理二元运算"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        return OPERATORS[type(node.op)](left, right)
    
    def visit_Num(self, node):
        """处理数字"""
        return node.n

@tool("calculator")
def calculate(operation: str) -> Dict[str, float]:
    """计算数学表达式的结果
    
    Args:
        operation: 要计算的数学表达式,如 "1 + 2 * (3 + 4^2)"
        
    Returns:
        Dict[str, float]: 包含计算结果的字典
    """
    try:
        # 解析表达式
        tree = ast.parse(operation, mode='eval')
        # 计算结果
        result = Calculator().visit(tree.body)
        return {"result": float(result)}
    except Exception as e:
        return {"error": f"计算错误: {str(e)}"} 