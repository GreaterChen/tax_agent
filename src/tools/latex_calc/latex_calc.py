from typing import Dict, Union, Any
from sympy import symbols, N, E, pi, factorial, sin, cos, tan, sqrt, exp
from sympy.parsing.latex import parse_latex
import logging

class LatexCalculator:
    """LaTeX计算器类，用于解析和计算LaTeX格式的数学表达式"""
    
    def __init__(self):
        """初始化计算器"""
        self.logger = logging.getLogger(__name__)
        # 预定义常用符号
        self.common_symbols = {
            'e': E,
            'pi': pi,
        }
    
    def _create_symbols(self, variables: Dict[str, float]) -> Dict[str, Any]:
        """为表达式中的变量创建符号对象
        
        Args:
            variables: 变量名和对应值的字典
            
        Returns:
            包含变量名和对应符号对象的字典
        """
        symbols_dict = {}
        for var_name in variables.keys():
            if var_name not in self.common_symbols:
                symbols_dict[var_name] = symbols(var_name)
        return symbols_dict

    def _parse_expression(self, latex_expr: str) -> Any:
        """解析LaTeX表达式
        
        Args:
            latex_expr: LaTeX格式的数学表达式
            
        Returns:
            解析后的SymPy表达式
        """
        try:
            expr = parse_latex(latex_expr)
            return expr
        except Exception as e:
            self.logger.error(f"解析LaTeX表达式时出错: {str(e)}")
            raise ValueError(f"无法解析LaTeX表达式: {latex_expr}")

    def calculate(self, latex_expr: str, values: Dict[str, float], precision: int = 6) -> Dict[str, Any]:
        """计算LaTeX表达式的值
        
        Args:
            latex_expr: LaTeX格式的数学表达式
            values: 变量值字典
            precision: 数值结果的精度（小数位数）
            
        Returns:
            包含计算结果的字典，包括符号形式和数值形式
        """
        try:
            # 创建符号
            sym_dict = self._create_symbols(values)
            
            # 解析表达式
            expr = self._parse_expression(latex_expr)
            
            # 准备变量值，包括预定义常量
            substitution_values = {}
            for name, symbol in sym_dict.items():
                if name in values:
                    substitution_values[symbol] = values[name]
            
            # 计算符号结果
            symbolic_result = expr.rhs.subs(substitution_values)
            
            # 计算数值结果
            numeric_result = N(symbolic_result, precision)
            
            return {
                "latex_expression": latex_expr,
                "variables": values,
                "symbolic_result": str(symbolic_result),
                # "numeric_result": float(numeric_result) if not numeric_result.is_complex else complex(numeric_result),
                "numeric_result": numeric_result,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"计算过程中出错: {str(e)}")
            return {
                "latex_expression": latex_expr,
                "variables": values,
                "error": str(e),
                "success": False
            }

    def validate_expression(self, latex_expr: str) -> bool:
        """验证LaTeX表达式是否可以被解析
        
        Args:
            latex_expr: LaTeX格式的数学表达式
            
        Returns:
            布尔值，表示表达式是否有效
        """
        try:
            self._parse_expression(latex_expr)
            return True
        except:
            return False

# 使用示例
if __name__ == "__main__":
    # 创建计算器实例
    calculator = LatexCalculator()
    
    # 测试表达式
    test_expr_1 = r"y = \sin(\theta) \times x^2 + \frac{n!}{2}"
    test_expr_2 = r"z = \frac{\sin(\pi x)}{\sqrt{n}} + e^{\theta}"
    
    test_values = {
        'x': 2,
        'n': 3,
        'theta': pi/4
    }
    
    # 计算结果
    result_1 = calculator.calculate(test_expr_1, test_values)
    result_2 = calculator.calculate(test_expr_2, test_values)
    
    # 打印结果
    print(f"\n变量值：")
    print(f"x = {test_values['x']}")
    print(f"n = {test_values['n']}")
    print(f"theta = π/4 ≈ {float(test_values['theta']):.4f} 弧度 (45度)")
    
    # 第一个表达式结果
    if result_1["success"]:
        print(f"\n表达式1: {result_1['latex_expression']}")
        print(f"符号结果: {result_1['symbolic_result']}")
        print(f"数值结果: {result_1['numeric_result']}")
    else:
        print(f"计算出错: {result_1['error']}")
    
    # 第二个表达式结果
    if result_2["success"]:
        print(f"\n表达式2: {result_2['latex_expression']}")
        print(f"符号结果: {result_2['symbolic_result']}")
        print(f"数值结果: {result_2['numeric_result']}")
    else:
        print(f"计算出错: {result_2['error']}") 