def Get_Hint(fields: list) -> str:
    """
    从Hints.txt文件中获取对应fields的提示文本
    
    Args:
        fields: 字段列表,如 ["A", "B"]
        
    Returns:
        str: 拼接的提示文本,如果没有匹配则返回空字符串
    """
    if not fields:
        return ""
        
    result = []
    try:
        with open("Dependencies/Hints/Hints.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith(">") and line[1:] in fields:
                # 如果当前行是字段标识(以>开头),获取到下一个字段标识前的所有内容
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.startswith(">"):
                        break
                    if next_line:  # 只添加非空行
                        result.append(next_line)
                    i += 1
                continue
            i += 1
                
        return "\n".join(result)
        
    except:
        # 任何错误都静默处理,直接返回空字符串
        return ""

a = Get_Hint(["B","D"]) 
print(a)