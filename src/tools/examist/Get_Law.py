#Exanist1 组件
import json
import re
import os

def normalize(text):
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

def find_content(data, query):
    level_order_pairs = [tuple(q.strip().split('-')) for q in query.split(',') if '-' in q]
    normalized_query = [(normalize(level), normalize(order)) for level, order in level_order_pairs]
    results = []

    def traverse(node, path):
        if not isinstance(node, dict):
            return

        node_level = node.get("Level") or ''
        node_order = node.get("Order") or ''
        node_content = node.get("Content")

        normalized_node = (normalize(node_level), normalize(node_order))

        if normalized_node == normalized_query[-1]:
            matched = True
            if len(normalized_query) > 1:
                recent_path = [(normalize(lvl), normalize(ord)) for lvl, ord in path[-(len(normalized_query)-1):]]
                if recent_path != normalized_query[:-1]:
                    matched = False
            if matched and node_content:
                results.append({"Path": path + [(node_level, node_order)], "Content": node_content})

        for child in node.get("Children", []):
            traverse(child, path + [(node_level, node_order)])

    for item in data.get("Text_of_Law", []):
        traverse(item, [])

    return results

def Get_Section(query: str, data=None) -> str:
    if data is None:
        return "未提供数据，且未指定来源文件。"

    matched_content = find_content(data, query)
    if matched_content:
        return "\n\n".join(
            f"**Section**\nTitle of law and location of section: \n{data.get('General_Metadata', {}).get('Full_Name', '')} -> {' -> '.join(f'{lvl}-{ord}' for lvl, ord in item['Path'][1:] if lvl)}\nText of the section:\n{item['Content']}\n"
            for item in matched_content
        )
    else:
        return "未找到匹配的内容，请检查输入的单元是否正确。"

def Get_Law_Title(law_name: str) -> str:
    law_name_norm = normalize(law_name)
    folder = os.path.join("Dependencies", "Laws")

    if not os.path.isdir(folder):
        return "未找到法律数据文件夹：Dependencies/Laws"

    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            full_path = os.path.join(folder, filename)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    law_data = json.load(f)
                    meta = law_data.get("General_Metadata", {})
                    full_name = meta.get("Full_Name", "")
                    other_names = meta.get("Other_Names", [])
                    candidates = [full_name] + other_names
                    for name in candidates:
                        if normalize(name) == law_name_norm:
                            return full_path
            except Exception:
                continue
    return "未找到匹配的法律文件。"


def Get_Law(Selected_Laws: list) -> str:
    """
    处理多个法律条文查询请求,返回拼接后的结果字符串
    
    Args:
        Selected_Laws: 包含多个查询字典的列表,每个字典包含Title和Path
        
    Returns:
        str: 所有查询结果拼接的字符串
    """
    results = []
    
    for query in Selected_Laws:
        # 检查必要字段是否存在
        if not isinstance(query, dict) or "Title" not in query or "Path" not in query:
            continue
            
        title = query["Title"]
        path = query["Path"]
        
        # 获取对应的法律文件路径
        json_path = Get_Law_Title(title)
        
        # 如果找不到对应的法律文件则跳过
        if not os.path.exists(json_path):
            continue
            
        # 读取法律文件内容
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                law_data = json.load(f)
        except Exception:
            continue
            
        # 获取具体条文内容
        section_content = Get_Section(path, data=law_data)
        if section_content:
            results.append(section_content)
            
    
    # 拼接所有结果
    return "\n\n".join(results) if results else "----"
