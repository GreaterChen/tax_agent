import os
import json

def Get_Law_Indices() -> str:
    """
    从Dependencies/Laws/目录下提取所有json文件的Indices字段并拼接
    
    Returns:
        str: 拼接后的Indices字符串
    """
    # 获取所有json文件路径
    law_dir = "Dependencies/Laws"
    json_files = [f for f in os.listdir(law_dir) if f.endswith('.json')]
    
    # 提取并拼接Indices
    indices_str = ""
    for json_file in json_files:
        file_path = os.path.join(law_dir, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "General_Metadata" in data and "Indices" in data["General_Metadata"]:
                    indices_str += data["General_Metadata"]["Indices"] + "\n"
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
            
    return indices_str.strip()


