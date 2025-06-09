import json
import os
import re

def convert_json_format(input_file, output_file):
    # 讀取輸入的 JSON 檔案

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pairs = re.findall(r'"Q(\d+)"\s*:\s*"(.+?)"\s*,\s*"A\1"\s*:\s*"(.+?)"', content, re.DOTALL)
    new_json = {}
    for index, question, answer in pairs:
        new_json[f"Q{index}"] = question.strip()
        new_json[f"A{index}"] = answer.strip()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_json, f, ensure_ascii=False, indent=2)
    # 創建一個新的字典來存儲格式化後的數據
    print(f"轉換完成。結果已保存到 {output_file}")

    
def process_folder(input_folder, output_folder):
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍歷輸入資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                convert_json_format(input_path, output_path)
            except json.JSONDecodeError:
                print(f"錯誤：{filename} 不是有效的 JSON 檔案，已跳過。")
            except Exception as e:
                print(f"處理 {filename} 時發生錯誤：{str(e)}")

# 使用示例
input_folder = 'output/3d_spatial/vqa_sets/'
output_folder = 'output_json_folder'
process_folder(input_folder, output_folder)
