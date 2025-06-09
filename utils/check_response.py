import re
import json
import os
from utils import chat_llama
def extract_json_from_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    

    messages = []
    messages.append(
        {
        "role" : "user",
        "content": content
        }
    )

    messages.append(
        {
        "role" : "user",
        "content": "please parse the response and output the refined prompt, the output should be JSON format.\\ntitle:\\nprompt:\\nexplantion:"
        }
    )
    response = chat_llama(messages)
    content = response
    # 新增可處理 null 值的JSON格式正則表達式
    null_title_json_pattern = r'{\s*"title":\s*null\s*,\s*"prompt":\s*"[^"]*"\s*,\s*"explation":\s*(?:"[^"]*"|(?:\s*"[^"]*")*)\s*}'
    
    # 其他格式的正則表達式
    markdown_with_header_pattern = r'\*\*Title:\*\*\s*(.*?)\s*\*\*Prompt:\*\*\s*(.*?)\s*\*\*Explanation:\*\*\s*(.*?)(?=\n\n|$)'
    empty_title_json_pattern = r'{\s*"title":\s*""\s*,\s*"prompt":\s*"[^"]*"\s*,\s*"explation":\s*"[^"]*"\s*}'
    inline_backtick_pattern = r'`title:\s*(.*?)(?=\s*prompt:)prompt:\s*(.*?)(?=\s*explation:)explation:\s*(.*?)`'
    markdown_list_pattern = r'\* title:\s*(.*?)\s*\* prompt:\s*(.*?)\s*\* explanation:\s*(.*?)(?=\n\n|$)'
    markdown_bold_pattern = r'\*\*title:\*\*\s*(.*?)\s*\*\*prompt:\*\*\s*(.*?)\s*\*\*explation:\*\*\s*(.*?)(?=\n\n|$)'
    direct_json_pattern = r'{\s*"title":\s*"[^"]*",\s*"prompt":\s*"[^"]*",\s*"explanation":\s*"[^"]*"\s*}'
    backtick_pattern = r'`title:\s*(.*?)`\s*`prompt:\s*(.*?)`\s*`explation:\s*(.*?)`'
    json_pattern = r'```json\s*(.*?)\s*```'
    backup_pattern = r'```\s*({\s*".*?}\s*)```'
    

    
    def process_json_object(json_str):
        try:
            # 嘗試解析JSON
            json_obj = json.loads(json_str)
            
            # 確保 title 欄位存在
            if "title" not in json_obj:
                json_obj["title"] = ""
                

                
            return json_obj
        except json.JSONDecodeError:
            return None
    
    # 所有JSON格式
    json_formats = [
        null_title_json_pattern,
        empty_title_json_pattern,
        direct_json_pattern, 
        json_pattern, 
        backup_pattern
        
    ]
    
    # 嘗試所有格式
    for pattern in json_formats:
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            valid_jsons = []
            for match in matches:
                if isinstance(match, str):
                    json_obj = process_json_object(match)
                    if json_obj:
                        valid_jsons.append(json_obj)
            if valid_jsons:
                return valid_jsons
    
    return []


def save_jsons(jsons, output_file):
    # 如果只有一個JSON物件，直接儲存
    if len(jsons) == 1:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(jsons[0], f, ensure_ascii=False, indent=2)
    # 如果有多個JSON物，儲存為陣列
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(jsons, f, ensure_ascii=False, indent=2)

def process_folder(input_folder, output_folder):
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    count = 0
    # 遍歷輸入資料夾中的所有檔案
    for filename in os.listdir(input_folder):

        if filename.replace('.txt', '.json') in os.listdir(output_folder):
            count += 1
            continue

        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            # 將輸出檔案的副檔名改為.json
            output_filename = os.path.splitext(filename)[0] + '.json'
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                jsons = extract_json_from_text(input_path)
                if jsons:
                    save_jsons(jsons, output_path)
                    print(f"成功轉換 {filename} 並儲存至 {output_filename}")
                else:
                    print(f"在 {filename} 中未找到有效的JSON內容")
            except Exception as e:
                print(f"處理 {filename} 時發生錯誤: {str(e)}")
    print(f"總共跳過了 {count} 個檔案")
# 設定輸入和輸出資料夾路徑
input_folder = 'output/3d_spatial/prompt_refinement/sd_xl/3/'  # 請修改為你的輸入資料夾路徑
output_folder = 'output/3d_spatial/prompt/mode3/sd_xl/3/'  # 請修改為你的輸出資料夾路徑

# 執行批次處理
if __name__ == "__main__":
    process_folder(input_folder, output_folder)