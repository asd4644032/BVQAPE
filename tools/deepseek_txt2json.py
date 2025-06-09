import re
import json
import os

def process_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正則表達式提取JSON和COT

    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    
    cot_pattern = r'<think>\s*([\s\S]*?)\s*</think>'
    

    # 提取所有JSON匹配項，取最後一個
    json_matches = re.findall(json_pattern, content, re.MULTILINE)
    json_data = None
    if json_matches:
        json_content = json_matches[-1]  # 取最後一個JSON內容
        try:
            json_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"JSON解析錯誤 in {file_path}: {str(e)}")
            print(f"問題內容: {json_content}")
    else:
        json_pattern = r'\{\s*"title":\s*".*?",\s*"prompt":\s*".*?",\s*"explanation":\s*".*?"\s*\}'
        json_matches = re.findall(json_pattern, content, re.MULTILINE)
        json_data = None
        if json_matches:
            json_content = json_matches[-1]  # 取最後一個JSON內容
            try:
                json_data = json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"JSON解析錯誤 in {file_path}: {str(e)}")
                print(f"問題內容: {json_content}")
        else:
            print(f"在檔案中未找到JSON內容: {file_path}")
        


    
    # 提取COT
    cot_match = re.search(cot_pattern, content, re.MULTILINE)
    cot_content = None
    if cot_match:
        cot_content = cot_match.group(1).strip()
    else:
        print(f"在檔案中未找到COT內容: {file_path}")
    
    return json_data, cot_content

def check_output_json_result(output_json_folder):
    for filename in os.listdir(output_json_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(output_json_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            #if json_data['title'] == '':
                #print(f"在檔案中未找到JSON內容: {file_path}")
            if json_data['prompt'] == '':
                print(f"在檔案中未找到prompt: {file_path}")
            
            #if json_data['explanation'] == '':
                #print(f"在檔案中未找到explanation: {file_path}")
    



# 設定輸入和輸出資料夾路徑
input_folder = 'output/3d_spatial/prompt_crafting/deepseek-r1-14b/'  # 請修改為你的輸入資料夾路徑
output_cot_folder = 'output/3d_spatial/prompt_crafting/deepseek-r1-14b/cot/'  # 請修改為你的輸出資料夾路徑
output_json_folder = 'output/3d_spatial/prompt_crafting/deepseek-r1-14b/json/'  # 請修改為你的輸出資料夾路徑


# 執行批次處理
if __name__ == "__main__":
    # 確保輸出資料夾存在
    if not os.path.exists(output_cot_folder):
        os.makedirs(output_cot_folder)
    if not os.path.exists(output_json_folder):
        os.makedirs(output_json_folder)
    

    # 遍歷輸入資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            base_filename = os.path.splitext(filename)[0]
            
            # 處理檔案並獲取JSON和COT內容
            json_data, cot_content = process_txt_file(input_path)
            
            # 儲存JSON檔案
            if json_data:
                json_output_path = os.path.join(output_json_folder, f"{base_filename}.json")
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

                print(f"成功轉換JSON並儲存至 {json_output_path}")
            
            # 儲存COT檔案
            if cot_content:
                cot_output_path = os.path.join(output_cot_folder, f"{base_filename}_cot.txt")
                with open(cot_output_path, 'w', encoding='utf-8') as f:
                    f.write(cot_content)

                print(f"成功轉換COT並儲存至 {cot_output_path}")
    check_output_json_result(output_json_folder)
    
    