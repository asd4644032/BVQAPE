
import json
from urllib import request, parse
import os
from PIL import Image
from io import BytesIO
import base64
import ollama
import re
from lavis.models import load_model_and_preprocess

def adjust_values(text):
    def adjust_number(match):
        full_match = match.group(0)
        content = match.group(1)
        value = float(match.group(2))
        
        if value > 1.1:
            # 將值縮放到1.0到1.1之間
            new_value = 1.0 + (value - 1.1) * (0.1 / (value - 1.0))
            new_value = round(new_value, 2)
            return f"({content}:{new_value})"
        return full_match

    # 使用正則表達式匹配括號內的內容和數值
    pattern = r'\((.*?):(\d+(\.\d+)?)\)'
    adjusted_text = re.sub(pattern, adjust_number, text)
    
    return adjusted_text

def load_workflow_from_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        return json.load(file)
    
def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)    

def load_prompt_book(book_file_path):
    with open(book_file_path, "r", encoding="utf-8") as file:
        # Read the lines and store them in a list
        sentences = file.readlines()

        # Remove any potential newline characters
        sentences = [line.strip() for line in sentences]

        return sentences    
    
def load_filename_to_description(output_folder, dtype ):
    # 取得所有檔案名稱
    filenames = os.listdir(output_folder)
    
    # 篩選出 *.png 檔
    if dtype  == ".png":
        png_filenames = [filename for filename in filenames if filename.endswith(".png")]
        output_prompt = [element.split("_")[0] for element in png_filenames]
    
    # 篩選出 *.txt 檔
    if dtype  == ".txt":
        txt_filenames = [filename for filename in filenames if filename.endswith(".txt")]
        output_prompt = txt_filenames
    
    if dtype  == ".json":
        json_filenames = [filename for filename in filenames if filename.endswith(".json")]
        output_prompt = json_filenames
       
    return output_prompt




def save_text_to_file(filename, text, output_folder="output"):
    """
    將 response 的內容儲存到 output_folder 資料夾下的 prompt_text 檔案中。

    Args:
        filename (str): 檔案名稱（包含副檔名，例如 .txt）。
        response (str): 要儲存的文字內容。
        output_folder (str, optional): 輸出資料夾的名稱。預設為 "output"。
    """

    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    file_path = os.path.join(output_folder, filename)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"成功將內容儲存到 {file_path}")
    except IOError as e:
        print(f"儲存檔案時發生錯誤: {e}")

def extract_value_to_txt(txt_file_path, key = "prompt", use_llm = True):
    """Extracts "prompt" values from JSON content in a .txt file and saves them to a new .txt file.

    Args:
        txt_file_path (str): The path to the input .txt file containing JSON content.
        output_txt_file_path (str): The path for the output .txt file.
    """

    try:
        with open(txt_file_path, 'r', encoding="utf-8") as txt_file:
            txt_content = txt_file.read()

            # Find all JSON objects within the .txt content
            json_objects = []
            start_index = 0
            start_index = txt_content.find('{', start_index)

            end_index = txt_content.find('}', start_index) + 1
            json_string = txt_content[start_index:end_index]

            try:
                json_objects.append(json.loads(json_string))

            except json.JSONDecodeError:
                pass  # Skip invalid JSON

            # Extract the "prompt" values
            value = [obj.get(key, '') for obj in json_objects]
            if value == []:
                if use_llm:
                    print("syntax error, apply llm to parse json file")
                    messages = [
                        {"role": "user", "content": "please give me the string of the value of prompt from json file without any explanation and don't add any comment: " + txt_content}
                    ]
                    value = chat_llama(messages)
                    value = [adjust_values(value)]
                    print(value)
                else:
                    print("syntax error, please check the file")
                    return value
            else:
                value = [adjust_values(value[0])]
        return value
    
    except FileNotFoundError:
        print(f"Error: File not found at '{txt_file_path}'")   

def read_raw_image(image_path, size = None):
    image = Image.open(image_path)
    if size != None:
        image = image.resize(size)
#    image.show()
    # Getting the base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image64        


def chat_llama(messages, model = 'llama3.1'):
    #messages = [{'role': 'user', 'content': 'Why is the sky blue?'}]

    response = ollama.chat(
        model = model, 
        messages = messages
    )
    #print(response['message']['content'])
    return response['message']['content']



