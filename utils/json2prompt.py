import json
import os

def extract_prompts_to_txt(txt_file_path):
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
            prompts = [obj.get('prompt', '') for obj in json_objects]

        return prompts



    except FileNotFoundError:
        print(f"Error: File not found at '{txt_file_path}'")

def load_output_filename(output_folder):
    # 取得所有檔案名稱
    filenames = os.listdir(output_folder)
    
    # 篩選出 *.txt 檔
    txt_filenames = [filename for filename in filenames if filename.endswith(".txt")]

    return txt_filenames
# Get input and output file paths from the user
output_folder = "output/"
#txt_file_path = input("Enter the path to your .txt file: ")
output_txt_file_path = "test.txt"

# Call the function to extract and save the prompts
#extract_prompts_to_txt(txt_file_path, output_txt_file_path)
txt_filenames = load_output_filename(output_folder)
prompt = ""
for name in txt_filenames:
    print(name)
    prompt = prompt + extract_prompts_to_txt("output/" + name)[0] + "\n"


with open(output_txt_file_path, 'w', encoding="utf-8") as output_file:
    output_file.write(prompt)
    print(f"Prompts extracted and saved to: {output_txt_file_path}")

