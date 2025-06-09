import os
from openai import OpenAI
api_key = "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"
client = OpenAI(api_key=api_key)

def save_text_to_file(filename, response, output_folder="output"):
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
            f.write(response)
        print(f"成功將內容儲存到 {file_path}")
    except IOError as e:
        print(f"儲存檔案時發生錯誤: {e}")

def load_prompt_book(book_file_path):
    with open(book_file_path, "r", encoding="utf-8") as file:
        # Read the lines and store them in a list
        sentences = file.readlines()

        # Remove any potential newline characters
        sentences = [line.strip() for line in sentences]

        return sentences

def load_output_filename(output_folder):
    # 取得所有檔案名稱
    filenames = os.listdir(output_folder)
    
    # 篩選出 *.png 檔
    png_filenames = [filename for filename in filenames if filename.endswith(".txt")]
    output_prompt = [element.split(".")[0] for element in png_filenames]

    return output_prompt


def chat_gpt(messages):

    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = messages
        )
    text = response.choices[0].message.content

    return text

file_path ="workflow_api_step1.json"


output_folder = "output/"
book_file_path = "complex_val.txt"

sentences = load_prompt_book(book_file_path)
output_prompt = load_output_filename(output_folder)
count = 0
for prompt_text in sentences:
    if prompt_text in output_prompt:
        continue
    print(count)

    messages = []
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are a AI artist by applying stable diffusion. Your job is creating a  prompt for stable diffusion by user description. A well-constructed prompt should indeed adhere to the 60/30/10 guideline for describing the main object, background, and other parameters, respectively. Focus on describing the spatial relationship between objects and adjusting their size accordingly.The text should be around 40 characters long. there's some good example for your reference: Example 1:\\nmasterpiece, geometric isometric cube pattern, opalescent effects, pastel opal hues, shimmery astral glow, warm and cool tone interplay, deep dark shadows, cinema4d rendering quality, metallic texture with naturalistic fusion, glossy opalescent surface, cybernatural aesthetics, scifi with surreal naturalism, abstract environment with floating geometric shapes, lighter mood with epic visual effects, varying materials with opalescent shimmer, Vincent van Gogh-inspired starry effects, Claude Monet-inspired light diffusion, soft ambient lighting with dramatic contrasts, starlit sky illumination, ethereal landscape reflections, chrome finish with opalescent sheen, holographic projections with a serene ambiance, reflective surfaces with a soft opalescent glow, enhanced depth of field for an immersive experience Example 2:\\nmasterpiece, best quality, (Anime:1.4), 90s sci-fi graphic novel illustration, bunny-shaped aliens, delivering Easter eggs, flying saucer, comical theme, 2D vector art, flat design, vibrant color grading, astral glow, bold visible outlines, hyperstylized, Moebius for inventive alien designs, Hayao Miyazaki for blending fantasy with sci-fi, Jamie Hewlett for a bold graphic style, astral and cosmic effects inspiration from Jack Kirby, creativity unleashed\\n Example 3:\\nUltra detailed jellyfish ((with iridiscent glow)), sun rays piercing through the sea water, at the bottom of the sea, very glowy jellyfish, (((holographic))), (((rainbowish))), glass effect .\nThe importance of parts of the prompt can be up or down-weighted by enclosing the specified part of the prompt in brackets using the following syntax: (prompt:weight). E.g. if we have a prompt flowers inside a blue vase and we want the diffusion model to empathize the flowers we could try reformulating our prompt into: (flowers:1.2) inside a blue vase. Nested loops multiply the weights inside them, e.g. in the prompt ((flowers:1.2):.5) inside a blue vase flowers end up with a weight of 0.6. Using only brackets without specifying a weight is shorthand for (prompt:1.1), e.g. (flower) is equal to (flower:1.1). To use brackets inside a prompt they have to be escaped, e.g. \\(1990\\). Note that the weight ratio should not higher than 1.21. e.g. (flower:1.5) is not suggested"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Now give me prompt example with following description : " + prompt_text  + "\nthe output should be JSON format.\\ntitle:\\nprompt:\\nexplation:"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "Here's a JSON formatted prompt based on your description:\n\n```json\n{\n  \"title\": \"Minimalist Chair and Table\",\n  \"prompt\": \"masterpiece, sleek black chair positioned to the left of a pristine white table, minimalist and elegant setting, single black chair, clear position to the left side, high-quality render, clean lines, soft ambient lighting, contrast between black chair and white table, minimalist decor, modern design\",\n  \"explanation\": \"This prompt focuses on rendering a single sleek black chair positioned to the left of a pristine white table in a minimalist and elegant setting. The main subjects are the chair and table, with the chair clearly placed on the left side to maintain clarity and balance in the composition. Soft ambient lighting and a clean, modern design enhance the overall aesthetic.\"\n}\n```\n\nThis prompt provides a clear directive for the composition (60% focus on the black chair and white table), the setting (30% minimalist and elegant), and other parameters (10% lighting, design elements, rendering quality)."
            }
        ]
        }
    ]    
    


    print(messages)
    response = chat_gpt(messages)

    save_text_to_file(prompt_text + ".txt", response, output_folder)



    if count == 700:
        break
    count +=1
