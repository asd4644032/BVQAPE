import json
from urllib import request, parse
import random
import os
from utils import  load_workflow_from_file, load_filename_to_description, extract_value_to_txt, queue_prompt, load_prompt_book,chat_llama
import argparse
import random
random.seed(42)


def main():
    parser = argparse.ArgumentParser(description="Generate images based on input parameters")
    parser.add_argument("--prompt", default="output/3d_spatial/prompt/gpt_4o/", help="Input prompt file")
    parser.add_argument("--workflow", default="workflow_api_step1.json", help="Workflow to use")
    parser.add_argument("--mode", type=int, default=2, help="Mode of operation")
    parser.add_argument("--output_path", default="C:/Users/lab929/Documents/ComfyUI-master/ComfyUI-master/output/T2I/", help="Output directory for generated images")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for image generation")
    parser.add_argument("--step", type=int, default=3, help="step number of iteration")
    args = parser.parse_args()

    # Your image generation logic here
    print(
        f"Generating images with: \n"
          f"Prompt: {args.prompt}\n"
          f"Workflow: {args.workflow}\n"
          f"Mode: {args.mode}\n"
          f"Output path: {args.output_path}\n"
          f"Batch size: {args.batch_size}"
          f"Step: {args.step}"
        )
    
    mode = args.mode
    mode_string = "_mode" + str(mode)
    step_string = str(args.step)
    
    prompt = load_workflow_from_file(args.workflow)

    # Load the prompt list based on the mode
    file_path = args.prompt
    if os.path.isdir(file_path):
        print("Workflow file path is a directory")
        description_list = load_filename_to_description(file_path, ".txt")
        if description_list == []:
            description_list = load_filename_to_description(file_path, ".json")
    elif os.path.isfile(file_path) and file_path.lower().endswith('.txt'):
        print("Workflow file path is a .txtfile")
        description_list = load_prompt_book(file_path)
    else:
        print("Invalid file path")
        return
    
    prompt_list = []
    if mode == 1:
        prompt_list = description_list
    elif mode == 2:
        count =0
        for file_name in description_list:
            with open(file_path + "/" + file_name, 'r', encoding='utf-8') as f:
                content = json.load(f)
                try:

                    prompt_list.append(content["prompt"])
                except:
                    print(f"Error loading {file_name}")
            count +=1
           # with open(file_path + "/" + file_name, 'r') as file:
           #     file_content = file.read()
           #     prompt_list.append(file_content)
            #prompt_list.append(extract_value_to_txt(file_path + "/" + file_name, key= "prompt", use_llm=True))
    elif mode == 3:
        count =0
        for file_name in description_list:
            with open(file_path + "/" + file_name, 'r', encoding='utf-8') as file:
                content = json.load(file)
                try:
                    prompt_list.append(content["prompt"])
                except:
                    print(f"Error loading {file_name}")
            count +=1
    output_folder = args.output_path + "sd_cascade/gpt_4o_new/"
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    else:
        print(f"Output folder already exists: {output_folder}")

    prefix = output_folder.split("output")[1][1:] + "/"
    png_filenames = load_filename_to_description(output_folder, ".png")

    for count, description in enumerate(description_list):
        if description[:-4] in png_filenames:
            print(f"Skipping {description} because it already exists")
            continue
        if mode ==2:
            description = description[:-4]
        if mode ==3:
            description = description[:-4]
        

        prompt_text = prompt_list[count]
        if prompt_text == None:
            prompt_text = description

        #set the text prompt for our positive CLIPTextEncode
        prompt["6"]["inputs"]["text"] = prompt_text

        #set the seed for our KSampler node
        prompt["3"]["inputs"]["seed"] = random.randint(0, 4294967295)
        prompt["33"]["inputs"]["seed"] = random.randint(0, 4294967295)

        #set image filename
        prompt["9"]["inputs"]["filename_prefix"] = prefix + description + mode_string

        queue_prompt(prompt)

        print("count: ", count)
        print("description: ", description)
        print("prompt_text: ", prompt_text) 
            





if __name__ == "__main__":
    main()


