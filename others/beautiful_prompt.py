from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print(torch.cuda.is_available())
import os

task = "magic_prompt"
#task = "beautiful_prompt"

file_path = "3d_spatial.txt"



with open(file_path, 'r') as file:
    prompt_list = file.readlines()

if task == "beautiful_prompt":
    output_folder = "output/3d_spatial/beautiful_prompt"
    tokenizer = AutoTokenizer.from_pretrained('alibaba-pai/pai-bloom-1b1-text2prompt-sd')
    model = AutoModelForCausalLM.from_pretrained('alibaba-pai/pai-bloom-1b1-text2prompt-sd').eval().cuda()
if task == "magic_prompt":
    output_folder = "output/3d_spatial/magic_prompt"
    tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
    model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion").eval().cuda()
    model.generation_config.pad_token_id = tokenizer.pad_token_id

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for prompt in prompt_list:
    raw_prompt = prompt[:-1]

#    input = f'Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {raw_prompt}\nOutput:'
    input = raw_prompt
    input_ids = tokenizer.encode(input, return_tensors='pt').cuda()
    outputs = model.generate(
        input_ids,
        max_length=90,
        min_length=20,
        do_sample=True,
        temperature=1.0,
        top_k=12,
        repetition_penalty=1.0,
        num_return_sequences=5,
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    )
    
    #prompts = tokenizer.batch_decode(outputs[:, input_ids.size(1):], skip_special_tokens=True)
    #prompts = [p.strip() for p in prompts]
    
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    #print(prompts)
    try:
        with open(os.path.join(output_folder, f"{raw_prompt}.txt"), 'w') as file:
            file.write(texts[0])
    except:
        print(f"Error with prompt: {raw_prompt}")
    