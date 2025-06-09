import argparse
import os
from utils import load_prompt_book, load_filename_to_description, save_text_to_file, load_filename_to_description, chat_llama, extract_value_to_txt
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Answer VQA")
    
    parser.add_argument("--task", type=str, default="prompt_crafting",
                        help="Task description")
    parser.add_argument("--prompt", type=str, default="3d_spatial.txt",
                        help="Prompt for the model")
    parser.add_argument("--output_path", type=str, default="output/3d_spatial/",
                        help="Path to save the output")


    parser.add_argument("--model", type=str, default="deepseek-r1-14b",
                        help="Model to use for generation")
    parser.add_argument("--image_folder", type=str, default="output/3d_spatial/images/sd_cascade/mode3/deepseek-r1-14b/2/",
                        help="Path to the image folder")

    parser.add_argument("--vqa_answer_folder", type=str, default="output/3d_spatial/vqa_answer/sd_cascade/deepseek-r1-14b/3/",
                        help="Path to the vqa answer folder")
    parser.add_argument("--step", type=int, default=3, help="step number of iteration")
    args = parser.parse_args()

    # Your code logic here
    print(f"Task: {args.task}")
    print(f"Prompt: {args.prompt}")
    print(f"Output path: {args.output_path}")
    print(f"Model: {args.model}")
    print(f"Step: {args.step}")
    # Load the prompt list based on the mode
    file_path = args.prompt
    if os.path.isdir(file_path):
        print("Prompt file path is a directory")
        description_list = load_filename_to_description(file_path, ".json")
    elif os.path.isfile(file_path) and file_path.lower().endswith('.txt'):
        print("Loading prompts from .txt file")
        description_list = load_prompt_book(file_path)
    else:
        print("Invalid prompt file path")
        return

    sentences = description_list
    task_output_folders = {
        "Generate VQA sets": "/vqa_sets",
        "prompt_crafting": "/prompt_crafting",
        "prompt_refinement": "/prompt_refinement",
        "Answer VQA": "/vqa_answer"
    }
    output_folder = args.output_path + task_output_folders.get(args.task, "") + "/deepseek-r1-14b/"
    #output_folder = args.output_path + "/vqa_sets/deepseek-r1-14b/"
    if args.task == "Answer VQA" or args.task == "prompt_refinement":
        output_folder = output_folder + "/" + str(args.step)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    else:
        print(f"Output folder already exists: {output_folder}")
        
    output_prompt = load_filename_to_description(output_folder, ".json")
    error_prompt = []
    count = 0
    for prompt_text in tqdm(sentences):
        if prompt_text in output_prompt:
            print(f"Skipping {prompt_text} because it already exists")
            continue
        #print(count)

        messages = []
        
        if args.task == "Generate VQA sets":
            messages.append(
                {
                "role" : "system",
                "content": "you are a professional VQA designer. I am analyzing the relationship between text and images. According to the user input, please help me design 5 English questions and answers pair to confirm whether the text and the image are consistent. the answer should be a simple word like yes or no."
                }
            )
            messages.append(
                {
                "role" : "user",
                "content": "user input : " + prompt_text
                }
            )
            messages.append(
                {
                "role" : "user",
                "content": "please output a json file with key values are : Q1, Q2, Q3, Q4, Q5, A1, A2, A3, A4, A5"
                }
            )
            response = chat_llama(messages, model = "deepseek-r1:14b")
            #print(response)
            """
            messages.append(
                {
                "role" : "user",
                "content": "please output a json file without any explanation"
                }
            )
            response = chat_llama(messages)
            """
        if args.task == "prompt_crafting":

            messages.append(
                {
                "role": "system",
                "content":  "You are a AI artist by applying stable diffusion. Your job is creating a  prompt for stable diffusion by user description. A well-constructed prompt should indeed adhere to the 60/30/10 guideline for describing the main object, background, and other parameters, respectively. Focus on describing the spatial relationship between objects and adjusting their size accordingly. There's some good example for your reference: Example 1:\\nmasterpiece, geometric isometric cube pattern, opalescent effects, pastel opal hues, shimmery astral glow, warm and cool tone interplay, deep dark shadows, cinema4d rendering quality, metallic texture with naturalistic fusion, glossy opalescent surface, cybernatural aesthetics, scifi with surreal naturalism, abstract environment with floating geometric shapes, lighter mood with epic visual effects, varying materials with opalescent shimmer, Vincent van Gogh-inspired starry effects, Claude Monet-inspired light diffusion, soft ambient lighting with dramatic contrasts, starlit sky illumination, ethereal landscape reflections, chrome finish with opalescent sheen, holographic projections with a serene ambiance, reflective surfaces with a soft opalescent glow, enhanced depth of field for an immersive experience Example 2:\\nmasterpiece, best quality, (Anime:1.4), 90s sci-fi graphic novel illustration, bunny-shaped aliens, delivering Easter eggs, flying saucer, comical theme, 2D vector art, flat design, vibrant color grading, astral glow, bold visible outlines, hyperstylized, Moebius for inventive alien designs, Hayao Miyazaki for blending fantasy with sci-fi, Jamie Hewlett for a bold graphic style, astral and cosmic effects inspiration from Jack Kirby, creativity unleashed\\n Example 3:\\nUltra detailed jellyfish ((with iridiscent glow)), sun rays piercing through the sea water, at the bottom of the sea, very glowy jellyfish, (((holographic))), (((rainbowish))), glass effect .\nThe importance of parts of the prompt can be up or down-weighted by enclosing the specified part of the prompt in brackets using the following syntax: (prompt:weight). E.g. if we have a prompt flowers inside a blue vase and we want the diffusion model to empathize the flowers we could try reformulating our prompt into: (flowers:1.2) inside a blue vase. Nested loops multiply the weights inside them, e.g. in the prompt ((flowers:1.2):.5) inside a blue vase flowers end up with a weight of 0.6. Using only brackets without specifying a weight is shorthand for (prompt:1.1), e.g. (flower) is equal to (flower:1.1). To use brackets inside a prompt they have to be escaped, e.g. \\(1990\\)."
                }           
            )
            messages.append(
                {
                "role": "user",
                "content": "Now give me a prompt example with following description : " + prompt_text  + "\nthe output should be JSON format.\\ntitle:\\nprompt:\\nexplanation:"
                }
            )

            response = chat_llama(messages, model = "deepseek-r1:14b")
            #print(response)
            '''
            messages.append(
                {
                "role" : "assistant",
                "content": response
                }
            )
            messages.append(
                {
                "role" : "user",
                "content": "the prompt is too short, please add more details to the prompt"
                }
            )
            response = chat_llama(messages, model = "deepseek-r1:14b")
            print(response)


            messages.append(
                {
                "role" : "user",
                "content": "please output a json file without any explanation"
                }
            )
    
        
            
        
            response = chat_llama(messages, model = "deepseek-r1:14b")            
            
            '''
            save_text_to_file(prompt_text.split(".json")[0] + ".txt", response, output_folder)

        if args.task == "Answer VQA":
            #a airplane behind a frog_mode1_00001_.png
            #a airplane behind a frog_mode2_00001_.png
            #a airplane behind a frog._mode3_00001_.png
            image_path = args.image_folder + "/" + prompt_text[:-4] + "_mode3_00001_.png"
            vqa_dict = {}
            question_list = []
            answer_list = []
            response_list = []
#            print("Answer VQA")
            qa_path = file_path + prompt_text 
            with open(qa_path, 'r') as fp:
                vqa_dict = json.load(fp)


            for i in range(5):
                messages = []
                Q_index = "Q" + str(i +1)

                A_index = "A" + str(i +1)
                #Q = extract_value_to_txt(qa_path, Q_index, use_llm = False)
                #A = extract_value_to_txt(qa_path, A_index, use_llm = False)
                Q = vqa_dict[Q_index]
                A = vqa_dict[A_index]



                question_list.append(Q)
                answer_list.append(A)
                
                messages.append(
                    {
                    "role" : "system",
                    "content": "Please answer the question based on the image. The answer should be a simple word like yes or no."
                    }
                )
                #print("Question : " + Q[0])
                
                messages.append(
                    {
                    "role" : "user",
                    "content": "Question : " + Q,
                    "images" : [image_path]
                    }
                )
                


                response = chat_llama(messages, model = 'llama3.2-vision')


                '''
                print(messages)
                print("image path : " + image_path)
                print("Question : " + Q)
                print("Answer : " + response)
                response_list.append(response)                
                '''


                vqa_dict[Q_index] = Q
                vqa_dict[A_index] = A
                vqa_dict["R" + str(i +1)] = response
                
            #print(vqa_dict)
#            print(question_list)
#            print(answer_list)
            '''
            with open(output_folder + "/" + prompt_text[:-4] + ".json", 'w') as fp:
                json.dump(vqa_dict, fp)
            '''
            with open(output_folder + "/" + prompt_text, 'w') as fp:
                json.dump(vqa_dict, fp)
             
        if args.task == "prompt_refinement":
            json_file = args.vqa_answer_folder + prompt_text
            with open(json_file, 'r') as fp:
                vqa_dict = json.load(fp)
            #print(vqa_dict)
            '''
            try:
                image_prompt = extract_value_to_txt(file_path + prompt_text, "prompt", use_llm = False)[0]
            except:
                image_prompt = prompt_text.split(".json")[0]
                error_prompt.append(prompt_text)            
            '''
            with open(file_path + prompt_text, 'r', encoding='utf-8') as fp:
                image_prompt = json.load(fp)["prompt"]
            
                            
            
            #image_prompt = prompt_text.split(".json")[0]


            messages.append(
                {
                "role" : "user",
                "content": "original prompt : " + image_prompt
                }
            )
            messages.append(
                {
                "role" : "user",
                "content": "description : " + prompt_text.split(".json")[0]
                }
            )
            for index in range(5):
                Q_index = "Q" + str(index + 1)
                A_index = "A" + str(index + 1)
                R_index = "R" + str(index + 1)
                messages.append(
                    {
                    "role" : "user",
                    "content": Q_index + " : " + vqa_dict[Q_index] + "\n" + A_index + " : " + vqa_dict[A_index] + "\n" + R_index + " : " + vqa_dict[R_index]
                    }
                )
            
            messages.append(
                {
                "role" : "user",
                "content": "According to the VQA answer, find out the reason why the answer is correct or incorrect. And then refine the original prompt to make generate better images. refined prompt will be used in stable diffusion. and the output should be JSON format.\\ntitle:\\nprompt:\\nexplanation:"
                }
            )

            #print(messages)
            response = chat_llama(messages, model = "deepseek-r1:14b")
            #print(response)

            '''
            messages = []
            messages.append(
                {
                "role" : "user",
                "content": response
                }
            )

            messages.append(
                {
                "role" : "user",
                "content": "please parse the response and output the refined prompt, the output should be JSON format.\\ntitle:\\nprompt:\\nexplanation:"
                }
            )
            response = chat_llama(messages)
            '''
            #print(response)
            save_text_to_file(prompt_text.split(".json")[0] + ".txt", response, output_folder)

        #save_text_to_file(prompt_text + ".txt", response, output_folder)


        count +=1

    print("end")
if __name__ == "__main__":
    main()