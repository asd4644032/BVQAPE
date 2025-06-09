import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import json
def get_folder_list(image_path, image_model_list, mode_list, llm_list, step_list):  
    folder_list = []
    for image_model in image_model_list:
        for mode in mode_list:
            if mode == "mode1":
                folder_list.append( os.path.join(image_path, image_model, mode))

            if mode == "mode2":
                for llm in llm_list:
                    folder_list.append(os.path.join(image_path, image_model, mode, llm))

            if mode == "mode3":
                for llm in llm_list:
                    if llm == "llama":
                        for step in step_list:
                            folder_list.append(os.path.join(image_path, image_model, mode, llm, step))
                    if llm == "gpt4" and image_model == "sd_cascade":
                        folder_list.append(os.path.join(image_path, image_model, mode, llm))
                    if llm == "deepseek-r1-14b":
                        for step in step_list:
                            folder_list.append(os.path.join(image_path, image_model, mode, llm, step))

    return folder_list



def main():
    parser = argparse.ArgumentParser(description="Evaluate the generated images")
    parser.add_argument("--output_path", default="output/3d_spatial/evaluate/", help="Output directory for generated images")
    parser.add_argument("--evaluate_metric", default="asthetics_scores", help="Evaluate metric")

    image_path = "output/3d_spatial/images/"
    bbox_path = "output/3d_spatial/detection/"
    depth_path = "output/3d_spatial/depth/"
    prompt_file_name = "3d_spatial.txt"
    image_model_list = ["sd_cascade"]
    #mode_list = ["mode1", "mode2", "mode3"]
    #llm_list = ["llama", "gpt4", "beautiful_prompt", "magic_prompt"]
    mode_list = ["mode3"]
    llm_list = ["deepseek-r1-14b"]
    step_list = ["3"]
    args = parser.parse_args()

    with open(prompt_file_name, 'r', encoding='utf-8') as file:
        prompt_list = file.readlines()
        prompt_list = [line.strip() for line in prompt_list]

    folder_list = get_folder_list(image_path, image_model_list, mode_list, llm_list, step_list)
    print(folder_list)

    if args.evaluate_metric == "clip_score":
        
        for folder in folder_list:
            output_json = {}

            with open(os.path.join(folder.replace("images", "evaluate"), "clip_score.txt"), "r") as file:
                score_list = file.readlines()
                score_list = [float(line.strip()) for line in score_list]

            image_list = os.listdir(folder)
            for i in range(len(image_list)):
                image_name = image_list[i]
                output_json["name" + str(i)] = image_name
                output_json["score" + str(i)] = score_list[i]
            
            with open(os.path.join(folder.replace("images", "evaluate"), "clip_score.json"), "w") as file:
                json.dump(output_json, file)

            print(f"Saved clip score to {folder.replace('images', 'evaluate')}/clip_score.json")
            
            
            #np.savetxt(os.path.join(output_folder, "clip_score.txt"), score_list)

    if args.evaluate_metric == "blip2_itc_itm":


        for folder in folder_list:
            with open(os.path.join(folder.replace("images", "evaluate"), "itc_scores.txt"), "r") as file:
                itc_score_list = file.readlines()
                itc_score_list = [float(line.strip()) for line in itc_score_list]

            with open(os.path.join(folder.replace("images", "evaluate"), "itm_scores.txt"), "r") as file:
                itm_score_list = file.readlines()
                itm_score_list = [float(line.strip()) for line in itm_score_list]

            image_list = os.listdir(folder)
            output_json = {}
            for i in range(len(image_list)):
                image_name = image_list[i]
                output_json["name" + str(i)] = image_name
                output_json["itc_score" + str(i)] = itc_score_list[i]
                output_json["itm_score" + str(i)] = itm_score_list[i]

            with open(os.path.join(folder.replace("images", "evaluate"), "blip2_itc_itm.json"), "w") as file:
                json.dump(output_json, file)

            print(f"Saved itc scores to {folder.replace('images', 'evaluate')}/itc_scores.json")



    if args.evaluate_metric == "blip2_vqa":
        for folder in folder_list:
            with open(os.path.join(folder.replace("images", "evaluate"), "vqa_scores.txt"), "r") as file:
                vqa_score_list = file.readlines()
                vqa_score_list = [float(line.strip()) for line in vqa_score_list]

            with open(os.path.join(folder.replace("images", "evaluate"), "soft_max_scores.txt"), "r") as file:
                soft_max_score_list = file.readlines()
                soft_max_score_list = [float(line.strip().split(",")[0]) for line in soft_max_score_list]

            output_json = {}
            image_list = os.listdir(folder)
            for i in range(len(image_list)):
                image_name = image_list[i]
                output_json["name" + str(i)] = image_name
                output_json["vqa_score" + str(i)] = vqa_score_list[i]
                output_json["soft_max_score" + str(i)] = soft_max_score_list[i]

            with open(os.path.join(folder.replace("images", "evaluate"), "blip2_vqa.json"), "w") as file:
                json.dump(output_json, file)

            print(f"Saved vqa scores to {folder.replace('images', 'evaluate')}/blip2_vqa.json")




    if args.evaluate_metric == "unidet":

        for folder in folder_list:
            detection_folder = os.path.join(bbox_path, folder.split("images/")[1])

            with open(os.path.join(folder.replace("images", "evaluate"), "unidet_scores.txt"), "r") as file:
                unidet_score_list = file.readlines()
                unidet_score_list = [float(line.strip()) for line in unidet_score_list]

            image_list = os.listdir(folder)
            detection_list = os.listdir(detection_folder)
            output_json = {}
            for i in range(len(image_list)):
                image_name = detection_list[i].replace(".json", ".png")
                output_json["name" + str(i)] = image_name
                output_json["unidet_score" + str(i)] = unidet_score_list[i]

            with open(os.path.join(folder.replace("images", "evaluate"), "unidet_scores.json"), "w") as file:
                json.dump(output_json, file)

            print(f"Saved unidet scores to {folder.replace('images', 'evaluate')}/unidet_scores.json")

    if args.evaluate_metric == "asthetics_scores":

        
        for folder in folder_list:
            with open(os.path.join(folder.replace("images", "evaluate"), "asthetics_scores.txt"), "r") as file:
                aesthetics_score_list = file.readlines()
                aesthetics_score_list = [float(line.strip()) for line in aesthetics_score_list]

            image_list = os.listdir(folder)
            output_json = {}
            for i in range(len(image_list)):
                image_name = image_list[i]
                output_json["name" + str(i)] = image_name
                output_json["aesthetics_score" + str(i)] = aesthetics_score_list[i]

            with open(os.path.join(folder.replace("images", "evaluate"), "aesthetics_scores.json"), "w") as file:
                json.dump(output_json, file)

            print(f"Saved aesthetics scores to {folder.replace('images', 'evaluate')}/aesthetics_scores.json")



if __name__ == "__main__":
    main()
