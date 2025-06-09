import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import spacy
import json
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")

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

    return folder_list

def get_objects(prompt):
    #get_objects(prompt = "a cat and a remote control") 
    doc = nlp(prompt)
    obj1= [token.text for token in doc if token.pos_=='NOUN'][0]
    obj2= [token.text for token in doc if token.pos_=='NOUN'][-1]    

    if obj1 == "front" or obj2 == "front":
        if 'key' in prompt:
            if obj1 == "front":
                obj1 = "key"
            if obj2 == "front":
                obj2 = "key"
        

    return obj1, obj2

image_path = "output/3d_spatial/images/"
image_path = "output/3d_spatial/images/"
bbox_path = "output/3d_spatial/detection/"
depth_path = "output/3d_spatial/depth/"
prompt_file_name = "3d_spatial.txt"
image_model_list = ["sd_xl", "sd_cascade"]
#mode_list = ["mode1", "mode2", "mode3"]
llm_list = ["gpt_4o_new"]
mode_list = ["mode2"]
step_list = []
folder_list = get_folder_list(image_path, image_model_list, mode_list, llm_list, step_list)
output_path = "output/3d_spatial/grounding_dino/"
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
BOX_THRESHOLD = 0.1
TEXT_THRESHOLD = 0.1

for folder in folder_list:
    img_list = os.listdir(folder)
    output_folder = os.path.join(output_path, folder.split("images/")[-1])
    os.makedirs(output_folder, exist_ok=True)
    for img in tqdm(img_list):
        img_path = os.path.join(folder, img)
        image = Image.open(img_path)
        prompt = img.split("_")[0]
        obj1, obj2 = get_objects(prompt)
        # Check for cats and remote controls
        # VERY important: text queries need to be lowercased + end with a dot
        text = f"{obj1}. {obj2}."

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]]
        )
        boxes = results[0]['boxes'].cpu().detach().numpy().tolist()
        confidences = results[0]['scores'].cpu().detach().numpy().tolist()
        class_names = results[0]['labels']    

        output_dict = {
            "obj1": obj1,
            "obj2": obj2,
            "prompt": prompt,
            "img_path": img_path,
            "boxes": boxes,
            "confidences": confidences,
            "class_names": class_names
        }
        
        with open(os.path.join(output_folder, img.replace(".png", ".json")), "w") as file:
            json.dump(output_dict, file)

