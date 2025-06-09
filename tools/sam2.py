from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import os
from PIL import Image
from tqdm import tqdm
import json
import spacy
import cv2
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

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3

image_path = "output/3d_spatial/images/"
image_model_list = ["sd_xl", "sd_cascade"]
mode_list = ["mode1", "mode2", "mode3"]
llm_list = ["llama", "gpt4", "beautiful_prompt", "magic_prompt"]
step_list = ["1", "2", "3"]
folder_list = get_folder_list(image_path, image_model_list, mode_list, llm_list, step_list)
output_path = "output/3d_spatial/sam2/"

for folder in folder_list:
    img_list = os.listdir(folder)
    output_folder = os.path.join(output_path, folder.split("images/")[-1])
    os.makedirs(output_folder, exist_ok=True)
    for img in tqdm(img_list):
        prompt = img.split("_")[0]
        obj1, obj2 = get_objects(prompt)

        img_path = os.path.join(folder, img)
        json_path = os.path.join(folder.replace("images", "grounding_dino"), img.replace(".png", ".json"))
        with open(json_path, "r") as file:
            data = json.load(file)
        image = Image.open(img_path)
        boxes = data["boxes"]
        confidences = data["confidences"]
        class_names = data["class_names"]

        

        # build SAM2 image predictor
        sam2_checkpoint = SAM2_CHECKPOINT
        model_cfg = SAM2_MODEL_CONFIG
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        sam2_predictor.set_image(image)

        #select highest confidence boxes in two objects
        if obj1 in class_names:
            obj1_boxes = boxes[class_names.index(obj1)]
        else:
            obj1_boxes = []

        if obj2 in class_names:
            obj2_boxes = boxes[class_names.index(obj2)]
        else:
            obj2_boxes = []

        new_boxes = obj1_boxes + obj2_boxes
        if new_boxes == []:
            continue
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=new_boxes,
            multimask_output=False,
        )

        #save masks
        if len(masks) == 2:
            mask = masks[0]
            #save grayscale
            cv2.imwrite(os.path.join(output_folder, f"{img}_obj1.png"), mask[0]*255)

            mask = masks[1]
            cv2.imwrite(os.path.join(output_folder, f"{img}_obj2.png"), mask[0]*255)

        if len(masks) == 1:
            mask = masks[0]
            if obj1_boxes != []:
                cv2.imwrite(os.path.join(output_folder, f"{img}_obj1.png"), mask*255)
            else:
                cv2.imwrite(os.path.join(output_folder, f"{img}_obj2.png"), mask*255)



