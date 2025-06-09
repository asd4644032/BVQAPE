# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE



import os

import torch
import os
import json
import copy
import PIL.Image as Image
import glob
import random
import spacy

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

from experts.model_bank_3d import load_expert_model
from experts.obj_detection.generate_dataset_3d import Dataset, collate_fn
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import argparse
from experts.depth.generate_dataset import Dataset as Dataset_depth

obj_label_map = torch.load('dataset/detection_features.pt')['labels']

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

def parse_args():
    parser = argparse.ArgumentParser(description="UniDet evaluation.")
    parser.add_argument("--output_path", default="output/3d_spatial/detection/", help="Output directory for generated images")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4, 
        help="Batch size",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

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

    #load detection model
    model, transform = load_expert_model(task='obj_detection')
    accelerator = Accelerator(mixed_precision='fp16')

    save_path= args.output_path
    batch_size = args.batch_size

    for folder in folder_list:
        depth_path = folder.replace("images", "depth")
        dataset = Dataset(folder, depth_path, transform)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        model, data_loader = accelerator.prepare(model, data_loader)

        with torch.no_grad():
            for _, test_data in enumerate(tqdm(data_loader)):
                
                test_pred = model(test_data)

                for k in range(len(test_pred)):

                    instance_boxes = test_pred[k]['instances'].get_fields()['pred_boxes'].tensor  
                    instance_id = test_pred[k]['instances'].get_fields()['pred_classes']   
                    instance_score = test_pred[k]['instances'].get_fields()['scores']
                    img_path = test_data[k]['image_path']

                    file_name = img_path.split('\\')[-1].replace('.png', '.json')
                    output_folder = folder.replace("images", "detection")

                    os.makedirs(output_folder, exist_ok=True)

                    output_dict = {}
                    output_dict['boxes'] = instance_boxes.tolist()
                    output_dict['labels'] = instance_id.tolist()
                    output_dict['scores'] = instance_score.tolist()

                    with open(os.path.join(output_folder, file_name), 'w') as f:
                        json.dump(output_dict, f)




                
    #print('depth map saved in {}'.format(im_save_path))
    
    

if __name__ == '__main__':
    main()




