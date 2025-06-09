import json
import os
import pandas as pd
import numpy as np

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

def folder_list_to_title_list(folder_list):
    title_list = []
    for folder in folder_list:
        temp = folder.split("evaluate/")[1]
        temp = temp.split("\\")
        string = temp[0]
        for i in range(1, len(temp)):
            string += "_" + temp[i]
        title_list.append(string)
    return title_list



image_model = "sd_xl"
metric = "max"
evaluate_path = "output/3d_spatial/evaluate/"
if image_model == "sd_cascade":
    image_model_list = ["sd_cascade"]
else:
    image_model_list = ["sd_xl"]
mode_list = ["mode1", "mode2", "mode3"]
image_model_list = ["sd_xl","sd_cascade"]
#mode_list = ["mode2", "mode3"]
llm_list = ["llama", "gpt4", "beautiful_prompt", "magic_prompt", "deepseek-r1-14b"]
step_list = ["1", "2", "3"]
folder_list = get_folder_list(evaluate_path, image_model_list, mode_list, llm_list, step_list)
print(folder_list)


#load all json files
aesthetics_file_name = "aesthetics_scores.json"
unidet_file_name = "unidet_scores.json"
blip2_vqa_file_name = "blip2_vqa.json"
blip2_itc_itm_file_name = "blip2_itc_itm.json"
clip_score_file_name = "clip_score.json"

json_files = []
batch_size = 4


    
'''
for folder in folder_list:
    with open(os.path.join(folder, aesthetics_file_name), "r") as file:
        aesthetics_scores = json.load(file)
        file_name_list_1 = []
        aesthetics_score_list = []
        for i in range(4000):
            file_name_list_1.append(aesthetics_scores["name" + str(i)])
            aesthetics_score_list.append(aesthetics_scores["aesthetics_score" + str(i)])
    
    
    
    with open(os.path.join(folder, unidet_file_name), "r") as file:
        unidet_scores = json.load(file)
        file_name_list_2 = []
        unidet_score_list = []
        for i in range(4000):
            file_name_list_2.append(unidet_scores["name" + str(i)])
            unidet_score_list.append(unidet_scores["unidet_score" + str(i)])
    
    with open(os.path.join(folder, blip2_vqa_file_name), "r") as file:
        blip2_vqa_scores = json.load(file)
        file_name_list_3 = []
        blip2_vqa_score_list = []
        for i in range(4000):
            file_name_list_3.append(blip2_vqa_scores["name" + str(i)])
            blip2_vqa_score_list.append(blip2_vqa_scores["vqa_score" + str(i)])
    
    with open(os.path.join(folder, blip2_itc_itm_file_name), "r") as file:
        blip2_itc_itm_scores = json.load(file)
        file_name_list_4 = []
        blip2_itc_score_list = []
        blip2_itm_score_list = []
        for i in range(4000):
            file_name_list_4.append(blip2_itc_itm_scores["name" + str(i)])
            blip2_itc_score_list.append(blip2_itc_itm_scores["itc_score" + str(i)])
            blip2_itm_score_list.append(blip2_itc_itm_scores["itm_score" + str(i)])
    
    with open(os.path.join(folder, clip_score_file_name), "r") as file:
        clip_scores = json.load(file)
        file_name_list_5 = []
        clip_score_list = []
        for i in range(4000):
            file_name_list_5.append(clip_scores["name" + str(i)])
            clip_score_list.append(clip_scores["score" + str(i)])

    assert file_name_list_1 == file_name_list_2 == file_name_list_3 == file_name_list_4 == file_name_list_5

    summary_table = pd.DataFrame({
        "file_name": file_name_list_1,
        "aesthetics_score": aesthetics_score_list,
        "unidet_score": unidet_score_list,
        "blip2_vqa_score": blip2_vqa_score_list,
        "blip2_itc_score": blip2_itc_score_list,
        "blip2_itm_score": blip2_itm_score_list,
        "clip_score": clip_score_list
    })
    
    summary_table.to_csv(os.path.join(folder, "summary_table.csv"), index=False)


'''
    
'''
if image_model == "sd_cascade": 
    title_list = ["mode1", 'llama', 'gpt4', 'beautiful_prompt', 'magic_prompt', 'mode3 iter1', 'mode3 iter2', 'mode3 iter3', 'mode3 gpt4']
else:
    title_list = ["mode1", 'llama', 'gpt4', 'beautiful_prompt', 'magic_prompt', 'mode3 iter1', 'mode3 iter2', 'mode3 iter3']
'''
title_list = folder_list_to_title_list(folder_list)
batch_size = 4
clip_score_list = []
blip2_vqa_score_list = []
blip2_itc_score_list = []
blip2_itm_score_list = []
unidet_score_list = []
aesthetics_score_list = []

for folder in folder_list:
    summary_table = pd.read_csv(os.path.join(folder, "summary_table.csv"))
    table_array = summary_table.to_numpy()[:, 1:].astype(np.float32)
    table_array = table_array.reshape(table_array.shape[0]//batch_size, batch_size, table_array.shape[1])
    label_list = summary_table.columns.to_list()[1:]

    clip_score_index = label_list.index("clip_score")
    blip2_vqa_score_index = label_list.index("blip2_vqa_score")
    blip2_itc_score_index = label_list.index("blip2_itc_score")
    blip2_itm_score_index = label_list.index("blip2_itm_score")
    unidet_score_index = label_list.index("unidet_score")
    aesthetics_score_index = label_list.index("aesthetics_score")

    if metric == "max":
        if "mode3" in folder:
            step = folder[-1]
            if step == "1" or "gpt4" in folder:
                new_clip_max_array = np.max(table_array[:, :, clip_score_index], axis=1)
                new_blip2_vqa_max_array = np.max(table_array[:, :, blip2_vqa_score_index], axis=1)
                new_blip2_itc_max_array = np.max(table_array[:, :, blip2_itc_score_index], axis=1)
                new_blip2_itm_max_array = np.max(table_array[:, :, blip2_itm_score_index], axis=1)
                new_unidet_max_array = np.max(table_array[:, :, unidet_score_index], axis=1)
                new_aesthetics_max_array = np.max(table_array[:, :, aesthetics_score_index], axis=1)
            elif step == "2" or step == "3":
                new_clip_max_array = np.max(np.stack([np.max(table_array[:, :, clip_score_index], axis=1), new_clip_max_array]), axis=0)
                new_blip2_vqa_max_array = np.max(np.stack([np.max(table_array[:, :, blip2_vqa_score_index], axis=1), new_blip2_vqa_max_array]), axis=0)
                new_blip2_itc_max_array = np.max(np.stack([np.max(table_array[:, :, blip2_itc_score_index], axis=1), new_blip2_itc_max_array]), axis=0)
                new_blip2_itm_max_array = np.max(np.stack([np.max(table_array[:, :, blip2_itm_score_index], axis=1), new_blip2_itm_max_array]), axis=0)
                new_unidet_max_array = np.max(np.stack([np.max(table_array[:, :, unidet_score_index], axis=1), new_unidet_max_array]), axis=0)
                new_aesthetics_max_array = np.max(np.stack([np.max(table_array[:, :, aesthetics_score_index], axis=1), new_aesthetics_max_array]), axis=0)
        else:
            new_clip_max_array = np.max(table_array[:, :, clip_score_index], axis=1)
            new_blip2_vqa_max_array = np.max(table_array[:, :, blip2_vqa_score_index], axis=1)
            new_blip2_itc_max_array = np.max(table_array[:, :, blip2_itc_score_index], axis=1)
            new_blip2_itm_max_array = np.max(table_array[:, :, blip2_itm_score_index], axis=1)
            new_unidet_max_array = np.max(table_array[:, :, unidet_score_index], axis=1)
            new_aesthetics_max_array = np.max(table_array[:, :, aesthetics_score_index], axis=1)

        

        '''
        clip_max_value_list = []
        blip2_vqa_max_value_list = []
        blip2_itc_max_value_list = []
        blip2_itm_max_value_list = []
        unidet_max_value_list = []
        aesthetics_max_value_list = []

        for i in range(1000):
            clip_max_value = max(summary_table["clip_score"][i*batch_size], summary_table["clip_score"][i*batch_size+1], summary_table["clip_score"][i*batch_size+2], summary_table["clip_score"][i*batch_size+3])
            blip2_vqa_max_value = max(summary_table["blip2_vqa_score"][i*batch_size], summary_table["blip2_vqa_score"][i*batch_size+1], summary_table["blip2_vqa_score"][i*batch_size+2], summary_table["blip2_vqa_score"][i*batch_size+3])
            blip2_itc_max_value = max(summary_table["blip2_itc_score"][i*batch_size], summary_table["blip2_itc_score"][i*batch_size+1], summary_table["blip2_itc_score"][i*batch_size+2], summary_table["blip2_itc_score"][i*batch_size+3])
            blip2_itm_max_value = max(summary_table["blip2_itm_score"][i*batch_size], summary_table["blip2_itm_score"][i*batch_size+1], summary_table["blip2_itm_score"][i*batch_size+2], summary_table["blip2_itm_score"][i*batch_size+3])
            unidet_max_value = max(summary_table["unidet_score"][i*batch_size], summary_table["unidet_score"][i*batch_size+1], summary_table["unidet_score"][i*batch_size+2], summary_table["unidet_score"][i*batch_size+3])
            aesthetics_max_value = max(summary_table["aesthetics_score"][i*batch_size], summary_table["aesthetics_score"][i*batch_size+1], summary_table["aesthetics_score"][i*batch_size+2], summary_table["aesthetics_score"][i*batch_size+3])

            clip_max_value_list.append(clip_max_value)
            blip2_vqa_max_value_list.append(blip2_vqa_max_value)
            blip2_itc_max_value_list.append(blip2_itc_max_value)
            blip2_itm_max_value_list.append(blip2_itm_max_value)
            unidet_max_value_list.append(unidet_max_value)
            aesthetics_max_value_list.append(aesthetics_max_value)

        diff_clip_max_value = np.sum(np.abs(np.array(clip_max_value_list) - np.array(new_clip_max_value)))
        diff_blip2_vqa_max_value = np.sum(np.abs(np.array(blip2_vqa_max_value_list) - np.array(new_blip2_vqa_max_value)))
        diff_blip2_itc_max_value = np.sum(np.abs(np.array(blip2_itc_max_value_list) - np.array(new_blip2_itc_max_value)))
        diff_blip2_itm_max_value = np.sum(np.abs(np.array(blip2_itm_max_value_list) - np.array(new_blip2_itm_max_value)))
        diff_unidet_max_value = np.sum(np.abs(np.array(unidet_max_value_list) - np.array(new_unidet_max_value)))
        diff_aesthetics_max_value = np.sum(np.abs(np.array(aesthetics_max_value_list) - np.array(new_aesthetics_max_value)))
        diff = diff_clip_max_value + diff_blip2_vqa_max_value + diff_blip2_itc_max_value + diff_blip2_itm_max_value + diff_unidet_max_value + diff_aesthetics_max_value
        print(diff)
        
        '''
        clip_score_list.append(np.mean(new_clip_max_array))
        blip2_vqa_score_list.append(np.mean(new_blip2_vqa_max_array))
        blip2_itc_score_list.append(np.mean(new_blip2_itc_max_array))
        blip2_itm_score_list.append(np.mean(new_blip2_itm_max_array))
        unidet_score_list.append(np.mean(new_unidet_max_array))
        aesthetics_score_list.append(np.mean(new_aesthetics_max_array))
    if metric == "avg":   
        new_clip_max_array = np.mean(table_array[:, :, clip_score_index], axis=1)
        new_blip2_vqa_max_array = np.mean(table_array[:, :, blip2_vqa_score_index], axis=1)
        new_blip2_itc_max_array = np.mean(table_array[:, :, blip2_itc_score_index], axis=1)
        new_blip2_itm_max_array = np.mean(table_array[:, :, blip2_itm_score_index], axis=1)
        new_unidet_max_array = np.mean(table_array[:, :, unidet_score_index], axis=1)
        new_aesthetics_max_array = np.mean(table_array[:, :, aesthetics_score_index], axis=1)

        clip_score_list.append(np.mean(new_clip_max_array))
        blip2_vqa_score_list.append(np.mean(new_blip2_vqa_max_array))
        blip2_itc_score_list.append(np.mean(new_blip2_itc_max_array))
        blip2_itm_score_list.append(np.mean(new_blip2_itm_max_array))
        unidet_score_list.append(np.mean(new_unidet_max_array))
        aesthetics_score_list.append(np.mean(new_aesthetics_max_array))



'''
ours_folder_list = folder_list[5:7]
if metric == "max":
    clip_max_value_list = []
    blip2_vqa_max_value_list = []
    blip2_itc_max_value_list = []
    blip2_itm_max_value_list = []
    unidet_max_value_list = []
    aesthetics_max_value_list = []


for folder in ours_folder_list:
    summary_table = pd.read_csv(os.path.join(folder, "summary_table.csv"))

    if metric == "max":
        for i in range(1000):
            clip_max_value = max(summary_table["clip_score"][i], summary_table["clip_score"][i+1], summary_table["clip_score"][i+2], summary_table["clip_score"][i+3])
            blip2_vqa_max_value = max(summary_table["blip2_vqa_score"][i], summary_table["blip2_vqa_score"][i+1], summary_table["blip2_vqa_score"][i+2], summary_table["blip2_vqa_score"][i+3])
            blip2_itc_max_value = max(summary_table["blip2_itc_score"][i], summary_table["blip2_itc_score"][i+1], summary_table["blip2_itc_score"][i+2], summary_table["blip2_itc_score"][i+3])
            blip2_itm_max_value = max(summary_table["blip2_itm_score"][i], summary_table["blip2_itm_score"][i+1], summary_table["blip2_itm_score"][i+2], summary_table["blip2_itm_score"][i+3])
            unidet_max_value = max(summary_table["unidet_score"][i], summary_table["unidet_score"][i+1], summary_table["unidet_score"][i+2], summary_table["unidet_score"][i+3])
            aesthetics_max_value = max(summary_table["aesthetics_score"][i], summary_table["aesthetics_score"][i+1], summary_table["aesthetics_score"][i+2], summary_table["aesthetics_score"][i+3])

            clip_max_value_list.append(clip_max_value)
            blip2_vqa_max_value_list.append(blip2_vqa_max_value)
            blip2_itc_max_value_list.append(blip2_itc_max_value)
            blip2_itm_max_value_list.append(blip2_itm_max_value)
            unidet_max_value_list.append(unidet_max_value)
            aesthetics_max_value_list.append(aesthetics_max_value)
        

    if metric == "avg":   
        clip_score_list.append(summary_table["clip_score"].mean())
        blip2_vqa_score_list.append(summary_table["blip2_vqa_score"].mean())
        blip2_itc_score_list.append(summary_table["blip2_itc_score"].mean())
        blip2_itm_score_list.append(summary_table["blip2_itm_score"].mean())
        unidet_score_list.append(summary_table["unidet_score"].mean())
        aesthetics_score_list.append(summary_table["aesthetics_score"].mean())

new_clip_max_value_list = []
new_blip2_vqa_max_value_list = []
new_blip2_itc_max_value_list = []
new_blip2_itm_max_value_list = []
new_unidet_max_value_list = []
new_aesthetics_max_value_list = []

for i in range(1000):
    new_clip_max_value = max(clip_max_value_list[i], clip_max_value_list[i+1])
    new_blip2_vqa_max_value = max(blip2_vqa_max_value_list[i], blip2_vqa_max_value_list[i+1])
    new_blip2_itc_max_value = max(blip2_itc_max_value_list[i], blip2_itc_max_value_list[i+1])
    new_blip2_itm_max_value = max(blip2_itm_max_value_list[i], blip2_itm_max_value_list[i+1])
    new_unidet_max_value = max(unidet_max_value_list[i], unidet_max_value_list[i+1])
    new_aesthetics_max_value = max(aesthetics_max_value_list[i], aesthetics_max_value_list[i+1])

    new_clip_max_value_list.append(new_clip_max_value)
    new_blip2_vqa_max_value_list.append(new_blip2_vqa_max_value)
    new_blip2_itc_max_value_list.append(new_blip2_itc_max_value)
    new_blip2_itm_max_value_list.append(new_blip2_itm_max_value)
    new_unidet_max_value_list.append(new_unidet_max_value)
    new_aesthetics_max_value_list.append(new_aesthetics_max_value)



clip_score_list.append(np.mean(np.array(new_clip_max_value_list)))
blip2_vqa_score_list.append(np.mean(np.array(new_blip2_vqa_max_value_list)))
blip2_itc_score_list.append(np.mean(np.array(new_blip2_itc_max_value_list)))
blip2_itm_score_list.append(np.mean(np.array(new_blip2_itm_max_value_list)))
unidet_score_list.append(np.mean(np.array(new_unidet_max_value_list)))
aesthetics_score_list.append(np.mean(np.array(new_aesthetics_max_value_list)))

'''

blip2_vqa_score_list = (np.array(blip2_vqa_score_list) +1) / 2
blip2_vqa_score_list = blip2_vqa_score_list.tolist()


output_table = pd.DataFrame({
    "title": title_list,
    "clip_score": clip_score_list,
    "blip2_vqa_score": blip2_vqa_score_list,
    "blip2_itc_score": blip2_itc_score_list,
    "blip2_itm_score": blip2_itm_score_list,
    "unidet_score": unidet_score_list,
    "aesthetics_score": aesthetics_score_list
})





output_table.to_csv("new_summary_table.csv", index=False)



