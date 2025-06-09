import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
mode_list = ["mode1", "mode3"]
image_model_list = ["sd_xl"]
#mode_list = ["mode2", "mode3"]
llm_list = ["llama"]
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

title_list = folder_list_to_title_list(folder_list)
batch_size = 4
clip_score_array = np.array([])
blip2_vqa_score_array = np.array([])
blip2_itc_score_array = np.array([])
blip2_itm_score_array = np.array([])
unidet_score_array = np.array([])
aesthetics_score_array = np.array([])
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
            new_blip2_vqa_max_array = (new_blip2_vqa_max_array +1) / 2



        else:
            new_clip_max_array = np.max(table_array[:, :, clip_score_index], axis=1)
            new_blip2_vqa_max_array = np.max(table_array[:, :, blip2_vqa_score_index], axis=1)
            new_blip2_itc_max_array = np.max(table_array[:, :, blip2_itc_score_index], axis=1)
            new_blip2_itm_max_array = np.max(table_array[:, :, blip2_itm_score_index], axis=1)
            new_unidet_max_array = np.max(table_array[:, :, unidet_score_index], axis=1)
            new_aesthetics_max_array = np.max(table_array[:, :, aesthetics_score_index], axis=1)
            new_blip2_vqa_max_array = (new_blip2_vqa_max_array +1) / 2

        clip_score_array = np.concatenate([clip_score_array, new_clip_max_array])
        blip2_vqa_score_array = np.concatenate([blip2_vqa_score_array, new_blip2_vqa_max_array])
        blip2_itc_score_array = np.concatenate([blip2_itc_score_array, new_blip2_itc_max_array])
        blip2_itm_score_array = np.concatenate([blip2_itm_score_array, new_blip2_itm_max_array])
        unidet_score_array = np.concatenate([unidet_score_array, new_unidet_max_array])
        aesthetics_score_array = np.concatenate([aesthetics_score_array, new_aesthetics_max_array])
            



#normalize x = (x - xmin) / (xmax - xmin)
clip_score_array = (clip_score_array - np.min(clip_score_array)) / (np.max(clip_score_array) - np.min(clip_score_array))
blip2_vqa_score_array = (blip2_vqa_score_array - np.min(blip2_vqa_score_array)) / (np.max(blip2_vqa_score_array) - np.min(blip2_vqa_score_array))
blip2_itc_score_array = (blip2_itc_score_array - np.min(blip2_itc_score_array)) / (np.max(blip2_itc_score_array) - np.min(blip2_itc_score_array))
blip2_itm_score_array = (blip2_itm_score_array - np.min(blip2_itm_score_array)) / (np.max(blip2_itm_score_array) - np.min(blip2_itm_score_array))
unidet_score_array = (unidet_score_array - np.min(unidet_score_array)) / (np.max(unidet_score_array) - np.min(unidet_score_array))
aesthetics_score_array = (aesthetics_score_array - np.min(aesthetics_score_array)) / (np.max(aesthetics_score_array) - np.min(aesthetics_score_array))


for i in range(4):
    clip_score_list.append(np.mean(clip_score_array[i*1000:(i+1)*1000]))
    blip2_vqa_score_list.append(np.mean(blip2_vqa_score_array[i*1000:(i+1)*1000]))
    blip2_itc_score_list.append(np.mean(blip2_itc_score_array[i*1000:(i+1)*1000]))
    blip2_itm_score_list.append(np.mean(blip2_itm_score_array[i*1000:(i+1)*1000]))
    unidet_score_list.append(np.mean(unidet_score_array[i*1000:(i+1)*1000]))
    aesthetics_score_list.append(np.mean(aesthetics_score_array[i*1000:(i+1)*1000]))



#plt.plot(np.sort(clip_score_array), label="clip_score")

#Probability density function
plt.hist(clip_score_array, bins=100, density=True)
plt.show()




#plot the score
plt.plot(clip_score_list, label="clip_score")
plt.plot(blip2_vqa_score_list, label="blip2_vqa_score")
plt.plot(blip2_itc_score_list, label="blip2_itc_score")
plt.plot(blip2_itm_score_list, label="blip2_itm_score")
plt.plot(unidet_score_list, label="unidet_score")
plt.plot(aesthetics_score_list, label="aesthetics_score")
plt.legend()
plt.show()

output_table = pd.DataFrame({
    "title": title_list,
    "clip_score": clip_score_list,
    "blip2_vqa_score": blip2_vqa_score_list,
    "blip2_itc_score": blip2_itc_score_list,
    "blip2_itm_score": blip2_itm_score_list,
    "unidet_score": unidet_score_list,
    "aesthetics_score": aesthetics_score_list
})




print(output_table)



