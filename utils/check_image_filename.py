import os

image_path = "output/3d_spatial/images/"
prompt_file_name = "3d_spatial.txt"

image_model_list = ["sd_xl", "sd_cascade"]
mode_list = ["mode1", "mode2", "mode3"]
llm_list = ["llama", "gpt4", "beautiful_prompt", "magic_prompt"]
step_list = ["1", "2", "3"]

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

def check_and_fix_image_filename(folder, filename):
    """
    檢查並修正圖片檔案名稱格式
    1. 將 'description._mode_number_' 修正為 'description_mode_number_'
    2. 將 'description.txt_mode_number_' 修正為 'description_mode_number_'
    """
    need_fix = False
    new_filename = filename

    # 檢查並修正 '._'
    if '._' in filename:
        new_filename = new_filename.replace('._', '_')
        need_fix = True
    
    # 檢查並修正 '.txt_'
    if '.txt_' in filename:
        new_filename = new_filename.replace('.txt_', '_')
        need_fix = True

    if 'boo_' in filename:
        new_filename = new_filename.replace('boo_', 'book_')
        need_fix = True

    if need_fix:
        # 取得檔案完整路徑
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"已修正: {filename} -> {new_filename}")
            return True
        except Exception as e:
            print(f"修正失敗 {filename}: {str(e)}")
            return False
    return False

folder_list = get_folder_list(image_path, image_model_list, mode_list, llm_list, step_list)
#print(folder_list)

with open(prompt_file_name, 'r', encoding='utf-8') as file:
    prompt_list = file.readlines()
    prompt_list = [line.strip() for line in prompt_list]


for folder in folder_list:
    print(f"\n檢查資料夾: {folder}")
    image_list = os.listdir(folder)
    fixed_count = 0
    
    for image in image_list:
        if check_and_fix_image_filename(folder, image):
            fixed_count += 1
    
    if fixed_count > 0:
        print(f"已修正 {fixed_count} 個檔案")
    else:
        print("沒有發現需要修正的檔案")

for image_model in image_model_list:
    for mode in mode_list:
        image_list2 = []
        if mode == "mode1":
            path = os.path.join(image_path, image_model, mode)
            image_list = os.listdir(path)
            for line in prompt_list:
                for i in range(4):               
                    image_list2.append(line + "_" + mode + "_0000" + str(i+1) + "_.png")
            
            image_list.sort()
            image_list2.sort()
            if image_list2 == image_list:
                print(f"資料夾 {path} 的檔案名稱正確")
            else:
                print(f"資料夾 {path} 的檔案名稱不正確")


        if mode == "mode2":
            for llm in llm_list:
                path = os.path.join(image_path, image_model, mode, llm)
                image_list = os.listdir(path)
                image_list2 = []
                for line in prompt_list:
                    for i in range(4):               
                        image_list2.append(line + "_" + mode + "_0000" + str(i+1) + "_.png")
                
                image_list.sort()
                image_list2.sort()
                if image_list2 == image_list:
                    print(f"資料夾 {path} 的檔案名稱正確")
                else:
                    for i in range(len(image_list)):
                        if image_list[i] != image_list2[i]:
                            print(f"資料夾 {path} 的檔案名稱不正確: {image_list[i]} 和 {image_list2[i]}")
                    print(f"資料夾 {path} 的檔案名稱不正確")

        if mode == "mode3":
            for llm in llm_list:
                if llm == "llama":
                    for step in step_list:
                        path = os.path.join(image_path, image_model, mode, llm, step)
                        image_list = os.listdir(path)
                        image_list2 = []
                        for line in prompt_list:
                            for i in range(4):               
                                image_list2.append(line + "_" + mode + "_0000" + str(i+1) + "_.png")
                        
                        image_list.sort()
                        image_list2.sort()
                        if image_list2 == image_list:
                            print(f"資料夾 {path} 的檔案名稱正確")
                        else:
                            print(f"資料夾 {path} 的檔案名稱不正確")
                if llm == "gpt4" and image_model == "sd_cascade":
                    path = os.path.join(image_path, image_model, mode, llm)
                    image_list = os.listdir(path)
                    image_list2 = []
                    for line in prompt_list:
                        
                        for i in range(4):               
                            image_list2.append(line + "_" + mode + "_0000" + str(i+1) + "_.png")
                        
                    image_list.sort()
                    image_list2.sort()
                    if image_list2 == image_list:
                        print(f"資料夾 {path} 的檔案名稱正確")
                    else:
                        for i in range(len(image_list)):
                            if image_list[i] != image_list2[i]:
                                print(f"資料夾 {path} 的檔案名稱不正確: {image_list[i]} 和 {image_list2[i]}")
                        print(f"資料夾 {path} 的檔案名稱不正確")



