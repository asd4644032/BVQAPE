import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torchvision.transforms as transforms
from torch import randint
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import CLIPProcessor, CLIPModel
import open_clip
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
import torch.nn as nn
import torch
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
import torch.nn.functional as F
from lavis.models.base_model import tile
import spacy
import json
import matplotlib.pyplot as plt
import cv2

nlp = spacy.load("en_core_web_sm")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
def determine_position(locality, box1, box2, iou_threshold=0.1,distance_threshold=150, depth_map=None, iou_threshold_3d=0.5):
    # Calculate centers of bounding boxes
    box1_center = ((box1['x_min'] + box1['x_max']) / 2, (box1['y_min'] + box1['y_max']) / 2)
    box2_center = ((box2['x_min'] + box2['x_max']) / 2, (box2['y_min'] + box2['y_max']) / 2)

    # Calculate horizontal and vertical distances
    x_distance = box2_center[0] - box1_center[0]
    y_distance = box2_center[1] - box1_center[1]

    # Calculate IoU
    x_overlap = max(0, min(box1['x_max'], box2['x_max']) - max(box1['x_min'], box2['x_min']))
    y_overlap = max(0, min(box1['y_max'], box2['y_max']) - max(box1['y_min'], box2['y_min']))
    intersection = x_overlap * y_overlap
    box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union = box1_area + box2_area - intersection
    iou = intersection / union

    # Determine position based on distances and IoU and give a soft score
    score=0
    if locality in ['next to', 'on side of', 'near']:
        if (abs(x_distance)< distance_threshold or abs(y_distance)< distance_threshold):
            score=1
        else:
            score=distance_threshold/max(abs(x_distance),abs(y_distance))
    elif locality == 'on the right of':
        if x_distance < 0:
            if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                score=1
            elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
        else:
            score=0
    elif locality == 'on the left of':
        if x_distance > 0:
            if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
                score=1
            elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
        else:
            score=0
    elif locality =='on the bottom of':
        if y_distance < 0:
            if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                score=1
            elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
    elif locality =='on the top of':
        if y_distance > 0:
            if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
                score=1
            elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
                score=iou_threshold/iou
    
    # 3d spatial relation
    elif locality == 'in front of':
        # use depth map to determine
        depth_A = depth_map[int(box1["x_min"]): int(box1["x_max"]), int(box1["y_min"]):int(box1["y_max"])]
        depth_B = depth_map[int(box2["x_min"]): int(box2["x_max"]), int(box2["y_min"]):int(box2["y_max"])]
        mean_depth_A = depth_A.mean()
        mean_depth_B = depth_B.mean()
        
        depth_diff = mean_depth_A - mean_depth_B
        # get the overlap of bbox1 and bbox2
        if iou > iou_threshold_3d:

            if (depth_diff > 0): # TODO set the threshold
                score=1
        
        elif iou < iou_threshold_3d and iou > 0:
            if (depth_diff > 0):
                score=iou/iou_threshold_3d
            else:
                score=0
        else:
            score = 0
        
    
    elif locality == 'behind' or locality == 'hidden':
        # use depth map to determine
        depth_A = depth_map[int(box1["x_min"]): int(box1["x_max"]), int(box1["y_min"]):int(box1["y_max"])]
        depth_B = depth_map[int(box2["x_min"]): int(box2["x_max"]), int(box2["y_min"]):int(box2["y_max"])]
        mean_depth_A = depth_A.mean()
        mean_depth_B = depth_B.mean()
        
        depth_diff = mean_depth_A - mean_depth_B
        # get the overlap of bbox1 and bbox2
        if iou > iou_threshold_3d:

            if (depth_diff < 0): # TODO set the threshold
                score=1
        
        elif iou < iou_threshold_3d and iou > 0:
            if (depth_diff < 0):
                score=iou/iou_threshold_3d
            else:
                score=0
        else:
            score = 0
     
        
    else:
        score=0
    return score

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
        
    #check if the noun is in the obj_label_map
    person = ['girl','boy','man','woman']
    phone  = ["phone"]
    computer = ["computer"]
    sofa = ["sofa"]
    non_object = ["painting", "bag", "wallet"] 
    if obj1 in person:
        obj1 = "person"
    if obj2 in person:
        obj2 = "person"
    if obj1 in phone:
        obj1 = "telephone"
    if obj2 in phone:
        obj2 = "telephone"
    if obj1 in computer:
        obj1 = 'tablet computer, tablet'
    if obj2 in computer:
        obj2 = 'tablet computer, tablet'
    if obj1 in sofa:
        obj1 = "couch"
    if obj2 in sofa:
        obj2 = "couch"    

    if obj1 == 'cow':
        obj1 = 'cattle, cow'
    if obj2 == 'cow':
        obj2 = 'cattle, cow'

    if obj1 == 'table':
        obj1 = 'table, desk'
    if obj2 == 'table':
        obj2 = 'table, desk'
    
    if obj1 == 'mouse':
        obj1 = 'mouse2, mouse'
    if obj2 == 'mouse':
        obj2 = 'mouse2, mouse'

    #non-object
    if obj1 in non_object:
        obj1 = ''
    if obj2 in non_object:
        obj2 = ''
    return obj1, obj2

def custom_rank_answers(self, samples, answer_list, num_ans_candidates):
        """
        Your custom implementation for _rank_answers

        Generate the first token of answers using decoder and select ${num_ans_candidates}
        most probable ones. Then select answers from answer list, which start with the probable tokens.
        Lastly, use the selected answers as the ground-truth labels for decoding and calculating LM loss.
        Return the answers that minimize the losses as result.

        """
        answer_candidates = self.tokenizer(
            answer_list, padding="longest", return_tensors="pt"
        ).to(self.device)
        answer_candidates.input_ids[:, 0] = self.tokenizer.bos_token_id

        answer_ids = answer_candidates.input_ids
        answer_atts = answer_candidates.attention_mask

        question_output, _ = self.forward_encoder(samples)
        question_states = question_output.last_hidden_state

        tokenized_question = samples["tokenized_text"]
        question_atts = tokenized_question.attention_mask

        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(
            start_ids,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            return_dict=True,
            reduction="none",
        )
        logits = start_output.logits[:, 0, :]  # first token's logit
        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token
        )
        topk_probs, topk_ids = prob_first_token.topk(num_ans_candidates, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100
        )

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, num_ans_candidates)
        question_atts = tile(question_atts, 0, num_ans_candidates)

        output = self.text_decoder(
            input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=targets_ids,
            return_dict=True,
            reduction="none",
        )

        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques, num_ans_candidates)

        max_topk_ids = log_probs_sum.argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]
        answers = [answer_list[max_id] for max_id in max_ids]
        topk_probs_ = topk_probs.detach().cpu().numpy()    
        probs = [(topk_probs_[i,0],topk_probs_[i,1]) if max_id==0 else (topk_probs_[i,1],topk_probs_[i,0]) for i,max_id in enumerate(max_ids)]
        return answers, probs

class VQAModel:
    '''BLIP-VQA model for computing DA-Scores'''
    def __init__(self, device='cuda'):
        # Load model and preprocessors
        self.blipvqa_model, self.blipvqa_vis_processors, self.blipvqa_txt_processors = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=device
        )
        
        ## Override the _rank_answers method with custom implementation
        self.blipvqa_model._rank_answers = custom_rank_answers.__get__(self.blipvqa_model, type(self.blipvqa_model))
        self.device = device

    def get_score(self, image, question):
        image_ = self.blipvqa_vis_processors["eval"](image).unsqueeze(0).to(self.device)
        question_ = self.blipvqa_txt_processors["eval"](question)
        
        with torch.no_grad():
            vqa_pred = self.blipvqa_model.predict_answers(
                samples={"image": image_, "text_input": question_}, 
                inference_method="rank", 
                answer_list=['yes','no'],
                num_ans_candidates=2
            )
        pos_score, neg_score = vqa_pred[1][0][0], vqa_pred[1][0][1]
        return pos_score, neg_score

def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m
def get_clip_score(image_path, prompt, model, processor):
    #image_path = "test.png"
    #prompt = "a airplane behind a frog"
    #score = get_clip_score(image_path, prompt)

    image = Image.open(image_path)
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image.detach().numpy()
    return logits_per_image[0][0]

def get_blip_vqa_score(image_path, prompt, model, neg_score_coef = 1):
    image = Image.open(image_path)
    question = f"Does this figure show ‘{prompt}’? Please answer yes or no."
    pos_score, neg_score = model.get_score(image, question)
    arr = np.array([pos_score, neg_score])
    soft_max_arr = np.exp(arr)/sum(np.exp(arr))

    diff_score = pos_score - neg_score_coef * neg_score
    return diff_score, soft_max_arr



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
    parser.add_argument("--evaluate_metric", default="unidet", help="Evaluate metric")

    image_path = "output/3d_spatial/images/"
    bbox_path = "output/3d_spatial/detection/"
    depth_path = "output/3d_spatial/depth/"
    prompt_file_name = "3d_spatial.txt"
    image_model_list = ["sd_xl", "sd_cascade"]
    #mode_list = ["mode1", "mode2", "mode3"]
    llm_list = ["gpt_4o_new"]
    mode_list = ["mode2"]
    step_list = []
    args = parser.parse_args()

    with open(prompt_file_name, 'r', encoding='utf-8') as file:
        prompt_list = file.readlines()
        prompt_list = [line.strip() for line in prompt_list]

    folder_list = get_folder_list(image_path, image_model_list, mode_list, llm_list, step_list)

    print(folder_list)

    if args.evaluate_metric == "clip_score":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        for folder in folder_list:
            score_list = []
            output_folder = os.path.join(args.output_path, folder.split("images/")[1])
            
            image_list = os.listdir(folder)
            for i, image in tqdm(enumerate(image_list)):
                text = prompt_list[i//4]
                image_path = os.path.join(folder, image)
                score = get_clip_score(image_path, text, model = model, processor = processor)
                score_list.append(score)
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            
            np.savetxt(os.path.join(output_folder, "clip_score.txt"), score_list)
            print(f"Saved clip score to {output_folder}/clip_score.txt")

    if args.evaluate_metric == "blip2_itc_itm":
        model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)


        for folder in folder_list:
            itm_scores = []
            itc_scores = []
            output_folder = os.path.join(args.output_path, folder.split("images/")[1])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            image_list = os.listdir(folder)

            for i in tqdm(range(len(image_list))):

                image = vis_processors["eval"](Image.open(folder+"/"+image_list[i]).convert("RGB")).unsqueeze(0)
                text = prompt_list[i//4]

                text = text_processors["eval"](text)
                img = image.to(device)
                text = text_processors["eval"](text)
                itm_output = model({"image": img, "text_input": text}, match_head="itm")
                itm_score = torch.nn.functional.softmax(itm_output, dim=1)
                #print(f'The image and text are matched with a probability of {itm_score[:, 1].item():.3%}')

                itc_score = model({"image": img, "text_input": text}, match_head='itc')
                #print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)

                itm_scores.append(itm_score[:, 1].item())
                itc_scores.append(itc_score.item())
            np.savetxt(os.path.join(output_folder, "itm_scores.txt"), itm_scores, delimiter=",")
            print(f"Saved itm scores to {output_folder}/itm_scores.txt")
            np.savetxt(os.path.join(output_folder, "itc_scores.txt"), itc_scores, delimiter=",")
            print(f"Saved itc scores to {output_folder}/itc_scores.txt")

    if args.evaluate_metric == "blip2_vqa":
        model = VQAModel(device=device)
        for folder in folder_list:
            vqa_scores = []
            soft_max_scores = []
            output_folder = os.path.join(args.output_path, folder.split("images/")[1])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            image_list = os.listdir(folder)
            for i in tqdm(range(len(image_list))):
                image_path = os.path.join(folder, image_list[i])
                prompt = prompt_list[i//4]
                diff_score, soft_max_arr = get_blip_vqa_score(image_path, prompt, model)
                vqa_scores.append(diff_score)
                soft_max_scores.append(soft_max_arr)
            np.savetxt(os.path.join(output_folder, "vqa_scores.txt"), vqa_scores, delimiter=",")
            print(f"Saved vqa scores to {output_folder}/vqa_scores.txt")
            np.savetxt(os.path.join(output_folder, "soft_max_scores.txt"), soft_max_scores, delimiter=",")
            print(f"Saved soft max scores to {output_folder}/soft_max_scores.txt")

    if args.evaluate_metric == "unidet":
        obj_label_map = torch.load('dataset/detection_features.pt')['labels']
        vocab_spatial_3d = ["in front of", "behind", "hidden"] 


        for folder in folder_list:
            unidet_scores = []
            output_folder = os.path.join(args.output_path, folder.split("images/")[1])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            detection_folder = os.path.join(bbox_path, folder.split("images/")[1])
            for i in tqdm(range(len(os.listdir(detection_folder)))):
                detection_file = os.path.join(detection_folder, os.listdir(detection_folder)[i])
                depth_file = os.path.join(depth_path, folder.split("images/")[1], os.listdir(detection_folder)[i]).replace(".json", ".png")
                image_file = os.path.join(folder, os.listdir(detection_folder)[i]).replace(".json", ".png")
                image = cv2.imread(image_file)
                depth_map = Image.open(depth_file).convert('L')
                depth_map = np.array(depth_map) / 255

                with open(detection_file, 'r') as f:
                    detection_data = json.load(f)
                boxes = np.array(detection_data['boxes']) / 480 * 1024
                labels = np.array(detection_data['labels'])
                scores = np.array(detection_data['scores'])
                prompt = os.listdir(detection_folder)[i].split("_")[0]
                prompt = prompt[:-1]


                #check if the prompt is in the prompt_list
                if prompt not in prompt_list:
                    print("incorrect")

                locality = None
                for word in vocab_spatial_3d:
                    if word in prompt:
                        locality = word
                        break

                obj = []  
                for num in range(len(boxes)):
                    obj_name = obj_label_map[labels[num]]  
                    obj.append(obj_name)

                obj1, obj2 = get_objects(prompt)
                if obj1 in obj and obj2 in obj:
                    # get obj_pos
                    obj1_index = obj.index(obj1)
                    obj2_index = obj.index(obj2)
                    obj1_bb = boxes[obj1_index]
                    obj2_bb = boxes[obj2_index]
                    obj1_score = scores[obj1_index]
                    obj2_score = scores[obj2_index]


                    # draw the bbox on the image
                    cv2.rectangle(image, (int(obj1_bb[0]), int(obj1_bb[1])), (int(obj1_bb[2]), int(obj1_bb[3])), (0, 0, 255), 2)
                    cv2.rectangle(image, (int(obj2_bb[0]), int(obj2_bb[1])), (int(obj2_bb[2]), int(obj2_bb[3])), (0, 255, 0), 2)

                    #add text on the image
                    cv2.putText(image, obj1, (int(obj1_bb[0] + 10), int(obj1_bb[1] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(image, obj2, (int(obj2_bb[0] + 10), int(obj2_bb[1] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    image_output_folder = os.path.join(output_folder, "images")
                    if not os.path.exists(image_output_folder):
                        os.makedirs(image_output_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(image_output_folder, image_file.split("\\")[-1]), image)

                    '''
                    print(obj1_bb)
                    print(obj2_bb)
                    square_coord1 =  ((obj1_bb[0], obj1_bb[2], obj1_bb[2], obj1_bb[0], obj1_bb[0]),
                 (obj1_bb[1], obj1_bb[1], obj1_bb[3], obj1_bb[3], obj1_bb[1]))
                    square_coord2 =  ((obj2_bb[0], obj2_bb[2], obj2_bb[2], obj2_bb[0], obj2_bb[0]),
                 (obj2_bb[1], obj2_bb[1], obj2_bb[3], obj2_bb[3], obj2_bb[1]))


                    plt.imshow(depth_map)
                    plt.plot(*square_coord1, c="r")
                    plt.plot(*square_coord2, c="b")

                    plt.show()
                    '''
                    
                    box_1 = {}
                    box_1["x_min"] = obj1_bb[0]
                    box_1["y_min"] = obj1_bb[1]
                    box_1["x_max"] = obj1_bb[2]
                    box_1["y_max"] = obj1_bb[3]


                    
                    box_2 = {}
                    box_2["x_min"] = obj2_bb[0]
                    box_2["y_min"] = obj2_bb[1]
                    box_2["x_max"] = obj2_bb[2]
                    box_2["y_max"] = obj2_bb[3]


                    score = 0.25 * obj1_score + 0.25 * obj2_score  # score = avg across two objects score
                    score += determine_position(locality, box_1, box_2, depth_map=depth_map) / 2

                    unidet_scores.append(score)
                else:
                    unidet_scores.append(0)
            np.savetxt(os.path.join(output_folder, "unidet_scores.txt"), unidet_scores, delimiter=",")
            print(f"Saved unidet scores to {output_folder}/unidet_scores.txt")

    if args.evaluate_metric == "asthetics_scores":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        amodel =get_aesthetic_model(clip_model="vit_l_14")
        amodel.eval()
        batch_size = 10
        
        for folder in folder_list:
            asthetics_scores = np.array([])
            image_list = os.listdir(folder)
            output_folder = os.path.join(args.output_path, folder.split("images/")[1])

            for i in tqdm(range(0, len(image_list), batch_size)):
                images = []
                batch_image_list = image_list[i:i+batch_size]
                for image_name in batch_image_list:
                    images.append(preprocess(Image.open(folder+"/"+image_name)))
                images = torch.stack(images)

                with torch.no_grad():
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    prediction = amodel(image_features)
                    asthetics_scores = np.append(asthetics_scores, prediction.cpu().numpy())
            np.savetxt(os.path.join(output_folder, "asthetics_scores.txt"), asthetics_scores, delimiter=",")
            print(f"Saved asthetics scores to {output_folder}/asthetics_scores.txt")

if __name__ == "__main__":
    main()
