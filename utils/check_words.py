import torch
import spacy
nlp = spacy.load("en_core_web_sm")
obj_label_map = torch.load('dataset/detection_features.pt')['labels']

def get_objects(sentence):
    doc = nlp(sentence)
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


with open('3d_spatial.txt', 'r') as f:
    prompt_list = []
    for line in f:
        #remove /n
        line = line.replace('\n', '')
        prompt_list.append(line)

noun_list = []
count = 0
for prompt in prompt_list:

    obj1, obj2 = get_objects(prompt)

    
    noun_list.append(obj1)
    noun_list.append(obj2)

        
#check if the noun is in the obj_label_map

non_obj_list = []
for noun in noun_list:
    if noun not in obj_label_map and noun not in non_obj_list:
        non_obj_list.append(noun)


print(non_obj_list)




