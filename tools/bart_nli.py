import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
nli_model.to(device)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
premise = "a airplane behind a frog"
hypothesis = "A serene scene captures a thoughtful green frog perched in lush green grass, with a blurred image of a sleek Airbus ACJ319 aircraft resting on a distant field under a soft, hazy sky."

# run through model pre-trained on MNLI
x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_first')
logits = nli_model(x.to(device))[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 
entail_contradiction_logits = logits[:,[0,2]]
probs = entail_contradiction_logits.softmax(dim=1)
prob_label_is_true = probs[:,1]
print(prob_label_is_true)