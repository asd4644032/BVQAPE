from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model

import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto", load_in_8bit=True) # load in int8

url = "http://images.cocodataset.org/val2017/000000039769.jpg"

img_path = "C:/Users/lab929/Downloads/test.jpg"
image = Image.open(img_path)

prompt = "Question: describe the image? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)