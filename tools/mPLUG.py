from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

model_id = 'iic/mplug_image-captioning_coco_base_en'
pipeline_caption = pipeline(Tasks.image_captioning, model=model_id)



img_captioning = pipeline(Tasks.image_captioning, model='iic/ofa_image-caption_coco_large_en', model_revision='v1.0.1')


folder_path = "output/3d_spatial/images/sd_cascade/mode3/deepseek-r1-14b/3"
file = "a airplane behind a frog._mode3_00003_.png"
input_caption = os.path.join(folder_path, file)

result = img_captioning(input_caption)
print(f"description of {file}:")
print(f"caption: {result}")

result2 = pipeline_caption(input_caption)
print(f"description of {file}:")
print(f"caption: {result2}")