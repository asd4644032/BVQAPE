import ollama
import base64

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

#'C:/Users/lab929/Downloads/ComfyUI-master/output/T2I/mode1/a airplane hidden by a vase_mode1_00001_.png'

image_path = 'C:/Users/lab929/Downloads/ComfyUI-master/output/T2I/mode1/a airplane in front of a giraffe_mode1_00001_.png'
image_base64 = encode_image(image_path)

messages = [
    {
        'role': 'system',
        'content': 'Please answer the question based on the image. The answer should be a simple word like yes or no.'
    }, 
    {
        'role': 'user', 
        'content':'Question : Are both objects from the real world mentioned in the text?', 
        'images': [image_path]
    }
    ]


'''
res = ollama.chat(
	model='llama3.2-vision',
	messages=[
		{
			'role': 'user',
			'content': 'Question :Is the airplane positioned above/beside/behind/front of the giraffe?',
			'images': [image_path]
		}
	]
)
'''
res = ollama.chat(
	model='llama3.2-vision',
	messages = messages
)


print(res['message']['content'])