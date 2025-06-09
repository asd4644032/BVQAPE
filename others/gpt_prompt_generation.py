from openai import OpenAI
import os
from utils import load_prompt_book
import json

prompt_book = load_prompt_book("3d_spatial.txt")

api_key = "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"
client = OpenAI(api_key=api_key)
output_folder = "output/3d_spatial/prompt/gpt_4o"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def generate_prompt(description):
    response = client.responses.create(
    model="gpt-4o",
    input=[
        {
        "role": "system",
        "content": [
            {
            "type": "input_text",
            "text": "<reasoning> - Simple Change: no - Reasoning: yes - Identify: \"interpret user input\", \"expand into image prompts\" - Conclusion: yes - Ordering: before - Structure: yes - Examples: yes - Representative: 4 - Complexity: 4 - Task: 4 - Necessity: (assumed moderate-to-high due to model-specific needs) - Specificity: 5 - Prioritization: Specificity, Output Format, Examples - Conclusion: Refactor the prompt to focus on generating highly specific, vivid image prompts optimized for Stable Diffusion. Include rich visual detail and relevant style cues, with helpful few-shot examples. </reasoning>\nTransform a user's plain language description into a vivid, detailed, and well-structured text prompt optimized for image generation with Stable Diffusion.\n\nYour output prompt should be directly usable with Stable Diffusion, emphasizing specific visual elements such as subject, style, lighting, composition, mood, and environment. Interpret and enhance the user's input to create a rich and coherent visual scene. Stay true to the core concept, but elevate it using artful, descriptive language.\n\nSteps\nExtract key elements: subjects, settings, actions, moods.\n\nEnrich the scene with visual modifiers: colors, lighting, atmosphere, camera angle.\n\nInclude style references (e.g. \"digital painting\", \"8k render\", \"isometric\", \"surreal\", \"dark fantasy\").\n\nAvoid vague or generic phrasing; prioritize specificity and realism or stylistic clarity.\n\nDo not include \"prompt engineering jargon\" like \"trending on ArtStation\" unless clearly implied by user input.\n\nOutput Format\nOutput: One highly detailed sentence or short paragraph, suitable for direct input into Stable Diffusion.\n\nLanguage: Clear, natural English.\n\nNo additional formatting (e.g. no markdown, JSON, or code blocks).\n\nMaintain proper grammar and a natural descriptive tone.\n\nExamples\nExample 1\nInput:\n\"A rabbit standing on a mountain during sunset\"\nOutput:\n\"A majestic white rabbit with long ears stands on a rocky cliff at sunset, overlooking a glowing orange and purple horizon, with wind blowing through its fur and dramatic clouds overhead — ultra-detailed, cinematic style.\"\n\nExample 2\nInput:\n\"A futuristic robot playing the violin in the rain\"\nOutput:\n\"A sleek humanoid robot, illuminated by neon reflections, plays a violin under heavy rain on a dimly lit city street, with raindrops sparkling on its metallic surface and soft blue lights glowing beneath its eyes — photorealistic, cyberpunk atmosphere.\"\n\nExample 3\nInput:\n\"A girl walking through a dream\"\nOutput:\n\"A young girl in a flowing dress walks barefoot through a surreal dreamscape of floating islands, giant glowing flowers, and pastel-colored mist, her hair suspended in the air — fantasy art style, soft ambient lighting.\"\n\nExample 4\nInput:\n\"A haunted house in the forest\"\nOutput:\n\"An old Victorian-style mansion stands deep in a foggy forest, its broken windows glowing faintly from within, vines wrapping around decaying wood, with crows circling above — horror illustration, moody and atmospheric.\"\n\nNotes\nPrioritize visual coherence — scenes should feel like they could exist visually.\n\nAvoid vague or overly poetic descriptions without grounding details.\n\nStable Diffusion handles descriptive imagery well — be bold with adjectives and scene-building.\n\nWhen style isn't specified, infer a fitting visual or artistic direction from the subject matter."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "input_text",
            "text": f"Description : {description}"
            }
        ]
        }
    ],
    text={
        "format": {
        "type": "text"
        }
    },
    reasoning={},
    tools=[],
    temperature=1,
    max_output_tokens=2048,
    top_p=1,
    store=True
    )
    prompt_text = response.output[0].content[0].text
    return prompt_text



for description in prompt_book:
    print(description)
    
    prompt_text = generate_prompt(description)
    prompt_dict = {}
    prompt_dict["title"] = description
    prompt_dict["prompt"] = prompt_text

    with open(os.path.join(output_folder, f"{description}.json"), "w") as f:
        json.dump(prompt_dict, f)
    




"""
example output
{
  "title": "A photo shows two people shaking hands, one on the left and one on the right",
  "prompt": "A professional setting where two individuals, in formal business attire, engage in a firm handshake. The person on the left, a middle-aged man with salt-and-pepper hair, wears a navy suit and a crisp white shirt. The person on the right, a young woman with dark hair pulled back, is dressed in a tailored grey blazer. They are in a modern office with large windows, sunlight streaming in, casting soft shadows. The atmosphere is one of mutual respect and collaboration — high-resolution, realistic style."
}
"""



