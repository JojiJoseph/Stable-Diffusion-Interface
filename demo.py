import gradio as gr
import torch
from diffusers.pipelines import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",safety_checker = None, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

def generate_image(text):
    image = pipe(text).images[0]
    return image

def generate_output(input_text):
    # Generate the output image
    output_image = generate_image(input_text)
    return output_image

def main(input_text):
    # Generate the output image based on the input text
    output_image = generate_output(input_text)
    return output_image

iface = gr.Interface(fn=main, inputs="text", outputs="image", title="Image Generator")
iface.launch()