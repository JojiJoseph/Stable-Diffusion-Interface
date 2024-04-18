import gradio as gr
import torch
from diffusers.pipelines import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    safety_checker=None,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")


def generate_image(text):
    image = pipe(text).images[0]
    return image


def generate_output(input_text):
    # Generate the output image
    output_image = generate_image(input_text)
    return output_image


with gr.Blocks() as demo:

    with gr.Row():
        inputs = gr.Textbox(value="A cat", label="Enter your prompt here!")
        btn_submit = gr.Button(value="Generate Image")
    with gr.Row():
        outputs = gr.Image(label="Generated Image")
    btn_submit.click(generate_output, inputs, outputs)

demo.launch()
