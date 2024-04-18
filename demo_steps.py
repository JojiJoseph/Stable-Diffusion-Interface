import gradio as gr
import torch
from diffusers.pipelines import StableDiffusionPipeline
from threading import Thread
import time

if torch.cuda.is_available():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )


last_image = None
last_time = 0


def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
    global last_image
    image = (
        pipe.vae.decode(1 / 0.18215 * callback_kwargs["latents"])
        .sample.cpu()
        .detach()
        .numpy()[0]
    )
    image = image.transpose(1, 2, 0)
    image = image - image.min()
    image = image / image.max()
    last_image = image
    return callback_kwargs


def generate_output(input_text):
    btn_submit.visible = False
    thread = Thread(
        target=pipe,
        args=(input_text,),
        kwargs={"callback_on_step_end": callback_dynamic_cfg},
    )
    thread.start()
    global last_time
    while thread.is_alive():
        if last_image is not None:
            yield last_image
            time.sleep(0.01)
    yield last_image


with gr.Blocks() as demo:

    with gr.Row():
        inputs = gr.Textbox(value="A cat", label="Enter your prompt here!")
        btn_submit = gr.Button(value="Generate Image")
    with gr.Row():
        outputs = gr.Image(label="Generated Image")
    btn_submit.click(generate_output, inputs, outputs)

demo.launch()
