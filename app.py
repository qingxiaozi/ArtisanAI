import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "models")

import time

import gradio as gr
import torch
from PIL import Image
from diffusers.utils import load_image
from gen_image import pipe, get_depth_map

# Ensure default input image exists
_default_image_path = os.path.join(os.path.dirname(__file__), "images", "cat.png")
if not os.path.exists(_default_image_path):
    os.makedirs(os.path.dirname(_default_image_path), exist_ok=True)
    cat_image = load_image(
        "https://hf-mirror.com/datasets/hf-internal-testing/diffusers-images/resolve/main"
        "/kandinsky/cat.png"
    ).resize((1024, 1024))
    cat_image.save(_default_image_path)

# Monkey-patch: fix gradio_client 1.3.0 bug where pydantic v2 produces bool schemas
from gradio_client import utils as gradio_client_utils
_original_get_type = gradio_client_utils.get_type
_original_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type

def _patched_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _original_get_type(schema)

def _patched_json_schema_to_python_type(schema, defs):
    if not isinstance(schema, dict):
        return "Any"
    return _original_json_schema_to_python_type(schema, defs)

gradio_client_utils.get_type = _patched_get_type
gradio_client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

def generate(image, prompt, negative_prompt, strength, steps, conditioning_scale, seed):
    if image is None:
        return None, ""

    t_start = time.time()

    # Resize to 1024x1024 (required by the pipeline)
    original_size = image.size
    image = image.resize((1024, 1024))

    # Get depth map
    depth_image = get_depth_map(image)

    # Set up generator
    if seed < 0:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.autocast("cuda"):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=image,
            control_image=depth_image,
            strength=strength,
            num_inference_steps=int(steps),
            controlnet_conditioning_scale=conditioning_scale,
            generator=generator,
        ).images[0]

    # Resize back to original input size
    result = result.resize(original_size)

    elapsed = time.time() - t_start
    timing_text = f"⏱ {elapsed:.1f}s ({int(steps)} steps, {elapsed / int(steps):.2f}s/step)"
    return result, timing_text


with gr.Blocks(title="Depth-Controlled Image Generation") as demo:
    gr.Markdown("# 🎨 SDXL Depth ControlNet Image-to-Image")
    gr.Markdown("Upload an image, enter a prompt, and generate a new image guided by depth estimation.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image", value=_default_image_path)
            prompt = gr.Textbox(
                value="A robot, 4k photo",
                label="Prompt",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                value="",
                label="Negative Prompt",
                lines=2,
            )
            with gr.Row():
                strength = gr.Slider(0.0, 1.0, value=0.99, step=0.01, label="Strength")
                conditioning_scale = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="ControlNet Scale")
            with gr.Row():
                steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
                seed = gr.Number(value=42, label="Seed (-1 = random)", precision=0)
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Generated Image")
            timing_output = gr.Textbox(label="Timing", interactive=False)

    generate_btn.click(
        fn=generate,
        inputs=[input_image, prompt, negative_prompt, strength, steps, conditioning_scale, seed],
        outputs=[output_image, timing_output],
    )

if __name__ == "__main__":
    # Warm up pipeline (first inference compiles CUDA kernels)
    print("Warming up pipeline...")
    warmup_img = Image.open(_default_image_path)
    generate(warmup_img, "A robot, 4k photo", "", 0.99, 20, 0.5, 42)
    print("Warmup done.")

    demo.launch(server_name="0.0.0.0")
