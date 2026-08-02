import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "models")

import time

import gradio as gr
import torch
from PIL import Image
from diffusers.utils import load_image
from gen_image import pipe, get_depth_map, LIGHTNING_ADAPTER

# LoRA manifest: display_name → {repo_id, prompt, scale}
# 首次使用时会自动从 HuggingFace 下载并缓存到 models/
LORA_MANIFEST = {
    "无": {"repo_id": None, "prompt": "A robot, 4k photo", "scale": 0.8},
    "中国水墨画": {
        "repo_id": "ming-yang/sdxl_chinese_ink_lora",
        "prompt": "traditional Chinese ink wash painting style, elegant brush strokes, misty mountains, black and white ink, poetic atmosphere",
        "scale": 1.0,
    },
    "吉卜力工作室": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style",
        "prompt": "A beautiful Ghibli style portrait, hand-drawn animation, soft lighting, Miyazaki aesthetic, vibrant colors",
        "scale": 1.0,
    },
    "日式动漫": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.anime",
        "prompt": "anime style, Japanese animation, cel shading, clean lines, vibrant colors, detailed character design",
        "scale": 1.0,
    },
    "油画": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.oil-painting",
        "prompt": "oil painting style, thick brushstrokes, rich textures, classical composition, impasto technique",
        "scale": 1.0,
    },
    "皮克斯动画": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.pixar-style",
        "prompt": "Pixar style, 3D animation render, cartoon aesthetic, soft lighting, expressive character, Disney CGI look",
        "scale": 1.0,
    },
    "水彩画": {
        "repo_id": "ostris/watercolor_style_lora_sdxl",
        "prompt": "watercolor painting style, soft color washes, gentle gradients, translucent tones, artistic flowing colors",
        "scale": 1.0,
    },
    "赛博朋克": {
        "repo_id": "issaccyj/lora-sdxl-cyberpunk",
        "prompt": "cyberpunk style, neon lights, futuristic city, high tech low life, rain-slicked streets, blade runner aesthetic",
        "scale": 1.0,
    },
}

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

def generate(image, prompt, negative_prompt, strength, steps, conditioning_scale, guidance_scale, seed, lora_name):
    if image is None:
        return None, ""

    # Apply style LoRA
    if lora_name and lora_name in LORA_MANIFEST and LORA_MANIFEST[lora_name]["repo_id"]:
        lora_info = LORA_MANIFEST[lora_name]
        try:
            # Remove previous style adapter if exists
            if "style" in pipe.get_active_adapters():
                pipe.set_adapters([LIGHTNING_ADAPTER], adapter_weights=[1.0])
                pipe.delete_adapters(["style"])
            pipe.load_lora_weights(lora_info["repo_id"], adapter_name="style")
            pipe.set_adapters([LIGHTNING_ADAPTER, "style"], adapter_weights=[1.0, lora_info.get("scale", 0.8)])
        except Exception as e:
            print(f"Failed to load LoRA '{lora_name}': {e}")
            pipe.set_adapters([LIGHTNING_ADAPTER], adapter_weights=[1.0])
    else:
        pipe.set_adapters([LIGHTNING_ADAPTER], adapter_weights=[1.0])

    t_total = time.time()

    # Resize to 1024x1024 (required by the pipeline)
    original_size = image.size
    image = image.resize((1024, 1024))

    # Get depth map
    t_depth_start = time.time()
    depth_image = get_depth_map(image)
    torch.cuda.synchronize()
    t_depth = time.time() - t_depth_start

    # Set up generator
    if seed < 0:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Track per-step denoising time
    step_durations = []
    t_last = time.time()
    def step_callback(pipe, step, timestep, callback_kwargs):
        nonlocal t_last
        torch.cuda.synchronize()
        now = time.time()
        step_durations.append(now - t_last)
        t_last = now
        return callback_kwargs

    t_pipe_start = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=image,
        control_image=depth_image,
        strength=strength,
        num_inference_steps=int(steps),
        controlnet_conditioning_scale=conditioning_scale,
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=step_callback,
    ).images[0]
    torch.cuda.synchronize()
    t_pipe = time.time() - t_pipe_start

    # step_durations[0] = encode+step1, [1..] = step2..N, t_pipe - sum = VAE decode
    n_steps = int(steps)
    if len(step_durations) > 1:
        t_setup = step_durations[0]  # VAE/text encode + step1
        t_per_step = sum(step_durations[1:]) / (len(step_durations) - 1)
    else:
        t_setup = t_pipe
        t_per_step = t_pipe

    # Resize back to original input size
    result = result.resize(original_size)

    elapsed = time.time() - t_total

    n_steps = int(steps)
    timing_lines = [
        f"⏱ Total: {elapsed:.2f}s",
        f"├─ Depth:      {t_depth:.2f}s",
        f"├─ Setup:      {t_setup:.2f}s (encode + step1)",
        f"├─ Denoise:    {t_per_step:.3f}s/step × {n_steps}",
        f"└─ Pipe total: {t_pipe:.2f}s",
    ]
    timing_text = "\n".join(timing_lines)
    return result, timing_text


with gr.Blocks(title="Depth-Controlled Image Generation") as demo:
    gr.Markdown("# 🎨 SDXL Depth ControlNet Image-to-Image")
    gr.Markdown("Upload an image, enter a prompt, and generate a new image guided by depth estimation.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image", value=_default_image_path)
            lora_selector = gr.Dropdown(
                choices=list(LORA_MANIFEST.keys()),
                value="无",
                label="Style LoRA",
                interactive=True,
            )
            prompt = gr.Textbox(
                value=LORA_MANIFEST["无"]["prompt"],
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
                steps = gr.Slider(1, 100, value=8, step=1, label="Steps")
                guidance_scale = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Guidance Scale")
            seed = gr.Number(value=42, label="Seed (-1 = random)", precision=0)
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Generated Image")
            timing_output = gr.Textbox(label="Timing", interactive=False, lines=6)

    def on_lora_change(lora_name):
        info = LORA_MANIFEST.get(lora_name, LORA_MANIFEST["无"])
        return info["prompt"]

    lora_selector.change(
        fn=on_lora_change,
        inputs=[lora_selector],
        outputs=[prompt],
    )

    generate_btn.click(
        fn=generate,
        inputs=[input_image, prompt, negative_prompt, strength, steps, conditioning_scale, guidance_scale, seed, lora_selector],
        outputs=[output_image, timing_output],
    )

if __name__ == "__main__":
    # Warm up pipeline (first inference compiles CUDA kernels)
    print("Warming up pipeline...")
    warmup_img = Image.open(_default_image_path)
    generate(warmup_img, "A robot, 4k photo", "", 0.99, 8, 0.5, 1.5, 42, "无")
    print("Warmup done.")

    demo.launch(server_name="0.0.0.0")
