# pip install accelerate transformers safetensors diffusers

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "models")

# ---- Quantization toggle (set to False for fp16) ----
USE_NF4 = False

import torch
import numpy as np
from PIL import Image

from transformers import DPTImageProcessor, DPTForDepthEstimation, BitsAndBytesConfig
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
from diffusers.utils import load_image

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0-small",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

if USE_NF4:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        quantization_config=quant_config,
        device_map={"": "cuda"},
    )
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        unet=unet,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    # UNet already on cuda via device_map; only move other components
    pipe.vae = pipe.vae.to("cuda")
    pipe.controlnet = pipe.controlnet.to("cuda")
    pipe.text_encoder = pipe.text_encoder.to("cuda")
    pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
else:
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
pipe.load_lora_weights("ByteDance/SDXL-Lightning", weight_name="sdxl_lightning_8step_lora.safetensors")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
# Lightning adapter name.
LIGHTNING_ADAPTER = pipe.get_active_adapters()[0]

# LoRA manifest: display_name -> {repo_id, adapter_name, prompt, scale}
# First use downloads and caches weights under models/.
LORA_MANIFEST = {
    "无": {"repo_id": None, "adapter_name": None, "prompt": "A robot, 4k photo", "scale": 0.8},
    "中国水墨画": {
        "repo_id": "ming-yang/sdxl_chinese_ink_lora",
        "adapter_name": "ink",
        "prompt": "traditional Chinese ink wash painting style, elegant brush strokes, misty mountains, black and white ink, poetic atmosphere",
        "scale": 1.0,
    },
    "吉卜力工作室": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style",
        "adapter_name": "ghibli",
        "prompt": "A beautiful Ghibli style portrait, hand-drawn animation, soft lighting, Miyazaki aesthetic, vibrant colors",
        "scale": 1.0,
    },
    "日式动漫": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.anime",
        "adapter_name": "anime",
        "prompt": "anime style, Japanese animation, cel shading, clean lines, vibrant colors, detailed character design",
        "scale": 1.0,
    },
    "油画": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.oil-painting",
        "adapter_name": "oil",
        "prompt": "oil painting style, thick brushstrokes, rich textures, classical composition, impasto technique",
        "scale": 1.0,
    },
    "皮克斯动画": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.pixar-style",
        "adapter_name": "pixar",
        "prompt": "Pixar style, 3D animation render, cartoon aesthetic, soft lighting, expressive character, Disney CGI look",
        "scale": 1.0,
    },
    "水彩画": {
        "repo_id": "ostris/watercolor_style_lora_sdxl",
        "adapter_name": "watercolor",
        "prompt": "watercolor painting style, soft color washes, gentle gradients, translucent tones, artistic flowing colors",
        "scale": 1.0,
    },
}


def preload_style_loras():
    print("Loading style LoRAs...")
    for display_name, info in LORA_MANIFEST.items():
        repo_id = info.get("repo_id")
        adapter_name = info.get("adapter_name")
        if not repo_id:
            continue
        if not adapter_name:
            raise ValueError(f"LoRA '{display_name}' is missing adapter_name")

        pipe.load_lora_weights(repo_id, adapter_name=adapter_name)
        print(f"  Loaded {display_name}: {repo_id} as '{adapter_name}'")

    pipe.set_adapters([LIGHTNING_ADAPTER], adapter_weights=[1.0])
    print("Style LoRAs loaded.")


def apply_lora(lora_name):
    info = LORA_MANIFEST.get(lora_name, LORA_MANIFEST["无"])
    adapter_name = info.get("adapter_name")
    if adapter_name:
        pipe.set_adapters(
            [LIGHTNING_ADAPTER, adapter_name],
            adapter_weights=[1.0, info.get("scale", 1.0)],
        )
    else:
        pipe.set_adapters([LIGHTNING_ADAPTER], adapter_weights=[1.0])


preload_style_loras()


def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


if __name__ == "__main__":
    seed = 42
    generator = torch.Generator(device="cuda").manual_seed(seed)

    prompt = "A robot, 4k photo"
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(images_dir, exist_ok=True)
    image = load_image(
        "https://hf-mirror.com/datasets/hf-internal-testing/diffusers-images/resolve/main"
        "/kandinsky/cat.png"
    ).resize((1024, 1024))
    image.save(os.path.join(images_dir, "cat.png"))
    controlnet_conditioning_scale = 0.5  # recommended for good generalization
    depth_image = get_depth_map(image)

    images = pipe(
        prompt,
        image=image,
        control_image=depth_image,
        strength=0.99,
        num_inference_steps=4,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=1.5,
        generator=generator,
    ).images
    output_dir = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(output_dir, exist_ok=True)
    images[0].save(os.path.join(output_dir, "robot_cat.png"))
