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

# Pre-download all style LoRAs (load then unload, files cached in models/)
_STYLE_LORA_REPOS = [
    "ming-yang/sdxl_chinese_ink_lora",
    "ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style",
    "ntc-ai/SDXL-LoRA-slider.anime",
    "ntc-ai/SDXL-LoRA-slider.oil-painting",
    "ntc-ai/SDXL-LoRA-slider.pixar-style",
    "ostris/watercolor_style_lora_sdxl",
    "issaccyj/lora-sdxl-cyberpunk",
]
print("Pre-downloading style LoRAs...")
for _repo in _STYLE_LORA_REPOS:
    try:
        pipe.load_lora_weights(_repo, adapter_name="_tmp")
        pipe.delete_adapters(["_tmp"])
        print(f"  Cached: {_repo}")
    except Exception as e:
        print(f"  Failed: {_repo} — {e}")
print("LoRA pre-download done.")
# Lightning adapter name (may be renamed to 'default_0' after pre-download dance)
LIGHTNING_ADAPTER = pipe.get_active_adapters()[0]


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