# pip install accelerate transformers safetensors diffusers

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "models")

# ---- Quantization toggle (set to False for fp16) ----
USE_NF4 = False

import torch

from controlnet_aux import HEDdetector
from transformers import BitsAndBytesConfig
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
from diffusers.utils import load_image

softedge_processor = HEDdetector.from_pretrained("lllyasviel/Annotators").to("cuda")
controlnet = ControlNetModel.from_pretrained(
    "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
    use_safetensors=False,
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
    "日式动漫": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.anime",
        "adapter_name": "anime",
        "prompt": "anime style, Japanese animation, cel shading, clean lines, vibrant colors, detailed character design",
        "scale": 1.0,
    },
    "像素艺术": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.pixel-art",
        "adapter_name": "pixel_art",
        "prompt": "pixel art style, 8-bit retro game aesthetic, blocky pixels, limited color palette, crisp dithering",
        "scale": 1.0,
    },
    "吉卜力工作室": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style",
        "adapter_name": "ghibli",
        "prompt": "A beautiful Ghibli style portrait, hand-drawn animation, soft lighting, Miyazaki aesthetic, vibrant colors",
        "scale": 1.0,
    },
    "卡通": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.cartoon",
        "adapter_name": "cartoon",
        "prompt": "cartoon style, bold outlines, flat bright colors, simplified shapes, playful exaggerated features",
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
    "超写实插画": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.ultra-realistic-illustration",
        "adapter_name": "ultra_realistic",
        "prompt": "ultra realistic illustration, highly detailed digital painting, refined shading, lifelike textures, artstation quality",
        "scale": 1.0,
    },
    "千禧独立漫画": {
        "repo_id": "ntc-ai/SDXL-LoRA-slider.2000s-indie-comic-art-style",
        "adapter_name": "indie_comic",
        "prompt": "2000s indie comic art style, inked linework, halftone shading, gritty alternative comic book aesthetic",
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


def get_softedge_map(image):
    return softedge_processor(
        image,
        detect_resolution=1024,
        image_resolution=1024,
        scribble=False,
    )


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
    softedge_image = get_softedge_map(image)

    images = pipe(
        prompt,
        image=image,
        control_image=softedge_image,
        strength=0.7,
        num_inference_steps=4,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=1.5,
        generator=generator,
    ).images
    output_dir = os.path.join(os.path.dirname(__file__), "output_images")
    os.makedirs(output_dir, exist_ok=True)
    images[0].save(os.path.join(output_dir, "robot_cat.png"))
