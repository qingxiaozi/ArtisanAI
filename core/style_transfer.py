import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from utils.gpu_utils import clear_gpu_cache


class StyleTransferProcessor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = None
        self.canny = None

    def load_model(self, model_dir: str = None):
        if self.pipe is not None:
            return
        if not model_dir:
            raise ValueError("model_dir 不能为空")

        base_dir = os.path.join(model_dir, "sdxl_base_v1")
        controlnet_dir = os.path.join(model_dir, "controlnet-canny-sdxl-1.0")

        if os.path.isdir(controlnet_dir):
            controlnet = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=torch.float16)
            self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                base_dir, controlnet=controlnet, torch_dtype=torch.float16
            )
            self.canny = CannyDetector()
            print("SDXL + ControlNet Canny 引擎加载完成。")
        else:
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                base_dir, torch_dtype=torch.float16
            )
            print("SDXL img2img 引擎加载完成（无 ControlNet）。")

        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

    def process(self, image: Image.Image, prompt: str, style_lora_path: str = None,
                strength: float = 0.65, lora_scale: float = 0.8,
                seed: int = None) -> Image.Image:
        if self.pipe is None:
            raise RuntimeError("请先调用 load_model()")

        image = image.resize((1024, 1024), Image.LANCZOS)

        if style_lora_path and os.path.exists(style_lora_path):
            self.pipe.load_lora_weights(style_lora_path, adapter_name="style")
            self.pipe.set_adapters(["style"], adapter_weights=[lora_scale])
        else:
            self.pipe.disable_lora()

        kwargs = dict(
            prompt=prompt, image=image,
            strength=strength, num_inference_steps=25, guidance_scale=7.5,
            height=1024, width=1024,
        )
        if seed is not None:
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)
        if self.canny is not None:
            kwargs["control_image"] = self.canny(image)

        result = self.pipe(**kwargs)
        return result.images[0]

    def unload_model(self):
        if self.pipe:
            del self.pipe
            self.pipe = None
            self.canny = None
            clear_gpu_cache()
