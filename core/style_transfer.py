import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, AutoencoderKL
from diffusers.schedulers import EulerDiscreteScheduler
from controlnet_aux import CannyDetector
from utils.gpu_utils import clear_gpu_cache

# 国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

class StyleTransferProcessor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = None
        self.canny_detector = CannyDetector()

    def load_model(self, model_dir: str = None):
        """
        model_dir: 本地模型目录,包含以下子目录:
            sdxl-base-1.0/        → SDXL Base 单文件权重 (sd_xl_base_1.0.safetensors)
            controlnet-canny-sdxl-1.0/ → ControlNet Canny 权重 + config.json
            sdxl-vae-fp16-fix/    → VAE fp16 fix 权重 + config.json
            sdxl-lightning/       → SDXL Lightning LoRA
        如果为 None,则从 HuggingFace 在线下载。
        """
        if self.pipe is not None: return

        if model_dir is None:
            print("请从 HuggingFace 镜像下载 SDXL + Lightning + ControlNet 模型...")
            return
        else:
            controlnet_dir = os.path.join(model_dir, "controlnet-canny-sdxl-1.0")
            vae_dir = os.path.join(model_dir, "sdxl-vae-fp16-fix")
            base_dir = os.path.join(model_dir, "sdxl-base-1.0")
            lightning_dir = os.path.join(model_dir, "sdxl-lightning")

            # 1. 从本地加载 ControlNet(diffusers 分割格式目录)
            controlnet = ControlNetModel.from_pretrained(
                controlnet_dir, torch_dtype=torch.float16
            )

            # 2. 从本地加载 VAE
            vae = AutoencoderKL.from_pretrained(
                vae_dir, torch_dtype=torch.float16
            )

            # 3. 从本地加载 SDXL Base 完整 pipeline(diffusers 分割格式目录)
            self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                base_dir, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
            )

            # 4. 从本地加载 Lightning LoRA
            lightning_path = os.path.join(lightning_dir, "sdxl_lightning_8step_lora.safetensors")
            self.pipe.load_lora_weights(lightning_path)

        # 5. 配置 Lightning 专用调度器
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload() # AMD 显卡防 OOM 必备
        print("SDXL + Lightning + ControlNet 引擎加载完成。")

    def process(self, image: Image.Image, prompt: str, style_lora_path: str = None,
                strength: float = 0.65, lora_scale: float = 0.8) -> Image.Image:
        if self.pipe is None: raise RuntimeError("请先调用 load_model()")

        # SDXL 原生分辨率 1024x1024，统一到此尺寸
        image = image.resize((1024, 1024), Image.LANCZOS)

        # 提取 Canny 骨架图
        canny_image = self.canny_detector(image)

        # 动态挂载/卸载 风格 LoRA
        if style_lora_path and os.path.exists(style_lora_path):
            self.pipe.load_lora_weights(style_lora_path, adapter_name="style")
            self.pipe.set_adapters(["style"], adapter_weights=[lora_scale])
        else:
            self.pipe.disable_lora() # 无风格 LoRA 时禁用

        # 执行推理 (Lightning 推荐 8步, CFG=1.5)
        result = self.pipe(
            prompt=prompt, image=image, control_image=canny_image,
            strength=strength, num_inference_steps=8, guidance_scale=1.5,
            height=1024, width=1024
        )

        return result.images[0]

    def unload_model(self):
        if self.pipe:
            del self.pipe; self.pipe = None
            clear_gpu_cache()