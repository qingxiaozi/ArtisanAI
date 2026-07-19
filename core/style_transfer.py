import os
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image
from utils.gpu_utils import clear_gpu_cache

# 直接指定 Hugging Face 镜像站点（国内加速）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用 Xet 存储以避免 401 认证错误
os.environ["HF_HUB_DISABLE_XET"] = "1"


class StyleTransferProcessor:
    """
    图像风格转换处理器
    基于 Stable Diffusion XL Turbo (diffusers/PyTorch) 的 Img2Img 管线实现
    AMD GPU 通过 ROCm 加速
    """

    def __init__(self, model_id: str = "stabilityai/sdxl-turbo", device: str = "cuda"):
        """
        初始化风格转换处理器

        :param model_id: Hugging Face 模型ID 或本地模型路径
        :param device: 运行设备 (ROCm 环境下使用 'cuda')
        """
        self.model_id = model_id
        self.device = device
        self.pipe = None

    def load_model(self):
        """加载模型到显存，并应用 ROCm 优化"""
        if self.pipe is not None:
            return

        print(f"正在加载模型: {self.model_id} ...")

        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=os.path.isdir(self.model_id),
        )

        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

        print("SDXL Turbo 模型加载完成 (PyTorch + ROCm)。")

    def process(self, image: Image.Image, prompt: str, strength: float = 0.5,
                num_inference_steps: int = 2, guidance_scale: float = 0.0) -> Image.Image:
        """
        执行风格转换

        :param image: 输入的 PIL 图像
        :param prompt: 风格转换提示词
        :param strength: 重绘幅度 (0.0 - 1.0)，Turbo 推荐 0.5 左右
        :param num_inference_steps: 推理步数，Turbo 仅需 1-4 步即可
        :param guidance_scale: 提示词引导系数，Turbo 训练时未使用，必须设为 0.0
        :return: 转换后的 PIL 图像
        """
        if self.pipe is None:
            raise RuntimeError("模型尚未加载，请先调用 load_model() 方法。")

        if image.mode != "RGB":
            image = image.convert("RGB")

        print(f"开始风格转换... 提示词: '{prompt}', 强度: {strength}, 步数: {num_inference_steps}")

        result = self.pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        output_image = result.images[0]
        print("风格转换完成。")
        return output_image

    def unload_model(self):
        """卸载模型，释放 GPU 显存"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            clear_gpu_cache()
            print("模型已卸载，显存已释放。")
