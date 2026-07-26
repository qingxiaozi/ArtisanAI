# ArtisanAI

基于 Gradio + Diffusers + PyTorch (ROCm) 的 AI 图像生成应用。

## 项目架构

```
┌─────────────────────────┐
│        Gradio           │  ← Web 界面 / 交互层
│   (app/app.py)          │
├─────────────────────────┤
│       Diffusers         │  ← 模型加载 / 推理流水线
│   (Hugging Face)        │
├─────────────────────────┤
│   PyTorch + ROCm        │  ← 深度学习框架 / AMD GPU 加速
└─────────────────────────┘
```

- **Gradio**（顶层）：提供 Web UI，用户通过浏览器交互，输入参数并查看生成结果。
- **Diffusers**（中间层）：Hugging Face 的扩散模型库，封装 Stable Diffusion 等模型的加载、调度与推理流程。
- **PyTorch + ROCm**（底层）：PyTorch 深度学习框架，通过 ROCm 实现 AMD GPU 加速推理。

## 启动 Gradio 应用

```bash
# 安装依赖（清华源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动应用（监听 0.0.0.0:10000）
cd /workspace/ArtisanAI && python app/app.py
```

启动后通过 `http://localhost:10000` 访问。

## LoRA 风格模块说明

项目 `models/lora/` 目录下包含以下风格化 LoRA 模块，均可通过 `style_lora_path` 参数加载使用：

| LoRA 文件 | 风格 | 推荐 Prompt 示例 | 说明 |
|-----------|------|------------------|------|
| `Chinese_Ink_Painting_Lora_SDXL.safetensors` | 中国水墨画 | `"traditional Chinese ink wash painting style, elegant brush strokes, misty mountains, black and white ink, poetic atmosphere"` | 中国传统水墨画风格，适用于山水、花鸟、人物等题材，呈现墨韵流淌的东方美学 |
| `Studio Ghibli style.safetensors` | 吉卜力工作室 | `"A beautiful Ghibli style portrait, hand-drawn animation, soft lighting, Miyazaki aesthetic, vibrant colors"` | 宫崎骏/吉卜力工作室动画风格，温暖细腻的手绘动画质感 |
| `anime.safetensors` | 日式动漫 | `"anime style, Japanese animation, cel shading, clean lines, vibrant colors, detailed character design"` | 经典日式动漫/二次元风格，线条清晰、色彩鲜艳 |
| `oil painting.safetensors` | 油画 | `"oil painting style, thick brushstrokes, rich textures, classical composition, impasto technique"` | 古典油画风格，厚重的笔触与丰富的纹理质感 |
| `pixar-style.safetensors` | 皮克斯动画 | `"Pixar style, 3D animation render, cartoon aesthetic, soft lighting, expressive character, Disney CGI look"` | 皮克斯/迪士尼3D动画风格，柔和光照与卡通质感 |
| `watercolor_v1_sdxl.safetensors` | 水彩画 | `"watercolor painting style, soft color washes, gentle gradients, translucent tones, artistic flowing colors"` | 水彩画风格，柔和晕染、通透的色彩层次 |
| `pytorch_lora_weights.safetensors` | 通用 LoRA（示例权重） | 根据实际训练数据确定，可配合各类题材使用 | 通用格式的 LoRA 权重文件（体积较大，371MB），来自 Diffusers 官方示例训练脚本，风格取决于训练数据集 |

### 使用方式

```python
from core.style_transfer import StyleTransferProcessor

processor = StyleTransferProcessor(device="cuda")
processor.load_model(model_dir="models")

# 指定风格 LoRA 路径进行推理
output = processor.process(
    image=your_image,
    prompt="your style-specific prompt here",  # 参考上表中的推荐 Prompt
    style_lora_path="models/lora/Studio Ghibli style.safetensors",
    strength=0.65,   # 风格化强度 (0.5~0.8)
    lora_scale=1.0   # LoRA 影响力
)
```

> **注**: `models/sdxl-lightning/` 下的 `sdxl_lightning_*step_lora.safetensors` 是 SDXL Lightning 加速 LoRA（2步/4步/8步），属于推理加速模块而非风格模块，由引擎自动加载。
