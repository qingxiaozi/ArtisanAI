#!/usr/bin/env python3
import os
import sys

# 1. 设置环境变量（使用国内镜像加速）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

def main():
    # 7 种常用风格 LoRA (均适配 SDXL，使用公开非 gated 仓库)
    loras = [
        # 1. 二次元动漫风
        ("ntc-ai/SDXL-LoRA-slider.anime", "anime.safetensors"),

        # 2. 水彩画风
        ("ostris/watercolor_style_lora_sdxl", "watercolor_v1_sdxl.safetensors"),

        # 3. 赛博朋克风
        ("issaccyj/lora-sdxl-cyberpunk", "pytorch_lora_weights.safetensors"),

        # 4. 油画风
        ("ntc-ai/SDXL-LoRA-slider.oil-painting", "oil painting.safetensors"),

        # 5. 国风/水墨画风
        ("ming-yang/sdxl_chinese_ink_lora", "Chinese_Ink_Painting_Lora_SDXL.safetensors"),

        # 6. 吉卜力风
        ("ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style", "Studio Ghibli style.safetensors"),

        # 7. 迪士尼/皮克斯风
        ("ntc-ai/SDXL-LoRA-slider.pixar-style", "pixar-style.safetensors"),
    ]

    lora_dir = "./models/lora"
    os.makedirs(lora_dir, exist_ok=True)

    print(f"🌐 当前使用的 HuggingFace 镜像: {os.environ.get('HF_ENDPOINT', 'NOT SET')}")
    print("🚀 开始下载 7 种常用风格 LoRA...\n")

    for repo_id, filename in loras:
        print(f"--- 正在下载: {filename} ---")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=lora_dir,
                allow_patterns=[filename],  # 仅下载权重文件，节省空间
            )
            print(f"✅ 下载成功: {filename}\n")
        except Exception as e:
            print(f"❌ 下载失败 ({filename}): {e}\n")

    print("🎉 所有风格 LoRA 下载完成！")
    print(f"📁 文件存放路径: {os.path.abspath(lora_dir)}")

if __name__ == "__main__":
    main()