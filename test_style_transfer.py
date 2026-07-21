import sys
import os
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.style_transfer import StyleTransferProcessor

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 优先使用本地下载的模型，否则从 HuggingFace 在线下载
    models_dir = os.path.join(base_dir, "models")
    base_model_path = os.path.join(models_dir, "sdxl-base-1.0")
    use_local = os.path.exists(base_model_path)
    print(f"模型来源: {'本地' if use_local else '在线下载 (HuggingFace)'}")

    processor = StyleTransferProcessor(device="cuda")

    try:
        if use_local:
            processor.load_model(model_dir=models_dir)
        else:
            processor.load_model()

        # 加载测试图片
        test_image_path = os.path.join(base_dir, "assets", "yi.jpg")
        if not os.path.exists(test_image_path):
            print(f"错误: 找不到测试图片 {test_image_path}")
            return
            
        input_image = Image.open(test_image_path)
        print(f"输入图片尺寸: {input_image.size}")

        # 执行风格转换 (新版内部已固化 Lightning 的 8步/CFG=1.5 配置)
        output_image = processor.process(
            image=input_image,
            prompt="Anime style, no background",
            strength=0.65,
            # 可选：传入本地风格 LoRA 路径进行测试
            # style_lora_path="./models/lora/ghibli_style.safetensors",
            # lora_scale=0.8
        )

        # 保存结果
        output_path = os.path.join(base_dir, "assets", "output_sdxl_lightning.jpg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_image.save(output_path)
        print(f"测试成功！结果已保存至: {output_path}")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        processor.unload_model()

if __name__ == "__main__":
    main()