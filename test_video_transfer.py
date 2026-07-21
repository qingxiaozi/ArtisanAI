import sys
import os

# 确保能导入 core 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.video_transfer import VideoTransferProcessor

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 配置输入输出路径
    input_video = os.path.join(base_dir, "assets", "test_video.mp4")
    output_video = os.path.join(base_dir, "assets", "output_styled.mp4")
    
    if not os.path.exists(input_video):
        print(f"找不到测试视频: {input_video}")
        print("请将一个短视频放入 assets/ 目录并命名为 test_video.mp4")
        return

    # 2. 模型目录
    models_dir = os.path.join(base_dir, "models")
    use_local = os.path.exists(os.path.join(models_dir, "sdxl-base-1.0"))
    print(f"模型来源: {'本地' if use_local else '在线下载 (HuggingFace)'}")

    # 3. 初始化视频处理器
    processor = VideoTransferProcessor(device="cuda")

    try:
        # 4. 执行视频风格转换
        print("开始视频风格化任务...")
        processor.process(
            video_path=input_video,
            output_path=output_video,
            prompt="Anime style, add trees in background",
            model_dir=models_dir if use_local else None,
            style_lora_path=None,  # 有本地 LoRA 时填入路径，如 "./models/lora/ghibli.safetensors"
            strength=0.65,         # 风格化强度 (0.5~0.8 效果较好)
            lora_scale=0.8,        # LoRA 影响力
            batch_size=1           # AMD 显存紧张时设为 1，显存充足可尝试 4
        )
        
        print(f"\n任务完成！")
        print(f"输出文件: {output_video}")

    except Exception as e:
        print(f"\n处理失败: {e}")
    finally:
        # 4. 清理显存
        processor.image_processor.unload_model()

if __name__ == "__main__":
    main()