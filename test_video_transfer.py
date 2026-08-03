import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.video_transfer import VideoTransferProcessor


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_video = os.path.join(base_dir, "assets", "洗面奶.mp4")
    output_video = os.path.join(base_dir, "assets", "output_styled.mp4")

    if not os.path.exists(input_video):
        print(f"找不到测试视频: {input_video}")
        return

    models_dir = os.path.join(base_dir, "models")
    use_local = os.path.exists(os.path.join(models_dir, "sdxl_base_v1"))
    print(f"模型来源: {'本地' if use_local else '在线下载 (HuggingFace)'}")

    processor = VideoTransferProcessor(device="cuda")

    try:
        print("开始视频风格化任务...")
        processor.process(
            video_path=input_video,
            output_path=output_video,
            prompt="Anime style, add trees in background",
            model_dir=models_dir if use_local else None,
            style_lora_path=None,
            strength=0.4,
            lora_scale=0.8,
        )

        print(f"\n任务完成！")
        print(f"输出文件: {output_video}")

    except Exception as e:
        print(f"\n处理失败: {e}")
    finally:
        processor.image_processor.unload_model()


if __name__ == "__main__":
    main()
