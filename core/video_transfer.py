import os
import cv2
import torch
import numpy as np
import subprocess
import shutil
from PIL import Image
from tqdm import tqdm
from core.style_transfer import StyleTransferProcessor
from utils.gpu_utils import clear_gpu_cache

class VideoTransferProcessor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        # 复用现有的图像风格转换引擎
        self.image_processor = StyleTransferProcessor(device=device)
        
    def process(self, video_path: str, output_path: str, prompt: str, 
                style_lora_path: str = None, strength: float = 0.65, 
                lora_scale: float = 0.8, batch_size: int = 4,
                model_dir: str = None):
        """
        执行视频风格转换
        """
        if not self.image_processor.pipe:
            self.image_processor.load_model(model_dir=model_dir)

        # 1. 读取视频元数据
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 2. 初始化视频写入器
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"开始视频风格化: {total_frames} 帧, 批次大小: {batch_size}")
        
        frame_buffer = []
        frame_indices = []
        success = False

        try:
            for idx in tqdm(range(total_frames), desc="处理视频帧"):
                ret, frame_bgr = cap.read()
                if not ret: break
                
                # BGR 转 RGB，再转 PIL
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                frame_buffer.append(pil_image)
                frame_indices.append(idx)

                # 当缓冲区满或到达最后一帧时，触发批量推理
                if len(frame_buffer) >= batch_size or idx == total_frames - 1:
                    processed_images = self._batch_infer(
                        frame_buffer, prompt, style_lora_path, strength, lora_scale
                    )
                    
                    # 缩放回原始分辨率并写入视频
                    for img in processed_images:
                        frame = img.resize((width, height), Image.LANCZOS)
                        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                    
                    # 清空缓冲区
                    frame_buffer.clear()
                    frame_indices.clear()
                    clear_gpu_cache() # 定期清理显存碎片
            success = True

        finally:
            cap.release()
            out.release()
            if success:
                print(f"视频处理完成，已保存至: {output_path}")
                self._ensure_h264_compatible(output_path)

    def _ensure_h264_compatible(self, output_path: str):
        if not shutil.which("ffmpeg"):
            print("\u26a0 系统未安装 ffmpeg，建议用 VLC 播放或手动转码:")
            print(f"  ffmpeg -i {output_path} -c:v libx264 {output_path.replace('.mp4', '_h264.mp4')}")
            return
        tmp_path = output_path + ".tmp.mp4"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", output_path,
                "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                "-pix_fmt", "yuv420p", tmp_path
            ], check=True, capture_output=True)
            shutil.move(tmp_path, output_path)
            print("\u2713 已自动转码为 H.264")
        except subprocess.CalledProcessError as e:
            print(f"\u26a0 ffmpeg 转码失败: {e.stderr.decode()}")

    def _batch_infer(self, images: list, prompt: str, style_lora_path: str, 
                     strength: float, lora_scale: float) -> list:
        """
        批量图像推理（可扩展为真正的 Batch 推理以榨干 AMD GPU 性能）
        当前采用串行调用以确保显存安全
        """
        results = []
        for img in images:
            result = self.image_processor.process(
                image=img,
                prompt=prompt,
                style_lora_path=style_lora_path,
                strength=strength,
                lora_scale=lora_scale
            )
            results.append(result)
        return results