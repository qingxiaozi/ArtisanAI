import os
import random
import cv2
import numpy as np
import subprocess
import shutil
from PIL import Image
from tqdm import tqdm
from scenedetect import open_video, SceneManager, ContentDetector
from core.style_transfer import StyleTransferProcessor
from utils.gpu_utils import clear_gpu_cache


class VideoTransferProcessor:
    def __init__(self, device: str = "cuda"):
        self.image_processor = StyleTransferProcessor(device=device)

    def process(self, video_path: str, output_path: str, prompt: str,
                style_lora_path: str = None, strength: float = 0.4,
                lora_scale: float = 0.8, model_dir: str = None,
                scene_threshold: float = 27.0):
        """视频风格转换。检测场景，场景内固定 seed 保持风格一致。"""
        if not self.image_processor.pipe:
            self.image_processor.load_model(model_dir=model_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out = self._init_writer(output_path, fps, width, height)

        scenes = self._detect_scenes(video_path, scene_threshold)

        for scene_start, scene_end in tqdm(scenes, desc="处理场景"):
            seed = random.randint(0, 2**31 - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene_start)

            for _ in range(scene_end - scene_start):
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None or frame_bgr.size == 0:
                    break
                if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
                    continue
                try:
                    styled = self._stylize_frame(frame_bgr, prompt, style_lora_path,
                                                 strength, lora_scale, width, height,
                                                 seed=seed)
                    out.write(styled)
                except Exception as e:
                    print(f"\n帧处理异常: {e}")
                    continue

            clear_gpu_cache()

        cap.release()
        out.release()
        print(f"视频处理完成，已保存至: {output_path}")
        self._ensure_h264_compatible(output_path)

    def _init_writer(self, path, fps, width, height):
        for codec in ["avc1", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(path, fourcc, fps, (width, height))
            if out.isOpened():
                return out
        raise RuntimeError("无法创建视频写入器")

    def _detect_scenes(self, video_path: str, threshold: float):
        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(video)
        return [(s.get_frames(), e.get_frames()) for s, e in manager.get_scene_list()]

    def _stylize_frame(self, frame_bgr, prompt, style_lora_path,
                       strength, lora_scale, width, height, seed=None):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.image_processor.process(
            image=Image.fromarray(frame_rgb),
            prompt=prompt,
            style_lora_path=style_lora_path,
            strength=strength,
            lora_scale=lora_scale,
            seed=seed,
        )
        result = result.resize((width, height), Image.LANCZOS)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    def _ensure_h264_compatible(self, output_path: str):
        if not shutil.which("ffmpeg"):
            return
        tmp_path = output_path + ".tmp.mp4"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", output_path,
                "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                "-pix_fmt", "yuv420p", tmp_path
            ], check=True, capture_output=True)
            shutil.move(tmp_path, output_path)
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg 转码失败: {e.stderr.decode()}")
