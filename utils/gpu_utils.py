# utils/gpu_utils.py
import torch

def clear_gpu_cache():
    """清理 AMD GPU 显存缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU 缓存已清理。")