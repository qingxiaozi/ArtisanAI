# pip install accelerate transformers safetensors diffusers

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "models")

import torch
import numpy as np
from PIL import Image

from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
import cv2

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0-small",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()


def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


seed = 42
generator = torch.Generator(device="cuda").manual_seed(seed)

prompt = "A robot, 4k photo"
controlnet_conditioning_scale = 0.5

def extract_keyframes(frames, threshold=0.3):
    """Extract keyframes using HSV histogram difference.
    Uses Bhattacharyya distance (0 = identical, 1 = complete mismatch).
    Returns list of segments [{start: int, end: int}, ...] where each start is a keyframe.
    """
    segments = []
    seg_start = 0
    prev_hist = None
    for i, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None:
            dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if dist > threshold:
                segments.append({"start": seg_start, "end": i - 1})
                seg_start = i
        prev_hist = hist
    segments.append({"start": seg_start, "end": len(frames) - 1})
    return segments


video_path = os.path.join(os.path.dirname(__file__), "videos", "1080-25-低.mp4")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Step 1: read all frames
print("Reading all frames...")
raw_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    raw_frames.append(frame)
cap.release()

# Step 2: extract keyframes → segments
segments = extract_keyframes(raw_frames)
print(f"Found {len(segments)} segments: {segments}")

# Step 3: process each segment, using keyframe as reference
output_dir = os.path.join(os.path.dirname(__file__), "output_videos")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "output.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (1024, 1024))

keyframes = []
for seg_idx, seg in enumerate(segments):
    keyframe_bgr = raw_frames[seg["start"]]
    keyframe_rgb = cv2.cvtColor(keyframe_bgr, cv2.COLOR_BGR2RGB)
    keyframe_pil = Image.fromarray(keyframe_rgb).resize((1024, 1024))
    keyframes.append(keyframe_pil)
    keyframe_depth = get_depth_map(keyframe_pil)
    print(f"Segment {seg_idx + 1}/{len(segments)}: frames {seg['start']}-{seg['end']}")
    # for i in range(seg["start"], seg["end"] + 1):
    #     frame_rgb = cv2.cvtColor(raw_frames[i], cv2.COLOR_BGR2RGB)
    #     frame_pil = Image.fromarray(frame_rgb).resize((1024, 1024))
    #     result = pipe(
    #         prompt,
    #         image=frame_pil,
    #         control_image=keyframe_depth,
    #         strength=0.99,
    #         num_inference_steps=50,
    #         controlnet_conditioning_scale=controlnet_conditioning_scale,
    #         generator=generator,
    #     ).images[0]
    #     out_frame = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    #     out.write(out_frame)

out.release()

# Save keyframes
keyframes_dir = os.path.join(output_dir, "keyframes")
os.makedirs(keyframes_dir, exist_ok=True)
for i, kf in enumerate(keyframes):
    kf.save(os.path.join(keyframes_dir, f"keyframe_{i:04d}.png"))
print(f"Saved {len(keyframes)} keyframes to {keyframes_dir}")
print(f"Saved video to {output_path}")