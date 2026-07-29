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
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

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

# Load RAFT for optical flow
raft_weights = Raft_Large_Weights.DEFAULT
raft_transforms = raft_weights.transforms()
raft = raft_large(weights=raft_weights).to("cuda").eval()


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

prompt = "anime style, Japanese animation, cel shading, clean lines, vibrant colors, detailed character design"
controlnet_conditioning_scale = 0.5


def warp_image(image_tensor, flow):
    """Warp image_tensor (C, H, W) using backward optical flow (H, W, 2).
    Backward flow: for pixel (x,y) in target, sample source at (x+flow_x, y+flow_y).
    """
    _, h, w = image_tensor.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).float().to(flow.device)
    sample_grid = grid + flow
    sample_grid[..., 0] = 2.0 * sample_grid[..., 0] / max(w - 1, 1) - 1.0
    sample_grid[..., 1] = 2.0 * sample_grid[..., 1] / max(h - 1, 1) - 1.0
    warped = F.grid_sample(
        image_tensor.unsqueeze(0), sample_grid.unsqueeze(0),
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    return warped.squeeze(0)


def compute_backward_flow(raft_model, frame_a_bgr, frame_b_bgr):
    """Compute backward optical flow from frame_b to frame_a.
    Returns flow (H, W, 2) float32.
    """
    def to_raft_input(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).to("cuda")
        return t.unsqueeze(0)
    img_a = to_raft_input(frame_a_bgr)
    img_b = to_raft_input(frame_b_bgr)
    img_a, img_b = raft_transforms(img_a, img_b)
    with torch.no_grad():
        flows = raft_model(img_b, img_a)
    flow = flows[-1].squeeze(0).permute(1, 2, 0)
    return flow


def extract_keyframes(frames, threshold=0.045):
    """Extract keyframes using HSV histogram difference.
    Uses Bhattacharyya distance (0 = identical, 1 = complete mismatch).
    Higher threshold → fewer keyframes, lower threshold → more keyframes.
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
max_frames = int(fps * 10)
raw_frames = []
while len(raw_frames) < max_frames:
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
    start, end = seg["start"], seg["end"]
    # Save original keyframe
    keyframe_bgr = cv2.resize(raw_frames[start], (1024, 1024))
    keyframe_rgb = cv2.cvtColor(keyframe_bgr, cv2.COLOR_BGR2RGB)
    keyframe_pil = Image.fromarray(keyframe_rgb)
    keyframes.append(keyframe_pil)

    # Use original keyframe directly (no pipe)
    keyframe_depth = get_depth_map(keyframe_pil)
    print(f"Segment {seg_idx + 1}/{len(segments)}: frames {start}-{end}")
    # result = pipe(
    #     prompt,
    #     image=keyframe_pil,
    #     control_image=keyframe_depth,
    #     strength=0.5,
    #     num_inference_steps=50,
    #     controlnet_conditioning_scale=controlnet_conditioning_scale,
    #     generator=generator,
    # ).images[0]
    keyframe_tensor = torch.from_numpy(keyframe_rgb).permute(2, 0, 1).float().to("cuda") / 255.0
    out.write(keyframe_bgr)

    # Warp each frame directly from keyframe via optical flow
    for i in range(start + 1, end + 1):
        curr_bgr = raw_frames[i]
        curr_resized = cv2.resize(curr_bgr, (1024, 1024))
        flow = compute_backward_flow(raft, keyframe_bgr, curr_resized)
        warped = warp_image(keyframe_tensor, flow)
        warped_np = (warped.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_frame = cv2.cvtColor(warped_np, cv2.COLOR_RGB2BGR)
        out.write(out_frame)

out.release()

# Save keyframes
keyframes_dir = os.path.join(output_dir, "keyframes")
os.makedirs(keyframes_dir, exist_ok=True)
for i, kf in enumerate(keyframes):
    kf.save(os.path.join(keyframes_dir, f"keyframe_{i:04d}.png"))
print(f"Saved {len(keyframes)} keyframes to {keyframes_dir}")
print(f"Saved video to {output_path}")