from huggingface_hub import snapshot_download

# 以 Canny 模型为例，下载到本地指定目录
snapshot_download(
    repo_id="diffusers/controlnet-canny-sdxl-1.0",
    local_dir="./models/controlnet-canny-sdxl-1.0"
)