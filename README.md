# ArtisanAI

基于 Gradio + Diffusers + PyTorch (ROCm) 的 AI 图像生成应用。

## 项目架构

```
┌─────────────────────────┐
│        Gradio           │  ← Web 界面 / 交互层
│   (app/app.py)          │
├─────────────────────────┤
│       Diffusers         │  ← 模型加载 / 推理流水线
│   (Hugging Face)        │
├─────────────────────────┤
│   PyTorch + ROCm        │  ← 深度学习框架 / AMD GPU 加速
└─────────────────────────┘
```

- **Gradio**（顶层）：提供 Web UI，用户通过浏览器交互，输入参数并查看生成结果。
- **Diffusers**（中间层）：Hugging Face 的扩散模型库，封装 Stable Diffusion 等模型的加载、调度与推理流程。
- **PyTorch + ROCm**（底层）：PyTorch 深度学习框架，通过 ROCm 实现 AMD GPU 加速推理。

## 启动 Gradio 应用

```bash
# 安装依赖（清华源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动应用（监听 0.0.0.0:10000）
cd /workspace/ArtisanAI && python app/app.py
```

启动后通过 `http://localhost:10000` 访问。
