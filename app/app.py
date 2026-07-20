import gradio as gr


def greet(name: str) -> str:
    return f"你好, {name}!"


demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="姓名"),
    outputs=gr.Textbox(label="问候"),
    title="最简单的 Gradio 示例",
    description="输入一个名字,点击提交查看问候。",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
