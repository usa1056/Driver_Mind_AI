import gradio as gr
import subprocess
import os

def run_detection(video_file=None):
    command = ["python", "drowsiness_detection.py"]
    if video_file:
        command += ["--video", video_file]
    subprocess.Popen(command)
    return "模型已啟動，請查看本機視窗！"

iface = gr.Interface(
    fn=run_detection,
    inputs=[
        gr.File(label="上傳影片檔（或留空使用攝影機）")
    ],
    outputs="text",
    title="駕駛疲勞偵測系統",
    description="可選擇影片或啟用 webcam 來進行眼睛閉合與打哈欠偵測"
)

iface.launch()
