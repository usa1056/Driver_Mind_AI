from tkinter import messagebox
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
import subprocess
import os

class DriverSafetyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("駕駛安全輔助系統")
        self.root.geometry("800x600")

        self.label = tk.Label(self.root, text="請選擇功能", font=("Arial", 20))
        self.label.pack(pady=20)

        self.btn_drowsy = tk.Button(self.root, text="即時疲勞偵測 (Webcam)", command=self.run_drowsiness, height=2, width=30)
        self.btn_drowsy.pack(pady=10)

        self.btn_lane = tk.Button(self.root, text="影片道路辨識 (上傳影片)", command=self.run_lane_detection, height=2, width=30)
        self.btn_lane.pack(pady=10)

    def run_drowsiness(self):
        def run():
            try:
                script_path = os.path.join(os.path.dirname(__file__), "fatigue_detection", "drowsiness_detection.py")
                subprocess.run(["python", script_path], check=True)
            except subprocess.CalledProcessError as e:
                messagebox.showerror("錯誤", f"程式執行失敗：\n{e}")
            except FileNotFoundError:
                messagebox.showerror("錯誤", "找不到疲勞偵測程式！請檢查路徑。")
        threading.Thread(target=run).start()

    def run_lane_detection(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            script_path = os.path.join(os.path.dirname(__file__), "lane_detection", "Land_detection.py")
            threading.Thread(target=lambda: subprocess.Popen(["python", script_path, file_path])).start()

def launch_app():
    root = tk.Tk()
    app = DriverSafetyGUI(root)
    root.mainloop()
