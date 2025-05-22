import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import subprocess
import os

class DriverSafetyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 駕駛安全輔助系統")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f4f7")  # 淺藍灰色背景

        self.style = ttk.Style()
        self.style.theme_use('clam')  # 使用現代風格主題
        self.style.configure("TButton",
                             font=("Microsoft JhengHei", 14),
                             padding=10,
                             background="#1f77b4",
                             foreground="white")
        self.style.map("TButton",
                       background=[('active', '#135c89')])

        # 標題區
        title_frame = tk.Frame(self.root, bg="#f0f4f7")
        title_frame.pack(pady=40)
        label = tk.Label(title_frame,
                         text="請選擇功能",
                         font=("Microsoft JhengHei", 24, "bold"),
                         bg="#f0f4f7",
                         fg="#1f77b4")
        label.pack()

        # 功能按鈕區
        btn_frame = tk.Frame(self.root, bg="#f0f4f7")
        btn_frame.pack(pady=20)

        self.btn_drowsy = ttk.Button(btn_frame,
                                     text="即時疲勞偵測 (Webcam)",
                                     command=self.run_drowsiness)
        self.btn_drowsy.pack(pady=15, ipadx=20)

        self.btn_lane = ttk.Button(btn_frame,
                                   text="影片道路辨識 (上傳影片)",
                                   command=self.run_lane_detection)
        self.btn_lane.pack(pady=15, ipadx=20)

    def run_drowsiness(self):
        def run():
            try:
                #script_path = os.path.join(os.path.dirname(__file__), "fatigue_detection", "drowsiness_detection.py")
                script_path = os.path.join(os.path.dirname(__file__), "fatigue_detection", "drowsiness_detection_mediapipe.py")
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
