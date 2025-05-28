
import tkinter as tk
from tkinter import ttk
import threading
import time

from fatigue_detection.drowsiness_detection_mediapipe import start_drowsiness_detection
#from driver_risk_alert_system.track_with_analytics import LaneTracker
from driver_risk_alert_system.lane_tracker_module import LaneTracker

class DriverSafetyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 駕駛安全輔助系統")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f4f7")

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TButton",
                             font=("Microsoft JhengHei", 14),
                             padding=10,
                             background="#1f77b4",
                             foreground="white")
        self.style.map("TButton", background=[('active', '#135c89')])

        label = tk.Label(self.root,
                         text="啟動整合系統：疲勞 + 車道辨識",
                         font=("Microsoft JhengHei", 20),
                         bg="#f0f4f7",
                         fg="#1f77b4")
        label.pack(pady=50)

        self.btn_run = ttk.Button(self.root,
                                  text="啟動",
                                  command=self.run_system)
        self.btn_run.pack(pady=20, ipadx=20)

    def run_system(self):
        shared_alert = [False]

        t1 = threading.Thread(target=start_drowsiness_detection, args=(shared_alert,), daemon=True)
        t2 = threading.Thread(target=LaneTracker(shared_alert).start, daemon=True)

        t1.start()
        time.sleep(3)  # 讓 webcam 優先開啟，3 秒後再執行追蹤
        t2.start()

def launch_app():
    root = tk.Tk()
    app = DriverSafetyGUI(root)
    root.mainloop()
