from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import time
from collections import defaultdict, deque
import gc # 導入 gc 模組，用於垃圾回收

# 為了支援 risk_modules 導入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from risk_modules.risk_analyzer import *
from risk_modules.Land_detection import *
from risk_modules.warning_controller import *
import yaml

# 導入語音輸出模組
from speech_alert_system import generate_and_play_audio


class LaneTracker:
    def __init__(self, shared_alert):
        self.shared_alert = shared_alert
        # 歷史數據隊列長度保持不變，因為它們通常佔用記憶體較少
        self.object_history = defaultdict(lambda: deque(maxlen=5))
        self.risk_score_history = defaultdict(lambda: deque(maxlen=10))

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.yaml_path = os.path.join(self.current_dir, 'risk_modules', 'risk_params.yaml')

        with open(self.yaml_path, 'r', encoding='utf-8') as file:
            self.risk_config = yaml.safe_load(file)['risk_params']

        self.flow_roi_top = self.risk_config['optical_flow']['roi_top_ratio']
        self.flow_roi_bottom = self.risk_config['optical_flow']['roi_bottom_ratio']

        model_path = os.path.join(self.current_dir, "weight", "best2.pt")
        # 初始化 YOLO 模型，考慮在需要時調整推斷設備 (device)
        self.model = YOLO(model_path) 
            
        # 可以考慮將模型加載到 GPU 如果有並且記憶體足夠：
        # self.model = YOLO(model_path).to('cuda') 

        self.video_path = os.path.join(self.current_dir, "assets", "videoplayback.mp4")

        # 此變數與記憶體無直接關係，保留
        self.red_alert_active = False

    def estimate_self_speed(self, prev_gray, curr_gray):
        h, w = curr_gray.shape
        top = int(h * self.flow_roi_top)
        bottom = int(h * self.flow_roi_bottom)
        # 注意：此處創建的 flow 和 mag 陣列，如果非常大，也可能佔用記憶體
        # 但通常光流計算的 ROI 較小，影響有限
        flow = cv2.calcOpticalFlowFarneback(prev_gray[top:bottom], curr_gray[top:bottom],
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        
        # 顯式刪除不再需要的局部變數
        del flow
        return np.mean(mag)

    def start(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 獲取原始幀的寬高
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 定義處理幀的目標尺寸，顯著降低記憶體消耗
        # YOLOv8 模型 imgsz=320，所以輸入給模型的圖像會被縮放到 320x320。
        # 但如果原始幀很大，縮放過程也可能耗費記憶體。
        # 這裡設置一個中間尺寸，可以減少整體記憶體負擔。
        # 保持長寬比，例如將寬度固定為 640 或 480，並計算相應的高度。
        target_width = 640 # 可以嘗試 480, 320 等更小的值
        target_height = int(original_height * (target_width / original_width))
        
        print(f"Original Frame Size: {original_width}x{original_height}")
        print(f"Processing Frame Size: {target_width}x{target_height}")

        frame_skip = 5
        frame_idx = 0
        prev_smoothed_speed = 0
        gray_history = deque(maxlen=3) # 歷史灰度幀，保持少量
        speed_fail_count = 0
        MAX_FAIL_COUNT = 3

        # 調整 VideoWriter 的輸出尺寸為處理後的尺寸
        out = cv2.VideoWriter("demo.mp4",
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              int(fps // frame_skip),
                              (target_width, target_height)) # 輸出尺寸與處理尺寸一致

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start_time = time.time()
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    # 在跳過幀時，顯式釋放當前幀的記憶體
                    del frame 
                    continue

                # 將讀取的原始幀縮小到目標尺寸
                processed_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                del frame # 釋放原始大幀的記憶體

                try:
                    # process_frame 可能會創建新的圖像，注意其記憶體使用
                    # 如果 process_frame 內部也處理了縮放，這裡可以調整
                    result = process_frame(processed_frame) # 使用縮小後的幀
                    # process_frame 返回的 result 中可能包含新的圖像數據
                    processed_frame, _, scene_valid, left_line, right_line = result 
                    # 假設 result[0] 是處理後的幀，如果它複製了數據，考慮減少複製
                    # 在這裡，processed_frame 變成了結果幀，所以原始 processed_frame 的引用會被替換，舊數據會被GC
                    
                    if not scene_valid:
                        # 在跳過時確保釋放處理過的幀
                        del processed_frame
                        continue

                    curr_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                    gray_history.append(curr_gray) # curr_gray 也是縮小後的尺寸

                    if len(gray_history) >= 2:
                        speeds = []
                        for i in range(len(gray_history) - 1):
                            try:
                                s = self.estimate_self_speed(gray_history[i], gray_history[i + 1])
                                speeds.append(s)
                            except:
                                pass
                        raw_speed = np.mean(speeds) if speeds else prev_smoothed_speed
                        speed_fail_count = 0 if speeds else speed_fail_count + 1
                    else:
                        raw_speed = prev_smoothed_speed
                        speed_fail_count += 1

                    alpha = 0.3
                    speed = alpha * raw_speed + (1 - alpha) * prev_smoothed_speed

                    if speed < 0.05:
                        speed = prev_smoothed_speed
                    if speed_fail_count >= MAX_FAIL_COUNT:
                        speed *= 0.5

                    prev_smoothed_speed = speed
                    # roi_dict 和 scale 都是計算結果，通常記憶體佔用不大
                    roi_dict, scale = get_lane_roi_dynamic(left_line, right_line, processed_frame.shape, speed=speed)

                except Exception as e:
                    print(f"[❌ Speed block error] {e}")
                    # 在錯誤發生時也嘗試釋放可能的記憶體
                    if 'processed_frame' in locals() and processed_frame is not None:
                        del processed_frame
                    continue

                # YOLO 模型推斷，輸入尺寸 imgsz=320 是比較小的，有助於控制記憶體
                # 但是 results[0].boxes.data.cpu().numpy() 會將結果複製到 CPU 記憶體
                results = self.model.track(source=processed_frame, imgsz=320, persist=True, show=False, stream=False)
                
                # 顯式刪除 processed_frame，因為 YOLO 推斷通常會內部複製
                # 或者在處理完 results 後再刪除，取決於 draw_risk_overlay 是否需要 processed_frame
                # 這裡，draw_risk_overlay 接收 annotated_frame，所以 processed_frame 可以在此處刪除
                del processed_frame 


                risky_objects = []
                seen_ids = set()
                
                is_any_red_risk_active = False 

                # 遍歷結果並處理風險物件
                for r in results[0].boxes.data.cpu().numpy():
                    if len(r) < 7:
                        continue

                    track_id = int(r[6])
                    if track_id in seen_ids:
                        continue
                    seen_ids.add(track_id)

                    x1, y1, x2, y2 = map(int, r[:4])
                    center = get_center((x1, y1, x2, y2))
                    self.object_history[track_id].append(center)
                    speed, is_jump, smoothed_center, vx = compute_speed(track_id, center, self.object_history, fps=fps)

                    roi_level = get_roi_level_bbox((x1, y1, x2, y2), roi_dict)
                    if roi_level is None:
                        continue

                    score, level, stay = analyze_risk(track_id, smoothed_center, roi_level, speed, is_jump, vx)
                    self.risk_score_history[track_id].append(score)
                    smoothed_score = np.mean(self.risk_score_history[track_id])

                    if smoothed_score > self.risk_config['score_threshold']['high']:
                        level = "high"
                    elif smoothed_score > self.risk_config['score_threshold']['mid']:
                        level = "mid"
                    else:
                        level = "low"

                    risky_objects.append((x1, y1, x2, y2, track_id, smoothed_score, level))

                    if level == "mid": 
                        print(f"⚠️ 提醒觸發！ID={track_id}, Level={level}, Score={smoothed_score:.2f}")
                        generate_and_play_audio("距離有點近了，建議您放慢速度", "risk_side_alert", cooldown_seconds=5)
                        
                    if level == "high":
                        is_any_red_risk_active = True 

                # 在所有物件處理完畢後，根據 is_any_red_risk_active 來控制紅色警報的語音
                if is_any_red_risk_active:
                    generate_and_play_audio("已進入危險範圍，請立即減速", "risk_high_alert", cooldown_seconds=1.5) 
                    self.red_alert_active = True 
                else:
                    self.red_alert_active = False 

                annotated_frame = results[0].plot() # YOLO 的 plot() 會返回一個新的圖像
                del results # 處理完 results 後顯式刪除，釋放記憶體

                draw_risk_overlay(annotated_frame, risky_objects, roi_dict) 

                cv2.putText(annotated_frame, f"ROI Scale: {scale:.3f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                cv2.putText(annotated_frame, f"Speed: {speed:.2f}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                if self.shared_alert[0]:
                    cv2.putText(annotated_frame, "DROWSINESS ALERT!", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    generate_and_play_audio("偵測到你疲勞了，請保持清醒或稍作休息", "drowsiness_alert", cooldown_seconds=5)

                frame_time = time.time() - frame_start_time
                fps = 1.0 / frame_time
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                out.write(annotated_frame)
                cv2.imshow("Tracked Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # 每次循環結束時，嘗試強制垃圾回收
                del annotated_frame # 釋放顯示和寫入後的幀
                gc.collect() # 嘗試強制垃圾回收

        except KeyboardInterrupt:
            print("\n[🛑 使用者中斷 Ctrl+C]")

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            gc.collect() # 程式結束前再次進行垃圾回收