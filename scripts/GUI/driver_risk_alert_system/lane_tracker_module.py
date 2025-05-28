from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import time
from collections import defaultdict, deque

# 為了支援 risk_modules 導入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from risk_modules.risk_analyzer import *
from risk_modules.Land_detection import *
from risk_modules.warning_controller import *
import yaml

# 新增語音輸出模組的導入
from speech_alert_system import generate_and_play_audio


class LaneTracker:
    def __init__(self, shared_alert):
        self.shared_alert = shared_alert
        self.object_history = defaultdict(lambda: deque(maxlen=5))
        self.risk_score_history = defaultdict(lambda: deque(maxlen=10))

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.yaml_path = os.path.join(self.current_dir, 'risk_modules', 'risk_params.yaml')

        with open(self.yaml_path, 'r', encoding='utf-8') as file:
            self.risk_config = yaml.safe_load(file)['risk_params']

        self.flow_roi_top = self.risk_config['optical_flow']['roi_top_ratio']
        self.flow_roi_bottom = self.risk_config['optical_flow']['roi_bottom_ratio']

        model_path = os.path.join(self.current_dir, "weight", "best2.pt")
        self.model = YOLO(model_path)

        self.video_path = os.path.join(self.current_dir, "assets", "videoplayback.mp4")

    def estimate_self_speed(self, prev_gray, curr_gray):
        h, w = curr_gray.shape
        top = int(h * self.flow_roi_top)
        bottom = int(h * self.flow_roi_bottom)
        flow = cv2.calcOpticalFlowFarneback(prev_gray[top:bottom], curr_gray[top:bottom],
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.linalg.norm(flow, axis=2)
        return np.mean(mag)

    def start(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = 5
        frame_idx = 0
        prev_smoothed_speed = 0
        gray_history = deque(maxlen=3)
        speed_fail_count = 0
        MAX_FAIL_COUNT = 3

        out = cv2.VideoWriter("demo.mp4",
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              int(fps // frame_skip),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start_time = time.time()
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                try:
                    result = process_frame(frame)
                    frame, _, scene_valid, left_line, right_line = result
                    if not scene_valid:
                        continue

                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_history.append(curr_gray)

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
                    roi_dict, scale = get_lane_roi_dynamic(left_line, right_line, frame.shape, speed=speed)

                except Exception as e:
                    print(f"[❌ Speed block error] {e}")
                    continue

                results = self.model.track(source=frame, imgsz=320, persist=True, show=False, stream=False)

                risky_objects = []
                seen_ids = set()

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

                    # 根據 smoothed_score 重新確認 level，因為 analyze_risk 返回的 level 只是初步判斷
                    if smoothed_score > self.risk_config['score_threshold']['high']:
                        level = "high"
                    elif smoothed_score > self.risk_config['score_threshold']['mid']:
                        level = "mid"
                    else:
                        level = "low"

                    risky_objects.append((x1, y1, x2, y2, track_id, smoothed_score, level))

                    now = time.time()
                    if should_warn(track_id, now, level, smoothed_score, stay, self.risk_config):
                        print(f"⚠️ 提醒觸發！ID={track_id}, Level={level}, Score={smoothed_score:.2f}")
                        # 根據風險等級播放不同的語音
                        if level == "yellow": # 黃色警示區 (來自 should_warn 中的 'yellow' level)
                            generate_and_play_audio("距離有點近了，建議您放慢速度", "risk_side_alert", cooldown_seconds=5)
                        elif level == "red": # 紅色警示區 (來自 should_warn 中的 'red' level)
                            generate_and_play_audio("已進入危險範圍，請立即減速", "risk_high_alert", cooldown_seconds=5)


                annotated_frame = results[0].plot()
                draw_risk_overlay(annotated_frame, risky_objects, roi_dict) # 此處 draw_risk_overlay 不再負責語音

                cv2.putText(annotated_frame, f"ROI Scale: {scale:.3f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                cv2.putText(annotated_frame, f"Speed: {speed:.2f}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                if self.shared_alert[0]:
                    cv2.putText(annotated_frame, "DROWSINESS ALERT!", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    # 疲勞警示語音：直接在這裡觸發，因為這是疲勞模組的輸出
                    generate_and_play_audio("你是來開車還是來睡覺的", "drowsiness_alert", cooldown_seconds=5)


                frame_time = time.time() - frame_start_time
                fps = 1.0 / frame_time
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                out.write(annotated_frame)
                cv2.imshow("Tracked Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n[🛑 使用者中斷 Ctrl+C]")

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()