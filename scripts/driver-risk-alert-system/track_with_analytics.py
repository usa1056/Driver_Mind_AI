from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
import time
from collections import defaultdict, deque
from risk_modules.risk_analyzer import *
from risk_modules.Land_detection import *
from risk_modules.warning_controller import *
import yaml

with open('risk_modules/risk_params.yaml', 'r', encoding='utf-8') as file:
    risk_config = yaml.safe_load(file)['risk_params']

flow_roi_top = risk_config['optical_flow']['roi_top_ratio']
flow_roi_bottom = risk_config['optical_flow']['roi_bottom_ratio']

model = YOLO("weight\best.pt")

def estimate_self_speed(prev_gray, curr_gray, roi_top_ratio=0.0, roi_bottom_ratio=0.3):
    h, w = curr_gray.shape
    top = int(h * roi_top_ratio)
    bottom = int(h * roi_bottom_ratio)
    flow = cv2.calcOpticalFlowFarneback(prev_gray[top:bottom], curr_gray[top:bottom],
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2)
    return np.mean(mag)

object_history = defaultdict(lambda: deque(maxlen=5))
risk_score_history = defaultdict(lambda: deque(maxlen=risk_config['smoothing']['history_length']))

video_path = "assets/danger.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_skip = 5
frame_idx = 0

out = cv2.VideoWriter("demo.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      int(fps // frame_skip),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

prev_smoothed_speed = 0
gray_history = deque(maxlen=3)
speed_fail_count = 0
MAX_FAIL_COUNT = 3

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
                        s = estimate_self_speed(gray_history[i], gray_history[i + 1],
                                                roi_top_ratio=flow_roi_top,
                                                roi_bottom_ratio=flow_roi_bottom)
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

        results = model.track(source=frame, imgsz=320, persist=True, show=False, stream=False)

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
            object_history[track_id].append(center)
            speed, is_jump, smoothed_center, vx = compute_speed(track_id, center, object_history, fps=fps)

            roi_level = get_roi_level_bbox((x1, y1, x2, y2), roi_dict)
            if roi_level is None:
                continue

            score, level, stay = analyze_risk(track_id, smoothed_center, roi_level, speed, is_jump, vx)
            risk_score_history[track_id].append(score)
            smoothed_score = np.mean(risk_score_history[track_id])

            if smoothed_score > risk_config['score_threshold']['high']:
                level = "high"
            elif smoothed_score > risk_config['score_threshold']['mid']:
                level = "mid"
            else:
                level = "low"

            risky_objects.append((x1, y1, x2, y2, track_id, smoothed_score, level))

            now = time.time()
            if should_warn(track_id, now, level, smoothed_score, stay, risk_config):
                print(f"⚠️ 提醒觸發！ID={track_id}, Level={level}, Score={smoothed_score:.2f}")

        annotated_frame = results[0].plot()
        draw_risk_overlay(annotated_frame, risky_objects, roi_dict)

        cv2.putText(annotated_frame, f"ROI Scale: {scale:.3f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(annotated_frame, f"Speed: {speed:.2f}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

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