from ultralytics import YOLO
import cv2
import numpy as np
import os
from collections import defaultdict, deque
from risk_modules.risk_analyzer import get_center, get_roi_level, analyze_risk, draw_risk_overlay, compute_speed
from risk_modules.Land_detection import process_frame
from risk_modules.risk_plotter import draw_risk_curve

# ✅ 載入 YOLO 模型
model = YOLO("yolov8n.pt")

# ✅ 初始化歷史紀錄（速度與分數）
object_history = defaultdict(lambda: deque(maxlen=5))  # 儲存中心點歷史
risk_score_history = defaultdict(list)  # 儲存風險分數歷史

# ✅ 讀取影片
video_path = "assets/videoplayback.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# ✅ 輸出影片設定
out = cv2.VideoWriter("output.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      int(fps),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# ✅ ROI 狀態追蹤
missing_counter = 0
previous_roi_dict = None

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 預處理：取得 ROI 區域 + 場景是否有效
        frame, roi_dict, scene_valid = process_frame(frame)
        if not scene_valid:
            print("[⚠️ 非車道場景] 本幀略過風險分析")
            continue

        if not roi_dict or not all(k in roi_dict for k in ["high", "mid", "low"]):
            print("[⚠️ ROI LOST] 使用前一幀 ROI")
            missing_counter += 1
            if previous_roi_dict:
                roi_dict = previous_roi_dict
            else:
                print("[⛔] 無法取得有效 ROI，跳過本幀分析")
                continue
        else:
            previous_roi_dict = roi_dict
            missing_counter = 0

        # ✅ YOLOv8 目標追蹤
        results = model.track(source=frame, persist=True, show=False, stream=False)

        # ✅ 分析每個追蹤物件
        risky_objects = []
        for r in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls, track_id = map(int, r[:7])
            center = get_center((x1, y1, x2, y2))
            object_history[track_id].append(center)

            roi_level = get_roi_level(center, roi_dict)
            if roi_level is None:
                continue

            speed, is_jump, smoothed_center = compute_speed(track_id, center, object_history, fps=fps)
            score, level = analyze_risk(track_id, smoothed_center, roi_level, speed, is_jump)

            risk_score_history[track_id].append(score)
            risky_objects.append((x1, y1, x2, y2, track_id, score, level))

        # ✅ 畫出 YOLO 框與 ROI 區域
        annotated_frame = results[0].plot()
        draw_risk_overlay(annotated_frame, risky_objects, roi_dict)

        # ❌ 停用軌跡線畫圖（畫面較乾淨）
        # for track_id, history in object_history.items():
        #     if len(history) >= 2:
        #         pts = np.array(history, dtype=np.int32)
        #         cv2.polylines(annotated_frame, [pts], isClosed=False, color=(255, 255, 0), thickness=2)

        # ✅ 顯示風險曲線圖（只畫分數最高的 high 物件）
        high_risks = [obj for obj in risky_objects if obj[-1] == "high"]
        if high_risks:
            top_risk = max(high_risks, key=lambda x: x[5])  # 按 score 排序
            top_id = top_risk[4]
            scores = risk_score_history[top_id]
            annotated_frame = draw_risk_curve(annotated_frame, top_id, scores, fps=fps)

        # ✅ 顯示畫面與寫入影片
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
