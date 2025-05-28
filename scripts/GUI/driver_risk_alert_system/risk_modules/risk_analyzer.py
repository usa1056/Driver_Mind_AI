import numpy as np
import cv2
from collections import defaultdict, deque
import math
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  # risk_analyzer.py 的絕對路徑
yaml_path = os.path.join(current_dir, 'risk_params.yaml')

with open(yaml_path, 'r', encoding='utf-8') as file:
    risk_config = yaml.safe_load(file)['risk_params']


# 儲存每個物件的停留狀態與歷史
object_state = defaultdict(lambda: {"last_level": None, "stay_counter": 0})
object_history = defaultdict(lambda: deque(maxlen=2))  # 新增：紀錄物件歷史中心點

def get_center(bbox):
    """計算追蹤框中心點"""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def bbox_intersects_roi(bbox, roi_polygon):
    # 將 bbox 轉為四邊形頂點
    x1, y1, x2, y2 = bbox
    box = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.int32)

    # 檢查兩個多邊形是否有交集（cv2 返回的 area > 0）
    inter_area, _ = cv2.intersectConvexConvex(box.astype(np.float32), roi_polygon.astype(np.float32))
    return inter_area > 0

def get_roi_level_bbox(bbox, roi_dict):
    for level in ["high", "side_right", "side_left", "mid", "low"]:
        if level in roi_dict and bbox_intersects_roi(bbox, roi_dict[level]):
            return level
    return None


# 每個追蹤 ID 對應的移動歷史中心點
object_history = defaultdict(lambda: deque(maxlen=2))

def compute_speed(track_id, current_center, object_history, fps=30):
    """
    計算物體移動速度（像素/秒），防止追蹤異常導致爆衝
    """
    jump_threshold = risk_config['speed']['jump_threshold']
    max_speed = risk_config['speed']['max_speed']

    history = object_history.get(track_id)

    if not history:
        history = deque(maxlen=5)
        history.append(current_center)
        object_history[track_id] = history
        return 0.0, False, current_center

    last_center = history[-1]
    distance = ((current_center[0] - last_center[0]) ** 2 + (current_center[1] - last_center[1]) ** 2) ** 0.5
    is_jump = distance > jump_threshold

    smoothed_center = last_center if is_jump else current_center
    history.append(smoothed_center)  # deque 自動維持長度，不需切片

    speed = min(distance * fps, max_speed)
    vx = current_center[0] - last_center[0]

    return speed, is_jump, smoothed_center, vx


# 全域記憶每個物體的靜止幀數
static_counter = defaultdict(int)

def decay_static_score(track_id, score, speed, config):
    if speed < config['decay']['speed_threshold']:
        static_counter[track_id] += 1
        if static_counter[track_id] >= config['decay']['decay_frame_threshold']:
            score *= config['decay']['decay_rate']
    else:
        static_counter[track_id] = 0  # 一動就歸零
    
    print(f"[Decay Triggered] ID={track_id}, static_frame={static_counter[track_id]}, score={score:.2f}")
    return score


def analyze_risk(track_id, center, roi_level, speed, is_jump, vx=0):
    state = object_state[track_id]

    if roi_level != state["last_level"]:
        state["stay_counter"] = 1
    else:
        if not is_jump:
            state["stay_counter"] += 1
        elif risk_config['id_stability']['decay_on_jump']:
            decay = risk_config['id_stability']['decay_rate']
            state["stay_counter"] = max(1, state["stay_counter"] - decay)

    state["last_level"] = roi_level
    stay = state["stay_counter"]

    # 讀參數
    gamma = risk_config['speed']['gamma']
    base = risk_config['base_score'][roi_level]
    stay_weight = risk_config['stay_weight'][roi_level]

    # 計算 log(speed)
    log_speed = np.log1p(speed) if risk_config['speed']['log_scale'] else speed

    # 核心風險分數公式
    horiz_speed_weight = 0.5
    score = base + stay_weight * stay + gamma * log_speed + 0.5 * abs(vx)

    # 衰退處理（分級前）
    score = decay_static_score(track_id, score, speed, risk_config)

    # 分級邏輯
    if score > risk_config['score_threshold']['high']:
        level = "high"
    elif score > risk_config['score_threshold']['mid']:
        level = "mid"
    else:
        level = "low"
    
    print(f"[track_id: {track_id}] ROI={roi_level}, stay={stay}, speed={speed:.2f}, score={score:.2f}, level={level}")

    return score, level, stay


def draw_risk_overlay(frame, risky_objects, roi_dict):
    """畫出 high 區風險框 + side_right 橘色區塊"""

    # 畫 side_right 色塊（橘色）
    if "side_right" in roi_dict:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_dict["side_right"]], color=(0, 165, 255))  # 橘色 BGR
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 畫 side_left 色塊（橘色）
    if "side_left" in roi_dict:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_dict["side_left"]], color=(0, 165, 255))
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for (x1, y1, x2, y2, track_id, risk_score, risk_level) in risky_objects:
        if risk_level in ["high", "side_right", "side_left"]:
            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"RISK ID: {track_id} ({risk_score:.1f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # # 畫高風險物件的紅框與 ID 分數
    # for (x1, y1, x2, y2, track_id, risk_score, risk_level) in risky_objects:
    #     if risk_level == "high":
    #         color = (0, 0, 255)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #         cv2.putText(frame, f"RISK ID: {track_id} ({risk_score:.1f})", (x1, y1 - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

