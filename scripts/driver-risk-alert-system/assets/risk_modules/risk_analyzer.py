import numpy as np
import cv2
from collections import defaultdict, deque
import math
import yaml

with open('risk_modules/risk_params.yaml', 'r', encoding='utf-8') as file:
    risk_config = yaml.safe_load(file)['risk_params']


# 儲存每個物件的停留狀態與歷史
object_state = defaultdict(lambda: {"last_level": None, "stay_counter": 0})
object_history = defaultdict(lambda: deque(maxlen=2))  # ✅ 新增：紀錄物件歷史中心點

def get_center(bbox):
    """計算追蹤框中心點"""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def get_roi_level(center, roi_dict):
    """判斷中心點落在哪個 ROI 區域"""
    for level in ["high", "mid", "low"]:
        if cv2.pointPolygonTest(roi_dict[level], center, False) >= 0:
            return level
    return None  # 沒落在任何 ROI 內

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
    history.append(smoothed_center)  # ✅ deque 自動維持長度，不需切片

    speed = min(distance * fps, max_speed)
    return speed, is_jump, smoothed_center


def analyze_risk(track_id, center, roi_level, speed, is_jump):
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
    score = base + stay_weight * stay + gamma * log_speed

    # 分級邏輯
    if score > risk_config['score_threshold']['high']:
        level = "high"
    elif score > risk_config['score_threshold']['mid']:
        level = "mid"
    else:
        level = "low"
    
    print(f"[track_id: {track_id}] ROI={roi_level}, stay={stay}, speed={speed:.2f}, score={score:.2f}, level={level}")

    return score, level



def draw_risk_overlay(frame, risky_objects, roi_dict):
    """畫出 ROI 區域與高風險物件標示"""
    color_map = {
        "low": (0, 255, 0),
        "mid": (0, 165, 255),
        "high": (0, 0, 255)
    }

    # for level, polygon in roi_dict.items():
    #     cv2.polylines(frame, [polygon], isClosed=True, color=color_map[level], thickness=2)

    for (x1, y1, x2, y2, track_id, risk_score, risk_level) in risky_objects:
        if risk_level == "high":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"RISK ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
