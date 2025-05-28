import cv2
import numpy as np
import sys
from collections import deque

# 儲存左右線歷史資料（最多 5 幀）
left_line_history = deque(maxlen=5)
right_line_history = deque(maxlen=5)

def smooth_line(history, new_line):
    """
    平滑化車道線（用最近幾幀平均）
    """
    if new_line is not None:
        history.append(new_line)
    if not history:
        return None
    avg_line = np.mean(history, axis=0).astype(int)
    return avg_line

def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),
        (int(width * 0.4), int(height * 0.55)),
        (int(width * 0.6), int(height * 0.55)),
        (width, height)
    ]])
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def detect_lines(edges):
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=80, maxLineGap=60)
    if raw_lines is None:
        return []
    filtered = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if 0.3 < abs(slope) < 5:
            filtered.append(line)
    return filtered

def average_slope_intercept(lines):
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    left_avg = np.average(left_lines, axis=0) if left_lines else None
    right_avg = np.average(right_lines, axis=0) if right_lines else None
    return left_avg, right_avg

def make_coordinates(frame, line_params):
    height, width, _ = frame.shape
    if line_params is None:
        return None
    slope, intercept = line_params
    y1 = height
    y2 = int(height * 0.6)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def draw_multicolor_lane(frame, left_line, right_line):
    """
    畫出紅橙綠三段風險區域，並保證回傳合法影像
    """
    if frame is None:
        print("[❌ draw_multicolor_lane] 警告：輸入 frame 為 None")
        return np.zeros((720, 1280, 3), dtype=np.uint8)  # 根據預設解析度調整

    frame_copy = frame.copy()

    if left_line is None or right_line is None:
        print("[⚠️ draw_multicolor_lane] 警告：缺少車道線，僅回傳原圖")
        return frame_copy

    try:
        lane_width = abs(left_line[0] - right_line[0])
        height, width = frame.shape[:2]
        y_bottom = height

        red_height = int(200)
        orange_height = 70
        green_height = 200

        red_y_top = y_bottom - red_height
        orange_y_top = red_y_top - orange_height
        left_y_min = min(left_line[1], left_line[3])
        right_y_min = min(right_line[1], right_line[3])
        max_top_y = max(left_y_min, right_y_min)
        green_y_top = max(orange_y_top - green_height, max_top_y)

        def interp_x(line, y):
            x1, y1, x2, y2 = line
            if y2 == y1:
                return x1
            return int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))

        left_x_red_bot = interp_x(left_line, y_bottom)
        left_x_red_top = interp_x(left_line, red_y_top)
        right_x_red_bot = interp_x(right_line, y_bottom)
        right_x_red_top = interp_x(right_line, red_y_top)
        left_x_orange_top = interp_x(left_line, orange_y_top)
        right_x_orange_top = interp_x(right_line, orange_y_top)
        left_x_green_top = interp_x(left_line, green_y_top)
        right_x_green_top = interp_x(right_line, green_y_top)

        red_zone = np.array([[left_x_red_bot, y_bottom], [left_x_red_top, red_y_top],
                             [right_x_red_top, red_y_top], [right_x_red_bot, y_bottom]])
        orange_zone = np.array([[left_x_red_top, red_y_top], [left_x_orange_top, orange_y_top],
                                [right_x_orange_top, orange_y_top], [right_x_red_top, red_y_top]])
        green_zone = np.array([[left_x_orange_top, orange_y_top], [left_x_green_top, green_y_top],
                               [right_x_green_top, green_y_top], [right_x_orange_top, orange_y_top]])

        overlay = np.zeros_like(frame_copy)
        cv2.fillPoly(overlay, [green_zone], (0, 255, 0))
        cv2.fillPoly(overlay, [orange_zone], (0, 165, 255))
        cv2.fillPoly(overlay, [red_zone], (0, 0, 255))

        red_width = abs(left_x_red_bot - right_x_red_bot)
        if red_width < lane_width * 0.4:
            cv2.putText(frame_copy, "WARNING: TOO CLOSE", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return cv2.addWeighted(frame_copy, 1, overlay, 0.4, 1)

    except Exception as e:
        print("[❌ draw_multicolor_lane 畫圖失敗]", e)
        return frame_copy


def get_lane_roi_dynamic(left_line, right_line, frame_shape, speed=0, scale_factor=1.0):
    """
    使用左線、右線建立動態 ROI（高、中、低 + 左右側切入區），允許根據車速動態調整長度
    """
    if left_line is None or right_line is None:
        return {}

    height = frame_shape[0]
    width = frame_shape[1]
    y_bottom = height

    # 車道寬度與縮放
    left_x_bot = left_line[0]
    right_x_bot = right_line[0]
    lane_width = abs(right_x_bot - left_x_bot)

    base_scale = min(max(lane_width / 400, 0.5), 0.6)
    dynamic_scale = min(base_scale + speed * 0.005 * scale_factor, 1.0)
    scale = dynamic_scale

    red_height = int(150 * scale)
    orange_height = int(70 * scale)
    green_height = int(180 * scale)

    # 精準銜接：每一區 top 為上一區 top 減高
    red_y_top = y_bottom - red_height
    orange_y_bottom = red_y_top
    orange_y_top = orange_y_bottom - orange_height
    green_y_bottom = orange_y_top
    green_y_top = green_y_bottom - green_height

    # 預防太高超出畫面
    left_y_min = min(left_line[1], left_line[3])
    right_y_min = min(right_line[1], right_line[3])
    max_top_y = max(left_y_min, right_y_min)
    green_y_top = max(green_y_top, max_top_y)

    def interp_x(line, y):
        x1, y1, x2, y2 = line
        if y2 == y1:
            return (x1 + x2) // 2
        return int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))

    # 左右邊界點
    left_x_red_bot = interp_x(left_line, y_bottom)
    left_x_red_top = interp_x(left_line, red_y_top)
    left_x_orange_top = interp_x(left_line, orange_y_top)
    left_x_green_top = interp_x(left_line, green_y_top)

    right_x_red_bot = interp_x(right_line, y_bottom)
    right_x_red_top = interp_x(right_line, red_y_top)
    right_x_orange_top = interp_x(right_line, orange_y_top)
    right_x_green_top = interp_x(right_line, green_y_top)

    # 左右切入區延伸寬度（以車道寬延伸 0.9）
    side_offset = int(lane_width * 0.9)
    side_y_top = max(red_y_top - 100, 0)  # 上緣再往上拉長 80px
    right_x_side_bot = min(right_x_red_bot + side_offset, width - 1)
    right_x_side_top = min(right_x_red_top + side_offset, width - 1)
    left_x_side_bot = max(left_x_red_bot - side_offset, 0)
    left_x_side_top = max(left_x_red_top - side_offset, 0)

    roi_dict = {
        "high": np.array([
            [left_x_red_bot, y_bottom],
            [left_x_red_top, red_y_top],
            [right_x_red_top, red_y_top],
            [right_x_red_bot, y_bottom]
        ]),
        "mid": np.array([
            [left_x_red_top, red_y_top],
            [left_x_orange_top, orange_y_top],
            [right_x_orange_top, orange_y_top],
            [right_x_red_top, red_y_top]
        ]),
        "low": np.array([
            [left_x_orange_top, orange_y_top],
            [left_x_green_top, green_y_top],
            [right_x_green_top, green_y_top],
            [right_x_orange_top, orange_y_top]
        ]),
        "side_right": np.array([
            [right_x_red_bot, y_bottom],
            [right_x_red_top, red_y_top],
            [right_x_red_top + side_offset, side_y_top],
            [right_x_red_bot + side_offset, y_bottom]
        ]),
        "side_left": np.array([
            [left_x_red_bot - side_offset, y_bottom],
            [left_x_red_top - side_offset, side_y_top],
            [left_x_red_top, red_y_top],
            [left_x_red_bot, y_bottom]
        ])
    }

    return roi_dict, scale


def is_valid_lane_scene(left_line, right_line, frame_shape):
    """
    簡單判斷目前場景是否為有效的車道環境
    條件：
    1. 左右線皆存在
    2. 左右線之間的距離 > 最小寬度
    3. 車道線長度夠長（非碎線）
    """
    if left_line is None or right_line is None:
        return False

    height = frame_shape[0]
    min_lane_width = 100  # 兩條線之間最小距離
    min_line_height = int(height * 0.2)

    # 比較底部兩點的距離
    _, y1_left, _, y2_left = left_line
    _, y1_right, _, y2_right = right_line

    lane_width = abs(left_line[0] - right_line[0])
    left_line_length = abs(y2_left - y1_left)
    right_line_length = abs(y2_right - y1_right)

    if lane_width < min_lane_width:
        return False
    if left_line_length < min_line_height or right_line_length < min_line_height:
        return False

    return True

def process_frame(frame):
    global left_line_history, right_line_history

    try:
        # 邊緣偵測 + Hough 轉換
        edges = detect_edges(frame)
        roi = region_of_interest(edges)
        lines = detect_lines(roi)
        left_params, right_params = average_slope_intercept(lines)

        # 平滑處理線條
        raw_left = make_coordinates(frame, left_params)
        raw_right = make_coordinates(frame, right_params)
        left_line = smooth_line(left_line_history, raw_left)
        right_line = smooth_line(right_line_history, raw_right)

        # 畫主車道區域與 lane 線條（紅橙綠）
        frame_with_colors = draw_multicolor_lane(frame, left_line, right_line)

        # 建立 ROI 字典（含 side_left/right）
        roi_dict, scale = get_lane_roi_dynamic(left_line, right_line, frame.shape)

        # 場景過濾：判斷車道是否有效
        scene_valid = is_valid_lane_scene(left_line, right_line, frame.shape)

        # 畫左側切入區（橘色）
        if "side_left" in roi_dict:
            overlay = frame_with_colors.copy()
            cv2.fillPoly(overlay, [roi_dict["side_left"]], (0, 140, 255))  # BGR 橘
            frame_with_colors = cv2.addWeighted(overlay, 0.4, frame_with_colors, 0.6, 0)

        # 畫右側切入區（橘色）
        if "side_right" in roi_dict:
            overlay = frame_with_colors.copy()
            cv2.fillPoly(overlay, [roi_dict["side_right"]], (0, 140, 255))  # BGR 橘
            frame_with_colors = cv2.addWeighted(overlay, 0.4, frame_with_colors, 0.6, 0)

        # 異常 fallback
        if frame_with_colors is None:
            print("[❌ process_frame] draw_multicolor_lane 回傳 None，回傳原圖")
            frame_with_colors = frame.copy()

        return frame_with_colors, roi_dict, scene_valid, left_line, right_line

    except Exception as e:
        print(f"[❌ process_frame Error] {e}")
        return frame.copy(), {}, False, None, None




if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "land2.mp4"  # 預設影片
    main(video_path)
