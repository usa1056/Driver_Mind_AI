import cv2
import numpy as np
import sys

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
    根據左右車道線位置，從底部往上依固定高度分為紅、橘、綠三段警戒區
    並動態畫出多邊形區塊，過近時顯示警告
    """
    if left_line is None or right_line is None:
        return frame

    height, width = frame.shape[:2]
    y_bottom = height  # 頂部為畫面底部

    # 設定三段區域的高度（像素）
    red_height = int(200)
    #red_height = 100
    orange_height = 70
    green_height = 200

    # 計算區段的 y 範圍
    red_y_top = y_bottom - red_height
    orange_y_top = red_y_top - orange_height
    
    # 取得左右線最高點（最小 y 值）
    left_y_min = min(left_line[1], left_line[3])
    right_y_min = min(right_line[1], right_line[3])
    max_top_y = max(left_y_min, right_y_min)
    
    # 限制 green_y_top 不得高於最大可插值範圍
    green_y_top = max(orange_y_top - green_height, max_top_y)


    # 線性插值函數
    def interp_x(line, y):
        x1, y1, x2, y2 = line
        if y2 == y1:
            return x1
        return int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))

    # 取得各段左右車道線上的 x 座標
    # 紅區
    left_x_red_bot = interp_x(left_line, y_bottom)
    left_x_red_top = interp_x(left_line, red_y_top)
    right_x_red_bot = interp_x(right_line, y_bottom)
    right_x_red_top = interp_x(right_line, red_y_top)

    # 橘區
    left_x_orange_top = interp_x(left_line, orange_y_top)
    right_x_orange_top = interp_x(right_line, orange_y_top)

    # 綠區
    left_x_green_top = interp_x(left_line, green_y_top)
    right_x_green_top = interp_x(right_line, green_y_top)

    # 建立三段多邊形區域
    red_zone = np.array([
        [left_x_red_bot, y_bottom],
        [left_x_red_top, red_y_top],
        [right_x_red_top, red_y_top],
        [right_x_red_bot, y_bottom]
    ])

    orange_zone = np.array([
        [left_x_red_top, red_y_top],
        [left_x_orange_top, orange_y_top],
        [right_x_orange_top, orange_y_top],
        [right_x_red_top, red_y_top]
    ])

    green_zone = np.array([
        [left_x_orange_top, orange_y_top],
        [left_x_green_top, green_y_top],
        [right_x_green_top, green_y_top],
        [right_x_orange_top, orange_y_top]
    ])

    # 畫區域
    overlay = np.zeros_like(frame)
    cv2.fillPoly(overlay, [green_zone], (0, 255, 0))      # 綠色
    cv2.fillPoly(overlay, [orange_zone], (0, 165, 255))   # 橘色
    cv2.fillPoly(overlay, [red_zone], (0, 0, 255))        # 紅色

    # 判斷警告：紅色區底部左右距離是否過小
    red_width = abs(left_x_red_bot - right_x_red_bot)
    if red_width < 150:
        cv2.putText(frame, "WARNING: TOO CLOSE", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 合併畫面與半透明疊層
    return cv2.addWeighted(frame, 1, overlay, 0.4, 1)


def process_frame(frame):
    edges = detect_edges(frame)
    roi = region_of_interest(edges)
    lines = detect_lines(roi)
    left_params, right_params = average_slope_intercept(lines)
    left_line = make_coordinates(frame, left_params)
    right_line = make_coordinates(frame, right_params)
    frame_with_colors = draw_multicolor_lane(frame, left_line, right_line)
    return frame_with_colors

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output = process_frame(frame)
        cv2.imshow("Lane with Multi-color Zones", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "land2.mp4"  # 預設影片
    main(video_path)
