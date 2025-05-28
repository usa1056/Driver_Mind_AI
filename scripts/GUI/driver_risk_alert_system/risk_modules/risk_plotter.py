import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
from PIL import Image

def draw_risk_curve(annotated_frame, track_id, scores, fps=30, window_size=160):
    """
    畫出即時風險折線圖（記憶體版本，避免磁碟 I/O）

    - annotated_frame: 畫完 YOLO 框與軌跡線的影像
    - track_id: 高風險物件的 ID
    - scores: 該物件的風險分數序列
    - fps: 幀率（用於計算停留時間）
    - window_size: 折線圖顯示的分數長度上限（滑動窗口）
    """
    if len(scores) < 5:
        return annotated_frame

    recent_scores = scores[-window_size:]

    # 用 BytesIO 儲存折線圖，不寫入硬碟
    buf = io.BytesIO()
    plt.figure(figsize=(2, 1.5))
    plt.plot(recent_scores, color='red')
    plt.title(f"RISK ID: {track_id}", fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # PIL → NumPy → BGR 給 OpenCV
    img_pil = Image.open(buf).convert("RGB")
    img_np = np.array(img_pil)
    risk_plot = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    buf.close()

    # 貼圖到左下角
    risk_plot = cv2.resize(risk_plot, (160, 120))
    h, w, _ = risk_plot.shape
    annotated_frame[-h:, :w] = risk_plot

    stay_time = len(scores) / fps
    cv2.putText(annotated_frame,
                f"ID {track_id} @ RED {stay_time:.1f}s | {int(scores[0])} -> {int(scores[-1])}",
                (10, annotated_frame.shape[0] - h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return annotated_frame
