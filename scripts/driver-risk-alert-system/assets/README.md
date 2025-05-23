# Driver Risk Alert System 🚗⚠️

本專案整合 YOLOv8 + Object Tracking + ROI 區域分析 + 風險評分邏輯，實現即時駕駛風險預測與視覺化提示效果。

---

## 專案組織結構

```
driver-risk-alert-system/
├── track_with_analytics.py       # 主程式：YOLO 追蹤 + ROI + 風險評估 + 視覺化曲線圖
├── output.mp4                    # 輸出影片（建議加入 .gitignore 排除）
├── assets/
│   └── videoplayback.mp4         # 測試影片素材
│
├── risk_modules/
│   ├── __init__.py               # 模組初始化
│   ├── risk_analyzer.py          # 風險計算邏輯（中心點、ROI、速度、風險分數）
│   ├── Land_detection.py         # 動態 ROI 檢測與場景過濾
│   ├── risk_plotter.py           # 即時風險曲線圖（Matplotlib + OpenCV 疊圖）
│   └── risk_params.yaml          # YAML 設定檔：分數參數、ROI 高度、速度設定等
│
├── README.md                     # 專案說明與模組架構圖（建議加上 Mermaid 流程圖）
└── .gitignore                    # 排除 output.mp4 / __pycache__ / VSCode 設定等

```

---

## 功能特色

* 支援物件偵測與追蹤（YOLOv8 + Track）
* 自動偵測車道帶並標示 ROI 區域（紅 / 橙 / 綠）
* 計算每個 track ID 風險分數（位置、速度、穩定性）
* 左下角即時顯示風險折線圖（儀表板式提示）
* 中斷 Ctrl+C 後自動釋放資源並保存影片

---

## 如何執行

```bash
python track_with_analytics.py
```

---

## 檔案與模組說明

| 檔案名稱                       | 功能                                                            |
| -------------------------- | ------------------------------------------------------------- |
| `track_with_analytics.py`  | 主程式，整合 YOLO + 追蹤 + ROI + 即時視覺化                                |
| `risk_analyzer.py`         | 含 `get_center()`、`analyze_risk()`、`draw_risk_overlay()` 等關鍵函式 |
| `Land_detection.py`        | 定義 `process_frame()` ，用於返回 ROI 與場景有效性                         |
| `risk_params.yaml`         | 自訂風險分數加減比重、ROI 區域設定                                           |
| `assets/videoplayback.mp4` | 測試影片，可自行更換                                                    |


---

## 環境需求

```text
Python 3.10+
ultralytics
opencv-python
matplotlib
numpy
PyYAML
```