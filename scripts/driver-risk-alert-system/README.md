# Driver Risk Alert System

本專案旨在建立一套即時駕駛風險預測系統，整合 YOLOv8 + ROI 分析 + 多因子風險評分 + 視覺與聲音提醒邏輯，並支援模組分離以利跨組協作與擴充。

---

## 專案組織結構

```
driver-risk-alert-system/
├── track_with_analytics.py       # 主程式：YOLO 追蹤 + ROI + 風險分析 + 視覺顯示 + 警示判斷
├── output.mp4                    # 輸出影片（建議加入 .gitignore 排除）
├── assets/
│   └── videoplayback.mp4         # 測試影片素材
│
├── risk_modules/
│   ├── __init__.py               # 模組初始化
│   ├── risk_analyzer.py          # 風險評分邏輯（滯留時間、速度、ROI 分級、跳動處理）
│   ├── Land_detection.py         # 動態 ROI 偵測與場景篩選（根據車道線與速度變化）
│   ├── risk_plotter.py           # 風險分數折線圖畫圖模組（左下角儀表板）
│   ├── warning_controller.py     # 警示觸發邏輯（分數門檻、頻率控制、只提醒一次邏輯）
│   └── risk_params.yaml          # 所有風險參數設定檔：分數權重、閾值、衰退與提醒邏輯
│
├── README.md                     # 專案說明與模組架構介紹
└── .gitignore                    # 排除 output.mp4 / __pycache__ / VSCode 設定等
```

---

## 功能特色

* YOLOv8 目標偵測與追蹤（持續追蹤每個物件 track ID）
* 動態 ROI 區域（紅 / 橙 / 綠 / 側邊切入區）隨車速縮放並畫出視覺區域
* 風險評分包含：ROI 區域等級 + 滯留時間 + 速度 + 橫向 vx + 穩定性懲罰
* 支援風險分數平滑與靜止衰退，避免紅燈長時間誤報
* 新增 warning\_controller 模組：紅區分數遞增提醒、橙區只提醒一次
* 留下提醒觸發點，聲音提示交由他人串接，確保模組分離與跨人協作
* Ctrl+C 中斷後，自動釋放資源與保存影片

---

## 執行方式

```bash
python track_with_analytics.py  # 執行後會生成 demo.mp4 分析影片
```

---

## 模組說明與用途

| 檔案名稱                       | 功能簡述                                           |
| -------------------------- | ---------------------------------------------- |
| `track_with_analytics.py`  | 主程式，整合 YOLO + ROI + 分數分析 + 視覺化 + 警示邏輯          |
| `risk_analyzer.py`         | 計算風險分數、跳動懲罰與 ROI 層級套用，為風險評分核心邏輯                |
| `Land_detection.py`        | 動態判斷車道線與場景是否可用，返回 ROI 區域與比例                    |
| `warning_controller.py`    | 負責是否提醒的決策模組：黃色區單次提醒、紅區遞增頻率，與分數門檻獨立控制           |
| `risk_plotter.py`          | 將風險歷史折線圖畫在畫面左下角，供即時分析與 debug 使用                |
| `risk_params.yaml`         | 所有分數與提醒邏輯設定檔，支援 speed、stay、vx 與 ROI 分區權重參數集中管理 |
| `assets/videoplayback.mp4` | 測試影片素材，可替換任意行車紀錄器畫面進行分析                        |

---

## 環境需求

```text
Python 3.10+
ultralytics 8.3.141
opencv-python
matplotlib
numpy
PyYAML
```

---
