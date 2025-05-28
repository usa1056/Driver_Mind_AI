# Driver Mind AI — 智慧盲點辨識與動態警示系統

本專案致力於建立一個基於 YOLOv8 的車車盲點辨識系統，結合 BDD100K 行車影像資料集進行訓練，最終提供視覺/語音警示以提升駕駛安全。

---

## 專案資料結構

```
Driver_Mind_AI/
├── BDD10K/                    # 10K 原始資料集（請手動下載放置）
│   ├── train/img + ann
│   ├── val/img + ann
│   └── test/img + ann
├── BDD10K_YOLO/               # 轉換後的 YOLOv8 資料（10K）
│   ├── images/train,val,test
│   ├── labels/train,val,test
│   └── bdd10k.yaml
├── BDD100K/                   # 100K 原始資料集（檔案過大/改以10K訓練）
│   ├── train/img + ann
│   ├── val/img + ann
│   └── test/img + ann
├── BDD100K_YOLO/              # 轉換後的 YOLOv8 資料（100K）
│   ├── images/train,val,test
│   ├── labels/train,val,test
│   └── bdd100k.yaml
├── datasets/                  # 資料轉換腳本
│   ├── convert_bdd_to_yolo.py
│   └── convert_bdd_to_yolo_2.py
├── docs/                      # 說明文檔與圖示
│   └── images/
│       └── architecture_v1.png
├── models/                    # 模型儲存區
├── notebooks/                 # EDA 與實驗記錄
├── scripts/                   # 模型訓練、推論主程式
├── .gitignore
├── .gitattributes
├── LICENSE
├── README.md
└── Requirements.txt
```
---

## 專案架構圖

下圖展示本專案的資料夾結構與主要組件流程，包含數據處理模組、YOLOv8 訓練配置與輸出對應。

![專案架構圖](docs/images/architecture_v1.png)

---

## 安裝方式

建議使用 Conda 建立處理環境：

```bash
conda create -n drivermind_ai python=3.12
conda activate drivermind_ai
pip install -r requirements.txt
```

---

## 資料準備

請自行下載 BDD10K dataset（DataSet Ninja 版本），並放置於 Driver_Mind_AI/BDD10K 資料夾下。
下載連結：https://datasetninja.com/bdd100k-10k

```
BDD10K/
├── train/
│   ├── img/
│   └── ann/
├── val/
│   ├── img/
│   └── ann/
└── test/
    ├── img/
    └── ann/
```

---

## 標準轉換 (JSON → YOLO)

執行：

```bash
python datasets/convert_bdd_to_yolo_2.py
```

輸出檔案會存到`BDD10K_YOLO/labels/train/`  
`BDD10K_YOLO/labels/val/`  
`BDD10K_YOLO/labels/test/`。

---

## 模型訓練

使用 YOLOv8 CLI 指令進行訓練：

```bash
yolo task=detect mode=train model=yolov8n.pt data=BDD100K_YOLO/bdd10k.yaml epochs=50 imgsz=640
```

---

## 語音與視覺警示模組（開發中）

* 使用 `pyttsx3` 語音輸出警示語句
* 使用 `OpenCV` 顯示動態視覺 UI

---

## 類別定義（共 6 類）

| 類別名稱            | 類別編號 |
| --------------- | ---- |
| car             | 0    |
| person          | 1    |
| truck ( 含 bus ) | 2    |
| motor           | 3    |
| bike            | 4    |
| rider           | 5    |

---

## 作者名單（依姓氏筆畫排序）
朱瑋傑、李旻翰、倪曼菱、張家齊、劉元新、蘇柏軒


---

## 授權 License

本專案採用 [MIT License](LICENSE) 授權。
