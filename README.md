# Driver Mind AI — 智慧盲點辨識與動態警示系統

本專案致力於建立一個基於 YOLOv8 的車車盲點辨識系統，結合 BDD100K 行車影像資料集進行訓練，最終提供視覺/語音警示以提升駕駛安全。

---

## 專案資料結構

```
Driver Mind AI/
├── datasets/                # BDD100K → YOLOv8 格式轉換器
│   └── convert_bdd_to_yolo.py
├── BDD100K/                 # 原始資料集 ( 請手動下載 )
│   ├── train/img, ann
│   ├── val/img, ann
│   └── test/img, ann
├── BDD100K_YOLO/
│   ├── images/train,val,test     # 對應圖片
│   ├── labels/train,val,test     # YOLO 格式標準 (.txt)
│   └── bdd100k.yaml              # YOLOv8 訓練設定檔
├── notebooks/              # EDA、標準統計等筆記本
├── scripts/                # 訓練/評估主程式
├── models/                 # 儲存模型檔案
├── docs/                   # 專案相關圖/文檔
│   ├── images/             
├── requirements.txt        # 本專案所使用的相關套件
└── README.md               # 本說明文件
```
---

## 專案架構圖

下圖展示本專案的資料夾結構與主要組件流程，包含數據處理模組、YOLOv8 訓練配置與輸出對應。

![專案架構圖](docs/images/architecture_v1.png)

---

## 安裝方式

建議使用 Conda 建立處理環境：

```bash
conda create -n blindguard python=3.12
conda activate blindguard
pip install -r requirements.txt
```

---

## 資料準備

請自行下載 **BDD100K dataset (DataSet Ninja JSON 版本)**，放置於 `BlindGuard/BDD100K` 資料夾下，結構如下：  
Link: https://datasetninja.com/bdd100k

```
BDD100K/
├── train/img + ann
├── val/img + ann
└── test/img + ann
```

---

## 標準轉換 (JSON → YOLO)

執行：

```bash
python datasets/convert_bdd_to_yolo.py
```

輸出檔案會存到 `BDD100K_YOLO/labels/`。

---

## 模型訓練

使用 YOLOv8 CLI 指令進行訓練：

```bash
yolo task=detect mode=train model=yolov8n.pt data=BDD100K_YOLO/bdd100k.yaml epochs=50 imgsz=640
```

---

## 語音與視覺警示模組（開發中）

* 使用 `pyttsx3` 語音輸出警示語句
* 使用 `OpenCV` 顯示動態視覺 UI

---

## 類別定義（共 9 類）

| 類別名稱            | 類別編號 |
| --------------- | ---- |
| car             | 0    |
| person          | 1    |
| truck ( 含 bus ) | 2    |
| motor           | 3    |
| bike            | 4    |
| rider           | 5    |
| traffic light   | 6    |

---

## 作者名單（依姓氏筆畫排序）
朱瑋傑、李旻翰、倪曼菱、張家齊、劉元新、蘇柏軒


---

## 授權 License

本專案採用 [MIT License](LICENSE) 授權。
