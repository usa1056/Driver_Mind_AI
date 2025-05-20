"""
轉換 BDD100K 資料集標註為 YOLOv8 格式
（適用 Dataset Ninja 版 BDD100K 格式）

說明：
    本腳本可將 Dataset Ninja 風格的 JSON 標註資料，
    轉換為 YOLO v8 所需的 .txt 格式標註檔案。
    轉換內容為 rectangle 類型的物件框，支援多類別指定。

📂 BDD100K/                   ← 原始Dataset Ninja資料集（JSON 與圖片分開）
├── 📂 train/
│   ├── 📂 img/               ← 訓練圖片（.jpg）
│   └── 📂 ann/               ← 訓練標註（.json）
├── 📂 val/
│   ├── 📂 img/
│   └── 📂 ann/
└── 📂 test/
    ├── 📂 img/
    └── 📂 ann/

📂 BDD100K_YOLO/               ← YOLOv8 訓練格式
├── 📂 images/                 ← 圖片（從原始資料集中複製而來）
│   ├── 📂 train/
│   ├── 📂 val/
│   └── 📂 test/
├── 📂 labels/                 ← 對應圖片的 .txt 標註檔（YOLO 格式）
│   ├── 📂 train/
│   ├── 📂 val/
│   └── 📂 test/
└── 📄 bdd100k.yaml            ← YOLOv8 訓練設定檔（含 path/class 數量）

每張圖片會產生一個對應的 .txt 標註檔，內容為：
    <class_id> <x_center> <y_center> <width> <height>

作者：Eric Chang
更新日期：
2025-05-20 (GMT+8) 9:09 (建立)
2025-05-20 (GMT+8) 11:16 (更改label2id條件)
2025-05-20 (GMT+8) 15:44 (和老師討論後，移除lane/drivable area)
"""
import os
import json
from glob import glob
from PIL import Image  # 用來讀取圖片尺寸

# =====================================
# 類別對應表：標籤名稱 ➝ 類別 ID
# - bus 被合併到 truck 類別
# - 其他未在此表內的類別會被忽略
# =====================================
label2id = {
    'car': 0,            # 車子（轎車）
    'person': 1,         # 行人
    'truck': 2,          # 卡車（含 bus）
    'motor': 3,          # 機車（不含騎士）
    'bike': 4,           # 腳踏車
    'rider': 5,          # 騎士（騎機車/腳踏車的人）
    'traffic light': 6   # 紅綠燈 / 交通燈
}

# =====================================
# 設定要處理的資料集 split（訓練/驗證/測試）
# 假設資料夾結構為：
# BlindGuard/
# ├── BDD100K/train/img + ann/
# └── BDD100K_YOLO/labels/
# =====================================
splits = ['train', 'val', 'test']

# 設定輸入輸出資料夾的根路徑
base_input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BDD100K'))
base_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BDD100K_YOLO/labels'))

# 逐一處理 train / val / test
for split in splits:
    print(f"\n開始處理資料集：{split}")

    image_dir = os.path.join(base_input_dir, split, "img")  # 影像資料夾
    ann_dir = os.path.join(base_input_dir, split, "ann")    # 標註 JSON 資料夾
    output_txt_dir = os.path.join(base_output_dir, split)   # 輸出的 YOLO txt 資料夾
    os.makedirs(output_txt_dir, exist_ok=True)              # 若不存在則建立

    # 找出所有 JSON 標註檔
    json_files = sorted(glob(os.path.join(ann_dir, "*.json")))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Dataset Ninja 的圖片名稱是從 JSON 檔名推得（圖片與 JSON 同名）
        image_name = os.path.basename(json_file).replace('.json', '')
        image_path = os.path.join(image_dir, image_name)

        try:
            width, height = Image.open(image_path).size  # 取得圖片尺寸（轉比例用）
        except Exception as e:
            print(f"[錯誤] 開啟圖片失敗：{image_path}，原因：{e}")
            continue

        yolo_labels = []  # 存放轉換後的每一行標註

        # 處理每一個物件標註
        for obj in data.get("objects", []):
            label = obj.get("classTitle")  # 類別名稱

            if label == 'bus':
                label = 'truck'  # bus 併入 truck 類別
            if label not in label2id:
                continue  # 如果該類別不在我們關注清單中就略過
            if obj.get("geometryType") != "rectangle":
                continue  # 只處理矩形框（忽略線段/多邊形）
            if "points" not in obj or "exterior" not in obj["points"]:
                continue  # 無效標註略過

            points = obj["points"]["exterior"]
            if len(points) != 2:
                continue  # rectangle 應該有左上與右下兩點，否則跳過

            (x1, y1), (x2, y2) = points  # 取出 bounding box 左上與右下座標

            # 計算中心座標與寬高（轉為相對比例）
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_w = abs(x2 - x1) / width
            box_h = abs(y2 - y1) / height

            class_id = label2id[label]  # 查對應 class ID
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        # 輸出 YOLO 格式標註 txt 檔（與圖片同名 .txt）
        output_path = os.path.join(output_txt_dir, image_name.replace(".jpg", ".txt"))
        with open(output_path, "w", encoding='utf-8') as f:
            f.write("\n".join(yolo_labels))

    print(f"完成轉換：{len(json_files)} 筆 JSON → YOLO txt")



