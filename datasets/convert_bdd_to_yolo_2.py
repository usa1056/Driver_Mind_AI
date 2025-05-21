# ============================================
# 🚗 專案用途：將 BDD10K 標註 JSON 檔轉為 YOLO txt
# ============================================

# 📁 資料夾結構：
# Driver_Mind_AI/
# ├── BDD10K/                      ← 原始 JSON 標註與圖片
# │   ├── train/
# │   │   ├── img/
# │   │   └── ann/
# │   ├── val/
# │   │   ├── img/
# │   │   └── ann/
# │   └── test/
# │       ├── img/
# │       └── ann/
# └── BDD10K_YOLO/
#     └── labels/                 ← 轉換後 YOLO txt 輸出
#         ├── train/
#         ├── val/
#         └── test/

import os
import json
from glob import glob
from PIL import Image

# ========================
# 類別對應表（label ➝ class id）
# ========================
label2id = {
    'car': 0,        # 小型車（轎車）
    'person': 1,     # 行人
    'truck': 2,      # 卡車（含 bus、拖車）
    'motor': 3,      # 機車
    'bike': 4,       # 腳踏車
    'rider': 5       # 騎士
}

# ========================
# 處理三個資料集 split（train / val / test）
# ========================
splits = ['train', 'val', 'test']

# 設定原始輸入與輸出資料夾的根路徑
base_input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BDD10K'))
base_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BDD10K_YOLO/labels'))

for split in splits:
    print(f"\n📦 開始處理資料集：{split}")

    image_dir = os.path.join(base_input_dir, split, "img")        # 圖片所在路徑
    ann_dir = os.path.join(base_input_dir, split, "ann")          # 標註 JSON 路徑
    output_txt_dir = os.path.join(base_output_dir, split)         # YOLO txt 輸出路徑
    os.makedirs(output_txt_dir, exist_ok=True)                    # 若資料夾不存在則建立

    json_files = sorted(glob(os.path.join(ann_dir, "*.json")))    # 找出所有標註檔

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 從 JSON 檔名推回對應圖片名稱
        image_name = os.path.basename(json_file).replace(".json", "")
        image_path = os.path.join(image_dir, image_name)

        # 讀取圖片尺寸
        try:
            width, height = Image.open(image_path).size
        except Exception as e:
            print(f"開啟圖片失敗：{image_path}，錯誤：{e}")
            continue

        yolo_labels = []  # 存放該圖片所有物件的 YOLO 格式標註

        for obj in data.get("objects", []):
            label = obj.get("classTitle")

            # 類別名稱清理與合併
            if label == "pedestrian":
                label = "person"
            elif label in ["bus", "trailer", "caravan"]:
                label = "truck"
            elif label == "motorcycle":
                label = "motor"
            elif label == "bicycle":
                label = "bike"
            elif label in ["train", "traffic light"]:
                continue  # 排除不納入訓練的類別

            # 若不在可處理類別中則跳過
            if label not in label2id:
                continue
            if obj.get("geometryType") != "rectangle":
                continue
            if "points" not in obj or "exterior" not in obj["points"]:
                continue
            points = obj["points"]["exterior"]
            if len(points) != 2:
                continue

            # 🧮 計算 YOLO 格式的 (中心點 x,y / 寬高) (都需除以圖片尺寸 ➝ 正規化)
            (x1, y1), (x2, y2) = points
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_w = abs(x2 - x1) / width
            box_h = abs(y2 - y1) / height
            class_id = label2id[label]

            # ➕ 加入 YOLO 格式標註（每列一個物件）
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        # 輸出 txt 檔（每張圖一個）
        output_path = os.path.join(output_txt_dir, image_name.replace(".jpg", ".txt"))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_labels))

    print(f"完成轉換：{len(json_files)} 筆 JSON → YOLO txt")