import os
import json
from glob import glob
from PIL import Image

# ========================
# 類別對應表（label ➝ class id）
# ========================
label2id = {
    'car': 0,
    'person': 1,
    'truck': 2,
    'motor': 3,
    'bike': 4,
    'rider': 5
}

# ========================
# 處理 train / val / test
# ========================
splits = ['train', 'val', 'test']
base_input_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BDD10K'))
base_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'BDD10K_YOLO/labels'))

for split in splits:
    print(f"\n處理資料集：{split}")

    image_dir = os.path.join(base_input_dir, split, "img")
    ann_dir = os.path.join(base_input_dir, split, "ann")
    output_txt_dir = os.path.join(base_output_dir, split)
    os.makedirs(output_txt_dir, exist_ok=True)

    json_files = sorted(glob(os.path.join(ann_dir, "*.json")))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_name = os.path.basename(json_file).replace(".json", "")
        image_path = os.path.join(image_dir, image_name)

        try:
            width, height = Image.open(image_path).size
        except Exception as e:
            print(f"開啟圖片失敗：{image_path}，錯誤：{e}")
            continue

        yolo_labels = []

        for obj in data.get("objects", []):
            label = obj.get("classTitle")

            # 類別名稱清洗與轉換
            if label == "pedestrian":
                label = "person"
            elif label in ["bus", "trailer", "caravan"]:
                label = "truck"
            elif label == "motorcycle":
                label = "motor"
            elif label == "bicycle":
                label = "bike"
            elif label in ["train", "traffic light"]:
                continue  # 排除不處理

            if label not in label2id:
                continue
            if obj.get("geometryType") != "rectangle":
                continue
            if "points" not in obj or "exterior" not in obj["points"]:
                continue
            points = obj["points"]["exterior"]
            if len(points) != 2:
                continue

            (x1, y1), (x2, y2) = points
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_w = abs(x2 - x1) / width
            box_h = abs(y2 - y1) / height
            class_id = label2id[label]

            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        output_path = os.path.join(output_txt_dir, image_name.replace(".jpg", ".txt"))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_labels))

    print(f"✅ 完成轉換：{len(json_files)} 筆 JSON → YOLO")