import os
import pandas as pd

# 標註資料夾
label_dir = '../BDD10K_YOLO/labels/train'
id2label = {
    0: 'car',
    1: 'person',
    2: 'truck',
    3: 'motor',
    4: 'bike',
    5: 'rider'
}

# 存放結果的 list
records = []

# 讀取每個標註檔
for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue
    file_path = os.path.join(label_dir, file)
    with open(file_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:5])
            area = w * h
            ratio = w / h if h != 0 else 0

            records.append({
                "filename": file.replace(".txt", ".jpg"),
                "class_id": class_id,
                "class_name": id2label[class_id],
                "x_center": x_center,
                "y_center": y_center,
                "width": w,
                "height": h,
                "area": area,
                "aspect_ratio": ratio
            })

# 轉為 DataFrame
df = pd.DataFrame(records)

# 顯示前幾筆
print(df.head())

# 各類別平均 bbox 面積
print(df.groupby("class_name")["area"].mean())

# 類別數量統計
print(df["class_name"].value_counts())
print(df["class_name"].value_counts())
