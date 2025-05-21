"""
這是一個針對BDD10K資料集進行的探勘數據分析(EDA)＆圖表視覺化
注意此檔為.py檔，需有本地資料集才能夠使用，後續會補上colab ipynb版本
"""

import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import numpy as np

# ========= 1. 設定標註資料夾路徑（YOLO 格式 .txt） =========
label_dir = '../BDD10K_YOLO/labels/train'  # ← 根據你的實際路徑修改

# ========= 2. 類別對應表（class_id ➝ 類別名稱）=========
id2label = {
    0: 'car',
    1: 'person',
    2: 'truck',
    3: 'motor',
    4: 'bike',
    5: 'rider'
}

# ========= 3. 初始化統計容器 =========
class_counter = Counter()                  # 計算每個類別出現的次數
bbox_counts = []                           # 每張圖片的標註框（bbox）數量
area_list = []                             # 所有 bbox 的面積 (比例)
aspect_ratios = []                         # 所有 bbox 的寬高比
class_area_dict = {i: [] for i in id2label.keys()}  # 各類別的 bbox 面積列表

# ========= 4. 遍歷每個標註檔案 .txt =========
for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue  # 跳過非 .txt 檔案

    file_path = os.path.join(label_dir, file)
    with open(file_path, "r") as f:
        lines = f.readlines()

    bbox_counts.append(len(lines))  # 該圖片的標註框數量

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])         # 取得類別 ID
        w, h = float(parts[3]), float(parts[4])  # bbox 的寬與高（相對比例）
        area = w * h                     # bbox 的面積（相對圖片大小）
        ratio = w / h if h != 0 else 0   # 長寬比（避免除以 0）

        class_counter[class_id] += 1     # 累加類別出現次數
        area_list.append(area)           # 記錄面積
        aspect_ratios.append(ratio)      # 記錄長寬比
        class_area_dict[class_id].append(area)  # 加入對應類別的面積列表中

# ========= 5. 建立母圖畫布（3 行 2 列，共 6 子圖） =========
fig, axs = plt.subplots(3, 2, figsize=(12, 10))  # ← 縮小畫布避免太擁擠
fig.suptitle("BDD10K Dataset EDA Summary", fontsize=16)  # 整體標題

# ========= 5-1. 類別分佈長條圖 =========
axs[0, 0].bar([id2label[i] for i in class_counter.keys()], class_counter.values(), color='skyblue')
axs[0, 0].set_title("Class Distribution (Bar)")
axs[0, 0].set_xlabel("Class")
axs[0, 0].set_ylabel("Count")

# ========= 5-2. 每張圖片的標註框數量分佈 =========
axs[0, 1].hist(bbox_counts, bins=range(0, max(bbox_counts)+1), edgecolor='black', color='lightgreen')
axs[0, 1].set_title("Annotations per Image")
axs[0, 1].set_xlabel("BBox Count")
axs[0, 1].set_ylabel("Image Count")

# ========= 5-3. BBox 面積分佈直方圖 =========
axs[1, 0].hist(area_list, bins=50, color='orange', edgecolor='black')
axs[1, 0].set_title("BBox Area Distribution")
axs[1, 0].set_xlabel("Width × Height")
axs[1, 0].set_ylabel("Count")

# ========= 5-4. BBox 長寬比分佈直方圖 =========
axs[1, 1].hist(aspect_ratios, bins=50, color='teal', edgecolor='black')
axs[1, 1].set_title("BBox Aspect Ratio Distribution")
axs[1, 1].set_xlabel("Width / Height")
axs[1, 1].set_ylabel("Frequency")

# ========= 5-5. 類別分佈 Pie Chart（只顯示佔比 ≥3%） =========

# ➤ 自訂格式化函式，排除佔比小於 3% 的項目
def autopct_format(pct):
    return f'{pct:.1f}%' if pct >= 3 else ''

labels = [id2label[i] for i in class_counter.keys()]
sizes = list(class_counter.values())
colors = sns.color_palette("pastel")

# ➤ 繪製 Pie Chart
wedges, texts, autotexts = axs[2, 0].pie(
    sizes,
    labels=None,                    # 不在圖上直接顯示類別名
    autopct=autopct_format,         # 根據比例決定是否顯示百分比
    startangle=140,
    colors=colors,
    radius=1,
    wedgeprops=dict(edgecolor='white'),
    pctdistance=1.2                 # 百分比移出圓餅圖
)

# ➤ 在圖外右側顯示圖例（label 對應 wedge）
axs[2, 0].legend(
    wedges,
    labels,
    title="Class",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)
axs[2, 0].set_title("Class Distribution (Pie Chart)")

# ========= 5-6. 類別 vs 面積 散佈圖 =========
for cid, areas in class_area_dict.items():
    axs[2, 1].scatter(
        [id2label[cid]] * len(areas), areas,
        alpha=0.5, s=6, label=id2label[cid]
    )
axs[2, 1].set_title("BBox Area by Class")
axs[2, 1].set_xlabel("Class")
axs[2, 1].set_ylabel("Area (w×h)")
axs[2, 1].tick_params(axis='x', rotation=15)

# ========= 6. 調整圖表排版與間距 =========
plt.tight_layout(rect=[0, 0.01, 1, 0.95])  # ← [left, bottom, right, top] 空間比例
plt.show()