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

"""
下方程式碼區塊目的為產生bbox中心點熱力圖
"""

# ========= 初始化統計容器（新增中心點清單 center_points） =========
class_counter = Counter()
bbox_counts = []
area_list = []
aspect_ratios = []
center_points = []  # ⬅ 新增：記錄 bbox 中心點 (x, y)
class_area_dict = {i: [] for i in id2label.keys()}

# ========= 遍歷每個標註檔案 .txt =========
for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue

    file_path = os.path.join(label_dir, file)
    with open(file_path, "r") as f:
        lines = f.readlines()

    bbox_counts.append(len(lines))

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center = float(parts[1]), float(parts[2])  # ⬅ 中心點座標
        w, h = float(parts[3]), float(parts[4])
        area = w * h
        ratio = w / h if h != 0 else 0

        class_counter[class_id] += 1
        area_list.append(area)
        aspect_ratios.append(ratio)
        class_area_dict[class_id].append(area)
        center_points.append((x_center, y_center))  # ⬅ 儲存中心點

# ========= 額外視覺化：bbox 中心點 Heatmap =========
import pandas as pd

# 轉為 DataFrame，方便 seaborn 處理
df_center = pd.DataFrame(center_points, columns=['x_center', 'y_center'])

# 使用 Seaborn KDE 繪製熱力圖（2D 分佈）
plt.figure(figsize=(6, 5))
sns.kdeplot(
    x=df_center['x_center'],
    y=df_center['y_center'],
    cmap='Reds',
    fill=True,
    thresh=0.05,
    bw_adjust=0.5
)
plt.title("BBox Center Position Heatmap")
plt.xlabel("x_center (normalized)")
plt.ylabel("y_center (normalized)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().invert_yaxis()  # 讓畫面符合圖片視角（左上為原點）
plt.tight_layout()
plt.show()