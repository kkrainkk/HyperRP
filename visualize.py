import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import numpy as np
import cv2
import os

# 配置路径
base_image_dir = "/root/autodl-tmp/VG_100K"  # 图像根目录（VG_100K和VG_100K_2的父目录）
coco_annotations = "/root/autodl-tmp/VGcoco_strict91.json"  # COCO标注文件路径
target_image_index = 20 # 要可视化的第51张图像（索引从0开始）

# 初始化COCO API
coco = COCO(coco_annotations)

# 获取图像信息
if target_image_index >= len(coco.dataset['images']):
    print(f"错误：标注文件中共有 {len(coco.dataset['images'])} 张图像，无法访问第{target_image_index + 1}张！")
else:
    img_id = coco.dataset['images'][target_image_index]['id']
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # 读取图像：直接从标注的file_name中获取相对路径
    img_relative_path = img_info['file_name']  # 假设标注中的file_name是类似 "VG_100K/12.jpg"
    img_abs_path = os.path.join(base_image_dir, img_relative_path)

    if not os.path.exists(img_abs_path):
        raise FileNotFoundError(
            f"图像文件 {img_relative_path} 未找到！检查路径：\n{img_abs_path}"
        )

    img = cv2.imread(img_abs_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建可视化画布
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # 绘制标注
    if len(annotations) == 0:
        plt.title(f"No Annotations for Image ID: {img_id}")
    else:
        for ann in annotations:
            # 获取类别名称和颜色
            category = coco.loadCats(ann['category_id'])[0]['name']
            color = np.random.rand(3)  # 随机颜色

            # 绘制边界框
            bbox = ann['bbox']
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
            plt.gca().add_patch(rect)

            # 绘制类别标签
            label = f"{category}"
            plt.text(x, y - 5, label, color=color, fontsize=10, backgroundcolor="white")

            # 绘制分割多边形（如果存在）
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((-1, 2))
                    plt.plot(poly[:, 0], poly[:, 1], color=color, linewidth=2, linestyle="--")

        plt.title(f"Image ID: {img_id} | Annotations: {len(annotations)}")

    # 隐藏坐标轴并显示
    plt.axis('off')
    plt.show()

