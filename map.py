import json

# 按你列出的48类ID，重新创建映射表（ID → 0~47）
coco_id_to_base_idx = {
    1: 0, 2: 1, 3: 2, 4: 3, 7: 4, 8: 5, 9: 6,
    15: 7, 16: 8, 19: 9, 20: 10, 23: 11, 24: 12, 25: 13,
    27: 14, 31: 15, 33: 16, 34: 17, 35: 18, 38: 19,
    42: 20, 44: 21, 48: 22, 50: 23, 51: 24, 52: 25,
    53: 26, 54: 27, 55: 28, 56: 29, 57: 30, 59: 31,
    60: 32, 62: 33, 65: 34, 70: 35, 72: 36, 73: 37,
    74: 38, 75: 39, 78: 40, 79: 41, 80: 42, 82: 43,
    84: 44, 85: 45, 86: 46, 90: 47
}

# 重新构建categories列表（ID 0-47 + 类别名）
base_categories = [
    {"id": new_id, "name": name}
    for new_id, name in enumerate([
        "person", "bicycle", "car", "motorcycle", "train", "truck", "boat",
        "bench", "bird", "horse", "sheep", "bear", "zebra", "giraffe",
        "backpack", "handbag", "suitcase", "frisbee", "skis", "kite",
        "surfboard", "bottle", "fork", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "pizza", "donut",
        "chair", "bed", "toilet", "tv", "laptop", "mouse", "remote",
        "microwave", "oven", "toaster", "refrigerator", "book", "clock",
        "vase", "toothbrush"
    ])
]

def remap_and_filter_annotations(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 过滤并重新映射annotations
    filtered_annotations = []
    image_id_with_annotations = set()

    for ann in coco_data['annotations']:
        old_id = ann['category_id']
        if old_id in coco_id_to_base_idx:
            ann['category_id'] = coco_id_to_base_idx[old_id]
            filtered_annotations.append(ann)
            image_id_with_annotations.add(ann['image_id'])

    # 过滤images（只保留有annotations的图片）
    filtered_images = [
        img for img in coco_data['images']
        if img['id'] in image_id_with_annotations
    ]

    # 更新coco_data
    coco_data['annotations'] = filtered_annotations
    coco_data['images'] = filtered_images
    coco_data['categories'] = base_categories  # 直接覆盖categories

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)

# 调用函数
remap_and_filter_annotations(
    '/root/autodl-tmp/COCO/instances_train2017_base.json',
    '/root/autodl-tmp/COCO/instances_train2017_base_mapped.json'
)

