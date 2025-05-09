import json

# 你的novel类ID映射表（原ID → 0~16）
novel_id_mapping = {
    5: 0, 6: 1, 17: 2, 18: 3, 21: 4, 22: 5,
    28: 6, 32: 7, 36: 8, 41: 9, 47: 10, 49: 11,
    61: 12, 63: 13, 76: 14, 81: 15, 87: 16
}

# 你的novel类别名表（与class_texts_novel顺序严格一致）
novel_classes = [
    "airplane", "bus", "cat", "dog", "cow", "elephant",
    "umbrella", "tie", "snowboard", "skateboard",
    "cup", "knife", "cake", "couch", "keyboard",
    "sink", "scissors"
]

def remap_novel_annotations(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # ✅ **确保 categories 里顺序正确**
    new_categories = [
        {"id": novel_id_mapping[old_id], "name": name}
        for old_id, name in zip(novel_id_mapping.keys(), novel_classes)
    ]

    # ✅ **深拷贝 `annotations`，确保不会影响原数据**
    filtered_annotations = []
    for ann in coco_data['annotations']:
        old_id = ann['category_id']
        if old_id in novel_id_mapping:
            new_ann = ann.copy()  # 避免修改原数据
            new_ann['category_id'] = novel_id_mapping[old_id]  # 重新映射
            filtered_annotations.append(new_ann)

    # ✅ **更新数据，确保`categories`被正确替换**
    coco_data['annotations'] = filtered_annotations
    coco_data['categories'] = new_categories  # 覆盖原 categories

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)

remap_novel_annotations(
    '/root/autodl-tmp/COCO/instances_val2017_novel.json',  # 原始 novel 标注
    '/root/autodl-tmp/COCO/instances_val2017_novel_mapped.json'  # 新标注
)


