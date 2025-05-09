import json
from collections import defaultdict

def check_coco_annotation_classes(ann_file):
    # 读取COCO格式标注文件
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 初始化统计容器
    category_counter = defaultdict(int)

    # 遍历annotations统计每个类别ID的出现次数
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        category_counter[cat_id] += 1

    # 如果文件里有categories字段，可以一起打印类别名称
    cat_id_to_name = {}
    if 'categories' in coco_data:
        for cat in coco_data['categories']:
            cat_id_to_name[cat['id']] = cat['name']

    # 打印统计信息
    print(f"📊 文件: {ann_file}")
    print(f"\n类别ID\t类别名称\t标注数量")
    print("-" * 40)

    for cat_id, count in sorted(category_counter.items()):
        cat_name = cat_id_to_name.get(cat_id, "未知类别")
        print(f"{cat_id}\t{cat_name}\t{count}")

    print(f"\n共发现 {len(category_counter)} 个独特类别ID")
    print("-" * 40)

# 调用示例
check_coco_annotation_classes('/root/autodl-tmp/COCO/instances_val2017_novel_mapped.json')  # 你可以换成任何coco标注路径



