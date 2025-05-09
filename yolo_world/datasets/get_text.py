import json
from collections import defaultdict


def get_top_classes(ann_file, max_classes=200):
    """统计并返回高频类别"""
    # 同义词映射规则（与数据集类保持一致）
    synonym_rules = {
        'person': ['man', 'woman', 'boy', 'girl', 'human'],
        'car': ['auto', 'automobile', 'sedan', 'truck', 'van'],
        'dog': ['puppy', 'doggy'],
        'tree': ['palm', 'oak', 'pine'],
        'building': ['house', 'skyscraper']
    }
    reverse_synonym = {alias: main for main, aliases in synonym_rules.items() for alias in aliases}

    category_counter = defaultdict(int)

    # 读取标注文件
    with open(ann_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
        for ann in annotations:
            for obj in ann.get('objects', []):
                if names := obj.get('names'):
                    raw_name = names[0].strip().lower()
                    mapped_name = reverse_synonym.get(raw_name, raw_name)
                    category_counter[mapped_name] += 1

    # 按频率排序
    sorted_categories = sorted(
        category_counter.items(),
        key=lambda x: x[1],
        reverse=True
    )[:max_classes]

    return [k for k, v in sorted_categories]


def save_class_list(class_list, output_file='/root/autodl-tmp/class_names——new.txt'):
    """保存类别列表到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, class_name in enumerate(class_list, 1):
            f.write(f"{idx}\t{class_name}\n")
    print(f"成功保存{len(class_list)}个类别到 {output_file}")


if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    ann_file = "/root/autodl-tmp//objects.json"  # 替换为实际标注文件路径

    # 获取并保存类别
    top_classes = get_top_classes(ann_file)
    save_class_list(top_classes)