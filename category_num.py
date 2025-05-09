import json
from collections import Counter


def print_top_200_categories(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 加载JSON数据

    # 收集所有类别（包括多别名）
    all_categories = []
    for image_data in data:  # 遍历每张图片
        for obj in image_data.get("objects", []):  # 遍历每个对象
            all_categories.extend(obj.get("names", []))  # 添加所有别名

    # 统计频率并获取前200个
    category_counter = Counter(all_categories)
    top_200 = category_counter.most_common(200)

    # 打印结果
    print("Top 200 frequent categories:")
    for idx, (category, count) in enumerate(top_200, 1):
        print(f"{idx}. {category}: {count}次")


# 使用示例
print_top_200_categories("/root/autodl-tmp/objects.json")