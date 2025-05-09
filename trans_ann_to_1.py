import json


def print_json_structure(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 打印结构的主要内容
    print("Keys in the JSON file:", data.keys())

    # 如果包含 'images'，打印前几个图像的信息
    if 'images' in data:
        print("\nFirst 2 image entries:")
        for img in data['images'][:2]:  # 打印前两个图像信息
            print(img)

    # 如果包含 'annotations'，打印前几个注释的信息
    if 'annotations' in data:
        print("\nFirst 2 annotation entries:")
        for ann in data['annotations'][:2]:  # 打印前两个注释信息
            print(ann)

    # 如果包含 'categories'，打印类别信息
    if 'categories' in data:
        print("\nCategories:")
        for category in data['categories']:
            print(category)


# 使用示例，指定注释文件路径
json_file = "/root/autodl-tmp/LVIS/lvis_v1_val.json"
print_json_structure(json_file)

