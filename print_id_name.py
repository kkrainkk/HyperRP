import json


def print_category_by_id(json_path, target_id=6):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 查找目标类别
        target_category = next(
            (cat for cat in data['categories'] if cat['id'] == target_id),
            None
        )

        if target_category:
            print(f"ID {target_id} 对应的类别信息:")
            print(f"名称: {target_category['name']}")
            print(f"同义词: {target_category.get('synonyms', [])}")
        else:
            print(f"警告: 未找到 ID={target_id} 的类别")

    except FileNotFoundError:
        print(f"错误: 文件 {json_path} 不存在")
    except json.JSONDecodeError:
        print(f"错误: 文件 {json_path} 不是有效的JSON格式")
    except KeyError:
        print("错误: JSON文件结构不符合预期，缺少'categories'字段")


if __name__ == "__main__":
    json_file = "/root/autodl-tmp/VGcoco_strict91.json"
    print_category_by_id(json_file)