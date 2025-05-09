import json
import os


def extract_single_image_annotations(input_json, output_json, target_filename):
    """支持模糊匹配文件名（如忽略大小写、子目录前缀）"""
    with open(input_json, 'r') as f:
        data = json.load(f)

    # 模糊匹配逻辑（忽略路径前缀和大小写）
    target_image = None
    for img in data['images']:
        # 提取文件名（去除子目录前缀）
        basename = os.path.basename(img['file_name']).lower()
        if basename == target_filename.lower():
            target_image = img
            break

    if not target_image:
        all_filenames = [os.path.basename(img['file_name']) for img in data['images']]
        raise ValueError(f"未找到文件: {target_filename}。可选文件名示例: {all_filenames[:5]}")

    # 提取标注（同上）
    target_image_id = target_image['id']
    target_annotations = [ann for ann in data['annotations'] if ann['image_id'] == target_image_id]

    new_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": [target_image],
        "annotations": target_annotations,
        "categories": data['categories']
    }

    with open(output_json, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"提取完成！输出文件: {os.path.abspath(output_json)}")


if __name__ == "__main__":
    input_file = "/root/autodl-tmp/VGcoco_strict91.json"
    output_file = "./277_annotation.json"
    target_image_name = "15.jpg"  # 即使输入为 "VG_100K/277.jpg" 也能匹配

    extract_single_image_annotations(input_file, output_file, target_image_name)