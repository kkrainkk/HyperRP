import json
import os


def extract_first_image_annotations(coco_anno_path, output_path):
    """
    提取COCO标注文件中第一张图像的完整标注信息
    :param coco_anno_path: COCO标注文件路径
    :param output_path: 输出文件路径
    """
    # 加载原始标注文件
    with open(coco_anno_path, 'r') as f:
        coco_data = json.load(f)

    # 获取第一张图像信息
    if len(coco_data['images']) == 0:
        raise ValueError("标注文件中没有图像信息")

    first_image = coco_data['images'][0]
    image_id = first_image['id']

    # 提取相关标注
    target_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] == image_id
    ]

    # 构建新数据（保留完整COCO格式）
    new_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": [first_image],
        "annotations": target_annotations,
        "categories": coco_data['categories']  # 保留所有类别定义
    }

    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)

    # 打印验证信息
    print(f"提取完成！输出文件: {os.path.abspath(output_path)}")
    print(f"图像文件名: {first_image['file_name']}")
    print(f"图像尺寸: {first_image['width']}x{first_image['height']}")
    print(f"关联标注数量: {len(target_annotations)}")


if __name__ == "__main__":
    # 配置路径（根据实际路径修改）
    coco_anno_path = "/root/autodl-tmp/COCO/annotations/instances_train2017.json"
    output_path = "/root/autodl-tmp/first_image_annotations_withseg.json"

    # 执行提取
    extract_first_image_annotations(coco_anno_path, output_path)