import json


def inspect_coco_annotations(json_path, num_images=10):
    with open(json_path) as f:
        coco_data = json.load(f)

    # 构建类别ID到名称的映射
    id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # 获取前N个图像ID
    image_ids = [img["id"] for img in coco_data["images"][:num_images]]

    print(f"检查前 {num_images} 个图像的标注信息：")
    print("=" * 60)

    for img in coco_data["images"][:num_images]:
        # 获取该图像所有标注
        annotations = [ann for ann in coco_data["annotations"]
                       if ann["image_id"] == img["id"]]

        print(f"\n图像ID: {img['id']}")
        print(f"文件名: {img['file_name']}")
        print(f"尺寸: {img['width']}x{img['height']}")
        print(f"标注数量: {len(annotations)}")

        if len(annotations) > 0:
            print("\n标注详情：")
            for i, ann in enumerate(annotations[:3], 1):  # 最多显示前3个标注
                print(f"{i}. 类别: {id_to_name[ann['category_id']]} (ID: {ann['category_id']})")
                print(f"   BBOX: [x:{ann['bbox'][0]}, y:{ann['bbox'][1]}, w:{ann['bbox'][2]}, h:{ann['bbox'][3]}]")
                print(f"   面积: {ann['area']}")
            if len(annotations) > 3:
                print(f"...剩余 {len(annotations) - 3} 个标注未显示")
        else:
            print("⚠️ 该图像无有效标注")
        print("-" * 60)


if __name__ == "__main__":
    json_path = "/root/autodl-tmp/COCO/annotations/instances_val2017.json"
    inspect_coco_annotations(json_path, num_images=5)