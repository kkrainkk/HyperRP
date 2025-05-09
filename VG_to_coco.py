import json
from pathlib import Path
from tqdm import tqdm

# 严格匹配的COCO类别定义（保持原始大小写）
COCO_CLASSES = [
    ["person"], ["bicycle"], ["car"], ["motorcycle"], ["airplane"],
    ["bus"], ["train"], ["truck"], ["boat"], ["traffic light"],
    ["fire hydrant"], ["stop sign"], ["parking meter"], ["bench"],
    ["bird"], ["cat"], ["dog"], ["horse"], ["sheep"], ["cow"],
    ["elephant"], ["bear"], ["zebra"], ["giraffe"], ["backpack"],
    ["umbrella"], ["handbag"], ["tie"], ["suitcase"], ["frisbee"],
    ["skis"], ["snowboard"], ["sports ball"], ["kite"], ["baseball bat"],
    ["baseball glove"], ["skateboard"], ["surfboard"], ["tennis racket"],
    ["bottle"], ["wine glass"], ["cup"], ["fork"], ["knife"],
    ["spoon"], ["bowl"], ["banana"], ["apple"], ["sandwich"],
    ["orange"], ["broccoli"], ["carrot"], ["hot dog"], ["pizza"],
    ["donut"], ["cake"], ["chair"], ["couch"], ["potted plant"],
    ["bed"], ["dining table"], ["toilet"], ["tv"], ["laptop"],
    ["mouse"], ["remote"], ["keyboard"], ["cell phone"], ["microwave"],
    ["oven"], ["toaster"], ["sink"], ["refrigerator"], ["book"],
    ["clock"], ["vase"], ["scissors"], ["teddy bear"], ["hair drier"],
    ["toothbrush"]
]

# 用户提供的COCO类别ID列表
COCO_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

# 生成严格匹配的类别字典
COCO_CATEGORIES = {}
for idx, (class_group, class_id) in enumerate(zip(COCO_CLASSES, COCO_IDS)):
    for class_name in class_group:
        COCO_CATEGORIES[class_name] = class_id  # 严格保留原始名称大小写

def convert_annotation(input_path, output_path):
    # 读取原始标注
    with open(input_path) as f:
        original_data = json.load(f)

    # 初始化COCO格式和统计信息
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": COCO_IDS[i], "name": cls[0]} for i, cls in enumerate(COCO_CLASSES)]
    }

    stats = {
        "total_images": len(original_data),
        "total_objects": 0,
        "valid_objects": 0,
        "skipped_objects": 0,
        "skipped_images": 0
    }

    annotation_id = 1

    # 处理每个图像（带进度条）
    for img in tqdm(original_data, desc="Converting"):
        has_valid_annotation = False

        # 添加图像信息（需补充实际尺寸）
        coco_format["images"].append({
            "id": img["id"],
            "width": 640,  # 替换为实际宽度
            "height": 480,  # 替换为实际高度
            "file_name": f"{img['id']}.jpg"
        })

        # 处理对象
        for obj in img["objects"]:
            stats["total_objects"] += 1
            matched = False

            # 严格检查所有名称
            for name in obj["names"]:
                if name in COCO_CATEGORIES:
                    category_id = COCO_CATEGORIES[name]
                    matched = True
                    break

            if matched:
                # 转换标注
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": img["id"],
                    "category_id": category_id,
                    "bbox": [obj["x"], obj["y"], obj["w"], obj["h"]],
                    "area": obj["w"] * obj["h"],
                    "iscrowd": 0,
                    "segmentation": []
                })
                annotation_id += 1
                stats["valid_objects"] += 1
                has_valid_annotation = True
            else:
                stats["skipped_objects"] += 1

        if not has_valid_annotation:
            stats["skipped_images"] += 1

    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    # 打印统计信息
    print("\n转换统计:")
    print(f"总图像数: {stats['total_images']}")
    print(f"有效标注图像: {stats['total_images'] - stats['skipped_images']}")
    print(f"总对象数: {stats['total_objects']}")
    print(f"有效对象: {stats['valid_objects']} (保留率: {stats['valid_objects'] / stats['total_objects']:.1%})")
    print(f"无效对象: {stats['skipped_objects']} (原因: 非COCO类别)")

if __name__ == "__main__":
    input_path = "/root/autodl-tmp/objects.json"
    output_path = "/root/autodl-tmp/VGcoco_strict.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    convert_annotation(input_path, output_path)