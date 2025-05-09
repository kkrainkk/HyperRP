import json
from pathlib import Path
from tqdm import tqdm



COCO_CLASSES = [
    ["person", "man", "woman", "people", "boy", "girl", "child", "guy", "lady"],  # ID 1
    ["bike"],                                                                     # ID 2 (bicycle → bike)
    ["car"],                                                                      # ID 3
    ["motorcycle"],                                                               # ID 4
    ["airplane", "plane"],                                                        # ID 5
    ["bus"],                                                                      # ID 6
    ["train"],                                                                    # ID 7
    ["truck"],                                                                    # ID 8
    ["boat"],                                                                     # ID 9
    ["light", "lights"],                                                          # ID 10 (traffic light → light)
    None,                                                                         # ID 11 (fire hydrant)
    None,                                                                         # ID 13 (stop sign)
    None,                                                                         # ID 14 (parking meter)
    ["bench"],                                                                    # ID 15
    ["bird"],                                                                     # ID 16
    ["cat"],                                                                      # ID 17
    ["dog"],                                                                      # ID 18
    ["horse"],                                                                    # ID 19
    ["sheep"],                                                                    # ID 20
    ["cow"],                                                                      # ID 21
    ["elephant"],                                                                 # ID 22
    ["bear"],                                                                     # ID 23
    ["zebra"],                                                                    # ID 24
    ["giraffe"],                                                                  # ID 25
    None,                                                                         # ID 27 (backpack)
    ["umbrella"],                                                                 # ID 28
    ["bag"],                                                                      # ID 31 (handbag → bag)
    ["tie"],                                                                      # ID 32
    ["suitcase"],                                                                 # ID 33
    ["frisbee"],                                                                  # ID 34
    None,                                                                         # ID 35 (skis)
    None,                                                                         # ID 36 (snowboard)
    ["ball"],                                                                     # ID 37 (sports ball → ball)
    ["kite"],                                                                     # ID 38
    None,                                                                         # ID 39 (baseball bat)
    None,                                                                         # ID 40 (baseball glove)
    ["skateboard"],                                                               # ID 41
    ["surfboard"],                                                                # ID 42
    ["racket"],                                                                   # ID 43 (tennis racket → racket)
    ["bottle"],                                                                   # ID 44
    None,                                                                         # ID 46 (wine glass)
    ["cup"],                                                                      # ID 47
    ["fork"],                                                                     # ID 48
    None,                                                                         # ID 49 (knife)
    None,                                                                         # ID 50 (spoon)
    ["bowl"],                                                                     # ID 51
    ["banana"],                                                                   # ID 52
    ["apple"],                                                                    # ID 53
    ["sandwich"],                                                                 # ID 54
    ["orange"],                                                                   # ID 55
    ["broccoli"],                                                                 # ID 56
    None,                                                                         # ID 57 (carrot)
    None,                                                                         # ID 58 (hot dog)
    ["pizza"],                                                                    # ID 59
    ["donut"],                                                                    # ID 60
    ["cake"],                                                                     # ID 61
    ["chair"],                                                                    # ID 62
    ["couch"],                                                                    # ID 63
    ["plant", "potted plant"],                                                    # ID 64
    ["bed"],                                                                      # ID 65
    ["table", "dining table"],                                                    # ID 67
    ["toilet"],                                                                   # ID 70
    None,                                                                         # ID 72 (tv)
    ["laptop"],                                                                   # ID 73
    None,                                                                         # ID 74 (mouse)
    None,                                                                         # ID 75 (remote)
    ["keyboard"],                                                                 # ID 76
    None,                                                                         # ID 77 (cell phone)
    None,                                                                         # ID 78 (microwave)
    None,                                                                         # ID 79 (oven)
    None,                                                                         # ID 80 (toaster)
    ["sink"],                                                                     # ID 81
    None,                                                                         # ID 82 (refrigerator)
    ["book"],                                                                     # ID 84
    ["clock"],                                                                    # ID 85
    ["vase"],                                                                     # ID 86
    None,                                                                         # ID 87 (scissors)
    None,                                                                         # ID 88 (teddy bear)
    None,                                                                         # ID 89 (hair drier)
    None,                                                                          # ID 90 (toothbrush)
    ["window", "windows"],        # ID 91
    ["tree", "trees"],            # ID 92
    ["building"],                 # ID 93
    ["sky", "clouds", "cloud"],   # ID 94
    ["shirt"],                    # ID 95
    ["wall"],                     # ID 96
    ["ground"],                   # ID 97
    ["sign"],                     # ID 98
    ["grass"],                    # ID 99
    ["water"],                    # ID 100
    ["pole"],                     # ID 101
    ["head", "hair", "ear", "eye", "nose", "face", "eyes"],  # ID 102
    ["plate"],                    # ID 103
    ["leg", "legs"],              # ID 104
    ["fence"],                    # ID 105
    ["floor"],                    # ID 106
    ["door"],                     # ID 107
    ["pants"],                    # ID 108
    ["road"],                     # ID 109
    ["hat"],                      # ID 110
    ["snow"],                     # ID 111
    ["leaves"],                   # ID 112
    ["street"],                   # ID 113
    ["wheel", "tire", "wheels"],  # ID 114
    ["jacket"],                   # ID 115
    ["shadow"],                   # ID 116
    ["line", "lines"],            # ID 117
    ["field", "a field"],         # ID 118
    ["sidewalk"],                 # ID 119
    ["handle"],                   # ID 120
    ["tail"],                     # ID 121
    ["flower", "flowers"],        # ID 122
    ["helmet"],                   # ID 123
    ["leaf"],                     # ID 124
    ["shorts"],                   # ID 125
    ["glass"],                    # ID 126
    ["food"],                     # ID 127
    ["rock", "rocks", "a rock"],  # ID 128
    ["tile"],                     # ID 129
    ["player"],                   # ID 130
    ["post"],                     # ID 131
    ["logo"],                     # ID 132
    ["mirror"],                   # ID 133
    ["stripe", "stripes"],        # ID 134
    ["number"],                   # ID 135
    ["roof"],                     # ID 136
    ["picture"],                  # ID 137
    ["box"],                      # ID 138
    ["cap"],                      # ID 139
    ["pillow"],                   # ID 140
    ["tracks", "track"],          # ID 141
    ["background"],               # ID 142
    ["dirt"],                     # ID 143
    ["house"],                    # ID 144
    ["shelf"],                    # ID 145
    ["mouth"],                    # ID 146
    ["beach"],                    # ID 147
    ["trunk"],                    # ID 148
    ["spot"],                     # ID 149
    ["board"],                    # ID 150
    ["counter"],                  # ID 151
    ["top"],                      # ID 152
    ["sand"],                     # ID 153
    ["wave"],                     # ID 154
    ["bush"],                     # ID 155
    ["lamp"],                     # ID 156
    ["button"],                   # ID 157
    ["paper"],                    # ID 158
    ["flag"],                     # ID 159
    ["writing"],                  # ID 160
    ["brick"],                    # ID 161
    ["seat"],                     # ID 162
    ["glove"],                    # ID 163
    ["wing"],                     # ID 164
    ["part"],                     # ID 165
    ["vehicle"],                  # ID 166
    ["tower"],                    # ID 167
    ["reflection"],               # ID 168
    ["branch"],                   # ID 169
    ["edge"],                     # ID 170
    ["letters"],                  # ID 171
    ["ocean"],                    # ID 172
    ["animal"],                   # ID 173
    ["mountain", "hill"],         # ID 174
    ["cabinet"],                  # ID 175
    ["headlight"],                # ID 176
    ["ceiling"],                  # ID 177
    ["container"],                # ID 178
    ["skier"],                    # ID 179
    ["towel"],                    # ID 180
    ["frame"],                    # ID 181
    ["windshield"],               # ID 182
    ["fruit"],                    # ID 183
    ["pot"],                      # ID 184
    ["bat"],                      # ID 185
    ["basket"],                   # ID 186
    ["back"],                     # ID 187
    ["the outdoors"],             # ID 188
    ["finger"],                   # ID 189
    ["the carpet"]                # ID 190
]

# 完整的类别ID列表（COCO 80类 + 用户扩展100类）
COCO_IDS = [
    # COCO原始80类ID（1-90，部分ID空缺）
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,

    # 用户扩展类别ID（91-190）
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
    151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
    171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
    181, 182, 183, 184, 185, 186, 187, 188, 189, 190
]

# 生成类别名称到ID的映射（跳过None）
COCO_CATEGORIES = {}
for cls_group, cls_id in zip(COCO_CLASSES, COCO_IDS):
    if cls_group is None:
        continue
    for name in cls_group:
        COCO_CATEGORIES[name] = cls_id


def convert_annotation(input_path, output_path, image_data_path):
    # 加载原始标注和图像尺寸数据
    with open(input_path) as f:
        original_data = json.load(f)
    with open(image_data_path) as f:
        image_data = json.load(f)

    # 建立图像ID到尺寸的映射
    image_size_map = {img["id"]: (img["width"], img["height"]) for img in image_data}

    # 初始化COCO格式数据
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": id, "name": cls[0]}
            for cls, id in zip(COCO_CLASSES, COCO_IDS)
            if cls is not None
        ]
    }

    stats = {
        "total_images": 0,
        "total_objects": 0,
        "valid_objects": 0,
        "skipped_objects": 0,
        "skipped_images_missing_size": 0,
        "skipped_images_no_annotations": 0
    }

    annotation_id = 1

    # 处理每个图像
    for img in tqdm(original_data, desc="Converting"):
        img_id = img["id"]
        stats["total_images"] += 1

        # 检查图像尺寸是否存在
        if img_id not in image_size_map:
            stats["skipped_images_missing_size"] += 1
            continue

        width, height = image_size_map[img_id]

        # 添加图像信息
        coco_format["images"].append({
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": f"{img_id}.jpg"
        })

        has_valid_annotation = False

        # 处理每个对象
        for obj in img.get("objects", []):
            stats["total_objects"] += 1
            category_id = None

            # 检查所有可能的名称
            for name in obj.get("names", []):
                if name in COCO_CATEGORIES:
                    category_id = COCO_CATEGORIES[name]
                    break

            if category_id is not None:
                # 添加标注
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
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
            stats["skipped_images_no_annotations"] += 1

    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)

    # 打印统计信息
    print("\n转换统计:")
    print(f"总图像数: {stats['total_images']}")
    print(
        f"有效标注图像: {stats['total_images'] - stats['skipped_images_missing_size'] - stats['skipped_images_no_annotations']}")
    print(f"  缺失尺寸跳过的图像: {stats['skipped_images_missing_size']}")
    print(f"  无有效标注跳过的图像: {stats['skipped_images_no_annotations']}")
    print(f"总对象数: {stats['total_objects']}")
    print(f"有效对象: {stats['valid_objects']} (保留率: {stats['valid_objects'] / stats['total_objects']:.1%})")
    print(f"无效对象: {stats['skipped_objects']}")


if __name__ == "__main__":
    input_path = "/root/autodl-tmp/objects.json"
    image_data_path = "/root/autodl-tmp/image_data.json"  # 新增尺寸文件路径
    output_path = "/root/autodl-tmp/VGcoco_strict200.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    convert_annotation(input_path, output_path, image_data_path)