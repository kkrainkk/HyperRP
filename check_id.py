import json
from collections import defaultdict


def check_bus_mapping(json_path):
    # 加载COCO格式标注文件
    with open(json_path) as f:
        data = json.load(f)

    # 查找bus类别的定义
    bus_id = None
    id_conflict = False
    category_names = {}

    # 第一步：检查categories中的定义
    for category in data["categories"]:
        # 获取标准名称和同义词
        primary_name = category["name"]
        synonyms = category.get("synonyms", [])

        # 检查是否包含bus定义
        if "bus" == primary_name.lower() or "bus" in [s.lower() for s in synonyms]:
            bus_id = category["id"]
            print(f"✅ 找到bus定义: ID={bus_id}")
            print(f"   主名称: {primary_name}")
            print(f"   同义词: {synonyms}")

        # 记录所有ID对应的名称（检查ID冲突）
        if category["id"] in category_names:
            id_conflict = True
            print(f"⚠️ ID冲突: ID {category['id']} 被多个类别使用")
            print(f"   已有类别: {category_names[category['id']]}")
            print(f"   当前类别: {primary_name}")
        category_names[category["id"]] = primary_name

    if bus_id is None:
        print("❌ 错误: 未找到bus类别的定义")
        return

    # 第二步：统计标注中的映射情况
    bus_annotations = []
    wrong_mappings = []

    for ann in data["annotations"]:
        if ann["category_id"] == bus_id:
            # 获取对应的对象名称（假设原始数据中有names字段）
            obj_names = ann.get("names", ["unknown"])

            # 验证名称是否确实属于bus的同义词
            is_valid = any(name.lower() in ["bus", "omnibus"] for name in obj_names)

            if is_valid:
                bus_annotations.append(ann["id"])
            else:
                wrong_mappings.append({
                    "annotation_id": ann["id"],
                    "names": obj_names,
                    "mapped_id": ann["category_id"]
                })

    # 输出统计结果
    print("\n🔍 检查结果:")
    print(f"标注中标记为bus类(ID={bus_id})的实例总数: {len(bus_annotations)}")
    print(f"疑似错误映射数量: {len(wrong_mappings)}")

    if len(wrong_mappings) > 0:
        print("\n❌ 发现错误映射（示例）：")
        for wrong in wrong_mappings[:3]:  # 显示前3个错误示例
            print(f"标注ID {wrong['annotation_id']}")
            print(f"  原始名称: {wrong['names']}")
            print(f"  被映射到: ID {wrong['mapped_id']}")

    # 第三步：检查ID冲突
    if id_conflict:
        print("\n⚠️ 警告: 检测到多个类别共享相同ID")
    else:
        print("\n✅ ID分配正常，无冲突")


if __name__ == "__main__":
    json_file = "/root/autodl-tmp/VGcoco_strict.json"
    check_bus_mapping(json_file)