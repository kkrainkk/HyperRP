import json
import os


def remove_segmentation_fields(input_json, output_json=None):
    """
    移除COCO标注文件中的所有segmentation字段
    :param input_json: 输入标注文件路径
    :param output_json: 输出文件路径（默认在原路径添加 '_noseg' 后缀）
    """
    # 自动生成输出路径
    if output_json is None:
        base_name = os.path.splitext(input_json)[0]
        output_json = f"{base_name}_noseg.json"

    try:
        # 加载原始标注文件
        with open(input_json, 'r') as f:
            data = json.load(f)

        # 遍历所有标注并移除segmentation字段
        removed_count = 0
        for ann in data.get('annotations', []):
            if 'segmentation' in ann:
                del ann['segmentation']
                removed_count += 1

        # 保存处理后的文件
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ 成功处理 {removed_count} 个标注 | 输出文件: {os.path.abspath(output_json)}")
        print(f"剩余标注总数: {len(data['annotations'])}")
        print("首标注字段示例:", list(data['annotations'][0].keys()) if data['annotations'] else "无标注")

    except FileNotFoundError:
        print(f"❌ 文件未找到: {input_json}")
    except json.JSONDecodeError:
        print(f"❌ 文件格式错误: {input_json} 不是有效的JSON")
    except KeyError as e:
        print(f"❌ 数据结构错误: 缺少必要字段 {str(e)}")
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")


if __name__ == "__main__":
    # 配置路径
    input_file = "/root/autodl-tmp/COCO/annotations/instances_train2017.json"
    output_file = "/root/autodl-tmp/COCO/annotations/instances_train2017_noseg.json"

    # 执行处理
    remove_segmentation_fields(input_file, output_file)