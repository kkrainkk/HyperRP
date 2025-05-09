import json
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", default="/root/autodl-tmp/LVIS/lvis_v1_val.json")  # 输入文件路径
args = parser.parse_args()

# 读取 JSON 文件
with open(args.json_path, 'r') as f:
    json_coco = json.load(f)

# 打印注释数量
num_annotations = len(json_coco['annotations'])
print(f"Total number of annotations: {num_annotations}")
