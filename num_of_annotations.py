import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_coco_annotations(json_path):
    """专业级COCO标注分析工具"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"标注文件未找到: {json_path}")
    except json.JSONDecodeError:
        raise ValueError("文件格式错误，无法解析为JSON")

    # 构建ID到类别的双向映射
    id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    name_to_id = {cat['name']: cat['id'] for cat in data['categories']}

    # 初始化统计结构（包含零计数）
    category_stats = defaultdict(int)
    for cat_id in id_to_name:
        category_stats[cat_id] = 0

    # 统计标注分布
    total_annotations = len(data['annotations'])
    for ann in data['annotations']:
        category_stats[ann['category_id']] += 1

    # 排序策略：按标注量降序，其次按ID升序
    sorted_stats = sorted(
        [(id_to_name[k], k, v) for k, v in category_stats.items()],
        key=lambda x: (-x[2], x[1])
    )

    # 专业级输出格式
    print(f"\n{' Analysis Report ':=^80}")
    print(f"数据集基本信息:")
    print(f"  - 图像总数: {len(data['images']):,}")
    print(f"  - 标注总数: {total_annotations:,}")
    print(f"  - 类别总数: {len(data['categories']):,}\n")

    print(f"{' Class Distribution ':-^80}")
    print(f"{'ID':<5}{'Category Name':<25}{'Count':<10}{'Percentage':<10}")
    print("-" * 50)

    for name, cat_id, count in sorted_stats:
        if count == 0:
            continue
        perc = count / total_annotations * 100
        print(f"{cat_id:<5}{name:<25}{count:<10,}{perc:>7.2f}%")

    # 长尾分析
    print("\n长尾分布特征:")
    sorted_counts = sorted([c for _, _, c in sorted_stats], reverse=True)
    head_20 = sum(sorted_counts[:int(len(sorted_counts) * 0.2)])
    tail_20 = sum(sorted_counts[-int(len(sorted_counts) * 0.2):])

    print(f"  - 头部20%类别标注量: {head_20 / total_annotations:.1%}")
    print(f"  - 尾部20%类别标注量: {tail_20 / total_annotations:.1%}")

    # 可视化分析
    plt.figure(figsize=(12, 6))

    # 长尾分布图
    plt.subplot(1, 2, 1)
    plt.plot(sorted_counts, marker='o')
    plt.title('Long-Tail Distribution')
    plt.xlabel('Class Rank')
    plt.ylabel('Annotation Count')
    plt.grid(True)

    # 类别占比饼图（前10）
    plt.subplot(1, 2, 2)
    top_10 = sorted_counts[:10]
    labels = [f"{name}\n({count})" for name, _, count in sorted_stats[:10]]
    plt.pie(top_10, labels=labels, startangle=90, autopct='%1.1f%%')
    plt.title('Top 10 Classes Distribution')

    plt.tight_layout()
    plt.savefig('annotation_analysis.png', dpi=300)
    print("\n可视化结果已保存至: annotation_analysis.png")


if __name__ == "__main__":
    json_path = "/root/autodl-tmp/VGcoco_strict91.json"

    # 验证文件存在性
    if not Path(json_path).exists():
        print(f"错误: 文件路径不存在 {json_path}")
        exit(1)

    try:
        analyze_coco_annotations(json_path)
    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")
        exit(1)