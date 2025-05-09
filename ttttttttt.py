import json
from collections import defaultdict, Counter


def find_synonyms(json_path, target_groups):
    # 提取目标类别列表
    targets = [group[0] for group in target_groups]

    # 加载JSON数据
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 初始化同义词计数器
    synonym_counts = defaultdict(Counter)

    # 遍历所有对象并统计同义词
    for image_data in data:
        for obj in image_data.get("objects", []):
            names = obj.get("names", [])
            for target in targets:
                if target in names:
                    for word in names:
                        if word != target:
                            synonym_counts[target][word] += 1

    # 打印结果
    print("近义词和同义词统计：")
    for target in targets:
        print(f"\n类 '{target}' 的近义词和同义词：")
        synonyms = synonym_counts[target]

        if not synonyms:
            print("  无")
            continue

        # 按次数降序和字母顺序排序
        sorted_synonyms = sorted(synonyms.items(), key=lambda x: (-x[1], x[0]))

        for idx, (synonym, count) in enumerate(sorted_synonyms, 1):
            print(f"  {idx}. {synonym}: {count}次")


# 使用示例
target_classes = [
    ["airplane"],
    ["bus"],
    ["cat"],
    ["dog"],
    ["cow"],
    ["elephant"],
    ["umbrella"],
    ["tie"],
    ["snowboard"],
    ["skateboard"],
    ["cup"],
    ["knife"],
    ["cake"],
    ["couch"],
    ["keyboard"],
    ["sink"],
    ["scissors"]
]

find_synonyms("/root/autodl-tmp/objects.json", target_classes)


