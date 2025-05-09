import torch

# -----------------------------
# 步骤 1：加载并修改 YOLOv8 权重键名
# -----------------------------
yolov8_weights = torch.load("/root/autodl-tmp/yolov8l-cls.pth")
modified_weights = {}

# 修改视觉主干键名（添加 .image_model 前缀）
for key, value in yolov8_weights["state_dict"].items():
    if key.startswith("backbone."):
        new_key = key.replace("backbone.", "backbone.image_model.")
        modified_weights[new_key] = value
    else:
        # 保留非主干部分（如 neck/head）
        modified_weights[key] = value

# -----------------------------
# 步骤 2：加载并提取 YOLO-World 文本编码器权重
# -----------------------------
yolo_world_weights = torch.load("/root/autodl-tmp/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth")

# 提取文本编码器权重（backbone.text_model.）
text_weights = {}
for key, value in yolo_world_weights["state_dict"].items():
    if key.startswith("backbone.text_model."):
        text_weights[key] = value

# -----------------------------
# 步骤 3：合并权重（覆盖冲突键）
# -----------------------------
# 将文本编码器权重添加到修改后的权重中
modified_weights.update(text_weights)

# -----------------------------
# 步骤 4：保存合并后的权重文件
# -----------------------------
torch.save(
    {
        "state_dict": modified_weights,
        #"meta": yolov8_weights["meta"]  # 保留原始元信息（可选）
    },
    "/root/autodl-tmp/merged_yolov8_yoloworld_weights_ImageNet_pretrain.pth"
)

print("权重合并完成！保存为 /root/autodl-tmp/merged_yolov8_yoloworld_weights_ImageNet_pretrain.pth")