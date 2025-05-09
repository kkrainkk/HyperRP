import torch.nn as nn
import torch.nn.functional as F

class TextGuidedAttention(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim=512, num_heads=8):
        super(TextGuidedAttention, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 文本特征映射到隐藏空间
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # 最小尺度图像特征映射到隐藏空间
        self.image_proj = nn.Conv2d(image_dim, hidden_dim, kernel_size=1)

        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)

        # 输出映射
        self.text_out = nn.Linear(hidden_dim, text_dim)
        self.image_out = nn.Conv2d(hidden_dim, image_dim, kernel_size=1)

        # ShapeAdapter 模块
        self.shape_adapter = ShapeAdapter(input_dim=80, output_dim=self.hidden_dim)

    def forward(self, text_feature, image_features):
        # 确保输入的最小尺度特征图的通道数与 image_dim 一致
        small_image_feature = image_features[2]  # 最小尺度特征图为 image_features[2]
        assert small_image_feature.size(
            1) == self.image_dim, f"Image feature channel mismatch: expected {self.image_dim}, got {small_image_feature.size(1)}"

        # 将文本特征和最小尺度图像特征映射到同一空间
        text_proj = self.text_proj(text_feature)  # (16, 80, hidden_dim)
        small_image_proj = self.image_proj(small_image_feature)  # (16, hidden_dim, 20, 20)

        # 将图像特征展平为序列形式
        small_image_proj_flat = small_image_proj.view(small_image_proj.size(0), self.hidden_dim, -1).permute(2, 0,
                                                                                                             1)  # (400, 16, hidden_dim)

        # 双向注意力机制
        # 文本作为Q，图像作为KV
        text_proj_permuted = text_proj.permute(1, 0, 2)  # (80, 16, hidden_dim)
        attn_output, _ = self.multihead_attn(text_proj_permuted, small_image_proj_flat,
                                             small_image_proj_flat)  # (80, 16, hidden_dim)
        updated_image_feature = attn_output.permute(1, 0, 2)  # (16, 80, hidden_dim)

        # 图像作为Q，文本作为KV
        text_proj_flat = text_proj_permuted.repeat(small_image_proj_flat.size(0) // text_proj_permuted.size(0), 1,
                                                   1)  # (400, 16, hidden_dim)
        updated_text_feature, _ = self.multihead_attn(small_image_proj_flat, text_proj_flat,
                                                      text_proj_flat)  # (400, 16, hidden_dim)
        updated_text_feature = updated_text_feature.mean(dim=0)  # (16, hidden_dim)

        # 将更新后的特征映射回原始空间
        updated_text_feature = self.text_out(updated_text_feature)  # (16, text_dim)

        # 使用 ShapeAdapter 模块调整特征图形状
        updated_image_feature = updated_image_feature[:16, :, :]  # 取前8个样本 (8, 80, hidden_dim)
        updated_image_feature = self.shape_adapter(updated_image_feature)  # (8, hidden_dim, 20, 20)
        updated_image_feature = self.image_out(updated_image_feature)  # (8, image_dim, 20, 20)



        return updated_text_feature, updated_image_feature



class ShapeAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ShapeAdapter, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=output_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=1)

    def forward(self, x):
        # 检查输入和设备是否一致
        if x.device != self.conv1.weight.device:
            x = x.to(self.conv1.weight.device)
        x = x.permute(0, 2, 1)  # 转换为 (8, 512, 80)
        x = self.conv1(x)  # (8, output_dim, 80)
        x = x.unsqueeze(-1).repeat(1, 1, 1, 20)  # 扩展为 (8, output_dim, 80, 20)
        x = self.conv2(x)  # (8, output_dim, 80, 20)
        x = x[:, :, :20, :]  # 截取前20行 (8, output_dim, 20, 20)
        return x