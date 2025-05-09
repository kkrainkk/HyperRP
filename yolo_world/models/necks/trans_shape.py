import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureShapeAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # 为第三个特征的通道转换（1024->512）定义1x1卷积
        self.channel_adjust = nn.Conv2d(1024, 512, kernel_size=1)

    def forward(self, features: list) -> list:
        """
        输入：包含三个特征图的列表，形状分别为
        [B,256,56,56], [B,512,28,28], [B,1024,14,14]

        输出：调整后的三个特征图列表，形状分别为
        [B,256,80,80], [B,512,40,40], [B,512,20,20]
        """
        # 解包输入特征
        feat1, feat2, feat3 = features

        # 调整第一个特征（56x56 -> 80x80）
        adj1 = F.interpolate(feat1, size=80, mode='bilinear', align_corners=False)

        # 调整第二个特征（28x28 -> 40x40）
        adj2 = F.interpolate(feat2, size=40, mode='bilinear', align_corners=False)

        # 调整第三个特征（14x14 -> 20x20 + 通道调整）
        adj3 = self.channel_adjust(feat3)  # 1024->512
        adj3 = F.interpolate(adj3, size=20, mode='bilinear', align_corners=False)

        return [adj1, adj2, adj3]
