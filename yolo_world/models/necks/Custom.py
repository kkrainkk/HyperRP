import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from collections import OrderedDict
from typing import List


# =============== 超图卷积层 ===============
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')  # Kaiming 初始化
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        G = G.to(x.device)

        # 线性变换：xW + b
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias

        # 基于超图邻接矩阵 G 进行信息传播
        degree = G.sum(dim=2, keepdim=True)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
        G_normalized = degree_inv_sqrt * G * degree_inv_sqrt.transpose(1, 2)

        x = G_normalized.matmul(x)
        return x


# =============== 两层 HGNN 网络 ===============
class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_class)
        self.norm1 = nn.LayerNorm(n_hid)  # 添加 LayerNorm
        self.norm2 = nn.LayerNorm(n_hid)  # 添加 LayerNorm

    def forward(self, x, G):
        x_regin = x
        x = F.relu(self.norm1(self.hgc1(x, G)))  # 加入 LayerNorm
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.norm2(self.hgc2(x, G)))  # 加入 LayerNorm
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x, G)
        return x + x_regin


# =============== CustomMultiScaleModule ===============
class CustomMultiScaleModule(nn.Module):
    def __init__(self, in_channels, out_channels,
                 H=None, W=None, threshold=0.5):
        super(CustomMultiScaleModule, self).__init__()
        self.hyperconv = HGNN(in_ch=in_channels, n_class=out_channels, n_hid=256)
        self.H = H
        self.W = W

        # 可学习的阈值 - 用于像素-像素之间
        self.threshold = nn.Parameter(
            torch.clamp(torch.Tensor([threshold]), min=0.01, max=1.0)
        )

    def create_hypergraph_adjacency(self, x_pixels, device):
        B, N, C = x_pixels.shape

        # 计算像素-像素子图
        distance = torch.cdist(x_pixels, x_pixels, p=2)
        distance = distance / distance.max()  # 归一化距离
        pixel_adj = (distance < self.threshold).float()

        # 添加自环
        adjacency = pixel_adj + torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
        return adjacency

    def forward(self, x):
        device = x.device
        B, C, H, W = x.shape

        N = H * W
        x_pixels = x.view(B, C, N).permute(0, 2, 1).contiguous()

        G = self.create_hypergraph_adjacency(x_pixels, device)

        out = self.hyperconv(x_pixels, G)
        out_pixels = out.permute(0, 2, 1).contiguous().view(B, -1, H, W)
        return out_pixels


# =============== SE模块（注意力机制） ===============
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


# =============== CustomFPN with PANet ===============
class CustomFPN(nn.Module):
    def __init__(self, in_channels_list: List[int]):
        super(CustomFPN, self).__init__()
        self.in_channels_list = in_channels_list
        self.num_levels = len(in_channels_list)

        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1) for in_channels in in_channels_list
        ])

        # 上采样调整通道数
        self.upsample_convs = nn.ModuleList([
            nn.Conv2d(in_channels_list[i], in_channels_list[i - 1], kernel_size=1)
            for i in range(1, self.num_levels)
        ])

        # 下采样增强路径（PANet）
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(in_channels_list[i], in_channels_list[i + 1], kernel_size=3, stride=2, padding=1)
            for i in range(self.num_levels - 1)
        ])

        # 输出卷积
        self.output_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) for in_channels in in_channels_list
        ])

        # SE注意力模块
        self.se_blocks = nn.ModuleList([SEBlock(in_channels) for in_channels in in_channels_list])

    def forward(self, inputs: List[torch.Tensor]) -> OrderedDict:
        assert len(inputs) == self.num_levels, "输入特征图数量与 in_channels_list 不匹配。"

        lateral_feats = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, inputs)]

        # 自顶向下特征融合
        for i in range(self.num_levels - 1, 0, -1):
            upsampled = F.interpolate(lateral_feats[i], size=lateral_feats[i - 1].shape[-2:], mode='bilinear')
            lateral_feats[i - 1] += self.upsample_convs[i - 1](upsampled)

        # 自底向上路径增强（PANet）
        for i in range(self.num_levels - 1):
            downsampled = self.downsample_convs[i](lateral_feats[i])
            lateral_feats[i + 1] += downsampled

        # 应用输出卷积和注意力
        outputs = [self.se_blocks[i](self.output_convs[i](feat)) for i, feat in enumerate(lateral_feats)]

        return OrderedDict(zip([f'fpn_{i}' for i in range(len(outputs))], outputs))












