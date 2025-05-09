import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List

# =============== SE模块（注意力机制） ===============
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, activation=F.relu):
        super(SEBlock, self).__init__()
        self.reduction = reduction
        self.activation = activation

        # 1x1卷积，用于压缩通道
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        # 1x1卷积，用于恢复通道
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

        # 批归一化
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # 全局平均池化
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        # 压缩通道
        scale = self.activation(self.bn1(self.fc1(scale)))
        # 恢复通道
        scale = torch.sigmoid(self.bn2(self.fc2(scale)))
        return x * scale

class HypergraphConv(nn.Module):
    def __init__(self, in_channels, hyperedge_num=16):
        """
        超图卷积模块
        Args:
            in_channels: 输入特征的通道数
            hyperedge_num: 超边的数量
        """
        super(HypergraphConv, self).__init__()
        self.hyperedge_num = hyperedge_num

        # 生成超边关联矩阵：将节点特征映射到超边数维度
        self.membership_conv = nn.Conv2d(in_channels, hyperedge_num, kernel_size=1)
        # 将超边特征转换回节点特征前，对超边特征进行转换时保持超边数不变
        self.hyperedge_conv = nn.Conv1d(hyperedge_num, hyperedge_num, kernel_size=1)
        # 节点特征的线性变换
        self.node_transform = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        输入:
            x: [B, C, H, W]
        输出:
            out: [B, C, H, W]
        """
        B, C, H, W = x.size()
        N = H * W  # 节点数

        # 1. 生成关联矩阵
        membership = self.membership_conv(x)  # 形状 [B, hyperedge_num, H, W]
        membership = membership.view(B, self.hyperedge_num, N)  # [B, hyperedge_num, N]
        membership = F.softmax(membership, dim=1)  # 在超边维度归一化

        # 2. 将节点特征展平
        x_flat = x.view(B, C, N)  # [B, C, N]

        # 3. 聚合超边特征，结果形状为 [B, C, hyperedge_num]
        hyperedge_features = torch.bmm(x_flat, membership.transpose(1, 2))
        # 4. 调整形状为 [B, hyperedge_num, C] 以符合 Conv1d 的要求
        hyperedge_features = hyperedge_features.transpose(1, 2)
        # 5. 对超边特征进行转换，保持超边数不变
        hyperedge_features = self.relu(self.hyperedge_conv(hyperedge_features))
        # 6. 将转换后的超边特征恢复为 [B, C, hyperedge_num]
        hyperedge_features = hyperedge_features.transpose(1, 2)
        # 7. 将超边特征反馈到节点，得到 [B, C, N]
        x_updated = torch.bmm(hyperedge_features, membership)
        # 8. 节点级别的变换
        x_transformed = self.node_transform(x_flat)
        # 9. 融合并添加非线性激活（残差连接）
        out = self.relu(x_transformed + x_updated)
        # 10. 重塑为原始空间尺寸 [B, C, H, W]
        out = out.view(B, C, H, W)
        return out


# =============== CustomFPN with PANet and Hypergraph Convolution ===============
class CustomFPN(nn.Module):
    def __init__(self, in_channels_list: List[int], reduction=16, dropout=0.5, hyperedge_num=16):
        """
        FPN网络，结合PANet结构和超图卷积优化
        Args:
            in_channels_list: 每个特征层的通道数列表
            reduction: SE模块中的通道压缩比例
            dropout: dropout层的概率
            hyperedge_num: 每个超图卷积模块中超边的数量
        """
        super(CustomFPN, self).__init__()
        self.in_channels_list = in_channels_list
        self.num_levels = len(in_channels_list)

        # 横向连接
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, kernel_size=1) for in_channels in in_channels_list]
        )

        # 上采样调整通道数（将低分辨率特征上采样后与高分辨率特征相加）
        self.upsample_convs = nn.ModuleList(
            [nn.Conv2d(in_channels_list[i], in_channels_list[i - 1], kernel_size=1)
             for i in range(1, self.num_levels)]
        )

        # 下采样增强路径（PANet：自底向上路径）
        self.downsample_convs = nn.ModuleList(
            [nn.Conv2d(in_channels_list[i], in_channels_list[i + 1], kernel_size=3, stride=2, padding=1)
             for i in range(self.num_levels - 1)]
        )

        # 输出卷积
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) for in_channels in in_channels_list]
        )

        # SE注意力模块
        self.se_blocks = nn.ModuleList(
            [SEBlock(in_channels, reduction) for in_channels in in_channels_list]
        )

        # 超图卷积模块，用于进一步优化特征表示
        self.hypergraph_blocks = nn.ModuleList(
            [HypergraphConv(in_channels, hyperedge_num) for in_channels in in_channels_list]
        )

        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)

        # BatchNorm层
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm2d(in_channels) for in_channels in in_channels_list]
        )

    def forward(self, inputs: List[torch.Tensor]) -> OrderedDict:
        """
        Args:
            inputs: 来自骨干网络的特征图列表，每个形状为 [B, C, H, W]
        Returns:
            OrderedDict 格式的特征图，键名如 'fpn_0', 'fpn_1', 等
        """
        assert len(inputs) == self.num_levels, "输入特征图数量与 in_channels_list 不匹配。"

        # 横向连接
        lateral_feats = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, inputs)]

        # 自顶向下特征融合
        for i in range(self.num_levels - 1, 0, -1):
            upsampled = F.interpolate(lateral_feats[i], size=lateral_feats[i - 1].shape[-2:], mode='bilinear',
                                      align_corners=False)
            lateral_feats[i - 1] += self.upsample_convs[i - 1](upsampled)

        # 自底向上路径增强（PANet）
        for i in range(self.num_levels - 1):
            downsampled = self.downsample_convs[i](lateral_feats[i])
            lateral_feats[i + 1] += downsampled

        outputs = []
        # 对每个特征层依次进行 BatchNorm、Dropout、超图卷积、输出卷积以及 SE 注意力调制
        for i, feat in enumerate(lateral_feats):
            feat = self.bn_layers[i](feat)
            feat = self.dropout_layer(feat)
            feat = self.hypergraph_blocks[i](feat)  # 应用超图卷积模块
            feat = self.output_convs[i](feat)
            feat = self.se_blocks[i](feat)  # 应用 SE 注意力模块
            outputs.append(feat)

        return OrderedDict(zip([f'fpn_{i}' for i in range(len(outputs))], outputs))
