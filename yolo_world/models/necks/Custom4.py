import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List


# =============== 超图卷积模块 ===============

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        """
        x: [B, N, in_ft] - 节点特征
        G: [B, N, num_edges] - 超边关联矩阵
        """
        # 线性变换
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias

        # Step 1: 节点 -> 超边
        hyperedge_features = torch.bmm(G.transpose(1, 2), x)  # [B, num_edges, out_ft]

        # Step 2: 超边 -> 节点
        x = torch.bmm(G, hyperedge_features)  # [B, N, out_ft]

        return x


# =============== 超图网络模块 ===============

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout

        # 增加卷积层深度
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_hid)  # 增加层次
        self.hgc4 = HGNN_conv(n_hid, n_class)  # 增加层次

        self.norm1 = nn.LayerNorm(n_hid)
        self.norm2 = nn.LayerNorm(n_hid)
        self.norm3 = nn.LayerNorm(n_hid)  # 新增归一化层
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, G):
        # 残差连接与层归一化
        residual = x
        x = F.relu(self.norm1(self.hgc1(x, G)))
        x = self.dropout_layer(x)
        x = F.relu(self.norm2(self.hgc2(x, G)))
        x = self.dropout_layer(x)
        x = F.relu(self.norm3(self.hgc3(x, G)))  # 增加一层处理
        x = self.dropout_layer(x)
        x = self.hgc4(x, G)  # 最后一层

        return x + residual  # 残差连接


# =============== 超边生成器（HyperEdgeGenerator） ===============

class HyperEdgeGenerator(nn.Module):
    def __init__(self, in_channels, num_edges, hidden_dim=128, dropout=0.5):
        super(HyperEdgeGenerator, self).__init__()
        self.num_edges = num_edges  # 超边的数量
        self.hidden_dim = hidden_dim

        # 增加更多隐藏层以增强网络的表达能力
        self.edge_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_edges)  # 输出每个节点属于哪些超边
        )

    def forward(self, x):
        """
        输入: x -> [B, N, C], 其中 B 是 batch size, N 是节点数, C 是特征维度
        输出: hyperedge_adj -> [B, N, num_edges], 表示每个节点与超边的关联强度
        """
        B, N, C = x.shape
        x = x.view(B * N, C)  # 展平为 [B*N, C]
        hyperedge_adj = self.edge_net(x)  # [B*N, num_edges]
        hyperedge_adj = hyperedge_adj.view(B, N, self.num_edges)  # [B, N, num_edges]
        return hyperedge_adj


# =============== 自定义多尺度模块（CustomMultiScaleModule） ===============

class CustomMultiScaleModule(nn.Module):
    def __init__(self, in_channels, out_channels, H=None, W=None, num_edges=64, dropout=0.5):
        super(CustomMultiScaleModule, self).__init__()
        self.H = H
        self.W = W
        self.num_edges = num_edges

        # 超图卷积网络（HGNN）
        self.hyperconv = HGNN(in_ch=in_channels, n_class=out_channels, n_hid=256, dropout=dropout)

        # 超边生成器
        self.hyperedge_generator = HyperEdgeGenerator(in_channels, num_edges, dropout=dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # 将特征图转换为节点特征
        x_pixels = x.view(B, C, N).permute(0, 2, 1).contiguous()  # [B, N, C]

        # 生成超边关联矩阵
        G = self.hyperedge_generator(x_pixels)  # [B, N, num_edges]

        # 超图卷积
        out = self.hyperconv(x_pixels, G)  # [B, N, out_channels]

        # 将输出转换回特征图形式
        out_pixels = out.permute(0, 2, 1).contiguous().view(B, -1, H, W)
        return out_pixels


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


# =============== CustomFPN with PANet ===============

class CustomFPN(nn.Module):
    def __init__(self, in_channels_list: List[int], reduction=16, dropout=0.5):
        super(CustomFPN, self).__init__()
        self.in_channels_list = in_channels_list
        self.num_levels = len(in_channels_list)

        # 横向连接
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, kernel_size=1) for in_channels in in_channels_list])

        # 上采样调整通道数
        self.upsample_convs = nn.ModuleList([nn.Conv2d(in_channels_list[i], in_channels_list[i - 1], kernel_size=1)
                                             for i in range(1, self.num_levels)])

        # 下采样增强路径（PANet）
        self.downsample_convs = nn.ModuleList(
            [nn.Conv2d(in_channels_list[i], in_channels_list[i + 1], kernel_size=3, stride=2, padding=1)
             for i in range(self.num_levels - 1)])

        # 输出卷积
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) for in_channels in in_channels_list])

        # SE注意力模块
        self.se_blocks = nn.ModuleList([SEBlock(in_channels, reduction) for in_channels in in_channels_list])

        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)

        # BatchNorm层
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(in_channels) for in_channels in in_channels_list])

    def forward(self, inputs: List[torch.Tensor]) -> OrderedDict:
        assert len(inputs) == self.num_levels, "输入特征图数量与 in_channels_list 不匹配。"

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

        # 应用输出卷积、SE注意力和Dropout
        outputs = []
        for i, feat in enumerate(lateral_feats):
            feat = self.bn_layers[i](feat)
            feat = self.dropout_layer(feat)
            feat = self.se_blocks[i](self.output_convs[i](feat))  # SE模块
            outputs.append(feat)

        return OrderedDict(zip([f'fpn_{i}' for i in range(len(outputs))], outputs))
