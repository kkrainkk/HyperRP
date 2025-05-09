import torch
import torch.nn as nn
import torch.nn.functional as F


# =============== LightweightCrossAttention (LCA) ===============
class LightweightCrossAttention(nn.Module):
    """
    轻量级跨尺度交互层 (LCA)
    - 用 Depthwise Conv 替代 MHSA
    - 用 Grouped Attention 降低计算复杂度
    """
    def __init__(self, in_channels, embed_dim, reduction=2):
        super(LightweightCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.reduction = reduction

        # 动态计算 groups，确保 in_channels 兼容
        num_groups = max(1, min(in_channels // 32, embed_dim // 32))

        # 轻量级 Query, Key, Value 计算
        self.query_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1, groups=num_groups)

        # Depthwise Conv 计算注意力
        self.depthwise_attn = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)

    def forward(self, feat_map, hg_feature):
        """
        feat_map:    [B, C, H, W]
        hg_feature:  [B, C, H, W]
        """
        # 确保 hg_feature 形状匹配
        if hg_feature.shape[1] != feat_map.shape[1]:
            hg_feature = F.interpolate(hg_feature, size=feat_map.shape[2:], mode='bilinear')

        # Query, Key, Value
        query = self.query_conv(hg_feature)  # [B, embed_dim, H, W]
        key = self.key_conv(feat_map)  # [B, embed_dim, H, W]
        value = self.value_conv(feat_map)  # [B, embed_dim, H, W]

        # Depthwise 计算交互注意力
        attn_weights = self.depthwise_attn(query * key)  # [B, embed_dim, H, W]
        attn_weights = F.softmax(attn_weights, dim=1)

        # 融合
        attn_result = attn_weights * value  # [B, embed_dim, H, W]

        # === 新增残差连接：将 attn_result 与 query 相加 ===
        out = attn_result + query  # [B, embed_dim, H, W]

        return out


# =============== HypergraphFeedbackFusion ===============
class HypergraphFeedbackFusion(nn.Module):
    def __init__(self, in_channels_list, hg_channels=256, embed_dim=256, reduction=2):
        super(HypergraphFeedbackFusion, self).__init__()
        self.reduction = reduction
        self.embed_dim = embed_dim
        self.num_scales = len(in_channels_list)
        self.hg_channels = hg_channels

        # 最顶层 Multi-Head Self-Attention
        self.mhsa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, dropout=0.1)

        # 其他层使用 LCA
        self.lca_layers = nn.ModuleList([
            LightweightCrossAttention(in_channels, embed_dim, reduction)
            for in_channels in in_channels_list
        ])

        # 1x1 Conv for channel fusion
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(embed_dim, in_channels, kernel_size=1) for in_channels in in_channels_list
        ])

        # 可学习融合权重
        self.alpha = nn.Parameter(torch.ones(self.num_scales), requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(self.num_scales), requires_grad=True)

        # 调整 hg_feature 的通道以匹配每层的输入
        self.hg_channel_adjust = nn.ModuleList([
            nn.Conv2d(hg_channels, in_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])

        # 确保顶层特征可以投影到 embed_dim
        self.projection = nn.Conv2d(in_channels_list[-1], embed_dim, kernel_size=1)

    def forward(self, feature_maps, hg_feature):
        updated_features = []

        for i, feat_map in enumerate(feature_maps):
            B, C, H, W = feat_map.shape

            # 调整并下采样 hg_feature
            hg_down = F.adaptive_avg_pool2d(hg_feature, (H, W))
            hg_down = self.hg_channel_adjust[i](hg_down)

            if i == len(feature_maps) - 1:
                # === 顶层使用 MHSA 并加残差 ===

                # 先投影到 [B, embed_dim, H, W] => [HW, B, embed_dim]
                query = self.projection(hg_down).flatten(2).permute(2, 0, 1)  # [HW, B, embed_dim]
                key = self.projection(feat_map).flatten(2).permute(2, 0, 1)  # [HW, B, embed_dim]
                value = self.projection(feat_map).flatten(2).permute(2, 0, 1)  # [HW, B, embed_dim]

                # MHSA
                attn_output, _ = self.mhsa(query, key, value)  # [HW, B, embed_dim]

                # === 新增残差: output + query ===
                attn_output = attn_output + query

                # 恢复形状 => [B, embed_dim, H, W]
                attn_output = attn_output.permute(1, 2, 0).view(B, self.embed_dim, H, W)

            else:
                # 其他层使用 LCA
                attn_output = self.lca_layers[i](feat_map, hg_down)  # [B, embed_dim, H, W]

            # 1x1 Conv 映射回原通道
            attn_result = self.fusion_layers[i](attn_output)  # [B, in_channels, H, W]

            # 加权融合 + 残差
            fused_feature = self.alpha[i] * attn_result + self.beta[i] * feat_map

            updated_features.append(fused_feature)

        return updated_features


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


# =============== HyperFPN ===============
class HyperFPN(nn.Module):
    def __init__(self, in_channels_list, hg_channels=256, embed_dim=256, reduction=2):
        super(HyperFPN, self).__init__()

        self.reduction = reduction
        self.embed_dim = embed_dim
        self.num_levels = len(in_channels_list)

        # FPN 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1) for in_channels in in_channels_list
        ])

        # 上下采样路径
        self.upsample_convs = nn.ModuleList([
            nn.Conv2d(in_channels_list[i], in_channels_list[i - 1], kernel_size=1)
            for i in range(1, self.num_levels)
        ])
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(in_channels_list[i], in_channels_list[i + 1], kernel_size=3, stride=2, padding=1, groups=in_channels_list[i])
            for i in range(self.num_levels - 1)
        ])

        # 超图特征交互
        self.hypergraph_fusion = HypergraphFeedbackFusion(in_channels_list, hg_channels, embed_dim, reduction)

        # SE 注意力模块
        self.se_blocks = nn.ModuleList([SEBlock(in_channels) for in_channels in in_channels_list])

    def forward(self, inputs, hg_feature):
        """
        inputs: 多尺度特征列表 [(B, C_i, H_i, W_i), ...]
        hg_feature: 超图特征 [B, hg_channels, H, W] (可能会被下采样适配)
        """
        assert len(inputs) == self.num_levels, "输入特征图数量不匹配"

        # 1) FPN横向连接
        lateral_feats = [
            lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, inputs)
        ]

        # 2) 超图交互
        lateral_feats = self.hypergraph_fusion(lateral_feats, hg_feature)

        # 3) 上下采样融合
        # 自顶向下
        for i in range(self.num_levels - 1, 0, -1):
            upsampled = F.interpolate(lateral_feats[i], size=lateral_feats[i - 1].shape[-2:], mode='bilinear')
            lateral_feats[i - 1] += self.upsample_convs[i - 1](upsampled)

        # 自下向上
        for i in range(self.num_levels - 1):
            downsampled = self.downsample_convs[i](lateral_feats[i])
            lateral_feats[i + 1] += downsampled

        # 4) 每层可选用 SE 模块增强
        for i in range(self.num_levels):
            lateral_feats[i] = self.se_blocks[i](lateral_feats[i])

        return lateral_feats


