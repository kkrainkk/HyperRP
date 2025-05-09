import torch
import torch.nn as nn
import torch.nn.functional as F

# =============== 文本特征对齐模块 ===============
class TextFeatureAligner(nn.Module):
    def __init__(self, text_channels=512, embed_dim=256, dropout_rate=0.1):
        super(TextFeatureAligner, self).__init__()
        self.text_projection = nn.Linear(text_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text_features):
        text_features = self.text_projection(text_features)
        text_features = self.norm(text_features)
        text_features = self.dropout(text_features)
        return text_features


# =============== Cross Attention With GroupNorm ===============
class CrossAttentionWithTextUpdate(nn.Module):
    def __init__(self, in_channels, text_channels=512, embed_dim=256, num_heads=4, dropout_rate=0.1):
        super(CrossAttentionWithTextUpdate, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.text_aligner = TextFeatureAligner(text_channels, embed_dim, dropout_rate)

        self.q_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.k_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # 输出变换
        self.fc_out = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels, kernel_size=1),
            nn.GroupNorm(4, in_channels)  # 改成GroupNorm
        )

        self.text_restore = nn.Linear(embed_dim, text_channels)

    def forward(self, feature_map, text_features, target_size):
        B, C, H, W = feature_map.shape
        seq_len = text_features.shape[1]

        # 文本特征变换
        text_features = self.text_aligner(text_features)
        Q = self.q_proj(text_features).view(B, self.num_heads, seq_len, self.head_dim)

        # 图像特征变换
        K = self.k_proj(feature_map).flatten(2).permute(0, 2, 1)
        V = self.v_proj(feature_map).flatten(2).permute(0, 2, 1)

        K = K.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Cross-Attention
        attention = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(attention, dim=-1)

        # 融合文本特征
        updated_text_features = torch.matmul(attention, V)
        updated_text_features = updated_text_features.permute(0, 2, 1, 3).contiguous().view(B, seq_len, self.embed_dim)

        updated_text_features = self.text_restore(updated_text_features)

        # 每次更新完文本特征后加LayerNorm
        updated_text_features = F.layer_norm(updated_text_features, [updated_text_features.shape[-1]])

        # 计算视觉特征输出，并插值到目标尺寸
        out = self.fc_out(V.permute(0, 2, 1, 3).contiguous().view(B, self.embed_dim, H, W))
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)

        return out, updated_text_features


# =============== Feature Fusion with Updated Text Features + GroupNorm ===============
class FeatureFusionWithTextUpdateModule(nn.Module):
    def __init__(self, in_channels_list, text_channels=512, embed_dim=256, dropout_rate=0.1):
        super(FeatureFusionWithTextUpdateModule, self).__init__()

        self.attention_modules = nn.ModuleList([
            CrossAttentionWithTextUpdate(in_channels, text_channels, embed_dim, dropout_rate=dropout_rate)
            for in_channels in in_channels_list
        ])

        total_channels = sum(in_channels_list)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, in_channels_list[0], kernel_size=1),
            nn.GroupNorm(4, in_channels_list[0])  # 这里关键改成GroupNorm
        )

        self.text_weight_layer = nn.Linear(len(in_channels_list), len(in_channels_list))
        self.text_proj = nn.Linear(text_channels, text_channels)

    def forward(self, feature_maps, text_features):
        target_size = (feature_maps[0].shape[2], feature_maps[0].shape[3])

        attended_features = []
        updated_text_features_list = []

        for attn_module, feature_map in zip(self.attention_modules, feature_maps):
            attended_feature, updated_text_features = attn_module(feature_map, text_features, target_size)
            attended_features.append(attended_feature)
            updated_text_features_list.append(updated_text_features)

        # 拼接多尺度特征图
        fused_feats = torch.cat(attended_features, dim=1)

        # 1x1卷积压缩+GroupNorm
        fused_feats = self.fusion_conv(fused_feats)

        # 逐尺度堆叠文本特征 [B, seq_len, text_channels, num_scales]
        updated_text_features = torch.stack(updated_text_features_list, dim=-1)

        # 计算每个batch的尺度加权权重
        B, T, C, S = updated_text_features.shape
        weights = self.text_weight_layer(torch.ones((B, S), device=updated_text_features.device))
        weights = F.softmax(weights, dim=-1).unsqueeze(1).unsqueeze(1)

        # 加权融合文本特征
        updated_text_features = torch.sum(updated_text_features * weights, dim=-1)

        # 最终线性投影+LayerNorm
        updated_text_features = self.text_proj(updated_text_features)
        updated_text_features = F.layer_norm(updated_text_features, [updated_text_features.shape[-1]])

        return fused_feats, updated_text_features












