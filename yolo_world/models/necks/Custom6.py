import torch
import torch.nn as nn
import torch.nn.functional as F

# =============== Transformer编码器层 ===============
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        """
        src: shape = [seq_len, batch_size, d_model]
        """
        # 自注意力
        src2 = self.self_attn(src, src, src)[0]  # [seq_len, batch_size, d_model]
        src = self.norm1(src + self.dropout(src2))  # 残差连接 + 归一化

        # 前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout(src2))  # 残差连接 + 归一化

        return src


# =============== 单尺度自注意力模块 ===============
class SingleScaleSelfAttention(nn.Module):
    def __init__(self, in_channels, head_num=8, embed_dim=128):
        super(SingleScaleSelfAttention, self).__init__()
        self.head_num = head_num
        self.head_dim = embed_dim // head_num  # Adjust embed_dim for the number of heads

        self.query = nn.Conv2d(in_channels, embed_dim, kernel_size=1)  # Match embed_dim
        self.key = nn.Conv2d(in_channels, embed_dim, kernel_size=1)  # Match embed_dim
        self.value = nn.Conv2d(in_channels, embed_dim, kernel_size=1)  # Match embed_dim

        self.fc_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)  # Output the same number of channels as input

    def forward(self, x):
        B, C, H, W = x.size()

        # Ensure the query, key, value have the correct dimensions for self-attention
        query = self.query(x).view(B, self.head_num, self.head_dim, H * W).transpose(2, 3)  # [B, head_num, H*W, head_dim]
        key = self.key(x).view(B, self.head_num, self.head_dim, H * W)  # [B, head_num, head_dim, H*W]
        value = self.value(x).view(B, self.head_num, self.head_dim, H * W).transpose(2, 3)  # [B, head_num, H*W, head_dim]

        attention = torch.matmul(query, key)  # [B, head_num, H*W, H*W]
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, value)  # [B, head_num, H*W, head_dim]
        out = out.transpose(2, 3).contiguous()  # [B, head_num, head_dim, H*W]
        out = out.view(B, self.head_num * self.head_dim, H, W)  # [B, embed_dim, H, W]

        out = self.fc_out(out)  # Output the final features
        return out


# =============== 单尺度超图卷积模块 ===============
class HypergraphConv(nn.Module):
    def __init__(self, in_channels, hyperedge_num=16):
        super(HypergraphConv, self).__init__()
        self.hyperedge_num = hyperedge_num

        self.membership_conv = nn.Conv2d(in_channels, hyperedge_num, kernel_size=1)
        self.hyperedge_conv = nn.Conv1d(hyperedge_num, hyperedge_num, kernel_size=1)
        self.node_transform = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W  # 节点数

        membership = self.membership_conv(x)  # 形状 [B, hyperedge_num, H, W]
        membership = membership.view(B, self.hyperedge_num, N)  # [B, hyperedge_num, N]
        membership = F.softmax(membership, dim=1)  # 在超边维度归一化

        x_flat = x.view(B, C, N)  # [B, C, N]
        hyperedge_features = torch.bmm(x_flat, membership.transpose(1, 2))
        hyperedge_features = hyperedge_features.transpose(1, 2)
        hyperedge_features = self.relu(self.hyperedge_conv(hyperedge_features))
        hyperedge_features = hyperedge_features.transpose(1, 2)
        x_updated = torch.bmm(hyperedge_features, membership)
        x_transformed = self.node_transform(x_flat)
        out = self.relu(x_transformed + x_updated)
        out = out.view(B, C, H, W)
        return out


# =============== Transformer整体结构 (单尺度) ===============
class SingleScaleTransformerWithHypergraph(nn.Module):
    def __init__(self, in_channels, hyperedge_num=8, head_num=4, nhead=4, num_layers=4, embed_dim=128):
        super(SingleScaleTransformerWithHypergraph, self).__init__()
        self.single_scale_attention = SingleScaleSelfAttention(in_channels, head_num, embed_dim)
        self.single_scale_hypergraph_conv = HypergraphConv(in_channels, hyperedge_num)

        # 新增线性投影层：将融合后的 2*in_channels 映射到 embed_dim
        self.fusion_projection = nn.Conv2d(2 * in_channels, embed_dim, kernel_size=1)

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, nhead) for _ in range(num_layers)]
        )
        self.output_conv = nn.Conv2d(embed_dim, in_channels, kernel_size=1)  # 输出恢复为原始通道数

    def forward(self, x):
        # 自注意力计算
        attention_feats = self.single_scale_attention(x)
        # 超图卷积
        hypergraph_feats = self.single_scale_hypergraph_conv(x)

        # 上采样/下采样至相同尺寸
        max_H = max(attention_feats.shape[2], hypergraph_feats.shape[2])
        max_W = max(attention_feats.shape[3], hypergraph_feats.shape[3])
        attention_feats = F.interpolate(attention_feats, size=(max_H, max_W), mode='bilinear')
        hypergraph_feats = F.interpolate(hypergraph_feats, size=(max_H, max_W), mode='bilinear')

        # 拼接特征并投影到embed_dim
        fused_feats = torch.cat([attention_feats, hypergraph_feats], dim=1)
        fused_feats = self.fusion_projection(fused_feats)  # [B, embed_dim, H, W]

        # 转换为Transformer输入格式 [seq_len, batch_size, embed_dim]
        B, C, H, W = fused_feats.size()
        fused_feats = fused_feats.view(B, C, H * W).permute(2, 0, 1)  # [H*W, B, C]

        # 通过Transformer编码层
        for layer in self.encoder_layers:
            fused_feats = layer(fused_feats)

        # 转换回空间特征图
        fused_feats = fused_feats.permute(1, 2, 0).view(B, C, H, W)
        out = self.output_conv(fused_feats)  # 恢复原始通道数
        return out




