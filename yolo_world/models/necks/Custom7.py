import torch
import torch.nn as nn
import torch.nn.functional as F

# =============== Transformer编码器层 ===============
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):  # Reduced dim_feedforward
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = self.norm1(src + self.dropout(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout(src2))
        return src

# =============== 稀疏自注意力模块 ===============
class SparseSelfAttention(nn.Module):
    def __init__(self, in_channels, head_num=4, embed_dim=128, sparsity=0.1):  # Reduced head_num
        super(SparseSelfAttention, self).__init__()
        self.head_num = head_num
        self.head_dim = embed_dim // head_num
        self.sparsity = sparsity

        # Use only one convolution layer for query/key/value projection
        self.qkv = nn.Conv2d(in_channels, embed_dim * 3, kernel_size=1)

        self.fc_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()

        qkv = self.qkv(x).view(B, 3, self.head_num, self.head_dim, H * W).transpose(1, 2)  # [B, head_num, 3, head_dim, H*W]
        query, key, value = qkv.chunk(3, dim=2)  # Split into query, key, value

        attention = torch.matmul(query, key.transpose(-2, -1))  # [B, head_num, H*W, H*W]
        attention = F.softmax(attention, dim=-1)
        attention = attention * (torch.rand_like(attention) < self.sparsity).float()

        out = torch.matmul(attention, value)  # [B, head_num, H*W, head_dim]
        out = out.transpose(2, 3).contiguous().view(B, self.head_num * self.head_dim, H, W)  # [B, embed_dim, H, W]
        out = self.fc_out(out)
        return out

# =============== 超图卷积模块 ===============
class HypergraphConv(nn.Module):
    def __init__(self, in_channels, hyperedge_num=8):  # Reduced hyperedge_num
        super(HypergraphConv, self).__init__()

        self.membership_conv = nn.Conv2d(in_channels, hyperedge_num, kernel_size=1)
        self.node_transform = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W

        membership = self.membership_conv(x).view(B, -1, N)
        membership = F.softmax(membership, dim=1)

        x_flat = x.view(B, C, N)
        hyperedge_features = torch.bmm(x_flat, membership.transpose(1, 2))
        x_updated = torch.bmm(hyperedge_features, membership)
        x_transformed = self.node_transform(x_flat)

        out = self.relu(x_transformed + x_updated)
        out = out.view(B, C, H, W)
        return out

# =============== Transformer整体结构 (单尺度) ===============
class SingleScaleTransformerWithHypergraph_LIGHT(nn.Module):
    def __init__(self, in_channels, hyperedge_num=8, head_num=4, nhead=4, num_layers=4, embed_dim=128, sparsity=0.1):
        super(SingleScaleTransformerWithHypergraph_LIGHT, self).__init__()

        self.sparse_attention = SparseSelfAttention(in_channels, head_num, embed_dim, sparsity)
        self.hypergraph_conv = HypergraphConv(in_channels, hyperedge_num)
        self.fusion_projection = nn.Conv2d(2 * in_channels, embed_dim, kernel_size=1)

        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, nhead) for _ in range(num_layers)])
        self.output_conv = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        attention_feats = self.sparse_attention(x)
        hypergraph_feats = self.hypergraph_conv(x)

        max_H = max(attention_feats.shape[2], hypergraph_feats.shape[2])
        max_W = max(attention_feats.shape[3], hypergraph_feats.shape[3])
        attention_feats = F.interpolate(attention_feats, size=(max_H, max_W), mode='bilinear')
        hypergraph_feats = F.interpolate(hypergraph_feats, size=(max_H, max_W), mode='bilinear')

        fused_feats = torch.cat([attention_feats, hypergraph_feats], dim=1)
        fused_feats = self.fusion_projection(fused_feats)

        B, C, H, W = fused_feats.size()
        fused_feats = fused_feats.view(B, C, H * W).permute(2, 0, 1)

        for layer in self.encoder_layers:
            fused_feats = layer(fused_feats)

        fused_feats = fused_feats.permute(1, 2, 0).view(B, C, H, W)
        out = self.output_conv(fused_feats)
        return out

