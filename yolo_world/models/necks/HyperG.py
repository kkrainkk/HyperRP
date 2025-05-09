import torch
import torch.nn as nn
import torch.nn.functional as F


# =============== MLP模块（保持原样） ===============
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.1):
        super(MLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Permute(nn.Module):
    """维度置换模块"""

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class LearnableHypergraphConstructor(nn.Module):
    def __init__(self, in_channels, text_channels=512, num_hyperedges=16,
                 embed_dim=256, num_heads=4):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.embed_dim = embed_dim

        # QKV生成器（保持通道兼容性）
        self.qkv_gen = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim * 3, 3, padding=1),
            nn.GroupNorm(4, embed_dim * 3)
        )

        # 注意力层（移除冗余参数）
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 文本特征适配器（维度安全）
        self.text_adapter = nn.Sequential(
            nn.Linear(text_channels, embed_dim),
            Permute((0, 2, 1)),
            nn.AdaptiveAvgPool1d(4),  # 固定输出4个窗口
            Permute((0, 2, 1))
        )

        # 超边预测器
        self.edge_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_hyperedges)
        )

    def _hierarchical_attention(self, q, k, v, text_feat):
        B, N, D = q.shape
        assert N % 64 == 0, "N必须能被64整除"
        window_size = 64
        num_windows = N // window_size

        # 窗口划分（安全重塑）
        q_win = q.reshape(B * num_windows, window_size, D).contiguous()
        k_win = k.reshape(B * num_windows, window_size, D).contiguous()
        v_win = v.reshape(B * num_windows, window_size, D).contiguous()

        # 局部注意力
        local_attn, _ = self.self_attn(q_win, k_win, v_win)
        local_attn = local_attn.reshape(B, num_windows, window_size, D)

        # 全局注意力
        global_q = local_attn.mean(dim=2).reshape(B, num_windows, D)
        global_attn, _ = self.self_attn(global_q, text_feat, text_feat)

        # 特征扩展
        global_attn = global_attn.unsqueeze(2).expand(-1, -1, window_size, -1)
        return local_attn.reshape(B, N, D), global_attn.reshape(B, N, D)

    def forward(self, feature_map, text_features):
        B, C, H, W = feature_map.shape
        N = H * W

        # QKV生成
        qkv = self.qkv_gen(feature_map)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q_seq = q.reshape(B, self.embed_dim, N).permute(0, 2, 1).contiguous()
        k_seq = k.reshape(B, self.embed_dim, N).permute(0, 2, 1).contiguous()
        v_seq = v.reshape(B, self.embed_dim, N).permute(0, 2, 1).contiguous()

        # 文本处理
        text_feat = self.text_adapter(text_features)

        # 分层注意力
        local_attn, global_attn = self._hierarchical_attention(q_seq, k_seq, v_seq, text_feat)

        # 特征融合
        combined = torch.cat([local_attn, global_attn], dim=-1)
        hyperedge_scores = self.edge_predictor(combined)
        return torch.sigmoid(hyperedge_scores)


class HyperedgeGuidedNodeAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        # 节点作为Q，超边作为K/V
        self.node_q = nn.Linear(in_channels, in_channels)
        self.hyper_kv = nn.Linear(in_channels, in_channels * 2)
        self.scale = in_channels  ** -0.5

    def forward(self, node_feats, hyper_feats):
        """
        node_feats: [B, N, C] (需要被更新的节点特征)
        hyper_feats: [B, E, C] (作为知识库的超边特征)
        输出: [B, N, C] (融合了跨超边信息的新节点特征)
        """
        Q = self.node_q(node_feats)  # [B, N, C]
        K, V = self.hyper_kv(hyper_feats).chunk(2, dim=-1)  # [B, E, C] each

        # 计算注意力权重
        attn = torch.einsum('bnc,bec->bne', Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)  # [B, N, E]

        # 聚合超边信息到节点
        return torch.einsum('bne,bec->bnc', attn, V)


class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels, num_hyperedges=16, dropout_rate=0.1):
        super().__init__()
        self.num_hyperedges = num_hyperedges

        # 超边自注意力模块（恢复）
        self.hyper_self_attn = HypergraphTransformerConv(in_channels)

        # 节点跨超边注意力模块
        self.node_cross_attn = HyperedgeGuidedNodeAttention(in_channels)

        # 归一化层
        self.norm_hyper = nn.LayerNorm(in_channels)
        self.norm_node_pre = nn.LayerNorm(in_channels)
        self.norm_fusion = nn.LayerNorm(in_channels * 2)

        # 特征变换
        self.hyper_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU()
        )

        # 动态门控
        self.gate = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.LayerNorm(in_channels),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, feature_map, hyperedge_scores):
        B, C, H, W = feature_map.shape
        N = H * W

        # ---- 节点特征初始化 ----
        node_feats = feature_map.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        node_feats = self.norm_node_pre(node_feats)

        # ---- 超边特征生成与更新 ----
        # 步骤1: 聚合节点特征生成初始超边特征
        hyper_feats = torch.bmm(hyperedge_scores.transpose(1, 2), node_feats)  # [B, E, C]

        # 步骤2: 超边自注意力更新（恢复丢失的部分）
        hyper_feats = self.hyper_self_attn(hyper_feats)  # ← 关键恢复点
        hyper_feats = self.norm_hyper(hyper_feats)
        hyper_feats = self.hyper_proj(hyper_feats)

        # ---- 双路信息传递 ----
        # 路径A: 常规超边内传播
        intra_feats = torch.bmm(hyperedge_scores, hyper_feats)  # [B, N, C]

        # 路径B: 跨超边注意力传播
        cross_feats = self.node_cross_attn(node_feats, hyper_feats)  # [B, N, C]

        # ---- 特征融合 ----
        combined = torch.cat([intra_feats, cross_feats], dim=-1)  # [B, N, 2C]
        combined = self.norm_fusion(combined)

        gate = self.gate(combined)  # [B, N, C]
        fused_feats = gate * intra_feats + (1 - gate) * cross_feats

        # ---- 更新节点特征 ----
        updated_feats = node_feats + self.dropout(fused_feats)

        return self.node_proj(updated_feats).permute(0, 2, 1).view(B, C, H, W)



# =============== 保持原有的Transformer层 ===============
class HypergraphTransformerConv(nn.Module):
    def __init__(self, in_channels, num_hyperedges=16, num_heads=4, dropout_rate=0.1):
        super(HypergraphTransformerConv, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels),
            nn.Dropout(dropout_rate)
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, hyperedge_features):
        # 自注意力更新
        attn_output, _ = self.multihead_attn(
            hyperedge_features, hyperedge_features, hyperedge_features
        )
        hyperedge_features = self.norm(hyperedge_features + attn_output)

        # 前馈网络
        return self.norm(hyperedge_features + self.ffn(hyperedge_features))




# =============== 改进后的超图学习模块 ===============
class HypergraphLearningModule(nn.Module):
    def __init__(self, in_channels, text_channels=512, num_hyperedges=16, embed_dim=256, dropout_rate=0.1):
        super(HypergraphLearningModule, self).__init__()
        # 超边构造器（保持原有逻辑）
        self.hypergraph_constructor = LearnableHypergraphConstructor(
            in_channels, text_channels, num_hyperedges, embed_dim
        )
        # 使用改进后的卷积层
        self.hypergraph_conv = HypergraphConvolution(in_channels, num_hyperedges, dropout_rate)

    def forward(self, feature_map, text_features):
        # 生成原始超边关联矩阵 H
        hyperedge_scores = self.hypergraph_constructor(feature_map, text_features)  # [B, N, E]

        # 仅传递原始H矩阵给卷积层
        return self.hypergraph_conv(feature_map, hyperedge_scores)


