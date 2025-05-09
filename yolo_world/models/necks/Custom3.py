import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # 转置 G 得到 [B, num_edges, N]
        hyperedge_features = torch.bmm(G.transpose(1, 2), x)  # [B, num_edges, out_ft]

        # Step 2: 超边 -> 节点
        x = torch.bmm(G, hyperedge_features)  # [B, N, out_ft]

        return x


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_class)
        self.norm1 = nn.LayerNorm(n_hid)
        self.norm2 = nn.LayerNorm(n_hid)

    def forward(self, x, G):
        x_regin = x
        x = F.relu(self.norm1(self.hgc1(x, G)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.norm2(self.hgc2(x, G)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x, G)
        return x + x_regin


class HyperEdgeGenerator(nn.Module):
    def __init__(self, in_channels, num_edges, hidden_dim=128):
        super(HyperEdgeGenerator, self).__init__()
        self.num_edges = num_edges  # 超边的数量
        self.hidden_dim = hidden_dim

        # 超边生成网络
        self.edge_net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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


class CustomMultiScaleModule(nn.Module):
    def __init__(self, in_channels, out_channels, H=None, W=None, num_edges=64):
        super(CustomMultiScaleModule, self).__init__()
        self.hyperconv = HGNN(in_ch=in_channels, n_class=out_channels, n_hid=256)
        self.H = H
        self.W = W
        self.num_edges = num_edges

        # 超边生成器
        self.hyperedge_generator = HyperEdgeGenerator(in_channels, num_edges)

    def forward(self, x):
        device = x.device
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
