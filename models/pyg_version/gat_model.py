"""
Light-weight Graph Attention Network (GAT) implementation
========================================================
Paper: Velickovic et al., “Graph Attention Networks”, ICLR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        out_dim: int = 7,
        heads: int = 8,
        dropout: float = 0.6,
    ):
        """
        Args
        ----
        in_dim     : 输入特征维度 (= dataset.num_node_features)
        hidden_dim : 隐层每头输出维度
        out_dim    : 类别数 (= dataset.num_classes)
        heads      : 多头注意力数量
        dropout    : Dropout 概率（应用于特征和注意力）
        """
        super().__init__()
        self.dropout = dropout

        # 第 1 层：拼接头
        self.conv1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
        )

        # 第 2 层（输出）：平均头
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)
