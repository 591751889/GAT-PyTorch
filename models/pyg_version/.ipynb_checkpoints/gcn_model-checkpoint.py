"""
Light-weight Graph Convolutional Network (GCN) implementation
-----------------------------------------------------------
Paper: Kipf & Welling, “Semi-Supervised Classification with Graph Convolutional Networks”, ICLR 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        out_dim: int = 7,
        dropout: float = 0.5,
    ):
        """GCN 双层架构

        Args
        ----
        in_dim     : 输入特征维度 (= dataset.num_node_features)
        hidden_dim : 隐层维度
        out_dim    : 类别数 (= dataset.num_classes)
        dropout    : Dropout 概率（应用于特征）
        """
        super().__init__()
        self.dropout = dropout

        # 第 1 层
        self.conv1 = GCNConv(in_channels=in_dim, out_channels=hidden_dim)

        # 第 2 层（输出）
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=out_dim)

    def forward(self, x, edge_index):
        # 输入特征 Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 卷积 + 激活
        x = F.relu(self.conv1(x, edge_index))
        # 再次 Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 输出层
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)
