"""
Light-weight Graph Isomorphism Network (GIN) implementation
----------------------------------------------------------
Paper: Xu et al., “How Powerful are Graph Neural Networks?”, ICLR 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class GIN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 7,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """GIN 多层架构

        Args
        ----
        in_dim     : 输入特征维度 (= dataset.num_node_features)
        hidden_dim : 每层隐藏维度
        out_dim    : 类别数 (= dataset.num_classes)
        num_layers : GINConv 层数
        dropout    : Dropout 概率（应用于特征）
        """
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linear_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_out(x)
        return F.log_softmax(x, dim=-1)
