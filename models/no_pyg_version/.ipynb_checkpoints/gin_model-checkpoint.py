

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F




def _augment_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index
    edge_index = torch.cat([edge_index, torch.stack([dst, src], dim=0)], dim=1)
    loops = torch.arange(num_nodes, device=edge_index.device)
    edge_index = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)
    return edge_index



class MLP(nn.Module):
    """Two‑layer MLP with BatchNorm and ReLU."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)

class GINLayer(nn.Module):
    """Message‑aggregation layer following the GIN formulation."""

    def __init__(self, in_features: int, out_features: int, eps: float = 0.0, train_eps: bool = True) -> None:
        super().__init__()
        self.mlp = MLP(in_features, out_features)
        self.eps = nn.Parameter(torch.tensor(eps)) if train_eps else torch.tensor(eps)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401
        N = x.size(0)
        edge_index = _augment_edge_index(edge_index, N)
        src, dst = edge_index

        aggr = torch.zeros_like(x)          # Σ_{u∈N(v)} h_u
        aggr.index_add_(0, dst, x[src])
        out = self.mlp((1 + self.eps) * x + aggr)
        return out



class GIN(nn.Module):
    """Three‑layer GIN with ReLU, dropout, and log‑softmax output."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        out_dim: int = 7,
        num_layers: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        for i in range(num_layers - 1):
            layers.append(GINLayer(dims[i], dims[i + 1]))
        self.gin_layers = nn.ModuleList(layers)
        self.readout = GINLayer(hidden_dim, out_dim, train_eps=False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401
        for gin in self.gin_layers:
            x = F.relu(gin(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.readout(x, edge_index)
        return F.log_softmax(x, dim=-1)
