

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F




def _augment_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return edge_index augmented with reverse edges and self-loops."""
    src, dst = edge_index
    # add reverse direction
    edge_index = torch.cat([edge_index, torch.stack([dst, src], dim=0)], dim=1)
    # add self-loops
    loops = torch.arange(num_nodes, device=edge_index.device)
    edge_index = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)
    return edge_index



class GCNLayer(nn.Module):
    """Graph convolution layer using symmetric normalisation (Â=D^-1/2 Â D^-1/2)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401
        N = x.size(0)
        edge_index = _augment_edge_index(edge_index, N)
        src, dst = edge_index  # both [E]

        # degrees d_i
        deg = torch.bincount(src, minlength=N).float().clamp(min=1)
        norm = (deg[src].rsqrt() * deg[dst].rsqrt()).unsqueeze(-1)  # 1/√(d_i d_j)

        h = self.lin(x)            # [N, F_out]
        msg = h[src] * norm        # message on each edge
        out = torch.zeros_like(h)
        out.index_add_(0, dst, msg)  # aggregate
        return out



class GCN(nn.Module):
    """Two-layer GCN with ReLU, dropout, and log-softmax output."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 16,
        out_dim: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        return F.log_softmax(x, dim=-1)
