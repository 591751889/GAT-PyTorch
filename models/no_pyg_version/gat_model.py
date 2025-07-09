

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single graph‑attention layer supporting multi‑head attention.*"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 8,
        dropout: float = 0.6,
        concat: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Linear projection W ∈ ℝ^{H × F_in × F_out}
        self.W = nn.Parameter(torch.empty(heads, in_features, out_features))
        # Attention parameters a_src, a_dst ∈ ℝ^{H × F_out}
        self.a_src = nn.Parameter(torch.empty(heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(heads, out_features))

        if bias:
            self.bias = nn.Parameter(
                torch.empty(heads * out_features if concat else out_features)
            )
        else:
            self.register_parameter("bias", None)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    # ---------------------------------------------------------------------
    # utilities
    # ---------------------------------------------------------------------
    def reset_parameters(self) -> None:  # noqa: D401
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute new node embeddings.

        Parameters
        ----------
        x          : [N, F_in] node features
        edge_index : [2, E]  COO edge list (src → dst)
        """
        N = x.size(0)
        src, dst = edge_index  # [E]

        # (1) Linear projection + multi‑head broadcast --------------
        #   h_i^k = W^k x_i   where W^k ∈ ℝ^{F_in×F_out}
        #   result shape: [N, H, F_out]
        h = torch.einsum("nd, hdf -> nhf", x, self.W)

        # Gather source & destination node representations for each edge
        h_src = h[src]  # [E, H, F_out]
        h_dst = h[dst]  # [E, H, F_out]

        # (2) Compute unnormalised attention coefficients e_ij ------
        e = (
            (h_src * self.a_src).sum(dim=-1)  # [E, H]
            + (h_dst * self.a_dst).sum(dim=-1)
        )
        e = self.leaky_relu(e)

        # (3) Softmax over incoming edges for each dst node ---------
        alpha = self._edge_softmax(dst, e)  # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # (4) Message aggregation -----------------------------------
        # out_j = Σ_i α_ij h_i
        msg = alpha.unsqueeze(-1) * h_src  # [E, H, F_out]
        out = torch.zeros(
            (N, self.heads, self.out_features), device=x.device, dtype=x.dtype
        )
        out.index_add_(0, dst, msg)  # aggregate over incoming edges

        # (5) Head concat or average --------------------------------
        if self.concat:
            out = out.reshape(N, self.heads * self.out_features)  # [N, H*F]
        else:  # output layer: average heads
            out = out.mean(dim=1)  # [N, F]

        if self.bias is not None:
            out = out + self.bias
        return out


    def _edge_softmax(self, index: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Edge‑softmax without external deps.

        Parameters
        ----------
        index : [E] destination node indices
        e     : [E, H] unnormalised scores
        """
        E, H = e.shape

        # exp -----------------------------------------------------
        e = e - e.max()  # global stabiliser (adequate for small graphs)
        exp_e = e.exp()

        # denominator: Σ_i exp(e_ij) for each dst j --------------
        N = int(index.max()) + 1
        denom = torch.zeros((N, H), device=e.device, dtype=e.dtype)
        denom.index_add_(0, index, exp_e)  # sum per dst per head

        return exp_e / (denom[index] + 1e-16)


class GAT(nn.Module):
    """Two‑layer (hidden → ELU → output) Graph Attention Network."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        out_dim: int = 7,
        heads: int = 8,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATLayer(
            in_features=in_dim,
            out_features=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        # output layer: 1 head, average instead of concat
        self.gat2 = GATLayer(
            in_features=hidden_dim * heads,
            out_features=out_dim,
            heads=1,
            dropout=dropout,
            concat=False,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=-1)
