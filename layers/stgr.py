"""Spatial graph modules for spatio-temporal reprogramming."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SpatialGraphReprogramming(nn.Module):
    """Graph attention over patch tokens prior to reprogramming."""

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        heads: int = 4,
        dropout: float = 0.1,
        node_feature_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or in_channels

        self.num_nodes = num_nodes
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.node_feature_proj: Optional[nn.Linear] = None
        if node_feature_dim:
            self.node_feature_proj = nn.Linear(node_feature_dim, hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply graph-aware attention.

        Parameters
        ----------
        x: torch.Tensor
            Tensor shaped as ``(batch, nodes, patches, channels)``.
        adjacency: torch.Tensor
            Binary adjacency matrix of shape ``(nodes, nodes)``.
        """

        batch, nodes, patches, channels = x.shape
        if nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes but received {nodes} in spatial encoder."
            )

        adjacency = adjacency.to(x.device)
        attention_mask = self._build_attention_mask(adjacency)

        x = self.input_projection(x)

        if node_features is not None and self.node_feature_proj is not None:
            node_features = node_features.to(x.device)
            if node_features.dim() == 2:
                node_features = node_features.unsqueeze(0)
            if node_features.size(1) != nodes:
                raise ValueError(
                    f"Node feature dimension {node_features.size(1)} does not match number of nodes {nodes}."
                )
            feature_bias = self.node_feature_proj(node_features)
            if feature_bias.size(0) == 1:
                feature_bias = feature_bias.expand(batch, -1, -1)
            x = x + feature_bias.unsqueeze(2)

        x = x.view(batch * patches, nodes, -1)

        attended, _ = self.attention(
            x,
            x,
            x,
            attn_mask=attention_mask,
            need_weights=False,
        )
        attended = self.output_projection(attended)
        attended = attended.view(batch, patches, nodes, -1).permute(0, 2, 1, 3)
        return self.activation(self.dropout(attended))

    def _build_attention_mask(self, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.float().clone()
        adjacency.fill_diagonal_(1.0)
        if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError("Adjacency must be a square matrix")
        mask = torch.where(adjacency > 0, torch.zeros_like(adjacency), torch.full_like(adjacency, float("-inf")))
        return mask

