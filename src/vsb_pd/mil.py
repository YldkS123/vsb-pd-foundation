"""MIL aggregation and per-phase classifier for the VSB pipeline."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class MILAggregator(nn.Module):
    """Aggregate K window representations into a single per-phase vector.

    Supported aggregations:
    - mean: simple average over windows
    - max: element-wise maximum over windows
    - attention: softmax-weighted sum
    - gated_attention: sigmoid-gated attention-weighted sum
    """

    def __init__(
        self,
        aggregation: Literal["mean", "max", "attention", "gated_attention"] = "gated_attention",
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.aggregation = aggregation
        self.hidden_dim = hidden_dim

        if aggregation == "attention":
            self.attn = nn.Linear(hidden_dim, 1)
        elif aggregation == "gated_attention":
            self.attn = nn.Linear(hidden_dim, 1)
            self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, hidden_dim)
        B, K, D = x.shape

        if self.aggregation == "mean":
            return x.mean(dim=1)  # (B, D)

        elif self.aggregation == "max":
            return x.max(dim=1).values  # (B, D)

        elif self.aggregation == "attention":
            # a = softmax(Linear(D, 1)(x) / sqrt(D))
            scores = self.attn(x).squeeze(-1)  # (B, K)
            alpha = F.softmax(scores / math.sqrt(D), dim=-1)  # (B, K)
            alpha = alpha.unsqueeze(-1)  # (B, K, 1)
            return (alpha * x).sum(dim=1)  # (B, D)

        elif self.aggregation == "gated_attention":
            # a = softmax(Linear(D, 1)(x) / sqrt(D))
            scores = self.attn(x).squeeze(-1)  # (B, K)
            alpha = F.softmax(scores / math.sqrt(D), dim=-1)
            # g = sigmoid(Linear(D, 1)(x))
            g = torch.sigmoid(self.gate(x).squeeze(-1))  # (B, K)
            weights = alpha * g  # (B, K)
            weights = weights.unsqueeze(-1)  # (B, K, 1)
            return (weights * x).sum(dim=1)  # (B, D)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class PhaseClassifier(nn.Module):
    """Per-phase binary classifier: Linear(128,64) -> ReLU -> Dropout(0.3) -> Linear(64,1)."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, hidden_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # (B, 1) -- raw logit
