"""Window encoder: depthwise-separable CNN + physical feature MLP (branch-selectable)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1D convolution: depthwise -> pointwise."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ConvBlock(nn.Module):
    """Conv1d -> GroupNorm -> SiLU, with optional residual connection."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
    ):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, stride)
        num_groups = min(out_channels // 4, 32)
        if num_groups < 1:
            num_groups = 1
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()
        self.use_residual = (in_channels == out_channels) and (stride == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.use_residual else None
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if residual is not None:
            x = x + residual
        return x


class CNNBranch(nn.Module):
    """4 depthwise-separable conv blocks -> GlobalAvgPool1d + GlobalMaxPool1d -> Linear(256, 128)."""

    def __init__(self, window_length: int = 8192, hidden_dim: int = 128):
        super().__init__()
        kernels = [15, 11, 7, 5]
        channels = [32, 64, 96, 128]
        stride = 4

        blocks = []
        in_ch = 1  # single-channel raw signal
        for k, out_ch in zip(kernels, channels):
            blocks.append(ConvBlock(in_ch, out_ch, k, stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Linear(256, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*K, 1, 8192)
        x = self.blocks(x)  # (B*K, 128, L_reduced)
        avg = x.mean(dim=-1)  # (B*K, 128)
        max_ = x.max(dim=-1).values  # (B*K, 128)
        pooled = torch.cat([avg, max_], dim=-1)  # (B*K, 256)
        return self.proj(pooled)  # (B*K, 128)


class FeatureMLP(nn.Module):
    """Linear(58, 128) -> LayerNorm -> SiLU -> Dropout(0.1) -> Linear(128, 64)."""

    def __init__(self, feature_dim: int = 58, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*K, 58)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)  # (B*K, 64)


class RobustNormalize(nn.Module):
    """Per-window robust standardization: (x - median) / (IQR / 1.349)."""

    def __init__(self, window_length: int = 8192):
        super().__init__()
        self.window_length = window_length

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # windows: (B, K, L)
        B, K, L = windows.shape
        if L != self.window_length:
            raise ValueError(f"Expected window length {self.window_length}, got {L}")
        windows_f = windows.reshape(B * K, L).float()
        median = windows_f.median(dim=-1, keepdim=True).values
        q75 = windows_f.quantile(0.75, dim=-1, keepdim=True)
        q25 = windows_f.quantile(0.25, dim=-1, keepdim=True)
        iqr = q75 - q25
        scale = iqr / 1.349
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
        return ((windows_f - median) / scale).reshape(B, K, L)


class WindowEncoder(nn.Module):
    """Branch-selectable window encoder, all variants output 128-dim per window.

    branch="dual":    CNN + feature MLP fusion (default, full model)
    branch="cnn":     raw-signal CNN branch only
    branch="feature": physical-feature MLP only
    """

    def __init__(
        self,
        window_length: int = 8192,
        feature_dim: int = 58,
        hidden_dim: int = 128,
        branch: Literal["dual", "cnn", "feature"] = "dual",
        apply_robust_normalize: bool = True,
    ):
        super().__init__()
        self.branch = branch
        self.hidden_dim = hidden_dim
        self.apply_robust_normalize = apply_robust_normalize
        self.window_length = window_length
        self.cnn = CNNBranch(window_length, hidden_dim)
        self.mlp = FeatureMLP(feature_dim, hidden_dim)

        if branch == "dual":
            self.fusion_norm = nn.LayerNorm(hidden_dim + 64)
            self.fusion = nn.Linear(hidden_dim + 64, hidden_dim)
            self.out_norm = nn.LayerNorm(hidden_dim)
        elif branch == "cnn":
            self.out_norm = nn.LayerNorm(hidden_dim)
        elif branch == "feature":
            self.proj = nn.Linear(64, hidden_dim)
            self.out_norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unknown encoder branch: {branch}")

    def _robust_normalize(self, windows: torch.Tensor) -> torch.Tensor:
        """Per-window robust standardization: (x - median) / (IQR / 1.349)."""
        # windows: (B*K, 8192)
        windows_f = windows.float()  # ensure float for quantile
        median = windows_f.median(dim=-1, keepdim=True).values
        q75 = windows_f.quantile(0.75, dim=-1, keepdim=True)
        q25 = windows_f.quantile(0.25, dim=-1, keepdim=True)
        iqr = q75 - q25
        scale = iqr / 1.349
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
        return (windows_f - median) / scale

    def forward(
        self, windows: torch.Tensor, features: torch.Tensor,
    ) -> torch.Tensor:
        # windows: (B, K, 8192)
        # features: (B, K, 58)
        B, K, L = windows.shape
        if L != self.window_length:
            raise ValueError(f"Expected window length {self.window_length}, got {L}")

        windows_flat = windows.reshape(B * K, 1, L)  # (B*K, 1, 8192)
        features_flat = features.reshape(B * K, -1)  # (B*K, 58)

        windows_flat_sq = windows_flat.squeeze(1)  # (B*K, 8192)
        if self.apply_robust_normalize:
            windows_norm = self._robust_normalize(windows_flat_sq).unsqueeze(1)
        else:
            windows_norm = windows_flat

        if self.branch == "dual":
            cnn_out = self.cnn(windows_norm)  # (B*K, 128)
            mlp_out = self.mlp(features_flat)  # (B*K, 64)
            fused = torch.cat([cnn_out, mlp_out], dim=-1)  # (B*K, 192)
            fused = self.fusion_norm(fused)
            fused = self.fusion(fused)
            out = self.out_norm(fused)  # (B*K, 128)
        elif self.branch == "cnn":
            cnn_out = self.cnn(windows_norm)  # (B*K, 128)
            out = self.out_norm(cnn_out)
        else:  # feature
            mlp_out = self.mlp(features_flat)  # (B*K, 64)
            out = self.out_norm(self.proj(mlp_out))  # (B*K, 128)

        return out.reshape(B, K, self.hidden_dim)
