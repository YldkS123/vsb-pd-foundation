# -*- coding: utf-8 -*-
"""DL baseline window encoders for the VSB pipeline.

All encoders expose the same interface as WindowEncoder: forward(windows,
features) with windows shaped (B, K, 8192) and emit (B, K, 128) embeddings,
so they drop into the shared MIL/phase-interaction pipeline. Implementations
are deliberately compact (a few hundred k params) so the comparison isolates
encoder expressiveness rather than raw capacity:

  - SimpleCNN      : existing CNNBranch (depthwise-separable 4-block CNN)
  - ResNet1D       : residual 1D blocks with downsampling stem
  - TCN            : dilated causal temporal-conv residual blocks
  - InceptionTime  : multi-kernel inception modules (kernels 11/21/41)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import RobustNormalize, WindowEncoder


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(channels // 4, 32), channels)


class _PoolProj(nn.Module):
    """Global avg+max pooling followed by a projection to hidden_dim."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=-1)
        mx = x.max(dim=-1).values
        return self.proj(torch.cat([avg, mx], dim=-1))


class _ResBlock1d(nn.Module):
    """Conv -> GN -> SiLU -> Conv -> GN + optional 1x1 shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.c1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.gn1 = _gn(out_ch)
        self.c2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, bias=False)
        self.gn2 = _gn(out_ch)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False), _gn(out_ch))
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(x)
        h = F.silu(self.gn1(h))
        h = self.c2(h)
        h = self.gn2(h)
        return F.silu(h + self.shortcut(x))


class ResNet1DEncoder(nn.Module):
    """Residual 1D encoder: stem(7, s=2) -> 2x[32,64,128] blocks -> pool+proj."""

    def __init__(self, window_length: int = 8192, hidden_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False), _gn(32), nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            _ResBlock1d(32, 32),
            _ResBlock1d(32, 64, stride=2),
            _ResBlock1d(64, 64),
            _ResBlock1d(64, 128, stride=2),
            _ResBlock1d(128, 128),
        )
        self.pool = _PoolProj(128, hidden_dim)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*K, 1, 8192) -> (B*K, 128)
        return self.pool(self.blocks(self.stem(x)))

    def forward(self, windows: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        B, K, L = windows.shape
        out = self._encode(windows.reshape(B * K, 1, L))
        return out.reshape(B, K, -1)


class _TCNBlock(nn.Module):
    """Causal dilated conv block with residual (padding left only)."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int, kernel: int = 7):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.c1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad, bias=False)
        self.gn1 = _gn(out_ch)
        self.c2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad, bias=False)
        self.gn2 = _gn(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gn1(self.c1(x)))[..., : x.shape[-1]]
        h = F.silu(self.gn2(self.c2(h)))[..., : x.shape[-1]]
        return h + self.shortcut(x)


class TCNEncoder(nn.Module):
    """Dilated causal TCN: stem conv -> blocks(dilations 1,2,4,8,16) -> pool+proj."""

    def __init__(self, window_length: int = 8192, hidden_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1, bias=False), _gn(32), nn.SiLU())
        chans = [32, 64, 96, 128, 128]
        dilations = [1, 2, 4, 8, 16]
        blocks = []
        for i, (c, d) in enumerate(zip(chans, dilations)):
            blocks.append(_TCNBlock(chans[i - 1] if i else 32, c, d))
        self.blocks = nn.Sequential(*blocks)
        self.pool = _PoolProj(chans[-1], hidden_dim)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(self.stem(x)))

    def forward(self, windows: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        B, K, L = windows.shape
        out = self._encode(windows.reshape(B * K, 1, L))
        return out.reshape(B, K, -1)


class _InceptionModule(nn.Module):
    """Multi-kernel 1D inception: convs k=11/21/41 + maxpool branch, residual."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.b10 = nn.Conv1d(in_ch, out_ch, 11, padding=5, bias=False)
        self.b20 = nn.Conv1d(in_ch, out_ch, 21, padding=10, bias=False)
        self.b40 = nn.Conv1d(in_ch, out_ch, 41, padding=20, bias=False)
        self.bp = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
        )
        self.gn = _gn(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.b10(x) + self.b20(x) + self.b40(x) + self.bp(x)
        return F.silu(self.gn(h) + self.shortcut(x))


class InceptionTimeEncoder(nn.Module):
    """InceptionTime-style 1D encoder: stem -> 3 inception modules -> pool+proj."""

    def __init__(self, window_length: int = 8192, hidden_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False), _gn(32), nn.SiLU(),
        )
        self.inception = nn.Sequential(
            _InceptionModule(32, 40),
            _InceptionModule(40, 56),
            _InceptionModule(56, 72),
        )
        self.pool = _PoolProj(72, hidden_dim)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.inception(self.stem(x)))

    def forward(self, windows: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        B, K, L = windows.shape
        out = self._encode(windows.reshape(B * K, 1, L))
        return out.reshape(B, K, -1)


ENCODERS = {
    "simple_cnn": "existing CNNBranch",
    "resnet1d": ResNet1DEncoder,
    "tcn": TCNEncoder,
}


class SimpleCNNEncoder(nn.Module):
    """Adapter exposing CNNBranch through the shared (windows, features) interface."""

    def __init__(self, window_length: int = 8192, hidden_dim: int = 128):
        super().__init__()
        from .encoder import CNNBranch

        self.cnn = CNNBranch(window_length, hidden_dim)

    def forward(self, windows: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        B, K, L = windows.shape
        out = self.cnn(windows.reshape(B * K, 1, L))
        return out.reshape(B, K, -1)


class LightTransformerEncoder(nn.Module):
    """Lightweight patch-based Transformer window encoder for industrial AI
    (TII track). Patches the 8192-sample window into P tokens, adds learnable
    positional embeddings, and runs a compact 2-layer Transformer encoder;
    global avg+max pooling projects to hidden_dim.

    Deliberately compact (~0.15M params) so it fits the lightweight,
    edge-deployable narrative while modernizing the time-series modeling
    compared with pure-CNN encoders.
    """

    def __init__(self, window_length: int = 8192, hidden_dim: int = 128,
                 patch: int = 64, d_model: int = 96, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 192,
                 dropout: float = 0.1):
        super().__init__()
        self.patch = patch
        n_patches = window_length // patch  # 8192/64 = 128 tokens
        self.proj = nn.Conv1d(1, d_model, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model * 2, hidden_dim)

    def forward(self, windows: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        B, K, L = windows.shape
        x = windows.reshape(B * K, 1, L)
        x = self.proj(x)                    # (BK, d_model, n_patches)
        x = x.transpose(1, 2)               # (BK, n_patches, d_model)
        x = x + self.pos
        x = self.encoder(x)
        x = self.norm(x)
        avg = x.mean(dim=1)
        mx = x.max(dim=1).values
        out = self.head(torch.cat([avg, mx], dim=-1))
        return out.reshape(B, K, -1)


class TimWindowEncoder(nn.Module):
    """Uniform preprocessing wrapper for the TIM matched-encoder comparison.

    Applies the same per-window robust normalization before every encoder so
    the preprocessing cannot vary across encoder rows. All variants emit
    (B, K, 128) window representations.
    """

    def __init__(
        self,
        encoder_name: str,
        window_length: int = 8192,
        feature_dim: int = 58,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.hidden_dim = hidden_dim
        # Inner encoder is constructed before the (parameter-free) preprocessor
        # so its parameter initialization consumes the same RNG stream as a
        # standalone WindowEncoder under the same seed.
        if encoder_name == "cnn":
            # WindowEncoder branch="cnn" already has the 80k architecture;
            # disable its internal normalization so it runs exactly once here.
            self.inner = WindowEncoder(
                window_length, feature_dim, hidden_dim,
                branch="cnn", apply_robust_normalize=False,
            )
        elif encoder_name == "simple_cnn":
            self.inner = SimpleCNNEncoder(window_length, hidden_dim)
        elif encoder_name == "resnet1d":
            self.inner = ResNet1DEncoder(window_length, hidden_dim)
        elif encoder_name == "inceptiontime":
            self.inner = InceptionTimeEncoder(window_length, hidden_dim)
        elif encoder_name == "tf_cnn":
            # Zheng et al. 2022 time-frequency CNN, re-tested under the E4
            # protocol (B2): STFT spectrogram + 2D CNN window encoder.
            self.inner = TFDCNNEncoder(window_length, hidden_dim)
        elif encoder_name == "lt_transformer":
            # Lightweight patch Transformer (TII track): modern time-series
            # modeling at edge-compatible capacity.
            self.inner = LightTransformerEncoder(window_length, hidden_dim)
        else:
            raise ValueError(f"Unknown TIM encoder: {encoder_name}")
        self.preprocess = RobustNormalize(window_length)

    def forward(self, windows: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # windows: (B, K, L); features: (B, K, 58) unused by raw-signal encoders
        return self.inner(self.preprocess(windows), features)


def build_dl_encoder(name: str, window_length: int = 8192, hidden_dim: int = 128) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNNEncoder(window_length, hidden_dim)
    if name == "tf_cnn":
        return TFDCNNEncoder(window_length, hidden_dim)
    if name not in ENCODERS or name == "simple_cnn":
        raise ValueError(f"Unknown DL encoder: {name}")
    return ENCODERS[name](window_length, hidden_dim)  # type: ignore[operator]


class TFDCNNEncoder(nn.Module):
    """Time-frequency CNN adapted from Zheng et al. 2022 for 1D windows.

    Computes log-magnitude STFT spectrograms per window, then applies three
    2D conv blocks and global avg+max pooling to produce 128-d window embeddings.
    """

    def __init__(self, window_length: int = 8192, hidden_dim: int = 128, n_fft: int = 256, hop_length: int = 128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), _gn(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), _gn(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), _gn(128), nn.SiLU(),
        )
        self.proj = nn.Linear(256, hidden_dim)

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, L) -> (N, 1, F, T)
        x = x.squeeze(1)
        win = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
            window=win, center=False, return_complex=True,
        )
        return torch.log1p(spec.abs()).unsqueeze(1)

    def forward(self, windows: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        B, K, L = windows.shape
        x = windows.reshape(B * K, 1, L)
        # Per-window standardization keeps STFT scales comparable across signals.
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        h = self.conv(self._spectrogram(x))
        avg = h.mean(dim=(-2, -1))
        mx = h.amax(dim=(-2, -1))
        out = self.proj(torch.cat([avg, mx], dim=-1))
        return out.reshape(B, K, -1)
