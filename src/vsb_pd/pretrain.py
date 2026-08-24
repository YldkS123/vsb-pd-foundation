"""VICReg self-supervised pretraining for the VSB window encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAugment(nn.Module):
    """Apply 4 augmentations to 1D window signals.

    Each call produces one augmented view. Two calls produce two different views.
    Augmentations applied independently:
    - Time shift: ±128 samples, reflect-pad
    - Amplitude scale: uniform [0.9, 1.1]
    - Gaussian noise: SNR uniform [20, 40] dB
    - Frequency masking: random band ≤ 5% Nyquist (applied in frequency domain)
    """

    def __init__(self, window_length: int = 8192, max_shift: int = 128):
        super().__init__()
        self.window_length = window_length
        self.max_shift = max_shift

    def _time_shift(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        B, C, L = x.shape
        shifts = torch.randint(-self.max_shift, self.max_shift + 1, (B,), device=x.device)
        out = torch.zeros_like(x)
        for i in range(B):
            s = shifts[i].item()
            if s > 0:
                out[i, 0, s:] = x[i, 0, :L - s]
                out[i, 0, :s] = torch.flip(x[i, 0, :s], dims=[-1])
            elif s < 0:
                s_abs = -s
                out[i, 0, :L - s_abs] = x[i, 0, s_abs:]
                out[i, 0, L - s_abs:] = torch.flip(x[i, 0, L - s_abs:], dims=[-1])
            else:
                out[i] = x[i]
        return out

    def _amplitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        scales = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(0.9, 1.1)
        return x * scales

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        snr_db = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(20, 40)
        snr_linear = 10 ** (snr_db / 10)
        signal_power = x.pow(2).mean(dim=(-1, -2), keepdim=True).clamp(min=1e-10)
        noise_std = torch.sqrt(signal_power / snr_linear)
        noise = torch.randn_like(x) * noise_std
        return x + noise

    def _freq_mask(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        X = torch.fft.rfft(x.squeeze(1), dim=-1)
        n_freq = X.shape[-1]
        max_band = max(1, int(n_freq * 0.05))

        for i in range(B):
            band_size = torch.randint(1, max_band + 1, (1,), device=x.device).item()
            start = torch.randint(0, n_freq - band_size + 1, (1,), device=x.device).item()
            X[i, start:start + band_size] = 0

        X_masked = torch.fft.irfft(X, n=L, dim=-1).unsqueeze(1)
        return X_masked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._time_shift(x)
        x = self._amplitude_scale(x)
        x = self._gaussian_noise(x)
        x = self._freq_mask(x)
        x = torch.clamp(x, -10.0, 10.0)
        return x


def vicreg_loss(
    za: torch.Tensor,
    zb: torch.Tensor,
    lambda_: float = 25.0,
    mu: float = 25.0,
    nu: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg loss: variance + invariance + covariance regularization."""
    B, D = za.shape

    inv_loss = F.mse_loss(za, zb)

    std_za = torch.sqrt(za.var(dim=0) + eps)
    std_zb = torch.sqrt(zb.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std_za)) + torch.mean(F.relu(1.0 - std_zb))

    za_centered = za - za.mean(dim=0, keepdim=True)
    zb_centered = zb - zb.mean(dim=0, keepdim=True)

    cov_za = (za_centered.T @ za_centered) / (B - 1)
    cov_zb = (zb_centered.T @ zb_centered) / (B - 1)

    diag_mask = torch.eye(D, device=za.device, dtype=torch.bool)
    cov_loss = (
        cov_za[~diag_mask].pow(2).sum() / D
        + cov_zb[~diag_mask].pow(2).sum() / D
    )

    return lambda_ * var_loss + mu * inv_loss + nu * cov_loss


def pretrain_vicreg(
    encoder: nn.Module,
    windows: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Pretrain the CNN branch of the encoder using VICReg."""
    encoder = encoder.to(device)
    augment = WindowAugment().to(device)

    projector = nn.Sequential(
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
    ).to(device)

    params = list(encoder.cnn.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    N = windows.shape[0]
    encoder.train()

    for epoch in range(epochs):
        perm = torch.randperm(N)

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            batch = windows[idx].to(device)
            batch = batch.unsqueeze(1)

            view_a = augment(batch)
            view_b = augment(batch)

            za = encoder.cnn(view_a)
            zb = encoder.cnn(view_b)

            za = projector(za)
            zb = projector(zb)

            loss = vicreg_loss(za, zb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

    return {k: v.cpu() for k, v in encoder.cnn.state_dict().items()}
