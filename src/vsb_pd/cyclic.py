"""Three-phase cyclic symmetry module, noisy-OR, and PhaseCyclicLoss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CyclicPhaseModule(nn.Module):
    """2-layer 1D circular convolution on the phase dimension with residual connection.

    Enforces cyclic equivariance: applying a phase roll to the input produces
    the corresponding roll in the output.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # Conv1d on the phase dimension (which has 3 channels, each of size hidden_dim)
        # We transpose so that hidden_dim becomes the feature dimension,
        # and the 3 phases become the sequence dimension.
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, 3, hidden_dim)
        # Transpose to (B, hidden_dim, 3) for Conv1d over the phase dimension
        residual = x
        x_t = x.transpose(1, 2)  # (B, hidden_dim, 3)

        x_t = self.conv1(x_t)
        x_t = self.act(x_t)
        x_t = self.conv2(x_t)
        x_t = self.act(x_t)

        out = x_t.transpose(1, 2)  # (B, 3, hidden_dim)
        out = out + residual  # residual connection

        if mask is not None:
            # mask: (B, 3) bool, True = present
            out = out * mask.unsqueeze(-1).float()

        return out


def noisy_or_probs(phase_probs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute noisy-OR measurement probability from per-phase probabilities.

    noisy_or = 1 - prod(1 - p_i) for unmasked phases.
    If all phases are masked, returns 0.

    Args:
        phase_probs: (B, num_phases) per-phase probabilities in [0, 1]
        mask: (B, num_phases) bool, True = phase present (not missing)

    Returns:
        measurement_prob: (B,) measurement-level probability
    """
    if mask is not None:
        # Zero out missing phases so they don't contribute
        phase_probs = phase_probs * mask.float()

    # noisy_or = 1 - prod(1 - p_i)
    complement = 1.0 - phase_probs
    prod_complement = complement.prod(dim=-1)  # (B,)
    return 1.0 - prod_complement


class PhaseCyclicLoss(nn.Module):
    """Combined phase-level + measurement-level BCE loss.

    Loss = BCE(phase_probs, phase_labels) + lambda_m * BCE(measurement_prob, measurement_label)
    where measurement_label = max(phase_labels).
    """

    def __init__(self, lambda_m: float = 0.25):
        super().__init__()
        self.lambda_m = lambda_m
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        phase_logits: torch.Tensor,
        phase_labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # phase_logits: (B, num_phases) raw logits
        # phase_labels: (B, num_phases) 0/1 float
        B, num_phases = phase_logits.shape

        # Phase loss: BCE on each phase independently
        phase_loss = self.bce(phase_logits, phase_labels)

        if self.lambda_m > 0:
            # Measurement label = max(phase_labels)
            measurement_label = phase_labels.max(dim=-1).values  # (B,)

            # Measurement probability via noisy-OR
            phase_probs = torch.sigmoid(phase_logits)
            measurement_prob = noisy_or_probs(phase_probs, mask)  # (B,)

            # BCE on measurement level needs logits, but we have probs.
            # We use the numerically stable formulation:
            # BCE(p, y) = -y*log(p) - (1-y)*log(1-p)
            eps = 1e-7
            meas_p = measurement_prob.clamp(eps, 1 - eps)
            meas_loss = -(
                measurement_label * torch.log(meas_p)
                + (1 - measurement_label) * torch.log(1 - meas_p)
            ).mean()

            return phase_loss + self.lambda_m * meas_loss

        return phase_loss


class PhaseInteractionModule(nn.Module):
    """Three-phase interaction variant for ablations (all output (B,3,hidden)).

    kind="none":    per-phase vectors unchanged (no cross-phase interaction)
    kind="cyclic":  2-layer circular conv with residual (reference full model)
    kind="concat":  concatenate all 3 phase vectors, project back (direct concat)
    kind="max":     element-wise max across phases, broadcast to each phase
    kind="mean":    element-wise mean across phases, broadcast to each phase
    kind="context_concat": per-phase vector concatenated with the mean of
        unmasked phase vectors, projected via Linear(256,128)+LayerNorm+SiLU
    kind="context_add": per-phase vector plus a projection of the unmasked
        phase mean via Linear(128,128)+LayerNorm+SiLU
    """

    def __init__(self, kind: str = "cyclic", hidden_dim: int = 128):
        super().__init__()
        self.kind = kind
        if kind == "cyclic":
            self.cyclic = CyclicPhaseModule(hidden_dim)
        elif kind == "concat":
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
        elif kind == "context_concat":
            self.context_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
        elif kind == "context_add":
            self.context_phi = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
        elif kind not in ("none", "max", "mean"):
            raise ValueError(f"Unknown phase interaction: {kind}")

    def _context_mean(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Mean over unmasked phases; zero context when all phases are missing."""
        B, P, D = x.shape
        if mask is None:
            return x.mean(dim=1, keepdim=True)  # (B, 1, D)
        mask_f = mask.float().unsqueeze(-1)  # (B, P, 1)
        counts = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1, 1)
        return (x * mask_f).sum(dim=1, keepdim=True) / counts  # (B, 1, D)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, 3, hidden_dim)
        if self.kind == "cyclic":
            return self.cyclic(x, mask=mask)
        if self.kind == "none":
            out = x
        elif self.kind == "concat":
            B, P, D = x.shape
            out = self.proj(x.reshape(B, -1)).unsqueeze(1).expand(B, P, D)
        elif self.kind == "max":
            out = x.max(dim=1, keepdim=True).values.expand_as(x)
        elif self.kind == "mean":
            out = x.mean(dim=1, keepdim=True).expand_as(x)
        elif self.kind == "context_concat":
            B, P, D = x.shape
            ctx = self._context_mean(x, mask).expand(B, P, D)
            out = self.context_proj(torch.cat([x, ctx], dim=-1))
        else:  # context_add
            B, P, D = x.shape
            ctx = self._context_mean(x, mask).expand(B, P, D)
            out = x + self.context_phi(ctx)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out
