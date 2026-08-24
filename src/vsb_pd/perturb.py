# -*- coding: utf-8 -*-
"""Inference-time perturbations for robustness evaluation."""

from __future__ import annotations

import numpy as np


def time_shift(windows: np.ndarray, shift: int) -> np.ndarray:
    """Shift window content along the sample axis with zero padding.

    Semantics: the recording arrives ``shift`` samples later (``shift > 0``) or
    earlier (``shift < 0``) relative to the fixed window, so the content moves
    right/left and the vacated edge is zero-filled. Content is genuinely *lost*
    at one edge and the lost amount is monotonic in ``|shift|``.

    Do NOT implement this with cyclic wrap-around (``np.roll``/``torch.roll``):
    a cyclic roll interacts with the reference encoder's 4x stride-4
    downsampling (total factor 256), producing a periodic alignment artifact.
    With the pre-fix implementation, a 256-sample roll was nearly invisible
    (PR-AUC ~ baseline) while a 64-sample roll was maximally destructive
    (PR-AUC 0.23 vs baseline 0.43), i.e. the reported "±64 worse than ±128"
    non-monotonicity was a strided-subsampling phase artifact, not a genuine
    time-shift robustness property.

    Args:
        windows: (..., L) array of window samples.
        shift: number of samples to move the content (positive = right/delayed,
            negative = left/early).

    Returns:
        Zero-padded shifted copy with the same shape and dtype.
    """
    if shift == 0:
        return windows.copy()
    out = np.zeros_like(windows)
    if shift > 0:
        out[..., shift:] = windows[..., :-shift]
    else:
        out[..., :shift] = windows[..., -shift:]
    return out
