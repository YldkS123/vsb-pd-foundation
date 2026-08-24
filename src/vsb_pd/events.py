from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .config import WindowPolicy
from .windows import WindowCandidate, deduplicate_and_fill, uniform_starts


def robust_nonnegative_z(values: np.ndarray) -> np.ndarray:
    """Return the locked nonnegative robust z-score for one feature."""
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        scale = np.mean(np.abs(values - median))
    if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        return np.zeros_like(values)
    return np.maximum((values - median) / scale, 0.0).astype(np.float32)


def event_score(x: np.ndarray, rolling_width: int = 256) -> np.ndarray:
    """Score amplitude, Teager, and rolling-difference energy without tuning."""
    if not isinstance(rolling_width, Integral) or isinstance(rolling_width, bool):
        raise ValueError("rolling_width must be a positive integer")
    if rolling_width < 1:
        raise ValueError("rolling_width must be a positive integer")

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if x.size == 0:
        raise ValueError("signal must not be empty")
    if not np.isfinite(x).all():
        raise ValueError("signal must contain only finite values")

    centered = x - np.median(x)
    amplitude = np.abs(centered)

    teager = np.zeros_like(centered)
    teager[1:-1] = np.abs(centered[1:-1] ** 2 - centered[:-2] * centered[2:])

    diff = np.diff(centered, prepend=centered[0])
    diff_energy = uniform_filter1d(diff ** 2, size=rolling_width, mode="reflect")
    diff_rms = np.sqrt(np.maximum(diff_energy, 0.0))

    return np.maximum.reduce([
        robust_nonnegative_z(amplitude),
        robust_nonnegative_z(teager),
        robust_nonnegative_z(diff_rms),
    ])


def select_hybrid_windows(x: np.ndarray, policy: WindowPolicy) -> list[WindowCandidate]:
    """Select deterministic uniform anchors plus score-ranked event windows."""
    policy.validate()
    signal_length = int(len(x))
    if signal_length < policy.window_length:
        raise ValueError("signal is shorter than one window")

    anchors = [
        WindowCandidate(start, "uniform", 0.0)
        for start in uniform_starts(signal_length, policy.window_length, policy.uniform_count)
    ] if policy.uniform_count else []
    score = event_score(x)
    peak_indices, _ = find_peaks(score, distance=policy.window_length // 2)
    max_start = signal_length - policy.window_length
    events = [
        WindowCandidate(
            start=int(np.clip(index - policy.window_length // 2, 0, max_start)),
            kind="event",
            score=float(score[index]),
        )
        for index in peak_indices
        if score[index] > 0
    ]
    return deduplicate_and_fill(
        signal_length=signal_length,
        window_length=policy.window_length,
        uniform=anchors,
        events=events,
        event_quota=policy.event_count,
        dedup_iou=policy.dedup_iou,
        grid_size=policy.fallback_grid_size,
    )
