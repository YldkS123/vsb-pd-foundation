"""Vectorized physical feature extraction — 58 features batch-computed over all windows.

All operations work on 2D arrays (N, 8192) where N = num_phases * num_windows,
eliminating the per-window Python for-loop for most operations.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew


def _time_domain(X: np.ndarray) -> dict[str, np.ndarray]:
    """20 time-domain features. X: (N, L) float64. Returns dict of (N,) arrays."""
    feats: dict[str, np.ndarray] = {}
    max_v = X.max(axis=1)
    min_v = X.min(axis=1)
    feats["max"] = max_v
    feats["min"] = min_v
    feats["peak_to_peak"] = max_v - min_v
    mean = X.mean(axis=1)
    feats["mean"] = mean
    std = X.std(axis=1)
    feats["std"] = std
    rms = np.sqrt(np.mean(X**2, axis=1))
    feats["rms"] = rms
    feats["skewness"] = np.where(std > 1e-12, skew(X, axis=1), 0.0)
    feats["kurtosis"] = np.where(std > 1e-12, kurtosis(X, axis=1), 0.0)
    peak = np.maximum(np.abs(max_v), np.abs(min_v))
    feats["crest_factor"] = np.where(rms > 1e-12, peak / rms, 0.0)
    mean_abs = np.abs(X).mean(axis=1)
    feats["shape_factor"] = np.where(mean_abs > 1e-12, rms / mean_abs, 0.0)
    feats["impulse_factor"] = np.where(mean_abs > 1e-12, peak / mean_abs, 0.0)
    sqrt_abs_mean = np.mean(np.sqrt(np.abs(X)), axis=1)
    feats["clearance_factor"] = np.where(mean_abs > 1e-12, peak / (sqrt_abs_mean**2), 0.0)
    feats["margin_factor"] = np.where(mean_abs > 1e-12, peak / (mean_abs**2), 0.0)
    feats["energy"] = np.sum(X**2, axis=1)
    diff = np.diff(X, axis=1)
    crossings = np.sum(np.diff(np.signbit(X), axis=1), axis=1)
    feats["zero_crossing_rate"] = crossings / (X.shape[1] - 1)
    feats["mean_abs_dev"] = np.mean(np.abs(X - mean[:, None]), axis=1)
    med = np.median(X, axis=1)
    feats["median_abs_dev"] = np.median(np.abs(X - med[:, None]), axis=1)
    q75 = np.percentile(X, 75, axis=1)
    q25 = np.percentile(X, 25, axis=1)
    feats["interquartile_range"] = q75 - q25
    feats["rms_of_diff"] = np.sqrt(np.mean(diff**2, axis=1))
    feats["peak_count_above_3sigma"] = np.sum(np.abs(X - mean[:, None]) > 3 * std[:, None], axis=1)
    return feats


def _frequency_domain(X: np.ndarray, sr: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """12 frequency-domain features. Returns (feats_dict, spectrum, freqs)."""
    N, L = X.shape
    X_f = np.fft.rfft(X, axis=1)
    spectrum = np.abs(X_f)
    freqs = np.fft.rfftfreq(L, d=1.0 / sr)
    power = spectrum**2
    total_power = power.sum(axis=1, keepdims=True)
    total_power = np.maximum(total_power, 1e-30)

    feats: dict[str, np.ndarray] = {}
    feats["dominant_freq_hz"] = freqs[np.argmax(spectrum, axis=1)]
    feats["dominant_magnitude"] = spectrum.max(axis=1)
    centroid_num = (freqs[None, :] * power).sum(axis=1)
    centroid = centroid_num / total_power[:, 0]
    feats["spectral_centroid_hz"] = centroid
    delta = freqs[None, :] - centroid[:, None]
    feats["spectral_bandwidth_hz"] = np.sqrt(
        ((delta**2) * power).sum(axis=1) / total_power[:, 0]
    )

    # Rolloff: for each row, find where cumulative power crosses 0.85
    cum_power = np.cumsum(power, axis=1)
    cum_frac = cum_power / total_power
    N_freq = cum_frac.shape[1]
    rolloff_idx = np.zeros(N, dtype=np.int64)
    for i in range(N):
        rolloff_idx[i] = np.searchsorted(cum_frac[i], 0.85)
    feats["spectral_rolloff_hz"] = freqs[np.clip(rolloff_idx, 0, N_freq - 1)]

    norm_power = power / total_power
    norm_power_safe = np.maximum(norm_power, 1e-30)
    feats["spectral_flatness"] = np.exp(np.mean(np.log(norm_power_safe), axis=1)) / (
        np.mean(norm_power, axis=1) + 1e-30
    )
    feats["spectral_entropy"] = -np.sum(norm_power_safe * np.log2(norm_power_safe), axis=1)
    feats["spectral_rms"] = np.sqrt(np.mean(spectrum**2, axis=1))
    feats["spectral_skewness"] = np.where(
        np.abs(spectrum).std(axis=1) > 1e-12, skew(spectrum, axis=1), 0.0
    )
    feats["spectral_kurtosis"] = np.where(
        np.abs(spectrum).std(axis=1) > 1e-12, kurtosis(spectrum, axis=1), 0.0
    )
    freqs_mean = np.mean(freqs)
    power_mean = power.mean(axis=1, keepdims=True)
    slope_num = ((freqs[None, :] - freqs_mean) * (power - power_mean)).sum(axis=1)
    slope_den = ((freqs[None, :] - freqs_mean) ** 2).sum()
    feats["spectral_slope"] = np.where(slope_den > 1e-30, slope_num / slope_den, 0.0)
    decrease = np.diff(spectrum, axis=1)
    decrease = np.maximum(decrease, 0)
    feats["spectral_decrease"] = np.where(
        total_power[:, 0] > 1e-30, decrease.sum(axis=1) / total_power[:, 0], 0.0,
    )
    return feats, spectrum, freqs


def _band_energy(power: np.ndarray, freqs: np.ndarray) -> dict[str, np.ndarray]:
    """13 band-energy features, 0-20 MHz equally divided."""
    feats: dict[str, np.ndarray] = {}
    band_edges = np.linspace(0, 20_000_000, 14)
    total_power = power.sum(axis=1, keepdims=True)
    total_power = np.maximum(total_power, 1e-30)
    for b in range(13):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        band_energy = power[:, mask].sum(axis=1)
        feats[f"band_{b}_{band_edges[b]/1e6:.2f}MHz"] = band_energy / total_power[:, 0]
    return feats


def _autocorr_ar(X: np.ndarray) -> dict[str, np.ndarray]:
    """9 auto-correlation / AR features. X: (N, L) float64."""
    N, L = X.shape
    mean = X.mean(axis=1, keepdims=True)
    demeaned = X - mean

    # Auto-correlation via FFT (O(N log N) vs O(N^2))
    fft_len = 2 * L - 1
    fft_a = np.fft.fft(demeaned, n=fft_len, axis=1)
    acf_all = np.fft.ifft(fft_a * np.conj(fft_a), axis=1).real
    acf = acf_all[:, :L] / np.maximum(acf_all[:, :1], 1e-30)

    acf_peaks = np.zeros((N, 3), dtype=np.float64)
    acf_pro = np.zeros((N, 3), dtype=np.float64)
    ar_coefs = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        idx, props = sp_signal.find_peaks(acf[i, 1:], distance=10)
        prominences = props.get("prominences", np.array([]))
        if len(idx) > 0:
            order = np.argsort(-prominences)
            for j in range(min(3, len(order))):
                acf_peaks[i, j] = float(idx[order[j]] + 1)
                acf_pro[i, j] = float(prominences[order[j]])
        try:
            ar = sp_signal.arburg(X[i], 3)
            ar_coefs[i] = ar[0][1:]
        except Exception:
            pass

    return {
        "acf_peak_1_lag": acf_peaks[:, 0],
        "acf_peak_1_prominence": acf_pro[:, 0],
        "acf_peak_2_lag": acf_peaks[:, 1],
        "acf_peak_2_prominence": acf_pro[:, 1],
        "acf_peak_3_lag": acf_peaks[:, 2],
        "acf_peak_3_prominence": acf_pro[:, 2],
        "ar_coef_1": ar_coefs[:, 0],
        "ar_coef_2": ar_coefs[:, 1],
        "ar_coef_3": ar_coefs[:, 2],
    }


def _peak_envelope(X: np.ndarray) -> dict[str, np.ndarray]:
    """4 peak/envelope features. X: (N, L) float64."""
    N, L = X.shape
    counts = np.zeros(N, dtype=np.float64)
    mean_prom = np.zeros(N, dtype=np.float64)

    analytic = sp_signal.hilbert(X, axis=1)
    envelope = np.abs(analytic)
    env_mean = envelope.mean(axis=1)
    env_std = envelope.std(axis=1)

    for i in range(N):
        peaks, props = sp_signal.find_peaks(X[i])
        counts[i] = len(peaks)
        prom = props.get("prominences", None)
        mean_prom[i] = float(np.mean(prom)) if prom is not None and len(prom) > 0 else 0.0

    return {
        "peak_count": counts,
        "mean_peak_prominence": mean_prom,
        "envelope_mean": env_mean,
        "envelope_std": env_std,
    }


def _extract_chunk(X: np.ndarray, sampling_rate_hz: int) -> dict[str, np.ndarray]:
    """Run the full 58-feature pipeline on a 2D (N, L) chunk."""
    feats = _time_domain(X)
    fd_feats, spectrum, freqs = _frequency_domain(X, sampling_rate_hz)
    feats.update(fd_feats)
    feats.update(_band_energy(spectrum**2, freqs))
    feats.update(_autocorr_ar(X))
    feats.update(_peak_envelope(X))
    assert len(feats) == 58, f"Expected 58 features, got {len(feats)}"
    return {k: np.asarray(v, dtype=np.float32) for k, v in feats.items()}


def extract_physical_features(
    windows: np.ndarray,
    sampling_rate_hz: int,
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Extract exactly 58 physical features from all windows - vectorized.

    Accepts 3D (phases, windows, length) or 4D (batch, phases, windows, length).
    All windows are reshaped to 2D (N, length) for batch computation.

    Returns dict of feature_name -> (N,) float32 array.
    """
    windows = np.asarray(windows, dtype=np.float32)

    # Normalize to 3D: (P, K, L) shapes
    if windows.ndim == 4:
        # (B, P, K, L) -> (B*P, K, L)
        B, P, K, L = windows.shape
        windows = windows.reshape(B * P, K, L)
        N = B * P * K
    elif windows.ndim == 3:
        P, K, L = windows.shape
        N = P * K
    else:
        raise ValueError(f"Expected 3D (P,K,L) or 4D (B,P,K,L), got {windows.ndim}D")

    X = windows.reshape(N, L).astype(np.float64)

    if batch_size is None or batch_size >= N:
        return _extract_chunk(X, sampling_rate_hz)

    # Chunked mode: bound peak memory of the FFT-based stages (rfft, autocorr,
    # Hilbert envelope) when N is large (e.g. 89,316 windows at K=12).
    chunk_size = max(1, int(batch_size))
    out: dict[str, np.ndarray] | None = None
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = _extract_chunk(X[start:end], sampling_rate_hz)
        if out is None:
            out = {k: np.empty(N, dtype=np.float32) for k in chunk}
        for k, v in chunk.items():
            out[k][start:end] = v
    return out
