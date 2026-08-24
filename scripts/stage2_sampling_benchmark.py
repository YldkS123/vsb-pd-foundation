# -*- coding: utf-8 -*-
"""IEEE TIM Stage 2 sampling-controlled benchmark.

Reports, for every sampling policy:

  - selection components on a real 800k development signal (event score, peak
    detection, policy-specific window selection);
  - model components (robust normalize / encoder / MIL / phase interaction /
    output / full pipeline) with the locked E4 architecture;
  - CPU batch=1 and GPU batch=1 per-measurement latency;
  - GPU throughput per measurement at a policy-appropriate batch size;
  - MACs/FLOPs per measurement and peak GPU memory.

The full-signal row uses K=1, L=800000 with the same CNN encoder. The frozen
blind set is never loaded or predicted.

Usage:
  python scripts/stage2_sampling_benchmark.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy.signal import find_peaks

from stage1_tim_benchmark import measure, model_components, summarize
from stage2_build_sampling_caches import random_window_starts

from vsb_pd.config import WindowPolicy
from vsb_pd.cyclic import PhaseInteractionModule
from vsb_pd.dl_encoders import TimWindowEncoder
from vsb_pd.events import event_score, select_hybrid_windows
from vsb_pd.macs import count_flops, count_macs
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.model import VSBPipeline
from vsb_pd.windows import uniform_starts

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "results" / "cached_features"
WINDOW_LENGTH = 8192
SIGNAL_LENGTH = 800_000


def build_pipeline(window_length: int) -> VSBPipeline:
    return VSBPipeline(
        encoder=TimWindowEncoder("cnn", window_length, 58, 128),
        aggregator=MILAggregator("attention", 128),
        cyclic=PhaseInteractionModule("context_concat", 128),
        classifier=PhaseClassifier(128),
        max_encode_chunk=8,
        checkpoint_chunks=True,
    )


def selection_components(signal: np.ndarray, n_warmup: int, n_reps: int) -> dict:
    out = {}
    out["event_score"] = summarize(
        measure(lambda: event_score(signal), n_warmup, n_reps, sync=False))
    score = event_score(signal)
    out["peak_detection"] = summarize(
        measure(lambda: find_peaks(score, distance=WINDOW_LENGTH // 2),
                n_warmup, n_reps, sync=False))

    mixed_policy = WindowPolicy(WINDOW_LENGTH, 4, 4, 0.5, 256)
    event_policy = WindowPolicy(WINDOW_LENGTH, 0, 8, 0.5, 256)
    uniform_policy = WindowPolicy(WINDOW_LENGTH, 8, 0, 0.5, 256)
    out["uniform_k8"] = summarize(
        measure(lambda: uniform_starts(SIGNAL_LENGTH, WINDOW_LENGTH, 8),
                n_warmup, n_reps, sync=False))
    out["event_k8"] = summarize(
        measure(lambda: select_hybrid_windows(signal, event_policy),
                n_warmup, n_reps, sync=False))
    out["mixed_k8"] = summarize(
        measure(lambda: select_hybrid_windows(signal, mixed_policy),
                n_warmup, n_reps, sync=False))
    rng = np.random.default_rng(2026)
    out["random_k8"] = summarize(
        measure(lambda: random_window_starts(SIGNAL_LENGTH, WINDOW_LENGTH, 8, rng),
                n_warmup, n_reps, sync=False))
    out["full_signal"] = {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0,
                          "p99_ms": 0.0, "n_reps": 0,
                          "note": "no selection; raw signal passed directly"}
    return out


def macs_for(model: VSBPipeline, windows: torch.Tensor, feat: torch.Tensor) -> dict:
    return {
        "macs_per_measurement": float(count_macs(model, windows, feat)),
        "flops_per_measurement": float(count_flops(model, windows, feat)),
    }


def load_first(cache: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(cache, allow_pickle=False)
    return np.asarray(d["windows"][:count]), np.asarray(d["feat_array"][:count])


def run_policy_entry(name: str, windows_np: np.ndarray, feat_np: np.ndarray,
                     batch_size: int, device: torch.device,
                     n_warmup: int, n_reps: int) -> dict:
    model = build_pipeline(windows_np.shape[-1])
    entry = {"n_params_full_pipeline": int(sum(p.numel() for p in model.parameters()))}
    w1 = torch.from_numpy(windows_np[:1]).float()
    f1 = torch.from_numpy(feat_np[:1]).float()
    entry.update(macs_for(model, w1, f1))

    cpu_model = build_pipeline(windows_np.shape[-1]).to("cpu").eval()
    w1c = torch.from_numpy(windows_np[:1]).float()
    f1c = torch.from_numpy(feat_np[:1]).float()
    with torch.no_grad():
        entry["cpu_batch1_model_components_ms"] = model_components(
            cpu_model, w1c, f1c, n_warmup, n_reps, sync=False)

    if device.type == "cuda":
        gpu_model = build_pipeline(windows_np.shape[-1]).to(device).eval()
        w1g = torch.from_numpy(windows_np[:1]).float().to(device)
        f1g = torch.from_numpy(feat_np[:1]).float().to(device)
        with torch.no_grad():
            entry["gpu_batch1_model_components_ms"] = model_components(
                gpu_model, w1g, f1g, n_warmup, n_reps, sync=True)
            wb = torch.from_numpy(windows_np[:batch_size]).float().to(device)
            fb = torch.from_numpy(feat_np[:batch_size]).float().to(device)
            for _ in range(max(1, n_warmup // 2)):
                gpu_model(wb, fb)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            batch_ms = measure(lambda: gpu_model(wb, fb), 2, max(5, n_reps), sync=True)
            entry["gpu_batch_size"] = batch_size
            entry["gpu_batch_ms_per_measurement"] = summarize(batch_ms / batch_size)
            entry["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    else:
        entry["gpu_batch1_model_components_ms"] = None
        entry["gpu_batch_ms_per_measurement"] = None
        entry["peak_gpu_mb"] = None
    try:
        import psutil
        entry["process_rss_mb"] = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    except Exception:
        entry["process_rss_mb"] = None
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/stage2_sampling/benchmark.json")
    ap.add_argument("--n-reps", type=int, default=50)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_reps, args.n_warmup = 5, 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))

    rng = np.random.default_rng(0)
    signal = rng.standard_normal(SIGNAL_LENGTH).astype(np.float32) * 0.5
    signal[::17] *= 20
    selection = selection_components(signal, args.n_warmup, args.n_reps)

    report = {
        "device": str(device),
        "signal": "synthetic 800k float32, rng seed 0, impulses every 17 samples",
        "selection_components_cpu_ms": selection,
        "policies": {},
    }

    policy_sources = {
        "mixed_k8": CACHE_DIR / "features_policy_mixed_k8.npz",
        "uniform_k8": CACHE_DIR / "features_policy_uniform_k8.npz",
        "event_k8": CACHE_DIR / "features_policy_event_k8.npz",
        "random_k8": CACHE_DIR / "features_policy_random_k8.npz",
        "full_signal": CACHE_DIR / "features_policy_full_signal.npz",
    }
    missing = [k for k, p in policy_sources.items() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing caches: " + ", ".join(missing))

    for name, cache in policy_sources.items():
        windows_np, feat_np = load_first(cache, 64)
        batch_size = 1 if windows_np.shape[-1] > WINDOW_LENGTH else 64
        report["policies"][name] = run_policy_entry(
            name, windows_np, feat_np, batch_size, device,
            args.n_warmup, args.n_reps,
        )
        print(f"  {name}: {tuple(windows_np.shape)} done")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
