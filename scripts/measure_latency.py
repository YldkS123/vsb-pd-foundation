# -*- coding: utf-8 -*-
"""P1: complete latency / throughput profiling of the locked reference model.

Measures, for the dual/gated-attention/cyclic reference model
(results/model_full fold-1 checkpoint):

  - GPU inference: batch=64 throughput and batch=1 latency per measurement;
  - CPU inference: batch=1 latency per measurement;
  - window selection time per phase signal (event score + hybrid selection);
  - physical feature extraction time per window (vectorized batch);
  - estimated end-to-end per-measurement latency
    (3x window selection + 3xK feature extraction + GPU batch=1 inference);
  - peak GPU memory during batch=64 inference.

Window-selection timing uses a synthetic 800k-point signal (content-independent
algorithmic cost) because the raw train parquet is not stored in this repo.

Usage:
  python scripts/measure_latency.py [--cache results/cached_features/features_full.npz]
                                    [--ckpts results/model_full] [--out results/latency_report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vsb_pd.encoder import WindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.cyclic import CyclicPhaseModule
from vsb_pd.model import VSBPipeline
from vsb_pd.features import extract_physical_features
from vsb_pd.events import select_hybrid_windows
from vsb_pd.config import WindowPolicy

SAMPLE_RATE = 40_000_000


def build_model() -> VSBPipeline:
    return VSBPipeline(
        encoder=WindowEncoder(8192, 58, 128, branch="dual"),
        aggregator=MILAggregator("gated_attention", 128),
        cyclic=CyclicPhaseModule(128),
        classifier=PhaseClassifier(128),
    )


def timed(fn, n_repeat: int = 10):
    best = float("inf")
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cached_features/features_full.npz")
    ap.add_argument("--ckpts", default="results/model_full")
    ap.add_argument("--out", default="results/latency_report.json")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    windows_np = np.asarray(d["windows"])  # (2481, 3, K, 8192)
    feat_np = np.asarray(d["feat_array"])  # (2481, 3, K, 58)
    print(f"data: {windows_np.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    ckpt = torch.load(Path(args.ckpts) / "model_fold1.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    report: dict = {"model": "dual + gated_attention + cyclic + noisy-OR",
                    "n_params": int(n_params), "device": str(device)}

    # ---- GPU batch=64 throughput ----
    wb = torch.from_numpy(windows_np[:64]).float().to(device)
    fb = torch.from_numpy(feat_np[:64]).float().to(device)
    with torch.no_grad():
        for _ in range(5):
            model(wb, fb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        n_rep = 20
        for _ in range(n_rep):
            model(wb, fb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_rep
    report["gpu_batch64"] = {
        "ms_per_batch": round(dt * 1000, 3),
        "ms_per_measurement_throughput": round(dt * 1000 / 64, 4),
    }
    if device.type == "cuda":
        report["gpu_batch64"]["peak_gpu_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024 / 1024, 1)

    # ---- GPU batch=1 latency ----
    w1 = torch.from_numpy(windows_np[:1]).float().to(device)
    f1 = torch.from_numpy(feat_np[:1]).float().to(device)
    with torch.no_grad():
        for _ in range(10):
            model(w1, f1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_rep = 50
        for _ in range(n_rep):
            model(w1, f1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_rep
    report["gpu_batch1_ms_per_measurement"] = round(dt * 1000, 4)

    # ---- CPU batch=1 latency ----
    cpu_model = build_model().to("cpu")
    cpu_model.load_state_dict(ckpt["state_dict"])
    cpu_model.eval()
    w1c = torch.from_numpy(windows_np[:1]).float()
    f1c = torch.from_numpy(feat_np[:1]).float()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    with torch.no_grad():
        for _ in range(3):
            cpu_model(w1c, f1c)
        t0 = time.perf_counter()
        n_rep = 20
        for _ in range(n_rep):
            cpu_model(w1c, f1c)
        dt = (time.perf_counter() - t0) / n_rep
    report["cpu_batch1_ms_per_measurement"] = round(dt * 1000, 4)

    # ---- window selection (event score + hybrid selection) ----
    policy = WindowPolicy(
        window_length=8192, uniform_count=4, event_count=4,
        dedup_iou=0.5, fallback_grid_size=256,
    )
    rng = np.random.default_rng(0)
    synth = rng.standard_normal(800_000).astype(np.float32) * 0.5
    synth[::17] *= 20  # sparse impulses so peaks exist
    n_sel = 20
    t_sel = timed(lambda: select_hybrid_windows(synth, policy), n_repeat=n_sel)
    report["window_selection_ms_per_signal"] = round(t_sel * 1000, 3)

    # ---- physical feature extraction per window ----
    # extract_physical_features expects 3D (P,K,L) or 4D (B,P,K,L);
    # wrap the single phase x K windows as (1, K, L).
    win_batch = windows_np[0, 0, None, :, :]  # (1, K=8, 8192)
    n_windows = win_batch.shape[1]
    t_feat = timed(lambda: extract_physical_features(win_batch, SAMPLE_RATE), n_repeat=20)
    report["feature_extraction_ms_per_window"] = round(t_feat * 1000 / n_windows, 4)
    report["feature_extraction_ms_per_window"] = round(t_feat * 1000 / win_batch.shape[0], 4)

    # ---- estimated end-to-end per measurement ----
    e2e_ms = (3 * t_sel * 1000
              + 3 * 8 * (t_feat * 1000 / n_windows)
              + report["gpu_batch1_ms_per_measurement"])
    report["end_to_end_ms_per_measurement_estimate"] = round(e2e_ms, 3)
    report["components_ms"] = {
        "window_selection_x3": round(3 * t_sel * 1000, 3),
        "feature_extraction_3x8": round(3 * 8 * (t_feat * 1000 / n_windows), 3),
        "gpu_batch1_inference": report["gpu_batch1_ms_per_measurement"],
    }

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
