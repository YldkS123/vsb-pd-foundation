# -*- coding: utf-8 -*-
"""Latency distribution (p50/p90/p95/p99) for the final 80k mainline.

Uses the locked dev-fold checkpoint #1 of the 80k mainline
(cnn + attention + mean, 80,113 params) on the K=8 mixed-window cache.
Reports GPU batch=64 throughput, GPU batch=1 per-measurement latency and
CPU batch=1 per-measurement latency with percentile summaries.

The blind holdout is never touched by this measurement.

Usage:
  python scripts/measure_latency_80k_p50p95.py [--out results/latency_80k_distribution.json]
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
from vsb_pd.encoder import WindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.cyclic import PhaseInteractionModule
from vsb_pd.model import VSBPipeline

ROOT = Path(__file__).resolve().parent.parent


def build_model() -> VSBPipeline:
    return VSBPipeline(
        encoder=WindowEncoder(8192, 58, 128, branch="cnn"),
        aggregator=MILAggregator("attention", 128),
        cyclic=PhaseInteractionModule("mean", 128),
        classifier=PhaseClassifier(128),
    )


def percentiles(values: np.ndarray) -> dict:
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "n_reps": int(len(values)),
    }


@torch.no_grad()
def time_loop(fn, n_reps: int, sync: bool = True) -> np.ndarray:
    vals = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cached_features/features_policy_mixed_k8.npz")
    ap.add_argument(
        "--ckpt",
        default="results/ablations/dev_k8_blind_ckpts/enc_cnn__mil_attention__ph_mean/model_fold1.pt",
    )
    ap.add_argument("--out", default="results/latency_80k_distribution.json")
    args = ap.parse_args()

    d = np.load(ROOT / args.cache, allow_pickle=True)
    windows_np = np.asarray(d["windows"])  # (M, 3, K, 8192)
    feat_np = np.asarray(d["feat_array"])  # (M, 3, K, 58)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    ckpt = torch.load(ROOT / args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    report = {
        "model": "cnn + attention + mean + noisy-OR",
        "n_params": int(n_params),
        "device": str(device),
        "checkpoint_sha256": None,
    }

    # ---- GPU batch=64 throughput distribution ----
    wb = torch.from_numpy(windows_np[:64]).float().to(device)
    fb = torch.from_numpy(feat_np[:64]).float().to(device)
    for _ in range(10):
        model(wb, fb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    batch_ms = time_loop(lambda: model(wb, fb), n_reps=200, sync=(device.type == "cuda"))
    report["gpu_batch64_ms_per_batch"] = percentiles(batch_ms)
    report["gpu_batch64_ms_per_measurement"] = percentiles(batch_ms / 64)

    # ---- GPU batch=1 per-measurement latency distribution ----
    w1 = torch.from_numpy(windows_np[:1]).float().to(device)
    f1 = torch.from_numpy(feat_np[:1]).float().to(device)
    for _ in range(20):
        model(w1, f1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu1_ms = time_loop(lambda: model(w1, f1), n_reps=500, sync=(device.type == "cuda"))
    report["gpu_batch1_ms_per_measurement"] = percentiles(gpu1_ms)

    # ---- CPU batch=1 per-measurement latency distribution ----
    cpu_model = build_model().to("cpu")
    cpu_model.load_state_dict(ckpt["state_dict"])
    cpu_model.eval()
    w1c = torch.from_numpy(windows_np[:1]).float()
    f1c = torch.from_numpy(feat_np[:1]).float()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    for _ in range(10):
        cpu_model(w1c, f1c)
    cpu1_ms = time_loop(lambda: cpu_model(w1c, f1c), n_reps=300, sync=False)
    report["cpu_batch1_ms_per_measurement"] = percentiles(cpu1_ms)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
