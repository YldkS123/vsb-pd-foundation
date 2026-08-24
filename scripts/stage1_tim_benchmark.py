# -*- coding: utf-8 -*-
"""IEEE TIM Stage 1 end-to-end benchmark.

Measures, for each matched encoder pipeline (TimWindowEncoder -> Attention MIL
-> context_concat -> PhaseClassifier):

  - selection components (event score / peak detection / hybrid selection) on
    a synthetic 800k signal (same recipe as scripts/measure_latency.py);
  - model components (robust normalize / encoder / MIL / phase interaction /
    output / full pipeline) on K=8 cached development windows;
  - CPU batch=1 and GPU batch=1 per-measurement latency distributions;
  - GPU batch=64 throughput distribution and peak GPU memory;
  - per-window and per-measurement MACs/FLOPs via manual hooks.

The frozen blind set is never loaded or predicted.

Usage:
  python scripts/stage1_tim_benchmark.py [--smoke]
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
from scipy.signal import find_peaks

from vsb_pd.config import WindowPolicy
from vsb_pd.cyclic import PhaseInteractionModule
from vsb_pd.dl_encoders import TimWindowEncoder
from vsb_pd.events import event_score, select_hybrid_windows
from vsb_pd.macs import count_flops, count_macs
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.model import VSBPipeline

ROOT = Path(__file__).resolve().parent.parent


def build_pipeline(encoder_name: str) -> VSBPipeline:
    return VSBPipeline(
        encoder=TimWindowEncoder(encoder_name, 8192, 58, 128),
        aggregator=MILAggregator("attention", 128),
        cyclic=PhaseInteractionModule("context_concat", 128),
        classifier=PhaseClassifier(128),
        max_encode_chunk=8,
        checkpoint_chunks=True,
    )


def summarize(values: np.ndarray) -> dict:
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "n_reps": int(len(values)),
    }


def measure(fn, n_warmup: int, n_reps: int, sync: bool) -> np.ndarray:
    for _ in range(n_warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    vals = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(vals)


def model_components(model, w1, f1, n_warmup: int, n_reps: int, sync: bool) -> dict:
    enc, agg, cyc, cls = model.encoder, model.aggregator, model.cyclic, model.classifier
    B, P, K, L = w1.shape
    # Mirrors VSBPipeline.forward: flatten (B, P, K, L) to (B*P, K, L).
    w_flat = w1.reshape(B * P, K, L)
    f_flat = f1.reshape(B * P, K, 58)
    out = {}
    out["robust_normalize"] = summarize(
        measure(lambda: enc.preprocess(w_flat), n_warmup, n_reps, sync))
    encoded = enc(w_flat, f_flat)  # (B*P, K, 128)
    out["encoder"] = summarize(
        measure(lambda: enc(w_flat, f_flat), n_warmup, n_reps, sync))
    aggregated = agg(encoded)  # (B*P, 128)
    out["mil"] = summarize(measure(lambda: agg(encoded), n_warmup, n_reps, sync))
    interacted = cyc(aggregated.reshape(B, P, -1))  # (B, P, 128)
    out["phase_interaction"] = summarize(
        measure(lambda: cyc(aggregated.reshape(B, P, -1)), n_warmup, n_reps, sync))
    flat = interacted.reshape(-1, 128)
    out["output"] = summarize(measure(lambda: cls(flat), n_warmup, n_reps, sync))
    out["full_pipeline"] = summarize(
        measure(lambda: model(w1, f1), n_warmup, n_reps, sync))
    return out


def selection_components(signal: np.ndarray, policy: WindowPolicy,
                         n_warmup: int, n_reps: int) -> dict:
    out = {}
    out["event_score"] = summarize(
        measure(lambda: event_score(signal), n_warmup, n_reps, sync=False))
    score = event_score(signal)
    out["peak_detection"] = summarize(
        measure(lambda: find_peaks(score, distance=policy.window_length // 2),
                n_warmup, n_reps, sync=False))
    out["select_hybrid_windows"] = summarize(
        measure(lambda: select_hybrid_windows(signal, policy),
                n_warmup, n_reps, sync=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="cnn,simple_cnn,resnet1d,inceptiontime")
    ap.add_argument("--cache", default="results/cached_features/features_policy_mixed_k8.npz")
    ap.add_argument("--out", default="results/stage1_tim/benchmark.json")
    ap.add_argument("--n-reps", type=int, default=50)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_reps, args.n_warmup = 5, 1

    d = np.load(ROOT / args.cache, allow_pickle=False)
    windows_np = np.asarray(d["windows"])  # (M, 3, 8, 8192)
    feat_np = np.asarray(d["feat_array"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = WindowPolicy(window_length=8192, uniform_count=4, event_count=4,
                          dedup_iou=0.5, fallback_grid_size=256)
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(800_000).astype(np.float32) * 0.5
    signal[::17] *= 20
    sel = selection_components(signal, policy, args.n_warmup, args.n_reps)

    report = {
        "device": str(device),
        "signal": "synthetic 800k float32, rng seed 0, impulses every 17 samples",
        "selection_components_cpu_ms": sel,
        "encoders": {},
    }

    torch.set_num_threads(min(8, torch.get_num_threads()))
    for name in [e.strip() for e in args.encoders.split(",") if e.strip()]:
        if name not in ("cnn", "simple_cnn", "resnet1d", "inceptiontime"):
            raise ValueError(f"Unknown encoder: {name}")
        entry = {}
        model = build_pipeline(name)
        entry["n_params_full_pipeline"] = int(sum(p.numel() for p in model.parameters()))

        w1w = torch.from_numpy(windows_np[0, 0, :1]).float().unsqueeze(0)  # (1, 1, 8192)
        f1w = torch.from_numpy(feat_np[0, 0, :1]).float().unsqueeze(0)
        entry["macs_per_window"] = float(count_macs(model.encoder, w1w, f1w))
        entry["flops_per_window"] = float(count_flops(model.encoder, w1w, f1w))
        wm = torch.from_numpy(windows_np[:1]).float()  # (1, 3, 8, 8192)
        fm = torch.from_numpy(feat_np[:1]).float()
        entry["macs_per_measurement"] = float(count_macs(model, wm, fm))
        entry["flops_per_measurement"] = float(count_flops(model, wm, fm))

        # CPU batch=1
        cpu_model = build_pipeline(name).to("cpu").eval()
        w1c = torch.from_numpy(windows_np[:1]).float()
        f1c = torch.from_numpy(feat_np[:1]).float()
        with torch.no_grad():
            entry["cpu_batch1_model_components_ms"] = model_components(
                cpu_model, w1c, f1c, args.n_warmup, args.n_reps, sync=False)

        # GPU scenarios
        if device.type == "cuda":
            gpu_model = build_pipeline(name).to(device).eval()
            w1 = torch.from_numpy(windows_np[:1]).float().to(device)
            f1 = torch.from_numpy(feat_np[:1]).float().to(device)
            with torch.no_grad():
                entry["gpu_batch1_model_components_ms"] = model_components(
                    gpu_model, w1, f1, args.n_warmup, args.n_reps, sync=True)
                wb = torch.from_numpy(windows_np[:64]).float().to(device)
                fb = torch.from_numpy(feat_np[:64]).float().to(device)
                for _ in range(max(1, args.n_warmup // 2)):
                    gpu_model(wb, fb)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                batch_ms = measure(lambda: gpu_model(wb, fb), 2, max(5, args.n_reps), sync=True)
                entry["gpu_batch64_ms_per_measurement"] = summarize(batch_ms / 64)
                entry["peak_gpu_mb_batch64"] = round(
                    torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        else:
            entry["gpu_batch1_model_components_ms"] = None
            entry["gpu_batch64_ms_per_measurement"] = None
            entry["peak_gpu_mb_batch64"] = None

        try:
            import psutil
            entry["process_rss_mb"] = round(
                psutil.Process().memory_info().rss / 1024 / 1024, 1)
        except Exception:
            entry["process_rss_mb"] = None

        report["encoders"][name] = entry
        print(json.dumps({name: entry}, indent=2, default=str))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
