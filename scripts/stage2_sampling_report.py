# -*- coding: utf-8 -*-
"""IEEE TIM Stage 2 sampling-controlled evidence report.

Reads results/stage2_sampling/<policy>/cv_summary.json and oof.npz plus the
Stage 1 E4 mixed_k8 row and the Stage 2 benchmark. Emits a compact markdown
report with PR-AUC / ROC-AUC / MCC, cost columns, and paired cluster-bootstrap
95% CIs for the mixed vs uniform / event / random / full-signal comparisons.

Usage:
  python scripts/stage2_sampling_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vsb_pd.evaluation import paired_bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "results" / "stage2_sampling"
STAGE1 = ROOT / "results" / "stage1_tim"
METRIC_KEYS = ("pr_auc", "roc_auc", "mcc")

POLICIES = ("mixed_k8", "uniform_k8", "event_k8", "random_k8", "full_signal")


def load_summary(policy: str, seed: int = 42) -> dict:
    if policy == "mixed_k8":
        path = STAGE1 / "e4_ctx_concat" / "seeds" / f"seed_{seed}" / "cv_summary.json"
    else:
        path = STAGE / policy / "seeds" / f"seed_{seed}" / "cv_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_oof(policy: str, seed: int = 42) -> dict:
    if policy == "mixed_k8":
        path = STAGE1 / "e4_ctx_concat" / "oof.npz"
    else:
        path = STAGE / policy / "oof.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=False)
    mids = d["measurement_ids"]
    return {
        "phase_probs": d[f"seed_{seed}_phase_probs"].reshape(-1),
        "phase_targets": d[f"seed_{seed}_phase_targets"].reshape(-1),
        "meas_probs": d[f"seed_{seed}_meas_probs"],
        "meas_targets": d[f"seed_{seed}_meas_targets"],
        "phase_mids": np.repeat(mids, 3),
        "meas_mids": mids,
    }


def paired_ci(a: str, b: str, level: str, seed: int = 42) -> dict:
    x = load_oof(a, seed)
    y = load_oof(b, seed)
    return paired_bootstrap_ci(
        model_scores=x[f"{level}_probs"],
        baseline_scores=y[f"{level}_probs"],
        labels=x[f"{level}_targets"],
        measurement_ids=x[f"{level}_mids"],
        n_bootstrap=2000,
        seed=42,
    )


def fmt3(value) -> str:
    return "-" if value is None else (f"{value:.3f}" if np.isfinite(value) else "nan")


def main() -> None:
    benchmark_path = STAGE / "benchmark.json"
    if not benchmark_path.exists():
        raise FileNotFoundError(benchmark_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

    lines = ["# IEEE TIM Stage 2 Sampling-Controlled Evidence Report", ""]
    lines.append("Protocol is identical across rows: TimWindowEncoder(cnn) + "
                 "Attention MIL + context-concat + phase BCE + AdamW + "
                 "StratifiedGroupKFold(5, seed=42). Only the sampling policy changes.")
    lines.append("")
    lines.append("| policy | K | phase PR-AUC | meas PR-AUC | phase ROC | meas ROC | "
                 "phase MCC | meas MCC | params | MACs/meas | GPU b1 ms | CPU b1 ms | "
                 "peak GPU MB |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for policy in POLICIES:
        try:
            s = load_summary(policy, 42)
        except FileNotFoundError as exc:
            lines.append(f"| {policy} | MISSING ({getattr(exc, 'filename', exc)}) |")
            continue
        b = benchmark["policies"].get(policy, {})
        k = 1 if policy == "full_signal" else 8
        lines.append(
            f"| {policy} | {k} | "
            f"{fmt3(s['fold_mean_phase']['pr_auc'])} | {fmt3(s['fold_mean_measurement']['pr_auc'])} | "
            f"{fmt3(s['fold_mean_phase']['roc_auc'])} | {fmt3(s['fold_mean_measurement']['roc_auc'])} | "
            f"{fmt3(s['fold_mean_phase']['mcc'])} | {fmt3(s['fold_mean_measurement']['mcc'])} | "
            f"{b.get('n_params_full_pipeline', '-')} | "
            f"{b.get('macs_per_measurement', '-')} | "
            f"{fmt3(b.get('gpu_batch1_model_components_ms', {}).get('full_pipeline', {}).get('p50_ms')) if b else '-'} | "
            f"{fmt3(b.get('cpu_batch1_model_components_ms', {}).get('full_pipeline', {}).get('p50_ms')) if b else '-'} | "
            f"{b.get('peak_gpu_mb', '-')} |")

    lines += ["", "## Paired cluster bootstrap, seed 42, x2000 (mixed_k8 minus policy)", ""]
    lines.append("| comparison | level | PR-AUC diff median | 95% CI |")
    lines.append("|---|---|---|---|")
    for policy in ("uniform_k8", "event_k8", "random_k8", "full_signal"):
        for level in ("phase", "meas"):
            try:
                d = paired_ci("mixed_k8", policy, level)
                lines.append(
                    f"| mixed_k8 vs {policy} | {level} | {d['diff_median']:.4f} | "
                    f"[{d['diff_lower']:.4f}, {d['diff_upper']:.4f}] |")
            except (FileNotFoundError, KeyError, ValueError) as exc:
                lines.append(f"| mixed_k8 vs {policy} | {level} | MISSING ({exc}) |")

    lines += ["", "## Selection pipeline (CPU, synthetic 800k signal)", ""]
    sel = benchmark["selection_components_cpu_ms"]
    lines.append("| component | p50 ms | p95 ms |")
    lines.append("|---|---|---|")
    for key in ("event_score", "peak_detection", "uniform_k8", "event_k8",
                "mixed_k8", "random_k8", "full_signal"):
        entry = sel.get(key)
        if entry is None:
            continue
        lines.append(f"| {key} | {fmt3(entry.get('p50_ms'))} | {fmt3(entry.get('p95_ms'))} |")

    out_md = STAGE / "report_sampling.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
