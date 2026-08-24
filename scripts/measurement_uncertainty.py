# -*- coding: utf-8 -*-
"""Measurement-level uncertainty analysis for the locked dev OOF predictions.

Produces, for the 80k mainline and every deep/published-PD baseline on the same
K=8 mixed-window folds (seed 42):
  - phase- and measurement-level PR-AUC / ROC-AUC cluster bootstrap 95% CI
    (measurements as resampling units, stratified by measurement label);
  - paired bootstrap 95% CI on the mainline-minus-baseline difference;
  - calibration diagnostics (ECE, Brier) on OOF;
  - threshold-aware metrics (0.5 / max-MCC / recall>=0.5 / recall>=0.8)
    with thresholds selected on OOF only.

The blind holdout is never touched by this analysis.

Usage:
  python scripts/measurement_uncertainty.py [--out results/measurement_uncertainty.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vsb_pd.evaluation import compute_bootstrap_ci, paired_bootstrap_ci, compute_calibration_metrics
from compare_baselines import full_metrics

N_BOOT = 2000
SEED = 42

METHODS = [
    {
        "key": "mainline_80k",
        "label": "80k mainline (cnn+attention+mean)",
        "oof": "results/ablations/dev_k8_combo/enc_cnn__mil_attention__ph_mean/oof.npz",
    },
    {
        "key": "resnet1d",
        "label": "ResNet1D (deep baseline)",
        "oof": "results/dl_baselines_k8_fair/dl_baselines_k8_fair/enc_resnet1d__mil_mean__ph_max/oof.npz",
    },
    {
        "key": "inception",
        "label": "InceptionTime (deep baseline)",
        "oof": "results/dl_baselines_k8_fair/dl_baselines_k8_fair/enc_inception__mil_mean__ph_max/oof.npz",
    },
    {
        "key": "simple_cnn",
        "label": "simple_cnn (deep baseline)",
        "oof": "results/dl_baselines_k8_fair/dl_baselines_k8_fair/enc_simple_cnn__mil_mean__ph_max/oof.npz",
    },
    {
        "key": "published_tfcnn_zheng2022",
        "label": "Zheng TF-CNN 2022 (adapted published PD baseline)",
        "oof": "results/sota_baselines/sota_tfcnn_zheng2022/oof.npz",
    },
    {
        "key": "published_cnn_qsvm_fei2024",
        "label": "Fei CNN+QSVM 2024 (adapted published PD baseline)",
        "oof": "results/sota_baselines/sota_cnn_qsvm_fei2024/oof.npz",
    },
]


def _ci_dict(ci) -> dict:
    return asdict(ci)


def analyze_method(root: Path, method: dict) -> dict:
    d = np.load(root / method["oof"], allow_pickle=True)
    probs = np.asarray(d["phase_probs"], dtype=np.float64)      # (M, 3)
    targets = np.asarray(d["phase_targets"], dtype=np.float64)  # (M, 3)
    mids = np.asarray(d["measurement_ids"])

    p_phase = probs.ravel()
    y_phase = targets.ravel().astype(int)
    ids_phase = np.repeat(mids, 3)

    p_meas = 1.0 - np.prod(1.0 - probs, axis=1)
    y_meas = targets.max(axis=1).astype(int)

    entry = {"key": method["key"], "label": method["label"], "n_measurements": int(len(mids))}

    # Bootstrap CIs (phase and measurement, PR-AUC and ROC-AUC)
    entry["phase"] = {
        "bootstrap_pr_auc_95ci": _ci_dict(compute_bootstrap_ci(
            p_phase, y_phase, ids_phase, metric_name="pr_auc", n_bootstrap=N_BOOT, seed=SEED)),
        "bootstrap_roc_auc_95ci": _ci_dict(compute_bootstrap_ci(
            p_phase, y_phase, ids_phase, metric_name="roc_auc", n_bootstrap=N_BOOT, seed=SEED)),
        "full_metrics": full_metrics(y_phase, p_phase),
        "calibration": compute_calibration_metrics(p_phase, y_phase),
    }
    entry["measurement"] = {
        "bootstrap_pr_auc_95ci": _ci_dict(compute_bootstrap_ci(
            p_meas, y_meas, mids, metric_name="pr_auc", n_bootstrap=N_BOOT, seed=SEED)),
        "bootstrap_roc_auc_95ci": _ci_dict(compute_bootstrap_ci(
            p_meas, y_meas, mids, metric_name="roc_auc", n_bootstrap=N_BOOT, seed=SEED)),
        "full_metrics": full_metrics(y_meas, p_meas),
        "calibration": compute_calibration_metrics(p_meas, y_meas),
    }
    entry["_arrays"] = {
        "phase_probs": p_phase, "phase_targets": y_phase, "ids_phase": ids_phase,
        "meas_probs": p_meas, "meas_targets": y_meas, "mids": mids,
    }
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/measurement_uncertainty.json")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    arrays = {}
    for method in METHODS:
        print(f"Analyzing {method['key']} ...", flush=True)
        entry = analyze_method(ROOT, method)
        arrays[entry["key"]] = entry.pop("_arrays")
        entries.append(entry)

    # Paired bootstrap: mainline minus each other method, same measurement clusters.
    main = arrays["mainline_80k"]
    paired = {}
    for key in arrays:
        if key == "mainline_80k":
            continue
        other = arrays[key]
        paired[key] = {
            "phase": paired_bootstrap_ci(
                main["phase_probs"], other["phase_probs"],
                main["phase_targets"], main["ids_phase"],
                n_bootstrap=N_BOOT, seed=SEED),
            "measurement": paired_bootstrap_ci(
                main["meas_probs"], other["meas_probs"],
                main["meas_targets"], main["mids"],
                n_bootstrap=N_BOOT, seed=SEED),
        }
        print(f"  paired vs {key}: phase diff={paired[key]['phase']['diff_median']:.4f} "
              f"meas diff={paired[key]['measurement']['diff_median']:.4f}", flush=True)

    report = {
        "protocol": (
            "dev OOF only; same StratifiedGroupKFold(5, seed=42) measurement splits; "
            "measurements are the resampling unit; blind holdout untouched"
        ),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "methods": entries,
        "paired_vs_mainline": paired,
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
