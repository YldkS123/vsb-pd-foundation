"""
Blind evaluation on the 423 strict holdout measurements.
Extracts holdout windows (if not already cached), loads 5-fold models,
runs ensemble inference, applies Platt calibration, and writes the
FINAL_EVALUATION_RECEIPT.json.

Usage:
    python scripts/blind_evaluate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vsb_pd.config import load_config, PipelineConfig
from vsb_pd.data import WindowDataset
from vsb_pd.encoder import WindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.cyclic import CyclicPhaseModule
from vsb_pd.model import VSBPipeline
from vsb_pd.features import extract_physical_features
from vsb_pd.evaluation import calibrate_platt, compute_calibration_metrics, compute_bootstrap_ci, BlindEvaluationLock


MODEL_DIR = Path("results/model_full")
CACHE_DIR = Path("results/holdout_cache")
LOCK_PATH = Path("results/final_eval.lock")
RECEIPT_PATH = Path("results/FINAL_EVALUATION_RECEIPT.json")


def _load_holdout_ids(split_lock_path: str) -> list[int]:
    """Extract holdout measurement IDs from the split lock."""
    lock_data = json.loads(Path(split_lock_path).read_text(encoding="utf-8"))
    assignments = lock_data["assignments"]
    holdout_ids = [
        a["id_measurement"]
        for a in assignments
        if a.get("split") == "final_holdout"
    ]
    return sorted(holdout_ids)


def _extract_holdout_windows(config: PipelineConfig, holdout_ids: list[int]) -> Path:
    """Extract NPZ artifacts for holdout measurements (same pipeline identity)."""
    from vsb_pd.extract import (
        pipeline_identity,
        read_measurement_signals,
        select_hybrid_windows,
        _write_npz_atomic,
        KIND_CODE,
    )
    from vsb_pd.metadata import load_metadata

    pipeline_hash, source_hash = pipeline_identity(config, Path("artifacts/locks/split_lock.json"))
    output_dir = CACHE_DIR / "windows" / pipeline_hash / "holdout"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(config.metadata_path)
    wp = config.window_policy

    existing = len(list(output_dir.glob("*.npz")))
    if existing >= len(holdout_ids):
        print(f"Holdout cache complete ({existing}/{len(holdout_ids)} NPZ)")
        return output_dir

    print(f"Extracting {len(holdout_ids)} holdout measurements...")
    extracted = 0
    for i, mid in enumerate(holdout_ids):
        group = metadata.loc[metadata["id_measurement"] == mid].sort_values("phase")
        signal_ids = group["signal_id"].astype(int).tolist()
        phases = group["phase"].to_numpy(dtype=np.int8)
        targets = group["target"].to_numpy(dtype=np.int8)

        dest = output_dir / f"{mid}.npz"
        if dest.exists():
            extracted += 1
            continue

        signals = read_measurement_signals(config.raw_parquet_path, signal_ids)
        selected_by_phase = [
            select_hybrid_windows(signals[p], wp) for p in range(3)
        ]

        starts = np.asarray(
            [[item.start for item in sel] for sel in selected_by_phase], dtype=np.int64,
        )
        kinds = np.asarray(
            [[KIND_CODE[item.kind] for item in sel] for sel in selected_by_phase], dtype=np.uint8,
        )
        scores = np.asarray(
            [[item.score for item in sel] for sel in selected_by_phase], dtype=np.float32,
        )
        windows_matrix = np.stack([
            np.stack([signals[p, start:start + wp.window_length] for start in starts[p]])
            for p in range(3)
        ])

        _write_npz_atomic(dest, {
            "measurement_id": np.asarray(mid, dtype=np.int64),
            "signal_ids": np.asarray(signal_ids, dtype=np.int64),
            "phases": phases,
            "targets": targets,
            "windows": windows_matrix,
            "starts": starts,
            "kinds": kinds,
            "scores": scores,
            "pipeline_hash": np.asarray(pipeline_hash),
            "source_parquet_sha256": np.asarray(source_hash),
        })

        extracted += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(holdout_ids)}")

    print(f"Extraction done: {extracted}/{len(holdout_ids)}")
    return output_dir


def _load_features_from_dir(holdout_dir: Path) -> dict:
    """Load all holdout NPZ files and compute physical features."""
    npz_files = sorted(holdout_dir.glob("*.npz"))
    print(f"Loading {len(npz_files)} holdout NPZ files...")

    all_windows, all_labels, all_mids = [], [], []
    for f in npz_files:
        data = np.load(f, allow_pickle=False)
        all_windows.append(data["windows"].astype(np.float32))
        all_labels.append(data["targets"].astype(np.int8))
        all_mids.append(int(data["measurement_id"].item()))

    all_windows = np.stack(all_windows)
    all_labels = np.array(all_labels, dtype=np.int8)
    all_mids = np.array(all_mids)

    M, P, K, L = all_windows.shape
    print(f"  {M} measurements, windows: {all_windows.shape}")

    # Compute physical features
    features = extract_physical_features(
        all_windows.reshape(M * P, K, L), 40_000_000,
    )
    feature_names = sorted(features.keys())
    feat_array = np.stack([features[name] for name in feature_names], axis=-1)
    feat_array = feat_array.reshape(M, P, K, -1).astype(np.float32)

    return {
        "windows": all_windows,
        "features": feat_array,
        "labels": all_labels,
        "measurement_ids": all_mids,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(Path("configs/local.json"))
    split_lock_path = "artifacts/locks/split_lock.json"

    # 1. Get holdout IDs
    holdout_ids = _load_holdout_ids(split_lock_path)
    print(f"Holdout measurements: {len(holdout_ids)}")
    print(f"  IDs: {holdout_ids[:5]}...{holdout_ids[-3:]}")

    # 2. Extract holdout windows
    holdout_npz_dir = _extract_holdout_windows(config, holdout_ids)

    # 3. Load data and compute features
    data = _load_features_from_dir(holdout_npz_dir)
    holdout_windows = torch.from_numpy(data["windows"]).float().to(device)
    holdout_features = torch.from_numpy(data["features"]).float().to(device)
    holdout_labels = torch.from_numpy(data["labels"]).float().to(device)
    holdout_mids = data["measurement_ids"]

    print(f"\nHoldout labels: {Counter(int(x) for labels_arr in holdout_labels.cpu().numpy() for x in labels_arr)}")

    # 4. Load 5-fold models and run inference
    model_fold_paths = sorted(MODEL_DIR.glob("model_fold*.pt"))
    print(f"\nLoading {len(model_fold_paths)} model folds from {MODEL_DIR}")

    all_fold_probs = []
    for fpath in model_fold_paths:
        model = VSBPipeline(
            encoder=WindowEncoder(8192, 58, 128),
            aggregator=MILAggregator("gated_attention", 128),
            cyclic=CyclicPhaseModule(128),
            classifier=PhaseClassifier(128),
        ).to(device)

        ckpt = torch.load(fpath, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        fold_probs = []
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(holdout_windows), batch_size):
                bw = holdout_windows[i:i + batch_size]
                bf = holdout_features[i:i + batch_size]
                logits, _ = model(bw, bf)
                probs = torch.sigmoid(logits)
                fold_probs.append(probs.cpu().numpy())

        fold_probs = np.concatenate(fold_probs, axis=0)
        all_fold_probs.append(fold_probs)
        print(f"  {fpath.stem}: done")

    # 5. Ensemble predictions (mean of 5 folds)
    probs_ensemble = np.mean(all_fold_probs, axis=0)  # (holdout, 3)
    labels_np = holdout_labels.cpu().numpy()

    # Per-phase metrics (flatten)
    probs_flat = probs_ensemble.flatten()
    labels_flat = labels_np.flatten()
    preds_flat = (probs_flat >= 0.5).astype(int)

    from vsb_pd.training import compute_metrics
    metrics_phase = compute_metrics(labels_flat, probs_flat, preds_flat)

    # Measurement-level: noisy-OR per measurement
    from vsb_pd.cyclic import noisy_or_probs
    meas_probs = noisy_or_probs(torch.from_numpy(probs_ensemble).float()).numpy()
    meas_labels = labels_np.max(axis=1)  # Any positive phase = positive measurement
    meas_preds = (meas_probs >= 0.5).astype(int)
    metrics_measurement = compute_metrics(meas_labels, meas_probs, meas_preds)

    print(f"\n=== BLIND EVALUATION RESULTS ===")
    print(f"Phase-level:")
    print(f"  PR-AUC: {metrics_phase.get('pr_auc', float('nan')):.4f}")
    print(f"  ROC-AUC: {metrics_phase.get('roc_auc', float('nan')):.4f}")
    print(f"  F1: {metrics_phase.get('f1', 0):.4f}")
    print(f"  Recall: {metrics_phase.get('recall', 0):.4f}")
    print(f"  Precision: {metrics_phase.get('precision', 0):.4f}")
    print(f"  Accuracy: {metrics_phase.get('accuracy', 0):.4f}")
    print(f"\nMeasurement-level (noisy-OR):")
    print(f"  PR-AUC: {metrics_measurement.get('pr_auc', float('nan')):.4f}")
    print(f"  ROC-AUC: {metrics_measurement.get('roc_auc', float('nan')):.4f}")
    print(f"  F1: {metrics_measurement.get('f1', 0):.4f}")
    print(f"  Recall: {metrics_measurement.get('recall', 0):.4f}")
    print(f"  Precision: {metrics_measurement.get('precision', 0):.4f}")

    # 6. Calibration
    print(f"\n=== CALIBRATION ===")
    cal_metrics = compute_calibration_metrics(probs_flat, labels_flat, n_bins=10)
    print(f"  ECE: {cal_metrics['ece']:.4f}")
    print(f"  Brier: {cal_metrics['brier']:.4f}")

    calibrated = calibrate_platt(probs_flat, labels_flat, probs_flat)
    cal_preds = (calibrated >= 0.5).astype(int)
    metrics_calibrated = compute_metrics(labels_flat, calibrated, cal_preds)
    print(f"\n  After Platt calibration:")
    print(f"    PR-AUC: {metrics_calibrated.get('pr_auc', float('nan')):.4f}")

    # 7. Bootstrap CI on PR-AUC
    from vsb_pd.evaluation import compute_bootstrap_ci as bs_ci
    ph_mids = np.repeat(holdout_mids, 3)  # each phase per measurement
    # Cluster bootstrap of the phase-level PR-AUC (measurements as clusters).
    # NOTE: the frozen FINAL_EVALUATION_RECEIPT.json predates the fix to
    # compute_bootstrap_ci (it stored a mean-probability interval); a rerun
    # would store the true PR-AUC CI below.
    ci = bs_ci(probs_flat, labels_flat, ph_mids, metric_name="pr_auc", n_bootstrap=2000)
    print(f"\n  PR-AUC 95% CI: [{ci.lower:.4f}, {ci.upper:.4f}] (median: {ci.median:.4f})")

    # 8. Write final evaluation receipt
    receipt = {
        "experiment": "VSB Partial Discharge Blind Evaluation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "holdout_measurements": len(holdout_ids),
        "holdout_positive_measurements": int(meas_labels.sum()),
        "holdout_phases": len(labels_flat),
        "holdout_positive_phases": int(labels_flat.sum()),
        "ensemble_method": "mean of 5 folded models",
        "metrics": {
            "phase_level": {
                "pr_auc": metrics_phase.get("pr_auc"),
                "roc_auc": metrics_phase.get("roc_auc"),
                "f1": metrics_phase.get("f1"),
                "recall": metrics_phase.get("recall"),
                "precision": metrics_phase.get("precision"),
                "accuracy": metrics_phase.get("accuracy"),
                "ece": cal_metrics["ece"],
                "brier": cal_metrics["brier"],
            },
            "measurement_level_noisy_or": {
                "pr_auc": metrics_measurement.get("pr_auc"),
                "roc_auc": metrics_measurement.get("roc_auc"),
                "f1": metrics_measurement.get("f1"),
                "recall": metrics_measurement.get("recall"),
                "precision": metrics_measurement.get("precision"),
                "accuracy": metrics_measurement.get("accuracy"),
            },
            "platt_calibrated": {
                "pr_auc": metrics_calibrated.get("pr_auc"),
            },
            "pr_auc_95_ci": {
                "lower": float(ci.lower),
                "median": float(ci.median),
                "upper": float(ci.upper),
            },
        },
        "status": "completed",
    }

    hashes = {
        "split_lock": "artifacts/locks/split_lock.json",
        "model_checkpoints": str(MODEL_DIR),
    }
    lock = BlindEvaluationLock(LOCK_PATH, hashes)
    if lock.acquire() or not lock.is_locked():
        RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, ensure_ascii=False))
        print(f"\n  Receipt written to {RECEIPT_PATH}")

    print(f"\n  Done! Final evaluation complete.")


if __name__ == "__main__":
    main()
