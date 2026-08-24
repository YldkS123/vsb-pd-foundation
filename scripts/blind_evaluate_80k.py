# -*- coding: utf-8 -*-
"""Blind evaluation of the final 80k mainline on the frozen VSB holdout.

The paper's proposed final model is the 80,113-param mainline
(encoder=cnn, MIL=attention, phase=mean), while the original
FINAL_EVALUATION_RECEIPT.json (2026-07-30) was produced with the older ~203k
reference model in results/model_full.  This script evaluates the 80k mainline
once on the same frozen 423-measurement holdout, in two optional protocols:

  --checkpoint PATH : single final dev-only checkpoint
                      (cnn80k_vsbdev_seed42.pt, trained on features_full)
  --model-dir DIR   : mean of 5 dev-fold checkpoints (same ensemble protocol
                      as the 203k reference receipt; trained on the paper's
                      features_policy_mixed_k8 cache)

The old receipt and lock are left untouched; each protocol writes its own
receipt/lock with checkpoint hashes.

Protocol (same as blind_evaluate.py except the model):
  - same frozen artifacts/locks/split_lock.json
  - same holdout window extraction cache (read-only)
  - same phase-level / measurement-level (noisy-OR) metrics
  - same ECE/Brier calibration diagnostics and cluster bootstrap CI
  - Platt scaling is fit on the matching 80k mainline 5-fold dev OOF,
    never on blind data

Usage:
    python scripts/blind_evaluate_80k.py --checkpoint results/external/checkpoints/cnn80k_vsbdev_seed42.pt
    python scripts/blind_evaluate_80k.py --model-dir results/ablations/dev_k8_blind_ckpts/enc_cnn__mil_attention__ph_mean
    python scripts/blind_evaluate_80k.py --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vsb_pd.encoder import WindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.cyclic import PhaseInteractionModule
from vsb_pd.model import VSBPipeline
from vsb_pd.training import compute_metrics
from vsb_pd.evaluation import (
    calibrate_platt,
    compute_bootstrap_ci,
    compute_calibration_metrics,
    BlindEvaluationLock,
)
from blind_evaluate import (
    _load_holdout_ids,
    _extract_holdout_windows,
    _load_features_from_dir,
    CACHE_DIR,
)

CHECKPOINT = ROOT / "results" / "external" / "checkpoints" / "cnn80k_vsbdev_seed42.pt"
RECEIPT_PATH = ROOT / "results" / "FINAL_EVALUATION_RECEIPT_80k.json"
LOCK_PATH = ROOT / "results" / "final_eval_80k.lock"
ENSEMBLE_RECEIPT_PATH = ROOT / "results" / "FINAL_EVALUATION_RECEIPT_80k_ensemble.json"
ENSEMBLE_LOCK_PATH = ROOT / "results" / "final_eval_80k_ensemble.lock"
SPLIT_LOCK_PATH = ROOT / "artifacts" / "locks" / "split_lock.json"
DEV_OOF_PATH = (
    ROOT
    / "results"
    / "ablations"
    / "dev_k8_combo"
    / "enc_cnn__mil_attention__ph_mean"
    / "oof.npz"
)
PREDICTIONS_PATH = ROOT / "results" / "blind_80k_predictions.npz"
ENSEMBLE_PREDICTIONS_PATH = ROOT / "results" / "blind_80k_ensemble_predictions.npz"

EXPECTED_PARAMS = 80113
EXPECTED_CONFIG = {"encoder": "cnn", "mil": "attention", "phase": "mean"}
EXPECTED_CONFIG_NAME = "enc_cnn__mil_attention__ph_mean"
N_FOLD_MODELS = 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_final_model() -> VSBPipeline:
    return VSBPipeline(
        encoder=WindowEncoder(8192, 58, 128, branch="cnn"),
        aggregator=MILAggregator("attention", 128),
        cyclic=PhaseInteractionModule("mean", 128),
        classifier=PhaseClassifier(128),
    )


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_platt_from_dev_oof(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened (probs, labels) from the 80k mainline dev OOF."""
    d = np.load(path, allow_pickle=True)
    probs = d["phase_probs"].reshape(-1)
    labels = d["phase_targets"].reshape(-1)
    return probs, labels


def load_holdout_data(cache_dir: Path, device: torch.device) -> tuple:
    data = _load_features_from_dir(cache_dir)
    windows = torch.from_numpy(data["windows"]).float().to(device)
    features = torch.from_numpy(data["features"]).float().to(device)
    labels = torch.from_numpy(data["labels"]).float().to(device)
    return data, windows, features, labels


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    windows: torch.Tensor,
    features: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probs = []
    for i in range(0, len(windows), batch_size):
        logits, _ = model(windows[i : i + batch_size], features[i : i + batch_size])
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def load_single_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    model = build_final_model().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    ckpt_config = ckpt.get("config", {})
    trained_on = ckpt.get("trained_on", "unknown")
    ckpt_params = int(ckpt.get("n_params", -1))
    actual_params = count_params(model)
    if ckpt_config != EXPECTED_CONFIG:
        raise SystemExit(f"unexpected checkpoint config: {ckpt_config}")
    if trained_on != "vsb_development_full":
        raise SystemExit(f"checkpoint must be dev-only, got: {trained_on}")
    if actual_params != EXPECTED_PARAMS or ckpt_params != EXPECTED_PARAMS:
        raise SystemExit(
            f"expected {EXPECTED_PARAMS} params, got {actual_params} / {ckpt_params}"
        )
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
    return model, ckpt


def load_fold_ensemble(model_dir: Path, device: torch.device) -> tuple[list[torch.nn.Module], list[dict]]:
    fold_paths = sorted(model_dir.glob("model_fold*.pt"))
    if len(fold_paths) != N_FOLD_MODELS:
        raise SystemExit(
            f"expected {N_FOLD_MODELS} fold checkpoints in {model_dir}, got {len(fold_paths)}"
        )
    models, metas = [], []
    for fpath in fold_paths:
        model = build_final_model().to(device)
        ckpt = torch.load(fpath, map_location=device, weights_only=True)
        config_name = ckpt.get("config", "")
        if config_name != EXPECTED_CONFIG_NAME:
            raise SystemExit(f"unexpected fold config {config_name!r} in {fpath}")
        if count_params(model) != EXPECTED_PARAMS:
            raise SystemExit(f"unexpected param count in {fpath}")
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
        if missing or unexpected:
            raise SystemExit(f"state_dict mismatch in {fpath}: {missing} / {unexpected}")
        models.append(model)
        metas.append({"path": str(fpath), "sha256": sha256_file(fpath)})
        print(f"  loaded {fpath.stem} ({count_params(model)} params)")
    return models, metas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--receipt", type=Path, default=None)
    ap.add_argument("--lock", type=Path, default=None)
    ap.add_argument("--calib-oof", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if (args.checkpoint is None) == (args.model_dir is None):
        raise SystemExit("exactly one of --checkpoint / --model-dir is required")
    if args.checkpoint is not None and not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.model_dir is not None and not args.model_dir.exists():
        raise SystemExit(f"model dir not found: {args.model_dir}")

    is_ensemble = args.model_dir is not None
    receipt_path = args.receipt or (ENSEMBLE_RECEIPT_PATH if is_ensemble else RECEIPT_PATH)
    lock_path = args.lock or (ENSEMBLE_LOCK_PATH if is_ensemble else LOCK_PATH)
    calib_oof_path = args.calib_oof or DEV_OOF_PATH
    if not calib_oof_path.exists():
        raise SystemExit(f"dev OOF not found for calibration: {calib_oof_path}")

    split_sha = sha256_text(SPLIT_LOCK_PATH)
    print(f"split_lock sha256: {split_sha[:16]}...")
    print(f"protocol: {'5-fold mean ensemble' if is_ensemble else 'single checkpoint'}")

    device = torch.device(args.device)

    # 1. Holdout IDs and window extraction (cache must already exist)
    holdout_ids = _load_holdout_ids(str(SPLIT_LOCK_PATH))
    print(f"Holdout measurements: {len(holdout_ids)}")

    import vsb_pd.config as vcfg
    from vsb_pd.extract import pipeline_identity
    config = vcfg.load_config(ROOT / "configs" / "local.json")
    pipeline_hash, _ = pipeline_identity(config, SPLIT_LOCK_PATH)
    cache_dir = CACHE_DIR / "windows" / pipeline_hash / "holdout"
    cached = list(cache_dir.glob("*.npz")) if cache_dir.exists() else []
    print(f"Holdout cache files: {len(cached)}")
    if len(cached) < len(holdout_ids):
        raise SystemExit(
            "holdout window cache incomplete; run scripts/blind_evaluate.py first "
            "(do not modify the split lock)"
        )

    data, holdout_windows, holdout_features, holdout_labels = load_holdout_data(
        cache_dir, device
    )
    holdout_mids = data["measurement_ids"]

    # 2. Load model(s)
    if is_ensemble:
        models, fold_metas = load_fold_ensemble(args.model_dir, device)
        print(f"\nRunning fold inference on {len(holdout_windows)} holdout measurements...")
        t0 = time.time()
        all_fold_probs = [
            predict_probs(m, holdout_windows, holdout_features, args.batch_size)
            for m in models
        ]
        probs_ensemble = np.mean(all_fold_probs, axis=0)
        model_meta = {
            "n_folds": len(models),
            "params_per_fold": EXPECTED_PARAMS,
            "config": EXPECTED_CONFIG,
            "fold_checkpoints": fold_metas,
        }
        print(f"  ensemble inference done in {time.time() - t0:.1f}s")
    else:
        model, ckpt = load_single_checkpoint(args.checkpoint, device)
        print(f"checkpoint config: {ckpt.get('config')}")
        print(
            f"trained_on: {ckpt.get('trained_on')}, seed: {ckpt.get('seed')}, "
            f"epochs: {ckpt.get('epochs')}"
        )
        print(f"\nRunning inference on {len(holdout_windows)} holdout measurements...")
        t0 = time.time()
        probs_ensemble = predict_probs(model, holdout_windows, holdout_features, args.batch_size)
        print(f"  inference done in {time.time() - t0:.1f}s")
        model_meta = {
            "name": args.checkpoint.stem,
            "params": EXPECTED_PARAMS,
            "config": EXPECTED_CONFIG,
            "seed": int(ckpt.get("seed", -1)),
            "epochs": int(ckpt.get("epochs", -1)),
            "trained_on": ckpt.get("trained_on"),
            "checkpoint_sha256": sha256_file(args.checkpoint),
        }

    labels_np = holdout_labels.cpu().numpy()
    print(f"\nHoldout labels: {Counter(int(x) for labels_arr in labels_np for x in labels_arr)}")

    # 3. Phase-level metrics
    probs_flat = probs_ensemble.flatten()
    labels_flat = labels_np.flatten()
    preds_flat = (probs_flat >= 0.5).astype(int)
    metrics_phase = compute_metrics(labels_flat, probs_flat, preds_flat)

    # 4. Measurement-level noisy-OR
    from vsb_pd.cyclic import noisy_or_probs
    meas_probs = noisy_or_probs(torch.from_numpy(probs_ensemble).float()).numpy()
    meas_labels = labels_np.max(axis=1)
    meas_preds = (meas_probs >= 0.5).astype(int)
    metrics_measurement = compute_metrics(meas_labels, meas_probs, meas_preds)

    print(f"\n=== BLIND EVALUATION RESULTS (80k mainline, {('ensemble' if is_ensemble else 'single')}) ===")
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

    # 5. Calibration (ECE/Brier raw; Platt fit on dev OOF, never on blind data)
    cal_metrics = compute_calibration_metrics(probs_flat, labels_flat, n_bins=10)
    print(f"\nECE: {cal_metrics['ece']:.4f}, Brier: {cal_metrics['brier']:.4f}")

    oof_probs, oof_labels = load_platt_from_dev_oof(calib_oof_path)
    calibrated = calibrate_platt(oof_probs, oof_labels, probs_flat)
    cal_preds = (calibrated >= 0.5).astype(int)
    metrics_calibrated = compute_metrics(labels_flat, calibrated, cal_preds)
    print(f"Platt (fit on dev OOF): PR-AUC {metrics_calibrated.get('pr_auc', float('nan')):.4f}")

    # 6. Cluster bootstrap CI on phase-level and measurement-level PR-AUC
    ph_mids = np.repeat(holdout_mids, 3)
    ci_phase = compute_bootstrap_ci(
        probs_flat, labels_flat, ph_mids, metric_name="pr_auc", n_bootstrap=2000
    )
    ci_meas = compute_bootstrap_ci(
        meas_probs, meas_labels, holdout_mids, metric_name="pr_auc", n_bootstrap=2000
    )
    print(
        f"Phase PR-AUC 95% CI: [{ci_phase.lower:.4f}, {ci_phase.upper:.4f}] "
        f"(median {ci_phase.median:.4f})"
    )
    print(
        f"Measurement PR-AUC 95% CI: [{ci_meas.lower:.4f}, {ci_meas.upper:.4f}] "
        f"(median {ci_meas.median:.4f})"
    )

    # 7. Save predictions and one-shot receipt (old receipt untouched)
    pred_path = ENSEMBLE_PREDICTIONS_PATH if is_ensemble else PREDICTIONS_PATH
    np.savez(
        pred_path,
        measurement_ids=holdout_mids,
        phase_probs=probs_ensemble,
        phase_targets=labels_np,
        meas_probs=meas_probs,
        meas_labels=meas_labels,
        ensemble=str(is_ensemble),
    )
    print(f"\nPredictions saved to {pred_path}")

    receipt = {
        "experiment": "VSB Partial Discharge Blind Evaluation (final 80k mainline)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_meta,
        "protocol": (
            "mean of 5 folded models (same protocol as 203k reference receipt)"
            if is_ensemble
            else "single final dev-only checkpoint, frozen 423-measurement holdout"
        ),
        "holdout_measurements": len(holdout_ids),
        "holdout_positive_measurements": int(meas_labels.sum()),
        "holdout_phases": len(labels_flat),
        "holdout_positive_phases": int(labels_flat.sum()),
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
            "platt_calibrated_on_dev_oof": {
                "pr_auc": metrics_calibrated.get("pr_auc"),
            },
            "phase_pr_auc_95_ci": {
                "lower": float(ci_phase.lower),
                "median": float(ci_phase.median),
                "upper": float(ci_phase.upper),
            },
            "measurement_pr_auc_95_ci": {
                "lower": float(ci_meas.lower),
                "median": float(ci_meas.median),
                "upper": float(ci_meas.upper),
            },
        },
        "hashes": {
            "split_lock": {"path": str(SPLIT_LOCK_PATH), "sha256": split_sha},
            "models": model_meta,
        },
        "status": "completed",
    }

    lock = BlindEvaluationLock(lock_path, receipt["hashes"])
    if lock.acquire() or not lock.is_locked():
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False))
        print(f"\nReceipt written to {receipt_path}")
    else:
        print(f"\nLock exists at {lock_path}; receipt not overwritten.")

    print("\nDone. Old 203k receipt and lock are untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
