# -*- coding: utf-8 -*-
"""Regenerate and persist E4 fold checkpoints for the Harvard blind test.

The original Stage 1 E4 run (results/stage1_tim/e4_ctx_concat, seed 42) did
not persist fold weights.  This script regenerates the five development-fold
checkpoints under the exact frozen protocol used by stage1_tim_runner.py:

  encoder=TimWindowEncoder("cnn", 8192, 58, 128)
  MIL=attention, phase interaction=context_concat, phase BCE (lambda_m=0)
  AdamW(lr=1e-3, weight_decay=1e-4), batch=64, epochs=40,
  early stopping on val phase PR-AUC (patience=15, min_delta=0.001),
  StratifiedGroupKFold(5, seed=42), per-fold RNG seed = 42 + fold_index.

Only the 2481-measurement development cache is read.  The Harvard blind data
is never opened here; checkpoints are written before the blind evaluation and
must not be modified afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stage1_tim_runner import build_pipeline, load_data, oof_metrics, predict, train_fold  # noqa: E402
from vsb_pd.cyclic import PhaseCyclicLoss  # noqa: E402
from vsb_pd.training import make_stratified_group_folds  # noqa: E402

CACHE = ROOT / "results" / "cached_features" / "features_policy_mixed_k8.npz"
OUT_DIR = ROOT / "results" / "e4_harvard_blind" / "checkpoints"
VERIFY_DIR = ROOT / "results" / "e4_harvard_blind" / "dev_oof_verify"
N_FOLDS = 5
SEED = 42
EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 15
CONFIG_NAME = "enc_cnn__mil_attention__ph_ctx_concat"
N_PARAMS = 113265


def save_fold_checkpoint(model: torch.nn.Module, fold: int, epochs_trained: int,
                         best_pr: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "config": CONFIG_NAME,
        "experiment": "e4_ctx_concat",
        "fold": int(fold),
        "seed": int(SEED),
        "n_params": N_PARAMS,
        "trained_on": "vsb_development_2481_mixed_k8",
        "protocol": "frozen_e4_stage1_protocol",
        "best_val_phase_pr_auc": round(float(best_pr), 6),
        "epochs_trained": int(epochs_trained),
        "purpose": "one-time Harvard blind test (created before blind evaluation)",
    }
    torch.save(ckpt, path)
    print(f"  saved {path.name} (best_val_pr={best_pr:.4f}, epochs={epochs_trained})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    if not CACHE.exists():
        raise SystemExit(f"dev cache not found: {CACHE}")

    windows, feat, labels, mids = load_data(CACHE, None)
    labels_np = labels.numpy()
    M = len(mids)
    if tuple(windows.shape) != (2481, 3, 8, 8192):
        raise SystemExit(f"unexpected dev cache shape: {windows.shape}")

    folds = make_stratified_group_folds(
        mids, labels_np.sum(axis=1).astype(int), n_splits=N_FOLDS, seed=SEED,
    )
    print(f"data={M} measurements, folds={[(len(tr), len(va)) for tr, va in folds[:args.max_folds]]}")
    print(f"device={args.device}")

    device = torch.device(args.device)
    oof_phase_p = np.zeros((M, 3), dtype=np.float64)
    oof_phase_t = np.zeros((M, 3), dtype=np.float64)
    oof_meas_p = np.zeros(M, dtype=np.float64)
    oof_meas_t = np.zeros(M, dtype=np.float64)
    fold_assign = np.zeros(M, dtype=np.int8)
    fold_rows = []

    for fi, (tr, va) in enumerate(folds[:args.max_folds]):
        t0 = time.time()
        model = build_pipeline("cnn", "context_concat").to(device)
        criterion = PhaseCyclicLoss(lambda_m=0.0)
        epochs_trained, best_pr = train_fold(
            model, criterion, windows, feat, labels, labels_np, tr, va,
            BATCH_SIZE, EPOCHS, PATIENCE, device, SEED + fi,
        )
        ckpt_path = OUT_DIR / f"model_fold{fi + 1}.pt"
        if ckpt_path.exists():
            raise SystemExit(
                f"refusing to overwrite existing checkpoint: {ckpt_path}"
            )
        save_fold_checkpoint(model, fi + 1, epochs_trained, best_pr, ckpt_path)

        final_probs, final_targets = predict(
            model, windows, feat, va, BATCH_SIZE, device, labels_np,
        )
        meas_probs = 1.0 - np.prod(1.0 - final_probs, axis=1)
        meas_targets = final_targets.max(axis=1)
        oof_phase_p[va] = final_probs
        oof_phase_t[va] = final_targets
        oof_meas_p[va] = meas_probs
        oof_meas_t[va] = meas_targets
        fold_assign[va] = fi
        fold_rows.append({
            "fold": fi + 1,
            "epochs_trained": int(epochs_trained),
            "best_val_phase_pr_auc": round(float(best_pr), 6),
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"  fold {fi + 1}: best_val_pr={best_pr:.4f} "
              f"({fold_rows[-1]['elapsed_s']}s)")

    summary = oof_metrics(oof_phase_p, oof_phase_t, oof_meas_p, oof_meas_t, fold_assign)
    summary.update({
        "config_name": "e4_ctx_concat",
        "encoder": "cnn",
        "interaction": "context_concat",
        "lambda_m": 0.0,
        "seed": SEED,
        "n_measurements": M,
        "n_params": N_PARAMS,
        "fold_rows": fold_rows,
    })
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    (VERIFY_DIR / "cv_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(
        VERIFY_DIR / "oof.npz",
        phase_probs=oof_phase_p, phase_targets=oof_phase_t,
        meas_probs=oof_meas_p, meas_targets=oof_meas_t,
        measurement_ids=mids, fold_assign=fold_assign,
    )
    print("dev OOF verification saved to", VERIFY_DIR)
    print("primary fold-mean phase PR-AUC:", summary["fold_mean_phase"]["pr_auc"])
    print("primary fold-mean measurement PR-AUC:", summary["fold_mean_measurement"]["pr_auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
