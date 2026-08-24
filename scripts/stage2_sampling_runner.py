# -*- coding: utf-8 -*-
"""IEEE TIM Stage 2 sampling-controlled runner.

Trains the locked E4 architecture (TimWindowEncoder(cnn) -> Attention MIL ->
context_concat -> PhaseClassifier) on different sampling caches while keeping
the encoder, MIL, interaction, loss, optimizer, fold split, and early stopping
protocol identical. Only the sampling policy changes between rows.

Usage:
  python scripts/stage2_sampling_runner.py --cache results/cached_features/features_policy_uniform_k8.npz \
      --config-name uniform_k8 --seeds 42 --out-dir results/stage2_sampling
  python scripts/stage2_sampling_runner.py --cache results/cached_features/features_policy_full_signal.npz \
      --config-name full_signal --batch-size 1 --seeds 42
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage1_tim_runner import load_data, oof_metrics, train_fold

from vsb_pd.cyclic import PhaseCyclicLoss, PhaseInteractionModule

from vsb_pd.dl_encoders import TimWindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.model import VSBPipeline
from vsb_pd.training import make_stratified_group_folds

N_FOLDS = 5
EPOCHS = 40
PATIENCE = 15
LR = 1e-3
WD = 1e-4
GRAD_CLIP = 1.0
METRIC_KEYS = ("pr_auc", "roc_auc", "mcc", "f1", "precision", "recall", "accuracy")


def build_pipeline(window_length: int) -> VSBPipeline:
    encoder = TimWindowEncoder("cnn", window_length, 58, 128)
    aggregator = MILAggregator("attention", 128)
    cyclic = PhaseInteractionModule("context_concat", 128)
    classifier = PhaseClassifier(128)
    return VSBPipeline(
        encoder=encoder,
        aggregator=aggregator,
        cyclic=cyclic,
        classifier=classifier,
        max_encode_chunk=8,
        checkpoint_chunks=True,
    )


@torch.no_grad()
def predict(model: nn.Module, windows: torch.Tensor, feat: torch.Tensor,
            indices, batch_size: int, device: torch.device, labels_np: np.ndarray):
    model.eval()
    probs_list, targets_list = [], []
    for i in range(0, len(indices), batch_size):
        bidx = indices[i:i + batch_size]
        logits, _ = model(windows[bidx].to(device), feat[bidx].to(device))
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
        targets_list.append(labels_np[bidx])
    return np.concatenate(probs_list), np.concatenate(targets_list)


def run_seed(config_name, seed, windows, feat, labels, mids, folds, seed_dir,
             epochs, batch_size, patience, max_folds, device, labels_np):
    seed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = seed_dir / "cv_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    M = len(mids)
    oof_phase_p = np.zeros((M, 3), dtype=np.float64)
    oof_phase_t = np.zeros((M, 3), dtype=np.float64)
    oof_meas_p = np.zeros(M, dtype=np.float64)
    oof_meas_t = np.zeros(M, dtype=np.float64)
    fold_assign = np.zeros(M, dtype=np.int8)

    for fi, (tr, va) in enumerate(folds[:max_folds]):
        fold_file = seed_dir / f"fold_{fi + 1}.json"
        oof_file = seed_dir / f"oof_fold_{fi + 1}.npz"
        if fold_file.exists() and oof_file.exists():
            fm = json.loads(fold_file.read_text(encoding="utf-8"))
            d = np.load(oof_file, allow_pickle=False)
            oof_phase_p[va] = d["phase_probs"]
            oof_phase_t[va] = d["phase_targets"]
            oof_meas_p[va] = d["meas_probs"]
            oof_meas_t[va] = d["meas_targets"]
            fold_assign[va] = fi
            continue

        t0 = time.time()
        model = build_pipeline(windows.shape[-1]).to(device)
        criterion = PhaseCyclicLoss(lambda_m=0.0)
        epochs_trained, best_pr = train_fold(
            model, criterion, windows, feat, labels, labels_np, tr, va,
            batch_size, epochs, patience, device, seed + fi,
        )
        final_probs, final_targets = predict(model, windows, feat, va, batch_size,
                                             device, labels_np)
        meas_probs = 1.0 - np.prod(1.0 - final_probs, axis=1)
        meas_targets = final_targets.max(axis=1)
        oof_phase_p[va] = final_probs
        oof_phase_t[va] = final_targets
        oof_meas_p[va] = meas_probs
        oof_meas_t[va] = meas_targets
        fold_assign[va] = fi

        fp, ft = final_probs.flatten(), final_targets.flatten()
        from vsb_pd.training import compute_metrics
        fm = compute_metrics(ft, fp, (fp >= 0.5).astype(int))
        mm = compute_metrics(meas_targets, meas_probs, (meas_probs >= 0.5).astype(int))
        fold_json = {
            "fold": fi + 1,
            "epochs_trained": int(epochs_trained),
            "best_val_phase_pr_auc": round(float(best_pr), 4),
            "elapsed_s": round(time.time() - t0, 1),
            "phase": {k: round(float(v), 4) for k, v in fm.items()},
            "measurement": {k: round(float(v), 4) for k, v in mm.items()},
        }
        (seed_dir / f"fold_{fi + 1}.json").write_text(
            json.dumps(fold_json, indent=2, default=str), encoding="utf-8")
        np.savez_compressed(
            seed_dir / f"oof_fold_{fi + 1}.npz",
            va_indices=np.asarray(va), phase_probs=final_probs,
            phase_targets=final_targets, meas_probs=meas_probs,
            meas_targets=meas_targets,
        )
        print(f"  [seed {seed}] fold {fi + 1}: phase_pr_auc={fm.get('pr_auc', float('nan')):.4f} "
              f"meas_pr_auc={mm.get('pr_auc', float('nan')):.4f} ({fold_json['elapsed_s']:.0f}s)")

    summary = oof_metrics(oof_phase_p, oof_phase_t, oof_meas_p, oof_meas_t, fold_assign)
    summary.update({
        "config_name": config_name,
        "encoder": "cnn",
        "interaction": "context_concat",
        "lambda_m": 0.0,
        "seed": int(seed),
        "n_measurements": int(M),
        "n_params": int(sum(p.numel() for p in build_pipeline(windows.shape[-1]).parameters())),
    })
    (seed_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    np.savez_compressed(
        seed_dir / "oof.npz",
        phase_probs=oof_phase_p, phase_targets=oof_phase_t,
        meas_probs=oof_meas_p, meas_targets=oof_meas_t,
        measurement_ids=mids, fold_assign=fold_assign,
    )
    return summary


def aggregate_summary(config_name, seed_summaries, window_length):
    def agg(key_level):
        out = {}
        for k in METRIC_KEYS:
            vals = [s[key_level][k] for s in seed_summaries]
            out[k] = float(np.mean(vals))
        return out

    return {
        "config": {
            "experiment": config_name,
            "encoder": "cnn",
            "mil": "attention",
            "phase_interaction": "context_concat",
            "loss": "phase_bce",
            "lambda_m": 0.0,
            "window_policy": config_name,
            "window_length": int(window_length),
            "folds": "StratifiedGroupKFold(5, seed=42)",
        },
        "n_measurements": int(seed_summaries[0]["n_measurements"]),
        "n_folds": int(seed_summaries[0]["n_folds"]),
        "n_params": int(seed_summaries[0]["n_params"]),
        "seeds": [int(s["seed"]) for s in seed_summaries],
        "primary_fold_mean_phase": agg("fold_mean_phase"),
        "primary_fold_mean_measurement": agg("fold_mean_measurement"),
        "per_seed": {str(int(s["seed"])): s for s in seed_summaries},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--config-name", default=None)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out-dir", default="results/stage2_sampling")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--max-folds", type=int, default=N_FOLDS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.max_folds = min(args.max_folds, 2)
        if args.batch_size is None:
            args.batch_size = 8
    windows, feat, labels, mids = load_data(cache, 200 if args.smoke else None)
    labels_np = labels.numpy()
    M = len(mids)
    window_length = int(windows.shape[-1])
    if args.batch_size is None:
        args.batch_size = 1 if window_length > 8192 else 64
    if args.smoke:
        args.batch_size = min(args.batch_size, 8)
    config_name = args.config_name or cache.stem.replace("features_policy_", "")
    print(f"Data: {M} measurements, windows={tuple(windows.shape)}, "
          f"batch_size={args.batch_size}, config={config_name}")

    label_counts = labels_np.sum(axis=1).astype(int)
    n_splits = min(N_FOLDS, M)
    folds = make_stratified_group_folds(mids, label_counts, n_splits=n_splits, seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_root = Path(args.out_dir)
    cfg_dir = out_root / config_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg_dir / "cv_summary.json"
    if summary_path.exists():
        print(f"resume: {config_name} already complete")
        return
    seed_summaries = []
    for seed in tuple(int(s) for s in args.seeds.split(",") if s.strip()):
        seed_dir = cfg_dir / "seeds" / f"seed_{seed}"
        print(f"[{config_name}] seed {seed}")
        s = run_seed(config_name, seed, windows, feat, labels, mids, folds,
                     seed_dir, args.epochs, args.batch_size, args.patience,
                     args.max_folds, device, labels_np)
        seed_summaries.append(s)

    summary = aggregate_summary(config_name, seed_summaries, window_length)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    save = {"measurement_ids": mids}
    for s in seed_summaries:
        seed = int(s["seed"])
        d = np.load(cfg_dir / "seeds" / f"seed_{seed}" / "oof.npz", allow_pickle=False)
        for key in ("phase_probs", "phase_targets", "meas_probs", "meas_targets", "fold_assign"):
            save[f"seed_{seed}_{key}"] = d[key]
    np.savez_compressed(cfg_dir / "oof.npz", **save)
    print(f"Done. Results under {cfg_dir}")


if __name__ == "__main__":
    main()
