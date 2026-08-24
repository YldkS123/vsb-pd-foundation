#!/usr/bin/env python3
"""VSB Partial Discharge Detection — Experiment Runner (CUDA-accelerated).

First cache features:
    python scripts/cache_features.py                              # full 2481 (~18 min)
    python scripts/cache_features.py --subset 500

Then run experiments (GPU auto-detected, ~5x faster):
    python scripts/run_experiments.py
    python scripts/run_experiments.py --full --epochs 50 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vsb_pd.config import load_config
from vsb_pd.encoder import WindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.cyclic import CyclicPhaseModule
from vsb_pd.model import VSBPipeline
from vsb_pd.baselines import train_lr_baseline, train_rf_baseline, train_lgbm_baseline
from vsb_pd.training import make_stratified_group_folds, compute_metrics

CACHE_DIR = Path("results/cached_features")


def _load_cached(subset: int | None) -> dict:
    if subset:
        path = CACHE_DIR / f"features_n{subset}.npz"
    else:
        path = CACHE_DIR / "features_full.npz"
    if not path.exists():
        print(f"Cache not found: {path}")
        print("Run 'python scripts/cache_features.py' first")
        sys.exit(1)
    print(f"Loading: {path} ({path.stat().st_size/1e6:.0f} MB)")
    data = np.load(path, allow_pickle=True)
    return {
        "features": data["feat_array"],
        "windows": data["windows"],
        "labels": data["labels"],
        "measurement_ids": data["measurement_ids"],
        "agg_X": data["aggregated_X"],
        "agg_y": data["aggregated_y"],
        "agg_groups": data["aggregated_groups"],
    }


def run_baselines(data: dict, output_dir: Path, seed: int = 42):
    print("\n" + "=" * 60)
    print("BASELINE MODELS")
    print("=" * 60)

    X, y, groups = data["agg_X"], data["agg_y"], data["agg_groups"]
    M = len(data["measurement_ids"])
    print(f"Features: {X.shape}, labels: {y.shape}, measurements: {M}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"Labels: {dict(zip(unique.astype(int), counts))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"n_measurements": int(M), "feature_dim": int(X.shape[1]), "n_samples": len(y)}

    for name, train_fn in [("lr", train_lr_baseline), ("rf", train_rf_baseline)]:
        t0 = datetime.now()
        model, params = train_fn(X, y, groups, seed=seed)
        elapsed = (datetime.now() - t0).total_seconds()
        results[name] = {"params": {k: str(v) for k, v in params.items()}, "train_time_s": elapsed}
        print(f"  {name}: {params} ({elapsed:.0f}s)")

    t0 = datetime.now()
    try:
        lgb_model, lgb_params = train_lgbm_baseline(X, y, groups, seed=seed)
        elapsed = (datetime.now() - t0).total_seconds()
        results["lgbm"] = {"params": {k: str(v) for k, v in lgb_params.items()}, "train_time_s": elapsed}
        print(f"  lgbm: {lgb_params} ({elapsed:.0f}s)")
    except Exception as e:
        print(f"  LightGBM failed: {e}")

    results_path = output_dir / "baseline_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved: {results_path}")


def run_model_training(data: dict, output_dir: Path, seed=42, epochs=50, batch_size=64, patience=20):
    print("\n" + "=" * 60)
    print("VSB MIL MODEL TRAINING (GPU-accelerated)")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    feat_array = torch.from_numpy(data["features"]).float().to(device)
    windows = torch.from_numpy(data["windows"]).float().to(device)
    labels = torch.from_numpy(data["labels"]).float().to(device)
    mids = data["measurement_ids"]

    M = len(mids)
    print(f"Data: {M} measurements, {feat_array.shape}")

    # numpy ops need CPU data
    labels_cpu = labels.cpu()
    label_counts = labels_cpu.sum(dim=1).numpy().astype(int)
    n_outer = 5 if M >= 100 else 3
    folds = make_stratified_group_folds(mids, label_counts, n_splits=n_outer, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n  === Fold {fold_idx+1}/{n_outer} ===")
        train_indices = train_idx.tolist()
        val_indices = val_idx.tolist()
        print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")

        model = VSBPipeline(
            encoder=WindowEncoder(8192, 58, 128),
            aggregator=MILAggregator("gated_attention", 128),
            cyclic=CyclicPhaseModule(128),
            classifier=PhaseClassifier(128),
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_val_loss = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            np.random.shuffle(train_indices)
            total_train_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_indices), batch_size):
                batch_idx = train_indices[i:i + batch_size]
                batch_windows = windows[batch_idx]
                batch_features = feat_array[batch_idx]
                batch_labels = labels[batch_idx]

                optimizer.zero_grad()
                phase_logits, _ = model(batch_windows, batch_features)
                loss = criterion(phase_logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()
                n_batches += 1

            # Validate on GPU
            model.eval()
            all_val_probs, all_val_targets = [], []
            with torch.no_grad():
                for i in range(0, len(val_indices), batch_size):
                    batch_idx = val_indices[i:i + batch_size]
                    batch_windows = windows[batch_idx]
                    batch_features = feat_array[batch_idx]
                    batch_labels = labels[batch_idx]
                    phase_logits, _ = model(batch_windows, batch_features)
                    probs = torch.sigmoid(phase_logits)
                    all_val_probs.append(probs.cpu().numpy())
                    all_val_targets.append(batch_labels.cpu().numpy())

            avg_train_loss = total_train_loss / max(n_batches, 1)
            val_probs_flat = np.concatenate(all_val_probs).flatten()
            val_targets_flat = np.concatenate(all_val_targets).flatten()
            val_preds_flat = (val_probs_flat >= 0.5).astype(int)
            metrics = compute_metrics(val_targets_flat, val_probs_flat, val_preds_flat)

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_train_loss:.4f}, "
                      f"pr_auc={metrics.get('pr_auc', float('nan')):.4f}, "
                      f"f1={metrics.get('f1',0):.4f}, recall={metrics.get('recall',0):.4f}")

            current_pr_auc = metrics.get("pr_auc", float("nan"))
            if not np.isnan(current_pr_auc) and current_pr_auc > best_val_loss + 0.001:
                best_val_loss = current_pr_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final eval
        model.eval()
        all_final_probs, all_final_targets = [], []
        with torch.no_grad():
            for idx in val_indices:
                w, f = windows[idx:idx+1], feat_array[idx:idx+1]
                t = labels[idx:idx+1]
                logits, _ = model(w, f)
                probs = torch.sigmoid(logits)
                all_final_probs.append(probs.cpu().numpy())
                all_final_targets.append(t.cpu().numpy())

        fp = np.concatenate(all_final_probs).flatten()
        ft = np.concatenate(all_final_targets).flatten()
        fm = compute_metrics(ft, fp, (fp >= 0.5).astype(int))
        print(f"    Final: pr_auc={fm.get('pr_auc', 0):.4f}, f1={fm.get('f1',0):.4f}, "
              f"recall={fm.get('recall',0):.4f}")

        ckpt = output_dir / f"model_fold{fold_idx+1}.pt"
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                     "metrics": fm}, ckpt)
        all_fold_metrics.append({f"fold_{fold_idx+1}": fm})

    aucs = [list(m.values())[0].get("pr_auc", float("nan")) for m in all_fold_metrics]
    valid = [a for a in aucs if not np.isnan(a)]
    if valid:
        print(f"\n  Mean PR-AUC: {np.mean(valid):.4f} +/- {np.std(valid):.4f}")
    summary = {"n_measurements": M, "n_folds": n_outer,
               "mean_pr_auc": float(np.mean(valid)) if valid else None}
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Saved: {output_dir / 'cv_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="VSB Experiment Runner")
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    subset = None if args.full else args.subset
    mode = "full" if args.full else f"subset{args.subset}"
    data = _load_cached(subset)

    if not args.skip_baselines:
        run_baselines(data, Path(f"results/baselines_{mode}"), seed=args.seed)

    if not args.skip_model:
        run_model_training(data, Path(f"results/model_{mode}"), seed=args.seed,
                           epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
