"""VSB pipeline CLI: foundation + model experiments + blind evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, TrainingConfig, ModelConfig, load_config
from .data import WindowDataset
from .encoder import WindowEncoder
from .mil import MILAggregator, PhaseClassifier
from .cyclic import CyclicPhaseModule, PhaseCyclicLoss
from .model import VSBPipeline
from .baselines import aggregate_features_per_phase, train_lr_baseline, train_rf_baseline, train_lgbm_baseline
from .features import extract_physical_features
from .training import (
    make_stratified_group_folds,
    train_one_epoch_batched,
    validate_batched,
    compute_metrics,
)
from .evaluation import compute_bootstrap_ci, calibrate_platt, final_blind_evaluate
from .extract import extract_development, pipeline_identity
from .integrity import AuditReport, audit_development
from .locks import create_split_lock, discover_historical_prediction_files


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_extract(args) -> int:
    if args.limit_measurements is not None and not args.smoke_test:
        raise ValueError("--limit-measurements requires --smoke-test")
    if args.smoke_test and args.limit_measurements is None:
        raise ValueError("--smoke-test requires --limit-measurements")

    config = load_config(args.config)
    manifest_path = extract_development(
        config, args.split_lock, limit_measurements=args.limit_measurements,
    )
    pipeline_hash, _ = pipeline_identity(config, args.split_lock)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps({
        "manifest_path": str(manifest_path.resolve()),
        "pipeline_hash": pipeline_hash,
    }, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    print(f"Pipeline hash: {pipeline_hash}")
    return 0


def cmd_audit(args) -> int:
    try:
        report = audit_development(
            load_config(args.config), args.split_lock, args.manifest,
        )
    except Exception as exc:
        report = AuditReport(False, 0, 0, (f"audit prerequisites invalid: {exc}",))
    result = {
        "ok": report.ok,
        "measurements": report.measurements,
        "windows": report.windows,
        "errors": list(report.errors),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if report.ok else 1


def cmd_train_model(args) -> int:
    """Train the VSB dual-branch MIL pipeline with nested CV."""
    config = load_config(args.config)
    manifest = args.manifest
    output_dir = _ensure_dir(args.output_dir)
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    print(f"Loading manifest: {manifest}")
    dataset = WindowDataset(manifest, config)
    print(f"Loaded {len(dataset)} measurements")

    # Collect measurement IDs and labels
    all_ids = []
    all_label_counts = []
    for idx in range(len(dataset)):
        _, _, _, _, targets, mid = dataset[idx]
        all_ids.append(mid)
        all_label_counts.append(int(targets.sum().item()))

    all_ids = np.array(all_ids)
    all_label_counts = np.array(all_label_counts)

    # Outer CV folds
    n_outer = args.outer_folds if hasattr(args, "outer_folds") else 5
    folds = make_stratified_group_folds(all_ids, all_label_counts, n_splits=n_outer, seed=seed)
    print(f"CV folds: {len(folds)} outer")

    all_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n=== Fold {fold_idx+1}/{len(folds)} ===")
        train_indices = train_idx.tolist()
        val_indices = val_idx.tolist()
        print(f"  Train: {len(train_indices)} measurements, Val: {len(val_indices)} measurements")

        # Build model
        model = VSBPipeline(
            encoder=WindowEncoder(8192, 58, 128),
            aggregator=MILAggregator("gated_attention", 128),
            cyclic=CyclicPhaseModule(128),
            classifier=PhaseClassifier(128),
        )

        criterion = PhaseCyclicLoss(lambda_m=0.25)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Simple training loop (simplified for initial run)
        best_val_pr_auc = -1.0
        best_state = None
        patience = 0
        max_patience = args.patience or 20

        for epoch in range(args.epochs or 50):
            train_loss = train_one_epoch_batched(
                model, optimizer, criterion, dataset,
                train_indices, batch_size=args.batch_size or 8,
                grad_clip_norm=1.0,
            )

            val_loss, val_probs, val_targets_np = validate_batched(
                model, criterion, dataset, val_indices, batch_size=args.batch_size or 8,
            )

            val_probs_flat = val_probs.flatten()
            val_targets_flat = val_targets_np.flatten()
            val_preds = (val_probs_flat >= 0.5).astype(int)
            m = compute_metrics(val_targets_flat, val_probs_flat, val_preds)

            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"pr_auc={m.get('pr_auc', float('nan')):.4f}, "
                      f"f1={m.get('f1', 0):.4f}, recall={m.get('recall', 0):.4f}")

            current_pr_auc = m.get("pr_auc", 0.0)
            if not np.isnan(current_pr_auc) and current_pr_auc > best_val_pr_auc + 0.001:
                best_val_pr_auc = current_pr_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        all_metrics.append({"fold": fold_idx + 1, "best_val_pr_auc": best_val_pr_auc})

        # Save fold model
        ckpt = output_dir / f"model_fold{fold_idx+1}.pt"
        torch.save({"state_dict": model.state_dict(), "metrics": m}, ckpt)
        print(f"  Saved: {ckpt}")

    # Summary
    print("\n=== CV Summary ===")
    aucs = [x["best_val_pr_auc"] for x in all_metrics if not np.isnan(x["best_val_pr_auc"])]
    if aucs:
        print(f"  Mean PR-AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    results = {"folds": all_metrics, "mean_pr_auc": float(np.mean(aucs)) if aucs else None}
    (output_dir / "cv_summary.json").write_text(json.dumps(results, indent=2))

    return 0


def cmd_train_baselines(args) -> int:
    """Train LR, RF, LightGBM baselines using aggregated features."""
    config = load_config(args.config)
    manifest = args.manifest
    output_dir = _ensure_dir(args.output_dir)
    seed = args.seed

    print(f"Loading manifest: {manifest}")
    dataset = WindowDataset(manifest, config)
    print(f"Loaded {len(dataset)} measurements")

    # Extract features and aggregate
    all_windows = []
    all_labels = []
    all_measurement_ids = []

    for idx in range(len(dataset)):
        windows, starts, kinds, scores, targets, mid = dataset[idx]
        # windows: (3, K, 8192)
        all_windows.append(windows.numpy())
        all_labels.append(targets.numpy())
        all_measurement_ids.append(mid)

    all_windows = np.stack(all_windows)  # (M, 3, K, 8192)
    all_labels = np.array(all_labels, dtype=np.int8)  # (M, 3)
    all_mids = np.array(all_measurement_ids)

    # Extract physical features
    M, P, K, L = all_windows.shape
    features = extract_physical_features(
        all_windows.reshape(M * P, K, L), config.sampling_rate_hz,
    )
    print(f"Extracted {len(features)} feature types")

    # Aggregate to per-phase features
    X, y, groups = aggregate_features_per_phase(features, all_labels, all_mids, num_phases=P, num_windows=K)
    print(f"Feature matrix: {X.shape}, labels: {y.shape}")

    results = {}

    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr_model, lr_params = train_lr_baseline(X, y, groups, seed=seed)
    print(f"Best params: {lr_params}")
    results["lr"] = {"params": lr_params}

    # Random Forest
    print("\n--- Random Forest ---")
    rf_model, rf_params = train_rf_baseline(X, y, groups, seed=seed)
    print(f"Best params: {rf_params}")
    results["rf"] = {"params": rf_params}

    # LightGBM
    print("\n--- LightGBM ---")
    try:
        lgb_model, lgb_params = train_lgbm_baseline(X, y, groups, seed=seed)
        print(f"Best params: {lgb_params}")
        results["lgbm"] = {"params": lgb_params}
    except Exception as e:
        print(f"LightGBM failed: {e}")

    (output_dir / "baseline_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_dir / 'baseline_results.json'}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="vsb-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- Foundation ---
    lock = sub.add_parser("lock-split")
    lock.add_argument("--candidate", type=Path, required=True)
    lock.add_argument("--historical-root", type=Path, action="append", required=True)
    lock.add_argument("--allow-shrink-holdout", action="store_true")
    lock.add_argument("--output", type=Path, required=True)

    extract = sub.add_parser("extract-development")
    extract.add_argument("--config", type=Path, required=True)
    extract.add_argument("--split-lock", type=Path, required=True)
    extract.add_argument("--receipt", type=Path, required=True)
    extract.add_argument("--smoke-test", action="store_true")
    extract.add_argument("--limit-measurements", type=int)

    audit = sub.add_parser("audit-development")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--split-lock", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, required=True)

    # --- Model ---
    train = sub.add_parser("train-model")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--patience", type=int, default=20)

    bl = sub.add_parser("train-baselines")
    bl.add_argument("--config", type=Path, required=True)
    bl.add_argument("--manifest", type=Path, required=True)
    bl.add_argument("--output-dir", type=Path, required=True)
    bl.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "lock-split":
        prediction_paths = discover_historical_prediction_files(args.historical_root)
        lock_path = create_split_lock(
            args.candidate, prediction_paths, args.output,
            allow_shrink_holdout=args.allow_shrink_holdout,
        )
        print(lock_path)
    elif args.command == "extract-development":
        return cmd_extract(args)
    elif args.command == "audit-development":
        return cmd_audit(args)
    elif args.command == "train-model":
        return cmd_train_model(args)
    elif args.command == "train-baselines":
        return cmd_train_baselines(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
