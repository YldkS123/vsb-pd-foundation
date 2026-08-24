# -*- coding: utf-8 -*-
"""DL baseline encoders on the locked dev folds (P0-6).

Trains deep window encoders (ResNet1D / TCN / InceptionTime / simple CNN)
inside the SAME lightweight MIL pipeline used by the proposed simple candidate
(Mean MIL + Max phase interaction + noisy-OR), on the same measurement-level
StratifiedGroupKFold(5, seed=42) development splits, same window data, same
early stopping and same evaluation metrics. This isolates encoder
expressiveness and answers whether the proposed sampling+aggregation story
holds against modern time-series encoders.

Usage:
  python scripts/run_dl_baselines.py [--cache results/cached_features/features_full.npz]
                                     [--encoders resnet1d,tcn,inception,simple_cnn]
                                     [--tag dl_baselines] [--epochs 40]
                                     [--batch-size 64] [--patience 15] [--seed 42]
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
from vsb_pd.dl_encoders import build_dl_encoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.cyclic import PhaseInteractionModule
from vsb_pd.model import VSBPipeline
from vsb_pd.training import make_stratified_group_folds, compute_metrics

N_FOLDS = 5


def build_model(encoder_name: str, mil_name: str = "mean", phase_kind: str = "max") -> VSBPipeline:
    encoder = build_dl_encoder(encoder_name, 8192, 128)
    aggregator = MILAggregator(mil_name, 128)
    cyclic = PhaseInteractionModule(phase_kind, 128)
    classifier = PhaseClassifier(128)
    # Bound activation memory on the 8 GB GPU: checkpoint each small encoder chunk
    # so deep encoders (esp. TCN) recompute activations instead of caching them all.
    return VSBPipeline(encoder=encoder, aggregator=aggregator, cyclic=cyclic,
                       classifier=classifier, max_encode_chunk=1, checkpoint_chunks=True)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_fold(model: nn.Module, windows: torch.Tensor, feat: torch.Tensor, indices, batch_size: int):
    model.eval()
    device = next(model.parameters()).device
    probs_list, targets_list = [], []
    for i in range(0, len(indices), batch_size):
        bidx = indices[i : i + batch_size]
        logits, _ = model(windows[bidx].to(device), feat[bidx].to(device))
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
        targets_list.append(labels_np[bidx])
    return np.concatenate(probs_list), np.concatenate(targets_list)


labels_np: np.ndarray


def train_config(
    config_name: str,
    encoder_name: str,
    windows: torch.Tensor,
    feat: torch.Tensor,
    labels: torch.Tensor,
    mids: np.ndarray,
    folds,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    out_dir: Path,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = len(mids)

    cfg_dir = out_dir / config_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg_dir / "cv_summary.json"
    if summary_path.exists():
        print(f"  resume: {config_name} already complete"); return json.loads(summary_path.read_text(encoding="utf-8"))

    fold_dir = cfg_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    oof_probs = np.zeros((M, 3), dtype=np.float64)
    oof_targets = np.zeros((M, 3), dtype=np.float64)
    fold_assign = np.zeros(M, dtype=np.int8)
    n_params = 0

    start_fold = 0
    for fi in range(N_FOLDS):
        fold_file = fold_dir / f"fold_{fi + 1}.json"
        oof_file = fold_dir / f"oof_fold_{fi + 1}.npz"
        if fold_file.exists() and oof_file.exists():
            fold_metrics.append(json.loads(fold_file.read_text(encoding="utf-8")))
            d = np.load(oof_file, allow_pickle=True)
            va = d["va_indices"]
            oof_probs[va] = d["phase_probs"]
            oof_targets[va] = d["phase_targets"]
            fold_assign[va] = fi
        else:
            start_fold = fi
            break
    if start_fold > 0:
        print(f"  resume: {config_name} from fold {start_fold + 1}")

    for fi, (tr, va) in enumerate(folds):
        if fi < start_fold:
            continue
        t0 = time.time()
        torch.manual_seed(seed + fi)
        np.random.seed(seed + fi)

        model = build_model(encoder_name).to(device)
        n_params = count_params(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        best_pr_auc = -1.0
        best_state = None
        patience_counter = 0
        epochs_trained = 0

        train_idx = tr.tolist()
        for epoch in range(epochs):
            model.train()
            np.random.shuffle(train_idx)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(train_idx), batch_size):
                bidx = train_idx[i : i + batch_size]
                optimizer.zero_grad()
                phase_logits, _ = model(windows[bidx].to(device), feat[bidx].to(device))
                loss = criterion(phase_logits, labels[bidx].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1

            val_probs, val_targets = predict_fold(model, windows, feat, va.tolist(), batch_size)
            p_flat = val_probs.flatten()
            t_flat = val_targets.flatten()
            m = compute_metrics(t_flat, p_flat, (p_flat >= 0.5).astype(int))
            pr_auc = m.get("pr_auc", float("nan"))
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={total_loss / max(n_batches, 1):.4f} "
                      f"pr_auc={pr_auc:.4f} f1={m.get('f1', 0):.4f}")
            epochs_trained = epoch + 1
            if np.isfinite(pr_auc) and pr_auc > best_pr_auc + 0.001:
                best_pr_auc = pr_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        final_probs, final_targets = predict_fold(model, windows, feat, va.tolist(), batch_size)
        oof_probs[va] = final_probs
        oof_targets[va] = final_targets
        fold_assign[va] = fi
        fp = final_probs.flatten()
        ft = final_targets.flatten()
        fm = compute_metrics(ft, fp, (fp >= 0.5).astype(int))
        elapsed = time.time() - t0

        meas_probs = 1.0 - np.prod(1.0 - final_probs, axis=1)
        meas_targets = final_targets.max(axis=1)
        mm = compute_metrics(meas_targets, meas_probs, (meas_probs >= 0.5).astype(int))

        fm_json = {
            "fold": fi + 1,
            "epochs_trained": epochs_trained,
            "elapsed_s": round(elapsed, 1),
            "phase": {k: round(float(v), 4) for k, v in fm.items()},
            "measurement": {k: round(float(v), 4) for k, v in mm.items()},
        }
        fold_metrics.append(fm_json)
        (fold_dir / f"fold_{fi + 1}.json").write_text(json.dumps(fm_json, indent=2, default=str), encoding="utf-8")
        np.savez_compressed(fold_dir / f"oof_fold_{fi + 1}.npz", va_indices=np.asarray(va),
                            phase_probs=final_probs, phase_targets=final_targets)
        print(f"  Fold {fi+1}: phase_pr_auc={fm.get('pr_auc', float('nan')):.4f} "
              f"meas_pr_auc={mm.get('pr_auc', float('nan')):.4f} ({elapsed:.0f}s)")

    np.savez_compressed(
        cfg_dir / "oof.npz",
        phase_probs=oof_probs, phase_targets=oof_targets,
        measurement_ids=mids, fold_assign=fold_assign,
    )

    prs = [f["phase"]["pr_auc"] for f in fold_metrics]
    rocs = [f["phase"]["roc_auc"] for f in fold_metrics]
    f1s = [f["phase"]["f1"] for f in fold_metrics]
    mprs = [f["measurement"]["pr_auc"] for f in fold_metrics]
    summary = {
        "config": {"encoder": encoder_name, "mil": "mean", "phase": "max"},
        "n_measurements": int(M),
        "n_folds": len(fold_metrics),
        "n_params": int(n_params),
        "mean_phase_pr_auc": round(float(np.mean(prs)), 4),
        "std_phase_pr_auc": round(float(np.std(prs)), 4),
        "mean_phase_roc_auc": round(float(np.mean(rocs)), 4),
        "mean_phase_f1_0.5": round(float(np.mean(f1s)), 4),
        "mean_measurement_pr_auc": round(float(np.mean(mprs)), 4),
        "folds": fold_metrics,
    }
    (cfg_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    global labels_np
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cached_features/features_full.npz")
    ap.add_argument("--encoders", default="resnet1d,tcn,inception,simple_cnn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="dl_baselines")
    ap.add_argument("--out-dir", default="results/dl_baselines")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    feat_array = d["feat_array"]
    windows_np = d["windows"]
    labels = np.asarray(d["labels"], dtype=np.float32)
    mids = np.asarray(d["measurement_ids"])
    M = len(mids)
    labels_np = labels
    print(f"Data: {M} measurements, windows={windows_np.shape}, features={feat_array.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    windows = torch.from_numpy(windows_np).float()
    feat = torch.from_numpy(feat_array).float()
    labels_t = torch.from_numpy(labels).float()

    label_counts = labels.sum(axis=1).astype(int)
    folds = make_stratified_group_folds(mids, label_counts, n_splits=N_FOLDS, seed=args.seed)
    print(f"Folds: {[(len(tr), len(va)) for tr, va in folds]}")

    names = [n.strip() for n in args.encoders.split(",")]
    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i, enc in enumerate(names):
        config_name = f"enc_{enc}__mil_mean__ph_max"
        print(f"\n[{i+1}/{len(names)}] {config_name}")
        s = train_config(
            config_name, enc, windows, feat, labels_t, mids, folds,
            args.epochs, args.batch_size, args.patience, args.seed, out_dir,
        )
        summaries.append(s)

    print("\n" + "=" * 100)
    print(f"{'config':<45} {'params':<9} {'phase PR-AUC':<18} {'meas PR-AUC':<14} {'phase ROC':<12} {'F1@0.5':<8}")
    print("-" * 100)
    for s in summaries:
        c = s["config"]
        name = f"enc_{c['encoder']}__mil_{c['mil']}__ph_{c['phase']}"
        print(f"{name:<45} {s['n_params']:<9} "
              f"{s['mean_phase_pr_auc']:.4f}±{s['std_phase_pr_auc']:.4f} "
              f"{s['mean_measurement_pr_auc']:<14.4f} {s['mean_phase_roc_auc']:<12.4f} "
              f"{s['mean_phase_f1_0.5']:<8.4f}")

    table = [{"name": f"enc_{s['config']['encoder']}__mil_{s['config']['mil']}__ph_{s['config']['phase']}", **s} for s in summaries]
    (out_dir / "dl_baseline_summary.json").write_text(json.dumps(table, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'dl_baseline_summary.json'}")


if __name__ == "__main__":
    main()
