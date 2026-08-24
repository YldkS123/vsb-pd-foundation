"""Ablation runner on the locked development folds (StratifiedGroupKFold 5, seed 42).

Grids (Cartesian product, deduplicated):
  encoder branch : dual | cnn | feature
  MIL aggregator : gated_attention | attention | mean | max
  phase interact : cyclic | none | concat | max | mean

Usage:
  python scripts/run_ablations.py [--cache results/cached_features/features_full.npz]
                                  [--encoders dual,cnn,feature]
                                  [--mils gated_attention,attention,mean,max]
                                  [--phases cyclic,none,concat,max,mean]
                                  [--epochs 40] [--batch-size 64] [--patience 15]
                                  [--seed 42] [--tag dev_k8]
                                  [--subset 200]           # smoke test only
                                  [--out-dir results/ablations]
                                  [--save-ckpts]
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
from vsb_pd.encoder import WindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.cyclic import PhaseInteractionModule
from vsb_pd.model import VSBPipeline
from vsb_pd.training import make_stratified_group_folds, compute_metrics

SEED = 42
N_FOLDS = 5


def build_model(encoder_branch: str, mil_name: str, phase_kind: str) -> VSBPipeline:
    encoder = WindowEncoder(8192, 58, 128, branch=encoder_branch)
    aggregator = MILAggregator(mil_name, 128)
    cyclic = PhaseInteractionModule(phase_kind, 128)
    classifier = PhaseClassifier(128)
    return VSBPipeline(encoder=encoder, aggregator=aggregator, cyclic=cyclic, classifier=classifier)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_fold(model: nn.Module, windows: torch.Tensor, feat: torch.Tensor, indices, batch_size: int):
    """Return phase probs and phase labels for the given indices."""
    model.eval()
    probs_list, targets_list = [], []
    for i in range(0, len(indices), batch_size):
        bidx = indices[i:i + batch_size]
        logits, _ = model(windows[bidx], feat[bidx])
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
        targets_list.append(labels_np[bidx])
    return np.concatenate(probs_list), np.concatenate(targets_list)


# Global label array set by main() (used inside predict_fold)
labels_np: np.ndarray


def train_config(
    config_name: str,
    encoder_branch: str,
    mil_name: str,
    phase_kind: str,
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
    save_ckpts: bool,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = len(mids)
    label_counts = labels_np.sum(axis=1).astype(int)

    cfg_dir = out_dir / config_name
    cfg_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    oof_probs = np.zeros((M, 3), dtype=np.float64)
    oof_targets = np.zeros((M, 3), dtype=np.float64)
    fold_assign = np.zeros(M, dtype=np.int8)
    n_params = 0

    for fi, (tr, va) in enumerate(folds):
        t0 = time.time()
        torch.manual_seed(seed + fi)
        np.random.seed(seed + fi)

        model = build_model(encoder_branch, mil_name, phase_kind).to(device)
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
                bidx = train_idx[i:i + batch_size]
                optimizer.zero_grad()
                phase_logits, _ = model(windows[bidx], feat[bidx])
                loss = criterion(phase_logits, labels[bidx])
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

        oof_probs[va] = val_probs
        oof_targets[va] = val_targets
        fold_assign[va] = fi

        final_probs, final_targets = predict_fold(model, windows, feat, va.tolist(), batch_size)
        oof_probs[va] = final_probs
        oof_targets[va] = final_targets
        fp = final_probs.flatten()
        ft = final_targets.flatten()
        fm = compute_metrics(ft, fp, (fp >= 0.5).astype(int))
        elapsed = time.time() - t0

        # measurement-level metrics via noisy-OR
        meas_probs = 1.0 - np.prod(1.0 - final_probs, axis=1)
        meas_targets = final_targets.max(axis=1)
        mm = compute_metrics(meas_targets, meas_probs, (meas_probs >= 0.5).astype(int))

        fold_metrics.append({
            "fold": fi + 1,
            "epochs_trained": epochs_trained,
            "elapsed_s": round(elapsed, 1),
            "phase": {k: round(float(v), 4) for k, v in fm.items()},
            "measurement": {k: round(float(v), 4) for k, v in mm.items()},
        })
        print(f"  Fold {fi+1}: phase_pr_auc={fm.get('pr_auc', float('nan')):.4f} "
              f"meas_pr_auc={mm.get('pr_auc', float('nan')):.4f} ({elapsed:.0f}s)")

        if save_ckpts:
            torch.save({"state_dict": best_state, "config": config_name}, cfg_dir / f"model_fold{fi+1}.pt")

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
        "config": {"encoder": encoder_branch, "mil": mil_name, "phase": phase_kind},
        "n_measurements": int(M),
        "n_folds": len(fold_metrics),
        "n_params": int(n_params),
        "mean_phase_pr_auc": float(np.mean(prs)),
        "std_phase_pr_auc": float(np.std(prs)),
        "mean_phase_roc_auc": float(np.mean(rocs)),
        "mean_phase_f1_0.5": float(np.mean(f1s)),
        "mean_measurement_pr_auc": float(np.mean(mprs)),
        "folds": fold_metrics,
    }
    (cfg_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    global labels_np
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cached_features/features_full.npz")
    ap.add_argument("--encoders", default="dual,cnn,feature")
    ap.add_argument("--mils", default="gated_attention,attention,mean,max")
    ap.add_argument("--phases", default="cyclic,none,concat,max,mean")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="dev_k8")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--out-dir", default="results/ablations")
    ap.add_argument("--save-ckpts", action="store_true")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    feat_array = d["feat_array"]
    windows_np = d["windows"]
    labels = d["labels"].astype(np.float32)
    mids = d["measurement_ids"]

    if args.subset:
        take = min(args.subset, len(mids))
        feat_array = feat_array[:take]
        windows_np = windows_np[:take]
        labels = labels[:take]
        mids = mids[:take]

    M = len(mids)
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    print(f"Data: {M} measurements, windows={windows_np.shape}, features={feat_array.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = torch.from_numpy(windows_np).float().to(device)
    feat = torch.from_numpy(feat_array).float().to(device)
    labels_t = torch.from_numpy(np.asarray(labels)).float().to(device)

    label_counts = np.asarray(labels).sum(axis=1).astype(int)
    folds = make_stratified_group_folds(mids, label_counts, n_splits=N_FOLDS, seed=args.seed)
    print(f"Folds: {[(len(tr), len(va)) for tr, va in folds]}")

    configs = []
    for enc in args.encoders.split(","):
        for mil in args.mils.split(","):
            for ph in args.phases.split(","):
                cfg = (enc, mil, ph)
                if cfg not in configs:
                    configs.append(cfg)
    print(f"Configs: {len(configs)}")

    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i, (enc, mil, ph) in enumerate(configs):
        name = f"enc_{enc}__mil_{mil}__ph_{ph}"
        print(f"\n[{i+1}/{len(configs)}] {name}")
        s = train_config(
            name, enc, mil, ph,
            windows, feat, labels_t, mids, folds,
            args.epochs, args.batch_size, args.patience, args.seed,
            out_dir, args.save_ckpts,
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
    (out_dir / "ablation_summary.json").write_text(json.dumps(table, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
