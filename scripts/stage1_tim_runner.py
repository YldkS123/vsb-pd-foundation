# -*- coding: utf-8 -*-
"""IEEE TIM Stage 1 runner: phase-interaction ablations, hierarchical loss,
lambda selection, matched encoders, multi-seed OOF, and per-fold resume.

Development-only: reads results/cached_features/features_policy_mixed_k8.npz,
writes results/stage1_tim/<config>/cv_summary.json + oof.npz. The frozen
423-measurement blind set is never loaded, opened, or predicted here.

Usage:
  python scripts/stage1_tim_runner.py --experiments e1,e2,e3,e4,e5,e6 \
      --encoders simple_cnn,resnet1d,inceptiontime --seeds 42,7,2024
  python scripts/stage1_tim_runner.py --experiments e4 --encoders simple_cnn \
      --seeds 42 --smoke
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
from sklearn.metrics import matthews_corrcoef

from vsb_pd.cyclic import PhaseCyclicLoss, PhaseInteractionModule
from vsb_pd.dl_encoders import TimWindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.model import VSBPipeline
from vsb_pd.training import compute_metrics, make_stratified_group_folds

N_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 15
LR = 1e-3
WD = 1e-4
GRAD_CLIP = 1.0
DEFAULT_SEEDS = (42, 7, 2024)
LAMBDA_GRID = (0.25, 0.5, 1.0)
METRIC_KEYS = ("pr_auc", "roc_auc", "mcc", "f1", "precision", "recall", "accuracy")

INTERACTIONS = {
    "e1": ("none", 0.0, "phase_bce"),
    "e2": ("mean", 0.0, "phase_bce"),
    "e3": ("max", 0.0, "phase_bce"),
    "e4": ("context_concat", 0.0, "phase_bce"),
    "e5": ("context_add", 0.0, "phase_bce"),
}


def build_pipeline(encoder_name: str, interaction: str) -> VSBPipeline:
    encoder = TimWindowEncoder(encoder_name, 8192, 58, 128)
    aggregator = MILAggregator("attention", 128)
    cyclic = PhaseInteractionModule(interaction, 128)
    classifier = PhaseClassifier(128)
    return VSBPipeline(
        encoder=encoder,
        aggregator=aggregator,
        cyclic=cyclic,
        classifier=classifier,
        max_encode_chunk=8,
        checkpoint_chunks=True,
    )


def load_data(cache_path: Path, subset: int | None):
    d = np.load(cache_path, allow_pickle=False)
    windows = np.asarray(d["windows"])
    feat = np.asarray(d["feat_array"])
    labels = np.asarray(d["labels"], dtype=np.float32)
    mids = np.asarray(d["measurement_ids"])
    if subset:
        take = min(subset, len(mids))
        windows = windows[:take]
        feat = feat[:take]
        labels = labels[:take]
        mids = mids[:take]
    return (
        torch.from_numpy(windows).float(),
        torch.from_numpy(feat).float(),
        torch.from_numpy(labels).float(),
        mids,
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


def max_mcc_threshold(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Max-MCC threshold over np.linspace(0.01, 0.99, 99)."""
    best_t, best_mcc = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        mcc = matthews_corrcoef(targets, (scores >= t).astype(int))
        if mcc > best_mcc:
            best_t, best_mcc = float(t), float(mcc)
    return best_t, best_mcc


def oof_metrics(phase_probs, phase_targets, meas_probs, meas_targets, fold_assign):
    """Per-seed summary: OOF max-MCC thresholds, fold-mean primary, pooled-OOF secondary."""
    phase_flat_p = phase_probs.reshape(-1)
    phase_flat_t = phase_targets.reshape(-1)
    phase_thr, _ = max_mcc_threshold(phase_flat_p, phase_flat_t)
    meas_thr, _ = max_mcc_threshold(meas_probs, meas_targets)

    fold_phase, fold_meas = [], []
    for fi in range(int(fold_assign.max()) + 1):
        sel = fold_assign == fi
        fp, ft = phase_probs[sel].reshape(-1), phase_targets[sel].reshape(-1)
        mp, mt = meas_probs[sel], meas_targets[sel]
        fm = compute_metrics(ft, fp, (fp >= phase_thr).astype(int))
        fm["mcc"] = float(matthews_corrcoef(ft, (fp >= phase_thr).astype(int)))
        mm = compute_metrics(mt, mp, (mp >= meas_thr).astype(int))
        mm["mcc"] = float(matthews_corrcoef(mt, (mp >= meas_thr).astype(int)))
        fold_phase.append(fm)
        fold_meas.append(mm)

    pooled_phase = compute_metrics(phase_flat_t, phase_flat_p, (phase_flat_p >= phase_thr).astype(int))
    pooled_phase["mcc"] = float(matthews_corrcoef(phase_flat_t, (phase_flat_p >= phase_thr).astype(int)))
    pooled_meas = compute_metrics(meas_targets, meas_probs, (meas_probs >= meas_thr).astype(int))
    pooled_meas["mcc"] = float(matthews_corrcoef(meas_targets, (meas_probs >= meas_thr).astype(int)))

    def mean_std(key: str, items) -> tuple[float, float]:
        vals = [it[key] for it in items if np.isfinite(it[key])]
        return float(np.mean(vals)), float(np.std(vals))

    return {
        "phase_threshold": phase_thr,
        "measurement_threshold": meas_thr,
        "fold_mean_phase": {k: mean_std(k, fold_phase)[0] for k in METRIC_KEYS},
        "fold_std_phase": {k: mean_std(k, fold_phase)[1] for k in METRIC_KEYS},
        "fold_mean_measurement": {k: mean_std(k, fold_meas)[0] for k in METRIC_KEYS},
        "fold_std_measurement": {k: mean_std(k, fold_meas)[1] for k in METRIC_KEYS},
        "pooled_oof_phase": {k: float(pooled_phase[k]) for k in METRIC_KEYS},
        "pooled_oof_measurement": {k: float(pooled_meas[k]) for k in METRIC_KEYS},
        "n_folds": len(fold_phase),
        "folds": [{"fold": i + 1, "phase": fp, "measurement": mm}
                  for i, (fp, mm) in enumerate(zip(fold_phase, fold_meas))],
    }


def train_fold(model, criterion, windows, feat, labels, labels_np, tr, va,
               batch_size, epochs, patience, device, seed_fold):
    torch.manual_seed(seed_fold)
    np.random.seed(seed_fold)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_pr = -1.0
    best_state = None
    patience_counter = 0
    epochs_trained = 0
    train_idx = tr.tolist()
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_idx)
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(train_idx), batch_size):
            bidx = train_idx[i:i + batch_size]
            optimizer.zero_grad()
            phase_logits, _ = model(windows[bidx].to(device), feat[bidx].to(device))
            loss = criterion(phase_logits, labels[bidx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        val_probs, val_targets = predict(model, windows, feat, va, batch_size, device, labels_np)
        p, t = val_probs.flatten(), val_targets.flatten()
        m = compute_metrics(t, p, (p >= 0.5).astype(int))
        pr = m.get("pr_auc", float("nan"))
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:3d}: loss={total_loss / max(n_batches, 1):.4f} "
                  f"phase_pr_auc={pr:.4f}")
        epochs_trained = epoch + 1
        if np.isfinite(pr) and pr > best_pr + 0.001:
            best_pr = pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return epochs_trained, best_pr


def run_seed(config_name, encoder_name, interaction, lambda_m, seed, windows, feat,
             labels, mids, folds, seed_dir, epochs, batch_size, patience, max_folds,
             device, labels_np):
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
        model = build_pipeline(encoder_name, interaction).to(device)
        criterion = PhaseCyclicLoss(lambda_m=lambda_m)
        epochs_trained, best_pr = train_fold(
            model, criterion, windows, feat, labels, labels_np, tr, va,
            batch_size, epochs, patience, device, seed + fi,
        )
        final_probs, final_targets = predict(model, windows, feat, va, batch_size, device, labels_np)
        meas_probs = 1.0 - np.prod(1.0 - final_probs, axis=1)
        meas_targets = final_targets.max(axis=1)
        oof_phase_p[va] = final_probs
        oof_phase_t[va] = final_targets
        oof_meas_p[va] = meas_probs
        oof_meas_t[va] = meas_targets
        fold_assign[va] = fi

        fp, ft = final_probs.flatten(), final_targets.flatten()
        fm = compute_metrics(ft, fp, (fp >= 0.5).astype(int))
        mm = compute_metrics(meas_targets, meas_probs, (meas_probs >= 0.5).astype(int))
        fm_json = {
            "fold": fi + 1,
            "epochs_trained": int(epochs_trained),
            "best_val_phase_pr_auc": round(float(best_pr), 4),
            "elapsed_s": round(time.time() - t0, 1),
            "phase": {k: round(float(v), 4) for k, v in fm.items()},
            "measurement": {k: round(float(v), 4) for k, v in mm.items()},
        }
        (seed_dir / f"fold_{fi + 1}.json").write_text(
            json.dumps(fm_json, indent=2, default=str), encoding="utf-8")
        np.savez_compressed(
            seed_dir / f"oof_fold_{fi + 1}.npz",
            va_indices=np.asarray(va), phase_probs=final_probs,
            phase_targets=final_targets, meas_probs=meas_probs,
            meas_targets=meas_targets,
        )
        print(f"  [seed {seed}] fold {fi + 1}: phase_pr_auc={fm.get('pr_auc', float('nan')):.4f} "
              f"meas_pr_auc={mm.get('pr_auc', float('nan')):.4f} ({fm_json['elapsed_s']:.0f}s)")

    summary = oof_metrics(oof_phase_p, oof_phase_t, oof_meas_p, oof_meas_t, fold_assign)
    summary.update({
        "config_name": config_name,
        "encoder": encoder_name,
        "interaction": interaction,
        "lambda_m": float(lambda_m),
        "seed": int(seed),
        "n_measurements": int(M),
        "n_params": int(sum(p.numel() for p in build_pipeline(encoder_name, interaction).parameters())),
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


def aggregate_summary(config_name, encoder_name, interaction, lambda_m, seed_summaries):
    def agg(key_level):
        out = {}
        for k in METRIC_KEYS:
            vals = [s[key_level][k] for s in seed_summaries]
            out[k] = float(np.mean(vals))
        return out

    def agg_std(key_level):
        out = {}
        for k in METRIC_KEYS:
            vals = [s[key_level][k] for s in seed_summaries]
            out[k] = float(np.mean(vals))
        return out

    return {
        "config": {
            "experiment": config_name,
            "encoder": encoder_name,
            "mil": "attention",
            "phase_interaction": interaction,
            "loss": "phase_bce" if lambda_m == 0.0 else "hierarchical_phase+lambda*measurement",
            "lambda_m": float(lambda_m),
            "window_policy": "mixed_k8",
            "folds": "StratifiedGroupKFold(5, seed=42)",
        },
        "n_measurements": int(seed_summaries[0]["n_measurements"]),
        "n_folds": int(seed_summaries[0]["n_folds"]),
        "n_params": int(seed_summaries[0]["n_params"]),
        "seeds": [int(s["seed"]) for s in seed_summaries],
        "primary_fold_mean_phase": agg("fold_mean_phase"),
        "primary_fold_mean_measurement": agg("fold_mean_measurement"),
        "primary_fold_std_phase": agg_std("fold_std_phase"),
        "primary_fold_std_measurement": agg_std("fold_std_measurement"),
        "per_seed": {str(int(s["seed"])): s for s in seed_summaries},
    }


def select_lambda(grid_dir: Path, out_path: Path) -> float:
    rows = []
    for lam in LAMBDA_GRID:
        p = grid_dir / f"lam_{lam:g}" / "seeds" / "seed_42" / "cv_summary.json"
        if not p.exists():
            raise FileNotFoundError(f"lambda grid run missing: {p}")
        s = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "lambda": float(lam),
            "measurement_pr_auc": s["pooled_oof_measurement"]["pr_auc"],
            "phase_pr_auc": s["pooled_oof_phase"]["pr_auc"],
        })
    best_meas = max(r["measurement_pr_auc"] for r in rows)
    tied = [r for r in rows if best_meas - r["measurement_pr_auc"] < 0.005]
    chosen = max(tied, key=lambda r: r["phase_pr_auc"])["lambda"]
    record = {"rule": "max pooled-OOF measurement PR-AUC; tie within 0.005 -> higher phase PR-AUC",
              "rows": rows, "chosen_lambda": chosen}
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  lambda selection: {record}")
    return chosen


def run_config(config_name, encoder_name, interaction, lambda_m, seeds, windows, feat,
               labels, mids, folds, out_dir, epochs, batch_size, patience, max_folds,
               device, labels_np):
    cfg_dir = out_dir / config_name
    summary_path = cfg_dir / "cv_summary.json"
    if summary_path.exists():
        print(f"  resume: {config_name} already complete")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    cfg_dir.mkdir(parents=True, exist_ok=True)
    seed_summaries = []
    for seed in seeds:
        seed_dir = cfg_dir / "seeds" / f"seed_{seed}"
        print(f"[{config_name}] seed {seed}")
        s = run_seed(config_name, encoder_name, interaction, lambda_m, seed,
                     windows, feat, labels, mids, folds, seed_dir,
                     epochs, batch_size, patience, max_folds, device, labels_np)
        seed_summaries.append(s)

    summary = aggregate_summary(config_name, encoder_name, interaction, lambda_m, seed_summaries)
    (cfg_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Combined per-seed OOF for later paired bootstrap analysis.
    save = {"measurement_ids": mids}
    for s in seed_summaries:
        seed = int(s["seed"])
        d = np.load(cfg_dir / "seeds" / f"seed_{seed}" / "oof.npz", allow_pickle=False)
        for key in ("phase_probs", "phase_targets", "meas_probs", "meas_targets", "fold_assign"):
            save[f"seed_{seed}_{key}"] = d[key]
    np.savez_compressed(cfg_dir / "oof.npz", **save)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="e1,e2,e3,e4,e5,e6")
    ap.add_argument("--encoders", default="simple_cnn,resnet1d,inceptiontime")
    ap.add_argument("--seeds", default="42,7,2024")
    ap.add_argument("--cache", default="results/cached_features/features_policy_mixed_k8.npz")
    ap.add_argument("--out-dir", default="results/stage1_tim")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--max-folds", type=int, default=N_FOLDS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        if args.subset is None:
            args.subset = 200
        args.subset = min(args.subset, 200)
        args.max_folds = min(args.max_folds, 2)
        args.epochs = min(args.epochs, 3)
        out_root = Path(args.out_dir) / "smoke"
    else:
        out_root = Path(args.out_dir)

    cache = Path(args.cache)
    windows, feat, labels, mids = load_data(cache, args.subset)
    labels_np = labels.numpy()
    M = len(mids)
    print(f"Data: {M} measurements, windows={tuple(windows.shape)}")

    if not args.smoke:
        assert windows.shape == (2481, 3, 8, 8192), f"unexpected cache shape: {windows.shape}"

    label_counts = labels_np.sum(axis=1).astype(int)
    folds = make_stratified_group_folds(mids, label_counts, n_splits=N_FOLDS, seed=42)
    print(f"Folds: {[(len(tr), len(va)) for tr, va in folds[:args.max_folds]]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    out_root.mkdir(parents=True, exist_ok=True)

    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    e1e5 = [e for e in experiments if e in INTERACTIONS]
    for exp in e1e5:
        interaction, lambda_m, _ = INTERACTIONS[exp]
        name_map = {"e1": "e1_ctx_none", "e2": "e2_ctx_mean",
                    "e3": "e3_ctx_max", "e4": "e4_ctx_concat", "e5": "e5_ctx_add"}
        run_config(name_map[exp], "cnn", interaction, lambda_m, seeds,
                   windows, feat, labels, mids, folds, out_root,
                   args.epochs, args.batch_size, args.patience, args.max_folds,
                   device, labels_np)

    # E6: lambda grid on seed 42, select lambda, then remaining seeds.
    if "e6" in experiments:
        grid_dir = out_root / "e6_lambda_grid"
        for lam in LAMBDA_GRID:
            run_config(f"lam_{lam:g}", "cnn", "context_concat", lam, (42,),
                       windows, feat, labels, mids, folds, grid_dir,
                       args.epochs, args.batch_size, args.patience, args.max_folds,
                       device, labels_np)
        chosen = select_lambda(grid_dir, out_root / "e6_lambda_selection.json")
        final_seeds = (42,) if args.smoke else seeds
        run_config(f"e6_ctx_concat_lam_{chosen:g}", "cnn", "context_concat", chosen,
                   final_seeds, windows, feat, labels, mids, folds, out_root,
                   args.epochs, args.batch_size, args.patience, args.max_folds,
                   device, labels_np)

    for enc in [e.strip() for e in args.encoders.split(",") if e.strip()]:
        if enc not in ("cnn", "simple_cnn", "resnet1d", "inceptiontime", "tf_cnn"):
            raise ValueError(f"Unknown encoder: {enc}")
        if enc == "cnn":
            continue  # matched cnn row reuses e4_ctx_concat
        run_config(f"enc_{enc}_ctx_concat", enc, "context_concat", 0.0, (42,),
                   windows, feat, labels, mids, folds, out_root,
                   args.epochs, args.batch_size, args.patience, args.max_folds,
                   device, labels_np)
    print(f"Done. Results under {out_root}")


if __name__ == "__main__":
    main()
