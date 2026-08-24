# -*- coding: utf-8 -*-
"""
L1: Label-ratio experiment (TIM submission, P1).

Quantifies the labeling-cost advantage of the weakly supervised pipeline:
with only a fraction f of phase labels kept (measurement-grouped, per-fold
random retention), how much performance is retained?

  f ∈ {0.05, 0.10, 0.20, 0.50}  (seed 42, 5 folds each; 100% = mainline 0.615,
  reused from the locked report, not rerun)
  f = 0.20 additionally with label-seeds {7, 2024} for stability (15 extra runs)

Protocol: same development folds (StratifiedGroupKFold(5, seed=42)); labels
are masked per measurement (all 3 phases of a kept measurement are kept, all
masked otherwise) with a label-subset seed; validation keeps FULL labels for
early stopping and threshold selection, measuring the true performance
ceiling at the given labeling budget. Development-only; blind/Harvard never
touched.

Outputs (results/label_fraction/):
  <f>/cv_summary.json       per-config summary
  summary.json              label-cost-performance table
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sklearn.metrics import matthews_corrcoef

from vsb_pd.cyclic import PhaseCyclicLoss, PhaseInteractionModule
from vsb_pd.dl_encoders import TimWindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.model import VSBPipeline
from vsb_pd.training import compute_metrics, make_stratified_group_folds

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "cached_features" / "features_policy_mixed_k8.npz"
OUT = ROOT / "results" / "label_fraction"
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 15
LR = 1e-3
WD = 1e-4
GRAD_CLIP = 1.0
FOLD_SEED = 42
LABEL_FRACTIONS = (0.05, 0.10, 0.20, 0.50)
STABILITY_FRACTION = 0.20
STABILITY_LABEL_SEEDS = (7, 2024)
MAINLINE_100 = {"phase_pr_auc_fold_mean": 0.615, "meas_pr_auc_fold_mean": 0.643,
                "phase_pr_auc_pooled": 0.533, "meas_pr_auc_pooled": 0.585}


def build_pipeline():
    encoder = TimWindowEncoder("simple_cnn", 8192, 58, 128)
    aggregator = MILAggregator("attention", 128)
    cyclic = PhaseInteractionModule("context_concat", 128)
    classifier = PhaseClassifier(128)
    return VSBPipeline(
        encoder=encoder, aggregator=aggregator, cyclic=cyclic,
        classifier=classifier, max_encode_chunk=8, checkpoint_chunks=True,
    )


@torch.no_grad()
def predict(model, windows, feat, indices, batch_size, device, labels_np):
    model.eval()
    probs_list, targets_list = [], []
    for i in range(0, len(indices), batch_size):
        bidx = indices[i:i + batch_size]
        logits, _ = model(windows[bidx].to(device), feat[bidx].to(device))
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
        targets_list.append(labels_np[bidx])
    return np.concatenate(probs_list), np.concatenate(targets_list)


def train_fold(model, criterion, windows, feat, labels_masked, labels_full_np,
               tr, va, batch_size, epochs, patience, device, seed_fold):
    torch.manual_seed(seed_fold)
    np.random.seed(seed_fold)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_pr, best_state, patience_counter = -1.0, None, 0
    train_idx = tr.tolist()
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_idx)
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(train_idx), batch_size):
            bidx = train_idx[i:i + batch_size]
            optimizer.zero_grad()
            phase_logits, _ = model(windows[bidx].to(device), feat[bidx].to(device))
            # BCE on the MASKED labels: unlabeled phases contribute no loss
            loss = criterion(phase_logits, labels_masked[bidx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        # early stopping on FULL labels (true ceiling)
        val_probs, val_targets = predict(model, windows, feat, va, batch_size, device, labels_full_np)
        p, t = val_probs.flatten(), val_targets.flatten()
        m = compute_metrics(t, p, (p >= 0.5).astype(int))
        pr = m.get("pr_auc", float("nan"))
        if np.isfinite(pr) and pr > best_pr + 0.001:
            best_pr = pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_pr


def make_label_mask(labels_np, fraction, label_seed, mids, folds, fold_idx):
    """Per-measurement label retention: keep `fraction` of MEASUREMENTS'
    full 3-phase labels in the training part of fold fold_idx. The
    validation part always keeps full labels (used for early stopping)."""
    rng = np.random.default_rng(label_seed * 1000 + fold_idx)
    tr_idx, va_idx = folds[fold_idx]
    mask = np.ones_like(labels_np, dtype=np.float32)
    # random subset of training measurements keeps labels
    n_tr = len(tr_idx)
    keep_n = max(1, int(round(fraction * n_tr)))
    keep_m = rng.choice(tr_idx, size=keep_n, replace=False)
    drop = np.setdiff1d(tr_idx, keep_m)
    mask[drop] = 0.0  # mask ALL phases of dropped measurements
    return mask


def run_config(fraction, label_seed, windows, feat, labels_np, mids, folds, device):
    cfg_dir = OUT / f"f{fraction:g}" / f"labelseed_{label_seed}"
    summary_path = cfg_dir / "cv_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    cfg_dir.mkdir(parents=True, exist_ok=True)
    M = len(mids)
    labels = torch.from_numpy(labels_np).float()
    oof_phase_p = np.zeros((M, 3))
    oof_phase_t = np.zeros((M, 3))
    oof_meas_p = np.zeros(M)
    oof_meas_t = np.zeros(M)
    fold_assign = np.zeros(M, dtype=np.int8)

    for fi, (tr_idx, va_idx) in enumerate(folds):
        mask = make_label_mask(labels_np, fraction, label_seed, mids, folds, fi)
        labels_masked = torch.from_numpy(labels_np * mask).float()
        model = build_pipeline().to(device)
        criterion = PhaseCyclicLoss(lambda_m=0.0)
        t0 = time.time()
        best_pr = train_fold(model, criterion, windows, feat, labels_masked,
                             labels_np, tr_idx, va_idx, BATCH_SIZE, EPOCHS,
                             PATIENCE, device, FOLD_SEED + fi)
        final_probs, final_targets = predict(model, windows, feat, va_idx, BATCH_SIZE, device, labels_np)
        meas_probs = 1.0 - np.prod(1.0 - final_probs, axis=1)
        meas_targets = final_targets.max(axis=1)
        oof_phase_p[va_idx] = final_probs
        oof_phase_t[va_idx] = final_targets
        oof_meas_p[va_idx] = meas_probs
        oof_meas_t[va_idx] = meas_targets
        fold_assign[va_idx] = fi
        fp, ft = final_probs.flatten(), final_targets.flatten()
        fm = compute_metrics(ft, fp, (fp >= 0.5).astype(int))
        mm = compute_metrics(meas_targets, meas_probs, (meas_probs >= 0.5).astype(int))
        print(f"  [f={fraction:g} ls={label_seed} fold {fi + 1}] "
              f"phase_pr={fm.get('pr_auc', float('nan')):.4f} "
              f"meas_pr={mm.get('pr_auc', float('nan')):.4f} ({time.time() - t0:.0f}s)")
        (cfg_dir / f"fold_{fi + 1}.json").write_text(json.dumps({
            "fold": fi + 1,
            "best_val_phase_pr_auc": round(float(best_pr), 4),
            "phase_pr_auc": round(float(fm.get("pr_auc", float("nan"))), 4),
            "meas_pr_auc": round(float(mm.get("pr_auc", float("nan"))), 4),
        }, indent=2), encoding="utf-8")

    # pooled metrics
    p_flat, t_flat = oof_phase_p.reshape(-1), oof_phase_t.reshape(-1)
    pooled_phase = compute_metrics(t_flat, p_flat, (p_flat >= 0.5).astype(int))
    pooled_meas = compute_metrics(oof_meas_t, oof_meas_p, (oof_meas_p >= 0.5).astype(int))
    # fold means
    ph_vals, me_vals = [], []
    for fi in range(N_FOLDS):
        sel = fold_assign == fi
        fp, ft = oof_phase_p[sel].reshape(-1), oof_phase_t[sel].reshape(-1)
        mp, mt = oof_meas_p[sel], oof_meas_t[sel]
        ph_vals.append(compute_metrics(ft, fp, (fp >= 0.5).astype(int))["pr_auc"])
        me_vals.append(compute_metrics(mt, mp, (mp >= 0.5).astype(int))["pr_auc"])
    summary = {
        "label_fraction": float(fraction),
        "label_seed": int(label_seed),
        "fold_mean_phase_pr_auc": round(float(np.mean(ph_vals)), 4),
        "fold_std_phase_pr_auc": round(float(np.std(ph_vals)), 4),
        "fold_mean_meas_pr_auc": round(float(np.mean(me_vals)), 4),
        "fold_std_meas_pr_auc": round(float(np.std(me_vals)), 4),
        "pooled_oof_phase_pr_auc": round(float(pooled_phase.get("pr_auc", float("nan"))), 4),
        "pooled_oof_meas_pr_auc": round(float(pooled_meas.get("pr_auc", float("nan"))), 4),
        "n_labeled_measurements": int(round(fraction * len(mids))),
    }
    (cfg_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(cfg_dir / "oof.npz",
                        phase_probs=oof_phase_p, phase_targets=oof_phase_t,
                        meas_probs=oof_meas_p, meas_targets=oof_meas_t,
                        measurement_ids=mids, fold_assign=fold_assign)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.05,0.10,0.20,0.50")
    ap.add_argument("--label-seeds", default="42")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    d = np.load(CACHE, allow_pickle=False)
    windows = torch.from_numpy(np.asarray(d["windows"])).float()
    feat = torch.from_numpy(np.asarray(d["feat_array"])).float()
    labels_np = np.asarray(d["labels"])
    mids = np.asarray(d["measurement_ids"])
    print(f"[data] {len(mids)} measurements")

    folds = make_stratified_group_folds(
        mids, np.clip(labels_np.sum(axis=1).astype(int), 0, 3),
        n_splits=N_FOLDS, seed=FOLD_SEED)

    fractions = [float(x) for x in args.fractions.split(",")]
    label_seeds = [int(x) for x in args.label_seeds.split(",")]
    if args.only:
        fractions = [float(x) for x in args.only.split(",")]

    summary_table = {"100%_reference_mainline": MAINLINE_100, "runs": {}}
    for f in fractions:
        for ls in label_seeds:
            print(f"\n=== label fraction {f:g} label-seed {ls} ===")
            s = run_config(f, ls, windows, feat, labels_np, mids, folds, device)
            summary_table["runs"][f"f{f:g}_ls{ls}"] = s
            print(f"  -> fold-mean phase PR-AUC {s['fold_mean_phase_pr_auc']}")

    (OUT / "summary.json").write_text(
        json.dumps(summary_table, indent=2), encoding="utf-8")
    print("\n=== LABEL FRACTION SUMMARY ===")
    print(f"{'f':>6} {'phase PR-AUC':>14} {'meas PR-AUC':>14} {'pooled phase':>14}")
    for key, s in summary_table["runs"].items():
        print(f"{key:>18} {s['fold_mean_phase_pr_auc']:>14} "
              f"{s['fold_mean_meas_pr_auc']:>14} {s['pooled_oof_phase_pr_auc']:>14}")
    print(f"{'100% mainline':>18} {MAINLINE_100['phase_pr_auc_fold_mean']:>14} "
          f"{MAINLINE_100['meas_pr_auc_fold_mean']:>14} {MAINLINE_100['phase_pr_auc_pooled']:>14}")
    print("\nDone. Output in", OUT)


if __name__ == "__main__":
    main()
