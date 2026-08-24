# -*- coding: utf-8 -*-
"""P0-1 audit: fold-mean PR-AUC vs pooled-OOF PR-AUC gap for the reference model.

The paper reports fold-mean phase PR-AUC 0.586 (main run, results/model_full) vs
pooled-OOF 0.426, a 0.16 gap that must be explained before publication. This
script checks the seven items listed in the revision brief:

  1. per-fold positive prevalence (phase and measurement);
  2. per-fold probability distribution (mean/std/quantiles/min/max);
  3. per-fold calibration shift (mean positive/negative probability, bin ECE);
  4. OOF prediction vs measurement_id alignment;
  5. metric-function consistency (both conventions use average_precision_score);
  6. per-fold PR-AUC computed with the same function (compute_metrics);
  7. per-fold probability scale differences.

It audits two runs:
  - main run: results/model_full/model_fold{1..5}.pt (fold-mean 0.586, pooled 0.426)
  - ablation reference: results/ablations/dev_k8/enc_dual__mil_gated_attention__ph_cyclic/oof.npz
    (fold-mean 0.590, pooled 0.495)

Usage:
  python scripts/audit_fold_oof_gap.py [--cache results/cached_features/features_full.npz]
                                       [--out results/fold_oof_gap_audit.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vsb_pd.encoder import WindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.cyclic import CyclicPhaseModule
from vsb_pd.model import VSBPipeline
from vsb_pd.training import make_stratified_group_folds, compute_metrics

SEED = 42
N_FOLDS = 5


def build_model() -> VSBPipeline:
    return VSBPipeline(
        encoder=WindowEncoder(8192, 58, 128, branch="dual"),
        aggregator=MILAggregator("gated_attention", 128),
        cyclic=CyclicPhaseModule(128),
        classifier=PhaseClassifier(128),
    )


@torch.no_grad()
def predict_indices(model: nn.Module, windows: torch.Tensor, feat: torch.Tensor, indices, batch_size: int):
    model.eval()
    probs_list = []
    for i in range(0, len(indices), batch_size):
        bidx = indices[i : i + batch_size]
        logits, _ = model(windows[bidx], feat[bidx])
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs_list)


def oof_predict(models, folds, windows, feat, batch_size) -> tuple[np.ndarray, np.ndarray]:
    M = len(windows)
    oof_p = np.zeros((M, 3), dtype=np.float64)
    fold_assign = np.full(M, -1, dtype=np.int8)
    for fi, (tr, va) in enumerate(folds):
        p = predict_indices(models[fi], windows, feat, va.tolist(), batch_size)
        assert p.shape == (len(va), 3)
        oof_p[va] = p
        fold_assign[va] = fi
    return oof_p, fold_assign


def bin_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(probs).ravel()
    t = np.asarray(targets).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = t[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def analyze_probs(probs: np.ndarray, targets: np.ndarray, fold_assign: np.ndarray) -> dict:
    """Per-fold + pooled analysis on flattened phase-level samples."""
    p = np.asarray(probs).ravel()
    t = np.asarray(targets).ravel()
    fa = np.repeat(np.asarray(fold_assign), 3)  # per-measurement -> per-phase
    assert p.shape == t.shape == fa.shape, (p.shape, t.shape, fa.shape)
    assert np.isfinite(p).all(), "non-finite probabilities found"
    assert (p >= 0.0).all() and (p <= 1.0).all(), "probabilities outside [0,1]"
    assert fa.min() >= 0 and fa.max() <= N_FOLDS - 1, "fold assignment incomplete"

    folds = []
    for fi in range(N_FOLDS):
        m = fa == fi
        tp, tt = p[m], t[m]
        pos = tt.sum()
        neg = (tt == 0).sum()
        fm = compute_metrics(tt, tp, (tp >= 0.5).astype(int))
        folds.append({
            "fold": fi + 1,
            "n_phase": int(len(tt)),
            "n_pos": int(pos),
            "n_neg": int(neg),
            "pos_prevalence": round(float(pos / len(tt)), 5),
            "pr_auc": round(float(fm.get("pr_auc", float("nan"))), 4),
            "roc_auc": round(float(fm.get("roc_auc", float("nan"))), 4),
            "prob_mean_pos": round(float(tp[tt == 1].mean()), 5) if pos else None,
            "prob_mean_neg": round(float(tp[tt == 0].mean()), 5) if neg else None,
            "prob_mean": round(float(tp.mean()), 5),
            "prob_std": round(float(tp.std()), 5),
            "prob_q25": round(float(np.percentile(tp, 25)), 5),
            "prob_median": round(float(np.percentile(tp, 50)), 5),
            "prob_q75": round(float(np.percentile(tp, 75)), 5),
            "prob_min": round(float(tp.min()), 5),
            "prob_max": round(float(tp.max()), 5),
            "ece_10bin": round(bin_ece(tp, tt), 5),
        })

    pooled_fm = compute_metrics(t, p, (p >= 0.5).astype(int))
    prs = [f["pr_auc"] for f in folds]
    return {
        "n_phase": int(len(t)),
        "n_pos": int(t.sum()),
        "n_neg": int((t == 0).sum()),
        "pos_prevalence": round(float(t.mean()), 5),
        "pooled_pr_auc": round(float(pooled_fm.get("pr_auc", float("nan"))), 4),
        "pooled_roc_auc": round(float(pooled_fm.get("roc_auc", float("nan"))), 4),
        "fold_mean_pr_auc": round(float(np.mean(prs)), 4),
        "fold_std_pr_auc": round(float(np.std(prs)), 4),
        "gap_foldmean_minus_pooled": round(float(np.mean(prs) - pooled_fm.get("pr_auc", float("nan"))), 4),
        "folds": folds,
    }


def measurement_analysis(probs: np.ndarray, targets: np.ndarray, fold_assign: np.ndarray) -> dict:
    mp = 1.0 - np.prod(1.0 - np.asarray(probs), axis=1)
    mt = np.asarray(targets).max(axis=1)
    fa = np.asarray(fold_assign)
    folds = []
    for fi in range(N_FOLDS):
        m = fa == fi
        tp, tt = mp[m], mt[m]
        fm = compute_metrics(tt, tp, (tp >= 0.5).astype(int))
        folds.append({
            "fold": fi + 1,
            "n_meas": int(len(tt)),
            "n_pos": int(tt.sum()),
            "pos_prevalence": round(float(tt.mean()), 5),
            "pr_auc": round(float(fm.get("pr_auc", float("nan"))), 4),
        })
    pooled = compute_metrics(mt, mp, (mp >= 0.5).astype(int))
    prs = [f["pr_auc"] for f in folds]
    return {
        "n_meas": int(len(mt)),
        "n_pos": int(mt.sum()),
        "pos_prevalence": round(float(mt.mean()), 5),
        "pooled_pr_auc": round(float(pooled.get("pr_auc", float("nan"))), 4),
        "fold_mean_pr_auc": round(float(np.mean(prs)), 4),
        "fold_std_pr_auc": round(float(np.std(prs)), 4),
        "gap_foldmean_minus_pooled": round(float(np.mean(prs) - pooled.get("pr_auc", float("nan"))), 4),
        "folds": folds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cached_features/features_full.npz")
    ap.add_argument("--ckpts", default="results/model_full")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="results/fold_oof_gap_audit.json")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    feat_array = d["feat_array"]
    windows_np = d["windows"]
    labels = np.asarray(d["labels"])
    mids = np.asarray(d["measurement_ids"])
    M = len(mids)
    print(f"Data: {M} measurements")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    windows = torch.from_numpy(windows_np).float().to(device)
    feat = torch.from_numpy(feat_array).float().to(device)

    label_counts = labels.sum(axis=1).astype(int)
    folds = make_stratified_group_folds(mids, label_counts, n_splits=N_FOLDS, seed=args.seed)

    # --- main run (results/model_full) ---
    ckpt_dir = Path(args.ckpts)
    models = []
    for fi in range(N_FOLDS):
        model = build_model().to(device)
        ckpt = torch.load(ckpt_dir / f"model_fold{fi + 1}.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        models.append(model)
        print(f"Loaded fold {fi + 1} ckpt")

    oof_p, fold_assign = oof_predict(models, folds, windows, feat, args.batch_size)
    assert set(fold_assign.tolist()) == set(range(N_FOLDS))
    assert (fold_assign >= 0).all()
    report = {
        "checkpoint_source": str(ckpt_dir),
        "model": "dual + gated_attention + cyclic + noisy-OR",
        "metric_function": "sklearn.metrics.average_precision_score (same for per-fold and pooled)",
        "alignment_check": "oof_p[va] = per-fold val predictions; fold_assign complete and unique; probs finite in [0,1]",
        "phase": analyze_probs(oof_p, labels, fold_assign),
        "measurement": measurement_analysis(oof_p, labels, fold_assign),
    }

    # --- ablation reference run (oof.npz already saved) ---
    ref_npz = Path("results/ablations/dev_k8/enc_dual__mil_gated_attention__ph_cyclic/oof.npz")
    if ref_npz.exists():
        rd = np.load(ref_npz, allow_pickle=True)
        rp = rd["phase_probs"]
        rt = rd["phase_targets"]
        rfa = rd["fold_assign"]
        rid = rd["measurement_ids"]
        ref_folds = make_stratified_group_folds(rid, np.asarray(rt).sum(axis=1).astype(int),
                                                n_splits=N_FOLDS, seed=args.seed)
        align_ok = all(bool((rfa[va] == fi).all()) for fi, (_, va) in enumerate(ref_folds))
        report["ablation_reference"] = {
            "source": str(ref_npz),
            "alignment_with_deterministic_folds": bool(align_ok),
            "phase": analyze_probs(rp, rt, rfa),
            "measurement": measurement_analysis(rp, rt, rfa),
        }
        print(f"Ablation reference alignment with deterministic folds: {align_ok}")

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- console table ---
    print("\n" + "=" * 100)
    for key in ("phase", "measurement"):
        r = report[key]
        print(f"[{key}] pooled={r['pooled_pr_auc']:.4f}  "
              f"fold-mean={r['fold_mean_pr_auc']:.4f}±{r.get('fold_std_pr_auc', float('nan')):.4f}  "
              f"gap={r['gap_foldmean_minus_pooled']:+.4f}  prev={r['pos_prevalence']:.4f}")
        for f in r["folds"]:
            print(f"  fold{f['fold']}: pr={f['pr_auc']:.4f} prev={f['pos_prevalence']:.4f} "
                  f"mean_p={f.get('prob_mean', float('nan')):.4f} std_p={f.get('prob_std', float('nan')):.4f} "
                  f"p_pos={f.get('prob_mean_pos', float('nan')):.4f} p_neg={f.get('prob_mean_neg', float('nan')):.4f} "
                  f"ece={f.get('ece_10bin', float('nan')):.4f}")
    if "ablation_reference" in report:
        a = report["ablation_reference"]
        print(f"[ablation ref] phase pooled={a['phase']['pooled_pr_auc']:.4f} "
              f"fold-mean={a['phase']['fold_mean_pr_auc']:.4f}±{a['phase']['fold_std_pr_auc']:.4f}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
