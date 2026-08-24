"""Metric-convention audit: pooled-OOF vs mean-of-fold for baselines and the model.

Both the baselines (results/baseline_oof/*.npz) and the reference model
(results/ablations/dev_k8/enc_dual__mil_gated_attention__ph_cyclic/oof.npz)
were trained on the same StratifiedGroupKFold(5, seed=42) development folds.
This script reports PR-AUC / ROC-AUC under both conventions so the paper can
pick one and stay consistent.

Usage:
  python scripts/audit_metrics.py [--out results/metric_convention_audit.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def per_fold_aucs(fold_assign: np.ndarray, probs: np.ndarray, targets: np.ndarray):
    """Return per-fold PR-AUC / ROC-AUC lists for phase-level samples."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    prs, rocs = [], []
    for fi in range(int(fold_assign.max()) + 1):
        m = fold_assign == fi
        if m.sum() == 0:
            continue
        t, p = targets[m], probs[m]
        if t.sum() == 0 or (t == 0).sum() == 0:
            continue
        prs.append(average_precision_score(t, p))
        rocs.append(roc_auc_score(t, p))
    return prs, rocs


def evaluate(name: str, npz: Path):
    d = np.load(npz, allow_pickle=True)
    if "phase_probs" in d.files:
        probs2d = d["phase_probs"]
        targets2d = d["phase_targets"]
    else:
        probs2d = d["oof_probs"].reshape(-1, 3)
        targets2d = d["labels"].reshape(-1, 3)
    p = probs2d.flatten()
    t = targets2d.flatten()
    fa = d["fold_assign"].repeat(3) if d["fold_assign"].shape[0] == len(t) // 3 else d["fold_assign"]

    from sklearn.metrics import average_precision_score, roc_auc_score

    prs, rocs = per_fold_aucs(fa, p, t)
    mp = 1.0 - np.prod(1.0 - probs2d, axis=1)
    mt = targets2d.max(axis=1)
    mprs, _ = per_fold_aucs(d["fold_assign"], mp, mt)
    return {
        "name": name,
        "pooled_oof": {
            "phase_pr_auc": round(float(average_precision_score(t, p)), 4),
            "phase_roc_auc": round(float(roc_auc_score(t, p)), 4),
            "measurement_pr_auc": round(float(average_precision_score(mt, mp)), 4),
        },
        "mean_of_folds": {
            "phase_pr_auc_mean": round(float(np.mean(prs)), 4) if prs else None,
            "phase_pr_auc_std": round(float(np.std(prs)), 4) if prs else None,
            "phase_roc_auc_mean": round(float(np.mean(rocs)), 4) if rocs else None,
            "measurement_pr_auc_mean": round(float(np.mean(mprs)), 4) if mprs else None,
            "n_folds": len(prs),
        },
    }


def main() -> None:
    rows = []
    oof_dir = Path("results/baseline_oof")
    for npz in sorted(oof_dir.glob("*.npz")):
        rows.append(evaluate(npz.stem, npz))
    model_npz = Path(
        "results/ablations/dev_k8/enc_dual__mil_gated_attention__ph_cyclic/oof.npz"
    )
    rows.append(evaluate("vsb_mil_reference", model_npz))

    out = Path("results/metric_convention_audit.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'name':<38}{'pooled PR':<12}{'mean PR':<14}{'std':<8}{'pooled ROC':<12}{'mean ROC':<12}")
    print("-" * 96)
    for r in rows:
        po = r["pooled_oof"]
        mf = r["mean_of_folds"]
        print(f"{r['name']:<38}{po['phase_pr_auc']:<12}{mf['phase_pr_auc_mean']:<14}"
              f"{mf['phase_pr_auc_std']:<8}{po['phase_roc_auc']:<12}{mf['phase_roc_auc_mean']:<12}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
