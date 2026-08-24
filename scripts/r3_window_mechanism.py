# -*- coding: utf-8 -*-
"""
R3: Window-level mechanistic evidence for event-focused information selection
(TIM submission, P1). Pure analysis on frozen window artifacts — zero GPU,
zero retraining, zero protocol risk.

Questions:
  Q1  Do event windows carry label information? Compare the event-score
      distribution of event windows between positive and negative phases
      (effect size + bootstrap CI).
  Q2  Does the event score itself rank phases? Rank correlation between
      per-phase max event score and the phase label (pooled OOF).
  Q3  Are event windows more discriminative than equidistant windows?
      Compare PR-AUC of per-window scores (event vs equidistant kinds) in
      a per-phase aggregation (max over windows of each kind).

Data: development window artifacts under
  artifacts/windows/<hash>/development/*.npz
  (keys: kinds (3,12) uint8 [0=equidistant?, 1=event?], scores (3,12) float32,
   targets (3,) int8, measurement_id)
We scan ALL hashed artifact directories and use the one matching the locked
development split (we take the union of all measurements found).

Outputs (results/window_mechanism/):
  window_mechanism.json   - all numbers (effect sizes, CIs, PR-AUCs)
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "window_mechanism"
OUT.mkdir(parents=True, exist_ok=True)


def collect():
    """Load all development window artifacts from ONE pipeline hash
    (each hash corresponds to the same 2,481-measurement development split);
    merge by measurement id (dedup across hashes is unnecessary)."""
    dirs = sorted(glob.glob(str(ROOT / "artifacts" / "windows" / "*" / "development")))
    if not dirs:
        raise FileNotFoundError("no window artifact dirs found")
    files = sorted(glob.glob(os.path.join(dirs[0], "*.npz")))
    print(f"[collect] scanning {len(files)} artifacts from {dirs[0]}")
    seen = {}
    for f in files:
        try:
            d = np.load(f, allow_pickle=False)
        except Exception:
            continue
        mid = int(d["measurement_id"])
        if mid in seen:
            continue
        seen[mid] = d
    print(f"[collect] {len(seen)} unique development measurements")
    return seen


def pr_auc(y, p):
    order = np.argsort(-p, kind="stable")
    yp = y[order]
    tp = np.cumsum(yp)
    fp = np.cumsum(1 - yp)
    n_pos = yp.sum()
    if n_pos == 0:
        return 0.0
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    if recall[0] == 0:
        recall, precision = recall[1:], precision[1:]
    return float(np.trapz(precision, recall))


def roc_auc(y, p):
    order = np.argsort(-p, kind="stable")
    yp = y[order]
    tp = np.cumsum(yp)
    fp = np.cumsum(1 - yp)
    n_pos, n_neg = yp.sum(), (1 - yp).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float(np.trapz(tp / n_pos, fp / n_neg))


def main():
    data = collect()

    # ---- assemble per-phase event scores and labels ----
    pos_scores, neg_scores = [], []
    phase_max_event, phase_max_uniform, phase_y = [], [], []
    for mid, d in data.items():
        targets = d["targets"].astype(int)
        kinds = d["kinds"]
        scores = d["scores"].astype(np.float64)
        for p in range(3):
            ev = scores[p][kinds[p] == 1]
            un = scores[p][kinds[p] != 1]   # uniform (0) + coverage fallback (2)
            if len(ev):
                phase_max_event.append(ev.max())
            if len(un):
                phase_max_uniform.append(un.max())
            phase_y.append(int(targets[p]))
            if targets[p] == 1:
                pos_scores.extend(ev.tolist())
            else:
                neg_scores.extend(ev.tolist())

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)
    phase_max_event = np.array(phase_max_event)
    phase_max_uniform = np.array(phase_max_uniform)
    phase_y = np.array(phase_y)

    # ---- Q1: event-score distribution, positive vs negative phases ----
    mu_pos, mu_neg = pos_scores.mean(), neg_scores.mean()
    sd_pos, sd_neg = pos_scores.std(), neg_scores.std()
    pooled_sd = np.sqrt((sd_pos**2 + sd_neg**2) / 2)
    cohens_d = (mu_pos - mu_neg) / max(pooled_sd, 1e-12)

    # bootstrap CI on the mean difference (resample windows with replacement)
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(2000):
        b_pos = rng.choice(pos_scores, size=len(pos_scores), replace=True).mean()
        b_neg = rng.choice(neg_scores, size=len(neg_scores), replace=True).mean()
        diffs.append(b_pos - b_neg)
    diffs = np.array(diffs)
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))

    # ---- Q2: rank correlation phase max-event-score vs label ----
    from scipy.stats import spearmanr
    rho, rho_p = spearmanr(phase_max_event, phase_y)

    # ---- Q3: per-phase discriminative power, event vs uniform windows ----
    # per-phase score = max over windows of each kind; PR-AUC vs phase label
    valid_ev = np.isfinite(phase_max_event) & np.isfinite(phase_y)
    valid_un = np.isfinite(phase_max_uniform) & np.isfinite(phase_y)
    pr_event = pr_auc(phase_y[valid_ev], phase_max_event[valid_ev])
    roc_event = roc_auc(phase_y[valid_ev], phase_max_event[valid_ev])
    pr_uniform = pr_auc(phase_y[valid_un], phase_max_uniform[valid_un])
    roc_uniform = roc_auc(phase_y[valid_un], phase_max_uniform[valid_un])

    summary = {
        "Q1_event_score_label_association": {
            "n_positive_phases": int(len(pos_scores)),
            "n_negative_phases": int(len(neg_scores)),
            "mean_event_score_positive": round(float(mu_pos), 4),
            "mean_event_score_negative": round(float(mu_neg), 4),
            "cohens_d": round(float(cohens_d), 4),
            "bootstrap_95ci_mean_diff": [round(ci[0], 4), round(ci[1], 4)],
            "interpretation": "event scores of event windows are "
                              "higher on positive phases" if ci[0] > 0 else "no separation",
        },
        "Q2_rank_correlation": {
            "spearman_rho": round(float(rho), 4),
            "p_value": float(rho_p),
        },
        "Q3_event_vs_uniform_discriminability": {
            "event_windows_pr_auc": round(pr_event, 4),
            "event_windows_roc_auc": round(roc_event, 4),
            "uniform_windows_pr_auc": round(pr_uniform, 4),
            "uniform_windows_roc_auc": round(roc_uniform, 4),
            "note": "per-phase score = max over windows of the kind; "
                    "the event score alone (no learned model) is used",
        },
        "reference": {
            "E4_mainline_phase_pr_auc_pooled": 0.530,
            "E4_mainline_phase_pr_auc_fold_mean": 0.615,
        },
    }
    (OUT / "window_mechanism.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== R3 WINDOW-LEVEL MECHANISM EVIDENCE ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nDone. Output in", OUT)


if __name__ == "__main__":
    main()
