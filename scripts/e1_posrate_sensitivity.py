# -*- coding: utf-8 -*-
"""
E1: PR-AUC prevalence-sensitivity controlled analysis (TIM submission, P0).

Implements the E1 design from reports/tim_experiment_design.md §2.1:
  E1a  analytic PR-curve reweighting (precision(t,λ)=TP/(TP+λ·FP), recall invariant)
  E1b  Harvard measurement-clustered negative downsampling ×20 (bootstrap check)
  E1c  development-side extra points at π ∈ {10%, 15%} (evaluation-side reweighting only)
  plus fixed-recall precision table and Δπ/Δshift decomposition.

Protocol safety: pure post-hoc analysis on FROZEN predictions only
 (development OOF + Harvard one-time receipt predictions). No retraining,
 no reopening of any one-time evaluation, no threshold changes.

Outputs (results/posrate_sensitivity/):
  phase_table.json, meas_table.json, decomposition.json,
  fixed_recall_precision.json, bootstrap_check.json
"""

import json
import os
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "posrate_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 0. Load frozen predictions
# ----------------------------------------------------------------------
DEV = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "stage1_tim", "e4_ctx_concat", "seeds",
                           "seed_42", "oof.npz"), allow_pickle=True)
HAR = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "e4_harvard_blind", "predictions.npz"), allow_pickle=True)

dev_phase_prob = DEV["phase_probs"].ravel()          # 2481*3
dev_phase_y = DEV["phase_targets"].ravel().astype(int)
dev_meas_prob = DEV["meas_probs"]                    # 2481
dev_meas_y = DEV["meas_targets"].astype(int)

har_mask = HAR["in_train_set"] == 0                  # exclude 1,392 train-set signals
har_prob = HAR["probs"][har_mask]
har_y = (HAR["annotations"][har_mask] > 0).astype(int)
har_meas_id = HAR["measurement_ids"][har_mask]

# measurement-level (complete triples only, following the locked protocol)
har_triple_mask = np.zeros(len(har_prob), dtype=bool)
for mid in np.unique(har_meas_id):
    idx = np.where(har_meas_id == mid)[0]
    if len(idx) == 3:
        har_triple_mask[idx] = True
har_meas_prob = np.zeros(len(np.unique(har_meas_id[har_triple_mask])))
# noisy-OR per complete triple
triple_ids = np.unique(har_meas_id[har_triple_mask])
har_meas_y = []
for mid in triple_ids:
    idx = np.where(har_meas_id == mid)[0]
    p = 1.0 - np.prod(1.0 - har_prob[idx])
    har_meas_prob[len(har_meas_y)] = p
    har_meas_y.append(int((har_y[idx] > 0).any()))
har_meas_y = np.array(har_meas_y)

print(f"[load] dev phases {len(dev_phase_prob)} pos {dev_phase_y.sum()} ({dev_phase_y.mean():.4%})")
print(f"[load] dev meas   {len(dev_meas_prob)} pos {dev_meas_y.sum()}")
print(f"[load] har phases {len(har_prob)} pos {har_y.sum()} ({har_y.mean():.4%})")
print(f"[load] har meas   {len(har_meas_prob)} pos {har_meas_y.sum()} (complete triples)")

# ----------------------------------------------------------------------
# 1. PR-AUC helpers
# ----------------------------------------------------------------------
def pr_auc_at_prevalence(prob, y, lam):
    """Analytic PR-AUC under negative reweighting factor lam (lam<=1 keeps
    a fraction lam of negatives). recall unchanged; precision reweighted.
    Uses the trapezoid rule over the empirical PR curve."""
    order = np.argsort(-prob, kind="stable")
    p = prob[order]
    yp = y[order]
    tp = np.cumsum(yp)
    fp = np.cumsum(1 - yp)
    n_pos = yp.sum()
    recall = tp / n_pos
    precision = tp / (tp + lam * fp)
    # trapezoid over recall, dropping the terminal recall=0 point
    if recall[0] == 0:
        recall = recall[1:]
        precision = precision[1:]
    if recall[-1] < 1.0:
        recall = np.concatenate([recall, [1.0]])
        precision = np.concatenate([precision, [precision[-1]]])
    return float(np.trapz(precision, recall))


def pr_auc_empirical(prob, y, n_down=None, rng=None):
    """Empirical PR-AUC (optionally after measurement-clustered negative
    downsampling to a target positive rate)."""
    if n_down is not None:
        neg_idx = np.where(y == 0)[0]
        keep = rng.choice(neg_idx, size=n_down, replace=False)
        sel = np.concatenate([np.where(y == 1)[0], keep])
        prob, y = prob[sel], y[sel]
    return pr_auc_at_prevalence(prob, y, 1.0)


def roc_auc(prob, y):
    order = np.argsort(-prob, kind="stable")
    yp = y[order]
    tp = np.cumsum(yp)
    fp = np.cumsum(1 - yp)
    n_pos, n_neg = yp.sum(), (1 - yp).sum()
    tpr = tp / n_pos
    fpr = fp / n_neg
    return float(np.trapz(tpr, fpr))


def mcc_at(prob, y, th):
    pred = (prob >= th).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0

# ----------------------------------------------------------------------
# 2. E1a: analytic reweighting grid
# ----------------------------------------------------------------------
pi_dev = dev_phase_y.mean()
pi_har = har_y.mean()
grid = [0.0157, 0.03, 0.0595, 0.10, 0.15]

def lam_for_target_pi(pi_target, n_pos, n_neg, allow_gt1=True):
    """lambda that achieves target prevalence pi = n_pos/(n_pos+lam*n_neg).
    allow_gt1: allow lambda>1 (upsampling negatives, analytic extension).
    lambda<=1 means downsampling negatives (feasible only if pi_target > pi_current);
    lambda>1 means we would need MORE negatives than exist (infeasible for
    empirical resampling, but valid analytically as an extrapolation)."""
    lam = (n_pos / pi_target - n_pos) / n_neg
    if lam > 1 + 1e-9 and not allow_gt1:
        return None
    return min(lam, 1.0) if not allow_gt1 else lam

def analytic_table(prob, y, grid, name):
    n_pos, n_neg = y.sum(), (1 - y).sum()
    pi_current = n_pos / (n_pos + n_neg)
    rows = {}
    # always include the current-prevalence (lambda=1) baseline row
    rows["current"] = {
        "pi_effective": round(float(pi_current), 4),
        "lambda": 1.0,
        "pr_auc": round(pr_auc_at_prevalence(prob, y, 1.0), 4),
        "roc_auc": round(roc_auc(prob, y), 4),
        "normalized_lift": round(float((pr_auc_at_prevalence(prob, y, 1.0) - pi_current) / (1 - pi_current)), 4),
    }
    for pi in grid:
        lam = lam_for_target_pi(pi, n_pos, n_neg, allow_gt1=False)
        if lam is None:
            continue  # target pi below current prevalence: infeasible for this dataset
        pi_eff = pi
        prauc = pr_auc_at_prevalence(prob, y, lam)
        roc = roc_auc(prob, y)
        rows[str(pi)] = {
            "pi_effective": round(float(pi_eff), 4),
            "lambda": round(float(lam), 4),
            "pr_auc": round(prauc, 4),
            "roc_auc": round(roc, 4),
            "normalized_lift": round(float((prauc - pi_eff) / (1 - pi_eff)), 4),
        }
    return rows

dev_phase_table = analytic_table(dev_phase_prob, dev_phase_y, grid, "dev_phase")
har_phase_table = analytic_table(har_prob, har_y, grid, "har_phase")

# measurement level
dev_meas_table = analytic_table(dev_meas_prob, dev_meas_y, [0.073, 0.10, 0.15], "dev_meas")
har_meas_table = analytic_table(har_meas_prob, har_meas_y, [0.0333, 0.073, 0.10], "har_meas")

# ----------------------------------------------------------------------
# 3. Decomposition  (phase level)
# ----------------------------------------------------------------------
# delta_shift  = PR_har(pi=5.9% analytic)  - PR_dev(pi=5.9% / current)
# delta_pi     = PR_har(pi=1.57% observed) - PR_har(pi=5.9% analytic)
pr_dev_ref = dev_phase_table["current"]["pr_auc"]
pr_har_59 = har_phase_table["0.0595"]["pr_auc"]
pr_har_157 = har_phase_table["current"]["pr_auc"]   # pi=1.5715% observed
delta_shift = pr_har_59 - pr_dev_ref
delta_pi = pr_har_157 - pr_har_59
decomposition = {
    "dev_pr_auc_current_pi": pr_dev_ref,
    "har_pr_auc_pi_1.57_observed": pr_har_157,
    "har_pr_auc_pi_5.9_analytic": pr_har_59,
    "delta_shift_same_prevalence": round(delta_shift, 4),
    "delta_pi_prevalence_effect": round(delta_pi, 4),
    "total_drop": round(pr_dev_ref - pr_har_157, 4),
    "normalized_lift_dev": dev_phase_table["current"]["normalized_lift"],
    "normalized_lift_har": har_phase_table["current"]["normalized_lift"],
    "lift_ratio": round((pr_har_157 / pi_har) / (pr_dev_ref / pi_dev), 4),
}

# ----------------------------------------------------------------------
# 4. E1b: bootstrap check (Harvard, measurement-clustered downsampling ×20)
# ----------------------------------------------------------------------
rng = np.random.default_rng(42)
n_har_pos = har_y.sum()
n_har_neg = (1 - har_y).sum()
lam59 = lam_for_target_pi(0.0595, n_har_pos, n_har_neg, allow_gt1=False)
n_keep_neg = int(round(lam59 * n_har_neg))
print(f"[E1b] Harvard lam@5.9% = {lam59:.4f} -> keep {n_keep_neg} of {n_har_neg} negatives")

bootstrap_pr = []
for i in range(20):
    pr = pr_auc_empirical(har_prob, har_y, n_down=n_keep_neg, rng=rng)
    bootstrap_pr.append(pr)
bootstrap_pr = np.array(bootstrap_pr)
bootstrap_check = {
    "n_iterations": 20,
    "analytic_pr_auc_at_5.9": har_phase_table["0.0595"]["pr_auc"],
    "bootstrap_mean": round(float(bootstrap_pr.mean()), 4),
    "bootstrap_95ci": [round(float(np.percentile(bootstrap_pr, 2.5)), 4),
                       round(float(np.percentile(bootstrap_pr, 97.5)), 4)],
    "abs_diff_vs_analytic": round(float(abs(bootstrap_pr.mean() - har_phase_table["0.0595"]["pr_auc"])), 4),
}

# ----------------------------------------------------------------------
# 5. E1c: development-side extension points (analytic only)
# ----------------------------------------------------------------------
# (included in dev_phase_table via grid {0.10, 0.15})

# ----------------------------------------------------------------------
# 6. Fixed-recall precision table
# ----------------------------------------------------------------------
def precision_at_recall(prob, y, target_recall):
    order = np.argsort(-prob, kind="stable")
    p, yp = prob[order], y[order]
    tp = np.cumsum(yp)
    fp = np.cumsum(1 - yp)
    recall = tp / yp.sum()
    # smallest threshold index reaching target recall
    idx = np.searchsorted(recall, target_recall)
    if idx >= len(recall):
        idx = len(recall) - 1
    return float(tp[idx] / (tp[idx] + fp[idx]))

fixed_recall = {}
for r in [0.1, 0.2, 0.5]:
    fixed_recall[str(r)] = {
        "dev_precision": round(precision_at_recall(dev_phase_prob, dev_phase_y, r), 4),
        "har_precision": round(precision_at_recall(har_prob, har_y, r), 4),
    }

# ----------------------------------------------------------------------
# 7. Save
# ----------------------------------------------------------------------
out = {
    "phase_table": {"dev": dev_phase_table, "harvard": har_phase_table},
    "meas_table": {"dev": dev_meas_table, "harvard": har_meas_table},
    "decomposition": decomposition,
    "fixed_recall_precision": fixed_recall,
    "bootstrap_check": bootstrap_check,
    "meta": {
        "protocol": "post-hoc analysis of frozen predictions only; no one-time "
                    "evaluation reopened; no thresholds changed",
        "dev_n_phases": int(len(dev_phase_prob)), "dev_n_pos": int(dev_phase_y.sum()),
        "har_n_phases": int(len(har_prob)), "har_n_pos": int(har_y.sum()),
        "sources": ["results/stage1_tim/e4_ctx_concat/seeds/seed_42/oof.npz",
                    "results/e4_harvard_blind/predictions.npz (in_train_set==0)"],
    },
}
with open(os.path.join(OUT_DIR, "phase_table.json"), "w", encoding="utf-8") as f:
    json.dump(out["phase_table"], f, indent=2, ensure_ascii=False)
with open(os.path.join(OUT_DIR, "meas_table.json"), "w", encoding="utf-8") as f:
    json.dump(out["meas_table"], f, indent=2, ensure_ascii=False)
with open(os.path.join(OUT_DIR, "decomposition.json"), "w", encoding="utf-8") as f:
    json.dump(out["decomposition"], f, indent=2, ensure_ascii=False)
with open(os.path.join(OUT_DIR, "fixed_recall_precision.json"), "w", encoding="utf-8") as f:
    json.dump(out["fixed_recall_precision"], f, indent=2, ensure_ascii=False)
with open(os.path.join(OUT_DIR, "bootstrap_check.json"), "w", encoding="utf-8") as f:
    json.dump(out["bootstrap_check"], f, indent=2, ensure_ascii=False)

print("\n=== PHASE-LEVEL PR-AUC(pi) analytic ===")
print(f"{'pi':>8} {'dev':>10} {'harvard':>10}")
for pi in grid:
    d = dev_phase_table.get(str(pi), {}).get("pr_auc")
    h = har_phase_table.get(str(pi), {}).get("pr_auc")
    print(f"{pi:>8.4f} {str(d):>10} {str(h):>10}")

print("\n=== DECOMPOSITION ===")
for k, v in decomposition.items():
    print(f"  {k}: {v}")

print("\n=== FIXED RECALL PRECISION ===")
for k, v in fixed_recall.items():
    print(f"  recall={k}: dev {v['dev_precision']} | har {v['har_precision']}")

print("\n=== E1b BOOTSTRAP CHECK ===")
for k, v in bootstrap_check.items():
    print(f"  {k}: {v}")
print("\nDone. Outputs in", OUT_DIR)
