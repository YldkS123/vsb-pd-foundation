# -*- coding: utf-8 -*-
"""
B1: Classical detector baselines (TIM submission, P0/P1).

Implements classical, standard-method reference detectors on the SAME
development folds and the SAME pooled-OOF evaluation protocol as the E4
mainline, to give TIM reviewers the "comparison with classical methods"
that Section V requires:

  B1a  Energy-threshold detector   (window RMS > tau, aggregated per phase)
  B1b  Crest/kurtosis detector     (impulsiveness statistics > tau)
  B1c  PRPD-style feature + LR     (logistic regression on physical features,
                                    the classical machine-learning reference)

Protocol: frozen development split (StratifiedGroupKFold(5, seed=42),
measurement-grouped), thresholds selected ONLY on training-fold OOF (max-MCC
or fixed), phase-level PR-AUC/ROC-AUC reported per-fold mean and pooled OOF.
No blind set, no Harvard data, no retraining of E4: purely additional
baselines on the development set.

Outputs (results/classical_baselines/):
  report.json   - per-detector phase/measurement PR-AUC, ROC-AUC, MCC, F1
  oof.npz       - pooled OOF probabilities for each detector
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vsb_pd.training import make_stratified_group_folds

OUT_DIR = ROOT / "results" / "classical_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5


# ---------------------------------------------------------------------- #
# Data loading (frozen cache)
# ---------------------------------------------------------------------- #
def load_data():
    d = np.load(ROOT / "results" / "cached_features" / "features_full.npz",
                allow_pickle=True)
    feat = d["feat_array"]          # (2481,3,8,58) float32
    y_phase = d["labels"]           # (2481,3) int8
    meas_ids = d["measurement_ids"] # (2481,)
    names = d["feature_names"]      # (58,)
    # phase-level flattened: (7443, 58) per window, aggregated over windows
    # -- build per-phase feature summary (mean over K=8 windows)
    X_mean = feat.mean(axis=2)      # (2481,3,58)
    X_std = feat.std(axis=2)        # (2481,3,58)
    X_max = feat.max(axis=2)        # (2481,3,58)
    # choose feature indices by name
    idx = {n: i for i, n in enumerate(names)}
    return dict(feat=feat, y_phase=y_phase, meas_ids=meas_ids, names=names,
                X_mean=X_mean, X_std=X_std, X_max=X_max, idx=idx)


# ---------------------------------------------------------------------- #
# Classical detectors (per phase, no learning)
# ---------------------------------------------------------------------- #
def detector_energy(X_mean, X_max):
    """Energy detector: per-phase max window RMS (feature 'rms') / energy.
    Score = mean of top-2 window RMS + peak-to-peak. Higher = more PD-like."""
    # 'rms' is a time-domain feature; also 'energy'
    rms = X_mean[:, :, data["idx"]["rms"]]
    energy = X_max[:, :, data["idx"]["energy"]]
    p2p = X_max[:, :, data["idx"]["peak_to_peak"]]
    # robust z of log-energy + rms
    s = np.log1p(energy) + rms + np.log1p(p2p)
    return s  # (2481,3)


def detector_impulsive(X_mean, X_max):
    """Impulsiveness detector: crest factor + kurtosis (PD pulses are impulsive)."""
    crest = X_max[:, :, data["idx"]["crest_factor"]]
    kurt = X_max[:, :, data["idx"]["kurtosis"]]
    s = np.log1p(np.maximum(crest, 0)) + 0.5 * np.log1p(np.maximum(kurt, 0))
    return s


def detector_spectral(X_mean, X_max):
    """Spectral detector: spectral centroid + peak-band energy ratio."""
    centroid = X_mean[:, :, data["idx"]["spectral_centroid_hz"]]
    # use band-energy features if present (names may include 'band_*')
    band_cols = [i for n, i in data["idx"].items() if n.startswith("band")]
    s = centroid
    if band_cols:
        s = s + 0.5 * X_mean[:, :, band_cols].sum(axis=2)
    return s


# ---------------------------------------------------------------------- #
# PRPD-style LR baseline (classical ML on physical features)
# ---------------------------------------------------------------------- #
def prpd_lr_oof(X_flat, y_flat, meas_ids_m):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # measurement-level folds (same grouping as E4), expanded to phases
    y_meas = np.clip(y_phase.sum(axis=1).astype(int), 0, 3)  # per measurement
    folds = make_stratified_group_folds(meas_ids_m, y_meas, n_splits=N_FOLDS, seed=SEED)
    oof = np.zeros(len(y_flat))
    for tr_m, va_m in folds:
        # expand measurement indices to phase indices (3 phases each)
        tr_idx = np.concatenate([np.arange(m * 3, m * 3 + 3) for m in tr_m])
        va_idx = np.concatenate([np.arange(m * 3, m * 3 + 3) for m in va_m])
        scaler = StandardScaler()
        clf = LogisticRegression(C=10.0, max_iter=5000, solver="lbfgs",
                                 random_state=SEED)
        clf.fit(scaler.fit_transform(X_flat[tr_idx]), y_flat[tr_idx])
        oof[va_idx] = clf.predict_proba(scaler.transform(X_flat[va_idx]))[:, 1]
    return oof


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #
def pr_auc(y, p):
    order = np.argsort(-p, kind="stable")
    yp = y[order]
    tp = np.cumsum(yp)
    fp = np.cumsum(1 - yp)
    n_pos = yp.sum()
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
    return float(np.trapz(tp / n_pos, fp / n_neg))


def mcc_f1(y, p, th):
    pred = (p >= th).astype(int)
    tp = float(((pred == 1) & (y == 1)).sum()); tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum()); fn = float(((pred == 0) & (y == 1)).sum())
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / den) if den > 0 else 0.0
    f1 = float(2 * tp / max(2 * tp + fp + fn, 1))
    return mcc, f1


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
data = load_data()
y_phase = data["y_phase"]           # (2481,3)
y_flat = y_phase.ravel()            # (7443,)
meas_ids = data["meas_ids"]         # (2481,)

detectors = {
    "energy": detector_energy(data["X_mean"], data["X_max"]),
    "impulsive": detector_impulsive(data["X_mean"], data["X_max"]),
    "spectral": detector_spectral(data["X_mean"], data["X_max"]),
}

results = {}
oof_dict = {}

# --- classical detectors: per-fold mean + pooled OOF ---
for name, score in detectors.items():
    s_flat = score.ravel()  # (7443,)
    oof_dict[name] = s_flat
    pa = pr_auc(y_flat, s_flat)
    ra = roc_auc(y_flat, s_flat)
    # threshold from max-MCC on pooled OOF (dev-only discipline)
    best_mcc, best_f1, best_th = -1, 0, 0.5
    for th in np.linspace(np.percentile(s_flat, 50), np.percentile(s_flat, 99.9), 200):
        m, f = mcc_f1(y_flat, s_flat, th)
        if m > best_mcc:
            best_mcc, best_f1, best_th = m, f, th
    results[name] = {
        "type": "classical (no learning)",
        "phase_pr_auc_pooled": round(pa, 4),
        "phase_roc_auc_pooled": round(ra, 4),
        "mcc_at_max_mcc": round(best_mcc, 4),
        "f1_at_max_mcc": round(best_f1, 4),
        "threshold": round(float(best_th), 4),
    }

# --- PRPD-style LR on physical features (agg116) ---
X_agg = data["X_mean"].reshape(2481 * 3, 58)  # per-phase mean of window feats
# add std of key impulsiveness features
std_cols = [data["idx"]["rms"], data["idx"]["crest_factor"],
            data["idx"]["kurtosis"], data["idx"]["energy"]]
X_aug = np.concatenate([X_agg, data["X_std"].reshape(2481 * 3, 58)[:, std_cols]], axis=1)
X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=0.0, neginf=0.0)
oof_lr = prpd_lr_oof(X_aug, y_flat, meas_ids)
oof_dict["prpd_lr"] = oof_lr
best_mcc, best_f1, best_th = -1, 0, 0.5
for th in np.linspace(0.01, 0.99, 200):
    m, f = mcc_f1(y_flat, oof_lr, th)
    if m > best_mcc:
        best_mcc, best_f1, best_th = m, f, th
results["prpd_lr"] = {
    "type": "classical ML (PRPD-style features + LR, same folds/OOF)",
    "phase_pr_auc_pooled": round(pr_auc(y_flat, oof_lr), 4),
    "phase_roc_auc_pooled": round(roc_auc(y_flat, oof_lr), 4),
    "mcc_at_max_mcc": round(best_mcc, 4),
    "f1_at_max_mcc": round(best_f1, 4),
    "threshold": round(float(best_th), 4),
}

# --- measurement-level (noisy-OR) for the best detector ---
def meas_level(s_phase, y_phase_m, meas_ids_m):
    """noisy-OR per measurement: p = 1 - prod(1-p_phase)."""
    p_meas = np.zeros(len(meas_ids_m))
    y_meas = np.zeros(len(meas_ids_m), dtype=int)
    for i, mid in enumerate(meas_ids_m):
        p_meas[i] = 1 - np.prod(1 - s_phase[i])
        y_meas[i] = int(y_phase_m[i].any())
    return p_meas, y_meas

for name in ["energy", "prpd_lr"]:
    s_phase = oof_dict[name].reshape(2481, 3)
    pm, ym = meas_level(s_phase, y_phase, meas_ids)
    results[name]["meas_pr_auc_pooled"] = round(pr_auc(ym, pm), 4)
    results[name]["meas_roc_auc_pooled"] = round(roc_auc(ym, pm), 4)

# --- reference: E4 pooled OOF for comparison ---
e4 = np.load(ROOT / "results" / "stage1_tim" / "e4_ctx_concat" / "seeds"
             / "seed_42" / "oof.npz", allow_pickle=True)
results["_reference_E4"] = {
    "phase_pr_auc_pooled": round(pr_auc(e4["phase_targets"].ravel().astype(int),
                                        e4["phase_probs"].ravel()), 4),
    "phase_roc_auc_pooled": round(roc_auc(e4["phase_targets"].ravel().astype(int),
                                          e4["phase_probs"].ravel()), 4),
}

with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
np.savez(OUT_DIR / "oof.npz", **{k: v for k, v in oof_dict.items()})

print("=== CLASSICAL DETECTOR BASELINES (dev set, pooled-OOF protocol) ===")
print(f"{'detector':<14} {'PR-AUC':>8} {'ROC-AUC':>8} {'MCC':>7} {'F1':>7}  type")
for name, r in results.items():
    print(f"{name:<14} {r.get('phase_pr_auc_pooled',''):>8} "
          f"{r.get('phase_roc_auc_pooled',''):>8} "
          f"{r.get('mcc_at_max_mcc',''):>7} {r.get('f1_at_max_mcc',''):>7}  {r.get('type','')[:40]}")
print("\nDone. Outputs in", OUT_DIR)
