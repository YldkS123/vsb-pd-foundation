"""Full baseline comparison with the SAME dev split locks as the VSB MIL model.

Reproduces the exact StratifiedGroupKFold(5, seed=42) development splits, computes
per-fold OOF probabilities for every baseline, saves them for later audit, and
reports metrics at multiple thresholds (0.5, max-MCC, recall>=0.5, recall>=0.8).

Feature sets:
  agg116   : per-phase mean+std over 8 windows (58*2=116 dims)  [existing cache]
  agg406   : per-phase 7 statistics (mean/std/min/max/median/skew/kurtosis) -> 406 dims
  flatten  : per-phase raw per-window features, 8*58=464 dims (no aggregation)

Usage:
  python scripts/compare_baselines.py [--cache results/cached_features/features_full.npz]
                                      [--features agg116,agg406,flatten]
                                      [--models lr,rf,lgbm]
                                      [--out results/baseline_full_comparison.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vsb_pd.training import make_stratified_group_folds

SEED = 42
N_FOLDS = 5


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def build_agg406(feat_array: np.ndarray) -> np.ndarray:
    """Per-phase 7-statistic aggregation: (M,3,8,58) -> (M*3, 406)."""
    from scipy.stats import skew, kurtosis

    M, P, K, F = feat_array.shape
    x = feat_array.astype(np.float64)  # (M,P,K,F)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = [
            x.mean(axis=2),
            x.std(axis=2),
            x.min(axis=2),
            x.max(axis=2),
            np.median(x, axis=2),
            skew(x, axis=2),
            kurtosis(x, axis=2),
        ]
    X = np.concatenate(stats, axis=-1)  # (M,P,7F)
    return np.nan_to_num(X.reshape(M * P, -1), nan=0.0, posinf=0.0, neginf=0.0)


def build_flatten(feat_array: np.ndarray) -> np.ndarray:
    """Per-phase window flattening: (M,3,8,58) -> (M*3, 464)."""
    M, P, K, F = feat_array.shape
    X = feat_array.astype(np.float64).reshape(M * P, K * F)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Per-fold OOF training (fixed hyperparameters, locked on dev folds)
# --------------------------------------------------------------------------- #
def _fit_predict_lr(X_tr, y_tr, X_va):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    model = LogisticRegression(
        C=10.0, max_iter=5000, solver="lbfgs", penalty="l2",
        class_weight=None, random_state=SEED,
    )
    model.fit(scaler.fit_transform(X_tr), y_tr)
    return model.predict_proba(scaler.transform(X_va))[:, 1]


def _fit_predict_rf(X_tr, y_tr, X_va):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=5,
        class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1]


def _fit_predict_lgbm(X_tr, y_tr, X_va):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=-1, num_leaves=63, learning_rate=0.1,
        reg_alpha=0.0, reg_lambda=0.0, class_weight="balanced",
        random_state=SEED, verbose=-1, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1]


FITTERS = {"lr": _fit_predict_lr, "rf": _fit_predict_rf, "lgbm": _fit_predict_lgbm}


def run_oof(model_name: str, X: np.ndarray, y: np.ndarray, folds) -> tuple[np.ndarray, dict]:
    """Return OOF probabilities and per-fold elapsed times."""
    oof = np.zeros(len(y), dtype=np.float64)
    per_fold_time = {}
    for fi, (tr, va) in enumerate(folds):
        t0 = time.time()
        oof[va] = FITTERS[model_name](X[tr], y[tr], X[va])
        per_fold_time[f"fold{fi + 1}_s"] = round(time.time() - t0, 1)
    return oof, per_fold_time


# --------------------------------------------------------------------------- #
# Threshold-aware metrics (thresholds chosen on OOF only)
# --------------------------------------------------------------------------- #
def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

    preds = (p >= t).astype(int)
    return {
        "threshold": float(t),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, preds)),
    }


def full_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.metrics import matthews_corrcoef

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}

    # Threshold scans on OOF
    grid = np.linspace(0.001, 0.999, 999)
    mccs = [matthews_corrcoef(y, (p >= t).astype(int)) for t in grid]
    t_mcc = float(grid[int(np.argmax(mccs))])

    # Recall-target thresholds: highest threshold with recall >= target
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    t_r50 = t_r80 = 0.5
    cand50 = thr[rec[:-1] >= 0.5]
    if len(cand50):
        t_r50 = float(cand50.max())
    cand80 = thr[rec[:-1] >= 0.8]
    if len(cand80):
        t_r80 = float(cand80.max())

    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "t_0.5": threshold_metrics(y, p, 0.5),
        "t_max_mcc": threshold_metrics(y, p, t_mcc),
        "t_recall_0.5": threshold_metrics(y, p, t_r50),
        "t_recall_0.8": threshold_metrics(y, p, t_r80),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cached_features/features_full.npz")
    ap.add_argument("--features", default="agg116,agg406,flatten")
    ap.add_argument("--models", default="lr,rf,lgbm")
    ap.add_argument("--out", default="results/baseline_full_comparison.json")
    ap.add_argument("--seed", type=int, default=SEED, help="fold split seed (must match the reported model run)")
    ap.add_argument("--oof-dir", default="results/baseline_oof", help="directory for per-fold OOF npz artifacts")
    args = ap.parse_args()

    cache_path = Path(args.cache)
    d = np.load(cache_path, allow_pickle=True)
    feat_array = d["feat_array"]          # (M, 3, 8, 58)
    labels = d["labels"]                  # (M, 3)
    mids = d["measurement_ids"]           # (M,)
    agg_X = d["aggregated_X"]             # (M*3, 116)
    agg_y = d["aggregated_y"]
    print(f"Cache: {cache_path}  feat={feat_array.shape} labels={labels.shape} mids={mids.shape}")

    y_phase = labels.reshape(-1).astype(int)
    strat = labels.sum(axis=1).astype(int)
    folds = make_stratified_group_folds(mids, strat, n_splits=N_FOLDS, seed=SEED)
    folds = make_stratified_group_folds(mids, strat, n_splits=N_FOLDS, seed=args.seed)
    fold_sizes = [(len(tr), len(va)) for tr, va in folds]
    print(f"Dev folds (train,val): {fold_sizes}")

    # Fold-assignment fingerprint for audit (same locks as VSB model training)
    fold_assign = np.zeros(len(mids), dtype=np.int8)
    for fi, (_, va) in enumerate(folds):
        fold_assign[va] = fi
    fp = hashlib.sha256(fold_assign.tobytes()).hexdigest()[:16]

    feature_sets = {}
    if "agg116" in args.features.split(","):
        feature_sets["agg116"] = (agg_X, agg_y)
    if "agg406" in args.features.split(","):
        feature_sets["agg406"] = (build_agg406(feat_array), agg_y)
    if "flatten" in args.features.split(","):
        feature_sets["flatten"] = (build_flatten(feat_array), agg_y)

    out_dir = Path("results/baseline_oof")
    out_dir = Path(args.oof_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "cache": str(cache_path),
        "n_measurements": int(len(mids)),
        "n_phases": int(len(y_phase)),
        "fold_sizes": fold_sizes,
        "fold_fingerprint_sha256_16": fp,
        "features": {},
    }

    for fs_name, (X, y) in feature_sets.items():
        print(f"\n=== Feature set: {fs_name}  X={X.shape} ===")
        results["features"][fs_name] = {"dim": int(X.shape[1]), "models": {}}
        for model_name in args.models.split(","):
            t0 = time.time()
            oof, per_fold_time = run_oof(model_name, X, y, folds)
            elapsed = time.time() - t0
            metrics = full_metrics(y, oof)
            print(f"  {model_name:<5} PR-AUC={metrics['pr_auc']:.4f} ROC-AUC={metrics['roc_auc']:.4f} "
                  f"({elapsed:.0f}s)  F1@0.5={metrics['t_0.5']['f1']:.4f} "
                  f"F1@maxMCC={metrics['t_max_mcc']['f1']:.4f}")

            oof_path = out_dir / f"{fs_name}_{model_name}_oof.npz"
            np.savez_compressed(oof_path, oof_probs=oof, labels=y, groups=d["aggregated_groups"], fold_assign=fold_assign)

            results["features"][fs_name]["models"][model_name] = {
                "oof_path": str(oof_path),
                "elapsed_s": round(elapsed, 1),
                "per_fold_time_s": per_fold_time,
                "metrics_oof": metrics,
            }

    # VSB MIL reference from the locked model CV
    vsb_path = Path("results/model_full/cv_summary.json")
    if vsb_path.exists():
        vsb = json.loads(vsb_path.read_text(encoding="utf-8"))
        results["vsb_mil"] = {
            "source": str(vsb_path),
            "mean_pr_auc": float(vsb.get("mean_pr_auc", float("nan"))),
        }
        print(f"\nVSB MIL reference: mean PR-AUC={results['vsb_mil']['mean_pr_auc']:.4f}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
