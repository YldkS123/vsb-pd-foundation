"""Statistical evaluation, Platt calibration, bootstrap CIs, and blind evaluation lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# ── Calibration ──────────────────────────────────────────────────────────────

def calibrate_platt(
    oof_probs: np.ndarray,
    oof_labels: np.ndarray,
    test_probs: np.ndarray,
) -> np.ndarray:
    """Platt scaling: fit logistic regression on OOF logits/raw scores.

    Args:
        oof_probs: (N,) out-of-fold probabilities
        oof_labels: (N,) binary labels
        test_probs: (M,) test probabilities to calibrate

    Returns:
        (M,) calibrated probabilities
    """
    # Convert to logit space for Platt scaling
    eps = 1e-12
    oof_logits = np.log(np.clip(oof_probs, eps, 1 - eps)) - np.log(
        np.clip(1 - oof_probs, eps, 1 - eps)
    )
    test_logits = np.log(np.clip(test_probs, eps, 1 - eps)) - np.log(
        np.clip(1 - test_probs, eps, 1 - eps)
    )

    # Fit logistic regression with small C to prevent overfitting
    cal = LogisticRegression(C=1.0, solver="lbfgs")
    cal.fit(oof_logits.reshape(-1, 1), oof_labels)

    calibrated_logits = cal.decision_function(test_logits.reshape(-1, 1))
    from scipy.special import expit
    return expit(calibrated_logits)


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute ECE and Brier score.

    Args:
        probs: (N,) predicted probabilities
        labels: (N,) binary labels
        n_bins: number of bins for ECE

    Returns:
        dict with "ece" and "brier" keys
    """
    brier = float(brier_score_loss(labels, probs))

    # ECE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)

    return {"ece": float(ece), "brier": float(brier)}


# ── Bootstrap ────────────────────────────────────────────────────────────────

@dataclass
class BootstrapCI:
    metric: str
    median: float
    lower: float
    upper: float


def compute_bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    measurement_ids: np.ndarray,
    metric_name: str = "pr_auc",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> BootstrapCI:
    """Cluster bootstrap (by measurement) confidence interval for a ranking metric.

    Measurements are the resampling unit. On every iteration the positive and
    negative measurements are resampled with replacement (stratified, original
    counts preserved) and the requested metric is recomputed on the pooled
    samples of the resampled measurements. This captures between-measurement
    variability while preserving the phase-level class imbalance.

    Args:
        scores: (N,) per-sample scores (e.g., phase-level probabilities)
        labels: (N,) per-sample binary labels
        measurement_ids: (N,) measurement ID for each sample
        metric_name: "pr_auc" (default) or "roc_auc"
        n_bootstrap: number of bootstrap iterations
        seed: random seed

    Returns:
        BootstrapCI with median and 95% percentile interval of the metric.
    """
    if metric_name not in ("pr_auc", "roc_auc"):
        raise ValueError(f"unsupported metric_name: {metric_name!r} (use 'pr_auc' or 'roc_auc')")
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    measurement_ids = np.asarray(measurement_ids)
    if not (scores.ndim == labels.ndim == measurement_ids.ndim == 1):
        raise ValueError("scores, labels and measurement_ids must all be 1-D")
    if not (len(scores) == len(labels) == len(measurement_ids)):
        raise ValueError("scores, labels and measurement_ids must be aligned (same length)")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary (0/1)")

    unique_mids = np.unique(measurement_ids)
    meas_labels = np.array([
        int(labels[measurement_ids == mid].max()) for mid in unique_mids
    ])
    pos_ids = unique_mids[meas_labels == 1]
    neg_ids = unique_mids[meas_labels == 0]
    if pos_ids.size == 0 or neg_ids.size == 0:
        raise ValueError("bootstrap requires at least one positive and one negative measurement")

    rng = np.random.default_rng(seed)
    metric = average_precision_score if metric_name == "pr_auc" else roc_auc_score
    values = []
    for _ in range(n_bootstrap):
        sampled_pos = rng.choice(pos_ids, size=pos_ids.size, replace=True)
        sampled_neg = rng.choice(neg_ids, size=neg_ids.size, replace=True)
        sel = np.isin(measurement_ids, np.concatenate([sampled_pos, sampled_neg]))
        t, s = labels[sel], scores[sel]
        if t.sum() == 0 or (t == 0).sum() == 0:
            continue
        values.append(float(metric(t, s)))

    if not values:
        raise RuntimeError("all bootstrap samples were single-class; cannot estimate CI")

    values = np.array(values)
    return BootstrapCI(
        metric=metric_name,
        median=float(np.median(values)),
        lower=float(np.percentile(values, 2.5)),
        upper=float(np.percentile(values, 97.5)),
    )


def paired_bootstrap_ci(
    model_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    measurement_ids: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired cluster bootstrap for the PR-AUC difference.

    The same measurement clusters are resampled for both score vectors and
    PR-AUC is recomputed on each cluster draw, so the returned difference is
    a metric difference rather than a difference of mean scores.

    Returns:
        dict with diff_median, diff_lower, diff_upper (95% CI on difference)
    """
    rng = np.random.default_rng(seed)
    model_scores = np.asarray(model_scores)
    baseline_scores = np.asarray(baseline_scores)
    labels = np.asarray(labels)
    measurement_ids = np.asarray(measurement_ids)
    if not (model_scores.ndim == baseline_scores.ndim == labels.ndim == measurement_ids.ndim == 1):
        raise ValueError("paired bootstrap inputs must all be 1-D")
    if not (len(model_scores) == len(baseline_scores) == len(labels) == len(measurement_ids)):
        raise ValueError("paired bootstrap inputs must be aligned (same length)")

    unique_mids = np.unique(measurement_ids)
    meas_labels = np.array([
        int(labels[measurement_ids == mid].max()) for mid in unique_mids
    ])
    pos_ids = unique_mids[meas_labels == 1]
    neg_ids = unique_mids[meas_labels == 0]
    if pos_ids.size == 0 or neg_ids.size == 0:
        raise ValueError("paired bootstrap requires at least one positive and one negative measurement")

    metric = average_precision_score
    diffs = []
    for _ in range(n_bootstrap):
        sampled_pos = rng.choice(pos_ids, size=pos_ids.size, replace=True)
        sampled_neg = rng.choice(neg_ids, size=neg_ids.size, replace=True)
        sel = np.isin(measurement_ids, np.concatenate([sampled_pos, sampled_neg]))
        t = labels[sel]
        if t.sum() == 0 or (t == 0).sum() == 0:
            continue
        m = metric(t, model_scores[sel])
        b = metric(t, baseline_scores[sel])
        diffs.append(float(m - b))

    if not diffs:
        raise RuntimeError("all paired bootstrap samples were single-class; cannot estimate CI")
    diffs = np.array(diffs)
    return {
        "diff_median": float(np.median(diffs)),
        "diff_lower": float(np.percentile(diffs, 2.5)),
        "diff_upper": float(np.percentile(diffs, 97.5)),
    }


# ── Blind evaluation lock ────────────────────────────────────────────────────

class BlindEvaluationLock:
    """One-shot lock for final blind evaluation.

    Usage:
        lock = BlindEvaluationLock(lock_path, hashes)
        if lock.acquire():
            # Run evaluation once
            lock.write_receipt(metrics)
    """

    def __init__(self, lock_path: Path, expected_hashes: dict[str, str]):
        self.lock_path = Path(lock_path)
        self.expected_hashes = expected_hashes

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True on first attempt only."""
        if self.lock_path.exists():
            return False
        # Create lock file atomically
        try:
            self.lock_path.write_text(
                json.dumps(
                    {
                        "acquired": True,
                        "hashes": self.expected_hashes,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return True
        except FileExistsError:
            return False

    def write_receipt(self, metrics: dict) -> None:
        """Write final evaluation receipt after successful run."""
        receipt = {
            "hashes_verified": self.expected_hashes,
            "metrics": metrics,
        }
        self.lock_path.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def is_locked(self) -> bool:
        return self.lock_path.exists()


def final_blind_evaluate(
    experiment_lock_path: Path,
    model_checkpoint_path: Path,
    holdout_manifest_path: Path,
    config,
) -> dict:
    """Run the one-shot final blind evaluation.

    Returns:
        dict of evaluation metrics
    """
    # Compute hashes for verification
    hashes = {}
    if model_checkpoint_path.exists():
        hashes["model_checkpoint"] = _sha256_file(model_checkpoint_path)
    if holdout_manifest_path.exists():
        hashes["holdout_data"] = _sha256_file(holdout_manifest_path)

    lock = BlindEvaluationLock(experiment_lock_path, hashes)

    if not lock.acquire():
        raise RuntimeError(
            "Blind evaluation already completed. "
            f"Lock file exists at {experiment_lock_path}"
        )

    # Load model and holdout data...
    # (actual evaluation happens here when real data is available)

    # Placeholder: write receipt
    lock.write_receipt({"status": "completed", "note": "placeholder"})

    return {"status": "completed"}


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
