"""Tests for statistical evaluation, calibration, and blind evaluation."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from vsb_pd.evaluation import (
    calibrate_platt,
    compute_calibration_metrics,
    paired_bootstrap_ci,
    compute_bootstrap_ci,
    BlindEvaluationLock,
)


def test_calibrate_platt_runs():
    rng = np.random.default_rng(42)
    oof_probs = rng.uniform(0, 1, 100)
    oof_labels = (oof_probs + 0.1 * rng.normal(0, 1, 100) > 0.5).astype(int)
    test_probs = rng.uniform(0, 1, 50)

    calibrated = calibrate_platt(oof_probs, oof_labels, test_probs)
    assert calibrated.shape == test_probs.shape
    assert (calibrated >= 0).all() and (calibrated <= 1).all()


def test_calibrate_platt_monotonic():
    rng = np.random.default_rng(42)
    oof_probs = np.sort(rng.uniform(0, 1, 100))
    oof_labels = (oof_probs > 0.5).astype(int)
    test_probs = np.linspace(0, 1, 20)

    calibrated = calibrate_platt(oof_probs, oof_labels, test_probs)
    diffs = np.diff(calibrated)
    assert (diffs >= -0.01).all(), "Calibrated probs should be monotonic"


def test_compute_calibration_metrics():
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, 100)
    labels = rng.integers(0, 2, 100)

    metrics = compute_calibration_metrics(probs, labels, n_bins=10)
    assert "ece" in metrics
    assert "brier" in metrics
    assert metrics["ece"] >= 0
    assert metrics["brier"] >= 0


def test_compute_calibration_metrics_perfect():
    probs = np.array([0.1, 0.9, 0.1, 0.9])
    labels = np.array([0, 1, 0, 1])
    metrics = compute_calibration_metrics(probs, labels, n_bins=2)
    assert metrics["brier"] < 0.02


def test_paired_bootstrap_ci():
    rng = np.random.default_rng(42)
    # Cluster bootstrap needs both positive and negative measurements, so
    # keep labels constant within each measurement and reserve both classes.
    n_meas = 20
    per_meas = 5
    meas_labels = np.array([1] * 8 + [0] * (n_meas - 8))
    measurement_ids = np.repeat(np.arange(n_meas), per_meas)
    labels = np.repeat(meas_labels, per_meas)
    model_scores = rng.normal(0.8, 0.05, measurement_ids.size)
    baseline_scores = rng.normal(0.75, 0.05, measurement_ids.size)

    result = paired_bootstrap_ci(
        model_scores, baseline_scores, labels, measurement_ids, n_bootstrap=200,
    )
    assert "diff_median" in result
    assert "diff_lower" in result
    assert "diff_upper" in result
    assert result["diff_lower"] <= result["diff_median"] <= result["diff_upper"]


def test_compute_bootstrap_ci_pr_auc_median_tracks_full_data():
    rng = np.random.default_rng(7)
    n_meas = 40
    measurement_ids = np.repeat(np.arange(n_meas), 5)
    base = np.linspace(-1.5, 1.5, n_meas)
    scores = np.repeat(base, 5) + rng.normal(0, 0.1, measurement_ids.size)
    labels = (scores > 0).astype(int)

    result = compute_bootstrap_ci(scores, labels, measurement_ids, n_bootstrap=300)
    full = average_precision_score(labels, scores)
    assert result.metric == "pr_auc"
    assert result.lower <= result.median <= result.upper
    assert abs(result.median - full) < 0.1, f"median {result.median:.3f} vs full {full:.3f}"


def test_compute_bootstrap_ci_perfect_classifier_returns_pr_auc_one():
    # scores == labels -> every resample is a perfect ranking, PR-AUC must be 1.0.
    # The old implementation (mean of scores) would have returned ~0.5 here.
    n_neg, n_pos = 10, 10
    measurement_ids = np.repeat(np.arange(n_neg + n_pos), 3)
    labels = np.array([0] * (n_neg * 3) + [1] * (n_pos * 3))
    scores = labels.astype(float).copy()

    result = compute_bootstrap_ci(scores, labels, measurement_ids, n_bootstrap=100)
    assert result.median == 1.0
    assert result.lower == 1.0
    assert result.upper == 1.0


def test_compute_bootstrap_ci_roc_auc():
    rng = np.random.default_rng(11)
    n_meas = 30
    measurement_ids = np.repeat(np.arange(n_meas), 4)
    scores = rng.normal(size=measurement_ids.size)
    labels = (scores > 0).astype(int)

    result = compute_bootstrap_ci(
        scores, labels, measurement_ids, metric_name="roc_auc", n_bootstrap=200,
    )
    full = roc_auc_score(labels, scores)
    assert result.metric == "roc_auc"
    assert result.lower <= result.median <= result.upper
    assert abs(result.median - full) < 0.1


def test_compute_bootstrap_ci_deterministic():
    rng = np.random.default_rng(3)
    measurement_ids = np.repeat(np.arange(25), 4)
    scores = rng.normal(size=measurement_ids.size)
    labels = (scores > 0).astype(int)

    a = compute_bootstrap_ci(scores, labels, measurement_ids, n_bootstrap=150)
    b = compute_bootstrap_ci(scores, labels, measurement_ids, n_bootstrap=150)
    assert (a.median, a.lower, a.upper) == (b.median, b.lower, b.upper)


def test_compute_bootstrap_ci_requires_both_classes():
    measurement_ids = np.repeat(np.arange(10), 3)
    labels = np.zeros(30, dtype=int)
    scores = np.zeros(30)
    with pytest.raises(ValueError, match="positive and one negative"):
        compute_bootstrap_ci(scores, labels, measurement_ids, n_bootstrap=50)


def test_compute_bootstrap_ci_unsupported_metric():
    rng = np.random.default_rng(1)
    measurement_ids = np.repeat(np.arange(20), 3)
    scores = rng.normal(size=60)
    labels = (scores > 0).astype(int)
    with pytest.raises(ValueError, match="unsupported metric_name"):
        compute_bootstrap_ci(scores, labels, measurement_ids, metric_name="f1", n_bootstrap=50)


def test_compute_bootstrap_ci_requires_aligned_inputs():
    measurement_ids = np.repeat(np.arange(10), 3)
    labels = np.zeros(30, dtype=int)
    scores = np.zeros(29)
    with pytest.raises(ValueError, match="aligned"):
        compute_bootstrap_ci(scores, labels, measurement_ids, n_bootstrap=50)


def test_blind_evaluation_lock_prevents_double_run(tmp_path):
    lock_path = tmp_path / "final_eval.lock"
    hashes = {"model_checkpoint": "abc123def456", "holdout_data": "789ghi012jkl"}

    lock = BlindEvaluationLock(lock_path, hashes)
    assert lock.acquire()

    lock2 = BlindEvaluationLock(lock_path, hashes)
    assert not lock2.acquire()


def test_blind_evaluation_lock_rejects_hash_mismatch(tmp_path):
    lock_path = tmp_path / "final_eval.lock"
    hashes = {"model_checkpoint": "abc123def456"}

    lock = BlindEvaluationLock(lock_path, hashes)
    assert lock.acquire()

    receipt = {"hashes_verified": hashes, "metrics": {"pr_auc": 0.85}}
    lock.write_receipt(receipt)

    # Verify lock still prevents re-acquisition
    lock3 = BlindEvaluationLock(lock_path, {"model_checkpoint": "different"})
    assert not lock3.acquire()
