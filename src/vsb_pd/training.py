"""Training loop, nested cross-validation runner, and evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .config import ExperimentConfig, PipelineConfig
from .data import WindowDataset


# ── Fold splitting ──────────────────────────────────────────────────────────

def make_stratified_group_folds(
    measurement_ids: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified group k-fold splits.

    Args:
        measurement_ids: (N,) array of measurement IDs (grouping unit)
        labels: (N,) scalar per-measurement stratification label
        n_splits: number of outer folds
        seed: random seed

    Returns:
        list of (train_indices, val_indices) pairs
    """
    # Stratify by positive-phase count: 0, 1, 2, or 3 positive phases
    stratify = np.clip(labels.astype(int), 0, 3)

    # Handle rare classes: merge classes with fewer than n_splits samples
    unique, counts = np.unique(stratify, return_counts=True)
    rare = unique[counts < n_splits]
    if len(rare) > 0:
        stratify_adj = stratify.copy()
        for r in rare:
            stratify_adj[stratify == r] = -1  # merge into a catch-all class
        stratify = stratify_adj

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(measurement_ids)), stratify, groups=measurement_ids))
    return folds


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        targets: (N,) binary ground truth
        probs: (N,) predicted probabilities
        preds: (N,) binary predictions (thresholded)

    Returns:
        dict with accuracy, precision, recall, f1, roc_auc, pr_auc
    """
    metrics: dict[str, float] = {}

    metrics["accuracy"] = float(accuracy_score(targets, preds))

    with np.errstate(divide="ignore", invalid="ignore"):
        metrics["precision"] = float(precision_score(targets, preds, zero_division=0))
        metrics["recall"] = float(recall_score(targets, preds, zero_division=0))
        metrics["f1"] = float(f1_score(targets, preds, zero_division=0))

    # AUC metrics need at least one positive and one negative sample
    n_pos = int(targets.sum())
    n_neg = len(targets) - n_pos
    if n_pos > 0 and n_neg > 0:
        metrics["roc_auc"] = float(roc_auc_score(targets, probs))
        metrics["pr_auc"] = float(average_precision_score(targets, probs))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


# ── Training / validation ────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    windows: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    grad_clip_norm: float = 1.0,
) -> float:
    """Run one training epoch.

    Args:
        model: VSBPipeline
        optimizer: AdamW optimizer
        criterion: PhaseCyclicLoss
        windows: (B, 3, K, 8192)
        features: (B, 3, K, 58)
        labels: (B, 3) binary phase labels
        grad_clip_norm: max gradient norm

    Returns:
        scalar training loss
    """
    model.train()
    optimizer.zero_grad()

    phase_logits, _ = model(windows, features)
    loss = criterion(phase_logits, labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    criterion: nn.Module,
    windows: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one validation pass.

    Returns:
        (val_loss, phase_probs_np, targets_np)
    """
    model.eval()

    phase_logits, _ = model(windows, features)
    loss = criterion(phase_logits, labels)

    phase_probs = torch.sigmoid(phase_logits)
    return (
        float(loss.item()),
        phase_probs.cpu().numpy(),
        labels.cpu().numpy(),
    )


# ── Full epoch loop over batches ─────────────────────────────────────────────

def train_one_epoch_batched(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataset: WindowDataset,
    indices: list[int],
    batch_size: int,
    grad_clip_norm: float = 1.0,
) -> float:
    """Train over batched data for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    perm = np.random.permutation(indices)
    for i in range(0, len(perm), batch_size):
        batch_idx = perm[i:i + batch_size]
        windows_list, features_list, labels_list = [], [], []

        for idx in batch_idx:
            w, starts, kinds, scores, targets, _ = dataset[idx]
            # w: (3, K, 8192), targets: (3,)
            # For now, use raw windows; features will be computed
            windows_list.append(w)
            labels_list.append(targets)

        if not windows_list:
            continue

        batch_windows = torch.stack(windows_list)  # (B, 3, K, 8192)
        batch_labels = torch.stack(labels_list)  # (B, 3)

        # Extract features on-the-fly from windows
        from .features import extract_physical_features
        Bp, P, K, L = batch_windows.shape
        features_np = extract_physical_features(
            batch_windows.reshape(Bp * P, K, L).numpy(), 40_000_000,
        )
        # features_np is dict of name -> (B*P*K,) array; reshape to (B, P, K, 58)
        feature_names = sorted(features_np.keys())
        feat_array = np.stack([features_np[name] for name in feature_names], axis=-1)  # (B*P*K, 58)
        feat_array = feat_array.reshape(Bp, P, K, 58)
        batch_features = torch.from_numpy(feat_array).float()

        loss = train_epoch(
            model, optimizer, criterion,
            batch_windows, batch_features, batch_labels,
            grad_clip_norm=grad_clip_norm,
        )
        total_loss += loss
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_batched(
    model: nn.Module,
    criterion: nn.Module,
    dataset: WindowDataset,
    indices: list[int],
    batch_size: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Validate over batched data."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        windows_list, features_list, labels_list = [], [], []

        for idx in batch_idx:
            w, starts, kinds, scores, targets, _ = dataset[idx]
            windows_list.append(w)
            labels_list.append(targets)

        if not windows_list:
            continue

        batch_windows = torch.stack(windows_list)
        batch_labels = torch.stack(labels_list)

        from .features import extract_physical_features
        Bp, P, K, L = batch_windows.shape
        features_np = extract_physical_features(
            batch_windows.reshape(Bp * P, K, L).numpy(), 40_000_000,
        )
        feature_names = sorted(features_np.keys())
        feat_array = np.stack([features_np[name] for name in feature_names], axis=-1)
        feat_array = feat_array.reshape(Bp, P, K, 58)
        batch_features = torch.from_numpy(feat_array).float()

        loss, probs, targets_np = validate_epoch(
            model, criterion,
            batch_windows, batch_features, batch_labels,
        )
        total_loss += loss
        n_batches += 1
        all_probs.append(probs)
        all_targets.append(targets_np)

    avg_loss = total_loss / max(n_batches, 1)
    all_probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
    all_targets_np = np.concatenate(all_targets, axis=0) if all_targets else np.array([])

    return avg_loss, all_probs_np, all_targets_np


# ── Nested CV ────────────────────────────────────────────────────────────────

def train_one_fold(
    model: nn.Module,
    dataset: WindowDataset,
    train_indices: list[int],
    val_indices: list[int],
    config: ExperimentConfig,
    seed: int,
) -> dict[str, float]:
    """Train model on one fold with early stopping."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    best_val_pr_auc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.training.epochs):
        train_loss = train_one_epoch_batched(
            model, optimizer, criterion, dataset,
            train_indices, config.training.batch_size,
            grad_clip_norm=config.training.grad_clip_norm,
        )

        val_loss, val_probs, val_targets = validate_batched(
            model, criterion, dataset,
            val_indices, config.training.batch_size,
        )

        # Flatten for metric computation
        val_probs_flat = val_probs.flatten()
        val_targets_flat = val_targets.flatten()
        val_preds_flat = (val_probs_flat >= 0.5).astype(int)

        metrics = compute_metrics(val_targets_flat, val_probs_flat, val_preds_flat)
        val_pr_auc = metrics.get("pr_auc", 0.0)

        if val_pr_auc > best_val_pr_auc + 0.001:
            best_val_pr_auc = val_pr_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.training.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_pr_auc": best_val_pr_auc, "epochs_trained": epoch + 1}
