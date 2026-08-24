"""Baseline models: LR, Random Forest, LightGBM."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def aggregate_features_per_phase(
    features_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    measurement_ids: np.ndarray,
    num_phases: int = 3,
    num_windows: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-window features into per-phase features using mean and std — vectorized.

    Args:
        features_dict: feature_name -> (N_total_windows,) float32
        labels: (num_measurements, num_phases) int8
        measurement_ids: (num_measurements,)

    Returns:
        X: (M*num_phases, num_features*2) — mean+std per feature per phase
        y: (M*num_phases,) flattened labels
        groups: (M*num_phases,) measurement IDs per phase
    """
    feature_names = sorted(features_dict.keys())
    F = len(feature_names)
    M = len(measurement_ids)
    P = num_phases
    K = num_windows

    # Stack all features into (M*P*K, F)
    feat_stack = np.column_stack([features_dict[name] for name in feature_names])  # (total_windows, F)

    # Reshape to (M, P, K, F)
    feat_stack = feat_stack.reshape(M, P, K, F)

    # Compute mean and std along window axis (K)
    mean = feat_stack.mean(axis=2)  # (M, P, F)
    std = feat_stack.std(axis=2)  # (M, P, F)

    # Concatenate: (M, P, 2F) -> (M*P, 2F)
    X = np.concatenate([mean, std], axis=-1).reshape(M * P, -1)

    # Flatten labels: (M, P) -> (M*P,)
    y = labels.reshape(-1).astype(np.int64)

    # Repeat measurement_ids: (M,) -> (M*P,)
    groups = np.repeat(measurement_ids.astype(np.int64), P)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, groups


def train_lr_baseline(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv: int = 3,
    seed: int = 42,
) -> tuple[Pipeline, dict]:
    """Train Logistic Regression with scaling + group-aware grid search."""
    param_grid = {
        "lr__C": [0.01, 0.1, 1.0, 10.0],
        "lr__penalty": ["l2"],
        "lr__solver": ["lbfgs"],
        "lr__class_weight": ["balanced", None],
        "lr__max_iter": [1000],
    }
    gkf = GroupKFold(n_splits=cv)
    pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(random_state=seed))])
    gs = GridSearchCV(pipeline, param_grid, cv=gkf, scoring="average_precision", n_jobs=1, refit=True)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_


def train_rf_baseline(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv: int = 3,
    seed: int = 42,
) -> tuple[RandomForestClassifier, dict]:
    """Train Random Forest with group-aware grid search."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced", "balanced_subsample", None],
    }
    gkf = GroupKFold(n_splits=cv)
    gs = GridSearchCV(
        RandomForestClassifier(random_state=seed, n_jobs=1),
        param_grid, cv=gkf, scoring="average_precision", n_jobs=1, refit=True,
    )
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_


def train_lgbm_baseline(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv: int = 3,
    seed: int = 42,
):
    """Train LightGBM with group-aware hyperparameter search."""
    import lightgbm as lgb
    from itertools import product
    from sklearn.metrics import average_precision_score

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, -1],
        "num_leaves": [31, 63],
        "learning_rate": [0.01, 0.05, 0.1],
        "class_weight": ["balanced", None],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [0.0, 0.1],
    }
    gkf = GroupKFold(n_splits=cv)

    best_score = -1.0
    best_model = None
    best_params = {}

    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        cw = params.pop("class_weight")

        scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            tr_x, tr_y = X[train_idx], y[train_idx]
            val_x, val_y = X[val_idx], y[val_idx]
            model = lgb.LGBMClassifier(**params, random_state=seed, verbose=-1, class_weight=cw)
            model.fit(tr_x, tr_y)
            scores.append(average_precision_score(val_y, model.predict_proba(val_x)[:, 1]))

        avg = float(np.mean(scores))
        if avg > best_score:
            best_score = avg
            best_params = params | {"class_weight": cw}
            best_model = lgb.LGBMClassifier(**params, random_state=seed, verbose=-1, class_weight=cw)
            best_model.fit(X, y)

    return best_model, best_params
