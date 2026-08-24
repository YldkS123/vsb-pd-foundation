# -*- coding: utf-8 -*-
"""Generate inference artifacts consumed by make_figures.py.

Produces three artifacts under results/figures_data/:

1. holdout_ensemble_probs.npz
   Per-phase ensemble probabilities on the 423 strict-holdout measurements
   (mean of the 5 locked fold models). Used for the reliability diagram.
   This is the *same* single blind evaluation already receipted; it only
   persists the per-sample probabilities for visualization.

2. dev_model_full_ensemble.npz
   True out-of-fold ensemble probabilities of the same 5 locked models over
   all 2481 development measurements (each measurement is scored only by the
   fold model whose validation split contains it). Reuses the cached
   features_full.npz. Kept for provenance/audit of the main-result run.

3. holdout_examples_attn.npz
   Two strict-holdout examples (one positive, one negative) with raw window
   starts/kinds, phase labels, ensemble phase probabilities, and per-window
   gated-attention weights extracted from the frozen models (no architecture
   change; weights are read out of the existing aggregator layers). This is
   visualization of the same single blind evaluation, not a second one.

Run from the project root:
    python scripts/make_figure_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vsb_pd.cyclic import CyclicPhaseModule  # noqa: E402
from vsb_pd.encoder import WindowEncoder  # noqa: E402
from vsb_pd.features import extract_physical_features  # noqa: E402
from vsb_pd.mil import MILAggregator, PhaseClassifier  # noqa: E402
from vsb_pd.model import VSBPipeline  # noqa: E402

MODEL_DIR = ROOT / "results" / "model_full"
OUT_DIR = ROOT / "results" / "figures_data"
HOLDOUT_CACHE = next((ROOT / "results" / "holdout_cache" / "windows").glob("*/holdout"))
DEV_FEATURES_CACHE = ROOT / "results" / "cached_features" / "features_full.npz"
REF_OOF = ROOT / "results" / "ablations" / "dev_k8" / "enc_dual__mil_gated_attention__ph_cyclic" / "oof.npz"
SAMPLING_RATE = 40_000_000
HIDDEN = 128
FEATURE_DIM = 58
WINDOW_LENGTH = 8192


def load_models(device: torch.device) -> list[tuple[VSBPipeline, Path]]:
    models = []
    for fpath in sorted(MODEL_DIR.glob("model_fold*.pt")):
        model = VSBPipeline(
            encoder=WindowEncoder(WINDOW_LENGTH, FEATURE_DIM, HIDDEN),
            aggregator=MILAggregator("gated_attention", HIDDEN),
            cyclic=CyclicPhaseModule(HIDDEN),
            classifier=PhaseClassifier(HIDDEN),
        ).to(device)
        ckpt = torch.load(fpath, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append((model, fpath))
    return models


def features_from_windows(windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """windows: (M, P, K, L) float32 -> (M, P, K, 58) float32."""
    M, P, K, L = windows.shape
    feats = extract_physical_features(
        windows.reshape(M * P, K, L), SAMPLING_RATE, batch_size=batch_size,
    )
    names = sorted(feats.keys())
    arr = np.stack([feats[n] for n in names], axis=-1)
    return arr.reshape(M, P, K, -1).astype(np.float32)


def ensemble_forward(
    models: list[tuple[VSBPipeline, Path]],
    windows: np.ndarray,
    features: np.ndarray,
    device: torch.device,
    batch: int = 64,
) -> np.ndarray:
    """Return ensemble mean phase probabilities (M, 3)."""
    fold_probs = []
    for model, _ in models:
        probs = []
        with torch.no_grad():
            for i in range(0, len(windows), batch):
                bw = torch.from_numpy(windows[i : i + batch]).float().to(device)
                bf = torch.from_numpy(features[i : i + batch]).float().to(device)
                logits, _ = model(bw, bf)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        fold_probs.append(np.concatenate(probs, axis=0))
    return np.mean(fold_probs, axis=0)


def gated_attention_weights(
    model: VSBPipeline,
    windows: np.ndarray,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Read out gated-attention weights for one measurement (3, K)."""
    with torch.no_grad():
        w = torch.from_numpy(windows).float().to(device)  # (1, 3, K, L)
        f = torch.from_numpy(features).float().to(device)  # (1, 3, K, 58)
        B, P, K, L = w.shape
        w_flat = w.reshape(B * P, K, L)
        f_flat = f.reshape(B * P, K, -1)
        encoded = model.encoder(w_flat, f_flat)  # (B*P, K, 128)
        scores = model.aggregator.attn(encoded).squeeze(-1)
        alpha = torch.softmax(scores / float(HIDDEN) ** 0.5, dim=-1)
        gate = torch.sigmoid(model.aggregator.gate(encoded).squeeze(-1))
        weights = (alpha * gate).cpu().numpy().reshape(P, K)
    return weights.astype(np.float64)


def make_holdout_probs(models: list[tuple[VSBPipeline, Path]], device: torch.device) -> None:
    files = sorted(HOLDOUT_CACHE.glob("*.npz"))
    all_probs, all_labels, all_mids = [], [], []
    chunk = 64
    for start in range(0, len(files), chunk):
        part = files[start : start + chunk]
        wins, labs, mids = [], [], []
        for f in part:
            data = np.load(f, allow_pickle=False)
            wins.append(data["windows"].astype(np.float32))
            labs.append(data["targets"].astype(np.int8))
            mids.append(int(data["measurement_id"].item()))
        windows = np.stack(wins)
        labels = np.stack(labs)
        features = features_from_windows(windows)
        probs = ensemble_forward(models, windows, features, device)
        all_probs.append(probs)
        all_labels.append(labels)
        all_mids.extend(mids)
        print(f"  holdout {len(all_mids)}/{len(files)} done")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "holdout_ensemble_probs.npz",
        probs=np.concatenate(all_probs, axis=0),
        labels=np.concatenate(all_labels, axis=0),
        measurement_ids=np.asarray(all_mids, dtype=np.int64),
    )
    print("saved:", OUT_DIR / "holdout_ensemble_probs.npz")


def make_dev_ensemble(models: list[tuple[VSBPipeline, Path]], device: torch.device) -> None:
    """True out-of-fold ensemble: each measurement is scored only by the fold
    model whose validation split contains it (i.e., the model that did NOT
    train on it). The fold assignment is taken from the locked dev8 OOF file,
    which shares the exact same 5-fold split as every baseline and ablation.
    """
    d = np.load(DEV_FEATURES_CACHE, allow_pickle=False)
    windows = d["windows"].astype(np.float32)
    feat = d["feat_array"].astype(np.float32)
    labels = d["labels"].astype(np.int8)
    mids = d["measurement_ids"]
    fold_assign = np.load(REF_OOF, allow_pickle=False)["fold_assign"]
    probs = np.zeros((len(mids), 3), dtype=np.float64)
    for fi, (model, _) in enumerate(models):
        idx = np.where(fold_assign == fi)[0]
        probs[idx] = ensemble_forward([(model, _)], windows[idx], feat[idx], device)
    np.savez(
        OUT_DIR / "dev_model_full_ensemble.npz",
        probs=probs,
        labels=labels,
        measurement_ids=mids,
        fold_assign=fold_assign,
    )
    print("saved:", OUT_DIR / "dev_model_full_ensemble.npz")


def pick_holdout_examples() -> tuple[int, int]:
    d = np.load(OUT_DIR / "holdout_ensemble_probs.npz", allow_pickle=False)
    probs, targets, mids = d["probs"], d["labels"], d["measurement_ids"]
    # Positive: a fully-positive measurement with the highest minimum probability.
    pos = [i for i in range(len(mids)) if targets[i].sum() == 3]
    pos_i = max(pos, key=lambda i: float(probs[i].min()))
    # Negative: a fully-negative measurement with the lowest maximum probability.
    neg = [i for i in range(len(mids)) if targets[i].sum() == 0]
    neg_i = min(neg, key=lambda i: float(probs[i].max()))
    print(f"holdout positive example: mid={int(mids[pos_i])}, probs={np.round(probs[pos_i], 4)}")
    print(f"holdout negative example: mid={int(mids[neg_i])}, max prob={probs[neg_i].max():.4f}")
    return int(mids[pos_i]), int(mids[neg_i])


def make_holdout_examples(models: list[tuple[VSBPipeline, Path]], device: torch.device) -> None:
    pos_mid, neg_mid = pick_holdout_examples()
    files = {int(f.stem): f for f in HOLDOUT_CACHE.glob("*.npz")}

    records = []
    for mid in (pos_mid, neg_mid):
        with np.load(files[mid], allow_pickle=False) as data:
            windows = data["windows"].astype(np.float32)  # (3, 8, 8192)
            starts = data["starts"].astype(np.int64)
            kinds = data["kinds"].astype(np.uint8)
            targets = data["targets"].astype(np.int8)
        features = features_from_windows(windows[None, ...])
        probs = ensemble_forward(models, windows[None, ...], features, device)[0]
        weights = np.mean(
            [gated_attention_weights(model, windows[None, ...], features, device) for model, _ in models],
            axis=0,
        )
        records.append(
            {
                "measurement_id": int(mid),
                "starts": starts,
                "kinds": kinds,
                "targets": targets,
                "probs": probs,
                "attn": weights,
            }
        )
        print(f"  example mid={mid}: probs={np.round(probs, 4)}, targets={targets.tolist()}")

    np.savez(
        OUT_DIR / "holdout_examples_attn.npz",
        positive_mid=records[0]["measurement_id"],
        negative_mid=records[1]["measurement_id"],
        starts_pos=records[0]["starts"],
        kinds_pos=records[0]["kinds"],
        targets_pos=records[0]["targets"],
        probs_pos=records[0]["probs"],
        attn_pos=records[0]["attn"],
        starts_neg=records[1]["starts"],
        kinds_neg=records[1]["kinds"],
        targets_neg=records[1]["targets"],
        probs_neg=records[1]["probs"],
        attn_neg=records[1]["attn"],
    )
    print("saved:", OUT_DIR / "holdout_examples_attn.npz")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    models = load_models(device)
    print(f"loaded {len(models)} fold models from {MODEL_DIR}")
    make_holdout_probs(models, device)
    make_dev_ensemble(models, device)
    make_holdout_examples(models, device)


if __name__ == "__main__":
    main()
