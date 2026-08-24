# -*- coding: utf-8 -*-
"""Generate 80k-mainline figure data consumed by make_figures.py.

Produces under results/figures_data/:

1. holdout_ensemble_probs_80k.npz
   Per-phase 5-fold-ensemble probabilities on the frozen 423-measurement
   holdout, copied from the one-shot 80k blind evaluation prediction file
   results/blind_80k_ensemble_predictions.npz (no re-inference).

2. holdout_examples_attn_80k.npz
   Two strict-holdout examples (one positive, one negative) with window
   starts/kinds, phase labels, ensemble phase probabilities, and per-window
   Attention-MIL weights averaged over the same 5 frozen fold models.

Run from the project root:
    python scripts/make_figure_data_80k.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vsb_pd.encoder import WindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.cyclic import PhaseInteractionModule
from vsb_pd.model import VSBPipeline

MODEL_DIR = (
    ROOT
    / "results"
    / "ablations"
    / "dev_k8_blind_ckpts"
    / "enc_cnn__mil_attention__ph_mean"
)
PRED_PATH = ROOT / "results" / "blind_80k_ensemble_predictions.npz"
OUT_DIR = ROOT / "results" / "figures_data"
WINDOW_LENGTH = 8192
FEATURE_DIM = 58
HIDDEN = 128
SAMPLING_RATE = 40_000_000


def find_holdout_cache() -> Path:
    """Return the holdout window cache directory containing 423 npz files."""
    root = ROOT / "results" / "holdout_cache" / "windows"
    candidates = []
    if root.exists():
        for sub in root.iterdir():
            holdout = sub / "holdout"
            if holdout.is_dir():
                n = len(list(holdout.glob("*.npz")))
                candidates.append((n, holdout))
    candidates.sort(reverse=True)
    if not candidates:
        raise RuntimeError("holdout window cache not found")
    return candidates[0][1]


def load_models(device: torch.device) -> list[torch.nn.Module]:
    models = []
    for fpath in sorted(MODEL_DIR.glob("model_fold*.pt")):
        model = VSBPipeline(
            encoder=WindowEncoder(WINDOW_LENGTH, FEATURE_DIM, HIDDEN, branch="cnn"),
            aggregator=MILAggregator("attention", HIDDEN),
            cyclic=PhaseInteractionModule("mean", HIDDEN),
            classifier=PhaseClassifier(HIDDEN),
        ).to(device)
        ckpt = torch.load(fpath, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def attention_weights(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return Attention-MIL softmax weights for one measurement (3, K)."""
    w = torch.from_numpy(windows).float().to(device)  # (3, K, L)
    f = torch.zeros((w.shape[0], w.shape[1], FEATURE_DIM), device=device)
    encoded = model.encoder(w, f)  # (3, K, 128)
    scores = model.aggregator.attn(encoded).squeeze(-1)  # (3, K)
    alpha = torch.softmax(scores / float(HIDDEN) ** 0.5, dim=-1)
    return alpha.cpu().numpy().astype(np.float64)


def make_holdout_probs() -> None:
    d = np.load(PRED_PATH, allow_pickle=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "holdout_ensemble_probs_80k.npz",
        probs=np.asarray(d["phase_probs"], dtype=np.float64),
        labels=np.asarray(d["phase_targets"], dtype=np.float64),
        measurement_ids=np.asarray(d["measurement_ids"], dtype=np.int64),
    )
    print("saved:", OUT_DIR / "holdout_ensemble_probs_80k.npz")


def make_holdout_examples(models: list[torch.nn.Module], device: torch.device) -> None:
    d = np.load(PRED_PATH, allow_pickle=False)
    probs = np.asarray(d["phase_probs"], dtype=np.float64)
    targets = np.asarray(d["phase_targets"], dtype=np.float64)
    mids = np.asarray(d["measurement_ids"], dtype=np.int64)

    pos = [i for i in range(len(mids)) if targets[i].sum() == 3]
    pos_i = max(pos, key=lambda i: float(probs[i].min()))
    neg = [i for i in range(len(mids)) if targets[i].sum() == 0]
    neg_i = min(neg, key=lambda i: float(probs[i].max()))
    pos_mid, neg_mid = int(mids[pos_i]), int(mids[neg_i])
    print(f"holdout positive example: mid={pos_mid}, probs={np.round(probs[pos_i], 4)}")
    print(f"holdout negative example: mid={neg_mid}, max prob={probs[neg_i].max():.4f}")

    cache = find_holdout_cache()
    files = {int(f.stem): f for f in cache.glob("*.npz")}
    prob_map = {int(m): p for m, p in zip(mids, probs)}

    records = []
    for mid in (pos_mid, neg_mid):
        with np.load(files[mid], allow_pickle=False) as data:
            windows = data["windows"].astype(np.float32)  # (3, 8, 8192)
            starts = data["starts"].astype(np.int64)
            kinds = data["kinds"].astype(np.uint8)
            tgt = data["targets"].astype(np.int8)
        weights = np.mean(
            [attention_weights(model, windows, device) for model in models],
            axis=0,
        )
        records.append(
            {
                "measurement_id": int(mid),
                "starts": starts,
                "kinds": kinds,
                "targets": tgt,
                "probs": prob_map[mid],
                "attn": weights,
            }
        )
        print(f"  example mid={mid}: probs={np.round(prob_map[mid], 4)}, targets={tgt.tolist()}")

    np.savez(
        OUT_DIR / "holdout_examples_attn_80k.npz",
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
    print("saved:", OUT_DIR / "holdout_examples_attn_80k.npz")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    make_holdout_probs()
    models = load_models(device)
    print(f"loaded {len(models)} fold models from {MODEL_DIR}")
    make_holdout_examples(models, device)


if __name__ == "__main__":
    main()
