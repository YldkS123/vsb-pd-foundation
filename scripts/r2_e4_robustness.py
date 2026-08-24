# -*- coding: utf-8 -*-
"""
R2: E4 perturbation robustness rerun on the IR-trained checkpoints
(TIM submission, P1). Mirrors the historical 80k robustness protocol
(scripts/run_robustness_80k.py) but for the E4 mainline architecture
(simple_cnn encoder + attention MIL + context-concat interaction) using the
checkpoints produced by scripts/e2_ir_evaluation.py (trained on the 80%
training domain under the exact mainline protocol).

This closes the known gap that perturbation evidence came only from the
historical locked mainline (paper Section V-F / limitations): the E4
architecture's perturbation sensitivity is now measured directly.

Perturbations (inference-time only, no retraining):
  1. additive Gaussian noise SNR 20/10/5 dB
  2. amplitude scaling 0.8x / 1.2x
  3. time shift ±64 / ±128 (zero-padded, features recomputed)
  4. missing phase (noisy-OR, measurement level)

Protocol safety: development set only; the 423 blind set and Harvard receipt
are never touched. Perturbation is applied to window signals at inference;
the per-window physical features are recomputed from the shifted windows for
time shifts (matching the historical protocol).

Outputs (results/ir_eval/robustness_E4.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vsb_pd.cyclic import PhaseInteractionModule
from vsb_pd.dl_encoders import TimWindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.model import VSBPipeline
from vsb_pd.features import extract_physical_features

OUT = ROOT / "results" / "ir_eval"
CACHE = ROOT / "results" / "cached_features" / "features_full.npz"
IR_DIR = ROOT / "results" / "ir_eval"


def build_model():
    encoder = TimWindowEncoder("simple_cnn", 8192, 58, 128)
    aggregator = MILAggregator("attention", 128)
    cyclic = PhaseInteractionModule("context_concat", 128)
    classifier = PhaseClassifier(128)
    return VSBPipeline(
        encoder=encoder, aggregator=aggregator, cyclic=cyclic,
        classifier=classifier, max_encode_chunk=8, checkpoint_chunks=True,
    )


@torch.no_grad()
def predict(model, windows, feat, indices, batch_size, device, labels_np):
    model.eval()
    probs_list, targets_list = [], []
    for i in range(0, len(indices), batch_size):
        bidx = indices[i:i + batch_size]
        logits, _ = model(windows[bidx].to(device), feat[bidx].to(device))
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
        targets_list.append(labels_np[bidx])
    return np.concatenate(probs_list), np.concatenate(targets_list)


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


def add_gaussian(w, snr_db):
    """w: (B,K,L) float; returns noisy copy with given SNR (dB)."""
    x = w.clone()
    sig = x.std(dim=-1, keepdim=True) + 1e-12
    noise = torch.randn_like(x) * (sig / (10 ** (snr_db / 20.0)))
    return x + noise


def scale_amp(w, factor):
    return w.clone() * factor


def shift_zero_pad(w, shift):
    """Roll with zero padding (content exits the window edge), matching the
    historical protocol (features recomputed on the shifted window)."""
    x = w.clone()
    if shift == 0:
        return x
    if shift > 0:
        x = torch.cat([torch.zeros_like(x[..., :shift]), x[..., :-shift]], dim=-1)
    else:
        s = -shift
        x = torch.cat([x[..., s:], torch.zeros_like(x[..., :s])], dim=-1)
    return x


def recompute_features(w_np, feature_names):
    """Recompute the 58-d physical features for shifted windows.
    w_np: (B,P,K,L) numpy (batch, phases, windows, length) or (P,K,L);
    feature_names: canonical 58-name list.
    Returns (B,P,K,58) or (P,K,58) float32 in the canonical order."""
    if w_np.ndim == 4:
        B, P, K, L = w_np.shape
        out = np.zeros((B, P, K, 58), dtype=np.float32)
        for b0 in range(0, B, 8):
            chunk = w_np[b0:b0 + 8]
            Bc = chunk.shape[0]
            try:
                # (Bc,P,K,L) -> (Bc*P,K,L) via extract (accepts 4D)
                d = extract_physical_features(chunk, 40_000_000)
            except Exception:
                continue
            arr = np.stack([d[n] for n in feature_names], axis=-1)  # (Bc*P*K, 58)
            out[b0:b0 + Bc] = arr.reshape(Bc, P, K, -1)[:, :, :, :58]
        return out
    else:
        P, K, L = w_np.shape
        d = extract_physical_features(w_np, 40_000_000)
        arr = np.stack([d[n] for n in feature_names], axis=-1)  # (P*K, 58)
        return arr.reshape(P, K, -1)[:, :, :58]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load the IR split so we can evaluate on the TRAINING DOMAIN OOF
    # (the checkpoints were trained on the 80% domain; perturbation OOF
    # evaluation must use the same domain folds, not the IR set).
    d = np.load(CACHE, allow_pickle=False)
    windows_np = np.asarray(d["windows"])
    feat_np = np.asarray(d["feat_array"])
    labels_np = np.asarray(d["labels"])
    mids = np.asarray(d["measurement_ids"])
    feature_names = [str(n) for n in d["feature_names"]]

    lock = json.loads((IR_DIR / "ir_split_lock.json").read_text(encoding="utf-8"))
    ir_ids = set(lock["ir_measurement_ids"])
    tr_mask = np.array([int(m) not in ir_ids for m in mids])
    tr_idx = np.where(tr_mask)[0]
    print(f"[data] training-domain measurements: {len(tr_idx)}")

    # The E2 checkpoints are 5 folds over the training domain; reconstruct the
    # inner folds exactly as E2 did (StratifiedGroupKFold(5, seed=42) on the
    # training-domain measurements, grouped by measurement id).
    from sklearn.model_selection import StratifiedGroupKFold
    y_tr = np.clip(labels_np[tr_mask].sum(axis=1).astype(int), 0, 3)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(inner.split(np.zeros(len(tr_idx)), y_tr, groups=mids[tr_mask]))

    models = []
    for fi in range(5):
        m = build_model().to(device)
        ckpt = torch.load(IR_DIR / f"fold_ckpt_{fi + 1}.pt", map_location=device)
        m.load_state_dict(ckpt)
        models.append(m)
    print("[models] loaded 5 E4 (IR-trained) checkpoints")

    windows = torch.from_numpy(windows_np).float()
    feat = torch.from_numpy(feat_np).float()

    # ---- baseline OOF on training domain ----
    def oof_with_perturb(perturb_fn=None, feats_override=None):
        M = len(tr_idx)
        oof_p = np.zeros((M, 3))
        for fi, (tr_i, va_i) in enumerate(folds):
            va_g = tr_idx[va_i]
            w = windows[va_g]
            f = feat[va_g]
            if perturb_fn is not None:
                w = perturb_fn(w)
            if feats_override is not None:
                # feats_override is indexed by training-domain rows; va_i are
                # inner-fold indices into the training domain
                f = torch.from_numpy(feats_override[va_i]).float()
            p, _ = predict(models[fi], w, f, np.arange(len(va_g)), 64, device, labels_np[va_g])
            oof_p[va_i] = p
        return oof_p

    base_p = oof_with_perturb()
    y_flat = labels_np[tr_idx].reshape(-1)
    base_pr = pr_auc(y_flat, base_p.reshape(-1))
    mp = 1.0 - np.prod(1.0 - base_p, axis=1)
    mt = labels_np[tr_idx].max(axis=1)
    base_meas_pr = pr_auc(mt, mp)
    print(f"[baseline] phase PR-AUC {base_pr:.4f}, meas PR-AUC {base_meas_pr:.4f}")

    report = {"model": "E4 (simple_cnn + attention MIL + context-concat), "
                       "IR-trained checkpoints (80% domain)",
              "baseline_phase_pr_auc": round(base_pr, 4),
              "baseline_meas_pr_auc": round(base_meas_pr, 4),
              "perturbations": {}}

    # ---- Gaussian noise ----
    for snr in [20, 10, 5]:
        p = oof_with_perturb(lambda w, s=snr: add_gaussian(w, s))
        pr = pr_auc(y_flat, p.reshape(-1))
        report["perturbations"][f"gaussian_{snr}dB"] = {
            "phase_pr_auc": round(pr, 4),
            "phase_relative_drop": round(float((pr - base_pr) / base_pr), 4),
        }
        print(f"  gaussian {snr}dB: {pr:.4f} ({(pr - base_pr) / base_pr:+.1%})")

    # ---- amplitude scaling ----
    for factor in [0.8, 1.2]:
        p = oof_with_perturb(lambda w, f=factor: scale_amp(w, f))
        pr = pr_auc(y_flat, p.reshape(-1))
        report["perturbations"][f"amp_{factor}x"] = {
            "phase_pr_auc": round(pr, 4),
            "phase_relative_drop": round(float((pr - base_pr) / base_pr), 4),
        }
        print(f"  amp {factor}x: {pr:.4f} ({(pr - base_pr) / base_pr:+.1%})")

    # ---- time shift (±64/±128) with feature recomputation ----
    for shift in [-128, -64, 64, 128]:
        w_shift = shift_zero_pad(windows[tr_idx], shift).numpy()
        f_re = recompute_features(w_shift.astype(np.float64), feature_names)  # (n_tr,P,K,58)
        # perturb the windows AND use the recomputed features
        p = oof_with_perturb(
            perturb_fn=lambda w, s=shift: shift_zero_pad(w, s),
            feats_override=f_re,
        )
        pr = pr_auc(y_flat, p.reshape(-1))
        report["perturbations"][f"shift_{shift:+d}"] = {
            "phase_pr_auc": round(pr, 4),
            "phase_relative_drop": round(float((pr - base_pr) / base_pr), 4),
        }
        print(f"  shift {shift:+d}: {pr:.4f} ({(pr - base_pr) / base_pr:+.1%})")

    # ---- missing phase (measurement-level noisy-OR) ----
    for miss in [0, 1, 2]:
        mask = torch.ones(3, dtype=torch.bool)
        mask[miss] = False
        # recompute noisy-OR from base probabilities with one phase zeroed
        p_zero = base_p.copy()
        p_zero[:, miss] = 0.0
        mp_miss = 1.0 - np.prod(1.0 - p_zero, axis=1)
        pr_miss = pr_auc(mt, mp_miss)
        report["perturbations"][f"missing_phase_{miss}"] = {
            "meas_pr_auc": round(pr_miss, 4),
            "meas_relative_drop": round(float((pr_miss - base_meas_pr) / base_meas_pr), 4),
        }
        print(f"  missing phase {miss}: {pr_miss:.4f} "
              f"({(pr_miss - base_meas_pr) / base_meas_pr:+.1%})")

    (OUT / "robustness_E4.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nDone. Output in", OUT / "robustness_E4.json")


if __name__ == "__main__":
    main()
