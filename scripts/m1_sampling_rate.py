# -*- coding: utf-8 -*-
"""
M1: Sampling-rate downsampling experiment (TII track, industrial sensing
cost-performance). For each target rate (40/20/10/5 MHz), the raw 40 MHz
windows are decimated by the integer factor 40/fs, the E4 mainline is
retrained and evaluated under the exact protocol, and the phase PR-AUC vs
data-rate (sensing hardware cost proxy) curve is produced.

Protocol: same development folds (StratifiedGroupKFold(5, seed=42)),
same training hyperparameters; only the input sampling rate changes.
Development-only; blind/Harvard never touched.

Outputs (results/tii_sampling_rate/):
  rate_<fs>MHz/cv_summary.json
  summary.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sklearn.metrics import matthews_corrcoef

from vsb_pd.cyclic import PhaseCyclicLoss, PhaseInteractionModule
from vsb_pd.dl_encoders import TimWindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.model import VSBPipeline
from vsb_pd.training import compute_metrics, make_stratified_group_folds

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "cached_features" / "features_policy_mixed_k8.npz"
OUT = ROOT / "results" / "tii_sampling_rate"
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 15
LR = 1e-3
WD = 1e-4
GRAD_CLIP = 1.0
SEED = 42
RATES_MHZ = [40, 20, 10, 5]      # 40 = native (reference mainline)
WINDOW_LEN = 8192


def build_pipeline(window_len: int = WINDOW_LEN):
    encoder = TimWindowEncoder("simple_cnn", window_len, 58, 128)
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


def train_fold(model, criterion, windows, feat, labels, labels_np, tr, va,
               batch_size, epochs, patience, device, seed_fold):
    torch.manual_seed(seed_fold)
    np.random.seed(seed_fold)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_pr, best_state, patience_counter = -1.0, None, 0
    train_idx = tr.tolist()
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_idx)
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(train_idx), batch_size):
            bidx = train_idx[i:i + batch_size]
            optimizer.zero_grad()
            phase_logits, _ = model(windows[bidx].to(device), feat[bidx].to(device))
            loss = criterion(phase_logits, labels[bidx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        val_probs, val_targets = predict(model, windows, feat, va, batch_size, device, labels_np)
        p, t = val_probs.flatten(), val_targets.flatten()
        m = compute_metrics(t, p, (p >= 0.5).astype(int))
        pr = m.get("pr_auc", float("nan"))
        if np.isfinite(pr) and pr > best_pr + 0.001:
            best_pr = pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_pr


def decimate(windows_np, fs_mhz):
    """Decimate 40 MHz windows to fs_mhz (integer divisor)."""
    factor = 40 // fs_mhz
    if factor == 1:
        return windows_np
    return windows_np[..., ::factor]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    d = np.load(CACHE, allow_pickle=False)
    windows_np = np.asarray(d["windows"])   # (M,3,K,8192) @40MHz
    labels_np = np.asarray(d["labels"])
    mids = np.asarray(d["measurement_ids"])
    print(f"[data] {len(mids)} measurements @40MHz")

    folds = make_stratified_group_folds(
        mids, np.clip(labels_np.sum(axis=1).astype(int), 0, 3),
        n_splits=N_FOLDS, seed=SEED)

    summary = {}
    for fs in RATES_MHZ:
        cfg_dir = OUT / f"rate_{fs}MHz"
        summary_path = cfg_dir / "cv_summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            summary[f"{fs}MHz"] = s
            print(f"[{fs}MHz] already done: {s['fold_mean_phase_pr_auc']}")
            continue

        cfg_dir.mkdir(parents=True, exist_ok=True)
        win = decimate(windows_np, fs)
        L = win.shape[-1]
        windows = torch.from_numpy(win).float()
        feat = torch.zeros(len(mids), 3, 8, 58, dtype=torch.float32)  # features unused by simple_cnn
        labels = torch.from_numpy(labels_np).float()
        M = len(mids)
        oof_p = np.zeros((M, 3))
        oof_t = np.zeros((M, 3))
        oof_mp = np.zeros(M)
        oof_mt = np.zeros(M)
        fold_assign = np.zeros(M, dtype=np.int8)
        print(f"\n=== {fs} MHz (window {L} pts) ===")
        for fi, (tr_idx, va_idx) in enumerate(folds):
            t0 = time.time()
            model = build_pipeline(L).to(device)
            criterion = PhaseCyclicLoss(lambda_m=0.0)
            best_pr = train_fold(model, criterion, windows, feat, labels, labels_np,
                                 tr_idx, va_idx, BATCH_SIZE, EPOCHS, PATIENCE,
                                 device, SEED + fi)
            probs, targets = predict(model, windows, feat, va_idx, BATCH_SIZE, device, labels_np)
            mp = 1.0 - np.prod(1.0 - probs, axis=1)
            mt = targets.max(axis=1)
            oof_p[va_idx], oof_t[va_idx] = probs, targets
            oof_mp[va_idx], oof_mt[va_idx] = mp, mt
            fold_assign[va_idx] = fi
            fp, ft = probs.flatten(), targets.flatten()
            fm = compute_metrics(ft, fp, (fp >= 0.5).astype(int))
            mm = compute_metrics(mt, mp, (mp >= 0.5).astype(int))
            print(f"  [fold {fi+1}] phase_pr={fm.get('pr_auc'):.4f} "
                  f"meas_pr={mm.get('pr_auc'):.4f} ({time.time()-t0:.0f}s)")
        # fold means
        ph, me = [], []
        for fi in range(N_FOLDS):
            sel = fold_assign == fi
            fp, ft = oof_p[sel].reshape(-1), oof_t[sel].reshape(-1)
            mp, mt = oof_mp[sel], oof_mt[sel]
            ph.append(compute_metrics(ft, fp, (fp >= 0.5).astype(int))["pr_auc"])
            me.append(compute_metrics(mt, mp, (mp >= 0.5).astype(int))["pr_auc"])
        s = {
            "sampling_rate_mhz": fs,
            "window_length": int(L),
            "fold_mean_phase_pr_auc": round(float(np.mean(ph)), 4),
            "fold_std_phase_pr_auc": round(float(np.std(ph)), 4),
            "fold_mean_meas_pr_auc": round(float(np.mean(me)), 4),
            "fold_std_meas_pr_auc": round(float(np.std(me)), 4),
            "data_reduction_vs_40mhz": 40.0 / fs,
        }
        (cfg_dir / "cv_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        summary[f"{fs}MHz"] = s
        print(f"  -> fold-mean phase PR-AUC {s['fold_mean_phase_pr_auc']}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SAMPLING-RATE COST-PERFORMANCE ===")
    print(f"{'rate':>6} {'phase PR-AUC':>14} {'meas PR-AUC':>14} {'data vs 40MHz':>14}")
    for fs in RATES_MHZ:
        s = summary[f"{fs}MHz"]
        print(f"{fs:>4}MHz {s['fold_mean_phase_pr_auc']:>14} "
              f"{s['fold_mean_meas_pr_auc']:>14} 1/{s['data_reduction_vs_40mhz']:.0f}")
    print("\nDone. Output in", OUT)


if __name__ == "__main__":
    main()
