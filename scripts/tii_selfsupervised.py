# -*- coding: utf-8 -*-
"""
TII-S1: Self-supervised (VICReg) pretraining + low-label fine-tuning
(TII track: label-cost narrative closed loop).

Question: does VICReg pretraining on unlabeled windows reduce the labeling
requirement of the weakly supervised pipeline? Compare, at low label
fractions (5% / 10% / 20%), the E4 mainline with and without VICReg
pretraining of the CNN encoder.

Design:
  - Pretrain: VICReg on ALL unlabeled windows of the training part
    (development only, 40 epochs, seed 42).
  - Fine-tune: the pretrained encoder is inserted into the E4 pipeline
    (simple_cnn + attention MIL + context-concat), then trained with only
    f% labels (per-measurement retention, same protocol as L1).
  - Contrast: L1 numbers without pretraining (5%: 0.272, 10%: 0.346,
    20%: 0.377±0.021) vs with pretraining.

Protocol: development-only; validation always keeps full labels;
blind/Harvard untouched.

Outputs (results/tii_selfsupervised/):
  vicreg_pretrain.pt          - pretrained CNN state dict
  f<f>_ls42/cv_summary.json   - fine-tuned results per label fraction
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
from vsb_pd.pretrain import pretrain_vicreg
from vsb_pd.training import compute_metrics, make_stratified_group_folds

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "cached_features" / "features_policy_mixed_k8.npz"
OUT = ROOT / "results" / "tii_selfsupervised"
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 15
LR = 1e-3
WD = 1e-4
GRAD_CLIP = 1.0
FOLD_SEED = 42
PRETRAIN_EPOCHS = 40
LABEL_FRACTIONS = (0.05, 0.10, 0.20)


def build_pipeline():
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


def train_fold(model, criterion, windows, feat, labels_masked, labels_full_np,
               tr, va, batch_size, epochs, patience, device, seed_fold):
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
            loss = criterion(phase_logits, labels_masked[bidx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        val_probs, val_targets = predict(model, windows, feat, va, batch_size, device, labels_full_np)
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


def make_label_mask(labels_np, fraction, label_seed, folds, fold_idx):
    rng = np.random.default_rng(label_seed * 1000 + fold_idx)
    tr_idx, _ = folds[fold_idx]
    mask = np.ones_like(labels_np, dtype=np.float32)
    keep_n = max(1, int(round(fraction * len(tr_idx))))
    keep_m = rng.choice(tr_idx, size=keep_n, replace=False)
    drop = np.setdiff1d(tr_idx, keep_m)
    mask[drop] = 0.0
    return mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    d = np.load(CACHE, allow_pickle=False)
    windows_np = np.asarray(d["windows"])
    labels_np = np.asarray(d["labels"])
    mids = np.asarray(d["measurement_ids"])
    M = len(mids)
    print(f"[data] {M} measurements")

    folds = make_stratified_group_folds(
        mids, np.clip(labels_np.sum(axis=1).astype(int), 0, 3),
        n_splits=N_FOLDS, seed=FOLD_SEED)

    # ---- VICReg pretrain on all unlabeled windows (window-level) ----
    ckpt_path = OUT / "vicreg_pretrain.pt"
    if ckpt_path.exists():
        print("[vicreg] loading pretrained state")
        pretrained_cnn = torch.load(ckpt_path, map_location=device)
    else:
        print("[vicreg] pretraining (40 epochs, all development windows)...")
        # pretrain_vicreg expects an encoder with .cnn (WindowEncoder-style).
        # Build a WindowEncoder-like CNN branch via SimpleCNN's inner cnn.
        from vsb_pd.dl_encoders import SimpleCNNEncoder
        enc = SimpleCNNEncoder(8192, 128).to(device)
        # windows_np: (M,3,K,8192) -> flatten to (M*3*K, 8192)
        w_flat = windows_np.reshape(-1, 8192)
        w_t = torch.from_numpy(w_flat).float()
        t0 = time.time()
        pretrain_vicreg(enc, w_t, epochs=PRETRAIN_EPOCHS, batch_size=BATCH_SIZE,
                        lr=1e-3, device=str(device))
        print(f"[vicreg] done in {time.time()-t0:.0f}s")
        pretrained_cnn = {k: v.cpu() for k, v in enc.cnn.state_dict().items()}
        torch.save(pretrained_cnn, ckpt_path)
        print("[vicreg] saved", ckpt_path)

    # ---- fine-tune per label fraction with pretrained encoder ----
    windows = torch.from_numpy(windows_np).float()
    feat = torch.zeros(M, 3, 8, 58, dtype=torch.float32)
    labels = torch.from_numpy(labels_np).float()
    summary = {"vicreg_epochs": PRETRAIN_EPOCHS, "runs": {}}
    for f in LABEL_FRACTIONS:
        cfg_dir = OUT / f"f{f:g}_ls{FOLD_SEED}"
        sp = cfg_dir / "cv_summary.json"
        if sp.exists():
            s = json.loads(sp.read_text(encoding="utf-8"))
            summary["runs"][str(f)] = s
            print(f"[f={f}] already done: {s['fold_mean_phase_pr_auc']}")
            continue
        cfg_dir.mkdir(parents=True, exist_ok=True)
        oof_p = np.zeros((M, 3)); oof_t = np.zeros((M, 3))
        oof_mp = np.zeros(M); oof_mt = np.zeros(M)
        fold_assign = np.zeros(M, dtype=np.int8)
        print(f"\n=== f={f} with VICReg pretrain ===")
        for fi, (tr_idx, va_idx) in enumerate(folds):
            mask = make_label_mask(labels_np, f, FOLD_SEED, folds, fi)
            labels_masked = torch.from_numpy(labels_np * mask).float()
            model = build_pipeline().to(device)
            # inject pretrained CNN weights
            cnn_state = model.encoder.inner.cnn.state_dict()
            cnn_state.update(pretrained_cnn)
            model.encoder.inner.cnn.load_state_dict(cnn_state)
            criterion = PhaseCyclicLoss(lambda_m=0.0)
            t0 = time.time()
            best_pr = train_fold(model, criterion, windows, feat, labels_masked,
                                 labels_np, tr_idx, va_idx, BATCH_SIZE, EPOCHS,
                                 PATIENCE, device, FOLD_SEED + fi)
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
        ph, me = [], []
        for fi in range(N_FOLDS):
            sel = fold_assign == fi
            fp, ft = oof_p[sel].reshape(-1), oof_t[sel].reshape(-1)
            mp, mt = oof_mp[sel], oof_mt[sel]
            ph.append(compute_metrics(ft, fp, (fp >= 0.5).astype(int))["pr_auc"])
            me.append(compute_metrics(mt, mp, (mp >= 0.5).astype(int))["pr_auc"])
        s = {
            "label_fraction": float(f),
            "with_vicreg": True,
            "fold_mean_phase_pr_auc": round(float(np.mean(ph)), 4),
            "fold_std_phase_pr_auc": round(float(np.std(ph)), 4),
            "fold_mean_meas_pr_auc": round(float(np.mean(me)), 4),
            "reference_without_vicreg": {
                0.05: 0.2718, 0.10: 0.3463, 0.20: 0.377,
            }[float(f)],
        }
        (cfg_dir / "cv_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
        summary["runs"][str(f)] = s
        print(f"  -> with VICReg: {s['fold_mean_phase_pr_auc']} "
              f"(without: {s['reference_without_vicreg']})")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== VICReg + LOW-LABEL SUMMARY ===")
    for f in LABEL_FRACTIONS:
        s = summary["runs"][str(f)]
        print(f"  f={f}: with={s['fold_mean_phase_pr_auc']} "
              f"without={s['reference_without_vicreg']}")
    print("\nDone. Output in", OUT)


if __name__ == "__main__":
    main()
