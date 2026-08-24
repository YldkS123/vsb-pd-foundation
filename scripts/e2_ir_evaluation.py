# -*- coding: utf-8 -*-
"""
E2-B: Internal-Reserved (IR) strict hold-out evaluation for the E4 mainline
(TIM submission, P0). Implements the t3 design decision D2 (方案 B):

  - From the 2,481 development measurements, stratify a 20% IR set
    (~496 measurements, grouped by measurement_id, stratified by number of
    positive phases) under a NEW hash-locked split (ir_split_lock.json).
  - Retrain the E4 mainline (pure CNN + attention MIL + context-concat,
    113,265 params) on the remaining 80% with the EXACT same protocol
    (StratifiedGroupKFold(5, seed=42) inside the 80% training domain,
    AdamW, batch 64, epochs 40, patience 15, gradient clip 1.0).
  - Thresholds: phase/measurement max-MCC selected ONLY on the 80%
    training-domain OOF. The IR set is evaluated EXACTLY ONCE with the
    frozen thresholds and frozen checkpoints.
  - Honest positioning (t3 D4): the IR set is same-distribution and
    participated in no NEW selection; the 20% split is drawn from the
    development set AFTER all original hyperparameter/lambda/threshold
    choices were locked, so the IR evaluation is a "frozen-protocol
    training-isolation verification", not an external validation.

Protocol safety: the 423 blind set and the Harvard receipt are NEVER
touched. No locked number from the main report is changed.

Outputs (results/ir_eval/):
  ir_split_lock.json   - SHA-256 locked IR split (measurement ids)
  fold_<i>.json        - per-fold training summary on the 80% domain
  ir_eval.json         - one-time IR evaluation summary + receipt hash
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold

from vsb_pd.cyclic import PhaseCyclicLoss, PhaseInteractionModule
from vsb_pd.dl_encoders import TimWindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.model import VSBPipeline
from vsb_pd.training import compute_metrics

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "cached_features" / "features_full.npz"
OUT = ROOT / "results" / "ir_eval"
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 15
LR = 1e-3
WD = 1e-4
GRAD_CLIP = 1.0
SEED = 42
IR_FRAC = 0.20


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


def train_fold(model, criterion, windows, feat, labels, labels_np, tr, va,
               batch_size, epochs, patience, device, seed_fold):
    torch.manual_seed(seed_fold)
    np.random.seed(seed_fold)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_pr, best_state, patience_counter = -1.0, None, 0
    train_idx = tr.tolist()
    epochs_trained = 0
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
        epochs_trained = epoch + 1
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
    return epochs_trained, best_pr


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


def roc_auc(y, p):
    order = np.argsort(-p, kind="stable")
    yp = y[order]
    tp = np.cumsum(yp)
    fp = np.cumsum(1 - yp)
    n_pos, n_neg = yp.sum(), (1 - yp).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float(np.trapz(tp / n_pos, fp / n_neg))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    d = np.load(CACHE, allow_pickle=False)
    windows = torch.from_numpy(np.asarray(d["windows"])).float()
    feat = torch.from_numpy(np.asarray(d["feat_array"])).float()
    labels = torch.from_numpy(np.asarray(d["labels"], dtype=np.float32)).float()
    labels_np = np.asarray(d["labels"])
    mids = np.asarray(d["measurement_ids"])
    M = len(mids)
    print(f"[data] {M} measurements")

    # ---- IR split: 20% stratified by positive-phase count, grouped by measurement ----
    y_meas = np.clip(labels_np.sum(axis=1).astype(int), 0, 3)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    # split returns (train_indices, test_indices); take the FIRST fold's test
    # (~20%) as the IR set; the corresponding train (~80%) is the training domain
    for tr_idx, ir_idx in sgkf.split(np.zeros(M), y_meas, groups=mids):
        break
    ir_mask = np.zeros(M, dtype=bool)
    ir_mask[ir_idx] = True
    tr_mask = ~ir_mask
    n_ir, n_tr = int(ir_mask.sum()), int(tr_mask.sum())
    print(f"[split] IR {n_ir} ({n_ir / M:.1%}), training domain {n_tr} ({n_tr / M:.1%})")
    ir_pos = int(labels_np[ir_mask].sum())
    print(f"[split] IR positive phases: {ir_pos} ({ir_pos / (n_ir * 3):.2%})")

    # ---- lock the split (SHA-256) ----
    lock_payload = {
        "protocol": "E2-B internal-reserved split (20%), grouped by measurement, "
                    "stratified by positive-phase count, seed 42; drawn AFTER all "
                    "original selection choices were locked",
        "n_total": int(M),
        "ir_measurement_ids": mids[ir_mask].tolist(),
        "train_measurement_ids": mids[tr_mask].tolist(),
        "ir_n_positive_phases": int(ir_pos),
    }
    payload_json = json.dumps(lock_payload, sort_keys=True).encode("utf-8")
    lock_sha = hashlib.sha256(payload_json).hexdigest()
    lock_payload["sha256"] = lock_sha
    (OUT / "ir_split_lock.json").write_text(
        json.dumps(lock_payload, indent=2), encoding="utf-8")
    print("[lock] sha256:", lock_sha)

    # ---- folds inside the 80% training domain (same protocol as mainline) ----
    tr_mids = mids[tr_mask]
    tr_y = np.clip(labels_np[tr_mask].sum(axis=1).astype(int), 0, 3)
    inner_folds = list(StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    ).split(np.zeros(n_tr), tr_y, groups=tr_mids))

    # position maps: measurement index -> row in training domain
    tr_pos = np.where(tr_mask)[0]           # global measurement indices in domain
    ir_pos_global = np.where(ir_mask)[0]

    oof_phase_p = np.zeros((n_tr, 3))
    oof_phase_t = np.zeros((n_tr, 3))
    oof_meas_p = np.zeros(n_tr)
    oof_meas_t = np.zeros(n_tr)
    fold_assign = np.zeros(n_tr, dtype=np.int8)

    for fi, (tr_i, va_i) in enumerate(inner_folds):
        t0 = time.time()
        model = build_pipeline().to(device)
        criterion = PhaseCyclicLoss(lambda_m=0.0)
        g_tr = tr_pos[tr_i]
        g_va = tr_pos[va_i]
        epochs_trained, best_pr = train_fold(
            model, criterion, windows, feat, labels, labels_np, g_tr, g_va,
            BATCH_SIZE, EPOCHS, PATIENCE, device, SEED + fi,
        )
        # save checkpoint for the one-time IR ensemble
        torch.save(model.state_dict(), OUT / f"fold_ckpt_{fi + 1}.pt")
        final_probs, final_targets = predict(model, windows, feat, g_va, BATCH_SIZE, device, labels_np)
        meas_probs = 1.0 - np.prod(1.0 - final_probs, axis=1)
        meas_targets = final_targets.max(axis=1)
        oof_phase_p[va_i] = final_probs
        oof_phase_t[va_i] = final_targets
        oof_meas_p[va_i] = meas_probs
        oof_meas_t[va_i] = meas_targets
        fold_assign[va_i] = fi

        fp, ft = final_probs.flatten(), final_targets.flatten()
        fm = compute_metrics(ft, fp, (fp >= 0.5).astype(int))
        mm = compute_metrics(meas_targets, meas_probs, (meas_probs >= 0.5).astype(int))
        print(f"  [fold {fi + 1}] phase_pr={fm.get('pr_auc', float('nan')):.4f} "
              f"meas_pr={mm.get('pr_auc', float('nan')):.4f} ({time.time() - t0:.0f}s)")
        (OUT / f"fold_{fi + 1}.json").write_text(json.dumps({
            "fold": fi + 1, "epochs_trained": int(epochs_trained),
            "best_val_phase_pr_auc": round(float(best_pr), 4),
            "phase_pr_auc": round(float(fm.get("pr_auc", float("nan"))), 4),
            "meas_pr_auc": round(float(mm.get("pr_auc", float("nan"))), 4),
        }, indent=2), encoding="utf-8")

    # ---- thresholds from 80% domain OOF only ----
    phase_flat_p = oof_phase_p.reshape(-1)
    phase_flat_t = oof_phase_t.reshape(-1)
    best_pt, best_pt_mcc = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        mcc = matthews_corrcoef(phase_flat_t, (phase_flat_p >= t).astype(int))
        if mcc > best_pt_mcc:
            best_pt, best_pt_mcc = t, mcc
    best_mt, best_mt_mcc = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        mcc = matthews_corrcoef(oof_meas_t, (oof_meas_p >= t).astype(int))
        if mcc > best_mt_mcc:
            best_mt, best_mt_mcc = t, mcc
    print(f"[thresholds] phase {best_pt:.2f} (mcc {best_pt_mcc:.3f}), "
          f"meas {best_mt:.2f} (mcc {best_mt_mcc:.3f})")

    # ---- ONE-TIME IR evaluation: 5-fold ensemble on frozen checkpoints ----
    print("[IR] evaluating IR set with 5-fold ensemble...")
    ir_probs_acc = np.zeros((n_ir, 3))
    for fi in range(N_FOLDS):
        ckpt_path = OUT / f"fold_ckpt_{fi + 1}.pt"
        model = build_pipeline().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        ir_probs, _ = predict(model, windows, feat, ir_pos_global, BATCH_SIZE, device, labels_np)
        ir_probs_acc += ir_probs / N_FOLDS
        print(f"  [fold {fi + 1}] IR inference done")

    # ---- IR metrics (frozen thresholds) ----
    ir_y = labels_np[ir_pos_global]
    ir_meas_y = ir_y.max(axis=1)
    ir_meas_p = 1.0 - np.prod(1.0 - ir_probs_acc, axis=1)

    ir_phase_pr = pr_auc(ir_y.reshape(-1), ir_probs_acc.reshape(-1))
    ir_phase_roc = roc_auc(ir_y.reshape(-1), ir_probs_acc.reshape(-1))
    ir_meas_pr = pr_auc(ir_meas_y, ir_meas_p)
    ir_meas_roc = roc_auc(ir_meas_y, ir_meas_p)
    ir_phase_mcc = matthews_corrcoef(ir_y.reshape(-1),
                                     (ir_probs_acc.reshape(-1) >= best_pt).astype(int))
    ir_meas_mcc = matthews_corrcoef(ir_meas_y, (ir_meas_p >= best_mt).astype(int))

    summary = {
        "protocol": "E2-B IR one-time evaluation (frozen thresholds from 80% domain OOF; "
                    "5-fold ensemble on the 80% domain; IR evaluated exactly once)",
        "ir_n_measurements": int(n_ir),
        "ir_n_positive_phases": int(ir_pos),
        "ir_positive_rate": round(float(ir_pos / (n_ir * 3)), 4),
        "phase_pr_auc": round(ir_phase_pr, 4),
        "phase_roc_auc": round(ir_phase_roc, 4),
        "phase_mcc": round(float(ir_phase_mcc), 4),
        "meas_pr_auc": round(ir_meas_pr, 4),
        "meas_roc_auc": round(ir_meas_roc, 4),
        "meas_mcc": round(float(ir_meas_mcc), 4),
        "phase_threshold": round(float(best_pt), 4),
        "meas_threshold": round(float(best_mt), 4),
        "split_lock_sha256": lock_sha,
        "reference": {
            "mainline_dev_phase_pr_auc_fold_mean": 0.615,
            "mainline_dev_phase_pr_auc_pooled": 0.533,
            "honest_note": "IR set is same-distribution internal validation drawn "
                           "from the development set after all original choices were "
                           "locked; not an external validation.",
        },
    }
    (OUT / "ir_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "ir_oof.npz",
                        ir_phase_probs=ir_probs_acc, ir_phase_targets=ir_y,
                        ir_meas_probs=ir_meas_p, ir_meas_targets=ir_meas_y,
                        ir_measurement_ids=mids[ir_pos_global])

    print("\n=== IR EVALUATION (E4 mainline, one-time, frozen thresholds) ===")
    for k, v in summary.items():
        if k != "reference":
            print(f"  {k}: {v}")
    print("  reference:", json.dumps(summary["reference"], ensure_ascii=False))
    print("\nDone. Outputs in", OUT)


if __name__ == "__main__":
    main()
