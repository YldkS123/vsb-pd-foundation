# -*- coding: utf-8 -*-
"""Time-shift augmentation study for the final 80k mainline.

The locked 80k mainline (cnn + attention + mean) is trained exactly as in the
main report (K=8 mixed windows, batch=64, epochs=40, patience=15, seed 42)
except that each training batch applies a random zero-padded time shift,
sampled per measurement from a fixed shift grid, to the raw windows. The
physical-feature tensor is passed unchanged because the final mainline uses
the pure-CNN encoder branch (features are ignored); the augmentation thus
teaches shift invariance directly on the waveform.

The study is auxiliary: it runs on the same development folds and never
touches the blind holdout, whose one-shot receipt stays locked. Robustness
is reported with the same zero-pad fixed-window protocol and recomputed
physical features as results/robustness_report_80k.json.

Usage:
  python scripts/run_shift_augment_80k.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vsb_pd.encoder import WindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.cyclic import PhaseInteractionModule
from vsb_pd.model import VSBPipeline
from vsb_pd.training import make_stratified_group_folds, compute_metrics
from vsb_pd.features import extract_physical_features
from vsb_pd.perturb import time_shift

ROOT = Path(__file__).resolve().parent.parent
SR_HZ = 40_000_000


def build_model() -> VSBPipeline:
    return VSBPipeline(
        encoder=WindowEncoder(8192, 58, 128, branch="cnn"),
        aggregator=MILAggregator("attention", 128),
        cyclic=PhaseInteractionModule("mean", 128),
        classifier=PhaseClassifier(128),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_fold(model: nn.Module, windows: torch.Tensor, feat: torch.Tensor, indices, batch_size: int):
    model.eval()
    probs_list, targets_list = [], []
    for i in range(0, len(indices), batch_size):
        bidx = indices[i:i + batch_size]
        logits, _ = model(windows[bidx], feat[bidx])
        probs_list.append(torch.sigmoid(logits).cpu().numpy())
        targets_list.append(labels_np[bidx])
    return np.concatenate(probs_list), np.concatenate(targets_list)


def shift_batch(w: torch.Tensor, shifts: np.ndarray) -> torch.Tensor:
    """Zero-padded shift per measurement: w (B, P, K, L) -> shifted copy."""
    out = torch.zeros_like(w)
    for b in range(w.shape[0]):
        s = int(shifts[b])
        if s == 0:
            out[b] = w[b]
        elif s > 0:
            out[b, ..., s:] = w[b, ..., :-s]
        else:
            out[b, ..., :s] = w[b, ..., -s:]
    return out


def phase_metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    p = probs.flatten()
    t = targets.flatten()
    m = compute_metrics(t, p, (p >= 0.5).astype(int))
    return {"pr_auc": m.get("pr_auc"), "roc_auc": m.get("roc_auc"), "f1": m.get("f1")}


def measurement_metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    mp = 1.0 - np.prod(1.0 - probs, axis=1)
    mt = targets.max(axis=1)
    m = compute_metrics(mt, mp, (mp >= 0.5).astype(int))
    return {"pr_auc": m.get("pr_auc"), "roc_auc": m.get("roc_auc"), "f1": m.get("f1")}


def oof_evaluate(models, folds, windows, feat, batch_size: int):
    M = len(windows)
    oof_p = np.zeros((M, 3), dtype=np.float64)
    for fi, (tr, va) in enumerate(folds):
        p, _ = predict_fold(models[fi], windows, feat, va.tolist(), batch_size)
        oof_p[va] = p
    return oof_p


def train_and_save(
    windows: torch.Tensor, feat: torch.Tensor, labels: torch.Tensor, mids: np.ndarray,
    folds, epochs: int, batch_size: int, patience: int, seed: int,
    shift_grid: np.ndarray, out_dir: Path,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = len(mids)
    cfg_dir = out_dir
    cfg_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    oof_probs = np.zeros((M, 3), dtype=np.float64)
    oof_targets = np.zeros((M, 3), dtype=np.float64)
    fold_assign = np.zeros(M, dtype=np.int8)
    n_params = 0

    for fi, (tr, va) in enumerate(folds):
        t0 = time.time()
        torch.manual_seed(seed + fi)
        np.random.seed(seed + fi)

        model = build_model().to(device)
        n_params = count_params(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        best_pr_auc = -1.0
        best_state = None
        patience_counter = 0
        epochs_trained = 0
        train_idx = tr.tolist()

        for epoch in range(epochs):
            model.train()
            np.random.shuffle(train_idx)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(train_idx), batch_size):
                bidx = train_idx[i:i + batch_size]
                wb = windows[bidx]
                shifts = np.random.choice(shift_grid, size=len(bidx))
                wb_aug = shift_batch(wb, shifts)
                optimizer.zero_grad()
                phase_logits, _ = model(wb_aug, feat[bidx])
                loss = criterion(phase_logits, labels[bidx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1

            val_probs, val_targets = predict_fold(model, windows, feat, va.tolist(), batch_size)
            p_flat = val_probs.flatten()
            t_flat = val_targets.flatten()
            m = compute_metrics(t_flat, p_flat, (p_flat >= 0.5).astype(int))
            pr_auc = m.get("pr_auc", float("nan"))
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={total_loss / max(n_batches, 1):.4f} "
                      f"pr_auc={pr_auc:.4f} f1={m.get('f1', 0):.4f}", flush=True)
            epochs_trained = epoch + 1
            if np.isfinite(pr_auc) and pr_auc > best_pr_auc + 0.001:
                best_pr_auc = pr_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch}", flush=True)
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        final_probs, final_targets = predict_fold(model, windows, feat, va.tolist(), batch_size)
        oof_probs[va] = final_probs
        oof_targets[va] = final_targets
        fold_assign[va] = fi

        fp = final_probs.flatten()
        ft = final_targets.flatten()
        fm = compute_metrics(ft, fp, (fp >= 0.5).astype(int))
        mm = measurement_metrics(final_probs, final_targets)
        elapsed = time.time() - t0
        fold_metrics.append({
            "fold": fi + 1,
            "epochs_trained": epochs_trained,
            "elapsed_s": round(elapsed, 1),
            "phase": {k: round(float(v), 4) for k, v in fm.items()},
            "measurement": {k: round(float(v), 4) for k, v in mm.items()},
        })
        print(f"  Fold {fi+1}: phase_pr_auc={fm.get('pr_auc', float('nan')):.4f} "
              f"meas_pr_auc={mm.get('pr_auc', float('nan')):.4f} ({elapsed:.0f}s)", flush=True)
        torch.save({"state_dict": best_state, "config": "shift_aug_80k"},
                   cfg_dir / f"model_fold{fi + 1}.pt")

    np.savez_compressed(
        cfg_dir / "oof.npz",
        phase_probs=oof_probs, phase_targets=oof_targets,
        measurement_ids=mids, fold_assign=fold_assign,
    )

    prs = [f["phase"]["pr_auc"] for f in fold_metrics]
    rocs = [f["phase"]["roc_auc"] for f in fold_metrics]
    mprs = [f["measurement"]["pr_auc"] for f in fold_metrics]
    summary = {
        "config": {"encoder": "cnn", "mil": "attention", "phase": "mean", "augmentation": "random_time_shift"},
        "n_measurements": int(M),
        "n_folds": len(fold_metrics),
        "n_params": int(n_params),
        "shift_grid": [int(s) for s in shift_grid],
        "mean_phase_pr_auc": float(np.mean(prs)),
        "std_phase_pr_auc": float(np.std(prs)),
        "mean_phase_roc_auc": float(np.mean(rocs)),
        "mean_measurement_pr_auc": float(np.mean(mprs)),
        "folds": fold_metrics,
    }
    (cfg_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def evaluate_shift_robustness(ckpt_dir: Path, windows_np, feat_names, folds, device, batch_size: int, seed: int):
    d = np.load(ROOT / "results" / "cached_features" / "features_policy_mixed_k8.npz", allow_pickle=True)
    labels_np_local = np.asarray(d["labels"])
    mids = np.asarray(d["measurement_ids"])
    windows = torch.from_numpy(windows_np).float().to(device)
    feat = torch.from_numpy(np.asarray(d["feat_array"])).float().to(device)

    models = []
    for fi in range(5):
        model = build_model().to(device)
        ckpt = torch.load(ckpt_dir / f"model_fold{fi + 1}.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        models.append(model)

    base_p = oof_evaluate(models, folds, windows, feat, batch_size)
    report = {
        "model": "80k cnn+attention+mean + random time-shift augmentation",
        "baseline_phase": phase_metrics(base_p, labels_np_local),
        "baseline_measurement": measurement_metrics(base_p, labels_np_local),
        "shift": {},
        "shift_metadata": {
            "method": "zero_pad_fixed_window",
            "features": "recomputed_from_shifted_windows",
        },
    }

    for shift in [-128, -64, 64, 128]:
        w_shifted = time_shift(windows_np, shift)
        feats_shifted = extract_physical_features(w_shifted, SR_HZ, batch_size=4096)
        feat_arr_shifted = np.stack([feats_shifted[n] for n in feat_names], axis=-1).reshape(
            windows_np.shape[0], 3, 8, -1)
        w_t = torch.from_numpy(w_shifted).float().to(device)
        f_t = torch.from_numpy(feat_arr_shifted).float().to(device)
        p = oof_evaluate(models, folds, w_t, f_t, batch_size)
        report["shift"][f"{shift:+d}"] = {
            "phase": phase_metrics(p, labels_np_local),
            "measurement": measurement_metrics(p, labels_np_local),
        }
        print(f"Shift {shift:+d}: phase PR-AUC={report['shift'][f'{shift:+d}']['phase']['pr_auc']:.4f}", flush=True)

    base_phase = report["baseline_phase"]["pr_auc"]
    base_meas = report["baseline_measurement"]["pr_auc"]
    for key, val in report["shift"].items():
        val["phase_pr_auc_drop_pct"] = round(float((base_phase - val["phase"]["pr_auc"]) / base_phase * 100), 2)
        val["measurement_pr_auc_drop_pct"] = round(float((base_meas - val["measurement"]["pr_auc"]) / base_meas * 100), 2)

    # Mainline reference for direct comparison.
    ref = json.loads((ROOT / "results" / "robustness_report_80k.json").read_text(encoding="utf-8"))
    report["mainline_reference_drop_pct"] = {
        key: ref["shift"][key]["phase_pr_auc_drop_pct"] for key in report["shift"]
    }
    return report


def main() -> None:
    global labels_np
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cached_features/features_policy_mixed_k8.npz")
    ap.add_argument("--out-dir", default="results/shift_aug_80k")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shifts", default="-128,-64,-32,0,32,64,128")
    args = ap.parse_args()

    d = np.load(ROOT / args.cache, allow_pickle=True)
    windows_np = np.asarray(d["windows"])
    feat_array = np.asarray(d["feat_array"])
    labels_np = np.asarray(d["labels"])
    mids = np.asarray(d["measurement_ids"])
    feat_names = list(d["feature_names"])
    shift_grid = np.asarray([int(s) for s in args.shifts.split(",")])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = torch.from_numpy(windows_np).float().to(device)
    feat = torch.from_numpy(feat_array).float().to(device)
    labels = torch.from_numpy(labels_np).float().to(device)

    label_counts = labels_np.sum(axis=1).astype(int)
    folds = make_stratified_group_folds(mids, label_counts, n_splits=5, seed=args.seed)
    print(f"Data: {len(mids)} measurements; folds: {[(len(tr), len(va)) for tr, va in folds]}", flush=True)

    out_dir = ROOT / args.out_dir
    summary = train_and_save(
        windows, feat, labels, mids, folds,
        args.epochs, args.batch_size, args.patience, args.seed,
        shift_grid, out_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    robust = evaluate_shift_robustness(
        out_dir, windows_np, feat_names, folds, device, args.batch_size, args.seed)
    (out_dir / "shift_robustness.json").write_text(
        json.dumps(robust, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'shift_robustness.json'}")


if __name__ == "__main__":
    main()
