"""Robustness evaluation of the locked reference model (dual/gated/cyclic).

Uses the 5 trained fold checkpoints (results/model_full/model_fold{1..5}.pt) on the
same StratifiedGroupKFold(5, seed=42) development splits. Perturbations are applied
only at inference time; models and thresholds are never retrained on perturbed data.

Tests:
  1. additive Gaussian noise (SNR 20/10/5 dB)
  2. amplitude scaling (0.8x / 1.2x)
  3. time shift (+-64 / +-128 samples)
     zero-padded fixed-window shift, features recomputed from shifted windows)
  4. missing phase (mask one phase, measurement-level noisy-OR)
  5. parameter count and end-to-end inference latency
  6. measurement-level stratified bootstrap 95% CI for PR-AUC

Usage:
  python scripts/run_robustness.py [--cache results/cached_features/features_full.npz]
                                   [--ckpts results/model_full]
                                   [--seed 42]
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
from vsb_pd.cyclic import CyclicPhaseModule
from vsb_pd.model import VSBPipeline
from vsb_pd.training import make_stratified_group_folds
from vsb_pd.evaluation import compute_calibration_metrics
from vsb_pd.features import extract_physical_features
from vsb_pd.perturb import time_shift


def build_model() -> VSBPipeline:
    return VSBPipeline(
        encoder=WindowEncoder(8192, 58, 128, branch="dual"),
        aggregator=MILAggregator("gated_attention", 128),
        cyclic=CyclicPhaseModule(128),
        classifier=PhaseClassifier(128),
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict(model: nn.Module, windows: torch.Tensor, feat: torch.Tensor, indices, batch_size: int, phase_mask=None, perturb=None):
    model.eval()
    probs_list, meas_list, targets_list = [], [], []
    for i in range(0, len(indices), batch_size):
        bidx = indices[i:i + batch_size]
        w = windows[bidx]
        if perturb is not None:
            w = perturb(w.clone())
        logits, _ = model(w, feat[bidx], phase_mask=phase_mask[bidx] if phase_mask is not None else None)
        p = torch.sigmoid(logits)
        probs_list.append(p.cpu().numpy())
        targets_list.append(labels_np[bidx])
    probs = np.concatenate(probs_list)
    targets = np.concatenate(targets_list)
    return probs, targets


def phase_metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    from vsb_pd.training import compute_metrics
    p = probs.flatten()
    t = targets.flatten()
    m = compute_metrics(t, p, (p >= 0.5).astype(int))
    return {"pr_auc": m.get("pr_auc"), "roc_auc": m.get("roc_auc"), "f1": m.get("f1")}


def measurement_metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    from vsb_pd.training import compute_metrics
    mp = 1.0 - np.prod(1.0 - probs, axis=1)
    mt = targets.max(axis=1)
    m = compute_metrics(mt, mp, (mp >= 0.5).astype(int))
    return {"pr_auc": m.get("pr_auc"), "roc_auc": m.get("roc_auc"), "f1": m.get("f1")}


def oof_evaluate(models, folds, windows, feat, batch_size, perturb=None, phase_mask=None):
    """Concatenate per-fold validation predictions into OOF probabilities."""
    M = len(windows)
    oof_p = np.zeros((M, 3), dtype=np.float64)
    for fi, (tr, va) in enumerate(folds):
        p, _ = predict(models[fi], windows, feat, va.tolist(), batch_size, phase_mask=phase_mask, perturb=perturb)
        oof_p[va] = p
    return oof_p


def measurement_bootstrap_ci(probs: np.ndarray, targets: np.ndarray, mids: np.ndarray, n_boot: int = 2000, seed: int = 42):
    """Stratified cluster bootstrap over measurements for measurement-level PR-AUC."""
    from sklearn.metrics import average_precision_score
    mp = 1.0 - np.prod(1.0 - probs, axis=1)
    mt = targets.max(axis=1)
    rng = np.random.default_rng(seed)
    pos_ids = np.unique(mids[mt == 1])
    neg_ids = np.unique(mids[mt == 0])
    vals = []
    for _ in range(n_boot):
        ps = rng.choice(pos_ids, size=len(pos_ids), replace=True)
        ns = rng.choice(neg_ids, size=len(neg_ids), replace=True)
        sel = np.isin(mids, np.concatenate([ps, ns]))
        if mt[sel].sum() == 0 or (mt[sel] == 0).sum() == 0:
            continue
        vals.append(average_precision_score(mt[sel], mp[sel]))
    vals = np.asarray(vals)
    return {
        "median": float(np.median(vals)),
        "ci95_lower": float(np.percentile(vals, 2.5)),
        "ci95_upper": float(np.percentile(vals, 97.5)),
        "n_bootstrap": int(len(vals)),
    }


def main() -> None:
    global labels_np
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cached_features/features_full.npz")
    ap.add_argument("--ckpts", default="results/model_full")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    feat_array = d["feat_array"]
    windows_np = d["windows"]
    labels = d["labels"]
    mids = d["measurement_ids"]
    labels_np = np.asarray(labels)
    M = len(mids)
    print(f"Data: {M} measurements")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = torch.from_numpy(windows_np).float().to(device)
    feat = torch.from_numpy(feat_array).float().to(device)

    label_counts = labels_np.sum(axis=1).astype(int)
    folds = make_stratified_group_folds(mids, label_counts, n_splits=5, seed=args.seed)

    ckpt_dir = Path(args.ckpts)
    models = []
    for fi in range(5):
        model = build_model().to(device)
        ckpt = torch.load(ckpt_dir / f"model_fold{fi + 1}.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        models.append(model)
        print(f"Loaded fold {fi + 1} ckpt")

    n_params = count_params(models[0])
    report: dict = {
        "model": "dual + gated_attention + cyclic + noisy-OR",
        "n_params": n_params,
        "n_measurements": int(M),
        "checkpoint_source": str(ckpt_dir),
    }

    # ---- baseline OOF ----
    base_p = oof_evaluate(models, folds, windows, feat, args.batch_size)
    report["baseline_phase"] = phase_metrics(base_p, labels_np)
    report["baseline_measurement"] = measurement_metrics(base_p, labels_np)
    print(f"Baseline phase PR-AUC: {report['baseline_phase']['pr_auc']:.4f}, "
          f"meas PR-AUC: {report['baseline_measurement']['pr_auc']:.4f}")

    # ---- latency ----
    torch.cuda.synchronize() if device.type == "cuda" else None
    model = models[0].eval()
    wb = windows[:64]
    fb = feat[:64]
    for _ in range(5):
        with torch.no_grad():
            model(wb, fb)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    n_rep = 20
    for _ in range(n_rep):
        with torch.no_grad():
            model(wb, fb)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = (time.perf_counter() - t0) / n_rep
    report["latency"] = {
        "ms_per_batch64": round(elapsed * 1000, 2),
        "ms_per_measurement": round(elapsed * 1000 / 64, 3),
        "device": str(device),
    }

    # ---- additive noise ----
    def add_noise(snr_db: float, gen: torch.Generator):
        def _p(w: torch.Tensor) -> torch.Tensor:
            sig_power = w.pow(2).mean(dim=-1, keepdim=True)
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(w, generator=gen) * torch.sqrt(noise_power)
            return w + noise
        return _p

    report["noise"] = {}
    for snr in [20, 10, 5]:
        noise_gen = torch.Generator(device=device).manual_seed(1000 + snr)
        p = oof_evaluate(models, folds, windows, feat, args.batch_size, perturb=add_noise(snr, noise_gen))
        report["noise"][f"snr_{snr}db"] = {
            "phase": phase_metrics(p, labels_np),
            "measurement": measurement_metrics(p, labels_np),
        }
        print(f"Noise SNR={snr}dB: phase PR-AUC={report['noise'][f'snr_{snr}db']['phase']['pr_auc']:.4f}")

    # ---- amplitude scaling ----
    report["scale"] = {}
    for scale in [0.8, 1.2]:
        p = oof_evaluate(models, folds, windows, feat, args.batch_size,
                         perturb=lambda w, s=scale: w * s)
        report["scale"][f"x{scale}"] = {
            "phase": phase_metrics(p, labels_np),
            "measurement": measurement_metrics(p, labels_np),
        }
        print(f"Scale x{scale}: phase PR-AUC={report['scale'][f'x{scale}']['phase']['pr_auc']:.4f}")

    # ---- time shift ----
    report["shift"] = {}
    report["shift_metadata"] = {
        "method": "zero_pad_fixed_window",
        "features": "recomputed_from_shifted_windows",
        "note": ("cyclic-roll implementation (pre-fix) produced a period-256 "
                 "strided-subsampling artifact (e.g. +-256 ~ baseline, +-64 max drop) "
                 "and was replaced; see results/shift_artifact_audit.md"),
    }
    sr_hz = 40_000_000
    feat_names = list(d["feature_names"])
    for shift in [-128, -64, 64, 128]:
        w_shifted = time_shift(windows_np, shift)
        feats_shifted = extract_physical_features(w_shifted, sr_hz, batch_size=4096)
        feat_arr_shifted = np.stack([feats_shifted[n] for n in feat_names], axis=-1).reshape(M, 3, 8, -1)
        w_t = torch.from_numpy(w_shifted).float().to(device)
        f_t = torch.from_numpy(feat_arr_shifted).float().to(device)
        p = oof_evaluate(models, folds, w_t, f_t, args.batch_size)
        report["shift"][f"{shift:+d}"] = {
            "phase": phase_metrics(p, labels_np),
            "measurement": measurement_metrics(p, labels_np),
        }
        print(f"Shift {shift:+d}: phase PR-AUC={report['shift'][f'{shift:+d}']['phase']['pr_auc']:.4f}")

    # ---- missing phase ----
    mask = torch.ones(M, 3, dtype=torch.bool, device=device)
    report["missing_phase"] = {}
    for miss in [0, 1, 2]:
        pm = mask.clone()
        pm[:, miss] = False
        p = oof_evaluate(models, folds, windows, feat, args.batch_size, phase_mask=pm)
        report["missing_phase"][f"miss_phase_{miss}"] = {
            "phase": phase_metrics(p, labels_np),
            "measurement": measurement_metrics(p, labels_np),
        }
        print(f"Missing phase {miss}: meas PR-AUC={report['missing_phase'][f'miss_phase_{miss}']['measurement']['pr_auc']:.4f}")

    # ---- bootstrap CI ----
    report["measurement_bootstrap_pr_auc_ci"] = measurement_bootstrap_ci(base_p, labels_np, mids, n_boot=2000, seed=args.seed)
    print("Measurement-level bootstrap CI:", report["measurement_bootstrap_pr_auc_ci"])

    # ---- calibration on OOF ----
    cal = compute_calibration_metrics(base_p.flatten(), labels_np.flatten())
    report["oof_calibration"] = cal

    # ---- relative degradation vs baseline ----
    base_phase = report["baseline_phase"]["pr_auc"]
    base_meas = report["baseline_measurement"]["pr_auc"]
    for group in ["noise", "scale", "shift"]:
        for key, val in report[group].items():
            p = val["phase"]["pr_auc"]
            m = val["measurement"]["pr_auc"]
            val["phase_pr_auc_drop_pct"] = round(float((base_phase - p) / base_phase * 100), 2) if base_phase else None
            val["measurement_pr_auc_drop_pct"] = round(float((base_meas - m) / base_meas * 100), 2) if base_meas else None
    for key, val in report["missing_phase"].items():
        m = val["measurement"]["pr_auc"]
        val["measurement_pr_auc_drop_pct"] = round(float((base_meas - m) / base_meas * 100), 2) if base_meas else None

    out = Path("results/robustness_report.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
