# -*- coding: utf-8 -*-
"""SOTA baselines on the locked development folds.

Runs two published-method adaptations on the same K=8 mixed-window cache,
same StratifiedGroupKFold(5, seed=42) measurement splits and same dev-only
tuning discipline as the mainline:

  sota_tfcnn_zheng2022:
      Time-frequency (STFT spectrogram) features + 2D CNN, following
      Zheng et al. 2022, adapted to the paper's 1D window/MIL pipeline.
      Trained with the same Mean MIL + Max phase interaction used by the
      deep generic encoders.

  sota_cnn_qsvm_fei2024:
      Ensemble lightweight CNN feature extractor + quadratic SVM,
      following Fei et al. 2024. Window labels are derived from phase labels
      for CNN pretraining; the quadratic SVM is trained on mean-pooled
      phase embeddings with phase labels. Window-level CNN predictions are
      not used directly in this adaptation.

Usage:
  python scripts/run_sota_baselines.py \
      --cache results/cached_features/features_policy_mixed_k8.npz \
      --methods tf_cnn,cnn_qsvm \
      --epochs 40 --batch-size 64 --patience 15 \
      --cnn-epochs 20 --ensemble-seeds 3 \
      --tag sota_baselines --out-dir results/sota_baselines
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
import run_dl_baselines as dl_runner
from vsb_pd.encoder import CNNBranch
from vsb_pd.training import make_stratified_group_folds, compute_metrics

N_FOLDS = 5


# --------------------------------------------------------------------------- #
# Zheng-style time-frequency CNN (reuses the standard DL pipeline)
# --------------------------------------------------------------------------- #
def run_tf_cnn(
    windows: torch.Tensor,
    feat: torch.Tensor,
    labels_t: torch.Tensor,
    mids: np.ndarray,
    folds,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    out_dir: Path,
) -> dict:
    config_name = "sota_tfcnn_zheng2022"
    summary_path = out_dir / config_name / "cv_summary.json"
    if summary_path.exists():
        print(f"  resume: {config_name} already complete")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    print(f"  training {config_name} ...", flush=True)
    return dl_runner.train_config(
        config_name, "tf_cnn", windows, feat, labels_t, mids, folds,
        epochs, batch_size, patience, seed, out_dir,
    )


# --------------------------------------------------------------------------- #
# Fei-style ensemble CNN + quadratic SVM
# --------------------------------------------------------------------------- #
class WindowCNN(nn.Module):
    def __init__(self, window_length: int = 8192, hidden_dim: int = 128):
        super().__init__()
        self.cnn = CNNBranch(window_length, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) -> (B, 128) -> (B, 1)
        return self.head(self.cnn(x))


def _windows_of(windows_np: np.ndarray, indices) -> np.ndarray:
    return np.asarray(windows_np[indices], dtype=np.float32).reshape(-1, 1, windows_np.shape[-1])


def _labels_of(labels_np: np.ndarray, indices, k: int) -> np.ndarray:
    return np.repeat(labels_np[indices], k, axis=1).reshape(-1).astype(np.float32)


def _train_window_cnn(
    windows_np: np.ndarray,
    labels_np: np.ndarray,
    tr_idx,
    va_idx,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    device: torch.device,
) -> tuple[nn.Module, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WindowCNN(windows_np.shape[-1], 128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    n_params = sum(p.numel() for p in model.parameters())

    best_auc = -1.0
    best_state = None
    patience_counter = 0
    train_idx = list(tr_idx)
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_idx)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_idx), batch_size):
            bidx = train_idx[i:i + batch_size]
            x = torch.from_numpy(_windows_of(windows_np, bidx)).to(device)
            y = torch.from_numpy(_labels_of(labels_np, bidx, windows_np.shape[2])).to(device)
            opt.zero_grad()
            loss = criterion(model(x).squeeze(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        val_probs = _predict_window_cnn(model, windows_np, va_idx, batch_size, device)
        val_y = _labels_of(labels_np, va_idx, windows_np.shape[2])
        m = compute_metrics(val_y, val_probs, (val_probs >= 0.5).astype(int))
        auc = m.get("pr_auc", float("nan"))
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"    CNN member seed={seed} epoch {epoch:3d}: loss={total_loss / max(n_batches, 1):.4f} "
                  f"window_pr_auc={auc:.4f}", flush=True)
        if np.isfinite(auc) and auc > best_auc + 0.001:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    CNN member seed={seed} early stop at epoch {epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, n_params


@torch.no_grad()
def _predict_window_cnn(
    model: nn.Module,
    windows_np: np.ndarray,
    indices,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(indices), batch_size):
        bidx = indices[i:i + batch_size]
        x = torch.from_numpy(_windows_of(windows_np, bidx)).to(device)
        outs.append(torch.sigmoid(model(x).squeeze(-1)).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.array([])


@torch.no_grad()
def _embed_windows(
    model: nn.Module,
    windows_np: np.ndarray,
    indices,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    embs = []
    for i in range(0, len(indices), batch_size):
        bidx = indices[i:i + batch_size]
        x = torch.from_numpy(_windows_of(windows_np, bidx)).to(device)
        embs.append(model.cnn(x).cpu().numpy())
    return np.concatenate(embs, axis=0) if embs else np.array([])


def _aggregate_phase(emb: np.ndarray, n_meas: int, p: int, k: int) -> np.ndarray:
    # emb: (n_meas*p*k, D) -> (n_meas, p, D)
    return emb.reshape(n_meas, p, k, -1).mean(axis=2)


def run_cnn_qsvm(
    windows_np: np.ndarray,
    labels_np: np.ndarray,
    mids: np.ndarray,
    folds,
    seed: int,
    batch_size: int,
    patience: int,
    cnn_epochs: int,
    ensemble_seeds: int,
    out_dir: Path,
) -> dict:
    from sklearn.svm import SVC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, P, K, L = windows_np.shape
    config_name = "sota_cnn_qsvm_fei2024"
    cfg_dir = out_dir / config_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg_dir / "cv_summary.json"
    if summary_path.exists():
        print(f"  resume: {config_name} already complete")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    fold_dir = cfg_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    oof_probs = np.zeros((M, P), dtype=np.float64)
    oof_targets = np.zeros((M, P), dtype=np.float64)
    fold_assign = np.zeros(M, dtype=np.int8)
    n_cnn_params = 0
    n_sv = 0

    start_fold = 0
    for fi in range(N_FOLDS):
        fold_file = fold_dir / f"fold_{fi + 1}.json"
        oof_file = fold_dir / f"oof_fold_{fi + 1}.npz"
        if fold_file.exists() and oof_file.exists():
            fold_metrics.append(json.loads(fold_file.read_text(encoding="utf-8")))
            d = np.load(oof_file, allow_pickle=True)
            va = d["va_indices"]
            oof_probs[va] = d["phase_probs"]
            oof_targets[va] = d["phase_targets"]
            fold_assign[va] = fi
        else:
            start_fold = fi
            break
    if start_fold > 0:
        print(f"  resume: {config_name} from fold {start_fold + 1}", flush=True)

    for fi, (tr, va) in enumerate(folds):
        if fi < start_fold:
            continue
        t0 = time.time()
        emb_tr = np.zeros((len(tr) * P * K, 128), dtype=np.float32)
        emb_va = np.zeros((len(va) * P * K, 128), dtype=np.float32)
        for si in range(ensemble_seeds):
            cnn, np_ = _train_window_cnn(
                windows_np, labels_np, tr, va, seed + si * 7,
                cnn_epochs, batch_size, patience, device,
            )
            n_cnn_params = max(n_cnn_params, np_)
            emb_tr += _embed_windows(cnn, windows_np, tr, batch_size, device).astype(np.float32)
            emb_va += _embed_windows(cnn, windows_np, va, batch_size, device).astype(np.float32)
            torch.cuda.empty_cache()
            del cnn
        emb_tr /= ensemble_seeds
        emb_va /= ensemble_seeds

        X_tr = _aggregate_phase(emb_tr, len(tr), P, K).reshape(len(tr) * P, -1)
        X_va = _aggregate_phase(emb_va, len(va), P, K).reshape(len(va) * P, -1)
        y_tr = labels_np[tr].reshape(-1)
        y_va = labels_np[va].reshape(-1)

        svm = SVC(
            C=1.0, kernel="poly", degree=2, gamma="scale",
            class_weight="balanced", probability=True, random_state=seed,
        )
        svm.fit(X_tr, y_tr)
        n_sv += int(getattr(svm, "n_support_", [0]).sum())

        phase_probs = svm.predict_proba(X_va)[:, 1].reshape(len(va), P)
        meas_probs = 1.0 - np.prod(1.0 - phase_probs, axis=1)
        meas_targets = labels_np[va].max(axis=1)

        p_flat = phase_probs.flatten()
        t_flat = y_va
        fm = compute_metrics(t_flat, p_flat, (p_flat >= 0.5).astype(int))
        mm = compute_metrics(meas_targets, meas_probs, (meas_probs >= 0.5).astype(int))
        elapsed = time.time() - t0

        fold_metrics.append({
            "fold": fi + 1,
            "ensemble_members": ensemble_seeds,
            "cnn_epochs": cnn_epochs,
            "elapsed_s": round(elapsed, 1),
            "phase": {k: round(float(v), 4) for k, v in fm.items()},
            "measurement": {k: round(float(v), 4) for k, v in mm.items()},
        })
        (fold_dir / f"fold_{fi + 1}.json").write_text(
            json.dumps(fold_metrics[-1], indent=2, default=str), encoding="utf-8")
        np.savez_compressed(
            fold_dir / f"oof_fold_{fi + 1}.npz",
            va_indices=np.asarray(va), phase_probs=phase_probs, phase_targets=labels_np[va],
        )
        print(f"  Fold {fi+1}: phase_pr_auc={fm.get('pr_auc', float('nan')):.4f} "
              f"meas_pr_auc={mm.get('pr_auc', float('nan')):.4f} ({elapsed:.0f}s)", flush=True)

        oof_probs[va] = phase_probs
        oof_targets[va] = labels_np[va]
        fold_assign[va] = fi

    np.savez_compressed(
        cfg_dir / "oof.npz",
        phase_probs=oof_probs, phase_targets=oof_targets,
        measurement_ids=mids, fold_assign=fold_assign,
    )

    prs = [f["phase"]["pr_auc"] for f in fold_metrics]
    rocs = [f["phase"]["roc_auc"] for f in fold_metrics]
    f1s = [f["phase"]["f1"] for f in fold_metrics]
    mprs = [f["measurement"]["pr_auc"] for f in fold_metrics]
    summary = {
        "config": {
            "method": "cnn_qsvm_fei2024",
            "encoder": "simple_cnn",
            "mil": "mean",
            "classifier": "quadratic_svm",
            "ensemble_members": ensemble_seeds,
        },
        "n_measurements": int(M),
        "n_folds": len(fold_metrics),
        "n_params_cnn": int(n_cnn_params),
        "n_support_vectors": int(n_sv),
        "mean_phase_pr_auc": round(float(np.mean(prs)), 4),
        "std_phase_pr_auc": round(float(np.std(prs)), 4),
        "mean_phase_roc_auc": round(float(np.mean(rocs)), 4),
        "mean_phase_f1_0.5": round(float(np.mean(f1s)), 4),
        "mean_measurement_pr_auc": round(float(np.mean(mprs)), 4),
        "folds": fold_metrics,
    }
    (cfg_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cached_features/features_policy_mixed_k8.npz")
    ap.add_argument("--methods", default="tf_cnn,cnn_qsvm")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--cnn-epochs", type=int, default=20)
    ap.add_argument("--ensemble-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--out-dir", default="results/sota_baselines")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    windows_np = np.asarray(d["windows"], dtype=np.float32)
    feat_array = np.asarray(d["feat_array"], dtype=np.float32)
    labels_np = np.asarray(d["labels"], dtype=np.float32)
    mids = np.asarray(d["measurement_ids"])

    if args.subset:
        take = min(args.subset, len(mids))
        windows_np = windows_np[:take]
        feat_array = feat_array[:take]
        labels_np = labels_np[:take]
        mids = mids[:take]

    dl_runner.labels_np = labels_np
    M = len(mids)
    print(f"Data: {M} measurements, windows={windows_np.shape}, features={feat_array.shape}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    windows = torch.from_numpy(windows_np).float()
    feat = torch.from_numpy(feat_array).float()
    labels_t = torch.from_numpy(labels_np).float()

    label_counts = labels_np.sum(axis=1).astype(int)
    folds = make_stratified_group_folds(mids, label_counts, n_splits=N_FOLDS, seed=args.seed)
    print(f"Folds: {[(len(tr), len(va)) for tr, va in folds]}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    summaries = []
    for i, method in enumerate(methods):
        print(f"\n[{i+1}/{len(methods)}] {method}", flush=True)
        if method == "tf_cnn":
            s = run_tf_cnn(
                windows, feat, labels_t, mids, folds,
                args.epochs, args.batch_size, args.patience, args.seed, out_dir,
            )
        elif method == "cnn_qsvm":
            s = run_cnn_qsvm(
                windows_np, labels_np, mids, folds,
                args.seed, args.batch_size, args.patience,
                args.cnn_epochs, args.ensemble_seeds, out_dir,
            )
        else:
            raise ValueError(f"Unknown SOTA method: {method}")
        summaries.append(s)

    print("\n" + "=" * 100, flush=True)
    for s in summaries:
        c = s["config"]
        print(f"{c}: phase_pr_auc={s['mean_phase_pr_auc']:.4f}±{s['std_phase_pr_auc']:.4f} "
              f"meas_pr_auc={s['mean_measurement_pr_auc']:.4f}", flush=True)

    (out_dir / "sota_summary.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_dir / 'sota_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
