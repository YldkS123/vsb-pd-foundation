# -*- coding: utf-8 -*-
"""External cross-domain experiments for the VSB 80k lightweight CNN.

Protocol
--------
1. Train the final lightweight model (encoder=cnn, MIL=attention,
   phase=mean, 80,113 params) on the full VSB development set with the same
   locked hyperparameters used in the ablations, then save checkpoints.
2. For external single-channel datasets, reuse the trained encoder +
   MIL aggregator + phase classifier as a per-measurement adapter (the
   three-phase interaction module is VSB-specific and is not applied).
3. Evaluate three arms:
     zero-shot   : VSB-trained weights, frozen
     from-scratch: train on the external training split
     fine-tune   : initialize from the VSB checkpoint, train on external data
   Metrics are ROC-AUC, PR-AUC, F1@0.5, accuracy and measurement-level
   bootstrap 95% CIs.  The VSB blind test is never touched.

Usage examples
--------------
python scripts/external_transfer.py train-vsb --out-dir results/external/checkpoints
python scripts/external_transfer.py eval --dataset figshare_24033225 \
    --checkpoint results/external/checkpoints/cnn80k_vsbdev_seed42.pt \
    --out-dir results/external/24033225
python scripts/external_transfer.py train-external --dataset figshare_24033225 \
    --mode from-scratch --out-dir results/external/24033225
python scripts/external_transfer.py train-external --dataset figshare_24033225 \
    --mode fine-tune --checkpoint results/external/checkpoints/cnn80k_vsbdev_seed42.pt \
    --out-dir results/external/24033225
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vsb_pd.encoder import WindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.cyclic import PhaseInteractionModule
from vsb_pd.model import VSBPipeline
from vsb_pd.training import compute_metrics
from vsb_pd.evaluation import compute_bootstrap_ci

WINDOW_LENGTH = 8192
HIDDEN_DIM = 128
FEATURE_DIM = 58
EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42


def build_model(branch: str = "cnn") -> VSBPipeline:
    return VSBPipeline(
        encoder=WindowEncoder(WINDOW_LENGTH, FEATURE_DIM, HIDDEN_DIM, branch=branch),
        aggregator=MILAggregator("attention", HIDDEN_DIM),
        cyclic=PhaseInteractionModule("mean", HIDDEN_DIM),
        classifier=PhaseClassifier(HIDDEN_DIM),
    )


class ExternalHead(nn.Module):
    """Encoder + MIL + classifier adapter for single-channel external data."""

    def __init__(self, branch: str = "cnn"):
        super().__init__()
        self.encoder = WindowEncoder(WINDOW_LENGTH, FEATURE_DIM, HIDDEN_DIM, branch=branch)
        self.aggregator = MILAggregator("attention", HIDDEN_DIM)
        self.classifier = PhaseClassifier(HIDDEN_DIM)

    def forward(self, windows: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # windows: (B, K, 8192); features: (B, K, 58)
        encoded = self.encoder(windows, features)  # (B, K, 128)
        agg = self.aggregator(encoded)  # (B, 128)
        return self.classifier(agg).squeeze(-1)  # (B,) logit

    @classmethod
    def from_pipeline(cls, pipeline: VSBPipeline) -> "ExternalHead":
        head = cls(branch=pipeline.encoder.branch)
        head.encoder.load_state_dict(pipeline.encoder.state_dict())
        head.aggregator.load_state_dict(pipeline.aggregator.state_dict())
        head.classifier.load_state_dict(pipeline.classifier.state_dict())
        return head


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# VSB development training
# ---------------------------------------------------------------------------

def load_vsb_dev() -> dict:
    cache = ROOT / "results" / "cached_features" / "features_full.npz"
    d = np.load(cache, allow_pickle=True)
    return {
        "windows": d["windows"].astype(np.float32),
        "features": d["feat_array"].astype(np.float32),
        "labels": d["labels"].astype(np.float32),
        "measurement_ids": d["measurement_ids"],
    }


def train_vsb(out_dir: Path, seed: int = SEED) -> Path:
    data = load_vsb_dev()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    windows = torch.from_numpy(data["windows"]).float()
    features = torch.from_numpy(data["features"]).float()
    labels = torch.from_numpy(data["labels"]).float()

    model = build_model("cnn").to(device)
    n_params = count_params(model)
    print(f"VSB dev: {len(windows)} measurements, params={n_params}, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    indices = np.arange(len(windows))
    losses = []

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(indices), BATCH_SIZE):
            bidx = indices[i : i + BATCH_SIZE]
            w = windows[bidx].to(device)
            f = features[bidx].to(device)
            y = labels[bidx].to(device)
            optimizer.zero_grad()
            logits, _ = model(w, f)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        losses.append(total_loss / n_batches)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:3d}: loss={losses[-1]:.4f} ({time.time()-t0:.0f}s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"cnn80k_vsbdev_seed{seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"encoder": "cnn", "mil": "attention", "phase": "mean"},
            "n_params": n_params,
            "seed": seed,
            "epochs": EPOCHS,
            "train_losses": losses,
            "trained_on": "vsb_development_full",
        },
        ckpt,
    )
    print(f"Saved {ckpt}")
    return ckpt


# ---------------------------------------------------------------------------
# External dataset loaders
# ---------------------------------------------------------------------------

def load_figshare_24033225(split: str) -> dict:
    path = (
        ROOT / "data" / "external_datasets" / "figshare_24033225" / "processed" / f"{split}.npz"
    )
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)
    # Tile each 400-sample signal to the fixed 8192-window length.
    n_repeat = int(np.ceil(WINDOW_LENGTH / X.shape[1]))
    Xw = np.tile(X[:, None, :], (1, 1, n_repeat))[:, :, :WINDOW_LENGTH]
    return {
        "windows": np.ascontiguousarray(Xw),  # (N, 1, 8192)
        "features": np.zeros((len(X), 1, FEATURE_DIM), dtype=np.float32),
        "labels": y,
        "names": np.array([f"{split}_{i}" for i in range(len(X))], dtype=object),
    }


def load_figshare_28523090(task: str) -> dict:
    path = Path(r"D:\datasets\figshare_28523090\windows\dataset_windows.npz")
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)  # (N, K, 8192)
    class_idx = d["class_idx"].astype(int)
    channels = np.asarray(d["channels"])
    names = np.asarray(d["filenames"])
    classes = [
        "background",
        "background_day2",
        "corona",
        "hv_bg",
        "pd",
        "pd_corona",
        "pd_corona_HI",
        "pd_HI",
    ]
    if task == "pd_vs_background":
        pos = {4, 5, 6, 7}
        neg = {0, 1, 3}
    elif task == "pd_vs_corona":
        pos = {4, 7}
        neg = {2}
    else:
        raise ValueError(f"unknown task: {task}")
    mask = np.isin(class_idx, list(pos | neg))
    y = np.where(np.isin(class_idx[mask], list(pos)), 1.0, 0.0).astype(np.float32)
    return {
        "windows": X[mask],
        "features": np.zeros((mask.sum(), 8, FEATURE_DIM), dtype=np.float32),
        "labels": y,
        "channels": channels[mask],
        "names": names[mask],
        "class_idx": class_idx[mask],
        "classes": classes,
    }


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def build_external_head(ckpt: Path | None, device) -> ExternalHead:
    head = ExternalHead("cnn").to(device)
    if ckpt is not None and ckpt.exists():
        sd = torch.load(ckpt, map_location=device)
        missing, unexpected = head.load_state_dict(sd["state_dict"], strict=False)
        required_missing = [k for k in missing if not k.startswith("cyclic.")]
        if required_missing:
            raise RuntimeError(f"checkpoint missing required keys: {required_missing[:8]}")
        print(f"Loaded checkpoint {ckpt} (ignored {len(unexpected)} cyclic keys)")
    return head


@torch.no_grad()
def predict_head(
    head: nn.Module,
    windows: np.ndarray,
    features: np.ndarray,
    batch_size: int = 256,
    device=None,
) -> np.ndarray:
    head.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs: list[np.ndarray] = []
    N = len(windows)
    for i in range(0, N, batch_size):
        w = torch.from_numpy(windows[i : i + batch_size]).float().to(device)
        f = torch.from_numpy(features[i : i + batch_size]).float().to(device)
        logits = head(w, f)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def evaluate_external(
    head: nn.Module,
    dataset: dict,
    out_dir: Path,
    tag: str,
    device=None,
    bootstrap_n: int = 1000,
) -> dict:
    probs = predict_head(head, dataset["windows"], dataset["features"], device=device)
    labels = dataset["labels"]
    preds = (probs >= 0.5).astype(int)
    m = compute_metrics(labels, probs, preds)
    mids = np.arange(len(labels))

    cis = {}
    for metric in ("roc_auc", "pr_auc"):
        try:
            ci = compute_bootstrap_ci(probs, labels, mids, metric_name=metric, n_bootstrap=bootstrap_n)
            cis[metric] = {
                "median": round(ci.median, 4),
                "lower": round(ci.lower, 4),
                "upper": round(ci.upper, 4),
            }
        except Exception as exc:
            cis[metric] = {"error": str(exc)}

    result = {
        "tag": tag,
        "n": int(len(labels)),
        "n_pos": int(labels.sum()),
        "n_neg": int(len(labels) - labels.sum()),
        "roc_auc": round(float(m["roc_auc"]), 4),
        "pr_auc": round(float(m["pr_auc"]), 4),
        "f1_0.5": round(float(m["f1"]), 4),
        "accuracy": round(float(m["accuracy"]), 4),
        "bootstrap_ci": cis,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez(out_dir / f"{tag}.npz", probs=probs, labels=labels)
    print(json.dumps(result, indent=2))
    return result


def train_external(
    dataset_name: str,
    mode: str,
    ckpt: Path | None,
    out_dir: Path,
    seed: int = SEED,
    task: str | None = None,
    split: str = "c1c2",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_name == "figshare_24033225":
        train_data = load_figshare_24033225("Tr0")
        val_data = load_figshare_24033225("Va0")
        test_data = load_figshare_24033225("Te0")
    elif dataset_name == "figshare_28523090":
        full = load_figshare_28523090(task or "pd_vs_background")
        names = np.asarray(full["names"])
        channels = np.asarray(full["channels"])
        rng = np.random.default_rng(seed)
        if split == "c1c2":
            # C1 recordings split into train/val groups; C2 stays untouched as test.
            c1_groups = np.unique(names[channels == "C1"])
            perm = rng.permutation(len(c1_groups))
            n_val = max(1, int(round(0.2 * len(c1_groups))))
            val_groups = set(c1_groups[perm[:n_val]].tolist())
            tr_groups = set(c1_groups[perm[n_val:]].tolist())
            te_groups = set(np.unique(names[channels == "C2"]).tolist())
        else:  # random 70/15/15 split by recording
            all_groups = np.unique(names)
            perm = rng.permutation(len(all_groups))
            n_tr = max(1, int(round(0.7 * len(all_groups))))
            n_val = max(1, int(round(0.15 * len(all_groups))))
            tr_groups = set(all_groups[perm[:n_tr]].tolist())
            val_groups = set(all_groups[perm[n_tr:n_tr + n_val]].tolist())
            te_groups = set(all_groups[perm[n_tr + n_val:]].tolist())

        def _subset(mask):
            return {k: v[mask] for k, v in full.items() if isinstance(v, np.ndarray)}

        train_data = _subset(np.isin(names, list(tr_groups)))
        val_data = _subset(np.isin(names, list(val_groups)))
        test_data = _subset(np.isin(names, list(te_groups)))
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    head = build_external_head(ckpt if mode == "fine-tune" else None, device)
    print(
        f"train-external {dataset_name} mode={mode} n_train={len(train_data['labels'])} "
        f"n_test={len(test_data['labels'])} device={device}"
    )

    windows = torch.from_numpy(train_data["windows"]).float()
    features = torch.from_numpy(train_data["features"]).float()
    labels = torch.from_numpy(train_data["labels"]).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    indices = np.arange(len(labels))
    best_val = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(EPOCHS):
        head.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(indices), BATCH_SIZE):
            bidx = indices[i : i + BATCH_SIZE]
            w = windows[bidx].to(device)
            f = features[bidx].to(device)
            y = labels[bidx].to(device)
            optimizer.zero_grad()
            loss = criterion(head(w, f), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        val_probs = predict_head(head, val_data["windows"], val_data["features"], device=device)
        val_preds = (val_probs >= 0.5).astype(int)
        vm = compute_metrics(val_data["labels"], val_probs, val_preds)
        val_pr = float(vm["pr_auc"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pr_auc": val_pr})
        if val_pr > best_val + 0.001:
            best_val = val_pr
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stop at epoch {epoch}")
                break
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:3d}: loss={train_loss:.4f} val_pr_auc={val_pr:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset_name}_{mode}"
    if dataset_name == "figshare_28523090":
        tag += f"_{task}_{split}"
    evaluate_external(head, test_data, out_dir, f"{tag}_test", device=device)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "mode": mode,
            "dataset": dataset_name,
            "task": task,
            "split": split,
            "history": history,
            "best_val_pr_auc": best_val,
        },
        out_dir / f"{tag}.pt",
    )
    return {"tag": tag, "best_val_pr_auc": best_val, "history_tail": history[-3:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("train-vsb")
    p1.add_argument("--out-dir", type=Path, default=ROOT / "results" / "external" / "checkpoints")
    p1.add_argument("--seed", type=int, default=SEED)

    p2 = sub.add_parser("train-external")
    p2.add_argument("--dataset", choices=["figshare_24033225", "figshare_28523090"], required=True)
    p2.add_argument("--mode", choices=["from-scratch", "fine-tune"], required=True)
    p2.add_argument("--checkpoint", type=Path, default=None)
    p2.add_argument("--out-dir", type=Path, default=ROOT / "results" / "external")
    p2.add_argument("--task", default=None)
    p2.add_argument("--split", default="c1c2")
    p2.add_argument("--seed", type=int, default=SEED)

    p3 = sub.add_parser("eval")
    p3.add_argument("--dataset", choices=["figshare_24033225", "figshare_28523090"], required=True)
    p3.add_argument("--checkpoint", type=Path, required=True)
    p3.add_argument("--out-dir", type=Path, default=ROOT / "results" / "external")
    p3.add_argument("--task", default=None)
    p3.add_argument("--split-name", default="Te0")

    args = ap.parse_args()

    if args.command == "train-vsb":
        train_vsb(args.out_dir, seed=args.seed)
        return 0

    if args.command == "train-external":
        train_external(
            args.dataset, args.mode, args.checkpoint, args.out_dir,
            seed=args.seed, task=args.task, split=args.split,
        )
        return 0

    if args.command == "eval":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        head = build_external_head(args.checkpoint, device)
        if args.dataset == "figshare_24033225":
            ds = load_figshare_24033225(args.split_name)
            tag = f"zero_shot_{args.dataset}_{args.split_name}"
        else:
            ds = load_figshare_28523090(args.task or "pd_vs_background")
            tag = f"zero_shot_{args.dataset}_{args.task or 'pd_vs_background'}"
            mask = np.asarray(ds["channels"]) == "C2"
            ds = {k: v[mask] for k, v in ds.items() if isinstance(v, np.ndarray)}
            tag = f"zero_shot_{args.dataset}_{args.task or 'pd_vs_background'}_C2"
        evaluate_external(head, ds, args.out_dir, tag, device=device)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
