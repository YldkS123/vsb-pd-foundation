# -*- coding: utf-8 -*-
"""
TII-D1: External multi-dataset validation with the E4 mainline and the
light-transformer encoder (TII track: industrial multi-source validation).

Upgrades the historical 80k external transfer study (scripts/external_transfer.py)
to (a) the E4 mainline encoder weights (simple_cnn + attention MIL), and
(b) the light-transformer window encoder, under the same three-arm protocol:

  zero-shot   : VSB-trained weights, frozen on external test
  from-scratch: train on external training split
  fine-tune   : init from VSB checkpoint, train on external data

Datasets:
  figshare 24033225  - motor PD vs noise (400-point signals; Te0 test)
  figshare 28523090  - oscilloscope 8-class captures (C1 train -> C2 test,
                       PD vs background and PD vs corona)

Protocol: external data only; VSB blind/Harvard never touched; all arms
report ROC-AUC / PR-AUC / F1@0.5 on the frozen external test split.

Outputs (results/tii_external/):
  <dataset>/<arm>_<encoder>.json + .npz
  summary.json
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

from vsb_pd.dl_encoders import TimWindowEncoder, LightTransformerEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier
from vsb_pd.training import compute_metrics
from vsb_pd.cyclic import PhaseInteractionModule
from vsb_pd.model import VSBPipeline

OUT = ROOT / "results" / "tii_external"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_LENGTH = 8192
HIDDEN_DIM = 128
FEATURE_DIM = 58
EPOCHS = 40
BATCH_SIZE = 64
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

DATA240 = ROOT / "data" / "external_datasets" / "figshare_24033225" / "processed"
DATA285 = ROOT / "data" / "external_datasets" / "figshare_28523090" / "processed"


# ---------------------------------------------------------------------------
# Encoder-aware adapter: encoder + attention MIL + phase classifier
# ---------------------------------------------------------------------------
class ExternalHead(nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        self.encoder = TimWindowEncoder(encoder_name, WINDOW_LENGTH, FEATURE_DIM, HIDDEN_DIM)
        self.aggregator = MILAggregator("attention", HIDDEN_DIM)
        self.classifier = PhaseClassifier(HIDDEN_DIM)

    def forward(self, windows: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(windows, features)   # (B, K, 128)
        agg = self.aggregator(encoded)              # (B, 128)
        return self.classifier(agg).squeeze(-1)     # (B,) logit


class _LazyWindows3D:
    """Xt is already (N, K, 8192); chunk slices rows directly."""

    def __init__(self, Xt, y, n_windows, feat_dim):
        self.Xt = Xt
        self.y = y
        self.n_windows = n_windows
        self.feat_dim = feat_dim

    def __len__(self):
        return len(self.y)

    def chunk(self, i0: int, i1: int):
        Xc = self.Xt[i0:i1]
        N = len(Xc)
        f = np.zeros((N, self.n_windows, self.feat_dim), dtype=np.float32)
        return (torch.from_numpy(Xc.astype(np.float32)),
                torch.from_numpy(f),
                self.y[i0:i1])

    def take(self, indices: np.ndarray):
        """Return a chunk for arbitrary (possibly shuffled) row indices."""
        Xc = self.Xt[indices]
        N = len(Xc)
        f = np.zeros((N, self.n_windows, self.feat_dim), dtype=np.float32)
        return (torch.from_numpy(Xc.astype(np.float32)),
                torch.from_numpy(f),
                self.y[indices])


def make_external_data(X: np.ndarray, y: np.ndarray, tile_to: int = 8192, n_windows: int = 8):
    """Convert (N, 400) external signals to (N, K, 8192) window batches by
    tiling (with zero-padding to the last window), plus zero features.
    Returns a lazy wrapper (signals kept in RAM, windows built per chunk)."""
    N, L = X.shape
    if L < tile_to:
        reps = int(np.ceil(tile_to / L))
        Xt = np.tile(X, (1, reps))[:, :tile_to]          # (N, 8192)
    else:
        Xt = X[:, :tile_to]
    # build (N, K, 8192) lazily per chunk inside _LazyWindows; here we keep
    # the tiled 2D and let chunk expand K copies to bound memory.
    return _LazyWindows2D(Xt, y.astype(np.float32), n_windows, FEATURE_DIM)


class _LazyWindows2D:
    """Xt is (N, 8192); chunk expands to (N, K, 8192) windows per slice."""

    def __init__(self, Xt, y, n_windows, feat_dim):
        self.Xt = Xt          # (N, 8192) float32
        self.y = y            # (N,)
        self.n_windows = n_windows
        self.feat_dim = feat_dim

    def __len__(self):
        return len(self.y)

    def chunk(self, i0: int, i1: int):
        Xc = self.Xt[i0:i1]
        N = len(Xc)
        w = np.stack([Xc] * self.n_windows, axis=1)      # (N, K, 8192)
        f = np.zeros((N, self.n_windows, self.feat_dim), dtype=np.float32)
        return (torch.from_numpy(w.astype(np.float32)),
                torch.from_numpy(f),
                self.y[i0:i1])

    def take(self, indices: np.ndarray):
        Xc = self.Xt[indices]
        N = len(Xc)
        w = np.stack([Xc] * self.n_windows, axis=1)
        f = np.zeros((N, self.n_windows, self.feat_dim), dtype=np.float32)
        return (torch.from_numpy(w.astype(np.float32)),
                torch.from_numpy(f),
                self.y[indices])


def load_240():
    te = np.load(DATA240 / "Te0.npz", allow_pickle=True)
    tr = np.load(DATA240 / "Tr0.npz", allow_pickle=True)
    va = np.load(DATA240 / "Va0.npz", allow_pickle=True)
    return {
        "train": (tr["X"], tr["y"]), "val": (va["X"], va["y"]),
        "test": (te["X"], te["y"]),
    }


def load_285(task: str):
    """task: 'pd_vs_background' or 'pd_vs_corona' (C1 -> C2)."""
    sub = "C1_C2" if task == "pd_vs_background" else "C1_C2_corona"
    # inspect the actual layout
    import glob
    files = sorted(glob.glob(str(DATA285 / "*.npz")))
    if not files:
        # search nested
        files = sorted(glob.glob(str(DATA285 / "**" / "*.npz"), recursive=True))
    return files


def train_head(head, data, val_data, device, chunk_size=256):
    """data/val_data are _LazyWindows; train with chunked batches."""
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    idx = np.arange(len(data))
    best_pr, best_state, patience = -1.0, None, 0
    for epoch in range(EPOCHS):
        head.train()
        np.random.shuffle(idx)
        total, nb = 0.0, 0
        for i in range(0, len(idx), chunk_size):
            cidx = idx[i:i + chunk_size]
            cw, cf, cy = data.take(cidx)
            opt = optimizer
            opt.zero_grad()
            logits = head(cw.to(device), cf.to(device))
            loss = criterion(logits, torch.from_numpy(cy).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt.step()
            total += float(loss.item()); nb += 1
        # val (chunked)
        head.eval()
        vp_list = []
        with torch.no_grad():
            for i in range(0, len(val_data), 512):
                cw, cf, cy = val_data.chunk(i, min(i + 512, len(val_data)))
                vp_list.append(torch.sigmoid(head(cw.to(device), cf.to(device))).cpu().numpy())
        vp = np.concatenate(vp_list)
        from sklearn.metrics import average_precision_score
        pr = average_precision_score(val_data.y, vp)
        if pr > best_pr + 0.001:
            best_pr = pr
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head


def evaluate(head, data, device, batch_size=256):
    head.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            cw, cf, cy = data.chunk(i, min(i + batch_size, len(data)))
            probs.append(torch.sigmoid(head(cw.to(device), cf.to(device))).cpu().numpy())
    p = np.concatenate(probs)
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    return {
        "roc_auc": round(float(roc_auc_score(data.y, p)), 4),
        "pr_auc": round(float(average_precision_score(data.y, p)), 4),
        "f1_05": round(float(f1_score(data.y, (p >= 0.5).astype(int))), 4),
        "n_test": int(len(data.y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="simple_cnn,lt_transformer")
    ap.add_argument("--datasets", default="24033225,28523090")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]

    # ---- VSB dev checkpoint weights (E4 mainline, seed 42, 5-fold avg is
    #      not needed: use the E4 config trained on full dev via stage1
    #      runner? We reuse the locked E4 OOF-trained weights if available,
    #      otherwise train a quick full-dev E4.) ----
    # Simplest robust choice: train E4 on full dev per encoder here (same
    # protocol), then run the three arms. This is the "VSB checkpoint".

    d = np.load(ROOT / "results" / "cached_features" / "features_full.npz", allow_pickle=False)
    wdev = np.asarray(d["windows"]).astype(np.float32)       # (M,3,K,8192)
    ydev = np.asarray(d["labels"], dtype=np.float32)         # (M,3)
    # ExternalHead is a single-channel adapter: flatten phases -> (M*3,K,8192)
    M = wdev.shape[0]
    wdev_flat = wdev.reshape(M * 3, 8, 8192)
    yflat = ydev.reshape(-1)                                  # (M*3,)
    # VSB dev windows are already (N, K, 8192); reuse _LazyWindows (2D-holder
    # is for 400-point external signals; for VSB we keep a 3D holder class)
    vsb_data = _LazyWindows3D(wdev_flat, yflat, 8, FEATURE_DIM)
    print(f"[vsb] dev {M} measurements -> {len(wdev_flat)} single-channel samples")

    summary = {}
    for enc_name in encoders:
        # 1) train VSB checkpoint (full dev, phase labels, single-channel)
        ckpt_path = OUT / f"vsb_ckpt_{enc_name}.pt"
        if ckpt_path.exists():
            head = ExternalHead(enc_name).to(device)
            head.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[{enc_name}] loaded VSB checkpoint")
        else:
            print(f"[{enc_name}] training VSB dev checkpoint...")
            head = ExternalHead(enc_name).to(device)
            t0 = time.time()
            head = train_head(head, vsb_data, vsb_data, device)
            torch.save(head.state_dict(), ckpt_path)
            print(f"[{enc_name}] VSB checkpoint saved ({time.time()-t0:.0f}s)")

        # 2) figshare 24033225 (motor PD vs noise)
        if "24033225" in args.datasets:
            data = load_240()
            trX, try_ = data["train"]; vaX, vay = data["val"]; teX, tey = data["test"]
            trw = make_external_data(trX, try_)
            vaw = make_external_data(vaX, vay)
            tew = make_external_data(teX, tey)
            # zero-shot
            zs = evaluate(head, tew, device)
            summary[f"24033225_{enc_name}_zero_shot"] = zs
            # from-scratch
            fs_head = ExternalHead(enc_name).to(device)
            fs_head = train_head(fs_head, trw, vaw, device)
            fs = evaluate(fs_head, tew, device)
            summary[f"24033225_{enc_name}_from_scratch"] = fs
            # fine-tune
            ft_head = ExternalHead(enc_name).to(device)
            ft_head.load_state_dict(head.state_dict())
            ft_head = train_head(ft_head, trw, vaw, device)
            ft = evaluate(ft_head, tew, device)
            summary[f"24033225_{enc_name}_fine_tune"] = ft
            print(f"[24033225][{enc_name}] zero {zs} | fs {fs} | ft {ft}")

        # 3) figshare 28523090 (oscilloscope, C1->C2)
        if "28523090" in args.datasets:
            import glob as _g
            files = sorted(_g.glob(str(DATA285 / "**" / "*.npz"), recursive=True))
            print(f"[28523090] files: {[Path(f).name for f in files]}")
            # structure depends on the processed layout; handle below in a
            # second pass after inspecting the actual file list.
            summary[f"28523090_{enc_name}_note"] = \
                f"files found: {[Path(f).name for f in files]}"

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== TII EXTERNAL SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nDone. Output in", OUT)


if __name__ == "__main__":
    main()
