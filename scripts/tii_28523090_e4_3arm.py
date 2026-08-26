# -*- coding: utf-8 -*-
"""
TII-E4-3arm: E4 mainline three-arm evaluation on figshare 28523090
(cross-device oscilloscope, C1 train -> C2 test).

Protocol mirrors the historical external_transfer.py (C1 split into train/val
by recording group; C2 untouched as test) but uses the E4 mainline encoder
(simple_cnn) through the ExternalHead adapter:

  zero-shot   : VSB-trained E4 checkpoint, frozen on C2 test
  from-scratch: train on C1 train, evaluate on C2 test
  fine-tune   : init from VSB checkpoint, train on C1 train, evaluate on C2

Tasks: pd_vs_background (PD vs background) and pd_vs_corona (PD vs corona).
Outputs: results/tii_external/28523090_<task>_<arm>_e4.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vsb_pd.dl_encoders import TimWindowEncoder
from vsb_pd.mil import MILAggregator, PhaseClassifier

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

DATA = ROOT / "data" / "external_datasets" / "figshare_28523090" / "processed" / "dataset_windows.npz"

CLASSES = ["background", "background_day2", "corona", "hv_bg", "pd",
           "pd_corona", "pd_corona_HI", "pd_HI"]


class ExternalHead(nn.Module):
    def __init__(self, encoder_name: str = "simple_cnn"):
        super().__init__()
        self.encoder = TimWindowEncoder(encoder_name, WINDOW_LENGTH, FEATURE_DIM, HIDDEN_DIM)
        self.aggregator = MILAggregator("attention", HIDDEN_DIM)
        self.classifier = PhaseClassifier(HIDDEN_DIM)

    def forward(self, windows, features):
        encoded = self.encoder(windows, features)
        agg = self.aggregator(encoded)
        return self.classifier(agg).squeeze(-1)


def load(task: str):
    d = np.load(DATA, allow_pickle=True)
    X = np.asarray(d["X"])            # (N, K, 8192)
    cls = np.asarray(d["class_idx"])
    channels = np.asarray(d["channels"])
    names = np.asarray(d["filenames"])
    if task == "pd_vs_background":
        keep = np.isin(cls, [CLASSES.index("pd"), CLASSES.index("background")])
        y = (cls == CLASSES.index("pd")).astype(np.float32)
    elif task == "pd_vs_corona":
        keep = np.isin(cls, [CLASSES.index("pd"), CLASSES.index("corona")])
        y = (cls == CLASSES.index("pd")).astype(np.float32)
    else:
        raise ValueError(task)
    X, y, channels, names = X[keep], y[keep], channels[keep], names[keep]
    return X, y, channels, names


def group_split(names, channels, seed=SEED):
    """C1 recordings split into train/val by group; C2 untouched as test."""
    rng = np.random.default_rng(seed)
    c1 = np.where(channels == "C1")[0]
    c2 = np.where(channels == "C2")[0]
    # group by recording (filename base without _window suffix -> use name)
    groups = np.unique(names[c1])
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * 0.2))
    val_groups = set(groups[:n_val].tolist())
    tr = np.array([i for i in c1 if names[i] not in val_groups])
    va = np.array([i for i in c1 if names[i] in val_groups])
    return tr, va, c2


class LazyData:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def take(self, idx):
        return (torch.from_numpy(self.X[idx].astype(np.float32)),
                torch.zeros(len(idx), self.X.shape[1], FEATURE_DIM, dtype=torch.float32),
                self.y[idx])


def train_head(head, data, val_data, device, chunk=128):
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()
    idx = np.arange(len(data))
    best_pr, best_state, patience = -1.0, None, 0
    for epoch in range(EPOCHS):
        head.train()
        np.random.shuffle(idx)
        for i in range(0, len(idx), chunk):
            cidx = idx[i:i + chunk]
            w, f, y = data.take(cidx)
            opt.zero_grad()
            logits = head(w.to(device), f.to(device))
            loss = crit(logits, torch.from_numpy(y).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt.step()
        head.eval()
        vp = []
        with torch.no_grad():
            for i in range(0, len(val_data), 512):
                w, f, _ = val_data.take(np.arange(i, min(i + 512, len(val_data))))
                vp.append(torch.sigmoid(head(w.to(device), f.to(device))).cpu().numpy())
        vp = np.concatenate(vp)
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


def evaluate(head, data, device, batch=256):
    head.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(data), batch):
            w, f, _ = data.take(np.arange(i, min(i + batch, len(data))))
            ps.append(torch.sigmoid(head(w.to(device), f.to(device))).cpu().numpy())
    p = np.concatenate(ps)
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    return {
        "n": int(len(data.y)), "n_pos": int(data.y.sum()),
        "roc_auc": round(float(roc_auc_score(data.y, p)), 4),
        "pr_auc": round(float(average_precision_score(data.y, p)), 4),
        "f1_05": round(float(f1_score(data.y, (p >= 0.5).astype(int))), 4),
    }


def load_vsb_checkpoint(enc_name):
    ckpt = OUT / f"vsb_ckpt_{enc_name}.pt"
    if not ckpt.exists():
        # train one on the fly: reuse the tii_external_multi_dataset path by
        # simply training here on VSB dev data.
        d = np.load(ROOT / "results" / "cached_features" / "features_full.npz", allow_pickle=False)
        wdev = np.asarray(d["windows"]).astype(np.float32).reshape(-1, 8, 8192)
        ydev = np.asarray(d["labels"], dtype=np.float32).reshape(-1)
        head = ExternalHead(enc_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        head = head.to(device)
        data = LazyData(wdev, ydev)
        head = train_head(head, data, data, device)
        torch.save(head.state_dict(), ckpt)
        return head
    head = ExternalHead(enc_name)
    head.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return head


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    summary = {}
    for task in ["pd_vs_background", "pd_vs_corona"]:
        X, y, channels, names = load(task)
        tr, va, te = group_split(names, channels)
        tr_d, va_d, te_d = LazyData(X, y), LazyData(X, y), LazyData(X, y)
        # subset by index for each split (LazyData holds full arrays; use take with indices)
        tr_d.X, tr_d.y = X[tr], y[tr]
        va_d.X, va_d.y = X[va], y[va]
        te_d.X, te_d.y = X[te], y[te]
        print(f"[{task}] train {len(tr)} val {len(va)} test {len(te)}")

        enc = "simple_cnn"  # E4 encoder
        head = load_vsb_checkpoint(enc).to(device)

        # zero-shot
        zs = evaluate(head, te_d, device)
        summary[f"28523090_{task}_zero_shot_e4"] = zs

        # from-scratch
        fs = ExternalHead(enc).to(device)
        t0 = time.time()
        fs = train_head(fs, tr_d, va_d, device)
        fsr = evaluate(fs, te_d, device)
        summary[f"28523090_{task}_from_scratch_e4"] = fsr

        # fine-tune
        ft = ExternalHead(enc).to(device)
        ft.load_state_dict(head.state_dict())
        ft = ft.to(device)
        ft = train_head(ft, tr_d, va_d, device)
        ftr = evaluate(ft, te_d, device)
        summary[f"28523090_{task}_fine_tune_e4"] = ftr
        print(f"[{task}] zero {zs} | fs {fsr} | ft {ftr}")

    (OUT / "summary_28523090_e4.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== E4 THREE-ARM ON 28523090 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("Done.")


if __name__ == "__main__":
    main()
