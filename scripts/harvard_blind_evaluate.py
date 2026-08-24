# -*- coding: utf-8 -*-
"""One-time blind evaluation of E4 on the frozen Harvard dataset chunk.

Reads the frozen protocol lock (results/blind_lock_E4_harvard.json), the
regenerated E4 development-fold checkpoints, and the downloaded Harvard
archive part1_1_A.  Extracts the archive once to D:\\harvard_blind\\signals,
extracts the locked mixed K=8 windows per measurement, runs the 5-fold
ensemble exactly once, and writes a tamper-evident evaluation receipt.

The receipt path is guarded: a second run is refused once the receipt exists.
No model selection, threshold tuning, or re-run is allowed after metrics are
seen.

Usage:
  python scripts/harvard_blind_evaluate.py --extract-only
  python scripts/harvard_blind_evaluate.py --smoke
  python scripts/harvard_blind_evaluate.py [--resume]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vsb_pd.config import WindowPolicy  # noqa: E402
from vsb_pd.cyclic import PhaseInteractionModule  # noqa: E402
from vsb_pd.dl_encoders import TimWindowEncoder  # noqa: E402
from vsb_pd.events import select_hybrid_windows  # noqa: E402
from vsb_pd.mil import MILAggregator, PhaseClassifier  # noqa: E402
from vsb_pd.model import VSBPipeline  # noqa: E402
from vsb_pd.evaluation import compute_bootstrap_ci, compute_calibration_metrics  # noqa: E402

LOCK_PATH = ROOT / "results" / "blind_lock_E4_harvard.json"
RECEIPT_PATH = ROOT / "results" / "FINAL_EVALUATION_RECEIPT_E4_harvard.json"
RUN_LOCK_PATH = ROOT / "results" / "final_eval_E4_harvard.lock"
CKPT_DIR = ROOT / "results" / "e4_harvard_blind" / "checkpoints"
PREDS_PATH = ROOT / "results" / "e4_harvard_blind" / "predictions.npz"
ARCHIVE = Path(r"C:\Users\hrfxgfx\Downloads\part1_1_A.tar.gz")
MEMBER_LIST = Path(r"C:\Users\hrfxgfx\Downloads\part1_1_A_members.txt")
FEATURE_CSV = Path(r"C:\Users\hrfxgfx\Downloads\feature_vector.csv")
TRAIN_SET = Path(r"C:\Users\hrfxgfx\Downloads\train_set.tab")
SIGNAL_ROOT = Path(r"D:\harvard_blind\signals")
EXTRACT_DONE = SIGNAL_ROOT / "extract_done.json"
POLICY = WindowPolicy(8192, 4, 4, 0.5, 256)
N_FOLDS = 5
EXPECTED_PARAMS = 113265
CONFIG_NAME = "enc_cnn__mil_attention__ph_ctx_concat"
BATCH = 64
N_WORKERS = 8
PRIMARY_POSITIVE = {1, 2, 3, 4, 5, 6}
CONTACT_POSITIVE = {1, 2, 5, 6}
PHASE_THRESHOLD = 0.5
MEASUREMENT_THRESHOLD = 0.88


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_member_list() -> list[tuple[int, int]]:
    pat = re.compile(r"part1_1_A/(\d+)_([123])\.bin\.gz$")
    keys: set[tuple[int, int]] = set()
    with open(MEMBER_LIST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = pat.match(line)
            if not m:
                raise SystemExit(f"unexpected member name: {line}")
            keys.add((int(m.group(1)), int(m.group(2))))
    return sorted(keys)


def load_annotations() -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    with open(FEATURE_CSV, encoding="utf-8", errors="replace") as f:
        r = csv.reader(f, delimiter=";")
        next(r)
        for row in r:
            if len(row) < 3:
                continue
            out[(int(row[0]), int(row[1]))] = int(row[2])
    return out


def load_train_set() -> set[tuple[int, int]]:
    q = chr(34)
    out: set[tuple[int, int]] = set()
    with open(TRAIN_SET, encoding="utf-8", errors="replace") as f:
        r = csv.reader(f, delimiter=";", quotechar=q)
        next(r)
        for row in r:
            if len(row) == 1 and ";" in row[0]:
                parts = row[0].split(";")
                mid, ph = parts[0].strip(q), parts[1].strip(q)
            elif len(row) >= 2:
                mid, ph = row[0].strip(q), row[1].strip(q)
            else:
                continue
            out.add((int(mid), int(ph) + 1))
    return out


def load_signal(path: Path) -> np.ndarray:
    raw = gzip.open(path, "rb").read()
    if len(raw) != 800_000:
        raise SystemExit(f"unexpected signal size {len(raw)}: {path}")
    return np.frombuffer(raw, dtype=np.int8).astype(np.float32)


def build_measurement_windows(mid: int, signal_root: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Return (mid, mask(3,), windows(3,8,8192) float32) for one measurement."""
    windows = np.zeros((3, 8, 8192), dtype=np.float32)
    mask = np.zeros(3, dtype=bool)
    for ph in (1, 2, 3):
        path = signal_root / "part1_1_A" / f"{mid}_{ph}.bin.gz"
        if not path.exists():
            continue
        x = load_signal(path)
        selected = select_hybrid_windows(x, POLICY)
        if len(selected) != 8:
            raise SystemExit(f"unexpected window count {len(selected)} for {path}")
        idx = ph - 1
        windows[idx] = np.stack([x[c.start:c.start + 8192] for c in selected])
        mask[idx] = True
    return mid, mask, windows


def extract_archive() -> None:
    if EXTRACT_DONE.exists():
        done = json.loads(EXTRACT_DONE.read_text(encoding="utf-8"))
        if done.get("member_count") == parse_member_list().__len__():
            print("archive already extracted")
            return
    SIGNAL_ROOT.mkdir(parents=True, exist_ok=True)
    member_keys = parse_member_list()
    target = SIGNAL_ROOT / "part1_1_A"
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    t0 = time.time()
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = Path(member.name).name
            out = target / name
            if out.exists() and out.stat().st_size == member.size:
                count += 1
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            with open(out, "wb") as f:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            count += 1
            if count % 10_000 == 0:
                print(f"  extracted {count}/{len(member_keys)} "
                      f"({time.time() - t0:.0f}s)")
    if count != len(member_keys):
        raise SystemExit(f"extracted {count} members, expected {len(member_keys)}")
    EXTRACT_DONE.write_text(
        json.dumps({"member_count": count, "date": "2026-08-18"}),
        encoding="utf-8")
    print(f"extraction complete: {count} members in {time.time() - t0:.0f}s")


def build_pipeline() -> VSBPipeline:
    encoder = TimWindowEncoder("cnn", 8192, 58, 128)
    aggregator = MILAggregator("attention", 128)
    cyclic = PhaseInteractionModule("context_concat", 128)
    classifier = PhaseClassifier(128)
    return VSBPipeline(
        encoder=encoder, aggregator=aggregator, cyclic=cyclic,
        classifier=classifier, max_encode_chunk=8, checkpoint_chunks=True,
    )


def load_models(device: torch.device) -> list[tuple[torch.nn.Module, str, int]]:
    paths = sorted(CKPT_DIR.glob("model_fold*.pt"))
    if len(paths) != N_FOLDS:
        raise SystemExit(f"expected {N_FOLDS} checkpoints in {CKPT_DIR}, got {len(paths)}")
    models = []
    for p in paths:
        model = build_pipeline().to(device)
        ckpt = torch.load(p, map_location=device, weights_only=True)
        if ckpt.get("config") != CONFIG_NAME:
            raise SystemExit(f"unexpected checkpoint config: {p}")
        if int(ckpt.get("n_params", -1)) != EXPECTED_PARAMS:
            raise SystemExit(f"unexpected n_params in {p}")
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
        if missing or unexpected:
            raise SystemExit(f"state_dict mismatch in {p}: {missing} / {unexpected}")
        models.append((model, str(p), int(ckpt.get("fold", 0))))
    return models


@torch.no_grad()
def predict_batch(models, windows, feat, mask, device) -> np.ndarray:
    windows = torch.from_numpy(windows).float().to(device)
    feat = torch.from_numpy(feat).float().to(device)
    mask_t = torch.from_numpy(mask).bool().to(device)
    probs = None
    for model, _, _ in models:
        model.eval()
        logits, _ = model(windows, feat, phase_mask=mask_t)
        p = torch.sigmoid(logits).cpu().numpy()
        probs = p if probs is None else probs + p
    return probs / len(models)


def save_partial(records: dict, path: Path) -> None:
    np.savez_compressed(path, **records)


def compute_classification_metrics(labels: np.ndarray, scores: np.ndarray,
                                   threshold: float) -> dict:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score,
        matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    )

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    preds = (scores >= threshold).astype(int)
    metrics = {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }
    metrics.update(compute_calibration_metrics(scores, labels))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke-mids", default="491873,713541,429192")
    args = ap.parse_args()

    if args.extract_only:
        extract_archive()
        return 0

    if not LOCK_PATH.exists():
        raise SystemExit(f"protocol lock missing: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("created_before_evaluation") is not True:
        raise SystemExit("lock does not assert created_before_evaluation")

    if RECEIPT_PATH.exists() and not args.smoke:
        raise SystemExit(
            f"one-time receipt already exists: {RECEIPT_PATH} "
            "(a second blind evaluation is refused)"
        )

    members = parse_member_list()
    members_set = set(members)
    ann = load_annotations()
    train = load_train_set()
    blind = members_set - train
    if len(blind) != lock["blind_set"]["n_phase_signals"]:
        raise SystemExit("blind set size does not match the frozen lock")

    device = torch.device(args.device)
    if not args.smoke and RUN_LOCK_PATH.exists():
        raise SystemExit(f"run lock already exists: {RUN_LOCK_PATH}")

    if args.smoke:
        mid_set = {int(x) for x in args.smoke_mids.split(",") if x.strip()}
        for mid in sorted(mid_set):
            _, mask, windows = build_measurement_windows(mid, SIGNAL_ROOT)
            print(f"smoke mid={mid} mask={mask.astype(int).tolist()} "
                  f"windows={windows.shape} finite={np.isfinite(windows).all()}")
        return 0

    extract_archive()
    models = load_models(device)
    print(f"loaded {len(models)} fold checkpoints, device={device}")

    # Group by measurement, present phases only.
    by_mid: dict[int, list[int]] = {}
    for mid, ph in members:
        by_mid.setdefault(mid, []).append(ph)
    mid_order = sorted(by_mid)

    records = {
        "measurement_ids": [],
        "phases": [],
        "probs": [],
        "annotations": [],
        "in_train_set": [],
    }

    # Resume support.
    if args.resume and PREDS_PATH.exists():
        d = np.load(PREDS_PATH, allow_pickle=False)
        records = {
            "measurement_ids": d["measurement_ids"].tolist(),
            "phases": d["phases"].tolist(),
            "probs": d["probs"].tolist(),
            "annotations": d["annotations"].tolist(),
            "in_train_set": d["in_train_set"].tolist(),
        }
        done_mids = set(zip(records["measurement_ids"], records["phases"]))
        print(f"resume: {len(done_mids)} phase predictions already recorded")
    else:
        done_mids = set()

    t_all = time.time()
    window_buf: list[tuple[int, np.ndarray, np.ndarray]] = []
    remaining = []
    for mid in mid_order:
        for ph in by_mid[mid]:
            if (mid, ph) in done_mids:
                continue
            remaining.append((mid, ph))
    n_remaining_mids = len({mid for mid, _ in remaining})
    print(f"remaining measurements to process: {n_remaining_mids}")

    def emit_batch(buf):
        mids = [b[0] for b in buf]
        masks = np.stack([b[1] for b in buf])
        wins = np.stack([b[2] for b in buf])
        feat = np.zeros((len(buf), 3, 8, 58), dtype=np.float32)
        probs = predict_batch(models, wins, feat, masks, device)
        for j, mid in enumerate(mids):
            for ph in by_mid[mid]:
                if ph == 0:
                    continue
                records["measurement_ids"].append(mid)
                records["phases"].append(ph)
                records["probs"].append(float(probs[j, ph - 1]))
                records["annotations"].append(int(ann[(mid, ph)]))
                records["in_train_set"].append(1 if (mid, ph) in train else 0)
        # Free batch memory explicitly.
        del wins, feat, masks

    processed = 0

    def drain_ready():
        nonlocal processed
        while len(window_buf) >= BATCH:
            emit_batch(window_buf[:BATCH])
            del window_buf[:BATCH]
            processed += BATCH
            if processed % 2000 == 0:
                save_partial({k: np.asarray(v) for k, v in records.items()}, PREDS_PATH)
                print(f"  processed ~{processed}/{n_remaining_mids} measurements "
                      f"({time.time() - t_all:.0f}s)")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        pending = {}
        for mid in mid_order:
            if all((mid, ph) in done_mids for ph in by_mid[mid]):
                continue
            f = ex.submit(build_measurement_windows, mid, SIGNAL_ROOT)
            pending[f] = mid
            if len(pending) >= N_WORKERS * 2:
                done_futs, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for fdone in done_futs:
                    window_buf.append(fdone.result())
                    del pending[fdone]
                drain_ready()
        while pending:
            done_futs, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fdone in done_futs:
                window_buf.append(fdone.result())
                del pending[fdone]
            drain_ready()
    if window_buf:
        emit_batch(window_buf)
        processed += len(window_buf)
        window_buf.clear()

    print(f"prediction pass done ({time.time() - t_all:.0f}s), "
          f"phase rows={len(records['measurement_ids'])}")

    arr = {k: np.asarray(v) for k, v in records.items()}
    save_partial(arr, PREDS_PATH)

    # Metrics on the frozen blind set (exclude official train-set rows).
    mids_np = arr["measurement_ids"]
    phases_np = arr["phases"]
    probs_np = arr["probs"]
    ann_np = arr["annotations"]
    train_np = arr["in_train_set"].astype(bool)
    blind_sel = ~train_np
    print(f"blind phase rows={blind_sel.sum()}, "
          f"train-set rows excluded={int(train_np.sum())}")

    b_mids = mids_np[blind_sel]
    b_phases = phases_np[blind_sel]
    b_probs = probs_np[blind_sel]
    b_ann = ann_np[blind_sel]
    b_y_any = np.isin(b_ann, sorted(PRIMARY_POSITIVE)).astype(int)
    b_y_contact = np.isin(b_ann, sorted(CONTACT_POSITIVE)).astype(int)

    phase_metrics = compute_classification_metrics(b_y_any, b_probs, PHASE_THRESHOLD)
    phase_contact = compute_classification_metrics(b_y_contact, b_probs, PHASE_THRESHOLD)

    # Complete-triple measurement level.
    mid_counts = Counter(b_mids)
    tri_mids = sorted(mid for mid, c in mid_counts.items() if c == 3)
    meas_rows = []
    for mid in tri_mids:
        sel = b_mids == mid
        ps = b_probs[sel]
        ys = b_y_any[sel]
        if len(ps) != 3:
            continue
        meas_rows.append((mid, 1.0 - np.prod(1.0 - ps), int(ys.max())))
    meas_mids = np.array([r[0] for r in meas_rows], dtype=np.int64)
    meas_probs = np.array([r[1] for r in meas_rows], dtype=float)
    meas_y = np.array([r[2] for r in meas_rows], dtype=int)
    meas_metrics = compute_classification_metrics(meas_y, meas_probs, MEASUREMENT_THRESHOLD)

    # Measurement-clustered bootstrap CIs.
    b_ids_uniq = np.array([int(m) for m in b_mids])
    phase_ci = compute_bootstrap_ci(b_probs, b_y_any, b_ids_uniq, "pr_auc", 2000, 42)
    meas_ci = compute_bootstrap_ci(meas_probs, meas_y, meas_mids, "pr_auc", 2000, 42)

    pred_hash = hashlib.sha256(PREDS_PATH.read_bytes()).hexdigest()
    lock_hash = sha256_file(LOCK_PATH)
    ckpt_hashes = [
        {"path": str(p), "sha256": sha256_file(p)}
        for p in sorted(CKPT_DIR.glob("model_fold*.pt"))
    ]
    receipt = {
        "experiment": "E4 one-time blind test on Harvard Dataverse VSB chunk "
                      "part1_1_A (never-used data)",
        "date": "2026-08-18",
        "protocol_lock": {
            "path": str(LOCK_PATH),
            "sha256": lock_hash,
        },
        "model": {
            "name": "E4 final mainline",
            "n_params": EXPECTED_PARAMS,
            "ensemble": "mean of 5 development-fold checkpoints (seed 42)",
            "fold_checkpoints": ckpt_hashes,
        },
        "data": {
            "archive": str(ARCHIVE),
            "archive_sha256": lock["data"]["archive"]["sha256"],
            "member_list_sha256": lock["data"]["archive"]["member_list_sha256"],
            "feature_vector_sha256": lock["data"]["feature_vector"]["sha256"],
            "train_set_sha256": lock["data"]["train_set"]["sha256"],
        },
        "blind_set": {
            "n_phase_signals": int(blind_sel.sum()),
            "n_positive_phases_annotation_gt_0": int(b_y_any.sum()),
            "n_contact_positive_phases": int(b_y_contact.sum()),
            "n_complete_triple_measurements": len(meas_rows),
            "n_positive_complete_triple_measurements": int(meas_y.sum()),
        },
        "thresholds": {
            "phase": PHASE_THRESHOLD,
            "measurement": MEASUREMENT_THRESHOLD,
            "source": "development OOF, frozen before this evaluation",
        },
        "metrics": {
            "phase_level_annotation_gt_0": phase_metrics,
            "phase_level_contact_1_2_5_6": phase_contact,
            "measurement_level_complete_triples": meas_metrics,
        },
        "bootstrap_95_ci": {
            "phase_pr_auc": {
                "lower": phase_ci.lower,
                "median": phase_ci.median,
                "upper": phase_ci.upper,
            },
            "measurement_pr_auc": {
                "lower": meas_ci.lower,
                "median": meas_ci.median,
                "upper": meas_ci.upper,
            },
        },
        "predictions": {
            "path": str(PREDS_PATH),
            "sha256": pred_hash,
            "rows": int(len(mids_np)),
        },
        "one_time": "run once; no model selection or threshold tuning after "
                    "seeing blind metrics",
        "status": "completed",
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    RUN_LOCK_PATH.write_text(
        json.dumps({
            "acquired": True,
            "protocol_lock_sha256": lock_hash,
            "receipt": str(RECEIPT_PATH),
            "predictions_sha256": pred_hash,
        }, indent=2), encoding="utf-8")
    print(json.dumps(receipt["metrics"], indent=2, ensure_ascii=False))
    print(f"receipt written: {RECEIPT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
