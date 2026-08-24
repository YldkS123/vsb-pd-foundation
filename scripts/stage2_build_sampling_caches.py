# -*- coding: utf-8 -*-
"""Build Stage 2 sampling-controlled caches from the locked development set.

Produces NPZ caches that share the exact measurement order (and therefore the
exact StratifiedGroupKFold indices) of results/cached_features/features_policy_mixed_k8.npz.

Policies:
  uniform_k8 : 8 equidistant anchors (WindowPolicy 8 uniform / 0 event)
  event_k8   : 8 event-centered windows (WindowPolicy 0 uniform / 8 event)
  random_k8  : 8 deterministic random, non-overlapping starts (documented RNG)
  full_signal: the whole 800k-point raw signal (K=1)

The frozen blind set is never read.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from vsb_pd.config import WindowPolicy
from vsb_pd.events import select_hybrid_windows

ROOT = Path(__file__).resolve().parent.parent
DEV_ARTIFACTS = (
    ROOT / "artifacts" / "windows"
    / "a5037d5fe2560d3a3456aba07e50fb0ed9457388b568550aeab40d1bac739de1" / "development"
)
MIXED_CACHE = ROOT / "results" / "cached_features" / "features_policy_mixed_k8.npz"
RAW_PARQUET = ROOT.parent / "1111" / "train.parquet"
OUT_DIR = ROOT / "results" / "cached_features"

WINDOW_LENGTH = 8192
SIGNAL_LENGTH = 800_000
RANDOM_SEED = 2026
RANDOM_MIN_GAP = WINDOW_LENGTH // 2


def load_measurement_order() -> tuple[np.ndarray, np.ndarray]:
    d = np.load(MIXED_CACHE, allow_pickle=False)
    mids = np.asarray(d["measurement_ids"])
    labels = np.asarray(d["labels"])
    return mids, labels


def load_signal_id_map() -> dict[int, list[int]]:
    mapping = {}
    for artifact in sorted(DEV_ARTIFACTS.glob("*.npz")):
        d = np.load(artifact, allow_pickle=False)
        mapping[int(d["measurement_id"].item())] = d["signal_ids"].astype(int).tolist()
    return mapping


def random_window_starts(signal_length: int, window_length: int, count: int,
                         rng: np.random.Generator) -> list[int]:
    """Deterministic random starts with at least window_length/2 spacing."""
    max_start = signal_length - window_length
    starts: list[int] = []
    for _ in range(count):
        for _ in range(10_000):
            candidate = int(rng.integers(0, max_start + 1))
            if all(abs(candidate - old) >= RANDOM_MIN_GAP for old in starts):
                starts.append(candidate)
                break
        else:
            raise RuntimeError("cannot place random non-overlapping windows")
    return sorted(starts)


def build_policy_windows(signals: np.ndarray, policy_name: str,
                         rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    """Return (windows (P,K,L), per-phase window metadata)."""
    phases = []
    meta_rows = []
    for phase in range(3):
        x = signals[phase]
        if policy_name == "uniform_k8":
            policy = WindowPolicy(WINDOW_LENGTH, 8, 0, 0.5, 256)
            selected = select_hybrid_windows(x, policy)
        elif policy_name == "event_k8":
            policy = WindowPolicy(WINDOW_LENGTH, 0, 8, 0.5, 256)
            selected = select_hybrid_windows(x, policy)
        elif policy_name == "random_k8":
            starts = random_window_starts(SIGNAL_LENGTH, WINDOW_LENGTH, 8, rng)
            selected = [{"start": start, "kind": "random", "score": 0.0}
                        for start in starts]
        else:
            raise ValueError(f"unknown window policy: {policy_name}")
        window_parts = []
        selected_meta = []
        for item in selected:
            start = item.start if hasattr(item, "start") else item["start"]
            kind = item.kind if hasattr(item, "kind") else item["kind"]
            score = item.score if hasattr(item, "score") else float(item["score"])
            window_parts.append(x[start:start + WINDOW_LENGTH])
            selected_meta.append({"start": int(start), "kind": kind, "score": float(score)})
        windows = np.stack(window_parts).astype(np.float32)
        phases.append(windows)
        meta_rows.append(selected_meta)
    return np.stack(phases), meta_rows


def read_raw_measurements(parquet_path: Path, rows: list[tuple[int, list[int]]],
                          batch_measurements: int = 12):
    """Yield (measurement_index, signals (3, L)) for each row in order."""
    for start in range(0, len(rows), batch_measurements):
        chunk = rows[start:start + batch_measurements]
        names = [str(sid) for _, sids in chunk for sid in sids]
        table = pq.ParquetFile(parquet_path).read(columns=names)
        arrays = [table[name].to_numpy(zero_copy_only=False) for name in names]
        for j, (midx, _) in enumerate(chunk):
            signals = np.stack(arrays[j * 3:(j + 1) * 3], axis=0)
            yield midx, signals
        del table, arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="uniform_k8,event_k8,random_k8,full_signal")
    ap.add_argument("--batch-measurements", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="smoke limit")
    ap.add_argument("--no-compress", action="store_true")
    args = ap.parse_args()

    if not RAW_PARQUET.exists():
        raise FileNotFoundError(RAW_PARQUET)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    mids, labels = load_measurement_order()
    if args.limit:
        mids, labels = mids[:args.limit], labels[:args.limit]
    signal_map = load_signal_id_map()
    missing = [int(m) for m in mids if int(m) not in signal_map]
    if missing:
        raise ValueError(f"missing dev artifacts for measurements: {missing[:5]}")
    rows = [(i, signal_map[int(mid)]) for i, mid in enumerate(mids)]

    print(f"measurements={len(mids)} policies={policies}")
    for policy in policies:
        if policy not in ("uniform_k8", "event_k8", "random_k8", "full_signal"):
            raise ValueError(f"unknown policy: {policy}")

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    metadata = {
        "measurement_source": "results/cached_features/features_policy_mixed_k8.npz",
        "dev_artifact_hash": DEV_ARTIFACTS.parent.name,
        "raw_parquet_path": str(RAW_PARQUET),
        "window_length": WINDOW_LENGTH,
        "signal_length": SIGNAL_LENGTH,
        "random_seed": RANDOM_SEED,
        "random_min_gap": RANDOM_MIN_GAP,
        "policies": {},
    }

    t_all = time.perf_counter()
    for policy in policies:
        t0 = time.perf_counter()
        if policy == "full_signal":
            shape = (len(mids), 3, 1, SIGNAL_LENGTH)
            windows_all = np.empty(shape, dtype=np.int8)
        else:
            shape = (len(mids), 3, 8, WINDOW_LENGTH)
            windows_all = np.empty(shape, dtype=np.float32)
        feat_all = np.zeros((len(mids), 3, shape[2], 58), dtype=np.float32)
        rows_done = 0
        for midx, signals in read_raw_measurements(RAW_PARQUET, rows, args.batch_measurements):
            if signals.shape != (3, SIGNAL_LENGTH):
                raise ValueError(f"measurement row {midx} shape {signals.shape}")
            if policy == "full_signal":
                windows_all[midx, :, 0, :] = signals
            else:
                windows_all[midx], _ = build_policy_windows(signals, policy, rng)
            rows_done += 1
            if rows_done % 200 == 0:
                print(f"  {policy}: {rows_done}/{len(mids)} "
                      f"({time.perf_counter() - t0:.0f}s)")
        suffix = "_raw" if args.no_compress else ""
        out_path = out_dir / f"features_policy_{policy}{suffix}.npz"
        if args.no_compress:
            np.savez(out_path, windows=windows_all, feat_array=feat_all,
                     labels=labels, measurement_ids=mids)
        else:
            np.savez_compressed(out_path, windows=windows_all, feat_array=feat_all,
                                labels=labels, measurement_ids=mids)
        elapsed = time.perf_counter() - t0
        metadata["policies"][policy] = {
            "cache": str(out_path),
            "windows_shape": list(windows_all.shape),
            "dtype": str(windows_all.dtype),
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  {policy}: saved {out_path} ({windows_all.shape}) in {elapsed:.0f}s")
        del windows_all, feat_all

    metadata["total_elapsed_s"] = round(time.perf_counter() - t_all, 1)
    meta_path = out_dir / "stage2_policies.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
