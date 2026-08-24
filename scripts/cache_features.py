"""
Cache feature extraction results to disk so training doesn't recompute them.
Run once before training.

Usage:
    python scripts/cache_features.py                     # all 2481
    python scripts/cache_features.py --subset 500        # subset
"""
import time
import sys
from pathlib import Path
import numpy as np
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vsb_pd.config import load_config
from vsb_pd.data import WindowDataset
from vsb_pd.features import extract_physical_features
from vsb_pd.baselines import aggregate_features_per_phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None, help="Only process N measurements")
    args = parser.parse_args()

    config = load_config(Path("configs/local.json"))
    manifest = Path(
        "artifacts/windows/a5037d5fe2560d3a3456aba07e50fb0ed9457388b568550aeab40d1bac739de1/"
        "development/manifest_full.parquet"
    )
    ds = WindowDataset(manifest, config)
    total = len(ds)
    n = min(args.subset, total) if args.subset else total

    outdir = Path("results/cached_features")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {n} of {total} measurements...")
    all_windows, all_labels, all_mids = [], [], []
    for i in range(n):
        w, s, k, sc, t, mid = ds[i]
        all_windows.append(np.asarray(w.numpy(), dtype=np.float32))
        all_labels.append(np.asarray(t.numpy(), dtype=np.int8))
        all_mids.append(mid)
        if (i + 1) % 200 == 0:
            print(f"  Loaded {i+1}/{n}")

    all_windows = np.stack(all_windows)
    all_labels = np.array(all_labels, dtype=np.int8)
    all_mids = np.array(all_mids)

    M, P, K, L = all_windows.shape
    total_windows = M * P * K
    print(f"Extracting features from {total_windows:,} windows ({M} measurements)...")
    print(f"  Estimated time: {total_windows * 0.44 / 24 / 60:.0f} min")

    t0 = time.time()
    features = extract_physical_features(all_windows.reshape(M * P, K, L), config.sampling_rate_hz)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print("Aggregating per-phase features...")
    X, y, groups = aggregate_features_per_phase(
        features, all_labels, all_mids, num_phases=P, num_windows=K
    )
    print(f"  X: {X.shape}, y: {y.shape}, groups: {groups.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Label distribution: {dict(zip(unique.astype(int), counts))}")

    # Save per-window features for model training
    feature_names = sorted(features.keys())
    feat_array = np.stack([features[name] for name in feature_names], axis=-1)  # (M*P*K, 58)
    feat_array = feat_array.reshape(M, P, K, -1)  # (M, 3, 8, 58)

    suffix = f"_n{n}" if args.subset else "_full"
    npz_path = outdir / f"features{suffix}.npz"
    np.savez_compressed(
        npz_path,
        feat_array=feat_array,
        windows=all_windows,
        labels=all_labels,
        measurement_ids=all_mids,
        aggregated_X=X,
        aggregated_y=y,
        aggregated_groups=groups,
        feature_names=feature_names,
    )
    print(f"Saved: {npz_path} ({npz_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
