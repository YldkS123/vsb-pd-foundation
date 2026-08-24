"""Pre-compute aggregated features for baseline training."""
from pathlib import Path
import numpy as np
from vsb_pd.data import WindowDataset
from vsb_pd.config import load_config
from vsb_pd.features import extract_physical_features
from vsb_pd.baselines import aggregate_features_per_phase

config = load_config(Path("configs/local.json"))
manifest = Path("artifacts/windows/a5037d5fe2560d3a3456aba07e50fb0ed9457388b568550aeab40d1bac739de1/development/manifest_full.parquet")
ds = WindowDataset(manifest, config)
print(f"Loading {len(ds)} measurements...")

all_windows, all_labels, all_mids = [], [], []
for i in range(len(ds)):
    w, s, k, sc, t, mid = ds[i]
    all_windows.append(w.numpy())
    all_labels.append(t.numpy())
    all_mids.append(mid)
    if (i + 1) % 500 == 0:
        print(f"  Loaded {i+1}/{len(ds)}")

all_windows = np.stack(all_windows)
all_labels = np.array(all_labels, dtype=np.int8)
all_mids = np.array(all_mids)
print(f"Windows: {all_windows.shape}, Labels: {all_labels.shape}")

M, P, K, L = all_windows.shape
print(f"Extracting physical features from {M*P*K} windows...")
features = extract_physical_features(all_windows.reshape(M * P, K, L), config.sampling_rate_hz)
print(f"Extracted {len(features)} feature types, {len(list(features.values())[0]):,} values each")

print("Aggregating to per-phase features...")
X, y, groups = aggregate_features_per_phase(features, all_labels, all_mids, num_phases=P, num_windows=K)
print(f"X: {X.shape}, y: {y.shape}, groups: {groups.shape}")

outdir = Path("results/baselines")
outdir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(outdir / "aggregated_features.npz", X=X, y=y, groups=groups)
print(f"Saved to {outdir / 'aggregated_features.npz'}")

unique, counts = np.unique(y, return_counts=True)
print(f"Label distribution: {dict(zip(unique.astype(int), counts))}")
print("Done!")
