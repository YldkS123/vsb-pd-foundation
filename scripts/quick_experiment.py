"""Quick experiment: subset of 200 measurements for fast pipeline validation."""
from pathlib import Path
import numpy as np
from vsb_pd.data import WindowDataset
from vsb_pd.config import load_config
from vsb_pd.features import extract_physical_features
from vsb_pd.baselines import aggregate_features_per_phase, train_lr_baseline, train_rf_baseline, train_lgbm_baseline
import json

SUBSET = 200
SEED = 42

config = load_config(Path("configs/local.json"))
manifest = Path("artifacts/windows/a5037d5fe2560d3a3456aba07e50fb0ed9457388b568550aeab40d1bac739de1/development/manifest_full.parquet")
ds = WindowDataset(manifest, config)
print(f"Loading first {SUBSET} of {len(ds)} measurements...")

all_windows, all_labels, all_mids = [], [], []
for i in range(SUBSET):
    w, s, k, sc, t, mid = ds[i]
    all_windows.append(w.numpy())
    all_labels.append(t.numpy())
    all_mids.append(mid)

all_windows = np.stack(all_windows)
all_labels = np.array(all_labels, dtype=np.int8)
all_mids = np.array(all_mids)
print(f"Windows: {all_windows.shape}")

M, P, K, L = all_windows.shape
print(f"Extracting features ({M*P*K} windows)...")
features = extract_physical_features(all_windows.reshape(M * P, K, L), config.sampling_rate_hz)
print(f"Features: {len(features)} types")

X, y, groups = aggregate_features_per_phase(features, all_labels, all_mids, num_phases=P, num_windows=K)
print(f"X: {X.shape}, y: {y.shape}")
unique, counts = np.unique(y, return_counts=True)
print(f"Labels: {dict(zip(unique.astype(int), counts))}")

results = {}

print("\n--- Logistic Regression ---")
lr_model, lr_params = train_lr_baseline(X, y, groups, seed=SEED)
print(f"  Best params: {lr_params}")
results["lr"] = {"params": {k: str(v) for k, v in lr_params.items()}}

print("\n--- Random Forest ---")
rf_model, rf_params = train_rf_baseline(X, y, groups, seed=SEED)
print(f"  Best params: {rf_params}")
results["rf"] = {"params": {k: str(v) for k, v in rf_params.items()}}

print("\n--- LightGBM ---")
try:
    lgb_model, lgb_params = train_lgbm_baseline(X, y, groups, seed=SEED)
    print(f"  Best params: {lgb_params}")
    results["lgbm"] = {"params": {k: str(v) for k, v in lgb_params.items()}}
except Exception as e:
    print(f"  Failed: {e}")

outdir = Path("results/baselines")
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "subset200_results.json").write_text(json.dumps(results, indent=2, default=str))
print(f"\nSaved to {outdir / 'subset200_results.json'}")
