"""Window-policy ablation: extract windows per policy, cache features, train reference model.

Policies (locked ablation grid):
  single       : 1 equidistant window per phase (K=1)
  equidistant  : 8 equidistant windows per phase (K=8, no event windows)
  event        : 8 event windows per phase (K=8, no uniform anchors)
  mixed_k4     : 2 uniform + 2 event (K=4)
  mixed_k8     : 4 uniform + 4 event (K=8, the locked main pipeline)
  mixed_k12    : 6 uniform + 6 event (K=12)

The reference model is the full pipeline (dual-branch encoder + gated-attention MIL +
cyclic phase module + noisy-OR), trained on the SAME StratifiedGroupKFold(5, seed=42)
development folds for every policy.

Usage:
  python scripts/run_policy_ablations.py --policies mixed_k4,mixed_k12
                                         [--limit 10]           # extraction smoke test
                                         [--epochs 50] [--batch-size 64] [--patience 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow importing run_ablations

from vsb_pd.config import load_config
from vsb_pd.extract import extract_development
from vsb_pd.data import WindowDataset
from vsb_pd.features import extract_physical_features
from vsb_pd.baselines import aggregate_features_per_phase
from vsb_pd.training import make_stratified_group_folds
import run_ablations as ra

POLICIES = {
    "single": "configs/policy_single.json",
    "equidistant": "configs/policy_equidistant.json",
    "event": "configs/policy_event.json",
    "mixed_k4": "configs/policy_mixed_k4.json",
    "mixed_k8": "configs/local.json",
    "mixed_k12": "configs/policy_mixed_k12.json",
}
SPLIT_LOCK = Path("artifacts/locks/split_lock.json")
CACHE_DIR = Path("results/cached_features")
OUT_ROOT = Path("results/ablations/window_policy")


def cache_policy_features(policy: str, cfg, limit: int | None, feature_batch_size: int | None = None) -> Path:
    """Extract windows (if needed) and cache features; return npz path."""
    npz_path = CACHE_DIR / f"features_policy_{policy}.npz"
    if npz_path.exists() and limit is None:
        print(f"  cache exists: {npz_path}, skipping extraction")
        return npz_path
    print(f"\n[policy:{policy}] extracting development windows...")
    t0 = time.time()
    manifest = extract_development(
        cfg, SPLIT_LOCK, limit_measurements=limit,
    )
    print(f"  manifest: {manifest} ({time.time() - t0:.0f}s)")

    ds = WindowDataset(manifest, cfg)
    total = len(ds)
    print(f"  measurements: {total}")
    all_windows, all_labels, all_mids = [], [], []
    t1 = time.time()
    for i in range(total):
        w, s, k, sc, t, mid = ds[i]
        all_windows.append(np.asarray(w.numpy(), dtype=np.float32))
        all_labels.append(np.asarray(t.numpy(), dtype=np.int8))
        all_mids.append(mid)
        if (i + 1) % 500 == 0:
            print(f"    loaded {i+1}/{total} ({time.time() - t1:.0f}s)")
    all_windows = np.stack(all_windows)
    all_labels = np.array(all_labels, dtype=np.int8)
    all_mids = np.array(all_mids)
    print(f"  windows: {all_windows.shape} ({time.time() - t1:.0f}s)")

    M, P, K, L = all_windows.shape
    total_windows = M * P * K
    t2 = time.time()
    features = extract_physical_features(
        all_windows.reshape(M * P, K, L), cfg.sampling_rate_hz,
        batch_size=feature_batch_size,
    )
    print(f"  features extracted in {time.time() - t2:.0f}s for {total_windows:,} windows")

    X, y, groups = aggregate_features_per_phase(
        features, all_labels, all_mids, num_phases=P, num_windows=K,
    )
    feature_names = sorted(features.keys())
    feat_array = np.stack([features[name] for name in feature_names], axis=-1).reshape(M, P, K, -1)

    suffix = f"_policy_{policy}" if limit is None else f"_policy_{policy}_n{limit}"
    npz_path = CACHE_DIR / f"features{suffix}.npz"
    np.savez_compressed(
        npz_path,
        feat_array=feat_array,
        windows=all_windows,
        labels=all_labels,
        measurement_ids=all_mids,
        aggregated_X=X,
        aggregated_y=y,
        aggregated_groups=groups,
        feature_names=np.array(feature_names),
    )
    print(f"  cached: {npz_path} ({npz_path.stat().st_size / 1e6:.0f} MB)")
    return npz_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="mixed_k4,mixed_k12")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--feature-batch-size", type=int, default=4096,
                    help="Chunk size for physical-feature extraction (bounds FFT peak memory)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args()

    summaries = []
    for policy in args.policies.split(","):
        cfg_path = POLICIES[policy]
        cfg = load_config(Path(cfg_path))
        npz_path = cache_policy_features(policy, cfg, args.limit, args.feature_batch_size)

        d = np.load(npz_path, allow_pickle=True)
        feat_array = d["feat_array"]
        windows_np = d["windows"]
        labels = d["labels"].astype(np.float32)
        mids = d["measurement_ids"]
        M = len(mids)
        print(f"\n[policy:{policy}] training reference model on {M} measurements "
              f"(windows {windows_np.shape})")

        device = ra.torch.device("cuda" if ra.torch.cuda.is_available() else "cpu")
        windows = ra.torch.from_numpy(windows_np).float().to(device)
        feat = ra.torch.from_numpy(feat_array).float().to(device)
        labels_t = ra.torch.from_numpy(np.asarray(labels)).float().to(device)
        ra.labels_np = np.asarray(labels)

        label_counts = np.asarray(labels).sum(axis=1).astype(int)
        folds = make_stratified_group_folds(mids, label_counts, n_splits=5, seed=args.seed)

        out_dir = OUT_ROOT / args.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"policy_{policy}"
        summary = ra.train_config(
            name, "dual", "gated_attention", "cyclic",
            windows, feat, labels_t, mids, folds,
            args.epochs, args.batch_size, args.patience, args.seed,
            out_dir, False,
        )
        summary["policy"] = policy
        summary["window_policy"] = {
            "uniform_count": cfg.window_policy.uniform_count,
            "event_count": cfg.window_policy.event_count,
            "K": cfg.window_policy.total_count,
        }
        summaries.append(summary)
        (out_dir / f"policy_{policy}_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"{'policy':<14}{'K':<5}{'combo':<14}{'phase PR-AUC':<18}{'meas PR-AUC':<14}")
    print("-" * 90)
    for s in summaries:
        wp = s["window_policy"]
        combo = f"{wp['uniform_count']}u+{wp['event_count']}e"
        print(f"{s['policy']:<14}{wp['K']:<5}{combo:<14}"
              f"{s['mean_phase_pr_auc']:.4f}+-{s['std_phase_pr_auc']:.4f}  "
              f"{s['mean_measurement_pr_auc']:.4f}")
    (OUT_ROOT / args.tag / "policy_ablation_summary.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {OUT_ROOT / args.tag / 'policy_ablation_summary.json'}")


if __name__ == "__main__":
    main()
