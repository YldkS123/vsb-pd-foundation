# -*- coding: utf-8 -*-
"""Rebuild the combined window-policy ablation summary from per-policy files.

Each background run overwrote results/ablations/window_policy/dev/
policy_ablation_summary.json with only its own rows; this script merges all
six policies deterministically.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("results/ablations/window_policy/dev")
ORDER = ["single", "equidistant", "event", "mixed_k4", "mixed_k8", "mixed_k12"]

def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

rows = []
for policy in ORDER:
    if policy == "mixed_k8":
        # locked mainline (dual/gated/cyclic, K=8) from the dev_k8 model ablation
        summary = load(Path(
            "results/ablations/dev_k8/enc_dual__mil_gated_attention__ph_cyclic/cv_summary.json"
        ))
        row = {
            "policy": policy,
            "window_policy": {"uniform_count": 4, "event_count": 4, "K": 8},
            "mean_phase_pr_auc": summary["mean_phase_pr_auc"],
            "std_phase_pr_auc": summary["std_phase_pr_auc"],
            "mean_measurement_pr_auc": summary["mean_measurement_pr_auc"],
        }
    else:
        path = OUT_DIR / f"policy_{policy}_summary.json"
        summary = load(path)
        row = {k: summary[k] for k in summary if k != "folds"}
    rows.append(row)

combined = OUT_DIR / "policy_ablation_summary.json"
combined.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
print(f"wrote {combined} with {len(rows)} policies:")
for row in rows:
    wp = row["window_policy"]
    print(f"  {row['policy']:<14} K={wp['K']:<3} {wp['uniform_count']}u+{wp['event_count']}e  "
          f"phase={row['mean_phase_pr_auc']:.4f} +- {row['std_phase_pr_auc']:.4f}  "
          f"meas={row['mean_measurement_pr_auc']:.4f}")
