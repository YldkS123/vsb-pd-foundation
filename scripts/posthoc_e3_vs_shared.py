# -*- coding: utf-8 -*-
"""Post-hoc paired cluster bootstrap: E3 vs E2 / E3 vs E5 / E2 vs E5.

Protocol-safe post-hoc analysis on development OOF predictions only
(results/stage1_tim/<config>/oof.npz, seed 42). Same methodology as the
locked Stage-1 report (paired_bootstrap_ci, 2000 resamples, seed 42,
measurement-clustered, difference = former - latter). Does not touch the
423 blind set or the Harvard held-out set and changes no locked numbers.

Usage:
  python scripts/posthoc_e3_vs_shared.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from vsb_pd.evaluation import paired_bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "results" / "stage1_tim"


def load_oof(config: str, seed: int = 42) -> dict:
    path = STAGE / config / "oof.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=False)
    mids = d["measurement_ids"]
    return {
        "phase_probs": d[f"seed_{seed}_phase_probs"].reshape(-1),
        "phase_targets": d[f"seed_{seed}_phase_targets"].reshape(-1),
        "meas_probs": d[f"seed_{seed}_meas_probs"],
        "meas_targets": d[f"seed_{seed}_meas_targets"],
        "phase_mids": np.repeat(mids, 3),
        "meas_mids": mids,
    }


def paired(a: str, b: str, level: str, seed: int = 42) -> dict:
    A = load_oof(a, seed)
    B = load_oof(b, seed)
    return paired_bootstrap_ci(
        model_scores=A[f"{level}_probs"],
        baseline_scores=B[f"{level}_probs"],
        labels=A[f"{level}_targets"],
        measurement_ids=A[f"{level}_mids"],
        n_bootstrap=2000,
        seed=42,
    )


def fmt(r: dict) -> str:
    return f"{r['diff_median']:+.3f} [{r['diff_lower']:+.3f}, {r['diff_upper']:+.3f}]"


def main() -> None:
    contrasts = [
        ("e3_ctx_max", "e2_ctx_mean", "E3 vs E2"),
        ("e3_ctx_max", "e5_ctx_add", "E3 vs E5"),
        ("e2_ctx_mean", "e5_ctx_add", "E2 vs E5"),
        ("e3_ctx_max", "e1_ctx_none", "E3 vs E1"),
        ("e4_ctx_concat", "e1_ctx_none", "E4 vs E1"),
    ]
    out = {}
    print(f"{'Contrast':<12} {'Level':<7} {'PR-AUC diff (95% CI)'}")
    print("-" * 48)
    for a, b, name in contrasts:
        for level in ("phase", "meas"):
            r = paired(a, b, level)
            out[f"{name} {level}"] = r
            print(f"{name:<12} {level:<7} {fmt(r)}")
    (STAGE / "posthoc_e3_vs_shared.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSaved: results/stage1_tim/posthoc_e3_vs_shared.json")


if __name__ == "__main__":
    main()
