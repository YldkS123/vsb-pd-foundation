# -*- coding: utf-8 -*-
"""Multi-seed robustness check for the K=8 vs K=12 decision.

Runs the locked reference model (dual/gated-attention/cyclic/noisy-OR) for
mixed_k8 and mixed_k12 with two additional seeds (2024, 7); seed 42 runs
already exist under results/ablations/window_policy/dev/. Each run writes to
its own tag dir (results/ablations/window_policy/seed{seed}/) so the combined
dev summary is not overwritten.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

RUNS = [
    ("mixed_k8", 2024),
    ("mixed_k8", 7),
    ("mixed_k12", 2024),
    ("mixed_k12", 7),
]

LOG = ROOT / "results" / "multi_seed_run.log"


def main() -> None:
    base = [
        PY, "-u", "scripts/run_policy_ablations.py",
        "--epochs", "50", "--batch-size", "64", "--patience", "20",
    ]
    with LOG.open("a", encoding="utf-8") as handle:
        for policy, seed in RUNS:
            tag = f"seed{seed}"
            cmd = base + ["--policies", policy, "--seed", str(seed), "--tag", tag]
            handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                         f"{policy} seed={seed} start ===\n")
            handle.flush()
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=handle,
                                  stderr=subprocess.STDOUT)
            handle.write(f"=== {policy} seed={seed} exit={proc.returncode} "
                         f"at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            handle.flush()
    print("multi-seed driver finished")


if __name__ == "__main__":
    main()
