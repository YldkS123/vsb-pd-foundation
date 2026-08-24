# -*- coding: utf-8 -*-
"""Queue the 80k mainline K=12 experiment after the current GPU queue exits.

The final lightweight mainline (cnn + attention MIL + mean phase, 80,113
params) is locked at K=8.  The reference model gains +0.047 phase PR-AUC from
K=8 to K=12; this runs the same K=12 protocol for the lightweight mainline on
the frozen dev folds (StratifiedGroupKFold 5, seed 42) as development evidence.

Usage:
    python scripts/queue_mainline_k12.py --wait-pid 24020
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def process_exists(pid: int) -> bool:
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"],
        capture_output=True, timeout=30,
    )
    return proc.returncode == 0


def wait_pid(pid: int, poll_sec: int = 30) -> None:
    print(f"waiting for PID {pid} to exit ...", flush=True)
    while process_exists(pid):
        time.sleep(poll_sec)
    print(f"PID {pid} exited", flush=True)


def run_k12() -> None:
    cmd = [
        sys.executable, "-u", "scripts/run_ablations.py",
        "--cache", "results/cached_features/features_policy_mixed_k12.npz",
        "--encoders", "cnn",
        "--mils", "attention",
        "--phases", "mean",
        "--epochs", "40",
        "--batch-size", "64",
        "--patience", "15",
        "--seed", "42",
        "--tag", "k12_mainline",
        "--out-dir", "results/ablations",
    ]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"K=12 mainline run failed with exit code {proc.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True)
    args = ap.parse_args()

    wait_pid(args.wait_pid)
    run_k12()
    print("K=12 mainline queue finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
