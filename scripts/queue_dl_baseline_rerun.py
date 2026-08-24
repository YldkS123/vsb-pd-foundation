# -*- coding: utf-8 -*-
"""Queue a deep-baseline / control run after the current GPU job exits.

Waits for a running PID (e.g. the current dl_baselines Inception job), then
launches run_dl_baselines.py for the requested encoders with the requested
hyperparameters.  Used to add the lightweight simple_cnn control at the same
batch-size/epochs protocol as the deep encoders without rerunning batch=64.

Usage:
    python scripts/queue_dl_baseline_rerun.py --wait-pid 26880 \
        --encoders simple_cnn --epochs 20 --batch-size 16 --patience 15 \
        --tag dl_baselines --out-dir results/dl_baselines
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


def run_rerun(encoders: list[str], epochs: int, batch_size: int, patience: int, tag: str, out_dir: str) -> None:
    cmd = [
        sys.executable, "-u", "scripts/run_dl_baselines.py",
        "--cache", "results/cached_features/features_full.npz",
        "--encoders", ",".join(encoders),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--patience", str(patience),
        "--seed", "42",
        "--tag", tag,
        "--out-dir", out_dir,
    ]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"run failed with exit code {proc.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True)
    ap.add_argument("--encoders", default="simple_cnn")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--tag", default="dl_baselines")
    ap.add_argument("--out-dir", default="results/dl_baselines")
    args = ap.parse_args()

    wait_pid(args.wait_pid)
    run_rerun(
        [e.strip() for e in args.encoders.split(",") if e.strip()],
        args.epochs, args.batch_size, args.patience, args.tag, args.out_dir,
    )
    print("rerun queue finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
