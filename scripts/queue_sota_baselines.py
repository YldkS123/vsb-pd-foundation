# -*- coding: utf-8 -*-
"""Queue the published SOTA baseline runs after the current GPU chain exits.

Waits for a running PID (e.g. the robustness queue), then launches
run_sota_baselines.py with the locked K=8 mixed-window cache, same
StratifiedGroupKFold(5, seed=42) folds and dev-only tuning protocol.

Usage:
    python scripts/queue_sota_baselines.py --wait-pid 13976
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def process_exists(pid: int) -> bool:
    cmd = (
        "if (Get-Process -Id " + str(pid) + " -ErrorAction SilentlyContinue) "
        "{ exit 0 } else { exit 1 }"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return True
    return proc.returncode == 0


def wait_pid(pid: int, poll_sec: int = 30) -> None:
    print(f"waiting for PID {pid} to exit ...", flush=True)
    while process_exists(pid):
        time.sleep(poll_sec)
    print(f"PID {pid} exited", flush=True)


def run_sota() -> None:
    cmd = [
        sys.executable, "-u", "scripts/run_sota_baselines.py",
        "--cache", "results/cached_features/features_policy_mixed_k8.npz",
        "--methods", "tf_cnn,cnn_qsvm",
        "--epochs", "40",
        "--batch-size", "64",
        "--patience", "15",
        "--cnn-epochs", "20",
        "--ensemble-seeds", "3",
        "--seed", "42",
        "--out-dir", "results/sota_baselines",
    ]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"SOTA baseline run failed with exit code {proc.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True)
    args = ap.parse_args()

    wait_pid(args.wait_pid)
    run_sota()
    print("SOTA baseline queue finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
