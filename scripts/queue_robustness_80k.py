# -*- coding: utf-8 -*-
"""Queue the 80k final-model robustness rerun after the ensemble blind eval.

The existing robustness report covers only the 203k reference model; this
queue reruns the identical inference-time perturbation protocol on the final
80k mainline fold checkpoints (trained by queue_mainline_blind_ckpts.py).

Usage:
    python scripts/queue_robustness_80k.py --wait-pid 34556
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


def run_robustness() -> None:
    cmd = [
        sys.executable, "-u", "scripts/run_robustness_80k.py",
        "--out", "results/robustness_report_80k.json",
    ]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"80k robustness run failed with exit code {proc.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True)
    args = ap.parse_args()

    wait_pid(args.wait_pid)
    run_robustness()
    print("80k robustness queue finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
