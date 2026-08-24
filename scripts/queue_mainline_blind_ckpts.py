# -*- coding: utf-8 -*-
"""Queue 80k mainline 5-fold checkpoint training + ensemble blind evaluation.

Waits for the current GPU queue (TCN, PID 4372) to exit, trains 5 dev-fold
checkpoints of the final lightweight mainline (encoder=cnn, MIL=attention,
phase=mean, 80,113 params) on the paper's locked features_policy_mixed_k8
cache with the same protocol as the multi-seed mainline runs, then runs the
5-fold-mean ensemble blind evaluation on the frozen 423-measurement holdout.
The ensemble receipt/lock are separate, so the single-model and 203k receipts
stay untouched.

Usage:
    python scripts/queue_mainline_blind_ckpts.py --wait-pid 4372
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


def run_training() -> None:
    cmd = [
        sys.executable, "-u", "scripts/run_ablations.py",
        "--cache", "results/cached_features/features_policy_mixed_k8.npz",
        "--encoders", "cnn",
        "--mils", "attention",
        "--phases", "mean",
        "--epochs", "40",
        "--batch-size", "64",
        "--patience", "15",
        "--seed", "42",
        "--tag", "dev_k8_blind_ckpts",
        "--out-dir", "results/ablations",
        "--save-ckpts",
    ]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"80k fold training failed with exit code {proc.returncode}")


def run_ensemble_blind() -> None:
    model_dir = (
        ROOT / "results" / "ablations" / "dev_k8_blind_ckpts"
        / "enc_cnn__mil_attention__ph_mean"
    )
    folds = sorted(model_dir.glob("model_fold*.pt")) if model_dir.exists() else []
    if len(folds) != 5:
        raise RuntimeError(f"expected 5 fold checkpoints, found {len(folds)} in {model_dir}")
    cmd = [
        sys.executable, "-u", "scripts/blind_evaluate_80k.py",
        "--model-dir", str(model_dir),
        "--calib-oof", str(model_dir / "oof.npz"),
        "--receipt", str(ROOT / "results" / "FINAL_EVALUATION_RECEIPT_80k_ensemble.json"),
        "--lock", str(ROOT / "results" / "final_eval_80k_ensemble.lock"),
        "--device", "cpu",
    ]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"80k ensemble blind eval failed with exit code {proc.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True)
    args = ap.parse_args()

    wait_pid(args.wait_pid)
    run_training()
    run_ensemble_blind()
    print("80k 5-fold ensemble blind queue finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
