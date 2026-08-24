# -*- coding: utf-8 -*-
"""Queue the K=8 vs K=12 reference-model multi-seed stability runs.

The post-combo driver already runs (serially, one GPU process at a time):
  combo search -> latency -> multi-seed(top-2 combo) -> DL baselines -> summary
This watcher waits until that driver chain has finished, then launches
scripts/run_multi_seed.py (mixed_k8 / mixed_k12 x seeds 7, 2024) so the
K=8 vs K=12 claim also gets 3-seed stability evidence. Started with
Start-Process -WindowStyle Hidden; safe to kill and restart (completed runs
are skipped by the runner scripts).

Usage:
  python scripts/queue_policy_multiseed.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
DRIVER_PID = ROOT / "results" / "post_combo_driver.pid"
DRIVER_LOG = ROOT / "results" / "post_combo_driver.log"
DONE_MARK = "post-combo driver finished."
WATCH_LOG = ROOT / "results" / "queue_policy_multiseed.log"
MULTI_LOG = ROOT / "results" / "multi_seed_run.log"
RUNS = [
    ("mixed_k8", 2024),
    ("mixed_k8", 7),
    ("mixed_k12", 2024),
    ("mixed_k12", 7),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with WATCH_LOG.open("a", encoding="utf-8") as h:
        h.write(line + "\n")


def pid_alive(pid_file: Path) -> bool:
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True, timeout=20)
        return f"{pid}" in out.stdout and "No tasks" not in out.stdout
    except Exception:
        return False


def main() -> None:
    log("watcher started")
    while pid_alive(DRIVER_PID):
        time.sleep(60)
    log("post-combo driver process is gone")
    if DRIVER_LOG.exists():
        tail = DRIVER_LOG.read_text(encoding="utf-8", errors="ignore").strip()
        if DONE_MARK in tail:
            log("driver finished normally; starting policy multi-seed")
        else:
            log("WARNING: driver finished without success marker; still starting policy multi-seed")
    else:
        log("WARNING: driver log missing; still starting policy multi-seed")

    # Skip runs whose cv_summary.json already exists (idempotent).
    remaining = []
    for policy, seed in RUNS:
        cfg_dir = ROOT / "results" / "ablations" / "window_policy" / f"seed{seed}" / f"policy_{policy}"
        if (cfg_dir / "cv_summary.json").exists():
            log(f"skip existing: {cfg_dir}")
            continue
        remaining.append((policy, seed))
    if not remaining:
        log("no remaining runs; nothing to do")
        return

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    with MULTI_LOG.open("a", encoding="utf-8") as handle:
        log("launching run_multi_seed.py (mixed_k8/mixed_k12 x seeds 7, 2024)")
        proc = subprocess.run(
            [PY, "-u", "scripts/run_multi_seed.py"],
            cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, env=env,
        )
        log(f"run_multi_seed.py exit={proc.returncode}")
    log("policy multi-seed driver finished")


if __name__ == "__main__":
    main()
