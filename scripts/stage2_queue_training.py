# -*- coding: utf-8 -*-
"""Queue the remaining Stage 2 sampling rows sequentially.

Each row uses the locked E4 protocol with seed 42. Completed configs are
skipped so the queue can be re-run after interruptions.

Usage:
  python scripts/stage2_queue_training.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "cached_features"
OUT = ROOT / "results" / "stage2_sampling"

JOBS = [
    ("event_k8", CACHE / "features_policy_event_k8.npz"),
    ("random_k8", CACHE / "features_policy_random_k8.npz"),
    ("full_signal", CACHE / "features_policy_full_signal.npz"),
]


def main() -> None:
    python = sys.executable
    for config_name, cache in JOBS:
        summary = OUT / config_name / "cv_summary.json"
        if summary.exists():
            print(f"skip completed: {config_name}")
            continue
        print(f"run: {config_name} ({cache.name})")
        subprocess.run([
            python, "-u", str(ROOT / "scripts" / "stage2_sampling_runner.py"),
            "--cache", str(cache), "--config-name", config_name,
            "--seeds", "42", "--out-dir", str(OUT),
        ], check=True)
    print("queue finished")


if __name__ == "__main__":
    main()
