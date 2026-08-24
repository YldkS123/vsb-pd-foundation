# -*- coding: utf-8 -*-
"""Post-combo experiment driver (P0-5 multi-seed + P0-6 DL baselines).

Waits for the running combination-search process (results/combo_search.pid) to
finish, then:

  1. reads results/ablations/dev_k8_combo/ablation_summary.json and picks the
     top-2 configurations by mean phase PR-AUC;
  2. runs those two configs with seeds 7 and 2024 under
     results/ablations/multi_seed_combo/ (seed 42 already exists from combo);
  3. runs the DL baseline encoders (resnet1d, tcn, inception) under
     results/dl_baselines/dl_baselines/ with Mean MIL + Max phase interaction;
  4. writes results/post_combo_driver_summary.json with per-seed fold-mean
     PR-AUC for the top configs and the DL baseline results.

All steps run sequentially (one GPU process at a time) and log to
results/post_combo_driver.log. Safe to kill and restart: completed runs are
skipped and step 1 waits for the combo summary only if the combo process is
still alive.

Usage:
  python scripts/run_post_combo_driver.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
COMBO_PID = ROOT / "results" / "combo_search.pid"
COMBO_SUMMARY = ROOT / "results" / "ablations" / "dev_k8_combo" / "ablation_summary.json"
LOG = ROOT / "results" / "post_combo_driver.log"
MULTI_TAG = "multi_seed_combo"
DL_OUT = ROOT / "results" / "dl_baselines" / "dl_baselines" / "dl_baseline_summary.json"
SEEDS = (7, 2024)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as h:
        h.write(line + "\n")


def combo_alive() -> bool:
    if not COMBO_PID.exists():
        return False
    try:
        pid = int(COMBO_PID.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                            capture_output=True, text=True, timeout=20)
        return f"{pid}" in out.stdout and "No tasks" not in out.stdout
    except Exception:
        return False


def wait_for_combo() -> None:
    while combo_alive():
        log("waiting for combination search to finish ...")
        time.sleep(60)
    if not COMBO_SUMMARY.exists():
        log(f"ERROR: combo summary missing: {COMBO_SUMMARY}")
        sys.exit(2)
    log("combination search finished; summary found.")


def run(cmd: list[str]) -> int:
    log("RUN: " + " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    with LOG.open("a", encoding="utf-8") as h:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=h, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def parse_config(name: str) -> tuple[str, str, str]:
    parts = name.split("__")
    enc = parts[0].replace("enc_", "")
    mil = parts[1].replace("mil_", "")
    ph = parts[2].replace("ph_", "")
    return enc, mil, ph


def top_configs(n: int = 2) -> list[dict]:
    rows = json.loads(COMBO_SUMMARY.read_text(encoding="utf-8"))
    if not COMBO_SUMMARY.exists():
        rows = []
        for p in sorted((ROOT / "results" / "ablations" / "dev_k8_combo").glob("*/cv_summary.json")):
            s = json.loads(p.read_text(encoding="utf-8"))
            rows.append({"name": p.parent.name, **s})
    else:
        rows = json.loads(COMBO_SUMMARY.read_text(encoding="utf-8"))
    rows = [r for r in rows if "mean_phase_pr_auc" in r and r["mean_phase_pr_auc"] is not None]
    rows.sort(key=lambda r: -r["mean_phase_pr_auc"])
    return rows[:n]




def latency_run() -> None:
    out = ROOT / "results" / "latency_report.json"
    if out.exists():
        log(f"skip existing latency report: {out}")
        return
    rc = run([PY, "-u", "scripts/measure_latency.py"])
    if rc != 0:
        log(f"ERROR: latency measurement failed with rc={rc}")
        sys.exit(6)


def multi_seed_runs(top: list[dict]) -> None:
    for row in top:
        name = row["name"]
        enc, mil, ph = parse_config(name)
        for seed in SEEDS:
            cfg_dir = ROOT / "results" / "ablations" / MULTI_TAG / f"seed{seed}" / name
            if (cfg_dir / "cv_summary.json").exists():
                log(f"skip existing {cfg_dir}")
                continue
            rc = run([
                PY, "-u", "scripts/run_ablations.py",
                "--encoders", enc, "--mils", mil, "--phases", ph,
                "--tag", MULTI_TAG, "--seed", str(seed),
                "--epochs", "40", "--batch-size", "64", "--patience", "15",
            ])
            if rc != 0:
                log(f"ERROR: multi-seed run failed with rc={rc} for {name} seed={seed}")
                sys.exit(3)


def dl_baseline_runs() -> None:
    if DL_OUT.exists():
        log(f"skip existing DL baseline summary: {DL_OUT}")
        return
    rc = run([
        PY, "-u", "scripts/run_dl_baselines.py",
        "--encoders", "resnet1d,tcn,inception",
        "--tag", "dl_baselines",
        "--epochs", "40", "--batch-size", "64", "--patience", "15",
    ])
    if rc != 0:
        log(f"ERROR: DL baseline run failed with rc={rc}")
        sys.exit(4)


def collect_summary(top: list[dict]) -> dict:
    seeds = {42: {r["name"]: r for r in json.loads(COMBO_SUMMARY.read_text(encoding="utf-8"))}}
    for seed in SEEDS:
        seeds[seed] = {}
        for row in top:
            name = row["name"]
            p = ROOT / "results" / "ablations" / MULTI_TAG / f"seed{seed}" / name / "cv_summary.json"
            if p.exists():
                seeds[seed][name] = json.loads(p.read_text(encoding="utf-8"))

    multi = {}
    for row in top:
        name = row["name"]
        vals = []
        for seed in (42, *SEEDS):
            s = seeds[seed].get(name)
            if s and s.get("mean_phase_pr_auc") is not None:
                vals.append(s["mean_phase_pr_auc"])
        if vals:
            mean = float(sum(vals) / len(vals))
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            multi[name] = {
                "n_seeds": len(vals),
                "per_seed_mean_phase_pr_auc": {str(k): seeds[k].get(name, {}).get("mean_phase_pr_auc") for k in (42, *SEEDS)},
                "mean_across_seeds": round(mean, 4),
                "std_across_seeds": round(std, 4),
            }

    dl = None
    if DL_OUT.exists():
        dl = json.loads(DL_OUT.read_text(encoding="utf-8"))

    summary = {
        "combo_top_configs": [{"name": r["name"], "mean_phase_pr_auc": r["mean_phase_pr_auc"],
                               "std_phase_pr_auc": r["std_phase_pr_auc"],
                               "mean_measurement_pr_auc": r.get("mean_measurement_pr_auc"),
                               "n_params": r.get("n_params")} for r in top],
        "multi_seed": multi,
        "dl_baselines": dl,
    }
    out = ROOT / "results" / "post_combo_driver_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"summary saved: {out}")
    return summary


def main() -> None:
    wait_for_combo()
    top = top_configs(2)
    if not top:
        log("ERROR: no valid combo rows")
        sys.exit(5)
    log(f"top configs: {[r['name'] for r in top]}")
    latency_run()
    multi_seed_runs(top)
    dl_baseline_runs()
    collect_summary(top)
    log("post-combo driver finished.")


if __name__ == "__main__":
    main()
