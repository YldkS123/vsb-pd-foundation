# -*- coding: utf-8 -*-
"""Resume the interrupted P0 pipeline with fixed bookkeeping.

The first post-combo driver run overwrote per-seed combo results (same tag
directory for two seeds) and died at the DL-baseline step with a CUDA OOM
while another GPU job was running. This driver re-runs the remaining pieces
with the GPU otherwise idle:

  1. DL baselines: resnet1d, tcn, inception (Mean MIL + Max phase inter.)
     -> results/dl_baselines/dl_baselines/
  2. combo top-2 multi-seed: seed 7 and 2024, per-seed tag directories
     (seed42 numbers come from results/ablations/dev_k8_combo/)
     -> results/ablations/multi_seed_combo_seed{seed}/
  3. writes results/combo_resume_summary.json with per-seed fold-mean
     PR-AUC for the top-2 configs plus DL baseline results.

Steps run sequentially, one GPU process at a time. Safe to kill and restart:
completed runs (cv_summary.json / summary files) are skipped.

Usage:
  python scripts/run_combo_resume_driver.py
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
LOG = ROOT / "results" / "combo_resume_driver.log"
COMBOSUMMARY = ROOT / "results" / "ablations" / "dev_k8_combo" / "ablation_summary.json"
DL_SUMMARY = ROOT / "results" / "dl_baselines" / "dl_baselines" / "dl_baseline_summary.json"
SEEDS = (7, 2024)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as h:
        h.write(line + "\n")


def run(cmd: list[str]) -> int:
    log("RUN: " + " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    with LOG.open("a", encoding="utf-8") as h:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=h, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def parse_config(name: str) -> tuple[str, str, str]:
    parts = name.split("__")
    return parts[0].replace("enc_", ""), parts[1].replace("mil_", ""), parts[2].replace("ph_", "")


def top_configs(n: int = 2) -> list[dict]:
    rows = json.loads(COMBOSUMMARY.read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("mean_phase_pr_auc") is not None]
    rows.sort(key=lambda r: -r["mean_phase_pr_auc"])
    return rows[:n]


def dl_baseline_runs() -> None:
    if DL_SUMMARY.exists():
        log(f"skip existing DL baseline summary: {DL_SUMMARY}")
        return
    rc = run([
        PY, "-u", "scripts/run_dl_baselines.py",
        "--encoders", "resnet1d",
        "--tag", "dl_baselines",
        "--epochs", "20", "--batch-size", "16", "--patience", "15",
    ])
    if rc != 0:
        log(f"ERROR: DL baseline run failed with rc={rc}")
        sys.exit(4)


def combo_multi_seed_runs(top: list[dict]) -> None:
    for row in top:
        name = row["name"]
        enc, mil, ph = parse_config(name)
        for seed in SEEDS:
            cfg_dir = ROOT / "results" / "ablations" / f"multi_seed_combo_seed{seed}" / name
            if (cfg_dir / "cv_summary.json").exists():
                log(f"skip existing {cfg_dir}")
                continue
            rc = run([
                PY, "-u", "scripts/run_ablations.py",
                "--encoders", enc, "--mils", mil, "--phases", ph,
                "--tag", f"multi_seed_combo_seed{seed}",
                "--seed", str(seed),
                "--epochs", "40", "--batch-size", "64", "--patience", "15",
            ])
            if rc != 0:
                log(f"ERROR: multi-seed run failed with rc={rc} for {name} seed={seed}")
                sys.exit(3)


def collect_summary(top: list[dict]) -> None:
    seed42 = {r["name"]: r for r in json.loads(COMBOSUMMARY.read_text(encoding="utf-8"))}
    multi = {}
    for row in top:
        name = row["name"]
        vals = []
        per_seed = {}
        for seed in (42, *SEEDS):
            if seed == 42:
                s = seed42.get(name)
            else:
                p = ROOT / "results" / "ablations" / f"multi_seed_combo_seed{seed}" / name / "cv_summary.json"
                s = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
            if s and s.get("mean_phase_pr_auc") is not None:
                vals.append(s["mean_phase_pr_auc"])
                per_seed[str(seed)] = s["mean_phase_pr_auc"]
        if vals:
            mean = float(sum(vals) / len(vals))
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            multi[name] = {
                "n_seeds": len(vals),
                "per_seed_mean_phase_pr_auc": per_seed,
                "mean_across_seeds": round(mean, 4),
                "std_across_seeds": round(std, 4),
            }

    dl = json.loads(DL_SUMMARY.read_text(encoding="utf-8")) if DL_SUMMARY.exists() else None
    out = ROOT / "results" / "combo_resume_summary.json"
    summary = {
        "combo_top_configs": [
            {"name": r["name"], "mean_phase_pr_auc": r["mean_phase_pr_auc"],
             "std_phase_pr_auc": r["std_phase_pr_auc"],
             "mean_measurement_pr_auc": r.get("mean_measurement_pr_auc"),
             "n_params": r.get("n_params")} for r in top
        ],
        "multi_seed": multi,
        "dl_baselines": dl,
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"summary saved: {out}")


def main() -> None:
    top = top_configs(2)
    log(f"top configs: {[r['name'] for r in top]}")
    if not top:
        log("ERROR: no valid combo rows")
        sys.exit(5)
    dl_baseline_runs()
    combo_multi_seed_runs(top)
    collect_summary(top)
    log("combo-resume driver finished.")


if __name__ == "__main__":
    main()
