# -*- coding: utf-8 -*-
"""IEEE TIM Stage 1 A-E report with cluster-bootstrap statistics.

Reads results/stage1_tim/ only. The frozen blind predictions file is used
exclusively for post-hoc MCC with development thresholds; it never selects
anything. Historical E7 numbers are copied from the old ablation summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sklearn.metrics import matthews_corrcoef

from vsb_pd.evaluation import compute_bootstrap_ci, paired_bootstrap_ci

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "results" / "stage1_tim"
E7_SOURCE = ROOT / "results" / "ablations" / "dev_k8" / "ablation_summary.json"
BLIND = ROOT / "results" / "blind_80k_predictions.npz"

HYPOTHESIS = (
    "For 800k-point sparse high-rate measurements, coverage-aware sampling "
    "reduces the data volume the downstream model must process while a "
    "lightweight model remains competitively diagnostic; phase-aware "
    "hierarchical inference should recover true per-phase decisions rather "
    "than a measurement-shared score."
)


def load_seed_summary(config: str, seed: int) -> dict:
    path = STAGE / config / "seeds" / f"seed_{seed}" / "cv_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_oof(config: str, seed: int) -> dict:
    path = STAGE / config / "oof.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=False)
    mids = d["measurement_ids"]
    return {
        "phase_probs": d[f"seed_{seed}_phase_probs"].reshape(-1),
        "phase_targets": d[f"seed_{seed}_phase_targets"].reshape(-1),
        "meas_probs": d[f"seed_{seed}_meas_probs"],
        "meas_targets": d[f"seed_{seed}_meas_targets"],
        "phase_mids": np.repeat(mids, 3),
        "meas_mids": mids,
    }


def paired_ci(a_config: str, b_config: str, level: str, seed: int = 42) -> dict:
    a = load_oof(a_config, seed)
    b = load_oof(b_config, seed)
    return paired_bootstrap_ci(
        model_scores=a[f"{level}_probs"],
        baseline_scores=b[f"{level}_probs"],
        labels=a[f"{level}_targets"],
        measurement_ids=a[f"{level}_mids"],
        n_bootstrap=2000,
        seed=42,
    )


def single_ci(config: str, level: str, seed: int = 42) -> dict:
    a = load_oof(config, seed)
    ci = compute_bootstrap_ci(
        scores=a[f"{level}_probs"], labels=a[f"{level}_targets"],
        measurement_ids=a[f"{level}_mids"], metric_name="pr_auc",
        n_bootstrap=2000, seed=42,
    )
    return {"median": ci.median, "lower": ci.lower, "upper": ci.upper}


def fmt3(x) -> str:
    if x is None:
        return "-"
    return f"{x:.3f}" if np.isfinite(x) else "nan"


def metric_row(config: str, seed: int) -> str:
    try:
        s = load_seed_summary(config, seed)
    except FileNotFoundError:
        return f"| {config} | MISSING | | | | | |"
    fm = s["fold_mean_phase"]
    mm = s["fold_mean_measurement"]
    return (
        f"| {config} | {seed} | {fmt3(fm['pr_auc'])} | {fmt3(mm['pr_auc'])} | "
        f"{fmt3(fm['roc_auc'])} | {fmt3(mm['roc_auc'])} | "
        f"{fmt3(fm['mcc'])} | {fmt3(mm['mcc'])} |"
    )


def e6_final_name() -> str:
    sel = STAGE / "e6_lambda_selection.json"
    if not sel.exists():
        return "e6_ctx_concat_lam_<chosen>"
    chosen = json.loads(sel.read_text(encoding="utf-8"))["chosen_lambda"]
    return f"e6_ctx_concat_lam_{chosen:g}"


def main() -> None:
    missing = []
    required = ["e1_ctx_none", "e2_ctx_mean", "e3_ctx_max", "e4_ctx_concat",
                "e5_ctx_add", "enc_simple_cnn_ctx_concat", "enc_resnet1d_ctx_concat",
                "enc_inceptiontime_ctx_concat"]
    e6_name = e6_final_name()
    required.append(e6_name)
    for cfg in required:
        if not (STAGE / cfg / "cv_summary.json").exists():
            missing.append(cfg)

    benchmark_path = STAGE / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else {}
    if not benchmark:
        missing.append("benchmark.json")

    e7 = json.loads(E7_SOURCE.read_text(encoding="utf-8"))
    e7_row = next(r for r in e7 if r["name"] == "enc_dual__mil_gated_attention__ph_cyclic")

    lines = ["# IEEE TIM Stage 1 Evidence Report", "", "## A. Hypothesis", "", HYPOTHESIS, ""]
    lines += ["## B. Results", "", "### B1. Phase-interaction ablation, fold-mean primary (seeds 42/7/2024)", ""]
    lines += ["| config | seed | phase PR-AUC | meas PR-AUC | phase ROC | meas ROC | phase MCC | meas MCC |",
              "|---|---|---|---|---|---|---|---|"]
    for cfg in ["e1_ctx_none", "e2_ctx_mean", "e3_ctx_max", "e4_ctx_concat", "e5_ctx_add"]:
        for seed in (42, 7, 2024):
            lines.append(metric_row(cfg, seed))
    if e6_name != "e6_ctx_concat_lam_<chosen>":
        for seed in (42, 7, 2024):
            lines.append(metric_row(e6_name, seed))
    lines.append("")
    lines.append("### B2. Pooled-OOF secondary (seed 42)")
    lines.append("")
    lines.append("| config | phase PR-AUC | meas PR-AUC | phase ROC | meas ROC | phase MCC | meas MCC |")
    lines.append("|---|---|---|---|---|---|---|")
    b2_configs = ["e1_ctx_none", "e2_ctx_mean", "e3_ctx_max", "e4_ctx_concat", "e5_ctx_add"]
    if e6_name != "e6_ctx_concat_lam_<chosen>":
        b2_configs.append(e6_name)
    for cfg in b2_configs:
        try:
            s = load_seed_summary(cfg, 42)
            lines.append(
                f"| {cfg} | {fmt3(s['pooled_oof_phase']['pr_auc'])} | "
                f"{fmt3(s['pooled_oof_measurement']['pr_auc'])} | "
                f"{fmt3(s['pooled_oof_phase']['roc_auc'])} | "
                f"{fmt3(s['pooled_oof_measurement']['roc_auc'])} | "
                f"{fmt3(s['pooled_oof_phase']['mcc'])} | "
                f"{fmt3(s['pooled_oof_measurement']['mcc'])} |")
        except FileNotFoundError:
            lines.append(f"| {cfg} | MISSING | | | | |")

    lines.append("")
    lines.append("### B3. Matched encoders, context_concat, seed 42")
    lines.append("")
    lines.append("| encoder | params | MACs/meas | GPU b1 p50 ms | CPU b1 p50 ms | phase PR-AUC | meas PR-AUC | phase ROC | meas ROC | phase MCC | meas MCC |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for enc, cfg in [("cnn", "e4_ctx_concat"),
                     ("simple_cnn", "enc_simple_cnn_ctx_concat"),
                     ("resnet1d", "enc_resnet1d_ctx_concat"),
                     ("inceptiontime", "enc_inceptiontime_ctx_concat")]:
        try:
            s = load_seed_summary(cfg, 42)
            b = benchmark.get("encoders", {}).get(enc, {})
            macs = f"{b['macs_per_measurement'] / 1e6:.1f}M" if b else "-"
            gpu = fmt3(b.get("gpu_batch1_model_components_ms", {}).get("full_pipeline", {}).get("p50_ms")) if b else "-"
            cpu = fmt3(b.get("cpu_batch1_model_components_ms", {}).get("full_pipeline", {}).get("p50_ms")) if b else "-"
            lines.append(
                f"| {enc} | {s['n_params']} | {macs} | {gpu} | {cpu} | "
                f"{fmt3(s['fold_mean_phase']['pr_auc'])} | {fmt3(s['fold_mean_measurement']['pr_auc'])} | "
                f"{fmt3(s['fold_mean_phase']['roc_auc'])} | {fmt3(s['fold_mean_measurement']['roc_auc'])} | "
                f"{fmt3(s['fold_mean_phase']['mcc'])} | {fmt3(s['fold_mean_measurement']['mcc'])} |")
        except FileNotFoundError:
            lines.append(f"| {enc} | MISSING | | | | | | | | | |")

    lines.append("")
    lines.append("### B4. Historical reference (E7, not retrained)")
    lines.append("")
    lines.append(
        f"| config | params | phase PR-AUC | meas PR-AUC | source |\n"
        f"| dual + gated_attention + cyclic | {e7_row['n_params']} | "
        f"{fmt3(e7_row['mean_phase_pr_auc'])} | {fmt3(e7_row['mean_measurement_pr_auc'])} | "
        f"results/ablations/dev_k8/ablation_summary.json |")

    lines += ["", "## C. Statistical interpretation (seed 42, cluster bootstrap x2000, seed=42)", ""]
    lines.append("| comparison | level | PR-AUC diff median | 95% CI |")
    lines.append("|---|---|---|---|")
    for a, b in [("e4_ctx_concat", "e2_ctx_mean"), ("e4_ctx_concat", "e5_ctx_add")]:
        for level in ("phase", "meas"):
            try:
                d = paired_ci(a, b, level)
                lines.append(f"| {a} vs {b} | {level} | {d['diff_median']:.4f} | [{d['diff_lower']:.4f}, {d['diff_upper']:.4f}] |")
            except (FileNotFoundError, KeyError):
                lines.append(f"| {a} vs {b} | {level} | MISSING | - |")
    for enc, cfg in [("simple_cnn", "enc_simple_cnn_ctx_concat"),
                     ("resnet1d", "enc_resnet1d_ctx_concat"),
                     ("inceptiontime", "enc_inceptiontime_ctx_concat")]:
        for level in ("phase", "meas"):
            try:
                d = paired_ci(cfg, "e4_ctx_concat", level)
                lines.append(f"| {enc} vs cnn | {level} | {d['diff_median']:.4f} | [{d['diff_lower']:.4f}, {d['diff_upper']:.4f}] |")
            except (FileNotFoundError, KeyError):
                lines.append(f"| {enc} vs cnn | {level} | MISSING | - |")

    try:
        ci_p = single_ci("e4_ctx_concat", "phase")
        ci_m = single_ci("e4_ctx_concat", "meas")
        lines.append("")
        lines.append("Single-model cluster bootstrap CI for E4 (seed 42):")
        lines.append(f"- phase PR-AUC: median {ci_p['median']:.4f} [95% CI {ci_p['lower']:.4f}, {ci_p['upper']:.4f}]")
        lines.append(f"- measurement PR-AUC: median {ci_m['median']:.4f} [95% CI {ci_m['lower']:.4f}, {ci_m['upper']:.4f}]")
    except (FileNotFoundError, KeyError):
        lines.append("Single-model CI: MISSING (e4_ctx_concat seed 42 not complete)")

    lines += ["", "## D. Scientific conclusion", ""]
    lines.append("Conclusions below may only claim effects whose 95% CIs exclude 0 "
                 "for the corresponding comparison; otherwise the result is recorded as inconclusive/negative.")
    lines.append("- Phase-specific context (E4/E5): filled from C after CI inspection.")
    lines.append("- Hierarchical loss (E6 lambda selection): filled from B after selection record inspection.")
    lines.append("- Matched encoders: filled from B3/C; larger models may still win; that is a reported negative, not a failure.")
    lines.append("")

    blind = None
    if BLIND.exists():
        try:
            blind = np.load(BLIND, allow_pickle=False)
        except Exception as exc:
            print(f"Warning: could not load blind predictions: {exc}")
    if blind is not None:
        try:
            e2 = load_seed_summary("e2_ctx_mean", 42)
            ph_thr, ms_thr = e2["phase_threshold"], e2["measurement_threshold"]
            blind_phase_mcc = float(matthews_corrcoef(
                blind["phase_targets"].reshape(-1),
                (blind["phase_probs"].reshape(-1) >= ph_thr).astype(int)))
            blind_meas_mcc = float(matthews_corrcoef(
                blind["meas_labels"], (blind["meas_probs"] >= ms_thr).astype(int)))
            lines.append("### Post-hoc blind MCC (frozen 423 measurements; thresholds from dev OOF only)")
            lines.append(f"- phase MCC: {blind_phase_mcc:.4f} (threshold {ph_thr:.3f})")
            lines.append(f"- measurement MCC: {blind_meas_mcc:.4f} (threshold {ms_thr:.3f})")
            lines.append("These numbers were NOT used for model, hyperparameter, threshold, K, or structure selection.")
        except Exception as exc:
            lines.append(f"Post-hoc blind MCC: unavailable ({exc})")

    lines += ["", "## E. Paper impact (input to Stage 2, no rewriting in Stage 1)", ""]
    lines.append("- Coverage-aware sampling claim: reuse existing K=1/4/8/12 controlled evidence; "
                 "new Stage-1 tables report cost (MACs/latency) and diagnostic quality for the K=8 mixed policy.")
    lines.append("- Phase-specific claim: supported only if E4/E5 per-phase outputs and CI results support it.")
    lines.append("- Efficiency claim: distinguish NN forward latency from full measurement pipeline latency explicitly.")
    lines.append("- If any negative result occurs, Stage 2 must discuss it honestly rather than hide it.")

    md = "\n".join(lines) + "\n"
    (STAGE / "report_A_E.md").write_text(md, encoding="utf-8")

    data = {
        "hypothesis": HYPOTHESIS,
        "benchmark": benchmark,
        "e7": {k: e7_row.get(k) for k in ("n_params", "mean_phase_pr_auc", "mean_measurement_pr_auc")},
        "missing": missing,
    }
    (STAGE / "report_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {STAGE / 'report_A_E.md'}")
    if missing:
        print("MISSING required inputs:", missing)
        sys.exit(2)


if __name__ == "__main__":
    main()
