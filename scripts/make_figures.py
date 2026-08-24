# -*- coding: utf-8 -*-
"""Generate the 8 paper figures (80k mainline story) into docs/figures/.

Dependencies on already-produced artifacts (scripts/make_figure_data_80k.py):
  results/figures_data/holdout_ensemble_probs_80k.npz
  results/figures_data/holdout_examples_attn_80k.npz
Existing experiment artifacts:
  results/baseline_oof/*.npz
  results/ablations/dev_k8_combo/*/oof.npz and cv_summary.json
  results/ablations/dev_k8/*/cv_summary.json
  results/ablations/window_policy/dev/policy_ablation_summary.json
  results/robustness_report_80k.json
  results/shift_aug_80k/shift_robustness.json
  C:/Users/hrfxgfx/Desktop/1111/train.parquet + metadata_train.csv (fig 2, 5)

Run from the project root:
    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vsb_pd.extract import read_measurement_signals  # noqa: E402
from vsb_pd.metadata import load_metadata  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ style ---
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun"],
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 100,
    }
)

C_UNIFORM = "#4C72B0"
C_EVENT = "#DD8452"
C_MODEL = "#C44E52"
C_POS = "#2E74B5"
C_AUG = "#3E7D44"
C_GRAY = "#9AA0A6"

RESULTS = ROOT / "results"
RAW_PARQUET = Path("C:/Users/hrfxgfx/Desktop/1111/train.parquet")
METADATA_PATH = Path("C:/Users/hrfxgfx/Desktop/1111/metadata_train.csv")

KIND_COLOR = {0: C_UNIFORM, 1: C_EVENT, 2: "#6B8E23"}


def save(fig, name: str, dpi: int = 200):
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", path.name)


def pr_curve(y_true, y_score):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return p, r, ap


# ------------------------------------------------------------- fig 1: 架构 ---
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF1FB", ec="#2E74B5", fs=10, dashed=False, weight="normal"):
        b = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.2, edgecolor=ec, facecolor=fc,
            linestyle="--" if dashed else "-", mutation_aspect=1.0,
        )
        ax.add_patch(b)
        ax.text(
            x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#1F2D3D", linespacing=1.35, fontweight=weight,
        )

    def arrow(x1, y1, x2, y2, color="#4A6785", lw=1.6):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="-|>", mutation_scale=16, linewidth=lw, color=color,
                shrinkA=2, shrinkB=2,
            )
        )

    # Row 1: input -> sampling -> lightweight encoder
    box(0.02, 0.68, 0.16, 0.24, "三相信号\nx_A, x_B, x_C\n(各 800,000 点 @ 40 MHz)", fc="#F6F8FA", ec="#8A94A6", fs=9.5)
    box(0.21, 0.68, 0.19, 0.24, "覆盖感知窗口采样\nK=8：4 等距 + 4 事件\n跨类型去重 · 分层回退", fc="#EAF1FB", ec="#2E74B5", fs=9.5)

    enc = FancyBboxPatch(
        (0.44, 0.56), 0.36, 0.36,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2, edgecolor="#2E74B5", facecolor="#FFFFFF",
    )
    ax.add_patch(enc)
    ax.text(0.62, 0.875, "窗口编码器（纯 CNN，轻量主线）", ha="center", fontsize=10.5, weight="bold", color="#1F2D3D")
    box(0.455, 0.69, 0.33, 0.155, "4 层深度可分离 1D CNN\n核 15/11/7/5 · 步长 4 · 通道 32→128\n全局均值/最大池化", fc="#E8F4E8", ec="#3E7D44", fs=9)
    box(0.455, 0.575, 0.33, 0.09, "58 维物理特征分支（可选，消融验证排除）", fc="#FBFBFB", ec="#8A94A6", fs=8.5, dashed=True)
    ax.text(0.62, 0.53, "→ 128 维窗口表示", ha="center", fontsize=8.5, color="#5A6B7C")

    # Row 2: MIL -> phase interaction -> classifier -> phase probs
    box(0.02, 0.30, 0.21, 0.20, "Attention MIL\n(窗口 → 相位聚合)", fc="#EAF1FB", ec="#2E74B5", fs=9.5)
    box(0.255, 0.30, 0.225, 0.20, "三相感知交互\ncontext-concat（主线）\n循环等变（参考验证）", fc="#EAF1FB", ec="#2E74B5", fs=9)
    box(0.505, 0.30, 0.185, 0.20, "相位分类头\nLinear(128→64→1) × 3", fc="#EAF1FB", ec="#2E74B5", fs=9.5)
    box(0.715, 0.30, 0.26, 0.20, "相位级概率\np_A, p_B, p_C\n(相位级弱监督标签)", fc="#FDF3E7", ec="#C77A2E", fs=9.5)

    # Row 3: measurement
    box(0.715, 0.03, 0.26, 0.18, "noisy-OR 融合\n测量级概率 p_m", fc="#FDF3E7", ec="#C77A2E", fs=9.5)

    # loss annotation
    ax.text(0.885, 0.55, "相位级弱监督标签\nBCEWithLogitsLoss", ha="center", fontsize=8.5,
            color="#8A2E33", style="italic")
    ax.add_patch(
        FancyArrowPatch(
            (0.885, 0.515), (0.845, 0.49),
            arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#8A2E33",
            linestyle=(0, (4, 3)), shrinkA=2, shrinkB=2,
        )
    )

    # arrows
    arrow(0.18, 0.80, 0.208, 0.80)
    arrow(0.40, 0.80, 0.438, 0.80)
    ax.add_patch(
        FancyArrowPatch(
            (0.62, 0.56), (0.125, 0.535),
            arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color="#4A6785",
            connectionstyle="arc3,rad=0.08", shrinkA=2, shrinkB=2,
        )
    )
    arrow(0.23, 0.40, 0.253, 0.40)
    arrow(0.48, 0.40, 0.503, 0.40)
    arrow(0.69, 0.40, 0.713, 0.40)
    arrow(0.845, 0.30, 0.845, 0.212)

    ax.text(0.315, 0.94, "窗口 (3 相 × K=8 × 8192)", ha="center", fontsize=8.5, color="#5A6B7C")

    fig.tight_layout()
    save(fig, "fig1_architecture.png", dpi=220)


# -------------------------------------------------------- fig 2: 窗口采样 ---
def signal_envelope(sig: np.ndarray, bins: int = 4000) -> np.ndarray:
    n = sig.size // bins
    x = np.abs(sig[: n * bins]).reshape(bins, n)
    return x.max(axis=1)


def fig2_window_sampling():
    data = np.load(RESULTS / "figures_data" / "holdout_examples_attn_80k.npz", allow_pickle=False)
    mid = int(data["positive_mid"])
    starts = data["starts_pos"]
    kinds = data["kinds_pos"]
    metadata = load_metadata(METADATA_PATH)
    group = metadata[metadata["id_measurement"] == mid].sort_values("phase")
    signals = read_measurement_signals(RAW_PARQUET, group["signal_id"].astype(int).tolist())
    phase = 2  # C 相
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.6))
    ax_top, ax_bot = axes
    env = signal_envelope(signals[phase])
    x = np.arange(len(env)) * (800000 / len(env))

    ax_top.plot(x, env, color="#33475B", lw=0.8)
    for k in range(starts.shape[1]):
        kind = int(kinds[phase, k])
        ax_top.axvspan(starts[phase, k], starts[phase, k] + 8192, color=KIND_COLOR[kind], alpha=0.28, lw=0)
    ax_top.set_ylabel("Amplitude envelope (max |x|)")
    ax_top.set_title(f"Measurement {mid} · Phase C with locked window sampling (K=8: 4 uniform + 4 event)", fontsize=11)
    ax_top.set_xlim(0, 800000)
    ax_top.set_xticks(np.arange(0, 800001, 100000))
    ax_top.ticklabel_format(axis="x", style="sci", scilimits=(5, 5))
    ax_top.grid(alpha=0.25)
    ax_top.legend(
        handles=[
            Patch(facecolor=C_UNIFORM, alpha=0.55, label="Uniform anchor windows (4)"),
            Patch(facecolor=C_EVENT, alpha=0.55, label="Event windows (4)"),
        ],
        loc="upper right", framealpha=0.9,
    )

    ax_bot.plot(x, env, color="#33475B", lw=0.8)
    for k in range(starts.shape[1]):
        kind = int(kinds[phase, k])
        ax_bot.axvspan(starts[phase, k], starts[phase, k] + 8192, color=KIND_COLOR[kind], alpha=0.30, lw=0)
        ax_bot.text(starts[phase, k] + 4096, env.max() * 0.72, f"{k+1}",
                    ha="center", fontsize=8, color=KIND_COLOR[kind], weight="bold")
    ax_bot.set_xlabel("Time (samples)")
    ax_bot.set_ylabel("Amplitude envelope (max |x|)")
    ax_bot.set_title("Window positions (numbers = window index; blue = uniform, orange = event)", fontsize=11)
    ax_bot.set_xlim(0, 250000)
    ax_bot.set_xticks(np.arange(0, 250001, 50000))
    ax_bot.ticklabel_format(axis="x", style="sci", scilimits=(5, 5))
    ax_bot.grid(alpha=0.25)

    fig.tight_layout()
    save(fig, "fig2_window_sampling.png", dpi=200)


# ------------------------------------------------------------- fig 3: PR ---
def fig3_pr_curves():
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    baseline_files = sorted((RESULTS / "baseline_oof").glob("*.npz"))
    curves = []
    for f in baseline_files:
        d = np.load(f, allow_pickle=False)
        p, r, ap = pr_curve(d["labels"], d["oof_probs"])
        curves.append((f.stem, p, r, ap))
    model = np.load(
        RESULTS / "stage1_tim" / "e4_ctx_concat" / "oof.npz",
        allow_pickle=False,
    )
    p_model, r_model, ap_model = pr_curve(
        model["seed_42_phase_targets"].flatten(), model["seed_42_phase_probs"].flatten(),
    )

    best = max(curves, key=lambda c: c[3])
    for stem, p, r, ap in curves:
        if stem == best[0]:
            ax.plot(r, p, color=C_EVENT, lw=1.8, ls="--", alpha=0.95,
                    label=f"Best conventional baseline {stem.split('_', 1)[1].upper()} (AP={ap:.3f})")
        else:
            ax.plot(r, p, color=C_GRAY, lw=0.9, alpha=0.55)
    ax.plot(r_model, p_model, color=C_MODEL, lw=2.6, label=f"E4 context-concat mainline (pooled OOF, AP={ap_model:.3f})")
    ax.plot([0, 1], [443 / 7443, 443 / 7443], color="#B0B0B0", ls=":", lw=1.0,
            label=f"Random baseline (positive rate {443 / 7443:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Development PR curves (pooled OOF; 443 positive phases)", fontsize=12)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig3_pr_curves.png", dpi=200)


# -------------------------------------------------------- fig 4: 校准图 ---
def fig4_reliability():
    d = np.load(RESULTS / "figures_data" / "holdout_ensemble_probs_80k.npz", allow_pickle=False)
    y = d["labels"].flatten()
    p = d["probs"].flatten()

    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.searchsorted(bins[1:-1], p, side="left"), 0, 9)
    counts = np.bincount(idx, minlength=10).astype(float)
    sums = np.bincount(idx, weights=p, minlength=10)
    obs = np.bincount(idx, weights=y, minlength=10)
    ece = float(np.sum(np.abs(obs / np.maximum(counts, 1) - sums / np.maximum(counts, 1)) * counts) / len(y))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax1.plot([0, 1], [0, 1], ls="--", color="#8A94A6", lw=1.0, label="Perfect calibration")
    ax1.plot(prob_pred, prob_true, "-o", color=C_MODEL, lw=2.0, ms=4, label="Model (10 bins)")
    for i in range(len(prob_pred)):
        gap = prob_true[i] - prob_pred[i]
        color = "#C44E52" if gap >= 0 else "#2E74B5"
        ax1.add_patch(plt.Rectangle((prob_pred[i], prob_pred[i]), 0.005, gap, color=color, alpha=0.35))
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title(f"Historical locked-mainline blind-test calibration reliability (ECE = {ece:.4f})", fontsize=11.5)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.25)

    ax2.hist(p, bins=25, range=(0, 1), color="#9FB6D4", edgecolor="white", log=True)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Number of phases (log scale)")
    ax2.set_title("Predicted probability distribution (1269 phases)", fontsize=11.5)
    ax2.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    save(fig, "fig4_reliability.png", dpi=200)


# ------------------------------------------------------------ fig 5: 示例 ---
def fig5_examples():
    d = np.load(RESULTS / "figures_data" / "holdout_examples_attn_80k.npz", allow_pickle=False)
    metadata = load_metadata(METADATA_PATH)

    examples = [
        (int(d["positive_mid"]), d["starts_pos"], d["kinds_pos"], d["targets_pos"], d["probs_pos"], d["attn_pos"]),
        (int(d["negative_mid"]), d["starts_neg"], d["kinds_neg"], d["targets_neg"], d["probs_neg"], d["attn_neg"]),
    ]
    signals_cache: dict[int, np.ndarray] = {}
    for mid, *_ in examples:
        if mid not in signals_cache:
            group = metadata[metadata["id_measurement"] == mid].sort_values("phase")
            signals_cache[mid] = read_measurement_signals(
                RAW_PARQUET, group["signal_id"].astype(int).tolist(),
            )

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 7.6))
    for row, (mid, starts, kinds, targets, probs, attn) in enumerate(examples):
        signals = signals_cache[mid]
        for ph in range(3):
            ax = axes[row, ph]
            env = signal_envelope(signals[ph])
            x = np.arange(len(env)) * (800000 / len(env))
            ax.plot(x, env, color="#33475B", lw=0.7)
            wmax = int(np.argmax(attn[ph]))
            amax = attn[ph].max() if attn[ph].max() > 0 else 1.0
            for k in range(starts.shape[1]):
                kind = int(kinds[ph, k])
                alpha = 0.18 + 0.55 * float(attn[ph, k] / amax)
                ax.axvspan(starts[ph, k], starts[ph, k] + 8192, color=KIND_COLOR[kind], alpha=alpha, lw=0)
            ax.axvspan(starts[ph, wmax], starts[ph, wmax] + 8192, facecolor="none", edgecolor=C_MODEL, lw=1.8, ls="--")
            ax.set_title(f"Phase {'ABC'[ph]}  p={probs[ph]:.2f} (label {int(targets[ph])})", fontsize=10)
            ax.set_xlim(0, 800000)
            ax.set_xticks(np.arange(0, 800001, 400000))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(5, 5))
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(alpha=0.2)
            if ph == 0:
                ax.set_ylabel("Amplitude envelope (max |x|)")
            if row == 1:
                ax.set_xlabel("Time (samples)")

        ax = axes[row, 3]
        xpos = np.arange(8)
        width = 0.25
        for ph in range(3):
            color = [KIND_COLOR[int(kinds[ph, k])] for k in range(8)]
            ax.bar(xpos + (ph - 1) * width, attn[ph], width=width * 0.9,
                   color=color, alpha=0.85, label=f"Phase {'ABC'[ph]}")
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{k+1}" for k in range(8)], fontsize=8)
        ax.set_xlabel("Window index")
        ax.set_ylabel("Attention weight")
        ax.set_title("Window attention weights (5-fold model average)", fontsize=10)
        ax.grid(alpha=0.2, axis="y")
        ax.legend(fontsize=8, loc="upper right")

    axes[0, 0].set_ylabel(f"Positive · measurement {examples[0][0]}\nAmplitude envelope (max |x|)")
    axes[1, 0].set_ylabel(f"Negative · measurement {examples[1][0]}\nAmplitude envelope (max |x|)")
    handles = [
        Patch(facecolor=C_UNIFORM, alpha=0.55, label="Uniform window"),
        Patch(facecolor=C_EVENT, alpha=0.55, label="Event window"),
        Line2D([0], [0], color=C_MODEL, ls="--", lw=1.8, label="Max-attention window"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Historical locked-mainline blind-test examples: three-phase signals, window sampling, and attention weights", fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save(fig, "fig5_examples.png", dpi=210)


# -------------------------------------------------------------- fig 6: 消融 ---
def fig6_ablation():
    combo = json.loads((RESULTS / "ablations" / "dev_k8_combo" / "ablation_summary.json").read_text(encoding="utf-8"))
    by = {}
    for s in combo:
        c = s["config"]
        by[(c["mil"], c["phase"])] = s
    mils = ["mean", "attention", "gated_attention"]
    phases = ["max", "mean"]
    mil_labels = {"mean": "Mean", "attention": "Attention", "gated_attention": "Gated-Att"}
    phase_labels = {"max": "Max", "mean": "Mean"}

    ref = {}
    for name in [
        "enc_cnn__mil_gated_attention__ph_cyclic",
        "enc_dual__mil_gated_attention__ph_cyclic",
        "enc_feature__mil_gated_attention__ph_cyclic",
    ]:
        p = RESULTS / "ablations" / "dev_k8" / name / "cv_summary.json"
        ref[name] = json.loads(p.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    x = np.arange(len(mils))
    width = 0.34
    for j, ph in enumerate(phases):
        vals = [by[(m, ph)]["mean_phase_pr_auc"] for m in mils]
        stds = [by[(m, ph)]["std_phase_pr_auc"] for m in mils]
        ax.bar(x + (j - 0.5) * width, vals, width=width * 0.9, yerr=stds, capsize=4,
               color=C_EVENT if ph == "max" else C_POS, alpha=0.9, edgecolor="white",
               error_kw={"lw": 1.0}, label=f"Phase interaction: {phase_labels[ph]}")
        for xi, v, sd in zip(x, vals, stds):
            ax.text(xi + (j - 0.5) * width, v + sd + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([mil_labels[m] for m in mils], fontsize=9.5)
    ax.set_ylim(0.5, 0.78)
    ax.set_ylabel("Phase PR-AUC (5-fold mean)")
    ax.set_title("Historical combination search: MIL aggregation × phase interaction", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    names = ["纯 CNN", "双分支（参考）", "纯统计特征"]
    names = ["CNN-only", "Dual (reference)", "Stats-only"]
    keys = [
        "enc_cnn__mil_gated_attention__ph_cyclic",
        "enc_dual__mil_gated_attention__ph_cyclic",
        "enc_feature__mil_gated_attention__ph_cyclic",
    ]
    vals = [ref[k]["mean_phase_pr_auc"] for k in keys]
    stds = [ref[k]["std_phase_pr_auc"] for k in keys]
    colors = [C_MODEL if i == 0 else "#5B8DB8" for i in range(len(names))]
    ax.bar(range(len(names)), vals, yerr=stds, capsize=4, color=colors, alpha=0.9,
           edgecolor="white", error_kw={"lw": 1.0})
    for i, (v, sd) in enumerate(zip(vals, stds)):
        ax.text(i, v + sd + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=12, ha="right")
    ax.set_ylim(0, 0.78)
    ax.set_title("Encoder (reference mechanism validation)", fontsize=11.5)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Ablations: historical combination search and reference encoder mechanism validation (K=8 mixed windows)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig6_ablation.png", dpi=200)


# ----------------------------------------------------------- fig 7: 鲁棒性 ---
def fig7_robustness():
    r = json.loads((RESULTS / "robustness_report_80k.json").read_text(encoding="utf-8"))
    aug = json.loads((RESULTS / "shift_aug_80k" / "shift_robustness.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.8, 4.6))

    snr_keys = ["snr_5db", "snr_10db", "snr_20db"]
    x = [0, 1, 2, 3]
    labels = ["5 dB", "10 dB", "20 dB", "无噪声"]
    labels = ["5 dB", "10 dB", "20 dB", "No noise"]
    phase = [r["noise"][k]["phase"]["pr_auc"] for k in snr_keys] + [r["baseline_phase"]["pr_auc"]]
    meas = [r["noise"][k]["measurement"]["pr_auc"] for k in snr_keys] + [r["baseline_measurement"]["pr_auc"]]
    ax1.plot(x, phase, "-o", color=C_MODEL, lw=2.0, label="Phase-level")
    ax1.plot(x, meas, "-s", color=C_POS, lw=2.0, label="Measurement-level (noisy-OR)")
    for xi, v in zip(x, phase):
        ax1.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Gaussian-noise SNR")
    ax1.set_ylabel("PR-AUC")
    ax1.set_title("Noise robustness (historical locked mainline)", fontsize=11.5)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.set_ylim(0, 0.75)

    def shift_drop(key):
        return r["shift"][key].get("measurement_pr_auc_drop_pct", 0.0)

    items = [
        ("Amp ×0.8", float(r["scale"]["x0.8"].get("measurement_pr_auc_drop_pct", 0.0)), C_POS),
        ("Amp ×1.2", float(r["scale"]["x1.2"].get("measurement_pr_auc_drop_pct", 0.0)), C_POS),
        ("Shift -64", shift_drop("-64"), C_EVENT),
        ("Shift +64", shift_drop("+64"), C_EVENT),
        ("Shift -128", shift_drop("-128"), C_EVENT),
        ("Shift +128", shift_drop("+128"), C_EVENT),
        ("Missing A", float(r["missing_phase"]["miss_phase_0"]["measurement_pr_auc_drop_pct"]), "#6B8E23"),
        ("Missing B", float(r["missing_phase"]["miss_phase_1"]["measurement_pr_auc_drop_pct"]), "#6B8E23"),
        ("Missing C", float(r["missing_phase"]["miss_phase_2"]["measurement_pr_auc_drop_pct"]), "#6B8E23"),
    ]
    names = [n for n, _, _ in items]
    vals = [v for _, v, _ in items]
    colors = [c for _, _, c in items]
    ax2.bar(range(len(names)), vals, color=colors, alpha=0.88, edgecolor="white")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=8.5)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=9, rotation=20, ha="right")
    ax2.set_ylabel("Measurement PR-AUC relative drop (%)")
    ax2.set_title("Amplitude / time-shift / missing-phase perturbations", fontsize=11.5)
    ax2.grid(alpha=0.25, axis="y")
    ax2.set_ylim(0, max(max(vals) * 1.35, 12))

    shift_keys = ["-128", "-64", "+64", "+128"]
    main_drop = [r["shift"][k]["phase_pr_auc_drop_pct"] for k in shift_keys]
    aug_drop = [aug["shift"][k]["phase_pr_auc_drop_pct"] for k in shift_keys]
    x3 = np.arange(len(shift_keys))
    width = 0.36
    ax3.bar(x3 - width / 2, main_drop, width, color=C_MODEL, alpha=0.9, edgecolor="white",
            label="Mainline (frozen blind-test checkpoints)")
    ax3.bar(x3 + width / 2, aug_drop, width, color=C_AUG, alpha=0.9, edgecolor="white",
            label="Time-shift augmentation variant")
    ax3.axhline(0, color="#333333", lw=1.0)
    ax3.axhspan(0, 3, color=C_AUG, alpha=0.10)
    for i, v in enumerate(main_drop):
        ax3.text(i - width / 2, v + 2, f"{v:.1f}", ha="center", fontsize=8.5, color=C_MODEL)
    for i, v in enumerate(aug_drop):
        ax3.text(i + width / 2, v - 4 if v < 0 else v + 2, f"{v:.1f}", ha="center", fontsize=8.5, color=C_AUG)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(["-128", "-64", "+64", "+128"])
    ax3.set_xlabel("Time shift (samples)")
    ax3.set_ylabel("Phase PR-AUC relative change (%, negative = improvement)")
    ax3.set_title("Time shift: mainline vs augmentation variant", fontsize=11.5)
    ax3.legend(fontsize=8.5, loc="upper right")
    ax3.grid(alpha=0.25, axis="y")
    ax3.set_ylim(-12, max(main_drop) * 1.15)

    fig.suptitle("Inference-time robustness and time-shift augmentation (historical locked mainline, frozen blind-test checkpoints)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig7_robustness.png", dpi=200)


# ----------------------------------------------------------- fig 8: 窗口策略 ---
def fig8_window_policy():
    s = json.loads(
        (RESULTS / "ablations" / "window_policy" / "dev" / "policy_ablation_summary.json").read_text(encoding="utf-8")
    )
    by_policy = {item["policy"]: item for item in s}

    ks = [1, 4, 8, 12]
    vals = [by_policy["single"]["mean_phase_pr_auc"],
            by_policy["mixed_k4"]["mean_phase_pr_auc"],
            by_policy["mixed_k8"]["mean_phase_pr_auc"],
            by_policy["mixed_k12"]["mean_phase_pr_auc"]]
    stds = [by_policy["single"]["std_phase_pr_auc"],
            by_policy["mixed_k4"]["std_phase_pr_auc"],
            by_policy["mixed_k8"]["std_phase_pr_auc"],
            by_policy["mixed_k12"]["std_phase_pr_auc"]]
    ev = by_policy["event"]["mean_phase_pr_auc"]
    eq = by_policy["equidistant"]["mean_phase_pr_auc"]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.errorbar(ks, vals, yerr=stds, fmt="-o", color=C_MODEL, lw=2.0, ms=7,
                capsize=5, capthick=1.5, label="Reference: mixed policy (uniform + event)")
    ax.errorbar([8, 12], [0.639, 0.670], yerr=[0.055, 0.070], fmt="-s", color="#2E74B5", lw=2.0, ms=7,
                capsize=5, capthick=1.5, label="80k mainline (seed 42)")
    ax.axhline(ev, color=C_EVENT, ls="--", lw=1.6, label=f"Event-only K=8 (reference {ev:.3f})")
    ax.axhline(eq, color=C_UNIFORM, ls=":", lw=1.6, label=f"Uniform-only K=8 (reference {eq:.3f})")
    for k, v in zip(ks, vals):
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    for k, v in [(8, 0.639), (12, 0.670)]:
        ax.annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9,
                    color="#2E74B5")
    ax.set_xlabel("Window coverage count K")
    ax.set_ylabel("Phase PR-AUC (5-fold mean)")
    ax.set_title("Window-policy ablation: coverage density and composition (reference mechanism validation + historical locked mainline)", fontsize=12)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylim(0.15, 0.78)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig8_window_policy.png", dpi=200)


# --------------------------------------------- fig 9: 采样受控对比 ---
def fig9_sampling_policy():
    stage = RESULTS / "stage2_sampling"
    stage1 = RESULTS / "stage1_tim"
    policies = ["uniform_k8", "event_k8", "random_k8", "mixed_k8", "full_signal"]
    labels = {
        "uniform_k8": "Uniform\nK=8",
        "event_k8": "Event\nK=8",
        "random_k8": "Random\nK=8",
        "mixed_k8": "Mixed K=8\n(proposed)",
        "full_signal": "Full\nsignal",
    }
    colors = [C_UNIFORM, C_EVENT, "#8A94A6", C_MODEL, "#5B8DB8"]

    def load_summary(policy):
        if policy == "mixed_k8":
            path = stage1 / "e4_ctx_concat" / "seeds" / "seed_42" / "cv_summary.json"
        else:
            path = stage / policy / "seeds" / "seed_42" / "cv_summary.json"
        return json.loads(path.read_text(encoding="utf-8"))

    phase = [load_summary(p)["fold_mean_phase"]["pr_auc"] for p in policies]
    meas = [load_summary(p)["fold_mean_measurement"]["pr_auc"] for p in policies]
    bench = json.loads((stage / "benchmark.json").read_text(encoding="utf-8"))
    macs = [bench["policies"][p]["macs_per_measurement"] / 1e6 for p in policies]
    gpu_ms = [bench["policies"][p]["gpu_batch1_model_components_ms"]["full_pipeline"]["p50_ms"]
              for p in policies]
    sel = bench["selection_components_cpu_ms"]
    sel_ms = [sel.get(p, {}).get("p50_ms", 0.0) for p in policies]

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.6))
    ax = axes[0]
    x = np.arange(len(policies))
    ax.bar(x - 0.2, phase, 0.4, color=colors, alpha=0.92, edgecolor="white", label="Phase PR-AUC")
    ax.bar(x + 0.2, meas, 0.4, color=colors, alpha=0.45, edgecolor="white", label="Measurement PR-AUC")
    for xi, v in zip(x, phase):
        ax.text(xi - 0.2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    for xi, v in zip(x, meas):
        ax.text(xi + 0.2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[p] for p in policies], fontsize=9)
    ax.set_ylabel("PR-AUC (5-fold mean)")
    ax.set_title("Sampling-controlled comparison (same architecture)", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, 0.85)

    ax = axes[1]
    ax.bar(x, macs, 0.58, color=colors, alpha=0.9, edgecolor="white")
    for xi, v in zip(x, macs):
        ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[p] for p in policies], fontsize=9)
    ax.set_ylabel("MACs per measurement (log scale)")
    ax.set_title("Neural processing cost", fontsize=11.5)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    ax.bar(x - 0.2, sel_ms, 0.4, color=colors, alpha=0.75, edgecolor="white",
           label="Selection CPU p50")
    ax.bar(x + 0.2, gpu_ms, 0.4, color=colors, alpha=0.45, edgecolor="white",
           label="Model GPU b1 p50")
    for xi, v in zip(x, sel_ms):
        ax.text(xi - 0.2, v + 3, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x, gpu_ms):
        ax.text(xi + 0.2, v + 0.8, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[p] for p in policies], fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Selection and model latency", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Stage 2 sampling-controlled evidence: uniform / event / random / mixed K=8 and full-signal baselines", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig9_sampling_policy.png", dpi=200)

def main():
    fig1_architecture()
    fig2_window_sampling()
    fig3_pr_curves()
    fig4_reliability()
    fig5_examples()
    fig6_ablation()
    fig7_robustness()
    fig8_window_policy()
    fig9_sampling_policy()
    print("all figures done ->", FIG_DIR)


if __name__ == "__main__":
    main()
