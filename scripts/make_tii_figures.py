# -*- coding: utf-8 -*-
"""Generate TII paper figures (sampling-rate curve, encoder comparison,
labeling-cost curve, framework overview placeholder).

Outputs (docs/figures_tii/):
  fig2_sampling_rate.png
  fig3_encoder_scale.png
  fig4_labeling_cost.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures_tii"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def fig2_sampling_rate():
    """Sampling-rate cost-performance curve."""
    rates = [40, 20, 10, 5]
    pr = [0.617, 0.569, 0.518, 0.510]
    meas = [0.629, 0.588, 0.551, 0.535]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rates, pr, "o-", lw=2, label="Phase PR-AUC")
    ax.plot(rates, meas, "s--", lw=2, label="Measurement PR-AUC")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{r} MHz" for r in rates])
    ax.set_xlabel("Sampling rate (log2 scale)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Sensing-Cost vs. Detection Performance")
    ax.annotate("83% retained\nat 1/8 data", xy=(5, 0.510), xytext=(8, 0.46),
                arrowprops=dict(arrowstyle="->"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig2_sampling_rate.png", dpi=300)
    plt.close(fig)
    print("fig2 done")


def fig3_encoder_scale():
    """Encoder comparison: data scale hypothesis (VSB 7.4K vs external 46K)."""
    datasets = ["VSB (7.4K)", "External (46K)"]
    cnn = [0.615, 0.991]      # simple_cnn fold-mean / from-scratch ROC
    tfe = [0.703, 0.993]      # TFE
    lpt = [0.588, 0.998]      # LPT
    x = np.arange(len(datasets))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w, cnn, w, label="simple_cnn", color="#4C72B0")
    ax.bar(x, tfe, w, label="TFE (STFT+2D CNN)", color="#55A868")
    ax.bar(x + w, lpt, w, label="LPT (Transformer)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Performance")
    ax.set_title("Encoder Choice Is Data-Scale Dependent")
    ax.legend()
    for xi, (c, t, l) in enumerate(zip(cnn, tfe, lpt)):
        ax.text(xi - w, c + 0.01, f"{c:.3f}", ha="center", fontsize=9)
        ax.text(xi, t + 0.01, f"{t:.3f}", ha="center", fontsize=9)
        ax.text(xi + w, l + 0.01, f"{l:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_encoder_scale.png", dpi=300)
    plt.close(fig)
    print("fig3 done")


def fig4_labeling_cost():
    """Labeling-cost curve with VICReg overlay."""
    fracs = [0.05, 0.10, 0.20, 0.50, 1.00]
    pr = [0.272, 0.346, 0.377, 0.542, 0.615]
    vicreg = [0.202, 0.285, 0.361, None, None]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fracs, pr, "o-", lw=2, label="Without pretraining (L1)")
    vx = [f for f, v in zip(fracs, vicreg) if v is not None]
    vy = [v for v in vicreg if v is not None]
    ax.plot(vx, vy, "s--", lw=2, color="#C44E52", label="With VICReg (negative result)")
    ax.set_xscale("log")
    ax.set_xticks(fracs)
    ax.set_xticklabels(["5%", "10%", "20%", "50%", "100%"])
    ax.set_xlabel("Labeled phase ratio")
    ax.set_ylabel("Phase PR-AUC")
    ax.set_title("Weak Supervision Retains Performance at Low Labeling Cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig4_labeling_cost.png", dpi=300)
    plt.close(fig)
    print("fig4 done")


if __name__ == "__main__":
    fig2_sampling_rate()
    fig3_encoder_scale()
    fig4_labeling_cost()
    print("All figures in", OUT)
