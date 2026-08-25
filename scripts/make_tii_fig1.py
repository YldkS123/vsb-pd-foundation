# -*- coding: utf-8 -*-
"""Fig.1 for the TII paper: framework overview with the three cost axes.

Pipeline: three-phase sensing -> coverage-aware sampling (CAS) ->
TFE/LPT window encoding -> attention MIL -> context-concat interaction ->
noisy-OR decision. Three cost axes annotated: sensing cost (sampling rate),
labeling cost (weak supervision), trust cost (leakage-safe evaluation).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures_tii"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10})


def box(ax, x, y, w, h, text, fc="#EAF2FB", ec="#2E5A88", fs=9.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color="#555555"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=14, lw=1.4, color=color)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # --- input stage ---
    box(ax, 0.2, 4.6, 1.7, 1.0, "Three-phase\nPD signals\n(3 × 800k @40 MHz)",
        fc="#FDF3E7", ec="#B5651D")
    box(ax, 0.2, 3.2, 1.7, 0.8, "Sampling rate\n40/20/10/5 MHz\n(design variable)",
        fc="#FDF3E7", ec="#B5651D", fs=8.5)

    # --- CAS ---
    box(ax, 2.4, 4.2, 1.9, 1.4, "Coverage-Aware\nSampling (CAS)\nK=8 windows\n(8.2% data)",
        fc="#E8F5E9", ec="#2E7D32")

    # --- encoder ---
    box(ax, 4.8, 4.2, 1.9, 1.4, "Window Encoder\nTFE (STFT+2D CNN)\nor LPT (Transformer)\n→ 128-d embeddings",
        fc="#E8F5E9", ec="#2E7D32")

    # --- MIL ---
    box(ax, 7.2, 4.2, 1.7, 1.4, "Attention MIL\n(window → phase)\nweak supervision",
        fc="#E3F2FD", ec="#1565C0")

    # --- interaction ---
    box(ax, 9.4, 4.2, 1.4, 1.4, "Context-concat\n3-phase\ninteraction",
        fc="#E3F2FD", ec="#1565C0")

    # --- decision ---
    box(ax, 9.4, 2.2, 1.4, 1.0, "Noisy-OR\nmeasurement\ndecision",
        fc="#F3E5F5", ec="#6A1B9A")

    # --- arrows main flow ---
    arrow(ax, 1.9, 5.1, 2.4, 5.1)
    arrow(ax, 4.3, 5.1, 4.8, 5.1)
    arrow(ax, 6.7, 5.1, 7.2, 5.1)
    arrow(ax, 8.9, 5.1, 9.4, 5.1)
    arrow(ax, 10.1, 4.2, 10.1, 3.2)
    # sampling rate feedback
    arrow(ax, 1.05, 3.2, 1.05, 4.6)

    # --- evaluation / trust block ---
    box(ax, 0.2, 0.3, 4.6, 1.4,
        "Leakage-Safe Evaluation\nSHA-256 split locks · one-time blind tests\nprevalence-normalized analytics · cluster bootstrap",
        fc="#FFF8E1", ec="#F57F17", fs=8.5)
    box(ax, 5.2, 0.3, 3.0, 1.4,
        "Multi-Dataset Validation\nVSB (main) · motor PD (external)\noscilloscope C1→C2 (cross-device)",
        fc="#FFF8E1", ec="#F57F17", fs=8.5)

    # --- cost axis annotations (left) ---
    ax.text(0.05, 5.9, "Cost axes:", fontsize=10, fontweight="bold", va="top")
    ax.annotate("Sensing cost\n↓ rate 40→5 MHz, 8× cheaper, 83% kept",
                xy=(1.05, 3.6), xytext=(0.1, 5.4),
                fontsize=8.2, color="#B5651D",
                arrowprops=dict(arrowstyle="->", color="#B5651D", lw=0.8))
    ax.annotate("Labeling cost\n↓ 50% labels, 88% kept",
                xy=(8.0, 5.5), xytext=(6.0, 5.9),
                fontsize=8.2, color="#1565C0",
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.8))
    ax.annotate("Trust cost\nleakage-safe protocol",
                xy=(2.5, 1.7), xytext=(4.2, 0.1),
                fontsize=8.2, color="#F57F17",
                arrowprops=dict(arrowstyle="->", color="#F57F17", lw=0.8))

    fig.tight_layout()
    fig.savefig(OUT / "fig1_framework.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig1 done ->", OUT / "fig1_framework.png")


if __name__ == "__main__":
    main()
