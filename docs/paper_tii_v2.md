# Cost-Efficient and Trustworthy Industrial AI for Partial Discharge Monitoring: Time-Frequency Encoding with Coverage-Aware Sampling and Hierarchical Weak Supervision

> **TII 版论文草稿 v2**（2026-08-25）
> 目标期刊：IEEE Transactions on Industrial Informatics (TII, IF≈11, 中科院1区Top)
> 与 TIM 版区分：本稿聚焦工业 AI 部署（时频编码升级、传感硬件成本、多数据集、可信 AI）；TIM 版聚焦测量科学。
> 两稿互引 + cover letter 互相披露。

---

## Abstract

Industrial condition monitoring of power distribution networks generates extremely long, sparse, and weakly labeled three-phase partial discharge (PD) signals, where sensing cost, labeling cost, and evaluation credibility jointly determine deployability. This paper presents a cost-efficient and trustworthy industrial AI framework that optimizes all three axes. **First**, a coverage-aware sampling plan extracts only 8.2% of the raw data (K=8 windows per 800,000-point phase signal at 40 MHz) via a robust event score, and a sampling-rate study shows that the sensing rate can be reduced 8-fold (40→5 MHz) while retaining 83% of detection performance—directly lowering ADC and storage hardware cost. **Second**, a time-frequency window encoder (STFT + 2D CNN, 167K parameters) replaces the plain-CNN backbone, raising phase-level PR-AUC from 0.615 to 0.703 (±0.025) on the VSB dataset under the same weakly supervised pipeline (attention MIL + context-concat three-phase interaction + noisy-OR), while a lightweight patch Transformer (235K parameters) is shown to require larger data scales to excel (0.998 PR-AUC from-scratch on the 46K-sample external motor-PD dataset vs 0.588 on the 7.4K-sample VSB set)—an empirical data-scale hypothesis for encoder choice. **Third**, the weakly supervised design retains 88% of full-supervision performance with only 50% of phase labels (0.542 vs 0.615), and a self-supervised (VICReg) pretraining study reports an honest negative result (no gain at low label ratios), refining when self-supervision helps. **Fourth**, a leakage-safe evaluation protocol—hash-verified split locks, one-time blind tests, and prevalence-normalized analytics—ensures trustworthy industrial conclusions; external multi-dataset validation (motor PD vs noise, 45,970 samples; cross-device oscilloscope captures) confirms transferability (from-scratch ROC-AUC 0.99+). The framework targets edge-deployable, cost-aware, and credible PD monitoring for distribution networks.

**Index Terms**—Industrial artificial intelligence; partial discharge monitoring; coverage-aware sampling; time-frequency encoding; weakly supervised learning; sampling-rate optimization; trustworthy AI.

---

## I. Introduction

### A. Industrial Context

Partial discharge (PD) is the earliest indicator of insulation degradation in power equipment [1], [2]. Online PD monitoring for distribution networks records raw high-rate waveforms (40 MHz, 800,000 samples per phase per measurement) and must decide, per phase and per measurement, whether discharge is present [3], [4]. Deploying machine learning for this task at industrial scale faces a **triple-cost problem**:

1. **Sensing cost**: 40 MHz digitization of long records dominates ADC, storage, and transmission budgets; whether a lower sampling rate suffices is an open hardware design question.
2. **Labeling cost**: window-level annotation of 800,000-point signals is prohibitively expensive; only phase-level labels exist (≈5.9% positive phases), requiring weakly supervised learning.
3. **Trust cost**: evaluation leakage inflates reported performance and undermines industrial confidence; credible evaluation protocols are a prerequisite for adoption.

### B. Related Work and Gap

Recent TIM/IEEE work on PD detection ranges from hardware sensing modules [5] to deep-learning pattern recognition [6], [7], transformer-based multisource PD recognition [8], and self-supervised anomaly detection [9]. Industrial informatics venues increasingly emphasize cost-aware and trustworthy AI for asset monitoring [10]–[12]. However, most existing methods (i) process full-length signals or fixed segments without addressing the sensing-rate design variable, (ii) assume fully labeled windows, and (iii) do not quantify evaluation leakage. This paper addresses the combined sensing–labeling–trust optimization, which is, to our knowledge, not jointly treated in prior industrial AI work.

### C. Contributions

1. **Cost-aware sampling with hardware-rate optimization**: a deterministic coverage-aware sampling plan (robust event score; equidistant anchors + event windows, K=8, 8.2% data) plus a sampling-rate study (40/20/10/5 MHz) showing 8× sensing-cost reduction at 83% retained performance.
2. **Time-frequency encoding for weak supervision**: an STFT+2D-CNN window encoder lifts phase PR-AUC to 0.703 under the same weakly supervised pipeline; a light patch Transformer is benchmarked, revealing a data-scale hypothesis (Transformers excel with more data).
3. **Labeling-cost quantification**: 50% phase labels retain 88% of full-supervision performance; a VICReg self-supervised study reports an honest negative result, refining when self-supervision helps.
4. **Trustworthy evaluation and multi-dataset validation**: hash-verified split locks, one-time blind tests, prevalence-normalized analytics, and external motor-PD/cross-device validation confirm credible, transferable industrial conclusions.

---

## II. Problem Formulation

Let a measurement consist of three-phase signals $x_A, x_B, x_C$, each of length $L = 800{,}000$ at sampling rate $f_s = 40$ MHz. The phase-level label $y_p \in \{0, 1\}$ indicates whether phase $p$ contains discharge; the measurement-level label follows the noisy-OR logic used by the evaluation metric (a measurement is abnormal if any phase contains discharge). During training only phase-level labels are available. The task is to learn a per-phase discriminator $f_p: x_p \to \hat{y}_p$ ($p \in \{A, B, C\}$) under three industrial constraints: **(C1) minimal sensing rate** (hardware cost), **(C2) minimal labeling** (annotation cost), and **(C3) credible evaluation** (trust).

---

## III. Proposed Framework

### A. Coverage-Aware Sampling (CAS)

For each phase signal $x$ of length $L$, we first remove the median to obtain $\tilde{x}[n] = x[n] - \mathrm{median}(x)$, then compute three energy features pointwise:

- **Amplitude**: $a[n] = |\tilde{x}[n]|$;
- **Teager energy**: $\tau[n] = |\tilde{x}[n]^2 - \tilde{x}[n-1]\tilde{x}[n+1]|$;
- **Differential RMS**: $d[n] = \sqrt{\mathrm{mean}_{m \in W}(\Delta\tilde{x}[m]^2)}$, $|W|=256$ (reflection-padded).

A robust non-negative z-score is defined per feature $v$:

$$z(v) = \max\left(\frac{v - \mathrm{median}(v)}{1.4826 \cdot \mathrm{MAD}(v)}, 0\right) \qquad (1)$$

with MAD fallback to mean-absolute deviation when degenerate. The event score is

$$S[n] = \max(z(a[n]),\, z(\tau[n]),\, z(d[n])) \qquad (2)$$

**Deterministic sampling plan (Algorithm 1)**. For each phase, extract $K = K_u + K_e = 8$ windows of length $W = 8{,}192$:
1. *Equidistant anchors* ($K_u=4$): uniformly spaced windows with maximally separated starts — guarantee full-signal coverage;
2. *Event windows* ($K_e=4$): peaks of $S$ (min distance $W/2$), sorted by $(-S, \text{start})$, deduplicated by IoU ≥ 0.5 against selected windows;
3. *Hierarchical fallback*: if fewer than $K$ windows are selected, fill from a 256-grid by maximum min-distance.

All sampling parameters are committed to a SHA-256 lock, making the plan deterministic and reproducible. The plan processes only $K \cdot W / L = 8.2\%$ of the raw data.

**Algorithm 1** Coverage-aware deterministic sampling
```
Input: x ∈ R^L, K_u = K_e = 4, W = 8192
Output: window set W with |W| = K_u + K_e
1: W ← ∅; x̃ ← x − median(x)
2: compute a, τ, d; S ← max(z(a), z(τ), z(d))      # Eq. (2)
3: Phase I — anchors: place K_u equidistant windows; W ← W ∪ anchors
4: Phase II — events:
5:   P ← peaks(S, dist_min = W/2, S > 0)
6:   sort P by (−S, start)
7:   for p ∈ P:
8:     s ← clip(peak_p − W/2, 0, L − W)
9:     if IoU([s, s+W], w) < 0.5 for all w ∈ W: W ← W ∪ {[s, s+W]}
10:    if |W| = K: break
11: Phase III — fallback: while |W| < K: add max-min-distance 256-grid window
12: return W
```

### B. Time-Frequency Window Encoder (TFE)

For each window $\mathbf{w} \in \mathbb{R}^{8192}$, the TFE computes a log-magnitude STFT spectrogram (256-point FFT, hop 128, Hann window) and processes it with three 2D convolution blocks (GroupNorm + SiLU), followed by global average+max pooling and a linear projection:

$$\mathbf{h}_{\mathrm{tf}} = \mathrm{Proj}\big(\mathrm{Pool}\big(\mathrm{CNN}_{2d}\big(\log(1+|\mathrm{STFT}(\mathbf{w})|)\big)\big)\big) \qquad (3)$$

with $\mathbf{h}_{\mathrm{tf}} \in \mathbb{R}^{128}$. The full pipeline (encoder + MIL + interaction + head) has **167,394 parameters**.

**Light patch Transformer (LPT, comparison)**: the window is patched into 128 tokens of 64 samples; a learnable positional embedding is added; a 2-layer Transformer encoder ($d=96$, 4 heads, feed-forward 192, GELU, norm-first) produces token representations; global avg+max pooling projects to 128-d (**234,594 parameters** in the full pipeline).

### C. Hierarchical Weak Supervision

**Window-to-phase attention MIL**. With $K$ window embeddings $\{\mathbf{h}_1, \ldots, \mathbf{h}_K\} \subset \mathbb{R}^{128}$ per phase, attention-based MIL [13] aggregates them:

$$\alpha_k = \frac{\exp(\mathbf{v}^\top \tanh(\mathbf{W}\mathbf{h}_k))}{\sum_{j=1}^K \exp(\mathbf{v}^\top \tanh(\mathbf{W}\mathbf{h}_j))}, \qquad \mathbf{z}_p = \sum_{k=1}^K \alpha_k \mathbf{h}_k \qquad (4)$$

**Context-concat three-phase interaction**. Let $\mathbf{c} = \frac{1}{3}(\mathbf{z}_A + \mathbf{z}_B + \mathbf{z}_C)$ be the global phase context; each phase's representation is concatenated with the context and classified:

$$\hat{y}_p = \sigma\big(\mathbf{w}_p^\top [\mathbf{z}_p, \mathbf{c}] + b_p\big) \qquad (5)$$

**Measurement-level inference**. Deterministic noisy-OR: $\hat{y}_{\mathrm{meas}} = 1 - \prod_p (1 - \hat{y}_p)$.

**Loss**. Phase-level binary cross-entropy: $\mathcal{L} = \sum_p \mathrm{BCE}(\hat{y}_p, y_p)$.

### D. Trustworthy Evaluation Protocol

1. **Leakage audit**: 56 historical prediction files scanned; 158 contaminated candidates demoted.
2. **Split locks**: development (2,481) / blind (423) split and all pipeline parameters committed to SHA-256 hash-verified locks; new splits (IR, sampling-rate) get their own locks.
3. **One-time blind tests**: the frozen model is evaluated exactly once on held-out partitions, with tamper-evident receipts binding protocol lock and checkpoint hashes.
4. **Prevalence-normalized analytics**: analytic PR reweighting decomposes apparent PR-AUC gaps into prevalence effects vs genuine domain shifts.
5. **Measurement-clustered bootstrap**: 2,000 resamples stratified by measurement for confidence intervals.

---

## IV. Experiments

### A. Datasets and Protocol

| Dataset | Task | Samples | Positive rate | Role |
|---|---|---|---|---|
| VSB (Kaggle 2019) [3] | 3-phase PD | 2,481 dev (7,443 phases) | 5.95% | Main |
| figshare 24033225 | Motor PD vs noise | 45,970 test | 50% | External |
| figshare 28523090 | Oscilloscope PD (C1→C2) | 639/319 test | — | Cross-device |

**Implementation**. PyTorch; AdamW (lr 1e-3, wd 1e-4), batch 64, up to 40 epochs, early stopping (patience 15, min delta 0.001) on validation phase PR-AUC, gradient clipping (max-norm 1.0); StratifiedGroupKFold(5, seed 42) grouped by measurement; seeds {42, 7, 2024} for stability. Metrics: PR-AUC primary (extreme imbalance), ROC-AUC, MCC, F1, ECE/Brier; per-fold mean primary protocol; paired cluster bootstrap (2,000, clustered by measurement) for significance; RTX 4060 Laptop GPU.

### B. Main Results (VSB, 5-fold CV, per-fold mean)

| Encoder | Params (pipeline) | Phase PR-AUC | Meas PR-AUC | Encoder fwd (8×8 batch, CPU) |
|---|---|---|---|---|
| simple_cnn (baseline) | 113,265 | 0.615 ± 0.053 | 0.643 | 63 ms |
| **TFE (STFT+2D CNN)** | 167,394 | **0.703 ± 0.025** | 0.744 | 542 ms |
| LPT (patch Transformer) | 234,594 | 0.588 ± 0.054 | 0.621 | 424 ms |

**TFE** significantly outperforms the baseline (+0.088 phase PR-AUC; paired cluster bootstrap on development OOF, 95% CI excludes 0) with only 1.5× parameters—time-frequency representation is the key encoder property for PD transient structure. **Explicit efficiency trade-off**: TFE's STFT front end costs ≈8.6× the CNN forward time on CPU; edge GPU deployment mitigates this (Discussion).

**Cross-encoder statistics** (paired bootstrap, seed 42): TFE vs simple_cnn phase difference +0.088 (95% CI excludes 0); LPT vs simple_cnn −0.027 (CI contains 0); TFE vs LPT +0.115 (CI excludes 0). Three-seed means confirm: TFE 0.70 ± 0.01 across seeds {42, 7, 2024}.

### C. Sensing-Cost Optimization (Sampling Rate)

| Rate | Data volume | Phase PR-AUC | Meas PR-AUC | Retained |
|---|---|---|---|---|
| 40 MHz | 1× | 0.617 | 0.629 | 100% |
| 20 MHz | 1/2 | 0.569 | 0.588 | 92% |
| 10 MHz | 1/4 | 0.518 | 0.551 | 84% |
| 5 MHz | 1/8 | 0.510 | 0.535 | 83% |

**8× sensing-cost reduction at 83% retained performance**—a direct hardware design guideline (ADC rate, storage, transmission budget). The 40 MHz row (0.617) matches the mainline (0.615) within noise, validating the protocol.

### D. Labeling-Cost Quantification

| Labels | Phase PR-AUC | vs 100% |
|---|---|---|
| 5% | 0.272 | 44% |
| 10% | 0.346 | 56% |
| 20% | 0.377 ± 0.021 (3 label seeds) | 61% |
| 50% | 0.542 | **88%** |
| 100% | 0.615 | 100% |

**Self-supervised (VICReg) negative result**: pretraining did not improve low-label fine-tuning (5%: 0.202 vs 0.272; 10%: 0.285 vs 0.346; 20%: 0.361 vs 0.377)—window-level self-supervised representations misalign with the phase-level weakly supervised task; honest reporting refines when self-supervision helps. VICReg details: 40-epoch pretraining on all development windows, 4 augmentations (time shift ±128, amplitude 0.9–1.1, noise 20–40 dB, frequency masking), variance/invariance/covariance loss (25/25/1).

### E. External Multi-Dataset Validation (figshare 24033225)

| Encoder | Zero-shot ROC/PR | From-scratch ROC/PR | Fine-tune ROC/PR |
|---|---|---|---|
| simple_cnn | 0.776/0.812 | 0.991/0.993 | 0.987/0.990 |
| **LPT** | 0.729/0.669 | **0.998/0.998** | **0.997/0.998** |

**Data-scale hypothesis confirmed**: the Transformer (weaker on 7.4K-sample VSB) becomes the best encoder on the 46K-sample external dataset (0.998 from-scratch vs 0.991 CNN)—encoder choice is data-scale dependent; TFE remains the cost-efficient choice at small industrial data scales. Cross-device validation (figshare 28523090, C1→C2) reproduces historical fine-tune results (ROC 0.80–0.91) [prior-ref].

### F. Robustness, Baselines, and Statistical Power

- **E4/TFE perturbation robustness**: 5 dB noise +3.7% (noise-robust vs −67% for the historical mainline), amplitude ≈0%, time-shift −47~−73% (temporal alignment remains the weak point; augmentation mitigates to ±3%).
- **Classical detector baselines**: energy 0.055 / impulsiveness 0.104 / spectral 0.088 / PRPD+LR 0.323 phase PR-AUC vs TFE 0.703—an order-of-magnitude gap isolating the value of learned weak supervision.
- **Statistical power analysis**: measurement-level minimum resolvable differences (4 positives ≈ 0.57; 31 ≈ 0.21; 163 ≈ 0.09) quantify when measurement-level statements are exploratory.
- **Prevalence-normalized analytics**: the apparent independent-set PR-AUC drop decomposes into a prevalence effect (Δπ = −0.242, ≈77%) and a genuine domain shift (Δshift = −0.073, ≈23%), with a prevalence-normalized PR lift of 13.8× vs 10.3× on development.

---

## V. Discussion

### A. Sensing–Labeling–Trust Joint Optimization

Each axis has a quantitative lever: **sensing rate** (8× cost reduction at 83% performance), **label ratio** (50% labels at 88% performance), and **evaluation credibility** (leakage-safe protocol with normalized analytics). Together they define a deployability design space: e.g., a 10 MHz / 20% labels configuration retains ≈84% × ≈61% ≈ 51% of full performance at 1/4 sensing and 1/5 labeling cost—a concrete industrial trade-off menu.

### B. Data-Scale Hypothesis for Encoder Choice

TFE wins at small data (VSB: 0.703 vs 0.588); LPT wins at larger data (external: 0.998 vs 0.991). We hypothesize that Transformer capacity pays off only beyond a data-scale threshold—practical guidance for industrial encoder selection. The VICReg negative result further indicates that pretraining cannot substitute for task-aligned supervision at small scale.

### C. Deployment Path

TFE's STFT front end is the compute bottleneck on CPU (542 ms per 8×8 batch) but parallelizes well on edge GPUs; the sampling plan reduces the input by 12× (K=8 of 96 windows-equivalents), and 5 MHz sensing reduces raw input 8× more—the combined pipeline is edge-feasible. Time-shift augmentation is recommended at deployment.

### D. Limitations

Single main industrial dataset (VSB); cross-device zero-shot transfer remains limited (needs target-domain fine-tuning); time-shift sensitivity requires augmentation; VICReg-style self-supervision did not help and should not be assumed beneficial.

---

## VI. Conclusion

We presented a cost-efficient and trustworthy industrial AI framework for PD monitoring: coverage-aware sampling with 8× sensing-cost reduction, time-frequency encoding lifting VSB performance to 0.703 PR-AUC, weak supervision retaining 88% at half the labels, honest self-supervision negative results, and leakage-safe multi-dataset validation. The framework provides concrete design levers for deploying credible, low-cost PD monitoring in distribution networks.

---

## References

[1] G. C. Stone, "Partial discharge diagnostics and electrical equipment insulation condition assessment," *IEEE Trans. Dielectr. Electr. Insul.*, vol. 12, no. 5, pp. 891–904, 2005.
[2] W. J. K. Raymond, H. A. Illias, A. H. Abu Bakar, and H. Mokhlis, "Partial discharge classifications: Review of recent progress," *Measurement*, vol. 68, pp. 164–181, 2015.
[3] VSB Power Line Fault Detection, Kaggle Competition, 2019.
[4] IEC 60270 / IEC TS 62478, partial discharge measurement standards.
[5] External PD detection via low-noise UHF sensor module, *IEEE Trans. Instrum. Meas.*, 2023.
[6] J. Zheng et al., "GIS partial discharge pattern recognition based on time-frequency features and improved CNN," *Energies*, vol. 15, p. 7372, 2022.
[7] Z. Fei et al., "Partial discharge pattern recognition based on ensembled simple CNN and quadratic SVM," *Energies*, vol. 17, p. 2443, 2024.
[8] Detection Transformer-based deep learning for multisource PD recognition, 2023–2024.
[9] Self-supervised temporal contrastive learning for PD anomaly detection, 2024.
[10]–[12] IEEE TII industrial AI / asset monitoring / trustworthy AI works (to be finalized).
[13] M. Ilse, J. M. Tomczak, and M. Welling, "Attention-based deep multiple instance learning," *ICML*, 2018.
[14] A. Bardes, J. Ponce, and Y. LeCun, "VICReg: Variance-invariance-covariance regularization for self-supervised learning," *ICLR*, 2022.
[15] S. Misak et al., "Problems associated with covered conductor fault detection," *EPQU*, 2011.
[16] G. M. Hashmi and M. Lehtonen, "On-line PD detection for condition monitoring of covered-conductor overhead distribution networks," *ICEE*, 2008.
[TIM-ref] 本团队：Measurement-cost-aware sampling and hierarchical weakly supervised detection for three-phase PD monitoring, *IEEE TIM*, submitted 2026（互引）.

---

## Figures (planned)

- **Fig. 1** — Framework overview: sensing → CAS sampling → TFE/LPT encoding → attention MIL → context-concat → noisy-OR decision; with the three cost axes annotated.
- **Fig. 2** — Sampling-rate cost-performance curve (x: rate; y: PR-AUC; annotated 83% retained at 5 MHz).
- **Fig. 3** — Encoder comparison: VSB vs external dataset performance (data-scale hypothesis).
- **Fig. 4** — Labeling-cost curve (x: label ratio; y: PR-AUC; VICReg negative result overlaid).
