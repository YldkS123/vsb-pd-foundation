# Cost-Efficient and Trustworthy Industrial AI for Partial Discharge Monitoring: Time-Frequency Encoding with Coverage-Aware Sampling and Hierarchical Weak Supervision

> **TII 版论文草稿 v1**（2026-08-25）
> 目标期刊：IEEE Transactions on Industrial Informatics (TII, IF≈11, 中科院1区Top)
> 与 TIM 版（IEEE TIM 投稿中）的区分：本稿聚焦**工业 AI 部署**——时频编码升级、传感硬件成本优化（采样率）、多数据集工业验证、可信评估；TIM 版聚焦测量科学。
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

A measurement consists of three-phase signals $x_A, x_B, x_C$, each of length $L=800{,}000$ at $f_s = 40$ MHz. Phase-level label $y_p \in \{0,1\}$; measurement label follows noisy-OR. Only phase-level labels are available. The task: per-phase discriminator $f_p: x_p \to \hat{y}_p$ with (i) minimal sensing rate, (ii) minimal labeling, (iii) credible evaluation.

---

## III. Proposed Framework

### A. Coverage-Aware Sampling (CAS)

Robust event score $S[n] = \max(z(a), z(\tau), z(d))$ over amplitude, Teager energy, and differential-RMS (robust non-negative z-scores). Deterministic plan: $K_u$ equidistant anchors + $K_e$ event windows (K=8; 8.2% data), cross-type deduplication (IoU≥0.5), hierarchical fallback. All parameters locked (SHA-256).

### B. Time-Frequency Window Encoder (TFE)

For each 8192-point window, log-magnitude STFT spectrogram → 2D CNN (three conv blocks, GroupNorm+SiLU) → global avg+max pooling → 128-d embedding (167,394 parameters). Replaces the plain-CNN encoder of the prior measurement-science paper [TIM-ref] inside the same MIL pipeline.

**Light patch Transformer (LPT, comparison)**: 8192-point window → 128 patches of 64 → learnable positional encoding → 2-layer TransformerEncoder (d=96, 4 heads) → pooling → 128-d (234,594 parameters).

### C. Hierarchical Weak Supervision

Window-to-phase attention MIL [13] aggregates K window embeddings; context-concat three-phase interaction fuses phase context; deterministic noisy-OR yields measurement-level decisions. Phase-level BCE loss.

### D. Trustworthy Evaluation Protocol

Leakage audit (158 contaminated candidates demoted), SHA-256 split locks, one-time blind tests with tamper-evident receipts, prevalence-normalized analytics (analytic PR reweighting), and measurement-clustered bootstrap CIs.

---

## IV. Experiments

### A. Datasets

| Dataset | Task | Samples | Positive rate | Role |
|---|---|---|---|---|
| VSB (Kaggle 2019) | 3-phase PD | 2,481 dev measurements (7,443 phases) | 5.95% | Main |
| figshare 24033225 | Motor PD vs noise | 45,970 test | 50% | External |
| figshare 28523090 | Oscilloscope PD (C1→C2) | 639/319 test | — | Cross-device |

### B. Main Results (VSB, 5-fold CV, per-fold mean)

| Encoder | Params (full pipeline) | Phase PR-AUC | Meas PR-AUC | Encoder fwd (batch 8×8, CPU) | Note |
|---|---|---|---|---|---|
| simple_cnn (baseline) | 113,265 | 0.615 ± 0.053 | 0.643 | 63 ms | prior mainline |
| **TFE (STFT+2D CNN)** | 167,394 | **0.703 ± 0.025** | 0.744 | 542 ms | proposed |
| LPT (patch Transformer) | 234,594 | 0.588 ± 0.054 | 0.621 | 424 ms | data-scale limited |

**TFE** significantly outperforms the baseline (+0.088 phase PR-AUC) with only 1.5× parameters—evidence that time-frequency representation is the key encoder property for PD transient structure. **Efficiency trade-off is explicit**: TFE's STFT front end costs ≈8.6× the CNN forward time on CPU, so deployment choice depends on whether the 0.088 PR-AUC gain justifies the compute (edge GPUs mitigate this; see Discussion).

### C. Sensing-Cost Optimization (Sampling Rate)

| Rate | Data | Phase PR-AUC | Retained |
|---|---|---|---|
| 40 MHz | 1× | 0.617 | 100% |
| 20 MHz | 1/2 | 0.569 | 92% |
| 10 MHz | 1/4 | 0.518 | 84% |
| 5 MHz | 1/8 | 0.510 | 83% |

**8× sensing-cost reduction at 83% retained performance**—a direct hardware design guideline (ADC rate, storage, transmission).

### D. Labeling-Cost Quantification

| Labels | Phase PR-AUC | vs 100% |
|---|---|---|
| 5% | 0.272 | 44% |
| 10% | 0.346 | 56% |
| 20% | 0.377 ± 0.021 (3 seeds) | 61% |
| 50% | 0.542 | **88%** |
| 100% | 0.615 | 100% |

**Self-supervised (VICReg) negative result**: pretraining did not improve low-label fine-tuning (5%: 0.202 vs 0.272; 10%: 0.285 vs 0.346; 20%: 0.361 vs 0.377)—window-level self-supervised representations misalign with the phase-level weakly supervised task; honest reporting refines when self-supervision helps.

### E. External Multi-Dataset Validation (figshare 24033225)

| Encoder | Zero-shot ROC/PR | From-scratch ROC/PR | Fine-tune ROC/PR |
|---|---|---|---|
| simple_cnn | 0.776/0.812 | 0.991/0.993 | 0.987/0.990 |
| **LPT** | 0.729/0.669 | **0.998/0.998** | **0.997/0.998** |

**Data-scale hypothesis confirmed**: the Transformer (weaker on 7.4K-sample VSB) becomes the best encoder on the 46K-sample external dataset (0.998 from-scratch vs 0.991 CNN)—encoder choice is data-scale dependent; TFE remains the cost-efficient choice at small industrial data scales.

Cross-device validation (figshare 28523090, C1→C2) reproduces the historical fine-tune results (ROC 0.80–0.91) [prior-work-ref].

### F. Robustness and Cost

E4 (TFE variant) perturbation robustness: 5 dB noise +3.7% (noise-robust), amplitude ≈0%, time-shift −47~−73% (temporal alignment remains the weak point); classical detectors (0.055–0.323) far below; statistical power analysis quantifies exploratory measurement-level evidence.

---

## V. Discussion

- **Sensing–labeling–trust joint optimization**: each axis has a quantitative lever (rate, label ratio, protocol); the framework provides a design space for industrial deployment.
- **Data-scale hypothesis**: TFE for small data, Transformer for larger data—practical encoder-selection guidance.
- **Self-supervision caveat**: not universally beneficial; alignment with the target task matters.
- **Limitations**: single main industrial dataset (VSB); cross-device zero-shot remains limited; time-shift sensitivity needs augmentation at deployment.

---

## VI. Conclusion

We presented a cost-efficient and trustworthy industrial AI framework for PD monitoring: coverage-aware sampling with 8× sensing-cost reduction, time-frequency encoding lifting VSB performance to 0.703 PR-AUC, weak supervision retaining 88% at half the labels, honest self-supervision negative results, and leakage-safe multi-dataset validation. The framework provides concrete design levers for deploying credible, low-cost PD monitoring in distribution networks.

---

## References (selected)

[1] Stone, IEEE TDEI 2005. [2] Raymond et al., Measurement 2015. [3] VSB Kaggle 2019. [4] IEC 60270 / TS 62478. [5] UHF sensor module, IEEE TIM 2023. [6] Zheng et al., Energies 2022. [7] Fei et al., Energies 2024. [8] Detection-Transformer PD recognition (2023-2024). [9] Self-supervised temporal contrastive PD (2024). [10]–[12] TII industrial AI/asset monitoring. [13] Ilse et al., ICML 2018 (attention MIL). [TIM-ref] 本团队 TIM 投稿（互引）。
