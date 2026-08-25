# Cover Letter — IEEE Transactions on Industrial Informatics

> TII 版投稿封面信（英文）。与 TIM 版 cover letter 的差异：突出**工业 AI 部署**叙事（传感成本/标注成本/可信 AI），并披露 TIM 版存在（互引）。

---

[Date]

Dear Editor-in-Chief,

We are pleased to submit our manuscript entitled **"Cost-Efficient and Trustworthy Industrial AI for Partial Discharge Monitoring: Time-Frequency Encoding with Coverage-Aware Sampling and Hierarchical Weak Supervision"** for consideration as a Regular Paper in the *IEEE Transactions on Industrial Informatics*.

**Industrial AI contribution and fit with TII scope.** This paper treats the deployment of industrial AI for asset monitoring as a joint optimization of three cost axes that industrial practice must balance—*sensing cost* (hardware/ADC/bandwidth), *labeling cost* (expert annotation), and *trust cost* (evaluation credibility). We provide quantitative levers for each axis and validate them on real industrial PD measurement data:

1. **8× sensing-cost reduction.** A deterministic coverage-aware sampling plan processes only 8.2% of the raw data (K=8 windows of an 800,000-point, 40 MHz three-phase PD signal), and a sampling-rate study shows that reducing the sensing rate from 40 to 5 MHz retains 83% of detection performance (phase PR-AUC 0.617 → 0.510)—a direct hardware design guideline for ADC rate, storage, and transmission budgets in distribution-network monitoring.

2. **Time-frequency encoding with explicit efficiency trade-off.** An STFT+2D-CNN window encoder (167K parameters) lifts phase-level PR-AUC from 0.615 to **0.703 ± 0.025** under the same weakly supervised pipeline (attention MIL + context-concat three-phase interaction + noisy-OR). A lightweight patch Transformer (235K parameters) is benchmarked and reveals an empirical **data-scale hypothesis**: it underperforms on the 7.4K-sample VSB set (0.588) but becomes the best encoder on the 46K-sample external motor-PD dataset (from-scratch PR-AUC 0.998 vs 0.991 CNN)—practical guidance for industrial encoder selection.

3. **Labeling-cost quantification with honest negative result.** With only 50% of phase labels the weakly supervised design retains 88% of full-supervision performance (0.542 vs 0.615); a VICReg self-supervised pretraining study reports an honest negative result (no gain at 5–20% labels), refining when self-supervision helps and when it does not.

4. **Trustworthy evaluation and multi-dataset validation.** Hash-verified split locks, one-time blind tests with tamper-evident receipts, prevalence-normalized analytics (decomposing apparent PR-AUC gaps into prevalence vs domain-shift effects), measurement-clustered bootstrap CIs, and external motor-PD / cross-device validation (from-scratch ROC-AUC 0.99+) confirm credible, transferable industrial conclusions.

**Use of real industrial data.** All experiments use the VSB Power Line Fault Detection dataset (real covered-conductor distribution-line PD recordings) as the main industrial benchmark, complemented by the figshare motor-PD dataset (45,970 test samples) and cross-device oscilloscope captures (28523090, C1→C2), plus an independent held-out partition archived at Harvard Dataverse. We position the work within the IEC 60270 / IEC TS 62478 measurement framework as a waveform-level online-monitoring route.

**Commitments to the reviewers.** Full source code, experiment scripts, locked split files, evaluation receipts, and environment specification are publicly available at https://github.com/YldkS123/vsb-pd-foundation. All numbers originate from locked experimental records.

**Declarations.** This manuscript has not been published previously and is not under consideration elsewhere; all authors have read and approved the submission; there are no conflicts of interest.

We believe this work is well aligned with the scope of TII—industrial AI, condition monitoring, predictive maintenance, and trustworthy industrial informatics—and would be of interest to the community working on cost-aware and credible asset monitoring. Thank you for your consideration.

Sincerely,

[Author names and affiliations]
[Corresponding author, email, ORCID]

---

**Enclosures**: Manuscript (IEEEtran, double column), figures, code repository link.
