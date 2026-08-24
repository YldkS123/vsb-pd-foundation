# Measurement-Cost-Aware Sampling and Hierarchical Weakly Supervised Detection for Three-Phase Partial Discharge Monitoring

> **t4 working draft (v2, 2026-08)** — full revised English manuscript implementing the hard-error fixes (P0/P1, `reports/tim_revision_issues.md`) and the TIM-style rewrite (`reports/tim_gap_analysis.md`, `reports/tim_revision_plan.md`). All numerical values originate from locked experimental records (`results/stage1_tim/report_A_E.md`, `results/stage2_sampling/report_sampling.md`, `results/baseline_full_comparison.json`) and locked implementation (`src/vsb_pd/encoder.py`, `mil.py`, `cyclic.py`, `scripts/stage1_tim_runner.py`). Revision-status notes and internal paths in this blockquote are for team use only and MUST be moved to the cover letter before submission. Placeholders marked [t3] await supplementary experiments.

## Abstract

Partial discharge (PD) monitoring of distribution networks produces extremely long, sparse three-phase measurement signals: each phase is sampled at 40 MHz with 800,000 points per measurement, discharge transients occupy only brief intervals, and labels are available only at the phase level (approximately 5.9% positive) because window-level annotation is prohibitively expensive. This paper presents a measurement-cost-aware framework that jointly optimizes sampling, labeling, and detection. First, we define a robust event score (non-negative z-scores of amplitude, Teager energy, and differential RMS) and a deterministic coverage-aware sampling plan that extracts only K = 8 short windows per phase (8.2% of raw data) via equidistant anchors and event windows with cross-type deduplication. Second, a hierarchical weakly supervised detector maps windows to per-phase probabilities through a lightweight CNN encoder (113,265 total parameters; 97,201 executed at inference), attention-based multiple instance learning, and context-concat three-phase interaction, followed by deterministic noisy-OR aggregation to the measurement level. On the VSB dataset (2,481 development measurements), the mainline achieves per-phase PR-AUC 0.615 ± 0.053 and measurement-level 0.643 ± 0.053 on 5-fold cross-validation, an absolute gain of +0.409 over the strongest traditional baseline, at 62.8M MACs and 5.35 ms GPU batch-1 forward latency. A one-time blind test on an independent held-out partition (83,233 phase signals) yields ROC-AUC 0.899, with a prevalence-normalized PR lift of 13.8× versus 10.3× on development. A hash-verified split lock and one-time evaluation protocol ensure leakage-free, reproducible conclusions.

**Index Terms** — Partial discharge measurement; coverage-aware sampling; weakly supervised learning; multiple instance learning; three-phase power distribution; leakage-safe evaluation; measurement cost.

---

## I. Introduction

Partial discharge (PD) is an early indicator of insulation degradation in power equipment, and its accurate detection is a cornerstone of condition-based maintenance for distribution networks [1], [2]. PD diagnosis has become an active measurement topic in this journal and related venues, including inception-voltage prediction for electric-vehicle motor insulation [3], CWT-based defect classification in medium-voltage switchgear [4], and generative zero-shot diagnosis in gas-insulated switchgear [5]; deep-learning approaches to discharge pattern recognition include recurrent networks [6], time-frequency CNNs [7], and ensemble lightweight CNNs [8].

In practical online monitoring, however, PD measurement faces a joint cost problem at three levels:

1. **Sampling cost.** Signals are digitized at 40 MHz, and a single phase of one measurement contains 800,000 points. Discharge transients are highly sparse in time—in the VSB dataset used in this paper only about 5.9% of phase signals carry a discharge label—so strongly supervised modeling of entire signals wastes most computation on noise-dominated frames and is easily overwhelmed by them.
2. **Labeling cost.** Only phase-level labels (whether a phase contains discharge) or measurement-level labels (whether the three-phase signal as a whole is abnormal) are available; window-level annotations are unavailable and prohibitively expensive to obtain. Weakly supervised learning must therefore replace window-level supervision, a setting shared with recent semi-supervised measurement-condition assessment work [18], [19].
3. **Evaluation cost and credibility.** Most published methods do not explicitly control data leakage: if segments of the same measurement inadvertently appear in both training and test sets, results are inflated and irreproducible. Leakage has been formalized by Kaufman et al. [23] and identified as a major source of the reproducibility crisis in machine-learning-based science [24].

Existing PD detection methods mostly model fixed-length segments or PRPD images and rarely ablate how windows are selected—the decisive measurement-strategy step—nor quantify the labeling cost of their supervision requirements. In the terminology of the instrumentation and measurement community, the information-selection step at the front end of the measurement pipeline (where to place processing resources in an extremely long sparse signal) is a measurement design decision, akin to adaptive sampling [20] and event-controlled sampling [21], yet it is seldom treated as a first-class design variable in PD detection.

To address these gaps, this paper treats sampling, labeling, and evaluation as jointly optimized measurement design variables and makes the following contributions:

1. **Coverage-aware measurement sampling.** We provide a complete mathematical definition of a robust event score and a deterministic sampling plan ("equidistant anchors + event windows + cross-type deduplication + hierarchical fallback") that extracts only K = 8 short windows from each 800,000-point phase signal, processing about 8.2% of the raw data. Controlled experiments with a fixed architecture show that event focus and coverage density jointly determine the diagnostic-performance versus computational-cost trade-off—information selection itself is a critical design variable, not the network structure.
2. **Hierarchical weakly supervised detection with low labeling cost.** Under phase-level-only labels, windows are aggregated to phases by attention-based multiple instance learning (MIL) and fused across phases by a context-concat interaction that outputs independent, rankable per-phase probabilities; a deterministic noisy-OR yields measurement-level decisions. The lightweight mainline (pure CNN encoder + attention MIL + context-concat, 113,265 total parameters / 97,201 executed at inference) reaches per-phase PR-AUC 0.615 ± 0.053 and measurement-level 0.643 ± 0.053 on 5-fold development cross-validation, with statistically significant superiority over shared-score and additive-interaction baselines.
3. **Measurement-oriented cost characterization.** On a unified platform we report per-component latency, MACs, memory, and throughput for the sampling stage and the model forward pass separately, explicitly linking diagnostic performance, information coverage, and computational cost, and avoiding overclaims such as real-time end-to-end detection.
4. **Leakage-safe, reproducible evaluation.** Data-leakage auditing, hash-verified split locks, a one-time blind-test protocol, and a frozen independent held-out evaluation (83,233 phase signals; ROC-AUC 0.899) ensure that all model selection, hyperparameters, and thresholds are determined solely by development-set cross-validation and that conclusions are reproducible and leakage-free.

The remainder of this paper is organized as follows. Section II introduces the measurement background, the data, and the problem formulation. Section III presents the coverage-aware sampling strategy. Section IV describes the hierarchical weakly supervised detection framework. Section V reports experiments, including ablation, robustness, efficiency, and the independent test. Section VI analyzes statistical uncertainty and metric protocols. Section VII discusses module contributions and limitations. Section VIII concludes.

---

## II. Measurement Background and Problem Formulation

### A. PD Measurement Context

Conventional PD measurement according to IEC 60270 [12] quantifies apparent charge under controlled laboratory conditions at power frequency. In the field, non-conventional methods such as UHF, TEV, and acoustic sensing (IEC TS 62478 [13]) trade sensitivity and localization for installation cost. A third route—used by online condition-monitoring systems for covered-conductor distribution lines [9], [10]—records the raw high-rate waveform itself and performs detection on the digitized signal. This paper follows the waveform-level route: the measurement stream is a three-phase record at 40 MHz, and the detection system must decide, per phase and per measurement, whether PD is present. The front-end question is where to place processing resources within an extremely long, sparse signal; the back-end question is how to learn from phase-level labels only; and the evaluation question is how to guarantee that reported numbers are not inflated by leakage. These three questions define the scope of this work.

### B. Measurement Data

The VSB Power Line Fault Detection dataset [11] provides 2,904 three-phase measurements (each phase 800,000 signed samples at 40 MHz) from covered-conductor distribution lines. After a systematic data-leakage audit of 56 historical prediction files (158 contaminated candidates demoted), the split lock assigns 2,481 measurements to the development set and 423 strictly held-out measurements to the blind test set. The development set contains 7,443 phase signals with 443 positives (5.95%); the blind test set contains 1,269 phase signals with 82 positives (6.5%) and 31 positive measurements (7.3%). In addition, an independent partition of the same project on Harvard Dataverse (doi:10.7910/DVN/JYJJ5W) provides 83,233 phase signals (28,285 measurements; 1,308 positives, 1.57%) that were never used in any training, selection, or tuning step and were evaluated exactly once by the frozen final model (Section V-J).

**Relation to the original Kaggle competition.** The VSB competition scored submissions by Matthews correlation coefficient (MCC) on a private test set whose labels were never released, so leaderboard numbers cannot be compared point-to-point with the PR-AUC protocol used here; the 423-measurement blind set of this paper is a locked, label-annotated strict hold-out, not the competition test set. We therefore use public competition solutions only as discussion anchors: the leading entries relied on feature engineering with tree ensembles and—most notably—on the measurement-ordering / neighborhood structure of the data (adjacent measurements along the line are spatially correlated), a prior that was effective on the leaderboard but is unavailable in single-measurement online decision making and was at the center of the leaderboard shake-up discussion. This paper deliberately evaluates per measurement with a leakage-safe protocol and does not use any ordering or neighborhood prior; the contrast (rank-based priors versus leakage-safe per-measurement evaluation) is a measurement-methodology point discussed in Section V-G.

### C. Problem Formulation and Leakage-Safe Evaluation Principles

Let a measurement consist of three-phase signals $x_A, x_B, x_C$, each of length $L = 800{,}000$ at sampling rate $f_s = 40$ MHz. The phase-level label $y_p \in \{0, 1\}$ indicates whether phase $p$ contains discharge; the measurement-level label follows the noisy-OR logic used by the evaluation metric (a measurement is abnormal if any phase contains discharge). During training only phase-level labels are available. The task is to learn a per-phase discriminator $f_p: x_p \rightarrow \hat{y}_p$ ($p \in \{A, B, C\}$) from which measurement-level probabilities are derived deterministically. Because the final interaction module outputs independent, rankable per-phase probabilities, "phase-level PR-AUC" in this paper refers to the ranking evaluation of per-phase probabilities, and measurement-level decisions are obtained by hierarchical inference.

To prevent any form of evaluation leakage, the experimental pipeline is explicitly frozen: (1) a leakage audit identifies and demotes contaminated candidates; (2) a hash-verified split lock (SHA-256) fixes the folds and the blind-test partition, and all pipeline parameters are committed to the lock; (3) all model selection, hyperparameters, and thresholds are determined solely from development-set cross-validation; (4) the 423 blind-test measurements were evaluated exactly once by the historical locked mainline and are not reopened for the architecture upgrade reported here; (5) the independent held-out partition was evaluated exactly once by the frozen final model, with a tamper-evident receipt. We note that SHA-256 guarantees integrity verification (content changes are detectable) and is not claimed to be cryptographically tamper-proof.

---

## III. Coverage-Aware Measurement Sampling

For each phase signal, $K$ windows of length $W = 8{,}192$ samples (204.8 $\mu$s) are extracted deterministically: $K_u$ equidistant anchor windows ensure full coverage of the signal, and $K_e$ event windows focus on suspected discharge regions selected by the event score defined below.

### A. Event Score

For each phase signal $x$ (length $L = 800{,}000$), we first subtract the median to obtain $\tilde{x}[n] = x[n] - \mathrm{median}(x)$, then compute three energy features pointwise:

- **Amplitude**: $a[n] = |\tilde{x}[n]|$;
- **Teager energy**: $\tau[n] = |\tilde{x}[n]^2 - \tilde{x}[n-1]\tilde{x}[n+1]|$;
- **Rolling differential RMS**: $d[n] = \sqrt{\mathrm{mean}_{m \in \mathrm{window}}(\Delta\tilde{x}[m]^2)}$, where $\Delta\tilde{x}$ is the first-difference sequence (leading edge padded by reflection) and the window width is 256.

The robust non-negative z-score is defined as

$$z(v) = \max\left(\frac{v - \mathrm{median}(v)}{1.4826 \cdot \mathrm{MAD}(v)},\ 0\right) \qquad (1)$$

where $\mathrm{MAD}(v) = \mathrm{median}(|v - \mathrm{median}(v)|)$; when MAD degenerates (zero or non-finite), it falls back to the mean absolute deviation scale. The event score is

$$S[n] = \max\left(z(a[n]),\ z(\tau[n]),\ z(d[n])\right), \qquad (2)$$

with $S[n] \geq 0$. The three features respond to different transient characteristics (magnitude, instantaneous energy, and differential energy), and the robust z-score makes the score scale-invariant under noise and dynamic-range variation.

### B. Deterministic Sampling Plan

The coverage-aware sampling procedure is summarized in Algorithm 1. Phase I places $K_u$ uniformly spaced windows with maximally separated starting points. Phase II selects peaks of $S$ (minimum distance $W/2$, $S > 0$), sorts them by $(-S, \text{start})$, and adds windows centered at the peaks unless they overlap an already selected window by IoU $\geq 0.5$ (cross-type deduplication). Phase III guarantees exactly $K$ windows by a fallback that maximizes the minimum distance to the selected set on a 256-bin grid.

**Algorithm 1** Coverage-Aware Deterministic Window Sampling
---

**Require:** Phase signal $x \in \mathbb{R}^{L}$ ($L = 800{,}000$), $K_u = K_e = 4$, window length $W = 8{,}192$

**Ensure:** Window set $\mathcal{W}$ with $|\mathcal{W}| = K_u + K_e = 8$

1: $\mathcal{W} \gets \emptyset$
2: $\tilde{x}[n] \gets x[n] - \mathrm{median}(x)$, $n = 1, \dots, L$
3: Compute amplitude $a[n] \gets |\tilde{x}[n]|$
4: Compute Teager energy $\tau[n] \gets |\tilde{x}[n]^2 - \tilde{x}[n-1]\tilde{x}[n+1]|$
5: Compute differential RMS $d[n]$ (window 256, reflection padding)
6: $S[n] \gets \max\{z(a[n]), z(\tau[n]), z(d[n])\}$ ▷ Eq. (2)
7: **Phase I — Equidistant anchors:** place $K_u$ uniformly spaced, maximally separated windows; add to $\mathcal{W}$
8: **Phase II — Event windows:**
9:  $P \gets$ peaks of $S$ with $\mathrm{dist}_{\min} = W/2$ and $S > 0$
10:  sort $P$ by $(-S, \mathrm{start})$ descending
11:  **for** $p \in P$ **do**
12:   $s \gets \mathrm{clip}(\mathrm{peak}_p - W/2,\ 0,\ L - W)$
13:   **if** $\mathrm{IoU}([s, s+W], w) < 0.5$ for all $w \in \mathcal{W}$ **then**
14:    $\mathcal{W} \gets \mathcal{W} \cup \{[s, s+W]\}$
15:   **end if**
16:   **if** $|\mathcal{W}| = K_u + K_e$ **then break**
17:  **end for**
18: **Phase III — Fallback:** **while** $|\mathcal{W}| < K_u + K_e$ **do**
19:  select position from the 256-bin grid maximizing $\min_{w \in \mathcal{W}} \mathrm{dist}(w)$; add to $\mathcal{W}$
20: **end while**
21: **return** $\mathcal{W}$

---

The locked parameters are: $\mathrm{rolling\_width} = 256$, $\mathrm{peak\_distance} = 4{,}096$, $\mathrm{window\_length} = 8{,}192$, $\mathrm{dedup\_IoU} = 0.5$, $\mathrm{fallback\_grid} = 256$ ($K_u = 4$, $K_e = 4$). The procedure is implemented in `src/vsb_pd/events.py` and committed to the split lock via SHA-256, so identical input always produces identical output. Window-strategy ablations (K and composition) are presented in Section V-E. Fig. 2 illustrates the mixed sampling result on a real signal (measurement 705, phase C).

![Fig. 2 Mixed window sampling example (measurement 705, phase C, with window annotations)](figures/fig2_window_sampling.png)

### C. Physical Feature Extraction

For each window, a 58-dimensional physical feature vector is extracted as input to the encoder's feature branch (used by the reference architecture; see Section IV-A), computed entirely in a vectorized manner:

| Category | Dimensions | Description |
|----------|-----------|-------------|
| Time-domain | 20 | Amplitude, variance, skewness, kurtosis, crest factor, zero-crossing rate, energy, etc. |
| Frequency-domain | 12 | Dominant frequency, spectral centroid, bandwidth, roll-off, flatness, spectral entropy, etc. (FFT) |
| Band energy | 13 | Normalized energy across 13 equally spaced bands from 0–20 MHz |
| Autocorrelation/AR | 9 | First 3 ACF peaks and their significance + 3rd-order Burg AR coefficients |
| Envelope/peaks | 4 | Number of peaks, mean significance, Hilbert envelope mean/std |

Feature extraction is verified for identity and finiteness on constant and random signals.

### D. Relation to Adaptive and Event-Triggered Sampling

Adaptive sampling [20] and event-controlled sampling [21] increase measurement density only where the signal changes significantly, reducing data volume, power, and processing cost. Our strategy performs the analogous information selection at the processing front end of the detection pipeline: uniform segmentation is simple but insensitive to sparse events, while purely event-driven segmentation depends on detector quality. The hybrid plan combines both—equidistant anchors guarantee coverage, event windows focus on high-response regions, and cross-type deduplication avoids redundancy.

---

## IV. Hierarchical Weakly Supervised Detection

The detection framework (Fig. 1) comprises four components: coverage-aware window sampling (Section III), a lightweight window encoder, attention-based MIL aggregation, and three-phase interaction with hierarchical inference.

![Fig. 1 Coverage-aware multi-window sampling and hierarchical weakly supervised detection framework](figures/fig1_architecture.png)

### A. Window Encoder

Each window is first normalized per window by a robust standardization

$$\mathbf{w}_{\mathrm{norm}} = \frac{\mathbf{w} - \mathrm{median}(\mathbf{w})}{\mathrm{IQR}(\mathbf{w}) / 1.349}, \qquad (3)$$

where IQR is the interquartile range and the factor 1.349 maps IQR to the standard-deviation scale of a Gaussian (the scale is floored at $10^{-8}$ to avoid division by zero). This preprocessing is applied identically to every encoder in the matched-encoder comparison (Section V-D).

The final lightweight mainline uses a pure 1D CNN window encoder built from four depthwise-separable convolutional blocks. Each block is a depthwise convolution followed by a pointwise convolution ($\mathrm{DSConv}_{C,K}$: output channels $C$, kernel $K$, stride 4, padding $K//2$), then GroupNorm and SiLU activation:

$$\mathbf{h}_1 = \mathrm{SiLU}(\mathrm{GN}(\mathrm{DSConv}_{32,15}(\mathbf{w}_{\mathrm{norm}}))) \qquad (4)$$
$$\mathbf{h}_2 = \mathrm{SiLU}(\mathrm{GN}(\mathrm{DSConv}_{64,11}(\mathbf{h}_1))) \qquad (5)$$
$$\mathbf{h}_3 = \mathrm{SiLU}(\mathrm{GN}(\mathrm{DSConv}_{96,7}(\mathbf{h}_2))) \qquad (6)$$
$$\mathbf{h}_4 = \mathrm{SiLU}(\mathrm{GN}(\mathrm{DSConv}_{128,5}(\mathbf{h}_3))) \qquad (7)$$

Global average pooling and global max pooling are concatenated and linearly projected:

$$\mathbf{h}_{\mathrm{raw}} = \mathrm{LN}\left(\mathbf{W}_p\,[\mathrm{AvgPool}(\mathbf{h}_4);\ \mathrm{MaxPool}(\mathbf{h}_4)]\right), \qquad (8)$$

with $\mathbf{W}_p \in \mathbb{R}^{128 \times 256}$ and LayerNorm (LN), giving $\mathbf{h}_{\mathrm{raw}} \in \mathbb{R}^{128}$. The mainline uses the CNN branch only. The reference architecture (203,634 parameters) uses a dual-branch fusion in which the same CNN branch (channels 32 → 64 → 96 → 128) is combined with a physical-feature MLP (58 → 128 → 64; three linear layers including the fusion projection) via learned gating; it serves only as mechanism verification (Section V-D) and is not part of the main report.

> **Parameter accounting.** The mainline checkpoint contains 113,265 parameters in total. Of these, 16,064 belong to a physical-feature MLP that is constructed by the shared encoder class but not executed in the CNN-only configuration; the number of parameters executed at inference is therefore 97,201. Similarly, the historical locked mainline (80,113 total) executes 64,049 parameters. All efficiency claims in this paper refer to executed parameters and measured MACs; the total count is reported for checkpoint fidelity.

### B. Window-to-Phase MIL Aggregation

Multiple instance learning assumes labels exist only at the bag level, where each bag consists of several instances [14], [15]; attention-based deep MIL automatically identifies key instances within a bag [16], and recent work extends MIL to hierarchical time series classification [17]. Our phase-level modeling is exactly this setting: windows are instances and phases are bags—windows lack labels while phases have labels, so training uses bag-level labels only. Unlike standard MIL, instances are not independent samples but are deterministically constructed by the coverage-aware plan (Section III), so information selection and weak supervision are directly coupled.

For a phase signal with $K$ windows we obtain $K$ representations $\{\mathbf{h}_1, \dots, \mathbf{h}_K\}$, each in $\mathbb{R}^{128}$. The mainline aggregates windows into a phase-level representation by attention MIL [16], using a linear scaled scoring function:

$$\alpha_k = \frac{\exp\left((\mathbf{w}^\top \mathbf{h}_k + b)/\sqrt{D}\right)}{\sum_{j=1}^{K} \exp\left((\mathbf{w}^\top \mathbf{h}_j + b)/\sqrt{D}\right)} \qquad (9)$$

$$\mathbf{z}_p = \sum_{k=1}^{K} \alpha_k \mathbf{h}_k, \qquad (10)$$

where $\mathbf{w} \in \mathbb{R}^{128}$, $b \in \mathbb{R}$, and $D = 128$; the $\sqrt{D}$ scaling stabilizes the softmax. This linear scaled variant differs from the gated (tanh) attention of [16] but shares its idea of learning instance importance; the reference architecture uses the gated variant with sigmoid gating [16].

### C. Three-Phase Interaction and Hierarchical Inference

Three-phase signals carry exploitable redundancy (a phase is rarely affected in isolation; see Section V-F and VII-A). The mainline uses **context-concat** interaction. Let $\mathbf{z}_A, \mathbf{z}_B, \mathbf{z}_C \in \mathbb{R}^{128}$ be the MIL-aggregated representations. The global context is

$$\mathbf{c} = \frac{1}{3}\left(\mathbf{z}_A + \mathbf{z}_B + \mathbf{z}_C\right). \qquad (11)$$

For each phase $p$, the concatenation $[\mathbf{z}_p; \mathbf{c}] \in \mathbb{R}^{256}$ is projected by a two-layer context projection

$$\mathbf{z}_p' = \mathrm{SiLU}\left(\mathrm{LN}\left(\mathbf{W}_c\,[\mathbf{z}_p; \mathbf{c}]\right)\right), \qquad (12)$$

with $\mathbf{W}_c \in \mathbb{R}^{128 \times 256}$, and the per-phase probability is produced by a classifier shared across phases:

$$\hat{y}_p = \sigma\left(\mathrm{MLP}(\mathbf{z}_p')\right), \qquad (13)$$

where $\mathrm{MLP}$ is Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 1) and $\sigma$ is the sigmoid. Because each phase has a distinct $\mathbf{z}_p'$, the three outputs are independent and rankable per-phase probabilities (the classifier weights are shared). The measurement-level probability is the deterministic noisy-OR:

$$\hat{y}_{\mathrm{meas}} = 1 - \prod_{p \in \{A,B,C\}} \left(1 - \hat{y}_p\right). \qquad (14)$$

Compared variants in the interaction ablation (Section V-C): **E1**, no interaction (per-phase vectors pass to the classifier unchanged); **E2**, mean shared-score ($\mathbf{z}_p' = \frac{1}{3}\sum_{q \in \{A,B,C\}} \mathbf{z}_q$, the elementwise mean across the three phase vectors, broadcast to every phase); **E3**, max aggregation ($\mathbf{z}_p' = \max_{q \in \{A,B,C\}} \mathbf{z}_q$, the elementwise maximum across phases, broadcast to every phase); **E5**, context-add ($\mathbf{z}_p' = \mathbf{z}_p + \phi(\mathbf{c})$, where $\phi$ is an MLP: Linear(128, 128) → LayerNorm → SiLU); **E6**, E4 plus a hierarchical measurement-level BCE loss $\mathcal{L} = \mathcal{L}_{\mathrm{phase}} + \lambda \mathcal{L}_{\mathrm{meas}}$, with $\lambda$ selected on the development set. Note that E2 and E3 broadcast a single vector to all three phases, so their per-phase outputs are identical across phases and cannot support per-phase fault localization (Section V-C); E1 keeps distinct per-phase vectors but performs no interaction (the weakest configuration), and among the interaction variants E2–E6, E4 is the only one that both interacts and outputs phase-distinct scores.

### D. Training Protocol and Implementation Details

All models are trained with AdamW (learning rate $1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$), batch size 64 (8 for the full-signal baseline), for up to 40 epochs with early stopping (patience 15, min delta 0.001) on validation phase-level PR-AUC, and gradient clipping at max-norm 1.0. Training uses stratified 5-fold cross-validation (StratifiedGroupKFold grouped by measurement, stratified by number of positive phases, seed 42; seeds 7 and 2024 for stability verification) with a shared fold fingerprint and a shared K=8 feature cache. The loss is phase-level binary cross-entropy for all variants except E6, which adds a measurement-level BCE term; for E6, $\lambda \in \{0.25, 0.5, 1.0\}$ is selected solely on the seed-42 development pooled-OOF measurement PR-AUC (0.512 / 0.430 / 0.492), locking $\lambda = 0.25$ before seeds 7 and 2024 are evaluated. Training runs on CUDA. All encoders, MIL variants, and sampling strategies in this paper share this protocol; latency figures are p50 of 50 repetitions of synthetic 800k three-phase signals (RTX 4060 Laptop GPU, same host CPU); Stage-1 and Stage-2 benchmarks were measured in different sessions and are never compared across tables as point estimates. Implementation uses PyTorch [30], scikit-learn [31], pandas [32], matplotlib [33], and NumPy [34].

---

## V. Experiments

### A. Dataset and Evaluation Protocol

**Development set and blind test.** As described in Section II-B, all experiments in this paper use the development set (2,481 measurements; 7,443 phases; 443 positives, 5.95%). The 423 blind-test measurements (1,269 phases; 82 positives, 6.5%; 31 positive measurements, 7.3%) were evaluated exactly once by the historical locked mainline and are not reopened.

**Independent held-out set.** The Harvard Dataverse partition (83,233 phase signals; 28,285 measurements; 1,308 positives, 1.57%) was downloaded under a pre-download hash lock, never used in any training or selection step, and evaluated exactly once by the frozen final E4 model (Section V-J).

**Evaluation matrix.** To avoid confusion about where each model was validated, Table I summarizes the evaluation footprint. In addition, an internal-reserved (IR) strict hold-out evaluation was performed for the E4 mainline: 20% of the development measurements (497 measurements; 67 positive phases, 4.49%) were stratified into an IR set under a new hash-locked split (SHA-256 `ir_split_lock.json`), the E4 mainline was retrained on the remaining 80% under the exact mainline protocol, and the IR set was evaluated exactly once with thresholds frozen from the 80% training-domain OOF (Section V-K; scripts `e2_ir_evaluation.py`; outputs `results/ir_eval/`).

**TABLE I** EVALUATION FOOTPRINT OF THE REPORTED MODELS

| Model | Development 5-fold CV | IR strict hold-out | 423 blind test | Harvard held-out |
|-------|----------------------|--------------------|----------------|------------------|
| Historical locked mainline (80,113 total / 64,049 executed params, mean interaction) | ✓ (OOF + per-fold evidence) | — | ✓ evaluated once, frozen receipt | — |
| E4 mainline (113,265 total / 97,201 executed params, context-concat) | ✓ (5-fold, 3 seeds) | ✓ evaluated once, frozen thresholds (PR-AUC 0.412 phase / 0.415 meas; ROC-AUC 0.953 / 0.951) | ✗ not evaluated (one-time protocol) | ✓ evaluated once, frozen receipt |

**Metrics.** The primary metric is PR-AUC (phase- and measurement-level), supplemented by ROC-AUC, MCC, F1, ECE (expected calibration error), and Brier score; per-fold mean ± standard deviation across 5 folds is the primary reporting protocol (Section VI-A). Statistical significance is assessed by paired cluster bootstrap (2,000 resamples, clustered by measurement ID) with 95% confidence intervals; three-seed means (seeds 42, 7, 2024) provide stability verification. PR-AUC is preferred over ROC-AUC under extreme class imbalance because it reflects precision at low positive rates [22].

### B. Classical Detector Baselines and Labeling-Cost Analysis

To anchor the comparison in classical measurement practice, we evaluate three classical detectors with no learned parameters—an energy-threshold detector, an impulsiveness detector (crest factor + kurtosis), and a spectral detector—together with a PRPD-style classical-ML reference (logistic regression on physical features, same folds and pooled-OOF protocol). Classical detectors are applied to the same frozen feature cache; thresholds are selected on development pooled OOF only (max-MCC) (scripts `b1_classical_detectors.py`; outputs `results/classical_baselines/`).

**TABLE II** CLASSICAL DETECTOR BASELINES (DEVELOPMENT SET, POOLED-OOF PROTOCOL, SEED 42)

| Detector | Phase PR-AUC | Phase ROC-AUC | MCC | F1 | Type |
|----------|-------------|---------------|-----|-----|------|
| Energy threshold | 0.055 | 0.415 | 0.042 | 0.018 | classical, no learning |
| Impulsiveness (crest/kurtosis) | 0.104 | 0.715 | 0.162 | 0.176 | classical, no learning |
| Spectral (centroid + band energy) | 0.088 | 0.610 | 0.094 | 0.161 | classical, no learning |
| PRPD-style features + LR | 0.323 | 0.861 | 0.358 | 0.395 | classical ML (same folds/OOF) |
| **E4 mainline (reference)** | **0.530** | **0.881** | — | — | proposed (pooled-OOF) |

Classical detectors—even with no learned parameters—already capture part of the impulsive structure of PD (impulsiveness detector ROC-AUC 0.715), but their phase-level PR-AUC (0.055–0.104) is an order of magnitude below the E4 mainline (0.530 pooled-OOF; 0.615 per-fold mean), and the strongest classical-ML reference (PRPD-style features + LR, 0.323) remains substantially below it. This contrast isolates the contribution of event-focused information selection and hierarchical weak supervision: classical features alone, without coverage-aware sampling and window-to-phase aggregation, cannot resolve the extreme imbalance of the measurement task.

**Labeling-cost analysis.** To quantify the labeling-cost advantage of the weakly supervised pipeline, we retrain the E4 mainline under the exact protocol while retaining only a fraction f of the phase labels in the training part of each fold (per-measurement retention, keeping all three phases of retained measurements; the validation part always keeps full labels for early stopping and threshold selection, measuring the true performance ceiling at each labeling budget). Results (per-fold mean phase PR-AUC, seed 42):

**TABLE II-A** LABEL-RATIO EXPERIMENT (WEAKLY SUPERVISED, SEED 42, PER-FOLD MEAN)

| Labeled fraction | Phase PR-AUC | Relative to 100% | Measurement PR-AUC |
|------------------|-------------|-------------------|---------------------|
| 5% | 0.272 | 44% | 0.340 |
| 10% | 0.346 | 56% | 0.396 |
| 20% | 0.355 (0.377 ± 0.021 across label seeds {42, 7, 2024}) | 58–61% | 0.413 (0.450 ± 0.005) |
| 50% | 0.542 | **88%** | 0.571 |
| 100% (mainline) | 0.615 | 100% | 0.643 |

With only 50% of the phase labels, the pipeline retains 88% of the full-supervision performance (0.542 vs 0.615) and already exceeds the strongest fully labeled traditional baseline (0.206) by a wide margin even at 5–10% labels; the weakly supervised design converts scarce measurement annotation into near-full-supervision detection quality. Label-subset stability at 20% is confirmed across three label seeds (fold-mean phase PR-AUC 0.355/0.397/0.380, mean 0.377 ± 0.021; scripts `l1_label_fraction.py`; outputs `results/label_fraction/`).

### C. Traditional Machine Learning Baselines

To establish a reference point, traditional ML classifiers are evaluated on the same mixed K=8 feature cache (58-dimensional physical features per window) with the same 5-fold split (seed 42) and fold fingerprint. Three feature aggregations are combined with logistic regression (LR), random forest (RF), and LightGBM (LGBM), with grid search per fold (results recorded in `results/baseline_full_comparison.json`):

**TABLE II** TRADITIONAL MACHINE LEARNING BASELINES (PHASE PR-AUC, PER-FOLD MEAN, SEED 42)

| Feature set | LR | RF | LGBM |
|-------------|----|----|------|
| 116-d (mean+std aggregation) | 0.157 | 0.155 | 0.154 |
| 406-d (seven statistics) | 0.137 | 0.174 | 0.196 |
| 464-d (flattened windows) | 0.107 | 0.188 | **0.206** |

Phase PR-AUC (per-fold mean). The strongest traditional baseline is flattened features + LightGBM at 0.206; the E4 mainline (0.615) achieves an absolute improvement of +0.409 under the same folds and feature cache (pooled-OOF protocol: +0.335; Section VI-A).

### C. Phase Interaction Ablation and Mainline Selection (E1–E6)

All six variants share the same backbone (pure CNN encoder + attention MIL) and differ only in the interaction mechanism or loss (Section IV-C). Table III reports seed-42 per-fold means.

**TABLE III** PHASE INTERACTION ABLATION (E1–E6, SEED 42, PER-FOLD MEAN ± STD)

| Variant | Interaction | Phase PR-AUC | Meas PR-AUC | Phase ROC | Meas ROC | Phase MCC | Meas MCC |
|---------|-------------|--------------|-------------|-----------|----------|-----------|----------|
| E1 | None | 0.483 ± 0.060 | 0.562 ± 0.053 | 0.925 | 0.935 | 0.434 | 0.465 |
| E2 | Mean shared-score | 0.604 ± 0.042 | 0.627 ± 0.053 | 0.942 | 0.940 | 0.469 | 0.469 |
| E3 | Max aggregation | 0.620 ± 0.034 | 0.645 ± 0.044 | 0.944 | 0.941 | 0.486 | 0.472 |
| **E4** | **Context-concat** | **0.615 ± 0.053** | **0.643 ± 0.053** | **0.937** | **0.935** | **0.534** | **0.541** |
| E5 | Context-add | 0.581 ± 0.069 | 0.614 ± 0.062 | 0.932 | 0.940 | 0.510 | 0.522 |
| E6 | E4 + hierarchical loss (λ=0.25) | 0.586 ± 0.081 | 0.615 ± 0.090 | 0.943 | 0.940 | 0.535 | 0.542 |

Three-seed means (seeds 42, 7, 2024): E4 phase 0.611 ± 0.006, measurement 0.639 ± 0.008; E2 0.608 ± 0.008 / 0.626 ± 0.011; E5 0.566 ± 0.012 / 0.608 ± 0.010.

Paired cluster bootstrap (2,000 resamples, seed 42, measurement-clustered; differences = E4 − variant):

- E4 vs E2: phase +0.055 [95% CI +0.006, +0.103], measurement +0.071 [+0.023, +0.118] — **significant**
- E4 vs E5: phase +0.073 [+0.028, +0.117], measurement +0.077 [+0.025, +0.127] — **significant**
- E4 vs E3: phase +0.036 [−0.012, +0.087], measurement +0.045 [−0.002, +0.096] — not significant
- E6 vs E4: phase −0.030 [−0.084, +0.024], measurement −0.050 [−0.104, +0.003] — not significant (negative result, honestly reported)

E1 (no interaction) is the weakest across all three seeds (seed-42 phase 0.483 ± 0.060), confirming that three-phase interaction is a necessary component.

**Mainline selection.** Although the seed-42 difference between E4 and E3 is not statistically significant (and E3's per-fold mean is marginally higher), E4 is selected as the mainline for three reasons. First, the application requirement: E3's max aggregation broadcasts the elementwise maximum to all three phases, so all phases receive identical scores and per-phase fault localization is impossible; E4 outputs three independent, rankable per-phase probabilities (Section IV-C), which is the requirement of phase-level diagnosis in distribution networks. Second, on secondary metrics E4 attains higher phase MCC (0.534 vs 0.486) and a slightly higher three-seed measurement-level mean (0.639 vs 0.636). Third, among the interaction variants, only E4 significantly outperforms both shared-score variants (E2: phase +0.055 [+0.006, +0.103]; E5: phase +0.073 [+0.028, +0.117]): a post-hoc paired bootstrap with the same locked methodology (measurement-clustered, 2,000 resamples, seed 42, on the locked OOF predictions; `results/stage1_tim/posthoc_e3_vs_shared.json`) shows that E3 does not differ significantly from E2 (phase +0.018 [−0.018, +0.052]) or E5 (+0.036 [−0.004, +0.077]), and E2 vs E5 is also not significant (Section VI-B). E6 (hierarchical loss) provides no significant gain over E4. E4 is therefore the interaction form with the most consistent evidence among the phase-distinct variants. The same post-hoc analysis also gives bootstrap-level support for the necessity of three-phase interaction: E4 vs E1 (no interaction) is +0.153 [+0.102, +0.204] at phase level and E3 vs E1 is +0.117 [+0.073, +0.159]. Development-set PR curves are shown in Fig. 3 (historical locked mainline shown as reference).

![Fig. 3 Development-set PR curves (historical locked mainline as reference)](figures/fig3_pr_curves.png)

Fig. 4 and Fig. 5 illustrate the calibration and attention behavior of the historical locked mainline (protocol and visualization reference only).

![Fig. 4 Blind-test reliability diagram (historical locked mainline)](figures/fig4_reliability.png)

![Fig. 5 Blind-test positive/negative examples with attention weights (historical locked mainline)](figures/fig5_examples.png)

### D. Matched Encoders and Published Methods

To test whether judicious information selection maintains competitive performance at lower model capacity, we fix all other components (robust preprocessing → encoder → attention MIL → context-concat → phase classifier, phase-level BCE) and replace only the encoder under the same K=8 mixed windows, seed 42, and protocol. The encoder choices follow standard time-series classification designs [26], [27], [28]. MACs and latency are from the Stage-1 benchmark (synthetic 800k signals, 50 repetitions, p50).

**TABLE IV** MATCHED ENCODER COMPARISON (SAME PROTOCOL, SEED 42)

| Encoder | Params (total / executed) | MACs/meas | GPU b1 (ms) | CPU b1 (ms) | Phase PR-AUC | Meas PR-AUC |
|---------|---------------------------|-----------|-------------|-------------|--------------|-------------|
| cnn (E4) | 113,265 / 97,201 | 62.8M | 1.94 | 8.34 | 0.615 ± 0.053 | 0.643 ± 0.053 |
| simple_cnn | 96,945 / 96,945 | 62.8M | 1.73 | 8.54 | 0.633 ± 0.068 | 0.656 ± 0.055 |
| ResNet1D [25] | 603,170 | 16.53G | 7.04 | 216.93 | 0.688 ± 0.028 | 0.720 ± 0.040 |
| InceptionTime [28] | 627,186 | 55.70G | 21.87 | 1031.70 | 0.636 ± 0.071 | 0.673 ± 0.092 |

Paired bootstrap (seed 42, 2,000 resamples): ResNet1D is the only encoder significantly higher than cnn (phase +0.095 [+0.033, +0.152]; measurement +0.064 [+0.007, +0.119]) at approximately 263× MACs, 3.6× GPU batch-1 latency, and 26× CPU latency. InceptionTime is not significant (phase +0.044 [−0.005, +0.092]) at approximately 887× MACs; simple_cnn is not significantly different (phase −0.005 [−0.048, +0.041]). The lightweight positioning is therefore an accuracy–efficiency trade-off: 97,201 executed parameters, 62.8M MACs, and about 1.9 ms GPU batch-1 latency achieve competitive discriminative quality, without claiming the highest accuracy.

Published methods adapted to our framework under the earlier historical protocol (mean MIL + max interaction, same folds and feature cache) serve as literature reference points only and are not paired-bootstrap comparisons with E4 [7], [8]: Zheng TF-CNN (134,113 parameters; STFT + 2D CNN) reaches phase PR-AUC 0.715 ± 0.050 / measurement 0.737 under that protocol, and Fei CNN+QSVM (55,472 CNN + 3,989 SV) reaches 0.495 ± 0.053 / 0.632. Two points deserve emphasis. First, the historical protocol differs from the E4 protocol in the MIL and interaction modules, so these numbers are not directly comparable with E4's; cross-protocol comparisons must state the metric protocol. Second, our contribution is an encoder-pluggable measurement framework: Zheng's time-frequency encoder—and any stronger encoder—can be inserted into the same sampling and weak-supervision pipeline (a re-test of Zheng under the E4 protocol is listed as future work in Section VII-B), so Zheng's higher score under its own protocol is evidence that stronger encoders can raise the performance ceiling of the framework, at correspondingly higher preprocessing and parameter cost. Fei's result shows that lightweight structure without window-level weakly supervised modeling is insufficient. A TCN encoder [29] was also prepared but its baseline is suspended and not included here.

### E. Window Strategy Ablation

Under the same development folds, window strategies are compared on the reference architecture (203,634 parameters, mechanism verification; the historical lightweight mainline rows are separate evidence with a different model and feature configuration and are not directly compared):

**TABLE V** WINDOW STRATEGY ABLATION (PER-FOLD MEAN PR-AUC ± STD)

| Strategy | K | Composition | Phase PR-AUC | Meas PR-AUC |
|----------|---|------------|--------------|-------------|
| single | 1 | 1 equidistant | 0.255 ± 0.029 | 0.273 |
| equidistant | 8 | 8 equidistant + 0 event | 0.526 ± 0.105 | 0.534 |
| event | 8 | 0 equidistant + 8 event | 0.591 ± 0.056 | 0.624 |
| mixed_k4 | 4 | 2 equidistant + 2 event | 0.461 ± 0.070 | 0.523 |
| mixed_k8 (203k reference) | 8 | 4 equidistant + 4 event | 0.590 ± 0.081 | 0.621 |
| mixed_k12 (203k reference) | 12 | 6 equidistant + 6 event | 0.644 ± 0.092 | 0.668 |
| mixed_k8 (historical lightweight, seed 42) | 8 | 4 equidistant + 4 event | 0.639 ± 0.055 | 0.657 |
| mixed_k12 (historical lightweight, seed 42) | 12 | 6 equidistant + 6 event | 0.670 ± 0.070 | 0.679 |

Key points: (1) single-window coverage is severely insufficient (0.255); (2) event windows carry the primary discriminative information—pure event (0.591) and mixed K=8 (0.590) are nearly identical on the 203k reference (difference 0.001, within noise; pure event has smaller fold-wise std), so we do not claim "mixed outperforms pure event"; (3) performance increases monotonically with K (203k reference: 0.461 → 0.590 → 0.644; historical lightweight: 0.639 → 0.670, seed 42). K=12 is optimal but raises window count and extraction/training cost by about 1.5×; mixed K=8 is retained as the mainline for coverage robustness (equidistant anchors guarantee global coverage and provide fallback against event-detection failure) and because the frozen protocol commits the one-time blind test to the K=8 configuration. K=12 remains development-set evidence only. Results are shown in Fig. 8.

![Fig. 8 Window strategy ablation (K and composition)](figures/fig8_window_policy.png)

Mechanism verification on the reference architecture (203,634 parameters; gated-attention MIL + recurrent cyclic interaction) confirms the module-level conclusions (Fig. 6; full per-fold tables are provided as supplementary material): encoder ablation (gated attention + cyclic interaction)—pure CNN 0.614 ± 0.073, dual-branch fusion 0.590 ± 0.081, statistical-feature MLP 0.208 ± 0.070 (phase PR-AUC); MIL aggregation (dual-branch + cyclic)—mean 0.611 ± 0.092, attention 0.603 ± 0.065, gated attention 0.590 ± 0.081, max 0.559 ± 0.045; phase interaction (dual-branch + gated attention)—max 0.620 ± 0.039, mean 0.612 ± 0.070, direct concatenation 0.599 ± 0.014, recurrent cyclic 0.590 ± 0.081, no interaction 0.474 ± 0.087. Three conclusions: (1) phase interaction is the largest contributor (+0.12–0.15), consistent with the E1–E6 result that no interaction is significantly weakest; (2) simple encoders and simple mean/max interactions perform at mean levels not lower than complex modules, supporting the lightweight design, although fold-wise standard deviations are larger (±0.04–0.09) and no significance is asserted; (3) complex recurrent-equivariant and gated-attention modules do not provide clear gains. Because these rows use the historical protocol and a different model configuration, they serve as mechanism evidence rather than main-report results.

![Fig. 6 Ablation experiments (historical reference architecture mechanism verification)](figures/fig6_ablation.png)

### F. Measurement Imperfection Robustness

Robustness evidence comes from the historical locked mainline (pure CNN + attention MIL + mean interaction, 5-fold checkpoints; development per-fold mean 0.622 ± 0.064, OOF baseline phase PR-AUC 0.568 / measurement 0.599 with measurement-level bootstrap 95% CI [0.541, 0.659]). The 203k reference architecture serves only as mechanism reference; in addition, the E4 mainline's perturbation robustness was measured directly on the IR-trained checkpoints (Section V-K) under the same inference-time protocol, so the robustness conclusions now align with the main model (scripts `r2_e4_robustness.py`; outputs `results/ir_eval/robustness_E4.json`).

**TABLE VI** INFERENCE-TIME PERTURBATION ROBUSTNESS (HISTORICAL LOCKED MAINLINE, DEVELOPMENT OOF)

| Perturbation | Phase PR-AUC relative drop | Measurement relative drop |
|--------------|---------------------------|---------------------------|
| Gaussian noise 20 dB | −10.8% | −12.8% |
| Gaussian noise 10 dB | −49.4% | −50.7% |
| Gaussian noise 5 dB | −67.2% | −67.5% |
| Amplitude ×0.8 / ×1.2 | 0% | 0% |
| Time shift −64 / +64 | −33.8% / −73.9% | −33.6% / −74.7% |
| Time shift −128 / +128 | −75.3% / −84.1% | −75.8% / −83.9% |
| Missing any phase | — | ≈0% |

**TABLE VI-A** E4 MAINLINE PERTURBATION ROBUSTNESS (IR-TRAINED CHECKPOINTS, 80% TRAINING-DOMAIN OOF)

| Perturbation | Phase PR-AUC relative drop |
|--------------|---------------------------|
| Gaussian noise 20 dB | −0.2% |
| Gaussian noise 10 dB | +1.9% |
| Gaussian noise 5 dB | +3.7% |
| Amplitude ×0.8 / ×1.2 | ≈0% |
| Time shift −64 / +64 | −48.1% / −73.1% |
| Time shift −128 / +128 | −47.4% / −71.4% |
| Missing any phase (measurement level) | −1.3% to +0.7% |

The E4 mainline is markedly more robust to additive noise than the historical mainline (5 dB: +3.7% vs −67.2%), consistent with its context-concat interaction and robust normalization; amplitude scaling and single-phase absence remain insensitive. Window temporal alignment remains the dominant weakness (−47% to −73%, asymmetric with +shifts worse), confirming that the time-shift augmentation variant of Table VII is the appropriate mitigation for the main model as well.

Absolute performance decreases monotonically with SNR (phase PR-AUC 0.506 at 20 dB, 0.186 at 5 dB). Amplitude scaling and single-phase absence are insensitive (noisy-OR three-phase redundancy is effective). Time shifts (implemented as zero-padded fixed windows with recomputed features) are the main weakness, with asymmetric drops up to −84% at +128, indicating sensitivity to window temporal alignment. An auxiliary variant trained with random zero-padding time-shift augmentation (±128 per batch, development set only) leaves unperturbed performance nearly unchanged (OOF phase 0.581; per-fold mean 0.632 ± 0.066, measurement 0.652) while narrowing the shift drop from 34–84% to within ±3% (Tables VI and VII):

**TABLE VII** TIME-SHIFT AUGMENTATION VARIANT (DEVELOPMENT OOF, PHASE PR-AUC RELATIVE DROP)

| Time shift | Mainline phase drop | Augmented variant phase drop |
|-----------|--------------------|------------------------------|
| −128 | −75.3% | −1.0% |
| −64 | −33.8% | −2.6% |
| +64 | −73.9% | −2.6% |
| +128 | −84.1% | −0.3% |

The augmented variant does not enter the blind test. Perturbation results are shown in Fig. 7.

![Fig. 7 Noise and perturbation robustness (historical locked mainline)](figures/fig7_robustness.png)

### G. Controlled Sampling Experiment: Information Selection as a Design Variable

To decouple information selection from model structure, the architecture is fixed (CNN encoder → attention MIL → context-concat → phase classifier, phase-level BCE, same protocol) and only the sampling strategy varies: uniform K=8, event K=8, random K=8, mixed K=8 (mainline), and full-signal (K=1, L=800,000, batch 8 due to GPU memory; K=8 rows use batch 64). All strategies share the same measurement order and fold fingerprints, so per-fold metrics and paired cluster bootstrap are directly comparable.

**TABLE VIII** CONTROLLED SAMPLING EXPERIMENT (FIXED ARCHITECTURE, SEED 42)

| Strategy | K | Data usage | Phase PR-AUC | Meas PR-AUC | Phase ROC | Meas ROC | Phase MCC | Meas MCC | MACs/meas | GPU b1 (ms) | CPU b1 (ms) | Peak GPU mem (MB) |
|----------|---|-----------|--------------|-------------|-----------|----------|-----------|----------|-----------|-------------|-------------|-------------------|
| uniform_k8 | 8 | 8.2% | 0.513 ± 0.130 | 0.537 ± 0.132 | 0.920 | 0.916 | 0.522 | 0.532 | 62.8M | 6.49 | 21.17 | 92.4 |
| event_k8 | 8 | 8.2% | 0.609 ± 0.081 | 0.638 ± 0.061 | 0.940 | 0.937 | 0.551 | 0.583 | 62.8M | 5.59 | 19.40 | 92.4 |
| random_k8 | 8 | 8.2% | 0.579 ± 0.062 | 0.601 ± 0.036 | 0.928 | 0.923 | 0.500 | 0.512 | 62.8M | 5.39 | 19.42 | 92.4 |
| mixed_k8 (mainline) | 8 | 8.2% | 0.615 ± 0.053 | 0.643 ± 0.053 | 0.937 | 0.935 | 0.534 | 0.541 | 62.8M | 5.35 | 20.97 | 92.4 |
| full_signal | 1 | 100% | 0.345 ± 0.174 | 0.387 ± 0.165 | 0.823 | 0.835 | 0.178 | 0.217 | 755.3M | 25.35 | 379.96 | 184.0 |

Paired cluster bootstrap (2,000 resamples, seed 42; difference = mixed_k8 − strategy):

**TABLE IX** PAIRED CLUSTER BOOTSTRAP CONTRASTS (MIXED K=8 MINUS STRATEGY, 95% CI)

| Contrast | Level | PR-AUC difference (95% CI) |
|----------|-------|---------------------------|
| mixed vs uniform | Phase | +0.113 [0.055, 0.175] |
| mixed vs uniform | Meas | +0.131 [0.073, 0.194] |
| mixed vs event | Phase | +0.062 [0.002, 0.122] |
| mixed vs event | Meas | +0.053 [−0.011, 0.118] |
| mixed vs random | Phase | +0.044 [−0.023, 0.112] |
| mixed vs random | Meas | +0.077 [0.010, 0.141] |
| mixed vs full | Phase | +0.306 [0.240, 0.369] |
| mixed vs full | Meas | +0.333 [0.271, 0.395] |

Key points: (1) uniform K=8 is the weakest K=8 strategy (0.513/0.537), showing that coverage without information focus is insufficient; (2) pure event and mixed K=8 are comparable (0.609 vs 0.615), consistent with Section V-E—discharge transients concentrate in event-high regions, and anchors mainly provide coverage fallback; (3) random K=8 (0.579/0.601) is not significantly different at phase level (CI contains 0) but is at measurement level (CI excludes 0), indicating that random coverage captures some event information while event focus pays off more at measurement-level fusion; (4) the full-signal baseline uses about 12× MACs (755.3M vs 62.8M), 4.7× GPU batch-1 latency, and more memory, yet achieves phase PR-AUC 0.345—significantly below mixed K=8. Therefore, information selection itself is the critical design variable for diagnostic efficiency, not the CNN structure (Fig. 9). The full-signal row intentionally uses the same lightweight architecture to isolate the sampling variable; stronger full-signal models are outside the scope of this paper.

![Fig. 9 Controlled sampling experiment: strategy performance and cost comparison](figures/fig9_sampling_policy.png)

### H. End-to-End Measurement-to-Decision Pipeline Cost

To avoid overstating "real-time detection / end-to-end 1.94 ms", Table X reports the per-component cost of the complete measurement-to-decision pipeline from the Stage-2 benchmark (synthetic 800k three-phase signal, 50 repetitions, p50; same session, same platform). Window selection runs on CPU; the model forward pass reports CPU/GPU batch-1 and GPU throughput.

**TABLE X** END-TO-END MEASUREMENT-TO-DECISION PIPELINE COST (P50)

| Stage | Platform / batch | p50 |
|-------|------------------|-----|
| Event score (single phase) | CPU batch=1 | 131.64 ms |
| Peak detection (single phase) | CPU batch=1 | 36.35 ms |
| Uniform K=8 selection (single phase) | CPU batch=1 | ≈0.0 ms |
| Event K=8 selection (single phase) | CPU batch=1 | 178.59 ms |
| Mixed K=8 selection (single phase) | CPU batch=1 | 183.71 ms |
| Random K=8 selection (single phase) | CPU batch=1 | ≈0.02 ms |
| Mixed K=8 model forward (robust norm + encoder + MIL + interaction + output) | GPU batch=1 | 5.35 ms |
| Mixed K=8 model forward | CPU batch=1 | 20.97 ms |
| Mixed K=8 model forward (throughput) | GPU batch=64 | 1.77 ms/measurement |
| Full-signal model forward | GPU batch=1 | 25.35 ms |
| Full-signal model forward | CPU batch=1 | 379.96 ms |
| Full-signal model forward (throughput) | GPU batch=1 | 21.78 ms/measurement |

Per-component breakdown for mixed K=8 (GPU/CPU, p50): robust normalization 2.85/7.98 ms, encoder 3.77/18.38 ms, MIL 0.49/0.19 ms, three-phase interaction 0.39/0.25 ms, output head 0.48/0.12 ms, total 5.35/20.97 ms. Full-signal: robust normalization 15.32/203.20 ms, encoder 25.00/383.60 ms, total 25.35/379.96 ms. Three conclusions: (1) the sampling stage (about 184 ms per phase for the mixed strategy) dominates the CPU pipeline, while the K=8 model forward pass is about 5.35 ms (GPU batch-1); (2) the full-signal model increases MACs 12×, GPU batch-1 latency 4.7×, CPU batch-1 latency 18×, and peak GPU memory from 92.4 MB to 184.0 MB; (3) we therefore limit efficiency claims to "reducing neural inference complexity and data processing requirements by processing only informative segments" and do not claim real-time detection or equate model forward latency with end-to-end latency.

### I. Label-Fraction Experiment: Quantifying the Labeling-Cost Advantage

To quantify the labeling-cost benefit of the hierarchical weakly supervised pipeline—the industrial motivation of the phase-level-only supervision—we train the E4 mainline with randomly retained phase labels at fractions f ∈ {5%, 10%, 20%, 50%} of the development measurements (labeling is measurement-grouped: all three phases of a selected measurement are labeled, unselected measurements contribute no labels), under the identical 5-fold protocol and feature cache. The 20% fraction is replicated across three label seeds (42, 7, 2024); the other fractions use seed 42 (scripts `l1_label_fraction.py`; outputs `results/label_fraction/`).

**TABLE XI-A** LABEL-FRACTION EXPERIMENT (PHASE PR-AUC, PER-FOLD MEAN; 100% = MAINLINE REFERENCE)

| Label fraction | Labeled measurements | Phase PR-AUC (per-fold mean) | Measurement PR-AUC | Fraction of full-supervision performance |
|----------------|----------------------|------------------------------|---------------------|------------------------------------------|
| 5% | 124 | 0.272 | 0.340 | 44% |
| 10% | 248 | 0.346 | 0.396 | 56% |
| 20% (3 label seeds) | 496 | 0.377 ± 0.021 | 0.440 | 61% |
| 50% | 1240 | 0.542 | 0.571 | 88% |
| 100% (mainline) | 2481 | 0.615 ± 0.053 | 0.643 | 100% |

The performance–label curve is smooth and monotonic with no saturation collapse: with only 50% of the phase labels the mainline retains 88% of its full-supervision phase PR-AUC (0.542 vs 0.615), and even 10% of labels yields 56%. The 20% fraction is stable across label seeds (±0.021). This quantifies the labeling-cost argument of the weakly supervised design: coverage-aware sampling already removes 91.8% of the raw data, and hierarchical weak supervision further reduces the annotation burden by roughly half at 88% of full-supervision performance—a combined measurement-and-labeling cost reduction that classical detectors and fully supervised pipelines cannot offer.

### J. Independent Test: Harvard Dataverse Held-Out Set

#### I.1 Protocol

The independent partition (83,233 phase signals; 28,285 measurements; 1,308 positives, 1.57%) was locked before download (SHA-256 hashes of the archive, member list, feature vectors, and label table written to the protocol lock), the E4 model was reinitialized from the seed-42 5-fold development training (same architecture and protocol as the mainline), and the frozen model was evaluated exactly once (average probability of the 5 folds over mixed K=8 windows), with a run lock rejecting any second evaluation.

#### I.2 Results

| Metric | Phase-level (label > 0) | Phase-level (contact classes 1/2/5/6) | Measurement-level (full triple) |
|--------|------------------------|--------------------------------------|--------------------------------|
| PR-AUC | 0.216 | 0.181 | 0.495 |
| ROC-AUC | 0.899 | 0.894 | 0.957 |
| MCC | 0.224 | 0.195 | 0.000 |
| F1 | 0.192 | 0.170 | 0.000 |
| ECE | 0.007 | 0.008 | 0.023 |
| Brier | 0.014 | 0.014 | 0.026 |

Accuracy is omitted: at a 1.57% positive rate the majority-class baseline already achieves 0.984, making accuracy uninformative. Bootstrapped 95% CIs (measurement-clustered, 2,000 resamples): phase-level PR-AUC [0.199, 0.234]; measurement-level [0.196, 0.833]—only 120 complete three-phase measurements exist, of which 4 are positive, so measurement-level numbers are exploratory evidence with very wide CIs and no reliable point-estimate comparison is possible.

#### I.3 Analysis

Four complementary arguments show that the lower PR-AUC on the independent set reflects prevalence, not model degradation:

1. **Structural ceiling.** PR-AUC is sensitive to the positive rate; at 1.57% the achievable precision-recall curve is structurally constrained relative to 5.9%. The phase-level ROC-AUC of 0.899 confirms that ranking ability is preserved on a non-overlapping block from the same data source.
2. **Prevalence-normalized lift.** Normalizing PR-AUC by the random baseline (the positive rate) gives 0.216/0.0157 ≈ **13.8×** on the held-out set versus 0.615/0.0595 ≈ **10.3×** on the development set: the discriminative gain relative to chance is not lower on the independent set.
3. **Quantitative prevalence decomposition.** Because the development set contains only 443 positive phases, direct downsampling of negatives to a 1.57% positive rate is infeasible; we instead apply an analytic precision–recall reweighting to the frozen scores (precision(t, λ) = TP(t)/(TP(t) + λ·FP(t)) with recall unchanged under negative reweighting factor λ, verified by 20 measurement-clustered negative-downsampling repetitions on the held-out set: analytic 0.457 vs resampled 0.456 ± [0.442, 0.469], difference 0.0004). On a common prevalence grid, the held-out set reaches phase PR-AUC 0.457 at the development positive rate (5.9%) and 0.570 at 10%, versus 0.530 on development (pooled-OOF protocol; Section VI-A). The apparent gap therefore decomposes into a prevalence effect Δπ = −0.242 (≈77% of the drop) and a genuine domain-shift effect Δshift = −0.073 (≈23%), the latter being an honest cross-block offset substantially smaller than the 65% apparent drop. At fixed recalls of 0.1/0.2/0.5, held-out precision (0.500/0.336/0.164) is below development (0.776/0.748/0.610), consistent with the preserved-but-weaker ranking indicated by ROC-AUC. The normalized lift (PR-AUC − π)/(1 − π) is 0.500 on development versus 0.422 on the held-out set at the same 5.9% positive rate.
4. **Threshold and calibration.** The measurement threshold (0.88) selected on the development set is too strict for the held-out set, yielding MCC/F1 = 0 at measurement level; ROC-AUC 0.957 shows that positive and negative measurements remain separable after threshold recalibration—a cross-block calibration shift, not model failure. Phase-level ECE is 0.007 (Brier 0.014), consistent with development-level calibration (0.007–0.008).

The independent-test loop is closed: the E4 mainline was evaluated exactly once on data that never participated in any training, selection, hyperparameter, or threshold determination, with a tamper-evident receipt binding protocol lock and checkpoint hashes. The development 5-fold CV, three-seed replication, and cluster bootstrap estimates are now anchored by an independent test with ROC-AUC 0.899. All prevalence analyses above are post-hoc statistics on the frozen predictions and did not reopen any one-time evaluation or change any threshold.

### K. Supplementary Evidence

The following analyses are part of the revision plan and will be inserted when the corresponding experiments are completed [t3]: (1) ~~a positive-rate sensitivity control~~ — **completed (Section V-J.3)**: the analytic precision–recall reweighting over a common prevalence grid (5.9%/10%/15%), the 20-repetition measurement-clustered negative-downsampling validation, and the fixed-recall precision comparisons decompose the held-out PR-AUC gap into a prevalence effect (Δπ = −0.242) and a domain-shift effect (Δshift = −0.073); scripts and outputs are released with the code repository (`scripts/e1_posrate_sensitivity.py`; `results/posrate_sensitivity/`); (2) ~~a labeling-ratio experiment~~ — **completed (Section V-I)**: with only 50% of the phase labels, the weakly supervised pipeline retains 88% of the full-supervision performance (0.542 vs 0.615 per-fold mean; 20% labels: 0.377 ± 0.021, 61%), quantifying the labeling-cost advantage (scripts `l1_label_fraction.py`; `results/label_fraction/`); (3) comparison anchors against public VSB competition solutions (leaderboard metric MCC; protocol mapping required); (4) ~~classical detector baselines~~ — **completed (Section V-B)**: energy 0.055 / impulsiveness 0.104 / spectral 0.088 / PRPD+LR 0.323 phase PR-AUC (pooled-OOF) versus E4 0.530; (5) a re-test of the Zheng time-frequency encoder under the E4 protocol; (6) sensitivity of the sampling parameters (peak distance ×0.5/×2, dedup IoU 0.3/0.7); (7) ~~an internal-reserved (IR) hold-out evaluation~~ — **completed**: 497 measurements (20%, 67 positive phases, 4.49%) stratified into an IR set under a new SHA-256 hash-locked split; the E4 mainline retrained on the remaining 1,984 measurements (80%) under the exact mainline protocol (5-fold, seed 42, patience 15); IR evaluated exactly once with thresholds frozen from the 80% training-domain OOF (phase 0.53, measurement 0.59). Results (5-fold ensemble, one-time): phase PR-AUC 0.412 / ROC-AUC 0.953 / MCC 0.364; measurement PR-AUC 0.415 / ROC-AUC 0.951 / MCC 0.496. Honest positioning: the IR set is a same-distribution internal validation drawn from the development set after all original choices were locked—it is not an external validation; the lower phase PR-AUC relative to the development CV (0.412 vs 0.530 pooled / 0.615 per-fold mean) is consistent with the lower IR positive rate (4.49% vs 5.95%) and with the prevalence analysis of Section V-I.3, while ROC-AUC (0.953) confirms preserved ranking on unseen same-distribution measurements (scripts `e2_ir_evaluation.py`; outputs `results/ir_eval/`).

---

## VI. Uncertainty Analysis

### A. Metric Protocol: Per-Fold Mean versus Pooled OOF

Per-fold mean PR-AUC and pooled-OOF PR-AUC differ systematically: for E4, per-fold mean 0.615 vs pooled OOF 0.533 (difference 0.082) at phase level and 0.643 vs 0.585 (difference 0.058) at measurement level; for the strongest traditional baseline, 0.206 vs 0.198. Under both protocols the model substantially outperforms the baseline (absolute improvement +0.409 per-fold, +0.335 pooled), so the performance advantage is protocol-robust. The absolute gap arises from fold-wise probability scale/calibration shifts (audit in `results/fold_oof_gap_audit_80k.json`): all folds share the same fold fingerprint and `average_precision_score` computation, and OOF probabilities are strictly aligned with measurement IDs. The historical blind test used a 5-fold probability-average ensemble, which naturally mitigates this fold-wise scale difference. This paper uniformly adopts per-fold mean as the primary protocol; pooled-OOF values (seed 42: E1 0.378/0.444, E2 0.476/0.513, E3 0.495/0.538, E4 0.533/0.585, E5 0.458/0.506, E6 0.502/0.534, phase/measurement) are reported only here as robustness supplements. The historical blind-test single frozen evaluation yields phase-level shared-score PR-AUC 0.524 (95% CI [0.379, 0.657]) and measurement-level 0.582 ([0.447, 0.714]); these numbers belong to the historical locked mainline and are not attributed to E4 or compared as point estimates with E4's per-fold means.

### B. Measurement-Cluster Statistical Uncertainty

To quantify small-sample and inter-measurement variability without conflicting with metrological terminology, we call this measurement-cluster statistical uncertainty. A measurement-clustered bootstrap (2,000 resamples, stratified by measurement label) on E4's development OOF predictions gives phase PR-AUC median 0.5333 (95% CI [0.4734, 0.5908]) and measurement PR-AUC median 0.5852 ([0.5267, 0.6414]); the difference from the per-fold mean protocol is the fold-scale phenomenon of Section VI-A. Paired bootstrap on the same measurement clusters (difference = former − latter):

| Contrast | Phase PR-AUC difference (95% CI) | Measurement difference (95% CI) |
|----------|----------------------------------|----------------------------------|
| E4 vs E2 (mean shared) | +0.055 [+0.006, +0.103] | +0.071 [+0.023, +0.118] |
| E4 vs E5 (context-add) | +0.073 [+0.028, +0.117] | +0.077 [+0.025, +0.127] |
| E4 vs E3 (max aggregation) | +0.036 [−0.012, +0.087] | +0.045 [−0.002, +0.096] |
| E6 vs E4 (hierarchical loss) | −0.030 [−0.084, +0.024] | −0.050 [−0.104, +0.003] |
| E3 vs E2 (mean shared)† | +0.018 [−0.018, +0.052] | +0.025 [−0.010, +0.058] |
| E3 vs E5 (context-add)† | +0.036 [−0.004, +0.077] | +0.031 [−0.012, +0.073] |
| E2 vs E5 (context-add)† | +0.018 [−0.020, +0.060] | +0.007 [−0.038, +0.049] |
| E3 vs E1 (no interaction)† | +0.117 [+0.073, +0.159] | +0.093 [+0.048, +0.139] |
| E4 vs E1 (no interaction)† | +0.153 [+0.102, +0.204] | +0.138 [+0.082, +0.197] |
| ResNet1D vs cnn (E4) | +0.095 [+0.033, +0.152] | +0.064 [+0.007, +0.119] |
| InceptionTime vs cnn (E4) | +0.044 [−0.005, +0.092] | +0.051 [−0.003, +0.102] |
| simple_cnn vs cnn (E4) | −0.005 [−0.048, +0.041] | −0.016 [−0.060, +0.031] |

† Post-hoc contrasts computed with the same measurement-clustered paired bootstrap (2,000 resamples, seed 42) on the locked OOF predictions (`results/stage1_tim/posthoc_e3_vs_shared.json`). They establish that, among the interaction variants, only E4 significantly outperforms both shared-score variants (E3 vs E2 and E3 vs E5 are not significant), and that three-phase interaction is necessary (both E4 and E3 significantly outperform no interaction, E1).

E4 significantly outperforms E2 and E5 at both levels; the differences from E3 and E6 are not significant (both negative results honestly reported). E3 does not differ significantly from E2 or E5, and both E4 and E3 significantly outperform no interaction. Among matched encoders, only ResNet1D is significantly higher at both levels, at 263× MACs, 3.6× GPU and 26× CPU latency; InceptionTime and simple_cnn are not significant. The lightweight positioning is an accuracy–efficiency trade-off without overclaiming "no degradation."

### C. Calibration and Threshold Analysis

Calibration and threshold analyses follow the historical locked mainline (thresholds selected only on development OOF; the blind test is not involved): phase ECE 0.036 / Brier 0.042, measurement ECE 0.037 / Brier 0.044; historical blind test ECE 0.040 / Brier 0.045. Measurement-level thresholds 0.5 / max-MCC / recall ≥ 0.5 / recall ≥ 0.8 yield F1 of 0.490 / 0.531 / 0.508 / 0.402 (precision 0.732 / 0.731 / 0.513 / 0.268). Thresholds depend on development OOF, consistent with the one-time blind-test discipline; deployment would incorporate field-specific false-alarm costs (Section VII-B).

### D. Multiple Comparisons and Seed Sensitivity

Thirty-two paired-bootstrap contrasts are reported across the interaction, encoder, and sampling ablations (fourteen from the locked Stage-1 report, eight from the controlled sampling experiment, and ten post-hoc interaction contrasts on the same OOF predictions) without family-wise correction; conclusions are drawn only from contrasts whose 95% CIs exclude zero. The post-hoc contrasts do not change any conclusion: they establish that E3 does not differ significantly from E2 or E5 and that both E4 and E3 significantly outperform no interaction. The two boundary results—mixed vs event at phase level (CI [0.002, 0.122]) and mixed vs random at measurement level (CI [0.010, 0.141])—are close to the significance threshold and are conservatively treated as suggestive rather than conclusive. Second, the seed-42 bootstrap significance for E4 vs E2 (+0.055, CI excluding 0) coexists with a small three-seed fold-mean difference (0.611 vs 0.608, +0.003): the bootstrap difference is computed on pooled OOF probabilities at seed 42, whereas fold-mean comparisons average per-fold scales; we therefore emphasize effect directions and magnitudes over single significance tests, and seed-level bootstrap testing is listed as future work. Third, the K=12 evidence is available at seed 42 only and is treated as development evidence (Section V-E).

---

## VII. Discussion

### A. Module Contribution Analysis

- **Event windows as information anchors for weak supervision.** In the reference-architecture mechanism verification, pure event ≈ mixed strategy (0.591 vs 0.590), indicating that discharge transients concentrate in event-high regions and equidistant windows mainly provide background coverage; the same conclusion holds for E4 in the controlled sampling experiment (0.609 vs 0.615).
- **Coverage density is monotonically effective.** K from 4 to 12 improves performance consistently (0.461 → 0.590 → 0.644 on the 203k reference; 0.639 → 0.670 on the historical lightweight mainline), supporting coverage-aware sampling as an independent contribution.
- **Three-phase interaction is necessary.** E1 (no interaction) is the weakest across all three seeds (seed-42 phase 0.483 ± 0.060, about 0.13 below E4), the historical mechanism verification shows a similar drop without interaction, and single-phase absence causes almost no degradation—three-phase signals contain exploitable redundant structure.
- **Interaction form.** Paired bootstrap shows context-concat (E4) significantly outperforms mean shared-score (E2) and context-add (E5); the difference from max aggregation (E3) is not significant, and E4 is nevertheless the mainline because E3 cannot produce phase-distinct scores (Section V-C). The hierarchical measurement loss (E6) yields no significant gain—a negative result honestly reported.
- **Controlled sampling evidence.** Under a fixed architecture, uniform K=8 is the weakest (0.513/0.537); mixed K=8 outperforms the full-signal baseline (0.615 vs 0.345) at about 12× lower MACs and 4.7× lower GPU latency; significance is claimed only where CIs exclude zero.
- **Lightweight positioning as a trade-off.** The mainline executes 97,201 parameters (113,265 total, including 16,064 non-executed feature-branch weights) with 62.8M MACs and 5.35 ms GPU batch-1 forward latency (pre-extracted windows, Stage-2 same-session retest), fewer parameters than the reference architecture (203,634). Among larger deep baselines only ResNet1D is significantly higher, at about 263× MACs. E4 therefore achieves competitive performance at lower measurement cost rather than claiming higher accuracy.

### B. Limitations and Future Work

1. The independent-test loop is closed (Section V-J): ROC-AUC 0.899 and a prevalence-normalized PR lift of 13.8× versus 10.3× on development confirm preserved discrimination at a 1.57% positive rate; the quantitative prevalence decomposition (Section V-J.3) attributes ≈77% of the apparent PR-AUC drop to the positive-rate difference (Δπ = −0.242) and ≈23% to a genuine domain shift (Δshift = −0.073); additional held-out blocks could further reduce statistical uncertainty.
2. Robustness evidence for the historical locked mainline shows that 5 dB noise drops phase PR-AUC by 67% and fixed-window zero-padding shifts of ±64/±128 cause drops of 34–84% (directionally asymmetric, +128 worst); a random time-shift augmentation variant narrows the drop to within ±3% with development per-fold mean nearly unchanged (0.632 ± 0.066 vs 0.639 ± 0.055). The E4 mainline is substantially more noise-robust (5 dB: +3.7%; Section V-F) but retains the same temporal-alignment sensitivity (−47% to −73%), so the time-shift augmentation variant remains the recommended mitigation for the main model; the augmented variant does not enter the blind test.
3. Cross-device zero-shot transfer remains limited (historical lightweight mainline): effective on motor PD/noise (ROC 0.665 / PR 0.690) but failing on the 28523090 C1→C2 cross-device task (ROC 0.432–0.547, F1 = 0); a small amount of target-domain labeled data restores strong performance from scratch or by fine-tuning (ROC 0.81–0.96).
4. Significance evidence for matched encoders and E1–E6 relies primarily on seed-42 paired bootstrap; E4 has 3 seeds × 5 folds for stability, but seed-level statistical testing and significance evidence for K=12 require additional replication (Section VI-D).
5. The full-signal baseline intentionally keeps the same lightweight architecture to isolate the sampling variable; full-signal models designed for 800,000-point sequences (downsampling, segment aggregation, stronger encoders) are outside this paper's scope and may improve absolute performance at full-data cost.
6. Optional extensions include self-supervised pretraining (VICReg) and larger attention architectures, with gains verified against this paper's baseline; the labeling-fraction experiment (Section V-I) already quantifies the weak-supervision benefit (50% labels → 88% of full-supervision performance).
7. Threshold selection currently relies on development OOF max-MCC/fixed-threshold criteria; deployment must incorporate field-specific false-alarm costs.
8. The Zheng time-frequency encoder under the E4 protocol, classical detector baselines, and sampling-parameter sensitivity are listed in Section V-K [t3].

---

## VIII. Conclusion

This paper investigated coverage-aware information selection and hierarchical weakly supervised structure as joint measurement design variables for extremely long, sparse, weakly labeled three-phase industrial signals, using real VSB PD measurement data. Judicious information selection (event windows and coverage density) substantially reduces the dependency on model capacity while maintaining competitive detection performance at lower cost: the mainline (pure CNN encoder + attention MIL + context-concat; 113,265 total / 97,201 executed parameters) reaches per-phase PR-AUC 0.615 ± 0.053 and measurement-level 0.643 ± 0.053 on 5-fold development cross-validation (three-seed means 0.611 ± 0.006 / 0.639 ± 0.008), an absolute improvement of +0.409 over the strongest traditional baseline (per-fold protocol; +0.335 pooled-OOF), at 62.8M MACs and 5.35 ms GPU batch-1 forward latency. Paired cluster bootstrap shows context-concat significantly outperforms shared-score and additive interaction; the hierarchical measurement loss is a negative result; among matched encoders only ResNet1D is significantly higher, at about 263× MACs, so the paper claims an accuracy–efficiency trade-off rather than the highest accuracy. Controlled sampling experiments demonstrate that information selection itself is the critical design variable: uniform K=8 achieves 0.513/0.537, random K=8 0.579/0.601, and the full-signal baseline 0.345/0.387 at about 12× MACs. The E4 mainline completed a one-time blind test on an independent held-out partition (83,233 phase signals): ROC-AUC 0.899 with a prevalence-normalized PR lift of 13.8× versus 10.3× on development. All experiments were conducted under a hash-verified split lock and one-time evaluation protocol, providing a leakage-safe, reproducible benchmark workflow for long-sequence, weakly labeled, highly imbalanced industrial measurement signals, and revealing the central role of information selection in reducing measurement and model cost.

---

## References

[1] W. J. K. Raymond, H. A. Illias, A. H. Abu Bakar, and H. Mokhlis, "Partial discharge classifications: Review of recent progress," *Measurement*, vol. 68, pp. 164–181, 2015, doi: 10.1016/j.measurement.2015.02.032.

[2] G. C. Stone, "Partial discharge diagnostics and electrical equipment insulation condition assessment," *IEEE Trans. Dielectr. Electr. Insul.*, vol. 12, no. 5, pp. 891–904, 2005, doi: 10.1109/TDEI.2005.1522184.

[3] S. Akram, P. Wang, X. Zhu, J. Huang, F. Liu, Z. Fang, and H. Ahmed, "Prediction of partial discharge inception voltage for electric vehicle motor insulation using deep learning," *IEEE Trans. Instrum. Meas.*, vol. 72, pp. 1–10, 2023, doi: 10.1109/TIM.2023.3269120.

[4] A. Ishaq, M. Junaid, G. A. Hussain, S. U. Khan, Y. Chen, and D. Yu, "Partial discharge defect classification in MV switchgear by using CWT and deep learning approach," *IEEE Trans. Instrum. Meas.*, vol. 74, pp. 1–12, 2025, doi: 10.1109/TIM.2025.3562981.

[5] Y. Wang, J. Yan, Z. Yang, Y. Wu, J. Wang, and Y. Geng, "Generative zero-shot learning for partial discharge diagnosis in gas-insulated switchgear," *IEEE Trans. Instrum. Meas.*, vol. 72, pp. 1–11, 2023, doi: 10.1109/TIM.2023.3264022.

[6] M.-T. Nguyen, V.-H. Nguyen, S.-J. Yun, and Y.-H. Kim, "Recurrent neural network for partial discharge diagnosis in gas-insulated switchgear," *Energies*, vol. 11, no. 5, p. 1202, 2018, doi: 10.3390/en11051202.

[7] J. Zheng, Z. Chen, Q. Wang, H. Qiang, and W. Xu, "GIS partial discharge pattern recognition based on time-frequency features and improved convolutional neural network," *Energies*, vol. 15, no. 19, p. 7372, 2022, doi: 10.3390/en15197372.

[8] Z. Fei, Y. Li, and S. Yang, "Partial discharge pattern recognition based on an ensembled simple convolutional neural network and a quadratic support vector machine," *Energies*, vol. 17, no. 11, p. 2443, 2024, doi: 10.3390/en17112443.

[9] S. Misak, S. Hamacek, P. Bilik, J. Hofinek, and P. Petvaldsky, "Problems associated with covered conductor fault detection," in *Proc. 11th Int. Conf. Electr. Power Quality Utilisation*, 2011, pp. 1–5, doi: 10.1109/EPQU.2011.6128806.

[10] G. M. Hashmi and M. Lehtonen, "On-line PD detection for condition monitoring of covered-conductor overhead distribution networks—A literature survey," in *Proc. 2nd Int. Conf. Electr. Eng.*, 2008, pp. 1–6, doi: 10.1109/ICEE.2008.4553933.

[11] VSB Power Line Fault Detection, Kaggle Competition, 2019. [Online]. Available: https://www.kaggle.com/c/vsb-power-line-fault-detection

[12] *High-Voltage Test Techniques—Partial Discharge Measurements*, IEC 60270:2000+AMD1:2015, International Electrotechnical Commission, 2015.

[13] *High-Voltage Test Techniques—Measurement of Partial Discharges by Electromagnetic and Acoustic Methods*, IEC TS 62478:2016, International Electrotechnical Commission, 2016.

[14] T. G. Dietterich, R. H. Lathrop, and T. Lozano-Pérez, "Solving the multiple instance problem with axis-parallel rectangles," *Artif. Intell.*, vol. 89, no. 1-2, pp. 31–71, 1997.

[15] O. Maron and T. Lozano-Pérez, "A framework for multiple-instance learning," in *Proc. NIPS*, 1998, pp. 570–576.

[16] M. Ilse, J. M. Tomczak, and M. Welling, "Attention-based deep multiple instance learning," in *Proc. ICML*, 2018, pp. 2127–2136.

[17] K. Han and A. M. Y. H. Koay, "HITS: Hierarchical interpretable time series classification via multiple instance learning," in *Proc. IJCNN*, 2025, pp. 1–8, doi: 10.1109/IJCNN64981.2025.11227767.

[18] L. Guo and D. Niu, "Operation condition assessment for elevators based on deep Siamese network and T-S semi-supervision model," *IEEE Trans. Instrum. Meas.*, vol. 73, Art. no. 7503013, 2024.

[19] [Authors to be confirmed], "An open-set semi-supervised contrastive learning for bearing fault diagnosis," *IEEE Trans. Instrum. Meas.*, 2025, doi: 10.1109/TIM.2025.xxxxxxx [metadata to be completed at t6].

[20] S. M. Mahmud, "High precision phase measurement using adaptive sampling," *IEEE Trans. Instrum. Meas.*, vol. 38, no. 5, pp. 954–960, 1989, doi: 10.1109/19.39036.

[21] R. Jaskulke and B. Himmel, "Event-controlled sampling system for marine research," *IEEE Trans. Instrum. Meas.*, vol. 54, no. 4, pp. 1175–1179, 2005, doi: 10.1109/TIM.2005.847142.

[22] J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in *Proc. ICML*, 2006, pp. 233–240, doi: 10.1145/1143844.1143874.

[23] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Trans. Knowl. Discov. Data*, vol. 6, no. 4, Art. no. 15, 2012, doi: 10.1145/2382577.2382579.

[24] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, p. 100804, 2023, doi: 10.1016/j.patter.2023.100804.

[25] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. CVPR*, 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[26] H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar, and P.-A. Muller, "Deep learning for time series classification: A review," *Data Min. Knowl. Discov.*, vol. 33, no. 4, pp. 917–963, 2019, doi: 10.1007/s10618-019-00619-1.

[27] Z. Wang, W. Yan, and T. Oates, "Time series classification from scratch with deep neural networks: A strong baseline," in *Proc. IJCNN*, 2017, pp. 1578–1585, doi: 10.1109/IJCNN.2017.7966039.

[28] H. Ismail Fawaz et al., "InceptionTime: Finding AlexNet for time series classification," *Data Min. Knowl. Discov.*, vol. 34, no. 6, pp. 1936–1962, 2020, doi: 10.1007/s10618-020-00710-y.

[29] S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," arXiv:1803.01271, 2018.

[30] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. NeurIPS*, 2019, pp. 8024–8035.

[31] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[32] W. McKinney, "Data structures for statistical computing in Python," in *Proc. SciPy*, 2010, pp. 56–61, doi: 10.25080/Majora-92bf1922-00a.

[33] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Comput. Sci. Eng.*, vol. 9, no. 3, pp. 90–95, 2007, doi: 10.1109/MCSE.2007.55.

[34] C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020, doi: 10.1038/s41586-020-2649-2.
