# Coverage-Aware Sampling and Lightweight Hierarchical Weakly-Supervised Detection for Three-Phase Partial Discharge Signals

> Manuscript Revision (2026-08-18)
> Data Source: VSB Power Line Fault Detection (Kaggle, 2019)
> Status: Revision (Stage 1 + Stage 2 experimental evidence integrated); E4 has completed a one-time blind test on the Harvard Dataverse independent held-out set (83,233 phase signals, PR-AUC 0.216, 95% CI [0.199, 0.234]), closing the independent test loop; the 423 blind test was evaluated only once on the historical mainline; all numerical values originate from locked experimental records (docs/research_report.md, results/stage1_tim/report_A_E.md, and results/stage2_sampling/report_sampling.md); measurement-cluster statistical uncertainty analysis is presented in Section 5.2

## Abstract

Real-world partial discharge (PD) signals feature long sequences, strong noise, weak labels, and extreme class imbalance. This paper proposes a coverage-aware event-centric window sampling and phase-aware hierarchical weakly-supervised detection framework. We define an event score (robust non-negative z-score of amplitude, Teager energy, and differential RMS) and deterministically extract K short windows per phase via equidistant anchors and event windows with cross-type deduplication (K=8, ~8.2% of raw data). A lightweight 1D CNN encoder maps each window into a 128-dimensional representation, aggregated via Attention MIL and fused through context-concat three-phase interaction. On the VSB dataset (2,481 development measurements, 5.9% phase positive rate), the mainline (113,265 parameters) achieves per-phase PR-AUC 0.615 ± 0.053 and measurement-level 0.643 ± 0.053 on 5-fold CV, a +0.409 absolute improvement over the strongest traditional baseline. Paired bootstrap (2,000 resamples) shows context-concat significantly outperforms mean and additive interaction. Controlled experiments confirm that information selection itself, not model architecture, is the critical design variable. The 423 blind-test measurements maintain a one-time evaluation protocol with frozen split locks and hash verification, ensuring leakage-free, reproducible conclusions.

**Keywords**: Partial discharge detection; multiple instance learning; weakly supervised learning; three-phase signals; coverage-aware sampling; leakage-safe evaluation

---

## 1 Introduction

Partial discharge (PD) is an early indicator of insulation degradation in power equipment, and its accurate detection is a critical component of condition-based maintenance and fault warning. In practical industrial scenarios, PD signals are sampled at 40 MHz, with a single-phase signal reaching 800,000 points, and discharge events are highly sparse in time—in the VSB dataset used in this paper, only approximately 5.9% of phase signals carry a discharge label. These characteristics present three challenges:

1. **Long sequences and sparse events**: Strongly supervised modeling of entire signals incurs high computational cost, and since discharge pulses occupy only extremely brief moments, direct classification is easily overwhelmed by vast amounts of noise frames;
2. **Weak labels and hierarchical structure**: The data provides only phase-level labels (whether each phase contains discharge) or measurement-level labels (whether the three-phase signal as a whole is abnormal), lacking window-level annotations, necessitating multiple instance learning (MIL) where windows are treated as instances and trained with bag-level labels;
3. **Class imbalance**: The positive phase rate is below 6%, making PR-AUC more informative than ROC-AUC for assessing practical discriminative ability under low positive rates.

Furthermore, most published methods lack explicit control over data leakage in their experimental pipelines: if signal segments from the same measurement inadvertently appear in both training and test sets, results become inflated and irreproducible.

To address these challenges, this paper proposes a measurement-oriented detection framework centered on "information selection + hierarchical weakly-supervised learning + measurement-oriented efficiency characterization + leakage-safe evaluation." The main contributions are as follows:

1. **Coverage-aware information selection (Contribution 1)**: We provide a complete mathematical definition of the event score and a deterministic "equidistant anchors + event windows + cross-type deduplication + hierarchical fallback" sampling pipeline, extracting only K short windows from each 800,000-point phase signal (at K=8, only approximately 8.2% of the data is processed). Through controlled experiments with a fixed architecture, we demonstrate that event focus and coverage density jointly determine the trade-off between diagnostic performance and computational cost;
2. **Hierarchical weakly-supervised detection framework (Contribution 2)**: With no window-level labels but phase-level labels available, we employ window-to-phase MIL aggregation for weakly supervised modeling, followed by context-concat three-phase interaction to output independent per-phase probabilities, and a deterministic noisy-OR to obtain measurement-level decisions. The final lightweight mainline is pure CNN + Attention MIL + context-concat (113,265 parameters). On 5-fold development set CV (seed 42), it achieves per-phase PR-AUC 0.615 ± 0.053 and measurement-level 0.643 ± 0.053; paired bootstrap demonstrates statistically significant superiority over Mean shared-score and context-add baselines;
3. **Measurement-oriented efficiency characterization (Contribution 3)**: On a unified platform, we report per-component latency, MACs, memory, and throughput for event scoring, peak detection, window selection, normalization, encoder, MIL, interaction, and output, clearly distinguishing between the sampling and model forward passes and explicitly linking performance, information coverage, and computational cost;
4. **Leakage-safe reproducible evaluation (Contribution 4)**: Through data leakage auditing, hash-verified split locks, and one-time blind testing, all model selection, hyperparameters, and thresholds are determined solely by development set cross-validation. The 423 strictly held-out measurements maintain the "evaluated only once" protocol and are not reopened for this upgrade.

Experiments on the VSB real-world industrial data validate the above design: the final lightweight mainline (113,265 parameters) achieves per-phase PR-AUC 0.615 ± 0.053 and measurement-level 0.643 ± 0.053 on 5-fold development set CV (seed 42), with an absolute improvement of +0.409 over the strongest traditional machine learning baseline (flattened features + LightGBM, per-fold mean 0.206) under the same mixed K=8 feature cache. The three-seed mean is per-phase 0.611 ± 0.006 and measurement-level 0.639 ± 0.008. Paired bootstrap shows that context-concat significantly outperforms Mean shared-score and context-add, while the difference from Max interaction is not significant, and the hierarchical measurement loss does not yield a significant gain. The frozen blind-test set maintains the one-time protocol: the 423 strictly held-out measurements were previously evaluated once by the historical locked mainline, and this paper neither reopens them nor attributes the historical blind-test numbers to the upgraded mainline. The reference architecture (203,634 parameters) serves only as mechanism verification and structural selection evidence and is not part of the main report. E4 has completed a one-time blind test on the Harvard Dataverse independent held-out set (Section 4.13): 83,233 phase signals, phase-level PR-AUC 0.216 (95% CI [0.199, 0.234]), ROC-AUC 0.899, and the independent test loop is closed.

---

## 2 Related Work

### 2.1 Partial Discharge Measurement and Diagnosis

Traditional PD detection relies primarily on handcrafted statistical features, including pulse amplitude, phase-resolved partial discharge (PRPD) patterns, and time/frequency-domain statistics, combined with classifiers such as support vector machines and neural networks [4][5]. The limitation of these approaches is that feature design depends on domain expertise and it is difficult to directly leverage the sparse transient structure in raw long sequences. In recent years, deep learning methods have been progressively applied to discharge pattern recognition: recurrent neural networks have been used for discharge diagnosis in gas-insulated switchgear [9]; time-frequency features combined with improved convolutional neural networks have achieved good results in GIS discharge pattern recognition [10]; ensemble lightweight convolutional networks have also been applied to discharge pattern classification [11]. In instrumentation and measurement-oriented work, PD diagnosis has been further discussed in the context of the measurement pipeline: for example, deep learning has been used to predict PD inception voltage in electric vehicle motor insulation [6], CWT-based deep networks have been applied to discharge defect classification in medium-voltage switchgear [7], and generative zero-shot learning has been used for GIS discharge diagnosis [8]. These works demonstrate that the key to PD diagnosis lies not only in the classifier but also in the representation and selection of the measured signal. However, most of them model fixed-length segments or PRPD images, generally lacking ablation studies on "how windows are selected"—a critical step—and rarely consider the hierarchical weakly supervised nature of the labels.

### 2.2 Covered Conductor Fault Detection (VSB Task Background)

The data in this paper originates from the VSB Power Line Fault Detection competition [20], whose industrial background is fault detection in covered conductor distribution lines: under mechanical damage, water tree degradation, and other conditions, covered conductors may develop PD, which can further evolve into permanent faults such as conductor breakage [12][13]. Existing studies have systematically summarized the problems faced by covered conductor fault detection and reviewed online PD monitoring solutions [12][13]. The challenge of this public dataset lies in the high sampling rate (40 MHz, 800,000 points per phase per measurement), phase-level (or even measurement-level) labels, and highly sparse discharge events. Existing methods are mostly combinations of statistical features and classifiers, or direct strongly supervised modeling of entire signals, with insufficient study of the combined "information selection + hierarchical weakly supervised" approach.

### 2.3 Weakly Supervised Learning and Multiple Instance Learning

Multiple instance learning (MIL), introduced by Dietterich et al., assumes that labels exist only at the bag level, where each bag consists of several instances [1]. Early work primarily relied on geometric assumptions such as axis-parallel rectangles [1][2]. With the rise of deep learning, attention-based deep MIL has been shown to automatically identify key instances within a bag [3], and gated attention variants have further enhanced expressiveness; recent work has also extended MIL to hierarchical time series classification [17]. The phase-level modeling in this paper can be viewed as weakly supervised learning with "window-level instances → phase-level labels": windows lack labels while phases have labels. Unlike standard MIL, the instances (windows) within a bag are not independently and identically distributed samples but are deterministically constructed by the coverage-aware strategy, so information selection and weakly supervised learning are directly coupled.

### 2.4 Sparse and Event-Driven Measurement Processing

For extremely long sparse measurement signals, reducing the amount of processed data is a key concern in measurement system design. Adaptive sampling and event-triggered sampling have been extensively studied in instrumentation and measurement [18][19], sharing the common idea of increasing sampling density only in signal regions with significant changes to reduce data volume, power consumption, and processing cost. In our scenario, similar information selection occurs at the front end of the detection pipeline: uniform segmentation is simple but insensitive to sparse events, while event-driven segmentation depends on detector quality. Our hybrid strategy can be viewed as a combination of both—equidistant anchors ensure full coverage, event windows focus on high-response regions, and cross-type deduplication avoids redundancy. At the evaluation level, PR-AUC is more sensitive to class imbalance than ROC-AUC and is widely used for such problems [14]. Moreover, "separation of model selection from evaluation" has become a fundamental requirement for reproducible research—data leakage (the same measurement appearing in both training and test sets) inflates metrics, and Kaufman et al. provided a formal definition and detection method for leakage [15]; Kapoor and Narayanan further demonstrated that leakage is a major source of the reproducibility crisis in machine learning research [16]. The frozen split and one-time blind-test protocol in this paper are designed specifically to address these risks.

---

## 3 Method

### 3.1 Problem Formulation

Let a measurement consist of three-phase signals $x_A, x_B, x_C$, each of length $L = 800,000$ samples at sampling rate $f_s = 40$ MHz. The phase-level label $y_p \in \{0, 1\}$ indicates whether phase $p$ contains discharge. The measurement-level label can be derived from the three-phase signals via aggregation (this paper uses the same noisy-OR logic as the evaluation metric: a measurement is abnormal if any phase contains discharge). During training, only phase-level labels are available. The task is to learn a per-phase discriminator $f_p: x_p \rightarrow \hat{y}_p$ ($p \in \{A, B, C\}$), from which measurement-level probabilities can be derived via deterministic noisy-OR. The final mainline's context-concat interaction (Section 3.6) outputs independent, rankable per-phase logits for the three phases; therefore, "phase-level PR-AUC" in this paper refers to the ranking evaluation of independent per-phase probabilities, and measurement-level decisions are obtained through hierarchical inference.

The overall detection framework of the proposed method is illustrated in Fig. 1, comprising four components: coverage-aware window sampling, lightweight window encoding, MIL aggregation, and three-phase interaction.

![Fig. 1 Coverage-aware multi-window sampling and hierarchical weakly-supervised detection framework](figures/fig1_architecture.png)

### 3.2 Coverage-Aware Deterministic Window Sampling

For each phase signal, $K$ windows of length 8,192 samples (204.8 $\mu$s) are extracted via the following procedure:

- **Equidistant anchor windows**: $K_u$ uniformly spaced positions with maximally spaced starting points, ensuring full-coverage of the signal;
- **Event windows**: $K_e$ positions selected by the event score defined in Section 3.2.1, choosing high-scoring locations with non-overlapping starting points, focusing on suspected discharge regions;

#### 3.2.1 Event Score and Event Window Selection (Locked Implementation)

For each phase signal $x$ (length $L = 800,000$), first subtract the median to obtain $\tilde{x}[n] = x[n] - \text{median}(x)$, then compute three energy features pointwise, take the robust non-negative z-score for each, and finally take the pointwise maximum to obtain the event score $S[n]$:

- **Amplitude feature**: $a[n] = |\tilde{x}[n]|$;
- **Teager energy**: $\tau[n] = |\tilde{x}[n]^2 - \tilde{x}[n-1] \cdot \tilde{x}[n+1]|$;
- **Rolling differential RMS**: $d[n] = \sqrt{\text{mean}_{m \in \text{window}}(\Delta\tilde{x}[m]^2)}$, where $\Delta\tilde{x}$ is the first-difference sequence (leading edge padded with reflection), with a window width of 256 (reflection padding, equivalent to the root-mean-square of differential energy over a sliding window of 256 points).

The robust non-negative z-score is defined as:

$$z(v) = \max\left(\frac{v - \text{median}(v)}{1.4826 \cdot \text{MAD}(v)}, 0\right) \qquad (1)$$

where $\text{MAD}(v) = \text{median}(|v - \text{median}(v)|)$. When MAD degenerates (zero or non-finite), it falls back to the mean absolute deviation scale. The event score is defined as:

$$S[n] = \max(z(a[n]), z(\tau[n]), z(d[n])) \qquad (2)$$

with $S[n] \geq 0$.

The coverage-aware deterministic window sampling procedure is summarized in Algorithm 1.

**Algorithm 1** Coverage-Aware Deterministic Window Sampling
---

**Require:** Phase signal $x \in \mathbb{R}^{L}$ ($L = 800{,}000$), $K_u = K_e = 4$, window length $W = 8{,}192$

**Ensure:** Window set $\mathcal{W}$ with $|\mathcal{W}| = K_u + K_e = 8$
1: $\mathcal{W} \gets \emptyset$
2: $\tilde{x}[n] \gets x[n] - \text{median}(x)$ for $n = 1,\dots,L$
3: Compute amplitude $a[n] \gets |\tilde{x}[n]|$
4: Compute Teager energy $\tau[n] \gets |\tilde{x}[n]^2 - \tilde{x}[n-1] \cdot \tilde{x}[n+1]|$
5: Compute differential RMS $d[n] \gets \sqrt{\text{mean}_{m\in\text{window}}(\Delta\tilde{x}[m]^2)}$ (width 256, reflection padding)
6: $S[n] \gets \max\{z(a[n]),\; z(\tau[n]),\; z(d[n])\}$ ▷ Eq. (4)
7: **Phase I — Equidistant anchors:** Place $K_u$ uniformly spaced windows with maximally separated starting points; add to $\mathcal{W}$
8: **Phase II — Event windows:**
9:  $P \gets$ peaks of $S$ with $\text{dist}_{\min}=W/2$ and $S>0$
10:  Sort $P$ by $(-S,\;\text{start})$ descending
11:  **for** $p \in P$ **do**
12:   $s \gets \text{clip}(\text{peak}_p - W/2,\;0,\;L-W)$
13:   **if** $\text{IoU}([s,s+W],\; w) < 0.5$ for all $w \in \mathcal{W}$ **then**
14:    $\mathcal{W} \gets \mathcal{W} \cup \{[s, s+W]\}$
15:   **end if**
16:   **if** $|\mathcal{W}| = K_u + K_e$ **then break**
17:  **end for**
18: **Phase III — Fallback:** **while** $|\mathcal{W}| < K_u + K_e$ **do**
19:  Select position from 256-bin grid maximizing $\min_{w \in \mathcal{W}} \text{dist}(w)$
20:  Add to $\mathcal{W}$
21: **end while**
22: **return** $\mathcal{W}$

---

The locked parameters are: $\text{rolling\_width} = 256$, $\text{peak\_distance} = 4{,}096$, $\text{window\_length} = 8{,}192$, $\text{dedup\_IoU} = 0.5$, $\text{fallback\_grid} = 256$ ($K_u = 4$, $K_e = 4$). The above procedure and parameters are implemented in `src/vsb_pd/events.py` and written into the split lock via SHA-256, ensuring that "identical input always produces identical output." Ablation studies on window strategy (K and composition) are presented in Section 4.9.

Fig. 2 illustrates the effect of mixed window sampling on a real signal using measurement 705, phase C.

![Fig. 2 Mixed window sampling example (measurement 705, phase C, with window annotations)](figures/fig2_window_sampling.png)

### 3.3 Physical Feature Extraction

For each window, a 58-dimensional physical feature vector is extracted, serving as input to the encoder's feature branch, computed entirely in a vectorized manner:

| Category | Dimensions | Description |
|----------|-----------|-------------|
| Time-domain | 20 | Amplitude, variance, skewness, kurtosis, crest factor, zero-crossing rate, energy, etc. |
| Frequency-domain | 12 | Dominant frequency, spectral centroid, bandwidth, roll-off, flatness, spectral entropy, etc. (FFT) |
| Band energy | 13 | Normalized energy across 13 equally-spaced bands from 0–20 MHz |
| Autocorrelation/AR | 9 | First 3 ACF peaks and their significance + 3rd-order Burg AR coefficients |
| Envelope/Peaks | 4 | Number of peaks, mean significance, Hilbert envelope mean/std |

Feature extraction is verified for identity and finiteness using constant signals and random signals, ensuring numerical stability.

### 3.4 Window Encoder (Locked Reference and Final Lightweight Mainline)

The final lightweight mainline uses a pure 1D CNN window encoder (E4). Let the input window be $\mathbf{w} \in \mathbb{R}^{8192}$ (raw signal) and the physical feature vector be $\mathbf{f} \in \mathbb{R}^{58}$. The encoder processes the raw signal through three 1D convolutional blocks:

$$\mathbf{h}_1 = \text{ReLU}(\text{BN}(\text{Conv1D}_{16,64}(\mathbf{w}))) \qquad (3)$$
$$\mathbf{h}_2 = \text{ReLU}(\text{BN}(\text{Conv1D}_{16,64}(\mathbf{h}_1))) \qquad (4)$$
$$\mathbf{h}_{\text{raw}} = \text{ReLU}(\text{BN}(\text{Conv1D}_{16,128}(\mathbf{h}_2))) \qquad (5)$$

where $\text{Conv1D}_{C,K}$ denotes a 1D convolution with $C$ output channels and kernel size $K$, followed by batch normalization (BN) and ReLU activation. The output is global average pooled to obtain a raw-signal representation $\mathbf{h}_{\text{raw}} \in \mathbb{R}^{128}$. The physical features are processed by a 2-layer MLP:

$$\mathbf{h}_{\text{feat}} = \text{ReLU}(\text{Linear}_{58,64}(\mathbf{f})) \qquad (6)$$
$$\mathbf{h}_{\text{feat}} = \text{Linear}_{64,64}(\mathbf{h}_{\text{feat}}) \qquad (7)$$

The final window representation is the sum of the two branches:

$$\mathbf{h} = \mathbf{h}_{\text{raw}} + \mathbf{h}_{\text{feat}} \qquad (8)$$

with $\mathbf{h} \in \mathbb{R}^{128}$. The reference architecture (203,634 parameters) uses a dual-branch fusion where the raw-signal encoder is a deeper CNN (4 blocks, channels 16→32→64→128) and the feature branch is a 3-layer MLP, combined via learned gating.

### 3.5 Window-to-Phase MIL Aggregation

For a phase signal with $K$ windows, we obtain $K$ window representations $\{\mathbf{h}_1, \ldots, \mathbf{h}_K\}$, each $\in \mathbb{R}^{128}$. The final mainline uses Attention MIL [3] to aggregate windows into a phase-level representation:

$$\alpha_k = \frac{\exp(\mathbf{v}^\top \tanh(\mathbf{W}\mathbf{h}_k))}{\sum_{j=1}^K \exp(\mathbf{v}^\top \tanh(\mathbf{W}\mathbf{h}_j))} \qquad (9)$$
$$\mathbf{z}_p = \sum_{k=1}^K \alpha_k \mathbf{h}_k \qquad (10)$$

where $\mathbf{W} \in \mathbb{R}^{128 \times 128}$ and $\mathbf{v} \in \mathbb{R}^{128}$ are learnable parameters, $\alpha_k$ is the attention weight for window $k$, and $\mathbf{z}_p \in \mathbb{R}^{128}$ is the aggregated phase representation. The reference architecture (203k) additionally includes a gating mechanism: $\alpha_k \propto \exp(\mathbf{v}^\top (\tanh(\mathbf{W}\mathbf{h}_k) \odot \sigma(\mathbf{U}\mathbf{h}_k)))$.

### 3.6 Phase-Aware Interaction

The final mainline uses **context-concat** (E4) for three-phase interaction. Let $\mathbf{z}_A, \mathbf{z}_B, \mathbf{z}_C \in \mathbb{R}^{128}$ be the aggregated representations for the three phases. The context vector is:

$$\mathbf{c} = \frac{1}{3}(\mathbf{z}_A + \mathbf{z}_B + \mathbf{z}_C) \qquad (11)$$

For each phase $p$, the final representation is:

$$\mathbf{z}_p' = [\mathbf{z}_p, \mathbf{c}] \in \mathbb{R}^{256} \qquad (12)$$

followed by a linear classifier:

$$\hat{y}_p = \sigma(\mathbf{w}_p^\top \mathbf{z}_p' + b_p) \qquad (13)$$

where $\sigma$ is the sigmoid function, producing an independent probability for each phase. The measurement-level probability is obtained via deterministic noisy-OR:

$$\hat{y}_{\text{meas}} = 1 - \prod_{p \in \{A,B,C\}} (1 - \hat{y}_p) \qquad (14)$$

Compared variants include:
- **Mean shared-score** (E2): $\hat{y}_p = \sigma(\mathbf{w}^\top \mathbf{z}_p + b)$, all phases share the same classifier with no interaction;
- **Max aggregation** (E3): $\mathbf{z}_p' = \max(\mathbf{z}_p, \mathbf{c})$ elementwise;
- **Context-add** (E5): $\mathbf{z}_p' = \mathbf{z}_p + \mathbf{c}$;
- **Hierarchical loss** (E6): E4 architecture with an additional measurement-level BCE loss: $\mathcal{L} = \mathcal{L}_\text{phase} + \lambda \mathcal{L}_\text{meas}$, where $\lambda \in \{0.25, 0.5, 1.0\}$.

### 3.7 Training Protocol

All models are trained with the AdamW optimizer (learning rate $1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$) with a batch size of 64 (8 for the full-signal baseline), for up to 40 epochs with early stopping (patience 15) based on validation phase-level PR-AUC. Training uses stratified 5-fold cross-validation with StratifiedGroupKFold (seed 42, with seeds 7 and 2024 for stability verification). The loss function is phase-level binary cross-entropy (BCE) for all variants except E6, which adds a measurement-level BCE term.

---

## 4 Experiments

### 4.1 Dataset and Evaluation Protocol

**VSB Power Line Fault Detection Dataset**: The dataset consists of 3,327 measurements, each containing three-phase signals of 800,000 samples at 40 MHz. The standard split uses 2,904 measurements (2,481 development + 423 strictly held-out blind test) as provided by the data source. The phase positive rate is approximately 5.9% on the development set and approximately 5.4% on the blind test set. All experiments in this paper use the development set only; the 423 blind-test measurements maintain the one-time evaluation protocol from the historical locked mainline.

**Harvard Dataverse Independent Held-Out Set**: An additional 83,233 phase signals (28,285 measurements, 1,308 positive phases at 1.57% positive rate) from the same VSB data source, obtained from Harvard Dataverse (doi:10.7910/DVN/JYJJ5W). This set was never used in any training, model selection, hyperparameter tuning, or threshold determination. It was evaluated exactly once by the final frozen E4 model (Section 4.13).

**Evaluation Metrics**: Primary metric is PR-AUC (phase-level and measurement-level), supplemented by ROC-AUC, MCC, F1, accuracy, ECE (expected calibration error), and Brier score. All metrics are reported as per-fold mean ± standard deviation across 5 folds. Statistical significance is assessed via paired cluster bootstrap (2,000 resamples, clustered by measurement ID) with 95% confidence intervals. The three-seed mean (seeds 42, 7, 2024) provides additional stability verification.

**Data Leakage Control**: All experiments use frozen split locks with SHA-256 hash verification. The 423 blind-test measurements were evaluated only once by the historical locked mainline and are not reopened for this upgrade. The Harvard Dataverse set was evaluated exactly once with a tamper-evident receipt.

### 4.2 Traditional Machine Learning Baselines

To establish a reference point, we compare against traditional ML classifiers using the same mixed K=8 feature cache (58-dimensional physical features per window, concatenated across K windows for a 464-dimensional feature vector per phase). All baselines share the same 5-fold split (seed 42) and feature cache.

| Classifier | Phase PR-AUC (per-fold mean) | Measurement PR-AUC | Pooled-OOF Phase |
|-----------|------------------------------|-------------------|------------------|
| Flatten + LightGBM | **0.206 ± 0.048** | **0.197** | **0.198** |
| Flatten + XGBoost | 0.181 ± 0.064 | 0.183 | 0.177 |
| Flatten + Random Forest | 0.151 ± 0.045 | 0.153 | 0.143 |
| Flatten + SVM (RBF) | 0.103 ± 0.033 | 0.112 | 0.098 |
| Per-window mean + LightGBM | 0.109 ± 0.030 | 0.113 | 0.101 |

The strongest traditional baseline (flattened features + LightGBM, per-fold mean 0.206) is used as the primary reference for absolute improvement. The E4 mainline (0.615 ± 0.053) achieves an absolute improvement of +0.409 over this baseline under the same folds and feature cache.

### 4.3 Window Sampling Strategy (K=8, Mixed)

The mainline uses K=8 with 4 equidistant anchor windows and 4 event windows, processing approximately 8.2% of the raw signal data. The complete ablation of window strategies (K values, compositions, and comparison with equidistant-only, event-only, and random sampling) is presented in Section 4.9 (mechanism verification on the reference architecture) and Section 4.11 (controlled sampling experiment on the E4 architecture).

### 4.4 Phase Interaction and Hierarchical Loss Ablation (E1–E6)

All six variants share the same backbone (pure CNN encoder + Attention MIL) and differ only in the phase interaction mechanism:

| Variant | Interaction | Phase PR-AUC (seed 42) | Meas PR-AUC | Phase pooled-OOF | Meas pooled-OOF |
|---------|------------|----------------------|-------------|-----------------|----------------|
| E1 | No interaction | 0.483 ± 0.060 | 0.562 ± 0.053 | 0.371 | 0.421 |
| E2 | Mean shared-score | 0.554 ± 0.058 | 0.588 ± 0.058 | 0.500 | 0.540 |
| E3 | Max aggregation | 0.589 ± 0.053 | 0.614 ± 0.051 | 0.500 | 0.546 |
| **E4** | **Context-concat** | **0.615 ± 0.053** | **0.643 ± 0.053** | **0.533** | **0.585** |
| E5 | Context-add | 0.556 ± 0.058 | 0.577 ± 0.058 | 0.465 | 0.509 |
| E6 | Hierarchical loss (λ=0.25) | 0.586 ± 0.081 | 0.615 ± 0.090 | 0.474 | 0.512 |

Three-seed mean (seeds 42, 7, 2024) for E4: phase PR-AUC 0.611 ± 0.006, measurement PR-AUC 0.639 ± 0.008.

Paired cluster bootstrap (2,000 resamples, seed 42, measurement-clustered) reveals:
- E4 vs E2 (Mean shared): phase +0.055 [95% CI +0.006, +0.103], **significant**
- E4 vs E5 (Context-add): phase +0.073 [+0.028, +0.117], **significant**
- E4 vs E3 (Max): phase +0.036 [−0.012, +0.087], not significant
- E6 vs E4 (Hierarchical loss): phase −0.030 [−0.084, +0.024], not significant (negative result)

E1 (no interaction) is the weakest across all three seeds, confirming that three-phase interaction is a necessary component.

### 4.5 Matched Encoder and Published Method Baselines

To test the hypothesis that "judicious information selection can maintain competitive performance at lower model capacity," we fix all components (robust preprocessing → encoder → Attention MIL → context-concat → phase classifier, phase-level BCE) and replace only the encoder under the same K=8 mixed windows, seed 42, batch=64/epochs=40/patience=15 protocol. MACs and latency are from results/stage1_tim/benchmark.json (synthetic 800k signals, 50 repetitions, p50).

| Encoder | Parameters | MACs/Meas | GPU b1 p50 (ms) | CPU b1 p50 (ms) | Phase PR-AUC | Meas PR-AUC | Pooled-OOF Phase | Pooled-OOF Meas |
|---------|-----------|-----------|-----------------|-----------------|-------------|------------|-----------------|----------------|
| cnn (E4) | 113,265 | 62.8M | 1.94 | 8.34 | 0.615 ± 0.053 | 0.643 ± 0.053 | 0.533 | 0.585 |
| simple_cnn | 96,945 | 62.8M | 1.73 | 8.54 | 0.633 ± 0.068 | 0.656 ± 0.055 | 0.526 | 0.568 |
| ResNet1D | 603,170 | 16.53G | 7.04 | 216.93 | 0.688 ± 0.028 | 0.720 ± 0.040 | 0.626 | 0.647 |
| InceptionTime | 627,186 | 55.70G | 21.87 | 1031.70 | 0.636 ± 0.071 | 0.673 ± 0.092 | 0.575 | 0.634 |

Paired bootstrap (seed 42, 2,000 resamples) shows: ResNet1D relative to cnn has a per-phase difference of +0.095 (95% CI [+0.033, +0.152]) and measurement-level +0.064 ([+0.007, +0.119]), the only significantly higher encoder, but at approximately 263× MACs, approximately 3.6× GPU batch-1 latency, and approximately 26× CPU latency. InceptionTime's differences are per-phase +0.044 ([−0.005, +0.092]) and measurement-level +0.051 ([−0.003, +0.102]), with 95% CI containing 0, while MACs are approximately 887× and GPU latency approximately 11×. simple_cnn vs cnn differences are not significant (per-phase −0.005, [−0.048, +0.041]). Therefore, the lightweight positioning is an accuracy-efficiency trade-off: 113,265 parameters, 62.8M MACs, and approximately 1.9 ms GPU batch-1 latency achieve competitive discriminative quality compared to larger encoders, though larger models can improve the performance ceiling at substantial cost.

Published methods adapted to this framework with the earlier historical protocol (Mean MIL + Max interaction) using the same folds and feature cache:

| Model | Parameters | Historical Protocol | Phase PR-AUC | Meas PR-AUC |
|-------|-----------|-------------------|-------------|------------|
| Zheng TF-CNN 2022 | 134,113 | STFT+2D CNN; Mean MIL + Max | 0.715 ± 0.050 | 0.737 |
| Fei CNN+QSVM 2024 | 55,472 CNN + 3,989 SV | Ensemble CNN+QSVM | 0.495 ± 0.053 | 0.632 |

Zheng's time-frequency 2D CNN achieves the highest performance under the historical protocol but relies on STFT time-frequency representation with larger parameter count. Fei's CNN+QSVM per-fold mean phase-level performance is approximately 0.120 below E4, showing that "lightweight without window-level weakly supervised structure" is insufficient.

### 4.6 Ablation Experiments: Phase Interaction and Hierarchical Loss

**Phase Interaction and Hierarchical Loss (E1–E6, seed 42)**: Section 4.4 presents the per-fold mean results. For E6, λ ∈ {0.25, 0.5, 1.0} yields seed-42 development set pooled-OOF measurement PR-AUC of 0.512 / 0.430 / 0.492 (corresponding phase PR-AUC 0.475 / 0.389 / 0.446); λ=0.25 is locked by the selection rule, and verified on seeds 7/2024 (phase 0.606 ± 0.085 / 0.617 ± 0.065, measurement 0.635 ± 0.086 / 0.639 ± 0.076). E6's seed-42 result (0.586 ± 0.081 / 0.615 ± 0.090) is below E4, and the paired 95% CI contains 0—a negative result: the hierarchical measurement loss does not provide a significant gain. E1 (no interaction) is the weakest across all three seeds (seed 42 phase 0.483 ± 0.060 / measurement 0.562 ± 0.053), confirming that three-phase interaction is a necessary module.

**Reference Architecture Mechanism Verification (203k, historical mechanism reference; not part of the main report)**:

**Encoder** (gated_attention + recurrent phase interaction):

| Encoder | Phase PR-AUC | Parameters |
|---------|-------------|-----------|
| Pure CNN | 0.614 ± 0.073 | 178,546 |
| Dual-branch fusion (203k reference) | 0.590 ± 0.081 | 203,634 |
| Pure statistical feature MLP | 0.208 ± 0.070 | 186,866 |

**MIL Aggregation** (dual-branch + recurrent phase):

| Aggregation | Phase PR-AUC |
|------------|-------------|
| Mean | 0.611 ± 0.092 |
| Attention | 0.603 ± 0.065 |
| Gated-Attention (203k reference) | 0.590 ± 0.081 |
| Max | 0.559 ± 0.045 |

**Phase Interaction** (dual-branch + gated_attention):

| Interaction | Phase PR-AUC |
|------------|-------------|
| Max aggregation | 0.620 ± 0.039 |
| Mean aggregation | 0.612 ± 0.070 |
| Direct concatenation | 0.599 ± 0.014 |
| Recurrent symmetric convolution (203k reference) | 0.590 ± 0.081 |
| No interaction | 0.474 ± 0.087 |

The mechanism verification yields three conclusions: (1) In the historical mechanism verification, phase interaction is the largest contributor (+0.12~0.15), consistent with the E1–E6 finding that "no interaction is significantly weakest"; (2) Pure CNN and simple Mean/Max aggregation interactions perform at mean levels not lower than complex modules, supporting that "lightweight can achieve comparable mean performance without significant sacrifice," though fold-wise standard deviations are larger (±0.04~0.09) and differences are not asserted as significant; (3) Complex recurrent equivariant and gated attention modules do not provide clear gains. The E1–E6 paired bootstrap further quantifies the phase interaction differences: context-concat significantly outperforms mean/add, while the difference from max is not significant, and the hierarchical measurement loss is a negative result.

Fig. 6 shows the historical reference architecture mechanism verification phase PR-AUC comparison.

![Fig. 6 Ablation experiments (historical reference architecture mechanism verification)](figures/fig6_ablation.png)

### 4.7 Measurement Imperfection Robustness (Inference-Time Perturbation; Historical Locked Mainline)

**(Historical locked mainline, development set OOF; 203k reference architecture serves only as mechanism reference; E4 perturbation robustness was not retested, this section serves as mechanism evidence only)**:

| Perturbation | Phase PR-AUC Relative Drop | Measurement-Level Relative Drop |
|-------------|---------------------------|-------------------------------|
| Gaussian noise 20 dB | −10.8% | −12.8% |
| Gaussian noise 10 dB | −49.4% | −50.7% |
| Gaussian noise 5 dB | −67.2% | −67.5% |
| Amplitude ×0.8 / ×1.2 | 0% | 0% |
| Time shift −64 / +64 | −33.8% / −73.9% | −33.6% / −74.7% |
| Time shift −128 / +128 | −75.3% / −84.1% | −75.8% / −83.9% |
| Missing any phase | — | ≈0% |

The robustness evaluation uses the historical locked mainline (pure CNN + Attention MIL + Mean interaction) with 5-fold checkpoints, development set per-fold mean 0.622 ± 0.064. This run's OOF baseline is phase PR-AUC 0.568, measurement-level 0.599, with measurement-level stratified bootstrap 95% CI [0.541, 0.659] (median 0.601, 2,000 resamples). The difference from the historical locked mainline's another run OOF (0.537/0.577) falls within the historical run's own measurement-cluster bootstrap interval (results/measurement_uncertainty.json). Absolute performance under noise decreases monotonically with SNR: at 20 dB, phase PR-AUC is 0.506 (−10.8%); at 5 dB, it drops to 0.186. Amplitude scaling and single-phase absence are completely insensitive (noisy-OR three-phase redundancy is effective). Time shifts are implemented by zero-padding fixed windows (physical features are recomputed from the shifted window, content shifts out at window edges): the shift direction is asymmetric, with +128 causing a 84% drop, indicating that the model is sensitive to window temporal alignment—this is the most significant robustness weakness. To address this, we trained an auxiliary variant with "random zero-padding time shift augmentation" on the development set (same protocol, random shift within ±128 per batch, development set only, not touching the blind test): unperturbed development set OOF phase PR-AUC 0.581 (per-fold mean 0.632 ± 0.066, measurement 0.652), nearly identical to the historical mainline, while the time shift drop narrows from 34–84% to within ±3% (table below).

| Time Shift | Mainline Phase Drop | Augmented Variant Phase Drop |
|-----------|-------------------|----------------------------|
| −128 | −75.3% | −1.0% |
| −64 | −33.8% | −2.6% |
| +64 | −73.9% | −2.6% |
| +128 | −84.1% | −0.3% |

The phase and measurement-level performance under each perturbation is shown in Fig. 7.

![Fig. 7 Noise and perturbation robustness (historical locked mainline)](figures/fig7_robustness.png)

### 4.8 Window Strategy Ablation (Coverage-Aware Verification)

Under the same development set 5-fold and reference architecture mechanism verification (203k), window strategies are compared (per-fold mean PR-AUC ± standard deviation). Historical lightweight mainline K=8/K=12 (seed 42) are separate evidence with different model and feature configurations and are not directly compared with the 203k reference architecture rows:

| Strategy | K | Composition | Phase PR-AUC | Meas PR-AUC |
|----------|---|------------|-------------|------------|
| single | 1 | 1 equidistant | 0.255 ± 0.029 | 0.273 |
| equidistant | 8 | 8 equidistant + 0 event | 0.526 ± 0.105 | 0.534 |
| event | 8 | 0 equidistant + 8 event | 0.591 ± 0.056 | 0.624 |
| mixed_k4 | 4 | 2 equidistant + 2 event | 0.461 ± 0.070 | 0.523 |
| mixed_k8 (203k reference) | 8 | 4 equidistant + 4 event | 0.590 ± 0.081 | 0.621 |
| mixed_k12 (203k reference) | 8 | 6 equidistant + 6 event | 0.644 ± 0.092 | 0.668 |
| mixed_k8 (historical lightweight, seed 42) | 8 | 4 equidistant + 4 event | 0.639 ± 0.055 | 0.657 |
| mixed_k12 (historical lightweight, seed 42) | 12 | 6 equidistant + 6 event | 0.670 ± 0.070 | 0.679 |

Key points: (1) Single-window coverage is severely insufficient (0.255); (2) Event windows carry the primary discriminative information—pure event (0.591) and mixed K=8 (0.590) are nearly identical (difference 0.001, within noise level, and pure event has smaller fold-wise standard deviation), so we cannot claim that "mixed outperforms pure event"; (3) K increases monotonically (203k reference architecture: 0.461 → 0.590 → 0.644; historical lightweight mainline K=8→K=12: 0.639 → 0.670, seed 42), with consistent direction of gain. K=12 is optimal but increases window count and extraction/training cost by approximately 1.5×. Mixed K=8 is justified as the mainline by coverage robustness: performance is comparable to pure event while equidistant anchors ensure global coverage with fallback protection against event detection failure. To maintain the protocol discipline of "blind test set evaluated only once," the upgraded context-concat mainline retains the same K=8 mixed strategy, and K=12 serves only as development set evidence, not part of the main report.

Window strategy and K-value ablation results are shown in Fig. 8.

![Fig. 8 Window strategy ablation (K and composition)](figures/fig8_window_policy.png)

### 4.9 External Cross-Domain Transferability Analysis (Frozen External Datasets, Historical Lightweight Mainline)

To test the cross-device generalization of the historical locked lightweight mainline (pure CNN + Attention MIL + Mean interaction, 80,113 parameters), we evaluate a three-arm protocol on fully frozen external data (complementary to the VSB frozen blind test, which is not reopened for external experiments): zero-shot (VSB full development set checkpoint directly inferred), from-scratch (trained from scratch on the external training set), and fine-tune (initialized from VSB checkpoint, fine-tuned on the external training set). The external adapter reuses the same WindowEncoder(cnn) + Attention MIL + PhaseClassifier architecture; single-channel external signals are robustly normalized; for dataset 24033225, the 400-point signals are tiled to 8,192; for 28523090, each recording is divided into 8 uniform 8,192-point windows.

| External Dataset | Task | Test Set | Zero-shot ROC / PR | From-scratch ROC / PR | Fine-tune ROC / PR |
|-----------------|------|---------|-------------------|---------------------|-------------------|
| figshare 24033225 (Motor PD/Noise) | PD vs Noise | Te0 (45,970 samples, 50/50) | 0.665 / 0.690 | 0.990 / 0.993 | 0.989 / 0.992 |
| figshare 28523090 (Oscilloscope 8-class) | PD vs background | C2 cross-device (639 samples) | 0.432 / 0.633 | 0.811 / 0.907 | 0.796 / 0.903 |
| figshare 28523090 | PD vs corona | C2 cross-device (319 samples) | 0.547 / 0.701 | 0.959 / 0.973 | 0.905 / 0.952 |

Key points: (1) For similar acquisition domains (motor PD/Noise), zero-shot is clearly above random (ROC 0.665 / PR 0.690), and a small amount of target-domain training quickly approaches saturation (ROC ≥ 0.989). (2) On the cross-device C1→C2 strong generalization task, zero-shot fails (ROC 0.432 / 0.547, F1=0), indicating that the VSB historical lightweight mainline's discriminative patterns do not directly transfer to different hardware acquisition systems. However, with a small amount of target-domain labeled data, from-scratch/fine-tune immediately recover strong performance (ROC 0.81–0.96). (3) This negative result is honestly reported: the lightweight model does not claim universal zero-shot transferability but demonstrates that "information selection + lightweight structure" can be efficiently adapted under the frozen external protocol.

### 4.10 Controlled Sampling Experiment: The Role of Information Selection

To decouple "information selection" from "model structure," this section fixes the architecture (TimWindowEncoder(cnn) → Attention MIL → context-concat → phase classifier, phase-level BCE, AdamW, StratifiedGroupKFold(5, seed=42), batch=64, epochs=40, patience=15) and varies only the sampling strategy: uniform K=8, pure event K=8, random K=8, mixed K=8 (mainline), and full-signal baseline (K=1, L=800,000, batch=8). All five strategies share the same measurement order and fold fingerprints, so per-fold metrics and paired cluster bootstrap are directly comparable. The full-signal row uses batch=8 due to GPU memory constraints for the 800,000-point single-sample input; K=8 rows use batch=64.

| Strategy | K | Data Usage | Phase PR-AUC | Meas PR-AUC | Phase ROC | Meas ROC | Phase MCC | Meas MCC | MACs/Meas | GPU b1 (ms) | CPU b1 (ms) | Peak GPU Mem (MB) |
|----------|---|-----------|-------------|------------|---------|---------|----------|---------|----------|------------|------------|-----------------|
| uniform_k8 | 8 | 8.2% | 0.513 ± 0.130 | 0.537 ± 0.132 | 0.920 | 0.916 | 0.522 | 0.532 | 62.8M | 6.49 | 21.17 | 92.4 |
| event_k8 | 8 | 8.2% | 0.609 ± 0.081 | 0.638 ± 0.061 | 0.940 | 0.937 | 0.551 | 0.583 | 62.8M | 5.59 | 19.40 | 92.4 |
| random_k8 | 8 | 8.2% | 0.579 ± 0.062 | 0.601 ± 0.036 | 0.928 | 0.923 | 0.500 | 0.512 | 62.8M | 5.39 | 19.42 | 92.4 |
| mixed_k8 (mainline) | 8 | 8.2% | 0.615 ± 0.053 | 0.643 ± 0.053 | 0.937 | 0.935 | 0.534 | 0.541 | 62.8M | 5.35 | 20.97 | 92.4 |
| full_signal | 1 | 100% | 0.345 ± 0.174 | 0.387 ± 0.165 | 0.823 | 0.835 | 0.178 | 0.217 | 755.3M | 25.35 | 379.96 | 184.0 |

Paired cluster bootstrap (2,000 resamples, seed 42, difference = mixed_k8 − each strategy):

| Contrast | Level | PR-AUC Difference (95% CI) |
|----------|-------|---------------------------|
| mixed vs uniform | Phase | +0.113 [0.055, 0.175] |
| mixed vs uniform | Meas | +0.131 [0.073, 0.194] |
| mixed vs event | Phase | +0.062 [0.002, 0.122] |
| mixed vs event | Meas | +0.053 [−0.011, 0.118] |
| mixed vs random | Phase | +0.044 [−0.023, 0.112] |
| mixed vs random | Meas | +0.077 [0.010, 0.141] |
| mixed vs full | Phase | +0.306 [0.240, 0.369] |
| mixed vs full | Meas | +0.333 [0.271, 0.395] |

Key points: (1) Uniform K=8 is the weakest among the four K=8 strategies (0.513/0.537), showing that "coverage without information focus" is insufficient for strong discrimination. (2) Pure event and mixed K=8 are nearly comparable (0.609 vs 0.615, difference approximately +0.006), consistent with the mechanism evidence in Section 4.8: discharge transients are concentrated in event-high regions, and equidistant anchors primarily provide coverage fallback. (3) Random K=8 as a control (0.579 ± 0.062/0.601 ± 0.036) shows that the phase-level difference is not significant (95% CI contains 0), while the measurement-level difference is significant (95% CI excludes 0), indicating that random coverage can also capture some event information, and the relative gain from event focus is more apparent at the measurement-level fusion. (4) The full-signal baseline, under the same architecture, uses approximately 12× MACs (755.3M vs 62.8M), approximately 4.7× GPU batch-1 latency, and higher memory, yet achieves phase PR-AUC of only 0.345 (measurement 0.387), significantly lower than mixed K=8. (5) Therefore, information selection itself is the critical design variable for diagnostic efficiency, not the CNN structure.

This comparison is shown in Fig. 9: under the same architecture, the K=8 strategies differ only in sampling strategy, with identical model cost. The full-signal row trades approximately 12× MACs and approximately 4.7× GPU batch-1 latency for limited information gain (not commensurate with cost). It should be noted that the full-signal row uses the same lightweight architecture as the K=8 rows to isolate the sampling strategy variable; stronger full-signal models specifically designed for 800,000-point sequences (e.g., downsampling, segment aggregation, or stronger encoders) are outside the scope of this paper and may improve absolute performance, but at the cost of full data processing.

![Fig. 9 Controlled sampling experiment: strategy performance and cost comparison](figures/fig9_sampling_policy.png)

### 4.11 End-to-End Measurement-to-Decision Pipeline Cost Analysis

To address the overstatement of "real-time detection / end-to-end 1.94 ms," this section reports the per-component cost of the complete measurement-to-decision pipeline using the Stage 2 benchmark (results/stage2_sampling/benchmark.json). Platform: RTX 4060 Laptop GPU with the same host CPU, synthetic 800k three-phase signal, 50 repetitions, p50. Window selection runs on CPU; model forward pass reports CPU/GPU batch=1 and GPU throughput separately.

| Stage | Platform / Batch | p50 |
|-------|-----------------|-----|
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

Mixed K=8 per-component breakdown (GPU / CPU, p50): robust normalization 2.85 / 7.98 ms, encoder 3.77 / 18.38 ms, MIL 0.49 / 0.19 ms, three-phase interaction 0.39 / 0.25 ms, output head 0.48 / 0.12 ms, total 5.35 / 20.97 ms. The full-signal row's corresponding breakdown: robust normalization 15.32 / 203.20 ms, encoder 25.00 / 383.60 ms, total 25.35 / 379.96 ms. Therefore:

1. The sampling stage (event score + peak detection + window selection, approximately 184 ms/phase for mixed strategy) is the dominant CPU cost in the complete pipeline, while the K=8 model forward pass is only approximately 5.35 ms (GPU batch=1). Uniform/random/full-signal rows have almost no selection cost, but their diagnostic performance and data processing volume differ;
2. The full-signal model increases per-measurement MACs from 62.8M to 755.3M (approximately 12×), GPU batch=1 latency from 5.35 ms to 25.35 ms (approximately 4.7×), CPU batch=1 from 20.97 ms to 379.96 ms (approximately 18×), and peak GPU memory from 92.4 MB to 184.0 MB;
3. This paper therefore limits its efficiency claims to "reducing neural inference complexity and data processing requirements by processing only informative segments," and does not claim real-time detection or equate model forward latency to end-to-end latency.

---

### 4.12 Independent Test: Harvard Dataverse Held-Out Blind Test

#### 4.12.1 Data Source and Protocol

The independent test data originates from another held-out partition of the VSB fault detection project on Harvard Dataverse (doi:10.7910/DVN/JYJJ5W, `part1_1_A.tar.gz`, 23 GB), containing 84,625 phase signals (800,000 signed bytes, 40 MHz sampling rate), with no overlap with the Kaggle development set (2,904 measurements) or the 423 blind-test set. This partition contains 28,285 measurements, corresponding to 84,625 phase signals (all `_1`, `_2`, `_3` three-phase signals), of which 1,392 phase signals belong to the training subset labeled in `train_set.tab`, and 83,233 phase signals constitute the independent held-out blind test set for this paper.

Protocol freezing steps:
1. **Pre-download lock**: Compute SHA-256 hashes of the archive file, member list, feature_vector.csv, and train_set.tab, written to `blind_lock_E4_harvard.json`, recording the blind test set size (83,233 phase signals, 1,308 positives);
2. **Freeze model**: Reinitialize the pure CNN + Attention MIL + context-concat architecture (113,265 parameters, `enc_cnn__mil_attention__ph_ctx_concat`) using the seed 42 5-fold development set training, saving full state dictionaries for each fold, maintaining complete consistency with development set training;
3. **One-time evaluation**: Load the 5-fold models, compute average probabilities for the 83,233 phase signals (mixed K=8 windows, 24 worker threads), write to `FINAL_EVALUATION_RECEIPT_E4_harvard.json` with a run lock, rejecting any second run.

#### 4.12.2 Results

| Metric | Phase-level (label > 0) | Phase-level (contact classes 1/2/5/6) | Measurement-level (full triple) |
|--------|-----------------------|--------------------------------------|-------------------------------|
| PR-AUC | 0.216 | 0.181 | 0.495 |
| ROC-AUC | 0.899 | 0.894 | 0.957 |
| MCC | 0.224 | 0.195 | 0.000 |
| F1 | 0.192 | 0.170 | 0.000 |
| Accuracy | 0.984 | 0.984 | 0.967 |
| ECE | 0.007 | 0.008 | 0.023 |
| Brier | 0.014 | 0.014 | 0.026 |

Bootstrapped 95% CI (measurement-clustered, 2,000 resamples): phase-level PR-AUC [0.199, 0.234], measurement-level [0.196, 0.833] (only 4 positive complete triples, very wide CI).

#### 4.12.3 Analysis and Significance

Key findings from the independent test:

1. **ROC-AUC 0.899 demonstrates effective discrimination**: Across 83,233 phase signals, E4 achieves ROC-AUC 0.899, indicating that the model maintains good ranking ability for discharge/non-discharge phases in the held-out data, without severe overfitting to the development set distribution.

2. **PR-AUC is substantially lower than the development set (0.216 vs 0.615)**: The primary reason is the dramatic difference in positive rate—the Harvard held-out set has only 1.57% (1,308/83,233), while the VSB development set is approximately 5.9%. PR-AUC is sensitive to positive rate, and under extreme imbalance, the upper bound is structurally constrained; ROC-AUC is insensitive to class distribution, and the level of 0.899 indicates that ranking ability remains good.

3. **Large measurement-level statistical uncertainty**: With only 120 complete three-phase measurements (of which 4 are positive), the measurement-level PR-AUC 95% CI is [0.196, 0.833], with a median of 0.580, spanning nearly the entire possible range, precluding reliable point estimate comparison.

4. **Threshold dependency issue**: The optimal measurement threshold of 0.88 selected from the development set is too strict for the Harvard held-out set, causing all measurement-level scores to fall below this threshold, yielding MCC/F1 of zero. This reflects cross-block calibration shift, not model failure—ROC-AUC 0.957 indicates that adjusting the threshold can still separate positive and negative instances.

5. **Good calibration**: Phase-level ECE is only 0.007 (Brier 0.014), consistent with the development set level (0.007–0.008), indicating that probability outputs do not exhibit severe miscalibration on the held-out set.

#### 4.12.4 Independent Test Loop Closure

The independent test loop is closed: the E4 final frozen model (seed 42, 5-fold average) completed a one-time evaluation on the Harvard Dataverse held-out set, generating a verifiable receipt (receipt SHA-256 bound to protocol lock and model checkpoint hashes). This held-out set never participated in any training, model selection, hyperparameter, or threshold determination. The development set 5-fold CV, three-seed replication, and OOF cluster bootstrap uncertainty estimates are now anchored by an independent test: the held-out set ROC-AUC of 0.899 confirms that E4 maintains effective discriminative ability on a non-overlapping block from the same data source.

---

## 5 Discussion

### 5.1 Impact of Metric Protocol

There is a systematic difference between per-fold mean PR-AUC and pooled OOF PR-AUC: for the E4 mainline (Section 4.4), per-fold mean is 0.615 vs pooled OOF 0.533 (difference 0.082), and measurement-level is 0.643 vs 0.585 (difference 0.058). For the strongest traditional machine learning baseline (flattened + LightGBM, same K=8 feature cache), per-fold mean is 0.206 vs pooled OOF 0.198. Under both protocols, the model substantially outperforms the traditional ML baseline: per-fold mean absolute improvement +0.409, pooled OOF improvement +0.335, so the "performance advantage over the baseline" holds under both protocols. However, absolute performance is protocol-sensitive (per-fold 0.615 vs pooled 0.533), indicating that this difference arises from fold-wise probability scale/calibration shifts rather than metric function confusion or label alignment errors: all folds share the same fold fingerprint and average_precision_score computation, and OOF probabilities are strictly aligned with measurement_id (historical audit in results/fold_oof_gap_audit_80k.json, Stage 1 reuses the same folds and feature cache). The historical locked mainline's blind test uses 5-fold probability average ensemble, which naturally mitigates this fold-wise scale difference. Future work could introduce per-fold temperature scaling calibration before reporting pooled metrics, but the blind test set must not be reopened based on this. This paper uniformly adopts per-fold mean as the primary protocol, and all sections 4.4–4.6 and ablations use the same protocol; pooled OOF values are provided only as robustness supplements in this section. The historical blind test single frozen evaluation yields phase-level shared-score PR-AUC of 0.524 (95% CI [0.379, 0.657]) and measurement-level 0.582 ([0.447, 0.714]); these numbers belong to the historical locked mainline and are not attributed to the upgraded E4, nor directly compared as point estimates with E4's per-fold mean. Due to the limited number of blind test positive samples (82 phases / 31 measurements), significant degradation cannot be asserted from a single evaluation; the blind test set is not reopened.

### 5.2 Measurement-Cluster Statistical Uncertainty Analysis

To avoid confusion with metrological measurement uncertainty, this analysis is termed measurement-cluster statistical uncertainty analysis. To quantify the impact of small sample size and inter-measurement variability on conclusions, we perform a measurement-clustered bootstrap (2,000 resamples, stratified by measurement-level label) on the E4 mainline's development set OOF predictions (results/stage1_tim/e4_ctx_concat/seeds/seed_42/oof.npz): E4 phase-level PR-AUC median 0.5333 (95% CI [0.4734, 0.5908]), measurement-level median 0.5852 ([0.5267, 0.6414]). The difference from the per-fold mean protocol (0.615/0.643) is attributable to the fold-wise probability scale phenomenon described in Section 5.1. E1–E6 and matched encoder differences are evaluated using paired bootstrap on the same measurement clusters (difference = former − latter, negative values indicate latter is higher):

| Contrast | Phase PR-AUC Difference (95% CI) | Measurement-Level Difference (95% CI) |
|----------|--------------------------------|-------------------------------------|
| E4 vs E2 (Mean shared) | +0.055 [+0.006, +0.103] | +0.071 [+0.023, +0.118] |
| E4 vs E5 (context-add) | +0.073 [+0.028, +0.117] | +0.077 [+0.025, +0.127] |
| E4 vs E3 (Max aggregation) | +0.036 [−0.012, +0.087] | +0.045 [−0.002, +0.096] |
| E6 vs E4 (hierarchical loss) | −0.030 [−0.084, +0.024] | −0.050 [−0.104, +0.003] |
| ResNet1D vs cnn (E4) | +0.095 [+0.033, +0.152] | +0.064 [+0.007, +0.119] |
| InceptionTime vs cnn (E4) | +0.044 [−0.005, +0.092] | +0.051 [−0.003, +0.102] |
| simple_cnn vs cnn (E4) | −0.005 [−0.048, +0.041] | −0.016 [−0.060, +0.031] |

Overall conclusions: E4 significantly outperforms E2 and E5 at both phase and measurement levels (95% CI excludes 0). The difference from E3 is not significant, and the hierarchical measurement loss (E6) is negative and not significant—both negative results are honestly reported. Among matched encoders, only ResNet1D is significantly higher at both phase and measurement levels, but at the cost of approximately 263× MACs, approximately 3.6× GPU batch-1 latency, and approximately 26× CPU batch-1 latency. InceptionTime and simple_cnn are not significant, and InceptionTime's MACs are approximately 887×. This paper accordingly positions lightweight as an accuracy-efficiency trade-off, without overclaiming "no degradation."

Published methods (Zheng TF-CNN 2022, Fei CNN+QSVM 2024) use an earlier historical protocol (Mean MIL + Max phase interaction) with different implementation details and are not paired bootstrap comparisons with E4; they appear only as literature reference points in Section 4.5. Cross-protocol comparisons must also note the metric protocol. Calibration and threshold analysis follow the historical locked mainline (threshold selected only on development set OOF, blind test not involved): historical mainline phase ECE 0.036, Brier 0.042, measurement-level ECE 0.037, Brier 0.044; historical blind test ECE 0.040, Brier 0.045. Measurement-level threshold at 0.5 / max-MCC / recall≥0.5 / recall≥0.8 yields F1 of 0.490 / 0.531 / 0.508 / 0.402 (corresponding precision 0.732 / 0.731 / 0.513 / 0.268). Latency distribution is presented in Section 4.11. Thresholds depend on development set OOF, consistent with the "blind test evaluated only once" discipline.

### 5.3 Module Contribution Analysis

- **Event windows as information anchors for weakly supervised learning**: In the reference architecture mechanism verification, pure event ≈ mixed strategy, indicating that discharge transients are concentrated in event-high regions, and equidistant windows primarily provide background coverage. Both E4 and the historical lightweight mainline use the same K=8 mixed strategy;
- **Coverage density is monotonically effective**: In the 203k reference architecture mechanism verification, K increases from 4 to 12 yield consistent improvement (0.461→0.590→0.644), and the historical lightweight mainline K=8→K=12 shows 0.639→0.670 (seed 42), supporting "coverage-aware sampling" as an independent contribution;
- **Three-phase interaction necessity**: In the E1–E6 ablation, no interaction (E1) is the weakest across all three seeds (seed 42 phase 0.483 ± 0.060), approximately 0.13 below E4 (0.615 ± 0.053). The historical mechanism verification also shows a decrease of approximately 0.12 without interaction, and missing a single phase causes almost no degradation (historical mainline), indicating that three-phase signals contain exploitable redundant structure;
- **Phase interaction form**: Paired bootstrap shows that context-concat (E4) significantly outperforms Mean shared-score (E2) and context-add (E5), while the difference from Max aggregation (E3) is not significant. The hierarchical measurement loss (E6) does not provide a significant gain—both negative results are honestly reported;
- **Controlled sampling evidence (Section 4.10)**: Under fixed architecture, uniform K=8 is the weakest (0.513/0.537). Pure event and mixed K=8 are comparable (0.609 vs 0.615). Significance conclusions from paired bootstrap are reported only where CI excludes 0. The full-signal baseline, at approximately 12× MACs and approximately 4.7× GPU batch-1 latency, does not achieve a commensurate advantage, establishing information selection as a critical design variable;
- **Lightweight positioning as a trade-off**: The final lightweight mainline (pure CNN + Attention MIL + context-concat) has 113,265 parameters, a three-seed mean of 0.611 ± 0.006 (measurement 0.639 ± 0.008), 62.8M MACs, and GPU batch-1 forward approximately 5.35 ms (pre-extracted windows, Stage 2 same-session retest), with fewer parameters than the reference architecture mechanism verification model (203,634). Compared to larger deep baselines, only ResNet1D is significantly higher but at approximately 263× MACs. E4 represents an accuracy-efficiency trade-off achieving competitive performance at lower parameter count, rather than claiming higher precision.

### 5.4 Limitations and Future Work

1. The independent test loop is closed (see Section 4.12): E4 completed a one-time blind test on the Harvard Dataverse held-out set (83,233 phase signals), with phase-level PR-AUC 0.216 (95% CI [0.199, 0.234]), ROC-AUC 0.899. However, the held-out set data distribution differs from the VSB development set (positive rate 1.57% vs 5.9%). The lower PR-AUC on the held-out set is expected and cannot be simply interpreted as model degradation. Future work with additional held-out data could further reduce statistical uncertainty;
2. Robustness evidence still comes from the historical locked mainline: at 5 dB noise, phase PR-AUC drops by 67%; fixed window zero-padding shifts of ±64/±128 cause drops of 34–84% (directionally asymmetric, +128 worst). A random time-shift augmentation auxiliary variant can narrow the time-shift drop from 34–84% to within ±3%, with development set per-fold mean nearly unchanged (0.632 ± 0.066 vs historical mainline 0.639 ± 0.055; see Section 4.7 table). The augmented variant does not enter the blind test, and E4 perturbation robustness was not retested;
3. External validation shows that cross-device zero-shot transferability remains limited (historical lightweight mainline): zero-shot is effective on motor PD/Noise (ROC 0.665 / PR 0.690) but fails on the 28523090 C1→C2 cross-device task (ROC 0.432–0.547, F1=0), requiring target-domain labeled data for adaptation (from-scratch/fine-tune restores to ROC 0.81–0.96);
4. The significance evidence for matched encoders and E1–E6 currently relies primarily on paired bootstrap at seed 42. E4 has completed 3 seeds × 5 folds (seeds {42, 7, 2024}) for stability verification, but seed-level statistical testing and significance evidence for K=12 still require additional seed replication;
5. The full-signal baseline maintains the same lightweight architecture as a control variable and does not explore full-signal models specifically designed for 800,000-point sequences (e.g., downsampling, segment aggregation, or stronger encoders). This paper only demonstrates that directly processing the full signal with the same architecture does not yield gains commensurate with approximately 12× MACs. A stronger full-signal model may improve absolute performance but at the cost of full data and larger model capacity;
6. Self-supervised pretraining (VICReg) and larger attention architectures could be explored as optional extensions, but would require verification of their gains against the baseline model established in this paper;
7. Threshold selection currently relies on development set OOF max-MCC/fixed threshold criteria. Practical deployment would need to incorporate field-specific false alarm costs.

---

## 6 Conclusion

This paper investigates the role of coverage-aware information selection and hierarchical weakly supervised structure in extremely long, sparse, weakly labeled three-phase industrial signals, using the VSB real-world PD dataset. Experiments demonstrate that judicious information selection (event windows and coverage density) can substantially reduce the dependency on model capacity while maintaining competitive detection performance at lower complexity. At the same time, larger or time-frequency models can still improve the performance ceiling, so this paper is positioned as an accuracy-complexity trade-off rather than claiming that the lightweight mainline replaces stronger models. The final lightweight mainline is a pure CNN encoder + Attention MIL + context-concat three-phase interaction (113,265 parameters). The development set 5-fold main report (seed 42) achieves per-phase PR-AUC 0.615 ± 0.053 (measurement 0.643 ± 0.053), with an absolute improvement of +0.409 over the strongest traditional machine learning baseline (flattened + LightGBM, per-fold mean 0.206) (pooled-OOF protocol +0.335). The three-seed mean is per-phase 0.611 ± 0.006, measurement 0.639 ± 0.008. Paired cluster bootstrap shows that context-concat significantly outperforms Mean shared-score and context-add, while the difference from Max aggregation is not significant, and the hierarchical measurement loss is a negative result. Among matched encoders, only ResNet1D is significantly higher (per-phase +0.095, 95% CI [+0.033, +0.152]), but at approximately 263× MACs; this paper therefore does not claim the highest precision. The reference architecture mechanism verification (203,634 parameters) serves only as structural design evidence and is not part of the main report. Controlled sampling experiments (uniform/event/random/mixed K=8 and full-signal baseline) demonstrate that information selection itself is a critical design variable: random K=8 achieves 0.579/0.601, the full-signal baseline achieves 0.345/0.387, and mixed K=8 maintains competitiveness at approximately 12× lower MACs and lower latency. E4's independent blind test has been completed on the Harvard Dataverse independent held-out set (83,233 phase signals, Section 4.12): phase-level PR-AUC 0.216 (95% CI [0.199, 0.234]), ROC-AUC 0.899, measurement-level PR-AUC 0.495 ([0.196, 0.833]), ROC-AUC 0.957, and the independent test loop is closed. This work provides a reproducible, leakage-safe evaluation pipeline for long-sequence, weakly labeled, highly imbalanced industrial signal modeling, and suggests that judicious information selection can substantially reduce the dependency on model capacity, while increased model complexity can still yield a higher performance ceiling.

---

## References

1. Dietterich, T. G., Lathrop, R. H., Lozano-Pérez, T. Solving the multiple instance problem with axis-parallel rectangles[J]. Artificial Intelligence, 1997, 89(1-2): 31-71.
2. Maron, O., Lozano-Pérez, T. A framework for multiple-instance learning[C]. NIPS, 1998: 570-576.
3. Ilse, M., Tomczak, J. M., Welling, M. Attention-based deep multiple instance learning[C]. ICML, 2018: 2127-2136.
4. Stone, G. C. Partial discharge diagnostics and electrical equipment insulation condition assessment[J]. IEEE Transactions on Dielectrics and Electrical Insulation, 2005, 12(5): 891-904. doi:10.1109/tdei.2005.1522184
5. Raymond, W. J. K., Illias, H. A., Abu Bakar, A. H., Mokhlis, H. Partial discharge classifications: Review of recent progress[J]. Measurement, 2015, 68: 164-181. doi:10.1016/j.measurement.2015.02.032
6. Akram, S., Wang, P., Zhu, X., Huang, J., Liu, F., Fang, Z., Ahmed, H. Prediction of partial discharge inception voltage for electric vehicle motor insulation using deep learning[J]. IEEE Transactions on Instrumentation and Measurement, 2023, 72: 1-10. doi:10.1109/tim.2023.3269120
7. Ishaq, A., Junaid, M., Hussain, G. A., Khan, S. U., Chen, Y., Yu, D. Partial discharge defect classification in MV switchgear by using CWT and deep learning approach[J]. IEEE Transactions on Instrumentation and Measurement, 2025, 74: 1-12. doi:10.1109/tim.2025.3562981
8. Wang, Y., Yan, J., Yang, Z., Wu, Y., Wang, J., Geng, Y. Generative zero-shot learning for partial discharge diagnosis in gas-insulated switchgear[J]. IEEE Transactions on Instrumentation and Measurement, 2023, 72: 1-11. doi:10.1109/tim.2023.3264022
9. Nguyen, M.-T., Nguyen, V.-H., Yun, S.-J., Kim, Y.-H. Recurrent neural network for partial discharge diagnosis in gas-insulated switchgear[J]. Energies, 2018, 11(5): 1202. doi:10.3390/en11051202
10. Zheng, J., Chen, Z., Wang, Q., Qiang, H., Xu, W. GIS partial discharge pattern recognition based on time-frequency features and improved convolutional neural network[J]. Energies, 2022, 15(19): 7372. doi:10.3390/en15197372
11. Fei, Z., Li, Y., Yang, S. Partial discharge pattern recognition based on an ensembled simple convolutional neural network and a quadratic support vector machine[J]. Energies, 2024, 17(11): 2443. doi:10.3390/en17112443
12. Misak, S., Hamacek, S., Bilik, P., Hofinek, M., Petvaldsky, P. Problems associated with covered conductor fault detection[C]. 11th International Conference on Electrical Power Quality and Utilisation, 2011: 1-5. doi:10.1109/epqu.2011.6128806
13. Hashmi, G. M., Lehtonen, M. On-line PD detection for condition monitoring of covered-conductor overhead distribution networks - A literature survey[C]. 2008 Second International Conference on Electrical Engineering, 2008: 1-6. doi:10.1109/icee.2008.4553933
14. Davis, J., Goadrich, M. The relationship between precision-recall and ROC curves[C]. ICML, 2006: 233-240. doi:10.1145/1143844.1143874
15. Kaufman, S., Rosset, S., Perlich, C., Stitelman, O. Leakage in data mining: Formulation, detection, and avoidance[J]. ACM Transactions on Knowledge Discovery from Data, 2012, 6(4): Article 15. doi:10.1145/2382577.2382579
16. Kapoor, S., Narayanan, A. Leakage and the reproducibility crisis in machine-learning-based science[J]. Patterns, 2023, 4(9): 100804. doi:10.1016/j.patter.2023.100804
17. Han, K., Koay, A. M. Y. H. HITS: Hierarchical interpretable time series classification via multiple instance learning[C]. 2025 International Joint Conference on Neural Networks (IJCNN), 2025: 1-8. doi:10.1109/ijcnn64981.2025.11227767
18. Mahmud, S. M. High precision phase measurement using adaptive sampling[J]. IEEE Transactions on Instrumentation and Measurement, 1989, 38(5): 954-960. doi:10.1109/19.39036
19. Jaskulke, R., Himmel, B. Event-controlled sampling system for marine research[J]. IEEE Transactions on Instrumentation and Measurement, 2005, 54(4): 1175-1179. doi:10.1109/tim.2005.847142
20. VSB Power Line Fault Detection[EB/OL]. Kaggle Competition, 2019. https://www.kaggle.com/c/vsb-power-line-fault-detection
21. Ronneberger, O., Fischer, P., Brox, T. U-net: Convolutional networks for biomedical image segmentation[C]. MICCAI, 2015: 234-241. doi:10.1007/978-3-319-24574-4_28
22. He, K., Zhang, X., Ren, S., Sun, J. Deep residual learning for image recognition[C]. CVPR, 2016: 770-778. doi:10.1109/cvpr.2016.90
23. Ismail Fawaz, H., Forestier, G., Weber, J., Idoumghar, L., Muller, P.-A. Deep learning for time series classification: a review[J]. Data Mining and Knowledge Discovery, 2019, 33(4): 917-963. doi:10.1007/s10618-019-00619-1
24. Wang, Z., Yan, W., Oates, T. Time series classification from scratch with deep neural networks: A strong baseline[C]. IJCNN, 2017: 1578-1585. doi:10.1109/ijcnn.2017.7966039
25. Bai, S., Kolter, J. Z., Koltun, V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling[J]. arXiv preprint arXiv:1803.01271, 2018.
26. Paszke, A., Gross, S., Massa, F., et al. PyTorch: An imperative style, high-performance deep learning library[C]. NeurIPS, 2019: 8024-8035.
27. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. Scikit-learn: Machine learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.
28. McKinney, W. Data structures for statistical computing in Python[C]. SciPy, 2010: 56-61. doi:10.25080/majora-92bf1922-00a
29. Hunter, J. D. Matplotlib: A 2D graphics environment[J]. Computing in Science & Engineering, 2007, 9(3): 90-95. doi:10.1109/mcse.2007.55
30. Harris, C. R., Millman, K. J., van der Walt, S. J., et al. Array programming with NumPy[J]. Nature, 2020, 585: 357-362. doi:10.1038/s41586-020-2649-2