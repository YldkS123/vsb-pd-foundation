# VSB-PD: Coverage-Aware Sampling and Hierarchical Weakly-Supervised PD Detection

Official implementation of the paper:

> **Measurement-Cost-Aware Sampling and Hierarchical Weakly Supervised Detection
> for Three-Phase Partial Discharge Monitoring** (IEEE TIM submission)

This repository provides a leakage-safe, reproducible benchmark pipeline for
partial-discharge (PD) detection on the VSB Power Line Fault Detection dataset
(Kaggle, 2019): 2,904 three-phase measurements, 800,000 samples per phase at
40 MHz, phase-level labels only, ~5.9% positive phases.

## Key results (development set, 5-fold CV, seed 42)

| Model | Phase PR-AUC | Meas PR-AUC | Params (total / executed) | MACs/meas | GPU b1 (ms) |
|---|---|---|---|---|---|
| **E4 mainline** (proposed) | **0.615 ± 0.053** | **0.643 ± 0.053** | 113,265 / 97,201 | 62.8M | 5.35 |
| Strongest traditional ML baseline (flattened + LightGBM) | 0.206 | — | — | — | — |
| Classical energy detector | 0.055 | — | 0 (no learning) | — | — |

Independent one-time blind test on a held-out partition (83,233 phase signals):
ROC-AUC 0.899, prevalence-normalized PR lift 13.8× (development: 10.3×).

## Repository layout

```
configs/           pipeline & training configuration (+ local.example.json)
src/vsb_pd/        core implementation (features, events, windows, encoder,
                   mil, cyclic, model, training, evaluation, locks, integrity)
scripts/           experiment runners and analysis
  stage1_tim_runner.py      E1–E6 phase-interaction ablations + matched encoders
  stage2_sampling_runner.py controlled sampling experiments
  e1_posrate_sensitivity.py PR-AUC prevalence-sensitivity control (analytic
                            reweighting + downsampling validation + fixed recall)
  b1_classical_detectors.py classical detector baselines
  e2_ir_evaluation.py       internal-reserved strict hold-out evaluation
  posthoc_e3_vs_shared.py   post-hoc paired bootstrap (E3 vs shared-score family)
  measurement_uncertainty.py measurement-clustered bootstrap CIs
  harvard_blind_evaluate.py one-time independent held-out evaluation (receipted)
results/           all outputs; frozen split locks & evaluation receipts
tests/             174 unit tests (all passing)
```

## Environment

- Python 3.10+, PyTorch ≥ 2.1 (CUDA recommended), scikit-learn ≥ 1.3,
  LightGBM ≥ 4.0, NumPy, SciPy, pandas. See `requirements.txt`.

## Data access

- VSB Power Line Fault Detection (Kaggle): https://www.kaggle.com/c/vsb-power-line-fault-detection
  (accept competition terms to download `train.parquet`).
- Independent held-out partition: Harvard Dataverse doi:10.7910/DVN/JYJJ5W
  (`part1_1_A.tar.gz`, ~23 GB) — used only by the one-time blind evaluation
  script; the frozen receipt binds archive/content hashes.

## Reproducibility protocol

1. **Leakage audit**: historical prediction files are scanned; contaminated
   candidates demoted (`src/vsb_pd/integrity.py`).
2. **Split lock**: development (2,481) / blind (423) split and all pipeline
   parameters are committed to a SHA-256 hash-verified lock
   (`artifacts/locks/split_lock.json`).
3. **One-time evaluation**: the 423 blind measurements were evaluated exactly
   once by the historical locked mainline; the independent held-out partition
   was evaluated exactly once by the frozen E4 mainline, with tamper-evident
   receipts (`results/FINAL_EVALUATION_RECEIPT_*.json`).
4. **Prevalence control**: `scripts/e1_posrate_sensitivity.py` performs
   post-hoc analytic PR reweighting on frozen predictions (no retraining,
   no reopening of any one-time evaluation).

## Quick start

```bash
pip install -e ".[dev]"
Copy-Item configs/local.example.json configs/local.json   # edit paths
python scripts/cache_features.py                          # precompute features
python scripts/stage1_tim_runner.py --experiments e4 --encoders simple_cnn --seeds 42
python scripts/e1_posrate_sensitivity.py
python scripts/b1_classical_detectors.py
python -m pytest -q
```

## License & citation

MIT License (see LICENSE). If you use this code or data pipeline in your
research, please cite the paper (bibtex provided in `CITATION.bib`).
