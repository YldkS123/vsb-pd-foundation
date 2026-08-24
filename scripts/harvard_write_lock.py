# -*- coding: utf-8 -*-
"""Write the frozen Harvard blind-test protocol lock for E4.

The lock is created before any blind metric is computed.  It fixes:

- the data archive (Harvard Dataverse 10.7910/DVN/JYJJ5W, chunk part1_1_A)
- the blind set: every measurement-phase signal in the chunk EXCEPT the
  dataset's own official training split (train_set.tab, phase 0/1/2 mapped
  to 1/2/3); duplicate archive member names are de-duplicated
- the positive label mapping: annotation > 0 (any fault) for the primary
  phase-level test; annotation in {1,2,5,6} (contact-related PD) is a frozen
  secondary label view
- the frozen E4 model ensemble, dev thresholds, window policy and metrics

Nothing in this file reads signal waveforms or label values beyond the
pre-computed composition statistics recorded below.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = ROOT / "results" / "blind_lock_E4_harvard.json"
ARCHIVE = Path(r"C:\Users\hrfxgfx\Downloads\part1_1_A.tar.gz")
MEMBER_LIST = Path(r"C:\Users\hrfxgfx\Downloads\part1_1_A_members.txt")
FEATURE_CSV = Path(r"C:\Users\hrfxgfx\Downloads\feature_vector.csv")
TRAIN_SET = Path(r"C:\Users\hrfxgfx\Downloads\train_set.tab")
TAR_SHA256 = "6709b72e9e9a6e8b8124ece9e1bad1172f3d191f477861e8c95ca90ed0590e74"
MEMBER_SHA256 = "73f82994a0e2b7f7b983feb44982b5fa7d13d2a237b4b02647c13b67a80b0a4f"
FEATURE_SHA256 = "b33bfabd6dd89d7df32ee078f1fdd3cc94edc266d8a0cdf247f36740ab1ea4a2"
TRAIN_SHA256 = "360b553c464290e5ccd69cd5ce8ae93651285fa89cf78d4fa119abfe69f5fb58"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_members() -> set[tuple[int, int]]:
    pat = re.compile(r"part1_1_A/(\d+)_([123])\.bin\.gz$")
    keys: set[tuple[int, int]] = set()
    with open(MEMBER_LIST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = pat.match(line)
            if not m:
                raise SystemExit(f"unexpected member name: {line}")
            keys.add((int(m.group(1)), int(m.group(2))))
    return keys


def load_annotations() -> dict[tuple[int, int], int]:
    import csv

    out: dict[tuple[int, int], int] = {}
    with open(FEATURE_CSV, encoding="utf-8", errors="replace") as f:
        r = csv.reader(f, delimiter=";")
        next(r)
        for row in r:
            if len(row) < 3:
                continue
            out[(int(row[0]), int(row[1]))] = int(row[2])
    return out


def load_train_set() -> set[tuple[int, int]]:
    import csv

    q = chr(34)
    out: set[tuple[int, int]] = set()
    with open(TRAIN_SET, encoding="utf-8", errors="replace") as f:
        r = csv.reader(f, delimiter=";", quotechar=q)
        next(r)
        for row in r:
            if len(row) == 1 and ";" in row[0]:
                parts = row[0].split(";")
                mid, ph = parts[0].strip(q), parts[1].strip(q)
            elif len(row) >= 2:
                mid, ph = row[0].strip(q), row[1].strip(q)
            else:
                continue
            out.add((int(mid), int(ph) + 1))  # train phase 0/1/2 -> 1/2/3
    return out


def main() -> int:
    if LOCK_PATH.exists():
        raise SystemExit(f"lock already exists: {LOCK_PATH} (refusing to overwrite)")
    if not ARCHIVE.exists():
        raise SystemExit(f"archive missing: {ARCHIVE}")
    if sha256_file(ARCHIVE) != TAR_SHA256:
        raise SystemExit("archive SHA-256 mismatch")
    if sha256_file(MEMBER_LIST) != MEMBER_SHA256:
        raise SystemExit("member list SHA-256 mismatch")
    if sha256_file(FEATURE_CSV) != FEATURE_SHA256:
        raise SystemExit("feature vector SHA-256 mismatch")
    if sha256_file(TRAIN_SET) != TRAIN_SHA256:
        raise SystemExit("train_set SHA-256 mismatch")

    members = load_members()
    ann = load_annotations()
    train = load_train_set()
    missing = sorted(k for k in members if k not in ann)
    if missing:
        raise SystemExit(f"members missing labels: {missing[:5]} ...")
    blind = members - train
    overlap = members & train
    cnt = Counter(ann[k] for k in blind)
    pos_any = sum(cnt.get(a, 0) for a in (1, 2, 3, 4, 5, 6))
    pos_contact = sum(1 for k in blind if ann[k] in {1, 2, 5, 6})
    by_mid: dict[int, list[int]] = {}
    for mid, ph in blind:
        by_mid.setdefault(mid, []).append(ph)
    triples = {mid: sorted(phs) for mid, phs in by_mid.items() if len(phs) == 3}
    pos_triples = sum(
        1 for mid, phs in triples.items() if any(ann[(mid, ph)] > 0 for ph in phs)
    )

    lock = {
        "purpose": "One-time independent blind test of the final E4 mainline "
                   "(context_concat) on a never-before-used public dataset",
        "created_before_evaluation": True,
        "date": "2026-08-18",
        "data": {
            "harvard_dataverse_doi": "10.7910/DVN/JYJJ5W",
            "dataset_paper_doi": "10.1038/s41597-026-07219-x",
            "signal_format": "800,000 signed bytes per phase signal, "
                             "40 MHz, gzip-compressed",
            "archive": {
                "path": str(ARCHIVE),
                "size_bytes": int(ARCHIVE.stat().st_size),
                "sha256": TAR_SHA256,
                "member_count": len(members),
                "member_list_sha256": MEMBER_SHA256,
            },
            "feature_vector": {
                "path": str(FEATURE_CSV),
                "sha256": FEATURE_SHA256,
                "rows": len(ann),
            },
            "train_set": {
                "path": str(TRAIN_SET),
                "sha256": TRAIN_SHA256,
                "phase_mapping": "train phase 0/1/2 -> feature phase 1/2/3",
                "overlap_with_archive": len(overlap),
            },
        },
        "blind_set": {
            "definition": "all measurement-phase signals in part1_1_A.tar.gz "
                          "minus the dataset's official train_set.tab rows; "
                          "duplicate archive member names de-duplicated; "
                          "no label-based selection",
            "n_phase_signals": len(blind),
            "n_positive_phase_signals_annotation_gt_0": int(pos_any),
            "positive_rate_annotation_gt_0": round(pos_any / len(blind), 6),
            "n_contact_positive_annotation_in_1_2_5_6": int(pos_contact),
            "n_complete_triple_measurements": len(triples),
            "n_positive_complete_triple_measurements": int(pos_triples),
        },
        "model": {
            "name": "E4 final mainline",
            "architecture": "TimWindowEncoder(cnn,8192,58,128) + Attention MIL "
                            "+ context_concat + PhaseClassifier(128)",
            "n_params": 113265,
            "checkpoints": "five development-fold checkpoints regenerated under "
                           "the frozen Stage 1 protocol (seed 42) before this "
                           "blind test; the original E4 run did not persist "
                           "weights; no Harvard data is used for training",
            "ensemble": "mean of the five fold phase probabilities",
            "dev_thresholds": {
                "phase": 0.5,
                "measurement": 0.88,
                "source": "results/stage1_tim/e4_ctx_concat/seeds/seed_42/cv_summary.json",
            },
            "dev_reference": {
                "fold_mean_phase_pr_auc": 0.6149,
                "fold_mean_measurement_pr_auc": 0.6432,
            },
        },
        "evaluation": {
            "window_policy": "mixed_k8: WindowPolicy(8192, uniform=4, event=4, "
                             "dedup_iou=0.5, fallback_grid_size=256)",
            "input_preprocessing": "same as development: raw signed-byte signals, "
                                   "per-window robust normalization inside "
                                   "TimWindowEncoder; features are unused by the "
                                   "CNN branch and passed as zeros",
            "missing_phases": "zero-filled; phase_mask marks present phases; "
                              "masked phases excluded from all metrics",
            "primary": "phase-level metrics on all blind phase signals; "
                       "positive = annotation > 0",
            "secondary_label_view": "phase-level metrics with positive = "
                                    "annotation in {1,2,5,6} (contact-related)",
            "measurement_level": "noisy-OR over the complete-triple "
                                 "measurements only (small subset, reported "
                                 "with caveats)",
            "metrics": ["pr_auc", "roc_auc", "mcc", "f1", "precision", "recall",
                        "accuracy", "ece", "brier"],
            "bootstrap": "measurement-clustered 2000x 95% CI for phase PR-AUC "
                         "and complete-triple measurement PR-AUC",
            "one_time": "run exactly once; no model selection, no threshold "
                        "tuning, no re-run after seeing metrics",
        },
    }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {LOCK_PATH}")
    print(f"blind phase signals={len(blind)}, positives={pos_any}, "
          f"rate={pos_any / len(blind):.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
