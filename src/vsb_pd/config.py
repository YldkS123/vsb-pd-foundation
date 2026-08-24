from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class WindowPolicy:
    window_length: int
    uniform_count: int
    event_count: int
    dedup_iou: float
    fallback_grid_size: int

    @property
    def total_count(self) -> int:
        return self.uniform_count + self.event_count

    def validate(self) -> None:
        if self.window_length != 8192:
            raise ValueError("window_length must be 8192")
        allowed = {
            (1, 0), (4, 0), (8, 0), (12, 0),   # single-peak / equidistant-only
            (0, 4), (0, 8), (0, 12),           # event-only
            (2, 2), (4, 4), (6, 6),            # locked hybrid (K=4/8/12)
        }
        if (self.uniform_count, self.event_count) not in allowed:
            raise ValueError("window composition not in the locked ablation grid")
        if self.dedup_iou != 0.5:
            raise ValueError("dedup_iou must be 0.5")
        if self.fallback_grid_size != 256:
            raise ValueError("fallback_grid_size must be 256")


@dataclass(frozen=True)
class PipelineConfig:
    metadata_path: Path
    raw_parquet_path: Path
    candidate_split_path: Path
    artifact_root: Path
    signal_length: int
    sampling_rate_hz: int
    window_policy: WindowPolicy

    def validate(self) -> None:
        if self.signal_length != 800_000:
            raise ValueError("signal_length must be 800000")
        if self.sampling_rate_hz != 40_000_000:
            raise ValueError("sampling_rate_hz must be 40000000")
        self.window_policy.validate()


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    early_stopping_patience: int = 20
    grad_clip_norm: float = 1.0
    lambda_m: float = 0.25

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.lambda_m not in (0.0, 0.25, 0.5):
            raise ValueError("lambda_m must be 0, 0.25, or 0.5")


@dataclass(frozen=True)
class ModelConfig:
    window_length: int = 8192
    num_windows: int = 8
    feature_dim: int = 58
    hidden_dim: int = 128
    aggregation: Literal["mean", "max", "attention", "gated_attention"] = "gated_attention"

    def validate(self) -> None:
        if self.window_length != 8192:
            raise ValueError("window_length must be 8192")
        if self.num_windows not in (4, 8, 12):
            raise ValueError("num_windows must be 4, 8, or 12")
        if self.feature_dim != 58:
            raise ValueError("feature_dim must be 58")
        if self.hidden_dim != 128:
            raise ValueError("hidden_dim must be 128")
        if self.aggregation not in ("mean", "max", "attention", "gated_attention"):
            raise ValueError("aggregation not supported")


@dataclass(frozen=True)
class ExperimentConfig:
    split_lock_path: Path
    experiment_root: Path
    training: TrainingConfig
    model: ModelConfig
    outer_folds: int = 5
    inner_folds: int = 3
    seeds: tuple[int, ...] = (42, 123, 2026)
    final_seeds: tuple[int, ...] = (42, 123, 2026, 3407, 8888)

    def validate(self) -> None:
        self.training.validate()
        self.model.validate()
        if self.outer_folds < 2:
            raise ValueError("outer_folds must be at least 2")
        if self.inner_folds < 2:
            raise ValueError("inner_folds must be at least 2")
        if len(self.seeds) < 1:
            raise ValueError("at least one seed required")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> PipelineConfig:
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    policy = WindowPolicy(**raw["window_policy"])
    cfg = PipelineConfig(
        metadata_path=_resolve(base, raw["metadata_path"]),
        raw_parquet_path=_resolve(base, raw["raw_parquet_path"]),
        candidate_split_path=_resolve(base, raw["candidate_split_path"]),
        artifact_root=_resolve(base, raw["artifact_root"]),
        signal_length=int(raw["signal_length"]),
        sampling_rate_hz=int(raw["sampling_rate_hz"]),
        window_policy=policy,
    )
    cfg.validate()
    return cfg
