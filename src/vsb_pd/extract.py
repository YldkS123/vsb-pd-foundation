from __future__ import annotations

import dataclasses
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import PipelineConfig
from .events import select_hybrid_windows
from .hashing import canonical_json_hash, sha256_file
from .locks import load_split_lock
from .metadata import load_metadata
from .windows import WindowCandidate, deduplicate_and_fill, uniform_starts


KIND_CODE = {"uniform": 0, "event": 1, "coverage_fallback": 2}
_KIND_NAME = {code: name for name, code in KIND_CODE.items()}
_NPZ_KEYS = {
    "measurement_id",
    "signal_ids",
    "phases",
    "targets",
    "windows",
    "starts",
    "kinds",
    "scores",
    "pipeline_hash",
    "source_parquet_sha256",
}


def read_measurement_signals(parquet_path: Path, signal_ids: list[int]) -> np.ndarray:
    """Read exactly the three raw signal columns belonging to one measurement."""
    if len(signal_ids) != 3:
        raise ValueError("one measurement must contain exactly three signal IDs")
    names = [str(value) for value in signal_ids]
    if len(set(names)) != 3:
        raise ValueError("one measurement must contain three distinct signal IDs")
    try:
        table = pq.ParquetFile(parquet_path).read(columns=names)
    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
        raise ValueError(f"cannot read requested Parquet columns {names}: {parquet_path}") from exc
    if table.column_names != names:
        raise ValueError(f"cannot read requested Parquet columns {names}: {parquet_path}")
    arrays = [table[name].to_numpy(zero_copy_only=False) for name in names]
    result = np.stack(arrays, axis=0)
    if result.ndim != 2 or result.shape[0] != 3:
        raise ValueError(f"unexpected signal matrix shape: {result.shape}")
    return result


def pipeline_identity(
    config: PipelineConfig,
    split_lock_path: Path,
) -> tuple[str, str]:
    """Return the immutable extraction identity and source-data digest."""
    source_hash = sha256_file(config.raw_parquet_path)
    payload = {
        "signal_length": config.signal_length,
        "sampling_rate_hz": config.sampling_rate_hz,
        "window_policy": dataclasses.asdict(config.window_policy),
        "split_lock_sha256": sha256_file(split_lock_path),
        "source_parquet_sha256": source_hash,
        "schema_version": 1,
    }
    return canonical_json_hash(payload), source_hash


def _write_npz_atomic(destination: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = destination.with_suffix(".npz.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_manifest_row(
    destination: Path,
    measurement_id: int,
) -> tuple[str, str, str] | None:
    """Return a recorded hash binding for an existing artifact, if one exists."""
    for manifest_path in sorted(destination.parent.glob("manifest_*.parquet")):
        try:
            manifest = pd.read_parquet(manifest_path)
        except Exception as exc:
            raise ValueError(f"cannot read existing manifest: {manifest_path}") from exc
        required = {
            "id_measurement",
            "artifact_path",
            "artifact_sha256",
            "pipeline_hash",
            "source_parquet_sha256",
        }
        if not required.issubset(manifest.columns):
            raise ValueError(f"existing manifest has invalid schema: {manifest_path}")
        matches = manifest.loc[manifest["id_measurement"].astype(int) == measurement_id]
        for row in matches.itertuples(index=False):
            if Path(row.artifact_path).resolve() == destination.resolve():
                return (
                    str(row.artifact_sha256),
                    str(row.pipeline_hash),
                    str(row.source_parquet_sha256),
                )
    return None


def _window_values_valid(
    starts: np.ndarray,
    kinds: np.ndarray,
    scores: np.ndarray,
    config: PipelineConfig,
) -> bool:
    """Validate the locked semantic contract for selected window metadata."""
    max_start = config.signal_length - config.window_policy.window_length
    if ((starts < 0) | (starts > max_start)).any() or not np.isfinite(scores).all():
        return False
    if not np.isin(kinds, [0, 1, 2]).all():
        return False
    uniform_count = config.window_policy.uniform_count
    event_count = config.window_policy.event_count
    expected_uniform = (
        uniform_starts(
            config.signal_length,
            config.window_policy.window_length,
            uniform_count,
        )
        if uniform_count else []
    )
    for phase in range(3):
        phase_kinds = kinds[phase]
        phase_starts = starts[phase]
        phase_scores = scores[phase]
        uniform = phase_kinds == 0
        event_or_fallback = np.isin(phase_kinds, [1, 2])
        if (
            int(np.sum(uniform)) != uniform_count
            or int(np.sum(event_or_fallback)) != event_count
            or sorted(phase_starts[uniform].tolist()) != expected_uniform
            or not np.array_equal(phase_kinds[:uniform_count], np.zeros(uniform_count, dtype=np.uint8))
            or not event_or_fallback[uniform_count:].all()
            or not np.all(phase_kinds[1:] >= phase_kinds[:-1])
            or not np.all(phase_scores[np.isin(phase_kinds, [0, 2])] == 0.0)
            or not np.all(phase_scores[phase_kinds == 1] > 0.0)
        ):
            return False
        uniform_candidates = [
            WindowCandidate(int(start), "uniform", float(score))
            for start, score in zip(
                phase_starts[phase_kinds == 0], phase_scores[phase_kinds == 0]
            )
        ]
        event_candidates = [
            WindowCandidate(int(start), "event", float(score))
            for start, score in zip(
                phase_starts[phase_kinds == 1], phase_scores[phase_kinds == 1]
            )
        ]
        try:
            expected = deduplicate_and_fill(
                signal_length=config.signal_length,
                window_length=config.window_policy.window_length,
                uniform=uniform_candidates,
                events=event_candidates,
                event_quota=event_count,
                dedup_iou=config.window_policy.dedup_iou,
                grid_size=config.window_policy.fallback_grid_size,
            )
        except (RuntimeError, ValueError):
            return False
        observed = [
            WindowCandidate(int(start), _KIND_NAME[int(kind)], float(score))
            for start, kind, score in zip(phase_starts, phase_kinds, phase_scores)
        ]
        if observed != expected:
            return False
    return True


def _existing_artifact_is_valid(
    destination: Path,
    measurement_id: int,
    signal_ids: list[int],
    phases: np.ndarray,
    targets: np.ndarray,
    pipeline_hash: str,
    source_hash: str,
    config: PipelineConfig,
) -> bool:
    binding = _read_manifest_row(destination, measurement_id)
    if binding is None:
        return False
    recorded_hash, recorded_pipeline, recorded_source = binding
    if recorded_pipeline != pipeline_hash or recorded_source != source_hash:
        raise ValueError(f"existing manifest hash mismatch: {destination}")
    if sha256_file(destination) != recorded_hash:
        raise ValueError(f"existing artifact hash mismatch: {destination}")
    try:
        schema = pq.ParquetFile(config.raw_parquet_path).schema_arrow
        raw_window_dtype = np.result_type(*[
            np.dtype(schema.field(str(signal_id)).type.to_pandas_dtype())
            for signal_id in signal_ids
        ])
    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
        raise ValueError(
            f"cannot determine source signal dtype for existing artifact: {destination}"
        ) from exc
    try:
        with np.load(destination, allow_pickle=False) as artifact:
            if set(artifact.files) != _NPZ_KEYS:
                raise ValueError("artifact key schema mismatch")
            checks = (
                artifact["measurement_id"].shape == ()
                and artifact["measurement_id"].dtype == np.dtype("int64")
                and int(artifact["measurement_id"].item()) == measurement_id
                and artifact["signal_ids"].dtype == np.dtype("int64")
                and artifact["signal_ids"].shape == (3,)
                and artifact["signal_ids"].tolist() == signal_ids
                and artifact["phases"].dtype == np.dtype("int8")
                and np.array_equal(artifact["phases"], phases)
                and artifact["targets"].dtype == np.dtype("int8")
                and np.array_equal(artifact["targets"], targets)
                and artifact["windows"].shape
                == (3, config.window_policy.total_count, config.window_policy.window_length)
                and artifact["windows"].dtype == raw_window_dtype
                and artifact["starts"].dtype == np.dtype("int64")
                and artifact["starts"].shape == (3, config.window_policy.total_count)
                and artifact["kinds"].dtype == np.dtype("uint8")
                and artifact["kinds"].shape == (3, config.window_policy.total_count)
                and artifact["scores"].dtype == np.dtype("float32")
                and artifact["scores"].shape == (3, config.window_policy.total_count)
                and _window_values_valid(
                    artifact["starts"], artifact["kinds"], artifact["scores"], config
                )
                and str(artifact["pipeline_hash"].item()) == pipeline_hash
                and str(artifact["source_parquet_sha256"].item()) == source_hash
            )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"existing artifact is invalid: {destination}") from exc
    if not checks:
        raise ValueError(f"existing artifact schema mismatch: {destination}")
    return True


def _manifest_path(output_dir: Path, extraction_mode: str) -> Path:
    return output_dir / f"manifest_{extraction_mode}.parquet"


def extract_development(
    config: PipelineConfig,
    split_lock_path: Path,
    *,
    allow_test_signal_length: bool = False,
    limit_measurements: int | None = None,
) -> Path:
    """Extract deterministic artifacts for the lock's development measurements only."""
    if allow_test_signal_length:
        config.window_policy.validate()
        if config.sampling_rate_hz != 40_000_000:
            raise ValueError("sampling_rate_hz must be 40000000")
    else:
        config.validate()
    if limit_measurements is not None and limit_measurements < 1:
        raise ValueError("limit_measurements must be positive")

    metadata = load_metadata(config.metadata_path)
    development = load_split_lock(split_lock_path, scope="development")
    development_ids = sorted(development["id_measurement"].astype(int).tolist())
    if limit_measurements is None:
        extraction_mode = "full"
    else:
        development_ids = development_ids[:limit_measurements]
        extraction_mode = "smoke"
    metadata_ids = set(metadata["id_measurement"].astype(int))
    missing = set(development_ids) - metadata_ids
    if missing:
        raise ValueError(f"development measurements missing from metadata: {sorted(missing)[:5]}")

    pipeline_hash, source_hash = pipeline_identity(config, split_lock_path)
    output_dir = config.artifact_root / "windows" / pipeline_hash / "development"
    rows: list[dict[str, object]] = []
    for measurement_id in development_ids:
        group = metadata.loc[metadata["id_measurement"] == measurement_id].sort_values("phase")
        signal_ids = group["signal_id"].astype(int).tolist()
        phases = group["phase"].to_numpy(dtype=np.int8)
        targets = group["target"].to_numpy(dtype=np.int8)
        if signal_ids.__len__() != 3 or phases.tolist() != [0, 1, 2]:
            raise ValueError(f"measurement {measurement_id} is not phase-complete")
        destination = output_dir / f"{measurement_id}.npz"
        write_artifact = not destination.exists()
        if not write_artifact:
            write_artifact = not _existing_artifact_is_valid(
                destination,
                measurement_id,
                signal_ids,
                phases,
                targets,
                pipeline_hash,
                source_hash,
                config,
            )
        if write_artifact:
            signals = read_measurement_signals(config.raw_parquet_path, signal_ids)
            if signals.shape != (3, config.signal_length):
                raise ValueError(
                    f"measurement {measurement_id} has shape {signals.shape}, "
                    f"expected (3, {config.signal_length})"
                )
            selected_by_phase = [
                select_hybrid_windows(signals[phase], config.window_policy)
                for phase in range(3)
            ]
            starts = np.asarray(
                [[item.start for item in selected] for selected in selected_by_phase],
                dtype=np.int64,
            )
            kinds = np.asarray(
                [[KIND_CODE[item.kind] for item in selected] for selected in selected_by_phase],
                dtype=np.uint8,
            )
            scores = np.asarray(
                [[item.score for item in selected] for selected in selected_by_phase],
                dtype=np.float32,
            )
            windows = np.stack([
                np.stack([
                    signals[phase, start:start + config.window_policy.window_length]
                    for start in starts[phase]
                ])
                for phase in range(3)
            ])
            _write_npz_atomic(destination, {
                "measurement_id": np.asarray(measurement_id, dtype=np.int64),
                "signal_ids": np.asarray(signal_ids, dtype=np.int64),
                "phases": phases,
                "targets": targets,
                "windows": windows,
                "starts": starts,
                "kinds": kinds,
                "scores": scores,
                "pipeline_hash": np.asarray(pipeline_hash),
                "source_parquet_sha256": np.asarray(source_hash),
            })
        rows.append({
            "id_measurement": measurement_id,
            "artifact_path": str(destination.resolve()),
            "artifact_sha256": sha256_file(destination),
            "pipeline_hash": pipeline_hash,
            "source_parquet_sha256": source_hash,
            "window_count": config.window_policy.total_count,
            "window_length": config.window_policy.window_length,
            "extraction_mode": extraction_mode,
        })
    manifest_path = _manifest_path(output_dir, extraction_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(manifest_path, index=False)
    return manifest_path
