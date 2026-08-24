from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import PipelineConfig
from .extract import _NPZ_KEYS, _window_values_valid, pipeline_identity
from .hashing import sha256_file
from .locks import load_split_lock
from .metadata import load_metadata


_MANIFEST_COLUMNS = {
    "id_measurement",
    "artifact_path",
    "artifact_sha256",
    "pipeline_hash",
    "source_parquet_sha256",
    "window_count",
    "window_length",
    "extraction_mode",
}


@dataclass(frozen=True)
class AuditReport:
    ok: bool
    measurements: int
    windows: int
    errors: tuple[str, ...]

    def require_ok(self) -> None:
        if not self.ok:
            raise RuntimeError("development artifact audit failed:\n" + "\n".join(self.errors))


def _failed(message: str) -> AuditReport:
    return AuditReport(False, 0, 0, (message,))


def _is_string_column(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _validate_manifest_schema(manifest: pd.DataFrame, errors: list[str]) -> bool:
    columns = set(manifest.columns)
    if columns != _MANIFEST_COLUMNS:
        errors.append(
            "manifest schema mismatch: "
            f"missing={sorted(_MANIFEST_COLUMNS - columns)}, extra={sorted(columns - _MANIFEST_COLUMNS)}"
        )
        return False
    expected = {
        "id_measurement": manifest["id_measurement"].dtype == np.dtype("int64"),
        "window_count": manifest["window_count"].dtype == np.dtype("int64"),
        "window_length": manifest["window_length"].dtype == np.dtype("int64"),
        "artifact_path": _is_string_column(manifest["artifact_path"]),
        "artifact_sha256": _is_string_column(manifest["artifact_sha256"]),
        "pipeline_hash": _is_string_column(manifest["pipeline_hash"]),
        "source_parquet_sha256": _is_string_column(manifest["source_parquet_sha256"]),
        "extraction_mode": _is_string_column(manifest["extraction_mode"]),
    }
    invalid = sorted(name for name, valid in expected.items() if not valid)
    if invalid:
        errors.append(f"manifest column types mismatch: {invalid}")
    if manifest.isna().any().any():
        errors.append("manifest contains null values")
    return not invalid and not manifest.isna().any().any()


def _expected_raw_dtype(config: PipelineConfig, signal_ids: list[int]) -> np.dtype | None:
    try:
        schema = pq.ParquetFile(config.raw_parquet_path).schema_arrow
        return np.result_type(*[
            np.dtype(schema.field(str(signal_id)).type.to_pandas_dtype())
            for signal_id in signal_ids
        ])
    except (OSError, ValueError, KeyError, pa.ArrowException):
        return None


def _outside_development_label(measurement_id: int, final_holdout_ids: set[int]) -> str:
    if measurement_id in final_holdout_ids:
        return "final_holdout"
    return "unknown"


def _audit_artifact(
    artifact_path: Path,
    row: object,
    *,
    config: PipelineConfig,
    pipeline_hash: str,
    source_hash: str,
    expected_signal_ids: list[int],
    expected_phases: np.ndarray,
    expected_targets: np.ndarray,
    errors: list[str],
) -> None:
    if not artifact_path.is_file():
        errors.append(f"artifact missing: {artifact_path}")
        return
    try:
        actual_hash = sha256_file(artifact_path)
    except OSError as exc:
        errors.append(f"artifact cannot be hashed: {artifact_path}: {exc}")
        return
    if actual_hash != row.artifact_sha256:
        errors.append(f"artifact sha256 mismatch: {artifact_path}")
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            if set(archive.files) != _NPZ_KEYS:
                errors.append(f"artifact key schema mismatch: {artifact_path}")
                return
            arrays = {key: archive[key] for key in archive.files}
    except Exception as exc:
        errors.append(f"artifact cannot be read: {artifact_path}: {exc}")
        return

    k = config.window_policy.total_count
    expected_dtype = _expected_raw_dtype(config, expected_signal_ids)
    checks = {
        "measurement_id": (
            arrays["measurement_id"].shape == ()
            and arrays["measurement_id"].dtype == np.dtype("int64")
            and int(arrays["measurement_id"].item()) == int(row.id_measurement)
        ),
        "signal_ids": (
            arrays["signal_ids"].shape == (3,)
            and arrays["signal_ids"].dtype == np.dtype("int64")
            and arrays["signal_ids"].tolist() == expected_signal_ids
        ),
        "phases": (
            arrays["phases"].shape == (3,)
            and arrays["phases"].dtype == np.dtype("int8")
            and np.array_equal(arrays["phases"], expected_phases)
        ),
        "targets": (
            arrays["targets"].shape == (3,)
            and arrays["targets"].dtype == np.dtype("int8")
            and np.array_equal(arrays["targets"], expected_targets)
        ),
        "windows": (
            arrays["windows"].shape == (3, k, config.window_policy.window_length)
            and expected_dtype is not None
            and arrays["windows"].dtype == expected_dtype
        ),
        "starts": arrays["starts"].shape == (3, k) and arrays["starts"].dtype == np.dtype("int64"),
        "kinds": arrays["kinds"].shape == (3, k) and arrays["kinds"].dtype == np.dtype("uint8"),
        "scores": arrays["scores"].shape == (3, k) and arrays["scores"].dtype == np.dtype("float32"),
        "embedded hashes": (
            arrays["pipeline_hash"].shape == ()
            and arrays["source_parquet_sha256"].shape == ()
            and str(arrays["pipeline_hash"].item()) == pipeline_hash
            and str(arrays["source_parquet_sha256"].item()) == source_hash
        ),
    }
    for name, valid in checks.items():
        if not valid:
            errors.append(f"artifact {name} mismatch: {artifact_path}")

    starts = arrays["starts"]
    kinds = arrays["kinds"]
    scores = arrays["scores"]
    if starts.shape == (3, k) and kinds.shape == (3, k) and scores.shape == (3, k):
        max_start = config.signal_length - config.window_policy.window_length
        if ((starts < 0) | (starts > max_start)).any():
            errors.append(f"illegal window start: {artifact_path}")
        if not np.isfinite(scores).all():
            errors.append(f"non-finite event score: {artifact_path}")
        if not _window_values_valid(starts, kinds, scores, config):
            errors.append(f"window selection replay mismatch: {artifact_path}")


def audit_development(
    config: PipelineConfig,
    split_lock_path: Path,
    manifest_path: Path,
) -> AuditReport:
    """Fail closed while auditing only development extraction artifacts."""
    try:
        manifest = pd.read_parquet(manifest_path)
    except Exception as exc:
        return _failed(f"manifest cannot be read: {manifest_path}: {exc}")
    if manifest.empty:
        return _failed("manifest is empty")

    errors: list[str] = []
    if not _validate_manifest_schema(manifest, errors):
        return AuditReport(False, len(manifest), 0, tuple(errors))
    try:
        development = load_split_lock(split_lock_path, scope="development")
        metadata = load_metadata(config.metadata_path)
        pipeline_hash, source_hash = pipeline_identity(config, split_lock_path)
    except Exception as exc:
        return AuditReport(False, len(manifest), 0, (f"audit prerequisites invalid: {exc}",))
    development_ids = set(development["id_measurement"].astype(int))
    try:
        candidate = pd.read_csv(config.candidate_split_path)
        final_holdout_ids = set(candidate.loc[
            candidate["split"] == "final_holdout", "id_measurement"
        ].astype(int))
    except Exception:
        final_holdout_ids = set()

    manifest_ids = manifest["id_measurement"].astype(int)
    if manifest_ids.duplicated().any():
        errors.append("manifest contains duplicate measurements")
    if manifest["artifact_path"].duplicated().any():
        errors.append("manifest contains duplicate artifact paths")
    outside = set(manifest_ids) - development_ids
    for measurement_id in outside:
        errors.append(
            "final_holdout or unknown measurement appears in development manifest: "
            f"{_outside_development_label(measurement_id, final_holdout_ids)}"
        )

    modes = set(manifest["extraction_mode"])
    if len(modes) != 1 or not modes.issubset({"full", "smoke"}):
        errors.append(f"invalid extraction_mode values: {sorted(modes)}")
    elif modes == {"full"} and set(manifest_ids) != development_ids:
        errors.append("full manifest does not contain every development measurement")
    elif modes == {"smoke"} and not set(manifest_ids) < development_ids:
        errors.append("smoke manifest must be a proper non-empty development subset")
    if (manifest["pipeline_hash"] != pipeline_hash).any():
        errors.append("manifest pipeline hash mismatch")
    if (manifest["source_parquet_sha256"] != source_hash).any():
        errors.append("manifest source hash mismatch")
    if (manifest["window_count"] != config.window_policy.total_count).any():
        errors.append("manifest window_count mismatch")
    if (manifest["window_length"] != config.window_policy.window_length).any():
        errors.append("manifest window_length mismatch")

    output_dir = (config.artifact_root / "windows" / pipeline_hash / "development").resolve()
    manifest_paths: set[Path] = set()
    metadata_by_measurement = {
        int(measurement_id): group.sort_values("phase")
        for measurement_id, group in metadata.groupby("id_measurement")
    }
    for row in manifest.itertuples(index=False):
        artifact_path = Path(row.artifact_path).resolve()
        manifest_paths.add(artifact_path)
        try:
            artifact_path.relative_to(output_dir)
        except ValueError:
            errors.append(f"artifact lies outside expected development directory: {artifact_path}")
            continue
        group = metadata_by_measurement.get(int(row.id_measurement))
        if group is None or len(group) != 3:
            errors.append(f"artifact has no phase-complete metadata: {artifact_path}")
            continue
        _audit_artifact(
            artifact_path,
            row,
            config=config,
            pipeline_hash=pipeline_hash,
            source_hash=source_hash,
            expected_signal_ids=group["signal_id"].astype(int).tolist(),
            expected_phases=group["phase"].to_numpy(dtype=np.int8),
            expected_targets=group["target"].to_numpy(dtype=np.int8),
            errors=errors,
        )

    try:
        artifact_paths = {path.resolve() for path in output_dir.rglob("*.npz")}
        for orphan in sorted(artifact_paths - manifest_paths):
            try:
                with np.load(orphan, allow_pickle=False) as archive:
                    measurement_id = int(archive["measurement_id"].item())
                label = _outside_development_label(measurement_id, final_holdout_ids)
                errors.append(f"orphan NPZ exists in development artifacts: {label}")
            except Exception:
                errors.append("orphan NPZ exists in development artifacts: unreadable")
        prediction_files = [
            path for path in output_dir.rglob("*")
            if path.is_file() and path.suffix == ".csv"
        ]
        if prediction_files:
            scanned: set[Path] = set()
            for csv_file in prediction_files:
                path_signals_prediction = any(
                    "predict" in component.casefold() or "oof" in component.casefold() or "pred" in component.casefold()
                    for component in csv_file.relative_to(output_dir).parts
                )
                if path_signals_prediction:
                    scanned.add(csv_file)
                    continue
                columns: set[str] = set()
                try:
                    columns = {str(column).casefold() for column in pd.read_csv(csv_file, nrows=0).columns}
                except Exception:
                    continue
                has_prediction_evidence = any(
                    "prob" in column
                    or "predict" in column
                    or column in {"pred", "score", "scores", "model_score", "signal_score"}
                    for column in columns
                )
                if "id_measurement" in columns and has_prediction_evidence:
                    scanned.add(csv_file)
            if scanned:
                errors.append("prediction files exist in development artifacts")
    except OSError as exc:
        errors.append(f"development artifact directory cannot be scanned: {exc}")

    windows = int(manifest["window_count"].sum()) * 3
    return AuditReport(not errors, len(manifest), windows, tuple(errors))
