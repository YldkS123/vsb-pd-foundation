from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import pandas as pd

from .hashing import canonical_json_hash, sha256_file


_REQUIRED_CANDIDATE_COLUMNS = {
    "signal_id",
    "id_measurement",
    "phase",
    "target",
    "split",
    "split_seed",
    "group_stratum",
}
_LOCK_FIELDS = {
    "schema_version",
    "candidate_split_path",
    "candidate_split_sha256",
    "historical_prediction_sha256",
    "split_seed",
    "candidate_holdout_count",
    "contaminated_candidate_holdout_ids",
    "effective_holdout_ids",
    "effective_holdout_count",
    "effective_holdout_fraction",
    "effective_holdout_positive_measurements",
    "effective_holdout_positive_phase_signals",
    "stratification_level",
    "assignments",
    "lock_sha256",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _integer_column(frame: pd.DataFrame, column: str, *, context: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or (values % 1 != 0).any():
        raise ValueError(f"{context} {column} must contain non-null integers")
    return values.astype("int64")


def _measurement_rows(candidate_path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(candidate_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"cannot read candidate split: {candidate_path}") from exc

    missing = sorted(_REQUIRED_CANDIDATE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"candidate split missing columns: {missing}")
    if frame.empty:
        raise ValueError("candidate split must not be empty")

    frame = frame.copy()
    for column in ("signal_id", "id_measurement", "phase", "target", "split_seed", "group_stratum"):
        frame[column] = _integer_column(frame, column, context="candidate split")
    if frame["signal_id"].duplicated().any():
        raise ValueError("candidate split signal_id must be unique")
    if not frame["target"].isin([0, 1]).all():
        raise ValueError("candidate split target must be binary")
    if not frame["split"].isin(["development", "final_holdout"]).all():
        raise ValueError("candidate split values must be development or final_holdout")
    if frame["split_seed"].nunique() != 1:
        raise ValueError("candidate split split_seed must be constant")

    counts = frame.groupby("id_measurement").agg(
        row_count=("signal_id", "size"),
        split_count=("split", "nunique"),
        phase_values=("phase", lambda values: tuple(sorted(values.tolist()))),
    )
    if (
        (counts["row_count"] != 3).any()
        or (counts["split_count"] != 1).any()
        or not counts["phase_values"].map(lambda values: values == (0, 1, 2)).all()
    ):
        raise ValueError("each measurement must have three phases in exactly one split")

    group_check = frame.groupby("id_measurement").agg(
        positive_phase_count=("target", "sum"),
        group_stratum=("group_stratum", "first"),
        stratum_count=("group_stratum", "nunique"),
    )
    if (group_check["stratum_count"] != 1).any():
        raise ValueError("group_stratum must be constant within a measurement")
    if not group_check["positive_phase_count"].equals(group_check["group_stratum"]):
        raise ValueError("group_stratum must equal positive phase count")
    return frame


def _prediction_measurement_ids(path: Path) -> set[int]:
    try:
        prediction = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"cannot read historical prediction file: {path}") from exc
    if "id_measurement" not in prediction.columns:
        raise ValueError(f"historical prediction file missing id_measurement: {path}")
    return set(_integer_column(prediction, "id_measurement", context="historical prediction").tolist())


def _historical_prediction_evidence(
    historical_prediction_paths: list[Path],
) -> tuple[dict[str, str], set[int]]:
    prediction_hashes: dict[str, str] = {}
    historically_predicted_ids: set[int] = set()
    for path in sorted({item.resolve() for item in historical_prediction_paths}):
        historically_predicted_ids.update(_prediction_measurement_ids(path))
        try:
            prediction_hashes[str(path)] = sha256_file(path)
        except OSError as exc:
            raise ValueError(f"cannot hash historical prediction file: {path}") from exc
    return prediction_hashes, historically_predicted_ids


def _build_lock_payload(
    candidate_path: Path,
    frame: pd.DataFrame,
    prediction_hashes: dict[str, str],
    historically_predicted_ids: set[int],
) -> dict[str, object]:
    candidate_holdout_ids = set(
        frame.loc[frame["split"] == "final_holdout", "id_measurement"].tolist()
    )
    candidate_development_ids = set(
        frame.loc[frame["split"] == "development", "id_measurement"].tolist()
    )
    if (
        not candidate_holdout_ids
        or not candidate_development_ids
        or candidate_holdout_ids & candidate_development_ids
    ):
        raise ValueError(
            "candidate split must contain disjoint development and final_holdout IDs"
        )

    contaminated = candidate_holdout_ids & historically_predicted_ids
    effective_holdout_ids = candidate_holdout_ids - contaminated
    if not effective_holdout_ids:
        raise ValueError("no strict holdout measurements remain after contamination audit")

    effective_split = frame[["id_measurement", "split"]].drop_duplicates().copy()
    effective_split.loc[
        effective_split["id_measurement"].isin(contaminated), "split"
    ] = "development"
    assignments = effective_split.sort_values("id_measurement").to_dict(orient="records")
    effective_holdout_rows = frame.loc[frame["id_measurement"].isin(effective_holdout_ids)]
    effective_positive_measurements = int(
        (effective_holdout_rows.groupby("id_measurement")["target"].max() == 1).sum()
    )
    effective_positive_phase_signals = int(effective_holdout_rows["target"].sum())
    return {
        "schema_version": 1,
        "candidate_split_path": str(candidate_path.resolve()),
        "candidate_split_sha256": sha256_file(candidate_path),
        "historical_prediction_sha256": prediction_hashes,
        "split_seed": int(frame["split_seed"].iloc[0]),
        "candidate_holdout_count": len(candidate_holdout_ids),
        "contaminated_candidate_holdout_ids": sorted(contaminated),
        "effective_holdout_ids": sorted(effective_holdout_ids),
        "effective_holdout_count": len(effective_holdout_ids),
        "effective_holdout_fraction": len(effective_holdout_ids) / frame["id_measurement"].nunique(),
        "effective_holdout_positive_measurements": effective_positive_measurements,
        "effective_holdout_positive_phase_signals": effective_positive_phase_signals,
        "stratification_level": "exact_0_1_2_3",
        "assignments": assignments,
    }


def discover_historical_prediction_files(roots: list[Path]) -> list[Path]:
    """Find CSV predictions by both measurement schema and prediction evidence."""
    discovered: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"historical prediction root is not a directory: {root}")
        try:
            paths = sorted(root.rglob("*.csv"))
        except OSError as exc:
            raise ValueError(f"cannot scan historical prediction root: {root}") from exc
        for path in paths:
            try:
                header = pd.read_csv(path, nrows=0)
            except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
                raise ValueError(f"cannot read historical CSV header: {path}") from exc
            columns = {str(column).casefold() for column in header.columns}
            path_signals_prediction = any(
                "predict" in component.casefold() or "oof" in component.casefold()
                for component in path.relative_to(root).parts
            )
            has_prediction_evidence = any(
                "prob" in column
                or "predict" in column
                or column in {"pred", "score", "scores", "model_score", "signal_score"}
                for column in columns
            )
            if "id_measurement" in columns and (
                path_signals_prediction or has_prediction_evidence
            ):
                discovered.add(path.resolve())
    return sorted(discovered)


def create_split_lock(
    candidate_path: Path,
    historical_prediction_paths: list[Path],
    output_path: Path,
    *,
    allow_shrink_holdout: bool = False,
) -> Path:
    """Create a tamper-evident development-only split lock.

    Candidate holdout IDs found in historical predictions are released to
    development only when the caller explicitly authorizes the shrink.
    """
    frame = _measurement_rows(candidate_path)
    prediction_hashes, historically_predicted_ids = _historical_prediction_evidence(
        historical_prediction_paths
    )
    candidate_holdout_ids = set(
        frame.loc[frame["split"] == "final_holdout", "id_measurement"].tolist()
    )
    contaminated = candidate_holdout_ids & historically_predicted_ids
    if contaminated and not allow_shrink_holdout:
        raise ValueError(
            f"{len(contaminated)} candidate holdout measurements were historically predicted; "
            "rerun only after review with allow_shrink_holdout=True"
        )
    payload = _build_lock_payload(
        candidate_path, frame, prediction_hashes, historically_predicted_ids
    )
    payload["lock_sha256"] = canonical_json_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def load_split_lock(path: Path, scope: Literal["development"]) -> pd.DataFrame:
    """Load the only scope available before final evaluation: development."""
    if scope != "development":
        raise PermissionError("only development scope is available before final evaluation")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read split lock: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("split lock must contain a JSON object")
    missing = sorted(_LOCK_FIELDS - set(payload))
    extra = sorted(set(payload) - _LOCK_FIELDS)
    if missing or extra:
        raise ValueError(f"split lock fields mismatch: missing={missing}, extra={extra}")
    expected = payload.get("lock_sha256")
    if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
        raise ValueError("split lock lock_sha256 must be a SHA-256 hex string")
    hashed_payload = {key: value for key, value in payload.items() if key != "lock_sha256"}
    if canonical_json_hash(hashed_payload) != expected:
        raise ValueError("split lock hash mismatch")

    if not isinstance(payload["schema_version"], int) or isinstance(payload["schema_version"], bool):
        raise ValueError("split lock schema_version must be an integer")
    if payload["schema_version"] != 1:
        raise ValueError("split lock schema_version is unsupported")
    if not isinstance(payload["candidate_split_path"], str) or not payload["candidate_split_path"]:
        raise ValueError("split lock candidate_split_path must be a non-empty string")
    if (
        not isinstance(payload["candidate_split_sha256"], str)
        or not _SHA256_RE.fullmatch(payload["candidate_split_sha256"])
    ):
        raise ValueError("split lock candidate_split_sha256 must be a SHA-256 hex string")
    if not isinstance(payload["historical_prediction_sha256"], dict):
        raise ValueError("split lock historical_prediction_sha256 must be an object")
    for recorded_path, digest in payload["historical_prediction_sha256"].items():
        if (
            not isinstance(recorded_path, str)
            or not recorded_path
            or not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
        ):
            raise ValueError("split lock historical prediction SHA-256 fields are invalid")
    for field in (
        "split_seed",
        "candidate_holdout_count",
        "effective_holdout_count",
        "effective_holdout_positive_measurements",
        "effective_holdout_positive_phase_signals",
    ):
        if not isinstance(payload[field], int) or isinstance(payload[field], bool):
            raise ValueError(f"split lock {field} must be an integer")
    if not isinstance(payload["effective_holdout_fraction"], float):
        raise ValueError("split lock effective_holdout_fraction must be a float")
    if not isinstance(payload["stratification_level"], str):
        raise ValueError("split lock stratification_level must be a string")
    for field in ("contaminated_candidate_holdout_ids", "effective_holdout_ids"):
        values = payload[field]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in values)
            or values != sorted(set(values))
        ):
            raise ValueError(f"split lock {field} must be sorted unique integer IDs")
    assignments = payload["assignments"]
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("split lock assignments must be a non-empty list")
    for assignment in assignments:
        if (
            not isinstance(assignment, dict)
            or set(assignment) != {"id_measurement", "split"}
            or not isinstance(assignment["id_measurement"], int)
            or isinstance(assignment["id_measurement"], bool)
            or not isinstance(assignment["split"], str)
            or assignment["split"] not in {"development", "final_holdout"}
        ):
            raise ValueError("split lock assignments are invalid")

    candidate_path = Path(payload["candidate_split_path"])
    try:
        current_candidate_hash = sha256_file(candidate_path)
    except OSError as exc:
        raise ValueError(f"cannot read candidate split input: {candidate_path}") from exc
    if current_candidate_hash != payload["candidate_split_sha256"]:
        raise ValueError("split lock candidate split sha256 mismatch")
    frame = _measurement_rows(candidate_path)
    historical_paths = [Path(item) for item in payload["historical_prediction_sha256"]]
    prediction_hashes, historically_predicted_ids = _historical_prediction_evidence(
        historical_paths
    )
    if prediction_hashes != payload["historical_prediction_sha256"]:
        raise ValueError("split lock historical prediction sha256 mismatch")
    expected_payload = _build_lock_payload(
        candidate_path, frame, prediction_hashes, historically_predicted_ids
    )
    for field, expected_value in expected_payload.items():
        if payload[field] != expected_value:
            if field == "assignments":
                raise ValueError("split lock assignments do not match candidate holdout policy")
            raise ValueError(f"split lock {field} mismatch")

    verified = pd.DataFrame(expected_payload["assignments"])
    result = verified.loc[verified["split"] == "development"].copy()
    if result.empty or (result["split"] != "development").any():
        raise PermissionError("development access resolved outside development split")
    return result.sort_values("id_measurement").reset_index(drop=True)
