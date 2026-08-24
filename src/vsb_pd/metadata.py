from pathlib import Path
from numbers import Integral

import pandas as pd


REQUIRED_COLUMNS = ["signal_id", "id_measurement", "phase", "target"]


def validate_metadata(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing metadata columns: {missing}")
    required_non_null = ["signal_id", "id_measurement", "phase"]
    if frame[required_non_null].isna().any().any():
        raise ValueError("signal_id, id_measurement, and phase must not contain null values")
    if frame["signal_id"].duplicated().any():
        raise ValueError("signal_id must be unique")
    if not set(frame["target"].unique()).issubset({0, 1}):
        raise ValueError("target must be binary")

    phases = frame.groupby("id_measurement")["phase"].agg(
        lambda values: tuple(sorted(values.tolist()))
    )
    bad = phases[phases.map(lambda values: values != (0, 1, 2))]
    if not bad.empty:
        raise ValueError(
            "every measurement must contain exactly phases (0, 1, 2); "
            f"bad={bad.index[:5].tolist()}"
        )


def load_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={
        "signal_id": "int64",
        "id_measurement": "int64",
        "phase": "int8",
        "target": "int8",
    })
    validate_metadata(frame)
    return frame.sort_values(["id_measurement", "phase"]).reset_index(drop=True)


def measurement_table(metadata: pd.DataFrame) -> pd.DataFrame:
    validate_metadata(metadata)
    return (
        metadata.groupby("id_measurement", as_index=False)
        .agg(
            positive_phase_count=("target", "sum"),
            signal_count=("signal_id", "size"),
        )
        .astype({"positive_phase_count": "int8", "signal_count": "int8"})
    )


def _valid(labels: pd.Series, n_splits: int) -> bool:
    counts = labels.value_counts()
    return not counts.empty and bool((counts >= n_splits).all())


def choose_group_strata(groups: pd.DataFrame, n_splits: int) -> tuple[pd.Series, str]:
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral) or n_splits <= 0:
        raise ValueError("n_splits must be a positive integer")
    exact = groups["positive_phase_count"].astype("int8")
    if _valid(exact, n_splits):
        return exact, "exact_0_1_2_3"

    merged = exact.map({0: 0, 1: 1, 2: 1, 3: 3}).astype("int8")
    if _valid(merged, n_splits):
        return merged, "merged_1_2"

    binary = (exact >= 1).astype("int8")
    if not _valid(binary, n_splits):
        raise ValueError(f"binary strata cannot support {n_splits} folds")
    return binary, "binary_any_positive"
