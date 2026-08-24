from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


@dataclass(frozen=True, order=True)
class WindowCandidate:
    start: int
    kind: str
    score: float


def uniform_starts(signal_length: int, window_length: int, count: int) -> list[int]:
    """Return exactly spaced legal starts, including both ends when possible."""
    if not all(_is_integer(value) for value in (signal_length, window_length, count)):
        raise ValueError("signal_length, window_length, and count must be integers")
    if not (0 < window_length <= signal_length):
        raise ValueError("window_length must be in [1, signal_length]")
    if count < 1:
        raise ValueError("count must be positive")

    max_start = signal_length - window_length
    if count == 1:
        return [max_start // 2]

    denominator = count - 1
    return [
        (2 * index * max_start + denominator) // (2 * denominator)
        for index in range(count)
    ]


def interval_iou(a_start: int, b_start: int, window_length: int) -> float:
    """Calculate IoU for two intervals with the same positive length."""
    if not all(_is_integer(value) for value in (a_start, b_start, window_length)):
        raise ValueError("interval starts and window_length must be integers")
    if window_length <= 0:
        raise ValueError("window_length must be positive")

    intersection = max(
        0,
        min(a_start, b_start) + window_length - max(a_start, b_start),
    )
    union = 2 * window_length - intersection
    return intersection / union


def _duplicates(
    candidate: WindowCandidate,
    selected: list[WindowCandidate],
    window_length: int,
    threshold: float,
) -> bool:
    return any(
        interval_iou(candidate.start, old.start, window_length) >= threshold
        for old in selected
    )


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _validate_candidates(
    candidates: list[WindowCandidate],
    expected_kind: str,
    max_start: int,
) -> None:
    for candidate in candidates:
        if candidate.kind != expected_kind:
            raise ValueError(f"{expected_kind} candidates must use kind={expected_kind!r}")
        if not _is_integer(candidate.start):
            raise ValueError("candidate start must be an integer")
        if not 0 <= candidate.start <= max_start:
            raise ValueError("candidate start must be within the legal range")
        if not _is_finite_real(candidate.score):
            raise ValueError("candidate score must be finite")


def deduplicate_and_fill(
    signal_length: int,
    window_length: int,
    uniform: list[WindowCandidate],
    events: list[WindowCandidate],
    event_quota: int,
    dedup_iou: float,
    grid_size: int,
) -> list[WindowCandidate]:
    """Keep prioritized events, then fill their quota with coverage windows."""
    uniform_starts(signal_length, window_length, 1)
    if not _is_integer(event_quota) or event_quota < 0:
        raise ValueError("event_quota must be non-negative")
    if not _is_finite_real(dedup_iou) or not 0 <= dedup_iou <= 1:
        raise ValueError("dedup_iou must be in [0, 1]")
    if not _is_integer(grid_size) or grid_size < 1:
        raise ValueError("grid_size must be positive")

    max_start = signal_length - window_length
    _validate_candidates(uniform, "uniform", max_start)
    _validate_candidates(events, "event", max_start)

    selected = sorted(uniform, key=lambda candidate: candidate.start)
    accepted_count = 0
    for event in sorted(events, key=lambda candidate: (-candidate.score, candidate.start)):
        if accepted_count == event_quota:
            break
        if not _duplicates(event, selected, window_length, dedup_iou):
            selected.append(event)
            accepted_count += 1

    valid_grid = set(uniform_starts(signal_length, window_length, grid_size))
    while accepted_count < event_quota:
        candidates = [
            start
            for start in valid_grid
            if not _duplicates(
                WindowCandidate(start, "coverage_fallback", 0.0),
                selected,
                window_length,
                dedup_iou,
            )
        ]
        if not candidates:
            raise RuntimeError("cannot fill event quota without duplicate windows")

        choice = max(
            candidates,
            key=lambda start: (
                min(abs(start - old.start) for old in selected),
                -start,
            ),
        )
        selected.append(WindowCandidate(choice, "coverage_fallback", 0.0))
        valid_grid.remove(choice)
        accepted_count += 1

    kind_order = {"uniform": 0, "event": 1, "coverage_fallback": 2}
    return sorted(selected, key=lambda candidate: (kind_order[candidate.kind], candidate.start))
