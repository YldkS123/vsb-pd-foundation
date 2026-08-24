# -*- coding: utf-8 -*-
"""Convert the figshare 28523090 .bin captures to compact 8192-sample windows.

Each .bin is a 10,004,096-byte oscilloscope capture; the first 0x1470 bytes
are header metadata and the rest is uint8 analog data.  Using the official
convertor.py scaling (5 V/div, 25 codes/div, -7.7 V offset), each recording is
converted to volts and K windows of 8192 samples are taken at evenly spaced
starts.  The result is one npz (X, class codes, channels, filenames) plus a
metadata json, suitable for zero-shot / cross-domain experiments with the
VSB 8192-window CNN.

Usage:
    python scripts/convert_figshare_28523090.py \
        D:/datasets/figshare_28523090/extracted \
        D:/datasets/figshare_28523090/windows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA_OFFSET = 0x1470
CH_VOLT_DIV_VAL = 5000.0
CODE_PER_DIV = 25.0
CH_VERT_OFFSET = -7.7
WINDOW_LENGTH = 8192
K_WINDOWS = 8

CLASSES = [
    "background",
    "background_day2",
    "corona",
    "hv_bg",
    "pd",
    "pd_corona",
    "pd_corona_HI",
    "pd_HI",
]


def uniform_starts(signal_length: int, window_length: int, count: int) -> list[int]:
    if count == 1:
        return [(signal_length - window_length) // 2]
    max_start = signal_length - window_length
    return [
        (2 * index * max_start + count - 1) // (2 * (count - 1))
        for index in range(count)
    ]


def parse_filename(name: str) -> tuple[str, str, str]:
    base = name[:-4]
    tokens = base.split("_")
    channel = tokens[0]
    sample = tokens[-1]
    fault_type = "_".join(tokens[1:-1])
    return channel, fault_type, sample


def convert_file(path: Path) -> np.ndarray:
    data = path.read_bytes()
    analog = np.frombuffer(data[DATA_OFFSET:], dtype=np.uint8).astype(np.float32)
    return (analog - 128.0) * CH_VOLT_DIV_VAL / 1000.0 / CODE_PER_DIV + CH_VERT_OFFSET


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    files = sorted(args.input_dir.rglob("*.bin"))
    if not files:
        print("no .bin files found")
        return 1

    starts = uniform_starts(9_998_864, WINDOW_LENGTH, K_WINDOWS)
    X = np.empty((len(files), K_WINDOWS, WINDOW_LENGTH), dtype=np.float32)
    class_idx = np.empty(len(files), dtype=np.int8)
    channels: list[str] = []
    names: list[str] = []
    skipped: list[str] = []
    valid = 0

    for i, path in enumerate(files):
        volt = convert_file(path)
        if len(volt) != 9_998_864:
            print(f"  skip {path.name}: analog length {len(volt)} != 9,998,864")
            skipped.append(path.name)
            continue
        channel, fault, _ = parse_filename(path.name)
        if fault not in CLASSES:
            raise ValueError(f"{path.name}: unknown class {fault!r}")
        X[valid] = np.stack([volt[s : s + WINDOW_LENGTH] for s in starts])
        class_idx[valid] = CLASSES.index(fault)
        channels.append(channel)
        names.append(path.name)
        valid += 1
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(files)} (valid {valid})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    X = X[:valid]
    class_idx = class_idx[:valid]
    np.savez_compressed(
        args.out_dir / "dataset_windows.npz",
        X=X,
        class_idx=class_idx,
        channels=np.array(channels, dtype=object),
        filenames=np.array(names, dtype=object),
    )
    meta = {
        "n_recordings": int(len(files)),
        "n_used": int(valid),
        "skipped": skipped,
        "skipped_note": "files skipped because analog length != 9,998,864 (truncated download)",
        "k_windows": K_WINDOWS,
        "window_length": WINDOW_LENGTH,
        "sampling_note": "uint8 analog after 0x1470-byte header, "
                         f"scaled {CH_VOLT_DIV_VAL} mV/div, {CODE_PER_DIV} codes/div, "
                         f"offset {CH_VERT_OFFSET} V",
        "classes": CLASSES,
        "class_counts": {c: int((class_idx == i).sum()) for i, c in enumerate(CLASSES)},
        "channel_counts": {
            ch: int(np.sum(np.array(channels) == ch)) for ch in ("C1", "C2")
        },
        "window_starts": starts,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
