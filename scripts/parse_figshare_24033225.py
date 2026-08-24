# -*- coding: utf-8 -*-
"""Parse the figshare 24033225 PD/noise dataset (motor insulation).

The .mat files store signals as a cell of float32 arrays and labels as a
MATLAB categorical whose codes/names live in the trailing MCOS workspace
matrix.  This script extracts both without MATLAB:

  - signals:   (N, 400) float32 waveform
  - labels:    (N,) int8, 0 = NonPD, 1 = PD

Outputs one npz per split plus a metadata json with class counts, source
checksums and the category-name mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import scipy.io as sio

# Splits shipped in the figshare article.
EXPECTED_N = {"Tr0": 12260, "Va0": 3066, "Te0": 45970}


def _read_tag(buf: bytes, pos: int) -> tuple[int, int, bool, int]:
    mt, sz = struct.unpack_from("<II", buf, pos)
    small = (mt >> 16) != 0
    if small:
        mt, sz = mt & 0xFFFF, mt >> 16
        tagw = 4
    else:
        tagw = 8
    return mt, sz, small, tagw


def _mini_payload(path: Path) -> bytes:
    """Return the trailing MCOS mini-mat payload for a v5 .mat file."""
    buf = path.read_bytes()
    elems: list[tuple[int, int, int]] = []
    pos = 128
    while pos < len(buf):
        mt, sz, small, tagw = _read_tag(buf, pos)
        ds = pos + tagw
        elems.append((mt, ds, sz))
        pos = ds + sz  # these MATLAB v7 files do not pad top-level elements
    mt, ds, sz = elems[-1]
    if mt == 15:  # miCOMPRESSED
        inner = zlib.decompress(buf[ds : ds + sz])
    else:
        inner = buf[ds : ds + sz]

    # inner is an unnamed uint8 matrix; recover its data payload.
    mtm, szm, smallm, tagwm = _read_tag(inner, 0)
    q = 0 + tagwm
    mtf, szf, smallf, tagwf = _read_tag(inner, q)
    q = q + tagwf + szf
    mtd, szd, smalld, tagwd = _read_tag(inner, q)
    q = q + tagwd + szd
    mtn, szn, smalln, tagwn = _read_tag(inner, q)
    q = q + tagwn + szn
    mtd2, szd2, smalld2, tagwd2 = _read_tag(inner, q)
    payload = inner[q + tagwd2 : q + tagwd2 + szd2]
    # Mini-mat header: version (0x0100) + "IM" + 4 zero bytes.
    return payload[8:]


def _advance(p: int, sz: int, pad: bool) -> int:
    p += sz
    if pad and p % 8:
        p += 8 - p % 8
    return p


def _parse_mini(mini: bytes, expected_n: int) -> tuple[np.ndarray, list[str]]:
    """Parse the MCOS mini-mat stream into (codes, category names)."""

    def parse_matrix(buf: bytes, pos: int, end: int, pad: bool):
        node = {
            "name": "", "class": 0, "dims": None, "fields": [],
            "children": [], "data": None, "data_type": None,
        }
        mt, sz, small, tagw = _read_tag(buf, pos)
        if mt != 14:
            return None, pos
        m_data = pos + tagw
        m_end = m_data + sz
        p = m_data
        mtf, szf, smallf, tagwf = _read_tag(buf, p)
        flags = struct.unpack_from("<I", buf, p + tagwf)[0]
        node["class"] = flags & 0xFF
        p = _advance(p + tagwf, szf, pad)
        mtd, szd, smalld, tagwd = _read_tag(buf, p)
        nd = szd // 4
        node["dims"] = struct.unpack_from("<" + "i" * nd, buf, p + tagwd)
        p = _advance(p + tagwd, szd, pad)
        mtn, szn, smalln, tagwn = _read_tag(buf, p)
        node["name"] = buf[p + tagwn : p + tagwn + szn].decode("latin1")
        p = _advance(p + tagwn, szn, pad)
        if node["class"] in (2, 3):
            mt4, sz4, small4, tagw4 = _read_tag(buf, p)
            p = _advance(p + tagw4, sz4, pad)
            mt5, sz5, small5, tagw5 = _read_tag(buf, p)
            node["fields"] = [
                fb.decode("latin1")
                for fb in buf[p + tagw5 : p + tagw5 + sz5].split(b"\x00")
                if fb
            ]
            p = _advance(p + tagw5, sz5, pad)
        while p < m_end:
            mtc, szc, smallc, tagwc = _read_tag(buf, p)
            dsc = p + tagwc
            if mtc == 14:
                sub, _ = parse_matrix(buf, p, dsc + szc, pad)
                if sub is not None:
                    node["children"].append(sub)
            elif mtc == 15:
                inner = zlib.decompress(buf[dsc : dsc + szc])
                sub, _ = parse_matrix(inner, 0, len(inner), pad)
                if sub is not None:
                    node["children"].append(sub)
            else:
                if node["data"] is None and mtc == 2:
                    node["data"] = np.frombuffer(
                        buf[dsc : dsc + szc], dtype=np.uint8
                    ).copy()
                    node["data_type"] = "uint8"
                elif node["data"] is None and mtc == 1:
                    node["data"] = np.frombuffer(
                        buf[dsc : dsc + szc], dtype=np.int8
                    ).copy()
                    node["data_type"] = "int8"
                elif node["data"] is None and mtc == 4:
                    node["data"] = np.frombuffer(
                        buf[dsc : dsc + szc], dtype=np.uint16
                    ).copy()
                    node["data_type"] = "uint16"
                elif node["data"] is None:
                    node["data"] = buf[dsc : dsc + szc]
                    node["data_type"] = f"raw{mtc}"
            p = _advance(dsc, szc, pad)
        return node, m_end

    def scan(buf: bytes, pos: int, end: int, pad: bool):
        out: list[dict] = []
        while pos < end:
            mt, sz, small, tagw = _read_tag(buf, pos)
            dsc = pos + tagw
            if mt == 15:
                inner = zlib.decompress(buf[dsc : dsc + sz])
                out += scan(inner, 0, len(inner), pad)
                pos = dsc + sz
            elif mt == 14:
                sub, _ = parse_matrix(buf, pos, dsc + sz, pad)
                if sub is not None:
                    out.append(sub)
                pos = dsc + sz
            else:
                pos = dsc + sz
        return out

    def collect(node: dict, acc: list[dict]) -> None:
        acc.append(node)
        for child in node.get("children", []):
            collect(child, acc)

    def char_strings(node: dict) -> list[str]:
        out: list[str] = []
        for child in node.get("children", []):
            if child["class"] == 4:
                raw = child.get("data")
                if isinstance(raw, bytes):
                    out.append(raw.decode("latin1"))
                elif isinstance(raw, np.ndarray):
                    out.append(bytes(raw).decode("latin1"))
            else:
                out += char_strings(child)
        return out

    roots = scan(mini, 0, len(mini), pad=True)
    all_nodes: list[dict] = []
    for root in roots:
        collect(root, all_nodes)

    codes_nodes = [
        n
        for n in all_nodes
        if n["class"] == 9 and n.get("data") is not None and len(n["data"]) == expected_n
    ]
    if not codes_nodes:
        raise ValueError(f"Could not locate the {expected_n}-element categorical codes")
    chars: list[str] = []
    for root in roots:
        chars += char_strings(root)
    cats = sorted(set(chars))
    return codes_nodes[0]["data"], cats


def parse_split(path: Path, name: str) -> dict:
    data = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    tr = data[name]
    signals = np.stack([np.asarray(s, dtype=np.float32).ravel() for s in tr.signals])
    n = len(signals)
    expected = EXPECTED_N.get(name)
    if expected is not None and n != expected:
        raise ValueError(f"{name}: expected {expected} signals, got {n}")

    mini = _mini_payload(path)
    codes, cats = _parse_mini(mini, n)
    if not {"PD", "NonPD"}.issubset(cats):
        raise ValueError(f"{name}: unexpected category names: {cats}")

    labels = np.where(codes == 2, 1, 0).astype(np.int8)  # PD -> 1
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "X": signals,
        "y": labels,
        "category_names": sorted(cats),
        "code_to_label": {1: "NonPD", 2: "PD"},
        "checksum": checksum,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--splits", default="Tr0,Va0,Te0")
    args = ap.parse_args()

    out_dir = args.out_dir or args.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.splits.split(","):
        path = args.data_dir / f"{name}.mat"
        if not path.exists():
            print(f"skip {name}: {path} not found")
            continue
        parsed = parse_split(path, name)
        np.savez_compressed(
            out_dir / f"{name}.npz",
            X=parsed["X"],
            y=parsed["y"],
            category_names=np.array(parsed["category_names"], dtype=object),
            source=str(path.name),
        )
        meta = {
            "source": str(path.name),
            "n": int(len(parsed["y"])),
            "n_pos": int(parsed["y"].sum()),
            "n_neg": int(len(parsed["y"]) - parsed["y"].sum()),
            "category_names": parsed["category_names"],
            "code_to_label": parsed["code_to_label"],
            "sha256": parsed["checksum"],
        }
        (out_dir / f"{name}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(
            f"{name}: {meta['n']} signals, "
            f"PD={meta['n_pos']}, NonPD={meta['n_neg']}, sha256={meta['sha256'][:12]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
