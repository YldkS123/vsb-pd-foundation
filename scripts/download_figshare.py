# -*- coding: utf-8 -*-
"""Resumable downloader for the two external Figshare datasets.

Usage:
  python scripts/download_figshare.py <out_path> <url>
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

BLOCK = 1024 * 1024


def download(out_path: Path, url: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    offset = tmp_path.stat().st_size if tmp_path.exists() else 0

    req = Request(url, headers={"Range": f"bytes={offset}-"})
    with urlopen(req, timeout=60) as resp:
        total = None
        content_range = resp.headers.get("Content-Range")
        if content_range and "/" in content_range:
            total = int(content_range.rsplit("/", 1)[1])
        mode = "ab" if offset else "wb"
        last_report = time.time()
        with open(tmp_path, mode) as h:
            while True:
                chunk = resp.read(BLOCK)
                if not chunk:
                    break
                h.write(chunk)
                offset += len(chunk)
                now = time.time()
                if now - last_report >= 30 or total and offset >= total:
                    last_report = now
                    print(f"downloaded {offset / 1024**3:.2f} GB"
                          + (f" / {total / 1024**3:.2f} GB" if total else ""), flush=True)
    tmp_path.replace(out_path)
    print(f"done: {out_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    download(Path(sys.argv[1]), sys.argv[2])
