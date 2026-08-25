# -*- coding: utf-8 -*-
"""Download figshare 28523090 with 202-handling (session + retry)."""
import sys
import time

import requests

URL = "https://ndownloader.figshare.com/files/52729469"
OUT = r"C:\Users\hrfxgfx\Desktop\1112\data\external_datasets\figshare_28523090\raw\corona_x_partial_discharge_dataset.zip"
API = "https://api.figshare.com/v2/articles/28523090"

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "*/*",
})

# warm up cookies via the API
try:
    r0 = s.get(API, timeout=30)
    print("api:", r0.status_code)
except Exception as e:
    print("api warmup:", type(e).__name__, e)

# handle 202: figshare prepares the download; retry until 200
import os
tmp = OUT + ".part"
offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0
print("existing part size:", offset)

prepared = False
for attempt in range(10):
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        r = s.get(URL, timeout=120, stream=True, allow_redirects=True, headers=headers)
        print(f"attempt {attempt+1}: status={r.status_code}")
        if r.status_code == 200:
            prepared = True
            break
        elif r.status_code in (202, 302):
            print("  preparing... retry in 15s")
            r.close()
            time.sleep(15)
        else:
            print("  body:", r.text[:150])
            r.close()
            time.sleep(10)
    except Exception as e:
        print(f"attempt {attempt+1}: {type(e).__name__}: {e}")
        time.sleep(10)

if not prepared:
    print("FAILED to prepare download")
    sys.exit(1)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
total = int(r.headers.get("Content-Length", 0))
print("downloading... total:", total)
mode = "ab" if offset else "wb"
last = time.time()
with open(tmp, mode) as f:
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            f.write(chunk)
            offset += len(chunk)
            if time.time() - last >= 30:
                last = time.time()
                print(f"  {offset/1e9:.2f} GB / {total/1e9:.2f} GB")
r.close()
if offset >= total:
    os.replace(tmp, OUT)
    print("DONE:", OUT)
else:
    print("incomplete:", offset, "/", total, "- resume by rerunning")
