# -*- coding: utf-8 -*-
"""Push the local working tree to GitHub via the Git Database REST API.

Robust version: exports the full file tree of the local HEAD commit with
`git archive`, uploads every file as a blob, builds one tree, one commit,
and updates the main ref. Works when github.com:443 is unreachable but
api.github.com is reachable.

Requires: gh authenticated (repo scope); requests installed.
"""
import base64
import json
import subprocess
import sys
import tarfile
import io

import requests

TOKEN = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
REPO = "YldkS123/vsb-pd-foundation"
API = f"https://api.github.com/repos/{REPO}/git"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}
MAX_BLOB = 90 * 1024 * 1024  # API blob size limit ~100MB


def api(path, method="GET", data=None):
    r = requests.request(method, f"{API}/{path}", headers=HEADERS, json=data, timeout=120)
    if r.status_code not in (200, 201):
        print(f"!! {method} {path} -> {r.status_code}: {r.text[:400]}")
        r.raise_for_status()
    return r.json()


def main():
    # 1. export HEAD tree as tar
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    msg = subprocess.run(["git", "log", "-1", "--format=%B", head], capture_output=True, text=True).stdout.strip()
    an = subprocess.run(["git", "log", "-1", "--format=%an", head], capture_output=True, text=True).stdout.strip()
    ae = subprocess.run(["git", "log", "-1", "--format=%ae", head], capture_output=True, text=True).stdout.strip()
    at = int(subprocess.run(["git", "log", "-1", "--format=%at", head], capture_output=True, text=True).stdout.strip())
    print("HEAD:", head, "| author:", an)

    tar_bytes = subprocess.run(
        ["git", "archive", "--format=tar", head], capture_output=True).stdout
    print(f"archive bytes: {len(tar_bytes)}")

    tf = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:")
    entries = []
    blob_cache = {}
    n = 0
    for member in tf.getmembers():
        if not member.isfile():
            continue
        f = tf.extractfile(member)
        if f is None:
            continue
        content = f.read()
        if len(content) > MAX_BLOB:
            print(f"!! skipping oversized file {member.name} ({len(content)} bytes)")
            continue
        # upload blob (dedupe by content hash)
        key = member.name
        try:
            text = content.decode("utf-8")
            payload = {"content": text, "encoding": "utf-8"}
        except UnicodeDecodeError:
            payload = {"content": base64.b64encode(content).decode(), "encoding": "base64"}
        created = api("blobs", "POST", payload)
        entries.append({"path": member.name, "mode": "100644", "type": "blob",
                        "sha": created["sha"]})
        n += 1
        if n % 50 == 0:
            print(f"  uploaded {n} blobs...")
    print(f"total blobs: {n}")

    # 2. build tree
    tree = api("trees", "POST", {"tree": entries})
    print("tree:", tree["sha"])

    # 3. create commit (no parent for a fresh repo; if main exists, use its head)
    try:
        existing = api("refs/heads/main")
        parent_sha = existing["object"]["sha"]
        print("existing main:", parent_sha)
    except Exception:
        parent_sha = None
        print("no existing main ref")
    commit = api("commits", "POST", {
        "message": msg,
        "tree": tree["sha"],
        "parents": [parent_sha] if parent_sha else [],
        "author": {"name": an, "email": ae, "date": f"{at} +0800"},
        "committer": {"name": an, "email": ae, "date": f"{at} +0800"},
    })
    print("commit:", commit["sha"])

    # 4. update ref
    if parent_sha:
        api("refs/heads/main", "PATCH", {"sha": commit["sha"], "force": True})
    else:
        api("refs", "POST", {"ref": "refs/heads/main", "sha": commit["sha"]})
    print("DONE -> https://github.com/YldkS123/vsb-pd-foundation")


if __name__ == "__main__":
    main()
