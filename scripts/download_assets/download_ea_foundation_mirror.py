#!/usr/bin/env python3
"""Resume pinned foundation downloads; never follow an unapproved HF redirect."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import shutil
from urllib.parse import urljoin, urlparse

import requests

REPO = "OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model"
ROOT = Path(__file__).resolve().parents[2]


def allowed_url(url, allow_mirror_cdn=False):
    host = urlparse(url).hostname or ""
    return urlparse(url).scheme == "https" and (
        host == "hf-mirror.com" or host.endswith(".hf-mirror.com")
        or (allow_mirror_cdn and host == "cas-bridge.xethub.hf.co")
    )


def get(session, url, *, allow_mirror_cdn=False, **kwargs):
    for _ in range(8):
        if not allowed_url(url, allow_mirror_cdn):
            raise RuntimeError(f"Unapproved redirect host: {urlparse(url).hostname}")
        r = session.get(url, allow_redirects=False, timeout=(30, 120), **kwargs)
        if not r.is_redirect:
            r.raise_for_status()
            return r
        next_url = urljoin(url, r.headers["Location"])
        r.close()
        url = next_url
    raise RuntimeError("Too many redirects")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def verify(path, item):
    size = item.get("size") or item.get("lfs", {}).get("size")
    expected = item.get("lfs", {}).get("sha256")
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"Size mismatch: {path.name}")
    digest = sha256(path)
    if expected and digest != expected:
        raise RuntimeError(f"SHA256 mismatch: {path.name}")
    if not expected:
        # HF stores non-LFS files as Git blobs, not SHA256 objects.
        body = path.read_bytes()
        git_hash = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        if item.get("blobId") and git_hash != item["blobId"]:
            raise RuntimeError(f"Git blob hash mismatch: {path.name}")
    return {"size": path.stat().st_size, "sha256": digest, "verified_upstream": bool(expected or item.get("blobId"))}


def download_ranges(url, partial, size, connections, allow_mirror_cdn, block_size=256 << 20):
    """Keep a downloaded prefix, fetch disjoint resumable ranges, then append in order."""
    prefix = partial.stat().st_size if partial.exists() else 0
    if prefix > size:
        raise RuntimeError("Partial file exceeds upstream size")
    if prefix == size:
        return
    directory = partial.with_name(partial.name + ".ranges")
    directory.mkdir(exist_ok=True)
    ranges = [(start, min(start + block_size, size) - 1)
              for start in range(prefix, size, block_size)]

    def fetch(bounds):
        start, end = bounds
        path = directory / f"{start}-{end}.partial"
        with requests.Session() as session:
            session.trust_env = False
            for attempt in range(5):
                offset = path.stat().st_size if path.exists() else 0
                if offset == end - start + 1:
                    return path
                if offset > end - start + 1:
                    raise RuntimeError("Oversized range partial")
                try:
                    with get(session, url, allow_mirror_cdn=allow_mirror_cdn, stream=True,
                             headers={"Range": f"bytes={start + offset}-{end}"}) as response:
                        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start + offset}-{end}/{size}":
                            raise RuntimeError("Server did not honor exact range; partials preserved")
                        with path.open("ab") as stream:
                            for chunk in response.iter_content(8 << 20):
                                stream.write(chunk)
                    if path.stat().st_size != end - start + 1:
                        raise requests.RequestException("Incomplete range")
                    print(f"range verified length: {start}-{end}", flush=True)
                    return path
                except requests.RequestException as error:
                    print(f"range retry {attempt + 1}: {start}-{end} ({type(error).__name__})", flush=True)
                    if attempt == 4:
                        raise RuntimeError("Range download failed; partials preserved") from None
                    time.sleep(2)

    with ThreadPoolExecutor(max_workers=connections) as pool:
        pieces = list(pool.map(fetch, ranges))
    if (partial.stat().st_size if partial.exists() else 0) != prefix:
        raise RuntimeError("Prefix changed during range download")
    with partial.open("ab") as destination:
        for piece in pieces:
            with piece.open("rb") as stream:
                shutil.copyfileobj(stream, destination, length=8 << 20)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=REPO, help="HF model repository to fetch through the mirror")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--allow-mirror-cdn", action="store_true", help="Requires explicit user approval; direct connection only")
    p.add_argument("--connections", type=int, default=1, help="Parallel resumable ranges for large files (1-16)")
    a = p.parse_args()
    repo = a.repo
    if not 1 <= a.connections <= 16:
        raise ValueError("connections must be between 1 and 16")
    if len(repo.split("/")) != 2 or any(p in ("", ".", "..") for p in repo.split("/")):
        raise ValueError("Expected owner/model repository name")
    if a.root is None:
        a.root = ROOT / "assets/openwam_ckpt/openwam_alpha" / repo.split("/")[1]
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    if not allowed_url(endpoint):
        raise RuntimeError("HF_ENDPOINT must be hf-mirror.com")
    a.root.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.trust_env = False  # Mirror/CDN direct only: never consume configured VPN proxy.
    inventory = a.root / "mirror_inventory.json"
    if inventory.exists():
        meta = json.loads(inventory.read_text())
        if meta.get("id", repo) != repo:
            raise ValueError("Existing inventory belongs to a different repository")
    else:
        meta = get(s, f"{endpoint}/api/models/{repo}?blobs=true").json()
        inventory.write_text(json.dumps(meta, indent=2))
    rev = meta["sha"]
    report = {"repo": repo, "revision": rev, "endpoint": endpoint,
              "proxy_used": False, "allow_mirror_cdn": a.allow_mirror_cdn, "files": {}}
    # Fetch all small reconstruction assets before the 24.8 GB weight file.
    for item in sorted(meta["siblings"], key=lambda x: x.get("size", 0)):
        name = item["rfilename"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid upstream filename")
        dst = a.root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            report["files"][name] = verify(dst, item)
            print(f"verified {name}", flush=True)
            continue
        partial = dst.with_name(dst.name + ".partial")
        if a.connections > 1 and item.get("size", 0) > (64 << 20):
            download_ranges(f"{endpoint}/{repo}/resolve/{rev}/{name}", partial, item["size"],
                            a.connections, a.allow_mirror_cdn)
            report["files"][name] = verify(partial, item)
            partial.replace(dst)
            print(f"downloaded and verified {name}", flush=True)
            (a.root / "download_verification.json").write_text(json.dumps(report, indent=2))
            continue
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                r = get(s, f"{endpoint}/{repo}/resolve/{rev}/{name}",
                        allow_mirror_cdn=a.allow_mirror_cdn, headers=headers, stream=True)
                with r:
                    if offset and (r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise RuntimeError("Server did not honor resume range; partial preserved")
                    with partial.open("ab" if offset else "wb") as f:
                        last = time.monotonic()
                        for chunk in r.iter_content(8 * 1024 * 1024):
                            f.write(chunk)
                            if time.monotonic() - last > 30:
                                print(f"{name}: {f.tell()} bytes", flush=True)
                                last = time.monotonic()
                record = verify(partial, item)
                partial.replace(dst)
                report["files"][name] = record
                print(f"downloaded and verified {name}", flush=True)
                break
            except requests.RequestException as e:
                print(f"retry {attempt + 1}: {name} ({type(e).__name__}); partial preserved", flush=True)
                if attempt == 4:
                    raise RuntimeError(f"Download failed: {name}") from None
                time.sleep(2)
        (a.root / "download_verification.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
