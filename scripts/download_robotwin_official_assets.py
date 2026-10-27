"""Download pinned RoboTwin assets through HF mirror, verify, extract and compare."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlparse

import requests


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--compare", type=Path, required=True)
    args = p.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    args.records.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    endpoint = "https://hf-mirror.com"
    info = session.get(endpoint + "/api/datasets/TianxingChen/RoboTwin2.0", timeout=45)
    info.raise_for_status()
    revision = info.json()["sha"]
    response = session.get(endpoint + f"/api/datasets/TianxingChen/RoboTwin2.0/tree/{revision}", timeout=45)
    response.raise_for_status()
    inventory = response.json()
    (args.records / "pinned_inventory.json").write_text(json.dumps({"revision": revision, "files": inventory}, indent=2))

    def fetch(entry):
        name, size, expected = entry["path"], entry["size"], entry["lfs"]["oid"]
        final = args.assets / name
        partial = args.assets / (name + ".partial")
        if not final.exists():
            for attempt in range(8):
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    if offset == size:
                        break
                    if offset > size:
                        raise RuntimeError("partial exceeds upstream size; retained for inspection")
                    url = endpoint + f"/datasets/TianxingChen/RoboTwin2.0/resolve/{revision}/{name}"
                    # Mirror-issued CAS redirect was explicitly approved in this project's
                    # 2026-10-13 download authorization. No proxy, official API, or fallback.
                    with requests.Session() as connection:
                        connection.trust_env = False
                        for _ in range(8):
                            host = urlparse(url).hostname
                            if host not in {"hf-mirror.com", "cas-bridge.xethub.hf.co"}:
                                raise RuntimeError(f"unapproved redirect host {host}; no fallback")
                            res = connection.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                                                 stream=True, allow_redirects=False, timeout=(30, 120))
                            if res.is_redirect:
                                url = urljoin(url, res.headers["Location"])
                                res.close()
                                continue
                            res.raise_for_status()
                            if offset and (res.status_code != 206 or not res.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                                raise RuntimeError("resume range rejected; partial retained")
                            last = offset
                            with partial.open("ab" if offset else "wb") as f:
                                for block in res.iter_content(8 << 20):
                                    f.write(block)
                                    offset += len(block)
                                    if offset - last >= 256 << 20:
                                        print(f"{name}: {offset}/{size} bytes", flush=True)
                                        last = offset
                            res.close()
                            break
                        else:
                            raise RuntimeError("too many redirects")
                    if partial.stat().st_size == size:
                        break
                except requests.RequestException as exc:
                    # Avoid printing signed URLs from exceptions.
                    print(f"{name}: network {type(exc).__name__}, resume attempt {attempt + 1}", flush=True)
                    time.sleep(min(10, attempt + 1))
            if not partial.exists() or partial.stat().st_size != size or digest(partial) != expected:
                raise RuntimeError(f"{name}: size/hash mismatch or incomplete; partial retained")
            partial.rename(final)
        if final.stat().st_size != size or digest(final) != expected:
            raise RuntimeError(f"{name}: existing archive does not match pinned upstream")
        print(f"{name}: pinned size/SHA256 verified", flush=True)
        manifest = []
        with zipfile.ZipFile(final) as archive:
            for member in archive.infolist():
                # Validate archive names lexically. Resolving every path causes
                # hundreds of thousands of metadata RPCs on the shared disk.
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise RuntimeError("unsafe archive path")
            archive.extractall(args.assets)
            for member in archive.infolist():
                if member.is_dir():
                    continue
                path = args.assets / member.filename
                actual = digest(path)
                old = args.compare / member.filename
                manifest.append({"path": member.filename, "size": path.stat().st_size, "sha256": actual,
                                 "previous_asset": "same" if old.is_file() and digest(old) == actual else "different_or_missing"})
        record = {"revision": revision, "archive": name, "size": size, "sha256": expected, "files": manifest}
        (args.records / (name + ".manifest.json")).write_text(json.dumps(record, indent=2))
        print(f"{name}: extracted {len(manifest)} files; previous identical={sum(v['previous_asset'] == 'same' for v in manifest)}", flush=True)

    names = {"embodiments.zip", "objects.zip", "background_texture.zip"}
    entries = [v for v in inventory if v["path"] in names]
    if len(entries) != 3:
        raise RuntimeError("mirror inventory missing required asset archives")
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(fetch, entries))


if __name__ == "__main__":
    main()
