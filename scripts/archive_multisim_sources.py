#!/usr/bin/env python3
"""Archive immutable source trees as independently verified regular files."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import stat
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def files(source, ancestors=(), boundary=None, metadata_links=frozenset(), external_files=frozenset()):
    source = Path(source)
    if str(source) in metadata_links:
        return
    if str(source) in external_files:
        if not source.is_file():
            raise ValueError(f"Declared external file is not a file: {source}")
        yield source
        return
    resolved = source.resolve(strict=True)
    boundary = resolved if boundary is None else boundary
    if not resolved.is_relative_to(boundary):
        raise ValueError(f"External symlink needs an explicit dependency plan: {source}")
    if resolved in ancestors:
        raise ValueError(f"Source directory symlink cycle: {source}")
    if source.is_dir():
        for child in sorted(source.iterdir()):
            yield from files(child, (*ancestors, resolved), boundary, metadata_links, external_files)
    elif stat.S_ISREG(source.stat().st_mode):
        yield source
    else:
        raise ValueError(f"Non-regular archive source: {source}")


def target_path(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Archive path escapes root: {relative}")
    return path


def inventory(connection, root, plan):
    payload = Path(plan).read_bytes()
    identity = hashlib.sha256(payload).hexdigest()
    previous = connection.execute("SELECT value FROM meta WHERE name='plan_sha256'").fetchone()
    if previous and previous[0] != identity:
        raise ValueError("Archive plan changed; use another state database")
    if connection.execute("SELECT value FROM meta WHERE name='inventory_complete'").fetchone():
        return
    spec = json.loads(payload)
    descriptors = spec.get("metadata_only_links", [])
    file_links = spec.get("materialize_external_files", [])
    for item in descriptors + file_links:
        path = Path(item["source"])
        if not path.is_symlink() or os.readlink(path) != item["target"]:
            raise ValueError(f"Declared execution-view link changed: {path}")
    if descriptors:
        save(root / "provenance" / f"source_links_{identity[:12]}.json", descriptors)
    if file_links:
        save(root / "provenance" / f"materialized_links_{identity[:12]}.json", file_links)
    metadata_links = frozenset(x["source"] for x in descriptors)
    external_files = frozenset(x["source"] for x in file_links)
    connection.execute("INSERT OR REPLACE INTO meta VALUES ('plan_sha256', ?)", (identity,))
    count = 0
    for entry in spec["roots"]:
        root_key = "root_done:" + hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        if connection.execute("SELECT value FROM meta WHERE name=?", (root_key,)).fetchone():
            continue
        source = Path(entry["source"])
        if root.resolve().is_relative_to(source.resolve()):
            raise ValueError("The archive must not be placed inside its source tree")
        for path in files(source, metadata_links=metadata_links, external_files=external_files):
            relative = (
                str(Path(entry["destination"]) / path.relative_to(source)) if source.is_dir() else entry["destination"]
            )
            target_path(root, relative)
            info = path.stat()
            values = (str(path), relative, info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino)
            found = connection.execute(
                "SELECT source, bytes, mtime, device, inode FROM files WHERE target=?", (relative,)
            ).fetchone()
            if found and found != (values[0], *values[2:]):
                raise ValueError(f"Inventory source changed or destination collision: {path}")
            connection.execute(
                "INSERT OR IGNORE INTO files(source,target,bytes,mtime,device,inode) VALUES (?,?,?,?,?,?)", values
            )
            count += 1
            if count % 1000 == 0:
                connection.commit()
                print(json.dumps(dict(inventoried=count)), flush=True)
        connection.execute("INSERT OR REPLACE INTO meta VALUES (?, '1')", (root_key,))
        connection.commit()
    connection.execute("INSERT INTO meta VALUES ('inventory_complete','1')")
    connection.commit()


def independent_copy(root, row):
    identifier, source, relative, size, mtime, device, inode, expected, verified = row
    source, target = Path(source), target_path(root, relative)
    info = source.stat()
    if (info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino) != (size, mtime, device, inode):
        raise ValueError(f"Source changed since inventory: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        other = target.stat()
        if target.is_symlink() or other.st_nlink != 1 or (other.st_dev, other.st_ino) == (device, inode):
            raise ValueError(f"Archive must contain independent regular files: {target}")
        if verified and other.st_size == size and other.st_mtime_ns == mtime:
            return identifier, expected, size
        digest = sha(source)
        if other.st_size != size or sha(target) != digest:
            raise ValueError(f"Unverified destination collision: {target}")
        os.utime(target, ns=(info.st_atime_ns, mtime))
        return identifier, digest, size
    temporary = root / "_archive_state/partials" / (hashlib.sha256(relative.encode()).hexdigest() + ".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists() or temporary.is_symlink():
        partial = temporary.stat()
        if temporary.is_symlink() or partial.st_nlink != 1 or (partial.st_dev, partial.st_ino) == (device, inode):
            raise ValueError(f"Unsafe archive partial: {temporary}")
    digest = hashlib.sha256()
    with source.open("rb") as src, temporary.open("wb") as dst:
        for chunk in iter(lambda: src.read(8 * 1024**2), b""):
            digest.update(chunk)
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    after = source.stat()
    if (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino) != (size, mtime, device, inode):
        raise ValueError(f"Source changed while copying: {source}")
    if sha(temporary) != digest.hexdigest():
        raise ValueError(f"Archive readback hash mismatch: {target}")
    os.chmod(temporary, stat.S_IMODE(info.st_mode))
    os.utime(temporary, ns=(info.st_atime_ns, mtime))
    temporary.replace(target)
    return identifier, digest.hexdigest(), size


def run(root, plan, state, mode, workers):
    root, state = Path(root).resolve(), Path(state).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state.parent.mkdir(parents=True, exist_ok=True)
    lock = state.with_name(state.name + ".lock")
    lock.mkdir()  # Atomic directory creation excludes writers across GPFS nodes too.
    save(lock / "owner.json", dict(pid=os.getpid(), host=os.uname().nodename))
    connection = sqlite3.connect(state)
    try:
        connection.executescript(
            "CREATE TABLE IF NOT EXISTS meta(name TEXT PRIMARY KEY,value TEXT); CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY,source TEXT,target TEXT UNIQUE,bytes INTEGER,mtime INTEGER,device INTEGER,inode INTEGER,sha256 TEXT,verified INTEGER DEFAULT 0);"
        )
        inventory(connection, root, plan)
        total, total_bytes = connection.execute("SELECT count(*),coalesce(sum(bytes),0) FROM files").fetchone()
        progress = dict(
            mode=mode, files=total, bytes=total_bytes, source_roots_complete=False, cleanup_authorized=False
        )
        save(state.with_suffix(".progress.json"), progress)
        if mode == "inventory":
            print(json.dumps(progress), flush=True)
            return
        last, completed, copied_bytes = 0, 0, 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                batch = connection.execute("SELECT * FROM files WHERE id>? ORDER BY id LIMIT 1000", (last,)).fetchall()
                if not batch:
                    break
                last = batch[-1][0]
                if mode == "copy":
                    pending = [pool.submit(independent_copy, root, row) for row in batch]
                    for future in concurrent.futures.as_completed(pending):
                        identifier, digest, size = future.result()
                        connection.execute("UPDATE files SET sha256=?,verified=1 WHERE id=?", (digest, identifier))
                        completed += 1
                        copied_bytes += size
                        if completed % 100 == 0:
                            connection.commit()
                            progress.update(completed=completed, verified_bytes=copied_bytes, time=time.time())
                            save(state.with_suffix(".progress.json"), progress)
                else:
                    for row in batch:
                        _, _, relative, size, _, device, inode, expected, verified = row
                        target = target_path(root, relative)
                        info = target.stat()
                        if (
                            not verified
                            or target.is_symlink()
                            or info.st_nlink != 1
                            or (info.st_dev, info.st_ino) == (device, inode)
                            or info.st_size != size
                            or sha(target) != expected
                        ):
                            raise ValueError(f"Independent archive audit failed: {relative}")
                        completed += 1
                        copied_bytes += size
                connection.commit()
        progress.update(
            completed=completed, verified_bytes=copied_bytes, source_roots_complete=completed == total, time=time.time()
        )
        save(state.with_suffix(".progress.json"), progress)
        print(json.dumps(progress), flush=True)
    finally:
        connection.close()
        (lock / "owner.json").unlink()
        lock.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--mode", choices=("inventory", "copy", "verify"), default="inventory")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.root, args.plan, args.state, args.mode, args.workers)
