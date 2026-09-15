#!/usr/bin/env python3
"""Download ModelScope objects through a signed Hangzhou OSS public URL.

This is a fallback for Shanghai PAI nodes when ModelScope's Shanghai request
is redirected to a slow CDN. The signed URL remains in memory and is never
logged. Downloads are range-parallel, resumable by completed-range markers,
and checked against the supplied SHA256 manifest before being published.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from urllib.parse import quote_plus

import requests


API_TEMPLATE = "https://modelscope.cn/api/v1/models/{repo}/repo"
DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024


def read_manifest(path: Path) -> list[tuple[str, int, str]]:
    objects: list[tuple[str, int, str]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or line == "path\tsize\tsha256":
            continue
        object_path, size, sha256 = line.split("\t")
        objects.append((object_path, int(size), sha256))
    return objects


def signed_public_url(
    repo: str,
    revision: str,
    object_path: str,
    signed_url_map: Path | None,
) -> str:
    if signed_url_map:
        urls = json.loads(signed_url_map.read_text())
        key = f"{repo}:{object_path}"
        if key in urls:
            return urls[key]

    api_url = API_TEMPLATE.format(repo=repo)
    params = f"Revision={quote_plus(revision)}&FilePath={quote_plus(object_path)}"
    session = requests.Session()
    for attempt in range(1, 31):
        try:
            response = session.get(
                f"{api_url}?{params}",
                headers={
                    "x-aliyun-region-id": "cn-hangzhou",
                    "snapshot-identifier": uuid.uuid4().hex,
                },
                allow_redirects=False,
                timeout=(10, 30),
            )
            location = response.headers.get("location")
            if location and "oss-cn-hangzhou-internal.aliyuncs.com" in location:
                public = location.replace(
                    "oss-cn-hangzhou-internal.aliyuncs.com",
                    "oss-cn-hangzhou.aliyuncs.com",
                )
                if public.startswith("http://"):
                    public = "https://" + public.removeprefix("http://")
                return public
        except requests.RequestException as exc:
            print(
                f"sign_retry path={object_path} attempt={attempt} "
                f"error={type(exc).__name__}",
                flush=True,
            )
        time.sleep(min(attempt, 5))
    raise RuntimeError(f"cannot obtain signed URL for {object_path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_range(
    url: str,
    fd: int,
    start: int,
    end: int,
    marker: Path,
) -> int:
    if marker.exists():
        return end - start + 1
    expected = end - start + 1
    for attempt in range(1, 11):
        try:
            response = requests.get(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "X-Request-ID": uuid.uuid4().hex,
                },
                stream=True,
                timeout=(10, 120),
            )
            if response.status_code != 206:
                raise RuntimeError(f"unexpected HTTP {response.status_code}")
            offset = start
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    os.pwrite(fd, chunk, offset)
                    offset += len(chunk)
            if offset - start != expected:
                raise RuntimeError(
                    f"short range: expected {expected}, received {offset - start}"
                )
            marker.touch()
            return expected
        except (requests.RequestException, RuntimeError) as exc:
            if attempt == 10:
                raise
            print(
                f"range_retry start={start} attempt={attempt} "
                f"error={type(exc).__name__}",
                flush=True,
            )
            time.sleep(min(attempt, 5))
    raise AssertionError("unreachable")


def download_object(
    repo: str,
    revision: str,
    output_root: Path,
    object_path: str,
    size: int,
    expected_sha256: str,
    workers: int,
    chunk_size: int,
    signed_url_map: Path | None,
) -> None:
    destination = output_root / object_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size:
        if sha256_file(destination) == expected_sha256:
            print(f"verified_existing path={object_path} bytes={size}", flush=True)
            return

    if size < chunk_size:
        chunk_size = size or 1
    temporary = destination.with_name(destination.name + ".multipart")
    markers = destination.with_name(destination.name + ".ranges")
    markers.mkdir(parents=True, exist_ok=True)
    fd = os.open(temporary, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    ranges = [
        (start, min(start + chunk_size, size) - 1)
        for start in range(0, size, chunk_size)
    ]
    url = signed_public_url(repo, revision, object_path, signed_url_map)
    started = time.monotonic()
    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    download_range,
                    url,
                    fd,
                    start,
                    end,
                    markers / f"{start}-{end}",
                ): (start, end)
                for start, end in ranges
            }
            for future in concurrent.futures.as_completed(futures):
                completed += future.result()
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"progress path={object_path} done={completed}/{size} "
                    f"speed={completed / elapsed / 1024 / 1024:.2f}MiB/s",
                    flush=True,
                )
    finally:
        os.close(fd)

    actual_sha256 = sha256_file(temporary)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {object_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    os.replace(temporary, destination)
    for marker in markers.iterdir():
        marker.unlink()
    markers.rmdir()
    elapsed = max(time.monotonic() - started, 0.001)
    print(
        f"complete path={object_path} bytes={size} "
        f"speed={size / elapsed / 1024 / 1024:.2f}MiB/s sha256={actual_sha256}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--signed-url-map", type=Path)
    args = parser.parse_args()

    objects = read_manifest(args.manifest)
    if args.prefix:
        objects = [item for item in objects if item[0].startswith(args.prefix)]
    total = sum(item[1] for item in objects)
    print(f"plan objects={len(objects)} bytes={total}", flush=True)
    for object_path, size, sha256 in objects:
        download_object(
            args.repo,
            args.revision,
            args.output,
            object_path,
            size,
            sha256,
            args.workers,
            args.chunk_mib * 1024 * 1024,
            args.signed_url_map,
        )
    print(f"all_complete objects={len(objects)} bytes={total}", flush=True)


if __name__ == "__main__":
    main()
