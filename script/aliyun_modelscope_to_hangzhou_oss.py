#!/usr/bin/env python3
"""Stage pinned LynnReal model objects into a same-region Hangzhou OSS mount."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from urllib.parse import quote_plus

import requests

import modelscope_hz_multipart_download as downloader


MODELS = {
    "standard": (
        "f1d6990e23496bef6c7d55bdcfe2925df2508325",
        "AI-ModelScope/LynnReal-Onmi-beta-0.1",
    ),
    "flash": (
        "9950cfc882b0e6b8600fbc49b58dc1d82d7a435b",
        "AI-ModelScope/LynnReal-Onmi-flash-beta-0.1",
    ),
    "light-vae": (
        "e453444c1a52b73a0c5eb0023d208c473c2bb26a",
        "AI-ModelScope/LynnReal-Onmi-light-vae",
    ),
}
LARGE_OBJECT_BYTES = 1024 * 1024


def internal_signed_url(
    repo: str,
    revision: str,
    object_path: str,
    _signed_url_map: Path | None,
) -> str:
    api = f"https://modelscope.cn/api/v1/models/{repo}/repo"
    params = f"Revision={quote_plus(revision)}&FilePath={quote_plus(object_path)}"
    session = requests.Session()
    for attempt in range(1, 31):
        try:
            response = session.get(
                f"{api}?{params}",
                headers={
                    "x-aliyun-region-id": "cn-hangzhou",
                    "snapshot-identifier": uuid.uuid4().hex,
                },
                allow_redirects=False,
                timeout=(10, 30),
            )
            location = response.headers.get("location")
            if location and "oss-cn-hangzhou-internal.aliyuncs.com" in location:
                return location
        except requests.RequestException as error:
            print(
                f"sign_retry path={object_path} attempt={attempt} "
                f"error={type(error).__name__}",
                flush=True,
            )
        time.sleep(min(attempt, 5))
    raise RuntimeError(f"cannot obtain internal signed URL for {object_path}")


def copy_verified(source: Path, destination: Path, size: int, digest: str) -> None:
    if destination.is_file() and destination.stat().st_size == size:
        if downloader.sha256_file(destination) == digest:
            print(f"verified_staged path={destination} bytes={size}", flush=True)
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(
        destination.name + f".partial.{os.environ.get('DLC_JOB_ID', os.getpid())}"
    )
    sha256 = hashlib.sha256()
    copied = 0
    with source.open("rb") as reader, partial.open("wb") as writer:
        while block := reader.read(16 * 1024 * 1024):
            writer.write(block)
            sha256.update(block)
            copied += len(block)
    if copied != size or sha256.hexdigest() != digest:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"publish verification failed for {destination}")
    os.replace(partial, destination)
    print(f"published path={destination} bytes={size} sha256={digest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/local/lynnreal-model-stage"))
    args = parser.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    downloader.signed_public_url = internal_signed_url

    for name, (revision, repo) in MODELS.items():
        manifest = args.manifest_dir / f"{name}.tsv"
        objects = downloader.read_manifest(manifest)
        destination_root = args.stage_root / f"{name}-{revision}"
        marker = destination_root / f".complete-{revision}"
        marker_payload = json.loads(marker.read_text()) if marker.is_file() else {}
        object_count = len(objects)
        total_bytes = sum(item[1] for item in objects)
        if (
            marker_payload.get("objects") == object_count
            and marker_payload.get("bytes") == total_bytes
        ):
            print(f"stage_complete_existing name={name} marker={marker}", flush=True)
            continue
        standard_legacy_complete = (
            name == "standard"
            and marker_payload.get("objects") == 43
            and marker_payload.get("bytes") == 66280591474
        )
        for path, size, digest in objects:
            if standard_legacy_complete and not path.startswith(("audio_vae/", "vae/")):
                continue
            destination = destination_root / path
            if size < LARGE_OBJECT_BYTES:
                source = args.manifest_dir / "model-seed" / name / path
                copy_verified(source, destination, size, digest)
                continue
            local_root = args.work_root / name
            downloader.download_object(
                repo,
                "master",
                local_root,
                path,
                size,
                digest,
                16,
                64 * 1024 * 1024,
                None,
            )
            source = local_root / path
            copy_verified(source, destination, size, digest)
            source.unlink()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "revision": revision,
                    "repo": repo,
                    "objects": object_count,
                    "bytes": total_bytes,
                }
            )
            + "\n"
        )
        print(f"stage_complete name={name} marker={marker}", flush=True)


if __name__ == "__main__":
    main()
