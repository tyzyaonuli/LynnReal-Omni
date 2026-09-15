#!/usr/bin/env python3
"""Copy hash-verified staged LynnReal model objects from Thailand OSS to CPFS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


MODELS = {
    "standard": "f1d6990e23496bef6c7d55bdcfe2925df2508325",
    "flash": "9950cfc882b0e6b8600fbc49b58dc1d82d7a435b",
    "light-vae": "e453444c1a52b73a0c5eb0023d208c473c2bb26a",
}


def read_manifest(path: Path) -> list[tuple[str, int, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("path\t"):
            continue
        object_path, size, digest = line.split("\t")
        rows.append((object_path, int(size), digest))
    return rows


def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            sha256.update(block)
    return sha256.hexdigest()


def sync_object(source: Path, destination: Path, size: int, digest: str) -> None:
    if destination.is_file() and destination.stat().st_size == size:
        if sha256_file(destination) == digest:
            print(f"verified_cpfs path={destination} bytes={size}", flush=True)
            return
    if not source.is_file() or source.stat().st_size != size:
        raise RuntimeError(f"missing or truncated staged object: {source}")
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
        raise RuntimeError(f"sync verification failed for {source}")
    os.replace(partial, destination)
    print(f"synced path={destination} bytes={size} sha256={digest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()

    for name, revision in MODELS.items():
        source_root = args.stage_root / f"{name}-{revision}"
        source_marker = source_root / f".complete-{revision}"
        if not source_marker.is_file():
            raise RuntimeError(f"stage is incomplete: {source_marker}")
        destination_root = args.model_root / f"{name}-{revision}"
        destination_marker = destination_root / ".downloaded-modelscope-master"
        marker_payload = (
            json.loads(destination_marker.read_text())
            if destination_marker.is_file()
            else {}
        )
        objects = read_manifest(args.manifest_dir / f"{name}.tsv")
        object_count = len(objects)
        total_bytes = sum(item[1] for item in objects)
        if (
            marker_payload.get("objects") == object_count
            and marker_payload.get("bytes") == total_bytes
        ):
            print(f"cpfs_complete_existing name={name} marker={destination_marker}", flush=True)
            continue
        standard_legacy_complete = (
            name == "standard"
            and marker_payload.get("objects") == 43
            and marker_payload.get("bytes") == 66280591474
        )
        for path, size, digest in objects:
            if standard_legacy_complete and not path.startswith(("audio_vae/", "vae/")):
                continue
            sync_object(source_root / path, destination_root / path, size, digest)
        destination_marker.parent.mkdir(parents=True, exist_ok=True)
        destination_marker.write_text(
            json.dumps(
                {
                    "revision": revision,
                    "source_marker": str(source_marker),
                    "objects": object_count,
                    "bytes": total_bytes,
                }
            )
            + "\n"
        )
        print(f"cpfs_complete name={name} marker={destination_marker}", flush=True)


if __name__ == "__main__":
    main()
