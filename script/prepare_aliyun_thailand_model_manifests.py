#!/usr/bin/env python3
"""Build hash-checked ModelScope manifests for the pinned LynnReal snapshots."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import uuid
from urllib.parse import quote, urlencode
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


MODELS = {
    "standard": (
        "stdstu123/LynnReal-Onmi-beta-0.1",
        "f1d6990e23496bef6c7d55bdcfe2925df2508325",
        "AI-ModelScope/LynnReal-Onmi-beta-0.1",
    ),
    "flash": (
        "stdstu123/LynnReal-Onmi-flash-beta-0.1",
        "9950cfc882b0e6b8600fbc49b58dc1d82d7a435b",
        "AI-ModelScope/LynnReal-Onmi-flash-beta-0.1",
    ),
    "light-vae": (
        "stdstu123/LynnReal-Onmi-light-vae",
        "e453444c1a52b73a0c5eb0023d208c473c2bb26a",
        "AI-ModelScope/LynnReal-Onmi-light-vae",
    ),
}
ROOT_METADATA = {
    "LICENSE", "NOTICE", "README.md", "inference_config.json",
    "model_index.json", "modular_model_index.json",
}
LARGE_OBJECT_BYTES = 1024 * 1024


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def get_json(url: str) -> object:
    for attempt in range(1, 6):
        try:
            request = Request(url, headers={"User-Agent": "LynnReal-Omni-manifest-builder"})
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except OSError:
            if attempt == 5:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")


def hf_tree(repo: str, revision: str) -> list[dict]:
    error = None
    for origin in ("https://huggingface.co", "https://hf-mirror.com"):
        url = f"{origin}/api/models/{repo}/tree/{revision}?recursive=true&limit=1000"
        try:
            entries = get_json(url)
            assert isinstance(entries, list)
            return entries
        except OSError as exc:
            error = exc
    assert error is not None
    raise error


def modelscope_files(repo: str, root: str = "") -> list[dict]:
    query = urlencode({"Revision": "master", "Root": root})
    payload = get_json(f"https://modelscope.cn/api/v1/models/{repo}/repo/files?{query}")
    assert payload["Success"], payload
    return payload["Data"]["Files"]


def sha256_url(url: str) -> str:
    urls = [url]
    if url.startswith("https://huggingface.co/"):
        urls.append(url.replace("https://huggingface.co/", "https://hf-mirror.com/", 1))
    error = None
    for candidate in urls:
        for attempt in range(1, 4):
            try:
                digest = hashlib.sha256()
                request = Request(
                    candidate,
                    headers={"User-Agent": "LynnReal-Omni-manifest-builder"},
                )
                with urlopen(request, timeout=60) as response:
                    for block in iter(lambda: response.read(4 << 20), b""):
                        digest.update(block)
                return digest.hexdigest()
            except OSError as exc:
                error = exc
                if attempt < 3:
                    time.sleep(attempt)
    assert error is not None
    raise error


def modelscope_object_url(repo: str, path: str) -> str:
    query = urlencode({"Revision": "master", "FilePath": path})
    return f"https://modelscope.cn/api/v1/models/{repo}/repo?{query}"


def modelscope_small_object(repo: str, path: str) -> bytes:
    for attempt in range(1, 6):
        try:
            request = Request(
                modelscope_object_url(repo, path),
                headers={"User-Agent": "LynnReal-Omni-artifact-builder"},
            )
            with urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 5:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")


def modelscope_signed_url(repo: str, path: str) -> tuple[str, str]:
    opener = build_opener(NoRedirect)
    for attempt in range(1, 11):
        try:
            request = Request(
                modelscope_object_url(repo, path),
                headers={
                    "User-Agent": "LynnReal-Omni-artifact-builder",
                    "x-aliyun-region-id": "cn-hangzhou",
                    "snapshot-identifier": uuid.uuid4().hex,
                },
            )
            try:
                response = opener.open(request, timeout=60)
            except HTTPError as error:
                if error.code not in {301, 302, 303, 307, 308}:
                    raise
                location = error.headers.get("Location")
            else:
                location = response.headers.get("Location")
                response.close()
            if location and "oss-cn-hangzhou-internal.aliyuncs.com" in location:
                public = location.replace(
                    "oss-cn-hangzhou-internal.aliyuncs.com",
                    "oss-cn-hangzhou.aliyuncs.com",
                )
                return f"{repo}:{path}", public
        except OSError:
            if attempt == 10:
                raise
        time.sleep(min(attempt, 3))
    raise RuntimeError(f"cannot sign {repo}:{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-signed-urls", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    large_objects: list[tuple[str, str]] = []

    for kind, (hf_repo, hf_revision, ms_repo) in MODELS.items():
        tree = hf_tree(hf_repo, hf_revision)
        hf_files = {entry["path"]: entry for entry in tree if entry["type"] == "file"}
        ms_files = modelscope_files(ms_repo)
        if kind != "light-vae":
            ms_files += modelscope_files(ms_repo, "transformer")
            if kind == "standard":
                ms_files += modelscope_files(ms_repo, "audio_vae")
                ms_files += modelscope_files(ms_repo, "vae")
            required = [
                entry
                for entry in ms_files
                if entry["Type"] == "blob"
                and (
                    entry["Path"] in ROOT_METADATA
                    or entry["Path"].startswith("transformer/")
                    or (
                        kind == "standard"
                        and entry["Path"].startswith(("audio_vae/", "vae/"))
                    )
                )
            ]
        else:
            required = [entry for entry in ms_files if entry["Type"] == "blob" and
                        (entry["Path"] in {"config.json", "decode_config.json"} or
                         entry["Path"].startswith("diffusion_pytorch_model"))]

        rows = []
        for entry in sorted(required, key=lambda item: item["Path"]):
            path = entry["Path"]
            expected = hf_files[path]
            assert entry["Size"] == expected["size"], (kind, path, entry["Size"], expected["size"])
            expected_sha = (expected.get("lfs") or {}).get("oid")
            if not expected_sha:
                raw = f"https://huggingface.co/{hf_repo}/resolve/{hf_revision}/{quote(path)}"
                expected_sha = sha256_url(raw)
            assert entry["Sha256"] == expected_sha, (kind, path, entry["Sha256"], expected_sha)
            rows.append((path, entry["Size"], entry["Sha256"]))

        manifest = args.output / f"{kind}.tsv"
        manifest.write_text("path\tsize\tsha256\n" + "".join(f"{path}\t{size}\t{digest}\n" for path, size, digest in rows))
        seed_root = args.output / "model-seed" / kind
        for path, size, digest in rows:
            if size >= LARGE_OBJECT_BYTES:
                large_objects.append((ms_repo, path))
                continue
            content = modelscope_small_object(ms_repo, path)
            assert len(content) == size, (kind, path, len(content), size)
            assert hashlib.sha256(content).hexdigest() == digest, (kind, path)
            destination = seed_root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        if kind == "standard":
            (args.output / "standard-hf-tree.json").write_text(json.dumps(tree, separators=(",", ":")))
        print({"kind": kind, "files": len(rows), "bytes": sum(row[1] for row in rows)})

    if args.include_signed_urls:
        signed_urls: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
            futures = [
                executor.submit(modelscope_signed_url, repo, path)
                for repo, path in large_objects
            ]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                key, url = future.result()
                signed_urls[key] = url
                print(f"signed={index}/{len(futures)}", flush=True)
        signed_path = args.output / "modelscope-signed-urls.json"
        signed_path.write_text(json.dumps(signed_urls, separators=(",", ":")))
        signed_path.chmod(0o600)


if __name__ == "__main__":
    main()
