#!/usr/bin/env python3
"""Run a JSONL T2V evaluation set with one resident model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def frame_count(value: str) -> int:
    """Parse seconds (10 or 10s) or an exact frame count (240f) at 24 fps."""
    try:
        count = float(value[:-1]) if value[-1:] in {"s", "f"} else float(value)
        if not math.isfinite(count) or count <= 0:
            raise ValueError
        frames = count if value.endswith("f") else count * 24
        if not frames.is_integer():
            raise ValueError
        return int(frames)
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError(
            "use seconds (10s or 10) or an integer frame count (240f), at 24 fps"
        )


def resolution(value: str) -> tuple[int, int]:
    presets = {
        "480p": (832, 480),
        "540p": (960, 540),
        "720p": (1280, 720),
        "768p": (1344, 768),
        "1080p": (1920, 1080),
    }
    if value.lower() in presets:
        return presets[value.lower()]
    if re.fullmatch(r"[0-9]+[xX][0-9]+", value):
        width, height = map(int, value.lower().split("x"))
        if min(width, height) > 0 and width % 2 == height % 2 == 0:
            return width, height
    raise argparse.ArgumentTypeError(
        "use 768p or even WIDTHxHEIGHT, e.g. 1344x768"
    )


def load_requests(path: Path) -> list[dict]:
    records = []
    seen = set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"line {line_number}: each request must be an object")
            sample_id = raw.get("id")
            prompt = raw.get("prompt")
            seed = raw.get("seed", 0)
            if not isinstance(sample_id, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", sample_id
            ):
                raise ValueError(
                    f"line {line_number}: id must be 1..128 safe filename characters"
                )
            if sample_id in seen:
                raise ValueError(f"line {line_number}: duplicate id {sample_id!r}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"line {line_number}: prompt must be a non-empty string")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError(f"line {line_number}: seed must be a nonnegative integer")
            seen.add(sample_id)
            records.append({"id": sample_id, "prompt": prompt.strip(), "seed": seed})
    if not records:
        raise ValueError("manifest contains no requests")
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", choices=("standard", "flash"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=frame_count, default="10s")
    parser.add_argument("--resolution", type=resolution, default="768p")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--attention-backend",
        choices=("native", "_native_cudnn"),
        default="native",
        help="native is the measured B300 winner; use another backend only for an explicit ablation",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        manifest = args.manifest.expanduser().resolve(strict=True)
        requests = load_requests(manifest)
        if args.warmups < 0:
            raise ValueError("warmups must be nonnegative")
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    width, height = args.resolution
    canvas_height = ((height + 31) // 32) * 32
    plan = {
        "schema": "lynnreal-t2v-eval-v1",
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "variant": args.variant,
        "samples": len(requests),
        "geometry": [width, height, args.frame],
        "fps": 24,
        "warmups": args.warmups,
        "attention_backend": args.attention_backend,
        "int8_gemm": "triton",
        "vae_tiles": "adaptive",
        "output": str(args.output.resolve()),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    import torch
    from model.acceleration import fastest_available
    from model.output import encode_video
    from model.pipeline import Pipeline, encode_conditioning, release_memory
    from model.provenance import changed_sources, snapshot_sources

    weights = ROOT / "weight" / args.variant
    light_vae = ROOT / "weight" / "light-vae"
    cache = ROOT / "output" / ".cache"
    output = args.output.resolve()
    samples_dir = output / "samples"
    metadata_dir = output / "metadata"
    output.mkdir(parents=True, exist_ok=False)
    samples_dir.mkdir()
    metadata_dir.mkdir()
    write_json(output / "run.json", plan)
    with (output / "requests.jsonl").open("w") as stream:
        for request in requests:
            stream.write(json.dumps(request, ensure_ascii=False) + "\n")

    source = snapshot_sources(cache)
    acceleration = fastest_available(args.attention_backend)
    if not acceleration["fused"]:
        raise RuntimeError("B300 eval requires the validated Triton fusion path")

    # Populate the content-addressed conditioning cache while retaining one CPU
    # text encoder. Release it before loading the resident DiT and VAE.
    resident: dict = {}
    conditioning_records = {}
    for index, request in enumerate(requests, 1):
        condition, timing = encode_conditioning(
            weights,
            request["prompt"],
            [],
            canvas_height,
            width,
            args.frame,
            cache,
            resident=resident,
            resident_device="cpu",
        )
        conditioning_records[request["id"]] = timing
        print(
            json.dumps(
                {"phase": "conditioning", "index": index, "total": len(requests),
                 "id": request["id"], **timing},
                ensure_ascii=False,
            ),
            flush=True,
        )
        del condition
    resident.clear()
    del resident
    gc.collect()

    loaded_at = time.perf_counter()
    pipe = Pipeline(
        weights,
        int8=True,
        fused=True,
        light_vae=light_vae,
        compile_vae=True,
        adaln_cache=True,
        int8_gemm="triton",
        attention_backend=args.attention_backend,
        vae_attention=args.attention_backend,
    )
    pipe.pipe.vae.tile_layout = "adaptive"
    pipe.vae_record["tile_layout"] = "adaptive"
    load_seconds = time.perf_counter() - loaded_at
    steps = 4 if args.variant == "standard" else 3

    first = requests[0]
    first_condition, _ = encode_conditioning(
        weights, first["prompt"], [], canvas_height, width, args.frame, cache
    )
    for index in range(args.warmups):
        warmup_state, timing = pipe.generate(
            first_condition, [], canvas_height, width, args.frame, first["seed"], steps
        )
        print(
            json.dumps({"phase": "warmup", "iteration": index, **timing}),
            flush=True,
        )
        del warmup_state
    del first_condition
    release_memory()

    progress_path = output / "progress.jsonl"
    results = []
    with progress_path.open("w", buffering=1) as progress:
        for index, request in enumerate(requests, 1):
            condition, cache_timing = encode_conditioning(
                weights,
                request["prompt"],
                [],
                canvas_height,
                width,
                args.frame,
                cache,
            )
            if not cache_timing["cache_hit"]:
                raise RuntimeError(f"conditioning cache unexpectedly missed for {request['id']}")
            state, timing = pipe.generate(
                condition, [], canvas_height, width, args.frame, request["seed"], steps
            )
            frames = state["videos"][0]
            if len(frames) < args.frame:
                raise RuntimeError(
                    f"decoder returned {len(frames)} frames for {request['id']}, expected {args.frame}"
                )
            top = (canvas_height - height) // 2
            frames = [frame.crop((0, top, width, top + height)) for frame in frames[: args.frame]]
            sample_rate = int(state["sampling_rate"])
            audio = state["audio"][0][..., : round(args.frame * sample_rate / 24)]
            video_path = samples_dir / f"{request['id']}.mp4"
            encode_video(
                frames,
                fps=24,
                output_path=str(video_path),
                audio=audio,
                audio_sample_rate=sample_rate,
            )
            record = {
                **request,
                "video": str(video_path),
                "video_sha256": sha256(video_path),
                "variant": args.variant,
                "precision": "W8A8",
                "int8_gemm": "triton",
                "attention_backend": args.attention_backend,
                "vae_tiles": "adaptive",
                "steps": steps,
                "width": width,
                "height": height,
                "frames": args.frame,
                "fps": 24,
                "seconds": args.frame / 24,
                "timing": timing,
                "conditioning": conditioning_records[request["id"]],
            }
            write_json(metadata_dir / f"{request['id']}.json", record)
            progress.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)
            print(
                json.dumps(
                    {"phase": "sample", "index": index, "total": len(requests), **record},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del condition, state, frames, audio
            release_memory()

    summary = {
        **plan,
        "status": "passed",
        "load_seconds": load_seconds,
        "acceleration": acceleration,
        "transformer": pipe.transformer_record,
        "quantization": pipe.quantization,
        "light_vae": pipe.vae_record,
        "source": source,
        "source_changed_during_run": changed_sources(source),
        "model_loads": 1,
        "completed_samples": len(results),
    }
    progress_path.replace(output / "results.jsonl")
    write_json(output / "summary.json", summary)
    (output / "READY").write_text("ready\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
