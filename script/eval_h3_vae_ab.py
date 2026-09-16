"""Official H3 BF16 sampling once, then two sequential decoders on one GPU."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def light_weight_view(path):
    # The release loader requires a directory inside weight/. Link files, not the directory.
    view = ROOT / "weight" / ("h3-ab-" + path.name)
    view.mkdir(parents=True, exist_ok=True)
    for source in path.iterdir():
        if source.is_file():
            destination = view / source.name
            if destination.exists():
                assert destination.resolve() == source.resolve()
            else:
                destination.symlink_to(source)
    return view


def completed_variant(folder, variant):
    """Only reuse a committed result whose inputs and output bytes still match."""
    try:
        record = json.loads((folder / f"{variant}.json").read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            return False
        files = {"latent_sha256": "latents.pt", "audio_sha256": "audio.pt",
                 "output_sha256": f"{variant}.mp4", "preview_sha256": f"{variant}-web.mp4"}
        return all(sha256(folder / name) == record[key] for key, name in files.items())
    except (OSError, ValueError, KeyError):
        return False


def resolve(manifest_path, case_id):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model"]["id"] == "MiniMaxAI/MiniMax-H3"
    assert manifest["model"]["variant"] == "fl2va"
    assert manifest["sampling"]["actual_nfe"] == 5
    assert manifest["sampling"]["num_inference_steps"] == 6
    cases = manifest["cases"]
    assert len(cases) == 10 and len({c["id"] for c in cases}) == 10
    for case in cases:
        q = case["request"]
        assert q["task"] == "t2va" and q["conditions"] == []
        assert q["model"] == manifest["model"]["id"]
        assert q["num_inference_steps"] == 6
        assert q["flow_shift"] == 12 and q["audio_flow_shift"] == 3
        assert q["seconds"] == 5 and q["target"] == {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5}
        assert isinstance(q["seed"], int) and q["prompt"].strip()
    geometry = manifest["geometry"]
    assert (geometry["output_width"], geometry["output_height"], geometry["output_frames"], geometry["fps"]) == (1344, 768, 124, 24)
    if case_id:
        cases = [c for c in cases if c["id"] == case_id]
        if not cases:
            raise ValueError(f"unknown case {case_id}")
    return manifest, cases


def main():
    execution_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "eval/h3_vae_ab/manifest.json")
    parser.add_argument("--h3", type=Path)
    parser.add_argument("--light-vae", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--variant", choices=("baseline", "light", "both"), default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest, cases = resolve(args.manifest, args.case)
    plan = {"manifest_sha256": sha256(args.manifest), "gpu_count": 1, "tp_size": 1,
            "model": manifest["model"], "light_vae": manifest["light_vae"], "sampling": manifest["sampling"],
            "geometry": manifest["geometry"], "case_ids": [c["id"] for c in cases],
            "variant": args.variant, "precision": {"dit": "bf16", "text": "bf16", "vae_weights": "fp32", "vae_decode": "fp16"},
            "attention_backend": "native", "torch_compile": False, "offload": "sequential model phases",
            "vae_tile_layout": "native", "vae_tile_batch": 1,
            "historical_runtime_attested": False, "historical_tp_size": 8,
            "e2e_definition": "shared conditioning+sampling+audio decode + each video decode+encode; phased sum, not independent request timing"}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        write_json(args.output / "plan.json", plan)
        print(json.dumps({"status": "dry_run_passed", "cases": len(cases), "gpu_count": 1}))
        return
    for name, path, rev in [("h3", args.h3, manifest["model"]["revision"]),
                            ("light", args.light_vae, manifest["light_vae"]["revision"])]:
        if path is None:
            raise ValueError(f"{name} path is required")
        marker = json.loads((path / "H3_VAE_AB_READY.json").read_text(encoding="utf-8"))
        assert marker["complete"] and marker["revision"] == rev
        for entry in marker["files"]:
            assert entry["verified"]
            assert (path / entry["path"]).stat().st_size == entry["size"], entry["path"]
        plan[name + "_marker_sha256"] = sha256(path / "H3_VAE_AB_READY.json")
        write_json(args.output / (name + "-weights.json"), marker)
    for key in ("latent_channels", "latents_mean", "latents_std", "spatial_compression_ratio", "temporal_compression_ratio"):
        baseline_config = json.loads((args.h3 / "vae/config.json").read_text(encoding="utf-8"))
        light_config = json.loads((args.light_vae / "config.json").read_text(encoding="utf-8"))
        if key in baseline_config:
            assert baseline_config[key] == light_config[key], key
    if args.preflight:
        write_json(args.output / "plan.json", plan)
        print("H3_VAE_AB_PREFLIGHT_PASS", flush=True)
        return

    import torch
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AudioDecodeStep, MiniMaxH3VideoDecodeStep
    from model.light_vae import LightVAE
    from model.output import configure_video_output, encode_video

    assert torch.cuda.device_count() == 1, "This experiment must use exactly one GPU"
    torch.set_grad_enabled(False)
    plan.update(job_id=os.environ.get("DLC_JOB_ID"), code_sha=os.environ.get("LYNNREAL_RELEASE_SHA"),
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                nvidia_smi=subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True))
    source_hash = hashlib.sha256()
    for path in sorted([Path(__file__), ROOT / "model/light_vae.py", ROOT / "model/output.py"]):
        source_hash.update(path.name.encode() + path.read_bytes())
    plan["runtime_source_sha256"] = source_hash.hexdigest()
    identity = {k: plan[k] for k in ("manifest_sha256", "h3_marker_sha256", "light_marker_sha256", "runtime_source_sha256")}
    if (args.output / "identity.json").exists():
        assert json.loads((args.output / "identity.json").read_text(encoding="utf-8")) == identity, "Refuse stale result reuse"
    write_json(args.output / "identity.json", identity)
    write_json(args.output / "run.json", plan)
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=(args.output / "pip-freeze.txt").open("w"), check=True)
    timings = json.loads((args.output / "timings.json").read_text(encoding="utf-8")) if (args.output / "timings.json").exists() else {}

    def measure(name, fn):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        used_samples = [torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]]
        stop = threading.Event()
        def sample_memory():
            while not stop.wait(0.1):
                free, total = torch.cuda.mem_get_info()
                used_samples.append(total - free)
        monitor = threading.Thread(target=sample_memory, daemon=True)
        monitor.start()
        started = time.perf_counter()
        try:
            value = fn()
            torch.cuda.synchronize()
        except Exception as error:
            failure_path = args.output / "failures.json"
            failures = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else []
            failures.append({"stage": name, "job_id": plan["job_id"], "unix": time.time(),
                             "error_type": type(error).__name__, "error": str(error),
                             "elapsed_seconds": time.perf_counter() - started})
            write_json(failure_path, failures)
            raise
        finally:
            stop.set()
            monitor.join()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        timings[name] = {"wall_seconds": elapsed, "allocated_before_bytes": before,
                         "reserved_before_bytes": reserved, "peak_allocated_bytes": peak,
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                         "incremental_peak_allocated_bytes": max(0, peak - before),
                         "device_used_sampled_peak_bytes": max(used_samples), "memory_sample_interval_seconds": 0.1}
        write_json(args.output / "timings.json", timings)
        print(name, json.dumps(timings[name]), flush=True)
        return value

    def load(blocks, dtype):
        pipe = blocks.init_pipeline(str(args.h3))
        pipe.load_components(dtype=dtype, pretrained_model_name_or_path=str(args.h3), local_files_only=True)
        pipe.to("cuda")
        return pipe

    def release():
        gc.collect()
        torch.cuda.empty_cache()

    def save_tensor(path, value):
        temp = path.with_suffix(".partial")
        torch.save(value, temp)
        os.replace(temp, path)

    # Preserve the separate warmup input; never use its latent as a measured case.
    warmup = dict(manifest["cases"][0], id="_warmup", request=dict(manifest["cases"][0]["request"], seed=40999))
    work = [warmup] + cases
    for case in work:
        (args.output / case["id"]).mkdir(exist_ok=True)
    pending = [c for c in work if not (args.output / c["id"] / "conditioning.pt").exists()]
    if pending:
        pipe = measure("text_model_load", lambda: load(MiniMaxH3TextEncoderStep(), torch.bfloat16))
        for case in pending:
            result = measure(case["id"] + "/conditioning", lambda: pipe(prompt=case["request"]["prompt"], keyframes=[], output=["prompt_embeds", "text_token_tags"]))
            save_tensor(args.output / case["id"] / "conditioning.pt", {k: v.cpu() for k, v in result.items()})
        del pipe, result
        release()
    pending = [c for c in work if not (args.output / c["id"] / "latents.pt").exists()]
    if pending:
        blocks = MiniMaxH3Blocks()
        for name in list(blocks.sub_blocks):
            if name not in {"setup", "prepare_layout", "prepare_latents", "set_timesteps", "denoise"}:
                del blocks.sub_blocks[name]
        pipe = measure("dit_model_load", lambda: load(blocks, torch.bfloat16))
        pipe.transformer.set_attention_backend("native")
        pipe.scheduler.set_shift(12)
        pipe.audio_scheduler.set_shift(3)
        forwards = []
        hook = pipe.transformer.register_forward_hook(lambda *unused: forwards.append(1))
        keys = ["latents", "audio_latents", "num_latent_frames", "latent_height", "latent_width", "num_audio_latents"]
        for case in pending:
            folder = args.output / case["id"]
            condition = torch.load(folder / "conditioning.pt", map_location="cpu", weights_only=True)
            forwards.clear()
            state = measure(case["id"] + "/sampling", lambda: pipe(
                prompt_embeds=condition["prompt_embeds"].to("cuda"), text_token_tags=condition["text_token_tags"],
                height=768, width=1344, num_frames=124, num_inference_steps=6,
                generator=torch.Generator().manual_seed(case["request"]["seed"]), output=keys))
            assert len(forwards) == 5, len(forwards)
            state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
            assert all(torch.isfinite(state[k]).all() for k in ("latents", "audio_latents"))
            save_tensor(folder / "latents.pt", state)
            write_json(folder / "latent.json", {"sha256": sha256(folder / "latents.pt"), "actual_nfe": len(forwards),
                       "video_sigma": pipe.scheduler.sigmas.cpu().tolist(), "audio_sigma": pipe.audio_scheduler.sigmas.cpu().tolist()})
        hook.remove()
        del pipe, state, condition
        release()
    pending = [c for c in work if not (args.output / c["id"] / "audio.pt").exists()]
    if pending:
        pipe = measure("audio_model_load", lambda: load(MiniMaxH3AudioDecodeStep(), torch.float32))
        for case in pending:
            folder = args.output / case["id"]
            latent = torch.load(folder / "latents.pt", map_location="cuda", weights_only=True)
            state = measure(case["id"] + "/audio_decode", lambda: pipe(audio_latents=latent["audio_latents"],
                            num_audio_latents=latent["num_audio_latents"], output_type="pt", output=["audio", "sampling_rate"]))
            save_tensor(folder / "audio.pt", {"audio": state["audio"].cpu(), "sampling_rate": state["sampling_rate"]})
        del pipe, state, latent
        release()
    variants = ["baseline", "light"] if args.variant == "both" else [args.variant]
    for variant in variants:
        pending = [c for c in work if not completed_variant(args.output / c["id"], variant)]
        if not pending:
            continue
        def load_video():
            pipe = MiniMaxH3VideoDecodeStep().init_pipeline(str(args.h3))
            if variant == "light":
                pipe.update_components(vae=LightVAE.from_pretrained(light_weight_view(args.light_vae), tile_batch=1, tile_layout="native"))
            else:
                pipe.load_components(dtype=torch.float32, pretrained_model_name_or_path=str(args.h3), local_files_only=True)
            pipe.to("cuda")
            configure_video_output(pipe.video_processor)
            return pipe
        pipe = measure(variant + "_model_load", load_video)
        core = pipe.vae.core if variant == "light" else pipe.vae
        core.set_attention_backend("native")
        tile = {key: getattr(core, key) for key in ("tile_sample_min_height", "tile_sample_min_width",
                 "tile_sample_min_overlap_height", "tile_sample_min_overlap_width", "tokens_chunk_size", "token_overlap")}
        tile.update(layout="native", tile_batch=pipe.vae.tile_batch if variant == "light" else 1,
                    use_tiling=core.use_tiling)
        decoder = pipe.vae.decode
        current = [""]
        def timed_decode(*a, **kw):
            decoded = measure(current[0] + "/" + variant + "_decoder", lambda: decoder(*a, **kw))
            if not torch.isfinite(decoded[0]).all():
                raise ValueError(f"Non-finite {variant} decoder output for {current[0]}")
            return decoded
        pipe.vae.decode = timed_decode
        for case in pending:
            folder = args.output / case["id"]
            latent_meta = json.loads((folder / "latent.json").read_text(encoding="utf-8"))
            assert sha256(folder / "latents.pt") == latent_meta["sha256"]
            latent = torch.load(folder / "latents.pt", map_location="cuda", weights_only=True)
            audio = torch.load(folder / "audio.pt", map_location="cpu", weights_only=True)
            current[0] = case["id"]
            start = time.perf_counter()
            video = pipe(**{k: latent[k] for k in ("latents", "num_latent_frames", "latent_height", "latent_width")}, output_type="pil", output="videos")[0]
            torch.cuda.synchronize()
            video_wall = time.perf_counter() - start
            assert len(video) == 124 and all(im.size == (1344, 768) for im in video)
            output = folder / f"{variant}.mp4"
            start = time.perf_counter()
            encode_video(video, 24, output, audio=audio["audio"][0], audio_sample_rate=audio["sampling_rate"])
            encode_seconds = time.perf_counter() - start
            import imageio_ffmpeg
            preview = output.with_name(variant + "-web.mp4")
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(output),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-movflags", "+faststart",
                            "-c:a", "copy", str(preview)], check=True)
            # Check the muxed audio too: identical source tensors alone do not validate muxing.
            import av
            audio_digest = hashlib.sha256()
            with av.open(str(output)) as container:
                for frame in container.decode(audio=0):
                    audio_digest.update(frame.to_ndarray().tobytes())
            record = {"case": case, "variant": variant, "latent_sha256": latent_meta["sha256"],
                      "audio_sha256": sha256(folder / "audio.pt"), "output_sha256": sha256(output),
                      "decoded_audio_sha256": audio_digest.hexdigest(), "preview_sha256": sha256(preview),
                      "video": output.name, "decoder": timings[case["id"] + "/" + variant + "_decoder"],
                      "decode_postprocess_seconds": video_wall, "encode_seconds": encode_seconds,
                      "tile": tile, "frames": len(video), "width": 1344, "height": 768, "fps": 24,
                      "job_id": plan["job_id"], "code_sha": plan["code_sha"], "status": "complete",
                      "completed_unix": time.time(), "elapsed_from_runner_start_seconds": time.perf_counter() - execution_started}
            shared = sum(timings[case["id"] + "/" + stage]["wall_seconds"] for stage in ("conditioning", "sampling", "audio_decode"))
            record["phased_e2e_seconds"] = shared + video_wall + encode_seconds
            record["e2e_definition"] = plan["e2e_definition"]
            write_json(folder / f"{variant}.json", record)
            del video, latent, audio
        del pipe, core, decoder
        release()
    aggregates = {}
    if args.variant == "both":
        for case in cases:
            folder = args.output / case["id"]
            a = json.loads((folder / "baseline.json").read_text(encoding="utf-8"))
            b = json.loads((folder / "light.json").read_text(encoding="utf-8"))
            assert all(a[k] == b[k] for k in ("latent_sha256", "audio_sha256", "decoded_audio_sha256")), case["id"]
    for variant in variants:
        records = [json.loads((args.output / c["id"] / (variant + ".json")).read_text(encoding="utf-8")) for c in cases]
        latencies = [r["decoder"]["wall_seconds"] for r in records]
        aggregates[variant] = {"decoder_mean_seconds": statistics.mean(latencies), "decoder_median_seconds": statistics.median(latencies),
                               "max_peak_allocated_bytes": max(r["decoder"]["peak_allocated_bytes"] for r in records),
                               "phased_e2e_mean_seconds": statistics.mean(r["phased_e2e_seconds"] for r in records)}
    write_json(args.output / "summary.json", {"status": "passed", "cases": len(cases), "variants": variants,
                                               "gpu_count": 1, "job_id": plan["job_id"], "identity": identity, "metrics": aggregates,
                                               "runner_wall_seconds": time.perf_counter() - execution_started,
                                               "overall_peak_allocated_bytes": max(t["peak_allocated_bytes"] for t in timings.values()),
                                               "overall_peak_reserved_bytes": max(t["peak_reserved_bytes"] for t in timings.values())})
    (args.output / "READY").write_text("complete\n")
    print("H3_VAE_AB_PASS", flush=True)


if __name__ == "__main__":
    main()
