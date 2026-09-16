"""CPU-only, hash-verified preparation of the official H3 / Light VAE cache."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path, entry):
    if "lfs" in entry:
        h = hashlib.sha256()
        expected = entry["lfs"]["oid"]
    else:
        h = hashlib.sha1(f"blob {entry['size']}\0".encode())
        expected = entry["oid"]
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest() == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/cpfs/world-model/lynnreal-omni/models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--wait-for-stage-seconds", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = json.loads((ROOT / "eval/h3_vae_ab/manifest.json").read_text())
    trees = {name: json.loads((ROOT / f"eval/h3_vae_ab/{name}-weight-tree.json").read_text()) for name in ("h3", "light")}
    components = {"transformer", "text_encoder", "tokenizer", "processor", "vae", "audio_vae", "scheduler", "audio_scheduler"}
    roots = [Path("/cpfs/world-model/checkpoints/MiniMax-H3-Ref2VA-diffusers-42ed227"),
             args.root / "standard-f1d6990e23496bef6c7d55bdcfe2925df2508325"]
    stage = Path("/mnt/world-model/lynnreal-omni/staged-models-crr/h3-fl2va-" + manifest["model"]["revision"])
    if args.wait_for_stage_seconds:
        deadline = time.monotonic() + args.wait_for_stage_seconds
        staged_files = [e for e in trees["h3"] if e["type"] == "file" and e["path"].startswith("transformer/")]
        while not ((stage / (".complete-" + manifest["model"]["revision"])).is_file() and
                   all((stage / e["path"]).is_file() and (stage / e["path"]).stat().st_size == e["size"] for e in staged_files)):
            if time.monotonic() >= deadline:
                raise TimeoutError("The complete transformer has not reached Thailand OSS")
            print("Waiting for complete Thailand transformer stage", flush=True)
            time.sleep(10)
    if (stage / (".complete-" + manifest["model"]["revision"])).is_file():
        roots.append(stage)
    summary = {"started_unix": started, "gpu_used": False, "models": {}}
    for name, model in [("h3", manifest["model"]), ("light", manifest["light_vae"])]:
        revision = model["revision"]
        repo = model["id"] if name == "h3" else model["repository"]
        target = args.root / f"{'h3' if name == 'h3' else 'light-vae'}-{revision}"
        target.mkdir(parents=True, exist_ok=True)
        entries = [e for e in trees[name] if e["type"] == "file" and
                   ((name == "light" and (e["path"].endswith(".safetensors") or e["path"] in {
                       "config.json", "decode_config.json", "diffusion_pytorch_model.safetensors.index.json"})) or
                    (name == "h3" and (e["path"].split("/")[0] in components or
                     e["path"] in {"model_index.json", "modular_model_index.json"})))]
        candidates = [target] + roots if name == "h3" else [target]
        def prepare(entry):
            relative, size = entry["path"], entry["size"]
            identity = {"path": relative, "size": size, "git_blob_oid": entry["oid"],
                        "sha256": entry["lfs"]["oid"] if "lfs" in entry else None}
            destination = target / relative
            for base in candidates:
                source = base / relative
                if source.is_file() and source.stat().st_size == size and digest(source, entry):
                    if source != destination:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        partial = destination.with_name(destination.name + ".verified-link")
                        partial.unlink(missing_ok=True)
                        if str(source).startswith("/mnt/world-model/"):
                            import shutil
                            shutil.copyfile(source, partial)
                            if not digest(partial, entry):
                                raise ValueError("staged copy hash mismatch")
                        else:
                            os.link(source, partial)
                        os.replace(partial, destination)
                    print("VERIFIED", name, relative, size, flush=True)
                    return dict(identity, verified=True)
            if not args.download:
                return dict(identity, verified=False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            # Always verify against the immutable HF tree, including mirrored bytes.
            errors = []
            for endpoint in ("https://huggingface.co", "https://hf-mirror.com"):
                try:
                    url = f"{endpoint}/{repo}/resolve/{revision}/{relative}"
                    with urllib.request.urlopen(url, timeout=120) as reader, partial.open("wb") as writer:
                        received, last_report = 0, 0
                        for block in iter(lambda: reader.read(8 << 20), b""):
                            writer.write(block)
                            received += len(block)
                            if received - last_report >= 256 << 20:
                                last_report = received
                                (args.output / (name + "-" + Path(relative).name + ".progress.json")).write_text(
                                    json.dumps({"path": relative, "received": received, "total": size, "time": time.time()}))
                    if partial.stat().st_size != size or not digest(partial, entry):
                        raise ValueError("download hash/size mismatch")
                    os.replace(partial, destination)
                    print("DOWNLOADED", name, relative, size, flush=True)
                    return dict(identity, verified=True)
                except Exception as exc:
                    errors.append(type(exc).__name__ + ": " + str(exc))
            return dict(identity, verified=False, errors=errors)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(prepare, entries))
        result = {"repository": repo, "revision": revision, "files": records,
                  "complete": all(r["verified"] for r in records), "root": str(target)}
        summary["models"][name] = result
        if result["complete"]:
            marker = target / "H3_VAE_AB_READY.json"
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            os.replace(temporary, marker)
        summary["elapsed_seconds"] = time.time() - started
        (args.output / "weights-preflight.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["environment_archives"] = [str(p) for p in (args.root.parent / "cache/env").glob("*.tar.gz")]
    summary["complete"] = all(m["complete"] for m in summary["models"].values())
    (args.output / "weights-preflight.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": summary["complete"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    if not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
