#!/usr/bin/env bash
set -Eeuo pipefail

readonly RELEASE_SHA="${LYNNREAL_RELEASE_SHA:?LYNNREAL_RELEASE_SHA is required}"
readonly RUN_MODE="${LYNNREAL_RUN_MODE:-speed-test}"
readonly STANDARD_REV="f1d6990e23496bef6c7d55bdcfe2925df2508325"
readonly FLASH_REV="9950cfc882b0e6b8600fbc49b58dc1d82d7a435b"
readonly LIGHT_VAE_REV="e453444c1a52b73a0c5eb0023d208c473c2bb26a"

readonly PERSIST_ROOT="/cpfs/world-model/lynnreal-omni"
readonly LOCAL_ROOT="/local/lynnreal-omni"
readonly RELEASE_DIR="$(pwd)"
readonly RESULT_ROOT="$PERSIST_ROOT/results/${DLC_JOB_ID:-manual-run}"
readonly OSS_RESULT_ROOT="/mnt/world-model/results/lynnreal-omni/${DLC_JOB_ID:-manual-run}"
readonly MODEL_ROOT="$PERSIST_ROOT/models"
readonly SHARED_CACHE_ROOT="$PERSIST_ROOT/cache/shared"
readonly KERNEL_SHA="$(sha256sum model/int8_gemm.py model/int8_tma.py | sha256sum | cut -c1-16)"
readonly DIFFUSERS_SHA="584a47eb49eaf60fda2843317708eacc6a7d9622f1630d9badb0b33fc480aaba"
readonly RUNTIME_SOURCE_SHA="$({
  find model -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
  sha256sum requirements.txt
  printf '%s  %s\n' "$DIFFUSERS_SHA" vendor/diffusers-abc5e9bf71fd.tar.gz
} | sha256sum | cut -c1-16)"
readonly CONTENT_CACHE="$PERSIST_ROOT/cache/compiled/b300-torch2121-cu130-$RUNTIME_SOURCE_SHA"
readonly VALIDATED_LEGACY_CACHE="$PERSIST_ROOT/cache/compiled/b300-torch2121-351f41e16ec29f9ae51b62ac5d531107e270e688-44e0d680bad9991a"
if [[ "$RUNTIME_SOURCE_SHA" == "f6265ace1607fac9" && -d "$VALIDATED_LEGACY_CACHE" ]]; then
  readonly COMPILED_CACHE="$VALIDATED_LEGACY_CACHE"
else
  readonly COMPILED_CACHE="$CONTENT_CACHE"
fi
readonly BASE_PYTHON="/opt/minwm/venv/bin/python"
readonly OVERLAY_DIR="$LOCAL_ROOT/overlay"

mkdir -p "$RESULT_ROOT"
exec > >(tee -a "$RESULT_ROOT/bootstrap.log") 2>&1

finish() {
  local rc=$?
  if ((rc == 0)); then
    printf 'ready %s\n' "$(date -u +%FT%TZ)" > "$RESULT_ROOT/READY"
  else
    printf '%s\n' "$rc" > "$RESULT_ROOT/FAILED"
  fi
}
trap finish EXIT

set -x
[[ "$RUN_MODE" == "speed-test" || "$RUN_MODE" == "eval" ]]
test "$(git rev-parse HEAD 2>/dev/null || printf '%s' "$RELEASE_SHA")" = "$RELEASE_SHA"
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')" = 1
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

python3 - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
major, minor = torch.cuda.get_device_capability(0)
assert major >= 10, (major, minor)
assert torch.cuda.get_device_properties(0).total_memory >= 250 * 2**30
print({"gpu": torch.cuda.get_device_name(0), "capability": [major, minor]})
PY

mkdir -p "$LOCAL_ROOT" "$MODEL_ROOT" "$SHARED_CACHE_ROOT" "$COMPILED_CACHE" \
  "$PERSIST_ROOT/cache/env" "$RELEASE_DIR/output"
rm -rf "$RELEASE_DIR/output/.cache"
ln -s "$COMPILED_CACHE" "$RELEASE_DIR/output/.cache"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$SHARED_CACHE_ROOT/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60
export PIP_CACHE_DIR="$SHARED_CACHE_ROOT/pip"
export TRITON_CACHE_DIR="$COMPILED_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$COMPILED_CACHE/inductor"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

test -x "$BASE_PYTHON"
env_key="$(sha256sum requirements.txt vendor/diffusers-abc5e9bf71fd.tar.gz | sha256sum | cut -c1-16)"
overlay_archive="$PERSIST_ROOT/cache/env/py312-torch2121-cu130-lynnreal-$env_key.tar.gz"
built_overlay=false
if [[ -s "$overlay_archive" ]]; then
  rm -rf "$OVERLAY_DIR"
  tar -xzf "$overlay_archive" -C "$LOCAL_ROOT"
else
  mkdir -p "$OVERLAY_DIR"
  "$BASE_PYTHON" -m pip install --no-deps --no-build-isolation --upgrade \
    --index-url http://mirrors.cloud.aliyuncs.com/pypi/simple/ \
    --trusted-host mirrors.cloud.aliyuncs.com --target "$OVERLAY_DIR" \
    "diffusers @ file://$RELEASE_DIR/vendor/diffusers-abc5e9bf71fd.tar.gz" \
    transformers==5.10.2 accelerate==1.13.0 peft==0.19.1 safetensors==0.8.0 \
    huggingface-hub==1.23.0 numpy==2.4.6 pillow==12.2.0 sentencepiece==0.2.1 \
    imageio-ffmpeg==0.6.0 av==17.0.1 einops==0.8.2 scipy==1.17.1
  built_overlay=true
fi

export PYTHONPATH="$OVERLAY_DIR:$RELEASE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LYNNREAL_PYTHON="$BASE_PYTHON"
"$BASE_PYTHON" - <<'PY'
from importlib.metadata import version
import torch
from diffusers import MiniMaxH3Transformer3DModel

assert torch.__version__ == "2.12.1+cu130", torch.__version__
for package, expected in {
    "transformers": "5.10.2", "accelerate": "1.13.0", "peft": "0.19.1",
    "safetensors": "0.8.0", "huggingface-hub": "1.23.0", "numpy": "2.4.6",
    "pillow": "12.2.0", "sentencepiece": "0.2.1", "imageio-ffmpeg": "0.6.0",
    "av": "17.0.1", "einops": "0.8.2", "scipy": "1.17.1",
}.items():
    assert version(package) == expected, (package, version(package), expected)
print({"torch": torch.__version__, "cuda": torch.version.cuda, "diffusers_model": MiniMaxH3Transformer3DModel.__name__})
PY
if [[ "$built_overlay" == true ]]; then
  partial="$overlay_archive.partial.${DLC_JOB_ID:-$$}"
  tar -czf "$partial" -C "$LOCAL_ROOT" overlay
  mv "$partial" "$overlay_archive"
fi
PIP_NO_CACHE_DIR=1 "$BASE_PYTHON" script/setup_env.py \
  --attention-only --attention _native_cudnn --report-dir "$RESULT_ROOT/setup"

test -f "$MODEL_ROOT/standard-$STANDARD_REV/.downloaded-modelscope-master"
test -f "$MODEL_ROOT/flash-$FLASH_REV/.downloaded-modelscope-master"
test -f "$MODEL_ROOT/light-vae-$LIGHT_VAE_REV/.downloaded-modelscope-master"

"$BASE_PYTHON" - <<PY
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

model_root = Path("$MODEL_ROOT")
h3_root = Path("/cpfs/world-model/checkpoints/MiniMax-H3-Ref2VA-diffusers-42ed227")
common = ("audio_scheduler", "audio_vae", "processor", "scheduler", "text_encoder", "tokenizer", "vae")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def component_valid(root: Path, files: list[dict]) -> bool:
    valid = bool(files)
    for entry in files:
        source = root / entry["path"]
        if not source.is_file() or source.stat().st_size != entry["size"]:
            return False
        lfs_oid = (entry.get("lfs") or {}).get("oid")
        if lfs_oid:
            valid = sha256(source) == lfs_oid
        else:
            content = source.read_bytes()
            valid = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == entry["oid"]
        if not valid:
            return False
    return valid


def reuse_h3_components(target: Path) -> tuple[list[str], list[str]]:
    entries = json.loads(Path("vendor/standard-hf-tree.json").read_text())
    reused = []
    local = []
    for component in common:
        files = [entry for entry in entries if entry.get("type") == "file" and entry["path"].startswith(component + "/")]
        if component_valid(target, files):
            local.append(component)
            continue
        if not h3_root.is_dir() or not component_valid(h3_root, files):
            continue
        for entry in files:
            source, destination = h3_root / entry["path"], target / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                if destination.exists() and os.path.samefile(source, destination):
                    continue
                destination.unlink()
            try:
                os.link(source, destination)
            except OSError:
                destination.symlink_to(source)
        reused.append(component)
    return reused, local


def complete(name: str, revision: str, reuse_h3: bool = False) -> Path:
    target = model_root / f"{name}-{revision}"
    marker = target / f".complete-{revision}"
    if marker.is_file():
        return target
    target.mkdir(parents=True, exist_ok=True)
    reused, local = reuse_h3_components(target) if reuse_h3 else ([], [])
    if reuse_h3 and set(reused) | set(local) != set(common):
        raise RuntimeError(
            f"common-component hash check failed: reused={reused}, local={local}"
        )
    marker.write_text(
        json.dumps(
            {
                "revision": revision,
                "reused_h3_components": reused,
                "local_components": local,
            }
        )
        + "\n"
    )
    return target


standard = complete("standard", "$STANDARD_REV", reuse_h3=True)
flash = complete("flash", "$FLASH_REV")
light = complete("light-vae", "$LIGHT_VAE_REV")
assert (standard / "transformer" / "diffusion_pytorch_model.safetensors.index.json").is_file()
assert (flash / "transformer" / "diffusion_pytorch_model.safetensors.index.json").is_file()
assert (light / "diffusion_pytorch_model.safetensors.index.json").is_file()
PY

standard_dir="$MODEL_ROOT/standard-$STANDARD_REV"
flash_dir="$MODEL_ROOT/flash-$FLASH_REV"
light_dir="$MODEL_ROOT/light-vae-$LIGHT_VAE_REV"
rm -rf "$RELEASE_DIR/weight"
mkdir -p "$RELEASE_DIR/weight/standard" "$RELEASE_DIR/weight/flash/transformer" \
  "$RELEASE_DIR/weight/light-vae"

for name in LICENSE NOTICE README.md inference_config.json model_index.json \
  modular_model_index.json audio_scheduler audio_vae processor scheduler \
  text_encoder tokenizer transformer vae; do
  ln -s "$standard_dir/$name" "$RELEASE_DIR/weight/standard/$name"
done
for item in "$light_dir"/* "$light_dir"/.[!.]*; do
  [[ -e "$item" ]] || continue
  ln -s "$item" "$RELEASE_DIR/weight/light-vae/$(basename "$item")"
done

for name in LICENSE NOTICE README.md inference_config.json model_index.json \
  modular_model_index.json; do
  ln -s "$flash_dir/$name" "$RELEASE_DIR/weight/flash/$name"
done
for item in "$flash_dir/transformer"/* "$flash_dir/transformer"/.[!.]*; do
  [[ -e "$item" ]] || continue
  ln -s "$item" "$RELEASE_DIR/weight/flash/transformer/$(basename "$item")"
done
for name in audio_scheduler audio_vae processor scheduler text_encoder tokenizer vae; do
  ln -s "$standard_dir/$name" "$RELEASE_DIR/weight/flash/$name"
done

"$BASE_PYTHON" script/validate_int8_gemm.py \
  --report "$RESULT_ROOT/int8-gemm-validation.json"

if [[ "$RUN_MODE" == "speed-test" ]]; then
  # Match the release speed-test protocol: identical prompt/seed and geometry,
  # two warmups, five measured calls, and hardware-specific backend search.
  bash script/sample/speed_test/standard.sh \
    --resolution 768p --frames 240 --seed 77 --warmups 2 --repeats 5 \
    --search fast --name aliyun_thailand_b300_official_10s
  cp -a output/speed_test/standard/aliyun_thailand_b300_official_10s "$RESULT_ROOT/standard"

  bash script/sample/speed_test/flash.sh \
    --resolution 768p --frames 240 --seed 77 --warmups 2 --repeats 5 \
    --search fast --name aliyun_thailand_b300_official_10s
  cp -a output/speed_test/flash/aliyun_thailand_b300_official_10s "$RESULT_ROOT/flash"

  "$BASE_PYTHON" - "$RESULT_ROOT" <<'PY'
import json
from pathlib import Path
import sys

import av

root = Path(sys.argv[1])
records = {}
for variant in ("standard", "flash"):
    path = root / variant / "video.mp4"
    with av.open(path) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
        duration = float(stream.duration * stream.time_base)
        fps = float(stream.average_rate)
        record = {"path": str(path), "frames": frames, "duration_seconds": duration,
                  "fps": fps, "width": stream.width, "height": stream.height,
                  "codec": stream.codec_context.name}
    assert frames == 240, record
    assert abs(duration - 10.0) <= 0.05, record
    assert abs(fps - 24.0) <= 0.01, record
    records[variant] = record
(root / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2), flush=True)
PY
  sha256sum "$RESULT_ROOT/standard/video.mp4" "$RESULT_ROOT/flash/video.mp4" > "$RESULT_ROOT/SHA256SUMS"
else
  readonly EVAL_MANIFEST="${LYNNREAL_EVAL_MANIFEST:?LYNNREAL_EVAL_MANIFEST is required in eval mode}"
  readonly EVAL_VARIANT="${LYNNREAL_EVAL_VARIANT:?LYNNREAL_EVAL_VARIANT is required in eval mode}"
  [[ "$EVAL_VARIANT" == "standard" || "$EVAL_VARIANT" == "flash" ]]
  test -f "$EVAL_MANIFEST"
  "$BASE_PYTHON" script/eval_t2v.py \
    --manifest "$EVAL_MANIFEST" --variant "$EVAL_VARIANT" \
    --frame 10s --resolution 768p --warmups 1 --attention-backend native \
    --output "$RESULT_ROOT/eval"
fi

printf '%s\n' \
  "run_mode=$RUN_MODE" \
  "release_sha=$RELEASE_SHA" \
  "standard_revision=$STANDARD_REV" \
  "flash_revision=$FLASH_REV" \
  "light_vae_revision=$LIGHT_VAE_REV" \
  "kernel_sha=$KERNEL_SHA" \
  "runtime_source_sha=$RUNTIME_SOURCE_SHA" \
  "environment_archive=$overlay_archive" \
  "model_root=$MODEL_ROOT" \
  "compiled_cache=$COMPILED_CACHE" \
  "result_oss_uri=oss://leap-worldmodel-thailand/world-model/results/lynnreal-omni/${DLC_JOB_ID:-manual-run}" \
  > "$RESULT_ROOT/runtime.env"
if [[ "$RUN_MODE" == "eval" ]]; then
  printf '%s\n' \
    "eval_variant=$EVAL_VARIANT" \
    "eval_manifest=$EVAL_MANIFEST" \
    >> "$RESULT_ROOT/runtime.env"
fi
echo "Bootstrap completed: mode=$RUN_MODE"
mkdir -p "$OSS_RESULT_ROOT"
cp -a "$RESULT_ROOT/." "$OSS_RESULT_ROOT/"
printf 'ready %s\n' "$(date -u +%FT%TZ)" > "$OSS_RESULT_ROOT/READY"
