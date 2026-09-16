#!/usr/bin/env bash
set -u -o pipefail

readonly RELEASE_SHA="${LYNNREAL_RELEASE_SHA:-97fdcb871b28b3a2b2316b5d19a9ea8e61185570}"
readonly STANDARD_REV="${LYNNREAL_STANDARD_REV:-f1d6990e23496bef6c7d55bdcfe2925df2508325}"
readonly LIGHT_VAE_REV="${LYNNREAL_LIGHT_VAE_REV:-e453444c1a52b73a0c5eb0023d208c473c2bb26a}"

readonly MOUNT_ROOT="${LYNNREAL_MOUNT_ROOT:-/mnt/lynnreal-omni}"
readonly LOCAL_ROOT="${LYNNREAL_LOCAL_ROOT:-/local/lynnreal-omni}"
readonly WORKSPACE="/root/workspace"
readonly RELEASE_DIR="$WORKSPACE/LynnReal-Omni"
readonly MODEL_ROOT="$MOUNT_ROOT/models"
readonly ENV_DIR="$LOCAL_ROOT/env"
readonly ENV_ARCHIVE="$MOUNT_ROOT/cache/env/env-py312-torch271-cu128-fa3-203b9b3-v3.tar.gz"
readonly RUN_NAME="official_540p_standard_${DLC_JOB_ID:-manual}"
readonly RESULT_ROOT="$MOUNT_ROOT/results/${DLC_JOB_ID:-manual}/official-speed/standard-540p"
readonly LOCAL_MODEL_ROOT="$LOCAL_ROOT/models"
readonly LOCAL_CACHE_ROOT="$LOCAL_ROOT/cache/official-standard-540p-$RELEASE_SHA"
readonly CACHE_ARCHIVE="$MOUNT_ROOT/cache/compiled/official-standard-540p-h100-$RELEASE_SHA.tar.gz"

mkdir -p "$RESULT_ROOT"
exec > >(tee -a "$RESULT_ROOT/job.log") 2>&1

benchmark() {
  set -euxo pipefail
  rm -f "$RESULT_ROOT/FAILED" "$RESULT_ROOT/READY"

  nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv
  python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
assert torch.cuda.get_device_capability(0) == (9, 0), torch.cuda.get_device_capability(0)
assert torch.cuda.get_device_properties(0).total_memory >= 79 * 2**30
print({"gpu": torch.cuda.get_device_name(0), "count": torch.cuda.device_count()})
PY

  test -s "$MOUNT_ROOT/code/LynnReal-Omni-$RELEASE_SHA.tar.gz"
  test -s "$ENV_ARCHIVE"
  test -f "$MODEL_ROOT/standard-$STANDARD_REV/transformer/diffusion_pytorch_model.safetensors.index.json"
  test -f "$MODEL_ROOT/light-vae-$LIGHT_VAE_REV/diffusion_pytorch_model.safetensors.index.json"

  mkdir -p "$WORKSPACE" "$LOCAL_ROOT" "$LOCAL_MODEL_ROOT" "$LOCAL_CACHE_ROOT"
  rm -rf "$RELEASE_DIR"
  tar -xzf "$MOUNT_ROOT/code/LynnReal-Omni-$RELEASE_SHA.tar.gz" -C "$WORKSPACE"

  if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    env_tmp="$LOCAL_ROOT/.env-${DLC_JOB_ID:-manual}.partial"
    rm -rf "$env_tmp"
    mkdir -p "$env_tmp"
    tar -xzf "$ENV_ARCHIVE" -C "$env_tmp"
    rm -rf "$ENV_DIR"
    mv "$env_tmp/env" "$ENV_DIR"
    rmdir "$env_tmp"
  fi

  ensure_local_model() {
    local kind="$1" revision="$2"
    local source="$MODEL_ROOT/$kind-$revision"
    local target="$LOCAL_MODEL_ROOT/$kind-$revision"
    local marker="$target/.local-copy-complete-$revision"
    if [[ ! -f "$marker" ]]; then
      local model_tmp="$LOCAL_MODEL_ROOT/.$kind-${DLC_JOB_ID:-manual}.partial"
      rm -rf "$model_tmp"
      mkdir -p "$model_tmp"
      cp -a "$source/." "$model_tmp/"
      touch "$model_tmp/.local-copy-complete-$revision"
      rm -rf "$target"
      mv "$model_tmp" "$target"
    fi
  }
  ensure_local_model standard "$STANDARD_REV"
  ensure_local_model light-vae "$LIGHT_VAE_REV"

  if [[ -s "$CACHE_ARCHIVE" && ! -e "$LOCAL_CACHE_ROOT/.archive-restored" ]]; then
    tar -xzf "$CACHE_ARCHIVE" -C "$LOCAL_CACHE_ROOT"
    touch "$LOCAL_CACHE_ROOT/.archive-restored"
  fi
  export CUDA_VISIBLE_DEVICES=0
  export LYNNREAL_PYTHON="$ENV_DIR/bin/python"
  export TRITON_CACHE_DIR="$LOCAL_CACHE_ROOT/triton"
  export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_ROOT/inductor"
  mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$RELEASE_DIR/output"
  rm -rf "$RELEASE_DIR/output/.cache"
  ln -s "$LOCAL_CACHE_ROOT" "$RELEASE_DIR/output/.cache"

  rm -rf "$RELEASE_DIR/weight"
  mkdir -p "$RELEASE_DIR/weight/standard" "$RELEASE_DIR/weight/light-vae"
  for name in .gitattributes LICENSE NOTICE README.md inference_config.json \
    model_index.json modular_model_index.json audio_scheduler audio_vae processor \
    scheduler text_encoder tokenizer transformer vae; do
    ln -s "$LOCAL_MODEL_ROOT/standard-$STANDARD_REV/$name" "$RELEASE_DIR/weight/standard/$name"
  done
  for item in "$LOCAL_MODEL_ROOT/light-vae-$LIGHT_VAE_REV"/* \
    "$LOCAL_MODEL_ROOT/light-vae-$LIGHT_VAE_REV"/.[!.]*; do
    [[ -e "$item" ]] || continue
    ln -s "$item" "$RELEASE_DIR/weight/light-vae/$(basename "$item")"
  done

  cd "$RELEASE_DIR"
  PIP_NO_CACHE_DIR=1 "$LYNNREAL_PYTHON" script/setup_env.py \
    --attention-only --attention _flash_3 --report-dir "$RESULT_ROOT/setup"

  bash script/sample/speed_test/standard.sh \
    --resolution 540p \
    --search fixed \
    --warmups 2 \
    --repeats 5 \
    --seed 77 \
    --name "$RUN_NAME"

  rm -rf "$RESULT_ROOT/run"
  cp -a "output/speed_test/standard/$RUN_NAME" "$RESULT_ROOT/run"
  mkdir -p "$(dirname "$CACHE_ARCHIVE")"
  cache_tmp="$LOCAL_ROOT/official-standard-cache-${DLC_JOB_ID:-manual}.tar.gz"
  tar -czf "$cache_tmp" -C "$LOCAL_CACHE_ROOT" .
  cp "$cache_tmp" "$CACHE_ARCHIVE.partial-${DLC_JOB_ID:-manual}"
  mv "$CACHE_ARCHIVE.partial-${DLC_JOB_ID:-manual}" "$CACHE_ARCHIVE"
  rm -f "$cache_tmp"
  sha256sum "$RESULT_ROOT/run/video.mp4" "$RESULT_ROOT/run/latency.json" > "$RESULT_ROOT/SHA256SUMS"
  date -Iseconds > "$RESULT_ROOT/READY"
}

set +e
(benchmark)
rc=$?
if ((rc == 0)); then
  echo "Official Standard 540p speed test completed."
else
  printf '%s\n' "$rc" > "$RESULT_ROOT/FAILED"
  echo "Official Standard 540p speed test failed with exit code $rc."
fi
exit "$rc"
