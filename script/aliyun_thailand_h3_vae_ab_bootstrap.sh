#!/usr/bin/env bash
set -Eeuo pipefail
readonly ROOT="$(pwd)"
readonly PERSIST=/cpfs/world-model/lynnreal-omni
readonly RESULT="$PERSIST/results/${DLC_JOB_ID:?DLC_JOB_ID required}"
readonly OSS_RESULT="/mnt/world-model/results/lynnreal-omni/$DLC_JOB_ID"
H3="$PERSIST/models/h3-bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
LIGHT="$PERSIST/models/light-vae-e453444c1a52b73a0c5eb0023d208c473c2bb26a"
readonly ARCHIVE="$PERSIST/cache/env/py312-torch2121-cu130-lynnreal-0384eca3d5d7482d.tar.gz"
readonly LOCAL="/local/h3-vae-ab/$DLC_JOB_ID"
readonly PYTHON=/opt/minwm/venv/bin/python
mkdir -p "$RESULT" "$OSS_RESULT" "$LOCAL"
exec > >(tee -a "$RESULT/bootstrap.log") 2>&1
finish() {
  local rc=$?
  if ((rc != 0)); then printf '%s\n' "$rc" > "$RESULT/FAILED"; fi
  if [[ "${LYNNREAL_H3_TAE_ONLY:-0}" == 1 ]]; then
    # Keep lossless masters on CPFS; publish small evidence first, then previews.
    python3 script/upload_h3_tae_results.py "$RESULT" "$OSS_RESULT"
  else
    cp -a "$RESULT/." "$OSS_RESULT/"
  fi
}
trap finish EXIT
date -u +%FT%TZ > "$RESULT/bootstrap-start.txt"
startup_root=""
if [[ -s eval/h3_vae_ab/startup-receipt.json ]]; then
  startup_root="$(python3 - <<'PY'
import json,os
from pathlib import Path
r=json.loads(Path('eval/h3_vae_ab/startup-receipt.json').read_text())
m=json.loads((Path(r['root'])/'READY.json').read_text())
assert r['complete'] and m['complete'] and r['code_sha']==m['code_sha']==os.environ['LYNNREAL_RELEASE_SHA']
assert r['archive_sha256']==m['archive_sha256']
print(r['root'])
PY
)"
  H3="$startup_root/h3"
  LIGHT="$startup_root/light"
  cp eval/h3_vae_ab/startup-receipt.json "$RESULT/startup-cache.json"
fi
test -s "$H3/H3_VAE_AB_READY.json"
test -s "$LIGHT/H3_VAE_AB_READY.json"
test -s "$ARCHIVE"
python3 - "$ARCHIVE" <<'PY'
import hashlib, json, sys
from pathlib import Path
archive=Path(sys.argv[1])
marker=json.loads((archive.parent/'h3-vae-ab-env-ready.json').read_text())
receipt=json.loads(Path('eval/h3_vae_ab/runtime-receipt.json').read_text())
assert marker['archive_sha256'] == receipt['archive_sha256']
h=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(16<<20),b''): h.update(block)
assert h.hexdigest() == marker['archive_sha256']
PY
date -u +%FT%TZ > "$RESULT/environment-restore-start.txt"
if [[ -n "$startup_root" ]]; then
  overlay="$startup_root/environment/overlay"
  test -d "$overlay"
else
  tar -xzf "$ARCHIVE" -C "$LOCAL"
  overlay="$LOCAL/overlay"
fi
export PYTHONPATH="$overlay:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runtime_key="$(sha256sum model/light_vae.py script/eval_h3_vae_ab.py | sha256sum | cut -c1-16)"
export TRITON_CACHE_DIR="$PERSIST/cache/compiled/h3-vae-ab-bf16-native-$runtime_key/triton"
export TORCHINDUCTOR_CACHE_DIR="$PERSIST/cache/compiled/h3-vae-ab-bf16-native-$runtime_key/inductor"
if [[ -n "$startup_root" ]]; then
  export TRITON_CACHE_DIR="$startup_root/compiled/$runtime_key/triton"
  export TORCHINDUCTOR_CACHE_DIR="$startup_root/compiled/$runtime_key/inductor"
fi
date -u +%FT%TZ > "$RESULT/environment-restore-finish.txt"
"$PYTHON" - <<'PY'
import torch
from diffusers import AutoencoderKLMiniMaxH3
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0)[0] >= 10
print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0)}, flush=True)
PY
if [[ "${LYNNREAL_H3_TAE_ONLY:-0}" == 1 ]]; then
  checkpoint="$PERSIST/models/taehv-011dfc2112197741c540e0bdd5b7b67bcc930771/taeh3.pth"
  mkdir -p "$(dirname "$checkpoint")"
  if [[ ! -s "$checkpoint" ]]; then
    cp /mnt/world-model/lynnreal-omni/taehv/011dfc2112197741c540e0bdd5b7b67bcc930771/taeh3.pth "$checkpoint.partial"
    echo "af92965c2d7986a89a757e7cccd26f9eeeff0c3f0d5495eb168aeb2d6d9be9ba  $checkpoint.partial" | sha256sum -c -
    mv "$checkpoint.partial" "$checkpoint"
  fi
  tae_args=(--source "$PERSIST/results/$LYNNREAL_H3_RESUME_JOB/eval" --output "$RESULT/eval" --checkpoint "$checkpoint")
  [[ -z "${LYNNREAL_H3_CASE:-}" ]] || tae_args+=(--case "$LYNNREAL_H3_CASE")
  "$PYTHON" -u script/eval_h3_tae.py "${tae_args[@]}"
  date -u +%FT%TZ > "$RESULT/bootstrap-finish.txt"
  printf 'complete\n' > "$RESULT/READY"
  exit 0
fi
if [[ -n "${LYNNREAL_H3_RESUME_JOB:-}" ]]; then
  [[ "$LYNNREAL_H3_RESUME_JOB" =~ ^dlc[a-z0-9]+$ ]]
  previous="$PERSIST/results/$LYNNREAL_H3_RESUME_JOB"
  test -s "$previous/eval/identity.json"
  # Copy to the new job; preserve the original evidence and its completion marker.
  mkdir -p "$RESULT/eval"
  cp -a "$previous/eval/." "$RESULT/eval/"
  rm -f "$RESULT/eval/READY"
  printf '%s\n' "$LYNNREAL_H3_RESUME_JOB" > "$RESULT/resumed-from-job.txt"
fi
args=(--h3 "$H3" --light-vae "$LIGHT" --output "$RESULT/eval" --variant "${LYNNREAL_H3_VARIANT:-both}")
if [[ "${LYNNREAL_H3_RETRY_CHECK:-0}" == 1 ]]; then
  test -n "${LYNNREAL_H3_RESUME_JOB:-}"
  "$PYTHON" script/check_h3_retry.py prepare --results "$RESULT/eval" --case "$LYNNREAL_H3_CASE"
fi
[[ -z "${LYNNREAL_H3_CASE:-}" ]] || args+=(--case "$LYNNREAL_H3_CASE")
"$PYTHON" -u script/eval_h3_vae_ab.py "${args[@]}" --preflight
"$PYTHON" -u script/eval_h3_vae_ab.py "${args[@]}"
if [[ "${LYNNREAL_H3_RETRY_CHECK:-0}" == 1 ]]; then
  "$PYTHON" script/check_h3_retry.py verify --results "$RESULT/eval" --case "$LYNNREAL_H3_CASE"
fi
if [[ "${LYNNREAL_H3_VARIANT:-both}" == both ]]; then
  "$PYTHON" script/report_h3_vae_ab.py --results "$RESULT/eval"
fi
date -u +%FT%TZ > "$RESULT/bootstrap-finish.txt"
printf 'complete\n' > "$RESULT/READY"
