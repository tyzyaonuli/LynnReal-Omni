#!/usr/bin/env bash
set -Eeuo pipefail
readonly ROOT="$(pwd)"
readonly PERSIST=/cpfs/world-model/lynnreal-omni
readonly RESULT="$PERSIST/results/${DLC_JOB_ID:?DLC_JOB_ID required}"
readonly OSS_RESULT="/mnt/world-model/results/lynnreal-omni/$DLC_JOB_ID"
readonly H3="$PERSIST/models/h3-bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
readonly LIGHT="$PERSIST/models/light-vae-e453444c1a52b73a0c5eb0023d208c473c2bb26a"
readonly ARCHIVE="$PERSIST/cache/env/py312-torch2121-cu130-lynnreal-0384eca3d5d7482d.tar.gz"
readonly LOCAL="/local/h3-vae-ab/$DLC_JOB_ID"
readonly PYTHON=/opt/minwm/venv/bin/python
mkdir -p "$RESULT" "$OSS_RESULT" "$LOCAL"
exec > >(tee -a "$RESULT/bootstrap.log") 2>&1
finish() {
  local rc=$?
  if ((rc != 0)); then printf '%s\n' "$rc" > "$RESULT/FAILED"; fi
  cp -a "$RESULT/." "$OSS_RESULT/"
}
trap finish EXIT
date -u +%FT%TZ > "$RESULT/bootstrap-start.txt"
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
tar -xzf "$ARCHIVE" -C "$LOCAL"
export PYTHONPATH="$LOCAL/overlay:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runtime_key="$(sha256sum model/light_vae.py script/eval_h3_vae_ab.py | sha256sum | cut -c1-16)"
export TRITON_CACHE_DIR="$PERSIST/cache/compiled/h3-vae-ab-bf16-native-$runtime_key/triton"
export TORCHINDUCTOR_CACHE_DIR="$PERSIST/cache/compiled/h3-vae-ab-bf16-native-$runtime_key/inductor"
date -u +%FT%TZ > "$RESULT/environment-restore-finish.txt"
"$PYTHON" - <<'PY'
import torch
from diffusers import AutoencoderKLMiniMaxH3
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0)[0] >= 10
print({'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0)}, flush=True)
PY
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
[[ -z "${LYNNREAL_H3_CASE:-}" ]] || args+=(--case "$LYNNREAL_H3_CASE")
"$PYTHON" -u script/eval_h3_vae_ab.py "${args[@]}" --preflight
"$PYTHON" -u script/eval_h3_vae_ab.py "${args[@]}"
if [[ "${LYNNREAL_H3_VARIANT:-both}" == both ]]; then
  "$PYTHON" script/report_h3_vae_ab.py --results "$RESULT/eval"
fi
date -u +%FT%TZ > "$RESULT/bootstrap-finish.txt"
printf 'complete\n' > "$RESULT/READY"
