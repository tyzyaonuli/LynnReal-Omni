#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="${ALIYUN_PROFILE:-shengdong}"
REGION="ap-southeast-7"
WORKSPACE_ID="384"
QUOTA_ID="quota1betfyi1ilu"
CPFS_DATASET_ID="d-ak7t158lhzaq7kdwp3"
IMAGE="zing-thailand-registry-vpc.ap-southeast-7.cr.aliyuncs.com/zing/zing-training-env:h3-fast-ulysses-torch2121-fa34-f7351c75-r3"
ACR_INSTANCE_ID="cri-ndh9vimgnm77f4tt"
VPC_ID="vpc-0jono6k4fmswh8exr9n0s"
VSWITCH_ID="vsw-0johcol4d8ekvh2zc0e9t"
SECURITY_GROUP_ID="sg-0jobja1grkgfne93adot"
RAM_ROLE="zing-thailand-training"
RESOURCE_GPU="${PAI_RESOURCE_GPU:-1}"
RESOURCE_CPU="${PAI_RESOURCE_CPU:-23}"
RESOURCE_MEMORY="${PAI_RESOURCE_MEMORY:-225Gi}"
RESOURCE_SHARED_MEMORY="${PAI_RESOURCE_SHARED_MEMORY:-64Gi}"
MAX_MINUTES="${PAI_MAX_MINUTES:-360}"
PAI_SESSION_ID="${PAI_SESSION_ID:-$(date +%s)-$RANDOM}"
USER_AGENT="${PAI_USER_AGENT:-AlibabaCloud-Agent-Skills/alibabacloud-pai-dlc-job/$PAI_SESSION_ID}"
ACTION=""
RUN_MODE="speed-test"
EVAL_MANIFEST=""
EVAL_VARIANT=""
H3_CASE=""
H3_PREFLIGHT=""
H3_RUNTIME_RECEIPT=""
H3_STARTUP_RECEIPT=""
H3_RESUME_JOB=""
H3_VARIANT="both"
H3_RETRY_CHECK="0"
H3_TAE_ONLY="0"
RECONSTRUCTION=""

while (($#)); do
  case "$1" in
    --submit|--dry-run)
      [[ -z "$ACTION" ]] || {
        echo "do not combine or repeat --submit and --dry-run" >&2
        exit 2
      }
      ACTION="${1#--}"
      shift
      ;;
    --eval-manifest) EVAL_MANIFEST="${2:?--eval-manifest requires a path}"; shift 2 ;;
    --variant) EVAL_VARIANT="${2:?--variant requires standard or flash}"; shift 2 ;;
    --h3-vae-ab) RUN_MODE="h3-vae-ab"; shift ;;
    --vae-reconstruction) RUN_MODE="h3-vae-ab"; RECONSTRUCTION="${2:?requires prepared manifest path}"; shift 2 ;;
    --case) H3_CASE="${2:?--case requires an ID}"; shift 2 ;;
    --resume-job) H3_RESUME_JOB="${2:?--resume-job requires a previous job ID}"; shift 2 ;;
    --retry-check) H3_RETRY_CHECK="1"; shift ;;
    --tae-only) H3_TAE_ONLY="1"; shift ;;
    --h3-variant) H3_VARIANT="${2:?--h3-variant requires baseline, light or both}"; shift 2 ;;
    --preflight-receipt) H3_PREFLIGHT="${2:?--preflight-receipt requires a JSON path}"; shift 2 ;;
    --runtime-receipt) H3_RUNTIME_RECEIPT="${2:?--runtime-receipt requires a JSON path}"; shift 2 ;;
    --startup-receipt) H3_STARTUP_RECEIPT="${2:?--startup-receipt requires a JSON path}"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash script/aliyun_thailand_b300_submit.sh [--dry-run|--submit] --h3-vae-ab \
    --preflight-receipt weights.json --runtime-receipt runtime.json \
    [--case CASE_ID] [--h3-variant baseline|light|both] [--resume-job JOB_ID] [--retry-check]
  bash script/aliyun_thailand_b300_submit.sh [--dry-run|--submit]
  bash script/aliyun_thailand_b300_submit.sh [--dry-run|--submit] \
    --eval-manifest prompts.jsonl --variant standard|flash

The default runs the official aligned Standard and Flash speed tests. Eval mode
accepts JSONL objects with id, prompt and optional seed, loads one model once,
and writes samples plus results.jsonl to the job's CPFS/OSS result directory.
EOF
      exit 0
      ;;
    *) echo "unexpected argument: $1" >&2; exit 2 ;;
  esac
done
SUBMIT=false
[[ "$ACTION" == "submit" ]] && SUBMIT=true
if [[ -n "$EVAL_MANIFEST" || -n "$EVAL_VARIANT" ]]; then
  [[ "$RUN_MODE" != "h3-vae-ab" ]] || { echo "H3 A/B cannot use Standard/Flash flags" >&2; exit 2; }
  [[ -n "$EVAL_MANIFEST" && "$EVAL_VARIANT" =~ ^(standard|flash)$ ]] || {
    echo "eval mode requires --eval-manifest and --variant standard|flash" >&2
    exit 2
  }
  EVAL_MANIFEST="$(cd "$(dirname "$EVAL_MANIFEST")" && pwd)/$(basename "$EVAL_MANIFEST")"
  test -f "$EVAL_MANIFEST"
  RUN_MODE="eval"
fi
[[ "$RESOURCE_GPU" == "1" ]] || { echo "this inference entry requires exactly one GPU" >&2; exit 2; }
for command in aliyun curl git jq python3 sha256sum tar; do
  command -v "$command" >/dev/null || {
    echo "required command not found: $command" >&2
    exit 2
  }
done
if [[ "$RUN_MODE" == "h3-vae-ab" ]]; then
  if [[ "$H3_TAE_ONLY" == 1 ]]; then
    [[ -n "$H3_RESUME_JOB" && "$H3_RETRY_CHECK" == 0 ]] || { echo "TAE-only requires a source job and no retry injection" >&2; exit 2; }
  fi
  if [[ "$H3_RETRY_CHECK" == 1 ]]; then
    [[ -n "$H3_RESUME_JOB" && -n "$H3_CASE" && "$H3_VARIANT" == both ]] || { echo "retry check requires resume job, case and both variants" >&2; exit 2; }
  fi
  [[ "$H3_VARIANT" =~ ^(baseline|light|both)$ ]] || { echo "invalid H3 variant" >&2; exit 2; }
  [[ -z "$H3_RESUME_JOB" || "$H3_RESUME_JOB" =~ ^dlc[a-z0-9]+$ ]] || { echo "invalid resume job ID" >&2; exit 2; }
  [[ -n "$H3_PREFLIGHT" ]] || { echo "H3 A/B requires a completed CPU preflight receipt" >&2; exit 2; }
  [[ -n "$H3_RUNTIME_RECEIPT" ]] || { echo "H3 A/B requires a CPU runtime receipt" >&2; exit 2; }
  python3 - "$H3_PREFLIGHT" "$H3_RUNTIME_RECEIPT" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
m = json.loads(Path('eval/h3_vae_ab/manifest.json').read_text(encoding="utf-8"))
assert r['complete'] and not r['gpu_used']
assert r['models']['h3']['revision'] == m['model']['revision']
assert r['models']['light']['revision'] == m['light_vae']['revision']
assert r['environment_archives']
e = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert e['status'] == 'passed' and not e['gpu_used'] and e['sigma_schedule_verified']
assert e['archive'] in r['environment_archives']
PY
  dry_args=(--dry-run --output /tmp/h3-vae-ab-dry-run)
  [[ -z "$H3_CASE" ]] || dry_args+=(--case "$H3_CASE")
  if [[ -n "$RECONSTRUCTION" ]]; then
    [[ "$H3_TAE_ONLY" == 0 && "$H3_RETRY_CHECK" == 0 && -z "$H3_STARTUP_RECEIPT" ]] || { echo 'incompatible reconstruction flags' >&2; exit 2; }
    python3 script/eval_vae_reconstruction.py "${dry_args[@]}" --manifest "$RECONSTRUCTION" --sources /unused
  else
    python3 script/eval_h3_vae_ab.py "${dry_args[@]}"
  fi
fi
git diff --cached --quiet || { echo "staged changes are not supported by this submitter" >&2; exit 2; }
git diff --quiet || {
  echo "tracked changes are not supported; commit them so release_sha matches the bundle" >&2
  exit 2
}

release_sha="$(git rev-parse HEAD)"
if [[ -n "$H3_STARTUP_RECEIPT" ]]; then
  [[ "$RUN_MODE" == h3-vae-ab && -z "$H3_RESUME_JOB" ]] || { echo "startup comparison requires fresh H3 results" >&2; exit 2; }
  python3 - "$H3_STARTUP_RECEIPT" "$release_sha" <<'PY'
import json,sys,re
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert r['complete'] and not r['gpu_used'] and r['code_sha']==sys.argv[2]
assert r['mode'] in ('cold','warm') and re.fullmatch('[a-z0-9-]+',r['experiment'])
assert r['root']=='/cpfs/world-model/lynnreal-omni/cache/startup-ab/'+r['experiment']
PY
fi
bootstrap="script/aliyun_thailand_b300_inference_bootstrap.sh"
[[ "$RUN_MODE" != "h3-vae-ab" ]] || bootstrap="script/aliyun_thailand_h3_vae_ab_bootstrap.sh"
downloader="script/modelscope_hz_multipart_download.py"
manifest_builder="script/prepare_aliyun_thailand_model_manifests.py"
kernel_validator="script/validate_int8_gemm.py"
eval_runner="script/eval_t2v.py"
kernel_sources=(model/int8_gemm.py model/int8_tma.py)
test -f "$bootstrap" && test -f "$downloader" && test -f "$manifest_builder" \
  && test -f "$kernel_validator" && test -f "$eval_runner"
bootstrap_sha="$(sha256sum "$bootstrap" "$downloader" "$manifest_builder" "$kernel_validator" "$eval_runner" "${kernel_sources[@]}" | sha256sum | cut -c1-12)"
diffusers_url="https://github.com/huggingface/diffusers/archive/abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc.tar.gz"
diffusers_sha="584a47eb49eaf60fda2843317708eacc6a7d9622f1630d9badb0b33fc480aaba"
bundle_id="$release_sha-$bootstrap_sha"
if [[ "$RUN_MODE" == "h3-vae-ab" ]]; then
  bundle_id="$release_sha-$bootstrap_sha-h3-vae-ab-${H3_CASE:-all}"
  [[ -z "$RECONSTRUCTION" ]] || bundle_id="$bundle_id-reconstruction-$(sha256sum "$RECONSTRUCTION" | cut -c1-12)"
  [[ -z "$H3_STARTUP_RECEIPT" ]] || bundle_id="$bundle_id-startup-$(sha256sum "$H3_STARTUP_RECEIPT" | cut -c1-12)"
fi
if [[ "$RUN_MODE" == "eval" ]]; then
  python3 "$eval_runner" --manifest "$EVAL_MANIFEST" --variant "$EVAL_VARIANT" \
    --output /tmp/lynnreal-eval-dry-run --dry-run >/dev/null
  eval_manifest_sha="$(sha256sum "$EVAL_MANIFEST" | cut -c1-12)"
  bundle_id="$bundle_id-eval-$EVAL_VARIANT-$eval_manifest_sha"
fi
bundle_uri="oss://leap-worldmodel-thailand/world-model/code/lynnreal-omni/$bundle_id.tar.gz"
job_name="${JOB_NAME:-lynnreal-b300-$RUN_MODE-${EVAL_VARIANT:-both}-${release_sha:0:8}-$(date -u +%Y%m%d-%H%M%S)}"

account_id="$(aliyun --profile "$PROFILE" --region "$REGION" --connect-timeout 15 \
  --read-timeout 30 --retry-count 2 --user-agent "$USER_AGENT" \
  sts GetCallerIdentity | jq -r .AccountId)"
role_arn="acs:ram::$account_id:role/$RAM_ROLE"
credential_config="$(jq -nc --arg account "$account_id" --arg arn "$role_arn" '{EnableCredentialInject:true,AliyunEnvRoleKey:"zing-runtime",CredentialConfigItems:[{Key:"zing-runtime",Type:"Role",Roles:[{AssumeRoleFor:$account,RoleType:"service",RoleArn:$arn}]}]}')"
data_sources="$(jq -nc --arg cpfs "$CPFS_DATASET_ID" '[{Uri:"oss://leap-worldmodel-thailand.oss-ap-southeast-7-internal.aliyuncs.com/world-model/",MountPath:"/mnt/world-model",MountAccess:"RW"},{DataSourceId:$cpfs,MountPath:"/cpfs",MountAccess:"RW"}]')"
user_vpc="$(jq -nc --arg vpc "$VPC_ID" --arg switch "$VSWITCH_ID" --arg sg "$SECURITY_GROUP_ID" '{VpcId:$vpc,SwitchId:$switch,SecurityGroupId:$sg,ExtendedCIDRs:["10.78.0.0/16"],DefaultRoute:"eth1"}')"
settings='{"Shell":"/bin/bash","EnableRDMA":false,"EnableSanityCheck":true,"OversoldType":"ForbiddenQuotaOverSold"}'
eval_exports=""
if [[ "$RUN_MODE" == "eval" ]]; then
  eval_exports="export LYNNREAL_EVAL_MANIFEST=/workspace/LynnReal-Omni/eval/input.jsonl LYNNREAL_EVAL_VARIANT=$EVAL_VARIANT;"
fi
user_command="set -euo pipefail; mkdir -p /workspace/LynnReal-Omni; tar -xzf /mnt/world-model/code/lynnreal-omni/$bundle_id.tar.gz -C /workspace/LynnReal-Omni; cd /workspace/LynnReal-Omni; export LYNNREAL_RELEASE_SHA=$release_sha LYNNREAL_RUN_MODE=$RUN_MODE; $eval_exports exec bash script/aliyun_thailand_b300_inference_bootstrap.sh"
if [[ "$RUN_MODE" == "h3-vae-ab" ]]; then
  [[ "$H3_CASE" =~ ^[a-z0-9-]*$ ]] || { echo "invalid H3 case ID" >&2; exit 2; }
  reconstruction_flag=0
  [[ -z "$RECONSTRUCTION" ]] || reconstruction_flag=1
  user_command="set -euo pipefail; mkdir -p /workspace/LynnReal-Omni; tar -xzf /mnt/world-model/code/lynnreal-omni/$bundle_id.tar.gz -C /workspace/LynnReal-Omni; cd /workspace/LynnReal-Omni; export LYNNREAL_RELEASE_SHA=$release_sha LYNNREAL_H3_CASE=$H3_CASE LYNNREAL_H3_VARIANT=$H3_VARIANT LYNNREAL_H3_RESUME_JOB=$H3_RESUME_JOB LYNNREAL_H3_RETRY_CHECK=$H3_RETRY_CHECK LYNNREAL_H3_TAE_ONLY=$H3_TAE_ONLY LYNNREAL_VAE_RECONSTRUCTION=$reconstruction_flag; exec bash $bootstrap"
fi

registry="${IMAGE%%/*}"
acr_username='<temporary-user>'
acr_password='<temporary-token>'
if [[ "$SUBMIT" == true ]]; then
  if [[ "${LYNNREAL_REUSE_BUNDLE:-false}" == true ]]; then
    aliyun --profile "$PROFILE" --user-agent "$USER_AGENT" oss stat "$bundle_uri" \
      --region "$REGION" --endpoint "oss-$REGION.aliyuncs.com" >/dev/null
  else
    bundle_dir="$(mktemp -d)"
    trap 'rm -rf "$bundle_dir"' EXIT
    mkdir -p "$bundle_dir/source"
    git archive "$release_sha" | tar -x -C "$bundle_dir/source"
    if [[ "$RUN_MODE" == "h3-vae-ab" ]]; then
      cp "$H3_PREFLIGHT" "$bundle_dir/source/eval/h3_vae_ab/preflight-receipt.json"
      cp "$H3_RUNTIME_RECEIPT" "$bundle_dir/source/eval/h3_vae_ab/runtime-receipt.json"
      [[ -z "$RECONSTRUCTION" ]] || cp "$RECONSTRUCTION" "$bundle_dir/source/eval/h3_vae_ab/reconstruction-manifest.json"
      [[ -z "$H3_STARTUP_RECEIPT" ]] || cp "$H3_STARTUP_RECEIPT" "$bundle_dir/source/eval/h3_vae_ab/startup-receipt.json"
    fi
    cp "$bootstrap" "$bundle_dir/source/$bootstrap"
    cp "$downloader" "$bundle_dir/source/$downloader"
    cp "$manifest_builder" "$bundle_dir/source/$manifest_builder"
    cp "$kernel_validator" "$bundle_dir/source/$kernel_validator"
    cp "$eval_runner" "$bundle_dir/source/$eval_runner"
    cp "${kernel_sources[@]}" "$bundle_dir/source/model/"
    if [[ "$RUN_MODE" == "eval" ]]; then
      mkdir -p "$bundle_dir/source/eval"
      cp "$EVAL_MANIFEST" "$bundle_dir/source/eval/input.jsonl"
    fi
    mkdir -p "$bundle_dir/source/vendor"
    if [[ "$RUN_MODE" != "h3-vae-ab" ]]; then
    if [[ -s /tmp/diffusers-abc5e9bf71fd.tar.gz ]] && \
        [[ "$(sha256sum /tmp/diffusers-abc5e9bf71fd.tar.gz | cut -d' ' -f1)" == "$diffusers_sha" ]]; then
      cp /tmp/diffusers-abc5e9bf71fd.tar.gz "$bundle_dir/source/vendor/diffusers-abc5e9bf71fd.tar.gz"
    else
      curl -fL --retry 3 --connect-timeout 15 "$diffusers_url" \
        -o "$bundle_dir/source/vendor/diffusers-abc5e9bf71fd.tar.gz"
    fi
    test "$(sha256sum "$bundle_dir/source/vendor/diffusers-abc5e9bf71fd.tar.gz" | cut -d' ' -f1)" = "$diffusers_sha"
    if [[ -n "${LYNNREAL_VENDOR_CACHE:-}" ]]; then
      test -f "$LYNNREAL_VENDOR_CACHE/standard.tsv"
      test -f "$LYNNREAL_VENDOR_CACHE/flash.tsv"
      test -f "$LYNNREAL_VENDOR_CACHE/light-vae.tsv"
      test -f "$LYNNREAL_VENDOR_CACHE/standard-hf-tree.json"
      test -d "$LYNNREAL_VENDOR_CACHE/model-seed"
      test "$(awk -F '\t' 'NR>1 {n++; b+=$2} END {printf "%d:%.0f", n, b}' "$LYNNREAL_VENDOR_CACHE/standard.tsv")" = "50:77301658214"
      test "$(awk -F '\t' 'NR>1 {n++; b+=$2} END {printf "%d:%.0f", n, b}' "$LYNNREAL_VENDOR_CACHE/flash.tsv")" = "32:39742516911"
      test "$(awk -F '\t' 'NR>1 {n++; b+=$2} END {printf "%d:%.0f", n, b}' "$LYNNREAL_VENDOR_CACHE/light-vae.tsv")" = "7:7729847284"
      cp -a "$LYNNREAL_VENDOR_CACHE/." "$bundle_dir/source/vendor/"
    else
      python3 "$manifest_builder" --output "$bundle_dir/source/vendor"
    fi
    fi
    tar -czf "$bundle_dir/source.tar.gz" -C "$bundle_dir/source" .
    upload_path="$bundle_dir/source.tar.gz"
    if command -v cygpath >/dev/null; then upload_path="$(cygpath -w "$upload_path")"; fi
    aliyun --profile "$PROFILE" --user-agent "$USER_AGENT" oss cp \
      "$upload_path" "$bundle_uri" --region "$REGION" \
      --endpoint "oss-$REGION.aliyuncs.com" --force >/dev/null
  fi
  acr_auth="$(aliyun --profile "$PROFILE" --region "$REGION" --connect-timeout 15 \
    --read-timeout 30 --retry-count 2 --user-agent "$USER_AGENT" \
    cr get-authorization-token --instance-id "$ACR_INSTANCE_ID")"
  acr_username="$(jq -r '.TempUsername // empty' <<<"$acr_auth")"
  acr_password="$(jq -r '.AuthorizationToken // empty' <<<"$acr_auth")"
  unset acr_auth
  [[ -n "$acr_username" && -n "$acr_password" ]]
fi

job_specs="$(jq -nc --arg image "$IMAGE" --arg registry "$registry" --arg username "$acr_username" --arg password "$acr_password" --arg cpu "$RESOURCE_CPU" --arg gpu "$RESOURCE_GPU" --arg memory "$RESOURCE_MEMORY" --arg shm "$RESOURCE_SHARED_MEMORY" '[{Type:"Worker",Image:$image,ImageConfig:{DockerRegistry:$registry,Username:$username,Password:$password},PodCount:1,ResourceConfig:{CPU:$cpu,GPU:$gpu,Memory:$memory,SharedMemory:$shm},RestartPolicy:"Never"}]')"
args=(pai-dlc create-job --display-name "$job_name" --job-type PyTorchJob \
  --workspace-id "$WORKSPACE_ID" --resource-id "$QUOTA_ID" --job-specs "$job_specs" \
  --user-command "$user_command" --credential-config "$credential_config" --data-sources "$data_sources" \
  --user-vpc "$user_vpc" --settings "$settings" --accessibility PRIVATE \
  --job-max-running-time-minutes "$MAX_MINUTES" --user-agent "$USER_AGENT")

variant_label="${EVAL_VARIANT:-standard+flash}"
[[ "$RUN_MODE" != "h3-vae-ab" ]] || variant_label="H3-$H3_VARIANT"
printf 'job_name=%s\nmode=%s variant=%s\nresource=%sGPU/%sCPU/%s/shm-%s\nbundle=%s\n' \
  "$job_name" "$RUN_MODE" "$variant_label" "$RESOURCE_GPU" "$RESOURCE_CPU" \
  "$RESOURCE_MEMORY" "$RESOURCE_SHARED_MEMORY" "$bundle_uri" >&2
if [[ "$SUBMIT" == true ]]; then
  aliyun --profile "$PROFILE" --region "$REGION" --connect-timeout 15 --read-timeout 30 \
    "${args[@]}" --retry-count 0
else
  aliyun --profile "$PROFILE" --region "$REGION" "${args[@]}" --cli-dry-run true
fi
