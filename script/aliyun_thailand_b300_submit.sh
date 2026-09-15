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
SUBMIT=false
RUN_MODE="speed-test"
EVAL_MANIFEST=""
EVAL_VARIANT=""

while (($#)); do
  case "$1" in
    --submit) SUBMIT=true; shift ;;
    --dry-run) shift ;;
    --eval-manifest) EVAL_MANIFEST="${2:?--eval-manifest requires a path}"; shift 2 ;;
    --variant) EVAL_VARIANT="${2:?--variant requires standard or flash}"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage:
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
if [[ -n "$EVAL_MANIFEST" || -n "$EVAL_VARIANT" ]]; then
  [[ -n "$EVAL_MANIFEST" && "$EVAL_VARIANT" =~ ^(standard|flash)$ ]] || {
    echo "eval mode requires --eval-manifest and --variant standard|flash" >&2
    exit 2
  }
  EVAL_MANIFEST="$(cd "$(dirname "$EVAL_MANIFEST")" && pwd)/$(basename "$EVAL_MANIFEST")"
  test -f "$EVAL_MANIFEST"
  RUN_MODE="eval"
fi
[[ "$RESOURCE_GPU" == "1" ]] || { echo "this inference entry requires exactly one GPU" >&2; exit 2; }
command -v aliyun >/dev/null
command -v git >/dev/null
command -v jq >/dev/null
git diff --cached --quiet || { echo "staged changes are not supported by this submitter" >&2; exit 2; }
unexpected_changes="$(git diff --name-only | grep -Ev '^model/(int8_gemm|int8_tma)\.py$' || true)"
[[ -z "$unexpected_changes" ]] || { printf 'unexpected tracked changes:\n%s\n' "$unexpected_changes" >&2; exit 2; }

release_sha="$(git rev-parse HEAD)"
bootstrap="script/aliyun_thailand_b300_inference_bootstrap.sh"
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
if [[ "$RUN_MODE" == "eval" ]]; then
  python3 "$eval_runner" --manifest "$EVAL_MANIFEST" --variant "$EVAL_VARIANT" \
    --output /tmp/lynnreal-eval-dry-run --dry-run >/dev/null
  eval_manifest_sha="$(sha256sum "$EVAL_MANIFEST" | cut -c1-12)"
  bundle_id="$bundle_id-eval-$EVAL_VARIANT-$eval_manifest_sha"
fi
bundle_uri="oss://leap-worldmodel-thailand/world-model/code/lynnreal-omni/$bundle_id.tar.gz"
job_name="${JOB_NAME:-lynnreal-b300-$RUN_MODE-${EVAL_VARIANT:-both}-${release_sha:0:8}-$(date -u +%Y%m%d-%H%M%S)}"

account_id="$(aliyun --profile "$PROFILE" --region "$REGION" --user-agent "$USER_AGENT" \
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
    tar -czf "$bundle_dir/source.tar.gz" -C "$bundle_dir/source" .
    aliyun --profile "$PROFILE" --user-agent "$USER_AGENT" oss cp \
      "$bundle_dir/source.tar.gz" "$bundle_uri" --region "$REGION" \
      --endpoint "oss-$REGION.aliyuncs.com" --force >/dev/null
  fi
  acr_auth="$(aliyun --profile "$PROFILE" --region "$REGION" --user-agent "$USER_AGENT" \
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

printf 'job_name=%s\nmode=%s variant=%s\nresource=%sGPU/%sCPU/%s/shm-%s\nbundle=%s\n' \
  "$job_name" "$RUN_MODE" "${EVAL_VARIANT:-standard+flash}" "$RESOURCE_GPU" "$RESOURCE_CPU" \
  "$RESOURCE_MEMORY" "$RESOURCE_SHARED_MEMORY" "$bundle_uri" >&2
if [[ "$SUBMIT" == true ]]; then
  aliyun --profile "$PROFILE" --region "$REGION" --connect-timeout 15 --read-timeout 30 \
    "${args[@]}" --retry-count 0
else
  aliyun --profile "$PROFILE" --region "$REGION" "${args[@]}" --cli-dry-run true
fi
