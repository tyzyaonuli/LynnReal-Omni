#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="${ALIYUN_PROFILE:-shengdong}"
MODE="${1:-}"
ACTION="${2:---dry-run}"
MODEL_SET="${3:-lynnreal}"
[[ "$MODEL_SET" == "lynnreal" || "$MODEL_SET" == "--h3-fl2va" || "$MODEL_SET" == "--h3-startup" || "$MODEL_SET" == "--h3-recover" ]] || { echo "invalid model selection" >&2; exit 2; }
if [[ "$MODEL_SET" == "--h3-recover" ]]; then
  [[ "$MODE" == --sync && "${H3_RECOVER_JOB:-}" =~ ^dlc[a-z0-9]+$ ]] || { echo "recovery requires --sync and H3_RECOVER_JOB" >&2; exit 2; }
  git diff --quiet && git diff --cached --quiet || { echo "commit recovery code first" >&2; exit 2; }
fi
if [[ "$MODEL_SET" == "--h3-startup" ]]; then
  [[ "$MODE" == --sync && "${H3_STARTUP_MODE:-}" =~ ^(cold|warm)$ && "${H3_STARTUP_EXPERIMENT:-}" =~ ^[a-z0-9-]+$ ]] || { echo "startup requires --sync and H3_STARTUP_MODE/EXPERIMENT" >&2; exit 2; }
  git diff --quiet && git diff --cached --quiet || { echo "commit startup code first" >&2; exit 2; }
fi
PAI_SESSION_ID="${PAI_SESSION_ID:-$(date +%s)-$RANDOM}"
USER_AGENT="${PAI_USER_AGENT:-AlibabaCloud-Agent-Skills/alibabacloud-pai-dlc-job/$PAI_SESSION_ID}"
THAILAND_REGION="ap-southeast-7"
THAILAND_OSS_URI="oss://leap-worldmodel-thailand.oss-ap-southeast-7.aliyuncs.com/world-model/"
THAILAND_OSS_CLI="oss://leap-worldmodel-thailand/world-model"
CPFS_ECS_DATASET_ID="d-qs9w70tf7ss5yw78vc"
INSTANCE_TYPE="ecs.g6.4xlarge"
MAX_MINUTES="${PAI_MAX_MINUTES:-720}"

if [[ "$MODE" != "--stage" && "$MODE" != "--sync" ]]; then
  echo "usage: bash script/aliyun_thailand_model_cache_submit.sh --stage|--sync [--dry-run|--submit]" >&2
  exit 2
fi
if [[ "$ACTION" != "--dry-run" && "$ACTION" != "--submit" ]]; then
  echo "usage: bash script/aliyun_thailand_model_cache_submit.sh --stage|--sync [--dry-run|--submit]" >&2
  exit 2
fi

stage_script="script/aliyun_modelscope_to_hangzhou_oss.py"
sync_script="script/aliyun_thailand_cpfs_model_sync.py"
downloader="script/modelscope_hz_multipart_download.py"
manifest_builder="script/prepare_aliyun_thailand_model_manifests.py"
for path in "$stage_script" "$sync_script" "$downloader" "$manifest_builder"; do test -f "$path"; done
bundle_hash="$(sha256sum "$stage_script" "$sync_script" "$downloader" "$manifest_builder" | sha256sum | cut -c1-12)"
if [[ "$MODEL_SET" == "--h3-fl2va" ]]; then
  bundle_hash="h3-fl2va-$({ sha256sum "$stage_script" "$downloader" "$sync_script" script/prepare_h3_vae_ab.py; find eval/h3_vae_ab -type f -print0 | sort -z | xargs -0 sha256sum; } | sha256sum | cut -c1-12)"
fi
bundle_name="model-cache-$bundle_hash.tar.gz"
if [[ "$MODEL_SET" == "--h3-startup" ]]; then
  bundle_hash="h3-startup-$(git rev-parse --short=12 HEAD)"
  bundle_name="model-cache-$bundle_hash.tar.gz"
fi
bundle_uri="$THAILAND_OSS_CLI/code/lynnreal-omni/$bundle_name"
if [[ "$MODEL_SET" == "--h3-recover" ]]; then
  bundle_name="h3-recover-$(git rev-parse --short=12 HEAD).tar.gz"
  bundle_uri="$THAILAND_OSS_CLI/code/lynnreal-omni/$bundle_name"
fi

account_id="$(aliyun --profile "$PROFILE" --connect-timeout 15 --read-timeout 30 \
  --retry-count 2 --user-agent "$USER_AGENT" sts GetCallerIdentity | jq -r .AccountId)"
role_arn="acs:ram::$account_id:role/zing-thailand-training"
credential_config="$(jq -nc --arg account "$account_id" --arg arn "$role_arn" '{EnableCredentialInject:true,AliyunEnvRoleKey:"zing-runtime",CredentialConfigItems:[{Key:"zing-runtime",Type:"Role",Roles:[{AssumeRoleFor:$account,RoleType:"service",RoleArn:$arn}]}]}')"

if [[ "$MODE" == "--stage" ]]; then
  REGION="cn-hangzhou"
  WORKSPACE_ID="660210"
  IMAGE="dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai/python:3.10.19-cpu-ubuntu22.04-de657118-1764321500"
  job_name="lynnreal-model-stage-$bundle_hash"
  user_command="set -euo pipefail; mkdir -p /workspace/lynnreal-model-cache; tar -xzf /mnt/world-model/code/lynnreal-omni/$bundle_name -C /workspace/lynnreal-model-cache; cd /workspace/lynnreal-model-cache; python -c 'import requests'; python script/aliyun_modelscope_to_hangzhou_oss.py --manifest-dir vendor --stage-root /mnt/hangzhou-stage/lynnreal-omni/staged-models-crr"
  data_sources="$(jq -nc --arg code "$THAILAND_OSS_URI" --arg stage "oss://codex-minmaxh3-cache-97318276.oss-cn-hangzhou-internal.aliyuncs.com/world-model/" '[{Uri:$code,MountPath:"/mnt/world-model",MountAccess:"RO"},{Uri:$stage,MountPath:"/mnt/hangzhou-stage",MountAccess:"RW"}]')"
  if [[ "$MODEL_SET" == "--h3-fl2va" ]]; then
    user_command="$user_command --h3-fl2va"
  fi
else
  REGION="$THAILAND_REGION"
  WORKSPACE_ID="385"
  IMAGE="dsw-registry-vpc.ap-southeast-7.cr.aliyuncs.com/pai/python:3.10.19-cpu-ubuntu22.04-de657118-1764321500"
  job_name="lynnreal-model-sync-$bundle_hash"
  user_command="set -euo pipefail; mkdir -p /workspace/lynnreal-model-cache; tar -xzf /mnt/world-model/code/lynnreal-omni/$bundle_name -C /workspace/lynnreal-model-cache; cd /workspace/lynnreal-model-cache; python script/aliyun_thailand_cpfs_model_sync.py --manifest-dir vendor --stage-root /mnt/world-model/lynnreal-omni/staged-models-crr --model-root /cpfs/world-model/lynnreal-omni/models"
  data_sources="$(jq -nc --arg oss "$THAILAND_OSS_URI" --arg cpfs "$CPFS_ECS_DATASET_ID" '[{Uri:$oss,MountPath:"/mnt/world-model",MountAccess:"RW"},{DataSourceId:$cpfs,DataSourceVersion:"v1",MountPath:"/cpfs",MountAccess:"RW"}]')"
  user_vpc="$(jq -nc '{VpcId:"vpc-0jono6k4fmswh8exr9n0s",SwitchId:"vsw-0johcol4d8ekvh2zc0e9t",SecurityGroupId:"sg-0jobja1grkgfne93adot",ExtendedCIDRs:["10.78.0.0/16"],DefaultRoute:"eth1"}')"
  if [[ "$MODEL_SET" == "--h3-fl2va" ]]; then
    user_command="set -euo pipefail; mkdir -p /workspace/lynnreal-model-cache; tar -xzf /mnt/world-model/code/lynnreal-omni/$bundle_name -C /workspace/lynnreal-model-cache; cd /workspace/lynnreal-model-cache; python script/prepare_h3_vae_ab.py --wait-for-stage-seconds 600 --output /mnt/world-model/results/lynnreal-omni/$job_name"
  fi
fi

if [[ "$MODEL_SET" == "--h3-startup" ]]; then
  job_name="h3-startup-$H3_STARTUP_EXPERIMENT-$H3_STARTUP_MODE"
  user_command="set -euo pipefail; mkdir -p /workspace/lynnreal-model-cache; tar -xzf /mnt/world-model/code/lynnreal-omni/$bundle_name -C /workspace/lynnreal-model-cache; cd /workspace/lynnreal-model-cache; export LYNNREAL_RELEASE_SHA=$(git rev-parse HEAD); python script/prepare_h3_startup_cache.py --experiment $H3_STARTUP_EXPERIMENT --mode $H3_STARTUP_MODE --output /mnt/world-model/results/lynnreal-omni/$job_name"
fi

if [[ "$MODEL_SET" == "--h3-recover" ]]; then
    job_name="h3-metadata-recover-$H3_RECOVER_JOB"
    user_command="set -euo pipefail; mkdir -p /workspace/h3-recover; tar -xzf /mnt/world-model/code/lynnreal-omni/$bundle_name -C /workspace/h3-recover; cd /workspace/h3-recover; python script/recover_h3_metadata.py --job $H3_RECOVER_JOB"
fi
if [[ "$ACTION" == "--submit" ]]; then
  bundle_dir="$(mktemp -d)"
  trap 'rm -rf "$bundle_dir"' EXIT
  mkdir -p "$bundle_dir/source/script" "$bundle_dir/source/vendor"
  cp "$stage_script" "$sync_script" "$downloader" "$bundle_dir/source/script/"
  if [[ "$MODEL_SET" == "--h3-recover" ]]; then
    cp script/recover_h3_metadata.py "$bundle_dir/source/script/"
  fi
  if [[ "$MODEL_SET" == "--h3-startup" ]]; then
    cp script/prepare_h3_startup_cache.py "$bundle_dir/source/script/"
  elif [[ "$MODEL_SET" == "--h3-fl2va" ]]; then
    cp -a eval/h3_vae_ab/staging/. "$bundle_dir/source/vendor/"
    cp script/prepare_h3_vae_ab.py "$bundle_dir/source/script/"
    mkdir -p "$bundle_dir/source/eval"
    cp -a eval/h3_vae_ab "$bundle_dir/source/eval/"
  elif [[ "$MODEL_SET" != "--h3-recover" ]]; then
    python3 "$manifest_builder" --output "$bundle_dir/source/vendor"
  fi
  tar -czf "$bundle_dir/$bundle_name" -C "$bundle_dir/source" .
  upload_path="$bundle_dir/$bundle_name"
  if command -v cygpath >/dev/null; then upload_path="$(cygpath -w "$upload_path")"; fi
  aliyun --profile "$PROFILE" --user-agent "$USER_AGENT" oss cp \
    "$upload_path" "$bundle_uri" --region "$THAILAND_REGION" \
    --endpoint "oss-$THAILAND_REGION.aliyuncs.com" --force >/dev/null
fi

job_specs="$(jq -nc --arg image "$IMAGE" --arg ecs "$INSTANCE_TYPE" '[{Type:"Worker",PodCount:1,EcsSpec:$ecs,Image:$image,RestartPolicy:"Never"}]')"
settings='{"Shell":"/bin/bash"}'
args=(pai-dlc create-job --display-name "$job_name" --job-type PyTorchJob \
  --workspace-id "$WORKSPACE_ID" --job-specs "$job_specs" --user-command "$user_command" \
  --credential-config "$credential_config" --data-sources "$data_sources")
if [[ "$MODE" == "--sync" ]]; then
  args+=(--user-vpc "$user_vpc")
fi
args+=(--settings "$settings" --accessibility PRIVATE --job-max-running-time-minutes "$MAX_MINUTES" \
  --user-agent "$USER_AGENT")

printf 'mode=%s region=%s workspace=%s instance=%s bundle=%s\n' \
  "$MODE" "$REGION" "$WORKSPACE_ID" "$INSTANCE_TYPE" "$bundle_uri" >&2
if [[ "$ACTION" == "--submit" ]]; then
  aliyun --profile "$PROFILE" --region "$REGION" --connect-timeout 15 --read-timeout 30 \
    --retry-count 0 "${args[@]}"
else
  aliyun --profile "$PROFILE" --region "$REGION" "${args[@]}" --cli-dry-run true
fi
