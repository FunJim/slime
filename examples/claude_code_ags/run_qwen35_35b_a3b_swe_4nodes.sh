#!/usr/bin/env bash
# End-to-end SWE coding-agent RL with Claude Code on AGS, using rollout_buffer
# on a 4-node Ray cluster. Run from a long-lived shell / tmux session on the
# Ray head node.

# Best-effort cleanup so a rerun does not collide with stale workers/services.
pkill -9 sglang || true
pkill -f "slime_plugins.rollout_buffer.buffer" || true
pkill -f "slime_plugins/rollout_buffer/buffer.py" || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3
pkill -9 ray || true

set -ex

export PYTHONUNBUFFERED=1

EXP="${EXP:?set EXP to an experiment directory, e.g. /data_train/ericxjzheng/experiments/<name>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-/data_train/ericxjzheng/workspace/slime}"

# ============ cluster size ============
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${MLP_WORKER_NUM:-4}}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
TOTAL_NUM_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))

# ============ model parallelism ============
export TP_SIZE="${TP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
export ETP_SIZE="${ETP_SIZE:-1}"

# ============ rollout engine ============
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${TOTAL_NUM_GPUS}}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-8}"
ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-4}"
ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-8}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.75}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

# ============ context length ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-96000}"
MAX_GEN_LEN="${MAX_GEN_LEN:-32768}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-${MAX_CONTEXT_LEN}}"

# ============ eval ============
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
EVAL_DATA="${EVAL_DATA:-/data_train/ericxjzheng/data/SWE-bench_Verified_slime_rl_eval/swebench_verified_from_yulei_filtered_slime.jsonl}"
SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"

# ============ paths — override before launching ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/data_train/ericxjzheng/models/Qwen3.5-35B-A3B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/data_train/ericxjzheng/models/Qwen3.5-35B-A3B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/data_train/ericxjzheng/data/SWE-rebench-filtered/filtered.jsonl}"

EXP_TAG="${EXP_TAG:-claude_code_ags_qwen35_35b_a3b_4nodes}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${EXP}/runs/${EXP_TAG}_${STAMP}}"

# ============ logging/artifacts ============
LOG_DIR="${RUN_ROOT}"
mkdir -p "${LOG_DIR}/rollout_dumps" "${LOG_DIR}/ags_artifacts"
LOG_FILE="${LOG_DIR}/run.log"
BUFFER_LOG_FILE="${LOG_DIR}/rollout_buffer.log"
export TRAJECTORY_DUMP_DIR="${TRAJECTORY_DUMP_DIR:-${LOG_DIR}/ags_artifacts}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXP_TAG}}"
echo "======================================================================"
echo "Training log:       ${LOG_FILE}"
echo "Rollout buffer log: ${BUFFER_LOG_FILE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "======================================================================"

# ============ ray cluster network ============
# Set MASTER_ADDR before AGS/SWE blocks: ADAPTER_PUBLIC_HOST below falls back to it.
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"

# ============ SWE / Claude Code / AGS rollout knobs ============
export SWE_AGENT="${SWE_AGENT:-claude_code}"

# AGS uses the E2B-compatible SDK surface. Export E2B_API_KEY in the launch
# environment (for Tencent AGS this is typically the AGS gateway key).
export E2B_DOMAIN="${E2B_DOMAIN:-ap-shanghai.tencentags.com}"
export AGS_BASE_TOOL="${AGS_BASE_TOOL:-sdt-3fzh6mv6}"
export AGS_IMAGE_REGISTRY_TYPE="${AGS_IMAGE_REGISTRY_TYPE:-enterprise}"
export AGS_SANDBOX_RESOURCES_JSON=${AGS_SANDBOX_RESOURCES_JSON:-'{"cpu":"4","memory":"16Gi"}'}

# ADAPTER_PUBLIC_HOST must be routable from inside the AGS sandbox (not 127.0.0.1).
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-1800}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
# false: grade in the agent sandbox; true: boot a second clean sandbox for grading.
export SWE_EVAL_ISOLATED_SANDBOX="${SWE_EVAL_ISOLATED_SANDBOX:-false}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-32}"
export SWE_BOOT_RETRIES="${SWE_BOOT_RETRIES:-10}"
export SWE_ROLLOUT_CONCURRENCY="${SWE_ROLLOUT_CONCURRENCY:-32}"

# # autoCompactWindow (80k) < MAX_CONTEXT_LEN (96k) so the CLI compacts before any
# # segment crosses the training-side cap. `investigator` is a read-only sub-agent.
# SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":80000}'
# AGENTS_JSON='{"investigator":{"description":"Searches the repo for relevant files before any edit","prompt":"You are an investigator sub-agent. Use Grep/Read/Glob to find every file relevant to the user task, then return a short bulleted summary. Do NOT edit anything.","tools":["Grep","Read","Glob"]}}'
# export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --agents '${AGENTS_JSON}' --disallowedTools WebFetch WebSearch"
export SLIME_AGENT_CC_MAX_TURNS="${SLIME_AGENT_CC_MAX_TURNS:-100}"
export SLIME_AGENT_CC_EXTRA_ARGS="${SLIME_AGENT_CC_EXTRA_ARGS:---max-turns ${SLIME_AGENT_CC_MAX_TURNS}}"

# Optional: require dispatching the investigator before any edit, to maximize sub-agent fan-out.
# export SWE_CC_PROMPT="Read PROBLEM_STATEMENT.md. BEFORE editing any file, dispatch the 'investigator' sub-agent (via the Agent tool with subagent_type=investigator) to locate every file relevant to the issue. Then fix the issue and run the tests."

# ============ proxy bypass for in-cluster/AGS traffic ============
export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST},${E2B_DOMAIN},.tencentags.com"
export NO_PROXY="${no_proxy}"

cd "${SLIME_DIR}"
source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

SAVE_DIR="${SAVE_DIR:-${EXP}/checkpoints}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
mkdir -p "${SAVE_DIR}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
   --load "${LOAD_DIR:-${SAVE_DIR}}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
   --rollout-function-path slime_plugins.rollout_buffer.rollout_buffer_example.generate_rollout
   # Used by periodic eval, which runs AGS through slime's standard sglang eval loop.
   --custom-generate-function-path slime_plugins.rollout_buffer.generator.ags_generator.generate
   --custom-rollout-log-function-path slime_plugins.rollout_buffer.generator.ags_generator.wandb_metrics.log_rollout_data
   --custom-eval-rollout-log-function-path slime_plugins.rollout_buffer.generator.ags_generator.wandb_metrics.log_eval_rollout_data
   --rollout-task-type ags
   --rollout-buffer-url "http://${MASTER_ADDR}:8889"
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-buffer-num-epoch 1
   --rollout-buffer-stop-timeout-sec "${ROLLOUT_BUFFER_STOP_TIMEOUT_SEC:-120}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${MAX_GEN_LEN}"
   --rollout-temperature 1.0
   --rollout-stop-token-ids 248046 248044
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --micro-batch-size "${MICRO_BATCH_SIZE}"
   --loss-mask-type qwen3_5
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)

EVAL_ARGS=(
   --eval-function-path slime.rollout.sglang_rollout.generate_rollout
   --eval-interval "${EVAL_INTERVAL}"
   --eval-prompt-data swebench_verified "${EVAL_DATA}"
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
   --eval-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
   --eval-max-response-len "${MAX_GEN_LEN}"
   --eval-temperature 0.6
   --eval-top-p 0.95
   --eval-top-k 20
)

if [[ "${SKIP_EVAL_BEFORE_TRAIN}" == "1" || "${SKIP_EVAL_BEFORE_TRAIN}" == "true" ]]; then
   EVAL_ARGS+=(--skip-eval-before-train)
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size "${ETP_SIZE}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # max-tokens-per-gpu is one CP rank's slice of MAX_CONTEXT_LEN; log-probs are
   # chunked along T to avoid OOM on long single trajectories.
   --max-tokens-per-gpu $((MAX_CONTEXT_LEN / CP_SIZE))
   --log-probs-chunk-size 1024
   --use-dynamic-batch-size
)

ALGO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}"
   --sglang-mem-fraction-static "${ROLLOUT_MEM_UTILIZATION}"
   --sglang-enable-dp-attention
   --sglang-dp-size "${ROLLOUT_DP_SIZE}"
   --sglang-ep-size "${ROLLOUT_EP_SIZE}"
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

if [[ -n "${WANDB_API_KEY:-}" ]]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-team "${WANDB_ENTITY:?WANDB_ENTITY is required when WandB is enabled}"
      --wandb-project "${WANDB_PROJECT:-slime-claude-code-ags}"
      --wandb-group "${WANDB_GROUP:-${EXP_TAG}}"
      --wandb-key "${WANDB_API_KEY}"
      --wandb-dir "${LOG_DIR}/wandb"
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
   --colocate
   --log-passrate
)

# ============ bring up rollout buffer ============
python3 -u -m slime_plugins.rollout_buffer.buffer >"${BUFFER_LOG_FILE}" 2>&1 &
BUFFER_PID=$!
trap 'kill ${BUFFER_PID} 2>/dev/null || true' EXIT
sleep 5

# ============ bring up ray cluster ============
HOSTFILE="${HOSTFILE:-/root/mpi_rack_hostfile}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [[ -f "${HOSTFILE}" ]]; then
  WORKER_LIMIT=$((ACTOR_NUM_NODES - 1))
  STARTED_WORKERS=0
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    [[ -z "${WORKER_IP}" ]] && continue
    [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]] && continue
    if (( STARTED_WORKERS >= WORKER_LIMIT )); then
      break
    fi
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh -o StrictHostKeyChecking=no "root@${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; \
       ray start --address=${MASTER_ADDR}:${MASTER_PORT} --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} \
         --node-ip-address ${WORKER_IP} --disable-usage-stats" &
    STARTED_WORKERS=$((STARTED_WORKERS + 1))
  done
  for pid in $(jobs -pr); do
    [[ "${pid}" == "${BUFFER_PID}" ]] && continue
    wait "${pid}"
  done
  if (( STARTED_WORKERS < WORKER_LIMIT )); then
    echo "WARNING: requested ${ACTOR_NUM_NODES} nodes but only started $((STARTED_WORKERS + 1)) including head."
  fi
else
  echo "WARNING: HOSTFILE=${HOSTFILE} not found; only the head node was started."
fi

echo "Waiting for Ray cluster to stabilize..."
sleep 30
ray status

# ============ runtime env propagated to ray workers ============
export SLIME_DIR
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "SWE_AGENT", "E2B_API_KEY", "E2B_DOMAIN", "AGS_BASE_TOOL",
    "AGS_IMAGE_REGISTRY_TYPE", "AGS_SANDBOX_RESOURCES_JSON",
    "EXPERIMENT_NAME", "TRAJECTORY_DUMP_DIR",
    "ADAPTER_PUBLIC_HOST", "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "SWE_AGENT_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC", "SWE_EVAL_ISOLATED_SANDBOX",
    "SWE_BOOT_CONCURRENCY",
    "SWE_BOOT_RETRIES", "SWE_ROLLOUT_GUARD_SEC", "SWE_ROLLOUT_CONCURRENCY",
    "SLIME_AGENT_CC_MAX_TURNS", "SLIME_AGENT_CC_EXTRA_ARGS", "SLIME_AGENT_CC_EXTRA_ENVS",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "SWE_CC_PROMPT",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
# Keep per-node socket interface env inherited from each Ray node; do not override workers with the head ifname.
env["PYTHONPATH"] = f"/root/Megatron-LM/:{os.environ['SLIME_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
