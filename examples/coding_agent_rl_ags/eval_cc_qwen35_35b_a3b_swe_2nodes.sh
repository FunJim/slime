#!/usr/bin/env bash
# Eval-only: score ONE checkpoint on the full SWE-bench Verified set with Claude
# Code on AGS. No training, no rollout buffer, no optimizer step.
#
# Derived from run_qwen35_35b_a3b_swe_2nodes.sh. The differences that matter:
#
#   --num-rollout 0   train.py's eval-only branch runs eval once at rollout_id=0
#                     and then exits, skipping the whole train loop.
#   no buffer         the eval path (ags_generator.generate -> AGSRolloutRunner)
#                     never talks to the rollout buffer, so buffer.py is not
#                     started and --rollout-function-path / --rollout-buffer-url
#                     are omitted. The AGS adapter is started by the eval path
#                     itself (see ADAPTER_PORT below).
#   scheduler args    train_iters = num_rollout * ... = 0 makes Megatron's
#                     OptimizerParamScheduler assert; see LR_SCHED_ARGS.
#
# Run from a long-lived shell / tmux session on the Ray head node. Budget ~95min
# for 500 samples at SWE_ROLLOUT_CONCURRENCY=32.
#
# Required:
#   EXP=/data_train/<user>/experiments/<name>
#   LOAD_DIR=<checkpoint dir containing latest_checkpointed_iteration.txt>
#   E2B_API_KEY=<AGS gateway key>
# Strongly recommended:
#   CKPT_STEP=<iteration>   pin the exact step; without it Megatron reads
#                           latest_checkpointed_iteration.txt and you score
#                           whatever happens to be newest.
#   TRAIN_NUM_ROLLOUT=<N>   the --num-rollout of the run that WROTE the
#                           checkpoint (see LR_SCHED_ARGS).
#
# Example:
#   EXP=/data_train/ericxjzheng/experiments/eval_step79 \
#   LOAD_DIR=/data_train/ericxjzheng/experiments/<run>/checkpoints \
#   CKPT_STEP=79 TRAIN_NUM_ROLLOUT=100 \
#   E2B_API_KEY=... WANDB_API_KEY=... WANDB_ENTITY=... \
#   bash examples/coding_agent_rl_ags/eval_cc_qwen35_35b_a3b_swe_2nodes.sh

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
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${MLP_WORKER_NUM:-2}}"
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
ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-2}"
ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-8}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.75}"

# num_rollout 0 selects train.py's eval-only branch. The rollout_batch_size /
# n_samples_per_prompt below are unused by eval (it iterates the eval dataset)
# but must stay positive: model.py divides by global_batch_size.
NUM_ROLLOUT=0
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

# ============ context length ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-96000}"
MAX_GEN_LEN="${MAX_GEN_LEN:-32768}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-${MAX_CONTEXT_LEN}}"

# ============ eval ============
# eval_interval must be non-None for the eval-only branch to fire; with
# num_rollout=0 the value itself is never used as a cadence.
EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
EVAL_DATA="${EVAL_DATA:-/data_train/ericxjzheng/data/SWE-bench_Verified_slime_rl_format_from_harbor/swebench_verified_slime_instruction_tcr.jsonl}"
# Names the dataset in slime's metrics, so the score lands on eval/<name>. Only
# worth overriding when EVAL_DATA is not SWE-bench Verified -- a difficulty scan
# over the *training* pool, say, which would otherwise log a "swebench_verified"
# number that is nothing of the sort.
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-swebench_verified}"
# 1 attempt x 500 rows = the full set, scored pass@1.
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"

# Benchmark sampling, matching the periodic eval in the training launcher so the
# two are comparable. Overridable because measuring how *trainable* a prompt is
# needs the training sampling instead (temperature 1.0, top_p 1.0, top_k -1 --
# slime's --rollout-top-p/--rollout-top-k defaults, which the training launcher
# leaves untouched): a prompt's solve rate is a property of the sampling
# distribution, so a scan run at 0.7/0.8/20 would not describe the rollouts GRPO
# actually sees.
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
EVAL_TOP_P="${EVAL_TOP_P:-0.8}"
EVAL_TOP_K="${EVAL_TOP_K:-20}"

# ============ paths — override before launching ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/data_train/ericxjzheng/models/Qwen3.5-35B-A3B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/data_train/ericxjzheng/models/Qwen3.5-35B-A3B_torch_dist}"

# The checkpoint under test. Without LOAD_DIR this scores the base model, which
# is a valid baseline but almost certainly not what you meant.
LOAD_DIR="${LOAD_DIR:-}"
CKPT_STEP="${CKPT_STEP:-}"

EXP_TAG="${EXP_TAG:-coding_agent_rl_ags_eval_cc_qwen35_35b_a3b${CKPT_STEP:+_step${CKPT_STEP}}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${EXP}/runs/${EXP_TAG}_${STAMP}}"

# ============ logging/artifacts ============
LOG_DIR="${RUN_ROOT}"
mkdir -p "${LOG_DIR}/ags_artifacts"
LOG_FILE="${LOG_DIR}/run.log"
export TRAJECTORY_DUMP_DIR="${TRAJECTORY_DUMP_DIR:-${LOG_DIR}/ags_artifacts}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXP_TAG}}"
echo "======================================================================"
echo "Eval log:      ${LOG_FILE}"
echo "RUN_ROOT=      ${RUN_ROOT}"
echo "Checkpoint:    ${LOAD_DIR:-<none: scoring the base HF model>}"
echo "Step:          ${CKPT_STEP:-<latest_checkpointed_iteration.txt>}"
echo "Eval data:     ${EVAL_DATA}"
echo "Dataset name:  ${EVAL_DATASET_NAME}"
echo "Attempts/row:  ${N_SAMPLES_PER_EVAL_PROMPT}"
echo "Sampling:      temperature=${EVAL_TEMPERATURE} top_p=${EVAL_TOP_P} top_k=${EVAL_TOP_K}"
echo "Prompt style:  ${SWE_EVAL_PROMPT_STYLE:-dataset}"
echo "======================================================================"

if [[ -n "${LOAD_DIR}" && ! -f "${LOAD_DIR}/latest_checkpointed_iteration.txt" ]]; then
  echo "ERROR: LOAD_DIR=${LOAD_DIR} has no latest_checkpointed_iteration.txt."
  echo "       slime treats such a path as 'no Megatron checkpoint' and silently"
  echo "       falls back to --ref-load, so you would score the base model."
  exit 1
fi
if [[ -n "${LOAD_DIR}" && -z "${CKPT_STEP}" ]]; then
  echo "WARNING: CKPT_STEP unset; Megatron will load whichever step"
  echo "         latest_checkpointed_iteration.txt names:"
  cat "${LOAD_DIR}/latest_checkpointed_iteration.txt" || true
fi

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

# No training rollout runs here, so nothing binds ADAPTER_PORT ahead of eval.
# get_adapter_service health-probes this port and, finding nothing, starts a
# local eval adapter on an ephemeral port inside the RolloutManager actor --
# advertising that actor's own node IP, since Ray does not pin it to the head.
# ADAPTER_PUBLIC_HOST is therefore only a probe target here, not the address
# handed to sandboxes.
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

# The default EVAL_DATA is the instruction.md-sourced set, whose `prompt` already
# carries the task text, so eval uses the dataset style: hand that prompt straight
# to the agent and write no PROBLEM_STATEMENT.md. Override to "instruction" when
# pointing EVAL_DATA at the training pool -- SWE-rebench rows carry a raw issue
# body rather than a rendered instruction, and the training rollouts they must be
# compared against run under SWE_PROMPT_STYLE=instruction.
# SWE_PROMPT_STYLE is set too because AGSGeneratorConfig.from_env validates both,
# though only the eval one is read here.
export SWE_PROMPT_STYLE="${SWE_PROMPT_STYLE:-instruction}"
export SWE_EVAL_PROMPT_STYLE="${SWE_EVAL_PROMPT_STYLE:-dataset}"

# ---- auto-compaction -------------------------------------------------------
# Compact before a segment crosses the training-side context cap, otherwise the
# adapter returns finish_reason="length" with zero output tokens once the prompt
# reaches rollout_max_context_len (slime/agent/adapters/common.py) and the turn is
# wasted.
#
# Expressed as a PERCENTAGE of the model's context window rather than an absolute
# token count: an absolute auto-compact window is clamped to [100k, 1M] by both
# CLIs, which is above our 96k cap, so it could never fire in time. Both CLIs are
# also told the real window, so the percentage is of MAX_CONTEXT_LEN on each side.
#
#   CodeBuddy   CODEBUDDY_AUTOCOMPACT_PCT_OVERRIDE (cbc 1.9.33) is a percentage of
#               the model's maxInputTokens, which the harness writes into
#               models.json from SLIME_AGENT_MAX_INPUT_TOKENS. Without that key
#               resolveCompactTriggerAt() falls back to the clamped absolute
#               window. Source: agent-cli src/node/context/context-protocol.ts.
#   Claude Code CLAUDE_AUTOCOMPACT_PCT_OVERRIDE, as a percentage of the context
#               window, which CLAUDE_CODE_MAX_CONTEXT_TOKENS sets. That variable
#               applies DIRECTLY for model names claude does not recognise as a
#               Claude model -- ours is "slime-actor", so it does (the binary
#               gates on `!normalize(model).startsWith("claude-")`; for a real
#               claude-* name it would need DISABLE_COMPACT too, which would
#               disable the compaction we want). Without it the CLI assumes its
#               200000 default and would compact at pct% of 200k, i.e. never
#               before our 96k cap.
# 60%, not the CLI's 70% default, because the per-turn output reservation eats
# into what is reachable: claude assumes MAX_OUTPUT_TOKENS (32000 by default for
# model ids it does not recognise, which includes ours) is available on top of the
# prompt, so the prompt cannot grow past MAX_CONTEXT_LEN - MAX_GEN_LEN ~= 63k
# before the adapter's hard stop. A 70% trigger (67200) sits ABOVE that and would
# never be reached; 60% (57600) fires with room to spare. Raising
# AGENT_AUTOCOMPACT_PCT re-opens that gap -- the check below says so out loud.
AGENT_AUTOCOMPACT_PCT="${AGENT_AUTOCOMPACT_PCT:-60}"
AGENT_MAX_CONTEXT_TOKENS="${AGENT_MAX_CONTEXT_TOKENS:-${MAX_CONTEXT_LEN}}"

# cbc: models.json maxInputTokens (written by CodeBuddyCodeHarness).
export SLIME_AGENT_MAX_INPUT_TOKENS="${SLIME_AGENT_MAX_INPUT_TOKENS:-${AGENT_MAX_CONTEXT_TOKENS}}"
# Built in a separate variable, not inline in ${VAR:-...}: a JSON default inside
# that expansion is mis-parsed -- the value's own "}" closes the expansion early
# and the trailing brace leaks into the result ("{...}}").
CBC_AUTOCOMPACT_ENVS="{\"CODEBUDDY_AUTOCOMPACT_PCT_OVERRIDE\":\"${AGENT_AUTOCOMPACT_PCT}\"}"
export SLIME_AGENT_CBC_EXTRA_ENVS="${SLIME_AGENT_CBC_EXTRA_ENVS:-${CBC_AUTOCOMPACT_ENVS}}"

# claude: declare the window and the per-turn output budget, then set the trigger
# percentage. MAX_OUTPUT_TOKENS is pinned to MAX_GEN_LEN rather than left at the
# CLI's 32000 default-for-unknown-model-ids so the reservation matches what the
# adapter will actually serve (--rollout-max-response-len).
CC_AUTOCOMPACT_ENVS="{\"CLAUDE_CODE_MAX_CONTEXT_TOKENS\":\"${AGENT_MAX_CONTEXT_TOKENS}\",\"CLAUDE_CODE_MAX_OUTPUT_TOKENS\":\"${MAX_GEN_LEN}\",\"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE\":\"${AGENT_AUTOCOMPACT_PCT}\"}"
export SLIME_AGENT_CC_EXTRA_ENVS="${SLIME_AGENT_CC_EXTRA_ENVS:-${CC_AUTOCOMPACT_ENVS}}"

AGENT_AUTOCOMPACT_AT=$((AGENT_AUTOCOMPACT_PCT * AGENT_MAX_CONTEXT_TOKENS / 100))
# The largest prompt that can still be served: the adapter caps prompt+output at
# rollout_max_context_len, so a turn needs MAX_GEN_LEN of headroom.
AGENT_PROMPT_CEILING=$((MAX_CONTEXT_LEN - MAX_GEN_LEN))
echo "Auto-compact:  ${AGENT_AUTOCOMPACT_PCT}% of ${AGENT_MAX_CONTEXT_TOKENS} = ${AGENT_AUTOCOMPACT_AT} tokens" \
     "(prompt ceiling ${AGENT_PROMPT_CEILING} = ${MAX_CONTEXT_LEN} - ${MAX_GEN_LEN})"
if (( AGENT_AUTOCOMPACT_AT >= AGENT_PROMPT_CEILING )); then
  echo "WARNING: the compaction trigger (${AGENT_AUTOCOMPACT_AT}) is at or above the prompt ceiling" \
       "(${AGENT_PROMPT_CEILING}); turns will hit finish_reason=length before compaction fires." \
       "Lower AGENT_AUTOCOMPACT_PCT to <= $((AGENT_PROMPT_CEILING * 100 / AGENT_MAX_CONTEXT_TOKENS))."
fi
if (( AGENT_MAX_CONTEXT_TOKENS > MAX_CONTEXT_LEN )); then
  echo "WARNING: declared context ${AGENT_MAX_CONTEXT_TOKENS} > MAX_CONTEXT_LEN ${MAX_CONTEXT_LEN};" \
       "the adapter hard-stops at the latter."
fi

# The only two harness knobs per agent: extra CLI flags, and extra env vars as
# JSON. Everything else (denied tools, the launch flags) is a class attribute in
# slime_plugins/.../harnesses.py, because it is a property of the harness rather
# than of a run. Both are applied LAST -- EXTRA_ARGS after the harness's own
# flags (claude takes the last occurrence of a repeated flag, verified against
# the real CLI) and EXTRA_ENVS after static_env -- so either can override a
# harness default. The *_EXTRA_ENVS pair is set in the auto-compaction block
# above; these two are the remaining passthroughs, declared so SWE_AGENT can be
# either harness from this script (run_cbc_*.sh just sets it and re-execs).
export SLIME_AGENT_CC_EXTRA_ARGS="${SLIME_AGENT_CC_EXTRA_ARGS:-}"
export SLIME_AGENT_CBC_EXTRA_ARGS="${SLIME_AGENT_CBC_EXTRA_ARGS:-}"

# ============ proxy bypass for in-cluster/AGS traffic ============
export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST},${E2B_DOMAIN},.tencentags.com"
export NO_PROXY="${no_proxy}"

cd "${SLIME_DIR}"
source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
)
if [[ -n "${LOAD_DIR}" ]]; then
   CKPT_ARGS+=(--load "${LOAD_DIR}")
fi
if [[ -n "${CKPT_STEP}" ]]; then
   # Megatron's get_load_checkpoint_path_by_args honours ckpt_step over the
   # tracker file, which is how one specific step gets scored.
   CKPT_ARGS+=(--ckpt-step "${CKPT_STEP}")
fi
# No --save/--save-interval: nothing is trained, so nothing should be written.

ROLLOUT_ARGS=(
   # Eval reaches AGS through slime's standard sglang eval loop plus this hook.
   # --rollout-function-path and --rollout-buffer-url are deliberately absent:
   # the eval path never writes to the rollout buffer, so buffer.py is not run.
   --custom-generate-function-path slime_plugins.rollout_buffer.generator.ags_generator.generate
   --custom-eval-rollout-log-function-path slime_plugins.rollout_buffer.generator.ags_generator.wandb_metrics.log_eval_rollout_data
   --rollout-task-type ags
   # NOTE: no --apply-chat-template, unlike the training script. Under
   # SWE_EVAL_PROMPT_STYLE=dataset the agent is handed sample.prompt verbatim
   # (swe_task._coerce_prompt), and the flag makes slime's Dataset run
   # tokenizer.apply_chat_template over the row first -- which would hand Claude
   # Code a prompt literally containing "<|im_start|>user ... <|im_end|>".
   # The eval path renders its own chat template per turn inside the adapter.
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${MAX_GEN_LEN}"
   --rollout-stop-token-ids 248046 248044
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --micro-batch-size "${MICRO_BATCH_SIZE}"
   --loss-mask-type qwen3_5
)

EVAL_ARGS=(
   --eval-function-path slime.rollout.sglang_rollout.generate_rollout
   --eval-interval "${EVAL_INTERVAL}"
   --eval-prompt-data "${EVAL_DATASET_NAME}" "${EVAL_DATA}"
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
   --eval-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
   --eval-max-response-len "${MAX_GEN_LEN}"
   --eval-temperature "${EVAL_TEMPERATURE}"
   --eval-top-p "${EVAL_TOP_P}"
   --eval-top-k "${EVAL_TOP_K}"
)

# train_iters = num_rollout * rollout_batch_size * n_samples_per_prompt //
# global_batch_size = 0 here, and Megatron's OptimizerParamScheduler asserts on
# a zero-length schedule. Three asserts fire in sequence otherwise:
#   1. lr_decay_steps > 0                    -> --lr-decay-iters
#   2. total-iterations mismatch vs the ckpt -> the value must equal the
#      *original* run's num_rollout, not any positive number
#   3. weight-decay-iterations mismatch      -> no override arg exists, so take
#      the whole scheduler state from the checkpoint
# TRAIN_NUM_ROLLOUT must therefore be the --num-rollout of the run that wrote
# this checkpoint. Only needed when actually loading a Megatron checkpoint.
# --lr-decay-iters is needed even with no --load: train_iters is 0 either way, so
# `assert self.lr_decay_steps > 0` fires while building the scheduler, before any
# checkpoint is touched. For the base model the value is arbitrary (nothing is
# trained and no scheduler state is compared), so default it to 1.
LR_SCHED_ARGS=(--lr-decay-iters "${LR_DECAY_ITERS_BASE:-1}")
if [[ -n "${LOAD_DIR}" ]]; then
   TRAIN_NUM_ROLLOUT="${TRAIN_NUM_ROLLOUT:?set TRAIN_NUM_ROLLOUT to the --num-rollout of the run that wrote this checkpoint (e.g. 100)}"
   LR_SCHED_ARGS=(
      --lr-decay-iters $((TRAIN_NUM_ROLLOUT * ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / GLOBAL_BATCH_SIZE))
      # Only the weights are needed: nothing is trained here, so the optimizer and
      # RNG state in the checkpoint are dead weight -- and loading them actually
      # crashes. Megatron's generate_state_dict -> DistributedOptimizer
      # .sharded_state_dict -> load_state_dict -> dummy_step() reaches
      # hybrid_optimizer._set_sub_optimizer_grads, whose torch.empty for the CPU
      # offload copy map dies with "CUDA error: invalid argument" (this
      # checkpoint was written with --optimizer-cpu-offload). slime itself takes
      # this same route whenever it wants weights only (see arguments.py where it
      # sets no_load_optim/no_load_rng/finetune together).
      #
      # --finetune also stops Megatron restoring the iteration counter, which is
      # what --use-checkpoint-opt-param-scheduler was working around; keep the
      # explicit --lr-decay-iters above so the scheduler still constructs.
      --no-load-optim
      --no-load-rng
      --finetune
   )
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
   --max-tokens-per-gpu $((MAX_CONTEXT_LEN / CP_SIZE))
   --log-probs-chunk-size 1024
   --use-dynamic-batch-size
)

# Kept because slime constructs the optimizer/scheduler even for eval-only; none
# of these values affect the score.
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

ALGO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
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
      --wandb-dir "${LOG_DIR}"
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

# ============ bring up ray cluster ============
# No rollout buffer here: the eval path talks to AGS directly.
HOSTFILE="${HOSTFILE:-/root/mpi_rack_hostfile}"

# Every Ray port is overridable because these H20 nodes routinely host other
# users' Ray clusters. The dashboard AGENT port matters most: it defaults to
# 52365, and when that is taken the agent dies with "address already in use"
# while `ray status` still reports a healthy cluster -- `ray job submit` then
# fails with "No available agent to submit job", which looks nothing like a port
# conflict. Ports below match the training launcher's defaults.
#
# --port must be MASTER_PORT: the worker loop below dials
# ${MASTER_ADDR}:${MASTER_PORT}, but without --port the head's GCS always binds
# 6379, so any non-default MASTER_PORT leaves workers dialing a dead port and
# timing out on "Failed to connect to GCS".
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
   --port "${MASTER_PORT}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 \
   --dashboard-port "${RAY_DASHBOARD_PORT:-8265}" \
   --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT:-28065}" \
   --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT:-28066}" \
   --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT:-28067}" \
   --metrics-export-port "${RAY_METRICS_EXPORT_PORT:-28068}"

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
    wait "${pid}"
  done
  if (( STARTED_WORKERS < WORKER_LIMIT )); then
    echo "WARNING: requested ${ACTOR_NUM_NODES} nodes but only started $((STARTED_WORKERS + 1)) including head."
  fi
else
  echo "WARNING: HOSTFILE=${HOSTFILE} not found; only the head node was started."
fi

# Wait for every GPU to actually register instead of sleeping a fixed 30s. Slime
# asks for a placement group of ACTOR_NUM_NODES x ACTOR_NUM_GPUS_PER_NODE GPUs and
# hangs on "1+ pending placement groups" if a worker is late or never joined --
# which is easy to miss, because `ray status` reports a perfectly healthy cluster
# with only the head's GPUs. This also covers externally started workers (the
# container has no root ssh key, so workers may be joined from outside).
EXPECTED_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
RAY_WAIT_SEC="${RAY_WAIT_SEC:-300}"
echo "Waiting for ${EXPECTED_GPUS} GPUs to register with Ray (timeout ${RAY_WAIT_SEC}s)..."
for ((waited = 0; waited < RAY_WAIT_SEC; waited += 10)); do
   TOTAL_GPUS=$(ray status 2>/dev/null | grep -oE '[0-9.]+/[0-9]+\.[0-9]+ GPU' | head -1 | sed -E 's|.*/([0-9]+)\.[0-9]+ GPU|\1|')
   if [[ "${TOTAL_GPUS:-0}" -ge "${EXPECTED_GPUS}" ]]; then
      echo "Ray has ${TOTAL_GPUS} GPUs after ${waited}s."
      break
   fi
   echo "  ${waited}s: ${TOTAL_GPUS:-0}/${EXPECTED_GPUS} GPUs registered"
   sleep 10
done
if [[ "${TOTAL_GPUS:-0}" -lt "${EXPECTED_GPUS}" ]]; then
   echo "ERROR: only ${TOTAL_GPUS:-0}/${EXPECTED_GPUS} GPUs registered after ${RAY_WAIT_SEC}s."
   echo "       Join the missing worker(s), or the job will hang on a pending placement group."
   ray status || true
   exit 1
fi
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
    "SWE_EMPTY_PATCH_GUARD", "SWE_PROMPT_STYLE", "SWE_EVAL_PROMPT_STYLE",
    "SLIME_AGENT_CC_EXTRA_ARGS", "SLIME_AGENT_CC_EXTRA_ENVS",
    "SLIME_AGENT_CBC_EXTRA_ARGS", "SLIME_AGENT_CBC_EXTRA_ENVS",
    # Read by CodeBuddyCodeHarness.write_config to set models.json maxInputTokens,
    # which is what the auto-compact percentage is a percentage OF.
    "SLIME_AGENT_MAX_INPUT_TOKENS",
    "SWE_CC_PROMPT",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
# The socket interface MUST be propagated. Inheriting it per-node only works when
# every Ray worker was started from a shell that exports it; workers launched by
# hand (as eval-only requires, since the container has no root ssh key) inherit
# nothing. Gloo then picks an address family per rank independently, and on nodes
# whose eth0 carries both IPv4 and a link-local IPv6 the ranks disagree:
#   RuntimeError: [enforce fail at .../tcp/device.cc:285]
#   ss1.ss_family == ss2.ss_family. 10 vs 2      (10=AF_INET6, 2=AF_INET)
# Naming the interface pins one family for every rank. GLOO_SOCKET_FAMILY makes
# the choice explicit rather than relying on the interface's first address.
for _k in ("GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME"):
    if _k in os.environ:
        env[_k] = os.environ[_k]
env.setdefault("GLOO_SOCKET_FAMILY", "AF_INET")
env["PYTHONPATH"] = f"/root/Megatron-LM/:{os.environ['SLIME_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${LR_SCHED_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "======================================================================"
echo "RUN_ROOT=${RUN_ROOT}"
echo "Score:      grep -E 'eval/${EVAL_DATASET_NAME}' ${LOG_FILE}"
echo "Trajectories/patches: ${TRAJECTORY_DUMP_DIR}"
echo "======================================================================"
