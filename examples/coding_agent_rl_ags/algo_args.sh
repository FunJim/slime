# Algorithm arguments for the AGS coding-agent runs -- SOURCED, not executed.
#
#   ADVANTAGE_ESTIMATOR=grpo (default)  group-relative baseline, no critic.
#   ADVANTAGE_ESTIMATOR=ppo             token-level GAE against a learned critic.
#
# Sourced by run_cc_*_{2,4}nodes.sh after SCRIPT_DIR, RUN_ROOT, SAVE_DIR,
# REF_MODEL_PATH and NUM_ROLLOUT are set. Sets ALGO_ARGS; the PPO branch also
# writes a Megatron role-config YAML into RUN_ROOT and appends
# --megatron-config-path pointing at it.
#
# Lives in its own file because the 2-node and 4-node launchers differ only in
# cluster size: a per-launcher copy of this block would drift the same way the
# per-harness launcher copies did before run_cbc_*.sh became a wrapper.

ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"

# Shared by both estimators. The clip range is asymmetric (clip-higher, DAPO
# arXiv:2503.14476): the upper bound is loosened so low-probability tokens can
# still gain mass, which matters for agent trajectories where the useful tokens
# (tool calls) are rare relative to prose.
ALGO_ARGS=(
   --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

case "${ADVANTAGE_ESTIMATOR}" in
grpo) ;;
ppo)
   # ---- why PPO here ------------------------------------------------------
   # One AGS rollout fans out into up to 8 training samples that share a
   # rollout_id (post-compaction segments). GRPO groups by prompt, so those
   # segments -- which have wildly different lengths and usefulness -- get one
   # shared group baseline, and a group whose members all solve or all fail
   # contributes exactly zero advantage. PPO's baseline is per-token V(s_t)
   # from the critic, so it needs no group structure and produces a nonzero
   # signal even from a 1-sample group.
   #
   # The price: a second full-size Megatron model on the same GPUs (slime pins
   # critic_num_nodes = actor_num_nodes and reuses the actor placement group),
   # so a train step runs critic-then-actor sequentially and costs roughly 2x
   # the actor-only step. slime also forces --offload-train under a critic,
   # which means the torch_memory_saver pause path is mandatory: nodes where
   # that path crashes (e.g. sh5-13) cannot run this arm.

   # gamma/lambd are deliberately pinned to 1.0 and should stay there for these
   # trajectories. Rewards arrive once, at the final token, and the response
   # contains long spans of tool output that the loss mask excludes. With
   # gamma=lambd=1 GAE degenerates to A_t = R - V(s_t), which is per-token and
   # therefore immune to whatever the value head predicts on masked positions.
   # Any lambd < 1 bootstraps each token's advantage through its neighbours'
   # values, i.e. through positions the critic is never trained on.
   PPO_GAMMA="${PPO_GAMMA:-1.0}"
   PPO_LAMBD="${PPO_LAMBD:-1.0}"
   PPO_VALUE_CLIP="${PPO_VALUE_CLIP:-0.2}"

   # The value head is freshly initialized (see _critic_output_layer_needs_reinit
   # in slime/backends/megatron_utils/model.py), so its first predictions are
   # noise. Train the critic alone for the first few rollouts; the actor is
   # frozen until then, and its weights are still pushed to the rollout engines
   # so generation is unaffected.
   NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-2}"

   # Critic LR is ~10x the actor's: it is fitting a scalar regression head from
   # scratch while the actor only needs small policy nudges.
   CRITIC_LR="${CRITIC_LR:-1e-5}"
   CRITIC_LR_WARMUP_ITERS="${CRITIC_LR_WARMUP_ITERS:-0}"

   # Separate checkpoint directory. Without this the critic inherits --save and
   # both roles write into the same iter_XXXXXXX directory, each clobbering the
   # other's weights under the same names.
   CRITIC_SAVE_DIR="${CRITIC_SAVE_DIR:-${EXP}/critic_checkpoints}"
   mkdir -p "${CRITIC_SAVE_DIR}"

   # Cold start vs resume. slime resolves --load once, for the actor, before the
   # role split, so the critic would silently follow the actor's checkpoint --
   # on a resume that means loading actor weights and reinitializing the value
   # head, throwing away everything the critic learned. Resolve it explicitly:
   # its own checkpoint if one exists, otherwise the base weights with a fresh
   # head (and no optimizer/RNG state to load from them).
   CRITIC_LOAD_DIR="${CRITIC_LOAD_DIR:-}"
   if [[ -z "${CRITIC_LOAD_DIR}" ]]; then
      if [[ -f "${CRITIC_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
         CRITIC_LOAD_DIR="${CRITIC_SAVE_DIR}"
      else
         CRITIC_LOAD_DIR="${REF_MODEL_PATH}"
      fi
   fi
   if [[ "${CRITIC_LOAD_DIR}" == "${CRITIC_SAVE_DIR}" ]]; then
      CRITIC_COLD_START=0
   else
      CRITIC_COLD_START=1
   fi

   MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-${RUN_ROOT}/megatron_ppo.yaml}"
   mkdir -p "$(dirname -- "${MEGATRON_CONFIG_PATH}")"
   # Keys are argparse attribute names (underscores), not CLI flags. Only
   # role-specific differences belong here -- actor and critic must keep the
   # same parallel topology, which stays on the CLI.
   {
      echo "megatron:"
      echo "  - name: default"
      echo "    role: critic"
      echo "    overrides:"
      # A bare 1e-5 is a *string* to PyYAML (its float resolver wants a dot,
      # as in 1.0e-5); slime coerces override strings to the type of the
      # existing arg, so either spelling lands as a float.
      echo "      lr: ${CRITIC_LR}"
      echo "      lr_decay_style: constant"
      echo "      lr_warmup_iters: ${CRITIC_LR_WARMUP_ITERS}"
      echo "      load: ${CRITIC_LOAD_DIR}"
      echo "      save: ${CRITIC_SAVE_DIR}"
      if (( CRITIC_COLD_START )); then
         # The base weights carry neither optimizer nor RNG state, and the
         # value head does not exist in them at all.
         echo "      finetune: true"
         echo "      no_load_optim: true"
         echo "      no_load_rng: true"
      fi
   } >"${MEGATRON_CONFIG_PATH}"

   ALGO_ARGS+=(
      --gamma "${PPO_GAMMA}"
      --lambd "${PPO_LAMBD}"
      --value-clip "${PPO_VALUE_CLIP}"
      # GAE advantages carry the reward's raw scale (unlike GRPO's, which are
      # already group-standardized), so whiten them across the DP group before
      # the policy loss sees them.
      --normalize-advantages
      --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS}"
      --megatron-config-path "${MEGATRON_CONFIG_PATH}"
   )

   if (( NUM_CRITIC_ONLY_STEPS >= NUM_ROLLOUT )); then
      echo "WARNING: NUM_CRITIC_ONLY_STEPS=${NUM_CRITIC_ONLY_STEPS} >= NUM_ROLLOUT=${NUM_ROLLOUT};" \
           "the actor will never train (critic-only run)."
   fi
   echo "PPO: critic lr=${CRITIC_LR} load=${CRITIC_LOAD_DIR} save=${CRITIC_SAVE_DIR}" \
        "cold_start=${CRITIC_COLD_START} critic_only_steps=${NUM_CRITIC_ONLY_STEPS}" \
        "gamma=${PPO_GAMMA} lambd=${PPO_LAMBD}"
   echo "PPO: megatron role config -> ${MEGATRON_CONFIG_PATH}"
   ;;
*)
   echo "ERROR: unsupported ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR} (expected grpo or ppo)" >&2
   exit 1
   ;;
esac
