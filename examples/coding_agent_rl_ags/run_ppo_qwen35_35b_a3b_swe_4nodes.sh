#!/usr/bin/env bash
# PPO (actor + critic) training on SWE-rebench with an AGS coding-agent harness,
# 4 nodes.
#
# Thin wrapper, same pattern as run_cbc_*.sh: the launcher logic lives in
# run_cc_qwen35_35b_a3b_swe_4nodes.sh and is estimator-agnostic --
# ADVANTAGE_ESTIMATOR selects the algorithm and algo_args.sh builds the
# corresponding arguments.
#
# Why PPO on this task: an AGS rollout fans out into several training samples
# that share a rollout_id, and GRPO gives that whole group a single baseline --
# an all-solved or all-failed group yields zero advantage. PPO scores every
# token against the critic's V(s_t), so no group structure is needed.
#
# Cost: the critic is a second full-size model sharing the actor's GPUs, so a
# train step is ~2x an actor-only step, and slime forces --offload-train
# whenever a critic is present.
#
# The CC launcher is configured purely through environment variables -- it reads
# no positional arguments -- so every knob it accepts is set the same way here,
# plus the PPO ones from algo_args.sh (CRITIC_LR, NUM_CRITIC_ONLY_STEPS,
# CRITIC_SAVE_DIR, CRITIC_LOAD_DIR, START_ROLLOUT_ID, PPO_GAMMA, PPO_LAMBD,
# PPO_VALUE_CLIP) and ACTOR_LR / SEED. Pick the harness with SWE_AGENT, e.g.
#   SWE_AGENT=codebuddy_code bash run_ppo_qwen35_35b_a3b_swe_4nodes.sh
#
# Usage (from a long-lived shell / tmux on the head node, inside the container):
#   bash examples/coding_agent_rl_ags/run_ppo_qwen35_35b_a3b_swe_4nodes.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export ADVANTAGE_ESTIMATOR=ppo
export SWE_AGENT="${SWE_AGENT:-claude_code}"
export EXP_TAG="${EXP_TAG:-coding_agent_rl_ags_ppo_${SWE_AGENT}_qwen35_35b_a3b_4nodes}"

# No "$@": the CC launcher never reads positional parameters, so forwarding them
# would accept and silently discard anything a user passed.
if (( $# )); then
   echo "ERROR: this launcher takes no positional arguments; set environment variables instead." >&2
   echo "  got: $*" >&2
   exit 2
fi

exec bash "${SCRIPT_DIR}/run_cc_qwen35_35b_a3b_swe_4nodes.sh"
