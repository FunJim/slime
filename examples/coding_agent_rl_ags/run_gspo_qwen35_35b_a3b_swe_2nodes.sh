#!/usr/bin/env bash
# GSPO training on SWE-rebench with an AGS coding-agent harness, 2 nodes.
#
# Thin wrapper, same pattern as run_cbc_*.sh and run_ppo_*.sh: the launcher
# logic lives in run_cc_qwen35_35b_a3b_swe_2nodes.sh and is estimator-agnostic
# -- ADVANTAGE_ESTIMATOR selects the algorithm and algo_args.sh builds the
# corresponding arguments.
#
# Why GSPO on this task: the policy is MoE (Qwen3.5-35B-A3B), so between the
# rollout engine and the trainer a token's router decision can change on its
# own, spiking that token's importance ratio for reasons unrelated to the
# policy update. GSPO clips one sequence-level ratio -- the length-normalized
# geometric mean over the trajectory -- which averages those router flips out
# and matches the granularity of the reward, a single terminal bit per
# trajectory. See arXiv:2507.18071.
#
# Cost vs GRPO: no critic, so no second model, but slime disables the log-prob
# reuse fast path under GSPO, adding one old-log-prob forward per train step.
#
# Note this does NOT address the zero-advantage rate: GSPO keeps GRPO's
# group-relative baseline, so an all-solved or all-failed group still yields
# zero advantage. That is what the PPO arm is for.
#
# Everything the CC launcher accepts works here too, plus EPS_CLIP /
# EPS_CLIP_HIGH (defaults 3e-4 / 4e-4 -- the sequence-level ratio needs a range
# ~3 orders of magnitude tighter than GRPO's 0.2, or clipping never binds).
# Pick the harness with SWE_AGENT, e.g.
#   SWE_AGENT=codebuddy_code bash run_gspo_qwen35_35b_a3b_swe_2nodes.sh
#
# Usage (from a long-lived shell / tmux on the head node, inside the container):
#   bash examples/coding_agent_rl_ags/run_gspo_qwen35_35b_a3b_swe_2nodes.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export ADVANTAGE_ESTIMATOR=gspo
export SWE_AGENT="${SWE_AGENT:-claude_code}"
export EXP_TAG="${EXP_TAG:-coding_agent_rl_ags_gspo_${SWE_AGENT}_qwen35_35b_a3b_2nodes}"

exec bash "${SCRIPT_DIR}/run_cc_qwen35_35b_a3b_swe_2nodes.sh" "$@"
