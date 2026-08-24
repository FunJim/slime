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
# Cost vs GRPO: no critic, so no second model. slime disables the log-prob reuse
# fast path under GSPO, which costs one extra old-log-prob forward -- but only at
# nspr=1, since the fast path requires a single global batch anyway; at the nspr=3
# default below both estimators pay it (measured: 232 s per rollout, ~9%).
#
# Note this does NOT address the zero-advantage rate: GSPO keeps GRPO's
# group-relative baseline, so an all-solved or all-failed group still yields
# zero advantage. That is what the PPO arm is for.
#
# Everything the CC launcher accepts works here too, plus EPS_CLIP /
# EPS_CLIP_HIGH (defaults 3e-4 / 4e-4 -- the sequence-level ratio needs a range
# ~3 orders of magnitude tighter than GRPO's 0.2, or clipping never binds) and
# NUM_STEPS_PER_ROLLOUT / ROLLOUT_BATCH_SIZE (defaults 3 and 6, see below).
# Pick the harness with SWE_AGENT, e.g.
#   SWE_AGENT=codebuddy_code bash run_gspo_qwen35_35b_a3b_swe_2nodes.sh
#
# Usage (from a long-lived shell / tmux on the head node, inside the container):
#   bash examples/coding_agent_rl_ags/run_gspo_qwen35_35b_a3b_swe_2nodes.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export ADVANTAGE_ESTIMATOR=gspo
export SWE_AGENT="${SWE_AGENT:-claude_code}"

# ---- more than one optimizer step per rollout, or this is not a GSPO run -----
# train_actor computes old_log_probs ONCE per rollout, so at the launcher's usual
# NUM_STEPS_PER_ROLLOUT=1 the single step happens at exactly those weights: the
# importance ratio is identically 1, the clip range never binds, and the objective
# is GRPO's. That is not a theory -- a 1-rollout run measured train/ppo_kl and
# train/pg_clipfrac as exactly 0.0 at nspr=1, and algo_args.sh warns if you put it
# back. GSPO only differs from GRPO on steps 1..n-1, which are off-policy with
# respect to those old_log_probs.
#
# 3 rather than 2, from a 20-rollout run on 4 nodes (60 steps, 20 per position):
#
#   step position   pg_clipfrac mean   |ppo_kl| mean   grad_norm mean
#   s0 (on-policy)  0.000 (20/20)      0              1.384
#   s1              0.338              2.2e-4         0.646
#   s2              0.217              1.8e-4         0.624
#
# The third step is not the runt of the litter: it is clipped LESS than the second
# and contributes the same gradient magnitude, so there is no marginal reason to
# stop at 2. Off-policy clipping averaged 0.27 over the first ten rollouts and 0.28
# over the last ten -- stable, with three quarters of the tokens still contributing.
# An extra step costs ~240 s against a ~2450 s rollout cycle (~10%).
#
# Watch train/pg_clipfrac: sustained >0.5 on s1 means the policy outruns the trust
# region each step -- lower this or raise EPS_CLIP. The run above peaked at 0.58 on
# a single step with a 0.338 mean, which is comfortable.
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-3}"
# 6 x 8 = 48 = 3 x 16 divides exactly. It has to: build_dp_schedule drops the
# trailing remainder silently, so the launcher's usual rbs=8 would cost one
# trajectory (~400 s of agent time) every rollout. n_samples_per_prompt stays at
# the launcher default of 8 -- it is the group size the advantages come from.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-6}"
export EXP_TAG="${EXP_TAG:-coding_agent_rl_ags_gspo_${SWE_AGENT}_qwen35_35b_a3b_2nodes}"

exec bash "${SCRIPT_DIR}/run_cc_qwen35_35b_a3b_swe_2nodes.sh" "$@"
