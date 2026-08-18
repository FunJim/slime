#!/usr/bin/env bash
# GRPO training on SWE-rebench with the CodeBuddy Code (cbc) harness, 2 nodes.
#
# Thin wrapper: the launcher logic lives in run_cc_qwen35_35b_a3b_swe_2nodes.sh
# and is harness-agnostic -- SWE_AGENT selects the harness, and that script sets
# both harnesses' EXTRA_ARGS/EXTRA_ENVS knobs. Duplicating ~370 lines per harness
# would guarantee the two copies drift, which is the failure this rename cleans up.
#
# Everything the CC script accepts works here too, e.g.
#   PROMPT_DATA=... NUM_ROLLOUT=50 bash run_cbc_qwen35_35b_a3b_swe_2nodes.sh
#
# Usage (from a long-lived shell / tmux on the head node, inside the container):
#   bash examples/coding_agent_rl_ags/run_cbc_qwen35_35b_a3b_swe_2nodes.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export SWE_AGENT=codebuddy_code
export EXP_TAG="${EXP_TAG:-coding_agent_rl_ags_cbc_qwen35_35b_a3b_2nodes}"

exec bash "${SCRIPT_DIR}/run_cc_qwen35_35b_a3b_swe_2nodes.sh" "$@"
