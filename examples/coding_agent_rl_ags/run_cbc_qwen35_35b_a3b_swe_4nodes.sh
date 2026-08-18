#!/usr/bin/env bash
# GRPO training on SWE-rebench with the CodeBuddy Code (cbc) harness, 4 nodes.
#
# Thin wrapper around the harness-agnostic 4-node launcher; see
# run_cbc_qwen35_35b_a3b_swe_2nodes.sh for why these are wrappers rather than
# copies.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export SWE_AGENT=codebuddy_code
export EXP_TAG="${EXP_TAG:-coding_agent_rl_ags_cbc_qwen35_35b_a3b_4nodes}"

exec bash "${SCRIPT_DIR}/run_cc_qwen35_35b_a3b_swe_4nodes.sh" "$@"
