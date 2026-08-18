#!/usr/bin/env bash
# Score one checkpoint on SWE-bench Verified with the CodeBuddy Code (cbc)
# harness, 2 nodes, no training.
#
# Thin wrapper: the launcher logic lives in eval_cc_qwen35_35b_a3b_swe_2nodes.sh
# and is harness-agnostic -- SWE_AGENT selects the harness, and that script sets
# both harnesses' EXTRA_ARGS/EXTRA_ENVS knobs -- so there is one copy to keep correct.
#
# Every variable the CC eval script reads works here too:
#   LOAD_DIR=<ckpt_root> CKPT_STEP=99 EVAL_DATA=<jsonl> \
#     bash examples/coding_agent_rl_ags/eval_cbc_qwen35_35b_a3b_swe_2nodes.sh
#
# Omit LOAD_DIR/CKPT_STEP to score the untrained HF model as the baseline.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export SWE_AGENT=codebuddy_code
export EXP_TAG="${EXP_TAG:-coding_agent_rl_ags_eval_cbc_qwen35_35b_a3b${CKPT_STEP:+_step${CKPT_STEP}}}"

exec bash "${SCRIPT_DIR}/eval_cc_qwen35_35b_a3b_swe_2nodes.sh" "$@"
