"""Configuration for the AGS coding-agent rollout-buffer generator."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AGSGeneratorConfig:
    agent_name: str
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    fork_merge_threshold: int | None
    agent_time_budget_sec: int
    eval_timeout_sec: int
    eval_bootstrap_cmd: str | None
    rollout_guard_sec: int
    boot_concurrency: int
    boot_retries: int
    artifact_dir: str | None
    prompt: str

    @classmethod
    def from_env(cls) -> AGSGeneratorConfig:
        agent_time_budget = int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "1800"))
        eval_timeout = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
        guard = int(os.environ.get("SWE_ROLLOUT_GUARD_SEC", "0") or 0) or (agent_time_budget + eval_timeout + 180)
        fork = int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        return cls(
            agent_name=os.environ.get("SWE_AGENT", "claude_code"),
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18001")),
            fork_merge_threshold=fork,
            agent_time_budget_sec=agent_time_budget,
            eval_timeout_sec=eval_timeout,
            eval_bootstrap_cmd=os.environ.get("SWE_EVAL_BOOTSTRAP_CMD") or None,
            rollout_guard_sec=guard,
            boot_concurrency=int(os.environ.get("SWE_BOOT_CONCURRENCY", "16")),
            boot_retries=int(os.environ.get("SWE_BOOT_RETRIES", "2")),
            artifact_dir=os.environ.get("TRAJECTORY_DUMP_DIR", "").strip() or None,
            prompt=os.environ.get(
                "SWE_CC_PROMPT",
                "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
                "Edit source files only (do NOT touch tests). After editing, run the relevant "
                "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
                "NOT commit. When finished, print a one-line summary and exit.",
            ),
        )
