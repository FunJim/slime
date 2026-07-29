"""Configuration for the AGS coding-agent rollout-buffer generator."""

from __future__ import annotations

import os
from dataclasses import dataclass

# What to do when the agent exits 0 but leaves no diff.
#   "off"     -- skip the check entirely
#   "metrics" -- label the samples and count it; no behaviour change (default)
#   "abort"   -- additionally drop the rollout via Sample.Status.ABORTED
#
# "abort" is deliberately not the default. An empty-patch trajectory still holds
# real on-policy tokens whose reward of 0 is correct, i.e. the negative half of
# a GRPO group; at the observed ~33% empty-patch rate, masking those out biases
# the group baseline toward successes. It also only requeues under
# slime.rollout.fully_async_rollout -- on the rollout_buffer path the AGS
# scripts actually use, an aborted sample is shipped with its loss masked and
# never retried.
EMPTY_PATCH_GUARD_POLICIES = frozenset({"off", "metrics", "abort"})

# How the agent learns what to do.
#   "instruction" -- send SWE_CC_PROMPT, which points at PROBLEM_STATEMENT.md
#   "dataset"     -- send the row's own prompt field verbatim
#
# Set independently for training (SWE_PROMPT_STYLE) and periodic eval
# (SWE_EVAL_PROMPT_STYLE), because the two want different things: eval should
# measure the model the way a benchmark would, handing over the task text
# directly the way Harbor does, while training may prefer the agent to work for
# it. Converted Harbor rows carry the same instruction.md text in both the prompt
# field and metadata.problem_statement, so the styles differ in *when* the agent
# sees the task, not in what it reads.
#
# Only "instruction" writes PROBLEM_STATEMENT.md into the workspace (see
# swe_task.prepare_workspace) -- under "dataset" the prompt already carries the
# task text, so the file would just be a stray artifact in the repo.
PROMPT_STYLES = frozenset({"dataset", "instruction"})


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
    eval_isolated_sandbox: bool
    rollout_guard_sec: int
    boot_concurrency: int
    rollout_concurrency: int
    boot_retries: int
    artifact_dir: str | None
    enable_token2text: bool
    prompt: str
    empty_patch_guard: str
    prompt_style: str
    eval_prompt_style: str

    @classmethod
    def from_env(cls, *, enable_token2text: bool = False) -> AGSGeneratorConfig:
        agent_time_budget = int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "1800"))
        eval_timeout = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
        guard = int(os.environ.get("SWE_ROLLOUT_GUARD_SEC", "0") or 0) or (agent_time_budget + eval_timeout + 180)
        fork = int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        rollout_concurrency = int(os.environ.get("SWE_ROLLOUT_CONCURRENCY", "1"))
        return cls(
            agent_name=os.environ.get("SWE_AGENT", "claude_code"),
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18001")),
            fork_merge_threshold=fork,
            agent_time_budget_sec=agent_time_budget,
            eval_timeout_sec=eval_timeout,
            eval_bootstrap_cmd=os.environ.get("SWE_EVAL_BOOTSTRAP_CMD") or None,
            eval_isolated_sandbox=_env_flag("SWE_EVAL_ISOLATED_SANDBOX", default=False),
            rollout_guard_sec=guard,
            boot_concurrency=int(os.environ.get("SWE_BOOT_CONCURRENCY", "16")),
            rollout_concurrency=max(1, rollout_concurrency),
            boot_retries=int(os.environ.get("SWE_BOOT_RETRIES", "2")),
            artifact_dir=os.environ.get("TRAJECTORY_DUMP_DIR", "").strip() or None,
            enable_token2text=enable_token2text,
            prompt=os.environ.get(
                "SWE_CC_PROMPT",
                "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. Edit source files only (do NOT touch tests). After editing, run the relevant tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do NOT commit. When finished, print a one-line summary and exit.",
            ),
            empty_patch_guard=_empty_patch_guard_policy(os.environ.get("SWE_EMPTY_PATCH_GUARD")),
            prompt_style=_prompt_style("SWE_PROMPT_STYLE", default="instruction"),
            eval_prompt_style=_prompt_style("SWE_EVAL_PROMPT_STYLE", default="dataset"),
        )

    def prompt_style_for(self, *, evaluation: bool) -> str:
        """Prompt style for this rollout: eval and training are set separately."""
        return self.eval_prompt_style if evaluation else self.prompt_style


def _empty_patch_guard_policy(raw: str | None) -> str:
    """Validate SWE_EMPTY_PATCH_GUARD, defaulting to "metrics".

    Raising here fails the run at construction time; silently falling back to a
    default would disable the guard on a typo without anyone noticing.
    """
    policy = (raw or "metrics").strip().lower()
    if policy not in EMPTY_PATCH_GUARD_POLICIES:
        raise ValueError(f"SWE_EMPTY_PATCH_GUARD={raw!r} is not one of {sorted(EMPTY_PATCH_GUARD_POLICIES)}")
    return policy


def _prompt_style(env_name: str, *, default: str) -> str:
    """Validate a prompt-style env var, naming it in the error."""
    raw = os.environ.get(env_name)
    style = (raw or default).strip().lower()
    if style not in PROMPT_STYLES:
        raise ValueError(f"{env_name}={raw!r} is not one of {sorted(PROMPT_STYLES)}")
    return style


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}
