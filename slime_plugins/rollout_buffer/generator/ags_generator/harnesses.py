"""Harness registry for AGS-backed coding agents."""

from __future__ import annotations

import json
import os
import shlex

from slime.agent.adapters import AnthropicAdapter, OpenAIAdapter
from slime.agent.harness import CodexHarness
from slime.agent.harness.common import BaseHarness, HarnessContext
from slime.agent.sandbox import Sandbox

from .runner import run_root_command


class AGSSidecarClaudeCodeHarness(BaseHarness):
    """Claude Code harness using the AGS sidecar binary instead of npm install."""

    name = "claude_code"
    extra_args_env = "SLIME_AGENT_CC_EXTRA_ARGS"
    extra_envs_env = "SLIME_AGENT_CC_EXTRA_ENVS"
    launch_flags = (
        "--dangerously-skip-permissions "
        "--verbose --output-format stream-json "
        "--include-partial-messages --include-hook-events"
    )
    static_env = {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }

    async def install_cli(self, sb: Sandbox) -> None:
        await sb.exec(
            "command -v node && node --version && command -v claude && claude --version",
            user="root",
            check=True,
            timeout=120,
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        await sb.exec(
            "mkdir -p /root/.claude /home/agent/.claude && "
            f"echo {shlex.quote(settings)} | tee "
            "/root/.claude.json /root/.claude/settings.json "
            "/home/agent/.claude.json /home/agent/.claude/settings.json > /dev/null && "
            "chown -R agent:agent /home/agent/.claude /home/agent/.claude.json",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        cmd = f"/usr/local/bin/claude -p {shlex.quote(prompt)} {self.launch_flags}"
        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            cmd = f"{cmd} {extra}"

        env = {
            "ANTHROPIC_BASE_URL": ctx.adapter_url,
            "ANTHROPIC_AUTH_TOKEN": ctx.session_id,
            "ANTHROPIC_API_KEY": ctx.session_id,
            "ANTHROPIC_MODEL": ctx.model_label,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": ctx.model_label,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": ctx.model_label,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": ctx.model_label,
            "CLAUDE_CODE_SUBAGENT_MODEL": ctx.model_label,
            **self.static_env,
            "IS_SANDBOX": "1",
        }
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))
        if os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"):
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = os.environ["CLAUDE_CODE_MAX_OUTPUT_TOKENS"]

        return await run_root_command(
            sb,
            workdir=ctx.workdir,
            start_cmd=cmd,
            env=env,
            time_budget_sec=time_budget_sec,
        )

    async def run(
        self,
        sb: Sandbox,
        *,
        workdir: str,
        session_id: str,
        adapter_url: str,
        time_budget_sec: int,
        prompt: str,
    ) -> int:
        from slime.agent import sandbox as agent_sandbox

        await agent_sandbox.ensure_agent_user(sb, workdir)
        ctx = HarnessContext(workdir=workdir, session_id=session_id, adapter_url=adapter_url)
        await self.write_config(sb, ctx)
        return await self.launch_and_wait(sb, ctx, prompt, time_budget_sec)


class CodeBuddyCodeHarness(BaseHarness):
    """Placeholder for AGS CodeBuddy Code sidecar integration.

    The package-level registry can add the harness once its non-interactive CLI
    contract is finalized without touching the rollout orchestration code.
    """

    name = "codebuddy_code"

    async def install_cli(self, sb: Sandbox) -> None:
        raise NotImplementedError("CodeBuddy Code AGS sidecar CLI contract is not configured yet")

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        raise NotImplementedError("CodeBuddy Code AGS sidecar CLI contract is not configured yet")

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        raise NotImplementedError("CodeBuddy Code AGS sidecar CLI contract is not configured yet")


HARNESS_REGISTRY: dict[str, tuple[type[BaseHarness], type]] = {
    "claude_code": (AGSSidecarClaudeCodeHarness, AnthropicAdapter),
    "codex": (CodexHarness, OpenAIAdapter),
    "codebuddy_code": (CodeBuddyCodeHarness, OpenAIAdapter),
}


def resolve_agent(agent_name: str) -> tuple[type[BaseHarness], type]:
    if agent_name not in HARNESS_REGISTRY:
        raise ValueError(f"SWE_AGENT={agent_name!r} not in {sorted(HARNESS_REGISTRY)}")
    return HARNESS_REGISTRY[agent_name]
