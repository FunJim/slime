"""Harness registry for AGS-backed coding agents."""

from __future__ import annotations

import base64
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
    """CodeBuddy Code (cbc) harness using the AGS sidecar binary."""

    name = "codebuddy_code"
    extra_args_env = "SLIME_AGENT_CBC_EXTRA_ARGS"
    extra_envs_env = "SLIME_AGENT_CBC_EXTRA_ENVS"
    max_turns_env = "SLIME_AGENT_CBC_MAX_TURNS"
    max_output_tokens_env = "SLIME_AGENT_CBC_MAX_OUTPUT_TOKENS"
    thinking_enabled_env = "SLIME_AGENT_CBC_THINKING_ENABLED"
    tools_env = "SLIME_AGENT_CBC_TOOLS"

    # Keep the default tool surface close to Claude Code's coding-agent use case
    # while disabling internet search by default for reproducible SWE rollouts.
    allowed_tools = (
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "TaskCreate",
        "TaskUpdate",
        "TaskGet",
        "TaskList",
        "Agent",
    )

    async def install_cli(self, sb: Sandbox) -> None:
        await sb.exec(
            "set -e\n"
            "if [ -x /opt/runtimes/node/bin/node ]; then\n"
            "  ln -sf /opt/runtimes/node/bin/node /usr/local/bin/node\n"
            "  ln -sf /opt/runtimes/node/bin/npm /usr/local/bin/npm\n"
            "  ln -sf /opt/runtimes/node/bin/npx /usr/local/bin/npx\n"
            "fi\n"
            "python_bin=$(ls /opt/runtimes/python/cpython-*/bin/python3 "
            "/envd-mount/opt/runtimes/python/cpython-*/bin/python3 2>/dev/null | head -1 || true)\n"
            'if [ -n "$python_bin" ]; then\n'
            '  ln -sf "$python_bin" /usr/local/bin/python3\n'
            '  pip_bin="${python_bin%/python3}/pip3"\n'
            '  [ -x "$pip_bin" ] && ln -sf "$pip_bin" /usr/local/bin/pip3\n'
            "fi\n"
            "test -x /opt/agents/cbc/bin/cbc\n"
            "ln -sf /opt/agents/cbc/bin/cbc /usr/local/bin/cbc\n"
            "command -v node && node --version\n"
            "command -v cbc && cbc --version",
            user="root",
            check=True,
            timeout=120,
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        models_json = {
            "models": [
                {
                    "id": ctx.model_label,
                    "name": ctx.model_label,
                    "vendor": "OpenAI",
                    "apiKey": ctx.session_id,
                    "url": self._chat_completions_url(ctx.adapter_url),
                    "maxOutputTokens": int(os.environ.get(self.max_output_tokens_env, "16384")),
                    "supportsToolCall": True,
                    "supportsImages": False,
                    "supportsReasoning": True,
                }
            ],
            "availableModels": [ctx.model_label],
        }
        settings_json = {
            "cleanupPeriodDays": 30,
            "includeCoAuthoredBy": False,
            "autoCompactEnabled": True,
            "alwaysThinkingEnabled": _env_flag(self.thinking_enabled_env, default=True),
            "showTokensCounter": False,
            "enablePasteImageFromClipboard": False,
            "enableTerminalProgressBar": False,
            "fileCheckpointingEnabled": False,
            "promptSuggestionEnabled": False,
            "enableAllProjectMcpServers": False,
        }
        models_b64 = _json_b64(models_json)
        settings_b64 = _json_b64(settings_json)
        await sb.exec(
            "set -e\n"
            "mkdir -p /root/.codebuddy/debug /root/.codebuddy/projects /root/.codebuddy/statsig "
            "/home/agent/.codebuddy/debug /home/agent/.codebuddy/projects /home/agent/.codebuddy/statsig\n"
            f"printf %s {shlex.quote(models_b64)} | base64 -d "
            "| tee /root/.codebuddy/models.json /home/agent/.codebuddy/models.json >/dev/null\n"
            f"printf %s {shlex.quote(settings_b64)} | base64 -d "
            "| tee /root/.codebuddy/settings.json /home/agent/.codebuddy/settings.json >/dev/null\n"
            "chown -R agent:agent /home/agent/.codebuddy",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        parts: list[str] = [
            f"--model {shlex.quote(ctx.model_label)}",
            "--verbose",
            "--output-format stream-json",
            "--include-partial-messages",
        ]
        tools = os.environ.get(self.tools_env, ",".join(self.allowed_tools)).strip()
        if tools:
            parts.append(f"--tools {shlex.quote(tools)}")
            parts.append("--disallowedTools WebSearch")
        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            # Keep caller-provided flags before the non-variadic tail and prompt.
            parts.append(extra)
        if not _env_flag(self.thinking_enabled_env, default=True):
            parts.append("--effort none")
        parts.append(f"--max-turns {int(os.environ.get(self.max_turns_env, '100'))}")
        parts.append("-y")

        session_log_dir = f"{ctx.workdir}/.harness/codebuddy_sessions"
        raw_cmd = (
            f"cbc {' '.join(parts)} {shlex.quote(prompt)}; "
            "rc=$?; "
            f"mkdir -p {shlex.quote(session_log_dir)}/projects; "
            f"cp -r /root/.codebuddy/projects/. {shlex.quote(session_log_dir)}/projects/ 2>/dev/null || true; "
            "exit $rc"
        )
        cmd = f"bash -lc {shlex.quote(raw_cmd)}"

        env = {
            "OPENAI_API_KEY": ctx.session_id,
            "OPENAI_BASE_URL": f"{ctx.adapter_url}/v1",
            "CBC_API_KEY": ctx.session_id,
            "CBC_BASE_URL": self._chat_completions_url(ctx.adapter_url),
            "NO_COLOR": "1",
            "CI": "1",
            "TERM": "dumb",
            "IS_SANDBOX": "1",
        }
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))
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

    @staticmethod
    def _chat_completions_url(adapter_url: str) -> str:
        url = adapter_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"


def _json_b64(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


HARNESS_REGISTRY: dict[str, tuple[type[BaseHarness], type]] = {
    "claude_code": (AGSSidecarClaudeCodeHarness, AnthropicAdapter),
    "codex": (CodexHarness, OpenAIAdapter),
    "codebuddy_code": (CodeBuddyCodeHarness, OpenAIAdapter),
}


def resolve_agent(agent_name: str) -> tuple[type[BaseHarness], type]:
    if agent_name not in HARNESS_REGISTRY:
        raise ValueError(f"SWE_AGENT={agent_name!r} not in {sorted(HARNESS_REGISTRY)}")
    return HARNESS_REGISTRY[agent_name]
