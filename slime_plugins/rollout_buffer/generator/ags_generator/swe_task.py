"""SWE task operations used by the AGS rollout-buffer generator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from slime.agent import sandbox as agent_sandbox
from slime.agent.adapters.common import flatten_content
from slime.agent.sandbox import Sandbox
from slime.utils.types import Sample

from .ags_sandbox import AGSSandbox

logger = logging.getLogger(__name__)

_PATCH = "/workspace/__cagent_patch__.diff"
_PRE = "/workspace/__cagent_pre__.sh"
_F2P = "/workspace/__cagent_f2p__.py"
_SWEPRO_DIR = "/workspace/swepro_eval"


def get_metadata(sample: Sample) -> dict[str, Any]:
    m = sample.metadata or {}
    rem = m.get("remote_env_info") or {}
    label = sample.label if (isinstance(sample.label, str) and len(sample.label) < 256) else None
    return {
        "instance_id": m.get("instance_id") or rem.get("instance_id") or label or "unknown",
        "image": m.get("image") or rem.get("image_url"),
        "workdir": m.get("workdir") or rem.get("workdir"),
        "problem_statement": m.get("problem_statement") or _coerce_prompt(sample.prompt),
        "swepro": m.get("swepro"),
        "eval_cmd": m.get("eval_cmd"),
        "f2p_script": rem.get("f2p_script"),
        "pre_commands": m.get("pre_commands") or rem.get("pre_commands"),
    }


def _coerce_prompt(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return flatten_content(message.get("content"))
    return ""


async def prepare_workspace(sb: Sandbox, workdir: str, md: dict[str, Any]) -> None:
    await agent_sandbox.ensure_agent_user(sb, workdir)
    swepro = md.get("swepro")
    if swepro:
        await apply_before_repo_set_cmd(sb, workdir, swepro)
    pre_commands = md.get("pre_commands")
    if pre_commands:
        await apply_pre_commands(sb, workdir, pre_commands)
    await sb.write_file(f"{workdir}/PROBLEM_STATEMENT.md", md.get("problem_statement") or "", user="agent")


async def apply_before_repo_set_cmd(sb: Sandbox, workdir: str, swepro: dict[str, Any]) -> None:
    before = swepro.get("before_repo_set_cmd")
    if not before:
        return
    payload = f"set -e\ncd {workdir}\n{before}\n"
    await sb.exec(
        "mkdir -p /workspace/swepro_setup && chown agent:agent /workspace/swepro_setup", user="root", check=True
    )
    await sb.write_file("/workspace/swepro_setup/before.sh", payload, user="agent")
    await sb.exec("bash /workspace/swepro_setup/before.sh", user="agent", check=False, timeout=600)


async def apply_pre_commands(sb: Sandbox, workdir: str, pre: list[str] | str) -> None:
    body = pre.replace("\\n", "\n") if isinstance(pre, str) else "\n".join(c for c in (pre or []) if c)
    await sb.write_file(_PRE, "set -e\n" + body, user="agent")
    await sb.exec(f"chmod 755 {_PRE} && cd {workdir} && bash {_PRE}", user="agent", check=False, timeout=600)


async def git_diff(sb: Sandbox, workdir: str) -> str:
    cmd = f"cd {workdir} && git add -N . && git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/'"
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120)
    return out


async def evaluate(
    *,
    image: str,
    workdir: str,
    diff_text: str,
    swepro: dict[str, Any] | None = None,
    eval_cmd: str | None = None,
    f2p_script: str | None = None,
    pre_commands: list[str] | str | None = None,
    eval_bootstrap_cmd: str | None = None,
    timeout_sec: int = 600,
) -> tuple[float, bool]:
    if not (swepro or eval_cmd or f2p_script):
        logger.warning("[ags_generator.evaluate] no swepro/eval_cmd/f2p_script; reward=0")
        return 0.0, True

    async with AGSSandbox(image) as ev:
        await agent_sandbox.ensure_agent_user(ev, workdir)
        if swepro:
            await _setup_swepro_assets(ev, swepro)
            await apply_before_repo_set_cmd(ev, workdir, swepro)
        if pre_commands:
            await apply_pre_commands(ev, workdir, pre_commands)
        if eval_bootstrap_cmd:
            await _run_eval_bootstrap(ev, workdir, eval_bootstrap_cmd, timeout=min(600, max(120, timeout_sec)))

        applied = await _apply_diff(ev, workdir, diff_text)
        if not applied:
            return 0.0, False

        if swepro:
            reward = await _run_swepro(ev, workdir, swepro, timeout_sec)
        elif eval_cmd:
            reward = await _run_eval_cmd(ev, workdir, eval_cmd, timeout_sec)
        else:
            reward = await _run_f2p_script(ev, workdir, f2p_script or "", timeout_sec)
        return reward, True


async def _setup_swepro_assets(ev: Sandbox, swepro: dict[str, Any]) -> None:
    await ev.exec(f"mkdir -p {_SWEPRO_DIR} && chmod 777 {_SWEPRO_DIR}", user="root", check=True)
    for key, dst in [("run_script_path", "run_script.sh"), ("parser_script_path", "parser.py")]:
        host_path = swepro.get(key)
        if host_path:
            await ev.write_file(f"{_SWEPRO_DIR}/{dst}", Path(host_path), user="root")
    await ev.exec(f"chmod 755 {_SWEPRO_DIR}/* && chown -R agent:agent {_SWEPRO_DIR}", user="root", check=True)


async def _apply_diff(ev: Sandbox, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True
    await ev.write_file(_PATCH, diff_text, user="agent")
    for cmd in [
        f"cd {workdir} && git apply --3way --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && git apply --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && patch -p1 --no-backup-if-mismatch < {_PATCH}",
    ]:
        ec, _, _ = await ev.exec(cmd, user="agent", check=False, timeout=120)
        if ec == 0:
            return True
    return False


async def _run_swepro(ev: Sandbox, workdir: str, swepro: dict[str, Any], timeout: int) -> float:
    test_arg = ",".join(swepro.get("selected_test_files") or [])
    stdout_f = f"{_SWEPRO_DIR}/stdout.log"
    stderr_f = f"{_SWEPRO_DIR}/stderr.log"
    result_f = f"{_SWEPRO_DIR}/result.json"
    await ev.exec(
        f"cd {workdir} && bash {_SWEPRO_DIR}/run_script.sh {json.dumps(test_arg)} > {stdout_f} 2> {stderr_f} || true",
        user="agent",
        check=False,
        timeout=timeout,
    )
    await ev.exec(
        f"python3 {_SWEPRO_DIR}/parser.py {stdout_f} {stderr_f} {result_f}", user="agent", check=False, timeout=120
    )
    raw = await ev.read_file(result_f, user="agent")
    parsed = json.loads(raw) if raw else {"tests": []}
    passed = {t["name"] for t in parsed.get("tests", []) if t.get("status") == "PASSED"}
    required = set(swepro.get("fail_to_pass") or []) | set(swepro.get("pass_to_pass") or [])
    solved = bool(required) and required.issubset(passed)
    return 1.0 if solved else 0.0


async def _run_eval_cmd(ev: Sandbox, workdir: str, cmd: str, timeout: int) -> float:
    ec, _, _ = await ev.exec(f"cd {workdir} && {cmd}", user="agent", check=False, timeout=timeout)
    return 1.0 if ec == 0 else 0.0


async def _run_eval_bootstrap(ev: Sandbox, workdir: str, cmd: str, timeout: int) -> None:
    ec, out, err = await ev.exec(f"cd {workdir} && {cmd}", user="agent", check=False, timeout=timeout)
    logger.info(
        "[ags_generator.evaluate] bootstrap exit=%s stdout_tail=%r stderr_tail=%r",
        ec,
        (out or "")[-2000:],
        (err or "")[-2000:],
    )


async def _run_f2p_script(ev: Sandbox, workdir: str, script: str, timeout: int) -> float:
    await ev.write_file(_F2P, script, user="agent")
    ec, out, err = await ev.exec(
        f"cd {workdir} && export PATH=/opt/conda/bin:/usr/local/bin:$PATH; "
        f"if [ -x /opt/conda/bin/python ]; then /opt/conda/bin/python {_F2P}; "
        f"elif command -v python >/dev/null 2>&1; then python {_F2P}; "
        f"else python3 {_F2P}; fi",
        user="agent",
        check=False,
        timeout=timeout,
    )
    logger.info(
        "[ags_generator.evaluate] f2p exit=%s stdout_tail=%r stderr_tail=%r",
        ec,
        (out or "")[-4000:],
        (err or "")[-4000:],
    )
    return 1.0 if ec == 0 else 0.0
