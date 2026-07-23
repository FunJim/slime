"""Command-running helpers for AGS sidecar harnesses."""

from __future__ import annotations

import asyncio
import logging
import shlex
import time

from slime.agent.sandbox import EXIT_TIME_BUDGET_EXCEEDED, Sandbox, terminate_process_group

logger = logging.getLogger(__name__)


async def run_root_command(
    sb: Sandbox,
    *,
    workdir: str,
    start_cmd: str,
    env: dict[str, str],
    time_budget_sec: int,
) -> int:
    """Run an AGS sidecar command as root and persist its stream-json output."""

    meta_dir = f"{workdir}/.harness"
    done = f"{meta_dir}/done"
    launcher = f"{meta_dir}/run.sh"
    traj = f"{meta_dir}/trajectory.jsonl"
    pid_file = f"{meta_dir}/pid"
    lock_dir = f"{meta_dir}/spawned"
    launcher_body = (
        "#!/bin/bash\n"
        f"cd {workdir}\n"
        "export HOME=/root\n"
        f"{start_cmd} 2>&1 | tee {shlex.quote(traj)}\n"
        f"echo ${{PIPESTATUS[0]}} > {done}\n"
    )
    await sb.exec(f"mkdir -p {meta_dir} && chmod 777 {meta_dir}", user="root", check=True, timeout=30)
    await sb.write_file(launcher, launcher_body, user="root")
    await sb.exec(f"chmod +x {launcher}", user="root", timeout=30, check=True)

    export_lines = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
    await sb.exec(
        f"mkdir {lock_dir} 2>/dev/null || exit 0; "
        f"rm -f {done} {pid_file}; "
        f"env {export_lines} setsid {launcher} < /dev/null > /dev/null 2>&1 & "
        f"echo $! > {pid_file}",
        user="root",
        timeout=30,
        check=True,
    )

    deadline = time.time() + time_budget_sec
    exit_code = EXIT_TIME_BUDGET_EXCEEDED
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(f"test -f {done} && cat {done}", user="root", timeout=15, check=False)
        if ec == 0 and (out or "").strip():
            exit_code = int((out or "").strip())
            break
    if exit_code == EXIT_TIME_BUDGET_EXCEEDED:
        logger.warning("AGS agent exceeded %ss; terminating process group", time_budget_sec)
        await terminate_process_group(sb, pid_file=pid_file, user="root")
    return exit_code
