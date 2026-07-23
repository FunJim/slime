"""Tencent AGS sandbox backend for the slime coding-agent RL experiment.

This module intentionally lives in the experiment directory so the slime source
checkout stays unchanged.  It implements the same async Sandbox contract as
``slime.agent.sandbox.E2BSandbox`` but creates sandboxes through Tencent AGS's
E2B-compatible gateway and sidecar mount.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

from slime.agent.sandbox import ExecResult, FileContent

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    return default if val is None or val == "" else val


def _append_no_proxy(host: str) -> None:
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    for item in (host, ".tencentags.com"):
        if item and item not in parts:
            parts.append(item)
    value = ",".join(parts)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _disable_proxy_for_ags() -> None:
    # The AGS E2B-compatible endpoint is on Tencent internal networking. In this
    # environment routing it through the generic HTTP(S)_PROXY intermittently
    # returns STGW 502 during sandbox create/readiness polling. Disable proxy
    # for SDK-side AGS RPCs in this Ray worker process.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    _append_no_proxy(_env("E2B_DOMAIN", "ap-shanghai.tencentags.com"))


ENVD_CMD = r"""
set -e
ln -sfn /proc/self/fd /dev/fd

for p in \
  runtimes/python runtimes/node \
  agents/craft agents/mini-swe agents/swe agents/claude agents/cbc
  do
    mkdir -p "/opt/${p%/*}"
    ln -sfn "/envd-mount/opt/$p" "/opt/$p"
  done

for t in uv uvx; do
  ln -sfn "/envd-mount/usr/local/bin/$t" "/usr/local/bin/$t"
done

for t in node npm npx; do
  ln -sfn "/opt/runtimes/node/bin/$t" "/usr/local/bin/$t"
done

ln -sfn /opt/agents/claude/bin/claude /usr/local/bin/claude

mkdir -p /etc/pip /root/.config/uv /etc/xdg/uv
printf '[global]\nindex-url = https://mirrors.cloud.tencent.com/pypi/simple\ntrusted-host = mirrors.cloud.tencent.com\n' > /etc/pip.conf
printf '[[index]]\nurl = "https://mirrors.cloud.tencent.com/pypi/simple"\ndefault = true\n' \
  | tee /root/.config/uv/uv.toml /etc/xdg/uv/uv.toml >/dev/null

/envd-mount/usr/bin/envd & exec sleep infinity
""".strip()


class AGSSandbox:
    """Async AGS sandbox wrapper compatible with ``slime.agent.sandbox.Sandbox``."""

    default_lifetime_sec = 3600
    default_rpc_retries = 3
    rpc_backoff_base_sec = 1.0

    def __init__(self, image: str, *, timeout: int | None = None, rpc_retries: int | None = None) -> None:
        self.image = image
        self.timeout = timeout or int(_env("SLIME_AGENT_SANDBOX_LIFETIME_SEC", str(self.default_lifetime_sec)))
        self.rpc_retries = rpc_retries or int(_env("SLIME_AGENT_SANDBOX_RPC_RETRIES", str(self.default_rpc_retries)))
        self._sb = None
        self.sandbox_id = ""

    @staticmethod
    def _resources() -> dict[str, str]:
        raw = _env("AGS_SANDBOX_RESOURCES_JSON", '{"cpu":"2","memory":"4Gi"}')
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"cpu": "2", "memory": "4Gi"}
        except Exception:
            return {"cpu": "2", "memory": "4Gi"}

    def _custom_config(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "imageRegistryType": _env("AGS_IMAGE_REGISTRY_TYPE", "enterprise"),
            "command": ["/bin/sh", "-c"],
            "args": [ENVD_CMD],
            "ports": [{"name": "envd", "port": 49983, "protocol": "TCP"}],
            "probe": {
                "httpGet": {"path": "/health", "port": 49983, "scheme": "HTTP"},
                "readyTimeoutMs": 30000,
                "probeTimeoutMs": 1000,
                "probePeriodMs": 2000,
                "successThreshold": 1,
                "failureThreshold": 15,
            },
            "resources": self._resources(),
        }

    @staticmethod
    def _is_transient_rpc_error(e: BaseException) -> bool:
        name = type(e).__name__
        if name in {
            "ProtocolError",
            "LocalProtocolError",
            "WriteError",
            "ReadError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "SSLError",
        }:
            return True
        msg = str(e)
        if name == "SandboxException":
            return not ("does not exist" in msg or "STOPPED state" in msg)
        return False

    async def _rpc_retry(self, op_name: str, coro_factory, *, idempotent: bool = True):
        last_err = None
        for attempt in range(self.rpc_retries):
            try:
                return await coro_factory()
            except Exception as e:
                if not self._is_transient_rpc_error(e):
                    raise
                if not idempotent:
                    raise
                last_err = e
                if attempt + 1 < self.rpc_retries:
                    backoff = self.rpc_backoff_base_sec * (2**attempt)
                    logger.debug(
                        "[ags_sandbox] %s transient %s retry %d/%d in %.1fs: %s",
                        op_name,
                        type(e).__name__,
                        attempt + 1,
                        self.rpc_retries,
                        backoff,
                        str(e)[:200],
                    )
                    await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    async def __aenter__(self) -> AGSSandbox:
        os.environ.setdefault("E2B_DOMAIN", _env("E2B_DOMAIN", "ap-shanghai.tencentags.com"))
        _disable_proxy_for_ags()

        # The upstream E2B SDK now validates API keys locally and only accepts
        # the public e2b_... format. Tencent AGS intentionally uses ark_...
        # gateway keys while keeping the E2B-compatible HTTP surface, so bypass
        # only this client-side format check and still send the configured key
        # as X-API-KEY to AGS. Keep this experiment-local; do not patch slime.
        import e2b.api as _e2b_api  # type: ignore

        def _allow_ags_api_key(_api_key: str) -> None:
            return None

        _e2b_api.validate_api_key = _allow_ags_api_key
        from e2b import AsyncSandbox  # type: ignore

        template = _env("AGS_BASE_TOOL", "sdt-3fzh6mv6")
        md = {
            "x-custom-config": json.dumps(self._custom_config(), ensure_ascii=False),
            "environment_name": _env("EXPERIMENT_NAME", "slime-coding-agent-rl"),
            "session_id": f"slime-coding-agent-rl-{os.getpid()}-{id(self)}",
        }
        envs = {
            "IS_SANDBOX": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        self._sb = await AsyncSandbox.create(template=template, timeout=self.timeout, metadata=md, envs=envs)
        self.sandbox_id = getattr(self._sb, "sandbox_id", getattr(self._sb, "id", ""))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._sb.kill()
        except Exception as e:
            logger.warning("[ags_sandbox] kill %s failed: %s", self.sandbox_id[:8], e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

        # AGS sidecar commands are root-friendly. Keep the contract's user arg,
        # but fall back to root if a base image lacks the requested user.
        async def _run():
            return await self._sb.commands.run(
                cmd,
                user=user,
                envs=env,
                timeout=timeout,
                on_stdout=lambda _s: None,
                on_stderr=lambda _s: None,
            )

        try:
            res = await self._rpc_retry(f"exec({cmd[:60]!r})", _run, idempotent=idempotent)
            return res.exit_code, res.stdout or "", res.stderr or ""
        except CommandExitException as e:
            if check:
                raise RuntimeError(
                    f"ags exec failed (exit={e.exit_code}): {cmd[:160]}\n{(e.stderr or '')[:800]}"
                ) from None
            return e.exit_code, e.stdout or "", e.stderr or ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            with open(content, "rb") as fp:
                data = fp.read()
            await self.write_file(sandbox_path, data, user=user)
            return
        if isinstance(content, bytes):
            await self._rpc_retry(
                f"write_file({sandbox_path}, bytes={len(content)})",
                lambda: self._sb.files.write(
                    sandbox_path, io.BytesIO(content), user=user, gzip=False, use_octet_stream=True
                ),
            )
            return
        await self._rpc_retry(
            f"write_file({sandbox_path})",
            lambda: self._sb.files.write(sandbox_path, content, user=user),
        )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await self._rpc_retry(
                f"read_file({sandbox_path})", lambda: self._sb.files.read(sandbox_path, user=user)
            )
        except Exception:
            return ""
