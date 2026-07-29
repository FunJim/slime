"""SGLang-to-agent adapter service for AGS generator workers."""

from __future__ import annotations

import asyncio
import logging
import os
from argparse import Namespace

import requests

from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.http_utils import get_host_info
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from .config import AGSGeneratorConfig

logger = logging.getLogger(__name__)


class RemoteAdapterProxy:
    """Control an already-running adapter service via slime control endpoints."""

    def __init__(self, control_url: str) -> None:
        self.control_url = control_url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        response = requests.post(f"{self.control_url}{path}", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
    ) -> None:
        self._post(
            "/_slime/open_session",
            {
                "sid": sid,
                "sampling_defaults": sampling_defaults or {},
                "max_context_tokens": int(max_context_tokens or 0),
            },
        )

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample,
        reward: float = 0.0,
        extra_metadata: dict | None = None,
        wait_timeout: float = 5.0,
    ) -> list[Sample]:
        result = await asyncio.to_thread(
            self._post,
            "/_slime/finish_session",
            {
                "sid": sid,
                "base_sample": base_sample.to_dict(),
                "reward": float(reward),
                "extra_metadata": extra_metadata or {},
                "wait_timeout": float(wait_timeout),
            },
        )
        return [Sample.from_dict(item) for item in result.get("samples", [])]

    async def drop_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        await asyncio.to_thread(
            self._post,
            "/_slime/drop_session",
            {"sid": sid, "wait_timeout": float(wait_timeout)},
        )


def _adapter_build_key(
    args: Namespace,
    config: AGSGeneratorConfig,
    adapter_cls: type,
    sglang_url: str,
) -> tuple:
    """The construction inputs baked into a live adapter, for staleness checks.

    AdapterService is a singleton, so only the first caller's values take
    effect; everything here is ignored on later calls. get_adapter_service
    compares this key and warns rather than silently serving an adapter built
    from a different tokenizer or context budget.
    """
    return (
        getattr(args, "hf_checkpoint", None),
        int(getattr(args, "rollout_max_context_len", 0) or 0),
        getattr(args, "sglang_tool_call_parser", None) or None,
        getattr(args, "sglang_reasoning_parser", None) or None,
        sglang_url,
        adapter_cls.__name__,
        config.fork_merge_threshold,
    )


class AdapterService(metaclass=SingletonMeta):
    def __init__(
        self,
        args: Namespace,
        config: AGSGeneratorConfig,
        adapter_cls: type,
        *,
        port: int | None = None,
    ) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        sglang_url = (
            os.environ.get("SWE_SGLANG_URL")
            or os.environ.get("AGS_GENERATOR_SGLANG_URL")
            or f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        )
        public_base_url = (os.environ.get("ADAPTER_PUBLIC_BASE_URL") or "").strip().rstrip("/")
        if not public_base_url and not config.adapter_public_host:
            raise RuntimeError(
                "ADAPTER_PUBLIC_HOST or ADAPTER_PUBLIC_BASE_URL is not set; "
                "AGS sandboxes need it to reach the adapter"
            )

        self.adapter = adapter_cls(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=config.fork_merge_threshold,
        )
        bind_port = config.adapter_port if port is None else port
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=config.adapter_bind_host,
            port=bind_port,
            thread_name="ags-rollout-adapter",
            runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
        )
        # An ephemeral bind (port=0) means this is the eval fallback adapter,
        # which lives in the RolloutManager actor rather than the rollout-buffer
        # process. That actor is not pinned to the head node, so neither
        # ADAPTER_PUBLIC_BASE_URL nor ADAPTER_PUBLIC_HOST -- both of which name
        # the head and a fixed port -- describe where this adapter is listening.
        # Advertise the local node's own routable IP and the port actually bound.
        public_host = config.adapter_public_host
        if bind_port == 0:
            local_ip = get_host_info()[1]
            if public_base_url:
                logger.warning(
                    "[ags_generator] ignoring ADAPTER_PUBLIC_BASE_URL=%s for the ephemeral eval adapter; "
                    "advertising %s instead",
                    public_base_url,
                    local_ip,
                )
                public_base_url = ""
            public_host = local_ip
        self.adapter_url = public_base_url or f"http://{public_host}:{self.app_handle.port}"
        self.build_key = _adapter_build_key(args, config, adapter_cls, sglang_url)
        logger.info(
            "[ags_generator] tokenizer=%s adapter=%s sglang_url=%s max_context_len=%s tool_parser=%s reasoning_parser=%s",
            args.hf_checkpoint,
            self.adapter_url,
            sglang_url,
            self.max_context_len,
            self.tool_parser,
            self.reasoning_parser,
        )


class RemoteAdapterService(metaclass=SingletonMeta):
    def __init__(self, args: Namespace, config: AGSGeneratorConfig) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        public_base_url = (os.environ.get("ADAPTER_PUBLIC_BASE_URL") or "").strip().rstrip("/")
        if not public_base_url and not config.adapter_public_host:
            raise RuntimeError(
                "ADAPTER_PUBLIC_HOST or ADAPTER_PUBLIC_BASE_URL is not set; "
                "AGS sandboxes need it to reach the adapter"
            )
        control_url = (
            os.environ.get("AGS_EVAL_ADAPTER_CONTROL_URL")
            or os.environ.get("ADAPTER_CONTROL_BASE_URL")
            or f"http://{config.adapter_public_host}:{config.adapter_port}"
        )
        self.adapter = RemoteAdapterProxy(control_url)
        self.adapter_url = public_base_url or f"http://{config.adapter_public_host}:{config.adapter_port}"
        self.control_url = control_url
        logger.info(
            "[ags_generator] using remote adapter control=%s public=%s max_context_len=%s",
            control_url,
            self.adapter_url,
            self.max_context_len,
        )


def _remote_adapter_alive(control_url: str, timeout: float = 5.0) -> bool:
    """True when something is already serving the adapter at control_url."""
    try:
        response = requests.get(f"{control_url.rstrip('/')}/healthz", timeout=timeout)
        return response.ok
    except Exception as exc:
        logger.info("[ags_generator] no adapter reachable at %s (%s)", control_url, type(exc).__name__)
        return False


def get_adapter_service(
    args: Namespace,
    config: AGSGeneratorConfig,
    adapter_cls: type,
    *,
    evaluation: bool = False,
):
    """Return the adapter service backing one AGS rollout or eval pass.

    Training always owns a local adapter. Eval prefers to reuse it -- the
    trajectory trees live in that process, and a second adapter on the same port
    would fail to bind -- but the training adapter only exists once
    entry.run_rollout has built it inside the rollout-buffer process. Under
    --num-rollout 0, or --skip-eval-before-train=0 on rollout 0, eval runs
    first and there is nothing to reuse, so fall back to a local adapter on an
    ephemeral port instead of failing every prompt with a connection error.
    """
    if not evaluation:
        return _local_adapter_service(args, config, adapter_cls)

    control_url = (
        os.environ.get("AGS_EVAL_ADAPTER_CONTROL_URL")
        or os.environ.get("ADAPTER_CONTROL_BASE_URL")
        or f"http://{config.adapter_public_host}:{config.adapter_port}"
    )
    if _remote_adapter_alive(control_url):
        return RemoteAdapterService(args, config)

    # Port 0: the training adapter may still claim config.adapter_port later in
    # this run, and two binds on one port would collide.
    logger.info(
        "[ags_generator] no training adapter at %s; starting a local eval adapter on an ephemeral port",
        control_url,
    )
    return _local_adapter_service(args, config, adapter_cls, port=0)


def _local_adapter_service(
    args: Namespace,
    config: AGSGeneratorConfig,
    adapter_cls: type,
    *,
    port: int | None = None,
):
    """AdapterService singleton, warning when a later caller's args are ignored."""
    service = AdapterService(args, config, adapter_cls, port=port)
    sglang_url = (
        os.environ.get("SWE_SGLANG_URL")
        or os.environ.get("AGS_GENERATOR_SGLANG_URL")
        or f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    )
    wanted = _adapter_build_key(args, config, adapter_cls, sglang_url)
    if wanted != service.build_key:
        logger.warning(
            "[ags_generator] reusing the live adapter built with %s; this call asked for %s. "
            "AdapterService is a singleton, so the requested values are ignored.",
            service.build_key,
            wanted,
        )
    return service
