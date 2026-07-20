"""SGLang-to-agent adapter service for AGS generator workers."""

from __future__ import annotations

import asyncio
import logging
import os
from argparse import Namespace

import requests

from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
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


class AdapterService(metaclass=SingletonMeta):
    def __init__(self, args: Namespace, config: AGSGeneratorConfig, adapter_cls: type) -> None:
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
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=config.adapter_bind_host,
            port=config.adapter_port,
            thread_name="ags-rollout-adapter",
            runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
        )
        self.adapter_url = public_base_url or f"http://{config.adapter_public_host}:{self.app_handle.port}"
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
        logger.info(
            "[ags_generator] using remote adapter control=%s public=%s max_context_len=%s",
            control_url,
            self.adapter_url,
            self.max_context_len,
        )
