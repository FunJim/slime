"""SGLang-to-agent adapter service for AGS generator workers."""

from __future__ import annotations

import logging
import os
from argparse import Namespace

from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer

from .config import AGSGeneratorConfig

logger = logging.getLogger(__name__)


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
        if not config.adapter_public_host:
            raise RuntimeError("ADAPTER_PUBLIC_HOST is not set; AGS sandboxes need it to reach the adapter")

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
        self.adapter_url = f"http://{config.adapter_public_host}:{self.app_handle.port}"
        logger.info(
            "[ags_generator] tokenizer=%s adapter=%s sglang_url=%s max_context_len=%s tool_parser=%s reasoning_parser=%s",
            args.hf_checkpoint,
            self.adapter_url,
            sglang_url,
            self.max_context_len,
            self.tool_parser,
            self.reasoning_parser,
        )
