"""Best-effort Weave tracing for AGS rollouts."""

from __future__ import annotations

import json
import logging
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class AGSWeaveTrace:
    def __init__(self, args: Namespace, tokenizer, *, enable_token2text: bool = False) -> None:
        self.project = _wandb_project(args)
        self.tokenizer = tokenizer
        self.enable_token2text = enable_token2text
        self.wandb_run_id = getattr(args, "wandb_run_id", None)
        self.wandb_group = getattr(args, "wandb_group", None)
        self.client = None
        if self.project is None:
            return
        try:
            import weave

            self.client = weave.init(self.project)
            if self.wandb_run_id:
                self.client.set_wandb_run_context(run_id=self.wandb_run_id)
        except Exception:
            logger.warning("[ags_generator] failed to initialize Weave tracing", exc_info=True)

    def start_rollout(
        self,
        *,
        instance_id: str,
        session_id: str,
        sample: Sample,
        sampling_params: dict[str, Any],
        agent: str,
    ):
        if self.client is None:
            return None
        inputs = {
            "instance_id": instance_id,
            "session_id": session_id,
            "prompt": sample.prompt,
            "sampling_params": sampling_params,
        }
        attributes = {
            "task_type": "ags",
            "agent": agent,
            "group_index": sample.group_index,
            "sample_index": sample.index,
            "rollout_id": sample.rollout_id,
            "wandb_run_id": self.wandb_run_id,
            "wandb_group": self.wandb_group,
        }
        try:
            return self.client.create_call(
                "slime.ags.rollout",
                inputs,
                attributes=attributes,
                display_name=instance_id,
                use_stack=False,
            )
        except Exception:
            logger.warning("[ags_generator] %s: failed to start Weave trace", instance_id, exc_info=True)
            return None

    def finish_rollout(
        self,
        call,
        *,
        samples: list[Sample],
        trajectory_path: str | None = None,
        output: dict[str, Any] | None = None,
        exception: BaseException | None = None,
    ) -> None:
        if self.client is None or call is None:
            return
        result = dict(output or {})
        try:
            if trajectory_path:
                self._log_trajectory(call, trajectory_path)
            result["samples"] = [self._sample_payload(sample) for sample in samples]
        except Exception:
            logger.warning("[ags_generator] failed to build Weave trace output", exc_info=True)
            result["trace_output_error"] = True
        try:
            self.client.finish_call(call, output=result, exception=exception)
        except Exception:
            logger.warning("[ags_generator] failed to finish Weave root call", exc_info=True)

    def _log_trajectory(self, parent, trajectory_path: str) -> None:
        tool_calls = {}
        for event in iter_trajectory_events(trajectory_path):
            tool_use_id = event.get("tool_use_id")
            if event["kind"] == "tool_result" and tool_use_id in tool_calls:
                self.client.finish_call(
                    tool_calls.pop(tool_use_id),
                    output=event.get("output"),
                    ended_at=event.get("started_at"),
                )
                continue
            child = self.client.create_call(
                f"slime.ags.{event['kind']}",
                event["inputs"],
                parent=parent,
                attributes=event.get("attributes"),
                display_name=event["display_name"],
                use_stack=False,
                started_at=event.get("started_at"),
            )
            if event["kind"] == "tool_call" and tool_use_id:
                tool_calls[tool_use_id] = child
            else:
                self.client.finish_call(child, output=event.get("output"), ended_at=event.get("started_at"))
        for child in tool_calls.values():
            self.client.finish_call(child, output={"missing_tool_result": True})

    def _sample_payload(self, sample: Sample) -> dict[str, Any]:
        tokens = [int(token) for token in sample.tokens]
        response_length = min(max(int(sample.response_length or 0), 0), len(tokens))
        prompt_tokens = tokens[:-response_length] if response_length else tokens
        response_tokens = tokens[-response_length:] if response_length else []
        payload = {
            "status": sample.status.value,
            "reward": sample.reward,
            "remove_sample": sample.remove_sample,
            "prompt_token_ids": prompt_tokens,
            "response_token_ids": response_tokens,
            "response_length": response_length,
            "metadata": sample.metadata,
        }
        if self.enable_token2text:
            payload["prompt_text"] = self.tokenizer.decode(prompt_tokens, skip_special_tokens=False)
            payload["response_text"] = self.tokenizer.decode(response_tokens, skip_special_tokens=False)
        return payload


def iter_trajectory_events(path: str | Path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("[ags_generator] failed to read trajectory for Weave trace: %s", path, exc_info=True)
        return

    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("[ags_generator] skipping invalid trajectory JSON at %s:%d", path, line_number)
            continue
        record_type = record.get("type")
        if record_type == "assistant":
            yield from _assistant_events(record)
        elif record_type == "user":
            yield from _tool_result_events(record)
        elif record_type == "result":
            yield {
                "kind": "result",
                "display_name": "agent result",
                "inputs": _common_fields(record),
                "output": record,
                "started_at": _parse_timestamp(record.get("timestamp")),
            }


def _assistant_events(record: dict[str, Any]):
    message = record.get("message") or {}
    common = _common_fields(record)
    for block in message.get("content") or []:
        block_type = block.get("type")
        if block_type == "tool_use":
            yield {
                "kind": "tool_call",
                "display_name": str(block.get("name") or "tool call"),
                "tool_use_id": block.get("id"),
                "inputs": {**common, "tool_use_id": block.get("id"), "input": block.get("input")},
                "attributes": {"tool_name": block.get("name")},
                "started_at": _parse_timestamp(record.get("timestamp")),
            }
        elif block_type in {"text", "thinking"}:
            text = block.get(block_type)
            if text:
                yield {
                    "kind": block_type,
                    "display_name": f"assistant {block_type}",
                    "inputs": common,
                    "output": {"text": text, "usage": message.get("usage")},
                    "started_at": _parse_timestamp(record.get("timestamp")),
                }


def _tool_result_events(record: dict[str, Any]):
    message = record.get("message") or {}
    common = _common_fields(record)
    for block in message.get("content") or []:
        if block.get("type") == "tool_result":
            yield {
                "kind": "tool_result",
                "display_name": "tool result",
                "tool_use_id": block.get("tool_use_id"),
                "inputs": {**common, "tool_use_id": block.get("tool_use_id")},
                "output": {
                    "content": block.get("content"),
                    "is_error": block.get("is_error", False),
                    "tool_use_result": record.get("tool_use_result"),
                },
                "started_at": _parse_timestamp(record.get("timestamp")),
            }


def _common_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": record.get("session_id"),
        "event_id": record.get("uuid"),
        "parent_tool_use_id": record.get("parent_tool_use_id"),
        "timestamp": record.get("timestamp"),
    }


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _wandb_project(args: Namespace) -> str | None:
    if not getattr(args, "use_wandb", False):
        return None
    wandb_mode = getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE")
    if wandb_mode in {"disabled", "offline"}:
        return None
    project = getattr(args, "wandb_project", None)
    if not project:
        return None
    team = getattr(args, "wandb_team", None)
    return f"{team}/{project}" if team else project
