"""rollout_buffer generator entry point for AGS coding-agent tasks."""

from __future__ import annotations

import asyncio
import copy
import logging
from argparse import Namespace
from typing import Any

import requests

from slime.utils.misc import SingletonMeta
from slime.utils.types import Sample

from .config import AGSGeneratorConfig
from .rollout import AGSRolloutRunner
from .serialization import output_item_from_samples, samples_from_payload
from .source import AGSPromptSource

TASK_TYPE = "ags"

logger = logging.getLogger(__name__)


class _AGSGenerateState(metaclass=SingletonMeta):
    """Process-local state for the per-sample AGS generate hook.

    The default slime eval loop calls the custom generate function once per
    sample. Keep the runner and concurrency semaphore process-local so periodic
    eval reuses the same adapter service instead of rebuilding it for every
    prompt.
    """

    def __init__(self, args: Namespace, *, evaluation: bool = False) -> None:
        self.config = AGSGeneratorConfig.from_env(
            enable_token2text=_as_bool(getattr(args, "enable_token2text", False))
        )
        self.runner = AGSRolloutRunner(args, self.config, use_remote_adapter=evaluation)
        self.semaphore = asyncio.Semaphore(self.config.rollout_concurrency)


async def generate(
    args: Namespace,
    base_sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Sample]:
    """Generate one AGS rollout for slime's default rollout/eval loop.

    This hook lets periodic eval use the AGS runner via:

      --eval-function-path slime.rollout.sglang_rollout.generate_rollout
      --custom-generate-function-path slime_plugins.rollout_buffer.generator.ags_generator.generate

    Training keeps AGSRolloutRunner's trainable segment output. Eval collapses
    the possibly multi-segment trajectory into one scored sample so pass-rate
    metrics count one eval attempt per prompt.
    """

    state = _AGSGenerateState(args, evaluation=evaluation)
    async with state.semaphore:
        samples = await state.runner.generate(base_sample, sampling_params)

    if not evaluation:
        return samples
    return [_collapse_eval_samples(base_sample, samples)]


def run_rollout(data: dict[str, Any]) -> str:
    """Generate AGS coding-agent trajectories and stream them into buffer.py."""

    logging.basicConfig(level=getattr(logging, data.get("log_level", "INFO"), logging.INFO))
    args = _build_args(data)
    config = AGSGeneratorConfig.from_env(enable_token2text=_as_bool(data.get("enable_token2text", False)))
    source = AGSPromptSource(args)
    runner = AGSRolloutRunner(args, config)
    remote_buffer_url = data["remote_buffer_url"].rstrip("/") + "/buffer/write"
    num_epoch = int(data.get("num_epoch", 1))
    groups_per_epoch = int(
        data.get("num_groups_per_epoch") or data.get("rollout_batch_size") or args.rollout_batch_size
    )
    skip_instance_ids = data.get("skip_instance_ids") or []
    stop_event = data.get("_stop_event")
    rollout_job_id = data.get("_rollout_job_id")

    logger.info(
        "[ags_generator] start task_type=%s groups_per_epoch=%s repeats=%s epochs=%s concurrency=%s buffer=%s",
        TASK_TYPE,
        groups_per_epoch,
        args.n_samples_per_prompt,
        num_epoch,
        config.rollout_concurrency,
        remote_buffer_url,
    )

    async def _run_sample(epoch: int, sample: Sample) -> None:
        outputs = await runner.generate(sample, args.sampling_params)
        first = outputs[0]
        instance_id = _instance_id(first)
        item = output_item_from_samples(
            outputs,
            instance_id=instance_id,
            extra_info={
                "epoch": epoch,
                "task_type": TASK_TYPE,
                "reward": first.reward,
                **(first.metadata or {}),
            },
        )
        if rollout_job_id is not None:
            item["rollout_job_id"] = rollout_job_id
        await asyncio.to_thread(_send_data_to_buffer, remote_buffer_url, item)

    async def _run_epoch(epoch: int, samples: list[Sample]) -> None:
        async def _guarded(sample: Sample) -> None:
            try:
                await _run_sample(epoch, sample)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                instance_id = _instance_id(sample)
                logger.exception(
                    "[ags_generator] %s: sample task failed; writing aborted rollout: %s",
                    instance_id,
                    exc,
                )
                outputs = runner._abort_result(sample, f"task_exception:{type(exc).__name__}", instance_id)
                first = outputs[0]
                item = output_item_from_samples(
                    outputs,
                    instance_id=instance_id,
                    extra_info={
                        "epoch": epoch,
                        "task_type": TASK_TYPE,
                        "reward": first.reward,
                        **(first.metadata or {}),
                    },
                )
                if rollout_job_id is not None:
                    item["rollout_job_id"] = rollout_job_id
                try:
                    await asyncio.to_thread(_send_data_to_buffer, remote_buffer_url, item)
                except Exception as send_exc:
                    logger.exception(
                        "[ags_generator] %s: failed to write aborted rollout after task failure: %s",
                        instance_id,
                        send_exc,
                    )

        pending: set[asyncio.Task[None]] = set()
        sample_iter = iter(samples)
        exhausted = False

        while pending or not exhausted:
            while not exhausted and len(pending) < config.rollout_concurrency and not _stop_requested(stop_event):
                try:
                    sample = next(sample_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(asyncio.create_task(_guarded(sample)))

            if not pending:
                break

            done, pending = await asyncio.wait(pending, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()

            if _stop_requested(stop_event):
                logger.info(
                    "[ags_generator] stop requested; cancelling %d in-flight tasks for epoch %d",
                    len(pending),
                    epoch,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                pending.clear()
                break

    for epoch in range(num_epoch):
        if _stop_requested(stop_event):
            logger.info("[ags_generator] stop requested before epoch %d; exiting", epoch)
            break
        samples = source.get_repeated_samples(groups_per_epoch, skip_instance_ids=skip_instance_ids)
        skip_instance_ids = []
        asyncio.run(_run_epoch(epoch, samples))
        if _stop_requested(stop_event):
            logger.info("[ags_generator] stop requested after epoch %d; exiting", epoch)
            break
    return "finished"


def _collapse_eval_samples(base_sample: Sample, samples: list[Sample]) -> Sample:
    """Return a single eval Sample from AGSRolloutRunner's segment output."""

    sample = copy.deepcopy(base_sample)
    first = samples[0] if samples else None
    metadata = dict(getattr(sample, "metadata", None) or {})
    if first is not None:
        metadata.update(first.metadata or {})
    metadata["eval_collapsed_segments"] = len(samples)

    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0 if first is None or first.reward is None else float(first.reward)
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED if first is None else first.status
    sample.metadata = metadata
    return sample


def transform_group(group, task_type: str = TASK_TYPE):
    return group


def is_valid_group(group, min_valid_group_size: int, task_type: str = TASK_TYPE) -> bool:
    _instance_id, items = group
    valid = 0
    for item in items:
        try:
            samples = samples_from_payload(item)
        except Exception:
            continue
        if samples and all(sample.response_length > 0 and sample.tokens for sample in samples):
            valid += 1
    return len(items) >= min_valid_group_size and valid >= min_valid_group_size


def get_group_data_meta_info(temp_data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rewards = []
    status_counts: dict[str, int] = {}
    artifact_counts = {"trajectory": 0, "patch": 0, "rollout_dump": 0, "complete": 0}
    elapsed_secs = []
    agent_exit_nonzero = 0
    applied_cleanly = 0
    rollout_concurrency = 0
    total_samples = 0
    total_rollouts = sum(len(items) for items in temp_data.values())

    for items in temp_data.values():
        for item in items:
            samples = samples_from_payload(item)
            for sample in samples:
                total_samples += 1
                if sample.reward is not None:
                    rewards.append(float(sample.reward))
                status_counts[sample.status.value] = status_counts.get(sample.status.value, 0) + 1
                metadata = sample.metadata or {}
                has_trajectory = bool(metadata.get("trajectory_path"))
                has_patch = bool(metadata.get("patch_path"))
                has_rollout_dump = bool(metadata.get("rollout_dump_path"))
                artifact_counts["trajectory"] += int(has_trajectory)
                artifact_counts["patch"] += int(has_patch)
                artifact_counts["rollout_dump"] += int(has_rollout_dump)
                artifact_counts["complete"] += int(has_trajectory and has_patch and has_rollout_dump)
                elapsed_sec = _float_or_none(metadata.get("ags_elapsed_sec"))
                if elapsed_sec is not None:
                    elapsed_secs.append(elapsed_sec)
                agent_exit_nonzero += int((metadata.get("agent_exit_code") or 0) != 0)
                applied_cleanly += int(bool(metadata.get("applied_cleanly")))
                rollout_concurrency = max(rollout_concurrency, int(metadata.get("ags_rollout_concurrency") or 0))

    completed = status_counts.get(Sample.Status.COMPLETED.value, 0)
    aborted = status_counts.get(Sample.Status.ABORTED.value, 0)
    solved = sum(1 for reward in rewards if reward == 1.0)
    nonzero = sum(1 for reward in rewards if reward != 0)
    return {
        "total_samples": total_samples,
        "total_rollouts": total_rollouts,
        "num_groups": len(temp_data),
        "avg_group_size": total_rollouts / len(temp_data) if temp_data else 0,
        "avg_samples_per_group": total_samples / len(temp_data) if temp_data else 0,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
        "nonzero_reward_samples": nonzero,
        "solved_samples": solved,
        "solve_rate": solved / len(rewards) if rewards else 0,
        "nonzero_reward_rate": nonzero / len(rewards) if rewards else 0,
        "completed_rate": completed / total_samples if total_samples else 0,
        "abort_rate": aborted / total_samples if total_samples else 0,
        "artifact_complete_rate": artifact_counts["complete"] / total_samples if total_samples else 0,
        "status_counts": status_counts,
        "artifact_counts": artifact_counts,
        "performance": {
            "rollout_concurrency": rollout_concurrency,
            "avg_elapsed_sec": sum(elapsed_secs) / len(elapsed_secs) if elapsed_secs else 0,
            "p50_elapsed_sec": _percentile(elapsed_secs, 0.50),
            "p95_elapsed_sec": _percentile(elapsed_secs, 0.95),
            "max_elapsed_sec": max(elapsed_secs) if elapsed_secs else 0,
            "elapsed_sec_values": elapsed_secs,
            "agent_exit_nonzero_count": agent_exit_nonzero,
            "applied_cleanly_count": applied_cleanly,
        },
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def _build_args(data: dict[str, Any]) -> Namespace:
    sampling_params = dict(data.get("sampling_params") or {})
    if "max_tokens" not in sampling_params and "max_tokens" in data:
        sampling_params["max_tokens"] = int(data["max_tokens"])
    max_tokens = int(sampling_params.get("max_tokens") or data.get("max_tokens") or 4096)
    prompt_data = data["input_file"]
    return Namespace(
        hf_checkpoint=data["tokenizer_path"],
        prompt_data=prompt_data,
        input_key=data.get("input_key", "prompt"),
        label_key=data.get("label_key", "label"),
        metadata_key=data.get("metadata_key", "metadata"),
        tool_key=data.get("tool_key"),
        multimodal_keys=data.get("multimodal_keys"),
        apply_chat_template=_as_bool(data.get("apply_chat_template", False)),
        apply_chat_template_kwargs=data.get("apply_chat_template_kwargs") or {},
        rollout_global_dataset=True,
        rollout_shuffle=_as_bool(data.get("rollout_shuffle", False)),
        rollout_seed=int(data.get("rollout_seed", 42)),
        rollout_max_prompt_len=data.get("rollout_max_prompt_len"),
        dump_details=None,
        rollout_max_context_len=int(data.get("rollout_max_context_len", 0) or 0),
        rollout_batch_size=int(data.get("rollout_batch_size", 1)),
        n_samples_per_prompt=int(data["num_repeat_per_sample"]),
        sglang_router_ip=_router_ip(data["remote_engine_url"]),
        sglang_router_port=_router_port(data["remote_engine_url"]),
        sglang_tool_call_parser=data.get("sglang_tool_call_parser"),
        sglang_reasoning_parser=data.get("sglang_reasoning_parser"),
        use_wandb=_as_bool(data.get("use_wandb", False)),
        wandb_mode=data.get("wandb_mode"),
        wandb_project=data.get("wandb_project"),
        wandb_team=data.get("wandb_team"),
        wandb_run_id=data.get("wandb_run_id"),
        wandb_group=data.get("wandb_group"),
        sampling_params=sampling_params | {"max_tokens": max_tokens},
    )


def _router_ip(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url if "://" in url else f"http://{url}")
    return parsed.hostname or "127.0.0.1"


def _router_port(url: str) -> int:
    from urllib.parse import urlparse

    parsed = urlparse(url if "://" in url else f"http://{url}")
    return int(parsed.port or 80)


def _instance_id(sample: Sample) -> str:
    metadata = sample.metadata or {}
    remote = metadata.get("remote_env_info") or {}
    label = sample.label if isinstance(sample.label, str) and len(sample.label) < 256 else None
    return str(metadata.get("instance_id") or remote.get("instance_id") or label or sample.index or "unknown")


def _send_data_to_buffer(remote_buffer_url: str, data: dict[str, Any]) -> None:
    last_err = None
    for _ in range(3):
        try:
            response = requests.post(remote_buffer_url, json=data, timeout=30)
            if response.status_code == 200:
                return
            last_err = RuntimeError(f"status={response.status_code} body={response.text[:200]}")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"send data to buffer failed: {last_err}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _stop_requested(stop_event: Any) -> bool:
    return bool(stop_event is not None and stop_event.is_set())
