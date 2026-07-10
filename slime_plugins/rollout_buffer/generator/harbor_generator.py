"""Harbor-backed rollout-buffer generator.

This generator keeps slime's rollout-buffer HTTP boundary intact and uses Harbor
as the agent/environment/verifier execution engine.  Completed Harbor trials are
converted into fully-formed ``Sample`` payloads before they are written to the
buffer, so the training-side rollout function does not need Harbor-specific
logic.
"""

from __future__ import annotations

import asyncio
import copy
import fnmatch
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests

from slime.utils.types import Sample
from slime_plugins.rollout_buffer.generator.ags_generator.serialization import (
    output_item_from_samples,
    samples_from_payload,
)

TASK_TYPE = "harbor"

_DEFAULT_HARBOR_REPO = "/data/workspace/harbor"


@dataclass(frozen=True)
class HarborTaskSpec:
    path: Path
    instance_id: str
    source: str | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class HarborRunConfig:
    harbor_repo_path: Path
    remote_buffer_url: str
    trials_dir: Path
    agent_name: str
    model_name: str | None
    environment_type: str
    agent_kwargs: dict[str, Any]
    environment_kwargs: dict[str, Any]
    proxy_kwargs: dict[str, Any]
    verifier_kwargs: dict[str, Any]
    reward_key: str | None
    allow_retokenize_fallback: bool
    tokenizer_path: str | None
    loss_mask_type: str
    concurrency: int
    repeats: int
    num_epoch: int
    skip_instance_ids: set[str]
    task_specs: list[HarborTaskSpec]
    task_overwrite: bool
    task_download_dir: Path | None
    materialized_tasks_dir: Path
    git_url: str | None
    git_commit_id: str | None

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> HarborRunConfig:
        harbor_repo_path = Path(
            data.get("harbor_repo_path") or os.environ.get("HARBOR_REPO_PATH") or _DEFAULT_HARBOR_REPO
        ).expanduser()
        trials_dir = Path(data.get("harbor_trials_dir") or data.get("trials_dir") or "runs/harbor-rollout-buffer")
        if not trials_dir.is_absolute():
            trials_dir = Path.cwd() / trials_dir
        materialized_tasks_dir = Path(
            data.get("harbor_materialized_tasks_dir")
            or data.get("materialized_task_dir")
            or "local/harbor-materialized-tasks"
        ).expanduser()
        if not materialized_tasks_dir.is_absolute():
            materialized_tasks_dir = Path.cwd() / materialized_tasks_dir

        agent_name = data.get("harbor_agent") or os.environ.get("HARBOR_AGENT") or "terminus-2"
        agent_kwargs = _dict_payload(data.get("harbor_agent_kwargs"))
        agent_kwargs.update(_dict_payload(data.get("agent_kwargs")))
        _apply_default_agent_kwargs(agent_name, data.get("remote_engine_url"), agent_kwargs)

        environment_kwargs = _dict_payload(data.get("harbor_environment_kwargs"))
        environment_kwargs.update(_dict_payload(data.get("environment_kwargs")))

        return cls(
            harbor_repo_path=harbor_repo_path,
            remote_buffer_url=str(data["remote_buffer_url"]),
            trials_dir=trials_dir,
            agent_name=str(agent_name),
            model_name=data.get("harbor_model") or data.get("model") or data.get("model_name"),
            environment_type=str(data.get("harbor_env") or os.environ.get("HARBOR_ENV") or "docker"),
            agent_kwargs=agent_kwargs,
            environment_kwargs=environment_kwargs,
            proxy_kwargs=_dict_payload(data.get("harbor_proxy_kwargs")),
            verifier_kwargs=_dict_payload(data.get("harbor_verifier_kwargs")),
            reward_key=data.get("harbor_reward_key") or data.get("reward_key"),
            allow_retokenize_fallback=_as_bool(data.get("allow_retokenize_fallback", False)),
            tokenizer_path=data.get("tokenizer_path") or data.get("hf_checkpoint"),
            loss_mask_type=str(data.get("loss_mask_type") or "qwen"),
            concurrency=max(1, int(data.get("harbor_rollout_concurrency") or data.get("num_process") or 1)),
            repeats=max(1, int(data.get("num_repeat_per_sample") or 1)),
            num_epoch=max(1, int(data.get("num_epoch") or 1)),
            skip_instance_ids={str(x) for x in _list_payload(data.get("skip_instance_ids"))},
            task_specs=_load_task_specs(data),
            task_overwrite=_as_bool(data.get("harbor_task_overwrite", False)),
            task_download_dir=(
                Path(data["harbor_task_download_dir"]).expanduser() if data.get("harbor_task_download_dir") else None
            ),
            materialized_tasks_dir=materialized_tasks_dir,
            git_url=data.get("harbor_git_url"),
            git_commit_id=data.get("harbor_git_commit_id"),
        )


@dataclass(frozen=True)
class TrialWorkItem:
    spec: HarborTaskSpec
    epoch: int
    repeat_index: int
    group_index: int
    sample_index: int

    @property
    def rollout_id(self) -> int:
        return self.sample_index


def run_rollout(data: dict[str, Any]) -> str:
    """Entry point discovered by ``buffer.py`` for ``task_type=harbor``."""

    config = HarborRunConfig.from_payload(data)
    if not config.task_specs:
        raise ValueError(
            "No Harbor tasks found. Pass harbor_task_path, harbor_task_paths, harbor_dataset, "
            "or an input_file with path-based or inline metadata.harbor_task rows."
        )

    print(
        "[harbor_generator] start "
        f"tasks={len(config.task_specs)} repeats={config.repeats} epochs={config.num_epoch} "
        f"concurrency={config.concurrency} agent={config.agent_name} env={config.environment_type}"
    )
    asyncio.run(_run_rollout_async(config))
    return "finished"


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
        if samples and all(_is_train_payload_sample(sample) for sample in samples):
            valid += 1
    return len(items) >= min_valid_group_size and valid >= min_valid_group_size


def get_group_data_meta_info(temp_data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rewards: list[float] = []
    status_counts: dict[str, int] = {}
    exception_counts: dict[str, int] = {}
    artifact_counts = {"result": 0, "trial_dir": 0, "trajectory": 0, "patch": 0, "complete": 0}
    total_samples = 0
    total_rollouts = sum(len(items) for items in temp_data.values())

    for items in temp_data.values():
        for item in items:
            try:
                samples = samples_from_payload(item)
            except Exception:
                continue
            for sample in samples:
                total_samples += 1
                if sample.reward is not None:
                    rewards.append(float(sample.reward))
                status_counts[sample.status.value] = status_counts.get(sample.status.value, 0) + 1
                metadata = sample.metadata or {}
                exc_type = metadata.get("harbor_exception_type")
                if exc_type:
                    exception_counts[str(exc_type)] = exception_counts.get(str(exc_type), 0) + 1
                has_result = bool(metadata.get("harbor_result_path"))
                has_trial = bool(metadata.get("harbor_trial_dir"))
                has_trajectory = bool(metadata.get("trajectory_path"))
                has_patch = bool(metadata.get("patch_path"))
                artifact_counts["result"] += int(has_result)
                artifact_counts["trial_dir"] += int(has_trial)
                artifact_counts["trajectory"] += int(has_trajectory)
                artifact_counts["patch"] += int(has_patch)
                artifact_counts["complete"] += int(has_result and has_trial)

    solved = sum(1 for reward in rewards if reward == 1.0)
    nonzero = sum(1 for reward in rewards if reward != 0.0)
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
        "status_counts": status_counts,
        "exception_counts": exception_counts,
        "artifact_counts": artifact_counts,
    }


async def _run_rollout_async(config: HarborRunConfig) -> None:
    _ensure_harbor_importable(config.harbor_repo_path)
    semaphore = asyncio.Semaphore(config.concurrency)
    tasks: list[asyncio.Task[None]] = []

    sample_index = 0
    for epoch in range(config.num_epoch):
        for group_index, spec in enumerate(config.task_specs):
            if spec.instance_id in config.skip_instance_ids:
                print(f"[harbor_generator] skip instance_id={spec.instance_id}")
                continue
            for repeat_index in range(config.repeats):
                item = TrialWorkItem(
                    spec=spec,
                    epoch=epoch,
                    repeat_index=repeat_index,
                    group_index=group_index,
                    sample_index=sample_index,
                )
                sample_index += 1
                tasks.append(asyncio.create_task(_run_one_with_semaphore(config, item, semaphore)))
    if tasks:
        await asyncio.gather(*tasks)


async def _run_one_with_semaphore(config: HarborRunConfig, item: TrialWorkItem, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        samples = await _run_one_trial(config, item)
        output = output_item_from_samples(
            samples,
            instance_id=item.spec.instance_id,
            extra_info={
                "task_type": TASK_TYPE,
                "harbor_task_path": str(item.spec.path),
                "harbor_epoch": item.epoch,
                "harbor_repeat_index": item.repeat_index,
            },
        )
        _send_data_to_buffer(config.remote_buffer_url, output)


async def _run_one_trial(config: HarborRunConfig, item: TrialWorkItem) -> list[Sample]:
    start = time.time()
    trial_result: Any | None = None
    try:
        trial_result = await _create_harbor_trial(config, item).run()
        result = _to_plain_dict(trial_result)
        samples = samples_from_harbor_result(
            result,
            reward_key=config.reward_key,
            base_sample=_base_sample(config, item),
            allow_retokenize_fallback=config.allow_retokenize_fallback,
            tokenizer_path=config.tokenizer_path,
            loss_mask_type=config.loss_mask_type,
        )
        _attach_runtime_metadata(samples, config, item, result, time.time() - start)
        return samples
    except Exception as exc:  # noqa: BLE001 - rollout workers must report failures as samples.
        print(f"[harbor_generator] trial failed for {item.spec.instance_id}: {exc}\n{traceback.format_exc()}")
        sample = _aborted_sample(
            _base_sample(config, item),
            reward=0.0,
            reason=f"exception:{type(exc).__name__}",
            metadata={
                "harbor_exception_type": type(exc).__name__,
                "harbor_exception_message": str(exc),
                "harbor_task_path": str(item.spec.path),
                "harbor_agent": config.agent_name,
                "harbor_env": config.environment_type,
                "harbor_elapsed_sec": time.time() - start,
            },
        )
        return [sample]


def _create_harbor_trial(config: HarborRunConfig, item: TrialWorkItem):
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        ProxyConfig,
        TaskConfig,
        TrialConfig,
        VerifierConfig,
    )
    from harbor.trial.trial import Trial

    trial_name = _safe_name(
        f"{item.spec.path.name}__harbor__e{item.epoch}__r{item.repeat_index}__i{item.sample_index}"
    )
    environment_type = EnvironmentType(config.environment_type)
    task_config = TaskConfig(
        path=item.spec.path,
        git_url=config.git_url,
        git_commit_id=config.git_commit_id,
        overwrite=config.task_overwrite,
        download_dir=config.task_download_dir,
        source=item.spec.source,
    )
    return Trial(
        TrialConfig(
            task=task_config,
            trial_name=trial_name,
            trials_dir=config.trials_dir,
            agent=AgentConfig(
                name=config.agent_name,
                model_name=config.model_name,
                kwargs=copy.deepcopy(config.agent_kwargs),
            ),
            environment=EnvironmentConfig(
                type=environment_type,
                kwargs=copy.deepcopy(config.environment_kwargs),
            ),
            proxy=ProxyConfig(**copy.deepcopy(config.proxy_kwargs)),
            verifier=VerifierConfig(**copy.deepcopy(config.verifier_kwargs)),
            dataset_name=item.spec.source,
        )
    )


def samples_from_harbor_result(
    result: dict[str, Any],
    *,
    reward_key: str | None = None,
    base_sample: Sample | None = None,
    allow_retokenize_fallback: bool = False,
    tokenizer_path: str | None = None,
    loss_mask_type: str = "qwen",
) -> list[Sample]:
    """Convert a Harbor TrialResult-like dictionary to slime Samples."""

    base = copy.deepcopy(base_sample) if base_sample is not None else Sample()
    reward, selected_reward_key, raw_rewards = _select_reward(result, reward_key)
    common_metadata = _result_metadata(result, selected_reward_key, raw_rewards)

    rollout_details = _get_nested(result, "agent_result", "rollout_details") or []
    samples = _samples_from_rollout_details(rollout_details, base, reward, common_metadata)
    if samples:
        return samples

    if allow_retokenize_fallback:
        sample = _retokenized_fallback_sample(result, base, reward, common_metadata, tokenizer_path, loss_mask_type)
        if sample is not None:
            return [sample]

    return [
        _aborted_sample(
            base,
            reward=0.0,
            reason="missing_token_rollout_details",
            metadata={**common_metadata, "harbor_raw_reward": reward},
        )
    ]


def _samples_from_rollout_details(
    rollout_details: list[dict[str, Any]],
    base: Sample,
    reward: float,
    common_metadata: dict[str, Any],
) -> list[Sample]:
    samples: list[Sample] = []
    rollout_id = base.rollout_id if base.rollout_id is not None else base.index
    next_index = base.index if isinstance(base.index, int) else 0

    for segment_index, detail in enumerate(rollout_details):
        prompt_turns = detail.get("prompt_token_ids") or []
        completion_turns = detail.get("completion_token_ids") or []
        logprob_turns = detail.get("logprobs") or []
        for turn_index, completion_ids in enumerate(completion_turns):
            completion = _int_list(completion_ids)
            if not completion:
                continue
            prompt = _int_list(prompt_turns[turn_index]) if turn_index < len(prompt_turns) else []
            logprobs = _float_list(logprob_turns[turn_index]) if turn_index < len(logprob_turns) else None
            if logprobs is not None and len(logprobs) != len(completion):
                logprobs = None

            sample = copy.deepcopy(base)
            sample.index = next_index
            next_index += 1
            sample.rollout_id = rollout_id
            sample.tokens = prompt + completion
            sample.response_length = len(completion)
            sample.loss_mask = [1] * len(completion)
            sample.rollout_log_probs = logprobs
            sample.reward = reward
            sample.status = Sample.Status.COMPLETED
            sample.remove_sample = False
            sample.metadata = {
                **(sample.metadata or {}),
                **common_metadata,
                "harbor_segment_index": segment_index,
                "harbor_turn_index": turn_index,
                "harbor_prompt_token_count": len(prompt),
                "harbor_completion_token_count": len(completion),
                "harbor_has_rollout_log_probs": logprobs is not None,
            }
            samples.append(sample)
    return samples


def _retokenized_fallback_sample(
    result: dict[str, Any],
    base: Sample,
    reward: float,
    common_metadata: dict[str, Any],
    tokenizer_path: str | None,
    loss_mask_type: str,
) -> Sample | None:
    if not tokenizer_path:
        return None
    messages = _extract_messages_for_retokenize(result)
    if not messages:
        return None

    from slime.utils.mask_utils import MultiTurnLossMaskGenerator
    from slime.utils.processing_utils import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=True)
    mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=loss_mask_type)
    token_ids, loss_mask = mask_generator.get_loss_mask(messages)
    response_length = mask_generator.get_response_lengths([loss_mask])[0]
    if response_length <= 0:
        return None

    sample = copy.deepcopy(base)
    sample.tokens = token_ids
    sample.response_length = response_length
    sample.loss_mask = loss_mask[-response_length:]
    sample.rollout_log_probs = None
    sample.reward = reward
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {**(sample.metadata or {}), **common_metadata, "retokenized": True}
    return sample


def _extract_messages_for_retokenize(result: dict[str, Any]) -> list[dict[str, str]]:
    metadata_messages = _get_nested(result, "agent_result", "metadata", "messages")
    if isinstance(metadata_messages, list):
        return [m for m in metadata_messages if isinstance(m, dict) and "role" in m and "content" in m]

    trajectory_path = _find_trajectory_path(result)
    if trajectory_path and Path(trajectory_path).is_file():
        try:
            data = json.loads(Path(trajectory_path).read_text())
        except Exception:
            return []
        messages = []
        for step in data.get("steps", []):
            if not isinstance(step, dict):
                continue
            source = step.get("source")
            role = "assistant" if source == "agent" else "user" if source == "user" else None
            content = step.get("message")
            if role and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})
        return messages
    return []


def _select_reward(result: dict[str, Any], reward_key: str | None) -> tuple[float, str | None, dict[str, Any] | None]:
    rewards = _get_nested(result, "verifier_result", "rewards")
    if not isinstance(rewards, dict) or not rewards:
        return 0.0, reward_key, rewards if isinstance(rewards, dict) else None

    if reward_key:
        value = rewards.get(reward_key, 0.0)
        return _coerce_float(value), reward_key, rewards

    for key, value in rewards.items():
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value), str(key), rewards
    return 0.0, None, rewards


def _result_metadata(
    result: dict[str, Any], reward_key: str | None, raw_rewards: dict[str, Any] | None
) -> dict[str, Any]:
    trial_dir = _trial_uri_to_path(result.get("trial_uri"))
    metadata = {
        "harbor_task_name": result.get("task_name"),
        "harbor_trial_name": result.get("trial_name"),
        "harbor_trial_dir": str(trial_dir) if trial_dir else None,
        "harbor_result_path": str(trial_dir / "result.json") if trial_dir else None,
        "harbor_agent": _get_nested(result, "agent_info", "name"),
        "harbor_reward_key": reward_key,
        "harbor_rewards": raw_rewards,
    }
    exception_info = result.get("exception_info")
    if isinstance(exception_info, dict):
        metadata.update(
            {
                "harbor_exception_type": exception_info.get("exception_type"),
                "harbor_exception_message": exception_info.get("exception_message"),
            }
        )
    trajectory_path = _find_trajectory_path(result)
    if trajectory_path:
        metadata["trajectory_path"] = trajectory_path
    patch_path = _find_patch_path(result)
    if patch_path:
        metadata["patch_path"] = patch_path
    return {key: value for key, value in metadata.items() if value is not None}


def _find_trajectory_path(result: dict[str, Any]) -> str | None:
    trial_dir = _trial_uri_to_path(result.get("trial_uri"))
    if not trial_dir:
        return None
    for rel in ("agent/trajectory.json", "agent/trajectory.jsonl"):
        path = trial_dir / rel
        if path.exists():
            return str(path)
    return None


def _find_patch_path(result: dict[str, Any]) -> str | None:
    trial_dir = _trial_uri_to_path(result.get("trial_uri"))
    if not trial_dir:
        return None
    path = trial_dir / "code_diff" / "agent.patch"
    return str(path) if path.exists() else None


def _trial_uri_to_path(value: Any) -> Path | None:
    if not value:
        return None
    raw = str(value)
    parsed = urlparse(raw)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        return None
    return Path(raw)


def _aborted_sample(base: Sample, *, reward: float, reason: str, metadata: dict[str, Any]) -> Sample:
    sample = copy.deepcopy(base)
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = reward
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), **metadata, "abort_reason": reason}
    return sample


def _base_sample(config: HarborRunConfig, item: TrialWorkItem) -> Sample:
    return Sample(
        index=item.sample_index,
        group_index=item.group_index,
        rollout_id=item.rollout_id,
        prompt=item.spec.prompt or str(item.spec.path),
        reward=0.0,
        status=Sample.Status.PENDING,
        metadata={
            "instance_id": item.spec.instance_id,
            "harbor_task_path": str(item.spec.path),
            "harbor_epoch": item.epoch,
            "harbor_repeat_index": item.repeat_index,
            "harbor_agent": config.agent_name,
            "harbor_env": config.environment_type,
        },
    )


def _attach_runtime_metadata(
    samples: list[Sample],
    config: HarborRunConfig,
    item: TrialWorkItem,
    result: dict[str, Any],
    elapsed_sec: float,
) -> None:
    for sample in samples:
        sample.metadata = {
            **(sample.metadata or {}),
            "instance_id": item.spec.instance_id,
            "harbor_task_path": str(item.spec.path),
            "harbor_epoch": item.epoch,
            "harbor_repeat_index": item.repeat_index,
            "harbor_agent": config.agent_name,
            "harbor_env": config.environment_type,
            "harbor_elapsed_sec": elapsed_sec,
        }
        if result.get("exception_info") and sample.status == Sample.Status.COMPLETED:
            sample.status = Sample.Status.FAILED


def _is_train_payload_sample(sample: Sample) -> bool:
    if not sample.tokens or sample.response_length <= 0:
        return False
    if sample.reward is None:
        return False
    if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
        return False
    if sample.rollout_log_probs is not None and len(sample.rollout_log_probs) != sample.response_length:
        return False
    return isinstance(sample.status, Sample.Status)


def _load_task_specs(data: dict[str, Any]) -> list[HarborTaskSpec]:
    raw_paths: list[Any] = []
    if data.get("harbor_task_paths"):
        raw_paths.extend(_list_payload(data["harbor_task_paths"]))
    for key in ("harbor_task_path", "task_path"):
        if data.get(key):
            raw_paths.append(data[key])

    specs = [_spec_from_path(path, source=data.get("harbor_dataset")) for path in raw_paths]
    if specs:
        return specs

    dataset = data.get("harbor_dataset")
    if dataset:
        return _specs_from_dataset(Path(dataset), data)

    input_file = data.get("input_file")
    if input_file:
        return _specs_from_input_file(Path(input_file), data)

    return []


def _specs_from_dataset(dataset: Path, data: dict[str, Any]) -> list[HarborTaskSpec]:
    dataset = dataset.expanduser().resolve()
    task_names = [str(x) for x in _list_payload(data.get("harbor_task_names"))]
    exclude_names = [str(x) for x in _list_payload(data.get("harbor_exclude_task_names"))]
    offset = int(data.get("harbor_task_offset") or 0)
    n_tasks = data.get("harbor_n_tasks")
    n_tasks = int(n_tasks) if n_tasks is not None else None

    paths = [p for p in sorted(dataset.iterdir()) if (p / "task.toml").is_file() and (p / "instruction.md").is_file()]
    if task_names:
        paths = [p for p in paths if any(fnmatch.fnmatch(p.name, pat) for pat in task_names)]
    if exclude_names:
        paths = [p for p in paths if not any(fnmatch.fnmatch(p.name, pat) for pat in exclude_names)]
    if offset:
        paths = paths[offset:]
    if n_tasks is not None:
        paths = paths[:n_tasks]
    return [_spec_from_path(path, source=dataset.name) for path in paths]


def _specs_from_input_file(input_file: Path, data: dict[str, Any]) -> list[HarborTaskSpec]:
    specs: list[HarborTaskSpec] = []
    if not input_file.exists():
        return specs
    for i, line in enumerate(input_file.read_text().splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        metadata_key = data.get("metadata_key") or "metadata"
        metadata = item.get(metadata_key) if isinstance(item.get(metadata_key), dict) else {}
        prompt = item.get(data.get("input_key") or "input") or item.get("prompt")
        inline_task = item.get("harbor_task") or metadata.get("harbor_task")
        if inline_task:
            specs.append(_spec_from_inline_task(inline_task, item, metadata, data, i, prompt))
            continue

        path = (
            item.get("harbor_task_path")
            or item.get("task_path")
            or item.get("path")
            or metadata.get("harbor_task_path")
            or metadata.get("task_path")
        )
        if not path:
            continue
        source = item.get("source") or metadata.get("source") or data.get("harbor_dataset")
        instance_id = item.get("instance_id") or metadata.get("instance_id") or Path(path).name or str(i)
        specs.append(
            HarborTaskSpec(
                path=Path(path).expanduser().resolve(),
                instance_id=str(instance_id),
                source=source,
                prompt=str(prompt) if prompt is not None else None,
            )
        )
    return specs


def _spec_from_inline_task(
    inline_task: Any,
    item: dict[str, Any],
    metadata: dict[str, Any],
    data: dict[str, Any],
    line_index: int,
    prompt: Any,
) -> HarborTaskSpec:
    if not isinstance(inline_task, dict):
        raise TypeError(f"Inline Harbor task at line {line_index + 1} must be a dict.")
    task_format = inline_task.get("format")
    if task_format not in (None, "harbor_task_inline_v1"):
        raise ValueError(f"Unsupported inline Harbor task format at line {line_index + 1}: {task_format!r}")

    instance_id = (
        item.get("instance_id")
        or metadata.get("instance_id")
        or inline_task.get("instance_id")
        or inline_task.get("name")
        or item.get("label")
        or f"task-{line_index}"
    )
    task_name = str(inline_task.get("name") or instance_id)
    materialized_root = Path(
        data.get("harbor_materialized_tasks_dir")
        or data.get("materialized_task_dir")
        or "local/harbor-materialized-tasks"
    ).expanduser()
    if not materialized_root.is_absolute():
        materialized_root = Path.cwd() / materialized_root
    task_dir = (materialized_root / _safe_name(task_name)).resolve()
    if _as_bool(data.get("harbor_materialized_task_overwrite", True)) and task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    files = _inline_task_files(inline_task, line_index)
    _write_inline_task_files(task_dir, files, line_index)
    _ensure_required_inline_task_files(task_dir, line_index)

    source = item.get("source") or metadata.get("source") or inline_task.get("source") or data.get("harbor_dataset")
    return HarborTaskSpec(
        path=task_dir,
        instance_id=str(instance_id),
        source=str(source) if source is not None else None,
        prompt=str(prompt) if prompt is not None else None,
    )


def _inline_task_files(inline_task: dict[str, Any], line_index: int) -> list[tuple[str, str, int | None]]:
    raw_files = inline_task.get("files")
    if isinstance(raw_files, dict):
        files: list[tuple[str, str, int | None]] = []
        for rel_path, value in raw_files.items():
            text: Any
            mode: int | None = None
            if isinstance(value, dict):
                text = value.get("text", value.get("content"))
                if value.get("mode") is not None:
                    mode = int(str(value["mode"]), 8)
            else:
                text = value
            if not isinstance(text, str):
                raise TypeError(f"Inline Harbor task file {rel_path!r} at line {line_index + 1} must contain text.")
            files.append((str(rel_path), text, mode))
        return files

    if isinstance(raw_files, list):
        files = []
        for file_index, entry in enumerate(raw_files):
            if not isinstance(entry, dict):
                raise TypeError(f"Inline Harbor task file entry {file_index} at line {line_index + 1} must be a dict.")
            rel_path = entry.get("path")
            text = entry.get("text", entry.get("content"))
            if not isinstance(rel_path, str) or not isinstance(text, str):
                raise TypeError(
                    f"Inline Harbor task file entry {file_index} at line {line_index + 1} "
                    "must contain string path and text."
                )
            mode = int(str(entry["mode"]), 8) if entry.get("mode") is not None else None
            files.append((rel_path, text, mode))
        return files

    raise ValueError(f"Inline Harbor task at line {line_index + 1} must contain a files dict or list.")


def _write_inline_task_files(task_dir: Path, files: list[tuple[str, str, int | None]], line_index: int) -> None:
    root = task_dir.resolve()
    for rel_path, text, mode in files:
        target = _safe_inline_file_target(root, rel_path, line_index)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        if mode is not None:
            target.chmod(mode)
        elif target.suffix == ".sh":
            target.chmod(0o755)


def _safe_inline_file_target(root: Path, rel_path: str, line_index: int) -> Path:
    path = Path(rel_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe inline Harbor task file path at line {line_index + 1}: {rel_path!r}")
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Unsafe inline Harbor task file path at line {line_index + 1}: {rel_path!r}")
    return target


def _ensure_required_inline_task_files(task_dir: Path, line_index: int) -> None:
    missing = [name for name in ("instruction.md", "task.toml") if not (task_dir / name).is_file()]
    if missing:
        raise ValueError(f"Inline Harbor task at line {line_index + 1} is missing required files: {missing}")


def _spec_from_path(path: Any, source: str | None = None) -> HarborTaskSpec:
    resolved = Path(str(path)).expanduser().resolve()
    return HarborTaskSpec(path=resolved, instance_id=resolved.name, source=source)


def _ensure_harbor_importable(harbor_repo_path: Path) -> None:
    candidates = [harbor_repo_path / "src", harbor_repo_path]
    for candidate in candidates:
        if candidate.exists():
            path_str = str(candidate.resolve())
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
    try:
        import harbor  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"Cannot import Harbor from {harbor_repo_path}") from exc


def _apply_default_agent_kwargs(agent_name: str, remote_engine_url: str | None, agent_kwargs: dict[str, Any]) -> None:
    if agent_name == "terminus-2":
        agent_kwargs.setdefault("collect_rollout_details", True)
    if not remote_engine_url:
        return
    openai_base = _openai_base_url(str(remote_engine_url))
    if agent_name == "cbc-agent":
        agent_kwargs.setdefault("CBC_BASE_URL", openai_base.rstrip("/") + "/chat/completions")
        agent_kwargs.setdefault("CBC_API_KEY", "EMPTY")
    else:
        agent_kwargs.setdefault("OPENAI_BASE_URL", openai_base)
        agent_kwargs.setdefault("OPENAI_API_BASE", openai_base)
        agent_kwargs.setdefault("OPENAI_API_KEY", "EMPTY")


def _openai_base_url(url: str) -> str:
    url = url.rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


def _send_data_to_buffer(remote_buffer_url: str, data: dict[str, Any]) -> None:
    url = remote_buffer_url.rstrip("/")
    if not url.endswith("/buffer/write"):
        url = f"{url}/buffer/write"
    last_err: BaseException | None = None
    for _ in range(3):
        try:
            response = requests.post(url, json=data, timeout=30)
            if response.status_code == 200:
                return
            last_err = RuntimeError(f"status={response.status_code} body={response.text[:200]}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(1)
    raise RuntimeError(f"send data to buffer failed: {last_err}")


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected dict or pydantic model, got {type(value)!r}")


def _get_nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _dict_payload(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError(f"Expected dict-like payload, got {type(value)!r}")


def _list_payload(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list | tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return [value]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _int_list(values: Any) -> list[int]:
    if values is None:
        return []
    return [int(v) for v in values]


def _float_list(values: Any) -> list[float]:
    if values is None:
        return []
    return [float(v) for v in values]


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:120]
