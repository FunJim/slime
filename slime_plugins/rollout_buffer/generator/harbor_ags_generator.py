"""Harbor-derived AGS rollout-buffer generator.

``harbor_ags`` consumes the semantic prompt-data rows produced by
``tools/harbor_task_to_slime_prompt_data.py``.  It intentionally does not run
Harbor Trial/Job objects at rollout time; the Harbor task has already been
lowered to the minimal AGS contract:

  - ``metadata.image``
  - ``metadata.workdir``
  - ``metadata.problem_statement``
  - ``metadata.pre_commands`` (optional)
  - ``metadata.eval_cmd`` / ``metadata.swepro`` / ``remote_env_info.f2p_script``

The actual agent execution, sandbox lifecycle, trajectory extraction, artifact
dumping, and reward computation are delegated to ``ags_generator`` so
``task_type=harbor_ags`` behaves like AGS while keeping a separate task type
for Harbor-derived datasets and future Harbor-specific validation/metrics.
"""

from __future__ import annotations

from typing import Any

from slime.utils.types import Sample
from slime_plugins.rollout_buffer.generator.ags_generator.entry import (
    get_group_data_meta_info as _ags_get_group_data_meta_info,
)
from slime_plugins.rollout_buffer.generator.ags_generator.entry import run_rollout_for_task_type
from slime_plugins.rollout_buffer.generator.ags_generator.entry import transform_group as _ags_transform_group
from slime_plugins.rollout_buffer.generator.ags_generator.serialization import samples_from_payload
from slime_plugins.rollout_buffer.generator.ags_generator.source import AGSPromptSource

TASK_TYPE = "harbor_ags"
HARBOR_AGS_INPUT_FORMAT = "harbor_semantic_ags_v1"

_REQUIRED_METADATA_KEYS = ("instance_id", "image", "workdir", "problem_statement")


class HarborAGSPromptSource(AGSPromptSource):
    """Prompt source for Harbor task rows lowered to the AGS metadata contract."""

    def get_groups(self, num_groups: int) -> list[list[Sample]]:
        groups = super().get_groups(num_groups)
        for group in groups:
            for sample in group:
                normalize_harbor_ags_sample(sample)
            if group:
                _ensure_prompt_group_rollout_ids(group)
        return groups


def run_rollout(data: dict[str, Any]) -> str:
    """Run Harbor-derived coding tasks through the shared AGS rollout path."""

    payload = dict(data)
    payload.setdefault("input_key", "prompt")
    payload.setdefault("label_key", "label")
    payload.setdefault("metadata_key", "metadata")
    return run_rollout_for_task_type(payload, task_type=TASK_TYPE, source_cls=HarborAGSPromptSource)


def transform_group(group, task_type: str = TASK_TYPE):
    return _ags_transform_group(group, task_type)


def is_valid_group(group, min_valid_group_size: int, task_type: str = TASK_TYPE) -> bool:
    """Accept groups only after enough AGS-compatible ``Sample`` payloads arrived."""

    _instance_id, items = group
    valid = 0
    for item in items:
        try:
            samples = samples_from_payload(item)
        except Exception:
            continue
        if samples and all(_is_complete_training_sample(sample) for sample in samples):
            valid += 1
    return len(items) >= min_valid_group_size and valid >= min_valid_group_size


def get_group_data_meta_info(temp_data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    meta = _ags_get_group_data_meta_info(temp_data)
    harbor_samples = 0
    missing_required = 0
    harbor_sources: dict[str, int] = {}

    for items in temp_data.values():
        for item in items:
            for sample in samples_from_payload(item):
                metadata = sample.metadata or {}
                harbor_meta = metadata.get("harbor") if isinstance(metadata.get("harbor"), dict) else {}
                harbor_ags_meta = metadata.get("harbor_ags") if isinstance(metadata.get("harbor_ags"), dict) else {}
                if harbor_meta or harbor_ags_meta.get("task_type") == TASK_TYPE:
                    harbor_samples += 1
                if harbor_ags_meta.get("missing_required_metadata"):
                    missing_required += 1
                source = metadata.get("source") or harbor_meta.get("source")
                if source:
                    harbor_sources[str(source)] = harbor_sources.get(str(source), 0) + 1

    meta["harbor_ags"] = {
        "total_samples": harbor_samples,
        "missing_required_metadata_samples": missing_required,
        "num_sources": len(harbor_sources),
        "source_counts": harbor_sources,
    }
    return meta


def normalize_harbor_ags_sample(sample: Sample) -> Sample:
    """Normalize a Harbor-derived prompt row to the AGS metadata contract.

    The converter writes the canonical fields at ``metadata.*`` and keeps
    Harbor provenance under ``metadata.harbor``.  This function also accepts
    the provenance values as fallbacks so future converter revisions can keep
    prompt rows compact without changing the AGS runner.
    """

    if sample.metadata is None:
        metadata: dict[str, Any] = {}
    elif isinstance(sample.metadata, dict):
        metadata = dict(sample.metadata)
    else:
        raise TypeError(f"harbor_ags sample metadata must be a dict, got {type(sample.metadata).__name__}")

    harbor = metadata.get("harbor") if isinstance(metadata.get("harbor"), dict) else {}
    remote = metadata.get("remote_env_info") if isinstance(metadata.get("remote_env_info"), dict) else {}

    instance_id = (
        metadata.get("instance_id")
        or harbor.get("task_name")
        or remote.get("instance_id")
        or (sample.label if isinstance(sample.label, str) and len(sample.label) < 256 else None)
        or sample.index
        or "unknown"
    )
    metadata["instance_id"] = str(instance_id)

    image = metadata.get("image") or harbor.get("docker_image") or remote.get("image_url")
    if image:
        metadata["image"] = str(image)

    workdir = metadata.get("workdir") or harbor.get("docker_workdir") or remote.get("workdir")
    if workdir:
        metadata["workdir"] = str(workdir)

    problem_statement = metadata.get("problem_statement") or _prompt_to_text(sample.prompt)
    if problem_statement:
        metadata["problem_statement"] = str(problem_statement)

    source = metadata.get("source") or harbor.get("source")
    if source:
        metadata["source"] = str(source)

    missing = [key for key in _REQUIRED_METADATA_KEYS if not metadata.get(key)]
    harbor_ags = {
        **(metadata.get("harbor_ags") if isinstance(metadata.get("harbor_ags"), dict) else {}),
        "input_format": HARBOR_AGS_INPUT_FORMAT,
        "task_type": TASK_TYPE,
        "uses_ags_rollout": True,
    }
    if harbor.get("task_name"):
        harbor_ags["task_name"] = harbor["task_name"]
    if missing:
        harbor_ags["missing_required_metadata"] = missing
    else:
        harbor_ags.pop("missing_required_metadata", None)
    metadata["harbor_ags"] = harbor_ags

    sample.metadata = metadata
    return sample


def _is_complete_training_sample(sample: Sample) -> bool:
    if not sample.tokens or sample.response_length <= 0:
        return False
    if sample.reward is None or sample.status is None:
        return False
    if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
        return False
    if sample.rollout_log_probs is not None and len(sample.rollout_log_probs) != sample.response_length:
        return False
    return True


def _task_identity(sample: Sample) -> str:
    metadata = sample.metadata or {}
    harbor = metadata.get("harbor") if isinstance(metadata.get("harbor"), dict) else {}
    return str(
        metadata.get("instance_id")
        or harbor.get("task_name")
        or (sample.label if isinstance(sample.label, str) and len(sample.label) < 256 else None)
        or sample.index
        or "unknown"
    )


def _ensure_prompt_group_rollout_ids(group: list[Sample]) -> None:
    """Keep AGS-repeated attempts from one prompt grouped for training stats.

    ``RolloutDataSource`` gives repeated samples different ``index`` values.
    If ``rollout_id`` stays equal to each sample index, slime's rollout-aware
    train splitter treats repeated attempts from the same prompt as unrelated
    rollouts.  Harbor-derived SWE rows are semantically one task with
    ``n_samples_per_prompt`` attempts, so we use the prompt group's stable
    ``group_index`` when available and fall back to the instance id.
    """

    first = group[0]
    rollout_id: int | str = first.group_index if first.group_index is not None else _task_identity(first)
    for sample in group:
        sample.rollout_id = rollout_id


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return ""
    for message in prompt:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(item.get("text"))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text") is not None
            ]
            if parts:
                return "\n".join(parts)
    return ""
