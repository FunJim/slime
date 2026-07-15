"""Minimal AGS-specific rollout/eval logging hooks for slime.

Wire these through::

    --custom-rollout-log-function-path \
      slime_plugins.rollout_buffer.generator.ags_generator.wandb_metrics.log_rollout_data
    --custom-eval-rollout-log-function-path \
      slime_plugins.rollout_buffer.generator.ags_generator.wandb_metrics.log_eval_rollout_data

The hooks intentionally keep slime's default rollout/eval metrics and add a
small AGS block derived from ``Sample.metadata`` produced by
``ags_generator.rollout.AGSRolloutRunner``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_BUCKET_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def log_rollout_data(rollout_id: int, args: Any, samples: list[Sample], rollout_extra_metrics, rollout_time) -> bool:
    """Log default rollout metrics plus a compact AGS diagnostics block.

    Return ``True`` after logging so slime's default logger is not called a
    second time.  If anything goes wrong, return ``False`` and let the default
    logger handle the batch.
    """

    try:
        if getattr(args, "load_debug_rollout_data", None):
            return False

        log_dict = {**(rollout_extra_metrics or {})}
        log_dict |= dict_add_prefix(_compute_default_metrics_from_samples(args, samples), "rollout/")
        log_dict |= dict_add_prefix(_compute_default_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
        log_dict |= dict_add_prefix(_compute_ags_metrics(args, samples), "rollout/ags/")

        step = compute_rollout_step(args, rollout_id)
        log_dict["rollout/step"] = step
        logger.info("ags rollout %s: %s", rollout_id, log_dict)
        logging_utils.log(args, log_dict, step_key="rollout/step")
        return True
    except Exception:  # noqa: BLE001 - logging must never break rollout/training
        logger.exception("AGS custom rollout logger failed; falling back to slime default logger")
        return False


def log_eval_rollout_data(rollout_id: int, args: Any, data: dict[str, dict[str, Any]], extra_metrics=None) -> bool:
    """Log default eval metrics plus per-dataset and pooled AGS diagnostics."""

    try:
        log_dict = {**(extra_metrics or {})}
        all_rewards: list[float] = []
        all_samples: list[Sample] = []

        for dataset_name, dataset_data in data.items():
            rewards = [_safe_float(r) for r in dataset_data.get("rewards", [])]
            all_rewards.extend(rewards)
            if rewards:
                log_dict[f"eval/{dataset_name}"] = sum(rewards) / len(rewards)

            samples = dataset_data.get("samples")
            if samples is not None:
                samples = list(samples)
                all_samples.extend(samples)
                log_dict |= dict_add_prefix(
                    _compute_default_metrics_from_samples(args, samples),
                    f"eval/{dataset_name}/",
                )
                log_dict |= dict_add_prefix(_compute_ags_metrics(args, samples), f"eval/{dataset_name}/ags/")

            if "truncated" in dataset_data:
                truncated = dataset_data["truncated"]
                if truncated:
                    log_dict[f"eval/{dataset_name}-truncated_ratio"] = sum(bool(x) for x in truncated) / len(truncated)

            if getattr(args, "log_passrate", False) and rewards:
                log_dict |= dict_add_prefix(
                    compute_pass_rate(
                        flat_rewards=rewards,
                        group_size=getattr(args, "n_samples_per_eval_prompt", 1),
                    ),
                    f"eval/{dataset_name}-",
                )

        if all_rewards:
            log_dict["eval/overall"] = sum(all_rewards) / len(all_rewards)
            log_dict |= dict_add_prefix(_stats(all_rewards), "eval/overall/reward/")
        if all_samples:
            log_dict |= dict_add_prefix(_compute_ags_metrics(args, all_samples), "eval/overall/ags/")

        step = compute_rollout_step(args, rollout_id)
        log_dict["eval/step"] = step
        logger.info("ags eval %s: %s", rollout_id, log_dict)
        logging_utils.log(args, log_dict, step_key="eval/step")
        return True
    except Exception:  # noqa: BLE001 - logging must never break eval
        logger.exception("AGS custom eval logger failed; falling back to slime default logger")
        return False


def _compute_default_metrics_from_samples(args: Any, samples: list[Sample]) -> dict[str, Any]:
    # Lazy import keeps this hook importable in lightweight environments where
    # rollout-only optional dependencies (for example sglang) are not installed.
    from slime.ray.rollout import compute_metrics_from_samples

    return compute_metrics_from_samples(args, samples)


def _compute_default_perf_metrics_from_samples(
    args: Any,
    samples: list[Sample],
    rollout_time: float,
) -> dict[str, Any]:
    # See _compute_default_metrics_from_samples for why this is lazy.
    from slime.ray.rollout import compute_perf_metrics_from_samples

    return compute_perf_metrics_from_samples(args, samples, rollout_time)


def _compute_ags_metrics(args: Any, samples: Iterable[Sample]) -> dict[str, float | int]:
    samples = list(samples)
    if not samples:
        return {}

    n = len(samples)
    rewards = [_reward_value(args, sample) for sample in samples]
    statuses = Counter(_status_value(sample) for sample in samples)
    metadata = [_metadata(sample) for sample in samples]

    valid_response_count = sum(1 for sample in samples if getattr(sample, "response_length", 0) > 0 and sample.tokens)
    remove_sample_count = sum(1 for sample in samples if bool(getattr(sample, "remove_sample", False)))
    nonzero_reward_count = sum(1 for reward in rewards if reward != 0.0)
    solve_count = sum(1 for sample, reward in zip(samples, rewards, strict=True) if _is_solved(sample, reward))

    metrics: dict[str, float | int] = {
        "samples/count": n,
        "reward/nonzero_count": nonzero_reward_count,
        "reward/nonzero_rate": _ratio(nonzero_reward_count, n),
        "solve/count": solve_count,
        "remove_sample/count": remove_sample_count,
        "remove_sample/rate": _ratio(remove_sample_count, n),
        "response/valid_count": valid_response_count,
        "response/valid_rate": _ratio(valid_response_count, n),
    }
    metrics |= dict_add_prefix(_stats(rewards), "reward/")
    metrics["solve/rate"] = _ratio(metrics["solve/count"], n)

    for status in sorted(statuses):
        count = statuses[status]
        metrics[f"status/{_bucket(status)}/count"] = count
        metrics[f"status/{_bucket(status)}/rate"] = _ratio(count, n)
    for status in Sample.Status:
        metrics.setdefault(f"status/{status.value}/count", 0)
        metrics.setdefault(f"status/{status.value}/rate", 0.0)

    metrics |= _artifact_metrics(metadata, n)
    metrics |= _runtime_metrics(metadata, n)
    metrics |= _abort_reason_metrics(metadata, n)
    metrics |= _rollout_level_metrics(samples, rewards)
    metrics |= _agent_metrics(samples, metadata, rewards)
    return metrics


def _artifact_metrics(metadata: list[dict[str, Any]], n: int) -> dict[str, float | int]:
    has_trajectory = [bool(md.get("trajectory_path")) for md in metadata]
    has_patch = [bool(md.get("patch_path")) for md in metadata]
    has_rollout_dump = [bool(md.get("rollout_dump_path")) for md in metadata]
    complete = [t and p and d for t, p, d in zip(has_trajectory, has_patch, has_rollout_dump, strict=True)]

    metrics: dict[str, float | int] = {}
    for name, values in {
        "trajectory": has_trajectory,
        "patch": has_patch,
        "rollout_dump": has_rollout_dump,
        "complete": complete,
    }.items():
        count = sum(values)
        metrics[f"artifact/{name}/count"] = count
        metrics[f"artifact/{name}/rate"] = _ratio(count, n)
    return metrics


def _runtime_metrics(metadata: list[dict[str, Any]], n: int) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}

    elapsed = [_safe_float(md.get("ags_elapsed_sec"), default=None) for md in metadata]
    elapsed = [value for value in elapsed if value is not None]
    if elapsed:
        metrics |= dict_add_prefix(_stats(elapsed), "elapsed_sec/")
        metrics["elapsed_sec/p50"] = _percentile(elapsed, 50)
        metrics["elapsed_sec/p95"] = _percentile(elapsed, 95)
        metrics["elapsed_sec/count"] = len(elapsed)

    segments = [_safe_float(md.get("ags_num_samples"), default=None) for md in metadata]
    segments = [value for value in segments if value is not None]
    if segments:
        metrics |= dict_add_prefix(_stats(segments), "segments/")

    concurrencies = [_safe_float(md.get("ags_rollout_concurrency"), default=None) for md in metadata]
    concurrencies = [value for value in concurrencies if value is not None]
    if concurrencies:
        metrics["rollout_concurrency/max"] = max(concurrencies)

    agent_exit_codes = [_safe_float(md.get("agent_exit_code"), default=None) for md in metadata]
    agent_exit_codes = [value for value in agent_exit_codes if value is not None]
    if agent_exit_codes:
        nonzero = sum(1 for code in agent_exit_codes if int(code) != 0)
        metrics["agent_exit/nonzero_count"] = nonzero
        metrics["agent_exit/nonzero_rate"] = _ratio(nonzero, len(agent_exit_codes))

    applied = [md.get("applied_cleanly") for md in metadata if "applied_cleanly" in md]
    if applied:
        count = sum(1 for value in applied if _safe_bool(value))
        metrics["patch/applied_cleanly_count"] = count
        metrics["patch/applied_cleanly_rate"] = _ratio(count, len(applied))
    else:
        metrics["patch/applied_cleanly_count"] = 0
        metrics["patch/applied_cleanly_rate"] = 0.0

    # ``n`` is kept as an explicit denominator for dashboards where some fields
    # are absent on aborted samples.
    metrics["metadata/coverage_rate"] = _ratio(sum(1 for md in metadata if md), n)
    return metrics


def _abort_reason_metrics(metadata: list[dict[str, Any]], n: int) -> dict[str, float | int]:
    reasons = Counter(str(md.get("abort_reason") or "unknown") for md in metadata if md.get("abort_reason"))
    metrics: dict[str, float | int] = {}
    for reason, count in sorted(reasons.items()):
        key = _bucket(reason)
        metrics[f"abort_reason/{key}/count"] = count
        metrics[f"abort_reason/{key}/rate"] = _ratio(count, n)
    metrics["abort_reason/total_count"] = sum(reasons.values())
    metrics["abort_reason/total_rate"] = _ratio(sum(reasons.values()), n)
    return metrics


def _rollout_level_metrics(samples: list[Sample], rewards: list[float]) -> dict[str, float | int]:
    by_rollout: dict[str, list[tuple[Sample, float]]] = defaultdict(list)
    for position, (sample, reward) in enumerate(zip(samples, rewards, strict=True)):
        by_rollout[_rollout_key(sample, position)].append((sample, reward))

    rollout_rewards = [max(reward for _, reward in items) for items in by_rollout.values()]
    rollout_solved = [any(_is_solved(sample, reward) for sample, reward in items) for items in by_rollout.values()]
    metrics: dict[str, float | int] = {
        "rollout/count": len(by_rollout),
        "rollout/solve_count": sum(rollout_solved),
        "rollout/solve_rate": _ratio(sum(rollout_solved), len(by_rollout)),
    }
    metrics |= dict_add_prefix(_stats(rollout_rewards), "rollout/reward/")
    return metrics


def _agent_metrics(
    samples: list[Sample],
    metadata: list[dict[str, Any]],
    rewards: list[float],
) -> dict[str, float | int]:
    by_agent: dict[str, list[tuple[Sample, float]]] = defaultdict(list)
    for sample, md, reward in zip(samples, metadata, rewards, strict=True):
        by_agent[str(md.get("agent") or "unknown")].append((sample, reward))

    metrics: dict[str, float | int] = {"agent/count": len(by_agent)}
    for agent, items in sorted(by_agent.items()):
        key = _bucket(agent)
        count = len(items)
        agent_rewards = [reward for _, reward in items]
        metrics[f"agent/{key}/count"] = count
        metrics[f"agent/{key}/rate"] = _ratio(count, len(samples))
        metrics[f"agent/{key}/solve_rate"] = _ratio(
            sum(1 for sample, reward in items if _is_solved(sample, reward)), count
        )
        metrics[f"agent/{key}/reward_mean"] = sum(agent_rewards) / count if count else 0.0
    return metrics


def _reward_value(args: Any, sample: Sample) -> float:
    reward = getattr(sample, "reward", 0.0)
    if isinstance(reward, dict):
        reward_key = getattr(args, "eval_reward_key", None) or getattr(args, "reward_key", None)
        if reward_key and reward_key in reward:
            return _safe_float(reward[reward_key]) or 0.0
        for key in ("score", "reward", "acc"):
            if key in reward:
                return _safe_float(reward[key]) or 0.0
        for value in reward.values():
            coerced = _safe_float(value, default=None)
            if coerced is not None:
                return coerced
        return 0.0
    return _safe_float(reward) or 0.0


def _is_solved(sample: Sample, reward: float) -> bool:
    md = _metadata(sample)
    if "grading_solved" in md:
        return _safe_bool(md["grading_solved"])
    return reward == 1.0


def _metadata(sample: Sample) -> dict[str, Any]:
    md = getattr(sample, "metadata", None)
    return md if isinstance(md, dict) else {}


def _status_value(sample: Sample) -> str:
    status = getattr(sample, "status", None)
    return status.value if isinstance(status, Sample.Status) else str(status or "unknown")


def _rollout_key(sample: Sample, position: int) -> str:
    rollout_id = getattr(sample, "rollout_id", None)
    if rollout_id is not None:
        return f"rollout:{rollout_id}"
    session_id = getattr(sample, "session_id", None)
    if session_id is not None:
        return f"session:{session_id}"
    return f"position:{position}"


def _stats(values: list[float]) -> dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {}
    return compute_statistics(values)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _ratio(count: float | int, total: float | int) -> float:
    return float(count) / float(total) if total else 0.0


def _bucket(value: str) -> str:
    value = value.strip() or "unknown"
    return _BUCKET_RE.sub("_", value).strip("_") or "unknown"
