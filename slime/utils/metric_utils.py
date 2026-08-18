import logging
import math
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    if len(flat_rewards) != num_groups * group_size:
        logger.warning(
            "skip fixed-shape passrate: len(flat_rewards)=%d num_groups=%d group_size=%d",
            len(flat_rewards),
            num_groups,
            group_size,
        )
        return {}
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def compute_grouped_pass_rate(
    flat_rewards: list[float],
    group_indices: list[int | str],
    rollout_ids: list[int | str],
    group_size: int,
):
    """Compute pass@k on prompt groups while tolerating fan-out samples.

    Agentic rollouts can emit multiple train samples for one rollout attempt
    (for example one sample per root-to-leaf chain in a tool-use tree).  Those
    fan-out siblings are training segments, not independent pass@k samples.  So
    this metric first deduplicates by ``(group_index, rollout_id)`` and then
    computes pass@k from the attempt-level rewards in each prompt group.
    """

    if group_size == 1:
        return {}

    if not (len(flat_rewards) == len(group_indices) == len(rollout_ids)):
        logger.warning(
            "skip grouped passrate: rewards=%d group_indices=%d rollout_ids=%d",
            len(flat_rewards),
            len(group_indices),
            len(rollout_ids),
        )
        return {}

    grouped_attempt_rewards: dict[int | str, dict[int | str, float]] = {}
    for position, (reward, group_index, rollout_id) in enumerate(
        zip(flat_rewards, group_indices, rollout_ids, strict=True)
    ):
        # Missing rollout ids are not expected on the normal train path because
        # rollout.py fills them in before packaging. Keep this fallback local so
        # a malformed custom path does not merge unrelated attempts.
        attempt_key = rollout_id if rollout_id is not None else f"position:{position}"
        attempts = grouped_attempt_rewards.setdefault(group_index, {})
        reward = float(reward)
        attempts[attempt_key] = max(attempts.get(attempt_key, reward), reward)

    if not grouped_attempt_rewards:
        return {}

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]
    log_dict = {}
    for k in pass_rate_name_list:
        num_samples = []
        num_correct = []
        for attempts in grouped_attempt_rewards.values():
            rewards = list(attempts.values())
            if len(rewards) < k:
                continue
            num_samples.append(len(rewards))
            num_correct.append(sum(1 for reward in rewards if reward == 1))
        if not num_samples:
            continue

        pass_k_estimates = _estimate_pass_at_k(np.array(num_samples), np.array(num_correct), k)
        log_dict[f"pass@{k}"] = np.mean(pass_k_estimates).item()

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
