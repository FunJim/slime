"""Reward normalization for group-relative advantage estimators.

Grouping matches how the loss aggregates. An agentic rollout can split one
attempt into several training samples (sub-agent dispatch, auto-compaction,
token-drift forks), and those siblings share ``rollout_id``. The loss already
collapses them into one per-attempt mean (``rollout_mask_sums`` ->
``cp_utils.get_sum_of_sample_mean``, which gives each attempt total weight 1.0
however many segments it produced), ``dp_schedule`` sizes a training step in
rollouts rather than samples, and ``compute_grouped_pass_rate`` deduplicates by
``(group_index, rollout_id)``. So the baseline has to be per-attempt too --
otherwise a heavily-forked attempt votes several times in its own group's mean.
"""

from collections import defaultdict

import torch


def normalize_rewards_by_group(
    rewards: list[float],
    group_indices: list[int | None],
    rollout_ids: list[int | str | None] | None = None,
    *,
    normalize_std: bool,
    fallback_group_size: int | None = None,
) -> list[float]:
    """Normalize rewards within the sample group that produced each response.

    Grouping is two-level: ``group_indices`` selects the prompt group, then
    ``rollout_ids`` reduces that group to one reward per attempt, so mean and std
    are attempt-level and match the loss reducer. Each attempt's normalized
    reward is broadcast back to all of its samples.

    ``rollout_ids=None``, or a ``None`` entry, treats the sample as its own
    attempt -- correct for the default path, where one execution is one training
    sample and ``Sample.rollout_id`` is left unset. That case reduces to plain
    per-group normalization, bit for bit.

    ``fallback_group_size`` preserves fixed-size custom rollouts that predate
    ``Sample.group_index``. Uneven groups must provide explicit identities.
    """
    if len(rewards) != len(group_indices):
        raise ValueError(
            f"rewards and group_indices must have the same length, got {len(rewards)} and {len(group_indices)}"
        )
    if rollout_ids is not None and len(rollout_ids) != len(rewards):
        raise ValueError(
            f"rewards and rollout_ids must have the same length, got {len(rewards)} and {len(rollout_ids)}"
        )

    if group_indices and all(group_index is None for group_index in group_indices):
        if fallback_group_size is None or fallback_group_size <= 0 or len(group_indices) % fallback_group_size != 0:
            raise ValueError("group_index is required when reward groups are not uniformly sized")
        group_indices = [position // fallback_group_size for position in range(len(group_indices))]

    positions_by_group: dict[int, list[int]] = defaultdict(list)
    for position, group_index in enumerate(group_indices):
        if group_index is None:
            raise ValueError(
                f"group_index is required for reward normalization, but sample at position {position} has none"
            )
        positions_by_group[group_index].append(position)

    reward_tensor = torch.tensor(rewards, dtype=torch.float)
    normalized_rewards = torch.empty_like(reward_tensor)
    for positions in positions_by_group.values():
        # One entry per attempt. A missing rollout id makes the sample its own
        # attempt; the tuple key keeps that apart from a real id of the same
        # value, since rollout_id may be a string on custom rollout paths.
        # Segments of one attempt carry the same reward, so the reduction is
        # lossless -- but take the max rather than first-write-wins so that a
        # custom path emitting disagreeing segments degrades to "solved if any
        # segment solved" instead of depending on sample order.
        attempt_rewards: dict[tuple[str, object], float] = {}
        for position in positions:
            rollout_id = None if rollout_ids is None else rollout_ids[position]
            key = ("position", position) if rollout_id is None else ("rollout", rollout_id)
            reward = float(rewards[position])
            attempt_rewards[key] = max(attempt_rewards.get(key, reward), reward)

        attempt_tensor = torch.tensor(list(attempt_rewards.values()), dtype=torch.float)
        mean = attempt_tensor.mean()
        # Take the std of the centered attempts, not of the raw ones. The two are
        # equal in exact arithmetic but not in float32, and centering first is
        # what per-sample normalization did -- so the one-attempt-per-sample case
        # stays bit-identical to it rather than drifting by ~1e-7.
        centered_attempts = attempt_tensor - mean
        # A lone attempt is already 0 after centering, and torch.std of one
        # element is NaN (it is the sample std), which would poison the gradient.
        # Reshaping could never produce a size-1 group; grouping by prompt can.
        std = centered_attempts.std() if normalize_std and len(attempt_rewards) > 1 else None

        group_rewards = reward_tensor[positions] - mean
        if std is not None:
            group_rewards = group_rewards / (std + 1e-6)
        normalized_rewards[positions] = group_rewards

    return normalized_rewards.tolist()
