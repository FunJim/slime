from pathlib import Path

import pytest

from slime.rollout.reward_utils import normalize_rewards_by_group

NUM_GPUS = 0


@pytest.mark.unit
def test_normalize_rewards_uses_explicit_uneven_groups():
    rewards = [0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 10.0, 11.0, 12.0, 13.0]
    group_indices = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    normalized = normalize_rewards_by_group(rewards, group_indices, normalize_std=False)

    assert normalized == pytest.approx([-1.5, -0.5, 0.5, 1.5, 0.0, 0.0, 0.0, -1.5, -0.5, 0.5, 1.5])


@pytest.mark.unit
def test_normalize_rewards_preserves_order_and_zeroes_singletons():
    rewards = [-1.0, 4.0, 0.0, 7.0, 1.0, 4.0]
    group_indices = [10, 20, 10, 30, 10, 20]

    normalized = normalize_rewards_by_group(rewards, group_indices, normalize_std=True)

    assert normalized == pytest.approx([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0], abs=1e-5)


@pytest.mark.unit
def test_normalize_rewards_requires_group_identity():
    with pytest.raises(ValueError, match="group_index is required.*position 1"):
        normalize_rewards_by_group([1.0, 2.0], [0, None], normalize_std=False)


@pytest.mark.unit
def test_normalize_rewards_supports_legacy_fixed_size_groups():
    normalized = normalize_rewards_by_group(
        [1.0, 3.0, 10.0, 10.0],
        [None, None, None, None],
        normalize_std=False,
        fallback_group_size=2,
    )

    assert normalized == pytest.approx([-1.0, 1.0, 0.0, 0.0])


@pytest.mark.unit
def test_normalize_rewards_rejects_unidentified_uneven_groups():
    with pytest.raises(ValueError, match="group_index is required when reward groups are not uniformly sized"):
        normalize_rewards_by_group(
            [1.0, 2.0, 3.0],
            [None, None, None],
            normalize_std=False,
            fallback_group_size=2,
        )


# ---------------------------------------------------------------------------
# Additions beyond the upstream PR: the properties that make this a bug fix
# rather than a refactor. Each of these fails against the reshape-based
# grouping it replaced.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_a_solved_prompt_does_not_lift_an_unsolved_one():
    # The concrete symptom of the collapsed grouping. Prompt 1 solved nothing, so
    # every one of its advantages must be 0 no matter what prompt 0 did; under
    # batch-wide centering they came out negative purely because prompt 0 solved.
    rewards = [1.0, 0.0] + [0.0] * 6
    group_indices = [0, 0] + [1] * 6

    normalized = normalize_rewards_by_group(rewards, group_indices, normalize_std=True)

    assert normalized[2:] == pytest.approx([0.0] * 6, abs=1e-6)
    assert normalized[0] > 0.0 > normalized[1]


@pytest.mark.unit
def test_a_lone_sample_is_finite():
    # Grouping by prompt can produce a size-1 group, which reshaping never could.
    # torch.std of one element is NaN, and a NaN advantage poisons the gradient.
    normalized = normalize_rewards_by_group([1.0, 1.0, 0.0], [0, 1, 1], normalize_std=True)

    assert all(value == value for value in normalized)  # NaN != NaN
    assert normalized[0] == pytest.approx(0.0)


@pytest.mark.unit
def test_matches_the_reshape_result_when_counts_are_uniform():
    # Backward compatibility: on the uniform layout the old reshape handled, the
    # grouping must be elementwise identical to it.
    import torch

    rewards = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    group_size = 4

    normalized = normalize_rewards_by_group(
        rewards, [i // group_size for i in range(len(rewards))], normalize_std=True
    )

    reference = torch.tensor(rewards, dtype=torch.float).reshape(-1, group_size)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    reference = reference / (reference.std(dim=-1, keepdim=True) + 1e-6)
    assert normalized == pytest.approx(reference.flatten().tolist(), abs=1e-5)


@pytest.mark.unit
def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        normalize_rewards_by_group([1.0, 2.0], [0], normalize_std=False)


# ---------------------------------------------------------------------------
# Call-site guard. The bug lived in RolloutManager._post_process_rewards, not in
# the function above, so the function passing its own tests is not enough -- a
# revert of the call site would leave every test here green. Asserted against
# the source text rather than by importing slime.ray.rollout, which pulls in
# sglang at module scope and cannot be imported in the CPU test job. Same
# read_text approach as tests/plugin_contracts/test_plugin_runtime_hook_contracts.py.
# ---------------------------------------------------------------------------

ROLLOUT_SOURCE = Path(__file__).resolve().parents[1] / "slime" / "ray" / "rollout.py"


@pytest.mark.unit
def test_rollout_manager_normalizes_through_this_function():
    source = ROLLOUT_SOURCE.read_text()

    assert "from slime.rollout.reward_utils import normalize_rewards_by_group" in source
    assert "rewards = normalize_rewards_by_group(" in source


@pytest.mark.unit
def test_rollout_manager_passes_per_prompt_group_identity():
    source = ROLLOUT_SOURCE.read_text()

    # Grouping must come from the samples, not from the batch shape.
    assert "[sample.group_index for sample in samples]" in source
    assert "fallback_group_size=self.args.n_samples_per_prompt" in source


@pytest.mark.unit
def test_the_batch_wide_collapse_is_gone():
    # The exact expression that normalized every prompt against one shared mean
    # whenever fan-out made the per-prompt counts uneven.
    source = ROLLOUT_SOURCE.read_text()

    assert "rewards.view(-1, rewards.shape[-1])" not in source
    assert "rewards.reshape(-1, self.args.n_samples_per_prompt)" not in source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
