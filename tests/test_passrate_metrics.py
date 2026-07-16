import pytest

from slime.utils.metric_utils import compute_grouped_pass_rate, compute_pass_rate

NUM_GPUS = 0


@pytest.mark.unit
def test_compute_pass_rate_fixed_shape():
    metrics = compute_pass_rate(
        flat_rewards=[
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
        ],
        group_size=4,
        num_groups=2,
    )

    assert metrics["pass@1"] == pytest.approx(0.375)
    assert metrics["pass@2"] == pytest.approx((0.5 + 5 / 6) / 2)
    assert metrics["pass@4"] == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_pass_rate_skips_mismatched_fixed_shape():
    assert compute_pass_rate(flat_rewards=[0] * 23, group_size=4, num_groups=4) == {}


@pytest.mark.unit
def test_compute_grouped_pass_rate_deduplicates_fanout_segments():
    # Two prompt groups, four rollout attempts per prompt.  Rollout attempts 0
    # and 6 fan out into multiple train samples; they should still count as one
    # independent pass@k sample each.
    metrics = compute_grouped_pass_rate(
        flat_rewards=[
            1,
            1,  # group 0, rollout 0 fan-out sibling
            0,
            0,
            0,
            0,
            0,
            1,
            1,  # group 1, rollout 6 fan-out sibling
            1,
        ],
        group_indices=[
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
        ],
        rollout_ids=[
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            6,
            7,
        ],
        group_size=4,
    )

    assert metrics["pass@1"] == pytest.approx(0.375)
    assert metrics["pass@2"] == pytest.approx((0.5 + 5 / 6) / 2)
    assert metrics["pass@4"] == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_grouped_pass_rate_skips_k_when_group_has_too_few_attempts():
    metrics = compute_grouped_pass_rate(
        flat_rewards=[1, 0, 1, 0, 0],
        group_indices=[0, 0, 1, 1, 1],
        rollout_ids=[0, 1, 2, 3, 4],
        group_size=4,
    )

    # pass@4 is not estimable because no group has four independent attempts.
    assert "pass@4" not in metrics
    assert metrics["pass@1"] == pytest.approx((0.5 + 1 / 3) / 2)
    assert metrics["pass@2"] == pytest.approx((1.0 + 2 / 3) / 2)
