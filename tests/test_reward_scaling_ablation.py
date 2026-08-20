"""Tests for the ``reward / K`` ablation: the reward hook and the offline analyzer.

Two units, one question each:

* ``slime.rollout._reward_scaling_ablation.post_process_rewards`` -- does each arm
  scale rewards the way the arm claims, and is the ``full`` arm a strict no-op
  relative to the ordinary grouped normalization?
* ``tools.ablate_reward_per_k`` -- does the group classifier put a group in the
  right bucket, especially the all-solved-with-varying-K case that is the whole
  point of the measurement?

Both are pure CPU and need no dumps: the analyzer's IO is one function
(``load_samples``) that these tests bypass, feeding sample dicts directly.
"""

from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.rollout import _reward_scaling_ablation as abl  # noqa: E402
from slime.rollout._fanout_test_helpers import grpo_normalize_by_group_index  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

ablate = importlib.import_module("tools.ablate_reward_per_k")


ARGS = Namespace(reward_key=None, grpo_std_normalization=True)


def make_samples(spec: list[tuple[int, float, int]]) -> list[Sample]:
    """Build samples from ``(group_index, reward, k)`` triples.

    Each triple expands to ``k`` siblings sharing a fresh ``rollout_id`` and
    carrying the same reward -- the shape the full-reward arm produces, and what
    both the hook and the analyzer consume.
    """
    samples: list[Sample] = []
    for rollout_id, (group_index, reward, k) in enumerate(spec):
        for _ in range(k):
            samples.append(
                Sample(index=len(samples), group_index=group_index, rollout_id=rollout_id, prompt="", reward=reward)
            )
    return samples


def as_dicts(samples: list[Sample]) -> list[dict]:
    return [s.to_dict() for s in samples]


@pytest.fixture(autouse=True)
def _reset_mode(monkeypatch):
    """Clear the arm env var and the once-per-process log latch between tests."""
    monkeypatch.delenv(abl.MODE_ENV, raising=False)
    monkeypatch.setattr(abl, "_logged_mode", False)


# ---------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------


def test_full_arm_matches_the_plain_grouped_normalization():
    # The `full` arm must be a strict identity on today's behaviour, otherwise
    # any difference measured against `div_k` is partly the hook's own doing.
    samples = make_samples([(0, 1.0, 3), (0, 0.0, 1), (0, 0.5, 2), (1, 0.0, 4), (1, 1.0, 1)])

    raw, out = abl.post_process_rewards(ARGS, samples)
    ref_raw, ref_out = grpo_normalize_by_group_index(ARGS, samples)

    assert raw == ref_raw
    assert out == pytest.approx(ref_out)


@pytest.mark.parametrize("mode", abl.MODES)
def test_arms_agree_when_every_k_is_one(mode, monkeypatch):
    # /K is the identity at K == 1, so the arms may only diverge on fan-out.
    monkeypatch.setenv(abl.MODE_ENV, mode)
    samples = make_samples([(0, 1.0, 1), (0, 0.0, 1), (0, 0.25, 1), (1, 1.0, 1), (1, 0.0, 1)])

    _, out = abl.post_process_rewards(ARGS, samples)
    _, ref_out = grpo_normalize_by_group_index(ARGS, samples)

    assert out == pytest.approx(ref_out)


def test_div_k_arm_divides_by_the_sibling_count(monkeypatch):
    monkeypatch.setenv(abl.MODE_ENV, abl.DIV_K)
    # One group: reward 1.0 at K=4, reward 1.0 at K=1, reward 0.0 at K=2.
    samples = make_samples([(0, 1.0, 4), (0, 1.0, 1), (0, 0.0, 2)])

    raw, out = abl.post_process_rewards(ARGS, samples)

    # raw is reported unscaled on every arm: it drives logging and pass-rate.
    assert raw == [1.0] * 4 + [1.0] + [0.0] * 2

    expected_scaled = np.array([0.25] * 4 + [1.0] + [0.0] * 2)
    # ddof=1: slime normalizes with torch.std, whose default is the sample std.
    centered = expected_scaled - expected_scaled.mean()
    assert out == pytest.approx((centered / (centered.std(ddof=1) + 1e-6)).tolist(), abs=1e-5)


def test_div_k_turns_an_all_solved_group_into_noise(monkeypatch):
    # The mechanism under test. Every sample solved => the full arm sees zero
    # variance and emits zero advantages (no signal, correctly). Dividing by a
    # varying K breaks the tie and manufactures gradient out of segment counts.
    samples = make_samples([(0, 1.0, 1), (0, 1.0, 2), (0, 1.0, 4)])

    _, full_out = abl.post_process_rewards(ARGS, samples)
    assert full_out == pytest.approx([0.0] * len(samples), abs=1e-5)

    monkeypatch.setenv(abl.MODE_ENV, abl.DIV_K)
    _, div_k_out = abl.post_process_rewards(ARGS, samples)
    assert max(abs(v) for v in div_k_out) > 1.0


def test_groups_are_centered_independently(monkeypatch):
    # Both arms must group per prompt, exactly as the default
    # _post_process_rewards does; a grouping difference between the arms would
    # confound the effect under test. Each group must sum to ~0 on its own.
    monkeypatch.setenv(abl.MODE_ENV, abl.DIV_K)
    samples = make_samples([(0, 1.0, 3), (0, 0.0, 1), (1, 1.0, 1), (1, 0.0, 2), (1, 0.5, 4)])

    _, out = abl.post_process_rewards(ARGS, samples)

    for group_index in (0, 1):
        vals = [v for v, s in zip(out, samples, strict=True) if s.group_index == group_index]
        assert sum(vals) == pytest.approx(0.0, abs=1e-4)


def test_unknown_arm_is_rejected(monkeypatch):
    monkeypatch.setenv(abl.MODE_ENV, "half")
    with pytest.raises(ValueError, match="SLIME_REWARD_K_MODE"):
        abl.post_process_rewards(ARGS, make_samples([(0, 1.0, 1)]))


# ---------------------------------------------------------------------------
# the offline analyzer
# ---------------------------------------------------------------------------


def classify(spec: list[tuple[int, float, int]]) -> str:
    samples = make_samples(spec)
    rewards = np.array([s.reward for s in samples], dtype=float)
    ks = np.array([spec[s.rollout_id][2] for s in samples], dtype=float)
    return ablate.classify_group(rewards, ks)


@pytest.mark.parametrize(
    "name, spec, expected",
    [
        # Nothing solved: 0/K == 0, so the arms cannot differ whatever K does.
        ("all zero, K varies", [(0, 0.0, 1), (0, 0.0, 4)], ablate.IDENTICAL),
        # Solved samples all sit at K == 1, so /K leaves them untouched.
        ("solved only at K=1", [(0, 1.0, 1), (0, 0.0, 3)], ablate.IDENTICAL),
        # No fan-out at all.
        ("every K is 1", [(0, 1.0, 1), (0, 0.0, 1)], ablate.IDENTICAL),
        # A single K among the solved samples scales the group by one constant,
        # which the per-group std divides right back out.
        ("solved at one K > 1", [(0, 1.0, 2), (0, 0.0, 1)], ablate.IDENTICAL),
        # Uniform K across the whole group: same constant-factor argument.
        ("all solved, K uniform", [(0, 1.0, 2), (0, 1.0, 2)], ablate.IDENTICAL),
        # A one-sample group is centered to 0 on both arms.
        ("single sample", [(0, 1.0, 3)], ablate.IDENTICAL),
        # The case the whole measurement exists for: no variance to normalize,
        # so /K is the only thing producing advantages.
        ("all solved, K varies", [(0, 1.0, 1), (0, 1.0, 2)], ablate.NOISE),
        # Two distinct K among the solved samples: not a constant factor, so the
        # relative magnitudes really move.
        ("solved at two different K", [(0, 1.0, 2), (0, 1.0, 4), (0, 0.0, 1)], ablate.DISTORTED),
    ],
)
def test_group_classification(name, spec, expected):
    assert classify(spec) == expected, name


def test_uniform_k_cancels_in_the_hook_too():
    # The counterpart of the "all solved, K uniform" classification above,
    # checked against the real hook rather than the analyzer: one constant factor
    # survives neither arm, so both emit zero advantages.
    samples = make_samples([(0, 1.0, 2), (0, 1.0, 2)])
    _, full_out = abl.post_process_rewards(ARGS, samples)
    assert full_out == pytest.approx([0.0] * len(samples), abs=1e-5)


def test_accumulate_counts_rollouts_groups_and_kinds(monkeypatch):
    # group 0: all solved with K in {1, 2}      -> noise
    # group 1: nothing solved                   -> identical
    # group 2: solved at two distinct K, plus 0 -> distorted
    spec = [(0, 1.0, 1), (0, 1.0, 2), (1, 0.0, 2), (1, 0.0, 1), (2, 1.0, 3), (2, 1.0, 1), (2, 0.0, 1)]
    samples = as_dicts(make_samples(spec))
    monkeypatch.setattr(ablate, "load_samples", lambda path: samples)

    stats = ablate.Stats()
    ablate.accumulate("<synthetic>", stats)

    assert stats.dumps == 1
    assert stats.dumps_all_zero == 0
    assert stats.rollouts == len(spec)
    assert stats.k_dist == {1: 4, 2: 2, 3: 1}
    assert stats.rollouts_k_gt_1 == 3
    assert stats.rollouts_solved == 4
    # Only these alter a reward value: solved AND fanned out.
    assert stats.rollouts_solved_k_gt_1 == 2

    assert stats.groups == 3
    assert dict(stats.kinds) == {ablate.NOISE: 1, ablate.IDENTICAL: 1, ablate.DISTORTED: 1}

    # Only group 2 trains under the full arm: group 0 is all-solved (zero
    # variance) and group 1 solves nothing. So the headline share is 1/1, not
    # 2/3 -- the all-groups denominator is diluted by groups that train nothing.
    assert stats.signal_groups == 1
    assert stats.signal_groups_changed == 1


def test_accumulate_flags_an_all_zero_dump(monkeypatch):
    samples = as_dicts(make_samples([(0, 0.0, 1), (0, 0.0, 2)]))
    monkeypatch.setattr(ablate, "load_samples", lambda path: samples)

    stats = ablate.Stats()
    ablate.accumulate("<synthetic>", stats)

    # Flagged, not rejected: one all-zero rollout is an ordinary hard batch. The
    # crash-segment guard fires in main() only when EVERY dump looks like this.
    assert stats.dumps == 1
    assert stats.dumps_all_zero == 1


def test_noise_group_records_the_invented_advantage(monkeypatch):
    samples = as_dicts(make_samples([(0, 1.0, 1), (0, 1.0, 4)]))
    monkeypatch.setattr(ablate, "load_samples", lambda path: samples)

    stats = ablate.Stats()
    ablate.accumulate("<synthetic>", stats)

    # The full arm gives this group zero advantages, so any magnitude here is
    # gradient conjured from segment counts alone.
    assert stats.noise_samples == len(samples)
    assert max(stats.noise_injected_abs) > 0.5
    assert stats.distorted_samples == 0


def test_solve_rate_buckets_split_none_from_all():
    assert ablate.solve_rate_bucket(np.array([0.0, 0.0])) == 0
    assert ablate.solve_rate_bucket(np.array([1.0, 1.0])) == len(ablate.SOLVE_RATE_EDGES) - 2
    # A single solve out of eight must not land in the "none solved" bucket.
    assert ablate.solve_rate_bucket(np.array([1.0] + [0.0] * 7)) == 1


def test_group_normalize_zeros_a_flat_group():
    assert ablate.group_normalize(np.array([1.0, 1.0, 1.0])) == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)


def test_analyzer_normalization_matches_the_training_path():
    # The tool's advantages are only meaningful if they equal what slime would
    # compute. The subtle half is ddof: slime uses torch.std (sample std), numpy
    # defaults to the population std, and the two differ by ~8% at n=7.
    samples = make_samples([(0, 1.0, 4), (0, 1.0, 1), (0, 0.0, 2)])
    _, hook_out = abl.post_process_rewards(ARGS, samples)

    rewards = np.array([s.reward for s in samples], dtype=float)
    assert ablate.group_normalize(rewards).tolist() == pytest.approx(hook_out, abs=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
