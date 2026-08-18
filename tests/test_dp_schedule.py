"""CPU unit tests for slime.utils.dp_schedule.build_dp_schedule.

The tests assert the invariants documented at the top of dp_schedule.py against
a range of static / dynamic / VPP / oversize / balance / uneven scenarios.
"""

from types import SimpleNamespace

import pytest

from slime.utils.dp_schedule import build_dp_schedule


NUM_GPUS = 0


def make_args(
    *,
    micro_batch_size=1,
    use_dynamic_batch_size=False,
    max_tokens_per_gpu=None,
    balance_data=False,
    balance_by_flops=False,
):
    return SimpleNamespace(
        micro_batch_size=micro_batch_size,
        use_dynamic_batch_size=use_dynamic_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        balance_data=balance_data,
        balance_by_flops=balance_by_flops,
        hidden_size=16,
        num_attention_heads=2,
        num_query_groups=2,
        vocab_size=32,
        ffn_hidden_size=64,
        num_experts=None,
        num_layers=2,
        kv_channels=8,
    )


def make_tp(dp_size=1, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1):
    return {
        "dp_size": dp_size,
        "cp_size": cp_size,
        "vpp_size": vpp_size,
        "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
    }


def assert_invariants(
    partitions,
    micro_batch_indices,
    num_microbatches,
    *,
    dp_size,
    expected_global_sample_indices,
    total_lengths,
    max_per_bin=None,
    allow_merged_over_cap=False,
):
    """Check the invariants documented at the top of dp_schedule.py.

    ``expected_global_sample_indices`` is the set of global sample indices
    that should end up covered (after trim). Trailing rollouts that don't
    fit are excluded.

    ``allow_merged_over_cap`` relaxes the per-mbs token cap to allow multi-sample
    mbs above it. Pass it for steps that had to merge bins down to satisfy the
    ``dp_size`` alignment (the path where every bin was an unsplittable singleton);
    on that path the cap is a target rather than a bound.
    """
    seen_global: set[int] = set()
    for r in range(dp_size):
        partition = partitions[r]
        mbi = micro_batch_indices[r]

        # Same num_mbs per rank (PP sync).
        assert len(mbi) == sum(num_microbatches), f"rank {r}: mbs count mismatch"

        # Flattened micro_batch_indices == range(len(partition)).
        flat = [i for mbs in mbi for i in mbs]
        assert flat == list(range(len(partition))), f"rank {r}: micro_batch_indices don't tile [0, n)"

        # Disjoint partitions whose union covers every kept sample.
        assert seen_global.isdisjoint(partition), f"rank {r}: overlap with other ranks"
        seen_global.update(partition)
    assert seen_global == set(expected_global_sample_indices), "covered sample set mismatch"

    if max_per_bin is None:
        return

    # Every mbs <= max_per_bin tokens, EXCEPT a singleton bin holding an oversized
    # sample (or, when allow_merged_over_cap, an mbs produced by alignment merging).
    for r in range(dp_size):
        partition = partitions[r]
        for mbs in micro_batch_indices[r]:
            bin_total = sum(total_lengths[partition[i]] for i in mbs)
            if bin_total > max_per_bin and not (allow_merged_over_cap and len(mbs) > 1):
                assert len(mbs) == 1, f"rank {r}: mbs sum {bin_total} > {max_per_bin} but contains {len(mbs)} samples"


@pytest.mark.unit
def test_static_stride_single_step():
    """Static + strided DP split, single step (1 rollout = 1 sample)."""
    total_lengths = [10] * 16
    rollout_indices = list(range(16))
    args = make_args(micro_batch_size=2)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=16, rollout_indices=rollout_indices
    )

    assert nmb == [2]
    assert gbs_per_step == [16]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=4,
        expected_global_sample_indices=range(16),
        total_lengths=total_lengths,
    )


@pytest.mark.unit
def test_static_balance_multi_step():
    """Static + balance_data + 2 training steps."""
    total_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1]
    rollout_indices = list(range(16))
    args = make_args(micro_batch_size=2, balance_data=True)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert nmb == [2, 2]
    assert gbs_per_step == [8, 8]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(16),
        total_lengths=total_lengths,
    )


@pytest.mark.unit
def test_balance_data_distributes_by_flops():
    """balance_data uses FLOPs weights for rank assignment, not raw token sums."""
    total_lengths = [1, 2, 3, 4, 5, 7, 9, 10]
    rollout_indices = list(range(8))
    args = make_args(micro_batch_size=1, balance_data=True)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert partitions == [[1, 2, 5, 6], [0, 3, 4, 7]]
    assert nmb == [4]
    assert gbs_per_step == [8]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
    )


@pytest.mark.unit
def test_dynamic_uniform():
    """Dynamic mbs on uniform-length samples."""
    total_lengths = [5] * 8
    rollout_indices = list(range(8))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert gbs_per_step == [8]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
        max_per_bin=10,
    )


@pytest.mark.unit
def test_dynamic_oversized_sample_lands_alone():
    """A sample larger than max_per_bin must end up alone in its mbs."""
    total_lengths = [15, 3, 3, 3, 3, 3, 3, 3]
    rollout_indices = list(range(8))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, rollout_indices=rollout_indices
    )

    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
        max_per_bin=10,
    )
    oversize_idx = total_lengths.index(15)
    found = False
    for r in range(2):
        if oversize_idx not in partitions[r]:
            continue
        local = partitions[r].index(oversize_idx)
        for mbs in mbi[r]:
            if local in mbs:
                assert mbs == [local], f"oversized sample shares an mbs: {mbs}"
                found = True
    assert found


@pytest.mark.unit
def test_dynamic_with_vpp_rounds_to_mb_group():
    """num_microbatches per rank should be a multiple of mb_group when vpp_size > 1."""
    total_lengths = [4] * 32
    rollout_indices = list(range(32))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=8)
    tp = make_tp(dp_size=2, vpp_size=2, microbatch_group_size_per_vp_stage=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=16, rollout_indices=rollout_indices
    )

    for n in nmb:
        assert n % 2 == 0, f"num_microbatches {n} is not a multiple of mb_group=2"
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(32),
        total_lengths=total_lengths,
        max_per_bin=8,
    )


@pytest.mark.unit
def test_rollout_grouping_keeps_samples_together():
    """compact / subagent simulation: rollout 0 emits 3 samples, rollout 1 emits 2,
    rollout 2 emits 4. Splitter keeps every rollout's samples in a single step."""
    rollout_indices = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    total_lengths = [3] * 9
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=1, rollout_indices=rollout_indices
    )

    # 3 rollouts / 1 per step → 3 steps, gbs constant.
    assert gbs_per_step == [1, 1, 1]
    # For each step, collect the samples (global indices) that landed in that step's mbs
    # on rank 0, then verify they exactly equal the rollout's sample positions.
    expected_per_step = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
    rank0_partition = partitions[0]
    mbs_cursor = 0
    for step_i, n_mbs in enumerate(nmb):
        step_locals = sorted(j for mbs in mbi[0][mbs_cursor : mbs_cursor + n_mbs] for j in mbs)
        step_globals = [rank0_partition[j] for j in step_locals]
        assert (
            sorted(step_globals) == expected_per_step[step_i]
        ), f"step {step_i} samples = {step_globals}, expected {expected_per_step[step_i]}"
        mbs_cursor += n_mbs
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=1,
        expected_global_sample_indices=range(9),
        total_lengths=total_lengths,
        max_per_bin=12,
    )


@pytest.mark.unit
def test_trims_trailing_rollouts_that_dont_fill_a_step():
    """5 rollouts, gbs=2 → 2 steps × 2 rollouts; trailing rollout 4 (sample positions 6, 7)
    is dropped."""
    rollout_indices = [0, 0, 1, 2, 2, 3, 4, 4]
    total_lengths = [3] * 8
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=2, rollout_indices=rollout_indices
    )

    assert gbs_per_step == [2, 2]
    # Sample positions 6 and 7 belong to the trimmed rollout 4 and must be absent.
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=1,
        expected_global_sample_indices=range(6),
        total_lengths=total_lengths,
        max_per_bin=12,
    )


@pytest.mark.unit
def test_rejects_when_fewer_rollouts_than_gbs():
    """gbs=4 with only 3 distinct rollouts → cannot form one step."""
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)
    with pytest.raises(AssertionError, match="num_rollouts"):
        build_dp_schedule(args, tp, [3] * 6, global_batch_size=4, rollout_indices=[0, 0, 1, 1, 2, 2])


# ---------------------------------------------------------------------------
# Alignment when no bin can be split: long-context runs where one sample alone
# fills max_per_bin, so first-fit emits one bin per sample. An odd sample count
# can then never be rounded UP to an even target_K, and the schedule must round
# down by merging instead of failing.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_odd_unsplittable_singletons_merge_instead_of_asserting():
    """Every sample fills a whole bin, and there is an odd number of them.

    Splitting is impossible (all bins are singletons), so the schedule merges the two
    smallest bins to reach an even mbs count rather than raising. Every sample must
    still be placed.
    """
    total_lengths = [10] * 9
    rollout_indices = list(range(9))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=9, rollout_indices=rollout_indices
    )

    # 9 singleton bins -> merged down to 8 -> 4 mbs per rank.
    assert nmb == [4]
    assert gbs_per_step == [9]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(9),
        total_lengths=total_lengths,
        max_per_bin=10,
        allow_merged_over_cap=True,
    )
    # Exactly one mbs holds two samples; the rest stay singletons.
    sizes = sorted(len(mbs) for r in range(2) for mbs in mbi[r])
    assert sizes == [1, 1, 1, 1, 1, 1, 1, 2], sizes


@pytest.mark.unit
@pytest.mark.parametrize("dp_size,num_samples", [(2, 3), (4, 5), (4, 7), (8, 9)])
def test_unsplittable_alignment_across_dp_sizes(dp_size, num_samples):
    """Merging down aligns to dp_size for any misaligned all-singleton bin count."""
    total_lengths = [10] * num_samples
    rollout_indices = list(range(num_samples))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=dp_size)

    partitions, mbi, nmb, _ = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=num_samples, rollout_indices=rollout_indices
    )

    total_mbs = sum(len(mbi[r]) for r in range(dp_size))
    assert total_mbs % dp_size == 0, f"{total_mbs} mbs not aligned to dp_size {dp_size}"
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=dp_size,
        expected_global_sample_indices=range(num_samples),
        total_lengths=total_lengths,
        max_per_bin=10,
        allow_merged_over_cap=True,
    )


# Real per-sample token lengths and rollout ids captured from the rollout dumps of run
# nspr3_stress_qwen35_35b_a3b_4nodes_20260810 (Qwen3.5-35B-A3B, 4 nodes, TP2/CP8 ->
# dp_size 2, max_tokens_per_gpu 12000 -> max_per_bin 96000). Rollout 1 crashed the
# unfixed scheduler with "could only produce 39 mbs ...; need 40" at step 2; rollout 0
# happened to survive. 48 rollouts each, but 77 / 74 training samples, because compact
# segmentation lets one rollout emit several samples.
#
# Stored as whitespace-separated strings rather than list literals purely so the
# formatter keeps them on one line instead of exploding to one element per line.
PROD_ROLLOUT0_LENGTHS = """
41660 48307 44888 37079 44934 46423 49780 55550 37387 46094 38929 34033 34547 39827 46419 50051 48041 51804
73102 46217 52259 72876 86641 44319 56019 46280 66098 86083 96000 53259 62065 65596 56250 73241 59030 84499
80569 88055 96000 96000 96000 96000 96000 96000 96000 96000 96000 81223 73704 96000 96000 96000 96000 96000
95456 96000 96000 96000 96000 96000 96000 90658 96000 96000 96000 44700 48326 59807 62062 44375 49897 61361
76566 68744 48080 47574 47846
"""
PROD_ROLLOUT0_ROLLOUT_IDS = """
1 5 0 7 7 2 6 4 3 3 10 13 8 15 12 11 14 9 44 46 42 47 40 45 45 41 41 43 28 29 27 26 25 25 30 31 24 22 20 20
20 20 19 19 21 21 21 17 17 16 16 16 16 18 18 18 18 23 23 23 23 23 23 23 23 34 33 36 36 38 37 35 35 39 32 32
32
"""
PROD_ROLLOUT1_LENGTHS = """
41582 76811 51088 70564 35093 67817 51789 46953 60184 62764 70756 40596 40050 40142 41407 40828 41753 42428
53024 42705 46273 32293 40500 42337 37544 32540 49951 32752 44972 49089 49369 46050 56614 58384 72760 77339
51852 60275 58984 78804 96000 96000 83523 96000 96000 95609 96000 96000 92476 95364 96000 96000 69612 96000
96000 96000 49568 57135 82158 59784 72398 90316 96000 96000 46691 59485 83890 80286 96000 96000 96000 96000
96000 96000
"""
PROD_ROLLOUT1_ROLLOUT_IDS = """
74 78 73 72 79 79 75 75 77 77 76 80 81 83 82 84 86 87 85 91 88 89 94 90 95 93 92 63 58 57 61 60 62 56 59 66
70 68 67 64 69 69 71 65 65 54 54 54 53 53 53 53 55 55 55 55 51 51 51 48 48 48 48 48 52 52 52 49 49 49 49 49
50 50
"""


def _ints(blob: str) -> list[int]:
    return [int(tok) for tok in blob.split()]


@pytest.mark.unit
@pytest.mark.parametrize(
    "lengths_blob,rollout_ids_blob",
    [
        (PROD_ROLLOUT0_LENGTHS, PROD_ROLLOUT0_ROLLOUT_IDS),
        (PROD_ROLLOUT1_LENGTHS, PROD_ROLLOUT1_ROLLOUT_IDS),
    ],
    ids=["rollout0", "rollout1"],
)
def test_production_long_context_rollouts_schedule(lengths_blob, rollout_ids_blob):
    """Regression: the real 96k-cap SWE rollouts that crashed the scheduler.

    num_steps_per_rollout=3 over 48 rollouts -> global_batch_size 16. Rollout 1's third
    step held 39 samples in 39 unsplittable bins and used to raise.
    """
    total_lengths = _ints(lengths_blob)
    rollout_indices = _ints(rollout_ids_blob)
    assert len(total_lengths) == len(rollout_indices), "fixture lengths/ids out of sync"
    assert len(set(rollout_indices)) == 48, "fixture should hold 48 distinct rollouts"

    max_per_bin = 12000 * 8
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12000)
    tp = make_tp(dp_size=2, cp_size=8)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=16, rollout_indices=rollout_indices
    )

    assert gbs_per_step == [16, 16, 16], "3 steps of 16 rollouts each"
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(len(total_lengths)),
        total_lengths=total_lengths,
        max_per_bin=max_per_bin,
        allow_merged_over_cap=True,
    )

    # Repacking is a last resort, so the overflow must stay marginal. Anything near 2x
    # the cap would risk OOM in real training.
    worst = max(sum(total_lengths[partitions[r][i]] for i in mbs) for r in range(2) for mbs in mbi[r])
    assert worst <= int(max_per_bin * 1.05), f"worst mbs {worst} exceeds the cap by more than 5%"


# ---------------------------------------------------------------------------
# Alignment fallback: failure mode and peak-memory behaviour.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rejects_when_fewer_mbs_than_alignment_threshold():
    """K < align_to cannot be aligned in either direction, and must say so.

    Splitting is impossible (all singletons) and rounding down lands on 0 mbs, so the
    step genuinely has no valid schedule. The error must name the real cause — too few
    samples for the alignment threshold — rather than blaming the repack.
    """
    total_lengths = [10] * 3  # each fills max_per_bin -> 3 unsplittable singleton bins
    rollout_indices = [0, 1, 2]
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    # vpp_size=2 with mb_group=2 makes align_to = dp_size * mb_group = 4 > 3 bins.
    tp = make_tp(dp_size=2, vpp_size=2, microbatch_group_size_per_vp_stage=2)

    with pytest.raises(AssertionError, match="below the alignment threshold"):
        build_dp_schedule(args, tp, total_lengths, global_batch_size=3, rollout_indices=rollout_indices)


@pytest.mark.unit
def test_repack_minimises_the_largest_mbs():
    """The repack must minimise the PEAK mbs, not just merge the smallest bins.

    Merging the two smallest bins repeatedly is the intuitive rule and is worse as soon
    as more than one merge is needed, because it keeps stacking onto the same bin. With
    align_to=4 and 7 singleton bins (3 merges) the difference is large enough to decide
    whether a real 96k-cap step OOMs.
    """
    # Every length is above max_per_bin/2 so no two samples ever share a bin.
    total_lengths = [49_000, 49_000, 49_000, 90_000, 90_000, 90_000, 90_000]
    rollout_indices = list(range(7))
    max_per_bin = 96_000
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=max_per_bin)
    tp = make_tp(dp_size=4)  # align_to = 4, so 7 bins must become 4

    partitions, mbi, nmb, _ = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=7, rollout_indices=rollout_indices
    )

    assert sum(len(mbi[r]) for r in range(4)) == 4, "7 singleton bins should repack to 4"
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=4,
        expected_global_sample_indices=range(7),
        total_lengths=total_lengths,
        max_per_bin=max_per_bin,
        allow_merged_over_cap=True,
    )

    worst = max(sum(total_lengths[partitions[r][i]] for i in mbs) for r in range(4) for mbs in mbi[r])
    # Optimal here is 139000 (90000+49000). The merge-two-smallest rule would stack all
    # three 49000s into one bin and peak at 147000; anything at or above that means the
    # peak-minimising property regressed.
    assert worst <= 139_000, f"peak mbs {worst} above the achievable minimum 139000"


@pytest.mark.unit
def test_single_merge_case_still_takes_the_smallest_pair():
    """With exactly one merge needed (the dp_size=2 production case), the peak-minimising
    repack must still combine the two smallest bins — that is optimal there, and it is
    what keeps the measured production overflow at ~0.3% of the cap."""
    total_lengths = [60_000, 61_000, 95_000, 96_000, 96_000]
    rollout_indices = list(range(5))
    max_per_bin = 96_000
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=max_per_bin)
    tp = make_tp(dp_size=2)  # align_to = 2, so 5 bins become 4

    partitions, mbi, nmb, _ = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=5, rollout_indices=rollout_indices
    )

    merged = [sorted(total_lengths[partitions[r][i]] for i in mbs) for r in range(2) for mbs in mbi[r] if len(mbs) > 1]
    assert merged == [[60_000, 61_000]], f"expected the two smallest bins merged, got {merged}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
