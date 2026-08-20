#!/usr/bin/env python3
"""Measure what ``reward / K`` would change, from dumped rollout data.

Background: on the fan-out path one trajectory becomes K training Samples
sharing a ``rollout_id``.  Until PR #2161 the outcome reward was split as
``reward / K`` across them; #2161 changed it to assign the full reward to
each.  This script quantifies the difference offline -- no GPUs, no training
run -- by replaying both reward assignments through GRPO's group
normalization and classifying every group by whether the two can differ at
all.

The classification is the point.  GRPO centers and std-normalizes within a
group, so scaling rewards only matters when it changes the group's *shape*:

* **identical** -- ``reward / K`` reproduces the full-reward vector exactly.
  Every all-zero-reward group lands here, as does any group where each
  solved sample happens to have ``K == 1``.
* **noise** -- the raw rewards are all equal (e.g. every sample solved), so
  the group's std is 0 and every advantage should be 0.  Dividing by a
  varying K breaks the tie, yielding non-zero advantages that encode nothing
  but how many segments each trajectory happened to produce.
* **distorted** -- the raw rewards genuinely vary and ``/K`` changes their
  relative magnitudes (possibly their order).  These are the groups GRPO
  actually learns from.

Most groups solve nothing, so their advantages are zero on both arms and they
train nothing either way.  Reporting ``changed / all groups`` therefore buries
the effect; the headline number here is ``changed / groups that carry signal``.
``--by-solve-rate`` breaks the classification down per group solve rate, which
shows where the damage concentrates: mixed-outcome groups, not the all-solved
ones (a group whose samples share a single K keeps its advantages intact,
because one constant factor is exactly what the per-group std divides out).

Usage:
    tools/ablate_reward_per_k.py 'runs/*/rollout_dumps/rollout_*.pt'
    tools/ablate_reward_per_k.py --by-solve-rate 'exp_a/**/rollout_*.pt' 'exp_b/**/rollout_*.pt'
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

# Groups are bucketed by their own solve rate. Bucket i covers
# [EDGES[i], EDGES[i+1]); the last bucket is closed on the right.
SOLVE_RATE_EDGES = (0.0, 0.001, 0.25, 0.5, 0.75, 1.0)

IDENTICAL, NOISE, DISTORTED = "identical", "noise", "distorted"


def group_normalize(rewards: np.ndarray) -> np.ndarray:
    """GRPO's within-group reward normalization: mean-center, then divide by std.

    Mirrors ``_post_process_rewards`` in ``slime/ray/rollout.py`` (and
    ``grpo_normalize_by_group_index`` in ``slime/rollout/_fanout_test_helpers.py``,
    which fixes that function's grouping for uneven fan-out). Two details are
    copied deliberately so the numbers here match what training would see:

    * ``ddof=1`` -- slime normalizes with ``torch.std``, whose default is the
      sample std, not numpy's population default.
    * ``+ 1e-6`` -- so a zero-variance group maps to all-zero advantages rather
      than to NaN. That is exactly the case ``/K`` corrupts.
    """
    centered = rewards - rewards.mean()
    if centered.size < 2:
        # ddof=1 is undefined for a single sample; slime would produce NaN here.
        # A one-sample group carries no relative signal either way.
        return np.zeros_like(centered)
    return centered / (centered.std(ddof=1) + 1e-6)


@dataclass
class Stats:
    """Everything one pass over the dumps collects."""

    dumps: int = 0
    dumps_all_zero: int = 0
    # Per-rollout (one trajectory, K sibling samples).
    rollouts: int = 0
    rollouts_k_gt_1: int = 0
    rollouts_solved: int = 0
    rollouts_solved_k_gt_1: int = 0  # the only rollouts whose reward /K alters
    k_dist: Counter = field(default_factory=Counter)
    # Per-group (one prompt's samples: GRPO's normalization unit).
    groups: int = 0
    kinds: Counter = field(default_factory=Counter)
    # Groups that produce a non-zero advantage under the full-reward arm, i.e.
    # the ones that actually train. Most groups are all-zero-reward and train
    # nothing, so a share of ALL groups understates the damage by ~40x; this is
    # the denominator that matters.
    signal_groups: int = 0
    signal_groups_changed: int = 0
    # kind counts per solve-rate bucket index.
    buckets: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    # Sample-level divergence. The two group kinds need different measures: in a
    # NOISE group every full-reward advantage is 0, so a sign comparison is
    # degenerate and a ratio has no denominator -- what matters there is the
    # magnitude of the advantage /K conjures up. Only DISTORTED groups support
    # the flip/ratio view.
    noise_samples: int = 0
    noise_injected_abs: list[float] = field(default_factory=list)
    distorted_samples: int = 0
    sign_flips: int = 0
    magnitude_ratios: list[float] = field(default_factory=list)


def classify_advantages(rewards: np.ndarray, full: np.ndarray, div_k: np.ndarray) -> str:
    """Classify one group from its two normalized advantage vectors.

    Split out from :func:`classify_group` so the accumulation pass can reuse
    advantages it already computed.
    """
    if np.allclose(full, div_k):
        return IDENTICAL
    # The full arm gives this group all-zero advantages (its rewards are all
    # equal), so whatever /K produces here is signal invented from segment counts.
    return NOISE if rewards.std() < 1e-12 else DISTORTED


def classify_group(rewards: np.ndarray, ks: np.ndarray) -> str:
    """Return IDENTICAL / NOISE / DISTORTED for one group.

    ``rewards`` are the raw per-sample rewards under the full-reward arm (every
    sibling of a trajectory carries the same value); ``ks`` is each sample's K.

    The comparison is made on the NORMALIZED advantages, not on the raw rewards,
    because normalization absorbs some scalings entirely and only the advantages
    reach training. Two cases look like a difference on raw values but are not:

    * a one-sample group -- centering sends it to 0 on both arms;
    * a group whose every K is equal -- ``/K`` is then one constant factor, which
      the per-group std divides straight back out.

    Classifying on raw values would count both as corrupted and overstate the
    effect (on one measured run, by 21 groups out of 21).
    """
    return classify_advantages(rewards, group_normalize(rewards), group_normalize(rewards / ks))


def solve_rate_bucket(rewards: np.ndarray) -> int:
    rate = float((rewards > 0).mean())
    for i in range(len(SOLVE_RATE_EDGES) - 1):
        if rate < SOLVE_RATE_EDGES[i + 1]:
            return i
    return len(SOLVE_RATE_EDGES) - 2


def bucket_label(i: int) -> str:
    lo, hi = SOLVE_RATE_EDGES[i], SOLVE_RATE_EDGES[i + 1]
    if i == 0:
        return "0 (none solved)"
    if i == len(SOLVE_RATE_EDGES) - 2:
        return f"[{lo:.0%}, {hi:.0%}]"
    return f"[{lo:.0%}, {hi:.0%})"


def load_samples(path: str) -> list[dict]:
    """Read one ``--save-debug-rollout-data`` dump as a list of sample dicts.

    The dump holds ``Sample.to_dict()`` output; the three fields used here
    (``rollout_id``, ``group_index``, ``reward``) are read straight off the
    dict rather than through ``Sample.from_dict``, which would rebuild the
    token lists for nothing.
    """
    return torch.load(path, weights_only=False)["samples"]


def accumulate(path: str, stats: Stats) -> None:
    samples = load_samples(path)
    if not samples:
        raise SystemExit(f"{path}: no samples in dump")

    stats.dumps += 1
    if not any(float(s.get("reward") or 0.0) for s in samples):
        # One such dump is ordinary (a hard batch where nothing was solved); a
        # whole run of them is the signature of an abandoned crash segment,
        # which would drag every rate in this report toward zero. Counted here
        # and checked across all dumps in main().
        stats.dumps_all_zero += 1

    k_by_rollout: Counter = Counter()
    for s in samples:
        k_by_rollout[s.get("rollout_id")] += 1

    reward_by_rollout: dict = {}
    for s in samples:
        reward_by_rollout.setdefault(s.get("rollout_id"), float(s.get("reward") or 0.0))

    for rollout_id, k in k_by_rollout.items():
        stats.rollouts += 1
        stats.k_dist[k] += 1
        solved = reward_by_rollout[rollout_id] > 0
        stats.rollouts_k_gt_1 += k > 1
        stats.rollouts_solved += solved
        stats.rollouts_solved_k_gt_1 += solved and k > 1

    groups: dict = defaultdict(list)
    for s in samples:
        groups[s.get("group_index")].append(s)

    for group in groups.values():
        rewards = np.array([float(s.get("reward") or 0.0) for s in group])
        ks = np.array([k_by_rollout[s.get("rollout_id")] for s in group], dtype=float)

        full = group_normalize(rewards)
        div_k = group_normalize(rewards / ks)
        kind = classify_advantages(rewards, full, div_k)
        stats.groups += 1
        stats.kinds[kind] += 1
        stats.buckets[solve_rate_bucket(rewards)][kind] += 1

        # A group whose full-reward advantages are all zero contributes no
        # gradient, so it cannot be harmed -- and it is the common case. Track
        # the trainable subset separately.
        if np.any(np.abs(full) > 1e-6):
            stats.signal_groups += 1
            stats.signal_groups_changed += kind is not IDENTICAL

        if kind is IDENTICAL:
            continue
        if kind is NOISE:
            # full is all-zero here by construction; record how large an
            # advantage /K invents out of a group that should carry no signal.
            stats.noise_samples += len(group)
            stats.noise_injected_abs.extend(np.abs(div_k).tolist())
        else:
            stats.distorted_samples += len(group)
            stats.sign_flips += int(np.sum(np.sign(full) != np.sign(div_k)))
            for a, b in zip(full, div_k, strict=True):
                if abs(a) > 1e-6:
                    stats.magnitude_ratios.append(abs(b) / abs(a))


def report(stats: Stats, *, by_solve_rate: bool) -> None:
    pct = lambda n, d: f"{100 * n / d:.1f}%" if d else "n/a"  # noqa: E731

    print(f"\ndumps read: {stats.dumps}")
    if stats.dumps_all_zero:
        print(f"  of which all-zero reward: {stats.dumps_all_zero} (check these are hard batches, not crash segments)")

    print(f"\nrollouts (one trajectory each): {stats.rollouts}")
    print(f"  K distribution:      {dict(sorted(stats.k_dist.items()))}")
    print(f"  K > 1:               {stats.rollouts_k_gt_1} ({pct(stats.rollouts_k_gt_1, stats.rollouts)})")
    print(f"  reward > 0:          {stats.rollouts_solved} ({pct(stats.rollouts_solved, stats.rollouts)})")
    print(
        f"  reward > 0 and K > 1: {stats.rollouts_solved_k_gt_1}"
        f" ({pct(stats.rollouts_solved_k_gt_1, stats.rollouts)})"
        "   <- the only rollouts whose reward value /K alters"
    )

    print(f"\ngroups (GRPO normalization unit): {stats.groups}")
    for kind, blurb in (
        (IDENTICAL, "reward/K reproduces the full-reward vector exactly"),
        (NOISE, "raw std == 0, so /K turns zero advantages into noise"),
        (DISTORTED, "raw rewards vary and /K changes their relative magnitudes"),
    ):
        n = stats.kinds[kind]
        print(f"  {kind:<10} {n:5d} ({pct(n, stats.groups)})  {blurb}")
    changed = stats.kinds[NOISE] + stats.kinds[DISTORTED]
    print(f"  => /K changes the gradient in {changed}/{stats.groups} groups ({pct(changed, stats.groups)})")

    # The headline. Groups whose full-reward advantages are all zero train
    # nothing, and they dominate the count, so the share above is diluted by
    # them. Among groups that DO carry signal, /K corrupts most of them.
    print(
        f"\ngroups carrying signal under the full arm: {stats.signal_groups} ({pct(stats.signal_groups, stats.groups)})"
    )
    print(
        f"  of those, corrupted by /K: {stats.signal_groups_changed}"
        f" ({pct(stats.signal_groups_changed, stats.signal_groups)})"
        "   <- the share that matters"
    )

    if stats.noise_samples:
        inj = np.array(stats.noise_injected_abs)
        print(f"\nnoise groups ({stats.kinds[NOISE]}), {stats.noise_samples} samples:")
        print("  every full-reward advantage here is 0; |advantage| that /K invents:")
        print(f"    median {np.median(inj):.3f}  p90 {np.percentile(inj, 90):.3f}  max {inj.max():.3f}")

    if stats.distorted_samples:
        ratios = np.array(stats.magnitude_ratios) if stats.magnitude_ratios else np.zeros(1)
        print(f"\ndistorted groups ({stats.kinds[DISTORTED]}), {stats.distorted_samples} samples:")
        print(f"  advantage sign flips:    {stats.sign_flips} ({pct(stats.sign_flips, stats.distorted_samples)})")
        print(f"  |adv_divk| / |adv_full|: median {np.median(ratios):.3f}  p90 {np.percentile(ratios, 90):.3f}")

    if not by_solve_rate:
        return

    print("\ngroups by their own solve rate:")
    print(f"  {'solve rate':<18} {'groups':>7} {'identical':>10} {'noise':>7} {'distorted':>10} {'changed':>9}")
    for i in sorted(stats.buckets):
        counts = stats.buckets[i]
        total = sum(counts.values())
        chg = counts[NOISE] + counts[DISTORTED]
        print(
            f"  {bucket_label(i):<18} {total:>7} {counts[IDENTICAL]:>10} {counts[NOISE]:>7}"
            f" {counts[DISTORTED]:>10} {pct(chg, total):>9}"
        )
    print(
        "\n  MIXED groups (a partial solve rate) are where /K does its damage: those\n"
        "  are the groups GRPO actually learns from, and there /K rescales real\n"
        "  advantages. The all-solved bucket is mostly safe because a group whose\n"
        "  samples share one K keeps its zero advantages -- only a group with two\n"
        "  distinct K among its solved samples becomes a noise source."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure offline what reward/K would change, from dumped rollout data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "patterns",
        nargs="+",
        metavar="GLOB",
        help="glob(s) matching --save-debug-rollout-data dumps (quote them so the shell does not expand)",
    )
    parser.add_argument(
        "--by-solve-rate",
        action="store_true",
        help="also break the group classification down by each group's solve rate",
    )
    parser.add_argument(
        "--allow-zero-reward",
        action="store_true",
        help="accept a dump set whose every reward is 0 (usually an abandoned crash segment)",
    )
    args = parser.parse_args(argv)

    paths = sorted({p for pattern in args.patterns for p in glob.glob(pattern, recursive=True)})
    if not paths:
        raise SystemExit(f"no dumps matched: {' '.join(args.patterns)}")

    stats = Stats()
    for path in paths:
        accumulate(path, stats)

    if stats.dumps_all_zero == stats.dumps and not args.allow_zero_reward:
        # Every rate below would be structurally zero, and the usual cause is a
        # glob that swept in an abandoned crash segment rather than a run where
        # genuinely nothing was solved.
        raise SystemExit(
            f"every one of the {stats.dumps} matched dumps has all-zero rewards.\n"
            "  That is the signature of an abandoned crash segment, not of a run "
            "where nothing was solved.\n"
            "  Narrow the glob, or pass --allow-zero-reward if you know these dumps are sound."
        )

    report(stats, by_solve_rate=args.by_solve_rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
