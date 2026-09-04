"""Test-internal compact-rollout helpers used by ``test_qwen2.5_0.5B_fanout_short.py``.

The underscore prefix marks this as test infrastructure — it is not part
of the user-facing slime API and is not re-exported anywhere. It lives
in ``slime/`` only so the test can reference it by a dotted module path
(``--custom-generate-function-path`` / ``--custom-reward-post-process-path``
resolve a string via ``importlib.import_module``, which can't handle the
dots in the e2e test's filename).

Two helpers:

  - ``compact_generate``: fans one input sample out to N siblings
    sharing the same ``rollout_id``. That's the contract the rest of the
    framework (per-rollout step splitter, per-rollout-mean reducer,
    ``_validate_rollout_id_annotated`` validator) is built around.

  - ``grpo_normalize_by_group_index``: per-prompt GRPO reward
    normalization, still wired into the e2e test above via
    ``--custom-reward-post-process-path``. It is now redundant with the
    default (see ``slime/rollout/reward_utils.py``), but is kept both
    because users may have the flag pointed at it and because it gives
    the normalization tests an oracle that is not the implementation
    under test.
"""

import copy
import os
from collections import defaultdict


MAX_FANOUT = 3

# Each invocation appends one line. The test file reads this after train
# completes to assert the framework actually drove the custom path for
# every prompt (no silent bypass / no double-submission).
COUNTER_FILE_ENV = "SLIME_FANOUT_TEST_COUNTER_FILE"


async def compact_generate(args, sample, sampling_params):
    """One prompt → N siblings, deterministic N = 1 + (index % MAX_FANOUT).

    Strategy: call sglang once, deepcopy N-1 times. Bounded GPU cost —
    we're pinning the framework's per-rollout handling, not generation
    diversity.
    """
    from slime.rollout.sglang_rollout import generate

    counter_path = os.environ.get(COUNTER_FILE_ENV)
    if counter_path:
        try:
            with open(counter_path, "a") as f:
                f.write(f"{sample.index}\n")
        except OSError:
            # Counter file is best-effort — never fail training because of it.
            pass

    base_sample = await generate(args, sample, sampling_params)

    n = 1 + (sample.index % MAX_FANOUT)
    siblings = []
    for _ in range(n):
        s = copy.deepcopy(base_sample)
        # Critical invariant: all siblings share ``rollout_id`` so the
        # per-rollout reducer aggregates them as ONE rollout (not N) and
        # the rollout-aware step splitter keeps them in the same step.
        # ``group_index`` is inherited via ``deepcopy`` so production reward
        # normalization keeps the siblings in their prompt group.
        s.rollout_id = sample.index
        siblings.append(s)
    return siblings


def grpo_normalize_by_group_index(args, samples):
    """Reference implementation of per-prompt GRPO reward normalization.

    Equivalent to what the default ``_post_process_rewards`` now does via
    ``slime.rollout.reward_utils.normalize_rewards_by_group``: group by
    ``Sample.group_index`` -- the data-source-set per-prompt counter, preserved
    through the deepcopy in ``compact_generate`` -- then mean-center and
    optionally std-normalize within each group.

    Written as a ``--custom-reward-post-process-path`` workaround back when the
    default built its groups by reshaping to ``(-1, n_samples_per_prompt)`` and
    fell back to one batch-wide group whenever fan-out made the per-prompt count
    uneven. That fallback is gone, so this is no longer load-bearing, but it is
    kept: the e2e fan-out test still passes the flag, users may have it
    configured too, and duplicating the logic gives the normalization tests an
    oracle that is not the implementation under test.

    Returns ``(raw_rewards, normalized_rewards)`` matching the input
    ``samples`` order — same shape as the default's return contract.
    """
    import torch

    raw_rewards = [s.get_reward_value(args) for s in samples]

    # group_index → list of (original_position, raw_reward)
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s.group_index].append((i, raw_rewards[i]))

    out = [0.0] * len(samples)
    use_std = getattr(args, "grpo_std_normalization", True)
    for indexed in groups.values():
        positions = [p for p, _ in indexed]
        rewards = torch.tensor([r for _, r in indexed], dtype=torch.float)
        rewards = rewards - rewards.mean()
        if use_std:
            rewards = rewards / (rewards.std() + 1e-6)
        for pos, r in zip(positions, rewards.tolist(), strict=True):
            out[pos] = r

    return raw_rewards, out
