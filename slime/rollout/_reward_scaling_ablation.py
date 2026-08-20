"""Reward-scaling arms for the ``reward / K`` ablation.

On the fan-out path one trajectory becomes K training Samples sharing a
``rollout_id`` (sub-agent dispatch, auto-compaction, and re-tokenization forks
all raise K).  Until PR #2161 the outcome reward was split as ``reward / K``
across those siblings; #2161 changed it to give each sibling the full reward
and named that the "full outcome reward per turn" invariant, without stating a
reason.  This module reproduces both assignments as a
``--custom-reward-post-process-path`` hook so the two can be compared without
touching ``slime/agent/trajectory.py``:

* ``full`` (the default, and today's behaviour) -- every sibling keeps the
  trajectory's reward.
* ``div_k`` -- each sibling gets ``reward / K``, the pre-#2161 behaviour.

Pick the arm with ``SLIME_REWARD_K_MODE``.  Both arms share one code path and
differ only in the per-sample scale, so ``full`` is elementwise identical to
the ordinary grouped normalization (and to ``div_k`` whenever every K is 1);
``tests/test_reward_scaling_ablation.py`` pins that.

Scaling happens BEFORE the group normalization, matching where the historical
``/K`` lived (inside ``get_trajectory``, upstream of ``_post_process_rewards``).
Dividing afterwards would be a different, and meaningless, intervention: GRPO
re-centers per group, so a post-hoc divide would partly cancel.

The underscore prefix marks this as experiment infrastructure -- it is not part
of the user-facing slime API.  It lives under ``slime/`` only because
``--custom-reward-post-process-path`` resolves a dotted module path via
``importlib.import_module``, which cannot reach a file whose name contains dots.

Offline counterpart: ``tools/ablate_reward_per_k.py`` measures what the two arms
would change on already-dumped rollouts, without running training at all.
"""

import logging
import os
from collections import Counter

from slime.rollout.reward_utils import normalize_rewards_by_group

logger = logging.getLogger(__name__)

MODE_ENV = "SLIME_REWARD_K_MODE"
FULL, DIV_K = "full", "div_k"
MODES = (FULL, DIV_K)

_logged_mode = False


def _resolve_mode() -> str:
    mode = os.environ.get(MODE_ENV, FULL).strip().lower()
    if mode not in MODES:
        raise ValueError(f"{MODE_ENV}={mode!r} is not one of {MODES}")

    global _logged_mode
    if not _logged_mode:
        # Announced once per worker: the arm arrives through the Ray runtime env,
        # and a mode that silently failed to propagate is indistinguishable from
        # `full` in the metrics. Grep the log for this line to confirm the arm.
        logger.info("[reward_k_ablation] arm=%s (%s)", mode, MODE_ENV)
        _logged_mode = True
    return mode


def _k_by_rollout(samples) -> Counter:
    """Sibling count per ``rollout_id`` -- the K a trajectory fanned out to.

    Every sibling of one trajectory carries the same ``rollout_id``; the rollout
    layer enforces that for fan-out shapes (``_validate_rollout_id_annotated`` in
    ``slime/ray/rollout.py``), so counting them gives K directly.
    """
    counts: Counter = Counter()
    for s in samples:
        counts[s.rollout_id] += 1
    return counts


def post_process_rewards(args, samples):
    """``--custom-reward-post-process-path`` hook: scale per the arm, then group-normalize.

    Returns ``(raw_rewards, normalized_rewards)`` in the input order, matching
    the default's contract. ``raw_rewards`` are the UNSCALED rewards regardless
    of arm -- they feed logging and pass-rate metrics, so dividing them would
    make ``rollout/raw_reward`` incomparable across arms and would misreport the
    solve rate.

    Normalization is delegated to :func:`slime.rollout.reward_utils.normalize_rewards_by_group`,
    the same function the default ``_post_process_rewards`` calls. Reimplementing
    the grouping here would let the two drift apart, and a grouping difference
    between the arms would confound the effect under test.
    """
    mode = _resolve_mode()
    raw_rewards = [s.get_reward_value(args) for s in samples]

    ks = _k_by_rollout(samples)
    scaled = raw_rewards if mode == FULL else [r / ks[s.rollout_id] for r, s in zip(raw_rewards, samples, strict=True)]

    normalized = normalize_rewards_by_group(
        scaled,
        [s.group_index for s in samples],
        normalize_std=getattr(args, "grpo_std_normalization", True),
        fallback_group_size=getattr(args, "n_samples_per_prompt", None),
    )
    return raw_rewards, normalized
