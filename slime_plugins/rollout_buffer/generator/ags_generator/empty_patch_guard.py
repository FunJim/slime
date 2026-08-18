"""Detect coding-agent rollouts that exit successfully without editing anything.

Some agent CLIs treat a text-only final turn as a finished answer: the model
stops mid-task ("Let me look at query.py:"), no tool call is parsed, the CLI
exits 0, and the rollout yields an empty diff. Measured on SWE-bench Verified
with CodeBuddy Code, that accounts for roughly a third of all rollouts, so it is
worth counting and attributing rather than leaving it indistinguishable from a
genuine "no change needed" outcome.

This module only classifies. The policy decision (see
``AGSGeneratorConfig.empty_patch_guard``) lives in the caller, and the default
does not change rollout behaviour -- an empty-patch trajectory is still real
on-policy data whose zero reward is correct.
"""

from __future__ import annotations

import dataclasses
import logging
import re

from .weave_trace import iter_trajectory_events

logger = logging.getLogger(__name__)

# Reasons are attribution labels, not trigger conditions. The guard fires on
# "exit 0 + empty diff" alone; on SWE-bench an empty patch is a failure no
# matter how the agent phrased its last message, and gating the counter on
# string heuristics would make it brittle.
REASON_MIDTASK_NARRATION = "empty_patch_midtask_narration"
REASON_CLAIMED_COMPLETE = "empty_patch_claimed_complete"
REASON_NO_FINAL_TEXT = "empty_patch_no_final_text"
REASON_UNCLASSIFIED = "empty_patch_unclassified"

# Checked against the tail of the final message, where the intent actually sits.
_CONTINUATION_RE = re.compile(
    r"\b(?:let me|let's|now let me|next,?|i'll|i will|i'm going to|i need to|i should)\b",
    re.IGNORECASE,
)
_COMPLETION_RE = re.compile(
    r"\b(?:fix is complete|all tests? pass|issue is resolved|changes? (?:are|is) complete|"
    r"successfully (?:fixed|resolved)|the fix works)\b",
    re.IGNORECASE,
)
_FINAL_TEXT_TAIL_CHARS = 400
_FINAL_TEXT_EXCERPT_CHARS = 400


@dataclasses.dataclass(frozen=True)
class GuardVerdict:
    """Outcome of the empty-patch check for one rollout."""

    triggered: bool
    reason: str = ""
    final_text: str = ""


def final_agent_text(trajectory_path: str | None, *, agent: str) -> str | None:
    """Return the agent's last user-visible text, or None if unavailable.

    None means "could not tell" rather than "empty": the trajectory dump is
    disabled (``TRAJECTORY_DUMP_DIR`` unset, so ``dump_trajectory`` returned
    None) or the agent has no trajectory parser (codex yields no events).
    """
    if not trajectory_path:
        return None
    result_text: str | None = None
    last_assistant_text: str | None = None
    for event in iter_trajectory_events(trajectory_path, agent=agent):
        kind = event.get("kind")
        if kind == "result":
            output = event.get("output")
            if isinstance(output, dict) and output.get("result") is not None:
                result_text = str(output["result"])
        elif kind == "text":
            output = event.get("output")
            if isinstance(output, dict) and output.get("text"):
                last_assistant_text = str(output["text"])
    if result_text is not None:
        return result_text
    return last_assistant_text


def classify_empty_patch(
    *,
    agent_exit_code: int,
    diff_text: str,
    trajectory_path: str | None,
    agent: str,
) -> GuardVerdict:
    """Flag a rollout that reported success but produced no diff."""
    # git_diff returns "" (not None) when nothing changed.
    if agent_exit_code != 0 or (diff_text or "").strip():
        return GuardVerdict(triggered=False)

    text = final_agent_text(trajectory_path, agent=agent)
    if text is None:
        return GuardVerdict(triggered=True, reason=REASON_UNCLASSIFIED)
    if not text.strip():
        return GuardVerdict(triggered=True, reason=REASON_NO_FINAL_TEXT)

    excerpt = text[-_FINAL_TEXT_EXCERPT_CHARS:]
    tail = text.replace("<|im_end|>", " ")[-_FINAL_TEXT_TAIL_CHARS:]
    # Completion claims win: "all tests pass. Let me summarise." is a claimed
    # completion, not mid-task narration. With no diff it is a distinct failure
    # mode -- the agent believes it edited something it never edited.
    if _COMPLETION_RE.search(tail):
        return GuardVerdict(triggered=True, reason=REASON_CLAIMED_COMPLETE, final_text=excerpt)
    if _CONTINUATION_RE.search(tail):
        return GuardVerdict(triggered=True, reason=REASON_MIDTASK_NARRATION, final_text=excerpt)
    return GuardVerdict(triggered=True, reason=REASON_UNCLASSIFIED, final_text=excerpt)
