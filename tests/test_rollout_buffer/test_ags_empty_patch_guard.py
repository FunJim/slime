"""Unit tests for the AGS empty-patch guardrail and its config knob.

The guard flags rollouts where the coding agent exited 0 but left no diff --
the signature of a CLI that accepted a text-only mid-task turn as a finished
answer. Trajectories here are written as real CodeBuddy Code stream-json JSONL
(``{"type": "assistant", "message": {...}}`` / ``{"type": "result", ...}``) so
the tests exercise the same reader the runner uses.

Only ``empty_patch_guard`` and ``config`` are loaded, and they are loaded from
their files rather than through the package: ``rollout_buffer.generator.__init__``
imports ``openai`` and ``ags_generator.__init__`` imports the whole entry module,
neither of which the CPU CI jobs install. The sibling ``test_ags_generator.py``
does import through the package, which is why it cannot run in those jobs.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AGS_DIR = REPO_ROOT / "slime_plugins" / "rollout_buffer" / "generator" / "ags_generator"
_PRIVATE_PACKAGE = "_ags_guard_under_test"


def _load_ags_module(name: str):
    """Load one ags_generator module through a private synthetic package.

    The modules are mounted under ``_ags_guard_under_test`` -- a package whose
    ``__path__`` is the real ags_generator directory -- rather than under their
    true dotted path. That keeps relative imports working (``empty_patch_guard``
    does ``from .weave_trace import ...``) while leaving the real
    ``slime_plugins...`` entries in ``sys.modules`` untouched: stubbing those
    ancestors would linger for the whole pytest session and break sibling
    modules that import through the genuine package.
    """
    if _PRIVATE_PACKAGE not in sys.modules:
        package = types.ModuleType(_PRIVATE_PACKAGE)
        package.__path__ = [str(AGS_DIR)]
        sys.modules[_PRIVATE_PACKAGE] = package
    return importlib.import_module(f".{name}", package=_PRIVATE_PACKAGE)


_load_ags_module("weave_trace")  # empty_patch_guard imports it relatively
guard = _load_ags_module("empty_patch_guard")
AGSGeneratorConfig = _load_ags_module("config").AGSGeneratorConfig

NUM_GPUS = 0

AGENT = "codebuddy_code"


def _write_trajectory(tmp_path: Path, *, result_text: str | None, assistant_text: str | None = None) -> str:
    """Write a minimal CBC-shaped stream-json trajectory and return its path."""
    events: list[dict] = [{"type": "system", "subtype": "init", "session_id": "s"}]
    if assistant_text is not None:
        events.append(
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
                "__timestamp": "2026-07-27T00:00:00.000Z",
            }
        )
    if result_text is not None:
        events.append(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": result_text,
                "__timestamp": "2026-07-27T00:00:01.000Z",
            }
        )
    path = tmp_path / "agent.trajectory.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return str(path)


def test_guard_flags_midtask_narration(tmp_path):
    # verbatim-style mid-task stop from a real CBC run
    path = _write_trajectory(
        tmp_path, result_text="Let me look at the query.py file to understand values().<|im_end|>"
    )
    verdict = guard.classify_empty_patch(agent_exit_code=0, diff_text="", trajectory_path=path, agent=AGENT)
    assert verdict.triggered
    assert verdict.reason == guard.REASON_MIDTASK_NARRATION
    assert "query.py" in verdict.final_text


def test_guard_flags_claimed_completion_without_diff(tmp_path):
    # Claiming success with nothing to show is a distinct failure mode from
    # stopping mid-task, so it must not collapse into the narration bucket.
    path = _write_trajectory(tmp_path, result_text="The fix is complete. All tests pass.")
    verdict = guard.classify_empty_patch(agent_exit_code=0, diff_text="", trajectory_path=path, agent=AGENT)
    assert verdict.triggered
    assert verdict.reason == guard.REASON_CLAIMED_COMPLETE


def test_guard_prefers_completion_claim_over_trailing_narration(tmp_path):
    path = _write_trajectory(tmp_path, result_text="All tests pass. Let me provide a summary.")
    verdict = guard.classify_empty_patch(agent_exit_code=0, diff_text="", trajectory_path=path, agent=AGENT)
    assert verdict.reason == guard.REASON_CLAIMED_COMPLETE


def test_guard_flags_blank_final_text(tmp_path):
    path = _write_trajectory(tmp_path, result_text="   ")
    verdict = guard.classify_empty_patch(agent_exit_code=0, diff_text="", trajectory_path=path, agent=AGENT)
    assert verdict.triggered
    assert verdict.reason == guard.REASON_NO_FINAL_TEXT


def test_guard_falls_back_to_last_assistant_text(tmp_path):
    path = _write_trajectory(tmp_path, result_text=None, assistant_text="Let me check the tests first.")
    verdict = guard.classify_empty_patch(agent_exit_code=0, diff_text="", trajectory_path=path, agent=AGENT)
    assert verdict.reason == guard.REASON_MIDTASK_NARRATION


def test_guard_ignores_nonempty_diff(tmp_path):
    path = _write_trajectory(tmp_path, result_text="Let me look at query.py.")
    verdict = guard.classify_empty_patch(
        agent_exit_code=0,
        diff_text="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
        trajectory_path=path,
        agent=AGENT,
    )
    assert not verdict.triggered
    assert verdict.reason == ""


def test_guard_ignores_whitespace_only_diff_as_empty(tmp_path):
    path = _write_trajectory(tmp_path, result_text="Let me look at query.py.")
    verdict = guard.classify_empty_patch(agent_exit_code=0, diff_text="   \n\t\n", trajectory_path=path, agent=AGENT)
    assert verdict.triggered


def test_guard_ignores_nonzero_exit(tmp_path):
    path = _write_trajectory(tmp_path, result_text="Let me look at query.py.")
    verdict = guard.classify_empty_patch(agent_exit_code=1, diff_text="", trajectory_path=path, agent=AGENT)
    assert not verdict.triggered


def test_guard_triggers_without_trajectory_path():
    # TRAJECTORY_DUMP_DIR unset makes dump_trajectory return None; an empty patch
    # is still an empty patch, we just cannot attribute it.
    verdict = guard.classify_empty_patch(agent_exit_code=0, diff_text="", trajectory_path=None, agent=AGENT)
    assert verdict.triggered
    assert verdict.reason == guard.REASON_UNCLASSIFIED


def test_guard_is_inert_for_agent_without_parser(tmp_path):
    path = _write_trajectory(tmp_path, result_text="Let me look at query.py.")
    verdict = guard.classify_empty_patch(agent_exit_code=0, diff_text="", trajectory_path=path, agent="codex")
    assert verdict.triggered
    assert verdict.reason == guard.REASON_UNCLASSIFIED


def test_guard_survives_malformed_trajectory(tmp_path):
    path = tmp_path / "bad.trajectory.jsonl"
    path.write_text('not json\n{"type": "result"\n\n', encoding="utf-8")
    verdict = guard.classify_empty_patch(agent_exit_code=0, diff_text="", trajectory_path=str(path), agent=AGENT)
    assert verdict.triggered  # no exception


def test_guard_survives_missing_trajectory_file(tmp_path):
    verdict = guard.classify_empty_patch(
        agent_exit_code=0, diff_text="", trajectory_path=str(tmp_path / "absent.jsonl"), agent=AGENT
    )
    assert verdict.triggered


def test_empty_patch_guard_policy_defaults_to_metrics(monkeypatch):
    monkeypatch.delenv("SWE_EMPTY_PATCH_GUARD", raising=False)
    assert AGSGeneratorConfig.from_env().empty_patch_guard == "metrics"


@pytest.mark.parametrize("value,expected", [("off", "off"), ("abort", "abort"), ("METRICS", "metrics")])
def test_empty_patch_guard_policy_from_env(monkeypatch, value, expected):
    monkeypatch.setenv("SWE_EMPTY_PATCH_GUARD", value)
    assert AGSGeneratorConfig.from_env().empty_patch_guard == expected


def test_empty_patch_guard_policy_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("SWE_EMPTY_PATCH_GUARD", "retry")
    with pytest.raises(ValueError, match="SWE_EMPTY_PATCH_GUARD"):
        AGSGeneratorConfig.from_env()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
