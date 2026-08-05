from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import sys
import types
from types import SimpleNamespace

import pytest
from aiohttp import web
from tests.test_agent._fakes import FakeSandbox

from slime.utils.misc import SingletonMeta
from slime.utils.types import Sample
from slime_plugins.rollout_buffer.generator.ags_generator import adapter_service, swe_task
from slime_plugins.rollout_buffer.generator.ags_generator.config import AGSGeneratorConfig
from slime_plugins.rollout_buffer.generator.ags_generator.entry import (
    _collapse_eval_samples,
    get_group_data_meta_info,
    is_valid_group,
    transform_group,
)
from slime_plugins.rollout_buffer.generator.ags_generator.harnesses import (
    AGSSidecarClaudeCodeHarness,
    CodeBuddyCodeHarness,
    resolve_agent,
)
from slime_plugins.rollout_buffer.generator.ags_generator.rollout import AGSRolloutRunner
from slime_plugins.rollout_buffer.generator.ags_generator.runner import run_root_command
from slime_plugins.rollout_buffer.generator.ags_generator.sampling import normalize_sampling_params
from slime_plugins.rollout_buffer.generator.ags_generator.serialization import (
    output_item_from_samples,
    samples_from_payload,
)
from slime_plugins.rollout_buffer.generator.ags_generator.weave_trace import AGSWeaveTrace, iter_trajectory_events
from slime_plugins.rollout_buffer.rollout_buffer_example import start_rollout


def _sample(*, reward=1.0, status=Sample.Status.COMPLETED):
    return Sample(
        index=3,
        group_index=1,
        rollout_id=3,
        prompt="p",
        tokens=[1, 2, 3],
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=[0.0, 0.0],
        reward=reward,
        status=status,
        metadata={"trajectory_path": "/tmp/t.jsonl", "patch_path": "/tmp/p.patch", "rollout_dump_path": "/tmp/r.json"},
    )


def test_output_item_round_trips_compact_samples():
    samples = [_sample(), _sample(reward=1.0)]
    item = output_item_from_samples(samples, instance_id="inst-1")

    restored = samples_from_payload(item)

    assert len(restored) == 2
    assert restored[0].status == Sample.Status.COMPLETED
    assert restored[0].reward == 1.0
    assert restored[0].metadata["patch_path"] == "/tmp/p.patch"


def test_group_hooks_accept_complete_sample_payloads():
    item = output_item_from_samples([_sample()], instance_id="inst-1")
    group = ("inst-1", [item])

    assert is_valid_group(group, min_valid_group_size=1)
    assert transform_group(group) is group

    meta = get_group_data_meta_info({"inst-1": [item]})
    assert meta["total_samples"] == 1
    assert meta["avg_reward"] == 1.0
    assert meta["nonzero_reward_samples"] == 1
    assert meta["artifact_counts"] == {"trajectory": 1, "patch": 1, "rollout_dump": 1, "complete": 1}


def test_eval_collapse_keeps_one_scored_sample_per_attempt():
    base = Sample(index=7, prompt="p", metadata={"dataset": "eval"})
    segments = [
        _sample(reward=1.0),
        _sample(reward=1.0),
    ]
    segments[0].metadata = {**segments[0].metadata, "instance_id": "inst-1"}

    collapsed = _collapse_eval_samples(base, segments)

    assert collapsed.reward == 1.0
    assert collapsed.status == Sample.Status.COMPLETED
    assert collapsed.remove_sample is True
    assert collapsed.tokens == [0, 0]
    assert collapsed.response_length == 1
    assert collapsed.metadata["dataset"] == "eval"
    assert collapsed.metadata["instance_id"] == "inst-1"
    assert collapsed.metadata["eval_collapsed_segments"] == 2


def test_eval_collapse_marks_empty_output_aborted():
    collapsed = _collapse_eval_samples(Sample(index=7, metadata={"dataset": "eval"}), [])

    assert collapsed.reward == 0.0
    assert collapsed.status == Sample.Status.ABORTED
    assert collapsed.metadata["eval_collapsed_segments"] == 0


def test_sampling_params_use_sglang_generate_names():
    assert normalize_sampling_params({"max_tokens": 128, "temperature": 1.0}) == {
        "max_new_tokens": 128,
        "temperature": 1.0,
    }


def test_eval_isolated_sandbox_defaults_to_false(monkeypatch):
    monkeypatch.delenv("SWE_EVAL_ISOLATED_SANDBOX", raising=False)

    assert AGSGeneratorConfig.from_env().eval_isolated_sandbox is False


def test_eval_isolated_sandbox_can_be_enabled(monkeypatch):
    monkeypatch.setenv("SWE_EVAL_ISOLATED_SANDBOX", "true")

    assert AGSGeneratorConfig.from_env().eval_isolated_sandbox is True


def _adapter_args(**overrides):
    args = SimpleNamespace(
        hf_checkpoint="/models/fake",
        rollout_max_context_len=4096,
        sglang_tool_call_parser="qwen3_coder",
        sglang_reasoning_parser="qwen3",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_eval_falls_back_to_a_local_adapter_when_none_is_running(monkeypatch):
    """Eval must not require the training rollout to have built the adapter.

    Under --num-rollout 0, or eval-before-train on rollout 0, nothing has bound
    ADAPTER_PORT yet, so a RemoteAdapterProxy would fail every prompt with a
    connection error. Fall back to a local adapter on an ephemeral port.
    """
    monkeypatch.setenv("ADAPTER_PUBLIC_HOST", "10.0.0.1")
    monkeypatch.delenv("ADAPTER_PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("AGS_EVAL_ADAPTER_CONTROL_URL", raising=False)
    monkeypatch.delenv("ADAPTER_CONTROL_BASE_URL", raising=False)
    monkeypatch.setattr(adapter_service, "_remote_adapter_alive", lambda *a, **k: False)
    built: dict = {}

    def _fake_local(args, config, adapter_cls, *, port=None):
        built["port"] = port
        return SimpleNamespace(kind="local")

    monkeypatch.setattr(adapter_service, "_local_adapter_service", _fake_local)

    service = adapter_service.get_adapter_service(
        _adapter_args(), AGSGeneratorConfig.from_env(), object, evaluation=True
    )

    assert service.kind == "local"
    # Ephemeral: the training adapter may still claim ADAPTER_PORT later in the run.
    assert built["port"] == 0


def test_eval_reuses_the_training_adapter_when_one_is_live(monkeypatch):
    """The live training adapter owns the trajectory trees, so prefer it."""
    monkeypatch.setenv("ADAPTER_PUBLIC_HOST", "10.0.0.1")
    monkeypatch.delenv("ADAPTER_PUBLIC_BASE_URL", raising=False)
    monkeypatch.setattr(adapter_service, "_remote_adapter_alive", lambda *a, **k: True)
    monkeypatch.setattr(
        adapter_service,
        "_local_adapter_service",
        lambda *a, **k: pytest.fail("must not start a local adapter when one is already live"),
    )
    monkeypatch.setattr(adapter_service, "RemoteAdapterService", lambda args, config: SimpleNamespace(kind="remote"))

    service = adapter_service.get_adapter_service(
        _adapter_args(), AGSGeneratorConfig.from_env(), object, evaluation=True
    )

    assert service.kind == "remote"


def test_training_never_probes_for_a_remote_adapter(monkeypatch):
    """Training owns its adapter; it must not depend on a health probe."""
    monkeypatch.setenv("ADAPTER_PUBLIC_HOST", "10.0.0.1")
    monkeypatch.setattr(
        adapter_service,
        "_remote_adapter_alive",
        lambda *a, **k: pytest.fail("training must not probe for a remote adapter"),
    )
    monkeypatch.setattr(
        adapter_service, "_local_adapter_service", lambda *a, **k: SimpleNamespace(kind="local", port=k.get("port"))
    )

    service = adapter_service.get_adapter_service(
        _adapter_args(), AGSGeneratorConfig.from_env(), object, evaluation=False
    )

    assert service.kind == "local"


def test_ephemeral_eval_adapter_advertises_its_own_node_and_port(monkeypatch):
    """The fallback adapter must not advertise the head node's host:port.

    It runs in the RolloutManager actor, which Ray does not pin to the head, so
    ADAPTER_PUBLIC_HOST/ADAPTER_PUBLIC_BASE_URL (head + fixed port) would send
    sandboxes to the wrong address. Only the fixed-port path may use them.
    """
    from slime.utils.http_utils import get_host_info

    monkeypatch.setenv("ADAPTER_PUBLIC_HOST", "10.255.255.1")
    monkeypatch.setenv("ADAPTER_PUBLIC_BASE_URL", "http://10.255.255.1:18001")
    monkeypatch.setenv("ADAPTER_PORT", "18903")
    monkeypatch.setattr(adapter_service, "load_tokenizer", lambda *a, **k: SimpleNamespace())

    class _StubAdapter:
        def __init__(self, **kwargs):
            self.app = web.Application()

    config = AGSGeneratorConfig.from_env()
    SingletonMeta.clear_instances(adapter_service.AdapterService)
    try:
        ephemeral = adapter_service._local_adapter_service(_adapter_args(), config, _StubAdapter, port=0)
        assert ephemeral.adapter_url == f"http://{get_host_info()[1]}:{ephemeral.app_handle.port}"
        assert ephemeral.app_handle.port not in (0, config.adapter_port)

        SingletonMeta.clear_instances(adapter_service.AdapterService)
        fixed = adapter_service._local_adapter_service(_adapter_args(), config, _StubAdapter)
        assert fixed.adapter_url == "http://10.255.255.1:18001"
    finally:
        SingletonMeta.clear_instances(adapter_service.AdapterService)


def test_reusing_the_adapter_singleton_warns_when_args_differ(monkeypatch, caplog):
    """AdapterService is a singleton: later callers' args are silently ignored.

    A mismatch means the live adapter was built with a different tokenizer or
    context budget than this caller asked for, which would otherwise be invisible.
    """
    monkeypatch.setenv("ADAPTER_PUBLIC_HOST", "10.0.0.1")
    config = AGSGeneratorConfig.from_env()
    first = _adapter_args()
    live = SimpleNamespace(
        build_key=adapter_service._adapter_build_key(first, config, object, "http://127.0.0.1:30000")
    )
    monkeypatch.setattr(adapter_service, "AdapterService", lambda *a, **k: live)

    with caplog.at_level("WARNING"):
        same = adapter_service._local_adapter_service(first, config, object)
    assert same is live
    assert not caplog.records, "identical args must not warn"

    with caplog.at_level("WARNING"):
        adapter_service._local_adapter_service(_adapter_args(rollout_max_context_len=131072), config, object)

    assert any("singleton" in r.message for r in caplog.records)


_MD = {"instance_id": "x__1", "dataset_prompt": "# Task\n\nFix the bug.", "problem_statement": "Fix the bug."}


def _async_return(value):
    """Async callable ignoring its arguments and returning ``value``."""

    async def _call(*args, **kwargs):
        return value

    return _call


@pytest.mark.parametrize(
    "style,expected_prompt,expects_file",
    [
        ("dataset", "# Task\n\nFix the bug.", False),
        ("instruction", "Read PROBLEM_STATEMENT.md and fix it.", True),
    ],
)
def test_agent_prompt_and_statement_file_agree_per_style(monkeypatch, style, expected_prompt, expects_file):
    """The prompt style must decide the prompt and the statement file together.

    Wiring these two independently is how you get "instruction" without the file
    it names, or "dataset" with a stray file in the repo, so both are asserted
    from one style.
    """
    monkeypatch.setenv("SWE_CC_PROMPT", "Read PROBLEM_STATEMENT.md and fix it.")
    runner = AGSRolloutRunner.__new__(AGSRolloutRunner)  # no sandbox/adapter needed
    runner.config = AGSGeneratorConfig.from_env()

    assert runner._agent_prompt(_MD, style) == expected_prompt

    async def run_case():
        sb = FakeSandbox()
        await swe_task.prepare_workspace(sb, "/testbed", _MD, write_problem_statement=style == "instruction")
        return sb

    assert ("/testbed/PROBLEM_STATEMENT.md" in asyncio.run(run_case()).files) is expects_file


@pytest.mark.parametrize(
    "evaluation,expected_prompt",
    [(False, "Read PROBLEM_STATEMENT.md and fix it."), (True, "# Task\n\nFix the bug.")],
)
def test_generate_picks_prompt_style_by_rollout_mode(monkeypatch, evaluation, expected_prompt):
    """Training and eval must resolve to their own style from one config.

    This is the wiring generate() does, exercised end to end from the env vars:
    the default split is instruction for training, dataset for eval.
    """
    monkeypatch.delenv("SWE_PROMPT_STYLE", raising=False)
    monkeypatch.delenv("SWE_EVAL_PROMPT_STYLE", raising=False)
    monkeypatch.setenv("SWE_CC_PROMPT", "Read PROBLEM_STATEMENT.md and fix it.")
    runner = AGSRolloutRunner.__new__(AGSRolloutRunner)
    runner.config = AGSGeneratorConfig.from_env()

    style = runner.config.prompt_style_for(evaluation=evaluation)
    assert runner._agent_prompt(_MD, style) == expected_prompt


@pytest.mark.parametrize(
    "evaluation,expected_prompt,expects_file",
    [(False, "Read PROBLEM_STATEMENT.md and fix it.", True), (True, "# Task\n\nFix the bug.", False)],
)
def test_generate_threads_evaluation_flag_to_harness_and_workspace(
    monkeypatch, evaluation, expected_prompt, expects_file
):
    """Drive the real generate() and capture what the harness was handed.

    The helper tests above verify the style→prompt mapping; this one verifies the
    plumbing, which is the part that silently breaks: generate() must resolve the
    style from its own `evaluation` argument and use that same value for both the
    harness prompt and the PROBLEM_STATEMENT.md decision.
    """
    monkeypatch.delenv("SWE_PROMPT_STYLE", raising=False)
    monkeypatch.delenv("SWE_EVAL_PROMPT_STYLE", raising=False)
    monkeypatch.setenv("SWE_CC_PROMPT", "Read PROBLEM_STATEMENT.md and fix it.")

    captured: dict = {}
    sandbox = FakeSandbox()

    class _Harness:
        async def run(self, sb, *, workdir, session_id, adapter_url, time_budget_sec, prompt):
            captured["prompt"] = prompt
            return 0

    @contextlib.asynccontextmanager
    async def _fake_boot(self, image, instance_id):
        yield sandbox

    runner = AGSRolloutRunner.__new__(AGSRolloutRunner)
    runner.config = AGSGeneratorConfig.from_env()
    runner.harness_cls = _Harness
    runner.artifacts = SimpleNamespace(
        dump_trajectory=_async_return(None), dump_patch=lambda *a, **k: None, dump_rollout=lambda *a, **k: None
    )
    runner.weave_trace = SimpleNamespace(start_rollout=lambda **k: None, finish_rollout=lambda *a, **k: None)
    # finish_session returning [] short-circuits into _abort_result, which is fine:
    # the prompt and the workspace file are already decided by then.
    runner.adapter_service = SimpleNamespace(
        adapter=SimpleNamespace(
            open_session=lambda *a, **k: None,
            finish_session=_async_return([]),
            drop_session=_async_return(None),
        ),
        max_context_len=4096,
        adapter_url="http://127.0.0.1:1",
    )
    monkeypatch.setattr(AGSRolloutRunner, "_boot_agent_sandbox", _fake_boot)
    monkeypatch.setattr("slime_plugins.rollout_buffer.generator.ags_generator.rollout.git_diff", _async_return(""))
    monkeypatch.setattr(
        "slime_plugins.rollout_buffer.generator.ags_generator.rollout.evaluate", _async_return((0.0, True))
    )

    sample = Sample(
        index=0,
        prompt="# Task\n\nFix the bug.",
        metadata={"instance_id": "x__1", "image": "img", "workdir": "/testbed"},
    )
    asyncio.run(runner.generate(sample, {}, evaluation=evaluation))

    assert captured["prompt"] == expected_prompt
    assert ("/testbed/PROBLEM_STATEMENT.md" in sandbox.files) is expects_file


def test_agent_prompt_falls_back_when_dataset_prompt_is_empty(monkeypatch):
    monkeypatch.setenv("SWE_CC_PROMPT", "fallback prompt")
    runner = AGSRolloutRunner.__new__(AGSRolloutRunner)
    runner.config = AGSGeneratorConfig.from_env()

    assert runner._agent_prompt({"instance_id": "x__1", "dataset_prompt": "   "}, "dataset") == "fallback prompt"


@pytest.mark.parametrize("write_problem_statement", [True, False])
def test_prepare_workspace_writes_problem_statement_only_when_asked(write_problem_statement):
    """Under the "dataset" prompt style the prompt already carries the task text,
    so PROBLEM_STATEMENT.md must not be created: it would be an untracked file in
    the repo that only git_diff's exclude pathspec keeps out of the patch."""

    async def run_case():
        sb = FakeSandbox()
        await swe_task.prepare_workspace(
            sb,
            "/testbed",
            {"problem_statement": "Fix the bug."},
            write_problem_statement=write_problem_statement,
        )
        return sb

    sb = asyncio.run(run_case())
    written = "/testbed/PROBLEM_STATEMENT.md" in sb.files
    assert written is write_problem_statement
    if written:
        assert sb.files["/testbed/PROBLEM_STATEMENT.md"] == "Fix the bug."


def test_prepare_workspace_writes_problem_statement_by_default():
    """Callers that predate the flag keep the old behaviour."""

    async def run_case():
        sb = FakeSandbox()
        await swe_task.prepare_workspace(sb, "/testbed", {"problem_statement": "Fix the bug."})
        return sb

    assert "/testbed/PROBLEM_STATEMENT.md" in asyncio.run(run_case()).files


def test_evaluate_can_reuse_agent_sandbox(monkeypatch):
    async def run_case():
        monkeypatch.setattr(
            swe_task,
            "AGSSandbox",
            lambda _image: (_ for _ in ()).throw(AssertionError("must not boot an isolated sandbox")),
        )
        sb = FakeSandbox()

        reward, applied = await swe_task.evaluate(
            sandbox=sb,
            image="unused-image",
            workdir="/workspace/repo",
            diff_text="diff --git a/a.py b/a.py",
            eval_cmd="pytest -q",
            pre_commands=["echo must-not-rerun"],
            eval_bootstrap_cmd="echo bootstrap",
        )

        assert reward == 1.0
        assert applied is True
        commands = [cmd for cmd, _user in sb.exec_log]
        assert "cd /workspace/repo && pytest -q" in commands
        assert "cd /workspace/repo && echo bootstrap" in commands
        assert not any("must-not-rerun" in cmd for cmd in commands)
        assert not any("git apply" in cmd or "patch -p1" in cmd for cmd in commands)

    asyncio.run(run_case())


def test_evaluate_isolated_sandbox_keeps_clean_apply_flow(monkeypatch):
    async def run_case():
        sandboxes = []

        def sandbox_factory(image):
            sb = FakeSandbox(image)
            sandboxes.append(sb)
            return sb

        monkeypatch.setattr(swe_task, "AGSSandbox", sandbox_factory)

        reward, applied = await swe_task.evaluate(
            sandbox=None,
            image="eval-image",
            workdir="/workspace/repo",
            diff_text="diff --git a/a.py b/a.py",
            eval_cmd="pytest -q",
        )

        assert reward == 1.0
        assert applied is True
        assert len(sandboxes) == 1
        commands = [cmd for cmd, _user in sandboxes[0].exec_log]
        assert any("git apply --3way" in cmd for cmd in commands)
        assert "cd /workspace/repo && pytest -q" in commands

    asyncio.run(run_case())


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def decode(self, tokens, skip_special_tokens=False):
        self.calls.append((tokens, skip_special_tokens))
        return "|".join(str(token) for token in tokens)


class _FakeWeaveClient:
    def __init__(self):
        self.created = []
        self.finished = []
        self.wandb_contexts = []

    def create_call(self, op, inputs, **kwargs):
        call = SimpleNamespace(op=op, inputs=inputs, kwargs=kwargs)
        self.created.append(call)
        return call

    def finish_call(self, call, output=None, exception=None, **kwargs):
        self.finished.append((call, output, exception, kwargs))

    def set_wandb_run_context(self, run_id, step=None):
        self.wandb_contexts.append((run_id, step))


def _trace(enable_token2text=False):
    trace = AGSWeaveTrace(_trace_args(), _FakeTokenizer(), enable_token2text=enable_token2text)
    trace.client = _FakeWeaveClient()
    return trace


def _trace_for_agent(agent):
    trace = AGSWeaveTrace(_trace_args(), _FakeTokenizer(), agent=agent)
    trace.client = _FakeWeaveClient()
    return trace


def _trace_args(**overrides):
    data = {
        "use_wandb": False,
        "wandb_mode": None,
        "wandb_team": "team",
        "wandb_project": "project",
        "wandb_run_id": "run-1",
        "wandb_group": "group-1",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_weave_trace_disabled_without_use_wandb(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "weave", types.SimpleNamespace(init=lambda project: (_ for _ in ()).throw(AssertionError))
    )

    trace = AGSWeaveTrace(_trace_args(use_wandb=False, wandb_mode="online"), _FakeTokenizer())

    assert trace.client is None


def test_weave_trace_disabled_for_offline_or_disabled_wandb(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "weave", types.SimpleNamespace(init=lambda project: (_ for _ in ()).throw(AssertionError))
    )

    assert AGSWeaveTrace(_trace_args(use_wandb=True, wandb_mode="disabled"), _FakeTokenizer()).client is None
    assert AGSWeaveTrace(_trace_args(use_wandb=True, wandb_mode="offline"), _FakeTokenizer()).client is None


def test_weave_trace_uses_wandb_project_and_run_context(monkeypatch):
    client = _FakeWeaveClient()
    seen = {}

    def fake_init(project):
        seen["project"] = project
        return client

    monkeypatch.setitem(sys.modules, "weave", types.SimpleNamespace(init=fake_init))

    trace = AGSWeaveTrace(
        _trace_args(use_wandb=True, wandb_mode="online", wandb_team="entity", wandb_project="train-proj"),
        _FakeTokenizer(),
    )

    assert trace.client is client
    assert seen["project"] == "entity/train-proj"
    assert client.wandb_contexts == [("run-1", None)]


def test_weave_trace_keeps_token_ids_without_decoding_by_default():
    trace = _trace()

    payload = trace._sample_payload(_sample())

    assert payload["prompt_token_ids"] == [1]
    assert payload["response_token_ids"] == [2, 3]
    assert "prompt_text" not in payload
    assert "response_text" not in payload
    assert trace.tokenizer.calls == []


def test_weave_trace_decodes_prompt_and_response_when_enabled():
    trace = _trace(enable_token2text=True)

    payload = trace._sample_payload(_sample())

    assert payload["prompt_text"] == "1"
    assert payload["response_text"] == "2|3"
    assert trace.tokenizer.calls == [([1], False), ([2, 3], False)]


def test_weave_trace_pairs_tool_call_and_result(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-07-13T01:02:03Z",
                        "message": {
                            "content": [
                                {"type": "thinking", "thinking": "inspect"},
                                {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"file": "a.py"}},
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": "2026-07-13T01:02:04Z",
                        "message": {
                            "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "source"}]
                        },
                    }
                ),
                json.dumps({"type": "stream_event", "event": {"type": "content_block_delta"}}),
            ]
        ),
        encoding="utf-8",
    )
    trace = _trace()
    parent = SimpleNamespace()

    trace._log_trajectory(parent, str(trajectory))

    assert [call.op for call in trace.client.created] == ["slime.ags.thinking", "slime.ags.tool_call"]
    tool_call = trace.client.created[1]
    tool_finish = next(item for item in trace.client.finished if item[0] is tool_call)
    assert tool_call.inputs["input"] == {"file": "a.py"}
    assert tool_finish[1]["content"] == "source"
    assert tool_finish[3]["ended_at"].isoformat() == "2026-07-13T01:02:04+00:00"
    assert len(list(iter_trajectory_events(trajectory))) == 3


def test_weave_trace_parses_codebuddy_code_stream_json(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "stream_event",
                        "event": {"type": "content_block_delta"},
                        "__timestamp": "2026-07-14T03:24:03.333Z",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": "think-1",
                        "session_id": "sess-cbc",
                        "message": {
                            "content": [{"type": "thinking", "thinking": "inspect cbc"}],
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                        "__timestamp": "2026-07-14T03:24:03.343Z",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": "msg-1",
                        "session_id": "sess-cbc",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "call-1",
                                    "name": "Read",
                                    "input": {"file_path": "/testbed/PROBLEM_STATEMENT.md"},
                                }
                            ],
                            "usage": {"input_tokens": 10, "output_tokens": 2},
                        },
                        "__timestamp": "2026-07-14T03:24:03.352Z",
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "uuid": "result-1",
                        "session_id": "sess-cbc",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "call-1",
                                    "content": [{"type": "text", "text": "source"}],
                                    "is_error": False,
                                }
                            ]
                        },
                        "parent_tool_use_id": "call-1",
                        "__timestamp": "2026-07-14T03:24:03.379Z",
                    }
                ),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": "done",
                        "session_id": "sess-cbc",
                        "__timestamp": "2026-07-14T03:24:04.000Z",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    trace = _trace_for_agent("codebuddy_code")
    parent = SimpleNamespace()

    trace._log_trajectory(parent, str(trajectory))

    assert [call.op for call in trace.client.created] == [
        "slime.ags.thinking",
        "slime.ags.tool_call",
        "slime.ags.result",
    ]
    thinking, tool_call, result_call = trace.client.created
    tool_finish = next(item for item in trace.client.finished if item[0] is tool_call)
    assert thinking.inputs["timestamp"] == "2026-07-14T03:24:03.343Z"
    assert tool_call.inputs["input"] == {"file_path": "/testbed/PROBLEM_STATEMENT.md"}
    assert tool_finish[1]["content"] == [{"type": "text", "text": "source"}]
    assert tool_finish[3]["ended_at"].isoformat() == "2026-07-14T03:24:03.379000+00:00"
    assert next(item for item in trace.client.finished if item[0] is result_call)[1]["result"] == "done"
    assert len(list(iter_trajectory_events(trajectory, agent="codebuddy_code"))) == 4


def test_weave_trace_codex_parser_placeholder_noops(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "type": "assistant",
                "timestamp": "2026-07-13T01:02:03Z",
                "message": {"content": [{"type": "text", "text": "not parsed yet"}]},
            }
        ),
        encoding="utf-8",
    )
    trace = _trace_for_agent("codex")

    trace._log_trajectory(SimpleNamespace(), str(trajectory))

    assert trace.client.created == []
    assert list(iter_trajectory_events(trajectory, agent="codex")) == []


def test_weave_trace_finishes_root_when_child_logging_fails(monkeypatch):
    trace = _trace(enable_token2text=True)
    root = SimpleNamespace()

    def fail_child_logging(parent, trajectory_path):
        raise RuntimeError("trace backend unavailable")

    monkeypatch.setattr(trace, "_log_trajectory", fail_child_logging)
    trace.finish_rollout(root, samples=[_sample()], trajectory_path="trajectory.jsonl")

    assert len(trace.client.finished) == 1
    call, output, exception, _ = trace.client.finished[0]
    assert call is root
    assert output == {"trace_output_error": True}
    assert exception is None


def test_start_rollout_forwards_enable_token2text(monkeypatch):
    captured = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": "Rollout started"}

    def fake_post(url, json, timeout):
        captured.update(json)
        return _Response()

    monkeypatch.setattr("slime_plugins.rollout_buffer.rollout_buffer_example.requests.post", fake_post)
    args = SimpleNamespace(
        rollout_num_process=1,
        num_epoch=1,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_buffer_url="http://127.0.0.1:8889",
        rollout_task_type="ags",
        prompt_data="smoke.jsonl",
        n_samples_per_prompt=1,
        rollout_max_response_len=16,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        hf_checkpoint="model",
        rollout_batch_size=1,
        enable_token2text=True,
        use_wandb=True,
        wandb_mode="online",
        wandb_project="train-proj",
        wandb_team="entity",
        wandb_run_id="run-1",
        wandb_group="group-1",
    )

    start_rollout(args.rollout_buffer_url, args, {})

    assert captured["num_epoch"] == "1"
    assert captured["num_groups_per_epoch"] == "1"
    assert captured["enable_token2text"] is True
    assert captured["use_wandb"] is True
    assert captured["wandb_mode"] == "online"
    assert captured["wandb_project"] == "train-proj"
    assert captured["wandb_team"] == "entity"
    assert captured["wandb_run_id"] == "run-1"
    assert captured["wandb_group"] == "group-1"


def test_start_rollout_uses_one_buffer_epoch_even_when_trainer_num_epoch_is_larger(monkeypatch):
    captured = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": "Rollout started"}

    def fake_post(url, json, timeout):
        captured.update(json)
        return _Response()

    monkeypatch.setattr("slime_plugins.rollout_buffer.rollout_buffer_example.requests.post", fake_post)
    args = SimpleNamespace(
        rollout_num_process=1,
        num_epoch=3,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_buffer_url="http://127.0.0.1:8889",
        rollout_task_type="ags",
        prompt_data="smoke.jsonl",
        n_samples_per_prompt=4,
        rollout_max_response_len=16,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        hf_checkpoint="model",
        rollout_batch_size=8,
    )

    start_rollout(args.rollout_buffer_url, args, {}, num_groups_per_epoch=2)

    assert captured["num_epoch"] == "1"
    assert captured["num_groups_per_epoch"] == "2"


def test_rollout_buffer_ignores_stale_ags_job_items():
    from slime_plugins.rollout_buffer.buffer import RolloutBuffer

    current = output_item_from_samples([_sample()], instance_id="inst-1")
    current["rollout_job_id"] = "job-current"
    stale = output_item_from_samples([_sample()], instance_id="inst-2")
    stale["rollout_job_id"] = "job-stale"

    buffer = RolloutBuffer(group_size=1, rollout_job_id="job-current")

    assert buffer.write(stale) is None
    assert buffer.write(current) == current

    data = buffer.read()["data"]
    assert len(data) == 1
    assert data[0]["instance_id"] == "inst-1"


def _ctx(workdir="/workspace/repo", sid="sess-1", url="http://host:18001"):
    from slime.agent.harness.common import HarnessContext

    return HarnessContext(workdir=workdir, session_id=sid, adapter_url=url)


def _decode_first_b64(cmd: str, path: str) -> dict:
    pattern = rf"printf %s ([^ ]+) \| base64 -d \| tee .*{re.escape(path)}"
    m = re.search(pattern, cmd)
    assert m, cmd
    return json.loads(base64.b64decode(m.group(1)).decode())


def test_codebuddy_code_registry_uses_openai_adapter():
    from slime.agent.adapters import OpenAIAdapter

    harness_cls, adapter_cls = resolve_agent("codebuddy_code")
    assert harness_cls is CodeBuddyCodeHarness
    assert adapter_cls is OpenAIAdapter


def test_codebuddy_code_install_uses_ags_sidecar_binary():
    async def run_case():
        sb = FakeSandbox()
        await CodeBuddyCodeHarness().install_cli(sb)

        cmd = "\n".join(c for c, _ in sb.exec_log)
        assert "/opt/agents/cbc/bin/cbc" in cmd
        assert "ln -sf /opt/agents/cbc/bin/cbc /usr/local/bin/cbc" in cmd
        assert "cbc --version" in cmd

    asyncio.run(run_case())


def test_codebuddy_code_write_config_points_to_adapter():
    async def run_case():
        sb = FakeSandbox()
        await CodeBuddyCodeHarness().write_config(sb, _ctx(sid="sess-cbc", url="http://host:18001"))

        cmd = next(c for c, _ in sb.exec_log if "/root/.codebuddy/models.json" in c)
        models = _decode_first_b64(cmd, "/root/.codebuddy/models.json")
        settings = _decode_first_b64(cmd, "/root/.codebuddy/settings.json")
        assert models["models"][0]["id"] == "slime-actor"
        assert models["models"][0]["apiKey"] == "sess-cbc"
        assert models["models"][0]["url"] == "http://host:18001/v1/chat/completions"
        assert models["models"][0]["supportsToolCall"] is True
        # No maxOutputTokens: the harness sets no per-turn cap, so the CLI applies
        # its own default. The adapter still bounds a turn via max_new_tokens.
        assert "maxOutputTokens" not in models["models"][0]
        assert settings["alwaysThinkingEnabled"] is True

    asyncio.run(run_case())


def test_codebuddy_code_launch_command_and_env():
    async def run_case():
        sb = FakeSandbox()
        rc = await CodeBuddyCodeHarness().launch_and_wait(
            sb,
            _ctx(sid="sess-cbc", url="http://host:18001"),
            prompt="solve it",
            time_budget_sec=0,
        )

        assert rc != 0  # time_budget=0 avoids waiting; launch still happens.
        body = next(v for k, v in sb.files.items() if k.endswith("run.sh"))
        assert "cbc --model slime-actor --verbose --output-format stream-json --include-partial-messages" in body
        assert "-y" in body and "solve it" in body
        # Tool restriction uses --disallowedTools, which the CLI enforces; --tools
        # does not restrict the surface, so it is never passed.
        assert "--disallowedTools WebSearch WebFetch" in body
        assert "--tools" not in body
        assert "codebuddy_sessions" in body
        # No turn cap from the harness: the run is bounded by
        # SWE_AGENT_TIME_BUDGET_SEC, and a caller who wants one passes it in
        # SLIME_AGENT_CBC_EXTRA_ARGS.
        assert "--max-turns" not in body
        # --disallowedTools is variadic, so the non-variadic tail and the prompt
        # must follow it, or it would swallow them.
        assert body.index("--disallowedTools") < body.index("-y") < body.index("solve it")

        launch_cmd = next(c for c, _ in sb.exec_log if "setsid" in c)
        assert "OPENAI_API_KEY=sess-cbc" in launch_cmd
        assert "OPENAI_BASE_URL=http://host:18001/v1" in launch_cmd
        assert "CBC_API_KEY=sess-cbc" in launch_cmd
        assert "CBC_BASE_URL=http://host:18001/v1/chat/completions" in launch_cmd
        assert "IS_SANDBOX=1" in launch_cmd
        assert any("kill -TERM" in cmd for cmd, _user in sb.exec_log)
        assert any("kill -KILL" in cmd for cmd, _user in sb.exec_log)

    asyncio.run(run_case())


def test_extra_args_come_after_defaults_so_they_win(monkeypatch):
    """EXTRA_ARGS is the only flag knob, so it must be able to beat a default.

    Both CLIs take the LAST occurrence of a repeated flag (verified against the
    real binaries: `--max-turns 99 --max-turns 1` stops after 1 turn on each), so
    "wins" here means "appears later in the command".
    """

    async def run_case():
        monkeypatch.setenv("SLIME_AGENT_CBC_EXTRA_ARGS", "--disallowedTools ImageGen --max-turns 7")
        sb = FakeSandbox()
        await CodeBuddyCodeHarness().launch_and_wait(
            sb, _ctx(sid="s", url="http://host:18001"), prompt="go", time_budget_sec=0
        )
        body = next(v for k, v in sb.files.items() if k.endswith("run.sh"))
        assert "--max-turns 7" in body
        assert (
            body.index("--disallowedTools WebSearch WebFetch")
            < body.index("--disallowedTools ImageGen")
            < body.index("go")
        )

        monkeypatch.setenv("SLIME_AGENT_CC_EXTRA_ARGS", "--disallowedTools ImageGen --max-turns 7")
        sb2 = FakeSandbox()
        await AGSSidecarClaudeCodeHarness().launch_and_wait(
            sb2, _ctx(sid="s", url="http://host:18001"), prompt="go", time_budget_sec=0
        )
        body2 = next(v for k, v in sb2.files.items() if k.endswith("run.sh"))
        assert "--max-turns 7" in body2
        assert body2.index("--disallowedTools WebSearch WebFetch") < body2.index("--disallowedTools ImageGen")

    asyncio.run(run_case())


def test_web_tools_denied_by_default_on_both_harnesses():
    """Web access makes a rollout unreproducible and can leak the graded fix.

    A 2026-07-29 eval matrix had this denied for CodeBuddy but not for Claude
    Code, which then used WebSearch/WebFetch on 1-2% of instances -- a difference
    of the same order as the training effect being measured.
    """

    async def run_case():
        for harness in (AGSSidecarClaudeCodeHarness(), CodeBuddyCodeHarness()):
            sb = FakeSandbox()
            await harness.launch_and_wait(sb, _ctx(sid="s", url="http://host:18001"), prompt="go", time_budget_sec=0)
            body = next(v for k, v in sb.files.items() if k.endswith("run.sh"))
            assert "--disallowedTools WebSearch WebFetch" in body, harness.name

    asyncio.run(run_case())


def test_claude_code_extra_envs_override_static_env(monkeypatch):
    """EXTRA_ENVS is merged after static_env, so it can override any of it."""

    async def run_case():
        sb = FakeSandbox()
        await AGSSidecarClaudeCodeHarness().launch_and_wait(
            sb, _ctx(sid="s", url="http://host:18001"), prompt="go", time_budget_sec=0
        )
        launch_cmd = next(c for c, _ in sb.exec_log if "setsid" in c)
        assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1" in launch_cmd
        # The harness sets no per-turn output cap of its own.
        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in launch_cmd

        monkeypatch.setenv(
            "SLIME_AGENT_CC_EXTRA_ENVS",
            '{"CLAUDE_CODE_MAX_OUTPUT_TOKENS":"4096","CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC":"0"}',
        )
        sb2 = FakeSandbox()
        await AGSSidecarClaudeCodeHarness().launch_and_wait(
            sb2, _ctx(sid="s", url="http://host:18001"), prompt="go", time_budget_sec=0
        )
        launch_cmd2 = next(c for c, _ in sb2.exec_log if "setsid" in c)
        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096" in launch_cmd2
        assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=0" in launch_cmd2

    asyncio.run(run_case())


def test_ags_timeout_fails_closed_if_agent_process_group_cannot_be_stopped():
    class StopFailureSandbox(FakeSandbox):
        async def exec(self, cmd, **kwargs):
            if "kill -TERM" in cmd:
                raise RuntimeError("cannot stop agent")
            return await super().exec(cmd, **kwargs)

    async def run_case():
        with pytest.raises(RuntimeError, match="cannot stop agent"):
            await run_root_command(
                StopFailureSandbox(),
                workdir="/workspace/repo",
                start_cmd="claude -p solve",
                env={},
                time_budget_sec=0,
            )

    asyncio.run(run_case())
