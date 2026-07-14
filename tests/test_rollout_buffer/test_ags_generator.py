from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
import types
from types import SimpleNamespace

from tests.test_agent._fakes import FakeSandbox

from slime.utils.types import Sample
from slime_plugins.rollout_buffer.generator.ags_generator.entry import (
    get_group_data_meta_info,
    is_valid_group,
    transform_group,
)
from slime_plugins.rollout_buffer.generator.ags_generator.harnesses import CodeBuddyCodeHarness, resolve_agent
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


def test_sampling_params_use_sglang_generate_names():
    assert normalize_sampling_params({"max_tokens": 128, "temperature": 1.0}) == {
        "max_new_tokens": 128,
        "temperature": 1.0,
    }


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

    assert captured["enable_token2text"] is True
    assert captured["use_wandb"] is True
    assert captured["wandb_mode"] == "online"
    assert captured["wandb_project"] == "train-proj"
    assert captured["wandb_team"] == "entity"
    assert captured["wandb_run_id"] == "run-1"
    assert captured["wandb_group"] == "group-1"


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


def test_codebuddy_code_write_config_points_to_adapter(monkeypatch):
    async def run_case():
        monkeypatch.setenv("SLIME_AGENT_CBC_MAX_OUTPUT_TOKENS", "8192")
        monkeypatch.setenv("SLIME_AGENT_CBC_THINKING_ENABLED", "false")
        sb = FakeSandbox()
        await CodeBuddyCodeHarness().write_config(sb, _ctx(sid="sess-cbc", url="http://host:18001"))

        cmd = next(c for c, _ in sb.exec_log if "/root/.codebuddy/models.json" in c)
        models = _decode_first_b64(cmd, "/root/.codebuddy/models.json")
        settings = _decode_first_b64(cmd, "/root/.codebuddy/settings.json")
        assert models["models"][0]["id"] == "slime-actor"
        assert models["models"][0]["apiKey"] == "sess-cbc"
        assert models["models"][0]["url"] == "http://host:18001/v1/chat/completions"
        assert models["models"][0]["maxOutputTokens"] == 8192
        assert models["models"][0]["supportsToolCall"] is True
        assert settings["alwaysThinkingEnabled"] is False

    asyncio.run(run_case())


def test_codebuddy_code_launch_command_and_env(monkeypatch):
    async def run_case():
        monkeypatch.setenv("SLIME_AGENT_CBC_MAX_TURNS", "7")
        monkeypatch.setenv("SLIME_AGENT_CBC_THINKING_ENABLED", "false")
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
        assert "--max-turns 7" in body
        assert "-y" in body and "solve it" in body
        assert "--effort none" in body
        assert "--tools Bash,Read,Write,Edit,Glob,Grep,TaskCreate,TaskUpdate,TaskGet,TaskList,Agent" in body
        assert "codebuddy_sessions" in body

        launch_cmd = next(c for c, _ in sb.exec_log if "setsid" in c)
        assert "OPENAI_API_KEY=sess-cbc" in launch_cmd
        assert "OPENAI_BASE_URL=http://host:18001/v1" in launch_cmd
        assert "CBC_API_KEY=sess-cbc" in launch_cmd
        assert "CBC_BASE_URL=http://host:18001/v1/chat/completions" in launch_cmd
        assert "IS_SANDBOX=1" in launch_cmd

    asyncio.run(run_case())
