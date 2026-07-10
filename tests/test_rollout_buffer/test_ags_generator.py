from __future__ import annotations

import asyncio
import base64
import json
import re

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
        assert "cbc --model slime-actor --output-format json" in body
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
