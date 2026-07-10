from __future__ import annotations

import pytest

from slime.utils.types import Sample
from slime_plugins.rollout_buffer.generator import harbor_ags_generator
from slime_plugins.rollout_buffer.generator.ags_generator.entry import (
    get_group_data_meta_info,
    is_valid_group,
    transform_group,
)
from slime_plugins.rollout_buffer.generator.ags_generator.sampling import normalize_sampling_params
from slime_plugins.rollout_buffer.generator.ags_generator.serialization import (
    output_item_from_samples,
    samples_from_payload,
)

NUM_GPUS = 0


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


def test_harbor_ags_normalizes_harbor_semantic_prompt_row():
    sample = Sample(
        index=7,
        prompt="Fix the failing tests.",
        label="label-id",
        metadata={
            "source": "swebench-verified",
            "harbor": {
                "task_name": "astropy__astropy-12907",
                "docker_image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
                "docker_workdir": "/testbed",
            },
        },
    )

    harbor_ags_generator.normalize_harbor_ags_sample(sample)

    assert sample.metadata["instance_id"] == "astropy__astropy-12907"
    assert sample.metadata["image"] == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    assert sample.metadata["workdir"] == "/testbed"
    assert sample.metadata["problem_statement"] == "Fix the failing tests."
    assert sample.metadata["harbor_ags"] == {
        "input_format": "harbor_semantic_ags_v1",
        "task_type": "harbor_ags",
        "uses_ags_rollout": True,
        "task_name": "astropy__astropy-12907",
    }


def test_harbor_ags_group_hooks_require_complete_training_samples():
    item = output_item_from_samples([_sample()], instance_id="inst-1")
    group = ("inst-1", [item])

    assert harbor_ags_generator.is_valid_group(group, min_valid_group_size=1)

    missing_loss_mask = _sample()
    missing_loss_mask.loss_mask = None
    invalid_item = output_item_from_samples([missing_loss_mask], instance_id="inst-2")
    assert not harbor_ags_generator.is_valid_group(("inst-2", [invalid_item]), min_valid_group_size=1)


def test_harbor_ags_run_rollout_uses_shared_ags_entry(monkeypatch):
    captured = {}

    def fake_run_rollout_for_task_type(payload, *, task_type, source_cls):
        captured["payload"] = payload
        captured["task_type"] = task_type
        captured["source_cls"] = source_cls
        return "ok"

    monkeypatch.setattr(harbor_ags_generator, "run_rollout_for_task_type", fake_run_rollout_for_task_type)

    assert harbor_ags_generator.run_rollout({}) == "ok"

    assert captured["task_type"] == "harbor_ags"
    assert captured["source_cls"] is harbor_ags_generator.HarborAGSPromptSource
    assert captured["payload"]["input_key"] == "prompt"
    assert captured["payload"]["label_key"] == "label"
    assert captured["payload"]["metadata_key"] == "metadata"


def test_harbor_ags_prompt_source_sets_shared_rollout_id_per_prompt(monkeypatch):
    base_groups = [
        [
            Sample(index=10, group_index=4, label="inst", prompt="p", metadata={"image": "img", "workdir": "/w"}),
            Sample(index=11, group_index=4, label="inst", prompt="p", metadata={"image": "img", "workdir": "/w"}),
        ]
    ]

    monkeypatch.setattr(
        harbor_ags_generator.AGSPromptSource,
        "get_groups",
        lambda self, num_groups: base_groups,
    )

    groups = harbor_ags_generator.HarborAGSPromptSource.__new__(harbor_ags_generator.HarborAGSPromptSource).get_groups(
        1
    )

    assert [sample.rollout_id for sample in groups[0]] == [4, 4]
    assert groups[0][0].metadata["instance_id"] == "inst"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
