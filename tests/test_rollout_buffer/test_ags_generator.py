from __future__ import annotations

from slime.utils.types import Sample
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
    assert meta["artifact_counts"] == {"trajectory": 1, "patch": 1, "rollout_dump": 1}


def test_sampling_params_use_sglang_generate_names():
    assert normalize_sampling_params({"max_tokens": 128, "temperature": 1.0}) == {
        "max_new_tokens": 128,
        "temperature": 1.0,
    }
