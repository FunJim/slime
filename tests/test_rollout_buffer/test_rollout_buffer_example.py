from types import SimpleNamespace

from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.types import Sample
from slime_plugins.rollout_buffer.rollout_buffer_example import _select_complete_rollout_groups


def _record(instance_id, timestamp):
    return {"instance_id": instance_id, "timestamp": timestamp}


def test_select_complete_rollout_groups_skips_partial_prompt_groups():
    args = SimpleNamespace(n_samples_per_prompt=8)
    results = [_record("complete", index) for index in range(8)]
    results += [_record("partial", index) for index in range(2)]

    selected, incomplete = _select_complete_rollout_groups(args, results, need_length=1)

    assert len(selected) == 1
    assert len(selected[0]) == 8
    assert {record["instance_id"] for record in selected[0]} == {"complete"}
    assert incomplete == {"partial": 2}


def test_buffer_accepts_flattened_agent_segments_above_requested_attempt_count():
    data_source = RolloutDataSourceWithBuffer.__new__(RolloutDataSourceWithBuffer)
    data_source.args = SimpleNamespace(n_samples_per_prompt=8)
    data_source.buffer = []

    segments = [Sample(index=index) for index in range(9)]

    data_source.add_samples([segments])

    assert data_source.buffer == [segments]
