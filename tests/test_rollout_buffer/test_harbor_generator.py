from __future__ import annotations

import json

import pytest

from slime.utils.types import Sample
from slime_plugins.rollout_buffer.generator.ags_generator.serialization import (
    output_item_from_samples,
    samples_from_payload,
)
from slime_plugins.rollout_buffer.generator.harbor_generator import (
    _specs_from_input_file,
    get_group_data_meta_info,
    is_valid_group,
    samples_from_harbor_result,
    transform_group,
)

NUM_GPUS = 0


def _base_sample() -> Sample:
    return Sample(index=7, group_index=2, rollout_id=7, prompt="task", metadata={"base": "kept"})


def _trial_result(*, rollout_details=None, rewards=None, exception_info=None):
    return {
        "task_name": "task-a",
        "trial_name": "trial-a",
        "trial_uri": "file:///tmp/harbor/trial-a",
        "agent_info": {"name": "terminus-2"},
        "agent_result": {"rollout_details": rollout_details},
        "verifier_result": {"rewards": rewards if rewards is not None else {"score": 1.0}},
        "exception_info": exception_info,
    }


def test_harbor_rollout_details_convert_to_trainable_samples():
    result = _trial_result(
        rollout_details=[
            {
                "prompt_token_ids": [[10, 11], [10, 11, 20, 21, 30]],
                "completion_token_ids": [[20, 21], [40]],
                "logprobs": [[-0.2, -0.3], [-0.4]],
            }
        ],
        rewards={"score": 0.75},
    )

    samples = samples_from_harbor_result(result, base_sample=_base_sample())

    assert len(samples) == 2
    assert samples[0].tokens == [10, 11, 20, 21]
    assert samples[0].response_length == 2
    assert samples[0].loss_mask == [1, 1]
    assert samples[0].rollout_log_probs == [-0.2, -0.3]
    assert samples[0].reward == 0.75
    assert samples[0].status == Sample.Status.COMPLETED
    assert samples[0].rollout_id == 7
    assert samples[0].metadata["harbor_reward_key"] == "score"
    assert samples[0].metadata["base"] == "kept"
    assert samples[1].tokens == [10, 11, 20, 21, 30, 40]
    assert samples[1].response_length == 1
    assert samples[1].rollout_id == 7


def test_harbor_reward_key_selects_dict_reward():
    result = _trial_result(
        rollout_details=[{"prompt_token_ids": [[1]], "completion_token_ids": [[2]], "logprobs": [[-0.1]]}],
        rewards={"aux": 0.1, "main": 0.9},
    )

    (sample,) = samples_from_harbor_result(result, reward_key="main", base_sample=_base_sample())

    assert sample.reward == 0.9
    assert sample.metadata["harbor_reward_key"] == "main"
    assert sample.metadata["harbor_rewards"] == {"aux": 0.1, "main": 0.9}


def test_harbor_missing_logprobs_keeps_tokens_without_rollout_log_probs():
    result = _trial_result(rollout_details=[{"prompt_token_ids": [[1, 2]], "completion_token_ids": [[3, 4]]}])

    (sample,) = samples_from_harbor_result(result, base_sample=_base_sample())

    assert sample.tokens == [1, 2, 3, 4]
    assert sample.response_length == 2
    assert sample.loss_mask == [1, 1]
    assert sample.rollout_log_probs is None
    assert sample.metadata["harbor_has_rollout_log_probs"] is False


def test_harbor_missing_token_rollout_details_aborts_by_default():
    result = _trial_result(rollout_details=None, rewards={"score": 1.0})

    (sample,) = samples_from_harbor_result(result, base_sample=_base_sample())

    assert sample.status == Sample.Status.ABORTED
    assert sample.remove_sample is True
    assert sample.reward == 0.0
    assert sample.tokens == [0, 0]
    assert sample.response_length == 1
    assert sample.loss_mask == [0]
    assert sample.rollout_log_probs == [0.0]
    assert sample.metadata["abort_reason"] == "missing_token_rollout_details"
    assert sample.metadata["harbor_raw_reward"] == 1.0


def test_harbor_exception_result_marks_completed_sample_failed():
    result = _trial_result(
        rollout_details=[{"prompt_token_ids": [[1]], "completion_token_ids": [[2]], "logprobs": [[-0.1]]}],
        exception_info={"exception_type": "AgentError", "exception_message": "boom"},
    )

    (sample,) = samples_from_harbor_result(result, base_sample=_base_sample())

    # The pure converter preserves token trainability; runtime metadata attachment
    # marks completed samples failed after real trial execution. Exception metadata
    # is still carried by the converter.
    assert sample.status == Sample.Status.COMPLETED
    assert sample.metadata["harbor_exception_type"] == "AgentError"
    assert sample.metadata["harbor_exception_message"] == "boom"


def test_harbor_output_payload_round_trips_and_group_hooks_accept_it():
    result = _trial_result(
        rollout_details=[{"prompt_token_ids": [[1]], "completion_token_ids": [[2]], "logprobs": [[-0.1]]}]
    )
    samples = samples_from_harbor_result(result, base_sample=_base_sample())
    item = output_item_from_samples(samples, instance_id="task-a")

    restored = samples_from_payload(item)

    assert len(restored) == 1
    assert restored[0].tokens == [1, 2]
    group = ("task-a", [item])
    assert transform_group(group) is group
    assert is_valid_group(group, min_valid_group_size=1)
    meta = get_group_data_meta_info({"task-a": [item]})
    assert meta["total_samples"] == 1
    assert meta["avg_reward"] == 1.0
    assert meta["artifact_counts"]["result"] == 1
    assert meta["artifact_counts"]["trial_dir"] == 1


def test_harbor_group_hook_rejects_incomplete_samples():
    incomplete = Sample(index=1, reward=1.0, status=Sample.Status.COMPLETED)
    item = output_item_from_samples([incomplete], instance_id="task-a")

    assert not is_valid_group(("task-a", [item]), min_valid_group_size=1)


def _inline_task_row():
    return {
        "prompt": "# Task\nFix the toy issue.",
        "label": "toy__repo-1",
        "metadata": {
            "instance_id": "toy__repo-1",
            "source": "toy-dataset",
            "harbor_task": {
                "format": "harbor_task_inline_v1",
                "name": "toy__repo-1",
                "files": {
                    "instruction.md": "# Task\nFix the toy issue.",
                    "task.toml": "[verifier]\ntimeout_sec = 10\n",
                    "environment/Dockerfile": "FROM python:3.11-slim\n",
                    "tests/test.sh": "#!/usr/bin/env bash\nset -euo pipefail\necho 1 > /verifier/reward.txt\n",
                    "tests/config.json": "{}\n",
                    "solution/solve.sh": "#!/usr/bin/env bash\ntrue\n",
                },
            },
        },
    }


def test_harbor_inline_prompt_data_materializes_task(tmp_path):
    input_file = tmp_path / "prompt_data.jsonl"
    input_file.write_text(json.dumps(_inline_task_row()) + "\n")
    materialized_dir = tmp_path / "materialized"

    specs = _specs_from_input_file(
        input_file,
        {
            "input_key": "prompt",
            "metadata_key": "metadata",
            "harbor_materialized_tasks_dir": str(materialized_dir),
        },
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.instance_id == "toy__repo-1"
    assert spec.source == "toy-dataset"
    assert spec.prompt == "# Task\nFix the toy issue."
    assert spec.path == materialized_dir / "toy__repo-1"
    for rel_path in [
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "tests/test.sh",
        "tests/config.json",
        "solution/solve.sh",
    ]:
        assert (spec.path / rel_path).is_file()
    assert spec.path.joinpath("tests/test.sh").stat().st_mode & 0o111


@pytest.mark.parametrize("bad_path", ["../escape.txt", "/tmp/escape.txt", "tests/../escape.txt"])
def test_harbor_inline_prompt_data_rejects_unsafe_file_paths(tmp_path, bad_path):
    row = _inline_task_row()
    files = row["metadata"]["harbor_task"]["files"]
    files[bad_path] = "bad"
    input_file = tmp_path / "prompt_data.jsonl"
    input_file.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="Unsafe inline Harbor task file path"):
        _specs_from_input_file(
            input_file,
            {
                "input_key": "prompt",
                "metadata_key": "metadata",
                "harbor_materialized_tasks_dir": str(tmp_path / "materialized"),
            },
        )


def test_harbor_inline_prompt_data_requires_minimal_task_files(tmp_path):
    row = _inline_task_row()
    del row["metadata"]["harbor_task"]["files"]["task.toml"]
    input_file = tmp_path / "prompt_data.jsonl"
    input_file.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="missing required files"):
        _specs_from_input_file(
            input_file,
            {
                "input_key": "prompt",
                "metadata_key": "metadata",
                "harbor_materialized_tasks_dir": str(tmp_path / "materialized"),
            },
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
