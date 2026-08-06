"""Unit tests for AGSPromptSource's seek-to-absolute-group-index behaviour.

The generator has no checkpoint of its own: the trainer tells it where to start
by absolute group index (``rollout_start_group = rollout_id *
rollout_batch_size``), and AGSPromptSource seeks ``RolloutDataSource`` there.
So the property under test is that seeking to group g yields exactly the
prompts an uninterrupted run would have produced at group g -- which is what
makes a resumed run replay the same per-step prompt set. The interesting case
is a seek past the end of the dataset, where the epoch (and with shuffle on,
the permutation) has to advance too.

``source`` is loaded from its file through a private synthetic package rather
than through ``slime_plugins...``: that package's ``__init__`` imports the whole
entry module (and ``openai``), which the CPU CI jobs do not install. See the
sibling ``test_ags_empty_patch_guard.py`` for the same pattern.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# slime.rollout.data_source -> slime.utils.processing_utils -> transformers, a
# heavy dep absent from the CPU-only CI env. No real tokenizer is ever used
# here (load_tokenizer/load_processor are patched, and Dataset skips tokenizing
# when apply_chat_template is off and max_length is None), so stub it out.
if "transformers" not in sys.modules:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

AGS_DIR = REPO_ROOT / "slime_plugins" / "rollout_buffer" / "generator" / "ags_generator"
_PRIVATE_PACKAGE = "_ags_source_under_test"

if _PRIVATE_PACKAGE not in sys.modules:
    _package = types.ModuleType(_PRIVATE_PACKAGE)
    _package.__path__ = [str(AGS_DIR)]
    sys.modules[_PRIVATE_PACKAGE] = _package

source_mod = importlib.import_module(".source", package=_PRIVATE_PACKAGE)
AGSPromptSource = source_mod.AGSPromptSource

NUM_GPUS = 0

DATASET_LEN = 5


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    """RolloutDataSource always loads a tokenizer/processor; neither is used."""
    data_source_mod = sys.modules["slime.rollout.data_source"]
    monkeypatch.setattr(data_source_mod, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(data_source_mod, "load_processor", lambda *a, **k: None)


@pytest.fixture
def prompt_data(tmp_path) -> str:
    path = tmp_path / "prompts.jsonl"
    lines = [
        json.dumps({"prompt": f"p{i}", "label": f"l{i}", "metadata": {"instance_id": f"i{i}"}})
        for i in range(DATASET_LEN)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _args(prompt_data: str, *, start_group: int, shuffle: bool) -> Namespace:
    return Namespace(
        hf_checkpoint="unused",
        prompt_data=prompt_data,
        input_key="prompt",
        label_key="label",
        metadata_key="metadata",
        tool_key=None,
        multimodal_keys=None,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        rollout_global_dataset=True,
        rollout_shuffle=shuffle,
        rollout_seed=42,
        rollout_max_prompt_len=None,
        dump_details=None,
        rollout_start_group=start_group,
        n_samples_per_prompt=2,
    )


def _prompts_from_seek(prompt_data: str, *, start_group: int, num_groups: int, shuffle: bool) -> list[str]:
    """Prompts a generator started at ``start_group`` produces for its first request."""
    source = AGSPromptSource(_args(prompt_data, start_group=start_group, shuffle=shuffle))
    return [group[0].prompt for group in source.get_groups(num_groups)]


def _prompts_from_start(prompt_data: str, *, num_groups: int, shuffle: bool) -> list[str]:
    """Prompts an uninterrupted run produces, drawn one group at a time from 0."""
    source = AGSPromptSource(_args(prompt_data, start_group=0, shuffle=shuffle))
    return [source.get_groups(1)[0][0].prompt for _ in range(num_groups)]


@pytest.mark.parametrize("shuffle", [False, True])
def test_seek_within_first_epoch_matches_uninterrupted_run(prompt_data, shuffle):
    uninterrupted = _prompts_from_start(prompt_data, num_groups=DATASET_LEN, shuffle=shuffle)
    resumed = _prompts_from_seek(prompt_data, start_group=2, num_groups=2, shuffle=shuffle)
    assert resumed == uninterrupted[2:4]


@pytest.mark.parametrize("shuffle", [False, True])
def test_seek_past_dataset_end_matches_uninterrupted_run(prompt_data, shuffle):
    """The regression: a seek that wraps must land in the right epoch.

    Group DATASET_LEN + 1 is the second group of epoch 1. Restoring only
    ``sample_offset`` left the source on epoch 0's permutation, so a resumed run
    diverged from the original once training passed one pass over the data.
    """
    start_group = DATASET_LEN + 1
    uninterrupted = _prompts_from_start(prompt_data, num_groups=start_group + 2, shuffle=shuffle)
    resumed = _prompts_from_seek(prompt_data, start_group=start_group, num_groups=2, shuffle=shuffle)
    assert resumed == uninterrupted[start_group : start_group + 2]


def test_seek_past_dataset_end_advances_epoch(prompt_data):
    source = AGSPromptSource(_args(prompt_data, start_group=2 * DATASET_LEN + 3, shuffle=True))
    assert source.data_source.epoch_id == 2
    assert source.data_source.sample_offset == 3
    assert source.data_source.dataset.epoch_id == 2


def test_shuffled_epochs_differ(prompt_data):
    """Guards the test above: epoch 1's order must actually differ from epoch 0's."""
    epoch0 = _prompts_from_seek(prompt_data, start_group=0, num_groups=DATASET_LEN, shuffle=True)
    epoch1 = _prompts_from_seek(prompt_data, start_group=DATASET_LEN, num_groups=DATASET_LEN, shuffle=True)
    assert sorted(epoch0) == sorted(epoch1)
    assert epoch0 != epoch1


def test_seek_is_noop_at_group_zero(prompt_data):
    source = AGSPromptSource(_args(prompt_data, start_group=0, shuffle=True))
    assert source.data_source.epoch_id == 0
    assert source.data_source.sample_offset == 0
    assert source.data_source.sample_group_index == 0
    assert source.data_source.sample_index == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
