"""Regression tests for Harbor-to-slime prompt conversion."""

from __future__ import annotations

import ast
import base64
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def converter_module():
    path = Path(__file__).parents[1] / "tools" / "harbor_task_to_slime_prompt_data.py"
    spec = importlib.util.spec_from_file_location("harbor_task_to_slime_prompt_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def harbor_task(tmp_path: Path) -> Path:
    task = tmp_path / "sample-task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("Fix the bug.")
    (task / "task.toml").write_text("[task]\nname = 'sample-task'\n")
    (task / "environment" / "Dockerfile").write_text("FROM registry.example/swe:sample\nWORKDIR /testbed\n")
    (task / "tests" / "config.json").write_text(
        json.dumps(
            {
                "instance_id": "sample__1",
                "repo": "example/sample",
                "base_commit": "deadbeef",
                "problem_statement": "Fix the bug.",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
            }
        )
    )
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\npython -m pip install -e .[test] --verbose\npytest tests/test_sample.py || true\n"
    )
    return task


def _embedded_test_script(eval_cmd: str) -> str:
    payloads_text = eval_cmd.split("payloads = [", 1)[1].split("]", 1)[0]
    payloads = ast.literal_eval(f"[{payloads_text}]")
    return gzip.decompress(base64.b64decode(payloads[1])).decode("utf-8")


def test_converter_preserves_image_head_by_default(converter_module, harbor_task: Path):
    row = converter_module.task_to_row(
        harbor_task,
        dataset_root=harbor_task.parent,
        source="test",
        input_key="prompt",
        prompt_alias_key="",
        label_key="label",
        metadata_key="metadata",
        prompt_source="problem_statement",
        image_override=None,
        default_workdir="/testbed",
        include_eval_cmd=True,
        include_inline_files=False,
        inline_files=(),
        provenance_root=False,
    )

    metadata = row["metadata"]
    assert "pre_commands" not in metadata
    assert _embedded_test_script(metadata["eval_cmd"]) == (harbor_task / "tests" / "test.sh").read_text()


def test_reset_to_base_commit_option_is_removed(converter_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harbor_task_to_slime_prompt_data.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--reset-to-base-commit",
        ],
    )

    with pytest.raises(SystemExit):
        converter_module.parse_args()


def test_gzip_payload_has_zero_mtime_and_is_reproducible(converter_module):
    first = base64.b64decode(converter_module._gzip_base64("same content"))
    second = base64.b64decode(converter_module._gzip_base64("same content"))

    assert first == second
    assert first[4:8] == b"\0\0\0\0"
    assert gzip.decompress(first) == b"same content"
