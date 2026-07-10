#!/usr/bin/env python3
"""Convert Harbor SWE-style tasks to semantic slime/AGS prompt-data JSONL.

The default output is not a mirror of Harbor task files.  It extracts the small
set of fields that slime's AGS rollout-buffer generator already consumes:

  - prompt text
  - metadata.instance_id
  - metadata.image
  - metadata.workdir
  - metadata.problem_statement
  - metadata.pre_commands
  - metadata.eval_cmd

The generated rows are still ordinary slime JSONL prompt data: use --input-key
prompt, --label-key label, and --metadata-key metadata.  The eval command is
built from Harbor's tests/test.sh plus tests/config.json so the row can be used
by ags_generator without a harbor_task_path.

Example:
    python tools/harbor_task_to_slime_prompt_data.py \
        --input /path/to/harbor-datasets/datasets/swebench-verified \
        --output ./local/swebench-verified-harbor-dataset/prompt_data.jsonl \
        --task astropy__astropy-12907 \
        --source swebench-verified \
        --pretty-output ./local/swebench-verified-harbor-dataset/example_pretty.json \
        --schema-output ./local/swebench-verified-harbor-dataset/schema.json
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shlex

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path
from typing import Any

INLINE_TASK_FORMAT = "harbor_task_inline_v1"
DEFAULT_INLINE_FILES = (
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "tests/test.sh",
    "tests/config.json",
    "solution/solve.sh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A Harbor task directory, or a dataset directory containing Harbor task subdirectories.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name or glob to include. Repeatable. If omitted, include all tasks under --input.",
    )
    parser.add_argument(
        "--exclude-task",
        action="append",
        default=[],
        help="Task name or glob to exclude. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of tasks to write after filtering.")
    parser.add_argument("--offset", type=int, default=0, help="Number of filtered tasks to skip before writing.")
    parser.add_argument(
        "--source",
        default=None,
        help="Dataset/source name stored in metadata. Defaults to --input basename for dataset inputs.",
    )
    parser.add_argument(
        "--input-key",
        default="prompt",
        help="Primary slime prompt key to write. Defaults to prompt.",
    )
    parser.add_argument(
        "--prompt-alias-key",
        default="",
        help="Optional prompt alias key. Use '' to disable. Disabled by default.",
    )
    parser.add_argument("--label-key", default="label", help="Label key to write. Use '' to disable.")
    parser.add_argument("--metadata-key", default="metadata", help="Metadata key to write.")
    parser.add_argument(
        "--prompt-source",
        choices=("problem_statement", "instruction"),
        default="problem_statement",
        help="Which extracted text to put in the primary prompt field.",
    )
    parser.add_argument(
        "--default-workdir",
        default="/testbed",
        help="Workdir fallback when environment/Dockerfile has no WORKDIR.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Override image for all rows. By default it is extracted from the active Dockerfile FROM line.",
    )
    parser.add_argument(
        "--no-pre-commands",
        action="store_true",
        help="Do not write metadata.pre_commands to reset the repo to base_commit.",
    )
    parser.add_argument(
        "--no-eval-cmd",
        action="store_true",
        help="Do not derive metadata.eval_cmd from tests/test.sh and tests/config.json.",
    )
    parser.add_argument(
        "--include-inline-files",
        action="store_true",
        help="Also include metadata.harbor_task.files for Harbor materialization/debugging. Disabled by default.",
    )
    parser.add_argument(
        "--inline-file",
        action="append",
        default=None,
        help="Relative Harbor task file to include when --include-inline-files is set. Repeatable.",
    )
    parser.add_argument(
        "--provenance-root",
        action="store_true",
        help="Record the source dataset root and task name in metadata.source_provenance.",
    )
    parser.add_argument(
        "--pretty-output", type=Path, default=None, help="Optional pretty JSON file for the first row."
    )
    parser.add_argument("--schema-output", type=Path, default=None, help="Optional JSON schema output path.")
    return parser.parse_args()


def find_task_dirs(input_path: Path, include_patterns: list[str], exclude_patterns: list[str]) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if _is_harbor_task_dir(input_path):
        tasks = [input_path]
    else:
        tasks = [path for path in sorted(input_path.iterdir()) if path.is_dir() and _is_harbor_task_dir(path)]

    if include_patterns:
        tasks = [path for path in tasks if _matches_any(path.name, include_patterns)]
    if exclude_patterns:
        tasks = [path for path in tasks if not _matches_any(path.name, exclude_patterns)]
    return tasks


def task_to_row(
    task_dir: Path,
    *,
    dataset_root: Path,
    source: str | None,
    input_key: str,
    prompt_alias_key: str,
    label_key: str,
    metadata_key: str,
    prompt_source: str,
    image_override: str | None,
    default_workdir: str,
    include_pre_commands: bool,
    include_eval_cmd: bool,
    include_inline_files: bool,
    inline_files: tuple[str, ...],
    provenance_root: bool,
) -> dict[str, Any]:
    task_dir = task_dir.resolve()
    instruction = (task_dir / "instruction.md").read_text()
    task_toml = read_task_toml(task_dir)
    swe_config = read_swe_config(task_dir)
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()

    instance_id = str(swe_config.get("instance_id") or task_dir.name)
    source_name = source or dataset_root.name
    problem_statement = str(swe_config.get("problem_statement") or instruction)
    prompt = problem_statement if prompt_source == "problem_statement" else instruction
    image = image_override or extract_dockerfile_image(dockerfile)
    if not image:
        raise ValueError(f"Cannot extract Docker image from {task_dir / 'environment' / 'Dockerfile'}")
    workdir = extract_dockerfile_workdir(dockerfile) or default_workdir
    base_commit = swe_config.get("base_commit")

    row: dict[str, Any] = {input_key: prompt}
    if prompt_alias_key and prompt_alias_key != input_key:
        row[prompt_alias_key] = prompt
    if label_key:
        row[label_key] = instance_id

    metadata: dict[str, Any] = {
        "instance_id": instance_id,
        "source": source_name,
        "image": image,
        "workdir": workdir,
        "problem_statement": problem_statement,
        "harbor": harbor_metadata(task_dir, source_name, task_toml, swe_config, image, workdir),
    }
    if include_pre_commands and base_commit:
        metadata["pre_commands"] = [
            f"git checkout {shlex.quote(str(base_commit))} -f",
            "git clean -fd",
        ]
    if include_eval_cmd:
        metadata["eval_cmd"] = build_eval_cmd(task_dir, swe_config)
    if include_inline_files:
        metadata["harbor_task"] = {
            "format": INLINE_TASK_FORMAT,
            "name": task_dir.name,
            "source": source_name,
            "files": read_inline_task_files(task_dir, inline_files),
        }
    if provenance_root:
        metadata["source_provenance"] = {
            "dataset_root": str(dataset_root),
            "task_name": task_dir.name,
        }
    row[metadata_key] = metadata
    return row


def read_task_toml(task_dir: Path) -> dict[str, Any]:
    return tomllib.loads((task_dir / "task.toml").read_text())


def read_swe_config(task_dir: Path) -> dict[str, Any]:
    path = task_dir / "tests" / "config.json"
    if not path.is_file():
        return {}
    config = json.loads(path.read_text())
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if isinstance(config.get(key), str):
            try:
                config[key] = json.loads(config[key])
            except json.JSONDecodeError:
                pass
    return config


def extract_dockerfile_image(dockerfile: str) -> str | None:
    image = None
    for raw_line in dockerfile.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"FROM\s+([^\s]+)", line, flags=re.IGNORECASE)
        if match:
            image = match.group(1)
    return image


def extract_dockerfile_workdir(dockerfile: str) -> str | None:
    workdir = None
    for raw_line in dockerfile.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"WORKDIR\s+(.+)", line, flags=re.IGNORECASE)
        if match:
            workdir = match.group(1).strip().strip("\"'")
    return workdir


def harbor_metadata(
    task_dir: Path,
    source_name: str,
    task_toml: dict[str, Any],
    swe_config: dict[str, Any],
    image: str,
    workdir: str,
) -> dict[str, Any]:
    return _drop_none(
        {
            "task_name": task_dir.name,
            "source": source_name,
            "repo": swe_config.get("repo"),
            "version": swe_config.get("version"),
            "base_commit": swe_config.get("base_commit"),
            "difficulty": swe_config.get("difficulty") or (task_toml.get("metadata") or {}).get("difficulty"),
            "docker_image": image,
            "docker_workdir": workdir,
            "fail_to_pass": swe_config.get("FAIL_TO_PASS"),
            "pass_to_pass": swe_config.get("PASS_TO_PASS"),
            "test_patch": swe_config.get("test_patch"),
            "reference_patch": swe_config.get("patch"),
            "verifier_timeout_sec": (task_toml.get("verifier") or {}).get("timeout_sec"),
            "agent_timeout_sec": (task_toml.get("agent") or {}).get("timeout_sec"),
        }
    )


def build_eval_cmd(task_dir: Path, swe_config: dict[str, Any]) -> str:
    test_script_path = task_dir / "tests" / "test.sh"
    if not test_script_path.is_file():
        raise FileNotFoundError(f"Cannot derive eval_cmd without {test_script_path}")
    task_slug = _safe_slug(task_dir.name)
    config_path = f"/tmp/slime_ags_{task_slug}_config.json"
    script_path = f"/tmp/slime_ags_{task_slug}_test.sh"
    verifier_dir = f"/tmp/slime_ags_{task_slug}_verifier"
    test_script = patch_harbor_test_script_for_ags(test_script_path.read_text(), config_path, verifier_dir)
    config_json = json.dumps(swe_config, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(verifier_dir)}",
            _heredoc(config_path, config_json, f"SLIME_AGS_CONFIG_{task_slug}"),
            _heredoc(script_path, test_script, f"SLIME_AGS_TEST_{task_slug}"),
            f"chmod +x {shlex.quote(script_path)}",
            f"bash {shlex.quote(script_path)}",
        ]
    )


def patch_harbor_test_script_for_ags(test_script: str, config_path: str, verifier_dir: str) -> str:
    return test_script.replace("/tests/config.json", config_path).replace("/logs/verifier", verifier_dir)


def read_inline_task_files(task_dir: Path, rel_paths: tuple[str, ...]) -> dict[str, str]:
    files: dict[str, str] = {}
    for rel_path in sorted(set(rel_paths)):
        _validate_relative_file_path(rel_path)
        path = task_dir / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"Inline Harbor task file not found: {path}")
        files[rel_path] = path.read_text()
    return files


def write_jsonl(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pretty_example(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows[0], ensure_ascii=False, indent=2) + "\n")


def write_schema(output: Path, *, input_key: str, prompt_alias_key: str, label_key: str, metadata_key: str) -> None:
    row_required = [input_key, metadata_key]
    properties: dict[str, Any] = {
        input_key: {"type": "string", "description": "Primary slime prompt key."},
        metadata_key: {
            "type": "object",
            "required": ["instance_id", "image", "workdir", "problem_statement"],
            "properties": {
                "instance_id": {"type": "string"},
                "source": {"type": "string"},
                "image": {"type": "string", "description": "Sandbox image consumed by ags_generator."},
                "workdir": {"type": "string", "description": "Repository path inside the sandbox."},
                "problem_statement": {"type": "string"},
                "pre_commands": {"type": "array", "items": {"type": "string"}},
                "eval_cmd": {"type": "string", "description": "Reward command; exit 0 means reward 1."},
                "harbor": {"type": "object", "description": "Extracted Harbor/SWE provenance and grading fields."},
                "harbor_task": {
                    "type": "object",
                    "description": "Optional inline files, present only with --include-inline-files.",
                },
            },
        },
    }
    if prompt_alias_key and prompt_alias_key != input_key:
        properties[prompt_alias_key] = {"type": "string", "description": "Prompt alias for --input-key prompt."}
    if label_key:
        properties[label_key] = {"type": "string", "description": "Optional label, usually the Harbor task name."}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "slime AGS prompt-data row converted from Harbor task",
                "type": "object",
                "required": row_required,
                "properties": properties,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def _is_harbor_task_dir(path: Path) -> bool:
    return (path / "instruction.md").is_file() and (path / "task.toml").is_file()


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _validate_relative_file_path(rel_path: str) -> None:
    path = Path(rel_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe task file path: {rel_path!r}")


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80] or "task"


def _heredoc(path: str, content: str, delimiter: str) -> str:
    while delimiter in content:
        delimiter += "_END"
    return f"cat > {shlex.quote(path)} <<'{delimiter}'\n{content.rstrip()}\n{delimiter}"


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    task_dirs = find_task_dirs(input_path, args.task, args.exclude_task)
    if args.offset:
        task_dirs = task_dirs[args.offset :]
    if args.limit is not None:
        task_dirs = task_dirs[: args.limit]
    if not task_dirs:
        raise ValueError(f"No Harbor tasks found under {input_path}")

    dataset_root = input_path if not _is_harbor_task_dir(input_path) else input_path.parent
    inline_files = tuple(args.inline_file) if args.inline_file else DEFAULT_INLINE_FILES
    rows = [
        task_to_row(
            task_dir,
            dataset_root=dataset_root,
            source=args.source,
            input_key=args.input_key,
            prompt_alias_key=args.prompt_alias_key,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
            prompt_source=args.prompt_source,
            image_override=args.image,
            default_workdir=args.default_workdir,
            include_pre_commands=not args.no_pre_commands,
            include_eval_cmd=not args.no_eval_cmd,
            include_inline_files=args.include_inline_files,
            inline_files=inline_files,
            provenance_root=args.provenance_root,
        )
        for task_dir in task_dirs
    ]

    write_jsonl(rows, args.output)
    if args.pretty_output:
        write_pretty_example(rows, args.pretty_output)
    if args.schema_output:
        write_schema(
            args.schema_output,
            input_key=args.input_key,
            prompt_alias_key=args.prompt_alias_key,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
        )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
