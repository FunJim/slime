"""Local artifact helpers for AGS rollout-buffer generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from slime.agent.sandbox import Sandbox
from slime.utils.types import Sample


def safe_artifact_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)[:200] or "unknown"


def sample_artifact_id(instance_id: str, sample: Sample) -> str:
    parts = [instance_id]
    if sample.group_index is not None:
        parts.append(f"g{sample.group_index}")
    if sample.index is not None:
        parts.append(f"i{sample.index}")
    if sample.session_id:
        parts.append(sample.session_id[-8:])
    return "__".join(parts)


class ArtifactWriter:
    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root) if root else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> ArtifactWriter:
        return cls(os.environ.get("TRAJECTORY_DUMP_DIR", "").strip() or None)

    def enabled(self) -> bool:
        return self.root is not None

    def _path(self, artifact_id: str, suffix: str) -> Path | None:
        if self.root is None:
            return None
        return self.root / f"{safe_artifact_name(artifact_id)}{suffix}"

    async def dump_trajectory(self, sb: Sandbox, workdir: str, artifact_id: str) -> str | None:
        path = self._path(artifact_id, ".trajectory.jsonl")
        if path is None:
            return None
        try:
            content = await sb.read_file(f"{workdir}/.harness/trajectory.jsonl", user="root")
        except Exception:
            content = ""
        path.write_text(content or "", encoding="utf-8")
        return str(path)

    def dump_patch(self, diff_text: str, artifact_id: str) -> str | None:
        path = self._path(artifact_id, ".patch")
        if path is None:
            return None
        path.write_text(diff_text or "", encoding="utf-8")
        return str(path)

    def dump_rollout(self, payload: dict[str, Any], artifact_id: str) -> str | None:
        path = self._path(artifact_id, ".rollout.json")
        if path is None:
            return None
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return str(path)
