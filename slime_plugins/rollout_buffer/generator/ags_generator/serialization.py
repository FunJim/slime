"""JSON-safe conversion between rollout-buffer payloads and slime Samples."""

from __future__ import annotations

from typing import Any

from slime.utils.types import Sample


SAMPLE_MARKER = "sample_dict_v1"


def sample_to_payload(sample: Sample) -> dict[str, Any]:
    return {"__type__": SAMPLE_MARKER, "sample": sample.to_dict()}


def samples_from_payload(payload: dict[str, Any]) -> list[Sample]:
    if "samples" in payload and isinstance(payload["samples"], list):
        return [Sample.from_dict(item) for item in payload["samples"]]
    return [sample_from_payload(payload)]


def sample_from_payload(payload: dict[str, Any]) -> Sample:
    if payload.get("__type__") == SAMPLE_MARKER and "sample" in payload:
        return Sample.from_dict(payload["sample"])
    if "sample" in payload and isinstance(payload["sample"], dict):
        return Sample.from_dict(payload["sample"])
    return Sample.from_dict(payload)


def output_item_from_samples(
    samples: list[Sample], *, instance_id: str, extra_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    first = samples[0]
    return {
        "uid": first.session_id or f"sample-{first.index}",
        "instance_id": instance_id,
        "messages": [],
        "reward": first.reward,
        "extra_info": extra_info or {},
        "samples": [sample.to_dict() for sample in samples],
        "__type__": SAMPLE_MARKER,
    }
