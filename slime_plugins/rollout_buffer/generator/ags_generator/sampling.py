"""Sampling-parameter normalization for OpenAI-style agent adapters."""

from __future__ import annotations

from typing import Any


def normalize_sampling_params(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params or {})
    if "max_tokens" in out and "max_new_tokens" not in out:
        out["max_new_tokens"] = out.pop("max_tokens")
    else:
        out.pop("max_tokens", None)
    if "max_response_len" in out and "max_new_tokens" not in out:
        out["max_new_tokens"] = out.pop("max_response_len")
    else:
        out.pop("max_response_len", None)
    out.pop("skip_special_tokens", None)
    out.pop("no_stop_trim", None)
    out.pop("spaces_between_special_tokens", None)
    return out
