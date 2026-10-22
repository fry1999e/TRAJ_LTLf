"""Formula-free lineage keys for reusable trace annotations."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

from .inputs import reject_hidden_fields
from .records import digest


def _query_view(query: object) -> dict[str, Any]:
    value = asdict(query) if is_dataclass(query) else dict(query)  # type: ignore[arg-type]
    allowed = {
        "query_id",
        "trace",
        "purpose",
        "obligation_ids",
        "ambiguity_axis_ids",
        "ir_node_ids",
        "semantic_parameter_paths",
        "beginner_view",
    }
    return {
        key: _stable_json_value(value[key])
        for key in sorted(allowed & set(value))
    }


def _stable_json_value(value: Any) -> Any:
    """Project native set-valued trace states into a deterministic cache key."""

    if is_dataclass(value):
        return _stable_json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _stable_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_stable_json_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported trace-lineage value: {type(value).__name__}")


def build_trace_cache_lineage(
    *,
    input_profile: Mapping[str, Any],
    ir_version: str,
    queries: Sequence[object],
    prompt_hash: str,
    reviewer_model_version: str,
) -> dict[str, Any]:
    contract = input_profile.get("non_quotable_contract_context", {})
    value = {
        "version": "trace-annotation-cache-lineage-v1",
        "source_hash": input_profile.get("source_input_hash", ""),
        "contract_hash": contract.get("contract_hash", "") if isinstance(contract, Mapping) else "",
        "ir_version": ir_version,
        "prompt_hash": prompt_hash,
        "reviewer_model_version": reviewer_model_version,
        "query_hash": digest([_query_view(query) for query in queries]),
        "formula_visible": False,
    }
    reject_hidden_fields(value)
    value["cache_key"] = digest(value)
    return value


__all__ = ["build_trace_cache_lineage"]
