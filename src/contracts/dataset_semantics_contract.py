"""Versioned, source-bound dataset semantics contracts.

The contract is deliberately formula-free.  It records only conventions that
were approved before an input is evaluated; missing fields stay explicitly
unknown and are never completed by runtime defaults.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .inputs import reject_hidden_fields
from .records import digest


CONTRACT_SCHEMA_VERSION = "dataset-semantics-v2"
LEGACY_CONTRACT_SCHEMA_VERSION = "dataset-semantics-v1"
LEGACY_SEMANTIC_FIELDS = (
    "finite_trace_semantics",
    "default_scope",
    "conditional_policy",
    "eventuality_policy",
    "coordination_policy",
    "state_event_policy",
    "interval_boundary_policy",
    "counting_policy",
)
SEMANTIC_FIELDS = (*LEGACY_SEMANTIC_FIELDS, "deontic_scope_policy")


def required_contract_fields(source_graph, candidate_ir=None):
    """Only executable differences introduced by this input create debt."""
    from semantic_obligation_matrix import required_contract_fields_for_nodes
    fields = set(required_contract_fields_for_nodes(source_graph.nodes))
    if (
        re.search(r"\b(?:shall|must)\b", source_graph.source_text, re.IGNORECASE)
        and not any(node.kind == "global_constraint" for node in source_graph.nodes)
    ):
        fields.add("deontic_scope_policy")
    # Explicit condition direction and Boolean membership need no defaults.
    return tuple(sorted(fields))
_AUTHORITIES = frozenset(
    {"dataset_guide", "annotation_guide", "project_specification", "unknown"}
)
DEONTIC_SCOPE_POLICIES = frozenset({
    "invariant_obligation", "initial_obligation", "source_explicit_only", "unknown",
})


def _validate_named_field(name: str, item: Mapping[str, Any]) -> None:
    if name == "deontic_scope_policy" and not item["unknown"]:
        if item["value"] not in DEONTIC_SCOPE_POLICIES - {"unknown"}:
            raise ValueError("unsupported deontic_scope_policy")


def _unknown_field() -> dict[str, Any]:
    return {
        "value": None,
        "authority": "unknown",
        "source_ref": None,
        "unknown": True,
    }


def _normalize_field(value: object) -> dict[str, Any]:
    if value is None:
        return _unknown_field()
    if not isinstance(value, Mapping):
        raise ValueError("dataset semantic fields require value/authority/source_ref records")
    unknown = value.get("unknown") is True or value.get("value") is None
    authority = str(value.get("authority", "unknown" if unknown else ""))
    if authority not in _AUTHORITIES:
        raise ValueError("unsupported dataset contract authority: " + authority)
    source_ref = value.get("source_ref")
    if not unknown and (authority == "unknown" or not isinstance(source_ref, str) or not source_ref.strip()):
        raise ValueError("known dataset semantics require an authoritative source_ref")
    return {
        "value": deepcopy(value.get("value")),
        "authority": authority,
        "source_ref": source_ref,
        "unknown": unknown,
    }


def build_dataset_semantics_contract(
    *,
    dataset_id: str,
    fields: Mapping[str, object] | None = None,
    source_documents: list[Mapping[str, object]] | tuple[Mapping[str, object], ...] = (),
    approved_by: list[str] | tuple[str, ...] = (),
    contract_version: str = CONTRACT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a canonical contract; unspecified semantics remain unknown."""

    if contract_version != CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported dataset semantics contract version")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id is required")
    supplied = dict(fields or {})
    extra = set(supplied) - set(SEMANTIC_FIELDS)
    if extra:
        raise ValueError("unknown dataset semantics fields: " + ", ".join(sorted(extra)))
    documents = []
    for item in source_documents:
        if not isinstance(item, Mapping):
            raise ValueError("source_documents entries must be objects")
        sha256 = str(item.get("sha256", ""))
        authority = str(item.get("authority", ""))
        if len(sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in sha256):
            raise ValueError("source document sha256 must be a 64-character hex digest")
        if authority not in _AUTHORITIES - {"unknown"}:
            raise ValueError("source document requires a recognized authority")
        documents.append(
            {
                "sha256": sha256.casefold(),
                "authority": authority,
                "source_ref": item.get("source_ref"),
            }
        )
    contract = {
        "contract_version": contract_version,
        "dataset_id": dataset_id,
        "source_documents": documents,
        "fields": {
            name: _normalize_field(supplied.get(name)) for name in SEMANTIC_FIELDS
        },
        "unknown_fields": [],
        "approved_by": [str(item) for item in approved_by if str(item).strip()],
        "contract_hash": "",
    }
    for name, item in contract["fields"].items():
        _validate_named_field(name, item)
    contract["unknown_fields"] = [
        name for name in SEMANTIC_FIELDS if contract["fields"][name]["unknown"]
    ]
    reject_hidden_fields(contract)
    contract["contract_hash"] = digest(
        {key: value for key, value in contract.items() if key != "contract_hash"}
    )
    validate_dataset_semantics_contract(contract, dataset_id=dataset_id)
    return contract


def validate_dataset_semantics_contract(
    contract: Mapping[str, Any], *, dataset_id: str | None = None
) -> None:
    required = {
        "contract_version",
        "dataset_id",
        "source_documents",
        "fields",
        "unknown_fields",
        "approved_by",
        "contract_hash",
    }
    if set(contract) != required:
        raise ValueError("dataset semantics contract has an invalid top-level shape")
    version = contract.get("contract_version")
    if version not in {CONTRACT_SCHEMA_VERSION, LEGACY_CONTRACT_SCHEMA_VERSION}:
        raise ValueError("unsupported dataset semantics contract version")
    if dataset_id is not None and contract.get("dataset_id") != dataset_id:
        raise ValueError("dataset semantics contract belongs to another dataset")
    # Validate field records without rebuilding or applying any defaults.
    fields = contract.get("fields")
    expected_fields = (
        SEMANTIC_FIELDS if version == CONTRACT_SCHEMA_VERSION else LEGACY_SEMANTIC_FIELDS
    )
    if not isinstance(fields, Mapping) or set(fields) != set(expected_fields):
        raise ValueError("dataset semantics contract fields are incomplete")
    normalized = {name: _normalize_field(fields[name]) for name in expected_fields}
    for name, item in normalized.items():
        _validate_named_field(name, item)
    documents = contract.get("source_documents")
    if not isinstance(documents, list):
        raise ValueError("dataset semantics source_documents must be an array")
    for item in documents:
        if not isinstance(item, Mapping) or set(item) != {"sha256", "authority", "source_ref"}:
            raise ValueError("invalid dataset semantics source document record")
        sha256 = str(item.get("sha256", ""))
        if len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256):
            raise ValueError("invalid dataset semantics source document hash")
        if item.get("authority") not in _AUTHORITIES - {"unknown"}:
            raise ValueError("invalid dataset semantics source document authority")
    unknown = [name for name in expected_fields if normalized[name]["unknown"]]
    if list(contract.get("unknown_fields", ())) != unknown:
        raise ValueError("dataset semantics contract unknown_fields mismatch")
    expected = digest({key: value for key, value in contract.items() if key != "contract_hash"})
    if contract.get("contract_hash") != expected:
        raise ValueError("dataset semantics contract hash mismatch")
    reject_hidden_fields(dict(contract))


def unknown_dataset_semantics_contract(dataset_id: str) -> dict[str, Any]:
    """Return an explicit all-unknown contract, never a semantic default."""

    return build_dataset_semantics_contract(dataset_id=dataset_id)


def contract_prompt_view(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the formula-free, explicitly non-quotable prompt projection."""

    validate_dataset_semantics_contract(contract)
    return {
        "context_kind": "non_quotable_dataset_semantics_contract",
        "quotation_allowed": False,
        "dataset_id": contract["dataset_id"],
        "contract_version": contract["contract_version"],
        "contract_hash": contract["contract_hash"],
        "fields": deepcopy(contract["fields"]),
        "unknown_fields": list(contract["unknown_fields"]),
    }


def load_dataset_semantics_contracts(directory: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate a frozen registry with exactly one contract per dataset."""

    result = {}
    paths = tuple(Path(directory).glob("*.dataset-semantics-v1.json")) + tuple(
        Path(directory).glob("*.dataset-semantics-v2.json")
    )
    for path in sorted(paths):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        validate_dataset_semantics_contract(value)
        dataset_id = value["dataset_id"]
        if dataset_id in result:
            raise ValueError("duplicate dataset semantics contract: " + dataset_id)
        result[dataset_id] = value
    return result


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "LEGACY_CONTRACT_SCHEMA_VERSION",
    "SEMANTIC_FIELDS",
    "DEONTIC_SCOPE_POLICIES",
    "build_dataset_semantics_contract",
    "contract_prompt_view",
    "load_dataset_semantics_contracts",
    "unknown_dataset_semantics_contract",
    "validate_dataset_semantics_contract",
]
