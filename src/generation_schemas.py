"""Machine-readable output schemas for proposal and patch calls."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from semantic_node_contracts import (
    node_parameter_json_schema,
    node_parameter_required,
)
from validated_generation import (
    PatchAuthorization,
    build_patch_authorization,
)


SEGMENTATION_PROPOSAL_SCHEMA = {
    "$id": "traj_ltlf_segmentation_proposal_v1",
    "type": "object",
    "properties": {"marked_text": {"type": "string"}},
    "required": ["marked_text"],
    "additionalProperties": False,
}


PLAIN_READING_PROPOSAL_SCHEMA = {
    "$id": "traj_ltlf_plain_reading_proposal_v5",
    "type": "object",
    "properties": {
        "overall_status": {
            "type": "string",
            "enum": ["no_ambiguity", "pending_plain_review"],
        },
        "ambiguities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ambiguity_id": {"type": "string"},
                    "source_segment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "evidence_quote": {"type": "string"},
                    "occurrence_index": {"type": ["integer", "null"], "minimum": 0},
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "option_id": {"type": "string", "enum": ["A", "B"]},
                                "plain_interpretation": {"type": "string"},
                                "difference_summary": {"type": "string"},
                            },
                            "required": [
                                "option_id",
                                "plain_interpretation",
                                "difference_summary",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "source_segment_ids",
                    "evidence_quote",
                    "question",
                    "options",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["overall_status", "ambiguities"],
    "additionalProperties": False,
}


PATCH_OPERATIONS_SCHEMA = {
    "$id": "traj_ltlf_atomic_patch_v2",
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "add"},
                            "path": {"type": "string", "pattern": "^/"},
                            "value": {},
                        },
                        "required": ["op", "path", "value"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "replace"},
                            "path": {"type": "string", "pattern": "^/"},
                            "value": {},
                        },
                        "required": ["op", "path", "value"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "remove"},
                            "path": {"type": "string", "pattern": "^/"},
                        },
                        "required": ["op", "path"],
                        "additionalProperties": False,
                    },
                ],
            },
        }
    },
    "required": ["operations"],
    "additionalProperties": False,
}


def patch_operations_schema(
    allowed_paths: Sequence[str] | PatchAuthorization,
    *,
    current_document: object | None = None,
    immutable_paths: Sequence[str] = (),
) -> dict:
    """Return the schema generated from one exact patch authorization."""

    if isinstance(allowed_paths, PatchAuthorization):
        authorization = allowed_paths
    else:
        paths = tuple(sorted({str(path) for path in allowed_paths if str(path)}))
        if not paths:
            return deepcopy(PATCH_OPERATIONS_SCHEMA)
        authorization = patch_authorization(
            paths,
            current_document=current_document,
            immutable_paths=immutable_paths,
        )
    if not authorization.targets:
        return deepcopy(PATCH_OPERATIONS_SCHEMA)
    variants: list[dict[str, object]] = []
    for target in authorization.targets:
        for operation in target.allowed_operations:
            properties: dict[str, object] = {
                "op": {"const": operation},
                "path": {"const": target.path},
            }
            required = ["op", "path"]
            if operation != "remove":
                properties["value"] = deepcopy(dict(target.value_schema))
                required.append("value")
            variants.append(
                {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                }
            )
    return {
        "$id": "traj_ltlf_atomic_patch_v3",
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "minItems": 1,
                "items": {"oneOf": variants},
            },
        },
        "required": ["operations"],
        "additionalProperties": False,
    }


def patch_authorization(
    allowed_paths: Sequence[str],
    *,
    current_document: object | None,
    immutable_paths: Sequence[str] = (),
) -> PatchAuthorization:
    """Compile exact operation/path pairs with domain-owned value contracts."""

    return build_patch_authorization(
        current_document,
        allowed_paths,
        immutable_paths=immutable_paths,
        value_contract_fn=_patch_path_parameter_contract,
    )


def _patch_path_parameter_contract(
    document: object | None,
    path: str,
) -> tuple[dict[str, object], bool | None]:
    plain_contract = _plain_reading_patch_path_contract(path)
    if plain_contract is not None:
        return plain_contract
    match = re.fullmatch(
        r"/(?:semantic/)?faithful_ir/nodes/(\d+)/parameters/([^/]+)",
        path,
    )
    if match is None or not isinstance(document, Mapping):
        return {}, None
    container: Any = document
    if path.startswith("/semantic/"):
        container = document.get("semantic")
    if not isinstance(container, Mapping):
        return {}, None
    ir = container.get("faithful_ir")
    if not isinstance(ir, Mapping):
        return {}, None
    nodes = ir.get("nodes")
    index = int(match.group(1))
    if not isinstance(nodes, list) or index >= len(nodes):
        return {}, None
    node = nodes[index]
    if not isinstance(node, Mapping):
        return {}, None
    node_type = str(node.get("node_type", ""))
    parameter = match.group(2).replace("~1", "/").replace("~0", "~")
    schema = node_parameter_json_schema(node_type, parameter)
    return (
        dict(schema) if schema is not None else {},
        node_parameter_required(node_type, parameter),
    )


def _plain_reading_patch_path_contract(
    path: str,
) -> tuple[dict[str, object], bool] | None:
    properties = PLAIN_READING_PROPOSAL_SCHEMA["properties"]
    if path == "/overall_status":
        return deepcopy(properties["overall_status"]), True
    if path == "/ambiguities":
        return {}, True
    ambiguity_schema = properties["ambiguities"]["items"]
    if re.fullmatch(r"/ambiguities/\d+", path):
        return {}, False
    if re.fullmatch(r"/ambiguities/\d+/options", path):
        return {}, True
    if re.fullmatch(
        r"/ambiguities/\d+/options/\d+/plain_interpretation",
        path,
    ):
        return {"type": "string"}, True
    return None


__all__ = [
    "PATCH_OPERATIONS_SCHEMA",
    "patch_authorization",
    "patch_operations_schema",
    "PLAIN_READING_PROPOSAL_SCHEMA",
    "SEGMENTATION_PROPOSAL_SCHEMA",
]
