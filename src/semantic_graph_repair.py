"""Atomic, stable-ID graph transactions for Semantic IR proposals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from contract_versions import SEMANTIC_GRAPH_REPAIR_CONTRACT_VERSION
from semantic_topology import (
    faithful_ir_proposal_mapping,
    raw_ir_graph_integrity_problems,
)


class SemanticGraphRepairError(ValueError):
    """A graph transaction failed before commit; the input remains untouched."""


@dataclass(frozen=True)
class RedirectEdge:
    parent_node_id: str
    old_child_node_id: str
    new_child_node_id: str
    operation: str = "redirect_edge"


@dataclass(frozen=True)
class ReplaceRoot:
    old_root_node_id: str
    new_root_node_id: str
    operation: str = "replace_root"


@dataclass(frozen=True)
class RemoveNode:
    node_id: str
    operation: str = "remove_node"


@dataclass(frozen=True)
class ReplaceChildren:
    node_id: str
    expected_children: tuple[str, ...]
    new_children: tuple[str, ...]
    operation: str = "replace_children"


@dataclass(frozen=True)
class AddNode:
    node_id: str
    node_type: str
    children: tuple[str, ...]
    parameters: tuple[tuple[str, object], ...]
    evidence_quote: str
    source_span: tuple[int, int]
    source_segment_ids: tuple[str, ...]
    operation: str = "add_node"


GraphOperation = RedirectEdge | ReplaceRoot | RemoveNode | ReplaceChildren | AddNode


@dataclass(frozen=True)
class SemanticGraphTransaction:
    transaction_id: str
    precondition_fingerprint: str
    operations: tuple[GraphOperation, ...]
    authorization: str
    reason_code: str
    contract_version: str = SEMANTIC_GRAPH_REPAIR_CONTRACT_VERSION


@dataclass(frozen=True)
class SemanticGraphTransactionReceipt:
    transaction_id: str
    before_fingerprint: str
    after_fingerprint: str
    operation_count: int
    reason_code: str
    contract_version: str = SEMANTIC_GRAPH_REPAIR_CONTRACT_VERSION


def semantic_graph_fingerprint(proposal: object) -> str:
    ir, _prefix = faithful_ir_proposal_mapping(proposal)
    if ir is None:
        raise SemanticGraphRepairError("graph fingerprint requires faithful_ir")
    graph = {
        "root_node_id": ir.get("root_node_id"),
        "nodes": ir.get("nodes"),
        "atomic_propositions": ir.get("atomic_propositions"),
        "schema_version": ir.get("schema_version"),
    }
    return _fingerprint(graph)


def apply_semantic_graph_transaction(
    proposal: Mapping[str, Any],
    transaction: SemanticGraphTransaction,
    *,
    canonicalize: Callable[[object], object] | None = None,
    validate_full: Callable[[object], Sequence[object]] | None = None,
    validate_source: Callable[[object], Sequence[object]] | None = None,
) -> tuple[dict[str, Any], SemanticGraphTransactionReceipt]:
    """Apply all operations to a copy and publish only after every gate passes."""

    if transaction.contract_version != SEMANTIC_GRAPH_REPAIR_CONTRACT_VERSION:
        raise SemanticGraphRepairError("unsupported semantic graph repair contract")
    before = semantic_graph_fingerprint(proposal)
    if transaction.precondition_fingerprint != before:
        raise SemanticGraphRepairError("semantic graph transaction parent is stale")
    if not transaction.operations:
        raise SemanticGraphRepairError("semantic graph transaction has no operations")
    _authorize(transaction)

    candidate = deepcopy(dict(proposal))
    ir, _prefix = faithful_ir_proposal_mapping(candidate)
    if not isinstance(ir, dict):
        raise SemanticGraphRepairError("faithful_ir must be mutable object")
    for operation in transaction.operations:
        _apply_operation(ir, operation, transaction.authorization)

    problems = raw_ir_graph_integrity_problems(candidate)
    if problems:
        summary = "; ".join(f"{problem.code}: {problem.message}" for problem in problems)
        raise SemanticGraphRepairError("graph transaction failed preflight: " + summary)

    published: object = candidate
    if canonicalize is not None:
        published = canonicalize(candidate)
        issues = tuple(getattr(published, "issues", ()))
        if issues:
            raise SemanticGraphRepairError(
                "graph transaction failed canonicalization: "
                + "; ".join(str(getattr(issue, "message", issue)) for issue in issues)
            )
        published = getattr(published, "artifact", published)
    if validate_full is not None:
        full_issues = tuple(validate_full(published))
        if full_issues:
            raise SemanticGraphRepairError(
                "graph transaction failed full validation: "
                + "; ".join(str(issue) for issue in full_issues)
            )
    if validate_source is not None:
        source_issues = tuple(validate_source(published))
        if source_issues:
            raise SemanticGraphRepairError(
                "graph transaction failed source validation: "
                + "; ".join(str(issue) for issue in source_issues)
            )
    if not isinstance(published, Mapping):
        raise SemanticGraphRepairError("graph transaction produced non-object artifact")
    result = deepcopy(dict(published))
    after = semantic_graph_fingerprint(result)
    return result, SemanticGraphTransactionReceipt(
        transaction_id=transaction.transaction_id,
        before_fingerprint=before,
        after_fingerprint=after,
        operation_count=len(transaction.operations),
        reason_code=transaction.reason_code,
    )


def normalize_unary_boolean_graph(
    proposal: Mapping[str, Any],
) -> tuple[dict[str, Any], SemanticGraphTransactionReceipt] | None:
    """Collapse provably redundant AND(x)/OR(x) wrappers by Boolean identity."""

    ir, _prefix = faithful_ir_proposal_mapping(proposal)
    if ir is None or not isinstance(ir.get("nodes"), list):
        return None
    nodes = tuple(node for node in ir["nodes"] if isinstance(node, Mapping))
    node_ids = {str(node.get("node_id", "")) for node in nodes}
    operations: list[GraphOperation] = []
    removed: set[str] = set()
    root = str(ir.get("root_node_id", ""))

    # Recompute parent references after every planned collapse so nested unary
    # wrappers are handled without introducing dangling edges.
    mutable_children = {
        str(node.get("node_id", "")): list(node.get("children", ()))
        for node in nodes
    }
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        node_type = str(node.get("node_type", ""))
        children = tuple(str(value) for value in node.get("children", ()))
        if (
            node_type not in {"conjunction", "disjunction"}
            or len(children) != 1
            or not node_id
            or children[0] not in node_ids
        ):
            continue
        replacement = children[0]
        for parent_id, parent_children in mutable_children.items():
            if parent_id in removed:
                continue
            for index, child_id in enumerate(tuple(parent_children)):
                if child_id != node_id:
                    continue
                operations.append(RedirectEdge(parent_id, node_id, replacement))
                parent_children[index] = replacement
        if root == node_id:
            operations.append(ReplaceRoot(node_id, replacement))
            root = replacement
        operations.append(RemoveNode(node_id))
        removed.add(node_id)

    if not operations:
        return None
    before = semantic_graph_fingerprint(proposal)
    transaction = SemanticGraphTransaction(
        transaction_id="SGTX-" + _fingerprint(
            {
                "parent": before,
                "operations": [asdict(operation) for operation in operations],
                "reason": "NORMALIZE_UNARY_BOOLEAN",
            }
        )[:20],
        precondition_fingerprint=before,
        operations=tuple(operations),
        authorization="backend_boolean_identity",
        reason_code="NORMALIZE_UNARY_BOOLEAN",
    )
    return apply_semantic_graph_transaction(proposal, transaction)


def semantic_graph_transaction_from_mapping(
    mapping: Mapping[str, Any],
    *,
    proposal: Mapping[str, Any],
    authorization: str = "source_grounded_semantic_repair",
) -> SemanticGraphTransaction:
    if "operations" not in mapping or not isinstance(mapping["operations"], list):
        raise SemanticGraphRepairError("semantic graph repair requires operations array")
    parent = str(mapping.get("precondition_fingerprint", ""))
    if parent != semantic_graph_fingerprint(proposal):
        raise SemanticGraphRepairError("semantic graph repair response has stale parent")
    parsed: list[GraphOperation] = []
    for raw in mapping["operations"]:
        if not isinstance(raw, Mapping):
            raise SemanticGraphRepairError("semantic graph operation must be an object")
        kind = str(raw.get("operation", ""))
        if kind == "redirect_edge":
            parsed.append(
                RedirectEdge(
                    str(raw["parent_node_id"]),
                    str(raw["old_child_node_id"]),
                    str(raw["new_child_node_id"]),
                )
            )
        elif kind == "replace_root":
            parsed.append(
                ReplaceRoot(
                    str(raw["old_root_node_id"]),
                    str(raw["new_root_node_id"]),
                )
            )
        elif kind == "remove_node":
            parsed.append(RemoveNode(str(raw["node_id"])))
        elif kind == "replace_children":
            parsed.append(
                ReplaceChildren(
                    str(raw["node_id"]),
                    tuple(str(value) for value in raw["expected_children"]),
                    tuple(str(value) for value in raw["new_children"]),
                )
            )
        elif kind == "add_node":
            parameters = raw.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise SemanticGraphRepairError("AddNode parameters must be an object")
            ir, _prefix = faithful_ir_proposal_mapping(proposal)
            source_text = (
                str(ir.get("source_text", ""))
                if isinstance(ir, Mapping)
                else ""
            )
            evidence_quote = str(raw["evidence_quote"])
            starts = [
                index
                for index in range(len(source_text))
                if source_text.startswith(evidence_quote, index)
            ]
            if len(starts) != 1:
                raise SemanticGraphRepairError(
                    "AddNode evidence_quote must resolve uniquely in current source"
                )
            parsed.append(
                AddNode(
                    node_id=str(raw["node_id"]),
                    node_type=str(raw["node_type"]),
                    children=tuple(str(value) for value in raw["children"]),
                    parameters=tuple(
                        (str(name), value)
                        for name, value in sorted(parameters.items())
                    ),
                    evidence_quote=evidence_quote,
                    source_span=(starts[0], starts[0] + len(evidence_quote)),
                    source_segment_ids=tuple(
                        str(value) for value in raw["source_segment_ids"]
                    ),
                )
            )
        else:
            raise SemanticGraphRepairError(
                f"unsupported semantic graph operation {kind!r}"
            )
    reason = str(mapping.get("reason_code", "")).strip()
    if not reason:
        raise SemanticGraphRepairError("semantic graph repair requires reason_code")
    transaction_id = "SGTX-" + _fingerprint(
        {
            "parent": parent,
            "operations": [asdict(operation) for operation in parsed],
            "reason": reason,
            "authorization": authorization,
        }
    )[:20]
    return SemanticGraphTransaction(
        transaction_id=transaction_id,
        precondition_fingerprint=parent,
        operations=tuple(parsed),
        authorization=authorization,
        reason_code=reason,
    )


def semantic_graph_transaction_response_schema() -> dict[str, object]:
    node_id = {"type": "string", "minLength": 1}
    node_ids = {"type": "array", "items": node_id}
    operation_variants = [
        _operation_schema(
            "redirect_edge",
            {
                "parent_node_id": node_id,
                "old_child_node_id": node_id,
                "new_child_node_id": node_id,
            },
        ),
        _operation_schema(
            "replace_root",
            {"old_root_node_id": node_id, "new_root_node_id": node_id},
        ),
        _operation_schema("remove_node", {"node_id": node_id}),
        _operation_schema(
            "replace_children",
            {
                "node_id": node_id,
                "expected_children": node_ids,
                "new_children": node_ids,
            },
        ),
        _operation_schema(
            "add_node",
            {
                "node_id": node_id,
                "node_type": node_id,
                "children": node_ids,
                "parameters": {"type": "object"},
                "evidence_quote": {"type": "string", "minLength": 1},
                "source_segment_ids": {
                    "type": "array",
                    "items": node_id,
                    "minItems": 1,
                },
            },
        ),
    ]
    return {
        "type": "object",
        "required": [
            "precondition_fingerprint",
            "reason_code",
            "operations",
        ],
        "properties": {
            "precondition_fingerprint": {"type": "string", "minLength": 64},
            "reason_code": {"type": "string", "minLength": 1},
            "operations": {
                "type": "array",
                "items": {"oneOf": operation_variants},
                "minItems": 1,
            },
        },
        "additionalProperties": False,
    }


def _operation_schema(
    operation: str,
    properties: Mapping[str, object],
) -> dict[str, object]:
    return {
        "type": "object",
        "required": ["operation", *properties],
        "properties": {
            "operation": {"const": operation},
            **dict(properties),
        },
        "additionalProperties": False,
    }


def _authorize(transaction: SemanticGraphTransaction) -> None:
    allowed = {
        "backend_boolean_identity": {
            "redirect_edge",
            "replace_root",
            "remove_node",
        },
        "source_grounded_semantic_repair": {
            "redirect_edge",
            "replace_root",
            "remove_node",
            "replace_children",
            "add_node",
        },
    }
    authorized = allowed.get(transaction.authorization)
    if authorized is None:
        raise SemanticGraphRepairError("unknown semantic graph authorization")
    forbidden = [
        operation.operation
        for operation in transaction.operations
        if operation.operation not in authorized
    ]
    if forbidden:
        raise SemanticGraphRepairError(
            "unauthorized semantic graph operations: " + ", ".join(forbidden)
        )


def _apply_operation(
    ir: dict[str, Any],
    operation: GraphOperation,
    authorization: str,
) -> None:
    nodes = ir.get("nodes")
    if not isinstance(nodes, list):
        raise SemanticGraphRepairError("faithful_ir nodes must be an array")
    by_id = {
        str(node.get("node_id", "")): node
        for node in nodes
        if isinstance(node, dict)
    }
    if isinstance(operation, RedirectEdge):
        parent = by_id.get(operation.parent_node_id)
        if parent is None:
            raise SemanticGraphRepairError("RedirectEdge parent does not exist")
        children = parent.get("children")
        if not isinstance(children, list):
            raise SemanticGraphRepairError("RedirectEdge parent children are invalid")
        matches = [
            index
            for index, value in enumerate(children)
            if str(value) == operation.old_child_node_id
        ]
        if len(matches) != 1:
            raise SemanticGraphRepairError(
                "RedirectEdge precondition requires exactly one old edge"
            )
        if operation.new_child_node_id not in by_id:
            raise SemanticGraphRepairError("RedirectEdge target does not exist")
        children[matches[0]] = operation.new_child_node_id
        return
    if isinstance(operation, ReplaceRoot):
        if str(ir.get("root_node_id", "")) != operation.old_root_node_id:
            raise SemanticGraphRepairError("ReplaceRoot old root precondition failed")
        if operation.new_root_node_id not in by_id:
            raise SemanticGraphRepairError("ReplaceRoot target does not exist")
        ir["root_node_id"] = operation.new_root_node_id
        return
    if isinstance(operation, RemoveNode):
        if str(ir.get("root_node_id", "")) == operation.node_id:
            raise SemanticGraphRepairError("RemoveNode cannot remove current root")
        if operation.node_id not in by_id:
            raise SemanticGraphRepairError("RemoveNode target does not exist")
        if any(
            operation.node_id in tuple(str(value) for value in node.get("children", ()))
            for node in nodes
            if isinstance(node, Mapping)
        ):
            raise SemanticGraphRepairError("RemoveNode target is still referenced")
        ir["nodes"] = [
            node
            for node in nodes
            if not (
                isinstance(node, Mapping)
                and str(node.get("node_id", "")) == operation.node_id
            )
        ]
        return
    if isinstance(operation, ReplaceChildren):
        node = by_id.get(operation.node_id)
        if node is None:
            raise SemanticGraphRepairError("ReplaceChildren target does not exist")
        current = tuple(str(value) for value in node.get("children", ()))
        if current != operation.expected_children:
            raise SemanticGraphRepairError("ReplaceChildren precondition failed")
        if any(child not in by_id for child in operation.new_children):
            raise SemanticGraphRepairError("ReplaceChildren references unknown node")
        node["children"] = list(operation.new_children)
        return
    if isinstance(operation, AddNode):
        if authorization != "source_grounded_semantic_repair":
            raise SemanticGraphRepairError("AddNode requires semantic repair authority")
        if operation.node_id in by_id:
            raise SemanticGraphRepairError("AddNode node_id already exists")
        if (
            not operation.evidence_quote
            or operation.source_span[0] < 0
            or operation.source_span[1] <= operation.source_span[0]
            or not operation.source_segment_ids
        ):
            raise SemanticGraphRepairError("AddNode requires complete source evidence")
        if any(child not in by_id for child in operation.children):
            raise SemanticGraphRepairError("AddNode references unknown child")
        nodes.append(
            {
                "node_id": operation.node_id,
                "node_type": operation.node_type,
                "children": list(operation.children),
                "parameters": dict(operation.parameters),
                "evidence_quote": operation.evidence_quote,
                "source_span": list(operation.source_span),
                "source_segment_ids": list(operation.source_segment_ids),
            }
        )
        return
    raise SemanticGraphRepairError("unsupported semantic graph operation")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "AddNode",
    "GraphOperation",
    "RedirectEdge",
    "RemoveNode",
    "ReplaceChildren",
    "ReplaceRoot",
    "SemanticGraphRepairError",
    "SemanticGraphTransaction",
    "SemanticGraphTransactionReceipt",
    "apply_semantic_graph_transaction",
    "normalize_unary_boolean_graph",
    "semantic_graph_transaction_from_mapping",
    "semantic_graph_transaction_response_schema",
    "semantic_graph_fingerprint",
]
