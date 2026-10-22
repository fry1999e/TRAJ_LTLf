"""Resolve trace disputes to existing IR nodes and exact source coordinates."""
from __future__ import annotations

from dataclasses import dataclass

from semantic_node_contracts import NODE_CONTRACTS


@dataclass(frozen=True)
class TraceConflictLocalization:
    node_ids: tuple[str, ...]
    source_spans: tuple[tuple[int, int], ...]
    operations: tuple[str, ...]
    status: str
    claim_ids: tuple[str, ...] = ()
    obligation_node_ids: tuple[str, ...] = ()


def localize_trace_conflict(ir, dispute, records, report=None):
    known = ir.node_map
    candidate_ids = set(getattr(report, "affected_node_ids", ()) if report else ())
    for evidence_id in dispute.evidence_ids:
        record = records.get(evidence_id)
        if record is not None:
            candidate_ids.update(getattr(record, "covered_ir_node_ids", ()))
            candidate_ids.update(
                getattr(getattr(record, "trace", None), "covered_ir_node_ids", ())
            )
    explanation = None
    if not any(node_id in known for node_id in candidate_ids):
        trace_shape = getattr(dispute, "trace_shape", ())
        if trace_shape:
            try:
                from ir_trace_semantics import explain_ir_trace
                explanation = explain_ir_trace(
                    ir,
                    [set(step) for step in trace_shape],
                )
                candidate_ids.update(explanation.affected_node_ids)
            except (ValueError, TypeError):
                explanation = None
    # No free-form model node IDs are accepted as authority.
    nodes = [known[node_id] for node_id in sorted(candidate_ids) if node_id in known]
    spans = []
    operations = set()
    for node in nodes:
        span = node.source_span
        if (len(span) == 2 and all(type(i) is int for i in span)
                and 0 <= span[0] < span[1] <= len(ir.source_text)
                and ir.source_text[span[0]:span[1]] == node.evidence_text):
            spans.append(tuple(span))
        else:
            return TraceConflictLocalization((), (), (), "unlocatable")
        contract = NODE_CONTRACTS.get(node.node_type)
        if contract is not None:
            if contract.parameters and node.node_type not in {"ap_condition", "boolean_constant"}:
                operations.add("replace_parameter")
            if node.node_type in {"implication", "until", "release", "before", "precedence", "after"}:
                operations.add("swap_children")
    if not nodes or not operations:
        return TraceConflictLocalization((), (), (), "unlocatable")
    return TraceConflictLocalization(
        tuple(node.node_id for node in nodes),
        tuple(sorted(set(spans))), tuple(sorted(operations)), "localized",
        (
            explanation.source_claim_ids
            if explanation is not None
            else tuple(
                "SCLAIM-" + __import__("hashlib").sha256(
                    f"{ir.source_hash}:{node.node_id}:{node.source_span[0]}:{node.source_span[1]}".encode()
                ).hexdigest()[:20]
                for node in nodes
            )
        ),
        tuple(node.node_id for node in nodes),
    )
