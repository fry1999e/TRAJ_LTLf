"""Backend-owned semantic obligations for SourceIntentGraph nodes.

Model debt records are hints.  This matrix decides whether missing information
can change the executable finite-trace language, separating AP denotation from
identity and formula topology.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping
from semantic_node_contracts import NODE_CONTRACTS


MATRIX_VERSION = "semantic-obligation-matrix-v1"


@dataclass(frozen=True)
class SemanticObligation:
    required_information: tuple[str, ...]
    irrelevant_information: tuple[str, ...] = ()
    contract_fields: tuple[str, ...] = ()


SEMANTIC_OBLIGATION_MATRIX: Mapping[str, SemanticObligation] = {
    "AP": SemanticObligation(
        ("unique_identity", "complete_predicate_span"),
        ("physical_state_event_definition",),
    ),
    "implication": SemanticObligation(
        ("condition_direction",), ("ap_edge_semantics",),
    ),
    "global_constraint": SemanticObligation(
        ("global_scope_authority",),
    ),
    "next_step": SemanticObligation(
        ("strong_or_weak_next", "last_position_semantics"), (), ("finite_trace_semantics",),
    ),
    "until": SemanticObligation(
        ("witness", "boundary", "current_or_future"), (), ("interval_boundary_policy",),
    ),
    "before": SemanticObligation(("strictness", "same_step", "witness")),
    "precedence": SemanticObligation(("strictness", "same_step", "witness")),
    "cardinality": SemanticObligation(("counted_predicate", "threshold", "window"), ("state_event_default",)),
    "fixed_duration": SemanticObligation(("origin", "duration", "terminal_policy")),
    "exact_run_length": SemanticObligation(("origin", "duration", "post_window_policy")),
    "bounded_window": SemanticObligation(("lower_bound", "upper_bound", "terminal_policy")),
    "eventually": SemanticObligation(("current_or_strict_future",), ("fixed_deadline_when_absent_from_source",), ("eventuality_policy",)),
}


_DENOTATION_TERMS = frozenset({
    "edge", "event", "state", "transition", "persistence", "persistent",
    "pressed", "cleared", "turned off", "turn on", "rising", "falling",
})


def obligation_for(kind: str) -> SemanticObligation:
    return SEMANTIC_OBLIGATION_MATRIX.get(kind, SemanticObligation(()))


def required_contract_fields_for_nodes(nodes) -> tuple[str, ...]:
    return tuple(sorted({
        field
        for node in nodes
        for field in obligation_for(str(getattr(node, "kind", ""))).contract_fields
    }))


def is_ap_denotation_only_debt(debt, nodes) -> bool:
    """True when a hint questions an AP's physics but not formula topology."""

    detail = str(getattr(debt, "detail", "")).casefold()
    if not any(term in detail for term in _DENOTATION_TERMS):
        return False
    topology_terms = (
        "scope", "direction", "and/or", "negation", "before", "until",
        "same step", "strict future", "weak next", "strong next", "count",
        "duration", "window", "witness", "reset",
    )
    if any(term in detail for term in topology_terms):
        return False
    start, end = tuple(getattr(debt, "source_span", (-1, -1)))
    return any(
        getattr(node, "kind", "") == "AP"
        and max(start, node.source_span[0]) < min(end, node.source_span[1])
        for node in nodes
    )


def is_unbounded_eventuality_observation(debt, source_text, nodes) -> bool:
    """A supported unbounded operator needs no invented measurable deadline.

    Only a descriptive missing-deadline observation is covered here. Actual time
    bounds and current/strict, scope, counting or other semantic differences are
    deliberately preserved for the normal candidate gates.
    """
    detail = str(getattr(debt, "detail", "")).casefold()
    if not re.search(r"\bunbounded\b|无界|无固定时限", detail):
        return False
    if not re.search(r"(?:does\s+not\s+specify|not\s+specified|no|without)\b[^.;]{0,90}\b(?:deadline|time\s+(?:bound|limit))\b|(?:未指定|未规定|没有)[^。；]{0,35}(?:截止|期限|时限)", detail):
        return False
    if re.search(r"\b(?:current|strict|scope|boundary|count|duration|reset|witness|weak|strong|order|negation)\b|当前|严格|作用域|边界|计数|持续|重置|否定", detail):
        return False
    if re.search(r"\b(?:within|between|deadline|by|no\s+later\s+than)\b|\b\d+\s*(?:steps?|seconds?|minutes?|hours?|days?)\b|之内|以内|截止|期限", source_text, re.I):
        return False
    start, end = tuple(getattr(debt, "source_span", (-1, -1)))
    return any(
        node.kind == "eventually" and NODE_CONTRACTS[node.kind].semantic_family == "unbounded_window"
        and node.source_span[0] <= start < end <= node.source_span[1]
        for node in nodes
    )


__all__ = [
    "MATRIX_VERSION",
    "SEMANTIC_OBLIGATION_MATRIX",
    "SemanticObligation",
    "is_ap_denotation_only_debt",
    "is_unbounded_eventuality_observation",
    "obligation_for",
    "required_contract_fields_for_nodes",
]
