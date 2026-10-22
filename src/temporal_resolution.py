"""Formula-free temporal semantic ranges for trajectory-guided completion.

The NLU stage may expose a small, text-supported range of temporal readings,
but it never chooses an LTLf formula.  This module is the shared contract that
turns those semantic readings into finite completion-hole choices and resolves
one materialized choice back to executable IR semantics.
"""

from __future__ import annotations

from typing import Mapping

from temporal_language import TEMPORAL_LANGUAGE_REGISTRY

TEMPORAL_READING_TO_OPERATOR: dict[str, str] = {
    **{
        reading: entry.operator
        for reading, entry in TEMPORAL_LANGUAGE_REGISTRY.items()
    },
    "until": "U",
    "release": "R",
    "before": "BEFORE",
    "after": "AFTER",
    "precedence": "PRECEDENCE",
    "ordered_sequence": "ORDERED",
    "fixed_duration": "FIXED_DURATION",
    "exact_run_length": "EXACT_RUN_LENGTH",
    "bounded_window": "BOUNDED_WINDOW",
    "cardinality": "CARDINALITY",
    "recurrence": "GF",
    "periodicity": "PERIODIC",
    "start_boundary": "START",
    "end_boundary": "END",
    "short_trace_boundary": "SHORT_TRACE",
}

OPERATOR_TO_EXECUTABLE_KIND: dict[str, str] = {
    "epsilon": "identity",
    "G": "global_constraint",
    "F": "eventually",
    "X": "next_step",
    "XF": "eventually_strict",
    "U": "until",
    "R": "release",
    "BEFORE": "before",
    "AFTER": "after",
    "PRECEDENCE": "precedence",
    "ORDERED": "ordered_sequence",
    "FIXED_DURATION": "fixed_duration",
    "EXACT_RUN_LENGTH": "exact_run_length",
    "BOUNDED_WINDOW": "bounded_window",
    "CARDINALITY": "cardinality",
    "GF": "recurrence",
    "PERIODIC": "periodicity",
    "START": "start_boundary",
    "END": "end_boundary",
    "SHORT_TRACE": "short_trace_boundary",
}

# Only unary scope/timing readings can currently be interchanged without
# changing the reviewed graph topology or inventing relation parameters.
TRACE_RESOLVABLE_UNARY_READINGS = frozenset(
    TEMPORAL_LANGUAGE_REGISTRY
)


def canonical_temporal_reading(node_type: str) -> str:
    """Return the semantic reading represented by a legacy node type."""

    return node_type if node_type in TEMPORAL_READING_TO_OPERATOR else ""


def allowed_semantic_readings(node: object) -> tuple[str, ...]:
    """Return the explicit range, or the canonical singleton for old IR."""

    explicit = tuple(
        str(item).strip()
        for item in getattr(node, "allowed_semantic_readings", ())
        if str(item).strip()
    )
    if explicit:
        return tuple(dict.fromkeys(explicit))
    canonical = canonical_temporal_reading(str(getattr(node, "node_type", "")))
    return (canonical,) if canonical else ()


def allowed_operators_for_node(node: object) -> tuple[str, ...]:
    """Compile formula-free semantic readings into a finite operator domain."""

    return tuple(
        dict.fromkeys(
            TEMPORAL_READING_TO_OPERATOR[reading]
            for reading in allowed_semantic_readings(node)
            if reading in TEMPORAL_READING_TO_OPERATOR
        )
    )


def resolved_node_kind(
    node: object,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
) -> str:
    """Resolve one materialized hole choice to independent executable semantics."""

    node_type = str(getattr(node, "node_type", ""))
    if node_type not in TEMPORAL_READING_TO_OPERATOR:
        return node_type
    resolution = temporal_resolution_by_node or {}
    node_id = str(getattr(node, "node_id", ""))
    selected = resolution.get(node_id)
    if selected is None:
        operators = allowed_operators_for_node(node)
        if len(operators) > 1:
            raise ValueError(
                f"temporal node {node_id} has unresolved operator range {operators}"
            )
        selected = operators[0] if operators else TEMPORAL_READING_TO_OPERATOR[node_type]
    if selected in TEMPORAL_READING_TO_OPERATOR:
        selected = TEMPORAL_READING_TO_OPERATOR[selected]
    allowed = allowed_operators_for_node(node)
    if selected not in allowed:
        raise ValueError(
            f"operator {selected!r} is outside the approved semantic range "
            f"for temporal node {node_id}: {allowed}"
        )
    try:
        return OPERATOR_TO_EXECUTABLE_KIND[selected]
    except KeyError as exc:
        raise ValueError(f"unsupported temporal operator choice: {selected!r}") from exc


def temporal_range_errors(node: object) -> tuple[str, ...]:
    """Validate that an exposed range is executable without topology changes."""

    node_id = str(getattr(node, "node_id", ""))
    node_type = str(getattr(node, "node_type", ""))
    mode = str(getattr(node, "temporal_resolution", "fixed") or "fixed")
    explicit = tuple(getattr(node, "allowed_semantic_readings", ()))
    if node_type not in TEMPORAL_READING_TO_OPERATOR:
        if explicit or mode != "fixed":
            return (
                f"non-temporal node {node_id} cannot declare a temporal semantic range",
            )
        return ()
    if mode not in {"fixed", "trace_resolvable"}:
        return (f"node {node_id} has unsupported temporal_resolution={mode!r}",)
    readings = allowed_semantic_readings(node)
    unknown = tuple(
        reading for reading in readings if reading not in TEMPORAL_READING_TO_OPERATOR
    )
    errors: list[str] = []
    if unknown:
        errors.append(f"node {node_id} has unsupported semantic readings {unknown}")
    distinct_ops = allowed_operators_for_node(node)
    canonical_operator = TEMPORAL_READING_TO_OPERATOR[node_type]
    if explicit and canonical_operator not in distinct_ops:
        errors.append(
            f"node {node_id} semantic range must retain its canonical "
            f"{node_type!r} behavior"
        )
    if mode == "fixed" and len(distinct_ops) != 1:
        errors.append(
            f"fixed temporal node {node_id} must expose exactly one operator behavior"
        )
    if mode == "trace_resolvable":
        if len(distinct_ops) < 2:
            errors.append(
                f"trace_resolvable node {node_id} must expose at least two distinct behaviors"
            )
        if any(reading not in TRACE_RESOLVABLE_UNARY_READINGS for reading in readings):
            errors.append(
                f"trace_resolvable node {node_id} may currently vary only unary "
                "scope/timing readings"
            )
        if len(tuple(getattr(node, "children", ()))) != 1:
            errors.append(
                f"trace_resolvable node {node_id} must preserve a unary graph topology"
            )
    return tuple(errors)
