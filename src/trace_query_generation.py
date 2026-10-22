"""Instantiate formula-free finite traces from a SemanticObligationPlan."""

from __future__ import annotations

from schemas import SemanticObligationPlan, SemanticStructure, TraceQuery
from trace_formatter import format_trace_for_beginner


def instantiate_trace_queries(
    plan: SemanticObligationPlan,
    semantic: SemanticStructure,
    *,
    target_trace_count: int = 6,
    target_trace_length: int | None = None,
    data_split: str = "test",
    label_source: str = "human",
) -> list[TraceQuery]:
    """Build pending query shapes without deriving or attaching truth labels."""

    target_count = max(6, target_trace_count)
    shapes = _query_shapes(plan, semantic)
    queries: list[TraceQuery] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()

    for role_name, trace, constraints in shapes:
        trace = _resize_trace(trace, target_trace_length)
        shape = _trace_key(trace)
        if shape in seen:
            continue
        seen.add(shape)
        queries.append(
            TraceQuery(
                query_id="QUERY-" + role_name.replace("_", "-").upper(),
                role=role_name,
                target_ambiguity=_target_ambiguity(plan, role_name),
                constraints=tuple(constraints),
                trace=trace,
                proposed_label=None,
                adjudicated_label="pending",
                reason=(
                    f"Query role {role_name}; asks the reviewer to resolve "
                    f"{_target_ambiguity(plan, role_name)} from the natural-language requirement."
                ),
                beginner_view=format_trace_for_beginner(trace),
                data_split=data_split,
                label_source=label_source,
                formula_source="none",
                formula_visibility=(),
            )
        )
        if len(queries) >= target_count:
            break

    if len(queries) < target_count:
        queries.extend(
            _generic_fillers(
                semantic=semantic,
                existing=queries,
                target_count=target_count,
                target_trace_length=target_trace_length,
                data_split=data_split,
                label_source=label_source,
            )
        )
    return queries[:target_count]


def _query_shapes(
    plan: SemanticObligationPlan,
    semantic: SemanticStructure,
) -> list[tuple[str, list[set[str]], tuple[str, ...]]]:
    trigger, response = _primary_pair(semantic)
    empty: set[str] = set()
    pattern = plan.pattern_family

    if pattern == "response":
        return [
            (
                "delayed_witness",
                [{trigger}, set(empty), {response}],
                (f"{trigger} holds at t0", f"{response} holds later"),
            ),
            (
                "missing_response",
                [{trigger}, set(empty), set(empty)],
                (f"{trigger} holds at t0", f"{response} is absent afterwards"),
            ),
            (
                "late_trigger_witness",
                [set(empty), {trigger}, {response}],
                (f"{trigger} first holds after t0", f"{response} holds later"),
            ),
            (
                "late_trigger_missing_response",
                [set(empty), {trigger}, set(empty)],
                (f"{trigger} first holds after t0", f"{response} is absent afterwards"),
            ),
            (
                "same_step_boundary",
                [{trigger, response}],
                (f"{trigger} and {response} co-occur",),
            ),
            (
                "final_state_boundary",
                [set(empty), set(empty), {trigger, response}],
                (f"{trigger} and {response} co-occur at the final state",),
            ),
            (
                "response_before_trigger",
                [{response}, {trigger}, set(empty)],
                (f"{response} occurs before {trigger}",),
            ),
            (
                "repeated_trigger_witness",
                [{trigger}, {trigger}, {response}],
                (f"{trigger} occurs more than once", f"{response} occurs later"),
            ),
        ]

    if pattern in {"precedence", "before"}:
        return [
            ("preceded_occurrence", [{response}, {trigger}], (f"{response} precedes {trigger}",)),
            ("missing_predecessor", [set(empty), {trigger}], (f"{trigger} occurs without prior {response}",)),
            ("same_step_boundary", [{trigger, response}], (f"{trigger} and {response} co-occur",)),
            ("late_predecessor_boundary", [{trigger}, {response}], (f"{response} occurs only after {trigger}",)),
            ("repeated_occurrence", [{response}, {trigger}, {trigger}], (f"{trigger} repeats after one {response}",)),
            ("minimal_length_boundary", [{trigger}], (f"single-state trace contains {trigger}",)),
        ]

    if pattern == "until":
        return [
            ("delayed_until_witness", [{trigger}, {trigger}, {response}], (f"{trigger} holds before {response}",)),
            ("immediate_until_witness", [{response}], (f"{response} holds immediately",)),
            ("left_break_before_witness", [{trigger}, set(empty), {response}], (f"{trigger} stops before {response}",)),
            ("missing_until_witness", [{trigger}, {trigger}, {trigger}], (f"{response} never occurs",)),
            ("final_state_witness", [{trigger}, {response}], (f"{response} holds only at the final state",)),
            ("minimal_length_boundary", [set(empty)], ("single empty state",)),
        ]

    if pattern in {"mutual_exclusion", "mutual_exclusion_with_coverage"}:
        return [
            ("left_only", [{trigger}], (f"only {trigger} holds",)),
            ("right_only", [{response}], (f"only {response} holds",)),
            ("cooccurrence_boundary", [{trigger, response}], (f"{trigger} and {response} co-occur",)),
            ("neither_boundary", [set(empty)], (f"neither {trigger} nor {response} holds",)),
            ("late_cooccurrence", [{trigger}, {trigger, response}], ("co-occurrence appears after t0",)),
            ("alternating_states", [{trigger}, {response}, {trigger}], ("participants alternate",)),
        ]

    if pattern in {"persistence", "global_state", "invariance"}:
        return [
            ("global_witness", [{trigger}, {trigger}, {trigger}], (f"{trigger} holds throughout",)),
            ("first_state_violation", [set(empty), {trigger}, {trigger}], (f"{trigger} is absent initially",)),
            ("middle_state_violation", [{trigger}, set(empty), {trigger}], (f"{trigger} is absent in the middle",)),
            ("final_state_violation", [{trigger}, {trigger}, set(empty)], (f"{trigger} is absent at the final state",)),
            ("minimal_length_boundary", [{trigger}], (f"single state contains {trigger}",)),
            ("late_state_boundary", [set(empty), {trigger}], (f"{trigger} starts after t0",)),
        ]

    aps = _semantic_aps(semantic)
    first = aps[0]
    second = aps[1] if len(aps) > 1 else first
    return [
        ("minimal_length_boundary", [set(empty)], ("single empty state",)),
        ("single_event_candidate", [{first}], (f"only {first} holds",)),
        ("delayed_event_candidate", [set(empty), {first}], (f"{first} holds after t0",)),
        ("missing_event_candidate", [set(empty), set(empty)], (f"{first} is absent",)),
        ("late_event_candidate", [set(empty), set(empty), {second}], (f"{second} holds at the final state",)),
        ("cooccurrence_boundary", [{first, second}], ("available participants co-occur",)),
    ]


def _generic_fillers(
    *,
    semantic: SemanticStructure,
    existing: list[TraceQuery],
    target_count: int,
    target_trace_length: int | None,
    data_split: str,
    label_source: str,
) -> list[TraceQuery]:
    aps = _semantic_aps(semantic)
    seen = {_trace_key(query.trace) for query in existing}
    fillers: list[TraceQuery] = []
    index = 1
    while len(existing) + len(fillers) < target_count:
        length = target_trace_length or 3
        trace = [set() for _ in range(max(1, length))]
        ap = aps[(index - 1) % len(aps)]
        trace[(index - 1) % len(trace)].add(ap)
        shape = _trace_key(trace)
        index += 1
        if shape in seen:
            if index > target_count * max(4, len(aps)):
                break
            continue
        seen.add(shape)
        fillers.append(
            TraceQuery(
                query_id=f"QUERY-SHAPE-{index:02d}",
                role="uncovered_trace_shape",
                target_ambiguity="uncovered_trace_shape",
                constraints=(f"{ap} appears at one selected position",),
                trace=trace,
                reason="Additional structurally distinct coverage query.",
                beginner_view=format_trace_for_beginner(trace),
                data_split=data_split,
                label_source=label_source,
            )
        )
    return fillers


def _target_ambiguity(
    plan: SemanticObligationPlan,
    role_name: str,
) -> str:
    if "same_step" in role_name:
        return "same_step_vs_strict_future"
    if "late_trigger" in role_name or "late_state" in role_name:
        return "current_vs_global_scope"
    if "final_state" in role_name or "minimal_length" in role_name:
        return "final_state_behavior"
    if "witness" in role_name and plan.pattern_family == "until":
        return "strong_vs_weak_until"
    if "cooccurrence" in role_name or "neither" in role_name:
        return "cooccurrence_policy"
    return plan.ambiguity_axes[0] if plan.ambiguity_axes else "semantic_coverage"


def _resize_trace(
    trace: list[set[str]],
    target_length: int | None,
) -> list[set[str]]:
    if target_length is None:
        return [set(step) for step in trace]
    if target_length < 1:
        raise ValueError("target_trace_length must be at least 1.")
    resized = [set(step) for step in trace[:target_length]]
    while len(resized) < target_length:
        resized.append(set())
    return resized


def _primary_pair(semantic: SemanticStructure) -> tuple[str, str]:
    aps = _semantic_aps(semantic)
    trigger = semantic.trigger if semantic.trigger in aps else aps[0]
    response = semantic.response if semantic.response in aps else (
        aps[1] if len(aps) > 1 else aps[0]
    )
    return trigger, response


def _semantic_aps(semantic: SemanticStructure) -> tuple[str, ...]:
    aps: list[str] = []
    for ap in (*semantic.aps, semantic.trigger, semantic.response):
        if ap and ap not in {"unknown", "unknown_ap"} and ap not in aps:
            aps.append(ap)
    return tuple(aps or ("event",))


def _trace_key(trace: list[set[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted(step)) for step in trace)
