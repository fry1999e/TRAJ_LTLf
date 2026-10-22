"""NL-anchored finite-trace query helpers for coverage debt closure."""

from __future__ import annotations

from itertools import product

from schemas import (
    IRCoverageReport,
    SemanticStructure,
    TraceCoverageReport,
    TraceObligation,
    TraceObligationPlan,
    TraceQuery,
)
from trace_formatter import format_trace_for_beginner


def generate_coverage_debt_queries(
    *,
    semantic: SemanticStructure,
    trace_plan: TraceObligationPlan,
    trace_coverage: TraceCoverageReport,
    ir_coverage: IRCoverageReport,
    existing_query_ids: set[str] | None = None,
    data_split: str = "test",
    label_source: str = "human",
    max_queries: int = 8,
) -> list[TraceQuery]:
    """Generate pending trace questions from NL/IR coverage debt only."""

    existing_query_ids = existing_query_ids or set()
    obligation_map = {item.obligation_id: item for item in trace_plan.obligations}
    debt_obligations = [
        obligation_map[item]
        for item in ir_coverage.missing_obligation_ids
        if item in obligation_map
    ]
    if not debt_obligations:
        debt_obligations = _obligations_for_missing_roles(
            trace_plan,
            trace_coverage.missing_roles,
            trace_coverage.missing_confusions,
        )
    if not debt_obligations:
        debt_obligations = list(trace_plan.obligations)

    queries: list[TraceQuery] = []
    seen_shapes: set[tuple[tuple[str, ...], ...]] = set()
    for obligation in debt_obligations:
        if len(queries) >= max_queries:
            break
        for variant in range(1, 5):
            trace = _probe_trace_for_obligation(
                semantic=semantic,
                obligation=obligation,
                variant=variant,
            )
            shape = tuple(tuple(sorted(step)) for step in trace)
            if shape in seen_shapes:
                continue
            seen_shapes.add(shape)
            query_id = _next_query_id(len(queries) + 1, existing_query_ids)
            target_ambiguity = _target_ambiguity(
                obligation,
                trace_plan.confusion_targets,
                variant,
            )
            queries.append(
                TraceQuery(
                    query_id=query_id,
                    role=obligation.required_trace_role,
                    target_ambiguity=target_ambiguity,
                    target_ambiguity_axis_ids=(
                        obligation.ambiguity_axis_ids
                        or (
                            (target_ambiguity,)
                            if target_ambiguity
                            and target_ambiguity
                            not in {"satisfy", "violate", "distinguish"}
                            else ()
                        )
                    ),
                    constraints=(
                        *obligation.construction_constraints,
                        f"target_debt_id={obligation.obligation_id}",
                        f"minimum_horizon={obligation.minimum_horizon}",
                        "label_source_must_be_NL_temporal_simulation",
                    ),
                    trace=trace,
                    reason=(
                        "NL/coverage-debt-directed query for missing "
                        f"obligation {obligation.obligation_id}; truth label "
                        "must be assigned by NL temporal simulation review."
                    ),
                    beginner_view=format_trace_for_beginner(trace),
                    data_split=data_split,
                    label_source=label_source,
                    ir_node_ids=obligation.ir_node_ids,
                    obligation_ids=(obligation.obligation_id,),
                    ir_version=trace_plan.ir_version,
                    semantic_parameter_paths=obligation.semantic_parameter_paths,
                    generation_source="nl_coverage_debt",
                )
            )
            break
    return queries


def _obligations_for_missing_roles(
    trace_plan: TraceObligationPlan,
    missing_roles: tuple[str, ...],
    missing_confusions: tuple[str, ...],
) -> list[TraceObligation]:
    roles = set(missing_roles)
    result = [
        obligation
        for obligation in trace_plan.obligations
        if obligation.required_trace_role in roles
    ]
    if missing_confusions:
        result.extend(
            obligation
            for obligation in trace_plan.obligations
            if obligation.required_trace_role == "boundary"
        )
    return list(dict.fromkeys(result))


def _probe_trace_for_obligation(
    *,
    semantic: SemanticStructure,
    obligation: TraceObligation,
    variant: int,
) -> list[set[str]]:
    aps = tuple(
        dict.fromkeys(
            (
                *(semantic.faithful_ir.ap_names if semantic.faithful_ir else ()),
                *semantic.aps,
                semantic.trigger,
                semantic.response,
            )
        )
    )
    aps = tuple(ap for ap in aps if ap and ap not in {"unknown", "unknown_ap"})
    horizon = max(1, obligation.minimum_horizon, 2)
    trace = [set() for _ in range(horizon)]
    trigger = semantic.trigger if semantic.trigger in aps else (aps[0] if aps else "")
    response = semantic.response if semantic.response in aps else (aps[-1] if aps else "")

    if obligation.expected_relation == "satisfy":
        if aps:
            if "conjunction" in " ".join(obligation.construction_constraints):
                for step in trace:
                    step.update(aps)
            elif trigger and response and trigger != response:
                trace[0].add(trigger)
                trace[-1].add(response)
            else:
                trace[-1].add(aps[0])
        return trace

    if obligation.expected_relation == "violate":
        if trigger:
            trace[0].add(trigger)
        if variant % 2 == 0 and len(trace) > 1 and aps:
            trace[-1].add(aps[0])
        return trace

    if trigger and response and trigger != response:
        if variant % 2 == 0 and len(trace) > 1:
            trace[-1].update((trigger, response))
        else:
            trace[0].update((trigger, response))
        return trace
    if aps:
        trace[-1].add(aps[0])
    return trace


def enumerate_formula_free_trace_shapes(
    aps: tuple[str, ...],
    *,
    horizon: int,
    limit: int,
) -> list[list[set[str]]]:
    """Small fallback enumerator that carries no truth labels."""

    states = [set(), *({ap} for ap in aps)]
    if len(aps) > 1:
        states.append(set(aps))
    traces: list[list[set[str]]] = []
    for raw in product(range(len(states)), repeat=horizon):
        traces.append([set(states[index]) for index in raw])
        if len(traces) >= limit:
            break
    return traces


def _target_ambiguity(
    obligation: TraceObligation,
    confusions: tuple[str, ...],
    variant: int,
) -> str:
    if obligation.boundary_case:
        return (
            confusions[(variant - 1) % len(confusions)]
            if confusions
            else obligation.boundary_case
        )
    return obligation.expected_relation


def _next_query_id(index: int, existing: set[str]) -> str:
    while True:
        query_id = f"QUERY-NL-DEBT-{index:03d}"
        if query_id not in existing:
            existing.add(query_id)
            return query_id
        index += 1
