"""Single policy boundary for finite-trace evidence sources.

The IR compiler always emits the canonical obligation inventory.  This module
creates an experiment-scoped view without mutating IR semantics or pretending
that omitted evidence was observed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from data_examples import LabeledTrace
from schemas import (
    TraceEvidencePolicy,
    TraceObligation,
    TraceObligationPlan,
    TraceQuery,
)


@dataclass(frozen=True)
class TraceEvidenceContext:
    """Canonical and policy-effective plans for one immutable IR version."""

    policy: TraceEvidencePolicy
    canonical_plan: TraceObligationPlan
    effective_plan: TraceObligationPlan


def build_trace_evidence_context(
    plan: TraceObligationPlan,
    policy: TraceEvidencePolicy | None = None,
) -> TraceEvidenceContext:
    normalized = policy or TraceEvidencePolicy()
    return TraceEvidenceContext(
        policy=normalized,
        canonical_plan=plan,
        effective_plan=project_trace_obligation_plan(plan, normalized),
    )


def project_trace_obligation_plan(
    plan: TraceObligationPlan,
    policy: TraceEvidencePolicy,
) -> TraceObligationPlan:
    """Return the evidence contract active for a run, preserving the canonical plan."""

    if policy.boundary_enabled:
        return plan

    obligations = tuple(
        obligation
        for obligation in plan.obligations
        if not is_boundary_obligation(obligation)
    )
    retained_confusions = {
        value
        for obligation in obligations
        for value in (
            *obligation.ambiguity_axis_ids,
            str(dict(obligation.semantic_parameters).get("confusion_id", "")),
        )
        if value
    }
    return replace(
        plan,
        obligations=obligations,
        confusion_targets=tuple(
            item for item in plan.confusion_targets if item in retained_confusions
        ),
        coverage_requirements=tuple(
            requirement
            for requirement in plan.coverage_requirements
            if "boundary" not in requirement.lower()
            and "distinguish" not in requirement.lower()
        )
        + ("boundary trace evidence is intentionally disabled for this experiment",),
        plan_purpose=plan.plan_purpose + ":boundary_disabled",
    )


def is_boundary_obligation(obligation: TraceObligation) -> bool:
    role = obligation.required_trace_role.strip().lower()
    axes = tuple(item.lower() for item in obligation.ambiguity_axis_ids)
    return bool(
        obligation.boundary_case
        or role == "boundary"
        or "boundary" in role
        or any("boundary" in axis for axis in axes)
    )


def is_boundary_query(query: TraceQuery) -> bool:
    role = query.role.strip().lower()
    ambiguity = query.target_ambiguity.strip().lower()
    axes = tuple(item.lower() for item in query.target_ambiguity_axis_ids)
    return bool(
        role == "boundary"
        or "boundary" in role
        or "boundary" in ambiguity
        or role in {"same_step", "cooccurrence_boundary", "neither_boundary"}
        or any("boundary" in axis for axis in axes)
    )


def is_boundary_trace(trace: LabeledTrace) -> bool:
    roles = (trace.trace_role, trace.query_role, trace.role)
    return bool(
        trace.kind == "boundary"
        or any(role == "boundary" or "boundary" in role for role in roles if role)
        or trace.boundary_role
        or any("boundary" in axis for axis in trace.target_ambiguity_axis_ids)
    )


def filter_queries_for_policy(
    queries: list[TraceQuery],
    policy: TraceEvidencePolicy,
) -> list[TraceQuery]:
    if policy.boundary_source == "generated":
        return list(queries)
    return [query for query in queries if not is_boundary_query(query)]


def filter_traces_for_policy(
    traces: list[LabeledTrace],
    policy: TraceEvidencePolicy,
) -> list[LabeledTrace]:
    if policy.boundary_enabled:
        return list(traces)
    return [trace for trace in traces if not is_boundary_trace(trace)]


__all__ = [
    "TraceEvidenceContext",
    "build_trace_evidence_context",
    "filter_queries_for_policy",
    "filter_traces_for_policy",
    "is_boundary_obligation",
    "is_boundary_query",
    "is_boundary_trace",
    "project_trace_obligation_plan",
]
