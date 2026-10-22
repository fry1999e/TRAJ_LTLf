"""Structured runtime error diagnosis and responsibility routing.

This module is deliberately independent from the pipeline orchestration.  It
turns already-produced evaluation evidence into an auditable diagnosis packet
and maps an error class to the earliest artifact-owning stage.  It never copies
candidate formulas, reference formulas, or complete repair formulas into the
diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from data_examples import LabeledTrace
from schemas import FormulaConsistencyEvaluation, TraceSimulationResult


PIPELINE_ERROR_CLASSES = frozenset(
    {
        "ap_extraction_error",
        "extra_ap",
        "missing_ap",
        "forbidden_boolean_constant",
        "temporal_offset_too_early",
        "temporal_offset_too_late",
        "missing_global_scope",
        "excess_global_scope",
        "simultaneous_vs_strict_future",
        "eventually_vs_next",
        "strong_until_vs_weak_until",
        "precedence_direction_error",
        "scope_parent_mismatch",
        "boolean_grouping_error",
        "period_phase_error",
        "period_gap_error",
        "cardinality_polarity_error",
        "short_trace_boundary_error",
        "trace_label_conflict",
        "trace_coverage_gap",
        "ir_ambiguity_state_mismatch",
        "completion_formula_assignment_mismatch",
        "no_effective_delta",
    }
)

# Operational umbrella classes are needed when the evidence identifies the
# responsible component but cannot justify a narrower semantic error class.
OPERATIONAL_ERROR_CLASSES = frozenset(
    {
        "compiler_error",
        "ir_error",
        "trace_label_error",
        "candidate_consistency_error",
        "blocked_by_frozen_upstream",
        "semantic_domain_not_approved",
        "stale_derived_plan",
    }
)
ERROR_CLASSES = PIPELINE_ERROR_CLASSES | OPERATIONAL_ERROR_CLASSES


ROUTABLE_STAGES = frozenset(
    {
        "segmentation",
        "plain_reading",
        "nlu",
        "semantic_ir",
        "semantic_approval",
        "trace_obligation_planning",
        "trace_adjudication",
        "skeleton",
        "completion",
        "compiler",
        "selection",
        "approval",
        "manual_semantic_review",
        "manual_approval",
    }
)

ARTIFACTS = frozenset(
    {
        "surface_segmentation",
        "plain_reading",
        "faithful_semantic_ir",
        "candidate_ir_set",
        "approved_semantic_ir",
        "trace_obligation_plan",
        "labeled_trace_evidence",
        "constrained_skeleton_plan",
        "completion_hole_plan",
        "candidate_set",
        "evaluation_set",
        "final_approval",
        "learning_writeback",
    }
)


@dataclass(frozen=True)
class ErrorRouteSpec:
    """Static route owned by one runtime error class."""

    earliest_responsible_stage: str
    invalidated_artifacts: tuple[str, ...]
    expected_changed_artifacts: tuple[str, ...]
    escalation_stages: tuple[str, ...] = ()
    recommended_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.earliest_responsible_stage not in ROUTABLE_STAGES:
            raise ValueError(
                "unsupported earliest responsible stage: "
                + self.earliest_responsible_stage
            )
        unknown_stages = set(self.escalation_stages) - ROUTABLE_STAGES
        if unknown_stages:
            raise ValueError(
                "unsupported escalation stages: " + ", ".join(sorted(unknown_stages))
            )
        unknown_artifacts = (
            set(self.invalidated_artifacts) | set(self.expected_changed_artifacts)
        ) - ARTIFACTS
        if unknown_artifacts:
            raise ValueError(
                "unsupported route artifacts: "
                + ", ".join(sorted(unknown_artifacts))
            )


@dataclass(frozen=True)
class ErrorRoute:
    """Resolved route after applying frozen-stage constraints."""

    error_class: str
    earliest_responsible_stage: str
    responsible_stage: str
    invalidated_artifacts: tuple[str, ...]
    expected_changed_artifacts: tuple[str, ...]
    status: str = "ready"
    frozen_stages: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.error_class not in ERROR_CLASSES:
            raise ValueError("unsupported error class: " + self.error_class)
        if self.earliest_responsible_stage not in ROUTABLE_STAGES:
            raise ValueError("unsupported earliest responsible stage")
        if self.responsible_stage not in ROUTABLE_STAGES:
            raise ValueError("unsupported resolved responsible stage")
        if self.status not in {"ready", "escalated", "blocked"}:
            raise ValueError("route status must be ready, escalated, or blocked")


@dataclass(frozen=True)
class ErrorDiagnosisPacket:
    """Complete runtime diagnosis artifact, intentionally formula-free."""

    diagnosis_id: str
    requirement_id: str
    source_run_id: str
    source_stage: str
    responsible_stage: str
    error_class: str
    affected_ir_node_ids: tuple[str, ...]
    affected_obligation_ids: tuple[str, ...]
    affected_hole_ids: tuple[str, ...]
    failed_trace_ids: tuple[str, ...]
    expected_labels: tuple[tuple[str, bool], ...]
    simulated_labels: tuple[tuple[str, bool], ...]
    confusion_targets: tuple[str, ...]
    evidence_summary: tuple[str, ...]
    recommended_actions: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    candidate_information_exposed: bool
    diagnoser_role: str
    diagnoser_independence_id: str
    confidence: str
    created_at: str

    def __post_init__(self) -> None:
        if not self.diagnosis_id.strip():
            raise ValueError("diagnosis_id must not be empty")
        if not self.requirement_id.strip():
            raise ValueError("requirement_id must not be empty")
        if not self.source_run_id.strip():
            raise ValueError("source_run_id must not be empty")
        if not self.source_stage.strip():
            raise ValueError("source_stage must not be empty")
        if self.responsible_stage not in ROUTABLE_STAGES:
            raise ValueError("unsupported responsible_stage")
        if self.error_class not in ERROR_CLASSES:
            raise ValueError("unsupported error_class: " + self.error_class)
        if not self.diagnoser_role.strip():
            raise ValueError("diagnoser_role must not be empty")
        if not self.diagnoser_independence_id.strip():
            raise ValueError("diagnoser_independence_id must not be empty")
        if self.confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence must be high, medium, or low")


_IR_DOWNSTREAM = (
    "faithful_semantic_ir",
    "candidate_ir_set",
    "approved_semantic_ir",
    "trace_obligation_plan",
    "labeled_trace_evidence",
    "constrained_skeleton_plan",
    "completion_hole_plan",
    "candidate_set",
    "evaluation_set",
    "final_approval",
    "learning_writeback",
)
_CANDIDATE_DOWNSTREAM = (
    "candidate_set",
    "evaluation_set",
    "final_approval",
    "learning_writeback",
)
_TRACE_DOWNSTREAM = (
    "labeled_trace_evidence",
    "approved_semantic_ir",
    "evaluation_set",
    "final_approval",
    "learning_writeback",
)
_APPROVAL_DOWNSTREAM = ("final_approval", "learning_writeback")


_STAGE_ESCALATIONS: dict[str, tuple[str, ...]] = {
    "segmentation": ("manual_semantic_review",),
    "plain_reading": ("segmentation", "manual_semantic_review"),
    "nlu": ("plain_reading", "segmentation", "manual_semantic_review"),
    "semantic_ir": ("plain_reading", "segmentation", "manual_semantic_review"),
    "semantic_approval": (
        "semantic_ir",
        "plain_reading",
        "manual_semantic_review",
    ),
    "trace_obligation_planning": (
        "semantic_ir",
        "plain_reading",
        "manual_semantic_review",
    ),
    "trace_adjudication": (
        "trace_obligation_planning",
        "semantic_ir",
        "plain_reading",
        "manual_semantic_review",
    ),
    "skeleton": ("semantic_ir", "plain_reading", "manual_semantic_review"),
    "completion": (
        "skeleton",
        "semantic_ir",
        "plain_reading",
        "manual_semantic_review",
    ),
    "compiler": (
        "completion",
        "semantic_ir",
        "plain_reading",
        "manual_semantic_review",
    ),
    "selection": (
        "completion",
        "semantic_ir",
        "plain_reading",
        "manual_semantic_review",
    ),
    "approval": ("manual_approval",),
    "manual_semantic_review": (),
    "manual_approval": (),
}

_STAGE_INVALIDATIONS: dict[str, tuple[str, ...]] = {
    "segmentation": ("surface_segmentation", "plain_reading", *_IR_DOWNSTREAM),
    "plain_reading": ("plain_reading", *_IR_DOWNSTREAM),
    "nlu": _IR_DOWNSTREAM,
    "semantic_ir": _IR_DOWNSTREAM,
    "semantic_approval": (
        "candidate_ir_set",
        "approved_semantic_ir",
        "trace_obligation_plan",
        "constrained_skeleton_plan",
        "completion_hole_plan",
        *_CANDIDATE_DOWNSTREAM,
    ),
    "trace_obligation_planning": (
        "trace_obligation_plan",
        *_TRACE_DOWNSTREAM,
    ),
    "trace_adjudication": _TRACE_DOWNSTREAM,
    "skeleton": ("constrained_skeleton_plan", "completion_hole_plan", *_CANDIDATE_DOWNSTREAM),
    "completion": ("completion_hole_plan", *_CANDIDATE_DOWNSTREAM),
    "compiler": _CANDIDATE_DOWNSTREAM,
    "selection": ("evaluation_set", *_APPROVAL_DOWNSTREAM),
    "approval": _APPROVAL_DOWNSTREAM,
    "manual_semantic_review": (
        "plain_reading",
        *_IR_DOWNSTREAM,
    ),
    "manual_approval": _APPROVAL_DOWNSTREAM,
}

_STAGE_EXPECTED_CHANGES: dict[str, tuple[str, ...]] = {
    "segmentation": ("surface_segmentation", "plain_reading"),
    "plain_reading": ("plain_reading", "faithful_semantic_ir"),
    "nlu": ("faithful_semantic_ir", "candidate_ir_set"),
    "semantic_ir": (
        "faithful_semantic_ir",
        "candidate_ir_set",
        "approved_semantic_ir",
    ),
    "semantic_approval": ("candidate_ir_set", "approved_semantic_ir"),
    "trace_obligation_planning": (
        "trace_obligation_plan",
        "labeled_trace_evidence",
    ),
    "trace_adjudication": ("labeled_trace_evidence", "approved_semantic_ir"),
    "skeleton": ("constrained_skeleton_plan", "completion_hole_plan"),
    "completion": ("completion_hole_plan", "candidate_set"),
    "compiler": ("candidate_set",),
    "selection": ("evaluation_set", "final_approval"),
    "approval": ("final_approval",),
    "manual_semantic_review": ("plain_reading", "faithful_semantic_ir"),
    "manual_approval": ("final_approval",),
}


def _route_spec(
    stage: str,
    *,
    invalidated: Sequence[str] | None = None,
    expected: Sequence[str] | None = None,
    action: str,
) -> ErrorRouteSpec:
    return ErrorRouteSpec(
        earliest_responsible_stage=stage,
        invalidated_artifacts=tuple(
            invalidated if invalidated is not None else _STAGE_INVALIDATIONS[stage]
        ),
        expected_changed_artifacts=tuple(
            expected if expected is not None else _STAGE_EXPECTED_CHANGES[stage]
        ),
        escalation_stages=_STAGE_ESCALATIONS[stage],
        recommended_actions=(action,),
    )


_route_specs: dict[str, ErrorRouteSpec] = {}

for _error_class in {"ap_extraction_error", "extra_ap", "missing_ap"}:
    _route_specs[_error_class] = _route_spec(
        "nlu",
        action="rebuild AP extraction and FaithfulSemanticIR from reviewed source evidence",
    )

for _error_class in {
    "missing_global_scope",
    "excess_global_scope",
    "scope_parent_mismatch",
    "boolean_grouping_error",
}:
    _route_specs[_error_class] = _route_spec(
        "plain_reading",
        action="reopen ordinary-language scope and topology review before rebuilding IR",
    )

for _error_class in {
    "temporal_offset_too_early",
    "temporal_offset_too_late",
    "simultaneous_vs_strict_future",
    "eventually_vs_next",
    "strong_until_vs_weak_until",
    "precedence_direction_error",
    "period_phase_error",
    "period_gap_error",
    "cardinality_polarity_error",
    "short_trace_boundary_error",
    "ir_error",
}:
    _route_specs[_error_class] = _route_spec(
        "semantic_ir",
        action="reopen affected IR nodes and review the typed temporal policy",
    )

_route_specs["ir_ambiguity_state_mismatch"] = _route_spec(
    "semantic_approval",
    action="recompute ambiguity debt and refreeze ApprovedSemanticIR",
)
_route_specs["semantic_domain_not_approved"] = _route_spec(
    "semantic_approval",
    action="resolve the reported semantic evidence gate reasons and refreeze ApprovedSemanticIR",
)
_route_specs["stale_derived_plan"] = _route_spec(
    "semantic_approval",
    action="discard stale derived plans and recompile them from the current ApprovedSemanticIR",
)
_route_specs["trace_label_conflict"] = _route_spec(
    "trace_adjudication",
    action="repeat blind NL-only adjudication for the conflicting traces",
)
_route_specs["trace_label_error"] = _route_spec(
    "trace_adjudication",
    action="revalidate trace provenance and repeat independent NL-only adjudication",
)
_route_specs["trace_coverage_gap"] = _route_spec(
    "trace_obligation_planning",
    action="generate formula-free queries for the missing obligations and confusion axes",
)
_route_specs["completion_formula_assignment_mismatch"] = _route_spec(
    "completion",
    action="rebuild affected assignments under the current hole constraints",
)
_route_specs["forbidden_boolean_constant"] = _route_spec(
    "compiler",
    action="repair the compiler or completion contract that emitted a forbidden constant",
)
_route_specs["compiler_error"] = _route_spec(
    "compiler",
    action="inspect IR-to-formula compilation and semantic-preservation checks",
)
_route_specs["candidate_consistency_error"] = _route_spec(
    "completion",
    action="diagnose the failed trace obligations before changing candidate generation",
)
_route_specs["no_effective_delta"] = _route_spec(
    "completion",
    action="stop the unchanged rerun and escalate to the stage that froze its inputs",
)
_route_specs["blocked_by_frozen_upstream"] = _route_spec(
    "manual_semantic_review",
    action="request explicit review because all authorized upstream stages are frozen",
)

if set(_route_specs) != ERROR_CLASSES:  # pragma: no cover - import-time invariant
    missing = ERROR_CLASSES - set(_route_specs)
    extra = set(_route_specs) - ERROR_CLASSES
    raise RuntimeError(
        "error route table is incomplete: "
        f"missing={sorted(missing)}, extra={sorted(extra)}"
    )

ERROR_ROUTE_SPECS: Mapping[str, ErrorRouteSpec] = MappingProxyType(_route_specs)


def resolve_error_route(
    error_class: str,
    *,
    frozen_stages: Iterable[str] = (),
    source_stage: str = "",
) -> ErrorRoute:
    """Resolve a typed route, escalating around frozen upstream stages.

    ``no_effective_delta`` starts at ``source_stage`` when that value is a
    routable stage.  Every other class starts at its static earliest owner.
    When the owner is frozen, the first editable stage in its explicit
    escalation chain becomes responsible.  If none is editable, the route is
    returned with ``status='blocked'``.
    """

    if error_class not in ERROR_ROUTE_SPECS:
        raise ValueError("unsupported error class: " + error_class)
    frozen = tuple(dict.fromkeys(str(stage) for stage in frozen_stages))
    unknown = set(frozen) - ROUTABLE_STAGES
    if unknown:
        raise ValueError("unsupported frozen stages: " + ", ".join(sorted(unknown)))

    spec = ERROR_ROUTE_SPECS[error_class]
    earliest = (
        source_stage
        if error_class == "no_effective_delta" and source_stage in ROUTABLE_STAGES
        else spec.earliest_responsible_stage
    )
    base_invalidated = (
        _STAGE_INVALIDATIONS[earliest]
        if error_class == "no_effective_delta"
        else spec.invalidated_artifacts
    )
    base_expected = (
        _STAGE_EXPECTED_CHANGES[earliest]
        if error_class == "no_effective_delta"
        else spec.expected_changed_artifacts
    )
    escalation = _STAGE_ESCALATIONS[earliest]
    frozen_set = set(frozen)

    if error_class == "blocked_by_frozen_upstream":
        return ErrorRoute(
            error_class=error_class,
            earliest_responsible_stage=earliest,
            responsible_stage=earliest,
            invalidated_artifacts=base_invalidated,
            expected_changed_artifacts=base_expected,
            status="blocked",
            frozen_stages=frozen,
            blocked_by=frozen or (earliest,),
            reason="upstream authority or editable semantic evidence is unavailable",
        )

    if earliest not in frozen_set:
        return ErrorRoute(
            error_class=error_class,
            earliest_responsible_stage=earliest,
            responsible_stage=earliest,
            invalidated_artifacts=base_invalidated,
            expected_changed_artifacts=base_expected,
            frozen_stages=frozen,
            reason="earliest responsible stage is editable",
        )

    responsible = next(
        (stage for stage in escalation if stage not in frozen_set),
        None,
    )
    if responsible is None:
        terminal = escalation[-1] if escalation else earliest
        return ErrorRoute(
            error_class=error_class,
            earliest_responsible_stage=earliest,
            responsible_stage=terminal,
            invalidated_artifacts=_ordered_union(
                base_invalidated,
                _STAGE_INVALIDATIONS[terminal],
            ),
            expected_changed_artifacts=_ordered_union(
                base_expected,
                _STAGE_EXPECTED_CHANGES[terminal],
            ),
            status="blocked",
            frozen_stages=frozen,
            blocked_by=tuple(
                stage for stage in (earliest, *escalation) if stage in frozen_set
            ),
            reason="all authorized responsible and escalation stages are frozen",
        )

    return ErrorRoute(
        error_class=error_class,
        earliest_responsible_stage=earliest,
        responsible_stage=responsible,
        invalidated_artifacts=_ordered_union(
            base_invalidated,
            _STAGE_INVALIDATIONS[responsible],
        ),
        expected_changed_artifacts=_ordered_union(
            base_expected,
            _STAGE_EXPECTED_CHANGES[responsible],
        ),
        status="escalated",
        frozen_stages=frozen,
        blocked_by=tuple(
            stage for stage in (earliest, *escalation) if stage in frozen_set
        ),
        reason=(
            f"{earliest} is frozen; responsibility escalated to {responsible}"
        ),
    )


def build_error_diagnosis(
    *,
    requirement_id: str,
    source_run_id: str,
    source_stage: str,
    candidate_information_exposed: bool,
    diagnoser_role: str,
    diagnoser_independence_id: str,
    evaluation: FormulaConsistencyEvaluation | None = None,
    traces: Sequence[LabeledTrace] = (),
    closure_report: object | None = None,
    compiler_error: BaseException | str | None = None,
    ir_simulated_labels: Mapping[str, bool] | None = None,
    error_class: str | None = None,
    affected_ir_node_ids: Iterable[str] = (),
    affected_obligation_ids: Iterable[str] = (),
    affected_hole_ids: Iterable[str] = (),
    confusion_targets: Iterable[str] = (),
    frozen_stages: Iterable[str] = (),
    confidence: str = "medium",
    diagnosis_id: str = "",
    created_at: str = "",
) -> ErrorDiagnosisPacket:
    """Build a formula-free diagnosis from candidate and trace evidence.

    Classification precedence is explicit override, closure violation, compiler
    exception, contradictory labels, then per-trace evaluation mismatch.  The
    function never reads or serializes ``evaluation.formula``.
    """

    trace_index = _trace_index(traces)
    failed_results = tuple(
        result
        for result in (evaluation.trace_results if evaluation is not None else ())
        if not result.matched
    )
    conflicting_traces = _conflicting_label_traces(traces)
    closure_classes, closure_evidence = _closure_diagnostics(closure_report)
    exception_class, exception_code = _exception_diagnostic(compiler_error)

    inferred_class = error_class
    if inferred_class is None and closure_classes:
        inferred_class = closure_classes[0]
    if inferred_class is None and compiler_error is not None:
        inferred_class = exception_class
    if inferred_class is None and conflicting_traces:
        inferred_class = "trace_label_conflict"
    if inferred_class is None and failed_results:
        inferred_class = _classify_failed_results(
            failed_results,
            trace_index,
            ir_simulated_labels or {},
        )
    if inferred_class is None:
        inferred_class = "candidate_consistency_error"
    if inferred_class not in ERROR_CLASSES:
        raise ValueError("unsupported error class: " + inferred_class)

    route = resolve_error_route(
        inferred_class,
        frozen_stages=frozen_stages,
        source_stage=source_stage,
    )

    failed_trace_ids: list[str] = []
    expected_labels: list[tuple[str, bool]] = []
    simulated_labels: list[tuple[str, bool]] = []
    derived_nodes: list[str] = []
    derived_obligations: list[str] = []
    derived_confusions: list[str] = []
    for result in failed_results:
        trace = trace_index.get(result.trace_name)
        trace_id = _trace_id(trace, result.trace_name)
        failed_trace_ids.append(trace_id)
        expected_labels.append((trace_id, bool(result.expected)))
        simulated_labels.append((trace_id, bool(result.actual)))
        if trace is not None:
            derived_nodes.extend(trace.covered_ir_node_ids)
            derived_obligations.extend(trace.obligation_ids)
            derived_confusions.extend(_trace_confusions(trace))
    for trace in conflicting_traces:
        trace_id = _trace_id(trace, trace.name)
        failed_trace_ids.append(trace_id)
        expected_labels.append((trace_id, bool(trace.expected)))
        derived_nodes.extend(trace.covered_ir_node_ids)
        derived_obligations.extend(trace.obligation_ids)
        derived_confusions.extend(_trace_confusions(trace))

    evidence: list[str] = []
    if failed_results:
        evidence.append(
            f"candidate disagreed with {len(failed_results)} reviewed trace(s)"
        )
    if conflicting_traces:
        evidence.append(
            f"{len(conflicting_traces)} trace record(s) participate in same-shape label conflicts"
        )
    evidence.extend(closure_evidence)
    if exception_code:
        evidence.append("compiler diagnostic code=" + exception_code)
    if route.status == "escalated":
        evidence.append(route.reason)
    elif route.status == "blocked":
        evidence.append("routing blocked by frozen upstream stages")
    if not evidence:
        evidence.append("structured upstream error classification supplied")

    recommended = list(ERROR_ROUTE_SPECS[inferred_class].recommended_actions)
    if route.status == "escalated":
        recommended.append(
            "apply the repair at the escalated stage and invalidate all listed downstream artifacts"
        )
    if route.status == "blocked":
        recommended.append(
            "request authority or human review before attempting another rerun"
        )

    forbidden = [
        "do_not_include_gold_or_reference_formula",
        "do_not_prescribe_a_complete_replacement_formula",
        "do_not_derive_trace_labels_from_candidate_predictions",
        "do_not_repeat_an_unchanged_generation_stage",
    ]
    if candidate_information_exposed:
        forbidden.append("do_not_self_approve_a_seen_candidate")

    return ErrorDiagnosisPacket(
        diagnosis_id=diagnosis_id or "diagnosis-" + uuid4().hex,
        requirement_id=requirement_id,
        source_run_id=source_run_id,
        source_stage=source_stage,
        responsible_stage=route.responsible_stage,
        error_class=inferred_class,
        affected_ir_node_ids=_ordered_union(
            affected_ir_node_ids,
            derived_nodes,
        ),
        affected_obligation_ids=_ordered_union(
            affected_obligation_ids,
            derived_obligations,
        ),
        affected_hole_ids=_ordered_union(affected_hole_ids),
        failed_trace_ids=_ordered_union(failed_trace_ids),
        expected_labels=_unique_label_pairs(expected_labels),
        simulated_labels=_unique_label_pairs(simulated_labels),
        confusion_targets=_ordered_union(confusion_targets, derived_confusions),
        evidence_summary=tuple(evidence),
        recommended_actions=tuple(dict.fromkeys(recommended)),
        forbidden_actions=tuple(forbidden),
        candidate_information_exposed=bool(candidate_information_exposed),
        diagnoser_role=diagnoser_role,
        diagnoser_independence_id=diagnoser_independence_id,
        confidence=confidence,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )


def _trace_index(traces: Sequence[LabeledTrace]) -> dict[str, LabeledTrace]:
    index: dict[str, LabeledTrace] = {}
    for trace in traces:
        index[trace.name] = trace
        if trace.trace_id:
            index[trace.trace_id] = trace
    return index


def _trace_id(trace: LabeledTrace | None, fallback: str) -> str:
    if trace is None:
        return fallback
    return trace.trace_id or trace.name or fallback


def _trace_confusions(trace: LabeledTrace) -> tuple[str, ...]:
    return _ordered_union(
        trace.target_ambiguity_axis_ids,
        (trace.target_ambiguity, trace.boundary_role),
        trace.semantics.confusion_targets,
    )


def _conflicting_label_traces(
    traces: Sequence[LabeledTrace],
) -> tuple[LabeledTrace, ...]:
    by_shape: dict[tuple[tuple[str, ...], ...], list[LabeledTrace]] = {}
    for trace in traces:
        shape = tuple(tuple(sorted(step)) for step in trace.trace)
        by_shape.setdefault(shape, []).append(trace)
    conflicts: list[LabeledTrace] = []
    for same_shape in by_shape.values():
        if len({trace.expected for trace in same_shape}) > 1:
            conflicts.extend(same_shape)
    return tuple(conflicts)


def _classify_failed_results(
    failed_results: Sequence[TraceSimulationResult],
    trace_index: Mapping[str, LabeledTrace],
    ir_simulated_labels: Mapping[str, bool],
) -> str:
    saw_untrusted_label = False
    saw_ir_disagreement = False
    saw_compiler_disagreement = False
    semantic_classes: list[str] = []

    for result in failed_results:
        trace = trace_index.get(result.trace_name)
        if trace is not None:
            semantic_class = _semantic_error_from_confusions(_trace_confusions(trace))
            if semantic_class:
                semantic_classes.append(semantic_class)
            trusted = (
                trace.annotation_locked
                and trace.label_source in {"human", "independent_judge"}
                and trace.confidence in {"", "high", "medium"}
            )
            if not trusted:
                saw_untrusted_label = True
            ir_label = _lookup_ir_label(ir_simulated_labels, trace, result.trace_name)
        else:
            saw_untrusted_label = True
            ir_label = ir_simulated_labels.get(result.trace_name)

        if ir_label is None:
            continue
        if bool(result.actual) != bool(ir_label):
            saw_compiler_disagreement = True
        elif bool(result.expected) != bool(ir_label):
            saw_ir_disagreement = True

    if saw_compiler_disagreement:
        return "compiler_error"
    if saw_ir_disagreement and not saw_untrusted_label:
        return semantic_classes[0] if semantic_classes else "ir_error"
    if saw_untrusted_label:
        return "trace_label_error"
    if semantic_classes:
        return semantic_classes[0]
    return "candidate_consistency_error"


def _lookup_ir_label(
    labels: Mapping[str, bool],
    trace: LabeledTrace,
    fallback: str,
) -> bool | None:
    for key in (trace.trace_id, trace.name, fallback):
        if key and key in labels:
            return bool(labels[key])
    return None


_CONFUSION_CLASS_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("temporal_offset_too_early", ("too_early", "offset_early")),
    ("temporal_offset_too_late", ("too_late", "offset_late")),
    ("eventually_vs_next", ("eventually_vs_next", "f_vs_x")),
    (
        "simultaneous_vs_strict_future",
        ("simultaneous_vs_strict_future", "current_vs_strict_future"),
    ),
    ("strong_until_vs_weak_until", ("strong_until_vs_weak_until",)),
    ("missing_global_scope", ("current_vs_global_scope", "missing_global_scope")),
    ("excess_global_scope", ("excess_global_scope",)),
    ("precedence_direction_error", ("precedence_direction", "strict_before")),
    ("scope_parent_mismatch", ("scope_parent", "parent_scope")),
    ("boolean_grouping_error", ("boolean_grouping", "grouping_scope")),
    ("period_phase_error", ("period_phase", "phase_origin")),
    ("period_gap_error", ("period_gap", "minimum_spacing", "exact_recurrence")),
    ("cardinality_polarity_error", ("cardinality_polarity",)),
    ("short_trace_boundary_error", ("short_trace", "terminal_boundary")),
)


def _semantic_error_from_confusions(confusions: Iterable[str]) -> str:
    text = " ".join(str(item).lower() for item in confusions if str(item).strip())
    for error_class, markers in _CONFUSION_CLASS_MARKERS:
        if any(marker in text for marker in markers):
            return error_class
    return ""


_CLOSURE_FIELD_CLASSES = (
    ("extra_ap", ("extra_ap", "extra_aps", "unexpected_aps")),
    ("missing_ap", ("missing_ap", "missing_aps", "missing_required_aps")),
    (
        "forbidden_boolean_constant",
        ("forbidden_boolean_constant", "boolean_constants", "forbidden_constants"),
    ),
    ("ap_extraction_error", ("ap_extraction_error", "unknown_aps")),
)

_CLOSURE_REASON_CLASSES = {
    "extra_ap": "extra_ap",
    "missing_required_ap": "missing_ap",
    "missing_ap": "missing_ap",
    "forbidden_boolean_constant": "forbidden_boolean_constant",
    "unresolved_placeholder": "completion_formula_assignment_mismatch",
    "unsupported_operator": "compiler_error",
    "stale_ir_binding": "ir_ambiguity_state_mismatch",
}


def _closure_diagnostics(report: object | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if report is None:
        return (), ()
    classes: list[str] = []
    evidence: list[str] = []
    items = report if isinstance(report, (list, tuple, set, frozenset)) else (report,)
    for item in items:
        if isinstance(item, str):
            normalized = item.strip().lower().split(":", 1)[0]
            mapped = _CLOSURE_REASON_CLASSES.get(normalized, normalized)
            if mapped in ERROR_CLASSES:
                classes.append(mapped)
                evidence.append("formula closure violation=" + mapped)
            continue
        values = item if isinstance(item, Mapping) else vars(item) if hasattr(item, "__dict__") else {}
        if str(values.get("status", "")).strip().lower() in {"passed", "pass"}:
            continue
        explicit = str(
            values.get("error_class", values.get("code", values.get("kind", "")))
        ).strip()
        mapped_explicit = _CLOSURE_REASON_CLASSES.get(explicit, explicit)
        if mapped_explicit in ERROR_CLASSES:
            classes.append(mapped_explicit)
            path = str(values.get("path", values.get("ast_path", ""))).strip()
            evidence.append(
                "formula closure violation="
                + mapped_explicit
                + (" at AST path " + path if path else "")
            )
        reasons = values.get("reasons", ()) or ()
        if isinstance(reasons, str):
            reasons = (reasons,)
        reason_classes: list[str] = []
        for reason in reasons:
            code = str(reason).strip().lower().split(":", 1)[0]
            mapped = _CLOSURE_REASON_CLASSES.get(code, "")
            if mapped:
                classes.append(mapped)
                reason_classes.append(mapped)
                evidence.append("formula closure violation=" + mapped)
        if reason_classes:
            paths = values.get("violating_node_paths", ()) or ()
            path_count = 1 if isinstance(paths, str) else len(tuple(paths))
            evidence.append(
                "formula closure violating AST path count=" + str(path_count)
            )
            continue
        for error_class, keys in _CLOSURE_FIELD_CLASSES:
            if any(_nonempty(values.get(key)) for key in keys):
                classes.append(error_class)
                evidence.append("formula closure violation=" + error_class)
    return tuple(dict.fromkeys(classes)), tuple(dict.fromkeys(evidence))


def _nonempty(value: Any) -> bool:
    if value is None or value is False or value == "":
        return False
    try:
        return len(value) > 0
    except TypeError:
        return bool(value)


def _exception_diagnostic(
    error: BaseException | str | None,
) -> tuple[str, str]:
    if error is None:
        return "", ""
    text = str(error).lower()
    type_name = type(error).__name__ if isinstance(error, BaseException) else "error"
    if "forbidden_boolean_constant" in text or (
        "compiler_contract_violation" in text and "boolean constant" in text
    ):
        return "forbidden_boolean_constant", "compiler_contract_violation"
    if "completion_formula_assignment_mismatch" in text:
        return (
            "completion_formula_assignment_mismatch",
            "completion_formula_assignment_mismatch",
        )
    if "trace_label_ir_mismatch" in text:
        return "trace_label_error", "trace_label_ir_mismatch"
    if (
        "blocked_by_unexecutable_ir" in text
        or "unresolved semantic policy" in text
        or "faithfulirvalidationerror" in type_name.lower()
    ):
        return "ir_error", "unexecutable_or_invalid_ir"
    if "compiler_semantics_mismatch" in text:
        return "compiler_error", "compiler_semantics_mismatch"
    if "compiler_contract_violation" in text:
        return "compiler_error", "compiler_contract_violation"
    return "compiler_error", "unclassified_compiler_exception"


def _ordered_union(*groups: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for group in groups:
        for value in group:
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
    return tuple(result)


def _unique_label_pairs(
    pairs: Iterable[tuple[str, bool]],
) -> tuple[tuple[str, bool], ...]:
    result: list[tuple[str, bool]] = []
    for trace_id, label in pairs:
        pair = (str(trace_id), bool(label))
        if pair not in result:
            result.append(pair)
    return tuple(result)


__all__ = [
    "ARTIFACTS",
    "ERROR_CLASSES",
    "ERROR_ROUTE_SPECS",
    "OPERATIONAL_ERROR_CLASSES",
    "PIPELINE_ERROR_CLASSES",
    "ROUTABLE_STAGES",
    "ErrorDiagnosisPacket",
    "ErrorRoute",
    "ErrorRouteSpec",
    "build_error_diagnosis",
    "resolve_error_route",
]
