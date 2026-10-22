"""Trajectory-backed resolution of temporal completion assignments.

This module deliberately consumes only reviewed whole-trace labels and already
materialized completion decisions.  It never evaluates IR subnodes and never
uses an expected/reference formula as operator evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import prod

from data_examples import LabeledTrace
from schemas import (
    CompletionDecision,
    CompletionHolePlan,
    FormulaConsistencyEvaluation,
    HoleOperatorResolution,
    OperatorResolutionReport,
    TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION,
    TRACE_PURPOSE_OPERATOR_DISCRIMINATION,
    TRACE_PURPOSE_SEMANTIC_VALIDATION,
)


@dataclass(frozen=True)
class OperatorTraceEvidenceSet:
    """Evidence bound to one fixed IR and CompletionHolePlan.

    IR-branch discrimination evidence is forbidden here. Baseline semantic
    traces may replay complete assignments, while only operator-purpose traces
    are allowed to claim operator discrimination.
    """

    ir_version: str
    hole_plan_version: str
    discrimination_traces: tuple[LabeledTrace, ...] = ()
    baseline_traces: tuple[LabeledTrace, ...] = ()
    parent_ir_status: str = "approved"

    def __post_init__(self) -> None:
        if any(
            trace.evidence_purpose != TRACE_PURPOSE_OPERATOR_DISCRIMINATION
            for trace in self.discrimination_traces
        ):
            raise ValueError(
                "OperatorTraceEvidenceSet discrimination_traces must use the "
                "operator-assignment purpose"
            )
        if any(
            trace.evidence_purpose != TRACE_PURPOSE_SEMANTIC_VALIDATION
            for trace in self.baseline_traces
        ):
            raise ValueError(
                "OperatorTraceEvidenceSet baseline_traces must use semantic_validation"
            )
        for trace in (*self.discrimination_traces, *self.baseline_traces):
            if trace.evidence_purpose == TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION:
                raise ValueError(
                    "IR-branch discrimination evidence cannot resolve operators"
                )
            if trace.ir_version and trace.ir_version != self.ir_version:
                raise ValueError("operator trace evidence references a stale IR version")
        if any(
            trace.parent_hole_plan_version
            and trace.parent_hole_plan_version != self.hole_plan_version
            for trace in self.discrimination_traces
        ):
            raise ValueError(
                "operator trace evidence references a stale CompletionHolePlan"
            )

    @property
    def traces(self) -> tuple[LabeledTrace, ...]:
        return (*self.baseline_traces, *self.discrimination_traces)


def assignment_temporal_resolution_by_node(
    *,
    hole_plan: CompletionHolePlan,
    assignment: Mapping[str, str] | Iterable[tuple[str, str]],
) -> dict[str, str]:
    """Translate one complete hole assignment into the IR compiler namespace.

    Completion decisions and resolution reports deliberately identify choices by
    ``hole_id``.  The certified IR compiler deliberately identifies the same
    choices by ``ir_node_id``.  Keeping this checked translation in one place
    prevents finalization code from accidentally treating those identifiers as
    interchangeable.
    """

    pairs = (
        tuple(assignment.items())
        if isinstance(assignment, Mapping)
        else tuple(assignment)
    )
    by_hole: dict[str, str] = {}
    for hole_id, operator in pairs:
        if hole_id in by_hole:
            raise ValueError(f"temporal assignment repeats hole {hole_id!r}")
        by_hole[hole_id] = operator

    constraints = {hole.hole_id: hole for hole in hole_plan.holes}
    if set(by_hole) != set(constraints):
        missing = sorted(set(constraints) - set(by_hole))
        extra = sorted(set(by_hole) - set(constraints))
        raise ValueError(
            "temporal assignment does not close CompletionHolePlan: "
            f"missing={missing}, extra={extra}"
        )

    resolution: dict[str, str] = {}
    for hole_id, constraint in constraints.items():
        operator = by_hole[hole_id]
        if operator not in constraint.allowed_operators:
            raise ValueError(
                f"operator {operator!r} is outside hole {hole_id!r} domain "
                f"{constraint.allowed_operators}"
            )
        if constraint.ir_node_id in resolution:
            raise ValueError(
                "CompletionHolePlan maps multiple holes to IR node "
                f"{constraint.ir_node_id!r}"
            )
        resolution[constraint.ir_node_id] = operator
    return resolution


def resolved_temporal_resolution_by_node(
    *,
    report: OperatorResolutionReport,
    hole_plan: CompletionHolePlan,
) -> dict[str, str]:
    """Return the sole authoritative assignment in IR-node form.

    This is the bridge between trajectory-backed operator resolution and the
    final certified IR compilation.  It fails closed for stale, ambiguous, or
    non-authoritative reports.
    """

    validate_operator_resolution_report(
        report=report,
        hole_plan=hole_plan,
        require_final=True,
    )
    if report.status not in {"complete", "not_required"}:
        raise ValueError(
            f"operator resolution is not final: status={report.status!r}"
        )
    if not report.authoritative:
        raise ValueError("operator resolution is not authoritative")
    if len(report.passing_assignments) != 1:
        raise ValueError(
            "operator resolution must contain exactly one passing assignment"
        )
    return assignment_temporal_resolution_by_node(
        hole_plan=hole_plan,
        assignment=report.passing_assignments[0],
    )


def validate_operator_resolution_report(
    *,
    report: OperatorResolutionReport,
    hole_plan: CompletionHolePlan,
    require_final: bool = False,
) -> None:
    """Validate that one report is internally consistent with one hole plan.

    Dataclass construction remains permissive for persisted diagnostic records.
    Production consumers call this validator so a report, assignment mapping,
    and certified compilation cannot be assembled from different decisions.
    """

    if report.ir_version != hole_plan.ir_version:
        raise ValueError("OperatorResolutionReport references a stale IR version")
    if report.hole_plan_version != hole_plan.skeleton_plan_version:
        raise ValueError("OperatorResolutionReport references a stale CompletionHolePlan")
    constraints = {hole.hole_id: hole for hole in hole_plan.holes}
    if len(constraints) != len(hole_plan.holes):
        raise ValueError("CompletionHolePlan repeats a hole_id")
    expected_count = prod(
        len(hole.allowed_operators) for hole in hole_plan.holes
    )
    if report.expected_assignment_count != expected_count:
        raise ValueError("OperatorResolutionReport expected assignment count is inconsistent")
    if report.passing_assignment_count != len(report.passing_assignments):
        raise ValueError("OperatorResolutionReport passing assignment count is inconsistent")
    if report.candidate_assignment_count > expected_count:
        raise ValueError("OperatorResolutionReport candidate assignment count exceeds the legal domain")
    if report.evaluated_assignment_count > report.candidate_assignment_count:
        raise ValueError("OperatorResolutionReport evaluated count exceeds materialized candidates")
    signatures = []
    for assignment in report.passing_assignments:
        assignment_temporal_resolution_by_node(
            hole_plan=hole_plan,
            assignment=assignment,
        )
        signature = tuple(sorted(assignment))
        if signature in signatures:
            raise ValueError("OperatorResolutionReport repeats a passing assignment")
        signatures.append(signature)

    resolutions = {item.hole_id: item for item in report.hole_resolutions}
    if len(resolutions) != len(report.hole_resolutions):
        raise ValueError("OperatorResolutionReport repeats a hole resolution")
    if set(resolutions) != set(constraints):
        raise ValueError("OperatorResolutionReport hole resolutions do not cover the hole plan")
    for hole_id, constraint in constraints.items():
        item = resolutions[hole_id]
        if (
            item.ir_node_id != constraint.ir_node_id
            or tuple(item.allowed_operators) != tuple(constraint.allowed_operators)
        ):
            raise ValueError("OperatorResolutionReport hole resolution is inconsistent")

    if require_final:
        if report.status not in {"complete", "not_required"}:
            raise ValueError(
                f"operator resolution is not final: status={report.status!r}"
            )
        if not report.authoritative or report.parent_ir_status != "approved":
            raise ValueError("operator resolution is not authoritative")
        if (
            report.candidate_assignment_count != expected_count
            or report.evaluated_assignment_count != expected_count
        ):
            raise ValueError("operator resolution did not exhaust the legal assignment domain")
        if len(report.passing_assignments) != 1:
            raise ValueError("operator resolution must contain exactly one passing assignment")
        selected = dict(report.passing_assignments[0])
        for hole_id, item in resolutions.items():
            if item.status != "complete" or item.selected_operator != selected[hole_id]:
                raise ValueError("OperatorResolutionReport final hole decision is inconsistent")
        expected_nodes = tuple(
            dict.fromkeys(hole.ir_node_id for hole in hole_plan.holes)
        )
        if set(report.resolved_ir_node_ids) != set(expected_nodes) or report.unresolved_ir_node_ids:
            raise ValueError("OperatorResolutionReport final node coverage is inconsistent")


def build_operator_trace_evidence(
    *,
    ir_version: str,
    hole_plan: CompletionHolePlan,
    traces: list[LabeledTrace],
    parent_ir_status: str = "approved",
) -> OperatorTraceEvidenceSet:
    """Partition reviewed traces without reusing IR-branch evidence."""

    if hole_plan.ir_version != ir_version:
        raise ValueError("CompletionHolePlan does not belong to the selected IR")
    forbidden = tuple(
        trace
        for trace in traces
        if trace.evidence_purpose == TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION
    )
    if forbidden:
        raise ValueError(
            "IR-branch discrimination traces reached operator resolution: "
            + ", ".join(trace.trace_id or trace.name for trace in forbidden)
        )
    return OperatorTraceEvidenceSet(
        ir_version=ir_version,
        hole_plan_version=hole_plan.skeleton_plan_version,
        discrimination_traces=tuple(
            trace
            for trace in traces
            if trace.evidence_purpose == TRACE_PURPOSE_OPERATOR_DISCRIMINATION
        ),
        baseline_traces=tuple(
            trace
            for trace in traces
            if trace.evidence_purpose == TRACE_PURPOSE_SEMANTIC_VALIDATION
        ),
        parent_ir_status=parent_ir_status,
    )


def evaluate_operator_resolution(
    *,
    completion_decisions: list[CompletionDecision],
    evaluations: list[FormulaConsistencyEvaluation],
    hole_plan: CompletionHolePlan,
    traces: list[LabeledTrace],
    strict_boundary: bool,
) -> OperatorResolutionReport:
    """Compatibility entry point for legacy callers with untyped traces.

    New runtime code must call :func:`evaluate_operator_resolution_from_evidence`.
    IR-branch-purpose traces are rejected even on this compatibility path.
    """

    if any(
        trace.evidence_purpose == TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION
        for trace in traces
    ):
        raise ValueError("IR-branch traces cannot be used as operator evidence")
    legacy = OperatorTraceEvidenceSet(
        ir_version=hole_plan.ir_version,
        hole_plan_version=hole_plan.skeleton_plan_version,
        discrimination_traces=tuple(
            replace_trace_purpose(
                trace,
                TRACE_PURPOSE_OPERATOR_DISCRIMINATION,
                ir_version=hole_plan.ir_version,
                hole_plan_version=hole_plan.skeleton_plan_version,
            )
            for trace in traces
        ),
    )
    return evaluate_operator_resolution_from_evidence(
        completion_decisions=completion_decisions,
        evaluations=evaluations,
        hole_plan=hole_plan,
        evidence=legacy,
        strict_boundary=strict_boundary,
    )


def evaluate_operator_resolution_from_evidence(
    *,
    completion_decisions: list[CompletionDecision],
    evaluations: list[FormulaConsistencyEvaluation],
    hole_plan: CompletionHolePlan,
    evidence: OperatorTraceEvidenceSet,
    strict_boundary: bool,
) -> OperatorResolutionReport:
    """Resolve operator assignments from purpose- and version-bound evidence."""

    del strict_boundary

    if evidence.ir_version != hole_plan.ir_version:
        raise ValueError("operator evidence IR version does not match CompletionHolePlan")
    if evidence.hole_plan_version != hole_plan.skeleton_plan_version:
        raise ValueError("operator evidence references a stale CompletionHolePlan")
    traces = list(evidence.traces)

    expected_holes = {hole.hole_id: hole for hole in hole_plan.holes}
    expected_assignment_count = prod(
        len(hole.allowed_operators) for hole in hole_plan.holes
    )
    valid_decisions: dict[
        tuple[tuple[str, str], ...], CompletionDecision
    ] = {}
    invalid_reasons: list[str] = []
    for decision in completion_decisions:
        signature = _validated_assignment_signature(
            decision=decision,
            expected_holes=expected_holes,
            ir_version=hole_plan.ir_version,
        )
        if signature is None:
            invalid_reasons.append(
                f"completion {decision.candidate_id or decision.formula} does not "
                "match the current CompletionHolePlan"
            )
            continue
        valid_decisions.setdefault(signature, decision)

    evaluation_by_formula = {
        str(evaluation.formula): evaluation for evaluation in evaluations
    }
    reviewed_decisions = {
        signature: decision
        for signature, decision in valid_decisions.items()
        if str(decision.formula) in evaluation_by_formula
    }
    passing = tuple(
        signature
        for signature, decision in sorted(reviewed_decisions.items())
        if _strictly_matches_reviewed_traces(
            evaluation_by_formula[str(decision.formula)],
            traces,
        )
    )
    candidate_assignment_count = len(valid_decisions)
    evaluated_assignment_count = len(reviewed_decisions)
    reasons: list[str] = list(dict.fromkeys(invalid_reasons))
    if candidate_assignment_count < expected_assignment_count:
        reasons.append(
            "not every IR-allowed joint temporal assignment was materialized: "
            f"generated={candidate_assignment_count}, "
            f"expected={expected_assignment_count}"
        )
    if evaluated_assignment_count < expected_assignment_count:
        reasons.append(
            "not every materialized temporal assignment was evaluated against "
            "the current reviewed trace set: "
            f"evaluated={evaluated_assignment_count}, "
            f"expected={expected_assignment_count}"
        )
    assignments_complete = (
        candidate_assignment_count == expected_assignment_count
        and evaluated_assignment_count == expected_assignment_count
    )
    trace_resolvable_holes = tuple(
        hole for hole in hole_plan.holes if len(hole.allowed_operators) > 1
    )
    if not assignments_complete:
        status = "incomplete"
    elif not trace_resolvable_holes:
        status = "not_required"
        if not passing:
            reasons.append(
                "the fixed assignment does not match every reviewed trace; "
                "final replay must fail"
            )
    elif not passing:
        status = "inconsistent"
        reasons.append("no temporal assignment strictly matches all reviewed traces")
    elif len(passing) > 1:
        status = "ambiguous"
        reasons.append(
            "multiple temporal assignments remain consistent with all reviewed traces"
        )
    else:
        status = "complete"
    if trace_resolvable_holes and not evidence.discrimination_traces:
        status = "incomplete"
        reasons.append(
            "operator-resolution holes require operator-purpose discrimination traces"
        )

    all_trace_ids = tuple(
        dict.fromkeys(trace.trace_id or trace.name for trace in traces)
    )
    passing_maps = [dict(signature) for signature in passing]
    hole_resolutions: list[HoleOperatorResolution] = []
    resolved_node_ids: list[str] = []
    unresolved_node_ids: list[str] = []
    for hole in hole_plan.holes:
        selected_values = {
            assignment[hole.hole_id]
            for assignment in passing_maps
            if hole.hole_id in assignment
        }
        selected_operator = (
            next(iter(selected_values))
            if len(selected_values) == 1
            else hole.allowed_operators[0]
            if status == "not_required" and len(hole.allowed_operators) == 1
            else ""
        )
        hole_complete = bool(selected_operator) and status in {
            "complete",
            "not_required",
        }
        rejection_trace_ids = _rejection_trace_ids_for_hole(
            hole_id=hole.hole_id,
            selected_operator=selected_operator,
            decisions=valid_decisions,
            evaluation_by_formula=evaluation_by_formula,
        )
        hole_resolutions.append(
            HoleOperatorResolution(
                hole_id=hole.hole_id,
                ir_node_id=hole.ir_node_id,
                allowed_operators=hole.allowed_operators,
                selected_operator=selected_operator,
                rejected_operators=tuple(
                    operator
                    for operator in hole.allowed_operators
                    if selected_operator and operator != selected_operator
                ),
                supporting_trace_ids=all_trace_ids if hole_complete else (),
                rejection_trace_ids=rejection_trace_ids,
                resolution_basis=(
                    "fixed_text_or_policy"
                    if hole_complete and len(hole.allowed_operators) == 1
                    else "trajectory_discriminated"
                    if hole_complete
                    else ""
                ),
                status="complete" if hole_complete else "unresolved",
            )
        )
        if hole_complete:
            resolved_node_ids.append(hole.ir_node_id)
        else:
            unresolved_node_ids.append(hole.ir_node_id)

    return OperatorResolutionReport(
        requirement_id=hole_plan.requirement_id,
        ir_version=hole_plan.ir_version,
        status=status,
        hole_resolutions=tuple(hole_resolutions),
        passing_assignment_count=len(passing),
        candidate_assignment_count=candidate_assignment_count,
        evaluated_assignment_count=evaluated_assignment_count,
        expected_assignment_count=expected_assignment_count,
        passing_assignments=passing,
        resolved_ir_node_ids=tuple(dict.fromkeys(resolved_node_ids)),
        unresolved_ir_node_ids=tuple(dict.fromkeys(unresolved_node_ids)),
        supporting_trace_ids=all_trace_ids if status == "complete" else (),
        reasons=tuple(dict.fromkeys(reasons)),
        evidence_purpose=TRACE_PURPOSE_OPERATOR_DISCRIMINATION,
        hole_plan_version=evidence.hole_plan_version,
        parent_ir_status=evidence.parent_ir_status,
        authoritative=(
            evidence.parent_ir_status == "approved" and status in {"complete", "not_required"}
        ),
    )


def replace_trace_purpose(
    trace: LabeledTrace,
    purpose: str,
    *,
    ir_version: str,
    hole_plan_version: str,
) -> LabeledTrace:
    """Explicit legacy adapter; never used by the authoritative runtime path."""

    from dataclasses import replace

    return replace(
        trace,
        evidence_purpose=purpose,
        ir_version=ir_version,
        parent_hole_plan_version=hole_plan_version,
    )


def _validated_assignment_signature(
    *,
    decision: CompletionDecision,
    expected_holes: dict,
    ir_version: str,
) -> tuple[tuple[str, str], ...] | None:
    if decision.ir_version != ir_version:
        return None
    assignments = dict(decision.assignments)
    if set(assignments) != set(expected_holes):
        return None
    for hole_id, operator in assignments.items():
        if operator not in expected_holes[hole_id].allowed_operators:
            return None
    closure = decision.closure_report
    if closure is None or getattr(closure, "status", "") != "passed":
        return None
    return tuple(sorted(assignments.items()))


def _strictly_matches_reviewed_traces(
    evaluation: FormulaConsistencyEvaluation,
    traces: list[LabeledTrace],
) -> bool:
    """Require a pointwise result for every reviewed evidence item."""

    return (
        bool(traces)
        and len(evaluation.trace_results) == len(traces)
        and all(result.matched for result in evaluation.trace_results)
    )


def select_operator_resolved_evaluation(
    *,
    report: OperatorResolutionReport,
    completion_decisions: list[CompletionDecision],
    evaluations: list[FormulaConsistencyEvaluation],
) -> FormulaConsistencyEvaluation | None:
    """Return the sole evaluation authorized by the resolution report.

    Scores and formula size are deliberately irrelevant.  The selected formula
    must come from the unique passing assignment recorded by exhaustive
    materialization, including the fixed-domain ``not_required`` case.
    """

    if report.status not in {"complete", "not_required"}:
        return None
    if len(report.passing_assignments) != 1:
        return None
    resolved = evaluations_for_passing_assignments(
        report=report,
        completion_decisions=completion_decisions,
        evaluations=evaluations,
    )
    if len(resolved) != 1:
        return None
    return resolved[0]


def evaluations_for_passing_assignments(
    *,
    report: OperatorResolutionReport,
    completion_decisions: list[CompletionDecision],
    evaluations: list[FormulaConsistencyEvaluation],
) -> list[FormulaConsistencyEvaluation]:
    """Map report assignments back to their complete-formula evaluations."""

    passing = set(report.passing_assignments)
    evaluation_by_formula = {
        str(evaluation.formula): evaluation for evaluation in evaluations
    }
    resolved: dict[str, FormulaConsistencyEvaluation] = {}
    for decision in completion_decisions:
        signature = tuple(sorted(decision.assignments))
        if signature not in passing:
            continue
        evaluation = evaluation_by_formula.get(str(decision.formula))
        if evaluation is not None:
            resolved[str(evaluation.formula)] = evaluation
    return list(resolved.values())


def _rejection_trace_ids_for_hole(
    *,
    hole_id: str,
    selected_operator: str,
    decisions: dict[tuple[tuple[str, str], ...], CompletionDecision],
    evaluation_by_formula: dict[str, FormulaConsistencyEvaluation],
) -> tuple[str, ...]:
    rejected: list[str] = []
    for signature, decision in decisions.items():
        assignment = dict(signature)
        if not selected_operator or assignment.get(hole_id) == selected_operator:
            continue
        evaluation = evaluation_by_formula.get(str(decision.formula))
        if evaluation is None:
            continue
        rejected.extend(
            result.trace_name
            for result in evaluation.trace_results
            if not result.matched
        )
    return tuple(dict.fromkeys(rejected))
