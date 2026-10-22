from __future__ import annotations

from dataclasses import replace
import unittest

from agents.consistency_agent import TrajectoryConsistencyAgent
from data_examples import (
    LabeledTrace,
    TRACE_PURPOSE_OPERATOR_DISCRIMINATION,
)
from faithful_ir import faithful_ir_from_mapping
from formula_closure import (
    build_strict_formula_closure_policy,
    validate_formula_closure,
)
from ir_compiler import (
    compile_completion_hole_plan,
    compile_constrained_skeleton_plan,
    compile_ir_to_formula,
)
from operator_resolution import (
    assignment_temporal_resolution_by_node,
    build_operator_trace_evidence,
    evaluate_operator_resolution_from_evidence,
    resolved_temporal_resolution_by_node,
    select_operator_resolved_evaluation,
)
from schemas import (
    ApprovedSemanticIR,
    CompletionDecision,
    IRCoverageReport,
    LabeledTraceEvidenceSet,
    OperatorResolutionReport,
    SemanticStructure,
)
from selection_rules import (
    completion_decision_reject_reasons,
    ir_structure_reject_reasons,
    select_with_ir_coverage_gate,
)
from semantic_node_contracts import node_contract_fingerprint


def _trace_resolvable_ir():
    source = "Eventually p"
    return faithful_ir_from_mapping(
        {
            "atomic_propositions": [
                {
                    "ap_id": "a",
                    "name": "p",
                    "source_span": [11, 12],
                    "evidence_text": "p",
                }
            ],
            "root_node_id": "t",
            "nodes": [
                {
                    "node_id": "a",
                    "node_type": "ap_condition",
                    "parameters": {"ap_name": "p"},
                    "source_span": [11, 12],
                    "evidence_text": "p",
                },
                {
                    "node_id": "t",
                    "node_type": "eventually",
                    "children": ["a"],
                    "allowed_semantic_readings": [
                        "eventually",
                        "eventually_strict",
                    ],
                    "temporal_resolution": "trace_resolvable",
                    "source_span": [0, 12],
                    "evidence_text": source,
                },
            ],
            "node_contract_fingerprint": node_contract_fingerprint(),
        },
        source_text=source,
    )


def _approved_semantic(ir) -> SemanticStructure:
    return SemanticStructure(
        trigger="p",
        response="p",
        temporal_keyword="eventually",
        scope_keyword="",
        pattern="eventually",
        aps=("p",),
        faithful_ir=ir,
        approved_semantic_ir=ApprovedSemanticIR(
            ir_version=ir.ir_version,
            approved_ir_branch_id="branch-1",
            nl_text=ir.source_text,
            approved_ap_mapping=(("p", "p"),),
            labeled_trace_evidence_ids=("trace-now",),
            approval_record="test-approval",
            approved=True,
            evidence_status="complete",
        ),
        labeled_trace_evidence=LabeledTraceEvidenceSet(
            requirement_id=ir.requirement_id,
            ir_version=ir.ir_version,
            trace_ids=("trace-now",),
            positive_trace_ids=("trace-now",),
            boundary_trace_ids=("trace-now",),
            status="complete",
        ),
    )


class TrajLTLfFinalizationTests(unittest.TestCase):
    def test_unique_trace_resolved_assignment_reaches_final_gate(self) -> None:
        ir = _trace_resolvable_ir()
        semantic = _approved_semantic(ir)
        skeleton_plan = compile_constrained_skeleton_plan(ir)
        hole_plan = compile_completion_hole_plan(ir, skeleton_plan)
        self.assertEqual(len(hole_plan.holes), 1)
        hole = hole_plan.holes[0]

        trace = LabeledTrace(
            name="current-state witness",
            kind="boundary",
            trace=[{"p"}],
            expected=True,
            reason="F p is true now while strong-future X F p is false",
            trace_id="trace-now",
            covered_ir_node_ids=("t",),
            ir_version=ir.ir_version,
            evidence_purpose=TRACE_PURPOSE_OPERATOR_DISCRIMINATION,
            parent_hole_plan_version=hole_plan.skeleton_plan_version,
            annotation_locked=True,
        )
        consistency = TrajectoryConsistencyAgent(retriever=None)
        decisions: list[CompletionDecision] = []
        evaluations = []
        for operator in hole.allowed_operators:
            assignment = ((hole.hole_id, operator),)
            resolution = assignment_temporal_resolution_by_node(
                hole_plan=hole_plan,
                assignment=assignment,
            )
            formula = compile_ir_to_formula(
                ir,
                temporal_resolution_by_node=resolution,
            )
            closure = validate_formula_closure(
                formula,
                build_strict_formula_closure_policy(
                    ir,
                    semantic.approved_semantic_ir,
                ),
                ir_version=ir.ir_version,
            )
            evaluation = replace(
                consistency.evaluate_formula(
                    formula,
                    [trace],
                    candidate_id=f"candidate-{operator}",
                ),
                ir_version=ir.ir_version,
                closure_report=closure,
            )
            evaluations.append(evaluation)
            decisions.append(
                CompletionDecision(
                    formula=formula,
                    assignments=assignment,
                    boundary_match=evaluation.bound_match,
                    pos_sat=evaluation.pos_sat,
                    neg_rej=evaluation.neg_rej,
                    score=evaluation.score,
                    candidate_id=evaluation.candidate_id,
                    ir_version=ir.ir_version,
                    closure_report=closure,
                    boundary_metric_available=True,
                    boundary_trace_count=1,
                )
            )

        # Every materialized completion is a certified member of the approved
        # IR domain before trajectory evidence selects one of them.
        for decision in decisions:
            structural_reasons = tuple(
                reason
                for reason in completion_decision_reject_reasons(
                    decision,
                    strict_boundary=False,
                    semantic=semantic,
                )
                if "PosSat=" not in reason and "BoundMatch=" not in reason
            )
            self.assertEqual(structural_reasons, ())

        evidence = build_operator_trace_evidence(
            ir_version=ir.ir_version,
            hole_plan=hole_plan,
            traces=[trace],
            parent_ir_status="approved",
        )
        report = evaluate_operator_resolution_from_evidence(
            completion_decisions=decisions,
            evaluations=evaluations,
            hole_plan=hole_plan,
            evidence=evidence,
            strict_boundary=True,
        )
        self.assertEqual(report.status, "complete")
        self.assertTrue(report.authoritative)
        self.assertEqual(report.passing_assignment_count, 1)

        selected = select_operator_resolved_evaluation(
            report=report,
            completion_decisions=decisions,
            evaluations=evaluations,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(str(selected.formula), "F p")

        winning_resolution = resolved_temporal_resolution_by_node(
            report=report,
            hole_plan=hole_plan,
        )
        self.assertEqual(winning_resolution, {"t": "F"})
        self.assertEqual(
            ir_structure_reject_reasons(
                selected.formula,
                semantic,
                temporal_resolution_by_node=winning_resolution,
            ),
            (),
        )

        coverage = IRCoverageReport(
            requirement_id=ir.requirement_id,
            ir_version=ir.ir_version,
            status="complete",
            covered_ir_node_ids=("a", "t"),
            structure_mapping_complete=True,
            version_consistent=True,
        )
        outcome = select_with_ir_coverage_gate(
            [selected],
            strict_boundary=True,
            semantic=semantic,
            coverage=coverage,
            operator_resolution=report,
            temporal_resolution_by_node=winning_resolution,
        )
        self.assertEqual(outcome.status, "final")
        self.assertTrue(outcome.automatic_finalization_allowed)
        self.assertEqual(str(outcome.selected.formula), "F p")

    def test_final_gate_rejects_non_authoritative_resolution(self) -> None:
        ir = _trace_resolvable_ir()
        semantic = _approved_semantic(ir)
        formula = compile_ir_to_formula(
            ir,
            temporal_resolution_by_node={"t": "F"},
        )
        evaluation = replace(
            TrajectoryConsistencyAgent(retriever=None).evaluate_formula(
                formula,
                [
                    LabeledTrace(
                        name="now",
                        kind="boundary",
                        trace=[{"p"}],
                        expected=True,
                        reason="fixture",
                    )
                ],
            ),
            ir_version=ir.ir_version,
        )
        report = OperatorResolutionReport(
            requirement_id=ir.requirement_id,
            ir_version=ir.ir_version,
            status="complete",
            passing_assignment_count=1,
            candidate_assignment_count=2,
            evaluated_assignment_count=2,
            expected_assignment_count=2,
            passing_assignments=((('H-t', 'F'),),),
            hole_plan_version=compile_constrained_skeleton_plan(ir).plan_version,
            authoritative=False,
        )
        outcome = select_with_ir_coverage_gate(
            [evaluation],
            strict_boundary=True,
            semantic=semantic,
            coverage=IRCoverageReport(
                requirement_id=ir.requirement_id,
                ir_version=ir.ir_version,
                status="complete",
            ),
            operator_resolution=report,
            temporal_resolution_by_node={"t": "F"},
        )
        self.assertNotEqual(outcome.status, "final")
        self.assertIn("operator resolution is not authoritative", outcome.reasons)


if __name__ == "__main__":
    unittest.main()
