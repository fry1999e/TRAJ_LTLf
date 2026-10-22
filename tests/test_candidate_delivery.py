from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agents.nlu_agent import NLUAgent
from candidate_delivery import deliver_formula_candidate
from data_examples import LabeledTrace
from faithful_ir import compute_ir_version, validate_faithful_ir
from ir_ambiguity_adjudication import adjudicate_ir_candidate_branches
from pipeline import _semantic_review_pending_result, format_pipeline_summary
from rag.retriever import KeywordRetriever
from schemas import ExperimentConfig, TraceCoverageReport
from semantic_freezing import attach_candidate_ir_set
from tests.test_ap_preparation import Client, TEXT


def unresolved_semantic():
    semantic = NLUAgent(KeywordRetriever([]), llm_client=Client()).generate(TEXT)
    ir = semantic.faithful_ir
    nodes = tuple(replace(node, allowed_semantic_readings=("eventually", "eventually_strict"),
                          temporal_resolution="trace_resolvable") if node.node_type == "eventually" else node
                  for node in ir.nodes)
    ir = replace(ir, nodes=nodes, ir_version="")
    ir = replace(ir, ir_version=compute_ir_version(ir))
    validate_faithful_ir(ir)
    return replace(semantic, faithful_ir=ir, candidate_ir_set=None, semantic_status="needs_semantic_review")


def pending(semantic, traces=()):
    return _semantic_review_pending_result(requirement=TEXT, semantic=semantic, traces=list(traces),
        review_records=[], trace_coverage=TraceCoverageReport(status="incomplete"), strict_boundary=True,
        review_mode="manual", experiment_config=ExperimentConfig(mode="human"), min_passed_skeletons=5)


class CandidateDeliveryTests(unittest.TestCase):
    def test_operator_range_is_materialized_before_promotion(self):
        source = unresolved_semantic()
        result = attach_candidate_ir_set(source, TEXT, max_branch_count=4)
        self.assertEqual(len(result.candidate_ir_set.branches), 2)
        self.assertEqual(result.candidate_ir_set.selection_status, "unresolved")
        self.assertEqual({node.node_type for branch in result.candidate_ir_set.branches
                          for node in branch.faithful_ir.nodes if node.node_type.startswith("eventually")},
                         {"eventually", "eventually_strict"})
        self.assertTrue(all(not branch.faithful_ir.provenance.approved for branch in result.candidate_ir_set.branches))

    def test_two_inconclusive_rounds_still_deliver_one_formula(self):
        source = attach_candidate_ir_set(unresolved_semantic(), TEXT, max_branch_count=4)
        with patch("ir_ambiguity_adjudication.synthesize_policy_discriminating_queries", return_value=[]):
            outcome = adjudicate_ir_candidate_branches(TEXT, source, annotation_provider=lambda **_kwargs: [])
        self.assertEqual(outcome.status, "provisional_fallback")
        result = pending(outcome.semantic)
        self.assertIsNotNone(result.provisional_candidate)
        self.assertEqual(result.selected_branch_id, outcome.selected_branch_id)
        self.assertIsNone(result.selected)
        self.assertFalse(result.automatic_finalization_allowed)
        self.assertIsNone(result.final_approval)
        self.assertEqual(result.learning_status, "writeback_blocked")
        summary = format_pipeline_summary(result)
        self.assertIn(str(result.provisional_candidate.formula), summary)
        self.assertNotIn("needs_review", summary)
        self.assertEqual(summary.count("最终候选:"), 1)

    def test_pending_operator_range_also_has_a_valid_fallback(self):
        result = pending(unresolved_semantic())
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.selection_status, "fallback_selected")
        self.assertEqual(result.approval_status, "not_approved")
        self.assertIs(deliver_formula_candidate(result), result)

    def test_human_judgment_ranks_the_fallback_without_relabeling(self):
        source = unresolved_semantic()
        for fits in (True, False):
            trace = LabeledTrace(name="one-step", kind="boundary", trace=[{"request_occurs", "grant_occurs"}],
                                 expected=fits, reason="fixture human choice", adjudicated_label="SAT" if fits else "UNSAT", label_source="human")
            result = pending(source, [trace])
            self.assertEqual(result.provisional_candidate.trace_results[0].matched, True)
            self.assertEqual(result.traces[0].expected, fits)
            self.assertEqual(result.review_records[-1]["human_trace_count"], 1)

    def test_model_answer_is_not_counted_as_a_human_judgment(self):
        trace = LabeledTrace(name="model", kind="boundary", trace=[{"request_occurs", "grant_occurs"}],
                             expected=False, reason="fixture model answer", adjudicated_label="UNSAT", label_source="model_assistant")
        result = pending(unresolved_semantic(), [trace])
        self.assertEqual(result.review_records[-1]["human_trace_count"], 0)
        self.assertEqual(result.traces[0].adjudicated_label, "UNSAT")

    def test_batch_output_contains_formula_and_keeps_unapproved_provenance(self):
        from batch_pipeline import BatchPipelineItemResult, write_final_unique_batch_outputs
        result = pending(unresolved_semantic())
        item = BatchPipelineItemResult(requirement_id="REQ-001", requirement=TEXT, result=result,
                                       summary_text=format_pipeline_summary(result), classification="needs_review")
        with tempfile.TemporaryDirectory() as folder:
            write_final_unique_batch_outputs([item], output_dir=folder)
            summary = (Path(folder) / "final_candidates.txt").read_text(encoding="utf-8")
            manifest = json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn(str(result.provisional_candidate.formula), summary)
        self.assertNotIn("needs_review", summary)
        self.assertEqual(manifest["items"][0]["result_status"], "fallback_selected")
        self.assertEqual(manifest["items"][0]["approval_status"], "not_approved")

    def test_full_batch_routes_a_pending_valid_ir_into_formula_fallback(self):
        from ap_preparation import prepare_reviewed_ap_context
        from batch_pipeline import run_batch_traj_ltlf_pipeline
        from tests.test_ap_preparation import context
        with redirect_stdout(io.StringIO()):
            approved = prepare_reviewed_ap_context(context(), Client(), interactive=True, input_fn=lambda _prompt: "a")
        semantic = NLUAgent(KeywordRetriever([]), llm_client=Client()).generate(TEXT, ap_binding_context=approved.context)
        item = SimpleNamespace(requirement_id="REQ-001", requirement=TEXT, semantic=semantic, traces=[], review_records=[],
            ap_binding_context=approved.context, trace_coverage=TraceCoverageReport(status="incomplete"),
            needs_review_error={"reason_code": "SEMANTIC_READINESS_PENDING"}, semantic_policy_pending=None,
            pre_ir_disposition=None, retriever=KeywordRetriever([]))
        with tempfile.TemporaryDirectory() as folder, patch("batch_pipeline.run_batch_semantic_trace_review", return_value=[item]):
            result = run_batch_traj_ltlf_pipeline([TEXT], review_mode="manual", experiment_mode="human", llm_mode="api",
                llm_client=Client(), approved_memory_file=None, records_dir=folder)[0]
        self.assertIsNotNone(result.result.provisional_candidate)
        self.assertNotIn("needs_review", result.summary_text)


if __name__ == "__main__":
    unittest.main()
