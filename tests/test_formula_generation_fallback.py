from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agents.nlu_agent import NLUAgent
from agents.review_loop import ReviewDecision, run_semantic_review
from ap_preparation import APPreparationError, prepare_reviewed_ap_context
from batch_review import _build_batch_state, _regenerate_batch_item
from batch_review import _restore_batch_checkpoints
from batch_pipeline import run_batch_traj_ltlf_pipeline
from checkpointing import semantic_from_checkpoint, save_needs_review_checkpoint
from candidate_delivery import recover_generation_candidate
from faithful_ir import FaithfulIRValidationError
from formula_generation_fallback import generate_formula_fallback, is_formula_generation_fallback
from ir_compiler import compile_ir_to_formula
from ir_proposal_router import IRProposalRouter
from pipeline import format_pipeline_summary, run_traj_ltlf_pipeline
from pre_ir_review import PreIRArtifacts
from rag.retriever import KeywordRetriever
from tests.test_ap_preparation import Client, TEXT, context, graph, segmentation
from tests.test_candidate_delivery import pending
from typed_source_relation_graph import source_graph_response_schema
from llm.schema_validation import validate_response_schema


FORMULA = "G(request_occurs -> F grant_occurs)"


class FallbackClient(Client):
    model = "fixture-main-generator"
    def __init__(self, formulas=None):
        broken = graph()
        broken["source_intent_graph"]["nodes"][0]["source_span"] = [9, 16]
        super().__init__(graph_replies=[broken])
        self.formulas = list(formulas or [FORMULA])

    def complete_json(self, system, payload, **kwargs):
        if payload.get("task") == "generate_bound_formula_fallback":
            self.calls.append(deepcopy(payload))
            reply = self.formulas.pop(0) if len(self.formulas) > 1 else self.formulas[0]
            return {"formula": reply, "reason": "Use current-or-future eventuality for this candidate."}
        return super().complete_json(system, payload, **kwargs)


def approved_context(client=None, seg=None):
    with redirect_stdout(io.StringIO()):
        return prepare_reviewed_ap_context(context(), client or Client(), segmentation=seg,
            interactive=True, input_fn=lambda _prompt: "a").context


def graph_error():
    return APPreparationError("AP_GRAPH_BINDING_REVIEW_REQUIRED", "fixture graph AP offset mismatch")


class FormulaGenerationFallbackTests(unittest.TestCase):
    def test_ap_names_replace_offset_arithmetic_and_schema_accepts_it(self):
        proposal = graph()
        for node, name in zip(proposal["source_intent_graph"]["nodes"][:2], ["request_occurs", "grant_occurs"]):
            node["ap_name"] = name
            node.pop("source_span")
        client = Client(graph_replies=[proposal])
        ap_context = approved_context(client)
        validate_response_schema(proposal, source_graph_response_schema(ap_context))
        result = IRProposalRouter().propose(TEXT, ap_context, proposal_client=client, feedback_history=["Regenerate graph."])
        self.assertTrue(result.candidates)
        self.assertEqual({node.source_span for node in result.graph.nodes if node.kind == "AP"}, {(9, 23), (25, 30)})
        self.assertFalse(any(call.get("task") == "repair_source_ap_spans" for call in client.calls))

    def test_named_ap_with_wrong_offset_is_bound_by_backend(self):
        proposal = graph()
        proposal["source_intent_graph"]["nodes"][0].update(ap_name="request_occurs", source_span=[0, 1])
        client = Client(graph_replies=[proposal])
        result = IRProposalRouter().propose(TEXT, approved_context(client), proposal_client=client,
            feedback_history=["Recheck source."])
        self.assertTrue(result.candidates)
        self.assertEqual(next(node for node in result.graph.nodes if node.canonical_name == "request_occurs").source_span, (9, 23))

    def test_existing_legal_candidate_survives_ap_binding_revision_failure(self):
        ap_context = approved_context()
        initial = NLUAgent(KeywordRetriever([]), llm_client=Client()).generate(TEXT, ap_binding_context=ap_context)
        agent = SimpleNamespace(generate=Mock(side_effect=[initial, graph_error()]), llm_enabled=True,
                                llm_client=FallbackClient(), retriever=KeywordRetriever([]))
        with tempfile.TemporaryDirectory() as folder:
            semantic, records = run_semantic_review(TEXT, agent, ap_binding_context=ap_context,
                records_dir=folder, structure_reviewer=lambda *_args: ReviewDecision(False, "Recheck source."))
        self.assertEqual(records[-1]["status"], "invalid_revision_rolled_back")
        self.assertFalse(agent.llm_client.calls)
        self.assertIsNotNone(pending(semantic).provisional_candidate)

    def test_explicit_ap_cancellation_is_not_overridden_by_fallback(self):
        ap_context = approved_context()
        initial = NLUAgent(KeywordRetriever([]), llm_client=Client()).generate(TEXT, ap_binding_context=ap_context)
        error = APPreparationError("AP_GROUNDING_REVIEW_REQUIRED", "AP review cancelled")
        with self.assertRaises(APPreparationError) as caught:
            recover_generation_candidate(TEXT, initial, error, client=FallbackClient(), ap_binding_context=ap_context)
        self.assertIs(caught.exception, error)

    def test_first_graph_failure_generates_one_formula_with_exact_ap_authority(self):
        client = FallbackClient()
        ap_context = approved_context(client)
        semantic, record = generate_formula_fallback(TEXT, client, ap_context, graph_error(), surface_segmentation=segmentation())
        result = pending(semantic)
        self.assertEqual(str(result.provisional_candidate.formula), FORMULA)
        self.assertEqual(semantic.faithful_ir.ap_context_fingerprint, ap_context.context_fingerprint)
        self.assertFalse(record["source_fidelity_proven"])
        self.assertFalse(semantic.faithful_ir.provenance.approved)
        self.assertFalse(semantic.persistent_rag_allowed)
        self.assertEqual(result.approval_status, "not_approved")

    def test_fallback_repairs_unknown_ap_and_syntax_before_output(self):
        client = FallbackClient(["G(invented -> F grant_occurs)", "G(request_occurs ->)", FORMULA])
        semantic, record = generate_formula_fallback(TEXT, client, approved_context(), graph_error())
        self.assertEqual(record["attempts"], 3)
        self.assertEqual(len(record["rejected_proposals"]), 2)
        self.assertEqual(str(compile_ir_to_formula(semantic.faithful_ir)), FORMULA)
        self.assertIn("validation_error", client.calls[-1])

    def test_invalid_fallback_does_not_fake_a_true_placeholder(self):
        client = FallbackClient(["true"])
        with self.assertRaises(FaithfulIRValidationError) as caught:
            generate_formula_fallback(TEXT, client, approved_context(), graph_error())
        self.assertEqual(caught.exception.code, "FORMULA_FALLBACK_GENERATION_FAILED")
        self.assertEqual(len(caught.exception.diagnostics["failed_attempts"]), 3)

    def test_fallback_supports_a_different_requirement_with_negation_and_until(self):
        source = "The alarm must remain off until the reset occurs."
        extraction = Client(ap_replies=[{"aps": [
            {"canonical_name": "alarm_on", "evidence_texts": ["alarm"]},
            {"canonical_name": "reset_occurs", "evidence_texts": ["reset occurs"]},
        ]}])
        with redirect_stdout(io.StringIO()):
            ap_context = prepare_reviewed_ap_context(context(source), extraction,
                interactive=True, input_fn=lambda _prompt: "a").context
        semantic, _record = generate_formula_fallback(source, FallbackClient(["!alarm_on U reset_occurs"]),
            ap_context, graph_error())
        self.assertEqual(str(compile_ir_to_formula(semantic.faithful_ir)), "!alarm_on U reset_occurs")
        self.assertEqual(set(semantic.faithful_ir.ap_names), {"alarm_on", "reset_occurs"})

    def test_semantic_and_candidate_checkpoint_preserve_unapproved_interpretation(self):
        semantic, _record = generate_formula_fallback(TEXT, FallbackClient(["G(request_occurs -> X F grant_occurs)"]),
            approved_context(), graph_error(), surface_segmentation=segmentation())
        restored = semantic_from_checkpoint(json.loads(json.dumps(asdict(semantic))))
        self.assertTrue(is_formula_generation_fallback(restored))
        self.assertEqual(restored.faithful_ir.provenance.extraction_mode, "working_interpretation_v1")
        self.assertEqual(str(pending(restored).provisional_candidate.formula), "G(request_occurs -> X F grant_occurs)")
        self.assertEqual(restored.faithful_ir.repair_history[-1]["stage"], "formula_generation_fallback")

    def test_single_manual_pipeline_recovers_repeated_graph_ap_failure(self):
        client = FallbackClient()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                patch("pipeline.run_pre_ir_review", return_value=PreIRArtifacts(segmentation(), None)), \
                patch("builtins.input", return_value="a") as read:
            result = run_traj_ltlf_pipeline(TEXT, llm_client=client, llm_mode="api", review_mode="manual",
                experiment_mode="human", approved_memory_file=None, records_dir=folder)
            audits = [json.loads(path.read_text(encoding="utf-8")) for path in Path(folder).glob("*.json")]
        self.assertEqual(read.call_count, 1)
        self.assertEqual(str(result.provisional_candidate.formula), FORMULA)
        self.assertNotIn("needs_review", format_pipeline_summary(result))
        self.assertTrue(any(record.get("stage") == "formula_generation_fallback" for record in audits))

    def test_batch_generation_uses_same_fallback_without_repeat_ap_confirmation(self):
        client = FallbackClient()
        state = _build_batch_state(index=1, requirement=TEXT, llm_client=client, ap_binding_context=approved_context(client))
        with tempfile.TemporaryDirectory() as folder, patch("builtins.input", side_effect=AssertionError("repeat review")):
            _regenerate_batch_item(state, checkpoint_dir=None, round_index=1, item_index=1, total_items=1,
                review_mode="manual", records_dir=folder, allow_degraded=False)
        self.assertTrue(is_formula_generation_fallback(state.semantic))
        self.assertEqual(state.semantic_policy_pending["status"], "candidate_generated")
        self.assertEqual(str(pending(state.semantic).provisional_candidate.formula), FORMULA)

    def test_batch_file_output_and_resume_keep_one_formula_without_new_model_calls(self):
        from pre_ir_review import complete_plain_reading_with_model
        from tests.test_nonexpert_review import AssistanceClient, proposal
        client = FallbackClient()
        seg = segmentation()
        reading = complete_plain_reading_with_model(TEXT, seg, proposal(seg), AssistanceClient())
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                patch("batch_review.run_pre_ir_review", return_value=PreIRArtifacts(seg, reading,
                    effective_source_text=TEXT)), \
                patch("builtins.input", side_effect=["a"]) as read:
            common = dict(llm_client=client, llm_mode="api", review_mode="manual", experiment_mode="human",
                records_dir=str(Path(folder) / "reviews"), checkpoint_dir=str(Path(folder) / "checkpoints"),
                classified_output_dir=str(Path(folder) / "outputs"), approved_memory_file=None)
            results = run_batch_traj_ltlf_pipeline([TEXT], **common)
            calls = len(client.calls)
            resumed = run_batch_traj_ltlf_pipeline([TEXT], **common)
            summary = (Path(folder) / "outputs" / "final_candidates.txt").read_text(encoding="utf-8")
        self.assertEqual(len(client.calls), calls)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(str(results[0].result.provisional_candidate.formula), FORMULA)
        self.assertEqual(str(resumed[0].result.provisional_candidate.formula), FORMULA)
        self.assertEqual(summary.count(FORMULA), 1)
        self.assertNotIn("needs_review", summary)

    def test_old_ap_graph_failure_checkpoint_retries_with_new_recovery(self):
        ap_context = approved_context()
        state = _build_batch_state(index=1, requirement=TEXT, llm_client=FallbackClient(), ap_binding_context=ap_context)
        with tempfile.TemporaryDirectory() as folder:
            save_needs_review_checkpoint(folder, requirement_id="REQ-001", stage="semantic",
                reason_code="AP_GRAPH_BINDING_REVIEW_REQUIRED", message="old graph offset failure",
                suggested_rerun_stage="semantic", details={"ap_context_fingerprint": ap_context.context_fingerprint})
            _restore_batch_checkpoints(states=[state], checkpoint_dir=folder, force_rerun_stages={}, progress=None)
        self.assertIsNone(state.needs_review_error)
        self.assertTrue(state.regenerate_semantic)


if __name__ == "__main__":
    unittest.main()
