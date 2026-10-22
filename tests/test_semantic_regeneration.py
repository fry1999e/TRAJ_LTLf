from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agents.nlu_agent import NLUAgent, NLURepairExhaustedError
from agents.review_loop import ReviewDecision, run_semantic_review
from ap_preparation import ensure_ap_candidates, prepare_reviewed_ap_context
from batch_review import _build_batch_state, _regenerate_batch_item
from candidate_delivery import retain_generation_candidate
from faithful_ir import FaithfulIRValidationError, validate_faithful_ir
from rag.retriever import KeywordRetriever
from temporal_language import temporal_language_prompt_view
from tests.test_ap_preparation import Client, TEXT, context
from tests.test_candidate_delivery import pending


def timing_error():
    return NLURepairExhaustedError(
        "node ir-n4 timing 'eventually_strict' conflicts with source alias "
        "'eventually' -> 'eventually'; node ir-n4 has non-canonical scope "
        "metadata 'strict_suffix'",
        first_failure_kind="faithful_ir_validation_error", first_failure="timing mismatch",
        reason_code="HUMAN_REVIEW_REQUIRED", attempt=1,
    )


class SemanticRegenerationTests(unittest.TestCase):
    def test_feedback_uses_source_graph_and_preserves_both_timing_candidates(self):
        feedback = ["Check whether eventually includes the current step."]
        client = Client()
        agent = NLUAgent(KeywordRetriever([]), llm_client=client)
        with patch.object(agent, "_generate_with_llm", side_effect=AssertionError("full model IR")):
            result = agent.generate(TEXT, feedback_history=feedback)
        request = next(call for call in client.calls if call.get("task") == "propose_source_intent_graph")
        self.assertEqual(request["feedback_history"], feedback)
        self.assertTrue(request["ap_mentions"])
        self.assertNotIn("faithful_ir", request["required_output"])
        self.assertEqual({node.node_type for branch in result.candidate_ir_set.branches
                          for node in branch.faithful_ir.nodes if node.node_type.startswith("eventually")},
                         {"eventually", "eventually_strict"})
        for branch in result.candidate_ir_set.branches:
            validate_faithful_ir(branch.faithful_ir)
            self.assertFalse(branch.faithful_ir.provenance.approved)

    def test_model_rejection_and_real_regeneration_return_to_candidate_review(self):
        agent = NLUAgent(KeywordRetriever([]), llm_client=Client())
        decisions = iter([ReviewDecision(False, "Recheck the temporal reading."), ReviewDecision(True, "Source aligned.")])
        with tempfile.TemporaryDirectory() as folder, patch("builtins.input", side_effect=AssertionError("expert form")):
            semantic, records = run_semantic_review(TEXT, agent, records_dir=folder,
                structure_reviewer=lambda *_args: next(decisions))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]["issue_type"], "awaiting_ir_trajectory_adjudication")
        self.assertEqual(len(semantic.candidate_ir_set.branches), 2)
        self.assertIsNotNone(pending(semantic).provisional_candidate)

    def test_invalid_revision_keeps_valid_candidate_and_delivers_one_formula(self):
        initial = NLUAgent(KeywordRetriever([]), llm_client=Client()).generate(TEXT)
        agent = SimpleNamespace(generate=Mock(side_effect=[initial, timing_error()]), llm_enabled=False,
                                retriever=KeywordRetriever([]))
        with tempfile.TemporaryDirectory() as folder, patch("builtins.input", side_effect=AssertionError("expert form")):
            semantic, records = run_semantic_review(TEXT, agent, records_dir=folder,
                structure_reviewer=lambda *_args: ReviewDecision(False, "Recheck timing."))
            saved = json.loads(Path(records[-1]["record_path"]).read_text(encoding="utf-8"))
        result = pending(semantic)
        self.assertEqual(agent.generate.call_count, 2)
        self.assertEqual(saved["status"], "invalid_revision_rolled_back")
        self.assertIn("strict_suffix", saved["error"])
        self.assertFalse(saved["approval_granted"])
        self.assertIsNotNone(result.provisional_candidate)
        self.assertEqual(result.approval_status, "not_approved")
        self.assertIsNone(result.selected)

    def test_no_prior_candidate_does_not_fabricate_formula(self):
        error = timing_error()
        agent = SimpleNamespace(generate=Mock(side_effect=error))
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(NLURepairExhaustedError) as caught:
            run_semantic_review(TEXT, agent, records_dir=folder)
        self.assertIs(caught.exception, error)

    def test_stale_source_or_ap_context_cannot_be_reused(self):
        semantic = NLUAgent(KeywordRetriever([]), llm_client=Client()).generate(TEXT)
        changed_aps = ensure_ap_candidates(context(), Client()).context
        with redirect_stdout(io.StringIO()):
            changed_aps = prepare_reviewed_ap_context(changed_aps, Client(), interactive=True,
                                                     input_fn=lambda _prompt: "a").context
        for source, ap_context in ((TEXT + " Changed.", None), (TEXT, changed_aps)):
            with self.subTest(source=source, ap_context=ap_context is not None), self.assertRaises(NLURepairExhaustedError):
                retain_generation_candidate(source, semantic, timing_error(), ap_binding_context=ap_context)

    def test_invalid_previous_ir_still_fails_validation(self):
        semantic = NLUAgent(KeywordRetriever([]), llm_client=Client()).generate(TEXT)
        invalid = replace(semantic, faithful_ir=replace(semantic.faithful_ir, root_node_id="missing"))
        with self.assertRaises(FaithfulIRValidationError):
            retain_generation_candidate(TEXT, invalid, timing_error())

    def test_batch_invalid_revision_preserves_candidate_and_audit(self):
        client = Client()
        with redirect_stdout(io.StringIO()):
            approved = prepare_reviewed_ap_context(context(), client, interactive=True,
                                                  input_fn=lambda _prompt: "a").context
        initial = NLUAgent(KeywordRetriever([]), llm_client=client).generate(TEXT, ap_binding_context=approved)
        state = _build_batch_state(index=1, requirement=TEXT, llm_client=client, ap_binding_context=approved)
        state.nlu_agent.generate = Mock(side_effect=[initial, timing_error()])
        def stop_after_candidates(item, **_kwargs):
            item.semantic_policy_pending = {"status": "provisional_fallback"}
        with tempfile.TemporaryDirectory() as folder, \
                patch("nonexpert_review.review_semantic_with_model", return_value={"approved": False, "reason": "Recheck timing."}), \
                patch("batch_review._adjudicate_ir_candidate_semantic", side_effect=stop_after_candidates), \
                patch("builtins.input", side_effect=AssertionError("repeat AP or expert form")):
            _regenerate_batch_item(state, checkpoint_dir=None, round_index=1, item_index=1, total_items=1,
                review_mode="manual", records_dir=folder, allow_degraded=False)
            audit = next(record for record in state.review_records if record.get("status") == "invalid_revision_rolled_back")
            self.assertTrue(Path(audit["record_path"]).exists())
        self.assertEqual(state.nlu_agent.generate.call_count, 2)
        self.assertEqual(state.semantic.semantic_status, "needs_semantic_review")
        self.assertIsNotNone(pending(state.semantic).provisional_candidate)

    def test_temporal_prompt_does_not_mix_witness_ranges_with_ir_scope(self):
        prompt = temporal_language_prompt_view()
        self.assertEqual(prompt["eventually_strict"]["scope"], "current_position")
        self.assertEqual(prompt["eventually_strict"]["witness_range"], "strict_suffix")
        self.assertEqual(prompt["eventually_strict"]["operator"], "XF")
        self.assertTrue(all(value["scope"] in {"current_position", "inherited", "all_positions"}
                            for value in prompt.values()))


if __name__ == "__main__":
    unittest.main()
