from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agents.nlu_agent import NLUAgent
from ap_preparation import APPreparationError, prepare_reviewed_ap_context
from candidate_delivery import deliver_formula_candidate
from execution_observer import observe_execution
from human_review_session import semantic_review_budget
from llm.langchain_client import LangChainJSONClient
from llm.overlap_client import OverlapJSONClient, overlap_client
from nonexpert_review import review_traces_with_simple_choices
from pre_ir_review import PreIRReviewConfig, run_pre_ir_review
from rag.retriever import KeywordRetriever
from schemas import TraceQuery
from semantic_adjudication import build_trace_adjudication_brief
from semantic_freezing import attach_candidate_ir_set
from tests.test_ap_preparation import APS, Client, TEXT, context, segmentation
from tests.test_candidate_delivery import pending, unresolved_semantic
from tests.test_nonexpert_review import AssistanceClient, proposal


class ProposalOverlapTests(unittest.TestCase):
    def test_existing_ap_candidates_do_not_trigger_an_unused_prefetch(self):
        client = OverlapJSONClient(AssistanceClient())
        with redirect_stdout(io.StringIO()), patch("builtins.input", return_value="a"), \
             patch.object(client, "prefetch_aps") as prefetch, \
             patch("pre_ir_review.generate_plain_reading_outcome",
                   side_effect=lambda _agent, _source, seg, **_kw: SimpleNamespace(review=proposal(seg))):
            run_pre_ir_review(requirement_id="REQ-001", requirement=TEXT,
                retriever=KeywordRetriever([]), llm_client=client, interactive=True,
                ap_candidates_available=True,
                config=PreIRReviewConfig(segmentation_review="manual", plain_reading_review="manual"))
        prefetch.assert_not_called()

    def test_local_surface_review_overlaps_ap_request_with_reading_check(self):
        started, release = Event(), Event()

        class ConcurrentClient(AssistanceClient):
            def complete_json(self, system, payload, **kwargs):
                if payload.get("task") == "extract_source_ap_candidates":
                    self.calls.append(deepcopy(payload))
                    started.set()
                    if not release.wait(3):
                        raise AssertionError("AP extraction was serialized after reading")
                    return deepcopy(APS)
                if not started.wait(3):
                    raise AssertionError("AP extraction did not start before reading")
                release.set()
                return super().complete_json(system, payload, **kwargs)

        base = ConcurrentClient()
        client = OverlapJSONClient(base)
        try:
            with redirect_stdout(io.StringIO()), patch("builtins.input", return_value="a"), \
                 patch("pre_ir_review.generate_plain_reading_outcome",
                       side_effect=lambda _agent, _source, seg, **_kw: SimpleNamespace(review=proposal(seg))):
                artifacts = run_pre_ir_review(requirement_id="REQ-001", requirement=TEXT,
                    retriever=KeywordRetriever([]), llm_client=client, interactive=True,
                    config=PreIRReviewConfig(segmentation_review="manual", plain_reading_review="manual"))
                ap = prepare_reviewed_ap_context(context(), client,
                    segmentation=artifacts.surface_segmentation, interactive=False)
        finally:
            release.set()
        self.assertEqual(sum(call.get("task") == "extract_source_ap_candidates" for call in base.calls), 1)
        self.assertFalse(any(call.get("task") == "lossless_surface_segmentation" for call in base.calls))
        self.assertEqual(artifacts.surface_segmentation.status, "surface_reviewed")
        self.assertFalse(ap.context.has_authoritative_catalog)

    def test_graph_is_requested_during_ap_input_and_reused_after_confirmation(self):
        started, release = Event(), Event()

        class BlockingGraphClient(Client):
            def complete_json(self, system, payload, **kwargs):
                if payload.get("task") == "propose_source_intent_graph":
                    started.set()
                    if not release.wait(3):
                        raise AssertionError("Graph was not overlapped with human input")
                return super().complete_json(system, payload, **kwargs)

        base = BlockingGraphClient()
        client = OverlapJSONClient(base)

        def approve(_prompt):
            self.assertTrue(started.wait(3))
            release.set()
            return "a"

        try:
            with redirect_stdout(io.StringIO()):
                ap = prepare_reviewed_ap_context(context(), client, segmentation=segmentation(),
                    interactive=True, input_fn=approve)
                semantic = NLUAgent(KeywordRetriever([]), llm_client=client).generate(TEXT,
                    surface_segmentation=segmentation(), ap_binding_context=ap.context)
        finally:
            release.set()
        self.assertTrue(ap.context.has_authoritative_catalog)
        self.assertFalse(semantic.faithful_ir.provenance.approved)
        self.assertEqual(sum(call.get("task") == "propose_source_intent_graph" for call in base.calls), 1)
        self.assertFalse(client._pending)

    def test_ap_edits_discard_the_old_graph_request(self):
        base = Client()
        client = OverlapJSONClient(base)
        with redirect_stdout(io.StringIO()):
            ap = prepare_reviewed_ap_context(context(), client, segmentation=segmentation(),
                interactive=True, input_fn=Mock(side_effect=["e request_occurs", "request_event", "", "a"]))
            semantic = NLUAgent(KeywordRetriever([]), llm_client=client).generate(TEXT,
                surface_segmentation=segmentation(), ap_binding_context=ap.context)
        self.assertIn("request_event", semantic.faithful_ir.ap_names)
        graphs = [call for call in base.calls if call.get("task") == "propose_source_intent_graph"]
        self.assertEqual(len(graphs), 2)
        self.assertNotEqual(graphs[0]["ap_mentions"], graphs[1]["ap_mentions"])

    def test_prefetch_does_not_cache_later_results_or_reuse_different_parameters(self):
        base = Mock(complete_json=Mock(side_effect=[{"answer": 1}, {"answer": 2}, {"answer": 3}]), request_observer=None)
        client = OverlapJSONClient(base)
        client.prefetch("proposal", {"source_text": "first"}, temperature=0.0)
        self.assertEqual(client.complete_json("proposal", {"source_text": "first"}, temperature=0.0), {"answer": 1})
        self.assertEqual(client.complete_json("proposal", {"source_text": "first"}, temperature=0.0), {"answer": 2})
        self.assertEqual(client.complete_json("proposal", {"source_text": "first"}, temperature=0.5), {"answer": 3})
        self.assertEqual(base.complete_json.call_count, 3)

    def test_observed_runs_do_not_start_unaccounted_background_requests(self):
        original = LangChainJSONClient(api_key="fixture-key", model="fixture")
        base = Mock(request_observer=None)
        client = OverlapJSONClient(base)
        with observe_execution(SimpleNamespace(event=lambda *_args: None)):
            self.assertIs(overlap_client(original), original)
            client.prefetch("proposal", {})
        base.complete_json.assert_not_called()

    def test_cancel_does_not_wait_for_speculative_graph_or_approve_anything(self):
        release = Event()

        class BlockingGraphClient(Client):
            def complete_json(self, system, payload, **kwargs):
                if payload.get("task") == "propose_source_intent_graph":
                    if not release.wait(3):
                        raise AssertionError("Cancellation waited for the graph")
                return super().complete_json(system, payload, **kwargs)

        client = OverlapJSONClient(BlockingGraphClient())
        try:
            with redirect_stdout(io.StringIO()), self.assertRaises(APPreparationError):
                prepare_reviewed_ap_context(context(), client, segmentation=segmentation(),
                    interactive=True, input_fn=lambda _: "c")
        finally:
            release.set()


class BudgetExitLatencyTests(unittest.TestCase):
    def review_to_limit(self, folder, *, answer="n"):
        semantic = attach_candidate_ir_set(unresolved_semantic(), TEXT)
        queries = [TraceQuery(f"Q-{index}", "boundary", "test", (),
                    [{"request_occurs", "grant_occurs"}] + [set() for _ in range(index)],
                    obligation_ids=(f"target-{index}",)) for index in range(10)]
        brief = build_trace_adjudication_brief(TEXT, semantic)
        client = Mock(model="fixture", complete_json=Mock(side_effect=AssertionError("metadata delayed fallback")))
        read = Mock(return_value=answer)
        annotations = review_traces_with_simple_choices(requirement=TEXT, semantic=semantic, queries=queries,
            round_index=1, records_dir=folder, llm_client=client, adjudication_brief=brief, input_fn=read)
        client.complete_json.assert_not_called()
        self.assertEqual(read.call_count, 5)
        self.assertEqual(annotations, [])
        return semantic, semantic_review_budget(TEXT, semantic, folder)

    def test_limit_outputs_formula_using_raw_votes_without_model_metadata_or_approval(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            semantic, budget = self.review_to_limit(folder)
            result = replace(pending(semantic), provisional_candidate=None, fallback_candidate=None,
                             review_records=[{"human_review_budget": budget}])
            result = deliver_formula_candidate(result)
            record = json.loads(next(Path(folder).glob("nonexpert_trace_review_*.json")).read_text(encoding="utf-8"))
        self.assertEqual(result.review_records[-1]["raw_human_trace_count"], 5)
        self.assertIn("X", str(result.provisional_candidate.formula))
        self.assertEqual(len(result.provisional_candidate.trace_results), 5)
        self.assertTrue(all(item.matched for item in result.provisional_candidate.trace_results))
        self.assertEqual(result.approval_status, "not_approved")
        self.assertIsNone(result.final_approval)
        self.assertFalse(result.traces)
        self.assertEqual(record["metadata_status"], "deferred_for_candidate_fallback")
        self.assertNotIn("model_response", record)

    def test_unclear_votes_remain_unclear_and_do_not_rank_a_formula(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            semantic, budget = self.review_to_limit(folder, answer="u")
            result = replace(pending(semantic), provisional_candidate=None, fallback_candidate=None,
                             review_records=[{"human_review_budget": budget}])
            result = deliver_formula_candidate(result)
        self.assertEqual(result.review_records[-1]["raw_human_trace_count"], 0)
        self.assertTrue(all(item["locked_label"] == "UNCLEAR" for item in budget["fallback_evidence"]["answers"].values()))

    def test_stale_or_changed_human_votes_cannot_be_used_for_fallback(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            semantic, budget = self.review_to_limit(folder)
            for corrupt in ("context", "label"):
                bad = deepcopy(budget)
                if corrupt == "context":
                    bad["fallback_evidence"]["context"]["ap_context_fingerprint"] = "changed"
                else:
                    next(iter(bad["fallback_evidence"]["answers"].values()))["locked_label"] = "SAT"
                result = replace(pending(semantic), provisional_candidate=None, fallback_candidate=None,
                                 review_records=[{"human_review_budget": bad}])
                with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                    deliver_formula_candidate(result)


if __name__ == "__main__":
    unittest.main()
