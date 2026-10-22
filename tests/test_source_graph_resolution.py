from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cli
from agents.nlu_agent import NLUAgent
from ap_preparation import prepare_reviewed_ap_context
from batch_review import _typed_needs_review_reason
from ir_proposal_router import IRProposalRouter
from llm import LLMClientError
from rag.retriever import KeywordRetriever
from source_graph_resolution import SourceGraphResolutionRequired, resolve_supported_axis_debts
from tests.test_ap_preparation import APS, TEXT, Client, context, graph, segmentation
from tests.test_manual_release import MANUAL_ARGS


DETAIL = "Whether eventually includes the current step or requires a strictly later step."


def proposal(*, reason_code="semantic_axis", detail=DETAIL):
    raw = graph()
    raw["source_intent_graph"]["semantic_debts"] = [
        {"reason_code": reason_code, "source_span": [38, 54], "detail": detail},
    ]
    return raw


class DebtClient(Client):
    model = "fixture-main-model"

    def __init__(self, raw=None, transform=None):
        aps = deepcopy(APS)
        aps["aps"][1]["evidence_texts"] = ["grant", "occur"]
        super().__init__(ap_replies=[aps], graph_replies=[raw or proposal()])
        self.transform = transform

    def complete_json(self, system, payload, **kwargs):
        if payload.get("task") != "map_source_semantic_debts_to_candidate_axes":
            return super().complete_json(system, payload, **kwargs)
        self.calls.append(deepcopy(payload))
        raw = {"resolutions": [{"debt_id": debt["debt_id"],
            "axis_ids": [axis["axis_id"] for axis in debt["allowed_axes"] if axis["axis_kind"] == "eventuality_policy"],
            "reason": "The supplied current-or-future and strict-future alternatives exhaust this timing distinction."}
            for debt in payload["debts"]]}
        return self.transform(raw) if self.transform else raw


class SourceGraphResolutionTests(unittest.TestCase):
    def build(self, client):
        return IRProposalRouter().propose(TEXT, context(), proposal_client=client)

    def test_unbounded_eventuality_is_not_a_missing_deadline_and_whenever_remains_global(self):
        raw = proposal(reason_code="unsupported_semantics", detail=(
            "The phrase 'eventually occur' does not specify a fixed or measurable deadline; "
            "the graph uses an unbounded eventual relation."))
        raw["source_intent_graph"]["nodes"] = [node for node in raw["source_intent_graph"]["nodes"] if node["ref"] != "g"]
        raw["source_intent_graph"]["root_ref"] = "i"
        client = DebtClient(raw)
        result = self.build(client)
        self.assertTrue(result.candidate_complete)
        self.assertFalse(result.semantic_debts)
        self.assertEqual(len(result.candidates), 2)
        self.assertTrue(any(entry.category == "AP_EVIDENCE" and entry.evidence_text == "occur"
                            for entry in result.graph.coverage_ledger.entries))
        for candidate in result.candidates:
            self.assertEqual(candidate.ir.node_map[candidate.ir.root_node_id].node_type, "global_constraint")
        self.assertFalse(any(call.get("task") == "map_source_semantic_debts_to_candidate_axes" for call in client.calls))

    def test_single_if_is_not_promoted_to_a_repeated_global_requirement(self):
        from ap_preparation import ensure_ap_candidates
        from typed_source_relation_graph import source_graph_from_proposal
        source = "If request occurs, grant should eventually occur."
        ap_context = ensure_ap_candidates(context(source), Client()).context
        raw = {"source_intent_graph": {"nodes": [
            {"ref": "r", "kind": "AP", "source_span": [3, 17], "children": []},
            {"ref": "g", "kind": "AP", "source_span": [19, 24], "children": []},
            {"ref": "e", "kind": "eventually", "source_span": [25, 48], "children": ["g"]},
            {"ref": "i", "kind": "implication", "source_span": [0, len(source)], "children": ["r", "e"]},
        ], "root_ref": "i", "semantic_debts": []}}
        built = source_graph_from_proposal(raw, source, ap_context)
        self.assertEqual(next(node for node in built.nodes if node.node_id == built.root_node_id).kind, "implication")

    def test_explicit_deadline_still_blocks_an_unbounded_only_graph(self):
        from ap_preparation import ensure_ap_candidates
        from typed_source_relation_graph import source_graph_from_proposal
        source = TEXT[:-1] + " within 3 steps."
        ap_context = ensure_ap_candidates(context(source), DebtClient()).context
        raw = graph()
        built = source_graph_from_proposal(raw, source, ap_context)
        self.assertTrue(any(debt.reason_code == "capability_gap" for debt in built.semantic_debts))

    def test_timing_debt_produces_complete_candidates_without_approval(self):
        client = DebtClient()
        result = self.build(client)
        self.assertTrue(result.candidate_complete)
        self.assertFalse(result.semantic_debts)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual({dict(candidate.selections)[result.axes[0].axis_id] for candidate in result.candidates},
                         {"current_or_future", "strict_future"})
        self.assertTrue(all(not candidate.ir.provenance.approved for candidate in result.candidates))
        binding = result.graph.axis_debt_resolutions[0]
        self.assertEqual(binding.debt.detail, DETAIL)
        self.assertEqual(binding.model, client.model)
        self.assertFalse(result.graph.coverage_ledger.blocking_debts)

    def test_user_confirmed_fragmented_aps_continue_into_typed_ir(self):
        client = DebtClient()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            prepared = prepare_reviewed_ap_context(context(), client, segmentation=segmentation(),
                interactive=True, records_dir=folder, input_fn=lambda _prompt: "a")
            semantic = NLUAgent(KeywordRetriever([]), llm_client=client).generate(
                TEXT, surface_segmentation=segmentation(), ap_binding_context=prepared.context)
        self.assertEqual(len(semantic.candidate_ir_set.branches), 2)
        self.assertEqual(semantic.semantic_status, "needs_semantic_review")
        self.assertEqual(len(semantic.source_axis_debt_resolutions), 1)
        self.assertEqual(semantic.source_axis_debt_resolutions[0]["debt"]["detail"], DETAIL)

    def test_declined_axis_keeps_original_debt_and_detailed_pending_error(self):
        def decline(raw):
            raw["resolutions"][0]["axis_ids"] = []
            return raw
        client = DebtClient(transform=decline)
        with self.assertRaises(SourceGraphResolutionRequired) as caught:
            NLUAgent(KeywordRetriever([]), llm_client=client).generate(TEXT)
        pending = caught.exception
        self.assertIn(DETAIL, str(pending))
        self.assertEqual(pending.diagnostics["source_reason_code"], "semantic_axis")
        self.assertEqual(pending.diagnostics["semantic_debts"][0]["detail"], DETAIL)
        self.assertEqual(_typed_needs_review_reason(pending, {}), pending.code)

    def test_unknown_axis_or_added_semantics_get_only_one_retry_and_remain_pending(self):
        def unknown(raw):
            raw["resolutions"][0]["axis_ids"] = ["invented-axis"]
            return raw
        def inject(raw):
            raw["resolutions"][0]["node"] = {"kind": "true"}
            return raw
        for transform in (unknown, inject):
            with self.subTest(transform=transform.__name__):
                client = DebtClient(transform=transform)
                result = self.build(client)
                self.assertEqual(result.reason_code, "semantic_axis")
                self.assertFalse(result.candidates)
                requests = [call for call in client.calls if call.get("task") == "map_source_semantic_debts_to_candidate_axes"]
                self.assertEqual(len(requests), 2)
                self.assertIn("validation_error", requests[1])

    def test_real_capability_debt_and_missing_details_cannot_be_mapped_away(self):
        for raw in (proposal(reason_code="capability_gap", detail="Missing reset and duration semantics."),
                    proposal(detail="")):
            with self.subTest(raw=raw):
                client = DebtClient(raw)
                result = self.build(client)
                self.assertFalse(result.candidates)
                self.assertTrue(result.semantic_debts)
                self.assertFalse(any(call.get("task") == "map_source_semantic_debts_to_candidate_axes" for call in client.calls))

    def test_same_span_debts_keep_distinct_ids_and_both_must_be_accounted_for(self):
        raw = proposal()
        raw["source_intent_graph"]["semantic_debts"].append(
            {"reason_code": "semantic_axis", "source_span": [38, 54],
             "detail": "The current-step versus strictly later boundary needs a complete choice."})
        result = self.build(DebtClient(raw))
        bindings = result.graph.axis_debt_resolutions
        self.assertEqual(len({binding.debt.debt_id for binding in bindings}), 2)
        self.assertTrue(result.candidate_complete)

    def test_incomplete_rebuild_never_discharges_a_debt(self):
        original = self.build(DebtClient(transform=lambda _raw: {"resolutions": []}))
        with patch("constructive_ir_builder.try_construct_from_proposed_graph", return_value=original):
            result = resolve_supported_axis_debts(original, context(), DebtClient())
        self.assertIs(result, original)
        self.assertTrue(result.semantic_debts)

    def test_missing_executable_option_cannot_claim_complete_resolution(self):
        original = self.build(DebtClient(transform=lambda _raw: {"resolutions": []}))
        complete = self.build(DebtClient())
        incomplete = replace(complete, candidates=complete.candidates[:1])
        with patch("constructive_ir_builder.try_construct_from_proposed_graph", return_value=incomplete):
            result = resolve_supported_axis_debts(original, context(), DebtClient())
        self.assertIs(result, original)

    def test_two_axes_require_every_option_combination(self):
        original = self.build(DebtClient(transform=lambda _raw: {"resolutions": []}))
        complete = self.build(DebtClient())
        first_axis = complete.axes[0]
        second_axis = replace(first_axis, axis_id="second-axis")
        # Both options of both axes occur, but cross-combinations are missing.
        coupled = replace(complete, axes=(first_axis, second_axis), candidates=tuple(
            replace(candidate, selections=(*candidate.selections, (second_axis.axis_id, candidate.selections[0][1])))
            for candidate in complete.candidates))
        with patch("constructive_ir_builder.try_construct_from_proposed_graph", return_value=coupled):
            result = resolve_supported_axis_debts(original, context(), DebtClient())
        self.assertIs(result, original)

    def test_network_failure_is_not_retried_as_semantic_mapping(self):
        def disconnect(_raw):
            raise LLMClientError("fixture connection failure")
        client = DebtClient(transform=disconnect)
        with self.assertRaises(LLMClientError):
            self.build(client)
        self.assertEqual(sum(call.get("task") == "map_source_semantic_debts_to_candidate_axes" for call in client.calls), 1)

    def test_cli_saves_source_graph_and_reports_pending_without_traceback(self):
        client = DebtClient(transform=lambda raw: {"resolutions": [
            {**item, "axis_ids": []} for item in raw["resolutions"]]})
        with self.assertRaises(SourceGraphResolutionRequired) as caught:
            NLUAgent(KeywordRetriever([]), llm_client=client).generate(TEXT)
        for progress in ("auto", "off"):
            with self.subTest(progress=progress), tempfile.TemporaryDirectory() as folder:
                with patch.object(sys, "argv", ["main.py", *MANUAL_ARGS, "--run-dir", folder, "--progress", progress, TEXT]), \
                     patch.object(sys.stdin, "isatty", return_value=True), \
                     patch.object(cli, "_prepare_artifacts"), \
                     patch.object(cli, "run_traj_ltlf_pipeline", side_effect=caught.exception), \
                     redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as error:
                    cli.main()
                saved = json.loads(next((Path(folder) / "review_records").glob("source_graph_pending_*.json")).read_text(encoding="utf-8"))
                self.assertEqual(saved["source_reason_code"], "semantic_axis")
                self.assertEqual(saved["semantic_debts"][0]["detail"], DETAIL)
                self.assertIsNotNone(saved["source_graph"])
                self.assertIn(DETAIL, output.getvalue())
                self.assertNotIn("[ERR]", output.getvalue())
                self.assertNotIn("Traceback", error.getvalue())
                self.assertFalse((Path(folder) / "outputs/result.txt").exists())


if __name__ == "__main__":
    unittest.main()
