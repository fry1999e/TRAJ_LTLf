from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agents.nlu_agent import NLUAgent
from ap_binding_context import build_ap_binding_context, build_runtime_ap_binding_context
from ap_preparation import APPreparationError, ensure_ap_candidates, prepare_reviewed_ap_context, source_symbol_ap_proposals
from batch_review import _regenerate_batch_item, _typed_needs_review_reason
from checkpointing import save_ap_inventory_checkpoint, load_ap_inventory_checkpoint
from ir_proposal_router import IRProposalRouter
from llm import LLMClientError
from pipeline import run_traj_ltlf_pipeline
from pre_ir_review import PreIRArtifacts, _parse_interactive_surface_input
from rag.retriever import KeywordRetriever
from semantic_source import natural_language_source
from surface_segmentation import build_surface_segmentation
import cli


TEXT = "Whenever request occurs, grant should eventually occur."
MARKED = "Whenever ⟦CUT⟧request occurs, ⟦CUT⟧grant ⟦CUT⟧should ⟦CUT⟧eventually ⟦CUT⟧occur."
APS = {"aps": [
    {"canonical_name": "request_occurs", "evidence_texts": ["request occurs"]},
    {"canonical_name": "grant_occurs", "evidence_texts": ["grant"]},
]}


def context(text=TEXT):
    return build_runtime_ap_binding_context(requirement_id="REQ-001", source_text=text)


def segmentation(marked=MARKED):
    return build_surface_segmentation(requirement_id="REQ-001", source_text=TEXT,
                                      marked_text=marked, marker="⟦CUT⟧", status="surface_reviewed")


def graph():
    return {"source_intent_graph": {"nodes": [
        {"ref": "p", "kind": "AP", "source_span": [9, 23], "children": []},
        {"ref": "q", "kind": "AP", "source_span": [25, 30], "children": []},
        {"ref": "e", "kind": "eventually", "source_span": [25, len(TEXT)-1], "children": ["q"]},
        {"ref": "i", "kind": "implication", "source_span": [0, len(TEXT)-1], "children": ["p", "e"]},
        {"ref": "g", "kind": "global_constraint", "source_span": [0, len(TEXT)-1], "children": ["i"]},
    ], "root_ref": "g", "semantic_debts": []}}


class Client:
    def __init__(self, ap_replies=None, graph_replies=None):
        self.calls = []
        self.ap_replies = list(ap_replies or [APS])
        self.graph_replies = list(graph_replies or [graph()])

    def complete_json(self, system, payload, **kwargs):
        self.calls.append(deepcopy(payload))
        queue = self.ap_replies if payload.get("task") == "extract_source_ap_candidates" else self.graph_replies
        return deepcopy(queue.pop(0) if len(queue) > 1 else queue[0])


class APPreparationTests(unittest.TestCase):
    def test_symbolic_holds_clauses_propose_separate_reused_aps_without_model_calls(self):
        samples = [
            ("Globally it is the case that if a holds then eventually a and b hold.", ("a", "b")),
            ("If p holds then eventually p or q holds.", ("p", "q")),
            ("Whenever prop_12 occurs, eventually prop_12 and prop_9 occur.", ("prop_12", "prop_9")),
            ("Globally p and p hold.", ("p",)),
        ]
        for source, expected in samples:
            with self.subTest(source=source):
                client = Mock()
                initial = build_ap_binding_context(requirement_id="REQ-001", source_text=source)
                self.assertEqual(source_symbol_ap_proposals(source), {
                    "aps": [{"canonical_name": name, "evidence_texts": [name]} for name in expected]})
                prepared = ensure_ap_candidates(initial, client)
                client.complete_json.assert_not_called()
                if prepared.inventory is None:
                    self.assertIs(prepared.context, initial)
                    self.assertEqual(tuple(ap.canonical_name for ap in initial.grounding_contract.identities), expected)
                    continue
                self.assertEqual(tuple(ap.canonical_name for ap in prepared.inventory.identities), expected)
                self.assertEqual(prepared.inventory.status, "pending_human_review")
                self.assertFalse(prepared.context.has_authoritative_catalog)
                self.assertTrue(all(mention.mapping_provenance == "source_symbol_predicates_v1"
                                    for mention in prepared.inventory.mentions))

    def test_repeated_symbol_evidence_binds_both_occurrences_to_one_ap(self):
        source = "Globally it is the case that if a holds then eventually a and b hold."
        prepared = ensure_ap_candidates(context(source), Mock())
        ap = next(ap for ap in prepared.inventory.identities if ap.canonical_name == "a")
        mentions = [mention for mention in prepared.inventory.mentions if mention.ap_identity_id == ap.ap_identity_id]
        self.assertEqual([mention.source_span for mention in mentions], [(32, 33), (56, 57)])
        self.assertTrue(all(mention.evidence_text == "a" for mention in mentions))

    def test_symbolic_proposals_still_need_explicit_human_approval(self):
        source = "Globally it is the case that if a holds then eventually a and b hold."
        client = Mock()
        with redirect_stdout(io.StringIO()) as output:
            approved = prepare_reviewed_ap_context(context(source), client, interactive=True,
                input_fn=Mock(side_effect=["", "a"]))
        client.complete_json.assert_not_called()
        self.assertTrue(approved.context.has_authoritative_catalog)
        self.assertEqual(set(approved.context.declared_catalog), {"a", "b"})
        self.assertNotIn("模型候选未通过", output.getvalue())
        self.assertEqual(approved.inventory.reviewer_source, "human")

    def test_symbolic_cancellation_does_not_approve_or_call_model(self):
        client = Mock()
        with redirect_stdout(io.StringIO()), self.assertRaises(APPreparationError) as caught:
            prepare_reviewed_ap_context(context("If p holds then q holds."), client,
                interactive=True, input_fn=lambda _: "c")
        self.assertEqual(caught.exception.code, "AP_GROUNDING_REVIEW_REQUIRED")
        client.complete_json.assert_not_called()

    def test_symbol_rule_never_drops_domain_predicates_articles_negation_or_deadlines(self):
        sources = [TEXT, "If p holds then request occurs.", "If p holds then a motor eventually stops.",
                   "If p holds then not q holds.", "If a holds then a and b hold within 2 steps.",
                   "If p holds then the door holds.", "If p holds then q and \"r\" hold."]
        for source in sources:
            with self.subTest(source=source):
                self.assertIsNone(source_symbol_ap_proposals(source))

    def test_symbolic_ap_prefetch_is_skipped_instead_of_making_an_unused_request(self):
        from llm.overlap_client import OverlapJSONClient
        base = Mock()
        client = OverlapJSONClient(base)
        client.prefetch_aps("Globally if a holds then eventually a and b hold.")
        base.complete_json.assert_not_called()
        self.assertFalse(client._pending)

    def test_discontinuous_predicate_ignores_occurs_when_quoting_occur(self):
        proposals = deepcopy(APS)
        proposals["aps"][1]["evidence_texts"] = ["grant", "occur"]
        client = Client([proposals])
        result = ensure_ap_candidates(context(), client, segmentation=segmentation())
        self.assertEqual(len(client.calls), 1)
        grant = next(ap for ap in result.inventory.identities if ap.canonical_name == "grant_occurs")
        mentions = [m for m in result.inventory.mentions if m.ap_identity_id == grant.ap_identity_id]
        self.assertEqual([m.evidence_text for m in mentions], ["grant", "occur"])
        self.assertEqual(result.inventory.status, "pending_human_review")

    def test_whole_word_evidence_ignores_identifier_substrings_including_case_variants(self):
        text = "ON then shutdown then on."
        client = Client([{"aps": [{"canonical_name": "is_on", "evidence_texts": ["on"]}]}])
        result = ensure_ap_candidates(context(text), client)
        self.assertEqual([m.evidence_text for m in result.inventory.mentions], ["ON", "on"])
        self.assertEqual(len(client.calls), 1)

    def test_failed_model_proposals_can_be_edited_and_then_explicitly_approved(self):
        invalid = deepcopy(APS)
        invalid["aps"][1]["evidence_texts"] = ["grant should eventually occur"]
        client = Client([invalid])
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
            result = prepare_reviewed_ap_context(context(), client, segmentation=segmentation(),
                interactive=True, records_dir=folder,
                input_fn=Mock(side_effect=["e " + json.dumps(APS), "a"]))
            record = json.loads(next(Path(folder).glob("ap_review_*.json")).read_text(encoding="utf-8"))
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(result.context.has_authoritative_catalog)
        self.assertIn("swallows", output.getvalue())
        self.assertEqual(result.inventory.reviewer_source, "human")
        self.assertEqual(record["proposal_recovery"]["resolution_source"], "human_ap_edit")

    def test_invalid_candidates_cannot_be_approved_by_pressing_a(self):
        client = Client([{"aps": []}])
        read = Mock(side_effect=["a", "c"])
        with redirect_stdout(io.StringIO()), self.assertRaises(APPreparationError) as caught:
            prepare_reviewed_ap_context(context(), client, interactive=True, input_fn=read)
        self.assertEqual(caught.exception.code, "AP_PROPOSAL_INVALID")
        self.assertEqual(read.call_count, 2)
        self.assertEqual(len(client.calls), 2)

    def test_failed_model_proposals_can_be_regenerated_before_approval(self):
        client = Client([{"aps": []}, {"aps": []}, APS])
        with redirect_stdout(io.StringIO()):
            result = prepare_reviewed_ap_context(context(), client, interactive=True,
                input_fn=Mock(side_effect=["r", "a"]))
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(result.context.has_authoritative_catalog)
        self.assertEqual(len(client.calls[2]["previous_failed_attempts"]), 2)

    def test_explicit_regeneration_replaces_only_unapproved_proposals(self):
        proposed = ensure_ap_candidates(context(), Client()).context
        edited = deepcopy(APS)
        edited["aps"][0]["canonical_name"] = "request_event"
        client = Client([edited])
        with redirect_stdout(io.StringIO()):
            result = prepare_reviewed_ap_context(proposed, client, interactive=True,
                input_fn=Mock(side_effect=["r", "a"]))
        self.assertEqual(len(client.calls), 1)
        self.assertIn("request_event", result.context.declared_catalog)
        self.assertEqual(proposed.grounding_contract.identities[0].canonical_name, "request_occurs")

    def test_failed_extraction_preserves_both_rejected_proposals_and_reasons(self):
        bad = {"aps": [{"canonical_name": "invented", "evidence_texts": ["missing event"]}]}
        client = Client([{"aps": []}, bad])
        with self.assertRaises(APPreparationError) as caught:
            ensure_ap_candidates(context(), client)
        error = caught.exception
        self.assertIn("missing event", str(error))
        self.assertEqual(error.diagnostics["source_hash"], context().effective_source_hash)
        self.assertEqual(len(error.diagnostics["failed_attempts"]), 2)
        self.assertEqual(error.diagnostics["failed_attempts"][1]["proposal"], bad)
        self.assertEqual(client.calls[1]["rejected_proposal"], {"aps": []})

    def test_case_mismatch_and_overlapping_predicates_still_fail(self):
        for raw in (
            {"aps": [{"canonical_name": "r", "evidence_texts": ["Request occurs"]}]},
            {"aps": [{"canonical_name": "r", "evidence_texts": ["request occurs"]},
                     {"canonical_name": "q", "evidence_texts": ["request"]}]},
        ):
            with self.subTest(raw=raw), self.assertRaises(APPreparationError):
                ensure_ap_candidates(context(), Client([raw]))

    def test_temporal_clause_is_repaired_before_ap_review(self):
        bad = deepcopy(APS)
        bad["aps"][1]["evidence_texts"] = ["grant should eventually occur"]
        client = Client(ap_replies=[bad, APS])
        result = ensure_ap_candidates(context(), client, segmentation=segmentation())
        self.assertEqual(len(client.calls), 2)
        self.assertIn("swallows", client.calls[1]["validation_error"])
        grant_mentions = [mention for mention in result.inventory.mentions if mention.evidence_text == "grant"]
        self.assertEqual(len(grant_mentions), 1)
        self.assertEqual(result.inventory.status, "pending_human_review")

    def test_user_surface_command_remains_valid_and_proposals_are_not_approval(self):
        seg = segmentation()
        offsets, _ = _parse_interactive_surface_input(seg, "e " + MARKED)
        self.assertEqual(offsets, [9, 25, 31, 38, 49])
        original = context()
        client = Client()
        result = ensure_ap_candidates(original, client, segmentation=seg)
        self.assertEqual(result.context.grounding_authority, "proposal_only")
        self.assertFalse(result.context.has_authoritative_catalog)
        self.assertEqual(result.inventory.status, "pending_human_review")
        self.assertEqual(len(result.context.grounding_contract.identities), 2)
        self.assertEqual(original.grounding_contract.identities, ())
        self.assertEqual(client.calls[0], {"task": "extract_source_ap_candidates", "source_text": TEXT})

    def test_exact_predicate_may_cross_multiple_reading_segments(self):
        proposals = {"aps": [{"canonical_name": "request_occurs", "evidence_texts": ["request occurs"]}]}
        cuts = MARKED.replace("request occurs", "request ⟦CUT⟧occurs")
        result = ensure_ap_candidates(context(), Client([proposals]), segmentation=segmentation(cuts))
        self.assertGreater(len(result.inventory.mentions[0].supporting_segment_ids), 1)

    def test_human_confirmation_creates_authority_and_audit_record(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            read = Mock(side_effect=["", "a"])
            result = prepare_reviewed_ap_context(context(), Client(), segmentation=segmentation(),
                                                 interactive=True, input_fn=read, records_dir=folder)
            files = list(Path(folder).glob("ap_review_*.json"))
            self.assertEqual(len(files), 1)
            record = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(read.call_count, 2)
        self.assertTrue(result.context.has_authoritative_catalog)
        self.assertEqual(result.inventory.reviewer_source, "human")
        self.assertEqual(record["ap_inventory"]["status"], "approved")

    def test_human_can_edit_before_confirming_without_extra_model_calls(self):
        edited = deepcopy(APS)
        edited["aps"][0]["canonical_name"] = "request_event"
        client = Client()
        with redirect_stdout(io.StringIO()):
            result = prepare_reviewed_ap_context(context(), client, interactive=True,
                input_fn=Mock(side_effect=["e " + json.dumps(edited), "a"]))
        self.assertIn("request_event", result.context.declared_catalog)
        self.assertEqual(len(client.calls), 1)

    def test_named_edit_preserves_other_aps_and_discontinuous_evidence(self):
        for command in ('e grant_occurs grant | occur',
                        'e grant_occurs ["grant", "occur"]'):
            with self.subTest(command=command), redirect_stdout(io.StringIO()):
                client = Client()
                result = prepare_reviewed_ap_context(context(), client, interactive=True,
                    input_fn=Mock(side_effect=[command, "a"]))
                proposals = {item.canonical_name: [m.evidence_text for m in item.mentions]
                             for item in result.context.grounding_contract.identities}
                self.assertEqual(proposals, {
                    "request_occurs": ["request occurs"], "grant_occurs": ["grant", "occur"]})
                self.assertEqual(len(client.calls), 1)

    def test_guided_edit_can_rename_one_ap_and_keep_its_evidence(self):
        with redirect_stdout(io.StringIO()):
            result = prepare_reviewed_ap_context(context(), Client(), interactive=True,
                input_fn=Mock(side_effect=["e", "request_occurs", "request_event", "", "a"]))
        proposals = {item.canonical_name: [m.evidence_text for m in item.mentions]
                     for item in result.context.grounding_contract.identities}
        self.assertEqual(proposals, {"request_event": ["request occurs"], "grant_occurs": ["grant"]})

    def test_invalid_named_edit_keeps_all_valid_candidates(self):
        commands = ["e missing grant", "e grant_occurs nonexistent", "e grant_occurs request",
                    "e grant_occurs grant |", 'e grant_occurs ["grant",', "a"]
        with redirect_stdout(io.StringIO()) as output:
            result = prepare_reviewed_ap_context(context(), Client(), interactive=True,
                input_fn=Mock(side_effect=commands))
        self.assertEqual(output.getvalue().count("AP 修改未通过校验"), 5)
        self.assertEqual({item.canonical_name: [m.evidence_text for m in item.mentions]
                          for item in result.context.grounding_contract.identities},
                         {"request_occurs": ["request occurs"], "grant_occurs": ["grant"]})

    def test_multiple_edits_accumulate_before_explicit_confirmation(self):
        commands = ["e grant_occurs grant | occur", "e request_occurs request",
                    "e grant_occurs grant", "e request_occurs request occurs", "a"]
        with redirect_stdout(io.StringIO()):
            result = prepare_reviewed_ap_context(context(), Client(), interactive=True,
                input_fn=Mock(side_effect=commands))
        self.assertEqual({item.canonical_name: [m.evidence_text for m in item.mentions]
                          for item in result.context.grounding_contract.identities},
                         {"request_occurs": ["request occurs"], "grant_occurs": ["grant"]})
        self.assertTrue(result.context.has_authoritative_catalog)

    def test_named_edit_requires_explicit_approval_and_guided_eof_stays_pending(self):
        for commands in (["e grant_occurs grant | occur", "c"], ["e", EOFError()]):
            with self.subTest(commands=commands), redirect_stdout(io.StringIO()), \
                 self.assertRaises(APPreparationError) as caught:
                prepare_reviewed_ap_context(context(), Client(), interactive=True,
                    input_fn=Mock(side_effect=commands))
            self.assertEqual(caught.exception.code, "AP_GROUNDING_REVIEW_REQUIRED")

    def test_named_edit_repairs_rejected_model_candidate_without_replacing_other_aps(self):
        bad = deepcopy(APS)
        bad["aps"][1]["evidence_texts"] = ["grant should eventually occur"]
        with redirect_stdout(io.StringIO()):
            client = Client([bad])
            result = prepare_reviewed_ap_context(context(), client, interactive=True,
                input_fn=Mock(side_effect=["e grant_occurs grant | occur", "a"]))
        self.assertEqual(set(result.context.declared_catalog), {"request_occurs", "grant_occurs"})
        self.assertEqual(len(client.calls), 2)

    def test_rejected_candidates_can_be_repaired_one_at_a_time(self):
        bad = deepcopy(APS)
        bad["aps"][0]["evidence_texts"] = ["nonexistent"]
        bad["aps"][1]["evidence_texts"] = ["grant should eventually occur"]
        with redirect_stdout(io.StringIO()):
            result = prepare_reviewed_ap_context(context(), Client([bad]), interactive=True,
                input_fn=Mock(side_effect=["e request_occurs request occurs", "e grant_occurs grant", "a"]))
        self.assertTrue(result.context.has_authoritative_catalog)
        self.assertEqual(set(result.context.declared_catalog), {"request_occurs", "grant_occurs"})

    def test_cancel_never_approves_or_generates_ir(self):
        client = Client()
        with redirect_stdout(io.StringIO()), self.assertRaises(APPreparationError) as caught:
            prepare_reviewed_ap_context(context(), client, interactive=True, input_fn=lambda _: "c")
        self.assertEqual(caught.exception.code, "AP_GROUNDING_REVIEW_REQUIRED")
        self.assertEqual([c["task"] for c in client.calls], ["extract_source_ap_candidates"])

    def test_existing_authority_is_reused_without_model_or_user_requests(self):
        with redirect_stdout(io.StringIO()):
            approved = prepare_reviewed_ap_context(context(), Client(), interactive=True, input_fn=lambda _: "a")
        client, read = Mock(), Mock()
        result = prepare_reviewed_ap_context(approved.context, client, interactive=True, input_fn=read)
        self.assertIs(result.context, approved.context)
        client.complete_json.assert_not_called()
        read.assert_not_called()

    def test_malformed_proposal_gets_one_bounded_retry(self):
        client = Client([{"aps": []}, APS])
        result = ensure_ap_candidates(context(), client)
        self.assertEqual(len(result.context.grounding_contract.identities), 2)
        self.assertEqual(len(client.calls), 2)

    def test_checkpoint_resume_reuses_the_identical_ap_context(self):
        base = context()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            approved = prepare_reviewed_ap_context(base, Client(), segmentation=segmentation(),
                interactive=True, input_fn=lambda _: "a")
            save_ap_inventory_checkpoint(folder, inventory=approved.inventory, make_current=True)
            inventory = load_ap_inventory_checkpoint(folder, "REQ-001", current=True,
                expected_source_hash=base.effective_source_hash)
            resumed = build_ap_binding_context(requirement_id=base.requirement_id, source_text=TEXT,
                representation_profile=base.representation_profile, active_inventory=inventory,
                provenance=base.provenance)
        self.assertEqual(resumed.context_fingerprint, approved.context.context_fingerprint)
        client = Mock()
        self.assertIs(ensure_ap_candidates(resumed, client).context, resumed)
        client.complete_json.assert_not_called()

    def test_stale_reading_segments_cannot_trigger_ap_extraction(self):
        client = Mock()
        with self.assertRaises(APPreparationError) as caught:
            ensure_ap_candidates(context("other source"), client, segmentation=segmentation())
        self.assertEqual(caught.exception.code, "AP_SOURCE_STALE")
        client.complete_json.assert_not_called()

    def test_invalid_proposals_stop_before_graph_generation(self):
        variants = [
            {"aps": []},
            {"aps": [{"canonical_name": "invented", "evidence_texts": ["shutdown"]}]},
            {"aps": [{"canonical_name": "timing", "evidence_texts": ["eventually"]}]},
            {"aps": [{"canonical_name": "x", "evidence_texts": ["request occurs"]},
                     {"canonical_name": "y", "evidence_texts": ["request"]}]},
            {"aps": [{"canonical_name": "x", "evidence_texts": ["request"]},
                     {"canonical_name": "X", "evidence_texts": ["grant"]}]},
            {"aps": APS["aps"], "formula": "G(request -> F grant)"},
        ]
        for value in variants:
            with self.subTest(value=value):
                client = Client([value])
                with self.assertRaises(APPreparationError) as caught:
                    NLUAgent(KeywordRetriever([]), llm_client=client).generate(TEXT)
                self.assertEqual(caught.exception.code, "AP_PROPOSAL_INVALID")
                self.assertEqual([c["task"] for c in client.calls], ["extract_source_ap_candidates"] * 2)

    def test_identifier_substrings_are_not_bound_as_predicates(self):
        client = Client([{"aps": [{"canonical_name": "on", "evidence_texts": ["on"]}]}])
        with self.assertRaises(APPreparationError):
            ensure_ap_candidates(context("shutdown"), client)

    def test_unavailable_generation_client_has_actionable_error(self):
        with self.assertRaises(APPreparationError) as caught:
            ensure_ap_candidates(context(), None)
        self.assertEqual(caught.exception.code, "AP_GROUNDING_REQUIRED")

    def test_network_error_is_not_repeated_as_a_bad_ap_proposal(self):
        client = Mock()
        client.complete_json.side_effect = LLMClientError("fixture network unavailable")
        with self.assertRaises(LLMClientError):
            ensure_ap_candidates(context(), client)
        self.assertEqual(client.complete_json.call_count, 1)

    def test_stale_effective_source_is_rejected_before_model_requests(self):
        client = Mock()
        with self.assertRaises(ValueError):
            NLUAgent(KeywordRetriever([]), llm_client=client).generate(
                TEXT, ap_binding_context=context(),
                effective_semantic_source=natural_language_source(requirement_id="REQ-001", source_text="other"),
            )
        client.complete_json.assert_not_called()


class APGraphIntegrationTests(unittest.TestCase):
    def test_cli_saves_failed_ap_candidates_and_displays_the_actual_reason(self):
        client = Client([{"aps": [{"canonical_name": "bad", "evidence_texts": ["missing event"]}]}])
        with self.assertRaises(APPreparationError) as failure:
            ensure_ap_candidates(context(), client)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output, \
             patch.object(sys, "argv", ["main.py", "--run-dir", folder, TEXT]), \
             patch.object(sys.stdin, "isatty", return_value=True), \
             patch.object(cli, "prepare_checkpoint_run", return_value=None), \
             patch.object(cli, "_prepare_artifacts"), \
             patch.object(cli, "run_traj_ltlf_pipeline", side_effect=failure.exception):
            with self.assertRaises(SystemExit):
                cli.main()
            record = json.loads(next((Path(folder) / "review_records").glob("ap_pending_*.json")).read_text(encoding="utf-8"))
        self.assertIn("missing event", output.getvalue())
        self.assertEqual(record["status"], "awaiting_ap_correction")
        self.assertEqual(len(record["diagnostics"]["failed_attempts"]), 2)

    def test_cli_reports_pending_ap_as_an_actionable_message(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as error, \
                patch.object(sys, "argv", ["main.py", TEXT]), \
                patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(cli, "_prepare_artifacts"), \
                patch.object(cli, "run_traj_ltlf_pipeline", side_effect=APPreparationError(
                    "AP_GROUNDING_REVIEW_REQUIRED", "AP 审核尚未完成，未生成 IR。")):
            with self.assertRaises(SystemExit) as caught:
                cli.main()
        self.assertEqual(str(caught.exception), "AP 审核尚未完成，未生成 IR。")
        self.assertNotIn("Traceback", error.getvalue())

    def test_plain_nl_agent_now_builds_source_bound_ir_candidates(self):
        client = Client()
        result = NLUAgent(KeywordRetriever([]), llm_client=client).generate(TEXT)
        self.assertIsNotNone(result.faithful_ir)
        self.assertTrue(result.candidate_ir_set.branches)
        self.assertEqual({ap.name for ap in result.faithful_ir.atomic_propositions}, {"request_occurs", "grant_occurs"})
        self.assertFalse(result.faithful_ir.provenance.approved)
        self.assertEqual(client.calls[0]["task"], "extract_source_ap_candidates")

    def test_router_also_guards_direct_calls_with_empty_ap_context(self):
        client = Client()
        result = IRProposalRouter().propose(TEXT, context(), proposal_client=client)
        self.assertTrue(result.candidates)
        self.assertEqual(client.calls[0]["task"], "extract_source_ap_candidates")

    def test_ap_span_mismatch_gets_one_narrow_repair(self):
        bad = graph()
        bad["source_intent_graph"]["nodes"][0]["source_span"] = [9, 16]
        client = Client(graph_replies=[bad, graph()])
        result = IRProposalRouter().propose(TEXT, context(), proposal_client=client)
        self.assertTrue(result.candidates)
        self.assertEqual(client.calls[-1]["task"], "repair_source_ap_spans")

    def test_unresolved_span_after_repair_becomes_human_review_request(self):
        bad = graph()
        bad["source_intent_graph"]["nodes"][0]["source_span"] = [9, 16]
        client = Client(graph_replies=[bad])
        with self.assertRaises(APPreparationError) as caught:
            IRProposalRouter().propose(TEXT, context(), proposal_client=client)
        self.assertEqual(caught.exception.code, "AP_GRAPH_BINDING_REVIEW_REQUIRED")
        self.assertEqual(len(client.calls), 3)

    def test_ap_span_repair_cannot_change_boolean_structure(self):
        bad = graph()
        bad["source_intent_graph"]["nodes"][0]["source_span"] = [9, 16]
        changed = graph()
        changed["source_intent_graph"]["nodes"][3]["kind"] = "conjunction"
        with self.assertRaises(APPreparationError):
            IRProposalRouter().propose(TEXT, context(), proposal_client=Client(graph_replies=[bad, changed]))

    def test_manual_single_pipeline_passes_approved_context_to_semantic_stage(self):
        class Captured(Exception):
            pass
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), \
                patch("pipeline.run_pre_ir_review", return_value=PreIRArtifacts(segmentation(), None)), \
                patch("builtins.input", return_value="a"), \
                patch("pipeline.run_semantic_review", side_effect=Captured) as stage:
            with self.assertRaises(Captured):
                run_traj_ltlf_pipeline(TEXT, llm_client=Client(), llm_mode="api", review_mode="manual",
                    experiment_mode="human", approved_memory_file=None, records_dir=folder)
        self.assertTrue(stage.call_args.kwargs["ap_binding_context"].has_authoritative_catalog)

    def test_batch_prepares_and_checkpoints_human_inventory_before_semantics(self):
        class Captured(Exception):
            pass
        agent = SimpleNamespace(llm_enabled=True, llm_client=Client(), generate=Mock(side_effect=Captured))
        state = SimpleNamespace(requirement_id="REQ-001", requirement=TEXT, regenerate_semantic=True,
            semantic=None, guided_context=None, ap_binding_context=context(), surface_segmentation=segmentation(),
            plain_reading_review=None, semantic_feedback_history=[], effective_semantic_source=None, expected_semantic=None)
        state.nlu_agent = agent
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), patch("builtins.input", return_value="a"), \
                patch("batch_review.save_ap_inventory_checkpoint") as save:
            with self.assertRaises(Captured):
                _regenerate_batch_item(state, checkpoint_dir="fixture", round_index=1, item_index=1,
                    total_items=1, review_mode="manual", records_dir=folder, allow_degraded=False)
        self.assertTrue(agent.generate.call_args.kwargs["ap_binding_context"].has_authoritative_catalog)
        self.assertEqual(save.call_args.kwargs["inventory"].reviewer_source, "human")
        self.assertTrue(save.call_args.kwargs["make_current"])
        self.assertEqual(_typed_needs_review_reason(APPreparationError("AP_PROPOSAL_INVALID", "fixture"), {}), "AP_PROPOSAL_INVALID")


if __name__ == "__main__":
    unittest.main()
