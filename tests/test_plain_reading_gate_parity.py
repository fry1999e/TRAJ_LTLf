"""The manual adapter and retained Judge adapter must share one gate policy."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm.errors import LLMResponseSchemaError
from plain_reading import (
    PlainReadingValidationError, assess_plain_reading_ir_readiness,
    assert_plain_reading_gate, assert_plain_reading_ready_for_candidate_ir,
    assert_plain_reading_ready_for_finalization,
)
from pre_ir_review import (
    PreIRGateError, complete_plain_reading_review, complete_plain_reading_with_model,
)
from pre_ir_batch_review import _reconcile_migrated_plain_reading_readiness, _route_item_by_readiness
from tests.test_nonexpert_review import AssistanceClient, SOURCE, proposal, segmentation


SOURCES = ("model_assistant", "independent_judge")


class RecordingClient(AssistanceClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []

    def complete_json(self, system, payload, **kwargs):
        self.requests.append(deepcopy((system, payload, kwargs)))
        return super().complete_json(system, payload, **kwargs)


def gate_outcome(review, seg):
    gate = assess_plain_reading_ir_readiness(review, seg)
    return (review.overall_status, gate.disposition, gate.can_enter_semantic_ir,
            gate.can_attempt_candidate_ir, dict(review.selected_options),
            gate.unresolved_ambiguity_ids, gate.validation_errors)


class PlainReadingGateParityTests(unittest.TestCase):
    def test_detected_deadline_and_advice_alternatives_receive_source_based_option_decisions(self):
        from tests.test_ap_preparation import TEXT, segmentation as request_segmentation

        class RequirementClient(RecordingClient):
            def complete_json(self, system, payload, **kwargs):
                self.requests.append(deepcopy((system, payload, kwargs)))
                if payload.get("proposal"):
                    return {"decision": "ambiguity_found", "reason": "Check proposed modal and deadline readings.",
                        "source_quotes": [{"start": 0, "end": len(TEXT), "text": TEXT}],
                        "ambiguities": [{"question": question,
                            "source_quote": {"start": 0, "end": len(TEXT), "text": TEXT},
                            "options": [{"plain_interpretation": first, "difference_summary": "First proposed reading."},
                                        {"plain_interpretation": second, "difference_summary": "Second proposed reading."}]}
                            for question, first, second in (
                                ("Is this a requirement or mere advice?", "Grant is required after every request.", "Grant is only recommended."),
                                ("Does eventually impose a fixed deadline?", "Grant has a fixed measurable deadline.", "Grant must occur without a specified deadline."),
                            )]}
                return {"decisions": [{"ambiguity_id": axis["ambiguity_id"],
                    "decision": axis["options"][index]["option_id"],
                    "reason": "Requirement context and unbounded eventuality do not add advice or a fixed deadline."}
                    for index, axis in enumerate(payload["ambiguities"])]}

        seg = request_segmentation()
        pending = proposal(seg)
        requests = []
        for source in SOURCES:
            client = RequirementClient()
            with patch("builtins.input", side_effect=AssertionError("No semantic questions for the user")):
                result = complete_plain_reading_review(TEXT, seg, pending, client, reviewer_source=source)
            self.assertEqual(result.overall_status, "plain_reading_reviewed")
            self.assertEqual(len(result.selected_options), 2)
            self.assertFalse(result.no_ambiguity_confirmation.human_confirmed)
            self.assertTrue(all(axis.comments for axis in result.ambiguities))
            assert_plain_reading_ready_for_finalization(result, seg)
            self.assertIn("fixed, numeric or measurable deadline", client.requests[1][0])
            requests.append(client.requests)
        self.assertEqual(requests[0], requests[1])

    def finish(self, review, *, source, client):
        with patch("builtins.input", side_effect=AssertionError("no professional review prompt")):
            return complete_plain_reading_review(SOURCE, segmentation(), review, client, reviewer_source=source)

    def test_manual_command_adapter_delegates_to_shared_policy(self):
        seg = segmentation()
        pending = proposal(seg)
        client = AssistanceClient()
        with patch("pre_ir_review.complete_plain_reading_review", return_value=pending) as policy:
            self.assertIs(complete_plain_reading_with_model(SOURCE, seg, pending, client), pending)
        policy.assert_called_once_with(SOURCE, seg, pending, client, reviewer_source="model_assistant")

    def test_same_proposal_and_answer_use_identical_prompts_schema_and_gate(self):
        seg = segmentation()
        for with_axis in (False, True):
            with self.subTest(with_axis=with_axis):
                pending = proposal(seg, with_axis=with_axis)
                results, requests = [], []
                for source in SOURCES:
                    client = RecordingClient()
                    result = self.finish(pending, source=source, client=client)
                    self.assertEqual(result.reviewer_source, source)
                    self.assertFalse(result.no_ambiguity_confirmation.human_confirmed)
                    results.append(gate_outcome(result, seg))
                    requests.append(client.requests)
                self.assertEqual(results[0], results[1])
                self.assertEqual(requests[0], requests[1])

    def test_legacy_a_b_and_exact_option_ids_share_identical_normalization(self):
        seg = segmentation()
        pending = proposal(seg, with_axis=True)
        for choice, expected in (("A", "OPT-now"), ("B", "OPT-later"),
                                 ("OPT-now", "OPT-now"), ("OPT-later", "OPT-later")):
            for source in SOURCES:
                with self.subTest(choice=choice, source=source):
                    result = self.finish(pending, source=source, client=AssistanceClient(reading_decision=choice))
                    self.assertEqual(dict(result.selected_options), {"AX-1": expected})
                    assert_plain_reading_ready_for_finalization(result, seg)

    def test_unresolved_answer_uses_candidate_route_for_either_source(self):
        seg = segmentation()
        pending = proposal(seg, with_axis=True)
        outcomes = []
        for source in SOURCES:
            result = self.finish(pending, source=source, client=AssistanceClient(reading_decision="U"))
            item = {}
            _route_item_by_readiness(item, result, seg)
            self.assertEqual(item["next_stage"], "candidate_ir")
            self.assertFalse(item["single_ir_ready"])
            self.assertFalse(item["finalization_allowed"])
            with self.assertRaises(PlainReadingValidationError):
                assert_plain_reading_ready_for_finalization(result, seg)
            with self.assertRaises(PlainReadingValidationError):
                assert_plain_reading_ready_for_candidate_ir(result, seg, mapped_axes={})
            mappings = {"AX-1": {option.option_id: {"operation": "choose_policy_branch", "node_id": "eventually"}
                                  for option in result.ambiguities[0].options}}
            assert_plain_reading_ready_for_candidate_ir(result, seg, mapped_axes=mappings)
            outcomes.append(gate_outcome(result, seg))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_rejecting_all_axes_requires_a_second_no_ambiguity_confirmation(self):
        seg = segmentation()
        outcomes = []
        for source in SOURCES:
            client = RecordingClient(reading_decision="N")
            result = self.finish(proposal(seg, with_axis=True), source=source, client=client)
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(result.overall_status, "no_ambiguity")
            self.assertTrue(result.no_ambiguity_confirmation.judge_confirmed)
            self.assertFalse(result.no_ambiguity_confirmation.human_confirmed)
            outcomes.append(gate_outcome(result, seg))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_migrated_batch_uses_the_same_candidate_gate_for_either_source(self):
        seg = segmentation()
        for source in SOURCES:
            review = self.finish(proposal(seg, with_axis=True), source=source,
                                 client=AssistanceClient(reading_decision="U"))
            item = {"status": "approved", "source_hash": seg.source_hash,
                    "segmentation_id": seg.segmentation_id, "reading_review_id": review.reading_review_id}
            with tempfile.TemporaryDirectory() as folder, \
                 patch("pre_ir_batch_review.load_segmentation_reviewed_checkpoint",
                       return_value={"surface_segmentation": asdict(seg)}), \
                 patch("pre_ir_batch_review.load_plain_reading_reviewed_checkpoint",
                       return_value={"plain_reading_review": asdict(review)}), \
                 patch("pre_ir_batch_review._semantic_descendant_for_reading", return_value=None):
                _reconcile_migrated_plain_reading_readiness(Path(folder), {"items": {seg.requirement_id: item}})
            self.assertEqual(item["next_stage"], "candidate_ir")
            self.assertFalse(item["single_ir_ready"])
            self.assertFalse(item["finalization_allowed"])

    def test_inspection_detected_ambiguity_keeps_same_gate_and_actual_reviewer(self):
        seg = segmentation()
        axis = proposal(seg, with_axis=True).ambiguities[0]

        def detect(raw):
            if "decision" not in raw:
                return raw
            raw["decision"] = "ambiguity_found"
            raw["ambiguities"] = [{
                "question": axis.question,
                "source_quote": {"start": 0, "end": len(SOURCE), "text": SOURCE},
                "options": [{"plain_interpretation": option.plain_interpretation,
                             "difference_summary": option.difference_summary} for option in axis.options],
            }]
            return raw

        outcomes, requests = [], []
        for source in SOURCES:
            client = RecordingClient(transform=detect, reading_decision="U")
            result = self.finish(proposal(seg), source=source, client=client)
            self.assertEqual(result.overall_status, "ambiguous_unresolved")
            self.assertTrue(assess_plain_reading_ir_readiness(result, seg).can_attempt_candidate_ir)
            self.assertEqual(result.ambiguities[0].reviewer_source, source)
            self.assertEqual(result.ambiguities[0].reviewer_id, client.model)
            self.assertEqual(len(client.requests), 2)
            self.assertTrue(any(item.get("operation") == "model_reading_evidence_before_option_adjudication"
                                for item in result.review_history))
            with self.assertRaises(PlainReadingValidationError):
                assert_plain_reading_ready_for_finalization(result, seg)
            outcomes.append(gate_outcome(result, seg))
            requests.append(client.requests)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(requests[0], requests[1])

    def test_successful_model_call_with_missing_decision_never_passes(self):
        for source in SOURCES:
            with self.subTest(source=source), self.assertRaises(LLMResponseSchemaError):
                self.finish(proposal(segmentation()), source=source,
                            client=AssistanceClient(transform=lambda _raw: {}))

    def test_model_request_for_source_clarification_blocks_both_sources(self):
        def clarify(raw):
            raw["decision"] = "needs_source_clarification"
            return raw
        for source in SOURCES:
            with self.subTest(source=source), self.assertRaises(PlainReadingValidationError):
                self.finish(proposal(segmentation()), source=source, client=AssistanceClient(transform=clarify))

    def test_stale_parent_is_rejected_before_any_model_request(self):
        for source in SOURCES:
            client = RecordingClient()
            stale = replace(proposal(segmentation(), with_axis=True), segmentation_id="stale-segmentation")
            with self.subTest(source=source), self.assertRaises(PlainReadingValidationError):
                self.finish(stale, source=source, client=client)
            self.assertEqual(client.requests, [])

    def test_unconfirmed_empty_proposal_does_not_pass_for_either_audit_source(self):
        seg = segmentation()
        for source in SOURCES:
            pending = replace(proposal(seg), reviewer_source=source)
            with self.subTest(source=source), self.assertRaises(PlainReadingValidationError):
                assert_plain_reading_gate(pending, seg)

    def test_missing_reviewer_cannot_be_silently_accepted(self):
        for source in SOURCES:
            with self.subTest(source=source), self.assertRaises(PreIRGateError) as error:
                self.finish(proposal(segmentation()), source=source, client=None)
            self.assertEqual(error.exception.status, "reviewer_unavailable")

    def test_unknown_or_duplicate_option_decisions_are_rejected_for_both_sources(self):
        def duplicate(raw):
            raw["decisions"].append(deepcopy(raw["decisions"][0]))
            return raw
        for source in SOURCES:
            for client in (AssistanceClient(reading_decision="unknown"), AssistanceClient(transform=duplicate)):
                with self.subTest(source=source), self.assertRaises(LLMResponseSchemaError):
                    self.finish(proposal(segmentation(), with_axis=True), source=source, client=client)


if __name__ == "__main__":
    unittest.main()
