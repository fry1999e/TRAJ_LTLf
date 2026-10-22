from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from ap_binding_context import build_ap_binding_context
from ap_preparation import APPreparationError, ensure_ap_candidates, prepare_reviewed_ap_context
from tests.test_ap_preparation import Client


def proposal(**aps):
    return {"aps": [{"canonical_name": name, "evidence_texts": quotes} for name, quotes in aps.items()]}


def context(source):
    return build_ap_binding_context(requirement_id="REQ-001", source_text=source)


class GeneralAPRepairTests(unittest.TestCase):
    def test_domain_and_chinese_coordination_are_repaired_before_human_prompt(self):
        samples = [
            ("Whenever pump runs, pump and valve should eventually run.",
             proposal(pump_running=["pump runs"], pump_and_valve_running=["pump", "valve", "run"]),
             proposal(pump_running=["pump"], valve_running=["valve"])),
            ("当设备启动时，设备应当最终启动并且警灯亮起。",
             proposal(device_starts=["设备启动"], device_and_lamp=["设备", "警灯亮起"]),
             proposal(device_starts=["设备"], lamp_lit=["警灯亮起"])),
            ("The door opens and then the door closes.",
             proposal(door_opens=["door opens"], door_closes=["door", "closes"]),
             proposal(door_opens=["door opens"], door_closes=["door closes"])),
        ]
        for source, invalid, valid in samples:
            client = Client([invalid, valid])
            with self.subTest(source=source), redirect_stdout(io.StringIO()) as output:
                approved = prepare_reviewed_ap_context(context(source), client,
                    interactive=True, input_fn=lambda _: "a")
            self.assertEqual(len(client.calls), 2)
            feedback = client.calls[1]["ap_repair_feedback"]
            overlaps = [item for item in feedback["issues"] if item["code"] == "AP_EVIDENCE_OVERLAP"]
            self.assertTrue(overlaps)
            self.assertTrue(all(source[start:end] == item["overlap_text"]
                                for item in overlaps for start, end in [item["overlap_span"]]))
            self.assertEqual(set(approved.context.declared_catalog),
                             {item["canonical_name"] for item in valid["aps"]})
            self.assertNotIn("模型候选未通过", output.getvalue())
            self.assertEqual(approved.inventory.reviewer_source, "human")

    def test_unrelated_predicate_and_its_negative_qualifier_are_preserved(self):
        source = "If request arrives and cancel stays absent, request and grant occur."
        invalid = proposal(request_arrives=["request arrives"], both=["request", "grant occur"],
                           cancel_absent=["cancel stays absent"])
        valid = proposal(request_arrives=["request"], grant_occurs=["grant occur"],
                         cancel_absent=["cancel stays absent"])
        client = Client([invalid, valid])
        prepared = ensure_ap_candidates(context(source), client)
        self.assertEqual(client.calls[1]["ap_repair_feedback"]["preserve_candidates"], [invalid["aps"][-1]])
        self.assertFalse(prepared.context.has_authoritative_catalog)
        self.assertEqual(prepared.proposal_records[0]["status"], "invalid")
        self.assertEqual(prepared.proposal_records[1]["status"], "valid")

    def test_repair_cannot_drop_a_valid_predicate_or_its_qualifiers(self):
        source = "If request arrives and cancel stays absent, request and grant occur."
        invalid = proposal(request_arrives=["request arrives"], both=["request", "grant occur"],
                           cancel_absent=["cancel stays absent"])
        lossy = proposal(request_arrives=["request"], grant_occurs=["grant occur"], cancel_absent=["cancel"])
        client = Client([invalid, lossy])
        with self.assertRaises(APPreparationError) as caught:
            ensure_ap_candidates(context(source), client)
        self.assertEqual(len(client.calls), 2)
        final = caught.exception.diagnostics["failed_attempts"][-1]
        self.assertEqual(final["ap_repair_feedback"]["issues"][0]["code"], "AP_REPAIR_DROPPED_VALID_PREDICATE")

    def test_multiple_bad_quotes_are_reported_together_for_one_repair(self):
        source = "When lamp shines, fan turns."
        client = Client([proposal(lamp=["lamp glows"], fan=["fan rotates"]),
                         proposal(lamp=["lamp shines"], fan=["fan turns"])])
        prepared = ensure_ap_candidates(context(source), client)
        issues = client.calls[1]["ap_repair_feedback"]["issues"]
        self.assertEqual({item["code"] for item in issues}, {"AP_QUOTE_NOT_EXACT"})
        self.assertEqual(len(issues), 2)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(prepared.inventory.identities), 2)

    def test_identical_repeated_proposals_are_collapsed_without_extra_model_request(self):
        source = "When the alarm rings, the alarm rings again."
        single = proposal(alarm_rings=["alarm rings"])
        repeated = {"aps": single["aps"] * 2}
        client = Client([repeated])
        prepared = ensure_ap_candidates(context(source), client)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(prepared.inventory.identities), 1)
        self.assertEqual(len(prepared.inventory.mentions), 2)
        self.assertEqual(prepared.proposal_records[-1]["normalization"], "identical_proposals_deduplicated")
        self.assertEqual(repeated["aps"], single["aps"] * 2)

    def test_fragment_order_changes_do_not_change_an_ap_definition(self):
        from ap_proposal_validation import same_ap_proposal, deduplicate_identical_proposals
        left = {"canonical_name": "grant_occurs", "evidence_texts": ["grant", "occur"]}
        right = {"canonical_name": "grant_occurs", "evidence_texts": ["occur", "grant"]}
        self.assertTrue(same_ap_proposal(left, right))
        self.assertEqual(deduplicate_identical_proposals({"aps": [left, right]}), {"aps": [left]})

    def test_differently_named_proposals_are_not_silently_merged(self):
        source = "When the alarm rings, the alarm rings again."
        invalid = proposal(alarm_rings=["alarm rings"], other_state=["alarm rings"])
        client = Client([invalid])
        with self.assertRaises(APPreparationError) as caught:
            ensure_ap_candidates(context(source), client)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(caught.exception.diagnostics["failed_attempts"][-1]["proposal"], invalid)

    def test_compound_entity_name_is_not_rejected_just_because_it_contains_and(self):
        client = Client([proposal(valve_opens=["oil and gas valve opens"])])
        result = ensure_ap_candidates(context("The oil and gas valve opens."), client)
        self.assertEqual(len(result.inventory.identities), 1)
        self.assertEqual(len(client.calls), 1)

    def test_automatic_repairs_and_original_conflicts_are_saved_in_approval_record(self):
        source = "The door opens and then the door closes."
        client = Client([proposal(door_opens=["door opens"], door_closes=["door", "closes"]),
                         proposal(door_opens=["door opens"], door_closes=["door closes"])])
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            prepare_reviewed_ap_context(context(source), client, interactive=True,
                input_fn=lambda _: "a", records_dir=folder)
            record = json.loads(next(Path(folder).glob("ap_review_*.json")).read_text(encoding="utf-8"))
        self.assertEqual(len(record["proposal_generation"]), 2)
        self.assertTrue(record["proposal_generation"][0]["ap_repair_feedback"]["issues"])
        self.assertEqual(record["proposal_generation"][-1]["status"], "valid")
        self.assertEqual(record["ap_inventory"]["reviewer_source"], "human")


if __name__ == "__main__":
    unittest.main()
