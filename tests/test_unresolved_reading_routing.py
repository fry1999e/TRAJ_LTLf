from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cli
from agents.nlu_agent import NLUAgent
from ap_binding_context import build_runtime_ap_binding_context
from ap_preparation import prepare_reviewed_ap_context
from constructive_ir_builder import bind_plain_review_to_source_axes
from ir_proposal_router import IRProposalRouter
from plain_reading import PlainReadingSemanticResolutionRequired, assess_plain_reading_ir_readiness
from pre_ir_review import complete_plain_reading_with_model
from rag.retriever import KeywordRetriever
from tests.test_manual_release import MANUAL_ARGS
from tests.test_nonexpert_review import AssistanceClient, SOURCE, proposal, segmentation


class MappingClient(AssistanceClient):
    def __init__(self, mapping_transform=None):
        super().__init__(reading_decision="U")
        self.mapping_transform = mapping_transform

    def complete_json(self, system, payload, **kwargs):
        if payload.get("task") == "extract_source_ap_candidates":
            return {"aps": [{"canonical_name": "p", "evidence_texts": ["p"]}]}
        raw = super().complete_json(system, payload, **kwargs)
        if payload.get("task") == "map_plain_reading_options_to_source_axes" and self.mapping_transform:
            raw = self.mapping_transform(deepcopy(raw))
        return raw


def parents(client):
    seg = segmentation()
    review = complete_plain_reading_with_model(SOURCE, seg, proposal(seg, with_axis=True, chinese=True), client)
    with redirect_stdout(io.StringIO()):
        prepared = prepare_reviewed_ap_context(build_runtime_ap_binding_context(source_text=SOURCE, requirement_id="REQ-001"),
            client, segmentation=seg, interactive=True, input_fn=lambda _prompt: "a")
    return seg, review, prepared.context


class UnresolvedReadingRoutingTests(unittest.TestCase):
    def test_mapping_cannot_add_ir_actions_or_collapse_two_options(self):
        def collapse(raw):
            raw["mappings"][0]["bindings"][1]["source_option_id"] = "current_or_future"
            return raw

        def inject(raw):
            raw["mappings"][0]["bindings"][0]["operation"] = "invent_new_ir"
            return raw

        for transform in (collapse, inject):
            with self.subTest(transform=transform.__name__):
                client = MappingClient(transform)
                seg, review, context = parents(client)
                built = IRProposalRouter().propose(SOURCE, context, surface_segmentation=seg,
                                                   plain_reading_review=review, proposal_client=client)
                gate, reverse = bind_plain_review_to_source_axes(review, built, mapping_client=client)
                self.assertEqual((gate, reverse), ({}, {}))
                self.assertFalse(assess_plain_reading_ir_readiness(review, seg).can_enter_semantic_ir)
                self.assertEqual(sum(call.get("task") == "map_plain_reading_options_to_source_axes" for call in client.calls), 2)

    def test_unrepresentable_reading_becomes_typed_pending_without_approval(self):
        def decline(raw):
            raw["mappings"][0].update(source_axis_id="", bindings=[])
            return raw

        client = MappingClient(decline)
        seg, review, context = parents(client)
        with self.assertRaises(PlainReadingSemanticResolutionRequired) as caught:
            NLUAgent(KeywordRetriever([]), llm_client=client).generate(SOURCE,
                surface_segmentation=seg, plain_reading_review=review, ap_binding_context=context)
        self.assertIs(caught.exception.review, review)
        self.assertEqual(caught.exception.segmentation, seg)
        self.assertEqual(review.overall_status, "ambiguous_unresolved")

    def test_cli_saves_pending_reading_and_exits_without_error_or_final_formula(self):
        client = MappingClient()
        seg, review, _ = parents(client)
        pending = PlainReadingSemanticResolutionRequired(review,
            assess_plain_reading_ir_readiness(review, seg), segmentation=seg)
        for progress in ("auto", "off"):
            with self.subTest(progress=progress), tempfile.TemporaryDirectory() as folder:
                with (
                    patch.object(sys, "argv", ["main.py", *MANUAL_ARGS, "--run-dir", folder, "--progress", progress, SOURCE]),
                    patch.object(sys.stdin, "isatty", return_value=True),
                    patch.object(cli, "_prepare_artifacts"),
                    patch.object(cli, "run_traj_ltlf_pipeline", side_effect=pending),
                    redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as error,
                ):
                    cli.main()
                self.assertNotIn("[ERR]", output.getvalue())
                self.assertNotIn("Traceback", error.getvalue())
                self.assertIn("无需填写专业语义表单", output.getvalue())
                saved = json.loads(next((Path(folder) / "review_records").glob("plain_reading_pending_*.json")).read_text(encoding="utf-8"))
                self.assertEqual(saved["plain_reading_review"]["overall_status"], "ambiguous_unresolved")
                self.assertFalse((Path(folder) / "outputs/result.txt").exists())


if __name__ == "__main__":
    unittest.main()
