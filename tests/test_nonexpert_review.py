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

from agents.review_loop import run_semantic_review
from llm.errors import LLMClientError
from nonexpert_review import review_traces_with_simple_choices
from human_review_session import REVIEW_STOP_DEBTS
from plain_reading import (
    assert_plain_reading_gate, build_plain_reading_review,
    plain_reading_from_mapping,
)
from pre_ir_review import (
    complete_plain_reading_with_model, run_pre_ir_review, resume_plain_reading_review,
    PreIRReviewConfig,
)
from pre_ir_batch_review import _route_item_by_readiness
from rag.retriever import KeywordRetriever
from schemas import (
    ArtifactGenerator, NoAmbiguityConfirmation, PlainReadingAmbiguity, PlainReadingOption,
    SemanticStructure, TraceQuery, TRACE_PURPOSE_OPERATOR_DISCRIMINATION,
)
from surface_segmentation import build_surface_segmentation


SOURCE = "Eventually p"


def segmentation():
    return build_surface_segmentation(requirement_id="REQ-001", source_text=SOURCE,
        marked_text=SOURCE, marker="⟦CUT⟧", status="surface_reviewed")


def semantic():
    return SemanticStructure(trigger="p", response="p", temporal_keyword="eventually",
                             scope_keyword="", pattern="eventually", aps=("p",))


def brief():
    return SimpleNamespace(ap_definitions=(("p", "p occurs"),), source_text=SOURCE)


def proposal(seg, with_axis=False, chinese=False):
    axes = ()
    if with_axis:
        axes = (PlainReadingAmbiguity(
            ambiguity_id="AX-1", source_segment_ids=tuple(segment.segment_id for segment in seg.segments),
            question="Does the current step count?", source_span=(0, len(seg.source_text)),
            evidence_text=seg.source_text, options=(
                PlainReadingOption("OPT-now", "可以在当前时刻或以后发生。" if chinese else "The present step also counts.", "包含当前时刻。" if chinese else "Current step included."),
                PlainReadingOption("OPT-later", "必须在严格靠后的时刻发生。" if chinese else "Only a later step counts.", "不包含当前时刻。" if chinese else "Current step excluded."),
            )),)
    return build_plain_reading_review(
        requirement_id=seg.requirement_id, segmentation=seg, ambiguities=axes,
        generator=ArtifactGenerator(mode="api"), review_basis="agent_detection",
        experiment_mode="manual",
        **({} if axes else {"overall_status": "pending_no_ambiguity_confirmation",
                               "no_ambiguity_confirmation": NoAmbiguityConfirmation(
                                   status="pending", llm_proposed_no_ambiguity=True)}),
    )


class AssistanceClient:
    model = "fixture-main-model"

    def __init__(self, transform=None, reading_decision="OPT-now"):
        self.calls = []
        self.transform = transform
        self.reading_decision = reading_decision

    def complete_json(self, system, payload, **kwargs):
        json.dumps(payload)  # Real clients also require JSON-serializable input.
        self.calls.append(deepcopy(payload))
        if payload.get("task") == "explain_human_trace_choices":
            result = {"annotations": [{
                "query_id": choice["query_id"], "label": choice["locked_label"],
                "confidence": "medium", "rationale": "模型根据原文解释该人工选择。",
                "temporal_simulation_summary": "模型按列出的有限步骤整理判断。",
            } for choice in payload["human_choices"]]}
        elif payload.get("task") == "model_semantic_structure_check":
            result = {"approved": True, "reason": "Source-aligned structure."}
        elif payload.get("task") == "model_plain_reading_check":
            result = {"decisions": [{"ambiguity_id": axis["ambiguity_id"],
                "decision": self.reading_decision, "reason": "Source-supported reading."}
                for axis in payload["ambiguities"]]}
        elif payload.get("task") == "map_plain_reading_options_to_source_axes":
            mappings = []
            for axis in payload["axes"]:
                source_axis = next(item for item in axis["allowed_source_axes"] if item["axis_kind"] == "eventuality_policy")
                mappings.append({"ambiguity_id": axis["ambiguity_id"], "source_axis_id": source_axis["axis_id"],
                    "bindings": [{"plain_option_id": axis["plain_options"][0]["option_id"], "source_option_id": "current_or_future"},
                                 {"plain_option_id": axis["plain_options"][1]["option_id"], "source_option_id": "strict_future"}],
                    "reason": "Exact correspondence to the supplied temporal policy alternatives."})
            result = {"mappings": mappings}
        elif payload.get("proposal") == "no material ambiguity detected":
            source = payload["source_text"]
            result = {"decision": "confirm_no_ambiguity", "reason": "The source is determinate.",
                "source_quotes": [{"start": 0, "end": len(source), "text": source}], "ambiguities": []}
        else:
            raise AssertionError(f"Unexpected fixture task: {payload.get('task')}")
        return self.transform(result) if self.transform else result


class SimpleTrajectoryTests(unittest.TestCase):
    def review(self, answers, *, client=None, queries=None, folder=None):
        queries = queries or [TraceQuery(f"QUERY-{i}", "positive", "test", (), [{"p"}, *(set() for _ in range(i))],
                                       beginner_view="MISLEADING GENERATED PROSE") for i in range(len(answers))]
        client = client or AssistanceClient()
        with patch("builtins.input", side_effect=answers) as read, redirect_stdout(io.StringIO()) as output:
            annotations = review_traces_with_simple_choices(requirement=SOURCE, semantic=None,
                queries=queries, round_index=1, records_dir=folder,
                llm_client=client, adjudication_brief=brief())
        return annotations, output.getvalue(), read

    def test_truth_tables_show_empty_steps_and_keep_the_same_ap_columns(self):
        queries = [
            TraceQuery("TABLE-1", "boundary", "test", (), [{"request", "grant"}, set(), set()]),
            TraceQuery("TABLE-2", "boundary", "test", (), [{"request"}, set()]),
        ]
        meanings = SimpleNamespace(ap_definitions=(("request", "request occurs"), ("grant", "grant occurs")))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
            annotations = review_traces_with_simple_choices(
                requirement="Whenever request occurs, grant should eventually occur.", semantic=None,
                queries=queries, round_index=1, records_dir=folder, llm_client=AssistanceClient(),
                adjudication_brief=meanings, input_fn=Mock(side_effect=["n", "y"]),
            )
            saved = json.loads(next(Path(folder).glob("nonexpert_trace_review_*.json")).read_text(encoding="utf-8"))
        rendered = output.getvalue()
        headers = [[cell.strip() for cell in line.split("|")] for line in rendered.splitlines() if line.startswith("时刻")]
        rows = [[cell.strip() for cell in line.split("|")] for line in rendered.splitlines() if line.startswith("t")]
        self.assertEqual(headers, [["时刻", "request", "grant"]] * 2)
        self.assertEqual(rows, [["t0", "true", "false"], ["t1", "false", "false"],
                                ["t0", "true", "true"], ["t1", "false", "false"], ["t2", "false", "false"]])
        self.assertNotIn("这些子命题均不成立", rendered)
        self.assertEqual(saved["trace_display"]["ap_columns"], ["request", "grant"])
        self.assertEqual(saved["queries"][0]["trace"], [["grant", "request"], [], []])
        self.assertEqual([annotation.label for annotation in annotations], ["SAT", "UNSAT"])

    def test_four_plain_choices_have_model_metadata_and_saved_human_verdicts(self):
        with tempfile.TemporaryDirectory() as folder:
            client = AssistanceClient()
            annotations, output, read = self.review(["y", "n", "u", "i"], client=client, folder=folder)
            saved = json.loads(next(Path(folder).glob("nonexpert_trace_review_*.json")).read_text(encoding="utf-8"))
        self.assertEqual([item.label for item in annotations], ["SAT", "UNSAT", "UNCLEAR", "INFEASIBLE"])
        self.assertTrue(all(item.label_source == "human" and item.model == client.model for item in annotations))
        self.assertEqual(saved["metadata_source"], "model_assistant")
        self.assertEqual([item["verdict"] for item in saved["human_choices"]], ["符合", "不符合", "无法判断", "情景不可能"])
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(read.call_count, 4)
        self.assertNotIn("MISLEADING", output)
        for technical in ("SAT", "UNSAT", "置信度", "理由", "语义结构"):
            self.assertNotIn(technical, output + "".join(call.args[0] for call in read.call_args_list))
        payload = client.calls[0]
        self.assertNotIn("semantic", payload)
        self.assertNotIn("formula", payload)

    def test_blank_and_technical_labels_do_not_count_as_human_choices(self):
        query = TraceQuery("QUERY-1", "positive", "test", (), [{"p"}])
        with tempfile.TemporaryDirectory() as folder:
            annotations, _, read = self.review(["", "SAT", "y"], queries=[query], folder=folder)
        self.assertEqual(read.call_count, 3)
        self.assertEqual(annotations[0].label, "SAT")

    def test_model_cannot_reverse_human_verdict_and_choice_survives_failure(self):
        def reverse(result):
            result["annotations"][0]["label"] = "UNSAT"
            return result
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(LLMClientError):
                self.review(["y"], client=AssistanceClient(reverse), folder=folder)
            saved = json.loads(next(Path(folder).glob("nonexpert_trace_review_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(saved["human_choices"][0]["verdict"], "符合")
            self.assertEqual(saved["status"], "model_metadata_failed")
            self.assertNotIn("annotations", saved)

    def test_duplicate_or_missing_model_annotations_are_rejected(self):
        def duplicate(result):
            result["annotations"][1] = result["annotations"][0]
            return result
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(LLMClientError):
                self.review(["y", "n"], client=AssistanceClient(duplicate), folder=folder)

    def test_metadata_failure_does_not_invent_confidence_or_ask_for_expert_fields(self):
        client = AssistanceClient()
        client.complete_json = Mock(side_effect=LLMClientError("fixture transport failure"))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(LLMClientError):
                self.review(["n"], client=client, folder=folder)
            saved = json.loads(next(Path(folder).glob("nonexpert_trace_review_*.json")).read_text(encoding="utf-8"))
        self.assertEqual(saved["human_choices"][0]["locked_label"], "UNSAT")
        self.assertNotIn("annotations", saved)

    def test_annotation_purpose_survives_ir_and_operator_discrimination(self):
        queries = [
            TraceQuery("IRQ-1", "boundary", "test", (), [{"p"}], purpose="ir_branch_discrimination", parent_candidate_set_id="set-1"),
            TraceQuery("OPQ-1", "boundary", "test", (), [{"p"}], purpose=TRACE_PURPOSE_OPERATOR_DISCRIMINATION, ir_version="ir-1"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            annotations, _, read = self.review(["y"], queries=queries, folder=folder)
        self.assertEqual([a.purpose for a in annotations], [q.purpose for q in queries])
        self.assertEqual(read.call_count, 1)
        self.assertEqual([a.label for a in annotations], ["SAT", "SAT"])


class ModelSemanticWorkTests(unittest.TestCase):
    def test_semantic_structure_is_checked_without_human_prompt(self):
        client = AssistanceClient()
        agent = SimpleNamespace(llm_enabled=True, llm_client=client,
                                generate=Mock(return_value=semantic()), retriever=KeywordRetriever([]))
        with tempfile.TemporaryDirectory() as folder, patch("builtins.input", side_effect=AssertionError("expert prompt")):
            result, records = run_semantic_review(SOURCE, agent, records_dir=folder)
        self.assertEqual(result.label_source, "model_assistant")
        self.assertEqual(records[0]["reviewer_source"], "model_assistant")
        self.assertEqual(client.calls[0]["task"], "model_semantic_structure_check")

    def test_model_semantic_rejection_feeds_bounded_regeneration(self):
        count = 0
        def second_pass(result):
            nonlocal count
            count += 1
            return {"approved": count > 1, "reason": "Check temporal scope."}
        agent = SimpleNamespace(llm_enabled=True, llm_client=AssistanceClient(second_pass),
                                generate=Mock(return_value=semantic()), retriever=KeywordRetriever([]))
        with tempfile.TemporaryDirectory() as folder, patch("builtins.input", side_effect=AssertionError("expert prompt")):
            result, records = run_semantic_review(SOURCE, agent, max_rounds=2, records_dir=folder)
        self.assertEqual(agent.generate.call_count, 2)
        self.assertFalse(records[0]["approved"])
        self.assertTrue(records[1]["approved"])
        self.assertEqual(records[0]["stage_errors"][0]["source"], "model_assistant")

    def test_no_ambiguity_model_evidence_round_trips_without_human_confirmation(self):
        seg = segmentation()
        with patch("builtins.input", side_effect=AssertionError("expert prompt")):
            result = complete_plain_reading_with_model(SOURCE, seg, proposal(seg), AssistanceClient())
        self.assertEqual(result.reviewer_source, "model_assistant")
        self.assertFalse(result.no_ambiguity_confirmation.human_confirmed)
        restored = plain_reading_from_mapping(asdict(result), seg)
        assert_plain_reading_gate(restored, seg)
        self.assertEqual(restored.reviewer_source, "model_assistant")
        self.assertIn("no-ambiguity-model-assistant-v1", restored.no_ambiguity_confirmation.judge_evidence_json)

    def test_plain_reading_selection_and_pending_resume_use_main_model(self):
        seg = segmentation()
        pending = proposal(seg, with_axis=True)
        with patch("builtins.input", side_effect=AssertionError("expert prompt")):
            result = resume_plain_reading_review(requirement=SOURCE, segmentation=seg,
                review=pending, retriever=KeywordRetriever([]), llm_client=AssistanceClient(), interactive=True)
        self.assertEqual(result.ambiguities[0].decision, "OPT-now")
        self.assertEqual(result.ambiguities[0].reviewer_source, "model_assistant")

    def test_unresolved_model_reading_only_enters_candidate_path(self):
        seg = segmentation()
        result = complete_plain_reading_with_model(SOURCE, seg, proposal(seg, with_axis=True), AssistanceClient(reading_decision="U"))
        item = {}
        _route_item_by_readiness(item, result, seg)
        self.assertEqual(item["next_stage"], "candidate_ir")
        self.assertFalse(item["finalization_allowed"])
        self.assertFalse(item["single_ir_ready"])

    def test_malformed_reading_decisions_cannot_silently_approve(self):
        seg = segmentation()
        with self.assertRaises(LLMClientError):
            complete_plain_reading_with_model(SOURCE, seg, proposal(seg, with_axis=True), AssistanceClient(reading_decision="unknown-option"))

    def test_pre_ir_keeps_only_surface_prompt_with_unchanged_manual_parameters(self):
        seg = segmentation()
        with (
            patch("pre_ir_review.SegmentationAgent.generate", return_value=seg),
            patch("pre_ir_review.generate_plain_reading_outcome", side_effect=lambda _agent, _requirement, current_seg, **_kwargs: SimpleNamespace(review=proposal(current_seg))),
            patch("builtins.input", return_value="a") as read,
            redirect_stdout(io.StringIO()),
        ):
            result = run_pre_ir_review(requirement_id="REQ-001", requirement=SOURCE,
                retriever=KeywordRetriever([]), llm_client=AssistanceClient(),
                config=PreIRReviewConfig(segmentation_review="manual", plain_reading_review="manual"), interactive=True)
        self.assertEqual(read.call_count, 1)
        self.assertIn("表面切分", read.call_args.args[0])
        self.assertEqual(result.plain_reading_review.reviewer_source, "model_assistant")


class NonexpertWorkflowTests(unittest.TestCase):
    def test_deferred_segmentation_completes_model_reading_and_resumes_without_expert_forms(self):
        from pre_ir_batch_review import ensure_deferred_pre_ir_batch
        from requirement_loader import RequirementRecord

        for unresolved in (False, True):
            with self.subTest(unresolved=unresolved), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                human_file = root / "human.md"
                human_file.write_text(
                    "format: traj-ltlf-human-segmentation\nschema_version: 1\n\n## REQ-001\n"
                    f"<!-- source_text:start -->\n```text\n{SOURCE}\n```\n<!-- source_text:end -->\n"
                    f"<!-- human_marked_text:start -->\n```text\n{SOURCE}\n```\n<!-- human_marked_text:end -->\n",
                    encoding="utf-8",
                )
                with (
                    patch("pre_ir_batch_review.SegmentationAgent.generate", return_value=segmentation()),
                    patch("pre_ir_batch_review.generate_plain_reading_outcome", side_effect=lambda _agent, _requirement, seg, **_kw: SimpleNamespace(review=proposal(seg, with_axis=unresolved))),
                    patch("builtins.input", side_effect=AssertionError("expert prompt")),
                ):
                    kwargs = dict(human_segmentation_file=str(human_file), checkpoint_dir=str(root / "checkpoints"),
                        records_dir=str(root / "reviews"), llm_mode="api", llm_client=AssistanceClient(reading_decision="U"))
                    first = ensure_deferred_pre_ir_batch([RequirementRecord("REQ-001", SOURCE)], **kwargs)
                    self.assertFalse(first.should_stop, first.state.get("items"))
                    self.assertEqual(first.state["downstream_eligible"], ["REQ-001"])
                    if unresolved:
                        self.assertFalse(first.state["items"]["REQ-001"]["finalization_allowed"])
                        from checkpointing import load_plain_reading_reviewed_checkpoint, load_segmentation_reviewed_checkpoint
                        from plain_reading import apply_plain_reading_decisions
                        from surface_segmentation import segmentation_from_mapping
                        from pre_ir_batch_review import persist_trace_selected_plain_reading
                        bound_seg = segmentation_from_mapping(load_segmentation_reviewed_checkpoint(kwargs["checkpoint_dir"], "REQ-001")["surface_segmentation"])
                        pending = plain_reading_from_mapping(load_plain_reading_reviewed_checkpoint(kwargs["checkpoint_dir"], "REQ-001")["plain_reading_review"], bound_seg)
                        closed = apply_plain_reading_decisions(pending, bound_seg, {"AX-1": "OPT-now"},
                            reviewer_source="human_trace", review_basis="human_trajectory_selected_reading")
                        persist_trace_selected_plain_reading(checkpoint_dir=kwargs["checkpoint_dir"],
                            segmentation=bound_seg, previous_review_id=pending.reading_review_id, review=closed)
                    second = ensure_deferred_pre_ir_batch([RequirementRecord("REQ-001", SOURCE)], **kwargs)
                    self.assertFalse(second.should_stop)
                    if unresolved:
                        self.assertEqual(second.state["items"]["REQ-001"]["reading_review_id"], closed.reading_review_id)

    def test_single_and_batch_workflows_ask_only_surface_ap_and_simple_trace_choices(self):
        from batch_review import run_batch_semantic_trace_review
        from pipeline import run_traj_ltlf_pipeline
        from tests.test_ap_preparation import APS, TEXT, MARKED, graph

        class Client(AssistanceClient):
            def complete_json(self, system, payload, **kwargs):
                if payload.get("task") == "extract_source_ap_candidates":
                    return deepcopy(APS) if payload["source_text"] == TEXT else {
                        "aps": [{"canonical_name": "p", "evidence_texts": ["p"]}]}
                if "SourceIntentGraph" in system:
                    generated = graph() if payload["source_text"] == TEXT else {"source_intent_graph": {
                        "nodes": [
                            {"ref": "p", "kind": "AP", "source_span": [11, 12], "children": []},
                            {"ref": "e", "kind": "eventually", "source_span": [0, 12], "children": ["p"]},
                        ], "root_ref": "e", "semantic_debts": [],
                    }}
                    if getattr(self, "source_axis_debt", False):
                        generated["source_intent_graph"]["semantic_debts"] = [{
                            "reason_code": "semantic_axis", "source_span": [0, len(payload["source_text"])],
                            "detail": "Whether the current step or only a strictly later step counts for eventuality.",
                        }]
                    if getattr(self, "spurious_deadline", False):
                        generated["source_intent_graph"]["nodes"] = [node for node in generated["source_intent_graph"]["nodes"] if node["ref"] != "g"]
                        generated["source_intent_graph"]["root_ref"] = "i"
                        generated["source_intent_graph"]["semantic_debts"] = [{
                            "reason_code": "unsupported_semantics", "source_span": [38, 54],
                            "detail": "The phrase 'eventually occur' does not specify a fixed or measurable deadline; the graph uses an unbounded eventual relation.",
                        }]
                    return generated
                if payload.get("task") == "map_source_semantic_debts_to_candidate_axes":
                    return {"resolutions": [{"debt_id": debt["debt_id"],
                        "axis_ids": [axis["axis_id"] for axis in debt["allowed_axes"] if axis["axis_kind"] == "eventuality_policy"],
                        "reason": "The complete supplied timing options express this difference."} for debt in payload["debts"]]}
                if getattr(self, "detected_ambiguity", False) and payload.get("proposal") == "no material ambiguity detected":
                    source = payload["source_text"]
                    return {"decision": "ambiguity_found", "reason": "Current versus strictly later is unresolved.",
                        "source_quotes": [{"start": 0, "end": len(source), "text": source}],
                        "ambiguities": [{"question": "是否包括当前时刻？",
                            "source_quote": {"start": 0, "end": len(source), "text": source},
                            "options": [{"plain_interpretation": "可以在当前时刻或以后发生。", "difference_summary": "包含当前时刻。"},
                                        {"plain_interpretation": "必须在严格靠后的时刻发生。", "difference_summary": "不包含当前时刻。"}]}]}
                if getattr(self, "spurious_deadline", False):
                    if payload.get("proposal") == "no material ambiguity detected":
                        return {"decision": "ambiguity_found", "reason": "Check advice versus obligation and deadline versus unbounded timing.",
                            "source_quotes": [{"start": 0, "end": len(TEXT), "text": TEXT}],
                            "ambiguities": [{"question": question,
                                "source_quote": {"start": 0, "end": len(TEXT), "text": TEXT},
                                "options": [{"plain_interpretation": first, "difference_summary": "First reading."},
                                            {"plain_interpretation": second, "difference_summary": "Second reading."}]}
                                for question, first, second in (
                                    ("Requirement or advice?", "Grant is required after each request.", "Grant is merely recommended."),
                                    ("Fixed deadline or unbounded?", "Grant has a fixed deadline.", "Grant must occur without a fixed deadline."),
                                )]}
                    if payload.get("task") == "model_plain_reading_check":
                        return {"decisions": [{"ambiguity_id": axis["ambiguity_id"],
                            "decision": axis["options"][index]["option_id"],
                            "reason": "Only the requirement reading without an invented deadline is source-supported."}
                            for index, axis in enumerate(payload["ambiguities"])]}
                return super().complete_json(system, payload, **kwargs)

        for source, batch in ((SOURCE, False), (SOURCE, True), (TEXT, False), (TEXT, True)):
            cases = (False, True, "detected", "source_axis_debt", "spurious_deadline") if source == TEXT else (False, True, "detected")
            for unresolved in cases:
                with self.subTest(source=source, batch=batch, unresolved=unresolved), tempfile.TemporaryDirectory() as folder:
                    record_root = Path(folder) / "reviews"
                    client = Client(reading_decision="U" if unresolved else "OPT-now")
                    client.detected_ambiguity = unresolved == "detected"
                    client.source_axis_debt = unresolved == "source_axis_debt"
                    client.spurious_deadline = unresolved == "spurious_deadline"
                    prompts = []

                    def answer(prompt=""):
                        prompts.append(prompt)
                        if "表面切分" in prompt or "AP" in prompt:
                            return "a"
                        if "请选择：y" in prompt:
                            files = list(record_root.rglob("nonexpert_trace_review_*.json"))
                            latest = max(files, key=lambda path: (path.stat().st_mtime_ns, path.name))
                            saved = json.loads(latest.read_text(encoding="utf-8"))
                            trace = next(query["trace"] for query in saved["queries"]
                                         if query["query_id"] == saved["current_query_id"])
                            # Simulated user judgment of these two fixture sentences;
                            # no production label or candidate evaluator is used.
                            if source == SOURCE:
                                fits = any("p" in step for step in trace)
                            else:
                                fits = all(any("grant_occurs" in later for later in trace[index:])
                                           for index, step in enumerate(trace) if "request_occurs" in step)
                            return "y" if fits else "n"
                        raise AssertionError("Unexpected expert prompt: " + prompt)

                    def generated_segmentation(_agent_self=None, *args, **kwargs):
                        return build_surface_segmentation(requirement_id=kwargs.get("requirement_id", "REQ-001"),
                            source_text=source, marked_text=MARKED if source == TEXT else source, marker="⟦CUT⟧", status="surface_reviewed")

                    with (
                        patch("pre_ir_review.SegmentationAgent.generate", side_effect=generated_segmentation),
                        patch("pre_ir_review.generate_plain_reading_outcome", side_effect=lambda _agent, _requirement, seg, **_kw: SimpleNamespace(review=proposal(seg, with_axis=unresolved is True, chinese=bool(unresolved)))),
                        patch("builtins.input", side_effect=answer), redirect_stdout(io.StringIO()),
                    ):
                        common = dict(llm_client=client, review_mode="manual", records_dir=str(record_root),
                                      segmentation_review="manual", plain_reading_review="manual")
                        if batch:
                            item = run_batch_semantic_trace_review([source],
                                checkpoint_dir=str(Path(folder) / "checkpoints"), **common)[0]
                            self.assertIsNone(item.pre_ir_error)
                            self.assertIsNone(item.needs_review_error)
                            self.assertTrue(item.traces or any(
                                (record.get("review_budget") or record.get("human_review_budget") or {}).get("fallback_evidence")
                                for record in item.review_records))
                            self.assertEqual(item.semantic.label_source, "model_assistant")
                            if unresolved in (True, "detected", "source_axis_debt", "spurious_deadline"):
                                if unresolved in (True, "detected"):
                                    self.assertEqual(item.plain_reading_review.overall_status, "plain_reading_reviewed")
                                    self.assertEqual(item.plain_reading_review.reviewer_source, "human_trace")
                                elif unresolved == "source_axis_debt":
                                    self.assertEqual(len(item.semantic.source_axis_debt_resolutions), 1)
                                else:
                                    self.assertEqual(item.plain_reading_review.overall_status, "plain_reading_reviewed")
                                    self.assertEqual(item.plain_reading_review.reviewer_source, "model_assistant")
                                calls_before_resume = len(client.calls)
                                with patch("builtins.input", side_effect=AssertionError("completed checkpoint must not reprompt")):
                                    resumed = run_batch_semantic_trace_review([source],
                                        checkpoint_dir=str(Path(folder) / "checkpoints"), **common)[0]
                                self.assertEqual(
                                    [(trace.trace, trace.adjudicated_label) for trace in resumed.traces],
                                    [(trace.trace, trace.adjudicated_label) for trace in item.traces])
                                self.assertEqual(resumed.plain_reading_review.reading_review_id, item.plain_reading_review.reading_review_id)
                                self.assertEqual(len(client.calls), calls_before_resume)
                                self.assertEqual(resumed.semantic.source_axis_debt_resolutions, item.semantic.source_axis_debt_resolutions)
                        else:
                            result = run_traj_ltlf_pipeline(source, llm_mode="api", experiment_mode="human",
                                approved_memory_file=None, **common)
                            self.assertIsNotNone(result.selected or result.provisional_candidate)
                            if result.semantic_status != "approved":
                                self.assertEqual(result.selection_method, "candidate_fallback")
                                self.assertFalse(result.automatic_finalization_allowed)
                                self.assertTrue(REVIEW_STOP_DEBTS.intersection(result.trace_coverage.coverage_debt))
                            if not result.traces:
                                self.assertEqual(result.approval_status, "not_approved")
                                fallback = next(record for record in result.review_records
                                                if record.get("stage") == "formula_candidate_fallback")
                                self.assertGreater(fallback["raw_human_trace_count"], 0)
                            if unresolved in (True, "detected"):
                                self.assertEqual(result.semantic.plain_reading_review.overall_status, "plain_reading_reviewed")
                                self.assertEqual(result.semantic.plain_reading_review.reviewer_source, "human_trace")
                            if unresolved == "source_axis_debt":
                                self.assertEqual(len(result.semantic.source_axis_debt_resolutions), 1)
                            if unresolved == "spurious_deadline":
                                self.assertEqual(result.semantic.plain_reading_review.overall_status, "plain_reading_reviewed")
                                self.assertEqual(result.semantic.faithful_ir.node_map[result.semantic.faithful_ir.root_node_id].node_type,
                                                 "global_constraint")
                    self.assertTrue(any("表面切分" in prompt for prompt in prompts))
                    self.assertTrue(any("AP" in prompt for prompt in prompts))
                    self.assertTrue(any("请选择：y" in prompt for prompt in prompts))
                    self.assertLessEqual(sum("请选择：y" in prompt for prompt in prompts), 5)
                    self.assertTrue(any(call.get("task") == "model_semantic_structure_check" for call in client.calls))


if __name__ == "__main__":
    unittest.main()
