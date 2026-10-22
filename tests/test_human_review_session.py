from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from human_review_session import HumanReviewSession, MAX_SCENARIO_COUNT
from llm.errors import LLMClientError
from nonexpert_review import review_traces_with_simple_choices
from schemas import TraceQuery, TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION, TRACE_PURPOSE_OPERATOR_DISCRIMINATION
from tests.test_nonexpert_review import AssistanceClient


SOURCE = "Eventually p"


def brief(meaning="p occurs"):
    return SimpleNamespace(ap_definitions=(("p", meaning),), requirement_id="REQ-001")


def queries(count=10):
    return [TraceQuery(f"KEY-{index}", "positive" if index % 2 else "boundary", "fixture", (),
        [{"p"}, *(set() for _ in range(index))], obligation_ids=(f"obligation-{index}",)) for index in range(count)]


class HumanReviewSessionTests(unittest.TestCase):
    def review(self, folder, asked, *, client=None, read=None, semantic=None, source=SOURCE, meaning=None):
        return review_traces_with_simple_choices(requirement=source, semantic=semantic, queries=asked,
            round_index=1, records_dir=folder, llm_client=client or AssistanceClient(),
            adjudication_brief=meaning or brief(), input_fn=read or Mock(return_value="y"))

    def test_ten_targets_ask_at_most_five_and_resume_without_new_questions_or_model_calls(self):
        client, read = AssistanceClient(), Mock(return_value="y")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
            annotations = self.review(folder, queries(), client=client, read=read)
            count = len(client.calls)
            again = self.review(folder, queries(), client=client, read=Mock(side_effect=AssertionError("repeat")))
            records = [json.loads(path.read_text(encoding="utf-8"))
                       for path in Path(folder).glob("nonexpert_trace_review_*.json")]
            session = HumanReviewSession(SOURCE, None, brief(), folder)
        self.assertEqual(read.call_count, MAX_SCENARIO_COUNT)
        self.assertEqual(annotations, [])
        self.assertEqual(again, [])
        self.assertEqual(count, 0)
        self.assertEqual(len(session.answers), 5)
        self.assertTrue(all("annotation" not in entry for entry in session.answers.values()))
        self.assertEqual(session.summary()["metadata_status"], "deferred_for_candidate_fallback")
        self.assertEqual(len(client.calls), count)
        self.assertEqual(session.remaining, 0)
        self.assertTrue(all(len(record["deferred_query_ids"]) == 5 for record in records))
        self.assertIn("仅在需要时补充", output.getvalue())
        self.assertNotIn("1/10", output.getvalue())

    def test_identical_table_is_asked_once_and_rebound_to_each_original_purpose(self):
        asked = queries(1)
        asked.append(replace(asked[0], query_id="OP-duplicate", purpose=TRACE_PURPOSE_OPERATOR_DISCRIMINATION,
                             ir_version="ir-fixture", obligation_ids=("other-obligation",)))
        read = Mock(return_value="n")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            annotations = self.review(folder, asked, read=read)
        self.assertEqual(read.call_count, 1)
        self.assertEqual([item.label for item in annotations], ["UNSAT", "UNSAT"])
        self.assertEqual([item.purpose for item in annotations], [query.purpose for query in asked])

    def test_three_judgments_cover_key_targets_without_asking_extra_tables(self):
        from human_review_session import review_budget_state, review_should_stop
        asked = [replace(query, obligation_ids=(f"target-{index % 3}",), role="boundary")
                 for index, query in enumerate(queries(6))]
        read = Mock(return_value="y")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            annotations = self.review(folder, asked, read=read)
            budget = review_budget_state(SOURCE, None, brief(), folder)
        self.assertEqual(read.call_count, 3)
        self.assertEqual(len(annotations), 3)
        self.assertEqual(budget["stop_reason"], "key_targets_covered")
        self.assertEqual(budget["remaining"], 2)
        self.assertTrue(review_should_stop(budget))

    def test_stages_share_budget_and_print_the_introduction_only_once(self):
        first = replace(queries(1)[0], query_id="BRANCH-first", purpose=TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION,
                        parent_candidate_set_id="candidate-fixture")
        read = Mock(return_value="y")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
            self.review(folder, [first], read=read)
            self.review(folder, queries(), read=read)
        text = output.getvalue()
        self.assertEqual(read.call_count, 5)
        self.assertEqual(text.count("请判断表格中的情景是否符合原始需求"), 1)
        self.assertIn("需求含义确认", text)
        self.assertIn("关键情景核对", text)
        self.assertIn("无需重复回答", text)

    def test_unique_branch_stops_after_one_answer_without_labeling_unasked_tables(self):
        from tests.test_candidate_delivery import unresolved_semantic, TEXT
        from semantic_freezing import attach_candidate_ir_set
        semantic = attach_candidate_ir_set(unresolved_semantic(), TEXT)
        asked = [TraceQuery(f"IR-{index}", "boundary", "timing", (), trace,
            purpose=TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION,
            parent_candidate_set_id=semantic.candidate_ir_set.branch_set_id or "fixture-timing-set")
            for index, trace in enumerate([
                [{"request_occurs", "grant_occurs"}],
                [{"request_occurs"}, {"grant_occurs"}],
                [{"request_occurs"}, set()],
            ])]
        meanings = SimpleNamespace(ap_definitions=(("request_occurs", "request occurs"), ("grant_occurs", "grant occurs")))
        read = Mock(return_value="y")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            annotations = self.review(folder, asked, read=read, semantic=semantic, source=TEXT, meaning=meanings)
            record = json.loads(next(Path(folder).glob("nonexpert_trace_review_*.json")).read_text(encoding="utf-8"))
        self.assertEqual(read.call_count, 1)
        self.assertEqual(len(annotations), 1)
        self.assertEqual(len(record["deferred_query_ids"]), 2)
        self.assertEqual(record["review_budget"]["stop_reason"], "unique_interpretation")

    def test_unknown_answer_is_reused_as_unknown_and_never_becomes_sat(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            first = self.review(folder, queries(1), read=Mock(return_value="u"))
            again = self.review(folder, [replace(queries(1)[0], query_id="other-id")],
                                read=Mock(side_effect=AssertionError("repeat unknown")))
        self.assertEqual(first[0].label, "UNCLEAR")
        self.assertEqual(again[0].label, "UNCLEAR")
        self.assertEqual(again[0].uncertainty_kind, "annotation_uncertainty")

    def test_model_failure_keeps_raw_answer_and_retries_metadata_without_reasking(self):
        failed = Mock(model="fixture", complete_json=Mock(side_effect=LLMClientError("fixture network")))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            with self.assertRaises(LLMClientError):
                self.review(folder, queries(1), client=failed, read=Mock(return_value="n"))
            annotations = self.review(folder, queries(1), read=Mock(side_effect=AssertionError("repeat")))
        self.assertEqual(annotations[0].label, "UNSAT")

    def test_source_or_ap_meaning_change_cannot_reuse_prior_judgment(self):
        read = Mock(return_value="y")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            self.review(folder, queries(1), read=read)
            self.review(folder, queries(1), read=read, source="Always p")
            self.review(folder, queries(1), read=read, meaning=brief("p is disabled"))
        self.assertEqual(read.call_count, 3)

    def test_full_batch_outputs_one_fallback_after_five_answers_and_resumes_without_reasking(self):
        from unittest.mock import patch
        from batch_pipeline import run_batch_traj_ltlf_pipeline
        from human_review_session import BUDGET_DEBT
        from pre_ir_review import PreIRArtifacts, complete_plain_reading_with_model
        from tests.test_ap_preparation import APS, TEXT, graph, segmentation
        from tests.test_nonexpert_review import proposal

        class PipelineClient(AssistanceClient):
            def complete_json(self, system, payload, **kwargs):
                if payload.get("task") in {"extract_source_ap_candidates", "propose_source_intent_graph"}:
                    self.calls.append(deepcopy(payload))
                    return deepcopy(APS if payload["task"] == "extract_source_ap_candidates" else graph())
                return super().complete_json(system, payload, **kwargs)

        client = PipelineClient()
        seg = segmentation()
        reading = complete_plain_reading_with_model(TEXT, seg, proposal(seg), client)
        prompts = []
        with tempfile.TemporaryDirectory() as folder:
            records = Path(folder) / "reviews"
            def answer(prompt=""):
                prompts.append(prompt)
                if "AP" in prompt:
                    return "a"
                current = max(records.glob("nonexpert_trace_review_*.json"), key=lambda path: path.stat().st_mtime_ns)
                record = json.loads(current.read_text(encoding="utf-8"))
                trace = next(query["trace"] for query in record["queries"] if query["query_id"] == record["current_query_id"])
                fits = all(any("grant_occurs" in step for step in trace[index:])
                           for index, step in enumerate(trace) if "request_occurs" in step)
                return "y" if fits else "n"
            with redirect_stdout(io.StringIO()) as output, \
                    patch("batch_review.run_pre_ir_review", return_value=PreIRArtifacts(seg, reading, effective_source_text=TEXT)), \
                    patch("builtins.input", side_effect=answer):
                common = dict(llm_client=client, llm_mode="api", review_mode="manual", experiment_mode="human",
                    records_dir=str(records), checkpoint_dir=str(Path(folder) / "checkpoints"),
                    classified_output_dir=str(Path(folder) / "outputs"), approved_memory_file=None)
                first = run_batch_traj_ltlf_pipeline([TEXT], **common)[0].result
                calls = len(client.calls)
                with patch("builtins.input", side_effect=AssertionError("repeat completed judgment")):
                    again = run_batch_traj_ltlf_pipeline([TEXT], **common)[0].result
                summary = (Path(folder) / "outputs" / "final_candidates.txt").read_text(encoding="utf-8")
            self.assertEqual(len(list(records.glob("human_trace_session_*.json"))), 1)
        self.assertEqual(sum("请选择：y" in prompt for prompt in prompts), 5)
        self.assertEqual(len(client.calls), calls)
        self.assertIn(BUDGET_DEBT, first.trace_coverage.coverage_debt)
        self.assertEqual(first.approval_status, "not_approved")
        fallback = next(record for record in first.review_records if record.get("stage") == "formula_candidate_fallback")
        self.assertEqual(fallback["human_trace_count"], 5)
        self.assertGreater(fallback["raw_human_trace_count"], 0)
        self.assertFalse(first.automatic_finalization_allowed)
        self.assertIsNone(first.final_approval)
        self.assertEqual(str(first.provisional_candidate.formula), str(again.provisional_candidate.formula))
        self.assertEqual(summary.count(str(first.provisional_candidate.formula)), 1)
        self.assertNotIn("needs_review", summary)
        self.assertEqual(output.getvalue().count("请判断表格中的情景是否符合原始需求"), 1)


if __name__ == "__main__":
    unittest.main()
