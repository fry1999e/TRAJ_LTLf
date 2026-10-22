from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace

import cli
from experiment_modes import resolve_experiment_mode
from llm import LLMClientError, build_judge_client_from_env
from llm.client import build_llm_client_from_env
from llm.langchain_client import LangChainJSONClient
from pre_ir_review import PreIRReviewConfig
from runtime_config import load_runtime_config
from rag.knowledge_base import load_default_kb
from agents.prompt_context import KB_LTLF_SEMANTICS, _source_matches
from schemas import SemanticStructure, TraceQuery, TraceAnnotation, CandidateIRSet
from trace_review_protocol import manual_trace_annotations, independent_judge_annotations
from trace_evidence import TraceEvidenceDispute, TraceEvidenceReconciliation
from trace_conflict_resolution import resolve_trace_evidence_disputes
from ir_branch_discriminator import PolicyBranchSelectionOutcome
import ir_ambiguity_adjudication as ir_review

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_inputs import build_exports
from check_reproducibility import verify_data

MANUAL_ARGS = [
    "--review", "manual", "--experiment-mode", "human", "--llm", "api",
    "--segmentation-review", "manual", "--plain-reading-review", "manual",
]


class CapturedPipeline(Exception):
    pass


class ManualEntryTests(unittest.TestCase):
    def _capture_batch(self, arguments):
        # Stop at the real batch boundary: no model requests, labels, or files.
        with (
            patch.object(sys, "argv", ["main.py", *arguments]),
            patch.object(sys.stdin, "isatty", return_value=True),
            patch.object(cli, "prepare_checkpoint_run", return_value=None),
            patch.object(cli, "_prepare_artifacts"),
            patch.object(cli, "ensure_deferred_pre_ir_batch") as deferred,
            patch.object(cli, "run_batch_traj_ltlf_pipeline", side_effect=CapturedPipeline) as run,
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(CapturedPipeline):
                cli.main()
        deferred.assert_not_called()
        return run.call_args

    def test_requested_command_enters_interactive_human_batch(self):
        call = self._capture_batch(MANUAL_ARGS)
        self.assertEqual(len(call.args[0]), 2)
        for key, value in {
            "review_mode": "manual", "experiment_mode": "human", "llm_mode": "api",
            "segmentation_review": "manual", "plain_reading_review": "manual",
            "delivery_policy": "strict", "approved_memory_file": None,
        }.items():
            self.assertEqual(call.kwargs[key], value)
        self.assertEqual(Path(call.kwargs["records_dir"]), Path("data/output/manual/review_records"))

    def test_defaults_match_requested_command(self):
        call = self._capture_batch([])
        self.assertEqual(call.kwargs["review_mode"], "manual")
        self.assertEqual(call.kwargs["experiment_mode"], "human")
        self.assertEqual(call.kwargs["llm_mode"], "api")
        self.assertEqual(call.kwargs["segmentation_review"], "manual")
        self.assertEqual(call.kwargs["plain_reading_review"], "manual")

    def test_noninteractive_invocation_never_falls_back_to_auto(self):
        with (
            patch.object(sys, "argv", ["main.py", *MANUAL_ARGS]),
            patch.object(sys.stdin, "isatty", return_value=False),
            patch.object(cli, "prepare_checkpoint_run") as checkpoint,
            patch.object(cli, "run_batch_traj_ltlf_pipeline") as run,
            redirect_stderr(io.StringIO()) as error,
        ):
            with self.assertRaises(SystemExit) as caught:
                cli.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("人工审核需要可交互终端", error.getvalue())
        checkpoint.assert_not_called()
        run.assert_not_called()

    def test_review_apply_command_uses_same_custom_run_checkpoint(self):
        with (
            patch.object(sys, "argv", ["main.py", "--run-dir", "data/output/A72", "--apply-no-ambiguity-review", "review.md"]),
            patch.object(cli, "_run_pre_ir_review_command") as apply,
        ):
            cli.main()
        self.assertEqual(apply.call_args.args[0].checkpoint_dir, str(Path("data/output/A72/checkpoints")))

    def test_all_model_reviewer_flags_are_rejected(self):
        for arguments in (
            ["--review", "auto"], ["--experiment-mode", "independent_judge"],
            ["--segmentation-review", "judge"], ["--plain-reading-review", "judge"],
            ["--experiment-mode", "formula_oracle"], ["--delivery-policy", "always_output"],
        ):
            with self.subTest(arguments=arguments), patch.object(sys, "argv", ["main.py", *arguments]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    cli.main()
                self.assertEqual(caught.exception.code, 2)

    def test_reviewer_factory_disabled_even_with_credentials(self):
        with patch.dict(os.environ, {"JUDGE_API_KEY": "fixture", "JUDGE_MODEL": "fixture"}):
            with self.assertRaisesRegex(LLMClientError, "disabled"):
                build_judge_client_from_env()
        with self.assertRaises(ValueError):
            resolve_experiment_mode("auto", "independent_judge")
        with self.assertRaises(ValueError):
            PreIRReviewConfig(segmentation_review="judge")

    def test_config_cannot_reenable_model_review(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "config.json"
            target.write_text(json.dumps({"experiment_mode": "independent_judge"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_runtime_config(target)

    def test_partial_environment_still_loads_missing_dotenv_values(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                Path(".env").write_text("LLM_API_KEY=fixture-key\nLLM_MODEL=fixture-model\n", encoding="utf-8")
                with patch.dict(os.environ, {"LLM_API_BASE_URL": "http://127.0.0.1:9/v1"}, clear=True):
                    client = build_llm_client_from_env("api")
                    self.assertEqual(client.model, "fixture-model")
                    self.assertEqual(client.base_url, "http://127.0.0.1:9/v1")
            finally:
                os.chdir(previous)

    def test_project_entry_can_be_called_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-B", str(ROOT / "main.py"), "--help"], cwd=folder, capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--review {manual}", result.stdout)


class DataReleaseTests(unittest.TestCase):
    def test_dataset_sources_and_all_exported_text_are_unchanged(self):
        self.assertEqual(verify_data(), {"dataset_files": 5, "raw_files": 10, "requirements": 1338})

    def test_exports_do_not_depend_on_offline_references(self):
        from openpyxl import load_workbook

        exports = build_exports()
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / "references_changed.xlsx"
            workbook = load_workbook(ROOT / "data" / "LTLf_Evaluation_Dataset.xlsx")
            for sheet in workbook:
                sheet["C2"] = "reference changed; must never reach input"
                sheet["D2"] = "another offline-only value"
            workbook.save(changed)
            workbook.close()
            self.assertEqual(build_exports(changed), exports)
        self.assertEqual(set(exports), {"A72.xlsx", "B240.xlsx", "C1026.xlsx", "manifest.json"})

    def test_moved_static_knowledge_is_still_selected_by_source_filters(self):
        chunks = load_default_kb()
        self.assertTrue(any(_source_matches(chunk.source, (KB_LTLF_SEMANTICS,)) for chunk in chunks))
        self.assertTrue(all(chunk.source != "builtin" for chunk in chunks))


class InstalledTransportTests(unittest.TestCase):
    @patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
    def test_generation_api_transport_with_local_http_fixture(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                value = {
                    "id": "fixture", "object": "chat.completion", "created": 0,
                    "model": "fixture-generator", "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": '{"ok":true}'},
                    "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                content = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = LangChainJSONClient(
                api_key="fixture-key", model="fixture-generator",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                max_retries=0, timeout_seconds=5,
            )
            result = client.complete_json("Generate fixture JSON", {"task": "fixture"}, response_schema={
                "type": "object", "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"], "additionalProperties": False,
            })
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["model"], "fixture-generator")


class InternalHumanAuthorityTests(unittest.TestCase):
    def test_trace_conflict_cannot_be_resolved_by_the_generation_model(self):
        dispute = TraceEvidenceDispute("D1", "C1", "F1", (("p",),), "label_conflict", (), ("SAT", "UNSAT"), ())
        reconciliation = TraceEvidenceReconciliation((), (), (dispute,))
        client = Mock()
        result = resolve_trace_evidence_disputes(
            semantic=None, brief=None, reconciliation=reconciliation,
            resolver_client=client, verifier_client=client,
        )
        self.assertIs(result.reconciliation, reconciliation)
        self.assertTrue(result.requires_existing_semantic_review)
        self.assertEqual(result.reports[0].status, "unresolved")
        client.complete_json.assert_not_called()

    def test_unsupported_trace_annotation_entry_rejects_model_clients(self):
        client = Mock()
        with self.assertRaisesRegex(ValueError, "disabled"):
            independent_judge_annotations("fixture", None, [], judge_client=client)
        client.complete_json.assert_not_called()

    def test_ir_branch_simple_choice_preserves_human_source(self):
        query = TraceQuery("IRQ-fixture", "boundary", "timing", (), [{"p"}], beginner_view="p is true.", purpose="ir_branch_discrimination", parent_candidate_set_id="fixture-set")
        brief = SimpleNamespace(source_text="Eventually p", brief_id="fixture", ap_definitions=(("p", "p holds"),), reviewed_plain_language_decisions=(), open_plain_language_axes=(), global_trace_semantics=())
        with tempfile.TemporaryDirectory() as folder:
            client = Mock(model="fixture-assistant")
            client.complete_json.return_value = {"annotations": [{
                "query_id": query.query_id, "label": "SAT", "confidence": "medium",
                "rationale": "Model interpretation of the human choice.",
                "temporal_simulation_summary": "p occurs in the finite trace.",
            }]}
            with patch("builtins.input", return_value="y"), redirect_stdout(io.StringIO()):
                result = manual_trace_annotations(requirement="Eventually p", semantic=None, queries=[query], round_index=1, records_dir=folder, adjudication_brief=brief, llm_client=client)
            saved = json.loads(next(Path(folder).glob("nonexpert_trace_review_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(saved["human_choices"][0]["verdict"], "符合")
            self.assertEqual(saved["metadata_source"], "model_assistant")
        self.assertEqual(result[0].label_source, "human")
        self.assertEqual(result[0].label, "SAT")
        self.assertEqual(result[0].purpose, "ir_branch_discrimination")

    def test_ir_branch_selection_uses_human_evidence(self):
        semantic = SemanticStructure(trigger="p", response="p", temporal_keyword="eventually", scope_keyword="", pattern="eventually", candidate_ir_set=CandidateIRSet("REQ-1", "Eventually p", branches=(Mock(ir_branch_id="B1"), Mock(ir_branch_id="B2"))))
        query = TraceQuery("IRQ-fixture", "boundary", "timing", (), [{"p"}], purpose="ir_branch_discrimination", parent_candidate_set_id="fixture-set")
        annotation = TraceAnnotation(query.query_id, "SAT", "high", "Fixture human label.", purpose=query.purpose)
        selected = PolicyBranchSelectionOutcome("unique", semantic, selected_branch_id="B1")
        provider = Mock(return_value=[annotation])
        with (
            patch.object(ir_review, "synthesize_policy_discriminating_queries", return_value=[query]),
            patch.object(ir_review, "build_trace_adjudication_brief", return_value=Mock()),
            patch.object(ir_review, "bind_trace_queries_to_brief", side_effect=lambda queries, brief: queries),
            patch.object(ir_review, "_candidate_scores", return_value=()),
            patch.object(ir_review, "select_policy_candidate_from_annotations", return_value=selected) as choose,
            patch.object(ir_review, "_report", return_value=Mock()),
        ):
            result = ir_review.adjudicate_ir_candidate_branches("Eventually p", semantic, annotation_provider=provider, records_dir="fixture-reviews")
        self.assertEqual(result.status, "unique")
        self.assertEqual(choose.call_args.kwargs["accepted_label_sources"], ("human",))
        self.assertEqual(choose.call_args.kwargs["decision_source"], "human_nl_trace_discrimination")
        self.assertEqual(provider.call_args.kwargs["records_dir"], "fixture-reviews")


if __name__ == "__main__":
    unittest.main()
