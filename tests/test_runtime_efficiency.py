from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from agents.nlu_agent import NLUAgent
from agents.review_loop import run_semantic_review
from ap_preparation import prepare_reviewed_ap_context
from batch_review import _build_batch_state, _regenerate_batch_item
from llm.langchain_client import LangChainJSONClient
from nonexpert_review import review_semantic_with_model
from rag.retriever import KeywordRetriever
from semantic_review_view import semantic_review_payload, semantic_review_fingerprint
from tests.test_ap_preparation import Client, TEXT, context
from tests.test_candidate_delivery import pending


class TransportReuseTests(unittest.TestCase):
    def client(self):
        return LangChainJSONClient(api_key="fixture-key", model="fixture", response_format_mode="json_schema")

    def test_schema_changes_share_transport_without_sharing_response_format(self):
        client = self.client()
        with patch("langchain_openai.ChatOpenAI") as factory:
            args = {"temperature": 0.0, "max_tokens": 1200, "use_json_response_format": True}
            for key in ("first", "second"):
                client._build_chat_model(**args, response_schema={"type": "object", "properties": {key: {"type": "string"}}})
            unbound = client._build_chat_model(**{**args, "use_json_response_format": False})
            factory.assert_called_once()
            self.assertIs(unbound, factory.return_value)
            bindings = factory.return_value.bind.call_args_list
        self.assertEqual(len(bindings), 2)
        self.assertIn("first", bindings[0].kwargs["response_format"]["json_schema"]["schema"]["properties"])
        self.assertIn("second", bindings[1].kwargs["response_format"]["json_schema"]["schema"]["properties"])

    def test_different_generation_parameters_and_clients_keep_separate_transports(self):
        client = self.client()
        with patch("langchain_openai.ChatOpenAI") as factory:
            base = {"temperature": 0.0, "max_tokens": 1200, "use_json_response_format": False}
            first = client._build_chat_model(**base)
            self.assertIs(first, client._build_chat_model(**base))
            client._build_chat_model(**{**base, "max_tokens": 1800})
            client._build_chat_model(**base, thinking_mode="disabled")
            self.client()._build_chat_model(**base)
        self.assertEqual(factory.call_count, 4)

    def test_retry_reuses_connection_and_still_performs_the_retry(self):
        client = self.client()
        with patch("langchain_openai.ChatOpenAI") as factory, \
             patch("llm.langchain_client._sleep_before_retry") as backoff:
            factory.return_value.invoke.side_effect = [TimeoutError("timed out"), '{"ok":true}']
            reply = client._invoke_chat_model([], temperature=0.0, max_tokens=1200, use_json_response_format=False)
            factory.assert_called_once()
            self.assertEqual(factory.return_value.invoke.call_count, 2)
            backoff.assert_called_once()
        self.assertEqual(json.loads(reply), {"ok": True})

    def test_direct_compatibility_path_reuses_connection(self):
        client = self.client()
        with patch("openai.OpenAI") as factory:
            factory.return_value.chat.completions.create.return_value = {"choices": [{"message": {"content": "{}"}}]}
            for _ in range(2):
                self.assertEqual(client._invoke_openai_chat_completion([], temperature=0.0,
                    max_tokens=1200, use_json_response_format=False), "{}")
            factory.assert_called_once()
            self.assertEqual(factory.return_value.chat.completions.create.call_count, 2)


class SemanticEfficiencyTests(unittest.TestCase):
    def native(self):
        return NLUAgent(KeywordRetriever([]), llm_client=Client()).generate(TEXT)

    def test_native_view_preserves_all_branches_and_omits_legacy_summaries_and_audit(self):
        semantic = self.native()
        semantic = replace(semantic, trigger="misleading_old_trigger", response="misleading_old_response")
        before = asdict(semantic)
        view = semantic_review_payload(TEXT, semantic)["semantic"]
        self.assertNotIn("response", view)
        self.assertNotIn("trigger", view)
        self.assertNotIn("rag_context", view)
        branches = view["candidate_ir_set"]["branches"]
        self.assertEqual(len(branches), len(semantic.candidate_ir_set.branches))
        for original, branch in zip(semantic.candidate_ir_set.branches, branches):
            graph = view["graphs"][branch["graph"]]
            self.assertEqual(graph["root_node_id"], original.faithful_ir.root_node_id)
            for expected, actual in zip(original.faithful_ir.nodes, graph["nodes"]):
                self.assertEqual(actual, {k: v for k, v in asdict(expected).items()
                                           if k not in {"confidence", "supporting_segment_ids"}})
        self.assertLess(len(json.dumps(view)), len(json.dumps(before)) // 2)
        self.assertEqual(asdict(semantic), before)

    def test_check_follows_conjunction_graph_instead_of_response_summary(self):
        source = "Globally it is the case that if a holds then eventually a and b hold."
        a1, a2, b = source.index("a holds"), source.index("a and"), source.index("b hold")
        nodes = [
            {"ref": "a1", "kind": "AP", "source_span": [a1, a1+1], "children": []},
            {"ref": "a2", "kind": "AP", "source_span": [a2, a2+1], "children": []},
            {"ref": "b", "kind": "AP", "source_span": [b, b+1], "children": []},
            {"ref": "and", "kind": "conjunction", "source_span": [a2, len(source)-1], "children": ["a2", "b"]},
            {"ref": "f", "kind": "eventually", "source_span": [source.index("eventually"), len(source)-1], "children": ["and"]},
            {"ref": "i", "kind": "implication", "source_span": [source.index("if"), len(source)-1], "children": ["a1", "f"]},
            {"ref": "g", "kind": "global_constraint", "source_span": [0, len(source)-1], "children": ["i"]},
        ]
        generator = Client(ap_replies=[{"aps": [{"canonical_name": name, "evidence_texts": [name]} for name in ("a", "b")]}],
            graph_replies=[{"source_intent_graph": {"nodes": nodes, "root_ref": "g", "semantic_debts": []}}])
        semantic = NLUAgent(KeywordRetriever([]), llm_client=generator).generate(source)
        checker = Mock()
        checker.complete_json.return_value = {"approved": True, "reason": "Both APs occur beneath the eventual conjunction."}
        review_semantic_with_model(source, replace(semantic, response="a"), checker)
        payload = checker.complete_json.call_args.args[1]
        view = payload["semantic"]
        graph = view["graphs"][view["primary_graph"]]
        self.assertEqual({ap["name"] for ap in graph["atomic_propositions"]}, {"a", "b"})
        self.assertIn("conjunction", {node["node_type"] for node in graph["nodes"]})
        self.assertNotIn("response", view)
        self.assertNotIn("formula_text", json.dumps(payload))

    def test_repeated_native_meaning_stops_before_second_check_and_third_generation(self):
        original = self.native()
        client = Mock()
        client.complete_json.return_value = {"approved": False, "reason": "Recheck the eventuality."}
        agent = SimpleNamespace(generate=Mock(return_value=original), llm_enabled=True,
                                llm_client=client, retriever=KeywordRetriever([]))
        with tempfile.TemporaryDirectory() as folder:
            semantic, records = run_semantic_review(TEXT, agent, records_dir=folder)
            saved = json.loads(Path(records[-1]["record_path"]).read_text(encoding="utf-8"))
        self.assertEqual(agent.generate.call_count, 2)
        client.complete_json.assert_called_once()
        self.assertEqual(saved["reason_code"], "SEMANTIC_REVISION_NO_PROGRESS")
        self.assertFalse(saved["approved"])
        result = pending(semantic)
        self.assertIsNotNone(result.provisional_candidate)
        self.assertEqual(result.approval_status, "not_approved")
        self.assertFalse(semantic.persistent_rag_allowed)

    def test_changed_branch_meaning_or_assumption_is_not_treated_as_a_repeat(self):
        original = self.native()
        key = semantic_review_fingerprint(original)
        branches = original.candidate_ir_set.branches
        changed = replace(original, candidate_ir_set=replace(original.candidate_ir_set, branches=branches[:1]))
        self.assertNotEqual(key, semantic_review_fingerprint(changed))
        assumption = replace(branches[0], semantic_assumptions=("different source assumption",))
        changed = replace(original, candidate_ir_set=replace(original.candidate_ir_set, branches=(assumption, *branches[1:])))
        self.assertNotEqual(key, semantic_review_fingerprint(changed))
        self.assertEqual(key, semantic_review_fingerprint(replace(original, response="old summary")))

    def test_changed_candidate_set_still_receives_a_new_check(self):
        original = self.native()
        changed = replace(original, candidate_ir_set=replace(original.candidate_ir_set,
                          branches=original.candidate_ir_set.branches[:1]))
        client = Mock()
        client.complete_json.side_effect = [
            {"approved": False, "reason": "Check the competing reading."},
            {"approved": True, "reason": "The remaining reading is source-aligned."},
        ]
        agent = SimpleNamespace(generate=Mock(side_effect=[original, changed]), llm_enabled=True,
                                llm_client=client, retriever=KeywordRetriever([]))
        with tempfile.TemporaryDirectory() as folder:
            _semantic, records = run_semantic_review(TEXT, agent, max_rounds=2, records_dir=folder)
        self.assertEqual(client.complete_json.call_count, 2)
        self.assertFalse(any(record.get("reason_code") == "SEMANTIC_REVISION_NO_PROGRESS" for record in records))

    def test_renaming_native_node_ids_does_not_hide_a_repeated_meaning(self):
        original = self.native()
        ir = original.faithful_ir
        names = {node.node_id: "renamed-" + node.node_id for node in ir.nodes}
        renamed = replace(ir, root_node_id=names[ir.root_node_id], nodes=tuple(
            replace(node, node_id=names[node.node_id], children=tuple(names[child] for child in node.children))
            for node in ir.nodes))
        self.assertEqual(semantic_review_fingerprint(original),
                         semantic_review_fingerprint(replace(original, faithful_ir=renamed)))

    def test_batch_repeated_meaning_uses_the_same_stop_policy(self):
        generator = Client()
        with redirect_stdout(io.StringIO()):
            ap = prepare_reviewed_ap_context(context(), generator, interactive=True, input_fn=lambda _: "a").context
        original = NLUAgent(KeywordRetriever([]), llm_client=generator).generate(TEXT, ap_binding_context=ap)
        state = _build_batch_state(index=1, requirement=TEXT, llm_client=generator, ap_binding_context=ap)
        state.nlu_agent.generate = Mock(return_value=original)

        def stop_after_candidates(item, **_kwargs):
            item.semantic_policy_pending = {"status": "provisional_fallback"}

        with tempfile.TemporaryDirectory() as folder, \
             patch("nonexpert_review.review_semantic_with_model", return_value={"approved": False, "reason": "Check eventuality."}) as check, \
             patch("batch_review._adjudicate_ir_candidate_semantic", side_effect=stop_after_candidates), \
             patch("builtins.input", side_effect=AssertionError("unexpected human form")):
            _regenerate_batch_item(state, checkpoint_dir=None, round_index=1, item_index=1,
                total_items=1, review_mode="manual", records_dir=folder, allow_degraded=False)
        check.assert_called_once()
        self.assertEqual(state.nlu_agent.generate.call_count, 2)
        self.assertTrue(any(record.get("reason_code") == "SEMANTIC_REVISION_NO_PROGRESS"
                            for record in state.review_records))
        self.assertFalse(state.semantic_approved)


if __name__ == "__main__":
    unittest.main()
