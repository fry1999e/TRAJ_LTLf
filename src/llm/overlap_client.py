"""Overlap proposal requests with human input; never precompute approval."""
from concurrent.futures import Future
from copy import deepcopy
import json
from threading import Lock, Thread

from execution_observer import current_observer
from terminal_progress import model_wait


class OverlapJSONClient:
    def __init__(self, client):
        self.client = client
        self._pending = {}
        self._lock = Lock()
        self._graph_started = set()

    def __getattr__(self, name):
        return getattr(self.client, name)

    @staticmethod
    def _key(system_prompt, user_payload, kwargs):
        return json.dumps([system_prompt, user_payload, kwargs], sort_keys=True,
                          ensure_ascii=False, separators=(",", ":"))

    def prefetch(self, system_prompt, user_payload, **kwargs):
        if current_observer() is not None or getattr(self.client, "request_observer", None) is not None:
            return
        key = self._key(system_prompt, user_payload, kwargs)
        with self._lock:
            if key in self._pending:
                return
            future = Future()
            self._pending[key] = future

        def run():
            # A fresh thread has no terminal/input context. Only the foreground
            # displays waiting feedback, and a cancelled review need not join.
            try:
                value = self.client.complete_json(system_prompt, deepcopy(user_payload), **deepcopy(kwargs))
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(value)

        Thread(target=run, name="proposal-prefetch", daemon=True).start()

    def complete_json(self, system_prompt, user_payload, **kwargs):
        key = self._key(system_prompt, user_payload, kwargs)
        with self._lock:
            future = self._pending.pop(key, None)
        if future is not None:
            with model_wait():
                return deepcopy(future.result())
        return self.client.complete_json(system_prompt, user_payload, **kwargs)

    def prefetch_aps(self, source_text):
        from ap_preparation import AP_PROPOSAL_PROMPT, AP_PROPOSAL_SCHEMA, source_symbol_ap_proposals
        if source_symbol_ap_proposals(source_text) is not None:
            return
        self.prefetch(AP_PROPOSAL_PROMPT,
            {"task": "extract_source_ap_candidates", "source_text": source_text},
            temperature=0.0, response_schema=AP_PROPOSAL_SCHEMA)

    def prefetch_graph(self, context, segmentation, reading):
        from constructive_ir_builder import try_construct_proposal_ir
        from ir_proposal_router import SOURCE_GRAPH_PROPOSAL_PROMPT
        from typed_source_relation_graph import source_graph_proposal_payload, source_graph_response_schema
        source = context.grounding_contract.source_text
        if source in self._graph_started or context.has_authoritative_catalog:
            return
        self._graph_started.add(source)
        policies = getattr(self.client, "dataset_semantics_contract", None)
        try:
            route = try_construct_proposal_ir(source, context, policies=policies,
                surface_segmentation=segmentation, plain_reading_review=reading)
        except ValueError:
            return
        if route.route != "G" or route.ir is not None:
            return
        self.prefetch(SOURCE_GRAPH_PROPOSAL_PROMPT,
            source_graph_proposal_payload(source, context, contract=policies, plain_reading_review=reading),
            temperature=0.0, response_schema=source_graph_response_schema(context))


def overlap_client(client):
    from llm.langchain_client import LangChainJSONClient
    if (current_observer() is None and isinstance(client, LangChainJSONClient)
            and client.request_observer is None):
        return OverlapJSONClient(client)
    return client
