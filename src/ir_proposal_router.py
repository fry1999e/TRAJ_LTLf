"""One source-bound entry point for constructive IR proposals.

The route is deliberately separate from approval.  A proposal AP inventory
can produce a useful candidate, but it cannot confer catalog authority.
"""
from __future__ import annotations
from copy import deepcopy

from ap_preparation import APPreparationError, ensure_ap_candidates

from constructive_ir_builder import (
    try_construct_from_proposed_graph,
    try_construct_ir,
    try_construct_proposal_ir,
)
from execution_observer import emit
from typed_source_relation_graph import (
    UnsupportedConstruct,
    source_graph_from_proposal,
    source_graph_proposal_payload,
    source_graph_response_schema,
)


SOURCE_GRAPH_PROPOSAL_PROMPT = (
    "Propose only a formula-free SourceIntentGraph. Use exact source "
    "spans for relations; reference APs by the supplied canonical "
    "ap_name and let the backend fill AP spans. Do not output LTL/LTLf, a "
    "formula, or a complete FaithfulSemanticIR. Mark unsupported "
    "source semantics as semantic_debts instead of dropping text."
)


class IRProposalRouter:
    def propose(self, source_text, ap_context, *, policies=None,
                surface_segmentation=None, plain_reading_review=None,
                proposal_client=None, feedback_history=None):
        ap_context.validate_source(source_text)
        ap_context = ensure_ap_candidates(
            ap_context, proposal_client, segmentation=surface_segmentation,
        ).context
        builder = (
            try_construct_ir if ap_context.has_authoritative_catalog
            else try_construct_proposal_ir
        )
        result = builder(
            source_text, ap_context, policies=policies,
            surface_segmentation=surface_segmentation,
            plain_reading_review=plain_reading_review,
        )
        # Grammar failures and review revisions use the same formula-free
        # SourceIntentGraph schema. Neither requests a complete model-written
        # FaithfulSemanticIR; backend construction remains authoritative.
        if proposal_client is not None and (
            (result.route == "G" and result.ir is None) or feedback_history
        ):
            try:
                request = source_graph_proposal_payload(
                    source_text,
                    ap_context,
                    contract=policies,
                    plain_reading_review=plain_reading_review,
                )
                if feedback_history:
                    request["feedback_history"] = list(feedback_history)
                    request["revision_instruction"] = (
                        "Reconsider the source graph using the review feedback. "
                        "Feedback cannot override the original source, reviewed AP "
                        "identities or reading constraints. Preserve unsupported "
                        "interpretations as semantic_debts; do not hand-write IR "
                        "node types, timing scopes, formulas or approvals."
                    )
                response_schema = source_graph_response_schema(ap_context)
                raw = proposal_client.complete_json(
                    SOURCE_GRAPH_PROPOSAL_PROMPT,
                    request,
                    temperature=0.0,
                    response_schema=response_schema,
                )
                try:
                    graph = source_graph_from_proposal(
                        raw,
                        source_text,
                        ap_context,
                        contract_hash=(policies or {}).get("contract_hash", ""),
                    )
                except UnsupportedConstruct as first_error:
                    ap_span_error = first_error.code == "AP_ALIAS_UNRESOLVED"
                    repairable = {
                        "AP_ALIAS_UNRESOLVED",
                        "SOURCE_GRAPH_PROPOSAL_INVALID",
                        "SOURCE_GRAPH_PROPOSAL_CHILD_REF_INVALID",
                        "SOURCE_GRAPH_PROPOSAL_ROOT_INVALID",
                        "SOURCE_GRAPH_PROPOSAL_NODE_COLLISION",
                        "SOURCE_GRAPH_PROPOSAL_ARITY_INVALID",
                        "SOURCE_GRAPH_PROPOSAL_CYCLE",
                        "SOURCE_GRAPH_PROPOSAL_DISCONNECTED",
                    }
                    if first_error.code not in repairable:
                        raise
                    repair_prompt = (
                        "Bind each AP to the same predicate using its supplied canonical "
                        "ap_name; the backend will fill its source_span. Never invent an AP or change "
                        "any other field, node, child, Boolean/temporal relation or policy."
                        if ap_span_error else (
                            "Repair only the representation of this formula-free SourceIntentGraph. "
                            "Preserve every AP span, relation kind, child order, Boolean choice, "
                            "scope, and semantic debt. You may only flatten nodes, repair local "
                            "references/root_ref, and satisfy the supplied arity/schema."
                        )
                    )
                    repaired = proposal_client.complete_json(
                        repair_prompt,
                        {
                            **request,
                            "task": ("repair_source_ap_spans" if ap_span_error
                                     else "repair_source_intent_graph_representation"),
                            "rejected_proposal": raw,
                            "backend_error": {
                                "reason_code": first_error.code,
                                "source_span": list(first_error.span),
                            },
                        },
                        temperature=0.0,
                        response_schema=response_schema,
                    )
                    if ap_span_error and not _only_ap_spans_changed(raw, repaired):
                        raise APPreparationError(
                            "AP_GRAPH_BINDING_REVIEW_REQUIRED",
                            "语义图修复改变了 AP 来源范围以外的内容，请重新审核 AP 与语义解释。",
                        )
                    graph = source_graph_from_proposal(
                        repaired,
                        source_text,
                        ap_context,
                        contract_hash=(policies or {}).get("contract_hash", ""),
                    )
                result = try_construct_from_proposed_graph(
                    graph,
                    ap_context,
                    policies=policies,
                    surface_segmentation=surface_segmentation,
                    plain_reading_review=plain_reading_review,
                )
            except APPreparationError:
                raise
            except (UnsupportedConstruct, ValueError, TypeError) as exc:
                if getattr(exc, "code", "") == "AP_ALIAS_UNRESOLVED":
                    raise APPreparationError(
                        "AP_GRAPH_BINDING_REVIEW_REQUIRED",
                        "语义图中的 AP 在一次修复后仍无法对应已绑定原文。请检查 AP 定义或重新生成；未继续编译。",
                        diagnostics={"source_span": list(exc.span)},
                    ) from None
                result = type(result)(
                    "unsupported",
                    graph=getattr(result, "graph", None),
                    reason_code=getattr(exc, "code", "SOURCE_GRAPH_PROPOSAL_INVALID"),
                    semantic_debts=getattr(result, "semantic_debts", ()),
                    candidate_complete=False,
                )
        if result.semantic_debts:
            from source_graph_resolution import resolve_supported_axis_debts
            result = resolve_supported_axis_debts(
                result, ap_context, proposal_client, policies=policies,
                surface_segmentation=surface_segmentation, plain_reading_review=plain_reading_review,
            )
        binding = {
            "requirement_id": ap_context.requirement_id,
            "ap_context_fingerprint": ap_context.context_fingerprint,
            "ap_grounding_authority": ap_context.grounding_authority,
            "entrypoint": "IRProposalRouter.propose",
            "route": result.route,
            "reason_code": result.reason_code,
        }
        emit("constructive_route_attempted", **binding)
        if result.ir is not None:
            emit(
                "proposal_ir_constructed", **binding,
                ir_version=result.ir.ir_version,
                ir_artifact_hash=result.ir.artifact_hash,
                approved=result.ir.provenance.approved,
            )
        if result.candidates:
            emit(
                "source_graph_candidates_constructed", **binding,
                candidate_count=len(result.candidates),
                candidate_complete=result.candidate_complete,
                source_graph_hash=(result.graph.graph_hash if result.graph is not None else ""),
                axis_ids=[axis.axis_id for axis in result.axes],
            )
        return result


def _only_ap_spans_changed(original, repaired):
    """AP binding repair cannot silently rewrite a reviewed interpretation."""
    def without_ap_spans(value):
        if not isinstance(value, dict):
            return None
        value = deepcopy(value)
        graph = value.get("source_intent_graph", value)
        if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
            return None
        for node in graph["nodes"]:
            if not isinstance(node, dict):
                return None
            if node.get("kind") == "AP":
                node.pop("source_span", None)
                node.pop("ap_name", None)
        return value
    old = without_ap_spans(original)
    new = without_ap_spans(repaired)
    return old is not None and new is not None and old == new
