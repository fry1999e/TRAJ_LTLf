"""Compact semantic graphs for model assistance, without audit replicas."""
from dataclasses import asdict

from semantic_progress import semantic_fingerprint
from typed_source_relation_graph import digest


def _graph(ir):
    return {
        "schema_version": ir.schema_version,
        "periodicity_profile_id": ir.periodicity_profile_id,
        "periodicity_profile_version": ir.periodicity_profile_version,
        "root_node_id": ir.root_node_id,
        "atomic_propositions": [asdict(ap) for ap in ir.atomic_propositions],
        "nodes": [{key: value for key, value in asdict(node).items()
                   if key not in {"confidence", "supporting_segment_ids"}}
                  for node in ir.nodes],
        "ambiguities": [asdict(axis) for axis in ir.ambiguities],
    }


def _reading(review):
    if review is None:
        return None
    return {
        "overall_status": review.overall_status,
        "review_basis": review.review_basis,
        "ambiguities": [{key: value for key, value in asdict(axis).items()
                         if key in {"ambiguity_id", "question", "options", "source_span",
                                    "evidence_text", "decision", "decision_status", "comments"}}
                        for axis in review.ambiguities],
    }


def semantic_review_payload(requirement, semantic):
    """Retain every graph/choice, omit misleading legacy trigger/response summaries."""
    view = {"trace_domain": semantic.trace_domain, "semantic_status": semantic.semantic_status}
    if semantic.faithful_ir is None:
        view.update({key: getattr(semantic, key) for key in (
            "trigger", "response", "temporal_keyword", "scope_keyword", "pattern", "aps", "components")})
    else:
        graphs = {}

        def register(ir):
            graph = _graph(ir)
            key = digest(graph)
            graphs.setdefault(key, graph)
            return key

        view["representation"] = "faithful_ir_review_v1"
        view["primary_graph"] = register(semantic.faithful_ir)
        candidates = semantic.candidate_ir_set
        if candidates is not None:
            view["candidate_ir_set"] = {
                "status": candidates.status,
                "selection_status": candidates.selection_status,
                "selected_branch_id": candidates.selected_branch_id,
                "ambiguity_debt": candidates.ambiguity_debt,
                "ambiguity_axes": asdict(candidates.ambiguity_axes) if candidates.ambiguity_axes else None,
                "branches": [{
                    "ir_branch_id": branch.ir_branch_id,
                    "graph": register(branch.faithful_ir),
                    "supported_text_spans": branch.supported_text_spans,
                    "ambiguity_axis_ids": branch.ambiguity_axis_ids,
                    "semantic_assumptions": branch.semantic_assumptions,
                    "plain_language_paraphrase": branch.plain_language_paraphrase,
                } for branch in candidates.branches],
            }
        view["graphs"] = graphs
        view["semantic_resolution"] = asdict(semantic.semantic_resolution) if semantic.semantic_resolution else None
    view["plain_reading_review"] = _reading(semantic.plain_reading_review)
    return {"task": "model_semantic_structure_check", "source_text": requirement, "semantic": view}


def semantic_review_fingerprint(semantic):
    """Detect unchanged native meanings before paying for the same check again."""
    ir = semantic.faithful_ir
    if ir is None:
        return None
    candidates = semantic.candidate_ir_set
    return digest({
        "source": ir.source_hash, "ap_context": ir.ap_context_fingerprint,
        "primary": semantic_fingerprint(ir),
        "branches": sorted(digest({"graph": semantic_fingerprint(branch.faithful_ir),
                                   "assumptions": branch.semantic_assumptions})
                           for branch in candidates.branches) if candidates else [],
        "ambiguity_debt": candidates.ambiguity_debt if candidates else (),
        "selection_status": candidates.selection_status if candidates else "",
        "selected_branch_id": candidates.selected_branch_id if candidates else "",
        "reading": _reading(semantic.plain_reading_review),
        "trace_domain": semantic.trace_domain,
        "semantic_resolution": asdict(semantic.semantic_resolution) if semantic.semantic_resolution else None,
    })
