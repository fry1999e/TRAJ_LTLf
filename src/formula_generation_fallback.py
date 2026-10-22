"""Generate an unapproved formula when source-graph generation cannot finish."""
from __future__ import annotations

from dataclasses import asdict, replace

from execution_observer import request_stage
from faithful_ir import (
    FaithfulIRValidationError, compute_ir_version, faithful_ir_from_mapping,
    semantic_summary_from_ir, validate_faithful_ir,
)
from ir_compiler import compile_ir_to_formula
from llm.errors import LLMResponseSchemaError
from llm.schema_validation import validate_response_schema
from ltlf_parser import parse_formula
from ltlf_syntax import validate_supported_ltlf_formula_text
from schemas import CandidateIRBranch, CandidateIRSet


EXTRACTOR = "main_model_formula_fallback_v1"
RESPONSE_SCHEMA = {
    "type": "object", "properties": {
        "formula": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 1},
    }, "required": ["formula", "reason"], "additionalProperties": False,
}


def is_formula_generation_fallback(semantic):
    ir = getattr(semantic, "faithful_ir", None)
    return ir is not None and ir.provenance.extractor == EXTRACTOR


def generate_formula_fallback(requirement, client, ap_context, error, *,
                              surface_segmentation=None, plain_reading_review=None):
    """Use the main generator, not a model reviewer or an oracle/reference answer."""
    ap_context.validate_source(requirement)
    if not ap_context.has_authoritative_catalog:
        raise error
    identities = ap_context.grounding_contract.identities
    request = {
        "task": "generate_bound_formula_fallback", "source_text": requirement,
        "atomic_propositions": [
            {"name": identity.canonical_name,
             "source_quotes": [mention.evidence_text for mention in identity.mentions]}
            for identity in identities
        ],
        "plain_reading_review": asdict(plain_reading_review) if plain_reading_review is not None else None,
        "graph_generation_error": {"code": getattr(error, "code", ""), "message": str(error)},
        "supported_operators": ["!", "&", "|", "->", "X", "F", "G", "U"],
    }
    failures = []
    for attempt in range(1, 4):
        raw = None
        try:
            with request_stage("formula_generation_fallback"):
                raw = client.complete_json(
                    "Generate one best LTLf translation of the original requirement using only "
                    "the supplied, human-confirmed AP names and meanings. Respect negation, "
                    "Boolean scope, repeated guards and the selected ordinary-language readings. "
                    "The source graph failed; you are the main formula generator, not a reviewer. "
                    "Return {formula,reason}. Do not return a candidate list, approval, new APs, "
                    "placeholder constants or reference answers. State any interpretation choice "
                    "in reason. A generated formula does not establish semantic approval.",
                    request, temperature=0.0, response_schema=RESPONSE_SCHEMA,
                )
            validate_response_schema(raw, RESPONSE_SCHEMA)
            validate_supported_ltlf_formula_text(raw["formula"])
            formula = parse_formula(raw["formula"])
            ir = _candidate_ir(formula, requirement, ap_context, surface_segmentation, plain_reading_review)
            compiled = compile_ir_to_formula(ir)
            record = {
                "stage": "formula_generation_fallback", "status": "candidate_generated",
                "source_error_code": getattr(error, "code", ""), "source_error": str(error),
                "source_diagnostic": getattr(error, "diagnostics", {}),
                "formula": str(compiled), "interpretation_reason": raw["reason"],
                "model": str(getattr(client, "model", type(client).__name__)),
                "attempts": attempt, "rejected_proposals": failures,
                "ap_context_fingerprint": ap_context.context_fingerprint,
                "approval_granted": False, "source_fidelity_proven": False,
            }
            # IR serialization remains formula-free. Complete proposals and
            # errors live in the separate generation audit returned below.
            history = {key: record[key] for key in (
                "stage", "status", "source_error_code", "source_error", "interpretation_reason",
                "model", "attempts", "ap_context_fingerprint", "approval_granted", "source_fidelity_proven",
            )}
            ir = replace(ir, repair_history=(*ir.repair_history, history), ir_version="")
            ir = replace(ir, ir_version=compute_ir_version(ir))
            semantic = semantic_summary_from_ir(ir, rag_context=())
            branch = CandidateIRBranch(ir_branch_id="generated-formula-fallback", faithful_ir=ir,
                semantic_assumptions=(raw["reason"], "Source graph generation failed; formula interpretation is unapproved."),
                generation_reason="Main-model formula generation after graph failure.")
            return replace(semantic, surface_segmentation=surface_segmentation,
                plain_reading_review=plain_reading_review, semantic_status="needs_semantic_review",
                persistent_rag_allowed=False, candidate_ir_set=CandidateIRSet(
                    requirement_id=ir.requirement_id, nl_text=requirement, branches=(branch,),
                    status="policy_candidates_ready", segmentation_id=ir.segmentation_id,
                    reading_review_id=ir.reading_review_id, max_branch_count=1,
                    candidate_catalog_version="generated-formula-fallback-v1")), record
        except (ValueError, TypeError, LLMResponseSchemaError) as exc:
            failures.append({"attempt": attempt, "validation_error": str(exc), "proposal": raw})
            request = {**request, "validation_error": str(exc), "rejected_proposal": raw}
    failure = FaithfulIRValidationError(
        "公式候选自动补生成未通过校验：" + failures[-1]["validation_error"],
        code="FORMULA_FALLBACK_GENERATION_FAILED", repairable=False,
    )
    failure.diagnostics = {"source_error": str(error), "failed_attempts": failures}
    raise failure from error


def _candidate_ir(formula, source, ap_context, segmentation, reading):
    """Backend AST lowering preserves AP identities, without claiming NL fidelity."""
    identities = {identity.canonical_name: identity for identity in ap_context.grounding_contract.identities}
    nodes, used = [], set()
    kinds = {"not": "negation", "and": "conjunction", "or": "disjunction", "implies": "implication",
             "G": "global_constraint", "F": "eventually", "X": "next_step", "U": "until",
             "true": "boolean_constant", "false": "boolean_constant"}
    def segments(span):
        return [part.segment_id for part in segmentation.segments
                if part.start < span[1] and span[0] < part.end] if segmentation is not None else []
    def build(current):
        children = [build(child) for child in current.args]
        node_id = f"fallback-n{len(nodes) + 1}"
        span, params = (0, len(source)), {}
        if current.op == "ap":
            if current.name not in identities:
                raise ValueError(f"公式引用未审核的 AP：{current.name}")
            used.add(current.name)
            span = identities[current.name].mentions[0].source_span
            kind, params = "ap_condition", {"ap_name": current.name}
        else:
            kind = kinds.get(current.op)
            if kind is None:
                raise ValueError(f"公式包含不支持的算子：{current.op}")
            if current.op == "X":
                params = {"steps": 1}
            elif current.op == "U":
                params = {"witness_policy": "required", "witness_timing": "current_or_future",
                          "left_at_witness": "not_required", "scope_origin": "current_position"}
            elif current.op in {"true", "false"}:
                params = {"value": current.op == "true"}
        nodes.append({"node_id": node_id, "node_type": kind, "children": children, "parameters": params,
            "scope": "all_positions" if current.op == "G" else "current_position",
            "source_span": span, "evidence_text": source[slice(*span)], "supporting_segment_ids": segments(span)})
        return node_id
    root = build(formula)
    if identities and not used:
        raise ValueError("公式未引用任何已审核 AP，不能用常量代替需求。")
    aps = [{"ap_id": f"fallback-ap{index}", "name": name,
            "source_span": identity.mentions[0].source_span, "evidence_text": identity.mentions[0].evidence_text,
            "supporting_segment_ids": segments(identity.mentions[0].source_span)}
           for index, (name, identity) in enumerate(identities.items(), 1) if name in used]
    ir = faithful_ir_from_mapping({"nodes": nodes, "atomic_propositions": aps, "root_node_id": root},
        source_text=source, requirement_id=ap_context.requirement_id, extractor=EXTRACTOR,
        ap_binding_context=ap_context, working_interpretation=True)
    # These are provenance links, not a claim that an unresolved reading was
    # adjudicated. The full parent reading remains attached to the semantic.
    ir = replace(ir, segmentation_id=segmentation.segmentation_id if segmentation is not None else "",
        reading_review_id=reading.reading_review_id if reading is not None else "",
        reading_review_basis=reading.review_basis if reading is not None else "", ir_version="")
    ir = replace(ir, ir_version=compute_ir_version(ir))
    validate_faithful_ir(ir, surface_segmentation=segmentation, plain_reading_review=reading)
    return ir
