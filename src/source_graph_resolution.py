"""Resolve source-graph debts only through complete existing typed candidates."""
from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product

from execution_observer import emit, request_stage
from faithful_ir import FaithfulIRValidationError
from llm.errors import LLMResponseSchemaError
from llm.schema_validation import validate_response_schema
from typed_source_relation_graph import SourceAxisDebtResolution, SourceCoverageEntry


class SourceGraphResolutionRequired(FaithfulIRValidationError):
    """A graph without exhaustive executable candidates remains pending."""

    suggested_rerun_stage = "semantic"

    def __init__(self, source_text, ap_context, result, *, segmentation=None, plain_reading_review=None):
        message = "语义图尚未形成完整候选，已保留待处理，未输出最终公式；无需填写专业语义表单。"
        details = [f"{debt.evidence_text!r}：{debt.detail or '模型未提供具体说明'}"
                   for debt in result.semantic_debts]
        if not details:
            details = ["构建原因：" + (result.reason_code or "没有可执行的完整候选")]
        message += "\n" + "\n".join(details)
        super().__init__(message, code="SOURCE_GRAPH_SEMANTIC_RESOLUTION_REQUIRED",
                         path="/source_intent_graph", repairable=False)
        self.requirement_id = ap_context.requirement_id
        self.source_hash = ap_context.effective_source_hash
        self.diagnostics = {
            "reason_code": self.code, "source_reason_code": result.reason_code,
            "suggested_rerun_stage": self.suggested_rerun_stage, "user_message": message,
            "source_text": source_text, "source_hash": self.source_hash,
            "requirement_id": self.requirement_id,
            "source_graph": asdict(result.graph) if result.graph is not None else None,
            "semantic_debts": [asdict(debt) for debt in result.semantic_debts],
            "candidate_axes": [asdict(axis) for axis in result.axes],
            "candidate_complete": result.candidate_complete,
            "ap_binding_context": asdict(ap_context),
            "surface_segmentation": asdict(segmentation) if segmentation is not None else None,
            "plain_reading_review": asdict(plain_reading_review) if plain_reading_review is not None else None,
        }


def resolve_supported_axis_debts(result, ap_context, client, *, policies=None,
                                surface_segmentation=None, plain_reading_review=None):
    """Let the main model match debt meanings to backend-owned alternatives.

    No model-authored node, parameter, option or approval is accepted. Discharging
    a debt requires a complete rebuilt candidate set containing every selected
    option of every cited axis. Other debts and unrepresentable axes stay pending.
    """
    debts = result.semantic_debts
    if (client is None or result.graph is None or not debts or not result.axes
            or any(debt.reason_code != "semantic_axis" or not debt.detail.strip() for debt in debts)):
        return result
    by_id = {debt.debt_id: debt for debt in debts}
    if len(by_id) != len(debts):
        return result
    allowed = {debt.debt_id: tuple(axis for axis in result.axes
                                 if debt.source_span[0] < axis.source_span[1]
                                 and axis.source_span[0] < debt.source_span[1]) for debt in debts}
    if any(not axes for axes in allowed.values()):
        return result
    schema = {"type": "object", "properties": {"resolutions": {
        "type": "array", "minItems": len(debts), "maxItems": len(debts),
        "items": {"type": "object", "properties": {
            "debt_id": {"type": "string", "enum": list(by_id)},
            "axis_ids": {"type": "array", "uniqueItems": True,
                         "items": {"type": "string", "enum": [axis.axis_id for axis in result.axes]}},
            "reason": {"type": "string", "minLength": 1},
        }, "required": ["debt_id", "axis_ids", "reason"], "additionalProperties": False},
    }}, "required": ["resolutions"], "additionalProperties": False}
    payload = {
        "task": "map_source_semantic_debts_to_candidate_axes",
        "source_text": result.graph.source_text,
        "source_nodes": [asdict(node) for node in result.graph.nodes],
        "debts": [{**asdict(debt), "allowed_axes": [asdict(axis) for axis in allowed[debt.debt_id]]}
                  for debt in debts],
    }
    raw = None
    for _attempt in range(2):
        try:
            with request_stage("source_axis_resolution"):
                raw = client.complete_json(
                    "Match each reported semantic difference to existing backend candidate axes only when "
                    "their supplied options exhaust the entire difference. Use every required axis ID. "
                    "If any part concerns a different scope, participant, order, domain assumption or an "
                    "unsupported meaning, use an empty axis_ids list instead of guessing. Do not change "
                    "nodes, spans, relations, policies or options; do not invent an axis or claim approval. "
                    "This correspondence permits only candidate construction for later human trajectory review.",
                    dict(payload), temperature=0.0, response_schema=schema,
                )
            validate_response_schema(raw, schema)
            items = raw["resolutions"]
            if len({item["debt_id"] for item in items}) != len(debts):
                raise ValueError("Missing or duplicate semantic-debt mapping")
            if any(not item["axis_ids"] for item in items):
                emit("source_axis_debt_mapping_declined", response=raw)
                return result
            for item in items:
                if not set(item["axis_ids"]) <= {axis.axis_id for axis in allowed[item["debt_id"]]}:
                    raise ValueError("Semantic-debt mapping references an unrelated axis")
            break
        except (ValueError, TypeError, LLMResponseSchemaError) as exc:
            payload["validation_error"] = str(exc)
    else:
        emit("source_axis_debt_mapping_invalid", response=raw, error=payload["validation_error"])
        return result
    actual = getattr(client, "client", client)
    model = str(getattr(actual, "model", None) or getattr(actual, "model_name", None) or type(actual).__name__)
    bindings = tuple(SourceAxisDebtResolution(by_id[item["debt_id"]], tuple(item["axis_ids"]),
                                             model, item["reason"]) for item in raw["resolutions"])
    ledger = result.graph.coverage_ledger
    if ledger is not None:
        ledger = replace(ledger, blocking_debts=tuple(debt for debt in ledger.blocking_debts if debt.debt_id not in by_id),
                         entries=(*ledger.entries, *(
            SourceCoverageEntry("SEMANTIC_AXIS", axis.source_span, axis.evidence_text, axis.axis_id)
            for axis in result.axes if any(axis.axis_id in binding.axis_ids for binding in bindings)
        )))
    graph = replace(result.graph, semantic_debts=tuple(debt for debt in result.graph.semantic_debts if debt.debt_id not in by_id),
                    coverage_ledger=ledger,
                    axis_debt_resolutions=(*result.graph.axis_debt_resolutions, *bindings))
    from constructive_ir_builder import try_construct_from_proposed_graph
    rebuilt = try_construct_from_proposed_graph(
        graph, ap_context, policies=policies, surface_segmentation=surface_segmentation,
        plain_reading_review=plain_reading_review,
    )
    if not rebuilt.candidate_complete or not rebuilt.candidates or rebuilt.semantic_debts:
        return result
    axes = {axis.axis_id: axis for axis in rebuilt.axes}
    for binding in bindings:
        for axis_id in binding.axis_ids:
            if axis_id not in axes:
                return result
    axis_ids = tuple(sorted(axes))
    expected = set(product(*(tuple(option.option_id for option in axes[axis_id].options) for axis_id in axis_ids)))
    selections = [dict(candidate.selections) for candidate in rebuilt.candidates]
    if any(set(selected) != set(axis_ids) for selected in selections):
        return result
    actual_choices = {tuple(selected[axis_id] for axis_id in axis_ids) for selected in selections}
    if actual_choices != expected:
        return result
    emit("source_axis_debts_bound_to_complete_candidates", resolutions=[asdict(binding) for binding in bindings],
         candidate_count=len(rebuilt.candidates))
    return rebuilt
