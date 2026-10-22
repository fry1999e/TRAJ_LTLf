"""Deliver one valid candidate while retaining its original approval status."""
from __future__ import annotations

from dataclasses import replace
from itertools import islice, product

from faithful_ir import compute_ir_version, validate_faithful_ir
from temporal_resolution import allowed_operators_for_node, allowed_semantic_readings, resolved_node_kind


def recover_generation_candidate(requirement, semantic, error, *, client=None,
                                 surface_segmentation=None, plain_reading_review=None,
                                 ap_binding_context=None):
    """Reuse a legal candidate, or generate one from the reviewed source/APs."""
    from faithful_ir import FaithfulIRValidationError
    eligible = isinstance(error, FaithfulIRValidationError) or getattr(error, "code", "") == "AP_GRAPH_BINDING_REVIEW_REQUIRED"
    if not eligible:
        raise error
    if getattr(semantic, "faithful_ir", None) is not None:
        return retain_generation_candidate(requirement, semantic, error,
            surface_segmentation=surface_segmentation, plain_reading_review=plain_reading_review,
            ap_binding_context=ap_binding_context)
    if (client is None or ap_binding_context is None
            or not ap_binding_context.has_authoritative_catalog):
        raise error
    from formula_generation_fallback import generate_formula_fallback
    return generate_formula_fallback(requirement, client, ap_binding_context, error,
        surface_segmentation=surface_segmentation, plain_reading_review=plain_reading_review)


def retain_generation_candidate(requirement, semantic, error, *, surface_segmentation=None,
                                plain_reading_review=None, ap_binding_context=None):
    """Keep a current validated candidate when its proposed revision is invalid."""
    ir = getattr(semantic, "faithful_ir", None)
    if ir is None or ir.source_text != requirement:
        raise error
    if ap_binding_context is not None:
        ap_binding_context.validate_source(requirement)
    sources = [ir]
    if semantic.candidate_ir_set is not None:
        sources.extend(branch.faithful_ir for branch in semantic.candidate_ir_set.branches)
    for candidate in sources:
        if candidate.source_text != requirement or (
            ap_binding_context is not None
            and candidate.ap_context_fingerprint != ap_binding_context.context_fingerprint
        ):
            raise error
        validate_faithful_ir(candidate, surface_segmentation=surface_segmentation,
                            plain_reading_review=plain_reading_review)
    record = {
        "stage": "semantic_revision", "status": "invalid_revision_rolled_back",
        "retained_ir_version": ir.ir_version, "error_type": type(error).__name__,
        "reason_code": getattr(error, "code", "IR_GENERATION_INVALID"), "error": str(error),
        "generation_diagnostic": getattr(error, "diagnostics", {}),
        "approval_granted": False,
    }
    return replace(semantic, semantic_status="needs_semantic_review"), record


def materialize_temporal_assignment(ir, assignment):
    """Close only declared operator choices; retain source and AP lineage."""
    nodes, redirects = [], {}
    for node in ir.nodes:
        if node.node_id not in assignment:
            nodes.append(node)
            continue
        kind = resolved_node_kind(node, assignment)
        if kind == "identity":
            if len(node.children) != 1:
                raise ValueError("identity temporal candidate must be unary")
            redirects[node.node_id] = node.children[0]
            continue
        from temporal_resolution import TEMPORAL_READING_TO_OPERATOR
        reading = next(reading for reading in allowed_semantic_readings(node)
                       if TEMPORAL_READING_TO_OPERATOR[reading] == assignment[node.node_id])
        parameters = dict(node.parameters)
        if kind == "next_step":
            parameters.setdefault("steps", 1)
        nodes.append(replace(node, node_type=kind, temporal_resolution="fixed",
                             allowed_semantic_readings=(reading,), parameters=tuple(parameters.items())))
    def target(node_id):
        while node_id in redirects:
            node_id = redirects[node_id]
        return node_id
    result = replace(ir, nodes=tuple(replace(node, children=tuple(target(child) for child in node.children)) for node in nodes),
                     root_node_id=target(ir.root_node_id), ir_version="",
                     provenance=replace(ir.provenance, approved=False, approval_source=""))
    result = replace(result, ir_version=compute_ir_version(result))
    validate_faithful_ir(result)
    return result


def temporal_assignments(ir):
    dimensions = [(node.node_id, allowed_operators_for_node(node)) for node in ir.nodes
                  if len(allowed_operators_for_node(node)) > 1]
    for options in product(*(domain for _node_id, domain in dimensions)):
        yield dict(zip((node_id for node_id, _domain in dimensions), options))


def _raw_fallback_traces(result, existing):
    """Use source-bound human votes only for fallback ranking, never approval."""
    from contracts.records import digest
    from data_examples import LabeledTrace
    from human_review_session import CHOICES
    from trace_formatter import trace_to_serializable
    ir = result.semantic.faithful_ir
    labels = {digest(trace_to_serializable(trace.trace)): trace.adjudicated_label for trace in existing}
    raw_traces = []
    for record in result.review_records:
        budget = record.get("human_review_budget") or record.get("review_budget") or {}
        evidence = budget.get("fallback_evidence")
        if not evidence or ir is None:
            continue
        context = evidence["context"]
        if (context.get("source_text") != result.requirement
                or context.get("requirement_id") != ir.requirement_id
                or context.get("ap_context_fingerprint") != ir.ap_context_fingerprint):
            raise ValueError("Fallback human votes have stale source/AP context")
        from semantic_adjudication import build_trace_adjudication_brief
        if context.get("ap_definitions") != dict(build_trace_adjudication_brief(result.requirement, result.semantic).ap_definitions):
            raise ValueError("Fallback human votes have stale AP meanings")
        for key, entry in evidence["answers"].items():
            if digest(entry["trace"]) != key or CHOICES.get(entry["answer"]) != (entry["verdict"], entry["locked_label"]):
                raise ValueError("Fallback human vote is inconsistent")
            label = entry["locked_label"]
            if label not in {"SAT", "UNSAT"}:
                continue
            if key in labels:
                if labels[key] != label:
                    raise ValueError("Conflicting human votes for the same fallback trace")
                continue
            labels[key] = label
            raw_traces.append(LabeledTrace(
                name="human-raw-" + key[:16], kind="positive" if label == "SAT" else "negative",
                trace=[set(step) for step in entry["trace"]], expected=label == "SAT",
                reason="已保存的人工选择；模型说明尚未生成，仅用于候选回退排序。",
                adjudicated_label=label, label_source="human", annotation_locked=True,
                confidence="", allowed_rag_stages=(), generation_source="human_raw_fallback_only",
            ))
    return raw_traces


def deliver_formula_candidate(result):
    """Choose one compiled/scored fallback if strict selection has no output."""
    if result.selected is not None or result.provisional_candidate is not None:
        return result
    from agents.consistency_agent import TrajectoryConsistencyAgent
    from ir_compiler import compile_ir_to_formula
    from rag.retriever import KeywordRetriever
    evaluator = TrajectoryConsistencyAgent(KeywordRetriever([]))
    traces = [trace for trace in result.traces if trace.label_source == "human"
              and trace.adjudicated_label in {"SAT", "UNSAT"}]
    raw_traces = _raw_fallback_traces(result, traces)
    traces.extend(raw_traces)
    semantic = result.semantic
    candidate_set = semantic.candidate_ir_set
    branches = list(candidate_set.branches) if candidate_set is not None else []
    selected_id = candidate_set.selected_branch_id if candidate_set is not None else ""
    if selected_id:
        selected = [branch for branch in branches if branch.ir_branch_id == selected_id]
        if selected:
            branches = selected
    sources = [(branch.ir_branch_id, branch.faithful_ir, branch.semantic_assumptions) for branch in branches]
    if not sources and semantic.faithful_ir is not None:
        sources = [("source-ir", semantic.faithful_ir, ())]
    candidates, failures = [], []
    for branch_id, ir, assumptions in sources:
        if ir.source_text != result.requirement:
            failures.append(f"{branch_id}: source mismatch")
            continue
        for assignment in islice(temporal_assignments(ir), 256):
            try:
                concrete = materialize_temporal_assignment(ir, assignment) if assignment else ir
                formula = compile_ir_to_formula(concrete)
                evaluation = replace(evaluator.evaluate_formula(formula, traces, candidate_id=branch_id),
                                     ir_version=concrete.ir_version)
                candidates.append((evaluation, branch_id, tuple(sorted(assignment.items())), assumptions))
            except ValueError as exc:
                failures.append(f"{branch_id}: {exc}")
    if not candidates:
        raise ValueError("候选回退失败：没有通过编译校验的公式。" + "; ".join(failures[:3]))
    def rank(item):
        evaluation = item[0]
        matches = sum(trace.matched for trace in evaluation.trace_results)
        return (matches, evaluation.score, -evaluation.formula.size(), str(evaluation.formula))
    evaluation, branch_id, assignment, assumptions = max(candidates, key=rank)
    record = {"stage": "formula_candidate_fallback", "selected_formula": str(evaluation.formula),
              "selected_ir_version": evaluation.ir_version,
              "source_ir_version": semantic.faithful_ir.ir_version if semantic.faithful_ir is not None else "",
              "selected_branch_id": branch_id, "temporal_assignment": list(assignment),
              "candidate_count": len(candidates), "human_trace_count": len(traces),
              "raw_human_trace_count": len(raw_traces),
              "ir_selection_basis": candidate_set.selection_basis if candidate_set is not None else "",
              "selection_basis": "highest_human_trace_agreement_then_stable_order",
              "approval_granted": False, "validation_failures": failures}
    return replace(result, provisional_candidate=evaluation, fallback_candidate=evaluation,
                   candidates=[item[0].formula for item in candidates], evaluations=[item[0] for item in candidates],
                   candidate_status="provisional", selection_status="fallback_selected",
                   selection_method="candidate_fallback", selected_branch_id=branch_id,
                   selected_assignment=assignment, semantic_assumptions=tuple(assumptions),
                   final_approval=None, automatic_finalization_allowed=False,
                   approval_status="not_approved", learning_status="writeback_blocked", learning_example=None,
                   review_records=[*result.review_records, record],
                   output_contract_version="candidate-fallback-v1")
