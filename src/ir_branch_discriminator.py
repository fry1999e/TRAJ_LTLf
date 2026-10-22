"""Online, formula-free discrimination for policy CandidateIRSet branches."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product
import hashlib
import json

from ir_trace_semantics import IRTraceSemanticsError, evaluate_ir_trace
from faithful_ir import compute_ir_version, validate_faithful_ir
from schemas import CandidateIRBranch, SemanticStructure, TraceAnnotation, TraceQuery
from schemas import TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION
from trace_formatter import format_trace_for_beginner
from ir_discriminator import shortest_discriminator
from execution_observer import emit


@dataclass(frozen=True)
class PolicyBranchSelectionOutcome:
    status: str
    semantic: SemanticStructure
    consistent_branch_ids: tuple[str, ...] = ()
    selected_branch_id: str = ""
    supporting_query_ids: tuple[str, ...] = ()
    reason: str = ""


def synthesize_policy_discriminating_queries(
    semantic: SemanticStructure,
    *,
    max_length: int = 5,
    max_count: int = 4,
    data_split: str = "test",
    round_index: int = 1,
    seen_traces: tuple[tuple[tuple[str, ...], ...], ...] = (),
    failure_reasons: tuple[str, ...] = (),
) -> list[TraceQuery]:
    """Return a small unlabeled trace set that separates policy branches."""

    candidate_set = semantic.candidate_ir_set
    if candidate_set is None or len(candidate_set.branches) < 2:
        return []
    branches = candidate_set.branches
    aps = tuple(
        dict.fromkeys(
            ap.name
            for branch in branches
            for ap in branch.faithful_ir.atomic_propositions
        )
    )
    if not aps or len(aps) > 4:
        return []
    unresolved_pairs = {
        tuple(sorted((left.ir_branch_id, right.ir_branch_id)))
        for left, right in combinations(branches, 2)
    }
    excluded = set(seen_traces)
    candidates: list[
        tuple[list[set[str]], tuple[bool, ...], set[tuple[str, str]]]
    ] = []
    for trace in _ranked_traces(aps, max_length=max_length):
        trace_key = tuple(tuple(sorted(step)) for step in trace)
        if trace_key in excluded:
            continue
        signature = _branch_signature(branches, trace)
        if signature is None or len(set(signature)) < 2:
            continue
        separated = {
            tuple(sorted((branches[left].ir_branch_id, branches[right].ir_branch_id)))
            for left, right in combinations(range(len(branches)), 2)
            if signature[left] != signature[right]
        }
        candidates.append((trace, signature, separated))

    selected: list[tuple[list[set[str]], tuple[bool, ...]]] = []
    remaining = set(unresolved_pairs)
    while remaining and candidates and len(selected) < max_count:
        best_index = max(
            range(len(candidates)),
            key=lambda index: (
                len(candidates[index][2] & remaining),
                -len(candidates[index][0]),
                -sum(len(step) for step in candidates[index][0]),
            ),
        )
        trace, signature, separated = candidates.pop(best_index)
        gain = separated & remaining
        if not gain:
            break
        selected.append((trace, signature))
        remaining -= gain

    shortest = shortest_discriminator(
        [branch.faithful_ir for branch in branches],
        max_length=min(max_length, 5), max_traces=4096,
    )
    if shortest.status == "witness_found":
        witness = [set(step) for step in shortest.trace]
        key = tuple(tuple(sorted(step)) for step in witness)
        if key not in excluded:
            selected = [(witness, shortest.partition),
                        *(item for item in selected if tuple(tuple(sorted(step)) for step in item[0]) != key)][:max_count]
            emit(
                "discriminator_witness_found",
                requirement_id=candidate_set.requirement_id,
                ir_versions=[branch.faithful_ir.ir_version for branch in branches],
                candidate_set_id=candidate_set.branch_set_id,
                trace=[sorted(step) for step in witness],
                entrypoint="synthesize_policy_discriminating_queries",
            )

    axis_ids = candidate_set.policy_axis_ids or tuple(
        dict.fromkeys(
            axis
            for branch in branches
            for axis in branch.ambiguity_axis_ids
        )
    )
    parameter_paths = tuple(
        dict.fromkeys(
            path
            for branch in branches
            for path in branch.controlled_parameter_paths
        )
    )
    node_ids = tuple(
        dict.fromkeys(
            path.split("/")[3]
            for path in parameter_paths
            if path.startswith("/faithful_ir/nodes/") and len(path.split("/")) > 3
        )
    )
    queries: list[TraceQuery] = []
    candidate_set_id = candidate_set.branch_set_id or (
        "candidate-set-"
        + _digest(
            {
                "requirement_id": candidate_set.requirement_id,
                "branches": [
                    (branch.ir_branch_id, branch.faithful_ir.ir_version)
                    for branch in branches
                ],
            },
            length=16,
        )
    )
    for trace, _hidden_signature in selected:
        query_id = f"IRQ-R{round_index}-" + _digest(
            {
                "branch_set_id": candidate_set_id,
                "round": round_index,
                "trace": [sorted(step) for step in trace],
            },
            length=16,
        )
        queries.append(
            TraceQuery(
                query_id=query_id,
                role="boundary",
                target_ambiguity=axis_ids[0] if axis_ids else "finite_trace_policy",
                constraints=(
                    "Judge the complete finite run exactly as listed.",
                    "No state exists after the final listed state.",
                    "Candidate identities and predictions are hidden from the reviewer.",
                    *(f"previous_round_failure={item}" for item in failure_reasons),
                ),
                trace=trace,
                proposed_label=None,
                adjudicated_label="pending",
                reason=(
                    "Backend-selected policy-candidate disagreement trace; it carries "
                    "no proposed truth label."
                ),
                beginner_view=format_trace_for_beginner(trace),
                data_split=data_split,
                label_source="human",
                formula_source="none",
                formula_visibility=(),
                ir_node_ids=node_ids,
                ir_version=candidate_set.branch_set_id,
                target_ambiguity_axis_ids=axis_ids,
                semantic_parameter_paths=parameter_paths,
                source_ir_branch_id="|".join(
                    branch.ir_branch_id for branch in branches
                ),
                generation_source="ir_candidate_branch_discrimination",
                purpose=TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION,
                parent_candidate_set_id=candidate_set_id,
            )
        )
    return queries


def select_policy_candidate_from_annotations(
    semantic: SemanticStructure,
    queries: list[TraceQuery],
    annotations: list[TraceAnnotation],
    *,
    accepted_label_sources: tuple[str, ...] = ("human",),
    selection_basis: str = "human_finite_trace_policy",
    decision_source: str = "human_finite_trace_policy",
    mapping_rule_id: str = "human-policy-trace-discrimination-v1",
    reviewer_default: str = "human-user",
) -> PolicyBranchSelectionOutcome:
    """Filter branches with locked human annotations."""

    candidate_set = semantic.candidate_ir_set
    if candidate_set is None or len(candidate_set.branches) < 2:
        return PolicyBranchSelectionOutcome(
            status="not_required",
            semantic=semantic,
            reason="No multi-branch policy candidate set is present.",
        )
    query_by_id = {query.query_id: query for query in queries}
    trusted = tuple(
        annotation
        for annotation in annotations
        if annotation.label_source in set(accepted_label_sources)
        and annotation.label in {"SAT", "UNSAT"}
        and annotation.query_id in query_by_id
    )
    if not trusted:
        return PolicyBranchSelectionOutcome(
            status="awaiting_human_trace_review",
            semantic=semantic,
            consistent_branch_ids=tuple(
                branch.ir_branch_id for branch in candidate_set.branches
            ),
            reason="No locked human SAT/UNSAT policy-boundary labels are available.",
        )

    consistent: list[CandidateIRBranch] = []
    for branch in candidate_set.branches:
        try:
            if all(
                evaluate_ir_trace(
                    branch.faithful_ir,
                    query_by_id[annotation.query_id].trace,
                )
                == (annotation.label == "SAT")
                for annotation in trusted
            ):
                consistent.append(branch)
        except IRTraceSemanticsError:
            continue
    query_ids = tuple(annotation.query_id for annotation in trusted)
    if len(consistent) != 1:
        return PolicyBranchSelectionOutcome(
            status=("multiple_remaining" if consistent else "none_consistent"),
            semantic=semantic,
            consistent_branch_ids=tuple(branch.ir_branch_id for branch in consistent),
            supporting_query_ids=query_ids,
            reason=(
                "Human trace evidence does not yet identify exactly one policy branch."
                if consistent
                else "Human trace evidence conflicts with every registered policy branch."
            ),
        )

    selected = consistent[0]
    selected_ir = replace(
        selected.faithful_ir,
        provenance=replace(
            selected.faithful_ir.provenance,
            extraction_mode=(
                "llm_nl_trace_selected_ir_branch"
                if "independent_judge" in accepted_label_sources
                else "human_selected_policy_branch"
            ),
        ),
        ir_version="",
    )
    selected_ir = replace(selected_ir, ir_version=compute_ir_version(selected_ir))
    validate_faithful_ir(selected_ir)
    selected = replace(selected, faithful_ir=selected_ir)
    selected_parameters = _selected_policy_parameters(selected)
    node_type, node_id, evidence_quote = _selected_policy_node(selected)
    decision_id = "POLICY-HUMAN-" + _digest(
        {
            "branch_set_id": candidate_set.branch_set_id,
            "branch_id": selected.ir_branch_id,
            "query_ids": query_ids,
        },
        length=20,
    )
    resolution = {
        "decision_id": decision_id,
        "axis_kind": (
            candidate_set.policy_axis_ids[0]
            if candidate_set.policy_axis_ids
            else "finite_trace_policy"
        ),
        "node_type": node_type,
        "requested_node_id": node_id,
        "evidence_quote": evidence_quote,
        "unresolved_parameters": sorted(selected_parameters),
        "profile_id": selected.policy_profile_id,
        "profile_version": selected.policy_profile_version,
        "parameters": selected_parameters,
        "controlled_parameters": list(selected_parameters),
        "selected_ambiguity_ids": [],
        "selected_readings": [],
        "decision_source": decision_source,
        "mapping_rule_id": mapping_rule_id,
        "supporting_trace_query_ids": list(query_ids),
        "reviewer": ",".join(
            dict.fromkeys(
                annotation.annotator
                for annotation in trusted
                if annotation.annotator.strip()
            )
        ) or reviewer_default,
        "user_label": selected.plain_language_paraphrase,
        "user_explanation": selected.generation_reason,
    }
    resolved_axes = candidate_set.ambiguity_axes
    if resolved_axes is not None:
        resolution_status = (
            "llm_trace_resolved"
            if "independent_judge" in accepted_label_sources
            else "resolved"
        )
        resolved_axes = replace(
            resolved_axes,
            axes=tuple(
                replace(axis, review_status=resolution_status)
                for axis in resolved_axes.axes
            ),
        )
    resolved_set = replace(
        candidate_set,
        branches=tuple(
            replace(
                branch,
                faithful_ir=(
                    selected_ir
                    if branch.ir_branch_id == selected.ir_branch_id
                    else branch.faithful_ir
                ),
                status=(
                    "selected" if branch.ir_branch_id == selected.ir_branch_id
                    else "rejected_by_expert_trace_evidence"
                ),
            )
            for branch in candidate_set.branches
        ),
        ambiguity_axes=resolved_axes,
        status="selected",
        ambiguity_debt=tuple(
            item
            for item in candidate_set.ambiguity_debt
            if not str(item).startswith("unresolved_ir_choice:")
        ),
        selection_status="unique",
        selection_basis=selection_basis,
        selected_branch_id=selected.ir_branch_id,
    )
    resolved_semantic = replace(
        semantic,
        faithful_ir=selected_ir,
        candidate_ir_set=resolved_set,
        semantic_policy_resolutions=(resolution,),
        semantic_resolution=None,
        semantic_status="ready",
    )
    return PolicyBranchSelectionOutcome(
        status="unique",
        semantic=resolved_semantic,
        consistent_branch_ids=(selected.ir_branch_id,),
        selected_branch_id=selected.ir_branch_id,
        supporting_query_ids=query_ids,
        reason="Exactly one policy branch matches all locked human trace labels.",
    )


def _selected_policy_parameters(branch: CandidateIRBranch) -> dict[str, object]:
    parameters: dict[str, object] = {}
    for path in branch.controlled_parameter_paths:
        parts = path.split("/")
        if len(parts) < 6:
            continue
        node_id = parts[3]
        name = parts[5]
        node = branch.faithful_ir.node_map.get(node_id)
        if node is not None and name in dict(node.parameters):
            parameters[name] = dict(node.parameters)[name]
    return parameters


def _selected_policy_node(branch: CandidateIRBranch) -> tuple[str, str, str]:
    for path in branch.controlled_parameter_paths:
        parts = path.split("/")
        if len(parts) >= 6:
            node = branch.faithful_ir.node_map.get(parts[3])
            if node is not None:
                return node.node_type, node.node_id, node.evidence_text
    return "", "", ""


def _branch_signature(
    branches: tuple[CandidateIRBranch, ...],
    trace: list[set[str]],
) -> tuple[bool, ...] | None:
    values: list[bool] = []
    for branch in branches:
        try:
            values.append(evaluate_ir_trace(branch.faithful_ir, trace))
        except IRTraceSemanticsError:
            return None
    return tuple(values)


def _ranked_traces(aps: tuple[str, ...], *, max_length: int):
    valuations = tuple(
        {
            ap for ap, enabled in zip(aps, bits) if enabled
        }
        for bits in product((False, True), repeat=len(aps))
    )
    for length in range(1, max_length + 1):
        traces = [list(items) for items in product(valuations, repeat=length)]
        traces.sort(
            key=lambda trace: (
                sum(len(step) for step in trace),
                tuple(tuple(sorted(step)) for step in trace),
            )
        )
        yield from traces


def _digest(payload: object, *, length: int) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:length]


__all__ = [
    "PolicyBranchSelectionOutcome",
    "select_policy_candidate_from_annotations",
    "synthesize_policy_discriminating_queries",
]
