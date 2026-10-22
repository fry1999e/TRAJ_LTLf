"""Candidate-IR review and ApprovedSemanticIR freeze helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
from itertools import islice, product
import json

from data_examples import LabeledTrace
from faithful_ir import (
    FaithfulIRValidationError,
    compute_ir_version,
    validate_faithful_ir,
)
from ir_trace_semantics import IRTraceSemanticsError, evaluate_ir_trace
from schemas import (
    AmbiguityAxis,
    AmbiguityAxisSet,
    ApprovedSemanticIR,
    CandidateIRBranch,
    CandidateIRSet,
    LabeledTraceEvidenceSet,
    SemanticEvidenceGateReport,
    SemanticStructure,
)
from contract_versions import SEMANTIC_EVIDENCE_CONTRACT_VERSION
from plain_reading import MAX_ACTIVE_CANDIDATE_IR_BRANCHES
from semantic_adjudication import semantic_challenge_debt
from semantic_readiness import CAN_EXECUTE_IR, require_semantic_capability


def freeze_working_evidence(payload: dict) -> dict:
    """Content-address a working interpretation without granting approval.

    A JSON round trip severs mutable references to model responses. Unknown
    labels remain in the snapshot and are never converted into positive ones.
    """
    body = json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    body.pop("evidence_version", None)
    body["output_contract_version"] = "always-output-v1"
    body["approval_status"] = "not_approved"
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**body, "evidence_version": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}


def attach_candidate_ir_set(
    semantic: SemanticStructure,
    requirement: str,
    *,
    max_branch_count: int = MAX_ACTIVE_CANDIDATE_IR_BRANCHES,
) -> SemanticStructure:
    """Attach a bounded CandidateIRSet before the IR is frozen."""

    if semantic.faithful_ir is None:
        return semantic
    ir = _bind_ir_to_review_parents(semantic.faithful_ir, semantic)
    semantic = replace(semantic, faithful_ir=ir)
    from temporal_resolution import allowed_operators_for_node
    unresolved_ranges = tuple(node for node in ir.nodes if len(allowed_operators_for_node(node)) > 1)
    if unresolved_ranges:
        from candidate_delivery import materialize_temporal_assignment, temporal_assignments
        total = 1
        for node in unresolved_ranges:
            total *= len(allowed_operators_for_node(node))
        if total <= max_branch_count:
            branches = []
            for index, assignment in enumerate(temporal_assignments(ir), start=1):
                concrete = materialize_temporal_assignment(ir, assignment)
                branches.append(CandidateIRBranch(
                    ir_branch_id=f"temporal-branch-{index}", faithful_ir=concrete,
                    semantic_assumptions=tuple(f"temporal_option:{node_id}={operator}" for node_id, operator in sorted(assignment.items())),
                    plain_language_paraphrase="; ".join(assignment.values()),
                    generation_reason="Materialized only from source-supported temporal readings.", status="proposed"))
            return replace(semantic, faithful_ir=branches[0].faithful_ir,
                           candidate_ir_set=CandidateIRSet(
                               requirement_id=ir.requirement_id, nl_text=requirement, branches=tuple(branches),
                               status="policy_candidates_ready", selection_status="unresolved", max_branch_count=max_branch_count,
                               segmentation_id=ir.segmentation_id, reading_review_id=ir.reading_review_id,
                               candidate_catalog_version="temporal-range-candidates-v1"),
                           semantic_status="needs_semantic_review", ir_ambiguity_adjudication=None)
    existing = semantic.candidate_ir_set
    if (
        existing is not None
        and existing.candidate_catalog_version == "source-intent-graph-candidates-v1"
        and len(existing.branches) > 1
    ):
        rebound_branches = tuple(
            replace(
                branch,
                faithful_ir=_bind_ir_to_review_parents(branch.faithful_ir, semantic),
            )
            for branch in existing.branches
        )
        if any(
            branch.faithful_ir.source_hash != ir.source_hash
            or branch.faithful_ir.ap_context_fingerprint != ir.ap_context_fingerprint
            for branch in rebound_branches
        ):
            raise FaithfulIRValidationError(
                "SourceIntentGraph candidate set has stale source/AP lineage"
            )
        rebound = replace(
            existing,
            branches=rebound_branches,
            segmentation_id=ir.segmentation_id,
            reading_review_id=ir.reading_review_id,
            max_branch_count=max_branch_count,
        )
        selected_base = next(
            (
                branch.faithful_ir for branch in rebound_branches
                if branch.faithful_ir.ir_version == ir.ir_version
            ),
            rebound_branches[0].faithful_ir,
        )
        return replace(
            semantic,
            faithful_ir=selected_base,
            candidate_ir_set=rebound,
        )
    if (
        existing is not None
        and existing.status in {"policy_candidates_ready", "selected"}
        and existing.segmentation_id == ir.segmentation_id
        and existing.reading_review_id == ir.reading_review_id
        and existing.branches
    ):
        return semantic
    if semantic.candidate_ir_set is not None and _candidate_set_matches_parents(
        semantic.candidate_ir_set,
        ir,
        semantic,
        max_branch_count=max_branch_count,
    ):
        return semantic

    axis_set = _ambiguity_axis_set(ir.requirement_id, ir.source_text or requirement, semantic)
    branches, ambiguity_debt = _candidate_branches(
        semantic,
        ir,
        axis_set,
        requirement,
        max_branch_count=max_branch_count,
    )
    status = "needs_review" if ambiguity_debt else "proposed"
    from execution_observer import emit
    emit(
        "candidate_set_materialized",
        requirement_id=ir.requirement_id,
        ir_version=ir.ir_version,
        ir_artifact_hash=ir.artifact_hash,
        ap_context_fingerprint=ir.ap_context_fingerprint,
        branch_count=len(branches),
        ambiguity_debt=list(ambiguity_debt),
        entrypoint="semantic_freezing",
    )
    return replace(
        semantic,
        candidate_ir_set=CandidateIRSet(
            requirement_id=ir.requirement_id,
            nl_text=ir.source_text or requirement,
            branches=branches,
            ambiguity_axes=axis_set,
            status=status,
            segmentation_id=ir.segmentation_id,
            reading_review_id=ir.reading_review_id,
            ambiguity_debt=ambiguity_debt,
            max_branch_count=max_branch_count,
        ),
        ir_ambiguity_adjudication=None,
    )


def invalidate_semantic_after_plain_reading_revision(
    semantic: SemanticStructure,
    revised_review,
) -> SemanticStructure:
    """Invalidate every downstream semantic artifact after a reading revision."""

    previous_review = semantic.plain_reading_review
    if (
        previous_review is not None
        and revised_review.reading_review_id == previous_review.reading_review_id
    ):
        raise FaithfulIRValidationError(
            "accepted plain-reading revision must create a new reading_review_id"
        )
    return replace(
        semantic,
        faithful_ir=None,
        candidate_ir_set=None,
        approved_semantic_ir=None,
        labeled_trace_evidence=None,
        semantic_evidence_gate=None,
        plain_reading_review=revised_review,
        ir_ambiguity_adjudication=None,
        semantic_status="needs_semantic_review",
        persistent_rag_allowed=False,
    )


def _bind_ir_to_review_parents(ir, semantic: SemanticStructure):
    segmentation = semantic.surface_segmentation
    reading_review = semantic.plain_reading_review
    expected_segmentation_id = (
        segmentation.segmentation_id
        if segmentation is not None
        else reading_review.segmentation_id
        if reading_review is not None
        else ""
    )
    expected_reading_review_id = (
        reading_review.reading_review_id if reading_review is not None else ""
    )
    if ir.segmentation_id and expected_segmentation_id and (
        ir.segmentation_id != expected_segmentation_id
    ):
        raise FaithfulIRValidationError(
            "FaithfulSemanticIR references a stale surface segmentation"
        )
    if ir.reading_review_id and expected_reading_review_id and (
        ir.reading_review_id != expected_reading_review_id
    ):
        raise FaithfulIRValidationError(
            "FaithfulSemanticIR references a stale plain reading review"
        )
    bound = replace(
        ir,
        segmentation_id=expected_segmentation_id or ir.segmentation_id,
        reading_review_id=expected_reading_review_id or ir.reading_review_id,
        reading_review_basis=(
            reading_review.review_basis
            if reading_review is not None and reading_review.review_basis
            else ir.reading_review_basis
        ),
        ir_version="",
    )
    bound = replace(bound, ir_version=compute_ir_version(bound))
    validate_faithful_ir(bound)
    return bound


def _candidate_set_matches_parents(
    candidate_set: CandidateIRSet,
    ir,
    semantic: SemanticStructure,
    *,
    max_branch_count: int = MAX_ACTIVE_CANDIDATE_IR_BRANCHES,
) -> bool:
    if (
        candidate_set.segmentation_id != ir.segmentation_id
        or candidate_set.reading_review_id != ir.reading_review_id
        or candidate_set.max_branch_count != max_branch_count
    ):
        return False
    if any(
        branch.faithful_ir.ir_version != ir.ir_version
        for branch in candidate_set.branches
    ):
        return False
    expected_debt = set(_plain_reading_ambiguity_debt(semantic))
    if not expected_debt.issubset(set(candidate_set.ambiguity_debt)):
        return False
    review = semantic.plain_reading_review
    if review is not None and review.selected_options:
        assumption_text = "\n".join(
            assumption
            for branch in candidate_set.branches
            for assumption in branch.semantic_assumptions
        )
        return all(
            f"reviewer_selected:{ambiguity_id}={option_id}" in assumption_text
            for ambiguity_id, option_id in review.selected_options
        )
    return True


def _candidate_branches(
    semantic: SemanticStructure,
    ir,
    axis_set: AmbiguityAxisSet,
    requirement: str,
    *,
    max_branch_count: int = MAX_ACTIVE_CANDIDATE_IR_BRANCHES,
) -> tuple[tuple[CandidateIRBranch, ...], tuple[str, ...]]:
    review = semantic.plain_reading_review
    ambiguity_debt = _plain_reading_ambiguity_debt(semantic)
    unresolved = _unresolved_plain_ambiguities(semantic)
    axis_ids = tuple(axis.axis_id for axis in axis_set.axes)
    supported_spans = _supported_spans(
        ir.source_text or requirement,
        ir.root_node_id,
        ir.node_map,
    )
    temporal_ambiguities = tuple(
        ambiguity
        for ambiguity in ir.ambiguities
        if not ambiguity.resolved and len(ambiguity.alternative_patches) >= 2
    )
    unmaterialized_ir_ambiguities = tuple(
        ambiguity
        for ambiguity in ir.ambiguities
        if not ambiguity.resolved and len(ambiguity.alternative_patches) < 2
    )
    if unmaterialized_ir_ambiguities:
        return (), tuple(
            dict.fromkeys(
                (
                    *ambiguity_debt,
                    *(
                        f"unmaterialized_ir_ambiguity:{ambiguity.ambiguity_id}"
                        for ambiguity in unmaterialized_ir_ambiguities
                    ),
                )
            )
        )
    if temporal_ambiguities:
        return _temporal_candidate_branches(
            semantic,
            ir,
            axis_set,
            requirement,
            temporal_ambiguities=temporal_ambiguities,
            unresolved_plain=unresolved,
            ambiguity_debt=ambiguity_debt,
            supported_spans=supported_spans,
            max_branch_count=max_branch_count,
        )
    if unresolved:
        if max_branch_count < 1:
            raise FaithfulIRValidationError("max candidate branch count must be positive")
        for ambiguity in unresolved:
            if len(ambiguity.options) != 2:
                raise FaithfulIRValidationError(
                    f"plain ambiguity {ambiguity.ambiguity_id} needs exactly two options"
                )
        combinations = islice(
            product(*(ambiguity.options for ambiguity in unresolved)),
            max_branch_count,
        )
        unresolved_ids = tuple(item.ambiguity_id for item in unresolved)
        total_combinations = 1
        for ambiguity in unresolved:
            total_combinations *= len(ambiguity.options)
        if total_combinations > max_branch_count:
            ambiguity_debt = (
                *ambiguity_debt,
                "plain_reading_branch_budget_exhausted:"
                f"total={total_combinations}:limit={max_branch_count}",
            )
            # A truncated semantic domain can exclude the correct reading and
            # must never be offered as though it were a complete candidate set.
            return (), ambiguity_debt
        branches = tuple(
            CandidateIRBranch(
                ir_branch_id=f"ir-branch-{index}",
                faithful_ir=ir,
                supported_text_spans=supported_spans,
                ambiguity_axis_ids=tuple(dict.fromkeys((*axis_ids, *unresolved_ids))),
                semantic_assumptions=(
                    *_semantic_assumptions(ir),
                    *(
                        f"plain_reading_option:{ambiguity.ambiguity_id}={option.option_id}"
                        for ambiguity, option in zip(unresolved, options)
                    ),
                    "ordinary-language reading is unresolved and non-authoritative",
                ),
                plain_language_paraphrase=" ".join(
                    option.plain_interpretation for option in options
                ),
                generation_reason=(
                    "Bounded CandidateIRBranch retained for an unresolved ordinary-"
                    "language reading; no branch is silently treated as primary."
                ),
                status="needs_review",
            )
            for index, options in enumerate(combinations, start=1)
        )
        return branches, ambiguity_debt

    selected_assumptions, selected_paraphrases = _selected_reading_provenance(review)
    branch = CandidateIRBranch(
        ir_branch_id="ir-branch-1",
        faithful_ir=ir,
        supported_text_spans=supported_spans,
        ambiguity_axis_ids=axis_ids,
        semantic_assumptions=(
            *_semantic_assumptions(ir),
            *selected_assumptions,
        ),
        plain_language_paraphrase=(
            " ".join(selected_paraphrases) or semantic.explanation
        ),
        generation_reason=(
            "Primary NL-supported branch constrained by the reviewed ordinary-"
            "language selections; those selections remain non-authoritative until "
            "NL trace evidence freezes the IR."
            if selected_assumptions
            else
            "Primary NL-supported branch generated by NLU/compatibility IR "
            "extraction; it is not a truth source until NL trace evidence freezes it."
        ),
        status="proposed",
    )
    return (branch,), ambiguity_debt


def _temporal_candidate_branches(
    semantic: SemanticStructure,
    ir,
    axis_set: AmbiguityAxisSet,
    requirement: str,
    *,
    temporal_ambiguities: tuple,
    unresolved_plain: tuple,
    ambiguity_debt: tuple[str, ...],
    supported_spans: tuple[str, ...],
    max_branch_count: int,
) -> tuple[tuple[CandidateIRBranch, ...], tuple[str, ...]]:
    """Materialize executable IR alternatives instead of paraphrase-only branches."""

    if max_branch_count < 1:
        raise FaithfulIRValidationError("max candidate branch count must be positive")
    dimensions = [ambiguity.alternative_patches for ambiguity in temporal_ambiguities]
    dimensions.extend(ambiguity.options for ambiguity in unresolved_plain)
    total_combinations = 1
    for dimension in dimensions:
        total_combinations *= len(dimension)
    debt = [*ambiguity_debt]
    debt.extend(
        f"unresolved_ir_choice:{ambiguity.ambiguity_id}"
        for ambiguity in temporal_ambiguities
    )
    if total_combinations > max_branch_count:
        debt.append(
            "candidate_ir_branch_budget_exhausted:"
            f"total={total_combinations}:limit={max_branch_count}"
        )
        # Keep the ambiguity pending instead of silently selecting from an
        # incomplete Cartesian prefix.
        return (), tuple(dict.fromkeys(debt))

    axis_ids = tuple(axis.axis_id for axis in axis_set.axes)
    branches: list[CandidateIRBranch] = []
    temporal_count = len(temporal_ambiguities)
    for index, choices in enumerate(
        islice(product(*dimensions), max_branch_count),
        start=1,
    ):
        temporal_choices = tuple(choices[:temporal_count])
        plain_choices = tuple(choices[temporal_count:])
        patched_ir = _apply_temporal_alternative_patches(
            ir,
            temporal_ambiguities,
            temporal_choices,
        )
        temporal_assumptions = tuple(
            f"temporal_alternative:{ambiguity.ambiguity_id}={patch.alternative_id}"
            for ambiguity, patch in zip(temporal_ambiguities, temporal_choices)
        )
        plain_assumptions = tuple(
            f"plain_reading_option:{ambiguity.ambiguity_id}={option.option_id}"
            for ambiguity, option in zip(unresolved_plain, plain_choices)
        )
        paraphrases = [
            patch.description or patch.alternative_id
            for patch in temporal_choices
        ]
        paraphrases.extend(option.plain_interpretation for option in plain_choices)
        branches.append(
            CandidateIRBranch(
                ir_branch_id=f"ir-branch-{index}",
                faithful_ir=patched_ir,
                supported_text_spans=supported_spans,
                ambiguity_axis_ids=tuple(
                    dict.fromkeys(
                        (
                            *axis_ids,
                            *(a.ambiguity_id for a in temporal_ambiguities),
                            *(a.ambiguity_id for a in unresolved_plain),
                        )
                    )
                ),
                semantic_assumptions=(
                    *_semantic_assumptions(ir),
                    *temporal_assumptions,
                    *plain_assumptions,
                    "branch is concrete but non-authoritative until NL-only trace review",
                ),
                plain_language_paraphrase=" ".join(paraphrases),
                generation_reason=(
                    "Executable temporal alternative patches were applied to a "
                    "versioned IR branch; no branch is silently preferred."
                ),
                applied_alternative_ids=tuple(
                    patch.alternative_id for patch in temporal_choices
                ),
                status="needs_review",
            )
        )
    return tuple(branches), tuple(dict.fromkeys(debt))


def _apply_temporal_alternative_patches(
    ir,
    ambiguities: tuple,
    patches: tuple,
):
    nodes = {node.node_id: node for node in ir.nodes}
    selected_by_axis = {
        ambiguity.ambiguity_id: patch
        for ambiguity, patch in zip(ambiguities, patches)
    }
    for patch in patches:
        if patch.node_id not in nodes:
            raise FaithfulIRValidationError(
                f"temporal alternative {patch.alternative_id} references "
                f"unknown node {patch.node_id}"
            )
        node = nodes[patch.node_id]
        parameters = dict(node.parameters)
        parameters.update(dict(patch.parameter_updates))
        if patch.node_type == "next_step" and "steps" not in parameters:
            parameters["steps"] = 1
        nodes[patch.node_id] = replace(
            node,
            node_type=patch.node_type or node.node_type,
            parameters=tuple(parameters.items()),
        )
    patched = replace(
        ir,
        nodes=tuple(nodes[node.node_id] for node in ir.nodes),
        ambiguities=tuple(
            replace(
                ambiguity,
                resolved=(
                    True
                    if ambiguity.ambiguity_id in selected_by_axis
                    else ambiguity.resolved
                ),
            )
            for ambiguity in ir.ambiguities
        ),
        ir_version="",
    )
    patched = replace(patched, ir_version=compute_ir_version(patched))
    validate_faithful_ir(patched)
    require_semantic_capability(patched, CAN_EXECUTE_IR)
    return patched


def _selected_reading_provenance(review) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if review is None:
        return (), ()
    assumptions: list[str] = []
    paraphrases: list[str] = []
    for ambiguity in review.ambiguities:
        if ambiguity.decision_status != "reviewer_selected" or not ambiguity.decision:
            continue
        selected = next(
            (
                option
                for option in ambiguity.options
                if option.option_id == ambiguity.decision
            ),
            None,
        )
        if selected is None:
            raise FaithfulIRValidationError(
                f"plain ambiguity {ambiguity.ambiguity_id} selected an unknown option"
            )
        assumptions.append(
            f"reviewer_selected:{ambiguity.ambiguity_id}={selected.option_id}"
        )
        assumptions.extend(
            f"unselected_plain_reading:{ambiguity.ambiguity_id}={option.option_id}"
            for option in ambiguity.options
            if option.option_id != selected.option_id
        )
        if ambiguity.review_basis or review.review_basis:
            assumptions.append(
                "plain_reading_review_basis:"
                + (ambiguity.review_basis or review.review_basis)
            )
        paraphrases.append(selected.plain_interpretation)
    return tuple(assumptions), tuple(paraphrases)


def _unresolved_plain_ambiguities(semantic: SemanticStructure) -> tuple:
    review = semantic.plain_reading_review
    if review is None:
        return ()
    unresolved = tuple(
        ambiguity
        for ambiguity in review.ambiguities
        if ambiguity.decision_status == "ambiguous_unresolved"
    )
    return unresolved


def _plain_reading_ambiguity_debt(
    semantic: SemanticStructure,
) -> tuple[str, ...]:
    review = semantic.plain_reading_review
    if review is None:
        return ()
    resolved_by_candidate = set()
    candidate_set = semantic.candidate_ir_set
    if (
        candidate_set is not None
        and candidate_set.selection_status == "unique"
        and candidate_set.selected_branch_id
        and candidate_set.ambiguity_axes is not None
        and all(
            axis.review_status in {"resolved", "expert_trace_resolved", "llm_trace_resolved"}
            for axis in candidate_set.ambiguity_axes.axes
        )
    ):
        selected = next(
            (
                branch for branch in candidate_set.branches
                if branch.ir_branch_id == candidate_set.selected_branch_id
            ),
            None,
        )
        if selected is not None:
            for assumption in selected.semantic_assumptions:
                if assumption.startswith("plain_reading_option:") and "=" in assumption:
                    resolved_by_candidate.add(
                        assumption[len("plain_reading_option:"):].split("=", 1)[0]
                    )
    debt = [
        f"unresolved_plain_reading:{ambiguity.ambiguity_id}"
        for ambiguity in _unresolved_plain_ambiguities(semantic)
        if ambiguity.ambiguity_id not in resolved_by_candidate
    ]
    segmentation = semantic.surface_segmentation
    if segmentation is not None:
        from plain_reading import assess_plain_reading_ir_readiness

        readiness = assess_plain_reading_ir_readiness(review, segmentation)
        unresolved_ids = set(readiness.unresolved_ambiguity_ids)
        if not readiness.can_enter_semantic_ir and not unresolved_ids.issubset(resolved_by_candidate):
            debt.append(f"plain_reading_readiness:{readiness.reason_code}")
    debt.extend(
        f"plain_reading_revision_pending:{request.request_id}"
        for request in review.revision_requests
        if request.request_status in {"submitted", "suggested", "confirmed"}
    )
    return tuple(dict.fromkeys(debt))


def freeze_approved_semantic_ir(
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    *,
    approval_source: str = "nl_temporal_simulation_review",
    approval_record: str = "",
    approved_ir_branch_id: str = "",
    trace_plan=None,
    experimental_only: bool = False,
) -> SemanticStructure:
    """Freeze the reviewed IR using NL-labeled finite trace evidence."""

    if semantic.faithful_ir is None:
        return semantic
    existing_candidates = semantic.candidate_ir_set
    has_selected_policy_candidate = bool(
        existing_candidates is not None
        and existing_candidates.status == "selected"
        and existing_candidates.selected_branch_id
        and any(
            branch.ir_branch_id == existing_candidates.selected_branch_id
            and branch.faithful_ir.ir_version == semantic.faithful_ir.ir_version
            for branch in existing_candidates.branches
        )
    )
    if not has_selected_policy_candidate:
        semantic = attach_candidate_ir_set(semantic, semantic.faithful_ir.source_text)
    candidate_set = semantic.candidate_ir_set
    selected_branch = _selected_candidate_branch(
        candidate_set,
        approved_ir_branch_id,
        traces=traces,
    )
    ir = selected_branch.faithful_ir if selected_branch is not None else semantic.faithful_ir
    validate_faithful_ir(ir)
    semantic_for_freeze = replace(semantic, faithful_ir=ir)

    evidence = _labeled_trace_evidence_set(
        semantic_for_freeze,
        ir.requirement_id,
        ir.ir_version,
        traces,
    )
    remaining_debt = _remaining_ambiguity_debt(
        semantic_for_freeze,
        evidence,
        traces,
        selected_branch=selected_branch,
    )
    gate, evidence = _semantic_evidence_gate(
        semantic_for_freeze,
        ir,
        traces,
        evidence=evidence,
        selected_branch=selected_branch,
        remaining_debt=remaining_debt,
        trace_plan=trace_plan,
    )
    approved = gate.complete and not experimental_only
    provenance = replace(
        ir.provenance,
        approved=approved,
        approval_source=approval_source if approved else "",
    )
    frozen_ir = replace(ir, provenance=provenance)
    if not frozen_ir.ir_version:
        frozen_ir = replace(frozen_ir, ir_version=compute_ir_version(frozen_ir))

    approved_record = ApprovedSemanticIR(
        ir_version=frozen_ir.ir_version,
        approved_ir_branch_id=(
            selected_branch.ir_branch_id if selected_branch is not None else ""
        ),
        nl_text=frozen_ir.source_text,
        approved_ap_mapping=tuple((ap.ap_id, ap.name) for ap in frozen_ir.atomic_propositions),
        resolved_ambiguity_axes=(
            _resolved_ambiguity_axis_ids(semantic_for_freeze)
            + _selected_temporal_axis_ids(semantic_for_freeze, selected_branch)
            if approved
            else ()
        ),
        remaining_ambiguity_debt=remaining_debt,
        labeled_trace_evidence_ids=evidence.trace_ids,
        approval_record=approval_record or _approval_record_id(frozen_ir.requirement_id),
        approved=approved,
        evidence_status=gate.status,
        evidence_contract_version=SEMANTIC_EVIDENCE_CONTRACT_VERSION,
        evidence_gate_fingerprint=gate.trace_evidence_fingerprint,
    )
    candidate_set = _mark_candidate_set_status(
        candidate_set,
        "approved" if approved_record.approved else "needs_review",
        selected_branch_id=approved_record.approved_ir_branch_id,
        selected_ir=frozen_ir,
    )
    return replace(
        semantic,
        faithful_ir=frozen_ir,
        candidate_ir_set=candidate_set,
        approved_semantic_ir=approved_record,
        labeled_trace_evidence=evidence,
        semantic_evidence_gate=gate,
        semantic_status="ready" if approved_record.approved else "needs_semantic_review",
    )


def _semantic_evidence_gate(
    semantic: SemanticStructure,
    ir,
    traces: list[LabeledTrace],
    *,
    evidence: LabeledTraceEvidenceSet,
    selected_branch: CandidateIRBranch | None,
    remaining_debt: tuple[str, ...],
    trace_plan=None,
) -> tuple[SemanticEvidenceGateReport, LabeledTraceEvidenceSet]:
    """Evaluate one plan-derived evidence contract and reuse it everywhere."""

    # Local import keeps schema/IR compilation dependencies acyclic at import time.
    from ir_compiler import (
        bind_traces_to_ir_plan,
        compile_trace_obligation_plan,
        evaluate_ir_coverage,
    )

    # Evidence coverage may bind independently reviewed NL labels to an
    # unresolved operator range.  This is structural accounting, not witness
    # execution, so retain the full structural plan here.
    plan = trace_plan or compile_trace_obligation_plan(ir)
    if plan.ir_version != ir.ir_version:
        canonical_plan = compile_trace_obligation_plan(ir)
        if str(plan.plan_purpose).endswith(":boundary_disabled"):
            from schemas import TraceEvidencePolicy
            from trace_evidence_policy import project_trace_obligation_plan

            plan = project_trace_obligation_plan(
                canonical_plan,
                TraceEvidencePolicy(boundary_source="disabled"),
            )
        else:
            plan = canonical_plan
    bound_traces = bind_traces_to_ir_plan(
        list(traces),
        plan,
        generation_source="semantic_evidence_gate",
    )
    coverage = evaluate_ir_coverage(ir, plan, bound_traces)
    gating_obligations = tuple(
        obligation
        for obligation in plan.obligations
        if obligation.evidence_mode not in {"diagnostic_only", "operator_resolution"}
    )
    required_ids = tuple(obligation.obligation_id for obligation in gating_obligations)
    covered_set = set(coverage.covered_obligation_ids)
    covered_ids = tuple(item for item in required_ids if item in covered_set)
    missing_ids = tuple(item for item in required_ids if item not in covered_set)
    boundary_ids = tuple(
        obligation.obligation_id
        for obligation in gating_obligations
        if obligation.required_trace_role == "boundary" or obligation.boundary_case
    )
    covered_boundary_ids = tuple(item for item in boundary_ids if item in covered_set)
    conflicts = _conflicting_trace_ids(bound_traces)
    ir_semantic_conflicts = tuple(
        reason[len("trace ") : reason.index(" label conflicts")]
        for reason in coverage.reasons
        if reason.startswith("trace ") and " label conflicts" in reason
    )
    conflicts = tuple(dict.fromkeys((*conflicts, *ir_semantic_conflicts)))
    reasons: list[str] = list(evidence.evidence_debt)
    if selected_branch is None:
        reasons.append("candidate_ir_branch_not_selected")
    reasons.extend(remaining_debt)
    reasons.extend("missing_trace_obligation:" + item for item in missing_ids)
    reasons.extend("trace_label_conflict:" + item for item in conflicts)
    if ir_semantic_conflicts:
        reasons.append("nl_ir_semantic_contract_conflict")
    trace_ids = tuple(trace.trace_id or trace.name for trace in bound_traces)
    fingerprint = _trace_evidence_fingerprint(ir.ir_version, bound_traces)
    status = "rejected" if conflicts else "incomplete" if reasons else "complete"
    unique_reasons = tuple(dict.fromkeys(reasons))
    gate = SemanticEvidenceGateReport(
        requirement_id=ir.requirement_id,
        ir_version=ir.ir_version,
        obligation_plan_version=plan.plan_version,
        evidence_contract_version=SEMANTIC_EVIDENCE_CONTRACT_VERSION,
        selected_branch_id=(
            selected_branch.ir_branch_id if selected_branch is not None else ""
        ),
        required_obligation_ids=required_ids,
        covered_obligation_ids=covered_ids,
        missing_obligation_ids=missing_ids,
        required_boundary_obligation_ids=boundary_ids,
        covered_boundary_obligation_ids=covered_boundary_ids,
        trace_ids=trace_ids,
        trace_evidence_fingerprint=fingerprint,
        trusted_annotation_count=sum(
            1
            for trace in bound_traces
            if trace.annotation_locked
            and trace.label_source
            in {
                "human",
                "independent_judge",
                "frozen_independent_judge",
                "warmup_fixture",
                "warmup_train_gold",
            }
        ),
        conflicting_trace_ids=conflicts,
        remaining_ambiguity_debt=remaining_debt,
        status=status,
        reasons=unique_reasons,
    )
    return gate, replace(
        evidence,
        ir_version=ir.ir_version,
        status=status,
        evidence_debt=unique_reasons,
        evidence_contract_version=SEMANTIC_EVIDENCE_CONTRACT_VERSION,
    )


def _conflicting_trace_ids(traces: list[LabeledTrace]) -> tuple[str, ...]:
    labels: dict[tuple[tuple[str, ...], ...], tuple[bool, str]] = {}
    conflicts: list[str] = []
    for trace in traces:
        shape = tuple(tuple(sorted(step)) for step in trace.trace)
        trace_id = trace.trace_id or trace.name
        previous = labels.get(shape)
        if previous is not None and previous[0] != trace.expected:
            conflicts.extend((previous[1], trace_id))
        else:
            labels[shape] = (trace.expected, trace_id)
    return tuple(dict.fromkeys(conflicts))


def _trace_evidence_fingerprint(ir_version: str, traces: list[LabeledTrace]) -> str:
    payload = [
        {
            "trace_id": trace.trace_id or trace.name,
            "trace": [sorted(step) for step in trace.trace],
            "expected": bool(trace.expected),
            "kind": trace.kind,
            "obligation_ids": list(trace.obligation_ids),
        }
        for trace in traces
    ]
    encoded = json.dumps(
        {"ir_version": ir_version, "traces": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ambiguity_axis_set(
    requirement_id: str,
    nl_text: str,
    semantic: SemanticStructure,
) -> AmbiguityAxisSet:
    axes: list[AmbiguityAxis] = []
    reading_review = semantic.plain_reading_review
    if reading_review is not None:
        for ambiguity in reading_review.ambiguities:
            axes.append(
                AmbiguityAxis(
                    axis_id=ambiguity.ambiguity_id,
                    nl_span=nl_text,
                    possible_readings=tuple(
                        option.plain_interpretation
                        for option in ambiguity.options
                    ),
                    why_ambiguous=ambiguity.question,
                    required_trace_roles=(
                        "positive_witness",
                        "minimal_violation",
                        "boundary",
                    ),
                    review_status=ambiguity.decision_status,
                    source_segment_ids=ambiguity.source_segment_ids,
                    plain_reading_decision=ambiguity.decision,
                    review_basis=(
                        ambiguity.review_basis or reading_review.review_basis
                    ),
                )
            )
    ir = semantic.faithful_ir
    known = {axis.axis_id for axis in axes}
    if ir is not None:
        for ambiguity in ir.ambiguities:
            if ambiguity.ambiguity_id in known:
                continue
            axes.append(
                AmbiguityAxis(
                    axis_id=ambiguity.ambiguity_id,
                    nl_span=nl_text,
                    possible_readings=ambiguity.alternatives,
                    why_ambiguous=ambiguity.description,
                    required_trace_roles=("positive_witness", "minimal_violation", "boundary"),
                    review_status="resolved" if ambiguity.resolved else "pending",
                )
            )
            known.add(ambiguity.ambiguity_id)

    inferred_axes = _infer_axis_ids(semantic)
    known = {axis.axis_id for axis in axes}
    for axis_id in inferred_axes:
        if axis_id in known:
            continue
        axes.append(
            AmbiguityAxis(
                axis_id=axis_id,
                nl_span=nl_text,
                possible_readings=_axis_readings(axis_id),
                why_ambiguous="Potential finite-trace temporal interpretation axis from NL/IR cues.",
                required_trace_roles=("boundary",),
                review_status="pending",
            )
        )
    return AmbiguityAxisSet(requirement_id=requirement_id, axes=tuple(axes))


def _infer_axis_ids(semantic: SemanticStructure) -> tuple[str, ...]:
    axes: list[str] = []
    text = " ".join((semantic.temporal_keyword, semantic.scope_keyword, semantic.pattern)).lower()
    if any(token in text for token in ("eventually", "finally", "最终", "最后")):
        axes.append("eventually_vs_globally")
    if any(token in text for token in ("next", "strict", "下一", "之后")):
        axes.append("current_vs_strict_future")
    if "until" in text:
        axes.extend(
            (
                "strong_until_vs_weak_until",
                "immediate_witness_vs_strict_future",
                "left_condition_scope",
                "left_required_at_witness",
                "final_state_behavior",
            )
        )
    if "precedence" in text or "before" in text:
        axes.extend(
            (
                "strict_before_vs_same_step",
                "witness_required_vs_vacuous",
                "prerequisite_reusable_vs_one_to_one",
                "first_occurrence_vs_every_occurrence",
                "trace_start_vs_local_scope",
            )
        )
    if any(token in text for token in ("whenever", "always", "global", "每当", "始终")):
        axes.append("current_vs_global_scope")
    return tuple(dict.fromkeys(axes))


def _axis_readings(axis_id: str) -> tuple[str, ...]:
    return {
        "eventually_vs_globally": ("eventually at some finite position", "globally at every position"),
        "current_vs_strict_future": ("current position may satisfy", "only a strict future position satisfies"),
        "strong_until_vs_weak_until": ("witness must occur", "left side may persist without witness"),
        "immediate_witness_vs_strict_future": ("current position may witness", "witness must be strictly later"),
        "left_condition_scope": ("left holds at every position before witness", "left need not persist continuously"),
        "left_required_at_witness": ("left also holds at witness", "only right is required at witness"),
        "final_state_behavior": ("final position may witness", "witness is required earlier"),
        "strict_before_vs_same_step": ("prerequisite must be strictly earlier", "same-step prerequisite is allowed"),
        "witness_required_vs_vacuous": ("prerequisite must occur", "absence of constrained event is vacuously allowed"),
        "prerequisite_reusable_vs_one_to_one": ("one prerequisite unlocks later occurrences", "each occurrence needs a fresh prerequisite"),
        "first_occurrence_vs_every_occurrence": ("only the first constrained occurrence is gated", "every constrained occurrence is gated"),
        "trace_start_vs_local_scope": ("gate starts at trace start", "gate starts after local activation"),
        "current_vs_global_scope": ("current-position obligation", "all-position obligation"),
    }.get(axis_id, ())


def _supported_spans(source_text: str, root_node_id: str, node_map: dict) -> tuple[str, ...]:
    spans: list[str] = []
    for node in node_map.values():
        if node.node_id != root_node_id and node.node_type != "ap_condition":
            continue
        if node.evidence_text and node.evidence_text not in spans:
            spans.append(node.evidence_text)
    return tuple(spans or (source_text,))


def _semantic_assumptions(ir) -> tuple[str, ...]:
    assumptions = [
        "IR branch is generated from NL evidence and remains provisional before trace review."
    ]
    if ir.ambiguities:
        assumptions.append("Unresolved IR ambiguities require NL-labeled boundary evidence.")
    return tuple(assumptions)


def _labeled_trace_evidence_set(
    semantic: SemanticStructure,
    requirement_id: str,
    ir_version: str,
    traces: list[LabeledTrace],
) -> LabeledTraceEvidenceSet:
    trace_ids = tuple(trace.trace_id or trace.name for trace in traces)
    evidence_debt: list[str] = []
    if not trace_ids:
        evidence_debt.append("missing_nl_labeled_trace_evidence")
    if any(trace.nl_verdict in {"pending", "unclear"} for trace in traces):
        evidence_debt.append("unresolved_nl_trace_annotation")
    # Positive, negative, boundary, and relation-specific requirements are all
    # declared by TraceObligationPlan and evaluated by SemanticEvidenceGateReport.
    # Duplicating fixed kind quotas here previously made pure Boolean IRs fail
    # after their complete plan-derived coverage had already passed.
    return LabeledTraceEvidenceSet(
        requirement_id=requirement_id,
        ir_version=ir_version,
        trace_ids=trace_ids,
        positive_trace_ids=tuple(
            trace.trace_id or trace.name
            for trace in traces
            if trace.expected and trace.kind != "boundary"
        ),
        negative_trace_ids=tuple(
            trace.trace_id or trace.name
            for trace in traces
            if not trace.expected and trace.kind != "boundary"
        ),
        boundary_trace_ids=tuple(
            trace.trace_id or trace.name for trace in traces if trace.kind == "boundary"
        ),
        review_ids=tuple(
            trace.review_record_id
            for trace in traces
            if trace.review_record_id
        ),
        status="complete" if not evidence_debt else "incomplete" if trace_ids else "pending",
        evidence_debt=tuple(dict.fromkeys(evidence_debt)),
    )


def _remaining_ambiguity_debt(
    semantic: SemanticStructure,
    evidence: LabeledTraceEvidenceSet,
    traces: list[LabeledTrace],
    *,
    selected_branch: CandidateIRBranch | None = None,
) -> tuple[str, ...]:
    debt: list[str] = list(_plain_reading_ambiguity_debt(semantic))
    debt.extend(semantic_challenge_debt(semantic))
    selected_temporal_axes = set(
        _selected_temporal_axis_ids(semantic, selected_branch)
    )
    if semantic.candidate_ir_set is not None:
        debt.extend(
            item
            for item in semantic.candidate_ir_set.ambiguity_debt
            if not any(
                item == f"unresolved_ir_choice:{axis_id}"
                for axis_id in selected_temporal_axes
            )
        )
    ir = semantic.faithful_ir
    if ir is not None:
        debt.extend(
            f"unresolved_ir_ambiguity:{ambiguity.ambiguity_id}"
            for ambiguity in ir.ambiguities
            if not ambiguity.resolved
        )
    if (
        semantic.candidate_ir_set is not None
        and len(semantic.candidate_ir_set.branches) > 1
        and semantic.candidate_ir_set.selection_status != "unique"
        and not semantic.candidate_ir_set.ambiguity_debt
    ):
        debt.append("multiple_candidate_ir_branches_unresolved")
    return tuple(dict.fromkeys(debt))


def _mark_candidate_set_status(
    candidate_set: CandidateIRSet | None,
    status: str,
    *,
    selected_branch_id: str = "",
    selected_ir=None,
) -> CandidateIRSet | None:
    if candidate_set is None:
        return None
    resolved_axis_ids = {
        ambiguity.ambiguity_id
        for ambiguity in (getattr(selected_ir, "ambiguities", ()) or ())
        if ambiguity.resolved
    }
    return replace(
        candidate_set,
        status=status,
        ambiguity_debt=tuple(
            item
            for item in candidate_set.ambiguity_debt
            if not (
                status == "approved"
                and any(
                    item == f"unresolved_ir_choice:{axis_id}"
                    for axis_id in resolved_axis_ids
                )
            )
        ),
        branches=tuple(
            replace(
                branch,
                status=(
                    "approved"
                    if status == "approved"
                    and branch.ir_branch_id == selected_branch_id
                    else "not_selected"
                    if status == "approved"
                    else status
                ),
                faithful_ir=(
                    selected_ir
                    if selected_ir is not None
                    and branch.ir_branch_id == selected_branch_id
                    else branch.faithful_ir
                ),
            )
            for branch in candidate_set.branches
        ),
    )


def _selected_candidate_branch(
    candidate_set: CandidateIRSet | None,
    requested_branch_id: str,
    *,
    traces: list[LabeledTrace] | None = None,
) -> CandidateIRBranch | None:
    if candidate_set is None or not candidate_set.branches:
        return None
    if candidate_set.selected_branch_id:
        selected = next(
            (
                branch
                for branch in candidate_set.branches
                if branch.ir_branch_id == candidate_set.selected_branch_id
            ),
            None,
        )
        if selected is None:
            raise FaithfulIRValidationError(
                "CandidateIRSet.selected_branch_id references an unknown branch"
            )
        return selected
    if requested_branch_id:
        selected = next(
            (
                branch
                for branch in candidate_set.branches
                if branch.ir_branch_id == requested_branch_id
            ),
            None,
        )
        if selected is None:
            raise FaithfulIRValidationError(
                "approved_ir_branch_id does not reference CandidateIRSet"
            )
        return selected
    if len(candidate_set.branches) == 1:
        return candidate_set.branches[0]
    reviewed_traces = [
        trace
        for trace in (traces or ())
        if trace.annotation_locked
        and trace.label_source
        in {
            "human",
            "independent_judge",
            "frozen_independent_judge",
            "warmup_fixture",
        }
    ]
    if reviewed_traces:
        consistent: list[CandidateIRBranch] = []
        for branch in candidate_set.branches:
            try:
                matches = all(
                    evaluate_ir_trace(branch.faithful_ir, trace.trace)
                    == trace.expected
                    for trace in reviewed_traces
                )
            except IRTraceSemanticsError:
                matches = False
            if matches:
                consistent.append(branch)
        if len(consistent) == 1:
            return consistent[0]
    return None


def _selected_temporal_axis_ids(
    semantic: SemanticStructure,
    selected_branch: CandidateIRBranch | None,
) -> tuple[str, ...]:
    if selected_branch is None or not selected_branch.applied_alternative_ids:
        return ()
    candidate_set = semantic.candidate_ir_set
    if (
        candidate_set is not None
        and candidate_set.selection_status == "unique"
        and candidate_set.selected_branch_id == selected_branch.ir_branch_id
        and candidate_set.policy_axis_ids
    ):
        return candidate_set.policy_axis_ids
    selected_ids = set(selected_branch.applied_alternative_ids)
    ir = semantic.faithful_ir
    if ir is None:
        return ()
    return tuple(
        ambiguity.ambiguity_id
        for ambiguity in ir.ambiguities
        if any(
            patch.alternative_id in selected_ids
            for patch in ambiguity.alternative_patches
        )
    )


def _resolved_ambiguity_axis_ids(
    semantic: SemanticStructure,
) -> tuple[str, ...]:
    resolved: list[str] = []
    if semantic.faithful_ir is not None:
        resolved.extend(
            ambiguity.ambiguity_id
            for ambiguity in semantic.faithful_ir.ambiguities
            if ambiguity.resolved
        )
    if (
        semantic.candidate_ir_set is not None
        and semantic.candidate_ir_set.ambiguity_axes is not None
    ):
        resolved.extend(
            axis.axis_id
            for axis in semantic.candidate_ir_set.ambiguity_axes.axes
            if axis.review_status in {"resolved", "expert_trace_resolved", "llm_trace_resolved"}
        )
    return tuple(dict.fromkeys(resolved))


def _approval_record_id(requirement_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"approved-semantic-ir-{requirement_id}-{timestamp}"
