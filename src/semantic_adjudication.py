"""Semantic authority and pending-challenge contracts for trace adjudication.

This module deliberately keeps pending review state outside FaithfulSemanticIR.
The IR remains the last complete, validated semantic artifact; a challenge gates
approval until a reviewer either reaffirms that artifact or materializes a new
validated IR version.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from typing import Any

from faithful_ir import compute_ir_version, validate_faithful_ir
from plain_reading import decision_fingerprint
from semantic_axes import semantic_axes_for_parameter
from semantic_readiness import (
    CAN_BIND_RESOLUTION,
    SemanticReadinessError,
    audit_semantic_readiness,
)
from schemas import (
    SemanticChallenge,
    SemanticChallengeSet,
    SemanticParameterDecision,
    SemanticResolutionRecord,
    SemanticStructure,
    TraceAdjudicationBrief,
    TraceQuery,
)


SEMANTIC_CHALLENGE_CONTRACT_VERSION = 1


def attach_semantic_resolution_record(
    semantic: SemanticStructure,
) -> SemanticStructure:
    """Bind a validated IR to an auditable parameter-decision parent artifact."""

    ir = semantic.faithful_ir
    if ir is None:
        return semantic
    validate_faithful_ir(ir)
    record = build_semantic_resolution_record(semantic)
    if (
        semantic.semantic_resolution == record
        and ir.resolution_id == record.resolution_id
        and ir.resolution_fingerprint == record.artifact_hash
    ):
        return semantic
    bound = replace(
        ir,
        resolution_id=record.resolution_id,
        resolution_fingerprint=record.artifact_hash,
        ir_version="",
    )
    bound = replace(bound, ir_version=compute_ir_version(bound))
    validate_faithful_ir(bound)
    candidate_set = semantic.candidate_ir_set
    if (
        candidate_set is not None
        and candidate_set.selection_status == "unique"
        and candidate_set.selected_branch_id
    ):
        candidate_set = replace(
            candidate_set,
            branches=tuple(
                replace(branch, faithful_ir=bound)
                if branch.ir_branch_id == candidate_set.selected_branch_id
                else branch
                for branch in candidate_set.branches
            ),
        )
    else:
        candidate_set = None
    resolved = replace(
        semantic,
        faithful_ir=bound,
        semantic_resolution=record,
        candidate_ir_set=candidate_set,
        approved_semantic_ir=None,
        labeled_trace_evidence=None,
        semantic_evidence_gate=None,
    )
    validate_semantic_resolution_binding(resolved)
    return resolved


def validate_semantic_resolution_binding(semantic: SemanticStructure) -> None:
    """Reject stale or partial resolution lineage before trace adjudication."""

    ir = semantic.faithful_ir
    record = semantic.semantic_resolution
    if ir is None:
        if record is not None:
            raise ValueError("SemanticResolutionRecord exists without FaithfulSemanticIR.")
        return
    if record is None:
        if ir.resolution_id or ir.resolution_fingerprint:
            raise ValueError("FaithfulSemanticIR has a stale semantic resolution binding.")
        return
    readiness = audit_semantic_readiness(ir)
    if not readiness.has_capability(CAN_BIND_RESOLUTION):
        raise SemanticReadinessError(CAN_BIND_RESOLUTION, readiness)
    if record.status != "resolved" or record.unresolved_questions:
        raise ValueError("SemanticResolutionRecord is not fully resolved.")
    if record.requirement_id != ir.requirement_id or record.source_hash != ir.source_hash:
        raise ValueError("SemanticResolutionRecord source lineage mismatch.")
    if record.resolution_id != ir.resolution_id:
        raise ValueError("SemanticResolutionRecord resolution_id mismatch.")
    if record.artifact_hash != ir.resolution_fingerprint:
        raise ValueError("SemanticResolutionRecord fingerprint mismatch.")
    expected_parameters = {
        f"/faithful_ir/nodes/{node.node_id}/parameters/{name}": (name, value)
        for node in ir.nodes
        for name, value in node.parameters
    }
    actual_parameters = {
        decision.semantic_path: (decision.parameter_name, decision.value)
        for decision in record.decisions
        if decision.approved
    }
    if actual_parameters != expected_parameters:
        raise ValueError(
            "SemanticResolutionRecord does not cover every current IR parameter."
        )


def build_semantic_resolution_record(
    semantic: SemanticStructure,
) -> SemanticResolutionRecord:
    ir = semantic.faithful_ir
    if ir is None:
        raise ValueError("Semantic resolution requires FaithfulSemanticIR.")
    readiness = audit_semantic_readiness(ir)
    if not readiness.has_capability(CAN_BIND_RESOLUTION):
        raise SemanticReadinessError(CAN_BIND_RESOLUTION, readiness)
    review = semantic.plain_reading_review
    validate_semantic_resolution_binding(semantic)
    decisions: list[SemanticParameterDecision] = []
    for node in ir.nodes:
        for parameter_name, value in node.parameters:
            parameter_reading_ids, parameter_reading_evidence = (
                _selected_parameter_reading_support(
                    review,
                    node_type=node.node_type,
                    node_source_span=node.source_span,
                    node_evidence=node.evidence_text,
                    parameter_name=parameter_name,
                )
            )
            policy_resolution = _matching_policy_resolution(
                semantic.semantic_policy_resolutions,
                node_type=node.node_type,
                node_evidence=node.evidence_text,
                parameter_name=parameter_name,
                value=value,
            )
            source = _parameter_decision_source(
                node.node_type,
                parameter_name,
                value,
                ir.source_text,
                has_selected_reading=bool(parameter_reading_ids),
                has_policy_resolution=policy_resolution is not None,
                policy_decision_source=(
                    str(policy_resolution.get("decision_source", ""))
                    if policy_resolution is not None
                    else ""
                ),
            )
            semantic_path = (
                f"/faithful_ir/nodes/{node.node_id}/parameters/{parameter_name}"
            )
            evidence = parameter_reading_evidence or node.evidence_text or ir.source_text
            parent_decision_ids = parameter_reading_ids
            mapping_rule_id = f"semantic-resolution-{node.node_type}-v1"
            if policy_resolution is not None:
                policy_decision_id = str(
                    policy_resolution.get("decision_id", "")
                ).strip()
                if policy_decision_id:
                    parent_decision_ids = tuple(
                        (*parameter_reading_ids, policy_decision_id)
                    )
                mapping_rule_id = str(
                    policy_resolution.get("mapping_rule_id", mapping_rule_id)
                )
            decision_payload = {
                "semantic_path": semantic_path,
                "parameter_name": parameter_name,
                "value": value,
                "decision_source": source,
                "source_evidence": evidence,
                "parent_decision_ids": parent_decision_ids,
                "mapping_rule_id": mapping_rule_id,
            }
            decisions.append(
                SemanticParameterDecision(
                    decision_id="SDEC-" + _stable_digest(decision_payload, length=16),
                    semantic_path=semantic_path,
                    parameter_name=parameter_name,
                    value=value,
                    decision_source=source,
                    source_evidence=evidence,
                    parent_decision_ids=parent_decision_ids,
                    mapping_rule_id=mapping_rule_id,
                    confidence=max(0.0, min(1.0, float(node.confidence))),
                    approved=True,
                )
            )

    reading_id = review.reading_review_id if review is not None else ir.reading_review_id
    reading_fingerprint = decision_fingerprint(review) if review is not None else ""
    record_payload = {
        "requirement_id": ir.requirement_id,
        "source_hash": ir.source_hash,
        "segmentation_id": ir.segmentation_id,
        "reading_review_id": reading_id,
        "decision_fingerprint": reading_fingerprint,
        "profile_id": ir.periodicity_profile_id,
        "decisions": [asdict(item) for item in decisions],
    }
    artifact_hash = _stable_digest(record_payload, length=64)
    return SemanticResolutionRecord(
        resolution_id="SRES-" + artifact_hash[:16],
        requirement_id=ir.requirement_id,
        source_hash=ir.source_hash,
        segmentation_id=ir.segmentation_id,
        reading_review_id=reading_id,
        decision_fingerprint=reading_fingerprint,
        profile_id=ir.periodicity_profile_id,
        status="resolved",
        decisions=tuple(decisions),
        unresolved_questions=(),
        reviewer=(
            next(
                (
                    str(resolution.get("reviewer", "")).strip()
                    for resolution in semantic.semantic_policy_resolutions
                    if resolution.get("decision_source")
                    == "expert_finite_trace_policy"
                    and str(resolution.get("reviewer", "")).strip()
                ),
                "",
            )
            or (
                review.reviewer_source
                if review is not None and review.reviewer_source
                else ir.provenance.approval_source or ir.provenance.extractor
            )
        ),
        artifact_hash=artifact_hash,
    )


def build_trace_adjudication_brief(
    requirement: str,
    semantic: SemanticStructure,
) -> TraceAdjudicationBrief:
    """Build the only semantic context that a formula-blind Judge may receive."""

    ir = semantic.faithful_ir
    if ir is None:
        fallback_payload = {
            "requirement": requirement,
            "aps": semantic.aps or (semantic.trigger, semantic.response),
            "trace_domain": semantic.trace_domain,
        }
        digest = _stable_digest(fallback_payload, length=16)
        return TraceAdjudicationBrief(
            brief_id="TBRIEF-legacy-" + digest,
            semantic_context_id="TCTX-legacy-" + digest,
            requirement_id="REQ-legacy",
            source_text=requirement,
            source_hash=_stable_digest(requirement, length=64),
            ir_version="",
            trace_domain=semantic.trace_domain,
            ap_definitions=tuple(
                (name, name)
                for name in (semantic.aps or (semantic.trigger, semantic.response))
                if name
            ),
            reviewed_plain_language_decisions=(
                "Legacy semantic input has no versioned IR; labels cannot create "
                "or revise semantic parameters.",
            ),
            global_trace_semantics=(
                "The last listed state is the end of the finite trace.",
            ),
        )
    validate_semantic_resolution_binding(semantic)
    review = semantic.plain_reading_review
    selected_ids, _selected_evidence = _selected_plain_readings(review)
    reviewed_decisions = _reviewed_plain_language_decisions(review)
    open_axes = tuple(dict.fromkeys(_open_plain_reading_axes(review)))
    trace_semantics = (
        "The trace is the complete finite run, indexed from its first listed state t0.",
        "The last listed state is the end of the run; no unlisted future state exists.",
        "An atomic proposition is true at exactly the listed states that contain it.",
    )
    ap_definitions = tuple(
        (
            ap.name,
            ap.evidence_text or (ap.aliases[0] if ap.aliases else ap.name),
        )
        for ap in ir.atomic_propositions
    )
    payload = {
        "requirement_id": ir.requirement_id,
        "source_hash": ir.source_hash,
        "ir_version": ir.ir_version,
        "reading_review_id": (
            review.reading_review_id if review is not None else ir.reading_review_id
        ),
        "decision_fingerprint": (
            decision_fingerprint(review) if review is not None else ""
        ),
        "resolution_id": ir.resolution_id,
        "resolution_fingerprint": ir.resolution_fingerprint,
        "reviewed_plain_language_decisions": reviewed_decisions,
        "open_plain_language_axes": open_axes,
        "global_trace_semantics": trace_semantics,
        "selected_plain_axis_ids": selected_ids,
    }
    semantic_context_id = "TCTX-" + _stable_digest(
        {
            "requirement_id": ir.requirement_id,
            "source_hash": ir.source_hash,
            "ir_version": ir.ir_version,
            "reading_review_id": payload["reading_review_id"],
            "decision_fingerprint": payload["decision_fingerprint"],
            "resolution_fingerprint": ir.resolution_fingerprint,
            "trace_domain": semantic.trace_domain,
            "ap_definitions": ap_definitions,
            "reviewed_plain_language_decisions": reviewed_decisions,
            "open_plain_language_axes": open_axes,
            "global_trace_semantics": trace_semantics,
        },
        length=20,
    )
    return TraceAdjudicationBrief(
        brief_id="TBRIEF-" + _stable_digest(payload, length=16),
        semantic_context_id=semantic_context_id,
        requirement_id=ir.requirement_id,
        source_text=requirement,
        source_hash=ir.source_hash,
        ir_version=ir.ir_version,
        reading_review_id=payload["reading_review_id"],
        decision_fingerprint=payload["decision_fingerprint"],
        resolution_id=ir.resolution_id,
        resolution_fingerprint=ir.resolution_fingerprint,
        trace_domain=semantic.trace_domain,
        ap_definitions=ap_definitions,
        reviewed_plain_language_decisions=reviewed_decisions,
        open_plain_language_axes=open_axes,
        global_trace_semantics=trace_semantics,
    )


def bind_trace_queries_to_brief(
    queries: list[TraceQuery],
    brief: TraceAdjudicationBrief,
) -> list[TraceQuery]:
    """Bind query provenance to the exact reviewed context shown to the Judge."""

    return [
        query
        if query.adjudication_brief_id == brief.brief_id
        else replace(query, adjudication_brief_id=brief.brief_id)
        for query in queries
    ]


def attach_semantic_challenges(
    semantic: SemanticStructure,
    challenges: tuple[SemanticChallenge, ...],
) -> SemanticStructure:
    if not challenges:
        return semantic
    ir = semantic.faithful_ir
    if ir is None:
        raise ValueError("Semantic challenges require a base FaithfulSemanticIR.")
    existing = (
        semantic.semantic_challenges.challenges
        if semantic.semantic_challenges is not None
        else ()
    )
    by_id = {challenge.challenge_id: challenge for challenge in existing}
    for challenge in challenges:
        by_id[challenge.challenge_id] = challenge
    challenge_set = SemanticChallengeSet(
        requirement_id=ir.requirement_id,
        base_ir_version=ir.ir_version,
        challenges=tuple(by_id.values()),
        status="open",
        contract_version=SEMANTIC_CHALLENGE_CONTRACT_VERSION,
    )
    return replace(
        semantic,
        semantic_challenges=challenge_set,
        semantic_status="needs_semantic_review",
        approved_semantic_ir=None,
        semantic_evidence_gate=None,
        persistent_rag_allowed=False,
    )


def open_semantic_challenges(
    semantic: SemanticStructure | None,
) -> tuple[SemanticChallenge, ...]:
    if semantic is None or semantic.semantic_challenges is None:
        return ()
    return semantic.semantic_challenges.open_challenges


def semantic_challenge_debt(
    semantic: SemanticStructure | None,
) -> tuple[str, ...]:
    return tuple(
        f"pending_semantic_challenge:{challenge.challenge_id}"
        for challenge in open_semantic_challenges(semantic)
    )


def _reviewed_plain_language_decisions(review) -> tuple[str, ...]:
    """Return only decisions made from explicit plain-language alternatives."""

    decisions: list[str] = []
    if review is not None:
        for ambiguity in review.ambiguities:
            if ambiguity.decision_status != "reviewer_selected":
                continue
            selected = next(
                (
                    option
                    for option in ambiguity.options
                    if option.option_id == ambiguity.decision
                ),
                None,
            )
            if selected is not None:
                decisions.append(
                    f"Reviewer selected reading {ambiguity.ambiguity_id}: "
                    + selected.plain_interpretation
                )
    return tuple(dict.fromkeys(decisions))


def _selected_plain_readings(review) -> tuple[tuple[str, ...], str]:
    if review is None:
        return (), ""
    ids: list[str] = []
    evidence: list[str] = []
    for ambiguity in review.ambiguities:
        if ambiguity.decision_status != "reviewer_selected":
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
            continue
        ids.append(ambiguity.ambiguity_id)
        evidence.append(selected.plain_interpretation)
    return tuple(ids), " | ".join(evidence)


def _selected_parameter_reading_support(
    review,
    *,
    node_type: str,
    node_source_span: tuple[int, int],
    node_evidence: str,
    parameter_name: str,
) -> tuple[tuple[str, ...], str]:
    """Bind a human decision only to typed parameters at its source locus."""

    if review is None or not semantic_axes_for_parameter(node_type, parameter_name):
        return (), ""
    ids: list[str] = []
    evidence: list[str] = []
    node_start, node_end = node_source_span
    normalized_node_evidence = " ".join(node_evidence.casefold().split())
    for ambiguity in review.ambiguities:
        if ambiguity.decision_status != "reviewer_selected":
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
            continue
        ambiguity_start, ambiguity_end = ambiguity.source_span
        spans_overlap = (
            node_start >= 0
            and ambiguity_start >= 0
            and node_start < ambiguity_end
            and ambiguity_start < node_end
        )
        ambiguity_evidence = " ".join(ambiguity.evidence_text.casefold().split())
        evidence_overlap = bool(
            normalized_node_evidence
            and ambiguity_evidence
            and (
                normalized_node_evidence in ambiguity_evidence
                or ambiguity_evidence in normalized_node_evidence
            )
        )
        if not spans_overlap and not evidence_overlap:
            continue
        ids.append(ambiguity.ambiguity_id)
        evidence.append(selected.plain_interpretation)
    return tuple(ids), " | ".join(evidence)


def _open_plain_reading_axes(review) -> tuple[str, ...]:
    if review is None:
        return ()
    return tuple(
        f"{ambiguity.ambiguity_id}: {ambiguity.question}"
        for ambiguity in review.ambiguities
        if ambiguity.decision_status
        not in {"reviewer_selected", "resolved", "not_ambiguous"}
    )


def _parameter_decision_source(
    node_type: str,
    parameter_name: str,
    value: Any,
    source_text: str,
    *,
    has_selected_reading: bool,
    has_policy_resolution: bool = False,
    policy_decision_source: str = "",
) -> str:
    if has_policy_resolution:
        if policy_decision_source == "expert_finite_trace_policy":
            return policy_decision_source
        return "project_policy"
    if has_selected_reading:
        return "human_selected_reading"
    normalized_source = " ".join(source_text.casefold().split())
    if str(value).casefold() in normalized_source:
        return "source_explicit"
    if node_type == "periodicity" and (
        (parameter_name == "terminal_policy" and value == "finite_prefix")
        or (parameter_name == "origin" and value == "first_occurrence")
    ):
        return "project_policy"
    return "deterministic_mapping"


def _matching_policy_resolution(
    resolutions: tuple[dict[str, Any], ...],
    *,
    node_type: str,
    node_evidence: str,
    parameter_name: str,
    value: Any,
) -> dict[str, Any] | None:
    """Match only parameters explicitly controlled by an applied policy."""

    normalized_evidence = " ".join(node_evidence.casefold().split())
    for resolution in resolutions:
        if not isinstance(resolution, dict):
            continue
        if str(resolution.get("node_type", "")) != node_type:
            continue
        controlled = resolution.get("controlled_parameters", ())
        if parameter_name not in controlled:
            continue
        parameters = resolution.get("parameters", {})
        if not isinstance(parameters, dict) or parameters.get(parameter_name) != value:
            continue
        requested_evidence = " ".join(
            str(resolution.get("evidence_quote", "")).casefold().split()
        )
        if requested_evidence and requested_evidence not in normalized_evidence:
            continue
        return resolution
    return None


def _stable_digest(payload: Any, *, length: int) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]
