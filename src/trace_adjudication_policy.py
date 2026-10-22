"""One authoritative state machine for finite-trace adjudication.

The policy separates weak annotation failures from genuine semantic review.
In particular, no outcome in this module mutates ``FaithfulSemanticIR``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from data_examples import LabeledTrace
from schemas import (
    SemanticChallenge,
    SemanticStructure,
    TraceAdjudicationBrief,
    TraceAnnotation,
    TraceQuery,
)
from trace_annotation import adjudicate_trace_queries


@dataclass(frozen=True)
class TraceAdjudicationRoundOutcome:
    """Classified result of one independent trace-review round."""

    accepted_traces: tuple[LabeledTrace, ...] = ()
    unresolved_queries: tuple[TraceQuery, ...] = ()
    annotation_debt: tuple[str, ...] = ()
    query_debt: tuple[str, ...] = ()
    semantic_challenges: tuple[SemanticChallenge, ...] = ()
    next_action: str = "continue_to_coverage"

    @property
    def has_semantic_challenge(self) -> bool:
        return bool(self.semantic_challenges)

    @property
    def has_adjudication_debt(self) -> bool:
        return bool(self.annotation_debt or self.query_debt)

    @property
    def coverage_debt(self) -> tuple[str, ...]:
        return self.annotation_debt + self.query_debt + tuple(
            f"pending_semantic_challenge:{item.challenge_id}"
            for item in self.semantic_challenges
        )


def adjudicate_trace_round(
    queries: list[TraceQuery],
    annotations: list[TraceAnnotation],
    semantic: SemanticStructure,
    brief: TraceAdjudicationBrief,
) -> TraceAdjudicationRoundOutcome:
    """Classify Judge evidence without inventing or reopening IR semantics."""

    ir = semantic.faithful_ir
    by_id = {annotation.query_id: annotation for annotation in annotations}
    if len(by_id) != len(annotations):
        raise ValueError("Duplicate TraceAnnotation.query_id values are not allowed.")

    accepted: list[LabeledTrace] = []
    unresolved: list[TraceQuery] = []
    annotation_debt: list[str] = []
    query_debt: list[str] = []
    challenges: list[SemanticChallenge] = []

    for query in queries:
        annotation = by_id.get(query.query_id)
        if annotation is None:
            unresolved.append(query)
            annotation_debt.append(f"missing_annotation:{query.query_id}")
            continue

        challenge_type = _challenge_type(annotation, semantic)
        if challenge_type and ir is None:
            unresolved.append(query)
            annotation_debt.append(
                f"semantic_context_unversioned:{query.query_id}"
            )
            continue
        if challenge_type:
            unresolved.append(query)
            challenges.append(
                _semantic_challenge(
                    query=query,
                    annotation=annotation,
                    semantic=semantic,
                    brief=brief,
                    challenge_type=challenge_type,
                )
            )
            continue

        if annotation.label == "INFEASIBLE" or annotation.uncertainty_kind == "query_infeasible":
            unresolved.append(query)
            query_debt.append(f"query_infeasible:{query.query_id}")
            continue

        labeled, rejected = adjudicate_trace_queries([query], [annotation])
        accepted.extend(labeled)
        if rejected:
            unresolved.extend(rejected)
            annotation_debt.append(f"annotation_uncertain:{query.query_id}")

    next_action = "continue_to_coverage"
    if challenges:
        next_action = "await_semantic_review"
    elif query_debt:
        next_action = "regenerate_queries"
    elif annotation_debt:
        next_action = "retry_annotation"

    return TraceAdjudicationRoundOutcome(
        accepted_traces=tuple(accepted),
        unresolved_queries=tuple(unresolved),
        annotation_debt=tuple(annotation_debt),
        query_debt=tuple(query_debt),
        semantic_challenges=tuple(challenges),
        next_action=next_action,
    )


def _challenge_type(
    annotation: TraceAnnotation,
    semantic: SemanticStructure,
) -> str:
    """Require structured alternatives before escalating uncertainty to semantics."""

    if annotation.uncertainty_kind not in {
        "new_semantic_axis",
        "reviewed_decision_conflict",
    }:
        return ""
    if len(tuple(dict.fromkeys(annotation.candidate_readings))) < 2:
        return ""
    selected_axis_ids = _selected_plain_axis_ids(semantic)
    if selected_axis_ids.intersection(annotation.challenged_axis_ids):
        return "reviewed_decision_conflict"
    return annotation.uncertainty_kind


def _semantic_challenge(
    *,
    query: TraceQuery,
    annotation: TraceAnnotation,
    semantic: SemanticStructure,
    brief: TraceAdjudicationBrief,
    challenge_type: str,
) -> SemanticChallenge:
    ir = semantic.faithful_ir
    assert ir is not None
    parameter_paths = tuple(
        dict.fromkeys(
            annotation.affected_parameter_paths or query.semantic_parameter_paths
        )
    )
    axis_ids = tuple(
        dict.fromkeys(
            annotation.challenged_axis_ids or query.target_ambiguity_axis_ids
        )
    )
    readings = tuple(dict.fromkeys(annotation.candidate_readings))
    payload = {
        "requirement_id": ir.requirement_id,
        "base_ir_version": ir.ir_version,
        "brief_id": brief.brief_id,
        "query_id": query.query_id,
        "challenge_type": challenge_type,
        "node_ids": query.ir_node_ids,
        "parameter_paths": parameter_paths,
        "axis_ids": axis_ids,
        "readings": readings,
        "spans": annotation.supported_text_spans,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    evidence = tuple(
        item
        for item in (
            annotation.rationale.strip(),
            annotation.temporal_simulation_summary.strip(),
        )
        if item
    )
    return SemanticChallenge(
        challenge_id=f"SCH-{digest}",
        requirement_id=ir.requirement_id,
        base_ir_version=ir.ir_version,
        brief_id=brief.brief_id,
        query_id=query.query_id,
        challenge_type=challenge_type,
        affected_node_ids=query.ir_node_ids,
        affected_parameter_paths=parameter_paths,
        related_axis_ids=axis_ids,
        proposed_readings=readings,
        supported_text_spans=annotation.supported_text_spans,
        evidence=evidence,
        annotation_label=annotation.label,
        annotation_confidence=annotation.confidence,
        status="pending",
    )


def _selected_plain_axis_ids(semantic: SemanticStructure) -> set[str]:
    review = semantic.plain_reading_review
    if review is None:
        return set()
    return {
        ambiguity.ambiguity_id
        for ambiguity in review.ambiguities
        if ambiguity.decision_status == "reviewer_selected"
    }
