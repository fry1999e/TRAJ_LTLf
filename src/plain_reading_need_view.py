"""Typed non-final view for unresolved Plain Reading axes.

This artifact is deliberately below Semantic IR.  It preserves every reviewed
ordinary-language reading without inventing an authoritative selection or
asking downstream semantic components to choose one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from typing import Any, Mapping

from contract_versions import PLAIN_READING_NEED_VIEW_CONTRACT_VERSION
from plain_reading import (
    MAX_ACTIVE_CANDIDATE_IR_BRANCHES,
    decision_fingerprint,
    validate_plain_reading_review,
)
from schemas import PlainReadingReview, SurfaceSegmentation


PLAIN_READING_NEED_VIEW_ACTION = "resolve_plain_reading_selection"
PLAIN_READING_BUDGET_ACTION = "split_or_clarify_plain_reading_axes"


@dataclass(frozen=True)
class PlainReadingNeedViewOption:
    ambiguity_id: str
    option_id: str
    plain_interpretation: str
    difference_summary: str
    option_text_hash: str


@dataclass(frozen=True)
class PlainReadingNeedViewAxis:
    ambiguity_id: str
    question: str
    source_segment_ids: tuple[str, ...]
    source_span: tuple[int, int]
    evidence_text: str
    options: tuple[PlainReadingNeedViewOption, ...]


@dataclass(frozen=True)
class PlainReadingAssignment:
    assignment_id: str
    selections: tuple[tuple[str, str], ...]
    assignment_fingerprint: str


@dataclass(frozen=True)
class PlainReadingNeedView:
    need_view_id: str
    requirement_id: str
    source_hash: str
    segmentation_id: str
    segmentation_fingerprint: str
    reading_review_id: str
    reading_review_fingerprint: str
    decision_fingerprint: str
    parent_fingerprint: str
    axes: tuple[PlainReadingNeedViewAxis, ...]
    assignments: tuple[PlainReadingAssignment, ...]
    exact_combination_count: int
    combination_budget: int
    budget_exceeded: bool
    review_completed: bool = True
    single_ir_ready: bool = False
    finalization_allowed: bool = False
    next_stage: str = "plain_reading_resolution"
    required_authority: tuple[str, ...] = (
        "source_owner",
        "semantic_expert",
        "explicit_human_adjudication",
    )
    required_action: str = PLAIN_READING_NEED_VIEW_ACTION
    status: str = "need_view"
    contract_version: str = PLAIN_READING_NEED_VIEW_CONTRACT_VERSION
    artifact_hash: str = ""

    def __post_init__(self) -> None:
        validate_plain_reading_need_view(self)
        expected_hash = _artifact_hash(self)
        if self.artifact_hash != expected_hash:
            object.__setattr__(self, "artifact_hash", expected_hash)


def build_plain_reading_need_view(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    combination_budget: int = MAX_ACTIVE_CANDIDATE_IR_BRANCHES,
) -> PlainReadingNeedView:
    """Build the complete, bounded reading view for one unresolved review."""

    errors = validate_plain_reading_review(
        review,
        segmentation,
        require_reviewed=True,
        raise_on_error=False,
    )
    if errors:
        raise ValueError("invalid Plain Reading parent: " + "；".join(errors))
    if review.overall_status != "ambiguous_unresolved":
        raise ValueError("PlainReadingNeedView requires ambiguous_unresolved parent")
    if combination_budget < 1:
        raise ValueError("Plain Reading combination budget must be positive")

    unresolved = [
        ambiguity
        for ambiguity in review.ambiguities
        if ambiguity.decision_status == "ambiguous_unresolved"
    ]
    if not unresolved:
        raise ValueError("ambiguous_unresolved parent has no unresolved axes")
    unresolved.sort(key=lambda value: value.ambiguity_id)

    axes: list[PlainReadingNeedViewAxis] = []
    option_ids_by_axis: list[tuple[str, ...]] = []
    for ambiguity in unresolved:
        options = tuple(
            PlainReadingNeedViewOption(
                ambiguity_id=ambiguity.ambiguity_id,
                option_id=option.option_id,
                plain_interpretation=option.plain_interpretation,
                difference_summary=option.difference_summary,
                option_text_hash=_fingerprint(
                    {
                        "option_id": option.option_id,
                        "plain_interpretation": option.plain_interpretation,
                        "difference_summary": option.difference_summary,
                    }
                ),
            )
            for option in sorted(ambiguity.options, key=lambda value: value.option_id)
        )
        if len(options) < 2:
            raise ValueError("unresolved Plain Reading axis requires at least two options")
        axes.append(
            PlainReadingNeedViewAxis(
                ambiguity_id=ambiguity.ambiguity_id,
                question=ambiguity.question,
                source_segment_ids=tuple(ambiguity.source_segment_ids),
                source_span=tuple(ambiguity.source_span),
                evidence_text=ambiguity.evidence_text,
                options=options,
            )
        )
        option_ids_by_axis.append(tuple(option.option_id for option in options))

    exact_count = 1
    for option_ids in option_ids_by_axis:
        exact_count *= len(option_ids)

    parent_payload = {
        "requirement_id": review.requirement_id,
        "source_hash": review.source_hash,
        "segmentation_id": review.segmentation_id,
        "segmentation_fingerprint": segmentation.artifact_hash,
        "reading_review_id": review.reading_review_id,
        "reading_review_fingerprint": review.artifact_hash,
        "decision_fingerprint": decision_fingerprint(review),
        "axes": [asdict(axis) for axis in axes],
        "contract_version": PLAIN_READING_NEED_VIEW_CONTRACT_VERSION,
    }
    parent_fingerprint = _fingerprint(parent_payload)

    assignments: list[PlainReadingAssignment] = []
    if exact_count <= combination_budget:
        axis_ids = tuple(axis.ambiguity_id for axis in axes)
        for combination in itertools.product(*option_ids_by_axis):
            selections = tuple(zip(axis_ids, combination))
            assignment_fingerprint = _fingerprint(
                {
                    "parent_fingerprint": parent_fingerprint,
                    "selections": selections,
                }
            )
            assignments.append(
                PlainReadingAssignment(
                    assignment_id="PRASSIGN-" + assignment_fingerprint[:20],
                    selections=selections,
                    assignment_fingerprint=assignment_fingerprint,
                )
            )

    required_action = (
        PLAIN_READING_BUDGET_ACTION
        if exact_count > combination_budget
        else PLAIN_READING_NEED_VIEW_ACTION
    )
    need_view_id = "PRVIEW-" + _fingerprint(
        {
            "parent_fingerprint": parent_fingerprint,
            "combination_budget": combination_budget,
            "required_action": required_action,
        }
    )[:20]
    return PlainReadingNeedView(
        need_view_id=need_view_id,
        requirement_id=review.requirement_id,
        source_hash=review.source_hash,
        segmentation_id=review.segmentation_id,
        segmentation_fingerprint=segmentation.artifact_hash,
        reading_review_id=review.reading_review_id,
        reading_review_fingerprint=review.artifact_hash,
        decision_fingerprint=decision_fingerprint(review),
        parent_fingerprint=parent_fingerprint,
        axes=tuple(axes),
        assignments=tuple(assignments),
        exact_combination_count=exact_count,
        combination_budget=combination_budget,
        budget_exceeded=exact_count > combination_budget,
        required_action=required_action,
    )


def validate_plain_reading_need_view(view: PlainReadingNeedView) -> None:
    if view.contract_version != PLAIN_READING_NEED_VIEW_CONTRACT_VERSION:
        raise ValueError("unsupported PlainReadingNeedView contract")
    if not view.need_view_id or not view.requirement_id or not view.parent_fingerprint:
        raise ValueError("PlainReadingNeedView requires stable identity and lineage")
    if view.status != "need_view":
        raise ValueError("PlainReadingNeedView status must be need_view")
    if not view.review_completed or view.single_ir_ready or view.finalization_allowed:
        raise ValueError("PlainReadingNeedView finalization flags are inconsistent")
    if view.next_stage != "plain_reading_resolution":
        raise ValueError("PlainReadingNeedView has unsupported next_stage")
    if not view.axes or view.exact_combination_count < 2:
        raise ValueError("PlainReadingNeedView requires unresolved reading alternatives")

    axis_ids = tuple(axis.ambiguity_id for axis in view.axes)
    if len(set(axis_ids)) != len(axis_ids):
        raise ValueError("PlainReadingNeedView contains duplicate axes")
    calculated_count = 1
    valid_options: dict[str, set[str]] = {}
    for axis in view.axes:
        option_ids = tuple(option.option_id for option in axis.options)
        if len(option_ids) < 2 or len(set(option_ids)) != len(option_ids):
            raise ValueError("PlainReadingNeedView axis options are incomplete")
        if any(option.ambiguity_id != axis.ambiguity_id for option in axis.options):
            raise ValueError("PlainReadingNeedView option axis lineage mismatch")
        valid_options[axis.ambiguity_id] = set(option_ids)
        calculated_count *= len(option_ids)
    if calculated_count != view.exact_combination_count:
        raise ValueError("PlainReadingNeedView combination count mismatch")

    if view.budget_exceeded:
        if view.assignments:
            raise ValueError("over-budget PlainReadingNeedView must not be truncated")
        if view.required_action != PLAIN_READING_BUDGET_ACTION:
            raise ValueError("over-budget PlainReadingNeedView action mismatch")
        return
    if len(view.assignments) != view.exact_combination_count:
        raise ValueError("PlainReadingNeedView assignments are not complete")

    seen: set[tuple[tuple[str, str], ...]] = set()
    for assignment in view.assignments:
        if assignment.selections in seen:
            raise ValueError("PlainReadingNeedView contains duplicate assignments")
        seen.add(assignment.selections)
        if tuple(axis_id for axis_id, _ in assignment.selections) != axis_ids:
            raise ValueError("PlainReadingNeedView assignment axis set mismatch")
        if any(
            option_id not in valid_options[axis_id]
            for axis_id, option_id in assignment.selections
        ):
            raise ValueError("PlainReadingNeedView assignment references unknown option")


def plain_reading_need_view_from_mapping(
    mapping: Mapping[str, Any],
) -> PlainReadingNeedView:
    axes = tuple(
        PlainReadingNeedViewAxis(
            ambiguity_id=str(axis["ambiguity_id"]),
            question=str(axis["question"]),
            source_segment_ids=tuple(str(value) for value in axis["source_segment_ids"]),
            source_span=tuple(int(value) for value in axis["source_span"]),
            evidence_text=str(axis["evidence_text"]),
            options=tuple(
                PlainReadingNeedViewOption(
                    ambiguity_id=str(option["ambiguity_id"]),
                    option_id=str(option["option_id"]),
                    plain_interpretation=str(option["plain_interpretation"]),
                    difference_summary=str(option["difference_summary"]),
                    option_text_hash=str(option["option_text_hash"]),
                )
                for option in axis["options"]
            ),
        )
        for axis in mapping["axes"]
    )
    assignments = tuple(
        PlainReadingAssignment(
            assignment_id=str(assignment["assignment_id"]),
            selections=tuple(
                (str(selection[0]), str(selection[1]))
                for selection in assignment["selections"]
            ),
            assignment_fingerprint=str(assignment["assignment_fingerprint"]),
        )
        for assignment in mapping.get("assignments", ())
    )
    return PlainReadingNeedView(
        need_view_id=str(mapping["need_view_id"]),
        requirement_id=str(mapping["requirement_id"]),
        source_hash=str(mapping["source_hash"]),
        segmentation_id=str(mapping["segmentation_id"]),
        segmentation_fingerprint=str(mapping["segmentation_fingerprint"]),
        reading_review_id=str(mapping["reading_review_id"]),
        reading_review_fingerprint=str(mapping["reading_review_fingerprint"]),
        decision_fingerprint=str(mapping["decision_fingerprint"]),
        parent_fingerprint=str(mapping["parent_fingerprint"]),
        axes=axes,
        assignments=assignments,
        exact_combination_count=int(mapping["exact_combination_count"]),
        combination_budget=int(mapping["combination_budget"]),
        budget_exceeded=bool(mapping["budget_exceeded"]),
        review_completed=bool(mapping.get("review_completed", True)),
        single_ir_ready=bool(mapping.get("single_ir_ready", False)),
        finalization_allowed=bool(mapping.get("finalization_allowed", False)),
        next_stage=str(mapping.get("next_stage", "plain_reading_resolution")),
        required_authority=tuple(
            str(value) for value in mapping.get("required_authority", ())
        ),
        required_action=str(mapping["required_action"]),
        status=str(mapping.get("status", "need_view")),
        contract_version=str(mapping["contract_version"]),
        artifact_hash=str(mapping.get("artifact_hash", "")),
    )


def _artifact_hash(value: PlainReadingNeedView) -> str:
    payload = asdict(value)
    payload["artifact_hash"] = ""
    return _fingerprint(payload)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "PLAIN_READING_BUDGET_ACTION",
    "PLAIN_READING_NEED_VIEW_ACTION",
    "PlainReadingAssignment",
    "PlainReadingNeedView",
    "PlainReadingNeedViewAxis",
    "PlainReadingNeedViewOption",
    "build_plain_reading_need_view",
    "plain_reading_need_view_from_mapping",
    "validate_plain_reading_need_view",
]
