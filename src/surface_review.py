"""Public compatibility facade for runtime surface and reading review APIs.

The implementation is split into :mod:`surface_segmentation` and
:mod:`plain_reading`; this facade keeps the original staged-rollout import names
stable for checkpoint and external callers.
"""

from __future__ import annotations

from typing import Any, Mapping

from plain_reading import (
    apply_plain_reading_decisions,
    apply_plain_reading_revision,
    assert_plain_reading_gate,
    build_plain_reading_review,
    close_plain_reading_revision_request,
    confirm_no_ambiguity,
    confirm_plain_reading_revision_classification,
    create_plain_reading_revision_request,
    decision_fingerprint,
    detect_rule_based_ambiguities,
    plain_reading_from_mapping as _plain_reading_from_mapping,
    request_ambiguity_generation,
    validate_plain_reading_review as _validate_plain_reading_review,
    validate_plain_reading_lexical_preservation,
    validate_rebuilt_plain_reading_axes,
    withdraw_ambiguity_concern,
)
from schemas import (
    ArtifactGenerator,
    PlainReadingAmbiguity,
    PlainReadingOption,
    PlainReadingReview,
    PlainReadingRevisionRequest,
)
from surface_segmentation import (
    DEFAULT_CUT_MARKER,
    SegmentationValidationError,
    assert_surface_segmentation_gate,
    build_surface_segmentation,
    choose_cut_marker,
    derive_segments,
    deterministic_marked_text,
    first_difference_offset,
    mark_segmentation_stale,
    revise_segmentation,
    segmentation_from_mapping,
    source_hash,
    validate_marked_text,
    validate_surface_segmentation,
)


def plain_reading_from_mapping(
    payload: Mapping[str, Any],
    segmentation: object | None = None,
) -> PlainReadingReview:
    """Parse a review; parent validation occurs immediately when available."""
    return _plain_reading_from_mapping(payload, segmentation)  # type: ignore[arg-type]


def validate_plain_reading_review(
    review: PlainReadingReview,
    segmentation: object | None = None,
    surface_segmentation: object | None = None,
) -> list[str]:
    parent = segmentation or surface_segmentation
    if parent is None:
        errors: list[str] = []
        for ambiguity in review.ambiguities:
            if len(ambiguity.options) != 2:
                errors.append(f"{ambiguity.ambiguity_id} must contain two options")
            normalized = {
                " ".join(option.plain_interpretation.split()).casefold()
                for option in ambiguity.options
            }
            if len(normalized) != len(ambiguity.options):
                errors.append(f"{ambiguity.ambiguity_id} options are not distinct")
        return errors
    try:
        _validate_plain_reading_review(review, parent)  # type: ignore[arg-type]
    except ValueError as exc:
        return [str(exc)]
    return []


plain_reading_decision_fingerprint = decision_fingerprint


__all__ = [
    "DEFAULT_CUT_MARKER",
    "SegmentationValidationError",
    "apply_plain_reading_decisions",
    "apply_plain_reading_revision",
    "assert_plain_reading_gate",
    "assert_surface_segmentation_gate",
    "build_plain_reading_review",
    "build_surface_segmentation",
    "choose_cut_marker",
    "close_plain_reading_revision_request",
    "confirm_no_ambiguity",
    "confirm_plain_reading_revision_classification",
    "create_plain_reading_revision_request",
    "derive_segments",
    "detect_rule_based_ambiguities",
    "deterministic_marked_text",
    "first_difference_offset",
    "mark_segmentation_stale",
    "plain_reading_decision_fingerprint",
    "plain_reading_from_mapping",
    "request_ambiguity_generation",
    "PlainReadingRevisionRequest",
    "revise_segmentation",
    "segmentation_from_mapping",
    "source_hash",
    "validate_marked_text",
    "validate_plain_reading_review",
    "validate_plain_reading_lexical_preservation",
    "validate_rebuilt_plain_reading_axes",
    "validate_surface_segmentation",
    "withdraw_ambiguity_concern",
]
