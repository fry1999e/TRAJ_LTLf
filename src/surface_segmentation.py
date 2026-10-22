"""Lossless surface segmentation primitives used before semantic IR generation.

The module deliberately treats the source string as opaque Unicode code points.
It never strips, normalizes, case-folds, or rewrites the source.  LLMs only
propose inserted markers; every offset and segment body is derived here.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import re
import secrets
from typing import Any, Mapping, Sequence

from schemas import (
    ArtifactGenerator,
    MechanicalSegmentationValidation,
    SegmentationReviewRecord,
    SurfaceSegment,
    SurfaceSegmentation,
)
from validated_generation import RepairClass, ValidationIssue


DEFAULT_CUT_MARKER = "⟦CUT⟧"


class SegmentationValidationError(ValueError):
    """Raised when a marked string or persisted segmentation is not lossless."""

    def __init__(self, message: str, validation: MechanicalSegmentationValidation):
        super().__init__(message)
        self.validation = validation


def source_hash(source_text: str) -> str:
    """Return the SHA-256 of the exact, unmodified source string."""

    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def choose_cut_marker(source_text: str, preferred: str = DEFAULT_CUT_MARKER) -> str:
    """Choose a reserved marker that does not occur in ``source_text``."""

    if preferred and preferred not in source_text:
        return preferred
    while True:
        marker = f"⟦CUT-{secrets.token_hex(8)}⟧"
        if marker not in source_text:
            return marker


def first_difference_offset(left: str, right: str) -> int | None:
    """Return the first Unicode-code-point offset at which two strings differ."""

    for index, (left_char, right_char) in enumerate(zip(left, right)):
        if left_char != right_char:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def validate_marked_text(
    source_text: str,
    marked_text: str,
    marker: str,
) -> MechanicalSegmentationValidation:
    """Mechanically verify that removing markers reconstructs the source exactly."""

    if not marker:
        return _failed_validation(
            "invalid_marker",
            0,
            "切分标记不能为空。",
        )
    if marker in source_text:
        return _failed_validation(
            "marker_collision",
            source_text.find(marker),
            "切分标记已存在于原文中，必须为本次请求选择新的保留标记。",
        )
    reconstructed = marked_text.replace(marker, "")
    difference = first_difference_offset(source_text, reconstructed)
    if difference is not None:
        excerpt_start = max(0, difference - 12)
        excerpt_end = difference + 24
        excerpt = (
            f"source={source_text[excerpt_start:excerpt_end]!r}; "
            f"reconstructed={reconstructed[excerpt_start:excerpt_end]!r}"
        )
        return _failed_validation(
            "source_reconstruction_mismatch",
            difference,
            excerpt,
        )
    pieces = marked_text.split(marker)
    if any(piece == "" for piece in pieces):
        empty_index = next(index for index, piece in enumerate(pieces) if not piece)
        return _failed_validation(
            "empty_segment",
            _piece_source_offset(pieces, empty_index),
            "连续标记或首尾标记产生了空片段。",
        )
    return MechanicalSegmentationValidation(
        status="passed",
        exact_reconstruction=True,
        gap_count=0,
        overlap_count=0,
        reordered=False,
    )


def marked_text_validation_issues(
    source_text: str,
    marked_text: object,
    marker: str,
    *,
    stage: str = "segmentation",
) -> tuple[ValidationIssue, ...]:
    """Expose lossless reconstruction findings through the common issue protocol."""

    if not isinstance(marked_text, str):
        return (
            ValidationIssue(
                code="INVALID_OUTPUT_SHAPE",
                stage=stage,
                path="/marked_text",
                observed={"type": type(marked_text).__name__},
                expected={"type": "string"},
                repair_class=RepairClass.CONSTRAINED,
                allowed_paths=("/marked_text",),
                message="marked_text 必须是字符串",
            ),
        )
    validation = validate_marked_text(source_text, marked_text, marker)
    if validation.status == "passed":
        return ()
    return (
        ValidationIssue(
            code=(validation.failure_type or "SEGMENTATION_MECHANICAL_FAILURE").upper(),
            stage=stage,
            path="/marked_text",
            observed={
                "first_difference_offset": validation.first_difference_offset,
                "excerpt": validation.missing_or_changed_excerpt,
            },
            expected={
                "exact_source_reconstruction": True,
                "allowed_change": f"insert {marker!r} only",
            },
            context={
                "source_length": len(source_text),
                "offset_unit": "unicode_codepoint",
                "instruction": validation.instruction,
            },
            repair_class=RepairClass.DETERMINISTIC,
            allowed_paths=("/marked_text",),
            message=(
                "marked_text failed lossless reconstruction: "
                + validation.failure_type
            ),
        ),
    )


def derive_segments(
    source_text: str,
    marked_text: str,
    marker: str,
) -> tuple[SurfaceSegment, ...]:
    """Derive trustworthy spans from a mechanically valid marked string."""

    validation = validate_marked_text(source_text, marked_text, marker)
    if validation.status != "passed":
        raise SegmentationValidationError(
            "marked_text failed lossless reconstruction: " + validation.failure_type,
            validation,
        )
    segments: list[SurfaceSegment] = []
    cursor = 0
    for index, piece in enumerate(marked_text.split(marker), start=1):
        end = cursor + len(piece)
        segments.append(
            SurfaceSegment(
                segment_id=f"seg-{index}",
                index=index,
                start=cursor,
                end=end,
                text=source_text[cursor:end],
            )
        )
        cursor = end
    return tuple(segments)


def validate_surface_segmentation(
    segmentation: SurfaceSegmentation,
    *,
    current_source_text: str | None = None,
) -> MechanicalSegmentationValidation:
    """Validate hashes, coverage, ordering, ids, and exact source slices."""

    source_text = segmentation.source_text
    if current_source_text is not None and current_source_text != source_text:
        return _failed_validation(
            "stale_source_text",
            first_difference_offset(source_text, current_source_text),
            "当前原文与切分产物中的 source_text 不一致。",
        )
    if segmentation.source_hash != source_hash(source_text):
        return _failed_validation(
            "source_hash_mismatch",
            None,
            "source_hash 与当前 source_text 不一致。",
        )
    if not segmentation.segments:
        return _failed_validation("empty_segments", 0, "segments 不能为空。")

    gap_count = 0
    overlap_count = 0
    reordered = False
    cursor = 0
    seen_ids: set[str] = set()
    errors: list[str] = []
    for expected_index, segment in enumerate(segmentation.segments, start=1):
        if segment.index != expected_index:
            errors.append("segment index 不连续")
        if not segment.segment_id or segment.segment_id in seen_ids:
            errors.append("segment_id 为空或重复")
        seen_ids.add(segment.segment_id)
        if segment.start > cursor:
            gap_count += 1
        elif segment.start < cursor:
            overlap_count += 1
            reordered = True
        if segment.start != cursor:
            errors.append("相邻片段 span 不连续")
        if not (0 <= segment.start < segment.end <= len(source_text)):
            errors.append("片段 span 非法或为空")
        elif source_text[segment.start:segment.end] != segment.text:
            errors.append("segment.text 与 source_span 不一致")
        cursor = segment.end
    if segmentation.segments[0].start != 0:
        errors.append("第一个片段必须从 0 开始")
    if segmentation.segments[-1].end != len(source_text):
        errors.append("最后一个片段必须覆盖到原文末尾")
    reconstructed = "".join(segment.text for segment in segmentation.segments)
    if reconstructed != source_text:
        errors.append("片段拼接不能精确重构原文")
    if errors:
        return MechanicalSegmentationValidation(
            status="failed",
            exact_reconstruction=False,
            gap_count=gap_count,
            overlap_count=overlap_count,
            reordered=reordered,
            failure_type="invalid_derived_segments",
            first_difference_offset=first_difference_offset(source_text, reconstructed),
            missing_or_changed_excerpt="；".join(dict.fromkeys(errors)),
            instruction="所有 span 必须由后端从已验证的 marked_text 派生。",
        )
    return MechanicalSegmentationValidation(
        status="passed",
        exact_reconstruction=True,
        gap_count=0,
        overlap_count=0,
        reordered=False,
    )


def assert_surface_segmentation_gate(
    segmentation: SurfaceSegmentation,
    *,
    current_source_text: str | None = None,
    require_reviewed: bool = True,
) -> None:
    """Raise unless an artifact is mechanically valid and surface-reviewed."""

    validation = validate_surface_segmentation(
        segmentation,
        current_source_text=current_source_text,
    )
    if validation.status != "passed":
        raise SegmentationValidationError(
            "surface segmentation failed the mechanical gate: "
            + validation.failure_type,
            validation,
        )
    if require_reviewed and segmentation.status not in {
        "surface_reviewed",
        "ambiguity_flagged",
    }:
        raise SegmentationValidationError(
            "surface segmentation has not passed surface review",
            _failed_validation(
                "surface_review_required",
                None,
                f"status={segmentation.status}",
            ),
        )


def build_surface_segmentation(
    *,
    requirement_id: str,
    source_text: str,
    marked_text: str,
    marker: str,
    revision: int = 1,
    status: str = "review_pending",
    generator: ArtifactGenerator | None = None,
    experiment_mode: str = "rules",
    data_split: str = "test",
    label_source: str = "rules",
    review_basis: str = "mechanical_validation",
    review_history: Sequence[SegmentationReviewRecord] = (),
) -> SurfaceSegmentation:
    """Build a versioned artifact from an exact marked string."""

    validation = validate_marked_text(source_text, marked_text, marker)
    if validation.status != "passed":
        raise SegmentationValidationError(
            "marked_text failed lossless reconstruction: " + validation.failure_type,
            validation,
        )
    segments = derive_segments(source_text, marked_text, marker)
    digest = _segmentation_digest(source_text, segments)
    now = _timestamp()
    artifact = SurfaceSegmentation(
        segmentation_id=f"SEG-{requirement_id}-v{revision}-{digest}",
        revision=revision,
        requirement_id=requirement_id,
        source_text=source_text,
        source_hash=source_hash(source_text),
        marker=marker,
        segments=segments,
        mechanical_validation=validation,
        status=status,
        generator=generator or ArtifactGenerator(),
        experiment_mode=experiment_mode,
        data_split=data_split,
        label_source=label_source,
        review_basis=review_basis,
        persistent_rag_allowed=(
            data_split == "train"
            and status in {"surface_reviewed", "ambiguity_flagged"}
            and label_source in {"human", "independent_judge", "warmup_fixture"}
        ),
        created_at=now,
        updated_at=now,
        review_history=tuple(review_history),
    )
    assert_surface_segmentation_gate(
        artifact,
        require_reviewed=status in {"surface_reviewed", "ambiguity_flagged"},
    )
    return artifact


def deterministic_marked_text(
    source_text: str,
    marker: str | None = None,
) -> tuple[str, str]:
    """Insert shallow, deterministic reading boundaries without rewriting text."""

    marker = marker or choose_cut_marker(source_text)
    candidates: set[int] = set()
    # Keep punctuation in the preceding segment and preserve following spaces.
    for match in re.finditer(r"[,;；，]", source_text):
        candidates.add(match.end())
    for match in re.finditer(
        r"\s+(?=(?:then|or|but|unless|until|while|whereas|"
        r"那么|否则|或者|但是|直到|同时)\b)",
        source_text,
        flags=re.IGNORECASE,
    ):
        candidates.add(match.start())
    # Avoid tiny/empty fragments; deterministic rules are a reading aid only.
    accepted: list[int] = []
    previous = 0
    for offset in sorted(candidates):
        if offset <= 0 or offset >= len(source_text):
            continue
        if offset - previous < 4 or len(source_text) - offset < 2:
            continue
        accepted.append(offset)
        previous = offset
    marked = source_text
    for offset in reversed(accepted):
        marked = marked[:offset] + marker + marked[offset:]
    return marked, marker


def revise_segmentation(
    segmentation: SurfaceSegmentation,
    *,
    cut_offsets: Sequence[int],
    decision: str = "surface_reviewed",
    feedback: str = "",
    reviewer_id: str = "",
    reviewer_role: str = "ordinary_reviewer",
    label_source: str = "human",
    review_basis: str = "surface_only",
) -> SurfaceSegmentation:
    """Apply direct split/merge edits represented as backend code-point offsets."""

    source_text = segmentation.source_text
    offsets = sorted(set(int(offset) for offset in cut_offsets))
    if any(offset <= 0 or offset >= len(source_text) for offset in offsets):
        raise ValueError("cut offsets must be strictly inside source_text")
    marker = choose_cut_marker(source_text, segmentation.marker)
    marked_text = source_text
    for offset in reversed(offsets):
        marked_text = marked_text[:offset] + marker + marked_text[offset:]
    revision = segmentation.revision + 1
    result_id = (
        f"SEG-{segmentation.requirement_id}-v{revision}-"
        f"{_segmentation_digest_from_offsets(source_text, offsets)}"
    )
    now = _timestamp()
    record = SegmentationReviewRecord(
        round=len(segmentation.review_history) + 1,
        decision=decision,
        previous_segmentation_id=segmentation.segmentation_id,
        result_segmentation_id=result_id,
        feedback=feedback,
        reviewer_id=reviewer_id,
        reviewer_role=reviewer_role,
        review_basis=review_basis,
        label_source=label_source,
        started_at=now,
        completed_at=now,
    )
    revised = build_surface_segmentation(
        requirement_id=segmentation.requirement_id,
        source_text=source_text,
        marked_text=marked_text,
        marker=marker,
        revision=revision,
        status=decision,
        generator=segmentation.generator,
        experiment_mode=segmentation.experiment_mode,
        data_split=segmentation.data_split,
        label_source=label_source,
        review_basis=review_basis,
        review_history=(*segmentation.review_history, record),
    )
    # ``build_surface_segmentation`` computes the same deterministic id.  Keep
    # the explicit review record linked even if an implementation changes it.
    if record.result_segmentation_id != revised.segmentation_id:
        record = replace(record, result_segmentation_id=revised.segmentation_id)
        revised = replace(
            revised,
            review_history=(*segmentation.review_history, record),
        )
    return revised


def mark_segmentation_stale(
    segmentation: SurfaceSegmentation,
    current_source_text: str,
) -> SurfaceSegmentation:
    """Return a stale copy when the exact source hash changes."""

    if source_hash(current_source_text) == segmentation.source_hash:
        return segmentation
    return replace(
        segmentation,
        status="stale",
        persistent_rag_allowed=False,
        updated_at=_timestamp(),
    )


def segmentation_from_mapping(payload: Mapping[str, Any]) -> SurfaceSegmentation:
    """Load and revalidate a persisted surface segmentation artifact."""

    validation_payload = payload.get("mechanical_validation", {})
    if not isinstance(validation_payload, Mapping):
        raise ValueError("mechanical_validation must be an object")
    generator_payload = payload.get("generator", {})
    if not isinstance(generator_payload, Mapping):
        raise ValueError("generator must be an object")
    history: list[SegmentationReviewRecord] = []
    for item in payload.get("review_history", ()):
        if not isinstance(item, Mapping):
            raise ValueError("segmentation review_history entries must be objects")
        history.append(SegmentationReviewRecord(**dict(item)))
    segments = tuple(
        SurfaceSegment(
            segment_id=str(item.get("segment_id", "")),
            index=int(item.get("index", 0)),
            start=int(item.get("start", -1)),
            end=int(item.get("end", -1)),
            text=str(item.get("text", "")),
        )
        for item in payload.get("segments", ())
        if isinstance(item, Mapping)
    )
    validation = MechanicalSegmentationValidation(**dict(validation_payload))
    artifact = SurfaceSegmentation(
        segmentation_id=str(payload.get("segmentation_id", "")),
        revision=int(payload.get("revision", 1)),
        requirement_id=str(payload.get("requirement_id", "")),
        source_text=str(payload.get("source_text", "")),
        source_hash=str(payload.get("source_hash", "")),
        marker=str(payload.get("marker", DEFAULT_CUT_MARKER)),
        segments=segments,
        mechanical_validation=validation,
        schema_version=str(payload.get("schema_version", "1")),
        source_hash_algorithm=str(payload.get("source_hash_algorithm", "sha256")),
        offset_unit=str(payload.get("offset_unit", "unicode_codepoint")),
        normalized_text=str(payload.get("normalized_text", "")),
        review_scope=str(payload.get("review_scope", "surface_only")),
        semantic_authority=str(payload.get("semantic_authority", "none")),
        status=str(payload.get("status", "review_pending")),
        blocked_stage=str(payload.get("blocked_stage", "")),
        generator=ArtifactGenerator(**dict(generator_payload)),
        experiment_mode=str(payload.get("experiment_mode", "rules")),
        data_split=str(payload.get("data_split", "test")),
        label_source=str(payload.get("label_source", "rules")),
        review_basis=str(payload.get("review_basis", "mechanical_and_rules")),
        persistent_rag_allowed=bool(payload.get("persistent_rag_allowed", False)),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
        review_history=tuple(history),
        contract_version=str(payload.get("contract_version", "surface-segmentation-contract-v2")),
        canonicalizer_version=str(payload.get("canonicalizer_version", "surface-segmentation-canonicalizer-v2")),
        validator_version=str(payload.get("validator_version", "surface-segmentation-validator-v2")),
        repair_policy_version=str(payload.get("repair_policy_version", "validated-generation-repair-policy-v1")),
        artifact_hash=str(payload.get("artifact_hash", "")),
        issue_fingerprint=str(payload.get("issue_fingerprint", "")),
        repair_history=tuple(
            dict(item) for item in payload.get("repair_history", ()) if isinstance(item, Mapping)
        ),
    )
    validation = validate_surface_segmentation(artifact)
    if validation.status != "passed" and artifact.status not in {
        "mechanically_invalid",
        "blocked_error",
        "stale",
    }:
        raise SegmentationValidationError(
            "persisted surface segmentation is invalid", validation
        )
    return artifact


def segmentation_to_mapping(segmentation: SurfaceSegmentation) -> dict[str, Any]:
    return asdict(segmentation)


def _failed_validation(
    failure_type: str,
    offset: int | None,
    excerpt: str,
) -> MechanicalSegmentationValidation:
    return MechanicalSegmentationValidation(
        status="failed",
        exact_reconstruction=False,
        failure_type=failure_type,
        first_difference_offset=offset,
        missing_or_changed_excerpt=excerpt,
        instruction="只能插入切分标记，不能修改、删除、增加或重排原文字符。",
    )


def _piece_source_offset(pieces: Sequence[str], empty_index: int) -> int:
    return sum(len(piece) for piece in pieces[:empty_index])


def _segmentation_digest(
    source_text: str,
    segments: Sequence[SurfaceSegment],
) -> str:
    return _segmentation_digest_from_offsets(
        source_text,
        [segment.end for segment in segments[:-1]],
    )


def _segmentation_digest_from_offsets(
    source_text: str,
    offsets: Sequence[int],
) -> str:
    raw = source_hash(source_text) + ":" + ",".join(str(offset) for offset in offsets)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
