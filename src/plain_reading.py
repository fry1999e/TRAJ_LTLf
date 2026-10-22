"""Ordinary-language ambiguity artifacts between surface review and IR."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
import re
import uuid
from typing import Any, Callable, Mapping, Sequence

from ap_identity import (
    APIdentityValidationError,
    AP_ADDS,
    AP_INVENTORY_UNAPPROVED,
    AP_MAPPING_REQUIRED,
    AP_OMITS,
    AP_RENAMES,
    build_declared_catalog_ap_inventory,
    build_symbolic_ap_inventory,
    compare_override_ap_inventories,
    controlled_symbolic_profile,
    propose_grounded_ap_inventory,
    v1_symbolic_tokens,
    validate_ap_inventory_integrity,
    validate_override_ap_comparison_integrity,
)
from contract_versions import (
    HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION,
    HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION,
    OVERRIDE_AP_COMPARISON_CONTRACT_VERSION,
    PLAIN_READING_READINESS_CONTRACT_VERSION,
)

from schemas import (
    AmbiguityGenerationTrigger,
    APIdentityMapping,
    APInventory,
    APRepresentationProfile,
    ArtifactGenerator,
    DeclaredAPCatalog,
    HumanSemanticOverride,
    NoAmbiguityConfirmation,
    OverrideAPComparison,
    PlainReadingAmbiguity,
    PlainReadingOption,
    PlainReadingReview,
    PlainReadingRevisionRequest,
    PLAIN_READING_SCHEMA_VERSION,
    SurfaceSegment,
    SurfaceSegmentation,
)
from semantic_source import is_complete_formula_like_source
from surface_segmentation import assert_surface_segmentation_gate, source_hash
from validated_generation import RepairClass, ValidationIssue, json_pointer


class PlainReadingValidationError(ValueError):
    """Raised when ordinary-language options or decisions violate the contract."""

    def __init__(self, message: str, *, errors: Sequence[str] = ()):
        super().__init__(message)
        self.errors = tuple(errors) if errors else (message,)


@dataclass(frozen=True)
class PlainReadingIRReadiness:
    """The sole routing decision between Plain Reading and Semantic IR."""

    disposition: str
    can_enter_semantic_ir: bool
    reason_code: str
    unresolved_ambiguity_ids: tuple[str, ...]
    pending_revision_request_ids: tuple[str, ...]
    suggested_rerun_stage: str
    reading_review_id: str
    decision_fingerprint: str
    validation_errors: tuple[str, ...] = ()
    contract_version: str = PLAIN_READING_READINESS_CONTRACT_VERSION

    @property
    def can_attempt_candidate_ir(self) -> bool:
        """Unresolved valid readings share one route, regardless of reviewer."""
        return self.disposition == "semantic_resolution_required" and not self.validation_errors


class PlainReadingSemanticResolutionRequired(PlainReadingValidationError):
    """A valid reviewed artifact still contains a real unresolved reading."""

    code = "PLAIN_READING_SEMANTIC_RESOLUTION_REQUIRED"
    stage = "plain_reading"
    status = "awaiting_plain_reading_resolution"
    suggested_rerun_stage = "plain_reading"

    def __init__(
        self,
        review: PlainReadingReview,
        readiness: PlainReadingIRReadiness,
        *, segmentation: SurfaceSegmentation | None = None,
    ) -> None:
        self.review = review
        self.readiness = readiness
        self.segmentation = segmentation
        self.requirement_id = review.requirement_id
        self.reading_review_id = readiness.reading_review_id
        self.unresolved_ambiguity_ids = readiness.unresolved_ambiguity_ids
        unresolved = set(readiness.unresolved_ambiguity_ids)
        self.unresolved_axes = tuple(
            {
                "ambiguity_id": ambiguity.ambiguity_id,
                "question": ambiguity.question,
            }
            for ambiguity in review.ambiguities
            if ambiguity.ambiguity_id in unresolved
        )
        self.decision_fingerprint = readiness.decision_fingerprint
        self.diagnostics = {
            "reason_code": self.code,
            "reading_review_id": review.reading_review_id,
            "unresolved_axes": self.unresolved_axes,
            "user_message": "模型读法尚未形成完整的可执行轨迹候选，已保留待处理；无需填写专业语义表单。",
        }
        super().__init__(
            self.code
            + ": unresolved plain-reading axes cannot enter FaithfulSemanticIR"
        )


POST_INDEPENDENT_REVISION_ALL_AXES_DISCARDED_BASIS = (
    "post_independent_revision_all_axes_discarded"
)
POST_INDEPENDENT_REVISION_REVIEW_OPERATION = (
    "post_independent_revision_axes_reviewed"
)


def _post_independent_revision_scope_records(
    review: PlainReadingReview,
) -> tuple[dict[str, dict[str, Any]], set[str], tuple[str, ...]]:
    """Return persisted independent-revision review scopes and structural errors."""

    applications: dict[str, dict[str, Any]] = {}
    completed: set[str] = set()
    errors: list[str] = []
    for item in review.review_history:
        operation = str(item.get("operation", ""))
        if operation == "plain_reading_revision_applied" and item.get(
            "post_revision_review_kind"
        ):
            scope_id = str(item.get("post_revision_review_scope_id", ""))
            kind = str(item.get("post_revision_review_kind", ""))
            axis_ids = tuple(
                str(axis_id)
                for axis_id in item.get("post_revision_review_axis_ids", ())
                if str(axis_id)
            )
            if kind != "new_independent_axis":
                errors.append("post revision review scope 必须来自 new_independent_axis")
            if not scope_id:
                errors.append("post revision review scope 缺少 scope_id")
                continue
            if scope_id in applications:
                errors.append(f"post revision review scope_id 重复: {scope_id}")
                continue
            if not axis_ids or len(axis_ids) != len(set(axis_ids)):
                errors.append(
                    f"post revision review scope {scope_id} 的 axis IDs 为空或重复"
                )
            applications[scope_id] = {
                **dict(item),
                "post_revision_review_axis_ids": axis_ids,
            }
        elif operation == POST_INDEPENDENT_REVISION_REVIEW_OPERATION:
            scope_id = str(item.get("scope_id", ""))
            if not scope_id:
                errors.append("post independent revision completion 缺少 scope_id")
            elif scope_id not in applications:
                errors.append(
                    "post independent revision completion 引用未知 scope_id: "
                    + scope_id
                )
            elif scope_id in completed:
                errors.append(
                    "post independent revision completion 重复关闭 scope_id: "
                    + scope_id
                )
            else:
                completed.add(scope_id)
    return applications, completed, tuple(errors)


def active_post_independent_revision_review_scope(
    review: PlainReadingReview,
) -> dict[str, Any] | None:
    """Return the latest unclosed independent-axis post-review scope, if any."""

    applications, completed, errors = _post_independent_revision_scope_records(review)
    if errors:
        raise PlainReadingValidationError("；".join(errors), errors=errors)
    for scope_id in reversed(tuple(applications)):
        if scope_id not in completed:
            return dict(applications[scope_id])
    return None


def is_post_independent_revision_all_axes_discarded(
    review: PlainReadingReview,
) -> bool:
    """Return whether the review is the explicit non-regenerating empty-axis terminal."""

    if (
        review.overall_status != "needs_source_clarification"
        or review.review_basis != POST_INDEPENDENT_REVISION_ALL_AXES_DISCARDED_BASIS
        or review.ambiguities
    ):
        return False
    completion = next(
        (
            item
            for item in reversed(review.review_history)
            if item.get("operation") == POST_INDEPENDENT_REVISION_REVIEW_OPERATION
        ),
        None,
    )
    return bool(completion and completion.get("all_axes_discarded"))


ALLOWED_REVIEWED_STATUSES = {
    "no_ambiguity",
    "plain_reading_reviewed",
    "ambiguous_unresolved",
}

MAX_PLAIN_READING_REVISION_ROUNDS = 2
MAX_ACTIVE_CANDIDATE_IR_BRANCHES = 5

_ENGLISH_EXPLICIT_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|without|cannot|neither|nor)\b|\b[a-z]+n['’]t\b",
    re.IGNORECASE,
)
_CHINESE_EXPLICIT_NEGATION_RE = re.compile(r"从不|没有|不|非")

_QUANTITY_WORD_VALUES = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "eleventh": "11",
    "twelfth": "12",
    "thirteenth": "13",
    "fourteenth": "14",
    "fifteenth": "15",
    "sixteenth": "16",
    "seventeenth": "17",
    "eighteenth": "18",
    "nineteenth": "19",
    "twentieth": "20",
    "thirtieth": "30",
    "fortieth": "40",
    "fiftieth": "50",
    "sixtieth": "60",
    "seventieth": "70",
    "eightieth": "80",
    "ninetieth": "90",
}
_QUANTITY_WORD_RE = re.compile(
    r"\b(?:"
    + "|".join(sorted(_QUANTITY_WORD_VALUES, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
_DIGIT_QUANTITY_RE = re.compile(
    r"(?<!\w)(\d+(?:\.\d+)?)(?:st|nd|rd|th)?(?!\w)",
    re.IGNORECASE,
)
_QUANTITY_AXIS_CUE_RE = re.compile(
    r"\b(?:count|counting|start|starting|step|position|index|offset|"
    r"interval|spacing|every)\b|计数|起点|第几|间隔|周期|偏移",
    re.IGNORECASE,
)


def build_plain_reading_review(
    *,
    requirement_id: str,
    segmentation: SurfaceSegmentation,
    ambiguities: Sequence[PlainReadingAmbiguity] = (),
    overall_status: str | None = None,
    review_basis: str = "",
    reviewer_source: str = "",
    generator: ArtifactGenerator | None = None,
    experiment_mode: str = "off",
    data_split: str = "test",
    review_history: Sequence[Mapping[str, Any]] = (),
    revision_requests: Sequence[PlainReadingRevisionRequest] = (),
    no_ambiguity_confirmation: NoAmbiguityConfirmation | None = None,
    ambiguity_generation_trigger: AmbiguityGenerationTrigger | None = None,
    downstream_status: str = "current",
    supersedes_reading_review_id: str = "",
) -> PlainReadingReview:
    """Build and validate a review linked to one reviewed segmentation."""

    assert_surface_segmentation_gate(segmentation)
    ambiguities = tuple(ambiguities)
    if overall_status is None:
        overall_status = "pending_plain_review" if ambiguities else "no_ambiguity"
    fingerprint = _reading_content_fingerprint(
        segmentation.segmentation_id,
        ambiguities,
        overall_status,
        revision_requests=revision_requests,
        no_ambiguity_confirmation=no_ambiguity_confirmation,
        ambiguity_generation_trigger=ambiguity_generation_trigger,
    )
    now = _timestamp()
    review = PlainReadingReview(
        reading_review_id=f"READ-{requirement_id}-v5-{fingerprint}",
        requirement_id=requirement_id,
        source_hash=segmentation.source_hash,
        segmentation_id=segmentation.segmentation_id,
        ambiguities=ambiguities,
        revision_requests=tuple(revision_requests),
        no_ambiguity_confirmation=(
            no_ambiguity_confirmation or NoAmbiguityConfirmation()
        ),
        ambiguity_generation_trigger=(
            ambiguity_generation_trigger or AmbiguityGenerationTrigger()
        ),
        overall_status=overall_status,
        review_basis=review_basis,
        reviewer_source=reviewer_source,
        generator=generator or ArtifactGenerator(),
        experiment_mode=experiment_mode,
        data_split=data_split,
        persistent_rag_allowed=(
            data_split == "train"
            and overall_status in ALLOWED_REVIEWED_STATUSES
            and reviewer_source in {"human", "independent_judge", "warmup_fixture"}
        ),
        created_at=now,
        updated_at=now,
        review_history=tuple(dict(item) for item in review_history),
        downstream_status=downstream_status,
        supersedes_reading_review_id=supersedes_reading_review_id,
    )
    validate_plain_reading_review(review, segmentation)
    return review


def validate_plain_reading_review(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    require_reviewed: bool = False,
    raise_on_error: bool = True,
) -> tuple[str, ...] | None:
    """Validate a review, or collect all errors when raising is disabled."""

    assert_surface_segmentation_gate(segmentation)
    errors: list[str] = []
    if review.requirement_id != segmentation.requirement_id:
        errors.append("requirement_id 与 SurfaceSegmentation 不一致")
    if review.source_hash != segmentation.source_hash:
        errors.append("source_hash 与 SurfaceSegmentation 不一致")
    if review.source_hash != source_hash(segmentation.source_text):
        errors.append("source_hash 与当前原文不一致")
    if review.segmentation_id != segmentation.segmentation_id:
        errors.append("segmentation_id 与当前审核切分不一致")
    segments_by_id = {
        segment.segment_id: segment for segment in segmentation.segments
    }
    known_segments = set(segments_by_id)
    ambiguity_ids: set[str] = set()
    for ambiguity in review.ambiguities:
        if not ambiguity.ambiguity_id or ambiguity.ambiguity_id in ambiguity_ids:
            errors.append("ambiguity_id 为空或重复")
        ambiguity_ids.add(ambiguity.ambiguity_id)
        if not ambiguity.source_segment_ids:
            errors.append(f"{ambiguity.ambiguity_id} 缺少 source_segment_ids")
        unknown = set(ambiguity.source_segment_ids) - known_segments
        if unknown:
            errors.append(
                f"{ambiguity.ambiguity_id} 引用未知 segment_id: "
                + ", ".join(sorted(unknown))
            )
        legacy_evidence_migration = any(
            item.get("operation") == "schema_migration_read_only"
            and str(item.get("from_schema_version", "")) in {"1", "2", "3"}
            for item in review.review_history
        )
        evidence_required = (
            review.generator.mode == "api"
            and review.review_basis == "agent_detection"
            and not legacy_evidence_migration
        )
        start, end = ambiguity.source_span
        has_evidence = start >= 0 or end >= 0 or bool(ambiguity.evidence_text)
        if evidence_required and not has_evidence:
            errors.append(f"{ambiguity.ambiguity_id} 缺少精确 source_span/evidence_text")
        if has_evidence:
            if not (0 <= start < end <= len(segmentation.source_text)):
                errors.append(f"{ambiguity.ambiguity_id} source_span 非法")
            elif segmentation.source_text[start:end] != ambiguity.evidence_text:
                errors.append(
                    f"{ambiguity.ambiguity_id} evidence_text 与 source_span 不一致"
                )
            else:
                cited_segments = [
                    segments_by_id[segment_id]
                    for segment_id in ambiguity.source_segment_ids
                    if segment_id in segments_by_id
                ]
                if not _cited_segments_contiguously_cover_span(
                    cited_segments,
                    start=start,
                    end=end,
                ):
                    errors.append(
                        f"{ambiguity.ambiguity_id} source_span 未被引用的连续 segments 完整覆盖"
                    )
        if len(ambiguity.options) != 2:
            errors.append(f"{ambiguity.ambiguity_id} 必须恰好包含两个读法选项")
        option_ids = [option.option_id for option in ambiguity.options]
        if any(not option_id for option_id in option_ids) or len(option_ids) != len(
            set(option_ids)
        ):
            errors.append(f"{ambiguity.ambiguity_id} option_id 为空或重复")
        normalized = [
            _normalize_option(option.plain_interpretation)
            for option in ambiguity.options
        ]
        if any(not value for value in normalized):
            errors.append(f"{ambiguity.ambiguity_id} 存在空解释")
        if len(set(normalized)) != len(normalized):
            errors.append(f"{ambiguity.ambiguity_id} 的两个解释规范化后相同")
        if any(not option.difference_summary.strip() for option in ambiguity.options):
            errors.append(f"{ambiguity.ambiguity_id} 缺少 difference_summary")
        if ambiguity.decision_status == "reviewer_selected":
            if ambiguity.decision not in option_ids:
                errors.append(f"{ambiguity.ambiguity_id} 选择项不属于当前 options")
            if not (
                ambiguity.review_basis or review.review_basis
            ) or not (
                ambiguity.reviewer_source or review.reviewer_source
            ):
                errors.append(
                    f"{ambiguity.ambiguity_id} 已选择结果缺少 review_basis/reviewer_source"
                )
        elif ambiguity.decision_status in {
            "ambiguous_unresolved",
            "neither_option_selected",
            "pending_plain_review",
            "options_revision_requested",
        }:
            if ambiguity.decision:
                errors.append(
                    f"{ambiguity.ambiguity_id} 的 {ambiguity.decision_status} 不得绑定选项"
                )
        else:
            errors.append(
                f"{ambiguity.ambiguity_id} 不支持 decision_status={ambiguity.decision_status}"
            )

    empty_axis_statuses = {
        "no_ambiguity",
        "pending_no_ambiguity_confirmation",
        "awaiting_human_semantic_override",
        "user_requested_ambiguity_generation",
        "needs_source_clarification",
    }
    if not review.ambiguities and review.overall_status not in empty_axis_statuses:
        errors.append(
            "无 ambiguity 时 overall_status 必须是 no_ambiguity、"
            "pending_no_ambiguity_confirmation 或 user_requested_ambiguity_generation"
        )
    if review.ambiguities and review.overall_status == "no_ambiguity":
        errors.append("存在 ambiguity 时不得标记 no_ambiguity")
    confirmation = review.no_ambiguity_confirmation
    trigger = review.ambiguity_generation_trigger
    judge_evidence_valid = _valid_no_ambiguity_judge_evidence(review, segmentation)
    if confirmation.judge_evidence_json and not judge_evidence_valid:
        errors.append("Judge no-ambiguity evidence is malformed, stale or unbound")
    if review.overall_status == "pending_no_ambiguity_confirmation":
        if review.ambiguities:
            errors.append("待确认无歧义状态不得包含 ambiguity")
        human_axis_rejection = (
            review.review_basis == "human_rejected_all_ambiguity_axes"
        )
        if (
            (not confirmation.llm_proposed_no_ambiguity and not human_axis_rejection)
            or confirmation.status != "pending"
        ):
            errors.append("待确认无歧义状态缺少 LLM 空轴提议记录")
    if review.overall_status == "user_requested_ambiguity_generation":
        if review.ambiguities:
            errors.append("等待强制生成歧义轴时不得保留旧 ambiguity")
        if confirmation.status != "rejected" or not confirmation.human_confirmed:
            errors.append("强制生成歧义轴必须来自人工否决无歧义结论")
        if trigger.source != "human_disagreement":
            errors.append("强制生成歧义轴缺少 human_disagreement trigger")
        if trigger.previous_reading_review_id != review.supersedes_reading_review_id:
            errors.append("强制生成 trigger 未绑定被否决的 reading review")
    if review.overall_status == "awaiting_human_semantic_override":
        if review.ambiguities or review.revision_requests:
            errors.append("等待完整语义替换时不得保留旧 ambiguity 或 revision request")
        if confirmation.status != "rejected" or not confirmation.human_confirmed:
            errors.append("完整语义替换必须来自普通审核者否决 fallback 结论")
        if review.review_basis != "human_requested_semantic_override":
            errors.append("完整语义替换缺少专用 review_basis")
    llm_generated_empty = (
        not review.ambiguities
        and review.generator.mode == "api"
        and review.review_basis == "agent_detection"
    )
    if review.overall_status == "no_ambiguity" and (
        confirmation.llm_proposed_no_ambiguity or llm_generated_empty
    ):
        if confirmation.status not in {"confirmed", "withdrawn"}:
            errors.append("LLM 空轴结论未经人工确认不得进入 no_ambiguity")
        if not confirmation.human_confirmed and not (confirmation.judge_confirmed and judge_evidence_valid):
            errors.append("LLM 空轴结论缺少人工或有效 Judge 确认")
    if trigger.suspected_segment_ids:
        unknown = set(trigger.suspected_segment_ids) - known_segments
        if unknown:
            errors.append(
                "ambiguity generation trigger 引用未知 segment_id: "
                + ", ".join(sorted(unknown))
            )
    if review.overall_status == "plain_reading_reviewed" and any(
        ambiguity.decision_status != "reviewer_selected"
        for ambiguity in review.ambiguities
    ):
        errors.append("plain_reading_reviewed 要求每个歧义轴都有审核选择")
    if review.overall_status == "ambiguous_unresolved" and not any(
        ambiguity.decision_status == "ambiguous_unresolved"
        for ambiguity in review.ambiguities
    ):
        errors.append("ambiguous_unresolved 必须至少保留一个未决歧义轴")
    if review.overall_status == "needs_source_clarification" and not any(
        ambiguity.decision_status == "neither_option_selected"
        for ambiguity in review.ambiguities
    ):
        if not (judge_evidence_valid and review.review_basis == "judge_requires_source_clarification") and not is_post_independent_revision_all_axes_discarded(review) and not (
            not review.ambiguities
            and review.review_basis == "forced_ambiguity_generation_exhausted"
            and trigger.source == "human_disagreement"
            and trigger.generation_attempts >= 1
            and trigger.last_errors
        ):
            errors.append(
                "needs_source_clarification 必须来自两项都不符合或强制生成耗尽"
            )
    if review.overall_status == "options_revision_requested" and not any(
        ambiguity.decision_status == "options_revision_requested"
        for ambiguity in review.ambiguities
    ):
        errors.append("options_revision_requested 必须绑定待修订歧义轴")
    active_request_statuses = {"submitted", "suggested", "confirmed"}
    revision_requests_by_ambiguity: dict[str, list[PlainReadingRevisionRequest]] = {}
    active_requests_by_ambiguity: dict[str, list[PlainReadingRevisionRequest]] = {}
    for request in review.revision_requests:
        revision_requests_by_ambiguity.setdefault(request.ambiguity_id, []).append(request)
        if request.request_status in active_request_statuses:
            active_requests_by_ambiguity.setdefault(request.ambiguity_id, []).append(request)
    ambiguity_statuses = {
        ambiguity.ambiguity_id: ambiguity.decision_status
        for ambiguity in review.ambiguities
    }
    for ambiguity in review.ambiguities:
        if ambiguity.decision_status != "options_revision_requested":
            continue
        requests_for_axis = revision_requests_by_ambiguity.get(
            ambiguity.ambiguity_id, []
        )
        active_for_axis = active_requests_by_ambiguity.get(ambiguity.ambiguity_id, [])
        if not requests_for_axis:
            errors.append(
                f"{ambiguity.ambiguity_id} options_revision_requested "
                "缺少 PlainReadingRevisionRequest"
            )
        elif not active_for_axis:
            terminal_statuses = ", ".join(
                sorted({request.request_status for request in requests_for_axis})
            )
            errors.append(
                f"{ambiguity.ambiguity_id} options_revision_requested "
                "没有活动的 PlainReadingRevisionRequest"
                f"（现有状态: {terminal_statuses}）"
            )
        elif len(active_for_axis) > 1:
            errors.append(
                f"{ambiguity.ambiguity_id} options_revision_requested "
                "必须恰好对应一个活动 PlainReadingRevisionRequest"
            )
    for ambiguity_id, requests in active_requests_by_ambiguity.items():
        if ambiguity_statuses.get(ambiguity_id) != "options_revision_requested":
            request_ids = ", ".join(
                sorted(request.request_id for request in requests if request.request_id)
            )
            errors.append(
                f"{ambiguity_id} 的活动 PlainReadingRevisionRequest"
                f" 未绑定 options_revision_requested 歧义轴: {request_ids}"
            )
    request_ids: set[str] = set()
    historical_ambiguity_ids = {
        str(item.get("ambiguity_id", ""))
        for item in review.review_history
        if item.get("operation") == "plain_reading_revision_applied"
    } - {""}
    historical_ambiguity_ids.update(
        str(ambiguity_id)
        for item in review.review_history
        for ambiguity_id in item.get("old_ambiguity_ids", ())
        if str(ambiguity_id)
    )
    allowed_request_parent_ids = {
        review.reading_review_id,
        review.supersedes_reading_review_id,
        *(
            str(item.get("old_reading_review_id", ""))
            for item in review.review_history
        ),
    } - {""}
    for request in review.revision_requests:
        if not request.request_id or request.request_id in request_ids:
            errors.append("revision request_id 为空或重复")
        request_ids.add(request.request_id)
        if request.reading_review_id not in allowed_request_parent_ids:
            errors.append(f"{request.request_id} reading_review_id 不属于当前修订链")
        if (
            request.ambiguity_id not in ambiguity_ids
            and request.ambiguity_id not in historical_ambiguity_ids
        ):
            errors.append(f"{request.request_id} 引用未知 ambiguity_id")
        if request.segmentation_id != review.segmentation_id:
            errors.append(f"{request.request_id} segmentation_id 与当前 review 不一致")
        if request.source_hash != review.source_hash:
            errors.append(f"{request.request_id} source_hash 与当前 review 不一致")
        if not request.source_segment_ids:
            errors.append(f"{request.request_id} 缺少 source_segment_ids")
        unknown = set(request.source_segment_ids) - known_segments
        if unknown:
            errors.append(
                f"{request.request_id} 引用未知 segment_id: "
                + ", ".join(sorted(unknown))
            )
        if not request.proposed_interpretation.strip():
            errors.append(f"{request.request_id} 补充解释为空")
        if not request.reason.strip():
            errors.append(f"{request.request_id} 缺少补充理由")
        if request.validation_status == "accepted" and not request.human_confirmed:
            errors.append(f"{request.request_id} 未经人工确认不得 accepted")
        if request.request_status in {"submitted", "suggested", "confirmed"} and (
            request.ambiguity_id not in ambiguity_ids
        ):
            errors.append(f"{request.request_id} 活动请求引用非当前 ambiguity_id")
    scope_applications, completed_scope_ids, scope_errors = (
        _post_independent_revision_scope_records(review)
    )
    errors.extend(scope_errors)
    requests_by_id = {request.request_id: request for request in review.revision_requests}
    completions_by_scope: dict[str, Mapping[str, Any]] = {}
    for item in review.review_history:
        if item.get("operation") == POST_INDEPENDENT_REVISION_REVIEW_OPERATION:
            scope_id = str(item.get("scope_id", ""))
            if scope_id:
                completions_by_scope[scope_id] = item
    for scope_id, application in scope_applications.items():
        axis_ids = tuple(application.get("post_revision_review_axis_ids", ()))
        original_axis_id = str(application.get("original_ambiguity_id", ""))
        new_axis_ids = tuple(
            str(axis_id)
            for axis_id in application.get("new_axis_ids", ())
            if str(axis_id)
        )
        request = requests_by_id.get(scope_id)
        if request is None:
            errors.append(f"post revision review scope {scope_id} 缺少 revision request")
        elif (
            request.classification != "new_independent_axis"
            or request.request_status != "applied"
        ):
            errors.append(
                f"post revision review scope {scope_id} 未绑定已应用的 independent request"
            )
        if application.get("post_revision_review_source_hash") != review.source_hash:
            errors.append(f"post revision review scope {scope_id} source_hash 不一致")
        if (
            application.get("post_revision_review_segmentation_id")
            != review.segmentation_id
        ):
            errors.append(f"post revision review scope {scope_id} segmentation_id 不一致")
        if not original_axis_id or original_axis_id not in axis_ids:
            errors.append(f"post revision review scope {scope_id} 缺少原轴")
        if not new_axis_ids or not set(new_axis_ids).issubset(axis_ids):
            errors.append(f"post revision review scope {scope_id} 缺少新增轴 IDs")
        if scope_id not in completed_scope_ids:
            missing_active_ids = set(axis_ids) - ambiguity_ids
            if missing_active_ids:
                errors.append(
                    f"活动 post revision review scope {scope_id} 缺少轴: "
                    + ", ".join(sorted(missing_active_ids))
                )
            continue
        completion = completions_by_scope.get(scope_id, {})
        discarded_ids = tuple(
            str(axis_id)
            for axis_id in completion.get("discarded_ambiguity_ids", ())
            if str(axis_id)
        )
        retained_ids = tuple(
            str(axis_id)
            for axis_id in completion.get("retained_ambiguity_ids", ())
            if str(axis_id)
        )
        if len(discarded_ids) != len(set(discarded_ids)):
            errors.append(f"post revision review scope {scope_id} 丢弃轴 IDs 重复")
        if len(retained_ids) != len(set(retained_ids)):
            errors.append(f"post revision review scope {scope_id} 保留轴 IDs 重复")
        if set(discarded_ids) & set(retained_ids):
            errors.append(f"post revision review scope {scope_id} 保留/丢弃轴重叠")
        if set(discarded_ids) | set(retained_ids) != set(axis_ids):
            errors.append(f"post revision review scope {scope_id} 未完整覆盖复审轴")
        decision_ids = {
            str(axis_id) for axis_id in completion.get("decisions", {}).keys()
        } if isinstance(completion.get("decisions", {}), Mapping) else set()
        if decision_ids != set(axis_ids):
            errors.append(f"post revision review scope {scope_id} decisions 未覆盖全部轴")
        all_axes_discarded = bool(completion.get("all_axes_discarded"))
        if all_axes_discarded != (not retained_ids and bool(discarded_ids)):
            errors.append(f"post revision review scope {scope_id} 全轴丢弃标记不一致")
    if is_post_independent_revision_all_axes_discarded(review):
        latest_completion = next(
            (
                item
                for item in reversed(review.review_history)
                if item.get("operation")
                == POST_INDEPENDENT_REVISION_REVIEW_OPERATION
            ),
            {},
        )
        if latest_completion.get("retained_ambiguity_ids"):
            errors.append("全轴丢弃澄清终态不得记录保留轴")
        if not latest_completion.get("discarded_ambiguity_ids"):
            errors.append("全轴丢弃澄清终态缺少 discarded_ambiguity_ids")
    if require_reviewed and review.overall_status not in ALLOWED_REVIEWED_STATUSES:
        errors.append("Plain Reading 尚未通过进入 IR 所需的门禁")
    unique_errors = tuple(dict.fromkeys(errors))
    if unique_errors and raise_on_error:
        raise PlainReadingValidationError(
            "；".join(unique_errors),
            errors=unique_errors,
        )
    return None if raise_on_error else unique_errors


def _cited_segments_contiguously_cover_span(
    cited_segments: Sequence[SurfaceSegment],
    *,
    start: int,
    end: int,
) -> bool:
    """Return whether exactly one contiguous cited run supports ``[start, end)``.

    Surface segmentation is lossless and gap-free.  Requiring the cited
    segments to be consecutive therefore permits relational evidence that
    crosses a reading boundary without allowing a citation to skip unsupported
    source text.  The first and last cited segments must both overlap the span;
    this also rejects padding a valid citation with unrelated adjacent segments.
    """

    ordered = list(cited_segments)
    if not ordered:
        return False
    if any(
        right.index != left.index + 1 or right.start != left.end
        for left, right in zip(ordered, ordered[1:])
    ):
        return False
    first = ordered[0]
    last = ordered[-1]
    return (
        first.start <= start < first.end
        and last.start < end <= last.end
    )


def assert_plain_reading_gate(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
) -> None:
    validate_plain_reading_review(review, segmentation, require_reviewed=True)


def assert_plain_reading_review_well_formed(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
) -> None:
    validate_plain_reading_review(review, segmentation, require_reviewed=False)


def assess_plain_reading_ir_readiness(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
) -> PlainReadingIRReadiness:
    """Return the authoritative, side-effect-free Plain Reading routing result."""

    validation_errors = tuple(
        validate_plain_reading_review(
            review,
            segmentation,
            require_reviewed=False,
            raise_on_error=False,
        )
    )
    common = {
        "unresolved_ambiguity_ids": tuple(review.unresolved_ambiguity_ids),
        "pending_revision_request_ids": tuple(
            review.pending_revision_request_ids
        ),
        "suggested_rerun_stage": "plain_reading",
        "reading_review_id": review.reading_review_id,
        "decision_fingerprint": decision_fingerprint(review),
        "validation_errors": validation_errors,
    }
    if validation_errors:
        return PlainReadingIRReadiness(
            disposition="invalid",
            can_enter_semantic_ir=False,
            reason_code="PLAIN_READING_ARTIFACT_INVALID",
            **common,
        )
    if review.overall_status in {"no_ambiguity", "plain_reading_reviewed"}:
        return PlainReadingIRReadiness(
            disposition="ready",
            can_enter_semantic_ir=True,
            reason_code="PLAIN_READING_READY",
            **common,
        )
    if review.overall_status == "ambiguous_unresolved":
        return PlainReadingIRReadiness(
            disposition="semantic_resolution_required",
            can_enter_semantic_ir=False,
            reason_code="PLAIN_READING_SEMANTIC_RESOLUTION_REQUIRED",
            **common,
        )
    if review.overall_status == "needs_source_clarification":
        return PlainReadingIRReadiness(
            disposition="source_clarification_required",
            can_enter_semantic_ir=False,
            reason_code="PLAIN_READING_SOURCE_CLARIFICATION_REQUIRED",
            **common,
        )
    return PlainReadingIRReadiness(
        disposition="human_action_required",
        can_enter_semantic_ir=False,
        reason_code="PLAIN_READING_HUMAN_ACTION_REQUIRED",
        **common,
    )


def assert_plain_reading_ready_for_semantic_ir(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
) -> None:
    readiness = assess_plain_reading_ir_readiness(review, segmentation)
    if readiness.can_enter_semantic_ir:
        return
    if readiness.disposition == "semantic_resolution_required":
        raise PlainReadingSemanticResolutionRequired(review, readiness, segmentation=segmentation)
    if readiness.disposition == "invalid":
        raise PlainReadingValidationError(
            "；".join(readiness.validation_errors),
            errors=readiness.validation_errors,
        )
    raise PlainReadingValidationError(
        f"{readiness.reason_code}: Plain Reading cannot enter Semantic IR"
    )


def assert_plain_reading_ready_for_candidate_ir(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    mapped_axes: Mapping[str, Mapping[str, object]] | None = None,
    max_candidates: int = 5,
) -> None:
    """Admit unresolved readings only with a complete executable choice map.

    A plain-language option alone is not a semantic branch.  The caller must
    bind every unresolved option to a typed IR operation before constructing
    candidate IRs.  This gate never grants finalization authority.
    """
    readiness = assess_plain_reading_ir_readiness(review, segmentation)
    if readiness.can_enter_semantic_ir:
        return
    if readiness.disposition != "semantic_resolution_required":
        assert_plain_reading_ready_for_semantic_ir(review, segmentation)
    axes = tuple(axis for axis in review.ambiguities
                 if axis.decision_status == "ambiguous_unresolved")
    mappings = mapped_axes or {}
    if set(mappings) != {axis.ambiguity_id for axis in axes}:
        raise PlainReadingValidationError("candidate IR requires every unresolved axis")
    count = 1
    for axis in axes:
        start, end = axis.source_span
        if not (0 <= start < end <= len(segmentation.source_text)
                and segmentation.source_text[start:end] == axis.evidence_text):
            raise PlainReadingValidationError("candidate axis source span is not exact")
        option_ids = {option.option_id for option in axis.options}
        choices = mappings[axis.ambiguity_id]
        if set(choices) != option_ids or len(option_ids) < 2:
            raise PlainReadingValidationError("candidate axis option map is incomplete")
        if any(not isinstance(choice, Mapping)
               or choice.get("operation") not in {"swap_children", "replace_parameter", "choose_policy_branch", "replace_node"}
               or not choice.get("node_id")
               for choice in choices.values()):
            raise PlainReadingValidationError("candidate option lacks typed IR authority")
        count *= len(option_ids)
    if count > max_candidates:
        raise PlainReadingValidationError("candidate Cartesian product exceeds frozen bound")


def assert_plain_reading_ready_for_finalization(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
) -> None:
    """Keep final approval behind the existing unique-reading gate."""
    assert_plain_reading_ready_for_semantic_ir(review, segmentation)


def confirm_no_ambiguity(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    reviewer_id: str = "",
    reviewer_role: str = "ordinary_reviewer",
    reviewer_source: str = "human",
    comment: str = "",
) -> PlainReadingReview:
    """Accept an LLM-proposed empty ambiguity set with explicit human lineage."""

    validate_plain_reading_review(review, segmentation)
    if review.overall_status != "pending_no_ambiguity_confirmation":
        raise PlainReadingValidationError("只有待确认的 LLM 空轴结论可以确认")
    completed_at = _timestamp()
    confirmation = NoAmbiguityConfirmation(
        status="confirmed",
        llm_proposed_no_ambiguity=True,
        human_confirmed=True,
        reviewer_id=reviewer_id,
        reviewer_role=reviewer_role,
        reviewer_source=reviewer_source,
        comment=comment,
        confirmed_at=completed_at,
    )
    return build_plain_reading_review(
        requirement_id=review.requirement_id,
        segmentation=segmentation,
        ambiguities=(),
        overall_status="no_ambiguity",
        review_basis="human_confirmed_no_ambiguity",
        reviewer_source=reviewer_source,
        generator=review.generator,
        experiment_mode=review.experiment_mode,
        data_split=review.data_split,
        review_history=(
            *review.review_history,
            {
                "round": len(review.review_history) + 1,
                "operation": "no_ambiguity_human_confirmed",
                "old_reading_review_id": review.reading_review_id,
                "reviewer_id": reviewer_id,
                "reviewer_role": reviewer_role,
                "reviewer_source": reviewer_source,
                "comment": comment,
                "completed_at": completed_at,
            },
        ),
        revision_requests=review.revision_requests,
        no_ambiguity_confirmation=confirmation,
        ambiguity_generation_trigger=review.ambiguity_generation_trigger,
        downstream_status=review.downstream_status,
        supersedes_reading_review_id=review.reading_review_id,
    )


def _judge_evidence_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _valid_no_ambiguity_judge_evidence(review, segmentation):
    c = review.no_ambiguity_confirmation
    if not c.judge_evidence_json:
        return False
    try:
        evidence = json.loads(c.judge_evidence_json)
        request, response = evidence["request"], evidence["response"]
        payload = request["payload"]
        decision = response["decision"]
        from source_evidence_views import validate_source_quotes
        valid_quotes = validate_source_quotes(
            response.get("source_quotes"), segmentation.source_text
        )
        return (
            evidence["version"] == ("no-ambiguity-model-assistant-v1" if review.reviewer_source == "model_assistant" else "no-ambiguity-judge-v1")
            and bool(evidence["judge_model"]) and bool(request["system"])
            and evidence["request_hash"] == _judge_evidence_hash(request)
            and evidence["response_hash"] == _judge_evidence_hash(response)
            and payload["source_text"] == segmentation.source_text
            and payload["source_hash"] == review.source_hash == segmentation.source_hash
            and payload["segmentation_id"] == review.segmentation_id == segmentation.segmentation_id
            and payload["proposal_id"] == review.supersedes_reading_review_id
            and isinstance(response["reason"], str) and bool(response["reason"].strip())
            and bool(valid_quotes)
            and review.reviewer_source == c.reviewer_source
            and review.reviewer_source in {"independent_judge", "model_assistant"}
            and not c.human_confirmed
            and ((decision == "confirm_no_ambiguity" and c.judge_confirmed and c.status == "confirmed"
                  and review.overall_status == "no_ambiguity" and review.review_basis == "judge_confirmed_no_ambiguity")
                 or (decision == "ambiguity_found" and not c.judge_confirmed
                     and c.status == "pending" and bool(review.ambiguities)
                     and review.overall_status == "ambiguous_unresolved"
                     and review.review_basis == "judge_detected_source_ambiguity")
                 or (decision in ("ambiguity_found", "needs_source_clarification") and not c.judge_confirmed
                     and c.status == "pending" and not review.ambiguities
                     and review.overall_status == "needs_source_clarification"
                     and review.review_basis == "judge_requires_source_clarification"))
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def apply_no_ambiguity_judge_decision(review, segmentation, *, request, response, judge_model, physical_evidence=None, reviewer_source="independent_judge"):
    """Record an actual second-stage decision, with no human approval claim."""
    validate_plain_reading_review(review, segmentation)
    if review.overall_status != "pending_no_ambiguity_confirmation":
        raise PlainReadingValidationError("Judge confirmation requires a pending empty-axis proposal")
    if reviewer_source not in {"independent_judge", "model_assistant"}:
        raise PlainReadingValidationError("Unsupported model review source")
    evidence = {"version": "no-ambiguity-model-assistant-v1" if reviewer_source == "model_assistant" else "no-ambiguity-judge-v1", "request": request, "response": response,
                "request_hash": _judge_evidence_hash(request), "response_hash": _judge_evidence_hash(response),
                "judge_model": judge_model, "physical_evidence": physical_evidence or {}}
    decision = response.get("decision")
    confirmed = decision == "confirm_no_ambiguity"
    ambiguities = (
        _ambiguities_from_no_ambiguity_judge_response(
            response, segmentation, reviewer_source=reviewer_source, reviewer_id=judge_model,
        )
        if decision == "ambiguity_found" and response.get("ambiguities")
        else ()
    )
    found_supported_axes = decision == "ambiguity_found" and bool(ambiguities)
    confirmation = NoAmbiguityConfirmation(status="confirmed" if confirmed else "pending",
        llm_proposed_no_ambiguity=True, human_confirmed=False, judge_confirmed=confirmed,
        judge_evidence_json=json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        reviewer_id=judge_model, reviewer_role="model_assistant" if reviewer_source == "model_assistant" else "model_reviewer", reviewer_source=reviewer_source,
        comment=response.get("reason", ""), confirmed_at=_timestamp())
    return build_plain_reading_review(requirement_id=review.requirement_id, segmentation=segmentation,
        ambiguities=ambiguities,
        overall_status=(
            "no_ambiguity" if confirmed
            else "ambiguous_unresolved" if found_supported_axes
            else "needs_source_clarification"
        ),
        review_basis=(
            "judge_confirmed_no_ambiguity" if confirmed
            else "judge_detected_source_ambiguity" if found_supported_axes
            else "judge_requires_source_clarification"
        ),
        reviewer_source=reviewer_source, generator=review.generator, experiment_mode=review.experiment_mode,
        data_split=review.data_split, no_ambiguity_confirmation=confirmation,
        review_history=(*review.review_history, {"operation": "no_ambiguity_model_assistant_decision" if reviewer_source == "model_assistant" else "no_ambiguity_judge_decision", "decision": response.get("decision"),
            "request_hash": evidence["request_hash"], "response_hash": evidence["response_hash"], "judge_model": judge_model}),
        revision_requests=review.revision_requests, ambiguity_generation_trigger=review.ambiguity_generation_trigger,
        downstream_status=review.downstream_status, supersedes_reading_review_id=review.reading_review_id)


def _ambiguities_from_no_ambiguity_judge_response(
    response, segmentation, *, reviewer_source="independent_judge", reviewer_id="",
):
    """Convert source-bound alternatives while preserving the actual reviewer."""

    payloads = response.get("ambiguities")
    if not isinstance(payloads, list) or not payloads:
        raise PlainReadingValidationError(
            "ambiguity_found requires at least one structured ambiguity axis"
        )
    result = []
    seen = set()
    for index, payload in enumerate(payloads, start=1):
        if not isinstance(payload, Mapping):
            raise PlainReadingValidationError("Judge ambiguity axis must be an object")
        quote = payload.get("source_quote")
        if not isinstance(quote, Mapping):
            raise PlainReadingValidationError("Judge ambiguity axis lacks source_quote")
        start, end = quote.get("start"), quote.get("end")
        text = quote.get("text")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(segmentation.source_text)
            or segmentation.source_text[start:end] != text
        ):
            raise PlainReadingValidationError("Judge ambiguity axis quote is not exact")
        options = payload.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise PlainReadingValidationError(
                "Judge ambiguity axis requires at least two source-supported readings"
            )
        # The current PlainReading contract is binary. More than two readings
        # are not truncated because that would falsely claim completeness.
        if len(options) != 2:
            raise PlainReadingValidationError(
                "Judge ambiguity axis exceeds the binary PlainReading contract"
            )
        option_records = []
        normalized = set()
        for option_index, option in enumerate(options, start=1):
            if not isinstance(option, Mapping):
                raise PlainReadingValidationError("Judge ambiguity option must be an object")
            interpretation = str(option.get("plain_interpretation", "")).strip()
            difference = str(option.get("difference_summary", "")).strip()
            if not interpretation or not difference:
                raise PlainReadingValidationError("Judge ambiguity option is incomplete")
            key = _normalize_option(interpretation)
            if key in normalized:
                raise PlainReadingValidationError("Judge ambiguity readings are duplicates")
            normalized.add(key)
            option_records.append(
                PlainReadingOption(
                    option_id=f"JUDGE-{index}-OPTION-{option_index}",
                    plain_interpretation=interpretation,
                    difference_summary=difference,
                )
            )
        segment_ids = tuple(
            segment.segment_id
            for segment in segmentation.segments
            if start < segment.end and end > segment.start
        )
        if not segment_ids:
            raise PlainReadingValidationError("Judge ambiguity quote is outside segmentation")
        digest = hashlib.sha256(
            json.dumps(
                [segmentation.source_hash, start, end, [o.plain_interpretation for o in option_records]],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        ambiguity_id = "JUDGE-AXIS-" + digest
        if ambiguity_id in seen:
            raise PlainReadingValidationError("Judge returned duplicate ambiguity axes")
        seen.add(ambiguity_id)
        result.append(
            PlainReadingAmbiguity(
                ambiguity_id=ambiguity_id,
                source_segment_ids=segment_ids,
                question=str(payload.get("question", "")).strip(),
                options=tuple(option_records),
                source_span=(start, end),
                evidence_text=text,
                decision_status="ambiguous_unresolved",
                review_basis="source_bound_model_axis",
                reviewer_id=reviewer_id,
                reviewer_role="model_assistant" if reviewer_source == "model_assistant" else "model_reviewer",
                reviewer_source=reviewer_source,
                comments=str(response.get("reason", "")),
            )
        )
    if any(not item.question for item in result):
        raise PlainReadingValidationError("Judge ambiguity axis question is empty")
    return tuple(result)


def build_plain_reading_semantic_fallback(
    *,
    requirement_id: str,
    segmentation: SurfaceSegmentation,
    generator: ArtifactGenerator,
    experiment_mode: str,
    data_split: str,
    reason_codes: Sequence[str],
    attempts: int,
    diagnostics: Mapping[str, Any],
    previous_review: PlainReadingReview | None = None,
) -> PlainReadingReview:
    """Discard rejected axes and expose only an ordinary no-ambiguity choice."""

    codes = tuple(dict.fromkeys(str(code) for code in reason_codes if str(code)))
    previous_history = previous_review.review_history if previous_review else ()
    return build_plain_reading_review(
        requirement_id=requirement_id,
        segmentation=segmentation,
        ambiguities=(),
        overall_status="pending_no_ambiguity_confirmation",
        review_basis="llm_no_source_supported_axis_after_repair",
        reviewer_source="",
        generator=generator,
        experiment_mode=experiment_mode,
        data_split=data_split,
        no_ambiguity_confirmation=NoAmbiguityConfirmation(
            status="pending",
            llm_proposed_no_ambiguity=True,
        ),
        review_history=(
            *previous_history,
            {
                "round": len(previous_history) + 1,
                "operation": "plain_reading_semantic_fallback",
                "semantic_fallback": True,
                "semantic_fallback_issue_codes": codes,
                "semantic_fallback_attempts": attempts,
                "last_rejected_candidate": dict(
                    diagnostics.get("last_rejected_candidate", {})
                ),
                "rollback_reason": str(diagnostics.get("reason_code", "")),
                "source_hash": segmentation.source_hash,
                "prompt_version": generator.prompt_version,
                "validator_version": str(diagnostics.get("validator_version", "")),
                "diagnostics": dict(diagnostics),
                "completed_at": _timestamp(),
            },
        ),
        ambiguity_generation_trigger=(
            previous_review.ambiguity_generation_trigger
            if previous_review is not None
            else None
        ),
        downstream_status=(
            previous_review.downstream_status
            if previous_review is not None
            else "current"
        ),
        supersedes_reading_review_id=(
            previous_review.reading_review_id
            if previous_review is not None
            else ""
        ),
    )


def request_human_semantic_override(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    reviewer_id: str = "",
    reviewer_source: str = "human",
    comment: str = "",
) -> PlainReadingReview:
    """Record R for a semantic fallback without forcing another LLM axis."""

    validate_plain_reading_review(review, segmentation)
    if (
        review.overall_status != "pending_no_ambiguity_confirmation"
        or review.review_basis != "llm_no_source_supported_axis_after_repair"
    ):
        raise PlainReadingValidationError(
            "只有两轮语义失败后的空轴结论可以进入完整语义替换"
        )
    requested_at = _timestamp()
    confirmation = NoAmbiguityConfirmation(
        status="rejected",
        llm_proposed_no_ambiguity=True,
        human_confirmed=True,
        reviewer_id=reviewer_id,
        reviewer_role="ordinary_reviewer",
        reviewer_source=reviewer_source,
        comment=comment,
        confirmed_at=requested_at,
    )
    return build_plain_reading_review(
        requirement_id=review.requirement_id,
        segmentation=segmentation,
        ambiguities=(),
        overall_status="awaiting_human_semantic_override",
        review_basis="human_requested_semantic_override",
        reviewer_source=reviewer_source,
        generator=review.generator,
        experiment_mode=review.experiment_mode,
        data_split=review.data_split,
        review_history=(
            *review.review_history,
            {
                "round": len(review.review_history) + 1,
                "operation": "human_requested_semantic_override",
                "old_reading_review_id": review.reading_review_id,
                "discarded_ambiguity_ids": (),
                "reviewer_id": reviewer_id,
                "reviewer_role": "ordinary_reviewer",
                "reviewer_source": reviewer_source,
                "comment": comment,
                "completed_at": requested_at,
            },
        ),
        revision_requests=(),
        no_ambiguity_confirmation=confirmation,
        ambiguity_generation_trigger=review.ambiguity_generation_trigger,
        downstream_status=review.downstream_status,
        supersedes_reading_review_id=review.reading_review_id,
    )


def request_ambiguity_generation(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    suspected_segment_ids: Sequence[str] = (),
    user_comment: str = "",
    reviewer_id: str = "",
    reviewer_role: str = "ordinary_reviewer",
    reviewer_source: str = "human",
) -> PlainReadingReview:
    """Record a human rejection before asking the LLM for candidate axes."""

    validate_plain_reading_review(review, segmentation)
    if review.overall_status != "pending_no_ambiguity_confirmation":
        raise PlainReadingValidationError("只有待确认的 LLM 空轴结论可以被否决")
    if review.review_basis == "llm_no_source_supported_axis_after_repair":
        raise PlainReadingValidationError(
            "semantic fallback不得再次强制生成歧义轴；请接受、暂缓或提供完整replacement"
        )
    known = {segment.segment_id for segment in segmentation.segments}
    requested_segments = tuple(dict.fromkeys(str(item) for item in suspected_segment_ids))
    unknown = set(requested_segments) - known
    if unknown:
        raise PlainReadingValidationError(
            "用户关注片段引用未知 segment_id: " + ", ".join(sorted(unknown))
        )
    requested_at = _timestamp()
    confirmation = NoAmbiguityConfirmation(
        status="rejected",
        llm_proposed_no_ambiguity=True,
        human_confirmed=True,
        reviewer_id=reviewer_id,
        reviewer_role=reviewer_role,
        reviewer_source=reviewer_source,
        comment=user_comment,
        confirmed_at=requested_at,
    )
    trigger = AmbiguityGenerationTrigger(
        source="human_disagreement",
        suspected_segment_ids=requested_segments,
        user_comment=user_comment,
        previous_reading_review_id=review.reading_review_id,
        requested_at=requested_at,
    )
    return build_plain_reading_review(
        requirement_id=review.requirement_id,
        segmentation=segmentation,
        ambiguities=(),
        overall_status="user_requested_ambiguity_generation",
        review_basis="human_disagreed_with_no_ambiguity",
        reviewer_source=reviewer_source,
        generator=review.generator,
        experiment_mode=review.experiment_mode,
        data_split=review.data_split,
        review_history=(
            *review.review_history,
            {
                "round": len(review.review_history) + 1,
                "operation": "human_requested_ambiguity_generation",
                "old_reading_review_id": review.reading_review_id,
                "suspected_segment_ids": requested_segments,
                "user_comment": user_comment,
                "reviewer_id": reviewer_id,
                "reviewer_role": reviewer_role,
                "reviewer_source": reviewer_source,
                "completed_at": requested_at,
            },
        ),
        revision_requests=review.revision_requests,
        no_ambiguity_confirmation=confirmation,
        ambiguity_generation_trigger=trigger,
        downstream_status=review.downstream_status,
        supersedes_reading_review_id=review.reading_review_id,
    )


def record_ambiguity_generation_exhausted(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    attempts: int,
    errors: Sequence[str],
) -> PlainReadingReview:
    """Persist a fail-closed forced-generation terminal state."""

    validate_plain_reading_review(review, segmentation)
    if review.overall_status != "user_requested_ambiguity_generation":
        raise PlainReadingValidationError(
            "只有等待强制生成的 review 可以记录生成耗尽"
        )
    total_attempts = review.ambiguity_generation_trigger.generation_attempts + attempts
    trigger = replace(
        review.ambiguity_generation_trigger,
        generation_attempts=total_attempts,
        last_errors=tuple(str(item) for item in errors if str(item)),
    )
    completed_at = _timestamp()
    return build_plain_reading_review(
        requirement_id=review.requirement_id,
        segmentation=segmentation,
        ambiguities=(),
        overall_status="needs_source_clarification",
        review_basis="forced_ambiguity_generation_exhausted",
        reviewer_source=review.reviewer_source,
        generator=review.generator,
        experiment_mode=review.experiment_mode,
        data_split=review.data_split,
        review_history=(
            *review.review_history,
            {
                "round": len(review.review_history) + 1,
                "operation": "forced_ambiguity_generation_exhausted",
                "old_reading_review_id": review.reading_review_id,
                "generation_attempts": total_attempts,
                "errors": trigger.last_errors,
                "completed_at": completed_at,
            },
        ),
        revision_requests=review.revision_requests,
        no_ambiguity_confirmation=review.no_ambiguity_confirmation,
        ambiguity_generation_trigger=trigger,
        downstream_status=review.downstream_status,
        supersedes_reading_review_id=review.reading_review_id,
    )


def withdraw_ambiguity_concern(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    reviewer_id: str = "",
    reviewer_role: str = "ordinary_reviewer",
    reviewer_source: str = "human",
    comment: str = "",
) -> PlainReadingReview:
    """Let the user withdraw a concern after inspecting generated candidates."""

    validate_plain_reading_review(review, segmentation)
    if review.ambiguity_generation_trigger.source != "human_disagreement":
        raise PlainReadingValidationError("只有人工请求生成的候选轴可以整体撤回")
    completed_at = _timestamp()
    confirmation = NoAmbiguityConfirmation(
        status="withdrawn",
        llm_proposed_no_ambiguity=True,
        human_confirmed=True,
        reviewer_id=reviewer_id,
        reviewer_role=reviewer_role,
        reviewer_source=reviewer_source,
        comment=comment,
        confirmed_at=completed_at,
    )
    return build_plain_reading_review(
        requirement_id=review.requirement_id,
        segmentation=segmentation,
        ambiguities=(),
        overall_status="no_ambiguity",
        review_basis="human_withdrew_ambiguity_concern",
        reviewer_source=reviewer_source,
        generator=review.generator,
        experiment_mode=review.experiment_mode,
        data_split=review.data_split,
        review_history=(
            *review.review_history,
            {
                "round": len(review.review_history) + 1,
                "operation": "human_withdrew_ambiguity_concern",
                "old_reading_review_id": review.reading_review_id,
                "discarded_ambiguity_ids": tuple(
                    ambiguity.ambiguity_id for ambiguity in review.ambiguities
                ),
                "reviewer_id": reviewer_id,
                "reviewer_role": reviewer_role,
                "reviewer_source": reviewer_source,
                "comment": comment,
                "completed_at": completed_at,
            },
        ),
        revision_requests=review.revision_requests,
        no_ambiguity_confirmation=confirmation,
        ambiguity_generation_trigger=review.ambiguity_generation_trigger,
        downstream_status=review.downstream_status,
        supersedes_reading_review_id=review.reading_review_id,
    )


def apply_plain_reading_decisions(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    decisions: Mapping[str, str],
    *,
    review_basis: str = "ordinary_language_reading",
    reviewer_id: str = "",
    reviewer_role: str = "ordinary_reviewer",
    reviewer_source: str = "human",
    comments: Mapping[str, str] | None = None,
    revision_requests: Mapping[str, PlainReadingRevisionRequest] | None = None,
) -> PlainReadingReview:
    """Apply A/B/N/U/R decisions one axis at a time."""

    comments = comments or {}
    revision_requests = revision_requests or {}
    if any(
        str(value).strip().casefold()
        in {"withdraw", "d", "撤回", "其实没有歧义"}
        for value in decisions.values()
    ):
        return withdraw_ambiguity_concern(
            review,
            segmentation,
            reviewer_id=reviewer_id,
            reviewer_role=reviewer_role,
            reviewer_source=reviewer_source,
            comment="; ".join(
                str(value) for value in comments.values() if str(value).strip()
            ),
        )
    active_scope = active_post_independent_revision_review_scope(review)
    scoped_review = active_scope is not None
    scoped_human_review = scoped_review and reviewer_source == "human"
    scoped_axis_ids = tuple(
        active_scope.get("post_revision_review_axis_ids", ())
        if active_scope is not None
        else ()
    )
    scoped_axis_id_set = set(scoped_axis_ids)
    if scoped_review:
        unresolved_decision_ids: list[str] = []
        axes_by_id = {item.ambiguity_id: item for item in review.ambiguities}
        for ambiguity_id in scoped_axis_ids:
            ambiguity = axes_by_id.get(ambiguity_id)
            if ambiguity is None:
                unresolved_decision_ids.append(ambiguity_id)
                continue
            normalized = str(decisions.get(ambiguity_id, "")).strip().casefold()
            option_ids = {
                option.option_id.casefold() for option in ambiguity.options
            }
            resolved = (
                normalized in option_ids
                or normalized
                in {
                    "ambiguous",
                    "unresolved",
                    "unable",
                    "无法判断",
                    "原句存在歧义",
                    "neither",
                    "none",
                    "两项都不符合",
                    "都不符合",
                    "n",
                    "not_distinct",
                    "paraphrase",
                }
                or (
                    normalized in {"r", "revise", "revision", "补充", "修订"}
                    and ambiguity_id in revision_requests
                )
            )
            if not resolved:
                unresolved_decision_ids.append(ambiguity_id)
        if unresolved_decision_ids:
            raise PlainReadingValidationError(
                "独立新轴重建后的复审必须原子提交全部轴决定，缺少: "
                + ", ".join(unresolved_decision_ids)
            )
    revised: list[PlainReadingAmbiguity] = []
    discarded_axis_ids: list[str] = []
    for ambiguity in review.ambiguities:
        raw = str(decisions.get(ambiguity.ambiguity_id, "")).strip()
        normalized = raw.casefold()
        option_ids = {option.option_id for option in ambiguity.options}
        selected = next(
            (option_id for option_id in option_ids if option_id.casefold() == normalized),
            "",
        )
        if selected:
            status = "reviewer_selected"
            decision = selected
        elif normalized in {"u", "ambiguous", "unresolved", "unable", "无法判断", "原句存在歧义"}:
            status = "ambiguous_unresolved"
            decision = ""
        elif normalized in {"n", "not_distinct", "paraphrase"}:
            discarded_axis_ids.append(ambiguity.ambiguity_id)
            continue
        elif normalized in {"neither", "none", "两项都不符合", "都不符合"}:
            if scoped_human_review and ambiguity.ambiguity_id in scoped_axis_id_set:
                discarded_axis_ids.append(ambiguity.ambiguity_id)
                continue
            status = "neither_option_selected"
            decision = ""
        elif normalized in {"r", "revise", "revision", "补充", "修订"}:
            if ambiguity.ambiguity_id not in revision_requests:
                raise PlainReadingValidationError(
                    f"{ambiguity.ambiguity_id} 选择 R 时必须提供修订请求"
                )
            status = "options_revision_requested"
            decision = ""
        else:
            status = "pending_plain_review"
            decision = ""
        revised.append(
            replace(
                ambiguity,
                decision=decision,
                decision_status=status,
                review_basis=review_basis if status != "pending_plain_review" else "",
                reviewer_id=reviewer_id,
                reviewer_role=reviewer_role,
                reviewer_source=(
                    reviewer_source if status != "pending_plain_review" else ""
                ),
                comments=str(comments.get(ambiguity.ambiguity_id, "")),
            )
        )
    statuses = {ambiguity.decision_status for ambiguity in revised}
    all_axes_discarded = bool(discarded_axis_ids and not revised)
    all_scoped_axes_discarded = bool(scoped_human_review and all_axes_discarded)
    if all_axes_discarded:
        overall_status = "pending_no_ambiguity_confirmation"
    elif "neither_option_selected" in statuses:
        overall_status = "needs_source_clarification"
    elif "options_revision_requested" in statuses:
        overall_status = "options_revision_requested"
    elif "pending_plain_review" in statuses:
        overall_status = "pending_plain_review"
    elif "ambiguous_unresolved" in statuses:
        overall_status = "ambiguous_unresolved"
    else:
        overall_status = "plain_reading_reviewed"
    effective_review_basis = (
        "human_rejected_all_ambiguity_axes"
        if all_axes_discarded
        else review_basis
    )
    decision_history = {
        "round": len(review.review_history) + 1,
        "old_reading_review_id": review.reading_review_id,
        "old_decision_fingerprint": decision_fingerprint(review),
        "decisions": dict(decisions),
        "review_basis": effective_review_basis,
        "reviewer_id": reviewer_id,
        "reviewer_role": reviewer_role,
        "reviewer_source": reviewer_source,
        "revision_request_ids": tuple(
            request.request_id for request in revision_requests.values()
        ),
        "discarded_ambiguity_ids": tuple(discarded_axis_ids),
        "all_axes_discarded": all_axes_discarded,
        "completed_at": _timestamp(),
    }
    if scoped_review:
        decision_history.update(
            {
                "operation": POST_INDEPENDENT_REVISION_REVIEW_OPERATION,
                "scope_id": str(active_scope["post_revision_review_scope_id"]),
                "decisions": {
                    ambiguity_id: str(decisions[ambiguity_id])
                    for ambiguity_id in scoped_axis_ids
                },
                "retained_ambiguity_ids": tuple(
                    ambiguity.ambiguity_id for ambiguity in revised
                ),
                "discarded_ambiguity_ids": tuple(discarded_axis_ids),
                "all_axes_discarded": all_scoped_axes_discarded,
                "original_ambiguity_id": str(
                    active_scope.get("original_ambiguity_id", "")
                ),
                "new_axis_ids": tuple(active_scope.get("new_axis_ids", ())),
            }
        )
    history = (
        *review.review_history,
        decision_history,
    )
    incoming_requests = {
        request.ambiguity_id: request for request in revision_requests.values()
    }
    superseded_at = _timestamp()
    merged_requests = [
        replace(
            item,
            request_status="superseded",
            updated_at=superseded_at,
        )
        if (
            item.ambiguity_id in incoming_requests
            and item.request_id
            != incoming_requests[item.ambiguity_id].request_id
            and item.request_status in {"submitted", "suggested", "confirmed"}
        )
        else item
        for item in review.revision_requests
    ]
    positions = {item.request_id: index for index, item in enumerate(merged_requests)}
    for request in revision_requests.values():
        if request.request_id in positions:
            merged_requests[positions[request.request_id]] = request
        else:
            positions[request.request_id] = len(merged_requests)
            merged_requests.append(request)
    result = build_plain_reading_review(
        requirement_id=review.requirement_id,
        segmentation=segmentation,
        ambiguities=revised,
        overall_status=overall_status,
        review_basis=effective_review_basis,
        reviewer_source=reviewer_source,
        generator=review.generator,
        experiment_mode=review.experiment_mode,
        data_split=review.data_split,
        review_history=history,
        revision_requests=tuple(merged_requests),
        no_ambiguity_confirmation=(
            NoAmbiguityConfirmation(
                status="pending", llm_proposed_no_ambiguity=True,
            )
            if all_axes_discarded
            else review.no_ambiguity_confirmation
        ),
        ambiguity_generation_trigger=review.ambiguity_generation_trigger,
        downstream_status=review.downstream_status,
        supersedes_reading_review_id=review.reading_review_id,
    )
    return result


def create_plain_reading_revision_request(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    ambiguity_id: str,
    proposed_interpretation: str,
    reason: str,
    source_segment_ids: Sequence[str],
    reviewer_id: str = "",
    reviewer_role: str = "ordinary_reviewer",
    reviewer_source: str = "human",
    max_revision_rounds: int = MAX_PLAIN_READING_REVISION_ROUNDS,
) -> PlainReadingRevisionRequest:
    """Create a non-authoritative revision proposal with mechanical validation."""

    validate_plain_reading_review(review, segmentation)
    ambiguity = next(
        (item for item in review.ambiguities if item.ambiguity_id == ambiguity_id),
        None,
    )
    if ambiguity is None:
        raise PlainReadingValidationError("revision request references unknown ambiguity_id")
    existing_rounds = sum(
        1 for item in review.revision_requests if item.ambiguity_id == ambiguity_id
    )
    revision_round = existing_rounds + 1
    if max_revision_rounds < 1:
        raise PlainReadingValidationError("max_revision_rounds must be at least 1")
    if revision_round > max_revision_rounds:
        raise PlainReadingValidationError(
            "plain-reading revision rounds exhausted; source clarification is required"
        )
    proposed = " ".join(proposed_interpretation.split())
    reason = " ".join(reason.split())
    segments = tuple(dict.fromkeys(str(item) for item in source_segment_ids if str(item)))
    errors = _revision_mechanical_errors(
        review,
        segmentation,
        ambiguity,
        proposed,
        reason,
        segments,
    )
    normalized = _normalize_option(proposed)
    equivalent = next(
        (
            option.option_id
            for option in ambiguity.options
            if _normalize_option(option.plain_interpretation) == normalized
        ),
        "",
    )
    source_warnings = tuple(
        error.removeprefix("out_of_source:")
        for error in errors
        if error.startswith("out_of_source:")
    )
    hard_errors = [
        error for error in errors if not error.startswith("out_of_source:")
    ]
    if hard_errors:
        raise PlainReadingValidationError("；".join(hard_errors))
    if equivalent:
        validation_status = "needs_human_confirmation"
        classification = "equivalent_to_existing"
    else:
        validation_status = "needs_human_confirmation"
        classification = "unresolved"
    basis = "; ".join(errors) if errors else (
        f"normalized text equals option {equivalent}"
        if equivalent
        else "mechanical checks passed; semantic classification requires human confirmation"
    )
    now = _timestamp()
    raw_id = json.dumps(
        (
            review.reading_review_id,
            ambiguity_id,
            proposed,
            reason,
            segments,
            revision_round,
        ),
        ensure_ascii=False,
    )
    request_id = "READREV-" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
    return PlainReadingRevisionRequest(
        request_id=request_id,
        reading_review_id=review.reading_review_id,
        ambiguity_id=ambiguity_id,
        segmentation_id=review.segmentation_id,
        source_hash=review.source_hash,
        proposed_interpretation=proposed,
        reason=reason,
        source_segment_ids=segments,
        reviewer_id=reviewer_id,
        reviewer_role=reviewer_role,
        reviewer_source=reviewer_source,
        validation_status=validation_status,
        classification=classification,
        classification_basis=basis,
        request_status="submitted",
        source_warnings=source_warnings,
        parent_decision_fingerprint=decision_fingerprint(review),
        revision_round=revision_round,
        created_at=now,
        updated_at=now,
    )


def confirm_plain_reading_revision_classification(
    request: PlainReadingRevisionRequest,
    *,
    classification: str,
    classification_basis: str,
    confirmed_by: str = "",
    confirmed_role: str = "ordinary_reviewer",
) -> PlainReadingRevisionRequest:
    """Record the required human confirmation of a classifier suggestion."""

    basis = classification_basis.strip()
    if not basis:
        raise PlainReadingValidationError("human classification requires a basis")
    if classification == "unresolved":
        raise PlainReadingValidationError(
            "unresolved is not a confirmable revision classification"
        )
    if request.request_status not in {"submitted", "suggested"}:
        raise PlainReadingValidationError(
            "only submitted or suggested revisions can be confirmed"
        )
    if (
        request.classification == "equivalent_to_existing"
        and classification != "equivalent_to_existing"
    ):
        raise PlainReadingValidationError(
            "an exact duplicate cannot be reclassified into a new reading axis"
        )
    status = "rejected" if classification == "out_of_source" else "accepted"
    now = _timestamp()
    return replace(
        request,
        validation_status=status,
        classification=classification,
        classification_basis=basis,
        human_confirmed=True,
        request_status="rejected" if classification == "out_of_source" else "confirmed",
        confirmed_by=confirmed_by,
        confirmed_role=confirmed_role,
        confirmed_at=now,
        updated_at=now,
    )


def apply_plain_reading_revision(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    request: PlainReadingRevisionRequest,
    *,
    replacement_axes: Sequence[PlainReadingAmbiguity] = (),
    reviewer_id: str = "",
    acknowledge_rebuilt_source_warnings: bool = False,
) -> PlainReadingReview:
    """Apply a confirmed classification and rebuild binary axes with a new id."""

    validate_plain_reading_review(review, segmentation)
    allowed_parent_fingerprints = {
        decision_fingerprint(review),
        *(
            str(item.get("old_decision_fingerprint", ""))
            for item in review.review_history
        ),
    } - {""}
    if (
        request.parent_decision_fingerprint
        and request.parent_decision_fingerprint not in allowed_parent_fingerprints
    ):
        raise PlainReadingValidationError(
            "revision request parent decision fingerprint is stale"
        )
    if (
        not request.human_confirmed
        or request.validation_status != "accepted"
        or request.request_status != "confirmed"
    ):
        raise PlainReadingValidationError(
            "revision classification must be accepted by a human before options change"
        )
    target = next(
        (item for item in review.ambiguities if item.ambiguity_id == request.ambiguity_id),
        None,
    )
    if target is None:
        raise PlainReadingValidationError("revision target ambiguity is stale")
    replacements = tuple(replacement_axes)
    if request.classification in {"same_axis_revision", "new_independent_axis"} and not replacements:
        raise PlainReadingValidationError(
            "axis revision classification requires rebuilt binary axes"
        )
    rebuilt_warnings = validate_rebuilt_plain_reading_axes(
        segmentation.source_text,
        segmentation,
        replacements,
    )
    if rebuilt_warnings and not acknowledge_rebuilt_source_warnings:
        raise PlainReadingValidationError(
            "rebuilt axes require explicit source-warning acknowledgement: "
            + "; ".join(rebuilt_warnings)
        )
    if request.classification == "equivalent_to_existing":
        rebuilt = tuple(
            replace(item, decision="", decision_status="pending_plain_review")
            if item.ambiguity_id == target.ambiguity_id
            else item
            for item in review.ambiguities
        )
    elif request.classification == "same_axis_revision":
        rebuilt_items: list[PlainReadingAmbiguity] = []
        for item in review.ambiguities:
            if item.ambiguity_id == target.ambiguity_id:
                rebuilt_items.extend(replacements)
            else:
                rebuilt_items.append(item)
        rebuilt = tuple(rebuilt_items)
    elif request.classification == "new_independent_axis":
        retained = tuple(
            replace(item, decision="", decision_status="pending_plain_review")
            if item.ambiguity_id == target.ambiguity_id
            else item
            for item in review.ambiguities
        )
        rebuilt = (*retained, *replacements)
    else:
        raise PlainReadingValidationError(
            "only confirmed equivalent/same-axis/new-axis revisions can rebuild options"
        )
    applied_at = _timestamp()
    updated_request = replace(
        request,
        request_status="applied",
        applied_at=applied_at,
        updated_at=applied_at,
    )
    requests = tuple(
        updated_request if item.request_id == request.request_id else item
        for item in review.revision_requests
    )
    if request.request_id not in {item.request_id for item in requests}:
        requests = (*requests, updated_request)
    history_entry = {
        "round": len(review.review_history) + 1,
        "operation": "plain_reading_revision_applied",
        "old_reading_review_id": review.reading_review_id,
        "ambiguity_id": target.ambiguity_id,
        "old_options": [asdict(option) for option in target.options],
        "reviewer_input": request.proposed_interpretation,
        "revision_request_id": request.request_id,
        "classification": request.classification,
        "classification_basis": request.classification_basis,
        "human_confirmed": request.human_confirmed,
        "new_axes": [asdict(item) for item in replacements],
        "reviewer_id": reviewer_id,
        "completed_at": _timestamp(),
        **(
            {
                "post_revision_review_scope_id": request.request_id,
                "post_revision_review_kind": "new_independent_axis",
                "post_revision_review_axis_ids": tuple(
                    item.ambiguity_id for item in rebuilt
                ),
                "post_revision_review_source_hash": review.source_hash,
                "post_revision_review_segmentation_id": review.segmentation_id,
                "original_ambiguity_id": target.ambiguity_id,
                "new_axis_ids": tuple(item.ambiguity_id for item in replacements),
            }
            if request.classification == "new_independent_axis"
            else {}
        ),
    }
    rebuilt_review = build_plain_reading_review(
        requirement_id=review.requirement_id,
        segmentation=segmentation,
        ambiguities=rebuilt,
        overall_status="pending_plain_review",
        review_basis="ordinary_language_revision",
        reviewer_source="human",
        generator=review.generator,
        experiment_mode=review.experiment_mode,
        data_split=review.data_split,
        review_history=(*review.review_history, history_entry),
        revision_requests=requests,
        no_ambiguity_confirmation=review.no_ambiguity_confirmation,
        ambiguity_generation_trigger=review.ambiguity_generation_trigger,
        downstream_status="stale_due_to_revision",
        supersedes_reading_review_id=review.reading_review_id,
    )
    final_history = (
        *rebuilt_review.review_history[:-1],
        {
            **rebuilt_review.review_history[-1],
            "new_reading_review_id": rebuilt_review.reading_review_id,
        },
    )
    return replace(rebuilt_review, review_history=final_history)


def close_plain_reading_revision_request(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    request: PlainReadingRevisionRequest,
    *,
    action: str = "return_to_review",
    reviewer_id: str = "",
) -> PlainReadingReview:
    """Close a rejected/withdrawn request without leaving the review deadlocked."""

    validate_plain_reading_review(review, segmentation)
    if action not in {"return_to_review", "needs_source_clarification", "withdraw"}:
        raise PlainReadingValidationError("unsupported revision close action")
    target = next(
        (item for item in review.ambiguities if item.ambiguity_id == request.ambiguity_id),
        None,
    )
    if target is None:
        raise PlainReadingValidationError("revision target ambiguity is stale")
    if request.request_status == "applied":
        raise PlainReadingValidationError("applied revision cannot be closed again")
    now = _timestamp()
    closed_status = "withdrawn" if action in {"return_to_review", "withdraw"} else "rejected"
    closed_request = replace(
        request,
        validation_status=(
            "rejected" if action == "needs_source_clarification" else request.validation_status
        ),
        human_confirmed=(
            True if action == "needs_source_clarification" else request.human_confirmed
        ),
        request_status=closed_status,
        confirmed_by=request.confirmed_by or reviewer_id,
        confirmed_role=request.confirmed_role or "ordinary_reviewer",
        confirmed_at=request.confirmed_at or now,
        updated_at=now,
    )
    requests = tuple(
        closed_request if item.request_id == request.request_id else item
        for item in review.revision_requests
    )
    if request.request_id not in {item.request_id for item in requests}:
        requests = (*requests, closed_request)
    clarification = action == "needs_source_clarification"
    rebuilt = tuple(
        replace(
            item,
            decision="",
            decision_status=(
                "neither_option_selected" if clarification else "pending_plain_review"
            ),
        )
        if item.ambiguity_id == target.ambiguity_id
        else item
        for item in review.ambiguities
    )
    return build_plain_reading_review(
        requirement_id=review.requirement_id,
        segmentation=segmentation,
        ambiguities=rebuilt,
        overall_status=(
            "needs_source_clarification" if clarification else "pending_plain_review"
        ),
        review_basis=review.review_basis,
        reviewer_source=review.reviewer_source,
        generator=review.generator,
        experiment_mode=review.experiment_mode,
        data_split=review.data_split,
        review_history=(
            *review.review_history,
            {
                "round": len(review.review_history) + 1,
                "operation": "plain_reading_revision_closed",
                "revision_request_id": request.request_id,
                "action": action,
                "reviewer_id": reviewer_id,
                "completed_at": now,
            },
        ),
        revision_requests=requests,
        no_ambiguity_confirmation=review.no_ambiguity_confirmation,
        ambiguity_generation_trigger=review.ambiguity_generation_trigger,
        downstream_status=review.downstream_status,
        supersedes_reading_review_id=review.reading_review_id,
    )


def validate_rebuilt_plain_reading_axes(
    source_text: str,
    segmentation: SurfaceSegmentation,
    axes: Sequence[PlainReadingAmbiguity],
) -> list[str]:
    """Validate model-built axes; return source heuristics requiring human acknowledgement."""

    warnings: list[str] = []
    known_segments = {item.segment_id for item in segmentation.segments}
    source_ids = _ap_like_tokens(source_text)
    source_numbers = set(re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", source_text))
    for axis in axes:
        if not axis.ambiguity_id or not axis.question.strip():
            raise PlainReadingValidationError("rebuilt ambiguity requires id and question")
        if not axis.source_segment_ids or set(axis.source_segment_ids) - known_segments:
            raise PlainReadingValidationError("rebuilt ambiguity references invalid source segments")
        if len(axis.options) != 2:
            raise PlainReadingValidationError("rebuilt ambiguity must contain exactly two options")
        normalized: set[str] = set()
        for option in axis.options:
            text = option.plain_interpretation.strip()
            if not option.option_id or not text or not option.difference_summary.strip():
                raise PlainReadingValidationError("rebuilt options must be complete")
            if re.search(r"(?:->|&&|\|\||\b[FGXURW]\b|[!&|])", text):
                raise PlainReadingValidationError("rebuilt options must not contain LTLf formulas")
            normalized.add(_normalize_option(text))
            extra_ids = _ap_like_tokens(text) - source_ids
            if extra_ids:
                warnings.append(
                    f"{axis.ambiguity_id}/{option.option_id} adds AP-like tokens: "
                    + ", ".join(sorted(extra_ids))
                )
            option_numbers = set(
                re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", text)
            )
            if option_numbers != source_numbers:
                warnings.append(
                    f"{axis.ambiguity_id}/{option.option_id} changes numeric cues"
                )
            if _has_negation(text) != _has_negation(source_text):
                warnings.append(
                    f"{axis.ambiguity_id}/{option.option_id} changes negation cues"
                )
        if len(normalized) != 2:
            raise PlainReadingValidationError("rebuilt options must be distinct")
    return list(dict.fromkeys(warnings))


def _revision_mechanical_errors(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    ambiguity: PlainReadingAmbiguity,
    proposed: str,
    reason: str,
    source_segment_ids: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    if len(proposed) < 8:
        errors.append("补充解释必须是完整、可独立理解的自然语言")
    if not reason:
        errors.append("补充理由不能为空")
    known_segments = {item.segment_id for item in segmentation.segments}
    if not source_segment_ids:
        errors.append("必须引用至少一个 source_segment_id")
    unknown = set(source_segment_ids) - known_segments
    if unknown:
        errors.append("引用未知 segment_id: " + ", ".join(sorted(unknown)))
    if re.search(r"(?:->|&&|\|\||\b[FGXRW]\s*\(|\sU\s)", proposed):
        errors.append("补充解释不得包含 LTLf 公式或公式运算符")
    source_ids = _ap_like_tokens(segmentation.source_text)
    proposed_ids = _ap_like_tokens(proposed)
    extra_ids = proposed_ids - source_ids
    if extra_ids:
        errors.append("out_of_source:新增 AP-like token " + ", ".join(sorted(extra_ids)))
    source_numbers = _quantity_values(segmentation.source_text)
    proposed_numbers = _quantity_values(proposed)
    if source_numbers != proposed_numbers:
        errors.append("out_of_source:数字或数量约束与原文不一致")
    if _has_negation(segmentation.source_text) != _has_negation(proposed):
        errors.append("out_of_source:关键否定与原文不一致")
    normalized = _normalize_option(proposed)
    if not normalized:
        errors.append("补充解释不能为空")
    if any(
        _normalize_option(option.plain_interpretation) == normalized
        for option in ambiguity.options
    ):
        # Exact equivalence is classified explicitly instead of creating C.
        pass
    return errors


def _ap_like_tokens(value: str) -> set[str]:
    """Legacy symbolic helper retained for non-override runtime checks."""

    return set(_v1_human_override_symbolic_tokens(value))


def _v1_human_override_symbolic_tokens(value: str) -> tuple[str, ...]:
    """Frozen v1 override reader; future AP extractors must not alter it."""

    return v1_symbolic_tokens(value)


def validate_human_override_ap_equality(
    original_text: str,
    replacement_text: str,
    *,
    parent_effective_source_hash: str | None = None,
) -> tuple[ValidationIssue, ...]:
    """Validate only the AP alphabet across a human whole-source replacement."""

    original = original_text.strip()
    replacement = replacement_text.strip()
    original_aps = _ap_like_tokens(original)
    replacement_aps = _ap_like_tokens(replacement)
    issues: list[ValidationIssue] = []

    def issue(code: str, message: str, *, observed: Any, expected: Any) -> None:
        issues.append(
            ValidationIssue(
                code=code,
                stage="human_semantic_override",
                path="/replacement_text",
                observed=observed,
                expected=expected,
                context={
                    "original_ap_set": sorted(original_aps),
                    "replacement_ap_set": sorted(replacement_aps),
                },
                repair_class=RepairClass.HUMAN_REQUIRED,
                allowed_paths=("/replacement_text",),
                message=message,
            )
        )

    if not replacement:
        issue(
            "HUMAN_OVERRIDE_EMPTY_TEXT",
            "replacement text must be a complete non-empty requirement",
            observed=replacement_text,
            expected="non-empty ordinary-language requirement",
        )
        return tuple(issues)
    if parent_effective_source_hash is not None and (
        source_hash(original_text) != parent_effective_source_hash
    ):
        issue(
            "HUMAN_OVERRIDE_STALE_PARENT",
            "semantic override parent no longer matches the effective source",
            observed=parent_effective_source_hash,
            expected=source_hash(original_text),
        )
        return tuple(issues)
    if not original_aps:
        issue(
            "HUMAN_OVERRIDE_AP_EXTRACTION_UNKNOWN",
            "the original requirement has no deterministically identifiable AP tokens",
            observed=original_text,
            expected="at least one prop_N or single-letter AP token",
        )
        return tuple(issues)

    omitted = original_aps - replacement_aps
    added = replacement_aps - original_aps
    if omitted and added and len(original_aps) == len(replacement_aps):
        issue(
            "HUMAN_OVERRIDE_RENAMES_AP",
            "replacement appears to rename AP tokens",
            observed={"omitted": sorted(omitted), "added": sorted(added)},
            expected=sorted(original_aps),
        )
        return tuple(issues)
    if omitted:
        issue(
            "HUMAN_OVERRIDE_OMITS_AP",
            "replacement omits AP tokens: " + ", ".join(sorted(omitted)),
            observed=sorted(replacement_aps),
            expected=sorted(original_aps),
        )
    if added:
        issue(
            "HUMAN_OVERRIDE_ADDS_AP",
            "replacement adds AP tokens: " + ", ".join(sorted(added)),
            observed=sorted(replacement_aps),
            expected=sorted(original_aps),
        )
    return tuple(issues)


def validate_human_semantic_override_integrity(
    override: HumanSemanticOverride,
    *,
    original_inventory: APInventory | None = None,
    replacement_inventory: APInventory | None = None,
    comparison: OverrideAPComparison | None = None,
) -> None:
    """Fail closed when persisted or in-memory override lineage is corrupted.

    ``HumanSemanticOverride.__post_init__`` protects hashes at construction time,
    but persisted mappings also contain derived AP sets and callers can receive
    objects from older checkpoint code.  Keep this check at the activation
    boundary as well as the checkpoint boundary so a corrupted proposal can
    never become the effective grounding source.
    """

    if not isinstance(override, HumanSemanticOverride):
        raise PlainReadingValidationError(
            "HUMAN_OVERRIDE_CORRUPT: expected HumanSemanticOverride"
        )
    errors: list[str] = []
    expected_original_hash = source_hash(override.original_source_text)
    expected_replacement_hash = (
        source_hash(override.replacement_text) if override.replacement_text else ""
    )
    if override.original_source_hash != expected_original_hash:
        errors.append("original_source_hash does not match original_source_text")
    if override.replacement_hash != expected_replacement_hash:
        errors.append("replacement_hash does not match replacement_text")
    contract = (
        override.override_contract_version
        or HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION
    )
    if contract == HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION:
        expected_original_aps = _v1_human_override_symbolic_tokens(
            override.original_source_text
        )
        expected_replacement_aps = _v1_human_override_symbolic_tokens(
            override.replacement_text
        )
        if tuple(override.original_ap_set) != expected_original_aps:
            errors.append("original_ap_set does not match v1 original source")
        if tuple(override.replacement_ap_set) != expected_replacement_aps:
            errors.append("replacement_ap_set does not match v1 replacement source")
    elif contract == HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION:
        if not override.representation_profile_id:
            errors.append("v2 override lacks representation profile")
        if override.ap_validation_contract_version != OVERRIDE_AP_COMPARISON_CONTRACT_VERSION:
            errors.append("v2 AP validation contract mismatch")
        inventory_required_statuses = {
            "awaiting_ap_mapping_confirmation",
            "ap_rejected",
            "awaiting_confirmation",
            "active",
            "cancelled",
            "superseded",
        }
        if override.status in inventory_required_statuses and not (
            override.original_ap_inventory_id
            and override.original_ap_inventory_hash
            and override.replacement_ap_inventory_id
            and override.replacement_ap_inventory_hash
        ):
            errors.append("v2 override lacks AP inventory references")
        if override.status in {"awaiting_confirmation", "active"} and not (
            override.ap_comparison_id and override.ap_comparison_hash
        ):
            errors.append("v2 override lacks passed comparison reference")
        if original_inventory is not None:
            try:
                validate_ap_inventory_integrity(
                    original_inventory,
                    allow_unapproved=original_inventory.status != "approved",
                )
            except APIdentityValidationError as exc:
                errors.append("original inventory invalid: " + str(exc))
            if (
                override.original_ap_inventory_id != original_inventory.inventory_id
                or override.original_ap_inventory_hash != original_inventory.artifact_hash
                or override.original_source_hash != original_inventory.source_hash
            ):
                errors.append("original inventory lineage mismatch")
            expected_original_ap_set = (
                tuple(
                    sorted(
                        item.ap_identity_id for item in original_inventory.identities
                    )
                )
                if original_inventory.status == "approved"
                else ()
            )
            if tuple(override.original_ap_set) != expected_original_ap_set:
                errors.append("original_ap_set is not derived from original inventory")
        if replacement_inventory is not None:
            try:
                validate_ap_inventory_integrity(
                    replacement_inventory,
                    allow_unapproved=replacement_inventory.status != "approved",
                )
            except APIdentityValidationError as exc:
                errors.append("replacement inventory invalid: " + str(exc))
            if (
                override.replacement_ap_inventory_id
                != replacement_inventory.inventory_id
                or override.replacement_ap_inventory_hash
                != replacement_inventory.artifact_hash
                or override.replacement_hash != replacement_inventory.source_hash
            ):
                errors.append("replacement inventory lineage mismatch")
            expected_replacement_ap_set = (
                tuple(
                    sorted(
                        item.ap_identity_id
                        for item in replacement_inventory.identities
                    )
                )
                if replacement_inventory.status == "approved"
                else ()
            )
            if tuple(override.replacement_ap_set) != expected_replacement_ap_set:
                errors.append("replacement_ap_set is not derived from replacement inventory")
        if comparison is not None:
            if original_inventory is None or replacement_inventory is None:
                errors.append("comparison validation requires both inventories")
            else:
                try:
                    validate_override_ap_comparison_integrity(
                        comparison,
                        original_inventory=original_inventory,
                        replacement_inventory=replacement_inventory,
                    )
                except APIdentityValidationError as exc:
                    errors.append("AP comparison invalid: " + str(exc))
            if (
                override.ap_comparison_id != comparison.comparison_id
                or override.ap_comparison_hash != comparison.artifact_hash
            ):
                errors.append("AP comparison lineage mismatch")
            if override.status in {"awaiting_confirmation", "active"} and comparison.status != "passed":
                errors.append("confirmation-ready override references non-passed comparison")
    else:
        errors.append("unsupported human semantic override contract")
    if errors:
        raise PlainReadingValidationError(
            "HUMAN_OVERRIDE_CORRUPT: " + "; ".join(errors),
            errors=tuple(errors),
        )


def create_human_semantic_override(
    *,
    requirement_id: str,
    original_source_text: str,
    replacement_text: str = "",
    parent_effective_source_hash: str,
    reviewer_id: str = "",
    reviewer_source: str = "human",
    representation_profile: APRepresentationProfile | None = None,
    original_segmentation: SurfaceSegmentation | None = None,
    declared_catalog: DeclaredAPCatalog | None = None,
    original_inventory: APInventory | None = None,
    replacement_inventory: APInventory | None = None,
    original_ap_proposals: Sequence[Mapping[str, Any]] = (),
    replacement_ap_proposals: Sequence[Mapping[str, Any]] = (),
    identity_mapping: Sequence[APIdentityMapping | Mapping[str, Any]] = (),
    ap_artifact_callback: Callable[
        [APInventory | OverrideAPComparison], None
    ]
    | None = None,
) -> HumanSemanticOverride:
    """Create an inactive override proposal and run the AP-only admission gate."""

    created_at = _timestamp()
    replacement_text = replacement_text.strip()
    replacement_hash = source_hash(replacement_text) if replacement_text else ""
    override_id = _new_human_semantic_override_id(
        requirement_id=requirement_id,
        original_source_text=original_source_text,
        replacement_hash=replacement_hash,
        created_at=created_at,
    )
    if replacement_text and is_complete_formula_like_source(replacement_text):
        return HumanSemanticOverride(
            override_id=override_id,
            requirement_id=requirement_id,
            original_source_text=original_source_text,
            original_source_hash=source_hash(original_source_text),
            replacement_text=replacement_text,
            replacement_hash=replacement_hash,
            original_ap_set=(),
            replacement_ap_set=(),
            ap_validation_status="pending",
            parent_effective_source_hash=parent_effective_source_hash,
            reviewer_id=reviewer_id,
            reviewer_source=reviewer_source,
            status="awaiting_source_modality_choice",
            issue_codes=("FORMULA_LIKE_SOURCE_REQUIRES_FORMAL_AUTHORITY",),
            old_effective_source_hash=parent_effective_source_hash,
            created_at=created_at,
            override_contract_version=(
                HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION
                if representation_profile is not None
                else HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION
            ),
            ap_validation_contract_version=(
                OVERRIDE_AP_COMPARISON_CONTRACT_VERSION
                if representation_profile is not None
                else "symbolic-token-v1"
            ),
            representation_profile_id=(
                representation_profile.profile_id
                if representation_profile is not None
                else ""
            ),
            ap_validation_mode=(
                representation_profile.mode
                if representation_profile is not None
                else "symbolic_token_v1"
            ),
        )
    if representation_profile is not None:
        if source_hash(original_source_text) != parent_effective_source_hash:
            raise PlainReadingValidationError("HUMAN_OVERRIDE_STALE_PARENT")
        if not replacement_text:
            return HumanSemanticOverride(
                override_id=override_id,
                requirement_id=requirement_id,
                original_source_text=original_source_text,
                original_source_hash=source_hash(original_source_text),
                replacement_text="",
                replacement_hash="",
                original_ap_set=(),
                replacement_ap_set=(),
                ap_validation_status="pending",
                parent_effective_source_hash=parent_effective_source_hash,
                reviewer_id=reviewer_id,
                reviewer_source=reviewer_source,
                status="awaiting_input",
                old_effective_source_hash=parent_effective_source_hash,
                created_at=created_at,
                override_contract_version=HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION,
                ap_validation_contract_version=(
                    OVERRIDE_AP_COMPARISON_CONTRACT_VERSION
                ),
                representation_profile_id=representation_profile.profile_id,
                ap_validation_mode=representation_profile.mode,
            )
        original_inventory = original_inventory or _build_profile_inventory(
            requirement_id=requirement_id,
            source_text=original_source_text,
            profile=representation_profile,
            segmentation=original_segmentation,
            declared_catalog=declared_catalog,
            proposals=original_ap_proposals,
        )
        replacement_inventory = replacement_inventory or _build_profile_inventory(
            requirement_id=requirement_id,
            source_text=replacement_text,
            profile=representation_profile,
            segmentation=None,
            declared_catalog=declared_catalog,
            proposals=replacement_ap_proposals,
            parent_inventory_id=original_inventory.inventory_id,
        )
        comparison = compare_override_ap_inventories(
            override_id=override_id,
            original_inventory=original_inventory,
            replacement_inventory=replacement_inventory,
            identity_mapping=identity_mapping,
            validation_mode=(
                "symbolic_exact_v1"
                if representation_profile.mode == "controlled_symbolic"
                else "declared_catalog_v1"
                if representation_profile.mode == "declared_catalog"
                else "manual_mapping_v1"
            ),
            reviewer_role=("ordinary_reviewer" if identity_mapping else "system"),
            reviewer_id=reviewer_id if identity_mapping else "",
            reviewer_source=reviewer_source if identity_mapping else "deterministic",
        )
        if ap_artifact_callback is not None:
            ap_artifact_callback(original_inventory)
            ap_artifact_callback(replacement_inventory)
            ap_artifact_callback(comparison)
        override = _v2_override_from_artifacts(
            override_id=override_id,
            requirement_id=requirement_id,
            original_source_text=original_source_text,
            replacement_text=replacement_text,
            parent_effective_source_hash=parent_effective_source_hash,
            reviewer_id=reviewer_id,
            reviewer_source=reviewer_source,
            profile=representation_profile,
            original_inventory=original_inventory,
            replacement_inventory=replacement_inventory,
            comparison=comparison,
            created_at=created_at,
        )
        validate_human_semantic_override_integrity(
            override,
            original_inventory=(
                original_inventory if original_inventory.status == "approved" else None
            ),
            replacement_inventory=(
                replacement_inventory
                if replacement_inventory.status == "approved"
                else None
            ),
            comparison=(
                comparison
                if comparison.status in {"passed", "mismatch", "needs_mapping_review"}
                and original_inventory.status == "approved"
                and replacement_inventory.status == "approved"
                else None
            ),
        )
        return override

    # Missing profile is an explicit v1 compatibility call.  New CLI/batch
    # writers always supply a v2 representation profile.
    issues = (
        validate_human_override_ap_equality(
            original_source_text,
            replacement_text,
        )
        if replacement_text.strip()
        else ()
    )
    original_aps = tuple(sorted(_ap_like_tokens(original_source_text)))
    replacement_aps = tuple(sorted(_ap_like_tokens(replacement_text)))
    if not replacement_text.strip():
        status = "awaiting_input"
        ap_status = "pending"
    elif issues:
        status = "ap_rejected"
        ap_status = "rejected"
    else:
        status = "awaiting_confirmation"
        ap_status = "passed"
    return HumanSemanticOverride(
        override_id=override_id,
        requirement_id=requirement_id,
        original_source_text=original_source_text,
        original_source_hash=source_hash(original_source_text),
        replacement_text=replacement_text,
        replacement_hash=replacement_hash,
        original_ap_set=original_aps,
        replacement_ap_set=replacement_aps,
        ap_validation_status=ap_status,
        parent_effective_source_hash=parent_effective_source_hash,
        reviewer_id=reviewer_id,
        reviewer_source=reviewer_source,
        status=status,
        issue_codes=tuple(issue.code for issue in issues),
        old_effective_source_hash=parent_effective_source_hash,
        created_at=created_at,
        override_contract_version=HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION,
        ap_validation_contract_version="symbolic-token-v1",
        ap_validation_mode="symbolic_token_v1",
    )


def advance_human_semantic_override_ap_review(
    override: HumanSemanticOverride,
    *,
    representation_profile: APRepresentationProfile,
    original_inventory: APInventory,
    replacement_inventory: APInventory,
    identity_mapping: Sequence[APIdentityMapping | Mapping[str, Any]] = (),
    reviewer_id: str,
    reviewer_source: str = "human",
) -> tuple[HumanSemanticOverride, OverrideAPComparison]:
    """Apply an explicit inventory/mapping review without activating text."""

    if override.override_contract_version != HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION:
        raise PlainReadingValidationError("AP review requires a v2 semantic override")
    if override.status not in {
        "awaiting_ap_inventory",
        "awaiting_ap_mapping_confirmation",
    }:
        raise PlainReadingValidationError("override is not awaiting AP review")
    comparison = compare_override_ap_inventories(
        override_id=override.override_id,
        original_inventory=original_inventory,
        replacement_inventory=replacement_inventory,
        identity_mapping=identity_mapping,
        validation_mode="manual_mapping_v1",
        reviewer_role="ordinary_reviewer",
        reviewer_id=reviewer_id,
        reviewer_source=reviewer_source,
        allow_formula_assisted=(
            representation_profile.mode == "offline_formula_assisted"
            and representation_profile.allow_formula_assisted_catalog
        ),
    )
    updated = _v2_override_from_artifacts(
        override_id=override.override_id,
        requirement_id=override.requirement_id,
        original_source_text=override.original_source_text,
        replacement_text=override.replacement_text,
        parent_effective_source_hash=override.parent_effective_source_hash,
        reviewer_id=reviewer_id,
        reviewer_source=override.reviewer_source,
        profile=representation_profile,
        original_inventory=original_inventory,
        replacement_inventory=replacement_inventory,
        comparison=comparison,
        created_at=override.created_at,
        supersedes_override_id=override.supersedes_override_id,
    )
    validate_human_semantic_override_integrity(
        updated,
        original_inventory=(
            original_inventory if original_inventory.status == "approved" else None
        ),
        replacement_inventory=(
            replacement_inventory if replacement_inventory.status == "approved" else None
        ),
        comparison=(
            comparison
            if original_inventory.status == "approved"
            and replacement_inventory.status == "approved"
            else None
        ),
    )
    return updated, comparison


def cancel_human_semantic_override_ap_review(
    override: HumanSemanticOverride,
    *,
    reviewer_id: str,
) -> HumanSemanticOverride:
    """Cancel a pending v2 AP review without accepting the replacement."""

    if override.override_contract_version != HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION:
        raise PlainReadingValidationError("AP review cancellation requires a v2 override")
    if override.status not in {
        "awaiting_ap_inventory",
        "awaiting_ap_mapping_confirmation",
    }:
        raise PlainReadingValidationError("override is not awaiting AP review")
    return replace(
        override,
        reviewer_id=reviewer_id or override.reviewer_id,
        confirmation_status="rejected",
        status="cancelled",
    )


def classify_v1_human_semantic_override_migration(
    override: HumanSemanticOverride,
) -> str:
    """Classify a v1 event without reinterpreting it through v2 extraction."""

    try:
        validate_human_semantic_override_integrity(override)
    except PlainReadingValidationError:
        return "corrupt"
    if (
        override.override_contract_version
        != HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION
    ):
        return "manual"
    if override.status in {"active", "awaiting_confirmation"} and (
        override.ap_validation_status == "passed"
    ):
        return "keep-v1"
    if (
        override.status == "ap_rejected"
        and "HUMAN_OVERRIDE_AP_EXTRACTION_UNKNOWN" in override.issue_codes
    ):
        return "upgradeable"
    return "manual"


def migrate_v1_unknown_human_semantic_override(
    override: HumanSemanticOverride,
    *,
    reviewer_id: str,
    ap_artifact_callback: Callable[
        [APInventory | OverrideAPComparison], None
    ]
    | None = None,
) -> HumanSemanticOverride:
    """Create a new v2 pending event; never mutate or activate the v1 event."""

    if classify_v1_human_semantic_override_migration(override) != "upgradeable":
        raise PlainReadingValidationError(
            "only v1 extraction-unknown override is explicitly upgradeable"
        )
    from ap_identity import undetermined_profile

    migrated = create_human_semantic_override(
        requirement_id=override.requirement_id,
        original_source_text=override.original_source_text,
        replacement_text=override.replacement_text,
        parent_effective_source_hash=override.parent_effective_source_hash,
        reviewer_id=reviewer_id,
        reviewer_source="human_v1_migration",
        representation_profile=undetermined_profile(
            provenance="explicit_v1_unknown_migration"
        ),
        ap_artifact_callback=ap_artifact_callback,
    )
    return replace(
        migrated,
        supersedes_override_id=override.override_id,
    )


def confirm_human_semantic_override(
    override: HumanSemanticOverride,
    *,
    confirmed: bool,
    current_effective_source_hash: str,
    reviewer_id: str = "",
    original_inventory: APInventory | None = None,
    replacement_inventory: APInventory | None = None,
    comparison: OverrideAPComparison | None = None,
) -> HumanSemanticOverride:
    """Activate only an explicitly confirmed, AP-equal, non-stale override."""

    validate_human_semantic_override_integrity(
        override,
        original_inventory=original_inventory,
        replacement_inventory=replacement_inventory,
        comparison=comparison,
    )
    if override.status != "awaiting_confirmation":
        raise PlainReadingValidationError(
            "只有 AP 校验通过且等待确认的 replacement 可以确认"
        )
    if override.override_contract_version == HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION:
        issues = validate_human_override_ap_equality(
            override.original_source_text,
            override.replacement_text,
        )
        if issues:
            raise PlainReadingValidationError(
                "人工 replacement AP 校验不再有效: "
                + ", ".join(issue.code for issue in issues)
            )
    else:
        if original_inventory is None or replacement_inventory is None or comparison is None:
            raise PlainReadingValidationError(
                "HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: v2 confirmation requires "
                "both inventories and the passed comparison"
            )
        if comparison.status != "passed":
            raise PlainReadingValidationError(
                "只有已通过 AP identity comparison 的 replacement 可以确认"
            )
    if override.parent_effective_source_hash != current_effective_source_hash:
        raise PlainReadingValidationError("HUMAN_OVERRIDE_STALE_PARENT")
    if not confirmed:
        return replace(
            override,
            reviewer_id=reviewer_id or override.reviewer_id,
            confirmation_status="rejected",
            status="cancelled",
        )
    confirmed_at = _timestamp()
    return replace(
        override,
        reviewer_id=reviewer_id or override.reviewer_id,
        confirmation_status="confirmed",
        status="active",
        invalidation_reason="semantic_override",
        old_effective_source_hash=current_effective_source_hash,
        new_effective_source_hash=override.replacement_hash,
        confirmed_at=confirmed_at,
    )


def _build_profile_inventory(
    *,
    requirement_id: str,
    source_text: str,
    profile: APRepresentationProfile,
    segmentation: SurfaceSegmentation | None,
    declared_catalog: DeclaredAPCatalog | None,
    proposals: Sequence[Mapping[str, Any]],
    parent_inventory_id: str = "",
) -> APInventory:
    if profile.mode == "controlled_symbolic":
        return build_symbolic_ap_inventory(
            requirement_id=requirement_id,
            source_text=source_text,
            segmentation=segmentation,
            profile=profile,
            parent_inventory_id=parent_inventory_id,
        )
    if profile.mode == "declared_catalog":
        if declared_catalog is None:
            raise PlainReadingValidationError(
                "HUMAN_OVERRIDE_AP_INVENTORY_REQUIRED: declared catalog is missing"
            )
        return build_declared_catalog_ap_inventory(
            requirement_id=requirement_id,
            source_text=source_text,
            catalog=declared_catalog,
            profile=profile,
            segmentation=segmentation,
            parent_inventory_id=parent_inventory_id,
        )
    return propose_grounded_ap_inventory(
        requirement_id=requirement_id,
        source_text=source_text,
        profile=profile,
        proposals=proposals,
        segmentation=segmentation,
        parent_inventory_id=parent_inventory_id,
    )


def _v2_override_from_artifacts(
    *,
    override_id: str,
    requirement_id: str,
    original_source_text: str,
    replacement_text: str,
    parent_effective_source_hash: str,
    reviewer_id: str,
    reviewer_source: str,
    profile: APRepresentationProfile,
    original_inventory: APInventory,
    replacement_inventory: APInventory,
    comparison: OverrideAPComparison,
    created_at: str,
    supersedes_override_id: str = "",
) -> HumanSemanticOverride:
    if comparison.status == "passed":
        status = "awaiting_confirmation"
        ap_status = "passed"
    elif comparison.status == "mismatch":
        status = "ap_rejected"
        ap_status = "rejected"
    elif comparison.status == "needs_mapping_review":
        status = "awaiting_ap_mapping_confirmation"
        ap_status = "needs_review"
    elif comparison.status == "needs_inventory":
        status = "awaiting_ap_inventory"
        ap_status = "needs_review"
    else:
        raise PlainReadingValidationError(
            "HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: "
            + ", ".join(comparison.issue_codes)
        )
    original_approved = original_inventory.status == "approved"
    replacement_approved = replacement_inventory.status == "approved"
    return HumanSemanticOverride(
        override_id=override_id,
        requirement_id=requirement_id,
        original_source_text=original_source_text,
        original_source_hash=source_hash(original_source_text),
        replacement_text=replacement_text,
        replacement_hash=source_hash(replacement_text),
        original_ap_set=(
            tuple(sorted(item.ap_identity_id for item in original_inventory.identities))
            if original_approved
            else ()
        ),
        replacement_ap_set=(
            tuple(sorted(item.ap_identity_id for item in replacement_inventory.identities))
            if replacement_approved
            else ()
        ),
        ap_validation_status=ap_status,
        parent_effective_source_hash=parent_effective_source_hash,
        reviewer_id=reviewer_id,
        reviewer_source=reviewer_source,
        status=status,
        issue_codes=comparison.issue_codes,
        old_effective_source_hash=parent_effective_source_hash,
        created_at=created_at,
        override_contract_version=HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION,
        ap_validation_contract_version=OVERRIDE_AP_COMPARISON_CONTRACT_VERSION,
        representation_profile_id=profile.profile_id,
        original_ap_inventory_id=original_inventory.inventory_id,
        original_ap_inventory_hash=original_inventory.artifact_hash,
        replacement_ap_inventory_id=replacement_inventory.inventory_id,
        replacement_ap_inventory_hash=replacement_inventory.artifact_hash,
        ap_comparison_id=comparison.comparison_id,
        ap_comparison_hash=comparison.artifact_hash,
        ap_validation_mode=comparison.validation_mode,
        supersedes_override_id=supersedes_override_id,
    )


def _new_human_semantic_override_id(
    *,
    requirement_id: str,
    original_source_text: str,
    replacement_hash: str,
    created_at: str,
) -> str:
    digest = hashlib.sha256(
        (
            requirement_id
            + "\0"
            + source_hash(original_source_text)
            + "\0"
            + replacement_hash
            + "\0"
            + created_at
            + "\0"
            + uuid.uuid4().hex
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"HOV-{requirement_id}-{digest}"


def effective_source_text(
    original_source_text: str,
    override: HumanSemanticOverride | None = None,
) -> str:
    """Resolve the only grounding source visible to downstream stages."""

    if override is not None and override.status == "active":
        return override.replacement_text
    return original_source_text


def _explicit_negation_terms(value: str) -> set[str]:
    """Return complete explicit negation tokens without substring matches."""

    lowered = value.casefold()
    terms = {
        match.group(0).casefold()
        for match in _ENGLISH_EXPLICIT_NEGATION_RE.finditer(lowered)
    }
    terms.update(_CHINESE_EXPLICIT_NEGATION_RE.findall(lowered))
    return terms


def _has_negation(value: str) -> bool:
    return bool(_explicit_negation_terms(value))


def detect_rule_based_ambiguities(
    source_text: str,
    segmentation: SurfaceSegmentation,
) -> tuple[PlainReadingAmbiguity, ...]:
    """Conservatively detect high-risk axes without pretending to resolve them."""

    text = source_text
    lowered = text.casefold()
    axes: list[PlainReadingAmbiguity] = []
    all_segments = tuple(segment.segment_id for segment in segmentation.segments)
    next_terms = ("next", "next step", "下一步", "随后")
    has_next = any(term in lowered for term in next_terms)
    has_branch = bool(re.search(r"\b(?:or|and)\b|或者|并且|或", text, re.IGNORECASE))
    if has_next and has_branch:
        axes.append(
            PlainReadingAmbiguity(
                ambiguity_id="amb-next-scope",
                source_segment_ids=all_segments,
                question="“下一步”修饰整个并列要求，还是只修饰紧随其后的第一个分支？",
                options=(
                    PlainReadingOption(
                        option_id="A",
                        plain_interpretation=(
                            f"对原句“{text}”的读法是：如果原句的条件成立，那么从下一步开始，"
                            f"原句并列的每个后件分支都在该时间范围内解释。"
                        ),
                        difference_summary="“下一步”同时修饰所有并列分支",
                    ),
                    PlainReadingOption(
                        option_id="B",
                        plain_interpretation=(
                            f"对原句“{text}”的读法是：如果原句的条件成立，那么只有紧随"
                            f"“下一步”的第一个后件分支从下一步开始，"
                            f"其他并列分支从当前步骤开始解释。"
                        ),
                        difference_summary="“下一步”只修饰第一个并列分支",
                    ),
                ),
            )
        )
    if re.search(r"^\s*(?:if|when|whenever)\b|^若|如果|每当", text, re.IGNORECASE):
        axes.append(
            PlainReadingAmbiguity(
                ambiguity_id="amb-condition-check-scope",
                source_segment_ids=all_segments,
                question="原句的条件是只在执行开始时检查，还是在执行过程的每个步骤检查？",
                options=(
                    PlainReadingOption(
                        option_id="A",
                        plain_interpretation=f"对原句“{text}”的读法是：只在整个执行开始时检查一次原句的条件，并据此决定后续要求是否适用。",
                        difference_summary="条件仅在初始位置检查一次",
                    ),
                    PlainReadingOption(
                        option_id="B",
                        plain_interpretation=f"对原句“{text}”的读法是：在执行过程的每个步骤都检查原句的条件；只要条件成立，就应用后续要求。",
                        difference_summary="条件在所有执行位置重复检查",
                    ),
                ),
            )
        )
    if re.search(r"\buntil\b|直到", text, re.IGNORECASE):
        axes.append(
            PlainReadingAmbiguity(
                ambiguity_id="amb-until-strength",
                source_segment_ids=all_segments,
                question="“直到”是否要求终止条件在有限执行中最终必须发生？",
                options=(
                    PlainReadingOption(
                        option_id="A",
                        plain_interpretation=f"对原句“{text}”的读法是：前一个状态必须保持，而且“直到”后的终止条件在有限执行结束前必须最终成立。",
                        difference_summary="终止条件最终必须发生",
                    ),
                    PlainReadingOption(
                        option_id="B",
                        plain_interpretation=f"对原句“{text}”的读法是：前一个状态在终止条件出现前必须保持，但有限执行结束时终止条件仍可以没有发生。",
                        difference_summary="终止条件可以在轨迹结束时仍未发生",
                    ),
                ),
            )
        )
    return tuple(axes)


def decision_fingerprint(review: PlainReadingReview | None) -> str:
    """Fingerprint semantic choices, excluding reviewer metadata-only edits."""

    if review is None:
        return ""
    payload = [
        (
            ambiguity.ambiguity_id,
            ambiguity.decision_status,
            ambiguity.decision,
            ambiguity.source_span,
            ambiguity.evidence_text,
            tuple(
                (option.option_id, _normalize_option(option.plain_interpretation))
                for option in ambiguity.options
            ),
        )
        for ambiguity in review.ambiguities
    ]
    raw = json.dumps(
        {
            "segmentation_id": review.segmentation_id,
            "overall_status": review.overall_status,
            "ambiguities": payload,
            "revision_requests": [
                (
                    request.request_id,
                    request.ambiguity_id,
                    _normalize_option(request.proposed_interpretation),
                    request.validation_status,
                    request.classification,
                    request.classification_basis,
                    request.human_confirmed,
                    request.request_status,
                    request.source_warnings,
                    request.parent_decision_fingerprint,
                    request.revision_round,
                )
                for request in review.revision_requests
            ],
            "no_ambiguity_confirmation": (
                review.no_ambiguity_confirmation.status,
                review.no_ambiguity_confirmation.llm_proposed_no_ambiguity,
                review.no_ambiguity_confirmation.human_confirmed,
                _normalize_option(review.no_ambiguity_confirmation.comment),
            ) + ((review.no_ambiguity_confirmation.judge_confirmed, review.no_ambiguity_confirmation.judge_evidence_json)
                 if review.no_ambiguity_confirmation.judge_evidence_json else ()),
            "ambiguity_generation_trigger": (
                review.ambiguity_generation_trigger.source,
                review.ambiguity_generation_trigger.suspected_segment_ids,
                _normalize_option(review.ambiguity_generation_trigger.user_comment),
                review.ambiguity_generation_trigger.previous_reading_review_id,
                review.ambiguity_generation_trigger.generation_attempts,
                review.ambiguity_generation_trigger.last_errors,
            ),
            "downstream_status": review.downstream_status,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_plain_reading_lexical_preservation(
    source_text: str,
    review: PlainReadingReview,
    *,
    lexical_contract: object | None = None,
) -> list[str]:
    """Conservatively check AP-like tokens, numbers, and explicit negation cues."""

    return [
        issue.message
        for issue in validate_plain_reading_lexical_issues(
            source_text,
            review,
            lexical_contract=lexical_contract,
        )
    ]


def validate_plain_reading_lexical_issues(
    source_text: str,
    review: PlainReadingReview,
    *,
    lexical_contract: object | None = None,
) -> list[ValidationIssue]:
    """Return lexical preservation failures with stable codes and exact paths."""

    contract_source_hash = str(
        getattr(lexical_contract, "source_hash", "")
    )
    if lexical_contract is not None and contract_source_hash != source_hash(
        source_text
    ):
        return [
            ValidationIssue(
                code="PLAIN_READING_LEXICAL_CONTRACT_STALE",
                stage="plain_reading",
                path="/",
                observed=contract_source_hash,
                expected={"source_hash": source_hash(source_text)},
                repair_class=RepairClass.HUMAN_REQUIRED,
                allowed_paths=(),
                message=(
                    "Plain Reading lexical contract belongs to a different "
                    "effective source"
                ),
            )
        ]
    source_identifiers = set(
        str(item)
        for item in getattr(lexical_contract, "ap_identifiers", ())
        if str(item)
    )
    if lexical_contract is not None:
        contract_quantities = tuple(
            item
            for item in getattr(
                lexical_contract,
                "quantity_evidence",
                (),
            )
            if isinstance(item, (tuple, list)) and len(item) == 2
        )
        source_numbers = {
            str(item[1]) for item in contract_quantities
        }
        source_quantity_evidence = [
            {
                "surface": str(item[0]),
                "canonical_value": str(item[1]),
                "kind": "contract",
            }
            for item in contract_quantities
        ]
        negation_terms = {
            str(item).casefold()
            for item in getattr(
                lexical_contract,
                "explicit_negation_terms",
                (),
            )
            if str(item)
        }
    else:
        source_numbers = _quantity_values(source_text)
        source_quantity_evidence = _quantity_evidence(source_text)
        negation_terms = _explicit_negation_terms(source_text)
    issues: list[ValidationIssue] = []
    for ambiguity_index, ambiguity in enumerate(review.ambiguities):
        option_number_sets = [
            _quantity_values(option.plain_interpretation)
            for option in ambiguity.options
        ]
        option_extra_number_sets = [
            values - source_numbers for values in option_number_sets
        ]
        common_extra_numbers = (
            set.intersection(*option_extra_number_sets)
            if option_extra_number_sets
            else set()
        )
        quantity_axis = _is_quantity_ambiguity_axis(
            ambiguity,
            source_numbers=source_numbers,
        )
        candidate_axis_numbers = set().union(*option_extra_number_sets)
        candidate_axis_numbers -= common_extra_numbers
        for option_index, option in enumerate(ambiguity.options):
            option_text = option.plain_interpretation.casefold()
            option_identifiers = {
                identifier
                for identifier in source_identifiers
                if re.search(
                    r"(?<![A-Za-z0-9_])"
                    + re.escape(identifier)
                    + r"(?![A-Za-z0-9_])",
                    option_text,
                    flags=re.IGNORECASE,
                )
            }
            missing_identifiers = source_identifiers - option_identifiers
            option_numbers = option_number_sets[option_index]
            missing_numbers = source_numbers - option_numbers
            # AP identity is supplied solely by PlainReadingLexicalContract.
            # Natural-language words absent from that contract are not promoted
            # to an exact AP catalog by this validator.
            extra_identifiers: set[str] = set()
            extra_numbers = option_numbers - source_numbers
            logical_explanation_numbers: set[str] = set()
            if (
                "1" in extra_numbers
                and re.search(r"(?:\bor\b|\|)", source_text, re.IGNORECASE)
                and re.search(r"\b(?:at\s+least\s+one|one\s+or\s+more|any\s+one)\b", option_text, re.IGNORECASE)
                and not re.search(r"\b(?:exactly|at\s+most|no\s+more\s+than)\s+one\b", option_text, re.IGNORECASE)
            ):
                # "at least one" can be a plain-language expansion of an
                # explicit disjunction, not a new business threshold. Only this
                # tightly scoped token is ignored here; structural grounding,
                # distinctness and source support still decide acceptance.
                logical_explanation_numbers.add("1")
                extra_numbers.remove("1")
            if quantity_axis:
                # Values that distinguish the two candidate readings belong to
                # the proposed ambiguity axis and require grounding review.
                # Values repeated by every option are axis-independent lexical
                # additions and remain hard preservation failures.
                extra_numbers &= common_extra_numbers
            option_negation_terms = _explicit_negation_terms(option_text)
            missing_negation = negation_terms - option_negation_terms
            path = json_pointer(
                "ambiguities",
                ambiguity_index,
                "options",
                option_index,
                "plain_interpretation",
            )
            findings = (
                (
                    "OPTION_OMITS_AP_TOKENS",
                    "omits AP-like tokens",
                    missing_identifiers,
                    source_identifiers,
                ),
                (
                    "OPTION_OMITS_NUMBERS",
                    "omits numbers",
                    missing_numbers,
                    source_numbers,
                ),
                (
                    "OPTION_ADDS_AP_TOKENS",
                    "adds AP-like tokens",
                    extra_identifiers,
                    source_identifiers,
                ),
                (
                    "OPTION_ADDS_NUMBERS",
                    "adds numbers",
                    extra_numbers,
                    source_numbers,
                ),
                (
                    "OPTION_OMITS_NEGATION",
                    "omits negation cues",
                    missing_negation,
                    negation_terms,
                ),
            )
            for code, label, values, allowed_values in findings:
                if not values:
                    continue
                ordered_values = _sorted_lexical_values(values, numeric="NUMBERS" in code)
                message = (
                    f"{ambiguity.ambiguity_id}/{option.option_id} {label}: "
                    + ", ".join(ordered_values)
                )
                if "OMITS" in code:
                    expected = {
                        "required_source_tokens": ordered_values,
                        "allowed_source_tokens": _sorted_lexical_values(
                            allowed_values,
                            numeric="NUMBERS" in code,
                        ),
                    }
                else:
                    expected = {
                        "allowed_source_tokens": _sorted_lexical_values(
                            allowed_values,
                            numeric="NUMBERS" in code,
                        ),
                        "remove_or_ground_tokens": ordered_values,
                    }
                expected["source_text"] = source_text
                if "NUMBERS" in code:
                    expected["source_quantities"] = source_quantity_evidence
                issues.append(
                    ValidationIssue(
                        code=code,
                        stage="plain_reading",
                        path=path,
                        observed=option.plain_interpretation,
                        expected=expected,
                        context={
                            "ambiguity_id": ambiguity.ambiguity_id,
                            "option_id": option.option_id,
                            "violating_tokens": ordered_values,
                            "violation_count": len(values),
                            "validated_field_scope": "plain_interpretation_only",
                            "quantity_axis_candidate": quantity_axis,
                            "candidate_axis_quantities": _sorted_lexical_values(
                                candidate_axis_numbers,
                                numeric=True,
                            ),
                            "logical_disjunction_explanation_quantities": _sorted_lexical_values(
                                logical_explanation_numbers,
                                numeric=True,
                            ),
                        },
                        repair_class=RepairClass.CONSTRAINED,
                        allowed_paths=(path,),
                        message=message,
                    )
                )
    return issues


def _canonical_quantity_value(value: str) -> str:
    if "." not in value:
        return str(int(value))
    whole, fraction = value.split(".", 1)
    fraction = fraction.rstrip("0")
    return str(int(whole)) + (f".{fraction}" if fraction else "")


def _quantity_evidence(value: str) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for match in _DIGIT_QUANTITY_RE.finditer(value):
        if _is_metalinguistic_quantity(value, match.start(), match.end()):
            continue
        surface = match.group(0)
        evidence.append(
            {
                "surface": surface,
                "canonical_value": _canonical_quantity_value(match.group(1)),
                "kind": "ordinal" if surface.casefold().endswith(("st", "nd", "rd", "th")) else "numeric",
            }
        )
    for match in _QUANTITY_WORD_RE.finditer(value):
        if _is_metalinguistic_quantity(value, match.start(), match.end()):
            continue
        surface = match.group(0)
        lowered = surface.casefold()
        evidence.append(
            {
                "surface": surface,
                "canonical_value": _QUANTITY_WORD_VALUES[lowered],
                "kind": "word",
            }
        )
    return evidence


def _is_metalinguistic_quantity(value: str, start: int, end: int) -> bool:
    labels = r"(?:reading|interpretation|option|alternative)"
    return bool(
        re.match(rf"\s+{labels}\b", value[end:], re.IGNORECASE)
        or re.search(rf"\b{labels}\s*$", value[:start], re.IGNORECASE)
    )


def _quantity_values(value: str) -> set[str]:
    return {item["canonical_value"] for item in _quantity_evidence(value)}


def _is_quantity_ambiguity_axis(
    ambiguity: PlainReadingAmbiguity,
    *,
    source_numbers: set[str],
) -> bool:
    if not source_numbers:
        return False
    axis_text = " ".join(
        (
            ambiguity.question,
            *(option.difference_summary for option in ambiguity.options),
        )
    )
    return bool(_QUANTITY_AXIS_CUE_RE.search(axis_text))


def _sorted_lexical_values(values: Sequence[str] | set[str], *, numeric: bool) -> list[str]:
    if not numeric:
        return sorted(values)

    def key(value: str) -> tuple[float, str]:
        try:
            return float(value), value
        except ValueError:
            return float("inf"), value

    return sorted(values, key=key)


def plain_reading_from_mapping(
    payload: Mapping[str, Any],
    segmentation: SurfaceSegmentation | None = None,
) -> PlainReadingReview:
    """Load a persisted review and reject stale/unknown segment references."""

    original_schema_version = str(payload.get("schema_version", "1"))
    if original_schema_version not in {"1", "2", "3", "4", PLAIN_READING_SCHEMA_VERSION}:
        raise PlainReadingValidationError(
            "unsupported plain reading schema_version=" + original_schema_version
        )
    generator_payload = payload.get("generator", {})
    if not isinstance(generator_payload, Mapping):
        raise PlainReadingValidationError("generator must be an object")
    ambiguities: list[PlainReadingAmbiguity] = []
    for ambiguity_payload in payload.get("ambiguities", ()):
        if not isinstance(ambiguity_payload, Mapping):
            raise PlainReadingValidationError("ambiguities entries must be objects")
        options = tuple(
            PlainReadingOption(
                option_id=str(option.get("option_id", "")),
                plain_interpretation=str(option.get("plain_interpretation", "")),
                difference_summary=str(option.get("difference_summary", "")),
            )
            for option in ambiguity_payload.get("options", ())
            if isinstance(option, Mapping)
        )
        ambiguities.append(
            PlainReadingAmbiguity(
                ambiguity_id=str(ambiguity_payload.get("ambiguity_id", "")),
                source_segment_ids=tuple(
                    str(item)
                    for item in ambiguity_payload.get("source_segment_ids", ())
                ),
                question=str(ambiguity_payload.get("question", "")),
                options=options,
                source_span=_plain_reading_source_span(
                    ambiguity_payload.get("source_span", (-1, -1))
                ),
                evidence_text=str(ambiguity_payload.get("evidence_text", "")),
                decision=str(ambiguity_payload.get("decision", "")),
                decision_status=str(
                    ambiguity_payload.get("decision_status", "pending_plain_review")
                ),
                review_basis=str(ambiguity_payload.get("review_basis", "")),
                reviewer_id=str(ambiguity_payload.get("reviewer_id", "")),
                reviewer_role=str(
                    ambiguity_payload.get("reviewer_role", "ordinary_reviewer")
                ),
                reviewer_source=str(
                    ambiguity_payload.get("reviewer_source", "")
                ),
                semantic_authority=str(
                    ambiguity_payload.get("semantic_authority", "non_authoritative")
                ),
                comments=str(ambiguity_payload.get("comments", "")),
            )
        )
    revision_requests = tuple(
        PlainReadingRevisionRequest(
            request_id=str(item.get("request_id", "")),
            reading_review_id=str(item.get("reading_review_id", "")),
            ambiguity_id=str(item.get("ambiguity_id", "")),
            segmentation_id=str(item.get("segmentation_id", payload.get("segmentation_id", ""))),
            source_hash=str(item.get("source_hash", payload.get("source_hash", ""))),
            proposed_interpretation=str(item.get("proposed_interpretation", "")),
            reason=str(item.get("reason", "")),
            source_segment_ids=tuple(str(value) for value in item.get("source_segment_ids", ())),
            reviewer_id=str(item.get("reviewer_id", "")),
            reviewer_role=str(item.get("reviewer_role", "ordinary_reviewer")),
            reviewer_source=str(item.get("reviewer_source", "human")),
            validation_status=str(item.get("validation_status", "pending_validation")),
            classification=str(item.get("classification", "unresolved")),
            classification_basis=str(item.get("classification_basis", "")),
            human_confirmed=bool(item.get("human_confirmed", False)),
            request_status=str(
                item.get(
                    "request_status",
                    _legacy_revision_request_status(item, payload),
                )
            ),
            source_warnings=tuple(str(value) for value in item.get("source_warnings", ())),
            parent_decision_fingerprint=str(item.get("parent_decision_fingerprint", "")),
            confirmed_by=str(item.get("confirmed_by", "")),
            confirmed_role=str(item.get("confirmed_role", "")),
            confirmed_at=str(item.get("confirmed_at", "")),
            applied_at=str(
                item.get("applied_at", "")
                or _legacy_revision_applied_at(item, payload)
            ),
            revision_round=int(item.get("revision_round", 1)),
            created_at=str(item.get("created_at", "")),
            updated_at=str(item.get("updated_at", "")),
        )
        for item in payload.get("revision_requests", ())
        if isinstance(item, Mapping)
    )
    history = tuple(
        dict(item)
        for item in payload.get("review_history", ())
        if isinstance(item, Mapping)
    )
    if original_schema_version in {"1", "2", "3", "4"}:
        history = (
            *history,
            {
                "operation": "schema_migration_read_only",
                "from_schema_version": original_schema_version,
                "to_schema_version": PLAIN_READING_SCHEMA_VERSION,
                "migrated_at": _timestamp(),
            },
        )
    confirmation_payload = payload.get("no_ambiguity_confirmation", {})
    if not isinstance(confirmation_payload, Mapping):
        raise PlainReadingValidationError("no_ambiguity_confirmation must be an object")
    trigger_payload = payload.get("ambiguity_generation_trigger", {})
    if not isinstance(trigger_payload, Mapping):
        raise PlainReadingValidationError("ambiguity_generation_trigger must be an object")
    overall_status = str(payload.get("overall_status", "pending_detection"))
    legacy_unconfirmed_empty = (
        original_schema_version in {"1", "2", "3"}
        and overall_status == "no_ambiguity"
        and not ambiguities
        and str(payload.get("review_basis", "")) not in {
            "curated_gold",
            "human_confirmed_no_ambiguity",
            "plain_reading_disabled",
        }
        and str(payload.get("reviewer_source", "")) not in {
            "human",
            "warmup_fixture",
        }
    )
    if legacy_unconfirmed_empty:
        overall_status = "pending_no_ambiguity_confirmation"
        confirmation = NoAmbiguityConfirmation(
            status="pending",
            llm_proposed_no_ambiguity=True,
        )
        history = (
            *history,
            {
                "operation": "legacy_unconfirmed_no_ambiguity_requires_confirmation",
                "from_schema_version": original_schema_version,
                "migrated_at": _timestamp(),
            },
        )
    else:
        confirmation = NoAmbiguityConfirmation(
            status=str(confirmation_payload.get("status", "not_required")),
            llm_proposed_no_ambiguity=bool(
                confirmation_payload.get("llm_proposed_no_ambiguity", False)
            ),
            human_confirmed=bool(confirmation_payload.get("human_confirmed", False)),
            judge_confirmed=bool(confirmation_payload.get("judge_confirmed", False)),
            judge_evidence_json=str(confirmation_payload.get("judge_evidence_json", "")),
            reviewer_id=str(confirmation_payload.get("reviewer_id", "")),
            reviewer_role=str(
                confirmation_payload.get("reviewer_role", "ordinary_reviewer")
            ),
            reviewer_source=str(confirmation_payload.get("reviewer_source", "")),
            comment=str(confirmation_payload.get("comment", "")),
            confirmed_at=str(confirmation_payload.get("confirmed_at", "")),
        )
    trigger = AmbiguityGenerationTrigger(
        source=str(trigger_payload.get("source", "")),
        suspected_segment_ids=tuple(
            str(item) for item in trigger_payload.get("suspected_segment_ids", ())
        ),
        user_comment=str(trigger_payload.get("user_comment", "")),
        previous_reading_review_id=str(
            trigger_payload.get("previous_reading_review_id", "")
        ),
        requested_at=str(trigger_payload.get("requested_at", "")),
        generation_attempts=int(trigger_payload.get("generation_attempts", 0)),
        last_errors=tuple(str(item) for item in trigger_payload.get("last_errors", ())),
    )
    review = PlainReadingReview(
        reading_review_id=str(payload.get("reading_review_id", "")),
        requirement_id=str(payload.get("requirement_id", "")),
        source_hash=str(payload.get("source_hash", "")),
        segmentation_id=str(payload.get("segmentation_id", "")),
        ambiguities=tuple(ambiguities),
        revision_requests=revision_requests,
        no_ambiguity_confirmation=confirmation,
        ambiguity_generation_trigger=trigger,
        schema_version=PLAIN_READING_SCHEMA_VERSION,
        overall_status=overall_status,
        review_basis=str(payload.get("review_basis", "")),
        reviewer_source=str(payload.get("reviewer_source", "")),
        semantic_authority=str(
            payload.get("semantic_authority", "non_authoritative")
        ),
        generator=ArtifactGenerator(**dict(generator_payload)),
        experiment_mode=str(payload.get("experiment_mode", "off")),
        data_split=str(payload.get("data_split", "test")),
        persistent_rag_allowed=bool(payload.get("persistent_rag_allowed", False)),
        blocked_stage=str(payload.get("blocked_stage", "")),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
        review_history=history,
        downstream_status=str(payload.get("downstream_status", "current")),
        supersedes_reading_review_id=str(
            payload.get("supersedes_reading_review_id", "")
        ),
        contract_version=str(payload.get("contract_version", "plain-reading-contract-v5")),
        canonicalizer_version=str(payload.get("canonicalizer_version", "plain-reading-canonicalizer-v1")),
        validator_version=str(payload.get("validator_version", "plain-reading-validator-v5")),
        repair_policy_version=str(payload.get("repair_policy_version", "validated-generation-repair-policy-v1")),
        artifact_hash=str(payload.get("artifact_hash", "")),
        issue_fingerprint=str(payload.get("issue_fingerprint", "")),
        repair_history=tuple(
            dict(item) for item in payload.get("repair_history", ()) if isinstance(item, Mapping)
        ),
    )
    if segmentation is not None:
        validate_plain_reading_review(review, segmentation)
    return review


def _plain_reading_source_span(value: object) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return (-1, -1)
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return (-1, -1)


def plain_reading_to_mapping(review: PlainReadingReview) -> dict[str, Any]:
    return asdict(review)


def _reading_content_fingerprint(
    segmentation_id: str,
    ambiguities: Sequence[PlainReadingAmbiguity],
    overall_status: str,
    *,
    revision_requests: Sequence[PlainReadingRevisionRequest] = (),
    no_ambiguity_confirmation: NoAmbiguityConfirmation | None = None,
    ambiguity_generation_trigger: AmbiguityGenerationTrigger | None = None,
) -> str:
    confirmation = no_ambiguity_confirmation or NoAmbiguityConfirmation()
    trigger = ambiguity_generation_trigger or AmbiguityGenerationTrigger()
    raw = json.dumps(
        {
            "segmentation_id": segmentation_id,
            "overall_status": overall_status,
            "axes": [
                (
                    ambiguity.ambiguity_id,
                    ambiguity.source_segment_ids,
                    ambiguity.source_span,
                    ambiguity.evidence_text,
                    ambiguity.question,
                    tuple(
                        (
                            option.option_id,
                            option.plain_interpretation,
                            option.difference_summary,
                        )
                        for option in ambiguity.options
                    ),
                    ambiguity.decision_status,
                    ambiguity.decision,
                )
                for ambiguity in ambiguities
            ],
            "revision_requests": [
                (
                    request.request_id,
                    request.ambiguity_id,
                    _normalize_option(request.proposed_interpretation),
                    request.validation_status,
                    request.classification,
                    request.classification_basis,
                    request.human_confirmed,
                    request.request_status,
                    request.source_warnings,
                    request.parent_decision_fingerprint,
                    request.revision_round,
                )
                for request in revision_requests
            ],
            "no_ambiguity_confirmation": (
                confirmation.status,
                confirmation.llm_proposed_no_ambiguity,
                confirmation.human_confirmed,
                _normalize_option(confirmation.comment),
            ) + ((confirmation.judge_confirmed, confirmation.judge_evidence_json) if confirmation.judge_evidence_json else ()),
            "ambiguity_generation_trigger": (
                trigger.source,
                trigger.suspected_segment_ids,
                _normalize_option(trigger.user_comment),
                trigger.previous_reading_review_id,
                trigger.generation_attempts,
                trigger.last_errors,
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def _normalize_option(value: str) -> str:
    return " ".join(value.split()).casefold()


def _legacy_revision_request_status(
    item: Mapping[str, Any],
    review_payload: Mapping[str, Any],
) -> str:
    request_id = str(item.get("request_id", ""))
    if item.get("applied_at") or any(
        isinstance(history, Mapping)
        and history.get("operation") == "plain_reading_revision_applied"
        and str(history.get("revision_request_id", "")) == request_id
        for history in review_payload.get("review_history", ())
    ):
        return "applied"
    if item.get("validation_status") == "rejected" and item.get("human_confirmed"):
        return "rejected"
    if item.get("validation_status") == "accepted" and item.get("human_confirmed"):
        return "confirmed"
    return "submitted"


def _legacy_revision_applied_at(
    item: Mapping[str, Any],
    review_payload: Mapping[str, Any],
) -> str:
    request_id = str(item.get("request_id", ""))
    return next(
        (
            str(history.get("completed_at", ""))
            for history in review_payload.get("review_history", ())
            if isinstance(history, Mapping)
            and history.get("operation") == "plain_reading_revision_applied"
            and str(history.get("revision_request_id", "")) == request_id
        ),
        "",
    )


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
