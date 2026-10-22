"""Independent Markdown contract for activated replacement Pre-IR review.

The objects in this module are deliberately draft-only.  A draft may contain
an LLM segmentation and Plain Reading candidates, but downstream code must only
consume the normal reviewed checkpoint directories after the batch application
orchestrator has promoted an explicit human decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from plain_reading import plain_reading_from_mapping
from schemas import PlainReadingReview, SurfaceSegmentation
from surface_segmentation import (
    derive_segments,
    segmentation_from_mapping,
    source_hash,
    validate_marked_text,
)


REPLACEMENT_REVIEW_TYPE = "replacement_surface_adjudication"
REPLACEMENT_REVIEW_SCHEMA_VERSION = "1"
REPLACEMENT_ACTION_CONTRACT_VERSION = "replacement-review-v1"
REPLACEMENT_DRAFT_SCHEMA_VERSION = "1"
REPLACEMENT_DRAFTS_DIR = "replacement_review_drafts"

_REQUIREMENT_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]*-\d+")
_TRUE = {"yes", "true", "是", "y", "1"}
_FALSE = {"no", "false", "否", "n", "0", ""}


class ReplacementReviewInputError(ValueError):
    """The submitted replacement Markdown failed side-effect-free preflight."""


@dataclass(frozen=True)
class ReplacementReviewDecision:
    """Typed decision shared by Markdown and terminal input adapters.

    ``cut_offsets`` is always the authoritative surface representation.  The
    optional ``marked_text`` is retained for audit and is mechanically reduced
    to the same offsets by the parser.
    """

    surface_action: str
    cut_offsets: tuple[int, ...] = ()
    marked_text: str = ""
    ambiguity_flagged: bool = False
    plain_reading_decisions: tuple[tuple[str, str], ...] = ()
    revision_requests: tuple[tuple[str, str], ...] = ()
    revision_actions: tuple[tuple[str, str, str], ...] = ()
    revision_candidate_action: str = ""
    acknowledge_source_warnings: bool = False
    no_ambiguity_action: str = ""
    comment: str = ""
    defer: bool = False

    @property
    def decisions(self) -> dict[str, str]:
        return dict(self.plain_reading_decisions)

    @property
    def revisions(self) -> dict[str, str]:
        return dict(self.revision_requests)

    @property
    def revision_action_map(self) -> dict[str, tuple[str, str]]:
        return {
            request_id: (action, basis)
            for request_id, action, basis in self.revision_actions
        }


@dataclass(frozen=True)
class ReplacementReviewDraft:
    requirement_id: str
    original_requirement: str
    effective_replacement: str
    original_source_hash: str
    source_hash: str
    active_override_id: str
    generation_contract: dict[str, Any]
    generated_segmentation: SurfaceSegmentation | None = None
    proposed_segmentation: SurfaceSegmentation | None = None
    plain_reading_review: PlainReadingReview | None = None
    revision_candidate_review: PlainReadingReview | None = None
    revision_candidate_request_id: str = ""
    revision_candidate_warnings: tuple[str, ...] = ()
    diagnostics: tuple[dict[str, Any], ...] = ()
    created_at: str = ""
    draft_fingerprint: str = ""

    @property
    def generation_succeeded(self) -> bool:
        return (
            self.generated_segmentation is not None
            and self.proposed_segmentation is not None
            and self.plain_reading_review is not None
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": REPLACEMENT_DRAFT_SCHEMA_VERSION,
            "action_contract_version": REPLACEMENT_ACTION_CONTRACT_VERSION,
            "requirement_id": self.requirement_id,
            "original_requirement": self.original_requirement,
            "effective_replacement": self.effective_replacement,
            "original_source_hash": self.original_source_hash,
            "source_hash": self.source_hash,
            "active_override_id": self.active_override_id,
            "generation_contract": dict(self.generation_contract),
            "generated_segmentation": (
                asdict(self.generated_segmentation)
                if self.generated_segmentation is not None
                else None
            ),
            "proposed_segmentation": (
                asdict(self.proposed_segmentation)
                if self.proposed_segmentation is not None
                else None
            ),
            "plain_reading_review": (
                asdict(self.plain_reading_review)
                if self.plain_reading_review is not None
                else None
            ),
            "revision_candidate_review": (
                asdict(self.revision_candidate_review)
                if self.revision_candidate_review is not None
                else None
            ),
            "revision_candidate_request_id": self.revision_candidate_request_id,
            "revision_candidate_warnings": list(self.revision_candidate_warnings),
            "diagnostics": [dict(item) for item in self.diagnostics],
            "created_at": self.created_at,
        }
        payload["draft_fingerprint"] = replacement_draft_fingerprint(payload)
        return payload


def replacement_draft_path(
    checkpoint_dir: str | Path, requirement_id: str
) -> Path:
    normalized = str(requirement_id).strip().upper()
    if _REQUIREMENT_ID.fullmatch(normalized) is None:
        raise ValueError("invalid replacement draft requirement_id")
    return Path(checkpoint_dir) / REPLACEMENT_DRAFTS_DIR / f"{normalized}.json"


def replacement_draft_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = {
        str(key): value
        for key, value in payload.items()
        if str(key) != "draft_fingerprint"
    }
    raw = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_replacement_review_draft(
    checkpoint_dir: str | Path, draft: ReplacementReviewDraft
) -> ReplacementReviewDraft:
    payload = draft.to_payload()
    path = replacement_draft_path(checkpoint_dir, draft.requirement_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return replacement_review_draft_from_mapping(payload)


def load_replacement_review_draft(
    checkpoint_dir: str | Path, requirement_id: str
) -> ReplacementReviewDraft | None:
    path = replacement_draft_path(checkpoint_dir, requirement_id)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("replacement review draft must be a JSON object")
    return replacement_review_draft_from_mapping(payload)


def replacement_review_draft_from_mapping(
    payload: Mapping[str, Any],
) -> ReplacementReviewDraft:
    if str(payload.get("schema_version", "")) != REPLACEMENT_DRAFT_SCHEMA_VERSION:
        raise ValueError("unsupported replacement review draft schema")
    if (
        str(payload.get("action_contract_version", ""))
        != REPLACEMENT_ACTION_CONTRACT_VERSION
    ):
        raise ValueError("unsupported replacement review action contract")
    expected_fingerprint = replacement_draft_fingerprint(payload)
    if str(payload.get("draft_fingerprint", "")) != expected_fingerprint:
        raise ValueError("replacement review draft fingerprint mismatch")

    requirement_id = str(payload.get("requirement_id", "")).strip().upper()
    if _REQUIREMENT_ID.fullmatch(requirement_id) is None:
        raise ValueError("invalid replacement review draft requirement_id")
    original = payload.get("original_requirement")
    effective = payload.get("effective_replacement")
    if not isinstance(original, str) or not isinstance(effective, str) or not effective:
        raise ValueError("replacement review draft source text is missing")
    if str(payload.get("original_source_hash", "")) != source_hash(original):
        raise ValueError("replacement review draft original source hash mismatch")
    if str(payload.get("source_hash", "")) != source_hash(effective):
        raise ValueError("replacement review draft effective source hash mismatch")

    generated = _segmentation_or_none(payload.get("generated_segmentation"))
    proposed = _segmentation_or_none(payload.get("proposed_segmentation"))
    review_payload = payload.get("plain_reading_review")
    review: PlainReadingReview | None = None
    if review_payload is not None:
        if not isinstance(review_payload, Mapping) or proposed is None:
            raise ValueError("replacement draft Plain Reading parent is missing")
        review = plain_reading_from_mapping(review_payload, proposed)
    candidate_payload = payload.get("revision_candidate_review")
    candidate_review: PlainReadingReview | None = None
    if candidate_payload is not None:
        if not isinstance(candidate_payload, Mapping) or proposed is None:
            raise ValueError("replacement draft revision candidate parent is missing")
        candidate_review = plain_reading_from_mapping(candidate_payload, proposed)

    artifacts = (generated, proposed, review)
    if any(item is None for item in artifacts) and not all(
        item is None for item in artifacts
    ):
        raise ValueError("replacement draft artifact chain is incomplete")
    if generated is not None:
        if (
            generated.requirement_id != requirement_id
            or generated.source_text != effective
            or generated.source_hash != source_hash(effective)
        ):
            raise ValueError("generated replacement segmentation is stale")
        assert proposed is not None and review is not None
        if (
            proposed.requirement_id != requirement_id
            or proposed.source_text != effective
            or proposed.source_hash != generated.source_hash
            or proposed.status not in {"surface_reviewed", "ambiguity_flagged"}
            or review.requirement_id != requirement_id
            or review.source_hash != proposed.source_hash
            or review.segmentation_id != proposed.segmentation_id
        ):
            raise ValueError("replacement draft reviewed artifact chain is stale")

    generation_contract = payload.get("generation_contract", {})
    if not isinstance(generation_contract, Mapping):
        raise ValueError("replacement draft generation contract is malformed")
    diagnostics_value = payload.get("diagnostics", ())
    if not isinstance(diagnostics_value, Sequence) or isinstance(
        diagnostics_value, (str, bytes)
    ):
        raise ValueError("replacement draft diagnostics are malformed")
    diagnostics = tuple(
        dict(item) for item in diagnostics_value if isinstance(item, Mapping)
    )
    candidate_request_id = str(
        payload.get("revision_candidate_request_id", "")
    )
    warnings_value = payload.get("revision_candidate_warnings", ())
    if not isinstance(warnings_value, Sequence) or isinstance(
        warnings_value, (str, bytes)
    ):
        raise ValueError("replacement draft revision warnings are malformed")
    candidate_warnings = tuple(str(item) for item in warnings_value)
    if candidate_review is None:
        if candidate_request_id or candidate_warnings:
            raise ValueError("replacement draft revision candidate metadata is orphaned")
    elif (
        not candidate_request_id
        or candidate_review.requirement_id != requirement_id
        or candidate_review.source_hash != proposed.source_hash
        or candidate_review.segmentation_id != proposed.segmentation_id
    ):
        raise ValueError("replacement draft revision candidate is stale")
    return ReplacementReviewDraft(
        requirement_id=requirement_id,
        original_requirement=original,
        effective_replacement=effective,
        original_source_hash=str(payload.get("original_source_hash", "")),
        source_hash=str(payload.get("source_hash", "")),
        active_override_id=str(payload.get("active_override_id", "")),
        generation_contract=dict(generation_contract),
        generated_segmentation=generated,
        proposed_segmentation=proposed,
        plain_reading_review=review,
        revision_candidate_review=candidate_review,
        revision_candidate_request_id=candidate_request_id,
        revision_candidate_warnings=candidate_warnings,
        diagnostics=diagnostics,
        created_at=str(payload.get("created_at", "")),
        draft_fingerprint=expected_fingerprint,
    )


def replacement_review_snapshot_id(
    *,
    checkpoint_fingerprint: str,
    round_number: int,
    drafts: Sequence[ReplacementReviewDraft],
) -> str:
    payload = {
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "round": int(round_number),
        "items": [
            {
                "requirement_id": item.requirement_id,
                "source_hash": item.source_hash,
                "active_override_id": item.active_override_id,
                "draft_fingerprint": item.draft_fingerprint,
            }
            for item in sorted(drafts, key=lambda value: value.requirement_id)
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def format_replacement_review_markdown(
    *,
    checkpoint_fingerprint: str,
    round_number: int,
    snapshot_id: str,
    drafts: Sequence[ReplacementReviewDraft],
) -> str:
    lines = [
        "# Replacement Pre-IR 批量审核",
        "",
        f"review_type: {REPLACEMENT_REVIEW_TYPE}",
        f"schema_version: {REPLACEMENT_REVIEW_SCHEMA_VERSION}",
        f"round: {round_number}",
        f"checkpoint_fingerprint: {checkpoint_fingerprint}",
        f"snapshot_id: {snapshot_id}",
        "review_complete: no",
        "",
        "将每项的 surface_action 和 Plain Reading 决定填写完整；暂不处理请填写 defer: yes。",
        "不要修改 metadata comment 中的 ID、hash 或版本。",
        "",
    ]
    for draft in sorted(drafts, key=lambda value: value.requirement_id):
        lines.extend(_format_replacement_item(draft))
    return "\n".join(lines).rstrip() + "\n"


def parse_replacement_review_markdown(
    text: str,
    *,
    checkpoint_fingerprint: str,
    round_number: int,
    snapshot_id: str,
    drafts: Sequence[ReplacementReviewDraft],
) -> dict[str, ReplacementReviewDecision]:
    """Validate an entire document and return typed decisions.

    No callback or write is performed here, making this function suitable for
    the mandatory all-file preflight before item transactions begin.
    """

    errors: list[str] = []
    expected_fields = {
        "review_type": REPLACEMENT_REVIEW_TYPE,
        "schema_version": REPLACEMENT_REVIEW_SCHEMA_VERSION,
        "round": str(round_number),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "snapshot_id": snapshot_id,
    }
    for field, expected in expected_fields.items():
        try:
            actual = _read_field(text, field)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if actual != expected:
            errors.append(f"{field} 已过期、缺失或被修改。")
    try:
        complete = _read_field(text, "review_complete").casefold()
    except ValueError as exc:
        errors.append(str(exc))
        complete = ""
    if complete not in _TRUE:
        errors.append("请检查整份文件后把 review_complete 改为 yes。")

    structural = _without_fenced_code_content(text)
    matches = list(
        re.finditer(
            r"^##\s+([A-Za-z][A-Za-z0-9_-]*-\d+)\s*$",
            structural,
            re.MULTILINE,
        )
    )
    headings = [match.group(1).upper() for match in matches]
    duplicates = sorted({value for value in headings if headings.count(value) > 1})
    expected_ids = [item.requirement_id for item in sorted(drafts, key=lambda x: x.requirement_id)]
    if duplicates:
        errors.append("存在重复 requirement 区块: " + ", ".join(duplicates))
    if set(headings) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(headings))
        extra = sorted(set(headings) - set(expected_ids))
        if missing:
            errors.append("缺少待审核区块: " + ", ".join(missing))
        if extra:
            errors.append("包含非当前待审核区块: " + ", ".join(extra))
    if errors:
        raise ReplacementReviewInputError("replacement 审核文件预检失败：\n" + "\n".join(errors))

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).upper()] = text[match.end() : end]

    decisions: dict[str, ReplacementReviewDecision] = {}
    item_errors: list[str] = []
    drafts_by_id = {item.requirement_id: item for item in drafts}
    for requirement_id in expected_ids:
        try:
            decisions[requirement_id] = _parse_replacement_item(
                sections[requirement_id], drafts_by_id[requirement_id]
            )
        except ValueError as exc:
            item_errors.append(f"{requirement_id}: {exc}")
    if item_errors:
        raise ReplacementReviewInputError(
            "replacement 审核文件预检失败：\n" + "\n".join(item_errors)
        )
    return decisions


def replacement_surface_decision_from_command(
    segmentation: SurfaceSegmentation,
    raw: str,
) -> ReplacementReviewDecision:
    """Translate the legacy terminal surface command into the shared model."""

    value = raw.strip()
    current_offsets = tuple(
        segment.end for segment in segmentation.segments[:-1]
    )
    if not value or value.casefold() == "a":
        return ReplacementReviewDecision("accept", current_offsets)
    if value.casefold() == "a!":
        return ReplacementReviewDecision(
            "accept", current_offsets, ambiguity_flagged=True
        )
    command, separator, payload = value.partition(" ")
    normalized = command.casefold()
    if normalized in {"e", "e!"}:
        if not separator or not payload:
            raise ValueError(
                f"{command} 后必须提供只插入 {segmentation.marker} 的完整原文"
            )
        validation = validate_marked_text(
            segmentation.source_text, payload, segmentation.marker
        )
        if validation.status != "passed":
            from surface_segmentation import SegmentationValidationError

            raise SegmentationValidationError(
                "marked_text failed lossless reconstruction: "
                + validation.failure_type,
                validation,
            )
        segments = derive_segments(
            segmentation.source_text, payload, segmentation.marker
        )
        return ReplacementReviewDecision(
            "marked_text",
            tuple(segment.end for segment in segments[:-1]),
            marked_text=payload,
            ambiguity_flagged=normalized.endswith("!"),
        )
    if normalized in {"o", "o!"}:
        if not separator or not payload.strip():
            raise ValueError(f"{command} 后必须提供至少一个 codepoint offset")
        return ReplacementReviewDecision(
            "offsets",
            _parse_terminal_offsets(payload, len(segmentation.source_text)),
            ambiguity_flagged=normalized.endswith("!"),
        )
    ambiguity = value.endswith("!")
    numeric_payload = value[:-1].strip() if ambiguity else value
    return ReplacementReviewDecision(
        "offsets",
        _parse_terminal_offsets(numeric_payload, len(segmentation.source_text)),
        ambiguity_flagged=ambiguity,
    )


def _format_replacement_item(draft: ReplacementReviewDraft) -> list[str]:
    lines = [
        f"## {draft.requirement_id}",
        "",
        "### Original requirement",
        "",
        *_fenced(draft.original_requirement),
        "",
        "### Effective replacement",
        "",
        *_fenced(draft.effective_replacement),
        "",
        "### Surface segmentation",
        "",
    ]
    if draft.generation_succeeded:
        assert draft.proposed_segmentation is not None
        proposed = draft.proposed_segmentation
        lines.extend(
            [
                "当前候选：",
                "",
                "candidate_marked_text:",
                *_fenced(_marked_text(proposed)),
                "",
                "当前片段：",
                "",
                *(
                    f"- {segment.segment_id} [{segment.start}:{segment.end}] {segment.text!r}"
                    for segment in proposed.segments
                ),
                "",
                "surface_action: pending",
                "cut_offsets:",
                "ambiguity_flagged: no",
                "marked_text:",
                *_fenced(""),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "草稿生成失败；本轮只能 defer，修复 API/输入后使用 --refresh 重新导出。",
                "",
                "surface_action: pending",
                "cut_offsets:",
                "ambiguity_flagged: no",
                "marked_text:",
                *_fenced(""),
                "",
            ]
        )

    lines.extend(["### Plain reading", ""])
    review = draft.plain_reading_review
    candidate = draft.revision_candidate_review
    if candidate is not None:
        lines.extend(
            [
                "以下是按人工 E/S/I/O 分类生成的待确认候选轴；不会自动应用。",
                "",
                f"revision_candidate_request_id: {draft.revision_candidate_request_id}",
                "",
            ]
        )
        for ambiguity in candidate.ambiguities:
            lines.extend(
                [
                    f"#### Candidate axis {ambiguity.ambiguity_id}",
                    "",
                    "question:",
                    *_fenced(ambiguity.question),
                    "",
                ]
            )
            for option in ambiguity.options:
                lines.extend(
                    [
                        f"option_{option.option_id}:",
                        *_fenced(option.plain_interpretation),
                        "",
                        f"difference_{option.option_id}:",
                        *_fenced(option.difference_summary),
                        "",
                    ]
                )
        if draft.revision_candidate_warnings:
            lines.extend(
                [
                    "source_warnings:",
                    *_fenced("\n".join(draft.revision_candidate_warnings)),
                    "",
                ]
            )
        lines.extend(
            [
                "revision_candidate_action: pending",
                "acknowledge_source_warnings: no",
                "可填写 accept 或 reject；存在 warning 时 accept 还必须明确 acknowledge=yes。",
                "",
            ]
        )
    elif review is not None and review.overall_status == "options_revision_requested":
        active_requests = [
            item
            for item in review.revision_requests
            if item.request_status in {"submitted", "suggested", "confirmed"}
        ]
        for request in active_requests:
            lines.extend(
                [
                    f"#### Revision {request.request_id}",
                    "",
                    f"ambiguity_id: {request.ambiguity_id}",
                    "proposed_interpretation:",
                    *_fenced(request.proposed_interpretation),
                    "",
                    f"revision_action_{request.request_id}: pending",
                    f"revision_basis_{request.request_id}:",
                    "可填写 E/S/I/O/B/C；除 B 外必须提供非空 basis。S/I 会先生成候选轴并进入下一轮确认。",
                    "",
                ]
            )
    elif review is not None and review.ambiguities:
        for ambiguity in review.ambiguities:
            lines.extend(
                [
                    f"#### Axis {ambiguity.ambiguity_id}",
                    "",
                    "question:",
                    *_fenced(ambiguity.question),
                    "",
                ]
            )
            for option in ambiguity.options:
                lines.extend(
                    [
                        f"option_{option.option_id}:",
                        *_fenced(option.plain_interpretation),
                        "",
                        f"difference_{option.option_id}:",
                        *_fenced(option.difference_summary),
                        "",
                    ]
                )
            lines.extend(["decision: pending", "revision_request:", ""])
    elif review is not None:
        lines.extend(
            [
                "LLM 未给出可安全区分的歧义轴。",
                "",
                "no_ambiguity_action: pending",
                "可填写 accept；request 表示请求下一轮候选轴；defer 表示暂缓。",
                "",
            ]
        )
    else:
        lines.extend(["Plain Reading 草稿不可用。", "", "no_ambiguity_action: pending", ""])
    diagnostics = json.dumps(
        [dict(item) for item in draft.diagnostics], ensure_ascii=False, indent=2
    )
    lines.extend(
        [
            "defer: no",
            "comment:",
            "",
            "### Generation diagnostics",
            "",
            *_fenced(diagnostics, language="json"),
            "",
            "<!-- traj-ltlf-meta",
            f"requirement_id: {draft.requirement_id}",
            f"original_source_hash: {draft.original_source_hash}",
            f"source_hash: {draft.source_hash}",
            f"active_override_id: {draft.active_override_id}",
            f"draft_fingerprint: {draft.draft_fingerprint}",
            f"action_contract_version: {REPLACEMENT_ACTION_CONTRACT_VERSION}",
            "-->",
            "",
        ]
    )
    return lines


def _parse_replacement_item(
    section: str, draft: ReplacementReviewDraft
) -> ReplacementReviewDecision:
    metadata_blocks = re.findall(
        r"<!--\s*traj-ltlf-meta[ \t]*\r?\n([\s\S]*?)\r?\n-->"
        , section
    )
    if len(metadata_blocks) != 1:
        raise ValueError("traj-ltlf-meta 区块缺失或重复。")
    metadata_block = metadata_blocks[0]
    metadata = {
        "requirement_id": draft.requirement_id,
        "original_source_hash": draft.original_source_hash,
        "source_hash": draft.source_hash,
        "active_override_id": draft.active_override_id,
        "draft_fingerprint": draft.draft_fingerprint,
        "action_contract_version": REPLACEMENT_ACTION_CONTRACT_VERSION,
    }
    for field, expected in metadata.items():
        if _read_field(metadata_block, field) != expected:
            raise ValueError(f"{field} 已过期、缺失或被修改。")
    if _read_fenced_after_heading(section, "Original requirement") != (
        draft.original_requirement
    ):
        raise ValueError("original requirement 展示内容被修改。")
    if _read_fenced_after_heading(section, "Effective replacement") != (
        draft.effective_replacement
    ):
        raise ValueError("effective replacement 展示内容被修改。")
    if draft.proposed_segmentation is not None and (
        _read_fenced_after_field(section, "candidate_marked_text")
        != _marked_text(draft.proposed_segmentation)
    ):
        raise ValueError("当前 segmentation 候选展示内容被修改。")

    defer = _parse_bool(_read_field(section, "defer"), field="defer")
    action_raw = _read_field(section, "surface_action").casefold()
    ambiguity_raw = _read_field(section, "ambiguity_flagged")
    ambiguity_flagged = _parse_bool(ambiguity_raw, field="ambiguity_flagged")
    comment = _read_field(section, "comment")
    review = draft.plain_reading_review

    axis_values: list[tuple[str, str]] = []
    revision_values: list[tuple[str, str]] = []
    revision_actions: list[tuple[str, str, str]] = []
    candidate_action = ""
    acknowledge_warnings = False
    candidate = draft.revision_candidate_review
    if candidate is not None:
        if _read_field(section, "revision_candidate_request_id") != (
            draft.revision_candidate_request_id
        ):
            raise ValueError("revision candidate request ID 被修改。")
        for ambiguity in candidate.ambiguities:
            axis_section = _candidate_axis_section(
                section, ambiguity.ambiguity_id
            )
            _validate_axis_display(axis_section, ambiguity)
        if draft.revision_candidate_warnings:
            if _read_fenced_after_field(section, "source_warnings") != "\n".join(
                draft.revision_candidate_warnings
            ):
                raise ValueError("revision candidate source warnings 被修改。")
        candidate_action = _read_field(
            section, "revision_candidate_action"
        ).strip().casefold()
        acknowledge_warnings = _parse_bool(
            _read_field(section, "acknowledge_source_warnings"),
            field="acknowledge_source_warnings",
        )
    elif review is not None and review.overall_status == "options_revision_requested":
        active_requests = [
            item
            for item in review.revision_requests
            if item.request_status in {"submitted", "suggested", "confirmed"}
        ]
        if not active_requests:
            raise ValueError("options_revision_requested 缺少活动 revision request。")
        for request in active_requests:
            revision_section = _revision_section(section, request.request_id)
            if _read_field(revision_section, "ambiguity_id") != request.ambiguity_id:
                raise ValueError(f"revision {request.request_id} ambiguity_id 被修改。")
            if _read_fenced_after_field(
                revision_section, "proposed_interpretation"
            ) != request.proposed_interpretation:
                raise ValueError(
                    f"revision {request.request_id} proposed interpretation 被修改。"
                )
            action = _read_field(
                revision_section, f"revision_action_{request.request_id}"
            ).strip().upper()
            basis = _read_field(
                revision_section, f"revision_basis_{request.request_id}"
            ).strip()
            revision_actions.append((request.request_id, action, basis))
    elif review is not None:
        for ambiguity in review.ambiguities:
            axis_section = _axis_section(section, ambiguity.ambiguity_id)
            _validate_axis_display(axis_section, ambiguity)
            decision = _read_field(axis_section, "decision").strip().upper()
            revision = _read_field(axis_section, "revision_request").strip()
            axis_values.append((ambiguity.ambiguity_id, decision))
            if revision:
                revision_values.append((ambiguity.ambiguity_id, revision))
    no_axis_action = _read_field(section, "no_ambiguity_action").strip().casefold()

    if defer or no_axis_action == "defer":
        if action_raw not in {"", "pending"}:
            raise ValueError("defer 项目的 surface_action 必须保持 pending。")
        if any(value not in {"", "PENDING"} for _, value in axis_values):
            raise ValueError("defer 项目的 Plain Reading 决定必须保持 pending。")
        if candidate_action not in {"", "pending"} or any(
            action not in {"", "PENDING"}
            for _, action, _ in revision_actions
        ):
            raise ValueError("defer 项目的 revision 决定必须保持 pending。")
        return ReplacementReviewDecision(
            surface_action="defer",
            no_ambiguity_action="defer",
            comment=comment,
            defer=True,
        )

    if not draft.generation_succeeded:
        raise ValueError("草稿生成失败的项目只能填写 defer: yes。")
    assert draft.proposed_segmentation is not None
    proposed = draft.proposed_segmentation
    current_offsets = tuple(segment.end for segment in proposed.segments[:-1])
    marked_text = ""
    inferred_flag = action_raw.endswith("!")
    normalized_action = action_raw.removesuffix("!")
    if normalized_action in {"a", "accept"}:
        cut_offsets = current_offsets
        surface_action = "accept"
    elif normalized_action in {"o", "offsets"}:
        cut_offsets = _parse_offsets(
            _read_field(section, "cut_offsets"), len(proposed.source_text)
        )
        surface_action = "offsets"
    elif normalized_action in {"e", "marked_text"}:
        marked_text = _read_fenced_after_field(section, "marked_text")
        validation = validate_marked_text(
            proposed.source_text, marked_text, proposed.marker
        )
        if validation.status != "passed":
            raise ValueError(
                "marked_text 不能精确重建 replacement 原文: "
                + validation.failure_type
            )
        segments = derive_segments(proposed.source_text, marked_text, proposed.marker)
        cut_offsets = tuple(segment.end for segment in segments[:-1])
        surface_action = "marked_text"
    else:
        raise ValueError("surface_action 必须填写 accept/offsets/marked_text 或 defer。")

    changed_surface = cut_offsets != current_offsets
    if review is None:
        raise ValueError("Plain Reading 草稿缺失。")
    if candidate is not None:
        if changed_surface:
            if candidate_action not in {"", "pending"}:
                raise ValueError("重切分会使 revision candidate 失效；本轮必须保持 pending。")
        elif candidate_action not in {"accept", "reject"}:
            raise ValueError("revision_candidate_action 必须填写 accept/reject 或 defer。")
        elif (
            candidate_action == "accept"
            and draft.revision_candidate_warnings
            and not acknowledge_warnings
        ):
            raise ValueError("revision candidate 存在 source warning，必须明确 acknowledge=yes。")
    elif review.overall_status == "options_revision_requested":
        allowed_revision_actions = {"E", "S", "I", "O", "B", "C"}
        if changed_surface:
            if any(
                action not in {"", "PENDING"}
                for _, action, _ in revision_actions
            ):
                raise ValueError("重切分会使 revision request 失效；本轮必须保持 pending。")
        else:
            invalid = [
                request_id
                for request_id, action, _basis in revision_actions
                if action not in allowed_revision_actions
            ]
            if invalid:
                raise ValueError(
                    "每个 revision request 必须填写 E/S/I/O/B/C: "
                    + ", ".join(invalid)
                )
            missing_basis = [
                request_id
                for request_id, action, basis in revision_actions
                if action != "B" and not basis
            ]
            if missing_basis:
                raise ValueError(
                    "revision classification 必须提供 basis: "
                    + ", ".join(missing_basis)
                )
    elif review.ambiguities:
        allowed = {"A", "B", "N", "U", "R"}
        values = {axis_id: value for axis_id, value in axis_values}
        if changed_surface:
            if any(value not in {"", "PENDING"} for value in values.values()):
                raise ValueError("重切分会使当前歧义轴失效；本轮决定必须保持 pending。")
        else:
            invalid = [
                axis_id for axis_id, value in axis_values if value not in allowed
            ]
            if invalid:
                raise ValueError(
                    "每个歧义轴必须填写 A/B/N/U/R: " + ", ".join(invalid)
                )
            revisions = dict(revision_values)
            missing_revisions = [
                axis_id
                for axis_id, value in axis_values
                if value == "R" and not revisions.get(axis_id)
            ]
            if missing_revisions:
                raise ValueError(
                    "选择 R 时必须填写 revision_request: "
                    + ", ".join(missing_revisions)
                )
            if sum(value == "R" for _, value in axis_values) > 1:
                raise ValueError("复杂 revision 每轮只能提交一个歧义轴，其他轴请留到下一轮。")
    else:
        if changed_surface:
            if no_axis_action not in {"", "pending"}:
                raise ValueError("重切分会使当前 Plain Reading 失效；本轮必须保持 pending。")
        elif no_axis_action not in {"accept", "y", "yes", "request", "r"}:
            raise ValueError("no_ambiguity_action 必须填写 accept/request 或 defer。")

    return ReplacementReviewDecision(
        surface_action=surface_action,
        cut_offsets=tuple(cut_offsets),
        marked_text=marked_text,
        ambiguity_flagged=ambiguity_flagged or inferred_flag,
        plain_reading_decisions=tuple(axis_values),
        revision_requests=tuple(revision_values),
        revision_actions=tuple(revision_actions),
        revision_candidate_action=candidate_action,
        acknowledge_source_warnings=acknowledge_warnings,
        no_ambiguity_action=(
            "accept" if no_axis_action in {"accept", "y", "yes"} else
            "request" if no_axis_action in {"request", "r"} else ""
        ),
        comment=comment,
    )


def _segmentation_or_none(value: Any) -> SurfaceSegmentation | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("replacement draft segmentation is malformed")
    return segmentation_from_mapping(value)


def _marked_text(segmentation: SurfaceSegmentation) -> str:
    return segmentation.marker.join(segment.text for segment in segmentation.segments)


def _fenced(value: str, *, language: str = "text") -> list[str]:
    fence = "```"
    while fence in value:
        fence += "`"
    return [f"{fence}{language}", value, fence]


def _without_fenced_code_content(text: str) -> str:
    def mask(match: re.Match[str]) -> str:
        value = match.group(0)
        return "".join("\n" if character == "\n" else " " for character in value)

    return re.sub(
        r"(?m)^(?P<fence>`{3,})[^\r\n]*\r?\n[\s\S]*?^\1\s*$",
        mask,
        text,
    )


def _read_field(text: str, field: str) -> str:
    matches = list(re.finditer(
        rf"^{re.escape(field)}[ \t]*:[ \t]*(.*?)[ \t]*$",
        _without_fenced_code_content(text),
        re.MULTILINE,
    ))
    if len(matches) > 1:
        raise ValueError(f"{field} 字段重复。")
    return matches[0].group(1).strip() if matches else ""


def _read_fenced_after_field(text: str, field: str) -> str:
    match = re.search(
        rf"^{re.escape(field)}\s*:\s*\r?\n(?P<fence>`{{3,}})[^\r\n]*\r?\n(?P<value>[\s\S]*?)\r?\n(?P=fence)\s*$",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"{field} 后缺少 fenced 完整原文。")
    return match.group("value")


def _read_fenced_after_heading(text: str, heading: str) -> str:
    match = re.search(
        rf"^###\s+{re.escape(heading)}\s*$\s*"
        r"(?P<fence>`{3,})[^\r\n]*\r?\n(?P<value>[\s\S]*?)\r?\n(?P=fence)\s*$",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"{heading} 后缺少 fenced 完整原文。")
    return match.group("value")


def _axis_section(section: str, ambiguity_id: str) -> str:
    structural = _without_fenced_code_content(section)
    matches = list(re.finditer(r"^####\s+Axis\s+(\S+)\s*$", structural, re.MULTILINE))
    found = [item for item in matches if item.group(1) == ambiguity_id]
    if len(found) != 1:
        raise ValueError(f"歧义轴 {ambiguity_id} 区块缺失或重复。")
    match = found[0]
    index = matches.index(match)
    end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
    return section[match.end() : end]


def _candidate_axis_section(section: str, ambiguity_id: str) -> str:
    structural = _without_fenced_code_content(section)
    matches = list(
        re.finditer(r"^####\s+Candidate axis\s+(\S+)\s*$", structural, re.MULTILINE)
    )
    found = [item for item in matches if item.group(1) == ambiguity_id]
    if len(found) != 1:
        raise ValueError(f"候选歧义轴 {ambiguity_id} 区块缺失或重复。")
    match = found[0]
    index = matches.index(match)
    end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
    return section[match.end() : end]


def _revision_section(section: str, request_id: str) -> str:
    structural = _without_fenced_code_content(section)
    matches = list(
        re.finditer(r"^####\s+Revision\s+(\S+)\s*$", structural, re.MULTILINE)
    )
    found = [item for item in matches if item.group(1) == request_id]
    if len(found) != 1:
        raise ValueError(f"revision {request_id} 区块缺失或重复。")
    match = found[0]
    index = matches.index(match)
    end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
    return section[match.end() : end]


def _validate_axis_display(axis_section: str, ambiguity: Any) -> None:
    if _read_fenced_after_field(axis_section, "question") != ambiguity.question:
        raise ValueError(f"歧义轴 {ambiguity.ambiguity_id} question 被修改。")
    for option in ambiguity.options:
        if _read_fenced_after_field(
            axis_section, f"option_{option.option_id}"
        ) != option.plain_interpretation:
            raise ValueError(
                f"歧义轴 {ambiguity.ambiguity_id} option {option.option_id} 被修改。"
            )
        if _read_fenced_after_field(
            axis_section, f"difference_{option.option_id}"
        ) != option.difference_summary:
            raise ValueError(
                f"歧义轴 {ambiguity.ambiguity_id} difference {option.option_id} 被修改。"
            )


def _parse_offsets(value: str, source_length: int) -> tuple[int, ...]:
    tokens = [item.strip() for item in value.split(",")]
    if not tokens or any(
        not token or not token.isascii() or not token.isdigit() for token in tokens
    ):
        raise ValueError("cut_offsets 必须是逗号分隔的 codepoint 整数。")
    offsets = tuple(int(token) for token in tokens)
    if list(offsets) != sorted(set(offsets)):
        raise ValueError("cut_offsets 必须严格递增且不能重复。")
    if any(offset <= 0 or offset >= source_length for offset in offsets):
        raise ValueError(f"每个 cut offset 必须满足 0 < offset < {source_length}。")
    return offsets


def _parse_terminal_offsets(value: str, source_length: int) -> tuple[int, ...]:
    try:
        return _parse_offsets(value, source_length)
    except ValueError as exc:
        if "逗号分隔" in str(exc):
            raise ValueError("请输入 a/a!，或 e/e! + 带 marker 的完整原文") from exc
        message = str(exc).replace("cut_offsets", "codepoint offsets")
        raise ValueError(message.rstrip("。")) from exc


def _parse_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(f"{field} 必须填写 yes 或 no。")


def now_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


__all__ = [
    "REPLACEMENT_ACTION_CONTRACT_VERSION",
    "REPLACEMENT_DRAFTS_DIR",
    "REPLACEMENT_REVIEW_SCHEMA_VERSION",
    "REPLACEMENT_REVIEW_TYPE",
    "ReplacementReviewDecision",
    "ReplacementReviewDraft",
    "ReplacementReviewInputError",
    "format_replacement_review_markdown",
    "load_replacement_review_draft",
    "now_timestamp",
    "parse_replacement_review_markdown",
    "replacement_surface_decision_from_command",
    "replacement_draft_path",
    "replacement_review_snapshot_id",
    "save_replacement_review_draft",
]
