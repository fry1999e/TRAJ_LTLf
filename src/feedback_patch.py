"""Reviewed, formula-free feedback patch proposals for routed reruns.

The proposal records which formal artifacts must be reopened or invalidated.
It never mutates an IR by itself: the owning pipeline stage must review and
validate the proposal before producing a new versioned artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

from error_diagnosis import ErrorDiagnosisPacket, resolve_error_route


FEEDBACK_PATCH_SCHEMA_VERSION = "feedback-patch-v1"
FEEDBACK_PATCH_STATUSES = frozenset(
    {"pending_stage_review", "approved", "rejected"}
)


def _unique(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


@dataclass(frozen=True)
class FeedbackArtifactPatch:
    """A non-executable proposal bound to one structured diagnosis."""

    patch_id: str
    diagnosis_id: str
    requirement_id: str
    source_ir_version: str
    rerun_start_stage: str
    reopened_ir_node_ids: tuple[str, ...] = ()
    affected_hole_ids: tuple[str, ...] = ()
    allowed_op_adjustments: tuple[tuple[str, tuple[str, ...]], ...] = ()
    new_ambiguity_debt: tuple[str, ...] = ()
    new_trace_obligation_requests: tuple[str, ...] = ()
    stale_artifact_declarations: tuple[str, ...] = ()
    expected_changed_artifacts: tuple[str, ...] = ()
    review_status: str = "pending_stage_review"
    reviewer_id: str = ""
    review_basis: tuple[str, ...] = ()
    schema_version: str = FEEDBACK_PATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "patch_id",
            "diagnosis_id",
            "requirement_id",
            "rerun_start_stage",
            "schema_version",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(field_name + " must not be empty")
        if self.review_status not in FEEDBACK_PATCH_STATUSES:
            raise ValueError("unsupported feedback patch review status")
        if self.review_status == "approved" and not self.reviewer_id.strip():
            raise ValueError("approved feedback patch requires reviewer_id")
        if self.rerun_start_stage not in {
            "pipeline",
            "segmentation",
            "plain_reading",
            "semantic",
            "trace",
            "skeleton",
            "completion",
            "selection",
        }:
            raise ValueError("unsupported feedback patch rerun_start_stage")
        for field_name in (
            "reopened_ir_node_ids",
            "affected_hole_ids",
            "new_ambiguity_debt",
            "new_trace_obligation_requests",
            "stale_artifact_declarations",
            "expected_changed_artifacts",
            "review_basis",
        ):
            object.__setattr__(self, field_name, _unique(getattr(self, field_name)))
        normalized_adjustments: list[tuple[str, tuple[str, ...]]] = []
        seen_nodes: set[str] = set()
        for node_id, operators in self.allowed_op_adjustments:
            normalized_node = str(node_id).strip()
            if not normalized_node or normalized_node in seen_nodes:
                raise ValueError("allowed-op adjustments require unique node ids")
            normalized_ops = _unique(operators)
            if not normalized_ops:
                raise ValueError("allowed-op adjustment must declare operators")
            seen_nodes.add(normalized_node)
            normalized_adjustments.append((normalized_node, normalized_ops))
        object.__setattr__(self, "allowed_op_adjustments", tuple(normalized_adjustments))

    @property
    def executable(self) -> bool:
        """Approval makes a patch eligible for stage processing, not auto-application."""

        return self.review_status == "approved"


_ROUTE_TO_BATCH_STAGE = {
    "segmentation": "segmentation",
    "plain_reading": "plain_reading",
    "nlu": "semantic",
    "semantic_ir": "semantic",
    "semantic_approval": "semantic",
    "trace_obligation_planning": "trace",
    "trace_adjudication": "trace",
    "skeleton": "skeleton",
    "completion": "completion",
    "compiler": "completion",
    "selection": "selection",
    "approval": "selection",
    "manual_semantic_review": "semantic",
    "manual_approval": "selection",
}


def propose_feedback_patch(
    diagnosis: ErrorDiagnosisPacket,
    *,
    source_ir_version: str = "",
) -> FeedbackArtifactPatch:
    """Convert diagnosis routing into a neutral, pending patch proposal."""

    route = resolve_error_route(
        diagnosis.error_class,
        source_stage=diagnosis.responsible_stage,
    )
    start_stage = _ROUTE_TO_BATCH_STAGE.get(diagnosis.responsible_stage, "pipeline")
    ambiguity_debt = (
        diagnosis.confusion_targets
        if diagnosis.responsible_stage
        in {"plain_reading", "semantic_ir", "semantic_approval"}
        else ()
    )
    trace_requests = (
        diagnosis.confusion_targets or diagnosis.affected_obligation_ids
        if diagnosis.responsible_stage
        in {"trace_obligation_planning", "trace_adjudication"}
        else ()
    )
    identity = {
        "diagnosis_id": diagnosis.diagnosis_id,
        "requirement_id": diagnosis.requirement_id,
        "source_ir_version": source_ir_version,
        "responsible_stage": diagnosis.responsible_stage,
        "nodes": diagnosis.affected_ir_node_ids,
        "holes": diagnosis.affected_hole_ids,
    }
    patch_id = "feedback-patch-" + sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return FeedbackArtifactPatch(
        patch_id=patch_id,
        diagnosis_id=diagnosis.diagnosis_id,
        requirement_id=diagnosis.requirement_id,
        source_ir_version=source_ir_version,
        rerun_start_stage=start_stage,
        reopened_ir_node_ids=diagnosis.affected_ir_node_ids,
        affected_hole_ids=diagnosis.affected_hole_ids,
        new_ambiguity_debt=ambiguity_debt,
        new_trace_obligation_requests=trace_requests,
        stale_artifact_declarations=route.invalidated_artifacts,
        expected_changed_artifacts=route.expected_changed_artifacts,
    )


def review_feedback_patch(
    patch: FeedbackArtifactPatch,
    *,
    approved: bool,
    reviewer_id: str,
    review_basis: Iterable[str],
    current_ir_version: str = "",
) -> FeedbackArtifactPatch:
    """Record owning-stage review without silently changing formal artifacts."""

    reviewer = str(reviewer_id).strip()
    if not reviewer:
        raise ValueError("feedback patch review requires reviewer_id")
    if (
        patch.source_ir_version
        and current_ir_version
        and patch.source_ir_version != current_ir_version
    ):
        raise ValueError("feedback patch is bound to a stale IR version")
    return replace(
        patch,
        review_status="approved" if approved else "rejected",
        reviewer_id=reviewer,
        review_basis=_unique(review_basis),
    )


def feedback_patch_from_mapping(payload: Mapping[str, Any]) -> FeedbackArtifactPatch:
    adjustments: list[tuple[str, tuple[str, ...]]] = []
    for raw in payload.get("allowed_op_adjustments", ()) or ():
        if isinstance(raw, Mapping):
            adjustments.append(
                (
                    str(raw.get("ir_node_id", raw.get("node_id", ""))),
                    tuple(raw.get("allowed_ops", raw.get("operators", ())) or ()),
                )
            )
        elif isinstance(raw, (list, tuple)) and len(raw) == 2:
            adjustments.append((str(raw[0]), tuple(raw[1] or ())))
        else:
            raise ValueError("invalid allowed_op_adjustments entry")
    return FeedbackArtifactPatch(
        patch_id=str(payload.get("patch_id", "")),
        diagnosis_id=str(payload.get("diagnosis_id", "")),
        requirement_id=str(payload.get("requirement_id", "")),
        source_ir_version=str(payload.get("source_ir_version", "")),
        rerun_start_stage=str(payload.get("rerun_start_stage", "")),
        reopened_ir_node_ids=tuple(payload.get("reopened_ir_node_ids", ()) or ()),
        affected_hole_ids=tuple(payload.get("affected_hole_ids", ()) or ()),
        allowed_op_adjustments=tuple(adjustments),
        new_ambiguity_debt=tuple(payload.get("new_ambiguity_debt", ()) or ()),
        new_trace_obligation_requests=tuple(
            payload.get("new_trace_obligation_requests", ()) or ()
        ),
        stale_artifact_declarations=tuple(
            payload.get("stale_artifact_declarations", ()) or ()
        ),
        expected_changed_artifacts=tuple(
            payload.get("expected_changed_artifacts", ()) or ()
        ),
        review_status=str(payload.get("review_status", "pending_stage_review")),
        reviewer_id=str(payload.get("reviewer_id", "")),
        review_basis=tuple(payload.get("review_basis", ()) or ()),
        schema_version=str(
            payload.get("schema_version", FEEDBACK_PATCH_SCHEMA_VERSION)
        ),
    )


__all__ = [
    "FEEDBACK_PATCH_SCHEMA_VERSION",
    "FEEDBACK_PATCH_STATUSES",
    "FeedbackArtifactPatch",
    "feedback_patch_from_mapping",
    "propose_feedback_patch",
    "review_feedback_patch",
]
