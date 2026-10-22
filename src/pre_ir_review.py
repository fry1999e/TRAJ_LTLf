"""Executable pre-IR review gates for single and batch requirements."""

from __future__ import annotations
from execution_observer import emit, observed_stage

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Callable, Mapping, Sequence

from agents.base_agent import JSONLLMClient
from agents.plain_reading_agent import (
    PlainReadingAgent,
    PlainReadingSemanticFallback,
)
from agents.plain_reading_revision_agent import PlainReadingRevisionAgent
from agents.segmentation_agent import SegmentationAgent, approve_rule_segmentation
from requirement_reading_contract import REQUIREMENT_READING_RULES
from guided_decision import build_guided_stage_decision
from ap_identity import (
    approve_ap_inventory,
    promote_replacement_ap_inventory,
    propose_grounded_ap_inventory,
    undetermined_profile,
)
from ap_grounding import APGroundingContract, build_ap_grounding_contract
from plain_reading import (
    PlainReadingValidationError,
    advance_human_semantic_override_ap_review,
    active_post_independent_revision_review_scope,
    apply_plain_reading_decisions,
    apply_plain_reading_revision,
    assert_plain_reading_gate,
    assert_plain_reading_review_well_formed,
    build_plain_reading_review,
    cancel_human_semantic_override_ap_review,
    close_plain_reading_revision_request,
    confirm_no_ambiguity,
    apply_no_ambiguity_judge_decision,
    confirm_human_semantic_override,
    confirm_plain_reading_revision_classification,
    create_plain_reading_revision_request,
    create_human_semantic_override,
    is_post_independent_revision_all_axes_discarded,
    record_ambiguity_generation_exhausted,
    request_human_semantic_override,
    request_ambiguity_generation,
    validate_rebuilt_plain_reading_axes,
)
from plain_reading_workflow import generate_plain_reading_outcome
from replacement_review import replacement_surface_decision_from_command
from rag.retriever import KeywordRetriever
from schemas import (
    APInventory,
    APRepresentationProfile,
    ArtifactGenerator,
    DeclaredAPCatalog,
    HumanSemanticOverride,
    MechanicalSegmentationValidation,
    NoAmbiguityConfirmation,
    OverrideAPComparison,
    PlainReadingReview,
    PlainReadingRevisionRequest,
    SurfaceSegmentation,
)
from surface_segmentation import (
    SegmentationValidationError,
    assert_surface_segmentation_gate,
    derive_segments,
    revise_segmentation,
    source_hash,
)
from source_evidence_views import (
    SourceQuoteValidationError,
    quote_repair_context,
    validate_source_quotes,
)


@dataclass(frozen=True)
class PreIRReviewConfig:
    segmentation_review: str = "rules"
    plain_reading_review: str = "off"
    max_segmentation_retries: int = 2
    max_surface_review_rounds: int = 3
    max_plain_reading_rounds: int = 2
    max_plain_reading_revision_rounds: int = 2
    segmentation_prompt_version: str = "plain-segmentation-v1"
    plain_reading_prompt_version: str = "plain-reading-v5"
    data_split: str = "test"
    semantic_fallback_enabled: bool = True

    def __post_init__(self) -> None:
        if self.segmentation_review not in {"manual", "rules"}:
            raise ValueError("segmentation_review must be manual or rules; model reviewers are disabled")
        if self.plain_reading_review not in {"manual", "off"}:
            raise ValueError("plain_reading_review must be manual or off; model reviewers are disabled")
        if self.max_segmentation_retries < 0:
            raise ValueError("max_segmentation_retries must be non-negative")
        if self.max_surface_review_rounds < 1:
            raise ValueError("max_surface_review_rounds must be at least 1")
        if self.max_plain_reading_rounds < 1:
            raise ValueError("max_plain_reading_rounds must be at least 1")
        if self.max_plain_reading_revision_rounds < 1:
            raise ValueError("max_plain_reading_revision_rounds must be at least 1")


@dataclass(frozen=True)
class PreIRArtifacts:
    surface_segmentation: SurfaceSegmentation
    plain_reading_review: PlainReadingReview
    effective_source_text: str = ""
    human_semantic_override: HumanSemanticOverride | None = None
    original_ap_inventory: APInventory | None = None
    replacement_ap_inventory: APInventory | None = None
    override_ap_comparison: OverrideAPComparison | None = None
    effective_ap_inventory: APInventory | None = None
    guided_review_records: tuple[dict[str, object], ...] = ()


class PreIRGateError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        status: str,
        diagnostics: Mapping[str, object] | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.status = status
        self.diagnostics = dict(diagnostics or {})


class _InteractiveAPReviewAction(RuntimeError):
    def __init__(self, action: str) -> None:
        super().__init__(action)
        self.action = action


def requires_interactive_pre_ir_review(
    segmentation_review: str,
    plain_reading_review: str,
) -> bool:
    """Return whether either pre-IR gate explicitly requests human input."""

    return "manual" in {segmentation_review, plain_reading_review}


def _normalize_plain_reading_no_axis_action(
    decision: str | None,
    *,
    semantic_fallback: bool,
) -> str:
    """Map one no-axis command to a protocol-specific state transition.

    Normal empty-axis review uses A/E/C, while the bounded semantic fallback
    uses Y/R/C.  Natural-language and historical long-form aliases remain
    accepted, but the protocol letters themselves never cross the boundary:
    in particular Y cannot approve a normal no-axis proposal and A cannot
    approve a semantic fallback.
    """

    normalized = (decision or "").strip().casefold()
    if normalized in {"", "c", "cancel", "later", "defer", "暂不处理"}:
        return "defer"
    if semantic_fallback:
        if normalized in {
            "y",
            "yes",
            "confirm",
            "confirmed",
            "accept",
            "同意",
            "继续",
        }:
            return "accept"
        if normalized in {
            "r",
            "b",
            "generate",
            "replace",
            "replacement",
            "reject",
            "disagree",
            "不同意",
            "可能有歧义",
        }:
            return "request_semantic_override"
        return "invalid"
    if normalized in {
        "a",
        "confirm",
        "confirmed",
        "accept",
        "同意",
        "继续",
    }:
        return "accept"
    if normalized in {
        "e",
        "b",
        "generate",
        "reject",
        "disagree",
        "可能有歧义",
    }:
        return "request_ambiguity_generation"
    return "invalid"


def _return_semantic_override_to_fallback(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    reviewer_id: str = "",
    cancelled_override: HumanSemanticOverride | None = None,
) -> PlainReadingReview:
    """Return an unactivated R proposal to the same auditable fallback."""

    event: dict[str, object] = {
        "round": len(review.review_history) + 1,
        "operation": "human_semantic_override_cancelled_returned_to_fallback",
        "old_reading_review_id": review.reading_review_id,
        "reviewer_id": reviewer_id,
        "reviewer_role": "ordinary_reviewer",
        "reviewer_source": "human",
    }
    if cancelled_override is not None:
        event.update(
            {
                "cancelled_override_id": cancelled_override.override_id,
                "override_status": cancelled_override.status,
            }
        )
    return build_plain_reading_review(
        requirement_id=review.requirement_id,
        segmentation=segmentation,
        ambiguities=(),
        overall_status="pending_no_ambiguity_confirmation",
        review_basis="llm_no_source_supported_axis_after_repair",
        reviewer_source="",
        generator=review.generator,
        experiment_mode=review.experiment_mode,
        data_split=review.data_split,
        review_history=(*review.review_history, event),
        revision_requests=(),
        no_ambiguity_confirmation=NoAmbiguityConfirmation(
            status="pending",
            llm_proposed_no_ambiguity=True,
        ),
        ambiguity_generation_trigger=review.ambiguity_generation_trigger,
        downstream_status=review.downstream_status,
        supersedes_reading_review_id=review.reading_review_id,
    )


@observed_stage("pre_ir")
def run_pre_ir_review(
    *,
    requirement_id: str,
    requirement: str,
    retriever: KeywordRetriever,
    llm_client: JSONLLMClient | None = None,
    judge_client: JSONLLMClient | None = None,
    config: PreIRReviewConfig | None = None,
    surface_cut_offsets: Sequence[int] | None = None,
    surface_ambiguity_flagged: bool = False,
    plain_decisions: Mapping[str, str] | None = None,
    no_ambiguity_decision: str | None = None,
    suspected_ambiguity_segment_ids: Sequence[str] = (),
    suspected_ambiguity_comment: str = "",
    human_semantic_replacement: str = "",
    human_semantic_override_confirmed: bool | None = None,
    reviewer_id: str = "",
    interactive: bool = False,
    guided_context: object | None = None,
    artifact_callback: Callable[[SurfaceSegmentation, PlainReadingReview], None]
    | None = None,
    segmentation_artifact_callback: Callable[[SurfaceSegmentation], None]
    | None = None,
    segmentation_generation_diagnostic_callback: Callable[
        [Mapping[str, object]], None
    ]
    | None = None,
    generation_diagnostic_callback: Callable[[Mapping[str, object]], None]
    | None = None,
    semantic_override_callback: Callable[[HumanSemanticOverride], None]
    | None = None,
    ap_artifact_callback: Callable[
        [APInventory | OverrideAPComparison], None
    ]
    | None = None,
    ap_representation_profile: APRepresentationProfile | None = None,
    declared_ap_catalog: DeclaredAPCatalog | None = None,
    current_ap_inventory: APInventory | None = None,
    ap_candidates_available: bool = False,
    original_source_text: str | None = None,
    active_semantic_override: HumanSemanticOverride | None = None,
) -> PreIRArtifacts:
    """Run both independent gates and return only IR-admissible artifacts."""

    config = config or PreIRReviewConfig()
    from llm.overlap_client import OverlapJSONClient
    if (isinstance(llm_client, OverlapJSONClient) and config.plain_reading_review == "manual"
            and guided_context is None and not ap_candidates_available
            and current_ap_inventory is None and declared_ap_catalog is None):
        llm_client.prefetch_aps(requirement)
    original_source = (
        original_source_text
        or (
            active_semantic_override.original_source_text
            if active_semantic_override is not None
            else requirement
        )
    )
    segmentation_agent = SegmentationAgent(
        retriever,
        llm_client=(llm_client if config.segmentation_review != "rules"
                    and not (config.segmentation_review == "manual" and guided_context is None) else None),
    )
    guided_records: list[dict[str, object]] = []
    segmentation_attempt_context = _guided_attempt_view(
        guided_context,
        "segmentation",
        1,
    )
    try:
        segmentation = segmentation_agent.generate(
            requirement,
            requirement_id=requirement_id,
            max_retries=config.max_segmentation_retries,
            prompt_version=config.segmentation_prompt_version,
            experiment_mode=config.segmentation_review,
            data_split=config.data_split,
            guided_context=segmentation_attempt_context,
            generation_diagnostic_callback=(
                segmentation_generation_diagnostic_callback
            ),
        )
        if segmentation_artifact_callback is not None:
            segmentation_artifact_callback(segmentation)
    except SegmentationValidationError as exc:
        raise PreIRGateError(
            str(exc), stage="segmentation_generation", status="blocked_error"
        ) from exc

    gold_segmentation = _guided_gold_artifact(
        guided_context,
        "gold_surface_segmentation",
    )
    gold_reading = _guided_gold_artifact(
        guided_context,
        "gold_plain_reading_review",
    )
    if gold_segmentation is not None and gold_reading is None:
        raise ValueError(
            "curated warmup Gold Surface Segmentation requires an explicit "
            "Gold Plain Reading Review or no_ambiguity artifact"
        )
    if gold_segmentation is not None:
        if not isinstance(gold_segmentation, SurfaceSegmentation):
            raise TypeError("gold_surface_segmentation must be validated")
        segmentation, segmentation_decision = _resolve_guided_segmentation_stage(
            requirement_id=requirement_id,
            requirement=requirement,
            generated_round_1=segmentation,
            gold_segmentation=gold_segmentation,
            segmentation_agent=segmentation_agent,
            guided_context=guided_context,
            config=config,
            generation_diagnostic_callback=(
                segmentation_generation_diagnostic_callback
            ),
        )
        guided_records.append(
            _guided_pre_ir_record("segmentation", segmentation_decision)
        )
    else:
        judge_surface_offsets: Sequence[int] | None = surface_cut_offsets
        judge_surface_ambiguity = surface_ambiguity_flagged
        if config.segmentation_review == "judge" and judge_client is not None:
            judge_surface_offsets, judge_surface_ambiguity = _judge_surface_decision(
                judge_client,
                segmentation,
            )
        segmentation = _review_segmentation(
            segmentation,
            mode=config.segmentation_review,
            cut_offsets=judge_surface_offsets,
            ambiguity_flagged=judge_surface_ambiguity,
            reviewer_id=reviewer_id,
            interactive=interactive,
            max_review_rounds=config.max_surface_review_rounds,
        )
    try:
        assert_surface_segmentation_gate(
            segmentation,
            current_source_text=requirement,
        )
    except SegmentationValidationError as exc:
        raise PreIRGateError(
            str(exc), stage="surface_review", status=segmentation.status
        ) from exc
    ap_grounding_contract = build_ap_grounding_contract(
        requirement_id=requirement_id,
        source_text=requirement,
        representation_profile=(
            ap_representation_profile
            or undetermined_profile(
                provenance="pre_ir_plain_reading_default"
            )
        ),
        segmentation=segmentation,
        declared_catalog=declared_ap_catalog,
        active_inventory=current_ap_inventory,
        provenance=("pre_ir_plain_reading",),
    )

    if gold_reading is not None:
        if gold_segmentation is None:
            raise ValueError(
                "Gold Plain Reading Review requires Gold Surface Segmentation"
            )
        if not isinstance(gold_reading, PlainReadingReview):
            raise TypeError("gold_plain_reading_review must be validated")
        reading_agent = PlainReadingAgent(retriever, llm_client=llm_client)
        review, reading_decision = _resolve_guided_plain_reading_stage(
            requirement_id=requirement_id,
            requirement=requirement,
            segmentation=segmentation,
            gold_review=gold_reading,
            reading_agent=reading_agent,
            guided_context=guided_context,
            config=config,
            ap_grounding_contract=ap_grounding_contract,
            generation_diagnostic_callback=generation_diagnostic_callback,
        )
        guided_records.append(
            _guided_pre_ir_record("plain_reading", reading_decision)
        )
        return PreIRArtifacts(
            segmentation,
            review,
            effective_source_text=requirement,
            human_semantic_override=active_semantic_override,
            effective_ap_inventory=current_ap_inventory,
            guided_review_records=tuple(guided_records),
        )

    # Reviewer adapters share the same decision protocol and gate policy.
    reading_reviewer = judge_client if config.plain_reading_review == "judge" else llm_client
    if reading_reviewer is not None and config.plain_reading_review in {"manual", "judge"}:
        reading_agent = PlainReadingAgent(retriever, llm_client=llm_client)
        review = generate_plain_reading_outcome(
            reading_agent, requirement, segmentation,
            semantic_fallback_enabled=config.semantic_fallback_enabled,
            requirement_id=requirement_id, mode=config.plain_reading_review,
            prompt_version=config.plain_reading_prompt_version,
            data_split=config.data_split, ap_grounding_contract=ap_grounding_contract,
            generation_diagnostic_callback=generation_diagnostic_callback,
        ).review
        review = complete_plain_reading_review(
            requirement, segmentation, review, reading_reviewer,
            reviewer_source="independent_judge" if config.plain_reading_review == "judge" else "model_assistant",
        )
        if artifact_callback is not None:
            artifact_callback(segmentation, review)
        return PreIRArtifacts(
            segmentation, review, effective_source_text=requirement,
            human_semantic_override=active_semantic_override,
            effective_ap_inventory=current_ap_inventory,
            guided_review_records=tuple(guided_records),
        )

    # Legacy explicit/programmatic review without an API client.
    reading_client = llm_client
    reading_agent = PlainReadingAgent(retriever, llm_client=reading_client)
    review: PlainReadingReview | None = None
    previous_rejected_review: PlainReadingReview | None = None
    for plain_round in range(1, config.max_plain_reading_rounds + 1):
        if previous_rejected_review is not None:
            emit("semantic_revision", source="plain_reading_regeneration", round_index=plain_round)
        try:
            review = generate_plain_reading_outcome(
                reading_agent,
                requirement,
                segmentation,
                semantic_fallback_enabled=config.semantic_fallback_enabled,
                requirement_id=requirement_id,
                mode=config.plain_reading_review,
                prompt_version=config.plain_reading_prompt_version,
                data_split=config.data_split,
                guided_context=guided_context,
                round_index=plain_round,
                previous_review=previous_rejected_review,
                ap_grounding_contract=ap_grounding_contract,
                generation_diagnostic_callback=generation_diagnostic_callback,
            ).review
        except PlainReadingValidationError as exc:
            raise PreIRGateError(
                str(exc),
                stage="plain_reading_generation",
                status="blocked_error",
                diagnostics=getattr(exc, "diagnostics", None),
            ) from exc
        if previous_rejected_review is not None:
            review = build_plain_reading_review(
                requirement_id=review.requirement_id,
                segmentation=segmentation,
                ambiguities=review.ambiguities,
                overall_status=review.overall_status,
                review_basis=review.review_basis,
                reviewer_source=review.reviewer_source,
                generator=review.generator,
                experiment_mode=review.experiment_mode,
                data_split=review.data_split,
                review_history=(
                    *previous_rejected_review.review_history,
                    {
                        "round": len(previous_rejected_review.review_history) + 1,
                        "operation": "plain_reading_options_regenerated_after_neither",
                        "old_reading_review_id": previous_rejected_review.reading_review_id,
                        "old_ambiguity_ids": tuple(
                            item.ambiguity_id
                            for item in previous_rejected_review.ambiguities
                        ),
                        "plain_round": plain_round,
                    },
                ),
                revision_requests=previous_rejected_review.revision_requests,
                no_ambiguity_confirmation=review.no_ambiguity_confirmation,
                ambiguity_generation_trigger=review.ambiguity_generation_trigger,
                downstream_status=previous_rejected_review.downstream_status,
                supersedes_reading_review_id=previous_rejected_review.reading_review_id,
            )
        if artifact_callback is not None:
            artifact_callback(segmentation, review)
        if review.overall_status == "pending_no_ambiguity_confirmation" and config.plain_reading_review == "judge":
            review = _judge_no_ambiguity(judge_client, segmentation, review)
            if artifact_callback is not None:
                artifact_callback(segmentation, review)
            if review.overall_status == "needs_source_clarification":
                break
        if review.overall_status == "pending_no_ambiguity_confirmation":
            decision = (no_ambiguity_decision or "").strip().casefold()
            segment_ids = tuple(suspected_ambiguity_segment_ids)
            comment = suspected_ambiguity_comment
            replacement_text = human_semantic_replacement
            replacement_confirmed = human_semantic_override_confirmed
            semantic_fallback = (
                review.review_basis
                == "llm_no_source_supported_axis_after_repair"
            )
            if interactive and config.plain_reading_review == "manual" and not decision:
                if semantic_fallback:
                    decision, replacement_text, replacement_confirmed = (
                        _interactive_semantic_fallback_confirmation(
                            review,
                            segmentation,
                            original_source_text=original_source,
                            parent_effective_source_hash=segmentation.source_hash,
                            representation_profile=ap_representation_profile,
                        )
                    )
                else:
                    decision, segment_ids, comment = _interactive_no_ambiguity_confirmation(
                        review,
                        segmentation,
                    )
            no_axis_action = _normalize_plain_reading_no_axis_action(
                decision,
                semantic_fallback=semantic_fallback,
            )
            if no_axis_action == "accept":
                review = confirm_no_ambiguity(
                    review,
                    segmentation,
                    reviewer_id=reviewer_id,
                    reviewer_source="human",
                    comment=comment,
                )
                if artifact_callback is not None:
                    artifact_callback(segmentation, review)
            elif no_axis_action == "request_semantic_override":
                pending_review = request_human_semantic_override(
                    review,
                    segmentation,
                    reviewer_id=reviewer_id,
                    reviewer_source="human",
                    comment=comment,
                )
                if artifact_callback is not None:
                    artifact_callback(segmentation, pending_review)
                override = create_human_semantic_override(
                    requirement_id=requirement_id,
                    original_source_text=(
                        requirement
                        if ap_representation_profile is not None
                        else original_source
                    ),
                    replacement_text=replacement_text,
                    parent_effective_source_hash=segmentation.source_hash,
                    reviewer_id=reviewer_id,
                    representation_profile=ap_representation_profile,
                    original_segmentation=segmentation,
                    declared_catalog=declared_ap_catalog,
                    original_inventory=current_ap_inventory,
                    ap_artifact_callback=ap_artifact_callback,
                )
                if semantic_override_callback is not None:
                    semantic_override_callback(override)
                if override.status == "ap_rejected":
                    raise PreIRGateError(
                        "人工 replacement 未通过 AP-only gate: "
                        + ", ".join(override.issue_codes),
                        stage="human_semantic_override",
                        status="ap_rejected",
                        diagnostics={
                            "override": override,
                            "issue_codes": override.issue_codes,
                            "original_ap_set": override.original_ap_set,
                            "replacement_ap_set": override.replacement_ap_set,
                        },
                    )
                if override.status == "awaiting_input":
                    raise PreIRGateError(
                        "等待普通审核者提供完整的新需求语义",
                        stage="human_semantic_override",
                        status="awaiting_human_semantic_override",
                        diagnostics={"override": override},
                    )
                if override.status in {
                    "awaiting_ap_inventory",
                    "awaiting_ap_mapping_confirmation",
                }:
                    raise PreIRGateError(
                        "replacement 等待独立 AP inventory/mapping 审核",
                        stage="human_semantic_override",
                        status=override.status,
                        diagnostics={
                            "override": override,
                            "issue_codes": override.issue_codes,
                        },
                    )
                if ap_representation_profile is not None:
                    # V2 mapping approval and full replacement confirmation are
                    # separate durable actions, even for deterministic symbolic
                    # equality.  Resume from the pending override to confirm.
                    raise PreIRGateError(
                        "replacement 已通过 AP identity gate，等待独立明确确认",
                        stage="human_semantic_override",
                        status="awaiting_confirmation",
                        diagnostics={"override": override},
                    )
                if replacement_confirmed is not True:
                    if replacement_confirmed is False:
                        override = confirm_human_semantic_override(
                            override,
                            confirmed=False,
                            current_effective_source_hash=segmentation.source_hash,
                            reviewer_id=reviewer_id,
                        )
                        if semantic_override_callback is not None:
                            semantic_override_callback(override)
                        if artifact_callback is not None:
                            artifact_callback(segmentation, review)
                        raise PreIRGateError(
                            "人工语义替换已取消；原 no-ambiguity 决策仍待处理",
                            stage="plain_reading",
                            status="pending_no_ambiguity_confirmation",
                        )
                    raise PreIRGateError(
                        "replacement 已通过 AP-only gate，等待明确确认",
                        stage="human_semantic_override",
                        status="awaiting_confirmation",
                        diagnostics={"override": override},
                    )
                override = confirm_human_semantic_override(
                    override,
                    confirmed=True,
                    current_effective_source_hash=segmentation.source_hash,
                    reviewer_id=reviewer_id,
                )
                if semantic_override_callback is not None:
                    semantic_override_callback(override)
                # The active replacement is the sole new grounding source.  No
                # old review, axis, decision, or revision request is forwarded.
                return run_pre_ir_review(
                    requirement_id=requirement_id,
                    requirement=override.replacement_text,
                    retriever=retriever,
                    llm_client=llm_client,
                    judge_client=judge_client,
                    config=config,
                    reviewer_id=reviewer_id,
                    interactive=interactive,
                    guided_context=None,
                    artifact_callback=artifact_callback,
                    segmentation_generation_diagnostic_callback=(
                        segmentation_generation_diagnostic_callback
                    ),
                    generation_diagnostic_callback=generation_diagnostic_callback,
                    semantic_override_callback=semantic_override_callback,
                    ap_artifact_callback=ap_artifact_callback,
                    ap_representation_profile=ap_representation_profile,
                    declared_ap_catalog=declared_ap_catalog,
                    original_source_text=original_source,
                    active_semantic_override=override,
                )
            elif no_axis_action == "request_ambiguity_generation":
                requested_review = request_ambiguity_generation(
                    review,
                    segmentation,
                    suspected_segment_ids=segment_ids,
                    user_comment=comment,
                    reviewer_id=reviewer_id,
                    reviewer_source="human",
                )
                if artifact_callback is not None:
                    artifact_callback(segmentation, requested_review)
                try:
                    outcome = generate_plain_reading_outcome(
                        reading_agent,
                        requirement,
                        segmentation,
                        semantic_fallback_enabled=config.semantic_fallback_enabled,
                        requirement_id=requirement_id,
                        mode=config.plain_reading_review,
                        prompt_version=config.plain_reading_prompt_version,
                        data_split=config.data_split,
                        guided_context=guided_context,
                        round_index=plain_round,
                        ap_grounding_contract=ap_grounding_contract,
                        ambiguity_generation_request=requested_review,
                        generation_diagnostic_callback=generation_diagnostic_callback,
                    )
                    review = outcome.review
                except PlainReadingSemanticFallback as exc:
                    if not config.semantic_fallback_enabled:
                        failed_review = record_ambiguity_generation_exhausted(
                            requested_review,
                            segmentation,
                            attempts=exc.attempts,
                            errors=exc.errors,
                        )
                        if artifact_callback is not None:
                            artifact_callback(segmentation, failed_review)
                        raise PreIRGateError(
                            str(exc),
                            stage="plain_reading_generation",
                            status="blocked_error",
                            diagnostics=exc.diagnostics,
                        ) from exc
                    raise AssertionError(
                        "enabled semantic fallback must be normalized by the workflow"
                    ) from exc
                except PlainReadingValidationError as exc:
                    failed_review = record_ambiguity_generation_exhausted(
                        requested_review,
                        segmentation,
                        attempts=int(getattr(exc, "attempts", 2)),
                        errors=tuple(getattr(exc, "errors", (str(exc),))),
                    )
                    if artifact_callback is not None:
                        artifact_callback(segmentation, failed_review)
                    raise PreIRGateError(
                        str(exc),
                        stage="plain_reading_generation",
                        status="needs_source_clarification",
                        diagnostics=getattr(exc, "diagnostics", None),
                    ) from exc
                if outcome.semantic_fallback:
                    if artifact_callback is not None:
                        artifact_callback(segmentation, review)
                    return resume_semantic_fallback_pre_ir(
                        requirement=requirement,
                        original_source_text=original_source,
                        segmentation=segmentation,
                        review=review,
                        retriever=retriever,
                        llm_client=llm_client,
                        judge_client=judge_client,
                        config=config,
                        reviewer_id=reviewer_id,
                        interactive=interactive,
                        decision=("r" if human_semantic_replacement else ""),
                        replacement_text=human_semantic_replacement,
                        replacement_confirmed=human_semantic_override_confirmed,
                        artifact_callback=artifact_callback,
                        segmentation_generation_diagnostic_callback=(
                            segmentation_generation_diagnostic_callback
                        ),
                        generation_diagnostic_callback=generation_diagnostic_callback,
                        semantic_override_callback=semantic_override_callback,
                        active_semantic_override=active_semantic_override,
                    )
                if artifact_callback is not None:
                    artifact_callback(segmentation, review)
        while review.overall_status == "pending_plain_review":
            decisions = dict(plain_decisions or {})
            revision_requests: dict[str, PlainReadingRevisionRequest] = {}
            if interactive and config.plain_reading_review == "manual":
                interactive_decisions, interactive_requests = _interactive_plain_decisions(
                    review,
                    segmentation,
                    reviewer_id=reviewer_id,
                    max_revision_rounds=config.max_plain_reading_revision_rounds,
                )
                decisions.update(interactive_decisions)
                revision_requests.update(interactive_requests)
            if config.plain_reading_review == "judge" and not decisions:
                decisions = _judge_plain_decisions(judge_client, requirement, review)
                if not decisions:
                    # A detector is not silently promoted to a semantic judge.
                    decisions = {
                        ambiguity.ambiguity_id: "ambiguous"
                        for ambiguity in review.ambiguities
                    }
            if (
                config.plain_reading_review == "manual"
                and not interactive
                and not decisions
            ):
                break
            review_before_decisions = review
            review = apply_plain_reading_decisions(
                review,
                segmentation,
                decisions,
                reviewer_id=reviewer_id,
                reviewer_source=(
                    "independent_judge"
                    if config.plain_reading_review == "judge"
                    else "human"
                    if config.plain_reading_review == "manual"
                    else "rules"
                ),
                review_basis=(
                    "ordinary_language_reading"
                    if config.plain_reading_review == "manual"
                    else "independent_plain_language_judgment"
                    if config.plain_reading_review == "judge"
                    else "plain_reading_disabled"
                ),
                revision_requests=revision_requests,
            )
            if (
                review.overall_status == "pending_no_ambiguity_confirmation"
                and review.review_history[-1].get("all_axes_discarded")
            ):
                old_axes = tuple(
                    (axis.question, tuple(option.plain_interpretation for option in axis.options))
                    for axis in review_before_decisions.ambiguities
                )
                if config.plain_reading_review == "judge":
                    review = _judge_no_ambiguity(judge_client, segmentation, review)
                    new_axes = tuple(
                        (axis.question, tuple(option.plain_interpretation for option in axis.options))
                        for axis in review.ambiguities
                    )
                    if new_axes and new_axes == old_axes:
                        raise PreIRGateError(
                            "independent reviewer repeated rejected ambiguity axes",
                            stage="plain_reading", status="ambiguity_review_cycle",
                        )
                elif (no_ambiguity_decision or "").strip().casefold() in {"a", "confirm", "accept"}:
                    review = confirm_no_ambiguity(
                        review, segmentation, reviewer_id=reviewer_id,
                        reviewer_source="human",
                        comment="independent confirmation after rejecting all axes",
                    )
            if artifact_callback is not None:
                artifact_callback(segmentation, review)
            if review.overall_status == "options_revision_requested":
                emit("semantic_revision", source="plain_reading_options_revision")
                review = _process_plain_reading_revisions(
                    review,
                    segmentation,
                    requirement=requirement,
                    retriever=retriever,
                    llm_client=reading_client,
                    reviewer_id=reviewer_id,
                    interactive=interactive and config.plain_reading_review == "manual",
                )
                if artifact_callback is not None:
                    artifact_callback(segmentation, review)
                if review.overall_status == "pending_plain_review":
                    plain_decisions = None
                    continue
            break
        if review.overall_status != "needs_source_clarification":
            break
        if is_post_independent_revision_all_axes_discarded(review):
            break
        previous_rejected_review = review
    if review is None:  # pragma: no cover - positive round count is validated
        raise PreIRGateError(
            "plain reading generation produced no artifact",
            stage="plain_reading",
            status="blocked_error",
        )
    try:
        assert_plain_reading_gate(review, segmentation)
    except PlainReadingValidationError as exc:
        status = review.overall_status
        stage = "plain_reading"
        if status == "needs_source_clarification":
            raise PreIRGateError(
                str(exc),
                stage=stage,
                status="needs_source_clarification",
                diagnostics=getattr(exc, "diagnostics", None),
            ) from exc
        raise PreIRGateError(
            str(exc),
            stage=stage,
            status=status,
            diagnostics=getattr(exc, "diagnostics", None),
        ) from exc
    return PreIRArtifacts(
        segmentation,
        review,
        effective_source_text=requirement,
        human_semantic_override=active_semantic_override,
        effective_ap_inventory=current_ap_inventory,
        guided_review_records=tuple(guided_records),
    )


def _resolve_guided_segmentation_stage(
    *,
    requirement_id: str,
    requirement: str,
    generated_round_1: SurfaceSegmentation,
    gold_segmentation: SurfaceSegmentation,
    segmentation_agent: SegmentationAgent,
    guided_context: object,
    config: PreIRReviewConfig,
    generation_diagnostic_callback: Callable[[Mapping[str, object]], None]
    | None,
) -> tuple[SurfaceSegmentation, dict]:
    """Automatically adjudicate a curated warmup segmentation in two attempts."""

    assert_surface_segmentation_gate(
        gold_segmentation,
        current_source_text=requirement,
    )
    round_1 = approve_rule_segmentation(generated_round_1)
    round_1_errors = _segmentation_expected_errors(round_1, gold_segmentation)
    round_2_artifact: object | None = None
    round_2_errors: list[str] = []
    artifact_origin = "teacher_adjudicated_round_1"
    verification_status = "passed"
    if round_1_errors:
        artifact_origin = "expected_fallback"
        verification_status = "passed_after_expected_adoption"
        try:
            round_2 = segmentation_agent.generate(
                requirement,
                requirement_id=requirement_id,
                max_retries=config.max_segmentation_retries,
                prompt_version=config.segmentation_prompt_version,
                experiment_mode="stage_guided",
                data_split=config.data_split,
                guided_context=_guided_attempt_view(
                    guided_context,
                    "segmentation",
                    2,
                ),
                generation_diagnostic_callback=generation_diagnostic_callback,
            )
            round_2 = approve_rule_segmentation(round_2)
            round_2_artifact = asdict(round_2)
            round_2_errors = _segmentation_expected_errors(
                round_2,
                gold_segmentation,
            )
            if not round_2_errors:
                artifact_origin = "teacher_adjudicated_round_2"
                verification_status = "passed"
        except Exception as exc:
            round_2_artifact = _attempt_error_artifact(exc)
            round_2_errors = [
                f"round 2 segmentation generation failed: {type(exc).__name__}: {exc}"
            ]

    decision_kwargs: dict[str, object] = {
        "artifact_origin": artifact_origin,
    }
    if round_1_errors:
        decision_kwargs.update(
            {
                "round_2_artifact": round_2_artifact,
                "round_2_verification_status": verification_status,
            }
        )
    decision = build_guided_stage_decision(
        stage="segmentation",
        requirement_id=requirement_id,
        requirement=requirement,
        round_index=1,
        generated_artifact=asdict(round_1),
        expected_artifact=asdict(gold_segmentation),
        mismatch_evidence=round_1_errors,
        llm_client=None,
        **decision_kwargs,
    )
    decision["round_2_mismatch_evidence"] = round_2_errors
    accepted = replace(
        gold_segmentation,
        experiment_mode="stage_guided",
        data_split=config.data_split,
        label_source="warmup_expected_teacher",
        review_basis="stage_scoped_expected_adjudication",
        persistent_rag_allowed=False,
    )
    assert_surface_segmentation_gate(
        accepted,
        current_source_text=requirement,
    )
    return accepted, decision


def _resolve_guided_plain_reading_stage(
    *,
    requirement_id: str,
    requirement: str,
    segmentation: SurfaceSegmentation,
    gold_review: PlainReadingReview,
    reading_agent: PlainReadingAgent,
    guided_context: object,
    config: PreIRReviewConfig,
    ap_grounding_contract: APGroundingContract | None = None,
    generation_diagnostic_callback: Callable[[Mapping[str, object]], None]
    | None = None,
) -> tuple[PlainReadingReview, dict]:
    """Use stage-scoped expected results instead of a human ambiguity judge."""

    assert_plain_reading_gate(gold_review, segmentation)
    round_1_artifact: object
    try:
        round_1_review = generate_plain_reading_outcome(
            reading_agent,
            requirement,
            segmentation,
            semantic_fallback_enabled=False,
            requirement_id=requirement_id,
            mode="stage_guided",
            prompt_version=config.plain_reading_prompt_version,
            data_split=config.data_split,
            guided_context=_guided_attempt_view(
                guided_context,
                "plain_reading",
                1,
            ),
            round_index=1,
            ap_grounding_contract=ap_grounding_contract,
            generation_diagnostic_callback=generation_diagnostic_callback,
        ).review
        round_1_artifact = asdict(round_1_review)
        round_1_errors = _plain_reading_expected_errors(
            round_1_review,
            gold_review,
        )
    except Exception as exc:
        round_1_artifact = _attempt_error_artifact(exc)
        round_1_errors = [
            f"round 1 plain-reading generation failed: {type(exc).__name__}: {exc}"
        ]

    round_2_artifact: object | None = None
    round_2_errors: list[str] = []
    artifact_origin = "teacher_adjudicated_round_1"
    verification_status = "passed"
    if round_1_errors:
        artifact_origin = "expected_fallback"
        verification_status = "passed_after_expected_adoption"
        try:
            round_2_review = generate_plain_reading_outcome(
                reading_agent,
                requirement,
                segmentation,
                semantic_fallback_enabled=False,
                requirement_id=requirement_id,
                mode="stage_guided",
                prompt_version=config.plain_reading_prompt_version,
                data_split=config.data_split,
                guided_context=_guided_attempt_view(
                    guided_context,
                    "plain_reading",
                    2,
                ),
                round_index=2,
                ap_grounding_contract=ap_grounding_contract,
                generation_diagnostic_callback=generation_diagnostic_callback,
            ).review
            round_2_artifact = asdict(round_2_review)
            round_2_errors = _plain_reading_expected_errors(
                round_2_review,
                gold_review,
            )
            if not round_2_errors:
                artifact_origin = "teacher_adjudicated_round_2"
                verification_status = "passed"
        except Exception as exc:
            round_2_artifact = _attempt_error_artifact(exc)
            round_2_errors = [
                f"round 2 plain-reading generation failed: {type(exc).__name__}: {exc}"
            ]

    decision_kwargs: dict[str, object] = {
        "artifact_origin": artifact_origin,
    }
    if round_1_errors:
        decision_kwargs.update(
            {
                "round_2_artifact": round_2_artifact,
                "round_2_verification_status": verification_status,
            }
        )
    decision = build_guided_stage_decision(
        stage="plain_reading",
        requirement_id=requirement_id,
        requirement=requirement,
        round_index=1,
        generated_artifact=round_1_artifact,
        expected_artifact=asdict(gold_review),
        mismatch_evidence=round_1_errors,
        llm_client=None,
        **decision_kwargs,
    )
    decision["round_2_mismatch_evidence"] = round_2_errors
    adjudicated_axes = tuple(
        replace(
            ambiguity,
            review_basis="stage_scoped_expected_adjudication",
            reviewer_id="",
            reviewer_source="warmup_expected_teacher",
        )
        for ambiguity in gold_review.ambiguities
    )
    accepted = replace(
        gold_review,
        ambiguities=adjudicated_axes,
        review_basis="stage_scoped_expected_adjudication",
        reviewer_source="warmup_expected_teacher",
        experiment_mode="stage_guided",
        data_split=config.data_split,
        persistent_rag_allowed=False,
        review_history=(
            *gold_review.review_history,
            {
                "operation": "warmup_expected_ambiguity_adjudication",
                "artifact_origin": artifact_origin,
                "attempt_count": decision["attempt_count"],
                "human_adjudication_used": False,
                "formula_visible": False,
            },
        ),
    )
    assert_plain_reading_gate(accepted, segmentation)
    return accepted, decision


def _guided_attempt_view(
    guided_context: object | None,
    stage: str,
    attempt: int,
) -> object | None:
    attempt_view = getattr(guided_context, "attempt_view", None)
    if callable(attempt_view):
        return attempt_view(stage, attempt)
    return guided_context


def _guided_gold_artifact(
    guided_context: object | None,
    name: str,
) -> object | None:
    return getattr(guided_context, name, None) if guided_context is not None else None


def _segmentation_expected_errors(
    generated: SurfaceSegmentation,
    expected: SurfaceSegmentation,
) -> list[str]:
    generated_boundaries = [
        (segment.start, segment.end, segment.text) for segment in generated.segments
    ]
    expected_boundaries = [
        (segment.start, segment.end, segment.text) for segment in expected.segments
    ]
    if generated_boundaries == expected_boundaries:
        return []
    return [
        "surface boundary mismatch: "
        f"generated={generated_boundaries!r}, expected={expected_boundaries!r}"
    ]


def _plain_reading_expected_errors(
    generated: PlainReadingReview,
    expected: PlainReadingReview,
) -> list[str]:
    generated_axes = _plain_reading_axis_signature(generated)
    expected_axes = _plain_reading_axis_signature(expected)
    if generated_axes == expected_axes:
        return []
    errors = [
        "plain-reading ambiguity axis mismatch: "
        f"generated_count={len(generated_axes)}, expected_count={len(expected_axes)}"
    ]
    for index, (generated_axis, expected_axis) in enumerate(
        zip(generated_axes, expected_axes),
        start=1,
    ):
        if generated_axis != expected_axis:
            errors.append(f"ambiguity axis {index} differs from stage-scoped gold")
    return errors


def _plain_reading_axis_signature(
    review: PlainReadingReview,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            tuple(ambiguity.source_segment_ids),
            tuple(ambiguity.source_span),
            _normalized_plain_text(ambiguity.evidence_text),
            _normalized_plain_text(ambiguity.question),
            tuple(
                (
                    option.option_id,
                    _normalized_plain_text(option.plain_interpretation),
                    _normalized_plain_text(option.difference_summary),
                )
                for option in ambiguity.options
            ),
        )
        for ambiguity in review.ambiguities
    )


def _normalized_plain_text(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def _attempt_error_artifact(exc: Exception) -> dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _guided_pre_ir_record(stage: str, decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "stage": f"{stage}_guided_decision",
        "requirement_id": str(decision.get("requirement_id", "")),
        "approved": True,
        "guided_decision": dict(decision),
        "comments": str(decision.get("decision_reason", "")),
    }


def resume_plain_reading_review(
    *,
    requirement: str,
    segmentation: SurfaceSegmentation,
    review: PlainReadingReview,
    retriever: KeywordRetriever,
    llm_client: JSONLLMClient | None = None,
    reviewer_id: str = "",
    interactive: bool = False,
    max_revision_rounds: int = 2,
    no_ambiguity_decision: str | None = None,
    suspected_ambiguity_segment_ids: Sequence[str] = (),
    suspected_ambiguity_comment: str = "",
    artifact_callback: Callable[[SurfaceSegmentation, PlainReadingReview], None]
    | None = None,
    generation_diagnostic_callback: Callable[[Mapping[str, object]], None]
    | None = None,
    semantic_fallback_enabled: bool = True,
) -> PlainReadingReview:
    """Resume a persisted pending R request without creating a duplicate request."""

    if llm_client is not None and review.overall_status in {
        "pending_plain_review", "pending_no_ambiguity_confirmation", "ambiguous_unresolved",
    }:
        result = complete_plain_reading_with_model(requirement, segmentation, review, llm_client)
        if artifact_callback is not None:
            artifact_callback(segmentation, result)
        return result

    validate_status = review.overall_status
    resumable = {
        "options_revision_requested",
        "pending_plain_review",
        "pending_no_ambiguity_confirmation",
        "user_requested_ambiguity_generation",
    }
    if validate_status not in resumable:
        assert_plain_reading_gate(review, segmentation)
        return review
    result = review
    reading_agent = PlainReadingAgent(retriever, llm_client=llm_client)
    if result.overall_status == "pending_no_ambiguity_confirmation":
        decision = (no_ambiguity_decision or "").strip().casefold()
        segment_ids = tuple(suspected_ambiguity_segment_ids)
        comment = suspected_ambiguity_comment
        semantic_fallback = (
            result.review_basis
            == "llm_no_source_supported_axis_after_repair"
        )
        if interactive and not decision:
            if semantic_fallback:
                decision = _interactive_semantic_fallback_action(
                    result,
                    segmentation,
                )
            else:
                decision, segment_ids, comment = _interactive_no_ambiguity_confirmation(
                    result,
                    segmentation,
                )
        no_axis_action = _normalize_plain_reading_no_axis_action(
            decision,
            semantic_fallback=semantic_fallback,
        )
        if no_axis_action == "accept":
            result = confirm_no_ambiguity(
                result,
                segmentation,
                reviewer_id=reviewer_id,
                reviewer_source="human",
                comment=comment,
            )
        elif no_axis_action == "request_semantic_override":
            result = request_human_semantic_override(
                result,
                segmentation,
                reviewer_id=reviewer_id,
                reviewer_source="human",
                comment=comment,
            )
            if artifact_callback is not None:
                artifact_callback(segmentation, result)
            raise PreIRGateError(
                "等待普通审核者提供并确认完整的新需求语义",
                stage="human_semantic_override",
                status="awaiting_human_semantic_override",
            )
        elif no_axis_action == "request_ambiguity_generation":
            result = request_ambiguity_generation(
                result,
                segmentation,
                suspected_segment_ids=segment_ids,
                user_comment=comment,
                reviewer_id=reviewer_id,
                reviewer_source="human",
            )
        if artifact_callback is not None:
            artifact_callback(segmentation, result)
    if result.overall_status == "user_requested_ambiguity_generation":
        try:
            outcome = generate_plain_reading_outcome(
                reading_agent,
                requirement,
                segmentation,
                semantic_fallback_enabled=semantic_fallback_enabled,
                requirement_id=review.requirement_id,
                mode=review.experiment_mode,
                prompt_version=review.generator.prompt_version or "plain-reading-v5",
                data_split=review.data_split,
                ambiguity_generation_request=result,
                generation_diagnostic_callback=generation_diagnostic_callback,
            )
            result = outcome.review
        except PlainReadingSemanticFallback as exc:
            raise PreIRGateError(
                str(exc),
                stage="plain_reading_generation",
                status="blocked_error",
                diagnostics=exc.diagnostics,
            ) from exc
        except PlainReadingValidationError as exc:
            failed_review = record_ambiguity_generation_exhausted(
                result,
                segmentation,
                attempts=int(getattr(exc, "attempts", 2)),
                errors=tuple(getattr(exc, "errors", (str(exc),))),
            )
            if artifact_callback is not None:
                artifact_callback(segmentation, failed_review)
            raise PreIRGateError(
                str(exc),
                stage="plain_reading_generation",
                status="needs_source_clarification",
                diagnostics=getattr(exc, "diagnostics", None),
            ) from exc
        if outcome.semantic_fallback:
            if artifact_callback is not None:
                artifact_callback(segmentation, result)
            raise PreIRGateError(
                "semantic fallback 仍等待普通审核者处理",
                stage="plain_reading",
                status="pending_no_ambiguity_confirmation",
                diagnostics=outcome.diagnostics,
            )
        if artifact_callback is not None:
            artifact_callback(segmentation, result)
    if result.overall_status == "options_revision_requested":
        result = _process_plain_reading_revisions(
            result,
            segmentation,
            requirement=requirement,
            retriever=retriever,
            llm_client=llm_client,
            reviewer_id=reviewer_id,
            interactive=interactive,
        )
        if artifact_callback is not None:
            artifact_callback(segmentation, result)
    while result.overall_status == "pending_plain_review" and interactive:
        decisions, requests = _interactive_plain_decisions(
            result,
            segmentation,
            reviewer_id=reviewer_id,
            max_revision_rounds=max_revision_rounds,
        )
        result = apply_plain_reading_decisions(
            result,
            segmentation,
            decisions,
            reviewer_id=reviewer_id,
            revision_requests=requests,
        )
        if artifact_callback is not None:
            artifact_callback(segmentation, result)
        if result.overall_status == "options_revision_requested":
            result = _process_plain_reading_revisions(
                result,
                segmentation,
                requirement=requirement,
                retriever=retriever,
                llm_client=llm_client,
                reviewer_id=reviewer_id,
                interactive=interactive,
            )
            if artifact_callback is not None:
                artifact_callback(segmentation, result)
    assert_plain_reading_gate(result, segmentation)
    return result


def resume_semantic_fallback_pre_ir(
    *,
    requirement: str,
    original_source_text: str,
    segmentation: SurfaceSegmentation,
    review: PlainReadingReview,
    retriever: KeywordRetriever,
    llm_client: JSONLLMClient | None,
    judge_client: JSONLLMClient | None = None,
    config: PreIRReviewConfig | None = None,
    reviewer_id: str = "",
    interactive: bool = False,
    decision: str = "",
    replacement_text: str = "",
    replacement_confirmed: bool | None = None,
    pending_override: HumanSemanticOverride | None = None,
    active_semantic_override: HumanSemanticOverride | None = None,
    artifact_callback: Callable[[SurfaceSegmentation, PlainReadingReview], None]
    | None = None,
    segmentation_generation_diagnostic_callback: Callable[
        [Mapping[str, object]], None
    ]
    | None = None,
    generation_diagnostic_callback: Callable[[Mapping[str, object]], None]
    | None = None,
    semantic_override_callback: Callable[[HumanSemanticOverride], None]
    | None = None,
    ap_artifact_callback: Callable[
        [APInventory | OverrideAPComparison], None
    ]
    | None = None,
    ap_representation_profile: APRepresentationProfile | None = None,
    declared_ap_catalog: DeclaredAPCatalog | None = None,
    original_ap_inventory: APInventory | None = None,
    replacement_ap_inventory: APInventory | None = None,
    override_ap_comparison: OverrideAPComparison | None = None,
) -> PreIRArtifacts:
    """Resume Y/R/C without spending another artifact attempt on the old source."""

    config = config or PreIRReviewConfig(plain_reading_review=review.experiment_mode)
    validate = (
        review.overall_status == "pending_no_ambiguity_confirmation"
        and review.review_basis
        == "llm_no_source_supported_axis_after_repair"
    ) or (
        review.overall_status == "awaiting_human_semantic_override"
        and review.review_basis == "human_requested_semantic_override"
        and any(
            item.get("operation") == "plain_reading_semantic_fallback"
            for item in review.review_history
        )
    )
    if not validate:
        raise PlainReadingValidationError(
            "resume_semantic_fallback_pre_ir requires a pending semantic fallback"
        )
    normalized = decision.strip().casefold()
    if pending_override is not None:
        if (
            pending_override.requirement_id != review.requirement_id
            or pending_override.original_source_hash
            != source_hash(original_source_text)
            or pending_override.parent_effective_source_hash
            != segmentation.source_hash
        ):
            raise PreIRGateError(
                "HUMAN_OVERRIDE_STALE_PARENT",
                stage="human_semantic_override",
                status="ap_rejected",
                diagnostics={"override": pending_override},
            )
        if pending_override.status in {
            "awaiting_ap_inventory",
            "awaiting_ap_mapping_confirmation",
            "awaiting_confirmation",
        }:
            normalized = "r"
            replacement_text = pending_override.replacement_text
            if (
                interactive
                and pending_override.status == "awaiting_confirmation"
                and replacement_confirmed is None
            ):
                replacement_confirmed = _interactive_existing_override_confirmation(
                    pending_override
                )
    if interactive and not normalized:
        normalized, replacement_text, replacement_confirmed = (
            _interactive_semantic_fallback_confirmation(
                review,
                segmentation,
                original_source_text=original_source_text,
                parent_effective_source_hash=segmentation.source_hash,
            )
        )
    no_axis_action = _normalize_plain_reading_no_axis_action(
        normalized,
        semantic_fallback=True,
    )
    if no_axis_action == "accept":
        confirmation_parent = review
        if review.overall_status == "awaiting_human_semantic_override":
            confirmation_parent = _return_semantic_override_to_fallback(
                review,
                segmentation,
                reviewer_id=reviewer_id,
            )
        confirmed_review = confirm_no_ambiguity(
            confirmation_parent,
            segmentation,
            reviewer_id=reviewer_id,
            reviewer_source="human",
        )
        if artifact_callback is not None:
            artifact_callback(segmentation, confirmed_review)
        return PreIRArtifacts(
            segmentation,
            confirmed_review,
            effective_source_text=requirement,
            human_semantic_override=active_semantic_override,
            effective_ap_inventory=(
                replacement_ap_inventory
                if active_semantic_override is not None
                else original_ap_inventory
            ),
        )
    if no_axis_action != "request_semantic_override":
        raise PreIRGateError(
            "semantic fallback 仍等待普通审核者处理",
            stage="plain_reading",
            status=review.overall_status,
        )
    if review.overall_status == "pending_no_ambiguity_confirmation":
        pending_review = request_human_semantic_override(
            review,
            segmentation,
            reviewer_id=reviewer_id,
        )
        if artifact_callback is not None:
            artifact_callback(segmentation, pending_review)
    else:
        pending_review = review
    override = pending_override or create_human_semantic_override(
        requirement_id=review.requirement_id,
        original_source_text=(
            requirement
            if ap_representation_profile is not None
            else original_source_text
        ),
        replacement_text=replacement_text,
        parent_effective_source_hash=segmentation.source_hash,
        reviewer_id=reviewer_id,
        representation_profile=ap_representation_profile,
        original_segmentation=segmentation,
        declared_catalog=declared_ap_catalog,
        ap_artifact_callback=ap_artifact_callback,
    )
    if semantic_override_callback is not None:
        semantic_override_callback(override)
    if override.status in {
        "awaiting_ap_inventory",
        "awaiting_ap_mapping_confirmation",
    }:
        if not interactive:
            raise PreIRGateError(
                "人工 replacement 等待 AP inventory/mapping 审核",
                stage="human_semantic_override",
                status=override.status,
                diagnostics={"override": override, "issue_codes": override.issue_codes},
            )
        if ap_representation_profile is None:
            raise PreIRGateError(
                "AP representation profile is required",
                stage="human_semantic_override",
                status=override.status,
                diagnostics={"override": override},
            )
        try:
            (
                override,
                original_ap_inventory,
                replacement_ap_inventory,
                override_ap_comparison,
            ) = _interactive_v2_ap_review(
                override=override,
                profile=ap_representation_profile,
                segmentation=segmentation,
                original_inventory=original_ap_inventory,
                replacement_inventory=replacement_ap_inventory,
                reviewer_id=reviewer_id or "interactive-reviewer",
            )
        except _InteractiveAPReviewAction as action:
            if action.action == "defer":
                raise PreIRGateError(
                    "AP inventory/mapping 审核已暂缓",
                    stage="human_semantic_override",
                    status=override.status,
                    diagnostics={"override": override},
                ) from None
            cancelled = cancel_human_semantic_override_ap_review(
                override,
                reviewer_id=reviewer_id or "interactive-reviewer",
            )
            if semantic_override_callback is not None:
                semantic_override_callback(cancelled)
            returned_review = _return_semantic_override_to_fallback(
                pending_review,
                segmentation,
                reviewer_id=reviewer_id,
                cancelled_override=cancelled,
            )
            if artifact_callback is not None:
                artifact_callback(segmentation, returned_review)
            raise PreIRGateError(
                "AP inventory/mapping 已拒绝；返回 semantic fallback 审核",
                stage="human_semantic_override",
                status="pending_no_ambiguity_confirmation",
                diagnostics={"override": cancelled, "review": returned_review},
            ) from None
        for current in (
            original_ap_inventory,
            replacement_ap_inventory,
            override_ap_comparison,
        ):
            if current is not None and ap_artifact_callback is not None:
                ap_artifact_callback(current)
        if semantic_override_callback is not None:
            semantic_override_callback(override)
        raise PreIRGateError(
            (
                "AP mapping 已确认，等待独立 replacement CONFIRM"
                if override.status == "awaiting_confirmation"
                else "AP inventory 已确认，等待独立 AP mapping 审核"
            ),
            stage="human_semantic_override",
            status=override.status,
            diagnostics={
                "override": override,
                "original_ap_inventory": original_ap_inventory,
                "replacement_ap_inventory": replacement_ap_inventory,
                "override_ap_comparison": override_ap_comparison,
            },
        )
    if override.status != "awaiting_confirmation":
        raise PreIRGateError(
            "人工 replacement 尚不能激活"
            + (": " + ", ".join(override.issue_codes) if override.issue_codes else ""),
            stage="human_semantic_override",
            status=override.status,
            diagnostics={"override": override, "issue_codes": override.issue_codes},
        )
    if replacement_confirmed is False:
        cancelled = confirm_human_semantic_override(
            override,
            confirmed=False,
            current_effective_source_hash=segmentation.source_hash,
            reviewer_id=reviewer_id,
            original_inventory=original_ap_inventory,
            replacement_inventory=replacement_ap_inventory,
            comparison=override_ap_comparison,
        )
        if semantic_override_callback is not None:
            semantic_override_callback(cancelled)
        returned_review = _return_semantic_override_to_fallback(
            pending_review,
            segmentation,
            reviewer_id=reviewer_id,
            cancelled_override=cancelled,
        )
        if artifact_callback is not None:
            artifact_callback(segmentation, returned_review)
        raise PreIRGateError(
            "人工语义替换已取消；已返回 semantic fallback 审核",
            stage="human_semantic_override",
            status="pending_no_ambiguity_confirmation",
            diagnostics={"override": cancelled, "review": returned_review},
        )
    if replacement_confirmed is not True:
        raise PreIRGateError(
            "replacement 已通过 AP-only gate，等待明确确认",
            stage="human_semantic_override",
            status="awaiting_confirmation",
            diagnostics={"override": override},
        )
    active = confirm_human_semantic_override(
        override,
        confirmed=True,
        current_effective_source_hash=segmentation.source_hash,
        reviewer_id=reviewer_id,
        original_inventory=original_ap_inventory,
        replacement_inventory=replacement_ap_inventory,
        comparison=override_ap_comparison,
    )
    promoted_inventory: APInventory | None = None
    if (
        original_ap_inventory is not None
        and replacement_ap_inventory is not None
        and override_ap_comparison is not None
    ):
        promoted_inventory = promote_replacement_ap_inventory(
            replacement_ap_inventory,
            original_inventory=original_ap_inventory,
            comparison=override_ap_comparison,
            profile=ap_representation_profile,
        )
        if ap_artifact_callback is not None:
            ap_artifact_callback(promoted_inventory)
    if semantic_override_callback is not None:
        semantic_override_callback(active)
    return run_pre_ir_review(
        requirement_id=review.requirement_id,
        requirement=active.replacement_text,
        retriever=retriever,
        llm_client=llm_client,
        judge_client=judge_client,
        config=config,
        reviewer_id=reviewer_id,
        interactive=interactive,
        artifact_callback=artifact_callback,
        segmentation_generation_diagnostic_callback=(
            segmentation_generation_diagnostic_callback
        ),
        generation_diagnostic_callback=generation_diagnostic_callback,
        semantic_override_callback=semantic_override_callback,
        ap_artifact_callback=ap_artifact_callback,
        ap_representation_profile=ap_representation_profile,
        declared_ap_catalog=declared_ap_catalog,
        current_ap_inventory=promoted_inventory,
        original_source_text=original_source_text,
        active_semantic_override=active,
    )


def run_batch_segmentation_review(
    requirements: Sequence[str],
    *,
    requirement_ids: Sequence[str] | None = None,
    retrievers: Sequence[KeywordRetriever] | None = None,
    llm_client: JSONLLMClient | None = None,
    mode: str = "rules",
    max_retries: int = 2,
    max_review_rounds: int = 3,
    data_split: str = "test",
    decisions_by_id: Mapping[str, Sequence[int]] | None = None,
    interactive: bool = False,
) -> dict[str, SurfaceSegmentation]:
    """Independent batch surface gate; one failed item does not stop generation."""

    ids = list(requirement_ids or [f"REQ-{i:03d}" for i in range(1, len(requirements) + 1)])
    if len(ids) != len(requirements):
        raise ValueError("requirement_ids length must match requirements")
    retriever_list = list(retrievers or [KeywordRetriever([]) for _ in requirements])
    results: dict[str, SurfaceSegmentation] = {}
    for requirement_id, requirement, retriever in zip(ids, requirements, retriever_list):
        try:
            agent = SegmentationAgent(retriever, llm_client=llm_client)
            segmentation = agent.generate(
                requirement,
                requirement_id=requirement_id,
                max_retries=max_retries,
                experiment_mode=mode,
                data_split=data_split,
            )
            results[requirement_id] = _review_segmentation(
                segmentation,
                mode=mode,
                cut_offsets=(decisions_by_id or {}).get(requirement_id),
                ambiguity_flagged=False,
                reviewer_id="",
                interactive=interactive,
                max_review_rounds=max_review_rounds,
            )
        except Exception as exc:
            results[requirement_id] = _blocked_segmentation(
                requirement_id,
                requirement,
                mode=mode,
                data_split=data_split,
                error=exc,
            )
    return results


def run_batch_plain_reading_review(
    requirements: Sequence[str],
    segmentations: Mapping[str, SurfaceSegmentation],
    *,
    requirement_ids: Sequence[str] | None = None,
    retrievers: Sequence[KeywordRetriever] | None = None,
    llm_client: JSONLLMClient | None = None,
    mode: str = "off",
    data_split: str = "test",
    decisions_by_id: Mapping[str, Mapping[str, str]] | None = None,
    interactive: bool = False,
    semantic_fallback_enabled: bool = True,
) -> dict[str, PlainReadingReview]:
    """Independent batch ordinary-language gate linked to reviewed spans."""

    ids = list(requirement_ids or [f"REQ-{i:03d}" for i in range(1, len(requirements) + 1)])
    retriever_list = list(retrievers or [KeywordRetriever([]) for _ in requirements])
    results: dict[str, PlainReadingReview] = {}
    for requirement_id, requirement, retriever in zip(ids, requirements, retriever_list):
        segmentation = segmentations[requirement_id]
        if segmentation.status == "blocked_error":
            results[requirement_id] = _blocked_plain_reading(
                requirement_id,
                segmentation,
                mode=mode,
            )
            continue
        try:
            agent = PlainReadingAgent(retriever, llm_client=llm_client)
            review = generate_plain_reading_outcome(
                agent,
                requirement,
                segmentation,
                semantic_fallback_enabled=semantic_fallback_enabled,
                requirement_id=requirement_id,
                mode=mode,
                data_split=data_split,
            ).review
            if llm_client is not None and mode == "manual":
                results[requirement_id] = complete_plain_reading_with_model(
                    requirement, segmentation, review, llm_client,
                )
                continue
            if review.overall_status == "pending_no_ambiguity_confirmation":
                no_axis_input = dict(
                    (decisions_by_id or {}).get(requirement_id, {})
                )
                decision = str(no_axis_input.get("__no_ambiguity__", "")).casefold()
                segment_ids: tuple[str, ...] = ()
                comment = str(no_axis_input.get("__comment__", ""))
                if interactive and mode == "manual" and not decision:
                    if (
                        review.review_basis
                        == "llm_no_source_supported_axis_after_repair"
                    ):
                        decision = _interactive_semantic_fallback_action(
                            review,
                            segmentation,
                        )
                    else:
                        decision, segment_ids, comment = _interactive_no_ambiguity_confirmation(
                            review,
                            segmentation,
                        )
                no_axis_action = _normalize_plain_reading_no_axis_action(
                    decision,
                    semantic_fallback=(
                        review.review_basis
                        == "llm_no_source_supported_axis_after_repair"
                    ),
                )
                if no_axis_action == "accept":
                    review = confirm_no_ambiguity(
                        review,
                        segmentation,
                        reviewer_source="human",
                        comment=comment,
                    )
                elif no_axis_action == "request_semantic_override":
                    review = request_human_semantic_override(
                        review,
                        segmentation,
                        reviewer_source="human",
                        comment=comment,
                    )
                elif no_axis_action == "request_ambiguity_generation":
                    requested = request_ambiguity_generation(
                        review,
                        segmentation,
                        suspected_segment_ids=segment_ids,
                        user_comment=comment,
                        reviewer_source="human",
                    )
                    review = generate_plain_reading_outcome(
                        agent,
                        requirement,
                        segmentation,
                        semantic_fallback_enabled=semantic_fallback_enabled,
                        requirement_id=requirement_id,
                        mode=mode,
                        data_split=data_split,
                        ambiguity_generation_request=requested,
                    ).review
                else:
                    results[requirement_id] = review
                    break
            if review.overall_status == "pending_plain_review":
                decisions = dict((decisions_by_id or {}).get(requirement_id, {}))
                revision_requests: dict[str, PlainReadingRevisionRequest] = {}
                if interactive and mode == "manual":
                    interactive_decisions, interactive_requests = _interactive_plain_decisions(
                        review,
                        segmentation,
                    )
                    decisions.update(interactive_decisions)
                    revision_requests.update(interactive_requests)
                if mode == "judge" and not decisions:
                    decisions = {
                        ambiguity.ambiguity_id: "ambiguous"
                        for ambiguity in review.ambiguities
                    }
                review = apply_plain_reading_decisions(
                    review,
                    segmentation,
                    decisions,
                    reviewer_source=(
                        "independent_judge" if mode == "judge" else "human" if mode == "manual" else "rules"
                    ),
                    review_basis=(
                        "ordinary_language_reading" if mode == "manual" else "rule_based_detection"
                    ),
                    revision_requests=revision_requests,
                )
            results[requirement_id] = review
            if review.overall_status in {
                "pending_no_ambiguity_confirmation",
                "user_requested_ambiguity_generation",
                "pending_plain_review",
                "needs_source_clarification",
                "awaiting_human_semantic_override",
            }:
                break
        except Exception:
            results[requirement_id] = _blocked_plain_reading(
                requirement_id,
                segmentation,
                mode=mode,
            )
            if mode == "manual":
                break
    waiting_for = next(
        (
            requirement_id
            for requirement_id, review in results.items()
            if review.overall_status
            in {
                "pending_no_ambiguity_confirmation",
                "user_requested_ambiguity_generation",
                "pending_plain_review",
                "awaiting_human_semantic_override",
            }
        ),
        "",
    )
    if not waiting_for and len(results) < len(ids) and results:
        waiting_for = next(reversed(results))
    if waiting_for:
        for requirement_id in ids:
            if requirement_id in results:
                continue
            results[requirement_id] = _deferred_plain_reading(
                requirement_id,
                segmentations[requirement_id],
                mode=mode,
                waiting_for=waiting_for,
            )
    return results


def _review_segmentation(
    segmentation: SurfaceSegmentation,
    *,
    mode: str,
    cut_offsets: Sequence[int] | None,
    ambiguity_flagged: bool,
    reviewer_id: str,
    interactive: bool,
    max_review_rounds: int = 3,
) -> SurfaceSegmentation:
    if mode == "rules":
        return approve_rule_segmentation(segmentation)
    if cut_offsets is None and interactive and mode == "manual":
        cut_offsets, ambiguity_flagged = _interactive_surface_decision(
            segmentation,
            max_rounds=max_review_rounds,
        )
    if cut_offsets is None:
        # Independent judge mode may accept mechanically valid boundaries; it
        # must retain its own provenance and never masquerade as human review.
        if mode == "judge":
            return replace(
                segmentation,
                status="surface_reviewed",
                label_source="independent_judge",
                review_basis="independent_plain_language_judgment",
                persistent_rag_allowed=segmentation.data_split == "train",
            )
        return segmentation
    return revise_segmentation(
        segmentation,
        cut_offsets=cut_offsets,
        decision="ambiguity_flagged" if ambiguity_flagged else "surface_reviewed",
        reviewer_id=reviewer_id,
        label_source="human" if mode == "manual" else "independent_judge",
        review_basis="surface_only",
    )


def _interactive_surface_decision(
    segmentation: SurfaceSegmentation,
    *,
    max_rounds: int = 3,
) -> tuple[list[int], bool]:
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    marker = segmentation.marker
    print("\n原始需求：\n" + segmentation.source_text)
    print(
        "\n带可视边界的完整原文：\n"
        + marker.join(segment.text for segment in segmentation.segments)
    )
    print("\n当前阅读片段：")
    for segment in segmentation.segments:
        print(f"  {segment.segment_id} [{segment.start}:{segment.end}] {segment.text!r}")
    print(
        "\n操作：a=接受，a!=接受并标记歧义；"
        f"e/e! 后输入只插入 {marker} 的完整原文。"
    )

    last_error = ""
    for round_index in range(1, max_rounds + 1):
        raw = input(f"表面切分审核 [{round_index}/{max_rounds}]: ")
        try:
            return _parse_interactive_surface_input(segmentation, raw)
        except SegmentationValidationError as exc:
            validation = exc.validation
            details = validation.missing_or_changed_excerpt or validation.instruction
            offset = (
                f"，首个差异 codepoint offset={validation.first_difference_offset}"
                if validation.first_difference_offset is not None
                else ""
            )
            last_error = f"{validation.failure_type}{offset}"
            print(f"输入无效：{last_error}。{details}")
        except ValueError as exc:
            last_error = str(exc)
            print(f"输入无效：{last_error}")

    raise PreIRGateError(
        f"人工表面切分审核在 {max_rounds} 次输入后仍未得到有效决定：{last_error}",
        stage="surface_review",
        status="review_pending",
    )


def _parse_interactive_surface_input(
    segmentation: SurfaceSegmentation,
    raw: str,
) -> tuple[list[int], bool]:
    """Parse one human surface-review command without asking an LLM for spans."""
    decision = replacement_surface_decision_from_command(segmentation, raw)
    return list(decision.cut_offsets), decision.ambiguity_flagged


def _parse_cut_offsets(raw: str, source_length: int) -> list[int]:
    tokens = [item.strip() for item in raw.split(",")]
    if not tokens or any(
        not token or not token.isascii() or not token.isdigit() for token in tokens
    ):
        raise ValueError(
            "请输入 a/a!，或 e/e! + 带 marker 的完整原文"
        )
    offsets = [int(token) for token in tokens]
    if offsets != sorted(set(offsets)):
        raise ValueError("codepoint offsets 必须严格递增且不能重复")
    if any(offset <= 0 or offset >= source_length for offset in offsets):
        raise ValueError(f"每个 codepoint offset 必须满足 0 < offset < {source_length}")
    return offsets


def _replace_revision_request(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    request: PlainReadingRevisionRequest,
) -> PlainReadingReview:
    target = next(
        (
            ambiguity
            for ambiguity in review.ambiguities
            if ambiguity.ambiguity_id == request.ambiguity_id
        ),
        None,
    )
    if (
        target is not None
        and target.decision_status == "options_revision_requested"
        and request.request_status
        not in {"submitted", "suggested", "confirmed"}
    ):
        raise PlainReadingValidationError(
            "内部修订状态转换错误: "
            f"{request.request_id} 已是 {request.request_status}，"
            "不得写回 options_revision_requested review；"
            "请通过 close_plain_reading_revision_request 或 "
            "apply_plain_reading_revision 原子转换"
        )
    requests = tuple(
        request if item.request_id == request.request_id else item
        for item in review.revision_requests
    )
    if request.request_id not in {item.request_id for item in requests}:
        requests = (*requests, request)
    return build_plain_reading_review(
        requirement_id=review.requirement_id,
        segmentation=segmentation,
        ambiguities=review.ambiguities,
        overall_status=review.overall_status,
        review_basis=review.review_basis,
        reviewer_source=review.reviewer_source,
        generator=review.generator,
        experiment_mode=review.experiment_mode,
        data_split=review.data_split,
        review_history=review.review_history,
        revision_requests=requests,
        no_ambiguity_confirmation=review.no_ambiguity_confirmation,
        ambiguity_generation_trigger=review.ambiguity_generation_trigger,
        downstream_status=review.downstream_status,
        supersedes_reading_review_id=review.reading_review_id,
    )


def _process_plain_reading_revisions(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    requirement: str,
    retriever: KeywordRetriever,
    llm_client: JSONLLMClient | None,
    reviewer_id: str,
    interactive: bool,
) -> PlainReadingReview:
    """Let a human classify each R request, then rebuild and confirm its axes."""

    agent = PlainReadingRevisionAgent(retriever, llm_client=llm_client)
    result = review
    active_ids = tuple(result.pending_revision_request_ids)
    for request_id in active_ids:
        request = next(
            item for item in result.revision_requests if item.request_id == request_id
        )
        ambiguity = next(
            (
                item
                for item in result.ambiguities
                if item.ambiguity_id == request.ambiguity_id
            ),
            None,
        )
        if ambiguity is None:
            raise PreIRGateError(
                "普通读法修订请求引用了过期歧义轴",
                stage="plain_reading_revision",
                status="blocked_error",
            )
        if not interactive:
            return result

        print("\n普通读法修订请求（分类必须由人工决定）")
        print("  请求: " + request.proposed_interpretation)
        print("  当前歧义轴: " + ambiguity.question)
        for option in ambiguity.options:
            print(f"    {option.option_id}. {option.plain_interpretation}")
        if request.source_warnings:
            print("  机械来源警告（不自动决定分类）: " + "; ".join(request.source_warnings))
        allowed = {
            "e": "equivalent_to_existing",
            "s": "same_axis_revision",
            "i": "new_independent_axis",
            "o": "out_of_source",
        }
        rebuild_failures = 0
        while True:
            raw = input(
                "人工分类 E=等价/S=同轴重建（替换原轴）/"
                "I=独立新轴（保留原轴）/O=超出原文/"
                "B=返回原 A/B/C=需求澄清（无默认值）: "
            ).strip().casefold()
            if raw in {"b", "back"}:
                result = close_plain_reading_revision_request(
                    result,
                    segmentation,
                    request,
                    action="return_to_review",
                    reviewer_id=reviewer_id,
                )
                break
            if raw in {"c", "clarify"}:
                basis = input("请输入需求澄清依据（不能为空）: ").strip()
                if not basis:
                    print("输入无效：需求澄清依据不能为空。")
                    continue
                clarification_request = replace(
                    request,
                    classification="out_of_source",
                    classification_basis=basis,
                )
                result = close_plain_reading_revision_request(
                    result,
                    segmentation,
                    clarification_request,
                    action="needs_source_clarification",
                    reviewer_id=reviewer_id,
                )
                break
            classification = allowed.get(raw)
            if classification is None:
                print("输入无效：必须由人工明确输入 E/S/I/O/B/C，不能留空。")
                continue
            basis = input("请输入人工分类依据（不能为空）: ").strip()
            if not basis:
                print("输入无效：人工分类依据不能为空。")
                continue
            try:
                confirmed = confirm_plain_reading_revision_classification(
                    request,
                    classification=classification,
                    classification_basis=basis,
                    confirmed_by=reviewer_id,
                )
            except PlainReadingValidationError as exc:
                print("分类无效：" + str(exc))
                continue
            if classification == "out_of_source":
                result = close_plain_reading_revision_request(
                    result,
                    segmentation,
                    confirmed,
                    action="return_to_review",
                    reviewer_id=reviewer_id,
                )
                break

            axes_to_apply = ()
            acknowledge = False
            if classification in {"same_axis_revision", "new_independent_axis"}:
                if rebuild_failures >= 2:
                    print("二元轴重建已达 2 次上限；请选择 B 返回或 C 请求澄清。")
                    continue
                try:
                    axes_to_apply = agent.rebuild_axes(
                        source_text=requirement,
                        segmentation=segmentation,
                        ambiguity=ambiguity,
                        request=request,
                        human_classification=classification,
                    )
                except (PlainReadingValidationError, TypeError, ValueError) as exc:
                    rebuild_failures += 1
                    print("二元轴重建无效：" + str(exc))
                    continue
                if not axes_to_apply:
                    rebuild_failures += 1
                    print("修订器未生成二元重建轴；请重试分类，或选择 B/C。")
                    continue
                print("  修订器按人工分类生成的候选轴：")
                for axis in axes_to_apply:
                    print("  重建轴: " + axis.question)
                    for option in axis.options:
                        print(f"    {option.option_id}. {option.plain_interpretation}")
                warnings = validate_rebuilt_plain_reading_axes(
                    requirement,
                    segmentation,
                    axes_to_apply,
                )
                if warnings:
                    print("  重建轴来源一致性警告: " + "; ".join(warnings))
                confirmation_prompt = (
                    "人工确认应用以上重建轴（新增独立轴并保留原轴）？[y/N]: "
                    if classification == "new_independent_axis"
                    else "人工确认应用以上重建轴（替换原轴）？[y/N]: "
                )
                accept_axes = input(confirmation_prompt).strip().casefold() in {
                    "y",
                    "yes",
                    "是",
                }
                if not accept_axes:
                    rebuild_failures += 1
                    print("重建轴未获人工确认，未应用任何修改。")
                    continue
                acknowledge = bool(warnings)
            try:
                result = apply_plain_reading_revision(
                    result,
                    segmentation,
                    confirmed,
                    replacement_axes=axes_to_apply,
                    reviewer_id=reviewer_id,
                    acknowledge_rebuilt_source_warnings=acknowledge,
                )
            except PlainReadingValidationError as exc:
                raise PreIRGateError(
                    str(exc),
                    stage="plain_reading_revision",
                    status="blocked_error",
                ) from exc
            if classification == "new_independent_axis":
                print(
                    f"重建轴已应用：新增 {len(axes_to_apply)} 个独立轴，"
                    "原歧义轴按定义保留。"
                )
                print(
                    "接下来会重新确认原轴和新增轴；先显示原轴，"
                    "因此下一题可能与最初相同，并不表示更新失败。"
                )
            elif classification == "same_axis_revision":
                print(
                    f"重建轴已应用：原歧义轴已替换为 {len(axes_to_apply)} 个重建轴，"
                    "接下来请重新确认。"
                )
            break
    return result


def _interactive_no_ambiguity_confirmation(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
) -> tuple[str, tuple[str, ...], str]:
    """Ask a non-expert whether to accept or challenge an LLM empty-axis result."""

    print("\nLLM 认为这句话没有明显的多种读法。你是否同意？")
    for segment in segmentation.segments:
        print(f"  {segment.segment_id}: {segment.display_text}")
    print(
        "说明：E 后的备注只用于定位原文中可能存在的不同读法，不会成为新的需求语义。"
        "如果原文最终无法支持两个读法，后续可用 R 提供并确认完整 replacement。"
    )
    while True:
        raw = input(
            "请选择 A=同意，没有歧义，继续；E=我觉得可能有歧义，请生成不同读法；"
            "C=暂不处理: "
        ).strip().casefold()
        if raw in {"a", "accept", "confirm", "同意", "继续"}:
            comment = input("可选备注（直接回车跳过）: ").strip()
            return "a", (), comment
        if raw in {"e", "b", "generate", "reject", "disagree", "可能有歧义"}:
            available = {segment.segment_id for segment in segmentation.segments}
            selected = input(
                "可选：输入觉得不确定的片段 ID（逗号分隔，直接回车可跳过）: "
            ).strip()
            segment_ids = tuple(
                dict.fromkeys(
                    item.strip()
                    for item in re.split(r"[,，\s]+", selected)
                    if item.strip()
                )
            )
            unknown = set(segment_ids) - available
            if unknown:
                print("片段 ID 无效: " + ", ".join(sorted(unknown)))
                continue
            comment = input(
                "可选：只说明原文哪些词可能有不同读法，不要在这里增加新约束"
                "（直接回车可跳过）: "
            ).strip()
            return "e", segment_ids, comment
        if raw in {"c", "cancel", "later", "defer", "暂不处理"}:
            return "c", (), ""
        print("输入无效，请选择 A、E 或 C。")


def _interactive_semantic_fallback_confirmation(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    original_source_text: str,
    parent_effective_source_hash: str,
    representation_profile: APRepresentationProfile | None = None,
) -> tuple[str, str, bool | None]:
    """Collect the runtime Y/R/C decision and an explicitly confirmed replacement."""

    print("\nLLM 认为这句话没有明显的多种读法。")
    for segment in segmentation.segments:
        print(f"  {segment.segment_id}: {segment.display_text}")
    while True:
        raw = input(
            "请选择 Y=同意，按当前需求继续；R=不同意，提供完整的新需求语义；C=暂不处理: "
        ).strip().casefold()
        if raw in {"y", "yes", "confirm", "同意", "继续"}:
            return "confirm", "", None
        if raw in {"c", "cancel", "later", "暂不处理"}:
            return "defer", "", None
        if raw not in {"r", "replace", "replacement", "不同意"}:
            print("输入无效，请选择 Y、R 或 C。")
            continue
        replacement = input(
            "请用完整、可独立理解的普通语言重新写出你希望系统采用的需求语义: "
        ).strip()
        if representation_profile is not None:
            # V2 first records the complete replacement.  AP inventory/mapping
            # and final activation are collected by later explicit actions.
            return "replace", replacement, None
        override = create_human_semantic_override(
            requirement_id=review.requirement_id,
            original_source_text=original_source_text,
            replacement_text=replacement,
            parent_effective_source_hash=parent_effective_source_hash,
        )
        print("当前 NL: " + original_source_text)
        print("replacement: " + replacement)
        print("当前 NL 的 AP: " + ", ".join(override.original_ap_set))
        print("replacement 的 AP: " + ", ".join(override.replacement_ap_set))
        if override.status == "ap_rejected":
            print("AP-only gate 未通过: " + ", ".join(override.issue_codes))
            continue
        if override.status == "awaiting_input":
            print("replacement 不能为空。")
            continue
        print("系统只检查 AP，不检查其他语义变化。")
        print("确认后，所有旧歧义轴和下游结果都将失效，并从切分重新开始。")
        while True:
            confirmation = input("是否确认激活 replacement？请输入 Y 或 N（无默认值）: ").strip().casefold()
            if confirmation in {"y", "yes", "确认"}:
                return "replace", replacement, True
            if confirmation in {"n", "no", "取消"}:
                return "defer", "", False
            print("输入无效，请明确输入 Y 或 N。")


def _interactive_v2_ap_review(
    *,
    override: HumanSemanticOverride,
    profile: APRepresentationProfile,
    segmentation: SurfaceSegmentation,
    original_inventory: APInventory | None,
    replacement_inventory: APInventory | None,
    reviewer_id: str,
) -> tuple[
    HumanSemanticOverride,
    APInventory,
    APInventory,
    OverrideAPComparison,
]:
    """Collect one explicit AP sub-action; never activate the replacement."""

    if override.status == "awaiting_ap_inventory":
        print("\n当前 replacement 需要独立 AP inventory 审核。")
        print("本步骤只标记命题短语，不确认布尔、时间、否定或因果语义。")
        print("请输入 JSON：{\"original\": [...], \"replacement\": [...]}。")
        print("每项包含 canonical_name 和 evidence_texts。")
        raw_payload = input(
            "AP inventory JSON（或输入 REJECT/C）: "
        ).strip()
        if raw_payload.casefold() in {"c", "cancel", "later", "暂缓"}:
            raise _InteractiveAPReviewAction("defer")
        if raw_payload.casefold() in {"reject", "拒绝", "取消replacement"}:
            raise _InteractiveAPReviewAction("reject")
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise PlainReadingValidationError("AP inventory JSON 无效") from exc
        if not isinstance(payload, Mapping):
            raise PlainReadingValidationError("AP inventory JSON 必须是对象")
        phrase_profile = profile.mode in {
            "natural_language_grounded",
            "undetermined",
            "offline_formula_assisted",
        }
        if phrase_profile:
            original_inventory = propose_grounded_ap_inventory(
                requirement_id=override.requirement_id,
                source_text=override.original_source_text,
                profile=profile,
                proposals=payload.get("original", ()),
                segmentation=segmentation,
            )
        elif original_inventory is None:
            raise PlainReadingValidationError(
                "controlled/catalog AP inventory proposal缺失"
            )
        replacement_proposals = payload.get("replacement", ())
        if phrase_profile:
            replacement_inventory = propose_grounded_ap_inventory(
                requirement_id=override.requirement_id,
                source_text=override.replacement_text,
                profile=profile,
                proposals=replacement_proposals,
                parent_inventory_id=original_inventory.inventory_id,
            )
        elif replacement_inventory is None:
            raise PlainReadingValidationError(
                "controlled/catalog replacement AP inventory proposal缺失"
            )
        approval = input(
            "确认以上两侧 identity 分组请输入 APPROVE（无默认值）: "
        ).strip()
        if approval != "APPROVE":
            raise PlainReadingValidationError("AP inventory 未明确批准")
        original_inventory = approve_ap_inventory(
            original_inventory,
            profile=profile,
            reviewer_id=reviewer_id,
            approved_empty=bool(payload.get("original_approved_empty", False)),
            approved_empty_reason=str(
                payload.get("original_approved_empty_reason", "")
            ).strip(),
            segmentation=segmentation,
        )
        if phrase_profile:
            replacement_inventory = propose_grounded_ap_inventory(
                requirement_id=override.requirement_id,
                source_text=override.replacement_text,
                profile=profile,
                proposals=replacement_proposals,
                parent_inventory_id=original_inventory.inventory_id,
            )
        replacement_inventory = approve_ap_inventory(
            replacement_inventory,
            profile=profile,
            reviewer_id=reviewer_id,
            approved_empty=bool(payload.get("replacement_approved_empty", False)),
            approved_empty_reason=str(
                payload.get("replacement_approved_empty_reason", "")
            ).strip(),
        )
        updated, comparison = advance_human_semantic_override_ap_review(
            override,
            representation_profile=profile,
            original_inventory=original_inventory,
            replacement_inventory=replacement_inventory,
            reviewer_id=reviewer_id,
        )
        return updated, original_inventory, replacement_inventory, comparison

    if original_inventory is None or replacement_inventory is None:
        raise PlainReadingValidationError("AP mapping 审核缺少两侧 approved inventory")
    print("\n请确认 replacement identity 到 original identity 的一一映射。")
    for identity in original_inventory.identities:
        print(f"  original {identity.ap_identity_id}: {identity.canonical_name}")
    for identity in replacement_inventory.identities:
        print(f"  replacement {identity.ap_identity_id}: {identity.canonical_name}")
    print("输入 JSON 数组，每项包含 replacement_ap_identity_id 和 original_ap_identity_id。")
    raw_mapping_text = input(
        "AP mapping JSON（或输入 REJECT/C）: "
    ).strip()
    if raw_mapping_text.casefold() in {"c", "cancel", "later", "暂缓"}:
        raise _InteractiveAPReviewAction("defer")
    if raw_mapping_text.casefold() in {"reject", "拒绝", "取消replacement"}:
        raise _InteractiveAPReviewAction("reject")
    try:
        raw_mapping = json.loads(raw_mapping_text)
    except json.JSONDecodeError as exc:
        raise PlainReadingValidationError("AP mapping JSON 无效") from exc
    if not isinstance(raw_mapping, list):
        raise PlainReadingValidationError("AP mapping JSON 必须是数组")
    approval = input("确认该 identity mapping 请输入 APPROVE（无默认值）: ").strip()
    if approval != "APPROVE":
        raise PlainReadingValidationError("AP mapping 未明确批准")
    updated, comparison = advance_human_semantic_override_ap_review(
        override,
        representation_profile=profile,
        original_inventory=original_inventory,
        replacement_inventory=replacement_inventory,
        identity_mapping=raw_mapping,
        reviewer_id=reviewer_id,
    )
    return updated, original_inventory, replacement_inventory, comparison


def _interactive_semantic_fallback_action(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
) -> str:
    """Collect Y/R/C when this entry point cannot activate a replacement."""

    del review
    print("\nLLM 认为这句话没有明显的多种读法。")
    for segment in segmentation.segments:
        print(f"  {segment.segment_id}: {segment.display_text}")
    while True:
        raw = input(
            "请选择 Y=同意，按当前需求继续；R=转到完整的新需求语义审核；"
            "C=暂不处理: "
        ).strip().casefold()
        action = _normalize_plain_reading_no_axis_action(
            raw,
            semantic_fallback=True,
        )
        if action != "invalid":
            return raw
        print("输入无效，请选择 Y、R 或 C。")


def _interactive_existing_override_confirmation(
    override: HumanSemanticOverride,
) -> bool:
    """Resume an already AP-validated replacement without silently activating it."""

    print("\n检测到尚未确认的完整需求语义 replacement：")
    print("当前 NL: " + override.original_source_text)
    print("replacement: " + override.replacement_text)
    print("当前 NL 的 AP: " + ", ".join(override.original_ap_set))
    print("replacement 的 AP: " + ", ".join(override.replacement_ap_set))
    print("系统只检查 AP；确认后旧歧义轴和下游结果都将失效。")
    while True:
        raw = input("是否确认激活 replacement？请输入 Y 或 N（无默认值）: ").strip().casefold()
        if raw in {"y", "yes", "确认"}:
            return True
        if raw in {"n", "no", "取消"}:
            return False
        print("输入无效，请明确输入 Y 或 N。")


def _interactive_plain_decisions(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    reviewer_id: str = "",
    max_revision_rounds: int = 2,
) -> tuple[dict[str, str], dict[str, PlainReadingRevisionRequest]]:
    decisions: dict[str, str] = {}
    revision_requests: dict[str, PlainReadingRevisionRequest] = {}
    post_revision_scope = active_post_independent_revision_review_scope(review)
    scoped_axis_ids = set(
        post_revision_scope.get("post_revision_review_axis_ids", ())
        if post_revision_scope is not None
        else ()
    )
    latest_revision = next(
        (
            item
            for item in reversed(review.review_history)
            if item.get("operation") == "plain_reading_revision_applied"
        ),
        None,
    )
    original_axis_id = (
        str(latest_revision.get("ambiguity_id", "")) if latest_revision else ""
    )
    new_axis_ids = {
        str(item.get("ambiguity_id", ""))
        for item in (latest_revision or {}).get("new_axes", ())
        if isinstance(item, Mapping)
    }
    classification = str((latest_revision or {}).get("classification", ""))
    axis_count = len(review.ambiguities)
    for axis_index, ambiguity in enumerate(review.ambiguities, start=1):
        axis_role = "歧义轴"
        if classification == "new_independent_axis":
            if ambiguity.ambiguity_id == original_axis_id:
                axis_role = "原歧义轴（独立新增不会替换此轴）"
            elif ambiguity.ambiguity_id in new_axis_ids:
                axis_role = "新增独立歧义轴"
        elif (
            classification == "same_axis_revision"
            and ambiguity.ambiguity_id in new_axis_ids
        ):
            axis_role = "同轴重建后的歧义轴"
        print(
            f"\n[{axis_index}/{axis_count}] {axis_role} "
            f"(ID: {ambiguity.ambiguity_id})"
        )
        print(ambiguity.question)
        for option in ambiguity.options:
            print(f"  {option.option_id}. {option.plain_interpretation}")
            print(f"     差异：{option.difference_summary}")
        can_withdraw = (
            review.ambiguity_generation_trigger.source == "human_disagreement"
            and axis_index == 1
        )
        while True:
            decision_prompt = (
                "请选择 A/B；N=当前轴无实质差异并丢弃；"
                "U=当前轴真实存在但仍无法唯一裁决；R=修订当前选项"
            )
            decisions[ambiguity.ambiguity_id] = input(
                decision_prompt
                + ("；D=看过后认为没有歧义: " if can_withdraw else ": ")
            ).strip()
            command = decisions[ambiguity.ambiguity_id].casefold()
            if command in {"a", "b", "n", "u", "r"} or (
                can_withdraw and command == "d"
            ):
                break
            print("输入无效，请选择 A/B/N/U/R" + ("/D。" if can_withdraw else "。"))
        if command == "d":
            return (
                {item.ambiguity_id: "withdraw" for item in review.ambiguities},
                {},
            )
        if command == "n":
            decisions[ambiguity.ambiguity_id] = "not_distinct"
        elif command == "u":
            decisions[ambiguity.ambiguity_id] = "ambiguous"
            continue
        elif command == "r":
            command = "revision"
        if command != "revision":
            continue
        last_error = ""
        for _ in range(max_revision_rounds):
            proposed = input("请输入完整的补充解释: ").strip()
            reason = "审核者认为当前 A/B 未覆盖该补充解释。"
            segment_ids = ambiguity.source_segment_ids
            try:
                request = create_plain_reading_revision_request(
                    review,
                    segmentation,
                    ambiguity_id=ambiguity.ambiguity_id,
                    proposed_interpretation=proposed,
                    reason=reason,
                    source_segment_ids=segment_ids,
                    reviewer_id=reviewer_id,
                    max_revision_rounds=max_revision_rounds,
                )
                decisions[ambiguity.ambiguity_id] = "revise"
                revision_requests[ambiguity.ambiguity_id] = request
                print(
                    "修订请求已记录，机械状态="
                    f"{request.validation_status}；分类必须由人工明确决定，"
                    "未经人工分类和轴确认不会改变正式选项。"
                )
                break
            except PlainReadingValidationError as exc:
                last_error = str(exc)
                print("补充解释无效：" + last_error)
        else:
            raise PreIRGateError(
                "普通读法修订轮次耗尽：" + last_error,
                stage="plain_reading_revision",
                status="needs_source_clarification",
            )
    return decisions, revision_requests


def _judge_surface_decision(
    judge_client: JSONLLMClient,
    segmentation: SurfaceSegmentation,
) -> tuple[list[int], bool]:
    raw = judge_client.complete_json(
        system_prompt=(
            "Act only as an ordinary-language surface-boundary reviewer. "
            "Do not infer IR, LTLf, operator scope, or author intent."
        ),
        user_payload={
            "source_text": segmentation.source_text,
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in segmentation.segments
            ],
            "output_schema": {
                "decision": "accept or revise",
                "cut_offsets": [10, 24],
                "ambiguity_flagged": False,
            },
        },
        temperature=0.0,
    )
    decision = str(raw.get("decision", "accept")).casefold()
    offsets = (
        [int(item) for item in raw.get("cut_offsets", ())]
        if decision == "revise"
        else [segment.end for segment in segmentation.segments[:-1]]
    )
    return offsets, bool(raw.get("ambiguity_flagged", False))


def _judge_plain_decisions(
    judge_client: JSONLLMClient | None,
    requirement: str,
    review: PlainReadingReview,
    *, decision_reasons: dict[str, str] | None = None,
) -> dict[str, str]:
    """Shared structured option decisions for both reviewer adapters."""
    from llm.schema_validation import validate_response_schema
    if judge_client is None:
        raise PreIRGateError("读法审核缺少模型客户端。", stage="plain_reading", status="reviewer_unavailable")
    if review.source_hash != source_hash(requirement):
        raise PlainReadingValidationError("读法审核原文与记录不一致")
    ids = [axis.ambiguity_id for axis in review.ambiguities]
    if not ids:
        raise PlainReadingValidationError("读法选项审核需要非空歧义集合")
    option_ids = [option.option_id for axis in review.ambiguities for option in axis.options]
    schema = {"type": "object", "properties": {"decisions": {
        "type": "array", "minItems": len(ids), "maxItems": len(ids),
        "items": {"type": "object", "properties": {
            "ambiguity_id": {"type": "string", "enum": ids},
            "decision": {"type": "string", "enum": list(dict.fromkeys([*option_ids, "A", "B", "U", "N"]))},
            "reason": {"type": "string", "minLength": 1},
        }, "required": ["ambiguity_id", "decision", "reason"], "additionalProperties": False},
    }}, "required": ["decisions"], "additionalProperties": False}
    raw = judge_client.complete_json(
        (
            "Choose the reading closer to the sentence's ordinary meaning. "
            "Use U only for a real behavior-changing axis that remains unresolved; "
            "Choose the source-supported option if another option invents a restriction or deadline. "
            "Use N when neither option forms a real source-supported distinction. "
            "Return an exact option_id when selecting a reading. A and B are legacy "
            "aliases for the first and second option of that axis. Do not use formulas, "
            "hidden intent, invented domain assumptions or claims of human approval. "
            + REQUIREMENT_READING_RULES
        ),
        {
            "task": "model_plain_reading_check",
            "source_text": requirement,
            "ambiguities": [{"ambiguity_id": axis.ambiguity_id, "question": axis.question,
                             "source_span": list(axis.source_span), "evidence_text": axis.evidence_text,
                             "options": [asdict(option) for option in axis.options]}
                            for axis in review.ambiguities],
        },
        temperature=0.0,
        response_schema=schema,
    )
    validate_response_schema(raw, schema)
    if decision_reasons is not None:
        decision_reasons.update({item["ambiguity_id"]: item["reason"] for item in raw["decisions"]})
    decisions = {item["ambiguity_id"]: item["decision"] for item in raw["decisions"]}
    if len(decisions) != len(ids) or set(decisions) != set(ids):
        raise PlainReadingValidationError("模型读法检查缺失或重复歧义编号")
    for axis in review.ambiguities:
        choice = decisions[axis.ambiguity_id]
        known = {option.option_id for option in axis.options}
        if choice not in known and choice in {"A", "B"}:
            choice = axis.options[0 if choice == "A" else 1].option_id
        if choice not in {"U", "N", *known}:
            raise PlainReadingValidationError("模型读法检查引用了其他歧义的选项")
        decisions[axis.ambiguity_id] = choice
    return decisions


def _judge_no_ambiguity(judge_client, segmentation, review, *, reviewer_source="independent_judge"):
    if judge_client is None:
        raise PreIRGateError("A separate Judge client is required to confirm an empty ambiguity set",
                             stage="plain_reading", status="judge_unavailable")
    prompt = ("Independently review the generator's proposal that this source has no material ordinary-language ambiguity. "
              "An empty proposal is not approval. Check participants/APs, negation, condition/consequence direction, "
              "coordination and temporal scope using only the source; do not invent domain meaning or consult formulas. "
              "Return confirm_no_ambiguity only if the source supports a determinate reading. Return ambiguity_found "
              "for materially different supported readings and include every detected axis with exactly two "
              "source-supported plain-language alternatives. Return needs_source_clarification for missing meaning. "
              "Give a concrete reason and at least one exact source span with Unicode-code-point start/end. "
              "The dataset contract, AP context, profiles and instructions are non-quotable. Never claim human approval. "
              + REQUIREMENT_READING_RULES)
    payload = {"source_text": segmentation.source_text, "source_hash": segmentation.source_hash,
               "segmentation_id": segmentation.segmentation_id, "proposal_id": review.reading_review_id,
               "proposal_basis": review.review_basis, "proposal": "no material ambiguity detected",
               "segments": [{"text": s.text, "start": s.start, "end": s.end} for s in segmentation.segments]}
    schema = {"type": "object", "properties": {
        "decision": {"type": "string", "enum": ["confirm_no_ambiguity", "ambiguity_found", "needs_source_clarification"]},
        "reason": {"type": "string", "minLength": 1},
        "source_quotes": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
            "start": {"type": "integer", "minimum": 0}, "end": {"type": "integer", "minimum": 1},
            "text": {"type": "string", "minLength": 1}}, "required": ["start", "end", "text"],
            "additionalProperties": False}},
        "ambiguities": {"type": "array", "items": {"type": "object", "properties": {
            "question": {"type": "string", "minLength": 1},
            "source_quote": {"type": "object", "properties": {
                "start": {"type": "integer", "minimum": 0},
                "end": {"type": "integer", "minimum": 1},
                "text": {"type": "string", "minLength": 1}},
                "required": ["start", "end", "text"], "additionalProperties": False},
            "options": {"type": "array", "minItems": 2, "maxItems": 2, "items": {
                "type": "object", "properties": {
                    "plain_interpretation": {"type": "string", "minLength": 1},
                    "difference_summary": {"type": "string", "minLength": 1}},
                "required": ["plain_interpretation", "difference_summary"],
                "additionalProperties": False}}},
            "required": ["question", "source_quote", "options"],
            "additionalProperties": False}}},
        "required": ["decision", "reason", "source_quotes", "ambiguities"], "additionalProperties": False}
    payload["output_schema"] = schema
    response = judge_client.complete_json(prompt, payload, temperature=0.0, response_schema=schema)
    from llm.schema_validation import validate_response_schema
    validate_response_schema(response, schema)
    try:
        quotes = validate_source_quotes(response.get("source_quotes"), segmentation.source_text)
    except SourceQuoteValidationError as exc:
        repair_payload = {
            "task": "repair_source_quote_format",
            "original_response": response,
            "repair_context": quote_repair_context(exc, segmentation.source_text),
            "output_schema": schema,
        }
        repaired = judge_client.complete_json(
            "Repair only the source quote coordinates. Do not change the semantic decision or reason.",
            repair_payload,
            temperature=0.0,
            response_schema=schema,
        )
        validate_response_schema(repaired, schema)
        def semantic_projection(value):
            """Remove quote coordinates while retaining every semantic field."""
            if isinstance(value, Mapping):
                return {
                    key: semantic_projection(child)
                    for key, child in value.items()
                    if key not in {"source_quote", "source_quotes"}
                }
            if isinstance(value, list):
                return [semantic_projection(child) for child in value]
            return value

        if semantic_projection(repaired) != semantic_projection(response):
            raise PlainReadingValidationError(
                "source quote repair attempted to change semantics"
            ) from exc
        try:
            quotes = validate_source_quotes(
                repaired.get("source_quotes"), segmentation.source_text
            )
        except SourceQuoteValidationError as repair_exc:
            raise PlainReadingValidationError(str(repair_exc)) from repair_exc
        response = dict(repaired)
    response = dict(response)
    response["source_quotes"] = list(quotes)
    actual = getattr(judge_client, "client", judge_client)
    model = getattr(actual, "model", None) or getattr(actual, "model_name", None) or type(actual).__name__
    physical = {}
    observer = getattr(actual, "request_observer", None)
    if observer is not None and getattr(observer, "parent", None):
        physical = {"attempt_id": observer.parent}
    reviewed = apply_no_ambiguity_judge_decision(review, segmentation, request={"system": prompt, "payload": payload},
        response=response, judge_model=model, physical_evidence=physical, reviewer_source=reviewer_source)
    emit("no_ambiguity_judge_review", review=reviewed)
    return reviewed


def complete_plain_reading_with_model(requirement, segmentation, review, client):
    """Manual-command adapter; the gate policy is shared with Judge review."""
    return complete_plain_reading_review(
        requirement, segmentation, review, client, reviewer_source="model_assistant",
    )


def complete_plain_reading_review(requirement, segmentation, review, client, *, reviewer_source):
    """One confirmation/state-transition policy; source is audit metadata only."""
    if reviewer_source not in {"model_assistant", "independent_judge"}:
        raise ValueError("Unsupported plain-reading reviewer source")
    if requirement != segmentation.source_text:
        raise PlainReadingValidationError("读法审核原文与切分不一致")
    assert_plain_reading_review_well_formed(review, segmentation)
    if client is None:
        raise PreIRGateError("读法审核缺少模型客户端。", stage="plain_reading", status="reviewer_unavailable")
    for _attempt in range(2):
        if review.overall_status == "pending_no_ambiguity_confirmation":
            review = _judge_no_ambiguity(
                client, segmentation, review, reviewer_source=reviewer_source,
            )
        if review.overall_status in {"pending_plain_review", "ambiguous_unresolved"}:
            reasons = {}
            decisions = _judge_plain_decisions(client, requirement, review, decision_reasons=reasons)
            if review.no_ambiguity_confirmation.judge_evidence_json:
                # Detection evidence is retained, while the option decision has
                # its own provenance and must not reuse an empty-axis approval.
                review = build_plain_reading_review(
                    requirement_id=review.requirement_id, segmentation=segmentation,
                    ambiguities=review.ambiguities, overall_status="ambiguous_unresolved",
                    review_basis="ordinary_language_model_review", reviewer_source=reviewer_source,
                    generator=review.generator, experiment_mode=review.experiment_mode, data_split=review.data_split,
                    revision_requests=review.revision_requests, downstream_status=review.downstream_status,
                    ambiguity_generation_trigger=review.ambiguity_generation_trigger,
                    review_history=(*review.review_history, {
                        "operation": "model_reading_evidence_before_option_adjudication",
                        "old_reading_review_id": review.reading_review_id,
                        "no_ambiguity_confirmation": asdict(review.no_ambiguity_confirmation),
                    }), supersedes_reading_review_id=review.reading_review_id,
                )
            review = apply_plain_reading_decisions(
                review, segmentation, decisions,
                reviewer_id=str(getattr(client, "model", type(client).__name__)),
                reviewer_role="model_assistant" if reviewer_source == "model_assistant" else "model_reviewer",
                reviewer_source=reviewer_source, review_basis="ordinary_language_model_review",
                comments=reasons,
            )
        if review.overall_status != "pending_no_ambiguity_confirmation":
            assert_plain_reading_gate(review, segmentation)
            return review
    raise PreIRGateError("模型未完成读法检查；请重试当前需求。", stage="plain_reading", status="model_check_pending")


def _blocked_segmentation(
    requirement_id: str,
    requirement: str,
    *,
    mode: str,
    data_split: str,
    error: Exception,
) -> SurfaceSegmentation:
    from surface_segmentation import choose_cut_marker, source_hash

    validation = (
        error.validation
        if isinstance(error, SegmentationValidationError)
        else MechanicalSegmentationValidation(
            status="failed",
            exact_reconstruction=False,
            failure_type=type(error).__name__,
            missing_or_changed_excerpt=str(error),
            instruction="检查 segmentation_generation 错误后仅重跑该需求。",
        )
    )
    return SurfaceSegmentation(
        segmentation_id=f"SEG-{requirement_id}-blocked",
        revision=1,
        requirement_id=requirement_id,
        source_text=requirement,
        source_hash=source_hash(requirement),
        marker=choose_cut_marker(requirement),
        segments=(),
        mechanical_validation=validation,
        status="blocked_error",
        blocked_stage="segmentation_generation",
        generator=ArtifactGenerator(mode="api" if mode != "rules" else "rules"),
        experiment_mode=mode,
        data_split=data_split,
        persistent_rag_allowed=False,
    )


def _blocked_plain_reading(
    requirement_id: str,
    segmentation: SurfaceSegmentation,
    *,
    mode: str,
) -> PlainReadingReview:
    return PlainReadingReview(
        reading_review_id=f"READ-{requirement_id}-blocked",
        requirement_id=requirement_id,
        source_hash=segmentation.source_hash,
        segmentation_id=segmentation.segmentation_id,
        overall_status="blocked_error",
        review_basis="blocked_upstream",
        reviewer_source="",
        experiment_mode=mode,
        data_split=segmentation.data_split,
        blocked_stage="plain_reading",
        persistent_rag_allowed=False,
    )


def _deferred_plain_reading(
    requirement_id: str,
    segmentation: SurfaceSegmentation,
    *,
    mode: str,
    waiting_for: str,
) -> PlainReadingReview:
    return PlainReadingReview(
        reading_review_id=f"READ-{requirement_id}-deferred",
        requirement_id=requirement_id,
        source_hash=segmentation.source_hash,
        segmentation_id=segmentation.segmentation_id,
        overall_status="deferred_pending_user_confirmation",
        review_basis=f"waiting_for:{waiting_for}",
        reviewer_source="",
        experiment_mode=mode,
        data_split=segmentation.data_split,
        blocked_stage="plain_reading",
        persistent_rag_allowed=False,
    )
