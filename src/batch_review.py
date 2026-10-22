"""
Batch Markdown review for semantic structures and finite traces.

The batch flow differs from the original terminal-by-terminal review loop:
all requirements are generated first, written into one Markdown file, and then
the human reviewer edits that file with pass/fail statuses and feedback.
Failed items are regenerated with the Markdown feedback in the next round.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import json
import re
from pathlib import Path
import traceback

from agents.base_agent import JSONLLMClient
from agents.nlu_agent import NLUAgent
from agents.trace_agent import (
    TraceGenerationAgent,
    assess_candidate_discrimination_coverage,
    assess_authoritative_trace_coverage,
    assess_trace_coverage,
    trace_coverage_feedback,
)
from semantic_promotion import (
    TRACE_READY,
    AuthoritativeTraceSession,
    create_authoritative_trace_session,
    promote_semantic_candidate,
)
from semantic_source import (
    EffectiveSemanticSource,
    bind_semantic_to_effective_source,
    natural_language_source,
    source_from_ordinary_override,
    validate_semantic_source_binding,
)
from ap_binding_context import (
    APBindingContext,
    build_ap_binding_context,
    text_only_ap_binding_context,
)
from ap_preparation import APPreparationError
from checkpointing import (
    load_human_semantic_override_ap_artifacts,
    load_human_semantic_override_checkpoint,
    load_needs_review_checkpoint,
    load_ap_inventory_checkpoint,
    load_semantic_policy_candidates_checkpoint,
    load_plain_reading_generated_checkpoint,
    load_plain_reading_reviewed_checkpoint,
    load_segmentation_reviewed_checkpoint,
    load_generated_review_checkpoint,
    load_reviewed_item_checkpoint,
    save_failed_checkpoint,
    save_ap_inventory_checkpoint,
    save_needs_review_checkpoint,
    save_ap_identity_artifact_checkpoint,
    save_human_semantic_override_checkpoint,
    save_generated_semantic_checkpoint,
    save_generated_review_checkpoint,
    save_plain_reading_generation_diagnostic,
    save_plain_reading_generated_checkpoint,
    save_plain_reading_need_view_checkpoint,
    save_plain_reading_reviewed_checkpoint,
    save_reviewed_item_checkpoint,
    save_semantic_review_pause_checkpoint,
    save_semantic_policy_candidates_checkpoint,
    save_semantic_policy_decision_checkpoint,
    save_segmentation_generated_checkpoint,
    save_segmentation_generation_diagnostic,
    save_segmentation_reviewed_checkpoint,
    semantic_from_checkpoint,
    trace_coverage_from_checkpoint,
    trace_evidence_dispute_from_checkpoint,
    trace_evidence_record_from_checkpoint,
    trace_from_checkpoint,
)
from data_examples import LabeledTrace, trace_semantics_to_dict
from guided_decision import build_guided_stage_decision
from guided_learning import GuidedLearningContext
from faithful_ir import (
    FaithfulIRValidationError,
    trace_generation_readiness_errors,
)
from pattern_registry import parse_learned_formula, pattern_requires_human_learning
from pattern_review import run_pattern_learning_review
from plain_reading import (
    PlainReadingValidationError,
    assess_plain_reading_ir_readiness,
    decision_fingerprint,
    plain_reading_from_mapping,
)
from plain_reading_need_view import build_plain_reading_need_view
from pre_ir_review import (
    PreIRGateError,
    PreIRReviewConfig,
    requires_interactive_pre_ir_review,
    resume_plain_reading_review,
    resume_semantic_fallback_pre_ir,
    run_pre_ir_review,
)
from rag.example_memory import (
    collect_approved_example_chunks,
    extend_retriever_once,
    record_approved_review_examples,
)
from rag.knowledge_base import KnowledgeChunk
from rag.memory_store import build_persistent_retriever
from rag.retriever import KeywordRetriever
from rag.runtime_memory import (
    record_guided_stage_decision,
    record_stage_errors,
    record_trace_length_issue,
)
from schemas import (
    APRepresentationProfile,
    BlockedFailureRecord,
    DeclaredAPCatalog,
    ExperimentConfig,
    HumanSemanticOverride,
    PlainReadingReview,
    PreIRDisposition,
    ReviewPendingRecord,
    SemanticStructure,
    StageErrorRecord,
    SurfaceSegmentation,
    TraceCoverageReport,
    TraceEvidencePolicy,
)
from surface_segmentation import (
    segmentation_from_mapping,
    source_hash,
)
from trace_annotation import oracle_annotations
from trace_conflict_resolution import resolve_trace_evidence_disputes
from trace_evidence import (
    TraceEvidenceDispute,
    TraceEvidenceRecord,
    reconcile_trace_evidence,
    reuse_confirmed_annotations,
)
from semantic_adjudication import (
    attach_semantic_challenges,
    attach_semantic_resolution_record,
    bind_trace_queries_to_brief,
    build_trace_adjudication_brief,
)
from trace_adjudication_policy import adjudicate_trace_round
from ir_ambiguity_adjudication import (
    adjudicate_ir_candidate_branches,
    requires_ir_candidate_adjudication,
)
from ir_compiler import compile_executable_trace_plan
from ir_relation_feasibility import IRRelationFeasibilityError
from semantic_policy_branching import is_policy_candidate_semantic
from semantic_readiness import (
    CAN_BIND_RESOLUTION,
    CAN_GENERATE_AUTHORITATIVE_TRACES,
    audit_semantic_readiness,
)
from semantic_freezing import attach_candidate_ir_set
from trace_review_protocol import (
    manual_trace_annotations,
    annotation_record as build_trace_annotation_record,
    frozen_train_gold_annotations,
    independent_judge_annotations,
)
from terminal_progress import TerminalProgress
from ltlf_syntax import SUPPORTED_LTLF_FORMULA_GUIDANCE
from trace_formatter import trace_to_serializable
from boundary_trace_fixture import (
    BoundaryTraceFixture,
    BoundaryTraceFixtureError,
    bind_frozen_boundary_evidence,
    load_boundary_trace_fixture,
)
from trace_evidence_policy import (
    build_trace_evidence_context,
    filter_queries_for_policy,
    filter_traces_for_policy,
)
from utils import format_trace, format_truth


SEMANTIC_FEEDBACK_START = "<!-- semantic_feedback:start -->"
SEMANTIC_FEEDBACK_END = "<!-- semantic_feedback:end -->"
TRACE_FEEDBACK_START = "<!-- trace_feedback:start -->"
TRACE_FEEDBACK_END = "<!-- trace_feedback:end -->"
WARMUP_FIXTURE_LABEL_SOURCE = "warmup_fixture"

_NORMAL_PRE_IR_PENDING_STATUSES = {
    "pending_no_ambiguity_confirmation",
    "user_requested_ambiguity_generation",
    "pending_plain_review",
    "ambiguous_unresolved",
    "needs_source_clarification",
    "options_revision_requested",
    "awaiting_human_semantic_override",
    "awaiting_confirmation",
    "awaiting_ap_inventory",
    "awaiting_ap_mapping_confirmation",
    "awaiting_source_modality_choice",
    "ap_rejected",
}
_TYPED_REVIEW_PENDING_STATUSES = frozenset(
    {
        "waiting_for_human_semantic_choice",
        "trace_query_regeneration_required",
        "trace_annotation_required",
        "semantic_review_required",
    }
)
_TYPED_REVIEW_PENDING_ERROR_TYPES = frozenset(
    {
        "UnresolvedTraceEvidence",
        "PendingSemanticChallenge",
        "PendingTraceAdjudication",
    }
)


@dataclass
class BatchReviewDecision:
    """The parsed review decision for one requirement in the batch file."""

    requirement_id: str
    semantic_status: str
    trace_status: str
    semantic_feedback: str = ""
    trace_feedback: str = ""
    trace_issue_type: str = "content"
    target_trace_length: int | None = None


@dataclass
class ReviewedBatchItem:
    """A reviewed requirement, or a pre-IR-blocked item retained for reporting."""

    requirement_id: str
    requirement: str
    semantic: SemanticStructure | None
    traces: list[LabeledTrace]
    retriever: KeywordRetriever
    review_records: list[dict]
    guided_context: GuidedLearningContext | None = None
    trace_coverage: TraceCoverageReport | None = None
    experiment_config: ExperimentConfig | None = None
    surface_segmentation: SurfaceSegmentation | None = None
    plain_reading_review: PlainReadingReview | None = None
    pre_ir_error: dict | None = None
    plain_reading_resolution_pending: dict | None = None
    semantic_policy_pending: dict | None = None
    original_requirement: str = ""
    human_semantic_override: HumanSemanticOverride | None = None
    trace_evidence_records: tuple[TraceEvidenceRecord, ...] = ()
    trace_evidence_disputes: tuple[TraceEvidenceDispute, ...] = ()
    ap_binding_context: APBindingContext | None = None
    needs_review_error: dict | None = None
    pre_ir_disposition: PreIRDisposition | None = None
    authoritative_trace_session: AuthoritativeTraceSession | None = None
    effective_semantic_source: EffectiveSemanticSource | None = None


@dataclass
class _BatchItemState:
    requirement_id: str
    requirement: str
    original_requirement: str
    retriever: KeywordRetriever
    nlu_agent: NLUAgent
    trace_agent: TraceGenerationAgent
    semantic: SemanticStructure | None = None
    curated_semantic: bool = False
    expected_semantic: SemanticStructure | None = None
    traces: list[LabeledTrace] = field(default_factory=list)
    curated_traces: list[LabeledTrace] = field(default_factory=list)
    curated_trace_source: str = ""
    expected_traces: list[LabeledTrace] = field(default_factory=list)
    guided_context: GuidedLearningContext | None = None
    semantic_feedback_history: list[str] = field(default_factory=list)
    trace_feedback_history: list[str] = field(default_factory=list)
    target_trace_length: int | None = None
    semantic_approved: bool = False
    trace_approved: bool = False
    regenerate_semantic: bool = True
    regenerate_traces: bool = True
    review_records: list[dict] = field(default_factory=list)
    trace_coverage: TraceCoverageReport = field(default_factory=TraceCoverageReport)
    experiment_config: ExperimentConfig | None = None
    surface_segmentation: SurfaceSegmentation | None = None
    plain_reading_review: PlainReadingReview | None = None
    pre_ir_error: dict | None = None
    plain_reading_resolution_pending: dict | None = None
    semantic_policy_pending: dict | None = None
    human_semantic_override: HumanSemanticOverride | None = None
    trace_evidence_records: list[TraceEvidenceRecord] = field(default_factory=list)
    trace_evidence_disputes: list[TraceEvidenceDispute] = field(default_factory=list)
    trace_evidence_policy: TraceEvidencePolicy = field(default_factory=TraceEvidencePolicy)
    boundary_fixture: BoundaryTraceFixture | None = None
    ap_representation_profile: APRepresentationProfile | None = None
    declared_ap_catalog: DeclaredAPCatalog | None = None
    ap_binding_context: APBindingContext | None = None
    active_ap_inventory: object | None = None
    needs_review_error: dict | None = None
    pre_ir_disposition: PreIRDisposition | None = None
    authoritative_trace_session: AuthoritativeTraceSession | None = None
    effective_semantic_source: EffectiveSemanticSource | None = None

    @property
    def approved(self) -> bool:
        return self.semantic_approved and self.trace_approved

    @property
    def terminal_for_current_run(self) -> bool:
        return bool(
            self.approved
            or self.pre_ir_error is not None
            or self.plain_reading_resolution_pending is not None
            or self.semantic_policy_pending is not None
            or self.needs_review_error is not None
        )


def _update_state_effective_source(state: _BatchItemState) -> None:
    override = state.human_semantic_override
    if override is not None and override.status == "active":
        source = source_from_ordinary_override(override)
    else:
        source = state.effective_semantic_source
        if source is None:
            source = natural_language_source(
                requirement_id=state.requirement_id,
                source_text=state.requirement,
            )
    if source.source_text != state.requirement:
        raise ValueError(
            "effective semantic source changed without an active authority record"
        )
    state.effective_semantic_source = source


def _bind_or_validate_state_semantic_source(
    state: _BatchItemState,
) -> SemanticStructure:
    if state.semantic is None or state.semantic.faithful_ir is None:
        raise ValueError("semantic source binding requires FaithfulSemanticIR")
    source = state.effective_semantic_source
    if source is None:
        raise ValueError("batch state lacks EffectiveSemanticSource")
    current = state.semantic
    binding_values = (
        current.semantic_source_contract_version,
        current.effective_source_fingerprint,
        current.effective_source_kind,
        current.effective_source_authority_record_id,
    )
    if not any(binding_values):
        current = bind_semantic_to_effective_source(current, source)
    else:
        validate_semantic_source_binding(current)
    if (
        current.effective_source_fingerprint != source.fingerprint
        or current.faithful_ir.source_hash != source.source_hash
    ):
        raise ValueError("semantic artifact references a stale effective source")
    return current


def _assess_state_trace_coverage(
    state: _BatchItemState,
    traces: list[LabeledTrace],
    **kwargs: object,
) -> TraceCoverageReport:
    """Use the promoted-session gate for every production Faithful-IR audit."""

    if state.semantic is None:
        raise RuntimeError("trace coverage requires semantic context")
    if state.semantic.faithful_ir is None:
        return assess_trace_coverage(traces, state.semantic, **kwargs)
    if state.authoritative_trace_session is None:
        raise RuntimeError(
            "Faithful-IR trace coverage requires an authoritative trace session"
        )
    return assess_authoritative_trace_coverage(
        traces,
        state.semantic,
        trace_session=state.authoritative_trace_session,
        **kwargs,
    )


def _generate_state_trace_queries(
    state: _BatchItemState,
    **kwargs: object,
) -> list:
    """Use the promoted-session gate for every production Faithful-IR query."""

    if state.semantic is None:
        raise RuntimeError("trace generation requires semantic context")
    if state.semantic.faithful_ir is None:
        return state.trace_agent.generate_queries(
            requirement=state.requirement,
            semantic=state.semantic,
            **kwargs,
        )
    if state.authoritative_trace_session is None:
        raise RuntimeError(
            "Faithful-IR trace generation requires an authoritative trace session"
        )
    return state.trace_agent.generate_authoritative_queries(
        requirement=state.requirement,
        semantic=state.semantic,
        trace_session=state.authoritative_trace_session,
        **kwargs,
    )


def run_batch_semantic_trace_review(
    requirements: list[str],
    *,
    requirement_ids: list[str] | None = None,
    llm_client: JSONLLMClient | None,
    review_mode: str = "manual",
    max_rounds: int = 3,
    records_dir: str = "review_records",
    initial_approved_chunks: list[KnowledgeChunk] | None = None,
    persistent_memory_file: str | None = None,
    requirement_id_offset: int = 0,
    requirement_id_prefix: str = "REQ",
    curated_semantics_by_id: dict[str, SemanticStructure] | None = None,
    curated_traces_by_id: dict[str, list[LabeledTrace]] | None = None,
    expected_semantics_by_id: dict[str, SemanticStructure] | None = None,
    expected_traces_by_id: dict[str, list[LabeledTrace]] | None = None,
    guided_contexts_by_id: dict[str, GuidedLearningContext] | None = None,
    experiment_configs_by_id: dict[str, ExperimentConfig] | None = None,
    checkpoint_dir: str | None = None,
    force_rerun_stages: dict[str, str] | None = None,
    rerun_feedback: dict[str, dict] | None = None,
    progress: TerminalProgress | None = None,
    experiment_config: ExperimentConfig | None = None,
    judge_client: JSONLLMClient | None = None,
    segmentation_review: str = "rules",
    plain_reading_review: str = "off",
    max_segmentation_retries: int = 2,
    max_surface_review_rounds: int = 3,
    max_plain_reading_rounds: int = 2,
    max_plain_reading_revision_rounds: int = 2,
    segmentation_prompt_version: str = "plain-segmentation-v1",
    plain_reading_prompt_version: str = "plain-reading-v5",
    plain_reading_semantic_fallback: bool = True,
    trace_evidence_policy: TraceEvidencePolicy = TraceEvidencePolicy(),
    ap_representation_profiles_by_id: dict[
        str, APRepresentationProfile
    ] | None = None,
    declared_ap_catalogs_by_id: dict[str, DeclaredAPCatalog] | None = None,
    ap_binding_contexts_by_id: dict[str, APBindingContext] | None = None,
    effective_semantic_sources_by_id: dict[
        str, EffectiveSemanticSource
    ] | None = None,
) -> list[ReviewedBatchItem]:
    """
    Generate and review semantic structures/traces for all requirements.

    In manual mode, each round writes one Markdown review file and waits for the
    reviewer to edit it. In auto mode, the same file is saved with pass statuses.
    """
    experiment_config = experiment_config or ExperimentConfig(mode="human")
    boundary_fixture = (
        load_boundary_trace_fixture(trace_evidence_policy.boundary_fixture_path)
        if trace_evidence_policy.boundary_source == "frozen_file"
        else None
    )
    if (
        boundary_fixture is not None
        and (
            boundary_fixture.sha256
            != trace_evidence_policy.boundary_fixture_sha256
            or boundary_fixture.schema_version
            != trace_evidence_policy.boundary_fixture_schema_version
        )
    ):
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            "boundary fixture content or schema changed after run policy was created",
        )

    if not requirements:
        raise ValueError("批量运行至少需要一条自然语言需求。")
    if requirement_ids is not None:
        if len(requirement_ids) != len(requirements):
            raise ValueError("requirement_ids must align one-to-one with requirements")
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("requirement_ids must be unique")

    approved_chunks = list(initial_approved_chunks or [])
    curated_semantics_by_id = curated_semantics_by_id or {}
    curated_traces_by_id = curated_traces_by_id or {}
    expected_semantics_by_id = expected_semantics_by_id or {}
    expected_traces_by_id = expected_traces_by_id or {}
    guided_contexts_by_id = guided_contexts_by_id or {}
    experiment_configs_by_id = experiment_configs_by_id or {}
    ap_representation_profiles_by_id = ap_representation_profiles_by_id or {}
    declared_ap_catalogs_by_id = declared_ap_catalogs_by_id or {}
    ap_binding_contexts_by_id = ap_binding_contexts_by_id or {}
    effective_semantic_sources_by_id = effective_semantic_sources_by_id or {}
    states = []
    for position, (index, requirement) in enumerate(
        enumerate(requirements, start=1 + requirement_id_offset)
    ):
        stable_requirement_id = (
            requirement_ids[position]
            if requirement_ids is not None
            else f"{requirement_id_prefix.upper()}-{index:03d}"
        )
        states.append(
            _build_batch_state(
                index=index,
                requirement_id=stable_requirement_id,
                requirement=requirement,
                llm_client=llm_client,
                approved_chunks=approved_chunks,
                persistent_memory_file=persistent_memory_file,
                requirement_id_prefix=requirement_id_prefix,
                curated_semantic=curated_semantics_by_id.get(
                    stable_requirement_id
                ),
                curated_traces=curated_traces_by_id.get(
                    stable_requirement_id,
                    [],
                ),
                expected_semantic=expected_semantics_by_id.get(
                    stable_requirement_id
                ),
                expected_traces=expected_traces_by_id.get(
                    stable_requirement_id,
                    [],
                ),
                guided_context=guided_contexts_by_id.get(stable_requirement_id),
                experiment_config=experiment_configs_by_id.get(
                    stable_requirement_id,
                    experiment_config,
                ),
                trace_evidence_policy=trace_evidence_policy,
                boundary_fixture=boundary_fixture,
                ap_representation_profile=ap_representation_profiles_by_id.get(
                    stable_requirement_id
                ),
                declared_ap_catalog=declared_ap_catalogs_by_id.get(
                    stable_requirement_id
                ),
                ap_binding_context=ap_binding_contexts_by_id.get(
                    stable_requirement_id
                ),
                effective_semantic_source=effective_semantic_sources_by_id.get(
                    stable_requirement_id
                ),
            )
        )
    _apply_structured_rerun_feedback(states, rerun_feedback or {})
    if progress is not None:
        progress.info(
            f"{requirement_id_prefix.upper()} batch prepared: {len(states)} item(s)"
        )
    _prepare_pre_ir_states(
        states,
        llm_client=llm_client,
        judge_client=judge_client,
        checkpoint_dir=checkpoint_dir,
        force_rerun_stages=force_rerun_stages or {},
        config=PreIRReviewConfig(
            segmentation_review=segmentation_review,
            plain_reading_review=plain_reading_review,
            max_segmentation_retries=max_segmentation_retries,
            max_surface_review_rounds=max_surface_review_rounds,
            max_plain_reading_rounds=max_plain_reading_rounds,
            max_plain_reading_revision_rounds=max_plain_reading_revision_rounds,
            segmentation_prompt_version=segmentation_prompt_version,
            plain_reading_prompt_version=plain_reading_prompt_version,
            data_split=(
                experiment_config.data_split
                if experiment_config is not None
                else "train" if requirement_id_prefix.upper() == "WARM" else "test"
            ),
            semantic_fallback_enabled=plain_reading_semantic_fallback,
        ),
        progress=progress,
    )
    if checkpoint_dir:
        _restore_batch_checkpoints(
            states=states,
            checkpoint_dir=checkpoint_dir,
            force_rerun_stages=force_rerun_stages or {},
            progress=progress,
        )
        restored_chunks = collect_approved_example_chunks(
            [state.retriever for state in states]
        )
        if restored_chunks:
            _share_approved_chunks(states, restored_chunks)
            approved_chunks = collect_approved_example_chunks(
                [state.retriever for state in states]
            )
        if all(state.terminal_for_current_run for state in states):
            if progress is not None:
                progress.ok(
                    f"{requirement_id_prefix.upper()} review restored from checkpoint: "
                    f"{len(states)}/{len(states)}"
                )
            return [_state_to_reviewed_item(state) for state in states]

    effective_max_rounds = (
        1 if any(state.guided_context for state in states) else max_rounds
    )
    for round_index in range(1, effective_max_rounds + 1):
        if progress is not None:
            progress.info(
                f"{requirement_id_prefix.upper()} review round "
                f"{round_index}/{max_rounds}"
            )
        for item_index, state in enumerate(states, start=1):
            if state.terminal_for_current_run:
                continue
            try:
                _regenerate_batch_item(
                    state,
                    checkpoint_dir=checkpoint_dir,
                    progress=progress,
                    round_index=round_index,
                    item_index=item_index,
                    total_items=len(states),
                    review_mode=review_mode,
                    records_dir=records_dir,
                    allow_degraded=round_index == effective_max_rounds,
                    judge_client=judge_client,
                )
            except Exception as exc:
                suggested_rerun_stage = str(
                    getattr(exc, "suggested_rerun_stage", "")
                )
                if not suggested_rerun_stage and isinstance(
                    exc, FaithfulIRValidationError
                ):
                    suggested_rerun_stage = "semantic"
                diagnostics = getattr(exc, "diagnostics", None)
                if state.ap_binding_context is not None:
                    diagnostics = {
                        **(
                            dict(diagnostics)
                            if isinstance(diagnostics, Mapping)
                            else {}
                        ),
                        "ap_profile_mode": state.ap_binding_context.profile_mode,
                        "ap_context_fingerprint": (
                            state.ap_binding_context.context_fingerprint
                        ),
                        "catalog_authority": (
                            state.ap_binding_context.catalog_authority
                        ),
                        "declared_catalog_size": len(
                            state.ap_binding_context.declared_catalog
                        ),
                        "active_inventory_id": (
                            state.ap_binding_context.active_inventory_id
                        ),
                        "active_inventory_hash": (
                            state.ap_binding_context.active_inventory_hash
                        ),
                    }
                if isinstance(diagnostics, dict) and state.semantic is not None:
                    diagnostics = {
                        **diagnostics,
                        "ir_version": (
                            state.semantic.faithful_ir.ir_version
                            if state.semantic.faithful_ir is not None
                            else ""
                        ),
                    }
                    if checkpoint_dir:
                        diagnostics["semantic_snapshot_path"] = str(
                            Path(checkpoint_dir)
                            / "semantic_generated"
                            / f"{state.requirement_id}.json"
                        )
                review_reason = _typed_needs_review_reason(exc, diagnostics)
                if review_reason:
                    suggested_rerun_stage = (
                        suggested_rerun_stage
                        or (
                            "trace"
                            if "RESOURCE_LIMIT" in review_reason
                            else "semantic"
                        )
                    )
                    state.needs_review_error = {
                        "stage": "semantic_trace_review",
                        "status": "needs_review",
                        "reason_code": review_reason,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "suggested_rerun_stage": suggested_rerun_stage,
                        **(
                            dict(diagnostics)
                            if isinstance(diagnostics, dict)
                            else {}
                        ),
                    }
                    state.review_records.append(
                        {
                            "round": round_index,
                            "requirement_id": state.requirement_id,
                            **state.needs_review_error,
                        }
                    )
                    save_needs_review_checkpoint(
                        checkpoint_dir,
                        requirement_id=state.requirement_id,
                        stage="semantic_trace_review",
                        reason_code=review_reason,
                        message=str(exc),
                        suggested_rerun_stage=suggested_rerun_stage,
                        details=state.needs_review_error,
                    )
                    if progress is not None:
                        progress.warn(
                            f"{state.requirement_id} isolated as needs_review: "
                            f"{review_reason}"
                        )
                    continue
                if (
                    isinstance(diagnostics, dict)
                    and diagnostics.get("reason_code") == "HUMAN_REVIEW_REQUIRED"
                ):
                    request = diagnostics.get("semantic_resolution_request")
                    options = diagnostics.get("semantic_resolution_options", [])
                    question = (
                        str(request.get("question", "")).strip()
                        if isinstance(request, dict)
                        else ""
                    )
                    save_semantic_review_pause_checkpoint(
                        checkpoint_dir,
                        requirement_id=state.requirement_id,
                        question=question,
                        request=request if isinstance(request, dict) else None,
                        options=options if isinstance(options, (list, tuple)) else (),
                        diagnostics=diagnostics,
                    )
                    state.pre_ir_error = {
                        "stage": "semantic_policy_review",
                        "status": "waiting_for_human_semantic_choice",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "question": question,
                        "options": options,
                        "suggested_rerun_stage": "semantic",
                        "user_message": (
                            "该需求已暂停，其他需求可继续处理。请先确认这句话在轨迹"
                            "接近末尾时应如何判断，然后从语义阶段重试。"
                        ),
                    }
                    state.review_records.append({
                        "round": round_index,
                        "requirement_id": state.requirement_id,
                        **state.pre_ir_error,
                    })
                    if progress is not None:
                        progress.warn(
                            f"{state.requirement_id} 已暂停等待语义确认"
                            + (f"：{question}" if question else "")
                        )
                    continue
                save_failed_checkpoint(
                    checkpoint_dir,
                    requirement_id=state.requirement_id,
                    stage="semantic_trace_review",
                    error=exc,
                    suggested_rerun_stage=suggested_rerun_stage,
                    details=diagnostics if isinstance(diagnostics, dict) else None,
                )
                raise
            if state.semantic is not None and (
                state.traces or state.trace_evidence_records
                or state.semantic_policy_pending is not None
            ):
                save_generated_review_checkpoint(
                    checkpoint_dir,
                    requirement_id=state.requirement_id,
                    requirement=state.requirement,
                    semantic=state.semantic,
                    traces=state.traces,
                    trace_coverage=state.trace_coverage,
                    surface_segmentation=state.surface_segmentation,
                    plain_reading_review=state.plain_reading_review,
                    trace_evidence_records=state.trace_evidence_records,
                    trace_evidence_disputes=state.trace_evidence_disputes,
                    review_records=state.review_records,
                )
            pattern_chunk_ids = [
                chunk_id
                for record in state.review_records
                if record.get("stage") == "pattern_learning"
                for chunk_id in record.get("runtime_rag_chunks", [])
            ]
            if pattern_chunk_ids:
                new_chunks = _chunks_by_ids(state.retriever, pattern_chunk_ids)
                _share_approved_chunks(states, new_chunks)

        active_states = [
            state for state in states
            if state.pre_ir_error is None
            and state.plain_reading_resolution_pending is None
            and state.semantic_policy_pending is None
            and state.needs_review_error is None
        ]
        if not active_states:
            return [_state_to_reviewed_item(state) for state in states]
        review_path = write_batch_review_markdown(
            states=active_states,
            round_index=round_index,
            records_dir=records_dir,
            auto_approve=True,
        )
        if progress is not None:
            progress.ok(f"Review file written: {review_path}")
        # Humans have already judged individual trajectories. Technical semantic
        # and coverage fields are finalized by the model and mechanical checks.
        decisions = _auto_review_decisions(active_states)
        if progress is not None and review_mode == "auto":
            passed = sum(
                1
                for decision in decisions.values()
                if decision.semantic_status == "pass"
                and decision.trace_status == "pass"
            )
            progress.ok(f"Auto review passed: {passed}/{len(active_states)}")

        for state in states:
            if state.terminal_for_current_run:
                continue
            decision = decisions[state.requirement_id]
            approved_chunk_ids = _apply_batch_decision(
                state=state,
                decision=decision,
                round_index=round_index,
                review_path=review_path,
                records_dir=records_dir,
            )
            if approved_chunk_ids:
                new_chunks = _chunks_by_ids(state.retriever, approved_chunk_ids)
                _share_approved_chunks(states, new_chunks)
                approved_chunks = collect_approved_example_chunks(
                    [state.retriever for state in states]
                )
            if progress is not None:
                if state.approved:
                    progress.ok(f"{state.requirement_id} semantic/traces approved")
                    if state.semantic is not None:
                        save_reviewed_item_checkpoint(
                            checkpoint_dir,
                            requirement_id=state.requirement_id,
                            requirement=state.requirement,
                            semantic=state.semantic,
                            traces=state.traces,
                            review_records=state.review_records,
                            trace_coverage=state.trace_coverage,
                            surface_segmentation=state.surface_segmentation,
                            plain_reading_review=state.plain_reading_review,
                            trace_evidence_records=state.trace_evidence_records,
                            trace_evidence_disputes=state.trace_evidence_disputes,
                        )
                else:
                    progress.warn(
                        f"{state.requirement_id} needs regeneration in next round"
                    )
            elif state.approved and state.semantic is not None:
                save_reviewed_item_checkpoint(
                    checkpoint_dir,
                    requirement_id=state.requirement_id,
                    requirement=state.requirement,
                    semantic=state.semantic,
                    traces=state.traces,
                    review_records=state.review_records,
                    trace_coverage=state.trace_coverage,
                    surface_segmentation=state.surface_segmentation,
                    plain_reading_review=state.plain_reading_review,
                    trace_evidence_records=state.trace_evidence_records,
                    trace_evidence_disputes=state.trace_evidence_disputes,
                )

        if all(state.terminal_for_current_run for state in states):
            if progress is not None:
                progress.ok(
                    f"{requirement_id_prefix.upper()} review completed: "
                    f"{len(states)}/{len(states)}"
                )
            return [_state_to_reviewed_item(state) for state in states]

    for state in states:
        if state.terminal_for_current_run:
            continue
        adjudication_debt = tuple(
            item
            for item in state.trace_coverage.coverage_debt
            if item.startswith(("annotation_uncertain:", "query_infeasible:"))
        )
        if not adjudication_debt:
            continue
        has_query_debt = any(
            item.startswith("query_infeasible:") for item in adjudication_debt
        )
        state.pre_ir_error = {
            "stage": "trace_adjudication",
            "status": (
                "trace_query_regeneration_required"
                if has_query_debt
                else "trace_annotation_required"
            ),
            "error_type": "PendingTraceAdjudication",
            "error": state.trace_coverage.reason,
            "suggested_rerun_stage": "trace",
            "coverage_debt": list(adjudication_debt),
            "ir_version": (
                state.semantic.faithful_ir.ir_version
                if state.semantic is not None
                and state.semantic.faithful_ir is not None
                else ""
            ),
        }
        state.review_records.append(
            {
                "stage": "trace_adjudication",
                "round": max_rounds,
                "requirement_id": state.requirement_id,
                **state.pre_ir_error,
            }
        )

    failed_states = [
        state
        for state in states
        if not state.approved
        and state.pre_ir_error is None
        and state.plain_reading_resolution_pending is None
        and state.semantic_policy_pending is None
        and state.needs_review_error is None
    ]
    if not failed_states:
        return [_state_to_reviewed_item(state) for state in states]
    for state in failed_states:
        readiness_errors = trace_generation_readiness_errors(
            state.semantic.faithful_ir
            if state.semantic is not None
            else None
        )
        if not readiness_errors:
            continue
        readiness = audit_semantic_readiness(state.semantic.faithful_ir)
        state.semantic_policy_pending = {
            "stage": "semantic_readiness",
            "status": "awaiting_semantic_resolution",
            "requirement_id": state.requirement_id,
            "suggested_rerun_stage": "semantic",
            "reason": "; ".join(readiness_errors),
            "required_capability": CAN_GENERATE_AUTHORITATIVE_TRACES,
            "readiness_fingerprint": readiness.fingerprint,
            "findings": [
                {
                    "code": item.code,
                    "path": item.path,
                    "message": item.message,
                    "node_id": item.node_id,
                    "node_type": item.node_type,
                    "parameter_name": item.parameter_name,
                    "ambiguity_id": item.ambiguity_id,
                    "semantic_axis_ids": list(item.semantic_axis_ids),
                }
                for item in readiness.blockers_for(
                    CAN_GENERATE_AUTHORITATIVE_TRACES
                )
            ],
        }
        state.review_records.append(dict(state.semantic_policy_pending))
    failed_states = [
        state for state in failed_states if state.semantic_policy_pending is None
    ]
    if not failed_states:
        return [_state_to_reviewed_item(state) for state in states]
    failed = ", ".join(state.requirement_id for state in failed_states)
    failure_report = {
        "stage": "semantic_trace_review",
        "failed_requirement_ids": [state.requirement_id for state in failed_states],
        "items": [
            {
                "requirement_id": state.requirement_id,
                "semantic_approved": state.semantic_approved,
                "trace_approved": state.trace_approved,
                "trace_coverage_status": state.trace_coverage.status,
                "trace_coverage_reason": state.trace_coverage.reason,
                "coverage_debt": list(state.trace_coverage.coverage_debt),
                "expected_semantic_errors": _expected_warmup_semantic_errors(state),
                "expected_trace_errors": _expected_warmup_trace_errors(state),
                "latest_review_record": (
                    state.review_records[-1] if state.review_records else None
                ),
            }
            for state in failed_states
        ],
    }
    failure_report_path = _save_json_record(
        failure_report,
        records_dir,
        "batch_semantic_trace_failure",
    )
    summaries = []
    for item in failure_report["items"][:8]:
        debt = item["coverage_debt"]
        reason = (
            debt[0]
            if debt
            else item["trace_coverage_reason"]
            or "semantic/trace review decision failed"
        )
        summaries.append(f"{item['requirement_id']}[{reason}]")
    suffix = "" if len(failed_states) <= 8 else f"; ... 另有 {len(failed_states)-8} 项"
    raise RuntimeError(
        "批量语义结构/有限轨迹评审未全部通过: "
        f"{failed}. 诊断文件: {failure_report_path}. "
        "首要原因: " + "; ".join(summaries) + suffix
    )


def _apply_structured_rerun_feedback(
    states: list[_BatchItemState],
    feedback_by_id: dict[str, dict],
) -> None:
    """Seed the requested front-half stage with human/Judge feedback."""

    for state in states:
        packet = feedback_by_id.get(state.requirement_id)
        if not isinstance(packet, dict):
            continue
        stage = str(packet.get("start_stage", "")).strip().lower()
        feedback = str(packet.get("feedback", "")).strip()
        constraints = [
            str(value).strip()
            for value in packet.get("constraints", [])
            if str(value).strip()
        ]
        patch = packet.get("feedback_patch")
        if isinstance(patch, dict):
            constraints.extend(
                value
                for value in (
                    "reopen IR nodes: "
                    + ", ".join(patch.get("reopened_ir_node_ids", ())),
                    "review ambiguity debt: "
                    + ", ".join(patch.get("new_ambiguity_debt", ())),
                    "plan trace obligations: "
                    + ", ".join(
                        patch.get("new_trace_obligation_requests", ())
                    ),
                    "invalidate stale artifacts after approved change: "
                    + ", ".join(
                        patch.get("stale_artifact_declarations", ())
                    ),
                    "feedback patch remains a proposal until this owning stage validates it",
                )
                if value.split(":", 1)[-1].strip()
            )
        message = "\n".join(
            value
            for value in (
                feedback,
                *(f"constraint: {constraint}" for constraint in constraints),
            )
            if value
        )
        if not message:
            continue
        if stage == "semantic":
            state.semantic_feedback_history.append(message)
        elif stage == "trace":
            state.trace_feedback_history.append(message)


def write_batch_review_markdown(
    *,
    states: list[_BatchItemState],
    round_index: int,
    records_dir: str,
    auto_approve: bool = False,
) -> Path:
    """Write one Markdown review file containing every requirement."""

    path = Path(records_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    review_path = path / f"batch_review_round_{round_index}_{timestamp}.md"
    review_path.write_text(
        _format_batch_review_markdown(
            states=states,
            round_index=round_index,
            auto_approve=auto_approve,
        ),
        encoding="utf-8",
    )
    return review_path


def parse_batch_review_markdown(path: str | Path) -> dict[str, BatchReviewDecision]:
    """Parse reviewer statuses and feedback from a batch review Markdown file."""

    review_path = Path(path)
    text = review_path.read_text(encoding="utf-8")
    matches = list(
        re.finditer(r"^##\s+([A-Z]+-\d+)\s*$", text, flags=re.MULTILINE)
    )
    decisions: dict[str, BatchReviewDecision] = {}

    for index, match in enumerate(matches):
        requirement_id = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end]
        decisions[requirement_id] = BatchReviewDecision(
            requirement_id=requirement_id,
            semantic_status=_normalize_status(_read_field(section, "semantic_status")),
            trace_status=_normalize_status(_read_field(section, "trace_status")),
            semantic_feedback=_extract_between(
                section,
                SEMANTIC_FEEDBACK_START,
                SEMANTIC_FEEDBACK_END,
            ),
            trace_feedback=_extract_between(
                section,
                TRACE_FEEDBACK_START,
                TRACE_FEEDBACK_END,
            ),
            trace_issue_type=_normalize_issue_type(
                _read_field(section, "trace_issue_type") or "content"
            ),
            target_trace_length=_read_optional_int(section, "target_trace_length"),
        )

    return decisions


def validate_batch_review_decisions(
    decisions: dict[str, BatchReviewDecision],
    states: list[_BatchItemState],
) -> list[str]:
    """Return human-readable validation errors for unfinished review fields."""

    errors: list[str] = []
    active_states = [
        state
        for state in states
        if not state.approved
        and state.pre_ir_error is None
        and state.plain_reading_resolution_pending is None
        and state.needs_review_error is None
    ]
    for state in active_states:
        decision = decisions.get(state.requirement_id)
        if decision is None:
            errors.append(f"{state.requirement_id}: 缺少评审区块。")
            continue

        if decision.semantic_status not in {"pass", "fail"}:
            errors.append(
                f"{state.requirement_id}: semantic_status 需填写 pass 或 fail。"
            )
        if decision.trace_status not in {"pass", "fail"}:
            errors.append(f"{state.requirement_id}: trace_status 需填写 pass 或 fail。")

        if decision.semantic_status == "pass":
            errors.extend(
                _semantic_trace_consistency_errors(
                    requirement_id=state.requirement_id,
                    semantic=state.semantic,
                    traces=state.traces,
                )
            )
            errors.extend(_expected_warmup_semantic_errors(state))

        if decision.semantic_status == "fail" and not decision.semantic_feedback:
            errors.append(
                f"{state.requirement_id}: 语义结构失败时需填写 Semantic Feedback。"
            )
        if decision.trace_status == "fail" and not decision.trace_feedback:
            errors.append(
                f"{state.requirement_id}: 有限轨迹失败时需填写 Trace Feedback。"
            )
        if (
            decision.trace_status == "fail"
            and decision.trace_issue_type == "length"
            and decision.target_trace_length is None
        ):
            errors.append(
                f"{state.requirement_id}: 轨迹长度错误时需填写 target_trace_length。"
            )
        if decision.semantic_status == "pass" and decision.trace_status == "pass":
            errors.extend(_expected_warmup_trace_errors(state))
            if state.trace_coverage.status in {"incomplete", "rejected"}:
                errors.append(
                    f"{state.requirement_id}: 当前轨迹覆盖状态为 "
                    f"{state.trace_coverage.status}，尚不能标记 pass。"
                )

    return errors


def _build_batch_state(
    *,
    index: int,
    requirement_id: str | None = None,
    requirement: str,
    llm_client: JSONLLMClient | None,
    approved_chunks: list[KnowledgeChunk] | None = None,
    persistent_memory_file: str | None = None,
    requirement_id_prefix: str = "REQ",
    curated_semantic: SemanticStructure | None = None,
    curated_traces: list[LabeledTrace] | None = None,
    expected_semantic: SemanticStructure | None = None,
    expected_traces: list[LabeledTrace] | None = None,
    guided_context: GuidedLearningContext | None = None,
    experiment_config: ExperimentConfig | None = None,
    trace_evidence_policy: TraceEvidencePolicy = TraceEvidencePolicy(),
    boundary_fixture: BoundaryTraceFixture | None = None,
    ap_representation_profile: APRepresentationProfile | None = None,
    declared_ap_catalog: DeclaredAPCatalog | None = None,
    ap_binding_context: APBindingContext | None = None,
    effective_semantic_source: EffectiveSemanticSource | None = None,
) -> _BatchItemState:
    from llm.overlap_client import overlap_client
    if guided_context is None:
        llm_client = overlap_client(llm_client)
    requirement_id = requirement_id or f"{requirement_id_prefix.upper()}-{index:03d}"
    retriever = build_persistent_retriever(persistent_memory_file)
    extend_retriever_once(retriever, list(approved_chunks or []))
    if guided_context is not None:
        retriever.exclude_example(requirement_id, requirement)
    loaded_traces = list(curated_traces or [])
    context = ap_binding_context or build_ap_binding_context(
        requirement_id=requirement_id,
        source_text=requirement,
        representation_profile=ap_representation_profile,
        declared_catalog=declared_ap_catalog,
        provenance=("batch_text_adapter",),
    )
    source = effective_semantic_source or natural_language_source(
        requirement_id=requirement_id,
        source_text=requirement,
    )
    if (
        source.requirement_id != requirement_id
        or source.source_text != requirement
    ):
        raise ValueError("batch effective semantic source lineage mismatch")
    return _BatchItemState(
        requirement_id=requirement_id,
        requirement=requirement,
        original_requirement=requirement,
        retriever=retriever,
        nlu_agent=NLUAgent(retriever, llm_client=llm_client),
        trace_agent=TraceGenerationAgent(retriever, llm_client=llm_client),
        semantic=curated_semantic,
        curated_semantic=curated_semantic is not None,
        expected_semantic=expected_semantic,
        regenerate_semantic=curated_semantic is None,
        traces=loaded_traces,
        curated_traces=loaded_traces,
        curated_trace_source=(
            "warmup 文件中已审核有限轨迹" if loaded_traces else ""
        ),
        expected_traces=list(expected_traces or []),
        guided_context=guided_context,
        regenerate_traces=not bool(loaded_traces),
        experiment_config=experiment_config,
        trace_evidence_policy=trace_evidence_policy,
        boundary_fixture=boundary_fixture,
        ap_representation_profile=(
            ap_representation_profile or context.representation_profile
        ),
        declared_ap_catalog=declared_ap_catalog,
        ap_binding_context=context,
        effective_semantic_source=source,
    )


_TYPED_NEEDS_REVIEW_CODES = frozenset(
    {
        "IR_AP_IDENTITY_COLLAPSE",
        "IR_AP_EVIDENCE_IDENTITY_CONFLICT",
        "IR_AP_REPAIR_UNAUTHORIZED_MERGE",
        "IR_AP_CONTEXT_STALE",
        "IR_AP_CATALOG_AUTHORITY_MISSING",
        "IR_RELATION_NON_CONTINGENT_REVIEW_REQUIRED",
        "IR_RELATION_FEASIBILITY_RESOURCE_LIMIT",
        "IR_RELATION_FEASIBILITY_UNSUPPORTED",
        "IR_WITNESS_RESOURCE_LIMIT",
        "IR_WITNESS_UNSUPPORTED_NODE",
        "IR_WITNESS_UNSAT_IN_DOMAIN",
    }
)


def _typed_needs_review_reason(
    exc: BaseException,
    diagnostics: Mapping[str, object] | None,
) -> str:
    from ap_preparation import APPreparationError
    from plain_reading import PlainReadingSemanticResolutionRequired
    from source_graph_resolution import SourceGraphResolutionRequired
    if isinstance(exc, SourceGraphResolutionRequired):
        return exc.code
    if isinstance(exc, PlainReadingSemanticResolutionRequired):
        return exc.code
    if isinstance(exc, APPreparationError):
        return exc.code
    candidates: list[str] = [
        str(getattr(exc, "code", "")),
        str(exc),
    ]

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) in {"code", "reason_code", "error", "message"}:
                    candidates.append(str(child))
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(diagnostics or {})
    for code in _TYPED_NEEDS_REVIEW_CODES:
        if any(code in candidate for candidate in candidates):
            return code
    if isinstance(exc, IRRelationFeasibilityError):
        return str(exc.code)
    return ""


def _prepare_pre_ir_states(
    states: list[_BatchItemState],
    *,
    llm_client: JSONLLMClient | None,
    judge_client: JSONLLMClient | None,
    checkpoint_dir: str | None,
    force_rerun_stages: dict[str, str],
    config: PreIRReviewConfig,
    progress: TerminalProgress | None,
) -> None:
    """Complete both runtime gates before any call to ``NLUAgent.generate``."""

    for state_index, state in enumerate(states):
        forced_stage = force_rerun_stages.get(state.requirement_id, "")
        segmentation: SurfaceSegmentation | None = None
        reading: PlainReadingReview | None = None
        pending_override: HumanSemanticOverride | None = None
        original_ap_inventory = None
        replacement_ap_inventory = None
        override_ap_comparison = None
        current_ap_inventory = None
        if checkpoint_dir:
            active_override = load_human_semantic_override_checkpoint(
                checkpoint_dir,
                state.requirement_id,
                active_only=True,
            )
            if active_override is not None:
                state.human_semantic_override = active_override
                state.requirement = active_override.replacement_text
                current_ap_inventory = load_ap_inventory_checkpoint(
                    checkpoint_dir,
                    state.requirement_id,
                    current=True,
                    expected_source_hash=source_hash(state.requirement),
                    expected_profile_id=(
                        state.ap_representation_profile.profile_id
                        if state.ap_representation_profile is not None
                        else None
                    ),
                )
                state.active_ap_inventory = current_ap_inventory
            pending_override = load_human_semantic_override_checkpoint(
                checkpoint_dir,
                state.requirement_id,
                pending_only=True,
            )
            if pending_override is not None and pending_override.status in {
                "awaiting_ap_inventory",
                "awaiting_ap_mapping_confirmation",
                "awaiting_confirmation",
            }:
                (
                    original_ap_inventory,
                    replacement_ap_inventory,
                    override_ap_comparison,
                ) = load_human_semantic_override_ap_artifacts(
                    checkpoint_dir,
                    pending_override,
                )
            elif pending_override is not None:
                pending_override = None
        if checkpoint_dir and forced_stage != "segmentation":
            payload = load_segmentation_reviewed_checkpoint(
                checkpoint_dir,
                state.requirement_id,
                expected_source_hash=source_hash(state.requirement),
            )
            mapping = payload.get("surface_segmentation") if payload else None
            if isinstance(mapping, dict):
                segmentation = segmentation_from_mapping(mapping)
        if segmentation is not None and checkpoint_dir and forced_stage not in {
            "segmentation",
            "plain_reading",
        }:
            payload = load_plain_reading_reviewed_checkpoint(
                checkpoint_dir,
                state.requirement_id,
                expected_source_hash=segmentation.source_hash,
                expected_segmentation_id=segmentation.segmentation_id,
            )
            mapping = payload.get("plain_reading_review") if payload else None
            if isinstance(mapping, dict):
                reading = plain_reading_from_mapping(mapping, segmentation)
            if reading is None:
                pending_payload = load_plain_reading_generated_checkpoint(
                    checkpoint_dir,
                    state.requirement_id,
                    expected_source_hash=segmentation.source_hash,
                    expected_segmentation_id=segmentation.segmentation_id,
                )
                pending_mapping = (
                    pending_payload.get("plain_reading_review")
                    if pending_payload
                    else None
                )
                if isinstance(pending_mapping, dict):
                    try:
                        pending_review = plain_reading_from_mapping(
                            pending_mapping,
                            segmentation,
                        )
                        resume_artifact_callback = (
                            lambda current_segmentation, current_review: (
                                save_plain_reading_generated_checkpoint(
                                    checkpoint_dir,
                                    requirement_id=state.requirement_id,
                                    plain_reading_review=current_review,
                                    segmentation_schema_version=(
                                        current_segmentation.schema_version
                                    ),
                                )
                            )
                        )
                        resume_diagnostic_callback = (
                            lambda diagnostic: save_plain_reading_generation_diagnostic(
                                checkpoint_dir,
                                requirement_id=state.requirement_id,
                                diagnostic=diagnostic,
                            )
                        )
                        if (
                            (
                                pending_review.overall_status
                                == "pending_no_ambiguity_confirmation"
                                and pending_review.review_basis
                                == "llm_no_source_supported_axis_after_repair"
                            )
                            or (
                                pending_review.overall_status
                                == "awaiting_human_semantic_override"
                                and pending_review.review_basis
                                == "human_requested_semantic_override"
                            )
                        ):
                            resumed = resume_semantic_fallback_pre_ir(
                                requirement=state.requirement,
                                original_source_text=state.original_requirement,
                                segmentation=segmentation,
                                review=pending_review,
                                retriever=state.retriever,
                                llm_client=llm_client,
                                judge_client=judge_client,
                                config=config,
                                interactive=(
                                    config.plain_reading_review == "manual"
                                ),
                                artifact_callback=resume_artifact_callback,
                                generation_diagnostic_callback=(
                                    resume_diagnostic_callback
                                ),
                                semantic_override_callback=(
                                    lambda override: save_human_semantic_override_checkpoint(
                                        checkpoint_dir,
                                        override=override,
                                    )
                                ),
                                ap_artifact_callback=(
                                    lambda artifact: save_ap_identity_artifact_checkpoint(
                                        checkpoint_dir,
                                        artifact,
                                    )
                                ),
                                ap_representation_profile=(
                                    state.ap_representation_profile
                                ),
                                declared_ap_catalog=state.declared_ap_catalog,
                                original_ap_inventory=original_ap_inventory,
                                replacement_ap_inventory=(
                                    replacement_ap_inventory
                                ),
                                override_ap_comparison=override_ap_comparison,
                                pending_override=pending_override,
                                active_semantic_override=(
                                    state.human_semantic_override
                                ),
                            )
                            segmentation = resumed.surface_segmentation
                            reading = resumed.plain_reading_review
                            state.requirement = resumed.effective_source_text
                            state.human_semantic_override = (
                                resumed.human_semantic_override
                            )
                            _update_state_effective_source(state)
                        else:
                            reading = resume_plain_reading_review(
                                requirement=state.requirement,
                                segmentation=segmentation,
                                review=pending_review,
                                retriever=state.retriever,
                                llm_client=(
                                    judge_client
                                    if config.plain_reading_review == "judge"
                                    else llm_client
                                ),
                                interactive=(
                                    config.plain_reading_review == "manual"
                                    and requires_interactive_pre_ir_review(
                                        config.segmentation_review,
                                        config.plain_reading_review,
                                    )
                                ),
                                max_revision_rounds=(
                                    config.max_plain_reading_revision_rounds
                                ),
                                artifact_callback=resume_artifact_callback,
                                generation_diagnostic_callback=(
                                    resume_diagnostic_callback
                                ),
                                semantic_fallback_enabled=(
                                    config.semantic_fallback_enabled
                                ),
                            )
                    except Exception as exc:
                        if getattr(exc, "status", "") in _NORMAL_PRE_IR_PENDING_STATUSES:
                            _set_plain_reading_gate_pending_from_exception(
                                state,
                                exc=exc,
                                checkpoint_dir=checkpoint_dir,
                                progress=progress,
                            )
                            continue
                        _record_pre_ir_failure(
                            state,
                            exc,
                            checkpoint_dir=checkpoint_dir,
                            progress=progress,
                        )
                        continue

        if segmentation is None or reading is None:
            if progress is not None:
                progress.info(f"{state.requirement_id} pre-IR review gates")
            try:
                artifacts = run_pre_ir_review(
                    requirement_id=state.requirement_id,
                    requirement=state.requirement,
                    retriever=state.retriever,
                    llm_client=state.nlu_agent.llm_client,
                    judge_client=judge_client,
                    config=config,
                    interactive=requires_interactive_pre_ir_review(
                        config.segmentation_review,
                        config.plain_reading_review,
                    ),
                    guided_context=state.guided_context,
                    artifact_callback=(
                        (
                            lambda current_segmentation, current_review: (
                                save_plain_reading_generated_checkpoint(
                                    checkpoint_dir,
                                    requirement_id=state.requirement_id,
                                    plain_reading_review=current_review,
                                    segmentation_schema_version=(
                                        current_segmentation.schema_version
                                    ),
                                )
                            )
                        )
                        if checkpoint_dir
                        else None
                    ),
                    segmentation_generation_diagnostic_callback=(
                        (
                            lambda diagnostic: save_segmentation_generation_diagnostic(
                                checkpoint_dir,
                                requirement_id=state.requirement_id,
                                diagnostic=diagnostic,
                            )
                        )
                        if checkpoint_dir
                        else None
                    ),
                    generation_diagnostic_callback=(
                        (
                            lambda diagnostic: save_plain_reading_generation_diagnostic(
                                checkpoint_dir,
                                requirement_id=state.requirement_id,
                                diagnostic=diagnostic,
                            )
                        )
                        if checkpoint_dir
                        else None
                    ),
                    semantic_override_callback=(
                        (
                            lambda override: save_human_semantic_override_checkpoint(
                                checkpoint_dir,
                                override=override,
                            )
                        )
                        if checkpoint_dir
                        else None
                    ),
                    ap_artifact_callback=(
                        (
                            lambda artifact: save_ap_identity_artifact_checkpoint(
                                checkpoint_dir,
                                artifact,
                            )
                        )
                        if checkpoint_dir
                        else None
                    ),
                    ap_representation_profile=state.ap_representation_profile,
                    declared_ap_catalog=state.declared_ap_catalog,
                    current_ap_inventory=current_ap_inventory,
                    ap_candidates_available=bool(state.ap_binding_context and (
                        state.ap_binding_context.has_authoritative_catalog
                        or state.ap_binding_context.grounding_contract.identities)),
                    original_source_text=state.original_requirement,
                    active_semantic_override=state.human_semantic_override,
                )
            except Exception as exc:
                if getattr(exc, "status", "") in _NORMAL_PRE_IR_PENDING_STATUSES:
                    _set_plain_reading_gate_pending_from_exception(
                        state,
                        exc=exc,
                        checkpoint_dir=checkpoint_dir,
                        progress=progress,
                    )
                    continue
                _record_pre_ir_failure(
                    state,
                    exc,
                    checkpoint_dir=checkpoint_dir,
                    progress=progress,
                )
                continue
            segmentation = artifacts.surface_segmentation
            reading = artifacts.plain_reading_review
            state.requirement = artifacts.effective_source_text
            state.human_semantic_override = artifacts.human_semantic_override
            _update_state_effective_source(state)
            state.active_ap_inventory = artifacts.effective_ap_inventory
            state.review_records.extend(artifacts.guided_review_records)
            save_segmentation_generated_checkpoint(
                checkpoint_dir,
                requirement_id=state.requirement_id,
                segmentation=segmentation,
            )
            save_segmentation_reviewed_checkpoint(
                checkpoint_dir,
                requirement_id=state.requirement_id,
                segmentation=segmentation,
            )
            save_plain_reading_generated_checkpoint(
                checkpoint_dir,
                requirement_id=state.requirement_id,
                plain_reading_review=reading,
                segmentation_schema_version=segmentation.schema_version,
            )
            save_plain_reading_reviewed_checkpoint(
                checkpoint_dir,
                requirement_id=state.requirement_id,
                plain_reading_review=reading,
                segmentation_schema_version=segmentation.schema_version,
            )
        state.surface_segmentation = segmentation
        state.plain_reading_review = reading
        if segmentation is not None and reading is not None:
            readiness = assess_plain_reading_ir_readiness(reading, segmentation)
            if readiness.disposition == "invalid":
                _record_pre_ir_failure(
                    state,
                    PreIRGateError(
                        "；".join(readiness.validation_errors),
                        stage="plain_reading",
                        status="blocked_error",
                    ),
                    checkpoint_dir=checkpoint_dir,
                    progress=progress,
                )
                continue
            if not readiness.can_enter_semantic_ir and not readiness.can_attempt_candidate_ir:
                _set_plain_reading_resolution_pending(
                    state,
                    readiness=readiness,
                    review=reading,
                    checkpoint_dir=checkpoint_dir,
                    progress=progress,
                )
                continue
            if checkpoint_dir:
                save_plain_reading_reviewed_checkpoint(
                    checkpoint_dir,
                    requirement_id=state.requirement_id,
                    plain_reading_review=reading,
                    segmentation_schema_version=segmentation.schema_version,
                )
        if checkpoint_dir and state.active_ap_inventory is None:
            state.active_ap_inventory = load_ap_inventory_checkpoint(
                checkpoint_dir,
                state.requirement_id,
                current=True,
                expected_source_hash=source_hash(state.requirement),
                expected_profile_id=(
                    state.ap_representation_profile.profile_id
                    if state.ap_representation_profile is not None
                    else None
                ),
            )
        state.ap_binding_context = build_ap_binding_context(
            requirement_id=state.requirement_id,
            source_text=state.requirement,
            representation_profile=state.ap_representation_profile,
            declared_catalog=state.declared_ap_catalog,
            active_inventory=state.active_ap_inventory,
            identity_mapping_id=(
                state.human_semantic_override.ap_comparison_id
                if state.human_semantic_override is not None
                else ""
            ),
            provenance=(
                "batch_effective_source",
                (
                    "human_semantic_replacement"
                    if state.human_semantic_override is not None
                    else "requirement_source"
                ),
            ),
        )
        if state.semantic is not None:
            state.semantic = replace(
                state.semantic,
                surface_segmentation=segmentation,
                plain_reading_review=reading,
            )
        pre_ir_records = [
                {
                    "stage": "segmentation",
                    "requirement_id": state.requirement_id,
                    "status": segmentation.status,
                    "segmentation_id": segmentation.segmentation_id,
                    "source_hash": segmentation.source_hash,
                    "mechanical_validation": asdict(
                        segmentation.mechanical_validation
                    ),
                    "review_basis": segmentation.review_basis,
                    "label_source": segmentation.label_source,
                },
                {
                    "stage": "plain_reading",
                    "requirement_id": state.requirement_id,
                    "status": reading.overall_status,
                    "reading_review_id": reading.reading_review_id,
                    "segmentation_id": reading.segmentation_id,
                    "decision_fingerprint": decision_fingerprint(reading),
                    "review_basis": reading.review_basis,
                    "reviewer_source": reading.reviewer_source,
                    "ambiguity_debt": list(reading.unresolved_ambiguity_ids),
                },
            ]
        if (
            config.segmentation_review != "rules"
            or config.plain_reading_review != "off"
        ):
            state.review_records.extend(pre_ir_records)


def _set_plain_reading_resolution_pending(
    state: _BatchItemState,
    *,
    readiness: object,
    review: PlainReadingReview,
    checkpoint_dir: str | None,
    progress: TerminalProgress | None,
) -> None:
    unresolved_ids = tuple(
        str(value)
        for value in getattr(readiness, "unresolved_ambiguity_ids", ())
    )
    unresolved = set(unresolved_ids)
    need_view = None
    if (
        review.overall_status == "ambiguous_unresolved"
        and state.surface_segmentation is not None
    ):
        need_view = build_plain_reading_need_view(
            review,
            state.surface_segmentation,
        )
        save_plain_reading_need_view_checkpoint(
            checkpoint_dir,
            need_view=need_view,
        )
    payload = {
        "stage": "plain_reading",
        "status": "awaiting_plain_reading_resolution",
        "classification": "needs_review",
        "reason_code": str(getattr(readiness, "reason_code", "")),
        "disposition": str(getattr(readiness, "disposition", "")),
        "requirement_id": state.requirement_id,
        "reading_review_id": review.reading_review_id,
        "decision_fingerprint": str(
            getattr(readiness, "decision_fingerprint", "")
        ),
        "unresolved_ambiguity_ids": list(unresolved_ids),
        "unresolved_axes": [
            {
                "ambiguity_id": ambiguity.ambiguity_id,
                "question": ambiguity.question,
            }
            for ambiguity in review.ambiguities
            if ambiguity.ambiguity_id in unresolved
        ],
        "pending_revision_request_ids": list(
            getattr(readiness, "pending_revision_request_ids", ())
        ),
        "suggested_rerun_stage": "plain_reading",
        "review_completed": bool(need_view),
        "single_ir_ready": False,
        "finalization_allowed": False,
        "next_stage": "plain_reading_resolution",
        **(
            {
                "plain_reading_need_view_id": need_view.need_view_id,
                "plain_reading_need_view": asdict(need_view),
            }
            if need_view is not None
            else {}
        ),
    }
    state.plain_reading_resolution_pending = payload
    state.review_records.append(dict(payload))
    if progress is not None:
        progress.warn(
            f"{state.requirement_id} awaits Plain Reading resolution; "
            "Semantic IR and NLU were skipped"
        )


def _set_plain_reading_gate_pending_from_exception(
    state: _BatchItemState,
    *,
    exc: Exception,
    checkpoint_dir: str | None,
    progress: TerminalProgress | None,
) -> None:
    """Route an expected human pause without a failed checkpoint or traceback."""

    status = str(getattr(exc, "status", ""))
    snapshot = _load_plain_reading_failure_snapshot(
        checkpoint_dir, state.requirement_id
    )
    reason_code = (
        "PLAIN_READING_SOURCE_CLARIFICATION_REQUIRED"
        if status == "needs_source_clarification"
        else "PLAIN_READING_SEMANTIC_RESOLUTION_REQUIRED"
        if status == "ambiguous_unresolved"
        else "PLAIN_READING_HUMAN_ACTION_REQUIRED"
    )
    payload = {
        "stage": "plain_reading",
        "status": status or "awaiting_plain_reading_resolution",
        "classification": "needs_review",
        "reason_code": reason_code,
        "requirement_id": state.requirement_id,
        "suggested_rerun_stage": "plain_reading",
        "message": str(exc),
        **(dict(snapshot) if isinstance(snapshot, Mapping) else {}),
    }
    state.plain_reading_resolution_pending = payload
    state.review_records.append(dict(payload))
    if progress is not None:
        progress.warn(
            f"{state.requirement_id} awaits Plain Reading action; later requirements continue"
        )


def _pre_ir_failure_payload(
    exc: Exception,
    *,
    plain_reading_state: dict | None = None,
) -> dict:
    stage = str(getattr(exc, "stage", "pre_ir")) or "pre_ir"
    status = str(getattr(exc, "status", "blocked_error")) or "blocked_error"
    suggested_stage = (
        "segmentation"
        if stage in {"segmentation_generation", "surface_review"}
        else "plain_reading"
        if stage in {
            "plain_reading_generation",
            "plain_reading",
            "plain_reading_revision",
        }
        else "pipeline"
    )
    cause = exc.__cause__
    validation = getattr(cause, "validation", None)
    mechanical_validation = (
        asdict(validation) if validation is not None else None
    )
    payload = {
        "stage": stage,
        "status": status,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "suggested_rerun_stage": suggested_stage,
        "mechanical_validation": mechanical_validation,
        "invalid_completions": [],
        "recent_candidates": [],
        "completed_artifacts": [],
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-12000:],
    }
    diagnostics = getattr(exc, "diagnostics", None)
    if isinstance(diagnostics, Mapping) and diagnostics:
        payload["generation_diagnostics"] = dict(diagnostics)
    if plain_reading_state is not None:
        payload["plain_reading_state"] = plain_reading_state
    return payload


def _record_pre_ir_failure(
    state: _BatchItemState,
    exc: Exception,
    *,
    checkpoint_dir: str | None,
    progress: TerminalProgress | None,
) -> None:
    snapshot = _load_plain_reading_failure_snapshot(
        checkpoint_dir,
        state.requirement_id,
    )
    state.pre_ir_error = _pre_ir_failure_payload(
        exc,
        plain_reading_state=snapshot,
    )
    state.review_records.append(
        {
            "stage": "pre_ir",
            "requirement_id": state.requirement_id,
            "status": "blocked_error",
            **state.pre_ir_error,
        }
    )
    save_failed_checkpoint(
        checkpoint_dir,
        requirement_id=state.requirement_id,
        stage=str(state.pre_ir_error["stage"]),
        error=exc,
        suggested_rerun_stage=str(
            state.pre_ir_error["suggested_rerun_stage"]
        ),
        details=state.pre_ir_error,
    )
    if progress is not None:
        progress.error(
            f"{state.requirement_id} pre-IR blocked: "
            f"{type(exc).__name__}: {exc}"
        )


def _load_plain_reading_failure_snapshot(
    checkpoint_dir: str | None,
    requirement_id: str,
) -> dict | None:
    try:
        payload = load_plain_reading_generated_checkpoint(
            checkpoint_dir,
            requirement_id,
        )
    except Exception as exc:
        return {
            "snapshot_error_type": type(exc).__name__,
            "snapshot_error": str(exc),
        }
    mapping = payload.get("plain_reading_review") if payload else None
    if not isinstance(mapping, dict):
        return None
    ambiguities = [
        {
            "ambiguity_id": str(item.get("ambiguity_id", "")),
            "decision_status": str(item.get("decision_status", "")),
        }
        for item in mapping.get("ambiguities", ())
        if isinstance(item, dict)
    ]
    requests = [
        {
            "request_id": str(item.get("request_id", "")),
            "ambiguity_id": str(item.get("ambiguity_id", "")),
            "classification": str(item.get("classification", "")),
            "validation_status": str(item.get("validation_status", "")),
            "request_status": str(item.get("request_status", "")),
        }
        for item in mapping.get("revision_requests", ())
        if isinstance(item, dict)
    ]
    return {
        "reading_review_id": str(mapping.get("reading_review_id", "")),
        "overall_status": str(mapping.get("overall_status", "")),
        "review_basis": str(mapping.get("review_basis", "")),
        "ambiguities": ambiguities,
        "revision_requests": requests,
        "semantic_fallback": next(
            (
                dict(item)
                for item in reversed(mapping.get("review_history", ()))
                if isinstance(item, dict)
                and item.get("operation")
                == "plain_reading_semantic_fallback"
            ),
            None,
        ),
    }


def _restore_batch_checkpoints(
    *,
    states: list[_BatchItemState],
    checkpoint_dir: str,
    force_rerun_stages: dict[str, str],
    progress: TerminalProgress | None,
) -> None:
    for state in states:
        if (
            state.pre_ir_error is not None
            or state.plain_reading_resolution_pending is not None
        ):
            continue
        forced_stage = force_rerun_stages.get(state.requirement_id, "")
        if forced_stage == "semantic":
            continue
        if not forced_stage:
            review_payload = load_needs_review_checkpoint(
                checkpoint_dir,
                state.requirement_id,
            )
            review_details = (
                review_payload.get("details", {})
                if isinstance(review_payload, Mapping)
                else {}
            )
            expected_context = (
                state.ap_binding_context.context_fingerprint
                if state.ap_binding_context is not None
                else ""
            )
            stored_context = (
                str(review_details.get("ap_context_fingerprint", ""))
                if isinstance(review_details, Mapping)
                else ""
            )
            if (
                isinstance(review_payload, Mapping)
                and expected_context
                and stored_context == expected_context
                and review_payload.get("reason_code") not in {
                    "AP_GRAPH_BINDING_REVIEW_REQUIRED", "SOURCE_GRAPH_SEMANTIC_RESOLUTION_REQUIRED",
                    "IR_SEMANTIC_AXIS_REQUIRED", "IR_GENERATION_SCHEMA_INVALID",
                    "UNSUPPORTED_SEMANTICS", "SOURCE_GRAPH_CANDIDATE_REQUIRED",
                }
            ):
                state.needs_review_error = {
                    "stage": str(review_payload.get("stage", "semantic")),
                    "status": "needs_review",
                    "reason_code": str(
                        review_payload.get("reason_code", "NEEDS_REVIEW")
                    ),
                    "error": str(review_payload.get("message", "")),
                    "suggested_rerun_stage": str(
                        review_payload.get("suggested_rerun_stage", "semantic")
                    ),
                    **dict(review_details),
                }
                if progress is not None:
                    progress.info(
                        f"{state.requirement_id} needs_review state loaded from checkpoint"
                    )
                continue
        policy_candidate_payload = load_semantic_policy_candidates_checkpoint(
            checkpoint_dir,
            state.requirement_id,
        )
        if policy_candidate_payload is not None:
            try:
                restored_policy_semantic = semantic_from_checkpoint(
                    policy_candidate_payload
                )
            except (TypeError, ValueError):
                restored_policy_semantic = None
            policy_parents_match = bool(
                restored_policy_semantic is not None
                and restored_policy_semantic.faithful_ir is not None
                and restored_policy_semantic.faithful_ir.source_text == state.requirement
                and (state.ap_binding_context is None or
                     restored_policy_semantic.faithful_ir.ap_context_fingerprint == state.ap_binding_context.context_fingerprint)
                and (
                    state.surface_segmentation is None
                    or restored_policy_semantic.surface_segmentation is not None
                    and restored_policy_semantic.surface_segmentation.segmentation_id
                    == state.surface_segmentation.segmentation_id
                )
                and (
                    state.plain_reading_review is None
                    or restored_policy_semantic.plain_reading_review is not None
                    and restored_policy_semantic.plain_reading_review.reading_review_id
                    == state.plain_reading_review.reading_review_id
                    and decision_fingerprint(
                        restored_policy_semantic.plain_reading_review
                    )
                    == decision_fingerprint(state.plain_reading_review)
                )
            )
            if (
                policy_parents_match
                and is_policy_candidate_semantic(restored_policy_semantic)
            ):
                state.semantic = restored_policy_semantic
                state.regenerate_semantic = False
                state.regenerate_traces = True
                from formula_generation_fallback import is_formula_generation_fallback
                if is_formula_generation_fallback(state.semantic):
                    state.regenerate_traces = False
                    state.semantic_policy_pending = {"stage": "formula_generation_fallback", "status": "candidate_generated",
                        "reason_code": "GRAPH_GENERATION_FALLBACK_SELECTED"}
                    state.review_records.extend(policy_candidate_payload.get("review_records", []))
                    continue
                if progress is not None:
                    progress.info(
                        f"{state.requirement_id} policy candidates loaded; "
                        "expert trace review will resume"
                    )
                continue
        reviewed_payload = load_reviewed_item_checkpoint(
            checkpoint_dir,
            state.requirement_id,
            expected_source_hash=(
                state.surface_segmentation.source_hash
                if state.surface_segmentation is not None
                else source_hash(state.requirement)
            ),
            expected_segmentation_id=(
                state.surface_segmentation.segmentation_id
                if state.surface_segmentation is not None
                else None
            ),
            expected_reading_review_id=(
                state.plain_reading_review.reading_review_id
                if state.plain_reading_review is not None
                else None
            ),
            expected_plain_reading_decision_fingerprint=(
                decision_fingerprint(state.plain_reading_review)
                if state.plain_reading_review is not None
                else None
            ),
            expected_ap_context_fingerprint=(
                state.ap_binding_context.context_fingerprint
                if state.ap_binding_context is not None
                else None
            ),
        )
        if reviewed_payload is not None and _restore_state_from_checkpoint(
            state,
            reviewed_payload,
            approved=forced_stage != "trace",
            artifact_root=checkpoint_dir,
        ):
            if forced_stage == "trace":
                state.traces = []
                state.trace_evidence_records = []
                state.trace_evidence_disputes = []
                state.regenerate_traces = True
                state.semantic_approved = True
                state.trace_approved = False
                if progress is not None:
                    progress.info(
                        f"{state.requirement_id} semantic loaded; traces will rerun"
                    )
            elif progress is not None:
                progress.ok(
                    f"{state.requirement_id} semantic/traces loaded from checkpoint"
                )
            continue

        generated_payload = load_generated_review_checkpoint(
            checkpoint_dir,
            state.requirement_id,
            expected_source_hash=(
                state.surface_segmentation.source_hash
                if state.surface_segmentation is not None
                else source_hash(state.requirement)
            ),
            expected_segmentation_id=(
                state.surface_segmentation.segmentation_id
                if state.surface_segmentation is not None
                else None
            ),
            expected_reading_review_id=(
                state.plain_reading_review.reading_review_id
                if state.plain_reading_review is not None
                else None
            ),
            expected_plain_reading_decision_fingerprint=(
                decision_fingerprint(state.plain_reading_review)
                if state.plain_reading_review is not None
                else None
            ),
            expected_ap_context_fingerprint=(
                state.ap_binding_context.context_fingerprint
                if state.ap_binding_context is not None
                else None
            ),
        )
        if generated_payload is not None and _restore_state_from_checkpoint(
            state,
            generated_payload,
            approved=False,
            artifact_root=checkpoint_dir,
        ):
            if forced_stage == "trace":
                state.traces = []
                state.trace_evidence_records = []
                state.trace_evidence_disputes = []
                state.regenerate_traces = True
            if progress is not None:
                progress.info(f"{state.requirement_id} generated traces loaded from checkpoint")
            continue

def _restore_state_from_checkpoint(
    state: _BatchItemState,
    payload: dict,
    *,
    approved: bool,
    artifact_root: str,
) -> bool:
    if payload.get("requirement") != state.requirement:
        return False
    semantic = semantic_from_checkpoint(payload["semantic"])
    traces = [
        trace_from_checkpoint(trace_payload)
        for trace_payload in payload.get("traces", [])
    ]

    records = [
        trace_evidence_record_from_checkpoint(item)
        for item in payload.get("trace_evidence_records", ())
        if isinstance(item, Mapping)
    ]
    disputes = [
        trace_evidence_dispute_from_checkpoint(item)
        for item in payload.get("trace_evidence_disputes", ())
        if isinstance(item, Mapping)
    ]
    brief = build_trace_adjudication_brief(state.requirement, semantic)
    reconciled = reconcile_trace_evidence(
        semantic=semantic,
        brief=brief,
        existing_records=records,
        existing_disputes=disputes,
        existing_traces=traces,
    )
    traces = list(reconciled.active_traces)
    has_evidence_disputes = bool(reconciled.unresolved_disputes)

    if approved and not has_evidence_disputes and _semantic_trace_consistency_errors(
        requirement_id=state.requirement_id,
        semantic=semantic,
        traces=traces,
    ):
        return False

    authoritative_session: AuthoritativeTraceSession | None = None
    if semantic.faithful_ir is not None:
        promotion = promote_semantic_candidate(
            semantic,
            artifact_root=artifact_root,
        )
        if promotion.disposition != TRACE_READY:
            return False
        authoritative_session = create_authoritative_trace_session(promotion)

    state.semantic = semantic
    state.authoritative_trace_session = authoritative_session
    state.traces = traces
    state.trace_evidence_records = list(reconciled.records)
    state.trace_evidence_disputes = list(reconciled.disputes)
    state.trace_coverage = trace_coverage_from_checkpoint(
        payload.get("trace_coverage")
    )
    if not payload.get("trace_coverage"):
        state.trace_coverage = _assess_state_trace_coverage(
            state,
            traces,
            requirement=state.requirement,
            feedback_history=state.trace_feedback_history,
            guided=bool(state.guided_context or state.expected_traces),
            allow_degraded=approved,
        )
    if has_evidence_disputes:
        state.trace_coverage = replace(
            state.trace_coverage,
            status="incomplete",
            actual_count=len(traces),
            coverage_debt=tuple(
                dict.fromkeys(
                    state.trace_coverage.coverage_debt
                    + reconciled.coverage_debt
                )
            ),
            executable_semantics=False,
            semantic_complete=False,
            quota_complete=False,
            reason=(
                "Stored trace labels were quarantined because they conflict "
                "with the current reviewed meaning; background resolution is pending."
            ),
        )
        approved = False
    if approved and state.trace_coverage.status in {"incomplete", "rejected"}:
        return False
    if approved and state.trace_coverage.degraded:
        semantic = replace(semantic, persistent_rag_allowed=False)
        state.semantic = semantic
    from human_review_session import REVIEW_STOP_DEBTS
    stop_debts = REVIEW_STOP_DEBTS.intersection(state.trace_coverage.coverage_debt)
    budget_stopped = bool(stop_debts)
    state.review_records = list(payload.get("review_records", [])) if approved or budget_stopped else []
    state.semantic_approved = approved
    state.trace_approved = approved
    state.regenerate_semantic = False
    state.regenerate_traces = has_evidence_disputes
    state.curated_semantic = False
    state.curated_traces = []
    state.curated_trace_source = ""
    if budget_stopped:
        state.semantic_approved = state.trace_approved = False
        state.regenerate_semantic = state.regenerate_traces = False
        state.semantic_policy_pending = {"stage": "human_review_budget", "status": "candidate_fallback",
                                         "reason_code": sorted(stop_debts)[0]}
    if (
        approved
        and state.semantic is not None
        and not state.trace_evidence_policy.experimental_only
    ):
        record_approved_review_examples(
            retriever=state.retriever,
            requirement_id=state.requirement_id,
            requirement=state.requirement,
            semantic=state.semantic,
            traces=state.traces,
            round_index=0,
            comments="从 checkpoint 恢复的已审核语义结构和有限轨迹。",
        )
    return True


def _regenerate_batch_item(
    state: _BatchItemState,
    *,
    checkpoint_dir: str | None,
    progress: TerminalProgress | None = None,
    round_index: int,
    item_index: int,
    total_items: int,
    review_mode: str,
    records_dir: str,
    allow_degraded: bool,
    judge_client: JSONLLMClient | None = None,
) -> None:
    item_label = f"{state.requirement_id} {item_index}/{total_items}"
    if state.regenerate_semantic or state.semantic is None:
        message = f"{item_label} semantic structure round {round_index}"
        round_1_semantic_artifact: object | None = None
        round_1_semantic_errors: list[str] = []
        semantic_context = (
            state.guided_context.attempt_view("semantic_extraction", 1)
            if state.guided_context is not None
            else None
        )
        try:
            if state.nlu_agent.llm_enabled:
                from ap_preparation import prepare_reviewed_ap_context
                source_ap_context = state.ap_binding_context or text_only_ap_binding_context(
                    state.requirement, requirement_id=state.requirement_id,
                )
                source_ap_context.validate_source(state.requirement)
                prepared = prepare_reviewed_ap_context(
                    source_ap_context,
                    state.nlu_agent.llm_client, segmentation=state.surface_segmentation,
                    interactive=review_mode == "manual", records_dir=records_dir,
                    plain_reading_review=state.plain_reading_review,
                )
                state.ap_binding_context = prepared.context
                if prepared.inventory is not None and prepared.inventory.status == "approved":
                    state.active_ap_inventory = prepared.inventory
                    save_ap_inventory_checkpoint(checkpoint_dir, inventory=prepared.inventory, make_current=True)
            if progress is None:
                state.semantic = state.nlu_agent.generate(
                    state.requirement,
                    feedback_history=state.semantic_feedback_history,
                    guided_context=semantic_context,
                    surface_segmentation=state.surface_segmentation,
                    plain_reading_review=state.plain_reading_review,
                    ap_binding_context=state.ap_binding_context,
                    effective_semantic_source=state.effective_semantic_source,
                )
            else:
                with progress.task(message):
                    state.semantic = state.nlu_agent.generate(
                        state.requirement,
                        feedback_history=state.semantic_feedback_history,
                        guided_context=semantic_context,
                        surface_segmentation=state.surface_segmentation,
                        plain_reading_review=state.plain_reading_review,
                        ap_binding_context=state.ap_binding_context,
                        effective_semantic_source=state.effective_semantic_source,
                    )
                progress.ok(f"{state.requirement_id} semantic generated")
            if state.nlu_agent.llm_enabled and state.expected_semantic is None:
                from nonexpert_review import review_semantic_with_model
                from semantic_review_view import semantic_review_fingerprint
                checked_meanings = set()
                for attempt in range(1, 4):
                    meaning = semantic_review_fingerprint(state.semantic)
                    if meaning is not None and meaning in checked_meanings:
                        record = {"stage": "semantic_revision", "round": attempt,
                            "status": "no_progress_candidate_retained", "approved": False,
                            "reviewer_source": "model_assistant", "reason_code": "SEMANTIC_REVISION_NO_PROGRESS",
                            "comments": "Regeneration repeated a rejected meaning; retain the unapproved candidate."}
                        record["record_path"] = str(_save_json_record(record, records_dir, record["stage"]))
                        state.review_records.append(record)
                        state.semantic = replace(state.semantic, semantic_status="needs_semantic_review",
                                                 persistent_rag_allowed=False)
                        break
                    if meaning is not None:
                        checked_meanings.add(meaning)
                    checked = review_semantic_with_model(
                        state.requirement, state.semantic, state.nlu_agent.llm_client,
                    )
                    state.review_records.append({
                        "stage": "semantic", "round": attempt,
                        "reviewer_source": "model_assistant", **checked,
                    })
                    if checked["approved"]:
                        break
                    if attempt == 3:
                        state.semantic = replace(state.semantic, semantic_status="needs_semantic_review")
                        break
                    state.semantic_feedback_history.append(checked["reason"])
                    state.semantic = state.nlu_agent.generate(
                        state.requirement,
                        feedback_history=state.semantic_feedback_history,
                        surface_segmentation=state.surface_segmentation,
                        plain_reading_review=state.plain_reading_review,
                        ap_binding_context=state.ap_binding_context,
                        effective_semantic_source=state.effective_semantic_source,
                    )
        except Exception as exc:
            if state.expected_semantic is None:
                if not isinstance(exc, (FaithfulIRValidationError, APPreparationError)):
                    raise
                from candidate_delivery import recover_generation_candidate
                state.semantic, record = recover_generation_candidate(
                    state.requirement, state.semantic, exc,
                    surface_segmentation=state.surface_segmentation,
                    plain_reading_review=state.plain_reading_review,
                    ap_binding_context=state.ap_binding_context,
                    client=state.nlu_agent.llm_client,
                )
                record["round"] = round_index
                record["requirement_id"] = state.requirement_id
                record["record_path"] = str(_save_json_record(record, records_dir, record["stage"]))
                state.review_records.append(record)
            else:
                state.semantic = None
                round_1_semantic_artifact = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                round_1_semantic_errors = [
                    f"round 1 semantic generation raised {type(exc).__name__}: {exc}"
                ]
        state.regenerate_semantic = False
        state.regenerate_traces = not bool(state.curated_traces)
        state.semantic_approved = False
        state.trace_approved = False
        if state.semantic is not None and state.experiment_config is not None:
            state.semantic = replace(
                state.semantic,
                data_split=state.experiment_config.data_split,
                label_source=("model_assistant" if state.nlu_agent.llm_enabled and state.expected_semantic is None
                              else _label_source_for_config(state.experiment_config)),
                formula_source="none",
                formula_visibility=(),
                persistent_rag_allowed=_persistent_rag_allowed_for_config(
                    state.semantic,
                    state.experiment_config,
                ),
            )
        from formula_generation_fallback import is_formula_generation_fallback
        if is_formula_generation_fallback(state.semantic):
            state.semantic_policy_pending = {"stage": "formula_generation_fallback",
                "status": "candidate_generated", "reason_code": "GRAPH_GENERATION_FALLBACK_SELECTED"}
            state.semantic = _bind_or_validate_state_semantic_source(state)
            save_semantic_policy_candidates_checkpoint(checkpoint_dir,
                requirement_id=state.requirement_id, requirement=state.requirement,
                semantic=state.semantic, queries=[], review_records=state.review_records)
            return
        if state.expected_semantic is not None:
            if round_1_semantic_artifact is None and state.semantic is not None:
                round_1_semantic_artifact = asdict(state.semantic)
                round_1_semantic_errors = _expected_warmup_semantic_errors(state)
            _resolve_expected_semantic_artifact(
                state=state,
                round_index=round_index,
                records_dir=records_dir,
                progress=progress,
                round_1_artifact=round_1_semantic_artifact,
                round_1_errors=round_1_semantic_errors,
            )

    elif state.curated_semantic and state.semantic is not None and progress is not None:
        progress.ok(f"{state.requirement_id} semantic loaded")

    if state.semantic is not None and pattern_requires_human_learning(state.semantic):
        if progress is not None:
            progress.warn(
                f"{state.requirement_id} unknown pattern: {state.semantic.pattern}"
            )
        pattern_result = run_pattern_learning_review(
            requirement_id=state.requirement_id,
            requirement=state.requirement,
            semantic=state.semantic,
            retriever=state.retriever,
            review_mode=review_mode,
            round_index=round_index,
            records_dir=records_dir,
            progress=progress,
        )
        if pattern_result.record:
            state.review_records.append(pattern_result.record)
        if not pattern_result.approved and not pattern_result.provisional:
            state.semantic_feedback_history.append(pattern_result.comments)
            state.semantic_approved = False
            state.trace_approved = False
            state.regenerate_semantic = True
            state.regenerate_traces = True
            return

        if pattern_result.semantic is None:
            raise RuntimeError(
                f"{state.requirement_id} pattern 审核通过但缺少语义结构。"
            )
        state.semantic = pattern_result.semantic
        state.traces = list(pattern_result.traces)
        # Coverage for a native Faithful IR is authoritative only after the
        # candidate has been persisted and a trace session has been created by
        # the promotion path.  Leave this provisional result unset here.
        state.trace_coverage = None
        state.curated_traces = list(pattern_result.traces)
        state.curated_trace_source = (
            "未知模式 provisional 证据，未生成单独 pattern 审核文件"
        )
        state.regenerate_semantic = False
        state.regenerate_traces = not bool(state.curated_traces)
        state.semantic_approved = False
        state.trace_approved = False

    if state.semantic is not None and state.semantic.faithful_ir is not None:
        state.semantic = attach_candidate_ir_set(state.semantic, state.requirement, max_branch_count=4)
    if requires_ir_candidate_adjudication(state.semantic):
        _adjudicate_ir_candidate_semantic(
            state,
            checkpoint_dir=checkpoint_dir,
            records_dir=records_dir,
            review_mode=review_mode,
            round_index=round_index,
            progress=progress,
            judge_client=judge_client,
        )
        if state.semantic_policy_pending is not None:
            return

    if state.semantic is not None and state.semantic.faithful_ir is not None:
        state.semantic = _bind_or_validate_state_semantic_source(state)
        if audit_semantic_readiness(
            state.semantic.faithful_ir
        ).has_capability(CAN_BIND_RESOLUTION):
            state.semantic = attach_semantic_resolution_record(state.semantic)
        # Candidate branches are part of the semantic artifact protected by
        # the authoritative trace session.  Attach them before promotion only
        # when the semantic is actually trace-ready; unresolved semantics must
        # stay on the semantic-review route.
        if audit_semantic_readiness(
            state.semantic.faithful_ir
        ).has_capability(CAN_GENERATE_AUTHORITATIVE_TRACES):
            state.semantic = attach_candidate_ir_set(
                state.semantic,
                state.requirement,
            )
        promotion = promote_semantic_candidate(
            state.semantic,
            artifact_root=(
                checkpoint_dir
                or str(Path(records_dir) / "semantic_artifacts")
            ),
        )
        state.review_records.append(
            {
                "stage": "semantic_promotion",
                "requirement_id": state.requirement_id,
                "status": promotion.disposition,
                "reason_code": promotion.reason_code,
                "candidate_path": promotion.candidate_path,
                "trace_ready_path": promotion.trace_ready_path,
            }
        )
        if promotion.disposition != TRACE_READY:
            state.traces = []
            state.semantic_approved = False
            state.trace_approved = False
            state.regenerate_traces = False
            state.needs_review_error = {
                "stage": "semantic_promotion",
                "status": "needs_review",
                "reason_code": promotion.reason_code,
                "error_type": "SemanticPromotionPending",
                "error": promotion.message,
                "suggested_rerun_stage": "semantic",
                "candidate_path": promotion.candidate_path,
                "promotion_fingerprint": promotion.promotion_fingerprint,
            }
            state.trace_coverage = TraceCoverageReport(
                status="incomplete",
                coverage_debt=(promotion.reason_code,),
                executable_semantics=False,
                semantic_complete=False,
                quota_complete=False,
                reason=promotion.message,
            )
            return
        state.authoritative_trace_session = (
            create_authoritative_trace_session(promotion)
        )

    trace_readiness_errors = trace_generation_readiness_errors(
        state.semantic.faithful_ir if state.semantic is not None else None
    )
    if trace_readiness_errors:
        state.traces = []
        state.semantic_approved = False
        state.trace_approved = False
        state.regenerate_traces = True
        state.trace_coverage = TraceCoverageReport(
            status="incomplete",
            coverage_debt=tuple(
                f"semantic_trace_readiness:{error}"
                for error in trace_readiness_errors
            ),
            executable_semantics=False,
            semantic_complete=False,
            quota_complete=False,
            reason=(
                "Semantic IR requires parameter review before finite-trace "
                "construction: " + "; ".join(trace_readiness_errors)
            ),
        )
        if progress is not None:
            progress.warn(
                f"{state.requirement_id} semantic review required before traces: "
                + "; ".join(trace_readiness_errors)
            )
        return

    state.traces = filter_traces_for_policy(
        state.traces,
        state.trace_evidence_policy,
    )
    state.curated_traces = filter_traces_for_policy(
        state.curated_traces,
        state.trace_evidence_policy,
    )
    canonical_trace_plan = (
        compile_executable_trace_plan(state.semantic.faithful_ir)
        if state.semantic.faithful_ir is not None
        else None
    )
    if (
        canonical_trace_plan is not None
        and canonical_trace_plan.feasibility_classification
        != "contingent_in_declared_domain"
    ):
        report = canonical_trace_plan.relation_feasibility
        state.traces = []
        state.semantic_approved = False
        state.trace_approved = False
        state.regenerate_traces = False
        state.needs_review_error = {
            "stage": "relation_feasibility",
            "status": "needs_review",
            "reason_code": "IR_RELATION_NON_CONTINGENT_REVIEW_REQUIRED",
            "error_type": "IRRelationFeasibilityReview",
            "error": (
                "Root IR is non-contingent in the declared finite domain; "
                "no impossible opposite witness was requested."
            ),
            "suggested_rerun_stage": "semantic",
            "relation_feasibility_classification": (
                canonical_trace_plan.feasibility_classification
            ),
            "feasibility_report_hash": (
                canonical_trace_plan.feasibility_report_hash
            ),
            "ap_context_fingerprint": (
                canonical_trace_plan.ap_context_fingerprint
            ),
            "source_supports_non_contingent": (
                report.source_supports_non_contingent
                if report is not None
                else False
            ),
        }
        state.review_records.append(
            {
                "requirement_id": state.requirement_id,
                **state.needs_review_error,
            }
        )
        save_generated_semantic_checkpoint(
            checkpoint_dir,
            requirement_id=state.requirement_id,
            requirement=state.requirement,
            semantic=state.semantic,
            surface_segmentation=state.surface_segmentation,
            plain_reading_review=state.plain_reading_review,
            generation_diagnostics=(
                state.nlu_agent.last_generation_diagnostic
                if isinstance(state.nlu_agent.last_generation_diagnostic, dict)
                else None
            ),
        )
        save_needs_review_checkpoint(
            checkpoint_dir,
            requirement_id=state.requirement_id,
            stage="relation_feasibility",
            reason_code="IR_RELATION_NON_CONTINGENT_REVIEW_REQUIRED",
            message=str(state.needs_review_error["error"]),
            suggested_rerun_stage="semantic",
            details=state.needs_review_error,
        )
        return
    trace_evidence_context = (
        build_trace_evidence_context(
            canonical_trace_plan,
            state.trace_evidence_policy,
        )
        if canonical_trace_plan is not None
        else None
    )

    if state.semantic is not None:
        state.semantic = attach_semantic_resolution_record(state.semantic)
        save_generated_semantic_checkpoint(
            checkpoint_dir,
            requirement_id=state.requirement_id,
            requirement=state.requirement,
            semantic=state.semantic,
            surface_segmentation=state.surface_segmentation,
            plain_reading_review=state.plain_reading_review,
            generation_diagnostics=(
                state.nlu_agent.last_generation_diagnostic
                if isinstance(state.nlu_agent.last_generation_diagnostic, dict)
                else None
            ),
        )

    round_1_trace_artifact: object | None = None
    round_1_trace_errors: list[str] = []
    adjudication_debt: tuple[str, ...] = ()
    if state.curated_traces and not state.regenerate_traces:
        if state.experiment_config is not None:
            state.traces = [
                replace(
                    trace,
                    data_split=state.experiment_config.data_split,
                    label_source=_label_source_for_config(state.experiment_config),
                )
                for trace in state.traces
            ]
        state.trace_coverage = _assess_state_trace_coverage(
            state,
            state.traces,
            requirement=state.requirement,
            feedback_history=state.trace_feedback_history,
            retry_count=round_index - 1,
            guided=bool(state.guided_context or state.expected_traces),
            allow_degraded=allow_degraded,
            trace_plan=(
                trace_evidence_context.effective_plan
                if trace_evidence_context is not None
                else None
            ),
        )
        if progress is not None:
            progress.ok(
                f"{state.requirement_id} traces loaded: {len(state.traces)}"
            )
        _resolve_expected_trace_artifact(
            state=state,
            round_index=round_index,
            records_dir=records_dir,
            progress=progress,
        )
        return

    if state.regenerate_traces or not state.traces:
        if state.semantic is None:
            raise RuntimeError(f"{state.requirement_id} 缺少语义结构，无法生成轨迹。")
        trace_context = (
            state.guided_context.attempt_view("trace_query_generation", 1)
            if state.guided_context is not None
            else None
        )
        if state.experiment_config is None:  # pragma: no cover - normalized at entry
            raise RuntimeError("ExperimentConfig is required for trace adjudication")
        # Every current path constructs pending queries before adjudication.
        if state.experiment_config is not None:
            try:
                generated_queries = _generate_state_trace_queries(
                    state,
                    feedback_history=state.trace_feedback_history,
                    target_trace_length=state.target_trace_length,
                    target_trace_count=None,
                    data_split=state.experiment_config.data_split,
                    label_source=_label_source_for_config(state.experiment_config),
                    guided_context=trace_context,
                )
                queries = filter_queries_for_policy(
                    generated_queries,
                    state.trace_evidence_policy,
                )
                fixture_annotations = []
                if state.trace_evidence_policy.boundary_source == "frozen_file":
                    if state.boundary_fixture is None:  # pragma: no cover
                        raise BoundaryTraceFixtureError(
                            "BOUNDARY_FIXTURE_NOT_FOUND",
                            "frozen boundary fixture was not loaded",
                        )
                    if trace_evidence_context is None:
                        raise BoundaryTraceFixtureError(
                            "BOUNDARY_FIXTURE_UNBOUND",
                            "frozen boundary evidence requires a validated "
                            "FaithfulSemanticIR",
                        )
                    fixture_queries, fixture_annotations = (
                        bind_frozen_boundary_evidence(
                            fixture=state.boundary_fixture,
                            requirement_id=state.requirement_id,
                            requirement=state.requirement,
                            semantic=state.semantic,
                            trace_plan=trace_evidence_context.canonical_plan,
                            data_split=state.experiment_config.data_split,
                        )
                    )
                    queries.extend(fixture_queries)
                adjudication_brief = build_trace_adjudication_brief(
                    state.requirement,
                    state.semantic,
                )
                queries = bind_trace_queries_to_brief(
                    queries,
                    adjudication_brief,
                )
            except Exception as exc:
                if not state.expected_traces:
                    raise
                state.traces = []
                round_1_trace_artifact = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                round_1_trace_errors = [
                    f"round 1 trace query generation raised {type(exc).__name__}: {exc}"
                ]
                queries = []
                fixture_annotations = []
                adjudication_brief = build_trace_adjudication_brief(
                    state.requirement,
                    state.semantic,
                )
            reused_annotations, pending_queries = reuse_confirmed_annotations(
                queries,
                state.trace_evidence_records,
                semantic_context_id=adjudication_brief.semantic_context_id,
            )
            frozen_by_id = {
                annotation.query_id: annotation
                for annotation in fixture_annotations
            }
            frozen_annotations = [
                frozen_by_id[query.query_id]
                for query in pending_queries
                if query.query_id in frozen_by_id
            ]
            pending_queries = [
                query
                for query in pending_queries
                if query.query_id not in frozen_by_id
            ]
            if state.experiment_config.mode == "formula_oracle":
                new_annotations = oracle_annotations(
                    pending_queries,
                    state.experiment_config,
                )
            elif review_mode == "auto":
                frozen_reviews = (
                    state.guided_context.attempt_view(
                        "trace_annotation", 2
                    ).gold_nl_temporal_reviews
                    if state.guided_context is not None
                    else ()
                )
                if frozen_reviews:
                    new_annotations = frozen_train_gold_annotations(
                        pending_queries,
                        frozen_reviews,
                    )
                else:
                    new_annotations = independent_judge_annotations(
                        state.requirement,
                        state.semantic,
                        pending_queries,
                        judge_client=judge_client,
                        adjudication_brief=adjudication_brief,
                    )
            else:
                new_annotations = manual_trace_annotations(
                    requirement=state.requirement, semantic=state.semantic,
                    queries=pending_queries, round_index=round_index,
                    records_dir=records_dir, adjudication_brief=adjudication_brief,
                    llm_client=state.nlu_agent.llm_client,
                )
            new_annotations = frozen_annotations + new_annotations
            annotations = reused_annotations + new_annotations
            outcome = (
                adjudicate_trace_round(
                    queries,
                    annotations,
                    state.semantic,
                    adjudication_brief,
                )
                if queries
                else None
            )
            newly_adjudicated = (
                list(outcome.accepted_traces) if outcome is not None else []
            )
            unresolved = (
                list(outcome.unresolved_queries) if outcome is not None else []
            )
            reconciliation = reconcile_trace_evidence(
                semantic=state.semantic,
                brief=adjudication_brief,
                existing_records=state.trace_evidence_records,
                existing_disputes=state.trace_evidence_disputes,
                existing_traces=(
                    state.traces if not state.trace_evidence_records else ()
                ),
                newly_adjudicated=newly_adjudicated,
            )
            conflict_resolution = resolve_trace_evidence_disputes(
                semantic=state.semantic,
                brief=adjudication_brief,
                reconciliation=reconciliation,
                resolver_client=None,
            )
            reconciliation = conflict_resolution.reconciliation
            state.trace_evidence_records = list(reconciliation.records)
            state.trace_evidence_disputes = list(reconciliation.disputes)
            state.traces = list(reconciliation.active_traces)
            if outcome is not None and outcome.has_semantic_challenge:
                state.semantic = attach_semantic_challenges(
                    state.semantic,
                    outcome.semantic_challenges,
                )
            adjudication_debt = tuple(
                dict.fromkeys(
                    (outcome.coverage_debt if outcome is not None else ())
                    + reconciliation.coverage_debt
                )
            )
            trace_annotation_record = {
                "stage": "trace_annotation",
                "round": round_index,
                "requirement_id": state.requirement_id,
                "requirement": state.requirement,
                "experiment_mode": state.experiment_config.mode,
                "oracle_assisted": False,
                "diagnostic_only": False,
                "query_count": len(queries),
                "labeled_count": len(state.traces),
                "approved": bool(
                    outcome is not None
                    and not outcome.coverage_debt
                    and not reconciliation.unresolved_disputes
                ),
                "adjudication_brief": asdict(adjudication_brief),
                "adjudication_outcome": (
                    {
                        "next_action": outcome.next_action,
                        "annotation_debt": list(outcome.annotation_debt),
                        "query_debt": list(outcome.query_debt),
                        "semantic_challenges": [
                            asdict(item) for item in outcome.semantic_challenges
                        ],
                    }
                    if outcome is not None
                    else {}
                ),
                "comments": (
                    "逐轨迹裁决完成。"
                    if not unresolved
                    else "存在待处理裁决债务；正式 IR 保持不变。"
                ),
                "annotations": [
                    build_trace_annotation_record(
                        query,
                        next(
                            (
                                annotation
                                for annotation in annotations
                                if annotation.query_id == query.query_id
                            ),
                            None,
                        ),
                        requirement=state.requirement,
                    )
                    for query in queries
                ],
                "unresolved_queries": [
                    {
                        **asdict(item),
                        "trace": [sorted(step) for step in item.trace],
                    }
                    for item in unresolved
                ],
                "trace_evidence_disputes": [
                    asdict(item) for item in reconciliation.disputes
                ],
                "background_conflict_resolutions": [
                    asdict(item) for item in conflict_resolution.reports
                ],
                "annotation_sources": sorted(
                    {annotation.label_source for annotation in annotations}
                ),
                "frozen_train_gold_used": any(
                    annotation.label_source == "warmup_train_gold"
                    for annotation in annotations
                ),
            }
            trace_annotation_record["record_path"] = str(
                _save_json_record(
                    trace_annotation_record,
                    records_dir,
                    "batch_trace_annotation",
                )
            )
            state.review_records.append(trace_annotation_record)
            if reconciliation.unresolved_disputes:
                state.regenerate_traces = False
                state.semantic_approved = False
                state.trace_approved = False
                state.trace_coverage = TraceCoverageReport(
                    status="incomplete",
                    actual_count=len(state.traces),
                    positive_count=sum(trace.expected for trace in state.traces),
                    negative_count=sum(not trace.expected for trace in state.traces),
                    boundary_count=sum(
                        trace.kind == "boundary" for trace in state.traces
                    ),
                    coverage_debt=reconciliation.coverage_debt,
                    retry_count=round_index - 1,
                    executable_semantics=False,
                    semantic_complete=False,
                    quota_complete=False,
                    reason=(
                        "Background source-first resolution could not safely "
                        "settle all trace evidence. The reviewed meaning and "
                        "original semantic review interface were preserved."
                    ),
                )
                state.pre_ir_error = {
                    "stage": "semantic_trace_review",
                    "status": "semantic_review_required",
                    "error_type": "UnresolvedTraceEvidence",
                    "error": state.trace_coverage.reason,
                    "suggested_rerun_stage": "semantic",
                    "dispute_ids": [
                        item.dispute_id
                        for item in reconciliation.unresolved_disputes
                    ],
                    "review_record_path": trace_annotation_record["record_path"],
                }
                if progress is not None:
                    progress.warn(
                        f"{state.requirement_id} background trace resolution "
                        "needs the existing semantic review flow"
                    )
                return
            if outcome is not None and outcome.has_semantic_challenge:
                state.regenerate_traces = False
                state.semantic_approved = False
                state.trace_approved = False
                state.trace_coverage = TraceCoverageReport(
                    status="incomplete",
                    actual_count=len(state.traces),
                    positive_count=sum(trace.expected for trace in state.traces),
                    negative_count=sum(not trace.expected for trace in state.traces),
                    boundary_count=sum(
                        trace.kind == "boundary" for trace in state.traces
                    ),
                    coverage_debt=outcome.coverage_debt,
                    retry_count=round_index - 1,
                    executable_semantics=False,
                    semantic_complete=False,
                    quota_complete=False,
                    reason=(
                        "Structured trace-discovered semantic challenge requires "
                        "review; the last validated IR was preserved."
                    ),
                )
                state.pre_ir_error = {
                    "stage": "semantic_trace_review",
                    "status": "semantic_review_required",
                    "error_type": "PendingSemanticChallenge",
                    "error": state.trace_coverage.reason,
                    "suggested_rerun_stage": "semantic",
                    "base_ir_version": state.semantic.faithful_ir.ir_version,
                    "challenge_ids": [
                        item.challenge_id for item in outcome.semantic_challenges
                    ],
                    "review_record_path": trace_annotation_record["record_path"],
                }
                if progress is not None:
                    progress.warn(
                        f"{state.requirement_id} semantic review required; "
                        "validated IR preserved"
                    )
                return
        if state.experiment_config is not None:
            state.traces = [
                replace(
                    trace,
                    data_split=state.experiment_config.data_split,
                )
                for trace in state.traces
            ]
        state.regenerate_traces = False
        state.trace_approved = False
        state.trace_coverage = _assess_state_trace_coverage(
            state,
            state.traces,
            requirement=state.requirement,
            feedback_history=state.trace_feedback_history,
            retry_count=round_index - 1,
            guided=bool(state.guided_context and state.guided_context.expected_traces),
            allow_degraded=allow_degraded,
            trace_plan=(
                trace_evidence_context.effective_plan
                if trace_evidence_context is not None
                else None
            ),
        )
        if adjudication_debt:
            state.trace_coverage = replace(
                state.trace_coverage,
                status=(
                    "degraded_unproven"
                    if allow_degraded
                    else "incomplete"
                ),
                coverage_debt=tuple(
                    dict.fromkeys(
                        state.trace_coverage.coverage_debt + adjudication_debt
                    )
                ),
                executable_semantics=False,
                semantic_complete=False,
                reason=(
                    "Trace adjudication has unresolved annotation/query debt; "
                    "no semantic change was inferred."
                ),
            )
    from human_review_session import review_budget_state, budget_limited_coverage, review_should_stop, REVIEW_STOP_DEBTS
    if review_mode == "manual" and state.semantic is not None and state.trace_coverage is not None:
        budget = review_budget_state(state.requirement, state.semantic,
            build_trace_adjudication_brief(state.requirement, state.semantic), records_dir)
        if review_should_stop(budget) and not state.trace_coverage.complete:
            state.trace_coverage = budget_limited_coverage(state.trace_coverage, budget)
            state.semantic_approved = state.trace_approved = False
            state.regenerate_semantic = state.regenerate_traces = False
            state.semantic_policy_pending = {"stage": "human_review_budget", "status": "candidate_fallback",
                "reason_code": sorted(REVIEW_STOP_DEBTS.intersection(state.trace_coverage.coverage_debt))[0],
                "review_budget": budget}
            state.review_records.append(dict(state.semantic_policy_pending))
            return
    if state.expected_traces:
        if round_1_trace_artifact is None:
            round_1_trace_artifact = [
                _trace_to_dict(trace) for trace in state.traces
            ]
            round_1_trace_errors = _expected_warmup_trace_errors(state)
        _resolve_expected_trace_artifact(
            state=state,
            round_index=round_index,
            records_dir=records_dir,
            progress=progress,
            round_1_artifact=round_1_trace_artifact,
            round_1_errors=round_1_trace_errors,
        )


def _wait_for_review_decisions(
    review_path: Path,
    states: list[_BatchItemState],
    *,
    progress: TerminalProgress | None = None,
) -> dict[str, BatchReviewDecision]:
    while True:
        if progress is None or not progress.enabled:
            print("")
            print("=" * 60)
            print(f"已生成批量评审文件: {review_path}")
            print("请在该 Markdown 文件中填写 pass/fail 和反馈意见。")
            print("填写完成后回到终端按 Enter 继续。")
            print("=" * 60)
            input()
        else:
            progress.wait(
                "Manual review required",
                details=[
                    f"Review file: {review_path}",
                    "Edit semantic_status / trace_status, then press Enter.",
                ],
            )
            input("填写完成后按 Enter 继续...")

        decisions = parse_batch_review_markdown(review_path)
        errors = validate_batch_review_decisions(decisions, states)
        if not errors:
            if progress is not None:
                passed = sum(
                    1
                    for decision in decisions.values()
                    if decision.semantic_status == "pass"
                    and decision.trace_status == "pass"
                )
                progress.ok(
                    f"Manual review file accepted: {passed}/{len(states)} pass"
                )
            return decisions

        if progress is None or not progress.enabled:
            print("")
            print("批量评审文件仍有未完成或格式错误的字段：")
            for error in errors:
                print(f"- {error}")
            print("请修改后再次按 Enter。")
        else:
            progress.warn("批量评审文件仍有未完成或格式错误的字段。")
            for error in errors:
                progress.info(f"- {error}")
            progress.info("请修改后再次按 Enter。")


def _resolve_expected_semantic_artifact(
    *,
    state: _BatchItemState,
    round_index: int,
    records_dir: str,
    progress: TerminalProgress | None,
    round_1_artifact: object | None = None,
    round_1_errors: list[str] | None = None,
) -> None:
    if state.expected_semantic is None:
        return
    generated = state.semantic
    errors = (
        list(round_1_errors)
        if round_1_errors is not None
        else _expected_warmup_semantic_errors(state)
    )
    generated_artifact = (
        round_1_artifact
        if round_1_artifact is not None
        else asdict(generated) if generated is not None else None
    )
    round_2_artifact: object = None
    round_2_verification = "not_required"
    artifact_origin = "generated_round_1"
    if errors:
        artifact_origin = "expected_after_round_2"
        teacher_feedback = _expected_semantic_feedback(state.expected_semantic)
        state.semantic_feedback_history.append(
            "Warmup attempt 1 mismatch. Attempt 2 may use the current-stage "
            "teacher artifact. " + teacher_feedback
        )
        if state.guided_context is not None:
            try:
                state.semantic = state.nlu_agent.generate(
                    state.requirement,
                    feedback_history=state.semantic_feedback_history,
                    guided_context=state.guided_context.attempt_view(
                        "semantic_extraction", 2
                    ),
                    surface_segmentation=state.surface_segmentation,
                    plain_reading_review=state.plain_reading_review,
                    ap_binding_context=state.ap_binding_context,
                    effective_semantic_source=state.effective_semantic_source,
                )
                if state.experiment_config is not None:
                    state.semantic = replace(
                        state.semantic,
                        data_split=state.experiment_config.data_split,
                        label_source=_label_source_for_config(state.experiment_config),
                        formula_source="none",
                        formula_visibility=(),
                        persistent_rag_allowed=_persistent_rag_allowed_for_config(
                            state.semantic,
                            state.experiment_config,
                        ),
                    )
                round_2_artifact = asdict(state.semantic)
                round_2_errors = _expected_warmup_semantic_errors(state)
                if not round_2_errors:
                    artifact_origin = "generated_round_2"
                    round_2_verification = "passed"
                else:
                    round_2_verification = "passed_after_expected_adoption"
            except Exception as exc:
                round_2_artifact = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                round_2_verification = "passed_after_expected_adoption"
        if artifact_origin != "generated_round_2":
            state.semantic = replace(
                state.expected_semantic,
                surface_segmentation=state.surface_segmentation,
                plain_reading_review=state.plain_reading_review,
            )
            if round_2_artifact is None:
                round_2_artifact = asdict(state.semantic)
        state.regenerate_traces = True
        state.semantic_approved = False
        state.trace_approved = False

    frozen_trace_gold = bool(
        state.guided_context is not None
        and state.guided_context.gold_nl_temporal_reviews
        and state.guided_context.gold_labeled_trace_evidence
    )
    if frozen_trace_gold:
        current_ir_version = (
            state.semantic.faithful_ir.ir_version
            if state.semantic is not None and state.semantic.faithful_ir is not None
            else ""
        )
        expected_ir_version = (
            state.expected_semantic.faithful_ir.ir_version
            if state.expected_semantic.faithful_ir is not None
            else ""
        )
        state.semantic = replace(
            state.expected_semantic,
            surface_segmentation=state.surface_segmentation,
            plain_reading_review=state.plain_reading_review,
        )
        artifact_origin = "frozen_train_approved_ir_snapshot"
        if current_ir_version != expected_ir_version:
            state.regenerate_traces = True
            state.trace_approved = False

    round_2_kwargs = (
        {
            "round_2_artifact": round_2_artifact,
            "round_2_verification_status": round_2_verification,
            "artifact_origin": artifact_origin,
        }
        if errors
        else {"artifact_origin": artifact_origin}
    )
    decision = build_guided_stage_decision(
        stage="semantic",
        requirement_id=state.requirement_id,
        requirement=state.requirement,
        round_index=round_index,
        generated_artifact=generated_artifact,
        expected_artifact=asdict(state.expected_semantic),
        mismatch_evidence=errors,
        llm_client=state.nlu_agent.llm_client,
        **round_2_kwargs,
    )
    if errors:
        if progress is not None:
            progress.warn(f"{state.requirement_id} semantic used guided attempt 2")
    elif progress is not None:
        progress.ok(f"{state.requirement_id} semantic matched expected artifact")
    _apply_warmup_fixture_semantic_provenance(state)
    _append_guided_decision_record(
        state=state,
        decision=decision,
        round_index=round_index,
        records_dir=records_dir,
        semantic=state.semantic,
    )


def _resolve_expected_trace_artifact(
    *,
    state: _BatchItemState,
    round_index: int,
    records_dir: str,
    progress: TerminalProgress | None,
    round_1_artifact: object | None = None,
    round_1_errors: list[str] | None = None,
) -> None:
    """Resolve warmup traces against the current IR obligation contract.

    ``Expected Finite Traces`` is not an exact target list.  A train-only seed
    may nevertheless be used as teacher evidence when, and only when, its
    labels are valid and it independently satisfies the current compiled
    ``TraceObligationPlan``.  This keeps old count/order/shape rules out of the
    main flow without making already adequate curated evidence unusable.
    """

    if not state.expected_traces:
        return
    generated = list(state.traces)
    generation_errors = (
        list(round_1_errors)
        if round_1_errors is not None
        else _expected_warmup_trace_errors(state)
    )
    generated_artifact = (
        round_1_artifact
        if round_1_artifact is not None
        else [_trace_to_dict(trace) for trace in generated]
    )
    generated_coverage = _assess_state_trace_coverage(
        state,
        generated,
        requirement=state.requirement,
        feedback_history=state.trace_feedback_history,
        guided=True,
    )
    generated_errors = [
        *generation_errors,
        *_trace_contract_errors("generated traces", generated_coverage),
    ]

    seed_traces = list(state.expected_traces)
    seed_coverage = _assess_state_trace_coverage(
        state,
        seed_traces,
        requirement=state.requirement,
        feedback_history=state.trace_feedback_history,
        guided=True,
    )
    seed_errors = [
        *_warmup_trace_label_errors(
            state,
            seed_traces,
            artifact_name="warmup seed traces",
        ),
        *_trace_contract_errors("warmup seed traces", seed_coverage),
    ]

    seed_artifact = [_trace_to_dict(trace) for trace in seed_traces]
    if generated_coverage.complete and not generation_errors:
        state.trace_coverage = generated_coverage
        decision = build_guided_stage_decision(
            stage="traces",
            requirement_id=state.requirement_id,
            requirement=state.requirement,
            round_index=round_index,
            generated_artifact=generated_artifact,
            expected_artifact=None,
            mismatch_evidence=(),
            llm_client=state.trace_agent.llm_client,
            artifact_origin="generated_round_1",
        )
        decision.update(
            status="current_contract_passed",
            decision="keep_current_obligation_evidence",
            verification_status="passed",
        )
        progress_message = "traces retained under current obligation contract"
        progress_failed = False
    elif state.guided_context is not None and (
        state.guided_context.gold_nl_temporal_reviews
        and state.guided_context.gold_labeled_trace_evidence
    ):
        (
            attempt_2_traces,
            attempt_2_coverage,
            attempt_2_artifact,
            attempt_2_errors,
        ) = _run_frozen_warmup_trace_attempt_2(state)
        if attempt_2_coverage.complete and not attempt_2_errors:
            state.traces = attempt_2_traces
            state.trace_coverage = attempt_2_coverage
            decision = build_guided_stage_decision(
                stage="traces",
                requirement_id=state.requirement_id,
                requirement=state.requirement,
                round_index=round_index,
                generated_artifact=generated_artifact,
                expected_artifact=attempt_2_artifact,
                mismatch_evidence=generated_errors,
                llm_client=state.trace_agent.llm_client,
                artifact_origin="frozen_train_gold_attempt_2",
            )
            decision.update(
                status="current_contract_passed_attempt_2",
                decision="adopt_frozen_gold_shape_after_exact_adjudication",
                verification_status="passed",
            )
            progress_message = "used frozen train-only trace gold on attempt 2"
            progress_failed = False
        else:
            state.trace_coverage = attempt_2_coverage
            decision = build_guided_stage_decision(
                stage="traces",
                requirement_id=state.requirement_id,
                requirement=state.requirement,
                round_index=round_index,
                generated_artifact=generated_artifact,
                expected_artifact=attempt_2_artifact,
                mismatch_evidence=(*generated_errors, *attempt_2_errors),
                llm_client=state.trace_agent.llm_client,
                artifact_origin="frozen_train_gold_attempt_2",
            )
            decision.update(
                status="frozen_train_gold_contract_failed",
                decision="block_without_heuristic_or_formula_fallback",
                verification_status="failed",
            )
            progress_message = "frozen train-only trace gold failed exact contract"
            progress_failed = True
    elif seed_coverage.complete and not seed_errors:
        state.traces = seed_traces
        state.trace_coverage = seed_coverage
        decision = build_guided_stage_decision(
            stage="traces",
            requirement_id=state.requirement_id,
            requirement=state.requirement,
            round_index=round_index,
            generated_artifact=generated_artifact,
            expected_artifact=seed_artifact,
            mismatch_evidence=generated_errors,
            llm_client=state.trace_agent.llm_client,
            artifact_origin="expected_current_contract_fallback",
        )
        decision.update(
            status="current_contract_seed_fallback",
            decision="adopt_seed_after_current_contract_validation",
            verification_status="passed",
        )
        progress_message = "used warmup seeds after current-contract validation"
        progress_failed = False
    else:
        state.trace_coverage = generated_coverage
        decision = build_guided_stage_decision(
            stage="traces",
            requirement_id=state.requirement_id,
            requirement=state.requirement,
            round_index=round_index,
            generated_artifact=generated_artifact,
            expected_artifact=None,
            mismatch_evidence=(*generated_errors, *seed_errors),
            llm_client=state.trace_agent.llm_client,
            artifact_origin="generated_round_1",
        )
        decision.update(
            status="warmup_evidence_migration_required",
            decision="keep_current_evidence_and_block_until_contract_complete",
            verification_status="failed",
        )
        progress_message = "warmup trace evidence needs current-contract migration"
        progress_failed = True

    decision["legacy_seed_traces"] = seed_artifact
    decision["seed_contract_status"] = seed_coverage.status
    decision["seed_contract_errors"] = seed_errors
    if progress is not None:
        if progress_failed:
            progress.warn(f"{state.requirement_id} {progress_message}")
        else:
            progress.ok(f"{state.requirement_id} {progress_message}")
    _apply_warmup_fixture_trace_provenance(state)
    _append_guided_decision_record(
        state=state,
        decision=decision,
        round_index=round_index,
        records_dir=records_dir,
        semantic=state.semantic,
    )


def _run_frozen_warmup_trace_attempt_2(
    state: _BatchItemState,
) -> tuple[list[LabeledTrace], TraceCoverageReport, list[dict], list[str]]:
    if state.semantic is None or state.guided_context is None:
        raise RuntimeError("Frozen warmup trace attempt requires semantic context.")
    try:
        generation_view = state.guided_context.attempt_view(
            "trace_query_generation", 2
        )
        queries = _generate_state_trace_queries(
            state,
            feedback_history=state.trace_feedback_history,
            target_trace_length=state.target_trace_length,
            target_trace_count=None,
            data_split="train",
            label_source="warmup_train_gold",
            guided_context=generation_view,
        )
        review_view = state.guided_context.attempt_view("trace_annotation", 2)
        annotations = frozen_train_gold_annotations(
            queries,
            review_view.gold_nl_temporal_reviews,
        )
        traces, unresolved = adjudicate_trace_queries(queries, annotations)
        errors = []
        if unresolved:
            errors.append(
                "frozen gold left unresolved queries: "
                + ", ".join(query.query_id for query in unresolved)
            )
        coverage = _assess_state_trace_coverage(
            state,
            traces,
            requirement=state.requirement,
            feedback_history=state.trace_feedback_history,
            guided=True,
        )
        errors.extend(_trace_contract_errors("frozen attempt 2", coverage))
        return traces, coverage, [_trace_to_dict(trace) for trace in traces], errors
    except Exception as exc:
        coverage = TraceCoverageReport(
            status="incomplete",
            coverage_debt=(f"frozen_attempt_2:{type(exc).__name__}:{exc}",),
            reason=str(exc),
        )
        return [], coverage, [], [
            f"frozen attempt 2 raised {type(exc).__name__}: {exc}"
        ]


def _append_guided_decision_record(
    *,
    state: _BatchItemState,
    decision: dict,
    round_index: int,
    records_dir: str,
    semantic: SemanticStructure | None,
) -> None:
    rag_chunk_ids = record_guided_stage_decision(
        retriever=state.retriever,
        requirement=state.requirement,
        semantic=semantic,
        decision_record=decision,
    )
    record = {
        "stage": f"{decision['stage']}_guided_decision",
        "round": round_index,
        "requirement_id": state.requirement_id,
        "requirement": state.requirement,
        "approved": True,
        "runtime_rag_chunks": rag_chunk_ids,
        "guided_decision": decision,
        "comments": decision["decision_reason"],
    }
    record["record_path"] = str(
        _save_json_record(record, records_dir, "batch_guided_decision")
    )
    state.review_records.append(record)


def _append_expected_fallback_record(
    *,
    state: _BatchItemState,
    stage: str,
    generated_artifact: object,
    expected_artifact: object,
    mismatch_evidence: list[str],
    round_index: int,
    records_dir: str,
    progress: TerminalProgress | None,
) -> None:
    decision = build_guided_stage_decision(
        stage=stage,
        requirement_id=state.requirement_id,
        requirement=state.requirement,
        round_index=round_index,
        generated_artifact=generated_artifact,
        expected_artifact=expected_artifact,
        mismatch_evidence=mismatch_evidence,
        llm_client=(
            state.nlu_agent.llm_client
            if stage == "semantic"
            else state.trace_agent.llm_client
        ),
    )
    if progress is not None:
        progress.warn(
            f"{state.requirement_id} {stage} fallback to expected artifact"
        )
    _append_guided_decision_record(
        state=state,
        decision=decision,
        round_index=round_index,
        records_dir=records_dir,
        semantic=state.semantic,
    )


def _apply_warmup_fixture_semantic_provenance(state: _BatchItemState) -> None:
    if state.semantic is None or state.expected_semantic is None:
        return
    data_split = (
        state.experiment_config.data_split
        if state.experiment_config is not None
        else state.semantic.data_split
    )
    persistent_rag_allowed = (
        _persistent_rag_allowed_for_config(state.semantic, state.experiment_config)
        if state.experiment_config is not None
        else state.semantic.persistent_rag_allowed
    )
    state.semantic = replace(
        state.semantic,
        data_split=data_split,
        label_source=WARMUP_FIXTURE_LABEL_SOURCE,
        formula_source="none",
        formula_visibility=(),
        persistent_rag_allowed=persistent_rag_allowed,
    )


def _apply_warmup_fixture_trace_provenance(state: _BatchItemState) -> None:
    if not state.expected_traces:
        return
    data_split = (
        state.experiment_config.data_split
        if state.experiment_config is not None
        else None
    )
    state.traces = [
        replace(
            trace,
            data_split=(data_split or trace.data_split),
            label_source=(
                "warmup_train_gold"
                if trace.label_source == "warmup_train_gold"
                else trace.label_source
            ),
            formula_source="none",
            formula_visibility=(),
            annotation_locked=True,
        )
        for trace in state.traces
    ]


def _label_source_for_config(config: ExperimentConfig) -> str:
    if config.mode == "formula_oracle":
        return "formula_oracle"
    if config.mode == "independent_judge":
        return "independent_judge"
    return "human"


def _persistent_rag_allowed_for_config(
    semantic: SemanticStructure,
    config: ExperimentConfig,
) -> bool:
    return semantic.persistent_rag_allowed and config.data_split == "train"


def _auto_review_decisions(states: list[_BatchItemState]) -> dict[str, BatchReviewDecision]:
    decisions: dict[str, BatchReviewDecision] = {}
    for state in states:
        consistency_errors = _semantic_trace_consistency_errors(
            requirement_id=state.requirement_id,
            semantic=state.semantic,
            traces=state.traces,
        )
        expected_semantic_errors = _expected_warmup_semantic_errors(state)
        expected_trace_errors = _expected_warmup_trace_errors(state)
        if consistency_errors or expected_semantic_errors:
            feedback = "\n".join(
                [
                    *consistency_errors,
                    *expected_semantic_errors,
                    (
                        "请重新抽取真实 AP。若触发或响应包含复杂布尔/时序结构，"
                        "请在 components 中写入 trigger_definition、"
                        "response_definition 或 response_formula，并在 aps 中列出"
                        "真实 AP；不要使用未绑定占位符。components 中的公式片段"
                        "只能使用当前支持的未来时 LTLf 片段："
                        + SUPPORTED_LTLF_FORMULA_GUIDANCE
                    ),
                ]
            )
            decisions[state.requirement_id] = BatchReviewDecision(
                requirement_id=state.requirement_id,
                semantic_status="fail",
                trace_status="fail",
                semantic_feedback=feedback,
                trace_feedback="语义结构不一致，轨迹需随修正后的语义重新生成。",
            )
            continue

        if state.trace_coverage.status in {"incomplete", "rejected"}:
            decisions[state.requirement_id] = BatchReviewDecision(
                requirement_id=state.requirement_id,
                semantic_status="pass",
                trace_status="fail",
                semantic_feedback="自动批量评审：语义结构通过。",
                trace_feedback=(
                    trace_coverage_feedback(state.trace_coverage)
                    or state.trace_coverage.reason
                ),
            )
            continue

        if expected_trace_errors:
            decisions[state.requirement_id] = BatchReviewDecision(
                requirement_id=state.requirement_id,
                semantic_status="pass",
                trace_status="fail",
                semantic_feedback="自动样本评审：语义结构符合 warmup 标准。",
                trace_feedback="\n".join(expected_trace_errors),
            )
            continue

        decisions[state.requirement_id] = BatchReviewDecision(
            requirement_id=state.requirement_id,
            semantic_status="pass",
            trace_status="pass",
            semantic_feedback=(
                "自动样本评审通过。"
                if state.expected_semantic is not None
                else "自动批量评审通过。"
            ),
            trace_feedback=(
                state.trace_coverage.reason
                if state.trace_coverage.degraded
                else "自动样本评审通过。"
                if state.expected_traces
                else "自动批量评审通过。"
            ),
        )
    return decisions


def _expected_warmup_semantic_errors(state: _BatchItemState) -> list[str]:
    expected = state.expected_semantic
    actual = state.semantic
    if expected is None or actual is None:
        return []

    errors: list[str] = []
    for field_name in (
        "trigger",
        "response",
        "pattern",
        "scope_keyword",
        "temporal_keyword",
    ):
        actual_value = _canonical_text(getattr(actual, field_name))
        expected_value = _canonical_text(getattr(expected, field_name))
        if actual_value != expected_value:
            errors.append(
                f"{state.requirement_id}: warmup 样本语义字段 {field_name} "
                f"应为 {getattr(expected, field_name)!r}，实际为 "
                f"{getattr(actual, field_name)!r}。"
            )

    expected_aps = set(expected.aps or ())
    actual_aps = set(actual.aps or ())
    if expected_aps and not expected_aps.issubset(actual_aps):
        errors.append(
            f"{state.requirement_id}: warmup 样本 AP 应包含 "
            f"{', '.join(sorted(expected_aps))}，实际为 "
            f"{', '.join(sorted(actual_aps)) or '(none)'}。"
        )

    expected_components = dict(expected.components)
    actual_components = dict(actual.components)
    for key, expected_value in expected_components.items():
        actual_value = actual_components.get(key)
        if actual_value is None:
            errors.append(
                f"{state.requirement_id}: warmup 样本缺少 components.{key}。"
            )
            continue
        if _canonical_text(actual_value) != _canonical_text(expected_value):
            errors.append(
                f"{state.requirement_id}: warmup 样本 components.{key} "
                f"应为 {expected_value!r}，实际为 {actual_value!r}。"
            )

    expected_ir = expected.faithful_ir
    actual_ir = actual.faithful_ir
    if expected_ir is not None:
        if actual_ir is None:
            errors.append(
                f"{state.requirement_id}: warmup 样本缺少 FaithfulSemanticIR。"
            )
        elif _canonical_warmup_ir(actual_ir) != _canonical_warmup_ir(expected_ir):
            errors.append(
                f"{state.requirement_id}: warmup 样本 FaithfulSemanticIR 的拓扑、"
                "时序语义范围或边界策略与 Expected 不一致。"
            )

    if errors:
        errors.append(_expected_semantic_feedback(expected))
    return errors


def _canonical_warmup_ir(ir: object) -> object:
    """Normalize IR semantics while ignoring ids, evidence text, and provenance."""

    node_map = getattr(ir, "node_map")

    def normalize_value(value: object) -> object:
        if isinstance(value, dict):
            return tuple(
                (str(key), normalize_value(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        if isinstance(value, (list, tuple)):
            return tuple(normalize_value(item) for item in value)
        return value

    def normalize_node(node_id: str) -> object:
        node = node_map[node_id]
        children = [normalize_node(child_id) for child_id in node.children]
        if node.node_type in {"conjunction", "disjunction", "equivalence"}:
            children = sorted(children, key=repr)
        parameters = tuple(
            (name, normalize_value(value))
            for name, value in sorted(node.parameters, key=lambda item: item[0])
            if name != "ap_id"
        )
        return (
            node.node_type,
            tuple(children),
            node.scope,
            node.temporal_relation,
            node.boundary_policy,
            tuple(sorted(node.allowed_semantic_readings)),
            node.temporal_resolution,
            parameters,
        )

    return (
        tuple(sorted(getattr(ir, "ap_names"))),
        normalize_node(getattr(ir, "root_node_id")),
    )


def _expected_warmup_trace_errors(state: _BatchItemState) -> list[str]:
    """Validate train-only labels without imposing a second coverage contract.

    Legacy warmup review required the generated trace list to match the seed
    examples in count, kind, order, and shape.  The current source of truth is
    the compiled TraceObligationPlan, so seed traces are no longer a closed
    target set and may be supplemented or replaced by better witnesses.
    """

    if not state.expected_traces:
        return []
    if state.semantic is None:
        return []

    return _warmup_trace_label_errors(
        state,
        state.traces,
        artifact_name="generated traces",
    )


def _warmup_trace_label_errors(
    state: _BatchItemState,
    traces: list[LabeledTrace],
    *,
    artifact_name: str,
) -> list[str]:
    expected_formula = _expected_formula_for_review(state.expected_semantic)
    if expected_formula is None:
        return []
    mismatches = [
        trace.name
        for trace in traces
        if _trace_formula_truth(trace, expected_formula) != trace.expected
    ]
    if not mismatches:
        return []
    return [
        f"{state.requirement_id}: {artifact_name} 中以下轨迹标签与 train-only "
        f"approved_formula 不一致: {', '.join(mismatches)}。",
        _expected_trace_feedback(state.expected_traces),
    ]


def _trace_contract_errors(label: str, coverage) -> list[str]:
    if coverage.complete:
        return []
    debt = tuple(dict.fromkeys((*coverage.missing, *coverage.coverage_debt)))
    detail = "; ".join(debt) or coverage.reason or coverage.status
    return [f"{label} do not satisfy the current TraceObligationPlan: {detail}"]


def _expected_formula_for_review(
    semantic: SemanticStructure | None,
):
    if semantic is None or not semantic.formula_text:
        return None
    try:
        return parse_learned_formula(semantic.formula_text)
    except ValueError:
        return None


def _trace_formula_truth(trace: LabeledTrace, formula) -> bool:
    from ltlf_checker import check_trace

    return check_trace(trace.trace, formula)


def _expected_semantic_feedback(expected: SemanticStructure) -> str:
    lines = [
        "请按 warmup 样本标准重新生成语义结构：",
        f"trigger: {expected.trigger}",
        f"response: {expected.response}",
        f"pattern: {expected.pattern}",
        f"scope_keyword: {expected.scope_keyword}",
        f"temporal_keyword: {expected.temporal_keyword}",
    ]
    if expected.aps:
        lines.append("aps: " + ", ".join(expected.aps))
    if expected.components:
        lines.append("components:")
        lines.extend(f"- {key}: {value}" for key, value in expected.components)
    lines.append("注意：这是第 2 次且唯一一次教师引导尝试；第 1 次自主结果已完成并被确定性差异检查判为不匹配。")
    return "\n".join(lines)


def _expected_trace_feedback(expected_traces: list[LabeledTrace]) -> str:
    lines = [
        "warmup 的 Expected Finite Traces 只是 train-only 参考种子，不定义独立的数量、类型、顺序或形状配额。",
        "请以当前 FaithfulSemanticIR 编译出的 TraceObligationPlan 为唯一覆盖契约，并补齐其报告的缺失 obligation。",
        "离线校验仍要求每条生成轨迹的标签与 approved_formula 的 LTLf 有限轨迹求值一致。",
        "参考种子如下；允许替换或增加轨迹：",
    ]
    for trace in expected_traces:
        lines.append(
            f"- {trace.kind} {trace.name}: expected={format_truth(trace.expected)}, "
            f"length={len(trace.trace)}, trace={format_trace(trace.trace)}"
        )
    return "\n".join(lines)


def _canonical_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _semantic_trace_consistency_errors(
    *,
    requirement_id: str,
    semantic: SemanticStructure | None,
    traces: list[LabeledTrace],
) -> list[str]:
    if semantic is None:
        return [f"{requirement_id}: 缺少语义结构。"]

    readiness_errors = trace_generation_readiness_errors(semantic.faithful_ir)
    if readiness_errors:
        return [
            f"{requirement_id}: Semantic IR 尚未完成参数审查，不能生成有限轨迹："
            + "; ".join(readiness_errors)
            + "。请在语义阶段依据已审核普通读法或明确的项目策略补齐参数；不得在轨迹阶段默认。"
        ]

    if not traces:
        return [
            f"{requirement_id}: 缺少有限轨迹；已审核语义必须配套 positive、"
            "negative 和 boundary 轨迹后才能进入 skeleton/completion 阶段。"
        ]

    if semantic.formula_text:
        try:
            parse_learned_formula(semantic.formula_text)
        except ValueError as exc:
            return [f"{requirement_id}: formula_text 无法解析: {exc}"]
        return []

    pattern = semantic.pattern.strip().lower().replace("-", "_").replace(" ", "_")
    if pattern != "response":
        return []

    trace_atoms = _trace_atoms(traces)
    if not trace_atoms:
        return []

    missing = [
        ap
        for field_name, ap in (
            ("trigger", semantic.trigger),
            ("response", semantic.response),
        )
        if ap
        and ap not in trace_atoms
        and not _endpoint_alias_is_bound(semantic, field_name, ap)
    ]
    if not missing:
        return []

    return [
        (
            f"{requirement_id}: semantic.trigger/response 包含未出现在有限轨迹中的"
            f"未绑定 AP {', '.join(missing)}；轨迹实际 AP 为 "
            f"{', '.join(sorted(trace_atoms))}。这通常表示使用了未绑定占位符，"
            "或该需求需要在 components 中给出 trigger/response 绑定定义。"
        )
    ]


def _endpoint_alias_is_bound(
    semantic: SemanticStructure,
    field_name: str,
    ap_name: str,
) -> bool:
    component_keys = (
        ("trigger_definition", "trigger_formula")
        if field_name == "trigger"
        else ("response_definition", "response_formula")
    )
    components = dict(semantic.components)
    for key in component_keys:
        lhs, rhs = _definition_sides(components.get(key, ""))
        if lhs == ap_name and rhs:
            return True
    return False


def _definition_sides(value: str) -> tuple[str, str]:
    text = value.strip()
    if not text:
        return "", ""
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|=)\s*(.+?)\s*$", text)
    if not match:
        return "", text
    return match.group(1).strip(), match.group(2).strip()


def _trace_atoms(traces: list[LabeledTrace]) -> set[str]:
    atoms: set[str] = set()
    for trace in traces:
        for step in trace.trace:
            atoms.update(step)
    return atoms


def _apply_batch_decision(
    *,
    state: _BatchItemState,
    decision: BatchReviewDecision,
    round_index: int,
    review_path: Path,
    records_dir: str,
) -> list[str]:
    semantic_passed = decision.semantic_status == "pass"
    trace_passed = semantic_passed and decision.trace_status == "pass"

    semantic_record = _save_semantic_batch_record(
        state=state,
        decision=decision,
        round_index=round_index,
        review_path=review_path,
        records_dir=records_dir,
        approved=semantic_passed,
    )
    trace_record = _save_trace_batch_record(
        state=state,
        decision=decision,
        round_index=round_index,
        review_path=review_path,
        records_dir=records_dir,
        approved=trace_passed,
        semantic_failed=not semantic_passed,
    )
    state.review_records.extend([semantic_record, trace_record])

    if semantic_passed:
        state.semantic_approved = True
    else:
        state.semantic_approved = False
        state.trace_approved = False
        state.regenerate_semantic = True
        state.regenerate_traces = True
        state.semantic_feedback_history.append(
            _feedback_or_default(
                decision.semantic_feedback,
                "语义结构未通过，但未填写具体意见。",
            )
        )

    if decision.trace_status == "fail":
        state.trace_feedback_history.append(
            _feedback_or_default(
                decision.trace_feedback,
                "有限轨迹未通过，但未填写具体意见。",
            )
        )
        if decision.trace_issue_type == "length":
            state.target_trace_length = decision.target_trace_length

    if semantic_passed and trace_passed:
        state.trace_approved = True
        return list(trace_record.get("runtime_rag_chunks", []))
    elif semantic_passed:
        state.trace_approved = False
        state.regenerate_traces = True
    else:
        state.trace_approved = False
    return []


def _save_semantic_batch_record(
    *,
    state: _BatchItemState,
    decision: BatchReviewDecision,
    round_index: int,
    review_path: Path,
    records_dir: str,
    approved: bool,
) -> dict:
    if state.semantic is None:
        raise RuntimeError(f"{state.requirement_id} 缺少语义结构。")

    stage_errors: list[StageErrorRecord] = []
    rag_chunk_ids: list[str] = []
    if state.curated_semantic:
        comments = "使用外部已审核语义结构，未调用 NLU Agent。"
    elif state.guided_context is not None:
        comments = (
            "warmup 两次尝试语义生成：第 1 次不暴露 Expected Semantic "
            "Structure；仅在确定性比较失败后，第 2 次开放本阶段教师信号。"
        )
    elif state.expected_semantic is not None:
        comments = (
            "自动样本评审通过：生成语义结构符合 warmup Expected Semantic "
            "Structure；样本答案只作为严格评审标准，未绕过 NLU Agent。"
        )
    else:
        comments = "批量 Markdown 评审通过。"
    if not approved:
        comments = _feedback_or_default(
            decision.semantic_feedback,
            "批量 Markdown 评审未通过，但未填写具体意见。",
        )
        stage_errors = [
            _build_semantic_batch_error(
                state=state,
                comments=comments,
                round_index=round_index,
            )
        ]
        rag_chunk_ids = record_stage_errors(
            retriever=state.retriever,
            requirement=state.requirement,
            semantic=state.semantic,
            stage="semantic",
            round_index=round_index,
            errors=stage_errors,
        )

    record = {
        "stage": "semantic",
        "round": round_index,
        "requirement_id": state.requirement_id,
        "requirement": state.requirement,
        "artifact": asdict(state.semantic),
        "approved": approved,
        "issue_type": "approved" if approved else "content",
        "batch_review_path": str(review_path),
        "runtime_rag_chunks": rag_chunk_ids,
        "stage_errors": [asdict(error) for error in stage_errors],
        "comments": comments,
    }
    record["record_path"] = str(_save_json_record(record, records_dir, "batch_semantic"))
    return record


def _save_trace_batch_record(
    *,
    state: _BatchItemState,
    decision: BatchReviewDecision,
    round_index: int,
    review_path: Path,
    records_dir: str,
    approved: bool,
    semantic_failed: bool,
) -> dict:
    if state.semantic is None:
        raise RuntimeError(f"{state.requirement_id} 缺少语义结构。")
    if approved and state.trace_coverage.degraded:
        state.semantic = replace(
            state.semantic,
            persistent_rag_allowed=False,
        )

    stage_errors: list[StageErrorRecord] = []
    rag_chunk_ids: list[str] = []
    discarded = not approved
    issue_type = (
        state.trace_coverage.status
        if approved and state.trace_coverage.degraded
        else "approved"
        if approved
        else decision.trace_issue_type
    )
    if state.curated_traces:
        comments = "使用外部已审核有限轨迹，未调用 Trace Agent。"
    elif state.guided_context is not None:
        comments = (
            "warmup 两次尝试轨迹生成：第 1 次不暴露 Expected Finite "
            "Traces 或预期条数；仅在比较失败后，第 2 次开放本阶段教师信号。"
        )
    elif state.expected_traces:
        comments = (
            "自动样本评审通过：生成有限轨迹符合 warmup Expected Finite "
            "Traces 覆盖和 approved_formula 标签校验；样本答案只作为严格"
            "评审标准，未绕过 Trace Agent。"
        )
    else:
        comments = "批量 Markdown 评审通过。"
    if state.curated_traces and state.curated_trace_source:
        comments = f"使用外部 {state.curated_trace_source}，未调用 Trace Agent。"

    if (
        approved
        and state.trace_coverage.complete
        and not state.trace_evidence_policy.experimental_only
    ):
        rag_chunk_ids = record_approved_review_examples(
            retriever=state.retriever,
            requirement_id=state.requirement_id,
            requirement=state.requirement,
            semantic=state.semantic,
            traces=state.traces,
            round_index=round_index,
            comments=comments,
        )
    elif approved and state.trace_coverage.degraded:
        comments = (
            f"{state.trace_coverage.reason} 当前证据仅用于本次后续推导，"
            "未写入持久化 RAG。"
        )
    elif not approved:
        comments = _feedback_or_default(
            decision.trace_feedback,
            "语义结构未通过，当前有限轨迹随语义结构重生成而丢弃。"
            if semantic_failed
            else "批量 Markdown 评审未通过，但未填写具体意见。",
        )
        if semantic_failed and decision.trace_status == "pass":
            issue_type = "semantic_changed"
        elif decision.trace_issue_type == "length" and decision.target_trace_length is not None:
            issue_type = "length"
            rag_chunk_ids = record_trace_length_issue(
                retriever=state.retriever,
                requirement=state.requirement,
                semantic=state.semantic,
                traces=state.traces,
                round_index=round_index,
                target_trace_length=decision.target_trace_length,
                comments=comments,
            )
        else:
            issue_type = "content"
            stage_errors = [
                _build_trace_batch_error(
                    state=state,
                    comments=comments,
                    round_index=round_index,
                )
            ]
            if state.trace_coverage.complete:
                rag_chunk_ids = record_stage_errors(
                    retriever=state.retriever,
                    requirement=state.requirement,
                    semantic=state.semantic,
                    stage="traces",
                    round_index=round_index,
                    errors=stage_errors,
                )

    record = {
        "stage": "traces",
        "round": round_index,
        "requirement_id": state.requirement_id,
        "requirement": state.requirement,
        "semantic": asdict(state.semantic),
        "experiment_mode": (
            state.experiment_config.mode if state.experiment_config is not None else "legacy"
        ),
        "oracle_assisted": False,
        "diagnostic_only": False,
        "artifact": [_trace_to_dict(trace) for trace in state.traces],
        "coverage_report": asdict(state.trace_coverage),
        "approved": approved,
        "issue_type": issue_type,
        "requested_trace_length": state.target_trace_length,
        "target_trace_length": decision.target_trace_length,
        "discarded": discarded,
        "batch_review_path": str(review_path),
        "runtime_rag_chunks": rag_chunk_ids,
        "stage_errors": [asdict(error) for error in stage_errors],
        "comments": comments,
    }
    if discarded:
        record["discarded_reason"] = (
            "semantic_changed" if semantic_failed and decision.trace_status == "pass" else issue_type
        )
    record["record_path"] = str(_save_json_record(record, records_dir, "batch_traces"))
    return record


def _state_to_reviewed_item(state: _BatchItemState) -> ReviewedBatchItem:
    if (
        state.semantic is None
        and state.pre_ir_error is None
        and state.plain_reading_resolution_pending is None
        and state.semantic_policy_pending is None
        and state.needs_review_error is None
    ):
        raise RuntimeError(f"{state.requirement_id} 缺少已通过评审的语义结构。")
    state.pre_ir_disposition = _typed_pre_ir_disposition(state)
    return ReviewedBatchItem(
        requirement_id=state.requirement_id,
        requirement=state.requirement,
        semantic=state.semantic,
        traces=state.traces,
        retriever=state.retriever,
        review_records=state.review_records,
        guided_context=state.guided_context,
        trace_coverage=state.trace_coverage,
        experiment_config=state.experiment_config,
        surface_segmentation=state.surface_segmentation,
        plain_reading_review=state.plain_reading_review,
        pre_ir_error=state.pre_ir_error,
        plain_reading_resolution_pending=(
            state.plain_reading_resolution_pending
        ),
        semantic_policy_pending=state.semantic_policy_pending,
        original_requirement=state.original_requirement,
        human_semantic_override=state.human_semantic_override,
        trace_evidence_records=tuple(state.trace_evidence_records),
        trace_evidence_disputes=tuple(state.trace_evidence_disputes),
        ap_binding_context=state.ap_binding_context,
        needs_review_error=state.needs_review_error,
        pre_ir_disposition=state.pre_ir_disposition,
        authoritative_trace_session=state.authoritative_trace_session,
        effective_semantic_source=state.effective_semantic_source,
    )


def _typed_pre_ir_disposition(state: _BatchItemState) -> PreIRDisposition:
    if state.approved:
        return PreIRDisposition(state.requirement_id, "approved")

    pending_payload = next(
        (
            payload
            for payload in (
                state.plain_reading_resolution_pending,
                state.semantic_policy_pending,
                state.needs_review_error,
            )
            if payload is not None
        ),
        None,
    )
    if pending_payload is None and state.pre_ir_error is not None:
        if _payload_is_review_pending(state.pre_ir_error):
            pending_payload = state.pre_ir_error
    if pending_payload is not None:
        stage = str(pending_payload.get("stage", "pre_ir")) or "pre_ir"
        restart_stage = str(
            pending_payload.get("suggested_rerun_stage", stage)
        ) or stage
        reason_code = str(
            pending_payload.get("reason_code")
            or pending_payload.get("status")
            or "PRE_IR_REVIEW_REQUIRED"
        ).upper()
        evidence_paths = tuple(
            dict.fromkeys(
                str(value)
                for key in ("record_path", "review_record_path")
                if (value := pending_payload.get(key))
            )
        )
        source_ir_version = ""
        if state.semantic is not None and state.semantic.faithful_ir is not None:
            source_ir_version = state.semantic.faithful_ir.ir_version
        record = ReviewPendingRecord(
            reason_code=reason_code,
            owner_stage=stage,
            safe_restart_stage=restart_stage,
            requirement_id=state.requirement_id,
            source_ir_version=source_ir_version,
            affected_ir_node_ids=tuple(
                str(value)
                for value in pending_payload.get("affected_ir_node_ids", ())
            ),
            dispute_ids=tuple(
                str(value) for value in pending_payload.get("dispute_ids", ())
            ),
            challenge_ids=tuple(
                str(value) for value in pending_payload.get("challenge_ids", ())
            ),
            evidence_record_paths=evidence_paths,
            required_action=str(
                pending_payload.get("required_action", "manual_review")
            ),
            retry_policy="retry_after_required_action",
        )
        return PreIRDisposition(
            state.requirement_id,
            "review_pending",
            review_pending=record,
        )

    failure = state.pre_ir_error or {
        "stage": "semantic_trace_review",
        "status": "blocked_error",
        "error_type": "ReviewExhausted",
        "error": "review rounds exhausted without an approved or reviewable state",
        "suggested_rerun_stage": "pipeline",
    }
    blocked = BlockedFailureRecord(
        reason_code=str(failure.get("reason_code") or failure.get("status") or "BLOCKED_ERROR").upper(),
        owner_stage=str(failure.get("stage", "pre_ir")),
        safe_restart_stage=str(failure.get("suggested_rerun_stage", "pipeline")),
        requirement_id=state.requirement_id,
        error_type=str(failure.get("error_type", "PipelineError")),
        message=str(failure.get("error", "pipeline failure")),
        source_ir_version=(
            state.semantic.faithful_ir.ir_version
            if state.semantic is not None and state.semantic.faithful_ir is not None
            else ""
        ),
        retryable=False,
    )
    return PreIRDisposition(
        state.requirement_id,
        "blocked_error",
        blocked_failure=blocked,
    )


def _payload_is_review_pending(payload: Mapping[str, object]) -> bool:
    status = str(payload.get("status", "")).casefold()
    error_type = str(payload.get("error_type", ""))
    return (
        status in _TYPED_REVIEW_PENDING_STATUSES
        or error_type in _TYPED_REVIEW_PENDING_ERROR_TYPES
    )


def _adjudicate_ir_candidate_semantic(
    state: _BatchItemState,
    *,
    checkpoint_dir: str | None,
    records_dir: str,
    review_mode: str,
    round_index: int,
    progress: TerminalProgress | None,
    judge_client: JSONLLMClient | None,
) -> None:
    """Resolve IR alternatives using human labels on ordinary finite traces."""

    del review_mode
    semantic = state.semantic
    if semantic is None or semantic.candidate_ir_set is None:
        raise RuntimeError("IR candidate adjudication requires CandidateIRSet")
    data_split = (
        state.experiment_config.data_split
        if state.experiment_config is not None
        else "test"
    )
    selection = adjudicate_ir_candidate_branches(
        state.requirement,
        semantic,
        records_dir=records_dir,
        llm_client=state.nlu_agent.llm_client,
        data_split=data_split,
    )
    report = selection.semantic.ir_ambiguity_adjudication
    from human_review_session import semantic_review_budget
    selection_record = {
        "stage": "ir_candidate_trace_adjudication",
        "round": round_index,
        "requirement_id": state.requirement_id,
        "candidate_set_id": report.candidate_set_id if report is not None else "",
        "status": selection.status,
        "selected_branch_id": selection.selected_branch_id,
        "consistent_branch_ids": list(selection.consistent_branch_ids),
        "supporting_query_ids": list(selection.supporting_query_ids),
        "reason": selection.reason,
        "provisional": selection.status == "provisional_fallback",
        "human_review_budget": semantic_review_budget(state.requirement, selection.semantic, records_dir),
    }
    state.review_records.append(selection_record)
    if selection.status != "unique":
        state.semantic = selection.semantic
        state.semantic_policy_pending = {
            **selection_record,
            "status": selection.status,
            "suggested_rerun_stage": "ir_candidate_trace_adjudication",
            "user_message": (
                "人工轨迹判别尚未唯一确定语义；当前候选仅为临时选择，"
                "请继续普通语言语义审核。"
            ),
        }
        if progress is not None:
            progress.warn(
                f"{state.requirement_id} IR candidate selection is "
                f"{selection.status}: {selection.selected_branch_id or 'none'}"
            )
        save_semantic_policy_decision_checkpoint(
            checkpoint_dir,
            requirement_id=state.requirement_id,
            semantic=state.semantic,
            selection=selection_record,
        )
        return

    previous_review = state.plain_reading_review
    state.plain_reading_review = selection.semantic.plain_reading_review
    if (checkpoint_dir and previous_review is not None and state.plain_reading_review is not None
            and previous_review.reading_review_id != state.plain_reading_review.reading_review_id):
        from pre_ir_batch_review import persist_trace_selected_plain_reading
        persist_trace_selected_plain_reading(
            checkpoint_dir=checkpoint_dir, segmentation=state.surface_segmentation,
            previous_review_id=previous_review.reading_review_id, review=state.plain_reading_review,
        )
    state.semantic = attach_semantic_resolution_record(selection.semantic)
    state.semantic_policy_pending = None
    state.regenerate_semantic = False
    state.regenerate_traces = True
    state.semantic_approved = False
    state.trace_approved = False
    save_semantic_policy_decision_checkpoint(
        checkpoint_dir,
        requirement_id=state.requirement_id,
        semantic=state.semantic,
        selection=selection_record,
    )


def _chunks_by_ids(
    retriever: KeywordRetriever,
    chunk_ids: list[str],
) -> list[KnowledgeChunk]:
    wanted = set(chunk_ids)
    return [chunk for chunk in retriever.chunks if chunk.chunk_id in wanted]


def _share_approved_chunks(
    states: list[_BatchItemState],
    chunks: list[KnowledgeChunk],
) -> None:
    for state in states:
        extend_retriever_once(state.retriever, chunks)


def _format_batch_review_markdown(
    *,
    states: list[_BatchItemState],
    round_index: int,
    auto_approve: bool,
) -> str:
    lines = [
        f"# Traj-LTLf Batch Review Round {round_index}",
        "",
        "将每个待评审项的状态改为 pass 或 fail。",
        "如果失败，请在对应 Feedback 标记之间写明原因；LLM 下一轮会读取这些意见。",
        "trace_issue_type 可填 content 或 length；length 需要填写 target_trace_length。",
        "",
    ]

    auto_decisions = _auto_review_decisions(states) if auto_approve else {}
    for state in states:
        semantic_status = _initial_status(state.semantic_approved, auto_approve)
        trace_status = _initial_status(state.trace_approved, auto_approve)
        auto_decision = auto_decisions.get(state.requirement_id)
        if auto_decision is not None:
            semantic_status = auto_decision.semantic_status
            trace_status = auto_decision.trace_status
        readiness_errors = trace_generation_readiness_errors(
            state.semantic.faithful_ir
            if state.semantic is not None
            else None
        )
        if auto_approve and readiness_errors:
            semantic_status = "fail"
            trace_status = "fail"
        if auto_approve and state.trace_coverage.status in {"incomplete", "rejected"}:
            trace_status = "fail"
        auto_semantic_feedback = (
            auto_decision.semantic_feedback
            if auto_decision is not None
            else
            "Semantic IR 尚未完成参数审查："
            + "; ".join(readiness_errors)
            + "。请在 semantic 阶段补齐，不得由 trace 阶段默认。"
            if readiness_errors
            else "自动批量评审通过。"
        )
        auto_trace_feedback = (
            auto_decision.trace_feedback
            if auto_decision is not None
            else
            (
                trace_coverage_feedback(state.trace_coverage)
                or state.trace_coverage.reason
            )
            if state.trace_coverage.status != "complete"
            else "自动批量评审通过。"
        )
        lines.extend(
            [
                f"## {state.requirement_id}",
                "",
                "### Requirement",
                "```text",
                state.requirement,
                "```",
                "",
                "### Semantic Structure",
                "```text",
                _format_semantic(state.semantic),
                "```",
                "",
                "### Finite Traces",
                _format_traces(state.traces),
                "",
                "### Trace Coverage",
                "```text",
                _format_trace_coverage(state.trace_coverage),
                "```",
                "",
                "### Review Fields",
                f"semantic_status: {semantic_status}",
                f"trace_status: {trace_status}",
                "trace_issue_type: content",
                "target_trace_length:",
                "",
                "### Semantic Feedback",
                SEMANTIC_FEEDBACK_START,
                "" if not auto_approve else auto_semantic_feedback,
                SEMANTIC_FEEDBACK_END,
                "",
                "### Trace Feedback",
                TRACE_FEEDBACK_START,
                "" if not auto_approve else auto_trace_feedback,
                TRACE_FEEDBACK_END,
                "",
            ]
        )

    return "\n".join(lines)


def _initial_status(already_approved: bool, auto_approve: bool) -> str:
    if already_approved or auto_approve:
        return "pass"
    return "pending"


def _format_semantic(semantic: SemanticStructure | None) -> str:
    if semantic is None:
        return "<not generated>"
    lines = [
        f"ir_semantic_status: {semantic.semantic_status}",
        f"trigger: {semantic.trigger}",
        f"response: {semantic.response}",
        f"pattern: {semantic.pattern}",
        f"pattern_status: {semantic.pattern_status}",
        f"scope_keyword: {semantic.scope_keyword}",
        f"temporal_keyword: {semantic.temporal_keyword}",
        f"trace_domain: {semantic.trace_domain}",
        f"persistent_rag_allowed: {semantic.persistent_rag_allowed}",
        f"confidence: {semantic.confidence:.2f}",
        f"explanation: {semantic.explanation}",
    ]
    if semantic.faithful_ir is not None:
        lines.append(f"ir_version: {semantic.faithful_ir.ir_version}")
        readiness_errors = trace_generation_readiness_errors(semantic.faithful_ir)
        if readiness_errors:
            lines.append("trace_readiness_errors:")
            lines.extend(f"- {error}" for error in readiness_errors)
        unresolved = tuple(
            ambiguity
            for ambiguity in semantic.faithful_ir.ambiguities
            if not ambiguity.resolved
        )
        if unresolved:
            lines.append("unresolved_ir_ambiguities:")
            lines.extend(
                f"- {item.ambiguity_id}: {item.description}"
                for item in unresolved
            )
    if semantic.original_pattern:
        lines.append(f"original_pattern: {semantic.original_pattern}")
    if semantic.reference_only:
        lines.append(f"domain_warning: {semantic.domain_warning}")
        lines.append(
            "domain_evidence: "
            + (
                ", ".join(semantic.domain_evidence)
                if semantic.domain_evidence
                else "none"
            )
        )
        lines.append(f"finite_rewrite: {semantic.finite_rewrite}")
    if semantic.aps:
        lines.append("aps: " + ", ".join(semantic.aps))
    if semantic.formula_text:
        lines.append(f"formula_text: {semantic.formula_text}")
    if semantic.components:
        lines.append("components:")
        lines.extend(f"- {key}: {value}" for key, value in semantic.components)
    if semantic.rag_context:
        lines.append("rag_context:")
        lines.extend(f"- {context}" for context in semantic.rag_context)
    return "\n".join(lines)


def _format_traces(traces: list[LabeledTrace]) -> str:
    if not traces:
        return "未生成有限轨迹。"

    lines: list[str] = []
    for trace in traces:
        lines.extend(
            [
                f"#### {trace.name}",
                "",
                f"- kind: {trace.kind}",
                f"- expected: {format_truth(trace.expected)}",
                f"- adjudicated_label: {trace.adjudicated_label}",
                f"- confidence: {trace.confidence or 'unspecified'}",
                f"- label_source: {trace.label_source}",
                f"- query_role: {trace.query_role or 'legacy'}",
                f"- semantic_label: {trace.semantics.semantic_label or 'unclassified'}",
                (
                    "- roles: "
                    + (", ".join(trace.semantics.roles) or "none")
                ),
                (
                    "- confusion_targets: "
                    + (", ".join(trace.semantics.confusion_targets) or "none")
                ),
                f"- reason: {trace.reason}",
                "- beginner_view:",
                "",
                "```text",
                trace.beginner_view,
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _format_trace_coverage(report: TraceCoverageReport) -> str:
    lines = [
        f"status: {report.status}",
        f"count: {report.actual_count}/{report.required_count}",
        f"positive: {report.positive_count}",
        f"negative: {report.negative_count}",
        f"boundary: {report.boundary_count}",
        f"distinct_boundary: {report.distinct_boundary_count}",
        f"retry_count: {report.retry_count}",
        f"executable_semantics: {report.executable_semantics}",
        f"quota_complete: {report.quota_complete}",
        f"semantic_complete: {report.semantic_complete}",
        (
            "covered_roles: "
            + (", ".join(report.covered_roles) or "none")
        ),
        (
            "missing_roles: "
            + (", ".join(report.missing_roles) or "none")
        ),
        (
            "covered_confusions: "
            + (", ".join(report.covered_confusions) or "none")
        ),
        (
            "missing_confusions: "
            + (", ".join(report.missing_confusions) or "none")
        ),
        f"validated_trace_count: {report.validated_trace_count}",
    ]
    if report.coverage_debt:
        lines.append("coverage_debt:")
        lines.extend(f"- {item}" for item in report.coverage_debt)
    if report.reason:
        lines.append(f"reason: {report.reason}")
    return "\n".join(lines)


def _read_field(section: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\s*:\s*(.*?)\s*$", section, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _read_optional_int(section: str, name: str) -> int | None:
    raw_value = _read_field(section, name)
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value >= 1 else None


def _extract_between(section: str, start_marker: str, end_marker: str) -> str:
    start = section.find(start_marker)
    end = section.find(end_marker)
    if start == -1 or end == -1 or end < start:
        return ""
    return section[start + len(start_marker) : end].strip()


def _normalize_status(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"pass", "passed", "ok", "yes", "y", "true", "通过"}:
        return "pass"
    if normalized in {"fail", "failed", "no", "n", "false", "不通过", "失败"}:
        return "fail"
    return "pending"


def _normalize_issue_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"length", "len", "l", "长度", "轨迹长度"}:
        return "length"
    return "content"


def _feedback_or_default(feedback: str, default: str) -> str:
    return feedback.strip() if feedback.strip() else default


def _build_semantic_batch_error(
    *,
    state: _BatchItemState,
    comments: str,
    round_index: int,
) -> StageErrorRecord:
    return StageErrorRecord(
        error_id=f"{state.requirement_id}-semantic-{round_index}-batch-review",
        stage="semantic",
        round_index=round_index,
        artifact=json.dumps(asdict(state.semantic), ensure_ascii=False, indent=2),
        error_type="content",
        source="batch_markdown_review",
        evidence=(comments,),
        suggestion=(
            "下一轮 NLUAgent 应根据批量评审 Markdown 中的意见修正 trigger、"
            "response、pattern 或时序关键词。"
        ),
        comments=comments,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


def _build_trace_batch_error(
    *,
    state: _BatchItemState,
    comments: str,
    round_index: int,
) -> StageErrorRecord:
    return StageErrorRecord(
        error_id=f"{state.requirement_id}-traces-{round_index}-batch-review",
        stage="traces",
        round_index=round_index,
        artifact=json.dumps(
            [_trace_to_dict(trace) for trace in state.traces],
            ensure_ascii=False,
            indent=2,
        ),
        error_type="content",
        source="batch_markdown_review",
        evidence=_trace_error_evidence(state.traces),
        suggestion=(
            "下一轮 TraceGenerationAgent 应结合批量评审 Markdown 中的意见"
            "重新生成正轨迹、反轨迹和边界轨迹。"
        ),
        comments=comments,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


def _trace_error_evidence(traces: list[LabeledTrace]) -> tuple[str, ...]:
    if not traces:
        return ("未生成任何有限轨迹。",)
    return tuple(
        f"{trace.kind} {trace.name}: length={len(trace.trace)}, "
        f"expected={format_truth(trace.expected)}, reason={trace.reason}"
        for trace in traces
    )


def _trace_to_dict(trace: LabeledTrace) -> dict:
    return {
        "name": trace.name,
        "kind": trace.kind,
        "trace": trace_to_serializable(trace.trace),
        "beginner_view": trace.beginner_view,
        "expected": trace.expected,
        "adjudicated_label": trace.adjudicated_label,
        "confidence": trace.confidence,
        "rationale": trace.rationale,
        "label_source": trace.label_source,
        "role": trace.role,
        "query_role": trace.query_role,
        "target_ambiguity": trace.target_ambiguity,
        "target_ambiguity_axis_ids": list(trace.target_ambiguity_axis_ids),
        "query_constraints": list(trace.query_constraints),
        "data_split": trace.data_split,
        "formula_source": trace.formula_source,
        "formula_visibility": list(trace.formula_visibility),
        "annotation_locked": trace.annotation_locked,
        "allowed_rag_stages": list(trace.allowed_rag_stages),
        "trace_id": trace.trace_id,
        "covered_ir_node_ids": list(trace.covered_ir_node_ids),
        "obligation_ids": list(trace.obligation_ids),
        "trace_role": trace.trace_role,
        "boundary_role": trace.boundary_role,
        "generation_source": trace.generation_source,
        "ir_version": trace.ir_version,
        "nl_verdict": trace.nl_verdict,
        "temporal_simulation_summary": trace.temporal_simulation_summary,
        "review_record_id": trace.review_record_id,
        "supported_text_spans": list(trace.supported_text_spans),
        "reason": trace.reason,
        "semantics": trace_semantics_to_dict(trace.semantics),
    }


def _save_json_record(record: dict, records_dir: str, prefix: str) -> Path:
    path = Path(records_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    requirement_id = record.get("requirement_id", "batch")
    round_label = record.get("round", "final")
    filename = f"{prefix}_{requirement_id}_round_{round_label}_{timestamp}.json"
    record_path = path / filename
    record["record_path"] = str(record_path)
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record_path
