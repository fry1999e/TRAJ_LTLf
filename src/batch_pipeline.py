"""Batch entrypoint for running Traj-LTLf requirements from a Markdown file."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from contract_versions import BATCH_OUTPUT_MANIFEST_SCHEMA_VERSION

from agents.base_agent import JSONLLMClient
from ap_binding_context import (
    APBindingContext,
    build_ap_binding_context,
    text_only_ap_binding_context,
)
from approval_protocol import (
    final_approval_from_independent_decision,
    request_fresh_independent_approval,
)
from artifact_audit import (
    ArtifactFingerprintSet,
    build_artifact_fingerprint_set,
    compare_artifact_fingerprint_sets,
)
from feedback_patch import (
    feedback_patch_from_mapping,
    propose_feedback_patch,
    review_feedback_patch,
)
from error_diagnosis import build_error_diagnosis
from batch_review import run_batch_semantic_trace_review
from checkpointing import (
    degraded_result_from_checkpoint,
    load_pipeline_result_checkpoint,
    save_failed_checkpoint,
    save_needs_review_checkpoint,
    save_pipeline_result_checkpoint,
    update_checkpoint_classification,
    update_pipeline_expert_review_status,
)
from llm import build_judge_client_from_env, build_llm_client_from_env
from experiment_modes import (
    is_sample_reference_formula,
    requires_judge_client,
    resolve_experiment_mode,
)
from ltlf_parser import Formula, parse_formula
from pipeline import (
    PipelineResult,
    build_degraded_result_record,
    format_degraded_result,
    format_degraded_summary,
    format_pipeline_result,
    format_pipeline_summary,
    run_traj_ltlf_pipeline_from_reviewed_inputs,
)
from rag.example_memory import collect_approved_example_chunks
from rag.example_memory import (
    build_guided_correction_chunks_from_review_records,
    record_guided_warmup_process,
)
from rag.knowledge_base import KnowledgeChunk
from rag.persistent_memory import (
    DEFAULT_APPROVED_MEMORY_FILE,
    merge_approved_memory,
)
from rag.memory_store import load_persistent_rag
from pattern_registry import register_pattern_specs
from requirement_loader import RequirementRecord
from ap_identity import offline_formula_assisted_profile
from plain_reading import assess_plain_reading_ir_readiness, decision_fingerprint
from surface_segmentation import source_hash
from terminal_progress import TerminalProgress
from warmup_loader import WarmupExample
from schemas import (
    APRepresentationProfile,
    BatchRunOutcome,
    BlockedFailureRecord,
    DeclaredAPCatalog,
    DegradedResultRecord,
    ExperimentConfig,
    FormulaConsistencyEvaluation,
    FeedbackRerunRequest,
    PreIRDisposition,
    ReviewPendingRecord,
    TraceEvidencePolicy,
)


FINAL_CLASSIFICATIONS = {
    "complete",
    "complete_with_corrections",
    "needs_review",
    "degraded_unproven",
    "degraded_proven",
    "blocked_error",
    "rejected",
}


@dataclass(frozen=True)
class BatchPipelineItemResult:
    """One finished pipeline result with its stable batch requirement ID."""

    requirement_id: str
    result: PipelineResult | None
    report_text: str = ""
    summary_text: str = ""
    review_records: list[dict] = field(default_factory=list)
    requirement: str = ""
    classification: str = ""
    degraded_record: DegradedResultRecord | None = None
    error: dict[str, Any] | None = None
    trace_evidence_policy: TraceEvidencePolicy = TraceEvidencePolicy()
    experimental_only: bool = False


class DuplicateBatchResultError(RuntimeError):
    """Raised when one requirement ID has multiple batch result records."""


class IncompleteBatchError(RuntimeError):
    """Raised after partial outputs are written when strict completeness is on."""

    def __init__(self, outcome: BatchRunOutcome) -> None:
        self.outcome = outcome
        super().__init__(
            "batch is incomplete; missing final results: "
            + ", ".join(outcome.missing_requirement_ids)
        )


_RERUN_STAGES = {
    "pipeline",
    "segmentation",
    "plain_reading",
    "semantic",
    "trace",
    "skeleton",
    "completion",
    "selection",
}


def _warmup_ap_binding_context(example: WarmupExample) -> APBindingContext:
    semantic = example.expected_semantic
    catalog = (
        semantic.faithful_ir.ap_names
        if semantic is not None and semantic.faithful_ir is not None
        else semantic.aps
        if semantic is not None
        else ()
    )
    if not catalog:
        return text_only_ap_binding_context(
            example.requirement,
            requirement_id=example.requirement_id,
        )
    return build_ap_binding_context(
        requirement_id=example.requirement_id,
        source_text=example.requirement,
        representation_profile=offline_formula_assisted_profile(
            profile_id=(
                "AP-PROFILE-warmup-offline-"
                + example.requirement_id.casefold()
            ),
            provenance="train_only_warmup_offline_catalog",
        ),
        trusted_offline_catalog=catalog,
        provenance=("train_only_warmup_offline_catalog",),
    )


def _load_review_feedback(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    feedback_path = Path(path)
    if not feedback_path.is_file():
        raise FileNotFoundError(f"review feedback 文件不存在: {feedback_path}")
    payload = json.loads(feedback_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("review feedback 必须使用 schema_version=2。")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("review feedback 必须是 items 数组或顶层数组。")

    packets: dict[str, dict[str, Any]] = {}
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            raise ValueError(f"review feedback items[{index}] 必须是对象。")
        if str(raw_item.get("action", "")).strip() != "rerun_with_feedback":
            continue
        requirement_id = str(raw_item.get("requirement_id", "")).strip()
        if not requirement_id:
            raise ValueError(f"review feedback items[{index}] 缺少 requirement_id。")
        required_v2_fields = {
            "owner_stage",
            "restart_stage",
            "reason_code",
            "source_ir_version",
            "source_artifact_fingerprints",
            "affected_ir_node_ids",
            "dispute_ids",
            "challenge_ids",
            "reviewed_nl_evidence",
            "expected_invalidation_set",
            "restart_prerequisites",
            "fresh_approval_requirement",
        }
        missing_fields = sorted(required_v2_fields - set(raw_item))
        if missing_fields:
            raise ValueError(
                f"review feedback items[{index}] 缺少 v2 字段: "
                + ", ".join(missing_fields)
            )
        for field_name in (
            "affected_ir_node_ids",
            "dispute_ids",
            "challenge_ids",
            "reviewed_nl_evidence",
            "expected_invalidation_set",
            "restart_prerequisites",
        ):
            field_value = raw_item.get(field_name)
            if not isinstance(field_value, list) or any(
                not isinstance(value, str) for value in field_value
            ):
                raise ValueError(
                    f"review feedback items[{index}] 的 {field_name} 必须是字符串数组。"
                )
        if not isinstance(raw_item.get("source_artifact_fingerprints"), dict):
            raise ValueError(
                f"review feedback items[{index}] 的 source_artifact_fingerprints 必须是对象。"
            )
        if "source_candidates" in raw_item or "candidate_formula" in raw_item:
            raise ValueError("review feedback v2 禁止携带候选公式或候选列表。")
        restart_stage = str(raw_item.get("restart_stage", "")).strip().lower()
        if restart_stage not in _RERUN_STAGES:
            raise ValueError(
                f"review feedback items[{index}] 的 restart_stage 无效: {restart_stage}"
            )
        packet = dict(raw_item)
        packet["restart_stage"] = restart_stage
        output_policy = str(
            packet.get("output_policy", "fresh_independent_approval")
        ).strip()
        if output_policy not in {
            "provisional_only",
            "fresh_independent_approval",
        }:
            raise ValueError(
                f"review feedback items[{index}] 的 output_policy 无效。"
            )
        affected = packet.get("affected_ir_node_ids", [])
        if not isinstance(affected, list) or any(
            not isinstance(node_id, str) for node_id in affected
        ):
            raise ValueError(
                f"review feedback items[{index}] 的 affected_ir_node_ids 必须是字符串数组。"
            )
        packet["output_policy"] = output_policy
        packet["affected_ir_node_ids"] = affected
        packet["ir_global_problem"] = bool(
            packet.get("ir_global_problem", not affected)
        )
        if (
            restart_stage not in {"segmentation", "plain_reading"}
            and not affected
            and not packet["ir_global_problem"]
        ):
            raise ValueError(
                f"review feedback items[{index}] 必须绑定 IR 节点或标记 ir_global_problem。"
            )
        feedback_patch_payload = packet.get("feedback_patch")
        if feedback_patch_payload is not None:
            if not isinstance(feedback_patch_payload, dict):
                raise ValueError(
                    f"review feedback items[{index}] 的 feedback_patch 必须是对象。"
                )
            feedback_patch_payload = asdict(
                feedback_patch_from_mapping(feedback_patch_payload)
            )
        request = FeedbackRerunRequest(
            requirement_id=requirement_id,
            owner_stage=str(packet.get("owner_stage", "")),
            restart_stage=restart_stage,
            reason_code=str(packet.get("reason_code", "")),
            source_ir_version=str(packet.get("source_ir_version", "")),
            source_artifact_fingerprints=(
                dict(packet.get("source_artifact_fingerprints", {}))
                if isinstance(packet.get("source_artifact_fingerprints"), dict)
                else None
            ),
            feedback=str(packet.get("feedback", "")),
            affected_ir_node_ids=tuple(affected),
            dispute_ids=tuple(str(item) for item in packet.get("dispute_ids", ())),
            challenge_ids=tuple(str(item) for item in packet.get("challenge_ids", ())),
            reviewed_nl_evidence=tuple(
                str(item) for item in packet.get("reviewed_nl_evidence", ())
            ),
            expected_invalidation_set=tuple(
                str(item) for item in packet.get("expected_invalidation_set", ())
            ),
            restart_prerequisites=tuple(
                str(item) for item in packet.get("restart_prerequisites", ())
            ),
            fresh_approval_requirement=str(
                packet.get("fresh_approval_requirement", "")
            ),
            constraints=tuple(str(item) for item in packet.get("constraints", ())),
            output_policy=output_policy,
            ir_global_problem=packet["ir_global_problem"],
            diagnosis_id=str(packet.get("diagnosis_id", "")),
            source_run_id=str(packet.get("source_run_id", "")),
            fresh_judge_required=bool(packet.get("fresh_judge_required", True)),
            feedback_patch=feedback_patch_payload,
            frozen_upstream=bool(packet.get("frozen_upstream", False)),
        )
        normalized_packet = asdict(request)
        for audit_key in (
            "status",
            "source_error_ids",
            "prior_independence_ids",
        ):
            if audit_key in packet:
                normalized_packet[audit_key] = packet[audit_key]
        packets[requirement_id] = normalized_packet
    return packets


def _pipeline_start_stage(packet: dict[str, Any] | None) -> str:
    if not packet:
        return "skeleton"
    stage = str(packet.get("restart_stage", "skeleton")).strip().lower()
    return stage if stage in {"completion", "selection"} else "skeleton"


def _add_review_feedback_chunk(item: object, packet: dict[str, Any]) -> None:
    retriever = getattr(item, "retriever", None)
    if retriever is None:
        return
    requirement_id = str(getattr(item, "requirement_id", ""))
    semantic = getattr(item, "semantic", None)
    faithful_ir = getattr(semantic, "faithful_ir", None)
    affected = tuple(packet.get("affected_ir_node_ids", ()))
    if faithful_ir is not None:
        unknown = set(affected) - set(faithful_ir.node_map)
        if unknown:
            raise ValueError(
                "review feedback references unknown IR nodes: "
                + ", ".join(sorted(unknown))
            )
        previous = str(packet.get("previous_ir_version", ""))
        if previous and previous != faithful_ir.ir_version:
            raise ValueError(
                "review feedback previous_ir_version does not match current IR"
            )
    chunk_id = f"runtime-review-feedback-{requirement_id}"
    if any(getattr(chunk, "chunk_id", "") == chunk_id for chunk in retriever.chunks):
        return
    constraints = [
        str(value).strip()
        for value in packet.get("constraints", [])
        if str(value).strip()
    ]
    patch = packet.get("feedback_patch")
    patch_lines: list[str] = []
    if isinstance(patch, dict):
        patch_lines = [
            f"feedback_patch_id: {patch.get('patch_id', '')}",
            f"feedback_patch_review_status: {patch.get('review_status', '')}",
            "reopened_ir_node_ids: "
            + ", ".join(patch.get("reopened_ir_node_ids", ())),
            "new_ambiguity_debt: "
            + ", ".join(patch.get("new_ambiguity_debt", ())),
            "new_trace_obligation_requests: "
            + ", ".join(patch.get("new_trace_obligation_requests", ())),
            "stale_artifact_declarations: "
            + ", ".join(patch.get("stale_artifact_declarations", ())),
        ]
    retriever.add_chunk(
        KnowledgeChunk(
            chunk_id=chunk_id,
            title=f"Review feedback for {requirement_id}",
            text="\n".join(
                [
                    f"requirement: {getattr(item, 'requirement', '')}",
                    f"start_stage: {packet.get('start_stage', '')}",
                    f"feedback: {packet.get('feedback', '')}",
                    f"affected_ir_node_ids: {', '.join(packet.get('affected_ir_node_ids', []))}",
                    f"output_policy: {packet.get('output_policy', 'fresh_independent_approval')}",
                    *patch_lines,
                    *(f"constraint: {constraint}" for constraint in constraints),
                ]
            ),
            source="runtime:review-feedback",
        )
    )


def run_batch_traj_ltlf_pipeline(
    requirements: list[str] | list[RequirementRecord],
    *,
    requirement_records: list[RequirementRecord] | None = None,
    requirement_ids: list[str] | None = None,
    strict_boundary: bool = True,
    review_mode: str = "manual",
    max_review_rounds: int = 3,
    min_passed_skeletons: int = 5,
    records_dir: str = "review_records",
    llm_mode: str | None = "rules",
    llm_client: JSONLLMClient | None = None,
    warmup_size: int = 0,
    warmup_examples: list[WarmupExample] | None = None,
    warmup_only: bool = False,
    approved_memory_file: str | None = DEFAULT_APPROVED_MEMORY_FILE,
    checkpoint_dir: str | None = None,
    partial_result_path: str | None = None,
    partial_summary_path: str | None = None,
    classified_output_dir: str | None = None,
    progress: TerminalProgress | None = None,
    experiment_mode: str | None = None,
    data_split: str = "test",
    reference_formula: Formula | str | None = None,
    learning_mode: str | None = None,
    offline_validation: str = "none",
    judge_client: JSONLLMClient | None = None,
    final_unique_results: bool | None = None,
    require_complete_batch: bool = False,
    review_feedback_file: str | None = None,
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
    delivery_policy: str | None = None,
) -> list[BatchPipelineItemResult]:
    """Run the batch review front half, then the full pipeline for each item."""

    from delivery_policy import resolve_delivery_policy
    mode = resolve_experiment_mode(review_mode, experiment_mode)
    policy = resolve_delivery_policy(delivery_policy, review_mode, mode)
    if warmup_only or warmup_examples or warmup_size:
        if delivery_policy == "always_output":
            raise ValueError("warmup requires strict delivery_policy")
        policy = "strict"
    raw_requirements = list(requirement_records or requirements)
    record_inputs = bool(requirement_records) or (
        bool(raw_requirements)
        and all(isinstance(item, RequirementRecord) for item in raw_requirements)
    )
    if record_inputs:
        records = [item for item in raw_requirements if isinstance(item, RequirementRecord)]
        record_ids = [item.requirement_id for item in records]
        if requirement_ids is not None and list(requirement_ids) != record_ids:
            raise ValueError("requirement_ids disagree with RequirementRecord ids")
        requirement_ids = record_ids
        requirements = [item.text for item in records]
        ap_representation_profiles_by_id = {
            **{item.requirement_id: item.ap_representation_profile for item in records},
            **dict(ap_representation_profiles_by_id or {}),
        }
        declared_ap_catalogs_by_id = {
            **{
                item.requirement_id: item.declared_ap_catalog
                for item in records
                if item.declared_ap_catalog is not None
            },
            **dict(declared_ap_catalogs_by_id or {}),
        }
        ap_binding_contexts_by_id = {
            **{
                item.requirement_id: build_ap_binding_context(
                    requirement_id=item.requirement_id,
                    source_text=item.text,
                    representation_profile=item.ap_representation_profile,
                    declared_catalog=item.declared_ap_catalog,
                    provenance=("RequirementRecord",),
                )
                for item in records
            },
            **dict(ap_binding_contexts_by_id or {}),
        }
        effective_semantic_sources_by_id = {
            item.requirement_id: item.effective_semantic_source
            for item in records
            if item.effective_semantic_source is not None
        }
    else:
        requirements = [str(item) for item in raw_requirements]
        effective_semantic_sources_by_id = {}

    if requirement_ids is not None and len(requirement_ids) != len(requirements):
        raise ValueError("requirement_ids must align one-to-one with requirements")

    effective_mode = resolve_experiment_mode(review_mode, experiment_mode)
    uses_sample_reference = is_sample_reference_formula(reference_formula)
    if learning_mode is None or not str(learning_mode).strip():
        learning_mode = (
            "legacy_formula_oracle"
            if effective_mode == "formula_oracle" or uses_sample_reference
            else "stage_guided"
        )
    if learning_mode not in {
        "stage_guided",
        "offline_formula_validation",
        "legacy_formula_oracle",
    }:
        raise ValueError("Unsupported learning_mode: " + learning_mode)
    if offline_validation not in {"none", "final_formula"}:
        raise ValueError("Unsupported offline_validation: " + offline_validation)
    review_feedback = _load_review_feedback(review_feedback_file)
    ap_representation_profiles_by_id = ap_representation_profiles_by_id or {}
    declared_ap_catalogs_by_id = declared_ap_catalogs_by_id or {}
    ap_binding_contexts_by_id = ap_binding_contexts_by_id or {}
    force_rerun_stages = {
        requirement_id: str(packet.get("start_stage", "")).strip().lower()
        for requirement_id, packet in review_feedback.items()
    }
    if final_unique_results is None:
        final_unique_results = _final_unique_results_enabled(
            review_mode=review_mode,
            experiment_mode=effective_mode,
            llm_mode=llm_mode,
        )
    if uses_sample_reference and learning_mode != "legacy_formula_oracle":
        raise ValueError(
            "--reference-formula sample 只允许 legacy_formula_oracle 兼容学习模式。"
        )
    if effective_mode == "formula_oracle" and learning_mode != "legacy_formula_oracle":
        raise ValueError(
            "formula_oracle 只作为 legacy_formula_oracle warmup 兼容路径存在。"
        )
    if learning_mode == "legacy_formula_oracle" and effective_mode != "formula_oracle":
        raise ValueError(
            "legacy_formula_oracle requires experiment_mode=formula_oracle."
        )
    if effective_mode == "formula_oracle" and not uses_sample_reference:
        raise ValueError(
            "formula_oracle 只允许 --warmup-file --warmup-only "
            "--reference-formula sample data_split=train。"
        )
    if uses_sample_reference:
        if effective_mode != "formula_oracle":
            raise ValueError(
                "--reference-formula sample 只允许和 formula_oracle 模式一起使用。"
            )
        if not warmup_examples:
            raise ValueError(
                "--reference-formula sample 需要提供 warmup_examples。"
            )
        if not warmup_only:
            raise ValueError(
                "--reference-formula sample 当前只支持 --warmup-only 批量样本运行。"
            )
        if data_split != "train":
            raise ValueError("formula_oracle warmup learning requires data_split=train.")
        parsed_reference = None
        experiment_config = None
    else:
        parsed_reference = (
            parse_formula(reference_formula)
            if isinstance(reference_formula, str)
            else reference_formula
        )
        experiment_config = ExperimentConfig(
            mode=effective_mode,
            data_split=data_split,
            reference_formula=parsed_reference,
            test_time_memory_frozen=True,
        )
    warmup_experiment_config = ExperimentConfig(
        mode="human",
        data_split="train",
    )
    if llm_client is None:
        message = (
            "Initializing LLM API client"
            if (llm_mode or "").lower() == "api"
            else "Initializing local rules mode"
        )
        if progress is None:
            llm_client = build_llm_client_from_env(llm_mode)
        else:
            with progress.task(message):
                llm_client = build_llm_client_from_env(llm_mode)
            progress.ok(message + " ready")
    if (
        judge_client is None
        and (
            requires_judge_client(review_mode, effective_mode)
            or segmentation_review == "judge"
            or plain_reading_review == "judge"
        )
    ):
        message = "Initializing Judge LLM client"
        if progress is None:
            judge_client = build_judge_client_from_env()
        else:
            with progress.task(message):
                judge_client = build_judge_client_from_env()
            progress.ok(message + " ready")

    persistent_chunks = []
    persistent_pattern_specs = []
    if approved_memory_file:
        persistent_chunks, persistent_pattern_specs = load_persistent_rag(
            approved_memory_file
        )
        if progress is not None and (persistent_chunks or persistent_pattern_specs):
            progress.info(
                "Loaded persistent RAG memory: "
                f"{len(persistent_chunks)} chunk(s), "
                f"{len(persistent_pattern_specs)} pattern spec(s)"
            )

    pattern_specs = [
        example.pattern_spec
        for example in (warmup_examples or [])
        if example.pattern_spec is not None
    ]
    register_pattern_specs(persistent_pattern_specs)
    approved_initial_chunks = list(persistent_chunks)
    explicit_warmup_requirements = [
        example.requirement for example in (warmup_examples or [])
    ]

    if warmup_only and not explicit_warmup_requirements:
        raise ValueError("warmup-only 运行至少需要一个 warmup 样例。")

    if explicit_warmup_requirements:
        expected_trace_examples = {
            example.requirement_id: list(
                example.gold_labeled_trace_evidence or example.expected_traces
            )
            for example in (warmup_examples or [])
            if example.gold_labeled_trace_evidence or example.expected_traces
        }
        expected_semantic_examples = {
            example.requirement_id: example.expected_semantic
            for example in (warmup_examples or [])
            if example.expected_semantic is not None
        }
        guided_context_examples = {
            example.requirement_id: example.guided_context
            for example in (warmup_examples or [])
            if example.guided_context is not None
        }
        warmup_ap_binding_contexts = {
            example.requirement_id: _warmup_ap_binding_context(example)
            for example in (warmup_examples or [])
            if example.expected_semantic is not None
        }
        missing_curated_traces = [
            example.requirement_id
            for example in (warmup_examples or [])
            if not example.expected_traces
        ]
        missing_frozen_trace_gold = [
            example.requirement_id
            for example in (warmup_examples or [])
            if not example.gold_nl_temporal_reviews
            or not example.gold_labeled_trace_evidence
        ]
        missing_curated_semantics = [
            example.requirement_id
            for example in (warmup_examples or [])
            if example.expected_semantic is None
        ]
        missing_pattern_specs = [
            example.requirement_id
            for example in (warmup_examples or [])
            if example.pattern_spec is None
        ]
        if missing_curated_semantics:
            raise ValueError(
                "warmup 文件中的样例必须包含 Expected Semantic Structure，"
                "它只在自主首轮失败后的第二次尝试中作为教师信号。缺失样例: "
                + ", ".join(missing_curated_semantics)
            )
        if missing_curated_traces:
            raise ValueError(
                "warmup 文件中的样例必须包含 Expected Finite Traces，"
                "它只在自主首轮失败后的第二次尝试中作为教师信号。缺失样例: "
                + ", ".join(missing_curated_traces)
            )
        if missing_frozen_trace_gold:
            raise ValueError(
                "warmup 自动裁决要求每个 train-only 样例包含冻结的 Gold "
                "NLTemporalSimulationReview 和 Gold LabeledTraceEvidenceSet；"
                "运行时不允许退回启发式 Judge 或公式标签。缺失样例: "
                + ", ".join(missing_frozen_trace_gold)
            )
        if missing_pattern_specs:
            raise ValueError(
                "warmup 文件中的样例必须包含 approved_formula 以学习 pattern handler。"
                "缺失样例: "
                + ", ".join(missing_pattern_specs)
            )
        sample_reference_configs = (
            _sample_reference_configs(warmup_examples)
            if uses_sample_reference
            else {}
        )
        if progress is not None:
            progress.stage("Warmup semantic/trace review")
            progress.info(f"Warmup examples: {len(explicit_warmup_requirements)}")
        warmup_items = run_batch_semantic_trace_review(
            requirements=explicit_warmup_requirements,
            llm_client=llm_client,
            review_mode=review_mode,
            max_rounds=1,
            records_dir=records_dir,
            initial_approved_chunks=approved_initial_chunks,
            persistent_memory_file=None,
            requirement_id_prefix="WARM",
            expected_semantics_by_id=expected_semantic_examples,
            expected_traces_by_id=expected_trace_examples,
            guided_contexts_by_id=guided_context_examples,
            experiment_configs_by_id=sample_reference_configs,
            checkpoint_dir=checkpoint_dir,
            force_rerun_stages=force_rerun_stages,
            rerun_feedback=review_feedback,
            progress=progress,
            experiment_config=(
                warmup_experiment_config if not uses_sample_reference else None
            ),
            judge_client=None,
            segmentation_review=segmentation_review,
            plain_reading_review=plain_reading_review,
            max_segmentation_retries=max_segmentation_retries,
            max_surface_review_rounds=max_surface_review_rounds,
            max_plain_reading_rounds=max_plain_reading_rounds,
            max_plain_reading_revision_rounds=max_plain_reading_revision_rounds,
            segmentation_prompt_version=segmentation_prompt_version,
            plain_reading_prompt_version=plain_reading_prompt_version,
            plain_reading_semantic_fallback=plain_reading_semantic_fallback,
            trace_evidence_policy=trace_evidence_policy,
            ap_representation_profiles_by_id=ap_representation_profiles_by_id,
            declared_ap_catalogs_by_id=declared_ap_catalogs_by_id,
            ap_binding_contexts_by_id=warmup_ap_binding_contexts,
            effective_semantic_sources_by_id=effective_semantic_sources_by_id,
        )
        register_pattern_specs(pattern_specs)
        if warmup_only:
            reviewed_items = warmup_items
        else:
            if progress is not None:
                progress.stage("Requirement semantic/trace review")
                progress.info(f"Main requirements: {len(requirements)}")
            remaining_items = run_batch_semantic_trace_review(
                requirements=requirements,
                requirement_ids=requirement_ids,
                llm_client=llm_client,
                review_mode=review_mode,
                max_rounds=max_review_rounds,
                records_dir=records_dir,
                initial_approved_chunks=persistent_chunks,
                persistent_memory_file=None,
                requirement_id_prefix="REQ",
                checkpoint_dir=checkpoint_dir,
                force_rerun_stages=force_rerun_stages,
                rerun_feedback=review_feedback,
                progress=progress,
                experiment_config=experiment_config,
                judge_client=judge_client,
                segmentation_review=segmentation_review,
                plain_reading_review=plain_reading_review,
                max_segmentation_retries=max_segmentation_retries,
                max_surface_review_rounds=max_surface_review_rounds,
                max_plain_reading_rounds=max_plain_reading_rounds,
                max_plain_reading_revision_rounds=max_plain_reading_revision_rounds,
                segmentation_prompt_version=segmentation_prompt_version,
                plain_reading_prompt_version=plain_reading_prompt_version,
                plain_reading_semantic_fallback=plain_reading_semantic_fallback,
                trace_evidence_policy=trace_evidence_policy,
                ap_representation_profiles_by_id=ap_representation_profiles_by_id,
                declared_ap_catalogs_by_id=declared_ap_catalogs_by_id,
                ap_binding_contexts_by_id=ap_binding_contexts_by_id,
                effective_semantic_sources_by_id=effective_semantic_sources_by_id,
            )
            reviewed_items = warmup_items + remaining_items
    elif warmup_size > 0 and warmup_size < len(requirements):
        if progress is not None:
            progress.stage("Warmup semantic/trace review")
            progress.info(f"Warmup requirements: {warmup_size}")
        warmup_items = run_batch_semantic_trace_review(
            requirements=requirements[:warmup_size],
            requirement_ids=(
                requirement_ids[:warmup_size] if requirement_ids is not None else None
            ),
            llm_client=llm_client,
            review_mode=review_mode,
            max_rounds=max_review_rounds,
            records_dir=records_dir,
            initial_approved_chunks=persistent_chunks,
            persistent_memory_file=None,
            requirement_id_prefix="WARM",
            checkpoint_dir=checkpoint_dir,
            force_rerun_stages=force_rerun_stages,
            rerun_feedback=review_feedback,
            progress=progress,
            experiment_config=warmup_experiment_config,
            judge_client=None,
            segmentation_review=segmentation_review,
            plain_reading_review=plain_reading_review,
            max_segmentation_retries=max_segmentation_retries,
            max_surface_review_rounds=max_surface_review_rounds,
            max_plain_reading_rounds=max_plain_reading_rounds,
            max_plain_reading_revision_rounds=max_plain_reading_revision_rounds,
            segmentation_prompt_version=segmentation_prompt_version,
            plain_reading_prompt_version=plain_reading_prompt_version,
            plain_reading_semantic_fallback=plain_reading_semantic_fallback,
            trace_evidence_policy=trace_evidence_policy,
            ap_representation_profiles_by_id=ap_representation_profiles_by_id,
            declared_ap_catalogs_by_id=declared_ap_catalogs_by_id,
            ap_binding_contexts_by_id=ap_binding_contexts_by_id,
            effective_semantic_sources_by_id=effective_semantic_sources_by_id,
        )
        approved_chunks = collect_approved_example_chunks(
            [item.retriever for item in warmup_items]
        )
        approved_chunks.extend(persistent_chunks)
        if progress is not None:
            progress.stage("Requirement semantic/trace review")
            progress.info(f"Main requirements: {len(requirements) - warmup_size}")
        remaining_items = run_batch_semantic_trace_review(
            requirements=requirements[warmup_size:],
            requirement_ids=(
                requirement_ids[warmup_size:] if requirement_ids is not None else None
            ),
            llm_client=llm_client,
            review_mode=review_mode,
            max_rounds=max_review_rounds,
            records_dir=records_dir,
            initial_approved_chunks=approved_chunks,
            persistent_memory_file=None,
            requirement_id_prefix="REQ",
            checkpoint_dir=checkpoint_dir,
            force_rerun_stages=force_rerun_stages,
            rerun_feedback=review_feedback,
            progress=progress,
            experiment_config=experiment_config,
            judge_client=judge_client,
            segmentation_review=segmentation_review,
            plain_reading_review=plain_reading_review,
            max_segmentation_retries=max_segmentation_retries,
            max_surface_review_rounds=max_surface_review_rounds,
            max_plain_reading_rounds=max_plain_reading_rounds,
            max_plain_reading_revision_rounds=max_plain_reading_revision_rounds,
            segmentation_prompt_version=segmentation_prompt_version,
            plain_reading_prompt_version=plain_reading_prompt_version,
            plain_reading_semantic_fallback=plain_reading_semantic_fallback,
            trace_evidence_policy=trace_evidence_policy,
            ap_representation_profiles_by_id=ap_representation_profiles_by_id,
            declared_ap_catalogs_by_id=declared_ap_catalogs_by_id,
            ap_binding_contexts_by_id=ap_binding_contexts_by_id,
            effective_semantic_sources_by_id=effective_semantic_sources_by_id,
        )
        reviewed_items = warmup_items + remaining_items
    else:
        if progress is not None:
            progress.stage("Requirement semantic/trace review")
            progress.info(f"Main requirements: {len(requirements)}")
        reviewed_items = run_batch_semantic_trace_review(
            requirements=requirements,
            requirement_ids=requirement_ids,
            llm_client=llm_client,
            review_mode=review_mode,
            max_rounds=max_review_rounds,
            records_dir=records_dir,
            initial_approved_chunks=persistent_chunks,
            persistent_memory_file=None,
            checkpoint_dir=checkpoint_dir,
            force_rerun_stages=force_rerun_stages,
            rerun_feedback=review_feedback,
            progress=progress,
            experiment_config=experiment_config,
            judge_client=judge_client,
            segmentation_review=segmentation_review,
            plain_reading_review=plain_reading_review,
            max_segmentation_retries=max_segmentation_retries,
            max_surface_review_rounds=max_surface_review_rounds,
            max_plain_reading_rounds=max_plain_reading_rounds,
            max_plain_reading_revision_rounds=max_plain_reading_revision_rounds,
            segmentation_prompt_version=segmentation_prompt_version,
            plain_reading_prompt_version=plain_reading_prompt_version,
            plain_reading_semantic_fallback=plain_reading_semantic_fallback,
            trace_evidence_policy=trace_evidence_policy,
            ap_representation_profiles_by_id=ap_representation_profiles_by_id,
            declared_ap_catalogs_by_id=declared_ap_catalogs_by_id,
            ap_binding_contexts_by_id=ap_binding_contexts_by_id,
            effective_semantic_sources_by_id=effective_semantic_sources_by_id,
        )

    results: list[BatchPipelineItemResult] = []
    if progress is not None:
        progress.stage("LTLf pipeline")
        progress.info(f"Reviewed items entering pipeline: {len(reviewed_items)}")
    for index, item in enumerate(reviewed_items, start=1):
        progress_label = f"{item.requirement_id} {index}/{len(reviewed_items)}"
        if progress is not None:
            progress.info(f"{progress_label} pipeline started")
        typed_disposition = getattr(item, "pre_ir_disposition", None)
        has_pending_semantics = bool(getattr(item, "needs_review_error", None)
                                     or getattr(item, "semantic_policy_pending", None)
                                     or isinstance(typed_disposition, PreIRDisposition)
                                     and typed_disposition.disposition == "review_pending")
        if (has_pending_semantics and item.semantic is not None and item.semantic.faithful_ir is not None
                and getattr(item, "ap_binding_context", None) is not None
                and item.ap_binding_context.has_authoritative_catalog):
            from pipeline import _semantic_review_pending_result
            from schemas import TraceCoverageReport
            fallback = _semantic_review_pending_result(
                requirement=item.requirement, semantic=item.semantic, traces=list(item.traces),
                review_records=list(item.review_records), trace_coverage=item.trace_coverage or TraceCoverageReport(status="incomplete"),
                strict_boundary=strict_boundary, review_mode=review_mode,
                experiment_config=ExperimentConfig(mode="human"), min_passed_skeletons=min_passed_skeletons,
                trace_evidence_policy=trace_evidence_policy)
            delivered = BatchPipelineItemResult(
                requirement_id=item.requirement_id, requirement=item.requirement, result=fallback,
                report_text=format_pipeline_result(fallback), summary_text=format_pipeline_summary(fallback),
                review_records=fallback.review_records, classification="complete_with_corrections")
            results.append(delivered)
            _write_incremental_outputs(results, partial_result_path=partial_result_path,
                partial_summary_path=partial_summary_path, classified_output_dir=classified_output_dir, reviewed_items=reviewed_items)
            continue
        if (
            isinstance(typed_disposition, PreIRDisposition)
            and typed_disposition.disposition != "approved"
        ):
            if typed_disposition.disposition == "review_pending":
                pending_record = typed_disposition.review_pending
                if pending_record is None:  # pragma: no cover - schema invariant
                    raise RuntimeError("review_pending disposition lacks its record")
                payload = asdict(pending_record)
                classification = "needs_review"
                report_text = (
                    f"## {item.requirement_id}\n\nstatus = review_pending\n\n"
                    f"reason = {pending_record.reason_code}\n\n"
                    f"required_action = {pending_record.required_action}"
                )
                summary_text = (
                    f"{item.requirement_id}: review_pending - "
                    f"{pending_record.reason_code}"
                )
            else:
                blocked_record = typed_disposition.blocked_failure
                if blocked_record is None:  # pragma: no cover - schema invariant
                    raise RuntimeError("blocked_error disposition lacks its record")
                payload = asdict(blocked_record)
                payload.update(
                    {
                        "stage": blocked_record.owner_stage,
                        "suggested_rerun_stage": blocked_record.safe_restart_stage,
                        "error": blocked_record.message,
                    }
                )
                classification = "blocked_error"
                report_text = _format_blocked_error_report(item, payload)
                summary_text = _format_blocked_error_summary(item, payload)
            disposition_item = BatchPipelineItemResult(
                requirement_id=item.requirement_id,
                requirement=item.requirement,
                result=None,
                report_text=report_text,
                summary_text=summary_text,
                review_records=list(item.review_records),
                classification=classification,
                error=payload,
                trace_evidence_policy=trace_evidence_policy,
                experimental_only=trace_evidence_policy.experimental_only,
            )
            results.append(disposition_item)
            _write_incremental_outputs(
                results,
                partial_result_path=partial_result_path,
                partial_summary_path=partial_summary_path,
                classified_output_dir=classified_output_dir,
                reviewed_items=reviewed_items,
            )
            if progress is not None:
                progress.warn(
                    f"{item.requirement_id} retained as {classification}"
                )
            continue
        plain_reading_pending = getattr(
            item, "plain_reading_resolution_pending", None
        )
        needs_review_error = (
            plain_reading_pending
            if plain_reading_pending is not None
            else getattr(item, "needs_review_error", None)
        )
        if needs_review_error is not None:
            review_payload = dict(needs_review_error)
            review_item = BatchPipelineItemResult(
                requirement_id=item.requirement_id,
                requirement=item.requirement,
                result=None,
                report_text=(
                    f"## {item.requirement_id}\n\n"
                    "status = needs_review\n\n"
                    f"reason = {review_payload.get('reason_code', '')}\n\n"
                    f"{review_payload.get('error', '')}"
                ),
                summary_text=(
                    f"{item.requirement_id}: needs_review - "
                    f"{review_payload.get('reason_code', '')}"
                ),
                review_records=list(item.review_records),
                classification="needs_review",
                error=review_payload,
                trace_evidence_policy=trace_evidence_policy,
                experimental_only=trace_evidence_policy.experimental_only,
            )
            results.append(review_item)
            _write_incremental_outputs(
                results,
                partial_result_path=partial_result_path,
                partial_summary_path=partial_summary_path,
                classified_output_dir=classified_output_dir,
                reviewed_items=reviewed_items,
            )
            if progress is not None:
                progress.warn(
                    f"{item.requirement_id} retained as recoverable needs_review"
                )
            continue
        semantic_policy_pending = getattr(item, "semantic_policy_pending", None)
        if semantic_policy_pending is not None:
            pending_payload = dict(semantic_policy_pending)
            pending_item = BatchPipelineItemResult(
                requirement_id=item.requirement_id,
                requirement=item.requirement,
                result=None,
                report_text=_format_semantic_policy_pending_report(
                    item,
                    pending_payload,
                ),
                summary_text=(
                    f"{item.requirement_id}: "
                    "等待专家有限轨迹边界审核；普通语言审核已完成。"
                ),
                review_records=list(item.review_records),
                classification="needs_review",
                error=pending_payload,
                trace_evidence_policy=trace_evidence_policy,
                experimental_only=trace_evidence_policy.experimental_only,
            )
            results.append(pending_item)
            _write_incremental_outputs(
                results,
                partial_result_path=partial_result_path,
                partial_summary_path=partial_summary_path,
                classified_output_dir=classified_output_dir,
                reviewed_items=reviewed_items,
            )
            if progress is not None:
                progress.warn(
                    f"{item.requirement_id} waits for deferred expert policy trace review"
                )
            continue
        if item.pre_ir_error is not None:
            error_payload = dict(item.pre_ir_error)
            if uses_sample_reference:
                raise RuntimeError(
                    f"{item.requirement_id} pre-IR blocked: "
                    f"{error_payload.get('error', '')}"
                )
            blocked_item = BatchPipelineItemResult(
                requirement_id=item.requirement_id,
                requirement=item.requirement,
                result=None,
                report_text=_format_blocked_error_report(item, error_payload),
                summary_text=_format_blocked_error_summary(item, error_payload),
                review_records=list(item.review_records),
                classification="blocked_error",
                error=error_payload,
                trace_evidence_policy=trace_evidence_policy,
                experimental_only=trace_evidence_policy.experimental_only,
            )
            results.append(blocked_item)
            _write_incremental_outputs(
                results,
                partial_result_path=partial_result_path,
                partial_summary_path=partial_summary_path,
                classified_output_dir=classified_output_dir,
                reviewed_items=reviewed_items,
            )
            if progress is not None:
                progress.error(
                    f"{item.requirement_id} skipped after pre-IR blocked_error"
                )
            continue
        feedback_packet = review_feedback.get(item.requirement_id)
        if feedback_packet is not None:
            _add_review_feedback_chunk(item, feedback_packet)
        cached_result = (
            None
            if feedback_packet is not None
            else _load_cached_pipeline_result(
                checkpoint_dir=checkpoint_dir,
                item=item,
                classified_output_dir=classified_output_dir,
                final_unique_results=(
                    final_unique_results and not review_feedback
                ),
            )
        )
        if cached_result is not None:
            results.append(cached_result)
            update_checkpoint_classification(
                checkpoint_dir,
                requirement_id=cached_result.requirement_id,
                classification=_item_classification(cached_result),
                degraded_item_path=(
                    str(
                        _degraded_item_path(
                            classified_output_dir,
                            cached_result.requirement_id,
                        )
                        or ""
                    )
                    if cached_result.degraded_record is not None
                    else ""
                ),
                expert_review_status=(
                    cached_result.degraded_record.expert_review_status
                    if cached_result.degraded_record is not None
                    else ""
                ),
                error=cached_result.error,
            )
            if progress is not None:
                progress.ok(f"{item.requirement_id} pipeline loaded from checkpoint")
            _write_incremental_outputs(
                results,
                partial_result_path=partial_result_path,
                partial_summary_path=partial_summary_path,
                classified_output_dir=classified_output_dir,
                reviewed_items=reviewed_items,
            )
            continue
        try:
            guided_context = item.guided_context
            result = run_traj_ltlf_pipeline_from_reviewed_inputs(
                requirement=item.requirement,
                semantic=item.semantic,
                traces=item.traces,
                initial_review_records=item.review_records,
                retriever=item.retriever,
                strict_boundary=strict_boundary,
                trace_evidence_policy=trace_evidence_policy,
                review_mode=review_mode,
                max_review_rounds=1 if guided_context is not None else max_review_rounds,
                min_passed_skeletons=min_passed_skeletons,
                records_dir=records_dir,
                llm_mode=llm_mode,
                llm_client=llm_client,
                approved_memory_file=approved_memory_file,
                progress=progress,
                progress_label=progress_label,
                guided_context=guided_context,
                trace_coverage=item.trace_coverage,
                experiment_config=(
                    item.experiment_config
                    or (
                        warmup_experiment_config
                        if guided_context is not None
                        else experiment_config
                    )
                ),
                judge_client=None if guided_context is not None else judge_client,
                start_stage=_pipeline_start_stage(feedback_packet),
                run_origin=("feedback_rerun" if feedback_packet else "normal"),
                ap_binding_context=getattr(item, "ap_binding_context", None),
                authoritative_trace_session=getattr(
                    item,
                    "authoritative_trace_session",
                    None,
                ),
            )
            _validate_guided_pipeline_result(item=item, result=result)
            if guided_context is not None:
                guided_chunk_ids = record_guided_warmup_process(
                    item.retriever,
                    context=guided_context,
                    semantic=result.semantic,
                    traces=result.traces,
                    skeletons=result.skeletons,
                    hole_candidates=result.hole_candidates,
                    completion_decisions=result.completion_decisions,
                    evaluations=result.evaluations,
                    selected=result.selected,
                    guided_decisions=_guided_decisions_from_review_records(
                        result.review_records
                    ),
                )
                if guided_chunk_ids:
                    result.review_records.append(
                        {
                            "stage": "guided_warmup_learning",
                            "round": 1,
                            "approved": True,
                            "record_path": "",
                            "storage": "inline",
                            "runtime_rag_chunks": guided_chunk_ids,
                            "comments": (
                                "warmup one-shot guided decision process "
                                "persisted for all pipeline stages."
                            ),
                        }
                    )
            if feedback_packet is not None:
                has_source_fingerprints = bool(
                    feedback_packet.get("source_artifact_fingerprints")
                )
                before_fingerprints = ArtifactFingerprintSet.from_mapping(
                    feedback_packet.get("source_artifact_fingerprints", {})
                )
                delta_report = compare_artifact_fingerprint_sets(
                    before_fingerprints,
                    result,
                    expected_changed_artifacts=tuple(
                        feedback_packet.get("expected_invalidation_set", ())
                        or _expected_artifact_changes_for_start_stage(
                            str(feedback_packet.get("restart_stage", "pipeline"))
                        )
                    ),
                    before_errors=feedback_packet.get("source_error_ids", ()),
                    after_errors=result.error_diagnoses,
                    diagnosis_id=str(feedback_packet.get("diagnosis_id", "")),
                    start_stage=str(feedback_packet.get("restart_stage", "pipeline")),
                    independent_approval_required=(
                        has_source_fingerprints
                        and
                        review_mode == "auto"
                        and effective_mode == "independent_judge"
                    ),
                    frozen_upstream=bool(
                        feedback_packet.get("frozen_upstream", False)
                    ),
                )
                result = _attach_feedback_patch_outcome(
                    result=result,
                    patch_payload=feedback_packet.get("feedback_patch"),
                    delta_report=delta_report,
                    reviewer_id=f"{review_mode}-owning-stage-validator",
                )
                result = _attach_rerun_delta_outcome(
                    result=result,
                    delta_report=delta_report,
                    requirement_id=item.requirement_id,
                )
                if (
                    review_mode == "auto"
                    and effective_mode == "independent_judge"
                    and has_source_fingerprints
                    and feedback_packet.get("fresh_judge_required", True)
                    and feedback_packet.get("output_policy")
                    == "fresh_independent_approval"
                    and result.provisional_candidate is not None
                    and result.operator_resolution.status
                    in {"complete", "not_required"}
                    and delta_report.status
                    in {"effective_delta", "needs_independent_approval"}
                ):
                    decision = request_fresh_independent_approval(
                        result=result,
                        delta_report=delta_report,
                        judge_client=judge_client,
                        prior_independence_ids=tuple(
                            feedback_packet.get("prior_independence_ids", ())
                        ),
                    )
                    if decision is not None and decision.approved:
                        approval = final_approval_from_independent_decision(
                            result=result,
                            delta_report=delta_report,
                            decision=decision,
                        )
                        selected_candidate = result.provisional_candidate
                        result = replace(
                            result,
                            selected=selected_candidate,
                            fallback_candidate=selected_candidate,
                            selection_status="fresh_independent_approved",
                            candidate_status="final",
                            approval_status="final",
                            learning_status="writeback_blocked_feedback_origin",
                            automatic_finalization_allowed=False,
                            provisional_candidate=None,
                            final_approval=approval,
                        )
                        result = replace(
                            result,
                            artifact_fingerprints=build_artifact_fingerprint_set(result),
                        )
        except Exception as exc:
            invalid_completions = list(
                getattr(exc, "invalid_completions", [])
            )
            pipeline_stage = str(getattr(exc, "pipeline_stage", "completion"))
            exception_reason_code = str(
                getattr(exc, "reason_code", "")
                or getattr(exc, "code", "")
                or type(exc).__name__
            ).upper()
            failure_diagnosis = build_error_diagnosis(
                requirement_id=item.requirement_id,
                source_run_id=(
                    "failed-run-"
                    + item.requirement_id
                    + "-"
                    + datetime.now().strftime("%Y%m%d%H%M%S%f")
                ),
                source_stage=pipeline_stage,
                candidate_information_exposed=bool(invalid_completions),
                diagnoser_role="pipeline_exception_diagnoser",
                diagnoser_independence_id="pipeline-exception-diagnoser-v1",
                compiler_error=exc,
                error_class=getattr(exc, "diagnosis_error_class", None),
                affected_ir_node_ids=getattr(
                    exc,
                    "affected_ir_node_ids",
                    (),
                ),
                affected_obligation_ids=getattr(
                    exc,
                    "affected_obligation_ids",
                    (),
                ),
                confidence="high",
            )
            error_payload = {
                "pipeline_stage": pipeline_stage,
                "stage": pipeline_stage,
                "reason_code": exception_reason_code,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "suggested_rerun_stage": _batch_stage_for_responsible_stage(
                    failure_diagnosis.responsible_stage
                ),
                "error_diagnosis": asdict(failure_diagnosis),
                "invalid_completions": invalid_completions,
                "allowed_aps": list(getattr(item.semantic, "aps", ()) or ()),
                "completed_artifacts": [
                    {
                        "stage": record.get("stage", ""),
                        "round": record.get("round", ""),
                        "record_path": record.get("record_path", ""),
                    }
                    for record in list(getattr(item, "review_records", []) or [])
                ],
                "recent_candidates": _recent_invalid_candidates(
                    invalid_completions
                ),
                "consistency_issues": list(
                    getattr(exc, "consistency_issues", ())
                ),
            }
            recoverable_reason = _recoverable_needs_review_reason(exc)
            if recoverable_reason:
                error_payload.update(
                    {
                        "status": "needs_review",
                        "reason_code": recoverable_reason,
                        "ap_context_fingerprint": getattr(
                            getattr(item.semantic, "faithful_ir", None),
                            "ap_context_fingerprint",
                            "",
                        ),
                    }
                )
                diagnostics = getattr(exc, "diagnostics", None)
                if isinstance(diagnostics, dict):
                    error_payload.update(diagnostics)
                save_needs_review_checkpoint(
                    checkpoint_dir,
                    requirement_id=item.requirement_id,
                    stage="ltlf_pipeline",
                    reason_code=recoverable_reason,
                    message=str(exc),
                    suggested_rerun_stage=str(
                        error_payload.get("suggested_rerun_stage", "semantic")
                    ),
                    details=error_payload,
                )
                results.append(
                    BatchPipelineItemResult(
                        requirement_id=item.requirement_id,
                        requirement=item.requirement,
                        result=None,
                        report_text=(
                            f"## {item.requirement_id}\n\n"
                            "status = needs_review\n\n"
                            f"reason = {recoverable_reason}\n\n{exc}"
                        ),
                        summary_text=(
                            f"{item.requirement_id}: needs_review - "
                            f"{recoverable_reason}"
                        ),
                        review_records=list(
                            getattr(item, "review_records", [])
                        ),
                        classification="needs_review",
                        error=error_payload,
                        trace_evidence_policy=trace_evidence_policy,
                        experimental_only=(
                            trace_evidence_policy.experimental_only
                        ),
                    )
                )
                _write_incremental_outputs(
                    results,
                    partial_result_path=partial_result_path,
                    partial_summary_path=partial_summary_path,
                    classified_output_dir=classified_output_dir,
                    reviewed_items=reviewed_items,
                )
                if progress is not None:
                    progress.warn(
                        f"{item.requirement_id} pipeline isolated as needs_review: "
                        f"{recoverable_reason}"
                    )
                continue
            save_failed_checkpoint(
                checkpoint_dir,
                requirement_id=item.requirement_id,
                stage="ltlf_pipeline",
                error=exc,
                suggested_rerun_stage=str(
                    error_payload.get("suggested_rerun_stage", "")
                ),
                details=error_payload,
            )
            if uses_sample_reference:
                raise RuntimeError(
                    f"{item.requirement_id} 未生成 warmup 样本的预期最终结果: {exc}"
                ) from exc
            results.append(
                BatchPipelineItemResult(
                    requirement_id=item.requirement_id,
                    requirement=item.requirement,
                    result=None,
                    report_text=_format_blocked_error_report(
                        item,
                        error_payload,
                    ),
                    summary_text=_format_blocked_error_summary(
                        item,
                        error_payload,
                    ),
                    review_records=list(getattr(item, "review_records", [])),
                    classification="blocked_error",
                    error=error_payload,
                    trace_evidence_policy=trace_evidence_policy,
                    experimental_only=trace_evidence_policy.experimental_only,
                )
            )
            _write_incremental_outputs(
                results,
                partial_result_path=partial_result_path,
                partial_summary_path=partial_summary_path,
                classified_output_dir=classified_output_dir,
                reviewed_items=reviewed_items,
            )
            if progress is not None:
                progress.error(
                    f"{item.requirement_id} pipeline rejected: {type(exc).__name__}: {exc}"
                )
            continue
        report_text = format_pipeline_result(result)
        summary_text = format_pipeline_summary(result)
        classification = _classification_for_result(result)
        degraded_record = None
        if classification in {"degraded_unproven", "degraded_proven"}:
            existing_status, existing_comments = _load_existing_expert_review(
                classified_output_dir,
                item.requirement_id,
            )
            if existing_status == "retry_requested":
                existing_status = "pending"
            degraded_record = build_degraded_result_record(
                item.requirement_id,
                result,
                expert_review_status=existing_status or "pending",
                expert_comments=existing_comments,
            )
        item_result = BatchPipelineItemResult(
            requirement_id=item.requirement_id,
            requirement=item.requirement,
            result=result,
            report_text=report_text,
            summary_text=summary_text,
            review_records=result.review_records,
            classification=classification,
            degraded_record=degraded_record,
            trace_evidence_policy=result.trace_evidence_policy,
            experimental_only=result.experimental_only,
        )
        results.append(item_result)
        degraded_item_path = _degraded_item_path(
            classified_output_dir,
            item.requirement_id,
        )
        save_pipeline_result_checkpoint(
            checkpoint_dir,
            requirement_id=item.requirement_id,
            requirement=item.requirement,
            report_text=report_text,
            summary_text=summary_text,
            review_records=result.review_records,
            classification=classification,
            trace_coverage=result.trace_coverage,
            selection_status=result.selection_status,
            selected_formula=(
                str(result.selected.formula)
                if result.selected is not None
                else ""
            ),
            degraded_record=degraded_record,
            degraded_item_path=degraded_item_path,
            ir_version=(
                result.semantic.faithful_ir.ir_version
                if result.semantic.faithful_ir is not None
                else ""
            ),
            ir_coverage=result.ir_coverage,
            operator_resolution=result.operator_resolution,
            candidate_status=result.candidate_status,
            run_origin=result.run_origin,
            automatic_finalization_allowed=result.automatic_finalization_allowed,
            final_approval=result.final_approval,
            semantic_status=result.semantic_status,
            evidence_status=result.evidence_status,
            approval_status=result.approval_status,
            learning_status=result.learning_status,
            source_run_id=result.source_run_id,
            formula_closure=result.formula_closure,
            ambiguity_closure=result.ambiguity_closure,
            artifact_fingerprints=result.artifact_fingerprints,
            error_diagnoses=result.error_diagnoses,
            rerun_delta=result.rerun_delta,
            feedback_patch=result.feedback_patch,
            trace_evidence_policy=result.trace_evidence_policy,
            canonical_trace_coverage=result.canonical_trace_coverage,
            experimental_only=result.experimental_only,
            boundary_metric_available=result.boundary_metric_available,
            ap_context_fingerprint=result.ap_context_fingerprint,
            relation_feasibility_classification=(
                result.relation_feasibility.classification
                if result.relation_feasibility is not None
                else ""
            ),
            feasibility_report_hash=(
                result.relation_feasibility.artifact_hash
                if result.relation_feasibility is not None
                else ""
            ),
            source_hash=(
                result.semantic.surface_segmentation.source_hash
                if result.semantic.surface_segmentation is not None
                else source_hash(item.requirement)
            ),
            segmentation_id=(
                result.semantic.surface_segmentation.segmentation_id
                if result.semantic.surface_segmentation is not None
                else ""
            ),
            reading_review_id=(
                result.semantic.plain_reading_review.reading_review_id
                if result.semantic.plain_reading_review is not None
                else ""
            ),
            plain_reading_decision_fingerprint=decision_fingerprint(
                result.semantic.plain_reading_review
            ),
            segmentation_schema_version=(
                result.semantic.surface_segmentation.schema_version
                if result.semantic.surface_segmentation is not None
                else ""
            ),
            plain_reading_schema_version=(
                result.semantic.plain_reading_review.schema_version
                if result.semantic.plain_reading_review is not None
                else ""
            ),
        )
        _write_incremental_outputs(
            results,
            partial_result_path=partial_result_path,
            partial_summary_path=partial_summary_path,
            classified_output_dir=classified_output_dir,
            reviewed_items=reviewed_items,
        )
        if progress is not None:
            if classification in {"degraded_unproven", "degraded_proven"}:
                progress.warn(
                    f"{item.requirement_id} pipeline ended: {classification}"
                )
            elif classification == "needs_review":
                progress.warn(f"{item.requirement_id} pipeline ended: needs_review")
            elif result.selected is None:
                progress.warn(f"{item.requirement_id} pipeline ended: needs_review")
            else:
                progress.ok(f"{item.requirement_id} pipeline finished")

    if (
        final_unique_results
        and not review_feedback
        and not trace_evidence_policy.experimental_only
    ):
        results = _rerun_intermediate_review_results(
            results=results,
            reviewed_items=reviewed_items,
            strict_boundary=strict_boundary,
            review_mode=review_mode,
            effective_mode=effective_mode,
            max_review_rounds=max_review_rounds,
            min_passed_skeletons=min_passed_skeletons,
            records_dir=records_dir,
            llm_mode=llm_mode,
            llm_client=llm_client,
            approved_memory_file=approved_memory_file,
            checkpoint_dir=checkpoint_dir,
            partial_result_path=partial_result_path,
            partial_summary_path=partial_summary_path,
            classified_output_dir=classified_output_dir,
            progress=progress,
            experiment_config=experiment_config,
            warmup_experiment_config=warmup_experiment_config,
            judge_client=judge_client,
            trace_evidence_policy=trace_evidence_policy,
        )

    persistable_chunks = _filter_persistable_chunks(
        [
            *_collect_retriever_chunks(
                [item.retriever for item in reviewed_items]
            ),
            *_collect_guided_correction_chunks(results),
        ],
        results,
    )
    persistable_pattern_specs = _filter_persistable_pattern_specs(
        pattern_specs,
        results,
    )
    if trace_evidence_policy.experimental_only:
        persistable_chunks = []
        persistable_pattern_specs = []
    faithful_examples = [
        item.result.learning_example
        for item in results
        if (
            item.result is not None
            and item.result.learning_example is not None
            and _result_allows_positive_rag_writeback(item)
        )
    ]
    if trace_evidence_policy.experimental_only:
        faithful_examples = []
    _persist_approved_memory(
        approved_memory_file,
        chunks=persistable_chunks,
        pattern_specs=persistable_pattern_specs,
        faithful_examples=faithful_examples,
        progress=progress,
    )
    _write_incremental_outputs(
        results,
        partial_result_path=partial_result_path,
        partial_summary_path=partial_summary_path,
        classified_output_dir=classified_output_dir,
        reviewed_items=reviewed_items,
    )
    if final_unique_results and classified_output_dir:
        write_final_unique_batch_outputs(
            results,
            output_dir=classified_output_dir,
            expected_items=reviewed_items,
        )

    if require_complete_batch:
        outcome = compute_batch_completeness(
            results,
            expected_items=reviewed_items,
        )
        if not outcome.complete:
            raise IncompleteBatchError(outcome)

    return results


def _final_unique_results_enabled(
    *,
    review_mode: str,
    experiment_mode: str,
    llm_mode: str | None,
) -> bool:
    if review_mode == "auto" and experiment_mode == "independent_judge":
        return True
    return (
        review_mode == "manual"
        and experiment_mode == "human"
        and (llm_mode or "").lower() == "api"
    )


def _rerun_intermediate_review_results(
    *,
    results: list[BatchPipelineItemResult],
    reviewed_items: list[object],
    strict_boundary: bool,
    review_mode: str,
    effective_mode: str,
    max_review_rounds: int,
    min_passed_skeletons: int,
    records_dir: str,
    llm_mode: str | None,
    llm_client: JSONLLMClient | None,
    approved_memory_file: str | None,
    checkpoint_dir: str | None,
    partial_result_path: str | None,
    partial_summary_path: str | None,
    classified_output_dir: str | None,
    progress: TerminalProgress | None,
    experiment_config: ExperimentConfig | None,
    warmup_experiment_config: ExperimentConfig,
    judge_client: JSONLLMClient | None,
    trace_evidence_policy: TraceEvidencePolicy,
) -> list[BatchPipelineItemResult]:
    reviewed_by_id = {
        str(getattr(item, "requirement_id", "")): item
        for item in reviewed_items
    }
    updated_results = list(results)
    unresolved_indices = [
        index
        for index, item in enumerate(updated_results)
        if item.result is not None
        if _item_classification(item)
        in {"needs_review", "degraded_unproven", "degraded_proven"}
    ]
    if not unresolved_indices:
        validate_result_uniqueness(updated_results)
        return updated_results

    if progress is not None:
        progress.stage("Final review feedback rerun")
        progress.info(f"Intermediate review item(s): {len(unresolved_indices)}")

    for index in unresolved_indices:
        previous_item = updated_results[index]
        reviewed_item = reviewed_by_id.get(previous_item.requirement_id)
        if reviewed_item is None:
            raise RuntimeError(
                f"{previous_item.requirement_id} 缺少 reviewed input，无法重跑待审项。"
            )

        feedback_chunks = _record_intermediate_review_feedback(
            reviewed_item=reviewed_item,
            item_result=previous_item,
            review_mode=review_mode,
            effective_mode=effective_mode,
        )
        if progress is not None and feedback_chunks:
            progress.info(
                f"{previous_item.requirement_id} review feedback RAG: "
                + ", ".join(feedback_chunks)
            )

        guided_context = getattr(reviewed_item, "guided_context", None)
        item_config = (
            getattr(reviewed_item, "experiment_config", None)
            or (
                warmup_experiment_config
                if guided_context is not None
                else experiment_config
            )
        )
        result = run_traj_ltlf_pipeline_from_reviewed_inputs(
            requirement=getattr(reviewed_item, "requirement"),
            semantic=getattr(reviewed_item, "semantic"),
            traces=getattr(reviewed_item, "traces"),
            initial_review_records=getattr(reviewed_item, "review_records", []),
            retriever=getattr(reviewed_item, "retriever"),
            strict_boundary=strict_boundary,
            trace_evidence_policy=trace_evidence_policy,
            review_mode=review_mode,
            max_review_rounds=1 if guided_context is not None else max_review_rounds,
            min_passed_skeletons=min_passed_skeletons,
            records_dir=records_dir,
            llm_mode=llm_mode,
            llm_client=llm_client,
            approved_memory_file=approved_memory_file,
            progress=progress,
            progress_label=f"{previous_item.requirement_id} final-rerun",
            guided_context=guided_context,
            trace_coverage=getattr(reviewed_item, "trace_coverage", None),
            experiment_config=item_config,
            judge_client=None if guided_context is not None else judge_client,
            run_origin="feedback_rerun",
            ap_binding_context=getattr(
                reviewed_item, "ap_binding_context", None
            ),
            authoritative_trace_session=getattr(
                reviewed_item,
                "authoritative_trace_session",
                None,
            ),
        )
        expected_changes = _expected_artifact_changes_for_start_stage("skeleton")
        diagnosis_id = (
            previous_item.result.error_diagnoses[0].diagnosis_id
            if previous_item.result is not None
            and previous_item.result.error_diagnoses
            else ""
        )
        delta_report = compare_artifact_fingerprint_sets(
            previous_item.result,
            result,
            expected_changed_artifacts=expected_changes,
            before_errors=(
                previous_item.result.error_diagnoses
                if previous_item.result is not None
                else ()
            ),
            after_errors=result.error_diagnoses,
            diagnosis_id=diagnosis_id,
            start_stage="skeleton",
            independent_approval_required=(
                review_mode == "auto"
                and effective_mode == "independent_judge"
                and not trace_evidence_policy.experimental_only
            ),
        )
        prior_diagnosis = _primary_error_diagnosis(previous_item.result)
        intermediate_patch = (
            asdict(
                replace(
                    propose_feedback_patch(
                        prior_diagnosis,
                        source_ir_version=(
                            previous_item.result.semantic.faithful_ir.ir_version
                            if previous_item.result is not None
                            and previous_item.result.semantic.faithful_ir is not None
                            else ""
                        ),
                    ),
                    rerun_start_stage="skeleton",
                    expected_changed_artifacts=expected_changes,
                )
            )
            if prior_diagnosis is not None
            else None
        )
        result = _attach_feedback_patch_outcome(
            result=result,
            patch_payload=intermediate_patch,
            delta_report=delta_report,
            reviewer_id=f"{review_mode}-owning-stage-validator",
        )
        result = _attach_rerun_delta_outcome(
            result=result,
            delta_report=delta_report,
            requirement_id=previous_item.requirement_id,
        )
        if (
            review_mode == "auto"
            and effective_mode == "independent_judge"
            and not result.experimental_only
            and result.provisional_candidate is not None
            and result.operator_resolution.status
            in {"complete", "not_required"}
            and delta_report.status
            in {"effective_delta", "needs_independent_approval"}
        ):
            prior_independence_ids = {
                diagnosis.diagnoser_independence_id
                for diagnosis in (
                    previous_item.result.error_diagnoses
                    if previous_item.result is not None
                    else ()
                )
            }
            if (
                previous_item.result is not None
                and previous_item.result.final_approval is not None
                and previous_item.result.final_approval.independence_id
            ):
                prior_independence_ids.add(
                    previous_item.result.final_approval.independence_id
                )
            decision = request_fresh_independent_approval(
                result=result,
                delta_report=delta_report,
                judge_client=judge_client,
                prior_independence_ids=prior_independence_ids,
            )
            if decision is not None and decision.approved:
                approval = final_approval_from_independent_decision(
                    result=result,
                    delta_report=delta_report,
                    decision=decision,
                )
                selected_candidate = result.provisional_candidate
                result = replace(
                    result,
                    selected=selected_candidate,
                    fallback_candidate=selected_candidate,
                    selection_status="fresh_independent_approved",
                    candidate_status="final",
                    approval_status="final",
                    learning_status="writeback_blocked_feedback_origin",
                    automatic_finalization_allowed=False,
                    provisional_candidate=None,
                    final_approval=approval,
                )
                result = replace(
                    result,
                    artifact_fingerprints=build_artifact_fingerprint_set(result),
                )
        report_text = format_pipeline_result(result)
        summary_text = format_pipeline_summary(result)
        rerun_classification = (
            "complete"
            if result.selected is not None and result.candidate_status == "final"
            else "needs_review"
        )
        final_item = BatchPipelineItemResult(
            requirement_id=previous_item.requirement_id,
            requirement=result.requirement,
            result=result,
            report_text=report_text,
            summary_text=summary_text,
            review_records=result.review_records,
            classification=rerun_classification,
            trace_evidence_policy=result.trace_evidence_policy,
            experimental_only=result.experimental_only,
        )
        updated_results[index] = final_item
        save_pipeline_result_checkpoint(
            checkpoint_dir,
            requirement_id=final_item.requirement_id,
            requirement=final_item.requirement,
            report_text=report_text,
            summary_text=summary_text,
            review_records=result.review_records,
            classification=rerun_classification,
            trace_coverage=result.trace_coverage,
            selection_status=result.selection_status,
            selected_formula=(
                str(result.selected.formula) if result.selected is not None else ""
            ),
            degraded_record=None,
            degraded_item_path="",
            ir_version=(
                result.semantic.faithful_ir.ir_version
                if result.semantic.faithful_ir is not None
                else ""
            ),
            ir_coverage=result.ir_coverage,
            operator_resolution=result.operator_resolution,
            candidate_status=result.candidate_status,
            run_origin=result.run_origin,
            automatic_finalization_allowed=result.automatic_finalization_allowed,
            final_approval=result.final_approval,
            semantic_status=result.semantic_status,
            evidence_status=result.evidence_status,
            approval_status=result.approval_status,
            learning_status=result.learning_status,
            source_run_id=result.source_run_id,
            formula_closure=result.formula_closure,
            ambiguity_closure=result.ambiguity_closure,
            artifact_fingerprints=result.artifact_fingerprints,
            error_diagnoses=result.error_diagnoses,
            rerun_delta=result.rerun_delta,
            feedback_patch=result.feedback_patch,
            trace_evidence_policy=result.trace_evidence_policy,
            canonical_trace_coverage=result.canonical_trace_coverage,
            experimental_only=result.experimental_only,
            boundary_metric_available=result.boundary_metric_available,
            source_hash=(
                result.semantic.surface_segmentation.source_hash
                if result.semantic.surface_segmentation is not None
                else source_hash(final_item.requirement)
            ),
            segmentation_id=(
                result.semantic.surface_segmentation.segmentation_id
                if result.semantic.surface_segmentation is not None
                else ""
            ),
            reading_review_id=(
                result.semantic.plain_reading_review.reading_review_id
                if result.semantic.plain_reading_review is not None
                else ""
            ),
            plain_reading_decision_fingerprint=decision_fingerprint(
                result.semantic.plain_reading_review
            ),
            segmentation_schema_version=(
                result.semantic.surface_segmentation.schema_version
                if result.semantic.surface_segmentation is not None
                else ""
            ),
            plain_reading_schema_version=(
                result.semantic.plain_reading_review.schema_version
                if result.semantic.plain_reading_review is not None
                else ""
            ),
        )
        _write_incremental_outputs(
            updated_results,
            partial_result_path=partial_result_path,
            partial_summary_path=partial_summary_path,
            classified_output_dir=classified_output_dir,
            reviewed_items=reviewed_items,
        )
        if progress is not None:
            if result.selected is not None:
                progress.ok(
                    f"{final_item.requirement_id} explicitly approved final: "
                    f"{result.selected.formula}"
                )
            else:
                progress.warn(
                    f"{final_item.requirement_id} feedback rerun remains provisional"
                )

    return updated_results


def _expected_artifact_changes_for_start_stage(stage: str) -> tuple[str, ...]:
    normalized = str(stage).strip().lower()
    mapping = {
        "pipeline": (
            "plain_reading",
            "semantic_ir",
            "approved_ir",
            "trace_plan",
            "trace_set",
            "skeleton_plan",
            "hole_plan",
            "candidate_set",
            "evaluation",
        ),
        "segmentation": (
            "plain_reading",
            "semantic_ir",
            "approved_ir",
            "trace_plan",
            "trace_set",
            "skeleton_plan",
            "hole_plan",
            "candidate_set",
            "evaluation",
        ),
        "plain_reading": (
            "plain_reading",
            "semantic_ir",
            "approved_ir",
            "trace_plan",
            "trace_set",
            "skeleton_plan",
            "hole_plan",
            "candidate_set",
            "evaluation",
        ),
        "semantic": (
            "semantic_ir",
            "approved_ir",
            "trace_plan",
            "trace_set",
            "skeleton_plan",
            "hole_plan",
            "candidate_set",
            "evaluation",
        ),
        "trace": ("trace_set", "evaluation"),
        "skeleton": ("skeleton_plan", "hole_plan", "candidate_set", "evaluation"),
        "completion": ("candidate_set", "evaluation"),
        "selection": ("evaluation", "approval"),
        "approval": ("approval",),
    }
    return mapping.get(normalized, ("candidate_set", "evaluation"))


def _record_intermediate_review_feedback(
    *,
    reviewed_item: object,
    item_result: BatchPipelineItemResult,
    review_mode: str,
    effective_mode: str,
) -> list[str]:
    result = item_result.result
    if result is None:
        return []

    source = (
        "independent_judge"
        if review_mode == "auto" and effective_mode == "independent_judge"
        else "human_review"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    chunk_id = f"runtime-final-review-feedback-{item_result.requirement_id}-{timestamp}"
    candidate_lines = [
        "- omitted: rerun feedback must not expose candidate formulas, scores, "
        "prior completions, or expected final candidates."
    ]
    comments = _objective_independent_feedback_comments(result)
    instruction = (
        "Rerun framework stages using only reviewed NL semantics, trace "
        "coverage status, IR coverage status, and objective review notes. "
        "Do not target any prior candidate formula or score. Do not design "
        "LTLf for the current traces or traces for any LTLf candidate; keep "
        "the result provisional until exhaustive operator resolution and all "
        "fresh approval gates pass."
    )
    chunk = KnowledgeChunk(
        chunk_id=chunk_id,
        title="Final review feedback for rerun",
        text="\n".join(
            [
                "example_type: final_review_feedback",
                f"requirement_id: {item_result.requirement_id}",
                f"requirement: {item_result.requirement or result.requirement}",
                f"feedback_source: {source}",
                f"intermediate_status: {_item_classification(item_result)}",
                f"selection_status: {result.selection_status}",
                f"trace_coverage_status: {result.trace_coverage.status}",
                "review_comments:",
                comments,
                "candidate_evaluations:",
                *(candidate_lines or ["- no candidates"]),
                "instruction:",
                instruction,
                f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
            ]
        ),
        source="runtime:final-review-feedback",
    )
    retriever = getattr(reviewed_item, "retriever")
    retriever.chunks.append(chunk)
    record = {
        "stage": "final_review_feedback",
        "round": 1,
        "approved": True,
        "feedback_source": source,
        "runtime_rag_chunks": [chunk_id],
        "comments": comments,
    }
    getattr(reviewed_item, "review_records").append(record)
    return [chunk_id]


def _objective_independent_feedback_comments(result: PipelineResult) -> str:
    """Build formula-free feedback for independent_judge reruns."""

    parts = [
        "independent_judge objective feedback only",
        f"selection_status={result.selection_status}",
        f"candidate_status={result.candidate_status}",
        f"trace_coverage_status={result.trace_coverage.status}",
        f"ir_coverage_status={result.ir_coverage.status}",
    ]
    if result.trace_coverage.reason:
        parts.append(f"trace_coverage_reason={result.trace_coverage.reason}")
    parts.extend(f"coverage_debt={item}" for item in result.trace_coverage.coverage_debt)
    parts.extend(f"ir_coverage_reason={item}" for item in result.ir_coverage.reasons)
    parts.append(
        "candidate formulas, scores, and prior completions intentionally omitted"
    )
    return "；".join(parts)


def validate_result_uniqueness(
    results: list[BatchPipelineItemResult],
) -> None:
    """Enforce at most one result record and one final result per requirement."""

    counts: dict[str, int] = {}
    final_counts: dict[str, int] = {}
    for item in results:
        counts[item.requirement_id] = counts.get(item.requirement_id, 0) + 1
        if _final_formula_text(item):
            final_counts[item.requirement_id] = final_counts.get(item.requirement_id, 0) + 1
    duplicates = sorted(
        requirement_id
        for requirement_id, count in counts.items()
        if count > 1
    )
    duplicate_finals = sorted(
        requirement_id
        for requirement_id, count in final_counts.items()
        if count > 1
    )
    if duplicates or duplicate_finals:
        affected = sorted(set(duplicates) | set(duplicate_finals))
        raise DuplicateBatchResultError(
            "multiple batch/final records for requirement IDs: "
            + ", ".join(affected)
        )


def compute_batch_completeness(
    results: list[BatchPipelineItemResult],
    *,
    expected_items: list[object] | None = None,
) -> BatchRunOutcome:
    """Compute completeness independently from the uniqueness invariant."""

    validate_result_uniqueness(results)
    result_ids = tuple(item.requirement_id for item in results)
    source_items = expected_items if expected_items is not None else results
    expected_ids = tuple(
        str(getattr(item, "requirement_id", ""))
        for item in source_items
        if str(getattr(item, "requirement_id", ""))
    )
    if len(expected_ids) != len(set(expected_ids)):
        raise DuplicateBatchResultError("expected batch contains duplicate requirement IDs")
    result_by_id = {item.requirement_id: item for item in results}
    missing = tuple(
        requirement_id
        for requirement_id in expected_ids
        if not _final_formula_text(result_by_id.get(requirement_id))
    )
    final_count = sum(
        1
        for requirement_id in expected_ids
        if _final_formula_text(result_by_id.get(requirement_id))
    )
    item_statuses = tuple(
        (
            requirement_id,
            "final"
            if _final_formula_text(result_by_id.get(requirement_id))
            else _item_classification(result_by_id.get(requirement_id)),
        )
        for requirement_id in expected_ids
    )
    pending_count = sum(
        1
        for _requirement_id, status in item_statuses
        if status in {
            "pending",
            "needs_review",
            "degraded_unproven",
            "degraded_proven",
            "rejected",
        }
    )
    blocked_count = sum(
        1 for _requirement_id, status in item_statuses if status == "blocked_error"
    )
    complete = not missing and set(result_ids) == set(expected_ids)
    return BatchRunOutcome(
        expected_count=len(expected_ids),
        result_count=len(result_ids),
        final_count=final_count,
        pending_count=pending_count,
        blocked_count=blocked_count,
        uniqueness_ok=True,
        complete=complete,
        release_ready=complete,
        missing_requirement_ids=missing,
        item_statuses=item_statuses,
    )


def _assert_unique_final_results(results: list[BatchPipelineItemResult]) -> None:
    """Compatibility alias: uniqueness no longer asserts batch completeness."""

    validate_result_uniqueness(results)


def _sample_reference_configs(
    warmup_examples: list[WarmupExample] | None,
) -> dict[str, ExperimentConfig]:
    configs: dict[str, ExperimentConfig] = {}
    missing: list[str] = []
    for example in warmup_examples or []:
        formula_text = _warmup_sample_reference_text(example)
        if not formula_text:
            missing.append(example.requirement_id)
            continue
        try:
            reference_formula = parse_formula(formula_text)
        except ValueError as exc:
            raise ValueError(
                f"{example.requirement_id} 的 warmup 样本参考公式无法解析: "
                f"{formula_text}"
            ) from exc
        configs[example.requirement_id] = ExperimentConfig(
            mode="formula_oracle",
            data_split="train",
            reference_formula=reference_formula,
            test_time_memory_frozen=True,
        )
    if missing:
        raise ValueError(
            "--reference-formula sample 需要每个 warmup 样本包含 "
            "approved_formula 或可解析的最终候选公式。缺失样例: "
            + ", ".join(missing)
        )
    return configs


def _guided_decisions_from_review_records(review_records: list[dict]) -> list[dict]:
    decisions: list[dict] = []
    for record in review_records:
        decision = record.get("guided_decision")
        if isinstance(decision, dict):
            decisions.append(decision)
        nested = record.get("guided_decisions")
        if isinstance(nested, list):
            decisions.extend(item for item in nested if isinstance(item, dict))
    return decisions


def _collect_guided_correction_chunks(
    results: list[BatchPipelineItemResult],
) -> list[object]:
    chunks: list[object] = []
    seen: set[str] = set()
    for item in results:
        if _item_classification(item) not in {"complete", "complete_with_corrections"}:
            continue
        records = item.review_records
        if item.result is not None:
            records = item.result.review_records
        for chunk in build_guided_correction_chunks_from_review_records(records):
            chunk_id = getattr(chunk, "chunk_id", "")
            if chunk_id in seen:
                continue
            if chunk_id:
                seen.add(chunk_id)
            chunks.append(chunk)
    return chunks


def _warmup_sample_reference_text(example: WarmupExample) -> str:
    guided_context = example.guided_context
    if guided_context is not None:
        for text in (
            guided_context.expected_final_candidate,
            guided_context.approved_formula,
        ):
            if text and text.strip():
                return text.strip()
    if example.pattern_spec is not None:
        for text in (
            example.pattern_spec.final_candidate_text,
            example.pattern_spec.formula_text,
        ):
            if text and text.strip():
                return text.strip()
    if example.expected_semantic is not None and example.expected_semantic.formula_text:
        return example.expected_semantic.formula_text.strip()
    return ""


def _persist_approved_memory(
    approved_memory_file: str | None,
    *,
    chunks: list[object],
    pattern_specs: list[object],
    faithful_examples: list[object],
    progress: TerminalProgress | None,
) -> None:
    if not approved_memory_file:
        return
    memory = merge_approved_memory(
        approved_memory_file,
        chunks=chunks,
        pattern_specs=pattern_specs,
        faithful_examples=faithful_examples,
    )
    if progress is not None:
        progress.info(
            "Saved persistent RAG memory: "
            f"{len(memory.chunks)} chunk(s), "
            f"{len(memory.pattern_specs)} pattern spec(s)"
        )


def _filter_persistable_chunks(
    chunks: list[object],
    results: list[BatchPipelineItemResult],
) -> list[object]:
    approved_chunks = [
        chunk
        for chunk in chunks
        if str(getattr(chunk, "source", "")).startswith("approved-example:")
    ]
    filtered: list[object] = []
    for chunk in approved_chunks:
        source = str(getattr(chunk, "source", ""))
        if "guided-correction" in source:
            filtered.append(chunk)
            continue
        item = _matching_result_for_chunk(chunk, results)
        if item is None:
            filtered.append(chunk)
            continue
        if not _result_allows_positive_rag_writeback(item):
            continue
        filtered.append(chunk)
    return filtered


def _matching_result_for_chunk(
    chunk: object,
    results: list[BatchPipelineItemResult],
) -> BatchPipelineItemResult | None:
    text = str(getattr(chunk, "text", ""))
    for item in results:
        if f"requirement_id: {item.requirement_id}" in text:
            return item
        if item.requirement and f"requirement: {item.requirement}" in text:
            return item
    return None


def _result_allows_positive_rag_writeback(item: BatchPipelineItemResult) -> bool:
    result = item.result
    if result is None:
        return False
    final_approval = result.final_approval
    approved_semantic = result.semantic.approved_semantic_ir
    segmentation = result.semantic.surface_segmentation
    reading = result.semantic.plain_reading_review
    reading_readiness = (
        assess_plain_reading_ir_readiness(reading, segmentation)
        if reading is not None and segmentation is not None
        else None
    )
    return bool(
        _item_classification(item) in {"complete", "complete_with_corrections"}
        and result.selected is not None
        and result.candidate_status == "final"
        and result.approval_status == "final"
        and result.learning_status == "writeback_allowed"
        and result.formula_closure is not None
        and result.formula_closure.passed
        and result.semantic.data_split == "train"
        and result.trace_coverage.complete
        and result.ir_coverage.complete
        and approved_semantic is not None
        and approved_semantic.approved
        and not approved_semantic.remaining_ambiguity_debt
        and final_approval is not None
        and final_approval.writeback_allowed
        and segmentation is not None
        and segmentation.mechanical_validation.status == "passed"
        and segmentation.status in {"surface_reviewed", "ambiguity_flagged"}
        and segmentation.label_source
        in {"human", "independent_judge", "warmup_fixture"}
        and reading is not None
        and reading_readiness is not None
        and reading_readiness.can_enter_semantic_ir
        and reading.downstream_status == "current"
        and reading.reviewer_source
        in {"human", "independent_judge", "warmup_fixture"}
    )


def _collect_retriever_chunks(retrievers: list[object]) -> list[object]:
    chunks: list[object] = []
    seen_ids: set[str] = set()
    for retriever in retrievers:
        for chunk in getattr(retriever, "chunks", []):
            chunk_id = str(getattr(chunk, "chunk_id", ""))
            if chunk_id and chunk_id in seen_ids:
                continue
            if chunk_id:
                seen_ids.add(chunk_id)
            chunks.append(chunk)
    return chunks


def _filter_persistable_pattern_specs(
    pattern_specs: list[object],
    results: list[BatchPipelineItemResult],
) -> list[object]:
    results_by_id = {item.requirement_id: item for item in results}
    return [
        spec
        for spec in pattern_specs
        if (
            (item := results_by_id.get(str(getattr(spec, "source_id", "")))) is None
            or _result_allows_positive_rag_writeback(item)
        )
    ]


def format_batch_pipeline_results(results: list[BatchPipelineItemResult]) -> str:
    """Format production-complete and explicitly experimental results."""

    normal_results = [
        item
        for item in results
        if _is_primary_report_item(item)
    ]
    lines = [
        "Traj-LTLf 批量运行结果",
        "=" * 40,
        f"需求数量: {len(normal_results)}",
        "",
    ]
    for index, item in enumerate(normal_results, start=1):
        lines.extend(
            [
                "",
                "#" * 60,
                f"批量项 {index}: {item.requirement_id}",
                "#" * 60,
                "",
                _format_item_report(item),
            ]
        )
    return "\n".join(lines)


def format_batch_pipeline_summary(results: list[BatchPipelineItemResult]) -> str:
    """Format production-complete and explicitly experimental candidates."""

    normal_results = [
        item
        for item in results
        if _is_primary_report_item(item)
    ]
    lines = [
        "Traj-LTLf 批量精简结果",
        f"需求数量: {len(normal_results)}",
    ]
    for index, item in enumerate(normal_results, start=1):
        lines.extend(
            [
                "",
                f"[{item.requirement_id}] 批量项 {index}",
                _format_item_summary(item),
            ]
        )
    return "\n".join(lines)


def format_final_unique_batch_results(
    results: list[BatchPipelineItemResult],
) -> str:
    """Format the final non-classified report with every requirement present."""

    lines = [
        "Traj-LTLf 最终唯一运行结果",
        "=" * 40,
        f"总需求数: {len(results)}",
        _format_final_count_line(batch_unique_result_counts(results)),
        "",
    ]
    for index, item in enumerate(results, start=1):
        lines.extend(
            [
                "",
                "#" * 60,
                f"最终项 {index}: {item.requirement_id}",
                "#" * 60,
                "",
                f"最终公式: {_final_formula_text(item) or '缺失'}",
                f"裁决依据: {_review_basis_for_item(item)}",
                "",
                _format_item_report(item),
            ]
        )
    return "\n".join(lines)


def format_final_unique_batch_summary(
    results: list[BatchPipelineItemResult],
) -> str:
    """Format the final concise result; no normal/review/degraded classes."""

    counts = batch_unique_result_counts(results)
    lines = [
        "Traj-LTLf 最终唯一候选",
        f"总需求数: {counts['total']}",
        _format_final_count_line(counts),
        f"回退选择: {sum(bool(item.result and item.result.provisional_candidate) for item in results)}/{counts['total']}",
    ]
    final_items = [item for item in results if _final_formula_text(item)]
    missing_items = [item for item in results if not _final_formula_text(item)]
    for item in final_items:
        lines.extend(
            [
                "",
                f"[{item.requirement_id}]",
                f"需求: {item.requirement or _requirement_text(item)}",
                f"最终候选: {_final_formula_text(item)}",
                f"裁决依据: {_review_basis_for_item(item)}",
            ]
        )
    if missing_items:
        lines.extend(("", "缺失最终结果:"))
        for item in missing_items:
            reason_code = str((item.error or {}).get("reason_code", ""))
            status = _item_classification(item)
            reason = f"{status}/{reason_code}" if reason_code else status
            lines.append(f"[{item.requirement_id}] {reason}")
    return "\n".join(lines)


def write_final_unique_batch_outputs(
    results: list[BatchPipelineItemResult],
    *,
    output_dir: str | Path,
    expected_items: list[object] | None = None,
) -> dict[str, int]:
    """Write the final non-classified manifest and unified result files."""

    root = Path(output_dir)
    source_items = expected_items or results
    outcome = compute_batch_completeness(
        results,
        expected_items=list(source_items),
    )
    result_by_id = {item.requirement_id: item for item in results}
    manifest_items: list[dict[str, Any]] = []
    for source in source_items:
        requirement_id = str(getattr(source, "requirement_id", ""))
        item = result_by_id.get(requirement_id)
        formula = _final_formula_text(item) if item is not None else ""
        provisional = _provisional_candidate_text(item) if item is not None else ""
        classification = _item_classification(item) if item is not None else "pending"
        manifest_items.append(
            {
                "requirement_id": requirement_id,
                "requirement": (
                    _requirement_text(item)
                    if item is not None
                    else str(getattr(source, "requirement", ""))
                ),
                "final_formula": formula,
                "provisional_candidate": "" if formula else provisional,
                "classification": classification,
                "result_status": ("fallback_selected" if item and item.result and item.result.provisional_candidate
                                  else "final" if formula else classification),
                "candidate_status": (
                    item.result.candidate_status
                    if item is not None and item.result is not None
                    else ""
                ),
                "trace_evidence_policy": (
                    asdict(item.result.trace_evidence_policy)
                    if item is not None and item.result is not None
                    else asdict(item.trace_evidence_policy)
                    if item is not None
                    else asdict(TraceEvidencePolicy())
                ),
                "experimental_only": bool(
                    item is not None
                    and (
                        item.result.experimental_only
                        if item.result is not None
                        else item.experimental_only
                    )
                ),
                "semantic_status": (
                    item.result.semantic_status
                    if item is not None and item.result is not None
                    else ""
                ),
                "evidence_status": (
                    item.result.evidence_status
                    if item is not None and item.result is not None
                    else ""
                ),
                "approval_status": (
                    item.result.approval_status
                    if item is not None and item.result is not None
                    else ""
                ),
                "learning_status": (
                    item.result.learning_status
                    if item is not None and item.result is not None
                    else ""
                ),
                "formula_closure_status": (
                    item.result.formula_closure.status
                    if item is not None
                    and item.result is not None
                    and item.result.formula_closure is not None
                    else ""
                ),
                "diagnosis_ids": (
                    [
                        diagnosis.diagnosis_id
                        for diagnosis in item.result.error_diagnoses
                    ]
                    if item is not None and item.result is not None
                    else []
                ),
                "rerun_delta": (
                    asdict(item.result.rerun_delta)
                    if item is not None
                    and item.result is not None
                    and item.result.rerun_delta is not None
                    else None
                ),
                "feedback_patch": (
                    asdict(item.result.feedback_patch)
                    if item is not None
                    and item.result is not None
                    and item.result.feedback_patch is not None
                    else None
                ),
                "review_basis": (
                    _review_basis_for_item(item) if item is not None else "pending"
                ),
                **(_pre_ir_manifest_fields(item) if item is not None else {}),
            }
        )

    counts = batch_unique_result_counts(results)
    _atomic_write_text(root / "result.txt", format_final_unique_batch_results(results))
    _atomic_write_text(
        root / "final_candidates.txt",
        format_final_unique_batch_summary(results),
    )
    _atomic_write_json(
        root / "manifest.json",
        {
            "schema_version": BATCH_OUTPUT_MANIFEST_SCHEMA_VERSION,
            "updated_at": _timestamp(),
            "counts": counts,
            "outcome": asdict(outcome),
            "unique": outcome.uniqueness_ok,
            "complete": outcome.complete,
            "release_ready": outcome.release_ready,
            "runtime_metrics": batch_runtime_metrics(results),
            "items": manifest_items,
        },
    )
    return counts


def _format_needs_review_results(
    results: list[BatchPipelineItemResult],
) -> str:
    lines = [
        "Traj-LTLf 待审查运行结果",
        "=" * 40,
        f"需求数量: {len(results)}",
        "",
    ]
    for index, item in enumerate(results, start=1):
        lines.extend(
            [
                "",
                "#" * 60,
                f"待审查项 {index}: {item.requirement_id}",
                "#" * 60,
                "",
                _format_item_report(item),
            ]
        )
    return "\n".join(lines)


def _format_needs_review_summary(
    results: list[BatchPipelineItemResult],
) -> str:
    lines = [
        "Traj-LTLf 待审查精简结果",
        f"需求数量: {len(results)}",
    ]
    for index, item in enumerate(results, start=1):
        lines.extend(
            [
                "",
                f"[{item.requirement_id}] 待审查项 {index}",
                _format_item_summary(item),
            ]
        )
    return "\n".join(lines)


def _format_blocked_error_report(item: object, error: dict[str, Any]) -> str:
    semantic = getattr(item, "semantic", None)
    traces = list(getattr(item, "traces", []) or [])
    coverage = getattr(item, "trace_coverage", None)
    review_records = list(getattr(item, "review_records", []) or [])
    invalid_completions = list(error.get("invalid_completions", []) or [])
    recent_candidates = list(error.get("recent_candidates", []) or [])
    suggested_stage = str(error.get("suggested_rerun_stage") or "pipeline")
    lines = [
        "Traj-LTLf 阻塞错误审查包",
        "=" * 40,
        "",
        "1. 自然语言需求",
        f"   {getattr(item, 'requirement', '')}",
        "",
        "2. 错误信息",
        "   status = blocked_error",
        f"   stage = {error.get('stage', '')}",
        f"   error_type = {error.get('error_type', '')}",
        f"   error = {error.get('error', '')}",
        f"   suggested_rerun_stage = {suggested_stage}",
    ]
    consistency_issues = list(error.get("consistency_issues", ()) or ())
    if consistency_issues:
        lines.append(f"   consistency issue count = {len(consistency_issues)}")
        for issue in consistency_issues:
            lines.append(
                "   - "
                f"trace={issue.get('trace_id', '')}; "
                f"type={issue.get('issue_type', '')}; "
                f"label={issue.get('expected', '')}; "
                f"ir={issue.get('ir_actual', '')}; "
                f"detail={issue.get('message', '')}"
            )
    lines.extend(["", "3. 已完成前置上下文"])
    if semantic is not None:
        lines.extend(
            [
                f"   AP = {{{', '.join(getattr(semantic, 'aps', ()) or ())}}}",
                f"   trigger = {getattr(semantic, 'trigger', '')}",
                f"   response = {getattr(semantic, 'response', '')}",
                f"   pattern = {getattr(semantic, 'pattern', '')}",
                f"   scope keyword = {getattr(semantic, 'scope_keyword', '')}",
                f"   temporal keyword = {getattr(semantic, 'temporal_keyword', '')}",
            ]
        )
    else:
        lines.append("   semantic = unavailable")
    if coverage is not None:
        lines.extend(
            [
                f"   trace coverage status = {getattr(coverage, 'status', '')}",
                f"   quota complete = {str(getattr(coverage, 'quota_complete', False)).lower()}",
                f"   semantic complete = {str(getattr(coverage, 'semantic_complete', False)).lower()}",
            ]
        )
    lines.append(f"   reviewed trace count = {len(traces)}")
    lines.append(f"   review record count = {len(review_records)}")
    if review_records:
        lines.append("")
        lines.append("4. 最近评审记录")
        for record in review_records[-5:]:
            lines.append(
                f"   {record.get('stage', '')} round {record.get('round', '')}: "
                f"approved={record.get('approved', '')} -> "
                f"{record.get('record_path', '')}"
            )
    if invalid_completions:
        lines.append("")
        lines.append("5. 无效 completion 记录")
        for record in invalid_completions:
            lines.append(
                "   - "
                f"candidate_id={record.get('candidate_id', '')}; "
                f"template={record.get('template_text', '')}; "
                f"formula={record.get('formula_text', '')}; "
                "invalid_reason="
                f"{record.get('invalid_reason', record.get('error', ''))}"
            )
    if recent_candidates:
        lines.append("")
        lines.append("6. 最近可用候选/模板")
        for candidate in recent_candidates:
            lines.append(f"   - {candidate}")
    lines.extend(
        [
            "",
            (
                "7. 建议处理"
                if invalid_completions and recent_candidates
                else "6. 建议处理"
                if invalid_completions or recent_candidates
                else "5. 建议处理"
            ),
            "   - 将该项作为待审项反馈，不写入最终候选。",
            f"   - 修正反馈后建议从 {suggested_stage} 阶段重跑。",
            "   - 若错误来自单个候选，应优先将该候选标记为 invalid，而不是放弃整项需求。",
        ]
    )
    return "\n".join(lines)


def _format_semantic_policy_pending_report(
    item: object,
    pending: dict[str, Any],
) -> str:
    return "\n".join([
        "Traj-LTLf 专家有限轨迹边界审核包",
        "=" * 40,
        "",
        "1. 自然语言需求",
        f"   {getattr(item, 'requirement', '')}",
        "",
        "2. 当前状态",
        "   普通语言切分与读法审核已完成。",
        "   当前不是阻塞错误，也不是再次向普通用户确认语义。",
        f"   status = {pending.get('status', '')}",
        f"   branch_set_id = {pending.get('branch_set_id', '')}",
        f"   candidate_branch_count = {pending.get('candidate_branch_count', '')}",
        f"   query_count = {pending.get('query_count', '')}",
        f"   expert_review_path = {pending.get('review_path', '')}",
    ])


def _recent_invalid_candidates(invalid_completions: list[dict]) -> list[str]:
    candidates: list[str] = []
    for record in invalid_completions:
        for key in ("formula_text", "template_text"):
            value = str(record.get(key, "")).strip()
            if value and value not in candidates:
                candidates.append(value)
    return candidates[:10]

def _format_blocked_error_summary(item: object, error: dict[str, Any]) -> str:
    return "\n".join(
        [
            "需求:",
            f"  {getattr(item, 'requirement', '')}",
            "当前状态:",
            "  blocked_error",
            "出错阶段:",
            f"  {error.get('stage', '')}",
            "错误:",
            f"  {error.get('error_type', '')}: {error.get('error', '')}",
            "建议重跑阶段:",
            f"  {error.get('suggested_rerun_stage', 'pipeline')}",
            "最终候选:",
            "  暂不输出（blocked_error）",
        ]
    )


def _suggest_rerun_stage_from_error(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".lower()
    if "blocked_by_unexecutable_ir" in text or "responsible_stage=semantic_ir" in text:
        return "semantic"
    if "trace_label_ir_mismatch" in text or "trace_label_error" in text:
        return "trace"
    if any(
        token in text
        for token in (
            "compiler_contract_violation",
            "compiler_semantics_mismatch",
            "completion_formula_assignment_mismatch",
        )
    ):
        return "completion"
    if any(token in text for token in ("segmentation", "segment", "切分", "重构")):
        return "segmentation"
    if any(token in text for token in ("plain reading", "plain_reading", "普通读法", "解释")):
        return "plain_reading"
    if any(token in text for token in ("completion", "candidate", "候选", "补全")):
        return "completion"
    if any(token in text for token in ("skeleton", "骨架")):
        return "skeleton"
    if any(token in text for token in ("trace", "轨迹")):
        return "trace"
    if any(token in text for token in ("semantic", "nlu", "语义")):
        return "semantic"
    if any(token in text for token in ("selection", "select", "最终选择")):
        return "selection"
    return "pipeline"


_RECOVERABLE_NEEDS_REVIEW_CODES = (
    "IR_AP_IDENTITY_COLLAPSE",
    "IR_AP_CONTEXT_STALE",
    "IR_AP_CATALOG_AUTHORITY_MISSING",
    "IR_RELATION_NON_CONTINGENT_REVIEW_REQUIRED",
    "IR_RELATION_FEASIBILITY_RESOURCE_LIMIT",
    "IR_RELATION_FEASIBILITY_UNSUPPORTED",
    "IR_WITNESS_UNSAT_IN_DOMAIN",
    "IR_WITNESS_RESOURCE_LIMIT",
    "IR_WITNESS_UNSUPPORTED_NODE",
)


def _recoverable_needs_review_reason(error: BaseException) -> str:
    values = [str(getattr(error, "code", "")), str(error)]
    diagnostics = getattr(error, "diagnostics", None)
    if isinstance(diagnostics, dict):
        values.extend(
            str(value)
            for key, value in diagnostics.items()
            if key in {"reason_code", "solver_status", "error", "findings"}
        )
    return next(
        (
            code
            for code in _RECOVERABLE_NEEDS_REVIEW_CODES
            if any(code in value for value in values)
        ),
        "",
    )


def _load_cached_pipeline_result(
    *,
    checkpoint_dir: str | None,
    item,
    classified_output_dir: str | None,
    final_unique_results: bool = False,
) -> BatchPipelineItemResult | None:
    segmentation = getattr(item.semantic, "surface_segmentation", None)
    reading = getattr(item.semantic, "plain_reading_review", None)
    payload = load_pipeline_result_checkpoint(
        checkpoint_dir,
        item.requirement_id,
        expected_ir_version=(
            item.semantic.faithful_ir.ir_version
            if item.semantic.faithful_ir is not None
            else None
        ),
        expected_source_hash=(
            segmentation.source_hash
            if segmentation is not None
            else source_hash(item.requirement)
        ),
        expected_segmentation_id=(
            segmentation.segmentation_id if segmentation is not None else None
        ),
        expected_reading_review_id=(
            reading.reading_review_id if reading is not None else None
        ),
        expected_plain_reading_decision_fingerprint=(
            decision_fingerprint(reading) if reading is not None else None
        ),
    )
    if payload is None or payload.get("requirement") != item.requirement:
        return None
    selection_status = str(payload.get("selection_status", "")).strip()
    expected_formula = _guided_expected_formula_text(item)
    if expected_formula:
        cached_classification = str(payload.get("classification", "")).strip()
        cached_formula = _canonical_formula_text(
            str(payload.get("selected_formula", ""))
        )
        if (
            cached_classification != "complete"
            or selection_status not in {"selected", "selected_reference_only"}
            or cached_formula != expected_formula
        ):
            return None
    classification = str(payload.get("classification", "")).strip()
    if payload.get("legacy_read_only"):
        classification = "needs_review"
    if (
        classification == "complete"
        and selection_status in {"ambiguous_candidates", "needs_review"}
    ):
        classification = "needs_review"
    if classification not in FINAL_CLASSIFICATIONS:
        classification = str(
            payload.get("trace_coverage", {}).get("status", "complete")
        )
    if classification not in FINAL_CLASSIFICATIONS:
        classification = "complete"
    if final_unique_results and classification not in {
        "complete",
        "complete_with_corrections",
    }:
        return None
    degraded_record = degraded_result_from_checkpoint(
        payload.get("degraded_record")
    )
    cached_trace_policy = _trace_evidence_policy_from_mapping(
        payload.get("trace_evidence_policy")
    )
    if classification in {"degraded_unproven", "degraded_proven"}:
        existing_status, existing_comments = _load_existing_expert_review(
            classified_output_dir,
            item.requirement_id,
        )
        checkpoint_status = str(payload.get("expert_review_status", ""))
        effective_status = existing_status or checkpoint_status or "pending"
        if existing_status:
            update_pipeline_expert_review_status(
                checkpoint_dir,
                requirement_id=item.requirement_id,
                expert_review_status=effective_status,
                expert_comments=existing_comments,
            )
        if effective_status == "retry_requested":
            return None
        if degraded_record is not None:
            degraded_record = replace(
                degraded_record,
                expert_review_status=effective_status,
                expert_comments=existing_comments or degraded_record.expert_comments,
            )
    return BatchPipelineItemResult(
        requirement_id=item.requirement_id,
        requirement=item.requirement,
        result=None,
        report_text=str(payload.get("report_text", "")),
        summary_text=str(payload.get("summary_text", "")),
        review_records=list(payload.get("review_records", [])),
        classification=classification,
        degraded_record=degraded_record,
        trace_evidence_policy=cached_trace_policy,
        experimental_only=bool(
            payload.get("experimental_only", cached_trace_policy.experimental_only)
        ),
    )


def _trace_evidence_policy_from_mapping(value: object) -> TraceEvidencePolicy:
    if not isinstance(value, dict):
        return TraceEvidencePolicy()
    return TraceEvidencePolicy(
        boundary_source=str(value.get("boundary_source", "generated")),
        boundary_fixture_path=str(value.get("boundary_fixture_path", "")),
        boundary_fixture_sha256=str(value.get("boundary_fixture_sha256", "")),
        boundary_fixture_schema_version=int(
            value.get("boundary_fixture_schema_version", 0) or 0
        ),
    )


def _validate_guided_pipeline_result(*, item, result: PipelineResult) -> None:
    expected_formula = _guided_expected_formula_text(item)
    if not expected_formula:
        return
    selected = result.selected
    if selected is None:
        provisional = result.provisional_candidate
        if provisional is not None and str(provisional.formula) == expected_formula:
            missing = ", ".join(result.ir_coverage.missing_obligation_ids) or "<none>"
            reasons = "; ".join(
                (
                    *result.operator_resolution.reasons,
                    *result.ir_coverage.reasons,
                    *result.selection_review_notes,
                )
            ) or "unspecified finalization gate"
            raise RuntimeError(
                "guided warmup expected candidate passed formula scoring but remained "
                "provisional: "
                f"candidate={expected_formula}, "
                f"operator_resolution={result.operator_resolution.status}, "
                f"missing_obligations={missing}, reasons={reasons}"
            )
        raise RuntimeError(
            "guided warmup pipeline did not select the expected final candidate: "
            f"{expected_formula}"
        )
    actual_formula = str(selected.formula)
    if actual_formula != expected_formula:
        raise RuntimeError(
            "guided warmup pipeline selected an unexpected final candidate: "
            f"selected={actual_formula}, expected={expected_formula}"
        )


def _guided_expected_formula_text(item) -> str:
    guided_context = getattr(item, "guided_context", None)
    if guided_context is None:
        return ""
    try:
        formula = guided_context.expected_final_formula()
    except ValueError as exc:
        raise RuntimeError(
            f"{item.requirement_id} 的 Expected Final Candidate 无法解析。"
        ) from exc
    if formula is None:
        raise RuntimeError(
            f"{item.requirement_id} 缺少 Expected Final Candidate。"
        )
    return str(formula)


def _canonical_formula_text(text: str) -> str:
    if not text.strip():
        return ""
    try:
        return str(parse_formula(text))
    except ValueError:
        return ""


def _write_incremental_outputs(
    results: list[BatchPipelineItemResult],
    *,
    partial_result_path: str | None,
    partial_summary_path: str | None,
    classified_output_dir: str | None,
    reviewed_items: list[object],
) -> None:
    if partial_result_path:
        _atomic_write_text(
            Path(partial_result_path),
            format_batch_pipeline_results(results),
        )
    if partial_summary_path:
        _atomic_write_text(
            Path(partial_summary_path),
            format_batch_pipeline_summary(results),
        )
    if classified_output_dir:
        write_classified_batch_outputs(
            results,
            output_dir=classified_output_dir,
            expected_items=reviewed_items,
        )


def write_classified_batch_outputs(
    results: list[BatchPipelineItemResult],
    *,
    output_dir: str | Path,
    expected_items: list[object] | None = None,
) -> dict[str, int]:
    """Atomically write normal, degraded, rejected, and manifest artifacts."""

    root = Path(output_dir)
    source_items = expected_items or results
    outcome = compute_batch_completeness(
        results,
        expected_items=list(source_items),
    )
    degraded_dir = root / "degraded"
    needs_review_dir = root / "needs_review"
    items_dir = degraded_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    needs_review_dir.mkdir(parents=True, exist_ok=True)

    normal_results = [
        item
        for item in results
        if _item_classification(item) in {"complete", "complete_with_corrections"}
    ]
    needs_review_results = [
        item for item in results if _item_classification(item) == "needs_review"
    ]
    blocked_error_results = [
        item for item in results if _item_classification(item) == "blocked_error"
    ]
    review_queue_results = [
        item
        for item in results
        if _item_classification(item)
        in {
            "needs_review",
            "blocked_error",
        }
    ]
    degraded_results = [
        item
        for item in results
        if _item_classification(item)
        in {"degraded_unproven", "degraded_proven"}
    ]
    rejected_results = [
        item for item in results if _item_classification(item) == "rejected"
    ]
    _atomic_write_text(
        root / "normal_result.txt",
        format_batch_pipeline_results(normal_results),
    )
    _atomic_write_text(
        root / "normal_summary.txt",
        format_batch_pipeline_summary(normal_results),
    )
    _atomic_write_text(
        needs_review_dir / "needs_review_result.txt",
        _format_needs_review_results(review_queue_results),
    )
    _atomic_write_text(
        needs_review_dir / "needs_review_summary.txt",
        _format_needs_review_summary(review_queue_results),
    )
    degraded_records: list[DegradedResultRecord] = []
    degraded_report_sections: list[str] = []
    for item in degraded_results:
        record = item.degraded_record
        if record is None and item.result is not None:
            record = build_degraded_result_record(item.requirement_id, item.result)
        if record is None:
            continue
        degraded_records.append(record)
        degraded_report_sections.append(
            format_degraded_result(record, _format_item_report(item))
        )
        _atomic_write_json(items_dir / f"{_safe_requirement_id(item.requirement_id)}.json", asdict(record))

    _atomic_write_text(
        degraded_dir / "degraded_result.txt",
        "\n\n" + ("\n\n" + "#" * 72 + "\n\n").join(degraded_report_sections)
        if degraded_report_sections
        else "Traj-LTLf 降级结果专家审查包\n降级项数量: 0\n",
    )
    _atomic_write_text(
        degraded_dir / "degraded_summary.txt",
        format_degraded_summary(degraded_records),
    )

    processed_ids = {item.requirement_id for item in results}
    degraded_ids = {item.requirement_id for item in degraded_results}
    expected_ids = {
        str(getattr(item, "requirement_id", ""))
        for item in (expected_items or [])
        if getattr(item, "requirement_id", "")
    }
    for path in items_dir.glob("*.json"):
        requirement_id = path.stem
        if requirement_id in processed_ids and requirement_id not in degraded_ids:
            path.unlink()
        elif expected_ids and requirement_id not in expected_ids:
            path.unlink()

    degraded_manifest = {
        "schema_version": 1,
        "updated_at": _timestamp(),
        "count": len(degraded_records),
        "items": [
            {
                "requirement_id": record.requirement_id,
                "classification": record.trace_coverage.get("status", ""),
                "item_file": f"items/{_safe_requirement_id(record.requirement_id)}.json",
                "expert_review_status": record.expert_review_status,
                "persistent_rag_written": record.persistent_rag_written,
            }
            for record in degraded_records
        ],
    }
    _atomic_write_json(degraded_dir / "manifest.json", degraded_manifest)

    result_by_id = {item.requirement_id: item for item in results}
    manifest_items: list[dict[str, Any]] = []
    for source in source_items:
        requirement_id = str(getattr(source, "requirement_id", ""))
        item = result_by_id.get(requirement_id)
        classification = _item_classification(item) if item is not None else "pending"
        entry: dict[str, Any] = {
            "requirement_id": requirement_id,
            "requirement": (
                item.requirement
                if item is not None and item.requirement
                else str(getattr(source, "requirement", ""))
            ),
            "classification": classification,
            "result_status": (
                "final" if _final_formula_text(item) else classification
            ),
        }
        if item is not None and item.result is not None:
            entry.update(
                {
                    "experiment_mode": item.result.experiment_mode,
                    "oracle_assisted": item.result.oracle_assisted,
                    "diagnostic_only": item.result.diagnostic_only,
                    "trace_evidence_policy": asdict(
                        item.result.trace_evidence_policy
                    ),
                    "experimental_only": item.result.experimental_only,
                    "canonical_trace_coverage": asdict(
                        item.result.canonical_trace_coverage
                    ),
                    **_pre_ir_manifest_fields(item),
                }
            )
        elif item is not None:
            entry.update(
                {
                    "trace_evidence_policy": asdict(item.trace_evidence_policy),
                    "experimental_only": item.experimental_only,
                }
            )
        if classification in {"complete", "complete_with_corrections"}:
            entry.update(
                {
                    "result_file": "normal_result.txt",
                    "summary_file": "normal_summary.txt",
                }
            )
        elif classification == "needs_review":
            entry.update(
                {
                    "result_file": "needs_review/needs_review_result.txt",
                    "summary_file": "needs_review/needs_review_summary.txt",
                }
            )
        elif classification == "blocked_error":
            entry.update(
                {
                    "result_file": "needs_review/needs_review_result.txt",
                    "summary_file": "needs_review/needs_review_summary.txt",
                    "error": item.error if item is not None else None,
                    "suggested_rerun_stage": (
                        (item.error or {}).get("suggested_rerun_stage", "pipeline")
                        if item is not None
                        else "pipeline"
                    ),
                }
            )
        elif classification in {"degraded_unproven", "degraded_proven"}:
            record = item.degraded_record if item is not None else None
            entry.update(
                {
                    "result_file": "degraded/degraded_result.txt",
                    "summary_file": "degraded/degraded_summary.txt",
                    "item_file": (
                        f"degraded/items/{_safe_requirement_id(requirement_id)}.json"
                    ),
                    "expert_review_status": (
                        record.expert_review_status if record is not None else "pending"
                    ),
                }
            )
        elif classification == "rejected":
            entry["error"] = item.error if item is not None else None
        manifest_items.append(entry)

    counts = {
        "normal": len(normal_results),
        "needs_review": len(needs_review_results),
        "degraded": len(degraded_results),
        "blocked_error": len(blocked_error_results),
        "rejected": len(rejected_results),
        "pending": sum(
            1 for entry in manifest_items if entry["classification"] == "pending"
        ),
    }
    feedback_items = [
        packet
        for item in results
        if (packet := _build_review_feedback_packet(item)) is not None
    ]
    _atomic_write_json(
        root / "review_feedback.json",
        {
            "schema_version": 2,
            "updated_at": _timestamp(),
            "items": feedback_items,
        },
    )
    for entry in manifest_items:
        if any(
            packet["requirement_id"] == entry["requirement_id"]
            for packet in feedback_items
        ):
            entry["feedback_file"] = "review_feedback.json"
    _atomic_write_json(
        root / "manifest.json",
        {
            "schema_version": BATCH_OUTPUT_MANIFEST_SCHEMA_VERSION,
            "updated_at": _timestamp(),
            "counts": counts,
            "outcome": asdict(outcome),
            "unique": outcome.uniqueness_ok,
            "complete": outcome.complete,
            "release_ready": outcome.release_ready,
            "items": manifest_items,
        },
    )
    return counts


def _build_review_feedback_packet(
    item: BatchPipelineItemResult,
) -> dict[str, Any] | None:
    classification = _item_classification(item)
    if classification not in {
        "needs_review",
        "degraded_unproven",
        "blocked_error",
    }:
        return None

    result = item.result
    diagnosis = _primary_error_diagnosis(result)
    constraints: list[str] = []
    if result is not None:
        allowed_aps = list(result.semantic.aps or tuple(result.ap))
        if allowed_aps:
            constraints.append(
                "完整公式只能使用 AP: " + ", ".join(allowed_aps)
            )
        constraints.extend(result.selection_review_notes)
        constraints.extend(result.trace_coverage.coverage_debt)
    else:
        allowed_aps = list((item.error or {}).get("allowed_aps", []))
        if allowed_aps:
            constraints.append(
                "完整公式只能使用 AP: " + ", ".join(allowed_aps)
            )
        if classification == "blocked_error" and (
            "epsilon" in str((item.error or {}).get("error", "")).lower()
        ):
            constraints.append(
                "epsilon 只能作为 H_xxx assignment，不能作为完整公式"
            )

    if classification == "blocked_error":
        start_stage = str(
            (item.error or {}).get("suggested_rerun_stage", "pipeline")
        )
        feedback = str((item.error or {}).get("error", "pipeline error"))
    elif classification == "degraded_unproven":
        start_stage = "trace"
        feedback = (
            result.trace_coverage.reason
            if result is not None
            else "轨迹覆盖不足，需要补充和复审轨迹。"
        )
    else:
        start_stage = (
            _batch_stage_for_responsible_stage(diagnosis.responsible_stage)
            if diagnosis is not None
            else "selection"
        )
        feedback = (
            "；".join(diagnosis.evidence_summary)
            if diagnosis is not None
            else "；".join(result.selection_review_notes)
            if result is not None and result.selection_review_notes
            else "最终候选未通过唯一选择，需要结合审查反馈重跑。"
        )

    if start_stage == "completion" and (
        result is None or not result.skeletons
    ):
        start_stage = "skeleton"
    if start_stage == "selection" and (
        result is None or not result.completion_decisions
    ):
        start_stage = "completion" if result is not None and result.skeletons else "skeleton"

    expected_changes = _expected_artifact_changes_for_start_stage(start_stage)
    previous_ir_version = (
        result.semantic.faithful_ir.ir_version
        if result is not None and result.semantic.faithful_ir is not None
        else ""
    )
    feedback_patch = None
    if diagnosis is not None:
        proposed_patch = propose_feedback_patch(
            diagnosis,
            source_ir_version=previous_ir_version,
        )
        feedback_patch = asdict(
            replace(
                proposed_patch,
                rerun_start_stage=start_stage,
                expected_changed_artifacts=expected_changes,
            )
        )

    affected_ir_node_ids = (
        list(diagnosis.affected_ir_node_ids)
        if diagnosis is not None
        else list(result.ir_coverage.missing_ir_node_ids)
        if result is not None
        else []
    )
    owner_stage = (
        diagnosis.responsible_stage
        if diagnosis is not None
        else str((item.error or {}).get("stage", start_stage))
    )
    reason_code = (
        diagnosis.error_class.upper()
        if diagnosis is not None
        else str((item.error or {}).get("reason_code") or classification).upper()
    )
    reviewed_nl_evidence = tuple(
        dict.fromkeys(
            value
            for value in (
                feedback,
                *(diagnosis.evidence_summary if diagnosis is not None else ()),
            )
            if value
        )
    )
    challenge_ids = tuple(
        str(value)
        for value in (item.error or {}).get("challenge_ids", ())
    )
    dispute_ids = tuple(
        str(value)
        for value in (item.error or {}).get("dispute_ids", ())
    )
    request = FeedbackRerunRequest(
        requirement_id=item.requirement_id,
        owner_stage=owner_stage,
        restart_stage=start_stage,
        reason_code=reason_code,
        source_ir_version=previous_ir_version,
        source_artifact_fingerprints=(
            result.artifact_fingerprints.as_dict() if result is not None else {}
        ),
        affected_ir_node_ids=tuple(affected_ir_node_ids),
        dispute_ids=dispute_ids,
        challenge_ids=challenge_ids,
        reviewed_nl_evidence=reviewed_nl_evidence,
        expected_invalidation_set=expected_changes,
        restart_prerequisites=_restart_prerequisites_for_stage(start_stage),
        fresh_approval_requirement="required",
        feedback=feedback,
        constraints=tuple(dict.fromkeys(value for value in constraints if value)),
        output_policy="fresh_independent_approval",
        ir_global_problem=not bool(affected_ir_node_ids),
        diagnosis_id=diagnosis.diagnosis_id if diagnosis is not None else "",
        source_run_id=result.source_run_id if result is not None else "",
        fresh_judge_required=True,
        feedback_patch=feedback_patch,
        frozen_upstream=bool(
            diagnosis is not None
            and diagnosis.error_class == "blocked_by_frozen_upstream"
        ),
    )
    return {
        "schema_version": request.schema_version,
        "requirement_id": item.requirement_id,
        "action": "rerun_with_feedback",
        "owner_stage": request.owner_stage,
        "restart_stage": request.restart_stage,
        "reason_code": request.reason_code,
        "source_ir_version": request.source_ir_version,
        "source_artifact_fingerprints": request.source_artifact_fingerprints or {},
        "affected_ir_node_ids": list(request.affected_ir_node_ids),
        "dispute_ids": list(request.dispute_ids),
        "challenge_ids": list(request.challenge_ids),
        "reviewed_nl_evidence": list(request.reviewed_nl_evidence),
        "expected_invalidation_set": list(request.expected_invalidation_set),
        "restart_prerequisites": list(request.restart_prerequisites),
        "fresh_approval_requirement": request.fresh_approval_requirement,
        "feedback": feedback,
        "constraints": list(dict.fromkeys(value for value in constraints if value)),
        "status": classification,
        "output_policy": "fresh_independent_approval",
        "diagnosis_id": diagnosis.diagnosis_id if diagnosis is not None else "",
        "source_run_id": result.source_run_id if result is not None else "",
        "source_error_ids": (
            [item.diagnosis_id for item in result.error_diagnoses]
            if result is not None
            else []
        ),
        "prior_independence_ids": (
            [
                item.diagnoser_independence_id
                for item in result.error_diagnoses
                if item.diagnoser_independence_id
            ]
            if result is not None
            else []
        ),
        "fresh_judge_required": True,
        "feedback_patch": feedback_patch,
        "frozen_upstream": bool(
            diagnosis is not None
            and diagnosis.error_class == "blocked_by_frozen_upstream"
        ),
        "ir_global_problem": request.ir_global_problem,
    }


def _restart_prerequisites_for_stage(stage: str) -> tuple[str, ...]:
    return {
        "plain_reading": ("approved_segmentation",),
        "semantic": ("approved_plain_reading",),
        "trace": ("approved_semantic_ir",),
        "skeleton": ("approved_semantic_ir", "approved_trace_evidence"),
        "completion": ("approved_skeleton",),
        "selection": ("approved_completion",),
    }.get(str(stage), ())


def _primary_error_diagnosis(result: PipelineResult | None):
    if result is None or not result.error_diagnoses:
        return None
    priority = {
        "blocked_by_frozen_upstream": 0,
        "no_effective_delta": 1,
        "ir_ambiguity_state_mismatch": 2,
        "trace_coverage_gap": 3,
    }
    return min(
        result.error_diagnoses,
        key=lambda item: priority.get(item.error_class, 10),
    )


def _batch_stage_for_responsible_stage(stage: str) -> str:
    return {
        "segmentation": "segmentation",
        "plain_reading": "plain_reading",
        "nlu": "semantic",
        "semantic_ir": "semantic",
        "semantic_approval": "semantic",
        "trace_obligation_planning": "trace",
        "trace_adjudication": "trace",
        "skeleton": "skeleton",
        "compiler": "skeleton",
        "completion": "completion",
        "selection": "selection",
        "approval": "selection",
        "manual_semantic_review": "semantic",
        "manual_approval": "selection",
    }.get(str(stage), "pipeline")


def _diagnosis_stage_for_batch_stage(stage: str) -> str:
    """Translate the coarse batch restart stage into a routed audit owner."""

    return {
        "pipeline": "plain_reading",
        "segmentation": "segmentation",
        "plain_reading": "plain_reading",
        "semantic": "semantic_ir",
        "trace": "trace_obligation_planning",
        "skeleton": "skeleton",
        "completion": "completion",
        "selection": "selection",
        "approval": "approval",
    }.get(str(stage), "plain_reading")


def _attach_feedback_patch_outcome(
    *,
    result: PipelineResult,
    patch_payload: object,
    delta_report: object,
    reviewer_id: str,
) -> PipelineResult:
    """Persist owning-stage review of a formal feedback patch proposal."""

    if not isinstance(patch_payload, dict):
        return result
    patch = feedback_patch_from_mapping(patch_payload)
    if patch.review_status == "pending_stage_review":
        status = str(getattr(delta_report, "status", ""))
        changed = set(getattr(delta_report, "actual_changed_artifacts", ()))
        expected = set(patch.expected_changed_artifacts)
        approved = (
            status in {"effective_delta", "needs_independent_approval"}
            and bool(changed)
            and (not expected or bool(changed.intersection(expected)))
        )
        patch = review_feedback_patch(
            patch,
            approved=approved,
            reviewer_id=reviewer_id,
            review_basis=(
                "owning stage reran its normal validation and review contract",
                f"rerun delta status={status}",
                "changed artifacts=" + ", ".join(sorted(changed)),
            ),
            current_ir_version=patch.source_ir_version,
        )
    return replace(result, feedback_patch=patch)


def _attach_rerun_delta_outcome(
    *,
    result: PipelineResult,
    delta_report: object,
    requirement_id: str,
) -> PipelineResult:
    """Attach the delta and turn non-progress into an upstream audit blocker.

    A no-op rerun is not merely a reporting detail: the repeated stage is
    treated as frozen for this diagnosis so the route resolver moves
    responsibility to the first editable upstream owner.  Any candidate is
    retained as provisional evidence, but release approval is cleared.
    """

    attached = replace(result, rerun_delta=delta_report)
    status = str(getattr(delta_report, "status", ""))
    if status not in {"no_effective_delta", "blocked_by_frozen_upstream"}:
        return attached

    source_stage = _diagnosis_stage_for_batch_stage(
        str(getattr(delta_report, "start_stage", "pipeline"))
    )
    error_class = (
        "no_effective_delta"
        if status == "no_effective_delta"
        else "blocked_by_frozen_upstream"
    )
    diagnosis = build_error_diagnosis(
        requirement_id=requirement_id,
        source_run_id=(
            attached.source_run_id
            or "rerun-audit-" + str(getattr(delta_report, "rerun_id", "unknown"))
        ),
        source_stage=source_stage,
        candidate_information_exposed=False,
        diagnoser_role="rerun_delta_auditor",
        diagnoser_independence_id=(
            "rerun-delta-auditor:"
            + str(getattr(delta_report, "rerun_id", "unknown"))
        ),
        error_class=error_class,
        confusion_targets=("identical_rerun",),
        frozen_stages=((source_stage,) if status == "no_effective_delta" else ()),
        confidence="high",
        diagnosis_id=(
            "diagnosis-"
            + str(getattr(delta_report, "rerun_id", "unknown"))
            + ("-no-delta" if status == "no_effective_delta" else "-frozen")
        ),
    )
    diagnoses = tuple(attached.error_diagnoses)
    if all(item.diagnosis_id != diagnosis.diagnosis_id for item in diagnoses):
        diagnoses = (*diagnoses, diagnosis)

    retained_candidate = (
        attached.provisional_candidate
        or attached.selected
        or attached.fallback_candidate
    )
    blocked = replace(
        attached,
        selected=None,
        provisional_candidate=retained_candidate,
        candidate_status=("provisional" if retained_candidate is not None else "none"),
        selection_status=status,
        selection_review_notes=tuple(
            dict.fromkeys(
                (
                    *attached.selection_review_notes,
                    *tuple(str(reason) for reason in getattr(delta_report, "reasons", ())),
                    "unchanged rerun stopped; responsibility routed upstream",
                )
            )
        ),
        approval_status=(
            "blocked_no_effective_delta"
            if status == "no_effective_delta"
            else "blocked_by_frozen_upstream"
        ),
        learning_status="writeback_blocked",
        automatic_finalization_allowed=False,
        final_approval=None,
        error_diagnoses=diagnoses,
    )
    return replace(
        blocked,
        artifact_fingerprints=build_artifact_fingerprint_set(blocked),
    )

def batch_classification_counts(
    results: list[BatchPipelineItemResult],
) -> dict[str, int]:
    return {
        "normal": sum(
            1
            for item in results
            if _item_classification(item) in {"complete", "complete_with_corrections"}
        ),
        "needs_review": sum(
            1
            for item in results
            if _item_classification(item) == "needs_review"
        ),
        "degraded": sum(
            1
            for item in results
            if _item_classification(item)
            in {"degraded_unproven", "degraded_proven"}
        ),
        "blocked_error": sum(
            1 for item in results if _item_classification(item) == "blocked_error"
        ),
        "rejected": sum(
            1 for item in results if _item_classification(item) == "rejected"
        ),
    }


def batch_unique_result_counts(
    results: list[BatchPipelineItemResult],
) -> dict[str, int]:
    final_formula_count = sum(1 for item in results if _final_formula_text(item))
    provisional_count = sum(
        1
        for item in results
        if not _final_formula_text(item) and _provisional_candidate_text(item)
    )
    blocked_count = sum(
        1 for item in results if _item_classification(item) == "blocked_error"
    )
    needs_review_count = sum(
        1
        for item in results
        if _item_classification(item) == "needs_review"
    )
    return {
        "total": len(results),
        "finalized": final_formula_count,
        "final_formula_count": final_formula_count,
        "provisional_candidate_count": provisional_count,
        "blocked_error_count": blocked_count,
        "needs_review_count": needs_review_count,
    }


def batch_runtime_metrics(
    results: list[BatchPipelineItemResult],
) -> dict[str, Any]:
    """Operational runtime metrics; gold-dependent metrics stay explicitly offline."""

    concrete = [item.result for item in results if item.result is not None]
    automatic = [r for r in concrete if r.delivery_policy == "always_output"]
    if automatic:
        # Include absent results in the fixed denominator; never conflate a
        # delivered approximation with strict approval or correctness.
        total = len(results)
        delivered = sum(r.selected is not None for r in automatic)
        normal = sum(r.normal_flow_completed for r in automatic)
        emergency = sum(r.selection_method == "emergency_fallback" for r in automatic)
        return {
            "output_contract_version": "always-output-v1",
            "fixed_denominator": total, "always_output_records": len(automatic),
            "strict_records": len(concrete) - len(automatic),
            "output_count": delivered, "output_rate": delivered / total if total else 0,
            "normal_flow_count": normal, "normal_flow_rate": normal / total if total else 0,
            "normal_flow_definition": "all working stages without degradation and complete candidate enumeration",
            "representation_valid_count": sum(r.representation_valid for r in automatic),
            "semantic_contract_pass_count": sum(r.semantic_contract_passed for r in automatic),
            "evidence_pass_count": sum(bool(r.evidence_pass_count) for r in automatic),
            "source_binding_count": sum(r.source_binding_status == 'bound' for r in automatic),
            "emergency_output_count": emergency, "emergency_output_rate": emergency / total if total else 0,
            "unique_evidence_pass_count": sum(r.strict_evidence_pass for r in automatic),
            "scored_fallback_count": sum(r.selection_method == "scored_fallback" for r in automatic),
            "correctness": "requires_independent_offline_scoring",
        }
    total = len(concrete)
    candidate_pass = sum(
        1
        for result in concrete
        if any(
            evaluation.positive_pass
            and evaluation.negative_pass
            and evaluation.boundary_pass
            and evaluation.closure_report is not None
            and evaluation.closure_report.passed
            for evaluation in result.evaluations
        )
    )
    closure_violations = sum(
        1
        for result in concrete
        for evaluation in result.evaluations
        if evaluation.closure_report is not None
        and not evaluation.closure_report.passed
    )
    closure_audits = sum(
        1
        for result in concrete
        for evaluation in result.evaluations
        if evaluation.closure_report is not None
    )
    reruns = [result for result in concrete if result.run_origin == "feedback_rerun"]
    delta_reports = [result.rerun_delta for result in reruns if result.rerun_delta is not None]
    effective = sum(
        1
        for report in delta_reports
        if report.status in {"effective_delta", "needs_independent_approval"}
    )
    identical = sum(
        1 for report in delta_reports if report.status == "no_effective_delta"
    )
    corrected = sum(1 for report in delta_reports if report.resolved_errors)
    reapproved = sum(
        1
        for result in reruns
        if result.final_approval is not None
        and result.final_approval.context_isolated
        and bool(result.final_approval.independence_id)
    )
    provisional_supported = sum(
        1
        for result in concrete
        if result.candidate_status == "provisional"
        and result.formula_closure is not None
        and result.formula_closure.passed
        and result.evidence_status == "complete"
    )
    provisional_total = sum(
        1 for result in concrete if result.candidate_status == "provisional"
    )
    confusion_counts: dict[str, int] = {}
    required_confusions: dict[str, int] = {}
    covered_confusions: dict[str, int] = {}
    for result in concrete:
        required_for_result = set(
            result.trace_obligation_plan.confusion_targets
            if result.trace_obligation_plan is not None
            else ()
        )
        covered_for_result: set[str] = set()
        for trace in result.traces:
            for confusion in trace.semantics.confusion_targets:
                confusion_counts[confusion] = confusion_counts.get(confusion, 0) + 1
                if trace.annotation_locked and trace.label_source in {
                    "human",
                    "independent_judge",
                    "approved_ir",
                    "warmup_fixture",
                }:
                    covered_for_result.add(confusion)
        for confusion in required_for_result:
            required_confusions[confusion] = required_confusions.get(confusion, 0) + 1
            if confusion in covered_for_result:
                covered_confusions[confusion] = covered_confusions.get(confusion, 0) + 1

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "candidate_semantic_pass_rate": rate(candidate_pass, total),
        "formula_closure_violation_count": closure_violations,
        "formula_closure_violation_rate": rate(closure_violations, closure_audits),
        "correct_provisionally_supported_retention_count": provisional_supported,
        "correct_provisionally_supported_retention_rate": rate(
            provisional_supported,
            provisional_total,
        ),
        "feedback_rerun_effective_delta_rate": rate(effective, len(delta_reports)),
        "feedback_correction_rate": rate(corrected, len(delta_reports)),
        "identical_rerun_rate": rate(identical, len(delta_reports)),
        "independent_reapproval_pass_rate": rate(reapproved, len(reruns)),
        "trace_discrimination_count_by_confusion": dict(sorted(confusion_counts.items())),
        "trace_discrimination_rate_by_confusion": {
            confusion: rate(
                covered_confusions.get(confusion, 0),
                required_count,
            )
            for confusion, required_count in sorted(required_confusions.items())
        },
        "gold_dependent_offline_metrics": {
            "final_precision": None,
            "final_recall_among_independently_supported_candidates": None,
            "false_gate_rejection_rate": None,
            "reason": "requires an isolated offline gold evaluator",
        },
    }


def _format_final_count_line(counts: dict[str, int]) -> str:
    return (
        "已输出 LTLf 公式: "
        f"{counts['final_formula_count']}/{counts['total']}"
    )


def _classification_for_result(result: PipelineResult) -> str:
    if result.provisional_candidate is not None:
        return "complete_with_corrections"
    status = result.trace_coverage.status
    if status == "complete" and result.selected is None:
        return "needs_review"
    if status in {"complete", "degraded_unproven", "degraded_proven"}:
        if status == "complete" and _result_has_guided_fallback(result):
            return "complete_with_corrections"
        return status
    return "rejected" if status == "rejected" else "complete"


def _is_primary_report_item(item: BatchPipelineItemResult) -> bool:
    """Keep ablation outputs visible without promoting their classification."""

    return bool(
        _item_classification(item) in {"complete", "complete_with_corrections"}
        or item.experimental_only
    )


def _result_has_guided_fallback(result: PipelineResult) -> bool:
    return any(
        bool(decision.get("fallback_used"))
        for decision in _guided_decisions_from_review_records(result.review_records)
    )


def _item_classification(item: BatchPipelineItemResult | None) -> str:
    if item is None:
        return "pending"
    if item.result is not None and item.result.provisional_candidate is not None:
        return "complete_with_corrections"
    if item.classification in FINAL_CLASSIFICATIONS:
        return item.classification
    if any(
        token in item.classification
        for token in ("pending", "awaiting", "review_required")
    ):
        return "needs_review"
    if item.result is not None:
        return _classification_for_result(item.result)
    return "complete"




def _final_formula_text(item: BatchPipelineItemResult | None) -> str:
    if item is None:
        return ""
    if item.result is not None and item.result.provisional_candidate is not None:
        return str(item.result.provisional_candidate.formula)
    if (
        item.result is not None
        and item.result.selected is not None
        and item.result.candidate_status == "final"
    ):
        return str(item.result.selected.formula)
    summary_candidate = _extract_candidate_from_summary_text(item.summary_text)
    if (
        summary_candidate
        and _item_classification(item) in {"complete", "complete_with_corrections"}
        and "needs_review" not in summary_candidate
        and "blocked_error" not in summary_candidate
        and "暂不输出" not in summary_candidate
    ):
        if not summary_candidate.startswith("provisional:"):
            return summary_candidate
    return ""


def _provisional_candidate_text(item: BatchPipelineItemResult | None) -> str:
    if item is None or _item_classification(item) == "blocked_error":
        return ""
    if item.result is not None and item.result.provisional_candidate is not None:
        return str(item.result.provisional_candidate.formula)
    summary_candidate = _extract_candidate_from_summary_text(item.summary_text)
    if summary_candidate.startswith("provisional:"):
        return summary_candidate.removeprefix("provisional:").strip()
    return ""


def _extract_candidate_from_summary_text(summary_text: str) -> str:
    for index, line in enumerate(summary_text.splitlines()):
        if line.strip() == "最终候选:":
            for candidate_line in summary_text.splitlines()[index + 1 :]:
                candidate = candidate_line.strip()
                if candidate:
                    return candidate
    return ""


def _review_basis_for_item(item: BatchPipelineItemResult | None) -> str:
    if item is not None and item.result is not None and item.result.selection_method == "candidate_fallback":
        return "候选回退：依据已有人工轨迹一致性及稳定排序选取。"
    if item is None:
        return "pending"
    records = item.result.review_records if item.result is not None else item.review_records
    for record in reversed(records):
        status = str(record.get("status", ""))
        if status == "human_approved":
            return status
        if record.get("stage") == "guided_warmup_learning":
            return "guided_warmup"
    if item.result is not None and item.result.selected is not None:
        if item.result.selection_status:
            return item.result.selection_status
        return "rerun_selected"
    return "checkpoint_selected" if _final_formula_text(item) else "missing"


def _pre_ir_manifest_fields(item: BatchPipelineItemResult | None) -> dict[str, Any]:
    if item is None or item.result is None:
        return {}
    segmentation = item.result.semantic.surface_segmentation
    reading = item.result.semantic.plain_reading_review
    return {
        "surface_segmentation_status": (
            segmentation.status if segmentation is not None else "missing"
        ),
        "mechanical_validation_status": (
            segmentation.mechanical_validation.status
            if segmentation is not None
            else "missing"
        ),
        "segmentation_id": (
            segmentation.segmentation_id if segmentation is not None else ""
        ),
        "segmentation_revision_count": (
            segmentation.revision if segmentation is not None else 0
        ),
        "segmentation_artifact_path": (
            f"segmentation_reviewed/{_safe_requirement_id(item.requirement_id)}.json"
            if segmentation is not None
            else ""
        ),
        "plain_reading_status": (
            reading.overall_status if reading is not None else "missing"
        ),
        "reading_review_id": (
            reading.reading_review_id if reading is not None else ""
        ),
        "plain_reading_artifact_path": (
            f"plain_reading_reviewed/{_safe_requirement_id(item.requirement_id)}.json"
            if reading is not None
            else ""
        ),
        "selected_plain_reading_axes": (
            [f"{axis}={option}" for axis, option in reading.selected_options]
            if reading is not None
            else []
        ),
        "unresolved_plain_reading_axes": (
            list(reading.unresolved_ambiguity_ids) if reading is not None else []
        ),
        "plain_reading_review_basis": (
            reading.review_basis if reading is not None else ""
        ),
        "plain_reading_reviewer_source": (
            reading.reviewer_source if reading is not None else ""
        ),
    }


def _requirement_text(item: BatchPipelineItemResult | None) -> str:
    if item is None:
        return ""
    if item.requirement:
        return item.requirement
    if item.result is not None:
        return item.result.requirement
    return ""


def _load_existing_expert_review(
    classified_output_dir: str | None,
    requirement_id: str,
) -> tuple[str, str]:
    if not classified_output_dir:
        return "", ""
    path = (
        Path(classified_output_dir)
        / "degraded"
        / "items"
        / f"{_safe_requirement_id(requirement_id)}.json"
    )
    if not path.exists():
        return "", ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", ""
    status = str(payload.get("expert_review_status", ""))
    if status not in {"pending", "approved", "rejected", "retry_requested"}:
        status = ""
    return status, str(payload.get("expert_comments", ""))


def _degraded_item_path(
    classified_output_dir: str | None,
    requirement_id: str,
) -> str:
    if not classified_output_dir:
        return ""
    return str(
        Path(classified_output_dir)
        / "degraded"
        / "items"
        / f"{_safe_requirement_id(requirement_id)}.json"
    )


def _safe_requirement_id(requirement_id: str) -> str:
    return requirement_id.replace("/", "_").replace("\\", "_")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    )


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _format_item_report(item: BatchPipelineItemResult) -> str:
    if item.result is not None:
        return format_pipeline_result(item.result)
    return item.report_text or "该批量项已从 checkpoint 恢复，但缺少完整报告文本。"


def _format_item_summary(item: BatchPipelineItemResult) -> str:
    if item.result is not None:
        return format_pipeline_summary(item.result)
    return item.summary_text or "暂不输出（checkpoint 缺少精简结果）"



