"""
Traj-LTLf 主流程。

这个文件把流程说明文档中的算法步骤连接起来：
    自然语言需求
    -> RAG + NLU Agent 生成语义结构
    -> 人工评审语义结构
    -> IR 义务驱动的 pending 有限轨迹查询
    -> 人工或独立 Judge 逐轨迹裁决
    -> ApprovedSemanticIR 冻结
    -> 确定性编译受限骨架与 typed holes
    -> 完整物化获准联合算子域
    -> Trajectory Consistency Agent 重放全部审核轨迹
    -> OperatorResolution 唯一性判定与最终门禁

当前版本支持 rules 与 api 两种模式。rules 使用本地规则便于离线演示和测试；
api 通过统一 LLM client 调用真实 OpenAI-compatible API。后半段的 LTLf
轨迹模拟、穷举完整性和算子解析逻辑保持确定性实现。
"""

from __future__ import annotations
from execution_observer import emit, observed_stage

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import re

from agents.base_agent import JSONLLMClient
from agents.completion_agent import (
    CandidateCompletionAgent,
    _render_formula_from_template,
    _validate_completion_assignments,
)
from agents.consistency_agent import TrajectoryConsistencyAgent
from agents.nlu_agent import NLUAgent
from agents.review_loop import run_semantic_review, run_trace_review
from agents.skeleton_agent import SkeletonAgent
from agents.trace_agent import (
    TraceGenerationAgent,
    assess_authoritative_trace_coverage,
)
from semantic_promotion import (
    TRACE_READY,
    AuthoritativeTraceSession,
    create_authoritative_trace_session,
    promote_semantic_candidate,
    validate_authoritative_trace_session,
)
from ap_binding_context import (
    APBindingContext,
    build_ap_binding_context,
    build_runtime_ap_binding_context,
    text_only_ap_binding_context,
)
from ambiguity_closure import (
    AmbiguityClosureReport,
    compute_ambiguity_closure_report,
)
from artifact_audit import ArtifactFingerprintSet, build_artifact_fingerprint_set
from error_diagnosis import ErrorDiagnosisPacket, build_error_diagnosis
from formula_closure import (
    FormulaClosurePolicy,
    FormulaClosureReport,
    build_strict_formula_closure_policy,
    stable_formula_hash,
    validate_formula_closure,
)
from checkpointing import (
    save_human_semantic_override_checkpoint,
    save_plain_reading_generation_diagnostic,
)
from data_examples import LabeledTrace, trace_semantics_to_dict
from guided_decision import build_guided_stage_decision
from guided_learning import GuidedLearningContext
from ltlf_parser import Formula, parse_formula
from ltlf_syntax import normalize_boolean_operator_text
from llm import build_judge_client_from_env, build_llm_client_from_env
from experiment_modes import (
    is_sample_reference_formula,
    requires_judge_client,
    resolve_experiment_mode,
)
from pattern_registry import pattern_requires_human_learning
from pattern_review import run_pattern_learning_review
from pre_ir_review import (
    PreIRReviewConfig,
    requires_interactive_pre_ir_review,
    run_pre_ir_review,
)
from rag.memory_store import build_persistent_retriever
from rag.persistent_memory import DEFAULT_APPROVED_MEMORY_FILE
from rag.example_memory import (
    build_faithful_learning_example,
    build_warmup_fixture_learning_example,
    record_approved_review_examples,
    record_approved_pre_ir_examples,
    record_approved_template_examples,
    record_trace_adequacy_issue,
)
from rag.retriever import KeywordRetriever
from rag.runtime_memory import (
    candidate_selection_review_notes,
    record_guided_stage_decision,
    record_skeleton_diagnostics,
    record_stage_errors,
)
from faithful_ir import attach_faithful_ir
from semantic_source import (
    bind_semantic_to_effective_source,
    natural_language_source,
    validate_semantic_source_binding,
)
from operator_resolution import (
    build_operator_trace_evidence,
    evaluate_operator_resolution_from_evidence,
    evaluations_for_passing_assignments,
    resolved_temporal_resolution_by_node,
    select_operator_resolved_evaluation,
)
from nl_trace_evidence import generate_coverage_debt_queries
from semantic_freezing import attach_candidate_ir_set, freeze_approved_semantic_ir
from semantic_adjudication import (
    attach_semantic_challenges,
    attach_semantic_resolution_record,
    bind_trace_queries_to_brief,
    build_trace_adjudication_brief,
    open_semantic_challenges,
)
from semantic_readiness import (
    CAN_BIND_RESOLUTION,
    CAN_GENERATE_AUTHORITATIVE_TRACES,
    audit_semantic_readiness,
)
from ir_compiler import (
    bind_traces_to_ir_plan,
    compile_completion_hole_plan,
    compile_constrained_skeleton_plan,
    compile_executable_trace_plan,
    compile_trace_obligation_plan,
    evaluate_ir_coverage,
)
from schemas import (
    APRepresentationProfile,
    DeclaredAPCatalog,
    CompletionHolePlan,
    CompletionDecision,
    ConstrainedSkeletonPlan,
    DegradedResultRecord,
    FinalApprovalRecord,
    FormulaConsistencyEvaluation,
    FaithfulLearningExample,
    HoleCandidate,
    IRCoverageReport,
    OperatorResolutionReport,
    SemanticStructure,
    SkeletonDiagnostic,
    StageErrorRecord,
    TemporalCompletion,
    TemporalHole,
    TraceCoverageReport,
    TraceEvidencePolicy,
    TraceObligationPlan,
    ExperimentConfig,
    TraceQuery,
    IRRelationFeasibilityReport,
)
from trace_evidence_policy import (
    build_trace_evidence_context,
    filter_queries_for_policy,
    filter_traces_for_policy,
    is_boundary_trace,
)
from terminal_progress import TerminalProgress
from skeleton_validator import (
    format_skeleton_diagnostics_for_agent,
    validate_skeletons,
)
from trace_formatter import trace_to_serializable
from trace_annotation import oracle_annotations
from trace_conflict_resolution import resolve_trace_evidence_disputes
from trace_evidence import reconcile_trace_evidence
from trace_adjudication_policy import adjudicate_trace_round
from trace_review_protocol import (
    manual_trace_annotations,
    independent_judge_annotations,
    trace_query_to_dict,
)
from selection_rules import (
    candidate_reject_reasons,
    completion_decision_reject_reasons,
    exact_formula_classification,
    rank_evaluations,
    select_fallback_evaluation,
    select_with_ir_coverage_gate,
)
from utils import format_trace, format_truth, ratio_text


@dataclass(frozen=True)
class PipelineResult:
    """完整流程的输出结果。"""

    requirement: str
    semantic: SemanticStructure
    ap: list[str]
    traces: list[LabeledTrace]
    skeletons: list[Formula]
    skeleton_diagnostics: list[SkeletonDiagnostic]
    skeleton_rounds: list[dict]
    min_passed_skeletons: int
    hole_candidates: list[HoleCandidate]
    completion_decisions: list[CompletionDecision]
    completion_rounds: list[dict]
    candidates: list[Formula]
    evaluations: list[FormulaConsistencyEvaluation]
    selected: FormulaConsistencyEvaluation | None
    fallback_candidate: FormulaConsistencyEvaluation | None
    selection_status: str
    selection_review_notes: tuple[str, ...]
    selection_rag_chunks: list[str]
    strict_boundary: bool
    review_mode: str
    review_records: list[dict]
    trace_coverage: TraceCoverageReport = TraceCoverageReport()
    experiment_mode: str = "human"
    oracle_assisted: bool = False
    diagnostic_only: bool = False
    ambiguous_candidates: tuple[str, ...] = ()
    trace_obligation_plan: TraceObligationPlan | None = None
    constrained_skeleton_plan: ConstrainedSkeletonPlan | None = None
    completion_hole_plan: CompletionHolePlan | None = None
    ir_coverage: IRCoverageReport = IRCoverageReport()
    operator_resolution: OperatorResolutionReport = OperatorResolutionReport()
    run_origin: str = "normal"
    candidate_status: str = "final"
    automatic_finalization_allowed: bool = False
    provisional_candidate: FormulaConsistencyEvaluation | None = None
    learning_example: FaithfulLearningExample | None = None
    final_approval: FinalApprovalRecord | None = None
    semantic_status: str = "pending"
    evidence_status: str = "pending"
    approval_status: str = "not_approved"
    learning_status: str = "writeback_blocked"
    source_run_id: str = ""
    formula_closure: FormulaClosureReport | None = None
    ambiguity_closure: AmbiguityClosureReport | None = None
    artifact_fingerprints: ArtifactFingerprintSet = ArtifactFingerprintSet()
    error_diagnoses: tuple[ErrorDiagnosisPacket, ...] = ()
    rerun_delta: object | None = None
    feedback_patch: object | None = None
    trace_evidence_policy: TraceEvidencePolicy = TraceEvidencePolicy()
    canonical_trace_coverage: TraceCoverageReport = TraceCoverageReport()
    experimental_only: bool = False
    boundary_metric_available: bool = True
    ap_context_fingerprint: str = ""
    relation_feasibility: IRRelationFeasibilityReport | None = None
    delivery_policy: str = "strict"
    output_contract_version: str = "strict-v1"
    selection_method: str = ""
    strict_evidence_pass: bool = False
    normal_flow_completed: bool = False
    enumeration_complete: bool = False
    unclear_trace_count: int = 0
    semantic_assumptions: tuple[str, ...] = ()
    degradation_reasons: tuple[str, ...] = ()
    evidence_version: str = ""
    selected_assignment: tuple[tuple[str, str], ...] = ()
    selected_branch_id: str = ""
    working_branch_count: int = 1
    discrimination_unresolved: bool = False
    source_binding_status: str = "unverified"
    prediction_namespace_contract: dict | None = None
    source_identity_diagnostics: dict | None = None
    technical_failure_categories: tuple[str, ...] = ()
    transport_completed: bool = False
    representation_valid: bool = False
    semantic_contract_passed: bool = False
    evidence_pass_count: int = 0
    evidence_contradicted: bool = False
    degradation_stage: str | None = None
    semantic_losses: tuple[str, ...] = ()
    truncation_count: int = 0
    unknown_execution_count: int = 0
    repair_count: int = 0


_REVIEW_RECORD_PATH_FALLBACK_KEYS = (
    "review_path",
    "batch_review_path",
    "manual_review_path",
    "annotation_review_path",
)

_REVIEW_RECORD_APPROVED_STATUSES = {
    "approved",
    "pass",
    "passed",
    "selected",
    "selected_reference_only",
    "human_approved",
    "no_supplement",
}

_REVIEW_RECORD_REJECTED_STATUSES = {
    "fail",
    "failed",
    "needs_review",
    "ambiguity_retained",
    "supplement_required",
    "rejected",
}


def normalize_review_record(record: dict) -> dict:
    """Return a display-safe review record with stable summary fields."""

    normalized = dict(record)
    normalized["stage"] = str(normalized.get("stage") or "unknown")
    normalized["round"] = normalized.get("round") or "?"

    inferred_approved = _infer_review_record_approved(normalized)
    if "approved" not in normalized:
        normalized["approved"] = bool(inferred_approved)
        normalized["approval_status"] = (
            "inferred" if inferred_approved is not None else "unknown"
        )
    else:
        normalized.setdefault("approval_status", "explicit")

    record_path = str(normalized.get("record_path") or "").strip()
    if record_path:
        normalized["record_path"] = record_path
        normalized.setdefault("record_path_source", "record_path")
        normalized.setdefault("storage", "file")
        return normalized

    fallback_key, fallback_path = _review_record_fallback_path(normalized)
    if fallback_path:
        normalized["record_path"] = fallback_path
        normalized["record_path_source"] = fallback_key
        normalized.setdefault("storage", "referenced")
    else:
        normalized["record_path"] = "inline"
        normalized["record_path_source"] = "inline"
        normalized.setdefault("storage", "inline")
    return normalized


def _infer_review_record_approved(record: dict) -> bool | None:
    status_values = (
        str(record.get("status", "")).strip().lower(),
        str(record.get("review_status", "")).strip().lower(),
        str(record.get("issue_type", "")).strip().lower(),
    )
    if any(status in _REVIEW_RECORD_APPROVED_STATUSES for status in status_values):
        return True
    if any(status in _REVIEW_RECORD_REJECTED_STATUSES for status in status_values):
        return False
    return None


def _review_record_fallback_path(record: dict) -> tuple[str, str]:
    for key in _REVIEW_RECORD_PATH_FALLBACK_KEYS:
        value = str(record.get(key) or "").strip()
        if value:
            return key, value
    return "", ""


def _review_record_status_label(record: dict) -> str:
    if record.get("approval_status") == "unknown":
        return "状态未知"
    return "通过" if record.get("approved") else "未通过"


def extract_semantic_structure(requirement: str) -> SemanticStructure:
    """
    使用 NLU Agent 从自然语言需求中抽取 AP 和语义结构。

    这个函数保留给单独调试使用。完整 pipeline 会调用 run_semantic_review，
    也就是“Agent 生成 -> 人工评审 -> 失败后重生成”的闭环。
    """

    retriever = build_persistent_retriever()
    agent = NLUAgent(retriever)
    return agent.generate(requirement)


def select_fallback_candidate(
    evaluations: list[FormulaConsistencyEvaluation],
    semantic: SemanticStructure | None = None,
) -> FormulaConsistencyEvaluation | None:
    """返回最高分参考候选，仅用于 needs_review 诊断展示。"""

    return select_fallback_evaluation(
        evaluations=evaluations,
        semantic=semantic,
    )


def run_skeleton_generation_loop(
    requirement: str,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    agent: SkeletonAgent,
    retriever: KeywordRetriever,
    min_passed_skeletons: int = 5,
    max_rounds: int = 3,
    records_dir: str = "review_records",
    progress: TerminalProgress | None = None,
    progress_label: str | None = None,
    guided_context: GuidedLearningContext | None = None,
) -> tuple[list[Formula], list[SkeletonDiagnostic], list[dict]]:
    """
    从当前 FaithfulSemanticIR 取得唯一权威骨架并验证一次。

    主流程不再通过多轮 LLM 候选或 deterministic fallback 改写已批准拓扑；
    验证失败会形成结构化诊断并阻断下游补全。
    """

    del min_passed_skeletons, max_rounds
    feedback_history: list[str] = []
    records: list[dict] = []
    last_diagnostics: list[SkeletonDiagnostic] = []
    label = progress_label or "Requirement"
    if semantic.faithful_ir is None:
        raise RuntimeError("Constrained skeleton compilation requires FaithfulSemanticIR")

    # The approved IR topology has one authoritative deterministic compilation.
    effective_max_rounds = 1
    for round_index in range(1, effective_max_rounds + 1):
        include_fallback = False
        message = f"{label} skeleton generation round {round_index}"
        skeleton_context = (
            guided_context.attempt_view("skeleton", 1)
            if guided_context is not None
            else None
        )
        if progress is None:
            skeletons = agent.generate(
                requirement=requirement,
                semantic=semantic,
                traces=traces,
                feedback_history=feedback_history,
                guided_context=skeleton_context,
                include_fallback=include_fallback,
                batch_size=1,
            )
            diagnostics = validate_skeletons(skeletons, traces, semantic)
        else:
            with progress.task(message):
                skeletons = agent.generate(
                    requirement=requirement,
                    semantic=semantic,
                    traces=traces,
                    feedback_history=feedback_history,
                    guided_context=skeleton_context,
                    include_fallback=include_fallback,
                    batch_size=1,
                )
                diagnostics = validate_skeletons(skeletons, traces, semantic)
        last_diagnostics = diagnostics
        passed_skeletons = [
            diagnostic.skeleton for diagnostic in diagnostics if diagnostic.passed
        ]
        guided_decisions: list[dict] = []
        guided_resolution = _resolve_guided_skeletons(
            requirement=requirement,
            semantic=semantic,
            traces=traces,
            generated_skeletons=skeletons,
            generated_passed_skeletons=passed_skeletons,
            generated_diagnostics=diagnostics,
            guided_context=guided_context,
            retriever=retriever,
            round_index=round_index,
            llm_client=agent.llm_client,
            agent=agent,
            feedback_history=feedback_history,
            include_fallback=include_fallback,
            batch_size=1,
        )
        if guided_resolution is not None:
            (
                skeletons,
                diagnostics,
                passed_skeletons,
                guided_decision,
                guided_rag_chunks,
            ) = guided_resolution
            guided_decision["runtime_rag_chunks"] = guided_rag_chunks
            guided_decisions.append(guided_decision)
        rag_chunk_ids = record_skeleton_diagnostics(
            retriever=retriever,
            requirement=requirement,
            semantic=semantic,
            diagnostics=diagnostics,
            round_index=round_index,
        )
        for guided_decision in guided_decisions:
            rag_chunk_ids.extend(guided_decision.get("runtime_rag_chunks", []))
        stage_errors = _build_skeleton_stage_errors(
            diagnostics=diagnostics,
            round_index=round_index,
        )
        rag_chunk_ids.extend(
            record_stage_errors(
                retriever=retriever,
                requirement=requirement,
                semantic=semantic,
                stage="skeleton",
                round_index=round_index,
                errors=stage_errors,
            )
        )
        approved = bool(passed_skeletons)
        comments = (
            f"结构校验和轨迹粗过滤后可用受限骨架 "
            f"{len(passed_skeletons)}/{len(skeletons)} 个。"
        )
        if not approved:
            comments += " 已将结构/轨迹粗过滤诊断写入当前 RAG，并反馈给 SkeletonAgent。"

        record = {
            "stage": "skeleton",
            "round": round_index,
            "requirement": requirement,
            "semantic": asdict(semantic),
            "artifact": [str(skeleton) for skeleton in skeletons],
            "batch_skeleton_count": len(skeletons),
            "compilation_policy": "single_authoritative_ir_plan",
            "diagnostics": [
                _skeleton_diagnostic_to_dict(diagnostic)
                for diagnostic in diagnostics
            ],
            "passed_count": len(passed_skeletons),
            "runtime_rag_chunks": rag_chunk_ids,
            "stage_errors": [asdict(error) for error in stage_errors],
            "guided_decisions": guided_decisions,
            "approved": approved,
            "comments": comments,
        }
        record["record_path"] = str(_save_skeleton_record(record, records_dir))
        records.append(record)

        if progress is not None:
            status_message = (
                f"{label} skeletons passed: "
                f"{len(passed_skeletons)}/{len(skeletons)}"
            )
            if approved:
                progress.ok(status_message)
            else:
                progress.warn(status_message)

        if approved:
            return passed_skeletons, diagnostics, records

        if guided_context is not None:
            raise RuntimeError(
                "warmup guided skeleton validation failed."
            )

        feedback_history.append(
            format_skeleton_diagnostics_for_agent(
                diagnostics=diagnostics,
            )
        )

    feedback = format_skeleton_diagnostics_for_agent(
        diagnostics=last_diagnostics,
    )
    raise RuntimeError(
        "权威受限骨架未通过验证。\n" + feedback
    )


@observed_stage("candidate_completion")
def run_candidate_completion_loop(
    requirement: str,
    semantic: SemanticStructure,
    skeletons: list[Formula],
    traces: list[LabeledTrace],
    agent: CandidateCompletionAgent,
    consistency_agent: TrajectoryConsistencyAgent,
    strict_boundary: bool = True,
    max_rounds: int = 3,
    records_dir: str = "review_records",
    progress: TerminalProgress | None = None,
    progress_label: str | None = None,
    guided_context: GuidedLearningContext | None = None,
) -> tuple[list[Formula], list[HoleCandidate], list[CompletionDecision], list[dict]]:
    """
    运行受限骨架到完整 LTLf 候选公式的补全闭环。

    Completion Agent 按 CompletionHolePlan 穷举所有联合算子赋值；随后由
    TrajectoryConsistencyAgent 对每个完整公式统一模拟全部审核轨迹。
    """

    del max_rounds
    feedback_history: list[str] = []
    records: list[dict] = []
    last_candidates: list[Formula] = []
    last_hole_candidates: list[HoleCandidate] = []
    last_decisions: list[CompletionDecision] = []
    label = progress_label or "Requirement"
    if semantic.faithful_ir is None:
        raise RuntimeError("Typed exhaustive completion requires FaithfulSemanticIR")
    emit("before_enumeration", ir=semantic.faithful_ir, traces=traces)
    accumulated_hole_candidates: list[HoleCandidate] = []
    accumulated_completions: list[TemporalCompletion] = []
    accumulated_decisions: list[CompletionDecision] = []
    seen_candidate_hashes: set[str] = set()

    effective_max_rounds = 1
    for round_index in range(1, effective_max_rounds + 1):
        guided_decisions: list[dict] = []
        completion_mismatch_evidence: list[str] = []
        invalid_completion_records: list[dict] = []
        attempted_hole_candidate_count = 0
        include_fallback = False
        template_context = (
            guided_context.attempt_view("template", 1)
            if guided_context is not None
            else None
        )
        completion_context = (
            guided_context.attempt_view("completion", 1)
            if guided_context is not None
            else None
        )
        if progress is None:
            hole_candidates = agent.generate_hole_candidates(
                requirement=requirement,
                semantic=semantic,
                skeletons=skeletons,
                traces=traces,
                feedback_history=feedback_history,
                guided_context=template_context,
                include_fallback=include_fallback,
                batch_size=1,
            )
            template_resolution = _resolve_guided_hole_candidates(
                requirement=requirement,
                semantic=semantic,
                skeletons=skeletons,
                generated_hole_candidates=hole_candidates,
                guided_context=guided_context,
                retriever=agent.retriever,
                round_index=round_index,
                llm_client=agent.llm_client,
            )
            if template_resolution is not None:
                hole_candidates, template_decision, template_rag_chunks = template_resolution
                template_decision["runtime_rag_chunks"] = template_rag_chunks
                guided_decisions.append(template_decision)
            attempted_hole_candidate_count = len(hole_candidates)
            completions = agent.complete_temporal_holes(
                semantic=semantic,
                hole_candidates=hole_candidates,
                traces=traces,
                guided_context=completion_context,
            )
            invalid_completion_records = list(
                getattr(agent, "last_invalid_completions", [])
            )
            completion_alignment_failed = False
            try:
                hole_candidates = _scoreable_hole_candidates(
                    hole_candidates=hole_candidates,
                    completions=completions,
                )
            except RuntimeError as exc:
                if _guided_expected_formula_text(guided_context):
                    completion_mismatch_evidence.append(str(exc))
                    decisions = []
                    completion_alignment_failed = True
                else:
                    raise
            if not completion_alignment_failed:
                try:
                    _assert_completion_count_matches_holes(
                        hole_candidates=hole_candidates,
                        completions=completions,
                    )
                except RuntimeError as exc:
                    if _guided_expected_formula_text(guided_context):
                        completion_mismatch_evidence.append(str(exc))
                        decisions = []
                    else:
                        raise
                else:
                    decisions = _score_temporal_completions(
                        completions=completions,
                        traces=traces,
                        consistency_agent=consistency_agent,
                    )
        else:
            with progress.task(
                f"{label} hole candidate generation round {round_index}"
            ):
                hole_candidates = agent.generate_hole_candidates(
                    requirement=requirement,
                    semantic=semantic,
                    skeletons=skeletons,
                    traces=traces,
                    feedback_history=feedback_history,
                    guided_context=template_context,
                    include_fallback=include_fallback,
                    batch_size=1,
                )
            progress.ok(f"{label} hole candidates: {len(hole_candidates)}")
            template_resolution = _resolve_guided_hole_candidates(
                requirement=requirement,
                semantic=semantic,
                skeletons=skeletons,
                generated_hole_candidates=hole_candidates,
                guided_context=guided_context,
                retriever=agent.retriever,
                round_index=round_index,
                llm_client=agent.llm_client,
            )
            if template_resolution is not None:
                hole_candidates, template_decision, template_rag_chunks = template_resolution
                template_decision["runtime_rag_chunks"] = template_rag_chunks
                guided_decisions.append(template_decision)
            attempted_hole_candidate_count = len(hole_candidates)
            with progress.task(
                f"{label} temporal completion round {round_index}"
            ):
                completions = agent.complete_temporal_holes(
                    semantic=semantic,
                    hole_candidates=hole_candidates,
                    traces=traces,
                    guided_context=completion_context,
                )
            invalid_completion_records = list(
                getattr(agent, "last_invalid_completions", [])
            )
            completion_alignment_failed = False
            try:
                hole_candidates = _scoreable_hole_candidates(
                    hole_candidates=hole_candidates,
                    completions=completions,
                )
            except RuntimeError as exc:
                if _guided_expected_formula_text(guided_context):
                    completion_mismatch_evidence.append(str(exc))
                    decisions = []
                    completion_alignment_failed = True
                else:
                    raise
            if not completion_alignment_failed:
                try:
                    _assert_completion_count_matches_holes(
                        hole_candidates=hole_candidates,
                        completions=completions,
                    )
                except RuntimeError as exc:
                    if _guided_expected_formula_text(guided_context):
                        completion_mismatch_evidence.append(str(exc))
                        decisions = []
                    else:
                        raise
                else:
                    decisions = None
            progress.ok(
                f"{label} temporal completions: "
                f"{len(completions)}/{len(hole_candidates)}"
            )
            if decisions is None:
                with progress.task(
                    f"{label} completion consistency scoring round {round_index}"
                ):
                    decisions = _score_temporal_completions(
                        completions=completions,
                        traces=traces,
                        consistency_agent=consistency_agent,
                    )
                progress.ok(
                    f"{label} scored completions: "
                    f"{len(decisions)}/{len(completions)}"
                )
        completion_resolution = _resolve_guided_completion_decisions(
            requirement=requirement,
            semantic=semantic,
            hole_candidates=hole_candidates,
            generated_completions=completions,
            generated_decisions=decisions,
            traces=traces,
            consistency_agent=consistency_agent,
            guided_context=guided_context,
            retriever=agent.retriever,
            round_index=round_index,
            strict_boundary=strict_boundary,
            llm_client=agent.llm_client,
            pre_mismatch_evidence=completion_mismatch_evidence,
        )
        if completion_resolution is not None:
            completions, decisions, completion_decision, completion_rag_chunks = (
                completion_resolution
            )
            completion_decision["runtime_rag_chunks"] = completion_rag_chunks
            guided_decisions.append(completion_decision)
        new_hole_candidates: list[HoleCandidate] = []
        new_completions: list[TemporalCompletion] = []
        new_decisions: list[CompletionDecision] = []
        for hole_candidate, completion, decision in zip(
            hole_candidates,
            completions,
            decisions,
        ):
            candidate_key = stable_formula_hash(completion.formula)
            if candidate_key in seen_candidate_hashes:
                continue
            seen_candidate_hashes.add(candidate_key)
            new_hole_candidates.append(hole_candidate)
            new_completions.append(completion)
            new_decisions.append(decision)
        accumulated_hole_candidates.extend(new_hole_candidates)
        accumulated_completions.extend(new_completions)
        accumulated_decisions.extend(new_decisions)
        current_hole_candidates = accumulated_hole_candidates
        current_completions = accumulated_completions
        current_decisions = accumulated_decisions
        candidates = [decision.formula for decision in current_decisions]
        last_candidates = candidates
        last_hole_candidates = current_hole_candidates
        last_decisions = current_decisions

        strict_match_decisions = [
            decision
            for decision in current_decisions
            if not completion_decision_reject_reasons(
                decision=decision,
                strict_boundary=strict_boundary,
                semantic=semantic,
            )
        ]
        no_effective_delta = round_index > 1 and not new_hole_candidates
        approved = bool(strict_match_decisions)
        rag_chunk_ids: list[str] = []
        stage_errors: list[StageErrorRecord] = _build_invalid_completion_stage_errors(
            invalid_completion_records,
            round_index=round_index,
        )
        stage_errors.extend(
            _build_completion_stage_errors(
                current_decisions,
                round_index,
                strict_boundary,
                semantic,
            )
        )
        comments = (
            f"本轮新增带孔候选模板 {len(new_hole_candidates)} 个，"
            f"累计带孔候选模板 {len(current_hole_candidates)} 个，"
            f"累计补全决策 {len(current_completions)} 个，"
            f"累计轨迹一致性评分 {len(current_decisions)} 个，"
            f"后验严格匹配候选 {len(strict_match_decisions)} 个；"
            "补全阶段保留所有累计候选，不按评分或严格指标筛选。"
        )
        # Completion-time matches are diagnostics only.  Approved-example
        # memory is written once, after unique operator resolution and every
        # final gate have succeeded (see the final writeback below).
        for guided_decision in guided_decisions:
            rag_chunk_ids.extend(guided_decision.get("runtime_rag_chunks", []))
        if guided_decisions:
            comments += " 已记录 guided 阶段取舍理由。"
        if stage_errors:
            rag_chunk_ids.extend(
                record_stage_errors(
                    agent.retriever,
                    requirement=requirement,
                    semantic=semantic,
                    stage="completion",
                    round_index=round_index,
                    errors=stage_errors,
                )
            )
            comments += (
                f" 本轮有 {len(stage_errors)} 个 LLM 补全项无效，"
                "已记录为 invalid_completion；这些候选未进入轨迹评分。"
            )

        record = {
            "stage": "completion",
            "round": round_index,
            "status": (
                "no_effective_delta"
                if no_effective_delta
                else
                "completed"
                if strict_match_decisions
                or guided_context is not None
                or round_index >= effective_max_rounds
                else "needs_next_batch"
            ),
            "requirement": requirement,
            "semantic": asdict(semantic),
            "hole_candidates": [
                _hole_candidate_to_dict(hole_candidate)
                for hole_candidate in current_hole_candidates
            ],
            "hole_candidate_count": len(current_hole_candidates),
            "completion_count": len(current_completions),
            "scored_completion_count": len(current_decisions),
            "batch_hole_candidate_count": attempted_hole_candidate_count,
            "new_hole_candidate_count": len(new_hole_candidates),
            "completion_policy": "exhaustive_ir_cartesian_product",
            "invalid_completion_count": len(invalid_completion_records),
            "invalid_completions": invalid_completion_records,
            "completion_decisions": [
                _completion_decision_to_dict(
                    decision=decision,
                    strict_boundary=strict_boundary,
                    semantic=semantic,
                )
                for decision in current_decisions
            ],
            "candidate_formulas": [str(candidate) for candidate in candidates],
            "approved_count": len(strict_match_decisions),
            "strict_match_count": len(strict_match_decisions),
            "approved": approved,
            "runtime_rag_chunks": rag_chunk_ids,
            "stage_errors": [asdict(error) for error in stage_errors],
            "guided_decisions": guided_decisions,
            "comments": comments,
        }
        record["record_path"] = str(_save_completion_record(record, records_dir))
        records.append(record)

        if progress is not None:
            status_message = (
                f"{label} completion candidates: {len(candidates)}, "
                f"templates/completions/scored: "
                f"{len(current_hole_candidates)}/{len(current_completions)}/{len(current_decisions)}, "
                f"retained: {len(current_decisions)}, strict match: {len(strict_match_decisions)}"
            )
            progress.ok(status_message)

        should_return = (
            bool(strict_match_decisions)
            or no_effective_delta
            or guided_context is not None
            or round_index >= effective_max_rounds
        )
        if should_return:
            return candidates, current_hole_candidates, current_decisions, records

        feedback_history.append(
            _format_completion_feedback(
                current_decisions,
                strict_boundary=strict_boundary,
                semantic=semantic,
            )
            + "\n已尝试的带孔模板:\n"
            + "\n".join(
                f"- {hole_candidate.template_text}"
                for hole_candidate in current_hole_candidates[:12]
            )
            + "\n请生成下一批不同的高质量候选，优先修复上述失败原因。"
        )

    return last_candidates, last_hole_candidates, last_decisions, records


def _score_temporal_completions(
    completions: list[TemporalCompletion],
    traces: list[LabeledTrace],
    consistency_agent: TrajectoryConsistencyAgent,
) -> list[CompletionDecision]:
    """把纯补全结果交给 TrajectoryConsistencyAgent 统一评分。"""

    decisions = [
            replace(
                consistency_agent.evaluate_completion_decision(
                formula=completion.formula,
                assignments=completion.assignments,
                traces=traces,
                candidate_id=completion.candidate_id,
                template_text=completion.template_text,
                operator_derivation=completion.operator_derivation,
                ),
                ir_version=completion.ir_version,
                closure_report=completion.closure_report,
            )
        for completion in completions
    ]
    if len(decisions) != len(completions):
        raise RuntimeError(
            "补全评分数量不一致: "
            f"completions={len(completions)}, decisions={len(decisions)}"
        )
    return decisions


def _scoreable_hole_candidates(
    *,
    hole_candidates: list[HoleCandidate],
    completions: list[TemporalCompletion],
) -> list[HoleCandidate]:
    if not completions:
        raise ValueError(
            "completion 阶段全部候选均为 invalid；没有候选可进入轨迹评分。"
        )
    remaining = list(hole_candidates)
    matched: list[HoleCandidate] = []
    for completion in completions:
        match_index = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if completion.candidate_id
                and candidate.candidate_id == completion.candidate_id
            ),
            None,
        )
        if match_index is None and not completion.candidate_id and remaining:
            match_index = 0
        if match_index is None:
            raise RuntimeError(
                "补全候选无法匹配带孔模板: "
                f"candidate_id={completion.candidate_id or '<empty>'}"
            )
        matched.append(remaining.pop(match_index))
    return matched

def _assert_completion_count_matches_holes(
    *,
    hole_candidates: list[HoleCandidate],
    completions: list[TemporalCompletion],
) -> None:
    if len(completions) != len(hole_candidates):
        raise RuntimeError(
            "带孔模板与补全决策数量不一致: "
            f"hole_candidates={len(hole_candidates)}, "
            f"completions={len(completions)}"
        )


def _validate_guided_completion_result(
    *,
    guided_context: GuidedLearningContext | None,
    decisions: list[CompletionDecision],
) -> None:
    if guided_context is None:
        return
    expected = _guided_expected_formula_text(guided_context)
    if not expected:
        return
    if any(str(decision.formula) == expected for decision in decisions):
        return
    raise RuntimeError(
        "warmup guided completion did not generate the expected final "
        f"candidate in its single allowed round: {expected}"
    )


def _guided_expected_formula_text(
    guided_context: GuidedLearningContext | None,
) -> str:
    if guided_context is None:
        return ""
    try:
        formula = guided_context.expected_final_formula()
    except ValueError as exc:
        raise RuntimeError(
            "warmup guided expected final candidate is not parseable: "
            f"{guided_context.expected_final_candidate or guided_context.approved_formula}"
        ) from exc
    return str(formula) if formula is not None else ""


def _resolve_guided_skeletons(
    *,
    requirement: str,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    generated_skeletons: list[Formula],
    generated_passed_skeletons: list[Formula],
    generated_diagnostics: list[SkeletonDiagnostic],
    guided_context: GuidedLearningContext | None,
    retriever: KeywordRetriever,
    round_index: int,
    llm_client: object | None,
    agent: SkeletonAgent,
    feedback_history: list[str],
    include_fallback: bool,
    batch_size: int,
) -> tuple[list[Formula], list[SkeletonDiagnostic], list[Formula], dict, list[str]] | None:
    expected_skeletons = _guided_expected_skeleton_formulas(guided_context)
    if not expected_skeletons:
        return None
    expected_diagnostics = validate_skeletons(expected_skeletons, traces, semantic)
    invalid_expected_diagnostics = [
        diagnostic for diagnostic in expected_diagnostics if not diagnostic.passed
    ]
    expected_texts = {str(skeleton) for skeleton in expected_skeletons}
    generated_passed_texts = {str(skeleton) for skeleton in generated_passed_skeletons}
    mismatch = _guided_invalid_expected_skeleton_evidence(
        invalid_expected_diagnostics
    )
    missing = sorted(expected_texts - generated_passed_texts)
    if missing:
        mismatch.append(
            "generated passed skeletons did not include expected restricted "
            "skeleton(s): " + ", ".join(missing)
        )
    first_mismatch = list(mismatch)
    resolved_skeletons = generated_skeletons
    resolved_diagnostics = generated_diagnostics
    resolved_passed = generated_passed_skeletons
    artifact_origin = "generated_round_1"
    round_2_artifact: object | None = None
    round_2_verification = "not_required"
    if mismatch and not invalid_expected_diagnostics and guided_context is not None:
        try:
            round_2_skeletons = agent.generate(
                requirement=requirement,
                semantic=semantic,
                traces=traces,
                feedback_history=[
                    *feedback_history,
                    "Warmup attempt 1 mismatch; use the current-stage teacher "
                    "skeleton signal on attempt 2. " + "; ".join(mismatch),
                ],
                guided_context=guided_context.attempt_view("skeleton", 2),
                include_fallback=include_fallback,
                batch_size=batch_size,
            )
            round_2_diagnostics = validate_skeletons(
                round_2_skeletons, traces, semantic
            )
            round_2_passed = [
                diagnostic.skeleton
                for diagnostic in round_2_diagnostics
                if diagnostic.passed
            ]
            round_2_artifact = [str(item) for item in round_2_skeletons]
            round_2_passed_texts = {str(item) for item in round_2_passed}
            round_2_mismatch = _guided_invalid_expected_skeleton_evidence(
                [item for item in round_2_diagnostics if not item.passed]
            )
            round_2_missing = sorted(expected_texts - round_2_passed_texts)
            if round_2_missing:
                round_2_mismatch.append(
                    "guided attempt 2 did not include expected restricted "
                    "skeleton(s): " + ", ".join(round_2_missing)
                )
            if not round_2_mismatch:
                resolved_skeletons = round_2_skeletons
                resolved_diagnostics = round_2_diagnostics
                resolved_passed = round_2_passed
                artifact_origin = "generated_round_2"
                round_2_verification = "passed"
            else:
                artifact_origin = "expected_after_round_2"
                round_2_verification = "passed_after_expected_adoption"
        except Exception as exc:
            round_2_artifact = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            artifact_origin = "expected_after_round_2"
            round_2_verification = "passed_after_expected_adoption"

        if artifact_origin != "generated_round_2":
            resolved_skeletons = expected_skeletons
            resolved_diagnostics = expected_diagnostics
            resolved_passed = [
                diagnostic.skeleton
                for diagnostic in resolved_diagnostics
                if diagnostic.passed
            ]
            if round_2_artifact is None:
                round_2_artifact = [str(item) for item in resolved_skeletons]

    round_2_kwargs = (
        {
            "round_2_artifact": round_2_artifact,
            "round_2_verification_status": round_2_verification,
            "artifact_origin": artifact_origin,
        }
        if first_mismatch and not invalid_expected_diagnostics
        else {"artifact_origin": artifact_origin}
    )
    decision = build_guided_stage_decision(
        stage="skeleton",
        requirement_id=guided_context.requirement_id if guided_context else "",
        requirement=requirement,
        round_index=round_index,
        generated_artifact=[str(skeleton) for skeleton in generated_skeletons],
        expected_artifact=[str(skeleton) for skeleton in expected_skeletons],
        mismatch_evidence=first_mismatch,
        llm_client=llm_client,
        **round_2_kwargs,
    )
    if invalid_expected_diagnostics and generated_passed_skeletons:
        decision = _mark_invalid_expected_skeleton_decision(decision)
    rag_chunks = record_guided_stage_decision(
        retriever,
        requirement=requirement,
        semantic=semantic,
        decision_record=decision,
    )
    if not first_mismatch:
        return (
            generated_skeletons,
            generated_diagnostics,
            generated_passed_skeletons,
            decision,
            rag_chunks,
        )
    if invalid_expected_diagnostics and generated_passed_skeletons:
        return (
            generated_skeletons,
            generated_diagnostics,
            generated_passed_skeletons,
            decision,
            rag_chunks,
        )
    return (
        resolved_skeletons,
        resolved_diagnostics,
        resolved_passed,
        decision,
        rag_chunks,
    )


def _guided_invalid_expected_skeleton_evidence(
    diagnostics: list[SkeletonDiagnostic],
) -> list[str]:
    evidence: list[str] = []
    for diagnostic in diagnostics:
        details = " ".join(diagnostic.trace_evidence).strip()
        evidence.append(
            "invalid_expected_skeleton: expected restricted skeleton failed "
            f"the restricted skeleton contract: {diagnostic.skeleton} "
            f"({diagnostic.error_type}). {details}"
        )
    return evidence


def _mark_invalid_expected_skeleton_decision(decision: dict) -> dict:
    updated = dict(decision)
    updated["status"] = "invalid_expected_skeleton"
    updated["decision"] = "keep_generated"
    updated["artifact_origin"] = "generated"
    updated["fallback_used"] = False
    updated["guided_correction"] = None
    updated["decision_reason"] = (
        "Expected warmup restricted skeleton failed deterministic validation; "
        "generated skeletons that already passed validation are retained."
    )
    updated["lesson_for_rag"] = (
        "Warmup expected restricted skeletons must satisfy the restricted "
        "skeleton contract before they can be used as fallback."
    )
    updated["rationale_source"] = "local"
    return updated


def _resolve_guided_hole_candidates(
    *,
    requirement: str,
    semantic: SemanticStructure,
    skeletons: list[Formula],
    generated_hole_candidates: list[HoleCandidate],
    guided_context: GuidedLearningContext | None,
    retriever: KeywordRetriever,
    round_index: int,
    llm_client: object | None,
) -> tuple[list[HoleCandidate], dict, list[str]] | None:
    if guided_context is None or not guided_context.expected_hole_template:
        return None
    expected_template = _canonical_guided_text(guided_context.expected_hole_template)
    generated_templates = {
        _canonical_guided_text(candidate.template_text)
        for candidate in generated_hole_candidates
    }
    mismatch = []
    if expected_template not in generated_templates:
        mismatch.append(
            "generated hole templates did not include expected template: "
            + guided_context.expected_hole_template
        )
    decision = build_guided_stage_decision(
        stage="template",
        requirement_id=guided_context.requirement_id,
        requirement=requirement,
        round_index=round_index,
        generated_artifact=[
            _hole_candidate_to_dict(candidate)
            for candidate in generated_hole_candidates
        ],
        expected_artifact={"template_text": guided_context.expected_hole_template},
        mismatch_evidence=mismatch,
        llm_client=llm_client,
    )
    if mismatch and semantic.faithful_ir is not None:
        decision = _mark_deterministic_expected_mismatch_decision(
            decision, stage="template"
        )
    rag_chunks = record_guided_stage_decision(
        retriever,
        requirement=requirement,
        semantic=semantic,
        decision_record=decision,
    )
    if not mismatch:
        return generated_hole_candidates, decision, rag_chunks
    if semantic.faithful_ir is not None:
        # Once a reviewed IR exists, a warmup template may explain an expected
        # formula but must not replace version-bound IR holes with legacy
        # H_scope/H_response fields.  Completion resolution can still compare
        # the resulting whole formula with the warmup expectation downstream.
        decision["decision_reason"] = (
            "Expected legacy hole names differ from the reviewed IR hole plan; "
            "the version-bound IR candidates are retained."
        )
        decision["rationale_source"] = "local"
        return generated_hole_candidates, decision, rag_chunks
    return (
        [
            _expected_hole_candidate(
                semantic=semantic,
                skeletons=skeletons,
                template_text=guided_context.expected_hole_template,
            )
        ],
        decision,
        rag_chunks,
    )


def _resolve_guided_completion_decisions(
    *,
    requirement: str,
    semantic: SemanticStructure,
    hole_candidates: list[HoleCandidate],
    generated_completions: list[TemporalCompletion],
    generated_decisions: list[CompletionDecision],
    traces: list[LabeledTrace],
    consistency_agent: TrajectoryConsistencyAgent,
    guided_context: GuidedLearningContext | None,
    retriever: KeywordRetriever,
    round_index: int,
    strict_boundary: bool,
    llm_client: object | None,
    pre_mismatch_evidence: list[str] | tuple[str, ...] = (),
) -> tuple[list[TemporalCompletion], list[CompletionDecision], dict, list[str]] | None:
    expected_formula = _guided_expected_formula_text(guided_context)
    if not expected_formula:
        return None
    generated_formulas = {str(decision.formula) for decision in generated_decisions}
    expected_assignments = tuple(
        guided_context.approved_assignments if guided_context is not None else ()
    )
    mismatch = [str(item) for item in pre_mismatch_evidence if str(item).strip()]
    if expected_formula not in generated_formulas:
        mismatch.append(
            "generated completions did not include expected final candidate: "
            + expected_formula
        )
    elif expected_assignments and semantic.faithful_ir is None:
        matching_assignments = [
            tuple(decision.assignments)
            for decision in generated_decisions
            if str(decision.formula) == expected_formula
        ]
        if expected_assignments not in matching_assignments:
            mismatch.append(
                "generated completion used expected formula but did not keep "
                "approved assignments: "
                + ", ".join(
                    f"{hole_id}={operator}"
                    for hole_id, operator in expected_assignments
                )
            )
    decision = build_guided_stage_decision(
        stage="completion",
        requirement_id=guided_context.requirement_id if guided_context else "",
        requirement=requirement,
        round_index=round_index,
        generated_artifact=[
            _completion_decision_to_dict(
                decision=item,
                strict_boundary=strict_boundary,
                semantic=semantic,
            )
            for item in generated_decisions
        ],
        expected_artifact={
            "formula": expected_formula,
            "assignments": [
                {"hole_id": hole_id, "selected_op": operator}
                for hole_id, operator in expected_assignments
            ],
        },
        mismatch_evidence=mismatch,
        llm_client=llm_client,
    )
    if mismatch and semantic.faithful_ir is not None:
        decision = _mark_deterministic_expected_mismatch_decision(
            decision, stage="completion"
        )
    rag_chunks = record_guided_stage_decision(
        retriever,
        requirement=requirement,
        semantic=semantic,
        decision_record=decision,
    )
    if not mismatch:
        return generated_completions, generated_decisions, decision, rag_chunks
    if semantic.faithful_ir is not None:
        return generated_completions, generated_decisions, decision, rag_chunks
    expected_completions = _expected_temporal_completions(
        guided_context=guided_context,
        hole_candidates=hole_candidates,
        semantic=semantic,
    )
    expected_decisions = _score_temporal_completions(
        completions=expected_completions,
        traces=traces,
        consistency_agent=consistency_agent,
    )
    return expected_completions, expected_decisions, decision, rag_chunks


def _resolve_guided_consistency_evaluations(
    *,
    requirement: str,
    semantic: SemanticStructure,
    generated_evaluations: list[FormulaConsistencyEvaluation],
    traces: list[LabeledTrace],
    consistency_agent: TrajectoryConsistencyAgent,
    guided_context: GuidedLearningContext | None,
    retriever: KeywordRetriever,
    round_index: int,
    records_dir: str,
    strict_boundary: bool,
    llm_client: object | None,
) -> tuple[list[FormulaConsistencyEvaluation], dict] | None:
    expected_formula = _guided_expected_formula_text(guided_context)
    if not expected_formula:
        return None
    generated_formulas = {str(evaluation.formula) for evaluation in generated_evaluations}
    mismatch = []
    evaluations = list(generated_evaluations)
    if expected_formula not in generated_formulas:
        mismatch.append(
            "final consistency evaluations did not include expected candidate: "
            + expected_formula
        )
        if semantic.faithful_ir is not None:
            decision = build_guided_stage_decision(
                stage="consistency",
                requirement_id=(guided_context.requirement_id if guided_context else ""),
                requirement=requirement,
                round_index=round_index,
                generated_artifact=[
                    _evaluation_to_decision_artifact(evaluation)
                    for evaluation in generated_evaluations
                ],
                expected_artifact={"formula": expected_formula},
                mismatch_evidence=mismatch,
                llm_client=llm_client,
            )
            decision = _mark_deterministic_expected_mismatch_decision(
                decision, stage="consistency"
            )
            rag_chunks = record_guided_stage_decision(
                retriever,
                requirement=requirement,
                semantic=semantic,
                decision_record=decision,
            )
            return generated_evaluations, _guided_decision_review_record(
                decision=decision,
                requirement=requirement,
                semantic=semantic,
                runtime_rag_chunks=rag_chunks,
                records_dir=records_dir,
            )
        evaluations.append(
            consistency_agent.evaluate_formula(
                parse_formula(normalize_boolean_operator_text(expected_formula)),
                traces,
            )
        )
    expected_evaluation = next(
        evaluation for evaluation in evaluations if str(evaluation.formula) == expected_formula
    )
    reject_reasons = candidate_reject_reasons(
        evaluation=expected_evaluation,
        strict_boundary=strict_boundary,
        semantic=semantic,
    )
    if reject_reasons:
        mismatch.extend(reject_reasons)
    decision = build_guided_stage_decision(
        stage="consistency",
        requirement_id=guided_context.requirement_id if guided_context else "",
        requirement=requirement,
        round_index=round_index,
        generated_artifact=[
            _evaluation_to_decision_artifact(evaluation)
            for evaluation in generated_evaluations
        ],
        expected_artifact=_evaluation_to_decision_artifact(expected_evaluation),
        mismatch_evidence=mismatch,
        llm_client=llm_client,
    )
    if mismatch and semantic.faithful_ir is not None:
        decision = _mark_deterministic_expected_mismatch_decision(
            decision, stage="consistency"
        )
    rag_chunks = record_guided_stage_decision(
        retriever,
        requirement=requirement,
        semantic=semantic,
        decision_record=decision,
    )
    record = _guided_decision_review_record(
        decision=decision,
        requirement=requirement,
        semantic=semantic,
        runtime_rag_chunks=rag_chunks,
        records_dir=records_dir,
    )
    return evaluations, record


def _build_guided_selection_record(
    *,
    requirement: str,
    semantic: SemanticStructure,
    evaluations: list[FormulaConsistencyEvaluation],
    selected: FormulaConsistencyEvaluation | None,
    guided_context: GuidedLearningContext | None,
    retriever: KeywordRetriever,
    round_index: int,
    records_dir: str,
    strict_boundary: bool,
    llm_client: object | None,
) -> dict | None:
    expected_formula = _guided_expected_formula_text(guided_context)
    if not expected_formula:
        return None
    selected_formula = str(selected.formula) if selected is not None else ""
    mismatch = []
    if selected_formula != expected_formula:
        mismatch.append(
            "selected final candidate did not equal expected final candidate: "
            f"selected={selected_formula or 'none'}, expected={expected_formula}"
        )
    expected_evaluation = next(
        (evaluation for evaluation in evaluations if str(evaluation.formula) == expected_formula),
        None,
    )
    if expected_evaluation is not None:
        reject_reasons = candidate_reject_reasons(
            evaluation=expected_evaluation,
            strict_boundary=strict_boundary,
            semantic=semantic,
        )
        if reject_reasons:
            mismatch.extend(reject_reasons)
    decision = build_guided_stage_decision(
        stage="selection",
        requirement_id=guided_context.requirement_id if guided_context else "",
        requirement=requirement,
        round_index=round_index,
        generated_artifact={"selected_formula": selected_formula},
        expected_artifact={"selected_formula": expected_formula},
        mismatch_evidence=mismatch,
        llm_client=llm_client,
    )
    if mismatch and semantic.faithful_ir is not None:
        decision = _mark_deterministic_expected_mismatch_decision(
            decision, stage="selection"
        )
    rag_chunks = record_guided_stage_decision(
        retriever,
        requirement=requirement,
        semantic=semantic,
        decision_record=decision,
    )
    return _guided_decision_review_record(
        decision=decision,
        requirement=requirement,
        semantic=semantic,
        runtime_rag_chunks=rag_chunks,
        records_dir=records_dir,
    )


def _mark_deterministic_expected_mismatch_decision(
    decision: dict,
    *,
    stage: str,
) -> dict:
    """Keep an IR-derived artifact when an offline warmup expectation differs."""

    updated = dict(decision)
    updated["status"] = "deterministic_expected_mismatch"
    updated["decision"] = "keep_generated_for_failure_diagnosis"
    updated["artifact_origin"] = "generated_from_approved_ir"
    updated["fallback_used"] = False
    updated["verification_status"] = "failed_expected_comparison"
    updated["guided_correction"] = None
    updated["decision_reason"] = (
        f"{stage} is derived from the approved IR and full-trace checks; the "
        "offline expected artifact is not inserted into the candidate set."
    )
    updated["lesson_for_rag"] = (
        "When a deterministic IR-derived artifact differs from warmup gold, "
        "repair the upstream semantic range or trace evidence instead of "
        "retrofitting the expected formula."
    )
    updated["rationale_source"] = "local"
    return updated


def _guided_expected_skeleton_formulas(
    guided_context: GuidedLearningContext | None,
) -> list[Formula]:
    if guided_context is None:
        return []
    skeletons: list[Formula] = []
    seen: set[str] = set()
    for text in guided_context.expected_restricted_skeletons:
        try:
            formula = parse_formula(normalize_boolean_operator_text(text))
        except ValueError:
            continue
        key = str(formula)
        if key in seen:
            continue
        seen.add(key)
        skeletons.append(formula)
    return skeletons


def _expected_hole_candidate(
    *,
    semantic: SemanticStructure,
    skeletons: list[Formula],
    template_text: str,
) -> HoleCandidate:
    return HoleCandidate(
        skeleton=skeletons[0] if skeletons else parse_formula(semantic.trigger or "ap"),
        template_text=normalize_boolean_operator_text(template_text),
        holes=_holes_from_template(template_text, semantic),
        rag_context=(),
        explanation="guided fallback adopted expected_hole_template.",
        candidate_id="HC-EXPECTED",
    )


def _holes_from_template(
    template_text: str,
    semantic: SemanticStructure,
) -> tuple[TemporalHole, ...]:
    holes: list[TemporalHole] = []
    seen: set[str] = set()
    for match in re.finditer(r"\bH_[A-Za-z0-9_]*\b", template_text):
        hole_id = match.group(0)
        if hole_id in seen:
            continue
        seen.add(hole_id)
        if hole_id == "H_scope":
            holes.append(
                TemporalHole(
                    hole_id=hole_id,
                    target="whole formula scope",
                    allowed_ops=("epsilon", "G"),
                    reason="guided fallback scope hole from expected template.",
                )
            )
        elif hole_id == "H_response":
            holes.append(
                TemporalHole(
                    hole_id=hole_id,
                    target=semantic.response or "response fragment",
                    allowed_ops=("epsilon", "X", "F", "G", "XF"),
                    reason="guided fallback response hole from expected template.",
                )
            )
        else:
            holes.append(
                TemporalHole(
                    hole_id=hole_id,
                    target="formula fragment",
                    allowed_ops=("epsilon", "X", "F", "G", "XF"),
                    reason="guided fallback generic hole from expected template.",
                )
            )
    return tuple(holes)


def _expected_temporal_completions(
    *,
    guided_context: GuidedLearningContext | None,
    hole_candidates: list[HoleCandidate],
    semantic: SemanticStructure,
) -> list[TemporalCompletion]:
    if guided_context is None:
        return []
    formula_text = _guided_expected_formula_text(guided_context)
    formula = parse_formula(normalize_boolean_operator_text(formula_text))
    assignments = tuple(guided_context.approved_assignments)
    candidates = hole_candidates or [
        HoleCandidate(
            skeleton=parse_formula("fallback_ap"),
            template_text="fallback_ap",
            holes=(),
            candidate_id="HC-EXPECTED",
        )
    ]
    completions: list[TemporalCompletion] = []
    for index, candidate in enumerate(candidates, start=1):
        if candidate.holes:
            _validate_completion_assignments(
                hole_candidate=candidate,
                assignments=assignments,
                index=index,
            )
            rendered = _render_formula_from_template(candidate, assignments)
            if rendered != formula:
                raise ValueError(
                    "completion_formula_assignment_mismatch: guided formula "
                    "does not match template+assignments"
                )
        ir = semantic.faithful_ir
        policy = (
            build_strict_formula_closure_policy(
                ir,
                semantic.approved_semantic_ir,
            )
            if ir is not None
            else FormulaClosurePolicy(
                allowed_aps=tuple(
                    dict.fromkeys(
                        (
                            *semantic.aps,
                            *(
                                item
                                for item in (semantic.trigger, semantic.response)
                                if item and item not in {"unknown", "unknown_ap"}
                            ),
                        )
                    )
                ),
                optional_aps=tuple(
                    dict.fromkeys(
                        (
                            *semantic.aps,
                            *(
                                item
                                for item in (semantic.trigger, semantic.response)
                                if item and item not in {"unknown", "unknown_ap"}
                            ),
                        )
                    )
                ),
            )
        )
        closure = validate_formula_closure(
            formula,
            policy,
            ir_version=(ir.ir_version if ir is not None else ""),
        )
        if not closure.passed:
            raise ValueError(
                "guided completion failed FormulaClosurePolicy: "
                + "; ".join(closure.reasons)
            )
        completions.append(
            TemporalCompletion(
                formula=formula,
                assignments=assignments,
                operator_derivation=(
                    "guided fallback adopted an independently validated expected candidate.",
                ),
                candidate_id=candidate.candidate_id or "HC-EXPECTED",
                template_text=candidate.template_text,
                ir_version=(ir.ir_version if ir is not None else ""),
                closure_report=closure,
            )
        )
    return completions


def _guided_decision_review_record(
    *,
    decision: dict,
    requirement: str,
    semantic: SemanticStructure,
    runtime_rag_chunks: list[str],
    records_dir: str,
) -> dict:
    record = {
        "stage": f"{decision['stage']}_guided_decision",
        "round": decision["round"],
        "requirement": requirement,
        "semantic": asdict(semantic),
        "approved": True,
        "runtime_rag_chunks": runtime_rag_chunks,
        "guided_decision": decision,
        "comments": decision["decision_reason"],
    }
    record["record_path"] = str(_save_guided_decision_record(record, records_dir))
    return record


def _save_guided_decision_record(record: dict, records_dir: str) -> Path:
    path = Path(records_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = (
        f"{record['stage']}_round_{record['round']}_{timestamp}.json"
    )
    record_path = path / filename
    record["record_path"] = str(record_path)
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record_path


def _evaluation_to_decision_artifact(
    evaluation: FormulaConsistencyEvaluation,
) -> dict:
    return {
        "formula": str(evaluation.formula),
        "pos_sat": evaluation.pos_sat,
        "neg_rej": evaluation.neg_rej,
        "bound_match": evaluation.bound_match,
        "score": evaluation.score,
        "failure_summary": list(evaluation.failure_summary),
    }


def _canonical_guided_text(text: str) -> str:
    return " ".join(normalize_boolean_operator_text(text).split())


def _label_source_for_experiment(config: ExperimentConfig) -> str:
    if config.mode == "formula_oracle":
        return "formula_oracle"
    if config.mode == "independent_judge":
        return "independent_judge"
    return "human"


def _persistent_rag_allowed_for_experiment(
    semantic: SemanticStructure,
    config: ExperimentConfig,
    *,
    trace_coverage: TraceCoverageReport | None = None,
) -> bool:
    return (
        semantic.persistent_rag_allowed
        and config.data_split == "train"
        and (trace_coverage is None or trace_coverage.complete)
    )


@observed_stage("pipeline")
def run_traj_ltlf_pipeline(
    requirement: str,
    strict_boundary: bool = True,
    review_mode: str = "auto",
    max_review_rounds: int = 3,
    min_passed_skeletons: int = 5,
    records_dir: str = "review_records",
    trace_supplement_records_dir: str | None = None,
    max_trace_supplement_rounds: int = 1,
    llm_mode: str | None = "rules",
    llm_client: JSONLLMClient | None = None,
    approved_memory_file: str | None = DEFAULT_APPROVED_MEMORY_FILE,
    progress: TerminalProgress | None = None,
    progress_label: str | None = None,
    experiment_mode: str | None = None,
    data_split: str = "test",
    reference_formula: Formula | str | None = None,
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
    ap_representation_profile: APRepresentationProfile | None = None,
    declared_ap_catalog: DeclaredAPCatalog | None = None,
    ap_binding_context: APBindingContext | None = None,
    retriever: KeywordRetriever | None = None,
    structure_reviewer=None,
    input_semantics_profile: dict | None = None,
    max_semantic_revision_rounds: int = 0,
    requirement_id: str = "REQ-001",
    delivery_policy: str | None = None,
    max_assignments: int = 4096,
    prediction_namespace_contract: dict | None = None,
    source_input_hash: str | None = None,
) -> PipelineResult:
    """执行完整 Traj-LTLf 原型流程。"""

    if trace_supplement_records_dir is None:
        trace_supplement_records_dir = str(Path(records_dir).parent / "outtraces")
    effective_mode = resolve_experiment_mode(review_mode, experiment_mode)
    from delivery_policy import resolve_delivery_policy
    resolve_delivery_policy(delivery_policy, review_mode, effective_mode)
    if effective_mode == "formula_oracle":
        raise ValueError(
            "formula_oracle 只允许批量 warmup-only 样本运行；"
            "请使用 --warmup-file --warmup-only --reference-formula sample "
            "--data-split train。"
        )
    if is_sample_reference_formula(reference_formula):
        raise ValueError(
            "--reference-formula sample 只支持批量 warmup-only 样本运行。"
        )
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
    if retriever is None:
        retriever = build_persistent_retriever(approved_memory_file)
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
            requires_judge_client(review_mode, experiment_config.mode)
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

    from llm.overlap_client import overlap_client
    if review_mode == "manual":
        llm_client = overlap_client(llm_client)
    nlu_agent = NLUAgent(retriever, llm_client=llm_client)
    trace_agent = TraceGenerationAgent(retriever, llm_client=llm_client)
    skeleton_agent = SkeletonAgent(retriever, llm_client=llm_client)
    completion_agent = CandidateCompletionAgent(retriever, llm_client=llm_client)
    consistency_agent = TrajectoryConsistencyAgent(retriever)

    initial_ap_context = build_runtime_ap_binding_context(
        requirement_id=requirement_id,
        source_text=requirement,
        input_semantics_profile=input_semantics_profile,
        representation_profile=ap_representation_profile,
        declared_catalog=declared_ap_catalog,
        provenance=("single_pipeline_input",),
        base_context=ap_binding_context,
    )
    pre_ir = run_pre_ir_review(
        requirement_id=requirement_id,
        requirement=requirement,
        retriever=retriever,
        llm_client=llm_client,
        judge_client=judge_client,
        config=PreIRReviewConfig(
            segmentation_review=segmentation_review,
            plain_reading_review=plain_reading_review,
            max_segmentation_retries=max_segmentation_retries,
            max_surface_review_rounds=max_surface_review_rounds,
            max_plain_reading_rounds=max_plain_reading_rounds,
            max_plain_reading_revision_rounds=max_plain_reading_revision_rounds,
            segmentation_prompt_version=segmentation_prompt_version,
            plain_reading_prompt_version=plain_reading_prompt_version,
            data_split=data_split,
            semantic_fallback_enabled=plain_reading_semantic_fallback,
        ),
        interactive=requires_interactive_pre_ir_review(
            segmentation_review,
            plain_reading_review,
        ),
        generation_diagnostic_callback=(
            lambda diagnostic: save_plain_reading_generation_diagnostic(
                records_dir,
                requirement_id=requirement_id,
                diagnostic=diagnostic,
            )
        ),
        semantic_override_callback=(
            lambda override: save_human_semantic_override_checkpoint(
                records_dir,
                override=override,
            )
        ),
        ap_representation_profile=initial_ap_context.representation_profile,
        declared_ap_catalog=declared_ap_catalog,
        ap_candidates_available=bool(initial_ap_context.grounding_contract.identities
                                     or initial_ap_context.has_authoritative_catalog),
    )
    original_requirement = requirement
    requirement = pre_ir.effective_source_text or requirement
    if (
        requirement == original_requirement
        and pre_ir.effective_ap_inventory is None
    ):
        # Rebuilding a proposal-only context here used to discard identities
        # that were already visible in the frozen online source profile.
        effective_ap_context = initial_ap_context
    else:
        effective_ap_context = build_ap_binding_context(
            requirement_id=requirement_id,
            source_text=requirement,
            representation_profile=initial_ap_context.representation_profile,
            declared_catalog=declared_ap_catalog,
            active_inventory=pre_ir.effective_ap_inventory,
            identity_mapping_id=(
                pre_ir.human_semantic_override.ap_comparison_id
                if pre_ir.human_semantic_override is not None
                else ""
            ),
            provenance=("single_pipeline_effective_source",),
        )

    from ap_preparation import prepare_reviewed_ap_context
    if llm_client is not None:
        effective_ap_context = prepare_reviewed_ap_context(
            effective_ap_context, llm_client, segmentation=pre_ir.surface_segmentation,
            interactive=review_mode == "manual", records_dir=records_dir,
            plain_reading_review=pre_ir.plain_reading_review,
        ).context
    semantic, semantic_records = run_semantic_review(
        requirement=requirement,
        agent=nlu_agent,
        review_mode=review_mode,
        max_rounds=max_review_rounds,
        records_dir=records_dir,
        surface_segmentation=pre_ir.surface_segmentation,
        plain_reading_review=pre_ir.plain_reading_review,
        ap_binding_context=effective_ap_context,
        structure_reviewer=structure_reviewer,
    )
    semantic = replace(
        semantic,
        data_split=experiment_config.data_split,
        label_source="model_assistant" if llm_client is not None else _label_source_for_experiment(experiment_config),
        formula_source="none",
        formula_visibility=(),
        persistent_rag_allowed=_persistent_rag_allowed_for_experiment(
            semantic,
            experiment_config,
        ),
    )
    from formula_generation_fallback import is_formula_generation_fallback
    if is_formula_generation_fallback(semantic):
        return _semantic_review_pending_result(
            requirement=requirement, semantic=semantic, traces=[], review_records=semantic_records,
            trace_coverage=TraceCoverageReport(status="incomplete", coverage_debt=("SOURCE_GRAPH_GENERATION_FAILED",)),
            strict_boundary=strict_boundary, review_mode=review_mode, experiment_config=experiment_config,
            min_passed_skeletons=min_passed_skeletons, trace_evidence_policy=trace_evidence_policy,
        )
    semantic = attach_candidate_ir_set(semantic, requirement, max_branch_count=4)
    # Complete CandidateIRSets must be resolved before semantic promotion.
    # Promotion signs one IR version, so attempting it on the compatibility
    # seed would strand the discriminator behind its own trace-ready gate.
    from ir_ambiguity_adjudication import (
        adjudicate_ir_candidate_branches,
        requires_ir_candidate_adjudication,
    )
    if requires_ir_candidate_adjudication(semantic):
        candidate_outcome = adjudicate_ir_candidate_branches(
            requirement,
            semantic,
            records_dir=records_dir,
            llm_client=llm_client,
            data_split=experiment_config.data_split,
        )
        semantic = candidate_outcome.semantic
        report = semantic.ir_ambiguity_adjudication
        from human_review_session import semantic_review_budget
        semantic_records.append({
            "stage": "ir_candidate_trace_adjudication",
            "round": len(report.rounds) if report is not None else 0,
            "requirement": requirement,
            "candidate_set_id": (
                report.candidate_set_id if report is not None else ""
            ),
            "status": candidate_outcome.status,
            "selected_branch_id": candidate_outcome.selected_branch_id,
            "consistent_branch_ids": list(candidate_outcome.consistent_branch_ids),
            "supporting_query_ids": list(candidate_outcome.supporting_query_ids),
            "comments": candidate_outcome.reason,
            "trace_evidence_review_required": candidate_outcome.status != "unique",
            "provisional": candidate_outcome.status == "provisional_fallback",
            "human_review_budget": semantic_review_budget(requirement, semantic, records_dir),
        })
        if candidate_outcome.status != "unique":
            pending_coverage = TraceCoverageReport(
                status="incomplete",
                coverage_debt=("IR_CANDIDATE_DISCRIMINATION_UNRESOLVED",),
                executable_semantics=False,
                semantic_complete=False,
                quota_complete=False,
                reason=candidate_outcome.reason,
            )
            return _semantic_review_pending_result(
                requirement=requirement,
                semantic=replace(
                    semantic,
                    semantic_status="needs_semantic_review",
                    persistent_rag_allowed=False,
                ),
                traces=[],
                review_records=semantic_records,
                trace_coverage=pending_coverage,
                strict_boundary=strict_boundary,
                review_mode=review_mode,
                experiment_config=experiment_config,
                min_passed_skeletons=min_passed_skeletons,
                trace_evidence_policy=trace_evidence_policy,
            )
    ap = list(
        semantic.faithful_ir.ap_names
        if semantic.faithful_ir is not None
        else (semantic.trigger, semantic.response)
    )

    if semantic.faithful_ir is None and pattern_requires_human_learning(semantic):
        if progress is not None:
            progress.warn(f"Unknown pattern: {semantic.pattern}")
        pattern_result = run_pattern_learning_review(
            requirement_id=requirement_id,
            requirement=requirement,
            semantic=semantic,
            retriever=retriever,
            review_mode=review_mode,
            round_index=1,
            records_dir=records_dir,
            progress=progress,
        )
        if (
            (not pattern_result.approved and not pattern_result.provisional)
            or pattern_result.semantic is None
        ):
            raise RuntimeError(
                "未知模式缺少可用的人工学习结果，流程已停止；"
                "请补充 warmup/人工评审样例，使该模式能生成受限骨架和孔位模板。"
            )
        semantic = pattern_result.semantic
        semantic = attach_candidate_ir_set(semantic, requirement, max_branch_count=4)
        ap = list(semantic.aps or (semantic.trigger, semantic.response))

    # Promotion must be the final semantic-binding step before authoritative
    # trace generation.  Binding a resolution record after the session is
    # signed changes the IR version and correctly makes that session stale.
    promotion_readiness = audit_semantic_readiness(semantic.faithful_ir)
    if promotion_readiness.has_capability(CAN_BIND_RESOLUTION):
        semantic = attach_semantic_resolution_record(semantic)
    promotion_readiness = audit_semantic_readiness(semantic.faithful_ir)
    if promotion_readiness.has_capability(CAN_GENERATE_AUTHORITATIVE_TRACES):
        semantic = attach_candidate_ir_set(semantic, requirement, max_branch_count=4)
    promotion = promote_semantic_candidate(
        semantic,
        artifact_root=str(Path(records_dir) / "semantic_artifacts"),
    )
    if promotion.disposition != TRACE_READY:
        pending_semantic = replace(
            semantic,
            semantic_status="needs_semantic_review",
            persistent_rag_allowed=False,
        )
        pending_coverage = TraceCoverageReport(
            status="incomplete",
            coverage_debt=(promotion.reason_code,),
            executable_semantics=False,
            semantic_complete=False,
            quota_complete=False,
            reason=promotion.message,
        )
        return _semantic_review_pending_result(
            requirement=requirement,
            semantic=pending_semantic,
            traces=[],
            review_records=[
                *semantic_records,
                {
                    "stage": "semantic_promotion",
                    "status": promotion.disposition,
                    "reason_code": promotion.reason_code,
                    "candidate_path": promotion.candidate_path,
                    "message": promotion.message,
                },
            ],
            trace_coverage=pending_coverage,
            strict_boundary=strict_boundary,
            review_mode=review_mode,
            experiment_config=experiment_config,
            min_passed_skeletons=min_passed_skeletons,
            trace_evidence_policy=trace_evidence_policy,
        )
    authoritative_trace_session = create_authoritative_trace_session(
        promotion
    )
    semantic, traces, trace_records = run_trace_review(
        requirement=requirement,
        semantic=semantic,
        agent=trace_agent,
        requirement_id=requirement_id,
        review_mode=review_mode,
        max_rounds=max_review_rounds,
        records_dir=records_dir,
        experiment_config=experiment_config,
        judge_client=judge_client,
        return_semantic=True,
        trace_evidence_policy=trace_evidence_policy,
        authoritative_trace_session=authoritative_trace_session,
        input_semantics_profile=input_semantics_profile,
    )
    # A source-first trace conflict may demonstrate that the executable IR is
    # wrong. Before evidence freezing, generate a genuinely new IR version and
    # rerun all semantic/promotion/trace gates. This is bounded, versioned and
    # never chooses by reference formula or candidate score.
    revision_limit = max(0, min(2, int(max_semantic_revision_rounds)))
    from human_review_session import REVIEW_STOP_DEBTS
    if trace_records and REVIEW_STOP_DEBTS.intersection(trace_records[-1].get("coverage_report", {}).get("coverage_debt", ())):
        revision_limit = 0
    from semantic_progress import ConflictVector, ProgressGuard, SemanticProgressStop
    revision_progress = None
    for revision_index in range(1, revision_limit + 1):
        packets = tuple(
            packet
            for packet in (
                trace_records[-1].get("semantic_failure_packets", ())
                if trace_records else ()
            )
            if packet.get("requires_semantic_version_change") is True
        )
        if not packets:
            break
        from semantic_failure_packet import semantic_revision_preflight_violations
        preflight_violations = semantic_revision_preflight_violations(
            semantic,
            packets,
            input_profile=input_semantics_profile,
        )
        if preflight_violations:
            trace_records.append({
                "stage": "semantic_revision_preflight",
                "round": revision_index,
                "status": "request_not_sent",
                "reason_code": "SEMANTIC_REVISION_AUTHORITY_INVALID",
                "violations": list(preflight_violations),
                "semantic_failure_packets": list(packets),
                "trace_evidence_review_required": True,
                "coverage_report": asdict(TraceCoverageReport(
                    status="incomplete",
                    coverage_debt=("SEMANTIC_REVISION_AUTHORITY_INVALID",),
                    executable_semantics=False,
                    semantic_complete=False,
                    quota_complete=False,
                    reason=(
                        "Semantic revision authority failed local preflight; "
                        "no provider request was sent."
                    ),
                )),
            })
            break
        emit(
            "failure_packet_preflight_passed",
            requirement_id=semantic.faithful_ir.requirement_id,
            ir_version=semantic.faithful_ir.ir_version,
            ir_artifact_hash=semantic.faithful_ir.artifact_hash,
            input_profile_hash=(input_semantics_profile or {}).get("profile_hash", ""),
            packet_ids=[packet["packet_id"] for packet in packets],
            entrypoint="run_traj_ltlf_pipeline",
        )
        if revision_progress is None:
            revision_progress = ProgressGuard(semantic.faithful_ir, ConflictVector(trace_label_conflicts=len(packets)))
        prior_ir_version = semantic.faithful_ir.ir_version
        conflict_cases = tuple(
            {
                "trace": packet.get("minimal_conflict_trace", ()),
                "expected": packet.get("resolved_trace_label") == "SAT",
            }
            for packet in packets
            if packet.get("resolved_trace_label") in {"SAT", "UNSAT"}
        )
        regression_cases = tuple(
            {"trace": trace.trace, "expected": trace.expected}
            for trace in traces
            if trace.annotation_locked
        )
        if not conflict_cases:
            trace_records.append({
                "stage": "typed_semantic_revision",
                "round": revision_index,
                "status": "request_not_sent",
                "reason_code": "SEMANTIC_REVISION_TYPED_AUTHORITY_INCOMPLETE",
                "semantic_failure_packets": list(packets),
                "trace_evidence_review_required": True,
            })
            break
        from typed_patch_formal_entry import run_formal_typed_patch
        formal_patch = run_formal_typed_patch(
            source_text=requirement,
            input_profile=input_semantics_profile or {},
            semantic=semantic,
            failure_packets=packets,
            main_client=llm_client,
            reviewer_client=judge_client,
            before_vector=ConflictVector(trace_label_conflicts=len(packets)),
            conflict_cases=conflict_cases,
            regression_cases=regression_cases,
        )
        if not formal_patch.committed:
            trace_records.append({
                "stage": "typed_semantic_revision",
                "round": revision_index,
                "status": "rolled_back",
                "reason_code": formal_patch.evidence.get("reason_code", "TYPED_PATCH_ROLLED_BACK"),
                "formal_entry_evidence": formal_patch.evidence,
                "semantic_failure_packets": list(packets),
                "trace_evidence_review_required": True,
            })
            break
        transaction_ir = formal_patch.ir
        transaction_evidence = formal_patch.evidence
        revised = replace(
            semantic,
            faithful_ir=transaction_ir,
            candidate_ir_set=None,
            approved_semantic_ir=None,
            labeled_trace_evidence=None,
            semantic_evidence_gate=None,
            semantic_resolution=None,
            ir_ambiguity_adjudication=None,
            semantic_status="ready",
            persistent_rag_allowed=False,
        )
        revision_records = [{
            "stage": "typed_semantic_revision",
            "round": revision_index,
            "status": "committed",
            "reason_code": transaction_evidence["reason_code"],
            "base_ir_version": prior_ir_version,
            "ir_version": transaction_ir.ir_version,
            "before_conflicts": transaction_evidence["before_conflicts"],
            "after_conflicts": transaction_evidence["after_conflicts"],
            "replayed_conflict_count": transaction_evidence["replayed_conflict_count"],
            "replayed_regression_count": transaction_evidence["replayed_regression_count"],
            "formal_entry_evidence": transaction_evidence,
        }]
        revised = replace(
            revised,
            data_split=experiment_config.data_split,
            label_source="model_assistant" if llm_client is not None else _label_source_for_experiment(experiment_config),
            formula_source="none",
            formula_visibility=(),
            persistent_rag_allowed=_persistent_rag_allowed_for_experiment(
                revised, experiment_config
            ),
        )
        revised = attach_candidate_ir_set(revised, requirement, max_branch_count=4)
        semantic_records.extend(revision_records)
        from semantic_failure_packet import semantic_revision_scope_violations
        scope_violations = semantic_revision_scope_violations(
            semantic, revised, packets
        )
        if scope_violations:
            trace_records.append({
                "stage": "semantic_revision",
                "round": revision_index,
                "status": "unauthorized_change_rolled_back",
                "reason_code": "SEMANTIC_REVISION_SCOPE_VIOLATION",
                "violations": list(scope_violations),
                "semantic_failure_packets": list(packets),
                "trace_evidence_review_required": True,
                "coverage_report": asdict(TraceCoverageReport(
                    status="incomplete",
                    coverage_debt=("SEMANTIC_REVISION_SCOPE_VIOLATION",),
                    executable_semantics=False,
                    semantic_complete=False,
                    quota_complete=False,
                    reason="Semantic revision changed fields outside packet authorization; rollback applied.",
                )),
            })
            break
        from semantic_progress import semantic_fingerprint
        if semantic_fingerprint(revised.faithful_ir) in revision_progress.seen:
            trace_records.append({
                "stage": "semantic_revision",
                "round": revision_index,
                "status": "no_progress",
                "reason_code": "SEMANTIC_REVISION_NO_PROGRESS",
                "trace_evidence_review_required": True,
                "semantic_failure_packets": list(packets),
                "coverage_report": asdict(TraceCoverageReport(
                    status="incomplete",
                    coverage_debt=("SEMANTIC_REVISION_NO_PROGRESS",),
                    executable_semantics=False,
                    semantic_complete=False,
                    quota_complete=False,
                    reason="Semantic revision reproduced the same IR version.",
                )),
            })
            break
        prior_semantic, prior_traces = semantic, traces
        semantic = revised
        revision_readiness = audit_semantic_readiness(semantic.faithful_ir)
        if revision_readiness.has_capability(CAN_BIND_RESOLUTION):
            semantic = attach_semantic_resolution_record(semantic)
        revision_readiness = audit_semantic_readiness(semantic.faithful_ir)
        if revision_readiness.has_capability(CAN_GENERATE_AUTHORITATIVE_TRACES):
            semantic = attach_candidate_ir_set(semantic, requirement, max_branch_count=4)
        revised_promotion = promote_semantic_candidate(
            semantic,
            artifact_root=str(Path(records_dir) / "semantic_artifacts"),
        )
        if revised_promotion.disposition != TRACE_READY:
            pending_semantic = replace(
                prior_semantic,
                semantic_status="needs_semantic_review",
                persistent_rag_allowed=False,
            )
            pending_coverage = TraceCoverageReport(
                status="incomplete",
                coverage_debt=(revised_promotion.reason_code,),
                executable_semantics=False,
                semantic_complete=False,
                quota_complete=False,
                reason=revised_promotion.message,
            )
            return _semantic_review_pending_result(
                requirement=requirement,
                semantic=pending_semantic,
                traces=[],
                review_records=[*semantic_records, *trace_records, {
                    "stage": "semantic_revision_promotion",
                    "round": revision_index,
                    "status": revised_promotion.disposition,
                    "reason_code": revised_promotion.reason_code,
                    "candidate_path": revised_promotion.candidate_path,
                    "message": revised_promotion.message,
                }],
                trace_coverage=pending_coverage,
                strict_boundary=strict_boundary,
                review_mode=review_mode,
                experiment_config=experiment_config,
                min_passed_skeletons=min_passed_skeletons,
                trace_evidence_policy=trace_evidence_policy,
            )
        revised_session = create_authoritative_trace_session(revised_promotion)
        semantic, traces, revised_trace_records = run_trace_review(
            requirement=requirement,
            semantic=semantic,
            agent=trace_agent,
            requirement_id=requirement_id,
            review_mode=review_mode,
            max_rounds=max_review_rounds,
            records_dir=records_dir,
            experiment_config=experiment_config,
            judge_client=judge_client,
            return_semantic=True,
            trace_evidence_policy=trace_evidence_policy,
            authoritative_trace_session=revised_session,
            input_semantics_profile=input_semantics_profile,
        )
        trace_records.extend(revised_trace_records)
        remaining_packets = tuple(
            packet
            for packet in (
                revised_trace_records[-1].get("semantic_failure_packets", ())
                if revised_trace_records else ()
            )
            if packet.get("requires_semantic_version_change") is True
        )
        before_vector = revision_progress.vector
        after_vector = ConflictVector(
            trace_label_conflicts=len(remaining_packets),
            readiness_debt=len(audit_semantic_readiness(semantic.faithful_ir).findings),
        )
        try:
            revision_progress.accept(semantic.faithful_ir, after_vector)
            made_progress = True
        except SemanticProgressStop:
            made_progress = False
        if not made_progress:
            semantic, traces = prior_semantic, prior_traces
            trace_records.append({
                "stage": "semantic_revision",
                "round": revision_index,
                "status": "no_strict_progress_rolled_back",
                "reason_code": "SEMANTIC_REVISION_NO_STRICT_PROGRESS",
                "previous_conflict_count": len(packets),
                "remaining_conflict_count": len(remaining_packets),
                "before_conflict_vector": asdict(before_vector),
                "after_conflict_vector": asdict(after_vector),
                "trace_evidence_review_required": True,
                "coverage_report": asdict(TraceCoverageReport(
                    status="incomplete",
                    coverage_debt=("SEMANTIC_REVISION_NO_STRICT_PROGRESS",),
                    executable_semantics=False,
                    semantic_complete=False,
                    quota_complete=False,
                    reason="Semantic revision did not strictly reduce source/trace conflicts; rollback applied.",
                )),
            })
            break
        authoritative_trace_session = revised_session
    ap = list(
        semantic.faithful_ir.ap_names
        if semantic.faithful_ir is not None
        else (semantic.trigger, semantic.response)
    )
    if open_semantic_challenges(semantic) or (
        trace_records
        and trace_records[-1].get("trace_evidence_review_required", False)
    ):
        raw_coverage = (
            trace_records[-1].get("coverage_report", {})
            if trace_records
            else {}
        )
        trace_coverage = TraceCoverageReport(**raw_coverage)
        return _semantic_review_pending_result(
            requirement=requirement,
            semantic=semantic,
            traces=traces,
            review_records=semantic_records + trace_records,
            trace_coverage=trace_coverage,
            strict_boundary=strict_boundary,
            review_mode=review_mode,
            experiment_config=experiment_config,
            min_passed_skeletons=min_passed_skeletons,
            trace_evidence_policy=trace_evidence_policy,
        )
    canonical_trace_plan = compile_executable_trace_plan(semantic.faithful_ir)
    trace_evidence_context = build_trace_evidence_context(
        canonical_trace_plan,
        trace_evidence_policy,
    )
    traces = filter_traces_for_policy(traces, trace_evidence_policy)
    trace_coverage = assess_authoritative_trace_coverage(
        traces,
        semantic,
        trace_session=authoritative_trace_session,
        requirement=requirement,
        retry_count=max(0, len(trace_records) - 1),
        allow_degraded=any(
            record.get("coverage_report", {}).get("status")
            == "degraded_unproven"
            for record in trace_records
        ),
        trace_plan=trace_evidence_context.effective_plan,
    )

    return run_traj_ltlf_pipeline_from_reviewed_inputs(
        requirement=requirement,
        semantic=semantic,
        traces=traces,
        initial_review_records=semantic_records + trace_records,
        retriever=retriever,
        strict_boundary=strict_boundary,
        review_mode=review_mode,
        max_review_rounds=max_review_rounds,
        min_passed_skeletons=min_passed_skeletons,
        records_dir=records_dir,
        trace_supplement_records_dir=trace_supplement_records_dir,
        max_trace_supplement_rounds=max_trace_supplement_rounds,
        llm_mode=llm_mode,
        llm_client=llm_client,
        approved_memory_file=approved_memory_file,
        progress=progress,
        progress_label=progress_label,
        trace_coverage=trace_coverage,
        experiment_config=experiment_config,
        judge_client=judge_client,
        trace_evidence_policy=trace_evidence_policy,
        ap_binding_context=effective_ap_context,
        authoritative_trace_session=authoritative_trace_session,
    )


def _semantic_review_pending_result(
    *,
    requirement: str,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    review_records: list[dict],
    trace_coverage: TraceCoverageReport,
    strict_boundary: bool,
    review_mode: str,
    experiment_config: ExperimentConfig,
    min_passed_skeletons: int,
    trace_evidence_policy: TraceEvidencePolicy = TraceEvidencePolicy(),
) -> PipelineResult:
    """Preserve semantic authority state and deliver a compiled fallback."""

    from candidate_delivery import deliver_formula_candidate
    return deliver_formula_candidate(PipelineResult(
        requirement=requirement,
        semantic=semantic,
        ap=list(
            semantic.faithful_ir.ap_names
            if semantic.faithful_ir is not None
            else semantic.aps or (semantic.trigger, semantic.response)
        ),
        traces=traces,
        skeletons=[],
        skeleton_diagnostics=[],
        skeleton_rounds=[],
        min_passed_skeletons=min_passed_skeletons,
        hole_candidates=[],
        completion_decisions=[],
        completion_rounds=[],
        candidates=[],
        evaluations=[],
        selected=None,
        fallback_candidate=None,
        selection_status="semantic_review_required",
        selection_review_notes=(
            "A structured trace-discovered semantic challenge is pending; "
            "the last validated IR remains unchanged.",
        ),
        selection_rag_chunks=[],
        strict_boundary=strict_boundary,
        review_mode=review_mode,
        review_records=review_records,
        trace_coverage=trace_coverage,
        experiment_mode=experiment_config.mode,
        oracle_assisted=False,
        diagnostic_only=False,
        candidate_status="blocked_by_semantic_review",
        automatic_finalization_allowed=False,
        semantic_status="needs_semantic_review",
        evidence_status="incomplete",
        approval_status="not_approved",
        learning_status="writeback_blocked",
        trace_evidence_policy=trace_evidence_policy,
        canonical_trace_coverage=trace_coverage,
        experimental_only=trace_evidence_policy.experimental_only,
        boundary_metric_available=any(
            is_boundary_trace(trace) for trace in traces
        ),
        ap_context_fingerprint=(
            semantic.faithful_ir.ap_context_fingerprint
            if semantic.faithful_ir is not None
            else ""
        ),
    ))


def _normalize_resume_stage(stage: str) -> str:
    normalized = (stage or "skeleton").strip().lower()
    aliases = {
        "pipeline": "skeleton",
        "semantic": "skeleton",
        "trace": "skeleton",
        "skeleton": "skeleton",
        "completion": "completion",
        "selection": "selection",
    }
    if normalized not in aliases:
        raise ValueError(f"不支持的 start_stage: {stage}")
    return aliases[normalized]


def _load_resume_formulas(
    *,
    records_dir: str,
    requirement: str,
    stage: str,
    ir_version: str,
) -> list[Formula]:
    root = Path(records_dir)
    if not root.exists():
        return []
    paths = sorted(
        root.glob(f"{stage}_round_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if payload.get("requirement") != requirement:
            continue
        if not payload.get("approved"):
            continue
        payload_ir_version = str(
            payload.get("semantic", {})
            .get("faithful_ir", {})
            .get("ir_version", "")
        )
        if payload_ir_version != ir_version:
            continue
        if stage == "skeleton":
            formula_texts = [
                diagnostic.get("skeleton", "")
                for diagnostic in payload.get("diagnostics", [])
                if diagnostic.get("passed")
            ]
        else:
            formula_texts = list(payload.get("candidate_formulas", []))
        formulas: list[Formula] = []
        for formula_text in formula_texts:
            try:
                formulas.append(parse_formula(str(formula_text)))
            except ValueError:
                continue
        if formulas:
            return formulas
    return []


def _load_resume_completion_decisions(
    *,
    records_dir: str,
    requirement: str,
    semantic: SemanticStructure,
) -> list[CompletionDecision]:
    """Restore explicit hole assignments for a selection-only rerun.

    The assignments come from the approved completion artifact.  They are not
    reconstructed from either the formula or the frozen IR.
    """

    ir = semantic.faithful_ir
    if ir is None:
        return []
    root = Path(records_dir)
    if not root.exists():
        return []
    paths = sorted(
        root.glob("completion_round_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    closure_policy = build_strict_formula_closure_policy(
        ir,
        semantic.approved_semantic_ir,
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if payload.get("requirement") != requirement:
            continue
        if not payload.get("approved"):
            continue
        payload_ir_version = str(
            payload.get("semantic", {})
            .get("faithful_ir", {})
            .get("ir_version", "")
        )
        if payload_ir_version != ir.ir_version:
            continue
        decisions: list[CompletionDecision] = []
        for item in payload.get("completion_decisions", []):
            try:
                formula = parse_formula(str(item.get("formula", "")))
                assignments = tuple(
                    (
                        str(assignment["hole_id"]),
                        str(assignment["selected_op"]),
                    )
                    for assignment in item.get("assignments", [])
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not assignments:
                continue
            closure_report = validate_formula_closure(
                formula,
                closure_policy,
                ir_version=ir.ir_version,
            )
            decisions.append(
                CompletionDecision(
                    formula=formula,
                    assignments=assignments,
                    boundary_match=float(item.get("boundary_match", 0.0)),
                    pos_sat=float(item.get("pos_sat", 0.0)),
                    neg_rej=float(item.get("neg_rej", 0.0)),
                    score=float(item.get("score", 0.0)),
                    operator_derivation=tuple(item.get("operator_derivation", [])),
                    trace_derivation=tuple(item.get("trace_derivation", [])),
                    reason=str(item.get("reason", "")),
                    candidate_id=str(item.get("candidate_id", "")),
                    template_text=str(item.get("template_text", "")),
                    ir_version=ir.ir_version,
                    closure_report=closure_report,
                )
            )
        if decisions:
            return decisions
    return []

def run_traj_ltlf_pipeline_from_reviewed_inputs(
    requirement: str,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    *,
    initial_review_records: list[dict] | None = None,
    retriever: KeywordRetriever | None = None,
    strict_boundary: bool = True,
    review_mode: str = "auto",
    max_review_rounds: int = 3,
    min_passed_skeletons: int = 5,
    records_dir: str = "review_records",
    trace_supplement_records_dir: str | None = None,
    max_trace_supplement_rounds: int = 1,
    llm_mode: str | None = "rules",
    llm_client: JSONLLMClient | None = None,
    approved_memory_file: str | None = DEFAULT_APPROVED_MEMORY_FILE,
    progress: TerminalProgress | None = None,
    progress_label: str | None = None,
    guided_context: GuidedLearningContext | None = None,
    trace_coverage: TraceCoverageReport | None = None,
    experiment_config: ExperimentConfig | None = None,
    judge_client: JSONLLMClient | None = None,
    start_stage: str = "skeleton",
    run_origin: str = "normal",
    trace_evidence_policy: TraceEvidencePolicy = TraceEvidencePolicy(),
    ap_binding_context: APBindingContext | None = None,
    authoritative_trace_session: AuthoritativeTraceSession | None = None,
) -> PipelineResult:
    """Run the remaining pipeline after semantic and trace review has passed."""

    if trace_supplement_records_dir is None:
        trace_supplement_records_dir = str(Path(records_dir).parent / "outtraces")
    if run_origin not in {"normal", "feedback_rerun"}:
        raise ValueError("run_origin must be normal or feedback_rerun")
    experiment_config = experiment_config or ExperimentConfig(
        mode=resolve_experiment_mode(review_mode, None),
    )
    if retriever is None:
        retriever = build_persistent_retriever(approved_memory_file)
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
        and requires_judge_client(review_mode, experiment_config.mode)
    ):
        message = "Initializing Judge LLM client"
        if progress is None:
            judge_client = build_judge_client_from_env()
        else:
            with progress.task(message):
                judge_client = build_judge_client_from_env()
            progress.ok(message + " ready")

    semantic = replace(semantic, formula_text="", formula_visibility=())
    semantic = attach_faithful_ir(
        semantic,
        requirement,
        ap_binding_context=ap_binding_context,
    )
    if semantic.faithful_ir is None:  # pragma: no cover
        raise RuntimeError("FaithfulSemanticIR is required")
    effective_source = natural_language_source(
        requirement_id=semantic.faithful_ir.requirement_id,
        source_text=requirement,
    )
    if not any(
        (
            semantic.semantic_source_contract_version,
            semantic.effective_source_fingerprint,
            semantic.effective_source_kind,
            semantic.effective_source_authority_record_id,
        )
    ):
        semantic = bind_semantic_to_effective_source(
            semantic,
            effective_source,
        )
    else:
        validate_semantic_source_binding(semantic)
        if (
            semantic.effective_source_fingerprint
            != effective_source.fingerprint
        ):
            raise ValueError(
                "reviewed semantic references a stale effective source"
            )
    semantic = attach_candidate_ir_set(semantic, requirement, max_branch_count=4)
    if authoritative_trace_session is None:
        reviewed_promotion = promote_semantic_candidate(
            semantic,
            artifact_root=str(Path(records_dir) / "semantic_artifacts"),
        )
        if reviewed_promotion.disposition != TRACE_READY:
            raise ValueError(
                "reviewed semantic is not trace-ready: "
                + reviewed_promotion.reason_code
            )
        authoritative_trace_session = create_authoritative_trace_session(
            reviewed_promotion
        )
    else:
        validate_authoritative_trace_session(
            authoritative_trace_session,
            semantic,
        )
    # Bind externally reviewed trace claims to the structural inventory before
    # candidate selection; this does not authorize IR witness execution.
    trace_obligation_plan = compile_trace_obligation_plan(semantic.faithful_ir)
    trace_evidence_context = build_trace_evidence_context(
        trace_obligation_plan,
        trace_evidence_policy,
    )
    traces = filter_traces_for_policy(traces, trace_evidence_policy)
    traces = bind_traces_to_ir_plan(
        traces,
        trace_obligation_plan,
        generation_source="reviewed_trace_binding",
    )
    previous_trace_coverage = trace_coverage
    trace_coverage = assess_authoritative_trace_coverage(
        traces,
        semantic,
        trace_session=authoritative_trace_session,
        requirement=requirement,
        guided=bool(guided_context and guided_context.expected_traces),
        retry_count=previous_trace_coverage.retry_count
        if previous_trace_coverage is not None
        else 0,
        allow_degraded=True,
        trace_plan=trace_evidence_context.effective_plan,
    )
    canonical_trace_coverage = assess_authoritative_trace_coverage(
        traces,
        semantic,
        trace_session=authoritative_trace_session,
        requirement=requirement,
        guided=bool(guided_context and guided_context.expected_traces),
        retry_count=trace_coverage.retry_count,
        allow_degraded=True,
        trace_plan=trace_evidence_context.canonical_plan,
    )
    if trace_coverage.status == "rejected":
        raise ValueError(trace_coverage.reason)
    semantic = freeze_approved_semantic_ir(
        semantic,
        traces,
        approval_record=(
            _latest_record_path(initial_review_records or (), stage="traces")
        ),
        trace_plan=trace_evidence_context.effective_plan,
        experimental_only=trace_evidence_policy.experimental_only,
    )
    if (
        semantic.faithful_ir is not None
        and trace_obligation_plan.ir_version != semantic.faithful_ir.ir_version
    ):
        # A uniquely discriminated semantic branch is a new IR artifact.  Rebind
        # reviewed formula-free claims before compiling downstream plans.
        trace_obligation_plan = compile_executable_trace_plan(semantic.faithful_ir)
        trace_evidence_context = build_trace_evidence_context(
            trace_obligation_plan,
            trace_evidence_policy,
        )
        traces = bind_traces_to_ir_plan(
            traces,
            trace_obligation_plan,
            generation_source="temporal_branch_rebinding",
        )
        reviewed_promotion = promote_semantic_candidate(
            semantic,
            artifact_root=str(Path(records_dir) / "semantic_artifacts"),
        )
        if reviewed_promotion.disposition != TRACE_READY:
            raise ValueError(
                "resolved semantic is not trace-ready: "
                + reviewed_promotion.reason_code
            )
        authoritative_trace_session = create_authoritative_trace_session(
            reviewed_promotion
        )
        trace_coverage = assess_authoritative_trace_coverage(
            traces,
            semantic,
            trace_session=authoritative_trace_session,
            requirement=requirement,
            guided=bool(guided_context and guided_context.expected_traces),
            retry_count=trace_coverage.retry_count,
            allow_degraded=True,
            trace_plan=trace_evidence_context.effective_plan,
        )
        canonical_trace_coverage = assess_authoritative_trace_coverage(
            traces,
            semantic,
            trace_session=authoritative_trace_session,
            requirement=requirement,
            guided=bool(guided_context and guided_context.expected_traces),
            retry_count=trace_coverage.retry_count,
            allow_degraded=True,
            trace_plan=trace_evidence_context.canonical_plan,
        )
        semantic = freeze_approved_semantic_ir(
            semantic,
            traces,
            approval_record=(
                _latest_record_path(initial_review_records or (), stage="traces")
            ),
            trace_plan=trace_evidence_context.effective_plan,
            experimental_only=trace_evidence_policy.experimental_only,
        )
    # The stale-plan baseline is the approved IR version from which the
    # authoritative downstream plans are about to be compiled, not the
    # pre-freeze draft version.
    final_freeze_input_version = semantic.faithful_ir.ir_version
    emit("evidence_frozen", semantic=semantic, traces=traces)

    approved_domain = semantic.approved_semantic_ir
    experimental_authorized = bool(
        trace_evidence_policy.experimental_only
        and semantic.semantic_evidence_gate is not None
        and semantic.semantic_evidence_gate.complete
    )
    if approved_domain is None or (
        not approved_domain.approved and not experimental_authorized
    ):
        raise _semantic_domain_not_approved_error(
            semantic,
            code="SEMANTIC_DOMAIN_NOT_APPROVED",
        )

    # Skeleton and hole plans become authoritative only after domain approval.
    constrained_skeleton_plan = compile_constrained_skeleton_plan(semantic.faithful_ir)
    completion_hole_plan = compile_completion_hole_plan(
        semantic.faithful_ir,
        constrained_skeleton_plan,
    )

    persistent_rag_allowed = semantic.persistent_rag_allowed
    semantic = replace(
        semantic,
        persistent_rag_allowed=_persistent_rag_allowed_for_experiment(
            semantic,
            experiment_config,
            trace_coverage=trace_coverage,
        ),
    )
    if trace_evidence_policy.experimental_only:
        semantic = replace(semantic, persistent_rag_allowed=False)

    skeleton_agent = SkeletonAgent(retriever, llm_client=llm_client)
    completion_agent = CandidateCompletionAgent(retriever, llm_client=llm_client)
    consistency_agent = TrajectoryConsistencyAgent(retriever)
    ap = list(semantic.faithful_ir.ap_names)
    base_review_records = list(initial_review_records or [])
    current_traces = list(traces)
    all_skeleton_records: list[dict] = []
    all_completion_records: list[dict] = []
    consistency_records: list[dict] = []
    guided_selection_records: list[dict] = []
    trace_supplement_records: list[dict] = []
    skeletons: list[Formula] = []
    skeleton_diagnostics: list[SkeletonDiagnostic] = []
    hole_candidates: list[HoleCandidate] = []
    completion_decisions: list[CompletionDecision] = []
    operator_resolution = OperatorResolutionReport(
        requirement_id=semantic.faithful_ir.requirement_id,
        ir_version=semantic.faithful_ir.ir_version,
    )
    candidates: list[Formula] = []
    evaluations: list[FormulaConsistencyEvaluation] = []
    selected: FormulaConsistencyEvaluation | None = None
    label = progress_label or "Requirement"
    source_run_id = (
        "run-"
        + semantic.faithful_ir.requirement_id
        + "-"
        + datetime.now().strftime("%Y%m%d%H%M%S%f")
    )
    normalized_start_stage = _normalize_resume_stage(start_stage)
    resume_skeletons = (
        _load_resume_formulas(
            records_dir=records_dir,
            requirement=requirement,
            stage="skeleton",
            ir_version=semantic.faithful_ir.ir_version,
        )
        if normalized_start_stage == "completion"
        else []
    )
    resume_completion_decisions = (
        _load_resume_completion_decisions(
            records_dir=records_dir,
            requirement=requirement,
            semantic=semantic,
        )
        if normalized_start_stage == "selection"
        else []
    )
    resume_candidates = [
        decision.formula for decision in resume_completion_decisions
    ]
    if normalized_start_stage == "completion" and not resume_skeletons:
        raise RuntimeError(
            "completion 阶段重跑缺少已批准 skeleton 阶段产物。"
        )
    if normalized_start_stage == "selection" and not resume_candidates:
        raise RuntimeError(
            "selection 阶段重跑缺少最近可用 completion 候选。"
        )
    resume_pending = normalized_start_stage in {"completion", "selection"}
    reuse_candidate_artifacts = False

    for supplement_round in range(1, max_trace_supplement_rounds + 2):
        if reuse_candidate_artifacts:
            # A trace-only delta invalidates evaluation/selection, not the
            # version-bound skeleton, hole plan, or candidate ASTs.
            reuse_candidate_artifacts = False
        elif resume_pending and normalized_start_stage == "selection":
            candidates = list(resume_candidates)
            skeletons = []
            skeleton_diagnostics = []
            hole_candidates = []
            completion_decisions = list(resume_completion_decisions)
        else:
            if resume_pending and normalized_start_stage == "completion":
                skeletons = list(resume_skeletons)
                skeleton_diagnostics = []
            else:
                skeletons, skeleton_diagnostics, skeleton_records = (
                    run_skeleton_generation_loop(
                        requirement=requirement,
                        semantic=semantic,
                        traces=current_traces,
                        agent=skeleton_agent,
                        retriever=retriever,
                        min_passed_skeletons=min_passed_skeletons,
                        max_rounds=max_review_rounds,
                        records_dir=records_dir,
                        progress=progress,
                        progress_label=label,
                        guided_context=guided_context,
                    )
                )
                all_skeleton_records.extend(skeleton_records)
            candidates, hole_candidates, completion_decisions, completion_records = (
                run_candidate_completion_loop(
                    requirement=requirement,
                    semantic=semantic,
                    skeletons=skeletons,
                    traces=current_traces,
                    agent=completion_agent,
                    consistency_agent=consistency_agent,
                    strict_boundary=strict_boundary,
                    max_rounds=max_review_rounds,
                    records_dir=records_dir,
                    progress=progress,
                    progress_label=label,
                    guided_context=guided_context,
                )
            )
            all_completion_records.extend(completion_records)
        resume_pending = False
        if progress is None:
            evaluations = consistency_agent.evaluate_candidates(
                candidates=candidates,
                traces=current_traces,
                strict_boundary=strict_boundary,
            )
        else:
            with progress.task(f"{label} final consistency evaluation"):
                evaluations = consistency_agent.evaluate_candidates(
                    candidates=candidates,
                    traces=current_traces,
                    strict_boundary=strict_boundary,
                )
            progress.ok(f"{label} evaluated candidates: {len(evaluations)}")
        decisions_by_formula = {
            decision.formula: decision for decision in completion_decisions
        }
        evaluations = [
            replace(
                evaluation,
                candidate_id=(
                    decisions_by_formula[evaluation.formula].candidate_id
                    if evaluation.formula in decisions_by_formula
                    else evaluation.candidate_id
                ),
                ir_version=(
                    decisions_by_formula[evaluation.formula].ir_version
                    if evaluation.formula in decisions_by_formula
                    else evaluation.ir_version
                ),
                closure_report=(
                    decisions_by_formula[evaluation.formula].closure_report
                    if evaluation.formula in decisions_by_formula
                    else evaluation.closure_report
                ),
            )
            for evaluation in evaluations
        ]
        consistency_resolution = _resolve_guided_consistency_evaluations(
            requirement=requirement,
            semantic=semantic,
            generated_evaluations=evaluations,
            traces=current_traces,
            consistency_agent=consistency_agent,
            guided_context=guided_context,
            retriever=retriever,
            round_index=1,
            records_dir=records_dir,
            strict_boundary=strict_boundary,
            llm_client=llm_client,
        )
        if consistency_resolution is not None:
            evaluations, consistency_record = consistency_resolution
            consistency_records.append(consistency_record)
            candidates = [evaluation.formula for evaluation in evaluations]
        operator_evidence = build_operator_trace_evidence(
            ir_version=semantic.faithful_ir.ir_version,
            hole_plan=completion_hole_plan,
            traces=current_traces,
            parent_ir_status=(
                "approved"
                if semantic.approved_semantic_ir is not None
                and semantic.approved_semantic_ir.approved
                else "experimental"
            ),
        )
        operator_resolution = evaluate_operator_resolution_from_evidence(
            completion_decisions=completion_decisions,
            evaluations=evaluations,
            hole_plan=completion_hole_plan,
            evidence=operator_evidence,
            strict_boundary=strict_boundary,
        )
        selected = select_operator_resolved_evaluation(
            report=operator_resolution,
            completion_decisions=completion_decisions,
            evaluations=evaluations,
        )
        if guided_context is not None:
            expected = _guided_expected_formula_text(guided_context)
            if selected is None or str(selected.formula) != expected:
                raise RuntimeError(
                    "warmup guided result is not the unique trajectory-resolved "
                    f"assignment: expected={expected or '<missing>'}"
                )
        if progress is not None:
            if selected is None:
                progress.warn(f"{label} final selection needs review")
            else:
                progress.ok(f"{label} selected: {selected.formula}")
        selection_record = _build_guided_selection_record(
            requirement=requirement,
            semantic=semantic,
            evaluations=evaluations,
            selected=selected,
            guided_context=guided_context,
            retriever=retriever,
            round_index=1,
            records_dir=records_dir,
            strict_boundary=strict_boundary,
            llm_client=llm_client,
        )
        if selection_record is not None:
            guided_selection_records.append(selection_record)

        interim_ir_coverage = evaluate_ir_coverage(
            semantic.faithful_ir,
            trace_evidence_context.effective_plan,
            current_traces,
            skeleton_plan=constrained_skeleton_plan,
            hole_plan=completion_hole_plan,
            operator_resolution=operator_resolution,
        )
        supplement_needed = (
            supplement_round <= max_trace_supplement_rounds
            and (
                operator_resolution.status == "ambiguous"
                or not trace_coverage.semantic_complete
                or not interim_ir_coverage.complete
            )
        )
        if supplement_needed:
            automatic_queries = generate_coverage_debt_queries(
                semantic=semantic,
                trace_plan=trace_evidence_context.effective_plan,
                trace_coverage=trace_coverage,
                ir_coverage=interim_ir_coverage,
                existing_query_ids={
                    trace.trace_id or trace.name for trace in current_traces
                },
                data_split=experiment_config.data_split,
                label_source=_label_source_for_experiment(experiment_config),
            )
            automatic_queries = filter_queries_for_policy(
                automatic_queries,
                trace_evidence_policy,
            )
            if automatic_queries:
                trace_supplement_records.append(
                    {
                        "stage": "trace_supplement_policy",
                        "round": supplement_round,
                        "approved": True,
                        "status": "nl_coverage_debt_queries_generated",
                        "experiment_mode": experiment_config.mode,
                        "record_path": "",
                        "storage": "inline",
                        "runtime_rag_chunks": [],
                        "ir_coverage": asdict(interim_ir_coverage),
                        "trace_coverage": asdict(trace_coverage),
                        "comments": (
                            "根据 NL/IR coverage debt 生成补充轨迹查询；"
                            "这些查询不使用候选 LTLf 公式，标签必须来自 NL 时序模拟审核。"
                        ),
                    }
                )
            else:
                trace_supplement_records.append(
                    {
                        "stage": "trace_supplement_policy",
                        "round": supplement_round,
                        "approved": True,
                        "status": "no_formula_free_debt_query_available",
                        "experiment_mode": experiment_config.mode,
                        "record_path": "",
                        "storage": "inline",
                        "runtime_rag_chunks": [],
                        "comments": (
                            "当前没有可生成的 NL/IR coverage-debt 查询；"
                            "流程保留未解析状态，不读取候选公式构造补轨。"
                        ),
                    }
                )
            supplement_outcome = _adjudicate_supplement_queries(
                requirement=requirement,
                semantic=semantic,
                queries=automatic_queries,
                experiment_config=experiment_config,
                review_mode=review_mode,
                round_index=supplement_round,
                records_dir=trace_supplement_records_dir,
                judge_client=judge_client,
                llm_client=llm_client,
            )
            automatic_traces = (
                list(supplement_outcome.accepted_traces)
                if supplement_outcome is not None
                else []
            )
            unresolved_automatic_queries = (
                list(supplement_outcome.unresolved_queries)
                if supplement_outcome is not None
                else []
            )
            if (
                supplement_outcome is not None
                and supplement_outcome.has_semantic_challenge
            ):
                semantic = attach_semantic_challenges(
                    semantic,
                    supplement_outcome.semantic_challenges,
                )
                trace_coverage = replace(
                    trace_coverage,
                    status="incomplete",
                    coverage_debt=tuple(
                        dict.fromkeys(
                            trace_coverage.coverage_debt
                            + supplement_outcome.coverage_debt
                        )
                    ),
                    executable_semantics=False,
                    semantic_complete=False,
                    reason=(
                        "Supplemental trace review found a structured semantic "
                        "challenge; the validated IR was preserved."
                    ),
                )
                trace_supplement_records.append(
                    {
                        "stage": "trace_supplement_annotation",
                        "round": supplement_round,
                        "approved": False,
                        "status": "semantic_review_required",
                        "experiment_mode": experiment_config.mode,
                        "record_path": "",
                        "storage": "inline",
                        "semantic_challenges": [
                            asdict(item)
                            for item in supplement_outcome.semantic_challenges
                        ],
                        "runtime_rag_chunks": [],
                        "comments": (
                            "结构化语义挑战已保存在正式 IR 外部；等待语义复核。"
                        ),
                    }
                )
                break
            if automatic_traces:
                previous_traces = list(current_traces)
                supplement_brief = build_trace_adjudication_brief(
                    requirement, semantic
                )
                reconciliation = reconcile_trace_evidence(
                    semantic=semantic,
                    brief=supplement_brief,
                    existing_traces=current_traces,
                    newly_adjudicated=automatic_traces,
                )
                conflict_resolution = resolve_trace_evidence_disputes(
                    semantic=semantic,
                    brief=supplement_brief,
                    reconciliation=reconciliation,
                    resolver_client=None,
                )
                reconciliation = conflict_resolution.reconciliation
                current_traces = list(reconciliation.active_traces)
                if reconciliation.unresolved_disputes:
                    trace_coverage = replace(
                        trace_coverage,
                        status="incomplete",
                        actual_count=len(current_traces),
                        coverage_debt=tuple(
                            dict.fromkeys(
                                trace_coverage.coverage_debt
                                + reconciliation.coverage_debt
                            )
                        ),
                        executable_semantics=False,
                        semantic_complete=False,
                        quota_complete=False,
                        reason=(
                            "Supplemental trace evidence could not be safely "
                            "resolved in the background; existing semantic review "
                            "is required."
                        ),
                    )
                    trace_supplement_records.append(
                        {
                            "stage": "trace_supplement_annotation",
                            "round": supplement_round,
                            "approved": False,
                            "status": "semantic_review_required",
                            "trace_evidence_disputes": [
                                asdict(item)
                                for item in reconciliation.unresolved_disputes
                            ],
                            "background_conflict_resolutions": [
                                asdict(item)
                                for item in conflict_resolution.reports
                            ],
                            "record_path": "",
                            "storage": "inline",
                            "runtime_rag_chunks": [],
                            "comments": (
                                "冲突证据已隔离，候选评分未使用这些轨迹；"
                                "保留原有语义复核流程。"
                            ),
                        }
                    )
                    break
                prior_ir_version = semantic.faithful_ir.ir_version
                semantic = freeze_approved_semantic_ir(
                    semantic,
                    current_traces,
                    approval_record=_latest_record_path(
                        base_review_records + trace_supplement_records,
                        stage="traces",
                    ),
                    trace_plan=trace_evidence_context.effective_plan,
                    experimental_only=trace_evidence_policy.experimental_only,
                )
                ir_changed = semantic.faithful_ir.ir_version != prior_ir_version
                if ir_changed:
                    trace_obligation_plan = compile_executable_trace_plan(
                        semantic.faithful_ir
                    )
                    trace_evidence_context = build_trace_evidence_context(
                        trace_obligation_plan,
                        trace_evidence_policy,
                    )
                    current_traces = bind_traces_to_ir_plan(
                        current_traces,
                        trace_obligation_plan,
                        generation_source="supplemental_evidence_rebinding",
                    )
                    constrained_skeleton_plan = compile_constrained_skeleton_plan(
                        semantic.faithful_ir
                    )
                    completion_hole_plan = compile_completion_hole_plan(
                        semantic.faithful_ir,
                        constrained_skeleton_plan,
                    )
                    ap = list(semantic.faithful_ir.ap_names)
                    final_freeze_input_version = semantic.faithful_ir.ir_version
                    reviewed_promotion = promote_semantic_candidate(
                        semantic,
                        artifact_root=str(
                            Path(records_dir) / "semantic_artifacts"
                        ),
                    )
                    if reviewed_promotion.disposition != TRACE_READY:
                        raise ValueError(
                            "supplemented semantic is not trace-ready: "
                            + reviewed_promotion.reason_code
                        )
                    authoritative_trace_session = (
                        create_authoritative_trace_session(
                            reviewed_promotion
                        )
                    )
                target_trace_count = len(current_traces)
                trace_coverage = assess_authoritative_trace_coverage(
                    current_traces,
                    semantic,
                    trace_session=authoritative_trace_session,
                    requirement=requirement,
                    target_trace_count=target_trace_count,
                    retry_count=trace_coverage.retry_count + 1,
                    allow_degraded=True,
                    trace_plan=trace_evidence_context.effective_plan,
                )
                canonical_trace_coverage = assess_authoritative_trace_coverage(
                    current_traces,
                    semantic,
                    trace_session=authoritative_trace_session,
                    requirement=requirement,
                    target_trace_count=target_trace_count,
                    retry_count=trace_coverage.retry_count,
                    allow_degraded=True,
                    trace_plan=trace_evidence_context.canonical_plan,
                )
                semantic = freeze_approved_semantic_ir(
                    semantic,
                    current_traces,
                    approval_record=_latest_record_path(
                        base_review_records + trace_supplement_records,
                        stage="traces",
                    ),
                    trace_plan=trace_evidence_context.effective_plan,
                    experimental_only=trace_evidence_policy.experimental_only,
                )
                if not _semantic_execution_authorized(
                    semantic,
                    trace_evidence_policy,
                ):
                    raise _semantic_domain_not_approved_error(
                        semantic,
                        code="SEMANTIC_DOMAIN_NOT_APPROVED_AFTER_EVIDENCE_UPDATE",
                    )
                semantic = replace(
                    semantic,
                    persistent_rag_allowed=_persistent_rag_allowed_for_experiment(
                        replace(semantic, persistent_rag_allowed=persistent_rag_allowed),
                        experiment_config,
                        trace_coverage=trace_coverage,
                    ),
                )
                trace_supplement_records.append(
                    _save_trace_supplement_generation_record(
                        requirement=requirement,
                        semantic=semantic,
                        round_index=supplement_round,
                        existing_traces=previous_traces,
                        generated_traces=automatic_traces,
                        supplemented_traces=automatic_traces,
                        combined_traces=current_traces,
                        target_trace_count=target_trace_count,
                        target_trace_length=None,
                        runtime_rag_chunks=[],
                        comments=(
                            "自动 NL/coverage debt 补轨：查询来自缺失的"
                            "语义角色、歧义轴或 IR obligation；标签来自"
                            "独立的逐轨迹 NL 裁决。"
                        ),
                        records_dir=trace_supplement_records_dir,
                    )
                )
                if progress is not None:
                    progress.ok(
                        f"{label} automatic NL debt traces: "
                        f"{len(automatic_traces)}"
                    )
                reuse_candidate_artifacts = not ir_changed
                continue
            if unresolved_automatic_queries:
                trace_supplement_records.append(
                    {
                        "stage": "trace_supplement_annotation",
                        "round": supplement_round,
                        "approved": False,
                        "status": "ambiguity_retained",
                        "experiment_mode": experiment_config.mode,
                        "record_path": "",
                        "storage": "inline",
                        "queries": [
                            trace_query_to_dict(query)
                            for query in unresolved_automatic_queries
                        ],
                        "runtime_rag_chunks": [],
                        "comments": (
                            "NL/coverage debt 查询被标为 UNCLEAR/INFEASIBLE，"
                            "未进入候选评分。"
                        ),
                    }
                )
        # No reviewed formula-free query produced new evidence.  Candidate
        # formulas are never disclosed to a second supplement constructor.
        break

    semantic = freeze_approved_semantic_ir(
        semantic,
        current_traces,
        approval_record=_latest_record_path(
            base_review_records + trace_supplement_records,
            stage="traces",
        ),
        trace_plan=trace_evidence_context.effective_plan,
        experimental_only=trace_evidence_policy.experimental_only,
    )
    if not _semantic_execution_authorized(semantic, trace_evidence_policy):
        raise _semantic_domain_not_approved_error(
            semantic,
            code="SEMANTIC_DOMAIN_NOT_APPROVED_AT_FINAL_FREEZE",
        )
    if semantic.faithful_ir.ir_version != final_freeze_input_version:
        error = RuntimeError(
            "ApprovedSemanticIR changed after completion; downstream plans are stale"
        )
        error.pipeline_stage = "semantic_approval"
        error.diagnosis_error_class = "stale_derived_plan"
        raise error
    operator_evidence = build_operator_trace_evidence(
        ir_version=semantic.faithful_ir.ir_version,
        hole_plan=completion_hole_plan,
        traces=current_traces,
        parent_ir_status=(
            "approved"
            if semantic.approved_semantic_ir is not None
            and semantic.approved_semantic_ir.approved
            else "experimental"
        ),
    )
    operator_resolution = evaluate_operator_resolution_from_evidence(
        completion_decisions=completion_decisions,
        evaluations=evaluations,
        hole_plan=completion_hole_plan,
        evidence=operator_evidence,
        strict_boundary=strict_boundary,
    )
    final_passing_evaluations = evaluations_for_passing_assignments(
        report=operator_resolution,
        completion_decisions=completion_decisions,
        evaluations=evaluations,
    )
    selected = select_operator_resolved_evaluation(
        report=operator_resolution,
        completion_decisions=completion_decisions,
        evaluations=evaluations,
    )
    final_temporal_resolution: dict[str, str] | None = None
    final_temporal_resolution_error = ""
    if selected is not None:
        try:
            final_temporal_resolution = resolved_temporal_resolution_by_node(
                report=operator_resolution,
                hole_plan=completion_hole_plan,
            )
        except ValueError as exc:
            final_temporal_resolution_error = str(exc)
    ambiguity_retained = operator_resolution.status == "ambiguous"
    ir_coverage = evaluate_ir_coverage(
        semantic.faithful_ir,
        trace_evidence_context.effective_plan,
        current_traces,
        skeleton_plan=constrained_skeleton_plan,
        hole_plan=completion_hole_plan,
        operator_resolution=operator_resolution,
    )
    selection_outcome = select_with_ir_coverage_gate(
        [selected] if selected is not None else [],
        strict_boundary=strict_boundary,
        semantic=semantic,
        coverage=ir_coverage,
        run_origin=run_origin,
        degraded=trace_coverage.degraded,
        fallback=False,
        candidates_distinguished=not ambiguity_retained,
        operator_resolution=operator_resolution,
        temporal_resolution_by_node=final_temporal_resolution,
        temporal_resolution_error=final_temporal_resolution_error,
    )
    if experiment_config.mode == "formula_oracle" and selection_outcome.status == "final":
        # Formula-oracle evidence is an offline train-only fixture.  It may
        # validate the pipeline but cannot authorize a runtime final candidate
        # or any approved-memory writeback.
        selection_outcome = replace(
            selection_outcome,
            status="provisional",
            automatic_finalization_allowed=False,
            reasons=(
                *selection_outcome.reasons,
                "formula_oracle evidence cannot authorize runtime finalization",
            ),
        )
    provisional_candidate = selection_outcome.selected
    selected = (
        selection_outcome.selected
        if selection_outcome.status == "final"
        else None
    )
    fallback_candidate = select_fallback_candidate(evaluations, semantic=semantic)
    selection_status = (
        "provisional"
        if selection_outcome.status == "provisional"
        else "ambiguous_candidates"
        if ambiguity_retained
        else
        "selected_reference_only"
        if selected is not None and semantic.reference_only
        else "final"
        if selected is not None
        else trace_coverage.status
        if trace_coverage.degraded
        else "needs_review"
    )
    selection_review_notes: tuple[str, ...] = ()
    selection_rag_chunks: list[str] = []
    selection_records: list[dict] = list(guided_selection_records)
    selection_records.append(
        {
            "stage": "operator_resolution",
            "round": 1,
            "status": operator_resolution.status,
            "approved": operator_resolution.complete,
            "artifact": asdict(operator_resolution),
            "comments": "; ".join(operator_resolution.reasons),
        }
    )
    final_approval: FinalApprovalRecord | None = None
    if selected is not None and selection_outcome.status == "final":
        final_approval = _build_strict_automatic_approval(
            semantic=semantic,
            selected=selected,
            ir_coverage=ir_coverage,
            run_origin=run_origin,
        )
        final_memory_chunks = record_approved_review_examples(
            retriever,
            requirement_id=semantic.faithful_ir.requirement_id,
            requirement=requirement,
            semantic=semantic,
            traces=current_traces,
            round_index=1,
            comments="IR coverage and strict consistency passed.",
            ir_coverage=ir_coverage,
            candidate_status="final",
            run_origin=run_origin,
            final_approval=final_approval,
        )
        final_memory_chunks.extend(
            record_approved_pre_ir_examples(
                retriever,
                requirement_id=semantic.faithful_ir.requirement_id,
                requirement=requirement,
                semantic=semantic,
                full_pipeline_approved=bool(
                    final_approval and final_approval.writeback_allowed
                ),
            )
        )
        final_memory_chunks.extend(
            record_approved_template_examples(
                retriever,
                requirement=requirement,
                semantic=semantic,
                hole_candidates=hole_candidates,
                approved_decisions=completion_decisions,
                round_index=1,
                ir_coverage=ir_coverage,
                candidate_status="final",
                run_origin=run_origin,
                final_approval=final_approval,
            )
        )
        if final_memory_chunks:
            selection_records.append(
                {
                    "stage": "approved_learning_writeback",
                    "round": 1,
                    "status": "approved",
                    "approved": True,
                    "runtime_rag_chunks": final_memory_chunks,
                    "final_approval": asdict(final_approval),
                }
            )
    if selected is None:
        selection_review_notes = (
            (
                "补轨预算耗尽后仍存在多个证据一致的联合算子赋值；"
                "系统保留歧义集合，不按分数或公式长度宣称唯一语义。",
                *(
                    f"歧义候选: {item.formula}"
                    for item in final_passing_evaluations
                ),
            )
            if ambiguity_retained
            else (
                (
                    "唯一解析出的赋值已通过整轨迹一致性检查，但被独立覆盖门禁保留为 provisional。",
                )
                if selection_outcome.status == "provisional"
                and provisional_candidate is not None
                else candidate_selection_review_notes(
                    evaluations=evaluations,
                    strict_boundary=strict_boundary,
                    semantic=semantic,
                )
            )
        )
        if trace_coverage.degraded:
            selection_review_notes = (
                *selection_review_notes,
                trace_coverage.reason,
                *trace_coverage.coverage_debt,
            )
        selection_review_notes = (
            *selection_review_notes,
            *selection_outcome.reasons,
            *ir_coverage.reasons,
        )
        selection_rag_chunks = []
        selection_errors = _build_selection_stage_errors(
            evaluations=evaluations,
            strict_boundary=strict_boundary,
            semantic=semantic,
        )
        # Formula-bearing selection diagnostics remain in the audit record but
        # are not inserted into the retriever.  A later trace-debt round may
        # consume only the formula-free adequacy record written below.
        selection_rag_chunks.extend(
            record_trace_adequacy_issue(
                retriever=retriever,
                requirement=requirement,
                semantic=semantic,
                traces=current_traces,
                evaluations=evaluations,
                round_index=1,
                target_trace_count=max(8, len(current_traces) + 3),
                comments=(
                    "算子解析或最终门禁未通过；后续只能依据 NL/IR 义务"
                    "补充公式无关的边界或语义覆盖轨迹。"
                ),
            )
        )
        selection_record = {
            "stage": "selection",
            "round": 1,
            "status": "needs_review",
            "requirement": requirement,
            "semantic": asdict(semantic),
            "trace_coverage": asdict(trace_coverage),
            "ir_coverage": asdict(ir_coverage),
            "operator_resolution": asdict(operator_resolution),
            "run_origin": run_origin,
            "candidate_status": selection_outcome.status,
            "artifact": [str(evaluation.formula) for evaluation in evaluations],
            "approved": False,
            "runtime_rag_chunks": selection_rag_chunks,
            "stage_errors": [asdict(error) for error in selection_errors],
            "comments": "；".join(selection_review_notes),
        }
        selection_record["record_path"] = str(
            _save_selection_record(selection_record, records_dir)
        )
        selection_records.append(selection_record)

    learning_example = None
    if (
        selected is not None
        and semantic.persistent_rag_allowed
        and experiment_config.data_split == "train"
    ):
        if guided_context is not None:
            learning_example = build_warmup_fixture_learning_example(
                context=guided_context,
                semantic=semantic,
                trace_plan=trace_obligation_plan,
                traces=current_traces,
                skeleton_plan=constrained_skeleton_plan,
                skeletons=skeletons,
                hole_plan=completion_hole_plan,
                approved_completion=(
                    guided_context.expected_completion or str(selected.formula)
                ),
                final_formula=str(selected.formula),
                ir_coverage=ir_coverage,
                candidate_status=selection_outcome.status,
                run_origin=run_origin,
                final_approval=final_approval,
            )
        else:
            learning_example = build_faithful_learning_example(
                requirement_id=semantic.faithful_ir.requirement_id,
                requirement=requirement,
                semantic=semantic,
                trace_plan=trace_obligation_plan,
                traces=current_traces,
                skeleton_plan=constrained_skeleton_plan,
                skeletons=skeletons,
                hole_plan=completion_hole_plan,
                approved_completion=str(selected.formula),
                final_formula=str(selected.formula),
                ir_coverage=ir_coverage,
                candidate_status=selection_outcome.status,
                run_origin=run_origin,
                approval_source="strict_consistency_and_ir_coverage",
                final_approval=final_approval,
            )

    audited_candidate = selected or provisional_candidate or fallback_candidate
    formula_closure = None
    if audited_candidate is not None and semantic.faithful_ir is not None:
        formula_closure = validate_formula_closure(
            audited_candidate.formula,
            build_strict_formula_closure_policy(
                semantic.faithful_ir,
                semantic.approved_semantic_ir,
            ),
            ir_version=(
                audited_candidate.ir_version
                or semantic.faithful_ir.ir_version
            ),
        )
    ambiguity_closure = compute_ambiguity_closure_report(
        semantic,
        trace_coverage=trace_coverage,
        ir_coverage=ir_coverage,
        operator_resolution=operator_resolution,
    )
    if semantic.approved_semantic_ir is not None:
        semantic = replace(
            semantic,
            approved_semantic_ir=replace(
                semantic.approved_semantic_ir,
                ambiguity_closure_id=ambiguity_closure.closure_id,
            ),
        )
    semantic_status = (
        "approved"
        if semantic.approved_semantic_ir is not None
        and semantic.approved_semantic_ir.approved
        and ambiguity_closure.passed
        else "needs_review"
    )
    evidence_status = (
        semantic.semantic_evidence_gate.status
        if semantic.semantic_evidence_gate is not None
        else "incomplete"
    )
    if final_approval is not None:
        final_approval = replace(
            final_approval,
            source_run_id=source_run_id,
            formula_closure_status=(
                formula_closure.status if formula_closure is not None else "missing"
            ),
            context_isolated=True,
        )
    approval_status = (
        "final"
        if final_approval is not None and selected is not None
        else "needs_independent_approval"
        if run_origin == "feedback_rerun"
        and provisional_candidate is not None
        and formula_closure is not None
        and formula_closure.passed
        and ambiguity_closure.passed
        and evidence_status == "complete"
        else "pending"
    )
    if approval_status == "final":
        import hashlib
        emit(
            "operator_report_bound_to_final_formula",
            requirement_id=semantic.faithful_ir.requirement_id,
            ir_version=semantic.faithful_ir.ir_version,
            ir_artifact_hash=semantic.faithful_ir.artifact_hash,
            report_ir_version=operator_resolution.ir_version,
            report_hole_plan_version=operator_resolution.hole_plan_version,
            passing_assignment_count=operator_resolution.passing_assignment_count,
            formula_sha256=hashlib.sha256(str(selected.formula).encode("utf-8")).hexdigest(),
            entrypoint="run_traj_ltlf_pipeline",
        )
    learning_status = (
        "writeback_allowed"
        if final_approval is not None and final_approval.writeback_allowed
        else "writeback_blocked_feedback_origin"
        if run_origin == "feedback_rerun"
        else "writeback_blocked_nonfinal"
    )
    diagnoses: list[ErrorDiagnosisPacket] = []
    for evaluation in evaluations:
        reject_reasons = candidate_reject_reasons(
            evaluation,
            strict_boundary,
            semantic,
            temporal_resolution_by_node=final_temporal_resolution,
        )
        if not reject_reasons:
            continue
        diagnoses.append(
            build_error_diagnosis(
                requirement_id=semantic.faithful_ir.requirement_id,
                source_run_id=source_run_id,
                source_stage="selection",
                candidate_information_exposed=True,
                diagnoser_role="deterministic_consistency_diagnoser",
                diagnoser_independence_id="ir-trace-diagnoser-v1",
                evaluation=evaluation,
                traces=current_traces,
                closure_report=evaluation.closure_report,
                confidence="high",
            )
        )
    if not ambiguity_closure.passed:
        diagnoses.append(
            build_error_diagnosis(
                requirement_id=semantic.faithful_ir.requirement_id,
                source_run_id=source_run_id,
                source_stage="semantic_approval",
                candidate_information_exposed=False,
                diagnoser_role="ambiguity_closure_validator",
                diagnoser_independence_id="ambiguity-closure-v1",
                error_class="ir_ambiguity_state_mismatch",
                confidence="high",
            )
        )
    elif evidence_status != "complete":
        gate_reasons = (
            semantic.semantic_evidence_gate.reasons
            if semantic.semantic_evidence_gate is not None
            else ()
        )
        if "nl_ir_semantic_contract_conflict" in gate_reasons:
            evidence_error_class = "ir_error"
            evidence_source_stage = "semantic_ir"
        elif any(reason.startswith("trace_label_conflict:") for reason in gate_reasons):
            evidence_error_class = "trace_label_conflict"
            evidence_source_stage = "trace_adjudication"
        else:
            evidence_error_class = "trace_coverage_gap"
            evidence_source_stage = "trace_obligation_planning"
        diagnoses.append(
            build_error_diagnosis(
                requirement_id=semantic.faithful_ir.requirement_id,
                source_run_id=source_run_id,
                source_stage=evidence_source_stage,
                candidate_information_exposed=False,
                diagnoser_role="coverage_validator",
                diagnoser_independence_id="coverage-validator-v1",
                error_class=evidence_error_class,
                affected_obligation_ids=(
                    semantic.semantic_evidence_gate.missing_obligation_ids
                    if semantic.semantic_evidence_gate is not None
                    else ()
                ),
                confidence="high",
            )
        )
    no_delta_records = (
        ("completion", all_completion_records, True),
        ("trace_obligation_planning", trace_supplement_records, False),
    )
    for source_stage, records, candidate_exposed in no_delta_records:
        for record in records:
            if str(record.get("status", "")) != "no_effective_delta":
                continue
            round_id = str(record.get("round", ""))
            diagnoses.append(
                build_error_diagnosis(
                    requirement_id=semantic.faithful_ir.requirement_id,
                    source_run_id=source_run_id,
                    source_stage=source_stage,
                    candidate_information_exposed=candidate_exposed,
                    diagnoser_role="generation_delta_auditor",
                    diagnoser_independence_id="generation-delta-auditor-v1",
                    error_class="no_effective_delta",
                    confusion_targets=("identical_generation",),
                    frozen_stages=(source_stage,),
                    confidence="high",
                    diagnosis_id=(
                        f"diagnosis-{source_run_id}-{source_stage}-{round_id}-no-delta"
                    ),
                )
            )

    result = PipelineResult(
        requirement=requirement,
        semantic=semantic,
        ap=ap,
        traces=current_traces,
        skeletons=skeletons,
        skeleton_diagnostics=skeleton_diagnostics,
        skeleton_rounds=all_skeleton_records,
        min_passed_skeletons=min_passed_skeletons,
        hole_candidates=hole_candidates,
        completion_decisions=completion_decisions,
        completion_rounds=all_completion_records,
        candidates=candidates,
        evaluations=evaluations,
        selected=selected,
        fallback_candidate=fallback_candidate,
        selection_status=selection_status,
        selection_review_notes=selection_review_notes,
        selection_rag_chunks=selection_rag_chunks,
        strict_boundary=strict_boundary,
        review_mode=review_mode,
        review_records=(
            base_review_records
            + all_skeleton_records
            + all_completion_records
            + consistency_records
            + trace_supplement_records
            + selection_records
        ),
        trace_coverage=trace_coverage,
        experiment_mode=experiment_config.mode,
        oracle_assisted=False,
        diagnostic_only=False,
        ambiguous_candidates=tuple(
            str(item.formula)
            for item in final_passing_evaluations
            if ambiguity_retained
        ),
        trace_obligation_plan=trace_obligation_plan,
        constrained_skeleton_plan=constrained_skeleton_plan,
        completion_hole_plan=completion_hole_plan,
        ir_coverage=ir_coverage,
        operator_resolution=operator_resolution,
        run_origin=run_origin,
        candidate_status=selection_outcome.status,
        automatic_finalization_allowed=selection_outcome.automatic_finalization_allowed,
        provisional_candidate=(
            provisional_candidate
            if selection_outcome.status == "provisional"
            else None
        ),
        learning_example=learning_example,
        final_approval=final_approval,
        semantic_status=semantic_status,
        evidence_status=evidence_status,
        approval_status=approval_status,
        learning_status=learning_status,
        source_run_id=source_run_id,
        formula_closure=formula_closure,
        ambiguity_closure=ambiguity_closure,
        error_diagnoses=tuple(diagnoses),
        trace_evidence_policy=trace_evidence_policy,
        canonical_trace_coverage=canonical_trace_coverage,
        experimental_only=trace_evidence_policy.experimental_only,
        boundary_metric_available=any(
            is_boundary_trace(trace) for trace in current_traces
        ),
        ap_context_fingerprint=semantic.faithful_ir.ap_context_fingerprint,
        relation_feasibility=trace_obligation_plan.relation_feasibility,
    )
    from candidate_delivery import deliver_formula_candidate
    result = deliver_formula_candidate(result)
    return replace(result, artifact_fingerprints=build_artifact_fingerprint_set(result))


def _latest_record_path(records, *, stage: str = "") -> str:
    for record in reversed(list(records)):
        if stage and str(record.get("stage", "")) != stage:
            continue
        path = str(record.get("record_path") or record.get("annotation_review_path") or "")
        if path:
            return path
    return ""


def _semantic_domain_not_approved_error(
    semantic: SemanticStructure,
    *,
    code: str,
) -> ValueError:
    gate = semantic.semantic_evidence_gate
    if gate is not None:
        reasons = gate.reasons
    elif semantic.approved_semantic_ir is not None:
        reasons = semantic.approved_semantic_ir.remaining_ambiguity_debt
    else:
        reasons = ("SemanticEvidenceGateReport is missing",)
    detail = "; ".join(reasons) or "semantic evidence gate is incomplete"
    error = ValueError(f"{code}: {detail}")
    error.pipeline_stage = "semantic_approval"
    error.diagnosis_error_class = "semantic_domain_not_approved"
    return error


def _semantic_execution_authorized(
    semantic: SemanticStructure,
    policy: TraceEvidencePolicy,
) -> bool:
    approved = semantic.approved_semantic_ir
    if approved is not None and approved.approved:
        return True
    return bool(
        policy.experimental_only
        and semantic.semantic_evidence_gate is not None
        and semantic.semantic_evidence_gate.complete
    )


def format_working_result(result: PipelineResult) -> str:
    return "\n".join([
        "自然语言需求:", result.requirement, "", "最终候选:",
        str(result.selected.formula) if result.selected else "missing",
        "", f"输出契约: {result.output_contract_version}",
        f"选择方式: {result.selection_method}",
        f"正常闭环完成: {result.normal_flow_completed}",
        f"严格证据唯一通过: {result.strict_evidence_pass}",
        f"枚举完整: {result.enumeration_complete}",
        f"UNCLEAR 轨迹数: {result.unclear_trace_count}",
        f"证据版本: {result.evidence_version}",
        "语义批准: 未批准；输出不代表语义正确",
        *("假设: " + item for item in result.semantic_assumptions),
        *("降级: " + item for item in result.degradation_reasons),
    ])


def format_pipeline_result(result: PipelineResult) -> str:
    if result.delivery_policy == "always_output":
        return format_working_result(result)
    """把完整流程结果格式化成一份可阅读报告。"""

    lines: list[str] = []

    lines.append("Traj-LTLf 原型程序运行结果")
    lines.append("=" * 40)
    lines.append("")
    lines.append("1. 自然语言需求")
    lines.append(f"   {result.requirement}")
    lines.append("")

    segmentation = result.semantic.surface_segmentation
    reading = result.semantic.plain_reading_review
    lines.append("1A. IR 前置普通语言审核")
    if segmentation is not None:
        lines.append(f"   surface segmentation status = {segmentation.status}")
        lines.append(
            "   mechanical validation status = "
            + segmentation.mechanical_validation.status
        )
        lines.append(f"   segmentation id = {segmentation.segmentation_id}")
        lines.append(f"   segmentation revision = {segmentation.revision}")
        lines.append(f"   segmentation review basis = {segmentation.review_basis}")
    else:
        lines.append("   surface segmentation status = missing")
    if reading is not None:
        lines.append(f"   plain reading status = {reading.overall_status}")
        lines.append(f"   reading review id = {reading.reading_review_id}")
        lines.append(f"   reading downstream status = {reading.downstream_status}")
        if reading.supersedes_reading_review_id:
            lines.append(
                "   supersedes reading review id = "
                + reading.supersedes_reading_review_id
            )
        lines.append(f"   reading review basis = {reading.review_basis}")
        lines.append(f"   reviewer source = {reading.reviewer_source}")
        confirmation = reading.no_ambiguity_confirmation
        lines.append(f"   no-ambiguity confirmation = {confirmation.status}")
        lines.append(
            "   no-ambiguity human confirmed = "
            + str(confirmation.human_confirmed).lower()
        )
        trigger = reading.ambiguity_generation_trigger
        if trigger.source:
            lines.append(f"   ambiguity generation trigger = {trigger.source}")
            lines.append(
                "   suspected ambiguity segments = "
                + (", ".join(trigger.suspected_segment_ids) or "none")
            )
        selected = [f"{axis}={option}" for axis, option in reading.selected_options]
        unresolved = list(reading.unresolved_ambiguity_ids)
        lines.append("   selected ambiguity axes = " + (", ".join(selected) or "none"))
        lines.append("   unresolved ambiguity axes = " + (", ".join(unresolved) or "none"))
        lines.append(
            "   pending revision requests = "
            + (", ".join(reading.pending_revision_request_ids) or "none")
        )
        for request in reading.revision_requests:
            lines.append(f"   revision request {request.request_id}:")
            lines.append(f"      ambiguity axis = {request.ambiguity_id}")
            lines.append(f"      human supplement = {request.proposed_interpretation}")
            lines.append(f"      reason = {request.reason}")
            lines.append(f"      source segments = {', '.join(request.source_segment_ids)}")
            lines.append(f"      validation status = {request.validation_status}")
            lines.append(f"      classification = {request.classification}")
            lines.append(f"      request status = {request.request_status}")
            lines.append(
                "      source warnings = "
                + ("; ".join(request.source_warnings) or "none")
            )
            lines.append(f"      confirmed by = {request.confirmed_by or 'none'}")
            lines.append(
                f"      human confirmed = {str(request.human_confirmed).lower()}"
            )
            if request.classification_basis:
                lines.append(
                    f"      classification basis = {request.classification_basis}"
                )
        revision_history = [
            entry
            for entry in reading.review_history
            if entry.get("operation") == "plain_reading_revision_applied"
        ]
        if revision_history:
            latest = revision_history[-1]
            lines.append(
                "   latest revision old options = "
                + str(latest.get("old_options", []))
            )
            lines.append(
                "   latest revision new axes = "
                + str(latest.get("new_axes", []))
            )
        if reading.downstream_status == "stale_due_to_revision":
            lines.append(
                "   注意：普通读法已修订，旧 IR、候选、轨迹证据和批准记录均已失效。"
            )
        if unresolved:
            lines.append(
                "   注意：当前候选基于普通语言工作读法，而非已证明的唯一语义。"
            )
    else:
        lines.append("   plain reading status = missing")
    lines.append("")

    ir_adjudication = result.semantic.ir_ambiguity_adjudication
    if ir_adjudication is not None:
        lines.append("1B. IR 候选自动轨迹判别")
        lines.append(f"   status = {ir_adjudication.status}")
        lines.append(f"   rounds = {len(ir_adjudication.rounds)}")
        lines.append(
            "   selected IR branch = "
            + (ir_adjudication.selected_branch_id or "none")
        )
        lines.append(
            "   provisional = " + str(ir_adjudication.provisional).lower()
        )
        lines.append(
            "   unresolved IR ambiguities = "
            + (
                ", ".join(ir_adjudication.unresolved_ambiguity_ids)
                or "none"
            )
        )
        lines.append("")

    lines.append("2. RAG + NLU Agent 语义结构")
    displayed_aps = result.semantic.aps or tuple(result.ap)
    lines.append(f"   AP = {{{', '.join(displayed_aps)}}}")
    lines.append(f"   trigger = {result.semantic.trigger}")
    lines.append(f"   response = {result.semantic.response}")
    lines.append(f"   pattern = {result.semantic.pattern}")
    if result.semantic.faithful_ir is not None:
        ir = result.semantic.faithful_ir
        lines.append(f"   FaithfulSemanticIR schema = {ir.schema_version}")
        lines.append(f"   ir_version = {ir.ir_version}")
        lines.append(f"   AP profile = {ir.ap_profile_mode or 'legacy_unbound'}")
        lines.append(f"   catalog authority = {ir.catalog_authority}")
        lines.append(
            f"   AP context fingerprint = {ir.ap_context_fingerprint or 'missing'}"
        )
        if result.relation_feasibility is not None:
            lines.append(
                "   relation feasibility = "
                + result.relation_feasibility.classification
            )
            lines.append(
                "   feasibility report hash = "
                + result.relation_feasibility.artifact_hash
            )
        lines.append(f"   root node = {ir.root_node_id}")
        lines.append(f"   semantic status = {result.semantic.semantic_status}")
        lines.append(
            "   ApprovedSemanticIR = "
            + (
                "approved"
                if result.semantic.approved_semantic_ir
                and result.semantic.approved_semantic_ir.approved
                else "pending"
            )
        )
        if (
            result.semantic.approved_semantic_ir
            and result.semantic.approved_semantic_ir.remaining_ambiguity_debt
        ):
            lines.append(
                "   ambiguity debt = "
                + ", ".join(
                    result.semantic.approved_semantic_ir.remaining_ambiguity_debt
                )
            )
    if result.semantic.pattern_status == "unknown":
        lines.append("   模式状态: 未知模式")
        lines.append(
            f"   LLM 识别模式: "
            f"{result.semantic.original_pattern or result.semantic.pattern}"
        )
        lines.append("   处理方式: 需要人工学习样例支持该模式的骨架与孔位模板")
        lines.append("   持久性 RAG: 未写入")
    else:
        lines.append("   模式状态: 已知模式")
    lines.append(f"   scope keyword = {result.semantic.scope_keyword}")
    lines.append(f"   temporal keyword = {result.semantic.temporal_keyword}")
    lines.append(f"   trace domain = {result.semantic.trace_domain}")
    lines.append(f"   experiment mode = {result.experiment_mode}")
    if result.oracle_assisted or result.diagnostic_only:
        lines.append(f"   oracle_assisted = {str(result.oracle_assisted).lower()}")
        lines.append(f"   diagnostic_only = {str(result.diagnostic_only).lower()}")
    lines.append(f"   data split = {result.semantic.data_split}")
    if result.semantic.formula_text:
        lines.append(f"   legacy formula_text = {result.semantic.formula_text}")
        if result.semantic.pattern == "response":
            lines.append("   legacy formula_text 在 response 补全阶段不作为旁路使用")
    if result.semantic.reference_only:
        lines.append("   LTL/LTLf 标注:")
        lines.append(f"   - {result.semantic.domain_warning}")
        if result.semantic.domain_evidence:
            lines.append(
                "   - 判别证据: "
                + ", ".join(result.semantic.domain_evidence)
            )
        if result.semantic.finite_rewrite:
            lines.append(f"   - LTLf 有限化 NL: {result.semantic.finite_rewrite}")
    lines.append(f"   confidence = {result.semantic.confidence:.2f}")
    lines.append(f"   explanation = {result.semantic.explanation}")
    if result.semantic.rag_context:
        lines.append("   RAG 检索片段:")
        for context in result.semantic.rag_context:
            lines.append(f"   - {context}")
    lines.append("")

    lines.append("3. RAG + Trace Agent 有限轨迹集合")
    lines.append(
        "   boundary trace policy = "
        + result.trace_evidence_policy.boundary_source
    )
    lines.append(
        f"   experimental only = {str(result.experimental_only).lower()}"
    )
    if result.trace_evidence_policy.boundary_fixture_sha256:
        lines.append(
            "   boundary fixture sha256 = "
            + result.trace_evidence_policy.boundary_fixture_sha256
        )
    lines.append(f"   coverage status = {result.trace_coverage.status}")
    lines.append(
        "   canonical coverage status = "
        + result.canonical_trace_coverage.status
    )
    lines.append(
        "   quota complete = "
        f"{str(result.trace_coverage.quota_complete).lower()}"
    )
    lines.append(
        "   semantic complete = "
        f"{str(result.trace_coverage.semantic_complete).lower()}"
    )
    lines.append(
        f"   coverage count = {result.trace_coverage.actual_count}/"
        f"{result.trace_coverage.required_count}"
    )
    if result.trace_coverage.coverage_debt:
        lines.append("   coverage debt:")
        lines.extend(
            f"   - {item}" for item in result.trace_coverage.coverage_debt
        )
    for labeled_trace in result.traces:
        lines.append(
            f"   [{_kind_name(labeled_trace.kind)}] {labeled_trace.name}: "
            f"{format_trace(labeled_trace.trace)}"
        )
        lines.append("      零基础视图:")
        for beginner_line in labeled_trace.beginner_view.splitlines():
            if beginner_line:
                lines.append(f"      {beginner_line}")
            else:
                lines.append("      ")
        lines.append(f"      人工标签: {format_truth(labeled_trace.expected)}")
        if labeled_trace.semantics.semantic_label:
            lines.append(
                f"      语义标签: {labeled_trace.semantics.semantic_label}"
            )
        if labeled_trace.semantics.roles:
            lines.append(
                "      证据角色: "
                + ", ".join(labeled_trace.semantics.roles)
            )
        if labeled_trace.semantics.confusion_targets:
            lines.append(
                "      区分目标: "
                + ", ".join(labeled_trace.semantics.confusion_targets)
            )
        lines.append(f"      说明: {labeled_trace.reason}")
    lines.append("")

    lines.append("4. RAG + Skeleton Agent 受限骨架生成")
    for record in result.skeleton_rounds:
        lines.append(
            f"   round {record['round']}: "
            f"可用 {record['passed_count']}/{len(record['artifact'])} 个骨架"
        )
        if record["runtime_rag_chunks"]:
            lines.append(
                "      写入当前 RAG: "
                + ", ".join(record["runtime_rag_chunks"])
            )
        for diagnostic in record["diagnostics"]:
            status = "通过" if diagnostic["passed"] else diagnostic["error_type"]
            lines.append(f"      [{status}] {diagnostic['skeleton']}")
            if not diagnostic["passed"]:
                for evidence in diagnostic["trace_evidence"][:2]:
                    lines.append(f"         证据: {evidence}")
    lines.append("")

    lines.append("   进入补全阶段的受限骨架:")
    for skeleton in result.skeletons:
        lines.append(f"   - {skeleton}")
    lines.append("")

    lines.append("5. RAG + Completion Agent 带孔候选与补全")
    lines.append(f"   带孔候选模板: {len(result.hole_candidates)} 个")
    for hole_candidate in result.hole_candidates:
        candidate_label = hole_candidate.candidate_id or "HC-???"
        lines.append(f"   - [{candidate_label}] {hole_candidate.template_text}")
        bound_versions = tuple(
            dict.fromkeys(
                hole.ir_version
                for hole in hole_candidate.holes
                if hole.ir_version
            )
        )
        if bound_versions:
            lines.append(
                "      source=hole_plan, ir_version=" + ", ".join(bound_versions)
            )
    if result.hole_candidates and result.hole_candidates[0].rag_context:
        lines.append("   RAG 检索片段:")
        for context in result.hole_candidates[0].rag_context:
            lines.append(f"   - {context}")
    lines.append(
        "   数量核对: "
        f"带孔模板 {len(result.hole_candidates)} 个，"
        f"补全决策 {len(result.completion_decisions)} 个"
    )
    lines.append("   补全决策（全部模板）:")
    for decision in result.completion_decisions:
        assignment_text = ", ".join(
            f"{hole}={operator}" for hole, operator in decision.assignments
        ) or "no holes"
        candidate_label = decision.candidate_id or "HC-???"
        lines.append(
            f"   - [{candidate_label}] {decision.formula} [{assignment_text}] "
            f"Bound={ratio_text(decision.boundary_match) if result.boundary_metric_available else 'N/A'}, "
            f"Pos={ratio_text(decision.pos_sat)}, "
            f"Neg={ratio_text(decision.neg_rej)}, "
            f"Score={decision.score:.3f}"
        )
        lines.append(
            "      source=completion, ir_version="
            + (decision.ir_version or "<unbound>")
            + ", closure="
            + (
                decision.closure_report.status
                if decision.closure_report is not None
                else "missing"
            )
        )
        for derivation in decision.operator_derivation[:1]:
            lines.append(f"      孔位推导: {derivation}")
    lines.append("")

    lines.append("6. Trajectory Consistency Agent 轨迹一致性模拟")
    lines.append(
        "   数量核对: "
        f"补全决策 {len(result.completion_decisions)} 个，"
        f"轨迹一致性评分 {len(result.evaluations)} 个"
    )
    lines.append(
        "   "
        + f"{'候选公式':<32}"
        + f"{'PosSat':>10}"
        + f"{'NegRej':>10}"
        + f"{'Bound':>10}"
        + f"{'Score':>10}"
        + "   结论"
    )
    for evaluation in rank_evaluations(result.evaluations):
        reject_reasons = candidate_reject_reasons(
            evaluation=evaluation,
            strict_boundary=result.strict_boundary,
            semantic=result.semantic,
        )
        conclusion = "严格匹配" if not reject_reasons else f"后验复核: {reject_reasons[0]}"
        lines.append(
            "   "
            + f"{str(evaluation.formula):<32}"
            + f"{ratio_text(evaluation.pos_sat):>10}"
            + f"{ratio_text(evaluation.neg_rej):>10}"
            + f"{(ratio_text(evaluation.bound_match) if result.boundary_metric_available else 'N/A'):>10}"
            + f"{evaluation.score:>10.3f}"
            + f"   {conclusion}"
        )
        lines.append(
            "      candidate_id="
            + (evaluation.candidate_id or "<unassigned>")
            + ", ir_version="
            + (evaluation.ir_version or "<unbound>")
            + ", closure="
            + (
                evaluation.closure_report.status
                if evaluation.closure_report is not None
                else "missing"
            )
        )
    lines.append("")

    lines.append("7. 最终选择")
    lines.append(f"   status = {result.selection_status}")
    lines.append(f"   semantic status = {result.semantic_status}")
    lines.append(f"   candidate status = {result.candidate_status}")
    lines.append(f"   evidence status = {result.evidence_status}")
    lines.append(f"   approval status = {result.approval_status}")
    lines.append(f"   learning status = {result.learning_status}")
    lines.append(f"   run origin = {result.run_origin}")
    lines.append(f"   IR coverage = {result.ir_coverage.status}")
    lines.append(
        "   operator resolution = "
        f"{result.operator_resolution.status} "
        f"(passing={result.operator_resolution.passing_assignment_count}, "
        f"materialized={result.operator_resolution.candidate_assignment_count}, "
        f"evaluated={result.operator_resolution.evaluated_assignment_count}, "
        f"expected={result.operator_resolution.expected_assignment_count})"
    )
    lines.append(
        "   operator evidence purpose = "
        f"{result.operator_resolution.evidence_purpose}"
    )
    lines.append(
        "   operator resolution authoritative = "
        + str(result.operator_resolution.authoritative).lower()
    )
    if result.formula_closure is not None:
        closure = result.formula_closure
        lines.append(f"   formula closure = {closure.status}")
        lines.append(f"   formula hash = {closure.formula_hash}")
        lines.append(f"   observed AP = {', '.join(closure.observed_aps) or '<none>'}")
        lines.append(f"   extra AP = {', '.join(closure.extra_aps) or '<none>'}")
        lines.append(
            f"   missing required AP = {', '.join(closure.missing_required_aps) or '<none>'}"
        )
        lines.append(
            f"   Boolean constants = {', '.join(closure.boolean_constants) or '<none>'}"
        )
        lines.append(
            f"   placeholders = {', '.join(closure.placeholders) or '<none>'}"
        )
        lines.append(
            "   unsupported operators = "
            + (", ".join(closure.unsupported_operators) or "<none>")
        )
        lines.append(
            "   violating AST paths = "
            + (", ".join(closure.violating_node_paths) or "<none>")
        )
        if closure.reasons:
            lines.append("   closure reasons = " + "; ".join(closure.reasons))
    if result.ambiguity_closure is not None:
        lines.append(f"   ambiguity closure = {result.ambiguity_closure.status}")
    if result.error_diagnoses:
        lines.append(
            "   diagnosis ids = "
            + ", ".join(item.diagnosis_id for item in result.error_diagnoses)
        )
    if result.rerun_delta is not None:
        lines.append(f"   rerun delta = {result.rerun_delta.status}")
        lines.append(f"   rerun id = {result.rerun_delta.rerun_id}")
        lines.append(f"   rerun start stage = {result.rerun_delta.start_stage}")
        lines.append(
            "   expected changed artifacts = "
            + (", ".join(result.rerun_delta.expected_changed_artifacts) or "<none>")
        )
        lines.append(
            "   actual changed artifacts = "
            + (", ".join(result.rerun_delta.actual_changed_artifacts) or "<none>")
        )
        lines.append(
            "   first-to-rerun candidate delta = new["
            + (", ".join(result.rerun_delta.new_candidates) or "<none>")
            + "] removed["
            + (", ".join(result.rerun_delta.removed_candidates) or "<none>")
            + "]"
        )
        lines.append(
            "   new traces = "
            + (", ".join(result.rerun_delta.new_traces) or "<none>")
        )
        lines.append(
            "   resolved errors = "
            + (", ".join(result.rerun_delta.resolved_errors) or "<none>")
        )
    if result.feedback_patch is not None:
        lines.append(f"   feedback patch = {result.feedback_patch.patch_id}")
        lines.append(
            f"   feedback patch review = {result.feedback_patch.review_status}"
        )
    if result.final_approval is not None:
        lines.append(f"   approval id = {result.final_approval.approval_id}")
        lines.append(f"   approval reviewer = {result.final_approval.reviewer}")
        lines.append(
            "   approval independence id = "
            + (result.final_approval.independence_id or "<initial normal approval>")
        )
        lines.append(
            f"   approval context isolated = {result.final_approval.context_isolated}"
        )
        lines.append(f"   approval gold exposed = {result.final_approval.gold_exposed}")
    if result.selected is not None:
        lines.append(f"   ψ* = {result.selected.formula}")
        if result.semantic.reference_only:
            lines.append("   注意：该公式是无限轨迹 LTL 需求的 LTLf 有限化参考结果。")
        lines.append("")
        lines.append("   选择理由:")
        if result.boundary_metric_available:
            lines.append(
                "   该公式通过语义硬过滤，满足所有正轨迹，拒绝所有反轨迹，并且匹配边界轨迹；"
                "在满足语义约束的候选中，它的综合得分最高。"
            )
        else:
            lines.append(
                "   该公式通过语义硬过滤，满足所有正轨迹并拒绝所有反轨迹；"
                "本实验未使用边界轨迹，结果仅按实验策略解释。"
            )
    else:
        lines.append("   ψ* = 暂不输出")
        if result.provisional_candidate is not None:
            lines.append(
                f"   provisional candidate = {result.provisional_candidate.formula}"
            )
        if result.ambiguous_candidates:
            lines.append("   歧义候选集:")
            for candidate in result.ambiguous_candidates:
                lines.append(f"   - {candidate}")
        lines.append("")
        lines.append("   评审意见:")
        for note in result.selection_review_notes:
            lines.append(f"   - {note}")
        if result.fallback_candidate is not None:
            lines.append(
                f"   参考候选: {result.fallback_candidate.formula} "
                f"(Score={result.fallback_candidate.score:.3f})"
            )
        if result.selection_rag_chunks:
            lines.append(
                "   已写入当前 RAG: "
                + ", ".join(result.selection_rag_chunks)
            )
    lines.append("")

    detail_evaluation = result.selected or result.fallback_candidate
    lines.append("8. 有限轨迹模拟明细")
    if detail_evaluation is None:
        lines.append("   暂无可展示的候选模拟明细。")
    else:
        if result.selected is None:
            lines.append("   以下为参考候选的模拟明细，不代表最终公式。")
        for trace_result in detail_evaluation.trace_results:
            lines.append(
                f"   {trace_result.trace_name:<32}"
                f" 模拟结果={format_truth(trace_result.actual):<4}"
                f" 人工标签={format_truth(trace_result.expected):<4}"
                f" 匹配={'是' if trace_result.matched else '否'}"
            )
            for step in trace_result.steps[:3]:
                lines.append(
                    f"      t{step.time_index} {step.formula_fragment}: "
                    f"{format_truth(step.result)}，{step.explanation}"
                )
    lines.append("")

    lines.append("9. 公式解释")
    if result.selected is not None:
        if result.semantic.pattern != "response" and result.semantic.formula_text:
            lines.append(
                f"   {result.selected.formula} 来自已学习的 "
                f"{result.semantic.pattern} 模式；其业务语义由 warmup 样例中的 "
                "approved_formula、components 和 derivation 固定。"
            )
            if result.semantic.components:
                lines.append("   组件:")
                for key, value in result.semantic.components:
                    lines.append(f"   - {key}: {value}")
        else:
            lines.append(
                f"   {result.selected.formula} 表示：在有限轨迹的每一个位置，"
                f"如果 {result.semantic.trigger} 成立，那么从当前位置到轨迹结束之前，"
                f"必须至少有一个位置出现 {result.semantic.response}。"
            )
        if result.semantic.reference_only:
            lines.append(
                "   原需求被标注为无限轨迹 LTL；上述解释只对应有限化后的 LTLf 语义。"
            )
    else:
        if result.trace_coverage.degraded and result.fallback_candidate is not None:
            lines.append(
                f"   现有证据支持 provisional 候选 "
                f"{result.fallback_candidate.formula}，但轨迹覆盖不完整，"
                "必须严格评审后才能确认。"
            )
        else:
            lines.append(
                "   当前候选集合没有公式通过严格过滤，需根据 needs_review 评审意见"
                "重新补全或人工检查后再确认最终 LTLf 规约。"
            )
    lines.append("")

    lines.append("10. Agent 评审记录")
    lines.append(f"   review mode = {result.review_mode}")
    for record in result.review_records:
        normalized_record = normalize_review_record(record)
        status = _review_record_status_label(normalized_record)
        lines.append(
            f"   {normalized_record['stage']} round {normalized_record['round']}: "
            f"{status} -> {normalized_record['record_path']}"
        )
        if normalized_record.get("runtime_rag_chunks"):
            lines.append(
                "      写入当前 RAG: "
                + ", ".join(normalized_record["runtime_rag_chunks"])
            )

    return "\n".join(lines)


def format_pipeline_summary(result: PipelineResult) -> str:
    if result.delivery_policy == "always_output":
        return format_working_result(result)
    if result.provisional_candidate is not None:
        return "\n".join(["需求:", f"  {result.requirement}", "最终候选:",
                          f"  {result.provisional_candidate.formula}", "选择方式:",
                          "  候选回退选择", "审核记录:", "  保留原始审核状态与回退依据。"])
    """只格式化自然语言需求和最终候选，供终端与精简结果文件使用。"""

    candidate = (
        str(result.selected.formula)
        if result.selected is not None
        else f"provisional: {result.provisional_candidate.formula}"
        if result.provisional_candidate is not None
        else f"provisional: {result.fallback_candidate.formula}"
        if result.trace_coverage.degraded and result.fallback_candidate is not None
        else "暂不输出（needs_review）"
    )
    lines = [
        "需求:",
        f"  {result.requirement}",
        "实验模式:",
        f"  {result.experiment_mode}",
        "轨迹覆盖状态:",
        f"  {result.trace_coverage.status}",
        "IR 覆盖状态:",
        f"  {result.ir_coverage.status}",
        "算子补全状态:",
        f"  {result.operator_resolution.status}",
        "候选状态:",
        f"  {result.candidate_status}",
        "语义状态:",
        f"  {result.semantic_status}",
        "证据状态:",
        f"  {result.evidence_status}",
        "审批状态:",
        f"  {result.approval_status}",
        "学习写回状态:",
        f"  {result.learning_status}",
    ]
    if result.semantic.surface_segmentation is not None:
        lines.extend(
            [
                "表面切分:",
                f"  {result.semantic.surface_segmentation.status} / "
                f"{result.semantic.surface_segmentation.mechanical_validation.status}",
            ]
        )
    if result.semantic.plain_reading_review is not None:
        reading = result.semantic.plain_reading_review
        lines.extend(
            [
                "普通读法审核:",
                f"  {reading.overall_status}",
                "普通读法修订:",
                f"  requests={len(reading.revision_requests)}, "
                f"pending={len(reading.pending_revision_request_ids)}, "
                f"downstream={reading.downstream_status}",
            ]
        )
    if result.final_approval is not None:
        lines.extend(
            [
                "最终审批:",
                f"  approval_id: {result.final_approval.approval_id}",
                f"  reviewer: {result.final_approval.reviewer}",
            ]
        )
    if result.oracle_assisted or result.diagnostic_only:
        lines.extend(
            [
                "诊断标记:",
                f"  oracle_assisted: {str(result.oracle_assisted).lower()}",
                f"  diagnostic_only: {str(result.diagnostic_only).lower()}",
            ]
        )
    if result.trace_coverage.coverage_debt:
        lines.extend(
            [
                "轨迹证据缺口:",
                *[f"  - {item}" for item in result.trace_coverage.coverage_debt],
            ]
        )
    if result.semantic.reference_only:
        lines.extend(
            [
                "LTL/LTLf 标注:",
                "  原需求疑似无限轨迹 LTL；以下 LTLf 结果仅供参考。",
            ]
        )
        if result.semantic.finite_rewrite:
            lines.extend(["有限化 NL:", f"  {result.semantic.finite_rewrite}"])
    if result.semantic.pattern_status == "unknown":
        lines.extend(
            [
                "模式状态:",
                "  未知模式",
                "LLM 识别模式:",
                f"  {result.semantic.original_pattern or result.semantic.pattern}",
                "处理方式:",
                "  需要人工学习样例支持该模式的骨架与孔位模板；未写入持久性 RAG。",
            ]
        )
    lines.extend(["最终候选:", f"  {candidate}"])
    return "\n".join(lines)


def build_degraded_result_record(
    requirement_id: str,
    result: PipelineResult,
    *,
    expert_review_status: str = "pending",
    expert_comments: str = "",
) -> DegradedResultRecord:
    """Build the JSON-safe expert-review package for one degraded result."""

    if not result.trace_coverage.degraded:
        raise ValueError(
            "Degraded review records require degraded_unproven or degraded_proven."
        )

    provisional = result.fallback_candidate
    formula_classification = (
        exact_formula_classification(provisional.formula)
        if provisional is not None
        else "unknown"
    )
    trace_kinds = (
        "正轨迹、反轨迹或边界轨迹"
        if result.trace_evidence_policy.boundary_enabled
        else "正轨迹或反轨迹"
    )
    recommended_actions = [
        f"补充覆盖缺口对应的{trace_kinds}。",
        "重新执行轨迹一致性与覆盖检查后再决定是否批准。",
    ]
    if result.trace_coverage.status == "degraded_proven":
        recommended_actions.insert(
            0,
            "复核精确公式分类结论及其是否足以解释缺失轨迹类别。",
        )
    if provisional is None:
        recommended_actions.insert(0, "人工检查候选生成过程并请求重新运行。")

    return DegradedResultRecord(
        requirement_id=requirement_id,
        requirement=result.requirement,
        semantic=asdict(result.semantic),
        trace_coverage=asdict(result.trace_coverage),
        coverage_debt=tuple(result.trace_coverage.coverage_debt),
        retry_count=result.trace_coverage.retry_count,
        executable_semantics=result.trace_coverage.executable_semantics,
        all_traces=tuple(
            {
                "name": trace.name,
                "kind": trace.kind,
                "trace": trace_to_serializable(trace.trace),
                "expected": trace.expected,
                "adjudicated_label": trace.adjudicated_label,
                "label_source": trace.label_source,
                "role": trace.role,
                "query_role": trace.query_role,
                "reason": trace.reason,
                "beginner_view": trace.beginner_view,
            }
            for trace in result.traces
        ),
        skeletons=tuple(str(skeleton) for skeleton in result.skeletons),
        candidates=tuple(str(candidate) for candidate in result.candidates),
        evaluations=tuple(
            {
                "formula": str(evaluation.formula),
                "pos_sat": evaluation.pos_sat,
                "neg_rej": evaluation.neg_rej,
                "bound_match": evaluation.bound_match,
                "score": evaluation.score,
                "failure_summary": list(evaluation.failure_summary),
                "trace_results": [
                    {
                        "trace_name": trace_result.trace_name,
                        "trace_kind": trace_result.trace_kind,
                        "expected": trace_result.expected,
                        "actual": trace_result.actual,
                        "matched": trace_result.matched,
                        "failure_reason": trace_result.failure_reason,
                    }
                    for trace_result in evaluation.trace_results
                ],
            }
            for evaluation in result.evaluations
        ),
        provisional_candidate=(
            {
                "formula": str(provisional.formula),
                "pos_sat": provisional.pos_sat,
                "neg_rej": provisional.neg_rej,
                "bound_match": provisional.bound_match,
                "score": provisional.score,
            }
            if provisional is not None
            else None
        ),
        exact_formula_classification=formula_classification,
        selection_status=result.selection_status,
        persistent_rag_written=False,
        review_records=tuple(
            _json_safe_record(normalize_review_record(record))
            for record in result.review_records
        ),
        expert_review_status=expert_review_status,
        expert_comments=expert_comments,
        recommended_actions=tuple(recommended_actions),
    )


def format_degraded_result(
    record: DegradedResultRecord,
    pipeline_report: str,
) -> str:
    """Format a degraded item as an explicitly provisional expert report."""

    provisional = record.provisional_candidate or {}
    lines = [
        "Traj-LTLf 降级结果专家审查包",
        "=" * 40,
        f"需求 ID: {record.requirement_id}",
        f"覆盖状态: {record.trace_coverage.get('status', '')}",
        f"专家状态: {record.expert_review_status}",
        "正式选择: 无",
        (
            "Provisional 候选: "
            + str(provisional.get("formula", "无可用候选"))
        ),
        f"精确公式分类: {record.exact_formula_classification}",
        "持久化 RAG: 未写入",
        "",
        "证据缺口:",
        *(
            [f"- {item}" for item in record.coverage_debt]
            or ["- 未提供具体 coverage debt，请人工检查 trace_coverage。"]
        ),
        "",
        "建议操作:",
        *[f"- {item}" for item in record.recommended_actions],
        "",
        "完整 pipeline 报告:",
        pipeline_report,
    ]
    return "\n".join(lines)


def format_degraded_summary(records: list[DegradedResultRecord]) -> str:
    """Format the concise list used by experts to triage degraded items."""

    lines = [
        "Traj-LTLf 降级结果摘要",
        f"降级项数量: {len(records)}",
    ]
    for record in records:
        provisional = record.provisional_candidate or {}
        lines.extend(
            [
                "",
                f"[{record.requirement_id}] {record.trace_coverage.get('status', '')}",
                f"需求: {record.requirement}",
                f"专家状态: {record.expert_review_status}",
                f"Provisional: {provisional.get('formula', '无')}",
                "核心缺口:",
                *(
                    [f"- {item}" for item in record.coverage_debt]
                    or ["- 未记录具体缺口"]
                ),
            ]
        )
    return "\n".join(lines)


def _json_safe_record(record: dict) -> dict:
    """Normalize nested review records before storing them in JSON packages."""

    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, set):
            return sorted(convert(item) for item in value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Formula):
            return str(value)
        return value

    return convert(record)


def _save_skeleton_record(record: dict, records_dir: str) -> Path:
    """保存单轮骨架生成与诊断记录。"""

    path = Path(records_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"skeleton_round_{record['round']}_{timestamp}.json"
    record_path = path / filename
    record["record_path"] = str(record_path)
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record_path


def _save_completion_record(record: dict, records_dir: str) -> Path:
    """保存单轮候选补全记录。"""

    path = Path(records_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"completion_round_{record['round']}_{timestamp}.json"
    record_path = path / filename
    record["record_path"] = str(record_path)
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record_path


def _save_selection_record(record: dict, records_dir: str) -> Path:
    """保存最终选择 needs_review 记录。"""

    path = Path(records_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"selection_round_{record['round']}_{timestamp}.json"
    record_path = path / filename
    record["record_path"] = str(record_path)
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record_path


def _build_strict_automatic_approval(
    *,
    semantic: SemanticStructure,
    selected: FormulaConsistencyEvaluation,
    ir_coverage: IRCoverageReport,
    run_origin: str,
) -> FinalApprovalRecord:
    ir = semantic.faithful_ir
    ir_version = ir.ir_version if ir is not None else ""
    requirement_id = ir.requirement_id if ir is not None else "REQ"
    candidate_id = selected.candidate_id or str(selected.formula)
    approved_at = datetime.now().isoformat(timespec="seconds")
    approval_id = (
        "approval-strict-auto-"
        + requirement_id
        + "-"
        + ir_version
        + "-"
        + re.sub(r"[^A-Za-z0-9_-]+", "_", candidate_id)[:48]
    )
    return FinalApprovalRecord(
        approval_id=approval_id,
        reviewer="strict_automatic_approval",
        approved_at=approved_at,
        approved_ir_version=ir_version,
        approved_candidate_id=candidate_id,
        approved_formula=str(selected.formula),
        approval_basis=(
            "ApprovedSemanticIR frozen",
            "trace coverage complete",
            "IR coverage complete",
            "ambiguity debt closed",
            "strict consistency pass",
            "unique exhaustive operator resolution",
            f"run_origin={run_origin}",
        ),
        ir_coverage_status=ir_coverage.status,
        consistency_status="strict_pass",
        writeback_allowed=(
            ir_coverage.complete and run_origin != "feedback_rerun"
        ),
        approval_status="final",
        independence_id="strict-mechanical-gate-v1",
        context_isolated=True,
        gold_exposed=False,
    )


def _save_trace_supplement_generation_record(
    *,
    requirement: str,
    semantic: SemanticStructure,
    round_index: int,
    existing_traces: list[LabeledTrace],
    generated_traces: list[LabeledTrace],
    supplemented_traces: list[LabeledTrace],
    combined_traces: list[LabeledTrace],
    target_trace_count: int,
    target_trace_length: int | None,
    runtime_rag_chunks: list[str],
    comments: str,
    records_dir: str,
) -> dict:
    """保存补轨生成结果。"""

    path = Path(records_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record = {
        "stage": "trace_supplement_generation",
        "round": round_index,
        "requirement": requirement,
        "semantic": asdict(semantic),
        "approved": True,
        "issue_type": "formula_free_semantic_debt",
        "target_trace_count": target_trace_count,
        "target_trace_length": target_trace_length,
        "existing_trace_count": len(existing_traces),
        "generated_trace_count": len(generated_traces),
        "supplemented_trace_count": len(supplemented_traces),
        "combined_trace_count": len(combined_traces),
        "existing_traces": [_labeled_trace_to_dict(trace) for trace in existing_traces],
        "generated_traces": [_labeled_trace_to_dict(trace) for trace in generated_traces],
        "supplemented_traces": [
            _labeled_trace_to_dict(trace) for trace in supplemented_traces
        ],
        "combined_traces": [_labeled_trace_to_dict(trace) for trace in combined_traces],
        "runtime_rag_chunks": runtime_rag_chunks,
        "comments": comments,
    }
    record_path = path / f"trace_supplement_generation_round_{round_index}_{timestamp}.json"
    record["record_path"] = str(record_path)
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


def _adjudicate_supplement_queries(
    *,
    requirement: str,
    semantic: SemanticStructure,
    queries: list[TraceQuery],
    experiment_config: ExperimentConfig,
    review_mode: str,
    round_index: int,
    records_dir: str,
    judge_client: JSONLLMClient | None = None,
    llm_client: JSONLLMClient | None = None,
) -> object | None:
    if not queries:
        return None
    brief = build_trace_adjudication_brief(requirement, semantic)
    queries = bind_trace_queries_to_brief(queries, brief)
    if experiment_config.mode == "formula_oracle":
        annotations = oracle_annotations(queries, experiment_config)
    elif experiment_config.mode == "independent_judge" and review_mode == "auto":
        annotations = independent_judge_annotations(
            requirement,
            semantic,
            queries,
            judge_client=judge_client,
            adjudication_brief=brief,
        )
    elif review_mode == "auto":
        annotations = []
    else:
        annotations = manual_trace_annotations(
            requirement=requirement, semantic=semantic, queries=queries,
            round_index=round_index, records_dir=records_dir,
            adjudication_brief=brief, llm_client=llm_client,
        )
    return adjudicate_trace_round(queries, annotations, semantic, brief)


def _trace_signature(trace: LabeledTrace) -> tuple:
    return (
        trace.kind,
        trace.expected,
        tuple(tuple(sorted(step)) for step in trace.trace),
    )


def _labeled_trace_to_dict(trace: LabeledTrace) -> dict:
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


def _build_skeleton_stage_errors(
    diagnostics: list[SkeletonDiagnostic],
    round_index: int,
) -> list[StageErrorRecord]:
    """把每个失败骨架诊断转换成一条可检索错误。"""

    errors: list[StageErrorRecord] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for item_index, diagnostic in enumerate(diagnostics, start=1):
        if diagnostic.passed:
            continue
        errors.append(
            StageErrorRecord(
                error_id=f"skeleton-{round_index}-{item_index}",
                stage="skeleton",
                round_index=round_index,
                artifact=str(diagnostic.skeleton),
                error_type=diagnostic.error_type,
                source="validator",
                evidence=diagnostic.trace_evidence,
                suggestion=diagnostic.suggestion,
                comments="骨架未通过正轨迹/反轨迹诊断。",
                timestamp=timestamp,
            )
        )
    return errors


def _build_completion_stage_errors(
    decisions: list[CompletionDecision],
    round_index: int,
    strict_boundary: bool,
    semantic: SemanticStructure,
) -> list[StageErrorRecord]:
    """把每个未通过严格过滤的补全候选转换成一条错误。"""

    errors: list[StageErrorRecord] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for item_index, decision in enumerate(decisions, start=1):
        reject_reasons = completion_decision_reject_reasons(
            decision=decision,
            strict_boundary=strict_boundary,
            semantic=semantic,
        )
        if not reject_reasons:
            continue
        errors.append(
            StageErrorRecord(
                error_id=f"completion-{round_index}-{item_index}",
                stage="completion",
                round_index=round_index,
                artifact=json.dumps(
                    _completion_decision_to_dict(
                        decision=decision,
                        strict_boundary=strict_boundary,
                        semantic=semantic,
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                error_type=_reject_error_type(reject_reasons),
                source="consistency_check",
                evidence=_completion_error_evidence(decision, reject_reasons),
                suggestion=(
                    "下一轮 CompletionAgent 应结合这些失败原因调整 H_scope "
                    "和 H_response 的 X/F/G/XF 选择。"
                ),
                comments="；".join(reject_reasons),
                timestamp=timestamp,
            )
        )
    return errors


def _build_invalid_completion_stage_errors(
    invalid_records: list[dict],
    *,
    round_index: int,
) -> list[StageErrorRecord]:
    """Convert invalid LLM completion items into reviewable stage errors."""

    errors: list[StageErrorRecord] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for item_index, record in enumerate(invalid_records, start=1):
        artifact = {
            "candidate_id": record.get("candidate_id", ""),
            "template_text": record.get("template_text", ""),
            "formula_text": record.get("formula_text", ""),
            "assignments": record.get("assignments", []),
            "fallback_used": bool(record.get("fallback_used")),
            "raw_item": record.get("raw_item"),
            "raw_completion": record.get("raw_completion"),
        }
        errors.append(
            StageErrorRecord(
                error_id=f"completion-invalid-{round_index}-{item_index}",
                stage="completion",
                round_index=round_index,
                artifact=json.dumps(
                    artifact,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                error_type="invalid_completion",
                source="llm_completion_parser",
                evidence=(
                    f"error_type={record.get('error_type', '')}",
                    "invalid_reason="
                    + str(record.get("invalid_reason", record.get("error", ""))),
                    "fallback_used="
                    + str(bool(record.get("fallback_used"))).lower(),
                ),
                suggestion=(
                    "下一轮 CompletionAgent 必须返回可解析完整公式，"
                    "并确保 epsilon 只出现在 H_xxx assignments 中。"
                ),
                comments=str(
                    record.get("invalid_reason", record.get("error", ""))
                ),
                timestamp=timestamp,
            )
        )
    return errors


def _build_selection_stage_errors(
    evaluations: list[FormulaConsistencyEvaluation],
    strict_boundary: bool,
    semantic: SemanticStructure,
) -> list[StageErrorRecord]:
    """把最终选择阶段每个未严格匹配候选转换成一条错误。"""

    errors: list[StageErrorRecord] = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    for item_index, evaluation in enumerate(rank_evaluations(evaluations), start=1):
        reject_reasons = candidate_reject_reasons(
            evaluation=evaluation,
            strict_boundary=strict_boundary,
            semantic=semantic,
        )
        if not reject_reasons:
            continue
        artifact = {
            "formula": str(evaluation.formula),
            "pos_sat": evaluation.pos_sat,
            "neg_rej": evaluation.neg_rej,
            "bound_match": evaluation.bound_match,
            "score": evaluation.score,
            "failure_summary": list(evaluation.failure_summary),
        }
        errors.append(
            StageErrorRecord(
                error_id=f"selection-1-{item_index}",
                stage="selection",
                round_index=1,
                artifact=json.dumps(artifact, ensure_ascii=False, indent=2),
                error_type=_reject_error_type(reject_reasons),
                source="consistency_check",
                evidence=_selection_error_evidence(evaluation, reject_reasons),
                suggestion=(
                    "下一次迭代应回到骨架或补全阶段，生成能同时满足语义硬过滤、"
                    + (
                        "正轨迹、反轨迹和边界轨迹的候选公式。"
                        if evaluation.boundary_metric_available
                        else "正轨迹和反轨迹的候选公式。"
                    )
                ),
                comments="；".join(reject_reasons),
                timestamp=timestamp,
            )
        )
    return errors


def _completion_error_evidence(
    decision: CompletionDecision,
    reject_reasons: tuple[str, ...],
) -> tuple[str, ...]:
    evidence = [*reject_reasons]
    evidence.extend(decision.operator_derivation[:3])
    evidence.extend(decision.trace_derivation[:3])
    return tuple(evidence)


def _selection_error_evidence(
    evaluation: FormulaConsistencyEvaluation,
    reject_reasons: tuple[str, ...],
) -> tuple[str, ...]:
    evidence = [*reject_reasons]
    evidence.extend(evaluation.failure_summary[:5])
    return tuple(evidence)


def _reject_error_type(reject_reasons: tuple[str, ...]) -> str:
    combined = "\n".join(reject_reasons)
    if "缺少" in combined or "trigger -> response" in combined:
        return "semantic_guard_failed"
    if "PosSat" in combined:
        return "positive_trace_mismatch"
    if "NegRej" in combined:
        return "negative_trace_mismatch"
    if "BoundMatch" in combined:
        return "boundary_trace_mismatch"
    return "candidate_rejected"


def _skeleton_diagnostic_to_dict(diagnostic: SkeletonDiagnostic) -> dict:
    """把 SkeletonDiagnostic 转换成可写入 JSON 的字典。"""

    return {
        "skeleton": str(diagnostic.skeleton),
        "passed": diagnostic.passed,
        "error_type": diagnostic.error_type,
        "trace_evidence": list(diagnostic.trace_evidence),
        "suggestion": diagnostic.suggestion,
    }


def _hole_candidate_to_dict(hole_candidate: HoleCandidate) -> dict:
    """把 HoleCandidate 转换成可写入 JSON 的字典。"""

    return {
        "candidate_id": hole_candidate.candidate_id,
        "skeleton": str(hole_candidate.skeleton),
        "template_text": hole_candidate.template_text,
        "holes": [
            {
                "hole_id": hole.hole_id,
                "target": hole.target,
                "allowed_ops": list(hole.allowed_ops),
                "reason": hole.reason,
                "role": hole.role,
                "parent_hole_id": hole.parent_hole_id,
                "source_hint": list(hole.source_hint),
            }
            for hole in hole_candidate.holes
        ],
        "rag_context": list(hole_candidate.rag_context),
        "explanation": hole_candidate.explanation,
    }


def _completion_decision_to_dict(
    decision: CompletionDecision,
    strict_boundary: bool,
    semantic: SemanticStructure,
) -> dict:
    """把 CompletionDecision 转换成可写入 JSON 的字典。"""

    reject_reasons = completion_decision_reject_reasons(
        decision=decision,
        strict_boundary=strict_boundary,
        semantic=semantic,
    )
    return {
        "candidate_id": decision.candidate_id,
        "template_text": decision.template_text,
        "formula": str(decision.formula),
        "assignments": [
            {"hole_id": hole_id, "selected_op": selected_op}
            for hole_id, selected_op in decision.assignments
        ],
        "boundary_match": decision.boundary_match,
        "pos_sat": decision.pos_sat,
        "neg_rej": decision.neg_rej,
        "score": decision.score,
        "operator_derivation": list(decision.operator_derivation),
        "trace_derivation": list(decision.trace_derivation),
        "retained": True,
        "posthoc_reject_reasons": list(reject_reasons),
        "reason": decision.reason,
        "ir_version": decision.ir_version,
        "closure_status": (
            decision.closure_report.status
            if decision.closure_report is not None
            else "missing"
        ),
    }


def _format_completion_feedback(
    decisions: list[CompletionDecision],
    strict_boundary: bool,
    semantic: SemanticStructure,
) -> str:
    """把补全后验复核信息压缩成 Agent 可用的反馈文本。"""

    boundary_available = any(
        decision.boundary_metric_available for decision in decisions
    )
    lines = [
        "候选补全后验模拟未找到严格匹配公式。",
        f"strict_boundary={strict_boundary}",
        (
            "请同时根据正轨迹、反轨迹、边界轨迹、NL 和 RAG 调整 X/F/G 孔位。"
            if boundary_available
            else "请同时根据正轨迹、反轨迹、NL 和 RAG 调整 X/F/G 孔位。"
        ),
    ]
    for decision in decisions[:5]:
        lines.append(f"候选: {decision.formula}")
        lines.append(
            "指标: "
            f"PosSat={decision.pos_sat:.2f}, "
            f"NegRej={decision.neg_rej:.2f}, "
            + (
                f"BoundMatch={decision.boundary_match:.2f}"
                if decision.boundary_metric_available
                else "BoundMatch=N/A"
            )
        )
        reject_reasons = completion_decision_reject_reasons(
            decision=decision,
            strict_boundary=strict_boundary,
            semantic=semantic,
        )
        if reject_reasons:
            lines.append("后验复核原因: " + "；".join(reject_reasons[:3]))
        for derivation in decision.trace_derivation[:2]:
            lines.append(f"轨迹推导: {derivation}")
    return "\n".join(lines)


def _kind_name(kind: str) -> str:
    if kind == "positive":
        return "正轨迹"
    if kind == "negative":
        return "反轨迹"
    if kind == "boundary":
        return "边界轨迹"
    return kind











