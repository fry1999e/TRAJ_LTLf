"""
模型语义检查、通俗人工轨迹判断与重生成循环。

这个模块实现两个门：
    1. 主模型处理语义结构检查
    2. 用户判断有限轨迹是否符合原文，主模型整理技术字段

只有评审通过后，pipeline 才会进入下一个阶段。若不通过，Agent 会接收评审
意见并重新生成。
"""

from __future__ import annotations
from execution_observer import emit, observed_stage

from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path

from agents.nlu_agent import NLUAgent
from ap_binding_context import APBindingContext
from ap_preparation import APPreparationError
from faithful_ir import FaithfulIRValidationError
from agents.base_agent import JSONLLMClient
from agents.trace_agent import (
    TraceGenerationAgent,
    assess_authoritative_trace_coverage,
    assess_trace_coverage,
    trace_coverage_feedback,
)
from semantic_promotion import AuthoritativeTraceSession
from data_examples import LabeledTrace, trace_semantics_to_dict
from rag.runtime_memory import record_stage_errors, record_trace_length_issue
from schemas import (
    ExperimentConfig,
    PlainReadingReview,
    SemanticStructure,
    StageErrorRecord,
    SurfaceSegmentation,
    TraceCoverageReport,
    TraceEvidencePolicy,
)
from boundary_trace_fixture import (
    BoundaryTraceFixtureError,
    bind_frozen_boundary_evidence,
    load_boundary_trace_fixture,
)
from ir_compiler import compile_executable_trace_plan
from trace_evidence_policy import (
    build_trace_evidence_context,
    filter_queries_for_policy,
    filter_traces_for_policy,
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
from trace_formatter import trace_to_serializable
from trace_review_protocol import (
    manual_trace_annotations,
    annotation_record,
    independent_judge_annotations,
    trace_query_to_dict,
)
from utils import format_truth


ReviewMode = str


@dataclass(frozen=True)
class ReviewDecision:
    """一次人工评审的结构化结果。"""

    approved: bool
    comments: str
    issue_type: str = "approved"
    target_trace_length: int | None = None
    confirmed_claim_ids: tuple[str, ...] = ()
    semantic_revision_authorized: bool | None = None


class ExpertPolicyReviewRequired(RuntimeError):
    """Normal pending state for a formula-blind expert policy review."""

    def __init__(self, review_path: str, branch_set_id: str):
        super().__init__(
            "finite-trace policy candidates require locked expert trace labels"
        )
        self.review_path = review_path
        self.branch_set_id = branch_set_id
        self.status = "awaiting_expert_semantic_review"


@observed_stage("structure_review")
def run_semantic_review(
    requirement: str,
    agent: NLUAgent,
    review_mode: ReviewMode = "manual",
    max_rounds: int = 3,
    records_dir: str = "review_records",
    surface_segmentation: SurfaceSegmentation | None = None,
    plain_reading_review: PlainReadingReview | None = None,
    ap_binding_context: APBindingContext | None = None,
    structure_reviewer=None,
    initial_feedback_history: list[str] | None = None,
) -> tuple[SemanticStructure, list[dict]]:
    """运行语义结构生成与人工评审闭环。"""

    feedback_history: list[str] = list(initial_feedback_history or ())
    records: list[dict] = []
    progress_guard = None
    last_legal_semantic = None
    semantic = None
    checked_meanings = set()

    for round_index in range(1, max_rounds + 1):
        if feedback_history:
            emit("semantic_revision", source="structure_review", round_index=round_index)
        try:
            revised = agent.generate(
                requirement,
                feedback_history=feedback_history,
                surface_segmentation=surface_segmentation,
                plain_reading_review=plain_reading_review,
                ap_binding_context=ap_binding_context,
            )
        except (FaithfulIRValidationError, APPreparationError) as exc:
            from candidate_delivery import recover_generation_candidate
            retained, record = recover_generation_candidate(
                requirement, semantic, exc, surface_segmentation=surface_segmentation,
                plain_reading_review=plain_reading_review, ap_binding_context=ap_binding_context,
                client=getattr(agent, "llm_client", None),
            )
            record["round"] = round_index
            record["record_path"] = str(_save_review_record(record, records_dir))
            records.append(record)
            return retained, records
        semantic = revised
        emit("ir_proposal_validated", semantic=semantic, round_index=round_index)
        from semantic_freezing import attach_candidate_ir_set
        semantic = attach_candidate_ir_set(semantic, requirement, max_branch_count=4)
        if progress_guard is not None:
            from semantic_progress import semantic_fingerprint
            if semantic_fingerprint(semantic.faithful_ir) in progress_guard.seen:
                records.append({"stage": "semantic_revision", "round": round_index,
                    "status": "cycle_rolled_back", "reason_code": "SEMANTIC_REVISION_NO_PROGRESS"})
                return replace(last_legal_semantic, semantic_status="needs_semantic_review"), records
        proxy_decision = (
            structure_reviewer(requirement, semantic, plain_reading_review, ap_binding_context)
            if structure_reviewer is not None else None
        )
        model_assisted = proxy_decision is None and review_mode == "manual" and agent.llm_enabled
        if model_assisted:
            from nonexpert_review import review_semantic_with_model
            from semantic_review_view import semantic_review_fingerprint
            meaning = semantic_review_fingerprint(semantic)
            if meaning is not None and meaning in checked_meanings:
                record = {"stage": "semantic_revision", "round": round_index,
                    "status": "no_progress_candidate_retained", "approved": False,
                    "reviewer_source": "model_assistant", "reason_code": "SEMANTIC_REVISION_NO_PROGRESS",
                    "comments": "Regeneration repeated a rejected meaning; skip duplicate model checks and retain the unapproved candidate.",
                    "artifact": asdict(semantic)}
                record["record_path"] = str(_save_review_record(record, records_dir))
                records.append(record)
                return replace(semantic, semantic_status="needs_semantic_review",
                               persistent_rag_allowed=False, label_source="model_assistant"), records
            if meaning is not None:
                checked_meanings.add(meaning)
            checked = review_semantic_with_model(requirement, semantic, agent.llm_client)
            proxy_decision = ReviewDecision(approved=checked["approved"], comments=checked["reason"])
            semantic = replace(semantic, label_source="model_assistant")
        if proxy_decision is not None and not isinstance(proxy_decision, ReviewDecision):
            raise TypeError("structure reviewer must return ReviewDecision")
        if requires_ir_candidate_adjudication(semantic) and (proxy_decision is None or proxy_decision.approved):
            emit("ir_reviewed", semantic=semantic, decision=proxy_decision, branch_adjudication_pending=True)
            record = {
                "stage": "semantic_policy_candidates",
                "round": round_index,
                "requirement": requirement,
                "artifact": asdict(semantic),
                "approved": False,
                "issue_type": "awaiting_ir_trajectory_adjudication",
                "reviewer_source": "model_assistant" if model_assisted else "custom" if structure_reviewer else "human",
                "comments": (
                    "Ordinary-language review is closed; legal finite-trace policy "
                    "branches will be resolved using candidate-blind human "
                    "trajectory annotations."
                ),
            }
            record["record_path"] = str(_save_review_record(record, records_dir))
            records.append(record)
            return semantic, records
        review_text = format_semantic_for_review(semantic)
        decision = proxy_decision if proxy_decision is not None else _get_review_decision("语义结构", review_text, review_mode)
        if decision.semantic_revision_authorized is not None:
            from semantic_progress import ConflictVector, ProgressGuard, SemanticProgressStop
            vector = ConflictVector(verified_source_contradictions=len(decision.confirmed_claim_ids),
                unresolved_required_contract_axes=int(not decision.approved and not decision.confirmed_claim_ids))
            if progress_guard is not None:
                try:
                    progress_guard.accept(semantic.faithful_ir, vector)
                except SemanticProgressStop:
                    records.append({"stage": "semantic_revision", "round": round_index,
                        "status": "no_progress_rolled_back", "reason_code": "SEMANTIC_REVISION_NO_PROGRESS"})
                    return replace(last_legal_semantic, semantic_status="needs_semantic_review"), records
            else:
                progress_guard = ProgressGuard(semantic.faithful_ir, vector)
            last_legal_semantic = semantic
        if semantic.semantic_status == "needs_semantic_review" and decision.approved:
            decision = ReviewDecision(
                approved=False,
                comments=(
                    "FaithfulSemanticIR contains unresolved high-risk ambiguity; "
                    "bind feedback to the affected IR node and resolve it explicitly."
                ),
                issue_type="semantic_ambiguity",
            )

        stage_errors: list[StageErrorRecord] = []
        rag_chunk_ids: list[str] = []
        if not decision.approved:
            stage_errors = [
                _build_semantic_review_error(
                    semantic=semantic,
                    decision=decision,
                    round_index=round_index,
                    reviewer_source="model_assistant" if model_assisted else "human_review",
                )
            ]
            rag_chunk_ids = record_stage_errors(
                retriever=agent.retriever,
                requirement=requirement,
                semantic=semantic,
                stage="semantic",
                round_index=round_index,
                errors=stage_errors,
            )

        record = {
            "stage": "semantic",
            "round": round_index,
            "requirement": requirement,
            "artifact": asdict(semantic),
            "approved": decision.approved,
            "issue_type": decision.issue_type,
            "runtime_rag_chunks": rag_chunk_ids,
            "stage_errors": [asdict(error) for error in stage_errors],
            "comments": decision.comments,
            "reviewer_source": "model_assistant" if model_assisted else "custom" if structure_reviewer else "human",
        }
        record["record_path"] = str(_save_review_record(record, records_dir))
        records.append(record)

        if decision.approved:
            emit("ir_reviewed", semantic=semantic, decision=decision, branch_adjudication_pending=False)
            return semantic, records

        if decision.semantic_revision_authorized is False:
            return replace(semantic, semantic_status="needs_semantic_review"), records

        feedback_history.append(decision.comments)

    if progress_guard is not None:
        return replace(last_legal_semantic, semantic_status="needs_semantic_review"), records
    if semantic is not None and semantic.faithful_ir is not None:
        records.append({"stage": "semantic_review_limit", "status": "candidate_fallback",
                        "comments": "Model review limit reached; retain the validated source candidate without approval."})
        return replace(semantic, semantic_status="needs_semantic_review"), records
    raise RuntimeError("语义结构检查未通过，且未生成可用候选。")


@observed_stage("trace_review")
def run_trace_review(
    requirement: str,
    semantic: SemanticStructure,
    agent: TraceGenerationAgent,
    review_mode: ReviewMode = "manual",
    max_rounds: int = 3,
    records_dir: str = "review_records",
    experiment_config: ExperimentConfig | None = None,
    judge_client: JSONLLMClient | None = None,
    return_semantic: bool = False,
    trace_evidence_policy: TraceEvidencePolicy = TraceEvidencePolicy(),
    requirement_id: str = "REQ-001",
    authoritative_trace_session: AuthoritativeTraceSession | None = None,
    input_semantics_profile: dict | None = None,
) -> (
    tuple[list[LabeledTrace], list[dict]]
    | tuple[SemanticStructure, list[LabeledTrace], list[dict]]
):
    """运行有限轨迹生成与人工评审闭环。"""

    experiment_config = experiment_config or ExperimentConfig(
        mode="human",
    )
    boundary_fixture = (
        load_boundary_trace_fixture(trace_evidence_policy.boundary_fixture_path)
        if trace_evidence_policy.boundary_source == "frozen_file"
        else None
    )
    if boundary_fixture is not None and (
        boundary_fixture.sha256 != trace_evidence_policy.boundary_fixture_sha256
        or boundary_fixture.schema_version
        != trace_evidence_policy.boundary_fixture_schema_version
    ):
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            "boundary fixture content or schema changed after run policy was created",
        )
    feedback_history: list[str] = []
    records: list[dict] = []
    target_trace_length: int | None = None
    accepted_trace_evidence: list[LabeledTrace] = []
    evidence_records: list[TraceEvidenceRecord] = []
    evidence_disputes: list[TraceEvidenceDispute] = []
    if requires_ir_candidate_adjudication(semantic):
        ir_selection = adjudicate_ir_candidate_branches(
            requirement,
            semantic,
            records_dir=records_dir,
            llm_client=agent.llm_client,
            data_split=experiment_config.data_split,
        )
        semantic = ir_selection.semantic
        report = semantic.ir_ambiguity_adjudication
        from human_review_session import semantic_review_budget
        records.append({
            "stage": "ir_candidate_trace_adjudication",
            "round": len(report.rounds) if report is not None else 0,
            "requirement": requirement,
            "candidate_set_id": (
                report.candidate_set_id if report is not None else ""
            ),
            "status": ir_selection.status,
            "selected_branch_id": ir_selection.selected_branch_id,
            "consistent_branch_ids": list(ir_selection.consistent_branch_ids),
            "supporting_query_ids": list(ir_selection.supporting_query_ids),
            "comments": ir_selection.reason,
            "trace_evidence_review_required": ir_selection.status != "unique",
            "provisional": ir_selection.status == "provisional_fallback",
            "human_review_budget": semantic_review_budget(requirement, semantic, records_dir),
        })
        if ir_selection.status != "unique":
            if return_semantic:
                return semantic, [], records
            return [], records
    semantic = attach_semantic_resolution_record(semantic)
    adjudication_brief = build_trace_adjudication_brief(requirement, semantic)
    trace_evidence_context = (
        build_trace_evidence_context(
            compile_executable_trace_plan(semantic.faithful_ir),
            trace_evidence_policy,
        )
        if semantic.faithful_ir is not None
        else None
    )

    for round_index in range(1, max_rounds + 1):
        query_kwargs = {
            "feedback_history": feedback_history,
            "target_trace_length": target_trace_length,
            "data_split": experiment_config.data_split,
            "label_source": (
                "formula_oracle"
                if experiment_config.mode == "formula_oracle"
                else "independent_judge"
                if experiment_config.mode == "independent_judge"
                else "human"
            ),
        }
        if (
            semantic.faithful_ir is not None
            and authoritative_trace_session is None
        ):
            raise ValueError(
                "Faithful-IR trace review requires an authoritative trace session"
            )
        generated_queries = (
            agent.generate_authoritative_queries(
                requirement,
                semantic,
                trace_session=authoritative_trace_session,
                **query_kwargs,
            )
            if authoritative_trace_session is not None
            else agent.generate_queries(
                requirement,
                semantic,
                **query_kwargs,
            )
        )
        queries = filter_queries_for_policy(
            generated_queries,
            trace_evidence_policy,
        )
        fixture_annotations = []
        if trace_evidence_policy.boundary_source == "frozen_file":
            if boundary_fixture is None:  # pragma: no cover
                raise BoundaryTraceFixtureError(
                    "BOUNDARY_FIXTURE_NOT_FOUND",
                    "frozen boundary fixture was not loaded",
                )
            if trace_evidence_context is None:
                raise BoundaryTraceFixtureError(
                    "BOUNDARY_FIXTURE_UNBOUND",
                    "frozen boundary evidence requires a validated FaithfulSemanticIR",
                )
            fixture_queries, fixture_annotations = bind_frozen_boundary_evidence(
                fixture=boundary_fixture,
                requirement_id=requirement_id,
                requirement=requirement,
                semantic=semantic,
                trace_plan=trace_evidence_context.canonical_plan,
                data_split=experiment_config.data_split,
            )
            queries.extend(fixture_queries)
        queries = bind_trace_queries_to_brief(queries, adjudication_brief)
        reused_annotations, pending_queries = reuse_confirmed_annotations(
            queries,
            evidence_records,
            semantic_context_id=adjudication_brief.semantic_context_id,
        )
        frozen_by_id = {
            annotation.query_id: annotation for annotation in fixture_annotations
        }
        frozen_annotations = [
            frozen_by_id[query.query_id]
            for query in pending_queries
            if query.query_id in frozen_by_id
        ]
        pending_queries = [
            query for query in pending_queries if query.query_id not in frozen_by_id
        ]
        if experiment_config.mode == "formula_oracle":
            new_annotations = oracle_annotations(
                pending_queries, experiment_config
            )
            annotation_review_path = ""
        elif review_mode == "auto":
            new_annotations = independent_judge_annotations(
                requirement,
                semantic,
                pending_queries,
                judge_client=(
                    judge_client
                    if experiment_config.mode == "independent_judge"
                    else None
                ),
                adjudication_brief=adjudication_brief,
            )
            annotation_review_path = ""
        else:
            annotation_review_path = ""
            new_annotations = manual_trace_annotations(
                requirement=requirement, semantic=semantic, queries=pending_queries,
                round_index=round_index, records_dir=records_dir,
                adjudication_brief=adjudication_brief, llm_client=agent.llm_client,
            )
        new_annotations = frozen_annotations + new_annotations
        annotations = reused_annotations + new_annotations
        outcome = adjudicate_trace_round(
            queries,
            annotations,
            semantic,
            adjudication_brief,
        )
        reconciliation = reconcile_trace_evidence(
            semantic=semantic,
            brief=adjudication_brief,
            existing_records=evidence_records,
            existing_disputes=evidence_disputes,
            existing_traces=(
                accepted_trace_evidence if not evidence_records else ()
            ),
            newly_adjudicated=outcome.accepted_traces,
        )
        conflict_resolution = resolve_trace_evidence_disputes(
            semantic=semantic,
            brief=adjudication_brief,
            reconciliation=reconciliation,
            resolver_client=None,
        )
        reconciliation = conflict_resolution.reconciliation
        from semantic_failure_packet import build_semantic_failure_packets
        failure_packets = build_semantic_failure_packets(
            source_text=requirement,
            input_profile=input_semantics_profile or {},
            semantic=semantic,
            reconciliation=reconciliation,
            resolution_reports=conflict_resolution.reports,
        )
        evidence_records = list(reconciliation.records)
        evidence_disputes = list(reconciliation.disputes)
        traces = filter_traces_for_policy(
            list(reconciliation.active_traces),
            trace_evidence_policy,
        )
        accepted_trace_evidence = traces
        unresolved_queries = list(outcome.unresolved_queries)
        if reconciliation.unresolved_disputes:
            coverage_report = TraceCoverageReport(
                status="incomplete",
                actual_count=len(traces),
                positive_count=sum(trace.expected for trace in traces),
                negative_count=sum(not trace.expected for trace in traces),
                boundary_count=sum(trace.kind == "boundary" for trace in traces),
                coverage_debt=tuple(
                    dict.fromkeys(
                        outcome.coverage_debt + reconciliation.coverage_debt
                    )
                ),
                retry_count=round_index - 1,
                executable_semantics=False,
                semantic_complete=False,
                quota_complete=False,
                reason=(
                    "Background source-first resolution could not safely settle "
                    "all trace evidence; use the existing semantic review flow."
                ),
            )
        elif outcome.has_semantic_challenge:
            semantic = attach_semantic_challenges(
                semantic,
                outcome.semantic_challenges,
            )
            coverage_report = TraceCoverageReport(
                status="incomplete",
                actual_count=len(traces),
                positive_count=sum(trace.expected for trace in traces),
                negative_count=sum(not trace.expected for trace in traces),
                boundary_count=sum(trace.kind == "boundary" for trace in traces),
                coverage_debt=outcome.coverage_debt,
                retry_count=round_index - 1,
                executable_semantics=False,
                semantic_complete=False,
                reason=(
                    "Trace review found a structured semantic challenge. The "
                    "last validated IR is preserved and semantic review is required."
                ),
            )
        else:
            coverage_kwargs = {
                "requirement": requirement,
                "feedback_history": feedback_history,
                "retry_count": round_index - 1,
                "allow_degraded": round_index == max_rounds,
                "trace_plan": (
                    trace_evidence_context.effective_plan
                    if trace_evidence_context is not None
                    else None
                ),
            }
            if (
                semantic.faithful_ir is not None
                and authoritative_trace_session is None
            ):
                raise ValueError(
                    "Faithful-IR coverage requires an authoritative trace session"
                )
            coverage_report = (
                assess_authoritative_trace_coverage(
                    traces,
                    semantic,
                    trace_session=authoritative_trace_session,
                    **coverage_kwargs,
                )
                if authoritative_trace_session is not None
                else assess_trace_coverage(
                    traces,
                    semantic,
                    **coverage_kwargs,
                )
            )
        decision = ReviewDecision(
            approved=True,
            comments="逐轨迹判断已完成，技术字段由模型整理。",
        )
        if outcome.has_adjudication_debt and coverage_report.status == "complete":
            coverage_report = replace(
                coverage_report,
                status="degraded_unproven",
                semantic_complete=False,
                coverage_debt=outcome.coverage_debt,
                reason=(
                    "存在 UNCLEAR/INFEASIBLE 轨迹；它们未进入候选评分，"
                    "需求歧义需要保留。"
                ),
            )
        from human_review_session import review_budget_state, budget_limited_coverage, review_should_stop
        review_budget = (review_budget_state(requirement, semantic, adjudication_brief, records_dir)
                         if review_mode == "manual" else None)
        budget_stopped = review_should_stop(review_budget) and not coverage_report.complete
        if budget_stopped:
            coverage_report = budget_limited_coverage(coverage_report, review_budget)
        if coverage_report.status in {"incomplete", "rejected"}:
            decision = ReviewDecision(
                approved=False,
                comments=trace_coverage_feedback(coverage_report)
                or coverage_report.reason,
                issue_type="content",
            )
        elif coverage_report.status == "degraded_unproven":
            decision = ReviewDecision(
                approved=True,
                comments=coverage_report.reason,
                issue_type="degraded",
            )
        discarded = not decision.approved and decision.issue_type == "length"
        rag_chunk_ids: list[str] = []
        stage_errors: list[StageErrorRecord] = []
        if discarded and decision.target_trace_length is not None:
            rag_chunk_ids = record_trace_length_issue(
                retriever=agent.retriever,
                requirement=requirement,
                semantic=semantic,
                traces=traces,
                round_index=round_index,
                target_trace_length=decision.target_trace_length,
                comments=decision.comments,
            )
        elif not decision.approved:
            stage_errors = [
                _build_trace_review_error(
                    traces=traces,
                    decision=decision,
                    round_index=round_index,
                )
            ]
            if coverage_report.complete:
                rag_chunk_ids = record_stage_errors(
                    retriever=agent.retriever,
                    requirement=requirement,
                    semantic=semantic,
                    stage="traces",
                    round_index=round_index,
                    errors=stage_errors,
                )

        record = {
            "stage": "traces",
            "round": round_index,
            "requirement_id": requirement_id,
            "requirement": requirement,
            "semantic": asdict(semantic),
            "experiment_mode": experiment_config.mode,
            "trace_evidence_policy": asdict(trace_evidence_policy),
            "oracle_assisted": False,
            "diagnostic_only": False,
            "annotation_review_path": annotation_review_path,
            "adjudication_brief": asdict(adjudication_brief),
            "adjudication_outcome": {
                "next_action": outcome.next_action,
                "annotation_debt": list(outcome.annotation_debt),
                "query_debt": list(outcome.query_debt),
                "semantic_challenges": [
                    asdict(item) for item in outcome.semantic_challenges
                ],
            },
            "trace_evidence_review_required": bool(reconciliation.unresolved_disputes) or budget_stopped,
            "human_review_budget": review_budget,
            "trace_evidence_disputes": [
                asdict(item) for item in reconciliation.disputes
            ],
            "background_conflict_resolutions": [
                asdict(item) for item in conflict_resolution.reports
            ],
            "semantic_failure_packets": list(failure_packets),
            "queries": [
                annotation_record(
                    query,
                    next(
                        (
                            annotation
                            for annotation in annotations
                            if annotation.query_id == query.query_id
                        ),
                        None,
                    ),
                    requirement=requirement,
                )
                for query in queries
            ],
            "unresolved_queries": [
                trace_query_to_dict(query) for query in unresolved_queries
            ],
            "artifact": [_trace_to_dict(trace) for trace in traces],
            "coverage_report": asdict(coverage_report),
            "approved": decision.approved,
            "issue_type": decision.issue_type,
            "requested_trace_length": target_trace_length,
            "target_trace_length": decision.target_trace_length,
            "discarded": discarded,
            "runtime_rag_chunks": rag_chunk_ids,
            "stage_errors": [asdict(error) for error in stage_errors],
            "comments": decision.comments,
        }
        if semantic.faithful_ir is not None:
            from contracts.trace_cache_lineage import build_trace_cache_lineage
            actual_reviewer = getattr(judge_client, "client", judge_client)
            reviewer_version = (
                getattr(actual_reviewer, "model", None)
                or getattr(actual_reviewer, "model_name", None)
                or "human-user"
            )
            record["annotation_cache_lineage"] = build_trace_cache_lineage(
                input_profile=input_semantics_profile or {},
                ir_version=semantic.faithful_ir.ir_version,
                queries=queries,
                prompt_hash=hashlib.sha256(
                    b"finite-trace-natural-language-adjudication-v1"
                ).hexdigest(),
                reviewer_model_version=str(reviewer_version),
            )
        if discarded:
            record["discarded_reason"] = "length_mismatch"
        record["record_path"] = str(_save_review_record(record, records_dir))
        records.append(record)

        if budget_stopped:
            semantic = replace(semantic, semantic_status="needs_semantic_review", persistent_rag_allowed=False)
        if budget_stopped or outcome.has_semantic_challenge or reconciliation.unresolved_disputes:
            return (
                (semantic, traces, records)
                if return_semantic
                else (traces, records)
            )

        if decision.approved:
            return (
                (semantic, traces, records)
                if return_semantic
                else (traces, records)
            )

        if decision.issue_type == "length":
            target_trace_length = decision.target_trace_length
            continue

        feedback_history.append(decision.comments)

    raise RuntimeError("有限轨迹未通过人工评审，已达到最大重试次数。")


def format_semantic_for_review(semantic: SemanticStructure) -> str:
    """把语义结构格式化成人工评审文本。"""

    lines = [
        "自然语言语义结构（模型处理）：",
        f"  AP = {{{semantic.trigger}, {semantic.response}}}",
        f"  trigger = {semantic.trigger}",
        f"  response = {semantic.response}",
        f"  pattern = {semantic.pattern}",
        f"  pattern_status = {semantic.pattern_status}",
        f"  scope keyword = {semantic.scope_keyword}",
        f"  temporal keyword = {semantic.temporal_keyword}",
        f"  trace domain = {semantic.trace_domain}",
        f"  persistent_rag_allowed = {semantic.persistent_rag_allowed}",
        f"  confidence = {semantic.confidence:.2f}",
        f"  explanation = {semantic.explanation}",
    ]
    if semantic.original_pattern:
        lines.append(f"  original_pattern = {semantic.original_pattern}")
    if semantic.formula_text:
        lines.append(f"  formula_text = {semantic.formula_text}")
    if semantic.reference_only:
        lines.append(f"  warning = {semantic.domain_warning}")
        if semantic.domain_evidence:
            lines.append("  evidence = " + ", ".join(semantic.domain_evidence))
        if semantic.finite_rewrite:
            lines.append(f"  finite rewrite = {semantic.finite_rewrite}")
    if semantic.rag_context:
        lines.append("  RAG context:")
        for context in semantic.rag_context:
            lines.append(f"    - {context}")
    return "\n".join(lines)


def format_traces_for_review(traces: list[LabeledTrace]) -> str:
    """把有限轨迹格式化成人工评审文本，优先展示 beginner_view。"""

    lines = ["请评审有限轨迹集合："]
    lines.extend(_format_trace_set_summary(traces))
    for trace in traces:
        lines.append("")
        lines.append(f"轨迹名称：{trace.name}")
        lines.append(f"轨迹类型：{_kind_name(trace.kind)}")
        lines.append("")
        lines.extend(f"  {line}" if line else "" for line in trace.beginner_view.splitlines())
        lines.append("")
        lines.append(f"人工标签：{format_truth(trace.expected)}")
        lines.append(f"裁决标签：{trace.adjudicated_label}")
        lines.append(f"置信度：{trace.confidence or 'unspecified'}")
        lines.append(f"标签来源：{trace.label_source}")
        if trace.rationale:
            lines.append(f"裁决理由：{trace.rationale}")
        if trace.semantics.semantic_label:
            lines.append(f"语义标签：{trace.semantics.semantic_label}")
        if trace.semantics.roles:
            lines.append("证据角色：" + ", ".join(trace.semantics.roles))
        if trace.semantics.confusion_targets:
            lines.append(
                "区分目标：" + ", ".join(trace.semantics.confusion_targets)
            )
        lines.append(f"生成理由：{trace.reason}")
    return "\n".join(lines)


def _format_trace_set_summary(traces: list[LabeledTrace]) -> list[str]:
    boundary_shapes = {
        tuple(tuple(sorted(step)) for step in trace.trace)
        for trace in traces
        if trace.kind == "boundary"
    }
    late_positive = any(
        trace.kind == "positive" and "late" in trace.name.lower()
        for trace in traces
    )
    late_negative = any(
        trace.kind == "negative" and "late" in trace.name.lower()
        for trace in traces
    )
    return [
        "",
        "轨迹集合摘要：",
        f"  总数 = {len(traces)}",
        f"  positive = {_count_trace_kind(traces, 'positive')}",
        f"  negative = {_count_trace_kind(traces, 'negative')}",
        f"  boundary = {_count_trace_kind(traces, 'boundary')}",
        f"  distinct boundary = {len(boundary_shapes)}",
        f"  late-trigger positive = {'yes' if late_positive else 'no'}",
        f"  late-trigger negative = {'yes' if late_negative else 'no'}",
    ]


def _count_trace_kind(traces: list[LabeledTrace], kind: str) -> int:
    return sum(1 for trace in traces if trace.kind == kind)


def _get_review_decision(
    stage_name: str,
    review_text: str,
    review_mode: ReviewMode,
) -> ReviewDecision:
    """
    获取评审结果。

    manual:
        打印待评审内容并等待人工输入。

    auto:
        自动通过，用于测试、CI 或非交互环境。它仍然会生成评审记录。
    """

    if review_mode == "auto":
        return ReviewDecision(
            approved=True,
            comments=f"自动评审通过：{stage_name} 在非交互/测试模式下通过。",
        )

    print("")
    print("=" * 60)
    print(review_text)
    print("=" * 60)
    answer = input(f"是否通过{stage_name}评审？输入 y/n: ").strip().lower()
    if answer in {"y", "yes", "是", "通过"}:
        return ReviewDecision(approved=True, comments="人工评审通过。")

    if stage_name == "有限轨迹":
        issue_answer = input(
            "未通过原因是内容错误还是轨迹长度错误？输入 c/l: "
        ).strip().lower()
        if issue_answer in {"l", "length", "len", "长度", "轨迹长度"}:
            target_length = _read_target_trace_length()
            return ReviewDecision(
                approved=False,
                issue_type="length",
                target_trace_length=target_length,
                comments=(
                    "人工评审指出当前有限轨迹内容可用，但长度不适合判断需求；"
                    f"目标轨迹长度为 {target_length}，当前轨迹集合已丢弃。"
                ),
            )

    comments = input("请输入评审意见，Agent 将结合意见重新生成: ").strip()
    if not comments:
        comments = "人工评审未通过，但未填写具体意见。"
    return ReviewDecision(approved=False, issue_type="content", comments=comments)


def _read_target_trace_length() -> int:
    """读取人工给出的目标有限轨迹长度。"""

    while True:
        raw_value = input(
            "请输入能够判断当前自然语言需求的有限轨迹长度（正整数）: "
        ).strip()
        try:
            target_length = int(raw_value)
        except ValueError:
            print("轨迹长度必须是正整数，请重新输入。")
            continue

        if target_length >= 1:
            return target_length

        print("轨迹长度必须至少为 1，请重新输入。")


def _build_semantic_review_error(
    semantic: SemanticStructure,
    decision: ReviewDecision,
    round_index: int,
    reviewer_source: str = "human_review",
) -> StageErrorRecord:
    return StageErrorRecord(
        error_id=f"semantic-{round_index}-human-review",
        stage="semantic",
        round_index=round_index,
        artifact=json.dumps(asdict(semantic), ensure_ascii=False, indent=2),
        error_type=decision.issue_type,
        source=reviewer_source,
        evidence=(decision.comments,),
        suggestion=(
            "下一轮 NLUAgent 应根据评审意见修正 trigger、response、"
            "pattern 或时序关键词，并避免重复输出相同语义结构。"
        ),
        comments=decision.comments,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )


def _build_trace_review_error(
    traces: list[LabeledTrace],
    decision: ReviewDecision,
    round_index: int,
) -> StageErrorRecord:
    return StageErrorRecord(
        error_id=f"traces-{round_index}-human-review",
        stage="traces",
        round_index=round_index,
        artifact=json.dumps([_trace_to_dict(trace) for trace in traces], ensure_ascii=False, indent=2),
        error_type=decision.issue_type,
        source="human_review",
        evidence=_trace_error_evidence(traces),
        suggestion=(
            "下一轮 TraceGenerationAgent 应结合人工意见重新生成正轨迹、"
            "反轨迹和边界轨迹，并避免重复当前未通过的轨迹模式。"
        ),
        comments=decision.comments,
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


def _save_review_record(record: dict, records_dir: str) -> Path:
    """保存单轮评审记录。"""

    path = Path(records_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{record['stage']}_round_{record['round']}_{timestamp}.json"
    record_path = path / filename
    record["record_path"] = str(record_path)
    record_path.write_text(
        json.dumps(_review_record_json_value(record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record_path


def _review_record_json_value(value):
    """Normalize native trace state containers at the persistence boundary."""

    if is_dataclass(value):
        return _review_record_json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _review_record_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_review_record_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_review_record_json_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported review-record value: {type(value).__name__}")


def _trace_to_dict(trace: LabeledTrace) -> dict:
    """把 LabeledTrace 转换成可以写 JSON 的字典。"""

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


def _kind_name(kind: str) -> str:
    if kind == "positive":
        return "正轨迹"
    if kind == "negative":
        return "反轨迹"
    if kind == "boundary":
        return "边界轨迹"
    return kind
