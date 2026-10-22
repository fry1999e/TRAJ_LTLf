"""
运行期正确样例 RAG 记忆。

错误记忆帮助 Agent 避免重复犯错；这里的正确样例记忆则用于大批量 NL
处理时的小样本学习。人工确认通过的语义、轨迹和带孔模板会被追加为
KnowledgeChunk，后续相似需求通过余弦相似度检索 Top-K 参考。
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json

from contract_versions import (
    FAITHFUL_IR_CONTRACT_VERSION,
    FAITHFUL_IR_VALIDATOR_VERSION,
)

from data_examples import LabeledTrace, trace_semantics_to_dict
from guided_learning import GuidedLearningContext
from formula_closure import (
    build_strict_formula_closure_policy,
    validate_formula_closure,
)
from ltlf_parser import Formula, parse_formula
from pattern_registry import (
    PatternSpec,
    derive_pattern_spec_template_derivations,
    normalize_formula_text,
)
from plain_reading import assess_plain_reading_ir_readiness
from ltlf_syntax import normalize_boolean_operator_text
from rag.knowledge_base import KnowledgeChunk
from rag.retriever import KeywordRetriever
from schemas import (
    CompletionDecision,
    FormulaConsistencyEvaluation,
    FaithfulLearningExample,
    HoleCandidate,
    IRCoverageReport,
    SemanticStructure,
    WARMUP_LEARNING_MODE,
    CompletionHolePlan,
    ConstrainedSkeletonPlan,
    FinalApprovalRecord,
    TraceObligationPlan,
)
from semantic_obligation_planner import plan_semantic_obligations
from trace_formatter import trace_to_serializable
from trace_obligations import (
    OBLIGATION_COMPILER_VERSION,
    analyze_obligations,
    enrich_trace_semantics,
    validate_trace_labels,
)
from utils import format_trace, format_truth


APPROVED_SOURCE_PREFIX = "approved-example"
SEMANTIC_SOURCE = f"{APPROVED_SOURCE_PREFIX}:semantic"
FAITHFUL_IR_SOURCE = f"{APPROVED_SOURCE_PREFIX}:faithful-ir"
TRACE_SOURCE = f"{APPROVED_SOURCE_PREFIX}:trace"
TEMPLATE_SOURCE = f"{APPROVED_SOURCE_PREFIX}:template"
OPERATOR_COMPLETION_SOURCE = f"{APPROVED_SOURCE_PREFIX}:operator-completion"
TRACE_ADEQUACY_SOURCE = f"{APPROVED_SOURCE_PREFIX}:trace-adequacy"
FULL_PROCESS_SOURCE = f"{APPROVED_SOURCE_PREFIX}:full-process"
PATTERN_SOURCE = f"{APPROVED_SOURCE_PREFIX}:pattern"
TEMPORAL_DOMAIN_SOURCE = f"{APPROVED_SOURCE_PREFIX}:temporal-domain"
GUIDED_SEMANTIC_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-semantic"
GUIDED_TRACE_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-trace"
GUIDED_SKELETON_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-skeleton"
GUIDED_TEMPLATE_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-template"
GUIDED_COMPLETION_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-completion"
GUIDED_CONSISTENCY_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-consistency"
GUIDED_SELECTION_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-selection"
GUIDED_FULL_PROCESS_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-full-process"
GUIDED_CORRECTION_SEMANTIC_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-correction-semantic"
SURFACE_SEGMENTATION_SOURCE = f"{APPROVED_SOURCE_PREFIX}:surface-segmentation"
PLAIN_READING_SOURCE = f"{APPROVED_SOURCE_PREFIX}:plain-reading"
GUIDED_CORRECTION_SEGMENTATION_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-correction-segmentation"
GUIDED_CORRECTION_PLAIN_READING_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-correction-plain-reading"
GUIDED_CORRECTION_TRACE_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-correction-trace"
GUIDED_CORRECTION_SKELETON_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-correction-skeleton"
GUIDED_CORRECTION_TEMPLATE_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-correction-template"
GUIDED_CORRECTION_COMPLETION_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-correction-completion"
GUIDED_CORRECTION_SELECTION_SOURCE = f"{APPROVED_SOURCE_PREFIX}:guided-correction-selection"
WARMUP_FIXTURE_LABEL_SOURCE = "warmup_fixture"
GUIDED_CORRECTION_SOURCES = {
    "segmentation": GUIDED_CORRECTION_SEGMENTATION_SOURCE,
    "plain_reading": GUIDED_CORRECTION_PLAIN_READING_SOURCE,
    "semantic": GUIDED_CORRECTION_SEMANTIC_SOURCE,
    "trace": GUIDED_CORRECTION_TRACE_SOURCE,
    "traces": GUIDED_CORRECTION_TRACE_SOURCE,
    "skeleton": GUIDED_CORRECTION_SKELETON_SOURCE,
    "template": GUIDED_CORRECTION_TEMPLATE_SOURCE,
    "completion": GUIDED_CORRECTION_COMPLETION_SOURCE,
    "consistency": GUIDED_CORRECTION_SELECTION_SOURCE,
    "selection": GUIDED_CORRECTION_SELECTION_SOURCE,
}


def record_approved_pre_ir_examples(
    retriever: KeywordRetriever,
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    full_pipeline_approved: bool,
) -> list[str]:
    """Write reviewed pre-IR examples only after the complete train gate passes."""

    segmentation = semantic.surface_segmentation
    reading = semantic.plain_reading_review
    reading_readiness = (
        assess_plain_reading_ir_readiness(reading, segmentation)
        if reading is not None and segmentation is not None
        else None
    )
    if (
        not full_pipeline_approved
        or semantic.data_split != "train"
        or segmentation is None
        or reading is None
        or segmentation.mechanical_validation.status != "passed"
        or segmentation.status not in {"surface_reviewed", "ambiguity_flagged"}
        or segmentation.label_source
        not in {"human", "independent_judge", "warmup_fixture"}
        or reading_readiness is None
        or not reading_readiness.can_enter_semantic_ir
        or reading.downstream_status != "current"
        or reading.reviewer_source
        not in {"human", "independent_judge", "warmup_fixture"}
    ):
        return []
    chunks = [
        KnowledgeChunk(
            chunk_id=_chunk_id("surface-segmentation", requirement_id, segmentation.segmentation_id),
            title="Approved lossless surface segmentation",
            text=json.dumps(
                {
                    "example_type": "surface_segmentation",
                    "requirement": requirement,
                    "artifact": asdict(segmentation),
                    "visible_to": ["segmentation", "plain_reading", "semantic_extraction"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            source=SURFACE_SEGMENTATION_SOURCE,
        ),
        KnowledgeChunk(
            chunk_id=_chunk_id("plain-reading", requirement_id, reading.reading_review_id),
            title="Approved ordinary-language reading review",
            text=json.dumps(
                {
                    "example_type": "plain_reading",
                    "requirement": requirement,
                    "artifact": asdict(reading),
                    "visible_to": ["plain_reading", "semantic_extraction"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            source=PLAIN_READING_SOURCE,
        ),
    ]
    extend_retriever_once(retriever, chunks)
    return [chunk.chunk_id for chunk in chunks]


def build_faithful_learning_example(
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    trace_plan: TraceObligationPlan,
    traces: list[LabeledTrace],
    skeleton_plan: ConstrainedSkeletonPlan,
    skeletons: list,
    hole_plan: CompletionHolePlan,
    approved_completion: str,
    final_formula: str,
    ir_coverage: IRCoverageReport,
    candidate_status: str,
    run_origin: str,
    approval_source: str,
    final_approval: FinalApprovalRecord | None = None,
) -> FaithfulLearningExample:
    if semantic.faithful_ir is None:
        raise ValueError("FaithfulSemanticIR is required for a learning example")
    if (
        candidate_status != "final"
        or not ir_coverage.complete
        or run_origin == "feedback_rerun"
        or semantic.label_source == "formula_oracle"
        or any(trace.label_source == "formula_oracle" for trace in traces)
        or final_approval is None
        or not final_approval.writeback_allowed
    ):
        raise ValueError("Only independently approved, coverage-complete final samples persist")
    _assert_learning_formula_closure(
        semantic,
        parse_formula(final_formula),
        final_approval=final_approval,
    )
    return FaithfulLearningExample(
        requirement_id=requirement_id,
        natural_language=requirement,
        approved_ir=semantic.faithful_ir,
        trace_obligation_plan=trace_plan,
        approved_traces=tuple(_trace_to_dict(trace) for trace in traces),
        skeleton_plan=skeleton_plan,
        approved_skeletons=tuple(str(item) for item in skeletons),
        completion_hole_plan=hole_plan,
        approved_completion=approved_completion,
        final_formula=final_formula,
        approval_metadata={
            "status": "final",
            "approval_source": approval_source,
            "approval_id": _chunk_id(
                "approval",
                requirement_id,
                semantic.faithful_ir.ir_version,
                final_formula,
            ),
            "approved_ir_version": semantic.faithful_ir.ir_version,
            "approved_formula": final_formula,
            "ir_coverage_status": ir_coverage.status,
            "run_origin": run_origin,
            "writeback_allowed": (
                candidate_status == "final"
                and ir_coverage.complete
                and run_origin != "feedback_rerun"
                and final_approval.writeback_allowed
            ),
            "final_approval_record": asdict(final_approval),
        },
        surface_segmentation=semantic.surface_segmentation,
        plain_reading_review=semantic.plain_reading_review,
    )


def build_warmup_fixture_learning_example(
    *,
    context: GuidedLearningContext,
    semantic: SemanticStructure,
    trace_plan: TraceObligationPlan,
    traces: list[LabeledTrace],
    skeleton_plan: ConstrainedSkeletonPlan,
    skeletons: list,
    hole_plan: CompletionHolePlan,
    approved_completion: str,
    final_formula: str,
    ir_coverage: IRCoverageReport,
    candidate_status: str,
    run_origin: str,
    final_approval: FinalApprovalRecord | None = None,
) -> FaithfulLearningExample:
    """Build a current-contract example from curated warmup artifacts.

    Ordinary runtime examples must pass independent approval; guided warmup
    fixtures instead persist the reviewed expected artifacts plus the recorded
    fallback decisions.  Sanitize provenance so formula-oracle trace labeling
    cannot leak into reusable memory.
    """

    if semantic.faithful_ir is None:
        raise ValueError("FaithfulSemanticIR is required for a warmup fixture")
    if final_approval is None or not final_approval.writeback_allowed:
        raise ValueError("FinalApprovalRecord is required for warmup fixture writeback")
    _assert_learning_formula_closure(
        semantic,
        parse_formula(final_formula),
        final_approval=final_approval,
    )
    fixture_semantic = replace(
        semantic,
        label_source=WARMUP_FIXTURE_LABEL_SOURCE,
        data_split="train",
        formula_source="none",
        formula_visibility=(),
    )
    fixture_traces = [
        replace(
            trace,
            label_source=WARMUP_FIXTURE_LABEL_SOURCE,
            data_split="train",
            formula_source="none",
            formula_visibility=(),
            annotation_locked=True,
        )
        for trace in traces
    ]
    return FaithfulLearningExample(
        requirement_id=(
            context.requirement_id or fixture_semantic.faithful_ir.requirement_id
        ),
        natural_language=context.requirement,
        approved_ir=fixture_semantic.faithful_ir,
        trace_obligation_plan=trace_plan,
        approved_traces=tuple(_trace_to_dict(trace) for trace in fixture_traces),
        skeleton_plan=skeleton_plan,
        approved_skeletons=tuple(str(item) for item in skeletons),
        completion_hole_plan=hole_plan,
        approved_completion=approved_completion,
        final_formula=final_formula,
        approval_metadata={
            "status": "final",
            "approval_source": WARMUP_FIXTURE_LABEL_SOURCE,
            "learning_mode": WARMUP_LEARNING_MODE,
            "fixture_guided": True,
            "approval_id": _chunk_id(
                "warmup-fixture-approval",
                context.requirement_id,
                fixture_semantic.faithful_ir.ir_version,
                final_formula,
            ),
            "approved_ir_version": fixture_semantic.faithful_ir.ir_version,
            "approved_formula": context.approved_formula or final_formula,
            "expected_final_candidate": (
                context.expected_final_candidate or context.approved_formula or final_formula
            ),
            "ir_coverage_status": ir_coverage.status,
            "candidate_status": candidate_status,
            "run_origin": run_origin,
            "writeback_allowed": (
                candidate_status == "final"
                and ir_coverage.complete
                and run_origin != "feedback_rerun"
                and final_approval.writeback_allowed
            ),
            "final_approval_record": asdict(final_approval),
        },
        surface_segmentation=fixture_semantic.surface_segmentation,
        plain_reading_review=fixture_semantic.plain_reading_review,
    )


def record_approved_review_examples(
    retriever: KeywordRetriever,
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    round_index: int,
    comments: str = "",
    ir_coverage: IRCoverageReport | None = None,
    candidate_status: str = "final",
    run_origin: str = "normal",
    final_approval: FinalApprovalRecord | None = None,
) -> list[str]:
    """把人工通过的语义解析和有限轨迹作为正确样例写入 RAG。"""

    if not semantic.persistent_rag_allowed:
        return []
    if final_approval is None or not final_approval.writeback_allowed:
        return []
    if (
        semantic.faithful_ir is None
        or semantic.faithful_ir.has_high_risk_ambiguity
        or candidate_status != "final"
        or run_origin == "feedback_rerun"
        or ir_coverage is None
        or not ir_coverage.complete
        or final_approval.ir_coverage_status != "complete"
    ):
        return []
    if semantic.label_source == "formula_oracle" or any(
        trace.label_source == "formula_oracle" for trace in traces
    ):
        return []
    if any(trace.query_role for trace in traces) and any(
        trace.data_split != "train" for trace in traces
    ):
        return []
    try:
        _assert_learning_formula_closure(
            semantic,
            parse_formula(final_approval.approved_formula),
            final_approval=final_approval,
        )
    except ValueError:
        return []

    legacy_profile = (
        analyze_obligations(requirement, semantic)
        if not any(trace.query_role for trace in traces)
        else None
    )
    if legacy_profile is not None:
        traces = enrich_trace_semantics(traces, legacy_profile)
    plan = plan_semantic_obligations(semantic)
    chunks = [
        _build_semantic_example_chunk(
            requirement_id=requirement_id,
            requirement=requirement,
            semantic=semantic,
            round_index=round_index,
            comments=comments,
            plan=plan,
            profile=legacy_profile,
        ),
        _build_trace_example_chunk(
            requirement_id=requirement_id,
            requirement=requirement,
            semantic=semantic,
            traces=traces,
            round_index=round_index,
            comments=comments,
            plan=plan,
            profile=legacy_profile,
        ),
    ]
    if semantic.reference_only:
        chunks.append(
            _build_temporal_domain_example_chunk(
                requirement_id=requirement_id,
                requirement=requirement,
                semantic=semantic,
                round_index=round_index,
                comments=comments,
            )
        )
    extend_retriever_once(retriever, chunks)
    return [chunk.chunk_id for chunk in chunks]


def record_approved_template_examples(
    retriever: KeywordRetriever,
    *,
    requirement: str,
    semantic: SemanticStructure,
    hole_candidates: list[HoleCandidate],
    approved_decisions: list[CompletionDecision],
    round_index: int,
    ir_coverage: IRCoverageReport | None = None,
    candidate_status: str = "final",
    run_origin: str = "normal",
    final_approval: FinalApprovalRecord | None = None,
) -> list[str]:
    """把已通过严格过滤的带孔模板和补全决策写入模板 RAG。"""

    if not semantic.persistent_rag_allowed:
        return []
    if final_approval is None or not final_approval.writeback_allowed:
        return []
    if (
        semantic.faithful_ir is None
        or candidate_status != "final"
        or run_origin == "feedback_rerun"
        or ir_coverage is None
        or not ir_coverage.complete
        or final_approval.ir_coverage_status != "complete"
    ):
        return []

    if not approved_decisions:
        return []
    for decision in approved_decisions:
        try:
            _assert_learning_formula_closure(
                semantic,
                decision.formula,
                final_approval=final_approval,
            )
        except ValueError:
            return []

    best_decision = sorted(
        approved_decisions,
        key=lambda item: (-item.score, item.formula.size(), str(item.formula)),
    )[0]
    approved_by_candidate_id = {
        decision.candidate_id: decision
        for decision in approved_decisions
        if decision.candidate_id
    }
    chunks: list[KnowledgeChunk] = []
    seen_templates: set[str] = set()
    for index, candidate in enumerate(hole_candidates, start=1):
        if candidate.template_text in seen_templates:
            continue
        decision = approved_by_candidate_id.get(candidate.candidate_id)
        if decision is None:
            if candidate.candidate_id:
                continue
            decision = best_decision
        seen_templates.add(candidate.template_text)
        chunks.append(
            _build_template_example_chunk(
                requirement=requirement,
                semantic=semantic,
                hole_candidate=candidate,
                decision=decision,
                round_index=round_index,
                item_index=index,
            )
        )
        chunks.append(
            _build_operator_completion_example_chunk(
                requirement=requirement,
                semantic=semantic,
                hole_candidate=candidate,
                decision=decision,
                round_index=round_index,
                item_index=index,
            )
        )

    extend_retriever_once(retriever, chunks)
    return [chunk.chunk_id for chunk in chunks]


def _assert_learning_formula_closure(
    semantic: SemanticStructure,
    formula: Formula,
    *,
    final_approval: FinalApprovalRecord,
) -> None:
    ir = semantic.faithful_ir
    if ir is None:
        raise ValueError("learning writeback requires FaithfulSemanticIR")
    report = validate_formula_closure(
        formula,
        build_strict_formula_closure_policy(
            ir,
            semantic.approved_semantic_ir,
        ),
        ir_version=ir.ir_version,
    )
    if not report.passed:
        raise ValueError(
            "learning writeback failed FormulaClosurePolicy: "
            + "; ".join(report.reasons)
        )
    if final_approval.approved_ir_version != ir.ir_version:
        raise ValueError("learning writeback approval is bound to a stale IR version")
    if final_approval.approved_formula != str(formula):
        raise ValueError("learning writeback formula differs from FinalApprovalRecord")


def build_approved_full_process_chunks(
    examples: list[tuple[str, str, str]],
) -> list[KnowledgeChunk]:
    """
    Build complete warmup-process chunks from manually curated examples.

    Each tuple is ``(requirement_id, requirement, process_text)``. The chunks are
    returned instead of immediately added so callers can share the same approved
    warmup memory across multiple per-requirement retrievers.
    """

    return [
        _build_full_process_example_chunk(
            requirement_id=requirement_id,
            requirement=requirement,
            process_text=process_text,
        )
        for requirement_id, requirement, process_text in examples
    ]


def build_approved_semantic_trace_chunks(
    examples: list[tuple[str, str, SemanticStructure | None, tuple[LabeledTrace, ...]]],
    *,
    round_index: int = 1,
) -> list[KnowledgeChunk]:
    """Build stage-specific semantic/trace RAG chunks from curated warmup examples."""

    chunks: list[KnowledgeChunk] = []
    for requirement_id, requirement, semantic, traces in examples:
        if semantic is None or not semantic.persistent_rag_allowed:
            continue
        plan = plan_semantic_obligations(semantic)
        validated_traces = list(traces)
        chunks.append(
            _build_semantic_example_chunk(
                requirement_id=requirement_id,
                requirement=requirement,
                semantic=semantic,
                round_index=round_index,
                comments=(
                    "warmup 文件中的已审核语义结构作为样本学习标准参考；"
                    "warmup 运行本身仍需执行 NLU Agent 生成和评审流程。"
                ),
                plan=plan,
            )
        )
        if validated_traces:
            chunks.append(
                _build_trace_example_chunk(
                    requirement_id=requirement_id,
                    requirement=requirement,
                    semantic=semantic,
                    traces=validated_traces,
                    round_index=round_index,
                    comments=(
                        "warmup 文件中的已审核有限轨迹作为样本学习标准参考；"
                        "warmup 运行本身仍需执行 Trace Agent 生成和评审流程。"
                    ),
                    plan=plan,
                )
            )
    return chunks


def build_approved_pattern_chunks(specs: list[PatternSpec]) -> list[KnowledgeChunk]:
    """Build RAG chunks for learned pattern handlers."""

    return [_build_pattern_example_chunk(spec) for spec in specs]


def build_approved_template_chunks_from_pattern_specs(
    specs: list[PatternSpec],
) -> list[KnowledgeChunk]:
    """Build template-stage RAG chunks from curated warmup pattern specs."""

    return [_build_template_example_chunk_from_pattern_spec(spec) for spec in specs]


def build_approved_operator_completion_chunks_from_pattern_specs(
    specs: list[PatternSpec],
) -> list[KnowledgeChunk]:
    """Build operator-completion RAG chunks from curated warmup pattern specs."""

    return [_build_operator_completion_chunk_from_pattern_spec(spec) for spec in specs]


def record_trace_adequacy_issue(
    retriever: KeywordRetriever,
    *,
    requirement: str,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    evaluations: list[FormulaConsistencyEvaluation],
    round_index: int,
    target_trace_count: int,
    comments: str,
    supplemented_traces: list[LabeledTrace] | None = None,
) -> list[str]:
    """
    记录公式无关的轨迹/语义覆盖不足样例。

    这条记忆会被 Trace Agent 检索；相似需求后续可提前生成更多、更有区分度
    的有限轨迹。候选公式、分数和 checker 结果不得进入该记忆。
    """

    if not semantic.persistent_rag_allowed:
        return []
    del evaluations

    chunk_id = _chunk_id(
        "trace-adequacy",
        requirement,
        str(round_index),
        str(target_trace_count),
    )
    status = (
        "insufficient_then_supplemented"
        if supplemented_traces
        else "insufficient_needs_supplement"
    )
    text = "\n".join(
        [
            "example_type: trace_adequacy",
            f"requirement: {requirement}",
            *_semantic_lines(semantic),
            "stage: completion_trace_adequacy",
            f"status: {status}",
            f"initial_trace_count: {len(traces)}",
            f"target_trace_count: {target_trace_count}",
            "problem: 当前 NL/IR 轨迹义务或语义覆盖不足。",
            f"review_comments: {comments}",
            "candidate_information: omitted",
            "existing_traces:",
            *_trace_lines(traces),
            "supplemented_traces:",
            *_trace_lines(supplemented_traces or []),
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    chunk = KnowledgeChunk(
        chunk_id=chunk_id,
        title="Approved trace adequacy supplement example",
        text=text,
        source=TRACE_ADEQUACY_SOURCE,
    )
    extend_retriever_once(retriever, [chunk])
    return [chunk_id]


def record_guided_warmup_process(
    retriever: KeywordRetriever,
    *,
    context: GuidedLearningContext,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    skeletons: list[object],
    hole_candidates: list[HoleCandidate],
    completion_decisions: list[CompletionDecision],
    evaluations: list[FormulaConsistencyEvaluation],
    selected: FormulaConsistencyEvaluation | None,
    round_index: int = 1,
    guided_decisions: list[dict] | None = None,
) -> list[str]:
    """Persist the validated two-attempt warmup process for later samples."""

    if not semantic.persistent_rag_allowed or not context.enabled:
        return []

    profile = analyze_obligations(context.requirement, semantic)
    stage_metadata = _guided_stage_metadata_by_stage(guided_decisions or [])
    chunks = [
        _build_guided_semantic_chunk(
            context=context,
            semantic=semantic,
            round_index=round_index,
            profile=profile,
            stage_metadata=stage_metadata,
        ),
        _build_guided_trace_chunk(
            context=context,
            semantic=semantic,
            traces=traces,
            round_index=round_index,
            profile=profile,
            stage_metadata=stage_metadata,
        ),
        _build_guided_skeleton_chunk(
            context=context,
            semantic=semantic,
            skeletons=skeletons,
            round_index=round_index,
            stage_metadata=stage_metadata,
        ),
        _build_guided_template_chunk(
            context=context,
            semantic=semantic,
            hole_candidates=hole_candidates,
            round_index=round_index,
            stage_metadata=stage_metadata,
        ),
        _build_guided_completion_chunk(
            context=context,
            semantic=semantic,
            decisions=completion_decisions,
            round_index=round_index,
            stage_metadata=stage_metadata,
        ),
        _build_guided_consistency_chunk(
            context=context,
            semantic=semantic,
            evaluations=evaluations,
            round_index=round_index,
            stage_metadata=stage_metadata,
        ),
        _build_guided_selection_chunk(
            context=context,
            semantic=semantic,
            evaluations=evaluations,
            selected=selected,
            round_index=round_index,
            stage_metadata=stage_metadata,
        ),
        _build_guided_full_process_chunk(
            context=context,
            semantic=semantic,
            traces=traces,
            selected=selected,
            round_index=round_index,
            stage_metadata=stage_metadata,
        ),
    ]
    return extend_retriever_once(retriever, chunks)


def build_guided_correction_chunks_from_review_records(
    review_records: list[dict],
) -> list[KnowledgeChunk]:
    """Build persistent correction chunks from guided fallback decisions."""

    chunks: list[KnowledgeChunk] = []
    seen: set[str] = set()
    for decision in _iter_guided_decisions(review_records):
        if not decision.get("fallback_used"):
            continue
        correction = decision.get("guided_correction")
        if not isinstance(correction, dict):
            continue
        stage = _normalize_guided_stage(str(correction.get("stage", "")))
        source = GUIDED_CORRECTION_SOURCES.get(stage)
        if source is None:
            continue
        chunk = _build_guided_correction_chunk(decision, correction, source)
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        chunks.append(chunk)
    return chunks


def collect_approved_example_chunks(
    retrievers: list[KeywordRetriever],
) -> list[KnowledgeChunk]:
    """从一批 retriever 中收集去重后的正确样例 chunk。"""

    chunks: list[KnowledgeChunk] = []
    seen: set[str] = set()
    for retriever in retrievers:
        for chunk in retriever.chunks:
            if not chunk.source.startswith(APPROVED_SOURCE_PREFIX):
                continue
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks


def extend_retriever_once(
    retriever: KeywordRetriever,
    chunks: list[KnowledgeChunk],
) -> list[str]:
    """把 chunk 加入 retriever，跳过已存在的 chunk_id。"""

    existing_ids = {chunk.chunk_id for chunk in retriever.chunks}
    new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
    retriever.extend_chunks(new_chunks)
    return [chunk.chunk_id for chunk in new_chunks]


def _build_semantic_example_chunk(
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    round_index: int,
    comments: str,
    profile=None,
    plan=None,
) -> KnowledgeChunk:
    plan = plan or plan_semantic_obligations(semantic)
    semantic_plan_lines = (
        _obligation_lines(profile)
        if profile is not None
        else _obligation_plan_lines(plan)
    )
    text = "\n".join(
        [
            "example_type: semantic",
            f"requirement_id: {requirement_id}",
            f"requirement: {requirement}",
            *_semantic_lines(semantic),
            *semantic_plan_lines,
            "approved_process:",
            "- 人工确认 AP、结构化 components 和作用域证据正确。",
            (
                "- 使用统一义务编译器生成可执行语义，并与有限轨迹共同审核。"
                if profile is not None
                else "- 上游阶段只保存歧义轴和查询角色，不保存公式答案。"
            ),
            f"review_comments: {comments or 'approved'}",
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
            "semantic_json:",
            json.dumps(
                _semantic_example_json_payload(semantic),
                ensure_ascii=False,
                indent=2,
            ),
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("semantic", requirement_id, requirement, str(round_index)),
        title="Approved semantic parsing example",
        text=text,
        source=(FAITHFUL_IR_SOURCE if semantic.faithful_ir is not None else SEMANTIC_SOURCE),
    )


def _build_trace_example_chunk(
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    round_index: int,
    comments: str,
    profile=None,
    plan=None,
) -> KnowledgeChunk:
    plan = plan or plan_semantic_obligations(semantic)
    legacy = profile is not None
    trace_plan_lines = (
        _obligation_lines(profile)
        if legacy
        else _obligation_plan_lines(plan)
    )
    text = "\n".join(
        [
            "example_type: trace",
            f"requirement_id: {requirement_id}",
            f"requirement: {requirement}",
            *_semantic_lines(semantic),
            *trace_plan_lines,
            f"trace_count: {len(traces)}",
            "approved_process:",
            (
                "- 每条 expected 标签均通过统一义务公式的有限轨迹求值审核。"
                if legacy
                else "- Trace Agent 仅学习查询角色和轨迹形状。"
            ),
            (
                "- 轨迹角色、区分目标和最小扰动用于指导后续主体需求生成。"
                if legacy
                else "- 逐轨迹标签由独立评审产生，不作为新查询的自动真值。"
            ),
            f"review_comments: {comments or 'approved'}",
            f"round: {round_index}",
            "traces:",
            *(_trace_lines(traces) if legacy else _trace_query_shape_lines(traces)),
            "trace_json:",
            json.dumps(
                [
                    (
                        _trace_to_dict(trace)
                        if legacy
                        else _trace_query_shape_to_dict(trace)
                    )
                    for trace in traces
                ],
                ensure_ascii=False,
                indent=2,
            ),
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("trace", requirement_id, requirement, str(round_index)),
        title="Approved finite trace example",
        text=text,
        source=TRACE_SOURCE,
    )


def _build_temporal_domain_example_chunk(
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    round_index: int,
    comments: str,
) -> KnowledgeChunk:
    text = "\n".join(
        [
            "example_type: temporal_domain",
            f"requirement_id: {requirement_id}",
            f"original_requirement: {requirement}",
            f"trace_domain: {semantic.trace_domain}",
            f"reference_only: {semantic.reference_only}",
            "domain_evidence:",
            *[f"- {item}" for item in semantic.domain_evidence],
            f"finite_rewrite: {semantic.finite_rewrite}",
            f"warning: {semantic.domain_warning}",
            *_semantic_lines(semantic),
            f"review_comments: {comments or 'approved'}",
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("temporal-domain", requirement_id, requirement, str(round_index)),
        title="Approved temporal-domain rewrite example",
        text=text,
        source=TEMPORAL_DOMAIN_SOURCE,
    )


def _build_template_example_chunk(
    *,
    requirement: str,
    semantic: SemanticStructure,
    hole_candidate: HoleCandidate,
    decision: CompletionDecision,
    round_index: int,
    item_index: int,
) -> KnowledgeChunk:
    template_text = normalize_formula_text(hole_candidate.template_text)
    hole_lines = []
    for hole in hole_candidate.holes:
        hole_lines.append(
            "- "
            f"hole_id={hole.hole_id}; "
            f"target={hole.target}; "
            f"allowed_ops={','.join(hole.allowed_ops)}; "
            f"role={hole.role}; "
            f"parent_hole_id={hole.parent_hole_id}; "
            f"source_hint={','.join(hole.source_hint)}; "
            f"reason={hole.reason}"
        )

    assignment_text = ", ".join(
        f"{hole_id}={operator}" for hole_id, operator in decision.assignments
    )
    text = "\n".join(
        [
            "example_type: template",
            f"requirement: {requirement}",
            *_semantic_lines(semantic),
            f"candidate_id: {hole_candidate.candidate_id}",
            f"skeleton: {hole_candidate.skeleton}",
            f"template_text: {template_text}",
            "holes:",
            *hole_lines,
            f"approved_formula: {normalize_formula_text(str(decision.formula))}",
            f"approved_assignments: {assignment_text}",
            f"PosSat: {decision.pos_sat:.2f}",
            f"NegRej: {decision.neg_rej:.2f}",
            f"BoundMatch: {decision.boundary_match:.2f}",
            f"Score: {decision.score:.3f}",
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id(
            "template",
            requirement,
            template_text,
            str(item_index),
        ),
        title="Approved hole template example",
        text=text,
        source=TEMPLATE_SOURCE,
    )


def _build_operator_completion_example_chunk(
    *,
    requirement: str,
    semantic: SemanticStructure,
    hole_candidate: HoleCandidate,
    decision: CompletionDecision,
    round_index: int,
    item_index: int,
) -> KnowledgeChunk:
    template_text = normalize_formula_text(hole_candidate.template_text)
    assignment_text = ", ".join(
        f"{hole_id}={operator}" for hole_id, operator in decision.assignments
    ) or "none"
    text = "\n".join(
        [
            "example_type: operator_completion",
            f"requirement: {requirement}",
            *_semantic_lines(semantic),
            f"candidate_id: {hole_candidate.candidate_id}",
            f"template_text: {template_text}",
            f"approved_assignments: {assignment_text}",
            "trace_use_policy: positive, negative, and boundary traces are joint references",
            "operator_derivation:",
            *([f"- {item}" for item in decision.operator_derivation] or ["- none"]),
            "trace_derivation:",
            *([f"- {item}" for item in decision.trace_derivation] or ["- none"]),
            f"reason: {decision.reason}",
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id(
            "operator-completion",
            requirement,
            template_text,
            str(item_index),
        ),
        title="Approved temporal-hole completion example",
        text=text,
        source=OPERATOR_COMPLETION_SOURCE,
    )


def _build_template_example_chunk_from_pattern_spec(spec: PatternSpec) -> KnowledgeChunk:
    derivation = derive_pattern_spec_template_derivations(spec)[0]
    template_text = normalize_formula_text(derivation.template_text)
    final_candidate = normalize_formula_text(spec.final_candidate_text or spec.formula_text)
    semantic_lines = [
        f"{key}: {value}" for key, value in spec.semantic_fields
    ]
    assignment_text = ", ".join(
        f"{hole_id}={operator}" for hole_id, operator in derivation.assignments
    ) or "none"
    text = "\n".join(
        [
            "example_type: template",
            f"requirement_id: {spec.source_id}",
            f"requirement: {spec.requirement}",
            f"pattern: {spec.pattern}",
            *semantic_lines,
            f"skeleton: {template_text}",
            f"template_text: {template_text}",
            "holes:",
            *(_temporal_hole_lines(derivation.holes) or ["- none"]),
            f"approved_formula: {final_candidate}",
            f"approved_assignments: {assignment_text}",
            f"trace_count: {spec.trace_count}",
            "derivation_rules:",
            *([f"- {item}" for item in spec.derivation_rules] or ["- none"]),
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id(
            "template",
            spec.requirement,
            template_text,
            spec.source_id,
        ),
        title="Approved hole template example",
        text=text,
        source=TEMPLATE_SOURCE,
    )


def _build_operator_completion_chunk_from_pattern_spec(spec: PatternSpec) -> KnowledgeChunk:
    derivation = derive_pattern_spec_template_derivations(spec)[0]
    template_text = normalize_formula_text(derivation.template_text)
    final_candidate = normalize_formula_text(spec.final_candidate_text or spec.formula_text)
    semantic_lines = [
        f"{key}: {value}" for key, value in spec.semantic_fields
    ]
    assignment_text = ", ".join(
        f"{hole_id}={operator}" for hole_id, operator in derivation.assignments
    ) or "none"
    text = "\n".join(
        [
            "example_type: operator_completion",
            f"requirement_id: {spec.source_id}",
            f"requirement: {spec.requirement}",
            f"pattern: {spec.pattern}",
            *semantic_lines,
            f"template_text: {template_text}",
            f"approved_formula: {final_candidate}",
            f"approved_assignments: {assignment_text}",
            "trace_use_policy: positive, negative, and boundary traces are joint references",
            "operator_derivation:",
            *([f"- {item}" for item in derivation.operator_derivation] or ["- none"]),
            "trace_derivation:",
            "- warmup positive, negative, and boundary traces are retained as joint references during sample learning",
            "derivation_rules:",
            *([f"- {item}" for item in spec.derivation_rules] or ["- none"]),
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id(
            "operator-completion",
            spec.requirement,
            template_text,
            spec.source_id,
        ),
        title="Approved temporal-hole completion example",
        text=text,
        source=OPERATOR_COMPLETION_SOURCE,
    )


def _temporal_hole_lines(holes: tuple[object, ...]) -> list[str]:
    lines: list[str] = []
    for hole in holes:
        hole_id = getattr(hole, "hole_id", "")
        target = getattr(hole, "target", "formula fragment")
        allowed_ops = getattr(hole, "allowed_ops", ())
        reason = getattr(hole, "reason", "")
        if not hole_id:
            continue
        lines.append(
            "- "
            f"hole_id={hole_id}; "
            f"target={target}; "
            f"allowed_ops={','.join(allowed_ops)}; "
            f"reason={reason}"
        )
    return lines


def _build_full_process_example_chunk(
    *,
    requirement_id: str,
    requirement: str,
    process_text: str,
) -> KnowledgeChunk:
    text = "\n".join(
        [
            "example_type: full_process",
            f"requirement_id: {requirement_id}",
            f"requirement: {requirement}",
            "approved_process:",
            normalize_boolean_operator_text(process_text.strip()),
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("full-process", requirement_id, requirement),
        title="Approved full Traj-LTLf warmup process",
        text=text,
        source=FULL_PROCESS_SOURCE,
    )


def _build_pattern_example_chunk(spec: PatternSpec) -> KnowledgeChunk:
    semantic_lines = [
        f"- {key}: {value}" for key, value in spec.semantic_fields
    ] or ["- none"]
    derivation_lines = [f"- {item}" for item in spec.derivation_rules] or ["- none"]
    skeleton_lines = [
        f"- {normalize_formula_text(item)}" for item in spec.skeleton_templates
    ] or ["- none"]
    formula_text = normalize_formula_text(spec.formula_text)
    final_candidate = normalize_formula_text(spec.final_candidate_text or spec.formula_text)
    hole_template = normalize_formula_text(spec.hole_template_text or spec.formula_text)
    text = "\n".join(
        [
            "example_type: learned_pattern",
            f"source_warmup_id: {spec.source_id}",
            f"pattern: {spec.pattern}",
            f"requirement: {spec.requirement}",
            f"approved_formula: {formula_text}",
            f"final_candidate: {final_candidate}",
            f"trace_count: {spec.trace_count}",
            "semantic_fields:",
            *semantic_lines,
            "derivation_rules:",
            *derivation_lines,
            "skeleton_templates:",
            *skeleton_lines,
            f"hole_template: {hole_template}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("pattern", spec.pattern, spec.source_id),
        title=f"Approved learned pattern: {spec.pattern}",
        text=text,
        source=PATTERN_SOURCE,
    )


def _iter_guided_decisions(review_records: list[dict]) -> list[dict]:
    decisions: list[dict] = []
    for record in review_records:
        decision = record.get("guided_decision")
        if isinstance(decision, dict):
            decisions.append(decision)
        nested = record.get("guided_decisions")
        if isinstance(nested, list):
            decisions.extend(item for item in nested if isinstance(item, dict))
    return decisions


def _guided_stage_metadata_by_stage(
    guided_decisions: list[dict],
) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for decision in guided_decisions:
        stage = _normalize_guided_stage(str(decision.get("stage", "")))
        if not stage:
            continue
        fallback_used = bool(decision.get("fallback_used"))
        existing = metadata.get(stage)
        if existing is not None and existing.get("fallback_used") == "true":
            continue
        metadata[stage] = {
            "artifact_origin": (
                "expected_fallback" if fallback_used else "generated"
            ),
            "fallback_used": str(fallback_used).lower(),
            "verification_status": str(
                decision.get("verification_status", "passed")
            ),
        }
    return metadata


def _learning_metadata_lines(
    stage: str,
    stage_metadata: dict[str, dict[str, str]] | None,
) -> list[str]:
    metadata = (stage_metadata or {}).get(
        _normalize_guided_stage(stage),
        {
            "artifact_origin": "generated",
            "fallback_used": "false",
            "verification_status": "passed",
        },
    )
    return [
        f"learning_mode: {WARMUP_LEARNING_MODE}",
        f"artifact_origin: {metadata['artifact_origin']}",
        f"fallback_used: {metadata['fallback_used']}",
        f"verification_status: {metadata['verification_status']}",
        "data_split: train",
    ]


def _normalize_guided_stage(stage: str) -> str:
    normalized = stage.strip().lower().replace("-", "_")
    if normalized in {"traces", "trace_generation"}:
        return "trace"
    if normalized == "hole_candidate_generation":
        return "template"
    if normalized == "temporal_hole_completion":
        return "completion"
    return normalized


def _build_guided_correction_chunk(
    decision: dict,
    correction: dict,
    source: str,
) -> KnowledgeChunk:
    stage = _normalize_guided_stage(str(correction.get("stage", "")))
    requirement_id = str(decision.get("requirement_id", "")).strip()
    mismatch_evidence = correction.get("mismatch_evidence") or []
    if not isinstance(mismatch_evidence, list):
        mismatch_evidence = [str(mismatch_evidence)]
    chunk_id = _chunk_id(
        f"guided-correction-{stage}",
        requirement_id,
        json.dumps(correction, ensure_ascii=False, sort_keys=True),
    )
    text = "\n".join(
        [
            f"example_type: guided_correction_{stage}",
            f"requirement_id: {requirement_id or 'unknown'}",
            f"stage: {stage}",
            f"learning_mode: {WARMUP_LEARNING_MODE}",
            f"artifact_origin: {decision.get('artifact_origin', 'expected_fallback')}",
            "fallback_used: "
            + ("true" if decision.get("fallback_used") else "false"),
            f"verification_status: {correction.get('verification_status', 'passed')}",
            "data_split: train",
            f"root_cause_category: {correction.get('root_cause_category', '')}",
            "root_cause_explanation:",
            str(correction.get("root_cause_explanation", "")),
            "llm_error_summary:",
            str(decision.get("decision_reason", "")),
            "lesson_for_rag:",
            str(decision.get("lesson_for_rag", "")),
            f"rationale_source: {decision.get('rationale_source', 'local')}",
            "mismatch_evidence:",
            *([f"- {item}" for item in mismatch_evidence] or ["- none"]),
            f"correction_action: {correction.get('correction_action', '')}",
            "round_1_artifact:",
            json.dumps(
                correction.get("round_1_artifact"),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "expected_artifact:",
            json.dumps(
                correction.get("expected_artifact"),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "round_2_artifact:",
            json.dumps(
                correction.get("round_2_artifact"),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=chunk_id,
        title=f"Guided correction {stage}",
        text=text,
        source=source,
    )


def _build_guided_semantic_chunk(
    *,
    context: GuidedLearningContext,
    semantic: SemanticStructure,
    round_index: int,
    profile=None,
    stage_metadata: dict[str, dict[str, str]] | None = None,
) -> KnowledgeChunk:
    expected = context.expected_semantic
    expected_lines = _semantic_lines(expected) if expected is not None else ["none"]
    plan = plan_semantic_obligations(semantic)
    text = "\n".join(
        [
            "example_type: guided_semantic",
            f"requirement_id: {context.requirement_id}",
            f"requirement: {context.requirement}",
            *_learning_metadata_lines("semantic", stage_metadata),
            "guided_policy: autonomous attempt 1, teacher-scoped attempt 2 on mismatch",
            "decision_steps:",
            "- generate attempt 1 from NL, validated upstream artifacts, and prior-sample RAG only",
            "- compare canonically with the sealed expected semantic artifact",
            "- expose only semantic teacher fields for one second attempt when needed",
            "expected_semantic:",
            *expected_lines,
            "generated_semantic:",
            *_semantic_lines(semantic),
            "obligation_plan:",
            *_obligation_plan_lines(plan),
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("guided-semantic", context.requirement_id, context.requirement),
        title="Guided semantic warmup decision",
        text=text,
        source=GUIDED_SEMANTIC_SOURCE,
    )


def _build_guided_trace_chunk(
    *,
    context: GuidedLearningContext,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    round_index: int,
    profile=None,
    stage_metadata: dict[str, dict[str, str]] | None = None,
) -> KnowledgeChunk:
    plan = plan_semantic_obligations(semantic)
    text = "\n".join(
        [
            "example_type: guided_trace",
            f"requirement_id: {context.requirement_id}",
            f"requirement: {context.requirement}",
            *_learning_metadata_lines("trace", stage_metadata),
            *_semantic_lines(semantic),
            *_obligation_plan_lines(plan),
            "guided_policy: stage-scoped trace-query shape learning",
            "decision_steps:",
            "- learn only query roles and finite-trace shapes",
            "- do not expose approved formulas or formula-oracle labels",
            "- keep adjudication independent from query generation",
            "expected_query_shapes:",
            *_trace_query_shape_lines(list(context.expected_traces)),
            "generated_query_shapes:",
            *_trace_query_shape_lines(traces),
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("guided-trace", context.requirement_id, context.requirement),
        title="Guided finite-trace warmup decision",
        text=text,
        source=GUIDED_TRACE_SOURCE,
    )


def _build_guided_skeleton_chunk(
    *,
    context: GuidedLearningContext,
    semantic: SemanticStructure,
    skeletons: list[object],
    round_index: int,
    stage_metadata: dict[str, dict[str, str]] | None = None,
) -> KnowledgeChunk:
    text = "\n".join(
        [
            "example_type: guided_skeleton",
            f"requirement_id: {context.requirement_id}",
            f"requirement: {context.requirement}",
            *_learning_metadata_lines("skeleton", stage_metadata),
            *_semantic_lines(semantic),
            "guided_policy: autonomous attempt 1, teacher-scoped attempt 2 on mismatch",
            "decision_steps:",
            "- expected skeleton/formula shape guides which Boolean structure is NL-supported",
            "- restricted skeleton generation must still avoid concrete X/F/G unless the learned pattern allows a formula-text path",
            "expected_restricted_skeletons:",
            *([f"- {item}" for item in context.expected_restricted_skeletons] or ["- none"]),
            "generated_skeletons:",
            *([f"- {item}" for item in skeletons] or ["- none"]),
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("guided-skeleton", context.requirement_id, context.requirement),
        title="Guided skeleton warmup decision",
        text=text,
        source=GUIDED_SKELETON_SOURCE,
    )


def _build_guided_template_chunk(
    *,
    context: GuidedLearningContext,
    semantic: SemanticStructure,
    hole_candidates: list[HoleCandidate],
    round_index: int,
    stage_metadata: dict[str, dict[str, str]] | None = None,
) -> KnowledgeChunk:
    generated = []
    for candidate in hole_candidates:
        hole_text = ", ".join(
            f"{hole.hole_id} in {{{','.join(hole.allowed_ops)}}}"
            for hole in candidate.holes
        ) or "no holes"
        generated.append(f"- {candidate.template_text}; {hole_text}")
    text = "\n".join(
        [
            "example_type: guided_template",
            f"requirement_id: {context.requirement_id}",
            f"requirement: {context.requirement}",
            *_learning_metadata_lines("template", stage_metadata),
            *_semantic_lines(semantic),
            "guided_policy: autonomous/deterministic attempt 1 before expected comparison",
            f"expected_hole_template: {context.expected_hole_template or 'none'}",
            "decision_steps:",
            "- use the expected template to decide where temporal operators should remain as H_xxx holes",
            "- do not choose final operators during the template stage",
            "generated_hole_templates:",
            *(generated or ["- none"]),
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("guided-template", context.requirement_id, context.requirement),
        title="Guided hole-template warmup decision",
        text=text,
        source=GUIDED_TEMPLATE_SOURCE,
    )


def _build_guided_completion_chunk(
    *,
    context: GuidedLearningContext,
    semantic: SemanticStructure,
    decisions: list[CompletionDecision],
    round_index: int,
    stage_metadata: dict[str, dict[str, str]] | None = None,
) -> KnowledgeChunk:
    assignment_text = ", ".join(
        f"{hole_id}={operator}" for hole_id, operator in context.approved_assignments
    ) or "none"
    generated = [
        f"- {decision.formula}; assignments="
        + (
            ", ".join(f"{hole}={operator}" for hole, operator in decision.assignments)
            or "none"
        )
        for decision in decisions
    ]
    text = "\n".join(
        [
            "example_type: guided_completion",
            f"requirement_id: {context.requirement_id}",
            f"requirement: {context.requirement}",
            *_learning_metadata_lines("completion", stage_metadata),
            *_semantic_lines(semantic),
            "guided_policy: trajectory-derived candidates first; teacher correction only after mismatch",
            f"approved_formula: {context.approved_formula}",
            f"approved_assignments: {assignment_text}",
            "decision_steps:",
            "- expected completion guides each H_xxx operator choice in the first generation",
            "- positive, negative, and boundary traces remain joint evidence for the choice",
            "generated_completions:",
            *(generated or ["- none"]),
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("guided-completion", context.requirement_id, context.requirement),
        title="Guided completion warmup decision",
        text=text,
        source=GUIDED_COMPLETION_SOURCE,
    )


def _build_guided_consistency_chunk(
    *,
    context: GuidedLearningContext,
    semantic: SemanticStructure,
    evaluations: list[FormulaConsistencyEvaluation],
    round_index: int,
    stage_metadata: dict[str, dict[str, str]] | None = None,
) -> KnowledgeChunk:
    lines = [
        f"- formula: {evaluation.formula}; PosSat={evaluation.pos_sat:.2f}; "
        f"NegRej={evaluation.neg_rej:.2f}; BoundMatch={evaluation.bound_match:.2f}; "
        f"Score={evaluation.score:.3f}"
        for evaluation in evaluations
    ]
    text = "\n".join(
        [
            "example_type: guided_consistency",
            f"requirement_id: {context.requirement_id}",
            f"requirement: {context.requirement}",
            *_learning_metadata_lines("consistency", stage_metadata),
            *_semantic_lines(semantic),
            "guided_policy: deterministic consistency audit of guided warmup candidates",
            f"expected_final_candidate: {context.expected_final_candidate or context.approved_formula}",
            "decision_steps:",
            "- evaluate generated candidates with the finite-trace checker, not LLM judgment",
            "- use mismatches only as fail-fast diagnostics for warmup, not as another generation round",
            "evaluations:",
            *(lines or ["- none"]),
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("guided-consistency", context.requirement_id, context.requirement),
        title="Guided consistency warmup audit",
        text=text,
        source=GUIDED_CONSISTENCY_SOURCE,
    )


def _build_guided_selection_chunk(
    *,
    context: GuidedLearningContext,
    semantic: SemanticStructure,
    evaluations: list[FormulaConsistencyEvaluation],
    selected: FormulaConsistencyEvaluation | None,
    round_index: int,
    stage_metadata: dict[str, dict[str, str]] | None = None,
) -> KnowledgeChunk:
    ranked = sorted(evaluations, key=lambda item: (-item.score, item.formula.size(), str(item.formula)))
    ranked_lines = [
        f"- {item.formula}; score={item.score:.3f}; PosSat={item.pos_sat:.2f}; "
        f"NegRej={item.neg_rej:.2f}; BoundMatch={item.bound_match:.2f}"
        for item in ranked
    ] or ["- none"]
    text = "\n".join(
        [
            "example_type: guided_selection",
            f"requirement_id: {context.requirement_id}",
            f"requirement: {context.requirement}",
            *_learning_metadata_lines("selection", stage_metadata),
            *_semantic_lines(semantic),
            "guided_policy: deterministic whole-IR validation before offline expected comparison",
            f"expected_final_candidate: {context.expected_final_candidate or context.approved_formula}",
            f"selected_formula: {selected.formula if selected is not None else 'none'}",
            "decision_steps:",
            "- select the expected final candidate when it is present and passes deterministic consistency checks",
            "- do not run another warmup generation round solely because an unguided ranking would differ",
            "ranked_candidates:",
            *ranked_lines,
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("guided-selection", context.requirement_id, context.requirement),
        title="Guided final-selection warmup decision",
        text=text,
        source=GUIDED_SELECTION_SOURCE,
    )


def _build_guided_full_process_chunk(
    *,
    context: GuidedLearningContext,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    selected: FormulaConsistencyEvaluation | None,
    round_index: int,
    stage_metadata: dict[str, dict[str, str]] | None = None,
) -> KnowledgeChunk:
    fallback_used = any(
        item.get("fallback_used") == "true"
        for item in (stage_metadata or {}).values()
    )
    text = "\n".join(
        [
            "example_type: guided_full_process",
            f"requirement_id: {context.requirement_id}",
            f"requirement: {context.requirement}",
            f"learning_mode: {WARMUP_LEARNING_MODE}",
            "artifact_origin: expected_fallback" if fallback_used else "artifact_origin: generated",
            f"fallback_used: {str(fallback_used).lower()}",
            "verification_status: passed",
            "data_split: train",
            "guided_policy: all warmup stages use expected artifacts in one generation round",
            *_semantic_lines(semantic),
            f"approved_formula: {context.approved_formula}",
            f"expected_final_candidate: {context.expected_final_candidate or context.approved_formula}",
            f"selected_formula: {selected.formula if selected is not None else 'none'}",
            f"trace_count: {len(traces)}",
            "process_summary:",
            "- semantic, trace, skeleton, template, completion, consistency, and selection stages were learned as guided choices",
            "- main requirements may retrieve this chunk as historical experience but never receive their own expected answer",
            f"round: {round_index}",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    return KnowledgeChunk(
        chunk_id=_chunk_id("guided-full-process", context.requirement_id, context.requirement),
        title="Guided full warmup process",
        text=text,
        source=GUIDED_FULL_PROCESS_SOURCE,
    )


def _semantic_lines(semantic: SemanticStructure) -> list[str]:
    lines = []
    if semantic.faithful_ir is not None:
        lines.extend(
            [
                f"schema_version: {semantic.faithful_ir.schema_version}",
                f"faithful_ir_contract_version: {semantic.faithful_ir.contract_version}",
                f"faithful_ir_validator_version: {semantic.faithful_ir.validator_version}",
                "semantic_representation: faithful_ir",
                f"ir_version: {semantic.faithful_ir.ir_version}",
                f"root_node_type: {semantic.faithful_ir.node_map[semantic.faithful_ir.root_node_id].node_type}",
            ]
        )
    else:
        lines.extend(
            [
                f"pattern: {semantic.pattern}",
                f"trigger: {semantic.trigger}",
                f"response: {semantic.response}",
                f"scope_keyword: {semantic.scope_keyword}",
                f"temporal_keyword: {semantic.temporal_keyword}",
            ]
        )
    if semantic.aps:
        lines.append("aps: " + ", ".join(semantic.aps))
    if semantic.components:
        lines.append("components:")
        lines.extend(f"- {key}: {value}" for key, value in semantic.components)
    if semantic.reference_only:
        lines.extend(
            [
                f"trace_domain: {semantic.trace_domain}",
                f"finite_rewrite: {semantic.finite_rewrite}",
                f"domain_warning: {semantic.domain_warning}",
            ]
        )
    return lines


def _semantic_example_json_payload(semantic: SemanticStructure) -> dict:
    if semantic.faithful_ir is None:
        return asdict(semantic)
    return {
        "faithful_ir": asdict(semantic.faithful_ir),
        "semantic_status": semantic.semantic_status,
    }


def _obligation_lines(profile) -> list[str]:
    return [
        f"obligation_compiler_version: {OBLIGATION_COMPILER_VERSION}",
        "obligation_usage_policy: use as derivation and label reference; NLU must not emit formula_text",
        f"obligation_formula: {profile.formula if profile.formula is not None else 'unavailable'}",
        f"obligation_source: {profile.source or 'unavailable'}",
        f"obligation_scope: {profile.scope}",
        f"temporal_operator: {profile.temporal_operator or 'none'}",
        f"required_horizon: {profile.required_horizon}",
        "required_roles: " + (", ".join(profile.required_roles) or "none"),
        "confusion_targets: " + (", ".join(profile.confusion_targets) or "none"),
        f"executable_semantics: {profile.executable}",
    ]


def _obligation_plan_lines(plan) -> list[str]:
    return [
        f"pattern_family: {plan.pattern_family}",
        "participants: "
        + (
            ", ".join(f"{key}={value}" for key, value in plan.participants)
            or "none"
        ),
        "ambiguity_axes: " + (", ".join(plan.ambiguity_axes) or "none"),
        "required_query_roles: "
        + (", ".join(plan.required_query_roles) or "none"),
        "domain_constraints: "
        + (", ".join(plan.domain_constraints) or "none"),
    ]


def _validated_learning_traces(
    *,
    requirement_id: str,
    traces: list[LabeledTrace],
    profile,
) -> list[LabeledTrace]:
    enriched = enrich_trace_semantics(list(traces), profile)
    _valid, conflicts = validate_trace_labels(enriched, profile)
    if conflicts:
        raise ValueError(
            f"{requirement_id} 学习样本轨迹标签与统一义务不一致: "
            + "；".join(conflicts)
        )

    labels_by_shape: dict[tuple[tuple[str, ...], ...], bool] = {}
    for trace in enriched:
        shape = tuple(tuple(sorted(step)) for step in trace.trace)
        previous = labels_by_shape.get(shape)
        if previous is not None and previous != trace.expected:
            raise ValueError(
                f"{requirement_id} 学习样本存在同形异标签轨迹: {trace.name}"
            )
        labels_by_shape[shape] = trace.expected
    return enriched


def _trace_lines(traces: list[LabeledTrace]) -> list[str]:
    if not traces:
        return ["- none"]
    return [
        f"- {trace.kind} {trace.name}: {format_trace(trace.trace)} "
        f"expected={format_truth(trace.expected)} "
        f"semantic_label={trace.semantics.semantic_label or 'unclassified'} "
        f"roles={','.join(trace.semantics.roles) or 'none'} "
        f"confusion_targets={','.join(trace.semantics.confusion_targets) or 'none'} "
        f"required_horizon={trace.semantics.required_horizon} "
        f"perturbation={trace.semantics.perturbation or 'none'} "
        f"reason={trace.reason}"
        for trace in traces
    ]


def _trace_query_shape_lines(traces: list[LabeledTrace]) -> list[str]:
    if not traces:
        return ["- none"]
    return [
        f"- query_role={trace.query_role or trace.role or trace.kind}; "
        f"trace={format_trace(trace.trace)}; "
        f"target_ambiguity={trace.target_ambiguity or 'unspecified'}"
        for trace in traces
    ]


def _trace_query_shape_to_dict(trace: LabeledTrace) -> dict:
    return {
        "name": trace.name,
        "query_role": trace.query_role or trace.role or trace.kind,
        "target_ambiguity": trace.target_ambiguity,
        "target_ambiguity_axis_ids": list(trace.target_ambiguity_axis_ids),
        "trace": trace_to_serializable(trace.trace),
        "trace_role": trace.trace_role,
        "boundary_role": trace.boundary_role,
        "obligation_ids": list(trace.obligation_ids),
        "covered_ir_node_ids": list(trace.covered_ir_node_ids),
        "ir_version": trace.ir_version,
        "nl_verdict": trace.nl_verdict,
        "review_record_id": trace.review_record_id,
        "data_split": trace.data_split,
        "label_source": trace.label_source,
        "formula_source": "none",
        "formula_visibility": [],
        "allowed_rag_stages": [
            "obligation_planning",
            "trace_query_generation",
        ],
    }


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
        "data_split": trace.data_split,
        "formula_source": trace.formula_source,
        "formula_visibility": list(trace.formula_visibility),
        "annotation_locked": trace.annotation_locked,
        "allowed_rag_stages": list(trace.allowed_rag_stages),
        "role": trace.role,
        "query_role": trace.query_role,
        "target_ambiguity": trace.target_ambiguity,
        "target_ambiguity_axis_ids": list(trace.target_ambiguity_axis_ids),
        "query_constraints": list(trace.query_constraints),
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


def _chunk_id(*parts: str) -> str:
    raw = "\n".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"approved-{parts[0]}-{digest}"
