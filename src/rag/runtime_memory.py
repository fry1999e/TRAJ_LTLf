"""
运行期 RAG 记忆。

这个模块不把内容写入磁盘，而是把本次 pipeline 运行中产生的诊断结果追加到
当前 KeywordRetriever 的知识片段列表里。这样下一轮 Agent 生成时，可以检索
到刚刚失败的骨架、错误类型和轨迹证据。
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json

from data_examples import LabeledTrace
from rag.knowledge_base import KnowledgeChunk
from rag.retriever import KeywordRetriever
from selection_rules import (
    candidate_selection_review_notes as build_candidate_selection_review_notes,
    completion_decision_reject_reasons,
    semantic_guard_reject_reasons,
)
from schemas import (
    CompletionDecision,
    FormulaConsistencyEvaluation,
    SemanticStructure,
    SkeletonDiagnostic,
    StageErrorRecord,
)


def _add_runtime_chunks(
    retriever: KeywordRetriever,
    chunks: list[KnowledgeChunk],
    semantic: SemanticStructure | None,
) -> None:
    if not chunks:
        return
    if semantic is not None and not semantic.persistent_rag_allowed:
        retriever.chunks.extend(chunks)
        return
    retriever.extend_chunks(chunks)


def record_stage_errors(
    retriever: KeywordRetriever,
    requirement: str,
    semantic: SemanticStructure | None,
    stage: str,
    round_index: int,
    errors: list[StageErrorRecord],
) -> list[str]:
    """把某一阶段出现的每一次错误逐条写入当前 RAG。"""

    chunks: list[KnowledgeChunk] = []
    for item_index, error in enumerate(errors, start=1):
        chunk_id = f"runtime-error-{stage}-{round_index}-{item_index}"
        evidence_text = _format_error_evidence(error.evidence)
        timestamp = error.timestamp or datetime.now().isoformat(timespec="seconds")
        text = "\n".join(
            [
                f"requirement: {requirement}",
                *_format_semantic_context(semantic),
                f"stage: {stage}",
                f"round: {round_index}",
                f"error_id: {error.error_id}",
                "status: error",
                f"error_type: {error.error_type}",
                f"source: {error.source}",
                f"timestamp: {timestamp}",
                "artifact:",
                _indent_multiline(error.artifact),
                f"comments: {error.comments or 'none'}",
                "evidence:",
                evidence_text,
                f"suggestion: {error.suggestion or 'review this stage before the next iteration'}",
            ]
        )
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                title=f"Stage error {stage} {error.error_type}",
                text=text,
                source="runtime",
            )
        )

    _add_runtime_chunks(retriever, chunks, semantic)
    return [chunk.chunk_id for chunk in chunks]


def record_trace_length_issue(
    retriever: KeywordRetriever,
    requirement: str,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    round_index: int,
    target_trace_length: int,
    comments: str,
) -> list[str]:
    """把因轨迹长度不合适导致的轨迹评审失败记录到当前 RAG。"""

    chunk_id = f"runtime-trace-length-{round_index}"
    trace_lengths = [len(trace.trace) for trace in traces]
    diagnostic = _trace_length_diagnostic(
        trace_lengths=trace_lengths,
        target_trace_length=target_trace_length,
    )
    evidence = _format_trace_length_evidence(traces)
    text = "\n".join(
        [
            f"requirement: {requirement}",
            *_format_semantic_context(semantic),
            "stage: trace_generation",
            "status: needs_regeneration",
            f"error_type: {diagnostic}",
            f"current_trace_lengths: {trace_lengths}",
            f"target_trace_length: {target_trace_length}",
            f"review_opinion: {comments}",
            "discard_policy: discard unapproved traces and regenerate immediately",
            "trace_length_evidence:",
            evidence,
            (
                "suggestion: next TraceGenerationAgent round must generate every "
                f"finite trace with exactly {target_trace_length} time steps."
            ),
        ]
    )
    chunk = KnowledgeChunk(
        chunk_id=chunk_id,
        title=f"Trace length diagnostic {diagnostic}",
        text=text,
        source="runtime",
    )
    _add_runtime_chunks(retriever, [chunk], semantic)
    return [chunk_id]


def record_guided_stage_decision(
    retriever: KeywordRetriever,
    *,
    requirement: str,
    semantic: SemanticStructure | None,
    decision_record: dict,
) -> list[str]:
    """Record a guided keep/fallback decision as runtime RAG experience."""

    stage = str(decision_record.get("stage", "guided"))
    round_index = str(decision_record.get("round", 1))
    raw_id = json.dumps(
        {
            "requirement": requirement,
            "stage": stage,
            "round": round_index,
            "status": decision_record.get("status"),
            "decision": decision_record.get("decision"),
            "reason": decision_record.get("decision_reason"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:12]
    chunk_id = f"runtime-guided-decision-{stage}-{round_index}-{digest}"
    evidence = decision_record.get("mismatch_evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    text = "\n".join(
        [
            "example_type: guided_stage_decision",
            f"requirement: {requirement}",
            *_format_semantic_context(semantic),
            f"stage: {stage}",
            f"round: {round_index}",
            f"status: {decision_record.get('status')}",
            f"decision: {decision_record.get('decision')}",
            f"decision_reason: {decision_record.get('decision_reason')}",
            f"lesson_for_rag: {decision_record.get('lesson_for_rag')}",
            f"rationale_source: {decision_record.get('rationale_source', 'unknown')}",
            "mismatch_evidence:",
            *([f"- {item}" for item in evidence] or ["- none"]),
            "decision_json:",
            json.dumps(decision_record, ensure_ascii=False, indent=2),
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        ]
    )
    chunk = KnowledgeChunk(
        chunk_id=chunk_id,
        title=f"Guided stage decision {stage} {decision_record.get('status')}",
        text=text,
        source="runtime:guided-decision",
    )
    _add_runtime_chunks(retriever, [chunk], semantic)
    return [chunk_id]


def record_skeleton_diagnostics(
    retriever: KeywordRetriever,
    requirement: str,
    semantic: SemanticStructure,
    diagnostics: list[SkeletonDiagnostic],
    round_index: int,
) -> list[str]:
    """把失败骨架的错误类型和轨迹证据记录到当前 RAG。"""

    chunks: list[KnowledgeChunk] = []
    failed = [diagnostic for diagnostic in diagnostics if not diagnostic.passed]

    for item_index, diagnostic in enumerate(failed, start=1):
        chunk_id = f"runtime-skeleton-{round_index}-{item_index}"
        evidence_text = "\n".join(
            f"- {evidence}" for evidence in diagnostic.trace_evidence
        )
        if not evidence_text:
            evidence_text = "- 未产生具体轨迹证据。"

        text = "\n".join(
            [
                f"requirement: {requirement}",
                *_format_semantic_context(semantic),
                f"skeleton: {diagnostic.skeleton}",
                f"error_type: {diagnostic.error_type}",
                "trace_evidence:",
                evidence_text,
                f"suggestion: {diagnostic.suggestion}",
            ]
        )
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                title=f"Skeleton diagnostic {diagnostic.error_type}",
                text=text,
                source="runtime",
            )
        )

    _add_runtime_chunks(retriever, chunks, semantic)
    return [chunk.chunk_id for chunk in chunks]


def record_completion_needs_review(
    retriever: KeywordRetriever,
    requirement: str,
    semantic: SemanticStructure,
    decisions: list[CompletionDecision],
    round_index: int,
    strict_boundary: bool,
) -> list[str]:
    """把候选补全阶段的 needs_review 评审意见归纳进当前 RAG。"""

    chunk_id = f"runtime-completion-needs-review-{round_index}"
    notes = _completion_review_notes(decisions, strict_boundary, semantic)
    diagnostics = _format_completion_diagnostics(
        decisions=decisions,
        strict_boundary=strict_boundary,
        semantic=semantic,
    )

    text = "\n".join(
        [
            f"requirement: {requirement}",
            *_format_semantic_context(semantic),
            "status: needs_review",
            f"strict_boundary: {strict_boundary}",
            "strict_filter:",
            "- Formula must pass the semantic guard for the requirement pattern",
            "- PosSat must equal 1.0",
            "- NegRej must equal 1.0",
            "- BoundMatch must equal 1.0 when strict_boundary is true",
            "review_opinion:",
            *[f"- {note}" for note in notes],
            "candidate_diagnostics:",
            diagnostics,
        ]
    )
    chunk = KnowledgeChunk(
        chunk_id=chunk_id,
        title="Completion needs_review",
        text=text,
        source="runtime",
    )
    _add_runtime_chunks(retriever, [chunk], semantic)
    return [chunk_id]


def candidate_selection_review_notes(
    evaluations: list[FormulaConsistencyEvaluation],
    strict_boundary: bool,
    semantic: SemanticStructure | None = None,
) -> tuple[str, ...]:
    """根据候选评估结果生成简明评审意见。"""

    return build_candidate_selection_review_notes(
        evaluations=evaluations,
        strict_boundary=strict_boundary,
        semantic=semantic,
    )
    return tuple(notes)


def _format_semantic_context(semantic: SemanticStructure | None) -> list[str]:
    if semantic is None:
        return []
    if semantic.faithful_ir is not None:
        ir = semantic.faithful_ir
        root = ir.node_map.get(ir.root_node_id)
        return [
            f"schema_version: {ir.schema_version}",
            "semantic_representation: faithful_ir",
            f"ir_version: {ir.ir_version}",
            f"root_node_type: {root.node_type if root is not None else 'unknown'}",
            "aps: " + ", ".join(ir.ap_names),
        ]
    return [
        f"pattern: {semantic.pattern}",
        f"trigger: {semantic.trigger}",
        f"response: {semantic.response}",
    ]


def _format_error_evidence(evidence: tuple[str, ...]) -> str:
    if not evidence:
        return "- no evidence"
    return "\n".join(f"- {item}" for item in evidence)


def _indent_multiline(text: str) -> str:
    if not text:
        return "  <empty>"
    return "\n".join(f"  {line}" if line else "" for line in text.splitlines())


def _trace_length_diagnostic(
    trace_lengths: list[int],
    target_trace_length: int,
) -> str:
    if trace_lengths and max(trace_lengths) < target_trace_length:
        return "length_insufficient"
    if trace_lengths and min(trace_lengths) > target_trace_length:
        return "length_excessive"
    return "length_mismatch"


def _format_trace_length_evidence(traces: list[LabeledTrace]) -> str:
    if not traces:
        return "- no traces generated"

    return "\n".join(
        f"- {trace.kind} {trace.name}: length={len(trace.trace)} "
        f"expected_label={trace.expected}"
        for trace in traces
    )


def _completion_review_notes(
    decisions: list[CompletionDecision],
    strict_boundary: bool,
    semantic: SemanticStructure,
) -> tuple[str, ...]:
    if not decisions:
        return (
            "补全 Agent 没有生成任何候选公式，需要重新检查带孔模板和允许算子集合。",
        )

    notes: list[str] = [
        "本轮补全没有候选公式同时通过语义硬过滤和严格轨迹过滤，下一轮应结合这些失败证据重新补全。",
    ]
    if all(semantic_guard_reject_reasons(decision.formula, semantic) for decision in decisions):
        notes.append("所有补全候选都未通过语义硬过滤，需要保留 trigger -> response 触发关系。")
    if max(decision.pos_sat for decision in decisions) < 1.0:
        notes.append("正轨迹满足率不足，候选可能把响应要求设置得过强。")
    if max(decision.neg_rej for decision in decisions) < 1.0:
        notes.append("反轨迹拒绝率不足，候选可能过弱或遗漏触发约束。")
    if strict_boundary and max(decision.boundary_match for decision in decisions) < 1.0:
        notes.append("边界轨迹匹配率不足，优先调整 response 孔位中的 X/F/XF 选择。")

    best = sorted(decisions, key=lambda item: (-item.score, item.formula.size(), str(item.formula)))[0]
    notes.append(
        f"当前最高分参考候选为 {best.formula}，但仍需修正后才能作为最终公式。"
    )
    return tuple(notes)


def _format_completion_diagnostics(
    decisions: list[CompletionDecision],
    strict_boundary: bool,
    semantic: SemanticStructure,
) -> str:
    if not decisions:
        return "- no candidates"

    lines: list[str] = []
    ranked = sorted(decisions, key=lambda item: (-item.score, item.formula.size(), str(item.formula)))
    for decision in ranked[:5]:
        reject_reasons = completion_decision_reject_reasons(
            decision=decision,
            strict_boundary=strict_boundary,
            semantic=semantic,
        )
        lines.append(
            f"- formula: {decision.formula}; "
            f"PosSat={decision.pos_sat:.2f}; "
            f"NegRej={decision.neg_rej:.2f}; "
            f"BoundMatch={decision.boundary_match:.2f}; "
            f"Score={decision.score:.3f}"
        )
        if reject_reasons:
            lines.append("  reject_reasons: " + "；".join(reject_reasons[:3]))
        for derivation in decision.operator_derivation[:2]:
            lines.append(f"  operator_derivation: {derivation}")
        for derivation in decision.trace_derivation[:2]:
            lines.append(f"  trace_derivation: {derivation}")
    return "\n".join(lines)
