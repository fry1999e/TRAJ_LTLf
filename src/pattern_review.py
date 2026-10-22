"""Runtime handling for unknown Traj-LTLf patterns.

Unknown patterns are accepted as provisional semantics for the current run.
They may use a complete ``formula_text`` when one is externally available, but
normal subject requirements continue from NL, APs, components, and traces
without requiring the NLU stage to invent a formula. Unknown-pattern outputs
are never written to persistent RAG automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from data_examples import LabeledTrace
from pattern_registry import collect_formula_atoms, parse_learned_formula
from rag.retriever import KeywordRetriever
from schemas import SemanticStructure
from terminal_progress import TerminalProgress


@dataclass(frozen=True)
class PatternLearningResult:
    """Outcome of the unknown-pattern runtime marker."""

    approved: bool
    provisional: bool
    semantic: SemanticStructure | None
    traces: list[LabeledTrace]
    record: dict
    runtime_rag_chunks: list[str]
    comments: str
    review_path: str = ""


def run_pattern_learning_review(
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    retriever: KeywordRetriever,
    review_mode: str,
    round_index: int,
    records_dir: str,
    progress: TerminalProgress | None = None,
) -> PatternLearningResult:
    """
    Mark an unsupported pattern as an unknown LLM-decided runtime pattern.

    The name is kept for compatibility with existing call sites, but this is no
    longer a human pattern-learning review.  No Markdown file is saved.
    """

    del requirement, retriever, review_mode, round_index, records_dir

    if not semantic.formula_text.strip():
        explanation = semantic.explanation.strip()
        comments = (
            f"未知模式 {semantic.pattern} 没有可执行 formula_text；"
            "本次运行将使用 NL、AP、components 和有限轨迹作为 provisional "
            "证据继续，不写入持久性 RAG。"
        )
        marked_semantic = replace(
            semantic,
            components=_with_unknown_pattern_components(
                semantic.components,
                semantic.pattern,
                policy="nl_components_traces_provisional",
            ),
            explanation=f"{explanation} {comments}".strip(),
            pattern_status="unknown",
            original_pattern=semantic.original_pattern or semantic.pattern,
            persistent_rag_allowed=False,
        )
        if progress is not None:
            progress.warn(f"{requirement_id} unknown pattern missing formula_text")
        return PatternLearningResult(
            approved=False,
            provisional=True,
            semantic=marked_semantic,
            traces=[],
            record={},
            runtime_rag_chunks=[],
            comments=comments,
        )

    try:
        formula = parse_learned_formula(semantic.formula_text)
    except Exception as exc:
        comments = (
            f"未知模式 {semantic.pattern} 的 formula_text 无法解析："
            f"{semantic.formula_text}。原因: {exc}"
        )
        if progress is not None:
            progress.warn(f"{requirement_id} unknown pattern formula_text invalid")
        return PatternLearningResult(
            approved=False,
            provisional=False,
            semantic=None,
            traces=[],
            record={},
            runtime_rag_chunks=[],
            comments=comments,
        )

    aps = semantic.aps or collect_formula_atoms(formula)
    explanation = semantic.explanation.strip()
    unknown_note = (
        f"未知模式 {semantic.pattern} 由 LLM 自由给出 formula_text，"
        "本次运行不检索相似参考模式，也不写入持久性 RAG。"
    )
    marked_semantic = replace(
        semantic,
        aps=aps,
        components=_with_unknown_pattern_components(
            semantic.components,
            semantic.pattern,
            policy="llm_formula_text_no_similar_reference",
        ),
        explanation=f"{explanation} {unknown_note}".strip(),
        pattern_status="unknown",
        original_pattern=semantic.original_pattern or semantic.pattern,
        persistent_rag_allowed=False,
    )
    if progress is not None:
        progress.warn(
            f"{requirement_id} unknown pattern accepted from LLM formula_text: "
            f"{semantic.pattern}"
        )
    return PatternLearningResult(
        approved=False,
        provisional=True,
        semantic=marked_semantic,
        traces=[],
        record={},
        runtime_rag_chunks=[],
        comments=unknown_note,
    )


def _with_unknown_pattern_components(
    components: tuple[tuple[str, str], ...],
    pattern: str,
    *,
    policy: str,
) -> tuple[tuple[str, str], ...]:
    existing_keys = {key for key, _value in components}
    additions: list[tuple[str, str]] = []
    if "pattern_status" not in existing_keys:
        additions.append(("pattern_status", "unknown"))
    if "original_pattern" not in existing_keys:
        additions.append(("original_pattern", pattern))
    if "unknown_pattern_policy" not in existing_keys:
        additions.append(
            (
                "unknown_pattern_policy",
                policy + "_no_persistent_rag",
            )
        )
    return (*components, *additions)
