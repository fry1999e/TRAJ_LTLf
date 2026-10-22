"""Structured prompt context helpers for LLM-backed agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from rag.knowledge_base import KnowledgeChunk, load_default_kb
from rag.retriever import KeywordRetriever


KB_LTLF_SEMANTICS = "data/examples/knowledge_base/ltlf_semantics.md"
KB_OPERATOR_COMPLETION = "data/examples/knowledge_base/operator_completion.md"
KB_PATTERNS = "data/examples/knowledge_base/patterns.md"
KB_SKELETON_DIAGNOSTICS = "data/examples/knowledge_base/skeleton_diagnostics.md"
KB_TRACE_EXAMPLES = "data/examples/knowledge_base/trace_examples.md"

FULL_PROCESS_SOURCE = "approved-example:full-process"


@dataclass(frozen=True)
class PromptContext:
    """RAG context split into the three prompt sections used by every agent."""

    ltlf_domain_knowledge: tuple[dict[str, Any], ...]
    top_5_examples: tuple[dict[str, Any], ...]
    agent_function_contract: dict[str, Any]

    def payload_fields(self) -> dict[str, Any]:
        """Return fields to merge into the LLM user payload."""

        return {
            "ltlf_domain_knowledge": list(self.ltlf_domain_knowledge),
            "top_5_examples": list(self.top_5_examples),
            "agent_function_contract": self.agent_function_contract,
        }

    def as_rag_context(self) -> tuple[str, ...]:
        """Backwards-compatible compact context strings for review records."""

        items: list[str] = []
        for chunk in self.ltlf_domain_knowledge:
            items.append(
                f"{chunk['title']} [{chunk['source']}]: {chunk['text']}"
            )
        for chunk in self.top_5_examples:
            items.append(
                f"{chunk['title']} [{chunk['source']} score={chunk['similarity']}]: "
                f"{chunk['text']}"
            )
        return tuple(items)


def build_stage_prompt_context(
    retriever: KeywordRetriever,
    *,
    stage: str,
    query_parts: list[str] | tuple[str, ...],
    domain_sources: tuple[str, ...],
    example_sources: tuple[str, ...],
    agent_function_contract: dict[str, Any],
    domain_top_k: int = 5,
    example_top_k: int = 5,
) -> PromptContext:
    """Retrieve stage-specific LTLf knowledge and Top-5 approved examples."""

    query = "\n".join(part for part in query_parts if part)
    domain_retriever = KeywordRetriever(
        _with_default_domain_chunks(retriever.chunks, domain_sources)
    )
    domain_chunks = domain_retriever.retrieve_with_scores(
        query,
        top_k=domain_top_k,
        source_filter=domain_sources,
    )
    example_chunks = retriever.retrieve_with_scores(
        query,
        top_k=example_top_k,
        source_filter=example_sources,
    )

    return PromptContext(
        ltlf_domain_knowledge=tuple(
            _chunk_payload(chunk, score, stage=stage)
            for chunk, score in domain_chunks
        ),
        top_5_examples=tuple(
            _chunk_payload(chunk, score, stage=stage)
            for chunk, score in example_chunks
        ),
        agent_function_contract=agent_function_contract,
    )


def _with_default_domain_chunks(
    chunks: list[KnowledgeChunk],
    domain_sources: tuple[str, ...],
) -> list[KnowledgeChunk]:
    """Include static data/examples/knowledge_base domain knowledge for prompt context only."""

    existing_ids = {chunk.chunk_id for chunk in chunks}
    combined = list(chunks)
    for chunk in load_default_kb():
        if chunk.chunk_id in existing_ids:
            continue
        if _source_matches(chunk.source, domain_sources):
            combined.append(chunk)
            existing_ids.add(chunk.chunk_id)
    return combined


def _source_matches(source: str, filters: tuple[str, ...]) -> bool:
    normalized_source = source.replace("\\", "/")
    return any(normalized_source.startswith(item.replace("\\", "/")) for item in filters)


def structured_system_prompt(
    *,
    agent_name: str,
    stage_name: str,
    extra_rules: tuple[str, ...] = (),
) -> str:
    """Build the common three-part system prompt shell."""

    rules = "".join(extra_rules)
    return (
        f"你是 Traj-LTLf 的 {agent_name}。"
        "请只返回 JSON object，不要输出 Markdown。"
        "你的提示词上下文固定由三部分组成："
        "payload.ltlf_domain_knowledge 是本阶段对应的 LTLf 领域知识；"
        "payload.top_5_examples 是本阶段对应的 Top-5 相似已通过示例；"
        "payload.agent_function_contract 是你当前唯一职责。"
        "必须优先遵守 agent_function_contract；领域知识只用于本阶段规则，"
        "Top-5 示例只用于格式和推理参考，不能照抄不匹配的 AP、公式或结论。"
        "不要执行其他 Agent 的职责，不要补充需求没有给出的业务语义，"
        "不要创建新的 pattern handler。"
        f"当前阶段: {stage_name}。"
        f"{rules}"
    )


def _chunk_payload(
    chunk: KnowledgeChunk,
    score: float,
    *,
    stage: str,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "source": chunk.source,
        "similarity": round(score, 3),
        "text": _trim_chunk_text(chunk, stage),
    }


def _trim_chunk_text(chunk: KnowledgeChunk, stage: str) -> str:
    if not chunk.source.startswith(FULL_PROCESS_SOURCE):
        return _sanitize_upstream_text(chunk.text, stage)

    headings = _FULL_PROCESS_STAGE_HEADINGS.get(stage, ())
    sections = [
        _extract_markdown_section(chunk.text, heading)
        for heading in headings
    ]
    sections = [section for section in sections if section]
    if not sections:
        return _sanitize_upstream_text(chunk.text, stage)

    metadata = _extract_example_metadata(chunk.text)
    return _sanitize_upstream_text(
        "\n\n".join([metadata, *sections]).strip(),
        stage,
    )


def _sanitize_upstream_text(text: str, stage: str) -> str:
    formula_free_stages = {
        "semantic_extraction",
        "obligation_planning",
        "trace_generation",
        "formula_text_trace_generation",
        "trace_query_generation",
    }
    upstream_structure_stages = {
        *formula_free_stages,
        "restricted_skeleton_generation",
        "hole_candidate_generation",
    }
    if stage not in upstream_structure_stages:
        return text
    forbidden = (
        "approved_formula",
        "expected_final_candidate",
        "final_candidate",
        "final_formula",
        "selected_formula",
        "obligation_formula",
        "formula_text",
        "formula-oracle",
        "formula_oracle",
        '"expected":',
        " expected=",
    )
    formula_like = (
        re.compile(r"\b[GXFA-Z]?\s*\([A-Za-z_][A-Za-z0-9_]*\s*->"),
        re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*->\s*(?:[XFG]\s*)?[A-Za-z_(]"),
        re.compile(r"\b(?:G|F|X)\s*\("),
    )
    sanitized: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(token in lowered for token in forbidden):
            continue
        if stage in formula_free_stages and any(
            pattern.search(line) for pattern in formula_like
        ):
            continue
        sanitized.append(line)
    return "\n".join(
        sanitized
    )


def _extract_example_metadata(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(("example_type:", "requirement_id:", "requirement:", "pattern:")):
            lines.append(line)
        if line == "approved_process:":
            break
    return "\n".join(lines)


def _extract_markdown_section(text: str, heading: str) -> str:
    pattern = rf"(?ms)^{re.escape(heading)}\s*\n(.*?)(?=^### |\Z)"
    match = re.search(pattern, text)
    if not match:
        return ""
    return f"{heading}\n{match.group(1).strip()}"


_FULL_PROCESS_STAGE_HEADINGS = {
    "semantic_extraction": ("### Expected Semantic Structure",),
    "trace_generation": ("### Expected Finite Traces",),
    "formula_text_trace_generation": ("### Expected Finite Traces",),
    "restricted_skeleton_generation": ("### Expected Restricted Skeletons",),
    "hole_candidate_generation": ("### Expected Hole Templates",),
    "temporal_hole_completion": (
        "### Expected Hole Templates",
        "### Expected Completion And Consistency",
        "### Expected Final Candidate",
    ),
}
