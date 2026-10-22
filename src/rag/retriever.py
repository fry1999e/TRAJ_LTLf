"""
轻量 RAG 检索器。

当前实现使用词频向量的余弦相似度，不依赖外部模型或向量数据库。这样既能
保留本地原型的可解释性，也能在运行期样例逐渐变多时只取最相似的 Top-K。
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re

from rag.knowledge_base import KnowledgeChunk
from schemas import FaithfulSemanticIR


class KeywordRetriever:
    """一个简单、可解释、零依赖的余弦相似度检索器。"""

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        persistent_memory_file: str | Path | None = None,
        memory_mode: str = "read_only",
    ):
        self.chunks = list(chunks)
        self.persistent_memory_file = (
            str(persistent_memory_file) if persistent_memory_file else None
        )
        if memory_mode not in {"read_only", "read_write"}:
            raise ValueError("Unsupported retriever memory_mode: " + memory_mode)
        self.memory_mode = memory_mode
        self._excluded_examples: list[tuple[str, str]] = []

    def exclude_example(self, requirement_id: str, requirement: str) -> None:
        """Hide one warmup sample's own chunks for the lifetime of this retriever."""

        marker = (requirement_id.strip(), requirement.strip())
        if marker not in self._excluded_examples:
            self._excluded_examples.append(marker)

    def add_chunk(self, chunk: KnowledgeChunk) -> None:
        """把运行期生成的知识片段加入当前 RAG，并按需持久化。"""

        self.chunks.append(chunk)
        self._persist_chunks([chunk])

    def extend_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        """批量加入运行期知识片段，并按需持久化。"""

        if not chunks:
            return
        self.chunks.extend(chunks)
        self._persist_chunks(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        *,
        source_filter: str | tuple[str, ...] | None = None,
    ) -> list[KnowledgeChunk]:
        """返回与 query 余弦相似度最高的 top_k 个知识片段。"""

        return [
            chunk
            for chunk, _score in self.retrieve_with_scores(
                query,
                top_k=top_k,
                source_filter=source_filter,
            )
        ]

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 3,
        *,
        source_filter: str | tuple[str, ...] | None = None,
    ) -> list[tuple[KnowledgeChunk, float]]:
        """返回知识片段及其余弦相似度，便于记录审计信息。"""

        query_vector = _term_counts(query)
        if not query_vector:
            return []

        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self.chunks:
            if self._is_excluded_example_chunk(chunk):
                continue
            if source_filter is not None and not _source_matches(
                chunk.source,
                source_filter,
            ):
                continue
            chunk_vector = _term_counts(f"{chunk.title}\n{chunk.text}")
            score = _cosine_similarity(query_vector, chunk_vector)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [(chunk, score) for score, chunk in scored[:top_k]]

    def _is_excluded_example_chunk(self, chunk: KnowledgeChunk) -> bool:
        if not chunk.source.startswith("approved-example:"):
            return False
        text = chunk.text
        for requirement_id, requirement in self._excluded_examples:
            id_markers = (
                f"requirement_id: {requirement_id}",
                f'"requirement_id": "{requirement_id}"',
            )
            requirement_markers = (
                f"requirement: {requirement}",
                f"original_requirement: {requirement}",
                f'"requirement": "{requirement}"',
                f'"original_requirement": "{requirement}"',
            )
            if any(marker in text for marker in id_markers + requirement_markers):
                return True
        return False

    def retrieve_for_nl(
        self,
        requirement: str,
        *,
        top_k: int = 3,
    ) -> list[KnowledgeChunk]:
        """Stage-one retrieval. Formula-bearing completion chunks are hidden."""

        allowed = (
            "approved-example:faithful-ir",
            "approved-example:semantic-v3",
            "kb/",
            "data/examples/knowledge_base/",
        )
        return self.retrieve(requirement, top_k=top_k, source_filter=allowed)

    def retrieve_for_ir(
        self,
        ir: FaithfulSemanticIR,
        *,
        top_k: int = 3,
        source_filter: str | tuple[str, ...] | None = None,
    ) -> list[KnowledgeChunk]:
        """Stage-two retrieval keyed by semantic topology rather than pattern."""

        return self.retrieve(
            ir_structural_fingerprint(ir),
            top_k=top_k,
            source_filter=source_filter,
        )

    def _persist_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        if (
            self.memory_mode == "read_only"
            or not self.persistent_memory_file
            or not chunks
        ):
            return

        from rag.persistent_memory import merge_approved_memory

        approved_chunks = [
            chunk
            for chunk in chunks
            if chunk.source.startswith("approved-example:")
        ]
        if approved_chunks:
            merge_approved_memory(self.persistent_memory_file, chunks=approved_chunks)


def _source_matches(source: str, source_filter: str | tuple[str, ...]) -> bool:
    filters = (source_filter,) if isinstance(source_filter, str) else source_filter
    normalized_source = source.replace("\\", "/")
    return any(
        normalized_source.startswith(item.replace("\\", "/"))
        for item in filters
    )


def _term_counts(text: str) -> Counter[str]:
    return Counter(_extract_terms(text))


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0

    common_terms = set(left) & set(right)
    dot_product = sum(left[term] * right[term] for term in common_terms)
    if dot_product == 0:
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _extract_terms(text: str) -> list[str]:
    """
    从英文和中文混合文本中抽取检索词。

    英文按单词抽取；中文会按连续中文片段抽取。这个策略很朴素，但足够支撑
    当前本地原型。
    """

    raw_terms = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]+", text.lower())
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "to",
        "of",
        "and",
        "or",
        "should",
        "must",
        "stage",
        "status",
        "requirement",
    }
    return [term for term in raw_terms if term not in stopwords]


def ir_structural_fingerprint(ir: FaithfulSemanticIR) -> str:
    node_map = ir.node_map
    root = node_map[ir.root_node_id]
    parents: dict[str, str] = {}
    for node in ir.nodes:
        for child in node.children:
            parents[child] = node.node_id

    def depth(node_id: str) -> int:
        value = 0
        while node_id in parents:
            value += 1
            node_id = parents[node_id]
        return value

    temporal_tree = " ".join(
        f"{node.node_type}:{','.join(node_map[child].node_type for child in node.children)}"
        for node in ir.nodes
        if node.temporal_relation != "none"
    )
    boolean_structure = " ".join(
        node.node_type
        for node in ir.nodes
        if node.node_type in {"negation", "conjunction", "disjunction", "implication", "equivalence"}
    )
    negations = ",".join(
        node.node_id for node in ir.nodes if node.node_type == "negation"
    ) or "none"
    cardinality = " ".join(
        node.node_type
        for node in ir.nodes
        if node.node_type in {"cardinality", "periodicity", "recurrence"}
    ) or "none"
    ordered_count = sum(
        int(dict(node.parameters).get("event_count", len(node.children)))
        for node in ir.nodes
        if node.node_type == "ordered_sequence"
    )
    return "\n".join(
        (
            f"root_node_type {root.node_type}",
            f"temporal_relation_tree {temporal_tree}",
            f"boolean_structure {boolean_structure}",
            f"scope_depth {max((depth(node.node_id) for node in ir.nodes), default=0)}",
            f"negation_positions {negations}",
            f"boundary_policy {' '.join(node.boundary_policy for node in ir.nodes)}",
            f"cardinality_or_periodicity {cardinality}",
            f"ordered_event_count {ordered_count}",
        )
    )
