"""Unified persistent RAG entrypoints."""

from __future__ import annotations

from pathlib import Path
import json

from contract_versions import (
    FAITHFUL_IR_CANONICALIZER_VERSION,
    FAITHFUL_IR_CONTRACT_VERSION,
    FAITHFUL_IR_SCHEMA_VERSION,
    FAITHFUL_IR_VALIDATOR_VERSION,
    OBLIGATION_COMPILER_VERSION,
)
from pattern_registry import PatternSpec, register_pattern_specs
from rag.knowledge_base import KnowledgeChunk
from rag.persistent_memory import (
    DEFAULT_APPROVED_MEMORY_FILE,
    load_approved_memory,
)
from rag.retriever import KeywordRetriever


_VERSIONED_LEARNING_SOURCES = {
    "approved-example:semantic",
    "approved-example:trace",
    "approved-example:guided-semantic",
    "approved-example:guided-trace",
}
_FAITHFUL_IR_LEARNING_SOURCE = "approved-example:faithful-ir"
_CURRENT_OBLIGATION_MARKER = (
    f"obligation_compiler_version: {OBLIGATION_COMPILER_VERSION}"
)
_CURRENT_IR_MARKERS = (
    f"faithful_ir_contract_version: {FAITHFUL_IR_CONTRACT_VERSION}",
    f"faithful_ir_validator_version: {FAITHFUL_IR_VALIDATOR_VERSION}",
)


def load_persistent_rag(
    memory_file: str | Path | None = DEFAULT_APPROVED_MEMORY_FILE,
) -> tuple[list[KnowledgeChunk], list[PatternSpec]]:
    """Load persistent chunks/specs. A missing file is an empty RAG."""

    if not memory_file:
        return [], []
    memory = load_approved_memory(memory_file)
    current_faithful_examples = [
        example
        for example in (memory.faithful_examples or [])
        if _current_faithful_example(example)
    ]
    faithful_chunks = [
        KnowledgeChunk(
            chunk_id=f"faithful-ir-{example.get('requirement_id', index)}",
            title="Approved FaithfulSemanticIR example",
            text="\n".join(
                (
                    "schema_version: "
                    + str(
                        example.get("approved_ir", {}).get(
                            "schema_version", FAITHFUL_IR_SCHEMA_VERSION
                        )
                        if isinstance(example.get("approved_ir"), dict)
                        else FAITHFUL_IR_SCHEMA_VERSION
                    ),
                    "semantic_representation: faithful_ir",
                    f"faithful_ir_contract_version: {FAITHFUL_IR_CONTRACT_VERSION}",
                    f"faithful_ir_validator_version: {FAITHFUL_IR_VALIDATOR_VERSION}",
                    f"requirement_id: {example.get('requirement_id', '')}",
                    f"requirement: {example.get('natural_language', '')}",
                    "approved_ir_json:",
                    json.dumps(example.get("approved_ir", {}), ensure_ascii=False),
                )
            ),
            source="approved-example:faithful-ir",
        )
        for index, example in enumerate(current_faithful_examples, start=1)
    ]
    completion_chunks = [
        KnowledgeChunk(
            chunk_id=f"faithful-completion-{example.get('requirement_id', index)}",
            title="Approved faithful completion example",
            text="\n".join(
                (
                    "schema_version: "
                    + str(
                        example.get("approved_ir", {}).get(
                            "schema_version", FAITHFUL_IR_SCHEMA_VERSION
                        )
                        if isinstance(example.get("approved_ir"), dict)
                        else FAITHFUL_IR_SCHEMA_VERSION
                    ),
                    f"requirement_id: {example.get('requirement_id', '')}",
                    f"approved_completion: {example.get('approved_completion', '')}",
                    f"final_formula: {example.get('final_formula', '')}",
                )
            ),
            source="approved-example:operator-completion",
        )
        for index, example in enumerate(current_faithful_examples, start=1)
    ]
    chunks = [
        chunk
        for chunk in [*memory.chunks, *faithful_chunks, *completion_chunks]
        if _current_learning_chunk(chunk)
    ]
    return chunks, list(memory.pattern_specs)


def build_persistent_retriever(
    memory_file: str | Path | None = DEFAULT_APPROVED_MEMORY_FILE,
    *,
    initial_chunks: list[KnowledgeChunk] | None = None,
    memory_mode: str = "read_only",
) -> KeywordRetriever:
    """Build a retriever over persistent memory.

    The default is read-only so test-time/runtime diagnostics cannot mutate the
    persistent RAG file. Warmup batch commits call ``merge_approved_memory``
    explicitly after all samples finish.
    """

    persistent_chunks, pattern_specs = load_persistent_rag(memory_file)
    register_pattern_specs(pattern_specs)
    retriever = KeywordRetriever(
        persistent_chunks,
        persistent_memory_file=memory_file,
        memory_mode=memory_mode,
    )
    _extend_once(retriever, initial_chunks or [])
    return retriever


def _extend_once(
    retriever: KeywordRetriever,
    chunks: list[KnowledgeChunk],
) -> list[str]:
    existing_ids = {chunk.chunk_id for chunk in retriever.chunks}
    new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
    retriever.extend_chunks(new_chunks)
    return [chunk.chunk_id for chunk in new_chunks]


def _current_learning_chunk(chunk: KnowledgeChunk) -> bool:
    if chunk.source == _FAITHFUL_IR_LEARNING_SOURCE:
        return all(marker in chunk.text for marker in _CURRENT_IR_MARKERS)
    if chunk.source not in _VERSIONED_LEARNING_SOURCES:
        return True
    if _CURRENT_OBLIGATION_MARKER not in chunk.text:
        return False
    if "semantic_representation: faithful_ir" in chunk.text:
        return all(marker in chunk.text for marker in _CURRENT_IR_MARKERS)
    return True


def _current_faithful_example(example: object) -> bool:
    if not isinstance(example, dict):
        return False
    ir = example.get("approved_ir")
    if not isinstance(ir, dict):
        return False
    return bool(
        str(ir.get("schema_version", "")) == FAITHFUL_IR_SCHEMA_VERSION
        and str(ir.get("contract_version", "")) == FAITHFUL_IR_CONTRACT_VERSION
        and str(ir.get("canonicalizer_version", ""))
        == FAITHFUL_IR_CANONICALIZER_VERSION
        and str(ir.get("validator_version", "")) == FAITHFUL_IR_VALIDATOR_VERSION
    )
