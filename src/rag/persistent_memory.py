"""Persistent RAG memory.

Warmup examples, approved runtime examples, and runtime diagnostics are useful
across separate CLI invocations. This module stores lightweight
``KnowledgeChunk`` and ``PatternSpec`` records on disk while keeping the
in-process retriever simple.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path

from contract_versions import SEMANTIC_EVIDENCE_CONTRACT_VERSION
from pattern_registry import PatternSpec, normalize_pattern
from formula_closure import FormulaClosurePolicy, validate_formula_closure
from ltlf_parser import parse_formula
from rag.knowledge_base import KnowledgeChunk
from schemas import CorrectionMemoryRecord, FaithfulLearningExample


DEFAULT_PERSISTENT_RAG_FILE = "data/output/memory/persistent_rag.jsonl"
DEFAULT_APPROVED_MEMORY_FILE = DEFAULT_PERSISTENT_RAG_FILE
PERSISTENT_MEMORY_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class PersistentApprovedMemory:
    """Chunks and pattern specs loaded from persistent storage."""

    chunks: list[KnowledgeChunk]
    pattern_specs: list[PatternSpec]
    faithful_examples: list[dict] | None = None
    corrections: list[dict] | None = None


def load_approved_memory(
    path: str | Path = DEFAULT_APPROVED_MEMORY_FILE,
) -> PersistentApprovedMemory:
    """Load persisted RAG memory, returning empty memory if absent."""

    memory_path = Path(path)
    if not memory_path.exists():
        return PersistentApprovedMemory(
            chunks=[], pattern_specs=[], faithful_examples=[], corrections=[]
        )

    chunks: list[KnowledgeChunk] = []
    pattern_specs: list[PatternSpec] = []
    faithful_examples: list[dict] = []
    corrections: list[dict] = []
    for line_number, raw_line in enumerate(
        memory_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"持久化 RAG 文件 {memory_path} 第 {line_number} 行不是合法 JSON。"
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(
                f"持久化 RAG 文件 {memory_path} 第 {line_number} 行必须是 JSON object。"
            )
        if record.get("schema_version") != PERSISTENT_MEMORY_SCHEMA_VERSION:
            # Old records were approved under a different semantic evidence
            # gate and must never enter the current faithful-IR retriever.
            continue
        if (
            record.get("semantic_evidence_contract_version")
            != SEMANTIC_EVIDENCE_CONTRACT_VERSION
        ):
            continue
        kind = record.get("kind")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(
                f"持久化 RAG 文件 {memory_path} 第 {line_number} 行缺少 payload object。"
            )
        if kind == "chunk":
            chunks.append(_chunk_from_payload(payload, memory_path, line_number))
        elif kind == "pattern_spec":
            pattern_specs.append(_pattern_spec_from_payload(payload, memory_path, line_number))
        elif kind == "faithful_example":
            _validate_faithful_example(payload, memory_path, line_number)
            faithful_examples.append(dict(payload))
        elif kind == "correction":
            _validate_correction(payload, memory_path, line_number)
            corrections.append(dict(payload))
        else:
            raise ValueError(
                f"持久化 RAG 文件 {memory_path} 第 {line_number} 行 kind 不支持: {kind}。"
            )

    return PersistentApprovedMemory(
        chunks=_dedupe_chunks(chunks),
        pattern_specs=_dedupe_pattern_specs(pattern_specs),
        faithful_examples=_dedupe_records(faithful_examples, "requirement_id"),
        corrections=_dedupe_records(corrections, "correction_id"),
    )


def merge_approved_memory(
    path: str | Path = DEFAULT_APPROVED_MEMORY_FILE,
    *,
    chunks: list[KnowledgeChunk] | None = None,
    pattern_specs: list[PatternSpec] | None = None,
    faithful_examples: list[dict | FaithfulLearningExample] | None = None,
    corrections: list[dict | CorrectionMemoryRecord] | None = None,
) -> PersistentApprovedMemory:
    """Merge records into persistent storage and return the saved set."""

    memory_path = Path(path)
    existing = load_approved_memory(memory_path)
    merged_chunks = _dedupe_chunks([*existing.chunks, *(chunks or [])])
    merged_pattern_specs = _dedupe_pattern_specs(
        [*existing.pattern_specs, *(pattern_specs or [])]
    )
    merged_faithful_examples = _dedupe_records(
        [
            *(existing.faithful_examples or []),
            *(_record_dict(item) for item in (faithful_examples or [])),
        ],
        "requirement_id",
    )
    merged_corrections = _dedupe_records(
        [
            *(existing.corrections or []),
            *(_record_dict(item) for item in (corrections or [])),
        ],
        "correction_id",
    )
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for chunk in sorted(merged_chunks, key=lambda item: item.chunk_id):
        records.append(
            {
                "schema_version": PERSISTENT_MEMORY_SCHEMA_VERSION,
                "semantic_evidence_contract_version": SEMANTIC_EVIDENCE_CONTRACT_VERSION,
                "kind": "chunk",
                "payload": asdict(chunk),
            }
        )
    for spec in sorted(
        merged_pattern_specs,
        key=lambda item: (normalize_pattern(item.pattern), item.source_id),
    ):
        records.append(
            {
                "schema_version": PERSISTENT_MEMORY_SCHEMA_VERSION,
                "semantic_evidence_contract_version": SEMANTIC_EVIDENCE_CONTRACT_VERSION,
                "kind": "pattern_spec",
                "payload": asdict(spec),
            }
        )
    for example in sorted(
        merged_faithful_examples,
        key=lambda item: str(item.get("requirement_id", "")),
    ):
        _validate_faithful_example(example, memory_path, 0)
        records.append(
            {
                "schema_version": PERSISTENT_MEMORY_SCHEMA_VERSION,
                "semantic_evidence_contract_version": SEMANTIC_EVIDENCE_CONTRACT_VERSION,
                "semantic_representation": "faithful_ir",
                "kind": "faithful_example",
                "payload": example,
            }
        )
    for correction in sorted(
        merged_corrections,
        key=lambda item: str(item.get("correction_id", "")),
    ):
        _validate_correction(correction, memory_path, 0)
        records.append(
            {
                "schema_version": PERSISTENT_MEMORY_SCHEMA_VERSION,
                "semantic_evidence_contract_version": SEMANTIC_EVIDENCE_CONTRACT_VERSION,
                "semantic_representation": "faithful_ir",
                "kind": "correction",
                "payload": correction,
            }
        )

    _atomic_write_text(
        memory_path,
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        + ("\n" if records else ""),
    )
    return PersistentApprovedMemory(
        chunks=merged_chunks,
        pattern_specs=merged_pattern_specs,
        faithful_examples=merged_faithful_examples,
        corrections=merged_corrections,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _chunk_from_payload(
    payload: dict,
    path: Path,
    line_number: int,
) -> KnowledgeChunk:
    try:
        return KnowledgeChunk(
            chunk_id=_required_string(payload, "chunk_id"),
            title=_required_string(payload, "title"),
            text=_required_string(payload, "text"),
            source=_required_string(payload, "source"),
        )
    except ValueError as exc:
        raise ValueError(
            f"持久化 RAG 文件 {path} 第 {line_number} 行 chunk 字段无效: {exc}"
        ) from exc


def _pattern_spec_from_payload(
    payload: dict,
    path: Path,
    line_number: int,
) -> PatternSpec:
    try:
        return PatternSpec(
            pattern=_required_string(payload, "pattern"),
            source_id=_required_string(payload, "source_id"),
            requirement=_required_string(payload, "requirement"),
            formula_text=_required_string(payload, "formula_text"),
            semantic_fields=_tuple_pairs(payload.get("semantic_fields", ())),
            derivation_rules=tuple(str(item) for item in payload.get("derivation_rules", ())),
            trace_count=int(payload.get("trace_count", 0)),
            skeleton_templates=tuple(str(item) for item in payload.get("skeleton_templates", ())),
            hole_template_text=str(payload.get("hole_template_text", "")),
            final_candidate_text=str(payload.get("final_candidate_text", "")),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"持久化 RAG 文件 {path} 第 {line_number} 行 pattern_spec 字段无效: {exc}"
        ) from exc


def _required_string(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} 必须是非空字符串")
    return value


def _tuple_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    pairs: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        pairs.append((str(item[0]), str(item[1])))
    return tuple(pairs)


def _dedupe_chunks(chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
    deduped: dict[str, KnowledgeChunk] = {}
    for chunk in chunks:
        existing = deduped.get(chunk.chunk_id)
        if existing is None or existing == chunk:
            deduped[chunk.chunk_id] = chunk
            continue

        collision_id = _collision_chunk_id(chunk)
        while collision_id in deduped and deduped[collision_id] != chunk:
            collision_id = _collision_chunk_id(
                replace(chunk, chunk_id=collision_id)
            )
        deduped[collision_id] = replace(chunk, chunk_id=collision_id)
    return list(deduped.values())


def _dedupe_pattern_specs(specs: list[PatternSpec]) -> list[PatternSpec]:
    deduped: dict[tuple[str, str], PatternSpec] = {}
    for spec in specs:
        deduped[(normalize_pattern(spec.pattern), spec.source_id)] = spec
    return list(deduped.values())


def _collision_chunk_id(chunk: KnowledgeChunk) -> str:
    raw = "\n".join([chunk.chunk_id, chunk.title, chunk.text, chunk.source])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{chunk.chunk_id}-{digest}"


def _validate_faithful_example(payload: dict, path: Path, line_number: int) -> None:
    required = (
        "requirement_id",
        "natural_language",
        "approved_ir",
        "trace_obligation_plan",
        "approved_traces",
        "skeleton_plan",
        "approved_skeletons",
        "completion_hole_plan",
        "approved_completion",
        "final_formula",
        "approval_metadata",
    )
    missing = [key for key in required if key not in payload]
    metadata = payload.get("approval_metadata", {})
    status = metadata.get("status", "") if isinstance(metadata, dict) else ""
    coverage_status = (
        payload.get("coverage_status")
        or (metadata.get("ir_coverage_status", "") if isinstance(metadata, dict) else "")
    )
    forbidden = bool(
        payload.get("provisional")
        or payload.get("forced")
        or payload.get("degraded")
        or payload.get("formula_oracle")
        or coverage_status != "complete"
        or (isinstance(metadata, dict) and metadata.get("run_origin") == "feedback_rerun")
    )
    if missing or status != "final" or forbidden:
        location = f"{path} line {line_number}" if line_number else str(path)
        raise ValueError(
            f"faithful example is not persistable at {location}: "
            + ("missing " + ", ".join(missing) if missing else "not approved final")
        )
    approved_ir = payload.get("approved_ir", {})
    if not isinstance(approved_ir, dict):
        raise ValueError("faithful example approved_ir must be an object")
    allowed_aps = tuple(
        str(item.get("name", ""))
        for item in approved_ir.get("atomic_propositions", ())
        if isinstance(item, dict) and str(item.get("name", ""))
    )
    ir_version = str(approved_ir.get("ir_version", ""))
    try:
        formula = parse_formula(str(payload.get("final_formula", "")))
    except ValueError as exc:
        raise ValueError("faithful example final_formula cannot be parsed") from exc
    closure = validate_formula_closure(
        formula,
        FormulaClosurePolicy(
            allowed_aps=allowed_aps,
            optional_aps=allowed_aps,
            approved_ir_version=ir_version,
        ),
        ir_version=ir_version,
    )
    if not closure.passed:
        raise ValueError(
            "faithful example failed FormulaClosurePolicy: "
            + "; ".join(closure.reasons)
        )
    if isinstance(metadata, dict):
        if str(metadata.get("approved_ir_version", "")) != ir_version:
            raise ValueError("faithful example approval metadata has stale IR binding")
        approval_record = metadata.get("final_approval_record", {})
        if not isinstance(approval_record, dict):
            raise ValueError("faithful example final approval record is missing")
        if str(approval_record.get("approved_formula", "")) != str(formula):
            raise ValueError("faithful example final approval formula mismatch")


def _validate_correction(payload: dict, path: Path, line_number: int) -> None:
    required = (
        "correction_id",
        "failed_ir_node_ids",
        "old_ir_fragment",
        "corrected_ir_fragment",
        "affected_artifacts",
        "correction_reason",
        "approval_status",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("correction memory missing fields: " + ", ".join(missing))


def _dedupe_records(records: list[dict], key: str) -> list[dict]:
    deduped: dict[str, dict] = {}
    for record in records:
        value = str(record.get(key, ""))
        if value:
            deduped[value] = dict(record)
    return list(deduped.values())


def _record_dict(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    return asdict(value)
