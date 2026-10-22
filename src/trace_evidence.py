"""Versioned finite-trace evidence registration and safe conflict isolation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Iterable

from data_examples import LabeledTrace
from faithful_ir import TEMPORAL_NODE_TYPES
from ir_trace_semantics import IRTraceSemanticsError, evaluate_ir_trace
from schemas import (
    SemanticStructure,
    TraceAdjudicationBrief,
    TraceAnnotation,
    TraceQuery,
)


TRACE_EVIDENCE_STATUSES = frozenset(
    {"active", "disputed", "superseded", "unresolved"}
)
TRACE_DISPUTE_STATUSES = frozenset({"pending", "resolved", "unresolved"})


@dataclass(frozen=True)
class TraceEvidenceRecord:
    """One immutable adjudication bound to a reviewed semantic context."""

    evidence_id: str
    semantic_context_id: str
    trace_fingerprint: str
    trace: LabeledTrace
    status: str = "active"
    adjudication_brief_id: str = ""
    source_ir_version: str = ""
    superseded_by_context_id: str = ""

    def __post_init__(self) -> None:
        if self.status not in TRACE_EVIDENCE_STATUSES:
            raise ValueError("Unsupported trace evidence status: " + self.status)


@dataclass(frozen=True)
class TraceEvidenceDispute:
    """Expected disagreement retained outside coverage and formula scoring."""

    dispute_id: str
    semantic_context_id: str
    trace_fingerprint: str
    trace_shape: tuple[tuple[str, ...], ...]
    reason_code: str
    evidence_ids: tuple[str, ...]
    labels: tuple[str, ...]
    query_ids: tuple[str, ...]
    status: str = "pending"
    resolution_id: str = ""
    suggested_rerun_stage: str = "semantic"

    def __post_init__(self) -> None:
        if self.status not in TRACE_DISPUTE_STATUSES:
            raise ValueError("Unsupported trace dispute status: " + self.status)


@dataclass(frozen=True)
class TraceEvidenceReconciliation:
    records: tuple[TraceEvidenceRecord, ...]
    active_traces: tuple[LabeledTrace, ...]
    disputes: tuple[TraceEvidenceDispute, ...]
    reused_evidence_count: int = 0

    @property
    def unresolved_disputes(self) -> tuple[TraceEvidenceDispute, ...]:
        return tuple(item for item in self.disputes if item.status != "resolved")

    @property
    def has_disputes(self) -> bool:
        return bool(self.unresolved_disputes)

    @property
    def coverage_debt(self) -> tuple[str, ...]:
        return tuple(
            f"trace_evidence_disputed:{item.dispute_id}"
            for item in self.unresolved_disputes
        )


def trace_shape(trace: LabeledTrace) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted(step)) for step in trace.trace)


def trace_fingerprint(trace: LabeledTrace) -> str:
    return _digest(
        (trace.evidence_purpose, trace_shape(trace)),
        length=32,
    )


def evidence_record_from_trace(
    trace: LabeledTrace,
    *,
    semantic_context_id: str,
    adjudication_brief_id: str,
    status: str = "active",
) -> TraceEvidenceRecord:
    fingerprint = trace_fingerprint(trace)
    payload = {
        "semantic_context_id": semantic_context_id,
        "trace_fingerprint": fingerprint,
        "expected": trace.expected,
        "kind": trace.kind,
        "trace_id": trace.trace_id or trace.name,
        "obligation_ids": trace.obligation_ids,
        "trace_role": trace.trace_role,
        "target_ambiguity_axis_ids": trace.target_ambiguity_axis_ids,
        "review_record_id": trace.review_record_id,
        "evidence_purpose": trace.evidence_purpose,
        "parent_candidate_set_id": trace.parent_candidate_set_id,
        "parent_hole_plan_version": trace.parent_hole_plan_version,
    }
    return TraceEvidenceRecord(
        evidence_id="TEV-" + _digest(payload, length=24),
        semantic_context_id=semantic_context_id,
        trace_fingerprint=fingerprint,
        trace=trace,
        status=status,
        adjudication_brief_id=adjudication_brief_id,
        source_ir_version=trace.ir_version,
    )


def reconcile_trace_evidence(
    *,
    semantic: SemanticStructure,
    brief: TraceAdjudicationBrief,
    existing_records: Iterable[TraceEvidenceRecord] = (),
    existing_disputes: Iterable[TraceEvidenceDispute] = (),
    existing_traces: Iterable[LabeledTrace] = (),
    newly_adjudicated: Iterable[LabeledTrace] = (),
) -> TraceEvidenceReconciliation:
    """Register evidence without silently overwriting labels or semantics."""

    context_id = brief.semantic_context_id
    records = list(existing_records)
    disputes = list(existing_disputes)
    records = [
        replace(
            record,
            status="superseded",
            superseded_by_context_id=context_id,
        )
        if record.status == "active" and record.semantic_context_id != context_id
        else record
        for record in records
    ]

    known_ids = {record.evidence_id for record in records}
    reused = 0
    for trace in existing_traces:
        inferred_context = _legacy_trace_context(trace, brief)
        record = evidence_record_from_trace(
            trace,
            semantic_context_id=inferred_context,
            adjudication_brief_id=brief.brief_id,
            status="active" if inferred_context == context_id else "superseded",
        )
        if record.evidence_id in known_ids:
            reused += 1
            continue
        records.append(record)
        known_ids.add(record.evidence_id)

    for trace in newly_adjudicated:
        record = evidence_record_from_trace(
            trace,
            semantic_context_id=context_id,
            adjudication_brief_id=brief.brief_id,
        )
        if record.evidence_id in known_ids:
            reused += 1
            continue
        records.append(record)
        known_ids.add(record.evidence_id)

    prior_disputed_shapes = {
        (record.semantic_context_id, record.trace_fingerprint)
        for record in records
        if record.status == "disputed"
    }
    records = [
        replace(record, status="disputed")
        if record.status == "active"
        and (record.semantic_context_id, record.trace_fingerprint)
        in prior_disputed_shapes
        else record
        for record in records
    ]

    dispute_keys = {
        (item.semantic_context_id, item.trace_fingerprint, item.reason_code)
        for item in disputes
        if item.status != "resolved"
    }
    grouped = _current_evidence_by_shape(records, context_id)
    for fingerprint, group in grouped.items():
        if len({record.trace.expected for record in group}) <= 1:
            continue
        records = _set_group_status(records, group, "disputed")
        dispute = _build_dispute(
            context_id=context_id,
            group=group,
            reason_code="SAME_TRACE_DIFFERENT_LABEL",
        )
        key = (context_id, fingerprint, dispute.reason_code)
        if key not in dispute_keys:
            disputes.append(dispute)
            dispute_keys.add(key)

    if _ir_has_fully_fixed_trace_semantics(semantic):
        grouped = _current_evidence_by_shape(records, context_id, active_only=True)
        for fingerprint, group in grouped.items():
            representative = group[0].trace
            try:
                ir_label = evaluate_ir_trace(semantic.faithful_ir, representative.trace)
            except (IRTraceSemanticsError, ValueError):
                continue
            if ir_label == representative.expected:
                continue
            records = _set_group_status(records, group, "disputed")
            dispute = _build_dispute(
                context_id=context_id,
                group=group,
                reason_code="FIXED_IR_LABEL_MISMATCH",
                reviewed_label=ir_label,
            )
            key = (context_id, fingerprint, dispute.reason_code)
            if key not in dispute_keys:
                disputes.append(dispute)
                dispute_keys.add(key)

    return TraceEvidenceReconciliation(
        records=tuple(records),
        active_traces=tuple(_project_active_traces(records, context_id)),
        disputes=tuple(
            item for item in disputes if item.semantic_context_id == context_id
        ),
        reused_evidence_count=reused,
    )


def reuse_confirmed_annotations(
    queries: Iterable[TraceQuery],
    records: Iterable[TraceEvidenceRecord],
    *,
    semantic_context_id: str,
) -> tuple[list[TraceAnnotation], list[TraceQuery]]:
    """Reuse identical confirmed evidence instead of repeatedly sampling a Judge."""

    reusable: dict[
        tuple[str, tuple[tuple[str, ...], ...]], LabeledTrace
    ] = {}
    conflicting: set[
        tuple[str, tuple[tuple[str, ...], ...]]
    ] = set()
    for record in records:
        if record.status != "active" or record.semantic_context_id != semantic_context_id:
            continue
        shape = (record.trace.evidence_purpose, trace_shape(record.trace))
        previous = reusable.get(shape)
        if previous is not None and previous.expected != record.trace.expected:
            conflicting.add(shape)
        else:
            reusable[shape] = record.trace
    for shape in conflicting:
        reusable.pop(shape, None)

    reused: list[TraceAnnotation] = []
    pending: list[TraceQuery] = []
    for query in queries:
        shape = (
            query.purpose,
            tuple(tuple(sorted(step)) for step in query.trace),
        )
        trace = reusable.get(shape)
        if trace is None:
            pending.append(query)
            continue
        reused.append(
            TraceAnnotation(
                query_id=query.query_id,
                label="SAT" if trace.expected else "UNSAT",
                confidence=trace.confidence or "high",
                rationale=(
                    "Reused the confirmed judgment for this identical trace under "
                    "the same reviewed requirement meaning."
                ),
                label_source=trace.label_source,
                annotator="evidence_registry",
                prompt_version="trace-evidence-reuse-v1",
                temporal_simulation_summary=(
                    trace.temporal_simulation_summary or trace.rationale
                ),
                supported_text_spans=trace.supported_text_spans,
                purpose=query.purpose,
            )
        )
    return reused, pending


def apply_resolved_trace_label(
    reconciliation: TraceEvidenceReconciliation,
    *,
    dispute: TraceEvidenceDispute,
    expected: bool,
    rationale: str,
    resolution_id: str,
) -> TraceEvidenceReconciliation:
    """Materialize an independently verified label without editing old evidence."""

    records = list(reconciliation.records)
    relevant = [
        record
        for record in records
        if record.semantic_context_id == dispute.semantic_context_id
        and record.trace_fingerprint == dispute.trace_fingerprint
    ]
    if not relevant:
        raise ValueError("Trace evidence dispute has no source evidence records.")
    records = [
        replace(record, status="superseded")
        if record in relevant
        else record
        for record in records
    ]
    template = relevant[0].trace
    resolved_trace = replace(
        template,
        expected=expected,
        adjudicated_label="SAT" if expected else "UNSAT",
        confidence="high",
        rationale=rationale,
        label_source="independent_judge",
        annotation_locked=True,
        generation_source="llm_semantic_conflict_resolution",
        review_record_id=resolution_id,
    )
    resolved_record = evidence_record_from_trace(
        resolved_trace,
        semantic_context_id=dispute.semantic_context_id,
        adjudication_brief_id=relevant[0].adjudication_brief_id,
    )
    records.append(resolved_record)
    disputes = tuple(
        replace(item, status="resolved", resolution_id=resolution_id)
        if item.dispute_id == dispute.dispute_id
        else item
        for item in reconciliation.disputes
    )
    return TraceEvidenceReconciliation(
        records=tuple(records),
        active_traces=tuple(
            _project_active_traces(records, dispute.semantic_context_id)
        ),
        disputes=disputes,
        reused_evidence_count=reconciliation.reused_evidence_count,
    )


def _legacy_trace_context(
    trace: LabeledTrace,
    brief: TraceAdjudicationBrief,
) -> str:
    if not trace.ir_version or trace.ir_version == brief.ir_version:
        return brief.semantic_context_id
    return "TCTX-legacy-ir-" + _digest(trace.ir_version, length=20)


def _current_evidence_by_shape(
    records: Iterable[TraceEvidenceRecord],
    context_id: str,
    *,
    active_only: bool = False,
) -> dict[str, list[TraceEvidenceRecord]]:
    grouped: dict[str, list[TraceEvidenceRecord]] = {}
    allowed = {"active"} if active_only else {"active", "disputed"}
    for record in records:
        if record.semantic_context_id != context_id or record.status not in allowed:
            continue
        grouped.setdefault(record.trace_fingerprint, []).append(record)
    return grouped


def _project_active_traces(
    records: Iterable[TraceEvidenceRecord],
    context_id: str,
) -> list[LabeledTrace]:
    projected: list[LabeledTrace] = []
    seen: set[tuple[object, ...]] = set()
    for record in records:
        if record.status != "active" or record.semantic_context_id != context_id:
            continue
        trace = record.trace
        key = (
            trace_shape(trace),
            trace.expected,
            trace.kind,
            tuple(trace.obligation_ids),
            trace.trace_role,
            tuple(trace.target_ambiguity_axis_ids),
            trace.evidence_purpose,
            trace.parent_candidate_set_id,
            trace.parent_hole_plan_version,
        )
        if key in seen:
            continue
        seen.add(key)
        projected.append(trace)
    return projected


def _ir_has_fully_fixed_trace_semantics(semantic: SemanticStructure) -> bool:
    ir = semantic.faithful_ir
    if ir is None:
        return False
    return not any(
        node.node_type in TEMPORAL_NODE_TYPES
        and node.temporal_resolution == "trace_resolvable"
        for node in ir.nodes
    )


def _set_group_status(
    records: list[TraceEvidenceRecord],
    group: list[TraceEvidenceRecord],
    status: str,
) -> list[TraceEvidenceRecord]:
    ids = {record.evidence_id for record in group}
    return [
        replace(record, status=status) if record.evidence_id in ids else record
        for record in records
    ]


def _build_dispute(
    *,
    context_id: str,
    group: list[TraceEvidenceRecord],
    reason_code: str,
    reviewed_label: bool | None = None,
) -> TraceEvidenceDispute:
    labels = tuple(
        sorted({"SAT" if record.trace.expected else "UNSAT" for record in group})
    )
    if reviewed_label is not None:
        labels = tuple(
            dict.fromkeys(
                labels
                + (("SAT" if reviewed_label else "UNSAT") + " (reviewed meaning)",)
            )
        )
    payload = {
        "context": context_id,
        "fingerprint": group[0].trace_fingerprint,
        "reason": reason_code,
        "labels": labels,
    }
    return TraceEvidenceDispute(
        dispute_id="TED-" + _digest(payload, length=20),
        semantic_context_id=context_id,
        trace_fingerprint=group[0].trace_fingerprint,
        trace_shape=trace_shape(group[0].trace),
        reason_code=reason_code,
        evidence_ids=tuple(record.evidence_id for record in group),
        labels=labels,
        query_ids=tuple(
            dict.fromkeys(record.trace.trace_id or record.trace.name for record in group)
        ),
    )


def _digest(value: object, *, length: int) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:length]
