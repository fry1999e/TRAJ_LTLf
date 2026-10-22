"""Source-first, formula-blind resolution of disputed finite-trace evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from agents.base_agent import JSONLLMClient
from schemas import SemanticStructure, TraceAdjudicationBrief
from trace_evidence import (
    TraceEvidenceDispute,
    TraceEvidenceReconciliation,
)


@dataclass(frozen=True)
class TraceConflictResolutionReport:
    """Stable audit schema; unresolved conflicts require actual human review."""

    resolution_id: str
    dispute_id: str
    status: str
    conflict_cause: str = ""
    trace_label: str = "UNCLEAR"
    ordinary_language_reading: str = ""
    confidence: str = "low"
    source_spans: tuple[str, ...] = ()
    affected_node_ids: tuple[str, ...] = ()
    requires_semantic_version_change: bool = False
    rationale: str = ""
    verifier_status: str = "unresolved"
    verifier_confidence: str = "low"
    verifier_rationale: str = ""


@dataclass(frozen=True)
class TraceConflictResolutionOutcome:
    reconciliation: TraceEvidenceReconciliation
    reports: tuple[TraceConflictResolutionReport, ...]

    @property
    def requires_existing_semantic_review(self) -> bool:
        return bool(self.reconciliation.unresolved_disputes)


def resolve_trace_evidence_disputes(
    *, semantic: SemanticStructure, brief: TraceAdjudicationBrief,
    reconciliation: TraceEvidenceReconciliation,
    resolver_client: JSONLLMClient | None,
    verifier_client: JSONLLMClient | None = None,
) -> TraceConflictResolutionOutcome:
    """Retain disputed labels for human review; never ask a model to decide."""
    reports = tuple(
        _unresolved_report(dispute, "Human review required; model adjudication is disabled.")
        for dispute in reconciliation.unresolved_disputes
    )
    return TraceConflictResolutionOutcome(reconciliation, reports)



def _unresolved_report(
    dispute: TraceEvidenceDispute,
    rationale: str,
) -> TraceConflictResolutionReport:
    return TraceConflictResolutionReport(
        resolution_id="TER-" + hashlib.sha256(
            (dispute.dispute_id + rationale).encode("utf-8")
        ).hexdigest()[:20],
        dispute_id=dispute.dispute_id,
        status="unresolved",
        rationale=rationale,
    )

