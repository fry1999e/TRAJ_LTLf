"""Cross-artifact ambiguity reconciliation for runtime finalization gates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class AmbiguityClosureReport:
    closure_id: str
    status: str
    ir_version: str = ""
    plain_reading_debt: tuple[str, ...] = ()
    candidate_ir_debt: tuple[str, ...] = ()
    faithful_ir_debt: tuple[str, ...] = ()
    approved_ir_debt: tuple[str, ...] = ()
    semantic_challenge_debt: tuple[str, ...] = ()
    operator_resolution_debt: tuple[str, ...] = ()
    evidence_debt: tuple[str, ...] = ()
    inconsistent_axes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "closed"


def compute_ambiguity_closure_report(
    semantic: Any,
    *,
    trace_coverage: Any = None,
    ir_coverage: Any = None,
    operator_resolution: Any = None,
) -> AmbiguityClosureReport:
    """Recompute ambiguity and evidence debt instead of trusting stale booleans."""

    ir = getattr(semantic, "faithful_ir", None)
    candidate_set = getattr(semantic, "candidate_ir_set", None)
    approved = getattr(semantic, "approved_semantic_ir", None)
    reading = getattr(semantic, "plain_reading_review", None)
    evidence = getattr(semantic, "labeled_trace_evidence", None)
    challenge_set = getattr(semantic, "semantic_challenges", None)
    ir_version = str(getattr(ir, "ir_version", "") or "")

    plain_debt: list[str] = []
    plain_axes: dict[str, str] = {}
    if reading is not None:
        for ambiguity in getattr(reading, "ambiguities", ()) or ():
            axis_id = str(getattr(ambiguity, "ambiguity_id", ""))
            status = str(getattr(ambiguity, "decision_status", ""))
            if axis_id:
                plain_axes[axis_id] = status
            if status not in {"reviewer_selected", "resolved", "not_ambiguous"}:
                plain_debt.append(f"plain_reading:{axis_id or '<unknown>'}:{status or 'pending'}")
        overall = str(getattr(reading, "overall_status", ""))
        if overall not in {"no_ambiguity", "plain_reading_reviewed"}:
            plain_debt.append(f"plain_reading_status:{overall or 'missing'}")
        if getattr(reading, "pending_revision_request_ids", ()):
            plain_debt.extend(
                f"plain_reading_revision_pending:{item}"
                for item in reading.pending_revision_request_ids
            )

    candidate_debt = list(getattr(candidate_set, "ambiguity_debt", ()) or ())
    faithful_debt = [
        f"unresolved_ir_ambiguity:{ambiguity.ambiguity_id}"
        for ambiguity in (getattr(ir, "ambiguities", ()) or ())
        if not ambiguity.resolved
    ]
    approved_debt = list(getattr(approved, "remaining_ambiguity_debt", ()) or ())
    challenge_debt = [
        f"pending_semantic_challenge:{challenge.challenge_id}"
        for challenge in (
            getattr(challenge_set, "open_challenges", ()) or ()
        )
    ]
    adjudication = getattr(semantic, "ir_ambiguity_adjudication", None)
    if adjudication is not None and bool(getattr(adjudication, "provisional", False)):
        candidate_debt.append(
            "provisional_ir_selection:"
            + str(getattr(adjudication, "selected_branch_id", "") or "<missing>")
        )
    operator_debt: list[str] = []
    if operator_resolution is not None:
        operator_status = str(getattr(operator_resolution, "status", "missing"))
        if operator_status not in {"complete", "not_required"}:
            operator_debt.append(f"operator_resolution:{operator_status}")
            operator_debt.extend(
                f"operator_resolution_reason:{item}"
                for item in (getattr(operator_resolution, "reasons", ()) or ())
            )
    evidence_prefixes = (
        "missing_nl_",
        "missing_relation_review_",
        "trace_coverage_",
        "ir_coverage_",
    )
    evidence_debt = [
        item
        for item in approved_debt
        if str(item).startswith(evidence_prefixes)
    ]
    approved_debt = [item for item in approved_debt if item not in evidence_debt]

    if evidence is None or str(getattr(evidence, "status", "pending")) != "complete":
        evidence_debt.append("labeled_trace_evidence_incomplete")
    if evidence is not None:
        evidence_debt.extend(
            str(item) for item in (getattr(evidence, "evidence_debt", ()) or ())
        )
    if trace_coverage is not None and not bool(getattr(trace_coverage, "complete", False)):
        evidence_debt.extend(
            str(item)
            for item in (getattr(trace_coverage, "coverage_debt", ()) or ())
        )
        evidence_debt.append("trace_coverage_incomplete")
    if ir_coverage is not None and not bool(getattr(ir_coverage, "complete", False)):
        evidence_debt.extend(
            str(item) for item in (getattr(ir_coverage, "reasons", ()) or ())
        )
        evidence_debt.append("ir_coverage_incomplete")

    inconsistent: list[str] = []
    approved_version = str(getattr(approved, "ir_version", "") or "")
    if approved is not None and approved_version != ir_version:
        inconsistent.append(
            f"stale_approved_ir_version:{approved_version or '<missing>'}!={ir_version or '<missing>'}"
        )
    resolved_approved = set(getattr(approved, "resolved_ambiguity_axes", ()) or ())
    unresolved_ir = {
        ambiguity.ambiguity_id
        for ambiguity in (getattr(ir, "ambiguities", ()) or ())
        if not ambiguity.resolved
    }
    for axis_id in sorted(resolved_approved & unresolved_ir):
        inconsistent.append(f"approved_resolved_but_ir_unresolved:{axis_id}")
    for axis_id, status in sorted(plain_axes.items()):
        if axis_id in resolved_approved and status not in {
            "reviewer_selected",
            "resolved",
            "not_ambiguous",
        }:
            inconsistent.append(
                f"approved_resolved_but_plain_pending:{axis_id}:{status}"
            )

    ambiguity_reasons = tuple(
        dict.fromkeys(
            (
                *plain_debt,
                *candidate_debt,
                *faithful_debt,
                *approved_debt,
                *challenge_debt,
                *operator_debt,
                *inconsistent,
            )
        )
    )
    payload = {
        "ir_version": ir_version,
        "ambiguity": ambiguity_reasons,
        "evidence": tuple(dict.fromkeys(evidence_debt)),
    }
    closure_id = "ambiguity-closure-" + hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return AmbiguityClosureReport(
        closure_id=closure_id,
        status="closed" if not ambiguity_reasons else "inconsistent",
        ir_version=ir_version,
        plain_reading_debt=tuple(dict.fromkeys(plain_debt)),
        candidate_ir_debt=tuple(dict.fromkeys(candidate_debt)),
        faithful_ir_debt=tuple(dict.fromkeys(faithful_debt)),
        approved_ir_debt=tuple(dict.fromkeys(approved_debt)),
        semantic_challenge_debt=tuple(dict.fromkeys(challenge_debt)),
        operator_resolution_debt=tuple(dict.fromkeys(operator_debt)),
        evidence_debt=tuple(dict.fromkeys(evidence_debt)),
        inconsistent_axes=tuple(dict.fromkeys(inconsistent)),
        reasons=ambiguity_reasons,
    )


__all__ = ["AmbiguityClosureReport", "compute_ambiguity_closure_report"]
