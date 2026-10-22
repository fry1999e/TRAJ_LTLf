"""Fresh, context-isolated approval for feedback-rerun candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable

from schemas import FinalApprovalRecord


FRESH_APPROVAL_PROMPT_VERSION = "fresh-independent-approval-v1"


@dataclass(frozen=True)
class IndependentApprovalDecision:
    approval_id: str
    reviewer_id: str
    independence_id: str
    source_run_id: str
    candidate_id: str
    approved: bool
    context_isolated: bool
    candidate_information_exposed: bool
    gold_exposed: bool
    rationale: tuple[str, ...] = ()
    created_at: str = ""


def feedback_candidate_is_approval_ready(result: Any, delta_report: Any) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if str(getattr(result, "run_origin", "")) != "feedback_rerun":
        reasons.append("result is not a feedback rerun")
    candidate = getattr(result, "provisional_candidate", None)
    if candidate is None:
        reasons.append("feedback rerun has no provisional candidate")
    delta_status = str(getattr(delta_report, "status", ""))
    if delta_status not in {"effective_delta", "needs_independent_approval"}:
        reasons.append("rerun delta is not effective or approval-only")
    closure = getattr(result, "formula_closure", None)
    if closure is None or not bool(getattr(closure, "passed", False)):
        reasons.append("FormulaClosureReport did not pass")
    ambiguity = getattr(result, "ambiguity_closure", None)
    if ambiguity is None or not bool(getattr(ambiguity, "passed", False)):
        reasons.append("AmbiguityClosureReport did not pass")
    if str(getattr(result, "evidence_status", "")) != "complete":
        reasons.append("evidence_status is incomplete")
    coverage = getattr(result, "ir_coverage", None)
    if coverage is None or not bool(getattr(coverage, "complete", False)):
        reasons.append("IR coverage is incomplete")
    feedback_patch = getattr(result, "feedback_patch", None)
    if feedback_patch is not None and str(
        getattr(feedback_patch, "review_status", "")
    ) != "approved":
        reasons.append("feedback artifact patch did not pass owning-stage review")
    if candidate is not None and not (
        bool(getattr(candidate, "positive_pass", False))
        and bool(getattr(candidate, "negative_pass", False))
        and bool(getattr(candidate, "boundary_pass", False))
    ):
        reasons.append("candidate does not strictly match every reviewed trace")
    return not reasons, tuple(reasons)


def request_fresh_independent_approval(
    *,
    result: Any,
    delta_report: Any,
    judge_client: Any,
    prior_independence_ids: Iterable[str] = (),
) -> IndependentApprovalDecision | None:
    ready, reasons = feedback_candidate_is_approval_ready(result, delta_report)
    if not ready or judge_client is None or not hasattr(judge_client, "complete_json"):
        return None
    candidate = result.provisional_candidate
    ir = result.semantic.faithful_ir
    payload = {
        "task": "fresh_feedback_candidate_approval",
        "prompt_version": FRESH_APPROVAL_PROMPT_VERSION,
        "source_run_id": result.source_run_id,
        "requirement_id": ir.requirement_id,
        "natural_language_requirement": result.requirement,
        "approved_ir": {
            "ir_version": ir.ir_version,
            "approved": bool(result.semantic.approved_semantic_ir.approved),
            "remaining_ambiguity_debt": list(
                result.semantic.approved_semantic_ir.remaining_ambiguity_debt
            ),
        },
        "candidate": {
            "candidate_id": candidate.candidate_id or result.formula_closure.formula_hash,
            "formula": str(candidate.formula),
            "formula_hash": result.formula_closure.formula_hash,
        },
        "mechanical_gate": {
            "formula_closure_status": result.formula_closure.status,
            "evidence_status": result.evidence_status,
            "ir_coverage_status": result.ir_coverage.status,
            "ambiguity_closure_status": result.ambiguity_closure.status,
            "strict_consistency": True,
            "rerun_delta_status": delta_report.status,
            "rerun_id": delta_report.rerun_id,
        },
        "review_constraints": {
            "generation_context_included": False,
            "prior_candidate_scores_included": False,
            "gold_or_reference_formula_included": False,
            "long_term_learning_permission_requested": False,
        },
        "required_output": {
            "approved": "boolean",
            "reviewer_id": "stable reviewer role",
            "independence_id": "fresh unique review instance id",
            "context_isolated": True,
            "gold_exposed": False,
            "rationale": ["short gate-oriented reasons"],
        },
    }
    raw = judge_client.complete_json(
        (
            "You are a fresh independent release approver. Review only the NL, "
            "approved IR identity, candidate, and mechanical gate summary in this "
            "request. You did not generate the candidate. Do not use or request a "
            "gold/reference formula and do not propose a replacement formula."
        ),
        payload,
        max_tokens=1200,
    )
    if not isinstance(raw, dict):
        return None
    reviewer_id = str(raw.get("reviewer_id", "")).strip()
    independence_id = str(raw.get("independence_id", "")).strip()
    context_isolated = raw.get("context_isolated") is True
    gold_exposed = raw.get("gold_exposed") is True
    approved = raw.get("approved") is True
    forbidden_ids = {str(item) for item in prior_independence_ids if str(item)}
    if (
        not reviewer_id
        or not independence_id
        or independence_id in forbidden_ids
        or not context_isolated
        or gold_exposed
    ):
        approved = False
    rationale_value = raw.get("rationale", ())
    if isinstance(rationale_value, str):
        rationale = (rationale_value.strip(),) if rationale_value.strip() else ()
    elif isinstance(rationale_value, (list, tuple)):
        rationale = tuple(str(item).strip() for item in rationale_value if str(item).strip())
    else:
        rationale = ()
    identity_payload = {
        "run": result.source_run_id,
        "candidate": candidate.candidate_id or result.formula_closure.formula_hash,
        "independence": independence_id,
        "rerun": delta_report.rerun_id,
    }
    approval_id = "independent-approval-" + hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return IndependentApprovalDecision(
        approval_id=approval_id,
        reviewer_id=reviewer_id,
        independence_id=independence_id,
        source_run_id=result.source_run_id,
        candidate_id=candidate.candidate_id or result.formula_closure.formula_hash,
        approved=approved,
        context_isolated=context_isolated,
        candidate_information_exposed=True,
        gold_exposed=gold_exposed,
        rationale=rationale or reasons,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def final_approval_from_independent_decision(
    *,
    result: Any,
    delta_report: Any,
    decision: IndependentApprovalDecision,
) -> FinalApprovalRecord:
    ready, reasons = feedback_candidate_is_approval_ready(result, delta_report)
    if not ready:
        raise ValueError("feedback candidate is not approval-ready: " + "; ".join(reasons))
    if not decision.approved:
        raise ValueError("fresh independent reviewer rejected the candidate")
    if decision.source_run_id != result.source_run_id:
        raise ValueError("independent approval is bound to a different source run")
    candidate = result.provisional_candidate
    return FinalApprovalRecord(
        approval_id=decision.approval_id,
        reviewer=decision.reviewer_id,
        approved_at=decision.created_at,
        approved_ir_version=result.semantic.faithful_ir.ir_version,
        approved_candidate_id=decision.candidate_id,
        approved_formula=str(candidate.formula),
        approval_basis=(
            "fresh independent feedback-rerun approval",
            *decision.rationale,
        ),
        ir_coverage_status=result.ir_coverage.status,
        consistency_status="strict_pass",
        writeback_allowed=False,
        approval_status="final",
        independence_id=decision.independence_id,
        source_run_id=result.source_run_id,
        rerun_delta_id=delta_report.rerun_id,
        formula_closure_status=result.formula_closure.status,
        context_isolated=decision.context_isolated,
        gold_exposed=decision.gold_exposed,
    )


__all__ = [
    "FRESH_APPROVAL_PROMPT_VERSION",
    "IndependentApprovalDecision",
    "feedback_candidate_is_approval_ready",
    "final_approval_from_independent_decision",
    "request_fresh_independent_approval",
]
