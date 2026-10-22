"""Structured guided-stage keep/fallback decisions.

Warmup guided runs have expected artifacts for each stage.  This module keeps
the audit record format consistent across stages while avoiding hidden
chain-of-thought: the recorded rationale is a short, reviewable reason and a
reusable lesson for RAG.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from schemas import GuidedCorrectionRecord


_UNSET = object()


def build_guided_stage_decision(
    *,
    stage: str,
    requirement_id: str,
    requirement: str,
    round_index: int,
    generated_artifact: Any,
    expected_artifact: Any,
    mismatch_evidence: list[str] | tuple[str, ...],
    llm_client: object | None = None,
    round_2_artifact: Any = _UNSET,
    round_2_verification_status: str | None = None,
    artifact_origin: str | None = None,
) -> dict:
    """Return the audit record for autonomous attempt 1 and guided attempt 2.

    The optional parameters preserve the legacy call shape while allowing the
    caller to record the artifact actually produced during the second attempt.
    """

    evidence = [str(item) for item in mismatch_evidence if str(item).strip()]
    has_expected = expected_artifact is not None
    matched = has_expected and not evidence
    has_real_round_2 = round_2_artifact is not _UNSET
    teacher_adjudicated_round_1 = artifact_origin == "teacher_adjudicated_round_1"
    teacher_adjudicated_round_2 = artifact_origin == "teacher_adjudicated_round_2"
    if matched:
        status = "matched_expected"
        decision = (
            "adopt_expected_decisions_on_matching_round_1_proposal"
            if teacher_adjudicated_round_1
            else "keep_generated"
        )
        resolved_origin = artifact_origin or "generated_round_1"
    elif not has_expected:
        status = "no_expected_available"
        decision = "keep_generated"
        resolved_origin = artifact_origin or "generated_round_1"
    elif has_real_round_2:
        resolved_origin = artifact_origin or "generated_round_2"
        if resolved_origin in {"generated_round_2", "teacher_adjudicated_round_2"}:
            status = "corrected_on_round_2"
            decision = (
                "adopt_expected_decisions_on_matching_round_2_proposal"
                if teacher_adjudicated_round_2
                else "keep_round_2_generated"
            )
        else:
            status = "fallback_to_expected"
            decision = "discard_round_2_adopt_expected"
    else:
        # Compatibility for old callers that still perform a direct fallback.
        status = "fallback_to_expected"
        decision = "discard_generated_adopt_expected"
        resolved_origin = artifact_origin or "expected_fallback"
    rationale = _decision_rationale(
        stage=stage,
        requirement=requirement,
        status=status,
        decision=decision,
        mismatch_evidence=evidence,
        llm_client=llm_client,
    )
    correction_record = None
    if status in {"fallback_to_expected", "corrected_on_round_2"}:
        actual_round_2 = (
            expected_artifact if round_2_artifact is _UNSET else round_2_artifact
        )
        verification_status = round_2_verification_status or "passed"
        correction_record = GuidedCorrectionRecord(
            stage=_normalize_correction_stage(stage),
            round_1_artifact=generated_artifact,
            expected_artifact=expected_artifact,
            mismatch_evidence=tuple(evidence),
            root_cause_category=_root_cause_category(stage, evidence),
            root_cause_explanation=_root_cause_explanation(stage, evidence),
            correction_action=(
                "round_2_teacher_guided_regeneration_and_adjudication"
                if resolved_origin
                in {"generated_round_2", "teacher_adjudicated_round_2"}
                else "round_2_teacher_guided_then_expected_adoption"
            ),
            round_2_artifact=actual_round_2,
            verification_status=verification_status,
        )
    return {
        "stage": stage,
        "requirement_id": requirement_id,
        "round": round_index,
        "attempt": 2 if has_real_round_2 else 1,
        "attempt_count": 2 if has_real_round_2 else 1,
        "status": status,
        "artifact_origin": resolved_origin,
        "fallback_used": resolved_origin.startswith("expected"),
        "verification_status": (
            round_2_verification_status
            or ("passed" if has_expected else "not_applicable")
        ),
        "generated_artifact": generated_artifact,
        "round_1_artifact": generated_artifact,
        "round_2_artifact": (
            None if round_2_artifact is _UNSET else round_2_artifact
        ),
        "expected_artifact": expected_artifact,
        "mismatch_evidence": evidence,
        "guided_correction": (
            asdict(correction_record) if correction_record is not None else None
        ),
        "decision": decision,
        "decision_reason": rationale["decision_reason"],
        "lesson_for_rag": rationale["lesson_for_rag"],
        "rationale_source": rationale["rationale_source"],
    }


def _normalize_correction_stage(stage: str) -> str:
    normalized = stage.strip().lower().replace("-", "_")
    if normalized in {"traces", "trace_generation"}:
        return "trace"
    if normalized == "hole_candidate_generation":
        return "template"
    if normalized == "temporal_hole_completion":
        return "completion"
    return normalized


def _root_cause_category(stage: str, evidence: list[str]) -> str:
    text = "\n".join(evidence).lower()
    if any(token in text for token in ("ap ", " ap", "atomic", "atom", "unknown ap")):
        return "ap_extraction_error"
    if any(token in text for token in ("scope", "trigger", "response", "component")):
        return "scope_or_component_error"
    if any(token in text for token in ("trace", "count", "kind", "label", "shape")):
        return "trace_coverage_or_label_error"
    if any(token in text for token in ("hole", "template", "h_")):
        return "hole_position_error"
    if any(token in text for token in ("assignment", "operator", "x", "f", "g")):
        return "temporal_operator_error"
    if any(token in text for token in ("candidate", "selected", "ranking")):
        return "candidate_selection_error"
    return _normalize_correction_stage(stage) + "_mismatch"


def _root_cause_explanation(stage: str, evidence: list[str]) -> str:
    if evidence:
        return "; ".join(evidence[:3])
    return (
        "Generated artifact did not satisfy the deterministic warmup expected "
        f"artifact for {_normalize_correction_stage(stage)}."
    )


def _decision_rationale(
    *,
    stage: str,
    requirement: str,
    status: str,
    decision: str,
    mismatch_evidence: list[str],
    llm_client: object | None,
) -> dict:
    complete_json = getattr(llm_client, "complete_json", None)
    if callable(complete_json):
        try:
            raw = complete_json(
                system_prompt=(
                    "You write concise audit rationales for a guided warmup "
                    "pipeline. Return JSON only. Do not include hidden "
                    "chain-of-thought; provide a short decision reason and a "
                    "reusable lesson."
                ),
                user_payload={
                    "task": "guided_stage_decision_rationale",
                    "stage": stage,
                    "requirement": requirement,
                    "status": status,
                    "decision": decision,
                    "mismatch_evidence": mismatch_evidence,
                    "output_schema": {
                        "decision_reason": "one concise sentence",
                        "lesson_for_rag": "one concise reusable lesson",
                    },
                },
                temperature=0,
                max_tokens=300,
            )
            decision_reason = str(raw.get("decision_reason", "")).strip()
            lesson_for_rag = str(raw.get("lesson_for_rag", "")).strip()
            if decision_reason and lesson_for_rag:
                return {
                    "decision_reason": decision_reason,
                    "lesson_for_rag": lesson_for_rag,
                    "rationale_source": "llm",
                }
        except Exception:
            pass

    if status == "matched_expected":
        decision_reason = (
            f"{stage} generated artifact matches the expected warmup artifact, "
            + (
                "so the stage-scoped expected decisions are applied without "
                "human adjudication."
                if decision
                == "adopt_expected_decisions_on_matching_round_1_proposal"
                else "so the generated value is retained."
            )
        )
        lesson_for_rag = (
            f"For similar {stage} tasks, preserve the field/formula/trace choices "
            "that align with the guided expected artifact."
        )
    elif status == "fallback_to_expected":
        summary = "; ".join(mismatch_evidence[:3]) or "generated artifact differed"
        decision_reason = (
            f"{stage} generated artifact is discarded because it differs from "
            f"the expected warmup artifact: {summary}."
        )
        lesson_for_rag = (
            f"For similar {stage} tasks, prefer the expected artifact pattern "
            "when these mismatch signals appear."
        )
    elif status == "corrected_on_round_2":
        summary = "; ".join(mismatch_evidence[:3]) or "round 1 differed"
        decision_reason = (
            f"{stage} was corrected during the teacher-guided second attempt "
            f"after these first-attempt mismatches: {summary}."
            + (
                " The stage-scoped expected decisions were then applied without "
                "human adjudication."
                if decision
                == "adopt_expected_decisions_on_matching_round_2_proposal"
                else ""
            )
        )
        lesson_for_rag = (
            f"For similar {stage} tasks, use the recorded mismatch and corrected "
            "second-attempt artifact as a reusable error/correction pair."
        )
    else:
        decision_reason = (
            f"{stage} has no expected artifact, so the generated value is kept "
            "for the normal review path."
        )
        lesson_for_rag = (
            f"No guided fallback is available for {stage}; rely on review and "
            "deterministic validation."
        )
    return {
        "decision_reason": decision_reason,
        "lesson_for_rag": lesson_for_rag,
        "rationale_source": "local",
    }
