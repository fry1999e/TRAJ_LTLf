"""Formal, formula-blind entrypoint for one-main/one-reviewer typed repair."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from semantic_failure_packet import (
    build_semantic_failure_packets,
    semantic_revision_preflight_violations,
)
from typed_ir_patch import (
    TypedPatchError,
    apply_typed_patch_transaction,
    authorized_axes_from_failure_packets,
    patch_schema,
)


_MAIN_PROMPT = """You are the single main semantic repair model in Traj-LTLf.
Use only the formula-blind failure packets and the closed typed choices supplied
by the backend. Select one or more listed targets and one listed value for each.
Do not emit a formula, free-text IR, source span, operation name, or new node ID.
Return only the requested JSON object."""

_REVIEW_PROMPT = """You are the single reviewer for this Traj-LTLf repair.
Review the main model's typed selection against the source-bound, formula-blind
failure packet. Approve only if every selected target/value follows the source
evidence. You may not propose a replacement formula, node, span, or operation.
Return only the requested JSON object."""


@dataclass(frozen=True)
class FormalTypedPatchResult:
    committed: bool
    ir: object
    evidence: dict


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def _main_schema(ir, axes):
    values = sorted({value for axis in axes for value in axis.get("values", ())})
    return {
        "$id": "traj_ltlf_typed_patch_selection_v1",
        "type": "object",
        "additionalProperties": False,
        "required": ["base_ir_hash", "selections"],
        "properties": {
            "base_ir_hash": {"type": "string", "enum": [ir.artifact_hash]},
            "selections": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["target", "value"],
                    "properties": {
                        "target": {"type": "string", "enum": [axis["axis_id"] for axis in axes]},
                        "value": {"type": "string", "enum": values},
                    },
                },
            },
        },
    }


def _review_schema(axis_ids):
    return {
        "$id": "traj_ltlf_typed_patch_review_v1",
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "reviewed_targets", "rationale"],
        "properties": {
            "decision": {"type": "string", "enum": ["approve", "reject"]},
            "reviewed_targets": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(axis_ids)},
                "uniqueItems": True,
            },
            "rationale": {"type": "string", "minLength": 1},
        },
    }


def _evidence(base, **updates):
    value = {**base, **updates}
    value["evidence_hash"] = _hash(value)
    return value


def run_formal_typed_patch(
    *,
    source_text,
    input_profile,
    semantic,
    reconciliation=None,
    resolution_reports=(),
    failure_packets=None,
    main_client,
    reviewer_client,
    before_vector,
    conflict_cases,
    regression_cases=(),
):
    """Run the complete S5 repair transaction and retain gate-by-gate evidence."""

    ir = getattr(semantic, "faithful_ir", None)
    base = {
        "workflow_version": "formal-typed-patch-v1",
        "entrypoint": "run_formal_typed_patch",
        "model_role_policy": "one_main_model_plus_one_reviewer",
        "formula_exposed": False,
        "applicable": ir is not None,
        "packet_created": False,
        "patch_sent": False,
        "review_sent": False,
        "reviewer_approved": False,
        "committed": False,
        "base_ir_hash": getattr(ir, "artifact_hash", None),
    }
    if ir is None:
        return FormalTypedPatchResult(False, None, _evidence(base, reason_code="PATCH_NOT_REACHABLE_FRONTEND_BLOCKED"))

    try:
        packets = tuple(failure_packets or ())
        if not packets:
            if reconciliation is None:
                return FormalTypedPatchResult(False, ir, _evidence(
                    base, reason_code="PATCH_NO_FAILURE_PACKET"
                ))
            packets = build_semantic_failure_packets(
                source_text=source_text,
                input_profile=input_profile,
                semantic=semantic,
                reconciliation=reconciliation,
                resolution_reports=resolution_reports,
            )
        base.update(
            packet_created=bool(packets),
            packet_ids=[packet["packet_id"] for packet in packets],
            packet_hashes=[packet["artifact_hash"] for packet in packets],
        )
        violations = semantic_revision_preflight_violations(
            semantic, packets, input_profile=input_profile
        )
        if violations:
            return FormalTypedPatchResult(False, ir, _evidence(
                base,
                reason_code="PATCH_PREFLIGHT_FAILED",
                preflight_violations=list(violations),
            ))
        axes = authorized_axes_from_failure_packets(ir, packets)
        closed_schema = patch_schema(ir, axes)
        if not axes:
            return FormalTypedPatchResult(False, ir, _evidence(base, reason_code="PATCH_NO_AUTHORIZED_AXIS"))
        closed_by_id = {item["axis_id"]: item for item in closed_schema["operations"]}
        selection_axes = tuple(
            {**axis, "values": closed_by_id[axis["axis_id"]]["values"]}
            for axis in axes
        )
        base.update(
            authorized_axis_ids=[axis["axis_id"] for axis in axes],
            closed_patch_schema_hash=_hash(closed_schema),
        )
        if main_client is None:
            return FormalTypedPatchResult(False, ir, _evidence(base, reason_code="PATCH_MAIN_MODEL_MISSING"))
        selection = main_client.complete_json(
            _MAIN_PROMPT,
            {
                "failure_packets": list(packets),
                "closed_choices": closed_schema,
            },
            temperature=0.0,
            response_schema=_main_schema(ir, selection_axes),
        )
        base["patch_sent"] = True
        if not isinstance(selection, dict) or selection.get("base_ir_hash") != ir.artifact_hash:
            raise TypedPatchError("main selection is stale or malformed")
        selections = selection.get("selections")
        if not isinstance(selections, list) or not selections:
            raise TypedPatchError("main selection is empty")
        axes_by_id = {axis["axis_id"]: axis for axis in selection_axes}
        chosen = []
        seen = set()
        for item in selections:
            if not isinstance(item, dict) or set(item) != {"target", "value"}:
                raise TypedPatchError("main selection has unexpected fields")
            target = item["target"]
            if target not in axes_by_id or target in seen:
                raise TypedPatchError("main selected an unauthorized or duplicate target")
            axis = axes_by_id[target]
            if item["value"] not in axis["values"]:
                raise TypedPatchError("main selected an illegal enum")
            seen.add(target)
            chosen.append(item)
        base["main_selection_hash"] = _hash(selection)
        if reviewer_client is None:
            return FormalTypedPatchResult(False, ir, _evidence(base, reason_code="PATCH_REVIEWER_MISSING"))
        review = reviewer_client.complete_json(
            _REVIEW_PROMPT,
            {
                "source_text": source_text,
                "failure_packets": list(packets),
                "main_selection": selection,
                "closed_choices": closed_schema,
            },
            temperature=0.0,
            response_schema=_review_schema(seen),
        )
        base["review_sent"] = True
        base["review_hash"] = _hash(review)
        if (
            not isinstance(review, dict)
            or set(review) != {"decision", "reviewed_targets", "rationale"}
            or review.get("decision") != "approve"
            or set(review.get("reviewed_targets", ())) != seen
            or not isinstance(review.get("rationale"), str)
            or not review["rationale"].strip()
        ):
            return FormalTypedPatchResult(False, ir, _evidence(base, reason_code="PATCH_REVIEW_REJECTED"))
        base["reviewer_approved"] = True

        patch = {
            "base_ir_hash": ir.artifact_hash,
            "operations": [
                {
                    "op": axes_by_id[item["target"]]["operation"],
                    "target": item["target"],
                    "value": item["value"],
                    "source_spans": axes_by_id[item["target"]]["source_spans"],
                    "contract_fields": [],
                }
                for item in chosen
            ],
        }
        transaction = apply_typed_patch_transaction(
            ir,
            patch,
            axes,
            before_vector=before_vector,
            conflict_cases=tuple(conflict_cases),
            regression_cases=tuple(regression_cases),
        )
        evidence = _evidence(
            base,
            committed=True,
            reason_code=transaction.reason_code,
            final_ir_hash=transaction.ir.artifact_hash,
            before_conflicts=transaction.before_conflicts,
            after_conflicts=transaction.after_conflicts,
            replayed_conflict_count=transaction.replayed_conflict_count,
            replayed_regression_count=transaction.replayed_regression_count,
        )
        from execution_observer import emit
        emit(
            "formal_typed_patch_completed",
            requirement_id=ir.requirement_id,
            base_ir_artifact_hash=ir.artifact_hash,
            ir_artifact_hash=transaction.ir.artifact_hash,
            evidence_hash=evidence["evidence_hash"],
            entrypoint="run_formal_typed_patch",
        )
        return FormalTypedPatchResult(True, transaction.ir, evidence)
    except TypedPatchError as exc:
        return FormalTypedPatchResult(False, ir, _evidence(
            base,
            reason_code="PATCH_TRANSACTION_REJECTED",
            error_type=type(exc).__name__,
            error=str(exc),
        ))
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        return FormalTypedPatchResult(False, ir, _evidence(
            base,
            reason_code="PATCH_ENTRY_INVALID",
            error_type=type(exc).__name__,
            error=str(exc),
        ))


__all__ = ["FormalTypedPatchResult", "run_formal_typed_patch"]
