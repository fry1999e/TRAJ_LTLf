"""Deterministic candidate-IR expansion for unresolved policy-only axes.

The LLM materializes the source-derived graph once.  This module then changes
only backend-registered policy parameters and produces a symmetric set of
complete, executable IR branches.  No branch is authoritative before locked
expert trace evidence selects it.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from typing import Mapping, Sequence

from faithful_ir import compute_ir_version, validate_faithful_ir
from schemas import (
    AmbiguityAxis,
    AmbiguityAxisSet,
    CandidateIRBranch,
    CandidateIRSet,
    FaithfulSemanticIR,
    SemanticStructure,
)
from semantic_policy_binding import (
    SemanticPolicyTargetBinding,
    policy_target_structure_fingerprint,
    validate_semantic_policy_target_binding,
)


POLICY_CANDIDATE_CATALOG_VERSION = "finite-trace-policy-candidates-v1"


class SemanticPolicyBranchingError(ValueError):
    """Raised when legal policy options cannot be expanded without guessing."""


def materialize_policy_candidate_set(
    semantic: SemanticStructure,
    *,
    target_binding: SemanticPolicyTargetBinding,
    options: Sequence[Mapping[str, object]],
) -> SemanticStructure:
    """Expand legal options into complete IR branches with shared structure."""

    base_ir = semantic.faithful_ir
    if base_ir is None:
        raise SemanticPolicyBranchingError(
            "policy candidate expansion requires one materialized seed IR"
        )
    if len(options) < 2:
        raise SemanticPolicyBranchingError(
            "policy candidate expansion requires at least two legal options"
        )
    validate_semantic_policy_target_binding(target_binding, base_ir)
    axis_id = target_binding.axis_id
    node = base_ir.node_map[target_binding.node_id]
    evidence_quote = target_binding.evidence_text
    axis_ids = {
        str(option.get("axis_id", "")).strip() for option in options
    }
    axis_ids.discard("")
    if len(axis_ids) != 1:
        raise SemanticPolicyBranchingError(
            "policy candidate options do not identify exactly one axis"
        )
    # The backend-owned option catalog canonicalizes fuzzy/model-authored axis
    # names from the request.
    axis_id = next(iter(axis_ids))

    branches: list[CandidateIRBranch] = []
    structure_fingerprints: set[str] = set()
    for option in options:
        option_axis = str(option.get("axis_id", "")).strip()
        option_node_type = str(option.get("node_type", "")).strip()
        if option_axis != axis_id or option_node_type != node.node_type:
            raise SemanticPolicyBranchingError(
                "policy option axis/node does not match the requested node"
            )
        profile_id = str(option.get("profile_id", "")).strip()
        if not profile_id:
            raise SemanticPolicyBranchingError("policy option is missing profile_id")
        controlled = tuple(
            str(item).strip()
            for item in option.get("controlled_parameters", ())
            if str(item).strip()
        )
        updates = option.get("parameters")
        if not controlled or not isinstance(updates, Mapping):
            raise SemanticPolicyBranchingError(
                f"policy option {profile_id} lacks a controlled parameter contract"
            )
        unknown_updates = set(updates) - set(controlled)
        if unknown_updates:
            raise SemanticPolicyBranchingError(
                f"policy option {profile_id} updates uncontrolled parameters: "
                + ", ".join(sorted(unknown_updates))
            )
        branch_ir, effective_controlled = _apply_policy_option(
            base_ir,
            node_id=node.node_id,
            updates=updates,
            controlled=controlled,
        )
        structure_fingerprint = policy_target_structure_fingerprint(
            branch_ir,
            node_id=node.node_id,
            controlled=effective_controlled,
        )
        structure_fingerprints.add(structure_fingerprint)
        branch_digest = _digest(
            {
                "source_hash": branch_ir.source_hash,
                "axis_id": axis_id,
                "profile_id": profile_id,
                "profile_version": str(option.get("profile_version", "1")),
                "ir_version": branch_ir.ir_version,
            },
            length=16,
        )
        branches.append(
            CandidateIRBranch(
                ir_branch_id="policy-branch-" + branch_digest,
                faithful_ir=branch_ir,
                supported_text_spans=(evidence_quote or node.evidence_text,),
                ambiguity_axis_ids=(axis_id,),
                semantic_assumptions=(
                    f"finite_trace_policy_candidate:{axis_id}={profile_id}",
                    "candidate is executable but non-authoritative before expert review",
                ),
                plain_language_paraphrase=str(option.get("label", profile_id)),
                generation_reason=(
                    "Backend-expanded legal finite-trace policy candidate; only "
                    "registered controlled parameters differ across branches."
                ),
                applied_alternative_ids=(profile_id,),
                status="needs_expert_review",
                policy_profile_id=profile_id,
                policy_profile_version=str(option.get("profile_version", "1")),
                controlled_parameter_paths=tuple(
                    f"/faithful_ir/nodes/{node.node_id}/parameters/{name}"
                    for name in effective_controlled
                ),
                structure_fingerprint=structure_fingerprint,
                policy_target_binding_id=target_binding.binding_id,
            )
        )

    if len(structure_fingerprints) != 1:
        raise SemanticPolicyBranchingError(
            "policy candidate expansion changed source-derived IR structure"
        )
    common_fingerprint = next(iter(structure_fingerprints))
    branch_set_id = "PCSET-" + _digest(
        {
            "requirement_id": base_ir.requirement_id,
            "source_hash": base_ir.source_hash,
            "reading_review_id": base_ir.reading_review_id,
            "axis_id": axis_id,
            "profiles": sorted(branch.policy_profile_id for branch in branches),
            "structure": common_fingerprint,
            "catalog_version": POLICY_CANDIDATE_CATALOG_VERSION,
        },
        length=20,
    )
    candidate_set = CandidateIRSet(
        requirement_id=base_ir.requirement_id,
        nl_text=base_ir.source_text,
        branches=tuple(branches),
        ambiguity_axes=AmbiguityAxisSet(
            requirement_id=base_ir.requirement_id,
            axes=(AmbiguityAxis(
                axis_id=axis_id,
                nl_span=evidence_quote or node.evidence_text,
                possible_readings=tuple(
                    str(option.get("label", option.get("profile_id", "")))
                    for option in options
                ),
                why_ambiguous=(
                    "Finite-trace policy candidate requires expert adjudication."
                ),
                required_trace_roles=("boundary",),
                review_status="pending_expert_policy_review",
                review_basis="finite_trace_policy_candidate_catalog",
            ),),
            source="finite_trace_policy_candidate_catalog",
        ),
        max_branch_count=max(5, len(branches)),
        status="policy_candidates_ready",
        segmentation_id=base_ir.segmentation_id,
        reading_review_id=base_ir.reading_review_id,
        ambiguity_debt=(f"unresolved_ir_choice:{axis_id}",),
        branch_set_id=branch_set_id,
        common_structure_fingerprint=common_fingerprint,
        policy_axis_ids=(axis_id,),
        selection_status="unresolved",
        candidate_catalog_version=POLICY_CANDIDATE_CATALOG_VERSION,
        policy_target_binding_ids=(target_binding.binding_id,),
    )
    from execution_observer import emit
    emit(
        "candidate_set_materialized",
        requirement_id=base_ir.requirement_id,
        ir_version=base_ir.ir_version,
        ir_artifact_hash=base_ir.artifact_hash,
        ap_context_fingerprint=base_ir.ap_context_fingerprint,
        candidate_set_id=branch_set_id,
        branch_count=len(branches),
        entrypoint="semantic_policy_branching",
    )
    # The seed remains only as a compatibility carrier.  The explicit status
    # prevents normal trace/formula stages from treating it as authoritative.
    return replace(
        semantic,
        faithful_ir=branches[0].faithful_ir,
        candidate_ir_set=candidate_set,
        semantic_policy_resolutions=(),
        semantic_resolution=None,
        approved_semantic_ir=None,
        labeled_trace_evidence=None,
        semantic_evidence_gate=None,
        ir_ambiguity_adjudication=None,
        semantic_status="needs_semantic_review",
        persistent_rag_allowed=False,
    )


def is_policy_candidate_semantic(semantic: SemanticStructure | None) -> bool:
    candidate_set = semantic.candidate_ir_set if semantic is not None else None
    return bool(
        candidate_set is not None
        and candidate_set.status == "policy_candidates_ready"
        and candidate_set.selection_status == "unresolved"
    )


def apply_singleton_policy_option(
    semantic: SemanticStructure,
    *,
    target_binding: SemanticPolicyTargetBinding,
    option: Mapping[str, object],
) -> SemanticStructure:
    base_ir = semantic.faithful_ir
    if base_ir is None:
        raise SemanticPolicyBranchingError(
            "singleton policy application requires a seed IR"
        )
    validate_semantic_policy_target_binding(target_binding, base_ir)
    if str(option.get("axis_id", "")) != target_binding.axis_id:
        raise SemanticPolicyBranchingError("singleton policy axis mismatch")
    if str(option.get("node_type", "")) != target_binding.node_type:
        raise SemanticPolicyBranchingError("singleton policy node type mismatch")
    controlled = tuple(
        str(value)
        for value in option.get("controlled_parameters", ())
        if str(value)
    )
    if not set(controlled).issubset(target_binding.controlled_parameter_names):
        raise SemanticPolicyBranchingError(
            "singleton policy controls parameters outside target binding"
        )
    updates = option.get("parameters")
    if not isinstance(updates, Mapping) or not set(updates).issubset(controlled):
        raise SemanticPolicyBranchingError(
            "singleton policy update contract is invalid"
        )
    applied, effective = _apply_policy_option(
        base_ir,
        node_id=target_binding.node_id,
        updates=updates,
        controlled=controlled,
    )
    if policy_target_structure_fingerprint(
        applied,
        node_id=target_binding.node_id,
        controlled=effective,
    ) != target_binding.structure_fingerprint:
        raise SemanticPolicyBranchingError(
            "singleton policy changed non-controlled IR structure"
        )
    return replace(
        semantic,
        faithful_ir=applied,
        semantic_policy_target_bindings=(asdict(target_binding),),
    )


def _apply_policy_option(
    ir: FaithfulSemanticIR,
    *,
    node_id: str,
    updates: Mapping[str, object],
    controlled: tuple[str, ...],
) -> tuple[FaithfulSemanticIR, tuple[str, ...]]:
    node = ir.node_map[node_id]
    parameters = dict(node.parameters)
    parameters.update(dict(updates))
    effective_controlled = list(controlled)
    if (
        node.node_type == "bounded_window"
        and parameters.get("short_trace_policy") == "strong_upper_bound"
    ):
        upper = int(parameters["upper_bound"])
        parameters["minimum_horizon"] = upper + 1
        if "minimum_horizon" not in effective_controlled:
            effective_controlled.append("minimum_horizon")
    patched_node = replace(node, parameters=tuple(parameters.items()))
    patched_ir = replace(
        ir,
        nodes=tuple(
            patched_node if item.node_id == node_id else item for item in ir.nodes
        ),
        ambiguities=(),
        resolution_id="",
        resolution_fingerprint="",
        ir_version="",
        provenance=replace(
            ir.provenance,
            extraction_mode="policy_candidate_branch",
            approved=False,
            approval_source="",
        ),
    )
    patched_ir = replace(patched_ir, ir_version=compute_ir_version(patched_ir))
    validate_faithful_ir(patched_ir)
    return patched_ir, tuple(effective_controlled)


def _digest(payload: object, *, length: int) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:length]


__all__ = [
    "POLICY_CANDIDATE_CATALOG_VERSION",
    "SemanticPolicyBranchingError",
    "apply_singleton_policy_option",
    "is_policy_candidate_semantic",
    "materialize_policy_candidate_set",
]
