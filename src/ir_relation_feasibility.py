"""Formula-free root-relation feasibility preflight for finite traces."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Mapping

from contract_versions import (
    IR_RELATION_FEASIBILITY_CONTRACT_VERSION,
    IR_WITNESS_SOLVER_CONTRACT_VERSION,
)
from ir_witness_solver import (
    INVALID_OBLIGATION,
    UNKNOWN_RESOURCE_LIMIT,
    UNSAT_IN_DECLARED_DOMAIN,
    UNSUPPORTED_SEMANTICS,
    WITNESS_FOUND,
    HorizonPolicy,
    WitnessRequest,
    synthesize_ir_witness,
)
from schemas import FaithfulSemanticIR, IRRelationFeasibilityReport
from semantic_node_contracts import structural_horizon
from semantic_feasibility import audit_semantic_feasibility
from witness_domain import WitnessDomainSpec


class IRRelationFeasibilityError(RuntimeError):
    """Typed non-finalizable relation classification."""

    suggested_rerun_stage = "semantic"

    def __init__(
        self,
        code: str,
        message: str,
        *,
        report: IRRelationFeasibilityReport,
        suggested_rerun_stage: str = "semantic",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.report = report
        self.suggested_rerun_stage = suggested_rerun_stage
        self.diagnostics = {
            "reason_code": code,
            "suggested_rerun_stage": suggested_rerun_stage,
            "relation_feasibility_classification": report.classification,
            "feasibility_report_hash": report.artifact_hash,
            "ap_context_fingerprint": report.ap_context_fingerprint,
            "satisfy_status": report.satisfy_status,
            "violate_status": report.violate_status,
            "proof_scope": report.proof_scope,
        }


def analyze_ir_relation_feasibility(
    ir: FaithfulSemanticIR,
    *,
    horizon_policy: HorizonPolicy | None = None,
    max_decisions: int = 250_000,
) -> IRRelationFeasibilityReport:
    """Classify whether the root can be true and false in one declared domain."""

    if horizon_policy is not None:
        policy = horizon_policy
        domain_spec = WitnessDomainSpec(
            minimum_evaluation_horizon=policy.minimum_length,
            minimum_positive_witness_horizon=policy.minimum_length,
            minimum_negative_witness_horizon=policy.minimum_length,
            discriminator_horizon=policy.minimum_length,
            minimum_length=policy.minimum_length,
            maximum_length=policy.maximum_length,
            exact_length=policy.exact_length,
            lengths=policy.lengths,
            basis=("explicit_legacy_horizon_policy",),
        )
    else:
        domain_spec = None
    unified = audit_semantic_feasibility(
        ir,
        domain_spec=domain_spec,
        max_decisions=max_decisions,
    )
    classification = {
        "two_sided_in_domain": "contingent_in_declared_domain",
        "positive_only_in_domain": "valid_in_declared_domain",
        "negative_only_in_domain": "unsatisfiable_in_declared_domain",
        "unknown_resource_limit": "unknown_due_to_resource_limit",
        "unsupported_semantics": "unsupported_semantics",
        "invalid_ir": "unsupported_semantics",
        "bounded_unsat": "unsupported_semantics",
        "stale_contract": "unsupported_semantics",
    }[unified.relation_classification]
    diagnostics = tuple(unified.resource_diagnostics)
    return IRRelationFeasibilityReport(
        ir_version=ir.ir_version,
        ap_context_fingerprint=ir.ap_context_fingerprint,
        horizon_minimum=unified.domain.minimum_length,
        horizon_maximum=unified.domain.maximum_length,
        satisfy_status=unified.positive_status,
        violate_status=unified.negative_status,
        proof_scope=unified.proof_scope,
        classification=classification,
        diagnostics=diagnostics,
        source_supports_non_contingent=_source_supports_non_contingent(
            ir, classification
        ),
        solver_contract_version=IR_WITNESS_SOLVER_CONTRACT_VERSION,
        contract_version=IR_RELATION_FEASIBILITY_CONTRACT_VERSION,
    )


def require_contingent_relation(
    report: IRRelationFeasibilityReport,
) -> IRRelationFeasibilityReport:
    """Route every non-contingent/unknown root to a typed review state."""

    if report.classification == "contingent_in_declared_domain":
        return report
    if report.classification == "unknown_due_to_resource_limit":
        raise IRRelationFeasibilityError(
            "IR_RELATION_FEASIBILITY_RESOURCE_LIMIT",
            "Root relation feasibility is unknown because the bounded solver budget was exhausted.",
            report=report,
            suggested_rerun_stage="trace",
        )
    if report.classification == "unsupported_semantics":
        raise IRRelationFeasibilityError(
            "IR_RELATION_FEASIBILITY_UNSUPPORTED",
            "Root relation feasibility cannot be established for the reviewed semantics.",
            report=report,
        )
    raise IRRelationFeasibilityError(
        "IR_RELATION_NON_CONTINGENT_REVIEW_REQUIRED",
        (
            "Root relation is non-contingent in the declared finite domain; "
            "the realizable side is retained and the item requires semantic approval."
        ),
        report=report,
    )


def _classification(satisfy: str, violate: str) -> str:
    if satisfy == WITNESS_FOUND and violate == WITNESS_FOUND:
        return "contingent_in_declared_domain"
    if satisfy == WITNESS_FOUND and violate == UNSAT_IN_DECLARED_DOMAIN:
        return "valid_in_declared_domain"
    if satisfy == UNSAT_IN_DECLARED_DOMAIN and violate == WITNESS_FOUND:
        return "unsatisfiable_in_declared_domain"
    if UNKNOWN_RESOURCE_LIMIT in {satisfy, violate}:
        return "unknown_due_to_resource_limit"
    if {satisfy, violate} & {UNSUPPORTED_SEMANTICS, INVALID_OBLIGATION}:
        return "unsupported_semantics"
    return "unsupported_semantics"


def _bounded_result(result) -> Mapping[str, object]:
    payload = asdict(result)
    witness = payload.get("witness")
    if witness:
        payload["witness"] = [sorted(step) for step in witness]
    diagnostics = dict(payload.get("diagnostics", {}))
    payload["diagnostics"] = {
        key: diagnostics[key]
        for key in sorted(diagnostics)
        if key not in {"trace", "provider_response"}
    }
    return payload


def _source_supports_non_contingent(
    ir: FaithfulSemanticIR,
    classification: str,
) -> bool:
    node_map = ir.node_map

    def unwrap(node_id: str) -> str:
        node = node_map[node_id]
        while node.node_type in {"global_constraint", "persist_to_end", "identity"} and len(node.children) == 1:
            node_id = node.children[0]
            node = node_map[node_id]
        return node_id

    def signature(node_id: str):
        node = node_map[unwrap(node_id)]
        if node.node_type == "ap_condition":
            return ("ap", str(node.parameter("ap_name", "")))
        return (
            node.node_type,
            tuple(signature(child) for child in node.children),
            tuple(
                (
                    name,
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                )
                for name, value in node.parameters
            ),
        )

    root = node_map[unwrap(ir.root_node_id)]
    if classification == "valid_in_declared_domain":
        return (
            root.node_type in {"implication", "equivalence"}
            and len(root.children) == 2
            and signature(root.children[0]) == signature(root.children[1])
        )
    if classification == "unsatisfiable_in_declared_domain" and root.node_type == "conjunction":
        positives = {signature(child) for child in root.children}
        negatives = {
            signature(node_map[unwrap(child)].children[0])
            for child in root.children
            if node_map[unwrap(child)].node_type == "negation"
            and node_map[unwrap(child)].children
        }
        return bool(positives & negatives)
    return False


__all__ = [
    "IRRelationFeasibilityError",
    "analyze_ir_relation_feasibility",
    "require_contingent_relation",
]
