"""Single semantic feasibility gate before authoritative trace work."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping

from contract_versions import (
    EFFECTIVE_SEMANTIC_SOURCE_CONTRACT_VERSION,
    IR_WITNESS_SOLVER_CONTRACT_VERSION,
    SEMANTIC_FEASIBILITY_CONTRACT_VERSION,
    TEMPORAL_LANGUAGE_CONTRACT_VERSION,
)
from ir_witness_solver import (
    INVALID_OBLIGATION,
    UNKNOWN_RESOURCE_LIMIT,
    UNSAT_IN_DECLARED_DOMAIN,
    UNSUPPORTED_SEMANTICS,
    WITNESS_FOUND,
    WitnessRequest,
    synthesize_ir_witness,
)
from semantic_readiness import CAN_EXECUTE_IR, audit_semantic_readiness
from semantic_node_contracts import node_contract_fingerprint
from witness_domain import (
    WitnessDomainSpec,
    horizon_policy_for_domain,
    plan_witness_domain,
)


@dataclass(frozen=True)
class FeasibilityObligationResult:
    obligation_id: str
    gating: bool
    polarity: str
    minimum_horizon: int
    searched_horizons: tuple[int, ...]
    result: str
    witness_hash: str
    solver_diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class SemanticFeasibilityReport:
    requirement_id: str
    ir_version: str
    readiness_fingerprint: str
    domain: WitnessDomainSpec
    positive_status: str
    positive_witness: tuple[tuple[str, ...], ...]
    negative_status: str
    negative_witness: tuple[tuple[str, ...], ...]
    relation_classification: str
    gating_obligations: tuple[str, ...]
    proof_scope: str
    resource_diagnostics: tuple[tuple[str, Mapping[str, object]], ...]
    evaluator_replay_status: str
    ap_grounding_fingerprint: str
    policy_binding_fingerprints: tuple[str, ...] = ()
    solver_contract_version: str = IR_WITNESS_SOLVER_CONTRACT_VERSION
    contract_version: str = SEMANTIC_FEASIBILITY_CONTRACT_VERSION
    obligation_plan_fingerprint: str = ""
    domain_policy_id: str = "semantic_witness_domain"
    domain_policy_version: str = ""
    per_obligation: tuple[FeasibilityObligationResult, ...] = ()
    root_relation_summary: str = ""
    overall_status: str = ""
    resource_usage: Mapping[str, object] = field(default_factory=dict)
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.relation_classification not in {
            "two_sided_in_domain",
            "positive_only_in_domain",
            "negative_only_in_domain",
            "unknown_resource_limit",
            "unsupported_semantics",
            "invalid_ir",
            "bounded_unsat",
            "stale_contract",
        }:
            raise ValueError("unsupported semantic feasibility classification")
        expected = _report_fingerprint(self)
        if self.fingerprint != expected:
            object.__setattr__(self, "fingerprint", expected)

    @property
    def trace_ready(self) -> bool:
        return self.relation_classification == "two_sided_in_domain"


def audit_semantic_feasibility(
    ir: object,
    *,
    exact_length: int | None = None,
    domain_spec: WitnessDomainSpec | None = None,
    max_decisions: int = 250_000,
    policy_binding_fingerprints: tuple[str, ...] = (),
) -> SemanticFeasibilityReport:
    readiness = audit_semantic_readiness(ir)
    try:
        domain = domain_spec or plan_witness_domain(
            ir,
            exact_length=exact_length,
        )
    except (KeyError, TypeError, ValueError):
        fallback_length = exact_length or 1
        domain = WitnessDomainSpec(
            minimum_evaluation_horizon=fallback_length,
            minimum_positive_witness_horizon=fallback_length,
            minimum_negative_witness_horizon=fallback_length,
            discriminator_horizon=fallback_length,
            minimum_length=fallback_length,
            maximum_length=fallback_length,
            exact_length=exact_length,
            lengths=(fallback_length,),
            basis=("invalid_ir_fallback_domain",),
        )
    if (
        str(getattr(ir, "temporal_language_contract_version", ""))
        not in {"", TEMPORAL_LANGUAGE_CONTRACT_VERSION}
        or str(getattr(ir, "node_contract_fingerprint", ""))
        not in {"", node_contract_fingerprint()}
        or str(getattr(ir, "semantic_source_contract_version", ""))
        not in {"", EFFECTIVE_SEMANTIC_SOURCE_CONTRACT_VERSION}
    ):
        return _nonexecuting_report(
            ir,
            readiness.fingerprint,
            domain,
            "stale_contract",
            policy_binding_fingerprints,
            ("semantic feasibility lineage contract is stale",),
        )
    if not readiness.structurally_valid:
        return _nonexecuting_report(
            ir,
            readiness.fingerprint,
            domain,
            "invalid_ir",
            policy_binding_fingerprints,
            readiness.messages_for(CAN_EXECUTE_IR),
        )
    if not readiness.has_capability(CAN_EXECUTE_IR):
        return _nonexecuting_report(
            ir,
            readiness.fingerprint,
            domain,
            "unsupported_semantics",
            policy_binding_fingerprints,
            readiness.messages_for(CAN_EXECUTE_IR),
        )
    policy = horizon_policy_for_domain(domain)
    positive = synthesize_ir_witness(
        WitnessRequest(
            ir=ir,
            node_id=ir.root_node_id,
            desired_value=True,
            horizon_policy=policy,
            purpose="semantic_feasibility_root_positive",
            max_decisions=max_decisions,
        )
    )
    negative = synthesize_ir_witness(
        WitnessRequest(
            ir=ir,
            node_id=ir.root_node_id,
            desired_value=False,
            horizon_policy=policy,
            purpose="semantic_feasibility_root_negative",
            max_decisions=max_decisions,
        )
    )
    classification = _classification(positive.status, negative.status)
    per_obligation = (
        _obligation_result(
            f"obl-{ir.root_node_id}-sat",
            "positive",
            positive,
            domain,
        ),
        _obligation_result(
            f"obl-{ir.root_node_id}-violate",
            "negative",
            negative,
            domain,
        ),
    )
    obligation_plan_fingerprint = _fingerprint(
        {
            "ir_version": ir.ir_version,
            "obligations": [
                {
                    "obligation_id": item.obligation_id,
                    "gating": item.gating,
                    "polarity": item.polarity,
                    "minimum_horizon": item.minimum_horizon,
                }
                for item in per_obligation
            ],
            "domain_policy_version": domain.policy_version,
        }
    )
    replay = (
        "passed"
        if all(
            result.witness is not None
            for result in (positive, negative)
            if result.status == WITNESS_FOUND
        )
        else "not_applicable"
    )
    return SemanticFeasibilityReport(
        requirement_id=ir.requirement_id,
        ir_version=ir.ir_version,
        readiness_fingerprint=readiness.fingerprint,
        domain=domain,
        positive_status=positive.status,
        positive_witness=_witness(positive.witness),
        negative_status=negative.status,
        negative_witness=_witness(negative.witness),
        relation_classification=classification,
        gating_obligations=(
            f"obl-{ir.root_node_id}-sat",
            f"obl-{ir.root_node_id}-violate",
        ),
        proof_scope="; ".join(
            value
            for value in (positive.proof_scope, negative.proof_scope)
            if value
        ),
        resource_diagnostics=(
            ("positive", _bounded_diagnostics(positive.diagnostics)),
            ("negative", _bounded_diagnostics(negative.diagnostics)),
        ),
        evaluator_replay_status=replay,
        ap_grounding_fingerprint=getattr(
            ir, "ap_grounding_fingerprint", ""
        ),
        policy_binding_fingerprints=policy_binding_fingerprints,
        obligation_plan_fingerprint=obligation_plan_fingerprint,
        domain_policy_version=domain.policy_version,
        per_obligation=per_obligation,
        root_relation_summary=classification,
        overall_status=(
            "feasible"
            if classification == "two_sided_in_domain"
            else classification
        ),
        resource_usage={
            "max_decisions": max_decisions,
            "searched_horizons": list(domain.lengths),
        },
    )


def _nonexecuting_report(
    ir: object,
    readiness_fingerprint: str,
    domain: WitnessDomainSpec,
    classification: str,
    policy_bindings: tuple[str, ...],
    messages: tuple[str, ...],
) -> SemanticFeasibilityReport:
    status = (
        INVALID_OBLIGATION
        if classification == "invalid_ir"
        else (
            "stale_contract"
            if classification == "stale_contract"
            else UNSUPPORTED_SEMANTICS
        )
    )
    obligation_ids = (
        f"obl-{getattr(ir, 'root_node_id', '')}-sat",
        f"obl-{getattr(ir, 'root_node_id', '')}-violate",
    )
    per_obligation = tuple(
        FeasibilityObligationResult(
            obligation_id=obligation_id,
            gating=True,
            polarity=polarity,
            minimum_horizon=domain.minimum_length,
            searched_horizons=domain.lengths,
            result=(
                "invalid_ir"
                if classification == "invalid_ir"
                else (
                    "stale_contract"
                    if classification == "stale_contract"
                    else "unsupported_semantics"
                )
            ),
            witness_hash="",
            solver_diagnostics={"messages": list(messages)},
        )
        for obligation_id, polarity in zip(
            obligation_ids,
            ("positive", "negative"),
        )
    )
    return SemanticFeasibilityReport(
        requirement_id=getattr(ir, "requirement_id", ""),
        ir_version=getattr(ir, "ir_version", ""),
        readiness_fingerprint=readiness_fingerprint,
        domain=domain,
        positive_status=status,
        positive_witness=(),
        negative_status=status,
        negative_witness=(),
        relation_classification=classification,
        gating_obligations=obligation_ids,
        proof_scope="readiness precondition",
        resource_diagnostics=(("readiness", {"messages": list(messages)}),),
        evaluator_replay_status="not_run",
        ap_grounding_fingerprint=getattr(
            ir, "ap_grounding_fingerprint", ""
        ),
        policy_binding_fingerprints=policy_bindings,
        obligation_plan_fingerprint=_fingerprint(
            {
                "ir_version": getattr(ir, "ir_version", ""),
                "obligations": obligation_ids,
                "domain_policy_version": domain.policy_version,
            }
        ),
        domain_policy_version=domain.policy_version,
        per_obligation=per_obligation,
        root_relation_summary=classification,
        overall_status=classification,
    )


def _classification(positive: str, negative: str) -> str:
    if positive == WITNESS_FOUND and negative == WITNESS_FOUND:
        return "two_sided_in_domain"
    if positive == WITNESS_FOUND and negative == UNSAT_IN_DECLARED_DOMAIN:
        return "positive_only_in_domain"
    if positive == UNSAT_IN_DECLARED_DOMAIN and negative == WITNESS_FOUND:
        return "negative_only_in_domain"
    if UNKNOWN_RESOURCE_LIMIT in {positive, negative}:
        return "unknown_resource_limit"
    if INVALID_OBLIGATION in {positive, negative}:
        return "invalid_ir"
    if UNSUPPORTED_SEMANTICS in {positive, negative}:
        return "unsupported_semantics"
    if (
        positive == UNSAT_IN_DECLARED_DOMAIN
        and negative == UNSAT_IN_DECLARED_DOMAIN
    ):
        return "bounded_unsat"
    return "unsupported_semantics"


def _obligation_result(
    obligation_id: str,
    polarity: str,
    result: object,
    domain: WitnessDomainSpec,
) -> FeasibilityObligationResult:
    status = str(getattr(result, "status", ""))
    normalized = {
        WITNESS_FOUND: "feasible",
        UNSAT_IN_DECLARED_DOMAIN: "bounded_unsat",
        UNKNOWN_RESOURCE_LIMIT: "unknown_resource_limit",
        UNSUPPORTED_SEMANTICS: "unsupported_semantics",
        INVALID_OBLIGATION: "invalid_ir",
    }.get(status, "unsupported_semantics")
    witness = getattr(result, "witness", None)
    witness_hash = (
        _fingerprint([sorted(step) for step in witness])
        if witness is not None
        else ""
    )
    return FeasibilityObligationResult(
        obligation_id=obligation_id,
        gating=True,
        polarity=polarity,
        minimum_horizon=domain.minimum_length,
        searched_horizons=domain.lengths,
        result=normalized,
        witness_hash=witness_hash,
        solver_diagnostics=_bounded_diagnostics(
            getattr(result, "diagnostics", {})
        ),
    )


def _witness(
    witness: tuple[frozenset[str], ...] | None,
) -> tuple[tuple[str, ...], ...]:
    return (
        tuple(tuple(sorted(step)) for step in witness)
        if witness is not None
        else ()
    )


def _bounded_diagnostics(
    diagnostics: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        key: value
        for key, value in diagnostics.items()
        if key not in {"trace", "provider_response"}
    }


def _report_fingerprint(report: SemanticFeasibilityReport) -> str:
    payload = asdict(report)
    payload["fingerprint"] = ""
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "FeasibilityObligationResult",
    "SemanticFeasibilityReport",
    "audit_semantic_feasibility",
]
