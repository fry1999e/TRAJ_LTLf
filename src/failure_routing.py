"""Single registry for live and persisted pipeline failure disposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from contract_versions import FAILURE_ROUTING_CONTRACT_VERSION


SEMANTIC_DISCRIMINATION_READY = "semantic_discrimination_ready"
AWAITING_PLAIN_READING_RESOLUTION = "awaiting_plain_reading_resolution"
AWAITING_SEMANTIC_RESOLUTION = "awaiting_semantic_resolution"
AWAITING_CANDIDATE_ADJUDICATION = "awaiting_candidate_adjudication"
BLOCKED_INVALID_ARTIFACT = "blocked_invalid_artifact"
BLOCKED_EXECUTOR_CONTRACT = "blocked_executor_contract"
BLOCKED_STALE_ARTIFACT = "blocked_stale_artifact"
FAILED_INTERNAL = "failed_internal"

# Compatibility names for callers being migrated.  They no longer describe a
# second routing taxonomy.
NEED_VIEW = AWAITING_SEMANTIC_RESOLUTION
RETRYABLE_GENERATION = FAILED_INTERNAL
BLOCKED_ERROR = FAILED_INTERNAL


@dataclass(frozen=True)
class FailureRoute:
    reason_code: str
    disposition: str
    owner_stage: str
    safe_restart_stage: str
    retryable: bool
    expected_pending: bool = False
    contract_version: str = FAILURE_ROUTING_CONTRACT_VERSION


_FAILURE_REGISTRY: Mapping[str, FailureRoute] = {
    "PLAIN_READING_SEMANTIC_RESOLUTION_REQUIRED": FailureRoute(
        "PLAIN_READING_SEMANTIC_RESOLUTION_REQUIRED",
        AWAITING_PLAIN_READING_RESOLUTION,
        "plain_reading",
        "plain_reading",
        False,
        True,
    ),
    "SEMANTIC_POLICY_RESOLUTION_REQUIRED": FailureRoute(
        "SEMANTIC_POLICY_RESOLUTION_REQUIRED",
        AWAITING_SEMANTIC_RESOLUTION,
        "semantic",
        "semantic",
        False,
        True,
    ),
    "IR_ROOT_RELATION_ONE_SIDED_IN_DOMAIN": FailureRoute(
        "IR_ROOT_RELATION_ONE_SIDED_IN_DOMAIN",
        AWAITING_CANDIDATE_ADJUDICATION,
        "semantic_feasibility",
        "semantic",
        False,
        True,
    ),
    "IR_WITNESS_UNSAT_IN_DOMAIN": FailureRoute(
        "IR_WITNESS_UNSAT_IN_DOMAIN",
        AWAITING_SEMANTIC_RESOLUTION,
        "semantic_feasibility",
        "semantic",
        False,
        True,
    ),
    "IR_WITNESS_RESOURCE_LIMIT": FailureRoute(
        "IR_WITNESS_RESOURCE_LIMIT",
        RETRYABLE_GENERATION,
        "semantic_feasibility",
        "semantic_feasibility",
        True,
    ),
    "LLM_GENERATION_FAILED": FailureRoute(
        "LLM_GENERATION_FAILED",
        RETRYABLE_GENERATION,
        "generation",
        "generation",
        True,
    ),
    "AP_GROUNDING_CONFLICT": FailureRoute(
        "AP_GROUNDING_CONFLICT",
        AWAITING_SEMANTIC_RESOLUTION,
        "ap_grounding",
        "pre_ir",
        False,
        True,
    ),
    "EXECUTOR_UNSUPPORTED": FailureRoute(
        "EXECUTOR_UNSUPPORTED",
        BLOCKED_EXECUTOR_CONTRACT,
        "semantic",
        "semantic",
        False,
    ),
    "CHECKPOINT_IO_ERROR": FailureRoute(
        "CHECKPOINT_IO_ERROR",
        FAILED_INTERNAL,
        "checkpoint",
        "checkpoint",
        True,
    ),
    "INTERNAL_PIPELINE_ERROR": FailureRoute(
        "INTERNAL_PIPELINE_ERROR",
        FAILED_INTERNAL,
        "pipeline",
        "pipeline",
        False,
    ),
    "PLAIN_READING_NEED_VIEW": FailureRoute(
        "PLAIN_READING_NEED_VIEW",
        AWAITING_PLAIN_READING_RESOLUTION,
        "plain_reading",
        "plain_reading",
        False,
        True,
    ),
    "SEMANTIC_READINESS_PENDING": FailureRoute(
        "SEMANTIC_READINESS_PENDING",
        SEMANTIC_DISCRIMINATION_READY,
        "semantic",
        "semantic",
        False,
        True,
    ),
    "SEMANTIC_CANDIDATE_ADJUDICATION_REQUIRED": FailureRoute(
        "SEMANTIC_CANDIDATE_ADJUDICATION_REQUIRED",
        AWAITING_CANDIDATE_ADJUDICATION,
        "semantic",
        "semantic",
        False,
        True,
    ),
    "EFFECTIVE_SEMANTIC_SOURCE_BINDING_INVALID": FailureRoute(
        "EFFECTIVE_SEMANTIC_SOURCE_BINDING_INVALID",
        BLOCKED_STALE_ARTIFACT,
        "semantic_source",
        "semantic",
        False,
    ),
    "SEMANTIC_POLICY_TARGET_BINDING_INVALID": FailureRoute(
        "SEMANTIC_POLICY_TARGET_BINDING_INVALID",
        BLOCKED_STALE_ARTIFACT,
        "semantic_policy",
        "semantic",
        False,
    ),
    "SEMANTIC_FEASIBILITY_INVALID": FailureRoute(
        "SEMANTIC_FEASIBILITY_INVALID",
        BLOCKED_INVALID_ARTIFACT,
        "semantic_feasibility",
        "semantic",
        False,
    ),
    "SEMANTIC_FEASIBILITY_STALE_CONTRACT": FailureRoute(
        "SEMANTIC_FEASIBILITY_STALE_CONTRACT",
        BLOCKED_STALE_ARTIFACT,
        "semantic_feasibility",
        "semantic",
        False,
    ),
}


def failure_route_for_code(reason_code: str) -> FailureRoute:
    normalized = reason_code.strip().upper()
    route = _FAILURE_REGISTRY.get(normalized)
    if route is not None:
        return route
    return FailureRoute(
        normalized or "INTERNAL_PIPELINE_ERROR",
        FAILED_INTERNAL,
        "pipeline",
        "pipeline",
        False,
    )


def classify_failure(value: object) -> FailureRoute:
    code = str(
        getattr(value, "code", "")
        or getattr(value, "reason_code", "")
        or (
            value.get("reason_code", "")
            if isinstance(value, Mapping)
            else ""
        )
    )
    return failure_route_for_code(code)


def failure_registry() -> Mapping[str, FailureRoute]:
    return dict(_FAILURE_REGISTRY)


__all__ = [
    "BLOCKED_ERROR",
    "BLOCKED_EXECUTOR_CONTRACT",
    "BLOCKED_INVALID_ARTIFACT",
    "BLOCKED_STALE_ARTIFACT",
    "FAILED_INTERNAL",
    "NEED_VIEW",
    "RETRYABLE_GENERATION",
    "AWAITING_CANDIDATE_ADJUDICATION",
    "AWAITING_PLAIN_READING_RESOLUTION",
    "AWAITING_SEMANTIC_RESOLUTION",
    "SEMANTIC_DISCRIMINATION_READY",
    "FailureRoute",
    "classify_failure",
    "failure_registry",
    "failure_route_for_code",
]
