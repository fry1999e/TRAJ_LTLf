"""Mutually exclusive failure routing based on types, codes and JSON pointers."""
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class ValidationRoute:
    domain: str
    code: str
    owner_stage: str
    retry_policy: str
    terminal_status: str
    consumes_semantic_budget: bool = False


def route_validation(error):
    from ap_binding_context import APBindingContextError
    from faithful_ir import FaithfulIRValidationError, FaithfulIRTraceReadinessError
    code = getattr(error, "code", None) or getattr(error, "reason_code", None) or type(error).__name__
    path = getattr(error, "path", "")
    if isinstance(error, (OSError, TimeoutError)) or code in {"PROVIDER_EXECUTION_UNKNOWN", "EXECUTION_UNKNOWN", "UnknownExecution"}:
        return ValidationRoute("infra", code, "transport", "reconcile_before_resend", "I")
    if isinstance(error, APBindingContextError) or code.startswith(("IR_AP_", "AP_")):
        return ValidationRoute("ap_binding", code, "ap_grounding", "grounding_review", "U")
    if isinstance(error, json.JSONDecodeError) or code in {"OUTPUT_TRUNCATED", "PRE_IR_OUTPUT_TRUNCATED", "RESPONSE_SCHEMA_INVALID", "JSON_DECODE_INVALID", "PATCH_SCHEMA_INVALID"}:
        return ValidationRoute("serialization", code, "format", "bounded_format_repair", "I")
    if code.startswith("SOURCE_GRAPH_SCHEMA_INVALID") or code in {
        "ARITY_INVALID", "ROOT_INVALID", "CYCLE",
        "SOURCE_GRAPH_PROPOSAL_ARITY_INVALID",
        "SOURCE_GRAPH_PROPOSAL_ROOT_INVALID",
        "SOURCE_GRAPH_PROPOSAL_CYCLE",
        "SOURCE_GRAPH_PROPOSAL_DISCONNECTED",
    }:
        return ValidationRoute("structure", code, "source_graph", "bounded_representation_repair", "I")
    if code == "VERIFIED_SOURCE_CONTRADICTION":
        return ValidationRoute("source_contradiction", code, "semantic", "bounded_monotone_revision", "U", True)
    if code in {"EVIDENCE_LABEL_CONFLICT", "SAME_TRACE_DIFFERENT_LABEL"}:
        return ValidationRoute("evidence", code, "annotation", "independent_adjudication", "U")
    if (
        isinstance(error, FaithfulIRTraceReadinessError)
        or code in {
            "IR_SEMANTIC_RESOLUTION_REQUIRED",
            "SEMANTIC_REVISION_NO_PROGRESS",
            "CANDIDATE_SET_INCOMPLETE",
            "SOURCE_REVIEW_UNDETERMINED",
            "SEMANTIC_AXIS_UNRESOLVED",
            "SOURCE_UNCLEAR",
            "PATCH_NOT_REACHABLE",
            "PATCH_PREFLIGHT_FAILED",
            "PATCH_NO_PROGRESS",
        }
        or str(code).startswith("CAPABILITY_GAP:")
    ):
        return ValidationRoute("underdetermination", code, "semantic_review", "candidate_or_human", "U")
    if isinstance(error, FaithfulIRValidationError) or path.startswith(("/nodes/", "/faithful_ir/nodes/")):
        return ValidationRoute("structure", code, "ir_builder", "invariant_preserving_repair", "I")
    return ValidationRoute("unclassified", code, "diagnosis", "no_automatic_retry", "I")
