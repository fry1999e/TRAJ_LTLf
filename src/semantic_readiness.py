"""Single-source semantic capability audit for FaithfulSemanticIR.

Structural validity, direct executability, and semantic authority are separate
properties.  This module derives those properties without mutating the IR and
without supplying defaults for unresolved semantic parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping
from contract_versions import SEMANTIC_READINESS_CONTRACT_VERSION
from faithful_ir import validate_faithful_ir
from relation_parameter_contracts import PERIODICITY_PARAMETER_VALUES
from schemas import FaithfulSemanticIR
from semantic_axes import semantic_axes_for_parameter
from semantic_node_contracts import required_node_parameter_names
from temporal_resolution import allowed_operators_for_node, resolved_node_kind


CAN_PERSIST_IR = "persist_ir"
CAN_BUILD_DISCRIMINATION_PLAN = "build_discrimination_plan"
CAN_EXECUTE_IR = "execute_ir"
CAN_SYNTHESIZE_WITNESS = "synthesize_witness"
CAN_BIND_RESOLUTION = "bind_resolution"
CAN_GENERATE_AUTHORITATIVE_TRACES = "generate_authoritative_traces"

_EXECUTION_CAPABILITIES = frozenset(
    {
        CAN_EXECUTE_IR,
        CAN_SYNTHESIZE_WITNESS,
        CAN_BIND_RESOLUTION,
        CAN_GENERATE_AUTHORITATIVE_TRACES,
    }
)
_DIRECT_EXECUTION_CAPABILITIES = frozenset(
    {CAN_EXECUTE_IR, CAN_SYNTHESIZE_WITNESS, CAN_GENERATE_AUTHORITATIVE_TRACES}
)
_RESOLUTION_CAPABILITIES = frozenset(
    {CAN_BIND_RESOLUTION, CAN_GENERATE_AUTHORITATIVE_TRACES}
)


@dataclass(frozen=True)
class SemanticReadinessFinding:
    code: str
    path: str
    message: str
    blocked_capabilities: tuple[str, ...]
    node_id: str = ""
    node_type: str = ""
    parameter_name: str = ""
    ambiguity_id: str = ""
    semantic_axis_ids: tuple[str, ...] = ()
    suggested_rerun_stage: str = "semantic"


@dataclass(frozen=True)
class SemanticReadinessReport:
    ir_version: str
    structurally_valid: bool
    parameter_complete: bool
    directly_executable: bool
    resolution_complete: bool
    capabilities: tuple[str, ...]
    findings: tuple[SemanticReadinessFinding, ...] = ()
    unresolved_ambiguity_ids: tuple[str, ...] = ()
    missing_parameter_paths: tuple[str, ...] = ()
    ap_context_fingerprint: str = ""
    contract_version: str = SEMANTIC_READINESS_CONTRACT_VERSION
    fingerprint: str = ""

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities

    def blockers_for(self, capability: str) -> tuple[SemanticReadinessFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if capability in finding.blocked_capabilities
        )

    def messages_for(self, capability: str) -> tuple[str, ...]:
        return tuple(finding.message for finding in self.blockers_for(capability))


class SemanticReadinessError(ValueError):
    """Typed capability failure that orchestration can route to semantic review."""

    suggested_rerun_stage = "semantic"

    def __init__(
        self,
        capability: str,
        report: SemanticReadinessReport,
        *,
        message: str = "",
    ) -> None:
        self.capability = capability
        self.report = report
        blockers = report.blockers_for(capability)
        detail = "; ".join(item.message for item in blockers)
        super().__init__(
            message
            or f"Semantic IR lacks capability {capability}: "
            + (detail or "readiness contract was not satisfied")
        )
        self.code = "IR_SEMANTIC_CAPABILITY_UNAVAILABLE"
        self.diagnostics = {
            "reason_code": self.code,
            "required_capability": capability,
            "ir_version": report.ir_version,
            "readiness_contract_version": report.contract_version,
            "readiness_fingerprint": report.fingerprint,
            "ap_context_fingerprint": report.ap_context_fingerprint,
            "findings": [asdict(item) for item in blockers],
            "missing_parameter_paths": list(report.missing_parameter_paths),
            "unresolved_ambiguity_ids": list(report.unresolved_ambiguity_ids),
        }


def audit_semantic_readiness(
    ir: FaithfulSemanticIR,
    *,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
) -> SemanticReadinessReport:
    """Derive IR capabilities without choosing or defaulting semantic values."""

    findings: list[SemanticReadinessFinding] = []
    structurally_valid = True
    try:
        validate_faithful_ir(ir)
    except Exception as exc:
        structurally_valid = False
        domain_findings = tuple(getattr(exc, "findings", ()))
        if domain_findings:
            for item in domain_findings:
                findings.append(
                    SemanticReadinessFinding(
                        code=str(getattr(item, "code", "FAITHFUL_IR_VALIDATION_FAILED")),
                        path=str(getattr(item, "path", "/faithful_ir")),
                        message=str(getattr(item, "message", str(exc))),
                        blocked_capabilities=tuple(sorted(_EXECUTION_CAPABILITIES)),
                    )
                )
        else:
            findings.append(
                SemanticReadinessFinding(
                    code=getattr(exc, "code", "FAITHFUL_IR_VALIDATION_FAILED"),
                    path=getattr(exc, "path", "/faithful_ir"),
                    message=str(exc),
                    blocked_capabilities=tuple(sorted(_EXECUTION_CAPABILITIES)),
                )
            )

    if structurally_valid:
        for node in ir.nodes:
            params = dict(node.parameters)
            for name in _required_parameters_for_node(node.node_type, params):
                if name not in params or params[name] in {None, ""}:
                    findings.append(
                        SemanticReadinessFinding(
                            code="IR_REQUIRED_PARAMETER_UNRESOLVED",
                            path=(
                                f"/faithful_ir/nodes/{node.node_id}/parameters/{name}"
                            ),
                            message=(
                                f"{node.node_type} node {node.node_id} is missing "
                                f"reviewed parameter {name}"
                            ),
                            blocked_capabilities=tuple(sorted(_EXECUTION_CAPABILITIES)),
                            node_id=node.node_id,
                            node_type=node.node_type,
                            parameter_name=name,
                            semantic_axis_ids=tuple(
                                axis.axis_id
                                for axis in semantic_axes_for_parameter(
                                    node.node_type,
                                    name,
                                )
                            ),
                        )
                    )
            findings.extend(
                _temporal_resolution_findings(node, temporal_resolution_by_node)
            )
            findings.extend(
                _executor_contract_findings(
                    node,
                    params,
                    temporal_resolution_by_node,
                )
            )

        for ambiguity in ir.ambiguities:
            if ambiguity.resolved:
                continue
            findings.append(
                SemanticReadinessFinding(
                    code="IR_SEMANTIC_AMBIGUITY_UNRESOLVED",
                    path=f"/faithful_ir/ambiguities/{ambiguity.ambiguity_id}/resolved",
                    message=(
                        f"semantic ambiguity {ambiguity.ambiguity_id} remains unresolved"
                    ),
                    blocked_capabilities=tuple(sorted(_RESOLUTION_CAPABILITIES)),
                    ambiguity_id=ambiguity.ambiguity_id,
                )
            )

        if ir.provenance.extraction_mode == "policy_candidate_branch":
            findings.append(
                SemanticReadinessFinding(
                    code="IR_POLICY_CANDIDATE_NOT_SELECTED",
                    path="/faithful_ir/provenance/extraction_mode",
                    message=(
                        "finite-trace policy candidate has not been selected by "
                        "expert evidence"
                    ),
                    blocked_capabilities=tuple(sorted(_RESOLUTION_CAPABILITIES)),
                )
            )

    blocked = {
        capability
        for finding in findings
        for capability in finding.blocked_capabilities
    }
    capabilities: list[str] = []
    if structurally_valid:
        capabilities.extend((CAN_PERSIST_IR, CAN_BUILD_DISCRIMINATION_PLAN))
    for capability in (
        CAN_EXECUTE_IR,
        CAN_SYNTHESIZE_WITNESS,
        CAN_BIND_RESOLUTION,
        CAN_GENERATE_AUTHORITATIVE_TRACES,
    ):
        if structurally_valid and capability not in blocked:
            capabilities.append(capability)

    missing_paths = tuple(
        item.path for item in findings if item.code == "IR_REQUIRED_PARAMETER_UNRESOLVED"
    )
    unresolved_ids = tuple(
        item.ambiguity_id
        for item in findings
        if item.code == "IR_SEMANTIC_AMBIGUITY_UNRESOLVED"
    )
    parameter_complete = structurally_valid and not missing_paths
    directly_executable = CAN_EXECUTE_IR in capabilities
    resolution_complete = CAN_BIND_RESOLUTION in capabilities
    payload = {
        "contract_version": SEMANTIC_READINESS_CONTRACT_VERSION,
        "ir_version": ir.ir_version,
        "ap_context_fingerprint": ir.ap_context_fingerprint,
        "structurally_valid": structurally_valid,
        "capabilities": capabilities,
        "findings": [asdict(item) for item in findings],
        "temporal_resolution_by_node": sorted(
            (temporal_resolution_by_node or {}).items()
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return SemanticReadinessReport(
        ir_version=ir.ir_version,
        structurally_valid=structurally_valid,
        parameter_complete=parameter_complete,
        directly_executable=directly_executable,
        resolution_complete=resolution_complete,
        capabilities=tuple(capabilities),
        findings=tuple(findings),
        unresolved_ambiguity_ids=unresolved_ids,
        missing_parameter_paths=missing_paths,
        ap_context_fingerprint=ir.ap_context_fingerprint,
        fingerprint=fingerprint,
    )


def require_semantic_capability(
    ir: FaithfulSemanticIR,
    capability: str,
    *,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
) -> SemanticReadinessReport:
    report = audit_semantic_readiness(
        ir,
        temporal_resolution_by_node=temporal_resolution_by_node,
    )
    if not report.has_capability(capability):
        raise SemanticReadinessError(capability, report)
    return report


def _required_parameters_for_node(
    node_type: str,
    params: dict[str, object],
) -> tuple[str, ...]:
    required = list(required_node_parameter_names(node_type))
    if node_type == "periodicity":
        mode = str(params.get("mode", "")).strip()
        if mode == "fixed_phase":
            required.extend(("phase", "off_schedule_policy"))
        elif mode in PERIODICITY_PARAMETER_VALUES["mode"]:
            required.append("occurrence_policy")
    return tuple(dict.fromkeys(required))


def _temporal_resolution_findings(
    node,
    temporal_resolution_by_node: Mapping[str, str] | None,
) -> tuple[SemanticReadinessFinding, ...]:
    operators = allowed_operators_for_node(node)
    if len(operators) <= 1:
        return ()
    resolution = temporal_resolution_by_node or {}
    if node.node_id in resolution:
        try:
            resolved_node_kind(node, resolution)
            return ()
        except ValueError as exc:
            return (
                SemanticReadinessFinding(
                    code="IR_TEMPORAL_OPERATOR_SELECTION_INVALID",
                    path=(
                        f"/faithful_ir/nodes/{node.node_id}/"
                        "allowed_semantic_readings"
                    ),
                    message=str(exc),
                    blocked_capabilities=tuple(
                        sorted(_DIRECT_EXECUTION_CAPABILITIES)
                    ),
                    node_id=node.node_id,
                    node_type=node.node_type,
                ),
            )
    return (
        SemanticReadinessFinding(
            code="IR_TEMPORAL_OPERATOR_UNRESOLVED",
            path=f"/faithful_ir/nodes/{node.node_id}/allowed_semantic_readings",
            message=(
                f"temporal node {node.node_id} has unresolved operator range {operators}"
            ),
            blocked_capabilities=tuple(sorted(_DIRECT_EXECUTION_CAPABILITIES)),
            node_id=node.node_id,
            node_type=node.node_type,
        ),
    )


def _executor_contract_findings(
    node,
    params: dict[str, object],
    temporal_resolution_by_node: Mapping[str, str] | None,
) -> tuple[SemanticReadinessFinding, ...]:
    """Report reviewed values that the direct evaluator cannot execute."""

    try:
        kind = resolved_node_kind(node, temporal_resolution_by_node)
    except ValueError:
        return ()  # The unresolved range has a dedicated finding above.

    messages: list[tuple[str, str]] = []
    if kind in {"precedence", "before"}:
        if params.get("reuse_policy") not in {None, "", "reusable"}:
            messages.append(
                ("reuse_policy", "direct semantics requires reuse_policy=reusable")
            )
        if params.get("scope_origin") not in {None, "", "trace_start"}:
            messages.append(
                ("scope_origin", "direct semantics requires scope_origin=trace_start")
            )
    elif kind == "fixed_duration" and all(
        params.get(name) not in {None, ""}
        for name in ("short_trace_policy", "terminal_obligation_policy")
    ):
        contract = (
            str(params["short_trace_policy"]),
            str(params["terminal_obligation_policy"]),
        )
        if contract not in {
            ("strong_next", "require_full_duration"),
            ("available_prefix", "truncate_at_end"),
        }:
            messages.append(
                ("short_trace_policy", f"unsupported fixed-duration contract {contract}")
            )
    elif kind == "bounded_window" and all(
        params.get(name) not in {None, ""}
        for name in ("short_trace_policy", "terminal_obligation_policy")
    ):
        contract = (
            str(params["short_trace_policy"]),
            str(params["terminal_obligation_policy"]),
        )
        if contract not in {
            ("strong_upper_bound", "require_witness"),
            ("available_prefix", "require_witness"),
            ("available_prefix", "vacuous_if_window_unavailable"),
        }:
            messages.append(
                ("short_trace_policy", f"unsupported bounded-window contract {contract}")
            )
    elif kind == "recurrence":
        expected = {
            "origin": "current_position",
            "gap_policy": "unbounded_eventual",
            "short_trace_policy": "finite_suffix",
            "terminal_obligation_policy": "final_state_witness",
        }
        for name, value in expected.items():
            if params.get(name) not in {None, "", value}:
                messages.append(
                    (name, f"direct recurrence semantics requires {name}={value}")
                )

    return tuple(
        SemanticReadinessFinding(
            code="IR_EXECUTOR_CONTRACT_UNSUPPORTED",
            path=f"/faithful_ir/nodes/{node.node_id}/parameters/{name}",
            message=f"{node.node_type} node {node.node_id}: {message}",
            blocked_capabilities=tuple(sorted(_DIRECT_EXECUTION_CAPABILITIES)),
            node_id=node.node_id,
            node_type=node.node_type,
            parameter_name=name,
        )
        for name, message in messages
    )


__all__ = [
    "CAN_BIND_RESOLUTION",
    "CAN_BUILD_DISCRIMINATION_PLAN",
    "CAN_EXECUTE_IR",
    "CAN_GENERATE_AUTHORITATIVE_TRACES",
    "CAN_PERSIST_IR",
    "CAN_SYNTHESIZE_WITNESS",
    "SEMANTIC_READINESS_CONTRACT_VERSION",
    "SemanticReadinessError",
    "SemanticReadinessFinding",
    "SemanticReadinessReport",
    "audit_semantic_readiness",
    "require_semantic_capability",
]
