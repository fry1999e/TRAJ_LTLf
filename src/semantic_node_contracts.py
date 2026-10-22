"""Closed executable contracts for FaithfulSemanticIR v6 nodes.

The registry in this module is deliberately denotation-free.  The compiler and
the independent trace evaluator remain separate implementations; they only
share structural facts that must not drift (arity, parameter vocabulary and
compositional trace horizons).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from contract_versions import FAITHFUL_IR_SCHEMA_VERSION
from relation_parameter_contracts import PERIODICITY_PARAMETER_VALUES


@dataclass(frozen=True)
class ParameterContract:
    description: str
    required: bool = True
    json_type: str = "string"
    minimum: int | None = None
    enum: tuple[object, ...] = ()

    def accepts(self, value: Any) -> bool:
        if self.json_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                return False
            if self.minimum is not None and value < self.minimum:
                return False
        elif self.json_type == "boolean":
            if not isinstance(value, bool):
                return False
        elif self.json_type == "string":
            if not isinstance(value, str) or not value.strip():
                return False
        if self.enum and value not in self.enum:
            return False
        return True

    def json_schema(self) -> dict[str, object]:
        schema: dict[str, object] = {
            "type": self.json_type,
            "description": self.description,
        }
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.enum:
            schema["enum"] = list(self.enum)
        return schema


@dataclass(frozen=True)
class SemanticNodeContract:
    min_children: int
    max_children: int | None
    parameters: Mapping[str, ParameterContract]
    semantic_family: str = "state"


def _integer_parameter(minimum: int, *, required: bool = True) -> ParameterContract:
    return ParameterContract(
        f"integer >= {minimum}",
        required=required,
        json_type="integer",
        minimum=minimum,
    )


def _enum_parameter(*values: str, required: bool = True) -> ParameterContract:
    return ParameterContract(
        " or ".join(values),
        required=required,
        json_type="string",
        enum=tuple(values),
    )


def _string_parameter(description: str, *, required: bool = True) -> ParameterContract:
    return ParameterContract(description, required=required, json_type="string")


def _boolean_parameter(*, required: bool = True) -> ParameterContract:
    return ParameterContract("boolean", required=required, json_type="boolean")


def _contract(
    min_children: int,
    max_children: int | None,
    *,
    family: str = "state",
    parameters: Mapping[str, ParameterContract] | None = None,
) -> SemanticNodeContract:
    return SemanticNodeContract(
        min_children=min_children,
        max_children=max_children,
        parameters=dict(parameters or {}),
        semantic_family=family,
    )


NODE_CONTRACTS: Mapping[str, SemanticNodeContract] = {
    "ap_condition": _contract(
        0,
        0,
        parameters={
            "ap_name": _string_parameter("non-empty AP name"),
            # Positive polarity is accepted only as explicit provenance.  A
            # negative proposition must be represented by a negation node.
            "polarity": _enum_parameter(
                "positive",
                required=False,
            ),
        },
    ),
    "boolean_constant": _contract(
        0,
        0,
        parameters={"value": _boolean_parameter()},
    ),
    "negation": _contract(1, 1),
    "conjunction": _contract(2, None),
    "disjunction": _contract(2, None),
    "implication": _contract(2, 2),
    "equivalence": _contract(2, 2),
    "global_constraint": _contract(1, 1, family="scope"),
    "eventually": _contract(1, 1, family="unbounded_window"),
    "eventually_strict": _contract(1, 1, family="unbounded_window"),
    "next_step": _contract(
        1,
        1,
        family="exact_offset",
        parameters={
            "steps": _integer_parameter(1),
            "anchor_node_id": _string_parameter(
                "non-empty provenance node id", required=False
            ),
            "relative_offset": _integer_parameter(0, required=False),
        },
    ),
    "until": _contract(
        2,
        2,
        family="ordered_relation",
        parameters={
            "witness_policy": _enum_parameter("required", "optional_if_missing"),
            "witness_timing": _enum_parameter("current_or_future", "strict_future"),
            "left_at_witness": _enum_parameter("required", "not_required"),
            "scope_origin": _enum_parameter("current_position", "inherited"),
        },
    ),
    "release": _contract(2, 2, family="ordered_relation"),
    "before": _contract(2, 2, family="precedence"),
    "precedence": _contract(2, 2, family="precedence"),
    "after": _contract(2, 2, family="ordered_relation"),
    "ordered_sequence": _contract(1, None, family="ordered_relation"),
    "persist_to_end": _contract(1, 1, family="scope"),
    "fixed_duration": _contract(
        1,
        1,
        family="universal_window",
        parameters={
            "duration": _integer_parameter(1),
            "origin": _enum_parameter("current_position"),
            "short_trace_policy": _enum_parameter("strong_next", "available_prefix"),
            "terminal_obligation_policy": _enum_parameter(
                "require_full_duration", "truncate_at_end"
            ),
            "minimum_horizon": _integer_parameter(1),
        },
    ),
    "exact_run_length": _contract(
        1,
        1,
        family="exact_duration",
        parameters={
            "start_offset": _integer_parameter(0),
            "duration": _integer_parameter(1),
            "origin": _enum_parameter("current_position"),
            "start_position_policy": _enum_parameter("required"),
            "terminal_obligation_policy": _enum_parameter("require_full_duration"),
            "post_window_policy": _enum_parameter("false_if_position_exists"),
            "minimum_horizon": _integer_parameter(1),
        },
    ),
    "bounded_window": _contract(
        1,
        1,
        family="existential_window",
        parameters={
            "lower_bound": _integer_parameter(0),
            "upper_bound": _integer_parameter(0),
            "origin": _enum_parameter("current_position"),
            "short_trace_policy": _enum_parameter(
                "strong_upper_bound", "available_prefix"
            ),
            "terminal_obligation_policy": _enum_parameter(
                "require_witness", "vacuous_if_window_unavailable"
            ),
            "minimum_horizon": _integer_parameter(1),
        },
    ),
    "cardinality": _contract(
        1,
        1,
        family="cardinality",
        parameters={
            "mode": _enum_parameter("at_least", "at_most", "exactly"),
            "count": _integer_parameter(0),
        },
    ),
    "recurrence": _contract(
        1,
        1,
        family="recurrence",
        parameters={
            "origin": _enum_parameter("current_position"),
            "gap_policy": _enum_parameter("unbounded_eventual"),
            "short_trace_policy": _enum_parameter("finite_suffix"),
            "terminal_obligation_policy": _enum_parameter("final_state_witness"),
            "minimum_horizon": _integer_parameter(1),
        },
    ),
    "periodicity": _contract(
        1,
        1,
        family="periodicity",
        parameters={
            "period": _integer_parameter(1),
            "phase": _integer_parameter(0, required=False),
            "mode": _enum_parameter(*sorted(PERIODICITY_PARAMETER_VALUES["mode"])),
            "origin": _enum_parameter(*sorted(PERIODICITY_PARAMETER_VALUES["origin"])),
            "off_schedule_policy": _enum_parameter(
                *sorted(PERIODICITY_PARAMETER_VALUES["off_schedule_policy"]),
                required=False,
            ),
            "occurrence_policy": _enum_parameter(
                *sorted(PERIODICITY_PARAMETER_VALUES["occurrence_policy"]),
                required=False,
            ),
            "terminal_policy": _enum_parameter(
                *sorted(PERIODICITY_PARAMETER_VALUES["terminal_policy"])
            ),
        },
    ),
    "start_boundary": _contract(1, 1, family="boundary"),
    "end_boundary": _contract(1, 1, family="boundary"),
    "short_trace_boundary": _contract(
        0,
        0,
        family="boundary",
        parameters={
            "maximum_length": _integer_parameter(1)
        },
    ),
}


_PRECEDENCE_PARAMETERS: Mapping[str, ParameterContract] = {
    "ordering": _enum_parameter("strict", "non_strict"),
    "witness_policy": _enum_parameter("required", "optional_if_no_constrained"),
    "same_step_policy": _enum_parameter("forbidden", "allowed"),
    "reuse_policy": _enum_parameter("reusable", "one_to_one"),
    "scope_origin": _enum_parameter("trace_start", "local_activation"),
}

NODE_CONTRACTS = dict(NODE_CONTRACTS)
NODE_CONTRACTS["before"] = _contract(
    2, 2, family="precedence", parameters=_PRECEDENCE_PARAMETERS
)
NODE_CONTRACTS["precedence"] = _contract(
    2, 2, family="precedence", parameters=_PRECEDENCE_PARAMETERS
)


LEGACY_NODE_TYPES = frozenset({"current_or_future", "strict_future"})


@dataclass(frozen=True)
class NodeContractFinding:
    code: str
    path_suffix: str
    message: str


def node_parameter_constraints_payload() -> dict[str, object]:
    """Return the model-facing parameter contract from the runtime registry."""

    return {
        node_type: {
            "type": "object",
            "properties": {
                name: parameter.json_schema()
                for name, parameter in sorted(contract.parameters.items())
            },
            "required": sorted(
                name
                for name, parameter in contract.parameters.items()
                if parameter.required
            ),
            "additionalProperties": False,
        }
        for node_type, contract in sorted(NODE_CONTRACTS.items())
        if contract.parameters
    }


def node_contract_prompt_view() -> dict[str, object]:
    """Return the complete model-facing structural view from one registry."""

    return {
        node_type: {
            "semantic_family": contract.semantic_family,
            "children": {
                "minimum": contract.min_children,
                **(
                    {"maximum": contract.max_children}
                    if contract.max_children is not None
                    else {}
                ),
            },
            "parameters": {
                name: parameter.json_schema()
                for name, parameter in sorted(contract.parameters.items())
            },
            "required_parameters": sorted(
                name
                for name, parameter in contract.parameters.items()
                if parameter.required
            ),
        }
        for node_type, contract in sorted(NODE_CONTRACTS.items())
    }


def node_contract_fingerprint() -> str:
    payload = {
        "faithful_ir_schema_version": FAITHFUL_IR_SCHEMA_VERSION,
        "nodes": node_contract_prompt_view(),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def node_proposal_json_schema() -> dict[str, object]:
    """Derive structural proposal alternatives from ``NODE_CONTRACTS``."""

    alternatives = []
    for node_type, contract in sorted(NODE_CONTRACTS.items()):
        children: dict[str, object] = {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": contract.min_children,
        }
        if contract.max_children is not None:
            children["maxItems"] = contract.max_children
        parameters = {
            "type": "object",
            "properties": {
                name: parameter.json_schema()
                for name, parameter in sorted(contract.parameters.items())
            },
            "required": sorted(
                name
                for name, parameter in contract.parameters.items()
                if parameter.required
            ),
            "additionalProperties": False,
        }
        alternatives.append(
            {
                "type": "object",
                "required": ["node_id", "node_type", "children", "parameters"],
                "properties": {
                    "node_id": {"type": "string", "minLength": 1},
                    "node_type": {"const": node_type},
                    "children": children,
                    "parameters": parameters,
                },
            }
        )
    return {"oneOf": alternatives}


def children_patch_json_schema() -> dict[str, object]:
    """Return stable-ID children edit schemas, never positional JSON paths."""

    alternatives = []
    for node_type, contract in sorted(NODE_CONTRACTS.items()):
        children: dict[str, object] = {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": contract.min_children,
        }
        if contract.max_children is not None:
            children["maxItems"] = contract.max_children
        alternatives.append(
            {
                "type": "object",
                "required": [
                    "operation",
                    "node_id",
                    "expected_node_type",
                    "expected_children",
                    "new_children",
                ],
                "properties": {
                    "operation": {"const": "replace_children"},
                    "node_id": {"type": "string", "minLength": 1},
                    "expected_node_type": {"const": node_type},
                    "expected_children": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "new_children": children,
                },
                "additionalProperties": False,
            }
        )
    return {"oneOf": alternatives}


def node_parameter_json_schema(node_type: str, name: str) -> dict[str, object] | None:
    contract = NODE_CONTRACTS.get(node_type)
    if contract is None or name not in contract.parameters:
        return None
    return contract.parameters[name].json_schema()


def node_parameter_required(node_type: str, name: str) -> bool | None:
    contract = NODE_CONTRACTS.get(node_type)
    if contract is None or name not in contract.parameters:
        return None
    return contract.parameters[name].required


def required_node_parameter_names(node_type: str) -> tuple[str, ...]:
    contract = NODE_CONTRACTS.get(node_type)
    if contract is None:
        return ()
    return tuple(
        name
        for name, parameter in contract.parameters.items()
        if parameter.required
    )


def node_parameter_enum_values(node_type: str) -> dict[str, frozenset[str]]:
    contract = NODE_CONTRACTS.get(node_type)
    if contract is None:
        return {}
    return {
        name: frozenset(str(value) for value in parameter.enum)
        for name, parameter in contract.parameters.items()
        if parameter.enum
    }


def node_contract_findings(
    node: object,
    *,
    schema_version: str,
    allow_missing_required: bool = False,
) -> tuple[NodeContractFinding, ...]:
    """Return leaf-addressed closed-world findings for one executable node."""

    if schema_version != FAITHFUL_IR_SCHEMA_VERSION:
        return (
            NodeContractFinding(
                "IR_SCHEMA_VERSION_UNSUPPORTED",
                "",
                "online node validation requires schema-v"
                f"{FAITHFUL_IR_SCHEMA_VERSION}, got {schema_version!r}",
            ),
        )
    node_type = str(getattr(node, "node_type", ""))
    if node_type in LEGACY_NODE_TYPES:
        replacement = (
            "eventually" if node_type == "current_or_future" else "eventually_strict"
        )
        return (
            NodeContractFinding(
                "IR_LEGACY_NODE_TYPE",
                "/node_type",
                f"legacy node_type {node_type!r} is forbidden in schema-v"
                f"{FAITHFUL_IR_SCHEMA_VERSION}; "
                f"migrate it offline to {replacement!r}",
            ),
        )
    contract = NODE_CONTRACTS.get(node_type)
    if contract is None:
        return (
            NodeContractFinding(
                "IR_NODE_TYPE_UNSUPPORTED",
                "/node_type",
                f"schema-v{FAITHFUL_IR_SCHEMA_VERSION} has no executable contract "
                f"for node_type {node_type!r}",
            ),
        )
    children = tuple(getattr(node, "children", ()))
    findings: list[NodeContractFinding] = []
    if len(children) < contract.min_children:
        findings.append(NodeContractFinding(
            "IR_NODE_ARITY_INVALID",
            "/children",
            f"{node_type} requires at least {contract.min_children} child nodes",
        ))
    if contract.max_children is not None and len(children) > contract.max_children:
        findings.append(NodeContractFinding(
            "IR_NODE_ARITY_INVALID",
            "/children",
            f"{node_type} permits at most {contract.max_children} child nodes",
        ))
    raw_parameters = tuple(getattr(node, "parameters", ()))
    names = [str(name) for name, _value in raw_parameters]
    if len(names) != len(set(names)):
        findings.append(NodeContractFinding(
            "IR_NODE_PARAMETER_DUPLICATE",
            "/parameters",
            f"{node_type} parameters must use unique names",
        ))
    parameters = dict(raw_parameters)
    for name in sorted(set(parameters) - set(contract.parameters)):
        findings.append(NodeContractFinding(
            "IR_NODE_PARAMETER_UNKNOWN",
            f"/parameters/{name}",
            f"{node_type} has unknown parameter {name}",
        ))
    for name, parameter_contract in contract.parameters.items():
        if name not in parameters:
            if parameter_contract.required and not allow_missing_required:
                findings.append(NodeContractFinding(
                    "IR_NODE_PARAMETER_MISSING",
                    f"/parameters/{name}",
                    f"{node_type} is missing required parameter {name}",
                ))
            continue
        if not parameter_contract.accepts(parameters[name]):
            findings.append(NodeContractFinding(
                "IR_NODE_PARAMETER_INVALID",
                f"/parameters/{name}",
                f"{node_type} parameter {name} must be {parameter_contract.description}",
            ))
    if node_type == "bounded_window":
        lower = parameters.get("lower_bound")
        upper = parameters.get("upper_bound")
        if (
            isinstance(lower, int)
            and not isinstance(lower, bool)
            and isinstance(upper, int)
            and not isinstance(upper, bool)
            and upper < lower
        ):
            findings.append(NodeContractFinding(
                "IR_BOUNDED_WINDOW_ORDER_INVALID",
                "/parameters/upper_bound",
                "bounded_window upper_bound must be >= lower_bound",
            ))
    if node_type == "next_step":
        anchor = parameters.get("anchor_node_id")
        relative = parameters.get("relative_offset")
        if (anchor is None) != (relative is None):
            missing = "relative_offset" if relative is None else "anchor_node_id"
            findings.append(NodeContractFinding(
                "IR_NEXT_STEP_PROVENANCE_INCOMPLETE",
                f"/parameters/{missing}",
                "next_step anchor_node_id and relative_offset must be declared together",
            ))
    if node_type in {"precedence", "before"}:
        ordering = parameters.get("ordering")
        same_step = parameters.get("same_step_policy")
        expected_same_step = "forbidden" if ordering == "strict" else "allowed"
        if (
            ordering in {"strict", "non_strict"}
            and same_step in {"forbidden", "allowed"}
            and same_step != expected_same_step
        ):
            findings.append(NodeContractFinding(
                "IR_PRECEDENCE_ORDERING_POLICY_INCONSISTENT",
                "/parameters/same_step_policy",
                "precedence ordering and same_step_policy must describe one decision",
            ))
    return tuple(findings)


def node_contract_errors(
    node: object,
    *,
    schema_version: str,
    allow_missing_required: bool = False,
) -> tuple[str, ...]:
    """Return closed-world structural errors for one current-schema node."""

    return tuple(
        finding.message
        for finding in node_contract_findings(
            node,
            schema_version=schema_version,
            allow_missing_required=allow_missing_required,
        )
    )


def structural_horizon(ir: object, node_id: str | None = None) -> int:
    """Compute the minimum executable horizon of an entire IR subtree."""

    node_map = getattr(ir, "node_map")
    target = node_id or str(getattr(ir, "root_node_id"))
    memo: dict[str, int] = {}

    def visit(current_id: str) -> int:
        if current_id in memo:
            return memo[current_id]
        node = node_map[current_id]
        children = tuple(getattr(node, "children", ()))
        child_horizons = tuple(visit(child) for child in children)
        child_max = max(child_horizons, default=1)
        kind = str(getattr(node, "node_type", ""))
        params = dict(getattr(node, "parameters", ()))
        if kind == "next_step":
            horizon = int(params.get("steps", 1)) + child_max
        elif kind == "bounded_window":
            horizon = int(params.get("upper_bound", 1)) + child_max
        elif kind == "fixed_duration":
            horizon = int(params.get("duration", 1)) - 1 + child_max
        elif kind == "exact_run_length":
            horizon = (
                int(params.get("start_offset", 0))
                + int(params.get("duration", 1))
                + child_max
            )
        elif kind == "ordered_sequence":
            horizon = max(child_max, len(children))
        elif kind == "periodicity":
            horizon = max(child_max, int(params.get("period", 1)) + 1)
        elif kind == "cardinality":
            horizon = max(child_max, int(params.get("count", 0)) + 1)
        else:
            horizon = child_max
        memo[current_id] = max(1, horizon)
        return memo[current_id]

    return visit(target)


def discriminator_horizon(node: object) -> int:
    """Minimum horizon needed to challenge the nearest semantic neighbours."""

    kind = str(getattr(node, "node_type", ""))
    params = dict(getattr(node, "parameters", ()))
    if kind in {"eventually", "eventually_strict", "next_step"}:
        return max(3, int(params.get("steps", 1)) + 2)
    if kind == "bounded_window":
        return int(params["upper_bound"]) + 2
    if kind == "exact_run_length":
        return int(params["start_offset"]) + int(params["duration"]) + 1
    if kind in {"precedence", "before", "until"}:
        return 2
    if kind == "cardinality":
        return int(params["count"]) + 1
    if kind in {"global_constraint", "implication"}:
        return 2
    return 1
