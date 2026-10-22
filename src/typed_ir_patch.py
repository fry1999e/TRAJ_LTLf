"""Closed, hash-bound local choices and atomic semantic patch transactions."""
from dataclasses import dataclass, replace
from faithful_ir import validate_faithful_ir, compute_ir_version
from semantic_node_contracts import node_parameter_enum_values


class TypedPatchError(ValueError):
    code = "PATCH_SCHEMA_INVALID"


@dataclass(frozen=True)
class TypedPatchTransactionResult:
    committed: bool
    ir: object
    before_conflicts: int
    after_conflicts: int
    replayed_conflict_count: int
    replayed_regression_count: int
    reason_code: str = "TYPED_PATCH_COMMITTED"


def patch_schema(ir, authorized_axes):
    choices = []
    for axis in authorized_axes:
        node = ir.node_map[axis["node_id"]]
        operation = axis.get("operation", "choose_policy_branch")
        if operation in {"choose_policy_branch", "replace_parameter"}:
            parameter = axis["parameter"]
            legal = node_parameter_enum_values(node.node_type).get(parameter, frozenset())
            values = tuple(axis["values"])
            if not values or not set(values) <= legal:
                raise TypedPatchError("axis values are not declared by node contract")
            choices.append({"axis_id": axis["axis_id"], "node_id": node.node_id,
                            "operation": operation, "parameter": parameter, "values": list(values)})
        elif operation == "swap_children":
            if len(node.children) != 2 or axis.get("parameter") or axis.get("values"):
                raise TypedPatchError("swap_children requires one binary node authority")
            choices.append({"axis_id": axis["axis_id"], "node_id": node.node_id,
                            "operation": operation, "parameter": "", "values": ["swap"]})
        else:
            raise TypedPatchError("unsupported authorized patch operation")
    return {"base_ir_hash": ir.artifact_hash, "operations": choices}


def apply_typed_patch(ir, patch, authorized_axes):
    schema = patch_schema(ir, authorized_axes)
    if not isinstance(patch, dict) or set(patch) != {"base_ir_hash", "operations"} or patch["base_ir_hash"] != ir.artifact_hash:
        raise TypedPatchError("stale or invalid patch envelope")
    operations = patch["operations"]
    if not isinstance(operations, list) or not operations:
        raise TypedPatchError("patch operations must be a nonempty list")
    axes = {axis["axis_id"]: axis for axis in schema["operations"]}
    if len(axes) != len(schema["operations"]): raise TypedPatchError("duplicate axis")
    nodes, seen = dict(ir.node_map), set()
    for operation in operations:
        if not isinstance(operation, dict) or set(operation) != {"op", "target", "value", "source_spans", "contract_fields"}:
            raise TypedPatchError("unexpected patch fields")
        target = operation["target"]
        if not isinstance(target, str) or target not in axes or target in seen:
            raise TypedPatchError("unauthorized or duplicate operation")
        seen.add(target); axis = axes[target]
        if operation["op"] != axis["operation"]:
            raise TypedPatchError("operation type is not authorized for this axis")
        if operation["value"] not in axis["values"]: raise TypedPatchError("illegal enum")
        authority = next(a for a in authorized_axes if a["axis_id"] == target)
        spans = operation["source_spans"]
        if (not isinstance(spans, list) or not spans or operation["contract_fields"] != []
                or any(not isinstance(s, list) or len(s) != 2 or any(type(i) is not int for i in s)
                       or not 0 <= s[0] < s[1] <= len(ir.source_text) for s in spans)
                or any(s not in authority.get("source_spans", []) for s in spans)):
            raise TypedPatchError("patch lacks exact authorized source evidence")
        node = nodes[axis["node_id"]]
        if axis["operation"] == "swap_children":
            nodes[node.node_id] = replace(node, children=node.children[::-1])
        else:
            params = dict(node.parameters)
            params[axis["parameter"]] = operation["value"]
            nodes[node.node_id] = replace(node, parameters=tuple(sorted(params.items())))
    revised = replace(ir, nodes=tuple(nodes[n.node_id] for n in ir.nodes), ir_version="")
    revised = replace(revised, ir_version=compute_ir_version(revised))
    validate_faithful_ir(revised)
    from execution_observer import emit
    emit(
        "typed_patch_applied",
        requirement_id=ir.requirement_id,
        base_ir_version=ir.ir_version,
        ir_version=revised.ir_version,
        base_ir_artifact_hash=ir.artifact_hash,
        ir_artifact_hash=revised.artifact_hash,
        authorized_axis_ids=sorted(seen),
        entrypoint="apply_typed_patch",
    )
    return revised


def authorized_axes_from_failure_packets(ir, packets):
    """Derive closed typed choices from preflighted node authority."""

    axes = []
    seen = set()
    for packet in packets:
        spans = [list(span) for span in packet.get("source_spans", ())]
        operations = set(packet.get("authorized_operations", ()))
        for node_id in packet.get("authorized_node_ids", ()):
            node = ir.node_map.get(str(node_id))
            if node is None:
                continue
            if "swap_children" in operations and len(node.children) == 2:
                key = (node.node_id, "swap_children", "")
                if key not in seen:
                    seen.add(key)
                    axes.append({
                        "axis_id": f"patch:{node.node_id}:swap_children",
                        "node_id": node.node_id,
                        "operation": "swap_children",
                        "source_spans": spans,
                    })
            if "replace_parameter" in operations:
                enums = node_parameter_enum_values(node.node_type)
                for parameter, values in sorted(enums.items()):
                    key = (node.node_id, "replace_parameter", parameter)
                    if key in seen or len(values) < 2:
                        continue
                    seen.add(key)
                    axes.append({
                        "axis_id": f"patch:{node.node_id}:{parameter}",
                        "node_id": node.node_id,
                        "operation": "replace_parameter",
                        "parameter": parameter,
                        "values": sorted(values),
                        "source_spans": spans,
                    })
    return tuple(axes)


def apply_typed_patch_transaction(
    ir,
    patch,
    authorized_axes,
    *,
    before_vector,
    conflict_cases,
    regression_cases=(),
):
    """Preflight, apply, validate, replay, require progress, then commit."""

    from ir_trace_semantics import evaluate_ir_trace
    from semantic_progress import ConflictVector, ProgressGuard, SemanticProgressStop

    # Local schema construction is the preflight and rejects stale/illegal
    # authority before any mutation is attempted.
    patch_schema(ir, authorized_axes)
    revised = apply_typed_patch(ir, patch, authorized_axes)
    validate_faithful_ir(revised)

    def mismatch(candidate, case):
        trace = case.get("trace")
        expected = case.get("expected")
        if not isinstance(trace, (list, tuple)) or type(expected) is not bool:
            raise TypedPatchError("trace replay case is malformed")
        return evaluate_ir_trace(candidate, [set(step) for step in trace]) != expected

    before_conflicts = sum(mismatch(ir, case) for case in conflict_cases)
    after_conflicts = sum(mismatch(revised, case) for case in conflict_cases)
    if not conflict_cases or after_conflicts >= before_conflicts:
        raise TypedPatchError("typed patch did not strictly reduce the conflict vector")
    for case in regression_cases:
        if mismatch(revised, case):
            raise TypedPatchError("typed patch failed regression trace replay")
    after_vector = ConflictVector(
        verified_source_contradictions=before_vector.verified_source_contradictions,
        unresolved_required_contract_axes=before_vector.unresolved_required_contract_axes,
        trace_label_conflicts=after_conflicts,
        readiness_debt=before_vector.readiness_debt,
        structural_errors=before_vector.structural_errors,
    )
    try:
        ProgressGuard(ir, before_vector).accept(revised, after_vector)
    except SemanticProgressStop as exc:
        raise TypedPatchError(str(exc)) from exc
    from execution_observer import emit
    emit(
        "typed_patch_committed",
        requirement_id=ir.requirement_id,
        base_ir_version=ir.ir_version,
        ir_version=revised.ir_version,
        before_conflicts=before_conflicts,
        after_conflicts=after_conflicts,
        replayed_regression_count=len(tuple(regression_cases)),
        entrypoint="apply_typed_patch_transaction",
    )
    return TypedPatchTransactionResult(
        True,
        revised,
        before_conflicts,
        after_conflicts,
        len(tuple(conflict_cases)),
        len(tuple(regression_cases)),
    )
