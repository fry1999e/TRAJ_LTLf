"""Independent, formula-free execution semantics for FaithfulSemanticIR.

The evaluator deliberately does not call ``compile_ir_to_formula``.  It is used
to check that reviewed trace labels, IR parameters, and the completion compiler
agree instead of allowing the compiler to serve as its own oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
from typing import Iterable, Mapping

from data_examples import Trace
from faithful_ir import validate_faithful_ir
from ltlf_checker import satisfies
from ltlf_parser import Formula
from schemas import FaithfulSemanticIR, SemanticNode
from semantic_readiness import CAN_EXECUTE_IR, audit_semantic_readiness
from temporal_resolution import resolved_node_kind


class IRTraceSemanticsError(ValueError):
    """Raised when an IR node is not executable without guessing semantics."""


@dataclass(frozen=True)
class IRFormulaEquivalenceReport:
    equivalent: bool
    checked_trace_count: int
    mismatch_trace: tuple[tuple[str, ...], ...] = ()
    ir_value: bool | None = None
    formula_value: bool | None = None
    reason: str = ""


@dataclass(frozen=True)
class IRTraceExplanation:
    result: bool
    affected_node_ids: tuple[str, ...]
    source_claim_ids: tuple[str, ...]
    obligations: tuple[tuple[str, str, tuple[int, int]], ...]


def evaluate_ir_trace(
    ir: FaithfulSemanticIR,
    trace: Trace,
    *,
    node_id: str | None = None,
    position: int = 0,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
) -> bool:
    """Evaluate one IR node directly on a non-empty finite trace."""

    # Preserve the domain validation exception type for explicit migration and
    # schema callers; readiness only classifies valid-but-not-executable IR.
    validate_faithful_ir(ir)
    readiness = audit_semantic_readiness(
        ir,
        temporal_resolution_by_node=temporal_resolution_by_node,
    )
    if not readiness.has_capability(CAN_EXECUTE_IR):
        blockers = readiness.blockers_for(CAN_EXECUTE_IR)
        raise IRTraceSemanticsError(
            "; ".join(item.message for item in blockers)
            or "FaithfulSemanticIR is not directly executable"
        )
    if not trace:
        raise IRTraceSemanticsError("LTLf traces must contain at least one state")
    node_map = ir.node_map
    target = node_id or ir.root_node_id
    if target not in node_map:
        raise IRTraceSemanticsError(f"unknown IR node: {target}")

    def ev(current_id: str, at: int) -> bool:
        node = node_map[current_id]
        children = node.children
        try:
            kind = resolved_node_kind(node, temporal_resolution_by_node)
        except ValueError as exc:
            raise IRTraceSemanticsError(str(exc)) from exc
        params = dict(node.parameters)

        if at < 0 or at >= len(trace):
            return False
        if kind == "ap_condition":
            return str(_required_parameter(node, "ap_name")) in trace[at]
        if kind == "boolean_constant":
            return bool(_required_parameter(node, "value"))
        if kind == "negation":
            return not ev(children[0], at)
        if kind == "conjunction":
            return all(ev(child, at) for child in children)
        if kind == "disjunction":
            return any(ev(child, at) for child in children)
        if kind == "implication":
            return (not ev(children[0], at)) or ev(children[1], at)
        if kind == "equivalence":
            return ev(children[0], at) == ev(children[1], at)
        if kind == "identity":
            return ev(children[0], at)
        if kind in {"global_constraint", "persist_to_end"}:
            return all(ev(children[0], index) for index in range(at, len(trace)))
        if kind == "eventually":
            return any(ev(children[0], index) for index in range(at, len(trace)))
        if kind == "next_step":
            steps = int(_required_parameter(node, "steps")) if node.node_type == "next_step" else 1
            return at + steps < len(trace) and ev(children[0], at + steps)
        if kind == "eventually_strict":
            return any(ev(children[0], index) for index in range(at + 1, len(trace)))
        if kind == "until":
            return _eval_until(node, children, at, len(trace), ev)
        if kind == "release":
            # phi R psi == !(!phi U !psi), evaluated directly over the suffix.
            return all(
                ev(children[1], target)
                or any(ev(children[0], witness) for witness in range(at, target + 1))
                for target in range(at, len(trace))
            )
        if kind in {"precedence", "before"}:
            return _eval_precedence(node, children, at, len(trace), ev)
        if kind == "after":
            return any(
                ev(children[0], first)
                and any(ev(children[1], later) for later in range(first + 1, len(trace)))
                for first in range(at, len(trace))
            )
        if kind == "fixed_duration":
            duration = int(_required_parameter(node, "duration"))
            short_policy = str(_required_parameter(node, "short_trace_policy"))
            terminal = str(
                _required_parameter(node, "terminal_obligation_policy")
            )
            if (short_policy, terminal) == (
                "strong_next",
                "require_full_duration",
            ):
                return at + duration <= len(trace) and all(
                    ev(children[0], index) for index in range(at, at + duration)
                )
            if (short_policy, terminal) == (
                "available_prefix",
                "truncate_at_end",
            ):
                return all(
                    ev(children[0], index)
                    for index in range(at, min(at + duration, len(trace)))
                )
            raise IRTraceSemanticsError(
                "unsupported fixed_duration short/terminal policy"
            )
        if kind == "exact_run_length":
            offset = int(_required_parameter(node, "start_offset"))
            duration = int(_required_parameter(node, "duration"))
            policies = (
                str(_required_parameter(node, "origin")),
                str(_required_parameter(node, "start_position_policy")),
                str(_required_parameter(node, "terminal_obligation_policy")),
                str(_required_parameter(node, "post_window_policy")),
            )
            if policies != (
                "current_position",
                "required",
                "require_full_duration",
                "false_if_position_exists",
            ):
                raise IRTraceSemanticsError(
                    "unsupported exact_run_length policy"
                )
            start = at + offset
            end = start + duration
            if start < at or end > len(trace):
                return False
            if not all(ev(children[0], index) for index in range(start, end)):
                return False
            return end >= len(trace) or not ev(children[0], end)
        if kind == "bounded_window":
            lower = int(_required_parameter(node, "lower_bound"))
            upper = int(_required_parameter(node, "upper_bound"))
            short_policy = str(_required_parameter(node, "short_trace_policy"))
            terminal = str(
                _required_parameter(node, "terminal_obligation_policy")
            )
            witness = any(
                ev(children[0], index)
                for index in range(at + lower, min(at + upper + 1, len(trace)))
            )
            if short_policy == "strong_upper_bound":
                if terminal != "require_witness":
                    raise IRTraceSemanticsError(
                        "strong_upper_bound requires a witness obligation"
                    )
                return at + upper < len(trace) and witness
            if short_policy == "available_prefix":
                if terminal == "require_witness":
                    return witness
                if terminal == "vacuous_if_window_unavailable":
                    return at + lower >= len(trace) or witness
            raise IRTraceSemanticsError(
                "unsupported bounded_window short/terminal policy"
            )
        if kind == "ordered_sequence":
            cursor = at
            for child in children:
                match = next(
                    (index for index in range(cursor, len(trace)) if ev(child, index)),
                    None,
                )
                if match is None:
                    return False
                cursor = match + 1
            return True
        if kind == "cardinality":
            count = max(0, int(_required_parameter(node, "count")))
            actual = sum(ev(children[0], index) for index in range(at, len(trace)))
            mode = str(_required_parameter(node, "mode"))
            if mode == "at_most":
                return actual <= count
            if mode == "exactly":
                return actual == count
            if mode == "at_least":
                return actual >= count
            raise IRTraceSemanticsError(f"unsupported cardinality mode: {mode}")
        if kind == "recurrence":
            recurrence_policy = {
                "origin": "current_position",
                "gap_policy": "unbounded_eventual",
                "short_trace_policy": "finite_suffix",
                "terminal_obligation_policy": "final_state_witness",
            }
            for name, expected in recurrence_policy.items():
                if str(_required_parameter(node, name)) != expected:
                    raise IRTraceSemanticsError(
                        f"unsupported recurrence policy for {name}"
                    )
            return all(
                any(ev(children[0], later) for later in range(index, len(trace)))
                for index in range(at, len(trace))
            )
        if kind == "periodicity":
            return _eval_periodicity(node, children[0], at, len(trace), ev)
        if kind == "start_boundary":
            return at == 0 and ev(children[0], 0)
        if kind == "end_boundary":
            return any(
                all(ev(children[0], later) for later in range(index, len(trace)))
                for index in range(at, len(trace))
            )
        if kind == "short_trace_boundary":
            maximum = max(1, int(_required_parameter(node, "maximum_length")))
            return len(trace) - at <= maximum
        raise IRTraceSemanticsError(f"IR node type {kind} has no direct trace semantics")

    return ev(target, position)


def explain_ir_trace(
    ir: FaithfulSemanticIR,
    trace: Trace,
    *,
    node_id: str | None = None,
    position: int = 0,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
) -> IRTraceExplanation:
    """Return source-bound executable edit points that can affect a verdict.

    This is deliberately generated by the local executor, not accepted from a
    model report.  The shallowest ordered/parameterized nodes form a bounded
    explanation frontier; descendants are included only when an ancestor has
    no typed semantic edit contract.
    """

    result = evaluate_ir_trace(
        ir,
        trace,
        node_id=node_id,
        position=position,
        temporal_resolution_by_node=temporal_resolution_by_node,
    )
    from semantic_node_contracts import NODE_CONTRACTS

    ordered_binary = {"implication", "until", "release", "before", "precedence", "after"}
    excluded_parameter_nodes = {"ap_condition", "boolean_constant"}
    frontier = []
    seen = set()

    def visit(current_id):
        if current_id in seen:
            return
        seen.add(current_id)
        node = ir.node_map[current_id]
        contract = NODE_CONTRACTS.get(node.node_type)
        editable = bool(
            contract is not None
            and (
                node.node_type in ordered_binary
                or (contract.parameters and node.node_type not in excluded_parameter_nodes)
            )
        )
        if editable:
            frontier.append(node)
            return
        for child in node.children:
            visit(child)

    visit(node_id or ir.root_node_id)
    claims = tuple(
        "SCLAIM-" + hashlib.sha256(
            f"{ir.source_hash}:{node.node_id}:{node.source_span[0]}:{node.source_span[1]}".encode()
        ).hexdigest()[:20]
        for node in frontier
    )
    obligations = tuple(
        (node.node_id, node.node_type, tuple(node.source_span))
        for node in frontier
    )
    return IRTraceExplanation(
        result,
        tuple(node.node_id for node in frontier),
        claims,
        obligations,
    )


def compare_ir_and_formula(
    ir: FaithfulSemanticIR,
    formula: Formula,
    *,
    max_length: int = 4,
    max_traces: int = 4096,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
) -> IRFormulaEquivalenceReport:
    """Boundedly compare the independent IR evaluator with a compiled formula."""

    aps = tuple(ir.ap_names)
    if len(aps) <= 4:
        states = [
            set(ap for ap, enabled in zip(aps, bits) if enabled)
            for bits in itertools.product((False, True), repeat=len(aps))
        ]
    else:
        # The lowering certificate proves compositional coverage.  For wider
        # AP alphabets, supplement it with a deterministic interaction basis
        # instead of treating exhaustive enumeration limits as a mismatch.
        all_aps = set(aps)
        states = [set(), all_aps]
        states.extend({ap} for ap in aps)
        states.extend(all_aps - {ap} for ap in aps)
        states.extend(
            {left, right}
            for index, left in enumerate(aps)
            for right in aps[index + 1 :]
        )
    checked = 0
    for length in range(1, max(1, max_length) + 1):
        for raw_trace in itertools.product(states, repeat=length):
            trace = [set(step) for step in raw_trace]
            ir_value = evaluate_ir_trace(
                ir,
                trace,
                temporal_resolution_by_node=temporal_resolution_by_node,
            )
            formula_value = satisfies(trace, formula)
            checked += 1
            if ir_value != formula_value:
                return IRFormulaEquivalenceReport(
                    False,
                    checked,
                    tuple(tuple(sorted(step)) for step in trace),
                    ir_value,
                    formula_value,
                    "compiled formula disagrees with independent IR semantics",
                )
            if checked >= max_traces:
                return IRFormulaEquivalenceReport(True, checked)
    return IRFormulaEquivalenceReport(True, checked)


def _required_parameter(node: SemanticNode, name: str) -> object:
    params = dict(node.parameters)
    if name not in params or params[name] in {"", None}:
        raise IRTraceSemanticsError(
            f"{node.node_type} node {node.node_id} requires explicit parameter {name}"
        )
    return params[name]


def _eval_until(node, children, at, length, ev) -> bool:
    params = dict(node.parameters)
    witness_policy = str(_required_parameter(node, "witness_policy"))
    timing = str(_required_parameter(node, "witness_timing"))
    left_at_witness = str(_required_parameter(node, "left_at_witness"))
    first = at + 1 if timing == "strict_future" else at
    for target in range(first, length):
        if not ev(children[1], target):
            continue
        before_end = target + 1 if left_at_witness == "required" else target
        if all(ev(children[0], index) for index in range(at, before_end)):
            return True
    return witness_policy == "optional_if_missing" and all(
        ev(children[0], index) for index in range(at, length)
    )


def _eval_precedence(node, children, at, length, ev) -> bool:
    ordering = str(_required_parameter(node, "ordering"))
    witness_policy = str(_required_parameter(node, "witness_policy"))
    reuse_policy = str(_required_parameter(node, "reuse_policy"))
    scope_origin = str(_required_parameter(node, "scope_origin"))
    if reuse_policy != "reusable" or scope_origin != "trace_start":
        raise IRTraceSemanticsError(
            "direct precedence semantics currently requires reusable/trace_start"
        )
    prerequisite_seen = False
    prerequisite_count = 0
    constrained_seen = False
    for index in range(0 if scope_origin == "trace_start" else at, length):
        prerequisite_now = ev(children[0], index)
        constrained_now = ev(children[1], index)
        if constrained_now:
            constrained_seen = True
            allowed = prerequisite_seen or (ordering == "non_strict" and prerequisite_now)
            if not allowed:
                return False
        if prerequisite_now:
            prerequisite_seen = True
            prerequisite_count += 1
    if witness_policy == "required":
        return prerequisite_seen
    if witness_policy == "optional_if_no_constrained":
        return True
    raise IRTraceSemanticsError(f"unsupported precedence witness policy: {witness_policy}")


def _eval_periodicity(node, child, at, length, ev) -> bool:
    period = int(_required_parameter(node, "period"))
    mode = str(_required_parameter(node, "mode"))
    terminal = str(_required_parameter(node, "terminal_policy"))
    if period < 1:
        raise IRTraceSemanticsError("period must be >= 1")
    positions = [index for index in range(at, length) if ev(child, index)]
    if mode == "fixed_phase":
        phase = int(_required_parameter(node, "phase"))
        off_schedule = str(_required_parameter(node, "off_schedule_policy"))
        if phase < 0 or phase >= period or off_schedule != "forbidden":
            raise IRTraceSemanticsError("invalid fixed_phase contract")
        required = list(range(phase, length, period))
        if not required:
            return terminal == "finite_prefix" and not positions
        return positions == required

    occurrence = str(_required_parameter(node, "occurrence_policy"))
    if terminal != "finite_prefix" or occurrence not in {"required", "optional"}:
        raise IRTraceSemanticsError("invalid event-relative periodicity contract")
    if not positions:
        return occurrence == "optional"
    gaps = [right - left for left, right in zip(positions, positions[1:])]
    if mode == "minimum_spacing":
        return all(gap >= period for gap in gaps)
    if mode == "exact_inter_occurrence_spacing":
        return all(gap == period for gap in gaps)
    if mode == "event_relative_exact_recurrence":
        expected = list(range(positions[0], length, period))
        return positions == expected
    raise IRTraceSemanticsError(f"unsupported periodicity mode: {mode}")
