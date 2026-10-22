"""Formula-free bounded witness synthesis for executable semantic IR.

The solver encodes the direct finite-trace semantics of ``FaithfulSemanticIR``
into a small propositional constraint system.  It never compiles an LTLf
formula and never supplies a truth label to the natural-language adjudicator.
Every returned witness is checked again by :func:`evaluate_ir_trace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping

from ir_trace_semantics import IRTraceSemanticsError, evaluate_ir_trace
from schemas import FaithfulSemanticIR
from semantic_readiness import CAN_EXECUTE_IR, audit_semantic_readiness
from temporal_resolution import resolved_node_kind


WITNESS_FOUND = "witness_found"
UNSAT_IN_DECLARED_DOMAIN = "unsat_in_declared_domain"
UNKNOWN_RESOURCE_LIMIT = "unknown_resource_limit"
UNSUPPORTED_SEMANTICS = "unsupported_semantics"
INVALID_OBLIGATION = "invalid_obligation"
NO_ADDITIONAL_DISTINCT_WITNESS = "no_additional_distinct_witness"


TraceKey = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class HorizonPolicy:
    """The explicit finite domain in which a witness may be synthesized."""

    minimum_length: int = 1
    maximum_length: int = 1
    exact_length: int | None = None

    def __post_init__(self) -> None:
        if self.minimum_length < 1 or self.maximum_length < 1:
            raise ValueError("IR witness horizons must be positive")
        if self.maximum_length < self.minimum_length:
            raise ValueError("IR witness maximum horizon is below its minimum")
        if self.exact_length is not None:
            if self.exact_length < 1:
                raise ValueError("IR witness exact horizon must be positive")
            if not self.minimum_length <= self.exact_length <= self.maximum_length:
                raise ValueError("IR witness exact horizon lies outside its domain")

    @property
    def lengths(self) -> tuple[int, ...]:
        if self.exact_length is not None:
            return (self.exact_length,)
        return tuple(range(self.minimum_length, self.maximum_length + 1))


@dataclass(frozen=True)
class WitnessRequest:
    ir: FaithfulSemanticIR
    node_id: str
    desired_value: bool
    horizon_policy: HorizonPolicy
    blocked_traces: tuple[TraceKey, ...] = ()
    purpose: str = "primary_witness"
    branch_variant: int = 0
    max_decisions: int = 250_000


@dataclass(frozen=True)
class WitnessResult:
    status: str
    node_id: str
    desired_value: bool
    witness: tuple[frozenset[str], ...] | None = None
    attempted_lengths: tuple[int, ...] = ()
    proof_scope: str = ""
    stop_reason: str = ""
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def trace(self) -> list[set[str]] | None:
        if self.witness is None:
            return None
        return [set(step) for step in self.witness]


class IRWitnessSynthesisError(RuntimeError):
    """A typed witness failure that batch routing can handle without guessing."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        result: WitnessResult,
        obligation_id: str = "",
        suggested_rerun_stage: str = "trace",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.result = result
        self.suggested_rerun_stage = suggested_rerun_stage
        self.diagnostics = {
            "reason_code": code,
            "obligation_id": obligation_id,
            "node_id": result.node_id,
            "desired_value": result.desired_value,
            "solver_status": result.status,
            "attempted_lengths": list(result.attempted_lengths),
            "proof_scope": result.proof_scope,
            "stop_reason": result.stop_reason,
            "suggested_rerun_stage": suggested_rerun_stage,
            **dict(result.diagnostics),
        }


def trace_key(trace: Iterable[Iterable[str]]) -> TraceKey:
    return tuple(tuple(sorted(step)) for step in trace)


def synthesize_ir_witness(request: WitnessRequest) -> WitnessResult:
    """Find or refute one bounded witness using only direct IR semantics."""

    ir = request.ir
    readiness = audit_semantic_readiness(ir)
    if not readiness.structurally_valid:
        blocker = readiness.findings[0] if readiness.findings else None
        return WitnessResult(
            INVALID_OBLIGATION,
            request.node_id,
            request.desired_value,
            attempted_lengths=(),
            stop_reason="IR validation failed before witness synthesis",
            diagnostics={
                "error_type": "FaithfulIRValidationError",
                "error": blocker.message if blocker is not None else "invalid IR",
                "readiness_fingerprint": readiness.fingerprint,
                "findings": [
                    {
                        "code": item.code,
                        "path": item.path,
                        "message": item.message,
                    }
                    for item in readiness.findings
                ],
            },
        )
    if not readiness.has_capability(CAN_EXECUTE_IR):
        blockers = readiness.blockers_for(CAN_EXECUTE_IR)
        return WitnessResult(
            UNSUPPORTED_SEMANTICS,
            request.node_id,
            request.desired_value,
            attempted_lengths=(),
            stop_reason="IR is not directly executable under reviewed semantics",
            diagnostics={
                "error_type": "SemanticReadinessError",
                "error": "; ".join(item.message for item in blockers),
                "readiness_fingerprint": readiness.fingerprint,
                "findings": [
                    {
                        "code": item.code,
                        "path": item.path,
                        "message": item.message,
                        "node_id": item.node_id,
                        "parameter_name": item.parameter_name,
                    }
                    for item in blockers
                ],
            },
        )
    if request.node_id not in ir.node_map:
        return WitnessResult(
            INVALID_OBLIGATION,
            request.node_id,
            request.desired_value,
            stop_reason="witness request names an unknown IR node",
        )
    if request.max_decisions < 1:
        return WitnessResult(
            INVALID_OBLIGATION,
            request.node_id,
            request.desired_value,
            stop_reason="max_decisions must be positive",
        )

    attempted: list[int] = []
    unsat_lengths: list[int] = []
    total_decisions = 0
    total_propagations = 0
    blocked_at_lengths = 0
    for length in request.horizon_policy.lengths:
        attempted.append(length)
        try:
            encoding = _IRCNFEncoding(ir, length)
            target = encoding.node_literal(request.node_id, 0)
            encoding.add_clause((target if request.desired_value else -target,))
            applicable_blocks = tuple(
                blocked for blocked in request.blocked_traces if len(blocked) == length
            )
            blocked_at_lengths += len(applicable_blocks)
            for blocked in applicable_blocks:
                encoding.block_trace(blocked)
        except (IRTraceSemanticsError, ValueError, KeyError) as exc:
            return WitnessResult(
                UNSUPPORTED_SEMANTICS,
                request.node_id,
                request.desired_value,
                attempted_lengths=tuple(attempted),
                proof_scope=f"length={length}",
                stop_reason="IR node could not be encoded without guessing semantics",
                diagnostics={"error_type": type(exc).__name__, "error": str(exc)},
            )

        solver = _DPLLSolver(
            encoding.clauses,
            encoding.variable_count,
            max_decisions=max(1, request.max_decisions - total_decisions),
            prefer_true=bool((request.branch_variant + length) % 2),
        )
        model = solver.solve()
        total_decisions += solver.decisions
        total_propagations += solver.propagations
        if solver.resource_limited:
            return WitnessResult(
                UNKNOWN_RESOURCE_LIMIT,
                request.node_id,
                request.desired_value,
                attempted_lengths=tuple(attempted),
                proof_scope=f"lengths={attempted[0]}..{attempted[-1]}",
                stop_reason="bounded SAT decision budget exhausted",
                diagnostics={
                    "decisions": total_decisions,
                    "propagations": total_propagations,
                    "variable_count": encoding.variable_count,
                    "clause_count": len(encoding.clauses),
                    "max_decisions": request.max_decisions,
                },
            )
        if model is None:
            unsat_lengths.append(length)
            continue

        trace = encoding.trace_from_model(model)
        try:
            checked = evaluate_ir_trace(ir, trace, node_id=request.node_id)
        except IRTraceSemanticsError as exc:
            return WitnessResult(
                UNSUPPORTED_SEMANTICS,
                request.node_id,
                request.desired_value,
                attempted_lengths=tuple(attempted),
                proof_scope=f"length={length}",
                stop_reason="independent IR evaluator rejected synthesized witness",
                diagnostics={"error_type": type(exc).__name__, "error": str(exc)},
            )
        if checked != request.desired_value:
            return WitnessResult(
                INVALID_OBLIGATION,
                request.node_id,
                request.desired_value,
                attempted_lengths=tuple(attempted),
                proof_scope=f"length={length}",
                stop_reason="symbolic encoding disagrees with independent IR evaluator",
                diagnostics={
                    "reason_code": "IR_WITNESS_VALIDATION_MISMATCH",
                    "encoded_value": request.desired_value,
                    "evaluated_value": checked,
                    "trace": [sorted(step) for step in trace],
                },
            )
        return WitnessResult(
            WITNESS_FOUND,
            request.node_id,
            request.desired_value,
            witness=tuple(frozenset(step) for step in trace),
            attempted_lengths=tuple(attempted),
            proof_scope=f"length={length}",
            stop_reason="verified witness found",
            diagnostics={
                "decisions": total_decisions,
                "propagations": total_propagations,
                "variable_count": encoding.variable_count,
                "clause_count": len(encoding.clauses),
                "blocked_trace_count": len(applicable_blocks),
            },
        )

    status = (
        NO_ADDITIONAL_DISTINCT_WITNESS
        if request.blocked_traces and blocked_at_lengths
        else UNSAT_IN_DECLARED_DOMAIN
    )
    return WitnessResult(
        status,
        request.node_id,
        request.desired_value,
        attempted_lengths=tuple(attempted),
        proof_scope=(
            f"exact_length={attempted[0]}"
            if len(attempted) == 1
            else f"lengths={attempted[0]}..{attempted[-1]}"
        ),
        stop_reason=(
            "no additional distinct witness exists in the declared domain"
            if status == NO_ADDITIONAL_DISTINCT_WITNESS
            else "relation is unsatisfiable in the complete declared finite domain"
        ),
        diagnostics={
            "decisions": total_decisions,
            "propagations": total_propagations,
            "unsat_lengths": unsat_lengths,
            "blocked_trace_count": blocked_at_lengths,
        },
    )


def replay_semantic_snapshot_witness(
    snapshot_path: str | Path,
    *,
    node_id: str,
    desired_value: bool,
    horizon_policy: HorizonPolicy,
    blocked_traces: tuple[TraceKey, ...] = (),
) -> WitnessResult:
    """Replay one witness request from a saved semantic snapshot without an LLM."""

    from checkpointing import semantic_from_checkpoint

    payload = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    semantic = semantic_from_checkpoint(payload)
    if semantic.faithful_ir is None:
        return WitnessResult(
            INVALID_OBLIGATION,
            node_id,
            desired_value,
            stop_reason="semantic snapshot has no faithful IR",
        )
    return synthesize_ir_witness(
        WitnessRequest(
            semantic.faithful_ir,
            node_id,
            desired_value,
            horizon_policy,
            blocked_traces=blocked_traces,
            purpose="offline_snapshot_replay",
        )
    )


class _IRCNFEncoding:
    def __init__(self, ir: FaithfulSemanticIR, length: int) -> None:
        self.ir = ir
        self.length = length
        self.clauses: list[tuple[int, ...]] = []
        self._variable_count = 0
        self._ap_variables: dict[tuple[str, int], int] = {}
        self._node_literals: dict[tuple[str, int], int] = {}
        self._gate_cache: dict[tuple[str, tuple[int, ...]], int] = {}
        self._true = self.new_variable()
        self._false = self.new_variable()
        self.add_clause((self._true,))
        self.add_clause((-self._false,))
        for ap in ir.ap_names:
            for at in range(length):
                self._ap_variables[(ap, at)] = self.new_variable()

    @property
    def variable_count(self) -> int:
        return self._variable_count

    def new_variable(self) -> int:
        self._variable_count += 1
        return self._variable_count

    def add_clause(self, literals: Iterable[int]) -> None:
        normalized = tuple(dict.fromkeys(int(item) for item in literals))
        if any(-literal in normalized for literal in normalized):
            return
        self.clauses.append(normalized)

    def and_gate(self, literals: Iterable[int]) -> int:
        items = tuple(literals)
        if not items:
            return self._true
        if len(items) == 1:
            return items[0]
        key = ("and", items)
        if key in self._gate_cache:
            return self._gate_cache[key]
        result = self.new_variable()
        for literal in items:
            self.add_clause((-result, literal))
        self.add_clause((result, *(-literal for literal in items)))
        self._gate_cache[key] = result
        return result

    def or_gate(self, literals: Iterable[int]) -> int:
        items = tuple(literals)
        if not items:
            return self._false
        if len(items) == 1:
            return items[0]
        key = ("or", items)
        if key in self._gate_cache:
            return self._gate_cache[key]
        result = self.new_variable()
        for literal in items:
            self.add_clause((result, -literal))
        self.add_clause((-result, *items))
        self._gate_cache[key] = result
        return result

    def equivalence_gate(self, left: int, right: int) -> int:
        return self.and_gate(
            (self.or_gate((-left, right)), self.or_gate((left, -right)))
        )

    def node_literal(self, node_id: str, at: int) -> int:
        if at < 0 or at >= self.length:
            return self._false
        key = (node_id, at)
        if key in self._node_literals:
            return self._node_literals[key]
        node = self.ir.node_map[node_id]
        kind = resolved_node_kind(node, None)
        children = node.children
        params = dict(node.parameters)

        def child(index: int, position: int = at) -> int:
            return self.node_literal(children[index], position)

        if kind == "ap_condition":
            result = self._ap_variables[(str(_required(params, kind, "ap_name")), at)]
        elif kind == "boolean_constant":
            result = self._true if bool(_required(params, kind, "value")) else self._false
        elif kind == "negation":
            result = -child(0)
        elif kind == "conjunction":
            result = self.and_gate(self.node_literal(item, at) for item in children)
        elif kind == "disjunction":
            result = self.or_gate(self.node_literal(item, at) for item in children)
        elif kind == "implication":
            result = self.or_gate((-child(0), child(1)))
        elif kind == "equivalence":
            result = self.equivalence_gate(child(0), child(1))
        elif kind == "identity":
            result = child(0)
        elif kind in {"global_constraint", "persist_to_end"}:
            result = self.and_gate(child(0, index) for index in range(at, self.length))
        elif kind == "eventually":
            result = self.or_gate(child(0, index) for index in range(at, self.length))
        elif kind == "next_step":
            result = child(0, at + int(_required(params, kind, "steps")))
        elif kind == "eventually_strict":
            result = self.or_gate(child(0, index) for index in range(at + 1, self.length))
        elif kind == "until":
            result = self._encode_until(node_id, at, params)
        elif kind == "release":
            result = self.and_gate(
                self.or_gate(
                    (
                        child(1, target),
                        self.or_gate(child(0, witness) for witness in range(at, target + 1)),
                    )
                )
                for target in range(at, self.length)
            )
        elif kind in {"precedence", "before"}:
            result = self._encode_precedence(node_id, params)
        elif kind == "after":
            result = self.or_gate(
                self.and_gate(
                    (
                        child(0, first),
                        self.or_gate(child(1, later) for later in range(first + 1, self.length)),
                    )
                )
                for first in range(at, self.length)
            )
        elif kind == "fixed_duration":
            duration = int(_required(params, kind, "duration"))
            policy = str(_required(params, kind, "short_trace_policy"))
            terminal = str(_required(params, kind, "terminal_obligation_policy"))
            if (policy, terminal) == ("strong_next", "require_full_duration"):
                result = (
                    self.and_gate(child(0, index) for index in range(at, at + duration))
                    if at + duration <= self.length
                    else self._false
                )
            elif (policy, terminal) == ("available_prefix", "truncate_at_end"):
                result = self.and_gate(
                    child(0, index)
                    for index in range(at, min(at + duration, self.length))
                )
            else:
                raise IRTraceSemanticsError("unsupported fixed_duration policy")
        elif kind == "bounded_window":
            result = self._encode_bounded_window(node_id, at, params)
        elif kind == "ordered_sequence":
            result = self._encode_sequence(children, 0, at, {})
        elif kind == "cardinality":
            result = self._encode_cardinality(node_id, at, params)
        elif kind == "recurrence":
            expected = {
                "origin": "current_position",
                "gap_policy": "unbounded_eventual",
                "short_trace_policy": "finite_suffix",
                "terminal_obligation_policy": "final_state_witness",
            }
            if any(str(_required(params, kind, name)) != value for name, value in expected.items()):
                raise IRTraceSemanticsError("unsupported recurrence policy")
            result = self.and_gate(
                self.or_gate(child(0, later) for later in range(index, self.length))
                for index in range(at, self.length)
            )
        elif kind == "periodicity":
            result = self._encode_periodicity(node_id, at, params)
        elif kind == "start_boundary":
            result = child(0, 0) if at == 0 else self._false
        elif kind == "end_boundary":
            result = self.or_gate(
                self.and_gate(child(0, later) for later in range(index, self.length))
                for index in range(at, self.length)
            )
        elif kind == "short_trace_boundary":
            maximum = max(1, int(_required(params, kind, "maximum_length")))
            result = self._true if self.length - at <= maximum else self._false
        else:
            raise IRTraceSemanticsError(f"IR node type {kind} has no witness encoding")
        self._node_literals[key] = result
        return result

    def _encode_until(self, node_id: str, at: int, params: Mapping[str, object]) -> int:
        node = self.ir.node_map[node_id]
        left, right = node.children
        timing = str(_required(params, "until", "witness_timing"))
        left_at_witness = str(_required(params, "until", "left_at_witness"))
        policy = str(_required(params, "until", "witness_policy"))
        first = at + 1 if timing == "strict_future" else at
        witnesses = []
        for target in range(first, self.length):
            before_end = target + 1 if left_at_witness == "required" else target
            witnesses.append(
                self.and_gate(
                    (
                        self.node_literal(right, target),
                        self.and_gate(
                            self.node_literal(left, index)
                            for index in range(at, before_end)
                        ),
                    )
                )
            )
        if policy == "optional_if_missing":
            witnesses.append(
                self.and_gate(
                    self.node_literal(left, index) for index in range(at, self.length)
                )
            )
        elif policy != "required":
            raise IRTraceSemanticsError("unsupported until witness policy")
        return self.or_gate(witnesses)

    def _encode_precedence(self, node_id: str, params: Mapping[str, object]) -> int:
        node = self.ir.node_map[node_id]
        prerequisite, constrained = node.children
        ordering = str(_required(params, node.node_type, "ordering"))
        policy = str(_required(params, node.node_type, "witness_policy"))
        reuse = str(_required(params, node.node_type, "reuse_policy"))
        origin = str(_required(params, node.node_type, "scope_origin"))
        if reuse != "reusable" or origin != "trace_start":
            raise IRTraceSemanticsError(
                "direct precedence semantics requires reusable/trace_start"
            )
        constraints: list[int] = []
        for index in range(self.length):
            earlier = self.or_gate(
                self.node_literal(prerequisite, before) for before in range(index)
            )
            allowed = (
                self.or_gate((earlier, self.node_literal(prerequisite, index)))
                if ordering == "non_strict"
                else earlier
            )
            constraints.append(
                self.or_gate((-self.node_literal(constrained, index), allowed))
            )
        if policy == "required":
            constraints.append(
                self.or_gate(
                    self.node_literal(prerequisite, index)
                    for index in range(self.length)
                )
            )
        elif policy != "optional_if_no_constrained":
            raise IRTraceSemanticsError("unsupported precedence witness policy")
        return self.and_gate(constraints)

    def _encode_bounded_window(
        self, node_id: str, at: int, params: Mapping[str, object]
    ) -> int:
        child_id = self.ir.node_map[node_id].children[0]
        lower = int(_required(params, "bounded_window", "lower_bound"))
        upper = int(_required(params, "bounded_window", "upper_bound"))
        policy = str(_required(params, "bounded_window", "short_trace_policy"))
        terminal = str(_required(params, "bounded_window", "terminal_obligation_policy"))
        witness = self.or_gate(
            self.node_literal(child_id, index)
            for index in range(at + lower, min(at + upper + 1, self.length))
        )
        if policy == "strong_upper_bound" and terminal == "require_witness":
            return witness if at + upper < self.length else self._false
        if policy == "available_prefix" and terminal == "require_witness":
            return witness
        if policy == "available_prefix" and terminal == "vacuous_if_window_unavailable":
            return self._true if at + lower >= self.length else witness
        raise IRTraceSemanticsError("unsupported bounded_window policy")

    def _encode_sequence(
        self,
        children: tuple[str, ...],
        child_index: int,
        cursor: int,
        memo: dict[tuple[int, int], int],
    ) -> int:
        key = (child_index, cursor)
        if key in memo:
            return memo[key]
        if child_index >= len(children):
            return self._true
        result = self.or_gate(
            self.and_gate(
                (
                    self.node_literal(children[child_index], index),
                    self._encode_sequence(children, child_index + 1, index + 1, memo),
                )
            )
            for index in range(cursor, self.length)
        )
        memo[key] = result
        return result

    def _encode_cardinality(
        self, node_id: str, at: int, params: Mapping[str, object]
    ) -> int:
        child_id = self.ir.node_map[node_id].children[0]
        literals = tuple(
            self.node_literal(child_id, index) for index in range(at, self.length)
        )
        count = max(0, int(_required(params, "cardinality", "count")))
        mode = str(_required(params, "cardinality", "mode"))
        exact = self._exact_count_literals(literals)
        if mode == "exactly":
            return exact[count] if count < len(exact) else self._false
        if mode == "at_most":
            return self.or_gate(exact[: min(count + 1, len(exact))])
        if mode == "at_least":
            return self.or_gate(exact[count:]) if count < len(exact) else self._false
        raise IRTraceSemanticsError(f"unsupported cardinality mode: {mode}")

    def _exact_count_literals(self, literals: tuple[int, ...]) -> tuple[int, ...]:
        distribution: tuple[int, ...] = (self._true,)
        for literal in literals:
            updated: list[int] = []
            for count in range(len(distribution) + 1):
                without = (
                    self.and_gate((distribution[count], -literal))
                    if count < len(distribution)
                    else self._false
                )
                with_current = (
                    self.and_gate((distribution[count - 1], literal))
                    if count > 0
                    else self._false
                )
                updated.append(self.or_gate((without, with_current)))
            distribution = tuple(updated)
        return distribution

    def _encode_periodicity(
        self, node_id: str, at: int, params: Mapping[str, object]
    ) -> int:
        child_id = self.ir.node_map[node_id].children[0]
        period = int(_required(params, "periodicity", "period"))
        mode = str(_required(params, "periodicity", "mode"))
        terminal = str(_required(params, "periodicity", "terminal_policy"))
        if period < 1:
            raise IRTraceSemanticsError("period must be >= 1")
        values = tuple(
            self.node_literal(child_id, index) for index in range(at, self.length)
        )
        if mode == "fixed_phase":
            phase = int(_required(params, "periodicity", "phase"))
            off_schedule = str(_required(params, "periodicity", "off_schedule_policy"))
            if phase < 0 or phase >= period or off_schedule != "forbidden":
                raise IRTraceSemanticsError("invalid fixed_phase contract")
            required_positions = set(range(phase, self.length, period))
            if any(position < at for position in required_positions):
                return self._false
            if not required_positions:
                return (
                    self.and_gate(-value for value in values)
                    if terminal == "finite_prefix"
                    else self._false
                )
            return self.and_gate(
                self.node_literal(child_id, index)
                if index in required_positions
                else -self.node_literal(child_id, index)
                for index in range(at, self.length)
            )

        occurrence = str(_required(params, "periodicity", "occurrence_policy"))
        if terminal != "finite_prefix" or occurrence not in {"required", "optional"}:
            raise IRTraceSemanticsError("invalid event-relative periodicity contract")
        constraints: list[int] = []
        for index in range(at, self.length):
            current = self.node_literal(child_id, index)
            for close in range(index + 1, min(index + period, self.length)):
                constraints.append(
                    self.or_gate((-current, -self.node_literal(child_id, close)))
                )
            scheduled = index + period
            if mode == "event_relative_exact_recurrence" and scheduled < self.length:
                constraints.append(
                    self.or_gate((-current, self.node_literal(child_id, scheduled)))
                )
            elif mode == "exact_inter_occurrence_spacing" and scheduled < self.length:
                later = self.or_gate(
                    self.node_literal(child_id, future)
                    for future in range(scheduled + 1, self.length)
                )
                constraints.append(
                    self.or_gate((-current, -later, self.node_literal(child_id, scheduled)))
                )
            elif mode not in {
                "minimum_spacing",
                "exact_inter_occurrence_spacing",
                "event_relative_exact_recurrence",
            }:
                raise IRTraceSemanticsError(f"unsupported periodicity mode: {mode}")
        if occurrence == "required":
            constraints.append(self.or_gate(values))
        return self.and_gate(constraints)

    def block_trace(self, blocked: TraceKey) -> None:
        differences: list[int] = []
        for at, state in enumerate(blocked):
            enabled = set(state)
            for ap in self.ir.ap_names:
                variable = self._ap_variables[(ap, at)]
                differences.append(-variable if ap in enabled else variable)
        self.add_clause(differences)

    def trace_from_model(self, model: Mapping[int, bool]) -> list[set[str]]:
        return [
            {
                ap
                for ap in self.ir.ap_names
                if model.get(self._ap_variables[(ap, at)], False)
            }
            for at in range(self.length)
        ]


class _DPLLSolver:
    def __init__(
        self,
        clauses: Iterable[Iterable[int]],
        variable_count: int,
        *,
        max_decisions: int,
        prefer_true: bool,
    ) -> None:
        self.clauses = tuple(tuple(clause) for clause in clauses)
        self.variable_count = variable_count
        self.max_decisions = max_decisions
        self.prefer_true = prefer_true
        self.decisions = 0
        self.propagations = 0
        self.resource_limited = False

    def solve(self) -> dict[int, bool] | None:
        return self._search({})

    def _search(self, assignment: dict[int, bool]) -> dict[int, bool] | None:
        if self.resource_limited:
            return None
        if not self._propagate(assignment):
            return None
        unresolved = self._unresolved_clauses(assignment)
        if not unresolved:
            return assignment
        if self.decisions >= self.max_decisions:
            self.resource_limited = True
            return None
        variable = self._choose_variable(unresolved, assignment)
        if variable is None:
            return None
        self.decisions += 1
        for value in (self.prefer_true, not self.prefer_true):
            branch = dict(assignment)
            branch[variable] = value
            result = self._search(branch)
            if result is not None:
                return result
            if self.resource_limited:
                return None
        return None

    def _propagate(self, assignment: dict[int, bool]) -> bool:
        changed = True
        while changed:
            changed = False
            for clause in self.clauses:
                satisfied = False
                unassigned: list[int] = []
                for literal in clause:
                    value = assignment.get(abs(literal))
                    if value is None:
                        unassigned.append(literal)
                    elif value == (literal > 0):
                        satisfied = True
                        break
                if satisfied:
                    continue
                if not unassigned:
                    return False
                if len(unassigned) == 1:
                    literal = unassigned[0]
                    variable = abs(literal)
                    required = literal > 0
                    existing = assignment.get(variable)
                    if existing is not None and existing != required:
                        return False
                    if existing is None:
                        assignment[variable] = required
                        self.propagations += 1
                        changed = True
        return True

    def _unresolved_clauses(
        self, assignment: Mapping[int, bool]
    ) -> tuple[tuple[int, ...], ...]:
        unresolved: list[tuple[int, ...]] = []
        for clause in self.clauses:
            if any(
                assignment.get(abs(literal)) == (literal > 0)
                for literal in clause
                if abs(literal) in assignment
            ):
                continue
            remaining = tuple(
                literal for literal in clause if abs(literal) not in assignment
            )
            if remaining:
                unresolved.append(remaining)
        return tuple(unresolved)

    @staticmethod
    def _choose_variable(
        clauses: tuple[tuple[int, ...], ...], assignment: Mapping[int, bool]
    ) -> int | None:
        scores: dict[int, tuple[int, int]] = {}
        for clause in clauses:
            weight = max(1, 16 - min(15, len(clause)))
            for literal in clause:
                variable = abs(literal)
                if variable in assignment:
                    continue
                count, current_weight = scores.get(variable, (0, 0))
                scores[variable] = (count + 1, current_weight + weight)
        if not scores:
            return None
        return min(
            scores,
            key=lambda variable: (
                -scores[variable][1],
                -scores[variable][0],
                variable,
            ),
        )


def _required(params: Mapping[str, object], kind: str, name: str) -> object:
    if name not in params or params[name] in {"", None}:
        raise IRTraceSemanticsError(f"{kind} requires explicit parameter {name}")
    return params[name]
