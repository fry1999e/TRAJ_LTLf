"""Executable, formula-free checks for semantic-neighbour trace obligations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Iterable

from ir_trace_semantics import IRTraceSemanticsError, evaluate_ir_trace


DISCRIMINATOR_WITNESS_FOUND = "witness_found"
DISCRIMINATOR_UNSAT_IN_SEARCH_DOMAIN = "unsat_in_search_domain"
DISCRIMINATOR_RESOURCE_LIMIT = "resource_limit"


@dataclass(frozen=True)
class DiscriminatorSynthesisResult:
    """Auditable result of a whole-IR semantic-neighbour search."""

    status: str
    node_id: str
    confusion_id: str
    witness: tuple[frozenset[str], ...] | None = None
    attempted_lengths: tuple[int, ...] = ()
    checked_candidate_count: int = 0
    stop_reason: str = ""

    @property
    def witness_length(self) -> int | None:
        return len(self.witness) if self.witness is not None else None

    def trace(self) -> list[set[str]] | None:
        if self.witness is None:
            return None
        return [set(step) for step in self.witness]


def discriminator_values(
    ir,
    node_id: str,
    confusion_id: str,
    trace: list[set[str]],
) -> tuple[bool, bool] | None:
    """Return the two neighbour readings on ``trace``, if implemented."""

    node = ir.node_map.get(node_id)
    if node is None or not trace:
        return None
    confusion = str(confusion_id)

    neighbour = None
    extra_nodes = ()

    if confusion in {"current_vs_global_scope", "local_implication_vs_global_response"}:
        if node.node_type in {"global_constraint", "persist_to_end"}:
            neighbour = replace(node, node_type="identity", parameters=())
        else:
            copy_id = _reserved_node_id(ir, node_id + "-local-copy")
            copy = replace(node, node_id=copy_id)
            neighbour = replace(
                node,
                node_type="global_constraint",
                children=(copy_id,),
                parameters=(),
            )
            extra_nodes = (copy,)
    elif confusion == "eventually_vs_next":
        if not node.children:
            return None
        if node.node_type == "next_step":
            neighbour = replace(node, node_type="eventually", parameters=())
        else:
            neighbour = replace(
                node,
                node_type="next_step",
                parameters=(("steps", 1),),
            )
    elif confusion == "strict_eventually_vs_next":
        if not node.children:
            return None
        if node.node_type == "next_step":
            neighbour = replace(node, node_type="eventually_strict", parameters=())
        else:
            neighbour = replace(
                node,
                node_type="next_step",
                parameters=(("steps", 1),),
            )
    elif confusion == "current_vs_strict_future":
        if not node.children:
            return None
        if node.node_type in {"eventually_strict", "strict_future"}:
            neighbour = replace(node, node_type="identity", parameters=())
        else:
            neighbour = replace(node, node_type="eventually_strict", parameters=())
    elif confusion == "next_one_vs_next_two":
        if node.node_type != "next_step":
            return None
        steps = int(node.parameter("steps", 1))
        neighbour = replace(node, parameters=(("steps", 2 if steps == 1 else 1),))
    elif confusion in {"next_exact_vs_early", "next_exact_vs_late"}:
        if node.node_type != "next_step":
            return None
        exact = max(1, int(node.parameter("steps", 1)))
        neighbour_offset = (
            max(0, exact - 1) if confusion.endswith("early") else exact + 1
        )
        if neighbour_offset == 0:
            neighbour = replace(node, node_type="identity", parameters=())
        else:
            neighbour = replace(node, parameters=(("steps", neighbour_offset),))
    elif confusion == "bounded_window_vs_unbounded_eventually":
        if node.node_type != "bounded_window" or not node.children:
            return None
        neighbour = replace(node, node_type="eventually", parameters=())
    elif confusion in {
        "strong_until_vs_weak_until",
        "until_current_vs_strict_future",
        "until_left_at_witness",
        "precedence_required_vs_optional",
        "precedence_strict_vs_non_strict",
        "periodicity_terminal_policy",
        "cardinality_threshold_neighbor",
    }:
        parameters = dict(node.parameters)
        if confusion == "strong_until_vs_weak_until":
            parameters["witness_policy"] = (
                "required"
                if parameters.get("witness_policy") == "optional_if_missing"
                else "optional_if_missing"
            )
        elif confusion == "until_current_vs_strict_future":
            if node.node_type != "until":
                return None
            parameters["witness_timing"] = (
                "strict_future"
                if parameters.get("witness_timing") == "current_or_future"
                else "current_or_future"
            )
        elif confusion == "until_left_at_witness":
            if node.node_type != "until":
                return None
            parameters["left_at_witness"] = (
                "not_required"
                if parameters.get("left_at_witness") == "required"
                else "required"
            )
        elif confusion == "precedence_required_vs_optional":
            parameters["witness_policy"] = (
                "required"
                if parameters.get("witness_policy") == "optional_if_no_constrained"
                else "optional_if_no_constrained"
            )
        elif confusion == "precedence_strict_vs_non_strict":
            if node.node_type not in {"precedence", "before"}:
                return None
            strict = parameters.get("ordering") == "strict"
            parameters["ordering"] = "non_strict" if strict else "strict"
            parameters["same_step_policy"] = "allowed" if strict else "forbidden"
        elif confusion == "periodicity_terminal_policy":
            if node.node_type != "periodicity" or parameters.get("mode") != "fixed_phase":
                return None
            parameters["terminal_policy"] = (
                "require_first_scheduled_witness"
                if parameters.get("terminal_policy") == "finite_prefix"
                else "finite_prefix"
            )
        else:
            parameters["count"] = int(parameters.get("count", 0)) + 1
        neighbour = replace(node, parameters=tuple(parameters.items()))
    elif confusion == "simultaneous_vs_independent_eventuality":
        if node.node_type == "eventually" and node.children:
            conjunction = ir.node_map.get(node.children[0])
            if conjunction is None or conjunction.node_type != "conjunction":
                return None
            wrappers = tuple(
                replace(
                    node,
                    node_id=_reserved_node_id(ir, f"{node_id}-eventually-{index}"),
                    children=(child,),
                    parameters=(),
                )
                for index, child in enumerate(conjunction.children, start=1)
            )
            neighbour = replace(
                node,
                node_type="conjunction",
                children=tuple(item.node_id for item in wrappers),
                parameters=(),
            )
            extra_nodes = wrappers
        elif node.node_type == "conjunction" and node.children:
            eventuals = tuple(ir.node_map[child] for child in node.children)
            if not all(
                item.node_type == "eventually" and len(item.children) == 1
                for item in eventuals
            ):
                return None
            conjunction_id = _reserved_node_id(ir, node_id + "-joint-condition")
            conjunction = replace(
                node,
                node_id=conjunction_id,
                children=tuple(item.children[0] for item in eventuals),
            )
            neighbour = replace(
                node,
                node_type="eventually",
                children=(conjunction_id,),
                parameters=(),
            )
            extra_nodes = (conjunction,)
        else:
            return None
    else:
        return None

    neighbour_ir = replace(
        ir,
        nodes=tuple(
            neighbour if item.node_id == node_id else item
            for item in ir.nodes
        )
        + tuple(extra_nodes),
        ir_version="",
    )
    return evaluate_ir_trace(ir, trace), evaluate_ir_trace(neighbour_ir, trace)


def _reserved_node_id(ir, stem: str) -> str:
    candidate = stem
    suffix = 2
    while candidate in ir.node_map:
        candidate = f"{stem}-{suffix}"
        suffix += 1
    return candidate


def trace_distinguishes_confusion(
    ir,
    node_id: str,
    confusion_id: str,
    trace: list[set[str]],
) -> bool:
    try:
        values = discriminator_values(ir, node_id, confusion_id, trace)
    except (IRTraceSemanticsError, KeyError, TypeError, ValueError):
        return False
    return (
        values is not None
        and values[0] != values[1]
        and _meets_discriminator_shape(ir, node_id, confusion_id, trace)
    )


def _meets_discriminator_shape(ir, node_id: str, confusion_id: str, trace) -> bool:
    """Enforce the semantic feature named by the catalog, not any accidental split."""

    if confusion_id not in {"eventually_vs_next", "strict_eventually_vs_next"}:
        return True
    node = ir.node_map.get(node_id)
    if node is None or not node.children or len(trace) < 3:
        return False
    child_id = node.children[0]
    child_values = tuple(
        evaluate_ir_trace(ir, trace, node_id=child_id, position=position)
        for position in range(len(trace))
    )
    delayed_shape = (
        not child_values[0]
        and not child_values[1]
        and any(child_values[2:])
    )
    if not delayed_shape:
        return False
    implication_parent = next(
        (
            candidate
            for candidate in ir.nodes
            if candidate.node_type == "implication"
            and len(candidate.children) == 2
            and candidate.children[1] == node_id
        ),
        None,
    )
    if implication_parent is None:
        return True
    trigger_id = implication_parent.children[0]
    trigger_values = tuple(
        evaluate_ir_trace(ir, trace, node_id=trigger_id, position=position)
        for position in range(len(trace))
    )
    return trigger_values[0] and not any(trigger_values[1:])


def synthesize_discriminator_trace(
    ir,
    node_id: str,
    confusion_id: str,
    *,
    minimum_horizon: int,
    exact_horizon: int | None = None,
    seen: Iterable[tuple[tuple[str, ...], ...]] = (),
    variant: int = 0,
    max_candidates: int = 100_000,
) -> DiscriminatorSynthesisResult:
    """Search the declared horizon domain for a whole-root discriminator."""

    aps = tuple(ir.ap_names)
    if len(aps) <= 5:
        states = [
            {ap for ap, enabled in zip(aps, mask) if enabled}
            for mask in product((False, True), repeat=len(aps))
        ] or [set()]
    else:
        states = [set(), *({ap} for ap in aps), set(aps)]
    offset = variant % len(states)
    states = states[offset:] + states[:offset]
    excluded = set(seen)
    checked = 0
    start = max(1, int(minimum_horizon))
    horizons = (
        (max(1, int(exact_horizon)),)
        if exact_horizon is not None
        else tuple(range(start, start + 2))
    )
    attempted: list[int] = []
    for horizon in horizons:
        attempted.append(horizon)
        for raw in product(states, repeat=horizon):
            checked += 1
            if checked > max_candidates:
                return DiscriminatorSynthesisResult(
                    DISCRIMINATOR_RESOURCE_LIMIT,
                    node_id,
                    str(confusion_id),
                    attempted_lengths=tuple(attempted),
                    checked_candidate_count=checked - 1,
                    stop_reason="discriminator candidate budget exhausted",
                )
            candidate = [set(step) for step in raw]
            key = tuple(tuple(sorted(step)) for step in candidate)
            if key in excluded:
                continue
            if trace_distinguishes_confusion(ir, node_id, confusion_id, candidate):
                return DiscriminatorSynthesisResult(
                    DISCRIMINATOR_WITNESS_FOUND,
                    node_id,
                    str(confusion_id),
                    witness=tuple(frozenset(step) for step in candidate),
                    attempted_lengths=tuple(attempted),
                    checked_candidate_count=checked,
                    stop_reason=(
                        "root IR and its semantic neighbour evaluate differently"
                    ),
                )
    return DiscriminatorSynthesisResult(
        DISCRIMINATOR_UNSAT_IN_SEARCH_DOMAIN,
        node_id,
        str(confusion_id),
        attempted_lengths=tuple(attempted),
        checked_candidate_count=checked,
        stop_reason="no discriminator exists in the declared horizon domain",
    )


def find_discriminator_trace(
    ir,
    node_id: str,
    confusion_id: str,
    *,
    minimum_horizon: int,
    exact_horizon: int | None = None,
    seen: Iterable[tuple[tuple[str, ...], ...]] = (),
    variant: int = 0,
    max_candidates: int = 100_000,
) -> list[set[str]] | None:
    """Compatibility wrapper for callers that only need the witness trace."""

    return synthesize_discriminator_trace(
        ir,
        node_id,
        confusion_id,
        minimum_horizon=minimum_horizon,
        exact_horizon=exact_horizon,
        seen=seen,
        variant=variant,
        max_candidates=max_candidates,
    ).trace()


__all__ = [
    "DISCRIMINATOR_RESOURCE_LIMIT",
    "DISCRIMINATOR_UNSAT_IN_SEARCH_DOMAIN",
    "DISCRIMINATOR_WITNESS_FOUND",
    "DiscriminatorSynthesisResult",
    "discriminator_values",
    "find_discriminator_trace",
    "synthesize_discriminator_trace",
    "trace_distinguishes_confusion",
]
