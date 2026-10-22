"""Typed registry for reviewable semantic axes.

This module identifies *which* IR parameters form one semantic decision.  The
node contract remains authoritative for legal values and temporal policy
profiles remain authoritative for executable option bundles.  Keeping those
responsibilities separate prevents prompts, UI code, and repair logic from
inventing their own lists.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from semantic_node_contracts import NODE_CONTRACTS


SEMANTIC_AXIS_REGISTRY_VERSION = "semantic-axes-v1"


@dataclass(frozen=True)
class SemanticAxisContract:
    axis_id: str
    node_type: str
    parameters: tuple[str, ...]
    question: str
    source_kind: str = "semantic"


SEMANTIC_AXES: Mapping[str, SemanticAxisContract] = {
    "bounded_window_bounds": SemanticAxisContract(
        "bounded_window_bounds",
        "bounded_window",
        ("lower_bound", "upper_bound", "origin", "minimum_horizon"),
        "Which positions belong to the bounded window?",
    ),
    "bounded_window_terminal": SemanticAxisContract(
        "bounded_window_terminal",
        "bounded_window",
        ("short_trace_policy", "terminal_obligation_policy"),
        "How is an incomplete bounded window handled at the end of a finite run?",
        "project_policy",
    ),
    "periodicity_terminal": SemanticAxisContract(
        "periodicity_terminal",
        "periodicity",
        ("terminal_policy",),
        "Must the first scheduled periodic witness exist before the finite run ends?",
        "project_policy",
    ),
    "periodicity_schedule": SemanticAxisContract(
        "periodicity_schedule",
        "periodicity",
        (
            "period",
            "phase",
            "mode",
            "origin",
            "off_schedule_policy",
            "occurrence_policy",
        ),
        "What anchors the periodic schedule and which occurrences are allowed?",
    ),
    "exact_offset": SemanticAxisContract(
        "exact_offset",
        "next_step",
        ("steps",),
        "How many successor positions later must the child hold?",
    ),
    "until_witness_policy": SemanticAxisContract(
        "until_witness_policy",
        "until",
        ("witness_policy",),
        "Is the right-hand witness required, or may it be absent?",
    ),
    "until_witness_timing": SemanticAxisContract(
        "until_witness_timing",
        "until",
        ("witness_timing",),
        "May the right-hand witness occur now, or only strictly later?",
    ),
    "until_left_at_witness": SemanticAxisContract(
        "until_left_at_witness",
        "until",
        ("left_at_witness",),
        "Must the left condition still hold at the witness state?",
    ),
    "precedence_ordering": SemanticAxisContract(
        "precedence_ordering",
        "precedence",
        ("ordering", "same_step_policy"),
        "May prerequisite and constrained event occur at the same state?",
    ),
    "precedence_scope": SemanticAxisContract(
        "precedence_scope",
        "precedence",
        ("scope_origin",),
        "Does precedence start at trace start or at a local activation?",
    ),
}


def _normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def validate_semantic_axes() -> None:
    for axis_id, axis in SEMANTIC_AXES.items():
        if axis.axis_id != axis_id:
            raise ValueError(f"semantic axis key/id mismatch: {axis_id}")
        contract = NODE_CONTRACTS.get(axis.node_type)
        if contract is None:
            raise ValueError(f"semantic axis {axis_id} has unknown node type")
        unknown = set(axis.parameters) - set(contract.parameters)
        if unknown:
            raise ValueError(
                f"semantic axis {axis_id} has unknown parameters: "
                + ", ".join(sorted(unknown))
            )


validate_semantic_axes()


def semantic_axis_for_request(
    *,
    node_type: object = "",
    axis_kind: object = "",
    unresolved_parameters: tuple[str, ...] = (),
) -> SemanticAxisContract | None:
    """Resolve a request to exactly one registered axis; never guess ties."""

    requested_node_type = _normalized(node_type)
    requested_axis = _normalized(axis_kind)
    names = set(unresolved_parameters)
    candidates = tuple(
        axis
        for axis in SEMANTIC_AXES.values()
        if (not requested_node_type or axis.node_type == requested_node_type)
        and names
        and names.issubset(set(axis.parameters))
    )
    if len(candidates) == 1:
        return candidates[0]
    exact = tuple(axis for axis in candidates if axis.axis_id == requested_axis)
    if len(exact) == 1:
        return exact[0]
    fuzzy = tuple(
        axis
        for axis in candidates
        if requested_axis
        and (
            requested_axis in axis.axis_id
            or all(token in requested_axis for token in axis.axis_id.split("_")[1:])
        )
    )
    return fuzzy[0] if len(fuzzy) == 1 else None


def semantic_axes_payload() -> dict[str, object]:
    return {
        "registry_version": SEMANTIC_AXIS_REGISTRY_VERSION,
        "axes": {
            axis_id: {
                "node_type": axis.node_type,
                "parameters": list(axis.parameters),
                "question": axis.question,
                "source_kind": axis.source_kind,
            }
            for axis_id, axis in sorted(SEMANTIC_AXES.items())
        },
    }


def semantic_axes_for_parameter(
    node_type: str,
    parameter_name: str,
) -> tuple[SemanticAxisContract, ...]:
    return tuple(
        axis
        for axis in SEMANTIC_AXES.values()
        if axis.node_type == node_type and parameter_name in axis.parameters
    )


__all__ = [
    "SEMANTIC_AXES",
    "SEMANTIC_AXIS_REGISTRY_VERSION",
    "SemanticAxisContract",
    "semantic_axis_for_request",
    "semantic_axes_for_parameter",
    "semantic_axes_payload",
]
