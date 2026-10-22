"""Dependency-light relation parameter contracts shared by prompts and validators.

The mode matrix is authoritative.  Callers may render it for an LLM or enforce
it mechanically, but must not maintain a second handwritten required/forbidden
field list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


PERIODICITY_PARAMETER_VALUES = {
    "mode": frozenset(
        {
            "fixed_phase",
            "event_relative_exact_recurrence",
            "exact_inter_occurrence_spacing",
            "minimum_spacing",
        }
    ),
    "origin": frozenset({"trace_start", "first_occurrence"}),
    "occurrence_policy": frozenset({"required", "optional"}),
    "off_schedule_policy": frozenset({"forbidden"}),
    "terminal_policy": frozenset(
        {"finite_prefix", "require_first_scheduled_witness"}
    ),
}


@dataclass(frozen=True)
class PeriodicityModeContract:
    required: frozenset[str]
    forbidden: frozenset[str]
    fixed_values: Mapping[str, str]


_COMMON_PERIODICITY_PARAMETERS = frozenset(
    {"period", "mode", "origin", "terminal_policy"}
)

PERIODICITY_MODE_CONTRACTS = {
    "fixed_phase": PeriodicityModeContract(
        required=_COMMON_PERIODICITY_PARAMETERS
        | frozenset({"phase", "off_schedule_policy"}),
        # occurrence_policy is the only mode-incompatible field rejected for
        # fixed schedules.
        forbidden=frozenset({"occurrence_policy"}),
        fixed_values={
            "origin": "trace_start",
            "off_schedule_policy": "forbidden",
        },
    ),
    "event_relative_exact_recurrence": PeriodicityModeContract(
        required=_COMMON_PERIODICITY_PARAMETERS | frozenset({"occurrence_policy"}),
        forbidden=frozenset({"phase", "off_schedule_policy", "off_phase_policy"}),
        fixed_values={
            "origin": "first_occurrence",
            "terminal_policy": "finite_prefix",
        },
    ),
    "exact_inter_occurrence_spacing": PeriodicityModeContract(
        required=_COMMON_PERIODICITY_PARAMETERS | frozenset({"occurrence_policy"}),
        forbidden=frozenset({"phase", "off_schedule_policy", "off_phase_policy"}),
        fixed_values={
            "origin": "first_occurrence",
            "terminal_policy": "finite_prefix",
        },
    ),
    "minimum_spacing": PeriodicityModeContract(
        required=_COMMON_PERIODICITY_PARAMETERS | frozenset({"occurrence_policy"}),
        forbidden=frozenset({"phase", "off_schedule_policy", "off_phase_policy"}),
        fixed_values={
            "origin": "first_occurrence",
            "terminal_policy": "finite_prefix",
        },
    ),
}


def periodicity_mode_contract(mode: object) -> PeriodicityModeContract | None:
    return PERIODICITY_MODE_CONTRACTS.get(str(mode or "").strip())


def periodicity_mode_constraints_payload() -> dict[str, dict[str, object]]:
    """Return a stable JSON-ready mode matrix for structured generation prompts."""

    return {
        mode: {
            "required": sorted(contract.required),
            "forbidden": sorted(contract.forbidden),
            "fixed_values": dict(sorted(contract.fixed_values.items())),
        }
        for mode, contract in sorted(PERIODICITY_MODE_CONTRACTS.items())
    }


__all__ = [
    "PERIODICITY_MODE_CONTRACTS",
    "PERIODICITY_PARAMETER_VALUES",
    "PeriodicityModeContract",
    "periodicity_mode_constraints_payload",
    "periodicity_mode_contract",
]
