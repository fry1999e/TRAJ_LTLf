"""Reviewed, reusable finite-trace policy bundles for semantic extraction.

Profiles do not guess a requirement reading. They fill only policy axes that
the source and the selected ordinary-language reading intentionally leave
unspecified. Keeping those responsibilities separate prevents a project
default from silently changing a human-selected meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from semantic_node_contracts import NODE_CONTRACTS


@dataclass(frozen=True)
class TemporalPolicyProfile:
    profile_id: str
    node_type: str
    reading: str
    parameters: Mapping[str, object]
    controlled_parameters: tuple[str, ...]
    policy_axis: str
    user_label: str
    user_explanation: str
    version: str = "1"
    derived_parameters: tuple[str, ...] = ()


TEMPORAL_POLICY_PROFILES: Mapping[str, TemporalPolicyProfile] = {
    "bounded-window-current-or-next-v1": TemporalPolicyProfile(
        profile_id="bounded-window-current-or-next-v1",
        node_type="bounded_window",
        reading="the child holds now or at the immediately following position",
        parameters={
            "lower_bound": 0,
            "upper_bound": 1,
            "origin": "current_position",
            "minimum_horizon": 1,
            "short_trace_policy": "available_prefix",
            "terminal_obligation_policy": "require_witness",
        },
        controlled_parameters=(
            "lower_bound",
            "upper_bound",
            "origin",
            "minimum_horizon",
            "short_trace_policy",
            "terminal_obligation_policy",
        ),
        policy_axis="complete_bounded_window_reading",
        user_label="当前或下一步出现即可",
        user_explanation=(
            "即使已经到轨迹末尾，只要当前这一步满足条件，也算找到见证。"
        ),
    ),
    "bounded-window-future-witness-v1": TemporalPolicyProfile(
        profile_id="bounded-window-future-witness-v1",
        node_type="bounded_window",
        reading=(
            "a witness must occur in the available part of a strict-future "
            "bounded window"
        ),
        parameters={
            "origin": "current_position",
            "minimum_horizon": 1,
            "short_trace_policy": "available_prefix",
            "terminal_obligation_policy": "require_witness",
        },
        controlled_parameters=(
            "origin",
            "minimum_horizon",
            "short_trace_policy",
            "terminal_obligation_policy",
        ),
        policy_axis="bounded_window_terminal",
        user_label="只检查实际存在的后续步骤，并且必须找到见证",
        user_explanation=(
            "如果末尾不足完整窗口，就检查仍然存在的后续步骤；只要还有可检查的步骤，"
            "就必须在其中至少找到一次满足条件的情况。窗口的起止步数仍由人工选定的句意决定。"
        ),
    ),
    "bounded-window-full-window-v1": TemporalPolicyProfile(
        profile_id="bounded-window-full-window-v1",
        node_type="bounded_window",
        reading="the complete bounded window must exist and contain a witness",
        parameters={
            "origin": "current_position",
            "short_trace_policy": "strong_upper_bound",
            "terminal_obligation_policy": "require_witness",
        },
        controlled_parameters=(
            "origin",
            "minimum_horizon",
            "short_trace_policy",
            "terminal_obligation_policy",
        ),
        policy_axis="bounded_window_terminal",
        user_label="必须存在完整窗口，并且必须找到见证",
        user_explanation=(
            "如果剩余轨迹不足完整窗口，该要求不成立；完整窗口内还必须至少出现一次满足条件的情况。"
        ),
        derived_parameters=("minimum_horizon",),
    ),
    "bounded-window-unavailable-vacuous-v1": TemporalPolicyProfile(
        profile_id="bounded-window-unavailable-vacuous-v1",
        node_type="bounded_window",
        reading=(
            "use the available prefix, but treat a window with no reachable "
            "positions as vacuously satisfied"
        ),
        parameters={
            "origin": "current_position",
            "minimum_horizon": 1,
            "short_trace_policy": "available_prefix",
            "terminal_obligation_policy": "vacuous_if_window_unavailable",
        },
        controlled_parameters=(
            "origin",
            "minimum_horizon",
            "short_trace_policy",
            "terminal_obligation_policy",
        ),
        policy_axis="bounded_window_terminal",
        user_label="没有任何可检查步骤时自动满足",
        user_explanation=(
            "如果末尾连窗口的第一步都不存在，该要求自动视为满足；否则仍在实际存在的窗口中寻找见证。"
        ),
    ),
    "periodicity-finite-prefix-v1": TemporalPolicyProfile(
        profile_id="periodicity-finite-prefix-v1",
        node_type="periodicity",
        reading="only scheduled positions that exist in the finite prefix are checked",
        parameters={"terminal_policy": "finite_prefix"},
        controlled_parameters=("terminal_policy",),
        policy_axis="periodicity_terminal",
        user_label="只检查有限轨迹中实际存在的周期位置",
        user_explanation=(
            "如果轨迹在第一个或下一个计划位置之前结束，不要求补出轨迹之外的状态。"
        ),
    ),
    "periodicity-require-first-witness-v1": TemporalPolicyProfile(
        profile_id="periodicity-require-first-witness-v1",
        node_type="periodicity",
        reading="the first scheduled periodic witness must exist in the finite run",
        parameters={"terminal_policy": "require_first_scheduled_witness"},
        controlled_parameters=("terminal_policy",),
        policy_axis="periodicity_terminal",
        user_label="有限轨迹内必须出现第一个计划周期位置",
        user_explanation=(
            "如果轨迹在第一个计划位置之前结束，该周期要求判为不满足。"
        ),
    ),
}


# Legal candidates are versioned, but no candidate is a hidden project default.
# Multi-option axes must be resolved by discriminating expert trace evidence.
SEMANTIC_POLICY_BINDINGS: Mapping[str, str] = {}


def validate_temporal_policy_profiles() -> None:
    """Fail fast when a registered policy is outside the executable contract."""

    for profile_id, profile in TEMPORAL_POLICY_PROFILES.items():
        if profile.profile_id != profile_id:
            raise ValueError(f"temporal policy key/id mismatch: {profile_id}")
        contract = NODE_CONTRACTS.get(profile.node_type)
        if contract is None:
            raise ValueError(
                f"temporal policy {profile_id} uses unknown node type "
                f"{profile.node_type!r}"
            )
        unknown = set(profile.parameters) - set(contract.parameters)
        if unknown:
            raise ValueError(
                f"temporal policy {profile_id} has unknown parameters: "
                + ", ".join(sorted(unknown))
            )
        if set(profile.parameters) | set(profile.derived_parameters) != set(
            profile.controlled_parameters
        ):
            raise ValueError(
                f"temporal policy {profile_id} controlled_parameters must match "
                "its parameter bundle"
            )
        for name, value in profile.parameters.items():
            if not contract.parameters[name].accepts(value):
                raise ValueError(
                    f"temporal policy {profile_id} has invalid {name}={value!r}"
                )
        short_policy = profile.parameters.get("short_trace_policy")
        terminal_policy = profile.parameters.get("terminal_obligation_policy")
        if (
            profile.node_type == "bounded_window"
            and short_policy == "strong_upper_bound"
            and terminal_policy != "require_witness"
        ):
            raise ValueError(
                f"temporal policy {profile_id} combines strong_upper_bound with "
                "an illegal terminal policy"
            )
    for reading_class, profile_id in SEMANTIC_POLICY_BINDINGS.items():
        if profile_id not in TEMPORAL_POLICY_PROFILES:
            raise ValueError(
                f"semantic policy binding {reading_class} refers to unknown "
                f"profile {profile_id}"
            )
    if SEMANTIC_POLICY_BINDINGS:
        raise ValueError(
            "finite-trace policy candidates must not define hidden default bindings"
        )
    from semantic_axes import SEMANTIC_AXES

    for axis in SEMANTIC_AXES.values():
        if axis.source_kind != "project_policy":
            continue
        profiles = tuple(
            profile
            for profile in TEMPORAL_POLICY_PROFILES.values()
            if profile.node_type == axis.node_type
            and profile.policy_axis == axis.axis_id
        )
        if not profiles:
            raise ValueError(
                f"project-policy axis {axis.axis_id} has no legal candidate profiles"
            )
        for profile in profiles:
            missing = set(axis.parameters) - set(profile.controlled_parameters)
            if missing:
                raise ValueError(
                    f"policy candidate {profile.profile_id} does not control axis "
                    f"{axis.axis_id}: " + ", ".join(sorted(missing))
                )


validate_temporal_policy_profiles()


def temporal_policy_profile(profile_id: str) -> TemporalPolicyProfile:
    try:
        return TEMPORAL_POLICY_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown temporal policy profile: {profile_id}") from exc


def temporal_policy_for_reading_class(
    reading_class: str,
) -> TemporalPolicyProfile:
    try:
        profile_id = SEMANTIC_POLICY_BINDINGS[reading_class]
    except KeyError as exc:
        raise ValueError(
            f"no project policy is registered for reading class: {reading_class}"
        ) from exc
    return temporal_policy_profile(profile_id)


def temporal_policy_profiles_payload() -> dict[str, object]:
    return {
        profile_id: {
            "node_type": profile.node_type,
            "reading": profile.reading,
            "parameters": dict(profile.parameters),
            "derived_parameters": list(profile.derived_parameters),
            "controlled_parameters": list(profile.controlled_parameters),
            "policy_axis": profile.policy_axis,
            "user_label": profile.user_label,
            "user_explanation": profile.user_explanation,
            "selection_rule": (
                "treat as one legal, non-authoritative candidate after the reviewed "
                "reading fixes semantic topology and source-derived parameters; expand "
                "all matching candidates and require expert trace evidence to select"
            ),
        }
        for profile_id, profile in sorted(TEMPORAL_POLICY_PROFILES.items())
    }
