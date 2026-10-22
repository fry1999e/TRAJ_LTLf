"""Resolve policy-only semantic requests without overriding reviewed readings.

The LLM may identify that an executable IR needs a finite-trace policy, but it
does not get to invent that policy.  This module matches a reviewed reading and
the unresolved contract axes to a backend-validated, versioned project profile.
Unknown or genuinely semantic questions remain human review items.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from schemas import PlainReadingReview
from semantic_axes import semantic_axis_for_request
from temporal_policy_profiles import TEMPORAL_POLICY_PROFILES


@dataclass(frozen=True)
class SemanticPolicyDecision:
    """Auditable input to a second, constrained IR-materialization call."""

    decision_id: str
    axis_kind: str
    node_type: str
    requested_node_id: str
    evidence_quote: str
    unresolved_parameters: tuple[str, ...]
    profile_id: str
    parameters: Mapping[str, object]
    controlled_parameters: tuple[str, ...]
    selected_ambiguity_ids: tuple[str, ...]
    selected_readings: tuple[str, ...]
    decision_source: str
    mapping_rule_id: str
    user_label: str
    user_explanation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_semantic_policy_request(
    raw_result: object,
    *,
    plain_reading_review: PlainReadingReview | None,
) -> SemanticPolicyDecision | None:
    """Resolve only a singleton backend candidate; never choose among options.

    ``plain_reading_review`` remains in the signature for API compatibility and
    lineage checks at the caller.  A reviewed reading does not silently select
    a finite-trace policy.  Multi-option axes are expanded into CandidateIRSet.
    """

    del plain_reading_review
    options = legal_policy_options_for_request(raw_result)
    if len(options) != 1:
        return None
    seed = candidate_seed_policy_decision(raw_result, options[0])
    return SemanticPolicyDecision(
        **{
            **seed.to_dict(),
            "decision_id": seed.decision_id.replace("POLICY-SEED-", "POLICY-SINGLETON-"),
            "decision_source": "singleton_policy_candidate_catalog",
            "mapping_rule_id": "singleton-policy-candidate-v1",
        }
    )


def legal_policy_options_for_request(raw_result: object) -> tuple[dict[str, object], ...]:
    """Return backend-owned options suitable for a nontechnical review screen."""

    request = semantic_resolution_request_payload(raw_result)
    if request is None:
        return ()
    unresolved = request.get("unresolved_parameters")
    if not isinstance(unresolved, list) or not unresolved:
        return ()
    names = frozenset(str(item).strip() for item in unresolved if str(item).strip())
    axis = semantic_axis_for_request(
        node_type=request.get("node_type", ""),
        axis_kind=request.get("axis_kind", ""),
        unresolved_parameters=tuple(sorted(names)),
    )
    if axis is None:
        return ()
    options = []
    for profile in TEMPORAL_POLICY_PROFILES.values():
        if profile.node_type != axis.node_type:
            continue
        if profile.policy_axis != axis.axis_id:
            continue
        if not names.issubset(set(profile.controlled_parameters)):
            continue
        options.append({
            "axis_id": axis.axis_id,
            "node_type": axis.node_type,
            "profile_id": profile.profile_id,
            "label": profile.user_label,
            "explanation": profile.user_explanation,
            "parameters": dict(profile.parameters),
            "controlled_parameters": list(profile.controlled_parameters),
            "derived_parameters": list(profile.derived_parameters),
            "profile_version": profile.version,
        })
    return tuple(options)


def candidate_seed_policy_decision(
    raw_result: object,
    option: Mapping[str, object],
) -> SemanticPolicyDecision:
    """Build a non-authoritative materialization seed for a policy branch set.

    The seed exists only so the LLM can emit the shared, source-derived IR
    structure once.  It is never recorded as the selected semantic policy; the
    backend deterministically expands every legal option from that structure.
    """

    request = semantic_resolution_request_payload(raw_result)
    if request is None:
        raise ValueError("candidate policy seed requires a resolution request")
    profile_id = str(option.get("profile_id", "")).strip()
    profile = TEMPORAL_POLICY_PROFILES.get(profile_id)
    if profile is None:
        raise ValueError(f"unknown candidate policy profile: {profile_id}")
    axis_id = str(option.get("axis_id", profile.policy_axis)).strip()
    requested_node_id = str(request.get("node_id", "")).strip()
    unresolved = tuple(
        sorted(
            str(item).strip()
            for item in request.get("unresolved_parameters", ())
            if str(item).strip()
        )
    )
    return SemanticPolicyDecision(
        decision_id=(
            "POLICY-SEED-"
            + axis_id
            + "-"
            + requested_node_id
            + "-"
            + profile.profile_id
        ),
        axis_kind=axis_id,
        node_type=profile.node_type,
        requested_node_id=requested_node_id,
        evidence_quote=str(request.get("evidence_quote", "")).strip(),
        unresolved_parameters=unresolved,
        profile_id=profile.profile_id,
        parameters=dict(profile.parameters),
        controlled_parameters=profile.controlled_parameters,
        selected_ambiguity_ids=(),
        selected_readings=(),
        decision_source="candidate_seed_materialization_only",
        mapping_rule_id="policy-candidate-seed-v1",
        user_label=profile.user_label,
        user_explanation=profile.user_explanation,
    )


def semantic_resolution_request_payload(
    raw_result: object,
) -> dict[str, object] | None:
    """Return a detached request payload for diagnostics and review UI."""

    if not isinstance(raw_result, Mapping):
        return None
    direct = raw_result.get("semantic_resolution_request")
    if isinstance(direct, Mapping):
        return dict(direct)
    semantic = raw_result.get("semantic")
    if isinstance(semantic, Mapping):
        nested = semantic.get("semantic_resolution_request")
        if isinstance(nested, Mapping):
            return dict(nested)
    return None


def _selected_readings(
    review: PlainReadingReview,
) -> tuple[tuple[str, str], ...]:
    selected: list[tuple[str, str]] = []
    for ambiguity in review.ambiguities:
        if ambiguity.decision_status != "reviewer_selected" or not ambiguity.decision:
            continue
        interpretation = next(
            (
                option.plain_interpretation
                for option in ambiguity.options
                if option.option_id == ambiguity.decision
            ),
            ambiguity.decision,
        )
        selected.append((ambiguity.ambiguity_id, interpretation.strip()))
    return tuple(selected)


def _is_bounded_future_witness_reading(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    witness_markers = (
        "at least one",
        "at least once",
        "must be false in",
        "must hold in",
        "must occur in",
        "至少一次",
        "至少一个",
        "必须在",
    )
    future_markers = (
        "following",
        "next",
        "future",
        "后续",
        "接下来",
        "未来",
    )
    bounded_markers = ("step", "timestamp", "步", "时刻", "时间点")
    explicitly_full = (
        "complete window must exist",
        "exactly three future steps must exist",
        "不足完整窗口则不满足",
        "必须存在完整窗口",
    )
    explicitly_vacuous = (
        "vacuously",
        "automatically satisfied",
        "自动满足",
    )
    return (
        any(marker in normalized for marker in witness_markers)
        and any(marker in normalized for marker in future_markers)
        and any(marker in normalized for marker in bounded_markers)
        and not any(marker in normalized for marker in explicitly_full)
        and not any(marker in normalized for marker in explicitly_vacuous)
    )
