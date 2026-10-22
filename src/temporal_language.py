"""Shared natural-language grounding for existing temporal operators."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from contract_versions import TEMPORAL_LANGUAGE_CONTRACT_VERSION


@dataclass(frozen=True)
class TemporalLanguageEntry:
    reading: str
    operator: str
    executable_kind: str
    aliases: tuple[str, ...]
    scope: str
    witness_position: str
    exact_offset: int | None = None
    contract_version: str = TEMPORAL_LANGUAGE_CONTRACT_VERSION


@dataclass(frozen=True)
class TemporalEvidenceDecision:
    status: str
    reading: str
    operator: str
    matched_alias: str
    evidence_text: str
    witness_position: str
    exact_offset: int | None
    contract_version: str = TEMPORAL_LANGUAGE_CONTRACT_VERSION


TEMPORAL_LANGUAGE_REGISTRY: Mapping[str, TemporalLanguageEntry] = {
    "identity": TemporalLanguageEntry(
        "identity",
        "epsilon",
        "identity",
        ("epsilon", "identity", "no operator", "current position"),
        "current",
        "current",
        0,
    ),
    "immediate": TemporalLanguageEntry(
        "immediate",
        "epsilon",
        "identity",
        ("immediate", "immediately", "at once", "now"),
        "current",
        "current",
        0,
    ),
    "global_constraint": TemporalLanguageEntry(
        "global_constraint",
        "G",
        "global_constraint",
        ("G", "always", "globally"),
        "suffix",
        "all_current_and_future",
    ),
    "persist_to_end": TemporalLanguageEntry(
        "persist_to_end",
        "G",
        "persist_to_end",
        ("persist to end", "holds forever", "thereafter"),
        "suffix",
        "all_current_and_future",
    ),
    "eventually": TemporalLanguageEntry(
        "eventually",
        "F",
        "eventually",
        ("F", "eventually", "finally", "sometime", "ultimately"),
        "suffix",
        "current_or_future",
        0,
    ),
    "eventually_strict": TemporalLanguageEntry(
        "eventually_strict",
        "XF",
        "eventually_strict",
        (
            "XF",
            "next F",
            "next eventually",
            "next finally",
            "strict future",
            "strictly later",
            "later but not now",
        ),
        "strict_suffix",
        "strict_future",
        1,
    ),
    "next_step": TemporalLanguageEntry(
        "next_step",
        "X",
        "next_step",
        ("X", "next", "next time", "next step", "following step"),
        "exact_offset",
        "exact_offset",
        1,
    ),
}


def temporal_language_prompt_view() -> dict[str, object]:
    return {
        reading: {
            "operator": entry.operator,
            "executable_kind": entry.executable_kind,
            "aliases": list(entry.aliases),
            # Witness ranges describe timing, not the IR's scope metadata.
            # In particular, strict_suffix is implemented by the node kind;
            # it is not a legal value for FaithfulIRNode.scope.
            "scope": "all_positions" if reading == "global_constraint" else "current_position",
            "witness_range": entry.scope,
            "witness_position": entry.witness_position,
            "exact_offset": entry.exact_offset,
        }
        for reading, entry in TEMPORAL_LANGUAGE_REGISTRY.items()
    }


def operator_atoms_for_alias(value: str) -> tuple[str, ...] | None:
    normalized = _normalize_alias(value)
    if not normalized:
        return ()
    for entry in _entries_longest_alias_first():
        if any(_normalize_alias(alias) == normalized for alias in entry.aliases):
            if entry.operator == "epsilon":
                return ()
            if entry.operator == "XF":
                return ("X", "F")
            if entry.operator in {"X", "F", "G"}:
                return (entry.operator,)
    return None


def decide_temporal_evidence(evidence_text: str) -> TemporalEvidenceDecision:
    """Resolve explicit timing language, preserving true ambiguity."""

    return decide_temporal_evidence_for_readings(
        evidence_text,
        tuple(TEMPORAL_LANGUAGE_REGISTRY),
    )


def decide_temporal_evidence_for_readings(
    evidence_text: str,
    readings: tuple[str, ...],
) -> TemporalEvidenceDecision:
    text = evidence_text.strip()
    if not text:
        return TemporalEvidenceDecision(
            "none", "", "", "", "", "", None
        )
    matches: list[tuple[int, int, TemporalLanguageEntry, str]] = []
    selected_entries = tuple(
        entry
        for entry in _entries_longest_alias_first()
        if entry.reading in readings
    )
    for entry in selected_entries:
        for alias in entry.aliases:
            match = _alias_search(text, alias)
            if match is not None:
                matches.append((match[0], match[1], entry, alias))
    if not matches:
        return TemporalEvidenceDecision(
            "none", "", "", "", text, "", None
        )
    matches.sort(key=lambda value: (value[0], -(value[1] - value[0])))
    # A strict composite ("next finally") dominates its component aliases.
    first = matches[0]
    overlapping = [
        match
        for match in matches
        if not (match[1] <= first[0] or first[1] <= match[0])
    ]
    readings = {match[2].reading for match in overlapping}
    strict = next(
        (
            match
            for match in overlapping
            if match[2].reading == "eventually_strict"
        ),
        None,
    )
    if strict is not None:
        first = strict
        readings = {"eventually_strict"}
    if len(readings) > 1:
        return TemporalEvidenceDecision(
            "unresolved",
            "",
            "",
            "",
            text,
            "multiple_explicit_timings",
            None,
        )
    entry = first[2]
    return TemporalEvidenceDecision(
        "fixed",
        entry.reading,
        entry.operator,
        first[3],
        text,
        entry.witness_position,
        entry.exact_offset,
    )


def temporal_readings_in_text(text: str) -> frozenset[str]:
    readings = {
        entry.reading
        for entry in TEMPORAL_LANGUAGE_REGISTRY.values()
        if any(_alias_search(text, alias) is not None for alias in entry.aliases)
    }
    if "eventually_strict" in readings:
        readings.discard("eventually")
        readings.discard("next_step")
    return frozenset(readings)


def temporal_grounding_errors(ir: object) -> tuple[str, ...]:
    """Reject a unique source timing that conflicts with the proposed node."""

    errors: list[str] = []
    for node in getattr(ir, "nodes", ()):
        errors.extend(temporal_node_grounding_errors(node, ir=ir))
    return tuple(errors)


def temporal_node_grounding_errors(
    node: object,
    *,
    ir: object | None = None,
) -> tuple[str, ...]:
    node_type = str(getattr(node, "node_type", ""))
    if node_type not in TEMPORAL_LANGUAGE_REGISTRY:
        return ()
    # A complete backend-generated policy axis may select a timing that the
    # source marker alone leaves open.  This exception is narrowly bound to
    # the constructive candidate extractor and a singleton typed selection;
    # arbitrary model IR cannot use it to bypass lexical grounding.
    if (
        ir is not None
        and getattr(getattr(ir, "provenance", None), "extractor", "")
        == "RCG-source-candidate-v1"
        and getattr(node, "semantic_role", "")
        == "policy_axis_selection:eventuality_policy"
        and tuple(getattr(node, "allowed_semantic_readings", ())) == (node_type,)
    ):
        return ()
    evidence = str(getattr(node, "evidence_text", ""))
    family = (
        ("eventually", "eventually_strict", "next_step")
        if node_type in {"eventually", "eventually_strict", "next_step"}
        else ("global_constraint", "persist_to_end")
        if node_type in {"global_constraint", "persist_to_end"}
        else ("identity", "immediate")
    )
    decision = decide_temporal_evidence_for_readings(evidence, family)
    if decision.status != "fixed":
        return ()
    proposed = TEMPORAL_LANGUAGE_REGISTRY[node_type]
    if decision.operator == proposed.operator:
        return ()
    # A strict-future formula may be represented canonically as the
    # composition X(F(phi)).  Reference/formal imports often give both unary
    # nodes the same enclosing evidence span, so neither component should be
    # rejected in isolation when the connected pair exactly realizes XF.
    if (
        decision.operator == "XF"
        and ir is not None
        and _is_connected_strict_future_component(ir, node)
    ):
        return ()
    allowed = tuple(
        str(value)
        for value in getattr(node, "allowed_semantic_readings", ())
    )
    if (
        str(getattr(node, "temporal_resolution", "fixed"))
        == "trace_resolvable"
        and decision.reading in allowed
    ):
        return ()
    return (
        f"node {getattr(node, 'node_id', '')} timing {node_type!r} conflicts "
        f"with source alias {decision.matched_alias!r} -> "
        f"{decision.reading!r}",
    )


def _is_connected_strict_future_component(
    ir: object,
    node: object,
) -> bool:
    node_id = str(getattr(node, "node_id", ""))
    node_type = str(getattr(node, "node_type", ""))
    node_map = {
        str(getattr(item, "node_id", "")): item
        for item in getattr(ir, "nodes", ())
    }
    if node_type == "next_step":
        children = tuple(getattr(node, "children", ()))
        return (
            len(children) == 1
            and str(getattr(node_map.get(children[0]), "node_type", ""))
            == "eventually"
        )
    if node_type == "eventually":
        return any(
            str(getattr(parent, "node_type", "")) == "next_step"
            and node_id in tuple(getattr(parent, "children", ()))
            for parent in node_map.values()
        )
    return False


def _entries_longest_alias_first() -> tuple[TemporalLanguageEntry, ...]:
    return tuple(
        sorted(
            TEMPORAL_LANGUAGE_REGISTRY.values(),
            key=lambda entry: max(len(alias) for alias in entry.aliases),
            reverse=True,
        )
    )


def _normalize_alias(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value).upper()


def _alias_search(text: str, alias: str) -> tuple[int, int] | None:
    if alias in {"F", "G", "X", "XF"}:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])"
    else:
        words = [re.escape(value) for value in re.split(r"[\s_-]+", alias)]
        pattern = r"(?<![A-Za-z0-9_])" + r"[\s_-]+".join(words) + r"(?![A-Za-z0-9_])"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return (match.start(), match.end()) if match else None


__all__ = [
    "TEMPORAL_LANGUAGE_REGISTRY",
    "TemporalEvidenceDecision",
    "TemporalLanguageEntry",
    "decide_temporal_evidence",
    "decide_temporal_evidence_for_readings",
    "operator_atoms_for_alias",
    "temporal_grounding_errors",
    "temporal_node_grounding_errors",
    "temporal_language_prompt_view",
    "temporal_readings_in_text",
]
