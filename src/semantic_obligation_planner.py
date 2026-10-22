"""Formula-free semantic obligation planning for finite-trace queries."""

from __future__ import annotations

import re

from schemas import SemanticObligationPlan, SemanticStructure
from ir_compiler import compile_review_trace_plan


def plan_semantic_obligations(
    semantic: SemanticStructure,
) -> SemanticObligationPlan:
    """
    Derive ambiguity axes and required query roles without compiling LTLf.

    Natural-language operator words are preserved as cues only. The returned
    object contains no executable formula and no truth labels.
    """

    if semantic.faithful_ir is not None:
        ir = semantic.faithful_ir
        trace_plan = compile_review_trace_plan(ir)
        root = ir.node_map[ir.root_node_id]
        return SemanticObligationPlan(
            pattern_family=root.node_type,
            participants=tuple(
                ("ap", ap.name) for ap in ir.atomic_propositions
            ),
            explicit_cues=(("ir_version", ir.ir_version),),
            ambiguity_axes=trace_plan.confusion_targets,
            required_query_roles=tuple(
                dict.fromkeys(
                    item.required_trace_role for item in trace_plan.obligations
                )
            ),
            trace_constraints=tuple(
                constraint
                for item in trace_plan.obligations
                for constraint in item.construction_constraints
            ),
            domain_constraints=trace_plan.coverage_requirements,
        )

    pattern = _normalized_pattern(semantic.pattern)
    participants = _participants(semantic)
    cues = tuple(
        (key, value)
        for key, value in (
            ("scope_keyword", semantic.scope_keyword.strip()),
            ("temporal_keyword", semantic.temporal_keyword.strip()),
        )
        if value
    )

    if pattern == "response":
        axes = [
            "same_step_vs_strict_future",
            "current_vs_global_scope",
            "final_state_behavior",
        ]
        roles = [
            "delayed_witness",
            "missing_response",
            "late_trigger_witness",
            "late_trigger_missing_response",
            "same_step_boundary",
            "final_state_boundary",
        ]
    elif pattern in {"persistence", "global_state", "invariance"}:
        axes = [
            "current_vs_global_scope",
            "middle_state_violation",
            "final_state_behavior",
        ]
        roles = [
            "global_witness",
            "first_state_violation",
            "middle_state_violation",
            "final_state_violation",
            "minimal_length_boundary",
            "late_state_boundary",
        ]
    elif pattern in {"precedence", "before"}:
        axes = [
            "strict_before_vs_same_step",
            "first_occurrence_vs_any_occurrence",
            "final_state_behavior",
        ]
        roles = [
            "preceded_occurrence",
            "missing_predecessor",
            "same_step_boundary",
            "late_predecessor_boundary",
            "repeated_occurrence",
            "minimal_length_boundary",
        ]
    elif pattern == "until":
        axes = [
            "strong_vs_weak_until",
            "same_step_witness",
            "left_condition_scope",
            "final_state_behavior",
        ]
        roles = [
            "delayed_until_witness",
            "immediate_until_witness",
            "left_break_before_witness",
            "missing_until_witness",
            "final_state_witness",
            "minimal_length_boundary",
        ]
    elif pattern in {"mutual_exclusion", "mutual_exclusion_with_coverage"}:
        axes = [
            "cooccurrence_policy",
            "coverage_requirement",
            "current_vs_global_scope",
        ]
        roles = [
            "left_only",
            "right_only",
            "cooccurrence_boundary",
            "neither_boundary",
            "late_cooccurrence",
            "alternating_states",
        ]
    else:
        axes = [
            "current_vs_global_scope",
            "same_step_vs_strict_future",
            "final_state_behavior",
        ]
        roles = [
            "minimal_length_boundary",
            "single_event_candidate",
            "delayed_event_candidate",
            "missing_event_candidate",
            "late_event_candidate",
            "cooccurrence_boundary",
        ]

    return SemanticObligationPlan(
        pattern_family=pattern,
        participants=participants,
        explicit_cues=cues,
        ambiguity_axes=tuple(axes),
        required_query_roles=tuple(roles),
        trace_constraints=(
            "use only reviewed atomic propositions",
            "finite trace must contain at least one time step",
            "query shapes must be structurally distinct",
            "query role must remain separate from adjudicated truth",
        ),
        domain_constraints=tuple(
            item
            for item in (
                semantic.domain_warning.strip(),
                semantic.finite_rewrite.strip() if semantic.reference_only else "",
            )
            if item
        ),
    )


def _normalized_pattern(pattern: str) -> str:
    value = re.sub(r"[\s-]+", "_", pattern.strip().lower())
    aliases = {
        "globally": "global_state",
        "always": "global_state",
        "invariant": "invariance",
        "precedence_before": "precedence",
        "mutex": "mutual_exclusion",
    }
    return aliases.get(value, value or "unknown")


def _participants(
    semantic: SemanticStructure,
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for role, value in (
        ("trigger", semantic.trigger),
        ("response", semantic.response),
    ):
        if value and value not in {"unknown", "unknown_ap"}:
            pairs.append((role, value))
    for key, value in semantic.components:
        if value and (key.endswith("_condition") or key.endswith("_witness")):
            pairs.append((key, value))
    return tuple(dict.fromkeys(pairs))
