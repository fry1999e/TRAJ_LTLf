"""
有限轨迹生成 Agent。

职责：
    1. 基于自然语言需求、已通过评审的语义结构和 RAG 知识生成有限轨迹。
    2. 同时生成程序可检查的 trace 和零基础可读的 beginner_view。
    3. 在人工评审不通过时，结合评审意见重新生成轨迹。
"""

from __future__ import annotations
from execution_observer import observed_stage

import json
from dataclasses import replace
from itertools import product
import re

from agents.base_agent import BaseAgent, JSONLLMClient
from agents.llm_parsing import (
    ALLOWED_TRACE_KINDS,
    coerce_bool,
    coerce_list_field,
    require_ap_name,
    require_list,
    require_mapping,
    require_string,
    optional_string,
)
from agents.prompt_context import (
    FULL_PROCESS_SOURCE,
    KB_LTLF_SEMANTICS,
    KB_TRACE_EXAMPLES,
    build_stage_prompt_context,
    structured_system_prompt,
)
from data_examples import (
    DEFAULT_RESPONSE_TRACE_COUNT,
    LabeledTrace,
    build_response_traces,
)
from guided_learning import GuidedLearningContext, guided_prompt_rule
from ltlf_checker import check_trace
from pattern_registry import get_pattern_handler
from pattern_registry import collect_formula_atoms, parse_learned_formula
from rag.example_memory import (
    GUIDED_CONSISTENCY_SOURCE,
    GUIDED_FULL_PROCESS_SOURCE,
    GUIDED_TRACE_SOURCE,
    TRACE_ADEQUACY_SOURCE,
    TRACE_SOURCE,
)
from schemas import SemanticStructure, TraceCoverageReport
from schemas import (
    TRACE_PURPOSE_OPERATOR_DISCRIMINATION,
    TRACE_PURPOSE_SEMANTIC_VALIDATION,
    TraceObligationPlan,
    TraceQuery,
)
from faithful_ir import (
    PERIODICITY_REQUIRED_PARAMETERS,
    assert_ir_ready_for_trace_generation,
    precedence_roles_from_semantic,
)
from ir_compiler import (
    bind_traces_to_ir_plan,
    compile_executable_trace_plan,
    evaluate_ir_coverage,
)
from ir_trace_semantics import IRTraceSemanticsError, evaluate_ir_trace
from ir_witness_solver import (
    HorizonPolicy,
    INVALID_OBLIGATION,
    IRWitnessSynthesisError,
    NO_ADDITIONAL_DISTINCT_WITNESS,
    UNKNOWN_RESOURCE_LIMIT,
    UNSAT_IN_DECLARED_DOMAIN,
    UNSUPPORTED_SEMANTICS,
    WITNESS_FOUND,
    WitnessRequest,
    WitnessResult,
    synthesize_ir_witness,
    trace_key,
)
from ir_relation_feasibility import require_contingent_relation
from semantic_discriminators import (
    DISCRIMINATOR_RESOURCE_LIMIT,
    DISCRIMINATOR_WITNESS_FOUND,
    discriminator_values,
    synthesize_discriminator_trace,
)
from semantic_node_contracts import structural_horizon
from semantic_obligation_planner import plan_semantic_obligations
from semantic_promotion import (
    AuthoritativeTraceSession,
    validate_authoritative_trace_session,
)
from temporal_domain import finite_requirement_for_pipeline
from trace_formatter import format_trace_for_beginner, trace_to_serializable
from trace_evidence_policy import is_boundary_trace
from trace_query_generation import instantiate_trace_queries
from trace_obligations import (
    analyze_obligations,
    build_obligation_trace_set,
    covered_confusions,
    covered_roles,
    enrich_trace_semantics,
    validate_trace_labels,
)


MIN_DISTINCT_BOUNDARY_TRACE_COUNT = 2
MIN_REQUIRED_TRACE_COUNT = 6
MIN_REQUIRED_KIND_TRACE_COUNT = 2


def _tag_faithful_ir_traces(
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
) -> list[LabeledTrace]:
    """Bind generated evidence to the IR-derived obligations it exercises."""

    if semantic.faithful_ir is None:
        return traces
    plan = compile_executable_trace_plan(semantic.faithful_ir)
    tagged: list[LabeledTrace] = []
    for index, trace in enumerate(traces, start=1):
        if trace.kind == "positive":
            relations = {"satisfy"}
        elif trace.kind == "negative":
            relations = {"violate"}
        else:
            relations = {"distinguish"}
        covered = tuple(
            obligation
            for obligation in plan.obligations
            if obligation.expected_relation in relations
            and len(trace.trace) >= obligation.minimum_horizon
        )
        confusions = (
            plan.confusion_targets if trace.kind == "boundary" else ()
        )
        semantics = replace(
            trace.semantics,
            roles=tuple(
                dict.fromkeys(
                    (*trace.semantics.roles, *(item.required_trace_role for item in covered))
                )
            ),
            obligations=tuple(
                dict.fromkeys(
                    (*trace.semantics.obligations, *(item.obligation_id for item in covered))
                )
            ),
            confusion_targets=tuple(
                dict.fromkeys((*trace.semantics.confusion_targets, *confusions))
            ),
        )
        tagged.append(
            replace(
                trace,
                semantics=semantics,
                trace_id=trace.trace_id or trace.name or f"trace-{index}",
                covered_ir_node_ids=tuple(
                    dict.fromkeys(
                        (*trace.covered_ir_node_ids, *(node_id for item in covered for node_id in item.ir_node_ids))
                    )
                ),
                obligation_ids=tuple(
                    dict.fromkeys(
                        (*trace.obligation_ids, *(item.obligation_id for item in covered))
                    )
                ),
                trace_role=(
                    trace.trace_role
                    or (covered[0].required_trace_role if covered else "")
                    or trace.role
                    or trace.kind
                ),
                boundary_role=(
                    trace.boundary_role
                    or (trace.target_ambiguity if trace.kind == "boundary" else "")
                ),
                generation_source=trace.generation_source or "ir_trace_agent",
                ir_version=plan.ir_version,
            )
        )
    return tagged


def _instantiate_ir_trace_queries(
    semantic: SemanticStructure,
    plan: TraceObligationPlan,
    *,
    target_trace_length: int | None,
    target_trace_count: int | None,
    data_split: str,
    label_source: str,
) -> list[TraceQuery]:
    """Create formula-free query shapes directly from compiled obligations."""

    aps = tuple(semantic.faithful_ir.ap_names) if semantic.faithful_ir else semantic.aps
    states = [set(), *({ap} for ap in aps)]
    if len(aps) > 1:
        states.append(set(aps))
    required_count = max(
        MIN_REQUIRED_TRACE_COUNT,
        len(plan.obligations),
        target_trace_count or 0,
    )
    obligations = list(plan.obligations)
    if not obligations:
        return []
    queries: list[TraceQuery] = []
    seen_by_purpose: dict[str, set[tuple[tuple[str, ...], ...]]] = {
        TRACE_PURPOSE_SEMANTIC_VALIDATION: set(),
        TRACE_PURPOSE_OPERATOR_DISCRIMINATION: set(),
    }
    shape_index = 0
    max_attempts = max(required_count * 20, 20)
    attempts = 0
    while len(queries) < required_count and attempts < max_attempts:
        attempts += 1
        obligation = obligations[(attempts - 1) % len(obligations)]
        purpose = (
            TRACE_PURPOSE_OPERATOR_DISCRIMINATION
            if obligation.evidence_mode == "operator_resolution"
            or obligation.required_trace_role == "operator_resolution"
            else TRACE_PURPOSE_SEMANTIC_VALIDATION
        )
        seen = seen_by_purpose[purpose]
        relation_discriminator = bool(
            dict(obligation.semantic_parameters).get("confusion_id", "")
        ) or _is_relation_discriminator_role(obligation.required_trace_role)
        if (
            target_trace_length is not None
            and target_trace_length < obligation.minimum_horizon
        ):
            result = _invalid_horizon_result(
                obligation=obligation,
                target_trace_length=target_trace_length,
            )
            raise IRWitnessSynthesisError(
                "IR_OBLIGATION_INVALID",
                "Exact trace length cannot satisfy the IR obligation horizon: "
                f"{obligation.obligation_id} requires at least "
                f"{obligation.minimum_horizon}, requested {target_trace_length}.",
                result=result,
                obligation_id=obligation.obligation_id,
                suggested_rerun_stage="trace",
            )
        horizon = target_trace_length or obligation.minimum_horizon
        if target_trace_length is None and not relation_discriminator:
            while len(states) ** horizon < required_count:
                horizon += 1
        trace = _nl_probe_trace_for_obligation(
            semantic=semantic,
            obligation=obligation,
            horizon=horizon,
            variant=shape_index,
        )
        relation_obligation = obligation.expected_relation in {"satisfy", "violate"}
        if relation_obligation:
            trace = _ensure_formula_free_relation(
                semantic=semantic,
                obligation=obligation,
                trace=trace,
                horizon=horizon,
                exact_horizon=target_trace_length is not None,
                seen=seen,
                variant=shape_index,
            )
            shape_index += 1
            if trace is None:
                continue
        elif target_trace_length is not None:
            trace = _fit_trace_to_exact_horizon(trace, target_trace_length)
            confusion_id = str(
                dict(obligation.semantic_parameters).get("confusion_id", "")
            )
            if confusion_id:
                ir = semantic.faithful_ir
                values = (
                    discriminator_values(
                        ir,
                        obligation.ir_node_ids[0],
                        confusion_id,
                        trace,
                    )
                    if ir is not None and obligation.ir_node_ids
                    else None
                )
                if values is None or values[0] == values[1]:
                    result = WitnessResult(
                        INVALID_OBLIGATION,
                        obligation.ir_node_ids[0] if obligation.ir_node_ids else "",
                        False,
                        attempted_lengths=(target_trace_length,),
                        proof_scope=f"exact_length={target_trace_length}",
                        stop_reason=(
                            "exact trace length cannot preserve the requested "
                            "semantic discriminator"
                        ),
                        diagnostics={"confusion_id": confusion_id},
                    )
                    raise IRWitnessSynthesisError(
                        "IR_OBLIGATION_INVALID",
                        f"Exact length {target_trace_length} cannot distinguish "
                        f"{confusion_id} for {obligation.obligation_id}.",
                        result=result,
                        obligation_id=obligation.obligation_id,
                        suggested_rerun_stage="trace",
                    )
        if not relation_obligation and not any(trace) and not relation_discriminator:
            # Mixed-radix enumeration supplies a distinct fallback shape. It
            # encodes no label and uses only APs declared by the IR.
            value = shape_index
            trace = []
            for _ in range(horizon):
                trace.append(set(states[value % len(states)]))
                value //= len(states)
        if not relation_obligation:
            shape_index += 1
        key = trace_key(trace)
        if key in seen and not relation_discriminator and not relation_obligation:
            fallback = _next_distinct_formula_free_trace(
                states=states,
                horizon=horizon,
                seen=seen,
                start_index=shape_index,
            )
            if fallback is None:
                continue
            trace, shape_index = fallback
            key = trace_key(trace)
        seen.add(key)
        obligation_confusion = str(
            dict(obligation.semantic_parameters).get("confusion_id", "")
        )
        target_ambiguity = (
            obligation_confusion
            or obligation.boundary_case
            or obligation.expected_relation
        )
        queries.append(
            TraceQuery(
                query_id=(
                    f"OPQ-{len(queries) + 1:03d}"
                    if purpose == TRACE_PURPOSE_OPERATOR_DISCRIMINATION
                    else f"QUERY-IR-{len(queries) + 1:03d}"
                ),
                role=obligation.required_trace_role,
                target_ambiguity=target_ambiguity,
                target_ambiguity_axis_ids=(
                    obligation.ambiguity_axis_ids
                    or (
                        (target_ambiguity,)
                        if target_ambiguity
                        and target_ambiguity
                        not in {"satisfy", "violate", "distinguish"}
                        else ()
                    )
                ),
                constraints=(
                    *obligation.construction_constraints,
                    f"minimum_horizon={obligation.minimum_horizon}",
                    f"obligation_id={obligation.obligation_id}",
                    f"trace_role={obligation.required_trace_role}",
                    f"target_ambiguity={target_ambiguity}",
                    "label_source_must_be_NL_temporal_simulation",
                ),
                trace=trace,
                reason=(
                    f"Formula-free query for IR obligation {obligation.obligation_id}."
                ),
                beginner_view=format_trace_for_beginner(trace),
                data_split=data_split,
                label_source=label_source,
                ir_node_ids=obligation.ir_node_ids,
                obligation_ids=(obligation.obligation_id,),
                ir_version=plan.ir_version,
                semantic_parameter_paths=obligation.semantic_parameter_paths,
                purpose=purpose,
                parent_hole_plan_version=(
                    plan.plan_version
                    if purpose == TRACE_PURPOSE_OPERATOR_DISCRIMINATION
                    else ""
                ),
                generation_source="nl_anchored_ir_obligation",
            )
        )
    return queries


def _ensure_formula_free_relation(
    *,
    semantic: SemanticStructure,
    obligation,
    trace: list[set[str]],
    horizon: int,
    exact_horizon: bool,
    seen: set[tuple[tuple[str, ...], ...]],
    variant: int,
) -> list[set[str]] | None:
    """Make satisfy/violate query shapes executable under the reviewed IR.

    This is a construction check, not a label source.  It evaluates only the
    formula-free IR node named by the obligation and never exposes the Boolean
    result to the later adjudicator.
    """

    ir = semantic.faithful_ir
    if ir is None or not obligation.ir_node_ids:
        return trace
    node_id = obligation.ir_node_ids[0]
    desired = obligation.expected_relation == "satisfy"
    proposed_key = trace_key(trace)
    try:
        if (
            len(trace) == horizon
            and proposed_key not in seen
            and evaluate_ir_trace(ir, trace, node_id=node_id) == desired
        ):
            return trace
    except IRTraceSemanticsError as exc:
        result = _unsupported_witness_result(
            node_id=node_id,
            desired=desired,
            horizon=horizon,
            error=exc,
        )
        raise IRWitnessSynthesisError(
            "IR_WITNESS_UNSUPPORTED_NODE",
            f"IR node {node_id} cannot be executed without guessing semantics: {exc}",
            result=result,
            obligation_id=obligation.obligation_id,
            suggested_rerun_stage="semantic",
        ) from exc

    horizon_policy = HorizonPolicy(
        minimum_length=horizon,
        maximum_length=horizon if exact_horizon else horizon + 1,
        exact_length=horizon if exact_horizon else None,
    )
    result = synthesize_ir_witness(
        WitnessRequest(
            ir=ir,
            node_id=node_id,
            desired_value=desired,
            horizon_policy=horizon_policy,
            blocked_traces=tuple(seen),
            purpose="distinct_query_witness" if seen else "primary_witness",
            branch_variant=variant,
        )
    )
    result = _with_witness_context(
        result,
        ir=ir,
        obligation=obligation,
        horizon=horizon,
        exact_horizon=exact_horizon,
    )
    if result.status == WITNESS_FOUND:
        return result.trace()

    if result.status == NO_ADDITIONAL_DISTINCT_WITNESS:
        # Diversity is a presentation concern. Retry the semantic obligation
        # without global blocking and permit reuse when the relation has only
        # one finite witness in the declared domain.
        unblocked = synthesize_ir_witness(
            WitnessRequest(
                ir=ir,
                node_id=node_id,
                desired_value=desired,
                horizon_policy=horizon_policy,
                purpose="primary_witness_reuse",
                branch_variant=variant,
            )
        )
        unblocked = _with_witness_context(
            unblocked,
            ir=ir,
            obligation=obligation,
            horizon=horizon,
            exact_horizon=exact_horizon,
        )
        if unblocked.status == WITNESS_FOUND:
            return unblocked.trace()
        result = unblocked

    if obligation.evidence_mode == "diagnostic_only" and result.status in {
        UNSAT_IN_DECLARED_DOMAIN,
        UNKNOWN_RESOURCE_LIMIT,
    }:
        return None

    feasibility_mismatch = bool(
        obligation.feasibility_report_hash
        and result.status == UNSAT_IN_DECLARED_DOMAIN
    )
    code = (
        "IR_FEASIBILITY_SYNTHESIS_MISMATCH"
        if feasibility_mismatch
        else {
        UNSAT_IN_DECLARED_DOMAIN: "IR_WITNESS_UNSAT_IN_DOMAIN",
        UNKNOWN_RESOURCE_LIMIT: "IR_WITNESS_RESOURCE_LIMIT",
        UNSUPPORTED_SEMANTICS: "IR_WITNESS_UNSUPPORTED_NODE",
        INVALID_OBLIGATION: "IR_WITNESS_VALIDATION_MISMATCH",
        }.get(result.status, "IR_WITNESS_SYNTHESIS_FAILED")
    )
    rerun_stage = (
        "trace"
        if feasibility_mismatch
        else "semantic"
        if result.status in {UNSAT_IN_DECLARED_DOMAIN, UNSUPPORTED_SEMANTICS}
        else "trace"
    )
    raise IRWitnessSynthesisError(
        code,
        "Unable to synthesize a verified formula-free IR witness for "
        f"{obligation.obligation_id}: {result.stop_reason}.",
        result=result,
        obligation_id=obligation.obligation_id,
        suggested_rerun_stage=rerun_stage,
    )


def _fit_trace_to_exact_horizon(
    trace: list[set[str]], exact_horizon: int
) -> list[set[str]]:
    fitted = [set(step) for step in trace[:exact_horizon]]
    while len(fitted) < exact_horizon:
        fitted.append(set())
    return fitted


def _invalid_horizon_result(*, obligation, target_trace_length: int):
    return WitnessResult(
        INVALID_OBLIGATION,
        obligation.ir_node_ids[0] if obligation.ir_node_ids else "",
        obligation.expected_relation == "satisfy",
        attempted_lengths=(target_trace_length,),
        proof_scope=f"exact_length={target_trace_length}",
        stop_reason="exact length is below the obligation minimum horizon",
        diagnostics={
            "minimum_horizon": obligation.minimum_horizon,
            "target_trace_length": target_trace_length,
        },
    )


def _unsupported_witness_result(
    *, node_id: str, desired: bool, horizon: int, error: BaseException
):
    return WitnessResult(
        UNSUPPORTED_SEMANTICS,
        node_id,
        desired,
        attempted_lengths=(horizon,),
        proof_scope=f"length={horizon}",
        stop_reason="independent IR evaluator rejected the proposed witness",
        diagnostics={"error_type": type(error).__name__, "error": str(error)},
    )


def _with_witness_context(
    result: WitnessResult,
    *,
    ir,
    obligation,
    horizon: int,
    exact_horizon: bool,
) -> WitnessResult:
    node_id = obligation.ir_node_ids[0] if obligation.ir_node_ids else ""
    node = ir.node_map.get(node_id)
    return replace(
        result,
        diagnostics={
            **dict(result.diagnostics),
            "ir_version": ir.ir_version,
            "node_type": node.node_type if node is not None else "",
            "node_parameters": dict(node.parameters) if node is not None else {},
            "node_children": list(node.children) if node is not None else [],
            "expected_relation": obligation.expected_relation,
            "evidence_mode": obligation.evidence_mode,
            "minimum_horizon": obligation.minimum_horizon,
            "requested_horizon": horizon,
            "exact_horizon": exact_horizon,
            "ap_count": len(ir.ap_names),
            "ap_context_fingerprint": ir.ap_context_fingerprint,
            "feasibility_report_hash": obligation.feasibility_report_hash,
        },
    )


def _is_relation_discriminator_role(role: str) -> bool:
    return role in {
        "immediate_until_witness",
        "delayed_until_witness",
        "left_break_before_witness",
        "missing_until_witness",
        "left_at_witness_boundary",
        "final_state_witness",
        "empty_prefix_gate",
        "missing_predecessor",
        "predecessor_only",
        "preceded_occurrence",
        "late_predecessor_boundary",
        "same_step_boundary",
        "repeated_occurrence",
        "repeated_precedence_cycle",
        "period_phase_boundary",
        "period_spacing_boundary",
        "period_off_phase_boundary",
        "period_terminal_boundary",
    }


def _nl_probe_trace_for_obligation(
    *,
    semantic: SemanticStructure,
    obligation,
    horizon: int,
    variant: int,
) -> list[set[str]]:
    confusion_id = str(
        dict(obligation.semantic_parameters).get("confusion_id", "")
    )
    if confusion_id:
        ir = semantic.faithful_ir
        if ir is None or not obligation.ir_node_ids:
            result = WitnessResult(
                INVALID_OBLIGATION,
                obligation.ir_node_ids[0] if obligation.ir_node_ids else "",
                False,
                attempted_lengths=(),
                proof_scope="no_search",
                stop_reason="discriminator obligation has no executable IR target",
                diagnostics={
                    "requirement_id": getattr(ir, "requirement_id", ""),
                    "recorded_minimum_horizon": obligation.minimum_horizon,
                    "suggested_rerun_stage": "semantic",
                },
            )
            raise IRWitnessSynthesisError(
                "IR_DISCRIMINATOR_INVALID_TARGET",
                f"Discriminator obligation {obligation.obligation_id} has no IR target.",
                result=result,
                obligation_id=obligation.obligation_id,
                suggested_rerun_stage="semantic",
            )
        node_id = obligation.ir_node_ids[0]
        node = ir.node_map.get(node_id)
        synthesis = synthesize_discriminator_trace(
            ir,
            node_id,
            confusion_id,
            minimum_horizon=horizon,
            exact_horizon=horizon,
            variant=variant,
        )
        trace = synthesis.trace()
        if synthesis.status != DISCRIMINATOR_WITNESS_FOUND or trace is None:
            parameters = dict(obligation.semantic_parameters)
            solver_status = (
                UNKNOWN_RESOURCE_LIMIT
                if synthesis.status == DISCRIMINATOR_RESOURCE_LIMIT
                else UNSAT_IN_DECLARED_DOMAIN
            )
            result = WitnessResult(
                solver_status,
                node_id,
                False,
                attempted_lengths=synthesis.attempted_lengths,
                proof_scope=f"exact_length={horizon}",
                stop_reason=synthesis.stop_reason,
                diagnostics={
                    "requirement_id": ir.requirement_id,
                    "ir_version": ir.ir_version,
                    "node_id": node_id,
                    "node_type": node.node_type if node is not None else "",
                    "confusion_id": confusion_id,
                    "recorded_minimum_horizon": obligation.minimum_horizon,
                    "root_structural_horizon": structural_horizon(ir),
                    "checked_candidate_count": synthesis.checked_candidate_count,
                    "preflight_witness_length": parameters.get(
                        "preflight_witness_length",
                        obligation.minimum_horizon,
                    ),
                    "suggested_rerun_stage": "trace",
                },
            )
            raise IRWitnessSynthesisError(
                (
                    "IR_DISCRIMINATOR_RESOURCE_LIMIT"
                    if synthesis.status == DISCRIMINATOR_RESOURCE_LIMIT
                    else "IR_DISCRIMINATOR_UNSAT_IN_DOMAIN"
                ),
                "Unable to instantiate semantic discriminator "
                f"{confusion_id} for {obligation.obligation_id}: "
                f"{synthesis.stop_reason}.",
                result=result,
                obligation_id=obligation.obligation_id,
                suggested_rerun_stage="trace",
            )
        return trace
    relation_trace = _relation_probe_trace(
        semantic=semantic,
        obligation=obligation,
        horizon=horizon,
        variant=variant,
    )
    if relation_trace is not None:
        return relation_trace
    aps = tuple(semantic.faithful_ir.ap_names) if semantic.faithful_ir else semantic.aps
    trace = [set() for _ in range(horizon)]
    if not aps:
        return trace
    trigger = semantic.trigger if semantic.trigger in aps else aps[0]
    response = semantic.response if semantic.response in aps else aps[-1]
    if obligation.expected_relation == "satisfy":
        if trigger and response and trigger != response:
            if variant % 3 == 1:
                trace[0].add(trigger)
                trace[-1].add(response)
            elif variant % 3 == 2:
                trace[0].update((trigger, response))
            else:
                for step in trace:
                    step.update(aps)
        else:
            trace[-1].add(aps[0])
    elif obligation.expected_relation == "violate":
        if trigger:
            trace[0 if variant % 2 else -1].add(trigger)
        elif aps:
            trace[0].add(aps[0])
    else:
        if trigger and response and trigger != response:
            target = -1 if variant % 2 else 0
            trace[target].update((trigger, response))
        else:
            trace[-1].add(aps[0])
    return trace


def _relation_probe_trace(
    *,
    semantic: SemanticStructure,
    obligation,
    horizon: int,
    variant: int = 0,
) -> list[set[str]] | None:
    ir = semantic.faithful_ir
    if ir is None or not obligation.ir_node_ids:
        return None
    node = ir.node_map.get(obligation.ir_node_ids[0])
    if node is None:
        return None
    if node.node_type == "periodicity":
        return _periodicity_probe_trace(
            ir,
            node,
            obligation=obligation,
            horizon=horizon,
            variant=variant,
        )
    if node.node_type not in {"until", "precedence"}:
        return None
    participants = tuple(
        _first_ap_descendant(ir, child_id) for child_id in node.children
    )
    if len(participants) != 2 or not all(participants):
        return None
    first, second = participants
    role = obligation.required_trace_role
    if node.node_type == "until":
        left_true = _condition_probe_step(ir, node.children[0], holds=True)
        left_false = _condition_probe_step(ir, node.children[0], holds=False)
        right_true = _condition_probe_step(ir, node.children[1], holds=True)
        shapes = {
            "immediate_until_witness": [right_true],
            "delayed_until_witness": [left_true, right_true],
            "left_break_before_witness": [left_true, left_false, right_true],
            "missing_until_witness": [left_true, left_true],
            "left_at_witness_boundary": [
                left_true,
                set(right_true) | set(left_false),
            ],
            "final_state_witness": [left_true, left_true, right_true],
        }
    else:
        prerequisite, constrained = first, second
        shapes = {
            "empty_prefix_gate": [set()],
            "missing_predecessor": [{constrained}],
            "predecessor_only": [{prerequisite}],
            "preceded_occurrence": [{prerequisite}, {constrained}],
            "late_predecessor_boundary": [{constrained}, {prerequisite}],
            "same_step_boundary": [{prerequisite, constrained}],
            "repeated_occurrence": [
                {prerequisite},
                {constrained},
                {constrained},
            ],
            "repeated_precedence_cycle": [
                {prerequisite},
                {constrained},
                {prerequisite},
                {constrained},
            ],
        }
    shape = shapes.get(role)
    if shape is None:
        return None
    fitted = [set(step) for step in shape[:horizon]]
    while len(fitted) < horizon:
        fitted.append(set())
    return fitted


def _periodicity_probe_trace(
    ir,
    node,
    *,
    obligation,
    horizon: int,
    variant: int,
) -> list[set[str]]:
    """Instantiate phase-aware periodic queries without assigning their labels."""

    params = dict(node.parameters)
    required = set(PERIODICITY_REQUIRED_PARAMETERS)
    if not required <= set(params):
        raise ValueError(
            f"periodicity node {node.node_id} is missing reviewed parameters: "
            + ", ".join(sorted(required - set(params)))
        )
    period = max(1, int(params["period"]))
    mode = str(params["mode"])
    phase = max(0, int(params.get("phase", 1)))
    ap = _first_ap_descendant(ir, node.node_id)
    trace = [set() for _ in range(max(1, horizon))]
    scheduled = tuple(range(phase, len(trace), period))

    if obligation.expected_relation == "satisfy":
        for position in scheduled:
            trace[position].add(ap)
        if mode in {"minimum_spacing", "exact_inter_occurrence_spacing"} and variant % 2:
            trace = [set() for _ in trace]
            trace[min(phase, len(trace) - 1)].add(ap)
        return trace

    if obligation.expected_relation == "violate":
        if variant % 2 and scheduled:
            for position in scheduled[1:]:
                trace[position].add(ap)
        else:
            for position in scheduled:
                trace[position].add(ap)
            off_phase = next(
                (index for index in range(len(trace)) if index not in scheduled),
                0,
            )
            trace[off_phase].add(ap)
        return trace

    # Boundary roles distinguish short-prefix, exact phase, spacing and off-phase.
    if obligation.required_trace_role == "period_terminal_boundary":
        return [set() for _ in range(max(1, min(phase or 1, len(trace))))]
    if obligation.required_trace_role == "period_phase_boundary":
        shifted_phase = (phase + 1) % period
        for position in range(shifted_phase, len(trace), period):
            trace[position].add(ap)
        return trace
    for position in scheduled:
        trace[position].add(ap)
    if obligation.required_trace_role == "period_off_phase_boundary":
        off_phase = next(
            (index for index in range(len(trace)) if index not in scheduled),
            0,
        )
        trace[off_phase].add(ap)
    elif obligation.required_trace_role == "period_spacing_boundary" and len(trace) > 1:
        trace[min(len(trace) - 1, max(1, period - 1))].add(ap)
    return trace


def _first_ap_descendant(ir, node_id: str) -> str:
    node = ir.node_map.get(node_id)
    if node is None:
        return ""
    if node.node_type == "ap_condition":
        return str(node.parameter("ap_name", ""))
    for child_id in node.children:
        ap = _first_ap_descendant(ir, child_id)
        if ap:
            return ap
    return ""


def _condition_probe_step(ir, node_id: str, *, holds: bool) -> set[str]:
    """Construct a small AP state that exercises a simple IR condition."""

    node = ir.node_map.get(node_id)
    if node is None:
        return set()
    if node.node_type == "ap_condition":
        ap = str(node.parameter("ap_name", ""))
        return {ap} if holds and ap else set()
    if node.node_type == "negation" and node.children:
        return _condition_probe_step(ir, node.children[0], holds=not holds)
    if node.node_type == "conjunction":
        if holds:
            result: set[str] = set()
            for child in node.children:
                result.update(_condition_probe_step(ir, child, holds=True))
            return result
        return (
            _condition_probe_step(ir, node.children[0], holds=False)
            if node.children
            else set()
        )
    if node.node_type == "disjunction":
        if holds:
            return (
                _condition_probe_step(ir, node.children[0], holds=True)
                if node.children
                else set()
            )
        result: set[str] = set()
        for child in node.children:
            result.update(_condition_probe_step(ir, child, holds=False))
        return result
    ap = _first_ap_descendant(ir, node_id)
    return {ap} if holds and ap else set()


def _next_distinct_formula_free_trace(
    *,
    states: list[set[str]],
    horizon: int,
    seen: set[tuple[tuple[str, ...], ...]],
    start_index: int,
) -> tuple[list[set[str]], int] | None:
    capacity = len(states) ** horizon
    for offset in range(capacity):
        value = (start_index + offset) % capacity
        trace: list[set[str]] = []
        for _ in range(horizon):
            trace.append(set(states[value % len(states)]))
            value //= len(states)
        key = tuple(tuple(sorted(step)) for step in trace)
        if key not in seen:
            return trace, start_index + offset + 1
    return None


class TraceGenerationAgent(BaseAgent):
    """有限轨迹生成 Agent。"""

    def __init__(self, retriever, llm_client: JSONLLMClient | None = None):
        super().__init__(retriever=retriever, name="TraceGenerationAgent", llm_client=llm_client)

    @observed_stage("trace_generation")
    def generate_queries(
        self,
        requirement: str,
        semantic: SemanticStructure,
        feedback_history: list[str] | None = None,
        target_trace_length: int | None = None,
        target_trace_count: int | None = None,
        *,
        data_split: str = "test",
        label_source: str = "human",
        guided_context: GuidedLearningContext | None = None,
    ) -> list[TraceQuery]:
        """
        Generate pending finite-trace questions from reviewed semantics.

        This path never compiles an obligation formula and never emits expected
        truth values. ``requirement`` and feedback are accepted for API
        compatibility, but only the reviewed SemanticStructure drives the
        formula-free query plan.
        """

        del requirement, feedback_history
        if (
            guided_context is not None
            and guided_context.attempt_index >= 2
            and guided_context.expected_traces
        ):
            queries: list[TraceQuery] = []
            for index, trace in enumerate(guided_context.expected_traces, start=1):
                role = trace.role or trace.kind
                queries.append(
                    TraceQuery(
                        query_id=trace.name or f"QUERY-{index:03d}",
                        role=role,
                        target_ambiguity=(
                            trace.target_ambiguity
                            or trace.query_role
                            or role
                        ),
                        constraints=trace.query_constraints
                        or (
                            "match warmup expected trace shape without exposing truth label",
                        ),
                        trace=[set(step) for step in trace.trace],
                        reason="Warmup guided query shape; label is assigned later.",
                        beginner_view=trace.beginner_view,
                        data_split=data_split,
                        label_source=label_source,
                        ir_node_ids=trace.covered_ir_node_ids,
                        obligation_ids=trace.obligation_ids,
                        ir_version=trace.ir_version,
                        target_ambiguity_axis_ids=trace.target_ambiguity_axis_ids,
                        generation_source="warmup_gold_shape_attempt_2",
                    )
                )
            return queries
        if semantic.faithful_ir is not None:
            assert_ir_ready_for_trace_generation(semantic.faithful_ir)
        plan = plan_semantic_obligations(semantic)
        ir_plan = (
            compile_executable_trace_plan(semantic.faithful_ir)
            if semantic.faithful_ir is not None
            else None
        )
        if ir_plan is not None and ir_plan.relation_feasibility is not None:
            require_contingent_relation(ir_plan.relation_feasibility)
        if ir_plan is not None:
            return _instantiate_ir_trace_queries(
                semantic,
                ir_plan,
                target_trace_length=target_trace_length,
                target_trace_count=target_trace_count,
                data_split=data_split,
                label_source=label_source,
            )
        queries = instantiate_trace_queries(
            plan,
            semantic,
            target_trace_count=max(
                target_trace_count or MIN_REQUIRED_TRACE_COUNT,
                len(ir_plan.obligations) if ir_plan is not None else 0,
            ),
            target_trace_length=target_trace_length,
            data_split=data_split,
            label_source=label_source,
        )
        return queries

    def generate_authoritative_queries(
        self,
        requirement: str,
        semantic: SemanticStructure,
        *,
        trace_session: AuthoritativeTraceSession,
        feedback_history: list[str] | None = None,
        target_trace_length: int | None = None,
        target_trace_count: int | None = None,
        data_split: str = "test",
        label_source: str = "human",
        guided_context: GuidedLearningContext | None = None,
    ) -> list[TraceQuery]:
        """Generate production trace queries only from a promoted session."""

        validate_authoritative_trace_session(trace_session, semantic)
        return self.generate_queries(
            requirement,
            semantic,
            feedback_history=feedback_history,
            target_trace_length=target_trace_length,
            target_trace_count=target_trace_count,
            data_split=data_split,
            label_source=label_source,
            guided_context=guided_context,
        )

    def generate(
        self,
        requirement: str,
        semantic: SemanticStructure,
        feedback_history: list[str] | None = None,
        target_trace_length: int | None = None,
        target_trace_count: int | None = None,
        guided_context: GuidedLearningContext | None = None,
        allow_degraded: bool = False,
    ) -> list[LabeledTrace]:
        """
        生成正轨迹、反轨迹和边界轨迹。

        当前原型先支持 response 模式。每条轨迹都带有 beginner_view，人工评审
        时优先展示这个字段。target_trace_length 用于长度评审失败后的
        立即重生成。
        """

        feedback_history = feedback_history or []
        if (
            semantic.faithful_ir is None
            and guided_context is not None
            and guided_context.expected_traces
            and target_trace_length is None
            and target_trace_count is None
        ):
            traces = _finalize_guided_trace_set(
                list(guided_context.expected_traces),
                target_trace_length=target_trace_length,
            )
            traces = [self._ensure_beginner_view(trace) for trace in traces]
            return _tag_faithful_ir_traces(traces, semantic)
        if semantic.faithful_ir is not None:
            raise ValueError(
                "Native FaithfulSemanticIR traces require independent query adjudication; "
                "use generate_queries with human or independent_judge review."
            )
        effective_requirement = finite_requirement_for_pipeline(requirement, semantic)
        obligation_profile = analyze_obligations(
            "\n".join((effective_requirement, *feedback_history)),
            semantic,
        )
        length_instruction = ""
        if target_trace_length is not None:
            length_instruction = f"目标有限轨迹长度: {target_trace_length}"

        stage = (
            "formula_text_trace_generation"
            if semantic.formula_text or semantic.pattern != "response"
            else "trace_generation"
        )
        prompt_context = build_stage_prompt_context(
            self.retriever,
            stage=stage,
            query_parts=(
                "有限轨迹生成",
                effective_requirement,
                semantic.pattern,
                semantic.trigger,
                semantic.response,
                length_instruction,
                "\n".join(feedback_history),
                _guided_query_text(guided_context, "trace"),
            ),
            domain_sources=(KB_LTLF_SEMANTICS, KB_TRACE_EXAMPLES),
            example_sources=(
                TRACE_SOURCE,
                GUIDED_TRACE_SOURCE,
                FULL_PROCESS_SOURCE,
                GUIDED_FULL_PROCESS_SOURCE,
            ),
            agent_function_contract=_trace_generation_contract(stage, guided_context),
        )
        combined_context = prompt_context.as_rag_context()
        context_trace_count = _suggested_trace_count_from_context(combined_context)
        guided_trace_count = _guided_expected_trace_count(guided_context)
        if guided_trace_count is not None:
            suggested_trace_count = guided_trace_count
        elif semantic.pattern == "response":
            suggested_trace_count = _minimum_required_trace_count(
                target_trace_count or DEFAULT_RESPONSE_TRACE_COUNT
            )
        else:
            suggested_trace_count = _minimum_required_trace_count(
                target_trace_count or context_trace_count
            )

        learned_handler = get_pattern_handler(semantic.pattern)
        if semantic.formula_text:
            if self.llm_enabled:
                traces = self._generate_learned_pattern_with_llm(
                    requirement=requirement,
                    finite_requirement=effective_requirement,
                    semantic=semantic,
                    feedback_history=feedback_history,
                    target_trace_length=target_trace_length,
                    suggested_trace_count=suggested_trace_count,
                    prompt_context=prompt_context,
                    guided_context=guided_context,
                )
            else:
                traces = _build_formula_text_traces(
                    semantic=semantic,
                    target_trace_length=target_trace_length,
                    target_trace_count=suggested_trace_count,
                    require_positive_negative=not allow_degraded,
                )
            traces = _finalize_trace_set_for_mode(
                traces=traces,
                semantic=semantic,
                target_trace_length=target_trace_length,
                target_trace_count=suggested_trace_count,
                guided_context=guided_context,
                allow_degraded=allow_degraded,
            )
            traces = self._apply_feedback(traces, feedback_history)
            return _finalize_obligation_aware_trace_output(
                traces=traces,
                semantic=semantic,
                profile=obligation_profile,
                target_trace_length=target_trace_length,
                target_trace_count=suggested_trace_count,
                guided=bool(guided_context and guided_context.expected_traces),
                preserve_generated=bool(feedback_history),
            )

        if semantic.pattern != "response":
            if self.llm_enabled:
                traces = self._generate_learned_pattern_with_llm(
                    requirement=requirement,
                    finite_requirement=effective_requirement,
                    semantic=semantic,
                    feedback_history=feedback_history,
                    target_trace_length=target_trace_length,
                    suggested_trace_count=suggested_trace_count,
                    prompt_context=prompt_context,
                    guided_context=guided_context,
                )
                traces = _finalize_trace_set_for_mode(
                    traces=traces,
                    semantic=semantic,
                    target_trace_length=target_trace_length,
                    target_trace_count=suggested_trace_count,
                    guided_context=guided_context,
                    allow_degraded=allow_degraded,
                )
                traces = self._apply_feedback(traces, feedback_history)
                return _finalize_obligation_aware_trace_output(
                    traces=traces,
                    semantic=semantic,
                    profile=obligation_profile,
                    target_trace_length=target_trace_length,
                    target_trace_count=suggested_trace_count,
                    guided=bool(guided_context and guided_context.expected_traces),
                    preserve_generated=bool(feedback_history),
                )

            learned_formula_semantic = _learned_formula_fallback_semantic(semantic)
            if learned_formula_semantic is not None:
                traces = _build_formula_text_traces(
                    semantic=learned_formula_semantic,
                    target_trace_length=target_trace_length,
                    target_trace_count=suggested_trace_count,
                )
            else:
                traces = _build_semantic_pattern_traces(
                    requirement=effective_requirement,
                    semantic=semantic,
                    target_trace_length=target_trace_length,
                    target_trace_count=suggested_trace_count,
                )
            if not traces:
                traces = build_obligation_trace_set(
                    obligation_profile,
                    semantic,
                    target_trace_count=suggested_trace_count,
                    target_trace_length=target_trace_length,
                )
            if not traces:
                raise ValueError(
                    f"模式 {semantic.pattern} 暂不能由当前语义字段和轨迹契约"
                    "确定性生成有限轨迹；请补充更明确的 components 或使用 --llm api。"
                )
            traces = _finalize_trace_set_for_mode(
                traces=traces,
                semantic=semantic,
                target_trace_length=target_trace_length,
                target_trace_count=suggested_trace_count,
                guided_context=guided_context,
                allow_degraded=allow_degraded,
            )
            traces = self._apply_feedback(traces, feedback_history)
            return _finalize_obligation_aware_trace_output(
                traces=traces,
                semantic=semantic,
                profile=obligation_profile,
                target_trace_length=target_trace_length,
                target_trace_count=suggested_trace_count,
                guided=bool(guided_context and guided_context.expected_traces),
                preserve_generated=bool(feedback_history),
            )

        if _is_degenerate_response_pair(semantic.trigger, semantic.response):
            traces = _build_degenerate_response_traces(
                trigger=semantic.trigger,
                target_trace_length=target_trace_length,
                target_trace_count=suggested_trace_count,
            )
        elif self.llm_enabled:
            traces = self._generate_with_llm(
                requirement=requirement,
                finite_requirement=effective_requirement,
                semantic=semantic,
                feedback_history=feedback_history,
                target_trace_length=target_trace_length,
                suggested_trace_count=suggested_trace_count,
                prompt_context=prompt_context,
                guided_context=guided_context,
            )
            if guided_trace_count is None:
                traces = _ensure_required_late_trigger_traces(
                    traces=traces,
                    trigger=semantic.trigger,
                    response=semantic.response,
                    target_trace_length=target_trace_length,
                )
                if suggested_trace_count == DEFAULT_RESPONSE_TRACE_COUNT:
                    traces = _canonicalize_default_response_traces(
                        traces=traces,
                        trigger=semantic.trigger,
                        response=semantic.response,
                        target_trace_length=target_trace_length,
                    )
        else:
            traces = build_response_traces(
                semantic.trigger,
                semantic.response,
                target_length=target_trace_length,
                target_count=suggested_trace_count,
            )

        traces = _finalize_trace_set_for_mode(
            traces=traces,
            semantic=semantic,
            target_trace_length=target_trace_length,
            target_trace_count=suggested_trace_count,
            guided_context=guided_context,
            allow_degraded=allow_degraded,
        )
        traces = self._apply_feedback(traces, feedback_history)
        traces = _finalize_obligation_aware_trace_output(
            traces=traces,
            semantic=semantic,
            profile=obligation_profile,
            target_trace_length=target_trace_length,
            target_trace_count=suggested_trace_count,
            guided=bool(guided_context and guided_context.expected_traces),
            preserve_generated=bool(feedback_history),
        )
        return _tag_faithful_ir_traces(traces, semantic)

    def _generate_with_llm(
        self,
        requirement: str,
        finite_requirement: str,
        semantic: SemanticStructure,
        feedback_history: list[str],
        target_trace_length: int | None,
        suggested_trace_count: int | None,
        prompt_context,
        guided_context: GuidedLearningContext | None = None,
    ) -> list[LabeledTrace]:
        extra_rules = (
            (
                guided_prompt_rule("trace_generation"),
                "有限轨迹数量、类型分布和 boundary 数量必须服从 "
                "payload.guided_learning.expected_traces；不要套用主体需求的 "
                "6 条覆盖契约，也不要额外补齐两条 boundary。",
                "trace.reason 只能说明对应 query role 和轨迹形状，不得引用公式或真值标签。",
            )
            if guided_context is not None
            else ()
        )
        coverage_rules = (
            (
                "当前是 warmup guided trace generation；只按 "
                "payload.guided_learning.expected_traces 的数量和类型生成有限轨迹。",
                "boundary 只在 expected_traces 中出现时才需要出现；不得为了主体测试覆盖契约额外添加 boundary。",
            )
            if guided_context is not None and guided_context.expected_traces
            else (
                "默认必须且仅生成 6 条有限轨迹：一条普通 positive、"
                "一条 late-trigger positive、一条普通 negative、"
                "一条 late-trigger negative 和两条不同 boundary。",
                "只有 payload.suggested_trace_count 大于 6 时，才可以生成更多补充轨迹。",
                "boundary 用于区分 F、XF、X 等边界语义，至少两条且轨迹内容必须不同。",
                "必须包含 late-trigger 正轨迹和 late-trigger 反轨迹："
                "也就是 trigger 不在 t0，而是在后续时间步出现。",
                "late-trigger 反轨迹用于区分 A -> F B 和 "
                "G(A -> F B)，不能省略。",
            )
        )
        trace_kind_rule = (
            "只为当前已确认的 semantic 生成 Expected Finite Traces 中列出的 kind；"
            "kind 可以是 positive、negative 或 boundary，但不要求三类都出现。"
            if guided_context is not None and guided_context.expected_traces
            else "只为当前已确认的 semantic 生成 positive、negative、boundary 三类有限轨迹。"
        )
        system_prompt = structured_system_prompt(
            agent_name="有限轨迹生成 Agent",
            stage_name="trace_generation",
            extra_rules=(
                trace_kind_rule,
                *coverage_rules,
                "trace 必须是时间步数组，每个时间步是 AP 字符串数组。",
                "整条 trace 不能是空数组；如果某个时间步没有 AP，用 [] 表示该时间步。",
                *extra_rules,
            ),
        )
        rag_context = prompt_context.as_rag_context()
        payload = {
            "task": "trace_generation",
            "requirement": finite_requirement,
            "original_requirement": requirement,
            "trace_domain": semantic.trace_domain,
            "domain_warning": semantic.domain_warning,
            "semantic": {
                "trigger": semantic.trigger,
                "response": semantic.response,
                "pattern": semantic.pattern,
                "scope_keyword": semantic.scope_keyword,
                "temporal_keyword": semantic.temporal_keyword,
                "aps": list(semantic.aps),
                "components": dict(semantic.components),
            },
            "target_trace_length": target_trace_length,
            "suggested_trace_count": suggested_trace_count,
            "feedback_history": feedback_history,
            "guided_learning": (
                guided_context.prompt_payload("trace_generation")
                if guided_context is not None
                else None
            ),
            **prompt_context.payload_fields(),
            "rag_context": rag_context,
            "output_schema": _trace_output_schema(semantic, guided_context),
        }
        raw_result = self.complete_json(system_prompt, payload, max_tokens=1800)
        items = coerce_list_field(
            raw_result,
            "traces",
            aliases=("trace_examples", "examples"),
            single_object_keys=("trace", "kind"),
        )
        traces = _parse_llm_trace_items(items)

        if target_trace_length is not None:
            for trace in traces:
                if len(trace.trace) != target_trace_length:
                    raise ValueError(
                        f"LLM 返回轨迹 {trace.name} 的长度为 {len(trace.trace)}，"
                        f"但目标长度是 {target_trace_length}。"
                    )

        if guided_context is not None and guided_context.expected_traces:
            return _finalize_guided_trace_set(
                traces,
                target_trace_length=target_trace_length,
            )

        kinds = {trace.kind for trace in traces}
        missing = ALLOWED_TRACE_KINDS - kinds
        if missing:
            traces = _ensure_response_required_trace_kinds(
                traces=traces,
                trigger=semantic.trigger,
                response=semantic.response,
                target_trace_length=target_trace_length,
                missing=missing,
            )
        traces = _ensure_response_trace_count(
            traces=traces,
            trigger=semantic.trigger,
            response=semantic.response,
            target_trace_length=target_trace_length,
            target_trace_count=suggested_trace_count,
        )
        return traces

    def _generate_learned_pattern_with_llm(
        self,
        requirement: str,
        finite_requirement: str,
        semantic: SemanticStructure,
        feedback_history: list[str],
        target_trace_length: int | None,
        suggested_trace_count: int | None,
        prompt_context,
        guided_context: GuidedLearningContext | None = None,
    ) -> list[LabeledTrace]:
        extra_rules = (
            (
                guided_prompt_rule("formula_text_trace_generation"),
                "有限轨迹数量、类型分布和 boundary 数量必须服从 "
                "payload.guided_learning.expected_traces；不要套用主体需求的 "
                "coverage contract。",
                "trace.reason 只能说明对应 query role 和轨迹形状，不得引用公式或真值标签。",
            )
            if guided_context is not None
            else ()
        )
        coverage_rules = (
            (
                "当前是 warmup guided formula_text trace generation；只按 "
                "payload.guided_learning.expected_traces 的数量和类型生成有限轨迹。",
                "boundary 只在 expected_traces 中出现时才需要出现；不得额外补齐两条 boundary。",
            )
            if guided_context is not None and guided_context.expected_traces
            else (
                "只根据 semantic.formula_text、components、需求文本、人工反馈、"
                "领域知识和 Top-5 示例格式生成 positive、negative、boundary 三类有限轨迹。",
                "boundary 轨迹至少两条，且有限轨迹内容必须不同。",
            )
        )
        system_prompt = structured_system_prompt(
            agent_name="有限轨迹生成 Agent",
            stage_name="formula_text_trace_generation",
            extra_rules=(
                "当前 pattern 可能是未知模式；不要创建新的 pattern handler。",
                *coverage_rules,
                "不要重新解释自然语言，不要补全候选公式。",
                "trace 必须是时间步数组，每个时间步是 AP 字符串数组。",
                "整条 trace 不能是空数组；如果某个时间步没有 AP，用 [] 表示该时间步。",
                *extra_rules,
            ),
        )
        rag_context = prompt_context.as_rag_context()
        payload = {
            "task": "formula_text_trace_generation",
            "requirement": finite_requirement,
            "original_requirement": requirement,
            "trace_domain": semantic.trace_domain,
            "domain_warning": semantic.domain_warning,
            "semantic": {
                "pattern": semantic.pattern,
                "aps": semantic.aps,
                "formula_text": semantic.formula_text,
                "components": dict(semantic.components),
                "scope_keyword": semantic.scope_keyword,
                "temporal_keyword": semantic.temporal_keyword,
            },
            "target_trace_length": target_trace_length,
            "suggested_trace_count": suggested_trace_count,
            "feedback_history": feedback_history,
            "guided_learning": (
                guided_context.prompt_payload("formula_text_trace_generation")
                if guided_context is not None
                else None
            ),
            **prompt_context.payload_fields(),
            "rag_context": rag_context,
            "output_schema": _trace_output_schema(semantic, guided_context),
        }
        raw_result = self.complete_json(system_prompt, payload, max_tokens=1800)
        items = coerce_list_field(
            raw_result,
            "traces",
            aliases=("trace_examples", "examples"),
            single_object_keys=("trace", "kind"),
        )
        traces = _parse_llm_trace_items(items)

        if target_trace_length is not None:
            for trace in traces:
                if len(trace.trace) != target_trace_length:
                    raise ValueError(
                        f"LLM 返回轨迹 {trace.name} 的长度为 {len(trace.trace)}，"
                        f"但目标长度是 {target_trace_length}。"
                    )

        if guided_context is not None and guided_context.expected_traces:
            return _finalize_guided_trace_set(
                traces,
                target_trace_length=target_trace_length,
            )

        return traces

    def _apply_feedback(
        self,
        traces: list[LabeledTrace],
        feedback_history: list[str],
    ) -> list[LabeledTrace]:
        """
        根据人工意见微调轨迹。

        示例：如果人工认为“同一时刻发生不算满足”，可以在意见里写：
            同时发生 不满足
        Agent 会把边界轨迹的 expected 改为 False。
        """

        feedback_text = "\n".join(feedback_history)
        if "同时" not in feedback_text and "同一" not in feedback_text:
            return traces
        if "不满足" not in feedback_text and "false" not in feedback_text.lower():
            return traces

        adjusted: list[LabeledTrace] = []
        for trace in traces:
            if trace.kind == "boundary" and "same_time" in trace.name:
                adjusted.append(
                    replace(
                        trace,
                        expected=False,
                        reason=(
                            trace.reason
                            + " 人工评审指出该需求要求响应必须发生在后续时刻，"
                            "因此同一时刻边界轨迹改为不满足。"
                        ),
                    )
                )
            else:
                adjusted.append(trace)
        return adjusted

    def _ensure_beginner_view(self, trace: LabeledTrace) -> LabeledTrace:
        """确保每条轨迹都有零基础可读视图。"""

        if trace.beginner_view:
            return trace
        return replace(trace, beginner_view=format_trace_for_beginner(trace.trace))


def _trace_generation_contract(
    stage: str,
    guided_context: GuidedLearningContext | None = None,
) -> dict:
    if guided_context is not None and guided_context.expected_traces:
        return {
            "agent": "TraceGenerationAgent",
            "stage": stage,
            "responsibilities": [
                "Generate finite traces for the already accepted semantic structure.",
                "Match the warmup expected trace count and expected trace kind distribution.",
                "Use boundary traces only when they are present in the warmup expected traces.",
                "Use Top-5 examples only as secondary style references.",
            ],
            "forbidden": [
                "Do not change trigger, response, pattern, formula_text, or components.",
                "Do not generate skeletons, temporal holes, completed formulas, or final rankings.",
                "Do not introduce APs that are absent from the semantic structure.",
                "Do not apply the main requirement coverage contract or top up to six traces.",
            ],
        }

    return {
        "agent": "TraceGenerationAgent",
        "stage": stage,
        "responsibilities": [
            "Generate finite traces for the already accepted semantic structure.",
            "Return positive, negative, and boundary traces with explicit expected labels.",
            "Use Top-5 examples only to mirror trace coverage patterns and naming style.",
        ],
        "forbidden": [
            "Do not change trigger, response, pattern, formula_text, or components.",
            "Do not generate skeletons, temporal holes, completed formulas, or final rankings.",
            "Do not introduce APs that are absent from the semantic structure.",
        ],
    }


def _guided_query_text(
    guided_context: GuidedLearningContext | None,
    stage: str,
) -> str:
    if guided_context is None:
        return ""
    return (
        f"warmup guided {stage} {guided_context.requirement_id} "
        "stage_scoped_teacher_signal"
    )


def _guided_expected_trace_count(
    guided_context: GuidedLearningContext | None,
) -> int | None:
    if guided_context is None or not guided_context.expected_traces:
        return None
    return len(guided_context.expected_traces)


def _trace_output_schema(
    semantic: SemanticStructure,
    guided_context: GuidedLearningContext | None,
) -> dict:
    if guided_context is not None and guided_context.expected_traces:
        return {
            "traces": [
                {
                    "name": trace.name,
                    "kind": trace.kind,
                    "trace": trace_to_serializable(trace.trace),
                    "reason": (
                        "explain how this generated trace corresponds to the "
                        "warmup expected query role and shape"
                    ),
                }
                for trace in guided_context.expected_traces
            ]
        }

    if semantic.formula_text or semantic.pattern != "response":
        return {
            "traces": [
                {
                    "name": "pos_satisfying",
                    "kind": "positive|negative|boundary",
                    "trace": [[semantic.aps[0] if semantic.aps else "prop_1"]],
                    "expected": True,
                    "reason": "why this trace matches the learned pattern semantics",
                }
            ]
        }

    return {
        "traces": [
            {
                "name": f"pos_{semantic.trigger}_then_{semantic.response}",
                "kind": "positive",
                "trace": [[semantic.trigger], [], [semantic.response]],
                "expected": True,
                "reason": "ordinary trigger at t0 is eventually followed by response",
            },
            {
                "name": f"pos_late_{semantic.trigger}_then_{semantic.response}",
                "kind": "positive",
                "trace": [[], [semantic.trigger], [semantic.response]],
                "expected": True,
                "reason": "trigger occurs after t0 and response appears later",
            },
            {
                "name": f"neg_{semantic.trigger}_without_{semantic.response}",
                "kind": "negative",
                "trace": [[semantic.trigger], [], []],
                "expected": False,
                "reason": "ordinary trigger at t0 is never followed by response",
            },
            {
                "name": f"neg_late_{semantic.trigger}_without_{semantic.response}",
                "kind": "negative",
                "trace": [[], [semantic.trigger], []],
                "expected": False,
                "reason": (
                    "trigger occurs after t0 but response never follows; "
                    "this distinguishes whether an outer G is required"
                ),
            },
            {
                "name": f"bound_{semantic.trigger}_and_{semantic.response}_same_time",
                "kind": "boundary",
                "trace": [[semantic.trigger, semantic.response]],
                "expected": True,
                "reason": "boundary trace that distinguishes F from XF/X",
            },
            {
                "name": f"bound_late_{semantic.trigger}_and_{semantic.response}_same_time",
                "kind": "boundary",
                "trace": [[], [], [semantic.trigger, semantic.response]],
                "expected": True,
                "reason": "a distinct late boundary trace for boundary diversity",
            },
        ]
    }


def _minimum_required_trace_count(target_trace_count: int | None) -> int:
    return max(MIN_REQUIRED_TRACE_COUNT, target_trace_count or 0)


def _finalize_trace_set_for_mode(
    *,
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    target_trace_length: int | None,
    target_trace_count: int | None,
    guided_context: GuidedLearningContext | None,
    allow_degraded: bool = False,
) -> list[LabeledTrace]:
    if guided_context is not None and guided_context.expected_traces:
        return _finalize_guided_trace_set(
            traces,
            target_trace_length=target_trace_length,
        )
    return _finalize_required_trace_set(
        traces=traces,
        semantic=semantic,
        target_trace_length=target_trace_length,
        target_trace_count=target_trace_count,
        allow_degraded=allow_degraded,
    )


def _finalize_guided_trace_set(
    traces: list[LabeledTrace],
    *,
    target_trace_length: int | None,
) -> list[LabeledTrace]:
    """Keep guided warmup traces under the expected-artifact contract only."""

    if target_trace_length is not None:
        mismatched = [
            f"{trace.name}={len(trace.trace)}"
            for trace in traces
            if len(trace.trace) != target_trace_length
        ]
        if mismatched:
            raise ValueError(
                "guided warmup 有限轨迹长度不符合 target_trace_length: "
                + ", ".join(mismatched)
            )
    return _deduplicate_trace_names(list(traces))


def _finalize_required_trace_set(
    *,
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    target_trace_length: int | None,
    target_trace_count: int | None,
    allow_degraded: bool = False,
) -> list[LabeledTrace]:
    """Normalize every generated trace set to the project-wide coverage contract."""

    required_count = _minimum_required_trace_count(target_trace_count)
    if semantic.pattern == "response" and not semantic.formula_text:
        traces = _ensure_required_response_trace_set(
            traces=traces,
            trigger=semantic.trigger,
            response=semantic.response,
            target_trace_length=target_trace_length,
            target_trace_count=required_count,
        )
    elif semantic.formula_text:
        traces = _ensure_formula_text_trace_set(
            traces=traces,
            semantic=semantic,
            target_trace_length=target_trace_length,
            target_trace_count=required_count,
        )
    else:
        traces = _deduplicate_trace_names(list(traces))
        learned_formula_semantic = _learned_formula_fallback_semantic(semantic)
        if learned_formula_semantic is not None and _formula_text_trace_set_needs_top_up(
            traces,
            required_count,
        ):
            traces = _ensure_formula_text_trace_set(
                traces=traces,
                semantic=learned_formula_semantic,
                target_trace_length=target_trace_length,
                target_trace_count=required_count,
            )
        if _formula_text_trace_set_needs_top_up(traces, required_count):
            traces = _ensure_semantic_pattern_trace_set(
                traces=traces,
                semantic=semantic,
                target_trace_length=target_trace_length,
                target_trace_count=required_count,
            )

    try:
        traces = _ensure_min_distinct_boundary_traces(
            traces=traces,
            semantic=semantic,
            target_trace_length=target_trace_length,
            target_trace_count=required_count,
        )
    except ValueError:
        if not allow_degraded or not _can_return_degraded_trace_set(traces):
            raise
    if semantic.formula_text:
        traces = _ensure_formula_text_trace_set(
            traces=traces,
            semantic=semantic,
            target_trace_length=target_trace_length,
            target_trace_count=required_count,
        )

    try:
        _validate_required_trace_set(traces, semantic)
    except ValueError:
        if not allow_degraded or not _can_return_degraded_trace_set(traces):
            raise
    return _deduplicate_trace_names(traces)


def _finalize_obligation_aware_trace_output(
    *,
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    profile,
    target_trace_length: int | None,
    target_trace_count: int | None,
    guided: bool = False,
    preserve_generated: bool = False,
) -> list[LabeledTrace]:
    enriched = enrich_trace_semantics(traces, profile)
    if guided:
        return _normalize_traces_against_profile(
            enriched,
            profile,
        )
    valid, conflicts = validate_trace_labels(enriched, profile)
    roles = set(covered_roles(enriched))
    confusions = set(covered_confusions(enriched))
    semantic_debt = (
        set(profile.required_roles) - roles
        or set(profile.confusion_targets) - confusions
    )
    should_rebuild = bool(conflicts) or (
        not preserve_generated
        and (
            profile.source.startswith("reviewed_")
            and semantic.trigger != semantic.response
            and bool(semantic_debt)
        )
    )
    if should_rebuild:
        rebuilt = build_obligation_trace_set(
            profile,
            semantic,
            target_trace_count=target_trace_count or MIN_REQUIRED_TRACE_COUNT,
            target_trace_length=target_trace_length,
        )
        if rebuilt:
            return _normalize_traces_against_profile(
                rebuilt,
                profile,
            )
    return _normalize_traces_against_profile(
        enriched,
        profile,
    )


def _normalize_traces_against_profile(
    traces: list[LabeledTrace],
    profile,
) -> list[LabeledTrace]:
    """Make generated labels and duplicate handling follow one obligation profile."""

    normalized: list[LabeledTrace] = []
    for trace in traces:
        if profile.formula is None:
            normalized.append(trace)
            continue
        expected = check_trace(trace.trace, profile.formula)
        kind = trace.kind
        if kind != "boundary":
            kind = "positive" if expected else "negative"
        normalized.append(
            replace(
                trace,
                kind=kind,
                expected=expected,
            )
        )

    by_shape: dict[tuple[tuple[str, ...], ...], list[LabeledTrace]] = {}
    shape_order: list[tuple[tuple[str, ...], ...]] = []
    for trace in normalized:
        shape = _trace_key(trace.trace)
        if shape not in by_shape:
            by_shape[shape] = []
            shape_order.append(shape)
        by_shape[shape].append(trace)

    resolved: list[LabeledTrace] = []
    for shape in shape_order:
        group = by_shape[shape]
        labels = {trace.expected for trace in group}
        if len(labels) > 1:
            resolved.append(
                sorted(
                    group,
                    key=lambda trace: (
                        trace.kind == "boundary",
                        trace.name,
                    ),
                )[0]
            )
            continue
        resolved.extend(group)
    return enrich_trace_semantics(_deduplicate_trace_names(resolved), profile)


def _ensure_required_response_trace_set(
    *,
    traces: list[LabeledTrace],
    trigger: str,
    response: str,
    target_trace_length: int | None,
    target_trace_count: int,
) -> list[LabeledTrace]:
    if _is_degenerate_response_pair(trigger, response):
        return _build_degenerate_response_traces(
            trigger=trigger,
            target_trace_length=target_trace_length,
            target_trace_count=target_trace_count,
        )

    canonical = _canonicalize_default_response_traces(
        traces=traces,
        trigger=trigger,
        response=response,
        target_trace_length=target_trace_length,
    )
    if len(canonical) >= target_trace_count:
        return canonical

    fallback_traces = build_response_traces(
        trigger=trigger,
        response=response,
        target_length=target_trace_length,
        target_count=target_trace_count,
    )
    completed = list(canonical)
    seen_shapes = {_trace_key(trace.trace) for trace in completed}
    for trace in [*traces, *fallback_traces]:
        if len(completed) >= target_trace_count:
            break
        shape = _trace_key(trace.trace)
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        completed.append(trace)
    return _deduplicate_trace_names(completed)


def _ensure_formula_text_trace_set(
    *,
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    target_trace_length: int | None,
    target_trace_count: int,
) -> list[LabeledTrace]:
    completed = _deduplicate_trace_names(list(traces))
    if not _formula_text_trace_set_needs_top_up(completed, target_trace_count):
        return completed

    fallback_traces = _build_formula_text_traces(
        semantic=semantic,
        target_trace_length=target_trace_length,
        target_trace_count=target_trace_count,
        require_positive_negative=False,
    )
    seen_shapes = {_trace_key(trace.trace) for trace in completed}
    for trace in fallback_traces:
        if not _formula_text_trace_set_needs_top_up(completed, target_trace_count):
            break
        shape = _trace_key(trace.trace)
        if shape in seen_shapes and not _formula_fallback_duplicate_can_help(
            completed,
            trace,
            target_trace_count,
        ):
            continue
        seen_shapes.add(shape)
        completed.append(trace)
    return _deduplicate_trace_names(completed)


def _formula_text_trace_set_needs_top_up(
    traces: list[LabeledTrace],
    target_trace_count: int,
) -> bool:
    return (
        len(traces) < target_trace_count
        or _kind_count(traces, "positive") < MIN_REQUIRED_KIND_TRACE_COUNT
        or _kind_count(traces, "negative") < MIN_REQUIRED_KIND_TRACE_COUNT
        or len(_boundary_trace_keys(traces)) < MIN_DISTINCT_BOUNDARY_TRACE_COUNT
    )


def _ensure_semantic_pattern_trace_set(
    *,
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    target_trace_length: int | None,
    target_trace_count: int,
) -> list[LabeledTrace]:
    completed = _deduplicate_trace_names(list(traces))
    if not _formula_text_trace_set_needs_top_up(completed, target_trace_count):
        return completed

    fallback_traces = _build_semantic_pattern_traces(
        requirement="",
        semantic=semantic,
        target_trace_length=target_trace_length,
        target_trace_count=target_trace_count,
    )
    seen_shapes = {_trace_key(trace.trace) for trace in completed}
    for trace in fallback_traces:
        if not _formula_text_trace_set_needs_top_up(completed, target_trace_count):
            break
        shape = _trace_key(trace.trace)
        if shape in seen_shapes and not _formula_fallback_duplicate_can_help(
            completed,
            trace,
            target_trace_count,
        ):
            continue
        seen_shapes.add(shape)
        completed.append(trace)
    return _deduplicate_trace_names(completed)


def _formula_fallback_duplicate_can_help(
    traces: list[LabeledTrace],
    trace: LabeledTrace,
    target_trace_count: int,
) -> bool:
    if trace.kind == "positive":
        return (
            _kind_count(traces, "positive") < MIN_REQUIRED_KIND_TRACE_COUNT
            or len(traces) < target_trace_count
        )
    if trace.kind == "negative":
        return (
            _kind_count(traces, "negative") < MIN_REQUIRED_KIND_TRACE_COUNT
            or len(traces) < target_trace_count
        )
    if trace.kind == "boundary":
        if _trace_key(trace.trace) in _boundary_trace_keys(traces):
            return False
        return len(_boundary_trace_keys(traces)) < MIN_DISTINCT_BOUNDARY_TRACE_COUNT
    return len(traces) < target_trace_count


def _validate_required_trace_set(
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
) -> None:
    errors = _required_trace_set_errors(traces, semantic)
    if errors:
        raise ValueError("有限轨迹集合不满足强制覆盖要求：" + "；".join(errors) + "。")


def _required_trace_set_errors(
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    profile=None,
) -> list[str]:
    errors: list[str] = []
    if len(traces) < MIN_REQUIRED_TRACE_COUNT:
        errors.append(f"轨迹数量为 {len(traces)}，至少需要 {MIN_REQUIRED_TRACE_COUNT} 条")

    if len(_boundary_trace_keys(traces)) < MIN_DISTINCT_BOUNDARY_TRACE_COUNT:
        errors.append("至少需要两条内容不同的 boundary 边界轨迹")

    use_canonical_response_contract = (
        semantic.pattern == "response"
        and not semantic.formula_text
        and (
            profile is None
            or (
                profile.source == "reviewed_response_semantics"
                and profile.scope == "global"
                and profile.temporal_operator == "F"
                and profile.required_horizon is None
            )
        )
    )
    if use_canonical_response_contract:
        if _is_degenerate_response_pair(semantic.trigger, semantic.response):
            _validate_degenerate_response_trace_set(traces, semantic, errors)
            return errors
        if not any(
            _is_general_response_positive(trace, semantic.trigger, semantic.response)
            for trace in traces
        ):
            errors.append("缺少普通正轨迹")
        if not _has_late_trigger_positive(traces, semantic.trigger, semantic.response):
            errors.append("缺少 late-trigger 正轨迹")
        if not any(
            _is_general_response_negative(trace, semantic.trigger, semantic.response)
            for trace in traces
        ):
            errors.append("缺少普通反轨迹")
        if not _has_late_trigger_negative(traces, semantic.trigger, semantic.response):
            errors.append("缺少 late-trigger 反轨迹")
    else:
        positive_count = _kind_count(traces, "positive")
        negative_count = _kind_count(traces, "negative")
        boundary_count = _kind_count(traces, "boundary")
        if positive_count < MIN_REQUIRED_KIND_TRACE_COUNT:
            errors.append(
                f"positive 轨迹为 {positive_count} 条，至少需要 "
                f"{MIN_REQUIRED_KIND_TRACE_COUNT} 条"
            )
        if negative_count < MIN_REQUIRED_KIND_TRACE_COUNT:
            errors.append(
                f"negative 轨迹为 {negative_count} 条，至少需要 "
                f"{MIN_REQUIRED_KIND_TRACE_COUNT} 条"
            )
        if boundary_count < MIN_DISTINCT_BOUNDARY_TRACE_COUNT:
            errors.append(
                f"boundary 轨迹为 {boundary_count} 条，至少需要 "
                f"{MIN_DISTINCT_BOUNDARY_TRACE_COUNT} 条"
            )

    return errors


def assess_trace_coverage(
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    *,
    requirement: str = "",
    feedback_history: list[str] | None = None,
    target_trace_count: int | None = None,
    retry_count: int = 0,
    guided: bool = False,
    allow_degraded: bool = False,
    trace_plan: TraceObligationPlan | None = None,
) -> TraceCoverageReport:
    """Assess coverage without changing or fabricating finite-trace labels."""

    # The current pipeline always carries a FaithfulSemanticIR.  Route every
    # such run through the IR-derived obligation contract, even when a legacy
    # warmup trace omitted ``query_role``.  Falling back based on that optional
    # display field previously let old formula/quota rules approve evidence
    # that the final IR coverage gate correctly rejected.
    if semantic.faithful_ir is not None:
        return _assess_adjudicated_query_coverage(
            traces,
            semantic,
            target_trace_count=target_trace_count,
            retry_count=retry_count,
            guided=guided,
            allow_degraded=allow_degraded,
            trace_plan=trace_plan,
        )

    from trace_obligations import (
        analyze_obligations,
        covered_confusions,
        covered_roles,
        enrich_trace_semantics,
        validate_trace_labels,
    )

    profile = analyze_obligations(
        "\n".join((requirement, *(feedback_history or []))),
        semantic,
    )
    enriched_traces = enrich_trace_semantics(traces, profile)
    positive_count = _kind_count(traces, "positive")
    negative_count = _kind_count(traces, "negative")
    boundary_count = _kind_count(traces, "boundary")
    distinct_boundary_count = len(_boundary_trace_keys(traces))
    required_count = (
        target_trace_count or len(traces)
        if guided
        else _minimum_required_trace_count(target_trace_count)
    )
    quota_missing = (
        ()
        if guided
        else tuple(_required_trace_set_errors(traces, semantic, profile))
    )
    executable_semantics = (
        _has_executable_trace_semantics(semantic) or profile.executable
    )
    label_valid_traces, semantic_conflicts = validate_trace_labels(
        enriched_traces,
        profile,
    )
    role_coverage = covered_roles(enriched_traces)
    confusion_coverage = covered_confusions(enriched_traces)
    missing_roles = (
        ()
        if guided
        else tuple(
            role
            for role in profile.required_roles
            if role not in role_coverage
        )
    )
    missing_confusions = (
        ()
        if guided
        else tuple(
            target
            for target in profile.confusion_targets
            if target not in confusion_coverage
        )
    )
    semantic_missing = tuple(
        [
            *[f"缺少语义角色: {role}" for role in missing_roles],
            *[f"缺少区分目标: {target}" for target in missing_confusions],
        ]
    )
    missing = tuple([*quota_missing, *semantic_missing])
    conflicts = tuple([*_trace_label_conflicts(traces), *semantic_conflicts])
    quota_complete = not quota_missing
    semantic_complete = not semantic_missing and not semantic_conflicts

    if conflicts:
        return TraceCoverageReport(
            status="rejected",
            required_count=required_count,
            actual_count=len(traces),
            positive_count=positive_count,
            negative_count=negative_count,
            boundary_count=boundary_count,
            distinct_boundary_count=distinct_boundary_count,
            missing=tuple(conflicts),
            coverage_debt=tuple(conflicts),
            retry_count=retry_count,
            executable_semantics=executable_semantics,
            reason="有限轨迹标签存在冲突，不能进入后续推导。",
            quota_complete=quota_complete,
            semantic_complete=False,
            covered_roles=role_coverage,
            missing_roles=missing_roles,
            covered_confusions=confusion_coverage,
            missing_confusions=missing_confusions,
            validated_trace_count=len(label_valid_traces),
        )

    if not traces:
        return TraceCoverageReport(
            status="rejected",
            required_count=required_count,
            actual_count=0,
            positive_count=0,
            negative_count=0,
            boundary_count=0,
            distinct_boundary_count=0,
            missing=missing or ("缺少全部有限轨迹",),
            coverage_debt=missing or ("缺少全部有限轨迹",),
            retry_count=retry_count,
            executable_semantics=executable_semantics,
            reason="没有任何有限轨迹证据，不能静默降级。",
            quota_complete=False,
            semantic_complete=False,
            covered_roles=role_coverage,
            missing_roles=missing_roles,
            covered_confusions=confusion_coverage,
            missing_confusions=missing_confusions,
            validated_trace_count=0,
        )

    if not missing:
        return TraceCoverageReport(
            status="complete",
            required_count=required_count,
            actual_count=len(traces),
            positive_count=positive_count,
            negative_count=negative_count,
            boundary_count=boundary_count,
            distinct_boundary_count=distinct_boundary_count,
            retry_count=retry_count,
            executable_semantics=executable_semantics,
            reason="有限轨迹配额和语义覆盖契约已满足。",
            quota_complete=True,
            semantic_complete=True,
            covered_roles=role_coverage,
            missing_roles=(),
            covered_confusions=confusion_coverage,
            missing_confusions=(),
            validated_trace_count=len(label_valid_traces),
        )

    status = "degraded_unproven" if allow_degraded else "incomplete"
    reason = (
        "多轮补充后仍缺少完整轨迹证据；保留现有无冲突证据继续推导，"
        "所有候选必须按降级结果严格审查。"
        if allow_degraded
        else "有限轨迹覆盖尚未满足，需要定向补充后重新评审。"
    )
    return TraceCoverageReport(
        status=status,
        required_count=required_count,
        actual_count=len(traces),
        positive_count=positive_count,
        negative_count=negative_count,
        boundary_count=boundary_count,
        distinct_boundary_count=distinct_boundary_count,
        missing=missing,
        coverage_debt=missing,
        retry_count=retry_count,
        executable_semantics=executable_semantics,
        reason=reason,
        quota_complete=quota_complete,
        semantic_complete=semantic_complete,
        covered_roles=role_coverage,
        missing_roles=missing_roles,
        covered_confusions=confusion_coverage,
        missing_confusions=missing_confusions,
        validated_trace_count=len(label_valid_traces),
    )


def _assess_adjudicated_query_coverage(
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    *,
    target_trace_count: int | None,
    retry_count: int,
    guided: bool,
    allow_degraded: bool,
    trace_plan: TraceObligationPlan | None = None,
) -> TraceCoverageReport:
    """Assess reviewed evidence using the current formula-free IR contract.

    ``guided`` and ``target_trace_count`` are retained only for call-site API
    compatibility.  They must not weaken the obligation plan or introduce an
    independent numeric quota.
    """

    del guided, target_trace_count
    ir_plan = trace_plan or (
        compile_executable_trace_plan(semantic.faithful_ir)
        if semantic.faithful_ir is not None
        else None
    )
    if ir_plan is None:
        raise ValueError(
            "adjudicated query coverage requires FaithfulSemanticIR"
        )
    assessed_traces = bind_traces_to_ir_plan(
        traces,
        ir_plan,
        generation_source="trace_coverage_assessment",
    )
    ir_coverage = evaluate_ir_coverage(
        semantic.faithful_ir,
        ir_plan,
        assessed_traces,
    )
    required_trace_obligations = tuple(
        obligation
        for obligation in ir_plan.obligations
        if obligation.evidence_mode
        not in {"diagnostic_only", "operator_resolution"}
    )
    required_obligation_ids = {
        obligation.obligation_id for obligation in required_trace_obligations
    }
    covered_obligation_ids = tuple(
        obligation_id
        for obligation_id in ir_coverage.covered_obligation_ids
        if obligation_id in required_obligation_ids
    )
    missing_obligation_ids = tuple(
        obligation.obligation_id
        for obligation in required_trace_obligations
        if obligation.obligation_id not in covered_obligation_ids
    )
    required_count = len(required_trace_obligations)
    positive_count = sum(1 for trace in traces if trace.expected)
    negative_count = sum(1 for trace in traces if not trace.expected)
    boundary_count = sum(1 for trace in traces if is_boundary_trace(trace))
    distinct_boundary_count = len(
        {
            _trace_key(trace.trace)
            for trace in traces
            if is_boundary_trace(trace)
        }
    )
    covered_query_roles = tuple(
        dict.fromkeys(
            role
            for trace in assessed_traces
            for role in _coverage_trace_roles(trace, include_semantic=False)
        )
    )
    covered_ambiguities = ir_coverage.covered_confusions
    missing_roles = tuple(
        dict.fromkeys(
            obligation.required_trace_role
            for obligation in required_trace_obligations
            if obligation.obligation_id in missing_obligation_ids
        )
    )
    missing_confusions = ir_coverage.missing_confusions
    missing: list[str] = []
    if missing_obligation_ids:
        obligation_map = {item.obligation_id: item for item in ir_plan.obligations}
        missing.extend(
            (
                "缺少 IR obligation: "
                f"{obligation_id} "
                f"(trace_role={obligation_map[obligation_id].required_trace_role}, "
                f"ir_nodes={','.join(obligation_map[obligation_id].ir_node_ids)})"
            )
            for obligation_id in missing_obligation_ids
        )
    missing.extend(f"缺少歧义轴覆盖: {axis}" for axis in missing_confusions)

    conflicts = _trace_label_conflicts(traces)
    if conflicts:
        return TraceCoverageReport(
            status="rejected",
            required_count=required_count,
            actual_count=len(traces),
            positive_count=positive_count,
            negative_count=negative_count,
            boundary_count=boundary_count,
            distinct_boundary_count=distinct_boundary_count,
            missing=conflicts,
            coverage_debt=conflicts,
            retry_count=retry_count,
            executable_semantics=False,
            reason="人工裁决中存在同形异标签，需复核后才能继续。",
            quota_complete=False,
            semantic_complete=False,
            covered_roles=covered_query_roles,
            missing_roles=missing_roles,
            covered_confusions=covered_ambiguities,
            missing_confusions=missing_confusions,
            validated_trace_count=len(traces),
        )

    quota_complete = not missing_obligation_ids
    semantic_complete = not missing and not conflicts
    if not missing:
        status = "complete"
        reason = "当前 IR TraceObligationPlan 的轨迹证据已完整覆盖。"
    else:
        status = "degraded_unproven" if allow_degraded and traces else "incomplete"
        reason = (
            "轨迹证据仍有覆盖债务；保留已裁决证据并输出歧义。"
            if allow_degraded and traces
            else "轨迹查询仍需补充或裁决。"
        )
    return TraceCoverageReport(
        status=status,
        required_count=required_count,
        actual_count=len(traces),
        positive_count=positive_count,
        negative_count=negative_count,
        boundary_count=boundary_count,
        distinct_boundary_count=distinct_boundary_count,
        missing=tuple(missing),
        coverage_debt=tuple(missing),
        retry_count=retry_count,
        executable_semantics=False,
        reason=reason,
        quota_complete=quota_complete,
        semantic_complete=semantic_complete,
        covered_roles=covered_query_roles,
        missing_roles=missing_roles,
        covered_confusions=covered_ambiguities,
        missing_confusions=missing_confusions,
        validated_trace_count=len(traces),
    )


def assess_authoritative_trace_coverage(
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    *,
    trace_session: AuthoritativeTraceSession,
    **kwargs,
) -> TraceCoverageReport:
    """Authoritative coverage entry point bound to one promoted semantic."""

    validate_authoritative_trace_session(trace_session, semantic)
    return assess_trace_coverage(traces, semantic, **kwargs)


def assess_candidate_discrimination_coverage(
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    **kwargs,
) -> TraceCoverageReport:
    """Non-authoritative coverage for unresolved candidate discrimination.

    Results from this entry point cannot authorize trace labels, formula
    compilation, or finalization.
    """

    ir = semantic.faithful_ir
    if ir is not None:
        from ir_compiler import compile_semantic_discrimination_plan

        kwargs["trace_plan"] = compile_semantic_discrimination_plan(ir)
    return assess_trace_coverage(traces, semantic, **kwargs)


def trace_coverage_feedback(report: TraceCoverageReport) -> str:
    """Build targeted regeneration feedback from a coverage report."""

    if not report.missing:
        return ""
    return (
        "当前有限轨迹未满足统一覆盖契约，请只补充缺失证据并避免重复轨迹：\n- "
        + "\n- ".join(report.missing)
    )


def _coverage_trace_roles(
    trace: LabeledTrace,
    *,
    include_semantic: bool,
) -> tuple[str, ...]:
    roles = [
        trace.trace_role,
        trace.query_role,
        trace.role,
    ]
    if include_semantic:
        roles.extend(trace.semantics.roles)
    return tuple(dict.fromkeys(role for role in roles if role))


def _can_return_degraded_trace_set(traces: list[LabeledTrace]) -> bool:
    return bool(traces) and not _trace_label_conflicts(traces)


def _trace_label_conflicts(traces: list[LabeledTrace]) -> list[str]:
    labels_by_shape: dict[tuple[tuple[str, ...], ...], bool] = {}
    conflicts: list[str] = []
    for trace in traces:
        key = _trace_key(trace.trace)
        previous = labels_by_shape.get(key)
        if previous is not None and previous != trace.expected:
            conflicts.append(
                f"轨迹 {trace.name} 与同形状轨迹的 expected 标签冲突"
            )
            continue
        labels_by_shape[key] = trace.expected
    return conflicts


def _has_executable_trace_semantics(semantic: SemanticStructure) -> bool:
    if semantic.formula_text or semantic.pattern == "response":
        return True
    if _learned_formula_fallback_semantic(semantic) is not None:
        return True
    pattern = semantic.pattern.strip().lower().replace("-", "_").replace(" ", "_")
    if pattern in {
        "precedence",
        "recurrence",
        "cardinality_constraint",
        "periodic",
    }:
        return True
    return _looks_like_mutual_exclusion(_semantic_pattern_evidence_text("", semantic))


def _validate_degenerate_response_trace_set(
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    errors: list[str],
) -> None:
    trigger = semantic.trigger
    if any(trace.kind == "negative" and not trace.expected for trace in traces):
        errors.append("trigger 与 response 相同时不存在 expected=False 的 response 反轨迹")
    if not any(
        trace.kind == "positive"
        and trace.expected
        and all(trigger not in step for step in trace.trace)
        for trace in traces
    ):
        errors.append("缺少 vacuous self-response 正轨迹")
    if not any(
        trace.kind == "positive"
        and trace.expected
        and any(trigger in step for step in trace.trace)
        for trace in traces
    ):
        errors.append("缺少 self-response 发生轨迹")
    if (
        any(len(trace.trace) > 1 for trace in traces)
        and not _has_late_trigger_positive(traces, trigger, trigger)
    ):
        errors.append("缺少 late-trigger self-response 正轨迹")


def _kind_count(traces: list[LabeledTrace], kind: str) -> int:
    return sum(1 for trace in traces if trace.kind == kind)


def _is_degenerate_response_pair(trigger: str, response: str) -> bool:
    return bool(trigger) and trigger == response


def _build_degenerate_response_traces(
    *,
    trigger: str,
    target_trace_length: int | None,
    target_trace_count: int | None,
) -> list[LabeledTrace]:
    """Build traces for G(A -> F A), where false response traces are impossible."""

    length = target_trace_length or 3
    if length < 1:
        raise ValueError("self-response 轨迹长度必须至少为 1。")

    last_index = length - 1
    late_index = 1 if length > 1 else 0
    initial_events = (0,)
    late_events = (late_index,)
    final_events = (last_index,)
    repeated_edge_events = tuple(dict.fromkeys((0, last_index)))
    repeated_adjacent_events = tuple(dict.fromkeys((0, min(1, last_index))))
    boundary_late_events = final_events if length > 1 else ()

    traces = [
        _degenerate_response_trace(
            name=f"pos_{trigger}_vacuous_no_trigger",
            kind="positive",
            trigger=trigger,
            length=length,
            positions=(),
            reason=(
                f"整条有限轨迹中没有 {trigger}，因此 G({trigger} -> F {trigger}) "
                "中的触发义务真空满足。"
            ),
        ),
        _degenerate_response_trace(
            name=f"pos_{trigger}_self_response_at_start",
            kind="positive",
            trigger=trigger,
            length=length,
            positions=initial_events,
            reason=(
                f"{trigger} 同时也是 response；F 允许当前时刻作为见证，"
                "所以首位置出现即满足自身响应。"
            ),
        ),
        _degenerate_response_trace(
            name=f"pos_late_{trigger}_self_response",
            kind="positive",
            trigger=trigger,
            length=length,
            positions=late_events,
            reason=(
                f"{trigger} 在 t0 之后出现，且该位置自身已经满足 F {trigger}。"
            ),
        ),
        _degenerate_response_trace(
            name=f"pos_final_{trigger}_self_response",
            kind="positive",
            trigger=trigger,
            length=length,
            positions=final_events,
            reason=(
                f"{trigger} 即使只在有限轨迹最后一个位置出现，也能由当前时刻满足 F {trigger}。"
            ),
        ),
        _degenerate_response_trace(
            name=f"bound_{trigger}_self_same_time_start",
            kind="boundary",
            trigger=trigger,
            length=length,
            positions=repeated_edge_events,
            reason=(
                f"边界样例：trigger 与 response 是同一个 AP {trigger}，"
                "同一时刻见证必须被视为满足。"
            ),
        ),
        _degenerate_response_trace(
            name=f"bound_late_{trigger}_self_same_time",
            kind="boundary",
            trigger=trigger,
            length=length,
            positions=boundary_late_events,
            reason=(
                f"第二条边界样例：用于记录 self-response 没有可构造的 false 反轨迹。"
            ),
        ),
    ]

    required_count = _minimum_required_trace_count(target_trace_count)
    if len(traces) >= required_count:
        return _deduplicate_trace_names(traces)

    seen_shapes = {_trace_key(trace.trace) for trace in traces}
    for positions in _single_ap_position_variants(length):
        if len(traces) >= required_count:
            break
        trace = _degenerate_response_trace(
            name=f"pos_{trigger}_self_response_variant_{len(traces) + 1}",
            kind="positive",
            trigger=trigger,
            length=length,
            positions=positions,
            reason=(
                f"补充 self-response 正轨迹；任一 {trigger} 出现位置都由自身满足 F {trigger}。"
            ),
        )
        shape = _trace_key(trace.trace)
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        traces.append(trace)

    while len(traces) < required_count:
        traces.append(
            _degenerate_response_trace(
                name=f"pos_{trigger}_self_response_extra_{len(traces) + 1}",
                kind="positive",
                trigger=trigger,
                length=length,
                positions=(),
                reason=(
                    f"补足目标轨迹数量；trigger=response={trigger} 时不存在 expected=False 反轨迹。"
                ),
            )
        )

    return _deduplicate_trace_names(traces)


def _degenerate_response_trace(
    *,
    name: str,
    kind: str,
    trigger: str,
    length: int,
    positions: tuple[int, ...],
    reason: str,
) -> LabeledTrace:
    trace = _empty_trace(length)
    for position in positions:
        trace[position].add(trigger)
    return LabeledTrace(
        name=name,
        kind=kind,
        trace=trace,
        expected=True,
        reason=reason,
    )


def _single_ap_position_variants(length: int) -> list[tuple[int, ...]]:
    variants: list[tuple[int, ...]] = []
    for mask in range(0, 2**length):
        positions = tuple(index for index in range(length) if mask & (1 << index))
        variants.append(positions)
    return sorted(variants, key=lambda item: (len(item), item))


def _ensure_required_late_trigger_traces(
    traces: list[LabeledTrace],
    trigger: str,
    response: str,
    target_trace_length: int | None,
) -> list[LabeledTrace]:
    """Guarantee LLM traces include non-t0 trigger coverage for response scope."""

    completed = list(traces)
    if not _has_late_trigger_positive(completed, trigger, response):
        completed.append(
            _build_late_trigger_positive_trace(trigger, response, target_trace_length)
        )
    if not _has_late_trigger_negative(completed, trigger, response):
        completed.append(
            _build_late_trigger_negative_trace(trigger, response, target_trace_length)
        )
    return _deduplicate_trace_names(completed)


def _canonicalize_default_response_traces(
    *,
    traces: list[LabeledTrace],
    trigger: str,
    response: str,
    target_trace_length: int | None,
) -> list[LabeledTrace]:
    """Reduce default response coverage to the 6 canonical trace slots."""

    fallback_traces = build_response_traces(
        trigger=trigger,
        response=response,
        target_length=target_trace_length,
        target_count=DEFAULT_RESPONSE_TRACE_COUNT,
    )
    fallback_by_name = {trace.name: trace for trace in fallback_traces}
    slots = [
        (
            f"pos_{trigger}_then_{response}",
            lambda trace: _is_general_response_positive(trace, trigger, response),
        ),
        (
            f"pos_late_{trigger}_then_{response}",
            lambda trace: _is_late_trigger_positive(trace, trigger, response),
        ),
        (
            f"neg_{trigger}_without_{response}",
            lambda trace: _is_general_response_negative(trace, trigger, response),
        ),
        (
            f"neg_late_{trigger}_without_{response}",
            lambda trace: _is_late_trigger_negative(trace, trigger, response),
        ),
        (
            f"bound_{trigger}_and_{response}_same_time",
            lambda trace: _is_initial_same_time_boundary(trace, trigger, response),
        ),
        (
            f"bound_late_{trigger}_and_{response}_same_time",
            lambda trace: _is_late_same_time_boundary(trace, trigger, response),
        ),
    ]

    selected: list[LabeledTrace] = []
    used_indexes: set[int] = set()
    for fallback_name, predicate in slots:
        matched = _first_matching_trace(traces, predicate, used_indexes)
        selected.append(matched or fallback_by_name[fallback_name])
    return _deduplicate_trace_names(selected)


def _first_matching_trace(
    traces: list[LabeledTrace],
    predicate,
    used_indexes: set[int],
) -> LabeledTrace | None:
    for index, trace in enumerate(traces):
        if index in used_indexes:
            continue
        if predicate(trace):
            used_indexes.add(index)
            return trace
    return None


def _is_general_response_positive(
    trace: LabeledTrace,
    trigger: str,
    response: str,
) -> bool:
    return (
        trace.kind == "positive"
        and trace.expected
        and bool(trace.trace)
        and trigger in trace.trace[0]
        and any(response in step for step in trace.trace)
    )


def _is_general_response_negative(
    trace: LabeledTrace,
    trigger: str,
    response: str,
) -> bool:
    return (
        trace.kind == "negative"
        and not trace.expected
        and bool(trace.trace)
        and trigger in trace.trace[0]
        and all(response not in step for step in trace.trace)
    )


def _is_late_trigger_positive(
    trace: LabeledTrace,
    trigger: str,
    response: str,
) -> bool:
    if trace.kind != "positive" or not trace.expected:
        return False
    return any(
        any(response in step for step in trace.trace[position:])
        for position in _late_trigger_positions(trace, trigger)
    )


def _is_late_trigger_negative(
    trace: LabeledTrace,
    trigger: str,
    response: str,
) -> bool:
    if trace.kind != "negative" or trace.expected:
        return False
    return any(
        all(response not in step for step in trace.trace[position:])
        for position in _late_trigger_positions(trace, trigger)
    )


def _is_initial_same_time_boundary(
    trace: LabeledTrace,
    trigger: str,
    response: str,
) -> bool:
    return (
        trace.kind == "boundary"
        and bool(trace.trace)
        and trigger in trace.trace[0]
        and response in trace.trace[0]
    )


def _is_late_same_time_boundary(
    trace: LabeledTrace,
    trigger: str,
    response: str,
) -> bool:
    return (
        trace.kind == "boundary"
        and any(
            position > 0 and trigger in step and response in step
            for position, step in enumerate(trace.trace)
        )
    )


def _build_formula_text_traces(
    *,
    semantic: SemanticStructure,
    target_trace_length: int | None,
    target_trace_count: int | None,
    require_positive_negative: bool = True,
) -> list[LabeledTrace]:
    """Generate deterministic traces by evaluating an LTLf formula_text."""

    formula = parse_learned_formula(semantic.formula_text)
    aps = tuple(dict.fromkeys(semantic.aps or collect_formula_atoms(formula)))
    if not aps:
        aps = tuple(
            item
            for item in (semantic.trigger, semantic.response)
            if item and item != "unknown"
        )
    if not aps:
        raise ValueError("未知模式 formula_text 未提供可枚举的 AP。")
    if len(aps) > 5:
        raise ValueError(
            "rules 模式最多支持为 formula_text 枚举 5 个 AP；"
            "请使用 --llm api 生成更复杂未知模式的有限轨迹。"
        )

    lengths = [target_trace_length] if target_trace_length is not None else [1, 2, 3]
    candidates: list[tuple[list[set[str]], bool]] = []
    for length in lengths:
        if length is None or length < 1:
            continue
        for trace in _enumerate_traces(aps, length):
            candidates.append((trace, check_trace(trace, formula)))

    if not candidates:
        raise ValueError("未能为未知模式 formula_text 枚举出有限轨迹。")

    selected: list[LabeledTrace] = []
    seen = set()
    positive_candidates = _best_distinct_formula_trace_candidates(
        candidates,
        expected=True,
        limit=MIN_REQUIRED_KIND_TRACE_COUNT,
    )
    negative_candidates = _best_distinct_formula_trace_candidates(
        candidates,
        expected=False,
        limit=MIN_REQUIRED_KIND_TRACE_COUNT,
    )

    for positive_index, positive in enumerate(positive_candidates, start=1):
        _append_formula_trace_if_new(
            selected=selected,
            seen=seen,
            name=f"formula_positive_{positive_index}",
            kind="positive",
            trace=positive[0],
            expected=positive[1],
            semantic=semantic,
        )
    for negative_index, negative in enumerate(negative_candidates, start=1):
        _append_formula_trace_if_new(
            selected=selected,
            seen=seen,
            name=f"formula_negative_{negative_index}",
            kind="negative",
            trace=negative[0],
            expected=negative[1],
            semantic=semantic,
        )

    boundary_pool = [
        item
        for item in candidates
        if len(item[0]) == (target_trace_length or 1)
    ] or candidates
    boundary_candidates = _best_distinct_formula_trace_candidates(
        boundary_pool or candidates,
        expected=None,
        limit=MIN_DISTINCT_BOUNDARY_TRACE_COUNT,
    )
    boundary_seen: set[tuple[tuple[str, ...], ...]] = set()
    for boundary_index, boundary in enumerate(boundary_candidates, start=1):
        _append_formula_boundary_trace_if_new(
            selected=selected,
            boundary_seen=boundary_seen,
            seen=seen,
            name=f"formula_boundary_{boundary_index}",
            trace=boundary[0],
            expected=boundary[1],
            semantic=semantic,
        )

    target_count = _minimum_required_trace_count(target_trace_count or len(selected))
    selected = _top_up_formula_trace_set_with_duplicates(
        selected=selected,
        candidates=candidates,
        semantic=semantic,
        target_count=target_count,
        fill_to_target=False,
    )

    extra_index = 1
    for trace, expected in candidates:
        if len(selected) >= target_count:
            break
        key = _trace_key(trace)
        if key in seen:
            continue
        seen.add(key)
        extra_index += 1
        selected.append(
            _formula_trace(
                name=f"formula_extra_{extra_index}",
                kind="positive" if expected else "negative",
                trace=trace,
                expected=expected,
                semantic=semantic,
            )
        )

    selected = _top_up_formula_trace_set_with_duplicates(
        selected=selected,
        candidates=candidates,
        semantic=semantic,
        target_count=target_count,
        fill_to_target=True,
    )

    kinds = {trace.kind for trace in selected}
    if require_positive_negative and (
        _kind_count(selected, "positive") < MIN_REQUIRED_KIND_TRACE_COUNT
        or _kind_count(selected, "negative") < MIN_REQUIRED_KIND_TRACE_COUNT
    ):
        missing = []
        if _kind_count(selected, "positive") < MIN_REQUIRED_KIND_TRACE_COUNT:
            missing.append("positive")
        if _kind_count(selected, "negative") < MIN_REQUIRED_KIND_TRACE_COUNT:
            missing.append("negative")
        missing.extend(sorted(ALLOWED_TRACE_KINDS - kinds))
        raise ValueError(
            "未知模式 formula_text 在当前枚举范围内缺少轨迹类型: "
            + ", ".join(missing)
        )
    return _deduplicate_trace_names(selected)


def _append_formula_trace_if_new(
    *,
    selected: list[LabeledTrace],
    seen: set[tuple[tuple[str, ...], ...]],
    name: str,
    kind: str,
    trace: list[set[str]],
    expected: bool,
    semantic: SemanticStructure,
) -> None:
    key = _trace_key(trace)
    if key in seen:
        return
    seen.add(key)
    selected.append(
        _formula_trace(
            name=name,
            kind=kind,
            trace=trace,
            expected=expected,
            semantic=semantic,
        )
    )


def _append_formula_boundary_trace_if_new(
    *,
    selected: list[LabeledTrace],
    boundary_seen: set[tuple[tuple[str, ...], ...]],
    seen: set[tuple[tuple[str, ...], ...]],
    name: str,
    trace: list[set[str]],
    expected: bool,
    semantic: SemanticStructure,
) -> None:
    key = _trace_key(trace)
    if key in boundary_seen:
        return
    boundary_seen.add(key)
    seen.add(key)
    selected.append(
        _formula_trace(
            name=name,
            kind="boundary",
            trace=trace,
            expected=expected,
            semantic=semantic,
        )
    )


def _top_up_formula_trace_set_with_duplicates(
    *,
    selected: list[LabeledTrace],
    candidates: list[tuple[list[set[str]], bool]],
    semantic: SemanticStructure,
    target_count: int,
    fill_to_target: bool,
) -> list[LabeledTrace]:
    completed = list(selected)

    def append_duplicate(kind: str, expected: bool | None, reason_kind: str) -> bool:
        candidate = _best_formula_trace_candidate(candidates, expected=expected)
        if candidate is None:
            return False
        completed.append(
            _formula_trace(
                name=f"formula_{reason_kind}_duplicate_{len(completed) + 1}",
                kind=kind,
                trace=candidate[0],
                expected=candidate[1],
                semantic=semantic,
            )
        )
        return True

    while _kind_count(completed, "positive") < MIN_REQUIRED_KIND_TRACE_COUNT:
        if not append_duplicate("positive", True, "positive"):
            break

    while _kind_count(completed, "negative") < MIN_REQUIRED_KIND_TRACE_COUNT:
        if not append_duplicate("negative", False, "negative"):
            break

    while fill_to_target and len(completed) < target_count:
        if not append_duplicate("positive", True, "extra"):
            if not append_duplicate("negative", False, "extra"):
                break

    return _deduplicate_trace_names(completed)


def _ensure_formula_text_required_trace_kinds(
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    target_trace_length: int | None,
    target_trace_count: int | None,
    missing: set[str],
) -> list[LabeledTrace]:
    """Fill missing learned-pattern trace categories by evaluating formula_text."""

    if not semantic.formula_text:
        raise ValueError(f"LLM 轨迹缺少类型: {', '.join(sorted(missing))}")

    fallback_traces = _build_formula_text_traces(
        semantic=semantic,
        target_trace_length=target_trace_length,
        target_trace_count=target_trace_count,
        require_positive_negative=False,
    )
    completed = list(traces)
    for required_kind in sorted(missing):
        replacement = next(
            (trace for trace in fallback_traces if trace.kind == required_kind),
            None,
        )
        if replacement is not None:
            completed.append(replacement)

    still_missing = ALLOWED_TRACE_KINDS - {trace.kind for trace in completed}
    if still_missing:
        raise ValueError(f"LLM 轨迹缺少类型: {', '.join(sorted(still_missing))}")
    return _deduplicate_trace_names(completed)


def _enumerate_traces(
    aps: tuple[str, ...],
    length: int,
) -> list[list[set[str]]]:
    step_options = _step_options(aps)
    return [
        [set(step) for step in raw_trace]
        for raw_trace in product(step_options, repeat=length)
    ]


def _step_options(aps: tuple[str, ...]) -> list[tuple[str, ...]]:
    options: list[tuple[str, ...]] = [()]
    for ap in aps:
        options.extend((*option, ap) for option in list(options))
    return [tuple(dict.fromkeys(option)) for option in options]


def _formula_trace(
    *,
    name: str,
    kind: str,
    trace: list[set[str]],
    expected: bool,
    semantic: SemanticStructure,
) -> LabeledTrace:
    return LabeledTrace(
        name=name,
        kind=kind,
        trace=trace,
        expected=expected,
        reason=(
            f"根据未知模式 {semantic.original_pattern or semantic.pattern} "
            f"的 LLM formula_text={semantic.formula_text} 直接求值。"
        ),
    )


def _best_formula_trace_candidate(
    candidates: list[tuple[list[set[str]], bool]],
    *,
    expected: bool | None,
) -> tuple[list[set[str]], bool] | None:
    pool = [
        item
        for item in candidates
        if expected is None or item[1] is expected
    ]
    if not pool:
        return None
    return sorted(
        pool,
        key=lambda item: (-_trace_ap_count(item[0]), len(item[0]), _trace_key(item[0])),
    )[0]


def _best_distinct_formula_trace_candidates(
    candidates: list[tuple[list[set[str]], bool]],
    *,
    expected: bool | None,
    limit: int,
) -> list[tuple[list[set[str]], bool]]:
    pool = [
        item
        for item in candidates
        if expected is None or item[1] is expected
    ]
    ordered = sorted(
        pool,
        key=lambda item: (-_trace_ap_count(item[0]), len(item[0]), _trace_key(item[0])),
    )

    selected: list[tuple[list[set[str]], bool]] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    for item in ordered:
        key = _trace_key(item[0])
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _trace_ap_count(trace: list[set[str]]) -> int:
    atoms: set[str] = set()
    for step in trace:
        atoms.update(step)
    return len(atoms)


def _trace_key(trace: list[set[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted(step)) for step in trace)


def _ensure_response_required_trace_kinds(
    traces: list[LabeledTrace],
    trigger: str,
    response: str,
    target_trace_length: int | None,
    missing: set[str],
) -> list[LabeledTrace]:
    """Fill missing response trace categories with deterministic local examples."""

    fallback_traces = build_response_traces(
        trigger=trigger,
        response=response,
        target_length=target_trace_length,
    )
    completed = list(traces)
    for required_kind in sorted(missing):
        replacement = next(
            (trace for trace in fallback_traces if trace.kind == required_kind),
            None,
        )
        if replacement is not None:
            completed.append(replacement)

    still_missing = ALLOWED_TRACE_KINDS - {trace.kind for trace in completed}
    if still_missing:
        raise ValueError(f"LLM 轨迹缺少类型: {', '.join(sorted(still_missing))}")
    return _deduplicate_trace_names(completed)


def _ensure_response_trace_count(
    traces: list[LabeledTrace],
    trigger: str,
    response: str,
    target_trace_length: int | None,
    target_trace_count: int | None,
) -> list[LabeledTrace]:
    """Top up response traces with deterministic examples when LLM returns too few."""

    if target_trace_count is None or len(traces) >= target_trace_count:
        return traces

    fallback_traces = build_response_traces(
        trigger=trigger,
        response=response,
        target_length=target_trace_length,
        target_count=target_trace_count,
    )
    completed = list(traces)
    seen_names = {trace.name for trace in completed}
    for trace in fallback_traces:
        if len(completed) >= target_trace_count:
            break
        if trace.name in seen_names:
            continue
        seen_names.add(trace.name)
        completed.append(trace)
    return _deduplicate_trace_names(completed)


def _ensure_min_distinct_boundary_traces(
    *,
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
    target_trace_length: int | None,
    target_trace_count: int | None,
) -> list[LabeledTrace]:
    """Guarantee at least two boundary traces with different finite traces."""

    completed = _deduplicate_boundary_trace_shapes(traces)
    boundary_keys = _boundary_trace_keys(completed)
    if len(boundary_keys) >= MIN_DISTINCT_BOUNDARY_TRACE_COUNT:
        return _deduplicate_trace_names(completed)

    for trace in _fallback_boundary_traces(
        semantic=semantic,
        target_trace_length=target_trace_length,
        target_trace_count=target_trace_count,
    ):
        if trace.kind != "boundary":
            continue
        key = _trace_key(trace.trace)
        if key in boundary_keys:
            continue
        completed.append(trace)
        boundary_keys.add(key)
        if len(boundary_keys) >= MIN_DISTINCT_BOUNDARY_TRACE_COUNT:
            break

    if len(boundary_keys) < MIN_DISTINCT_BOUNDARY_TRACE_COUNT:
        for trace in _boundary_traces_from_existing_labels(
            completed,
            excluded_keys=boundary_keys,
        ):
            completed.append(trace)
            boundary_keys.add(_trace_key(trace.trace))
            if len(boundary_keys) >= MIN_DISTINCT_BOUNDARY_TRACE_COUNT:
                break

    if len(boundary_keys) < MIN_DISTINCT_BOUNDARY_TRACE_COUNT:
        raise ValueError("边界轨迹至少需要 2 条不同有限轨迹。")
    return _deduplicate_trace_names(completed)


def _boundary_traces_from_existing_labels(
    traces: list[LabeledTrace],
    *,
    excluded_keys: set[tuple[tuple[str, ...], ...]],
) -> list[LabeledTrace]:
    """Reuse known labels when a pattern has no deterministic boundary builder."""

    fallback: list[LabeledTrace] = []
    seen_keys = set(excluded_keys)
    for trace in traces:
        if trace.kind == "boundary":
            continue
        key = _trace_key(trace.trace)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        fallback.append(
            replace(
                trace,
                name=f"bound_distinct_fallback_{len(fallback) + 1}_{trace.name}",
                kind="boundary",
                reason=(
                    "模式专用边界兜底不足；复用已有已标注有限轨迹及其 expected "
                    f"真值作为不同形状的边界样例。原轨迹：{trace.name}。"
                ),
            )
        )
    return fallback


def _deduplicate_boundary_trace_shapes(traces: list[LabeledTrace]) -> list[LabeledTrace]:
    seen_boundary_keys: set[tuple[tuple[str, ...], ...]] = set()
    unique: list[LabeledTrace] = []
    for trace in traces:
        if trace.kind == "boundary":
            key = _trace_key(trace.trace)
            if key in seen_boundary_keys:
                continue
            seen_boundary_keys.add(key)
        unique.append(trace)
    return unique


def _boundary_trace_keys(
    traces: list[LabeledTrace],
) -> set[tuple[tuple[str, ...], ...]]:
    return {
        _trace_key(trace.trace)
        for trace in traces
        if trace.kind == "boundary"
    }


def _fallback_boundary_traces(
    *,
    semantic: SemanticStructure,
    target_trace_length: int | None,
    target_trace_count: int | None,
) -> list[LabeledTrace]:
    if semantic.formula_text:
        return _build_formula_text_traces(
            semantic=semantic,
            target_trace_length=target_trace_length,
            target_trace_count=max(
                target_trace_count or 0,
                MIN_REQUIRED_TRACE_COUNT,
            ),
            require_positive_negative=False,
        )

    learned_formula_semantic = _learned_formula_fallback_semantic(semantic)
    if learned_formula_semantic is not None:
        return _build_formula_text_traces(
            semantic=learned_formula_semantic,
            target_trace_length=target_trace_length,
            target_trace_count=max(
                target_trace_count or 0,
                MIN_REQUIRED_TRACE_COUNT,
            ),
            require_positive_negative=False,
        )

    if semantic.pattern == "response":
        return build_response_traces(
            trigger=semantic.trigger,
            response=semantic.response,
            target_length=target_trace_length,
            target_count=max(
                target_trace_count or 0,
                DEFAULT_RESPONSE_TRACE_COUNT,
            ),
        )

    return _build_semantic_pattern_traces(
        requirement="",
        semantic=semantic,
        target_trace_length=target_trace_length,
        target_trace_count=max(
            target_trace_count or 0,
            MIN_REQUIRED_TRACE_COUNT,
        ),
    )


def _build_semantic_pattern_traces(
    *,
    requirement: str,
    semantic: SemanticStructure,
    target_trace_length: int | None,
    target_trace_count: int | None,
) -> list[LabeledTrace]:
    evidence = _semantic_pattern_evidence_text(requirement, semantic)
    pattern = semantic.pattern.strip().lower().replace("-", "_").replace(" ", "_")
    if pattern == "precedence":
        return _build_precedence_traces(
            semantic=semantic,
            target_trace_length=target_trace_length,
            target_trace_count=target_trace_count,
        )
    if pattern == "recurrence":
        return _build_recurrence_traces(
            semantic=semantic,
            evidence=evidence,
            target_trace_length=target_trace_length,
            target_trace_count=target_trace_count,
        )
    if pattern == "cardinality_constraint":
        return _build_cardinality_traces(
            semantic=semantic,
            evidence=evidence,
            target_trace_length=target_trace_length,
            target_trace_count=target_trace_count,
        )
    if pattern in {"periodic", "periodicity"}:
        return _build_periodic_traces(
            semantic=semantic,
            evidence=evidence,
            target_trace_length=target_trace_length,
            target_trace_count=target_trace_count,
        )
    if not _looks_like_mutual_exclusion(evidence):
        return []

    aps = _semantic_binary_aps(semantic)
    if len(aps) < 2:
        return []
    left, right = aps[0], aps[1]
    coverage_required = _looks_like_coverage(evidence)
    length = max(2, target_trace_length or 3)

    def trace_with(events: tuple[tuple[int, tuple[str, ...]], ...]) -> list[set[str]]:
        trace = [set() for _ in range(length)]
        for raw_index, step_events in events:
            index = min(max(raw_index, 0), length - 1)
            trace[index].update(step_events)
        return trace

    def expected(trace: list[set[str]]) -> bool:
        no_overlap = all(not ({left, right} <= step) for step in trace)
        if not coverage_required:
            return no_overlap
        covered = all(left in step or right in step for step in trace)
        return no_overlap and covered

    positive_shapes = (
        trace_with(((0, (left,)), (length - 1, (right,)))),
        trace_with(((0, (right,)), (length - 1, (left,)))),
    )
    if coverage_required:
        positive_shapes = (
            [({left} if index % 2 == 0 else {right}) for index in range(length)],
            [{left} for _index in range(length)],
        )

    negative_shapes = (
        trace_with(((0, (left, right)),)),
        trace_with(((1, (left, right)),)),
    )
    if coverage_required:
        negative_shapes = (
            trace_with(((0, (left, right)),)),
            trace_with(()),
        )

    boundary_shapes = (
        trace_with(()),
        trace_with(((0, (left,)),)),
    )
    if coverage_required:
        boundary_shapes = (
            [{right} for _index in range(length)],
            trace_with(((0, (left,)), (1, (left, right)))),
        )

    traces: list[LabeledTrace] = []
    for index, trace in enumerate(positive_shapes, start=1):
        traces.append(
            LabeledTrace(
                name=f"pos_mutual_exclusion_{index}",
                kind="positive",
                trace=trace,
                expected=expected(trace),
                reason=(
                    f"由语义模式 {semantic.pattern} 推导：{left} 和 {right} "
                    "没有在同一时间步同时成立。"
                ),
            )
        )
    for index, trace in enumerate(negative_shapes, start=1):
        traces.append(
            LabeledTrace(
                name=f"neg_mutual_exclusion_{index}",
                kind="negative",
                trace=trace,
                expected=expected(trace),
                reason=(
                    f"由语义模式 {semantic.pattern} 推导：该轨迹违反 "
                    f"{left}/{right} 互斥或覆盖约束。"
                ),
            )
        )
    for index, trace in enumerate(boundary_shapes, start=1):
        traces.append(
            LabeledTrace(
                name=f"bound_mutual_exclusion_{index}",
                kind="boundary",
                trace=trace,
                expected=expected(trace),
                reason=(
                    f"由语义模式 {semantic.pattern} 推导的边界轨迹，用于区分"
                    "空步、单 AP 成立和同时成立。"
                ),
            )
        )

    target_count = _minimum_required_trace_count(target_trace_count)
    return traces[:target_count]


def _build_precedence_traces(
    *,
    semantic: SemanticStructure,
    target_trace_length: int | None,
    target_trace_count: int | None,
) -> list[LabeledTrace]:
    constrained, prerequisite = precedence_roles_from_semantic(semantic)
    if not constrained or not prerequisite or constrained == prerequisite:
        return []
    length = _pattern_trace_length(
        target_trace_length=target_trace_length,
        default_length=3,
    )

    def trace_with(events: tuple[tuple[int, tuple[str, ...]], ...]) -> list[set[str]]:
        return _trace_with_events(length, events)

    def expected(trace: list[set[str]]) -> bool:
        seen_prerequisite = False
        for step in trace:
            if constrained in step and not seen_prerequisite:
                return False
            if prerequisite in step:
                seen_prerequisite = True
        return True

    shapes = (
        (
            "pos_prior_then_constrained",
            "positive",
            trace_with(((0, (prerequisite,)), (min(1, length - 1), (constrained,)))),
            f"{prerequisite} 严格早于 {constrained}，满足 precedence。",
        ),
        (
            "pos_no_constrained_vacuous",
            "positive",
            trace_with(((0, (prerequisite,)),)),
            f"{constrained} 从未发生，因此 precedence 约束真值为 vacuous true。",
        ),
        (
            "neg_constrained_before_prior",
            "negative",
            trace_with(((0, (constrained,)), (min(1, length - 1), (prerequisite,)))),
            f"{constrained} 在任何 prior {prerequisite} 之前发生，违反严格 precedence。",
        ),
        (
            "neg_same_step_not_strict",
            "negative",
            trace_with(((0, (prerequisite, constrained)),)),
            f"{prerequisite} 与 {constrained} 同步发生，不能作为严格 prior。",
        ),
        (
            "bound_prior_at_start_constrained_at_end",
            "boundary",
            trace_with(((0, (prerequisite,)), (length - 1, (constrained,)))),
            "边界轨迹：先决事件在起点，受限事件在末尾。",
        ),
        (
            "bound_constrained_at_start",
            "boundary",
            trace_with(((0, (constrained,)),)),
            "边界轨迹：受限事件在首步发生，有限轨迹中不存在更早位置。",
        ),
    )
    return _pattern_labeled_traces(
        semantic=semantic,
        shapes=shapes,
        expected_fn=expected,
        target_trace_count=target_trace_count,
    )


def _build_recurrence_traces(
    *,
    semantic: SemanticStructure,
    evidence: str,
    target_trace_length: int | None,
    target_trace_count: int | None,
) -> list[LabeledTrace]:
    ap = _primary_semantic_ap(semantic)
    if not ap:
        return []
    period = _period_from_text(evidence, default=5)
    length = _pattern_trace_length(
        target_trace_length=target_trace_length,
        default_length=period + 1,
    )
    required_positions = tuple(range(0, length, period))

    def required_trace(*, omit: int | None = None, extra: int | None = None) -> list[set[str]]:
        trace = [set() for _index in range(length)]
        for position in required_positions:
            if position != omit:
                trace[position].add(ap)
        if extra is not None and 0 <= extra < length:
            trace[extra].add(ap)
        return trace

    def expected(trace: list[set[str]]) -> bool:
        return all(ap in trace[position] for position in range(0, len(trace), period))

    first_non_required = next(
        (index for index in range(length) if index not in required_positions),
        0,
    )
    period_position = period if period < length else 0
    shapes = (
        (
            "pos_required_steps_hold",
            "positive",
            required_trace(),
            f"{ap} 在每个第 {period} 步周期位置成立。",
        ),
        (
            "pos_required_steps_with_extra_occurrence",
            "positive",
            required_trace(extra=first_non_required),
            f"{ap} 在所有要求位置成立；非要求位置额外成立不破坏 recurrence。",
        ),
        (
            "neg_missing_initial_required_step",
            "negative",
            required_trace(omit=0),
            f"{ap} 缺失于初始周期位置，违反 recurrence。",
        ),
        (
            "neg_missing_later_required_step",
            "negative",
            required_trace(omit=period_position),
            f"{ap} 缺失于后续周期位置，违反 recurrence。",
        ),
        (
            "bound_exact_period_boundary",
            "boundary",
            required_trace(omit=period_position),
            "边界轨迹：刚到达下一个周期检查点但 AP 缺失。",
        ),
        (
            "bound_short_prefix_requirement",
            "boundary",
            required_trace(),
            "边界轨迹：有限前缀只检查实际存在的周期位置。",
        ),
    )
    return _pattern_labeled_traces(
        semantic=semantic,
        shapes=shapes,
        expected_fn=expected,
        target_trace_count=target_trace_count,
    )


def _build_cardinality_traces(
    *,
    semantic: SemanticStructure,
    evidence: str,
    target_trace_length: int | None,
    target_trace_count: int | None,
) -> list[LabeledTrace]:
    ap = _primary_semantic_ap(semantic)
    if not ap:
        return []
    limit = _cardinality_limit_from_text(evidence, default=2)
    count_absence = _cardinality_counts_absence(evidence, ap)
    length = _pattern_trace_length(
        target_trace_length=target_trace_length,
        default_length=max(3, limit + 2),
    )
    if length <= limit:
        raise ValueError(
            f"cardinality_constraint 需要至少 {limit + 1} 个时间步生成反例。"
        )

    def trace_with_count(count: int, *, offset: int = 0) -> list[set[str]]:
        trace = [set() for _index in range(length)]
        condition_positions = {
            (offset + index) % length for index in range(min(count, length))
        }
        for position in range(length):
            condition_holds = position in condition_positions
            ap_holds = not condition_holds if count_absence else condition_holds
            if ap_holds:
                trace[position].add(ap)
        return trace

    def condition_count(trace: list[set[str]]) -> int:
        if count_absence:
            return sum(1 for step in trace if ap not in step)
        return sum(1 for step in trace if ap in step)

    def expected(trace: list[set[str]]) -> bool:
        return condition_count(trace) <= limit

    condition_text = f"!{ap}" if count_absence else ap
    shapes = (
        (
            "pos_below_cardinality_limit",
            "positive",
            trace_with_count(max(0, limit - 1)),
            f"{condition_text} 出现次数低于 at-most-{limit} 上界。",
        ),
        (
            "pos_at_cardinality_limit",
            "positive",
            trace_with_count(limit),
            f"{condition_text} 出现次数正好等于 at-most-{limit} 上界。",
        ),
        (
            "neg_exceeds_cardinality_limit",
            "negative",
            trace_with_count(limit + 1),
            f"{condition_text} 出现 {limit + 1} 次，超过 cardinality 上界。",
        ),
        (
            "neg_many_violations",
            "negative",
            trace_with_count(min(length, limit + 2)),
            f"{condition_text} 出现次数明显超过 at-most-{limit} 上界。",
        ),
        (
            "bound_exact_limit_shifted",
            "boundary",
            trace_with_count(limit, offset=1),
            "边界轨迹：计数刚好达到上界但不超过。",
        ),
        (
            "bound_one_over_limit_shifted",
            "boundary",
            trace_with_count(limit + 1, offset=1),
            "边界轨迹：只比上界多一次，应该被拒绝。",
        ),
    )
    return _pattern_labeled_traces(
        semantic=semantic,
        shapes=shapes,
        expected_fn=expected,
        target_trace_count=target_trace_count,
    )


def _build_periodic_traces(
    *,
    semantic: SemanticStructure,
    evidence: str,
    target_trace_length: int | None,
    target_trace_count: int | None,
) -> list[LabeledTrace]:
    ap = _primary_semantic_ap(semantic)
    ir = semantic.faithful_ir
    if not ap:
        return []
    if ir is None:
        return _build_legacy_periodic_spacing_traces(
            semantic=semantic,
            ap=ap,
            evidence=evidence,
            target_trace_length=target_trace_length,
            target_trace_count=target_trace_count,
        )
    root = ir.node_map.get(ir.root_node_id)
    periodic_node = root if root and root.node_type == "periodicity" else next(
        (node for node in ir.nodes if node.node_type == "periodicity"),
        None,
    )
    if periodic_node is None:
        return []
    params = dict(periodic_node.parameters)
    required = set(PERIODICITY_REQUIRED_PARAMETERS)
    if not required <= set(params):
        return []
    period = max(1, int(params["period"]))
    mode = str(params["mode"])
    phase = max(0, int(params.get("phase", 1)))
    length = _pattern_trace_length(
        target_trace_length=target_trace_length,
        default_length=max(period + 1, phase + period + 1),
    )
    if length < 2:
        raise ValueError("periodic 模式至少需要 2 个时间步生成判别轨迹。")

    def trace_with_positions(positions: tuple[int, ...]) -> list[set[str]]:
        trace = [set() for _index in range(length)]
        for position in positions:
            if 0 <= position < length:
                trace[position].add(ap)
        return trace

    def expected(trace: list[set[str]]) -> bool:
        try:
            return evaluate_ir_trace(ir, trace)
        except IRTraceSemanticsError as exc:
            raise ValueError(f"periodicity IR is not executable: {exc}") from exc

    scheduled = tuple(range(phase, length, period))
    satisfying = trace_with_positions(scheduled)
    missing = trace_with_positions(
        scheduled[:1]
        if mode == "event_relative_exact_recurrence"
        else scheduled[1:] if scheduled else ()
    )
    off_phase_index = next(
        (index for index in range(length) if index not in scheduled),
        0,
    )
    with_off_phase = trace_with_positions(tuple(dict.fromkeys((*scheduled, off_phase_index))))
    short_length = max(1, min(phase, length - 1))
    short_trace = [set() for _index in range(short_length)]
    shapes = (
        (
            "pos_periodic_schedule",
            "positive",
            satisfying,
            f"{ap} follows the reviewed phase={phase}, period={period} schedule.",
        ),
        (
            "neg_missing_scheduled_occurrence",
            "negative",
            missing,
            "A required scheduled occurrence is missing.",
        ),
        (
            "neg_off_phase_occurrence",
            "negative",
            with_off_phase,
            "An occurrence is placed outside the reviewed periodic phase.",
        ),
        (
            "bound_before_first_phase",
            "boundary",
            short_trace,
            "Boundary trace ends before the first reviewed phase.",
        ),
        (
            "bound_exact_schedule",
            "boundary",
            satisfying,
            "Boundary occurrences are exactly one period apart.",
        ),
        (
            "bound_one_off_phase",
            "boundary",
            with_off_phase,
            "Boundary occurrence is one schedule class outside the reviewed phase.",
        ),
    )
    return _pattern_labeled_traces(
        semantic=semantic,
        shapes=shapes,
        expected_fn=expected,
        target_trace_count=target_trace_count,
    )


def _build_legacy_periodic_spacing_traces(
    *,
    semantic: SemanticStructure,
    ap: str,
    evidence: str,
    target_trace_length: int | None,
    target_trace_count: int | None,
) -> list[LabeledTrace]:
    """Read-only compatibility behavior for pre-FaithfulIR callers."""

    period = _period_from_text(evidence, default=3)
    length = _pattern_trace_length(
        target_trace_length=target_trace_length,
        default_length=period + 1,
    )

    def trace_with_positions(positions: tuple[int, ...]) -> list[set[str]]:
        trace = [set() for _index in range(length)]
        for position in positions:
            if 0 <= position < length:
                trace[position].add(ap)
        return trace

    def expected(trace: list[set[str]]) -> bool:
        positions = [index for index, step in enumerate(trace) if ap in step]
        return all(
            right - left >= period
            for left, right in zip(positions, positions[1:])
        )

    near = min(max(period - 1, 1), length - 1)
    exact = min(period, length - 1)
    shapes = (
        ("pos_periodic_spacing", "positive", trace_with_positions((0, exact)), "exact spacing"),
        ("pos_single_occurrence", "positive", trace_with_positions((min(1, length - 1),)), "single occurrence"),
        ("neg_adjacent_occurrences", "negative", trace_with_positions((0, 1)), "adjacent occurrences"),
        ("neg_too_close_occurrences", "negative", trace_with_positions((0, near)), "too-close occurrences"),
        ("bound_exact_period_gap", "boundary", trace_with_positions((0, exact)), "exact-period boundary"),
        ("bound_one_before_period_gap", "boundary", trace_with_positions((0, near)), "one-before-period boundary"),
    )
    return _pattern_labeled_traces(
        semantic=semantic,
        shapes=shapes,
        expected_fn=expected,
        target_trace_count=target_trace_count,
    )


def _pattern_labeled_traces(
    *,
    semantic: SemanticStructure,
    shapes: tuple[tuple[str, str, list[set[str]], str], ...],
    expected_fn,
    target_trace_count: int | None,
) -> list[LabeledTrace]:
    traces = [
        LabeledTrace(
            name=f"{name}_{semantic.pattern}",
            kind=("boundary" if kind == "boundary" else "positive" if expected_fn(trace) else "negative"),
            trace=trace,
            expected=expected_fn(trace),
            reason=reason,
        )
        for name, kind, trace, reason in shapes
    ]
    target_count = _minimum_required_trace_count(target_trace_count)
    return traces[:target_count]


def _trace_with_events(
    length: int,
    events: tuple[tuple[int, tuple[str, ...]], ...],
) -> list[set[str]]:
    trace = [set() for _index in range(length)]
    for raw_index, step_events in events:
        index = min(max(raw_index, 0), length - 1)
        trace[index].update(step_events)
    return trace


def _pattern_trace_length(
    *,
    target_trace_length: int | None,
    default_length: int,
) -> int:
    if target_trace_length is not None:
        if target_trace_length < 1:
            raise ValueError("有限轨迹长度至少为 1。")
        return target_trace_length
    return max(1, default_length)


def _primary_semantic_ap(semantic: SemanticStructure) -> str:
    for ap in (*semantic.aps, semantic.trigger, semantic.response):
        if ap and ap not in {"unknown", "unknown_ap"}:
            return ap
    return ""


def _period_from_text(text: str, *, default: int) -> int:
    ordinal_numbers = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    cardinal_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    patterns = (
        r"every[_\s-]+(\d+)",
        r"every[_\s-]+([A-Za-z]+)",
        r"(\d+)(?:st|nd|rd|th)?[_\s-]+step",
        r"([A-Za-z]+)[_\s-]+step",
        r"period[_\s-]*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1).lower()
        if raw.isdigit():
            return max(1, int(raw))
        if raw in ordinal_numbers:
            return ordinal_numbers[raw]
        if raw in cardinal_numbers:
            return cardinal_numbers[raw]
    return default


def _cardinality_limit_from_text(text: str, *, default: int) -> int:
    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    match = re.search(
        r"(?:at\s+most|no\s+more\s+than|<=)\s+(\d+|[A-Za-z]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return default
    raw = match.group(1).lower()
    if raw.isdigit():
        return max(0, int(raw))
    return number_words.get(raw, default)


def _cardinality_counts_absence(text: str, ap: str) -> bool:
    escaped = re.escape(ap)
    return bool(
        re.search(rf"\bnot\s+{escaped}\b", text)
        or re.search(rf"!\s*{escaped}\b", text)
    )


def _semantic_binary_aps(semantic: SemanticStructure) -> tuple[str, ...]:
    aps: list[str] = []
    for ap in (*semantic.aps, semantic.trigger, semantic.response):
        if ap and ap not in {"unknown", "unknown_ap"} and ap not in aps:
            aps.append(ap)
    return tuple(aps)


def _semantic_pattern_evidence_text(
    requirement: str,
    semantic: SemanticStructure,
) -> str:
    component_text = " ".join(
        f"{key} {value}" for key, value in semantic.components
    )
    return " ".join(
        [
            requirement,
            semantic.pattern,
            semantic.scope_keyword,
            semantic.temporal_keyword,
            semantic.explanation,
            component_text,
        ]
    ).lower()


def _looks_like_mutual_exclusion(text: str) -> bool:
    return (
        "mutual_exclusion" in text
        or "mutual exclusion" in text
        or "exclusive" in text
        or (
            ("same time" in text or "simultaneous" in text or "together" in text)
            and any(token in text for token in ("never", "not", "cannot", "can't", "won't", "without"))
        )
    )


def _looks_like_coverage(text: str) -> bool:
    return (
        "coverage" in text
        or "one of them" in text
        or "one of the" in text
        or "at least one" in text
        or "exactly one" in text
    )


def _learned_formula_fallback_semantic(
    semantic: SemanticStructure,
) -> SemanticStructure | None:
    if semantic.formula_text:
        return semantic

    learned_handler = get_pattern_handler(semantic.pattern)
    if learned_handler is None:
        return None

    try:
        spec = learned_handler._spec_for_semantic(semantic)
    except ValueError:
        return None

    formula_text = spec.final_candidate_text or spec.formula_text
    if not formula_text:
        return None

    try:
        formula_atoms = collect_formula_atoms(parse_learned_formula(formula_text))
    except ValueError:
        formula_atoms = ()
    if formula_atoms and not set(formula_atoms).issubset(set(semantic.aps)):
        aps = tuple(formula_atoms)
    else:
        aps = semantic.aps or tuple(formula_atoms)

    return replace(
        semantic,
        formula_text=formula_text,
        aps=aps,
        original_pattern=semantic.original_pattern or semantic.pattern,
    )


def _has_late_trigger_positive(
    traces: list[LabeledTrace],
    trigger: str,
    response: str,
) -> bool:
    for trace in traces:
        if trace.kind != "positive" or not trace.expected:
            continue
        for position in _late_trigger_positions(trace, trigger):
            if any(response in step for step in trace.trace[position:]):
                return True
    return False


def _has_late_trigger_negative(
    traces: list[LabeledTrace],
    trigger: str,
    response: str,
) -> bool:
    for trace in traces:
        if trace.kind != "negative" or trace.expected:
            continue
        for position in _late_trigger_positions(trace, trigger):
            if all(response not in step for step in trace.trace[position:]):
                return True
    return False


def _late_trigger_positions(trace: LabeledTrace, trigger: str) -> list[int]:
    return [
        position
        for position, step in enumerate(trace.trace)
        if position > 0 and trigger in step
    ]


def _build_late_trigger_positive_trace(
    trigger: str,
    response: str,
    target_trace_length: int | None,
) -> LabeledTrace:
    length = _late_trigger_trace_length(target_trace_length)
    trigger_index = 1
    response_index = length - 1
    trace = _empty_trace(length)
    trace[trigger_index].add(trigger)
    trace[response_index].add(response)
    return LabeledTrace(
        name=f"pos_late_{trigger}_then_{response}",
        kind="positive",
        trace=trace,
        expected=True,
        reason=(
            f"{trigger} 不在 t0 出现，仍然应该被全局约束检查到；"
            f"随后有限轨迹中出现了 {response}。"
        ),
    )


def _build_late_trigger_negative_trace(
    trigger: str,
    response: str,
    target_trace_length: int | None,
) -> LabeledTrace:
    length = _late_trigger_trace_length(target_trace_length)
    trace = _empty_trace(length)
    trace[1].add(trigger)
    return LabeledTrace(
        name=f"neg_late_{trigger}_without_{response}",
        kind="negative",
        trace=trace,
        expected=False,
        reason=(
            f"{trigger} 在 t0 之后出现，但从该位置到有限轨迹结束都没有 "
            f"{response}；这能区分缺少外层 G 的弱公式。"
        ),
    )


def _late_trigger_trace_length(target_trace_length: int | None) -> int:
    length = target_trace_length or 3
    if length < 2:
        raise ValueError("late-trigger 轨迹至少需要 2 个时间步。")
    return length


def _empty_trace(length: int) -> list[set[str]]:
    return [set() for _ in range(length)]


def _deduplicate_trace_names(traces: list[LabeledTrace]) -> list[LabeledTrace]:
    seen: set[str] = set()
    unique: list[LabeledTrace] = []
    for trace in traces:
        if trace.name not in seen:
            seen.add(trace.name)
            unique.append(trace)
            continue

        suffix = 2
        new_name = f"{trace.name}_{suffix}"
        while new_name in seen:
            suffix += 1
            new_name = f"{trace.name}_{suffix}"
        seen.add(new_name)
        unique.append(replace(trace, name=new_name))
    return unique


def _suggested_trace_count_from_context(context: tuple[str, ...]) -> int | None:
    """从相似轨迹充分性样例里读取人工建议的目标轨迹数量。"""

    target_counts: list[int] = []
    for item in context:
        for match in re.finditer(r"target_trace_count\s*:\s*(\d+)", item):
            target_counts.append(int(match.group(1)))
    if not target_counts:
        return None
    return max(target_counts)


def _parse_llm_trace_items(items: list) -> list[LabeledTrace]:
    """Parse LLM traces, ignoring empty trace placeholders."""

    traces: list[LabeledTrace] = []
    for index, item in enumerate(items, start=1):
        if _is_empty_trace_placeholder(item):
            continue
        traces.append(_trace_from_llm_item(item, index))
    return traces


def _is_empty_trace_placeholder(item: object) -> bool:
    return isinstance(item, dict) and item.get("trace") == []


def _trace_from_llm_item(item: object, index: int) -> LabeledTrace:
    mapping = require_mapping(item, f"traces[{index}]")
    name = optional_string(mapping, "name", f"llm_trace_{index}") or f"llm_trace_{index}"
    kind = require_string(mapping, "kind", f"traces[{index}]").lower()
    if kind not in ALLOWED_TRACE_KINDS:
        raise ValueError(f"traces[{index}].kind 不支持: {kind}")

    raw_steps = _coerce_llm_trace_steps(
        mapping.get("trace"),
        f"traces[{index}].trace",
    )
    if not raw_steps:
        raise ValueError(f"traces[{index}].trace 不能为空。")

    trace: list[set[str]] = []
    for step_index, raw_step in enumerate(raw_steps):
        raw_events = require_list(raw_step, f"traces[{index}].trace[{step_index}]")
        events: set[str] = set()
        for event in raw_events:
            if not isinstance(event, str):
                raise ValueError(f"traces[{index}].trace[{step_index}] 只能包含 AP 字符串。")
            events.add(require_ap_name(event, f"traces[{index}].trace[{step_index}]") )
        trace.append(events)

    return LabeledTrace(
        name=name,
        kind=kind,
        trace=trace,
        expected=coerce_bool(mapping.get("expected"), f"traces[{index}].expected"),
        reason=optional_string(mapping, "reason", "LLM 生成的有限轨迹。"),
    )


def _coerce_llm_trace_steps(value: object, context: str) -> list:
    """Accept the required JSON array plus common string forms copied from examples."""

    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return require_list(value, context)

    text = value.strip()
    if not text:
        raise ValueError(f"{context} 必须是 JSON array。")

    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context} 字符串不是合法 JSON array: {exc.msg}。") from exc
        return require_list(decoded, context)

    parsed = _parse_text_trace_steps(text, context)
    if parsed is not None:
        return parsed
    raise ValueError(f"{context} 必须是 JSON array。")


def _parse_text_trace_steps(text: str, context: str) -> list[list[str]] | None:
    normalized = re.sub(r"\s*->\s*", "\n", text)
    steps_by_index: dict[int, list[str]] = {}
    saw_step = False

    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(r"t(\d+)\s*:\s*(?:\{(.*?)\}|(.*?))\s*", line)
        if match is None:
            return None

        saw_step = True
        step_index = int(match.group(1))
        raw_events = match.group(2) if match.group(2) is not None else match.group(3)
        steps_by_index[step_index] = _parse_text_trace_events(
            raw_events or "",
            f"{context}[{step_index}]",
        )

    if not saw_step:
        return None
    if not steps_by_index:
        return []

    max_index = max(steps_by_index)
    return [steps_by_index.get(index, []) for index in range(max_index + 1)]


def _parse_text_trace_events(raw_events: str, context: str) -> list[str]:
    value = raw_events.strip()
    if not value:
        return []
    events: list[str] = []
    for raw_event in value.split(","):
        event = raw_event.strip()
        if not event:
            continue
        events.append(require_ap_name(event, context))
    return events


