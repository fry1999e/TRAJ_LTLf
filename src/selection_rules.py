"""Diagnostic ranking plus final gates for an operator-resolved formula.

Score/size ordering is retained only for reports and provisional diagnostics.
It is not an operator-resolution mechanism. Automatic finalization accepts at
most the single evaluation already authorized by OperatorResolutionReport.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from itertools import product

from ltlf_parser import Formula
from formula_closure import (
    build_strict_formula_closure_policy,
    validate_formula_closure,
)
from ambiguity_closure import compute_ambiguity_closure_report
from pattern_registry import collect_formula_atoms, parse_learned_formula
from schemas import (
    CandidateSelectionOutcome,
    CompletionDecision,
    FormulaConsistencyEvaluation,
    IRCoverageReport,
    OperatorResolutionReport,
    SemanticStructure,
)


_CERTIFIED_COMPILATION_CACHE_MAX = 128
_CERTIFIED_COMPILATION_CACHE = OrderedDict()
_CERTIFIED_COMPILATION_CACHE_HITS = 0
_CERTIFIED_COMPILATION_CACHE_MISSES = 0


def certified_compilation_cache_info() -> dict[str, int]:
    return {
        "size": len(_CERTIFIED_COMPILATION_CACHE),
        "max_size": _CERTIFIED_COMPILATION_CACHE_MAX,
        "hits": _CERTIFIED_COMPILATION_CACHE_HITS,
        "misses": _CERTIFIED_COMPILATION_CACHE_MISSES,
    }


def clear_certified_compilation_cache() -> None:
    global _CERTIFIED_COMPILATION_CACHE_HITS, _CERTIFIED_COMPILATION_CACHE_MISSES
    _CERTIFIED_COMPILATION_CACHE.clear()
    _CERTIFIED_COMPILATION_CACHE_HITS = 0
    _CERTIFIED_COMPILATION_CACHE_MISSES = 0


def _certified_ir_artifacts(ir, temporal_resolution_by_node):
    """Compile and validate an exact IR/domain once per versioned contract."""

    global _CERTIFIED_COMPILATION_CACHE_HITS, _CERTIFIED_COMPILATION_CACHE_MISSES
    resolution_key = (
        tuple(sorted(dict(temporal_resolution_by_node).items()))
        if temporal_resolution_by_node is not None
        else ("__complete_domain__",)
    )
    key = (
        ir.artifact_hash,
        ir.ir_version,
        ir.node_contract_fingerprint,
        ir.temporal_language_contract_version,
        resolution_key,
    )
    cached = _CERTIFIED_COMPILATION_CACHE.get(key)
    if cached is not None:
        _CERTIFIED_COMPILATION_CACHE_HITS += 1
        _CERTIFIED_COMPILATION_CACHE.move_to_end(key)
        return cached
    _CERTIFIED_COMPILATION_CACHE_MISSES += 1

    from ir_compiler import (
        compile_completion_hole_plan,
        compile_constrained_skeleton_plan,
        compile_ir_artifact,
    )
    from operator_resolution import assignment_temporal_resolution_by_node

    if temporal_resolution_by_node is not None:
        artifacts = (
            compile_ir_artifact(
                ir,
                temporal_resolution_by_node=dict(temporal_resolution_by_node),
            ),
        )
    else:
        skeleton_plan = compile_constrained_skeleton_plan(ir)
        hole_plan = compile_completion_hole_plan(ir, skeleton_plan)
        domains = tuple(hole.allowed_operators for hole in hole_plan.holes)
        assignments = product(*domains) if domains else ((),)
        materialized = []
        for values in assignments:
            by_hole = tuple(
                (hole.hole_id, operator)
                for hole, operator in zip(hole_plan.holes, values)
            )
            resolution = assignment_temporal_resolution_by_node(
                hole_plan=hole_plan,
                assignment=by_hole,
            )
            try:
                materialized.append(
                    compile_ir_artifact(
                        ir,
                        temporal_resolution_by_node=resolution,
                    )
                )
            except ValueError:
                continue
        artifacts = tuple(materialized)
    for artifact in artifacts:
        artifact.validate_for(ir)
    _CERTIFIED_COMPILATION_CACHE[key] = artifacts
    _CERTIFIED_COMPILATION_CACHE.move_to_end(key)
    while len(_CERTIFIED_COMPILATION_CACHE) > _CERTIFIED_COMPILATION_CACHE_MAX:
        _CERTIFIED_COMPILATION_CACHE.popitem(last=False)
    return artifacts


def select_best_evaluation(
    evaluations: list[FormulaConsistencyEvaluation],
    strict_boundary: bool,
    semantic: SemanticStructure | None = None,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
) -> FormulaConsistencyEvaluation | None:
    """Return a strict diagnostic representative; never prove uniqueness."""

    filtered = strict_passed_evaluations(
        evaluations=evaluations,
        strict_boundary=strict_boundary,
        semantic=semantic,
        temporal_resolution_by_node=temporal_resolution_by_node,
    )
    if not filtered:
        return None
    return rank_evaluations(filtered)[0]


def select_fallback_evaluation(
    evaluations: list[FormulaConsistencyEvaluation],
    semantic: SemanticStructure | None = None,
) -> FormulaConsistencyEvaluation | None:
    """
    选择 needs_review 时的参考候选。

    参考候选优先从语义硬过滤通过的公式里选；如果全部语义不合格，再退回到
    全部候选中最高分者，避免报告空白。
    """

    if not evaluations:
        return None

    semantic_passed = [
        evaluation
        for evaluation in evaluations
        if not semantic_guard_reject_reasons(evaluation.formula, semantic)
    ]
    pool = semantic_passed if semantic_passed else evaluations
    return rank_evaluations(pool)[0]


def strict_passed_evaluations(
    evaluations: list[FormulaConsistencyEvaluation],
    strict_boundary: bool,
    semantic: SemanticStructure | None = None,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
) -> list[FormulaConsistencyEvaluation]:
    """返回满足语义硬过滤和严格轨迹一致性的候选。"""

    return [
        evaluation
        for evaluation in evaluations
        if not candidate_reject_reasons(
            evaluation=evaluation,
            strict_boundary=strict_boundary,
            semantic=semantic,
            temporal_resolution_by_node=temporal_resolution_by_node,
        )
    ]


def rank_evaluations(
    evaluations: list[FormulaConsistencyEvaluation],
) -> list[FormulaConsistencyEvaluation]:
    """按 Score、Size、公式文本进行稳定排序。"""

    return sorted(
        evaluations,
        key=lambda item: (-item.score, item.formula.size(), str(item.formula)),
    )


def select_output_evaluation(evaluations, *, evidence_complete, enumeration_complete,
                             degraded=False, candidate_priority=None):
    """runtime selection. An assignment is a candidate, even if texts coincide.

    Scores average ordinary/boundary match rates over nonempty groups. Unknown
    labels are excluded by the caller; an empty evidence set scores zero.
    """
    from dataclasses import replace

    if not evaluations:
        raise ValueError("output selection requires a nonempty candidate domain")
    scored = []
    for evaluation in evaluations:
        groups = [
            [r.matched for r in evaluation.trace_results if r.trace_kind != "boundary"],
            [r.matched for r in evaluation.trace_results if r.trace_kind == "boundary"],
        ]
        rates = [sum(group) / len(group) for group in groups if group]
        scored.append(replace(evaluation, score=sum(rates) / len(rates) if rates else 0.0))
    passing = [e for e in scored if e.trace_results and all(r.matched for r in e.trace_results)]
    pool = passing or scored
    selected = min(pool, key=lambda e: (-e.score, (candidate_priority or {}).get(e.candidate_id, 0),
                                       e.formula.size(), str(e.formula), e.candidate_id))
    unique = bool(evidence_complete and enumeration_complete and not degraded
                  and len(passing) == 1 and selected.trace_results)
    method = "emergency_fallback" if degraded else "unique_evidence_pass" if unique else "scored_fallback"
    return selected, scored, passing, method


def exact_formula_classification(formula: Formula) -> str:
    """Conservatively prove whether a formula is valid or unsatisfiable."""

    if formula.op == "true":
        return "valid"
    if formula.op == "false":
        return "unsatisfiable"
    if formula.op == "ap":
        return "contingent"

    if formula.op == "not":
        child = exact_formula_classification(formula.args[0])
        if child == "valid":
            return "unsatisfiable"
        if child == "unsatisfiable":
            return "valid"
        return "contingent"

    if formula.op in {"and", "or"}:
        left, right = formula.args
        left_class = exact_formula_classification(left)
        right_class = exact_formula_classification(right)
        complements = _are_formula_complements(left, right)
        if formula.op == "and":
            if complements or "unsatisfiable" in {left_class, right_class}:
                return "unsatisfiable"
            if left_class == right_class == "valid":
                return "valid"
            return "contingent"
        if complements or "valid" in {left_class, right_class}:
            return "valid"
        if left_class == right_class == "unsatisfiable":
            return "unsatisfiable"
        return "contingent"

    if formula.op == "implies":
        left_class = exact_formula_classification(formula.args[0])
        right_class = exact_formula_classification(formula.args[1])
        if left_class == "unsatisfiable" or right_class == "valid":
            return "valid"
        if left_class == "valid" and right_class == "unsatisfiable":
            return "unsatisfiable"
        return "contingent"

    if formula.op in {"F", "G"}:
        return exact_formula_classification(formula.args[0])

    if formula.op == "X":
        child_class = exact_formula_classification(formula.args[0])
        return "unsatisfiable" if child_class == "unsatisfiable" else "contingent"

    if formula.op == "U":
        left_class = exact_formula_classification(formula.args[0])
        right_class = exact_formula_classification(formula.args[1])
        if right_class == "valid":
            return "valid"
        if right_class == "unsatisfiable":
            return "unsatisfiable"
        if left_class == "unsatisfiable":
            return right_class
        return "contingent"

    return "contingent"


def _are_formula_complements(left: Formula, right: Formula) -> bool:
    return (
        left.op == "not"
        and left.args[0] == right
        or right.op == "not"
        and right.args[0] == left
    )


def candidate_reject_reasons(
    evaluation: FormulaConsistencyEvaluation,
    strict_boundary: bool,
    semantic: SemanticStructure | None = None,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """返回完整候选公式未通过最终过滤的原因。"""

    return (
        *semantic_guard_reject_reasons(
            evaluation.formula,
            semantic,
            temporal_resolution_by_node=temporal_resolution_by_node,
        ),
        *metric_reject_reasons(
            pos_sat=evaluation.pos_sat,
            neg_rej=evaluation.neg_rej,
            bound_match=evaluation.bound_match,
            strict_boundary=strict_boundary,
        ),
    )


def completion_decision_reject_reasons(
    decision: CompletionDecision,
    strict_boundary: bool,
    semantic: SemanticStructure | None = None,
) -> tuple[str, ...]:
    """返回补全候选未通过过滤的原因。"""

    resolution: dict[str, str] | None = None
    assignment_reasons: tuple[str, ...] = ()
    if semantic is not None and semantic.faithful_ir is not None:
        try:
            from ir_compiler import (
                compile_completion_hole_plan,
                compile_constrained_skeleton_plan,
            )
            from operator_resolution import assignment_temporal_resolution_by_node

            skeleton_plan = compile_constrained_skeleton_plan(semantic.faithful_ir)
            hole_plan = compile_completion_hole_plan(
                semantic.faithful_ir,
                skeleton_plan,
            )
            resolution = assignment_temporal_resolution_by_node(
                hole_plan=hole_plan,
                assignment=decision.assignments,
            )
        except ValueError as exc:
            assignment_reasons = (f"补全赋值未闭合受限时序孔计划：{exc}",)
    return (
        *assignment_reasons,
        *semantic_guard_reject_reasons(
            decision.formula,
            semantic,
            temporal_resolution_by_node=resolution,
        ),
        *metric_reject_reasons(
            pos_sat=decision.pos_sat,
            neg_rej=decision.neg_rej,
            bound_match=decision.boundary_match,
            strict_boundary=strict_boundary,
        ),
    )


def semantic_guard_reject_reasons(
    formula: Formula,
    semantic: SemanticStructure | None,
    *,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """
    语义硬过滤。

    对 response 模式，最终公式必须显式保留 trigger 和 response，并包含
    trigger -> response-like 的触发关系。这样可以防止 F response 这类过短、
    但遗漏触发语义的公式被选为最终规约。
    """

    if semantic is None:
        return ()

    if semantic.faithful_ir is not None:
        return ir_structure_reject_reasons(
            formula,
            semantic,
            temporal_resolution_by_node=temporal_resolution_by_node,
        )

    reasons: list[str] = []
    if semantic.formula_text:
        try:
            expected_formula = parse_learned_formula(semantic.formula_text)
        except ValueError as exc:
            return (f"semantic.formula_text 无法解析: {exc}",)
        expected_atoms = semantic.aps or collect_formula_atoms(expected_formula)
        atoms = _collect_atoms(formula)
        for ap in expected_atoms:
            if ap not in atoms:
                reasons.append(f"缺少 learned pattern 原子命题 {ap}。")
        if str(formula) != str(expected_formula):
            reasons.append(
                f"候选公式未匹配 learned pattern 公式 {expected_formula}。"
            )
        return tuple(reasons)

    atoms = _collect_atoms(formula)
    if semantic.trigger not in atoms:
        reasons.append(f"缺少 trigger 原子命题 {semantic.trigger}。")
    if semantic.response not in atoms:
        reasons.append(f"缺少 response 原子命题 {semantic.response}。")

    if semantic.pattern == "response" and not _has_response_implication(
        formula=formula,
        trigger=semantic.trigger,
        response=semantic.response,
    ):
        reasons.append("缺少 response 模式的 trigger -> response 触发关系。")

    return tuple(reasons)


def ir_structure_reject_reasons(
    formula: Formula,
    semantic: SemanticStructure,
    *,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Check a candidate against certified compilations of the reviewed IR.

    Before operator resolution, membership in any authorized assignment is a
    structural validity check only.  Finalization supplies the unique winning
    assignment and therefore requires an exact certified compilation match.
    """

    ir = semantic.faithful_ir
    if ir is None:
        return ()
    reasons: list[str] = []
    closure_report = validate_formula_closure(
        formula,
        build_strict_formula_closure_policy(
            ir,
            semantic.approved_semantic_ir,
        ),
        ir_version=ir.ir_version,
    )
    reasons.extend(
        f"FormulaClosurePolicy: {reason}"
        for reason in closure_report.reasons
    )
    try:
        artifacts = _certified_ir_artifacts(
            ir,
            temporal_resolution_by_node,
        )
    except ValueError as exc:
        reasons.append(f"IR 编译产物或 lowering certificate 无效：{exc}")
    else:
        if not any(formula == artifact.formula for artifact in artifacts):
            reasons.append("候选公式不是该 IR 及指定时序赋值的认证编译产物。")
    return tuple(reasons)


def select_with_ir_coverage_gate(
    evaluations: list[FormulaConsistencyEvaluation],
    *,
    strict_boundary: bool,
    semantic: SemanticStructure,
    coverage: IRCoverageReport,
    run_origin: str = "normal",
    degraded: bool = False,
    fallback: bool = False,
    candidates_distinguished: bool = True,
    operator_resolution: OperatorResolutionReport | None = None,
    temporal_resolution_by_node: Mapping[str, str] | None = None,
    temporal_resolution_error: str = "",
) -> CandidateSelectionOutcome:
    """Apply consistency and the independent IR coverage/status policy."""

    candidate = select_best_evaluation(
        evaluations,
        strict_boundary=strict_boundary,
        semantic=semantic,
        temporal_resolution_by_node=temporal_resolution_by_node,
    )
    reasons: list[str] = []
    automatic = True
    if len(evaluations) > 1:
        automatic = False
        reasons.append(
            "finalization gate accepts only one OperatorResolution-authorized evaluation"
        )
    if run_origin == "feedback_rerun":
        automatic = False
        reasons.append("feedback rerun candidates require independent approval")
    if semantic.faithful_ir is None:
        automatic = False
        reasons.append("FaithfulSemanticIR is missing")
    else:
        approved_record = semantic.approved_semantic_ir
        if semantic.faithful_ir.provenance.extraction_mode == "working_interpretation_v1":
            automatic = False
            reasons.append("working interpretation is not strict semantic approval")
        if approved_record is None:
            automatic = False
            reasons.append("ApprovedSemanticIR is missing")
        elif not approved_record.approved:
            automatic = False
            reasons.append("ApprovedSemanticIR exists but is not approved")
        if approved_record is not None and approved_record.remaining_ambiguity_debt:
            automatic = False
            reasons.append(
                "ApprovedSemanticIR still has ambiguity debt: "
                + ", ".join(approved_record.remaining_ambiguity_debt)
            )
        if approved_record is not None and approved_record.evidence_status != "complete":
            automatic = False
            reasons.append(
                "labeled trace evidence is incomplete: "
                + approved_record.evidence_status
            )
        if semantic.faithful_ir.has_high_risk_ambiguity:
            automatic = False
            reasons.append("high-risk semantic ambiguity is unresolved")
        ambiguity_closure = compute_ambiguity_closure_report(
            semantic,
            ir_coverage=coverage,
        )
        if not ambiguity_closure.passed:
            automatic = False
            reasons.append(
                "ir_ambiguity_state_mismatch: "
                + ", ".join(ambiguity_closure.reasons)
            )
    if not coverage.complete:
        automatic = False
        reasons.append("IR coverage gate is incomplete")
    if degraded or fallback:
        automatic = False
        reasons.append("degraded or fallback candidates cannot become final")
    if not candidates_distinguished:
        automatic = False
        reasons.append("approved operator assignments are not uniquely resolved")
    if operator_resolution is None:
        automatic = False
        reasons.append("OperatorResolutionReport is missing")
    else:
        if not operator_resolution.complete:
            automatic = False
            reasons.append(
                "operator resolution is not complete: "
                + operator_resolution.status
            )
        if not operator_resolution.authoritative:
            automatic = False
            reasons.append("operator resolution is not authoritative")
        if len(operator_resolution.passing_assignments) != 1:
            automatic = False
            reasons.append(
                "operator resolution does not contain exactly one passing assignment"
            )
        if semantic.faithful_ir is not None:
            try:
                from ir_compiler import (
                    compile_completion_hole_plan,
                    compile_constrained_skeleton_plan,
                )
                from operator_resolution import resolved_temporal_resolution_by_node

                gate_skeleton_plan = compile_constrained_skeleton_plan(
                    semantic.faithful_ir
                )
                gate_hole_plan = compile_completion_hole_plan(
                    semantic.faithful_ir,
                    gate_skeleton_plan,
                )
                report_resolution = resolved_temporal_resolution_by_node(
                    report=operator_resolution,
                    hole_plan=gate_hole_plan,
                )
                if (
                    temporal_resolution_by_node is not None
                    and dict(temporal_resolution_by_node) != report_resolution
                ):
                    raise ValueError(
                        "winning node assignment does not match OperatorResolutionReport"
                    )
            except ValueError as exc:
                automatic = False
                reasons.append(
                    "operator resolution/report/IR decision mismatch: " + str(exc)
                )
    if temporal_resolution_error:
        automatic = False
        reasons.append(
            "winning temporal assignment is invalid: " + temporal_resolution_error
        )
    elif semantic.faithful_ir is not None and temporal_resolution_by_node is None:
        automatic = False
        reasons.append("winning temporal assignment is missing")
    if candidate is None:
        automatic = False
        reasons.append("no candidate strictly passes consistency")

    if candidate is not None and automatic:
        return CandidateSelectionOutcome(candidate, "final", True, ())
    if candidate is not None:
        return CandidateSelectionOutcome(candidate, "provisional", False, tuple(reasons))
    return CandidateSelectionOutcome(None, "needs_review", False, tuple(reasons))


def metric_reject_reasons(
    pos_sat: float,
    neg_rej: float,
    bound_match: float,
    strict_boundary: bool,
) -> tuple[str, ...]:
    """返回轨迹一致性指标未通过严格过滤的原因。"""

    reasons: list[str] = []
    if pos_sat != 1.0:
        reasons.append(f"PosSat={pos_sat:.2f}，存在正轨迹未满足。")
    if neg_rej != 1.0:
        reasons.append(f"NegRej={neg_rej:.2f}，存在反轨迹未拒绝。")
    if strict_boundary and bound_match != 1.0:
        reasons.append(f"BoundMatch={bound_match:.2f}，边界轨迹未完全匹配。")
    return tuple(reasons)


def candidate_selection_review_notes(
    evaluations: list[FormulaConsistencyEvaluation],
    strict_boundary: bool,
    semantic: SemanticStructure | None = None,
) -> tuple[str, ...]:
    """根据候选评估结果生成 needs_review 评审意见。"""

    if not evaluations:
        return (
            "没有生成任何完整候选公式，需要重新检查骨架生成和 X/F/G 补全阶段。",
        )

    notes: list[str] = [
        "没有候选公式同时满足语义硬过滤和严格轨迹过滤，当前不能确认最终 LTLf 规约。",
    ]

    if semantic is not None:
        semantic_passed = [
            evaluation
            for evaluation in evaluations
            if not semantic_guard_reject_reasons(evaluation.formula, semantic)
        ]
        if not semantic_passed:
            notes.append("所有候选都未通过语义硬过滤，应重新生成保留触发关系的候选公式。")

    if _max_metric(evaluations, "pos_sat") < 1.0:
        notes.append("存在正轨迹未被候选公式满足，应检查响应算子是否过强或范围是否错误。")
    if _max_metric(evaluations, "neg_rej") < 1.0:
        notes.append("存在反轨迹未被候选公式拒绝，应检查公式是否过弱或缺少触发关系。")
    if strict_boundary and _max_metric(evaluations, "bound_match") < 1.0:
        notes.append("边界轨迹未完全匹配，应优先根据边界标签调整 X、F、G 或 XF。")

    best = select_fallback_evaluation(evaluations, semantic=semantic)
    if best is not None:
        reasons = candidate_reject_reasons(
            evaluation=best,
            strict_boundary=strict_boundary,
            semantic=semantic,
        )
        reason_text = "；".join(reasons[:2]) if reasons else "未通过严格过滤"
        notes.append(
            f"当前参考候选为 {best.formula}，但它仍不能作为最终公式：{reason_text}"
        )

    return tuple(notes)


def _collect_atoms(formula: Formula) -> set[str]:
    if formula.op == "ap":
        return {formula.name}

    atoms: set[str] = set()
    for child in formula.args:
        atoms.update(_collect_atoms(child))
    return atoms


def _has_response_implication(
    formula: Formula,
    trigger: str,
    response: str,
) -> bool:
    if formula.op == "implies":
        left, right = formula.args
        return trigger in _collect_atoms(left) and response in _collect_atoms(right)

    return any(
        _has_response_implication(
            formula=child,
            trigger=trigger,
            response=response,
        )
        for child in formula.args
    )


def _max_metric(
    evaluations: list[FormulaConsistencyEvaluation],
    field_name: str,
) -> float:
    if not evaluations:
        return 0.0
    return max(float(getattr(evaluation, field_name)) for evaluation in evaluations)
