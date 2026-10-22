"""
受限骨架结构校验器。

本模块只检查受限骨架是否能进入后续带孔模板阶段：
    1. 公式已经由 parser 成功解析；
    2. 叶子元素只使用 NLU 阶段抽取出的 AP，不允许 true/false 常量；
    3. 不包含受限骨架阶段禁用的 X/F/G。

批量校验入口还会根据已审核的正/反轨迹做一层保守粗过滤，只排除明显违背
response 方向或缺少关键 AP 的骨架；最终正确性仍交给后续轨迹一致性模拟阶段。
"""

from __future__ import annotations

from data_examples import LabeledTrace
from ltlf_parser import Atom, Formula, parse_formula
from ltlf_syntax import has_repeated_negation, normalize_boolean_operator_text
from schemas import SemanticStructure, SkeletonDiagnostic
from semantic_topology import build_ir_semantic_view


FORBIDDEN_SKELETON_OPS = {"X", "F", "G"}
NON_AP_LITERAL_OPS = {"true", "false"}
ALLOWED_SKELETON_OPS = {
    "ap",
    "not",
    "and",
    "or",
    "implies",
    "U",
}


def validate_skeletons(
    skeletons: list[Formula],
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
) -> list[SkeletonDiagnostic]:
    """批量校验受限骨架，并按正/反轨迹做保守粗过滤。"""

    diagnostics: list[SkeletonDiagnostic] = []
    for skeleton in skeletons:
        diagnostic = validate_skeleton(skeleton, [], semantic)
        if diagnostic.passed and semantic.faithful_ir is None:
            diagnostic = _apply_response_trace_heuristic(
                diagnostic=diagnostic,
                traces=traces,
                semantic=semantic,
            )
        diagnostics.append(diagnostic)
    return diagnostics


def validate_skeleton(
    skeleton: Formula,
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
) -> SkeletonDiagnostic:
    """结构校验单个受限骨架，不做轨迹粗过滤。"""

    del traces
    ops = _collect_ops(skeleton)
    forbidden_ops = sorted(ops & FORBIDDEN_SKELETON_OPS)

    allowed_atoms = _allowed_atoms(semantic)
    non_ap_literals = sorted(ops & NON_AP_LITERAL_OPS)
    unknown_atoms = sorted(_collect_atoms(skeleton) - allowed_atoms)
    if non_ap_literals or unknown_atoms:
        evidence = []
        if non_ap_literals:
            evidence.append(
                "骨架包含非 AP 常量: "
                f"{', '.join(non_ap_literals)}。"
            )
        if unknown_atoms:
            evidence.extend(
                (
                    f"骨架包含 AP 白名单外元素: {', '.join(unknown_atoms)}。",
                    f"允许 AP: {', '.join(sorted(allowed_atoms)) or '(none)'}。",
                )
            )
        if forbidden_ops:
            evidence.append(
                "骨架还包含受限阶段禁用的具体时序算子: "
                f"{', '.join(forbidden_ops)}。"
            )
        return SkeletonDiagnostic(
            skeleton=skeleton,
            passed=False,
            error_type="non_ap_element" if non_ap_literals else "unknown_ap",
            trace_evidence=tuple(evidence),
            suggestion=(
                "受限骨架的叶子元素只能来自 AP 白名单；不要使用 true/false、"
                "新造变量或占位符。"
            ),
        )

    if forbidden_ops:
        return SkeletonDiagnostic(
            skeleton=skeleton,
            passed=False,
            error_type="forbidden_temporal_operator",
            trace_evidence=(
                f"骨架包含受限阶段禁用的具体时序算子: {', '.join(forbidden_ops)}。",
            ),
            suggestion="受限骨架阶段只能保留 AP、括号和布尔/非 XFG 结构；X/F/G 留给补全阶段。",
        )

    if has_repeated_negation(skeleton):
        return SkeletonDiagnostic(
            skeleton=skeleton,
            passed=False,
            error_type="repeated_negation",
            trace_evidence=(
                "骨架包含连续或仅隔括号嵌套的重复否定，例如 !!p 或 !(!p)。",
            ),
            suggestion=(
                "删除重复否定并保留等价的内部公式；"
                "否定可以用于不同布尔层级，例如 !(p & !q)，"
                "但不能直接作用于另一个否定。"
            ),
        )

    unsupported_ops = sorted(
        ops - ALLOWED_SKELETON_OPS - FORBIDDEN_SKELETON_OPS - NON_AP_LITERAL_OPS
    )
    if unsupported_ops:
        return SkeletonDiagnostic(
            skeleton=skeleton,
            passed=False,
            error_type="unsupported_operator",
            trace_evidence=(
                f"骨架包含当前 parser 不支持的算子: {', '.join(unsupported_ops)}。",
            ),
            suggestion="请只使用 AP 白名单中的原子命题、!、&、|、->、括号；必要时可保留 U。",
        )

    diagnostic = SkeletonDiagnostic(
        skeleton=skeleton,
        passed=True,
        error_type="none",
        trace_evidence=(
            "骨架通过结构校验：语法可解析，叶子元素均来自 AP 白名单，且不含 X/F/G。",
        ),
        suggestion="骨架可进入 RAG + 带孔候选模板生成阶段。",
    )
    if semantic.faithful_ir is None:
        return diagnostic
    return _validate_ir_skeleton_topology(diagnostic, semantic)


def _validate_ir_skeleton_topology(
    diagnostic: SkeletonDiagnostic,
    semantic: SemanticStructure,
) -> SkeletonDiagnostic:
    """Reject native-IR skeletons that change the certified boolean topology."""

    from ir_compiler import compile_ir_skeleton

    ir = semantic.faithful_ir
    if ir is None:  # pragma: no cover - guarded by caller
        return diagnostic
    candidate_set = semantic.candidate_ir_set
    expected_skeletons = (
        tuple(
            compile_ir_skeleton(branch.faithful_ir)
            for branch in candidate_set.branches
        )
        if candidate_set is not None
        and len(candidate_set.branches) > 1
        and not (
            semantic.approved_semantic_ir is not None
            and semantic.approved_semantic_ir.approved
        )
        else (compile_ir_skeleton(ir),)
    )
    if any(
        _same_formula(diagnostic.skeleton, expected)
        for expected in expected_skeletons
    ):
        return diagnostic
    return SkeletonDiagnostic(
        skeleton=diagnostic.skeleton,
        passed=False,
        error_type="ir_topology_mismatch",
        trace_evidence=(
            "骨架与 FaithfulSemanticIR 编译出的受限布尔拓扑不一致。",
            "IR version: "
            + ir.ir_version
            + "; expected skeletons: "
            + ", ".join(str(expected) for expected in expected_skeletons),
        ),
        suggestion="按 IR 节点类型、children 顺序和 AP 映射重新生成骨架。",
    )


def format_skeleton_diagnostics_for_agent(
    diagnostics: list[SkeletonDiagnostic],
) -> str:
    """把结构校验失败结果压缩成下一轮 SkeletonAgent 可用的反馈文本。"""

    passed_count = sum(1 for diagnostic in diagnostics if diagnostic.passed)
    lines = [
        f"当前可进入补全阶段的受限骨架数量为 {passed_count}。",
        "请重新生成受限骨架，并满足：语法可解析、叶子元素只使用已抽取 AP、不使用 true/false、不含 X/F/G、不使用 !!p 或 !(!p) 等重复否定、统一使用 & 和 |，且 response 模式中保持 trigger -> response 的语义方向。",
    ]
    for diagnostic in diagnostics:
        if diagnostic.passed:
            continue
        lines.append(f"骨架: {diagnostic.skeleton}")
        lines.append(f"错误类型: {diagnostic.error_type}")
        for evidence in diagnostic.trace_evidence:
            lines.append(f"证据: {evidence}")
        lines.append(f"建议: {diagnostic.suggestion}")
    return "\n".join(lines)


def _apply_response_trace_heuristic(
    *,
    diagnostic: SkeletonDiagnostic,
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
) -> SkeletonDiagnostic:
    evidence = _response_trace_heuristic_rejection(
        diagnostic.skeleton,
        traces,
        semantic,
    )
    if evidence is None:
        return diagnostic
    return SkeletonDiagnostic(
        skeleton=diagnostic.skeleton,
        passed=False,
        error_type="trace_heuristic_mismatch",
        trace_evidence=evidence,
        suggestion=(
            "response 模式的受限骨架应保留 trigger -> response 方向；"
            "优先生成 trigger -> response、!trigger | response、"
            "!(trigger & !response) 或其逆否等价形式。"
        ),
    )


def _response_trace_heuristic_rejection(
    skeleton: Formula,
    traces: list[LabeledTrace],
    semantic: SemanticStructure,
) -> tuple[str, ...] | None:
    if not _semantic_allows_response_guard(semantic):
        return None
    trigger_formula, response_formula = _semantic_response_endpoint_formulas(semantic)
    if not _is_state_formula(trigger_formula) or not _is_state_formula(response_formula):
        return None

    context = _response_trace_context(traces, trigger_formula, response_formula)
    if not context["has_relevant_trace"]:
        return None

    atoms = _collect_atoms(skeleton)
    required_atoms = _collect_atoms(trigger_formula) | _collect_atoms(response_formula)
    if required_atoms - atoms:
        missing = ", ".join(sorted(required_atoms - atoms))
        return (
            f"骨架缺少 response 语义所需 AP: {missing}。",
            context["summary"],
        )

    if _is_response_direction_formula(skeleton, trigger_formula, response_formula):
        return None

    if _is_reverse_response_formula(skeleton, trigger_formula, response_formula):
        return (
            "骨架方向更像 response -> trigger，与自然语言中 trigger 触发 response 的方向相反。",
            context["summary"],
        )

    if _is_same_step_conjunction(skeleton, trigger_formula, response_formula):
        if context["positive_late_response"]:
            return (
                "骨架要求 trigger 与 response 在同一位置同时成立，但正轨迹显示 response 可以晚于 trigger。",
                context["summary"],
            )

    if _is_broad_disjunction(skeleton, trigger_formula, response_formula):
        return (
            "骨架使用 trigger | response，无法表达 trigger 出现后必须得到 response 的约束。",
            context["summary"],
        )

    if _is_negated_same_step_conjunction(skeleton, trigger_formula, response_formula):
        return (
            "骨架排斥 trigger 与 response 同时成立，明显偏离 response 需求的触发-响应关系。",
            context["summary"],
        )

    return None


def _semantic_allows_response_guard(semantic: SemanticStructure) -> bool:
    if semantic.faithful_ir is not None:
        relation = build_ir_semantic_view(semantic.faithful_ir).primary_relation
        return relation is not None and relation.relation_type == "implication"
    return semantic.pattern == "response"


def _response_trace_context(
    traces: list[LabeledTrace],
    trigger: Formula,
    response: Formula,
) -> dict[str, object]:
    positive_late: list[str] = []
    negative_missing: list[str] = []
    relevant: list[str] = []
    for trace in traces:
        if trace.kind not in {"positive", "negative"}:
            continue
        if not any(
            _state_formula_holds(trigger, step)
            or _state_formula_holds(response, step)
            for step in trace.trace
        ):
            continue
        relevant.append(trace.name)
        if trace.kind == "positive" and trace.expected:
            for index, step in enumerate(trace.trace):
                if not _state_formula_holds(trigger, step):
                    continue
                if not _state_formula_holds(response, step) and any(
                    _state_formula_holds(response, future_step)
                    for future_step in trace.trace[index + 1 :]
                ):
                    positive_late.append(trace.name)
                    break
        if trace.kind == "negative" and not trace.expected:
            for index, step in enumerate(trace.trace):
                if _state_formula_holds(trigger, step) and not any(
                    _state_formula_holds(response, future_step)
                    for future_step in trace.trace[index:]
                ):
                    negative_missing.append(trace.name)
                    break

    details = []
    if positive_late:
        details.append("正轨迹显示延迟响应: " + ", ".join(positive_late))
    if negative_missing:
        details.append("反轨迹显示缺失响应违例: " + ", ".join(negative_missing))
    if not details and relevant:
        details.append("存在正/反轨迹可用于 response 方向粗过滤: " + ", ".join(relevant))
    return {
        "has_relevant_trace": bool(relevant),
        "positive_late_response": bool(positive_late),
        "negative_missing_response": bool(negative_missing),
        "summary": "；".join(details),
    }


def _is_response_direction_formula(
    formula: Formula,
    trigger: Formula,
    response: Formula,
) -> bool:
    formula = _strip_double_negation(formula)
    if formula.op == "implies":
        left, right = formula.args
        return (
            _same_formula(left, trigger)
            and _same_formula(right, response)
        ) or (
            _is_negated_formula(left, response)
            and _is_negated_formula(right, trigger)
        )
    if formula.op == "or":
        left, right = formula.args
        return _unordered_match(
            left,
            right,
            lambda item: _is_negated_formula(item, trigger),
            lambda item: _same_formula(item, response),
        )
    if formula.op == "not":
        child = _strip_double_negation(formula.args[0])
        if child.op == "and":
            left, right = child.args
            return _unordered_match(
                left,
                right,
                lambda item: _same_formula(item, trigger),
                lambda item: _is_negated_formula(item, response),
            )
    return False


def _is_reverse_response_formula(
    formula: Formula,
    trigger: Formula,
    response: Formula,
) -> bool:
    formula = _strip_double_negation(formula)
    if formula.op == "implies":
        left, right = formula.args
        return (
            _same_formula(left, response)
            and _same_formula(right, trigger)
        ) or (
            _is_negated_formula(left, trigger)
            and _is_negated_formula(right, response)
        )
    if formula.op == "or":
        left, right = formula.args
        return _unordered_match(
            left,
            right,
            lambda item: _is_negated_formula(item, response),
            lambda item: _same_formula(item, trigger),
        )
    if formula.op == "not":
        child = _strip_double_negation(formula.args[0])
        if child.op == "and":
            left, right = child.args
            return _unordered_match(
                left,
                right,
                lambda item: _same_formula(item, response),
                lambda item: _is_negated_formula(item, trigger),
            )
    return False


def _is_same_step_conjunction(
    formula: Formula,
    trigger: Formula,
    response: Formula,
) -> bool:
    formula = _strip_double_negation(formula)
    if formula.op != "and":
        return False
    left, right = formula.args
    return _unordered_match(
        left,
        right,
        lambda item: _same_formula(item, trigger),
        lambda item: _same_formula(item, response),
    )


def _is_broad_disjunction(
    formula: Formula,
    trigger: Formula,
    response: Formula,
) -> bool:
    formula = _strip_double_negation(formula)
    if formula.op != "or":
        return False
    left, right = formula.args
    return _unordered_match(
        left,
        right,
        lambda item: _same_formula(item, trigger),
        lambda item: _same_formula(item, response),
    )


def _is_negated_same_step_conjunction(
    formula: Formula,
    trigger: Formula,
    response: Formula,
) -> bool:
    formula = _strip_double_negation(formula)
    if formula.op != "not":
        return False
    return _is_same_step_conjunction(formula.args[0], trigger, response)


def _strip_double_negation(formula: Formula) -> Formula:
    current = formula
    while current.op == "not" and current.args and current.args[0].op == "not":
        current = current.args[0].args[0]
    return current


def _same_formula(formula: Formula, expected: Formula) -> bool:
    return _strip_double_negation(formula) == _strip_double_negation(expected)


def _is_negated_formula(formula: Formula, expected: Formula) -> bool:
    formula = _strip_double_negation(formula)
    return (
        formula.op == "not"
        and bool(formula.args)
        and _same_formula(formula.args[0], expected)
    )


def _unordered_match(
    left: Formula,
    right: Formula,
    first_predicate,
    second_predicate,
) -> bool:
    return (
        first_predicate(left)
        and second_predicate(right)
    ) or (
        first_predicate(right)
        and second_predicate(left)
    )


def _semantic_response_endpoint_formulas(
    semantic: SemanticStructure,
) -> tuple[Formula, Formula]:
    if semantic.faithful_ir is not None:
        from ir_compiler import compile_ir_skeleton

        skeleton = compile_ir_skeleton(semantic.faithful_ir)
        if skeleton.op == "implies" and len(skeleton.args) == 2:
            return skeleton.args[0], skeleton.args[1]
        return skeleton, skeleton
    trigger_text, response_text = _semantic_response_endpoint_texts(semantic)
    try:
        trigger = parse_formula(trigger_text)
    except ValueError:
        trigger = Atom(semantic.trigger or "trigger")
    try:
        response = parse_formula(response_text)
    except ValueError:
        response = Atom(semantic.response or "response")
    return trigger, response


def _semantic_response_endpoint_texts(
    semantic: SemanticStructure,
) -> tuple[str, str]:
    trigger_text = _component_formula_text(
        semantic,
        "trigger_definition",
        "trigger_formula",
    )
    response_text = _component_formula_text(
        semantic,
        "response_formula",
        "response_definition",
    )
    return (
        trigger_text or semantic.trigger or "trigger",
        response_text or semantic.response or "response",
    )


def _component_formula_text(
    semantic: SemanticStructure,
    *keys: str,
) -> str:
    components = dict(semantic.components)
    for key in keys:
        text = _component_rhs(components.get(key, ""))
        if not text:
            continue
        text = normalize_boolean_operator_text(text)
        try:
            return str(_strip_xfg(parse_formula(text)))
        except ValueError:
            return text
    return ""


def _component_rhs(value: str) -> str:
    text = value.strip().strip("`\"'“”‘’")
    if not text:
        return ""
    marker_index = -1
    for marker in (":=", "="):
        marker_index = text.find(marker)
        if marker_index >= 0:
            text = text[marker_index + len(marker) :].strip()
            break
    return text.rstrip(".,;:，。；：")


def _strip_xfg(formula: Formula) -> Formula:
    if formula.op in {"X", "F", "G"} and formula.args:
        return _strip_xfg(formula.args[0])
    if not formula.args:
        return formula
    return Formula(
        formula.op,
        tuple(_strip_xfg(child) for child in formula.args),
        formula.name,
    )


def _is_state_formula(formula: Formula) -> bool:
    if formula.op in {"ap", "true", "false"}:
        return True
    if formula.op in {"not", "and", "or", "implies"}:
        return all(_is_state_formula(child) for child in formula.args)
    return False


def _state_formula_holds(formula: Formula, step: set[str]) -> bool:
    if formula.op == "ap":
        return formula.name in step
    if formula.op == "true":
        return True
    if formula.op == "false":
        return False
    if formula.op == "not":
        return not _state_formula_holds(formula.args[0], step)
    if formula.op == "and":
        left, right = formula.args
        return _state_formula_holds(left, step) and _state_formula_holds(right, step)
    if formula.op == "or":
        left, right = formula.args
        return _state_formula_holds(left, step) or _state_formula_holds(right, step)
    if formula.op == "implies":
        left, right = formula.args
        return (not _state_formula_holds(left, step)) or _state_formula_holds(right, step)
    return False


def _allowed_atoms(semantic: SemanticStructure) -> set[str]:
    if semantic.faithful_ir is not None:
        return set(semantic.faithful_ir.ap_names)
    atoms = set(semantic.aps)
    if not atoms:
        if semantic.trigger and semantic.trigger != "unknown_ap":
            atoms.add(semantic.trigger)
        if semantic.response and semantic.response != "unknown_ap":
            atoms.add(semantic.response)
    return atoms


def _collect_atoms(formula: Formula) -> set[str]:
    if formula.op == "ap":
        return {formula.name}
    atoms: set[str] = set()
    for child in formula.args:
        atoms.update(_collect_atoms(child))
    return atoms


def _collect_ops(formula: Formula) -> set[str]:
    ops = {formula.op}
    for child in formula.args:
        ops.update(_collect_ops(child))
    return ops
