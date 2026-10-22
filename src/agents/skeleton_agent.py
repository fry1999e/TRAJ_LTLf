"""
受限骨架生成 Agent。

rules 模式使用本地规则；api 模式要求 LLM 返回可解析、只含 AP 白名单叶子元素
且不含 X/F/G 的受限 LTLf 骨架。后续 validator 只做结构校验，最终正确性由轨迹一致性模拟判断。
"""

from __future__ import annotations

import re

from agents.base_agent import BaseAgent, JSONLLMClient
from agents.llm_parsing import coerce_list_field
from agents.prompt_context import (
    FULL_PROCESS_SOURCE,
    KB_PATTERNS,
    KB_SKELETON_DIAGNOSTICS,
    build_stage_prompt_context,
    structured_system_prompt,
)
from data_examples import LabeledTrace
from guided_learning import GuidedLearningContext, guided_prompt_rule
from ltlf_parser import And, Atom, Formula, Implies, Not, Or, parse_formula
from ltlf_syntax import (
    has_repeated_negation,
    normalize_boolean_operator_text,
    validate_canonical_boolean_operators,
)
from pattern_registry import (
    get_pattern_handler_for_semantic,
    learned_pattern_direct_execution_allowed,
)
from rag.example_memory import (
    GUIDED_FULL_PROCESS_SOURCE,
    GUIDED_SKELETON_SOURCE,
    PATTERN_SOURCE,
)
from schemas import SemanticStructure
from temporal_domain import finite_requirement_for_pipeline
from faithful_ir import precedence_roles_from_semantic
from ir_compiler import compile_ir_skeleton


class SkeletonAgent(BaseAgent):
    """RAG + LLM 风格的受限 LTLf 骨架生成 Agent。"""

    def __init__(self, retriever, llm_client: JSONLLMClient | None = None):
        super().__init__(retriever=retriever, name="SkeletonAgent", llm_client=llm_client)

    def generate(
        self,
        requirement: str,
        semantic: SemanticStructure,
        traces: list[LabeledTrace] | None = None,
        feedback_history: list[str] | None = None,
        guided_context: GuidedLearningContext | None = None,
        include_fallback: bool = True,
        batch_size: int = 5,
    ) -> list[Formula]:
        """
        生成一批受限 LTLf 骨架。

        受限骨架阶段不生成 X、F、G，也不做轨迹正确性判断。
        """

        traces = traces or []
        feedback_history = feedback_history or []
        if semantic.faithful_ir is not None:
            candidate_set = semantic.candidate_ir_set
            if (
                candidate_set is not None
                and len(candidate_set.branches) > 1
                and not (
                    semantic.approved_semantic_ir is not None
                    and semantic.approved_semantic_ir.approved
                )
            ):
                skeletons: list[Formula] = []
                for branch in candidate_set.branches:
                    skeleton = compile_ir_skeleton(branch.faithful_ir)
                    if skeleton not in skeletons:
                        skeletons.append(skeleton)
                return skeletons
            return [compile_ir_skeleton(semantic.faithful_ir)]
        effective_requirement = finite_requirement_for_pipeline(requirement, semantic)
        trace_context = _positive_negative_trace_summary(traces)
        allowed_ap_text = ", ".join(_allowed_aps(semantic, traces))
        prompt_context = build_stage_prompt_context(
            self.retriever,
            stage="restricted_skeleton_generation",
            query_parts=(
                "受限骨架生成",
                effective_requirement,
                semantic.pattern,
                semantic.trigger,
                semantic.response,
                allowed_ap_text,
                trace_context,
                "\n".join(feedback_history),
                _guided_query_text(guided_context, "skeleton"),
            ),
            domain_sources=(KB_SKELETON_DIAGNOSTICS, KB_PATTERNS),
            example_sources=(
                FULL_PROCESS_SOURCE,
                GUIDED_FULL_PROCESS_SOURCE,
                GUIDED_SKELETON_SOURCE,
                PATTERN_SOURCE,
            ),
            agent_function_contract=_restricted_skeleton_contract(),
        )
        rag_context = prompt_context.as_rag_context()

        learned_handler = get_pattern_handler_for_semantic(semantic)
        if (
            learned_handler is not None
            and semantic.pattern != "response"
            and learned_pattern_direct_execution_allowed(semantic)
        ):
            learned_skeletons = _ensure_min_restricted_skeletons(
                learned_handler.generate_skeletons(semantic),
                semantic=semantic,
                traces=traces,
                include_fallback=include_fallback,
            )
            if learned_skeletons:
                return learned_skeletons

        if self.llm_enabled:
            return self._generate_with_llm(
                requirement=requirement,
                finite_requirement=effective_requirement,
                semantic=semantic,
                traces=traces,
                trace_context=trace_context,
                allowed_ap_text=allowed_ap_text,
                feedback_history=feedback_history,
                prompt_context=prompt_context,
                guided_context=guided_context,
                include_fallback=include_fallback,
                batch_size=batch_size,
            )

        guided_skeletons = _guided_restricted_skeletons(
            guided_context=guided_context,
            semantic=semantic,
            traces=traces,
        )
        if guided_skeletons:
            return _ensure_min_restricted_skeletons(
                guided_skeletons,
                semantic=semantic,
                traces=traces,
            )

        semantic_skeletons = _semantic_restricted_skeletons(
            semantic=semantic,
            traces=traces,
        )
        if semantic_skeletons:
            return _ensure_min_restricted_skeletons(
                semantic_skeletons,
                semantic=semantic,
                traces=traces,
            )

        trigger_name, response_name = _trigger_response_for_variants(semantic, traces)
        trigger = Atom(trigger_name)
        response = Atom(response_name)

        if self._has_failure_feedback(feedback_history, rag_context):
            return _ensure_min_restricted_skeletons(
                [
                    Implies(trigger, response),
                    Implies(Not(response), Not(trigger)),
                    Not(And(trigger, Not(response))),
                    Or(Not(trigger), response),
                ],
                semantic=semantic,
                traces=traces,
            )

        return _ensure_min_restricted_skeletons(
                [
                    Implies(trigger, response),
                    Implies(Not(response), Not(trigger)),
                    Not(And(trigger, Not(response))),
                    Or(Not(trigger), response),
                    response,
                    And(trigger, response),
                Implies(response, trigger),
                Or(trigger, response),
                Not(And(trigger, response)),
            ],
            semantic=semantic,
            traces=traces,
        )

    def _generate_with_llm(
        self,
        requirement: str,
        finite_requirement: str,
        semantic: SemanticStructure,
        traces: list[LabeledTrace],
        trace_context: str,
        allowed_ap_text: str,
        feedback_history: list[str],
        prompt_context,
        guided_context: GuidedLearningContext | None = None,
        include_fallback: bool = True,
        batch_size: int = 5,
    ) -> list[Formula]:
        extra_rules = (
            (
                guided_prompt_rule("restricted_skeleton_generation"),
                "expected_restricted_skeletons 只用于选择 NL 支持的布尔结构；除非本阶段 contract 允许，不要输出具体 X/F/G。",
            )
            if guided_context is not None
            else ()
        )
        system_prompt = structured_system_prompt(
            agent_name="受限骨架生成 Agent",
            stage_name="restricted_skeleton_generation",
            extra_rules=(
                f"只返回当前最有根据的 Top-{batch_size} 个受限 LTLf 骨架；不要超过 {batch_size} 个 skeleton。",
                "如果 feedback_history 中列出上一批失败原因或已有候选，不要重复上一批；优先修复失败原因。",
                "受限骨架禁止使用 X、F、G；不要在本阶段补全具体时序算子。",
                "公式叶子元素只能使用 allowed_aps 中列出的 AP；禁止使用 true、false、未知变量或占位符。",
                "公式结构只可使用 !、&、|、->、括号；必要时可保留 U。",
                "否定 ! 不得直接重复或仅隔括号嵌套重复；禁止 !!p 和 !(!p)。",
                "布尔连接符必须统一使用 & 和 |，不要输出 && 或 ||。",
                "正轨迹、反轨迹、NL 和 RAG 示例只作为 AP 与括号/布尔组合参考；不要判断骨架是否正确。",
                "优先覆盖 NL 直接支持的蕴含/析取/合取结构；只加入少量等价改写。",
                "不要生成 temporal holes，不要补全候选公式，不要给候选评分。",
                *extra_rules,
            ),
        )
        rag_context = prompt_context.as_rag_context()
        payload = {
            "task": "restricted_skeleton_generation",
            "requirement": finite_requirement,
            "original_requirement": requirement,
            "trace_domain": semantic.trace_domain,
            "domain_warning": semantic.domain_warning,
            "semantic": {
                "trigger": semantic.trigger,
                "response": semantic.response,
                "pattern": semantic.pattern,
                "aps": list(semantic.aps),
            },
            "allowed_aps": allowed_ap_text,
            "positive_negative_trace_summary": trace_context,
            "feedback_history": feedback_history,
            "guided_learning": (
                guided_context.prompt_payload("restricted_skeleton_generation")
                if guided_context is not None
                else None
            ),
            **prompt_context.payload_fields(),
            "rag_context": rag_context,
            "output_schema": {
                "skeletons": [
                    "request -> grant",
                    "!(request & !grant)",
                    "!request | grant",
                ],
                "max_items": batch_size,
            },
        }
        raw_result = self.complete_json(system_prompt, payload, max_tokens=1200)
        raw_skeletons = coerce_list_field(
            raw_result,
            "skeletons",
            aliases=("candidates", "formulas"),
        )

        skeletons: list[Formula] = []
        seen: set[str] = set()
        for index, raw_skeleton in enumerate(raw_skeletons, start=1):
            if not isinstance(raw_skeleton, str) or not raw_skeleton.strip():
                raise ValueError(f"skeletons[{index}] 必须是非空公式字符串。")
            validate_canonical_boolean_operators(
                raw_skeleton,
                context=f"skeletons[{index}]",
            )
            formula = parse_formula(
                normalize_boolean_operator_text(raw_skeleton.strip())
            )
            if _contains_forbidden_temporal(formula):
                raise ValueError(
                    f"skeletons[{index}] 不是受限骨架，包含 X/F/G: {raw_skeleton}"
                )
            formula_key = str(formula)
            if formula_key not in seen:
                seen.add(formula_key)
                skeletons.append(formula)

        parsed_skeletons = list(skeletons)
        skeletons = _ensure_min_restricted_skeletons(
            skeletons,
            semantic=semantic,
            traces=traces,
            include_fallback=include_fallback,
            maximum=max(batch_size, 1) if not include_fallback else 8,
        )
        if not skeletons:
            if parsed_skeletons:
                return parsed_skeletons
            raise ValueError("LLM 未返回任何可用受限骨架。")
        return skeletons

    def _has_failure_feedback(
        self,
        feedback_history: list[str],
        rag_context: tuple[str, ...],
    ) -> bool:
        """判断当前轮是否已经收到骨架失败诊断。"""

        combined = "\n".join([*feedback_history, *rag_context]).lower()
        return any(
            marker in combined
            for marker in (
                "missing_component",
                "logic_operator_error",
                "redundant_component",
                "forbidden_temporal_operator",
                "non_ap_element",
                "unknown_ap",
                "repeated_negation",
                "当前可进入补全阶段的受限骨架数量为 0",
            )
        )


def _restricted_skeleton_contract() -> dict:
    return {
        "agent": "SkeletonAgent",
        "stage": "restricted_skeleton_generation",
        "responsibilities": [
            "Generate parseable restricted skeleton formulas for the accepted semantic structure.",
            "Generate as many distinct skeletons as practical using NL, extracted APs, positive traces, negative traces, and RAG examples as references.",
            "Preserve only allowed AP leaves and Boolean/non-XFG structure supported by the requirement and semantic fields.",
            "Use Top-5 examples only as skeleton-shape references.",
        ],
        "forbidden": [
            "Do not use X, F, or G in restricted skeletons.",
            "Do not use true, false, unknown variables, trigger placeholders, response placeholders, or any token outside allowed_aps as a formula leaf.",
            "Do not output && or ||; use & and | only.",
            "Do not repeat or directly nest negation; !!p and !(!p) are invalid.",
            "Do not generate traces, hole candidates, completed formulas, or rankings.",
            "Do not change semantic trigger, response, pattern, or components.",
        ],
    }


def _contains_forbidden_temporal(formula: Formula) -> bool:
    if formula.op in {"X", "F", "G"}:
        return True
    return any(_contains_forbidden_temporal(child) for child in formula.args)


def _guided_restricted_skeletons(
    *,
    guided_context: GuidedLearningContext | None,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
) -> list[Formula]:
    if guided_context is None:
        return []
    allowed = set(_allowed_aps(semantic, traces))
    skeletons: list[Formula] = []
    seen: set[str] = set()
    for text in guided_context.expected_restricted_skeletons:
        try:
            formula = parse_formula(normalize_boolean_operator_text(text))
        except ValueError:
            continue
        if _contains_forbidden_temporal(formula):
            continue
        atoms = _formula_atoms(formula)
        if atoms and any(atom not in allowed for atom in atoms):
            continue
        key = str(formula)
        if key not in seen:
            seen.add(key)
            skeletons.append(formula)
    return skeletons


def _formula_atoms(formula: Formula) -> tuple[str, ...]:
    if formula.op == "ap":
        return (formula.name,)
    atoms: list[str] = []
    for child in formula.args:
        atoms.extend(_formula_atoms(child))
    return tuple(dict.fromkeys(atoms))


def _semantic_restricted_skeletons(
    *,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
) -> list[Formula]:
    allowed_aps = _allowed_aps(semantic, traces)
    texts = _semantic_fallback_skeleton_texts(semantic, allowed_aps)
    texts.extend(_component_restricted_skeleton_texts(semantic))

    skeletons: list[Formula] = []
    seen: set[str] = set()
    allowed = set(allowed_aps)
    for text in texts:
        try:
            formula = _strip_xfg(parse_formula(normalize_boolean_operator_text(text)))
        except ValueError:
            continue
        if _contains_forbidden_temporal(formula):
            continue
        if _contains_non_ap_literal(formula):
            continue
        if _collect_atoms(formula) - allowed:
            continue
        key = str(formula)
        if key not in seen:
            seen.add(key)
            skeletons.append(formula)
    return skeletons


def _component_restricted_skeleton_texts(semantic: SemanticStructure) -> list[str]:
    components = dict(semantic.components)
    texts: list[str] = []

    def add(value: str) -> None:
        value = _component_rhs(value)
        if value and value not in texts:
            texts.append(value)

    for key in (
        "operand",
        "formula",
        "formula_fragment",
        "condition",
        "constraint",
        "left_component",
        "right_component",
        "left_condition",
        "right_condition",
        "trigger_formula",
        "response_formula",
        "trigger_definition",
        "response_definition",
    ):
        add(components.get(key, ""))

    left = (
        _component_rhs(components.get("left_component", ""))
        or _component_rhs(components.get("left_condition", ""))
    )
    right = (
        _component_rhs(components.get("right_component", ""))
        or _component_rhs(components.get("right_condition", ""))
    )
    pattern = semantic.pattern.lower()
    if left and right:
        if "conjunction" in pattern:
            add(f"({left}) & ({right})")
        elif "disjunction" in pattern:
            add(f"({left}) | ({right})")
        elif "implication" in pattern or "response" in pattern:
            add(f"({left}) -> ({right})")
    return texts


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


def _allowed_aps(
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
) -> tuple[str, ...]:
    aps: list[str] = []
    for ap in semantic.aps:
        if ap and ap not in aps:
            aps.append(ap)
    if aps:
        return tuple(aps)
    for ap in (semantic.trigger, semantic.response):
        if ap and ap != "unknown_ap" and ap not in aps:
            aps.append(ap)
    if aps:
        return tuple(aps)
    for trace in traces:
        if trace.kind not in {"positive", "negative"}:
            continue
        for step in trace.trace:
            for ap in sorted(step):
                if ap not in aps:
                    aps.append(ap)
    return tuple(aps)


def _trigger_response_for_variants(
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
) -> tuple[str, str]:
    aps = _allowed_aps(semantic, traces)
    trigger_component = _restricted_component_formula_text(
        semantic,
        "trigger_definition",
        "trigger_formula",
    )
    response_component = _restricted_component_formula_text(
        semantic,
        "response_formula",
        "response_definition",
    )
    trigger = (
        trigger_component
        or (semantic.trigger if semantic.trigger in aps else (aps[0] if aps else "request"))
    )
    response = response_component or (
        semantic.response
        if semantic.response in aps
        else (aps[1] if len(aps) > 1 else trigger)
    )
    return trigger, response


def _restricted_component_formula_text(
    semantic: SemanticStructure,
    *keys: str,
) -> str:
    components = dict(semantic.components)
    for key in keys:
        raw_value = components.get(key, "")
        text = _component_rhs(raw_value)
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
    match = re.search(r"(?::=|=)", text)
    if match is not None:
        text = text[match.end() :].strip()
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


def _positive_negative_trace_summary(traces: list[LabeledTrace]) -> str:
    lines: list[str] = []
    for trace in traces:
        if trace.kind not in {"positive", "negative"}:
            continue
        rendered_steps = []
        for index, step in enumerate(trace.trace):
            events = "{" + ", ".join(sorted(step)) + "}" if step else "{}"
            rendered_steps.append(f"t{index}={events}")
        lines.append(
            f"{trace.kind} {trace.name}: "
            + "; ".join(rendered_steps)
            + f"; expected={trace.expected}; reason={trace.reason}"
        )
    return "\n".join(lines)


def _ensure_min_restricted_skeletons(
    skeletons: list[Formula],
    *,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    minimum: int = 5,
    maximum: int = 8,
    include_fallback: bool = True,
) -> list[Formula]:
    del minimum
    result: list[Formula] = []
    seen: set[str] = set()
    allowed_aps_tuple = _allowed_aps(semantic, traces)
    allowed_aps = set(allowed_aps_tuple)

    def add(formula: Formula) -> None:
        if len(result) >= maximum:
            return
        if _contains_forbidden_temporal(formula):
            return
        if has_repeated_negation(formula):
            return
        if _contains_non_ap_literal(formula):
            return
        if _collect_atoms(formula) - allowed_aps:
            return
        key = str(formula)
        if key in seen:
            return
        seen.add(key)
        result.append(formula)

    for skeleton in skeletons:
        add(skeleton)

    if include_fallback:
        semantic_fallbacks = _semantic_restricted_skeletons(
            semantic=semantic,
            traces=traces,
        )
        for formula in semantic_fallbacks:
            add(formula)

        for text in _semantic_fallback_skeleton_texts(semantic, allowed_aps_tuple):
            try:
                add(parse_formula(text))
            except ValueError:
                continue

        if semantic.pattern == "response" or not semantic_fallbacks:
            trigger, response = _trigger_response_for_variants(semantic, traces)
            for text in _fallback_skeleton_texts(trigger, response, allowed_aps_tuple):
                try:
                    add(parse_formula(text))
                except ValueError:
                    continue

    return result


def _contains_non_ap_literal(formula: Formula) -> bool:
    if formula.op in {"true", "false"}:
        return True
    return any(_contains_non_ap_literal(child) for child in formula.args)


def _collect_atoms(formula: Formula) -> set[str]:
    if formula.op == "ap":
        return {formula.name}
    atoms: set[str] = set()
    for child in formula.args:
        atoms.update(_collect_atoms(child))
    return atoms


def _semantic_fallback_skeleton_texts(
    semantic: SemanticStructure,
    aps: tuple[str, ...],
) -> list[str]:
    texts: list[str] = []

    def add(value: str) -> None:
        value = normalize_boolean_operator_text(value.strip())
        if value and value not in texts:
            texts.append(value)

    semantic_text = _semantic_evidence_text(semantic)
    pattern = semantic.pattern.strip().lower().replace("-", "_").replace(" ", "_")
    if pattern == "precedence":
        constrained, prerequisite = precedence_roles_from_semantic(semantic)
        if constrained and prerequisite and constrained != prerequisite:
            add(f"!{constrained}")
            add(f"!{constrained} U ({prerequisite} & !{constrained})")
            add(f"!{constrained} | (!{constrained} U ({prerequisite} & !{constrained}))")
    elif pattern in {"recurrence", "periodic"}:
        ap = _primary_semantic_ap(semantic)
        if ap:
            add(ap)
            add(f"{ap} | !{ap}")
    elif pattern == "cardinality_constraint":
        ap = _primary_semantic_ap(semantic)
        if ap:
            counted = f"!{ap}" if _cardinality_counts_absence(semantic_text, ap) else ap
            add(counted)
            add(f"!({counted} & {counted})")
            add(f"{ap} | !{ap}")
    elif len(aps) >= 2 and _looks_like_mutual_exclusion(semantic_text):
        left, right = aps[0], aps[1]
        add(f"!({left} & {right})")
        add(f"!({right} & {left})")
        add(f"!{left} | !{right}")
        add(f"!{right} | !{left}")
        if _looks_like_coverage(semantic_text):
            add(f"!({left} & {right}) & ({left} | {right})")
            add(f"({left} | {right}) & !({left} & {right})")
            add(f"({left} | {right})")
    elif len(aps) >= 2 and _looks_like_coverage(semantic_text):
        add(" | ".join(aps[: min(len(aps), 4)]))

    return texts


def _primary_semantic_ap(semantic: SemanticStructure) -> str:
    for ap in (*semantic.aps, semantic.trigger, semantic.response):
        if ap and ap not in {"unknown", "unknown_ap"}:
            return ap
    return ""


def _cardinality_counts_absence(text: str, ap: str) -> bool:
    escaped = re.escape(ap)
    return bool(
        re.search(rf"\bnot\s+{escaped}\b", text)
        or re.search(rf"!\s*{escaped}\b", text)
    )


def _semantic_evidence_text(semantic: SemanticStructure) -> str:
    component_text = " ".join(
        f"{key} {value}" for key, value in semantic.components
    )
    return " ".join(
        [
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


def _fallback_skeleton_texts(
    trigger: str,
    response: str,
    aps: tuple[str, ...],
) -> list[str]:
    texts: list[str] = []

    def add(value: str) -> None:
        if value not in texts:
            texts.append(value)

    def atom_variants(ap: str) -> list[str]:
        return [ap, f"({ap})", f"(({ap}))"]

    for trigger_text in atom_variants(trigger):
        for response_text in atom_variants(response):
            add(f"{trigger_text} -> {response_text}")
            add(f"({trigger_text} -> {response_text})")
            add(f"!{trigger_text} | {response_text}")
            add(f"({response_text} | !{trigger_text})")
            add(f"!({trigger_text} & !{response_text})")
            add(f"!(({trigger_text}) & (!{response_text}))")
            add(f"{trigger_text} & {response_text}")
            add(f"({trigger_text} & {response_text})")
            add(f"{trigger_text} | {response_text}")
            add(f"({trigger_text} | {response_text})")
            add(f"!({trigger_text} & {response_text})")

    if len(aps) >= 2:
        for left_index, left in enumerate(aps):
            for right in aps[left_index + 1 :]:
                add(f"{left} & {right}")
                add(f"({left} & {right})")
                add(f"{right} & {left}")
                add(f"{left} | {right}")
                add(f"({left} | {right})")
                add(f"{right} | {left}")
                add(f"!({left} & {right})")
                add(f"!{left} | !{right}")
                add(f"{left} -> {right}")
                add(f"{right} -> {left}")
                add(f"!({left} & !{right})")
                add(f"!({right} & !{left})")
    if len(aps) >= 3:
        for index, consequent in enumerate(aps):
            antecedents = [ap for ap_index, ap in enumerate(aps) if ap_index != index]
            if len(antecedents) < 2:
                continue
            left, right = antecedents[0], antecedents[1]
            add(f"({left} & {right}) -> {consequent}")
            add(f"(({left}) & ({right})) -> ({consequent})")
            add(f"!({left} & {right} & !{consequent})")
            add(f"!(({left} & {right}) & !{consequent})")
            add(f"({left} | {right}) -> {consequent}")
            add(f"!({left} | {right}) | {consequent}")
        if len(aps) >= 4:
            joined_and = " & ".join(aps[:4])
            joined_or = " | ".join(aps[:4])
            add(f"({joined_and})")
            add(f"({joined_or})")
    return texts
