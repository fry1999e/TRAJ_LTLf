"""Learned pattern registry for Traj-LTLf.

Unknown patterns are accepted only when they are learned from curated warmup
examples. Runtime requirements may reuse learned handlers, but they do not
create new handlers from LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re

from data_examples import LabeledTrace
from ltlf_parser import Formula, parse_formula
from ltlf_syntax import (
    normalize_boolean_operator_text,
    validate_no_repeated_negation,
    validate_supported_ltlf_formula_text,
)
from schemas import (
    HoleCandidate,
    SemanticStructure,
    SkeletonDiagnostic,
    TemporalCompletion,
    TemporalHole,
)


@dataclass(frozen=True)
class PatternSpec:
    """A pattern learned from one curated warmup example."""

    pattern: str
    source_id: str
    requirement: str
    formula_text: str
    semantic_fields: tuple[tuple[str, str], ...] = ()
    derivation_rules: tuple[str, ...] = ()
    trace_count: int = 0
    skeleton_templates: tuple[str, ...] = ()
    hole_template_text: str = ""
    final_candidate_text: str = ""


@dataclass(frozen=True)
class PatternTemplateDerivation:
    """A learned warmup formula represented as a temporal-hole template."""

    template_text: str
    holes: tuple[TemporalHole, ...]
    assignments: tuple[tuple[str, str], ...]
    operator_derivation: tuple[str, ...]


@dataclass
class LearnedPatternHandler:
    """Handler backed by curated examples for one concrete pattern."""

    pattern: str
    specs: list[PatternSpec] = field(default_factory=list)

    def add_spec(self, spec: PatternSpec) -> None:
        if spec.source_id not in {item.source_id for item in self.specs}:
            self.specs.append(spec)

    def generate_skeletons(self, semantic: SemanticStructure) -> list[Formula]:
        spec = self._spec_for_semantic(semantic)
        skeleton_texts = _learned_skeleton_texts(spec)
        return _unique_formulas(parse_learned_formula(text) for text in skeleton_texts)

    def validate_skeleton(
        self,
        skeleton: Formula,
        traces: list[LabeledTrace],
        semantic: SemanticStructure,
    ) -> SkeletonDiagnostic:
        del traces
        spec = self._spec_for_semantic(semantic)
        expected = {
            str(parse_learned_formula(text))
            for text in _learned_skeleton_texts(spec)
        }
        if str(skeleton) in expected:
            return SkeletonDiagnostic(
                skeleton=skeleton,
                passed=True,
                error_type="none",
                trace_evidence=(
                    f"骨架来自 warmup 样例 {spec.source_id} 的已审核 {self.pattern} 模式。",
                ),
                suggestion="该 learned pattern 骨架可进入补全阶段。",
            )
        return SkeletonDiagnostic(
            skeleton=skeleton,
            passed=False,
            error_type="logic_operator_error",
            trace_evidence=(
                f"骨架不匹配 learned {self.pattern} 模式的人工审核模板。",
            ),
            suggestion="使用 warmup 学到的模式模板，不要让 LLM 自由改写。",
        )

    def generate_hole_candidates(
        self,
        semantic: SemanticStructure,
        skeletons: list[Formula],
        rag_context: tuple[str, ...] = (),
    ) -> list[HoleCandidate]:
        spec = self._spec_for_semantic(semantic)
        derivations = derive_pattern_spec_template_derivations(spec)
        return [
            HoleCandidate(
                skeleton=skeleton,
                template_text=derivation.template_text,
                holes=derivation.holes,
                rag_context=rag_context,
                explanation=(
                    f"使用 warmup 样例 {spec.source_id} 学到的 "
                    f"{self.pattern} 模式模板；每个 H_xxx 孔位仍需显式记录补全算子。"
                ),
            )
            for skeleton in skeletons
            for derivation in derivations
        ]

    def complete_temporal_holes(
        self,
        semantic: SemanticStructure,
        hole_candidates: list[HoleCandidate],
    ) -> list[TemporalCompletion]:
        spec = self._spec_for_semantic(semantic)
        formula_text = spec.final_candidate_text or spec.formula_text
        fallback_formula = parse_learned_formula(formula_text)
        completions: list[TemporalCompletion] = []
        for hole_candidate in hole_candidates:
            derivation = _derivation_for_template(spec, hole_candidate.template_text)
            try:
                formula = _render_formula_from_template(
                    hole_candidate.template_text,
                    derivation.assignments,
                )
            except ValueError:
                formula = fallback_formula
            completions.append(
                TemporalCompletion(
                    formula=formula,
                    assignments=derivation.assignments,
                    operator_derivation=derivation.operator_derivation,
                    candidate_id=hole_candidate.candidate_id,
                    template_text=hole_candidate.template_text,
                )
            )
        return completions

    def _spec_for_semantic(self, semantic: SemanticStructure) -> PatternSpec:
        if semantic.formula_text:
            return PatternSpec(
                pattern=self.pattern,
                source_id="runtime-semantic",
                requirement="",
                formula_text=semantic.formula_text,
                semantic_fields=semantic.components,
                derivation_rules=(),
                trace_count=0,
                skeleton_templates=(semantic.formula_text,),
                hole_template_text=semantic.formula_text,
                final_candidate_text=semantic.formula_text,
            )
        if not self.specs:
            raise ValueError(f"模式 {self.pattern} 尚未从 warmup 样例学习。")

        allowed_atoms = set(semantic.aps)
        compatible_specs: list[PatternSpec] = []
        for spec in self.specs:
            formula_text = spec.final_candidate_text or spec.formula_text
            try:
                formula_atoms = set(
                    collect_formula_atoms(parse_learned_formula(formula_text))
                )
            except ValueError:
                continue
            if allowed_atoms and not formula_atoms.issubset(allowed_atoms):
                continue
            compatible_specs.append(spec)

        if len(compatible_specs) == 1:
            return compatible_specs[0]

        semantic_fields = dict(semantic.components)
        scored_specs: list[tuple[int, PatternSpec]] = []
        for spec in compatible_specs:
            spec_fields = dict(spec.semantic_fields)
            score = sum(
                2 if spec_fields.get(key) == value else 1
                for key, value in semantic_fields.items()
                if key in spec_fields
            )
            scored_specs.append((score, spec))
        if scored_specs:
            best_score = max(score for score, _spec in scored_specs)
            best_specs = [
                spec for score, spec in scored_specs if score == best_score
            ]
            if best_score > 0 and len(best_specs) == 1:
                return best_specs[0]

        raise ValueError(
            f"模式 {self.pattern} 的 warmup PatternSpec 只能作为运行时参考；"
            "没有唯一且与当前 AP/语义字段兼容的样例，不能直接执行某个样例公式。"
        )


class PatternRegistry:
    """In-memory registry populated by curated warmup examples."""

    def __init__(self) -> None:
        self._handlers: dict[str, LearnedPatternHandler] = {}

    def register(self, spec: PatternSpec) -> LearnedPatternHandler:
        spec = _normalize_pattern_spec(spec)
        pattern = normalize_pattern(spec.pattern)
        handler = self._handlers.setdefault(pattern, LearnedPatternHandler(pattern))
        handler.add_spec(spec)
        return handler

    def register_many(self, specs: list[PatternSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def get(self, pattern: str) -> LearnedPatternHandler | None:
        return self._handlers.get(normalize_pattern(pattern))

    def require(self, pattern: str) -> LearnedPatternHandler:
        handler = self.get(pattern)
        if handler is None:
            raise ValueError(
                f"模式 {pattern} 未通过 warmup 样例学习，不能由 LLM 动态创建 handler。"
            )
        return handler

    def specs(self) -> list[PatternSpec]:
        items: list[PatternSpec] = []
        for handler in self._handlers.values():
            items.extend(handler.specs)
        return items


GLOBAL_PATTERN_REGISTRY = PatternRegistry()


def normalize_pattern(pattern: str) -> str:
    return pattern.strip().lower().replace("-", "_").replace(" ", "_")


def register_pattern_specs(specs: list[PatternSpec]) -> None:
    GLOBAL_PATTERN_REGISTRY.register_many(specs)


def get_pattern_handler(pattern: str) -> LearnedPatternHandler | None:
    return GLOBAL_PATTERN_REGISTRY.get(pattern)


def get_pattern_handler_for_semantic(
    semantic: SemanticStructure,
) -> LearnedPatternHandler | None:
    """Return a registered or runtime formula_text-backed handler."""

    handler = get_pattern_handler(semantic.pattern)
    if handler is not None:
        return handler
    if semantic.pattern != "response" and semantic.formula_text.strip():
        return LearnedPatternHandler(normalize_pattern(semantic.pattern))
    return None


def learned_pattern_direct_execution_allowed(
    semantic: SemanticStructure,
) -> bool:
    """Return whether a learned handler may execute the current formula directly."""

    formula_text = semantic.formula_text.strip()
    if not formula_text:
        return False
    try:
        formula = parse_learned_formula(formula_text)
    except ValueError:
        return False

    allowed_atoms = set(semantic.aps)
    if not allowed_atoms:
        for atom in (semantic.trigger, semantic.response):
            if atom and atom not in {"unknown", "unknown_ap"}:
                allowed_atoms.add(atom)
    return not (set(collect_formula_atoms(formula)) - allowed_atoms)


def require_pattern_handler(pattern: str) -> LearnedPatternHandler:
    return GLOBAL_PATTERN_REGISTRY.require(pattern)


def registered_pattern_specs() -> list[PatternSpec]:
    """Return learned pattern specs currently available in memory."""

    return GLOBAL_PATTERN_REGISTRY.specs()


def is_builtin_pattern(pattern: str) -> bool:
    """Return whether the pattern is supported without curated learning."""

    return normalize_pattern(pattern) in {
        "response",
        "mutual_exclusion",
        "mutual_exclusion_with_coverage",
        "precedence",
        "recurrence",
        "cardinality_constraint",
        "periodic",
    }


def is_pattern_learned(pattern: str) -> bool:
    """Return whether a pattern can be handled by the current registry."""

    normalized = normalize_pattern(pattern)
    return is_builtin_pattern(normalized) or GLOBAL_PATTERN_REGISTRY.get(normalized) is not None


def pattern_requires_human_learning(semantic: SemanticStructure) -> bool:
    """Return True when a formal requirement names an unsupported pattern."""

    return not is_pattern_learned(semantic.pattern)


def parse_learned_formula(text: str) -> Formula:
    normalized = normalize_formula_text(text)
    validate_supported_ltlf_formula_text(normalized)
    formula = parse_formula(normalized)
    validate_no_repeated_negation(formula, context="formula_text")
    return formula


def normalize_formula_text(text: str) -> str:
    normalized = text.strip()
    normalized = normalize_boolean_operator_text(normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _normalize_pattern_spec(spec: PatternSpec) -> PatternSpec:
    return replace(
        spec,
        formula_text=normalize_formula_text(spec.formula_text),
        skeleton_templates=tuple(
            normalize_formula_text(text) for text in spec.skeleton_templates
        ),
        hole_template_text=normalize_formula_text(spec.hole_template_text)
        if spec.hole_template_text
        else "",
        final_candidate_text=normalize_formula_text(spec.final_candidate_text)
        if spec.final_candidate_text
        else "",
    )


def collect_formula_atoms(formula: Formula) -> tuple[str, ...]:
    atoms: set[str] = set()

    def visit(node: Formula) -> None:
        if node.op == "ap":
            atoms.add(node.name)
            return
        for child in node.args:
            visit(child)

    visit(formula)
    return tuple(sorted(atoms))


def derive_pattern_spec_template_derivations(
    spec: PatternSpec,
) -> tuple[PatternTemplateDerivation, ...]:
    """Derive temporal-hole templates and H_xxx assignments from a warmup spec."""

    normalized_spec = _normalize_pattern_spec(spec)
    formula = parse_learned_formula(
        normalized_spec.final_candidate_text or normalized_spec.formula_text
    )
    base = _formula_to_hole_template_derivation(formula)

    derivations: list[PatternTemplateDerivation] = []
    seen: set[str] = set()

    def add(derivation: PatternTemplateDerivation) -> None:
        template = normalize_formula_text(derivation.template_text)
        if not template or template in seen:
            return
        seen.add(template)
        derivations.append(replace(derivation, template_text=template))

    add(base)

    for candidate_text in (
        normalized_spec.hole_template_text,
        *normalized_spec.skeleton_templates,
    ):
        derivation = _template_derivation_from_text(candidate_text, base)
        if derivation is not None:
            add(derivation)

    for variant in _template_parenthesis_variants(base.template_text):
        add(replace(base, template_text=variant))

    return tuple(derivations)


def _learned_skeleton_texts(spec: PatternSpec) -> tuple[str, ...]:
    texts: list[str] = []

    def add(text: str) -> None:
        if not text:
            return
        normalized = normalize_formula_text(text)
        if normalized not in texts:
            texts.append(normalized)

    source_texts = spec.skeleton_templates or (spec.formula_text,)
    for source_text in source_texts:
        try:
            stripped = _strip_concrete_temporal(parse_learned_formula(source_text))
        except ValueError:
            continue
        stripped_text = str(stripped)
        add(stripped_text)
        add(f"({stripped_text})")

    if not texts:
        try:
            stripped = _strip_concrete_temporal(parse_learned_formula(spec.formula_text))
            add(str(stripped))
        except ValueError:
            pass
    return tuple(texts)


def _learned_template_texts(spec: PatternSpec) -> tuple[str, ...]:
    return tuple(
        derivation.template_text
        for derivation in derive_pattern_spec_template_derivations(spec)
    )


def _template_derivation_from_text(
    text: str,
    base: PatternTemplateDerivation,
) -> PatternTemplateDerivation | None:
    if not text:
        return None
    normalized = normalize_formula_text(text)
    if _contains_hole_id(normalized):
        holes = _holes_for_template(normalized, base)
        assignments = _assignments_for_holes(holes, base.assignments)
        return PatternTemplateDerivation(
            template_text=normalized,
            holes=holes,
            assignments=assignments,
            operator_derivation=_operator_derivation_lines(holes, assignments),
        )
    try:
        formula = parse_learned_formula(normalized)
    except ValueError:
        return None
    return _formula_to_hole_template_derivation(formula)


def _formula_to_hole_template_derivation(
    formula: Formula,
) -> PatternTemplateDerivation:
    assignments: list[tuple[str, str]] = []
    hole_targets: dict[str, str] = {}
    counters = {"temporal": 1}
    template = _formula_to_hole_template_text(
        formula,
        is_root=True,
        assignments=assignments,
        hole_targets=hole_targets,
        counters=counters,
    )
    holes = tuple(
        TemporalHole(
            hole_id=hole_id,
            target=hole_targets.get(hole_id, "formula fragment"),
            allowed_ops=("epsilon", "X", "F", "G", "XF"),
            reason="从 warmup 最终公式中的具体时序算子反推得到。",
        )
        for hole_id, _operator in assignments
    )
    return PatternTemplateDerivation(
        template_text=template,
        holes=holes,
        assignments=tuple(assignments),
        operator_derivation=_operator_derivation_lines(holes, tuple(assignments)),
    )


def _formula_to_hole_template_text(
    formula: Formula,
    *,
    is_root: bool,
    assignments: list[tuple[str, str]],
    hole_targets: dict[str, str],
    counters: dict[str, int],
) -> str:
    if formula.op in {"X", "F", "G"}:
        operators, inner = _collect_temporal_chain(formula)
        hole_id = _next_hole_id(
            is_root=is_root,
            assignments=assignments,
            counters=counters,
        )
        operator = _operator_from_chain(operators)
        assignments.append((hole_id, operator))
        hole_targets[hole_id] = str(_strip_concrete_temporal(inner))
        inner_text = _formula_to_hole_template_text(
            inner,
            is_root=False,
            assignments=assignments,
            hole_targets=hole_targets,
            counters=counters,
        )
        return f"{hole_id}({inner_text})"

    if formula.op == "ap":
        return formula.name
    if formula.op == "true":
        return "true"
    if formula.op == "false":
        return "false"
    if formula.op == "not":
        child = formula.args[0]
        child_text = _formula_to_hole_template_text(
            child,
            is_root=False,
            assignments=assignments,
            hole_targets=hole_targets,
            counters=counters,
        )
        if child.op in {"and", "or", "implies", "U"}:
            return f"!({child_text})"
        return f"!{child_text}"
    if formula.op in {"and", "or", "implies", "U"}:
        left, right = formula.args
        symbol = {
            "and": " & ",
            "or": " | ",
            "implies": " -> ",
            "U": " U ",
        }[formula.op]
        left_text = _formula_to_hole_template_text(
            left,
            is_root=False,
            assignments=assignments,
            hole_targets=hole_targets,
            counters=counters,
        )
        right_text = _formula_to_hole_template_text(
            right,
            is_root=False,
            assignments=assignments,
            hole_targets=hole_targets,
            counters=counters,
        )
        if left.op == "implies":
            left_text = f"({left_text})"
        if right.op in {"and", "or", "implies", "U"}:
            right_text = f"({right_text})"
        return f"{left_text}{symbol}{right_text}"
    return str(formula)


def _next_hole_id(
    *,
    is_root: bool,
    assignments: list[tuple[str, str]],
    counters: dict[str, int],
) -> str:
    used = {hole_id for hole_id, _operator in assignments}
    if is_root and "H_scope" not in used:
        return "H_scope"
    if "H_response" not in used:
        return "H_response"
    while True:
        hole_id = f"H_temporal_{counters['temporal']}"
        counters["temporal"] += 1
        if hole_id not in used:
            return hole_id


def _collect_temporal_chain(formula: Formula) -> tuple[tuple[str, ...], Formula]:
    operators: list[str] = []
    current = formula
    while current.op in {"X", "F", "G"} and current.args:
        operators.append(current.op)
        current = current.args[0]
    return tuple(operators), current


def _operator_from_chain(operators: tuple[str, ...]) -> str:
    if operators == ("X", "F"):
        return "XF"
    if len(operators) == 1 and operators[0] in {"X", "F", "G"}:
        return operators[0]
    if not operators:
        return "epsilon"
    for operator in operators:
        if operator in {"X", "F", "G"}:
            return operator
    return "epsilon"


def _template_parenthesis_variants(template_text: str) -> tuple[str, ...]:
    return (
        f"({template_text})",
    )


def _contains_hole_id(text: str) -> bool:
    return re.search(r"\bH_[A-Za-z0-9_]*\b", text) is not None


def _holes_for_template(
    template_text: str,
    base: PatternTemplateDerivation,
) -> tuple[TemporalHole, ...]:
    known = {hole.hole_id: hole for hole in base.holes}
    holes: list[TemporalHole] = []
    for match in re.finditer(r"\bH_[A-Za-z0-9_]*\b", template_text):
        hole_id = match.group(0)
        if any(hole.hole_id == hole_id for hole in holes):
            continue
        holes.append(
            known.get(
                hole_id,
                TemporalHole(
                    hole_id=hole_id,
                    target="formula fragment",
                    allowed_ops=("epsilon", "X", "F", "G", "XF"),
                    reason="从 warmup 带孔模板中发现的 H_xxx 孔位。",
                ),
            )
        )
    return tuple(holes)


def _assignments_for_holes(
    holes: tuple[TemporalHole, ...],
    base_assignments: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    base_map = dict(base_assignments)
    return tuple(
        (hole.hole_id, base_map.get(hole.hole_id, "epsilon"))
        for hole in holes
    )


def _operator_derivation_lines(
    holes: tuple[TemporalHole, ...],
    assignments: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    assignment_map = dict(assignments)
    return tuple(
        f"{hole.hole_id}={assignment_map.get(hole.hole_id, 'epsilon')}: "
        f"由 warmup 最终公式、NL 需求、正/反/边界轨迹和 RAG 样例共同确认。"
        for hole in holes
    )


def _derivation_for_template(
    spec: PatternSpec,
    template_text: str,
) -> PatternTemplateDerivation:
    derivations = derive_pattern_spec_template_derivations(spec)
    normalized = normalize_formula_text(template_text)
    for derivation in derivations:
        if derivation.template_text == normalized:
            return derivation
    base = derivations[0]
    holes = _holes_for_template(normalized, base)
    assignments = _assignments_for_holes(holes, base.assignments)
    return PatternTemplateDerivation(
        template_text=normalized,
        holes=holes,
        assignments=assignments,
        operator_derivation=_operator_derivation_lines(holes, assignments),
    )


def _render_formula_from_template(
    template_text: str,
    assignments: tuple[tuple[str, str], ...],
) -> Formula:
    text = normalize_formula_text(template_text)
    assignment_map = dict(assignments)
    text = _render_hole_calls(text, assignment_map)
    text = _render_hole_prefixes(text, assignment_map)
    if _contains_hole_id(text):
        raise ValueError(f"模板仍包含未补全孔位: {text}")
    return parse_learned_formula(text)


def _render_hole_calls(text: str, assignment_map: dict[str, str]) -> str:
    while True:
        best_index = -1
        best_hole = ""
        for hole_id in assignment_map:
            index = text.rfind(f"{hole_id}(")
            if index > best_index:
                best_index = index
                best_hole = hole_id
        if best_index < 0:
            return text

        open_index = best_index + len(best_hole)
        close_index = _matching_paren_index(text, open_index)
        inner = text[open_index + 1 : close_index]
        replacement = _apply_operator_text(inner, assignment_map[best_hole])
        text = text[:best_index] + replacement + text[close_index + 1 :]


def _render_hole_prefixes(text: str, assignment_map: dict[str, str]) -> str:
    for hole_id, operator in assignment_map.items():
        pattern = re.compile(rf"\b{re.escape(hole_id)}([A-Za-z_][A-Za-z0-9_]*)\b")
        text = pattern.sub(
            lambda match: _apply_operator_text(match.group(1), operator),
            text,
        )
    return text


def _matching_paren_index(text: str, open_index: int) -> int:
    if open_index >= len(text) or text[open_index] != "(":
        raise ValueError(f"模板孔位缺少左括号: {text}")
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"模板孔位缺少右括号: {text}")


def _apply_operator_text(inner: str, operator: str) -> str:
    inner = inner.strip()
    if operator == "epsilon":
        return inner
    if operator == "XF":
        return f"X(F({inner}))"
    if operator in {"X", "F", "G"}:
        return f"{operator}({inner})"
    return inner


def _hole_lines(holes: tuple[TemporalHole, ...]) -> tuple[str, ...]:
    return tuple(
        "- "
        f"hole_id={hole.hole_id}; "
        f"target={hole.target}; "
        f"allowed_ops={','.join(hole.allowed_ops)}; "
        f"role={hole.role}; "
        f"parent_hole_id={hole.parent_hole_id}; "
        f"source_hint={','.join(hole.source_hint)}; "
        f"reason={hole.reason}"
        for hole in holes
    )


def _strip_concrete_temporal(formula: Formula) -> Formula:
    if formula.op in {"X", "F", "G"} and formula.args:
        return _strip_concrete_temporal(formula.args[0])
    if not formula.args:
        return formula
    return Formula(
        formula.op,
        tuple(_strip_concrete_temporal(child) for child in formula.args),
        formula.name,
    )


def _unique_formulas(formulas: object) -> list[Formula]:
    result: list[Formula] = []
    seen: set[str] = set()
    for formula in formulas:
        if not isinstance(formula, Formula):
            continue
        key = str(formula)
        if key in seen:
            continue
        seen.add(key)
        result.append(formula)
    return result
