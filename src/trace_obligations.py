"""Semantic obligation analysis and finite-trace evidence classification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import re

from contract_versions import OBLIGATION_COMPILER_VERSION
from data_examples import LabeledTrace, TraceSemantics
from ltlf_checker import check_trace, satisfies
from ltlf_parser import And, Atom, F, Formula, G, Implies, Not, Or, U, X
from pattern_registry import parse_learned_formula
from schemas import SemanticStructure


ROLE_BOUNDARY = "boundary"
ROLE_MINIMUM_WITNESS = "minimum_witness"
ROLE_VACUOUS = "vacuous"
ROLE_SAME_STEP = "same_step"
ROLE_EARLY = "early"
ROLE_EXACT = "exact"
ROLE_LATE = "late"
ROLE_FINAL_STATE = "final_state"
ROLE_FIRST_VIOLATION = "first_violation"
ROLE_MIDDLE_VIOLATION = "middle_violation"
ROLE_FINAL_VIOLATION = "final_violation"
ROLE_SINGLE_TRIGGER = "single_trigger"
ROLE_LATE_TRIGGER = "late_trigger"
ROLE_MULTIPLE_OBLIGATION = "multiple_obligation"
ROLE_MISSING_RESPONSE = "missing_response"
ROLE_LATE_MISSING_RESPONSE = "late_missing_response"
ROLE_GLOBAL_WITNESS = "global_witness"
ROLE_CANDIDATE_DISTINGUISHING = "candidate_distinguishing"


@dataclass(frozen=True)
class ObligationProfile:
    """Executable trace obligations derived from reviewed semantics."""

    scope: str = "current"
    temporal_operator: str = ""
    required_horizon: int | None = None
    required_roles: tuple[str, ...] = ()
    confusion_targets: tuple[str, ...] = ()
    trigger: str = ""
    response: str = ""
    formula: Formula | None = None
    source: str = ""

    @property
    def executable(self) -> bool:
        return self.formula is not None


def analyze_obligations(
    requirement: str,
    semantic: SemanticStructure,
) -> ObligationProfile:
    """Build an obligation profile without depending on candidate generation."""

    evidence = _semantic_evidence(requirement, semantic)
    formula, source = _reference_formula(requirement, semantic, evidence)
    operators = _formula_operators(formula) if formula is not None else set()
    horizon = _max_next_depth(formula) if formula is not None else _text_next_depth(evidence)
    global_scope = (
        (formula is not None and formula.op == "G")
        or _scope_is_global(requirement, semantic)
    )

    temporal_operator = _temporal_operator_name(operators, horizon, evidence)
    roles: list[str] = []
    confusions: list[str] = []

    if semantic.pattern == "response":
        if semantic.trigger == semantic.response:
            roles.extend((ROLE_VACUOUS, ROLE_SAME_STEP))
        else:
            roles.extend(
                (
                    ROLE_SINGLE_TRIGGER,
                    ROLE_LATE_TRIGGER,
                    ROLE_MISSING_RESPONSE,
                    ROLE_LATE_MISSING_RESPONSE,
                )
            )
        if "F" in operators or (formula is None and _looks_eventual(evidence)):
            roles.append(ROLE_SAME_STEP)
            confusions.extend(("F_vs_XF", "local_vs_global_scope"))
        elif horizon:
            roles.extend((ROLE_SAME_STEP, ROLE_EXACT))
            confusions.extend(("same_step_vs_next", "next_depth"))
        if global_scope:
            confusions.append("local_vs_global_scope")
    elif global_scope:
        roles.extend(
            (
                ROLE_GLOBAL_WITNESS,
                ROLE_MIDDLE_VIOLATION,
                ROLE_FINAL_VIOLATION,
            )
        )
        confusions.append("current_vs_global_scope")

    if horizon:
        roles.extend((ROLE_SAME_STEP, ROLE_EXACT, ROLE_FINAL_STATE))
        if horizon > 1:
            roles.extend((ROLE_EARLY, ROLE_LATE))
        confusions.extend(("same_step_vs_next", "next_depth", "finite_suffix"))
    if "U" in operators:
        roles.extend(("immediate_witness", "left_break", "witness_absent"))
        confusions.extend(("strong_until", "until_witness"))

    return ObligationProfile(
        scope="global" if global_scope else "current",
        temporal_operator=temporal_operator,
        required_horizon=horizon or None,
        required_roles=_unique(roles),
        confusion_targets=_unique(confusions),
        trigger=semantic.trigger,
        response=semantic.response,
        formula=formula,
        source=source,
    )


def enrich_trace_semantics(
    traces: list[LabeledTrace],
    profile: ObligationProfile,
) -> list[LabeledTrace]:
    """Attach semantic labels, obligation instances, and boundary roles."""

    positive_lengths = [
        len(trace.trace)
        for trace in traces
        if trace.expected
    ]
    minimum_positive_length = min(positive_lengths) if positive_lengths else None
    enriched = [
        replace(
            trace,
            semantics=_classify_trace(
                trace,
                profile,
                minimum_positive_length=minimum_positive_length,
            ),
        )
        for trace in traces
    ]
    return _attach_minimal_perturbations(enriched)


def validate_trace_labels(
    traces: list[LabeledTrace],
    profile: ObligationProfile,
) -> tuple[list[LabeledTrace], tuple[str, ...]]:
    """Return traces whose labels match executable semantics and conflict messages."""

    if profile.formula is None:
        return list(traces), ()
    valid: list[LabeledTrace] = []
    conflicts: list[str] = []
    for trace in traces:
        actual = check_trace(trace.trace, profile.formula)
        if actual != trace.expected:
            conflicts.append(
                f"{trace.name}: expected={trace.expected} 与义务公式 "
                f"{profile.formula} 的有限轨迹求值 {actual} 冲突"
            )
            continue
        valid.append(trace)
    return valid, tuple(conflicts)


def covered_roles(traces: list[LabeledTrace]) -> tuple[str, ...]:
    return _unique(
        role
        for trace in traces
        for role in trace.semantics.roles
    )


def covered_confusions(traces: list[LabeledTrace]) -> tuple[str, ...]:
    return _unique(
        target
        for trace in traces
        for target in trace.semantics.confusion_targets
    )


def semantic_label_for_trace(
    trace: list[set[str]],
    profile: ObligationProfile,
) -> bool | None:
    if profile.formula is None:
        return None
    return check_trace(trace, profile.formula)


def build_obligation_trace_set(
    profile: ObligationProfile,
    semantic: SemanticStructure,
    *,
    target_trace_count: int = 6,
    target_trace_length: int | None = None,
) -> list[LabeledTrace]:
    """Build a compact role-covering trace set from executable obligations."""

    if profile.formula is None:
        return []
    aps = tuple(dict.fromkeys((*semantic.aps, *_formula_atoms(profile.formula))))
    if not aps or len(aps) > 4:
        return []
    max_enumerated_length = (
        5
        if len(aps) <= 2
        else 4
        if len(aps) == 3
        else 3
    )
    lengths = (
        (target_trace_length,)
        if target_trace_length is not None
        else tuple(
            range(
                1,
                min(
                    max_enumerated_length,
                    max(3, (profile.required_horizon or 0) + 2),
                )
                + 1,
            )
        )
    )
    raw_candidates: list[tuple[list[set[str]], bool, TraceSemantics]] = []
    for length in lengths:
        if length < 1:
            continue
        for candidate in _enumerate_traces(aps, length):
            expected = check_trace(candidate, profile.formula)
            provisional = LabeledTrace(
                name="candidate",
                kind="boundary",
                trace=candidate,
                expected=expected,
                reason="Obligation-driven candidate.",
            )
            raw_candidates.append(
                (
                    candidate,
                    expected,
                    _classify_trace(
                        provisional,
                        profile,
                        minimum_positive_length=length if expected else None,
                    ),
                )
            )

    selected: list[LabeledTrace] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    covered: set[str] = set()
    required = set(profile.required_roles) | set(profile.confusion_targets)

    def append_best(kind: str, expected: bool | None) -> bool:
        pool = [
            item
            for item in raw_candidates
            if (expected is None or item[1] is expected)
            and _trace_shape(item[0]) not in seen
        ]
        if not pool:
            return False

        def score(item) -> tuple:
            trace, _expected, semantics = item
            evidence = set(semantics.roles) | set(semantics.confusion_targets)
            new_evidence = len((evidence & required) - covered)
            return (
                -new_evidence,
                len(trace),
                sum(len(step) for step in trace),
                _trace_shape(trace),
            )

        candidate, candidate_expected, semantics = min(pool, key=score)
        index = 1 + sum(1 for trace in selected if trace.kind == kind)
        selected.append(
            LabeledTrace(
                name=f"obligation_{kind}_{index}",
                kind=kind,
                trace=candidate,
                expected=candidate_expected,
                reason=(
                    f"根据已审核语义义务 {profile.formula} 生成；"
                    f"覆盖角色 {', '.join(semantics.roles) or 'ordinary'}。"
                ),
            )
        )
        seen.add(_trace_shape(candidate))
        covered.update(semantics.roles)
        covered.update(semantics.confusion_targets)
        return True

    for _index in range(2):
        append_best("positive", True)
    for _index in range(2):
        append_best("negative", False)
    for _index in range(2):
        append_best("boundary", None)

    target_count = max(6, target_trace_count)
    while len(selected) < target_count:
        if not append_best("positive", True):
            if not append_best("negative", False):
                if not append_best("boundary", None):
                    break
    return enrich_trace_semantics(selected, profile)


def _classify_trace(
    trace: LabeledTrace,
    profile: ObligationProfile,
    *,
    minimum_positive_length: int | None,
) -> TraceSemantics:
    roles: list[str] = []
    confusion_targets: list[str] = []
    trigger_positions = _ap_positions(trace.trace, profile.trigger)
    response_positions = _ap_positions(trace.trace, profile.response)

    if trace.kind == "boundary":
        roles.append(ROLE_BOUNDARY)
    if trace.expected and minimum_positive_length == len(trace.trace):
        roles.append(ROLE_MINIMUM_WITNESS)
    if profile.trigger and not trigger_positions:
        roles.append(ROLE_VACUOUS)
    if len(trigger_positions) == 1:
        roles.append(ROLE_SINGLE_TRIGGER)
    if len(trigger_positions) > 1:
        roles.append(ROLE_MULTIPLE_OBLIGATION)
    if any(position > 0 for position in trigger_positions):
        roles.append(ROLE_LATE_TRIGGER)
        confusion_targets.append("local_vs_global_scope")
    if _has_same_step(trigger_positions, response_positions):
        roles.append(ROLE_SAME_STEP)
        if profile.required_horizon:
            confusion_targets.append("same_step_vs_next")
        if "F" in _formula_operators(profile.formula):
            confusion_targets.append("F_vs_XF")

    horizon = profile.required_horizon
    obligations: list[str] = []
    remaining_suffix: int | None = None
    if trigger_positions and profile.response:
        for position in trigger_positions:
            if horizon:
                target = position + horizon
                obligations.append(
                    f"{profile.trigger}@t{position} requires "
                    f"{profile.response}@t{target}"
                )
                remaining_suffix = len(trace.trace) - 1 - position
                if target >= len(trace.trace):
                    roles.append(ROLE_FINAL_STATE)
                    confusion_targets.append("finite_suffix")
                elif profile.response in trace.trace[target]:
                    roles.append(ROLE_EXACT)
                    confusion_targets.append("next_depth")
                if horizon > 1:
                    if position + horizon - 1 < len(trace.trace) and (
                        profile.response in trace.trace[position + horizon - 1]
                    ):
                        roles.append(ROLE_EARLY)
                    if position + horizon + 1 < len(trace.trace) and (
                        profile.response in trace.trace[position + horizon + 1]
                    ):
                        roles.append(ROLE_LATE)
            else:
                obligations.append(
                    f"{profile.trigger}@t{position} requires {profile.response}"
                )

    if trigger_positions and not trace.expected and profile.response:
        if all(
            not any(profile.response in step for step in trace.trace[position:])
            for position in trigger_positions
        ):
            roles.append(ROLE_MISSING_RESPONSE)
            if any(position > 0 for position in trigger_positions):
                roles.append(ROLE_LATE_MISSING_RESPONSE)

    until_operands = _first_until_operands(profile.formula)
    if until_operands is not None:
        left, right = until_operands
        right_positions = [
            position
            for position in range(len(trace.trace))
            if satisfies(trace.trace, right, position)
        ]
        if right_positions and right_positions[0] == 0:
            roles.append("immediate_witness")
        if not right_positions and not trace.expected:
            roles.append("witness_absent")
        if right_positions:
            first_witness = right_positions[0]
            if any(
                not satisfies(trace.trace, left, position)
                for position in range(first_witness)
            ):
                roles.append("left_break")
        if any(
            role in roles
            for role in ("immediate_witness", "left_break", "witness_absent")
        ):
            confusion_targets.extend(("strong_until", "until_witness"))

    if profile.formula is not None and profile.formula.op == "G":
        child = profile.formula.args[0]
        failing_positions = [
            position
            for position in range(len(trace.trace))
            if not satisfies(trace.trace, child, position)
        ]
        if not failing_positions and trace.expected:
            roles.append(ROLE_GLOBAL_WITNESS)
        if failing_positions == [0]:
            roles.append(ROLE_FIRST_VIOLATION)
        if (
            len(failing_positions) == 1
            and 0 < failing_positions[0] < len(trace.trace) - 1
        ):
            roles.append(ROLE_MIDDLE_VIOLATION)
        if len(trace.trace) > 1 and failing_positions == [len(trace.trace) - 1]:
            roles.append(ROLE_FINAL_VIOLATION)
        if failing_positions:
            confusion_targets.append("current_vs_global_scope")

    semantic_label = "UNSAT"
    if trace.expected:
        semantic_label = (
            "SAT_VACUOUS"
            if ROLE_VACUOUS in roles
            else "SAT_WITNESS"
        )
    return TraceSemantics(
        semantic_label=semantic_label,
        roles=_unique(roles),
        trigger_instances=tuple(
            f"{profile.trigger}@t{position}" for position in trigger_positions
        ),
        obligations=tuple(obligations),
        required_horizon=profile.required_horizon,
        remaining_suffix=remaining_suffix,
        confusion_targets=_unique(confusion_targets),
        perturbation=trace.semantics.perturbation,
    )


def _reference_formula(
    requirement: str,
    semantic: SemanticStructure,
    evidence: str,
) -> tuple[Formula | None, str]:
    # A native IR run has exactly one executable authority.  Compilation
    # failure is semantic debt and must never fall through to the legacy text
    # heuristics below, which exist only for pre-IR warmup artifacts.
    if semantic.faithful_ir is not None:
        return _certified_ir_reference_formula(semantic)

    if semantic.formula_text.strip():
        try:
            return parse_learned_formula(semantic.formula_text), "formula_text"
        except ValueError:
            return None, ""

    if semantic.pattern == "response":
        if not _is_ap(semantic.trigger) or not _is_ap(semantic.response):
            return None, ""
        components = dict(semantic.components)
        trigger = _first_component_formula(
            components,
            ("trigger_formula", "trigger_definition"),
        ) or Atom(semantic.trigger)
        explicit_response = _first_component_formula(
            components,
            ("response_formula", "response_definition"),
        )
        if explicit_response is not None:
            formula = Implies(trigger, explicit_response)
            if _scope_is_global(requirement, semantic):
                formula = G(formula)
            return formula, "reviewed_response_components"

        response_atom = Atom(semantic.response)
        if _looks_like_future_prohibition(requirement):
            formula = G(Implies(trigger, X(G(Not(response_atom)))))
            return formula, "reviewed_response_prohibition"

        relative_depth = _relative_response_next_depth(
            requirement,
            response=semantic.response,
        )
        window_depth = _next_window_depth(requirement)
        exact_depth = _exact_next_depth(requirement)
        if relative_depth:
            response: Formula = _apply_next(response_atom, relative_depth)
        elif window_depth:
            response = _next_window_formula(response_atom, window_depth)
        elif exact_depth:
            response = _apply_next(response_atom, exact_depth)
        else:
            response = response_atom
            if _looks_eventual(evidence):
                response = F(response)
            if _requires_strict_future(requirement, evidence):
                response = X(response)
            elif _text_next_depth(evidence):
                response = _apply_next(response, _text_next_depth(evidence))

        formula = Implies(trigger, response)
        if _scope_is_global(requirement, semantic):
            formula = G(formula)
        return formula, "reviewed_response_semantics"

    components = dict(semantic.components)
    if semantic.pattern == "conjunction" and _scope_is_global(requirement, semantic):
        left = _parse_component(components.get("left_component", ""))
        right = _parse_component(components.get("right_component", ""))
        if left is not None and right is not None:
            formula = And(left, right)
            formula = G(formula)
            return formula, "reviewed_conjunction_components"

    if semantic.pattern in {"mutual_exclusion", "mutual_exclusion_with_coverage"}:
        aps = tuple(dict.fromkeys(semantic.aps))
        if len(aps) >= 2:
            left, right = Atom(aps[0]), Atom(aps[1])
            formula: Formula = G(Not(And(left, right)))
            if semantic.pattern == "mutual_exclusion_with_coverage":
                formula = G(And(Not(And(left, right)), _or(left, right)))
            return formula, "reviewed_pattern_semantics"

    if semantic.pattern == "until":
        left = _parse_component(components.get("left_condition", ""))
        right = _parse_component(components.get("until_witness", ""))
        if left is not None and right is not None:
            return U(left, right), "reviewed_until_components"
    return None, ""


def _certified_ir_reference_formula(
    semantic: SemanticStructure,
) -> tuple[Formula | None, str]:
    ir = semantic.faithful_ir
    if ir is None:
        return None, ""
    try:
        from ir_compiler import compile_ir_artifact

        artifact = compile_ir_artifact(ir)
        artifact.validate_for(ir)
        return artifact.formula, "certified_faithful_ir"
    except ValueError:
        return None, "ir_compilation_blocked"


def _parse_component(text: str) -> Formula | None:
    value = text.strip()
    if not value:
        return None
    definition = re.match(
        r"^[A-Za-z_][A-Za-z0-9_]*\s*(?::=|=)\s*(.+?)\s*$",
        value,
    )
    if definition:
        value = definition.group(1).strip()
    normalized = (
        value.replace("&&", "&")
        .replace("||", "|")
        .replace("finally", "F")
        .replace("globally", "G")
    )
    try:
        return parse_learned_formula(normalized)
    except ValueError:
        return None


def _semantic_evidence(requirement: str, semantic: SemanticStructure) -> str:
    return " ".join(
        (
            requirement,
            semantic.pattern,
            semantic.scope_keyword,
            semantic.temporal_keyword,
            semantic.explanation,
            " ".join(f"{key} {value}" for key, value in semantic.components),
        )
    ).lower()


def _looks_global(text: str) -> bool:
    return any(
        token in text
        for token in (
            "global",
            "globally",
            "always",
            "whenever",
            "every position",
            "every time",
            "始终",
            "总是",
            "每当",
        )
    )


def _scope_is_global(requirement: str, semantic: SemanticStructure) -> bool:
    scope = semantic.scope_keyword.strip().lower()
    if semantic.pattern == "until" and " or always " in f" {requirement.lower()} ":
        return False
    if any(
        token in scope
        for token in (
            "current position",
            "current step",
            "no outer global",
            "same-step",
            "same step",
        )
    ):
        return False
    if scope in {"if", "when", "once", "current", "local"}:
        return False
    if any(
        token in scope
        for token in (
            "global",
            "whenever",
            "every",
            "always",
            "all finite-trace positions",
            "每当",
            "始终",
            "总是",
        )
    ):
        return True

    normalized = requirement.strip().lower()
    return bool(
        re.match(
            r"^(whenever|every(?:time)?\b|it is always the case|always\b|每当|始终|总是)",
            normalized,
        )
    )


def _looks_eventual(text: str) -> bool:
    return any(
        token in text
        for token in ("eventually", "finally", "sometime", "最终", "终将")
    )


def _same_step_is_rejected(text: str) -> bool:
    return (
        ("同时" in text or "同一" in text or "same time" in text)
        and (
            "不满足" in text
            or "false" in text
            or "must occur later" in text
        )
    )


_NUMBER_WORDS = {
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


def _number_value(value: str) -> int:
    normalized = value.strip().lower()
    if normalized.isdigit():
        return int(normalized)
    return _NUMBER_WORDS.get(normalized, 0)


def _exact_next_depth(text: str) -> int:
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?:steps?|time units?|timestamps?)\s+(?:later|after)\b",
        text.lower(),
    )
    return _number_value(match.group(1)) if match else 0


def _next_window_depth(text: str) -> int:
    match = re.search(
        r"\b(?:in|within)\s+the\s+next\s+"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+(?:steps?|time units?|timestamps?)\b",
        text.lower(),
    )
    return _number_value(match.group(1)) if match else 0


def _relative_response_next_depth(text: str, *, response: str) -> int:
    normalized = text.lower()
    intermediate = re.search(
        r"\b([a-z_][a-z0-9_]*)\s+holds?\s+in\s+the\s+next\s+step\b",
        normalized,
    )
    if not intermediate:
        return 0
    relation = re.search(
        rf"\b{re.escape(response.lower())}\s+holds?\s+"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        rf"\s+steps?\s+after\s+{re.escape(intermediate.group(1))}\b",
        normalized,
    )
    if not relation:
        return 0
    return 1 + _number_value(relation.group(1))


def _requires_strict_future(requirement: str, evidence: str) -> bool:
    normalized = requirement.lower()
    if any(
        token in normalized
        for token in (
            "from the next step",
            "starting from the next step",
            "must occur later",
            "之后",
            "下一步开始",
        )
    ):
        return True
    if "followed by" in normalized and "eventually followed by" not in normalized:
        return True
    return _same_step_is_rejected(evidence)


def _looks_like_future_prohibition(requirement: str) -> bool:
    normalized = requirement.lower()
    return any(
        token in normalized
        for token in (
            "won't happen again",
            "will not happen again",
            "must not happen again",
            "never happen again",
            "never occurs again",
            "之后不再",
            "以后不再",
        )
    )


def _looks_like_until_or_always(requirement: str, left_ap: str) -> bool:
    normalized = requirement.lower()
    return (
        " until " in f" {normalized} "
        and " or always " in f" {normalized} "
        and re.search(rf"\balways\s+{re.escape(left_ap.lower())}\b", normalized)
        is not None
    )


def _first_component_formula(
    components: dict[str, str],
    keys: tuple[str, ...],
) -> Formula | None:
    for key in keys:
        formula = _parse_component(components.get(key, ""))
        if formula is not None:
            return formula
    return None


def _apply_next(formula: Formula, depth: int) -> Formula:
    result = formula
    for _index in range(max(0, depth)):
        result = X(result)
    return result


def _next_window_formula(formula: Formula, depth: int) -> Formula:
    alternatives = [_apply_next(formula, index) for index in range(1, depth + 1)]
    result = alternatives[0]
    for alternative in alternatives[1:]:
        result = Or(result, alternative)
    return result


def _text_next_depth(text: str) -> int:
    exact_depth = _exact_next_depth(text)
    if exact_depth:
        return exact_depth
    window_depth = _next_window_depth(text)
    if window_depth:
        return window_depth
    chains = [
        len(match.group(0).replace(" ", ""))
        for match in re.finditer(
            r"(?<![A-Za-z0-9_])(?:X\s*)+",
            text,
        )
    ]
    if chains:
        return max(chains)
    if "next" in text or "下一步" in text:
        return 1
    return 0


def _formula_operators(formula: Formula | None) -> set[str]:
    if formula is None:
        return set()
    result = {formula.op}
    for child in formula.args:
        result.update(_formula_operators(child))
    return result


def _max_next_depth(formula: Formula | None) -> int:
    if formula is None:
        return 0
    if formula.op == "X":
        return 1 + _max_next_depth(formula.args[0])
    return max((_max_next_depth(child) for child in formula.args), default=0)


def _temporal_operator_name(
    operators: set[str],
    horizon: int,
    evidence: str,
) -> str:
    if horizon:
        return f"X^{horizon}"
    for operator in ("U", "F", "G"):
        if operator in operators:
            return operator
    if _looks_eventual(evidence):
        return "F"
    return ""


def _ap_positions(trace: list[set[str]], ap: str) -> list[int]:
    if not _is_ap(ap):
        return []
    return [
        position
        for position, step in enumerate(trace)
        if ap in step
    ]


def _has_same_step(left: list[int], right: list[int]) -> bool:
    return bool(set(left) & set(right))


def _is_ap(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value or ""))


def _formula_atoms(formula: Formula) -> tuple[str, ...]:
    atoms: list[str] = []

    def visit(node: Formula) -> None:
        if node.op == "ap" and node.name not in atoms:
            atoms.append(node.name)
        for child in node.args:
            visit(child)

    visit(formula)
    return tuple(atoms)


def _first_until_operands(
    formula: Formula | None,
) -> tuple[Formula, Formula] | None:
    if formula is None:
        return None
    if formula.op == "U":
        return formula.args[0], formula.args[1]
    for child in formula.args:
        found = _first_until_operands(child)
        if found is not None:
            return found
    return None


def _enumerate_traces(
    aps: tuple[str, ...],
    length: int,
):
    valuations = [
        {ap for ap, enabled in zip(aps, flags) if enabled}
        for flags in product((False, True), repeat=len(aps))
    ]
    valuations.sort(key=lambda step: (len(step), tuple(sorted(step))))
    for steps in product(valuations, repeat=length):
        yield [set(step) for step in steps]


def _trace_shape(trace: list[set[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted(step)) for step in trace)


def _attach_minimal_perturbations(
    traces: list[LabeledTrace],
) -> list[LabeledTrace]:
    enriched: list[LabeledTrace] = []
    for trace in traces:
        perturbation = trace.semantics.perturbation
        if not perturbation:
            for other in traces:
                if other.expected == trace.expected:
                    continue
                perturbation = _minimal_perturbation(trace.trace, other.trace)
                if perturbation:
                    perturbation = f"{perturbation}:{other.name}"
                    break
        enriched.append(
            replace(
                trace,
                semantics=replace(
                    trace.semantics,
                    perturbation=perturbation,
                ),
            )
        )
    return enriched


def _minimal_perturbation(
    source: list[set[str]],
    target: list[set[str]],
) -> str:
    if len(source) != len(target):
        return ""
    removed: list[tuple[int, str]] = []
    added: list[tuple[int, str]] = []
    for position, (left, right) in enumerate(zip(source, target)):
        removed.extend((position, ap) for ap in left - right)
        added.extend((position, ap) for ap in right - left)
    if len(removed) == 1 and not added:
        return "REMOVE"
    if len(added) == 1 and not removed:
        return "ADD"
    if (
        len(removed) == 1
        and len(added) == 1
        and removed[0][1] == added[0][1]
    ):
        return "SHIFT"
    return ""


def _or(left: Formula, right: Formula) -> Formula:
    from ltlf_parser import Or

    return Or(left, right)


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))
