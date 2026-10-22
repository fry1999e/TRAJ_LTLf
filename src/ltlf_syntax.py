"""Shared guardrails for the LTLf formula fragment supported by this project."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ltlf_parser import Formula


SUPPORTED_TEMPORAL_OPERATORS = ("X", "F", "G", "U")
SUPPORTED_TEMPORAL_ALIASES = (
    "next",
    "eventually",
    "finally",
    "always",
    "globally",
    "until",
)

SUPPORTED_LTLF_FORMULA_GUIDANCE = (
    "formula_text 只能使用当前支持的未来时 LTLf 片段：AP、true/false、"
    "!、&、|、->、括号，以及 X/F/G/U "
    "(别名 next/eventually/finally/always/globally/until)。"
    "不要使用 O/once、Y/previous、H/historically、S/since、"
    "W/weak-until、R/release、before 或 after 作为公式算子。"
)

CANONICAL_BOOLEAN_OPERATOR_GUIDANCE = (
    "表达式中的布尔连接符统一使用 '&' 和 '|'；不要输出 '&&' 或 '||'，"
    "也不要在同一表达式中混用单字符和双字符写法。"
)

UNARY_OPERATOR_NESTING_GUIDANCE = (
    "否定算子 ! 不允许直接重复或仅隔括号嵌套重复，例如 !!p 和 !(!p)；"
    "X、F、G 时序算子允许重复、嵌套和组合，例如 X X p、F G p、G X F p。"
)


_UNSUPPORTED_TEMPORAL_TOKENS = {
    "AFTER": "after",
    "BEFORE": "before",
    "H": "historically/past always",
    "HISTORICALLY": "historically/past always",
    "O": "once/past eventually",
    "ONCE": "once/past eventually",
    "PREV": "previous",
    "PREVIOUS": "previous",
    "PRIOR": "prior",
    "R": "release",
    "RELEASE": "release",
    "S": "since",
    "SINCE": "since",
    "W": "weak until",
    "WEAKUNTIL": "weak until",
    "Y": "yesterday/previous",
    "YESTERDAY": "yesterday/previous",
}

_UNSUPPORTED_TEMPORAL_PHRASES = (
    (r"\bweak\s+until\b", "weak until"),
    (r"\bpast\s+once\b", "past once"),
    (r"\bpast\s+eventually\b", "past eventually"),
    (r"\bpast\s+always\b", "past always"),
)

_TOKEN_PATTERN = re.compile(
    r"\s*(<->|<=>|->|&&|\|\||[()!&|]|¬|∧|∨|→|↔|"
    r"[A-Za-z_][A-Za-z0-9_]*|.)"
)


def unsupported_temporal_operator_errors(
    text: str,
    *,
    context: str = "formula_text",
) -> list[str]:
    """Return errors for temporal operators outside the supported LTLf fragment."""

    errors: list[str] = []
    seen: set[str] = set()
    for token in _formula_tokens(text):
        normalized = token.upper().replace("-", "").replace("_", "")
        meaning = _UNSUPPORTED_TEMPORAL_TOKENS.get(normalized)
        if meaning is None:
            continue
        label = f"{context} 使用了不支持的时序算子 {token} ({meaning})。"
        if label not in seen:
            seen.add(label)
            errors.append(label)

    for pattern, meaning in _UNSUPPORTED_TEMPORAL_PHRASES:
        if not re.search(pattern, text, flags=re.IGNORECASE):
            continue
        label = f"{context} 使用了不支持的时序算子短语 {meaning}。"
        if label not in seen:
            seen.add(label)
            errors.append(label)

    return errors


def validate_supported_ltlf_formula_text(
    text: str,
    *,
    context: str = "formula_text",
) -> None:
    """Raise when a formula uses unsupported temporal syntax."""

    errors = unsupported_temporal_operator_errors(text, context=context)
    if errors:
        raise ValueError(" ".join([*errors, SUPPORTED_LTLF_FORMULA_GUIDANCE]))


def normalize_boolean_operator_text(text: str) -> str:
    """Return text with accepted Boolean aliases rewritten to canonical symbols."""

    return (
        text.replace("&&", "&")
        .replace("||", "|")
        .replace("\\/", "|")
        .replace("/\\", "&")
        .replace("∧", "&")
        .replace("∨", "|")
    )


def boolean_operator_style_errors(
    text: str,
    *,
    context: str = "expression",
) -> list[str]:
    """Return style errors for non-canonical or mixed Boolean operator spelling."""

    errors: list[str] = []
    has_double_and = "&&" in text
    has_double_or = "||" in text
    has_single_and = re.search(r"(?<!&)&(?!&)", text) is not None
    has_single_or = re.search(r"(?<!\|)\|(?!\|)", text) is not None

    if has_double_and:
        errors.append(f"{context} 使用了 '&&'，应统一写作 '&'。")
    if has_double_or:
        errors.append(f"{context} 使用了 '||'，应统一写作 '|'。")
    if has_single_and and has_double_and:
        errors.append(f"{context} 同时出现 '&' 和 '&&'。")
    if has_single_or and has_double_or:
        errors.append(f"{context} 同时出现 '|' 和 '||'。")
    return errors


def validate_canonical_boolean_operators(
    text: str,
    *,
    context: str = "expression",
) -> None:
    """Raise when generated text does not use canonical Boolean symbols."""

    errors = boolean_operator_style_errors(text, context=context)
    if errors:
        raise ValueError(" ".join([*errors, CANONICAL_BOOLEAN_OPERATOR_GUIDANCE]))


def has_repeated_negation(formula: Formula) -> bool:
    """Return whether a negation directly wraps another negation."""

    if formula.op == "not" and formula.args and formula.args[0].op == "not":
        return True
    return any(has_repeated_negation(child) for child in formula.args)


def validate_no_repeated_negation(
    formula: Formula,
    *,
    context: str = "formula",
) -> None:
    """Raise when a formula contains an adjacent or parenthesized ! chain."""

    if has_repeated_negation(formula):
        raise ValueError(
            f"{context} 包含重复否定；"
            + UNARY_OPERATOR_NESTING_GUIDANCE
        )


def forbidden_temporal_operator_errors(
    text: str,
    *,
    forbidden: tuple[str, ...] = ("X", "F", "G"),
    context: str = "template_text",
) -> list[str]:
    """Return errors for concrete temporal operators forbidden in this stage."""

    forbidden_set = {item.upper() for item in forbidden}
    alias_to_operator = {
        "next": "X",
        "eventually": "F",
        "finally": "F",
        "always": "G",
        "globally": "G",
    }
    errors: list[str] = []
    seen: set[str] = set()
    for token in _formula_tokens(text):
        operator = alias_to_operator.get(token.lower(), token.upper())
        if operator not in forbidden_set:
            continue
        label = f"{context} 包含本阶段禁用的具体时序算子 {token}。"
        if label not in seen:
            seen.add(label)
            errors.append(label)
    return errors


def validate_no_forbidden_temporal_operators(
    text: str,
    *,
    forbidden: tuple[str, ...] = ("X", "F", "G"),
    context: str = "template_text",
) -> None:
    """Raise when text contains concrete temporal operators forbidden in a stage."""

    errors = forbidden_temporal_operator_errors(
        text,
        forbidden=forbidden,
        context=context,
    )
    if errors:
        raise ValueError(" ".join(errors))


def _formula_tokens(text: str) -> list[str]:
    tokens = re.findall(_TOKEN_PATTERN, text)
    return [token for token in tokens if token.strip()]
