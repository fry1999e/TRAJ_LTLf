"""One versioned source-language relation registry.

Tokenizer, coverage, parameter grounding and tests consume this module instead
of maintaining subtly different synonym lists.  The registry describes source
language only; it never contains dataset IDs or reference-formula templates.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping


LEXICON_VERSION = "source-relation-lexicon-v1"


@dataclass(frozen=True)
class RelationLexeme:
    relation: str
    pattern: str
    tokens: frozenset[str]


def _lexeme(relation: str, pattern: str, *tokens: str) -> RelationLexeme:
    return RelationLexeme(relation, pattern, frozenset(token.casefold() for token in tokens))


SOURCE_RELATION_LEXICON: Mapping[str, RelationLexeme] = {
    "implication": _lexeme(
        "implication",
        r"\b(?:if|only\s+if|when|only\s+when|whenever|upon|on\s+condition\s+that)\b",
        "if", "only", "when", "whenever", "upon", "on", "condition", "that",
        "then", "unless", "every", "time", "everytime", "in", "case",
    ),
    "conjunction": _lexeme("conjunction", r"\b(?:and|both)\b", "and", "both"),
    "disjunction": _lexeme("disjunction", r"\b(?:or|either)\b", "or", "either"),
    "global_constraint": _lexeme(
        "global_constraint", r"\b(?:globally|always|whenever|every\s+time|at\s+all\s+times)\b",
        "globally", "always", "whenever", "every", "time", "at", "all", "times",
    ),
    "eventually": _lexeme(
        "eventually", r"\b(?:eventually|ultimately|finally|sometime|later)\b",
        "eventually", "ultimately", "finally", "sometime", "future", "later",
    ),
    "next_step": _lexeme("next_step", r"\bnext\b", "next"),
    "until": _lexeme("until", r"\b(?:until|till)\b", "until", "till"),
    "fixed_duration": _lexeme(
        "fixed_duration", r"\bfor\s+(?:\d+|[a-z]+)\s+steps?\b", "for", "step", "steps",
    ),
    "exact_run_length": _lexeme(
        "exact_run_length", r"\b(?:for\s+)?only\s+(?:\d+|[a-z]+)\s+steps?\b",
        "for", "only", "step", "steps",
    ),
    "bounded_window": _lexeme(
        "bounded_window", r"\b(?:within|between|from|no\s+later\s+than)\b",
        "within", "between", "from", "to", "no", "later", "than", "steps",
    ),
    "before": _lexeme(
        "before", r"\b(?:at\s+or\s+before|no\s+later\s+than|before|prior\s+to|preced(?:e|es|ed|ing|ence))\b",
        "at", "or", "before", "no", "later", "than", "prior", "to",
        "precede", "precedes", "preceded", "preceding", "precedence",
    ),
    "precedence": _lexeme(
        "precedence", r"\b(?:at\s+or\s+before|no\s+later\s+than|before|prior\s+to|preced(?:e|es|ed|ing|ence))\b",
        "at", "or", "before", "no", "later", "than", "prior", "to",
        "precede", "precedes", "preceded", "preceding", "precedence",
    ),
    "after": _lexeme("after", r"\b(?:after|following)\b", "after", "following"),
    "cardinality": _lexeme(
        "cardinality", r"\b(?:at\s+(?:least|most)|exactly)\b",
        "at", "least", "most", "exactly", "times", "occurrences",
    ),
    "recurrence": _lexeme(
        "recurrence", r"\b(?:recur(?:s|ring)?|repeatedly|again)\b",
        "recur", "recurs", "recurring", "repeatedly", "again",
    ),
    "periodicity": _lexeme(
        "periodicity", r"\b(?:every\s+(?:\d+|[a-z]+)\s+steps?|periodically)\b",
        "every", "step", "steps", "periodically",
    ),
    "negation": _lexeme("negation", r"\b(?:not|never|no)\b", "not", "never", "no"),
}


def relation_pattern(*relations: str) -> str:
    """Return a non-capturing union for the requested registered relations."""

    missing = set(relations) - set(SOURCE_RELATION_LEXICON)
    if missing:
        raise KeyError("unknown source relation: " + ", ".join(sorted(missing)))
    return "(?:" + "|".join(SOURCE_RELATION_LEXICON[name].pattern for name in relations) + ")"


def relation_tokens(relation: str) -> frozenset[str]:
    return SOURCE_RELATION_LEXICON[relation].tokens


def scan_relations(text: str, relations: Iterable[str] | None = None):
    names = tuple(relations or SOURCE_RELATION_LEXICON)
    return tuple(
        (name, match.span(), match.group(0))
        for name in names
        for match in re.finditer(SOURCE_RELATION_LEXICON[name].pattern, text, re.IGNORECASE)
    )


def conditional_direction(marker: str) -> str:
    """Return the ordinary-language direction contract for a condition marker."""

    normalized = re.sub(r"\s+", " ", marker.strip().casefold())
    if normalized in {"if", "when", "whenever", "upon", "on condition that"}:
        return "condition_implies_result"
    if normalized in {"only if", "only when"}:
        return "result_implies_condition"
    if normalized == "unless":
        return "requires_source_contract"
    raise KeyError("unsupported conditional marker: " + marker)


__all__ = [
    "LEXICON_VERSION",
    "RelationLexeme",
    "SOURCE_RELATION_LEXICON",
    "conditional_direction",
    "relation_pattern",
    "relation_tokens",
    "scan_relations",
]
