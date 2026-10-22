"""Temporal-domain detection and finite-trace rewriting."""

from __future__ import annotations

from dataclasses import replace
import re

from schemas import SemanticStructure


INFINITE_DOMAIN = "ltl_infinite"
FINITE_DOMAIN = "ltlf"
AMBIGUOUS_DOMAIN = "ambiguous"


_INFINITE_MARKERS: tuple[tuple[str, str], ...] = (
    (r"\binfinitely\s+often\b", "infinitely often"),
    (r"\binfinite\s+(trace|run|execution|trajectory)\b", "infinite trace/run"),
    (r"\bforever\b", "forever"),
    (r"\bfor\s+all\s+future\s+time\b", "for all future time"),
    (r"\bwithout\s+end\b", "without end"),
    (r"\bunbounded\b", "unbounded"),
    (r"\bnever\s+terminates?\b", "never terminates"),
    (r"\bnon[-\s]?terminating\b", "non-terminating"),
    (r"\balways\s+eventually\b", "always eventually / GF"),
    (r"\beventually\s+always\b", "eventually always / FG"),
    (r"\bG\s*F\b", "GF"),
    (r"\bF\s*G\b", "FG"),
    (r"无限|无穷|永远|不终止", "中文无限轨迹表达"),
)


_FINITE_REWRITES: tuple[tuple[str, str], ...] = (
    (r"\binfinitely\s+often\b", "at least once in the finite trace"),
    (r"\bforever\b", "until the end of the finite trace"),
    (r"\bfor\s+all\s+future\s+time\b", "for every remaining position in the finite trace"),
    (r"\bwithout\s+end\b", "until the finite trace ends"),
    (r"\bunbounded\b", "within the finite trace"),
    (r"\bnever\s+terminates?\b", "is evaluated over the observed finite trace"),
    (r"\bnon[-\s]?terminating\b", "finite"),
    (r"\balways\s+eventually\b", "at every finite-trace position, eventually"),
    (r"\beventually\s+always\b", "eventually and then always until the finite trace ends"),
    (r"无限|无穷|永远|不终止", "直到有限轨迹结束"),
)


def apply_temporal_domain_assessment(
    semantic: SemanticStructure,
    requirement: str,
) -> SemanticStructure:
    """Attach an LTL/LTLf domain assessment to a semantic structure."""

    evidence = _infinite_evidence(requirement)
    if not evidence:
        return replace(
            semantic,
            trace_domain=FINITE_DOMAIN,
            finite_rewrite="",
            domain_warning="",
            domain_evidence=(),
            reference_only=False,
        )

    finite_rewrite = _finite_rewrite(requirement)
    warning = (
        "原始自然语言疑似表达无限轨迹 LTL 性质；本流程会继续执行，"
        "但会按有限轨迹 LTLf 近似改写生成结果，最终公式仅供参考。"
    )
    explanation = semantic.explanation
    if warning not in explanation:
        explanation = (explanation + " " + warning).strip()
    return replace(
        semantic,
        trace_domain=INFINITE_DOMAIN,
        finite_rewrite=finite_rewrite,
        domain_warning=warning,
        domain_evidence=evidence,
        reference_only=True,
        explanation=explanation,
    )


def finite_requirement_for_pipeline(
    requirement: str,
    semantic: SemanticStructure,
) -> str:
    """Return the finite-trace requirement used by generation stages."""

    if semantic.reference_only and semantic.finite_rewrite:
        return semantic.finite_rewrite
    return requirement


def _infinite_evidence(requirement: str) -> tuple[str, ...]:
    evidence: list[str] = []
    for pattern, label in _INFINITE_MARKERS:
        if re.search(pattern, requirement, flags=re.IGNORECASE):
            evidence.append(label)
    return tuple(dict.fromkeys(evidence))


def _finite_rewrite(requirement: str) -> str:
    rewritten = requirement
    for pattern, replacement in _FINITE_REWRITES:
        rewritten = re.sub(
            pattern,
            replacement,
            rewritten,
            flags=re.IGNORECASE,
        )
    if rewritten == requirement:
        rewritten = (
            requirement.rstrip(".")
            + " within the observed finite trace."
        )
    return rewritten
