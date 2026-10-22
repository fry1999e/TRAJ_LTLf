"""Experiment-mode policy shared by CLI and pipeline entrypoints."""

from __future__ import annotations

from schemas import EXPERIMENT_MODES


SAMPLE_REFERENCE_FORMULA = "sample"


def resolve_experiment_mode(review_mode: str, experiment_mode: str | None) -> str:
    """Return the effective experiment mode without implicitly enabling a judge."""

    mode = experiment_mode or "human"
    if mode != "human":
        raise ValueError("This release supports human mode only; model reviewers are disabled.")
    return mode


def implicit_auto_human_warning(
    review_mode: str,
    experiment_mode: str | None,
    *,
    warmup_only: bool = False,
) -> str:
    """Explain the ambiguous compatibility default used by auto review runs."""

    if review_mode != "auto" or experiment_mode is not None or warmup_only:
        return ""
    return (
        "警告: --review auto 未显式指定 --experiment-mode；当前兼容默认仍会记录为 "
        "human。该组合只适合非实验性本地冒烟。人工实验请使用 --review manual "
        "--experiment-mode human。"
    )


def requires_judge_client(review_mode: str, experiment_mode: str) -> bool:
    """Only the explicit auto + independent_judge combination uses Judge LLM."""

    if experiment_mode != "human":
        raise ValueError("Model reviewers are disabled; use human mode.")
    return False


def is_oracle_mode(experiment_mode: str) -> bool:
    return experiment_mode == "formula_oracle"


def is_sample_reference_formula(reference_formula: str | object | None) -> bool:
    """Return whether the CLI reference formula requests per-sample formulas."""

    return (
        isinstance(reference_formula, str)
        and reference_formula.strip().lower() == SAMPLE_REFERENCE_FORMULA
    )
