"""Shared orchestration boundary for validated plain-reading generation.

The generation agent deliberately raises :class:`PlainReadingSemanticFallback`
after its bounded artifact budget when the terminal artifact is structurally
usable but has no safe source-grounded ambiguity axis.  Production callers must
not interpret that exception independently: this module is the single policy
boundary that either materializes the auditable fallback review or preserves
the exception when the rollout switch disables fallback handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from agents.plain_reading_agent import (
    PlainReadingAgent,
    PlainReadingSemanticFallback,
)
from plain_reading import build_plain_reading_semantic_fallback
from schemas import ArtifactGenerator, PlainReadingReview, SurfaceSegmentation


PlainReadingGenerationKind = Literal["normal", "semantic_fallback"]


@dataclass(frozen=True)
class PlainReadingGenerationOutcome:
    """A validated review plus the policy path that produced it."""

    kind: PlainReadingGenerationKind
    review: PlainReadingReview
    diagnostics: dict[str, Any]
    reason_codes: tuple[str, ...] = ()
    attempts: int = 0

    @property
    def semantic_fallback(self) -> bool:
        return self.kind == "semantic_fallback"


def generate_plain_reading_outcome(
    agent: PlainReadingAgent,
    requirement: str,
    segmentation: SurfaceSegmentation,
    *,
    semantic_fallback_enabled: bool = True,
    fallback_previous_review: PlainReadingReview | None = None,
    **generation_kwargs: Any,
) -> PlainReadingGenerationOutcome:
    """Run one bounded generation through the shared fallback policy.

    Structural, transport, patch-protocol and other hard failures continue to
    raise their original exceptions.  Only ``PlainReadingSemanticFallback`` is
    normalized, and only while the explicit rollout policy is enabled.
    """

    explicit_requirement_id = str(generation_kwargs.get("requirement_id") or "")
    if (
        explicit_requirement_id
        and explicit_requirement_id != segmentation.requirement_id
    ):
        raise ValueError(
            "plain-reading generation requirement_id does not match segmentation parent"
        )
    try:
        review = agent.generate(
            requirement,
            segmentation,
            **generation_kwargs,
        )
    except PlainReadingSemanticFallback as exc:
        if not semantic_fallback_enabled:
            raise
        if fallback_previous_review is None:
            requested = generation_kwargs.get("ambiguity_generation_request")
            if isinstance(requested, PlainReadingReview):
                fallback_previous_review = requested
        prompt_version = str(
            generation_kwargs.get("prompt_version") or "plain-reading-v5"
        )
        experiment_mode = str(generation_kwargs.get("mode") or "manual")
        data_split = str(generation_kwargs.get("data_split") or "test")
        fallback = build_plain_reading_semantic_fallback(
            requirement_id=(
                explicit_requirement_id
                or segmentation.requirement_id
                or review_requirement_id(fallback_previous_review)
            ),
            segmentation=segmentation,
            generator=ArtifactGenerator(
                mode="api" if bool(getattr(agent, "llm_enabled", True)) else "rules",
                prompt_version=prompt_version,
            ),
            experiment_mode=experiment_mode,
            data_split=data_split,
            reason_codes=exc.reason_codes,
            attempts=exc.attempts,
            diagnostics=exc.diagnostics,
            previous_review=fallback_previous_review,
        )
        return PlainReadingGenerationOutcome(
            kind="semantic_fallback",
            review=fallback,
            diagnostics=dict(exc.diagnostics),
            reason_codes=tuple(exc.reason_codes),
            attempts=exc.attempts,
        )
    return PlainReadingGenerationOutcome(
        kind="normal",
        review=review,
        diagnostics=dict(getattr(agent, "last_generation_diagnostic", {})),
    )


def review_requirement_id(review: PlainReadingReview | None) -> str:
    """Return a bounded fallback id for defensive legacy callers."""

    if review is None:
        raise ValueError("plain-reading fallback requires a stable requirement_id")
    return review.requirement_id


__all__ = [
    "PlainReadingGenerationKind",
    "PlainReadingGenerationOutcome",
    "generate_plain_reading_outcome",
]
