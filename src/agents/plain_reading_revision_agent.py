"""Non-authoritative binary-axis rebuilder for human-classified revisions."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Mapping

from agents.base_agent import BaseAgent, JSONLLMClient
from schemas import (
    PlainReadingAmbiguity,
    PlainReadingOption,
    PlainReadingRevisionRequest,
    SurfaceSegmentation,
)
from plain_reading import validate_rebuilt_plain_reading_axes


class PlainReadingRevisionAgent(BaseAgent):
    """Rebuild binary axes only after an ordinary reviewer classifies a request."""

    def __init__(self, retriever, llm_client: JSONLLMClient | None = None):
        super().__init__(
            retriever=retriever,
            name="PlainReadingRevisionAgent",
            llm_client=llm_client,
        )

    def rebuild_axes(
        self,
        *,
        source_text: str,
        segmentation: SurfaceSegmentation,
        ambiguity: PlainReadingAmbiguity,
        request: PlainReadingRevisionRequest,
        human_classification: str,
    ) -> tuple[PlainReadingAmbiguity, ...]:
        if human_classification not in {
            "same_axis_revision",
            "new_independent_axis",
        }:
            raise ValueError(
                "rebuilt axes require a human same-axis or independent-axis classification"
            )
        if not self.llm_enabled:
            return ()
        raw = self.complete_json(
            system_prompt=(
                "The ordinary-language reviewer has already classified this "
                "revision. Do not classify, approve, override, or reinterpret "
                "that human decision. Rebuild only the requested natural-language "
                "ambiguity axes. Do not infer LTLf, output formulas, or add source "
                "facts. Every axis must contain exactly two complete natural-language "
                "options differing on one dimension."
            ),
            user_payload={
                "source_text": source_text,
                "segments": [
                    {"segment_id": item.segment_id, "text": item.text}
                    for item in segmentation.segments
                ],
                "current_ambiguity": {
                    "ambiguity_id": ambiguity.ambiguity_id,
                    "question": ambiguity.question,
                    "options": [
                        {
                            "option_id": option.option_id,
                            "plain_interpretation": option.plain_interpretation,
                            "difference_summary": option.difference_summary,
                        }
                        for option in ambiguity.options
                    ],
                },
                "human_proposal": request.proposed_interpretation,
                "human_reason": request.reason,
                "human_classification": human_classification,
                "classification_instruction": (
                    "Replace the current ambiguity axis with the rebuilt axes."
                    if human_classification == "same_axis_revision"
                    else "Keep the current ambiguity axis and generate only new independent axes."
                ),
                "output_schema": {
                    "rebuilt_axes": [
                        {
                            "ambiguity_id": "stable id",
                            "source_segment_ids": ["seg-1"],
                            "question": "one ordinary-language difference",
                            "options": [
                                {
                                    "option_id": "A",
                                    "plain_interpretation": "complete reading",
                                    "difference_summary": "unique difference",
                                },
                                {
                                    "option_id": "B",
                                    "plain_interpretation": "complete reading",
                                    "difference_summary": "unique difference",
                                },
                            ],
                        }
                    ],
                },
            },
            temperature=0.0,
        )
        if not isinstance(raw, Mapping):
            raise ValueError("plain-reading axis rebuild must be an object")
        parsed_axes = tuple(
            _ambiguity_from_mapping(item)
            for item in raw.get("rebuilt_axes", ())
            if isinstance(item, Mapping)
        )
        axes = tuple(
            replace(
                axis,
                ambiguity_id=_stable_ambiguity_id(request, axis, index),
            )
            for index, axis in enumerate(parsed_axes, start=1)
        )
        validate_rebuilt_plain_reading_axes(source_text, segmentation, axes)
        return axes


def _ambiguity_from_mapping(payload: Mapping) -> PlainReadingAmbiguity:
    options = tuple(
        PlainReadingOption(
            option_id=str(item.get("option_id", "")),
            plain_interpretation=str(item.get("plain_interpretation", "")),
            difference_summary=str(item.get("difference_summary", "")),
        )
        for item in payload.get("options", ())
        if isinstance(item, Mapping)
    )
    if len(options) != 2:
        raise ValueError("every rebuilt ambiguity axis must contain exactly two options")
    return PlainReadingAmbiguity(
        ambiguity_id=str(payload.get("ambiguity_id", "")),
        source_segment_ids=tuple(
            str(item) for item in payload.get("source_segment_ids", ())
        ),
        question=str(payload.get("question", "")),
        options=options,
    )


def _stable_ambiguity_id(
    request: PlainReadingRevisionRequest,
    axis: PlainReadingAmbiguity,
    index: int,
) -> str:
    """Derive retry-stable ids instead of trusting an LLM-generated identifier."""

    semantic_key = "|".join(
        (
            request.request_id,
            str(index),
            axis.question.strip().casefold(),
            *(option.difference_summary.strip().casefold() for option in axis.options),
        )
    )
    digest = hashlib.sha256(semantic_key.encode("utf-8")).hexdigest()[:12]
    return f"amb-revision-{digest}"
