"""Physical separation and deterministic validation for source quotations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SourceQuoteValidationError(ValueError):
    pointer: str
    reason_code: str
    message: str

    def __str__(self) -> str:
        return f"{self.reason_code} at {self.pointer}: {self.message}"


def validate_source_quotes(
    quotes: object, raw_nl: str
) -> tuple[dict[str, Any], ...]:
    """Return source-bound span records; contract/profile text is never accepted."""

    if not isinstance(quotes, Sequence) or isinstance(quotes, (str, bytes)) or not quotes:
        raise SourceQuoteValidationError(
            "/source_quotes", "SOURCE_QUOTES_REQUIRED", "expected a nonempty array"
        )
    result = []
    for index, item in enumerate(quotes):
        pointer = f"/source_quotes/{index}"
        if isinstance(item, str):
            quote = item
            start = raw_nl.find(quote)
            if not quote.strip() or start < 0:
                raise SourceQuoteValidationError(
                    pointer, "SOURCE_QUOTE_NOT_RAW_SPAN", "quote is not a raw_nl substring"
                )
            if raw_nl.find(quote, start + 1) >= 0:
                raise SourceQuoteValidationError(
                    pointer,
                    "SOURCE_QUOTE_SPAN_AMBIGUOUS",
                    "repeated text requires explicit start/end coordinates",
                )
            end = start + len(quote)
        elif isinstance(item, Mapping):
            if set(item) != {"start", "end", "text"}:
                raise SourceQuoteValidationError(
                    pointer, "SOURCE_QUOTE_SHAPE_INVALID", "allowed keys are start/end/text"
                )
            start, end, quote = item.get("start"), item.get("end"), item.get("text")
            if type(start) is not int or type(end) is not int:
                raise SourceQuoteValidationError(
                    pointer, "SOURCE_QUOTE_COORDINATES_INVALID", "start/end must be integers"
                )
            if not isinstance(quote, str) or not quote.strip():
                raise SourceQuoteValidationError(
                    pointer + "/text", "SOURCE_QUOTE_TEXT_INVALID", "text must be nonempty"
                )
        else:
            raise SourceQuoteValidationError(
                pointer, "SOURCE_QUOTE_SHAPE_INVALID", "quote must be a span object"
            )
        if not (0 <= start < end <= len(raw_nl)) or raw_nl[start:end] != quote:
            raise SourceQuoteValidationError(
                pointer,
                "SOURCE_QUOTE_NOT_RAW_SPAN",
                f"legal coordinate range is 0..{len(raw_nl)} and text must equal raw_nl[start:end]",
            )
        result.append({"start": start, "end": end, "text": quote})
    return tuple(result)


def quote_repair_context(error: SourceQuoteValidationError, raw_nl: str) -> dict[str, Any]:
    return {
        "invalid_pointer": error.pointer,
        "reason_code": error.reason_code,
        "raw_nl": raw_nl,
        "legal_span": {"minimum": 0, "maximum": len(raw_nl), "end_exclusive": True},
        "instruction": "Repair source_quotes only; preserve decision and reason exactly.",
    }


__all__ = [
    "SourceQuoteValidationError",
    "quote_repair_context",
    "validate_source_quotes",
]
