"""Shared transport exception types used by both formal and legacy callers."""

from llm.errors import LLMClientError


class OutputTruncatedError(LLMClientError):
    """A response ended at its output or reasoning cap."""

    reason_code = 'OUTPUT_TRUNCATED'
