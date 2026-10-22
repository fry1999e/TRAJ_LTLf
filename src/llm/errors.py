"""Shared exceptions for LLM integrations."""


class LLMClientError(RuntimeError):
    """Raised when the LLM API call, configuration, or response parsing fails."""


class LLMResponseSchemaError(LLMClientError):
    """Raised when a parsed LLM response violates its declared JSON Schema."""

    def __init__(
        self,
        message: str,
        *,
        path: str = "",
        validator: str = "",
    ) -> None:
        super().__init__(message)
        self.path = path
        self.validator = validator
