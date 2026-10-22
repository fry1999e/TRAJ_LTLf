"""Provider-independent validation for structured LLM responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from llm.errors import LLMClientError, LLMResponseSchemaError


def check_response_schema(schema: Mapping[str, Any]) -> None:
    """Reject an invalid response schema before any provider call is made."""

    _checked_validator_for(schema)


def validate_response_schema(value: object, schema: Mapping[str, Any]) -> None:
    """Require ``value`` to satisfy ``schema`` regardless of provider behavior."""

    validator_class = _checked_validator_for(schema)

    errors = sorted(
        validator_class(schema).iter_errors(value),
        key=lambda item: tuple(str(token) for token in item.absolute_path),
    )
    if not errors:
        return

    error = errors[0]
    path = _json_pointer(tuple(error.absolute_path))
    location = path or "/"
    validator = str(error.validator or "")
    detail = _schema_error_detail(error)
    raise LLMResponseSchemaError(
        f"LLM response violates response_schema at {location} "
        f"({validator or 'schema'}): {detail}",
        path=path,
        validator=validator,
    )


def _checked_validator_for(schema: Mapping[str, Any]):
    """Return the applicable validator class after validating the schema itself."""

    try:
        from jsonschema import SchemaError
        from jsonschema.validators import validator_for
    except ImportError as exc:  # pragma: no cover - packaging guard
        raise LLMClientError(
            "Structured LLM responses require the direct dependency 'jsonschema'."
        ) from exc

    try:
        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
    except SchemaError as exc:
        raise LLMClientError(
            "Configured LLM response_schema is invalid: " + _compact(exc.message)
        ) from exc
    return validator_class


def parse_and_validate_json_object(
    text: str,
    *,
    response_schema: Mapping[str, Any] | None,
    parser,
) -> dict[str, Any]:
    """Parse one JSON object and enforce an optional response schema."""

    value = parser(text)
    if response_schema is not None:
        validate_response_schema(value, response_schema)
    return value


def _json_pointer(tokens: tuple[object, ...]) -> str:
    if not tokens:
        return ""
    return "/" + "/".join(
        str(token).replace("~", "~0").replace("/", "~1") for token in tokens
    )


def _compact(value: object, *, limit: int = 500) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _schema_error_detail(error: object) -> str:
    message = _compact(getattr(error, "message", "schema validation failed"))
    if getattr(error, "validator", "") != "oneOf":
        return message
    instance = getattr(error, "instance", None)
    variants = getattr(error, "validator_value", ())
    if not isinstance(instance, Mapping) or not isinstance(variants, list):
        return message
    received_op = str(instance.get("op", ""))
    received_path = str(instance.get("path", ""))
    allowed: list[str] = []
    for variant in variants:
        if not isinstance(variant, Mapping):
            continue
        properties = variant.get("properties")
        if not isinstance(properties, Mapping):
            continue
        op_schema = properties.get("op")
        path_schema = properties.get("path")
        if not isinstance(op_schema, Mapping) or not isinstance(path_schema, Mapping):
            continue
        operation = op_schema.get("const")
        path = path_schema.get("const")
        if operation is None or path is None:
            continue
        allowed.append(f"{operation} {path}")
    if not allowed:
        return message
    received = f"{received_op} {received_path}".strip()
    return _compact(
        f"{message}; received operation/path: {received}; "
        "allowed operation/path pairs: " + ", ".join(allowed),
        limit=1200,
    )


__all__ = [
    "check_response_schema",
    "parse_and_validate_json_object",
    "validate_response_schema",
]
