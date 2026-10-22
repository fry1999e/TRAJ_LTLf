"""LangChain-backed JSON client for OpenAI-compatible chat models."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from threading import RLock
import time
from llm.errors import LLMClientError, LLMResponseSchemaError
from llm.schema_validation import check_response_schema, parse_and_validate_json_object
from terminal_progress import model_wait


JSON_OUTPUT_CONSTRAINT = (
    "输出约束：只输出一个 JSON object。第一个非空字符必须是 {，"
    "最后一个非空字符必须是 }。不要输出 Markdown、解释、确认语、"
    "思考过程或任何额外文本。"
)


def _response_format(
    response_schema: dict | None,
    *,
    mode: str = "json_schema",
) -> dict[str, object]:
    if mode == "json_object" or response_schema is None:
        return {"type": "json_object"}
    if mode != "json_schema":
        raise ValueError("response format mode must be json_schema or json_object")
    raw_name = str(response_schema.get("$id", "traj_ltlf_response"))
    name = "".join(character if character.isalnum() or character == "_" else "_" for character in raw_name)
    provider_schema = {
        key: value for key, value in response_schema.items() if key != "$id"
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name[:64] or "traj_ltlf_response",
            "strict": True,
            "schema": provider_schema,
        },
    }


@dataclass(frozen=True)
class LangChainJSONClient:
    """Small adapter matching the agent layer's ``complete_json`` protocol."""

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 1200
    max_retries: int = 2
    single_physical_request: bool = False
    retry_backoff_seconds: float = 0.5
    use_json_response_format: bool = True
    response_format_mode: str = "auto"
    thinking_mode: str | None = None
    reasoning_effort: str | None = None
    use_legacy_max_tokens: bool = False
    # Optional request accounting; omitted for ordinary API requests.
    request_observer: object | None = field(default=None, repr=False, compare=False)
    _response_format_disabled_after_error: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )
    _chat_models: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _transport_lock: object = field(default_factory=RLock, init=False, repr=False, compare=False)
    _direct_client: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mode = str(self.response_format_mode or "auto").strip().lower()
        if mode not in {"auto", "json_schema", "json_object", "none"}:
            raise ValueError(
                "response_format_mode must be auto, json_schema, json_object, or none"
            )
        object.__setattr__(self, "response_format_mode", mode)

    def _effective_response_format_mode(
        self,
        response_schema: dict | None,
    ) -> str:
        if (
            not self.use_json_response_format
            or self.response_format_mode == "none"
            or self._response_format_disabled_after_error
        ):
            return "none"
        if self.response_format_mode in {"json_schema", "json_object"}:
            return self.response_format_mode
        if response_schema is None:
            return "json_object"
        normalized_base = self.base_url.casefold().rstrip("/")
        return (
            "json_schema"
            if normalized_base in {
                "https://api.openai.com",
                "https://api.openai.com/v1",
            }
            else "json_object"
        )

    def complete_json(
        self,
        system_prompt: str,
        user_payload: dict,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
        reasoning_effort: str | None = None,
        thinking_mode: str | None = None,
    ) -> dict:
        """Generate JSON with CLI waiting feedback through retries and repair."""

        with model_wait():
            return self._complete_json(
                system_prompt, user_payload,
                temperature=temperature, max_tokens=max_tokens,
                response_schema=response_schema, reasoning_effort=reasoning_effort,
                thinking_mode=thinking_mode,
            )

    def _complete_json(
        self,
        system_prompt: str,
        user_payload: dict,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
        reasoning_effort: str | None = None,
        thinking_mode: str | None = None,
    ) -> dict:
        """Send one LangChain chat request and parse a JSON object response.

        The configured capability policy selects ``json_schema``,
        ``json_object``, or no provider response format. Local schema
        validation remains authoritative in every mode. Provider capability
        failures are retried without ``response_format`` and cached for this
        client; malformed output receives one bounded JSON repair request.
        """

        if response_schema is not None:
            check_response_schema(response_schema)
        resolved_temperature = self.temperature if temperature is None else temperature
        resolved_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        resolved_reasoning = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        resolved_thinking = self.thinking_mode if thinking_mode is None else thinking_mode
        messages = self._build_messages(system_prompt, user_payload)

        if self.single_physical_request:
            content = self._invoke_chat_model(
                messages, temperature=resolved_temperature, max_tokens=resolved_max_tokens,
                use_json_response_format=self._effective_response_format_mode(response_schema) != 'none',
                response_schema=response_schema, thinking_mode=resolved_thinking,
                reasoning_effort=resolved_reasoning)
            try:
                return _parse_and_validate_response(content, response_schema=response_schema)
            except LLMClientError as exc:
                raise LLMResponseSchemaError(str(exc)) from exc

        # ── first attempt ──────────────────────────────────────────
        first_mode = self._effective_response_format_mode(response_schema)
        first_use_json = first_mode != "none"
        first_result: str | None = None
        first_error: LLMClientError | None = None
        try:
            first_result = self._invoke_chat_model(
                messages,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                use_json_response_format=first_use_json,
                response_schema=response_schema,
                thinking_mode=resolved_thinking,
                reasoning_effort=resolved_reasoning,
            )
            return _parse_and_validate_response(
                first_result,
                response_schema=response_schema,
            )
        except LLMClientError as exc:
            first_error = exc
            if not first_use_json:
                if first_result is not None:
                    try:
                        return self._repair_json_response(
                            system_prompt=system_prompt,
                            user_payload=user_payload,
                            failed_response=first_result,
                            parse_error=first_error,
                            temperature=resolved_temperature,
                            max_tokens=resolved_max_tokens,
                            response_schema=response_schema,
                            reasoning_effort=resolved_reasoning,
                            thinking_mode=resolved_thinking,
                        )
                    except LLMClientError as repair_error:
                        raise _with_json_recovery_hint(
                            repair_error,
                            attempted_fallback=False,
                            attempted_repair=True,
                        ) from repair_error
                raise _with_json_recovery_hint(first_error) from first_error
            if first_result is None and not _should_retry_without_response_format(
                first_error
            ):
                raise _with_json_recovery_hint(
                    first_error,
                    attempted_fallback=False,
                    attempted_repair=False,
                ) from first_error
            if first_result is None and _is_response_format_capability_error(first_error):
                object.__setattr__(
                    self,
                    "_response_format_disabled_after_error",
                    True,
                )

        # ── automatic retry without JSON response_format ───────────
        retry_content: str | None = None
        try:
            retry_content = self._invoke_chat_model(
                messages,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                use_json_response_format=False,
                response_schema=None,
                thinking_mode=resolved_thinking,
                reasoning_effort=resolved_reasoning,
            )
        except LLMClientError as retry_api_error:
            first_detail = _format_first_attempt_detail(first_result, first_error)
            raise LLMClientError(
                f"{_with_json_recovery_hint(retry_api_error, attempted_repair=False)} "
                f"首次尝试(JSON response_format)详情: {first_detail}"
            ) from retry_api_error

        try:
            return _parse_and_validate_response(
                retry_content,
                response_schema=response_schema,
            )
        except LLMClientError as retry_error:
            repair_failure: LLMClientError | None = None
            try:
                return self._repair_json_response(
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    failed_response=retry_content,
                    parse_error=retry_error,
                    temperature=resolved_temperature,
                    max_tokens=resolved_max_tokens,
                    response_schema=response_schema,
                    reasoning_effort=resolved_reasoning,
                    thinking_mode=resolved_thinking,
                )
            except LLMClientError as exc:
                repair_failure = exc

            first_detail = _format_first_attempt_detail(first_result, first_error)
            recovery_error = _with_json_recovery_hint(
                repair_failure,
                attempted_repair=True,
            )
            message = (
                f"{recovery_error} "
                f"首次尝试(JSON response_format)详情: {first_detail}"
            )
            if isinstance(recovery_error, LLMResponseSchemaError):
                raise LLMResponseSchemaError(
                    message,
                    path=recovery_error.path,
                    validator=recovery_error.validator,
                ) from repair_failure
            raise LLMClientError(message) from repair_failure

    def _invoke_chat_model(
        self,
        messages: object,
        *,
        temperature: float,
        max_tokens: int,
        use_json_response_format: bool,
        response_schema: dict | None = None,
        thinking_mode: str | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        max_attempts = max(1, self.max_retries + 1)
        for attempt in range(1, max_attempts + 1):
            chat_model = self._build_chat_model(
                temperature=temperature,
                max_tokens=max_tokens,
                use_json_response_format=use_json_response_format,
                response_schema=response_schema,
                thinking_mode=thinking_mode,
                reasoning_effort=reasoning_effort,
            )
            try:
                response = (
                    self.request_observer.invoke(lambda: chat_model.invoke(messages), messages, max_tokens)
                    if self.request_observer is not None else chat_model.invoke(messages)
                )
                return _message_content_to_text(getattr(response, "content", response))
            except Exception as exc:  # LangChain wraps provider-specific exceptions.
                if getattr(exc, "stop_retries", False):
                    raise
                invoke_error: Exception = exc
                if _should_bypass_langchain_response_format_parser(
                    exc,
                    use_json_response_format=use_json_response_format,
                ):
                    try:
                        return self._invoke_openai_chat_completion(
                            messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            use_json_response_format=use_json_response_format,
                            response_schema=response_schema,
                            thinking_mode=thinking_mode,
                            reasoning_effort=reasoning_effort,
                        )
                    except Exception as direct_exc:
                        if getattr(direct_exc, "stop_retries", False):
                            raise
                        invoke_error = _combine_langchain_and_direct_errors(
                            exc,
                            direct_exc,
                        )
                if attempt < max_attempts and _is_retryable_api_exception(invoke_error):
                    if self.request_observer is not None:
                        self.request_observer.before_retry(invoke_error, self.retry_backoff_seconds * (2 ** (attempt - 1)))
                    else:
                        _sleep_before_retry(attempt, self.retry_backoff_seconds)
                    continue
                raise LLMClientError(
                    _format_api_failure(
                        invoke_error,
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )
                ) from invoke_error

        raise LLMClientError("LLM API request failed.")

    def _invoke_openai_chat_completion(
        self,
        messages: object,
        *,
        temperature: float,
        max_tokens: int,
        use_json_response_format: bool,
        response_schema: dict | None = None,
        thinking_mode: str | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMClientError(
                "API 模式需要安装 openai。请先运行 "
                "python -m pip install -r requirements.txt。"
            ) from exc

        payload: dict[str, object] = {
            "model": self.model,
            "messages": _messages_to_openai_payload(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        extra_body = self._build_extra_body(
            max_tokens=max_tokens,
            thinking_mode=thinking_mode,
        )
        if extra_body:
            payload["extra_body"] = extra_body
        if thinking_mode == "enabled" and reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if use_json_response_format:
            payload["response_format"] = _response_format(
                response_schema,
                mode=self._effective_response_format_mode(response_schema),
            )

        with self._transport_lock:
            if self._direct_client is None:
                object.__setattr__(self, "_direct_client", OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                    max_retries=0,
                ))
            client = self._direct_client
        response = (
            self.request_observer.invoke(lambda: client.chat.completions.create(**payload), messages, max_tokens)
            if self.request_observer is not None else client.chat.completions.create(**payload)
        )
        return _extract_openai_chat_completion_text(response)

    def _build_chat_model(
        self,
        *,
        temperature: float,
        max_tokens: int,
        use_json_response_format: bool,
        response_schema: dict | None = None,
        thinking_mode: str | None = None,
        reasoning_effort: str | None = None,
    ):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise LLMClientError(
                "API 模式需要安装 langchain-openai。请先运行 "
                "python -m pip install -r requirements.txt。"
            ) from exc

        chat_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.timeout_seconds,
            "max_retries": 0,
        }
        extra_body = self._build_extra_body(
            max_tokens=max_tokens,
            thinking_mode=thinking_mode,
        )
        if extra_body:
            chat_kwargs["extra_body"] = extra_body
        if thinking_mode == "enabled" and reasoning_effort:
            chat_kwargs["reasoning_effort"] = reasoning_effort

        # Keep connections alive across stages and retries. Schemas are bound
        # per request so one stage cannot inherit another stage's output shape.
        key = (temperature, max_tokens, thinking_mode, reasoning_effort)
        with self._transport_lock:
            chat_model = self._chat_models.get(key)
            if chat_model is None:
                chat_model = ChatOpenAI(**chat_kwargs)
                if len(self._chat_models) >= 16:
                    self._chat_models.pop(next(iter(self._chat_models)))
                self._chat_models[key] = chat_model
        if use_json_response_format:
            return chat_model.bind(
                response_format=_response_format(
                    response_schema,
                    mode=self._effective_response_format_mode(response_schema),
                )
            )
        return chat_model

    def _build_extra_body(
        self,
        *,
        max_tokens: int,
        thinking_mode: str | None,
    ) -> dict[str, object]:
        extra_body: dict[str, object] = {}
        if thinking_mode in {"enabled", "disabled"}:
            extra_body["thinking"] = {"type": thinking_mode}
        if self.use_legacy_max_tokens:
            extra_body["max_tokens"] = max_tokens
        return extra_body

    def _build_messages(self, system_prompt: str, user_payload: dict):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as exc:
            raise LLMClientError(
                "API 模式需要安装 langchain-core。请先运行 "
                "python -m pip install -r requirements.txt。"
            ) from exc

        return [
            SystemMessage(content=_with_json_output_constraint(system_prompt)),
            HumanMessage(content=json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))),
        ]

    def _repair_json_response(
        self,
        *,
        system_prompt: str,
        user_payload: dict,
        failed_response: str,
        parse_error: LLMClientError,
        temperature: float,
        max_tokens: int,
        response_schema: dict | None,
        reasoning_effort: str | None,
        thinking_mode: str | None,
    ) -> dict:
        repair_messages = self._build_json_repair_messages(
            system_prompt=system_prompt,
            user_payload=user_payload,
            failed_response=failed_response,
            parse_error=parse_error,
            response_schema=response_schema,
        )
        repair_content = self._invoke_chat_model(
            repair_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            use_json_response_format=False,
            response_schema=None,
            thinking_mode=(
                "disabled" if self.use_legacy_max_tokens else thinking_mode
            ),
            reasoning_effort=reasoning_effort,
        )
        return _parse_and_validate_response(
            repair_content,
            response_schema=response_schema,
        )

    def _repair_thinking_mode(self) -> str | None:
        if self.use_legacy_max_tokens:
            return "disabled"
        return self.thinking_mode

    def _build_json_repair_messages(
        self,
        *,
        system_prompt: str,
        user_payload: dict,
        failed_response: str,
        parse_error: LLMClientError,
        response_schema: dict | None,
    ):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as exc:
            raise LLMClientError(
                "API 模式需要安装 langchain-core。请先运行 "
                "python -m pip install -r requirements.txt。"
            ) from exc

        repair_payload = {
            "task": "rewrite_previous_answer_as_json_object",
            "original_system_prompt": system_prompt,
            "original_user_payload": user_payload,
            "invalid_response": failed_response,
            "parse_error": str(parse_error),
            "response_schema": response_schema,
            "required_output": (
                "Return exactly one valid JSON object matching the original task "
                "and response_schema when one is provided. "
                "Do not include Markdown, explanations, apologies, or any text "
                "outside the JSON object."
            ),
        }
        return [
            SystemMessage(
                content=(
                    "你是 JSON 修复器。你的唯一任务是把上一次模型输出改写成"
                    "一个合法 JSON object，并保持原任务语义。"
                    f"{JSON_OUTPUT_CONSTRAINT}"
                )
            ),
            HumanMessage(content=json.dumps(repair_payload, ensure_ascii=False)),
        ]


def _message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    raise LLMClientError(f"LLM response content is not text: {type(content).__name__}")


def _messages_to_openai_payload(messages: object) -> list[dict[str, str]]:
    if isinstance(messages, (list, tuple)):
        message_items = list(messages)
    else:
        message_items = [messages]

    payload: list[dict[str, str]] = []
    for index, message in enumerate(message_items):
        if isinstance(message, dict):
            raw_role = message.get("role") or message.get("type")
            content = message.get("content", "")
        else:
            raw_role = getattr(message, "role", None) or getattr(
                message,
                "type",
                None,
            )
            content = getattr(message, "content", message)
        role = _openai_role_from_message(raw_role, index=index)
        payload.append({"role": role, "content": _message_content_to_text(content)})
    return payload


def _openai_role_from_message(raw_role: object, *, index: int) -> str:
    role = str(raw_role or "").lower()
    role_map = {
        "ai": "assistant",
        "assistant": "assistant",
        "human": "user",
        "user": "user",
        "system": "system",
        "developer": "developer",
        "tool": "tool",
    }
    if role in role_map:
        return role_map[role]
    return "system" if index == 0 else "user"


def _extract_openai_chat_completion_text(response: object) -> str:
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return response
        if isinstance(parsed, dict) and "choices" in parsed:
            return _extract_openai_chat_completion_text(parsed)
        return response

    if isinstance(response, dict):
        return _extract_openai_response_dict_text(response)

    choices = getattr(response, "choices", None)
    if choices:
        return _extract_openai_choice_text(choices[0])

    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return _extract_openai_response_dict_text(dumped)

    return _message_content_to_text(getattr(response, "content", response))


def _extract_openai_response_dict_text(response: dict) -> str:
    if response.get("error"):
        raise LLMClientError(f"LLM API returned error: {response['error']}")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMClientError("LLM API response did not include choices.")
    return _extract_openai_choice_text(choices[0])


def _extract_openai_choice_text(choice: object) -> str:
    if isinstance(choice, dict):
        message = choice.get("message")
        if isinstance(message, dict):
            return _message_content_to_text(message.get("content", ""))
        if "text" in choice:
            return _message_content_to_text(choice["text"])
        return _message_content_to_text(message)

    message = getattr(choice, "message", None)
    if message is not None:
        return _message_content_to_text(getattr(message, "content", message))
    if hasattr(choice, "text"):
        return _message_content_to_text(getattr(choice, "text"))
    return _message_content_to_text(choice)


def _parse_json_object(text: str) -> dict:
    value = _decode_first_json_value(text)

    if not isinstance(value, dict):
        raise LLMClientError("LLM response must be a JSON object.")
    return value


def _parse_and_validate_response(
    text: str,
    *,
    response_schema: dict | None,
) -> dict:
    return parse_and_validate_json_object(
        text,
        response_schema=response_schema,
        parser=_parse_json_object,
    )


def _decode_first_json_value(text: str) -> object:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if not stripped:
        raise LLMClientError("LLM response did not contain JSON: <empty>")

    try:
        value, _end = decoder.raw_decode(stripped)
        return value
    except json.JSONDecodeError as exc:
        if stripped.startswith("{"):
            raise LLMClientError(
                "LLM response began with a JSON object but it was invalid "
                f"or incomplete: {_truncate(text)}"
            ) from exc

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue

    raise LLMClientError(
        f"LLM response did not contain valid JSON object: {_truncate(text)}"
    )


def _with_json_output_constraint(system_prompt: str) -> str:
    return f"{system_prompt}\n\n{JSON_OUTPUT_CONSTRAINT}"


def _truncate(text: str, limit: int = 500) -> str:
    normalized = text.replace("\n", " ").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


def _format_first_attempt_detail(
    content: str | None, error: LLMClientError | None
) -> str:
    if content is not None:
        return _truncate(content)
    if error is not None:
        return _truncate(str(error))
    return "<首次请求未返回内容>"


def _should_retry_without_response_format(error: LLMClientError) -> bool:
    text = str(error).lower()
    if _looks_like_connection_error_text(text):
        return False

    blocking_markers = [
        "authentication",
        "unauthorized",
        "invalid api key",
        "permission",
        "quota",
        "rate limit",
        "too many requests",
        "model_not_found",
        "model not found",
        "does not exist",
    ]
    if any(marker in text for marker in blocking_markers):
        return False

    response_format_markers = [
        "response_format",
        "json_object",
        "json mode",
        "has no attribute 'choices'",
        'has no attribute "choices"',
        "lengthfinishreasonerror",
        "length limit",
        "finish_reason=\"length\"",
        "finish_reason='length'",
        "unsupported parameter",
        "not support",
        "not supported",
        "extra fields not permitted",
    ]
    return any(marker in text for marker in response_format_markers)


def _is_response_format_capability_error(error: LLMClientError) -> bool:
    """Return whether the provider explicitly rejected response_format support."""

    text = str(error).lower()
    capability_markers = (
        "unsupported parameter: response_format",
        "response_format type is unavailable",
        "response_format is unavailable",
        "response_format is not supported",
        "response_format not supported",
        "unsupported response_format",
        "extra fields not permitted",
    )
    return any(marker in text for marker in capability_markers)


def _should_bypass_langchain_response_format_parser(
    exc: Exception,
    *,
    use_json_response_format: bool,
) -> bool:
    if not use_json_response_format:
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        isinstance(exc, AttributeError)
        and (
            "has no attribute 'choices'" in text
            or 'has no attribute "choices"' in text
        )
    )


def _combine_langchain_and_direct_errors(
    langchain_exc: Exception,
    direct_exc: Exception,
) -> LLMClientError:
    langchain_detail = _truncate(f"{type(langchain_exc).__name__}: {langchain_exc}")
    direct_detail = _truncate(f"{type(direct_exc).__name__}: {direct_exc}")
    return LLMClientError(
        "LangChain response_format parser failed: "
        f"{langchain_detail}; direct OpenAI SDK fallback failed: {direct_detail}"
    )


def _is_retryable_api_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    text = f"{type(exc).__name__}: {exc}".lower()
    retryable_markers = [
        "rate limit",
        "too many requests",
        "server error",
        "service unavailable",
        "temporarily unavailable",
    ]
    return _looks_like_connection_error_text(text) or any(
        marker in text for marker in retryable_markers
    )


def _looks_like_connection_error_text(text: str) -> bool:
    connection_markers = [
        "connection error",
        "apiconnectionerror",
        "api connection",
        "connecterror",
        "connection reset",
        "connection aborted",
        "remoteprotocolerror",
        "timeout",
        "timed out",
        "read timeout",
        "ssl",
        "tls",
        "dns",
        "name resolution",
        "proxy",
    ]
    return any(marker in text for marker in connection_markers)


def _sleep_before_retry(attempt: int, backoff_seconds: float) -> None:
    if backoff_seconds <= 0:
        return
    time.sleep(min(backoff_seconds * (2 ** (attempt - 1)), 8.0))


def _format_api_failure(
    exc: Exception, *, attempt: int, max_attempts: int
) -> str:
    if isinstance(exc, LLMClientError):
        detail = _truncate(str(exc))
    else:
        detail = _truncate(f"{type(exc).__name__}: {exc}")
    if max_attempts > 1 and attempt >= max_attempts:
        return f"LLM API request failed after {max_attempts} attempts: {detail}"
    return f"LLM API request failed: {detail}"


def _with_json_recovery_hint(
    error: LLMClientError,
    *,
    attempted_fallback: bool = True,
    attempted_repair: bool = False,
) -> LLMClientError:
    recovery_steps: list[str] = []
    if attempted_fallback:
        recovery_steps.append("系统已自动尝试不带 response_format 的请求。")
    if attempted_repair:
        recovery_steps.append("系统已自动尝试 JSON 修复请求。")
    if not recovery_steps:
        recovery_steps.append("系统未能从本次响应中恢复 JSON object。")

    hints: list[str] = []
    if isinstance(error, LLMResponseSchemaError):
        hints.append(
            "响应已经解析为 JSON object，但未通过本地 response_schema 验证。"
        )
        hints.append(
            "这不是 response_format 能力问题；请核对错误中的实际路径、操作与允许项。"
        )
    elif _looks_like_connection_error_text(str(error).lower()):
        hints.append(
            "这更像是网络、API base URL、代理、TLS 或服务端连接问题，"
            "不是 JSON 输出格式问题。"
        )
        hints.append(
            "请确认当前运行的 Python 环境能访问 LLM_API_BASE_URL，"
            "并按需配置 HTTPS_PROXY/HTTP_PROXY。"
        )
    else:
        hints.append("如果仍失败，请确认模型支持输出纯 JSON object。")
        hints.append(
            "如果目标服务不支持 response_format，可在 .env 中设置 "
            "LLM_DISABLE_RESPONSE_FORMAT=1 跳过首次 JSON mode 尝试。"
        )

    message = " ".join([str(error), *recovery_steps, *hints])
    if isinstance(error, LLMResponseSchemaError):
        return LLMResponseSchemaError(
            message,
            path=error.path,
            validator=error.validator,
        )
    return LLMClientError(message)
