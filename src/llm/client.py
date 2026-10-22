"""Build LLM clients for Traj-LTLf agents from environment variables.

The public factory keeps ``rules`` mode dependency-light. LangChain and
python-dotenv are imported only when API mode needs them, so offline tests can
still run without third-party packages.
"""

from __future__ import annotations

import os
from pathlib import Path

from llm.errors import LLMClientError
from llm.langchain_client import LangChainJSONClient


# Backward-compatible name kept for older imports. The implementation now uses
# LangChain under the hood.
OpenAICompatibleClient = LangChainJSONClient

_NETWORK_ENV_KEYS = {
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
}


def build_llm_client_from_env(mode: str | None = None) -> LangChainJSONClient | None:
    """
    Build an API client from environment variables.

    mode:
        "rules" returns None.
        "api" requires an API key and LLM_MODEL.
        None reads TRAJ_LTLF_LLM_MODE or LLM_MODE, defaulting to "rules".
    """

    _load_dotenv_if_available()

    selected_mode = (
        mode or os.getenv("TRAJ_LTLF_LLM_MODE") or os.getenv("LLM_MODE") or "rules"
    ).lower()
    if selected_mode in {"rules", "rule", "local", "off", "none"}:
        return None
    if selected_mode != "api":
        raise LLMClientError(
            "未知 LLM 模式。请使用 rules 或 api，"
            f"当前为: {selected_mode}"
        )

    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("DEEPSEEK_MODEL")
    if not api_key:
        raise LLMClientError(
            "API 模式需要设置 LLM_API_KEY、OPENAI_API_KEY 或 DEEPSEEK_API_KEY。"
        )
    if not model:
        raise LLMClientError("API 模式需要设置 LLM_MODEL。")

    base_url = _resolve_base_url()
    return LangChainJSONClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=_float_env("LLM_TIMEOUT_SECONDS", 60.0),
        temperature=_float_env("LLM_TEMPERATURE", 0.0),
        max_tokens=_int_env("LLM_MAX_TOKENS", 1200),
        max_retries=max(0, _int_env("LLM_MAX_RETRIES", 2)),
        retry_backoff_seconds=max(
            0.0, _float_env("LLM_RETRY_BACKOFF_SECONDS", 0.5)
        ),
        use_json_response_format=os.getenv("LLM_DISABLE_RESPONSE_FORMAT", "").lower()
        not in {"1", "true", "yes"},
        response_format_mode=_response_format_mode_env("LLM"),
        thinking_mode=_resolve_thinking_mode(model=model, base_url=base_url),
        reasoning_effort=_optional_env("LLM_REASONING_EFFORT"),
        use_legacy_max_tokens=_is_deepseek_target(model=model, base_url=base_url),
    )


def build_judge_client_from_env() -> LangChainJSONClient:
    """Compatibility name: reviewer models are disabled in this release."""
    raise LLMClientError("Model reviewers are disabled; use manual review in human mode.")


def _load_dotenv_if_available() -> None:
    _load_dotenv_network_env_if_available()
    _load_dotenv_file_if_available()




def _load_dotenv_file_if_available(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_dotenv_fallback(path)
        return

    load_dotenv(dotenv_path=env_path, override=False)


def _load_dotenv_network_env_if_available(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        _load_dotenv_fallback(path, allowed_keys=_NETWORK_ENV_KEYS)
        return

    for key, value in dotenv_values(env_path).items():
        if key not in _NETWORK_ENV_KEYS or value is None or key in os.environ:
            continue
        os.environ[key] = str(value)


def _has_llm_env() -> bool:
    keys = {
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "LLM_MODEL",
        "OPENAI_MODEL",
        "DEEPSEEK_MODEL",
        "LLM_API_BASE_URL",
        "OPENAI_API_BASE_URL",
        "OPENAI_API_URL",
        "DEEPSEEK_API_BASE_URL",
        "DEEPSEEK_API_URL",
    }
    return any(os.getenv(key) for key in keys)




def _load_dotenv_fallback(
    path: str = ".env",
    *,
    allowed_keys: set[str] | None = None,
) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if allowed_keys is not None and key not in allowed_keys:
            continue
        if not key or key in os.environ:
            continue
        value = value.strip()
        if "←" in value:
            value = value.split("←", 1)[0].strip()
        if "#" in value and not (value.startswith('"') or value.startswith("'")):
            value = value.split("#", 1)[0].strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        os.environ[key] = value


def _resolve_base_url() -> str:
    explicit_base_url = (
        os.getenv("LLM_API_BASE_URL")
        or os.getenv("OPENAI_API_BASE_URL")
        or os.getenv("OPENAI_API_URL")
        or os.getenv("DEEPSEEK_API_BASE_URL")
        or os.getenv("DEEPSEEK_API_URL")
    )
    if explicit_base_url:
        return explicit_base_url

    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        return "https://api.deepseek.com"
    return "https://api.openai.com/v1"


def _resolve_thinking_mode(*, model: str, base_url: str) -> str | None:
    raw_value = (
        os.getenv("LLM_THINKING_MODE")
        or os.getenv("DEEPSEEK_THINKING_MODE")
        or ""
    ).strip().lower()
    if raw_value in {"", "auto"}:
        return "enabled" if _is_deepseek_target(model=model, base_url=base_url) else None
    if raw_value in {"default", "provider-default", "none"}:
        return None
    if raw_value in {"1", "true", "yes", "on", "enabled", "enable"}:
        return "enabled"
    if raw_value in {"0", "false", "no", "off", "disabled", "disable"}:
        return "disabled"
    raise LLMClientError(
        "LLM_THINKING_MODE must be one of auto, default, enabled, or disabled."
    )




def _is_deepseek_target(*, model: str, base_url: str) -> bool:
    return "deepseek.com" in base_url.lower() or model.lower().startswith("deepseek-")


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _response_format_mode_env(prefix: str) -> str:
    value = (os.getenv(f"{prefix}_RESPONSE_FORMAT_MODE") or "auto").strip().lower()
    if value not in {"auto", "json_schema", "json_object", "none"}:
        raise LLMClientError(
            f"{prefix}_RESPONSE_FORMAT_MODE must be auto, json_schema, "
            "json_object, or none."
        )
    return value


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise LLMClientError(f"{name} must be a number.") from exc


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise LLMClientError(f"{name} must be an integer.") from exc
