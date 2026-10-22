"""
Agent 基类。

项目支持两种运行方式：
    1. rules：使用本地规则，便于离线演示和单元测试。
    2. api：通过统一 LLM client 调用真实 OpenAI-compatible API。
"""

from __future__ import annotations

from typing import Protocol

from llm.errors import LLMClientError
from rag.retriever import KeywordRetriever


class JSONLLMClient(Protocol):
    """Agent 需要的最小 LLM client 接口。"""

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
        """Return a JSON object from one LLM call."""


class BaseAgent:
    """所有 Traj-LTLf Agent 的公共父类。"""

    def __init__(
        self,
        retriever: KeywordRetriever,
        name: str,
        llm_client: JSONLLMClient | None = None,
    ):
        self.retriever = retriever
        self.name = name
        self.llm_client = llm_client

    @property
    def llm_enabled(self) -> bool:
        """是否使用真实 LLM API 生成本阶段结果。"""

        return self.llm_client is not None

    def retrieve_context(
        self,
        *parts: str,
        top_k: int = 5,
        source_filter: str | tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        """
        使用 RAG 检索和当前任务相关的知识片段。

        parts 可以包含自然语言需求、阶段名称、语义结构、人工评审意见等。
        """

        query = "\n".join(part for part in parts if part)
        chunks = self.retriever.retrieve(
            query,
            top_k=top_k,
            source_filter=source_filter,
        )
        return tuple(f"{chunk.title}: {chunk.text}" for chunk in chunks)

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
        """Call the configured LLM client and require a JSON object response."""

        if self.llm_client is None:
            raise RuntimeError(f"{self.name} 没有配置 LLM API client。")
        kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_schema is not None:
            kwargs["response_schema"] = response_schema
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if thinking_mode is not None:
            kwargs["thinking_mode"] = thinking_mode
        try:
            result = self.llm_client.complete_json(
                system_prompt=system_prompt,
                user_payload=user_payload,
                **kwargs,
            )
        except TypeError as exc:
            if response_schema is None or "response_schema" not in str(exc):
                raise
            raise LLMClientError(
                f"{self.name} configured LLM client does not implement the required "
                "response_schema contract."
            ) from exc
        return result
