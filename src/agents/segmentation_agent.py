"""Agent that may only insert lossless surface-boundary markers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping

from agents.base_agent import BaseAgent, JSONLLMClient
from schemas import (
    ArtifactGenerator,
    MechanicalSegmentationValidation,
    SurfaceSegmentation,
)
from generation_schemas import SEGMENTATION_PROPOSAL_SCHEMA
from surface_segmentation import (
    SegmentationValidationError,
    build_surface_segmentation,
    choose_cut_marker,
    deterministic_marked_text,
    source_hash,
    validate_marked_text,
)
from validated_generation import (
    CANONICALIZER_VERSION,
    CONTRACT_VERSION,
    REPAIR_POLICY_VERSION,
    VALIDATOR_VERSION,
    PatchOperation,
    RepairClass,
    RepairPolicy,
    ValidatedGenerationRunner,
    ValidationIssue,
    artifact_hash,
    canonical_json,
)


class SegmentationAgent(BaseAgent):
    """Generate marked text under the shared validation and progress controller."""

    def __init__(self, retriever, llm_client: JSONLLMClient | None = None):
        super().__init__(retriever=retriever, name="SegmentationAgent", llm_client=llm_client)
        self.last_generation_diagnostic: dict[str, Any] = {}

    def generate(
        self,
        requirement: str,
        *,
        requirement_id: str = "REQ-001",
        max_retries: int = 2,
        prompt_version: str = "plain-segmentation-v1",
        experiment_mode: str = "rules",
        data_split: str = "test",
        guided_context: object | None = None,
        generation_diagnostic_callback: Callable[[Mapping[str, Any]], None]
        | None = None,
    ) -> SurfaceSegmentation:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        marker = choose_cut_marker(requirement)
        generator = ArtifactGenerator(
            mode="api" if self.llm_enabled else "rules",
            model="",
            prompt_version=prompt_version,
        )
        if not self.llm_enabled:
            marked_text, marker = deterministic_marked_text(requirement, marker)
            return build_surface_segmentation(
                requirement_id=requirement_id,
                source_text=requirement,
                marked_text=marked_text,
                marker=marker,
                status="review_pending",
                generator=generator,
                experiment_mode=experiment_mode,
                data_split=data_split,
                label_source="rules",
                review_basis="mechanical_and_rules",
            )

        max_attempts = max_retries + 1
        current_attempt = 0
        previous_error: dict[str, Any] | None = None

        def call_model() -> dict[str, Any]:
            nonlocal current_attempt
            current_attempt += 1
            payload: dict[str, Any] = {
                "task": "lossless_surface_segmentation",
                "source_text": requirement,
                "marker": marker,
                "attempt": current_attempt,
                "output_schema": {"marked_text": "exact source with marker insertions only"},
            }
            if previous_error is not None:
                payload["previous_mechanical_error"] = previous_error
            if guided_context is not None and hasattr(guided_context, "prompt_payload"):
                payload["guided_learning"] = guided_context.prompt_payload(
                    "segmentation"
                )
            return self.complete_json(
                _segmentation_system_prompt(marker),
                payload,
                temperature=0.0,
                response_schema=SEGMENTATION_PROPOSAL_SCHEMA,
            )

        def validate_proposal(raw: object) -> tuple[ValidationIssue, ...]:
            nonlocal previous_error
            marked_text = raw.get("marked_text") if isinstance(raw, dict) else None
            if not isinstance(marked_text, str):
                validation = _invalid_output_shape_validation(raw)
                previous_error = {
                    "failure_type": "invalid_output_shape",
                    "first_difference_offset": None,
                    "missing_or_changed_excerpt": validation.missing_or_changed_excerpt,
                    "instruction": "只输出包含字符串 marked_text 的 JSON 对象。",
                }
                _emit_generation_diagnostic(
                    generation_diagnostic_callback,
                    requirement_id=requirement_id,
                    requirement=requirement,
                    marker=marker,
                    attempt=current_attempt,
                    max_attempts=max_attempts,
                    validation=validation,
                    raw=raw,
                    fallback_used=current_attempt == max_attempts,
                )
                return (
                    ValidationIssue(
                        code="INVALID_OUTPUT_SHAPE",
                        stage="segmentation",
                        path="/marked_text",
                        observed=_sanitized_rejected_response(raw),
                        expected={"type": "string"},
                        context=previous_error,
                        repair_class=RepairClass.CONSTRAINED,
                        allowed_paths=("/marked_text",),
                        message=validation.instruction,
                    ),
                )
            validation = validate_marked_text(requirement, marked_text, marker)
            if validation.status != "passed":
                previous_error = {
                    "failure_type": validation.failure_type,
                    "first_difference_offset": validation.first_difference_offset,
                    "missing_or_changed_excerpt": validation.missing_or_changed_excerpt,
                    "instruction": validation.instruction,
                }
                _emit_generation_diagnostic(
                    generation_diagnostic_callback,
                    requirement_id=requirement_id,
                    requirement=requirement,
                    marker=marker,
                    attempt=current_attempt,
                    max_attempts=max_attempts,
                    validation=validation,
                    raw=raw,
                    fallback_used=current_attempt == max_attempts,
                )
                return (
                    ValidationIssue(
                        code=(validation.failure_type or "MECHANICAL_FAILURE").upper(),
                        stage="segmentation",
                        path="/marked_text",
                        observed={
                            "marked_text": marked_text,
                            "first_difference_offset": validation.first_difference_offset,
                        },
                        expected={"exact_source_reconstruction": True},
                        context=previous_error,
                        repair_class=RepairClass.CONSTRAINED,
                        allowed_paths=("/marked_text",),
                        message=validation.instruction,
                    ),
                )
            return ()

        def request_patch(
            artifact: object,
            issues: object,
            repair_class: RepairClass,
        ) -> dict[str, object]:
            del issues, repair_class
            repaired = call_model()
            value = repaired.get("marked_text") if isinstance(repaired, Mapping) else None
            op = (
                "replace"
                if isinstance(artifact, Mapping) and "marked_text" in artifact
                else "add"
            )
            return {
                "operations": [PatchOperation(op, "/marked_text", value).to_dict()]
            }

        result = ValidatedGenerationRunner(
            RepairPolicy(
                max_domain_repairs=max_retries,
                max_constrained_repairs=max_retries,
                max_semantic_patches=0,
                immutable_paths=(),
            )
        ).generate(
            proposal_fn=call_model,
            validate_fn=validate_proposal,
            patch_fn=request_patch,
            prompt_version=prompt_version,
        )
        self.last_generation_diagnostic = result.diagnostic()
        if result.succeeded:
            marked_text = result.artifact.get("marked_text")
            return build_surface_segmentation(
                requirement_id=requirement_id,
                source_text=requirement,
                marked_text=marked_text,
                marker=marker,
                status="review_pending",
                generator=generator,
                experiment_mode=experiment_mode,
                data_split=data_split,
                label_source="rules",
                review_basis="mechanical_validation",
            )
        # Boundary proposals are non-semantic.  When an LLM cannot preserve the
        # source exactly, use the backend-only proposal and keep the configured
        # human/Judge surface review instead of blocking the whole requirement.
        fallback_marked_text, marker = deterministic_marked_text(requirement, marker)
        return build_surface_segmentation(
            requirement_id=requirement_id,
            source_text=requirement,
            marked_text=fallback_marked_text,
            marker=marker,
            status="review_pending",
            generator=generator,
            experiment_mode=experiment_mode,
            data_split=data_split,
            label_source="rules",
            review_basis="mechanical_validation_after_deterministic_fallback",
        )


def approve_rule_segmentation(segmentation: SurfaceSegmentation) -> SurfaceSegmentation:
    """Mark deterministic rules output as reviewed without claiming human review."""

    return replace(
        segmentation,
        status="surface_reviewed",
        label_source="rules",
        review_basis="mechanical_and_rules",
        persistent_rag_allowed=False,
    )


def _segmentation_system_prompt(marker: str) -> str:
    return (
        "你的任务只是在原始文本中插入切分标记。"
        "不得删除、增加、改写、翻译、纠正或重新排列任何原始字符。"
        f"本次唯一允许插入的标记是 {marker!r}。"
        "删除全部切分标记后，输出必须与原文逐字符完全一致。"
        "只输出包含 marked_text 的 JSON 对象，不要输出 IR、LTLf、解释、标题或编号。"
    )


def _invalid_output_shape_validation(
    raw: object,
) -> MechanicalSegmentationValidation:
    keys = sorted(str(key) for key in raw) if isinstance(raw, Mapping) else []
    detail = f"response_type={type(raw).__name__}"
    if keys:
        detail += "; keys=" + ",".join(keys[:50])
    return MechanicalSegmentationValidation(
        status="failed",
        exact_reconstruction=False,
        failure_type="invalid_output_shape",
        missing_or_changed_excerpt=detail,
        instruction="只输出包含字符串 marked_text 的 JSON 对象。",
    )


def _emit_generation_diagnostic(
    callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    requirement_id: str,
    requirement: str,
    marker: str,
    attempt: int,
    max_attempts: int,
    validation: object,
    raw: object,
    fallback_used: bool,
) -> None:
    if callback is None:
        return
    callback(
        {
            "requirement_id": requirement_id,
            "source_hash": source_hash(requirement),
            "attempt": attempt,
            "max_attempts": max_attempts,
            "error_type": getattr(validation, "failure_type", ""),
            "first_difference_offset": getattr(
                validation, "first_difference_offset", None
            ),
            "missing_or_changed_excerpt": _bounded_text(
                getattr(validation, "missing_or_changed_excerpt", "")
            ),
            "instruction": _bounded_text(getattr(validation, "instruction", "")),
            "will_retry": attempt < max_attempts,
            "fallback_used": fallback_used,
            "marker": marker,
            "rejected_response": _sanitized_rejected_response(raw),
            "contract_version": CONTRACT_VERSION,
            "canonicalizer_version": CANONICALIZER_VERSION,
            "validator_version": VALIDATOR_VERSION,
            "repair_policy_version": REPAIR_POLICY_VERSION,
            "artifact_hash": artifact_hash(raw),
            "issue_fingerprint": __import__("hashlib").sha256(
                canonical_json(
                    {
                        "code": getattr(validation, "failure_type", ""),
                        "observed": getattr(validation, "missing_or_changed_excerpt", ""),
                    }
                ).encode("utf-8")
            ).hexdigest(),
            "repair_strategy": RepairClass.CONSTRAINED.value,
            "reason_code": "VALIDATION_FAILED",
        }
    )


def _sanitized_rejected_response(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"response_type": type(raw).__name__}
    marked_text = raw.get("marked_text")
    return {
        "response_type": type(raw).__name__,
        "response_keys": sorted(str(key) for key in raw)[:50],
        "marked_text": (
            _bounded_text(marked_text) if marked_text is not None else None
        ),
        "marked_text_type": type(marked_text).__name__,
    }


def _bounded_text(value: object, *, limit: int = 4000) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"
