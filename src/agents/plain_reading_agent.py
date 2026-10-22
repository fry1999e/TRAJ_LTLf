"""Agent for formula-free ordinary-language ambiguity options."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import re
from typing import Any, Callable, Mapping

from agents.base_agent import BaseAgent, JSONLLMClient
from requirement_reading_contract import REQUIREMENT_READING_RULES
from plain_reading import (
    PlainReadingValidationError,
    build_plain_reading_review,
    plain_reading_from_mapping,
    validate_plain_reading_review,
    validate_plain_reading_lexical_issues,
)
from schemas import (
    ArtifactGenerator,
    NoAmbiguityConfirmation,
    PlainReadingReview,
    SurfaceSegmentation,
)
from generation_schemas import (
    PLAIN_READING_PROPOSAL_SCHEMA,
)
from plain_reading_repair import (
    compile_plain_reading_repair_plan,
    deterministic_axis_removal_plan,
    normalize_plain_reading_issues,
    parse_plain_reading_repair_plan,
    repair_plan_response_schema,
)
from ap_grounding import APGroundingContract
from ap_binding_context import text_only_ap_binding_context
from plain_reading_lexical import build_plain_reading_lexical_contract
from evidence_grounding import canonicalize_plain_reading_proposal
from validated_generation import (
    CANONICALIZER_VERSION,
    CandidateSubmission,
    CONTRACT_VERSION,
    REPAIR_POLICY_VERSION,
    VALIDATOR_VERSION,
    CanonicalizedArtifact,
    RepairClass,
    RepairPolicy,
    Severity,
    ValidatedGenerationRunner,
    ValidationIssue,
    artifact_hash,
    canonical_json,
    json_pointer,
)


PLAIN_READING_SEMANTIC_FALLBACK_POLICY_VERSION = "plain-reading-fallback-v1"


class PlainReadingGenerationExhaustedError(PlainReadingValidationError):
    """Bounded generation failed; exposes persistable attempts and errors."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        errors: list[str],
        diagnostics: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.errors = tuple(errors)
        self.diagnostics = dict(diagnostics or {})


class PlainReadingSemanticFallback(PlainReadingGenerationExhaustedError):
    """Two structurally usable artifacts failed only semantic validation."""

    def __init__(
        self,
        message: str,
        *,
        reason_codes: tuple[str, ...],
        attempts: int,
        errors: list[str],
        last_rejected_candidate: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
    ):
        super().__init__(
            message,
            attempts=attempts,
            errors=errors,
            diagnostics=diagnostics,
        )
        self.reason_codes = tuple(reason_codes)
        self.last_rejected_candidate = dict(last_rejected_candidate)


class PlainReadingAgent(BaseAgent):
    """Detect one-axis-at-a-time readings without generating IR or formulas."""

    def __init__(self, retriever, llm_client: JSONLLMClient | None = None):
        super().__init__(retriever=retriever, name="PlainReadingAgent", llm_client=llm_client)
        self.last_generation_diagnostic: dict[str, Any] = {}

    def generate(
        self,
        requirement: str,
        segmentation: SurfaceSegmentation,
        *,
        requirement_id: str = "REQ-001",
        mode: str = "off",
        prompt_version: str = "plain-reading-v5",
        data_split: str = "test",
        guided_context: object | None = None,
        round_index: int = 1,
        previous_review: PlainReadingReview | None = None,
        ambiguity_generation_request: PlainReadingReview | None = None,
        segmentation_comparison_context: Mapping[str, Any] | None = None,
        ap_grounding_contract: APGroundingContract | None = None,
        generation_diagnostic_callback: Callable[[Mapping[str, Any]], None]
        | None = None,
    ) -> PlainReadingReview:
        if mode not in {"off", "manual", "judge", "stage_guided"}:
            raise ValueError("unsupported plain reading mode: " + mode)
        if (
            ambiguity_generation_request is not None
            and ambiguity_generation_request.overall_status
            != "user_requested_ambiguity_generation"
        ):
            raise PlainReadingValidationError(
                "forced ambiguity generation requires a user-requested review"
            )
        if previous_review is not None and (
            previous_review.source_hash != segmentation.source_hash
            or previous_review.segmentation_id != segmentation.segmentation_id
        ):
            raise PlainReadingValidationError(
                "previous plain-reading review belongs to a different effective source"
            )
        if ambiguity_generation_request is not None and (
            ambiguity_generation_request.source_hash != segmentation.source_hash
            or ambiguity_generation_request.segmentation_id != segmentation.segmentation_id
        ):
            raise PlainReadingValidationError(
                "ambiguity generation request belongs to a different effective source"
            )
        generator = ArtifactGenerator(
            mode="api" if self.llm_enabled else "rules",
            model="",
            prompt_version=prompt_version,
        )
        if ap_grounding_contract is None:
            ap_grounding_contract = text_only_ap_binding_context(
                requirement,
                requirement_id=requirement_id,
            ).grounding_contract
        if ap_grounding_contract is None:
            raise PlainReadingValidationError(
                "Plain Reading requires an APGroundingContract"
            )
        ap_grounding_contract.validate_source(requirement)
        if (
            ap_grounding_contract.segmentation_id
            and ap_grounding_contract.segmentation_id
            != segmentation.segmentation_id
        ):
            raise PlainReadingValidationError(
                "AP grounding contract belongs to a different segmentation"
            )
        lexical_contract = build_plain_reading_lexical_contract(
            requirement,
            ap_grounding_contract,
        )
        if mode == "off":
            return build_plain_reading_review(
                requirement_id=requirement_id,
                segmentation=segmentation,
                ambiguities=(),
                overall_status="no_ambiguity",
                review_basis="plain_reading_disabled",
                reviewer_source="rules",
                generator=generator,
                experiment_mode="off",
                data_split=data_split,
            )
        if not self.llm_enabled:
            raise PlainReadingValidationError(
                "LLM-backed plain reading is required when review is enabled; "
                "use mode=off only for an explicit bypass"
            )

        payload: dict[str, Any] = {
            "task": "plain_reading_ambiguity_detection",
            "source_text": requirement,
            "surface_segmentation": {
                "segmentation_id": segmentation.segmentation_id,
                "segments": [
                    {
                        "segment_id": segment.segment_id,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                    }
                    for segment in segmentation.segments
                ],
            },
            "plain_reading_lexical_contract": (
                {
                    "grounding_fingerprint": lexical_contract.grounding_fingerprint,
                    "ap_identity_ids": list(lexical_contract.ap_identity_ids),
                    "ap_identifiers": list(lexical_contract.ap_identifiers),
                    "quantity_evidence": [
                        list(item) for item in lexical_contract.quantity_evidence
                    ],
                    "explicit_negation_terms": list(
                        lexical_contract.explicit_negation_terms
                    ),
                    "contract_version": lexical_contract.contract_version,
                    "fingerprint": lexical_contract.fingerprint,
                }
            ),
            "rules": [
                REQUIREMENT_READING_RULES,
                "Handle one ambiguity axis per item.",
                "Each item has exactly two complete and independently readable natural-language options.",
                "The options differ only on that axis.",
                (
                    "Only emit an ambiguity when the source supports two materially distinct "
                    "ordinary-language readings. If it does not, omit that item; if no items "
                    "remain, return no_ambiguity with an empty ambiguities list."
                ),
                (
                    "The two plain_interpretation values must remain different after "
                    "case-folding and collapsing whitespace. Different option ids, questions, "
                    "or difference_summary values do not make duplicate interpretations valid."
                ),
                "Do not output IR, operator tokens, LTLf, or any formula.",
                "Write questions and options for a non-expert; do not expose terms such as phase, terminal policy, off-phase policy, or temporal operator.",
                (
                    "Do not add or omit APs, participants, events, negation, numbers, timing, "
                    "or conditionality outside the target ambiguity axis. Candidate quantity "
                    "values that differ only on a genuine quantity axis must remain explicit "
                    "for grounding review."
                ),
                (
                    "Treat every explicit if/when/whenever guard as invariant unless the exact "
                    "source wording supports a genuine guard-scope ambiguity. Never extend a "
                    "guarded consequence to the opposite or complementary guard case."
                ),
                (
                    "Lexical preservation is enforced on each plain_interpretation. Questions "
                    "and difference summaries are explanatory metadata: they may restate the "
                    "same axis but must not introduce a commitment absent from both complete options."
                ),
                "Repeat every explicit source negation token verbatim in each option, even when resolving its scope.",
                "Every ambiguity cites source_segment_ids.",
                (
                    "Every ambiguity returns an exact evidence_quote and source_segment_ids. "
                    "The backend derives Unicode-code-point source_span and evidence_text; "
                    "do not calculate numeric offsets. If the quote repeats within the cited "
                    "segments, return a zero-based occurrence_index."
                ),
            ],
            "output_schema": {
                "overall_status": "no_ambiguity or pending_plain_review",
                "ambiguities": [
                    {
                        "ambiguity_id": "optional proposal id; backend derives the persisted id",
                        "source_segment_ids": ["seg-1"],
                        "evidence_quote": "exact source substring",
                        "occurrence_index": "required only for a repeated quote",
                        "question": "ordinary-language question",
                        "options": [
                            {
                                "option_id": "A",
                                "plain_interpretation": "complete sentence",
                                "difference_summary": "unique difference",
                            },
                            {
                                "option_id": "B",
                                "plain_interpretation": "complete sentence",
                                "difference_summary": "unique difference",
                            },
                        ],
                    }
                ],
            },
        }
        if segmentation_comparison_context:
            payload["segmentation_comparison_context"] = dict(
                segmentation_comparison_context
            )
            payload["rules"].append(
                "Use the blind-LLM versus human surface-boundary comparison only as an attention cue. A boundary difference is not itself a semantic decision; emit an axis only when the source wording supports two materially different ordinary-language readings."
            )
        forced_generation = ambiguity_generation_request is not None
        if forced_generation:
            trigger = ambiguity_generation_request.ambiguity_generation_trigger
            payload["task"] = "user_requested_plain_reading_ambiguity_generation"
            payload["user_disagrees_with_no_ambiguity"] = True
            payload["minimum_required_ambiguity_axes"] = 1
            payload["user_suspected_segment_ids"] = list(
                trigger.suspected_segment_ids
            )
            payload["user_comment"] = trigger.user_comment
            payload["previous_no_ambiguity_review"] = {
                "reading_review_id": trigger.previous_reading_review_id,
                "overall_status": "no_ambiguity",
            }
            payload["rules"].extend(
                (
                    "The user rejected the previous no-ambiguity judgment. Generate at least one source-supported candidate ambiguity axis.",
                    "An empty ambiguities list is invalid for this request.",
                    "Use suspected segments and the user comment only as attention guidance; never invent source content.",
                    "The generated axes remain non-authoritative until the user reviews them.",
                )
            )
            payload["output_schema"]["ambiguities_min_items"] = 1
        if previous_review is not None:
            payload["regeneration_feedback"] = {
                "instruction": (
                    "The reviewer rejected the previous option pairs. Generate materially "
                    "different, source-supported binary readings and do not repeat them."
                ),
                "previous_reading_review_id": previous_review.reading_review_id,
                "previous_axes": [
                    {
                        "ambiguity_id": ambiguity.ambiguity_id,
                        "question": ambiguity.question,
                        "options": [
                            option.plain_interpretation for option in ambiguity.options
                        ],
                        "decision_status": ambiguity.decision_status,
                        "comments": ambiguity.comments,
                    }
                    for ambiguity in previous_review.ambiguities
                ],
            }
        if guided_context is not None and hasattr(guided_context, "prompt_payload"):
            payload["guided_learning"] = guided_context.prompt_payload(
                "plain_reading", round_index=round_index
            )
        repair_policy = RepairPolicy(
            # runtime fixes artifact work to one proposal plus one bounded patch.
            max_domain_repairs=1,
            max_constrained_repairs=1,
            max_semantic_patches=1,
            immutable_paths=(
                "/source_hash",
                "/contract_version",
                "/canonicalizer_version",
            ),
        )
        max_generation_attempts = 1 + repair_policy.max_domain_repairs
        generation_attempt = 0
        legacy_model_contract = False
        attempt_diagnostics: list[dict[str, Any]] = []
        validation_state: dict[str, Any] = {
            "parsed": None,
            "error_type": "",
            "errors": [],
            "issues": (),
            "rejected_response": {},
            "grounding_review_history": [],
        }

        def generate_proposal() -> dict[str, Any]:
            nonlocal generation_attempt, legacy_model_contract
            generation_attempt += 1
            raw = self.complete_json(
                "Generate formula-free ordinary-language reading alternatives only.",
                payload,
                temperature=0.0,
                response_schema=PLAIN_READING_PROPOSAL_SCHEMA,
            )
            if isinstance(raw, Mapping) and isinstance(raw.get("ambiguities"), list):
                legacy_model_contract = legacy_model_contract or any(
                    isinstance(axis, Mapping)
                    and "evidence_quote" not in axis
                    and ("source_span" in axis or "evidence_text" in axis)
                    for axis in raw["ambiguities"]
                )
            return _plain_model_fields(raw)

        def canonicalize_proposal(raw: Any) -> CanonicalizedArtifact:
            result = canonicalize_plain_reading_proposal(
                raw,
                source_text=requirement,
                segments=segmentation.segments,
                source_hash=segmentation.source_hash,
            )
            # Existing proposal ids remain stable during backend contract
            # migration; new ids are derived when the model omitted one.
            if isinstance(raw, Mapping) and isinstance(result.artifact, Mapping):
                raw_axes = raw.get("ambiguities")
                canonical_axes = result.artifact.get("ambiguities")
                if isinstance(raw_axes, list) and isinstance(canonical_axes, list):
                    for index, axis in enumerate(canonical_axes):
                        if index >= len(raw_axes) or not isinstance(axis, dict) or not isinstance(raw_axes[index], Mapping):
                            continue
                        proposed_id = str(raw_axes[index].get("ambiguity_id", ""))
                        if proposed_id:
                            axis["ambiguity_id"] = proposed_id
            return result

        def validate_proposal(raw: Any) -> tuple[ValidationIssue, ...]:
            rejected_response = _sanitized_rejected_response(raw)
            validation_state["rejected_response"] = rejected_response
            try:
                parsed = _plain_review_from_model_output(
                    raw,
                    segmentation=segmentation,
                    requirement_id=requirement_id,
                    generator=generator,
                    mode=mode,
                    data_split=data_split,
                    forced_generation=forced_generation,
                    ambiguity_generation_request=ambiguity_generation_request,
                    validate_contract=False,
                )
            except (PlainReadingValidationError, TypeError, ValueError) as exc:
                contract_errors = list(getattr(exc, "errors", (str(exc),)))
                validation_state.update(
                    parsed=None,
                    error_type="contract_validation",
                    errors=contract_errors,
                )
                _emit_generation_diagnostic(
                    attempt_diagnostics.append,
                    requirement_id=requirement_id,
                    segmentation=segmentation,
                    attempt=generation_attempt,
                    max_attempts=max_generation_attempts,
                    error_type="contract_validation",
                    errors=contract_errors,
                    rejected_response=rejected_response,
                    repair_strategy=RepairClass.CONSTRAINED,
                )
                issues = tuple(
                    _plain_reading_contract_issue(
                        error,
                        rejected_response=rejected_response,
                    )
                    for error in contract_errors
                )
                validation_state["issues"] = issues
                return issues

            contract_errors = list(
                validate_plain_reading_review(
                    parsed,
                    segmentation,
                    raise_on_error=False,
                )
            )
            contract_issues = tuple(
                _plain_reading_contract_issue(
                    error,
                    rejected_response=rejected_response,
                )
                for error in contract_errors
            )
            lexical_issues = tuple(
                _plain_reading_generation_lexical_issue(issue)
                for issue in validate_plain_reading_lexical_issues(
                    requirement,
                    parsed,
                    lexical_contract=lexical_contract,
                )
            )
            structural_grounding_issues = _axis_structural_grounding_issues(
                requirement=requirement,
                review=parsed,
                rejected_response=rejected_response,
            )
            forced_empty_issues = (
                (_forced_generation_empty_axis_issue(rejected_response),)
                if forced_generation and not parsed.ambiguities
                else ()
            )
            grounding_issues: tuple[ValidationIssue, ...] = (
                *structural_grounding_issues,
                *forced_empty_issues,
            )
            structurally_rejected_axis_ids = {
                str(issue.context.get("ambiguity_id", ""))
                for issue in structural_grounding_issues
            }
            grounding_history = validation_state["grounding_review_history"]
            if structural_grounding_issues:
                grounding_history.append(
                    {
                        "attempt": generation_attempt,
                        "reviewer": "mechanical_conditionality_guard",
                        "candidate_axis_ids": sorted(structurally_rejected_axis_ids),
                        "result": "rejected",
                        "issues": [
                            issue.to_dict() for issue in structural_grounding_issues
                        ],
                    }
                )
            if _plain_reading_grounding_reviewable(parsed, segmentation):
                grounding_issues = (
                    *grounding_issues,
                    *_axis_grounding_review_issues(
                        self,
                        requirement=requirement,
                        segmentation=segmentation,
                        review=parsed,
                        rejected_response=rejected_response,
                        skip_axis_ids=structurally_rejected_axis_ids,
                        attempt=generation_attempt,
                        audit_history=grounding_history,
                    ),
                )
            issues = (*contract_issues, *lexical_issues, *grounding_issues)
            if issues:
                error_type = _plain_reading_error_type(issues)
                errors = [issue.message or issue.code for issue in issues]
                repair_strategy = _plain_reading_next_repair_class(issues)
                validation_state.update(
                    parsed=parsed,
                    error_type=error_type,
                    errors=errors,
                    issues=issues,
                )
                _emit_generation_diagnostic(
                    attempt_diagnostics.append,
                    requirement_id=requirement_id,
                    segmentation=segmentation,
                    attempt=generation_attempt,
                    max_attempts=max_generation_attempts,
                    error_type=error_type,
                    errors=errors,
                    rejected_response=rejected_response,
                    repair_strategy=repair_strategy,
                    grounding_review_records=tuple(
                        record
                        for record in grounding_history
                        if record.get("attempt") == generation_attempt
                    ),
                )
                return tuple(issues)
            validation_state.update(parsed=parsed, error_type="", errors=[], issues=())
            return ()

        def patch_proposal(
            current: Any,
            issues: tuple[ValidationIssue, ...],
            repair_class: RepairClass,
        ) -> dict[str, Any]:
            nonlocal generation_attempt
            rejected_response = validation_state["rejected_response"]
            # Plain-reading has exactly one domain repair.  When validation found
            # both a constrained lexical issue and a semantic grounding issue,
            # the runner selects the constrained class first.  Give that single
            # atomic patch the complete current issue set so semantic feedback is
            # not starved behind a repair class that can never run in this budget.
            current_issues = tuple(validation_state.get("issues", ()))
            if current_issues and repair_class in {
                RepairClass.CONSTRAINED,
                RepairClass.SEMANTIC_PATCH,
            }:
                issues = current_issues
            payload.pop("contract_validation_feedback", None)
            payload.pop("lexical_preservation_feedback", None)
            payload.pop("grounding_review_feedback", None)
            contract_issues = tuple(
                issue for issue in issues if _is_plain_reading_contract_issue(issue)
            )
            lexical_issues = tuple(
                issue for issue in issues if _is_plain_reading_lexical_issue(issue)
            )
            grounding_issues = tuple(
                issue
                for issue in issues
                if issue not in contract_issues and issue not in lexical_issues
            )
            if contract_issues:
                payload["contract_validation_feedback"] = {
                    "attempt": generation_attempt,
                    "errors": [issue.message or issue.code for issue in contract_issues],
                    "source_span_contract": _source_span_contract_feedback(segmentation),
                    "instruction": (
                        "Return patch-equivalent corrected fields that satisfy the output contract. "
                        "For duplicate options, re-evaluate the complete ambiguity axis: its "
                        "question, both complete interpretations, and both difference summaries "
                        "must describe the same single distinction and remain mutually consistent. "
                        "A paraphrase that only changes wording is not a materially distinct reading. "
                        + (
                            "Because the user rejected no_ambiguity, return at least one valid ambiguity axis; ambiguities=[] is not allowed."
                            if forced_generation
                            else "If no supported ambiguity remains, return no_ambiguity with ambiguities=[]; the result will await human confirmation."
                        )
                        + " Do not rely on ids, summaries, or unsupported timing or conditions to distinguish duplicate readings."
                    ),
                    "rejected_response": rejected_response,
                }
            if lexical_issues:
                payload["lexical_preservation_feedback"] = {
                    "attempt": generation_attempt,
                    "errors": [issue.message or issue.code for issue in lexical_issues],
                    "issue_codes": sorted({issue.code for issue in lexical_issues}),
                    "instruction": _lexical_repair_instruction(
                        lexical_issues,
                        forced_generation=forced_generation,
                    ),
                    "rejected_ambiguities": current.get("ambiguities", []),
                    "rejected_response": rejected_response,
                }
            if grounding_issues:
                no_supported_axis = any(
                    issue.code == "NO_SOURCE_SUPPORTED_AXIS_DESPITE_USER_CONCERN"
                    for issue in grounding_issues
                )
                payload["grounding_review_feedback"] = {
                    "attempt": generation_attempt,
                    "errors": [issue.message or issue.code for issue in grounding_issues],
                    "instruction": (
                        (
                            "The forced proposal returned no candidate axis. Make one bounded "
                            "attempt to provide at least one materially distinct, source-grounded "
                            "axis; do not turn the user comment into source evidence. "
                            if no_supported_axis
                            else
                            "Repair only the rejected candidate axes so they are source-grounded, "
                            "materially distinct, free of invented content, and understandable "
                            "to a non-expert. If a rejected axis has no materially distinct pair "
                            "that is supported by the source, remove that entire axis instead of "
                            "duplicating or merely paraphrasing its options. "
                        )
                        + (
                            "The user requires at least one axis, so do not remove the final axis; "
                            "never invent a replacement reading."
                            if forced_generation
                            else "If removing it leaves no axes, return no_ambiguity with ambiguities=[]."
                        )
                    ),
                    "rejected_response": rejected_response,
                }
            tasks = normalize_plain_reading_issues(
                current,
                issues,
                forced_generation=forced_generation,
            )
            if not tasks:
                raise PlainReadingValidationError(
                    "plain-reading findings have no authorized stable-ID repair task"
                )
            patch_payload = {
                **payload,
                "task": "plain_reading_typed_repair_plan",
                "current_artifact": current,
                "issues": [issue.to_dict() for issue in issues],
                "repair_tasks": [asdict(task) for task in tasks],
                "legal_minimal_example": {
                    "artifact_hash": tasks[0].artifact_hash,
                    "actions": [{
                        "action": tasks[0].allowed_actions[0],
                        "ambiguity_id": tasks[0].ambiguity_id,
                        **(
                            {"option_id": tasks[0].option_id, "plain_interpretation": "complete source-grounded reading"}
                            if tasks[0].allowed_actions[0] == "replace_option_text"
                            else {}
                        ),
                    }],
                },
                "repair_instruction": (
                    "Return a PlainReadingRepairPlan with artifact_hash and "
                    "actions only. 'Stable-ID' means reuse the provided "
                    "ambiguity_id and option_id fields; never add a stable_id field. "
                    "Never return repair_tasks, value, JSON Pointer, "
                    "array index, patch op, overall_status, source/hash/evidence, "
                    "or any target absent from repair_tasks."
                ),
            }
            generation_attempt += 1
            deterministic_plan = deterministic_axis_removal_plan(
                current,
                tasks,
                allowed_issue_codes=frozenset(
                    {
                        "AXIS_WIDENS_EXPLICIT_CONDITION",
                        "PARAPHRASE_OR_NON_DISTINCT_AXIS",
                    }
                ),
            )
            plan = deterministic_plan
            if plan is None:
                raw_plan = self.complete_json(
                    (
                        "Propose only stable-ID Plain Reading repair intent. "
                        "Do not return JSON Patch."
                    ),
                    patch_payload,
                    temperature=0.0,
                    response_schema=repair_plan_response_schema(tasks),
                )
                plan = parse_plain_reading_repair_plan(raw_plan, tasks)
            compilation = compile_plain_reading_repair_plan(
                current,
                tasks,
                plan,
                forced_generation=forced_generation,
            )
            return CandidateSubmission(
                candidate=compilation.candidate,
                changed_targets=compilation.changed_targets,
                contract_fingerprint=compilation.plan_fingerprint,
            )

        result = ValidatedGenerationRunner(repair_policy).generate(
            proposal_fn=generate_proposal,
            canonicalize_fn=canonicalize_proposal,
            validate_fn=validate_proposal,
            patch_fn=patch_proposal,
            prompt_version=prompt_version,
        )
        _flush_generation_diagnostics(
            attempt_diagnostics,
            result=result,
            generation_attempts=generation_attempt,
            callback=generation_diagnostic_callback,
        )
        result_diagnostic = result.diagnostic()
        result_diagnostic["generation_attempts"] = generation_attempt
        result_diagnostic["accepted_current_artifact"] = {
            "artifact_hash": result.artifact_hash,
            "issue_fingerprint": result.issue_fingerprint,
            "issues": [issue.to_dict() for issue in result.issues],
        }
        if (
            not result.succeeded
            and result.attempts
            and attempt_diagnostics
        ):
            last_rejected = attempt_diagnostics[-1]
            result_diagnostic["last_rejected_candidate"] = {
                "artifact_hash": str(last_rejected.get("artifact_hash", "")),
                "sanitized_response_hash": str(
                    last_rejected.get("artifact_hash", "")
                ),
                "canonical_artifact_hash": str(
                    result.rejected_candidate.get(
                        "artifact_hash", result.artifact_hash
                    )
                ),
                "hash_scope": "sanitized_rejected_response",
                "issue_fingerprint": str(
                    last_rejected.get("issue_fingerprint", "")
                ),
                "error_type": str(last_rejected.get("error_type", "")),
                "errors": list(last_rejected.get("errors", ())),
                "rejected_response": dict(
                    last_rejected.get("rejected_response", {})
                ),
                "rollback_reason": result.reason_code,
            }
        result_diagnostic["validation_history"] = [
            {
                "attempt": item["attempt"],
                "error_type": item["error_type"],
                "errors": list(item["errors"]),
            }
            for item in attempt_diagnostics
        ]
        result_diagnostic["grounding_review_history"] = list(
            validation_state["grounding_review_history"]
        )
        self.last_generation_diagnostic = result_diagnostic
        parsed = validation_state.get("parsed")
        if not result.succeeded or not isinstance(parsed, PlainReadingReview):
            errors = [issue.message or issue.code for issue in result.issues]
            if not errors:
                errors = [result.reason_code]
            error_type = (
                _plain_reading_error_type(result.issues)
                if result.issues
                else "validation"
            )
            repair_error = next(
                (
                    str(attempt.repair_error)
                    for attempt in reversed(result.attempts)
                    if attempt.repair_error
                ),
                "",
            )
            history = " | ".join(
                f"attempt {item['attempt']} {item['error_type']}: "
                + "; ".join(str(error) for error in item["errors"])
                for item in attempt_diagnostics
            )
            terminal_issues = tuple(validation_state.get("issues", ()))
            if not terminal_issues:
                terminal_issues = tuple(result.issues)
            last_attempt_codes = tuple(
                str(item.get("code", ""))
                for item in (result.attempts[-1].issues if result.attempts else ())
                if isinstance(item, Mapping) and item.get("code")
            )
            fallback_codes = tuple(
                dict.fromkeys(
                    last_attempt_codes
                    or tuple(issue.code for issue in terminal_issues)
                )
            )
            if forced_generation:
                fallback_codes = tuple(
                    dict.fromkeys(
                        (
                            *fallback_codes,
                            "NO_SOURCE_SUPPORTED_AXIS_DESPITE_USER_CONCERN",
                        )
                    )
                )
            repair_protocol_failed = any(
                bool(attempt.repair_error) for attempt in result.attempts[1:]
            )
            if (
                generation_attempt == 2
                and isinstance(parsed, PlainReadingReview)
                and terminal_issues
                and _plain_reading_semantic_only_codes(fallback_codes)
                and not repair_protocol_failed
                and result.reason_code
                not in {
                    "REPAIR_REJECTED",
                    "INTERNAL_DETERMINISTIC_REPAIR_BUG",
                    "DETERMINISTIC_REPAIR_UNAVAILABLE",
                    "JSON_REPAIR_UNAVAILABLE",
                    "PATCH_REPAIR_UNAVAILABLE",
                }
            ):
                result_diagnostic.update(
                    semantic_fallback=True,
                    semantic_fallback_issue_codes=list(fallback_codes),
                    semantic_fallback_attempts=2,
                    semantic_fallback_reason=(
                        "no_source_supported_axis_after_repair"
                    ),
                    semantic_fallback_policy_version=(
                        PLAIN_READING_SEMANTIC_FALLBACK_POLICY_VERSION
                    ),
                )
                self.last_generation_diagnostic = result_diagnostic
                raise PlainReadingSemanticFallback(
                    "plain reading semantic validation produced no safe axis "
                    "after exactly two artifact attempts",
                    reason_codes=fallback_codes,
                    attempts=2,
                    errors=errors,
                    last_rejected_candidate=result_diagnostic.get(
                        "last_rejected_candidate", {}
                    ),
                    diagnostics=result_diagnostic,
                )
            raise PlainReadingGenerationExhaustedError(
                f"plain reading {error_type.replace('_', ' ')} failed after "
                f"{generation_attempt} attempts "
                f"({generation_attempt}/{max_generation_attempts} artifact attempts; "
                f"{result.reason_code}): current artifact errors: "
                + "; ".join(errors)
                + (f"; repair error: {repair_error}" if repair_error else "")
                + (f"; validation history: {history}" if history else ""),
                attempts=generation_attempt,
                errors=errors,
                diagnostics=result_diagnostic,
            )
        return build_plain_reading_review(
            requirement_id=requirement_id,
            segmentation=segmentation,
            ambiguities=_backend_plain_ambiguity_ids(
                parsed.ambiguities,
                segmentation=segmentation,
                preserve_proposed_ids=legacy_model_contract,
            ),
            overall_status=parsed.overall_status,
            review_basis="agent_detection",
            reviewer_source="",
            generator=generator,
            experiment_mode=mode,
            data_split=data_split,
            review_history=(
                *(
                    ambiguity_generation_request.review_history
                    if ambiguity_generation_request is not None
                    else ()
                ),
                {
                    "operation": "plain_reading_lexical_contract_bound",
                    "ap_grounding_fingerprint": (
                        ap_grounding_contract.fingerprint
                    ),
                    "plain_reading_lexical_fingerprint": (
                        lexical_contract.fingerprint
                    ),
                    "ap_grounding_authority": (
                        ap_grounding_contract.authority
                    ),
                },
            ),
            revision_requests=(
                ambiguity_generation_request.revision_requests
                if ambiguity_generation_request is not None
                else ()
            ),
            no_ambiguity_confirmation=parsed.no_ambiguity_confirmation,
            ambiguity_generation_trigger=parsed.ambiguity_generation_trigger,
            supersedes_reading_review_id=(
                ambiguity_generation_request.reading_review_id
                if ambiguity_generation_request is not None
                else ""
            ),
        )


def _plain_model_fields(raw: Any) -> dict[str, Any]:
    """Keep the semantic proposal envelope and discard provider-only metadata."""

    if not isinstance(raw, Mapping):
        return {"_invalid_output_type": type(raw).__name__, "ambiguities": None}
    return {
        "overall_status": raw.get("overall_status", ""),
        "ambiguities": deepcopy(raw.get("ambiguities", [])),
    }


def _backend_plain_ambiguity_ids(
    ambiguities: tuple,
    *,
    segmentation: SurfaceSegmentation,
    preserve_proposed_ids: bool = False,
) -> tuple:
    """Persist retry-stable ids derived from semantic evidence, never position."""

    result = []
    for ambiguity in ambiguities:
        if preserve_proposed_ids and ambiguity.ambiguity_id:
            result.append(ambiguity)
            continue
        digest = __import__("hashlib").sha256(
            (
                segmentation.source_hash
                + "\0"
                + "\0".join(ambiguity.source_segment_ids)
                + "\0"
                + repr(tuple(ambiguity.source_span))
                + "\0"
                + ambiguity.evidence_text
                + "\0"
                + " ".join(ambiguity.question.split()).casefold()
            ).encode("utf-8")
        ).hexdigest()[:12]
        result.append(replace(ambiguity, ambiguity_id=f"amb-{digest}"))
    return tuple(result)


def _plain_review_from_model_output(
    raw: Any,
    *,
    segmentation: SurfaceSegmentation,
    requirement_id: str,
    generator: ArtifactGenerator,
    mode: str,
    data_split: str,
    forced_generation: bool,
    ambiguity_generation_request: PlainReadingReview | None,
    validate_contract: bool = True,
) -> PlainReadingReview:
    if not isinstance(raw, Mapping):
        raise PlainReadingValidationError("PlainReadingAgent output must be an object")
    raw_ambiguities = raw.get("ambiguities", [])
    if not isinstance(raw_ambiguities, (list, tuple)):
        raise PlainReadingValidationError("ambiguities must be an array of objects")
    if forced_generation and raw_ambiguities:
        normalized_status = "pending_plain_review"
        confirmation_payload = asdict(
            ambiguity_generation_request.no_ambiguity_confirmation
        )
        trigger_payload = asdict(
            ambiguity_generation_request.ambiguity_generation_trigger
        )
    elif not raw_ambiguities:
        normalized_status = "pending_no_ambiguity_confirmation"
        confirmation_payload = asdict(
            NoAmbiguityConfirmation(
                status="pending",
                llm_proposed_no_ambiguity=True,
            )
        )
        trigger_payload = (
            asdict(ambiguity_generation_request.ambiguity_generation_trigger)
            if forced_generation
            else {}
        )
    else:
        normalized_status = "pending_plain_review"
        confirmation_payload = {}
        trigger_payload = {}
    normalized = {
        "schema_version": "5",
        "reading_review_id": "READ-PENDING",
        "requirement_id": requirement_id,
        "source_hash": segmentation.source_hash,
        "segmentation_id": segmentation.segmentation_id,
        "overall_status": normalized_status,
        "review_basis": "agent_detection",
        "reviewer_source": "",
        "generator": {
            "mode": generator.mode,
            "model": generator.model,
            "prompt_version": generator.prompt_version,
        },
        "experiment_mode": mode,
        "data_split": data_split,
        "ambiguities": raw_ambiguities,
        "no_ambiguity_confirmation": confirmation_payload,
        "ambiguity_generation_trigger": trigger_payload,
    }
    return plain_reading_from_mapping(
        normalized,
        segmentation if validate_contract else None,
    )


def _plain_reading_grounding_reviewable(
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
) -> bool:
    """Return whether every proposed axis is structurally safe to critique.

    Semantic review is intentionally allowed for contract-invalid duplicate
    readings.  It is skipped only when the reviewer could not reliably address
    a complete, source-linked binary axis.
    """

    if not review.ambiguities:
        return False
    known_segments = {segment.segment_id for segment in segmentation.segments}
    for ambiguity in review.ambiguities:
        if not ambiguity.ambiguity_id or not ambiguity.question.strip():
            return False
        if len(ambiguity.options) != 2:
            return False
        if any(
            not option.option_id
            or not option.plain_interpretation.strip()
            or not option.difference_summary.strip()
            for option in ambiguity.options
        ):
            return False
        if not ambiguity.source_segment_ids or any(
            segment_id not in known_segments
            for segment_id in ambiguity.source_segment_ids
        ):
            return False
        start, end = ambiguity.source_span
        if not (0 <= start < end <= len(segmentation.source_text)):
            return False
        if segmentation.source_text[start:end] != ambiguity.evidence_text:
            return False
    return True


def _forced_generation_empty_axis_issue(
    rejected_response: Mapping[str, Any],
) -> ValidationIssue:
    """Treat a structurally valid forced empty result as a semantic outcome."""

    return ValidationIssue(
        code="NO_SOURCE_SUPPORTED_AXIS_DESPITE_USER_CONCERN",
        stage="plain_reading",
        path="/ambiguities",
        observed=rejected_response.get("ambiguities", []),
        expected={
            "minimum_source_supported_axes": 1,
            "invented_content_allowed": False,
        },
        context={
            "ambiguity_id": "",
            "validation_visibility": "forced_generation_empty_result",
            "violation_count": 1,
        },
        repair_class=RepairClass.SEMANTIC_PATCH,
        allowed_paths=("/ambiguities", "/overall_status"),
        message=(
            "forced ambiguity generation returned no source-supported candidate axis"
        ),
        severity=Severity.ERROR,
    )


_AP_ATOM = r"(?:prop_\d+|[a-z])"
_CONDITIONAL_RELATION_RE = re.compile(
    rf"\b(?:if|when|whenever)\b(?P<guard>[^,.;]{{1,160}})"
    rf"(?:,|\bthen\b)(?P<body>[^.;]{{1,240}})",
    re.IGNORECASE,
)
_CONDITIONAL_GUARD_RE = re.compile(
    r"\b(?:if|when|whenever)\b(?P<guard>[^,.;]{1,160})",
    re.IGNORECASE,
)
_AP_AFTER_NEGATION_RE = re.compile(
    rf"\b(?:not|never|without)\s+(?P<ap>{_AP_ATOM})\b",
    re.IGNORECASE,
)
_AP_BEFORE_NEGATION_RE = re.compile(
    rf"\b(?P<ap>{_AP_ATOM})\b\s+"
    r"(?:(?:does|do|did|is|are|was|were|will|would|can|could|must|shall|should|has|have|had)\s+)?"
    r"(?:not|never|false|absent|disabled|inactive|cannot|can't|doesn't|isn't|aren't)\b",
    re.IGNORECASE,
)


def _ap_atoms(value: str) -> set[str]:
    return set(re.findall(rf"(?<!['’])\b{_AP_ATOM}\b", value.casefold()))


def _negative_ap_atoms(value: str) -> set[str]:
    """Return APs explicitly placed under lexical negation in a short clause."""

    return {
        match.group("ap").casefold()
        for pattern in (_AP_AFTER_NEGATION_RE, _AP_BEFORE_NEGATION_RE)
        for match in pattern.finditer(value)
    }


def _negative_conditional_guard_atoms(value: str) -> set[str]:
    result: set[str] = set()
    for match in _CONDITIONAL_GUARD_RE.finditer(value):
        result.update(_negative_ap_atoms(match.group("guard")))
    return result


def _source_negative_conditional_relations(
    requirement: str,
) -> tuple[tuple[set[str], set[str]], ...]:
    """Extract conclusive ``positive guard -> negative AP`` source relations.

    This intentionally handles only explicit English condition clauses.  It is
    a fail-closed pre-screen for obvious widening, not a general NL parser.
    Unsupported shapes continue to the independent grounding reviewer.
    """

    relations: list[tuple[set[str], set[str]]] = []
    for match in _CONDITIONAL_RELATION_RE.finditer(requirement):
        guard_text = match.group("guard")
        positive_guards = _ap_atoms(guard_text) - _negative_ap_atoms(guard_text)
        negative_responses = _negative_ap_atoms(match.group("body"))
        if positive_guards and negative_responses:
            relations.append((positive_guards, negative_responses))
    return tuple(relations)


def _axis_structural_grounding_issues(
    *,
    requirement: str,
    review: PlainReadingReview,
    rejected_response: Mapping[str, Any],
) -> tuple[ValidationIssue, ...]:
    """Reject options that mechanically widen an explicit source condition."""

    relations = _source_negative_conditional_relations(requirement)
    rejected_axes = rejected_response.get("ambiguities", ())
    if not isinstance(rejected_axes, (list, tuple)):
        rejected_axes = ()
    rejected_by_id = {
        str(axis.get("ambiguity_id", "")): axis
        for axis in rejected_axes
        if isinstance(axis, Mapping)
    }
    issues: list[ValidationIssue] = []
    for axis_index, ambiguity in enumerate(review.ambiguities):
        if (
            len(ambiguity.options) == 2
            and _only_reorders_commutative_conditions(
                ambiguity.options[0].plain_interpretation,
                ambiguity.options[1].plain_interpretation,
            )
        ):
            axis_path = json_pointer("ambiguities", axis_index)
            message = (
                f"{ambiguity.ambiguity_id} only reorders commutative condition terms"
            )
            issues.append(
                ValidationIssue(
                    code="PARAPHRASE_OR_NON_DISTINCT_AXIS",
                    stage="plain_reading",
                    path=axis_path,
                    observed=rejected_by_id.get(
                        ambiguity.ambiguity_id, asdict(ambiguity)
                    ),
                    expected={
                        "materially_distinct": True,
                        "distinguishing_scenario": "A and B require different outcomes",
                    },
                    context={
                        "ambiguity_id": ambiguity.ambiguity_id,
                        "failed_checks": ["materially_distinct"],
                        "rationale": message,
                        "validation_visibility": "mechanical_commutative_order_guard",
                    },
                    repair_class=RepairClass.SEMANTIC_PATCH,
                    allowed_paths=(axis_path, "/ambiguities", "/overall_status"),
                    message=message,
                    severity=Severity.ERROR,
                )
            )
            continue
        violations: list[dict[str, Any]] = []
        for option in ambiguity.options:
            text = option.plain_interpretation
            option_atoms = _ap_atoms(text)
            option_negative_atoms = _negative_ap_atoms(text)
            negative_guard_atoms = _negative_conditional_guard_atoms(text)
            for positive_guards, negative_responses in relations:
                retained_negative_responses = sorted(
                    negative_responses & option_negative_atoms
                )
                if not retained_negative_responses:
                    continue
                omitted_guards = sorted(positive_guards - option_atoms)
                complementary_guards = sorted(positive_guards & negative_guard_atoms)
                if omitted_guards:
                    violations.append(
                        {
                            "option_id": option.option_id,
                            "kind": "condition_guard_removed",
                            "guard_aps": omitted_guards,
                            "negative_response_aps": retained_negative_responses,
                        }
                    )
                if complementary_guards:
                    violations.append(
                        {
                            "option_id": option.option_id,
                            "kind": "complementary_guard_added",
                            "guard_aps": complementary_guards,
                            "negative_response_aps": retained_negative_responses,
                        }
                    )
        if not violations:
            continue
        summaries = []
        for violation in violations:
            action = (
                "removes source guard"
                if violation["kind"] == "condition_guard_removed"
                else "adds the complementary negated guard"
            )
            summaries.append(
                f"option {violation['option_id']} {action} "
                + ", ".join(violation["guard_aps"])
                + " while retaining negative consequence "
                + ", ".join(violation["negative_response_aps"])
            )
        message = (
            f"{ambiguity.ambiguity_id} widens an explicit source condition: "
            + "; ".join(summaries)
        )
        axis_path = json_pointer("ambiguities", axis_index)
        issues.append(
            ValidationIssue(
                code="AXIS_WIDENS_EXPLICIT_CONDITION",
                stage="plain_reading",
                path=axis_path,
                observed=rejected_by_id.get(ambiguity.ambiguity_id, asdict(ambiguity)),
                expected={
                    "source_grounded": True,
                    "no_invented_content": True,
                    "preserve_explicit_condition_guards": True,
                },
                context={
                    "ambiguity_id": ambiguity.ambiguity_id,
                    "failed_checks": ["source_grounded", "no_invented_content"],
                    "rationale": message,
                    "violations": violations,
                    "violation_count": 2,
                    "validation_visibility": "mechanical_relation_guard",
                },
                repair_class=RepairClass.SEMANTIC_PATCH,
                allowed_paths=(axis_path, "/ambiguities", "/overall_status"),
                message=message,
                severity=Severity.ERROR,
            )
        )
    return tuple(issues)


def _only_reorders_commutative_conditions(first: str, second: str) -> bool:
    """Detect A∧B/B∧A pseudo-axes without consulting formulas or an LLM."""

    conjunction = re.compile(r"\b(?:and|as well as)\b|&&|∧|以及|并且|且", re.I)
    if conjunction.search(first) is None or conjunction.search(second) is None:
        return False
    token_pattern = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
    first_tokens = [token.casefold() for token in token_pattern.findall(first)]
    second_tokens = [token.casefold() for token in token_pattern.findall(second)]
    return (
        first_tokens != second_tokens
        and sorted(first_tokens) == sorted(second_tokens)
    )


def _axis_grounding_review_issues(
    agent: PlainReadingAgent,
    *,
    requirement: str,
    segmentation: SurfaceSegmentation,
    review: PlainReadingReview,
    rejected_response: Mapping[str, Any],
    skip_axis_ids: set[str] | None = None,
    attempt: int = 0,
    audit_history: list[dict[str, Any]] | None = None,
) -> tuple[ValidationIssue, ...]:
    """Use an isolated LLM call to critique candidate axes, never to select one."""

    skipped = set(skip_axis_ids or ())
    review_axes = tuple(
        ambiguity
        for ambiguity in review.ambiguities
        if ambiguity.ambiguity_id not in skipped
    )
    if not review_axes:
        return ()

    raw = agent.complete_json(
        (
            "Act only as an independent ordinary-language ambiguity-axis reviewer. "
            "Do not choose an interpretation and do not output IR or formulas."
        ),
        {
            "task": "review_user_requested_ambiguity_axes",
            "source_text": requirement,
            "surface_segments": [
                {
                    "segment_id": segment.segment_id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                for segment in segmentation.segments
            ],
            "candidate_axes": [
                {
                    "ambiguity_id": ambiguity.ambiguity_id,
                    "source_segment_ids": list(ambiguity.source_segment_ids),
                    "source_span": list(ambiguity.source_span),
                    "evidence_text": ambiguity.evidence_text,
                    "question": ambiguity.question,
                    "options": [
                        {
                            "option_id": option.option_id,
                            "plain_interpretation": option.plain_interpretation,
                            "difference_summary": option.difference_summary,
                        }
                        for option in ambiguity.options
                    ],
                }
                for ambiguity in review_axes
            ],
            "rules": [
                "Review every candidate axis independently.",
                "source_grounded means both readings are plausible interpretations of the exact source evidence.",
                "materially_distinct means the A/B difference changes meaning, not merely wording.",
                "distinguishing_scenario must give one concrete source-supported state/event sequence where A requires an outcome and B does not, or requires a different outcome.",
                "no_invented_content means no new AP, number, participant, event, condition, or unsupported requirement.",
                "For every explicit if/when/whenever clause, test each option once with the guard true and once with it false; reject an option that constrains the consequence where the source is silent.",
                "The rationale must identify the exact source words that support the A/B difference; if no such words exist, mark source_grounded false.",
                "non_expert_language means the question and options avoid formal-logic jargon.",
                "Return no interpretation choice.",
            ],
            "output_schema": {
                "axis_reviews": [
                    {
                        "ambiguity_id": "existing axis id",
                        "source_grounded": True,
                        "materially_distinct": True,
                        "distinguishing_scenario": "concrete A/B behavior difference",
                        "no_invented_content": True,
                        "non_expert_language": True,
                        "rationale": "short source-based explanation",
                    }
                ]
            },
        },
        temperature=0.0,
    )
    if audit_history is not None:
        audit_history.append(
            {
                "attempt": attempt,
                "reviewer": "llm_independent_axis_reviewer",
                "candidate_axis_ids": [axis.ambiguity_id for axis in review_axes],
                "result": "received",
                "response": _sanitized_grounding_response(raw),
            }
        )
    if not isinstance(raw, Mapping):
        return (_grounding_protocol_issue("grounding reviewer output must be an object"),)
    items = raw.get("axis_reviews", ())
    if not isinstance(items, (list, tuple)):
        return (_grounding_protocol_issue("grounding reviewer axis_reviews must be an array"),)
    by_id = {
        str(item.get("ambiguity_id", "")): item
        for item in items
        if isinstance(item, Mapping)
    }
    issues: list[ValidationIssue] = []
    required_checks = (
        "source_grounded",
        "materially_distinct",
        "no_invented_content",
        "non_expert_language",
    )
    rejected_axes = rejected_response.get("ambiguities", ())
    if not isinstance(rejected_axes, (list, tuple)):
        rejected_axes = ()
    rejected_by_id = {
        str(axis.get("ambiguity_id", "")): axis
        for axis in rejected_axes
        if isinstance(axis, Mapping)
    }
    review_axis_ids = {axis.ambiguity_id for axis in review_axes}
    for index, ambiguity in enumerate(review.ambiguities):
        if ambiguity.ambiguity_id not in review_axis_ids:
            continue
        item = by_id.get(ambiguity.ambiguity_id)
        if item is None:
            failed_checks = ("missing_independent_review",)
            rationale = ""
        else:
            failed_checks = tuple(
                check for check in required_checks if item.get(check) is not True
            )
            distinguishing_scenario = _bounded_text(
                item.get("distinguishing_scenario", ""), limit=500
            )
            if not distinguishing_scenario:
                failed_checks = (*failed_checks, "distinguishing_scenario")
            rationale = _bounded_text(item.get("rationale", ""), limit=500)
        if not failed_checks:
            continue
        message = (
            f"{ambiguity.ambiguity_id} failed " + ", ".join(failed_checks)
            + (f": {rationale}" if rationale else "")
        )
        axis_path = json_pointer("ambiguities", index)
        issues.append(
            ValidationIssue(
                code="AXIS_NOT_SOURCE_GROUNDED",
                stage="plain_reading",
                path=axis_path,
                observed=rejected_by_id.get(
                    ambiguity.ambiguity_id,
                    asdict(ambiguity),
                ),
                expected={check: True for check in required_checks},
                context={
                    "ambiguity_id": ambiguity.ambiguity_id,
                    "failed_checks": list(failed_checks),
                    "rationale": rationale,
                    "distinguishing_scenario": (
                        distinguishing_scenario if item is not None else ""
                    ),
                    "violation_count": len(failed_checks),
                    "validation_visibility": "full_axis_review",
                },
                repair_class=RepairClass.SEMANTIC_PATCH,
                # Removing the final unsupported axis is a coherent two-field
                # transition.  The narrower axis path also records the actual
                # fault location for progress diagnostics.
                allowed_paths=(axis_path, "/ambiguities", "/overall_status"),
                message=message,
                severity=Severity.ERROR,
            )
        )
    unknown = set(by_id) - review_axis_ids
    if unknown:
        issues.append(
            _grounding_protocol_issue(
                "grounding reviewer returned unknown axes: "
                + ", ".join(sorted(unknown))
            )
        )
    return tuple(issues)


def _grounding_protocol_issue(message: str) -> ValidationIssue:
    return ValidationIssue(
        code="GROUNDING_REVIEW_PROTOCOL_INVALID",
        stage="plain_reading",
        path="/ambiguities",
        observed=None,
        expected={"axis_reviews": "one entry per candidate axis"},
        context={"protocol_error": message},
        repair_class=RepairClass.SEMANTIC_PATCH,
        allowed_paths=("/ambiguities", "/overall_status"),
        message=message,
    )


def _sanitized_grounding_response(raw: object) -> dict[str, Any]:
    """Persist bounded pass/fail reviewer evidence without provider metadata."""

    if not isinstance(raw, Mapping):
        return {"response_type": type(raw).__name__, "axis_reviews": []}
    items = raw.get("axis_reviews", ())
    if not isinstance(items, (list, tuple)):
        return {"response_type": "mapping", "axis_reviews": []}
    reviews: list[dict[str, Any]] = []
    for item in items[:20]:
        if not isinstance(item, Mapping):
            continue
        reviews.append(
            {
                "ambiguity_id": _bounded_text(item.get("ambiguity_id", "")),
                "source_grounded": item.get("source_grounded") is True,
                "materially_distinct": item.get("materially_distinct") is True,
                "distinguishing_scenario": _bounded_text(
                    item.get("distinguishing_scenario", ""), limit=500
                ),
                "no_invented_content": item.get("no_invented_content") is True,
                "non_expert_language": item.get("non_expert_language") is True,
                "rationale": _bounded_text(item.get("rationale", ""), limit=500),
            }
        )
    return {"axis_reviews": reviews}


def _plain_reading_generation_lexical_issue(
    issue: ValidationIssue,
) -> ValidationIssue:
    axis_index = int(issue.path.split("/")[2])
    return ValidationIssue(
        code=issue.code,
        stage=issue.stage,
        path=issue.path,
        observed=issue.observed,
        expected=issue.expected,
        context=issue.context,
        repair_class=issue.repair_class,
        allowed_paths=(
            issue.path,
            json_pointer("ambiguities", axis_index, "options"),
            "/ambiguities",
            "/overall_status",
        ),
        message=issue.message,
        severity=issue.severity,
    )


def _is_plain_reading_contract_issue(issue: ValidationIssue) -> bool:
    return issue.code in {
        "DUPLICATE_PLAIN_INTERPRETATIONS",
        "PLAIN_READING_CONTRACT_VIOLATION",
    }


def _is_plain_reading_lexical_issue(issue: ValidationIssue) -> bool:
    return issue.code.startswith("OPTION_")


_PLAIN_READING_SEMANTIC_FALLBACK_CODES = {
    "AXIS_NOT_SOURCE_GROUNDED",
    "AXIS_WIDENS_EXPLICIT_CONDITION",
    "PARAPHRASE_OR_NON_DISTINCT_AXIS",
    "DUPLICATE_PLAIN_INTERPRETATIONS",
    "UNKNOWN_CONDITION_SEMANTICS",
    "UNKNOWN_NEGATION_SCOPE",
    "UNKNOWN_TEMPORAL_SCOPE",
    "UNKNOWN_QUANTIFIER_SCOPE",
    "UNKNOWN_REFERENCE",
    "UNKNOWN_COORDINATION_SCOPE",
    "UNKNOWN_MODALITY",
    "NO_SOURCE_SUPPORTED_AXIS_DESPITE_USER_CONCERN",
}


def _plain_reading_semantic_only_issues(
    issues: tuple[ValidationIssue, ...],
) -> bool:
    """Return true only when every terminal issue has a usable artifact shape."""

    return bool(issues) and all(
        issue.code in _PLAIN_READING_SEMANTIC_FALLBACK_CODES
        or issue.code.startswith("OPTION_")
        for issue in issues
    )


def _plain_reading_semantic_only_codes(codes: tuple[str, ...]) -> bool:
    return bool(codes) and all(
        code in _PLAIN_READING_SEMANTIC_FALLBACK_CODES
        or code.startswith("OPTION_")
        for code in codes
    )


def _plain_reading_error_type(issues: tuple[ValidationIssue, ...]) -> str:
    if any(_is_plain_reading_contract_issue(issue) for issue in issues):
        return "contract_validation"
    if any(_is_plain_reading_lexical_issue(issue) for issue in issues):
        return "lexical_preservation"
    if any(
        issue.code
        in {
            "AXIS_NOT_SOURCE_GROUNDED",
            "AXIS_WIDENS_EXPLICIT_CONDITION",
            "PARAPHRASE_OR_NON_DISTINCT_AXIS",
            "NO_SOURCE_SUPPORTED_AXIS_DESPITE_USER_CONCERN",
            "GROUNDING_REVIEW_PROTOCOL_INVALID",
        }
        for issue in issues
    ):
        return (
            "mechanical_grounding_review"
            if any(
                issue.code
                in {
                    "AXIS_WIDENS_EXPLICIT_CONDITION",
                    "PARAPHRASE_OR_NON_DISTINCT_AXIS",
                }
                for issue in issues
            )
            else "llm_grounding_review"
        )
    return "validation"


def _plain_reading_next_repair_class(
    issues: tuple[ValidationIssue, ...],
) -> RepairClass:
    for repair_class in (
        RepairClass.DETERMINISTIC,
        RepairClass.CONSTRAINED,
        RepairClass.SEMANTIC_PATCH,
    ):
        if any(issue.repair_class == repair_class for issue in issues):
            return repair_class
    return RepairClass.HUMAN_REQUIRED


def _lexical_repair_instruction(
    issues: tuple[ValidationIssue, ...],
    *,
    forced_generation: bool,
) -> str:
    """Build code-specific repair guidance for finite lexical contracts."""

    codes = {issue.code for issue in issues}
    instructions = [
        "Repair only the reported plain_interpretation fields and keep both options complete and distinct."
    ]
    if codes & {"OPTION_OMITS_AP_TOKENS", "OPTION_OMITS_NUMBERS"}:
        instructions.append(
            "Restore every reported source AP or quantity, using an equivalent cardinal, ordinal, or digit form for quantities."
        )
    if "OPTION_OMITS_AP_TOKENS" in codes:
        instructions.append(
            "If an omitted AP is part of an explicit if/when/whenever guard, do not restore it by adding the opposite or complementary condition. If removing that guard is what makes the alternative different, remove the unsupported ambiguity axis instead."
        )
    if "OPTION_ADDS_AP_TOKENS" in codes:
        instructions.append(
            "Remove reported AP-like tokens that are absent from the immutable source; do not replace them with new participants or events."
        )
    if "OPTION_ADDS_NUMBERS" in codes:
        instructions.append(
            "Remove reported axis-independent quantities that are absent from the source. Values that distinguish a genuine quantity axis must be left explicit for grounding review, not hidden in metadata."
        )
    if "OPTION_OMITS_NEGATION" in codes:
        instructions.append(
            "Repeat each reported source negation token verbatim; do not replace it with a synonym such as absent or false."
        )
    instructions.append(
        (
            "The user requires at least one source-supported axis. Do not remove the final axis or invent a replacement reading."
            if forced_generation
            else "If no distinct source-supported pair survives the repair, remove that axis; if no axes remain, return no_ambiguity with ambiguities=[]."
        )
    )
    return " ".join(instructions)


def _source_span_contract_feedback(
    segmentation: SurfaceSegmentation,
) -> dict[str, Any]:
    """Describe valid evidence coordinates in a machine-actionable retry shape."""

    return {
        "offset_unit": "unicode_codepoint",
        "end_exclusive": True,
        "source_length": len(segmentation.source_text),
        "rule": (
            "source_span must be covered by exactly one contiguous run of cited "
            "segments; cite every segment intersected by the span and no unrelated "
            "padding segments"
        ),
        "segments": [
            {
                "segment_id": segment.segment_id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
            for segment in segmentation.segments
        ],
    }


def _plain_reading_contract_issue(
    error: str,
    *,
    rejected_response: Mapping[str, Any],
) -> ValidationIssue:
    """Map known contract failures to the narrowest stable repair location."""

    duplicate_suffix = " 的两个解释规范化后相同"
    ambiguities = rejected_response.get("ambiguities", ())
    if error.endswith(duplicate_suffix) and isinstance(ambiguities, (list, tuple)):
        ambiguity_id = error[: -len(duplicate_suffix)]
        for index, ambiguity in enumerate(ambiguities):
            if not isinstance(ambiguity, Mapping):
                continue
            if str(ambiguity.get("ambiguity_id", "")) != ambiguity_id:
                continue
            options = ambiguity.get("options", ())
            observed = [
                str(option.get("plain_interpretation", ""))
                for option in options
                if isinstance(option, Mapping)
            ] if isinstance(options, (list, tuple)) else options
            options_path = json_pointer("ambiguities", index, "options")
            return ValidationIssue(
                code="DUPLICATE_PLAIN_INTERPRETATIONS",
                stage="plain_reading",
                path=options_path,
                observed=observed,
                expected={
                    "option_count": 2,
                    "unique_after": ["casefold", "collapse_whitespace"],
                },
                context={"ambiguity_id": ambiguity_id},
                repair_class=RepairClass.CONSTRAINED,
                # The model may repair the pair or remove an unsupported axis.
                allowed_paths=(options_path, "/ambiguities", "/overall_status"),
                message=error,
            )
    return ValidationIssue(
        code="PLAIN_READING_CONTRACT_VIOLATION",
        stage="plain_reading",
        path="/ambiguities",
        observed=rejected_response.get("ambiguities", []),
        expected={"schema": "plain-reading-contract-v5"},
        context={"error": error},
        repair_class=RepairClass.CONSTRAINED,
        allowed_paths=("/ambiguities", "/overall_status"),
        message=error,
    )


def _flush_generation_diagnostics(
    diagnostics: list[dict[str, Any]],
    *,
    result: object,
    generation_attempts: int,
    callback: Callable[[Mapping[str, Any]], None] | None,
) -> None:
    """Finalize retry facts after orchestration decides whether work continued."""

    attempts = tuple(getattr(result, "attempts", ()))
    terminal_reason = str(getattr(result, "reason_code", ""))
    for diagnostic in diagnostics:
        attempt = int(diagnostic["attempt"])
        next_strategy = str(diagnostic.get("repair_strategy", ""))
        applied_strategy = (
            str(attempts[attempt - 1].repair_strategy)
            if 0 < attempt <= len(attempts)
            else ""
        )
        diagnostic["applied_repair_strategy"] = applied_strategy
        diagnostic["next_recommended_repair_strategy"] = next_strategy
        diagnostic["will_retry"] = attempt < generation_attempts
        diagnostic["terminal_reason_code"] = (
            terminal_reason if attempt == max(item["attempt"] for item in diagnostics) else ""
        )
        if callback is not None:
            callback(diagnostic)


def _emit_generation_diagnostic(
    callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    requirement_id: str,
    segmentation: SurfaceSegmentation,
    attempt: int,
    max_attempts: int,
    error_type: str,
    errors: list[str],
    rejected_response: Mapping[str, Any],
    repair_strategy: RepairClass,
    grounding_review_records: tuple[Mapping[str, Any], ...] = (),
) -> None:
    if callback is None:
        return
    diagnostic = {
            "stage": "plain_reading_generation",
            "requirement_id": requirement_id,
            "source_hash": segmentation.source_hash,
            "segmentation_id": segmentation.segmentation_id,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "error_type": error_type,
            "errors": list(errors),
            "will_retry": attempt < max_attempts,
            "rejected_response": dict(rejected_response),
            "contract_version": CONTRACT_VERSION,
            "canonicalizer_version": CANONICALIZER_VERSION,
            "validator_version": VALIDATOR_VERSION,
            "repair_policy_version": REPAIR_POLICY_VERSION,
            "artifact_hash": artifact_hash(rejected_response),
            "sanitized_response_hash": artifact_hash(rejected_response),
            "hash_scope": "sanitized_rejected_response",
            "issue_fingerprint": __import__("hashlib").sha256(
                canonical_json(sorted(errors)).encode("utf-8")
            ).hexdigest(),
            "repair_strategy": repair_strategy.value,
            "reason_code": "VALIDATION_FAILED",
        }
    if grounding_review_records:
        diagnostic["grounding_review_records"] = [
            dict(record) for record in grounding_review_records
        ]
    callback(diagnostic)


def _sanitized_rejected_response(raw: object) -> dict[str, Any]:
    """Keep only bounded plain-reading schema fields for retry/checkpoint diagnostics."""

    if not isinstance(raw, Mapping):
        return {
            "response_type": type(raw).__name__,
            "overall_status": "",
            "ambiguities": [],
        }
    result: dict[str, Any] = {
        "overall_status": _bounded_text(raw.get("overall_status", "")),
        "ambiguities": [],
    }
    ambiguities = raw.get("ambiguities", ())
    if not isinstance(ambiguities, (list, tuple)):
        return result
    sanitized_axes: list[dict[str, Any]] = []
    for ambiguity in ambiguities[:20]:
        if not isinstance(ambiguity, Mapping):
            continue
        source_segment_ids = ambiguity.get("source_segment_ids", ())
        if not isinstance(source_segment_ids, (list, tuple)):
            source_segment_ids = ()
        options = ambiguity.get("options", ())
        sanitized_options: list[dict[str, str]] = []
        if isinstance(options, (list, tuple)):
            for option in options[:4]:
                if not isinstance(option, Mapping):
                    continue
                sanitized_options.append(
                    {
                        "option_id": _bounded_text(option.get("option_id", "")),
                        "plain_interpretation": _bounded_text(
                            option.get("plain_interpretation", "")
                        ),
                        "difference_summary": _bounded_text(
                            option.get("difference_summary", "")
                        ),
                    }
                )
        sanitized_axes.append(
            {
                "ambiguity_id": _bounded_text(ambiguity.get("ambiguity_id", "")),
                "source_segment_ids": [
                    _bounded_text(item, limit=200) for item in source_segment_ids[:50]
                ],
                "source_span": list(ambiguity.get("source_span", ()))[:2],
                "evidence_text": _bounded_text(ambiguity.get("evidence_text", "")),
                "question": _bounded_text(ambiguity.get("question", "")),
                "options": sanitized_options,
            }
        )
    result["ambiguities"] = sanitized_axes
    return result


def _bounded_text(value: object, *, limit: int = 2000) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"
