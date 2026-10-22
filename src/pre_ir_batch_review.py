"""Deferred Markdown adjudication for large batch pre-IR processing.

The generator completes every item without interactive pauses.  Human review is
then applied through explicit Markdown action contracts for normal empty-axis,
semantic-fallback, override-confirmation and detected-axis states. Existing
SurfaceSegmentation and PlainReadingReview checkpoints remain the only artifacts
consumed by the downstream pipeline.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import uuid

from agents.base_agent import JSONLLMClient
from agents.plain_reading_agent import (
    PlainReadingAgent,
    PlainReadingSemanticFallback,
)
from agents.plain_reading_revision_agent import PlainReadingRevisionAgent
from agents.segmentation_agent import SegmentationAgent
from checkpointing import (
    load_human_semantic_override_ap_artifacts,
    load_human_semantic_override_checkpoint,
    load_ap_inventory_checkpoint,
    load_plain_reading_generated_checkpoint,
    load_plain_reading_need_view_checkpoint,
    load_plain_reading_reviewed_checkpoint,
    load_segmentation_reviewed_checkpoint,
    save_ap_identity_artifact_checkpoint,
    save_ap_inventory_checkpoint,
    save_human_semantic_override_checkpoint,
    save_plain_reading_generation_diagnostic,
    save_plain_reading_generated_checkpoint,
    save_plain_reading_need_view_checkpoint,
    save_plain_reading_reviewed_checkpoint,
    save_segmentation_generated_checkpoint,
    save_segmentation_reviewed_checkpoint,
)
from human_segmentation_loader import (
    SegmentationComparisonReport,
    compare_segmentations,
    load_human_segmentations_from_markdown,
)
from llm import LLMClientError, build_llm_client_from_env
from plain_reading import (
    PlainReadingValidationError,
    assess_plain_reading_ir_readiness,
    advance_human_semantic_override_ap_review,
    apply_plain_reading_revision,
    apply_plain_reading_decisions,
    cancel_human_semantic_override_ap_review,
    confirm_human_semantic_override,
    confirm_no_ambiguity,
    confirm_plain_reading_revision_classification,
    close_plain_reading_revision_request,
    create_plain_reading_revision_request,
    create_human_semantic_override,
    plain_reading_from_mapping,
    request_ambiguity_generation,
    validate_rebuilt_plain_reading_axes,
)
from plain_reading_workflow import generate_plain_reading_outcome
from plain_reading_need_view import build_plain_reading_need_view
from rag.memory_store import build_persistent_retriever
from pre_ir_review import PreIRReviewConfig, run_pre_ir_review, complete_plain_reading_with_model
from requirement_loader import RequirementRecord
from schemas import (
    APIdentityMapping,
    APInventory,
    APRepresentationProfile,
    DeclaredAPCatalog,
    HumanSemanticOverride,
    OverrideAPComparison,
    PlainReadingReview,
    SurfaceSegmentation,
)
from ap_identity import (
    ap_representation_profile_from_mapping,
    approve_ap_inventory,
    declared_ap_catalog_from_mapping,
    propose_grounded_ap_inventory,
    promote_replacement_ap_inventory,
    undetermined_profile,
)
from ap_grounding import build_ap_grounding_contract
from surface_segmentation import segmentation_from_mapping, source_hash
from surface_segmentation import revise_segmentation
from terminal_progress import TerminalProgress
from replacement_review import (
    REPLACEMENT_DRAFTS_DIR,
    ReplacementReviewDecision,
    ReplacementReviewDraft,
    ReplacementReviewInputError,
    format_replacement_review_markdown,
    load_replacement_review_draft,
    now_timestamp as replacement_review_timestamp,
    parse_replacement_review_markdown,
    replacement_draft_path,
    replacement_review_snapshot_id,
    save_replacement_review_draft,
)
from validated_generation import (
    CONTRACT_VERSION as VALIDATED_GENERATION_CONTRACT_VERSION,
    PATCH_AUTHORIZATION_CONTRACT_VERSION,
    REPAIR_POLICY_VERSION,
    VALIDATOR_VERSION as VALIDATED_GENERATION_VALIDATOR_VERSION,
)


# State v3 is already the runtime AP-substage contract.  runtime therefore
# occupies v4 while retaining the planned review-document v2 -> v3 upgrade.
PRE_IR_BATCH_STATE_SCHEMA_VERSION = 5
PRE_IR_REVIEW_SCHEMA_VERSION = 3
PRE_IR_BATCH_STATE_FILE = "pre_ir_batch_state.json"
PREPROCESSING_ERRORS_FILE = "pre_ir_preprocessing_errors.md"
PRE_IR_TRANSACTIONS_DIR = "pre_ir_transactions"
PRE_IR_BATCHES_DIR = "pre_ir_batches"
SEMANTIC_FALLBACK_REVIEW_BASIS = "llm_no_source_supported_axis_after_repair"
REQUIREMENT_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*-\d+")


@dataclass(frozen=True)
class PreIRBatchOutcome:
    should_stop: bool
    message: str
    state: dict[str, Any]


@dataclass(frozen=True)
class ReviewApplyOutcome:
    message: str
    next_review_files: tuple[str, ...] = ()
    pre_ir_complete: bool = False
    had_errors: bool = False


@dataclass(frozen=True)
class ReplacementReviewQueueOutcome:
    """Structured result from one interactive replacement Pre-IR queue run."""

    message: str
    initial_ids: tuple[str, ...] = ()
    attempted_ids: tuple[str, ...] = ()
    completed_ids: tuple[str, ...] = ()
    remaining_ids: tuple[str, ...] = ()
    halt_reason: str = ""
    pre_ir_complete: bool = False
    had_errors: bool = False


@dataclass(frozen=True)
class ReplacementReviewExportOutcome:
    """Result of producing one isolated replacement Markdown review round."""

    message: str
    review_file: str = ""
    pending_ids: tuple[str, ...] = ()
    failed_ids: tuple[str, ...] = ()
    reused_ids: tuple[str, ...] = ()


class ReviewFileInputError(ValueError):
    """The submitted review file failed side-effect-free preflight."""


class PreIRStateConsistencyError(RuntimeError):
    """Canonical state changed but its derived review views could not recover."""


class _InteractiveReplacementActivated(BaseException):
    """Internal non-error control transfer after a durable human transition."""

    def __init__(self, outcome: ReviewApplyOutcome) -> None:
        super().__init__(outcome.message)
        self.outcome = outcome


@dataclass(frozen=True)
class _ItemTransaction:
    transaction_id: str
    directory: Path


def _canonical_path(path: str | Path) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def _same_path(left: str | Path, right: str | Path) -> bool:
    return _canonical_path(left) == _canonical_path(right)


def _checkpoint_fingerprint(checkpoint_dir: str | Path) -> str:
    """Return the stable identity used to isolate this checkpoint's views."""

    return hashlib.sha256(_canonical_path(checkpoint_dir).encode("utf-8")).hexdigest()


def _review_namespace_root(
    records_dir: str | Path,
    checkpoint_dir: str | Path,
) -> Path:
    return (
        Path(records_dir)
        / PRE_IR_BATCHES_DIR
        / _checkpoint_fingerprint(checkpoint_dir)
    )


@contextmanager
def _review_apply_lock(root: Path):
    """Hold one OS-released process lock for a deferred review mutation."""

    lock_root = Path(tempfile.gettempdir()) / "traj_ltlf_pre_ir_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    identity = _checkpoint_fingerprint(root)
    path = lock_root / f"{identity}.lock"
    handle = path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise ValueError("另一个Pre-IR审核命令正在修改同一checkpoint。") from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _serialized_review_apply(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        checkpoint_dir = kwargs.get("checkpoint_dir")
        if not checkpoint_dir and function.__name__ in {
            "load_pre_ir_batch_state",
            "mark_pre_ir_downstream_started",
        } and args:
            checkpoint_dir = args[0]
        if not checkpoint_dir:
            raise ValueError("审核命令需要非空checkpoint_dir。")
        root = Path(str(checkpoint_dir))
        state_path = root / PRE_IR_BATCH_STATE_FILE
        with _review_apply_lock(root):
            before_state = state_path.read_bytes() if state_path.exists() else None
            try:
                if function.__name__ == "reopen_pre_ir_item":
                    requirement_id = str(args[0] if args else "").strip().upper()
                    if REQUIREMENT_ID_PATTERN.fullmatch(requirement_id) is None:
                        raise ValueError(
                            "--reopen-pre-ir 的 requirement_id 格式无效。"
                        )
                    loaded_state = _require_state(root)
                    with _item_transaction_context(
                        root, requirement_id
                    ) as transaction:
                        call_kwargs = dict(kwargs)
                        call_kwargs["_transaction"] = transaction
                        call_kwargs["_loaded_state"] = loaded_state
                        try:
                            return function(*args, **call_kwargs)
                        except _InteractiveReplacementActivated as transition:
                            return transition.outcome
                return function(*args, **kwargs)
            except BaseException:
                after_state = state_path.read_bytes() if state_path.exists() else None
                if (
                    (
                        function.__name__ == "reopen_pre_ir_item"
                        or (
                            function.__name__.startswith("apply_")
                            and before_state != after_state
                        )
                    )
                    and after_state is not None
                ):
                    try:
                        state = _load_state(root)
                        if state and state.get("generation_complete"):
                            _refresh_review_files(
                                state,
                                checkpoint_dir=str(root),
                                records_dir=str(
                                    state.get("records_dir") or "review_records"
                                ),
                                force=True,
                            )
                            if function.__name__ != "reopen_pre_ir_item":
                                _close_if_complete(state)
                                _write_state(root, state)
                    except BaseException as recovery_exc:
                        raise PreIRStateConsistencyError(
                            "Pre-IR state changed and derived review recovery failed"
                        ) from recovery_exc
                raise

    return wrapped


@_serialized_review_apply
def reopen_pre_ir_item(
    requirement_id: str,
    *,
    checkpoint_dir: str,
    records_dir: str | None = None,
    llm_mode: str | None = "api",
    llm_client: JSONLLMClient | None = None,
    approved_memory_file: str | None = None,
    max_segmentation_retries: int = 2,
    max_surface_review_rounds: int = 3,
    max_plain_reading_rounds: int = 2,
    max_plain_reading_revision_rounds: int = 2,
    segmentation_prompt_version: str = "plain-segmentation-v1",
    plain_reading_prompt_version: str = "plain-reading-v5",
    semantic_fallback_enabled: bool = True,
    _required_status: str | None = None,
    _transaction: _ItemTransaction | None = None,
    _loaded_state: dict[str, Any] | None = None,
) -> ReviewApplyOutcome:
    """Reopen one item in the original interactive Pre-IR interface."""

    root = Path(checkpoint_dir)
    state = _loaded_state if _loaded_state is not None else _require_state(root)
    records_dir = str(state.get("records_dir") or records_dir or "review_records")
    normalized_id = requirement_id.strip().upper()
    item = state.get("items", {}).get(normalized_id)
    if not isinstance(item, dict):
        raise ValueError(f"未知 requirement_id: {normalized_id}")
    if _required_status is not None and item.get("status") != _required_status:
        raise ValueError(
            f"{normalized_id} 当前状态为 {item.get('status')!r}，"
            f"不再是队列预期的 {_required_status!r}。"
        )
    if state.get("downstream_started"):
        if item.get("status") not in {
            "plain_reading_need_view",
            "deferred_source_clarification",
        }:
            raise ValueError(
                "下游已启动后只能单项重开 deferred Plain Reading。"
            )
        descendant = _semantic_descendant_for_reading(
            root,
            normalized_id,
            str(item.get("reading_review_id", "")),
        )
        if descendant is not None:
            _write_semantic_quarantine_record(root, normalized_id, descendant)
            raise PreIRStateConsistencyError(
                f"{normalized_id}: deferred Plain Reading has a quarantined "
                "Semantic IR descendant; manual inspection is required"
            )
    if llm_client is None:
        llm_client = build_llm_client_from_env(llm_mode)
    if llm_client is None:
        raise ValueError("交互式重开需要 --llm api。")

    original = str(item["original_requirement"])
    effective = str(item.get("effective_requirement") or original)
    active_override = load_human_semantic_override_checkpoint(
        checkpoint_dir,
        normalized_id,
        active_only=True,
    )
    generated_segmentations: list[SurfaceSegmentation] = []
    retriever = build_persistent_retriever(approved_memory_file)

    def persist_semantic_override(override: HumanSemanticOverride) -> None:
        save_human_semantic_override_checkpoint(
            checkpoint_dir,
            override=override,
        )
        if (
            override.status != "active"
            or override.override_id
            == str(item.get("active_override_id", ""))
        ):
            return
        if _transaction is None:
            raise PreIRStateConsistencyError(
                "interactive replacement activation requires an item transaction"
            )
        updated_item = _activate_confirmed_replacement(dict(item), override)
        _commit_item_state(
            root,
            state,
            normalized_id,
            updated_item,
            transaction=_transaction,
        )
        reconciliation_error = ""
        try:
            _refresh_review_files(
                state,
                checkpoint_dir=checkpoint_dir,
                records_dir=records_dir,
                force=True,
            )
            _close_if_complete(state)
            _write_state(root, state)
        except Exception as exc:
            # The human authorization and canonical item transition were
            # already committed.  A derived-view failure must not undo them.
            reconciliation_error = f"{type(exc).__name__}: {exc}"
        lines = [
            (
                "[PARTIAL] Replacement activated; review-view reconciliation failed."
                if reconciliation_error
                else "[WAIT] Replacement activated; manual surface review is required."
            ),
            f"Activated requirement: {normalized_id}",
            "No replacement segmentation or plain reading was generated in this transition.",
        ]
        if reconciliation_error:
            lines.extend(("Error:", "  " + reconciliation_error))
        lines.extend(_next_file_lines(state, include_commands=True))
        raise _InteractiveReplacementActivated(
            ReviewApplyOutcome(
                "\n".join(lines),
                _pending_files(state),
                False,
                bool(reconciliation_error),
            )
        )

    artifacts = run_pre_ir_review(
        requirement_id=normalized_id,
        requirement=effective,
        original_source_text=original,
        active_semantic_override=active_override,
        retriever=retriever,
        llm_client=llm_client,
        config=PreIRReviewConfig(
            segmentation_review="manual",
            plain_reading_review="manual",
            max_segmentation_retries=max_segmentation_retries,
            max_surface_review_rounds=max_surface_review_rounds,
            max_plain_reading_rounds=max_plain_reading_rounds,
            max_plain_reading_revision_rounds=max_plain_reading_revision_rounds,
            segmentation_prompt_version=segmentation_prompt_version,
            plain_reading_prompt_version=plain_reading_prompt_version,
            data_split=str(item.get("data_split", "test")),
            semantic_fallback_enabled=semantic_fallback_enabled,
        ),
        interactive=True,
        reviewer_id="interactive-reopen-reviewer",
        ap_representation_profile=_item_ap_profile(item),
        declared_ap_catalog=_item_declared_catalog(item),
        current_ap_inventory=_item_current_ap_inventory(
            checkpoint_dir, item
        ),
        ap_artifact_callback=(
            lambda artifact: save_ap_identity_artifact_checkpoint(
                checkpoint_dir, artifact
            )
        ),
        segmentation_artifact_callback=(
            lambda segmentation: (
                generated_segmentations.append(segmentation),
                save_segmentation_generated_checkpoint(
                    checkpoint_dir,
                    requirement_id=normalized_id,
                    segmentation=segmentation,
                ),
            )
        ),
        artifact_callback=(
            lambda segmentation, review: save_plain_reading_generated_checkpoint(
                checkpoint_dir,
                requirement_id=normalized_id,
                plain_reading_review=review,
                segmentation_schema_version=segmentation.schema_version,
            )
        ),
        generation_diagnostic_callback=_plain_reading_diagnostic_callback(
            checkpoint_dir,
            normalized_id,
        ),
        semantic_override_callback=persist_semantic_override,
    )
    segmentation = artifacts.surface_segmentation
    review = artifacts.plain_reading_review
    save_segmentation_reviewed_checkpoint(
        checkpoint_dir,
        requirement_id=normalized_id,
        segmentation=segmentation,
    )
    _save_reviewed(checkpoint_dir, segmentation, review)
    generated = generated_segmentations[-1] if generated_segmentations else segmentation
    if generated.source_hash != segmentation.source_hash:
        # A whole-source semantic replacement can regenerate the effective
        # segmentation inside the interactive flow.  Keep the final checkpoint
        # internally consistent even when that nested proposal is not exposed by
        # the initial-generation callback.
        generated = segmentation
        save_segmentation_generated_checkpoint(
            checkpoint_dir,
            requirement_id=normalized_id,
            segmentation=generated,
        )
    comparison = compare_segmentations(generated, segmentation)
    _write_comparison(root, comparison)
    returned_override = artifacts.human_semantic_override
    active_override_id = (
        returned_override.override_id if returned_override is not None else ""
    )
    superseded_artifacts = list(item.get("superseded_pre_ir_artifacts", ()))
    if active_override_id and active_override_id != str(
        item.get("active_override_id", "")
    ):
        if returned_override is None or (
            returned_override.parent_effective_source_hash
            != str(item.get("source_hash", ""))
        ):
            raise PreIRStateConsistencyError(
                f"{normalized_id}: interactive semantic override parent is stale"
            )
        superseded_artifacts = _append_superseded_artifact_record(
            item,
            active_override_id,
        )
    updated_item = dict(item)
    updated_item.update(
        effective_requirement=artifacts.effective_source_text or segmentation.source_text,
        source_hash=segmentation.source_hash,
        segmentation_id=segmentation.segmentation_id,
        llm_segmentation_id=generated.segmentation_id,
        reading_review_id=review.reading_review_id,
        plain_status=review.overall_status,
        comparison=comparison.to_dict(),
        active_override_id=active_override_id,
        pending_override_id="",
        superseded_pre_ir_artifacts=superseded_artifacts,
        error="",
    )
    updated_item["override_id"] = updated_item["active_override_id"]
    updated_item.pop("replacement_segmentation_review_required", None)
    _clear_item_errors(updated_item)
    _route_item_by_readiness(
        updated_item,
        review,
        segmentation,
        checkpoint_dir=checkpoint_dir,
    )
    _commit_item_state(
        root,
        state,
        normalized_id,
        updated_item,
        transaction=_transaction,
    )
    _refresh_review_files(
        state,
        checkpoint_dir=checkpoint_dir,
        records_dir=records_dir,
    )
    _close_if_complete(state)
    _write_state(root, state)
    return ReviewApplyOutcome(
        "\n".join(
            (
                f"[OK] {normalized_id} Pre-IR interaction completed.",
                *_next_file_lines(state, include_commands=True),
                (
                    "Pre-IR review is complete; run the main command to continue."
                    if state.get("pre_ir_closed")
                    else "Other Pre-IR review items are still pending."
                ),
            )
        ),
        _pending_files(state),
        bool(state.get("pre_ir_closed")),
    )


@_serialized_review_apply
def mark_pre_ir_downstream_started(checkpoint_dir: str) -> None:
    root = Path(checkpoint_dir)
    state = _load_state(root)
    if not state:
        return
    if not state.get("pre_ir_closed"):
        raise ValueError("Pre-IR review is not complete.")
    _validate_canonical_item_artifacts(state, root)
    if not state.get("downstream_eligible"):
        raise ValueError("Pre-IR review has no downstream-eligible item.")
    state["downstream_started"] = True
    state["run_status"] = "downstream_started"
    state["updated_at"] = _timestamp()
    _write_state(root, state)


@_serialized_review_apply
def ensure_deferred_pre_ir_batch(
    records: Sequence[RequirementRecord],
    *,
    human_segmentation_file: str,
    checkpoint_dir: str,
    records_dir: str,
    llm_mode: str | None,
    llm_client: JSONLLMClient | None = None,
    approved_memory_file: str | None = None,
    max_segmentation_retries: int = 2,
    segmentation_prompt_version: str = "plain-segmentation-v1",
    plain_reading_prompt_version: str = "plain-reading-v5",
    data_split: str = "test",
    semantic_fallback_enabled: bool = True,
    progress: Any | None = None,
) -> PreIRBatchOutcome:
    """Generate all pending Pre-IR candidates, or report the current gate."""

    root = Path(_canonical_path(checkpoint_dir))
    checkpoint_dir = str(root)
    state = _load_state(root)
    if state:
        # The persisted records root is part of the batch identity.  A later
        # invocation must not silently relocate active review files merely
        # because it was launched with a different records_dir argument.
        records_dir = str(state.get("records_dir") or records_dir)
    else:
        records_dir = _canonical_path(records_dir)
    if state and state.get("generation_complete") and llm_mode == "api":
        pending = [
            (requirement_id, item) for requirement_id, item in state.get("items", {}).items()
            if item.get("status") in {"awaiting_no_ambiguity_review", "awaiting_ambiguity_review", "awaiting_semantic_fallback_review"}
        ]
        if pending:
            llm_client = llm_client or build_llm_client_from_env(llm_mode)
            for requirement_id, item in pending:
                segmentation, reading = _load_pending_artifacts(checkpoint_dir, requirement_id, item)
                reading = complete_plain_reading_with_model(segmentation.source_text, segmentation, reading, llm_client)
                _save_reviewed(checkpoint_dir, segmentation, reading)
                _route_item_by_readiness(item, reading, segmentation, checkpoint_dir=checkpoint_dir)
            _refresh_review_files(state, checkpoint_dir=checkpoint_dir, records_dir=records_dir, force=True)
            _close_if_complete(state)
            _write_state(root, state)
    if state and state.get("generation_complete"):
        _validate_canonical_item_artifacts(state, root)
    if state and state.get("generation_complete") and _active_review_views_inconsistent(
        state,
        checkpoint_dir=checkpoint_dir,
    ):
        _refresh_review_files(
            state,
            checkpoint_dir=checkpoint_dir,
            records_dir=records_dir,
            force=True,
        )
        _close_if_complete(state)
        _write_state(root, state)
    generation_contract = _preprocessing_generation_contract(
        segmentation_prompt_version=segmentation_prompt_version,
        plain_reading_prompt_version=plain_reading_prompt_version,
        semantic_fallback_enabled=semantic_fallback_enabled,
    )
    retry_blocked_for_contract_upgrade = False
    if state and state.get("generation_complete"):
        blocked_items = [
            item
            for item in state.get("items", {}).values()
            if item.get("status") == "blocked"
        ]
        retry_blocked_for_contract_upgrade = bool(blocked_items) and (
            state.get("preprocessing_generation_contract") != generation_contract
        )
        if not retry_blocked_for_contract_upgrade:
            if state.get("pre_ir_closed"):
                if not state.get("downstream_eligible"):
                    return PreIRBatchOutcome(
                        True,
                        "Pre-IR review cycle is closed, but no item is downstream eligible.",
                        state,
                    )
                return PreIRBatchOutcome(False, "Pre-IR review is complete.", state)
            return PreIRBatchOutcome(True, _pending_message(state), state)

    if llm_client is None:
        llm_client = build_llm_client_from_env(llm_mode)
    if llm_client is None:
        raise ValueError("延迟Pre-IR人工审核需要 --llm api。")

    expected = {record.requirement_id: record.text for record in records}
    human_load = load_human_segmentations_from_markdown(
        human_segmentation_file,
        expected_requirements=expected,
        data_split=data_split,
    )
    previous_state = state
    items: dict[str, dict[str, Any]] = (
        dict(state.get("items", {}))
        if state
        and (
            not state.get("generation_complete")
            or retry_blocked_for_contract_upgrade
        )
        else {}
    )
    pending_before = {
        "no_ambiguity": set(_pending_ids(state, "no_ambiguity")) if state else set(),
        "semantic_fallback": set(_pending_ids(state, "semantic_fallback"))
        if state
        else set(),
        "ap_inventory": set(_pending_ids(state, "ap_inventory")) if state else set(),
        "ap_mapping": set(_pending_ids(state, "ap_mapping")) if state else set(),
        "ambiguity": set(_pending_ids(state, "ambiguity")) if state else set(),
    }
    # Schema migration and newer writers may add compatibility/audit fields
    # which this state-machine version does not need to interpret.  A blocked
    # contract-upgrade retry must not silently discard those durable fields.
    state = dict(previous_state or {})
    state.update({
        "schema_version": PRE_IR_BATCH_STATE_SCHEMA_VERSION,
        "preprocessing_generation_contract": generation_contract,
        "generation_complete": False,
        "no_ambiguity_review_complete": bool(
            previous_state.get("no_ambiguity_review_complete", False)
        ),
        "ambiguity_review_complete": bool(
            previous_state.get("ambiguity_review_complete", False)
        ),
        "semantic_fallback_review_complete": bool(
            previous_state.get("semantic_fallback_review_complete", False)
        ),
        "ap_inventory_review_complete": bool(
            previous_state.get("ap_inventory_review_complete", False)
        ),
        "ap_mapping_review_complete": bool(
            previous_state.get("ap_mapping_review_complete", False)
        ),
        "pre_ir_closed": False,
        "downstream_started": bool(
            previous_state.get("downstream_started", False)
        ),
        "no_ambiguity_round": int(
            previous_state.get("no_ambiguity_round", 1)
        ),
        "ambiguity_round": int(previous_state.get("ambiguity_round", 1)),
        "semantic_fallback_round": int(
            previous_state.get("semantic_fallback_round", 1)
        ),
        "ap_inventory_round": int(previous_state.get("ap_inventory_round", 1)),
        "ap_mapping_round": int(previous_state.get("ap_mapping_round", 1)),
        "pending_no_ambiguity_file": str(
            previous_state.get("pending_no_ambiguity_file", "")
        ),
        "pending_ambiguity_file": str(
            previous_state.get("pending_ambiguity_file", "")
        ),
        "pending_semantic_fallback_file": str(
            previous_state.get("pending_semantic_fallback_file", "")
        ),
        "pending_ap_inventory_file": str(
            previous_state.get("pending_ap_inventory_file", "")
        ),
        "pending_ap_mapping_file": str(
            previous_state.get("pending_ap_mapping_file", "")
        ),
        "pending_replacement_review_file": str(
            previous_state.get("pending_replacement_review_file", "")
        ),
        "replacement_review_round": int(
            previous_state.get("replacement_review_round", 1)
        ),
        "replacement_review_snapshot_id": str(
            previous_state.get("replacement_review_snapshot_id", "")
        ),
        "run_status": "generating_pre_ir",
        "checkpoint_dir": str(checkpoint_dir),
        "records_dir": str(records_dir),
        "review_namespace_dir": str(
            _review_namespace_root(records_dir, checkpoint_dir)
        ),
        "human_segmentation_file": str(human_segmentation_file),
        "items": items,
        "created_at": (
            previous_state.get("created_at", _timestamp())
            if previous_state
            else _timestamp()
        ),
        "updated_at": _timestamp(),
    })
    _write_state(root, state)
    preprocessing_errors: dict[str, str] = dict(human_load.errors)
    for requirement_id in human_load.missing_requirement_ids:
        preprocessing_errors[requirement_id] = "人工切分文件缺少该需求。"

    for index, record in enumerate(records, start=1):
        requirement_id = record.requirement_id
        if progress is not None:
            progress.info(
                f"{requirement_id} deferred Pre-IR {index}/{len(records)}"
            )
        existing_item = items.get(requirement_id)
        if existing_item and existing_item.get("status") != "blocked":
            continue
        human = human_load.segmentations.get(requirement_id)
        if human is None:
            items[requirement_id] = _blocked_item(
                record,
                preprocessing_errors.get(requirement_id, "人工切分不可用。"),
            )
            state["updated_at"] = _timestamp()
            _write_state(root, state)
            continue
        retriever = build_persistent_retriever(approved_memory_file)
        generation_error: dict[str, Any] | None = None
        try:
            llm_segmentation = SegmentationAgent(
                retriever, llm_client=llm_client
            ).generate(
                record.text,
                requirement_id=requirement_id,
                max_retries=max_segmentation_retries,
                prompt_version=segmentation_prompt_version,
                experiment_mode="manual",
                data_split=data_split,
            )
            comparison = compare_segmentations(llm_segmentation, human)
            save_segmentation_generated_checkpoint(
                checkpoint_dir,
                requirement_id=requirement_id,
                segmentation=llm_segmentation,
            )
            save_segmentation_reviewed_checkpoint(
                checkpoint_dir,
                requirement_id=requirement_id,
                segmentation=human,
            )
            _write_comparison(root, comparison)
            reading_agent = PlainReadingAgent(retriever, llm_client=llm_client)
            try:
                reading = generate_plain_reading_outcome(
                    reading_agent,
                    record.text,
                    human,
                    semantic_fallback_enabled=semantic_fallback_enabled,
                    requirement_id=requirement_id,
                    mode="manual",
                    prompt_version=plain_reading_prompt_version,
                    data_split=data_split,
                    generation_diagnostic_callback=(
                        _plain_reading_diagnostic_callback(
                            checkpoint_dir,
                            requirement_id,
                        )
                    ),
                    segmentation_comparison_context={
                        "boundary_relation": comparison.boundary_relation,
                        "llm_blind_marked_text": _marked_text(llm_segmentation),
                        "human_presegmented_marked_text": _marked_text(human),
                        "llm_only_cut_offsets": list(comparison.llm_only_offsets),
                        "human_only_cut_offsets": list(comparison.human_only_offsets),
                    },
                    ap_grounding_contract=build_ap_grounding_contract(
                        requirement_id=requirement_id,
                        source_text=record.text,
                        representation_profile=(
                            record.ap_representation_profile
                            or undetermined_profile(
                                provenance="deferred_pre_ir_default"
                            )
                        ),
                        segmentation=human,
                        declared_catalog=record.declared_ap_catalog,
                        provenance=("deferred_pre_ir_generation",),
                    ),
                ).review
                if llm_client is not None:
                    reading = complete_plain_reading_with_model(record.text, human, reading, llm_client)
            except Exception as exc:
                generation_error = _attach_plain_reading_generation_error(
                    exc,
                    checkpoint_dir=checkpoint_dir,
                    requirement_id=requirement_id,
                )
                raise
            save_plain_reading_generated_checkpoint(
                checkpoint_dir,
                requirement_id=requirement_id,
                plain_reading_review=reading,
                segmentation_schema_version=human.schema_version,
            )
            items[requirement_id] = _generated_item(
                record, human, llm_segmentation, comparison, reading
            )
            if llm_client is not None:
                _save_reviewed(checkpoint_dir, human, reading)
                _route_item_by_readiness(items[requirement_id], reading, human, checkpoint_dir=checkpoint_dir)
        except Exception as exc:
            items[requirement_id] = _blocked_item(
                record,
                str(exc),
                generation_error=generation_error,
            )
        state["updated_at"] = _timestamp()
        _write_state(root, state)

    state["generation_complete"] = True
    state["run_status"] = "awaiting_pre_ir_review"
    added_pending_ids: set[str] = set()
    if retry_blocked_for_contract_upgrade:
        added_pending_ids = (
            set(_pending_ids(state, "no_ambiguity"))
            - pending_before["no_ambiguity"]
        ) | (
            set(_pending_ids(state, "semantic_fallback"))
            - pending_before["semantic_fallback"]
        ) | (
            set(_pending_ids(state, "ambiguity")) - pending_before["ambiguity"]
        )
        _advance_completed_review_rounds_for_retry(
            state,
            records_dir=records_dir,
            checkpoint_dir=checkpoint_dir,
            pending_before=pending_before,
            added_requirement_ids=added_pending_ids,
        )
    _refresh_review_files(
        state,
        checkpoint_dir=checkpoint_dir,
        records_dir=records_dir,
        force=not retry_blocked_for_contract_upgrade,
        added_requirement_ids=tuple(added_pending_ids),
    )
    _write_preprocessing_errors(records_dir, state)
    _close_if_complete(state)
    _write_state(root, state)
    if state.get("pre_ir_closed") and state.get("downstream_eligible"):
        return PreIRBatchOutcome(False, "读法模型检查完成，继续 AP 和轨迹审核。", state)
    return PreIRBatchOutcome(True, _initial_message(state), state)


@_serialized_review_apply
def apply_no_ambiguity_review(
    review_file: str,
    *,
    checkpoint_dir: str,
    records_dir: str | None = None,
    llm_mode: str | None = "api",
    llm_client: JSONLLMClient | None = None,
    approved_memory_file: str | None = None,
    plain_reading_prompt_version: str = "plain-reading-v5",
    semantic_fallback_enabled: bool = True,
    progress: TerminalProgress | None = None,
) -> ReviewApplyOutcome:
    root = Path(checkpoint_dir)
    if progress is not None:
        progress.stage("Loading no-ambiguity review")
        progress.info(f"Review file: {review_file}")
        progress.info(f"Checkpoint directory: {checkpoint_dir}")
    state = _require_state(root)
    records_dir = str(state.get("records_dir") or records_dir or "review_records")
    review_path = _require_active_review_file(
        review_file,
        state,
        path_key="pending_no_ambiguity_file",
    )
    text = review_path.read_text(encoding="utf-8")
    pending_ids = _pending_ids(state, "no_ambiguity")
    _validate_review_document(
        text,
        review_type="no_ambiguity_screening",
        round_number=int(state.get("no_ambiguity_round", 1)),
        requirement_ids=pending_ids,
    )
    sections = _requirement_sections(text)
    if progress is not None:
        progress.ok("Review file is marked complete")
        progress.info(f"Pending no-ambiguity requirements: {len(pending_ids)}")
    prepared = _preflight_no_ambiguity_review(
        sections,
        pending_ids,
        state=state,
        checkpoint_dir=checkpoint_dir,
    )
    errors: list[str] = []
    promoted: list[str] = []
    fallbacks: list[str] = []
    accepted: list[str] = []
    deferred: list[str] = []
    needs_llm = any(
        command == "e" for _segmentation, _review, command, _comment in prepared.values()
    )
    if llm_client is None and needs_llm:
        if progress is not None:
            progress.task_start("Initializing LLM client for promoted requirements")
        try:
            llm_client = build_llm_client_from_env(llm_mode)
        finally:
            if progress is not None:
                progress.task_stop()
        if progress is not None:
            progress.ok("LLM client ready")

    for index, requirement_id in enumerate(pending_ids, start=1):
        label = f"{requirement_id} [{index}/{len(pending_ids)}]"
        item = state["items"][requirement_id]
        try:
            if progress is not None:
                progress.info(f"{label} validating metadata and checkpoints")
            segmentation, review, command, comment = prepared[requirement_id]
            if command == "a":
                reviewed = confirm_no_ambiguity(
                    review,
                    segmentation,
                    reviewer_id="batch-markdown-reviewer",
                    reviewer_source="human",
                    comment="Accepted in deferred no-ambiguity review.",
                )
                with _item_transaction_context(root, requirement_id) as transaction:
                    _save_reviewed(checkpoint_dir, segmentation, reviewed)
                    updated_item = dict(item)
                    _route_item_by_readiness(
                        updated_item,
                        reviewed,
                        segmentation,
                        checkpoint_dir=checkpoint_dir,
                    )
                    _commit_item_state(
                        root,
                        state,
                        requirement_id,
                        updated_item,
                        transaction=transaction,
                    )
                accepted.append(requirement_id)
                if progress is not None:
                    progress.ok(f"{label} accepted: no ambiguity")
            elif command == "e":
                if llm_client is None:
                    raise ValueError("e 需要可用的 --llm api client。")
                request = request_ambiguity_generation(
                    review,
                    segmentation,
                    suspected_segment_ids=(),
                    user_comment=comment,
                    reviewer_id="batch-markdown-reviewer",
                    reviewer_source="human",
                )
                retriever = build_persistent_retriever(approved_memory_file)
                if progress is not None:
                    progress.task_start(
                        f"{label} generating candidate ambiguity axes"
                    )
                try:
                    try:
                        outcome = generate_plain_reading_outcome(
                            PlainReadingAgent(retriever, llm_client=llm_client),
                            item["effective_requirement"],
                            segmentation,
                            semantic_fallback_enabled=semantic_fallback_enabled,
                            requirement_id=requirement_id,
                            mode="manual",
                            prompt_version=plain_reading_prompt_version,
                            data_split=item.get("data_split", "test"),
                            ambiguity_generation_request=request,
                            ap_grounding_contract=(
                                _item_ap_grounding_contract(
                                    item,
                                    segmentation=segmentation,
                                    checkpoint_dir=checkpoint_dir,
                                )
                            ),
                            generation_diagnostic_callback=(
                                _plain_reading_diagnostic_callback(
                                    checkpoint_dir,
                                    requirement_id,
                                )
                            ),
                        )
                    except Exception as exc:
                        _attach_plain_reading_generation_error(
                            exc,
                            checkpoint_dir=checkpoint_dir,
                            requirement_id=requirement_id,
                        )
                        raise
                    generated = outcome.review
                finally:
                    if progress is not None:
                        progress.task_stop()
                with _item_transaction_context(root, requirement_id) as transaction:
                    save_plain_reading_generated_checkpoint(
                        checkpoint_dir,
                        requirement_id=requirement_id,
                        plain_reading_review=generated,
                        segmentation_schema_version=segmentation.schema_version,
                    )
                    updated_item = dict(item)
                    _set_pending_item(updated_item, generated)
                    _commit_item_state(
                        root,
                        state,
                        requirement_id,
                        updated_item,
                        transaction=transaction,
                    )
                if outcome.semantic_fallback:
                    fallbacks.append(requirement_id)
                else:
                    promoted.append(requirement_id)
                if progress is not None:
                    progress.ok(
                        f"{label} routed to semantic fallback review"
                        if outcome.semantic_fallback
                        else f"{label} promoted to ambiguity review"
                    )
            elif command == "c":
                deferred.append(requirement_id)
                if progress is not None:
                    progress.info(f"{label} deferred without state change")
        except (ValueError, PlainReadingValidationError, LLMClientError) as exc:
            errors.append(f"{requirement_id}: {exc}")
            _record_item_apply_error(item, exc)
            if progress is not None:
                progress.warn(f"{label} failed: {exc}")

    if progress is not None:
        progress.stage("Refreshing deferred Pre-IR review state")
    state["no_ambiguity_review_complete"] = not bool(
        _pending_ids(state, "no_ambiguity")
    )
    state["semantic_fallback_review_complete"] = not bool(
        _pending_ids(state, "semantic_fallback")
    )
    _refresh_review_files(
        state,
        checkpoint_dir=checkpoint_dir,
        records_dir=records_dir,
        force_ambiguity=True,
    )
    _close_if_complete(state)
    _write_state(root, state)
    next_files = _pending_files(state)
    if progress is not None:
        progress.ok("Deferred Pre-IR state and review files refreshed")
        progress.info(
            f"No-ambiguity result: accepted={len(accepted)}, "
            f"promoted={len(promoted)}, fallback={len(fallbacks)}, "
            f"deferred={len(deferred)}, errors={len(errors)}"
        )
        for next_file in next_files:
            progress.info(f"Next review file: {next_file}")
    message_lines = [
        "[PARTIAL] No-ambiguity review applied with item errors."
        if errors
        else "[OK] No-ambiguity review applied.",
        f"Accepted: {len(accepted)}",
        f"Promoted to ambiguity review: {len(promoted)}",
        f"Routed to semantic fallback review: {len(fallbacks)}",
        f"Deferred: {len(deferred)}",
    ]
    if errors:
        message_lines.extend(("Errors:", *("  " + error for error in errors)))
    message_lines.extend(_next_file_lines(state))
    return ReviewApplyOutcome(
        "\n".join(message_lines),
        next_files,
        bool(state.get("pre_ir_closed")),
        bool(errors),
    )


@_serialized_review_apply
def apply_semantic_fallback_review(
    review_file: str,
    *,
    checkpoint_dir: str,
    records_dir: str | None = None,
    llm_mode: str | None = "api",
    llm_client: JSONLLMClient | None = None,
    approved_memory_file: str | None = None,
    max_segmentation_retries: int = 2,
    segmentation_prompt_version: str = "plain-segmentation-v1",
    plain_reading_prompt_version: str = "plain-reading-v5",
    semantic_fallback_enabled: bool = True,
    progress: TerminalProgress | None = None,
) -> ReviewApplyOutcome:
    """Apply Y/R/C fallback decisions and explicit override confirmations."""

    root = Path(checkpoint_dir)
    if progress is not None:
        progress.stage("Loading semantic-fallback review")
        progress.info(f"Review file: {review_file}")
    state = _require_state(root)
    records_dir = str(state.get("records_dir") or records_dir or "review_records")
    review_path = _require_active_review_file(
        review_file,
        state,
        path_key="pending_semantic_fallback_file",
    )
    text = review_path.read_text(encoding="utf-8")
    pending_ids = _pending_ids(state, "semantic_fallback")
    _validate_review_document(
        text,
        review_type="semantic_fallback_adjudication",
        round_number=int(state.get("semantic_fallback_round", 1)),
        requirement_ids=pending_ids,
    )
    sections = _requirement_sections(text)
    prepared = _preflight_semantic_fallback_review(
        sections,
        pending_ids,
        state=state,
        checkpoint_dir=checkpoint_dir,
    )
    accepted: list[str] = []
    proposed: list[str] = []
    confirmed: list[str] = []
    rejected: list[str] = []
    deferred: list[str] = []
    errors: list[str] = []
    for index, requirement_id in enumerate(pending_ids, start=1):
        label = f"{requirement_id} [{index}/{len(pending_ids)}]"
        item = state["items"][requirement_id]
        try:
            segmentation, review, command, replacement, override, ap_artifacts = prepared[
                requirement_id
            ]
            if command == "y":
                reviewed = confirm_no_ambiguity(
                    review,
                    segmentation,
                    reviewer_id="batch-markdown-reviewer",
                    reviewer_source="human",
                    comment="Accepted after semantic fallback review.",
                )
                with _item_transaction_context(root, requirement_id) as transaction:
                    _save_reviewed(checkpoint_dir, segmentation, reviewed)
                    updated_item = dict(item)
                    _route_item_by_readiness(
                        updated_item,
                        reviewed,
                        segmentation,
                        checkpoint_dir=checkpoint_dir,
                    )
                    _commit_item_state(
                        root,
                        state,
                        requirement_id,
                        updated_item,
                        transaction=transaction,
                    )
                accepted.append(requirement_id)
            elif command == "r":
                if override is None:
                    raise ValueError("缺少已通过预检的replacement proposal。")
                with _item_transaction_context(root, requirement_id) as transaction:
                    for artifact in ap_artifacts:
                        save_ap_identity_artifact_checkpoint(
                            checkpoint_dir, artifact
                        )
                    save_human_semantic_override_checkpoint(
                        checkpoint_dir, override=override
                    )
                    updated_item = dict(item)
                    if override.status == "ap_rejected":
                        _set_pending_item(updated_item, review)
                        updated_item["pending_override_id"] = ""
                    else:
                        updated_item["status"] = _batch_status_for_override(override)
                        updated_item["pending_override_id"] = override.override_id
                    _commit_item_state(
                        root,
                        state,
                        requirement_id,
                        updated_item,
                        transaction=transaction,
                    )
                if override.status == "ap_rejected":
                    rejected.append(requirement_id)
                else:
                    proposed.append(requirement_id)
            elif command == "confirm":
                if override is None:
                    raise ValueError("缺少等待确认的replacement。")
                original_inventory, replacement_inventory, comparison = (
                    load_human_semantic_override_ap_artifacts(
                        checkpoint_dir, override
                    )
                )
                active = confirm_human_semantic_override(
                    override,
                    confirmed=True,
                    current_effective_source_hash=str(item["source_hash"]),
                    reviewer_id="batch-markdown-reviewer",
                    original_inventory=original_inventory,
                    replacement_inventory=replacement_inventory,
                    comparison=comparison,
                )
                promoted_inventory = None
                if (
                    original_inventory is not None
                    and replacement_inventory is not None
                    and comparison is not None
                ):
                    promoted_inventory = promote_replacement_ap_inventory(
                        replacement_inventory,
                        original_inventory=original_inventory,
                        comparison=comparison,
                        profile=_item_ap_profile(item),
                    )
                with _item_transaction_context(root, requirement_id) as transaction:
                    if promoted_inventory is not None:
                        save_ap_inventory_checkpoint(
                            checkpoint_dir,
                            inventory=promoted_inventory,
                            make_current=True,
                        )
                    save_human_semantic_override_checkpoint(
                        checkpoint_dir, override=active
                    )
                    updated_item = _activate_confirmed_replacement(
                        dict(item), active
                    )
                    if promoted_inventory is not None:
                        updated_item["effective_ap_inventory_id"] = (
                            promoted_inventory.inventory_id
                        )
                        updated_item["effective_ap_inventory_hash"] = (
                            promoted_inventory.artifact_hash
                        )
                    _commit_item_state(
                        root,
                        state,
                        requirement_id,
                        updated_item,
                        transaction=transaction,
                    )
                confirmed.append(requirement_id)
            elif command == "reject":
                if override is None:
                    raise ValueError("缺少等待确认的replacement。")
                original_inventory, replacement_inventory, comparison = (
                    load_human_semantic_override_ap_artifacts(
                        checkpoint_dir, override
                    )
                )
                cancelled = confirm_human_semantic_override(
                    override,
                    confirmed=False,
                    current_effective_source_hash=str(item["source_hash"]),
                    reviewer_id="batch-markdown-reviewer",
                    original_inventory=original_inventory,
                    replacement_inventory=replacement_inventory,
                    comparison=comparison,
                )
                with _item_transaction_context(root, requirement_id) as transaction:
                    save_human_semantic_override_checkpoint(
                        checkpoint_dir, override=cancelled
                    )
                    updated_item = dict(item)
                    _set_pending_item(updated_item, review)
                    updated_item["pending_override_id"] = ""
                    _commit_item_state(
                        root,
                        state,
                        requirement_id,
                        updated_item,
                        transaction=transaction,
                    )
                rejected.append(requirement_id)
            elif command == "c":
                deferred.append(requirement_id)
            if progress is not None:
                progress.ok(f"{label} action={command}")
        except (ValueError, PlainReadingValidationError, LLMClientError) as exc:
            errors.append(f"{requirement_id}: {exc}")
            _record_item_apply_error(item, exc)
            if progress is not None:
                progress.warn(f"{label} failed: {exc}")

    if proposed or rejected:
        state["semantic_fallback_round"] = int(
            state.get("semantic_fallback_round", 1)
        ) + 1
    state["semantic_fallback_review_complete"] = not bool(
        _pending_ids(state, "semantic_fallback")
    )
    state["no_ambiguity_review_complete"] = not bool(
        _pending_ids(state, "no_ambiguity")
    )
    state["ambiguity_review_complete"] = not bool(
        _pending_ids(state, "ambiguity")
    )
    _refresh_review_files(
        state,
        checkpoint_dir=checkpoint_dir,
        records_dir=records_dir,
        force=True,
    )
    _close_if_complete(state)
    _write_state(root, state)
    next_files = _pending_files(state)
    message_lines = [
        "[PARTIAL] Semantic-fallback review applied with item errors."
        if errors
        else "[OK] Semantic-fallback review applied.",
        f"Accepted current source: {len(accepted)}",
        f"Replacement proposals awaiting confirmation: {len(proposed)}",
        f"Confirmed replacements awaiting segmentation review: {len(confirmed)}",
        f"Rejected replacements: {len(rejected)}",
        f"Deferred: {len(deferred)}",
    ]
    if errors:
        message_lines.extend(("Errors:", *("  " + error for error in errors)))
    message_lines.extend(_next_file_lines(state, include_commands=True))
    return ReviewApplyOutcome(
        "\n".join(message_lines),
        next_files,
        bool(state.get("pre_ir_closed")),
        bool(errors),
    )


@_serialized_review_apply
def apply_semantic_override_ap_inventory_review(
    review_file: str,
    *,
    checkpoint_dir: str,
    records_dir: str | None = None,
    progress: TerminalProgress | None = None,
) -> ReviewApplyOutcome:
    """Apply a complete AP-inventory Markdown round without activating text."""

    root = Path(checkpoint_dir)
    state = _require_state(root)
    records_dir = str(state.get("records_dir") or records_dir or "review_records")
    review_path = _require_active_review_file(
        review_file, state, path_key="pending_ap_inventory_file"
    )
    text = review_path.read_text(encoding="utf-8")
    pending_ids = _pending_ids(state, "ap_inventory")
    _validate_review_document(
        text,
        review_type="semantic_override_ap_inventory",
        round_number=int(state.get("ap_inventory_round", 1)),
        requirement_ids=pending_ids,
    )
    prepared = _preflight_ap_inventory_review(
        _requirement_sections(text),
        pending_ids,
        state=state,
        checkpoint_dir=checkpoint_dir,
    )
    approved: list[str] = []
    rejected: list[str] = []
    deferred: list[str] = []
    for requirement_id in pending_ids:
        command, updated_override, artifacts = prepared[requirement_id]
        if command == "c":
            deferred.append(requirement_id)
            continue
        item = state["items"][requirement_id]
        with _item_transaction_context(root, requirement_id) as transaction:
            for artifact in artifacts:
                save_ap_identity_artifact_checkpoint(checkpoint_dir, artifact)
            if updated_override is None:
                raise PreIRStateConsistencyError("AP inventory transition is missing override")
            save_human_semantic_override_checkpoint(
                checkpoint_dir, override=updated_override
            )
            updated_item = dict(item)
            if command == "reject" or updated_override.status == "ap_rejected":
                segmentation, reading = _load_pending_artifacts(
                    checkpoint_dir, requirement_id, item
                )
                del segmentation
                _set_pending_item(updated_item, reading)
                updated_item["pending_override_id"] = ""
                rejected.append(requirement_id)
            else:
                updated_item["status"] = _batch_status_for_override(updated_override)
                updated_item["pending_override_id"] = updated_override.override_id
                approved.append(requirement_id)
            _commit_item_state(
                root,
                state,
                requirement_id,
                updated_item,
                transaction=transaction,
            )
    if approved or rejected:
        state["ap_inventory_round"] = int(state.get("ap_inventory_round", 1)) + 1
    _refresh_review_files(
        state, checkpoint_dir=checkpoint_dir, records_dir=records_dir, force=True
    )
    _close_if_complete(state)
    _write_state(root, state)
    next_files = _pending_files(state)
    return ReviewApplyOutcome(
        "\n".join(
            [
                "[OK] AP inventory review applied.",
                f"Approved inventories: {len(approved)}",
                f"Rejected or known mismatch: {len(rejected)}",
                f"Deferred: {len(deferred)}",
                *_next_file_lines(state, include_commands=True),
            ]
        ),
        next_files,
        bool(state.get("pre_ir_closed")),
        False,
    )


@_serialized_review_apply
def apply_semantic_override_ap_mapping_review(
    review_file: str,
    *,
    checkpoint_dir: str,
    records_dir: str | None = None,
    progress: TerminalProgress | None = None,
) -> ReviewApplyOutcome:
    """Apply AP identity mappings; replacement confirmation remains separate."""

    root = Path(checkpoint_dir)
    state = _require_state(root)
    records_dir = str(state.get("records_dir") or records_dir or "review_records")
    review_path = _require_active_review_file(
        review_file, state, path_key="pending_ap_mapping_file"
    )
    text = review_path.read_text(encoding="utf-8")
    pending_ids = _pending_ids(state, "ap_mapping")
    _validate_review_document(
        text,
        review_type="semantic_override_ap_mapping",
        round_number=int(state.get("ap_mapping_round", 1)),
        requirement_ids=pending_ids,
    )
    prepared = _preflight_ap_mapping_review(
        _requirement_sections(text),
        pending_ids,
        state=state,
        checkpoint_dir=checkpoint_dir,
    )
    approved: list[str] = []
    rejected: list[str] = []
    deferred: list[str] = []
    for requirement_id in pending_ids:
        command, updated_override, artifacts = prepared[requirement_id]
        if command == "c":
            deferred.append(requirement_id)
            continue
        item = state["items"][requirement_id]
        with _item_transaction_context(root, requirement_id) as transaction:
            for artifact in artifacts:
                save_ap_identity_artifact_checkpoint(checkpoint_dir, artifact)
            if updated_override is None:
                raise PreIRStateConsistencyError("AP mapping transition is missing override")
            save_human_semantic_override_checkpoint(
                checkpoint_dir, override=updated_override
            )
            updated_item = dict(item)
            if command == "reject":
                _segmentation, reading = _load_pending_artifacts(
                    checkpoint_dir, requirement_id, item
                )
                _set_pending_item(updated_item, reading)
                updated_item["pending_override_id"] = ""
                rejected.append(requirement_id)
            else:
                updated_item["status"] = _batch_status_for_override(updated_override)
                updated_item["pending_override_id"] = updated_override.override_id
                approved.append(requirement_id)
            _commit_item_state(
                root,
                state,
                requirement_id,
                updated_item,
                transaction=transaction,
            )
    if approved or rejected:
        state["ap_mapping_round"] = int(state.get("ap_mapping_round", 1)) + 1
    _refresh_review_files(
        state, checkpoint_dir=checkpoint_dir, records_dir=records_dir, force=True
    )
    _close_if_complete(state)
    _write_state(root, state)
    next_files = _pending_files(state)
    return ReviewApplyOutcome(
        "\n".join(
            [
                "[OK] AP mapping review applied.",
                f"Mappings approved; awaiting replacement confirmation: {len(approved)}",
                f"Rejected: {len(rejected)}",
                f"Deferred: {len(deferred)}",
                *_next_file_lines(state, include_commands=True),
            ]
        ),
        next_files,
        bool(state.get("pre_ir_closed")),
        False,
    )


@_serialized_review_apply
def apply_ambiguity_review(
    review_file: str,
    *,
    checkpoint_dir: str,
    records_dir: str | None = None,
    llm_mode: str | None = "api",
    llm_client: JSONLLMClient | None = None,
    approved_memory_file: str | None = None,
    max_segmentation_retries: int = 2,
    segmentation_prompt_version: str = "plain-segmentation-v1",
    plain_reading_prompt_version: str = "plain-reading-v5",
    semantic_fallback_enabled: bool = True,
    progress: TerminalProgress | None = None,
) -> ReviewApplyOutcome:
    root = Path(checkpoint_dir)
    if progress is not None:
        progress.stage("Loading ambiguity review")
        progress.info(f"Review file: {review_file}")
        progress.info(f"Checkpoint directory: {checkpoint_dir}")
    state = _require_state(root)
    fallback_view_existed = bool(state.get("pending_semantic_fallback_file"))
    records_dir = str(state.get("records_dir") or records_dir or "review_records")
    review_path = _require_active_review_file(
        review_file,
        state,
        path_key="pending_ambiguity_file",
    )
    text = review_path.read_text(encoding="utf-8")
    pending_ids = _pending_ids(state, "ambiguity")
    _validate_review_document(
        text,
        review_type="ambiguity_adjudication",
        round_number=int(state.get("ambiguity_round", 1)),
        requirement_ids=pending_ids,
    )
    sections = _requirement_sections(text)
    prepared = _preflight_ambiguity_review(
        sections,
        pending_ids,
        state=state,
        checkpoint_dir=checkpoint_dir,
    )
    if progress is not None:
        progress.ok("Review file is marked complete")
        progress.info(f"Pending ambiguity requirements: {len(pending_ids)}")
    errors: list[str] = []
    completed: list[str] = []
    regenerated: list[str] = []
    replacement_proposed: list[str] = []
    needs_llm = False
    if llm_client is None and needs_llm:
        if progress is not None:
            progress.task_start("Initializing LLM client for replacement review")
        try:
            llm_client = build_llm_client_from_env(llm_mode)
        finally:
            if progress is not None:
                progress.task_stop()
        if progress is not None:
            progress.ok("LLM client ready")

    for index, requirement_id in enumerate(pending_ids, start=1):
        label = f"{requirement_id} [{index}/{len(pending_ids)}]"
        item = state["items"][requirement_id]
        try:
            if progress is not None:
                progress.info(f"{label} validating metadata and checkpoints")
            segmentation, review, decisions, replacement, override, ap_artifacts = prepared[
                requirement_id
            ]
            if replacement:
                if override is None:
                    raise ValueError("缺少已通过预检的semantic override。")
                with _item_transaction_context(root, requirement_id) as transaction:
                    for artifact in ap_artifacts:
                        save_ap_identity_artifact_checkpoint(
                            checkpoint_dir, artifact
                        )
                    save_human_semantic_override_checkpoint(
                        checkpoint_dir, override=override
                    )
                    updated_item = dict(item)
                    if override.status == "ap_rejected":
                        _set_pending_item(updated_item, review)
                        updated_item["pending_override_id"] = ""
                    else:
                        updated_item["status"] = _batch_status_for_override(override)
                        updated_item["pending_override_id"] = override.override_id
                    _commit_item_state(
                        root,
                        state,
                        requirement_id,
                        updated_item,
                        transaction=transaction,
                    )
                if override.status != "ap_rejected":
                    replacement_proposed.append(requirement_id)
                if progress is not None:
                    progress.ok(f"{label} replacement awaiting explicit confirmation")
                continue
            if progress is not None:
                decision_summary = ", ".join(
                    f"{ambiguity_id}={decision}"
                    for ambiguity_id, decision in decisions.items()
                )
                progress.info(f"{label} decisions: {decision_summary}")
            translated = {
                ambiguity_id: (
                    "ambiguous"
                    if value == "u"
                    else "not_distinct"
                    if value == "n"
                    else value
                )
                for ambiguity_id, value in decisions.items()
            }
            reviewed = apply_plain_reading_decisions(
                review,
                segmentation,
                translated,
                reviewer_id="batch-markdown-reviewer",
                reviewer_source="human",
                review_basis="ordinary_language_reading",
            )
            if reviewed.overall_status not in {
                "plain_reading_reviewed",
                "ambiguous_unresolved",
                "pending_no_ambiguity_confirmation",
            }:
                raise ValueError(
                    "审核仍未关闭全部歧义轴: " + reviewed.overall_status
                )
            with _item_transaction_context(root, requirement_id) as transaction:
                _save_reviewed(checkpoint_dir, segmentation, reviewed)
                updated_item = dict(item)
                _route_item_by_readiness(
                    updated_item,
                    reviewed,
                    segmentation,
                    checkpoint_dir=checkpoint_dir,
                )
                _commit_item_state(
                    root,
                    state,
                    requirement_id,
                    updated_item,
                    transaction=transaction,
                )
            completed.append(requirement_id)
            if progress is not None:
                progress.ok(f"{label} ambiguity decisions applied")
        except (ValueError, PlainReadingValidationError, LLMClientError) as exc:
            errors.append(f"{requirement_id}: {exc}")
            _record_item_apply_error(item, exc)
            if progress is not None:
                progress.warn(f"{label} failed: {exc}")

    if progress is not None:
        progress.stage("Refreshing deferred Pre-IR review state")
    if replacement_proposed and fallback_view_existed:
        state["semantic_fallback_round"] = int(
            state.get("semantic_fallback_round", 1)
        ) + 1
    state["no_ambiguity_review_complete"] = not bool(
        _pending_ids(state, "no_ambiguity")
    )
    state["ambiguity_review_complete"] = not bool(
        _pending_ids(state, "ambiguity")
    )
    state["semantic_fallback_review_complete"] = not bool(
        _pending_ids(state, "semantic_fallback")
    )
    _refresh_review_files(
        state,
        checkpoint_dir=checkpoint_dir,
        records_dir=records_dir,
        force=True,
    )
    _close_if_complete(state)
    _write_state(root, state)
    next_files = _pending_files(state)
    if progress is not None:
        progress.ok("Deferred Pre-IR state and review files refreshed")
        progress.info(
            f"Ambiguity result: completed={len(completed)}, "
            f"replacement_proposed={len(replacement_proposed)}, "
            f"regenerated={len(regenerated)}"
        )
        for next_file in next_files:
            progress.info(f"Next review file: {next_file}")
    if errors:
        message = "\n".join(
            (
                "[PARTIAL] Ambiguity review applied with item errors.",
                f"Completed requirements: {len(completed)}",
                f"Regenerated requirements: {len(regenerated)}",
                f"Replacement proposals awaiting confirmation: {len(replacement_proposed)}",
                "Errors:",
                *("  " + error for error in errors),
                *_next_file_lines(state, include_commands=True),
            )
        )
    elif next_files:
        message = "\n".join(
            (
                "[WAIT] Ambiguity review requires another round.",
                f"Completed requirements: {len(completed)}",
                f"Regenerated requirements: {len(regenerated)}",
                f"Replacement proposals awaiting confirmation: {len(replacement_proposed)}",
                "",
                *_next_file_lines(state, include_commands=True),
                "No downstream IR or formula generation was started.",
            )
        )
    else:
        message = "\n".join(
            (
                "[OK] Ambiguity review completed.",
                f"Reviewed requirements: {len(completed)}",
                f"Replacement proposals awaiting confirmation: {len(replacement_proposed)}",
                "Pre-IR review is now complete.",
                "Run the main command again to continue downstream.",
            )
        )
    return ReviewApplyOutcome(
        message,
        next_files,
        bool(state.get("pre_ir_closed")),
        bool(errors),
    )


@_serialized_review_apply
def load_pre_ir_batch_state(checkpoint_dir: str) -> dict[str, Any]:
    return _load_state(Path(checkpoint_dir))


def _replacement_review_generation_contract(
    *,
    max_segmentation_retries: int,
    segmentation_prompt_version: str,
    plain_reading_prompt_version: str,
    semantic_fallback_enabled: bool,
) -> dict[str, Any]:
    return {
        "segmentation_prompt_version": segmentation_prompt_version,
        "plain_reading_prompt_version": plain_reading_prompt_version,
        "max_segmentation_retries": int(max_segmentation_retries),
        "semantic_fallback_enabled": bool(semantic_fallback_enabled),
        "validated_generation_contract_version": (
            VALIDATED_GENERATION_CONTRACT_VERSION
        ),
        "validated_generation_validator_version": (
            VALIDATED_GENERATION_VALIDATOR_VERSION
        ),
        "repair_policy_version": REPAIR_POLICY_VERSION,
        "patch_authorization_contract_version": (
            PATCH_AUTHORIZATION_CONTRACT_VERSION
        ),
    }


def _replacement_draft_matches_item(
    draft: ReplacementReviewDraft,
    item: Mapping[str, Any],
    *,
    generation_contract: Mapping[str, Any] | None = None,
) -> bool:
    return bool(
        draft.requirement_id == str(item.get("requirement_id", ""))
        and draft.original_requirement == str(item.get("original_requirement", ""))
        and draft.original_source_hash
        == source_hash(str(item.get("original_requirement", "")))
        and draft.effective_replacement
        == str(item.get("effective_requirement", ""))
        and draft.source_hash == str(item.get("source_hash", ""))
        and draft.active_override_id == str(item.get("active_override_id", ""))
        and (
            generation_contract is None
            or draft.generation_contract == dict(generation_contract)
        )
    )


def _replacement_review_draft(
    *,
    item: Mapping[str, Any],
    generation_contract: Mapping[str, Any],
    llm_client: JSONLLMClient,
    retriever: Any,
    checkpoint_dir: str,
    semantic_fallback_enabled: bool,
    plain_reading_prompt_version: str,
    max_segmentation_retries: int,
    segmentation_prompt_version: str,
    generated_segmentation: SurfaceSegmentation | None = None,
    proposed_segmentation: SurfaceSegmentation | None = None,
) -> ReplacementReviewDraft:
    requirement_id = str(item["requirement_id"])
    effective = str(item.get("effective_requirement", ""))
    diagnostics: list[dict[str, Any]] = []
    if generated_segmentation is None:
        generated_segmentation = SegmentationAgent(
            retriever, llm_client=llm_client
        ).generate(
            effective,
            requirement_id=requirement_id,
            max_retries=max_segmentation_retries,
            prompt_version=segmentation_prompt_version,
            experiment_mode="manual",
            data_split=str(item.get("data_split", "test")),
            generation_diagnostic_callback=(
                lambda value: diagnostics.append(dict(value))
            ),
        )
    if proposed_segmentation is None:
        proposed_segmentation = revise_segmentation(
            generated_segmentation,
            cut_offsets=[
                segment.end
                for segment in generated_segmentation.segments[:-1]
            ],
            decision="surface_reviewed",
            feedback="draft candidate; promotion requires Markdown application",
            reviewer_id="replacement-markdown-draft",
            label_source="human",
            review_basis="pending_batch_markdown_decision",
        )
    reading_agent = PlainReadingAgent(retriever, llm_client=llm_client)
    outcome = generate_plain_reading_outcome(
        reading_agent,
        effective,
        proposed_segmentation,
        semantic_fallback_enabled=semantic_fallback_enabled,
        requirement_id=requirement_id,
        mode="manual",
        prompt_version=plain_reading_prompt_version,
        data_split=str(item.get("data_split", "test")),
        ap_grounding_contract=_item_ap_grounding_contract(
            item,
            segmentation=proposed_segmentation,
            checkpoint_dir=checkpoint_dir,
        ),
        generation_diagnostic_callback=(
            lambda value: diagnostics.append(dict(value))
        ),
    )
    if outcome.diagnostics:
        diagnostics.append(dict(outcome.diagnostics))
    return ReplacementReviewDraft(
        requirement_id=requirement_id,
        original_requirement=str(item.get("original_requirement", "")),
        effective_replacement=effective,
        original_source_hash=source_hash(
            str(item.get("original_requirement", ""))
        ),
        source_hash=str(item.get("source_hash", "")),
        active_override_id=str(item.get("active_override_id", "")),
        generation_contract=dict(generation_contract),
        generated_segmentation=generated_segmentation,
        proposed_segmentation=proposed_segmentation,
        plain_reading_review=outcome.review,
        diagnostics=tuple(diagnostics),
        created_at=replacement_review_timestamp(),
    )


def _failed_replacement_review_draft(
    *,
    item: Mapping[str, Any],
    generation_contract: Mapping[str, Any],
    exc: BaseException,
) -> ReplacementReviewDraft:
    diagnostics = getattr(exc, "diagnostics", {})
    details = dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
    details.update(
        {
            "stage": "replacement_review_draft_generation",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    )
    return ReplacementReviewDraft(
        requirement_id=str(item["requirement_id"]),
        original_requirement=str(item.get("original_requirement", "")),
        effective_replacement=str(item.get("effective_requirement", "")),
        original_source_hash=source_hash(
            str(item.get("original_requirement", ""))
        ),
        source_hash=str(item.get("source_hash", "")),
        active_override_id=str(item.get("active_override_id", "")),
        generation_contract=dict(generation_contract),
        diagnostics=(details,),
        created_at=replacement_review_timestamp(),
    )


def _replacement_review_file_path(
    state: Mapping[str, Any], *, round_number: int
) -> Path:
    namespace = Path(
        str(state.get("review_namespace_dir") or "")
        or str(
            _review_namespace_root(
                str(state.get("records_dir") or "review_records"),
                str(state.get("checkpoint_dir") or ".traj_ltlf_checkpoints"),
            )
        )
    )
    return namespace / f"pre_ir_replacement_round_{round_number}.md"


def _active_replacement_review_is_current(
    state: Mapping[str, Any],
    *,
    checkpoint_dir: str,
    pending_ids: Sequence[str],
) -> bool:
    active = str(state.get("pending_replacement_review_file", ""))
    snapshot = str(state.get("replacement_review_snapshot_id", ""))
    if not active or not snapshot or not Path(active).is_file():
        return False
    drafts: list[ReplacementReviewDraft] = []
    try:
        for requirement_id in pending_ids:
            draft = load_replacement_review_draft(
                checkpoint_dir, requirement_id
            )
            item = state.get("items", {}).get(requirement_id, {})
            if draft is None or not _replacement_draft_matches_item(draft, item):
                return False
            drafts.append(draft)
    except (OSError, TypeError, ValueError):
        return False
    expected = replacement_review_snapshot_id(
        checkpoint_fingerprint=_checkpoint_fingerprint(checkpoint_dir),
        round_number=int(state.get("replacement_review_round", 1)),
        drafts=drafts,
    )
    return snapshot == expected


@_serialized_review_apply
def export_pending_replacement_review(
    *,
    checkpoint_dir: str,
    records_dir: str | None = None,
    llm_mode: str | None = "api",
    llm_client: JSONLLMClient | None = None,
    approved_memory_file: str | None = None,
    max_segmentation_retries: int = 2,
    segmentation_prompt_version: str = "plain-segmentation-v1",
    plain_reading_prompt_version: str = "plain-reading-v5",
    semantic_fallback_enabled: bool = True,
    refresh: bool = False,
    progress: TerminalProgress | None = None,
) -> ReplacementReviewExportOutcome:
    """Generate or reuse draft-only candidates and one editable Markdown view."""

    root = Path(checkpoint_dir)
    state = _require_state(root)
    pending_ids = _pending_replacement_review_ids_from_state(state)
    if not pending_ids:
        state["pending_replacement_review_file"] = ""
        state["replacement_review_snapshot_id"] = ""
        _write_state(root, state)
        return ReplacementReviewExportOutcome(
            "[OK] No activated replacements are awaiting Pre-IR review."
        )
    if not refresh and _active_replacement_review_is_current(
        state, checkpoint_dir=checkpoint_dir, pending_ids=pending_ids
    ):
        active = str(state["pending_replacement_review_file"])
        return ReplacementReviewExportOutcome(
            "\n".join(
                (
                    "[OK] Existing replacement review round is still current.",
                    "Review file:",
                    "  " + active,
                    "No draft or canonical artifact was regenerated.",
                )
            ),
            active,
            pending_ids,
            (),
            pending_ids,
        )

    records_dir = str(state.get("records_dir") or records_dir or "review_records")
    state["records_dir"] = records_dir
    generation_contract = _replacement_review_generation_contract(
        max_segmentation_retries=max_segmentation_retries,
        segmentation_prompt_version=segmentation_prompt_version,
        plain_reading_prompt_version=plain_reading_prompt_version,
        semantic_fallback_enabled=semantic_fallback_enabled,
    )
    retriever = build_persistent_retriever(approved_memory_file)
    drafts: list[ReplacementReviewDraft] = []
    failed: list[str] = []
    reused: list[str] = []
    for index, requirement_id in enumerate(pending_ids, start=1):
        item = state["items"][requirement_id]
        if progress is not None:
            progress.info(
                f"{requirement_id} replacement draft {index}/{len(pending_ids)}"
            )
        existing: ReplacementReviewDraft | None = None
        if not refresh:
            try:
                existing = load_replacement_review_draft(
                    checkpoint_dir, requirement_id
                )
            except (OSError, TypeError, ValueError):
                existing = None
        if existing is not None and _replacement_draft_matches_item(
            existing, item, generation_contract=generation_contract
        ):
            drafts.append(existing)
            reused.append(requirement_id)
            if not existing.generation_succeeded:
                failed.append(requirement_id)
            continue
        if llm_client is None:
            llm_client = build_llm_client_from_env(llm_mode)
        if llm_client is None:
            raise ValueError("replacement review 导出需要 --llm api。")
        try:
            generated = _replacement_review_draft(
                item=item,
                generation_contract=generation_contract,
                llm_client=llm_client,
                retriever=retriever,
                checkpoint_dir=checkpoint_dir,
                semantic_fallback_enabled=semantic_fallback_enabled,
                plain_reading_prompt_version=plain_reading_prompt_version,
                max_segmentation_retries=max_segmentation_retries,
                segmentation_prompt_version=segmentation_prompt_version,
            )
        except Exception as exc:
            generated = _failed_replacement_review_draft(
                item=item,
                generation_contract=generation_contract,
                exc=exc,
            )
            failed.append(requirement_id)
            if progress is not None:
                progress.warn(f"{requirement_id} draft generation failed: {exc}")
        saved = save_replacement_review_draft(checkpoint_dir, generated)
        drafts.append(saved)

    had_active = bool(str(state.get("pending_replacement_review_file", "")))
    round_number = int(state.get("replacement_review_round", 1))
    if had_active:
        round_number += 1
    snapshot = replacement_review_snapshot_id(
        checkpoint_fingerprint=_checkpoint_fingerprint(checkpoint_dir),
        round_number=round_number,
        drafts=drafts,
    )
    review_path = _replacement_review_file_path(
        state, round_number=round_number
    )
    _atomic_write_text(
        review_path,
        format_replacement_review_markdown(
            checkpoint_fingerprint=_checkpoint_fingerprint(checkpoint_dir),
            round_number=round_number,
            snapshot_id=snapshot,
            drafts=drafts,
        ),
    )
    state.update(
        pending_replacement_review_file=str(review_path),
        replacement_review_round=round_number,
        replacement_review_snapshot_id=snapshot,
        updated_at=_timestamp(),
    )
    _write_state(root, state)
    lines = [
        "[PARTIAL] Replacement review draft exported with item errors."
        if failed
        else "[OK] Replacement review Markdown exported.",
        f"Pending replacements: {len(pending_ids)}",
        f"Reused drafts: {len(reused)}",
        f"Generation failures: {len(failed)}",
        "Review file:",
        "  " + str(review_path),
        "No item was approved and no downstream processing was started.",
    ]
    return ReplacementReviewExportOutcome(
        "\n".join(lines),
        str(review_path),
        pending_ids,
        tuple(failed),
        tuple(reused),
    )


def _validate_replacement_active_override(
    checkpoint_dir: str,
    requirement_id: str,
    item: Mapping[str, Any],
) -> None:
    override = load_human_semantic_override_checkpoint(
        checkpoint_dir, requirement_id, active_only=True
    )
    if (
        override is None
        or override.override_id != str(item.get("active_override_id", ""))
        or override.replacement_text != str(item.get("effective_requirement", ""))
        or override.new_effective_source_hash != str(item.get("source_hash", ""))
    ):
        raise ValueError("active replacement override 已过期或不匹配。")


def _replacement_decision_segmentation(
    draft: ReplacementReviewDraft,
    decision: ReplacementReviewDecision,
) -> tuple[SurfaceSegmentation, bool]:
    assert draft.proposed_segmentation is not None
    proposed = draft.proposed_segmentation
    current_offsets = tuple(segment.end for segment in proposed.segments[:-1])
    changed = tuple(decision.cut_offsets) != current_offsets
    if changed:
        reviewed = revise_segmentation(
            proposed,
            cut_offsets=decision.cut_offsets,
            decision=(
                "ambiguity_flagged"
                if decision.ambiguity_flagged
                else "surface_reviewed"
            ),
            feedback=decision.comment,
            reviewer_id="batch-replacement-markdown-reviewer",
            label_source="human",
            review_basis="surface_only",
        )
        return reviewed, True
    desired_status = (
        "ambiguity_flagged"
        if decision.ambiguity_flagged
        else "surface_reviewed"
    )
    return replace(
        proposed,
        status=desired_status,
        label_source="human",
        review_basis="surface_only",
        persistent_rag_allowed=(
            proposed.data_split == "train"
            and desired_status in {"surface_reviewed", "ambiguity_flagged"}
        ),
    ), False


def _apply_replacement_plain_decision(
    draft: ReplacementReviewDraft,
    segmentation: SurfaceSegmentation,
    decision: ReplacementReviewDecision,
) -> PlainReadingReview:
    review = draft.plain_reading_review
    if review is None:
        raise ValueError("replacement Plain Reading draft is missing")
    if draft.revision_candidate_review is not None:
        if decision.revision_candidate_action == "accept":
            return draft.revision_candidate_review
        if decision.revision_candidate_action == "reject":
            return review
        raise ValueError("replacement revision candidate decision is incomplete")
    if review.overall_status == "options_revision_requested":
        action_map = decision.revision_action_map
        active_requests = [
            item
            for item in review.revision_requests
            if item.request_status in {"submitted", "suggested", "confirmed"}
        ]
        result = review
        for request in active_requests:
            action, basis = action_map.get(request.request_id, ("", ""))
            if action in {"S", "I"}:
                raise ValueError(
                    "S/I revision classification requires candidate generation"
                )
            if action == "B":
                result = close_plain_reading_revision_request(
                    result,
                    segmentation,
                    request,
                    action="return_to_review",
                    reviewer_id="batch-replacement-markdown-reviewer",
                )
                continue
            if action == "C":
                clarification = replace(
                    request,
                    classification="out_of_source",
                    classification_basis=basis,
                )
                result = close_plain_reading_revision_request(
                    result,
                    segmentation,
                    clarification,
                    action="needs_source_clarification",
                    reviewer_id="batch-replacement-markdown-reviewer",
                )
                continue
            classification = {
                "E": "equivalent_to_existing",
                "O": "out_of_source",
            }.get(action)
            if classification is None:
                raise ValueError(
                    f"unsupported replacement revision action: {action}"
                )
            confirmed = confirm_plain_reading_revision_classification(
                request,
                classification=classification,
                classification_basis=basis,
                confirmed_by="batch-replacement-markdown-reviewer",
            )
            if action == "O":
                result = close_plain_reading_revision_request(
                    result,
                    segmentation,
                    confirmed,
                    action="return_to_review",
                    reviewer_id="batch-replacement-markdown-reviewer",
                )
            else:
                result = apply_plain_reading_revision(
                    result,
                    segmentation,
                    confirmed,
                    reviewer_id="batch-replacement-markdown-reviewer",
                )
        return result
    if review.ambiguities:
        revision_requests = {}
        for ambiguity in review.ambiguities:
            proposed = decision.revisions.get(ambiguity.ambiguity_id, "")
            if not proposed:
                continue
            revision_requests[ambiguity.ambiguity_id] = (
                create_plain_reading_revision_request(
                    review,
                    segmentation,
                    ambiguity_id=ambiguity.ambiguity_id,
                    proposed_interpretation=proposed,
                    reason="Markdown reviewer requested revised Plain Reading options.",
                    source_segment_ids=ambiguity.source_segment_ids,
                    reviewer_id="batch-replacement-markdown-reviewer",
                )
            )
        translated = {
            ambiguity_id: (
                "not_distinct"
                if value == "N"
                else "ambiguous"
                if value == "U"
                else "revise"
                if value == "R"
                else value
            )
            for ambiguity_id, value in decision.decisions.items()
        }
        return apply_plain_reading_decisions(
            review,
            segmentation,
            translated,
            reviewer_id="batch-replacement-markdown-reviewer",
            reviewer_source="human",
            review_basis="ordinary_language_reading",
            comments={key: decision.comment for key in translated},
            revision_requests=revision_requests,
        )
    if decision.no_ambiguity_action == "accept":
        return confirm_no_ambiguity(
            review,
            segmentation,
            reviewer_id="batch-replacement-markdown-reviewer",
            reviewer_source="human",
            comment=decision.comment,
        )
    if decision.no_ambiguity_action == "request":
        return request_ambiguity_generation(
            review,
            segmentation,
            suspected_segment_ids=(),
            user_comment=decision.comment,
            reviewer_id="batch-replacement-markdown-reviewer",
            reviewer_source="human",
        )
    raise ValueError("replacement Plain Reading decision is incomplete")


def _replacement_draft_with_review(
    draft: ReplacementReviewDraft,
    *,
    segmentation: SurfaceSegmentation,
    review: PlainReadingReview,
    diagnostics: Sequence[Mapping[str, Any]] = (),
    revision_candidate_review: PlainReadingReview | None = None,
    revision_candidate_request_id: str = "",
    revision_candidate_warnings: Sequence[str] = (),
) -> ReplacementReviewDraft:
    return ReplacementReviewDraft(
        requirement_id=draft.requirement_id,
        original_requirement=draft.original_requirement,
        effective_replacement=draft.effective_replacement,
        original_source_hash=draft.original_source_hash,
        source_hash=draft.source_hash,
        active_override_id=draft.active_override_id,
        generation_contract=dict(draft.generation_contract),
        generated_segmentation=draft.generated_segmentation,
        proposed_segmentation=segmentation,
        plain_reading_review=review,
        revision_candidate_review=revision_candidate_review,
        revision_candidate_request_id=revision_candidate_request_id,
        revision_candidate_warnings=tuple(revision_candidate_warnings),
        diagnostics=tuple(dict(item) for item in diagnostics),
        created_at=replacement_review_timestamp(),
    )


def _preflight_replacement_review_decision(
    draft: ReplacementReviewDraft,
    decision: ReplacementReviewDecision,
) -> None:
    if decision.defer:
        return
    segmentation, changed = _replacement_decision_segmentation(draft, decision)
    if changed:
        return
    generation_action = _replacement_revision_generation_action(draft, decision)
    if generation_action is not None:
        request, classification, basis = generation_action
        confirm_plain_reading_revision_classification(
            request,
            classification=classification,
            classification_basis=basis,
            confirmed_by="batch-replacement-markdown-reviewer",
        )
        return
    _apply_replacement_plain_decision(draft, segmentation, decision)


def _replacement_revision_generation_action(
    draft: ReplacementReviewDraft,
    decision: ReplacementReviewDecision,
) -> tuple[Any, str, str] | None:
    review = draft.plain_reading_review
    if review is None or review.overall_status != "options_revision_requested":
        return None
    actions = decision.revision_action_map
    for request in review.revision_requests:
        action, basis = actions.get(request.request_id, ("", ""))
        classification = {
            "S": "same_axis_revision",
            "I": "new_independent_axis",
        }.get(action)
        if classification is not None:
            return request, classification, basis
    return None


def _generate_replacement_revision_candidate(
    draft: ReplacementReviewDraft,
    segmentation: SurfaceSegmentation,
    decision: ReplacementReviewDecision,
    *,
    llm_client: JSONLLMClient,
    retriever: Any,
) -> ReplacementReviewDraft:
    generation_action = _replacement_revision_generation_action(draft, decision)
    if generation_action is None:
        raise ValueError("replacement revision candidate action is missing")
    request, classification, basis = generation_action
    review = draft.plain_reading_review
    if review is None:
        raise ValueError("replacement revision review is missing")
    ambiguity = next(
        (
            item
            for item in review.ambiguities
            if item.ambiguity_id == request.ambiguity_id
        ),
        None,
    )
    if ambiguity is None:
        raise ValueError("replacement revision target ambiguity is stale")
    confirmed = confirm_plain_reading_revision_classification(
        request,
        classification=classification,
        classification_basis=basis,
        confirmed_by="batch-replacement-markdown-reviewer",
    )
    axes = PlainReadingRevisionAgent(
        retriever, llm_client=llm_client
    ).rebuild_axes(
        source_text=draft.effective_replacement,
        segmentation=segmentation,
        ambiguity=ambiguity,
        request=request,
        human_classification=classification,
    )
    if not axes:
        raise PlainReadingValidationError(
            "revision generator did not return a binary candidate axis"
        )
    warnings = validate_rebuilt_plain_reading_axes(
        draft.effective_replacement, segmentation, axes
    )
    candidate = apply_plain_reading_revision(
        review,
        segmentation,
        confirmed,
        replacement_axes=axes,
        reviewer_id="batch-replacement-markdown-reviewer",
        acknowledge_rebuilt_source_warnings=True,
    )
    return _replacement_draft_with_review(
        draft,
        segmentation=segmentation,
        review=review,
        revision_candidate_review=candidate,
        revision_candidate_request_id=request.request_id,
        revision_candidate_warnings=warnings,
    )


def _archive_replacement_draft(
    checkpoint_dir: str,
    draft: ReplacementReviewDraft,
) -> None:
    source = replacement_draft_path(checkpoint_dir, draft.requirement_id)
    if not source.is_file():
        return
    target = (
        Path(checkpoint_dir)
        / REPLACEMENT_DRAFTS_DIR
        / "applied"
        / draft.requirement_id
        / f"{draft.draft_fingerprint}.json"
    )
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(target)


@_serialized_review_apply
def apply_pending_replacement_review(
    review_file: str,
    *,
    checkpoint_dir: str,
    records_dir: str | None = None,
    llm_mode: str | None = "api",
    llm_client: JSONLLMClient | None = None,
    approved_memory_file: str | None = None,
    max_segmentation_retries: int = 2,
    segmentation_prompt_version: str = "plain-segmentation-v1",
    plain_reading_prompt_version: str = "plain-reading-v5",
    semantic_fallback_enabled: bool = True,
    progress: TerminalProgress | None = None,
) -> ReviewApplyOutcome:
    """Preflight the whole file, then promote each item transactionally."""

    root = Path(checkpoint_dir)
    state = _require_state(root)
    records_dir = str(state.get("records_dir") or records_dir or "review_records")
    review_path = _require_active_review_file(
        review_file,
        state,
        path_key="pending_replacement_review_file",
    )
    pending_ids = _pending_replacement_review_ids_from_state(state)
    if not pending_ids:
        raise ReviewFileInputError("当前没有待处理的 activated replacement。")
    drafts: list[ReplacementReviewDraft] = []
    preflight_errors: list[str] = []
    for requirement_id in pending_ids:
        try:
            draft = load_replacement_review_draft(
                checkpoint_dir, requirement_id
            )
            item = state["items"][requirement_id]
            if draft is None or not _replacement_draft_matches_item(draft, item):
                raise ValueError("draft checkpoint 缺失、过期或与 active override 不匹配。")
            _validate_replacement_active_override(
                checkpoint_dir, requirement_id, item
            )
            drafts.append(draft)
        except (OSError, TypeError, ValueError) as exc:
            preflight_errors.append(f"{requirement_id}: {exc}")
    if preflight_errors:
        raise ReviewFileInputError(
            "replacement 审核文件预检失败：\n" + "\n".join(preflight_errors)
        )
    round_number = int(state.get("replacement_review_round", 1))
    snapshot = replacement_review_snapshot_id(
        checkpoint_fingerprint=_checkpoint_fingerprint(checkpoint_dir),
        round_number=round_number,
        drafts=drafts,
    )
    if snapshot != str(state.get("replacement_review_snapshot_id", "")):
        raise ReviewFileInputError(
            "replacement review snapshot 已过期；请重新导出，未执行任何写操作。"
        )
    try:
        decisions = parse_replacement_review_markdown(
            review_path.read_text(encoding="utf-8"),
            checkpoint_fingerprint=_checkpoint_fingerprint(checkpoint_dir),
            round_number=round_number,
            snapshot_id=snapshot,
            drafts=drafts,
        )
    except ReplacementReviewInputError as exc:
        raise ReviewFileInputError(str(exc)) from exc
    for draft in drafts:
        try:
            _preflight_replacement_review_decision(
                draft, decisions[draft.requirement_id]
            )
        except (TypeError, ValueError, PlainReadingValidationError) as exc:
            preflight_errors.append(f"{draft.requirement_id}: {exc}")
    if preflight_errors:
        raise ReviewFileInputError(
            "replacement 审核文件领域预检失败：\n"
            + "\n".join(preflight_errors)
        )

    retriever = build_persistent_retriever(approved_memory_file)
    completed: list[str] = []
    deferred: list[str] = []
    continued: list[str] = []
    errors: list[str] = []
    drafts_by_id = {draft.requirement_id: draft for draft in drafts}
    for index, requirement_id in enumerate(pending_ids, start=1):
        decision = decisions[requirement_id]
        item = state["items"][requirement_id]
        draft = drafts_by_id[requirement_id]
        if decision.defer:
            deferred.append(requirement_id)
            continue
        if progress is not None:
            progress.info(
                f"{requirement_id} replacement apply {index}/{len(pending_ids)}"
            )
        try:
            segmentation, changed_surface = _replacement_decision_segmentation(
                draft, decision
            )
            if changed_surface:
                if llm_client is None:
                    llm_client = build_llm_client_from_env(llm_mode)
                if llm_client is None:
                    raise ValueError("重切分后的下一轮 Plain Reading 需要 --llm api。")
                next_draft = _replacement_review_draft(
                    item=item,
                    generation_contract=draft.generation_contract,
                    llm_client=llm_client,
                    retriever=retriever,
                    checkpoint_dir=checkpoint_dir,
                    semantic_fallback_enabled=semantic_fallback_enabled,
                    plain_reading_prompt_version=plain_reading_prompt_version,
                    max_segmentation_retries=max_segmentation_retries,
                    segmentation_prompt_version=segmentation_prompt_version,
                    generated_segmentation=draft.generated_segmentation,
                    proposed_segmentation=segmentation,
                )
                with _item_transaction_context(root, requirement_id):
                    save_replacement_review_draft(checkpoint_dir, next_draft)
                continued.append(requirement_id)
                continue

            if _replacement_revision_generation_action(
                draft, decision
            ) is not None:
                if llm_client is None:
                    llm_client = build_llm_client_from_env(llm_mode)
                if llm_client is None:
                    raise ValueError("S/I revision candidate generation 需要 --llm api。")
                next_draft = _generate_replacement_revision_candidate(
                    draft,
                    segmentation,
                    decision,
                    llm_client=llm_client,
                    retriever=retriever,
                )
                with _item_transaction_context(root, requirement_id):
                    save_replacement_review_draft(checkpoint_dir, next_draft)
                continued.append(requirement_id)
                continue

            reviewed = _apply_replacement_plain_decision(
                draft, segmentation, decision
            )
            if reviewed.overall_status == "user_requested_ambiguity_generation":
                if llm_client is None:
                    llm_client = build_llm_client_from_env(llm_mode)
                if llm_client is None:
                    raise ValueError("请求下一轮歧义轴需要 --llm api。")
                generated_outcome = generate_plain_reading_outcome(
                    PlainReadingAgent(retriever, llm_client=llm_client),
                    draft.effective_replacement,
                    segmentation,
                    semantic_fallback_enabled=semantic_fallback_enabled,
                    requirement_id=requirement_id,
                    mode="manual",
                    prompt_version=plain_reading_prompt_version,
                    data_split=str(item.get("data_split", "test")),
                    ambiguity_generation_request=reviewed,
                    ap_grounding_contract=_item_ap_grounding_contract(
                        item,
                        segmentation=segmentation,
                        checkpoint_dir=checkpoint_dir,
                    ),
                )
                reviewed = generated_outcome.review

            readiness = assess_plain_reading_ir_readiness(reviewed, segmentation)
            if readiness.disposition == "human_action_required":
                next_draft = _replacement_draft_with_review(
                    draft,
                    segmentation=segmentation,
                    review=reviewed,
                )
                with _item_transaction_context(root, requirement_id):
                    save_replacement_review_draft(checkpoint_dir, next_draft)
                continued.append(requirement_id)
                continue
            if readiness.disposition == "invalid":
                raise PlainReadingValidationError(
                    "；".join(readiness.validation_errors),
                    errors=readiness.validation_errors,
                )
            assert draft.generated_segmentation is not None
            assert draft.plain_reading_review is not None
            comparison = compare_segmentations(
                draft.generated_segmentation, segmentation
            )
            with _item_transaction_context(root, requirement_id) as transaction:
                save_segmentation_generated_checkpoint(
                    checkpoint_dir,
                    requirement_id=requirement_id,
                    segmentation=draft.generated_segmentation,
                )
                save_segmentation_reviewed_checkpoint(
                    checkpoint_dir,
                    requirement_id=requirement_id,
                    segmentation=segmentation,
                )
                _write_comparison(root, comparison)
                save_plain_reading_generated_checkpoint(
                    checkpoint_dir,
                    requirement_id=requirement_id,
                    plain_reading_review=draft.plain_reading_review,
                    segmentation_schema_version=segmentation.schema_version,
                )
                _save_reviewed(checkpoint_dir, segmentation, reviewed)
                updated_item = dict(item)
                updated_item.update(
                    segmentation_id=segmentation.segmentation_id,
                    llm_segmentation_id=(
                        draft.generated_segmentation.segmentation_id
                    ),
                    reading_review_id=reviewed.reading_review_id,
                    plain_status=reviewed.overall_status,
                    comparison=comparison.to_dict(),
                    error="",
                )
                updated_item.pop(
                    "replacement_segmentation_review_required", None
                )
                _clear_item_errors(updated_item)
                _route_item_by_readiness(
                    updated_item,
                    reviewed,
                    segmentation,
                    checkpoint_dir=checkpoint_dir,
                )
                _commit_item_state(
                    root,
                    state,
                    requirement_id,
                    updated_item,
                    transaction=transaction,
                )
            try:
                _archive_replacement_draft(checkpoint_dir, draft)
            except OSError as exc:
                # The canonical promotion is already committed.  The active
                # draft remains in place, so an optional audit-copy failure is
                # diagnostic only and must not undo or misreport the item.
                if progress is not None:
                    progress.warn(
                        f"{requirement_id} draft archive copy failed: {exc}"
                    )
            completed.append(requirement_id)
        except Exception as exc:
            errors.append(f"{requirement_id}: {exc}")
            # A transaction-finalization failure may already have restored the
            # on-disk state.  Rebase the in-memory mapping before recording the
            # non-canonical diagnostic so a later global state write cannot
            # accidentally resurrect rolled-back artifact pointers.
            recovered_state = _load_state(root)
            state.clear()
            state.update(recovered_state)
            recovered_item = state["items"][requirement_id]
            _record_item_apply_error(recovered_item, exc)
            if progress is not None:
                progress.warn(f"{requirement_id} apply failed: {exc}")

    remaining_ids = _pending_replacement_review_ids_from_state(state)
    if remaining_ids:
        next_round = round_number + 1
        remaining_drafts: list[ReplacementReviewDraft] = []
        for requirement_id in remaining_ids:
            draft = load_replacement_review_draft(
                checkpoint_dir, requirement_id
            )
            if draft is None:
                raise PreIRStateConsistencyError(
                    f"{requirement_id}: next replacement review draft is missing"
                )
            remaining_drafts.append(draft)
        next_snapshot = replacement_review_snapshot_id(
            checkpoint_fingerprint=_checkpoint_fingerprint(checkpoint_dir),
            round_number=next_round,
            drafts=remaining_drafts,
        )
        next_path = _replacement_review_file_path(
            state, round_number=next_round
        )
        _atomic_write_text(
            next_path,
            format_replacement_review_markdown(
                checkpoint_fingerprint=_checkpoint_fingerprint(checkpoint_dir),
                round_number=next_round,
                snapshot_id=next_snapshot,
                drafts=remaining_drafts,
            ),
        )
        state.update(
            pending_replacement_review_file=str(next_path),
            replacement_review_round=next_round,
            replacement_review_snapshot_id=next_snapshot,
        )
    else:
        state["pending_replacement_review_file"] = ""
        state["replacement_review_snapshot_id"] = ""
    _refresh_review_files(
        state,
        checkpoint_dir=checkpoint_dir,
        records_dir=records_dir,
        force=True,
    )
    _close_if_complete(state)
    state["updated_at"] = _timestamp()
    _write_state(root, state)
    next_files = _pending_files(state)
    lines = [
        "[PARTIAL] Replacement review applied with item errors."
        if errors
        else "[WAIT] Replacement review requires another round."
        if remaining_ids
        else "[OK] Replacement review applied.",
        f"Completed requirements: {len(completed)}",
        f"Continued in next replacement round: {len(continued)}",
        f"Deferred requirements: {len(deferred)}",
        f"Item errors: {len(errors)}",
    ]
    if errors:
        lines.extend(("Errors:", *("  " + value for value in errors)))
    lines.extend(_next_file_lines(state, include_commands=True))
    lines.append("No downstream processing was started.")
    return ReviewApplyOutcome(
        "\n".join(lines),
        next_files,
        bool(state.get("pre_ir_closed")),
        bool(errors),
    )


@_serialized_review_apply
def pending_replacement_review_ids(*, checkpoint_dir: str) -> tuple[str, ...]:
    """Return the canonical, deterministic replacement Pre-IR work queue."""

    state = _require_state(Path(checkpoint_dir))
    return _pending_replacement_review_ids_from_state(state)


def review_pending_replacement_pre_ir(
    *,
    checkpoint_dir: str,
    records_dir: str | None = None,
    llm_mode: str | None = "api",
    llm_client: JSONLLMClient | None = None,
    approved_memory_file: str | None = None,
    max_segmentation_retries: int = 2,
    max_surface_review_rounds: int = 3,
    max_plain_reading_rounds: int = 2,
    max_plain_reading_revision_rounds: int = 2,
    segmentation_prompt_version: str = "plain-segmentation-v1",
    plain_reading_prompt_version: str = "plain-reading-v5",
    semantic_fallback_enabled: bool = True,
    progress: TerminalProgress | None = None,
) -> ReplacementReviewQueueOutcome:
    """Review every activated replacement serially using per-item transactions."""

    initial_ids = pending_replacement_review_ids(checkpoint_dir=checkpoint_dir)
    if not initial_ids:
        state = load_pre_ir_batch_state(checkpoint_dir)
        return _replacement_review_queue_outcome(
            state,
            initial_ids=(),
            attempted_ids=(),
            completed_ids=(),
            halt_reason="empty",
        )

    if llm_client is None:
        llm_client = build_llm_client_from_env(llm_mode)
    if llm_client is None:
        raise ValueError("连续replacement Pre-IR审核需要 --llm api。")

    attempted: list[str] = []
    completed: list[str] = []
    halt_reason = ""
    had_errors = False
    while True:
        pending_ids = pending_replacement_review_ids(
            checkpoint_dir=checkpoint_dir
        )
        if not pending_ids:
            break
        requirement_id = pending_ids[0]
        before_state = load_pre_ir_batch_state(checkpoint_dir)
        before_item = before_state.get("items", {}).get(requirement_id, {})
        before_fingerprint = _replacement_queue_item_fingerprint(before_item)
        attempted.append(requirement_id)
        if progress is not None:
            progress.stage(
                f"Replacement Pre-IR review: {requirement_id}",
                step=min(len(completed) + 1, len(initial_ids)),
                total=len(initial_ids),
            )
        try:
            item_outcome = reopen_pre_ir_item(
                requirement_id,
                checkpoint_dir=checkpoint_dir,
                records_dir=records_dir,
                llm_mode=llm_mode,
                llm_client=llm_client,
                approved_memory_file=approved_memory_file,
                max_segmentation_retries=max_segmentation_retries,
                max_surface_review_rounds=max_surface_review_rounds,
                max_plain_reading_rounds=max_plain_reading_rounds,
                max_plain_reading_revision_rounds=(
                    max_plain_reading_revision_rounds
                ),
                segmentation_prompt_version=segmentation_prompt_version,
                plain_reading_prompt_version=plain_reading_prompt_version,
                semantic_fallback_enabled=semantic_fallback_enabled,
                _required_status="awaiting_replacement_segmentation_review",
            )
        except KeyboardInterrupt:
            halt_reason = "interrupted"
            if progress is not None:
                progress.wait(
                    f"Replacement Pre-IR review interrupted at {requirement_id}"
                )
            break

        after_state = load_pre_ir_batch_state(checkpoint_dir)
        after_item = after_state.get("items", {}).get(requirement_id, {})
        after_status = str(after_item.get("status", ""))
        if after_status == "ready_for_single_ir":
            if requirement_id not in completed:
                completed.append(requirement_id)
            if progress is not None:
                progress.ok(f"{requirement_id} approved")
            continue
        if item_outcome.had_errors:
            halt_reason = "item_error"
            had_errors = True
            if progress is not None:
                progress.error(f"{requirement_id} review did not complete")
            break

        after_fingerprint = _replacement_queue_item_fingerprint(after_item)
        if (
            after_status == "awaiting_replacement_segmentation_review"
            and after_fingerprint != before_fingerprint
        ):
            if progress is not None:
                progress.wait(
                    f"{requirement_id} activated a new replacement; "
                    "starting its surface review in a new transaction"
                )
            continue

        halt_reason = (
            "no_state_progress"
            if after_fingerprint == before_fingerprint
            else "additional_review_required"
        )
        if progress is not None:
            progress.wait(
                f"{requirement_id} requires additional human review before "
                "the queue can continue"
            )
        break

    final_state = load_pre_ir_batch_state(checkpoint_dir)
    return _replacement_review_queue_outcome(
        final_state,
        initial_ids=initial_ids,
        attempted_ids=tuple(attempted),
        completed_ids=tuple(completed),
        halt_reason=halt_reason,
        had_errors=had_errors,
    )


def _activate_confirmed_replacement(
    item: dict[str, Any],
    override: HumanSemanticOverride,
) -> dict[str, Any]:
    """Activate human semantics without bypassing the manual surface gate."""

    requirement_id = str(item["requirement_id"])
    current_hash = str(item["source_hash"])
    if (
        override.status != "active"
        or override.requirement_id != requirement_id
        or override.parent_effective_source_hash != current_hash
    ):
        raise ValueError("active semantic override与当前项目不一致。")

    superseded = _append_superseded_artifact_record(item, override.override_id)
    updated_item = dict(item)
    updated_item.update(
        effective_requirement=override.replacement_text,
        source_hash=source_hash(override.replacement_text),
        segmentation_id="",
        llm_segmentation_id="",
        reading_review_id="",
        plain_status="awaiting_surface_segmentation_review",
        status="awaiting_replacement_segmentation_review",
        active_override_id=override.override_id,
        pending_override_id="",
        override_id=override.override_id,
        comparison={},
        replacement_segmentation_review_required=True,
        superseded_pre_ir_artifacts=superseded,
    )
    _clear_item_errors(updated_item)
    return updated_item


def _append_superseded_artifact_record(
    item: Mapping[str, Any],
    override_id: str,
) -> list[dict[str, Any]]:
    records = [
        dict(entry)
        for entry in item.get("superseded_pre_ir_artifacts", ())
        if isinstance(entry, Mapping)
    ]
    records.append(
        {
            "source_hash": str(item.get("source_hash", "")),
            "effective_requirement": str(item.get("effective_requirement", "")),
            "segmentation_id": str(item.get("segmentation_id", "")),
            "llm_segmentation_id": str(item.get("llm_segmentation_id", "")),
            "reading_review_id": str(item.get("reading_review_id", "")),
            "plain_status": str(item.get("plain_status", "")),
            "superseded_by_override_id": override_id,
            "superseded_at": _timestamp(),
        }
    )
    return records


def _generated_item(
    record: RequirementRecord,
    human: SurfaceSegmentation,
    llm: SurfaceSegmentation,
    comparison: SegmentationComparisonReport,
    reading: PlainReadingReview,
) -> dict[str, Any]:
    return {
        "requirement_id": record.requirement_id,
        "original_requirement": record.text,
        "effective_requirement": record.text,
        "source_hash": human.source_hash,
        "segmentation_id": human.segmentation_id,
        "llm_segmentation_id": llm.segmentation_id,
        "reading_review_id": reading.reading_review_id,
        "plain_status": reading.overall_status,
        "status": _pending_status_for_review(reading),
        "active_override_id": "",
        "pending_override_id": "",
        "override_id": "",
        "comparison": comparison.to_dict(),
        "data_split": reading.data_split,
        "ap_representation_profile": (
            asdict(record.ap_representation_profile)
            if _record_uses_v2_ap_profile(record)
            else None
        ),
        "declared_ap_catalog": (
            asdict(record.declared_ap_catalog)
            if record.declared_ap_catalog is not None
            else None
        ),
        "error": "",
    }


def _marked_text(segmentation: SurfaceSegmentation) -> str:
    return segmentation.marker.join(segment.text for segment in segmentation.segments)


def _plain_reading_diagnostic_callback(
    checkpoint_dir: str,
    requirement_id: str,
):
    """Build the one durable sink used by every batch plain-reading gateway."""

    def persist(diagnostic: Mapping[str, Any]) -> None:
        save_plain_reading_generation_diagnostic(
            checkpoint_dir,
            requirement_id=requirement_id,
            diagnostic=diagnostic,
        )

    return persist


def _plain_reading_generation_error(
    exc: BaseException,
    *,
    checkpoint_dir: str,
    requirement_id: str,
) -> dict[str, Any]:
    """Return a bounded state summary; full attempts live in their checkpoint."""

    diagnostics_value = getattr(exc, "diagnostics", {})
    diagnostics = (
        dict(diagnostics_value)
        if isinstance(diagnostics_value, Mapping)
        else {}
    )
    raw_reason_codes = getattr(exc, "reason_codes", ())
    if isinstance(raw_reason_codes, str):
        reason_codes = [raw_reason_codes] if raw_reason_codes else []
    else:
        reason_codes = [str(code) for code in (raw_reason_codes or ()) if code]
    if not reason_codes:
        diagnostic_reason = str(diagnostics.get("reason_code", "")).strip()
        if diagnostic_reason:
            reason_codes = [diagnostic_reason]
    raw_attempts = getattr(
        exc,
        "attempts",
        diagnostics.get("generation_attempts", diagnostics.get("attempt", 0)),
    )
    try:
        attempts = max(0, int(raw_attempts or 0))
    except (TypeError, ValueError):
        attempts = 0
    relative_diagnostic = (
        Path("plain_reading_diagnostics") / f"{requirement_id}.json"
    )
    diagnostic_path = Path(checkpoint_dir) / relative_diagnostic
    return {
        "kind": (
            "semantic_fallback_disabled"
            if isinstance(exc, PlainReadingSemanticFallback)
            else "hard_generation_failure"
        ),
        "stage": "plain_reading_generation",
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "reason_codes": reason_codes,
        "attempts": attempts,
        "diagnostic_ref": (
            relative_diagnostic.as_posix() if diagnostic_path.is_file() else ""
        ),
    }


def _attach_plain_reading_generation_error(
    exc: BaseException,
    *,
    checkpoint_dir: str,
    requirement_id: str,
) -> dict[str, Any]:
    details = _plain_reading_generation_error(
        exc,
        checkpoint_dir=checkpoint_dir,
        requirement_id=requirement_id,
    )
    try:
        setattr(exc, "_pre_ir_plain_reading_generation_error", details)
    except Exception:
        pass
    return details


def _record_item_apply_error(item: dict[str, Any], exc: BaseException) -> None:
    item["last_apply_error"] = str(exc)
    generation_error = getattr(
        exc,
        "_pre_ir_plain_reading_generation_error",
        None,
    )
    if isinstance(generation_error, Mapping):
        item["last_apply_generation_error"] = dict(generation_error)
        item.pop("last_apply_diagnostics", None)
        return
    item.pop("last_apply_generation_error", None)
    diagnostics = getattr(exc, "diagnostics", {})
    item["last_apply_diagnostics"] = (
        dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
    )


def _clear_item_errors(item: dict[str, Any]) -> None:
    for key in (
        "error",
        "generation_error",
        "last_apply_error",
        "last_apply_diagnostics",
        "last_apply_generation_error",
    ):
        item.pop(key, None)
    item["error"] = ""


def _blocked_item(
    record: RequirementRecord,
    error: str,
    *,
    generation_error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        "requirement_id": record.requirement_id,
        "original_requirement": record.text,
        "effective_requirement": record.text,
        "source_hash": "",
        "segmentation_id": "",
        "llm_segmentation_id": "",
        "reading_review_id": "",
        "plain_status": "blocked_error",
        "status": "blocked",
        "active_override_id": "",
        "pending_override_id": "",
        "override_id": "",
        "comparison": {},
        "data_split": "test",
        "ap_representation_profile": (
            asdict(record.ap_representation_profile)
            if _record_uses_v2_ap_profile(record)
            else None
        ),
        "declared_ap_catalog": (
            asdict(record.declared_ap_catalog)
            if record.declared_ap_catalog is not None
            else None
        ),
        "error": error,
    }
    if generation_error is not None:
        item["generation_error"] = dict(generation_error)
    return item


def _set_pending_item(item: dict[str, Any], review: PlainReadingReview) -> None:
    item["reading_review_id"] = review.reading_review_id
    item["plain_status"] = review.overall_status
    item["status"] = _pending_status_for_review(review)
    item["error"] = ""


def _record_uses_v2_ap_profile(record: RequirementRecord) -> bool:
    return bool(
        record.ap_profile_declared
        or record.declared_ap_catalog is not None
        or record.ap_representation_profile.mode != "undetermined"
    )


def _pending_status_for_review(review: PlainReadingReview) -> str:
    if review.overall_status == "pending_no_ambiguity_confirmation":
        if review.review_basis == SEMANTIC_FALLBACK_REVIEW_BASIS:
            return "awaiting_semantic_fallback_review"
        return "awaiting_no_ambiguity_review"
    if review.overall_status == "pending_plain_review":
        return "awaiting_ambiguity_review"
    if review.overall_status == "awaiting_human_semantic_override":
        return "awaiting_semantic_override_confirmation"
    return "blocked"


def _batch_status_for_override(override: HumanSemanticOverride) -> str:
    status = {
        "awaiting_source_modality_choice": "awaiting_source_modality_choice",
        "awaiting_ap_inventory": "awaiting_semantic_override_ap_inventory",
        "awaiting_ap_mapping_confirmation": "awaiting_semantic_override_ap_mapping",
        "awaiting_confirmation": "awaiting_semantic_override_confirmation",
    }.get(override.status)
    if status is None:
        raise ValueError(
            f"unsupported pending semantic override status: {override.status}"
        )
    return status


def _route_item_by_readiness(
    item: dict[str, Any],
    review: PlainReadingReview,
    segmentation: SurfaceSegmentation,
    *,
    checkpoint_dir: str | None = None,
) -> None:
    readiness = assess_plain_reading_ir_readiness(review, segmentation)
    if readiness.disposition == "invalid":
        raise PlainReadingValidationError(
            "；".join(readiness.validation_errors),
            errors=readiness.validation_errors,
        )
    item["reading_review_id"] = review.reading_review_id
    item["plain_status"] = review.overall_status
    item["error"] = ""
    item["plain_reading_readiness"] = asdict(readiness)
    item["classification"] = (
        "eligible" if readiness.can_enter_semantic_ir else "needs_review"
    )
    item["suggested_rerun_stage"] = (
        "" if readiness.can_enter_semantic_ir else "plain_reading"
    )
    item["review_completed"] = False
    item["single_ir_ready"] = False
    item["finalization_allowed"] = False
    item["next_stage"] = "plain_reading"
    candidate_route = readiness.can_attempt_candidate_ir
    if readiness.can_enter_semantic_ir or candidate_route:
        item["status"] = "ready_for_single_ir"
        item["review_completed"] = True
        item["single_ir_ready"] = readiness.can_enter_semantic_ir
        item["finalization_allowed"] = readiness.can_enter_semantic_ir
        item["next_stage"] = "candidate_ir" if candidate_route else "semantic_ir"
        item["classification"] = "eligible"
        item["suggested_rerun_stage"] = ""
    elif readiness.disposition == "semantic_resolution_required":
        need_view = build_plain_reading_need_view(review, segmentation)
        save_plain_reading_need_view_checkpoint(
            checkpoint_dir,
            need_view=need_view,
        )
        item["status"] = "plain_reading_need_view"
        item["plain_reading_need_view"] = asdict(need_view)
        item["plain_reading_need_view_id"] = need_view.need_view_id
        item["review_completed"] = True
        item["single_ir_ready"] = False
        item["finalization_allowed"] = False
        item["next_stage"] = need_view.next_stage
    elif readiness.disposition == "source_clarification_required":
        item["status"] = "deferred_source_clarification"
    else:
        item["status"] = _pending_status_for_review(review)


@_serialized_review_apply
def persist_trace_selected_plain_reading(*, checkpoint_dir, segmentation, previous_review_id, review):
    """Advance the current reading after human trace discrimination, atomically."""
    if review.reviewer_source != "human_trace" or review.overall_status != "plain_reading_reviewed":
        raise ValueError("trace-selected checkpoint requires a closed human trajectory reading")
    root = Path(checkpoint_dir)
    state = _load_state(root)
    item = state.get("items", {}).get(review.requirement_id) if state else None
    if item is not None and (item.get("reading_review_id") != previous_review_id
                             or item.get("source_hash") != review.source_hash
                             or item.get("segmentation_id") != review.segmentation_id):
        raise ValueError("trace-selected reading has a stale deferred parent")
    _save_reviewed(checkpoint_dir, segmentation, review)
    if item is not None:
        item.setdefault("trace_selected_reading_parents", []).append(previous_review_id)
        _route_item_by_readiness(item, review, segmentation, checkpoint_dir=checkpoint_dir)
        _close_if_complete(state)
        _write_state(root, state)


def _save_reviewed(
    checkpoint_dir: str,
    segmentation: SurfaceSegmentation,
    review: PlainReadingReview,
) -> None:
    save_plain_reading_reviewed_checkpoint(
        checkpoint_dir,
        requirement_id=review.requirement_id,
        plain_reading_review=review,
        segmentation_schema_version=segmentation.schema_version,
    )


def _load_pending_artifacts(
    checkpoint_dir: str,
    requirement_id: str,
    item: Mapping[str, Any],
) -> tuple[SurfaceSegmentation, PlainReadingReview]:
    segmentation_payload = load_segmentation_reviewed_checkpoint(
        checkpoint_dir,
        requirement_id,
        expected_source_hash=str(item["source_hash"]),
        expected_segmentation_id=str(item["segmentation_id"]),
    )
    if not segmentation_payload:
        raise ValueError("缺少或已过期的 reviewed segmentation checkpoint。")
    segmentation_mapping = segmentation_payload.get("surface_segmentation")
    if not isinstance(segmentation_mapping, Mapping):
        raise ValueError("reviewed segmentation checkpoint 格式错误。")
    segmentation = segmentation_from_mapping(segmentation_mapping)
    reading_payload = load_plain_reading_generated_checkpoint(
        checkpoint_dir,
        requirement_id,
        expected_source_hash=segmentation.source_hash,
        expected_segmentation_id=segmentation.segmentation_id,
    )
    if not reading_payload:
        raise ValueError("缺少或已过期的 generated plain-reading checkpoint。")
    reading_mapping = reading_payload.get("plain_reading_review")
    if not isinstance(reading_mapping, Mapping):
        raise ValueError("plain-reading checkpoint 格式错误。")
    review = plain_reading_from_mapping(reading_mapping, segmentation)
    if review.reading_review_id != item["reading_review_id"]:
        raise ValueError("审核文件引用了已过期的 reading review。")
    return segmentation, review


def _item_ap_profile(item: Mapping[str, Any]) -> APRepresentationProfile | None:
    value = item.get("ap_representation_profile")
    if value is None:
        # A state written before runtime must retain the frozen v1 proposal path.
        return None
    if not isinstance(value, Mapping):
        raise ValueError("AP representation profile metadata is malformed")
    return ap_representation_profile_from_mapping(value)


def _item_declared_catalog(item: Mapping[str, Any]) -> DeclaredAPCatalog | None:
    value = item.get("declared_ap_catalog")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("declared AP catalog metadata is malformed")
    return declared_ap_catalog_from_mapping(value)


def _load_pending_override_for_item(
    checkpoint_dir: str,
    requirement_id: str,
    item: Mapping[str, Any],
    expected_status: str,
) -> HumanSemanticOverride:
    override = load_human_semantic_override_checkpoint(
        checkpoint_dir,
        requirement_id,
        pending_only=True,
        expected_original_source_hash=source_hash(_item_override_source_text(item)),
    )
    if (
        override is None
        or override.override_id != str(item.get("pending_override_id", ""))
        or override.parent_effective_source_hash != str(item.get("source_hash", ""))
        or override.status != expected_status
    ):
        raise ValueError("pending semantic override checkpoint缺失、过期或状态不匹配。")
    return override


def _item_override_source_text(item: Mapping[str, Any]) -> str:
    return str(
        item.get("effective_requirement")
        if _item_ap_profile(item) is not None
        else item.get("original_requirement")
    )


def _item_current_ap_inventory(
    checkpoint_dir: str,
    item: Mapping[str, Any],
) -> APInventory | None:
    inventory_id = str(item.get("effective_ap_inventory_id", ""))
    if not inventory_id:
        return None
    profile = _item_ap_profile(item)
    inventory = load_ap_inventory_checkpoint(
        checkpoint_dir,
        str(item["requirement_id"]),
        inventory_id=inventory_id,
        expected_source_hash=str(item["source_hash"]),
        expected_profile_id=profile.profile_id if profile is not None else None,
    )
    if inventory is None or inventory.artifact_hash != str(
        item.get("effective_ap_inventory_hash", "")
    ):
        raise ValueError("effective AP inventory checkpoint缺失或已过期。")
    return inventory


def _item_ap_grounding_contract(
    item: Mapping[str, Any],
    *,
    segmentation: SurfaceSegmentation,
    checkpoint_dir: str,
):
    profile = _item_ap_profile(item) or undetermined_profile(
        provenance="deferred_pre_ir_default"
    )
    inventory = (
        _item_current_ap_inventory(checkpoint_dir, item)
        if checkpoint_dir
        else None
    )
    return build_ap_grounding_contract(
        requirement_id=str(item["requirement_id"]),
        source_text=str(item["effective_requirement"]),
        representation_profile=profile,
        segmentation=segmentation,
        declared_catalog=_item_declared_catalog(item),
        active_inventory=inventory,
        provenance=("deferred_pre_ir_plain_reading",),
    )


def _refresh_review_files(
    state: dict[str, Any],
    *,
    checkpoint_dir: str,
    records_dir: str,
    force: bool = False,
    force_ambiguity: bool = False,
    added_requirement_ids: Sequence[str] = (),
) -> None:
    prior_paths = {
        key: str(state.get(key, "")).strip()
        for key in (
            "pending_no_ambiguity_file",
            "pending_semantic_fallback_file",
            "pending_ap_inventory_file",
            "pending_ap_mapping_file",
            "pending_ambiguity_file",
        )
    }
    review_root = _review_namespace_root(records_dir, checkpoint_dir)
    review_root.mkdir(parents=True, exist_ok=True)
    state["review_namespace_dir"] = str(review_root)
    no_ids = _pending_ids(state, "no_ambiguity")
    fallback_ids = _pending_ids(state, "semantic_fallback")
    inventory_ids = _pending_ids(state, "ap_inventory")
    mapping_ids = _pending_ids(state, "ap_mapping")
    ambiguity_ids = _pending_ids(state, "ambiguity")
    if no_ids:
        path = review_root / (
            f"pre_ir_no_ambiguity_round_{int(state.get('no_ambiguity_round', 1))}.md"
        )
        _write_reconciled_review(
            path,
            _format_no_ambiguity_review(
                state, no_ids, checkpoint_dir=checkpoint_dir
            ),
            no_ids,
            state=state,
            action_contract="normal-no-axis-v1",
            migration_source=_review_migration_source(
                prior_paths["pending_no_ambiguity_file"], path
            ),
        )
        state["pending_no_ambiguity_file"] = str(path)
        state["no_ambiguity_review_complete"] = False
    else:
        state["pending_no_ambiguity_file"] = ""
        state["no_ambiguity_review_complete"] = True
    if fallback_ids:
        path = review_root / (
            "pre_ir_semantic_fallback_round_"
            + str(int(state.get("semantic_fallback_round", 1)))
            + ".md"
        )
        _write_reconciled_review(
            path,
            _format_semantic_fallback_review(
                state, fallback_ids, checkpoint_dir=checkpoint_dir
            ),
            fallback_ids,
            state=state,
            action_contract="semantic-fallback-v1",
            migration_source=_review_migration_source(
                prior_paths["pending_semantic_fallback_file"], path
            ),
        )
        state["pending_semantic_fallback_file"] = str(path)
        state["semantic_fallback_review_complete"] = False
    else:
        state["pending_semantic_fallback_file"] = ""
        state["semantic_fallback_review_complete"] = True
    if inventory_ids:
        path = review_root / (
            "pre_ir_semantic_override_ap_inventory_round_"
            + str(int(state.get("ap_inventory_round", 1)))
            + ".md"
        )
        _write_reconciled_review(
            path,
            _format_ap_inventory_review(
                state, inventory_ids, checkpoint_dir=checkpoint_dir
            ),
            inventory_ids,
            state=state,
            action_contract="semantic-override-ap-inventory-v1",
            migration_source=_review_migration_source(
                prior_paths["pending_ap_inventory_file"], path
            ),
        )
        state["pending_ap_inventory_file"] = str(path)
        state["ap_inventory_review_complete"] = False
    else:
        state["pending_ap_inventory_file"] = ""
        state["ap_inventory_review_complete"] = True
    if mapping_ids:
        path = review_root / (
            "pre_ir_semantic_override_ap_mapping_round_"
            + str(int(state.get("ap_mapping_round", 1)))
            + ".md"
        )
        _write_reconciled_review(
            path,
            _format_ap_mapping_review(
                state, mapping_ids, checkpoint_dir=checkpoint_dir
            ),
            mapping_ids,
            state=state,
            action_contract="semantic-override-ap-mapping-v1",
            migration_source=_review_migration_source(
                prior_paths["pending_ap_mapping_file"], path
            ),
        )
        state["pending_ap_mapping_file"] = str(path)
        state["ap_mapping_review_complete"] = False
    else:
        state["pending_ap_mapping_file"] = ""
        state["ap_mapping_review_complete"] = True
    if ambiguity_ids:
        path = review_root / (
            f"pre_ir_ambiguity_round_{int(state.get('ambiguity_round', 1))}.md"
        )
        _write_reconciled_review(
            path,
            _format_ambiguity_review(
                state, ambiguity_ids, checkpoint_dir=checkpoint_dir
            ),
            ambiguity_ids,
            state=state,
            action_contract="ambiguity-adjudication-v2",
            migration_source=_review_migration_source(
                prior_paths["pending_ambiguity_file"], path
            ),
        )
        state["pending_ambiguity_file"] = str(path)
        state["ambiguity_review_complete"] = False
    else:
        state["pending_ambiguity_file"] = ""
        state["ambiguity_review_complete"] = True
    state["updated_at"] = _timestamp()


def _write_reconciled_review(
    path: Path,
    rendered: str,
    requirement_ids: Sequence[str],
    *,
    state: Mapping[str, Any],
    action_contract: str,
    migration_source: Path | None = None,
) -> None:
    """Rebuild one active review view while preserving current human input.

    A section is reusable only while all parent metadata still matches the
    canonical state.  Stale and already-completed sections are therefore
    removed, while newly promoted requirements are always materialized.
    """

    rendered_sections = _requirement_sections(rendered)
    existing_path = migration_source or (path if path.exists() else None)
    existing_text = (
        existing_path.read_text(encoding="utf-8")
        if existing_path is not None and existing_path.is_file()
        else ""
    )
    existing_sections = _requirement_sections(existing_text) if existing_text else {}
    first_section = re.search(
        r"^##\s+[A-Za-z][A-Za-z0-9_-]*-\d+\s*$",
        rendered,
        re.MULTILINE,
    )
    header = rendered[: first_section.start()].rstrip() if first_section else rendered.rstrip()
    sections: list[str] = []
    reused_ids: set[str] = set()
    for requirement_id in requirement_ids:
        section = rendered_sections.get(requirement_id)
        existing = existing_sections.get(requirement_id)
        item = state.get("items", {}).get(requirement_id, {})
        if existing is not None:
            try:
                _validate_section_metadata(existing, item)
            except ValueError:
                pass
            else:
                rendered_contract = _read_field(
                    section or "", "action_contract_version"
                )
                existing_contract = _read_field(
                    existing, "action_contract_version"
                )
                legacy_compatible = (
                    not existing_contract
                    and rendered_contract
                    in {
                        "normal-no-axis-v1",
                        "ambiguity-adjudication-v1",
                    }
                )
                legacy_ambiguity_v1 = (
                    existing_contract == "ambiguity-adjudication-v1"
                    and rendered_contract == "ambiguity-adjudication-v2"
                )
                identity_fields = (
                    "override_id",
                    "original_inventory_id",
                    "original_inventory_hash",
                    "replacement_inventory_id",
                    "replacement_inventory_hash",
                    "comparison_id",
                    "comparison_hash",
                )
                override_matches = all(
                    not _read_field(section or "", field)
                    or _read_field(existing, field)
                    == _read_field(section or "", field)
                    for field in identity_fields
                )
                if (
                    (
                        existing_contract == rendered_contract
                        or legacy_compatible
                        or legacy_ambiguity_v1
                    )
                    and override_matches
                ):
                    section = existing
                    if legacy_ambiguity_v1:
                        section = _migrate_legacy_ambiguity_v1_section(section)
                    reused_ids.add(requirement_id)
        if section is None:
            raise ValueError(f"无法生成 {requirement_id} 的 Pre-IR 审核区块。")
        rendered_contract = _read_field(
            rendered_sections.get(requirement_id, ""),
            "action_contract_version",
        ) or action_contract
        section = _upgrade_review_section_contract(section, rendered_contract)
        sections.append(f"## {requirement_id}{section.rstrip()}")
    text = header
    if (
        set(requirement_ids) == reused_ids
        and _read_field(existing_text, "review_complete").casefold()
        in {"yes", "true", "是"}
    ):
        text = re.sub(
            r"^review_complete\s*:\s*.*?$",
            "review_complete: yes",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    if sections:
        text += "\n\n" + "\n\n".join(sections)
    _atomic_write_text(path, text.rstrip() + "\n")


def _review_migration_source(source_value: str, target: Path) -> Path | None:
    """Select a state-declared legacy view as a safe reconciliation source."""

    if not source_value:
        return None
    source = Path(source_value)
    if _same_path(source, target) or not source.is_file():
        return None
    return source


def _upgrade_review_section_contract(section: str, contract: str) -> str:
    # Repair the short-lived v2 migration bug that inserted the field into the
    # replacement envelope instead of the traj-ltlf metadata envelope.
    section = re.sub(
        r"^<!-- replacement:start\s+action_contract_version:\s*[^\r\n]+$",
        "<!-- replacement:start -->",
        section,
        flags=re.MULTILINE,
    )
    if _read_field(section, "action_contract_version"):
        return section
    match = re.search(r"<!-- traj-ltlf-meta[\s\S]*?^-->\s*$", section, re.MULTILINE)
    if match is None:
        raise ValueError("Pre-IR审核区块缺少traj-ltlf metadata envelope。")
    envelope = match.group(0)
    upgraded = re.sub(
        r"^-->\s*$",
        f"action_contract_version: {contract}\n-->",
        envelope,
        count=1,
        flags=re.MULTILINE,
    )
    return section[: match.start()] + upgraded + section[match.end() :]


def _migrate_legacy_ambiguity_v1_section(section: str) -> str:
    """Preserve unsubmitted v1 work while making legacy ``both`` explicit U."""

    migrated, count = re.subn(
        r"^(用户填写\s*:\s*)both\s*$",
        r"\1U",
        section,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    migrated = re.sub(
        r"^action_contract_version\s*:\s*ambiguity-adjudication-v1\s*$",
        "action_contract_version: ambiguity-adjudication-v2",
        migrated,
        flags=re.MULTILINE,
    )
    if count:
        warning = (
            "<!-- compatibility-warning: legacy both was migrated to U; "
            "U means a real unresolved semantic axis, not two equivalent readings. -->\n"
        )
        metadata = migrated.find("<!-- traj-ltlf-meta")
        if metadata >= 0:
            migrated = migrated[:metadata] + warning + migrated[metadata:]
        else:
            migrated = warning + migrated
    return migrated


def _append_review_sections(
    path: Path,
    rendered_additions: str,
    requirement_ids: Sequence[str],
) -> None:
    """Append new generated items without erasing existing human commands."""

    existing = path.read_text(encoding="utf-8")
    existing_sections = _requirement_sections(existing)
    rendered_sections = _requirement_sections(rendered_additions)
    additions: list[str] = []
    for requirement_id in requirement_ids:
        if requirement_id in existing_sections:
            continue
        section = rendered_sections.get(requirement_id)
        if section is None:
            raise ValueError(f"无法生成 {requirement_id} 的 Pre-IR 审核区块。")
        additions.append(f"## {requirement_id}{section.rstrip()}")
    if not additions:
        return
    existing = re.sub(
        r"^review_complete\s*:\s*.*?$",
        "review_complete: no",
        existing,
        count=1,
        flags=re.MULTILINE,
    )
    path.write_text(
        existing.rstrip() + "\n\n" + "\n\n".join(additions) + "\n",
        encoding="utf-8",
    )


def _advance_completed_review_rounds_for_retry(
    state: dict[str, Any],
    *,
    records_dir: str,
    checkpoint_dir: str,
    pending_before: Mapping[str, set[str]],
    added_requirement_ids: set[str],
) -> None:
    """Use a new review round only when the previous round was already closed."""

    review_root = _review_namespace_root(records_dir, checkpoint_dir)
    for kind, round_key, filename_prefix in (
        ("no_ambiguity", "no_ambiguity_round", "pre_ir_no_ambiguity_round"),
        (
            "semantic_fallback",
            "semantic_fallback_round",
            "pre_ir_semantic_fallback_round",
        ),
        ("ambiguity", "ambiguity_round", "pre_ir_ambiguity_round"),
    ):
        new_ids = set(_pending_ids(state, kind)) & added_requirement_ids
        if not new_ids or pending_before.get(kind):
            continue
        round_number = int(state.get(round_key, 1))
        prior_path = review_root / f"{filename_prefix}_{round_number}.md"
        if prior_path.exists():
            state[round_key] = round_number + 1


def _preprocessing_generation_contract(
    *,
    segmentation_prompt_version: str,
    plain_reading_prompt_version: str,
    semantic_fallback_enabled: bool,
) -> dict[str, Any]:
    """Describe the exact software contract that produced blocked batch items."""

    return {
        "validated_generation_contract_version": (
            VALIDATED_GENERATION_CONTRACT_VERSION
        ),
        "validated_generation_validator_version": (
            VALIDATED_GENERATION_VALIDATOR_VERSION
        ),
        "repair_policy_version": REPAIR_POLICY_VERSION,
        "patch_authorization_contract_version": (
            PATCH_AUTHORIZATION_CONTRACT_VERSION
        ),
        "segmentation_prompt_version": segmentation_prompt_version,
        "plain_reading_prompt_version": plain_reading_prompt_version,
        "semantic_fallback_enabled": bool(semantic_fallback_enabled),
    }


def _format_no_ambiguity_review(
    state: Mapping[str, Any],
    requirement_ids: Sequence[str],
    *,
    checkpoint_dir: str,
) -> str:
    lines = [
        "# LLM未发现歧义——用户抽查",
        "",
        "review_type: no_ambiguity_screening",
        f"schema_version: {PRE_IR_REVIEW_SCHEMA_VERSION}",
        f"round: {state.get('no_ambiguity_round', 1)}",
        "review_complete: no",
        "",
        "填写说明：A 表示接受；E <非权威歧义关注说明> 请求候选轴；C 暂缓。",
        "检查完成后把文件头的 `review_complete` 改为 `yes`。",
        "",
        "## 填写示例（解析器忽略）",
        "",
        "```text",
        "用户填写: e “or always c holds”可能是全局选择，也可能只属于局部响应。",
        "```",
        "",
    ]
    for requirement_id in requirement_ids:
        item = state["items"][requirement_id]
        segmentation, review = _load_pending_artifacts(
            checkpoint_dir, requirement_id, item
        )
        lines.extend(
            [
                f"## {requirement_id}",
                "",
                "原文：",
                "",
                "```text",
                item["effective_requirement"],
                "```",
                "",
                "LLM判断：未发现普通语言歧义。",
                "",
                "用户填写: a",
                "",
                "<!-- traj-ltlf-meta",
                f"requirement_id: {requirement_id}",
                f"source_hash: {item['source_hash']}",
                f"segmentation_id: {segmentation.segmentation_id}",
                f"reading_review_id: {review.reading_review_id}",
                "action_contract_version: normal-no-axis-v1",
                "-->",
                "",
            ]
        )
    return "\n".join(lines)


def _format_semantic_fallback_review(
    state: Mapping[str, Any],
    requirement_ids: Sequence[str],
    *,
    checkpoint_dir: str,
) -> str:
    lines = [
        "# Plain Reading语义Fallback人工裁决",
        "",
        "review_type: semantic_fallback_adjudication",
        f"schema_version: {PRE_IR_REVIEW_SCHEMA_VERSION}",
        f"round: {state.get('semantic_fallback_round', 1)}",
        "review_complete: no",
        "",
        "这些项目的两个结构化artifact均未留下安全的来源支持歧义轴。",
        "普通裁决填写 Y（接受当前原文）、R（提供完整replacement）或 C（暂缓）。",
        "选择 R 后系统只保存待确认替换；下一轮展示AP集合，必须再填写 CONFIRM 才会生效。",
        "",
    ]
    for requirement_id in requirement_ids:
        item = state["items"][requirement_id]
        segmentation, review = _load_pending_artifacts(
            checkpoint_dir, requirement_id, item
        )
        status = str(item.get("status", ""))
        lines.extend(
            [
                f"## {requirement_id}",
                "",
                "当前有效原文：",
                "",
                "```text",
                item["effective_requirement"],
                "```",
                "",
            ]
        )
        override_id = ""
        if status in {
            "awaiting_semantic_override_ap_inventory",
            "awaiting_semantic_override_ap_mapping",
            "awaiting_semantic_override_confirmation",
        }:
            override = load_human_semantic_override_checkpoint(
                checkpoint_dir,
                requirement_id,
                pending_only=True,
                expected_original_source_hash=source_hash(
                    _item_override_source_text(item)
                ),
            )
            if override is None or override.status != "awaiting_confirmation":
                raise ValueError(
                    f"{requirement_id}: 缺少等待确认的semantic override checkpoint。"
                )
            override_id = override.override_id
            action_contract = "semantic-override-confirmation-v1"
            lines.extend(
                [
                    "待确认replacement：",
                    "",
                    "```text",
                    override.replacement_text,
                    "```",
                    "",
                    "原文AP集合: " + ", ".join(override.original_ap_set),
                    "replacement AP集合: " + ", ".join(override.replacement_ap_set),
                    "AP校验: " + override.ap_validation_status,
                ]
            )
            original_inventory, replacement_inventory, comparison = (
                load_human_semantic_override_ap_artifacts(
                    checkpoint_dir, override
                )
            )
            if (
                original_inventory is not None
                and replacement_inventory is not None
                and comparison is not None
            ):
                lines.extend(
                    [
                        "original inventory: "
                        + original_inventory.inventory_id
                        + " / "
                        + original_inventory.artifact_hash,
                        "replacement inventory: "
                        + replacement_inventory.inventory_id
                        + " / "
                        + replacement_inventory.artifact_hash,
                        "comparison: "
                        + comparison.comparison_id
                        + " / "
                        + comparison.status
                        + " / "
                        + comparison.validation_version,
                        "identity mapping:",
                        "",
                        "```json",
                        json.dumps(
                            [asdict(mapping) for mapping in comparison.identity_mapping],
                            ensure_ascii=False,
                            indent=2,
                        ),
                        "```",
                        "",
                        "注意：系统只检查AP身份，不检查其它语义变化；确认后旧Pre-IR artifact失效。",
                    ]
                )
            lines.extend(["", "用户填写: c"])
        else:
            if review.review_basis != SEMANTIC_FALLBACK_REVIEW_BASIS:
                raise ValueError(
                    f"{requirement_id}: semantic fallback状态与review_basis不一致。"
                )
            action_contract = "semantic-fallback-decision-v1"
            lines.extend(
                [
                    "用户填写: c",
                    "",
                    "完整replacement（仅当填写 R 时使用）：",
                    "",
                    "<!-- replacement:start -->",
                    "```text",
                    "",
                    "```",
                    "<!-- replacement:end -->",
                ]
            )
        lines.extend(
            [
                "",
                "<!-- traj-ltlf-meta",
                f"requirement_id: {requirement_id}",
                f"source_hash: {item['source_hash']}",
                f"segmentation_id: {segmentation.segmentation_id}",
                f"reading_review_id: {review.reading_review_id}",
                f"override_id: {override_id}",
                f"action_contract_version: {action_contract}",
                "-->",
                "",
            ]
        )
    return "\n".join(lines)


def _format_ap_inventory_review(
    state: Mapping[str, Any],
    requirement_ids: Sequence[str],
    *,
    checkpoint_dir: str,
) -> str:
    lines = [
        "# Semantic Override AP Inventory审核",
        "",
        "review_type: semantic_override_ap_inventory",
        f"schema_version: {PRE_IR_REVIEW_SCHEMA_VERSION}",
        f"round: {state.get('ap_inventory_round', 1)}",
        "review_complete: no",
        "",
        "本步骤只批准两侧原子命题身份及其原文证据，不批准replacement语义。",
        "填写 REVISE 保存修改后进入新一轮，APPROVE 明确批准，REJECT 取消replacement，C 暂缓。",
        "每个 evidence_text 必须逐字出现在对应文本中；空清单必须显式给出理由。",
        "",
    ]
    for requirement_id in requirement_ids:
        item = state["items"][requirement_id]
        segmentation, review = _load_pending_artifacts(
            checkpoint_dir, requirement_id, item
        )
        override = _load_pending_override_for_item(
            checkpoint_dir, requirement_id, item, "awaiting_ap_inventory"
        )
        original, replacement, comparison = (
            load_human_semantic_override_ap_artifacts(checkpoint_dir, override)
        )
        if original is None or replacement is None or comparison is None:
            raise ValueError(f"{requirement_id}: AP inventory artifact链不完整。")
        payload = {
            "schema_version": 1,
            "original": _editable_inventory_proposals(original),
            "replacement": _editable_inventory_proposals(replacement),
            "original_approved_empty": bool(original.approved_empty),
            "original_approved_empty_reason": original.approved_empty_reason,
            "replacement_approved_empty": bool(replacement.approved_empty),
            "replacement_approved_empty_reason": replacement.approved_empty_reason,
        }
        lines.extend(
            [
                f"## {requirement_id}",
                "",
                f"profile: {override.representation_profile_id}",
                f"extraction_mode: {original.extraction_mode} / {replacement.extraction_mode}",
                "",
                "原文：",
                "",
                "```text",
                override.original_source_text,
                "```",
                "",
                "replacement：",
                "",
                "```text",
                override.replacement_text,
                "```",
                "",
                "结构化 AP inventory（可编辑）：",
                "",
                "```ap-inventory-json",
                json.dumps(payload, ensure_ascii=False, indent=2),
                "```",
                "",
                "用户填写: c",
                "",
                "<!-- traj-ltlf-meta",
                f"requirement_id: {requirement_id}",
                f"source_hash: {item['source_hash']}",
                f"segmentation_id: {segmentation.segmentation_id}",
                f"reading_review_id: {review.reading_review_id}",
                f"override_id: {override.override_id}",
                f"original_inventory_id: {original.inventory_id}",
                f"original_inventory_hash: {original.artifact_hash}",
                f"replacement_inventory_id: {replacement.inventory_id}",
                f"replacement_inventory_hash: {replacement.artifact_hash}",
                f"comparison_id: {comparison.comparison_id}",
                "action_contract_version: semantic-override-ap-inventory-v1",
                "-->",
                "",
            ]
        )
    return "\n".join(lines)


def _format_ap_mapping_review(
    state: Mapping[str, Any],
    requirement_ids: Sequence[str],
    *,
    checkpoint_dir: str,
) -> str:
    lines = [
        "# Semantic Override AP Identity Mapping审核",
        "",
        "review_type: semantic_override_ap_mapping",
        f"schema_version: {PRE_IR_REVIEW_SCHEMA_VERSION}",
        f"round: {state.get('ap_mapping_round', 1)}",
        "review_complete: no",
        "",
        "本步骤只确认两侧是否指向同一个原子命题；不确认时间、否定、布尔或因果语义。",
        "填写 APPROVE 批准完整一一映射，REJECT 取消replacement，C 暂缓。",
        "映射通过后仍会进入独立的 CONFIRM/REJECT replacement 确认。",
        "",
    ]
    for requirement_id in requirement_ids:
        item = state["items"][requirement_id]
        segmentation, review = _load_pending_artifacts(
            checkpoint_dir, requirement_id, item
        )
        override = _load_pending_override_for_item(
            checkpoint_dir,
            requirement_id,
            item,
            "awaiting_ap_mapping_confirmation",
        )
        original, replacement, comparison = (
            load_human_semantic_override_ap_artifacts(checkpoint_dir, override)
        )
        if original is None or replacement is None or comparison is None:
            raise ValueError(f"{requirement_id}: AP mapping artifact链不完整。")
        payload = {
            "schema_version": 1,
            "identity_mapping": [
                {
                    "replacement_ap_identity_id": mapping.replacement_ap_identity_id,
                    "original_ap_identity_id": mapping.original_ap_identity_id,
                    "mapping_provenance": "human",
                }
                for mapping in (
                    comparison.identity_mapping
                    or _suggest_identity_mapping(original, replacement)
                )
            ],
        }
        lines.extend(
            [
                f"## {requirement_id}",
                "",
                "原文 identities：",
                "",
                "```json",
                json.dumps(_identity_evidence_view(original), ensure_ascii=False, indent=2),
                "```",
                "",
                "replacement identities：",
                "",
                "```json",
                json.dumps(_identity_evidence_view(replacement), ensure_ascii=False, indent=2),
                "```",
                "",
                "结构化 identity mapping（可编辑）：",
                "",
                "```ap-mapping-json",
                json.dumps(payload, ensure_ascii=False, indent=2),
                "```",
                "",
                "用户填写: c",
                "",
                "<!-- traj-ltlf-meta",
                f"requirement_id: {requirement_id}",
                f"source_hash: {item['source_hash']}",
                f"segmentation_id: {segmentation.segmentation_id}",
                f"reading_review_id: {review.reading_review_id}",
                f"override_id: {override.override_id}",
                f"original_inventory_id: {original.inventory_id}",
                f"original_inventory_hash: {original.artifact_hash}",
                f"replacement_inventory_id: {replacement.inventory_id}",
                f"replacement_inventory_hash: {replacement.artifact_hash}",
                f"comparison_id: {comparison.comparison_id}",
                f"comparison_hash: {comparison.artifact_hash}",
                "action_contract_version: semantic-override-ap-mapping-v1",
                "-->",
                "",
            ]
        )
    return "\n".join(lines)


def _editable_inventory_proposals(inventory: APInventory) -> list[dict[str, Any]]:
    evidence_by_id: dict[str, list[str]] = {}
    for mention in inventory.mentions:
        evidence_by_id.setdefault(mention.ap_identity_id, []).append(
            mention.evidence_text
        )
    return [
        {
            "canonical_name": identity.canonical_name,
            "aliases": list(identity.aliases),
            "evidence_texts": evidence_by_id.get(identity.ap_identity_id, []),
        }
        for identity in inventory.identities
    ]


def _identity_evidence_view(inventory: APInventory) -> list[dict[str, Any]]:
    evidence_by_id: dict[str, list[dict[str, Any]]] = {}
    for mention in inventory.mentions:
        evidence_by_id.setdefault(mention.ap_identity_id, []).append(
            {
                "evidence_text": mention.evidence_text,
                "source_span": list(mention.source_span),
                "supporting_segment_ids": list(mention.supporting_segment_ids),
            }
        )
    return [
        {
            "ap_identity_id": identity.ap_identity_id,
            "canonical_name": identity.canonical_name,
            "evidence": evidence_by_id.get(identity.ap_identity_id, []),
        }
        for identity in inventory.identities
    ]


def _suggest_identity_mapping(
    original: APInventory,
    replacement: APInventory,
) -> tuple[APIdentityMapping, ...]:
    originals = {item.canonical_name.casefold(): item for item in original.identities}
    return tuple(
        APIdentityMapping(
            replacement_ap_identity_id=replacement_identity.ap_identity_id,
            original_ap_identity_id=originals[
                replacement_identity.canonical_name.casefold()
            ].ap_identity_id,
            mapping_provenance="human",
        )
        for replacement_identity in replacement.identities
        if replacement_identity.canonical_name.casefold() in originals
    )


def _format_ambiguity_review(
    state: Mapping[str, Any],
    requirement_ids: Sequence[str],
    *,
    checkpoint_dir: str,
) -> str:
    lines = [
        "# Pre-IR统一歧义裁决",
        "",
        "review_type: ambiguity_adjudication",
        f"schema_version: {PRE_IR_REVIEW_SCHEMA_VERSION}",
        f"round: {state.get('ambiguity_round', 1)}",
        "review_complete: no",
        "",
        "填写说明：每个问题填写 A、B、N、U 或 R。A/B 唯一选择读法；N 表示当前轴无实质差异；U 表示轴真实存在但仍无法裁决；R 修订选项或填写完整替换语义（其余轴保留 pending）。",
        "U 不表示“两种说法都可以”；U 表示存在真实、行为不同且尚未裁决的语义。若 A/B 只是同义改写，应选择 N。",
        "检查完成后把文件头的 `review_complete` 改为 `yes`。",
        "",
        "## 填写示例（解析器忽略）",
        "",
        "```text",
        "用户填写: B",
        "```",
        "",
    ]
    for requirement_id in requirement_ids:
        item = state["items"][requirement_id]
        _segmentation, review = _load_pending_artifacts(
            checkpoint_dir, requirement_id, item
        )
        lines.extend(
            [
                f"## {requirement_id}",
                "",
                "原文：",
                "",
                "```text",
                item["effective_requirement"],
                "```",
                "",
            ]
        )
        for ambiguity in review.ambiguities:
            lines.extend(
                [
                    f"### {ambiguity.ambiguity_id}",
                    "",
                    "歧义问题：",
                    "",
                    "```text",
                    ambiguity.question,
                    "```",
                    "",
                ]
            )
            for option in ambiguity.options:
                lines.extend(
                    [
                        f"{option.option_id}：",
                        "",
                        "```text",
                        option.plain_interpretation,
                        "```",
                        "",
                    ]
                )
            lines.extend(["用户填写: pending", ""])
        lines.extend(
            [
                "替换语义（仅当填写 r 时使用）：",
                "",
                "<!-- replacement:start -->",
                "```text",
                "",
                "```",
                "<!-- replacement:end -->",
                "",
                "<!-- traj-ltlf-meta",
                f"requirement_id: {requirement_id}",
                f"source_hash: {item['source_hash']}",
                f"segmentation_id: {item['segmentation_id']}",
                f"reading_review_id: {item['reading_review_id']}",
                "action_contract_version: ambiguity-adjudication-v2",
                "-->",
                "",
            ]
        )
    return "\n".join(lines)


def _no_ambiguity_command(section: str) -> tuple[str, str]:
    match = re.search(
        r"^用户填写\s*:\s*(.*?)\s*$",
        _without_fenced_code_content(section),
        re.MULTILINE,
    )
    if not match:
        return "", ""
    value = match.group(1).strip()
    if not value:
        return "", ""
    command, _, remainder = value.partition(" ")
    return command.casefold(), remainder.strip()


def _ambiguity_commands(
    section: str,
    review: PlainReadingReview,
) -> dict[str, str]:
    commands: dict[str, str] = {}
    structural = _without_fenced_code_content(section)
    for index, ambiguity in enumerate(review.ambiguities):
        marker = re.search(
            rf"^###\s+{re.escape(ambiguity.ambiguity_id)}\s*$",
            structural,
            re.MULTILINE,
        )
        if marker is None:
            raise ValueError(f"缺少歧义问题 {ambiguity.ambiguity_id}。")
        next_positions = [
            found.start()
            for found in re.finditer(r"^###\s+", structural[marker.end() :], re.MULTILINE)
        ]
        end = (
            marker.end() + next_positions[0]
            if next_positions
            else structural.find("替换语义", marker.end())
        )
        if end < 0:
            end = len(section)
        axis_section = structural[marker.end() : end]
        match = re.search(
            r"^用户填写\s*:\s*(.*?)\s*$", axis_section, re.MULTILINE
        )
        value = match.group(1).strip() if match else ""
        normalized = value.casefold()
        allowed = {option.option_id.casefold() for option in ambiguity.options}
        if normalized in allowed:
            selected = next(
                option.option_id
                for option in ambiguity.options
                if option.option_id.casefold() == normalized
            )
            commands[ambiguity.ambiguity_id] = selected
        elif normalized in {"n", "u", "r"}:
            commands[ambiguity.ambiguity_id] = normalized
        else:
            raise ValueError(
                f"{ambiguity.ambiguity_id} 用户填写必须为 A、B、N、U 或 R。"
            )
    return commands


def _section_requests_replacement(section: str) -> bool:
    return bool(
        re.search(
            r"^用户填写\s*:\s*r\s*$",
            _without_fenced_code_content(section),
            re.MULTILINE | re.IGNORECASE,
        )
    )


def _replacement_text(section: str) -> str:
    start = section.find("<!-- replacement:start -->")
    end = section.find("<!-- replacement:end -->", start + 1)
    if start < 0 or end < 0:
        return ""
    envelope = section[start:end]
    match = re.search(r"```(?:text)?\s*\r?\n([\s\S]*?)\r?\n```", envelope)
    return match.group(1).strip() if match else ""


def _preflight_no_ambiguity_review(
    sections: Mapping[str, str],
    pending_ids: Sequence[str],
    *,
    state: Mapping[str, Any],
    checkpoint_dir: str,
) -> dict[str, tuple[SurfaceSegmentation, PlainReadingReview, str, str]]:
    """Validate the complete static submission before any item is committed."""

    prepared: dict[
        str, tuple[SurfaceSegmentation, PlainReadingReview, str, str]
    ] = {}
    errors: list[str] = []
    for requirement_id in pending_ids:
        section = sections.get(requirement_id)
        if section is None:
            errors.append(f"{requirement_id}: 审核文件缺少区块。")
            continue
        item = state["items"][requirement_id]
        try:
            _validate_section_metadata(section, item)
            _validate_action_contract(section, "normal-no-axis-v1")
            segmentation, review = _load_pending_artifacts(
                checkpoint_dir, requirement_id, item
            )
            if review.review_basis == SEMANTIC_FALLBACK_REVIEW_BASIS:
                raise ValueError("semantic fallback项目必须使用专用Y/R/C审核文件。")
            command, comment = _no_ambiguity_command(section)
            if command not in {"a", "e", "c"}:
                raise ValueError("用户填写只允许 A、E <歧义关注说明> 或 C。")
            if command == "e" and not comment:
                raise ValueError("E 后必须填写非权威的歧义关注说明。")
            prepared[requirement_id] = (
                segmentation,
                review,
                command,
                comment,
            )
        except (ValueError, TypeError, PlainReadingValidationError) as exc:
            errors.append(f"{requirement_id}: {exc}")
    if errors:
        raise ReviewFileInputError(
            "无歧义审核文件预检失败，未应用任何项目：\n" + "\n".join(errors)
        )
    return prepared


def _preflight_semantic_fallback_review(
    sections: Mapping[str, str],
    pending_ids: Sequence[str],
    *,
    state: Mapping[str, Any],
    checkpoint_dir: str,
) -> dict[
    str,
    tuple[
        SurfaceSegmentation,
        PlainReadingReview,
        str,
        str,
        Any | None,
        tuple[APInventory | OverrideAPComparison, ...],
        tuple[APInventory | OverrideAPComparison, ...],
    ],
]:
    prepared: dict[
        str,
        tuple[
            SurfaceSegmentation,
            PlainReadingReview,
            str,
            str,
            Any | None,
            tuple[APInventory | OverrideAPComparison, ...],
            tuple[APInventory | OverrideAPComparison, ...],
        ],
    ] = {}
    errors: list[str] = []
    for requirement_id in pending_ids:
        section = sections.get(requirement_id)
        if section is None:
            errors.append(f"{requirement_id}: 审核文件缺少区块。")
            continue
        item = state["items"][requirement_id]
        try:
            _validate_section_metadata(section, item)
            expected_contract = (
                "semantic-override-confirmation-v1"
                if item.get("status") == "awaiting_semantic_override_confirmation"
                else "semantic-fallback-decision-v1"
            )
            _validate_action_contract(section, expected_contract)
            segmentation, review = _load_pending_artifacts(
                checkpoint_dir, requirement_id, item
            )
            command, _comment = _no_ambiguity_command(section)
            replacement = ""
            override = None
            ap_artifacts: list[APInventory | OverrideAPComparison] = []
            ap_artifacts: list[APInventory | OverrideAPComparison] = []
            if item.get("status") == "awaiting_semantic_fallback_review":
                if (
                    review.overall_status != "pending_no_ambiguity_confirmation"
                    or review.review_basis != SEMANTIC_FALLBACK_REVIEW_BASIS
                ):
                    raise ValueError("fallback state与plain-reading artifact不一致。")
                if command not in {"y", "r", "c"}:
                    raise ValueError("semantic fallback用户填写只允许 Y、R 或 C。")
                if command == "r":
                    replacement = _replacement_text(section)
                    if not replacement:
                        raise ValueError("R 必须填写完整replacement。")
                    override = create_human_semantic_override(
                        requirement_id=requirement_id,
                        original_source_text=_item_override_source_text(item),
                        replacement_text=replacement,
                        parent_effective_source_hash=str(item["source_hash"]),
                        reviewer_id="batch-markdown-reviewer",
                        original_segmentation=segmentation,
                        representation_profile=_item_ap_profile(item),
                        declared_catalog=_item_declared_catalog(item),
                        original_inventory=_item_current_ap_inventory(
                            checkpoint_dir, item
                        ),
                        ap_artifact_callback=ap_artifacts.append,
                    )
                    if override.status not in {
                        "awaiting_confirmation",
                        "awaiting_ap_inventory",
                        "awaiting_ap_mapping_confirmation",
                        "ap_rejected",
                    }:
                        raise ValueError(
                            "replacement未通过AP一致性检查: "
                            + ", ".join(override.issue_codes)
                        )
            elif item.get("status") == "awaiting_semantic_override_confirmation":
                if command not in {"confirm", "reject", "c"}:
                    raise ValueError("override确认只允许 CONFIRM、REJECT 或 C。")
                override = load_human_semantic_override_checkpoint(
                    checkpoint_dir,
                    requirement_id,
                    pending_only=True,
                    expected_original_source_hash=source_hash(
                        _item_override_source_text(item)
                    ),
                )
                if override is None or override.status != "awaiting_confirmation":
                    raise ValueError("缺少等待确认的semantic override checkpoint。")
                if _read_field(section, "override_id") != override.override_id:
                    raise ValueError("override_id 已过期或被修改。")
                replacement = override.replacement_text
            else:
                raise ValueError("项目不属于semantic fallback审核状态。")
            prepared[requirement_id] = (
                segmentation,
                review,
                command,
                replacement,
                override,
                tuple(ap_artifacts),
            )
        except (ValueError, TypeError, PlainReadingValidationError) as exc:
            errors.append(f"{requirement_id}: {exc}")
    if errors:
        raise ReviewFileInputError(
            "semantic fallback审核文件预检失败，未应用任何项目：\n"
            + "\n".join(errors)
        )
    return prepared


def _preflight_ap_inventory_review(
    sections: Mapping[str, str],
    pending_ids: Sequence[str],
    *,
    state: Mapping[str, Any],
    checkpoint_dir: str,
) -> dict[
    str,
    tuple[
        str,
        HumanSemanticOverride | None,
        tuple[APInventory | OverrideAPComparison, ...],
    ],
]:
    prepared: dict[
        str,
        tuple[
            str,
            HumanSemanticOverride | None,
            tuple[APInventory | OverrideAPComparison, ...],
        ],
    ] = {}
    errors: list[str] = []
    for requirement_id in pending_ids:
        section = sections.get(requirement_id)
        if section is None:
            errors.append(f"{requirement_id}: 审核文件缺少区块。")
            continue
        item = state["items"][requirement_id]
        try:
            _validate_section_metadata(section, item)
            _validate_action_contract(section, "semantic-override-ap-inventory-v1")
            segmentation, _reading = _load_pending_artifacts(
                checkpoint_dir, requirement_id, item
            )
            override = _load_pending_override_for_item(
                checkpoint_dir, requirement_id, item, "awaiting_ap_inventory"
            )
            original, replacement, comparison = (
                load_human_semantic_override_ap_artifacts(checkpoint_dir, override)
            )
            if original is None or replacement is None or comparison is None:
                raise ValueError("AP inventory artifact链不完整。")
            _validate_ap_review_metadata(
                section,
                override=override,
                original=original,
                replacement=replacement,
                comparison=comparison,
            )
            command, _ = _no_ambiguity_command(section)
            if command not in {"approve", "revise", "reject", "c"}:
                raise ValueError(
                    "AP inventory用户填写只允许 APPROVE、REVISE、REJECT 或 C。"
                )
            if command == "c":
                prepared[requirement_id] = (command, None, ())
                continue
            if command == "reject":
                prepared[requirement_id] = (
                    command,
                    cancel_human_semantic_override_ap_review(
                        override, reviewer_id="batch-markdown-reviewer"
                    ),
                    (),
                )
                continue
            profile = _item_ap_profile(item)
            if profile is None or profile.profile_id != override.representation_profile_id:
                raise ValueError("AP representation profile缺失或已过期。")
            payload = _structured_json_envelope(section, "ap-inventory-json")
            reviewed_original = _reviewed_inventory_from_payload(
                payload,
                side="original",
                existing=original,
                profile=profile,
                segmentation=segmentation,
                parent_inventory_id=original.parent_inventory_id,
            )
            if command == "revise":
                reviewed_replacement = _reviewed_inventory_from_payload(
                    payload,
                    side="replacement",
                    existing=replacement,
                    profile=profile,
                    segmentation=None,
                    parent_inventory_id=reviewed_original.inventory_id,
                )
                updated, updated_comparison = (
                    advance_human_semantic_override_ap_review(
                        override,
                        representation_profile=profile,
                        original_inventory=reviewed_original,
                        replacement_inventory=reviewed_replacement,
                        reviewer_id="batch-markdown-reviewer",
                        reviewer_source="batch_markdown_ap_inventory_revision",
                    )
                )
                prepared[requirement_id] = (
                    command,
                    updated,
                    (reviewed_original, reviewed_replacement, updated_comparison),
                )
                continue
            approved_original = approve_ap_inventory(
                reviewed_original,
                profile=profile,
                reviewer_id="batch-markdown-reviewer",
                reviewer_source="batch_markdown_ap_inventory",
                approved_empty=bool(payload.get("original_approved_empty", False)),
                approved_empty_reason=str(
                    payload.get("original_approved_empty_reason", "")
                ).strip(),
                segmentation=segmentation,
            )
            reviewed_replacement = _reviewed_inventory_from_payload(
                payload,
                side="replacement",
                existing=replacement,
                profile=profile,
                segmentation=None,
                parent_inventory_id=approved_original.inventory_id,
            )
            approved_replacement = approve_ap_inventory(
                reviewed_replacement,
                profile=profile,
                reviewer_id="batch-markdown-reviewer",
                reviewer_source="batch_markdown_ap_inventory",
                approved_empty=bool(
                    payload.get("replacement_approved_empty", False)
                ),
                approved_empty_reason=str(
                    payload.get("replacement_approved_empty_reason", "")
                ).strip(),
            )
            updated, updated_comparison = advance_human_semantic_override_ap_review(
                override,
                representation_profile=profile,
                original_inventory=approved_original,
                replacement_inventory=approved_replacement,
                reviewer_id="batch-markdown-reviewer",
                reviewer_source="batch_markdown_ap_inventory",
            )
            prepared[requirement_id] = (
                command,
                updated,
                (approved_original, approved_replacement, updated_comparison),
            )
        except (ValueError, TypeError, PlainReadingValidationError) as exc:
            errors.append(f"{requirement_id}: {exc}")
    if errors:
        raise ReviewFileInputError(
            "AP inventory审核文件预检失败，未应用任何项目：\n"
            + "\n".join(errors)
        )
    return prepared


def _preflight_ap_mapping_review(
    sections: Mapping[str, str],
    pending_ids: Sequence[str],
    *,
    state: Mapping[str, Any],
    checkpoint_dir: str,
) -> dict[
    str,
    tuple[
        str,
        HumanSemanticOverride | None,
        tuple[OverrideAPComparison, ...],
    ],
]:
    prepared: dict[
        str,
        tuple[
            str,
            HumanSemanticOverride | None,
            tuple[OverrideAPComparison, ...],
        ],
    ] = {}
    errors: list[str] = []
    for requirement_id in pending_ids:
        section = sections.get(requirement_id)
        if section is None:
            errors.append(f"{requirement_id}: 审核文件缺少区块。")
            continue
        item = state["items"][requirement_id]
        try:
            _validate_section_metadata(section, item)
            _validate_action_contract(section, "semantic-override-ap-mapping-v1")
            override = _load_pending_override_for_item(
                checkpoint_dir,
                requirement_id,
                item,
                "awaiting_ap_mapping_confirmation",
            )
            original, replacement, comparison = (
                load_human_semantic_override_ap_artifacts(checkpoint_dir, override)
            )
            if original is None or replacement is None or comparison is None:
                raise ValueError("AP mapping artifact链不完整。")
            _validate_ap_review_metadata(
                section,
                override=override,
                original=original,
                replacement=replacement,
                comparison=comparison,
                require_comparison_hash=True,
            )
            command, _ = _no_ambiguity_command(section)
            if command not in {"approve", "reject", "c"}:
                raise ValueError("AP mapping用户填写只允许 APPROVE、REJECT 或 C。")
            if command == "c":
                prepared[requirement_id] = (command, None, ())
                continue
            if command == "reject":
                prepared[requirement_id] = (
                    command,
                    cancel_human_semantic_override_ap_review(
                        override, reviewer_id="batch-markdown-reviewer"
                    ),
                    (),
                )
                continue
            payload = _structured_json_envelope(section, "ap-mapping-json")
            mappings = payload.get("identity_mapping")
            if not isinstance(mappings, list):
                raise ValueError("identity_mapping必须是JSON数组。")
            profile = _item_ap_profile(item)
            if profile is None or profile.profile_id != override.representation_profile_id:
                raise ValueError("AP representation profile缺失或已过期。")
            updated, updated_comparison = advance_human_semantic_override_ap_review(
                override,
                representation_profile=profile,
                original_inventory=original,
                replacement_inventory=replacement,
                identity_mapping=mappings,
                reviewer_id="batch-markdown-reviewer",
                reviewer_source="batch_markdown_ap_mapping",
            )
            if updated.status != "awaiting_confirmation":
                raise ValueError(
                    "AP mapping未形成完整一一映射: "
                    + ", ".join(updated.issue_codes)
                )
            prepared[requirement_id] = (
                command,
                updated,
                (updated_comparison,),
            )
        except (ValueError, TypeError, PlainReadingValidationError) as exc:
            errors.append(f"{requirement_id}: {exc}")
    if errors:
        raise ReviewFileInputError(
            "AP mapping审核文件预检失败，未应用任何项目：\n"
            + "\n".join(errors)
        )
    return prepared


def _structured_json_envelope(section: str, language: str) -> Mapping[str, Any]:
    matches = re.findall(
        rf"```{re.escape(language)}\s*\r?\n([\s\S]*?)\r?\n```",
        section,
    )
    if len(matches) != 1:
        raise ValueError(f"必须且只能提供一个 {language} envelope。")
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{language}不是有效JSON: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"{language} schema_version必须为1。")
    return payload


def _reviewed_inventory_from_payload(
    payload: Mapping[str, Any],
    *,
    side: str,
    existing: APInventory,
    profile: APRepresentationProfile,
    segmentation: SurfaceSegmentation | None,
    parent_inventory_id: str,
) -> APInventory:
    proposals = payload.get(side)
    if not isinstance(proposals, list) or any(
        not isinstance(item, Mapping) for item in proposals
    ):
        raise ValueError(f"{side} inventory必须是JSON对象数组。")
    if profile.mode in {
        "natural_language_grounded",
        "undetermined",
        "offline_formula_assisted",
    }:
        return propose_grounded_ap_inventory(
            requirement_id=existing.requirement_id,
            source_text=existing.source_text,
            profile=profile,
            proposals=proposals,
            segmentation=segmentation,
            extraction_mode=(
                "offline_formula_assisted_v1"
                if profile.mode == "offline_formula_assisted"
                else "manual_grounded_inventory_v1"
            ),
            proposal_source="batch_markdown_ap_inventory",
            parent_inventory_id=parent_inventory_id,
        )
    if proposals != _editable_inventory_proposals(existing):
        raise ValueError(
            "controlled/declared profile的identity由profile或catalog决定，不能在此改写。"
        )
    if existing.status == "approved":
        raise ValueError("已批准inventory不应重新进入inventory review。")
    return existing


def _validate_ap_review_metadata(
    section: str,
    *,
    override: HumanSemanticOverride,
    original: APInventory,
    replacement: APInventory,
    comparison: OverrideAPComparison,
    require_comparison_hash: bool = False,
) -> None:
    expected = {
        "override_id": override.override_id,
        "original_inventory_id": original.inventory_id,
        "original_inventory_hash": original.artifact_hash,
        "replacement_inventory_id": replacement.inventory_id,
        "replacement_inventory_hash": replacement.artifact_hash,
        "comparison_id": comparison.comparison_id,
    }
    if require_comparison_hash:
        expected["comparison_hash"] = comparison.artifact_hash
    for field, value in expected.items():
        if _read_field(section, field) != value:
            raise ValueError(f"{field} 已过期或被修改。")


def _preflight_ambiguity_review(
    sections: Mapping[str, str],
    pending_ids: Sequence[str],
    *,
    state: Mapping[str, Any],
    checkpoint_dir: str,
) -> dict[
    str,
    tuple[
        SurfaceSegmentation,
        PlainReadingReview,
        dict[str, str],
        str,
        Any | None,
        tuple[APInventory | OverrideAPComparison, ...],
    ],
]:
    prepared: dict[
        str,
        tuple[
            SurfaceSegmentation,
            PlainReadingReview,
            dict[str, str],
            str,
            Any | None,
            tuple[APInventory | OverrideAPComparison, ...],
        ],
    ] = {}
    errors: list[str] = []
    for requirement_id in pending_ids:
        section = sections.get(requirement_id)
        if section is None:
            errors.append(f"{requirement_id}: 审核文件缺少区块。")
            continue
        item = state["items"][requirement_id]
        try:
            _validate_section_metadata(section, item)
            _validate_action_contract(section, "ambiguity-adjudication-v2")
            segmentation, review = _load_pending_artifacts(
                checkpoint_dir, requirement_id, item
            )
            replacement = ""
            override = None
            ap_artifacts: list[APInventory | OverrideAPComparison] = []
            if _section_requests_replacement(section):
                _validate_ambiguity_replacement_commands(section, review)
                replacement = _replacement_text(section)
                if not replacement:
                    raise ValueError("选择 r 时必须填写完整替换语义。")
                override = create_human_semantic_override(
                    requirement_id=requirement_id,
                    original_source_text=_item_override_source_text(item),
                    replacement_text=replacement,
                    parent_effective_source_hash=str(item["source_hash"]),
                    reviewer_id="batch-markdown-reviewer",
                    original_segmentation=segmentation,
                    representation_profile=_item_ap_profile(item),
                    declared_catalog=_item_declared_catalog(item),
                    original_inventory=_item_current_ap_inventory(
                        checkpoint_dir, item
                    ),
                    ap_artifact_callback=ap_artifacts.append,
                )
                if override.status not in {
                    "awaiting_confirmation",
                    "awaiting_ap_inventory",
                    "awaiting_ap_mapping_confirmation",
                    "ap_rejected",
                }:
                    raise ValueError(
                        "替换语义未通过AP一致性检查: "
                        + ", ".join(override.issue_codes)
                    )
                decisions: dict[str, str] = {}
            else:
                decisions = _ambiguity_commands(section, review)
            prepared[requirement_id] = (
                segmentation,
                review,
                decisions,
                replacement,
                override,
                tuple(ap_artifacts),
            )
        except (ValueError, TypeError, PlainReadingValidationError) as exc:
            errors.append(f"{requirement_id}: {exc}")
    if errors:
        raise ReviewFileInputError(
            "歧义审核文件预检失败，未应用任何项目：\n" + "\n".join(errors)
        )
    return prepared


def _validate_ambiguity_replacement_commands(
    section: str,
    review: PlainReadingReview,
) -> None:
    values: list[str] = []
    structural = _without_fenced_code_content(section)
    for ambiguity in review.ambiguities:
        marker = re.search(
            rf"^###\s+{re.escape(ambiguity.ambiguity_id)}\s*$",
            structural,
            re.MULTILINE,
        )
        if marker is None:
            raise ValueError(f"缺少歧义问题 {ambiguity.ambiguity_id}。")
        next_marker = re.search(r"^###\s+", structural[marker.end() :], re.MULTILINE)
        end = (
            marker.end() + next_marker.start()
            if next_marker is not None
            else structural.find("替换语义", marker.end())
        )
        if end < 0:
            end = len(section)
        match = re.search(
            r"^用户填写\s*:\s*(.*?)\s*$",
            structural[marker.end() : end],
            re.MULTILINE,
        )
        values.append(match.group(1).strip().casefold() if match else "")
    if values.count("r") != 1 or any(value not in {"r", "pending"} for value in values):
        raise ValueError("replacement要求恰好一个轴填写 r，其余轴保持 pending。")


def _validate_section_metadata(section: str, item: Mapping[str, Any]) -> None:
    for field in ("requirement_id", "source_hash", "segmentation_id", "reading_review_id"):
        value = _read_field(section, field)
        expected = str(item.get(field, ""))
        if value != expected:
            raise ValueError(f"{field} 已过期或被修改。")


def _validate_action_contract(section: str, expected: str) -> None:
    value = _read_field(section, "action_contract_version")
    if value != expected:
        raise ValueError("action_contract_version 已过期或与审核文件类型不匹配。")


def _require_review_complete(text: str) -> None:
    if _read_field(text, "review_complete").casefold() not in {"yes", "true", "是"}:
        raise ValueError("请检查文件后把 review_complete 改为 yes。")


def _require_active_review_file(
    submitted_file: str,
    state: Mapping[str, Any],
    *,
    path_key: str,
) -> Path:
    """Reject stale, unscoped, or cross-batch review files before parsing."""

    active_value = str(state.get(path_key, "")).strip()
    if not active_value:
        raise ReviewFileInputError("当前批次没有该类型的待审核文件。")
    if not _same_path(submitted_file, active_value):
        raise ReviewFileInputError(
            "提交的审核文件不是当前checkpoint的活动审核文件。\n"
            f"当前活动文件: {active_value}"
        )
    active_path = Path(active_value)
    if not active_path.is_file():
        raise ReviewFileInputError("当前活动审核文件不存在: " + active_value)
    return active_path


def _validate_review_document(
    text: str,
    *,
    review_type: str,
    round_number: int,
    requirement_ids: Sequence[str],
) -> None:
    """Validate the file envelope and exact pending-item coverage."""

    errors: list[str] = []
    if _read_field(text, "review_type") != review_type:
        errors.append("review_type 与当前审核命令不匹配。")
    if _read_field(text, "schema_version") != str(PRE_IR_REVIEW_SCHEMA_VERSION):
        errors.append("schema_version 已过期或不受支持。")
    if _read_field(text, "round") != str(round_number):
        errors.append("round 已过期或不匹配。")
    if _read_field(text, "review_complete").casefold() not in {
        "yes",
        "true",
        "是",
    }:
        errors.append("请检查文件后把 review_complete 改为 yes。")
    structural = _without_fenced_code_content(text)
    headings = [
        match.group(1).upper()
        for match in re.finditer(
            r"^##\s+([A-Za-z][A-Za-z0-9_-]*-\d+)\s*$",
            structural,
            re.MULTILINE,
        )
    ]
    duplicates = sorted(
        requirement_id
        for requirement_id in set(headings)
        if headings.count(requirement_id) > 1
    )
    expected = set(requirement_ids)
    actual = set(headings)
    if duplicates:
        errors.append("存在重复requirement区块: " + ", ".join(duplicates))
    if expected - actual:
        errors.append("缺少待审核区块: " + ", ".join(sorted(expected - actual)))
    if actual - expected:
        errors.append("包含非当前待审核区块: " + ", ".join(sorted(actual - expected)))
    if errors:
        raise ReviewFileInputError("审核文件预检失败：\n" + "\n".join(errors))


def _requirement_sections(text: str) -> dict[str, str]:
    structural = _without_fenced_code_content(text)
    matches = list(
        re.finditer(
            r"^##\s+([A-Za-z][A-Za-z0-9_-]*-\d+)\s*$",
            structural,
            re.MULTILINE,
        )
    )
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        requirement_id = match.group(1).upper()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[requirement_id] = text[match.end() : end]
    return sections


def _read_field(text: str, field: str) -> str:
    match = re.search(
        rf"^{re.escape(field)}\s*:\s*(.*?)\s*$",
        _without_fenced_code_content(text),
        re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _without_fenced_code_content(text: str) -> str:
    """Mask fenced payloads while preserving offsets used by structural parsers."""

    def mask(match: re.Match[str]) -> str:
        value = match.group(0)
        return "".join("\n" if character == "\n" else " " for character in value)

    return re.sub(r"```[^\r\n]*\r?\n[\s\S]*?\r?\n```", mask, text)


def _pending_ids(state: Mapping[str, Any], kind: str) -> list[str]:
    targets = {
        "no_ambiguity": {"awaiting_no_ambiguity_review"},
        "semantic_fallback": {
            "awaiting_semantic_fallback_review",
            "awaiting_source_modality_choice",
            "awaiting_semantic_override_confirmation",
        },
        "ap_inventory": {"awaiting_semantic_override_ap_inventory"},
        "ap_mapping": {"awaiting_semantic_override_ap_mapping"},
        "ambiguity": {"awaiting_ambiguity_review"},
    }.get(kind)
    if targets is None:
        raise ValueError("unsupported deferred review kind: " + kind)
    return [
        requirement_id
        for requirement_id, item in state.get("items", {}).items()
        if item.get("status") in targets
    ]


def _close_if_complete(state: dict[str, Any]) -> None:
    items_by_id = state.get("items", {})
    items = list(items_by_id.values())
    state["no_ambiguity_review_complete"] = not bool(
        _pending_ids(state, "no_ambiguity")
    )
    state["semantic_fallback_review_complete"] = not bool(
        _pending_ids(state, "semantic_fallback")
    )
    state["ap_inventory_review_complete"] = not bool(
        _pending_ids(state, "ap_inventory")
    )
    state["ap_mapping_review_complete"] = not bool(
        _pending_ids(state, "ap_mapping")
    )
    state["ambiguity_review_complete"] = not bool(
        _pending_ids(state, "ambiguity")
    )
    terminal_statuses = {
        "ready_for_single_ir",
        "plain_reading_need_view",
        "deferred_source_clarification",
        "blocked",
    }
    complete = (
        bool(state.get("generation_complete"))
        and bool(items)
        and all(item.get("status") in terminal_statuses for item in items)
    )
    eligible = sorted(
        requirement_id
        for requirement_id, item in items_by_id.items()
        if item.get("status") == "ready_for_single_ir"
    )
    excluded = sorted(
        requirement_id
        for requirement_id, item in items_by_id.items()
        if item.get("status") != "ready_for_single_ir"
    )
    state["review_cycle_closed"] = complete
    state["downstream_eligible"] = eligible
    state["downstream_excluded"] = excluded
    state["pre_ir_closed"] = complete
    state["run_status"] = (
        "ready_for_downstream"
        if complete and eligible
        else "review_cycle_closed_no_eligible"
        if complete
        else "awaiting_pre_ir_review"
    )
    state["updated_at"] = _timestamp()


def _pending_files(state: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(value)
        for value in (
            state.get("pending_replacement_review_file", ""),
            state.get("pending_no_ambiguity_file", ""),
            state.get("pending_semantic_fallback_file", ""),
            state.get("pending_ap_inventory_file", ""),
            state.get("pending_ap_mapping_file", ""),
            state.get("pending_ambiguity_file", ""),
        )
        if value
    )


def _pending_replacement_review_ids_from_state(
    state: Mapping[str, Any],
) -> tuple[str, ...]:
    items = state.get("items", {})
    if not isinstance(items, Mapping):
        raise PreIRStateConsistencyError(
            "canonical Pre-IR items mapping is malformed"
        )
    return tuple(
        sorted(
            str(requirement_id)
            for requirement_id, item in items.items()
            if isinstance(item, Mapping)
            and item.get("status")
            == "awaiting_replacement_segmentation_review"
        )
    )


def _replacement_queue_item_fingerprint(item: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(item.get(key, ""))
        for key in (
            "status",
            "source_hash",
            "active_override_id",
            "segmentation_id",
            "reading_review_id",
        )
    )


def _replacement_review_queue_outcome(
    state: Mapping[str, Any],
    *,
    initial_ids: tuple[str, ...],
    attempted_ids: tuple[str, ...],
    completed_ids: tuple[str, ...],
    halt_reason: str,
    had_errors: bool = False,
) -> ReplacementReviewQueueOutcome:
    remaining_ids = _pending_replacement_review_ids_from_state(state)
    if had_errors:
        heading = "[PARTIAL] Replacement Pre-IR queue stopped with an item error."
    elif halt_reason == "interrupted":
        heading = "[WAIT] Replacement Pre-IR queue was interrupted."
    elif halt_reason in {"additional_review_required", "no_state_progress"}:
        heading = "[WAIT] Replacement Pre-IR queue requires additional human review."
    elif halt_reason == "empty":
        heading = "[OK] No pending replacement Pre-IR reviews."
    else:
        heading = "[OK] Pending replacement Pre-IR reviews completed."
    lines = [
        heading,
        f"Pending before continuous review: {len(initial_ids)}",
        f"Completed in this run: {len(completed_ids)}",
        f"Remaining replacement reviews: {len(remaining_ids)}",
    ]
    if completed_ids:
        lines.extend(("Completed requirements:", "  " + ", ".join(completed_ids)))
    lines.extend(_next_file_lines(state, include_commands=True))
    if state.get("pre_ir_closed"):
        lines.extend(
            (
                "Pre-IR review is complete; run the main command to continue.",
                "No downstream processing was started.",
            )
        )
    elif not remaining_ids and not _pending_files(state):
        lines.append("Other Pre-IR items are still pending.")
    return ReplacementReviewQueueOutcome(
        message="\n".join(lines),
        initial_ids=initial_ids,
        attempted_ids=attempted_ids,
        completed_ids=completed_ids,
        remaining_ids=remaining_ids,
        halt_reason=halt_reason,
        pre_ir_complete=bool(state.get("pre_ir_closed")),
        had_errors=had_errors,
    )


def _next_file_lines(
    state: Mapping[str, Any],
    *,
    include_commands: bool = False,
) -> list[str]:
    lines: list[str] = []
    no_file = str(state.get("pending_no_ambiguity_file", ""))
    fallback_file = str(state.get("pending_semantic_fallback_file", ""))
    inventory_file = str(state.get("pending_ap_inventory_file", ""))
    mapping_file = str(state.get("pending_ap_mapping_file", ""))
    ambiguity_file = str(state.get("pending_ambiguity_file", ""))
    replacement_file = str(
        state.get("pending_replacement_review_file", "")
    )
    checkpoint_dir = str(
        state.get("checkpoint_dir") or ".traj_ltlf_checkpoints"
    )
    replacement_segmentation_ids = _pending_replacement_review_ids_from_state(
        state
    )
    if replacement_segmentation_ids:
        lines.extend(
            (
                "Confirmed replacement requires manual surface review:",
                "  " + ", ".join(replacement_segmentation_ids),
            )
        )
        if include_commands:
            if replacement_file:
                lines.extend(
                    (
                        "Next replacement review file:",
                        "  " + replacement_file,
                        "python -B main.py --apply-pending-replacement-review "
                        + _command_arg(replacement_file)
                        + " --llm api --checkpoint-dir "
                        + _command_arg(checkpoint_dir)
                        + " --progress auto",
                    )
                )
            else:
                lines.extend(
                    (
                        "python -B main.py --export-pending-replacement-review "
                        "--llm api --checkpoint-dir "
                        + _command_arg(checkpoint_dir)
                        + " --progress auto",
                        "Compatible interactive mode:",
                        "python -B main.py --review-pending-replacements "
                        "--llm api --checkpoint-dir "
                        + _command_arg(checkpoint_dir)
                        + " --progress auto",
                    )
                )
    if no_file:
        lines.extend(("Next no-ambiguity review file:", f"  {no_file}"))
        if include_commands:
            lines.append(
                "python -B main.py --apply-no-ambiguity-review "
                + _command_arg(no_file)
                + " --llm api --checkpoint-dir "
                + _command_arg(checkpoint_dir)
                + " --progress auto"
            )
    if ambiguity_file:
        lines.extend(("Next ambiguity review file:", f"  {ambiguity_file}"))
        if include_commands:
            lines.append(
                "python -B main.py --apply-ambiguity-review "
                + _command_arg(ambiguity_file)
                + " --llm api --checkpoint-dir "
                + _command_arg(checkpoint_dir)
                + " --progress auto"
            )
    if fallback_file:
        insert_at = 0
        fallback_lines = [
            "Next semantic-fallback review file:",
            f"  {fallback_file}",
        ]
        if include_commands:
            fallback_lines.append(
                "python -B main.py --apply-semantic-fallback-review "
                + _command_arg(fallback_file)
                + " --llm api --checkpoint-dir "
                + _command_arg(checkpoint_dir)
                + " --progress auto"
            )
        lines[insert_at:insert_at] = fallback_lines
    if mapping_file:
        mapping_lines = [
            "Next semantic-override AP mapping review file:",
            f"  {mapping_file}",
        ]
        if include_commands:
            mapping_lines.append(
                "python -B main.py --apply-semantic-override-ap-mapping-review "
                + _command_arg(mapping_file)
                + " --checkpoint-dir "
                + _command_arg(checkpoint_dir)
                + " --progress auto"
            )
        lines[0:0] = mapping_lines
    if inventory_file:
        inventory_lines = [
            "Next semantic-override AP inventory review file:",
            f"  {inventory_file}",
        ]
        if include_commands:
            inventory_lines.append(
                "python -B main.py --apply-semantic-override-ap-inventory-review "
                + _command_arg(inventory_file)
                + " --checkpoint-dir "
                + _command_arg(checkpoint_dir)
                + " --progress auto"
            )
        lines[0:0] = inventory_lines
    return lines


def _command_arg(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:\\-]+", value):
        return value
    return '"' + value.replace('"', '\\"') + '"'


def _initial_message(state: Mapping[str, Any]) -> str:
    items = state.get("items", {})
    blocked_count = sum(
        1 for item in items.values() if item.get("status") == "blocked"
    )
    return "\n".join(
        (
            "Pre-IR processing completed.",
            f"Total requirements: {len(items)}",
            f"LLM detected ambiguity: {len(_pending_ids(state, 'ambiguity'))}",
            f"LLM detected no ambiguity: {len(_pending_ids(state, 'no_ambiguity'))}",
            f"Semantic fallback awaiting review: {len(_pending_ids(state, 'semantic_fallback'))}",
            f"AP inventories awaiting review: {len(_pending_ids(state, 'ap_inventory'))}",
            f"AP mappings awaiting review: {len(_pending_ids(state, 'ap_mapping'))}",
            f"Blocked preprocessing errors: {blocked_count}",
            "",
            *_next_file_lines(state, include_commands=True),
            *_preprocessing_error_lines(state, blocked_count),
            "No downstream IR or formula generation was started.",
        )
    )


def _pending_message(state: Mapping[str, Any]) -> str:
    blocked_ids = [
        requirement_id
        for requirement_id, item in state.get("items", {}).items()
        if item.get("status") == "blocked"
    ]
    lines = ["[WAIT] Pre-IR review is not complete."]
    lines.extend(_next_file_lines(state, include_commands=True))
    if blocked_ids:
        lines.extend(("Blocked preprocessing requirements:", "  " + ", ".join(blocked_ids)))
        lines.extend(_preprocessing_error_lines(state, len(blocked_ids)))
    lines.append("No downstream processing was started.")
    return "\n".join(lines)


def _preprocessing_error_lines(
    state: Mapping[str, Any], blocked_count: int
) -> list[str]:
    if not blocked_count:
        return []
    error_file = str(state.get("preprocessing_errors_file") or "")
    if not error_file:
        review_root = Path(
            str(state.get("review_namespace_dir") or "")
            or str(
                _review_namespace_root(
                    str(state.get("records_dir") or "review_records"),
                    str(state.get("checkpoint_dir") or ".traj_ltlf_checkpoints"),
                )
            )
        )
        error_file = str(review_root / PREPROCESSING_ERRORS_FILE)
    return [
        "Preprocessing error file:",
        "  " + error_file,
        (
            "Fix genuine source/human segmentation input errors before starting a new "
            "input batch. Software generation-contract upgrades automatically retry only "
            "the affected blocked items in the same checkpoint."
        ),
    ]


def _write_preprocessing_errors(records_dir: str, state: dict[str, Any]) -> None:
    blocked = [
        (requirement_id, item.get("error", ""))
        for requirement_id, item in state.get("items", {}).items()
        if item.get("status") == "blocked"
    ]
    review_root = Path(
        str(state.get("review_namespace_dir") or "")
        or str(
            _review_namespace_root(
                records_dir,
                str(state.get("checkpoint_dir") or ".traj_ltlf_checkpoints"),
            )
        )
    )
    path = review_root / PREPROCESSING_ERRORS_FILE
    if not blocked:
        if path.exists():
            path.unlink()
        state["preprocessing_errors_file"] = ""
        return
    lines = ["# Pre-IR预处理错误", ""]
    for requirement_id, error in blocked:
        lines.extend((f"## {requirement_id}", "", str(error), ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, "\n".join(lines))
    state["preprocessing_errors_file"] = str(path)


def _write_comparison(root: Path, comparison: SegmentationComparisonReport) -> None:
    path = root / "segmentation_comparison" / f"{comparison.requirement_id}.json"
    _atomic_write_json(path, comparison.to_dict())


def _load_state(root: Path) -> dict[str, Any]:
    path = root / PRE_IR_BATCH_STATE_FILE
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if _recover_item_transactions(root, payload):
        payload = json.loads(path.read_text(encoding="utf-8"))
    storage_paths_changed = _normalize_state_storage_paths(payload, root)
    if payload.get("schema_version") == 1:
        payload = _migrate_pre_ir_batch_state_v1(root, payload)
        storage_paths_changed = False
    elif payload.get("schema_version") == 2:
        payload = _migrate_pre_ir_batch_state_v2(root, payload)
        storage_paths_changed = False
    elif payload.get("schema_version") == 3:
        payload = _migrate_pre_ir_batch_state_v3(root, payload)
        storage_paths_changed = False
    elif payload.get("schema_version") == 4:
        payload = _migrate_pre_ir_batch_state_v4(root, payload)
        storage_paths_changed = False
    if payload.get("schema_version") != PRE_IR_BATCH_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported pre-IR batch state schema")
    # runtime fields are additive so old v4 checkpoints remain readable without
    # reinterpretation or a mandatory migration write.
    payload.setdefault("pending_replacement_review_file", "")
    payload.setdefault("replacement_review_round", 1)
    payload.setdefault("replacement_review_snapshot_id", "")
    override_fields_changed = _normalize_override_state_fields(payload, root)
    if payload.get("generation_complete") and _review_namespace_needs_migration(
        payload, root
    ):
        records_dir = str(payload.get("records_dir") or "review_records")
        _refresh_review_files(
            payload,
            checkpoint_dir=str(root),
            records_dir=records_dir,
            force=True,
        )
        _write_preprocessing_errors(records_dir, payload)
        _close_if_complete(payload)
        _atomic_write_json(path, payload)
        storage_paths_changed = False
    if storage_paths_changed or override_fields_changed:
        _atomic_write_json(path, payload)
    return payload


def _normalize_override_state_fields(
    state: dict[str, Any],
    root: Path,
) -> bool:
    """Migrate the former overloaded override_id into active/pending roles."""

    changed = False
    for requirement_id, raw_item in state.get("items", {}).items():
        if not isinstance(raw_item, dict):
            continue
        item = raw_item
        original = str(item.get("original_requirement", ""))
        effective = str(item.get("effective_requirement") or original)
        status = str(item.get("status", ""))
        legacy_id = str(item.get("override_id", ""))
        active = load_human_semantic_override_checkpoint(
            str(root),
            requirement_id,
            active_only=True,
        )
        active_id = str(item.get("active_override_id", ""))
        if effective != original and active is not None:
            active_id = active.override_id
        elif effective == original and status != (
            "awaiting_replacement_segmentation_review"
        ):
            active_id = ""

        pending_id = str(item.get("pending_override_id", ""))
        if status in {
            "awaiting_source_modality_choice",
            "awaiting_semantic_override_ap_inventory",
            "awaiting_semantic_override_ap_mapping",
            "awaiting_semantic_override_confirmation",
        }:
            if not pending_id and legacy_id and legacy_id != active_id:
                pending_id = legacy_id
        else:
            pending_id = ""

        normalized = {
            "active_override_id": active_id,
            "pending_override_id": pending_id,
            # Kept as a compatibility alias for the active effective source.
            "override_id": active_id,
        }
        for key, value in normalized.items():
            if item.get(key) != value:
                item[key] = value
                changed = True
    return changed


def _normalize_state_storage_paths(state: dict[str, Any], root: Path) -> bool:
    """Make persisted storage paths independent of the process working directory."""

    changed = False
    path_keys = (
        "pending_replacement_review_file",
        "pending_no_ambiguity_file",
        "pending_semantic_fallback_file",
        "pending_ap_inventory_file",
        "pending_ap_mapping_file",
        "pending_ambiguity_file",
    )
    records_root: Path | None = None

    # Existing active files are the strongest legacy evidence: they identify
    # the records location that actually contains human work, even if an old
    # relative records_dir would resolve differently under today's cwd.
    for path_key in path_keys:
        value = str(state.get(path_key, "")).strip()
        if not value:
            continue
        candidate = Path(value)
        if candidate.is_file():
            resolved = Path(_canonical_path(candidate))
            records_root = _records_root_from_review_path(resolved)
            break

    if records_root is None:
        namespace_value = str(state.get("review_namespace_dir", "")).strip()
        namespace = Path(namespace_value) if namespace_value else None
        if namespace is not None and namespace.is_absolute():
            records_root = _records_root_from_namespace(namespace)

    if records_root is None:
        records_value = str(state.get("records_dir") or "review_records")
        records_path = Path(records_value)
        if records_path.is_absolute():
            records_root = records_path
        else:
            # Legacy defaults were workspace-relative.  The checkpoint parent
            # is the only stable workspace anchor available after cwd changes.
            records_root = root.parent / records_path
    records_root = Path(_canonical_path(records_root))

    if str(state.get("records_dir", "")) != str(records_root):
        state["records_dir"] = str(records_root)
        changed = True
    canonical_checkpoint = _canonical_path(root)
    if str(state.get("checkpoint_dir", "")) != canonical_checkpoint:
        state["checkpoint_dir"] = canonical_checkpoint
        changed = True

    expected_namespace = _review_namespace_root(records_root, root)
    for path_key in path_keys:
        value = str(state.get(path_key, "")).strip()
        if not value:
            continue
        resolved = _resolve_legacy_review_path(
            value,
            records_root=records_root,
            expected_namespace=expected_namespace,
        )
        if value != str(resolved):
            state[path_key] = str(resolved)
            changed = True
    return changed


def _records_root_from_review_path(path: Path) -> Path:
    parent = path.parent
    if parent.parent.name == PRE_IR_BATCHES_DIR:
        return parent.parent.parent
    return parent


def _records_root_from_namespace(path: Path) -> Path:
    if path.parent.name == PRE_IR_BATCHES_DIR:
        return path.parent.parent
    return path


def _resolve_legacy_review_path(
    value: str,
    *,
    records_root: Path,
    expected_namespace: Path,
) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return Path(_canonical_path(path))
    parts = {part.casefold() for part in path.parts}
    if PRE_IR_BATCHES_DIR.casefold() in parts:
        candidate = expected_namespace / path.name
    else:
        candidate = records_root / path.name
    return Path(_canonical_path(candidate))


def _review_namespace_needs_migration(
    state: Mapping[str, Any],
    checkpoint_dir: str | Path,
) -> bool:
    records_dir = str(state.get("records_dir") or "review_records")
    expected_root = _review_namespace_root(records_dir, checkpoint_dir)
    namespace_value = str(state.get("review_namespace_dir", "")).strip()
    if not namespace_value or not _same_path(namespace_value, expected_root):
        return True
    has_blocked_items = any(
        isinstance(item, Mapping) and item.get("status") == "blocked"
        for item in state.get("items", {}).values()
    )
    has_error_file = bool(str(state.get("preprocessing_errors_file", "")).strip())
    if has_blocked_items != has_error_file:
        return True
    for path_key in (
        "pending_replacement_review_file",
        "pending_no_ambiguity_file",
        "pending_semantic_fallback_file",
        "pending_ap_inventory_file",
        "pending_ap_mapping_file",
        "pending_ambiguity_file",
        "preprocessing_errors_file",
    ):
        path_value = str(state.get(path_key, "")).strip()
        if path_value and not _same_path(Path(path_value).parent, expected_root):
            return True
        if (
            path_key == "preprocessing_errors_file"
            and path_value
            and not Path(path_value).is_file()
        ):
            return True
    return False


def _migrate_pre_ir_batch_state_v1(
    root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify legacy pending reviews and rebuild their derived Markdown views."""

    state = dict(payload)
    backups: list[str] = []
    for path_key in ("pending_no_ambiguity_file", "pending_ambiguity_file"):
        source_value = str(state.get(path_key, "")).strip()
        if not source_value:
            continue
        source_path = Path(source_value)
        if not source_path.is_file():
            continue
        backup_path = source_path.with_suffix(source_path.suffix + ".schema-v1.bak")
        if not backup_path.exists():
            shutil.copy2(source_path, backup_path)
        backups.append(str(backup_path))
    state["schema_version"] = PRE_IR_BATCH_STATE_SCHEMA_VERSION
    state.setdefault("semantic_fallback_round", 1)
    state.setdefault("pending_semantic_fallback_file", "")
    state.setdefault("ap_inventory_round", 1)
    state.setdefault("ap_mapping_round", 1)
    state.setdefault("pending_ap_inventory_file", "")
    state.setdefault("pending_ap_mapping_file", "")
    items = {
        str(requirement_id): dict(item)
        for requirement_id, item in state.get("items", {}).items()
        if isinstance(item, Mapping)
    }
    state["items"] = items
    state["migration_history"] = [
        *list(state.get("migration_history", ())),
        {
            "from_schema": 1,
            "to_schema": PRE_IR_BATCH_STATE_SCHEMA_VERSION,
            "migrated_at": _timestamp(),
            "review_file_backups": backups,
        },
    ]
    for requirement_id, item in items.items():
        if item.get("status") != "awaiting_no_ambiguity_review":
            continue
        reading_payload = load_plain_reading_generated_checkpoint(
            str(root),
            requirement_id,
            expected_source_hash=str(item.get("source_hash", "")),
            expected_segmentation_id=str(item.get("segmentation_id", "")),
            expected_reading_review_id=str(item.get("reading_review_id", "")),
        )
        reading_mapping = (
            reading_payload.get("plain_reading_review")
            if isinstance(reading_payload, Mapping)
            else None
        )
        if (
            isinstance(reading_mapping, Mapping)
            and reading_mapping.get("review_basis")
            == SEMANTIC_FALLBACK_REVIEW_BASIS
        ):
            item["status"] = "awaiting_semantic_fallback_review"
    _reconcile_migrated_plain_reading_readiness(root, state)
    state["no_ambiguity_review_complete"] = not bool(
        _pending_ids(state, "no_ambiguity")
    )
    state["semantic_fallback_review_complete"] = not bool(
        _pending_ids(state, "semantic_fallback")
    )
    state["ambiguity_review_complete"] = not bool(
        _pending_ids(state, "ambiguity")
    )
    records_dir = str(state.get("records_dir") or "review_records")
    _refresh_review_files(
        state,
        checkpoint_dir=str(root),
        records_dir=records_dir,
        force=True,
    )
    _write_preprocessing_errors(records_dir, state)
    _close_if_complete(state)
    _atomic_write_json(root / PRE_IR_BATCH_STATE_FILE, state)
    return state


def _migrate_pre_ir_batch_state_v2(
    root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Add runtime AP sub-stage views without reinterpreting old overrides."""

    state = dict(payload)
    state["schema_version"] = PRE_IR_BATCH_STATE_SCHEMA_VERSION
    state.setdefault("ap_inventory_round", 1)
    state.setdefault("ap_mapping_round", 1)
    state.setdefault("pending_ap_inventory_file", "")
    state.setdefault("pending_ap_mapping_file", "")
    state.setdefault("ap_inventory_review_complete", True)
    state.setdefault("ap_mapping_review_complete", True)
    state["migration_history"] = [
        *list(state.get("migration_history", ())),
        {
            "from_schema": 2,
            "to_schema": PRE_IR_BATCH_STATE_SCHEMA_VERSION,
            "migrated_at": _timestamp(),
            "semantic_change": (
                "runtime AP fields added; terminal item status reclassified only "
                "from persisted Plain Reading readiness"
            ),
            "llm_calls": 0,
        },
    ]
    _reconcile_migrated_plain_reading_readiness(root, state)
    if state.get("generation_complete"):
        records_dir = str(state.get("records_dir") or "review_records")
        _refresh_review_files(
            state,
            checkpoint_dir=str(root),
            records_dir=records_dir,
            force=True,
        )
        _close_if_complete(state)
    _atomic_write_json(root / PRE_IR_BATCH_STATE_FILE, state)
    return state


def _migrate_pre_ir_batch_state_v3(
    root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Add runtime readiness/deferred routing to the existing runtime v3."""

    state = dict(payload)
    state["items"] = {
        str(requirement_id): dict(item)
        for requirement_id, item in state.get("items", {}).items()
        if isinstance(item, Mapping)
    }
    _reconcile_migrated_plain_reading_readiness(root, state)
    state["schema_version"] = PRE_IR_BATCH_STATE_SCHEMA_VERSION
    state["migration_history"] = [
        *list(state.get("migration_history", ())),
        {
            "from_schema": 3,
            "to_schema": PRE_IR_BATCH_STATE_SCHEMA_VERSION,
            "migrated_at": _timestamp(),
            "migration": "plain-reading-readiness-v1",
            "llm_calls": 0,
        },
    ]
    if state.get("generation_complete"):
        records_dir = str(state.get("records_dir") or "review_records")
        _refresh_review_files(
            state,
            checkpoint_dir=str(root),
            records_dir=records_dir,
            force=True,
        )
    _close_if_complete(state)
    _atomic_write_json(root / PRE_IR_BATCH_STATE_FILE, state)
    return state


def _migrate_pre_ir_batch_state_v4(
    root: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Migrate broad approved/deferred states to explicit Plain Reading modes."""

    state = dict(payload)
    state["items"] = {
        str(requirement_id): dict(item)
        for requirement_id, item in state.get("items", {}).items()
        if isinstance(item, Mapping)
    }
    _reconcile_migrated_plain_reading_readiness(root, state)
    state["schema_version"] = PRE_IR_BATCH_STATE_SCHEMA_VERSION
    state["migration_history"] = [
        *list(state.get("migration_history", ())),
        {
            "from_schema": 4,
            "to_schema": PRE_IR_BATCH_STATE_SCHEMA_VERSION,
            "migrated_at": _timestamp(),
            "migration": "plain-reading-need-view-v1",
            "llm_calls": 0,
        },
    ]
    _close_if_complete(state)
    _atomic_write_json(root / PRE_IR_BATCH_STATE_FILE, state)
    return state


def _reconcile_migrated_plain_reading_readiness(
    root: Path,
    state: dict[str, Any],
) -> None:
    """Reclassify only formerly approved/deferred items, without any LLM call."""

    items = state.get("items", {})
    for requirement_id, item in items.items():
        if item.get("status") not in {
            "approved",
            "ready_for_single_ir",
            "deferred_plain_reading_resolution",
            "plain_reading_need_view",
            "deferred_source_clarification",
        }:
            continue
        failures = []
        for key in (
            "error",
            "generation_error",
            "last_apply_error",
            "last_apply_generation_error",
            "last_apply_diagnostics",
        ):
            value = item.get(key)
            if value:
                failures.append(
                    {
                        "field": key,
                        "record": value,
                        "superseded_by": "plain-reading-readiness-v1-migration",
                    }
                )
        if failures:
            item["migration_superseded_failures"] = [
                *list(item.get("migration_superseded_failures", ())),
                *failures,
            ]
        try:
            segmentation_payload = load_segmentation_reviewed_checkpoint(
                str(root),
                requirement_id,
                expected_source_hash=str(item.get("source_hash", "")),
                expected_segmentation_id=str(item.get("segmentation_id", "")),
            )
            reviewed_payload = load_plain_reading_reviewed_checkpoint(
                str(root),
                requirement_id,
                expected_source_hash=str(item.get("source_hash", "")),
                expected_segmentation_id=str(item.get("segmentation_id", "")),
                expected_reading_review_id=str(
                    item.get("reading_review_id", "")
                ),
            )
            segmentation_mapping = (
                segmentation_payload.get("surface_segmentation")
                if isinstance(segmentation_payload, Mapping)
                else None
            )
            review_mapping = (
                reviewed_payload.get("plain_reading_review")
                if isinstance(reviewed_payload, Mapping)
                else None
            )
            if not isinstance(segmentation_mapping, Mapping) or not isinstance(
                review_mapping, Mapping
            ):
                raise ValueError("reviewed Plain Reading artifact chain is missing")
            segmentation = segmentation_from_mapping(segmentation_mapping)
            review = plain_reading_from_mapping(review_mapping, segmentation)
            readiness = assess_plain_reading_ir_readiness(review, segmentation)
            if readiness.disposition == "invalid":
                raise ValueError("；".join(readiness.validation_errors))
            if not (
                readiness.can_enter_semantic_ir
                or readiness.can_attempt_candidate_ir
                or readiness.disposition == "source_clarification_required"
            ):
                raise ValueError(
                    "formerly terminal item now requires an unfinished human action"
                )
            descendant = (
                _semantic_descendant_for_reading(
                    root, requirement_id, review.reading_review_id
                )
                if not readiness.can_enter_semantic_ir
                else None
            )
            if descendant is not None:
                _write_semantic_quarantine_record(
                    root, requirement_id, descendant
                )
                item["migration_superseded_descendant"] = descendant
            _route_item_by_readiness(
                item, review, segmentation, checkpoint_dir=str(root),
            )
        except (TypeError, ValueError) as exc:
            item["status"] = "blocked"
            item["classification"] = "blocked_error"
            item["suggested_rerun_stage"] = "plain_reading"
            item["migration_error"] = {
                "reason_code": "PLAIN_READING_MIGRATION_ARTIFACT_INVALID",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "migrated_at": _timestamp(),
            }


def _require_state(root: Path) -> dict[str, Any]:
    state = _load_state(root)
    if not state or not state.get("generation_complete"):
        raise ValueError("尚未运行主体命令完成Pre-IR生成。")
    _validate_canonical_item_artifacts(state, root)
    if _active_review_views_inconsistent(state, checkpoint_dir=root):
        _refresh_review_files(
            state,
            checkpoint_dir=str(root),
            records_dir=str(state.get("records_dir") or "review_records"),
            force=True,
        )
        _close_if_complete(state)
        _write_state(root, state)
    return state


def _validate_canonical_item_artifacts(
    state: Mapping[str, Any],
    root: Path,
) -> None:
    """Fail closed when state points at missing or stale Pre-IR artifacts."""

    items_mapping = state.get("items", {})
    if not isinstance(items_mapping, Mapping):
        raise PreIRStateConsistencyError("canonical Pre-IR items mapping is malformed")
    items = list(items_mapping.values())
    terminal_statuses = {
        "ready_for_single_ir",
        "plain_reading_need_view",
        "deferred_source_clarification",
        "blocked",
    }
    expected_closed = (
        bool(state.get("generation_complete"))
        and bool(items)
        and all(
            isinstance(item, Mapping)
            and item.get("status") in terminal_statuses
            for item in items
        )
    )
    if bool(state.get("pre_ir_closed")) != expected_closed:
        raise PreIRStateConsistencyError(
            "canonical pre_ir_closed flag does not match item states"
        )
    expected_eligible = sorted(
        requirement_id
        for requirement_id, item in items_mapping.items()
        if (
            isinstance(item, Mapping)
            and item.get("status") == "ready_for_single_ir"
        )
    )
    expected_excluded = sorted(set(items_mapping) - set(expected_eligible))
    if sorted(state.get("downstream_eligible", ())) != expected_eligible:
        raise PreIRStateConsistencyError(
            "canonical downstream_eligible does not match approved items"
        )
    if sorted(state.get("downstream_excluded", ())) != expected_excluded:
        raise PreIRStateConsistencyError(
            "canonical downstream_excluded does not match non-approved items"
        )
    if state.get("downstream_started") and (
        not expected_closed
        or not any(
            isinstance(item, Mapping)
            and item.get("status") == "ready_for_single_ir"
            for item in items
        )
    ):
        raise PreIRStateConsistencyError(
            "downstream_started requires a closed review cycle and an approved item"
        )

    pending_artifact_statuses = {
        "awaiting_no_ambiguity_review",
        "awaiting_semantic_fallback_review",
        "awaiting_source_modality_choice",
        "awaiting_ambiguity_review",
        "awaiting_semantic_override_confirmation",
        "awaiting_semantic_override_ap_inventory",
        "awaiting_semantic_override_ap_mapping",
    }
    allowed_statuses = pending_artifact_statuses | {
        "ready_for_single_ir",
        "plain_reading_need_view",
        "deferred_source_clarification",
        "blocked",
        "awaiting_replacement_segmentation_review",
    }
    for requirement_id, raw_item in items_mapping.items():
        if not isinstance(raw_item, Mapping):
            raise PreIRStateConsistencyError(
                f"{requirement_id}: canonical Pre-IR item is malformed"
            )
        item = raw_item
        status = str(item.get("status", ""))
        if status not in allowed_statuses:
            raise PreIRStateConsistencyError(
                f"{requirement_id}: unsupported canonical Pre-IR status {status!r}"
            )
        original = str(item.get("original_requirement", ""))
        effective = str(item.get("effective_requirement") or original)
        if str(item.get("requirement_id", "")) != requirement_id:
            raise PreIRStateConsistencyError(
                f"{requirement_id}: item requirement_id does not match state key"
            )
        if status != "blocked" and str(item.get("source_hash", "")) != source_hash(
            effective
        ):
            raise PreIRStateConsistencyError(
                f"{requirement_id}: item source_hash does not match effective source"
            )
        try:
            active_current = load_human_semantic_override_checkpoint(
                str(root),
                requirement_id,
                active_only=True,
            )
        except (TypeError, ValueError) as exc:
            raise PreIRStateConsistencyError(
                f"{requirement_id}: active semantic override checkpoint is corrupt"
            ) from exc

        try:
            if status in pending_artifact_statuses:
                _load_pending_artifacts(str(root), requirement_id, item)
            elif status in {
                "ready_for_single_ir",
                "plain_reading_need_view",
                "deferred_source_clarification",
            }:
                segmentation_payload = load_segmentation_reviewed_checkpoint(
                    str(root),
                    requirement_id,
                    expected_source_hash=str(item.get("source_hash", "")),
                    expected_segmentation_id=str(item.get("segmentation_id", "")),
                )
                reviewed_payload = load_plain_reading_reviewed_checkpoint(
                    str(root),
                    requirement_id,
                    expected_source_hash=str(item.get("source_hash", "")),
                    expected_segmentation_id=str(item.get("segmentation_id", "")),
                    expected_reading_review_id=str(
                        item.get("reading_review_id", "")
                    ),
                )
                if segmentation_payload is None or reviewed_payload is None:
                    raise ValueError("approved reviewed artifact chain is missing or stale")
                segmentation_mapping = segmentation_payload.get(
                    "surface_segmentation"
                )
                reviewed_mapping = reviewed_payload.get("plain_reading_review")
                if not isinstance(segmentation_mapping, Mapping) or not isinstance(
                    reviewed_mapping, Mapping
                ):
                    raise ValueError("approved reviewed artifact chain is malformed")
                segmentation = segmentation_from_mapping(segmentation_mapping)
                reviewed = plain_reading_from_mapping(
                    reviewed_mapping, segmentation
                )
                readiness = assess_plain_reading_ir_readiness(
                    reviewed, segmentation
                )
                candidate_route = readiness.can_attempt_candidate_ir
                expected_status = (
                    "ready_for_single_ir"
                    if readiness.can_enter_semantic_ir or candidate_route
                    else "plain_reading_need_view"
                    if readiness.disposition == "semantic_resolution_required"
                    else "deferred_source_clarification"
                    if readiness.disposition == "source_clarification_required"
                    else ""
                )
                if status != expected_status:
                    raise ValueError(
                        f"item status {status!r} conflicts with Plain Reading "
                        f"readiness {readiness.disposition!r}"
                    )
                if candidate_route and (item.get("single_ir_ready") or item.get("finalization_allowed")):
                    raise ValueError("candidate IR routing cannot grant finalization authority")
                if status == "plain_reading_need_view":
                    view = load_plain_reading_need_view_checkpoint(
                        str(root),
                        requirement_id,
                        expected_source_hash=reviewed.source_hash,
                        expected_segmentation_id=reviewed.segmentation_id,
                        expected_reading_review_id=reviewed.reading_review_id,
                    )
                    if (
                        view is None
                        or view.need_view_id
                        != str(item.get("plain_reading_need_view_id", ""))
                        or view.finalization_allowed
                        or view.single_ir_ready
                    ):
                        raise ValueError(
                            "plain-reading need-view is missing, stale, or finalizable"
                        )
                descendant = (
                    _semantic_descendant_for_reading(
                        root, requirement_id, reviewed.reading_review_id
                    )
                    if not readiness.can_enter_semantic_ir
                    else None
                )
                if descendant is not None:
                    _write_semantic_quarantine_record(
                        root, requirement_id, descendant
                    )
                    raise ValueError(
                        "non-ready Plain Reading has a Semantic IR descendant"
                    )
        except (TypeError, ValueError) as exc:
            raise PreIRStateConsistencyError(
                f"{requirement_id}: canonical Pre-IR artifact chain is inconsistent: {exc}"
            ) from exc

        if status in {
            "awaiting_source_modality_choice",
            "awaiting_semantic_override_ap_inventory",
            "awaiting_semantic_override_ap_mapping",
            "awaiting_semantic_override_confirmation",
        }:
            pending = load_human_semantic_override_checkpoint(
                str(root),
                requirement_id,
                pending_only=True,
                expected_original_source_hash=source_hash(
                    _item_override_source_text(item)
                ),
            )
            if (
                pending is None
                or pending.override_id
                != str(item.get("pending_override_id", ""))
                or pending.parent_effective_source_hash
                != str(item.get("source_hash", ""))
                or pending.status
                != {
                    "awaiting_source_modality_choice": (
                        "awaiting_source_modality_choice"
                    ),
                    "awaiting_semantic_override_ap_inventory": "awaiting_ap_inventory",
                    "awaiting_semantic_override_ap_mapping": "awaiting_ap_mapping_confirmation",
                    "awaiting_semantic_override_confirmation": "awaiting_confirmation",
                }[status]
            ):
                raise PreIRStateConsistencyError(
                    f"{requirement_id}: pending semantic override is missing or stale"
                )

        if status == "awaiting_replacement_segmentation_review" and any(
            str(item.get(key, ""))
            for key in (
                "segmentation_id",
                "llm_segmentation_id",
                "reading_review_id",
            )
        ):
            raise PreIRStateConsistencyError(
                f"{requirement_id}: replacement surface review still references stale artifacts"
            )

        depends_on_active_override = (
            status == "awaiting_replacement_segmentation_review"
            or effective != original
        )
        if not depends_on_active_override:
            if active_current is not None:
                raise PreIRStateConsistencyError(
                    f"{requirement_id}: unreferenced active semantic override pointer"
                )
            if item.get("active_override_id") and status != "blocked":
                raise PreIRStateConsistencyError(
                    f"{requirement_id}: active_override_id has no effective source"
                )
            continue

        active = active_current
        if (
            active is None
            or active.override_id != str(item.get("active_override_id", ""))
            or active.requirement_id != requirement_id
            or active.status != "active"
            or active.replacement_text != effective
            or active.replacement_hash != str(item.get("source_hash", ""))
            or active.new_effective_source_hash != str(item.get("source_hash", ""))
        ):
            raise PreIRStateConsistencyError(
                f"{requirement_id}: active semantic override is missing or stale"
            )

        parent_records = [
            entry
            for entry in item.get("superseded_pre_ir_artifacts", ())
            if isinstance(entry, Mapping)
            and str(entry.get("superseded_by_override_id", ""))
            == active.override_id
        ]
        if parent_records:
            expected_parent_hash = str(parent_records[-1].get("source_hash", ""))
        else:
            expected_parent_hash = source_hash(original)
        if active.parent_effective_source_hash != expected_parent_hash:
            raise PreIRStateConsistencyError(
                f"{requirement_id}: active semantic override parent lineage is stale"
            )
        effective_inventory_id = str(item.get("effective_ap_inventory_id", ""))
        effective_inventory_hash = str(item.get("effective_ap_inventory_hash", ""))
        if effective_inventory_id or effective_inventory_hash:
            try:
                current_inventory = load_ap_inventory_checkpoint(
                    str(root),
                    requirement_id,
                    current=True,
                    expected_source_hash=str(item.get("source_hash", "")),
                    expected_profile_id=(
                        active.representation_profile_id or None
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise PreIRStateConsistencyError(
                    f"{requirement_id}: effective AP inventory pointer is corrupt or stale"
                ) from exc
            if (
                current_inventory is None
                or current_inventory.inventory_id != effective_inventory_id
                or current_inventory.artifact_hash != effective_inventory_hash
            ):
                raise PreIRStateConsistencyError(
                    f"{requirement_id}: effective AP inventory lineage is stale"
                )


def _semantic_descendant_for_reading(
    root: Path,
    requirement_id: str,
    reading_review_id: str,
) -> dict[str, Any] | None:
    """Return a matching semantic snapshot without treating it as usable."""

    path = root / "semantic_generated" / f"{requirement_id}.json"
    if not reading_review_id or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "path": str(path),
            "reading_review_id": reading_review_id,
            "status": "unreadable",
        }
    semantic = payload.get("semantic", {})
    faithful_ir = (
        semantic.get("faithful_ir", {}) if isinstance(semantic, Mapping) else {}
    )
    embedded = payload.get("plain_reading_review", {})
    parent_ids = {
        str(payload.get("reading_review_id", "")),
        (
            str(faithful_ir.get("reading_review_id", ""))
            if isinstance(faithful_ir, Mapping)
            else ""
        ),
        (
            str(embedded.get("reading_review_id", ""))
            if isinstance(embedded, Mapping)
            else ""
        ),
    }
    if reading_review_id not in parent_ids:
        return None
    return {
        "path": str(path),
        "reading_review_id": reading_review_id,
        "ir_version": (
            str(faithful_ir.get("ir_version", ""))
            if isinstance(faithful_ir, Mapping)
            else ""
        ),
        "status": "quarantined",
    }


def _write_semantic_quarantine_record(
    root: Path,
    requirement_id: str,
    descendant: Mapping[str, Any],
) -> None:
    """Record the inconsistency without deleting or rewriting the descendant."""

    _atomic_write_json(
        root
        / "quarantined_semantic_descendants"
        / f"{requirement_id}.json",
        {
            "requirement_id": requirement_id,
            "reason_code": "NON_READY_PLAIN_READING_HAS_SEMANTIC_DESCENDANT",
            "suggested_rerun_stage": "manual_consistency_review",
            "descendant": dict(descendant),
            "quarantined_at": _timestamp(),
        },
    )


def _active_review_views_inconsistent(
    state: Mapping[str, Any],
    *,
    checkpoint_dir: str | Path,
) -> bool:
    expected_root = _review_namespace_root(
        str(state.get("records_dir") or "review_records"),
        checkpoint_dir,
    )
    namespace_value = str(state.get("review_namespace_dir", "")).strip()
    if not namespace_value or not _same_path(namespace_value, expected_root):
        return True
    categories = (
        ("no_ambiguity", "pending_no_ambiguity_file", "normal-no-axis-v1"),
        (
            "semantic_fallback",
            "pending_semantic_fallback_file",
            "semantic-fallback-decision-v1",
        ),
        (
            "ap_inventory",
            "pending_ap_inventory_file",
            "semantic-override-ap-inventory-v1",
        ),
        (
            "ap_mapping",
            "pending_ap_mapping_file",
            "semantic-override-ap-mapping-v1",
        ),
        ("ambiguity", "pending_ambiguity_file", "ambiguity-adjudication-v2"),
    )
    for kind, path_key, default_contract in categories:
        requirement_ids = _pending_ids(state, kind)
        path_value = str(state.get(path_key, ""))
        if not requirement_ids:
            if path_value:
                return True
            continue
        if not path_value:
            return True
        path = Path(path_value)
        if not _same_path(path.parent, expected_root):
            return True
        if not path.exists():
            return True
        sections = _requirement_sections(path.read_text(encoding="utf-8"))
        if set(sections) != set(requirement_ids):
            return True
        for requirement_id in requirement_ids:
            item = state["items"][requirement_id]
            section = sections[requirement_id]
            try:
                _validate_section_metadata(section, item)
            except ValueError:
                return True
            expected_contract = default_contract
            expected_override_id = ""
            if kind == "semantic_fallback" and item.get("status") == (
                "awaiting_semantic_override_confirmation"
            ):
                expected_contract = "semantic-override-confirmation-v1"
                expected_override_id = str(item.get("pending_override_id", ""))
            elif kind in {"ap_inventory", "ap_mapping"}:
                expected_override_id = str(item.get("pending_override_id", ""))
            if _read_field(section, "action_contract_version") != expected_contract:
                return True
            if kind in {"semantic_fallback", "ap_inventory", "ap_mapping"} and _read_field(
                section, "override_id"
            ) != expected_override_id:
                return True
    return False


def _write_state(root: Path, state: Mapping[str, Any]) -> None:
    _atomic_write_json(root / PRE_IR_BATCH_STATE_FILE, state)


def _item_transaction_paths(root: Path, requirement_id: str) -> tuple[Path, ...]:
    return (
        root / PRE_IR_BATCH_STATE_FILE,
        root / "segmentation_generated" / f"{requirement_id}.json",
        root / "segmentation_reviewed" / f"{requirement_id}.json",
        root / "segmentation_comparison" / f"{requirement_id}.json",
        root / "plain_reading_generated" / f"{requirement_id}.json",
        root / "plain_reading_diagnostics" / f"{requirement_id}.json",
        root / "plain_reading_reviewed" / f"{requirement_id}.json",
        root / "ap_inventory_current" / f"{requirement_id}.json",
        root / "pending_semantic_override" / f"{requirement_id}.json",
        root / "human_semantic_overrides" / f"{requirement_id}.json",
        root / REPLACEMENT_DRAFTS_DIR / f"{requirement_id}.json",
    )


@contextmanager
def _item_transaction_context(root: Path, requirement_id: str):
    transaction_id = "TXN-" + requirement_id + "-" + uuid.uuid4().hex
    directory = root / PRE_IR_TRANSACTIONS_DIR / transaction_id
    directory.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    for index, target in enumerate(_item_transaction_paths(root, requirement_id)):
        backup_name = f"backup-{index}.bin"
        existed = target.exists()
        if existed:
            (directory / backup_name).write_bytes(target.read_bytes())
        entries.append(
            {
                "target": str(target.relative_to(root)),
                "existed": existed,
                "backup": backup_name,
            }
        )
    manifest = {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "requirement_id": requirement_id,
        "phase": "prepared",
        "entries": entries,
        "created_at": _timestamp(),
    }
    _atomic_write_json(directory / "manifest.json", manifest)
    transaction = _ItemTransaction(transaction_id, directory)
    try:
        yield transaction
    except BaseException:
        _rollback_item_transaction(root, directory, manifest)
        raise
    else:
        try:
            manifest["phase"] = "committed"
            _atomic_write_json(directory / "manifest.json", manifest)
        except BaseException:
            _rollback_item_transaction(root, directory, manifest)
            raise
        shutil.rmtree(directory, ignore_errors=True)


def _rollback_item_transaction(
    root: Path,
    directory: Path,
    manifest: Mapping[str, Any],
) -> None:
    for entry in manifest.get("entries", ()):
        if not isinstance(entry, Mapping):
            continue
        target = (root / str(entry.get("target", ""))).resolve()
        if root.resolve() not in target.parents and target != root.resolve():
            raise ValueError("transaction target escaped checkpoint root")
        if bool(entry.get("existed")):
            backup = directory / str(entry.get("backup", ""))
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".rollback.tmp")
            temporary.write_bytes(backup.read_bytes())
            temporary.replace(target)
        elif target.exists():
            target.unlink()
    shutil.rmtree(directory, ignore_errors=True)


def _recover_item_transactions(root: Path, state: Mapping[str, Any]) -> bool:
    transaction_root = root / PRE_IR_TRANSACTIONS_DIR
    if not transaction_root.exists():
        return False
    rolled_back = False
    for directory in sorted(path for path in transaction_root.iterdir() if path.is_dir()):
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            shutil.rmtree(directory, ignore_errors=True)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_item_transaction_manifest(root, directory, manifest)
        requirement_id = str(manifest.get("requirement_id", ""))
        transaction_id = str(manifest.get("transaction_id", ""))
        phase = str(manifest.get("phase") or "prepared")
        item = state.get("items", {}).get(requirement_id, {})
        if phase == "committed" or item.get("last_transaction_id") == transaction_id:
            shutil.rmtree(directory, ignore_errors=True)
            continue
        _rollback_item_transaction(root, directory, manifest)
        rolled_back = True
    return rolled_back


def _validate_item_transaction_manifest(
    root: Path,
    directory: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Reject a malformed journal before it can modify checkpoint files."""

    if int(manifest.get("schema_version", 0)) != 1:
        raise PreIRStateConsistencyError("unsupported Pre-IR transaction schema")
    requirement_id = str(manifest.get("requirement_id", ""))
    transaction_id = str(manifest.get("transaction_id", ""))
    if (
        REQUIREMENT_ID_PATTERN.fullmatch(requirement_id) is None
        or transaction_id != directory.name
        or not transaction_id.startswith("TXN-" + requirement_id + "-")
    ):
        raise PreIRStateConsistencyError("invalid Pre-IR transaction identity")
    phase = str(manifest.get("phase") or "prepared")
    if phase not in {"prepared", "committed"}:
        raise PreIRStateConsistencyError("invalid Pre-IR transaction phase")

    allowed_targets = {
        str(path.relative_to(root))
        for path in _item_transaction_paths(root, requirement_id)
    }
    seen: set[str] = set()
    seen_backups: set[str] = set()
    entries = manifest.get("entries", ())
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise PreIRStateConsistencyError("invalid Pre-IR transaction entries")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PreIRStateConsistencyError("invalid Pre-IR transaction entry")
        if not isinstance(entry.get("existed"), bool):
            raise PreIRStateConsistencyError(
                "invalid Pre-IR transaction existence marker"
            )
        target_value = str(entry.get("target", ""))
        target_path = Path(target_value)
        target = str(target_path)
        if (
            target_path.is_absolute()
            or ".." in target_path.parts
            or target not in allowed_targets
            or target in seen
        ):
            raise PreIRStateConsistencyError("invalid Pre-IR transaction target")
        seen.add(target)
        backup_name = str(entry.get("backup", ""))
        if (
            not backup_name
            or Path(backup_name).name != backup_name
            or backup_name in seen_backups
        ):
            raise PreIRStateConsistencyError("invalid Pre-IR transaction backup name")
        seen_backups.add(backup_name)
        if phase == "prepared" and bool(entry.get("existed")):
            backup = directory / backup_name
            if not backup.is_file() or backup.parent != directory:
                raise PreIRStateConsistencyError(
                    "missing or invalid Pre-IR transaction backup"
                )
    if seen != allowed_targets:
        raise PreIRStateConsistencyError(
            "Pre-IR transaction journal does not cover the complete item write set"
        )


def _commit_item_state(
    root: Path,
    state: dict[str, Any],
    requirement_id: str,
    updated_item: Mapping[str, Any],
    *,
    transaction: _ItemTransaction | None = None,
) -> None:
    """Commit one item with the state file as the final durable marker."""

    items = state["items"]
    previous = items[requirement_id]
    consistency_keys = (
        "no_ambiguity_review_complete",
        "semantic_fallback_review_complete",
        "ambiguity_review_complete",
        "pre_ir_closed",
        "review_cycle_closed",
        "downstream_eligible",
        "downstream_excluded",
        "run_status",
        "updated_at",
    )
    missing = object()
    previous_consistency = {
        key: state.get(key, missing) for key in consistency_keys
    }
    committed_item = dict(updated_item)
    _clear_item_errors(committed_item)
    if transaction is not None:
        committed_item["last_transaction_id"] = transaction.transaction_id
    items[requirement_id] = committed_item
    if str(state.get("pending_replacement_review_file", "")) and (
        previous.get("status") == "awaiting_replacement_segmentation_review"
        or committed_item.get("status")
        == "awaiting_replacement_segmentation_review"
    ) and _replacement_queue_item_fingerprint(
        previous
    ) != _replacement_queue_item_fingerprint(
        committed_item
    ):
        # Any canonical replacement transition invalidates the whole snapshot.
        # Increment before clearing so a recovery export never overwrites the
        # edited file that was current when the transition occurred.
        state["replacement_review_round"] = int(
            state.get("replacement_review_round", 1)
        ) + 1
        state["pending_replacement_review_file"] = ""
        state["replacement_review_snapshot_id"] = ""
    _close_if_complete(state)
    try:
        _write_state(root, state)
    except Exception:
        items[requirement_id] = previous
        for key, value in previous_consistency.items():
            if value is missing:
                state.pop(key, None)
            else:
                state[key] = value
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


__all__ = [
    "PRE_IR_BATCHES_DIR",
    "PRE_IR_BATCH_STATE_FILE",
    "PreIRBatchOutcome",
    "PreIRStateConsistencyError",
    "ReplacementReviewExportOutcome",
    "ReplacementReviewQueueOutcome",
    "ReviewApplyOutcome",
    "ReviewFileInputError",
    "apply_ambiguity_review",
    "apply_no_ambiguity_review",
    "apply_pending_replacement_review",
    "apply_semantic_fallback_review",
    "ensure_deferred_pre_ir_batch",
    "export_pending_replacement_review",
    "load_pre_ir_batch_state",
    "mark_pre_ir_downstream_started",
    "pending_replacement_review_ids",
    "reopen_pre_ir_item",
    "review_pending_replacement_pre_ir",
]
