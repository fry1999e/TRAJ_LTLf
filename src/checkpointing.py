"""Checkpoint helpers for resumable batch runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
from typing import Any
import uuid

from contract_versions import (
    AP_INVENTORY_SCHEMA_VERSION,
    HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION,
    HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION,
    OBLIGATION_COMPILER_VERSION,
    SEMANTIC_EVIDENCE_CONTRACT_VERSION,
    TRACE_CONTRACT_VERSION,
    ap_identity_runtime_contract,
    faithful_ir_runtime_contract,
)
from ap_identity import (
    APIdentityValidationError,
    ap_inventory_from_mapping,
    override_ap_comparison_from_mapping,
    undetermined_profile,
    validate_ap_inventory_integrity,
    validate_override_ap_comparison_integrity,
)
from data_examples import (
    LabeledTrace,
    trace_semantics_from_mapping,
    trace_semantics_to_dict,
)
from schemas import (
    APInventory,
    AmbiguityAxis,
    AmbiguityAxisSet,
    ArtifactGenerator,
    ApprovedSemanticIR,
    CandidateIRBranch,
    CandidateIRSet,
    BlockedFailureRecord,
    DegradedResultRecord,
    FinalApprovalRecord,
    HumanSemanticOverride,
    IRCoverageReport,
    IRAmbiguityAdjudicationReport,
    IRAmbiguityAdjudicationRound,
    IRCandidateSemanticScore,
    LabeledTraceEvidenceSet,
    OverrideAPComparison,
    SemanticChallenge,
    SemanticChallengeSet,
    SemanticEvidenceGateReport,
    SemanticParameterDecision,
    SemanticResolutionRecord,
    PLAIN_READING_SCHEMA_VERSION,
    SURFACE_SEGMENTATION_SCHEMA_VERSION,
    PlainReadingReview,
    ReviewPendingRecord,
    SemanticStructure,
    SurfaceSegmentation,
    TraceCoverageReport,
    TraceEvidencePolicy,
)
from faithful_ir import (
    compute_ir_version,
    faithful_ir_from_mapping,
    recertify_legacy_faithful_ir_mapping,
)
from plain_reading import (
    PlainReadingValidationError,
    decision_fingerprint,
    plain_reading_from_mapping,
    validate_human_semantic_override_integrity,
)
from plain_reading_need_view import (
    PlainReadingNeedView,
    plain_reading_need_view_from_mapping,
)
from surface_segmentation import segmentation_from_mapping, source_hash
from ir_compiler import compile_review_trace_plan
from semantic_readiness import (
    CAN_GENERATE_AUTHORITATIVE_TRACES,
    audit_semantic_readiness,
)
from formula_closure import stable_formula_hash
from ltlf_parser import parse_formula
from trace_formatter import trace_to_serializable
from trace_evidence import TraceEvidenceDispute, TraceEvidenceRecord


CHECKPOINT_SCHEMA_VERSION = 10
PRE_IR_CHECKPOINT_CONTRACT_VERSION = 2
DEFAULT_CHECKPOINT_DIR = "data/output/manual/checkpoints"
SEMANTIC_OVERRIDE_HISTORY_CONTRACT_VERSION = 1
_LEGACY_PRE_IR_CHECKPOINT_SCHEMA_VERSIONS = frozenset({9})
_PRE_IR_PRESERVED_SUBDIRS = frozenset(
    {
        "segmentation_generated",
        "segmentation_reviewed",
        "plain_reading_generated",
        "plain_reading_reviewed",
        "human_semantic_overrides",
        "pending_semantic_override",
    }
)


def _checkpoint_trace_plan(ir):
    """Persist a plan whose purpose matches the audited semantic capability."""

    return compile_review_trace_plan(ir)


def prepare_checkpoint_run(
    checkpoint_dir: str | None,
    *,
    manifest: dict[str, Any],
    fresh: bool = False,
) -> Path | None:
    """Create or validate the run manifest for a checkpoint directory."""

    if not checkpoint_dir:
        return None

    root = Path(checkpoint_dir)
    if fresh and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    input_manifest = dict(manifest)
    runtime_contract = input_manifest.pop(
        "faithful_ir_runtime_contract",
        faithful_ir_runtime_contract(),
    )
    input_fingerprint = _fingerprint(input_manifest)
    runtime_fingerprint = _fingerprint(runtime_contract)
    manifest_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": input_fingerprint,
        "input_fingerprint": input_fingerprint,
        "semantic_runtime_fingerprint": runtime_fingerprint,
        "faithful_ir_runtime_contract": runtime_contract,
        "manifest": manifest,
        "updated_at": _timestamp(),
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_manifest = existing.get("manifest", {})
        existing_input_manifest = (
            dict(existing_manifest)
            if isinstance(existing_manifest, Mapping)
            else {}
        )
        existing_input_manifest.pop("faithful_ir_runtime_contract", None)
        legacy_policy_missing = "trace_evidence_policy" not in existing_input_manifest
        if legacy_policy_missing:
            existing_input_manifest["trace_evidence_policy"] = asdict(
                TraceEvidencePolicy()
            )
        existing_input_fingerprint = (
            _fingerprint(existing_input_manifest)
            if legacy_policy_missing
            else existing.get("input_fingerprint")
            or _fingerprint(existing_input_manifest)
        )
        if existing_input_fingerprint != input_fingerprint:
            raise RuntimeError(
                "checkpoint 与当前输入或参数不匹配。请使用 --fresh 清空断点后重跑，"
                f"或指定新的 --checkpoint-dir。当前目录: {root}"
            )
        # A compatible run manifest may be promoted in place, but item
        # checkpoints from an older schema remain unreadable.  This lets the
        # caller regenerate the new pre-IR gates without requiring a destructive
        # ``--fresh`` solely because the checkpoint contract was upgraded.
        if (
            existing.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or existing.get("semantic_runtime_fingerprint") != runtime_fingerprint
            or existing.get("input_fingerprint") != input_fingerprint
        ):
            _write_json(manifest_path, manifest_payload)
        else:
            existing["updated_at"] = manifest_payload["updated_at"]
            _write_json(manifest_path, existing)
    else:
        _write_json(manifest_path, manifest_payload)
    return root


def save_segmentation_generated_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    segmentation: Any,
    review_history: list[dict] | tuple[Any, ...] | None = None,
    generator: Any | None = None,
    experiment_mode: str | None = None,
    data_split: str | None = None,
) -> None:
    """Save a mechanically validated segmentation before surface review."""

    _save_segmentation_checkpoint(
        checkpoint_dir,
        subdir="segmentation_generated",
        requirement_id=requirement_id,
        segmentation=segmentation,
        reviewed=False,
        review_history=review_history,
        generator=generator,
        experiment_mode=experiment_mode,
        data_split=data_split,
    )


def load_segmentation_generated_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    expected_source_hash: str | None = None,
    expected_segmentation_id: str | None = None,
) -> dict | None:
    return _load_segmentation_checkpoint(
        checkpoint_dir,
        "segmentation_generated",
        requirement_id,
        expected_source_hash=expected_source_hash,
        expected_segmentation_id=expected_segmentation_id,
    )


def save_segmentation_reviewed_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    segmentation: Any,
    review_history: list[dict] | tuple[Any, ...] | None = None,
    generator: Any | None = None,
    experiment_mode: str | None = None,
    data_split: str | None = None,
) -> None:
    """Save a surface-reviewed segmentation that may feed plain reading."""

    _save_segmentation_checkpoint(
        checkpoint_dir,
        subdir="segmentation_reviewed",
        requirement_id=requirement_id,
        segmentation=segmentation,
        reviewed=True,
        review_history=review_history,
        generator=generator,
        experiment_mode=experiment_mode,
        data_split=data_split,
    )


def load_segmentation_reviewed_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    expected_source_hash: str | None = None,
    expected_segmentation_id: str | None = None,
) -> dict | None:
    return _load_segmentation_checkpoint(
        checkpoint_dir,
        "segmentation_reviewed",
        requirement_id,
        expected_source_hash=expected_source_hash,
        expected_segmentation_id=expected_segmentation_id,
    )


def save_plain_reading_generated_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    plain_reading_review: Any,
    review_history: list[dict] | tuple[Any, ...] | None = None,
    generator: Any | None = None,
    experiment_mode: str | None = None,
    data_split: str | None = None,
    segmentation_schema_version: str = SURFACE_SEGMENTATION_SCHEMA_VERSION,
) -> None:
    """Save generated ordinary-language reading options before review."""

    _save_plain_reading_checkpoint(
        checkpoint_dir,
        subdir="plain_reading_generated",
        requirement_id=requirement_id,
        plain_reading_review=plain_reading_review,
        reviewed=False,
        review_history=review_history,
        generator=generator,
        experiment_mode=experiment_mode,
        data_split=data_split,
        segmentation_schema_version=segmentation_schema_version,
    )


def save_plain_reading_generation_diagnostic(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    diagnostic: Mapping[str, Any],
) -> None:
    """Persist bounded rejected model responses for plain-reading generation."""

    if not checkpoint_dir:
        return
    if str(diagnostic.get("requirement_id", "")) != requirement_id:
        raise ValueError("plain-reading diagnostic requirement_id does not match item")
    attempt = int(diagnostic.get("attempt", 0))
    if attempt < 1:
        raise ValueError("plain-reading diagnostic attempt must be positive")
    path = _checkpoint_path(
        checkpoint_dir,
        "plain_reading_diagnostics",
        requirement_id,
    )
    attempts: list[dict[str, Any]] = []
    if attempt != 1 and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
            and existing.get("source_hash") == diagnostic.get("source_hash")
            and existing.get("segmentation_id") == diagnostic.get("segmentation_id")
        ):
            attempts = [
                dict(item)
                for item in existing.get("attempts", ())
                if isinstance(item, Mapping)
            ]
    attempts.append(dict(diagnostic))
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "kind": "plain_reading_generation_diagnostic",
        "requirement_id": requirement_id,
        "source_hash": str(diagnostic.get("source_hash", "")),
        "segmentation_id": str(diagnostic.get("segmentation_id", "")),
        "response_sanitization": (
            "plain-reading schema fields only; bounded text; provider metadata omitted"
        ),
        "attempts": attempts,
        "contract_version": str(diagnostic.get("contract_version", "")),
        "canonicalizer_version": str(diagnostic.get("canonicalizer_version", "")),
        "validator_version": str(diagnostic.get("validator_version", "")),
        "repair_policy_version": str(diagnostic.get("repair_policy_version", "")),
        "artifact_hash": str(diagnostic.get("artifact_hash", "")),
        "sanitized_response_hash": str(
            diagnostic.get(
                "sanitized_response_hash",
                diagnostic.get("artifact_hash", ""),
            )
        ),
        "hash_scope": str(
            diagnostic.get("hash_scope", "sanitized_rejected_response")
        ),
        "issue_fingerprint": str(diagnostic.get("issue_fingerprint", "")),
        "repair_history": attempts,
        "updated_at": _timestamp(),
    }
    _write_json(path, payload)


def save_ap_inventory_checkpoint(
    checkpoint_dir: str | None,
    *,
    inventory: APInventory,
    make_current: bool = False,
) -> None:
    """Write one immutable AP inventory and optionally its validated pointer."""

    if not checkpoint_dir:
        return
    validate_ap_inventory_integrity(
        inventory,
        allow_unapproved=inventory.status != "approved",
    )
    root = Path(checkpoint_dir)
    path = (
        root
        / "ap_inventories"
        / _safe_semantic_override_history_component(inventory.requirement_id)
        / f"{_safe_semantic_override_history_component(inventory.inventory_id)}.json"
    )
    payload = {
        "kind": "ap_inventory",
        "schema_version": AP_INVENTORY_SCHEMA_VERSION,
        "runtime_contract": ap_identity_runtime_contract(),
        "requirement_id": inventory.requirement_id,
        "inventory_id": inventory.inventory_id,
        "artifact_hash": inventory.artifact_hash,
        "ap_inventory": asdict(inventory),
    }
    _write_immutable_json(path, payload, conflict_code="HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT")
    if make_current:
        if inventory.status != "approved":
            raise ValueError("only an approved AP inventory may become current")
        _write_json(
            _checkpoint_path(
                checkpoint_dir,
                "ap_inventory_current",
                inventory.requirement_id,
            ),
            {
                "kind": "ap_inventory_current_pointer",
                "schema_version": AP_INVENTORY_SCHEMA_VERSION,
                "runtime_contract": ap_identity_runtime_contract(),
                "requirement_id": inventory.requirement_id,
                "inventory_id": inventory.inventory_id,
                "artifact_hash": inventory.artifact_hash,
                "source_hash": inventory.source_hash,
                "segmentation_id": inventory.segmentation_id,
                "representation_profile_id": inventory.representation_profile_id,
                "updated_at": _timestamp(),
            },
        )


def load_ap_inventory_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    inventory_id: str | None = None,
    current: bool = False,
    expected_source_hash: str | None = None,
    expected_segmentation_id: str | None = None,
    expected_profile_id: str | None = None,
    allow_unapproved: bool = False,
) -> APInventory | None:
    if not checkpoint_dir:
        return None
    if current and inventory_id is not None:
        raise ValueError("AP inventory load cannot use current and inventory_id together")
    root = Path(checkpoint_dir)
    pointer: Mapping[str, Any] | None = None
    if current:
        pointer_path = _checkpoint_path(
            checkpoint_dir,
            "ap_inventory_current",
            requirement_id,
        )
        if not pointer_path.is_file():
            return None
        pointer = _read_json_mapping(
            pointer_path,
            code="HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT",
        )
        if (
            pointer.get("kind") != "ap_inventory_current_pointer"
            or pointer.get("runtime_contract") != ap_identity_runtime_contract()
            or pointer.get("requirement_id") != requirement_id
        ):
            raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: invalid current pointer")
        inventory_id = str(pointer.get("inventory_id", ""))
    if not inventory_id:
        return None
    path = (
        root
        / "ap_inventories"
        / _safe_semantic_override_history_component(requirement_id)
        / f"{_safe_semantic_override_history_component(inventory_id)}.json"
    )
    if not path.is_file():
        if current:
            raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: dangling current pointer")
        return None
    payload = _read_json_mapping(
        path,
        code="HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT",
    )
    if (
        payload.get("kind") != "ap_inventory"
        or payload.get("schema_version") != AP_INVENTORY_SCHEMA_VERSION
        or payload.get("runtime_contract") != ap_identity_runtime_contract()
        or payload.get("requirement_id") != requirement_id
        or payload.get("inventory_id") != inventory_id
    ):
        raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: invalid AP inventory envelope")
    mapping = payload.get("ap_inventory")
    if not isinstance(mapping, Mapping):
        raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: missing AP inventory")
    try:
        inventory = ap_inventory_from_mapping(mapping)
        validate_ap_inventory_integrity(
            inventory,
            allow_unapproved=allow_unapproved or inventory.status != "approved",
        )
    except (TypeError, ValueError, APIdentityValidationError) as exc:
        raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: " + str(exc)) from exc
    checks = (
        (payload.get("artifact_hash"), inventory.artifact_hash),
        (expected_source_hash, inventory.source_hash),
        (expected_segmentation_id, inventory.segmentation_id),
        (expected_profile_id, inventory.representation_profile_id),
    )
    if any(expected is not None and expected != actual for expected, actual in checks):
        raise ValueError("HUMAN_OVERRIDE_AP_INVENTORY_STALE")
    if pointer is not None and (
        pointer.get("artifact_hash") != inventory.artifact_hash
        or pointer.get("source_hash") != inventory.source_hash
        or pointer.get("segmentation_id") != inventory.segmentation_id
        or pointer.get("representation_profile_id")
        != inventory.representation_profile_id
    ):
        raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: current pointer mismatch")
    return inventory


def save_override_ap_comparison_checkpoint(
    checkpoint_dir: str | None,
    *,
    comparison: OverrideAPComparison,
    original_inventory: APInventory,
    replacement_inventory: APInventory,
) -> None:
    if not checkpoint_dir:
        return
    validate_override_ap_comparison_integrity(
        comparison,
        original_inventory=original_inventory,
        replacement_inventory=replacement_inventory,
    )
    root = Path(checkpoint_dir)
    path = (
        root
        / "override_ap_comparisons"
        / _safe_semantic_override_history_component(original_inventory.requirement_id)
        / f"{_safe_semantic_override_history_component(comparison.comparison_id)}.json"
    )
    payload = {
        "kind": "override_ap_comparison",
        "schema_version": AP_INVENTORY_SCHEMA_VERSION,
        "runtime_contract": ap_identity_runtime_contract(),
        "requirement_id": original_inventory.requirement_id,
        "comparison_id": comparison.comparison_id,
        "artifact_hash": comparison.artifact_hash,
        "override_ap_comparison": asdict(comparison),
    }
    _write_immutable_json(path, payload, conflict_code="HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT")


def load_override_ap_comparison_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    comparison_id: str,
    *,
    original_inventory: APInventory,
    replacement_inventory: APInventory,
) -> OverrideAPComparison | None:
    if not checkpoint_dir:
        return None
    path = (
        Path(checkpoint_dir)
        / "override_ap_comparisons"
        / _safe_semantic_override_history_component(requirement_id)
        / f"{_safe_semantic_override_history_component(comparison_id)}.json"
    )
    if not path.is_file():
        return None
    payload = _read_json_mapping(
        path,
        code="HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT",
    )
    if (
        payload.get("kind") != "override_ap_comparison"
        or payload.get("schema_version") != AP_INVENTORY_SCHEMA_VERSION
        or payload.get("runtime_contract") != ap_identity_runtime_contract()
        or payload.get("requirement_id") != requirement_id
        or payload.get("comparison_id") != comparison_id
    ):
        raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: invalid comparison envelope")
    mapping = payload.get("override_ap_comparison")
    if not isinstance(mapping, Mapping):
        raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: missing AP comparison")
    try:
        comparison = override_ap_comparison_from_mapping(mapping)
        validate_override_ap_comparison_integrity(
            comparison,
            original_inventory=original_inventory,
            replacement_inventory=replacement_inventory,
        )
    except (TypeError, ValueError, APIdentityValidationError) as exc:
        raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: " + str(exc)) from exc
    if payload.get("artifact_hash") != comparison.artifact_hash:
        raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: comparison hash mismatch")
    return comparison


def _validate_human_override_checkpoint_artifacts(
    checkpoint_dir: str,
    override: HumanSemanticOverride,
) -> None:
    contract = (
        override.override_contract_version
        or HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION
    )
    if contract == HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION:
        validate_human_semantic_override_integrity(override)
        return
    if contract != HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION:
        raise ValueError("HUMAN_OVERRIDE_AP_CONTRACT_UNSUPPORTED")
    original = None
    replacement = None
    comparison = None
    if override.original_ap_inventory_id:
        original = load_ap_inventory_checkpoint(
            checkpoint_dir,
            override.requirement_id,
            inventory_id=override.original_ap_inventory_id,
            expected_source_hash=override.original_source_hash,
            expected_profile_id=override.representation_profile_id,
        )
        if original is None or original.artifact_hash != override.original_ap_inventory_hash:
            raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: original inventory")
    if override.replacement_ap_inventory_id:
        replacement = load_ap_inventory_checkpoint(
            checkpoint_dir,
            override.requirement_id,
            inventory_id=override.replacement_ap_inventory_id,
            expected_source_hash=override.replacement_hash,
            expected_profile_id=override.representation_profile_id,
        )
        if (
            replacement is None
            or replacement.artifact_hash != override.replacement_ap_inventory_hash
        ):
            raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: replacement inventory")
    if override.ap_comparison_id:
        if original is None or replacement is None:
            raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: comparison inputs")
        comparison = load_override_ap_comparison_checkpoint(
            checkpoint_dir,
            override.requirement_id,
            override.ap_comparison_id,
            original_inventory=original,
            replacement_inventory=replacement,
        )
        if comparison is None or comparison.artifact_hash != override.ap_comparison_hash:
            raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: AP comparison")
    validate_human_semantic_override_integrity(
        override,
        original_inventory=original,
        replacement_inventory=replacement,
        comparison=comparison,
    )


def save_ap_identity_artifact_checkpoint(
    checkpoint_dir: str | None,
    artifact: APInventory | OverrideAPComparison,
) -> None:
    """Persist artifacts emitted in original/replacement/comparison order."""

    if not checkpoint_dir:
        return
    if isinstance(artifact, APInventory):
        save_ap_inventory_checkpoint(
            checkpoint_dir,
            inventory=artifact,
            make_current=(
                artifact.status == "approved"
                and artifact.reviewer_source == "approved_override_mapping"
            ),
        )
        return
    if not isinstance(artifact, OverrideAPComparison):
        raise TypeError("unsupported AP identity artifact")
    original = load_ap_inventory_checkpoint(
        checkpoint_dir,
        _comparison_requirement_id(checkpoint_dir, artifact.original_inventory_id),
        inventory_id=artifact.original_inventory_id,
        allow_unapproved=True,
    )
    if original is None:
        raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: original inventory missing")
    replacement = load_ap_inventory_checkpoint(
        checkpoint_dir,
        original.requirement_id,
        inventory_id=artifact.replacement_inventory_id,
        allow_unapproved=True,
    )
    if replacement is None:
        raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: replacement inventory missing")
    save_override_ap_comparison_checkpoint(
        checkpoint_dir,
        comparison=artifact,
        original_inventory=original,
        replacement_inventory=replacement,
    )


def load_human_semantic_override_ap_artifacts(
    checkpoint_dir: str | None,
    override: HumanSemanticOverride,
) -> tuple[APInventory | None, APInventory | None, OverrideAPComparison | None]:
    if (
        not checkpoint_dir
        or override.override_contract_version
        != HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION
    ):
        return None, None, None
    original = load_ap_inventory_checkpoint(
        checkpoint_dir,
        override.requirement_id,
        inventory_id=override.original_ap_inventory_id,
        expected_source_hash=override.original_source_hash,
        expected_profile_id=override.representation_profile_id,
        allow_unapproved=True,
    )
    replacement = load_ap_inventory_checkpoint(
        checkpoint_dir,
        override.requirement_id,
        inventory_id=override.replacement_ap_inventory_id,
        expected_source_hash=override.replacement_hash,
        expected_profile_id=override.representation_profile_id,
        allow_unapproved=True,
    )
    comparison = None
    if override.ap_comparison_id and original is not None and replacement is not None:
        comparison = load_override_ap_comparison_checkpoint(
            checkpoint_dir,
            override.requirement_id,
            override.ap_comparison_id,
            original_inventory=original,
            replacement_inventory=replacement,
        )
    return original, replacement, comparison


def _comparison_requirement_id(
    checkpoint_dir: str,
    inventory_id: str,
) -> str:
    root = Path(checkpoint_dir) / "ap_inventories"
    if not root.is_dir():
        return ""
    matches = [
        path.parent.name
        for path in root.glob(f"*/{_safe_semantic_override_history_component(inventory_id)}.json")
    ]
    if len(matches) != 1:
        raise ValueError("HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: inventory identity is ambiguous")
    return matches[0]


def migrate_v1_pending_semantic_overrides(
    checkpoint_dir: str | None,
    *,
    commit: bool = False,
    requirement_ids: tuple[str, ...] = (),
    reviewer_id: str = "migration-reviewer",
) -> dict[str, Any]:
    """Dry-run or explicitly migrate only v1 extraction-unknown proposals."""

    report: dict[str, Any] = {
        "mode": "commit" if commit else "dry-run",
        "items": [],
        "counts": {"keep-v1": 0, "upgradeable": 0, "manual": 0, "corrupt": 0},
    }
    if not checkpoint_dir:
        return report
    root = Path(checkpoint_dir)
    pending_dir = root / "pending_semantic_override"
    if not pending_dir.is_dir():
        return report
    wanted = set(requirement_ids)
    from plain_reading import (
        classify_v1_human_semantic_override_migration,
        migrate_v1_unknown_human_semantic_override,
    )

    for path in sorted(pending_dir.glob("*.json")):
        requirement_id = path.stem
        if wanted and requirement_id not in wanted:
            continue
        entry: dict[str, Any] = {"requirement_id": requirement_id}
        try:
            override = load_human_semantic_override_checkpoint(
                checkpoint_dir,
                requirement_id,
                pending_only=True,
            )
            if override is None:
                continue
            classification = classify_v1_human_semantic_override_migration(override)
            entry.update(
                {
                    "classification": classification,
                    "override_id": override.override_id,
                    "status": override.status,
                    "issue_codes": list(override.issue_codes),
                }
            )
            report["counts"][classification] += 1
            if commit and classification == "upgradeable":
                artifacts: list[APInventory | OverrideAPComparison] = []
                migrated = migrate_v1_unknown_human_semantic_override(
                    override,
                    reviewer_id=reviewer_id,
                    ap_artifact_callback=artifacts.append,
                )
                inventories = [
                    item for item in artifacts if isinstance(item, APInventory)
                ]
                comparisons = [
                    item
                    for item in artifacts
                    if isinstance(item, OverrideAPComparison)
                ]
                for inventory in inventories:
                    save_ap_inventory_checkpoint(
                        checkpoint_dir,
                        inventory=inventory,
                    )
                if comparisons:
                    if len(inventories) != 2:
                        raise ValueError(
                            "HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT: migration inputs"
                        )
                    save_override_ap_comparison_checkpoint(
                        checkpoint_dir,
                        comparison=comparisons[0],
                        original_inventory=inventories[0],
                        replacement_inventory=inventories[1],
                    )
                save_human_semantic_override_checkpoint(
                    checkpoint_dir,
                    override=replace(override, status="superseded"),
                )
                save_human_semantic_override_checkpoint(
                    checkpoint_dir,
                    override=migrated,
                )
                entry["migrated_override_id"] = migrated.override_id
                entry["migrated_status"] = migrated.status
        except Exception as exc:
            previous_classification = entry.get("classification")
            entry.update(
                {
                    "classification": "corrupt",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if previous_classification != "corrupt":
                if previous_classification in report["counts"]:
                    report["counts"][previous_classification] -= 1
                report["counts"]["corrupt"] += 1
        report["items"].append(entry)
    report["items"] = sorted(
        report["items"], key=lambda item: str(item.get("requirement_id", ""))
    )
    return report


def save_human_semantic_override_checkpoint(
    checkpoint_dir: str | None,
    *,
    override: HumanSemanticOverride,
) -> None:
    """Persist the current override pointer and append immutable audit events.

    The per-requirement files remain the compatibility/current-state view used
    by ``load_human_semantic_override_checkpoint``.  Every current record that
    could be overwritten, plus the incoming lifecycle state, is first archived
    under an override-id-specific history path.  History is deliberately not a
    load candidate for the current pointer.
    """

    if not checkpoint_dir:
        return
    if not isinstance(override, HumanSemanticOverride):
        raise TypeError("semantic override checkpoint requires HumanSemanticOverride")
    _validate_human_override_checkpoint_artifacts(checkpoint_dir, override)
    root = Path(checkpoint_dir)
    existing_current: list[HumanSemanticOverride] = []
    current_payloads: dict[str, dict[str, Any]] = {}
    for current_subdir in (
        "human_semantic_overrides",
        "pending_semantic_override",
    ):
        current_path = _checkpoint_path(
            checkpoint_dir,
            current_subdir,
            override.requirement_id,
        )
        if not current_path.exists():
            continue
        try:
            current_payload = json.loads(current_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "HUMAN_OVERRIDE_CORRUPT: unreadable current checkpoint "
                f"{current_path}"
            ) from exc
        if (
            current_payload.get("schema_version")
            not in {
                CHECKPOINT_SCHEMA_VERSION,
                *_LEGACY_PRE_IR_CHECKPOINT_SCHEMA_VERSIONS,
            }
            or current_payload.get("pre_ir_contract_version")
            != PRE_IR_CHECKPOINT_CONTRACT_VERSION
            or current_payload.get("kind") != "human_semantic_override"
        ):
            raise ValueError(
                "HUMAN_OVERRIDE_CORRUPT: incompatible current checkpoint "
                f"{current_path}"
            )
        current_override = _validated_human_semantic_override_payload(
            current_payload,
            expected_requirement_id=override.requirement_id,
            context=str(current_path),
        )
        existing_current.append(current_override)
        current_payloads[current_subdir] = current_payload

    subdir = (
        "human_semantic_overrides"
        if override.status == "active"
        else "pending_semantic_override"
    )
    payload = _human_semantic_override_checkpoint_payload(override)

    superseded_override: HumanSemanticOverride | None = None
    pending_payload = current_payloads.get("pending_semantic_override")
    if override.status == "active" and pending_payload is not None:
        pending_mapping = pending_payload.get("human_semantic_override")
        if (
            isinstance(pending_mapping, Mapping)
            and pending_mapping.get("override_id") == override.override_id
        ):
            pending_override = _validated_human_semantic_override_payload(
                pending_payload,
                expected_requirement_id=override.requirement_id,
                context="pending semantic override",
            )
            superseded_override = replace(pending_override, status="superseded")
            validate_human_semantic_override_integrity(superseded_override)

    audit_events = [*existing_current, override]
    if superseded_override is not None:
        audit_events.append(superseded_override)
    _validate_semantic_override_event_batch(audit_events)
    for event in audit_events:
        _preflight_human_semantic_override_history_event(root, event)
    for event in audit_events:
        _write_human_semantic_override_history_event(root, event)

    _write_json(
        _checkpoint_path(checkpoint_dir, subdir, override.requirement_id),
        payload,
    )
    if superseded_override is not None:
        _write_json(
            _checkpoint_path(
                checkpoint_dir,
                "pending_semantic_override",
                override.requirement_id,
            ),
            _human_semantic_override_checkpoint_payload(superseded_override),
        )


def list_human_semantic_override_history(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    override_id: str | None = None,
) -> tuple[HumanSemanticOverride, ...]:
    """List validated immutable override lifecycle events for one requirement."""

    if not checkpoint_dir:
        return ()
    requirement_dir = _semantic_override_history_requirement_dir(
        Path(checkpoint_dir), requirement_id
    )
    if not requirement_dir.is_dir():
        return ()
    candidates = (
        [_semantic_override_history_override_dir(
            Path(checkpoint_dir), requirement_id, override_id
        )]
        if override_id is not None
        else sorted(path for path in requirement_dir.iterdir() if path.is_dir())
    )
    records: list[tuple[str, HumanSemanticOverride]] = []
    for override_dir in candidates:
        if not override_dir.is_dir():
            continue
        for path in sorted(override_dir.glob("*.json")):
            payload = _load_human_semantic_override_history_payload(path)
            event = _validated_human_semantic_override_history_payload(
                payload,
                expected_requirement_id=requirement_id,
                expected_override_id=override_id,
                context=str(path),
            )
            _validate_human_semantic_override_history_path(
                path,
                event,
                checkpoint_root=Path(checkpoint_dir),
            )
            _validate_human_override_checkpoint_artifacts(checkpoint_dir, event)
            records.append((str(payload.get("recorded_at", "")), event))
    records.sort(key=lambda item: (item[0], item[1].override_id, item[1].status))
    return tuple(event for _, event in records)


def _human_semantic_override_checkpoint_payload(
    override: HumanSemanticOverride,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "kind": "human_semantic_override",
        "requirement_id": override.requirement_id,
        "original_source_hash": override.original_source_hash,
        "parent_effective_source_hash": override.parent_effective_source_hash,
        "replacement_hash": override.replacement_hash,
        "status": override.status,
        "human_semantic_override": asdict(override),
        "invalidation_reason": override.invalidation_reason,
        "old_effective_source_hash": override.old_effective_source_hash,
        "new_effective_source_hash": override.new_effective_source_hash,
        "updated_at": _timestamp(),
    }


_SEMANTIC_OVERRIDE_STABLE_FIELDS = (
    "override_id",
    "requirement_id",
    "original_source_text",
    "original_source_hash",
    "replacement_text",
    "replacement_hash",
    "original_ap_set",
    "replacement_ap_set",
    "ap_validation_status",
    "parent_effective_source_hash",
    "reviewer_role",
    "reviewer_source",
    "issue_codes",
    "old_effective_source_hash",
    "created_at",
)


def _semantic_override_stable_payload(
    override: HumanSemanticOverride,
) -> dict[str, Any]:
    mapping = asdict(override)
    if (
        override.override_contract_version
        == HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION
    ):
        v2_fields = (
            "override_id",
            "requirement_id",
            "original_source_text",
            "original_source_hash",
            "replacement_text",
            "replacement_hash",
            "parent_effective_source_hash",
            "reviewer_role",
            "reviewer_source",
            "old_effective_source_hash",
            "created_at",
            "override_contract_version",
            "ap_validation_contract_version",
            "representation_profile_id",
            "supersedes_override_id",
        )
        return {key: mapping[key] for key in v2_fields}
    return {key: mapping[key] for key in _SEMANTIC_OVERRIDE_STABLE_FIELDS}


def _validate_semantic_override_event_batch(
    events: list[HumanSemanticOverride],
) -> None:
    lineage: dict[str, str] = {}
    event_payloads: dict[tuple[str, str], str] = {}
    for event in events:
        validate_human_semantic_override_integrity(event)
        lineage_fingerprint = _canonical_json(_semantic_override_stable_payload(event))
        previous_lineage = lineage.setdefault(event.override_id, lineage_fingerprint)
        if previous_lineage != lineage_fingerprint:
            raise ValueError(
                "HUMAN_OVERRIDE_HISTORY_CONFLICT: override_id has divergent lineage "
                f"({event.override_id})"
            )
        event_variant = event.status
        if (
            event.override_contract_version
            == HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION
            and event.status
            in {"awaiting_ap_inventory", "awaiting_ap_mapping_confirmation"}
        ):
            event_variant += "--" + hashlib.sha256(
                _canonical_json(asdict(event)).encode("utf-8")
            ).hexdigest()[:16]
        key = (event.override_id, event_variant)
        event_fingerprint = _canonical_json(asdict(event))
        previous_event = event_payloads.setdefault(key, event_fingerprint)
        if previous_event != event_fingerprint:
            raise ValueError(
                "HUMAN_OVERRIDE_HISTORY_CONFLICT: immutable event payload changed "
                f"({event.override_id}/{event.status})"
            )


def _semantic_override_history_requirement_dir(
    checkpoint_root: Path,
    requirement_id: str,
) -> Path:
    return (
        checkpoint_root
        / "semantic_override_history"
        / _safe_semantic_override_history_component(requirement_id)
    )


def _semantic_override_history_override_dir(
    checkpoint_root: Path,
    requirement_id: str,
    override_id: str,
) -> Path:
    return _semantic_override_history_requirement_dir(
        checkpoint_root, requirement_id
    ) / _safe_semantic_override_history_component(override_id)


def _semantic_override_history_event_path(
    checkpoint_root: Path,
    override: HumanSemanticOverride,
) -> Path:
    status_component = _safe_semantic_override_history_component(override.status)
    if (
        override.override_contract_version
        == HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION
        and override.status
        in {"awaiting_ap_inventory", "awaiting_ap_mapping_confirmation"}
    ):
        status_component += "--" + hashlib.sha256(
            _canonical_json(asdict(override)).encode("utf-8")
        ).hexdigest()[:16]
    return _semantic_override_history_override_dir(
        checkpoint_root,
        override.requirement_id,
        override.override_id,
    ) / f"{status_component}.json"


def _validate_human_semantic_override_history_path(
    path: Path,
    override: HumanSemanticOverride,
    *,
    checkpoint_root: Path,
) -> None:
    expected = _semantic_override_history_event_path(checkpoint_root, override)
    if path.resolve() != expected.resolve():
        raise ValueError(
            "HUMAN_OVERRIDE_CORRUPT: history event path does not match its identity "
            f"({path})"
        )


def _safe_semantic_override_history_component(value: str) -> str:
    component = str(value).replace("/", "_").replace("\\", "_")
    if not component:
        raise ValueError("semantic override history id cannot be empty")
    if component in {".", ".."}:
        component = "_" + component.replace(".", "dot")
    return component


def _human_semantic_override_history_payload(
    override: HumanSemanticOverride,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "history_contract_version": SEMANTIC_OVERRIDE_HISTORY_CONTRACT_VERSION,
        "kind": "human_semantic_override_history_event",
        "requirement_id": override.requirement_id,
        "override_id": override.override_id,
        "event_status": override.status,
        "human_semantic_override": asdict(override),
    }


def _preflight_human_semantic_override_history_event(
    checkpoint_root: Path,
    override: HumanSemanticOverride,
) -> None:
    override_dir = _semantic_override_history_override_dir(
        checkpoint_root,
        override.requirement_id,
        override.override_id,
    )
    desired_lineage = _canonical_json(_semantic_override_stable_payload(override))
    if override_dir.is_dir():
        for path in override_dir.glob("*.json"):
            payload = _load_human_semantic_override_history_payload(path)
            existing = _validated_human_semantic_override_history_payload(
                payload,
                expected_requirement_id=override.requirement_id,
                expected_override_id=override.override_id,
                context=str(path),
            )
            _validate_human_semantic_override_history_path(
                path,
                existing,
                checkpoint_root=checkpoint_root,
            )
            if _canonical_json(_semantic_override_stable_payload(existing)) != desired_lineage:
                raise ValueError(
                    "HUMAN_OVERRIDE_HISTORY_CONFLICT: override_id has divergent "
                    f"lineage ({override.override_id})"
                )
    event_path = _semantic_override_history_event_path(checkpoint_root, override)
    if event_path.exists():
        _assert_human_semantic_override_history_event_matches(event_path, override)


def _write_human_semantic_override_history_event(
    checkpoint_root: Path,
    override: HumanSemanticOverride,
) -> None:
    path = _semantic_override_history_event_path(checkpoint_root, override)
    if path.exists():
        _assert_human_semantic_override_history_event_matches(path, override)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **_human_semantic_override_history_payload(override),
        "recorded_at": datetime.now().isoformat(timespec="microseconds"),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            _assert_human_semantic_override_history_event_matches(path, override)
        except OSError:
            # Some filesystems do not support hard links.  Exclusive creation
            # still prevents an existing immutable event from being replaced.
            try:
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(text)
            except FileExistsError:
                _assert_human_semantic_override_history_event_matches(path, override)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_human_semantic_override_history_event_matches(
    path: Path,
    override: HumanSemanticOverride,
) -> None:
    payload = _load_human_semantic_override_history_payload(path)
    _validated_human_semantic_override_history_payload(
        payload,
        expected_requirement_id=override.requirement_id,
        expected_override_id=override.override_id,
        context=str(path),
    )
    existing = dict(payload)
    existing.pop("recorded_at", None)
    desired = _human_semantic_override_history_payload(override)
    if _canonical_json(existing) != _canonical_json(desired):
        raise ValueError(
            "HUMAN_OVERRIDE_HISTORY_CONFLICT: immutable event payload changed "
            f"({override.override_id}/{override.status})"
        )


def _load_human_semantic_override_history_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"HUMAN_OVERRIDE_CORRUPT: unreadable history event {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"HUMAN_OVERRIDE_CORRUPT: invalid history event {path}")
    return payload


def _validated_human_semantic_override_history_payload(
    payload: Mapping[str, Any],
    *,
    expected_requirement_id: str,
    expected_override_id: str | None,
    context: str,
) -> HumanSemanticOverride:
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("pre_ir_contract_version")
        != PRE_IR_CHECKPOINT_CONTRACT_VERSION
        or payload.get("history_contract_version")
        != SEMANTIC_OVERRIDE_HISTORY_CONTRACT_VERSION
        or payload.get("kind") != "human_semantic_override_history_event"
        or not str(payload.get("recorded_at", ""))
    ):
        raise ValueError(f"HUMAN_OVERRIDE_CORRUPT: invalid history event {context}")
    override = _validated_human_semantic_override_mapping(
        payload.get("human_semantic_override"), context=context
    )
    if (
        override.requirement_id != expected_requirement_id
        or payload.get("requirement_id") != override.requirement_id
        or payload.get("override_id") != override.override_id
        or payload.get("event_status") != override.status
        or (expected_override_id is not None and override.override_id != expected_override_id)
    ):
        raise ValueError(
            f"HUMAN_OVERRIDE_CORRUPT: history identity mismatch in {context}"
        )
    return override


def _validated_human_semantic_override_payload(
    payload: Mapping[str, Any],
    *,
    expected_requirement_id: str,
    context: str,
) -> HumanSemanticOverride:
    override = _validated_human_semantic_override_mapping(
        payload.get("human_semantic_override"), context=context
    )
    expected_fields = {
        "requirement_id": override.requirement_id,
        "original_source_hash": override.original_source_hash,
        "parent_effective_source_hash": override.parent_effective_source_hash,
        "replacement_hash": override.replacement_hash,
        "status": override.status,
        "invalidation_reason": override.invalidation_reason,
        "old_effective_source_hash": override.old_effective_source_hash,
        "new_effective_source_hash": override.new_effective_source_hash,
    }
    if override.requirement_id != expected_requirement_id or any(
        payload.get(key) != value for key, value in expected_fields.items()
    ):
        raise ValueError(
            f"HUMAN_OVERRIDE_CORRUPT: checkpoint lineage mismatch in {context}"
        )
    return override


def _validated_human_semantic_override_mapping(
    mapping: Any,
    *,
    context: str,
) -> HumanSemanticOverride:
    if not isinstance(mapping, Mapping):
        raise ValueError(
            f"HUMAN_OVERRIDE_CORRUPT: missing override payload in {context}"
        )
    try:
        override = _human_semantic_override_from_mapping(mapping)
        validate_human_semantic_override_integrity(override)
    except (TypeError, ValueError, PlainReadingValidationError) as exc:
        if str(exc).startswith("HUMAN_OVERRIDE_CORRUPT"):
            raise
        raise ValueError(
            f"HUMAN_OVERRIDE_CORRUPT: invalid override payload in {context}: {exc}"
        ) from exc
    return override


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_human_semantic_override_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    active_only: bool = False,
    pending_only: bool = False,
    expected_original_source_hash: str | None = None,
) -> HumanSemanticOverride | None:
    """Restore an override, preferring the active record over pending input."""

    if active_only and pending_only:
        raise ValueError("override checkpoint cannot be both active-only and pending-only")
    subdirs = (
        ("human_semantic_overrides",)
        if active_only
        else ("pending_semantic_override",)
        if pending_only
        else ("human_semantic_overrides", "pending_semantic_override")
    )
    for subdir in subdirs:
        payload = _load_checkpoint(checkpoint_dir, subdir, requirement_id)
        if payload is None:
            continue
        if (
            payload.get("pre_ir_contract_version")
            != PRE_IR_CHECKPOINT_CONTRACT_VERSION
            or payload.get("kind") != "human_semantic_override"
        ):
            continue
        mapping = payload.get("human_semantic_override")
        if not isinstance(mapping, Mapping):
            raise ValueError(
                "HUMAN_OVERRIDE_CORRUPT: missing override payload in "
                f"{subdir}/{requirement_id}"
            )
        override = _validated_human_semantic_override_payload(
            payload,
            expected_requirement_id=requirement_id,
            context=f"{subdir}/{requirement_id}",
        )
        if override.requirement_id != requirement_id:
            continue
        if (
            expected_original_source_hash is not None
            and override.original_source_hash != expected_original_source_hash
        ):
            continue
        if payload.get("status") != override.status:
            continue
        if subdir == "human_semantic_overrides" and override.status != "active":
            continue
        if subdir == "pending_semantic_override" and override.status == "active":
            continue
        _validate_human_override_checkpoint_artifacts(checkpoint_dir, override)
        return override
    return None


def _human_semantic_override_from_mapping(
    value: Mapping[str, Any],
) -> HumanSemanticOverride:
    return HumanSemanticOverride(
        override_id=str(value.get("override_id", "")),
        requirement_id=str(value.get("requirement_id", "")),
        original_source_text=str(value.get("original_source_text", "")),
        original_source_hash=str(value.get("original_source_hash", "")),
        replacement_text=str(value.get("replacement_text", "")),
        replacement_hash=str(value.get("replacement_hash", "")),
        original_ap_set=tuple(str(item) for item in value.get("original_ap_set", ())),
        replacement_ap_set=tuple(
            str(item) for item in value.get("replacement_ap_set", ())
        ),
        ap_validation_status=str(value.get("ap_validation_status", "pending")),
        parent_effective_source_hash=str(
            value.get("parent_effective_source_hash", "")
        ),
        reviewer_role=str(value.get("reviewer_role", "ordinary_reviewer")),
        reviewer_id=str(value.get("reviewer_id", "")),
        reviewer_source=str(value.get("reviewer_source", "human")),
        confirmation_status=str(value.get("confirmation_status", "pending")),
        status=str(value.get("status", "awaiting_input")),
        issue_codes=tuple(str(item) for item in value.get("issue_codes", ())),
        invalidation_reason=str(value.get("invalidation_reason", "")),
        old_effective_source_hash=str(value.get("old_effective_source_hash", "")),
        new_effective_source_hash=str(value.get("new_effective_source_hash", "")),
        created_at=str(value.get("created_at", "")),
        confirmed_at=str(value.get("confirmed_at", "")),
        override_contract_version=str(
            value.get("override_contract_version", "")
            or HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION
        ),
        ap_validation_contract_version=str(
            value.get("ap_validation_contract_version", "symbolic-token-v1")
        ),
        representation_profile_id=str(value.get("representation_profile_id", "")),
        original_ap_inventory_id=str(value.get("original_ap_inventory_id", "")),
        original_ap_inventory_hash=str(value.get("original_ap_inventory_hash", "")),
        replacement_ap_inventory_id=str(
            value.get("replacement_ap_inventory_id", "")
        ),
        replacement_ap_inventory_hash=str(
            value.get("replacement_ap_inventory_hash", "")
        ),
        ap_comparison_id=str(value.get("ap_comparison_id", "")),
        ap_comparison_hash=str(value.get("ap_comparison_hash", "")),
        ap_validation_mode=str(
            value.get("ap_validation_mode", "symbolic_token_v1")
        ),
        supersedes_override_id=str(value.get("supersedes_override_id", "")),
    )


def save_segmentation_generation_diagnostic(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    diagnostic: Mapping[str, Any],
) -> None:
    """Persist bounded rejected responses from lossless segmentation generation."""

    if not checkpoint_dir:
        return
    if str(diagnostic.get("requirement_id", "")) != requirement_id:
        raise ValueError("segmentation diagnostic requirement_id does not match item")
    attempt = int(diagnostic.get("attempt", 0))
    if attempt < 1:
        raise ValueError("segmentation diagnostic attempt must be positive")
    path = _checkpoint_path(
        checkpoint_dir,
        "segmentation_diagnostics",
        requirement_id,
    )
    attempts: list[dict[str, Any]] = []
    if attempt != 1 and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
            and existing.get("source_hash") == diagnostic.get("source_hash")
        ):
            attempts = [
                dict(item)
                for item in existing.get("attempts", ())
                if isinstance(item, Mapping)
            ]
    attempts.append(dict(diagnostic))
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "segmentation_generation_diagnostic",
        "requirement_id": requirement_id,
        "source_hash": str(diagnostic.get("source_hash", "")),
        "response_sanitization": (
            "marked_text and response shape only; bounded text; provider metadata omitted"
        ),
        "fallback_used": any(bool(item.get("fallback_used")) for item in attempts),
        "attempts": attempts,
        "contract_version": str(diagnostic.get("contract_version", "")),
        "canonicalizer_version": str(diagnostic.get("canonicalizer_version", "")),
        "validator_version": str(diagnostic.get("validator_version", "")),
        "repair_policy_version": str(diagnostic.get("repair_policy_version", "")),
        "artifact_hash": str(diagnostic.get("artifact_hash", "")),
        "issue_fingerprint": str(diagnostic.get("issue_fingerprint", "")),
        "repair_history": attempts,
        "updated_at": _timestamp(),
    }
    _write_json(path, payload)


def load_plain_reading_generated_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    expected_source_hash: str | None = None,
    expected_segmentation_id: str | None = None,
    expected_reading_review_id: str | None = None,
    expected_plain_reading_decision_fingerprint: str | None = None,
) -> dict | None:
    return _load_plain_reading_checkpoint(
        checkpoint_dir,
        "plain_reading_generated",
        requirement_id,
        expected_source_hash=expected_source_hash,
        expected_segmentation_id=expected_segmentation_id,
        expected_reading_review_id=expected_reading_review_id,
        expected_plain_reading_decision_fingerprint=(
            expected_plain_reading_decision_fingerprint
        ),
    )


def save_plain_reading_reviewed_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    plain_reading_review: Any,
    review_history: list[dict] | tuple[Any, ...] | None = None,
    generator: Any | None = None,
    experiment_mode: str | None = None,
    data_split: str | None = None,
    segmentation_schema_version: str = SURFACE_SEGMENTATION_SCHEMA_VERSION,
) -> None:
    """Save a reviewed/no-ambiguity ordinary-language interpretation."""

    _save_plain_reading_checkpoint(
        checkpoint_dir,
        subdir="plain_reading_reviewed",
        requirement_id=requirement_id,
        plain_reading_review=plain_reading_review,
        reviewed=True,
        review_history=review_history,
        generator=generator,
        experiment_mode=experiment_mode,
        data_split=data_split,
        segmentation_schema_version=segmentation_schema_version,
    )


def load_plain_reading_reviewed_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    expected_source_hash: str | None = None,
    expected_segmentation_id: str | None = None,
    expected_reading_review_id: str | None = None,
    expected_plain_reading_decision_fingerprint: str | None = None,
) -> dict | None:
    return _load_plain_reading_checkpoint(
        checkpoint_dir,
        "plain_reading_reviewed",
        requirement_id,
        expected_source_hash=expected_source_hash,
        expected_segmentation_id=expected_segmentation_id,
        expected_reading_review_id=expected_reading_review_id,
        expected_plain_reading_decision_fingerprint=(
            expected_plain_reading_decision_fingerprint
        ),
    )


def save_plain_reading_need_view_checkpoint(
    checkpoint_dir: str | None,
    *,
    need_view: PlainReadingNeedView,
) -> None:
    """Persist an unresolved-reading view separately from semantic candidates."""

    if not checkpoint_dir:
        return
    if not isinstance(need_view, PlainReadingNeedView):
        raise TypeError("plain-reading need-view checkpoint requires typed artifact")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "kind": "plain_reading_need_view",
        "requirement_id": need_view.requirement_id,
        "source_hash": need_view.source_hash,
        "segmentation_id": need_view.segmentation_id,
        "reading_review_id": need_view.reading_review_id,
        "parent_fingerprint": need_view.parent_fingerprint,
        "artifact_hash": need_view.artifact_hash,
        "plain_reading_need_view": asdict(need_view),
        "updated_at": _timestamp(),
    }
    _write_json(
        _checkpoint_path(
            checkpoint_dir,
            "plain_reading_need_view",
            need_view.requirement_id,
        ),
        payload,
    )


def load_plain_reading_need_view_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    expected_source_hash: str | None = None,
    expected_segmentation_id: str | None = None,
    expected_reading_review_id: str | None = None,
    expected_parent_fingerprint: str | None = None,
) -> PlainReadingNeedView | None:
    payload = _load_checkpoint(
        checkpoint_dir,
        "plain_reading_need_view",
        requirement_id,
    )
    if payload is None:
        return None
    mapping = payload.get("plain_reading_need_view")
    if not isinstance(mapping, Mapping):
        return None
    try:
        view = plain_reading_need_view_from_mapping(mapping)
    except (KeyError, TypeError, ValueError):
        return None
    expected = {
        "requirement_id": requirement_id,
        "source_hash": expected_source_hash,
        "segmentation_id": expected_segmentation_id,
        "reading_review_id": expected_reading_review_id,
        "parent_fingerprint": expected_parent_fingerprint,
    }
    for field, value in expected.items():
        if value is not None and getattr(view, field) != value:
            return None
    if payload.get("artifact_hash") != view.artifact_hash:
        return None
    return view


# Keep both natural naming orders available while callers are migrated.
save_generated_segmentation_checkpoint = save_segmentation_generated_checkpoint
load_generated_segmentation_checkpoint = load_segmentation_generated_checkpoint
save_reviewed_segmentation_checkpoint = save_segmentation_reviewed_checkpoint
load_reviewed_segmentation_checkpoint = load_segmentation_reviewed_checkpoint
save_generated_plain_reading_checkpoint = save_plain_reading_generated_checkpoint
load_generated_plain_reading_checkpoint = load_plain_reading_generated_checkpoint
save_reviewed_plain_reading_checkpoint = save_plain_reading_reviewed_checkpoint
load_reviewed_plain_reading_checkpoint = load_plain_reading_reviewed_checkpoint


def save_generated_review_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    trace_coverage: TraceCoverageReport | None = None,
    surface_segmentation: Any | None = None,
    plain_reading_review: Any | None = None,
    source_hash: str = "",
    segmentation_id: str = "",
    reading_review_id: str = "",
    plain_reading_decision_fingerprint: str = "",
    segmentation_schema_version: str = "",
    plain_reading_schema_version: str = "",
    trace_evidence_records: tuple[TraceEvidenceRecord, ...] | list[TraceEvidenceRecord] = (),
    trace_evidence_disputes: tuple[TraceEvidenceDispute, ...] | list[TraceEvidenceDispute] = (),
    review_records: list[dict] | None = None,
) -> None:
    """Save semantic/traces generated for one item before review approval."""

    _save_review_checkpoint(
        checkpoint_dir,
        subdir="review_generated",
        requirement_id=requirement_id,
        requirement=requirement,
        semantic=semantic,
        traces=traces,
        review_records=list(review_records or ()),
        approved=False,
        trace_coverage=trace_coverage,
        surface_segmentation=surface_segmentation,
        plain_reading_review=plain_reading_review,
        source_hash=source_hash,
        segmentation_id=segmentation_id,
        reading_review_id=reading_review_id,
        plain_reading_decision_fingerprint=plain_reading_decision_fingerprint,
        segmentation_schema_version=segmentation_schema_version,
        plain_reading_schema_version=plain_reading_schema_version,
        trace_evidence_records=trace_evidence_records,
        trace_evidence_disputes=trace_evidence_disputes,
    )


def save_generated_semantic_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    surface_segmentation: Any | None = None,
    plain_reading_review: Any | None = None,
    generation_diagnostics: Mapping[str, object] | None = None,
) -> None:
    """Persist validated semantic IR before any trace synthesis can fail."""

    if not checkpoint_dir:
        return
    surface_segmentation = surface_segmentation or getattr(
        semantic, "surface_segmentation", None
    )
    plain_reading_review = plain_reading_review or getattr(
        semantic, "plain_reading_review", None
    )
    surface_mapping = _artifact_mapping(surface_segmentation)
    plain_mapping = _artifact_mapping(plain_reading_review)
    semantic_payload = asdict(semantic)
    semantic_payload["surface_segmentation"] = surface_mapping
    semantic_payload["plain_reading_review"] = plain_mapping
    ir = semantic.faithful_ir
    metadata = _pre_ir_metadata(
        requirement=requirement,
        surface_segmentation=surface_segmentation,
        plain_reading_review=plain_reading_review,
        source_hash_value="",
        segmentation_id="",
        reading_review_id="",
        decision_fingerprint="",
        segmentation_schema_version="",
        plain_reading_schema_version="",
    )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "obligation_compiler_version": OBLIGATION_COMPILER_VERSION,
        "trace_contract_version": TRACE_CONTRACT_VERSION,
        "semantic_evidence_contract_version": SEMANTIC_EVIDENCE_CONTRACT_VERSION,
        "faithful_ir_runtime_contract": faithful_ir_runtime_contract(),
        "kind": "semantic_snapshot",
        "requirement_id": requirement_id,
        "requirement": requirement,
        "semantic": semantic_payload,
        "surface_segmentation": surface_mapping,
        "plain_reading_review": plain_mapping,
        **metadata,
        "ir_version": ir.ir_version if ir is not None else "",
        "ap_context_fingerprint": (
            ir.ap_context_fingerprint if ir is not None else ""
        ),
        "ap_grounding_fingerprint": (
            ir.ap_grounding_fingerprint if ir is not None else ""
        ),
        "effective_source_fingerprint": (
            ir.effective_source_fingerprint if ir is not None else ""
        ),
        "trace_obligation_plan": None,
        "trace_obligation_plan_error": None,
        "semantic_readiness": None,
        "generation_diagnostics": dict(generation_diagnostics or {}),
        "updated_at": _timestamp(),
    }
    snapshot_path = _checkpoint_path(
        checkpoint_dir,
        "semantic_generated",
        requirement_id,
    )
    # Candidate persistence is the first state transition.  Readiness and
    # obligation planning are downstream diagnostics and must never run while
    # the only candidate exists in memory.
    _write_json(snapshot_path, payload)

    if ir is not None:
        try:
            readiness = audit_semantic_readiness(ir)
            payload["semantic_readiness"] = asdict(readiness)
            plan = compile_review_trace_plan(ir)
            payload["trace_obligation_plan"] = asdict(plan)
        except Exception as exc:  # The persisted candidate remains recoverable.
            payload["trace_obligation_plan_error"] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        _write_json(snapshot_path, payload)


def load_generated_semantic_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    expected_ap_context_fingerprint: str | None = None,
) -> dict | None:
    """Load a replayable pre-trace semantic snapshot without mutating state."""

    payload = _load_checkpoint(checkpoint_dir, "semantic_generated", requirement_id)
    return _reject_stale_ap_context_payload(
        payload,
        expected_ap_context_fingerprint,
    )


def save_reviewed_item_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    review_records: list[dict],
    trace_coverage: TraceCoverageReport | None = None,
    surface_segmentation: Any | None = None,
    plain_reading_review: Any | None = None,
    source_hash: str = "",
    segmentation_id: str = "",
    reading_review_id: str = "",
    plain_reading_decision_fingerprint: str = "",
    segmentation_schema_version: str = "",
    plain_reading_schema_version: str = "",
    trace_evidence_records: tuple[TraceEvidenceRecord, ...] | list[TraceEvidenceRecord] = (),
    trace_evidence_disputes: tuple[TraceEvidenceDispute, ...] | list[TraceEvidenceDispute] = (),
) -> None:
    """Save a semantic/trace reviewed item that can enter the later pipeline."""

    _save_review_checkpoint(
        checkpoint_dir,
        subdir="reviewed_items",
        requirement_id=requirement_id,
        requirement=requirement,
        semantic=semantic,
        traces=traces,
        review_records=review_records,
        approved=True,
        trace_coverage=trace_coverage,
        surface_segmentation=surface_segmentation,
        plain_reading_review=plain_reading_review,
        source_hash=source_hash,
        segmentation_id=segmentation_id,
        reading_review_id=reading_review_id,
        plain_reading_decision_fingerprint=plain_reading_decision_fingerprint,
        segmentation_schema_version=segmentation_schema_version,
        plain_reading_schema_version=plain_reading_schema_version,
        trace_evidence_records=trace_evidence_records,
        trace_evidence_disputes=trace_evidence_disputes,
    )


def load_reviewed_item_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    expected_ir_version: str | None = None,
    expected_source_hash: str | None = None,
    expected_segmentation_id: str | None = None,
    expected_reading_review_id: str | None = None,
    expected_plain_reading_decision_fingerprint: str | None = None,
    expected_ap_context_fingerprint: str | None = None,
) -> dict | None:
    payload = _load_checkpoint(checkpoint_dir, "reviewed_items", requirement_id)
    payload = _migrate_review_evidence_contract_payload(payload)
    payload = _reject_stale_pre_ir_payload(
        payload,
        expected_source_hash=expected_source_hash,
        expected_segmentation_id=expected_segmentation_id,
        expected_reading_review_id=expected_reading_review_id,
        expected_plain_reading_decision_fingerprint=(
            expected_plain_reading_decision_fingerprint
        ),
    )
    payload = _reject_stale_ap_context_payload(
        payload,
        expected_ap_context_fingerprint,
    )
    return _reject_stale_ir_payload(payload, expected_ir_version)


def load_generated_review_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    expected_ir_version: str | None = None,
    expected_source_hash: str | None = None,
    expected_segmentation_id: str | None = None,
    expected_reading_review_id: str | None = None,
    expected_plain_reading_decision_fingerprint: str | None = None,
    expected_ap_context_fingerprint: str | None = None,
) -> dict | None:
    payload = _load_checkpoint(checkpoint_dir, "review_generated", requirement_id)
    payload = _migrate_review_evidence_contract_payload(payload)
    payload = _reject_stale_pre_ir_payload(
        payload,
        expected_source_hash=expected_source_hash,
        expected_segmentation_id=expected_segmentation_id,
        expected_reading_review_id=expected_reading_review_id,
        expected_plain_reading_decision_fingerprint=(
            expected_plain_reading_decision_fingerprint
        ),
    )
    payload = _reject_stale_ap_context_payload(
        payload,
        expected_ap_context_fingerprint,
    )
    return _reject_stale_ir_payload(payload, expected_ir_version)


def save_pipeline_result_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    requirement: str,
    report_text: str,
    summary_text: str,
    review_records: list[dict],
    classification: str = "complete",
    trace_coverage: TraceCoverageReport | None = None,
    selection_status: str = "",
    selected_formula: str = "",
    degraded_record: DegradedResultRecord | None = None,
    degraded_item_path: str = "",
    expert_review_status: str = "",
    ir_version: str = "",
    ir_coverage: IRCoverageReport | None = None,
    operator_resolution: Any = None,
    candidate_status: str = "",
    run_origin: str = "normal",
    automatic_finalization_allowed: bool = False,
    final_approval: FinalApprovalRecord | None = None,
    source_hash: str = "",
    segmentation_id: str = "",
    reading_review_id: str = "",
    plain_reading_decision_fingerprint: str = "",
    segmentation_schema_version: str = "",
    plain_reading_schema_version: str = "",
    semantic_status: str = "",
    evidence_status: str = "",
    approval_status: str = "",
    learning_status: str = "",
    source_run_id: str = "",
    formula_closure: Any = None,
    ambiguity_closure: Any = None,
    artifact_fingerprints: Any = None,
    error_diagnoses: tuple[Any, ...] = (),
    rerun_delta: Any = None,
    feedback_patch: Any = None,
    trace_evidence_policy: Any = None,
    canonical_trace_coverage: TraceCoverageReport | None = None,
    experimental_only: bool = False,
    boundary_metric_available: bool = True,
    ap_context_fingerprint: str = "",
    relation_feasibility_classification: str = "",
    feasibility_report_hash: str = "",
) -> None:
    if not checkpoint_dir:
        return
    metadata = _pre_ir_metadata(
        requirement=requirement,
        surface_segmentation=None,
        plain_reading_review=None,
        source_hash_value=source_hash,
        segmentation_id=segmentation_id,
        reading_review_id=reading_review_id,
        decision_fingerprint=plain_reading_decision_fingerprint,
        segmentation_schema_version=segmentation_schema_version,
        plain_reading_schema_version=plain_reading_schema_version,
    )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "obligation_compiler_version": OBLIGATION_COMPILER_VERSION,
        "trace_contract_version": TRACE_CONTRACT_VERSION,
        "semantic_evidence_contract_version": SEMANTIC_EVIDENCE_CONTRACT_VERSION,
        "faithful_ir_runtime_contract": faithful_ir_runtime_contract(),
        "kind": "pipeline_result",
        "requirement_id": requirement_id,
        "requirement": requirement,
        "report_text": report_text,
        "summary_text": summary_text,
        "review_records": review_records,
        "classification": classification,
        "trace_coverage": asdict(trace_coverage or TraceCoverageReport()),
        "canonical_trace_coverage": asdict(
            canonical_trace_coverage or trace_coverage or TraceCoverageReport()
        ),
        "trace_evidence_policy": (
            asdict(trace_evidence_policy)
            if is_dataclass(trace_evidence_policy)
            else trace_evidence_policy
        ),
        "experimental_only": experimental_only,
        "boundary_metric_available": boundary_metric_available,
        "selection_status": selection_status,
        "candidate_status": candidate_status,
        "run_origin": run_origin,
        "automatic_finalization_allowed": automatic_finalization_allowed,
        "final_approval": (
            asdict(final_approval) if final_approval is not None else None
        ),
        "semantic_status": semantic_status,
        "evidence_status": evidence_status,
        "approval_status": approval_status,
        "learning_status": learning_status,
        "source_run_id": source_run_id,
        "formula_closure": (
            asdict(formula_closure)
            if is_dataclass(formula_closure)
            else formula_closure
        ),
        "ambiguity_closure": (
            asdict(ambiguity_closure)
            if is_dataclass(ambiguity_closure)
            else ambiguity_closure
        ),
        "artifact_fingerprints": (
            artifact_fingerprints.as_dict()
            if hasattr(artifact_fingerprints, "as_dict")
            else artifact_fingerprints
        ),
        "error_diagnoses": [
            asdict(item) if is_dataclass(item) else item
            for item in error_diagnoses
        ],
        "rerun_delta": (
            asdict(rerun_delta) if is_dataclass(rerun_delta) else rerun_delta
        ),
        "feedback_patch": (
            asdict(feedback_patch)
            if is_dataclass(feedback_patch)
            else feedback_patch
        ),
        "ir_version": ir_version,
        "ap_context_fingerprint": ap_context_fingerprint,
        "relation_feasibility_classification": (
            relation_feasibility_classification
        ),
        "feasibility_report_hash": feasibility_report_hash,
        "ir_coverage": asdict(ir_coverage or IRCoverageReport()),
        "operator_resolution": (
            asdict(operator_resolution)
            if is_dataclass(operator_resolution)
            else operator_resolution
        ),
        "selected_formula": selected_formula,
        **metadata,
        "degraded_record": (
            asdict(degraded_record) if degraded_record is not None else None
        ),
        "degraded_item_path": degraded_item_path,
        "expert_review_status": (
            expert_review_status
            or (
                degraded_record.expert_review_status
                if degraded_record is not None
                else ""
            )
        ),
        "updated_at": _timestamp(),
    }
    _write_json(_checkpoint_path(checkpoint_dir, "pipeline_results", requirement_id), payload)
    update_checkpoint_classification(
        checkpoint_dir,
        requirement_id=requirement_id,
        classification=classification,
        degraded_item_path=degraded_item_path,
        expert_review_status=payload["expert_review_status"],
    )


def load_pipeline_result_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
    *,
    expected_ir_version: str | None = None,
    expected_source_hash: str | None = None,
    expected_segmentation_id: str | None = None,
    expected_reading_review_id: str | None = None,
    expected_plain_reading_decision_fingerprint: str | None = None,
) -> dict | None:
    payload = _load_checkpoint(checkpoint_dir, "pipeline_results", requirement_id)
    payload = _reject_stale_evidence_contract_payload(payload)
    payload = _reject_stale_pre_ir_payload(
        payload,
        expected_source_hash=expected_source_hash,
        expected_segmentation_id=expected_segmentation_id,
        expected_reading_review_id=expected_reading_review_id,
        expected_plain_reading_decision_fingerprint=(
            expected_plain_reading_decision_fingerprint
        ),
    )
    if (
        payload is not None
        and expected_ir_version
        and payload.get("ir_version") != expected_ir_version
    ):
        return None
    if payload is not None:
        closure = payload.get("formula_closure")
        final_approval = payload.get("final_approval")
        ir_version = str(payload.get("ir_version", ""))
        if payload.get("candidate_status") == "final":
            if not isinstance(closure, Mapping):
                return None
            else:
                if closure.get("status") != "passed":
                    return None
                selected_formula = str(payload.get("selected_formula", ""))
                try:
                    selected_hash = stable_formula_hash(
                        parse_formula(selected_formula)
                    )
                except ValueError:
                    return None
                if selected_hash != str(closure.get("formula_hash", "")):
                    return None
                if str(closure.get("ir_version", "")) != ir_version:
                    return None
                if not isinstance(final_approval, Mapping):
                    return None
                if str(final_approval.get("approved_ir_version", "")) != ir_version:
                    return None
                if str(final_approval.get("approved_formula", "")) != selected_formula:
                    return None
    return payload


def update_pipeline_expert_review_status(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    expert_review_status: str,
    expert_comments: str = "",
) -> None:
    """Persist expert edits discovered in the degraded item JSON."""

    if not checkpoint_dir:
        return
    path = _checkpoint_path(checkpoint_dir, "pipeline_results", requirement_id)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["expert_review_status"] = expert_review_status
    degraded_record = payload.get("degraded_record")
    if isinstance(degraded_record, dict):
        degraded_record["expert_review_status"] = expert_review_status
        if expert_comments:
            degraded_record["expert_comments"] = expert_comments
    payload["updated_at"] = _timestamp()
    _write_json(path, payload)
    update_checkpoint_classification(
        checkpoint_dir,
        requirement_id=requirement_id,
        classification=str(payload.get("classification", "degraded_unproven")),
        degraded_item_path=str(payload.get("degraded_item_path", "")),
        expert_review_status=expert_review_status,
    )


def save_failed_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    stage: str,
    error: BaseException,
    suggested_rerun_stage: str = "",
    details: dict | None = None,
) -> None:
    if not checkpoint_dir:
        return
    from failure_routing import classify_failure

    route = classify_failure(error)
    if route.expected_pending:
        save_needs_review_checkpoint(
            checkpoint_dir,
            requirement_id=requirement_id,
            stage=route.owner_stage or stage,
            reason_code=route.reason_code,
            message=str(error),
            suggested_rerun_stage=(
                route.safe_restart_stage
                or suggested_rerun_stage
                or "semantic"
            ),
            details=details,
        )
        return
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "failed_item",
        "requirement_id": requirement_id,
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "suggested_rerun_stage": suggested_rerun_stage,
        "traceback": _bounded_traceback(error),
        "updated_at": _timestamp(),
    }
    payload["disposition"] = route.disposition
    payload["blocked_failure"] = asdict(
        BlockedFailureRecord(
            reason_code=str(
                getattr(error, "reason_code", "")
                or getattr(error, "code", "")
                or type(error).__name__
            ).upper(),
            owner_stage=route.owner_stage or stage,
            safe_restart_stage=(
                route.safe_restart_stage
                or suggested_rerun_stage
                or "pipeline"
            ),
            requirement_id=requirement_id,
            error_type=type(error).__name__,
            message=str(error),
            source_ir_version=str((details or {}).get("ir_version", "")),
            retryable=route.retryable,
        )
    )
    if details:
        payload["details"] = details
    _write_json(_checkpoint_path(checkpoint_dir, "failed_items", requirement_id), payload)
    _write_json(Path(checkpoint_dir) / "failed_item.json", payload)
    update_checkpoint_classification(
        checkpoint_dir,
        requirement_id=requirement_id,
        classification=route.disposition,
        error=payload,
    )


def save_needs_review_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    stage: str,
    reason_code: str,
    message: str,
    suggested_rerun_stage: str,
    details: Mapping[str, object] | None = None,
) -> None:
    """Persist a recoverable semantic/solver review state without traceback."""

    if not checkpoint_dir:
        return
    from failure_routing import failure_route_for_code

    route = failure_route_for_code(reason_code)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "needs_review_item",
        "status": "needs_review",
        "requirement_id": requirement_id,
        "stage": stage,
        "reason_code": reason_code,
        "message": message,
        "suggested_rerun_stage": suggested_rerun_stage,
        "details": dict(details or {}),
        "updated_at": _timestamp(),
    }
    evidence_paths = tuple(
        str(value)
        for key in ("record_path", "review_record_path")
        if (value := (details or {}).get(key))
    )
    payload["disposition"] = (
        route.disposition
        if route.expected_pending
        else "awaiting_semantic_resolution"
    )
    payload["review_pending"] = asdict(
        ReviewPendingRecord(
            reason_code=reason_code,
            owner_stage=stage,
            safe_restart_stage=suggested_rerun_stage,
            requirement_id=requirement_id,
            source_ir_version=str((details or {}).get("ir_version", "")),
            affected_ir_node_ids=tuple(
                str(value)
                for value in (details or {}).get("affected_ir_node_ids", ())
            ),
            dispute_ids=tuple(
                str(value) for value in (details or {}).get("dispute_ids", ())
            ),
            challenge_ids=tuple(
                str(value) for value in (details or {}).get("challenge_ids", ())
            ),
            evidence_record_paths=evidence_paths,
            required_action="manual_review",
        )
    )
    _write_json(
        _checkpoint_path(checkpoint_dir, "needs_review_items", requirement_id),
        payload,
    )
    update_checkpoint_classification(
        checkpoint_dir,
        requirement_id=requirement_id,
        classification=payload["disposition"],
        error=payload,
    )


def load_needs_review_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
) -> dict | None:
    return _load_checkpoint(checkpoint_dir, "needs_review_items", requirement_id)


def save_semantic_review_pause_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    question: str,
    request: dict | None = None,
    options: list | tuple = (),
    diagnostics: dict | None = None,
) -> None:
    """Persist a normal human-decision pause without recording a traceback."""

    if not checkpoint_dir:
        return
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "semantic_review_pause",
        "status": "waiting_for_human_semantic_choice",
        "requirement_id": requirement_id,
        "stage": "semantic_policy_review",
        "question": question,
        "request": request or {},
        "options": list(options),
        "suggested_rerun_stage": "semantic",
        "updated_at": _timestamp(),
    }
    payload["disposition"] = "review_pending"
    payload["review_pending"] = asdict(
        ReviewPendingRecord(
            reason_code="HUMAN_SEMANTIC_CHOICE_REQUIRED",
            owner_stage="semantic_policy_review",
            safe_restart_stage="semantic",
            requirement_id=requirement_id,
            required_action="select_reviewed_semantic_reading",
        )
    )
    if diagnostics:
        payload["diagnostics"] = diagnostics
    _write_json(
        _checkpoint_path(
            checkpoint_dir,
            "semantic_review_pauses",
            requirement_id,
        ),
        payload,
    )
    update_checkpoint_classification(
        checkpoint_dir,
        requirement_id=requirement_id,
        classification="waiting_for_human_semantic_choice",
        error=payload,
    )


def save_semantic_policy_candidates_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    queries: list[Any] | tuple[Any, ...],
    review_records: list[dict] | tuple[dict, ...] = (),
) -> None:
    """Persist an unresolved, non-error policy CandidateIRSet and its queries."""

    if not checkpoint_dir:
        return
    candidate_set = semantic.candidate_ir_set
    if candidate_set is None:
        raise ValueError("semantic policy candidate checkpoint requires CandidateIRSet")
    from formula_generation_fallback import is_formula_generation_fallback
    generated_formula = is_formula_generation_fallback(semantic)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "kind": "semantic_policy_candidates",
        "status": "candidate_generated" if generated_formula else "awaiting_expert_trace_review",
        "requirement_id": requirement_id,
        "requirement": requirement,
        "source_hash": (
            semantic.faithful_ir.source_hash
            if semantic.faithful_ir is not None
            else ""
        ),
        "segmentation_id": (
            semantic.surface_segmentation.segmentation_id
            if semantic.surface_segmentation is not None
            else ""
        ),
        "reading_review_id": (
            semantic.plain_reading_review.reading_review_id
            if semantic.plain_reading_review is not None
            else ""
        ),
        "plain_reading_decision_fingerprint": (
            decision_fingerprint(semantic.plain_reading_review)
            if semantic.plain_reading_review is not None
            else ""
        ),
        "segmentation_schema_version": (
            semantic.surface_segmentation.schema_version
            if semantic.surface_segmentation is not None
            else ""
        ),
        "plain_reading_schema_version": (
            semantic.plain_reading_review.schema_version
            if semantic.plain_reading_review is not None
            else ""
        ),
        "branch_set_id": candidate_set.branch_set_id,
        "candidate_catalog_version": candidate_set.candidate_catalog_version,
        "semantic": asdict(semantic),
        "review_records": list(review_records),
        "queries": [
            {
                **asdict(query),
                "trace": [sorted(step) for step in query.trace],
            }
            for query in queries
        ],
        "updated_at": _timestamp(),
    }
    _write_json(
        _checkpoint_path(
            checkpoint_dir,
            "semantic_policy_candidates",
            requirement_id,
        ),
        payload,
    )
    _supersede_legacy_semantic_pause(checkpoint_dir, requirement_id)
    update_checkpoint_classification(
        checkpoint_dir,
        requirement_id=requirement_id,
        classification="complete_with_corrections" if generated_formula else "awaiting_expert_semantic_review",
        expert_review_status="not_approved" if generated_formula else "pending",
        error=None,
    )


def load_semantic_policy_candidates_checkpoint(
    checkpoint_dir: str | None,
    requirement_id: str,
) -> dict | None:
    if not checkpoint_dir:
        return None
    return _load_checkpoint(
        checkpoint_dir,
        "semantic_policy_candidates",
        requirement_id,
    )


def save_semantic_policy_decision_checkpoint(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    semantic: SemanticStructure,
    selection: Mapping[str, Any],
) -> None:
    """Persist resolved or provisional automated IR branch selection."""

    if not checkpoint_dir:
        return
    selection_status = str(selection.get("status", "resolved"))
    resolved = selection_status in {"unique", "unique_resolved", "resolved"}
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "semantic_policy_decision",
        "status": "resolved" if resolved else "provisional",
        "requirement_id": requirement_id,
        "selection": dict(selection),
        "semantic": asdict(semantic),
        "updated_at": _timestamp(),
    }
    _write_json(
        _checkpoint_path(
            checkpoint_dir,
            "semantic_policy_decisions",
            requirement_id,
        ),
        payload,
    )
    update_checkpoint_classification(
        checkpoint_dir,
        requirement_id=requirement_id,
        classification=(
            "semantic_policy_resolved"
            if resolved
            else "provisional_ir_candidate_selected"
        ),
        expert_review_status="resolved" if resolved else "pending",
        error=None,
    )


def _supersede_legacy_semantic_pause(
    checkpoint_dir: str,
    requirement_id: str,
) -> None:
    path = _checkpoint_path(
        checkpoint_dir,
        "semantic_review_pauses",
        requirement_id,
    )
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(payload, dict) or payload.get("status") == "superseded":
        return
    payload["status"] = "superseded"
    payload["superseded_by"] = "semantic_policy_candidates"
    payload["updated_at"] = _timestamp()
    _write_json(path, payload)


def _bounded_traceback(error: BaseException, *, limit: int = 12000) -> str:
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    if len(rendered) <= limit:
        return rendered
    return "...<traceback truncated>...\n" + rendered[-limit:]


def update_checkpoint_classification(
    checkpoint_dir: str | None,
    *,
    requirement_id: str,
    classification: str,
    degraded_item_path: str = "",
    expert_review_status: str = "",
    error: dict | None = None,
) -> None:
    """Atomically update the per-run checkpoint classification index."""

    if not checkpoint_dir:
        return
    path = Path(checkpoint_dir) / "classification_manifest.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "items": {},
        }
    )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "items": {},
        }
    payload.setdefault("items", {})[requirement_id] = {
        "classification": classification,
        "degraded_item_path": degraded_item_path,
        "expert_review_status": expert_review_status,
        "error": error,
        "updated_at": _timestamp(),
    }
    payload["updated_at"] = _timestamp()
    _write_json(path, payload)


def _is_generated_formula_ir(payload):
    from formula_generation_fallback import EXTRACTOR
    provenance = payload.get("provenance", {})
    return (isinstance(provenance, Mapping) and provenance.get("extractor") == EXTRACTOR
            and provenance.get("extraction_mode") == "working_interpretation_v1")


def semantic_from_checkpoint(payload: dict) -> SemanticStructure:
    checkpoint_payload = payload if isinstance(payload.get("semantic"), dict) else None
    data = dict(checkpoint_payload["semantic"] if checkpoint_payload else payload)
    if checkpoint_payload is not None:
        for field_name in (
            "surface_segmentation",
            "plain_reading_review",
            "candidate_ir_set",
        ):
            if data.get(field_name) is None and checkpoint_payload.get(field_name) is not None:
                data[field_name] = checkpoint_payload[field_name]
    data["rag_context"] = tuple(data.get("rag_context", ()))
    data["aps"] = tuple(data.get("aps", ()))
    data["components"] = tuple(tuple(item) for item in data.get("components", ()))
    data["domain_evidence"] = tuple(data.get("domain_evidence", ()))
    data["formula_visibility"] = tuple(data.get("formula_visibility", ()))
    data["allowed_rag_stages"] = tuple(data.get("allowed_rag_stages", ()))
    data["semantic_policy_resolutions"] = tuple(
        dict(item)
        for item in data.get("semantic_policy_resolutions", ())
        if isinstance(item, Mapping)
    )
    data["semantic_policy_target_bindings"] = tuple(
        dict(item)
        for item in data.get("semantic_policy_target_bindings", ())
        if isinstance(item, Mapping)
    )
    data["source_axis_debt_resolutions"] = tuple(
        dict(item) for item in data.get("source_axis_debt_resolutions", ()) if isinstance(item, Mapping)
    )
    surface_segmentation = data.get("surface_segmentation")
    if isinstance(surface_segmentation, Mapping):
        data["surface_segmentation"] = _surface_segmentation_from_mapping(
            surface_segmentation
        )
    plain_reading_review = data.get("plain_reading_review")
    if isinstance(plain_reading_review, Mapping):
        data["plain_reading_review"] = _plain_reading_from_mapping(
            plain_reading_review,
            segmentation=data.get("surface_segmentation"),
        )
    faithful_ir = data.get("faithful_ir")
    if isinstance(faithful_ir, dict):
        working = _is_generated_formula_ir(faithful_ir)
        data["faithful_ir"] = faithful_ir_from_mapping(
            faithful_ir,
            source_text=str(faithful_ir.get("source_text", "")),
            requirement_id=str(faithful_ir.get("requirement_id", "")),
            extractor="checkpoint",
            trusted_provenance=True,
            working_interpretation=working,
            surface_segmentation=None if working else data.get("surface_segmentation"),
            plain_reading_review=None if working else data.get("plain_reading_review"),
        )
        if working:
            from faithful_ir import validate_faithful_ir
            validate_faithful_ir(data["faithful_ir"], surface_segmentation=data.get("surface_segmentation"),
                                 plain_reading_review=data.get("plain_reading_review"))
    if isinstance(data.get("candidate_ir_set"), Mapping):
        data["candidate_ir_set"] = _candidate_ir_set_from_mapping(
            data["candidate_ir_set"]
        )
    semantic_resolution = data.get("semantic_resolution")
    if isinstance(semantic_resolution, Mapping):
        semantic_resolution = dict(semantic_resolution)
        semantic_resolution["decisions"] = tuple(
            item
            if isinstance(item, SemanticParameterDecision)
            else _semantic_parameter_decision_from_mapping(item)
            for item in semantic_resolution.get("decisions", ())
        )
        semantic_resolution["unresolved_questions"] = tuple(
            semantic_resolution.get("unresolved_questions", ())
        )
        data["semantic_resolution"] = SemanticResolutionRecord(
            **semantic_resolution
        )
    semantic_challenges = data.get("semantic_challenges")
    if isinstance(semantic_challenges, Mapping):
        semantic_challenges = dict(semantic_challenges)
        semantic_challenges["challenges"] = tuple(
            _semantic_challenge_from_mapping(item)
            for item in semantic_challenges.get("challenges", ())
        )
        data["semantic_challenges"] = SemanticChallengeSet(
            **semantic_challenges
        )
    ir_adjudication = data.get("ir_ambiguity_adjudication")
    if isinstance(ir_adjudication, Mapping):
        data["ir_ambiguity_adjudication"] = (
            _ir_ambiguity_adjudication_from_mapping(ir_adjudication)
        )
    approved_semantic_ir = data.get("approved_semantic_ir")
    if isinstance(approved_semantic_ir, dict):
        approved_semantic_ir = dict(approved_semantic_ir)
        approved_semantic_ir["approved_ap_mapping"] = tuple(
            tuple(item)
            for item in approved_semantic_ir.get("approved_ap_mapping", ())
        )
        approved_semantic_ir["resolved_ambiguity_axes"] = tuple(
            approved_semantic_ir.get("resolved_ambiguity_axes", ())
        )
        approved_semantic_ir["remaining_ambiguity_debt"] = tuple(
            approved_semantic_ir.get("remaining_ambiguity_debt", ())
        )
        approved_semantic_ir["labeled_trace_evidence_ids"] = tuple(
            approved_semantic_ir.get("labeled_trace_evidence_ids", ())
        )
        data["approved_semantic_ir"] = ApprovedSemanticIR(**approved_semantic_ir)
    labeled_trace_evidence = data.get("labeled_trace_evidence")
    if isinstance(labeled_trace_evidence, dict):
        labeled_trace_evidence = dict(labeled_trace_evidence)
        for key in (
            "trace_ids",
            "positive_trace_ids",
            "negative_trace_ids",
            "boundary_trace_ids",
            "unclear_trace_ids",
            "review_ids",
        ):
            labeled_trace_evidence[key] = tuple(labeled_trace_evidence.get(key, ()))
        data["labeled_trace_evidence"] = LabeledTraceEvidenceSet(
            **labeled_trace_evidence
        )
    semantic_evidence_gate = data.get("semantic_evidence_gate")
    if isinstance(semantic_evidence_gate, dict):
        semantic_evidence_gate = dict(semantic_evidence_gate)
        for key in (
            "required_obligation_ids",
            "covered_obligation_ids",
            "missing_obligation_ids",
            "required_boundary_obligation_ids",
            "covered_boundary_obligation_ids",
            "trace_ids",
            "conflicting_trace_ids",
            "remaining_ambiguity_debt",
            "reasons",
        ):
            semantic_evidence_gate[key] = tuple(
                semantic_evidence_gate.get(key, ())
            )
        data["semantic_evidence_gate"] = SemanticEvidenceGateReport(
            **semantic_evidence_gate
        )
    semantic = SemanticStructure(**data)
    from semantic_adjudication import validate_semantic_resolution_binding

    if semantic.faithful_ir is not None:
        readiness = audit_semantic_readiness(semantic.faithful_ir)
        if (
            semantic.semantic_resolution is not None
            and not readiness.has_capability(CAN_GENERATE_AUTHORITATIVE_TRACES)
        ):
            stale_ir = semantic.faithful_ir
            migrated_ir = replace(
                stale_ir,
                resolution_id="",
                resolution_fingerprint="",
                ir_version="",
                migration_history=(
                    *stale_ir.migration_history,
                    {
                        "kind": "semantic_readiness_reclassification",
                        "previous_resolution_id": stale_ir.resolution_id,
                        "readiness_fingerprint": readiness.fingerprint,
                        "reason": (
                            "persisted resolution binding was incomplete under "
                            "the current semantic capability contract"
                        ),
                    },
                ),
            )
            migrated_ir = replace(
                migrated_ir,
                ir_version=compute_ir_version(migrated_ir),
            )
            semantic = replace(
                semantic,
                faithful_ir=migrated_ir,
                semantic_resolution=None,
                approved_semantic_ir=None,
                labeled_trace_evidence=None,
                semantic_evidence_gate=None,
                semantic_status="needs_semantic_review",
                persistent_rag_allowed=False,
            )
    validate_semantic_resolution_binding(semantic)
    _validate_semantic_pre_ir_parents(semantic, checkpoint_payload)
    return semantic


def _semantic_challenge_from_mapping(value: Any) -> SemanticChallenge:
    if isinstance(value, SemanticChallenge):
        return value
    data = dict(value)
    for key in (
        "affected_node_ids",
        "affected_parameter_paths",
        "related_axis_ids",
        "proposed_readings",
        "supported_text_spans",
        "evidence",
    ):
        data[key] = tuple(data.get(key, ()))
    return SemanticChallenge(**data)


def _ir_ambiguity_adjudication_from_mapping(
    value: Mapping[str, Any],
) -> IRAmbiguityAdjudicationReport:
    data = dict(value)
    rounds: list[IRAmbiguityAdjudicationRound] = []
    for raw_round in data.get("rounds", ()):
        round_data = dict(raw_round)
        round_data["query_ids"] = tuple(round_data.get("query_ids", ()))
        round_data["accepted_query_ids"] = tuple(
            round_data.get("accepted_query_ids", ())
        )
        round_data["unresolved_query_ids"] = tuple(
            round_data.get("unresolved_query_ids", ())
        )
        round_data["failure_reasons"] = tuple(
            round_data.get("failure_reasons", ())
        )
        round_data["candidate_scores"] = tuple(
            IRCandidateSemanticScore(
                **{
                    **dict(item),
                    "reasons": tuple(dict(item).get("reasons", ())),
                }
            )
            for item in round_data.get("candidate_scores", ())
        )
        rounds.append(IRAmbiguityAdjudicationRound(**round_data))
    data["rounds"] = tuple(rounds)
    for key in (
        "supporting_query_ids",
        "unresolved_ambiguity_ids",
        "reasons",
    ):
        data[key] = tuple(data.get(key, ()))
    return IRAmbiguityAdjudicationReport(**data)


def _semantic_parameter_decision_from_mapping(
    value: Any,
) -> SemanticParameterDecision:
    data = dict(value)
    data["parent_decision_ids"] = tuple(data.get("parent_decision_ids", ()))
    return SemanticParameterDecision(**data)


def trace_from_checkpoint(payload: dict) -> LabeledTrace:
    return LabeledTrace(
        name=str(payload["name"]),
        kind=str(payload["kind"]),
        trace=[set(step) for step in payload["trace"]],
        expected=bool(payload["expected"]),
        reason=str(payload.get("reason", "")),
        beginner_view=str(payload.get("beginner_view", "")),
        semantics=trace_semantics_from_mapping(payload.get("semantics")),
        role=str(payload.get("role", "")),
        query_role=str(payload.get("query_role", "")),
        target_ambiguity=str(payload.get("target_ambiguity", "")),
        target_ambiguity_axis_ids=tuple(
            payload.get("target_ambiguity_axis_ids", ())
        ),
        query_constraints=tuple(payload.get("query_constraints", ())),
        adjudicated_label=str(payload.get("adjudicated_label", "")),
        confidence=str(payload.get("confidence", "")),
        rationale=str(payload.get("rationale", "")),
        label_source=str(payload.get("label_source", "human")),
        data_split=str(payload.get("data_split", "test")),
        formula_source=str(payload.get("formula_source", "none")),
        formula_visibility=tuple(payload.get("formula_visibility", ())),
        annotation_locked=bool(payload.get("annotation_locked", False)),
        allowed_rag_stages=tuple(
            payload.get(
                "allowed_rag_stages",
                ("skeleton", "completion", "selection"),
            )
        ),
        trace_id=str(payload.get("trace_id", "")),
        covered_ir_node_ids=tuple(payload.get("covered_ir_node_ids", ())),
        obligation_ids=tuple(payload.get("obligation_ids", ())),
        trace_role=str(payload.get("trace_role", "")),
        boundary_role=str(payload.get("boundary_role", "")),
        generation_source=str(payload.get("generation_source", "")),
        ir_version=str(payload.get("ir_version", "")),
        nl_verdict=str(payload.get("nl_verdict", "")),
        temporal_simulation_summary=str(
            payload.get("temporal_simulation_summary", "")
        ),
        evidence_purpose=str(
            payload.get("evidence_purpose", "semantic_validation")
        ),
        parent_candidate_set_id=str(
            payload.get("parent_candidate_set_id", "")
        ),
        parent_hole_plan_version=str(
            payload.get("parent_hole_plan_version", "")
        ),
        review_record_id=str(payload.get("review_record_id", "")),
        supported_text_spans=tuple(payload.get("supported_text_spans", ())),
    )


def trace_coverage_from_checkpoint(payload: dict | None) -> TraceCoverageReport:
    if not payload:
        return TraceCoverageReport()
    data = dict(payload)
    data["missing"] = tuple(data.get("missing", ()))
    data["coverage_debt"] = tuple(data.get("coverage_debt", ()))
    data["covered_roles"] = tuple(data.get("covered_roles", ()))
    data["missing_roles"] = tuple(data.get("missing_roles", ()))
    data["covered_confusions"] = tuple(data.get("covered_confusions", ()))
    data["missing_confusions"] = tuple(data.get("missing_confusions", ()))
    return TraceCoverageReport(**data)


def degraded_result_from_checkpoint(
    payload: dict | None,
) -> DegradedResultRecord | None:
    if not payload:
        return None
    data = dict(payload)
    data["coverage_debt"] = tuple(data.get("coverage_debt", ()))
    data["all_traces"] = tuple(data.get("all_traces", ()))
    data["skeletons"] = tuple(data.get("skeletons", ()))
    data["candidates"] = tuple(data.get("candidates", ()))
    data["evaluations"] = tuple(data.get("evaluations", ()))
    data["review_records"] = tuple(data.get("review_records", ()))
    data["recommended_actions"] = tuple(data.get("recommended_actions", ()))
    return DegradedResultRecord(**data)


def surface_segmentation_from_checkpoint(payload: Mapping[str, Any]) -> Any:
    """Rebuild and validate a SurfaceSegmentation stored in a payload."""

    value = payload.get("surface_segmentation", payload.get("segmentation"))
    if value is None:
        raise ValueError("checkpoint does not contain surface_segmentation")
    if not isinstance(value, Mapping):
        return value
    return _surface_segmentation_from_mapping(value)


def plain_reading_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    segmentation: Any | None = None,
) -> Any:
    """Rebuild and validate a PlainReadingReview stored in a payload."""

    value = payload.get("plain_reading_review")
    if value is None:
        raise ValueError("checkpoint does not contain plain_reading_review")
    if not isinstance(value, Mapping):
        return value
    return _plain_reading_from_mapping(value, segmentation=segmentation)


def _save_segmentation_checkpoint(
    checkpoint_dir: str | None,
    *,
    subdir: str,
    requirement_id: str,
    segmentation: Any,
    reviewed: bool,
    review_history: list[dict] | tuple[Any, ...] | None,
    generator: Any | None,
    experiment_mode: str | None,
    data_split: str | None,
) -> None:
    if not checkpoint_dir:
        return
    mapping = _artifact_mapping(segmentation)
    if mapping is None:
        raise ValueError("segmentation checkpoint requires an artifact")
    artifact = _surface_segmentation_from_mapping(mapping)
    if str(getattr(artifact, "requirement_id", "")) != requirement_id:
        raise ValueError("segmentation requirement_id does not match checkpoint item")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "kind": "surface_segmentation",
        "checkpoint_stage": subdir,
        "reviewed": reviewed,
        "requirement_id": requirement_id,
        "source_hash": str(getattr(artifact, "source_hash", "")),
        "segmentation_id": str(getattr(artifact, "segmentation_id", "")),
        "segmentation_schema_version": str(
            getattr(artifact, "schema_version", "")
        ),
        "plain_reading_schema_version": "",
        "reading_review_id": "",
        "plain_reading_decision_fingerprint": "",
        "surface_segmentation": mapping,
        "review_history": _json_sequence(
            review_history
            if review_history is not None
            else getattr(artifact, "review_history", ())
        ),
        "generator": _artifact_mapping(
            generator if generator is not None else getattr(artifact, "generator", None)
        )
        or {},
        "experiment_mode": str(
            experiment_mode
            if experiment_mode is not None
            else getattr(artifact, "experiment_mode", "")
        ),
        "data_split": str(
            data_split
            if data_split is not None
            else getattr(artifact, "data_split", "")
        ),
        "updated_at": _timestamp(),
    }
    _write_json(_checkpoint_path(checkpoint_dir, subdir, requirement_id), payload)


def _load_segmentation_checkpoint(
    checkpoint_dir: str | None,
    subdir: str,
    requirement_id: str,
    *,
    expected_source_hash: str | None,
    expected_segmentation_id: str | None,
) -> dict | None:
    payload = _load_checkpoint(checkpoint_dir, subdir, requirement_id)
    if payload is None or not _has_pre_ir_contract(payload):
        return None
    mapping = payload.get("surface_segmentation")
    if not isinstance(mapping, Mapping):
        return None
    try:
        artifact = _surface_segmentation_from_mapping(mapping)
    except (KeyError, TypeError, ValueError):
        return None
    if str(getattr(artifact, "requirement_id", "")) != requirement_id:
        return None
    if payload.get("source_hash") != getattr(artifact, "source_hash", ""):
        return None
    if payload.get("segmentation_id") != getattr(artifact, "segmentation_id", ""):
        return None
    if payload.get("segmentation_schema_version") != getattr(
        artifact, "schema_version", ""
    ):
        return None
    if expected_source_hash is not None and payload.get("source_hash") != expected_source_hash:
        return None
    if (
        expected_segmentation_id is not None
        and payload.get("segmentation_id") != expected_segmentation_id
    ):
        return None
    restored = dict(payload)
    restored["surface_segmentation_mapping"] = dict(mapping)
    restored["surface_segmentation"] = artifact
    restored["segmentation"] = artifact
    return restored


def _plain_reading_grounding_lineage(
    review: PlainReadingReview,
) -> dict[str, str]:
    for item in reversed(review.review_history):
        if (
            isinstance(item, Mapping)
            and item.get("operation")
            == "plain_reading_lexical_contract_bound"
        ):
            return {
                "ap_grounding_fingerprint": str(
                    item.get("ap_grounding_fingerprint", "")
                ),
                "plain_reading_lexical_fingerprint": str(
                    item.get("plain_reading_lexical_fingerprint", "")
                ),
                "ap_grounding_authority": str(
                    item.get("ap_grounding_authority", "")
                ),
            }
    return {}


def _save_plain_reading_checkpoint(
    checkpoint_dir: str | None,
    *,
    subdir: str,
    requirement_id: str,
    plain_reading_review: Any,
    reviewed: bool,
    review_history: list[dict] | tuple[Any, ...] | None,
    generator: Any | None,
    experiment_mode: str | None,
    data_split: str | None,
    segmentation_schema_version: str,
) -> None:
    if not checkpoint_dir:
        return
    mapping = _artifact_mapping(plain_reading_review)
    if mapping is None:
        raise ValueError("plain-reading checkpoint requires an artifact")
    artifact = _plain_reading_from_mapping(mapping, segmentation=None)
    if str(getattr(artifact, "requirement_id", "")) != requirement_id:
        raise ValueError("plain-reading requirement_id does not match checkpoint item")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "kind": "plain_reading_review",
        "checkpoint_stage": subdir,
        "reviewed": reviewed,
        "requirement_id": requirement_id,
        "source_hash": str(getattr(artifact, "source_hash", "")),
        "segmentation_id": str(getattr(artifact, "segmentation_id", "")),
        "segmentation_schema_version": segmentation_schema_version,
        "plain_reading_schema_version": str(
            getattr(artifact, "schema_version", "")
        ),
        "reading_review_id": str(getattr(artifact, "reading_review_id", "")),
        "plain_reading_decision_fingerprint": (
            _plain_reading_decision_fingerprint(artifact)
        ),
        "plain_reading_review": mapping,
        "review_history": _json_sequence(
            review_history
            if review_history is not None
            else getattr(artifact, "review_history", ())
        ),
        "generator": _artifact_mapping(
            generator if generator is not None else getattr(artifact, "generator", None)
        )
        or {},
        "experiment_mode": str(
            experiment_mode
            if experiment_mode is not None
            else getattr(artifact, "experiment_mode", "")
        ),
        "data_split": str(
            data_split
            if data_split is not None
            else getattr(artifact, "data_split", "")
        ),
        "updated_at": _timestamp(),
    }
    _write_json(_checkpoint_path(checkpoint_dir, subdir, requirement_id), payload)


def _load_plain_reading_checkpoint(
    checkpoint_dir: str | None,
    subdir: str,
    requirement_id: str,
    *,
    expected_source_hash: str | None,
    expected_segmentation_id: str | None,
    expected_reading_review_id: str | None,
    expected_plain_reading_decision_fingerprint: str | None,
) -> dict | None:
    payload = _load_checkpoint(checkpoint_dir, subdir, requirement_id)
    if payload is None or not _has_pre_ir_contract(payload):
        return None
    mapping = payload.get("plain_reading_review")
    if not isinstance(mapping, Mapping):
        return None
    try:
        artifact = _plain_reading_from_mapping(mapping, segmentation=None)
        fingerprint = _plain_reading_decision_fingerprint(artifact)
    except (KeyError, TypeError, ValueError):
        return None
    if str(getattr(artifact, "requirement_id", "")) != requirement_id:
        return None
    if payload.get("segmentation_schema_version") != (
        SURFACE_SEGMENTATION_SCHEMA_VERSION
    ):
        return None
    actual_fields = {
        "source_hash": str(getattr(artifact, "source_hash", "")),
        "segmentation_id": str(getattr(artifact, "segmentation_id", "")),
        "reading_review_id": str(getattr(artifact, "reading_review_id", "")),
        "plain_reading_schema_version": str(
            getattr(artifact, "schema_version", "")
        ),
        "plain_reading_decision_fingerprint": fingerprint,
    }
    if any(payload.get(key) != value for key, value in actual_fields.items()):
        return None
    expected_fields = {
        "source_hash": expected_source_hash,
        "segmentation_id": expected_segmentation_id,
        "reading_review_id": expected_reading_review_id,
        "plain_reading_decision_fingerprint": (
            expected_plain_reading_decision_fingerprint
        ),
    }
    if any(
        expected is not None and payload.get(key) != expected
        for key, expected in expected_fields.items()
    ):
        return None
    restored = dict(payload)
    restored["plain_reading_review_mapping"] = dict(mapping)
    restored["plain_reading_review"] = artifact
    return restored


def _save_review_checkpoint(
    checkpoint_dir: str | None,
    *,
    subdir: str,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    review_records: list[dict],
    approved: bool,
    trace_coverage: TraceCoverageReport | None,
    surface_segmentation: Any | None,
    plain_reading_review: Any | None,
    source_hash: str,
    segmentation_id: str,
    reading_review_id: str,
    plain_reading_decision_fingerprint: str,
    segmentation_schema_version: str,
    plain_reading_schema_version: str,
    trace_evidence_records: tuple[TraceEvidenceRecord, ...] | list[TraceEvidenceRecord],
    trace_evidence_disputes: tuple[TraceEvidenceDispute, ...] | list[TraceEvidenceDispute],
) -> None:
    if not checkpoint_dir:
        return
    surface_segmentation = surface_segmentation or getattr(
        semantic, "surface_segmentation", None
    )
    plain_reading_review = plain_reading_review or getattr(
        semantic, "plain_reading_review", None
    )
    surface_mapping = _artifact_mapping(surface_segmentation)
    plain_mapping = _artifact_mapping(plain_reading_review)
    surface_object = (
        _surface_segmentation_from_mapping(surface_mapping)
        if surface_mapping is not None
        else None
    )
    plain_object = (
        _plain_reading_from_mapping(plain_mapping, segmentation=surface_object)
        if plain_mapping is not None
        else None
    )
    metadata = _pre_ir_metadata(
        requirement=requirement,
        surface_segmentation=surface_object,
        plain_reading_review=plain_object,
        source_hash_value=source_hash,
        segmentation_id=segmentation_id,
        reading_review_id=reading_review_id,
        decision_fingerprint=plain_reading_decision_fingerprint,
        segmentation_schema_version=segmentation_schema_version,
        plain_reading_schema_version=plain_reading_schema_version,
    )
    semantic_payload = asdict(semantic)
    semantic_payload["surface_segmentation"] = surface_mapping
    semantic_payload["plain_reading_review"] = plain_mapping
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "obligation_compiler_version": OBLIGATION_COMPILER_VERSION,
        "trace_contract_version": TRACE_CONTRACT_VERSION,
        "semantic_evidence_contract_version": SEMANTIC_EVIDENCE_CONTRACT_VERSION,
        "faithful_ir_runtime_contract": faithful_ir_runtime_contract(),
        "kind": "review_item",
        "approved": approved,
        "requirement_id": requirement_id,
        "requirement": requirement,
        "semantic": semantic_payload,
        "surface_segmentation": surface_mapping,
        "plain_reading_review": plain_mapping,
        "candidate_ir_set": _artifact_mapping(semantic.candidate_ir_set),
        **metadata,
        "ir_version": (
            semantic.faithful_ir.ir_version
            if semantic.faithful_ir is not None
            else ""
        ),
        "ap_context_fingerprint": (
            semantic.faithful_ir.ap_context_fingerprint
            if semantic.faithful_ir is not None
            else ""
        ),
        "trace_obligation_plan": (
            asdict(_checkpoint_trace_plan(semantic.faithful_ir))
            if semantic.faithful_ir is not None
            else None
        ),
        "semantic_readiness": (
            asdict(audit_semantic_readiness(semantic.faithful_ir))
            if semantic.faithful_ir is not None
            else None
        ),
        "traces": [_trace_to_payload(trace) for trace in traces],
        "trace_evidence_records": [
            _trace_evidence_record_to_payload(item)
            for item in trace_evidence_records
        ],
        "trace_evidence_disputes": [
            asdict(item) for item in trace_evidence_disputes
        ],
        "trace_coverage": asdict(trace_coverage or TraceCoverageReport()),
        "review_records": review_records,
        "updated_at": _timestamp(),
    }
    _write_json(_checkpoint_path(checkpoint_dir, subdir, requirement_id), payload)


def _trace_to_payload(trace: LabeledTrace) -> dict:
    return {
        "name": trace.name,
        "kind": trace.kind,
        "trace": trace_to_serializable(trace.trace),
        "expected": trace.expected,
        "reason": trace.reason,
        "beginner_view": trace.beginner_view,
        "semantics": trace_semantics_to_dict(trace.semantics),
        "role": trace.role,
        "query_role": trace.query_role,
        "target_ambiguity": trace.target_ambiguity,
        "target_ambiguity_axis_ids": list(trace.target_ambiguity_axis_ids),
        "query_constraints": list(trace.query_constraints),
        "adjudicated_label": trace.adjudicated_label,
        "confidence": trace.confidence,
        "rationale": trace.rationale,
        "label_source": trace.label_source,
        "data_split": trace.data_split,
        "formula_source": trace.formula_source,
        "formula_visibility": list(trace.formula_visibility),
        "annotation_locked": trace.annotation_locked,
        "allowed_rag_stages": list(trace.allowed_rag_stages),
        "trace_id": trace.trace_id,
        "covered_ir_node_ids": list(trace.covered_ir_node_ids),
        "obligation_ids": list(trace.obligation_ids),
        "trace_role": trace.trace_role,
        "boundary_role": trace.boundary_role,
        "generation_source": trace.generation_source,
        "ir_version": trace.ir_version,
        "nl_verdict": trace.nl_verdict,
        "temporal_simulation_summary": trace.temporal_simulation_summary,
        "review_record_id": trace.review_record_id,
        "supported_text_spans": list(trace.supported_text_spans),
    }


def _trace_evidence_record_to_payload(record: TraceEvidenceRecord) -> dict:
    return {
        "evidence_id": record.evidence_id,
        "semantic_context_id": record.semantic_context_id,
        "trace_fingerprint": record.trace_fingerprint,
        "trace": _trace_to_payload(record.trace),
        "status": record.status,
        "adjudication_brief_id": record.adjudication_brief_id,
        "source_ir_version": record.source_ir_version,
        "superseded_by_context_id": record.superseded_by_context_id,
    }


def trace_evidence_record_from_checkpoint(payload: Mapping[str, Any]) -> TraceEvidenceRecord:
    data = dict(payload)
    return TraceEvidenceRecord(
        evidence_id=str(data["evidence_id"]),
        semantic_context_id=str(data["semantic_context_id"]),
        trace_fingerprint=str(data["trace_fingerprint"]),
        trace=trace_from_checkpoint(dict(data["trace"])),
        status=str(data.get("status", "active")),
        adjudication_brief_id=str(data.get("adjudication_brief_id", "")),
        source_ir_version=str(data.get("source_ir_version", "")),
        superseded_by_context_id=str(
            data.get("superseded_by_context_id", "")
        ),
    )


def trace_evidence_dispute_from_checkpoint(payload: Mapping[str, Any]) -> TraceEvidenceDispute:
    data = dict(payload)
    for key in ("trace_shape", "evidence_ids", "labels", "query_ids"):
        data[key] = tuple(data.get(key, ()))
    data["trace_shape"] = tuple(
        tuple(step) for step in data.get("trace_shape", ())
    )
    return TraceEvidenceDispute(**data)


def _load_checkpoint(
    checkpoint_dir: str | None,
    subdir: str,
    requirement_id: str,
) -> dict | None:
    if not checkpoint_dir:
        return None
    path = _checkpoint_path(checkpoint_dir, subdir, requirement_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION and not (
        subdir in _PRE_IR_PRESERVED_SUBDIRS
        and schema_version in _LEGACY_PRE_IR_CHECKPOINT_SCHEMA_VERSIONS
    ):
        return None
    return payload


def _reject_stale_ir_payload(
    payload: dict | None,
    expected_ir_version: str | None,
) -> dict | None:
    if payload is None or not expected_ir_version:
        return payload
    semantic = payload.get("semantic", {})
    faithful_ir = semantic.get("faithful_ir", {}) if isinstance(semantic, dict) else {}
    actual = faithful_ir.get("ir_version", "") if isinstance(faithful_ir, dict) else ""
    if actual != expected_ir_version:
        return None
    return payload


def _reject_stale_ap_context_payload(
    payload: dict | None,
    expected_ap_context_fingerprint: str | None,
) -> dict | None:
    if payload is None or not expected_ap_context_fingerprint:
        return payload
    semantic = payload.get("semantic", {})
    faithful_ir = (
        semantic.get("faithful_ir", {}) if isinstance(semantic, Mapping) else {}
    )
    actual = (
        str(faithful_ir.get("ap_context_fingerprint", ""))
        if isinstance(faithful_ir, Mapping)
        else ""
    )
    if actual != expected_ap_context_fingerprint:
        return None
    if str(payload.get("ap_context_fingerprint", actual)) != expected_ap_context_fingerprint:
        return None
    return payload


def _reject_stale_evidence_contract_payload(
    payload: dict | None,
) -> dict | None:
    """Reject cached review/selection artifacts built under an old gate contract."""

    if payload is None:
        return None
    if (
        payload.get("obligation_compiler_version")
        != OBLIGATION_COMPILER_VERSION
        or payload.get("trace_contract_version") != TRACE_CONTRACT_VERSION
        or payload.get("semantic_evidence_contract_version")
        != SEMANTIC_EVIDENCE_CONTRACT_VERSION
        or payload.get("faithful_ir_runtime_contract")
        != faithful_ir_runtime_contract()
    ):
        return None
    return payload


def recertify_legacy_semantic_checkpoint(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Explicit migration entrypoint; never called by normal checkpoint loads."""

    semantic = payload.get("semantic")
    if not isinstance(semantic, Mapping):
        return None, {
            "status": "stale_semantic_and_downstream",
            "reason_code": "CHECKPOINT_SEMANTIC_MISSING",
            "semantic_behavior_preserved": False,
        }
    ir_payload = semantic.get("faithful_ir")
    source_text = str(
        payload.get("requirement")
        or (ir_payload or {}).get("source_text", "")
        if isinstance(ir_payload, Mapping)
        else payload.get("requirement", "")
    )
    if not isinstance(ir_payload, Mapping) or not source_text:
        return None, {
            "status": "stale_semantic_and_downstream",
            "reason_code": "CHECKPOINT_FAITHFUL_IR_MISSING",
            "semantic_behavior_preserved": False,
        }
    recertified, audit = recertify_legacy_faithful_ir_mapping(
        ir_payload,
        source_text=source_text,
    )
    if recertified is None:
        return None, audit
    migrated = dict(payload)
    migrated_semantic = dict(semantic)
    migrated_semantic["faithful_ir"] = asdict(recertified)
    migrated["semantic"] = migrated_semantic
    for key in (
        "traces",
        "trace_coverage",
        "trace_obligation_plan",
        "constrained_skeleton_plan",
        "completion_hole_plan",
        "skeletons",
        "completion_decisions",
        "candidates",
        "evaluations",
        "selected_formula",
        "final_approval",
    ):
        migrated.pop(key, None)
    migrated["faithful_ir_runtime_contract"] = faithful_ir_runtime_contract()
    migrated["semantic_migration_audit"] = audit
    migrated["downstream_rederivation_required"] = True
    return migrated, audit


def _migrate_review_evidence_contract_payload(
    payload: dict | None,
) -> dict | None:
    """Load only the immediately previous review contract as quarantined input.

    Legacy trace labels are retained for audit, then rebound by the batch layer
    to the current semantic-context registry before they may become active.
    Pipeline-result checkpoints deliberately continue to use the strict rejector.
    """

    current = _reject_stale_evidence_contract_payload(payload)
    if current is not None:
        return current
    if payload is None:
        return None
    previous_contract = (
        payload.get("trace_contract_version"),
        payload.get("semantic_evidence_contract_version"),
    )
    if (
        payload.get("obligation_compiler_version")
        != OBLIGATION_COMPILER_VERSION
        or previous_contract not in {(11, 5), (8, 4)}
    ):
        return None
    legacy_runtime = payload.get("faithful_ir_runtime_contract")
    if not isinstance(legacy_runtime, Mapping):
        return None
    normalized_runtime = dict(legacy_runtime)
    normalized_runtime["trace_contract_version"] = TRACE_CONTRACT_VERSION
    normalized_runtime[
        "semantic_evidence_contract_version"
    ] = SEMANTIC_EVIDENCE_CONTRACT_VERSION
    if normalized_runtime != faithful_ir_runtime_contract():
        return None
    migrated = dict(payload)
    migrated["trace_contract_version"] = TRACE_CONTRACT_VERSION
    migrated[
        "semantic_evidence_contract_version"
    ] = SEMANTIC_EVIDENCE_CONTRACT_VERSION
    migrated["faithful_ir_runtime_contract"] = faithful_ir_runtime_contract()
    migrated.setdefault("trace_evidence_records", [])
    migrated.setdefault("trace_evidence_disputes", [])
    migrated["trace_evidence_migration"] = (
        "v8-v4_labels_require_semantic_context_reconciliation"
    )
    return migrated


def _artifact_mapping(artifact: Any | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    if is_dataclass(artifact):
        return asdict(artifact)
    if isinstance(artifact, Mapping):
        return dict(artifact)
    raise TypeError("checkpoint artifact must be a dataclass or mapping")


def _surface_segmentation_from_mapping(
    payload: Mapping[str, Any],
) -> SurfaceSegmentation:
    return segmentation_from_mapping(payload)


def _plain_reading_from_mapping(
    payload: Mapping[str, Any],
    *,
    segmentation: SurfaceSegmentation | None,
) -> PlainReadingReview:
    if segmentation is None:
        raise ValueError("plain reading checkpoint is missing its segmentation parent")
    return plain_reading_from_mapping(payload, segmentation)


def _save_segmentation_checkpoint(
    checkpoint_dir: str | None,
    *,
    subdir: str,
    requirement_id: str,
    segmentation: Any,
    reviewed: bool,
    review_history: list[dict] | tuple[Any, ...] | None,
    generator: Any | None,
    experiment_mode: str | None,
    data_split: str | None,
) -> None:
    if not checkpoint_dir:
        return
    artifact = (
        segmentation
        if isinstance(segmentation, SurfaceSegmentation)
        else segmentation_from_mapping(dict(segmentation))
    )
    if reviewed and artifact.status not in {"surface_reviewed", "ambiguity_flagged"}:
        raise ValueError("reviewed segmentation checkpoint requires a reviewed status")
    if artifact.mechanical_validation.status != "passed":
        raise ValueError("mechanically invalid segmentation cannot enter checkpoints")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "kind": "surface_segmentation",
        "reviewed": reviewed,
        "requirement_id": requirement_id,
        "source_hash": artifact.source_hash,
        "segmentation_id": artifact.segmentation_id,
        "segmentation_schema_version": artifact.schema_version,
        "review_history": [
            _artifact_mapping(item) or {} for item in (review_history or artifact.review_history)
        ],
        "generator": _artifact_mapping(generator or artifact.generator),
        "experiment_mode": experiment_mode or artifact.experiment_mode,
        "data_split": data_split or artifact.data_split,
        "surface_segmentation": asdict(artifact),
        "updated_at": _timestamp(),
    }
    _write_json(_checkpoint_path(checkpoint_dir, subdir, requirement_id), payload)


def _load_segmentation_checkpoint(
    checkpoint_dir: str | None,
    subdir: str,
    requirement_id: str,
    *,
    expected_source_hash: str | None,
    expected_segmentation_id: str | None,
) -> dict | None:
    payload = _load_checkpoint(checkpoint_dir, subdir, requirement_id)
    if payload is None:
        return None
    if payload.get("pre_ir_contract_version") != PRE_IR_CHECKPOINT_CONTRACT_VERSION:
        return None
    if payload.get("segmentation_schema_version") != SURFACE_SEGMENTATION_SCHEMA_VERSION:
        return None
    if expected_source_hash and payload.get("source_hash") != expected_source_hash:
        return None
    if expected_segmentation_id and payload.get("segmentation_id") != expected_segmentation_id:
        return None
    mapping = payload.get("surface_segmentation")
    if not isinstance(mapping, Mapping):
        return None
    artifact = segmentation_from_mapping(mapping)
    if artifact.requirement_id != requirement_id:
        return None
    if bool(payload.get("reviewed")) and artifact.status not in {
        "surface_reviewed",
        "ambiguity_flagged",
    }:
        return None
    result = dict(payload)
    result["surface_segmentation_object"] = artifact
    return result


def _save_plain_reading_checkpoint(
    checkpoint_dir: str | None,
    *,
    subdir: str,
    requirement_id: str,
    plain_reading_review: Any,
    reviewed: bool,
    review_history: list[dict] | tuple[Any, ...] | None,
    generator: Any | None,
    experiment_mode: str | None,
    data_split: str | None,
    segmentation_schema_version: str,
) -> None:
    if not checkpoint_dir:
        return
    if not isinstance(plain_reading_review, PlainReadingReview):
        raise TypeError(
            "plain reading checkpoints require a validated PlainReadingReview object"
        )
    review = plain_reading_review
    if reviewed and review.overall_status not in {
        "no_ambiguity",
        "plain_reading_reviewed",
        "ambiguous_unresolved",
    }:
        raise ValueError("reviewed plain reading checkpoint has a non-gate status")
    if (
        reviewed
        and review.overall_status == "no_ambiguity"
        and not review.ambiguities
        and review.generator.mode == "api"
        and review.review_basis == "agent_detection"
        and (
            review.no_ambiguity_confirmation.status
            not in {"confirmed", "withdrawn"}
            or not review.no_ambiguity_confirmation.human_confirmed
        )
    ):
        raise ValueError(
            "LLM no-ambiguity checkpoint requires explicit human confirmation"
        )
    grounding_lineage = _plain_reading_grounding_lineage(review)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pre_ir_contract_version": PRE_IR_CHECKPOINT_CONTRACT_VERSION,
        "kind": "plain_reading_review",
        "reviewed": reviewed,
        "requirement_id": requirement_id,
        "source_hash": review.source_hash,
        "segmentation_id": review.segmentation_id,
        "reading_review_id": review.reading_review_id,
        "plain_reading_decision_fingerprint": decision_fingerprint(review),
        "segmentation_schema_version": (
            segmentation_schema_version or SURFACE_SEGMENTATION_SCHEMA_VERSION
        ),
        "plain_reading_schema_version": review.schema_version,
        "review_history": [dict(item) for item in (review_history or review.review_history)],
        "generator": _artifact_mapping(generator or review.generator),
        "experiment_mode": experiment_mode or review.experiment_mode,
        "data_split": data_split or review.data_split,
        "plain_reading_review": asdict(review),
        **grounding_lineage,
        "updated_at": _timestamp(),
    }
    _write_json(_checkpoint_path(checkpoint_dir, subdir, requirement_id), payload)


def _load_plain_reading_checkpoint(
    checkpoint_dir: str | None,
    subdir: str,
    requirement_id: str,
    *,
    expected_source_hash: str | None,
    expected_segmentation_id: str | None,
    expected_reading_review_id: str | None,
    expected_plain_reading_decision_fingerprint: str | None,
) -> dict | None:
    payload = _load_checkpoint(checkpoint_dir, subdir, requirement_id)
    if payload is None:
        return None
    if payload.get("pre_ir_contract_version") != PRE_IR_CHECKPOINT_CONTRACT_VERSION:
        return None
    if payload.get("segmentation_schema_version") != SURFACE_SEGMENTATION_SCHEMA_VERSION:
        return None
    stored_reading_version = str(payload.get("plain_reading_schema_version", ""))
    if stored_reading_version in {"1", "2", "3", "4"}:
        mapping = payload.get("plain_reading_review")
        if not isinstance(mapping, Mapping):
            return None
        try:
            from plain_reading import decision_fingerprint, plain_reading_from_mapping

            migrated = plain_reading_from_mapping(mapping, None)
        except (TypeError, ValueError):
            return None
        payload = {
            **payload,
            "plain_reading_schema_version": PLAIN_READING_SCHEMA_VERSION,
            "plain_reading_review": asdict(migrated),
            "plain_reading_decision_fingerprint": decision_fingerprint(migrated),
            "migration_status": (
                "read_only_migrated_from_v" + stored_reading_version
            ),
            "read_only_migration": True,
        }
    elif stored_reading_version != PLAIN_READING_SCHEMA_VERSION:
        return None
    mapping = payload.get("plain_reading_review")
    if not isinstance(mapping, Mapping):
        return None
    try:
        from plain_reading import plain_reading_from_mapping

        restored_review = plain_reading_from_mapping(mapping, None)
    except (TypeError, ValueError):
        return None
    grounding_lineage = _plain_reading_grounding_lineage(
        restored_review
    )
    if any(
        key in payload and payload.get(key) != value
        for key, value in grounding_lineage.items()
    ):
        return None
    if bool(payload.get("reviewed")) and restored_review.overall_status not in {
        "no_ambiguity",
        "plain_reading_reviewed",
        "ambiguous_unresolved",
    }:
        return None
    checks = (
        (expected_source_hash, payload.get("source_hash")),
        (expected_segmentation_id, payload.get("segmentation_id")),
        (expected_reading_review_id, payload.get("reading_review_id")),
        (
            expected_plain_reading_decision_fingerprint,
            payload.get("plain_reading_decision_fingerprint"),
        ),
    )
    if any(expected and expected != actual for expected, actual in checks):
        return None
    return payload


def _pre_ir_metadata(
    *,
    requirement: str,
    surface_segmentation: SurfaceSegmentation | None,
    plain_reading_review: PlainReadingReview | None,
    source_hash_value: str,
    segmentation_id: str,
    reading_review_id: str,
    decision_fingerprint: str,
    segmentation_schema_version: str,
    plain_reading_schema_version: str,
) -> dict[str, str]:
    return {
        "source_hash": (
            source_hash_value
            or (surface_segmentation.source_hash if surface_segmentation else source_hash(requirement))
        ),
        "segmentation_id": (
            segmentation_id
            or (surface_segmentation.segmentation_id if surface_segmentation else "")
        ),
        "reading_review_id": (
            reading_review_id
            or (plain_reading_review.reading_review_id if plain_reading_review else "")
        ),
        "plain_reading_decision_fingerprint": (
            decision_fingerprint
            or globals()["decision_fingerprint"](plain_reading_review)
        ),
        "segmentation_schema_version": (
            segmentation_schema_version
            or (surface_segmentation.schema_version if surface_segmentation else "")
        ),
        "plain_reading_schema_version": (
            plain_reading_schema_version
            or (plain_reading_review.schema_version if plain_reading_review else "")
        ),
    }


def _reject_stale_pre_ir_payload(
    payload: dict | None,
    *,
    expected_source_hash: str | None,
    expected_segmentation_id: str | None,
    expected_reading_review_id: str | None,
    expected_plain_reading_decision_fingerprint: str | None,
) -> dict | None:
    if payload is None:
        return None
    # Old checkpoints cannot masquerade as reviewed under the new gate.
    if payload.get("pre_ir_contract_version") != PRE_IR_CHECKPOINT_CONTRACT_VERSION:
        return None
    checks = (
        (expected_source_hash, payload.get("source_hash")),
        (expected_segmentation_id, payload.get("segmentation_id")),
        (expected_reading_review_id, payload.get("reading_review_id")),
        (
            expected_plain_reading_decision_fingerprint,
            payload.get("plain_reading_decision_fingerprint"),
        ),
    )
    if any(expected and expected != actual for expected, actual in checks):
        return None
    return payload


def _validate_semantic_pre_ir_parents(
    semantic: SemanticStructure,
    checkpoint_payload: dict | None,
) -> None:
    segmentation = semantic.surface_segmentation
    reading = semantic.plain_reading_review
    if segmentation is None and reading is None:
        return
    if segmentation is None or reading is None:
        raise ValueError("checkpoint semantic has an incomplete pre-IR parent chain")
    if reading.segmentation_id != segmentation.segmentation_id:
        raise ValueError("checkpoint plain reading references a stale segmentation")
    if semantic.faithful_ir is not None:
        if semantic.faithful_ir.segmentation_id != segmentation.segmentation_id:
            raise ValueError("checkpoint IR references a stale segmentation")
        if semantic.faithful_ir.reading_review_id != reading.reading_review_id:
            raise ValueError("checkpoint IR references a stale plain reading review")
    if checkpoint_payload is not None:
        if checkpoint_payload.get("source_hash") != segmentation.source_hash:
            raise ValueError("checkpoint top-level source_hash is stale")


def _candidate_ir_set_from_mapping(payload: Mapping[str, Any]) -> CandidateIRSet:
    axes_payload = payload.get("ambiguity_axes")
    axis_set = None
    if isinstance(axes_payload, Mapping):
        axis_set = AmbiguityAxisSet(
            requirement_id=str(axes_payload.get("requirement_id", "")),
            axes=tuple(
                AmbiguityAxis(
                    axis_id=str(item.get("axis_id", "")),
                    nl_span=str(item.get("nl_span", "")),
                    possible_readings=tuple(item.get("possible_readings", ())),
                    why_ambiguous=str(item.get("why_ambiguous", "")),
                    required_trace_roles=tuple(item.get("required_trace_roles", ())),
                    review_status=str(item.get("review_status", "pending")),
                    source_segment_ids=tuple(item.get("source_segment_ids", ())),
                    plain_reading_decision=str(item.get("plain_reading_decision", "")),
                    review_basis=str(item.get("review_basis", "")),
                )
                for item in axes_payload.get("axes", ())
                if isinstance(item, Mapping)
            ),
            source=str(axes_payload.get("source", "natural_language")),
        )
    branches: list[CandidateIRBranch] = []
    for item in payload.get("branches", ()):
        if not isinstance(item, Mapping) or not isinstance(item.get("faithful_ir"), Mapping):
            continue
        ir_payload = item["faithful_ir"]
        ir = faithful_ir_from_mapping(
            ir_payload,
            source_text=str(ir_payload.get("source_text", "")),
            requirement_id=str(ir_payload.get("requirement_id", "")),
            extractor="checkpoint",
            trusted_provenance=True,
            working_interpretation=_is_generated_formula_ir(ir_payload),
        )
        branches.append(
            CandidateIRBranch(
                ir_branch_id=str(item.get("ir_branch_id", "")),
                faithful_ir=ir,
                supported_text_spans=tuple(item.get("supported_text_spans", ())),
                ambiguity_axis_ids=tuple(item.get("ambiguity_axis_ids", ())),
                semantic_assumptions=tuple(item.get("semantic_assumptions", ())),
                plain_language_paraphrase=str(item.get("plain_language_paraphrase", "")),
                generation_reason=str(item.get("generation_reason", "")),
                applied_alternative_ids=tuple(item.get("applied_alternative_ids", ())),
                status=str(item.get("status", "proposed")),
                policy_profile_id=str(item.get("policy_profile_id", "")),
                policy_profile_version=str(item.get("policy_profile_version", "")),
                controlled_parameter_paths=tuple(
                    item.get("controlled_parameter_paths", ())
                ),
                structure_fingerprint=str(item.get("structure_fingerprint", "")),
                policy_target_binding_id=str(
                    item.get("policy_target_binding_id", "")
                ),
            )
        )
    return CandidateIRSet(
        requirement_id=str(payload.get("requirement_id", "")),
        nl_text=str(payload.get("nl_text", "")),
        branches=tuple(branches),
        ambiguity_axes=axis_set,
        max_branch_count=int(payload.get("max_branch_count", 5)),
        status=str(payload.get("status", "proposed")),
        segmentation_id=str(payload.get("segmentation_id", "")),
        reading_review_id=str(payload.get("reading_review_id", "")),
        ambiguity_debt=tuple(payload.get("ambiguity_debt", ())),
        branch_set_id=str(payload.get("branch_set_id", "")),
        common_structure_fingerprint=str(
            payload.get("common_structure_fingerprint", "")
        ),
        policy_axis_ids=tuple(payload.get("policy_axis_ids", ())),
        selection_status=str(payload.get("selection_status", "unresolved")),
        selection_basis=str(payload.get("selection_basis", "")),
        selected_branch_id=str(payload.get("selected_branch_id", "")),
        candidate_catalog_version=str(payload.get("candidate_catalog_version", "")),
        policy_target_binding_ids=tuple(
            payload.get("policy_target_binding_ids", ())
        ),
    )


def _reject_stale_pre_ir_payload(
    payload: dict | None,
    *,
    expected_source_hash: str | None,
    expected_segmentation_id: str | None,
    expected_reading_review_id: str | None,
    expected_plain_reading_decision_fingerprint: str | None,
) -> dict | None:
    if payload is None or not _has_pre_ir_contract(payload):
        return None
    required_fields = {
        "source_hash",
        "segmentation_id",
        "reading_review_id",
        "plain_reading_decision_fingerprint",
        "segmentation_schema_version",
        "plain_reading_schema_version",
    }
    if not required_fields.issubset(payload):
        return None
    actual_source_hash = str(payload.get("source_hash", ""))
    if not actual_source_hash:
        return None
    requirement = payload.get("requirement")
    if isinstance(requirement, str) and requirement:
        if actual_source_hash != _source_hash(requirement):
            return None
    expected_fields = {
        "source_hash": expected_source_hash,
        "segmentation_id": expected_segmentation_id,
        "reading_review_id": expected_reading_review_id,
        "plain_reading_decision_fingerprint": (
            expected_plain_reading_decision_fingerprint
        ),
    }
    if any(
        expected is not None and str(payload.get(field, "")) != expected
        for field, expected in expected_fields.items()
    ):
        return None
    if payload.get("segmentation_id") and not payload.get(
        "segmentation_schema_version"
    ):
        return None
    if payload.get("reading_review_id") and (
        not payload.get("plain_reading_schema_version")
        or not payload.get("segmentation_id")
        or not payload.get("plain_reading_decision_fingerprint")
    ):
        return None

    surface_mapping = payload.get("surface_segmentation")
    plain_mapping = payload.get("plain_reading_review")
    try:
        surface = (
            _surface_segmentation_from_mapping(surface_mapping)
            if isinstance(surface_mapping, Mapping)
            else None
        )
        plain = (
            _plain_reading_from_mapping(plain_mapping, segmentation=surface)
            if isinstance(plain_mapping, Mapping)
            else None
        )
    except (KeyError, TypeError, ValueError):
        return None
    if surface is not None:
        if any(
            (
                str(payload.get("source_hash", ""))
                != str(getattr(surface, "source_hash", "")),
                str(payload.get("segmentation_id", ""))
                != str(getattr(surface, "segmentation_id", "")),
                str(payload.get("segmentation_schema_version", ""))
                != str(getattr(surface, "schema_version", "")),
            )
        ):
            return None
    if plain is not None:
        if any(
            (
                str(payload.get("source_hash", ""))
                != str(getattr(plain, "source_hash", "")),
                str(payload.get("segmentation_id", ""))
                != str(getattr(plain, "segmentation_id", "")),
                str(payload.get("reading_review_id", ""))
                != str(getattr(plain, "reading_review_id", "")),
                str(payload.get("plain_reading_schema_version", ""))
                != str(getattr(plain, "schema_version", "")),
                str(payload.get("plain_reading_decision_fingerprint", ""))
                != _plain_reading_decision_fingerprint(plain),
            )
        ):
            return None
    return payload


def _has_pre_ir_contract(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
        and payload.get("pre_ir_contract_version")
        == PRE_IR_CHECKPOINT_CONTRACT_VERSION
    )


def _pre_ir_metadata(
    *,
    requirement: str,
    surface_segmentation: Any | None,
    plain_reading_review: Any | None,
    source_hash_value: str,
    segmentation_id: str,
    reading_review_id: str,
    decision_fingerprint: str,
    segmentation_schema_version: str,
    plain_reading_schema_version: str,
) -> dict[str, str]:
    expected_source_hash = _source_hash(requirement)
    surface_source_hash = str(
        getattr(surface_segmentation, "source_hash", "")
        if surface_segmentation is not None
        else ""
    )
    plain_source_hash = str(
        getattr(plain_reading_review, "source_hash", "")
        if plain_reading_review is not None
        else ""
    )
    resolved_source_hash = _coalesce_parent_value(
        "source_hash",
        source_hash_value,
        surface_source_hash,
        plain_source_hash,
        expected_source_hash,
    )
    if resolved_source_hash != expected_source_hash:
        raise ValueError("checkpoint source_hash does not match exact requirement text")

    surface_segmentation_id = str(
        getattr(surface_segmentation, "segmentation_id", "")
        if surface_segmentation is not None
        else ""
    )
    plain_segmentation_id = str(
        getattr(plain_reading_review, "segmentation_id", "")
        if plain_reading_review is not None
        else ""
    )
    resolved_segmentation_id = _coalesce_parent_value(
        "segmentation_id",
        segmentation_id,
        surface_segmentation_id,
        plain_segmentation_id,
    )
    resolved_reading_review_id = _coalesce_parent_value(
        "reading_review_id",
        reading_review_id,
        str(
            getattr(plain_reading_review, "reading_review_id", "")
            if plain_reading_review is not None
            else ""
        ),
    )
    resolved_decision_fingerprint = _coalesce_parent_value(
        "plain_reading_decision_fingerprint",
        decision_fingerprint,
        (
            _plain_reading_decision_fingerprint(plain_reading_review)
            if plain_reading_review is not None
            else ""
        ),
    )
    resolved_segmentation_schema = _coalesce_parent_value(
        "segmentation_schema_version",
        segmentation_schema_version,
        str(
            getattr(surface_segmentation, "schema_version", "")
            if surface_segmentation is not None
            else ""
        ),
        (
            SURFACE_SEGMENTATION_SCHEMA_VERSION
            if plain_reading_review is not None
            else ""
        ),
        (
            SURFACE_SEGMENTATION_SCHEMA_VERSION
            if resolved_segmentation_id
            else ""
        ),
    )
    resolved_plain_schema = _coalesce_parent_value(
        "plain_reading_schema_version",
        plain_reading_schema_version,
        str(
            getattr(plain_reading_review, "schema_version", "")
            if plain_reading_review is not None
            else ""
        ),
        PLAIN_READING_SCHEMA_VERSION if resolved_reading_review_id else "",
    )
    if resolved_reading_review_id and not (
        resolved_segmentation_id
        and resolved_decision_fingerprint
        and resolved_plain_schema
    ):
        raise ValueError("plain-reading parent metadata is incomplete")
    return {
        "source_hash": resolved_source_hash,
        "segmentation_id": resolved_segmentation_id,
        "reading_review_id": resolved_reading_review_id,
        "plain_reading_decision_fingerprint": resolved_decision_fingerprint,
        "segmentation_schema_version": resolved_segmentation_schema,
        "plain_reading_schema_version": resolved_plain_schema,
    }


def _coalesce_parent_value(name: str, *values: object) -> str:
    present = [str(value) for value in values if value is not None and str(value)]
    if len(set(present)) > 1:
        raise ValueError(f"conflicting checkpoint {name} values")
    return present[0] if present else ""


def _artifact_mapping(value: Any | None) -> dict | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"checkpoint artifact must be a dataclass or mapping, got {type(value).__name__}")


def _json_sequence(values: object) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return [str(values)]
    result: list[Any] = []
    for value in values:  # type: ignore[union-attr]
        if is_dataclass(value):
            result.append(asdict(value))
        elif isinstance(value, Mapping):
            result.append(dict(value))
        else:
            result.append(value)
    return result


def _surface_segmentation_from_mapping(payload: Mapping[str, Any]) -> Any:
    from surface_review import segmentation_from_mapping, validate_surface_segmentation

    artifact = segmentation_from_mapping(dict(payload))
    _assert_validation_passed(
        validate_surface_segmentation(artifact),
        "surface segmentation",
    )
    return artifact


def _plain_reading_from_mapping(
    payload: Mapping[str, Any],
    *,
    segmentation: Any | None,
) -> Any:
    from surface_review import plain_reading_from_mapping, validate_plain_reading_review

    artifact = plain_reading_from_mapping(dict(payload))
    # The validator's first-iteration public contract accepts the review and
    # may optionally accept its parent segmentation.  Use signature names to
    # avoid masking a TypeError raised by validation itself.
    import inspect

    parameters = inspect.signature(validate_plain_reading_review).parameters
    if segmentation is not None and "segmentation" in parameters:
        result = validate_plain_reading_review(artifact, segmentation=segmentation)
    elif segmentation is not None and "surface_segmentation" in parameters:
        result = validate_plain_reading_review(
            artifact,
            surface_segmentation=segmentation,
        )
    elif segmentation is not None and len(parameters) >= 2:
        result = validate_plain_reading_review(artifact, segmentation)
    else:
        result = validate_plain_reading_review(artifact)
    _assert_validation_passed(result, "plain reading review")
    return artifact


def _assert_validation_passed(result: Any, label: str) -> None:
    if result is False:
        raise ValueError(f"invalid {label}")
    if isinstance(result, (list, tuple)) and result:
        raise ValueError(f"invalid {label}: " + "; ".join(map(str, result)))


def _source_hash(source_text: str) -> str:
    try:
        from surface_review import source_hash
    except ImportError:  # pragma: no cover - only supports staged module rollout
        return hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    return source_hash(source_text)


def _plain_reading_decision_fingerprint(review: Any) -> str:
    from surface_review import plain_reading_decision_fingerprint

    return plain_reading_decision_fingerprint(review)


def _candidate_ir_set_from_mapping(payload: Mapping[str, Any]) -> CandidateIRSet:
    axis_payload = payload.get("ambiguity_axes")
    axis_set = None
    if isinstance(axis_payload, Mapping):
        axes = tuple(
            AmbiguityAxis(
                axis_id=str(item.get("axis_id", "")),
                nl_span=str(item.get("nl_span", "")),
                possible_readings=tuple(item.get("possible_readings", ())),
                why_ambiguous=str(item.get("why_ambiguous", "")),
                required_trace_roles=tuple(item.get("required_trace_roles", ())),
                review_status=str(item.get("review_status", "pending")),
                source_segment_ids=tuple(item.get("source_segment_ids", ())),
                plain_reading_decision=str(
                    item.get("plain_reading_decision", "")
                ),
                review_basis=str(item.get("review_basis", "")),
            )
            for item in axis_payload.get("axes", ())
            if isinstance(item, Mapping)
        )
        axis_set = AmbiguityAxisSet(
            requirement_id=str(axis_payload.get("requirement_id", "")),
            axes=axes,
            source=str(axis_payload.get("source", "natural_language")),
        )
    branches: list[CandidateIRBranch] = []
    for item in payload.get("branches", ()):
        if not isinstance(item, Mapping):
            continue
        ir_payload = item.get("faithful_ir")
        if not isinstance(ir_payload, Mapping):
            raise ValueError("candidate IR branch is missing faithful_ir")
        faithful_ir = faithful_ir_from_mapping(
            ir_payload,
            source_text=str(ir_payload.get("source_text", "")),
            requirement_id=str(ir_payload.get("requirement_id", "")),
            extractor="checkpoint",
            trusted_provenance=True,
            working_interpretation=_is_generated_formula_ir(ir_payload),
        )
        branches.append(
            CandidateIRBranch(
                ir_branch_id=str(item.get("ir_branch_id", "")),
                faithful_ir=faithful_ir,
                supported_text_spans=tuple(item.get("supported_text_spans", ())),
                ambiguity_axis_ids=tuple(item.get("ambiguity_axis_ids", ())),
                semantic_assumptions=tuple(item.get("semantic_assumptions", ())),
                plain_language_paraphrase=str(
                    item.get("plain_language_paraphrase", "")
                ),
                generation_reason=str(item.get("generation_reason", "")),
                applied_alternative_ids=tuple(item.get("applied_alternative_ids", ())),
                status=str(item.get("status", "proposed")),
                policy_profile_id=str(item.get("policy_profile_id", "")),
                policy_profile_version=str(item.get("policy_profile_version", "")),
                controlled_parameter_paths=tuple(
                    item.get("controlled_parameter_paths", ())
                ),
                structure_fingerprint=str(item.get("structure_fingerprint", "")),
                policy_target_binding_id=str(
                    item.get("policy_target_binding_id", "")
                ),
            )
        )
    return CandidateIRSet(
        requirement_id=str(payload.get("requirement_id", "")),
        nl_text=str(payload.get("nl_text", "")),
        branches=tuple(branches),
        ambiguity_axes=axis_set,
        max_branch_count=int(payload.get("max_branch_count", 5)),
        status=str(payload.get("status", "proposed")),
        segmentation_id=str(payload.get("segmentation_id", "")),
        reading_review_id=str(payload.get("reading_review_id", "")),
        ambiguity_debt=tuple(payload.get("ambiguity_debt", ())),
        branch_set_id=str(payload.get("branch_set_id", "")),
        common_structure_fingerprint=str(
            payload.get("common_structure_fingerprint", "")
        ),
        policy_axis_ids=tuple(payload.get("policy_axis_ids", ())),
        selection_status=str(payload.get("selection_status", "unresolved")),
        selection_basis=str(payload.get("selection_basis", "")),
        selected_branch_id=str(payload.get("selected_branch_id", "")),
        candidate_catalog_version=str(payload.get("candidate_catalog_version", "")),
        policy_target_binding_ids=tuple(
            payload.get("policy_target_binding_ids", ())
        ),
    )


def _validate_semantic_pre_ir_parents(
    semantic: SemanticStructure,
    checkpoint_payload: Mapping[str, Any] | None,
) -> None:
    surface = getattr(semantic, "surface_segmentation", None)
    plain = getattr(semantic, "plain_reading_review", None)
    faithful_ir = semantic.faithful_ir
    candidate_set = semantic.candidate_ir_set
    segmentation_id = str(getattr(surface, "segmentation_id", ""))
    reading_review_id = str(getattr(plain, "reading_review_id", ""))
    if plain is not None:
        if not segmentation_id:
            raise ValueError("plain reading checkpoint is missing surface segmentation")
        if str(getattr(plain, "segmentation_id", "")) != segmentation_id:
            raise ValueError("plain reading references a different segmentation")
        if str(getattr(plain, "source_hash", "")) != str(
            getattr(surface, "source_hash", "")
        ):
            raise ValueError("plain reading source_hash does not match segmentation")
    for artifact_name, artifact in (
        ("faithful_ir", faithful_ir),
        ("candidate_ir_set", candidate_set),
    ):
        if artifact is None:
            continue
        artifact_segmentation_id = str(getattr(artifact, "segmentation_id", ""))
        artifact_reading_id = str(getattr(artifact, "reading_review_id", ""))
        if artifact_segmentation_id and artifact_segmentation_id != segmentation_id:
            raise ValueError(f"{artifact_name} segmentation parent is stale")
        if artifact_reading_id and artifact_reading_id != reading_review_id:
            raise ValueError(f"{artifact_name} plain-reading parent is stale")
    if checkpoint_payload is not None:
        validation_payload = dict(checkpoint_payload)
        # Direct semantic migration may inspect the immediately preceding
        # checkpoint schema.  Promote only the envelope for parent-lineage
        # validation; normal checkpoint loaders still invalidate stale derived
        # plans and traces before resume.
        try:
            previous_schema = int(validation_payload.get("schema_version", -1))
        except (TypeError, ValueError):
            previous_schema = -1
        if (
            previous_schema == CHECKPOINT_SCHEMA_VERSION - 1
            and validation_payload.get("pre_ir_contract_version")
            == PRE_IR_CHECKPOINT_CONTRACT_VERSION
        ):
            validation_payload["schema_version"] = CHECKPOINT_SCHEMA_VERSION
        validated = _reject_stale_pre_ir_payload(
            validation_payload,
            expected_source_hash=(
                str(getattr(surface, "source_hash", "")) if surface is not None else None
            ),
            expected_segmentation_id=segmentation_id or None,
            expected_reading_review_id=reading_review_id or None,
            expected_plain_reading_decision_fingerprint=(
                _plain_reading_decision_fingerprint(plain)
                if plain is not None
                else None
            ),
        )
        if validated is None:
            raise ValueError("semantic checkpoint pre-IR parent fingerprint is stale")


def _checkpoint_path(checkpoint_dir: str, subdir: str, requirement_id: str) -> Path:
    safe_id = requirement_id.replace("/", "_").replace("\\", "_")
    return Path(checkpoint_dir) / subdir / f"{safe_id}.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _read_json_mapping(path: Path, *, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{code}: unreadable artifact {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{code}: artifact is not an object {path}")
    return payload


def _write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    conflict_code: str,
) -> None:
    """Create a content-addressed artifact without overwriting history."""

    desired = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _read_json_mapping(path, code=conflict_code)
        if _canonical_json(existing) != _canonical_json(payload):
            raise ValueError(f"{conflict_code}: immutable artifact changed {path}")
        return
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(desired)
    except FileExistsError:
        existing = _read_json_mapping(path, code=conflict_code)
        if _canonical_json(existing) != _canonical_json(payload):
            raise ValueError(f"{conflict_code}: immutable artifact changed {path}")


def _fingerprint(manifest: dict[str, Any]) -> str:
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
