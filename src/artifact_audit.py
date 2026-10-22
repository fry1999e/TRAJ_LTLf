"""Stable artifact fingerprints and rerun delta reports for runtime.

This module deliberately depends only on the Python standard library.  It can
therefore be imported by schemas, checkpointing, the pipeline, or batch
orchestration without creating a project-level import cycle.  All public
builders accept ``Any`` and work with dataclasses, mappings, or ordinary
objects exposing the expected attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from enum import Enum
import hashlib
import json
import math
from pathlib import Path, PurePath
from typing import Any, Iterable, Mapping, Sequence


CANONICAL_HASH_VERSION = "artifact-canonical-v1"

ARTIFACT_NAMES = (
    "plain_reading",
    "semantic_ir",
    "approved_ir",
    "trace_plan",
    "trace_set",
    "skeleton_plan",
    "hole_plan",
    "candidate_set",
    "evaluation",
    "approval",
)

RERUN_DELTA_STATUSES = frozenset(
    {
        "effective_delta",
        "no_effective_delta",
        "regression",
        "blocked_by_frozen_upstream",
        "needs_independent_approval",
    }
)

_ARTIFACT_ALIASES = {
    "plain_reading_review": "plain_reading",
    "faithful_ir": "semantic_ir",
    "faithful_semantic_ir": "semantic_ir",
    "approved_semantic_ir": "approved_ir",
    "trace_obligation_plan": "trace_plan",
    "traces": "trace_set",
    "constrained_skeleton_plan": "skeleton_plan",
    "completion_hole_plan": "hole_plan",
    "candidates": "candidate_set",
    "evaluations": "evaluation",
    "final_approval": "approval",
}

_VOLATILE_KEYS = frozenset(
    {
        "approved_at",
        "completed_at",
        "created_at",
        "finished_at",
        "generated_at",
        "started_at",
        "timestamp",
        "updated_at",
        "elapsed_seconds",
        "duration_seconds",
        "wall_time_seconds",
        "record_path",
        "record_paths",
        "review_path",
        "manual_review_path",
        "annotation_review_path",
        "checkpoint_dir",
        "classified_output_dir",
        "output_dir",
        "records_dir",
        "runtime_rag_chunks",
        "report_text",
        "summary_text",
    }
)

_FORMULA_FIELD_NAMES = frozenset(
    {
        "formula",
        "formula_text",
        "approved_formula",
        "candidate_formula",
        "final_formula",
        "selected_formula",
    }
)

_FORMULA_OPERATORS = frozenset(
    {
        "ap",
        "true",
        "false",
        "not",
        "and",
        "or",
        "implies",
        "X",
        "F",
        "G",
        "U",
    }
)

_ERROR_IDENTITY_KEYS = frozenset(
    {
        "diagnosis_id",
        "error_id",
        "rerun_id",
        "source_run_id",
    }
)

_UNSET = object()


@dataclass(frozen=True)
class ArtifactFingerprintSet:
    """Content fingerprints for every artifact named by runtime."""

    plain_reading_fingerprint: str = ""
    semantic_ir_fingerprint: str = ""
    approved_ir_fingerprint: str = ""
    trace_plan_fingerprint: str = ""
    trace_set_fingerprint: str = ""
    skeleton_plan_fingerprint: str = ""
    hole_plan_fingerprint: str = ""
    candidate_set_fingerprint: str = ""
    evaluation_fingerprint: str = ""
    approval_fingerprint: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            f"{name}_fingerprint": getattr(self, f"{name}_fingerprint")
            for name in ARTIFACT_NAMES
        }

    def fingerprint_for(self, artifact_name: str) -> str:
        normalized = normalize_artifact_name(artifact_name)
        return getattr(self, f"{normalized}_fingerprint")

    @property
    def present_artifacts(self) -> tuple[str, ...]:
        return tuple(name for name in ARTIFACT_NAMES if self.fingerprint_for(name))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactFingerprintSet":
        values: dict[str, str] = {}
        for name in ARTIFACT_NAMES:
            field_name = f"{name}_fingerprint"
            values[field_name] = str(payload.get(field_name, "") or "")
        return cls(**values)


@dataclass(frozen=True)
class RerunDeltaReport:
    """Auditable comparison between the source and rerun artifact states."""

    rerun_id: str
    diagnosis_id: str = ""
    start_stage: str = ""
    expected_changed_artifacts: tuple[str, ...] = ()
    actual_changed_artifacts: tuple[str, ...] = ()
    unchanged_artifacts: tuple[str, ...] = ()
    new_candidates: tuple[str, ...] = ()
    removed_candidates: tuple[str, ...] = ()
    new_traces: tuple[str, ...] = ()
    resolved_errors: tuple[str, ...] = ()
    remaining_errors: tuple[str, ...] = ()
    status: str = "no_effective_delta"
    before_fingerprints: ArtifactFingerprintSet = field(
        default_factory=ArtifactFingerprintSet
    )
    after_fingerprints: ArtifactFingerprintSet = field(
        default_factory=ArtifactFingerprintSet
    )
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in RERUN_DELTA_STATUSES:
            raise ValueError(f"unsupported rerun delta status: {self.status}")
        for attribute in (
            "expected_changed_artifacts",
            "actual_changed_artifacts",
            "unchanged_artifacts",
            "new_candidates",
            "removed_candidates",
            "new_traces",
            "resolved_errors",
            "remaining_errors",
            "reasons",
        ):
            value = tuple(str(item) for item in getattr(self, attribute))
            object.__setattr__(self, attribute, value)

    @property
    def missing_expected_artifacts(self) -> tuple[str, ...]:
        actual = set(self.actual_changed_artifacts)
        return tuple(
            name for name in self.expected_changed_artifacts if name not in actual
        )

    @property
    def unexpected_changed_artifacts(self) -> tuple[str, ...]:
        if not self.expected_changed_artifacts:
            return ()
        expected = set(self.expected_changed_artifacts)
        return tuple(
            name for name in self.actual_changed_artifacts if name not in expected
        )


@dataclass(frozen=True)
class APIdentityAuditReport:
    """Read-only runtime audit of AP inventory/comparison/override lineage."""

    checkpoint_dir: str
    items: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    corrupt_or_stale: tuple[Mapping[str, str], ...] = ()
    formula_blind_exclusions: tuple[Mapping[str, str], ...] = ()
    generated_at: str = ""

    @property
    def passed(self) -> bool:
        return not self.corrupt_or_stale

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_contract_version": "ap-identity-audit-v1",
            "checkpoint_dir": self.checkpoint_dir,
            "passed": self.passed,
            "items": [dict(item) for item in self.items],
            "metrics": dict(self.metrics),
            "corrupt_or_stale": [dict(item) for item in self.corrupt_or_stale],
            "formula_blind_exclusions": [
                dict(item) for item in self.formula_blind_exclusions
            ],
            "generated_at": self.generated_at,
        }


def audit_ap_identity_artifacts(
    checkpoint_dir: str | Path,
) -> APIdentityAuditReport:
    """Validate every runtime artifact without mutating checkpoints.

    Imports are intentionally local: ``artifact_audit`` is a dependency-light
    module used by schemas and checkpointing, so top-level runtime imports would
    create a cycle.
    """

    from ap_identity import (
        override_ap_comparison_from_mapping,
        validate_ap_inventory_integrity,
        validate_override_ap_comparison_integrity,
    )
    from checkpointing import (
        list_human_semantic_override_history,
        load_ap_inventory_checkpoint,
        load_human_semantic_override_ap_artifacts,
        load_human_semantic_override_checkpoint,
    )
    from contract_versions import (
        HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION,
        HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION,
    )

    root = Path(checkpoint_dir)
    requirement_ids: set[str] = set()
    for subdir in (
        "ap_inventories",
        "override_ap_comparisons",
        "semantic_override_history",
    ):
        directory = root / subdir
        if directory.is_dir():
            requirement_ids.update(path.name for path in directory.iterdir() if path.is_dir())
    for subdir in ("pending_semantic_override", "human_semantic_overrides"):
        directory = root / subdir
        if directory.is_dir():
            requirement_ids.update(path.stem for path in directory.glob("*.json"))

    entries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    modes: dict[str, int] = {}
    v1_load_count = 0
    inventory_required = 0
    mapping_required = 0
    comparison_passed = 0
    known_mismatch = 0
    unknown = 0
    formula_assisted = 0
    seen_events: set[tuple[str, str, str]] = set()
    seen_inventory_metrics: set[tuple[str, str]] = set()
    approved_inventories: set[tuple[str, str]] = set()
    manually_approved_inventories: set[tuple[str, str]] = set()

    for requirement_id in sorted(requirement_ids):
        try:
            events = list(
                list_human_semantic_override_history(str(root), requirement_id)
            )
        except Exception as exc:
            failures.append(
                {
                    "requirement_id": requirement_id,
                    "artifact_id": "semantic_override_history",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            events = []
        if not events:
            for active_only, pending_only in ((True, False), (False, True)):
                try:
                    current = load_human_semantic_override_checkpoint(
                        str(root),
                        requirement_id,
                        active_only=active_only,
                        pending_only=pending_only,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "requirement_id": requirement_id,
                            "artifact_id": (
                                "active_override" if active_only else "pending_override"
                            ),
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    current = None
                if current is not None:
                    events.append(current)
        for override in events:
            event_key = (override.override_id, override.status, override.confirmed_at)
            if event_key in seen_events:
                continue
            seen_events.add(event_key)
            contract = (
                override.override_contract_version
                or HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION
            )
            entry: dict[str, Any] = {
                "requirement_id": requirement_id,
                "override_id": override.override_id,
                "override_contract_version": contract,
                "status": override.status,
                "ap_validation_status": override.ap_validation_status,
                "ap_validation_mode": override.ap_validation_mode,
                "representation_profile_id": override.representation_profile_id,
                "issue_codes": list(override.issue_codes),
                "original_inventory_id": override.original_ap_inventory_id,
                "original_inventory_hash": override.original_ap_inventory_hash,
                "replacement_inventory_id": override.replacement_ap_inventory_id,
                "replacement_inventory_hash": override.replacement_ap_inventory_hash,
                "comparison_id": override.ap_comparison_id,
                "comparison_hash": override.ap_comparison_hash,
                "supersedes_override_id": override.supersedes_override_id,
                "source_lineage": {
                    "original_source_hash": override.original_source_hash,
                    "parent_effective_source_hash": override.parent_effective_source_hash,
                    "replacement_hash": override.replacement_hash,
                },
                "audit_status": "passed",
            }
            try:
                if contract == HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION:
                    v1_load_count += 1
                    entry["compatibility_mode"] = "frozen_v1_symbolic_token"
                elif contract == HUMAN_SEMANTIC_OVERRIDE_CONTRACT_VERSION:
                    original, replacement, comparison = (
                        load_human_semantic_override_ap_artifacts(str(root), override)
                    )
                    inventories = tuple(
                        item for item in (original, replacement) if item is not None
                    )
                    for inventory in inventories:
                        inventory_key = (requirement_id, inventory.inventory_id)
                        if inventory_key not in seen_inventory_metrics:
                            seen_inventory_metrics.add(inventory_key)
                            mode = inventory.extraction_mode
                            modes[mode] = modes.get(mode, 0) + 1
                        if inventory.reviewer_role == "ordinary_reviewer":
                            manually_approved_inventories.add(inventory_key)
                        if inventory.status == "approved":
                            approved_inventories.add(inventory_key)
                        if (
                            not inventory.eligible_for_formula_blind_evaluation
                            and not any(
                                item.get("inventory_id") == inventory.inventory_id
                                for item in exclusions
                            )
                        ):
                            formula_assisted += 1
                            exclusion = {
                                "requirement_id": requirement_id,
                                "override_id": override.override_id,
                                "inventory_id": inventory.inventory_id,
                                "reason": "formula_assisted_not_formula_blind",
                            }
                            exclusions.append(exclusion)
                    entry["inventory_modes"] = [
                        inventory.extraction_mode for inventory in inventories
                    ]
                    entry["identity_mapping"] = (
                        [
                            {
                                "replacement_ap_identity_id": item.replacement_ap_identity_id,
                                "original_ap_identity_id": item.original_ap_identity_id,
                                "mapping_provenance": item.mapping_provenance,
                            }
                            for item in comparison.identity_mapping
                        ]
                        if comparison is not None
                        else []
                    )
                else:
                    raise ValueError("unsupported semantic override contract")
            except Exception as exc:
                entry["audit_status"] = "corrupt_or_stale"
                failure = {
                    "requirement_id": requirement_id,
                    "artifact_id": override.override_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)

            if override.status == "awaiting_ap_inventory":
                inventory_required += 1
            if override.status == "awaiting_ap_mapping_confirmation":
                mapping_required += 1
            if override.ap_validation_status == "passed":
                comparison_passed += 1
            if override.status == "ap_rejected":
                known_mismatch += 1
            if override.ap_validation_status == "needs_review":
                unknown += 1
            entries.append(entry)

    # Audit orphan inventories and comparisons as well as override-referenced ones.
    inventories_by_id: dict[tuple[str, str], Any] = {}
    inventory_root = root / "ap_inventories"
    if inventory_root.is_dir():
        for path in sorted(inventory_root.glob("*/*.json")):
            requirement_id = path.parent.name
            inventory_id = path.stem
            try:
                inventory = load_ap_inventory_checkpoint(
                    str(root),
                    requirement_id,
                    inventory_id=inventory_id,
                    allow_unapproved=True,
                )
                if inventory is None:
                    raise ValueError("inventory file is not loadable")
                inventories_by_id[(requirement_id, inventory_id)] = inventory
            except Exception as exc:
                failures.append(
                    {
                        "requirement_id": requirement_id,
                        "artifact_id": inventory_id,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    comparison_root = root / "override_ap_comparisons"
    if comparison_root.is_dir():
        for path in sorted(comparison_root.glob("*/*.json")):
            requirement_id = path.parent.name
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                mapping = envelope.get("override_ap_comparison")
                if not isinstance(mapping, Mapping):
                    raise ValueError("missing comparison payload")
                comparison = override_ap_comparison_from_mapping(mapping)
                original = inventories_by_id.get(
                    (requirement_id, comparison.original_inventory_id)
                )
                replacement = inventories_by_id.get(
                    (requirement_id, comparison.replacement_inventory_id)
                )
                if original is None or replacement is None:
                    raise ValueError("comparison has missing inventory input")
                validate_ap_inventory_integrity(
                    original, allow_unapproved=original.status != "approved"
                )
                validate_ap_inventory_integrity(
                    replacement, allow_unapproved=replacement.status != "approved"
                )
                validate_override_ap_comparison_integrity(
                    comparison,
                    original_inventory=original,
                    replacement_inventory=replacement,
                )
                if envelope.get("artifact_hash") != comparison.artifact_hash:
                    raise ValueError("comparison envelope hash mismatch")
            except Exception as exc:
                failures.append(
                    {
                        "requirement_id": requirement_id,
                        "artifact_id": path.stem,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )

    # Older human Y decisions did not create an override artifact.  Report
    # likely runtime semantic-fallback acceptances read-only; never reopen them.
    override_requirements = {
        str(item.get("requirement_id", ""))
        for item in entries
        if item.get("override_id")
    }
    reviewed_root = root / "plain_reading_reviewed"
    if reviewed_root.is_dir():
        for path in sorted(reviewed_root.glob("*.json")):
            requirement_id = path.stem
            if requirement_id in override_requirements:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            review = payload.get("plain_reading_review")
            if not isinstance(review, Mapping):
                continue
            history = review.get("review_history", ())
            if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
                continue
            operations = {
                str(event.get("operation", ""))
                for event in history
                if isinstance(event, Mapping)
            }
            if "plain_reading_semantic_fallback" not in operations:
                continue
            entries.append(
                {
                    "requirement_id": requirement_id,
                    "override_id": "",
                    "override_contract_version": "none",
                    "status": "legacy_human_y_not_reopened",
                    "audit_status": "possible_v1_ap_limit_read_only",
                    "issue_codes": [],
                    "source_lineage": {
                        "source_hash": str(review.get("source_hash", "")),
                        "segmentation_id": str(review.get("segmentation_id", "")),
                        "reading_review_id": str(
                            review.get("reading_review_id", "")
                        ),
                    },
                }
            )

    denominator = max(1, len(entries))
    metrics: dict[str, Any] = {
        "human_override_ap_mode_total": dict(sorted(modes.items())),
        "human_override_ap_inventory_required_rate": inventory_required / denominator,
        "human_override_ap_mapping_required_rate": mapping_required / denominator,
        "human_override_ap_comparison_pass_rate": comparison_passed / denominator,
        "human_override_ap_known_mismatch_rate": known_mismatch / denominator,
        "human_override_ap_unknown_rate": unknown / denominator,
        "human_override_ap_manual_approval_rate": len(
            manually_approved_inventories
        )
        / max(1, len(approved_inventories)),
        "human_override_ap_stale_or_corrupt_count": len(failures),
        "human_override_v1_load_count": v1_load_count,
        "human_override_v1_migration_count": sum(
            1 for item in entries if item.get("supersedes_override_id")
        ),
        "human_override_formula_assisted_count": formula_assisted,
        "human_override_formula_leak_violation_count": sum(
            1
            for item in exclusions
            if item.get("reason") != "formula_assisted_not_formula_blind"
        ),
    }
    return APIdentityAuditReport(
        checkpoint_dir=str(root.resolve()),
        items=tuple(entries),
        metrics=metrics,
        corrupt_or_stale=tuple(failures),
        formula_blind_exclusions=tuple(exclusions),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def normalize_artifact_name(name: str) -> str:
    """Normalize public aliases and reject unknown artifact names."""

    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized.endswith("_fingerprint"):
        normalized = normalized[: -len("_fingerprint")]
    normalized = _ARTIFACT_ALIASES.get(normalized, normalized)
    if normalized not in ARTIFACT_NAMES:
        raise ValueError(f"unknown artifact name: {name}")
    return normalized


def canonicalize(value: Any, *, artifact_kind: str | None = None) -> Any:
    """Return a deterministic JSON-compatible representation of ``value``.

    Mapping order, set iteration order, timestamps, and filesystem paths do not
    affect the result.  Formula-like objects are represented from ``op/args/name``
    rather than their display string.
    """

    kind = normalize_artifact_name(artifact_kind) if artifact_kind else None
    return _canonical_artifact_payload(value, kind, seen=set())


def canonical_json(value: Any, *, artifact_kind: str | None = None) -> str:
    """Serialize the canonical representation using a versioned contract."""

    kind = normalize_artifact_name(artifact_kind) if artifact_kind else "generic"
    envelope = {
        "canonical_hash_version": CANONICAL_HASH_VERSION,
        "artifact_kind": kind,
        "value": canonicalize(
            value,
            artifact_kind=None if kind == "generic" else kind,
        ),
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_stable_hash(value: Any, *, artifact_kind: str | None = None) -> str:
    """Return a SHA-256 content hash under the versioned canonical contract."""

    raw = canonical_json(value, artifact_kind=artifact_kind)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


stable_hash = canonical_stable_hash


def build_artifact_fingerprint_set(
    pipeline_result: Any | None = None,
    *,
    plain_reading: Any | None = None,
    semantic_ir: Any | None = None,
    approved_ir: Any | None = None,
    trace_plan: Any | None = None,
    trace_set: Any | None = None,
    skeleton_plan: Any | None = None,
    hole_plan: Any | None = None,
    candidate_set: Any | None = None,
    evaluation: Any | None = None,
    approval: Any | None = None,
) -> ArtifactFingerprintSet:
    """Fingerprint explicit artifacts or extract them from a pipeline result.

    Explicit non-``None`` arguments override attributes discovered on
    ``pipeline_result``.  No concrete project class is imported.
    """

    if pipeline_result is not None:
        semantic = _read_value(pipeline_result, "semantic")
        plain_reading = _prefer(
            plain_reading,
            _read_value(semantic, "plain_reading_review"),
        )
        semantic_ir = _prefer(
            semantic_ir,
            _read_value(semantic, "faithful_ir"),
        )
        approved_ir = _prefer(
            approved_ir,
            _read_value(semantic, "approved_semantic_ir"),
        )
        trace_plan = _prefer(
            trace_plan,
            _read_value(pipeline_result, "trace_obligation_plan"),
        )
        trace_set = _prefer(trace_set, _read_value(pipeline_result, "traces"))
        skeleton_plan = _prefer(
            skeleton_plan,
            _read_value(pipeline_result, "constrained_skeleton_plan"),
        )
        hole_plan = _prefer(
            hole_plan,
            _read_value(pipeline_result, "completion_hole_plan"),
        )
        candidate_set = _prefer(
            candidate_set,
            _read_value(pipeline_result, "candidates"),
        )
        evaluation = _prefer(
            evaluation,
            _read_value(pipeline_result, "evaluations"),
        )
        approval = _prefer(
            approval,
            _read_value(pipeline_result, "final_approval"),
        )

    values = {
        "plain_reading": plain_reading,
        "semantic_ir": semantic_ir,
        "approved_ir": approved_ir,
        "trace_plan": trace_plan,
        "trace_set": trace_set,
        "skeleton_plan": skeleton_plan,
        "hole_plan": hole_plan,
        "candidate_set": candidate_set,
        "evaluation": evaluation,
        "approval": approval,
    }
    return ArtifactFingerprintSet(
        **{
            f"{name}_fingerprint": (
                ""
                if artifact is None
                else canonical_stable_hash(artifact, artifact_kind=name)
            )
            for name, artifact in values.items()
        }
    )


fingerprint_pipeline_artifacts = build_artifact_fingerprint_set


def compare_artifact_fingerprint_sets(
    before: ArtifactFingerprintSet | Mapping[str, Any] | Any,
    after: ArtifactFingerprintSet | Mapping[str, Any] | Any,
    *,
    expected_changed_artifacts: Iterable[str] = (),
    before_candidates: Any = _UNSET,
    after_candidates: Any = _UNSET,
    before_traces: Any = _UNSET,
    after_traces: Any = _UNSET,
    before_errors: Iterable[Any] = (),
    after_errors: Iterable[Any] = (),
    rerun_id: str = "",
    diagnosis_id: str = "",
    start_stage: str = "",
    frozen_upstream: bool = False,
    independent_approval_required: bool = False,
    regression_detected: bool = False,
) -> RerunDeltaReport:
    """Compare two artifact states and classify the rerun outcome.

    ``before`` and ``after`` may be fingerprint sets, serialized fingerprint
    mappings, or PipelineResult-like objects.  Candidate/trace collections are
    inferred from result-like objects unless explicitly supplied.
    """

    before_raw = before
    after_raw = after
    before_fingerprints = _coerce_fingerprint_set(before)
    after_fingerprints = _coerce_fingerprint_set(after)
    expected = _normalize_artifact_names(expected_changed_artifacts)

    actual = tuple(
        name
        for name in ARTIFACT_NAMES
        if before_fingerprints.fingerprint_for(name)
        != after_fingerprints.fingerprint_for(name)
    )
    present = {
        name
        for name in ARTIFACT_NAMES
        if before_fingerprints.fingerprint_for(name)
        or after_fingerprints.fingerprint_for(name)
    }
    unchanged = tuple(
        name for name in ARTIFACT_NAMES if name in present and name not in actual
    )

    if before_candidates is _UNSET:
        before_candidates = _collection_from_result(before_raw, "candidates")
    if after_candidates is _UNSET:
        after_candidates = _collection_from_result(after_raw, "candidates")
    if before_traces is _UNSET:
        before_traces = _collection_from_result(before_raw, "traces")
    if after_traces is _UNSET:
        after_traces = _collection_from_result(after_raw, "traces")

    before_candidate_entries = _entity_entries(before_candidates, "candidate")
    after_candidate_entries = _entity_entries(after_candidates, "candidate")
    new_candidates = _entry_difference(
        after_candidate_entries,
        before_candidate_entries,
    )
    removed_candidates = _entry_difference(
        before_candidate_entries,
        after_candidate_entries,
    )

    before_trace_entries = _entity_entries(before_traces, "trace")
    after_trace_entries = _entity_entries(after_traces, "trace")
    new_traces = _entry_difference(after_trace_entries, before_trace_entries)

    before_error_entries = _entity_entries(before_errors, "error")
    after_error_entries = _entity_entries(after_errors, "error")
    resolved_errors = _entry_difference(
        before_error_entries,
        after_error_entries,
    )
    remaining_errors = tuple(
        after_error_entries[key] for key in sorted(after_error_entries)
    )
    new_error_keys = set(after_error_entries) - set(before_error_entries)

    meaningful_delta = bool(
        actual
        or new_candidates
        or removed_candidates
        or new_traces
        or resolved_errors
    )
    reasons: list[str] = []
    missing_expected = tuple(name for name in expected if name not in set(actual))
    unexpected = tuple(name for name in actual if expected and name not in set(expected))
    if missing_expected:
        reasons.append(
            "expected artifacts did not change: " + ", ".join(missing_expected)
        )
    if unexpected:
        reasons.append(
            "artifacts changed outside the declared expectation: "
            + ", ".join(unexpected)
        )
    if new_error_keys:
        reasons.append(
            "new errors appeared: "
            + ", ".join(after_error_entries[key] for key in sorted(new_error_keys))
        )

    if frozen_upstream:
        status = "blocked_by_frozen_upstream"
        reasons.append("the responsible upstream artifact is frozen")
    elif regression_detected or new_error_keys:
        status = "regression"
    elif not meaningful_delta:
        status = "no_effective_delta"
        reasons.append("no artifact, candidate, trace, or error state changed")
    elif independent_approval_required:
        status = "needs_independent_approval"
        reasons.append("the effective rerun delta requires fresh independent approval")
    else:
        status = "effective_delta"

    if not rerun_id:
        rerun_id = "rerun-" + canonical_stable_hash(
            {
                "before": before_fingerprints.as_dict(),
                "after": after_fingerprints.as_dict(),
                "diagnosis_id": diagnosis_id,
                "start_stage": start_stage,
                "expected": expected,
            }
        )[:16]

    return RerunDeltaReport(
        rerun_id=rerun_id,
        diagnosis_id=diagnosis_id,
        start_stage=start_stage,
        expected_changed_artifacts=expected,
        actual_changed_artifacts=actual,
        unchanged_artifacts=unchanged,
        new_candidates=new_candidates,
        removed_candidates=removed_candidates,
        new_traces=new_traces,
        resolved_errors=resolved_errors,
        remaining_errors=remaining_errors,
        status=status,
        before_fingerprints=before_fingerprints,
        after_fingerprints=after_fingerprints,
        reasons=tuple(dict.fromkeys(reasons)),
    )


build_rerun_delta_report = compare_artifact_fingerprint_sets
compare_before_after = compare_artifact_fingerprint_sets


def _canonical_artifact_payload(
    value: Any,
    artifact_kind: str | None,
    *,
    seen: set[int],
) -> Any:
    if artifact_kind == "candidate_set":
        payloads = [
            _canonical_candidate(item, seen=seen)
            for item in _artifact_items(
                value,
                collection_keys=("candidates", "members", "items"),
            )
        ]
        return {"candidates": _sorted_unique_payloads(payloads)}
    if artifact_kind == "trace_set":
        payloads = [
            _canonical_value(item, seen=seen)
            for item in _artifact_items(value, collection_keys=("traces", "items"))
        ]
        return {"traces": _sorted_unique_payloads(payloads)}
    if artifact_kind == "evaluation":
        payloads = [
            _canonical_value(item, seen=seen)
            for item in _artifact_items(
                value,
                collection_keys=("evaluations", "items", "members"),
            )
        ]
        return {"evaluations": _sorted_unique_payloads(payloads)}
    return _canonical_value(value, seen=seen)


def _canonical_value(
    value: Any,
    *,
    seen: set[int],
    extra_excluded_keys: frozenset[str] = frozenset(),
) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "infinity" if value > 0 else "-infinity"}
        return 0.0 if value == 0.0 else value
    if isinstance(value, Decimal):
        return {"__decimal__": format(value.normalize(), "f")}
    if isinstance(value, Enum):
        return _canonical_value(value.value, seen=seen)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes_sha256__": hashlib.sha256(bytes(value)).hexdigest()}
    if isinstance(value, (datetime, date, time)):
        return {"__temporal_metadata__": "<excluded>"}
    if isinstance(value, (Path, PurePath)):
        return {"__filesystem_path__": "<excluded>"}
    if _is_formula_like(value):
        return _canonical_formula(value, seen=seen)

    identity = id(value)
    if identity in seen:
        return {"__cycle__": type(value).__qualname__}
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for raw_key, raw_value in value.items():
                key = str(raw_key)
                if _is_volatile_key(key) or key.casefold() in extra_excluded_keys:
                    continue
                if (
                    key.casefold() in _FORMULA_FIELD_NAMES
                    and raw_value is not None
                    and raw_value != ""
                ):
                    result[key] = _canonical_formula_value(raw_value, seen=seen)
                else:
                    result[key] = _canonical_value(
                        raw_value,
                        seen=seen,
                        extra_excluded_keys=extra_excluded_keys,
                    )
            return result
        if is_dataclass(value) and not isinstance(value, type):
            result = {}
            for dataclass_field in fields(value):
                key = dataclass_field.name
                if _is_volatile_key(key) or key.casefold() in extra_excluded_keys:
                    continue
                raw_value = getattr(value, key)
                if (
                    key.casefold() in _FORMULA_FIELD_NAMES
                    and raw_value is not None
                    and raw_value != ""
                ):
                    result[key] = _canonical_formula_value(raw_value, seen=seen)
                else:
                    result[key] = _canonical_value(
                        raw_value,
                        seen=seen,
                        extra_excluded_keys=extra_excluded_keys,
                    )
            return result
        if isinstance(value, (set, frozenset)):
            payloads = [
                _canonical_value(
                    item,
                    seen=seen,
                    extra_excluded_keys=extra_excluded_keys,
                )
                for item in value
            ]
            return _sorted_unique_payloads(payloads)
        if isinstance(value, (list, tuple)):
            return [
                _canonical_value(
                    item,
                    seen=seen,
                    extra_excluded_keys=extra_excluded_keys,
                )
                for item in value
            ]
        if hasattr(value, "__dict__"):
            return _canonical_value(
                {
                    key: item
                    for key, item in vars(value).items()
                    if not key.startswith("_") and not callable(item)
                },
                seen=seen,
                extra_excluded_keys=extra_excluded_keys,
            )
        return {"__type__": type(value).__qualname__, "value": str(value)}
    finally:
        seen.remove(identity)


def _is_formula_like(value: Any) -> bool:
    if isinstance(value, Mapping):
        op = value.get("op")
        return op in _FORMULA_OPERATORS and "args" in value
    op = getattr(value, "op", None)
    return op in _FORMULA_OPERATORS and hasattr(value, "args")


def _canonical_formula(value: Any, *, seen: set[int]) -> dict[str, Any]:
    op = _read_value(value, "op", "")
    args = _read_value(value, "args", ()) or ()
    name = _read_value(value, "name", "")
    return {
        "op": str(op),
        "name": str(name) if op == "ap" else "",
        "args": [_canonical_formula(item, seen=seen) for item in args],
    }


def _canonical_formula_value(value: Any, *, seen: set[int]) -> Any:
    if _is_formula_like(value):
        return _canonical_formula(value, seen=seen)
    if isinstance(value, str):
        try:
            from ltlf_parser import parse_formula

            parsed = parse_formula(value)
        except (ImportError, TypeError, ValueError):
            return {"formula_text": " ".join(value.split())}
        return _canonical_formula(parsed, seen=seen)
    return _canonical_value(value, seen=seen)


def _canonical_candidate(value: Any, *, seen: set[int]) -> Any:
    formula = _candidate_formula(value)
    return _canonical_formula_value(formula, seen=seen)


def _candidate_formula(value: Any) -> Any:
    if _is_formula_like(value) or isinstance(value, str):
        return value
    for key in (
        "formula",
        "candidate_formula",
        "approved_formula",
        "final_formula",
        "selected_formula",
    ):
        candidate = _read_value(value, key, _UNSET)
        if candidate is not _UNSET and candidate not in {None, ""}:
            return candidate
    return value


def _artifact_items(
    value: Any,
    *,
    collection_keys: Sequence[str],
) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)) or _is_formula_like(value):
        return [value]
    if isinstance(value, Mapping):
        for key in collection_keys:
            nested = value.get(key, _UNSET)
            if nested is not _UNSET:
                return _artifact_items(nested, collection_keys=())
        return [value]
    for key in collection_keys:
        nested = getattr(value, key, _UNSET)
        if nested is not _UNSET:
            return _artifact_items(nested, collection_keys=())
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _sorted_unique_payloads(payloads: Iterable[Any]) -> list[Any]:
    by_text = {_payload_text(payload): payload for payload in payloads}
    return [by_text[text] for text in sorted(by_text)]


def _payload_text(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _is_volatile_key(key: str) -> bool:
    normalized = key.casefold()
    if normalized in _VOLATILE_KEYS or "timestamp" in normalized:
        return True
    return normalized.endswith(
        (
            "_path",
            "_paths",
            "_directory",
            "_directories",
        )
    )


def _prefer(explicit: Any, discovered: Any) -> Any:
    return discovered if explicit is None else explicit


def _read_value(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalize_artifact_names(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    normalized = {normalize_artifact_name(value) for value in values}
    return tuple(name for name in ARTIFACT_NAMES if name in normalized)


def _coerce_fingerprint_set(value: Any) -> ArtifactFingerprintSet:
    if isinstance(value, ArtifactFingerprintSet):
        return value
    if isinstance(value, Mapping) and any(
        f"{name}_fingerprint" in value for name in ARTIFACT_NAMES
    ):
        return ArtifactFingerprintSet.from_mapping(value)
    return build_artifact_fingerprint_set(value)


def _collection_from_result(value: Any, attribute: str) -> Any:
    if isinstance(value, ArtifactFingerprintSet):
        return None
    if isinstance(value, Mapping) and any(
        f"{name}_fingerprint" in value for name in ARTIFACT_NAMES
    ):
        return None
    return _read_value(value, attribute)


def _entity_entries(value: Any, entity_kind: str) -> dict[str, str]:
    if value is None or value is _UNSET:
        return {}
    items = _artifact_items(value, collection_keys=(f"{entity_kind}s", "items"))
    entries: dict[str, str] = {}
    for item in items:
        if entity_kind == "candidate":
            payload = _canonical_candidate(item, seen=set())
            display = _candidate_display(item)
        elif entity_kind == "trace":
            payload = _canonical_value(item, seen=set())
            display = _trace_display(item)
        elif entity_kind == "error":
            payload = _canonical_value(
                item,
                seen=set(),
                extra_excluded_keys=_ERROR_IDENTITY_KEYS,
            )
            display = _error_display(item)
        else:  # pragma: no cover - private API guard
            raise ValueError(f"unknown entity kind: {entity_kind}")
        key = hashlib.sha256(_payload_text(payload).encode("utf-8")).hexdigest()
        entries.setdefault(key, display or f"{entity_kind}-{key[:12]}")
    return entries


def _candidate_display(value: Any) -> str:
    formula = _candidate_formula(value)
    if isinstance(formula, str):
        try:
            from ltlf_parser import parse_formula

            return str(parse_formula(formula))
        except (ImportError, TypeError, ValueError):
            return " ".join(formula.split())
    return str(formula)


def _trace_display(value: Any) -> str:
    return str(
        _read_value(value, "trace_id", "")
        or _read_value(value, "name", "")
        or ""
    )


def _error_display(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    for key in (
        "diagnosis_id",
        "error_id",
        "error_class",
        "error_type",
        "message",
        "error",
    ):
        display = _read_value(value, key, "")
        if display:
            return str(display)
    return str(value)


def _entry_difference(
    left: Mapping[str, str],
    right: Mapping[str, str],
) -> tuple[str, ...]:
    return tuple(left[key] for key in sorted(set(left) - set(right)))


__all__ = [
    "ARTIFACT_NAMES",
    "ArtifactFingerprintSet",
    "CANONICAL_HASH_VERSION",
    "RERUN_DELTA_STATUSES",
    "RerunDeltaReport",
    "build_artifact_fingerprint_set",
    "build_rerun_delta_report",
    "canonical_json",
    "canonical_stable_hash",
    "canonicalize",
    "compare_artifact_fingerprint_sets",
    "compare_before_after",
    "fingerprint_pipeline_artifacts",
    "normalize_artifact_name",
    "stable_hash",
]
