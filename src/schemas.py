"""
Traj-LTLf 中跨模块共享的数据结构。

把这些结构单独放在一个文件里，是为了让 pipeline、Agent 和人工评审模块
都使用同一种数据格式，避免每个文件各写一套字段。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from contract_versions import (
    AP_INVENTORY_CANONICALIZER_VERSION,
    AP_INVENTORY_CONTRACT_VERSION,
    AP_INVENTORY_SCHEMA_VERSION,
    AP_INVENTORY_VALIDATOR_VERSION,
    AP_REPRESENTATION_PROFILE_CONTRACT_VERSION,
    BATCH_DISPOSITION_CONTRACT_VERSION,
    FEEDBACK_RERUN_SCHEMA_VERSION,
    FAITHFUL_IR_CANONICALIZER_VERSION,
    FAITHFUL_IR_CONTRACT_VERSION,
    FAITHFUL_IR_SCHEMA_VERSION,
    FAITHFUL_IR_VALIDATOR_VERSION,
    HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION,
    IR_RELATION_FEASIBILITY_CONTRACT_VERSION,
    IR_WITNESS_SOLVER_CONTRACT_VERSION,
    OVERRIDE_AP_COMPARISON_CONTRACT_VERSION,
    TEMPORAL_LANGUAGE_CONTRACT_VERSION,
)
from ltlf_parser import Formula


WARMUP_LEARNING_MODE = "trajectory_guided_warmup_v2"
WARMUP_MEMORY_MODES = ("read_only", "read_write")
LEARNING_MODES = (
    "stage_guided",
    "offline_formula_validation",
    "legacy_formula_oracle",
)
OFFLINE_VALIDATION_MODES = ("none", "final_formula")

DERIVED_PLAN_VERSION = "4"
SURFACE_SEGMENTATION_SCHEMA_VERSION = "1"
PLAIN_READING_SCHEMA_VERSION = "5"

TRACE_PURPOSE_SEMANTIC_VALIDATION = "semantic_validation"
TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION = "ir_branch_discrimination"
TRACE_PURPOSE_OPERATOR_DISCRIMINATION = "operator_assignment_discrimination"
TRACE_EVIDENCE_PURPOSES = (
    TRACE_PURPOSE_SEMANTIC_VALIDATION,
    TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION,
    TRACE_PURPOSE_OPERATOR_DISCRIMINATION,
)

SURFACE_SEGMENTATION_STATUSES = (
    "pending",
    "generated",
    "mechanically_valid",
    "mechanically_invalid",
    "review_pending",
    "surface_reviewed",
    "boundary_revision_requested",
    "ambiguity_flagged",
    "stale",
    "blocked_error",
)
PLAIN_READING_STATUSES = (
    "pending_detection",
    "pending_no_ambiguity_confirmation",
    "awaiting_human_semantic_override",
    "user_requested_ambiguity_generation",
    "no_ambiguity",
    "pending_plain_review",
    "reviewer_selected",
    "plain_reading_reviewed",
    "ambiguous_unresolved",
    "options_revision_requested",
    "neither_option_selected",
    "needs_source_clarification",
    "deferred_pending_user_confirmation",
    "blocked_error",
    "stale",
)

HUMAN_SEMANTIC_OVERRIDE_STATUSES = (
    "awaiting_input",
    "awaiting_source_modality_choice",
    "awaiting_ap_inventory",
    "awaiting_ap_mapping_confirmation",
    "ap_rejected",
    "awaiting_confirmation",
    "active",
    "cancelled",
    "superseded",
)

AP_REPRESENTATION_MODES = (
    "controlled_symbolic",
    "declared_catalog",
    "natural_language_grounded",
    "offline_formula_assisted",
    "undetermined",
)
AP_INVENTORY_STATUSES = (
    "proposed",
    "pending_human_review",
    "approved",
    "rejected",
    "stale",
)
OVERRIDE_AP_COMPARISON_STATUSES = (
    "needs_inventory",
    "needs_mapping_review",
    "passed",
    "mismatch",
    "stale",
    "corrupt",
)


@dataclass(frozen=True)
class SurfaceSegment:
    """A read-only, backend-derived slice of the exact source text."""

    segment_id: str
    index: int
    start: int
    end: int
    text: str

    @property
    def display_text(self) -> str:
        """Whitespace-trimmed UI view; never replaces ``text``."""

        return self.text.strip()


@dataclass(frozen=True)
class MechanicalSegmentationValidation:
    status: str
    exact_reconstruction: bool
    gap_count: int = 0
    overlap_count: int = 0
    reordered: bool = False
    failure_type: str = ""
    first_difference_offset: int | None = None
    missing_or_changed_excerpt: str = ""
    instruction: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed"}:
            raise ValueError("mechanical validation status must be passed or failed")
        if self.status == "passed" and not self.exact_reconstruction:
            raise ValueError("passed mechanical validation requires exact reconstruction")


@dataclass(frozen=True)
class ArtifactGenerator:
    mode: str = "rules"
    model: str = ""
    prompt_version: str = ""


@dataclass(frozen=True)
class SegmentationReviewRecord:
    round: int
    decision: str
    previous_segmentation_id: str
    result_segmentation_id: str
    feedback: str = ""
    reviewer_id: str = ""
    reviewer_role: str = "ordinary_reviewer"
    review_basis: str = "surface_only"
    label_source: str = "human"
    started_at: str = ""
    completed_at: str = ""


@dataclass(frozen=True)
class SurfaceSegmentation:
    """Lossless surface boundaries with no semantic authority."""

    segmentation_id: str
    revision: int
    requirement_id: str
    source_text: str
    source_hash: str
    marker: str
    segments: tuple[SurfaceSegment, ...]
    mechanical_validation: MechanicalSegmentationValidation
    schema_version: str = SURFACE_SEGMENTATION_SCHEMA_VERSION
    source_hash_algorithm: str = "sha256"
    offset_unit: str = "unicode_codepoint"
    normalized_text: str = ""
    review_scope: str = "surface_only"
    semantic_authority: str = "none"
    status: str = "review_pending"
    blocked_stage: str = ""
    generator: ArtifactGenerator = ArtifactGenerator()
    experiment_mode: str = "rules"
    data_split: str = "test"
    label_source: str = "rules"
    review_basis: str = "mechanical_and_rules"
    persistent_rag_allowed: bool = False
    created_at: str = ""
    updated_at: str = ""
    review_history: tuple[SegmentationReviewRecord, ...] = ()
    contract_version: str = "surface-segmentation-contract-v2"
    canonicalizer_version: str = "surface-segmentation-canonicalizer-v2"
    validator_version: str = "surface-segmentation-validator-v2"
    repair_policy_version: str = "validated-generation-repair-policy-v1"
    artifact_hash: str = ""
    issue_fingerprint: str = ""
    repair_history: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SURFACE_SEGMENTATION_SCHEMA_VERSION:
            raise ValueError("unsupported surface segmentation schema_version")
        if self.source_hash_algorithm != "sha256":
            raise ValueError("surface segmentation source hash must use sha256")
        if self.offset_unit != "unicode_codepoint":
            raise ValueError("surface segmentation offsets must use unicode_codepoint")
        if self.review_scope != "surface_only" or self.semantic_authority != "none":
            raise ValueError("surface segmentation must not claim semantic authority")
        if self.status not in SURFACE_SEGMENTATION_STATUSES:
            raise ValueError("unsupported surface segmentation status: " + self.status)
        from validated_generation import artifact_hash

        computed_hash = artifact_hash(self)
        if self.artifact_hash != computed_hash:
            object.__setattr__(self, "artifact_hash", computed_hash)


@dataclass(frozen=True)
class PlainReadingOption:
    option_id: str
    plain_interpretation: str
    difference_summary: str


@dataclass(frozen=True)
class PlainReadingAmbiguity:
    ambiguity_id: str
    source_segment_ids: tuple[str, ...]
    question: str
    options: tuple[PlainReadingOption, ...]
    source_span: tuple[int, int] = (-1, -1)
    evidence_text: str = ""
    decision: str = ""
    decision_status: str = "pending_plain_review"
    review_basis: str = ""
    reviewer_id: str = ""
    reviewer_role: str = "ordinary_reviewer"
    reviewer_source: str = ""
    semantic_authority: str = "non_authoritative"
    comments: str = ""

    def __post_init__(self) -> None:
        if self.semantic_authority != "non_authoritative":
            raise ValueError("plain reading decisions are non-authoritative")


PLAIN_READING_REVISION_VALIDATION_STATUSES = (
    "pending_validation",
    "accepted",
    "rejected",
    "needs_human_confirmation",
)

PLAIN_READING_REVISION_CLASSIFICATIONS = (
    "equivalent_to_existing",
    "same_axis_revision",
    "new_independent_axis",
    "out_of_source",
    "unresolved",
)

PLAIN_READING_REVISION_REQUEST_STATUSES = (
    "submitted",
    "suggested",
    "confirmed",
    "applied",
    "rejected",
    "withdrawn",
    "superseded",
)


@dataclass(frozen=True)
class PlainReadingRevisionRequest:
    """Non-authoritative proposal to repair one binary reading question."""

    request_id: str
    reading_review_id: str
    ambiguity_id: str
    segmentation_id: str
    source_hash: str
    proposed_interpretation: str
    reason: str
    source_segment_ids: tuple[str, ...]
    reviewer_id: str = ""
    reviewer_role: str = "ordinary_reviewer"
    reviewer_source: str = "human"
    validation_status: str = "pending_validation"
    classification: str = "unresolved"
    classification_basis: str = ""
    human_confirmed: bool = False
    request_status: str = "submitted"
    source_warnings: tuple[str, ...] = ()
    parent_decision_fingerprint: str = ""
    confirmed_by: str = ""
    confirmed_role: str = ""
    confirmed_at: str = ""
    applied_at: str = ""
    revision_round: int = 1
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if self.validation_status not in PLAIN_READING_REVISION_VALIDATION_STATUSES:
            raise ValueError("unsupported plain-reading revision validation_status")
        if self.classification not in PLAIN_READING_REVISION_CLASSIFICATIONS:
            raise ValueError("unsupported plain-reading revision classification")
        if self.request_status not in PLAIN_READING_REVISION_REQUEST_STATUSES:
            raise ValueError("unsupported plain-reading revision request_status")
        if self.revision_round < 1:
            raise ValueError("plain-reading revision_round must be positive")
        if self.human_confirmed and self.validation_status not in {"accepted", "rejected"}:
            raise ValueError("confirmed revision must be accepted or rejected")
        if self.request_status in {"confirmed", "applied", "rejected"} and not self.human_confirmed:
            raise ValueError("terminal revision status requires human confirmation")
        if self.request_status == "applied" and not self.applied_at:
            raise ValueError("applied revision requires applied_at")


@dataclass(frozen=True)
class NoAmbiguityConfirmation:
    """Explicit human or evidence-bound Judge disposition of an empty axis set."""

    status: str = "not_required"
    llm_proposed_no_ambiguity: bool = False
    human_confirmed: bool = False
    reviewer_id: str = ""
    reviewer_role: str = "ordinary_reviewer"
    reviewer_source: str = ""
    comment: str = ""
    confirmed_at: str = ""
    judge_confirmed: bool = False
    judge_evidence_json: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"not_required", "pending", "confirmed", "rejected", "withdrawn"}:
            raise ValueError("unsupported no-ambiguity confirmation status")
        if self.judge_confirmed and (self.human_confirmed or self.status != "confirmed" or self.reviewer_source not in {"independent_judge", "model_assistant"} or not self.judge_evidence_json or not self.confirmed_at):
            raise ValueError("Judge confirmation requires separate evidence and must not claim human action")
        if self.status in {"confirmed", "rejected", "withdrawn"} and not (self.human_confirmed or self.judge_confirmed):
            raise ValueError("terminal no-ambiguity confirmation requires human action")
        if self.human_confirmed and not self.confirmed_at:
            raise ValueError("human no-ambiguity action requires confirmed_at")


@dataclass(frozen=True)
class APRepresentationProfile:
    """Versioned declaration of how AP identities are represented in a source."""

    profile_id: str = "AP-PROFILE-undetermined-v1"
    mode: str = "undetermined"
    symbolic_token_pattern: str = ""
    symbolic_grammar_id: str = ""
    declared_catalog_id: str = ""
    allow_formula_assisted_catalog: bool = False
    requires_human_grounding: bool = True
    language: str = ""
    domain: str = ""
    contract_version: str = AP_REPRESENTATION_PROFILE_CONTRACT_VERSION
    provenance: str = "implicit_untagged"

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("AP representation profile requires profile_id")
        if self.mode not in AP_REPRESENTATION_MODES:
            raise ValueError("unsupported AP representation mode")
        if self.contract_version != AP_REPRESENTATION_PROFILE_CONTRACT_VERSION:
            raise ValueError("unsupported AP representation profile contract")
        if self.mode == "declared_catalog" and not self.declared_catalog_id:
            raise ValueError("declared_catalog profile requires catalog id")
        if self.mode != "offline_formula_assisted" and self.allow_formula_assisted_catalog:
            raise ValueError("formula-assisted catalog is forbidden for this profile")


@dataclass(frozen=True)
class APCatalogEntry:
    catalog_ap_id: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    symbolic_declared: bool = False

    def __post_init__(self) -> None:
        if not self.catalog_ap_id or not self.canonical_name:
            raise ValueError("AP catalog entry requires id and canonical_name")


@dataclass(frozen=True)
class DeclaredAPCatalog:
    catalog_id: str
    catalog_version: str
    entries: tuple[APCatalogEntry, ...]
    catalog_hash: str
    provenance: str
    source: str = "declared_catalog"

    def __post_init__(self) -> None:
        if not self.catalog_id or not self.catalog_version or not self.catalog_hash:
            raise ValueError("declared AP catalog requires versioned identity")
        ids = [entry.catalog_ap_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("declared AP catalog entry ids must be unique")
        if not self.provenance:
            raise ValueError("declared AP catalog requires provenance")


@dataclass(frozen=True)
class APIdentity:
    ap_identity_id: str
    canonical_name: str
    ordinal: int
    aliases: tuple[str, ...] = ()
    identity_origin: str = "source_inventory"

    def __post_init__(self) -> None:
        if not self.ap_identity_id or not self.canonical_name:
            raise ValueError("AP identity requires id and canonical_name")
        if self.ordinal < 1:
            raise ValueError("AP identity ordinal must be positive")


@dataclass(frozen=True)
class APMention:
    mention_id: str
    ap_identity_id: str
    evidence_text: str
    source_span: tuple[int, int]
    supporting_segment_ids: tuple[str, ...]
    predicate_span: tuple[int, int]
    occurrence_index: int
    mapping_provenance: str

    def __post_init__(self) -> None:
        if not self.mention_id or not self.ap_identity_id or not self.evidence_text:
            raise ValueError("AP mention requires identity and exact evidence")
        start, end = self.source_span
        predicate_start, predicate_end = self.predicate_span
        if start < 0 or end <= start:
            raise ValueError("AP mention source_span is invalid")
        if predicate_start < start or predicate_end > end or predicate_end <= predicate_start:
            raise ValueError("AP mention predicate_span is invalid")
        if self.occurrence_index < 1:
            raise ValueError("AP mention occurrence_index must be positive")


@dataclass(frozen=True)
class APInventory:
    inventory_id: str
    requirement_id: str
    source_text: str
    source_hash: str
    segmentation_id: str
    representation_profile_id: str
    extraction_mode: str
    extractor_version: str
    status: str
    identities: tuple[APIdentity, ...]
    mentions: tuple[APMention, ...]
    reviewer_role: str = "system"
    reviewer_id: str = ""
    reviewer_source: str = "deterministic"
    provenance: str = ""
    parent_inventory_id: str = ""
    approved_empty: bool = False
    approved_empty_reason: str = ""
    eligible_for_formula_blind_evaluation: bool = True
    persistent_rag_allowed: bool = False
    catalog_source: str = ""
    formula_source_hash: str = ""
    issue_codes: tuple[str, ...] = ()
    schema_version: int = AP_INVENTORY_SCHEMA_VERSION
    contract_version: str = AP_INVENTORY_CONTRACT_VERSION
    canonicalizer_version: str = AP_INVENTORY_CANONICALIZER_VERSION
    validator_version: str = AP_INVENTORY_VALIDATOR_VERSION
    artifact_hash: str = ""
    created_at: str = ""
    confirmed_at: str = ""

    def __post_init__(self) -> None:
        if not self.inventory_id or not self.requirement_id:
            raise ValueError("AP inventory requires stable ids")
        if self.status not in AP_INVENTORY_STATUSES:
            raise ValueError("unsupported AP inventory status")
        if self.schema_version != AP_INVENTORY_SCHEMA_VERSION:
            raise ValueError("unsupported AP inventory schema")
        if self.contract_version != AP_INVENTORY_CONTRACT_VERSION:
            raise ValueError("unsupported AP inventory contract")
        identity_ids = [identity.ap_identity_id for identity in self.identities]
        if len(identity_ids) != len(set(identity_ids)):
            raise ValueError("AP inventory identity ids must be unique")
        mention_ids = [mention.mention_id for mention in self.mentions]
        if len(mention_ids) != len(set(mention_ids)):
            raise ValueError("AP inventory mention ids must be unique")
        if self.approved_empty and self.identities:
            raise ValueError("approved-empty AP inventory must have no identities")
        if self.approved_empty and not self.approved_empty_reason:
            raise ValueError("approved-empty AP inventory requires a reason")


@dataclass(frozen=True)
class APIdentityMapping:
    replacement_ap_identity_id: str
    original_ap_identity_id: str
    mapping_provenance: str = "human"

    def __post_init__(self) -> None:
        if not self.replacement_ap_identity_id or not self.original_ap_identity_id:
            raise ValueError("AP identity mapping requires both identities")


@dataclass(frozen=True)
class OverrideAPComparison:
    comparison_id: str
    override_id: str
    original_inventory_id: str
    original_inventory_hash: str
    replacement_inventory_id: str
    replacement_inventory_hash: str
    identity_mapping: tuple[APIdentityMapping, ...]
    unmapped_original_ids: tuple[str, ...]
    unmapped_replacement_ids: tuple[str, ...]
    validation_mode: str
    validation_version: str
    status: str
    issue_codes: tuple[str, ...]
    reviewer_role: str = "system"
    reviewer_id: str = ""
    reviewer_source: str = "deterministic"
    artifact_hash: str = ""
    confirmed_at: str = ""

    def __post_init__(self) -> None:
        if not self.comparison_id or not self.override_id:
            raise ValueError("override AP comparison requires stable ids")
        if self.status not in OVERRIDE_AP_COMPARISON_STATUSES:
            raise ValueError("unsupported override AP comparison status")
        if self.validation_version != OVERRIDE_AP_COMPARISON_CONTRACT_VERSION:
            raise ValueError("unsupported override AP comparison contract")


@dataclass(frozen=True)
class HumanSemanticOverride:
    """Auditable whole-source replacement supplied by an ordinary reviewer."""

    override_id: str
    requirement_id: str
    original_source_text: str
    original_source_hash: str
    replacement_text: str
    replacement_hash: str
    original_ap_set: tuple[str, ...]
    replacement_ap_set: tuple[str, ...]
    ap_validation_status: str
    parent_effective_source_hash: str
    reviewer_role: str = "ordinary_reviewer"
    reviewer_id: str = ""
    reviewer_source: str = "human"
    confirmation_status: str = "pending"
    status: str = "awaiting_input"
    issue_codes: tuple[str, ...] = ()
    invalidation_reason: str = ""
    old_effective_source_hash: str = ""
    new_effective_source_hash: str = ""
    created_at: str = ""
    confirmed_at: str = ""
    override_contract_version: str = HUMAN_SEMANTIC_OVERRIDE_V1_CONTRACT_VERSION
    ap_validation_contract_version: str = "symbolic-token-v1"
    representation_profile_id: str = ""
    original_ap_inventory_id: str = ""
    original_ap_inventory_hash: str = ""
    replacement_ap_inventory_id: str = ""
    replacement_ap_inventory_hash: str = ""
    ap_comparison_id: str = ""
    ap_comparison_hash: str = ""
    ap_validation_mode: str = "symbolic_token_v1"
    supersedes_override_id: str = ""

    def __post_init__(self) -> None:
        if not self.override_id or not self.requirement_id:
            raise ValueError("human semantic override requires stable ids")
        if self.status not in HUMAN_SEMANTIC_OVERRIDE_STATUSES:
            raise ValueError("unsupported human semantic override status")
        if self.reviewer_role != "ordinary_reviewer":
            raise ValueError("human semantic override requires ordinary_reviewer")
        if self.ap_validation_status not in {
            "pending",
            "passed",
            "rejected",
            "needs_review",
        }:
            raise ValueError("unsupported override AP validation status")
        if self.confirmation_status not in {
            "pending",
            "confirmed",
            "rejected",
        }:
            raise ValueError("unsupported override confirmation status")
        computed_original_hash = hashlib.sha256(
            self.original_source_text.encode("utf-8")
        ).hexdigest()
        if self.original_source_hash != computed_original_hash:
            raise ValueError("human semantic override original source hash mismatch")
        if self.replacement_text:
            computed_replacement_hash = hashlib.sha256(
                self.replacement_text.encode("utf-8")
            ).hexdigest()
            if self.replacement_hash != computed_replacement_hash:
                raise ValueError("human semantic override replacement hash mismatch")
        elif self.replacement_hash:
            raise ValueError("empty semantic replacement must not have a hash")
        if self.status == "active":
            if self.ap_validation_status != "passed":
                raise ValueError("active semantic override requires passed AP validation")
            if self.confirmation_status != "confirmed" or not self.confirmed_at:
                raise ValueError("active semantic override requires explicit confirmation")
            if self.replacement_hash != self.new_effective_source_hash:
                raise ValueError("active semantic override hash lineage is inconsistent")


@dataclass(frozen=True)
class AmbiguityGenerationTrigger:
    """User request that asks the LLM to propose at least one candidate axis."""

    source: str = ""
    suspected_segment_ids: tuple[str, ...] = ()
    user_comment: str = ""
    previous_reading_review_id: str = ""
    requested_at: str = ""
    generation_attempts: int = 0
    last_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.generation_attempts < 0:
            raise ValueError("ambiguity generation attempts must be non-negative")


@dataclass(frozen=True)
class PlainReadingReview:
    """Ordinary-language readings linked to one reviewed segmentation."""

    reading_review_id: str
    requirement_id: str
    source_hash: str
    segmentation_id: str
    ambiguities: tuple[PlainReadingAmbiguity, ...] = ()
    revision_requests: tuple[PlainReadingRevisionRequest, ...] = ()
    no_ambiguity_confirmation: NoAmbiguityConfirmation = NoAmbiguityConfirmation()
    ambiguity_generation_trigger: AmbiguityGenerationTrigger = AmbiguityGenerationTrigger()
    schema_version: str = PLAIN_READING_SCHEMA_VERSION
    overall_status: str = "pending_detection"
    review_basis: str = ""
    reviewer_source: str = ""
    semantic_authority: str = "non_authoritative"
    generator: ArtifactGenerator = ArtifactGenerator()
    experiment_mode: str = "off"
    data_split: str = "test"
    persistent_rag_allowed: bool = False
    blocked_stage: str = ""
    created_at: str = ""
    updated_at: str = ""
    review_history: tuple[dict[str, Any], ...] = ()
    downstream_status: str = "current"
    supersedes_reading_review_id: str = ""
    contract_version: str = "plain-reading-contract-v6"
    canonicalizer_version: str = "plain-reading-canonicalizer-v1"
    validator_version: str = "plain-reading-validator-v6"
    repair_policy_version: str = "validated-generation-repair-policy-v1"
    artifact_hash: str = ""
    issue_fingerprint: str = ""
    repair_history: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != PLAIN_READING_SCHEMA_VERSION:
            raise ValueError("unsupported plain reading schema_version")
        if self.overall_status not in PLAIN_READING_STATUSES:
            raise ValueError("unsupported plain reading status: " + self.overall_status)
        if self.semantic_authority != "non_authoritative":
            raise ValueError("plain reading review must remain non-authoritative")
        if self.downstream_status not in {"current", "stale_due_to_revision"}:
            raise ValueError("unsupported plain reading downstream_status")
        from validated_generation import artifact_hash

        computed_hash = artifact_hash(self)
        if self.artifact_hash != computed_hash:
            object.__setattr__(self, "artifact_hash", computed_hash)

    @property
    def unresolved_ambiguity_ids(self) -> tuple[str, ...]:
        return tuple(
            ambiguity.ambiguity_id
            for ambiguity in self.ambiguities
            if ambiguity.decision_status == "ambiguous_unresolved"
        )

    @property
    def selected_options(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (ambiguity.ambiguity_id, ambiguity.decision)
            for ambiguity in self.ambiguities
            if ambiguity.decision_status == "reviewer_selected"
            and ambiguity.decision
        )

    @property
    def pending_revision_request_ids(self) -> tuple[str, ...]:
        return tuple(
            request.request_id
            for request in self.revision_requests
            if request.request_status in {"submitted", "suggested", "confirmed"}
        )


@dataclass(frozen=True)
class AtomicProposition:
    """One AP declaration and its traceable natural-language evidence."""

    ap_id: str
    name: str
    ap_identity_id: str = ""
    ap_mention_ids: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    source_span: tuple[int, int] = (-1, -1)
    evidence_text: str = ""
    supporting_segment_ids: tuple[str, ...] = ()
    formula_presence: str = "unspecified"
    required_in_formula: bool = False
    optional_in_formula: bool = False


@dataclass(frozen=True)
class SemanticNode:
    """A formula-free node in the faithful semantic IR graph."""

    node_id: str
    node_type: str
    children: tuple[str, ...] = ()
    semantic_role: str = ""
    scope: str = "current_position"
    temporal_relation: str = "none"
    boundary_policy: str = "finite_trace_default"
    allowed_semantic_readings: tuple[str, ...] = ()
    temporal_resolution: str = "fixed"
    parameters: tuple[tuple[str, Any], ...] = ()
    source_span: tuple[int, int] = (-1, -1)
    evidence_text: str = ""
    confidence: float = 0.0
    supporting_segment_ids: tuple[str, ...] = ()

    def parameter(self, name: str, default: Any = None) -> Any:
        return dict(self.parameters).get(name, default)


@dataclass(frozen=True)
class SemanticAlternativePatch:
    """One executable, NL-supported update for an unresolved IR choice."""

    alternative_id: str
    node_id: str
    parameter_updates: tuple[tuple[str, Any], ...] = ()
    node_type: str = ""
    description: str = ""


@dataclass(frozen=True)
class SemanticAmbiguity:
    ambiguity_id: str
    description: str
    affected_node_ids: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    alternative_patches: tuple[SemanticAlternativePatch, ...] = ()
    risk: str = "medium"
    resolved: bool = False
    origin_layer: str = "ir_extraction"
    axis_kind: str = ""
    related_axis_ids: tuple[str, ...] = ()
    relation_kind: str = "independent"


@dataclass(frozen=True)
class IRProvenance:
    extractor: str = "unknown"
    extraction_mode: str = "generated"
    source: str = "natural_language"
    created_at: str = ""
    approved: bool = False
    approval_source: str = ""


@dataclass(frozen=True)
class PeriodicitySemanticProfile:
    profile_id: str = "periodicity-event-relative-v1"
    default_origin: str = "first_occurrence"
    timestamp_indexing: str = "zero_based"
    default_terminal_policy: str = "finite_prefix"
    allow_explicit_fixed_phase: bool = True
    empty_trace_domain: str = "unsupported"
    model_decision_without_basis: str = "forbidden"
    version: str = "1"
    provenance: str = "project_policy"


@dataclass(frozen=True)
class SemanticParameterDecision:
    decision_id: str
    semantic_path: str
    parameter_name: str
    value: Any
    decision_source: str
    source_evidence: str = ""
    parent_decision_ids: tuple[str, ...] = ()
    mapping_rule_id: str = ""
    confidence: float = 0.0
    approved: bool = False


@dataclass(frozen=True)
class SemanticResolutionRecord:
    resolution_id: str
    requirement_id: str
    source_hash: str
    segmentation_id: str = ""
    reading_review_id: str = ""
    decision_fingerprint: str = ""
    profile_id: str = "periodicity-event-relative-v1"
    status: str = "pending_resolution"
    decisions: tuple[SemanticParameterDecision, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    reviewer: str = ""
    created_at: str = ""
    completed_at: str = ""
    artifact_hash: str = ""


@dataclass(frozen=True)
class FaithfulSemanticIR:
    """Versioned, evidence-backed semantic source of truth.

    This is a domain semantic graph, not an LTLf formula AST.  In particular,
    neither this object nor any node has a formula/formula_text field.
    """

    requirement_id: str
    source_text: str
    atomic_propositions: tuple[AtomicProposition, ...]
    root_node_id: str
    nodes: tuple[SemanticNode, ...]
    ambiguities: tuple[SemanticAmbiguity, ...] = ()
    extraction_confidence: float = 0.0
    provenance: IRProvenance = IRProvenance()
    schema_version: str = FAITHFUL_IR_SCHEMA_VERSION
    ir_version: str = ""
    segmentation_id: str = ""
    reading_review_id: str = ""
    reading_review_basis: str = ""
    resolution_id: str = ""
    resolution_fingerprint: str = ""
    periodicity_profile_id: str = "periodicity-event-relative-v1"
    periodicity_profile_version: str = "1"
    contract_version: str = FAITHFUL_IR_CONTRACT_VERSION
    canonicalizer_version: str = FAITHFUL_IR_CANONICALIZER_VERSION
    validator_version: str = FAITHFUL_IR_VALIDATOR_VERSION
    repair_policy_version: str = "validated-generation-repair-policy-v1"
    artifact_hash: str = ""
    issue_fingerprint: str = ""
    repair_history: tuple[dict[str, Any], ...] = ()
    migration_history: tuple[dict[str, Any], ...] = ()
    source_hash: str = ""
    ap_context_fingerprint: str = ""
    ap_profile_mode: str = ""
    catalog_authority: str = "none"
    ap_grounding_fingerprint: str = ""
    ap_grounding_authority: str = "proposal_only"
    node_contract_fingerprint: str = ""
    temporal_language_contract_version: str = TEMPORAL_LANGUAGE_CONTRACT_VERSION
    semantic_source_contract_version: str = ""
    effective_source_fingerprint: str = ""
    effective_source_kind: str = ""
    effective_source_authority_record_id: str = ""

    def __post_init__(self) -> None:
        import hashlib

        if not self.node_contract_fingerprint:
            from semantic_node_contracts import node_contract_fingerprint

            object.__setattr__(
                self,
                "node_contract_fingerprint",
                node_contract_fingerprint(),
            )
        computed_source_hash = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_hash != computed_source_hash:
            object.__setattr__(self, "source_hash", computed_source_hash)
        from validated_generation import artifact_hash

        computed_hash = artifact_hash(self)
        if self.artifact_hash != computed_hash:
            object.__setattr__(self, "artifact_hash", computed_hash)

    @property
    def node_map(self) -> dict[str, SemanticNode]:
        return {node.node_id: node for node in self.nodes}

    @property
    def ap_names(self) -> tuple[str, ...]:
        return tuple(ap.name for ap in self.atomic_propositions)

    @property
    def has_high_risk_ambiguity(self) -> bool:
        return any(
            ambiguity.risk == "high" and not ambiguity.resolved
            for ambiguity in self.ambiguities
        )


@dataclass(frozen=True)
class AmbiguityAxis:
    """One NL-supported ambiguity axis considered before IR freeze."""

    axis_id: str
    nl_span: str = ""
    possible_readings: tuple[str, ...] = ()
    why_ambiguous: str = ""
    required_trace_roles: tuple[str, ...] = ()
    review_status: str = "pending"
    source_segment_ids: tuple[str, ...] = ()
    plain_reading_decision: str = ""
    review_basis: str = ""


@dataclass(frozen=True)
class AmbiguityAxisSet:
    """Finite set of NL ambiguity axes tracked for a requirement."""

    requirement_id: str
    axes: tuple[AmbiguityAxis, ...] = ()
    source: str = "natural_language"


@dataclass(frozen=True)
class CandidateIRBranch:
    """A bounded, NL-supported semantic interpretation before freeze."""

    ir_branch_id: str
    faithful_ir: FaithfulSemanticIR
    supported_text_spans: tuple[str, ...] = ()
    ambiguity_axis_ids: tuple[str, ...] = ()
    semantic_assumptions: tuple[str, ...] = ()
    plain_language_paraphrase: str = ""
    generation_reason: str = ""
    applied_alternative_ids: tuple[str, ...] = ()
    status: str = "proposed"
    policy_profile_id: str = ""
    policy_profile_version: str = ""
    controlled_parameter_paths: tuple[str, ...] = ()
    structure_fingerprint: str = ""
    policy_target_binding_id: str = ""


@dataclass(frozen=True)
class CandidateIRSet:
    """Candidate IR branches that may be reviewed, repaired, or rejected."""

    requirement_id: str
    nl_text: str
    branches: tuple[CandidateIRBranch, ...] = ()
    ambiguity_axes: AmbiguityAxisSet | None = None
    max_branch_count: int = 5
    status: str = "proposed"
    segmentation_id: str = ""
    reading_review_id: str = ""
    ambiguity_debt: tuple[str, ...] = ()
    branch_set_id: str = ""
    common_structure_fingerprint: str = ""
    policy_axis_ids: tuple[str, ...] = ()
    selection_status: str = "unresolved"
    selection_basis: str = ""
    selected_branch_id: str = ""
    candidate_catalog_version: str = ""
    policy_target_binding_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class IRCandidateSemanticScore:
    """NL-trajectory agreement for one complete CandidateIRBranch."""

    branch_id: str
    matched_count: int = 0
    compared_count: int = 0
    agreement: float = 0.0
    validation_passed: bool = True
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class IRAmbiguityAdjudicationRound:
    """One candidate-blind NL trajectory adjudication round."""

    round_index: int
    query_ids: tuple[str, ...] = ()
    accepted_query_ids: tuple[str, ...] = ()
    unresolved_query_ids: tuple[str, ...] = ()
    candidate_scores: tuple[IRCandidateSemanticScore, ...] = ()
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class IRAmbiguityAdjudicationReport:
    """Automated resolution of complete IR branches from NL-only trace labels."""

    adjudication_id: str
    requirement_id: str
    candidate_set_id: str
    reading_review_id: str
    status: str
    selected_branch_id: str = ""
    rounds: tuple[IRAmbiguityAdjudicationRound, ...] = ()
    supporting_query_ids: tuple[str, ...] = ()
    unresolved_ambiguity_ids: tuple[str, ...] = ()
    selection_basis: str = ""
    provisional: bool = False
    authoritative: bool = False
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {
            "not_required",
            "unique_resolved",
            "provisional_fallback",
            "no_valid_candidate",
            "incomplete",
        }:
            raise ValueError("unsupported IR ambiguity adjudication status")
        if self.provisional and self.authoritative:
            raise ValueError("provisional IR adjudication cannot be authoritative")


@dataclass(frozen=True)
class NLTemporalSimulationReview:
    """NL-only finite-trace review result used as trace truth source."""

    review_id: str
    trace_id: str
    nl_text: str
    trace_steps: tuple[tuple[str, ...], ...] = ()
    review_basis: str = "NL_only"
    temporal_simulation_summary: str = ""
    nl_verdict: str = "unclear"
    reason: str = ""
    supported_text_spans: tuple[str, ...] = ()
    reviewer_id: str = ""
    review_time: str = ""
    confidence: str = "medium"
    query_role: str = ""
    target_ambiguity: str = ""
    obligation_ids: tuple[str, ...] = ()
    ir_version: str = ""
    trace_fingerprint: str = ""
    data_split: str = "train"
    label_source: str = "warmup_train_gold"
    formula_source: str = "none"
    formula_visibility: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.data_split not in DATA_SPLITS:
            raise ValueError("Unsupported data_split: " + self.data_split)
        if self.label_source not in LABEL_SOURCES:
            raise ValueError("Unsupported label_source: " + self.label_source)
        if self.formula_source not in FORMULA_SOURCES:
            raise ValueError("Unsupported formula_source: " + self.formula_source)
        if self.formula_visibility:
            raise ValueError(
                "NLTemporalSimulationReview must remain formula-blind at runtime."
            )


@dataclass(frozen=True)
class LabeledTraceEvidenceSet:
    """Accepted NL-labeled evidence under the current IR obligation plan."""

    requirement_id: str
    ir_version: str = ""
    trace_ids: tuple[str, ...] = ()
    positive_trace_ids: tuple[str, ...] = ()
    negative_trace_ids: tuple[str, ...] = ()
    boundary_trace_ids: tuple[str, ...] = ()
    unclear_trace_ids: tuple[str, ...] = ()
    review_ids: tuple[str, ...] = ()
    status: str = "pending"
    evidence_debt: tuple[str, ...] = ()
    evidence_contract_version: int = 0


@dataclass(frozen=True)
class SemanticEvidenceGateReport:
    """Single source of truth for semantic-domain evidence completeness."""

    requirement_id: str
    ir_version: str
    obligation_plan_version: str
    evidence_contract_version: int
    selected_branch_id: str = ""
    required_obligation_ids: tuple[str, ...] = ()
    covered_obligation_ids: tuple[str, ...] = ()
    missing_obligation_ids: tuple[str, ...] = ()
    required_boundary_obligation_ids: tuple[str, ...] = ()
    covered_boundary_obligation_ids: tuple[str, ...] = ()
    trace_ids: tuple[str, ...] = ()
    trace_evidence_fingerprint: str = ""
    trusted_annotation_count: int = 0
    conflicting_trace_ids: tuple[str, ...] = ()
    remaining_ambiguity_debt: tuple[str, ...] = ()
    status: str = "incomplete"
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"complete", "incomplete", "rejected"}:
            raise ValueError("unsupported semantic evidence gate status")

    @property
    def complete(self) -> bool:
        return self.status == "complete"


@dataclass(frozen=True)
class ApprovedSemanticIR:
    """Frozen semantic source of truth after NL-labeled trace review."""

    ir_version: str
    approved_ir_branch_id: str
    nl_text: str
    approved_ap_mapping: tuple[tuple[str, str], ...] = ()
    resolved_ambiguity_axes: tuple[str, ...] = ()
    remaining_ambiguity_debt: tuple[str, ...] = ()
    labeled_trace_evidence_ids: tuple[str, ...] = ()
    approval_record: str = ""
    approved: bool = False
    evidence_status: str = "pending"
    ambiguity_closure_id: str = ""
    evidence_contract_version: int = 0
    evidence_gate_fingerprint: str = ""


@dataclass(frozen=True)
class TraceObligation:
    obligation_id: str
    ir_node_ids: tuple[str, ...]
    required_trace_role: str
    construction_constraints: tuple[str, ...] = ()
    expected_relation: str = "distinguish"
    minimum_horizon: int = 1
    boundary_case: str = ""
    semantic_parameters: tuple[tuple[str, Any], ...] = ()
    semantic_parameter_paths: tuple[str, ...] = ()
    ambiguity_axis_ids: tuple[str, ...] = ()
    evidence_mode: str = "root_label"
    feasibility_report_hash: str = ""
    gating: bool = False


@dataclass(frozen=True)
class IRRelationFeasibilityReport:
    """Formula-free, bounded realizability classification of the IR root."""

    ir_version: str
    ap_context_fingerprint: str
    horizon_minimum: int
    horizon_maximum: int
    satisfy_status: str
    violate_status: str
    proof_scope: str
    classification: str
    diagnostics: tuple[tuple[str, Any], ...] = ()
    source_supports_non_contingent: bool = False
    solver_contract_version: str = IR_WITNESS_SOLVER_CONTRACT_VERSION
    contract_version: str = IR_RELATION_FEASIBILITY_CONTRACT_VERSION
    artifact_hash: str = ""

    def __post_init__(self) -> None:
        import hashlib
        import json

        if self.horizon_minimum < 1 or self.horizon_maximum < self.horizon_minimum:
            raise ValueError("invalid relation feasibility horizon domain")
        if self.classification not in {
            "contingent_in_declared_domain",
            "valid_in_declared_domain",
            "unsatisfiable_in_declared_domain",
            "unknown_due_to_resource_limit",
            "unsupported_semantics",
        }:
            raise ValueError("unsupported relation feasibility classification")
        payload = {
            "ir_version": self.ir_version,
            "ap_context_fingerprint": self.ap_context_fingerprint,
            "horizon_minimum": self.horizon_minimum,
            "horizon_maximum": self.horizon_maximum,
            "satisfy_status": self.satisfy_status,
            "violate_status": self.violate_status,
            "proof_scope": self.proof_scope,
            "classification": self.classification,
            "diagnostics": self.diagnostics,
            "source_supports_non_contingent": self.source_supports_non_contingent,
            "solver_contract_version": self.solver_contract_version,
            "contract_version": self.contract_version,
        }
        computed = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if self.artifact_hash != computed:
            object.__setattr__(self, "artifact_hash", computed)


@dataclass(frozen=True)
class TraceObligationPlan:
    requirement_id: str
    ir_version: str
    obligations: tuple[TraceObligation, ...]
    confusion_targets: tuple[str, ...] = ()
    coverage_requirements: tuple[str, ...] = ()
    plan_purpose: str = "structural_analysis"
    readiness_fingerprint: str = ""
    execution_blockers: tuple[str, ...] = ()
    ap_context_fingerprint: str = ""
    feasibility_report_hash: str = ""
    feasibility_classification: str = ""
    relation_feasibility: IRRelationFeasibilityReport | None = None
    plan_version: str = DERIVED_PLAN_VERSION


@dataclass(frozen=True)
class SkeletonPlanNode:
    skeleton_node_id: str
    ir_node_id: str
    parent_id: str = ""
    children: tuple[str, ...] = ()
    node_type: str = ""
    fixed: bool = True
    allowed_operators: tuple[str, ...] = ()
    scope_boundary: bool = False


@dataclass(frozen=True)
class ConstrainedSkeletonPlan:
    requirement_id: str
    ir_version: str
    root_skeleton_node_id: str
    nodes: tuple[SkeletonPlanNode, ...]
    fixed_ir_node_ids: tuple[str, ...] = ()
    plan_version: str = DERIVED_PLAN_VERSION


@dataclass(frozen=True)
class CompletionHoleConstraint:
    hole_id: str
    ir_node_id: str
    skeleton_node_id: str
    allowed_operators: tuple[str, ...]
    parent_hole_id: str = ""
    scope_locked: bool = True


@dataclass(frozen=True)
class CompletionHolePlan:
    requirement_id: str
    ir_version: str
    skeleton_plan_version: str
    holes: tuple[CompletionHoleConstraint, ...]
    plan_version: str = DERIVED_PLAN_VERSION


@dataclass(frozen=True)
class IRCoverageReport:
    """Independent gate between consistency scoring and final selection."""

    requirement_id: str = ""
    ir_version: str = ""
    status: str = "incomplete"
    covered_ir_node_ids: tuple[str, ...] = ()
    missing_ir_node_ids: tuple[str, ...] = ()
    covered_obligation_ids: tuple[str, ...] = ()
    missing_obligation_ids: tuple[str, ...] = ()
    covered_confusions: tuple[str, ...] = ()
    missing_confusions: tuple[str, ...] = ()
    structure_mapping_complete: bool = False
    version_consistent: bool = False
    reasons: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.status == "complete"


@dataclass(frozen=True)
class HoleOperatorResolution:
    """Trajectory evidence for one temporal completion hole."""

    hole_id: str
    ir_node_id: str
    allowed_operators: tuple[str, ...] = ()
    selected_operator: str = ""
    rejected_operators: tuple[str, ...] = ()
    supporting_trace_ids: tuple[str, ...] = ()
    rejection_trace_ids: tuple[str, ...] = ()
    resolution_basis: str = ""
    status: str = "unresolved"


@dataclass(frozen=True)
class OperatorResolutionReport:
    """Whether reviewed whole-trace labels uniquely resolve temporal holes."""

    requirement_id: str = ""
    ir_version: str = ""
    status: str = "not_required"
    hole_resolutions: tuple[HoleOperatorResolution, ...] = ()
    passing_assignment_count: int = 0
    candidate_assignment_count: int = 0
    evaluated_assignment_count: int = 0
    expected_assignment_count: int = 0
    passing_assignments: tuple[tuple[tuple[str, str], ...], ...] = ()
    resolved_ir_node_ids: tuple[str, ...] = ()
    unresolved_ir_node_ids: tuple[str, ...] = ()
    supporting_trace_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    evidence_purpose: str = TRACE_PURPOSE_OPERATOR_DISCRIMINATION
    hole_plan_version: str = ""
    parent_ir_status: str = "approved"
    authoritative: bool = False

    @property
    def complete(self) -> bool:
        return self.status in {"complete", "not_required"}


@dataclass(frozen=True)
class CandidateSelectionOutcome:
    selected: FormulaConsistencyEvaluation | None
    status: str
    automatic_finalization_allowed: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class FinalApprovalRecord:
    """Explicit approval metadata for turning a candidate into a final result.

    Automatic finalization does not need this record.  It is required whenever
    a candidate was produced by a feedback rerun or by any path that cannot
    automatically satisfy the IR coverage/status gate.
    """

    approval_id: str
    reviewer: str
    approved_at: str
    approved_ir_version: str
    approved_candidate_id: str
    approved_formula: str
    approval_basis: tuple[str, ...] = ()
    ir_coverage_status: str = ""
    consistency_status: str = ""
    writeback_allowed: bool = False
    approval_status: str = "final"
    independence_id: str = ""
    source_run_id: str = ""
    rerun_delta_id: str = ""
    formula_closure_status: str = ""
    context_isolated: bool = False
    gold_exposed: bool = False


@dataclass(frozen=True)
class FaithfulLearningExample:
    requirement_id: str
    natural_language: str
    approved_ir: FaithfulSemanticIR
    trace_obligation_plan: TraceObligationPlan
    approved_traces: tuple[dict[str, Any], ...]
    skeleton_plan: ConstrainedSkeletonPlan
    approved_skeletons: tuple[str, ...]
    completion_hole_plan: CompletionHolePlan
    approved_completion: str
    final_formula: str
    approval_metadata: dict[str, Any]
    surface_segmentation: SurfaceSegmentation | None = None
    plain_reading_review: PlainReadingReview | None = None


@dataclass(frozen=True)
class CorrectionMemoryRecord:
    correction_id: str
    failed_ir_node_ids: tuple[str, ...]
    old_ir_fragment: dict[str, Any]
    corrected_ir_fragment: dict[str, Any]
    affected_artifacts: tuple[str, ...]
    correction_reason: str
    approval_status: str


@dataclass(frozen=True)
class ReviewPendingRecord:
    """A recoverable pause that awaits evidence, review, or a scoped retry."""

    reason_code: str
    owner_stage: str
    safe_restart_stage: str
    requirement_id: str
    source_ir_version: str = ""
    affected_ir_node_ids: tuple[str, ...] = ()
    dispute_ids: tuple[str, ...] = ()
    challenge_ids: tuple[str, ...] = ()
    evidence_record_paths: tuple[str, ...] = ()
    required_action: str = "manual_review"
    retry_policy: str = "retry_after_required_action"
    contract_version: str = BATCH_DISPOSITION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.reason_code or not self.requirement_id:
            raise ValueError("review pending record requires reason_code and requirement_id")


@dataclass(frozen=True)
class BlockedFailureRecord:
    """A true execution failure, distinct from normal reviewable incompleteness."""

    reason_code: str
    owner_stage: str
    safe_restart_stage: str
    requirement_id: str
    error_type: str
    message: str
    source_ir_version: str = ""
    evidence_record_paths: tuple[str, ...] = ()
    retryable: bool = False
    contract_version: str = BATCH_DISPOSITION_CONTRACT_VERSION


@dataclass(frozen=True)
class PreIRDisposition:
    """Typed status at the pre-IR/batch boundary."""

    requirement_id: str
    disposition: str
    review_pending: ReviewPendingRecord | None = None
    blocked_failure: BlockedFailureRecord | None = None
    contract_version: str = BATCH_DISPOSITION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.disposition not in {"approved", "review_pending", "blocked_error"}:
            raise ValueError("unsupported pre-IR disposition")
        if self.disposition == "review_pending" and self.review_pending is None:
            raise ValueError("review_pending disposition requires its typed record")
        if self.disposition == "blocked_error" and self.blocked_failure is None:
            raise ValueError("blocked_error disposition requires its typed record")
        if self.disposition == "approved" and (
            self.review_pending is not None or self.blocked_failure is not None
        ):
            raise ValueError("approved disposition cannot carry failure records")


@dataclass(frozen=True)
class BatchRunOutcome:
    """Independent uniqueness and completeness dimensions for one batch run."""

    expected_count: int
    result_count: int
    final_count: int
    pending_count: int
    blocked_count: int
    uniqueness_ok: bool
    complete: bool
    release_ready: bool
    missing_requirement_ids: tuple[str, ...] = ()
    duplicate_requirement_ids: tuple[str, ...] = ()
    item_statuses: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class FeedbackRerunRequest:
    requirement_id: str
    owner_stage: str = "pipeline"
    restart_stage: str = ""
    reason_code: str = "RERUN_REQUESTED"
    source_ir_version: str = ""
    source_artifact_fingerprints: dict[str, str] | None = None
    affected_ir_node_ids: tuple[str, ...] = ()
    dispute_ids: tuple[str, ...] = ()
    challenge_ids: tuple[str, ...] = ()
    reviewed_nl_evidence: tuple[str, ...] = ()
    expected_invalidation_set: tuple[str, ...] = ()
    restart_prerequisites: tuple[str, ...] = ()
    fresh_approval_requirement: str = "required"
    feedback: str = ""
    schema_version: int = FEEDBACK_RERUN_SCHEMA_VERSION
    # Compatibility aliases for in-memory callers. Persisted v2 packets use
    # the typed fields above and these are normalized to the same values.
    start_stage: str = ""
    constraints: tuple[str, ...] = ()
    previous_ir_version: str = ""
    action: str = "rerun_with_feedback"
    output_policy: str = "fresh_independent_approval"
    ir_global_problem: bool = False
    diagnosis_id: str = ""
    source_run_id: str = ""
    expected_changed_artifacts: tuple[str, ...] = ()
    source_fingerprints: dict[str, str] | None = None
    fresh_judge_required: bool = True
    feedback_patch: dict[str, Any] | None = None
    frozen_upstream: bool = False

    def __post_init__(self) -> None:
        restart_stage = self.restart_stage or self.start_stage
        if not restart_stage:
            restart_stage = "pipeline"
        source_ir_version = self.source_ir_version or self.previous_ir_version
        source_fingerprints = (
            self.source_artifact_fingerprints
            if self.source_artifact_fingerprints is not None
            else self.source_fingerprints
        )
        invalidation_set = (
            self.expected_invalidation_set or self.expected_changed_artifacts
        )
        object.__setattr__(self, "restart_stage", restart_stage)
        object.__setattr__(self, "start_stage", restart_stage)
        object.__setattr__(self, "source_ir_version", source_ir_version)
        object.__setattr__(self, "previous_ir_version", source_ir_version)
        object.__setattr__(self, "source_artifact_fingerprints", source_fingerprints)
        object.__setattr__(self, "source_fingerprints", source_fingerprints)
        object.__setattr__(self, "expected_invalidation_set", invalidation_set)
        object.__setattr__(self, "expected_changed_artifacts", invalidation_set)
        if self.schema_version != FEEDBACK_RERUN_SCHEMA_VERSION:
            raise ValueError("unsupported feedback rerun schema_version")
        if self.action != "rerun_with_feedback":
            raise ValueError("feedback action must be rerun_with_feedback")
        if self.output_policy not in {
            "provisional_only",
            "fresh_independent_approval",
        }:
            raise ValueError("unsupported feedback output_policy")
        if restart_stage not in {
            "pipeline",
            "segmentation",
            "plain_reading",
            "semantic",
            "trace",
            "skeleton",
            "completion",
            "selection",
        }:
            raise ValueError("unsupported feedback start_stage")
        if not self.owner_stage or not self.reason_code:
            raise ValueError("feedback requires owner_stage and reason_code")
        if self.fresh_approval_requirement not in {"required", "not_required"}:
            raise ValueError("unsupported fresh_approval_requirement")
        prerequisites = {item.casefold() for item in self.restart_prerequisites}
        required_prerequisites = {
            "plain_reading": {"approved_segmentation"},
            "semantic": {"approved_plain_reading"},
            "trace": {"approved_semantic_ir"},
            "skeleton": {"approved_semantic_ir", "approved_trace_evidence"},
            "completion": {"approved_skeleton"},
            "selection": {"approved_completion"},
        }.get(restart_stage, set())
        missing_prerequisites = required_prerequisites - prerequisites
        if missing_prerequisites:
            raise ValueError(
                f"{restart_stage} rerun missing prerequisites: "
                + ", ".join(sorted(missing_prerequisites))
            )
        if (
            restart_stage not in {"segmentation", "plain_reading"}
            and not self.affected_ir_node_ids
            and not self.ir_global_problem
        ):
            raise ValueError("feedback must bind IR nodes or be an IR-global problem")
        if self.feedback_patch is not None:
            if not isinstance(self.feedback_patch, dict):
                raise ValueError("feedback_patch must be a mapping")
            patch = self.feedback_patch
            if str(patch.get("requirement_id", "")) != self.requirement_id:
                raise ValueError("feedback_patch requirement_id mismatch")
            if str(patch.get("rerun_start_stage", "")) != restart_stage:
                raise ValueError("feedback_patch start_stage mismatch")
            if self.diagnosis_id and str(patch.get("diagnosis_id", "")) != self.diagnosis_id:
                raise ValueError("feedback_patch diagnosis_id mismatch")
            patch_ir = str(patch.get("source_ir_version", ""))
            if source_ir_version and patch_ir != source_ir_version:
                raise ValueError("feedback_patch IR version mismatch")


@dataclass(frozen=True)
class WarmupLearningConfig:
    """Strict configuration for warmup/sample learning."""

    warmup_file: str
    warmup_only: bool
    reference_formula: str = ""
    data_split: str = "train"
    memory_mode: str = "read_write"
    learning_mode: str = "stage_guided"
    offline_validation: str = "none"

    def __post_init__(self) -> None:
        if not self.warmup_file:
            raise ValueError("warmup learning requires --warmup-file.")
        if not self.warmup_only:
            raise ValueError("warmup learning requires --warmup-only.")
        if self.learning_mode not in LEARNING_MODES:
            raise ValueError("Unsupported learning_mode: " + self.learning_mode)
        if self.offline_validation not in OFFLINE_VALIDATION_MODES:
            raise ValueError(
                "Unsupported offline_validation: " + self.offline_validation
            )
        if (
            self.learning_mode == "legacy_formula_oracle"
            and self.reference_formula.strip().lower() != "sample"
        ):
            raise ValueError(
                "legacy_formula_oracle warmup requires --reference-formula sample."
            )
        if (
            self.learning_mode != "legacy_formula_oracle"
            and self.reference_formula.strip()
        ):
            raise ValueError(
                "--reference-formula is only available for legacy_formula_oracle; "
                "use --offline-validation final_formula for stage-guided checks."
            )
        if self.data_split != "train":
            raise ValueError("warmup learning requires data_split=train.")
        if self.memory_mode not in WARMUP_MEMORY_MODES:
            raise ValueError("Unsupported warmup memory_mode: " + self.memory_mode)


@dataclass(frozen=True)
class GuidedCorrectionRecord:
    """One autonomous mismatch followed by one teacher-guided attempt."""

    stage: str
    round_1_artifact: Any
    expected_artifact: Any
    mismatch_evidence: tuple[str, ...]
    root_cause_category: str
    root_cause_explanation: str
    correction_action: str
    round_2_artifact: Any
    verification_status: str = "passed"


@dataclass(frozen=True)
class StageErrorRecord:
    """
    一个 pipeline 阶段中出现的一次错误。

    该结构用于把人工评审失败、自动诊断失败、候选过滤失败和运行异常统一
    写入运行期 RAG。每个 StageErrorRecord 对应一次可检索的错误记忆，
    下一轮同阶段 Agent 可以用它避免重复犯同类错误。
    """

    error_id: str
    stage: str
    round_index: int
    artifact: str
    error_type: str
    source: str
    evidence: tuple[str, ...] = ()
    suggestion: str = ""
    comments: str = ""
    timestamp: str = ""


@dataclass(frozen=True)
class TraceCoverageReport:
    """Finite-trace coverage status carried across the full pipeline."""

    status: str = "complete"
    required_count: int = 0
    actual_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    boundary_count: int = 0
    distinct_boundary_count: int = 0
    missing: tuple[str, ...] = ()
    coverage_debt: tuple[str, ...] = ()
    retry_count: int = 0
    executable_semantics: bool = False
    reason: str = ""
    quota_complete: bool = True
    semantic_complete: bool = True
    covered_roles: tuple[str, ...] = ()
    missing_roles: tuple[str, ...] = ()
    covered_confusions: tuple[str, ...] = ()
    missing_confusions: tuple[str, ...] = ()
    validated_trace_count: int = 0

    @property
    def complete(self) -> bool:
        return self.status == "complete" and self.semantic_complete

    @property
    def degraded(self) -> bool:
        return self.status in {"degraded_unproven", "degraded_proven"}

    @property
    def permits_automatic_selection(self) -> bool:
        return self.status == "complete"


EXPERT_REVIEW_STATUSES = (
    "pending",
    "approved",
    "rejected",
    "retry_requested",
)


@dataclass(frozen=True)
class DegradedResultRecord:
    """Structured expert-review package for a degraded pipeline result."""

    requirement_id: str
    requirement: str
    semantic: dict[str, Any]
    trace_coverage: dict[str, Any]
    coverage_debt: tuple[str, ...]
    retry_count: int
    executable_semantics: bool
    all_traces: tuple[dict[str, Any], ...]
    skeletons: tuple[str, ...]
    candidates: tuple[str, ...]
    evaluations: tuple[dict[str, Any], ...]
    provisional_candidate: dict[str, Any] | None
    exact_formula_classification: str
    selection_status: str
    persistent_rag_written: bool = False
    review_records: tuple[dict[str, Any], ...] = ()
    expert_review_status: str = "pending"
    expert_comments: str = ""
    recommended_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.expert_review_status not in EXPERT_REVIEW_STATUSES:
            raise ValueError(
                "expert_review_status must be one of: "
                + ", ".join(EXPERT_REVIEW_STATUSES)
            )


SEMANTIC_CHALLENGE_STATUSES = (
    "pending",
    "confirmed_existing_decision",
    "confirmed_new_axis",
    "rejected_as_annotation_uncertainty",
    "needs_source_clarification",
    "resolved",
    "stale",
)


@dataclass(frozen=True)
class TraceAdjudicationBrief:
    """Natural-language-only context supplied to a trace reviewer.

    IR node parameters and compiler decisions are intentionally absent.  They
    are compared with the resulting annotation only after the reviewer has
    produced an independent source-based judgment.
    """

    brief_id: str
    semantic_context_id: str
    requirement_id: str
    source_text: str
    source_hash: str
    ir_version: str
    reading_review_id: str = ""
    decision_fingerprint: str = ""
    resolution_id: str = ""
    resolution_fingerprint: str = ""
    trace_domain: str = "ltlf"
    ap_definitions: tuple[tuple[str, str], ...] = ()
    reviewed_plain_language_decisions: tuple[str, ...] = ()
    open_plain_language_axes: tuple[str, ...] = ()
    global_trace_semantics: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticChallenge:
    """A pending semantic concern discovered during trace adjudication.

    Challenges are deliberately external to ``FaithfulSemanticIR``.  They may
    block approval, but they never make the last validated IR structurally
    invalid while a reviewer is still deciding what the source means.
    """

    challenge_id: str
    requirement_id: str
    base_ir_version: str
    brief_id: str
    query_id: str
    challenge_type: str
    affected_node_ids: tuple[str, ...] = ()
    affected_parameter_paths: tuple[str, ...] = ()
    related_axis_ids: tuple[str, ...] = ()
    proposed_readings: tuple[str, ...] = ()
    supported_text_spans: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    annotation_label: str = "UNCLEAR"
    annotation_confidence: str = "medium"
    status: str = "pending"
    replacement_ir_version: str = ""

    def __post_init__(self) -> None:
        if self.status not in SEMANTIC_CHALLENGE_STATUSES:
            raise ValueError("Unsupported semantic challenge status: " + self.status)
        if self.challenge_type not in {
            "new_semantic_axis",
            "reviewed_decision_conflict",
        }:
            raise ValueError(
                "Unsupported semantic challenge type: " + self.challenge_type
            )
        if len(self.proposed_readings) < 2:
            raise ValueError(
                "A semantic challenge requires at least two proposed readings."
            )


@dataclass(frozen=True)
class SemanticChallengeSet:
    requirement_id: str
    base_ir_version: str
    challenges: tuple[SemanticChallenge, ...] = ()
    status: str = "open"
    contract_version: int = 1

    @property
    def open_challenges(self) -> tuple[SemanticChallenge, ...]:
        return tuple(
            challenge
            for challenge in self.challenges
            if challenge.status
            in {"pending", "confirmed_new_axis", "needs_source_clarification"}
        )


@dataclass(frozen=True)
class SemanticStructure:
    """
    自然语言理解阶段得到的语义结构。

    trigger:
        触发事件，例如 request。

    response:
        响应事件，例如 grant。

    pattern:
        需求所属的时序模式。当前原型主要支持 response 模式。

    explanation:
        Agent 给出的自然语言解释，方便人工评审。

    rag_context:
        RAG 检索到的关键知识片段摘要。

    aps:
        从完整公式或人工样例中确认的全部 AP 名称。非 response 模式优先使用
        该字段做语义硬过滤。

    formula_text:
        旧版兼容字段。NLU 阶段不再生成完整 LTLf 公式；response 主流程也不
        使用该字段绕过受限骨架和 H_xxx 孔位补全。warmup/历史记录中若仍
        带有该字段，只作为外部产物保留。

    components:
        模式特定的语义字段，例如 conjunction 的 left_component /
        right_component，until 的 left_condition / until_witness。

    trace_domain:
        NLU 后的轨迹语义域判别。ltlf 表示有限轨迹语义；ltl_infinite 表示
        原始自然语言更像无限轨迹 LTL 性质；ambiguous 表示需要人工确认。

    finite_rewrite:
        当 trace_domain 为 ltl_infinite 时，系统用于继续有限轨迹流程的
        LTLf 近似自然语言改写。

    reference_only:
        True 表示最终 LTLf 结果只是对原无限轨迹 LTL 需求的有限化参考。

    pattern_status:
        known 表示当前模式已由内置逻辑、扩展中间模板或 warmup 学习支持；
        unknown 表示 LLM 识别到了尚未注册的模式，不能跳过有限轨迹证据链
        直接进入候选公式。

    original_pattern:
        unknown 模式下记录 LLM 原始识别出的模式名，便于最终结果显式标注。

    persistent_rag_allowed:
        False 表示该语义结构及其后续运行产物不得写入持久性 RAG。
    """

    trigger: str
    response: str
    temporal_keyword: str
    scope_keyword: str
    pattern: str
    explanation: str = ""
    confidence: float = 0.0
    rag_context: tuple[str, ...] = ()
    aps: tuple[str, ...] = ()
    formula_text: str = ""
    components: tuple[tuple[str, str], ...] = ()
    trace_domain: str = "ltlf"
    finite_rewrite: str = ""
    domain_warning: str = ""
    domain_evidence: tuple[str, ...] = ()
    reference_only: bool = False
    pattern_status: str = "known"
    original_pattern: str = ""
    persistent_rag_allowed: bool = True
    data_split: str = "test"
    label_source: str = "human"
    formula_source: str = "none"
    formula_visibility: tuple[str, ...] = ()
    allowed_rag_stages: tuple[str, ...] = (
        "semantic",
        "obligation_planning",
        "trace_query_generation",
        "skeleton",
        "completion",
        "selection",
    )
    faithful_ir: FaithfulSemanticIR | None = None
    semantic_status: str = "ready"
    candidate_ir_set: CandidateIRSet | None = None
    approved_semantic_ir: ApprovedSemanticIR | None = None
    labeled_trace_evidence: LabeledTraceEvidenceSet | None = None
    semantic_evidence_gate: SemanticEvidenceGateReport | None = None
    surface_segmentation: SurfaceSegmentation | None = None
    plain_reading_review: PlainReadingReview | None = None
    semantic_policy_resolutions: tuple[dict[str, Any], ...] = ()
    source_axis_debt_resolutions: tuple[dict[str, Any], ...] = ()
    semantic_policy_target_bindings: tuple[dict[str, Any], ...] = ()
    semantic_resolution: SemanticResolutionRecord | None = None
    semantic_challenges: SemanticChallengeSet | None = None
    ir_ambiguity_adjudication: IRAmbiguityAdjudicationReport | None = None
    semantic_source_contract_version: str = ""
    effective_source_fingerprint: str = ""
    effective_source_kind: str = ""
    effective_source_authority_record_id: str = ""

    def __post_init__(self) -> None:
        if self.data_split not in DATA_SPLITS:
            raise ValueError("Unsupported data_split: " + self.data_split)
        if self.label_source not in LABEL_SOURCES:
            raise ValueError("Unsupported label_source: " + self.label_source)
        if self.formula_source not in FORMULA_SOURCES:
            raise ValueError("Unsupported formula_source: " + self.formula_source)
        if self.semantic_status not in {"ready", "needs_semantic_review"}:
            raise ValueError("Unsupported semantic_status: " + self.semantic_status)


TRACE_ADJUDICATION_LABELS = (
    "pending",
    "SAT",
    "UNSAT",
    "UNCLEAR",
    "INFEASIBLE",
)
TRACE_ANNOTATION_CONFIDENCES = ("high", "medium", "low")
EXPERIMENT_MODES = ("human", "independent_judge", "formula_oracle")
DATA_SPLITS = ("train", "validation", "test")
LABEL_SOURCES = (
    "human",
    "model_assistant",
    "independent_judge",
    "frozen_independent_judge",
    "formula_oracle",
    "warmup_fixture",
    "warmup_train_gold",
)
FORMULA_SOURCES = (
    "human_approved",
    "dataset_reference",
    "generated",
    "none",
)


@dataclass(frozen=True)
class SemanticObligationPlan:
    """
    Formula-free finite-trace query plan derived from reviewed semantics.

    The plan records semantic roles, ambiguity axes, and query coverage needs.
    It intentionally has no Formula/formula_text/expected field.
    """

    pattern_family: str
    participants: tuple[tuple[str, str], ...] = ()
    explicit_cues: tuple[tuple[str, str], ...] = ()
    ambiguity_axes: tuple[str, ...] = ()
    required_query_roles: tuple[str, ...] = ()
    trace_constraints: tuple[str, ...] = ()
    domain_constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceQuery:
    """An unlabeled finite-trace question submitted for independent review."""

    query_id: str
    role: str
    target_ambiguity: str
    constraints: tuple[str, ...]
    trace: list[set[str]]
    proposed_label: None = None
    adjudicated_label: str = "pending"
    reason: str = ""
    beginner_view: str = ""
    data_split: str = "test"
    label_source: str = "human"
    formula_source: str = "none"
    formula_visibility: tuple[str, ...] = ()
    allowed_rag_stages: tuple[str, ...] = (
        "obligation_planning",
        "trace_query_generation",
    )
    ir_node_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    ir_version: str = ""
    target_ambiguity_axis_ids: tuple[str, ...] = ()
    semantic_parameter_paths: tuple[str, ...] = ()
    adjudication_brief_id: str = ""
    source_ir_branch_id: str = ""
    generation_source: str = "nl_anchored_query"
    purpose: str = TRACE_PURPOSE_SEMANTIC_VALIDATION
    parent_candidate_set_id: str = ""
    parent_hole_plan_version: str = ""

    def __post_init__(self) -> None:
        if self.proposed_label is not None:
            raise ValueError("TraceQuery.proposed_label must remain null.")
        if self.adjudicated_label != "pending":
            raise ValueError(
                "A TraceQuery must stay pending until a TraceAnnotation is applied."
            )
        if not self.trace:
            raise ValueError("TraceQuery.trace must contain at least one time step.")
        if self.data_split not in DATA_SPLITS:
            raise ValueError("Unsupported data_split: " + self.data_split)
        if self.label_source not in LABEL_SOURCES:
            raise ValueError("Unsupported label_source: " + self.label_source)
        if self.formula_source not in FORMULA_SOURCES:
            raise ValueError("Unsupported formula_source: " + self.formula_source)
        if self.formula_visibility:
            raise ValueError(
                "Unlabeled trace queries must not expose formula visibility."
            )
        if self.purpose not in TRACE_EVIDENCE_PURPOSES:
            raise ValueError("Unsupported trace query purpose: " + self.purpose)
        if (
            self.purpose == TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION
            and not self.parent_candidate_set_id
        ):
            raise ValueError(
                "IR branch discrimination queries require parent_candidate_set_id"
            )
        if (
            self.purpose == TRACE_PURPOSE_OPERATOR_DISCRIMINATION
            and not self.ir_version
        ):
            raise ValueError(
                "operator discrimination queries require an IR version"
            )


@dataclass(frozen=True)
class TraceAnnotation:
    """Independent natural-language adjudication for one TraceQuery."""

    query_id: str
    label: str
    confidence: str
    rationale: str
    label_source: str = "human"
    annotator: str = ""
    model: str = ""
    prompt_version: str = ""
    temporal_simulation_summary: str = ""
    supported_text_spans: tuple[str, ...] = ()
    uncertainty_kind: str = "none"
    candidate_readings: tuple[str, ...] = ()
    challenged_axis_ids: tuple[str, ...] = ()
    affected_parameter_paths: tuple[str, ...] = ()
    purpose: str = TRACE_PURPOSE_SEMANTIC_VALIDATION

    def __post_init__(self) -> None:
        if self.label not in TRACE_ADJUDICATION_LABELS[1:]:
            raise ValueError(
                "TraceAnnotation.label must be SAT, UNSAT, UNCLEAR, or INFEASIBLE."
            )
        if self.confidence not in TRACE_ANNOTATION_CONFIDENCES:
            raise ValueError("Unsupported annotation confidence: " + self.confidence)
        if self.label_source not in LABEL_SOURCES:
            raise ValueError("Unsupported label_source: " + self.label_source)
        if not self.rationale.strip():
            raise ValueError("TraceAnnotation.rationale must not be empty.")
        if self.uncertainty_kind not in {
            "none",
            "annotation_uncertainty",
            "query_infeasible",
            "reviewed_decision_conflict",
            "new_semantic_axis",
        }:
            raise ValueError(
                "Unsupported annotation uncertainty kind: " + self.uncertainty_kind
            )
        if self.purpose not in TRACE_EVIDENCE_PURPOSES:
            raise ValueError("Unsupported trace annotation purpose: " + self.purpose)


@dataclass(frozen=True)
class ExperimentConfig:
    """Reference-formula visibility policy for one pipeline run."""

    mode: str = "human"
    data_split: str = "test"
    reference_formula: Formula | None = None
    test_time_memory_frozen: bool = True

    def __post_init__(self) -> None:
        if self.mode not in EXPERIMENT_MODES:
            raise ValueError("Unsupported experiment mode: " + self.mode)
        if self.data_split not in DATA_SPLITS:
            raise ValueError("Unsupported data_split: " + self.data_split)
        if self.mode != "formula_oracle" and self.reference_formula is not None:
            raise ValueError(
                "reference_formula is inaccessible outside formula_oracle mode."
            )
        if self.mode == "formula_oracle" and self.reference_formula is None:
            raise ValueError(
                "formula_oracle mode requires an explicit reference_formula."
            )


BOUNDARY_TRACE_POLICIES = ("generated", "disabled", "frozen_file")


@dataclass(frozen=True)
class TraceEvidencePolicy:
    """Orthogonal policy for the source of boundary trace evidence.

    ExperimentConfig controls who assigns labels.  This policy controls only
    whether boundary queries are generated, omitted for an ablation, or loaded
    from a frozen formula-blind fixture.
    """

    boundary_source: str = "generated"
    boundary_fixture_path: str = ""
    boundary_fixture_sha256: str = ""
    boundary_fixture_schema_version: int = 0

    def __post_init__(self) -> None:
        if self.boundary_source not in BOUNDARY_TRACE_POLICIES:
            raise ValueError(
                "Unsupported boundary trace policy: " + self.boundary_source
            )
        if self.boundary_source == "frozen_file":
            if not self.boundary_fixture_path:
                raise ValueError(
                    "frozen_file boundary policy requires a fixture path"
                )
            if not self.boundary_fixture_sha256:
                raise ValueError(
                    "frozen_file boundary policy requires a fixture SHA-256"
                )
            if self.boundary_fixture_schema_version <= 0:
                raise ValueError(
                    "frozen_file boundary policy requires a positive schema version"
                )
        elif (
            self.boundary_fixture_path
            or self.boundary_fixture_sha256
            or self.boundary_fixture_schema_version
        ):
            raise ValueError(
                "boundary fixture metadata is only valid for frozen_file policy"
            )

    @property
    def boundary_enabled(self) -> bool:
        return self.boundary_source != "disabled"

    @property
    def experimental_only(self) -> bool:
        return self.boundary_source != "generated"


@dataclass(frozen=True)
class SkeletonDiagnostic:
    """
    受限骨架结构检查结果。

    validator 先检查结构合法性：语法已解析、叶子元素只来自 NLU 抽取 AP、
    不使用 true/false，且不包含受限阶段禁用的 X/F/G。批量校验还会根据
    正/反轨迹做一层保守粗过滤，排除明显不贴近 NL response 方向的骨架。
    """

    skeleton: Formula
    passed: bool
    error_type: str
    trace_evidence: tuple[str, ...] = ()
    suggestion: str = ""


@dataclass(frozen=True)
class TemporalHole:
    """
    候选补全阶段中的一个时序算子孔位。

    hole_id:
        孔位编号，例如 H_scope 或 H_response。

    target:
        孔位作用的公式片段说明。

    allowed_ops:
        该孔位允许选择的算子。epsilon 表示不插入时序算子。

    role:
        孔位的结构角色，例如 root_scope、subscope、response 或 subresponse。

    parent_hole_id:
        嵌套孔位的直接父孔位；顶层孔位为空。

    source_hint:
        从非法提前具体化模板中恢复出的原始 X/F/G，仅作为补全提示。
    """

    hole_id: str
    target: str
    allowed_ops: tuple[str, ...]
    reason: str = ""
    role: str = ""
    parent_hole_id: str = ""
    source_hint: tuple[str, ...] = ()
    ir_node_id: str = ""
    ir_version: str = ""


@dataclass(frozen=True)
class HoleCandidate:
    """
    从受限骨架扩展出的带孔 LTLf 候选。

    这一层仍不直接决定 X/F/G，而是显式暴露可补全槽位，供后续根据
    边界轨迹、正轨迹和反轨迹进行补全。
    """

    skeleton: Formula
    template_text: str
    holes: tuple[TemporalHole, ...]
    rag_context: tuple[str, ...] = ()
    explanation: str = ""
    candidate_id: str = ""


@dataclass(frozen=True)
class TemporalCompletion:
    """
    Completion Agent 对一个带孔候选做出的纯补全结果。

    它保存补全后的公式、孔位赋值和来源带孔模板；轨迹一致性评分由
    TrajectoryConsistencyAgent 在后续阶段统一计算。
    """

    formula: Formula
    assignments: tuple[tuple[str, str], ...]
    operator_derivation: tuple[str, ...] = ()
    candidate_id: str = ""
    template_text: str = ""
    ir_version: str = ""
    closure_report: Any = None


@dataclass(frozen=True)
class TraceSimulationStep:
    """
    TrajectoryConsistencyAgent 模拟公式在有限轨迹某个位置上的一步解释。

    该结构记录的是 Agent 的语义模拟过程，而不是只保存最终 True/False。
    """

    time_index: int
    formula_fragment: str
    result: bool
    explanation: str


@dataclass(frozen=True)
class TraceSimulationResult:
    """
    一个候选公式在一条人工审核轨迹上的模拟结果。

    actual 是 Agent 按 LTLf 有限轨迹语义模拟得到的标签；expected 是人工
    审核标签；matched 表示两者是否一致。
    """

    formula: Formula
    trace_name: str
    trace_kind: str
    expected: bool
    actual: bool
    matched: bool
    steps: tuple[TraceSimulationStep, ...] = ()
    failure_reason: str = ""


@dataclass(frozen=True)
class FormulaConsistencyEvaluation:
    """
    一个完整候选 LTLf 公式在全部人工审核轨迹上的一致性评估。

    PosSat、NegRej 和 BoundMatch 都由 TrajectoryConsistencyAgent 逐条模拟
    有限轨迹后聚合得到。
    """

    formula: Formula
    pos_sat: float
    neg_rej: float
    bound_match: float
    score: float
    trace_results: tuple[TraceSimulationResult, ...] = ()
    failure_summary: tuple[str, ...] = ()
    candidate_id: str = ""
    ir_version: str = ""
    closure_report: Any = None
    boundary_metric_available: bool = True
    boundary_trace_count: int = 0

    @property
    def positive_pass(self) -> bool:
        return self.pos_sat == 1.0

    @property
    def negative_pass(self) -> bool:
        return self.neg_rej == 1.0

    @property
    def boundary_pass(self) -> bool:
        return self.bound_match == 1.0

    @property
    def per_trace_results(self) -> tuple[TraceSimulationResult, ...]:
        return self.trace_results


@dataclass(frozen=True)
class CompletionDecision:
    """
    一个完整候选公式的补全决策记录。

    operator_derivation 记录每个 H_xxx 孔位为何选择当前算子；trace_derivation
    记录正轨迹、反轨迹和边界轨迹的统一模拟结果。补全阶段保留所有候选，
    这些字段只用于解释和后验诊断。
    """

    formula: Formula
    assignments: tuple[tuple[str, str], ...]
    boundary_match: float
    pos_sat: float
    neg_rej: float
    score: float
    operator_derivation: tuple[str, ...] = ()
    trace_derivation: tuple[str, ...] = ()
    reason: str = ""
    candidate_id: str = ""
    template_text: str = ""
    ir_version: str = ""
    closure_report: Any = None
    boundary_metric_available: bool = True
    boundary_trace_count: int = 0
