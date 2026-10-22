"""Versioned, formula-blind AP identity artifacts for semantic replacement.

This module deliberately lives before FaithfulSemanticIR.  It establishes only
the identity alphabet and exact source evidence; it never decides Boolean or
temporal meaning and is not an NLU input.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from contract_versions import (
    AP_INVENTORY_CANONICALIZER_VERSION,
    AP_INVENTORY_CONTRACT_VERSION,
    AP_INVENTORY_SCHEMA_VERSION,
    AP_INVENTORY_VALIDATOR_VERSION,
    AP_REPRESENTATION_PROFILE_CONTRACT_VERSION,
    OVERRIDE_AP_COMPARISON_CONTRACT_VERSION,
)
from schemas import (
    APCatalogEntry,
    APIdentity,
    APIdentityMapping,
    APInventory,
    APMention,
    APRepresentationProfile,
    DeclaredAPCatalog,
    OverrideAPComparison,
    SurfaceSegmentation,
)


V1_SYMBOLIC_TOKEN_PATTERN = r"(?<!['’])\b(?:prop_\d+|[a-z])\b"
LEGACY_CONTROLLED_SYMBOLIC_TOKEN_PATTERN = (
    r"(?<!['’])\b(?:prop_\d+|[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|[a-z])\b"
)
PROP_PLACEHOLDER_TOKEN_PATTERN = (
    r"(?<![A-Za-z0-9_])prop_\d+(?![A-Za-z0-9_])"
)
SINGLE_LOWER_LETTER_TOKEN_PATTERN = (
    r"(?<![A-Za-z0-9_'’])(?:[a-z])(?![A-Za-z0-9_'’])"
)
CONTROLLED_SYMBOLIC_TOKEN_PATTERN = SINGLE_LOWER_LETTER_TOKEN_PATTERN
SYMBOLIC_GRAMMAR_PATTERNS = {
    "prop_placeholder_v1": PROP_PLACEHOLDER_TOKEN_PATTERN,
    "single_lower_letter_v1": SINGLE_LOWER_LETTER_TOKEN_PATTERN,
    "legacy_symbolic_union_v1": LEGACY_CONTROLLED_SYMBOLIC_TOKEN_PATTERN,
}
SYMBOLIC_EXACT_EXTRACTOR_VERSION = "symbolic-exact-v1"
DECLARED_CATALOG_EXTRACTOR_VERSION = "declared-catalog-v1"
GROUNDED_PHRASE_EXTRACTOR_VERSION = "grounded-phrase-v1"
MANUAL_MAPPING_EXTRACTOR_VERSION = "manual-mapping-v1"
OFFLINE_FORMULA_EXTRACTOR_VERSION = "offline-formula-alphabet-v1"

AP_PROFILE_REQUIRED = "HUMAN_OVERRIDE_AP_PROFILE_REQUIRED"
AP_INVENTORY_REQUIRED = "HUMAN_OVERRIDE_AP_INVENTORY_REQUIRED"
AP_INVENTORY_UNAPPROVED = "HUMAN_OVERRIDE_AP_INVENTORY_UNAPPROVED"
AP_INVENTORY_STALE = "HUMAN_OVERRIDE_AP_INVENTORY_STALE"
AP_EVIDENCE_INVALID = "HUMAN_OVERRIDE_AP_EVIDENCE_INVALID"
AP_MAPPING_REQUIRED = "HUMAN_OVERRIDE_AP_MAPPING_REQUIRED"
AP_MAPPING_INCOMPLETE = "HUMAN_OVERRIDE_AP_MAPPING_INCOMPLETE"
AP_MAPPING_NON_BIJECTIVE = "HUMAN_OVERRIDE_AP_MAPPING_NON_BIJECTIVE"
AP_MAPPING_CONFLICT = "HUMAN_OVERRIDE_AP_MAPPING_CONFLICT"
AP_ADDS = "HUMAN_OVERRIDE_ADDS_AP"
AP_OMITS = "HUMAN_OVERRIDE_OMITS_AP"
AP_RENAMES = "HUMAN_OVERRIDE_RENAMES_AP"
AP_COMPARISON_STALE = "HUMAN_OVERRIDE_AP_COMPARISON_STALE"
AP_CONTRACT_UNSUPPORTED = "HUMAN_OVERRIDE_AP_CONTRACT_UNSUPPORTED"
AP_ARTIFACT_CORRUPT = "HUMAN_OVERRIDE_AP_ARTIFACT_CORRUPT"
FORMULA_PROVENANCE_FORBIDDEN = "HUMAN_OVERRIDE_FORMULA_PROVENANCE_FORBIDDEN"


class APIdentityValidationError(ValueError):
    """Stable fail-closed AP artifact validation error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def source_text_hash(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def controlled_symbolic_profile(
    *,
    profile_id: str = "AP-PROFILE-controlled-symbolic-v1",
    provenance: str = "explicit_runtime_profile",
    language: str = "symbolic",
    domain: str = "synthetic",
    symbolic_grammar_id: str = "single_lower_letter_v1",
) -> APRepresentationProfile:
    if symbolic_grammar_id not in SYMBOLIC_GRAMMAR_PATTERNS:
        raise ValueError(f"unsupported symbolic grammar: {symbolic_grammar_id}")
    return APRepresentationProfile(
        profile_id=profile_id,
        mode="controlled_symbolic",
        symbolic_token_pattern=SYMBOLIC_GRAMMAR_PATTERNS[symbolic_grammar_id],
        symbolic_grammar_id=symbolic_grammar_id,
        requires_human_grounding=False,
        language=language,
        domain=domain,
        provenance=provenance,
    )


def undetermined_profile(
    *,
    profile_id: str = "AP-PROFILE-undetermined-v1",
    provenance: str = "implicit_untagged",
    language: str = "",
    domain: str = "",
) -> APRepresentationProfile:
    return APRepresentationProfile(
        profile_id=profile_id,
        mode="undetermined",
        requires_human_grounding=True,
        language=language,
        domain=domain,
        provenance=provenance,
    )


def natural_language_profile(
    *,
    profile_id: str = "AP-PROFILE-natural-language-v1",
    provenance: str = "explicit_runtime_profile",
    language: str = "",
    domain: str = "",
) -> APRepresentationProfile:
    return APRepresentationProfile(
        profile_id=profile_id,
        mode="natural_language_grounded",
        requires_human_grounding=True,
        language=language,
        domain=domain,
        provenance=provenance,
    )


def declared_catalog_profile(
    catalog: DeclaredAPCatalog,
    *,
    profile_id: str = "",
    provenance: str = "declared_catalog_metadata",
    language: str = "",
    domain: str = "",
) -> APRepresentationProfile:
    validate_declared_ap_catalog(catalog)
    return APRepresentationProfile(
        profile_id=(
            profile_id
            or "AP-PROFILE-catalog-"
            + _digest(catalog.catalog_id + "\0" + catalog.catalog_hash, 12)
        ),
        mode="declared_catalog",
        declared_catalog_id=catalog.catalog_id,
        requires_human_grounding=False,
        language=language,
        domain=domain,
        provenance=provenance,
    )


def offline_formula_assisted_profile(
    *,
    profile_id: str = "AP-PROFILE-offline-formula-assisted-v1",
    provenance: str = "explicit_offline_adjudication",
    language: str = "",
    domain: str = "",
) -> APRepresentationProfile:
    return APRepresentationProfile(
        profile_id=profile_id,
        mode="offline_formula_assisted",
        allow_formula_assisted_catalog=True,
        requires_human_grounding=True,
        language=language,
        domain=domain,
        provenance=provenance,
    )


def create_declared_ap_catalog(
    *,
    catalog_id: str,
    catalog_version: str,
    entries: Sequence[APCatalogEntry | Mapping[str, Any]],
    provenance: str,
    source: str = "declared_catalog",
) -> DeclaredAPCatalog:
    normalized = tuple(_catalog_entry(item) for item in entries)
    payload = {
        "catalog_id": catalog_id,
        "catalog_version": catalog_version,
        "entries": [asdict(item) for item in normalized],
        "provenance": provenance,
        "source": source,
    }
    catalog = DeclaredAPCatalog(
        catalog_id=catalog_id,
        catalog_version=catalog_version,
        entries=normalized,
        catalog_hash=_hash_payload(payload),
        provenance=provenance,
        source=source,
    )
    validate_declared_ap_catalog(catalog)
    return catalog


def validate_declared_ap_catalog(catalog: DeclaredAPCatalog) -> None:
    expected = _hash_payload(
        {
            "catalog_id": catalog.catalog_id,
            "catalog_version": catalog.catalog_version,
            "entries": [asdict(item) for item in catalog.entries],
            "provenance": catalog.provenance,
            "source": catalog.source,
        }
    )
    if catalog.catalog_hash != expected:
        raise APIdentityValidationError(
            AP_ARTIFACT_CORRUPT,
            "declared AP catalog hash mismatch",
        )


def v1_symbolic_tokens(value: str) -> tuple[str, ...]:
    """Frozen runtime reader.  Never redirect this to a newer extractor."""

    return tuple(
        sorted(
            set(
                match.group(0).casefold()
                for match in re.finditer(
                    V1_SYMBOLIC_TOKEN_PATTERN,
                    value,
                    flags=re.IGNORECASE,
                )
            )
        )
    )


def build_symbolic_ap_inventory(
    *,
    requirement_id: str,
    source_text: str,
    segmentation: SurfaceSegmentation | None = None,
    profile: APRepresentationProfile | None = None,
    approved_empty: bool = False,
    approved_empty_reason: str = "",
    reviewer_id: str = "",
    reviewer_source: str = "deterministic_symbolic_catalog",
    parent_inventory_id: str = "",
) -> APInventory:
    profile = profile or controlled_symbolic_profile()
    if profile.mode != "controlled_symbolic":
        raise APIdentityValidationError(
            AP_CONTRACT_UNSUPPORTED,
            "symbolic_exact_v1 requires controlled_symbolic profile",
        )
    pattern = (
        SYMBOLIC_GRAMMAR_PATTERNS.get(profile.symbolic_grammar_id)
        or profile.symbolic_token_pattern
        or V1_SYMBOLIC_TOKEN_PATTERN
    )
    matches = list(re.finditer(pattern, source_text, flags=re.IGNORECASE))
    grouped: dict[str, list[re.Match[str]]] = {}
    for match in matches:
        grouped.setdefault(match.group(0).casefold(), []).append(match)
    ordered_names = sorted(grouped, key=lambda name: grouped[name][0].start())
    identities: list[APIdentity] = []
    mentions: list[APMention] = []
    source_hash = source_text_hash(source_text)
    for ordinal, name in enumerate(ordered_names, start=1):
        identity_id = _source_identity_id(
            requirement_id,
            source_hash,
            ordinal,
            profile.profile_id,
        )
        identities.append(
            APIdentity(
                ap_identity_id=identity_id,
                canonical_name=name,
                ordinal=ordinal,
                identity_origin="controlled_symbolic",
            )
        )
        for occurrence_index, match in enumerate(grouped[name], start=1):
            mentions.append(
                _mention(
                    identity_id=identity_id,
                    evidence_text=source_text[match.start() : match.end()],
                    span=(match.start(), match.end()),
                    occurrence_index=occurrence_index,
                    segmentation=segmentation,
                    provenance="symbolic_exact_v1",
                )
            )
    if not identities and not approved_empty:
        status = "pending_human_review"
        issues = (AP_INVENTORY_REQUIRED,)
    else:
        status = "approved"
        issues = ()
    inventory = APInventory(
        inventory_id="pending",
        requirement_id=requirement_id,
        source_text=source_text,
        source_hash=source_hash,
        segmentation_id=segmentation.segmentation_id if segmentation else "",
        representation_profile_id=profile.profile_id,
        extraction_mode="symbolic_exact_v1",
        extractor_version=SYMBOLIC_EXACT_EXTRACTOR_VERSION,
        status=status,
        identities=tuple(identities),
        mentions=tuple(mentions),
        reviewer_role=("ordinary_reviewer" if approved_empty else "system"),
        reviewer_id=reviewer_id,
        reviewer_source=reviewer_source,
        provenance=profile.provenance,
        parent_inventory_id=parent_inventory_id,
        approved_empty=approved_empty,
        approved_empty_reason=approved_empty_reason,
        issue_codes=issues,
        created_at=_timestamp(),
        confirmed_at=_timestamp() if status == "approved" else "",
    )
    inventory = _finalize_inventory(inventory)
    validate_ap_inventory_integrity(
        inventory,
        profile=profile,
        segmentation=segmentation,
        allow_unapproved=status != "approved",
    )
    return inventory


def build_declared_catalog_ap_inventory(
    *,
    requirement_id: str,
    source_text: str,
    catalog: DeclaredAPCatalog,
    profile: APRepresentationProfile,
    segmentation: SurfaceSegmentation | None = None,
    parent_inventory_id: str = "",
) -> APInventory:
    validate_declared_ap_catalog(catalog)
    if (
        profile.mode != "declared_catalog"
        or profile.declared_catalog_id != catalog.catalog_id
    ):
        raise APIdentityValidationError(
            AP_CONTRACT_UNSUPPORTED,
            "declared catalog does not match representation profile",
        )
    identities: list[APIdentity] = []
    mentions: list[APMention] = []
    missing: list[str] = []
    for ordinal, entry in enumerate(catalog.entries, start=1):
        identity_id = "APID-CATALOG-" + _digest(
            catalog.catalog_id
            + "\0"
            + catalog.catalog_version
            + "\0"
            + entry.catalog_ap_id,
            20,
        )
        identities.append(
            APIdentity(
                ap_identity_id=identity_id,
                canonical_name=entry.canonical_name,
                ordinal=ordinal,
                aliases=entry.aliases,
                identity_origin=(
                    "declared_symbolic" if entry.symbolic_declared else "declared_catalog"
                ),
            )
        )
        spans = _alias_spans(
            source_text,
            (entry.canonical_name, *entry.aliases),
        )
        if not spans and not entry.symbolic_declared:
            missing.append(entry.catalog_ap_id)
        for occurrence_index, (start, end) in enumerate(spans, start=1):
            mentions.append(
                _mention(
                    identity_id=identity_id,
                    evidence_text=source_text[start:end],
                    span=(start, end),
                    occurrence_index=occurrence_index,
                    segmentation=segmentation,
                    provenance="declared_catalog_v1",
                )
            )
    status = "approved" if not missing else "pending_human_review"
    inventory = APInventory(
        inventory_id="pending",
        requirement_id=requirement_id,
        source_text=source_text,
        source_hash=source_text_hash(source_text),
        segmentation_id=segmentation.segmentation_id if segmentation else "",
        representation_profile_id=profile.profile_id,
        extraction_mode="declared_catalog_v1",
        extractor_version=DECLARED_CATALOG_EXTRACTOR_VERSION,
        status=status,
        identities=tuple(identities),
        mentions=tuple(mentions),
        reviewer_role="system",
        reviewer_source="declared_catalog_validator",
        provenance=catalog.provenance,
        parent_inventory_id=parent_inventory_id,
        catalog_source=catalog.catalog_id,
        issue_codes=((AP_EVIDENCE_INVALID,) if missing else ()),
        created_at=_timestamp(),
        confirmed_at=_timestamp() if status == "approved" else "",
    )
    inventory = _finalize_inventory(inventory)
    validate_ap_inventory_integrity(
        inventory,
        profile=profile,
        segmentation=segmentation,
        allow_unapproved=True,
    )
    return inventory


def propose_grounded_ap_inventory(
    *,
    requirement_id: str,
    source_text: str,
    profile: APRepresentationProfile,
    proposals: Sequence[Mapping[str, Any]],
    segmentation: SurfaceSegmentation | None = None,
    extraction_mode: str = "grounded_phrase_v1",
    proposal_source: str = "human_or_llm_proposal",
    parent_inventory_id: str = "",
) -> APInventory:
    if profile.mode not in {
        "natural_language_grounded",
        "undetermined",
        "offline_formula_assisted",
    }:
        raise APIdentityValidationError(
            AP_CONTRACT_UNSUPPORTED,
            "grounded phrase proposal requires a phrase-grounded profile",
        )
    identities: list[APIdentity] = []
    mentions: list[APMention] = []
    issues: list[str] = []
    normalized_proposals = sorted(
        (_normalized_proposal(item) for item in proposals),
        key=lambda item: _proposal_first_span(source_text, item),
    )
    source_hash = source_text_hash(source_text)
    for ordinal, proposal in enumerate(normalized_proposals, start=1):
        canonical_name = proposal["canonical_name"]
        spans: list[tuple[int, int]] = []
        for evidence in proposal["evidence_texts"]:
            if _pure_non_predicate_fragment(evidence):
                issues.append(AP_EVIDENCE_INVALID)
                continue
            spans.extend(_exact_spans(source_text, evidence))
        spans = sorted(set(spans))
        if not spans:
            issues.append(AP_EVIDENCE_INVALID)
            continue
        identity_id = "APTMP-" + _digest(
            requirement_id
            + "\0"
            + source_hash
            + "\0"
            + str(ordinal)
            + "\0"
            + AP_INVENTORY_CONTRACT_VERSION,
            20,
        )
        identities.append(
            APIdentity(
                ap_identity_id=identity_id,
                canonical_name=canonical_name,
                ordinal=ordinal,
                aliases=proposal["aliases"],
                identity_origin="grounded_phrase_proposal",
            )
        )
        for occurrence_index, (start, end) in enumerate(spans, start=1):
            mentions.append(
                _mention(
                    identity_id=identity_id,
                    evidence_text=source_text[start:end],
                    span=(start, end),
                    occurrence_index=occurrence_index,
                    segmentation=segmentation,
                    provenance=proposal_source,
                )
            )
    if not identities:
        issues.append(
            AP_PROFILE_REQUIRED if profile.mode == "undetermined" else AP_INVENTORY_REQUIRED
        )
    inventory = APInventory(
        inventory_id="pending",
        requirement_id=requirement_id,
        source_text=source_text,
        source_hash=source_hash,
        segmentation_id=segmentation.segmentation_id if segmentation else "",
        representation_profile_id=profile.profile_id,
        extraction_mode=extraction_mode,
        extractor_version=(
            OFFLINE_FORMULA_EXTRACTOR_VERSION
            if profile.mode == "offline_formula_assisted"
            else GROUNDED_PHRASE_EXTRACTOR_VERSION
        ),
        status="pending_human_review",
        identities=tuple(identities),
        mentions=tuple(mentions),
        reviewer_role="system",
        reviewer_source=proposal_source,
        provenance=profile.provenance,
        parent_inventory_id=parent_inventory_id,
        eligible_for_formula_blind_evaluation=(
            profile.mode != "offline_formula_assisted"
        ),
        persistent_rag_allowed=False,
        catalog_source=(
            "gold_formula_assisted"
            if profile.mode == "offline_formula_assisted"
            else ""
        ),
        issue_codes=tuple(dict.fromkeys(issues or (AP_INVENTORY_UNAPPROVED,))),
        created_at=_timestamp(),
    )
    inventory = _finalize_inventory(inventory)
    validate_ap_inventory_integrity(
        inventory,
        profile=profile,
        segmentation=segmentation,
        allow_unapproved=True,
    )
    return inventory


def approve_ap_inventory(
    inventory: APInventory,
    *,
    profile: APRepresentationProfile,
    reviewer_id: str,
    reviewer_source: str = "human",
    approved_empty: bool = False,
    approved_empty_reason: str = "",
    segmentation: SurfaceSegmentation | None = None,
) -> APInventory:
    validate_ap_inventory_integrity(
        inventory,
        profile=profile,
        segmentation=segmentation,
        allow_unapproved=True,
    )
    if inventory.status not in {"proposed", "pending_human_review"}:
        raise APIdentityValidationError(
            AP_INVENTORY_UNAPPROVED,
            "only a pending AP inventory can be approved",
        )
    if not reviewer_id:
        raise APIdentityValidationError(
            AP_INVENTORY_UNAPPROVED,
            "AP inventory approval requires an explicit reviewer",
        )
    if not inventory.identities and not approved_empty:
        raise APIdentityValidationError(
            AP_INVENTORY_REQUIRED,
            "unknown empty AP inventory cannot be approved as equality evidence",
        )
    approved = replace(
        inventory,
        status="approved",
        reviewer_role="ordinary_reviewer",
        reviewer_id=reviewer_id,
        reviewer_source=reviewer_source,
        approved_empty=approved_empty,
        approved_empty_reason=approved_empty_reason,
        issue_codes=(),
        confirmed_at=_timestamp(),
        artifact_hash="",
    )
    approved = _stabilize_initial_identity_ids(approved)
    approved = _finalize_inventory(approved)
    validate_ap_inventory_integrity(
        approved,
        profile=profile,
        segmentation=segmentation,
    )
    return approved


def compare_override_ap_inventories(
    *,
    override_id: str,
    original_inventory: APInventory,
    replacement_inventory: APInventory,
    identity_mapping: Sequence[APIdentityMapping | Mapping[str, Any]] = (),
    validation_mode: str = "manual_mapping_v1",
    reviewer_role: str = "system",
    reviewer_id: str = "",
    reviewer_source: str = "deterministic",
    allow_formula_assisted: bool = False,
) -> OverrideAPComparison:
    issues: list[str] = []
    mapping = tuple(_identity_mapping(item) for item in identity_mapping)
    if original_inventory.status != "approved" or replacement_inventory.status != "approved":
        issues.append(AP_INVENTORY_UNAPPROVED)
        return _comparison(
            override_id,
            original_inventory,
            replacement_inventory,
            mapping,
            status="needs_inventory",
            issues=issues,
            validation_mode=validation_mode,
            reviewer_role=reviewer_role,
            reviewer_id=reviewer_id,
            reviewer_source=reviewer_source,
        )
    validate_ap_inventory_integrity(original_inventory)
    validate_ap_inventory_integrity(replacement_inventory)
    if (
        not allow_formula_assisted
        and (
            not original_inventory.eligible_for_formula_blind_evaluation
            or not replacement_inventory.eligible_for_formula_blind_evaluation
        )
    ):
        issues.append(FORMULA_PROVENANCE_FORBIDDEN)
        return _comparison(
            override_id,
            original_inventory,
            replacement_inventory,
            mapping,
            status="corrupt",
            issues=issues,
            validation_mode=validation_mode,
            reviewer_role=reviewer_role,
            reviewer_id=reviewer_id,
            reviewer_source=reviewer_source,
        )

    original_ids = {item.ap_identity_id for item in original_inventory.identities}
    replacement_ids = {
        item.ap_identity_id for item in replacement_inventory.identities
    }
    if not original_ids and not replacement_ids:
        if original_inventory.approved_empty and replacement_inventory.approved_empty:
            return _comparison(
                override_id,
                original_inventory,
                replacement_inventory,
                (),
                status="passed",
                issues=(),
                validation_mode="approved_empty_v1",
                reviewer_role=reviewer_role,
                reviewer_id=reviewer_id,
                reviewer_source=reviewer_source,
            )
        issues.append(AP_INVENTORY_REQUIRED)
        return _comparison(
            override_id,
            original_inventory,
            replacement_inventory,
            (),
            status="needs_inventory",
            issues=issues,
            validation_mode=validation_mode,
            reviewer_role=reviewer_role,
            reviewer_id=reviewer_id,
            reviewer_source=reviewer_source,
        )

    if not mapping and _can_exactly_map_by_declared_identity(
        original_inventory, replacement_inventory
    ):
        mapping = tuple(
            APIdentityMapping(
                replacement_ap_identity_id=replacement_id,
                original_ap_identity_id=original_id,
                mapping_provenance="deterministic_declared_identity",
            )
            for replacement_id, original_id in _declared_identity_pairs(
                original_inventory, replacement_inventory
            )
        )
        validation_mode = "declared_identity_exact_v1"
    elif not mapping and _is_symbolic_inventory(original_inventory) and _is_symbolic_inventory(
        replacement_inventory
    ):
        original_by_name = {
            item.canonical_name: item.ap_identity_id
            for item in original_inventory.identities
        }
        replacement_by_name = {
            item.canonical_name: item.ap_identity_id
            for item in replacement_inventory.identities
        }
        omitted_names = set(original_by_name) - set(replacement_by_name)
        added_names = set(replacement_by_name) - set(original_by_name)
        if omitted_names or added_names:
            if omitted_names and added_names and len(original_by_name) == len(replacement_by_name):
                issues.append(AP_RENAMES)
            else:
                if omitted_names:
                    issues.append(AP_OMITS)
                if added_names:
                    issues.append(AP_ADDS)
            return _comparison(
                override_id,
                original_inventory,
                replacement_inventory,
                (),
                status="mismatch",
                issues=issues,
                validation_mode="symbolic_exact_v1",
                reviewer_role=reviewer_role,
                reviewer_id=reviewer_id,
                reviewer_source=reviewer_source,
            )
        mapping = tuple(
            APIdentityMapping(
                replacement_ap_identity_id=replacement_by_name[name],
                original_ap_identity_id=original_by_name[name],
                mapping_provenance="symbolic_exact_v1",
            )
            for name in sorted(original_by_name)
        )
        validation_mode = "symbolic_exact_v1"

    if len(replacement_ids) > len(original_ids):
        issues.append(AP_ADDS)
    if len(replacement_ids) < len(original_ids):
        issues.append(AP_OMITS)
    if issues:
        return _comparison(
            override_id,
            original_inventory,
            replacement_inventory,
            mapping,
            status="mismatch",
            issues=issues,
            validation_mode=validation_mode,
            reviewer_role=reviewer_role,
            reviewer_id=reviewer_id,
            reviewer_source=reviewer_source,
        )

    mapped_replacement = [item.replacement_ap_identity_id for item in mapping]
    mapped_original = [item.original_ap_identity_id for item in mapping]
    if any(item not in replacement_ids for item in mapped_replacement) or any(
        item not in original_ids for item in mapped_original
    ):
        issues.append(AP_MAPPING_CONFLICT)
    if len(mapped_replacement) != len(set(mapped_replacement)) or len(
        mapped_original
    ) != len(set(mapped_original)):
        issues.append(AP_MAPPING_NON_BIJECTIVE)
    unmapped_original = tuple(sorted(original_ids - set(mapped_original)))
    unmapped_replacement = tuple(sorted(replacement_ids - set(mapped_replacement)))
    if unmapped_original or unmapped_replacement:
        issues.append(AP_MAPPING_INCOMPLETE if mapping else AP_MAPPING_REQUIRED)
    status = "needs_mapping_review" if issues else "passed"
    return _comparison(
        override_id,
        original_inventory,
        replacement_inventory,
        mapping,
        status=status,
        issues=issues,
        validation_mode=validation_mode,
        reviewer_role=reviewer_role,
        reviewer_id=reviewer_id,
        reviewer_source=reviewer_source,
        unmapped_original_ids=unmapped_original,
        unmapped_replacement_ids=unmapped_replacement,
    )


def validate_override_ap_comparison_integrity(
    comparison: OverrideAPComparison,
    *,
    original_inventory: APInventory,
    replacement_inventory: APInventory,
) -> None:
    allow_unapproved = comparison.status == "needs_inventory"
    validate_ap_inventory_integrity(
        original_inventory,
        allow_unapproved=allow_unapproved,
    )
    validate_ap_inventory_integrity(
        replacement_inventory,
        allow_unapproved=allow_unapproved,
    )
    errors: list[str] = []
    if comparison.original_inventory_id != original_inventory.inventory_id:
        errors.append("original inventory id mismatch")
    if comparison.original_inventory_hash != original_inventory.artifact_hash:
        errors.append("original inventory hash mismatch")
    if comparison.replacement_inventory_id != replacement_inventory.inventory_id:
        errors.append("replacement inventory id mismatch")
    if comparison.replacement_inventory_hash != replacement_inventory.artifact_hash:
        errors.append("replacement inventory hash mismatch")
    if comparison.artifact_hash != override_ap_comparison_artifact_hash(comparison):
        errors.append("comparison artifact hash mismatch")
    if comparison.status == "passed":
        original_ids = {item.ap_identity_id for item in original_inventory.identities}
        replacement_ids = {
            item.ap_identity_id for item in replacement_inventory.identities
        }
        mapped_original = {
            item.original_ap_identity_id for item in comparison.identity_mapping
        }
        mapped_replacement = {
            item.replacement_ap_identity_id for item in comparison.identity_mapping
        }
        empty_pair = (
            not original_ids
            and not replacement_ids
            and original_inventory.approved_empty
            and replacement_inventory.approved_empty
        )
        if not empty_pair and (
            mapped_original != original_ids or mapped_replacement != replacement_ids
        ):
            errors.append("passed comparison mapping is incomplete")
    if errors:
        raise APIdentityValidationError(
            AP_ARTIFACT_CORRUPT,
            "; ".join(errors),
        )


def promote_replacement_ap_inventory(
    replacement_inventory: APInventory,
    *,
    original_inventory: APInventory,
    comparison: OverrideAPComparison,
    profile: APRepresentationProfile | None = None,
    segmentation: SurfaceSegmentation | None = None,
) -> APInventory:
    validate_override_ap_comparison_integrity(
        comparison,
        original_inventory=original_inventory,
        replacement_inventory=replacement_inventory,
    )
    if comparison.status != "passed":
        raise APIdentityValidationError(
            AP_MAPPING_REQUIRED,
            "replacement inventory promotion requires passed AP comparison",
        )
    mapping = {
        item.replacement_ap_identity_id: item.original_ap_identity_id
        for item in comparison.identity_mapping
    }
    originals = {
        item.ap_identity_id: item for item in original_inventory.identities
    }
    identities: list[APIdentity] = []
    for identity in replacement_inventory.identities:
        inherited_id = mapping.get(identity.ap_identity_id)
        if inherited_id is None:
            raise APIdentityValidationError(
                AP_MAPPING_INCOMPLETE,
                "replacement identity has no inherited original identity",
            )
        original = originals[inherited_id]
        identities.append(
            replace(
                identity,
                ap_identity_id=inherited_id,
                ordinal=original.ordinal,
                identity_origin="replacement_inherited_identity",
            )
        )
    mentions = tuple(
        replace(
            mention,
            ap_identity_id=mapping.get(
                mention.ap_identity_id,
                mention.ap_identity_id,
            ),
            mention_id="pending",
            mapping_provenance="approved_override_mapping",
        )
        for mention in replacement_inventory.mentions
    )
    mentions = tuple(
        replace(
            mention,
            mention_id=_mention_id(
                mention.ap_identity_id,
                mention.source_span,
                mention.occurrence_index,
            ),
        )
        for mention in mentions
    )
    promoted = replace(
        replacement_inventory,
        inventory_id="pending",
        identities=tuple(sorted(identities, key=lambda item: item.ordinal)),
        mentions=mentions,
        parent_inventory_id=original_inventory.inventory_id,
        reviewer_source="approved_override_mapping",
        artifact_hash="",
    )
    promoted = _finalize_inventory(promoted)
    validate_ap_inventory_integrity(
        promoted,
        profile=profile,
        segmentation=segmentation,
    )
    return promoted


def build_offline_formula_assisted_inventory(
    *,
    requirement_id: str,
    source_text: str,
    formula: Any,
    evidence_by_ap: Mapping[str, Sequence[str]],
    profile: APRepresentationProfile,
    segmentation: SurfaceSegmentation | None = None,
) -> APInventory:
    if profile.mode != "offline_formula_assisted" or not profile.allow_formula_assisted_catalog:
        raise APIdentityValidationError(
            FORMULA_PROVENANCE_FORBIDDEN,
            "formula AP catalog requires explicit offline_formula_assisted profile",
        )
    names = sorted(_formula_ap_names(formula))
    proposals = [
        {
            "canonical_name": name,
            "evidence_texts": list(evidence_by_ap.get(name, ())),
        }
        for name in names
    ]
    inventory = propose_grounded_ap_inventory(
        requirement_id=requirement_id,
        source_text=source_text,
        profile=profile,
        proposals=proposals,
        segmentation=segmentation,
        extraction_mode="grounded_phrase_v1",
        proposal_source="gold_formula_assisted",
    )
    formula_hash = hashlib.sha256(str(formula).encode("utf-8")).hexdigest()
    return _finalize_inventory(
        replace(
            inventory,
            extraction_mode="offline_formula_assisted",
            extractor_version=OFFLINE_FORMULA_EXTRACTOR_VERSION,
            formula_source_hash=formula_hash,
            catalog_source="gold_formula_assisted",
            eligible_for_formula_blind_evaluation=False,
            persistent_rag_allowed=False,
            artifact_hash="",
        )
    )


def validate_ap_inventory_integrity(
    inventory: APInventory,
    *,
    profile: APRepresentationProfile | None = None,
    segmentation: SurfaceSegmentation | None = None,
    allow_unapproved: bool = False,
) -> None:
    errors: list[str] = []
    if inventory.source_hash != source_text_hash(inventory.source_text):
        errors.append("source hash mismatch")
    if profile is not None:
        if profile.profile_id != inventory.representation_profile_id:
            errors.append("representation profile mismatch")
        if profile.contract_version != AP_REPRESENTATION_PROFILE_CONTRACT_VERSION:
            errors.append("representation profile contract mismatch")
    if segmentation is not None:
        if segmentation.source_text != inventory.source_text:
            errors.append("segmentation source mismatch")
        if inventory.segmentation_id != segmentation.segmentation_id:
            errors.append("segmentation id mismatch")
    identity_ids = {item.ap_identity_id for item in inventory.identities}
    mention_identity_ids = {item.ap_identity_id for item in inventory.mentions}
    if not mention_identity_ids.issubset(identity_ids):
        errors.append("mention references unknown identity")
    segment_ids = (
        {item.segment_id for item in segmentation.segments}
        if segmentation is not None
        else set()
    )
    for mention in inventory.mentions:
        start, end = mention.source_span
        predicate_start, predicate_end = mention.predicate_span
        if inventory.source_text[start:end] != mention.evidence_text:
            errors.append(f"mention evidence mismatch: {mention.mention_id}")
        if predicate_start < start or predicate_end > end:
            errors.append(f"predicate span outside evidence: {mention.mention_id}")
        if segmentation is not None and not set(
            mention.supporting_segment_ids
        ).issubset(segment_ids):
            errors.append(f"unknown supporting segment: {mention.mention_id}")
    identities_with_mentions = mention_identity_ids
    missing_evidence = {
        item.ap_identity_id
        for item in inventory.identities
        if item.ap_identity_id not in identities_with_mentions
        and item.identity_origin != "declared_symbolic"
    }
    if missing_evidence:
        errors.append("identities lack exact source evidence")
    if not inventory.identities and inventory.status == "approved" and not inventory.approved_empty:
        errors.append("unknown empty inventory cannot be approved")
    if inventory.approved_empty and not inventory.approved_empty_reason:
        errors.append("approved empty inventory lacks reason")
    if inventory.status != "approved" and not allow_unapproved:
        errors.append("inventory is not approved")
    if inventory.extraction_mode == "offline_formula_assisted":
        if inventory.eligible_for_formula_blind_evaluation:
            errors.append("formula-assisted inventory is marked formula-blind eligible")
        if inventory.persistent_rag_allowed:
            errors.append("formula-assisted inventory allows persistent RAG")
        if not inventory.formula_source_hash:
            errors.append("formula-assisted inventory lacks formula source hash")
    if inventory.artifact_hash != ap_inventory_artifact_hash(inventory):
        errors.append("inventory artifact hash mismatch")
    if errors:
        raise APIdentityValidationError(
            AP_ARTIFACT_CORRUPT if inventory.artifact_hash else AP_EVIDENCE_INVALID,
            "; ".join(dict.fromkeys(errors)),
        )


def ap_inventory_artifact_hash(inventory: APInventory) -> str:
    payload = asdict(inventory)
    for key in ("artifact_hash", "created_at", "confirmed_at", "reviewer_id"):
        payload.pop(key, None)
    return _hash_payload(payload)


def override_ap_comparison_artifact_hash(
    comparison: OverrideAPComparison,
) -> str:
    payload = asdict(comparison)
    for key in ("artifact_hash", "confirmed_at", "reviewer_id"):
        payload.pop(key, None)
    return _hash_payload(payload)


def ap_inventory_from_mapping(value: Mapping[str, Any]) -> APInventory:
    return APInventory(
        inventory_id=str(value.get("inventory_id", "")),
        requirement_id=str(value.get("requirement_id", "")),
        source_text=str(value.get("source_text", "")),
        source_hash=str(value.get("source_hash", "")),
        segmentation_id=str(value.get("segmentation_id", "")),
        representation_profile_id=str(value.get("representation_profile_id", "")),
        extraction_mode=str(value.get("extraction_mode", "")),
        extractor_version=str(value.get("extractor_version", "")),
        status=str(value.get("status", "proposed")),
        identities=tuple(_identity(item) for item in value.get("identities", ())),
        mentions=tuple(_mention_from_mapping(item) for item in value.get("mentions", ())),
        reviewer_role=str(value.get("reviewer_role", "system")),
        reviewer_id=str(value.get("reviewer_id", "")),
        reviewer_source=str(value.get("reviewer_source", "deterministic")),
        provenance=str(value.get("provenance", "")),
        parent_inventory_id=str(value.get("parent_inventory_id", "")),
        approved_empty=bool(value.get("approved_empty", False)),
        approved_empty_reason=str(value.get("approved_empty_reason", "")),
        eligible_for_formula_blind_evaluation=bool(
            value.get("eligible_for_formula_blind_evaluation", True)
        ),
        persistent_rag_allowed=bool(value.get("persistent_rag_allowed", False)),
        catalog_source=str(value.get("catalog_source", "")),
        formula_source_hash=str(value.get("formula_source_hash", "")),
        issue_codes=tuple(str(item) for item in value.get("issue_codes", ())),
        schema_version=int(value.get("schema_version", 0)),
        contract_version=str(value.get("contract_version", "")),
        canonicalizer_version=str(value.get("canonicalizer_version", "")),
        validator_version=str(value.get("validator_version", "")),
        artifact_hash=str(value.get("artifact_hash", "")),
        created_at=str(value.get("created_at", "")),
        confirmed_at=str(value.get("confirmed_at", "")),
    )


def override_ap_comparison_from_mapping(
    value: Mapping[str, Any],
) -> OverrideAPComparison:
    return OverrideAPComparison(
        comparison_id=str(value.get("comparison_id", "")),
        override_id=str(value.get("override_id", "")),
        original_inventory_id=str(value.get("original_inventory_id", "")),
        original_inventory_hash=str(value.get("original_inventory_hash", "")),
        replacement_inventory_id=str(value.get("replacement_inventory_id", "")),
        replacement_inventory_hash=str(value.get("replacement_inventory_hash", "")),
        identity_mapping=tuple(
            _identity_mapping(item) for item in value.get("identity_mapping", ())
        ),
        unmapped_original_ids=tuple(
            str(item) for item in value.get("unmapped_original_ids", ())
        ),
        unmapped_replacement_ids=tuple(
            str(item) for item in value.get("unmapped_replacement_ids", ())
        ),
        validation_mode=str(value.get("validation_mode", "")),
        validation_version=str(value.get("validation_version", "")),
        status=str(value.get("status", "needs_inventory")),
        issue_codes=tuple(str(item) for item in value.get("issue_codes", ())),
        reviewer_role=str(value.get("reviewer_role", "system")),
        reviewer_id=str(value.get("reviewer_id", "")),
        reviewer_source=str(value.get("reviewer_source", "deterministic")),
        artifact_hash=str(value.get("artifact_hash", "")),
        confirmed_at=str(value.get("confirmed_at", "")),
    )


def ap_representation_profile_from_mapping(
    value: Mapping[str, Any] | None,
) -> APRepresentationProfile:
    if not isinstance(value, Mapping):
        return undetermined_profile()
    return APRepresentationProfile(
        profile_id=str(value.get("profile_id", "AP-PROFILE-undetermined-v1")),
        mode=str(value.get("mode", "undetermined")),
        symbolic_token_pattern=str(value.get("symbolic_token_pattern", "")),
        symbolic_grammar_id=str(value.get("symbolic_grammar_id", "")),
        declared_catalog_id=str(value.get("declared_catalog_id", "")),
        allow_formula_assisted_catalog=bool(
            value.get("allow_formula_assisted_catalog", False)
        ),
        requires_human_grounding=bool(value.get("requires_human_grounding", True)),
        language=str(value.get("language", "")),
        domain=str(value.get("domain", "")),
        contract_version=str(
            value.get(
                "contract_version",
                AP_REPRESENTATION_PROFILE_CONTRACT_VERSION,
            )
        ),
        provenance=str(value.get("provenance", "implicit_untagged")),
    )


def declared_ap_catalog_from_mapping(value: Mapping[str, Any]) -> DeclaredAPCatalog:
    catalog = DeclaredAPCatalog(
        catalog_id=str(value.get("catalog_id", "")),
        catalog_version=str(value.get("catalog_version", "")),
        entries=tuple(_catalog_entry(item) for item in value.get("entries", ())),
        catalog_hash=str(value.get("catalog_hash", "")),
        provenance=str(value.get("provenance", "")),
        source=str(value.get("source", "declared_catalog")),
    )
    validate_declared_ap_catalog(catalog)
    return catalog


def ap_identity_observability(
    *,
    profile: APRepresentationProfile,
    inventory: APInventory | None = None,
    comparison: OverrideAPComparison | None = None,
    v1_load: bool = False,
    v1_migration: bool = False,
) -> dict[str, int | str]:
    """Minimal structured metric labels; artifacts remain the source of truth."""

    issue_codes = set(comparison.issue_codes if comparison else ())
    return {
        "mode": profile.mode,
        "inventory_required": int(
            inventory is None or AP_INVENTORY_REQUIRED in inventory.issue_codes
        ),
        "mapping_required": int(
            comparison is not None and comparison.status == "needs_mapping_review"
        ),
        "comparison_passed": int(
            comparison is not None and comparison.status == "passed"
        ),
        "known_mismatch": int(
            comparison is not None and comparison.status == "mismatch"
        ),
        "stale_or_corrupt": int(
            comparison is not None and comparison.status in {"stale", "corrupt"}
        ),
        "formula_assisted": int(profile.mode == "offline_formula_assisted"),
        "formula_leak_violation": int(FORMULA_PROVENANCE_FORBIDDEN in issue_codes),
        "v1_load": int(v1_load),
        "v1_migration": int(v1_migration),
    }


def _comparison(
    override_id: str,
    original: APInventory,
    replacement: APInventory,
    mapping: tuple[APIdentityMapping, ...],
    *,
    status: str,
    issues: Iterable[str],
    validation_mode: str,
    reviewer_role: str,
    reviewer_id: str,
    reviewer_source: str,
    unmapped_original_ids: tuple[str, ...] | None = None,
    unmapped_replacement_ids: tuple[str, ...] | None = None,
) -> OverrideAPComparison:
    mapped_original = {item.original_ap_identity_id for item in mapping}
    mapped_replacement = {item.replacement_ap_identity_id for item in mapping}
    unmapped_original_ids = unmapped_original_ids if unmapped_original_ids is not None else tuple(
        sorted(
            {item.ap_identity_id for item in original.identities} - mapped_original
        )
    )
    unmapped_replacement_ids = (
        unmapped_replacement_ids
        if unmapped_replacement_ids is not None
        else tuple(
            sorted(
                {item.ap_identity_id for item in replacement.identities}
                - mapped_replacement
            )
        )
    )
    core = {
        "override_id": override_id,
        "original_inventory_id": original.inventory_id,
        "original_inventory_hash": original.artifact_hash,
        "replacement_inventory_id": replacement.inventory_id,
        "replacement_inventory_hash": replacement.artifact_hash,
        "mapping": [asdict(item) for item in mapping],
        "status": status,
        "issues": list(dict.fromkeys(issues)),
        "validation_mode": validation_mode,
        "validation_version": OVERRIDE_AP_COMPARISON_CONTRACT_VERSION,
        "reviewer_role": reviewer_role,
        "reviewer_source": reviewer_source,
    }
    comparison = OverrideAPComparison(
        comparison_id="APCMP-" + _digest(_canonical_json(core), 20),
        override_id=override_id,
        original_inventory_id=original.inventory_id,
        original_inventory_hash=original.artifact_hash,
        replacement_inventory_id=replacement.inventory_id,
        replacement_inventory_hash=replacement.artifact_hash,
        identity_mapping=mapping,
        unmapped_original_ids=unmapped_original_ids,
        unmapped_replacement_ids=unmapped_replacement_ids,
        validation_mode=validation_mode,
        validation_version=OVERRIDE_AP_COMPARISON_CONTRACT_VERSION,
        status=status,
        issue_codes=tuple(dict.fromkeys(issues)),
        reviewer_role=reviewer_role,
        reviewer_id=reviewer_id,
        reviewer_source=reviewer_source,
        confirmed_at=_timestamp() if status == "passed" else "",
    )
    return replace(
        comparison,
        artifact_hash=override_ap_comparison_artifact_hash(comparison),
    )


def _finalize_inventory(inventory: APInventory) -> APInventory:
    without_identity = replace(inventory, inventory_id="pending", artifact_hash="")
    payload = asdict(without_identity)
    for key in (
        "inventory_id",
        "artifact_hash",
        "created_at",
        "confirmed_at",
        "reviewer_id",
    ):
        payload.pop(key, None)
    inventory_id = "APINV-" + _digest(_canonical_json(payload), 24)
    identified = replace(without_identity, inventory_id=inventory_id)
    return replace(
        identified,
        artifact_hash=ap_inventory_artifact_hash(identified),
    )


def _stabilize_initial_identity_ids(inventory: APInventory) -> APInventory:
    if inventory.parent_inventory_id or inventory.extraction_mode in {
        "symbolic_exact_v1",
        "declared_catalog_v1",
    }:
        return inventory
    id_map: dict[str, str] = {}
    identities: list[APIdentity] = []
    for identity in sorted(inventory.identities, key=lambda item: item.ordinal):
        stable_id = _source_identity_id(
            inventory.requirement_id,
            inventory.source_hash,
            identity.ordinal,
            inventory.representation_profile_id,
        )
        id_map[identity.ap_identity_id] = stable_id
        identities.append(
            replace(
                identity,
                ap_identity_id=stable_id,
                identity_origin="approved_source_inventory",
            )
        )
    mentions = tuple(
        replace(
            mention,
            ap_identity_id=id_map[mention.ap_identity_id],
            mention_id=_mention_id(
                id_map[mention.ap_identity_id],
                mention.source_span,
                mention.occurrence_index,
            ),
        )
        for mention in inventory.mentions
    )
    return replace(
        inventory,
        inventory_id="pending",
        identities=tuple(identities),
        mentions=mentions,
        artifact_hash="",
    )


def _mention(
    *,
    identity_id: str,
    evidence_text: str,
    span: tuple[int, int],
    occurrence_index: int,
    segmentation: SurfaceSegmentation | None,
    provenance: str,
) -> APMention:
    supporting = ()
    if segmentation is not None:
        supporting = tuple(
            segment.segment_id
            for segment in segmentation.segments
            if span[0] < segment.end and span[1] > segment.start
        )
    return APMention(
        mention_id=_mention_id(identity_id, span, occurrence_index),
        ap_identity_id=identity_id,
        evidence_text=evidence_text,
        source_span=span,
        supporting_segment_ids=supporting,
        predicate_span=span,
        occurrence_index=occurrence_index,
        mapping_provenance=provenance,
    )


def _mention_id(
    identity_id: str,
    span: tuple[int, int],
    occurrence_index: int,
) -> str:
    return "APMENTION-" + _digest(
        identity_id
        + "\0"
        + str(span[0])
        + "\0"
        + str(span[1])
        + "\0"
        + str(occurrence_index),
        20,
    )


def _source_identity_id(
    requirement_id: str,
    source_hash: str,
    ordinal: int,
    profile_id: str,
) -> str:
    return "APID-" + requirement_id + "-" + _digest(
        requirement_id
        + "\0"
        + source_hash
        + "\0"
        + str(ordinal)
        + "\0"
        + profile_id
        + "\0"
        + AP_INVENTORY_CONTRACT_VERSION,
        16,
    )


def _can_exactly_map_by_declared_identity(
    original: APInventory,
    replacement: APInventory,
) -> bool:
    if original.extraction_mode != "declared_catalog_v1" or replacement.extraction_mode != "declared_catalog_v1":
        return False
    return {
        item.ap_identity_id for item in original.identities
    } == {item.ap_identity_id for item in replacement.identities}


def _declared_identity_pairs(
    original: APInventory,
    replacement: APInventory,
) -> tuple[tuple[str, str], ...]:
    original_ids = {item.ap_identity_id for item in original.identities}
    return tuple(
        (item.ap_identity_id, item.ap_identity_id)
        for item in replacement.identities
        if item.ap_identity_id in original_ids
    )


def _is_symbolic_inventory(inventory: APInventory) -> bool:
    return inventory.extraction_mode == "symbolic_exact_v1"


def _alias_spans(source_text: str, aliases: Sequence[str]) -> list[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    for alias in aliases:
        spans.update(_exact_spans(source_text, alias))
    return sorted(spans)


def _exact_spans(source_text: str, evidence: str) -> list[tuple[int, int]]:
    return exact_ap_evidence_spans(source_text, evidence)


def exact_ap_evidence_spans(
    source_text: str, evidence: str, *, case_sensitive: bool = False,
) -> list[tuple[int, int]]:
    """Find source fragments without matching inside ASCII identifiers.

    Chinese predicates need no whitespace boundaries. Case-insensitive inventory
    mentions retain their actual source spelling; proposals can demand exact case.
    """
    evidence = evidence.strip()
    if not evidence:
        return []
    def identifier_character(char):
        return char.isascii() and (char.isalnum() or char == "_")

    return [
        (match.start(), match.end())
        for match in re.finditer(re.escape(evidence), source_text,
                                 flags=0 if case_sensitive else re.IGNORECASE)
        if not (match.start() and identifier_character(source_text[match.start() - 1])
                and identifier_character(source_text[match.start()]))
        and not (match.end() < len(source_text) and identifier_character(source_text[match.end() - 1])
                 and identifier_character(source_text[match.end()]))
    ]


def _pure_non_predicate_fragment(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value.strip().casefold())
    return normalized in {
        "not",
        "and",
        "or",
        "when",
        "while",
        "eventually",
        "always",
        "shall",
        "must",
        "only one step",
        "不",
        "并且",
        "或者",
        "最终",
        "始终",
    }


def _normalized_proposal(value: Mapping[str, Any]) -> dict[str, Any]:
    canonical_name = str(value.get("canonical_name", "")).strip()
    evidence = value.get("evidence_texts", value.get("mentions", ()))
    if isinstance(evidence, str):
        evidence = (evidence,)
    evidence_texts = tuple(
        dict.fromkeys(str(item).strip() for item in evidence or () if str(item).strip())
    )
    aliases = value.get("aliases", ())
    if isinstance(aliases, str):
        aliases = (aliases,)
    if not canonical_name:
        raise APIdentityValidationError(
            AP_EVIDENCE_INVALID,
            "grounded AP proposal requires canonical_name",
        )
    return {
        "canonical_name": canonical_name,
        "evidence_texts": evidence_texts,
        "aliases": tuple(str(item).strip() for item in aliases if str(item).strip()),
    }


def _proposal_first_span(source_text: str, proposal: Mapping[str, Any]) -> int:
    starts = [
        spans[0][0]
        for evidence in proposal["evidence_texts"]
        if (spans := _exact_spans(source_text, evidence))
    ]
    return min(starts) if starts else len(source_text) + 1


def _formula_ap_names(formula: Any) -> set[str]:
    names: set[str] = set()
    stack = [formula]
    while stack:
        node = stack.pop()
        if str(getattr(node, "op", "")) == "ap":
            name = str(getattr(node, "name", "")).strip()
            if name:
                names.add(name)
        stack.extend(getattr(node, "args", ()) or ())
    return names


def _catalog_entry(value: APCatalogEntry | Mapping[str, Any]) -> APCatalogEntry:
    if isinstance(value, APCatalogEntry):
        return value
    return APCatalogEntry(
        catalog_ap_id=str(value.get("catalog_ap_id", "")),
        canonical_name=str(value.get("canonical_name", "")),
        aliases=tuple(str(item) for item in value.get("aliases", ())),
        symbolic_declared=bool(value.get("symbolic_declared", False)),
    )


def _identity(value: APIdentity | Mapping[str, Any]) -> APIdentity:
    if isinstance(value, APIdentity):
        return value
    return APIdentity(
        ap_identity_id=str(value.get("ap_identity_id", "")),
        canonical_name=str(value.get("canonical_name", "")),
        ordinal=int(value.get("ordinal", 0)),
        aliases=tuple(str(item) for item in value.get("aliases", ())),
        identity_origin=str(value.get("identity_origin", "source_inventory")),
    )


def _mention_from_mapping(value: APMention | Mapping[str, Any]) -> APMention:
    if isinstance(value, APMention):
        return value
    return APMention(
        mention_id=str(value.get("mention_id", "")),
        ap_identity_id=str(value.get("ap_identity_id", "")),
        evidence_text=str(value.get("evidence_text", "")),
        source_span=tuple(int(item) for item in value.get("source_span", (0, 0))),
        supporting_segment_ids=tuple(
            str(item) for item in value.get("supporting_segment_ids", ())
        ),
        predicate_span=tuple(
            int(item) for item in value.get("predicate_span", (0, 0))
        ),
        occurrence_index=int(value.get("occurrence_index", 0)),
        mapping_provenance=str(value.get("mapping_provenance", "")),
    )


def _identity_mapping(
    value: APIdentityMapping | Mapping[str, Any],
) -> APIdentityMapping:
    if isinstance(value, APIdentityMapping):
        return value
    return APIdentityMapping(
        replacement_ap_identity_id=str(
            value.get("replacement_ap_identity_id", "")
        ),
        original_ap_identity_id=str(value.get("original_ap_identity_id", "")),
        mapping_provenance=str(value.get("mapping_provenance", "human")),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest(value: str, length: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="microseconds")


__all__ = [
    "APIdentityValidationError",
    "AP_ADDS",
    "AP_ARTIFACT_CORRUPT",
    "AP_COMPARISON_STALE",
    "AP_CONTRACT_UNSUPPORTED",
    "AP_EVIDENCE_INVALID",
    "AP_INVENTORY_REQUIRED",
    "AP_INVENTORY_STALE",
    "AP_INVENTORY_UNAPPROVED",
    "AP_MAPPING_CONFLICT",
    "AP_MAPPING_INCOMPLETE",
    "AP_MAPPING_NON_BIJECTIVE",
    "AP_MAPPING_REQUIRED",
    "AP_OMITS",
    "AP_PROFILE_REQUIRED",
    "AP_RENAMES",
    "FORMULA_PROVENANCE_FORBIDDEN",
    "V1_SYMBOLIC_TOKEN_PATTERN",
    "CONTROLLED_SYMBOLIC_TOKEN_PATTERN",
    "ap_identity_observability",
    "ap_inventory_artifact_hash",
    "ap_inventory_from_mapping",
    "ap_representation_profile_from_mapping",
    "approve_ap_inventory",
    "build_declared_catalog_ap_inventory",
    "build_offline_formula_assisted_inventory",
    "build_symbolic_ap_inventory",
    "compare_override_ap_inventories",
    "controlled_symbolic_profile",
    "create_declared_ap_catalog",
    "declared_ap_catalog_from_mapping",
    "declared_catalog_profile",
    "natural_language_profile",
    "offline_formula_assisted_profile",
    "override_ap_comparison_artifact_hash",
    "override_ap_comparison_from_mapping",
    "promote_replacement_ap_inventory",
    "propose_grounded_ap_inventory",
    "source_text_hash",
    "undetermined_profile",
    "v1_symbolic_tokens",
    "validate_ap_inventory_integrity",
    "validate_declared_ap_catalog",
    "validate_override_ap_comparison_integrity",
]
