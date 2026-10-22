"""Versioned authority for AP identity throughout the semantic pipeline.

The context is deliberately formula-blind.  It may be built only from the
requirement representation profile, explicit metadata, or an approved AP
inventory tied to the effective source text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import re
from typing import Any, Iterable, Mapping

from ap_identity import undetermined_profile
from ap_grounding import (
    APGroundingContract,
    AUTHORITY_CONFLICT,
    AUTHORITY_EXACT,
    AUTHORITY_PROPOSAL_ONLY,
    APGroundingIdentity,
    APGroundingMention,
    build_ap_grounding_contract,
    normalize_ap_key,
)
from contract_versions import AP_BINDING_CONTEXT_CONTRACT_VERSION
from schemas import APInventory, APRepresentationProfile, DeclaredAPCatalog


CATALOG_AUTHORITY_NONE = "none"
CATALOG_AUTHORITY_CONTROLLED = "controlled_symbolic_profile"
CATALOG_AUTHORITY_DECLARED = "declared_catalog_metadata"
CATALOG_AUTHORITY_INVENTORY = "approved_human_inventory"
CATALOG_AUTHORITY_OFFLINE = "trusted_offline_formula_catalog"

CATALOG_AUTHORITIES = frozenset(
    {
        CATALOG_AUTHORITY_NONE,
        CATALOG_AUTHORITY_CONTROLLED,
        CATALOG_AUTHORITY_DECLARED,
        CATALOG_AUTHORITY_INVENTORY,
        CATALOG_AUTHORITY_OFFLINE,
    }
)


class APBindingContextError(ValueError):
    """A typed AP authority/context failure."""

    suggested_rerun_stage = "semantic"

    def __init__(self, code: str, message: str, *, diagnostics: dict | None = None):
        super().__init__(message)
        self.code = code
        self.diagnostics = {
            "reason_code": code,
            "suggested_rerun_stage": self.suggested_rerun_stage,
            **dict(diagnostics or {}),
        }


@dataclass(frozen=True)
class APBindingContext:
    requirement_id: str
    effective_source_hash: str
    representation_profile: APRepresentationProfile
    declared_catalog: tuple[str, ...] = ()
    catalog_authority: str = CATALOG_AUTHORITY_NONE
    active_inventory_id: str = ""
    active_inventory_hash: str = ""
    identity_mapping_id: str = ""
    identity_alias_groups: tuple[tuple[str, ...], ...] = ()
    context_contract_version: str = AP_BINDING_CONTEXT_CONTRACT_VERSION
    context_fingerprint: str = ""
    provenance: tuple[str, ...] = ()
    grounding_contract: APGroundingContract | None = None

    def __post_init__(self) -> None:
        if not self.requirement_id or not self.effective_source_hash:
            raise ValueError("AP binding context requires requirement/source identity")
        if self.catalog_authority not in CATALOG_AUTHORITIES:
            raise ValueError("unsupported AP catalog authority")
        if self.context_contract_version != AP_BINDING_CONTEXT_CONTRACT_VERSION:
            raise ValueError("unsupported AP binding context contract")
        normalized_catalog = tuple(dict.fromkeys(str(item) for item in self.declared_catalog))
        if normalized_catalog != self.declared_catalog:
            object.__setattr__(self, "declared_catalog", normalized_catalog)
        if self.catalog_authority == CATALOG_AUTHORITY_NONE and self.declared_catalog:
            raise ValueError("AP catalog without authority is forbidden")
        if self.grounding_contract is not None:
            if self.grounding_contract.requirement_id != self.requirement_id:
                raise ValueError("AP grounding/context requirement mismatch")
            if self.grounding_contract.source_hash != self.effective_source_hash:
                raise ValueError("AP grounding/context source mismatch")
        fingerprint = _context_fingerprint(self)
        if self.context_fingerprint != fingerprint:
            object.__setattr__(self, "context_fingerprint", fingerprint)

    @property
    def profile_mode(self) -> str:
        return self.representation_profile.mode

    @property
    def has_authoritative_catalog(self) -> bool:
        return (
            self.grounding_contract.authority == AUTHORITY_EXACT
            if self.grounding_contract is not None
            else self.catalog_authority != CATALOG_AUTHORITY_NONE
        )

    @property
    def grounding_authority(self) -> str:
        return (
            self.grounding_contract.authority
            if self.grounding_contract is not None
            else (
                AUTHORITY_EXACT
                if self.catalog_authority != CATALOG_AUTHORITY_NONE
                else "proposal_only"
            )
        )

    def validate_source(self, source_text: str) -> None:
        actual = source_text_hash(source_text)
        if actual != self.effective_source_hash:
            raise APBindingContextError(
                "IR_AP_CONTEXT_STALE",
                "AP binding context does not match the effective source text",
                diagnostics={
                    "requirement_id": self.requirement_id,
                    "expected_source_hash": self.effective_source_hash,
                    "actual_source_hash": actual,
                    "ap_context_fingerprint": self.context_fingerprint,
                },
            )
        if self.grounding_contract is not None:
            try:
                self.grounding_contract.validate_source(source_text)
            except Exception as exc:
                if isinstance(exc, APBindingContextError):
                    raise
                code = getattr(exc, "code", "IR_AP_GROUNDING_CONFLICT")
                diagnostics = getattr(exc, "diagnostics", {})
                raise APBindingContextError(
                    code,
                    str(exc),
                    diagnostics=dict(diagnostics),
                ) from exc

    def resolve_identity(self, name: str):
        """Resolve only a unique, authoritative catalog name or declared alias.

        Case folding does not authorize merging two distinct catalog identities.
        """
        if not self.has_authoritative_catalog:
            return None
        return self.resolve_candidate_identity(name)

    def resolve_candidate_identity(self, name: str):
        """Resolve a unique source-bound identity without granting authority.

        Structure diagnostics need stable identity keys for proposal-only
        candidates. Keeping this separate from ``resolve_identity`` prevents
        diagnostic canonicalization from becoming an approval mechanism.
        """
        contract = self.grounding_contract
        if contract is None:
            return None
        exact = contract.identity_by_name.get(name)
        if exact is not None:
            return exact
        matches = {
            item.identity_key: item for item in contract.identities
            if normalize_ap_key(name) in {
                normalize_ap_key(item.canonical_name),
                *(normalize_ap_key(alias) for alias in item.aliases)
            }
        }
        return next(iter(matches.values())) if len(matches) == 1 else None

    def prompt_payload(self) -> dict[str, object]:
        payload = {
            "requirement_id": self.requirement_id,
            "effective_source_hash": self.effective_source_hash,
            "representation_profile": self.profile_mode,
            "active_inventory_id": self.active_inventory_id,
            "active_inventory_hash": self.active_inventory_hash,
            "context_contract_version": self.context_contract_version,
            "context_fingerprint": self.context_fingerprint,
        }
        if self.grounding_contract is not None:
            payload["ap_grounding_contract"] = (
                self.grounding_contract.prompt_payload()
            )
        return payload


def source_text_hash(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def build_ap_binding_context(
    *,
    requirement_id: str,
    source_text: str,
    representation_profile: APRepresentationProfile | None = None,
    declared_catalog: DeclaredAPCatalog | None = None,
    active_inventory: APInventory | None = None,
    identity_mapping_id: str = "",
    trusted_offline_catalog: Iterable[str] = (),
    provenance: Iterable[str] = (),
) -> APBindingContext:
    """Build the single authoritative AP context for an effective source."""

    profile = representation_profile or undetermined_profile(
        provenance="text_only_compatibility_adapter"
    )
    effective_hash = source_text_hash(source_text)
    grounding = build_ap_grounding_contract(
        requirement_id=requirement_id,
        source_text=source_text,
        representation_profile=profile,
        declared_catalog=declared_catalog,
        active_inventory=active_inventory,
        trusted_offline_catalog=trusted_offline_catalog,
        provenance=provenance,
    )
    if grounding.authority == AUTHORITY_CONFLICT:
        raise APBindingContextError(
            "IR_AP_GROUNDING_CONFLICT",
            "AP grounding source/profile/inventory authority is conflicting",
            diagnostics={
                "requirement_id": requirement_id,
                "issue_codes": list(grounding.issue_codes),
                "grounding_fingerprint": grounding.fingerprint,
            },
        )
    authority = CATALOG_AUTHORITY_NONE
    catalog: tuple[str, ...] = ()
    alias_groups: tuple[tuple[str, ...], ...] = ()
    inventory_id = ""
    inventory_hash = ""
    context_provenance = list(provenance)
    context_provenance.append(profile.provenance)

    if active_inventory is not None:
        authority = CATALOG_AUTHORITY_INVENTORY
        inventory_id = active_inventory.inventory_id
        inventory_hash = active_inventory.artifact_hash
        context_provenance.append("approved_ap_inventory")
    elif profile.mode == "controlled_symbolic":
        authority = CATALOG_AUTHORITY_CONTROLLED
        context_provenance.append("controlled_symbolic_exact_extractor")
    elif profile.mode == "declared_catalog":
        if declared_catalog is None or declared_catalog.catalog_id != profile.declared_catalog_id:
            raise APBindingContextError(
                "IR_AP_CATALOG_AUTHORITY_MISSING",
                "declared_catalog profile requires its matching versioned catalog",
                diagnostics={"profile_id": profile.profile_id},
            )
        authority = CATALOG_AUTHORITY_DECLARED
        context_provenance.append(
            f"declared_catalog:{declared_catalog.catalog_id}:{declared_catalog.catalog_version}"
        )
    elif profile.mode == "offline_formula_assisted":
        authority = CATALOG_AUTHORITY_OFFLINE
        context_provenance.append("trusted_train_only_offline_catalog")

    if grounding.authority == AUTHORITY_EXACT:
        catalog = grounding.canonical_names
        alias_groups = tuple(
            tuple(
                dict.fromkeys(
                    (
                        identity.canonical_name,
                        *identity.aliases,
                        *(item.evidence_text for item in identity.mentions),
                    )
                )
            )
            for identity in grounding.identities
        )

    return APBindingContext(
        requirement_id=requirement_id,
        effective_source_hash=effective_hash,
        representation_profile=profile,
        declared_catalog=catalog,
        catalog_authority=authority,
        active_inventory_id=inventory_id,
        active_inventory_hash=inventory_hash,
        identity_mapping_id=identity_mapping_id,
        identity_alias_groups=alias_groups,
        provenance=tuple(dict.fromkeys(item for item in context_provenance if item)),
        grounding_contract=grounding,
    )


def text_only_ap_binding_context(
    source_text: str,
    *,
    requirement_id: str = "",
) -> APBindingContext:
    digest = source_text_hash(source_text)
    return build_ap_binding_context(
        requirement_id=requirement_id or "REQ-TEXT-" + digest[:12].upper(),
        source_text=source_text,
        representation_profile=undetermined_profile(
            provenance="text_only_compatibility_adapter"
        ),
        provenance=("text_only_compatibility_adapter",),
    )


def bind_proposal_ap_mentions(
    context: APBindingContext,
    proposals: Iterable[dict],
) -> APBindingContext:
    """Return an evidence-bound, explicitly non-authoritative AP context.

    This is the candidate-construction path for natural language without a
    catalog.  It does not upgrade the grounding authority and therefore cannot
    be mistaken for human- or catalog-approved AP identity.
    """

    grounding = context.grounding_contract
    if grounding is None or grounding.authority != AUTHORITY_PROPOSAL_ONLY:
        raise APBindingContextError(
            "IR_AP_PROPOSAL_PATH_NOT_APPLICABLE",
            "proposal AP mentions require a proposal-only grounding context",
        )
    staged: list[dict[str, Any]] = []
    for item in proposals:
        if not isinstance(item, dict):
            raise APBindingContextError(
                "IR_AP_PROPOSAL_INVALID", "proposal AP mention must be an object"
            )
        span = item.get("source_span")
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or any(type(value) is not int for value in span)
        ):
            raise APBindingContextError(
                "IR_AP_PROPOSAL_INVALID", "proposal AP mention requires an exact source_span"
            )
        canonical_name = str(item.get("canonical_name", "")).strip()
        evidence_text = str(item.get("evidence_text", ""))
        start, end = span
        if (
            not canonical_name
            or not 0 <= start < end <= len(context.grounding_contract.source_text)
            or context.grounding_contract.source_text[start:end] != evidence_text
        ):
            raise APBindingContextError(
                "IR_AP_EVIDENCE_IDENTITY_MISMATCH",
                "proposal AP mention requires a name and exact backend-derived source evidence",
            )
        staged.append({
            "canonical_name": canonical_name,
            "normalized_name": normalize_ap_key(canonical_name),
            "evidence_text": evidence_text,
            "source_span": (start, end),
            "supporting_segment_ids": tuple(item.get("supporting_segment_ids", ())),
        })
    if not staged:
        raise APBindingContextError(
            "IR_AP_PROPOSAL_INVALID", "at least one proposal AP mention is required"
        )

    # One normalized AP name may have multiple exact occurrences.  A source
    # span, however, may not be assigned to two different AP identities.
    span_owners: dict[tuple[int, int], str] = {}
    for item in staged:
        owner = span_owners.setdefault(item["source_span"], item["normalized_name"])
        if owner != item["normalized_name"]:
            raise APBindingContextError(
                "AP_CASEFOLD_COLLISION",
                "one source mention was assigned to multiple AP identities",
            )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in staged:
        grouped.setdefault(item["normalized_name"], []).append(item)
    identities = []
    for normalized_name, items in sorted(
        grouped.items(), key=lambda pair: min(value["source_span"] for value in pair[1])
    ):
        unique_items = {
            value["source_span"]: value for value in items
        }
        ordered = [unique_items[key] for key in sorted(unique_items)]
        identity_key = "apg-" + hashlib.sha256(
            f"{grounding.requirement_id}\0{grounding.source_hash}\0proposal:{normalized_name}".encode("utf-8")
        ).hexdigest()[:20]
        mentions = tuple(
            APGroundingMention(
                mention_id="apm-" + hashlib.sha256(
                    f"{identity_key}\0{value['source_span'][0]}\0{value['source_span'][1]}\0{index}".encode("utf-8")
                ).hexdigest()[:20],
                evidence_text=value["evidence_text"],
                source_span=value["source_span"],
                supporting_segment_ids=value["supporting_segment_ids"],
                occurrence_index=index,
            )
            for index, value in enumerate(ordered, start=1)
        )
        aliases = tuple(dict.fromkeys(
            value for item in items
            for value in (item["canonical_name"], item["evidence_text"])
            if value
        ))
        identities.append(APGroundingIdentity(
            identity_key=identity_key,
            canonical_name=_identifier(items[0]["canonical_name"]),
            aliases=aliases,
            mentions=mentions,
        ))
    if len({normalize_ap_key(item.canonical_name) for item in identities}) != len(identities):
        raise APBindingContextError(
            "AP_CASEFOLD_COLLISION",
            "proposal AP identities collide after case-insensitive normalization",
        )
    proposed_grounding = replace(
        grounding,
        identities=tuple(identities),
        issue_codes=("AP_PROPOSAL_REVIEW_REQUIRED",),
        provenance=(*grounding.provenance, "source_bound_proposal_mentions"),
        fingerprint="",
    )
    return replace(
        context,
        grounding_contract=proposed_grounding,
        provenance=(*context.provenance, "source_bound_proposal_mentions"),
        context_fingerprint="",
    )


def proposal_ap_mentions_from_semantics_profile(
    profile: Mapping[str, Any] | None,
    source_text: str,
) -> tuple[dict[str, object], ...]:
    """Extract only source-bound AP mentions from an online semantics profile.

    The profile remains non-authoritative.  This adapter closes the production
    gap where the same mentions were visible to the model but absent from the
    backend AP contract.
    """

    if not isinstance(profile, Mapping):
        return ()
    raw_nl = profile.get("raw_nl")
    if raw_nl is not None and raw_nl != source_text:
        raise APBindingContextError(
            "IR_AP_CONTEXT_STALE",
            "input semantics profile belongs to a different source text",
        )
    result: list[dict[str, object]] = []
    propositions = profile.get("atomic_propositions", ())
    if not isinstance(propositions, (list, tuple)):
        raise APBindingContextError(
            "IR_AP_PROPOSAL_INVALID",
            "input semantics profile atomic_propositions must be an array",
        )
    for proposition in propositions:
        if not isinstance(proposition, Mapping):
            raise APBindingContextError(
                "IR_AP_PROPOSAL_INVALID", "profile AP proposition must be an object"
            )
        name = str(
            proposition.get("ap")
            or proposition.get("canonical_name")
            or proposition.get("surface")
            or ""
        ).strip()
        mentions = proposition.get("mentions", ())
        if not name or not isinstance(mentions, (list, tuple)):
            raise APBindingContextError(
                "IR_AP_PROPOSAL_INVALID", "profile AP requires a name and mentions"
            )
        for mention in mentions:
            if not isinstance(mention, Mapping):
                raise APBindingContextError(
                    "IR_AP_PROPOSAL_INVALID", "profile AP mention must be an object"
                )
            start, end = mention.get("start"), mention.get("end")
            if (
                type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(source_text)
            ):
                raise APBindingContextError(
                    "IR_AP_PROPOSAL_INVALID", "profile AP mention has an invalid span"
                )
            result.append({
                "canonical_name": name,
                "evidence_text": source_text[start:end],
                "source_span": [start, end],
                "supporting_segment_ids": tuple(mention.get("supporting_segment_ids", ())),
            })
    return tuple(result)


def build_runtime_ap_binding_context(
    *,
    requirement_id: str,
    source_text: str,
    input_semantics_profile: Mapping[str, Any] | None = None,
    representation_profile: APRepresentationProfile | None = None,
    declared_catalog: DeclaredAPCatalog | None = None,
    active_inventory: APInventory | None = None,
    identity_mapping_id: str = "",
    trusted_offline_catalog: Iterable[str] = (),
    provenance: Iterable[str] = (),
    base_context: APBindingContext | None = None,
) -> APBindingContext:
    """Build one production AP context and bind source-only proposals once.

    A supplied context must already match the same requirement and source.
    Proposal mentions stay proposal-only; this adapter never promotes them to
    catalog or inventory authority.
    """

    if base_context is not None:
        if base_context.requirement_id != requirement_id:
            raise APBindingContextError(
                "IR_AP_CONTEXT_STALE",
                "base AP context belongs to a different requirement",
            )
        base_context.validate_source(source_text)
        context = base_context
    else:
        context = build_ap_binding_context(
            requirement_id=requirement_id,
            source_text=source_text,
            representation_profile=representation_profile,
            declared_catalog=declared_catalog,
            active_inventory=active_inventory,
            identity_mapping_id=identity_mapping_id,
            trusted_offline_catalog=trusted_offline_catalog,
            provenance=provenance,
        )
    mentions = proposal_ap_mentions_from_semantics_profile(
        input_semantics_profile,
        source_text,
    )
    if (
        mentions
        and context.grounding_authority == AUTHORITY_PROPOSAL_ONLY
        and context.grounding_contract is not None
        and not context.grounding_contract.identities
    ):
        context = bind_proposal_ap_mentions(context, mentions)
    return context


def _catalog_entry_identifier(catalog_ap_id: str, canonical_name: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", catalog_ap_id):
        return catalog_ap_id
    return _identifier(canonical_name)


def _identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_")
    if not normalized:
        return "ap"
    if normalized[0].isdigit():
        normalized = "ap_" + normalized
    return normalize_ap_key(normalized)


def _context_fingerprint(context: APBindingContext) -> str:
    payload = asdict(context)
    payload.pop("context_fingerprint", None)
    payload["representation_profile"] = asdict(context.representation_profile)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "APBindingContext",
    "APBindingContextError",
    "CATALOG_AUTHORITIES",
    "CATALOG_AUTHORITY_CONTROLLED",
    "CATALOG_AUTHORITY_DECLARED",
    "CATALOG_AUTHORITY_INVENTORY",
    "CATALOG_AUTHORITY_NONE",
    "CATALOG_AUTHORITY_OFFLINE",
    "build_ap_binding_context",
    "build_runtime_ap_binding_context",
    "bind_proposal_ap_mentions",
    "proposal_ap_mentions_from_semantics_profile",
    "source_text_hash",
    "text_only_ap_binding_context",
]
