"""Single, versioned AP identity authority for the semantic mainline.

The module deliberately owns only AP identity and exact source evidence.  It
does not decide Boolean/temporal meaning and never consumes formula gold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from ap_identity import (
    PROP_PLACEHOLDER_TOKEN_PATTERN,
    SINGLE_LOWER_LETTER_TOKEN_PATTERN,
    SYMBOLIC_GRAMMAR_PATTERNS,
)
from contract_versions import (
    AP_GROUNDING_CONTRACT_VERSION,
    AP_GROUNDING_EXTRACTOR_REGISTRY_VERSION,
)
from schemas import (
    APInventory,
    APMention,
    APRepresentationProfile,
    DeclaredAPCatalog,
)


AUTHORITY_EXACT = "exact"
AUTHORITY_REQUIRED_SUBSET = "required_subset"
AUTHORITY_PROPOSAL_ONLY = "proposal_only"
AUTHORITY_CONFLICT = "conflict"
GROUNDING_AUTHORITIES = frozenset(
    {
        AUTHORITY_EXACT,
        AUTHORITY_REQUIRED_SUBSET,
        AUTHORITY_PROPOSAL_ONLY,
        AUTHORITY_CONFLICT,
    }
)


def normalize_ap_key(value: object) -> str:
    """Return the one case-insensitive AP comparison key.

    Display names and exact source evidence remain unchanged.  This key is
    used only for identity comparison, collision detection, and cache/scoring
    normalization.
    """

    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def bind_source_quote(source: str, quote: str, span=None, *, occurrence=None) -> tuple[int, int]:
    """Resolve whole source fragments, retaining offsets through NFKC/casefold.

    Ambiguous repeated fragments require an explicit source span. Normalizing
    prefixes handles combining marks and length-changing compatibility forms.
    """
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("empty source quote")
    key = normalize_ap_key(quote)
    def boundary(start, end):
        return (not (source[start].isalnum() and start and source[start-1].isalnum())
                and not (source[end-1].isalnum() and end < len(source) and source[end].isalnum()))
    if span is not None:
        if (isinstance(span, (list, tuple)) and len(span) == 2
                and all(type(x) is int for x in span)
                and 0 <= span[0] < span[1] <= len(source)
                and normalize_ap_key(source[span[0]:span[1]]) == key
                and boundary(*span)):
            return tuple(span)
        # A bad model offset is recoverable when the quote has one legal match.
    normalized = unicodedata.normalize("NFKC", source).casefold()
    # Only stable prefix boundaries can be mapped back into source text.
    offsets = {}
    for index in range(len(source) + 1):
        prefix = unicodedata.normalize("NFKC", source[:index]).casefold()
        if normalized.startswith(prefix):
            offsets.setdefault(len(prefix), []).append(index)
    matches = set()
    for match in re.finditer(re.escape(key), normalized):
        for start in offsets.get(match.start(), []):
            for end in offsets.get(match.end(), []):
                if start < end and boundary(start, end) and normalize_ap_key(source[start:end]) == key:
                    matches.add((start, end))
    choices = sorted(matches)
    if occurrence is not None:
        if type(occurrence) is not int or not 1 <= occurrence <= len(choices):
            raise ValueError("invalid source quote occurrence")
        return choices[occurrence - 1]
    if len(choices) != 1:
        raise ValueError("source quote absent or ambiguous; provide source_span")
    return choices[0]

EXTRACTOR_PATTERNS: Mapping[str, str] = {
    "prop_placeholder_v1": PROP_PLACEHOLDER_TOKEN_PATTERN,
    "single_lower_letter_v1": SINGLE_LOWER_LETTER_TOKEN_PATTERN,
}


class APGroundingError(ValueError):
    """Typed, fail-closed AP grounding error."""

    suggested_rerun_stage = "semantic"

    def __init__(self, code: str, message: str, *, diagnostics: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.diagnostics = {
            "reason_code": code,
            "suggested_rerun_stage": self.suggested_rerun_stage,
            **dict(diagnostics or {}),
        }


@dataclass(frozen=True)
class APGroundingMention:
    mention_id: str
    evidence_text: str
    source_span: tuple[int, int]
    supporting_segment_ids: tuple[str, ...] = ()
    occurrence_index: int = 1

    def __post_init__(self) -> None:
        start, end = self.source_span
        if not self.mention_id or not self.evidence_text:
            raise ValueError("AP grounding mention requires identity and evidence")
        if start < 0 or end <= start:
            raise ValueError("AP grounding mention has invalid source span")
        if self.occurrence_index < 1:
            raise ValueError("AP grounding mention occurrence must be positive")


@dataclass(frozen=True)
class APGroundingIdentity:
    identity_key: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    mentions: tuple[APGroundingMention, ...] = ()

    def __post_init__(self) -> None:
        if not self.identity_key or not self.canonical_name:
            raise ValueError("AP grounding identity requires key and canonical name")
        mention_ids = [item.mention_id for item in self.mentions]
        if len(mention_ids) != len(set(mention_ids)):
            raise ValueError("AP grounding mention ids must be unique per identity")


@dataclass(frozen=True)
class APGroundingContract:
    requirement_id: str
    source_text: str = field(repr=False)
    source_hash: str
    representation_profile_id: str
    representation_mode: str
    symbolic_grammar_id: str
    authority: str
    identities: tuple[APGroundingIdentity, ...] = ()
    segmentation_id: str = ""
    parent_inventory_id: str = ""
    parent_inventory_hash: str = ""
    extractor_id: str = ""
    extractor_registry_version: str = AP_GROUNDING_EXTRACTOR_REGISTRY_VERSION
    contract_version: str = AP_GROUNDING_CONTRACT_VERSION
    issue_codes: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.requirement_id or not self.source_hash:
            raise ValueError("AP grounding contract requires requirement/source identity")
        if self.authority not in GROUNDING_AUTHORITIES:
            raise ValueError("unsupported AP grounding authority")
        if self.contract_version != AP_GROUNDING_CONTRACT_VERSION:
            raise ValueError("unsupported AP grounding contract")
        if _source_hash(self.source_text) != self.source_hash:
            raise ValueError("AP grounding source hash mismatch")
        names = [normalize_ap_key(item.canonical_name) for item in self.identities]
        keys = [item.identity_key for item in self.identities]
        if len(names) != len(set(names)) or len(keys) != len(set(keys)):
            raise APGroundingError(
                "AP_CASEFOLD_COLLISION",
                "AP grounding identities must have unique case-insensitive names and keys",
            )
        if self.authority == AUTHORITY_CONFLICT and not self.issue_codes:
            raise ValueError("conflicting AP grounding requires issue codes")
        computed = grounding_fingerprint(self)
        if self.fingerprint != computed:
            object.__setattr__(self, "fingerprint", computed)

    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(item.canonical_name for item in self.identities)

    @property
    def identity_by_name(self) -> dict[str, APGroundingIdentity]:
        return {item.canonical_name: item for item in self.identities}

    @property
    def identity_by_key(self) -> dict[str, APGroundingIdentity]:
        return {item.identity_key: item for item in self.identities}

    def validate_source(self, source_text: str) -> None:
        actual = _source_hash(source_text)
        if actual != self.source_hash:
            raise APGroundingError(
                "IR_AP_GROUNDING_STALE",
                "AP grounding contract belongs to a different effective source",
                diagnostics={
                    "expected_source_hash": self.source_hash,
                    "actual_source_hash": actual,
                    "grounding_fingerprint": self.fingerprint,
                },
            )
        if self.authority == AUTHORITY_CONFLICT:
            raise APGroundingError(
                "IR_AP_GROUNDING_CONFLICT",
                "AP grounding source/profile/inventory authority is conflicting",
                diagnostics={
                    "issue_codes": list(self.issue_codes),
                    "grounding_fingerprint": self.fingerprint,
                },
            )

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "representation_profile_id": self.representation_profile_id,
            "representation_mode": self.representation_mode,
            "symbolic_grammar_id": self.symbolic_grammar_id,
            "required_ap_identities": [
                {
                    "identity_key": item.identity_key,
                    "canonical_name": item.canonical_name,
                    "aliases": list(item.aliases),
                    "mentions": [
                        {
                            "mention_id": mention.mention_id,
                            "evidence_text": mention.evidence_text,
                            "source_span": list(mention.source_span),
                            "supporting_segment_ids": list(
                                mention.supporting_segment_ids
                            ),
                        }
                        for mention in item.mentions
                    ],
                }
                for item in self.identities
            ],
            "contract_version": self.contract_version,
            "fingerprint": self.fingerprint,
            "instruction": (
                "AP identities are backend-owned. Refer to canonical_name and exact "
                "source evidence; never merge, split, rename, or invent identities."
            ),
        }


def build_ap_grounding_contract(
    *,
    requirement_id: str,
    source_text: str,
    representation_profile: APRepresentationProfile,
    segmentation: object | None = None,
    declared_catalog: DeclaredAPCatalog | None = None,
    active_inventory: APInventory | None = None,
    trusted_offline_catalog: Iterable[str] = (),
    provenance: Iterable[str] = (),
) -> APGroundingContract:
    """Resolve exactly one AP authority for an effective source."""

    source_hash = _source_hash(source_text)
    segmentation_id = str(getattr(segmentation, "segmentation_id", "") or "")
    base = {
        "requirement_id": requirement_id,
        "source_text": source_text,
        "source_hash": source_hash,
        "representation_profile_id": representation_profile.profile_id,
        "representation_mode": representation_profile.mode,
        "symbolic_grammar_id": representation_profile.symbolic_grammar_id,
        "segmentation_id": segmentation_id,
        "provenance": tuple(
            dict.fromkeys(
                item
                for item in (*provenance, representation_profile.provenance)
                if item
            )
        ),
    }

    if active_inventory is not None:
        conflicts: list[str] = []
        if active_inventory.source_hash != source_hash:
            conflicts.append("AP_INVENTORY_SOURCE_STALE")
        if active_inventory.requirement_id != requirement_id:
            conflicts.append("AP_INVENTORY_REQUIREMENT_STALE")
        if active_inventory.representation_profile_id != representation_profile.profile_id:
            conflicts.append("AP_INVENTORY_PROFILE_STALE")
        if segmentation_id and active_inventory.segmentation_id not in {"", segmentation_id}:
            conflicts.append("AP_INVENTORY_SEGMENTATION_STALE")
        if active_inventory.status != "approved":
            conflicts.append("AP_INVENTORY_NOT_APPROVED")
        if conflicts:
            return APGroundingContract(
                **base,
                authority=AUTHORITY_CONFLICT,
                parent_inventory_id=active_inventory.inventory_id,
                parent_inventory_hash=active_inventory.artifact_hash,
                extractor_id="approved_inventory_adapter_v1",
                issue_codes=tuple(conflicts),
            )
        return APGroundingContract(
            **base,
            authority=AUTHORITY_EXACT,
            identities=_identities_from_inventory(active_inventory),
            parent_inventory_id=active_inventory.inventory_id,
            parent_inventory_hash=active_inventory.artifact_hash,
            extractor_id="approved_inventory_adapter_v1",
        )

    if representation_profile.mode == "declared_catalog":
        if (
            declared_catalog is None
            or declared_catalog.catalog_id != representation_profile.declared_catalog_id
        ):
            return APGroundingContract(
                **base,
                authority=AUTHORITY_CONFLICT,
                extractor_id="declared_catalog_adapter_v1",
                issue_codes=("AP_DECLARED_CATALOG_MISSING",),
            )
        return APGroundingContract(
            **base,
            authority=AUTHORITY_EXACT,
            identities=_identities_from_declared_catalog(
                source_text,
                declared_catalog,
                segmentation,
            ),
            extractor_id="declared_catalog_adapter_v1",
        )

    if representation_profile.mode == "controlled_symbolic":
        grammar_id = representation_profile.symbolic_grammar_id
        pattern = EXTRACTOR_PATTERNS.get(grammar_id)
        if pattern is None:
            return APGroundingContract(
                **base,
                authority=AUTHORITY_CONFLICT,
                extractor_id=grammar_id or "missing",
                issue_codes=("AP_SYMBOLIC_GRAMMAR_UNSUPPORTED",),
            )
        return APGroundingContract(
            **base,
            authority=AUTHORITY_EXACT,
            identities=_identities_from_pattern(
                requirement_id,
                source_text,
                pattern,
                grammar_id,
                segmentation,
            ),
            extractor_id=grammar_id,
        )

    if representation_profile.mode == "offline_formula_assisted":
        values = tuple(dict.fromkeys(_identifier(item) for item in trusted_offline_catalog))
        if not values:
            return APGroundingContract(
                **base,
                authority=AUTHORITY_CONFLICT,
                extractor_id="trusted_offline_catalog_v1",
                issue_codes=("AP_OFFLINE_CATALOG_MISSING",),
            )
        identities = tuple(
            APGroundingIdentity(
                identity_key=_identity_key(requirement_id, source_hash, name),
                canonical_name=name,
                aliases=(name,),
                mentions=(),
            )
            for name in values
        )
        return APGroundingContract(
            **base,
            authority=AUTHORITY_EXACT,
            identities=identities,
            extractor_id="trusted_offline_catalog_v1",
        )

    explicit = _identities_from_pattern(
        requirement_id,
        source_text,
        PROP_PLACEHOLDER_TOKEN_PATTERN,
        "prop_placeholder_v1",
        segmentation,
    )
    if explicit:
        return APGroundingContract(
            **base,
            authority=AUTHORITY_REQUIRED_SUBSET,
            identities=explicit,
            extractor_id="prop_placeholder_v1",
        )
    return APGroundingContract(
        **base,
        authority=AUTHORITY_PROPOSAL_ONLY,
        identities=(),
        extractor_id="grounded_phrase_proposal_v1",
    )


def proposal_identity(
    contract: APGroundingContract,
    *,
    canonical_name: str,
    evidence_text: str,
    source_span: tuple[int, int],
    supporting_segment_ids: Sequence[str] = (),
) -> APGroundingIdentity:
    """Create a deterministic evidence-bound identity in proposal-only mode."""

    contract.validate_source(contract.source_text)
    start, end = source_span
    if start < 0 or end <= start or contract.source_text[start:end] != evidence_text:
        raise APGroundingError(
            "IR_AP_EVIDENCE_IDENTITY_MISMATCH",
            "proposal AP identity requires exact backend-derived source evidence",
        )
    normalized_name = _identifier(canonical_name)
    key = _identity_key(
        contract.requirement_id,
        contract.source_hash,
        normalized_name + f":{start}:{end}",
    )
    mention = APGroundingMention(
        mention_id=_mention_key(key, start, end, 1),
        evidence_text=evidence_text,
        source_span=(start, end),
        supporting_segment_ids=tuple(supporting_segment_ids),
        occurrence_index=1,
    )
    return APGroundingIdentity(
        identity_key=key,
        canonical_name=normalized_name,
        aliases=(canonical_name, evidence_text),
        mentions=(mention,),
    )


def identity_for_evidence(
    contract: APGroundingContract,
    evidence_text: str,
    source_span: tuple[int, int] | None = None,
) -> APGroundingIdentity | None:
    """Return an identity only when the evidence proves a unique match."""

    candidates: list[APGroundingIdentity] = []
    normalized = normalize_ap_key(evidence_text)
    for identity in contract.identities:
        if any(
            (
                source_span is not None
                and (
                    mention.source_span == source_span
                    or (
                        source_span[0] >= 0
                        and mention.source_span[0] >= source_span[0]
                        and mention.source_span[1] <= source_span[1]
                    )
                )
            )
            or normalize_ap_key(mention.evidence_text) == normalized
            for mention in identity.mentions
        ):
            candidates.append(identity)
            continue
        aliases = {
            normalize_ap_key(identity.canonical_name),
            *(normalize_ap_key(alias) for alias in identity.aliases),
        }
        if normalized in aliases or any(
            re.search(
                r"(?<![A-Za-z0-9_])"
                + re.escape(alias)
                + r"(?![A-Za-z0-9_])",
                evidence_text,
                flags=re.IGNORECASE,
            )
            for alias in aliases
            if alias
        ):
            candidates.append(identity)
    unique = {item.identity_key: item for item in candidates}
    return next(iter(unique.values())) if len(unique) == 1 else None


def grounding_fingerprint(contract: APGroundingContract) -> str:
    payload = asdict(contract)
    payload.pop("fingerprint", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _identities_from_inventory(
    inventory: APInventory,
) -> tuple[APGroundingIdentity, ...]:
    mentions_by_identity: dict[str, list[APGroundingMention]] = {}
    for mention in inventory.mentions:
        mentions_by_identity.setdefault(mention.ap_identity_id, []).append(
            _mention_from_inventory(mention)
        )
    return tuple(
        APGroundingIdentity(
            identity_key=identity.ap_identity_id,
            canonical_name=_identifier(identity.canonical_name),
            aliases=tuple(
                dict.fromkeys((identity.canonical_name, *identity.aliases))
            ),
            mentions=tuple(mentions_by_identity.get(identity.ap_identity_id, ())),
        )
        for identity in inventory.identities
    )


def _mention_from_inventory(mention: APMention) -> APGroundingMention:
    return APGroundingMention(
        mention_id=mention.mention_id,
        evidence_text=mention.evidence_text,
        source_span=mention.predicate_span,
        supporting_segment_ids=mention.supporting_segment_ids,
        occurrence_index=mention.occurrence_index,
    )


def _identities_from_declared_catalog(
    source_text: str,
    catalog: DeclaredAPCatalog,
    segmentation: object | None,
) -> tuple[APGroundingIdentity, ...]:
    identities: list[APGroundingIdentity] = []
    source_hash = _source_hash(source_text)
    for entry in catalog.entries:
        canonical = _identifier(entry.catalog_ap_id or entry.canonical_name)
        mentions = _mentions_for_aliases(
            canonical,
            source_text,
            (entry.canonical_name, *entry.aliases),
            segmentation,
        )
        identities.append(
            APGroundingIdentity(
                identity_key=entry.catalog_ap_id or _identity_key(
                    catalog.catalog_id,
                    source_hash,
                    canonical,
                ),
                canonical_name=canonical,
                aliases=tuple(
                    dict.fromkeys((entry.canonical_name, *entry.aliases))
                ),
                mentions=mentions,
            )
        )
    return tuple(identities)


def _identities_from_pattern(
    requirement_id: str,
    source_text: str,
    pattern: str,
    extractor_id: str,
    segmentation: object | None,
) -> tuple[APGroundingIdentity, ...]:
    grouped: dict[str, list[re.Match[str]]] = {}
    for match in re.finditer(pattern, source_text, flags=re.IGNORECASE):
        grouped.setdefault(normalize_ap_key(match.group(0)), []).append(match)
    source_hash = _source_hash(source_text)
    ordered = sorted(grouped, key=lambda item: grouped[item][0].start())
    identities: list[APGroundingIdentity] = []
    for canonical in ordered:
        key = _identity_key(
            requirement_id,
            source_hash,
            extractor_id + ":" + canonical,
        )
        mentions = tuple(
            APGroundingMention(
                mention_id=_mention_key(
                    key,
                    match.start(),
                    match.end(),
                    index,
                ),
                evidence_text=source_text[match.start():match.end()],
                source_span=(match.start(), match.end()),
                supporting_segment_ids=_segments_for_span(
                    (match.start(), match.end()),
                    segmentation,
                ),
                occurrence_index=index,
            )
            for index, match in enumerate(grouped[canonical], start=1)
        )
        identities.append(
            APGroundingIdentity(
                identity_key=key,
                canonical_name=_identifier(canonical),
                aliases=(canonical,),
                mentions=mentions,
            )
        )
    return tuple(identities)


def _mentions_for_aliases(
    identity_name: str,
    source_text: str,
    aliases: Sequence[str],
    segmentation: object | None,
) -> tuple[APGroundingMention, ...]:
    key = _identity_key(identity_name, _source_hash(source_text), identity_name)
    spans: list[tuple[int, int]] = []
    for alias in aliases:
        if not alias:
            continue
        pattern = (
            r"(?<![A-Za-z0-9_])"
            + re.escape(alias)
            + r"(?![A-Za-z0-9_])"
        )
        spans.extend(
            (match.start(), match.end())
            for match in re.finditer(pattern, source_text, flags=re.IGNORECASE)
        )
    unique = sorted(set(spans))
    return tuple(
        APGroundingMention(
            mention_id=_mention_key(key, start, end, index),
            evidence_text=source_text[start:end],
            source_span=(start, end),
            supporting_segment_ids=_segments_for_span(
                (start, end),
                segmentation,
            ),
            occurrence_index=index,
        )
        for index, (start, end) in enumerate(unique, start=1)
    )


def _segments_for_span(
    span: tuple[int, int],
    segmentation: object | None,
) -> tuple[str, ...]:
    if segmentation is None:
        return ()
    result: list[str] = []
    for segment in getattr(segmentation, "segments", ()):
        segment_span = getattr(segment, "source_span", (-1, -1))
        if not isinstance(segment_span, (tuple, list)) or len(segment_span) != 2:
            continue
        start, end = int(segment_span[0]), int(segment_span[1])
        if span[0] < end and span[1] > start:
            segment_id = str(getattr(segment, "segment_id", ""))
            if segment_id:
                result.append(segment_id)
    return tuple(result)


def _identity_key(requirement_id: str, source_hash: str, value: str) -> str:
    digest = hashlib.sha256(
        f"{requirement_id}\0{source_hash}\0{value}".encode("utf-8")
    ).hexdigest()[:20]
    return "apg-" + digest


def _mention_key(
    identity_key: str,
    start: int,
    end: int,
    occurrence: int,
) -> str:
    digest = hashlib.sha256(
        f"{identity_key}\0{start}\0{end}\0{occurrence}".encode("utf-8")
    ).hexdigest()[:20]
    return "apm-" + digest


def _identifier(value: object) -> str:
    normalized = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        str(value).strip(),
    ).strip("_")
    if not normalized:
        return "ap"
    if normalized[0].isdigit():
        return "ap_" + normalized
    return normalize_ap_key(normalized)


def _source_hash(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


__all__ = [
    "APGroundingContract",
    "APGroundingError",
    "APGroundingIdentity",
    "APGroundingMention",
    "AUTHORITY_CONFLICT",
    "AUTHORITY_EXACT",
    "AUTHORITY_PROPOSAL_ONLY",
    "AUTHORITY_REQUIRED_SUBSET",
    "EXTRACTOR_PATTERNS",
    "GROUNDING_AUTHORITIES",
    "build_ap_grounding_contract",
    "grounding_fingerprint",
    "identity_for_evidence",
    "normalize_ap_key",
    "proposal_identity",
]
