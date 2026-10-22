"""Typed and fail-closed authority boundary for semantic source text."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from typing import Any

from contract_versions import (
    EFFECTIVE_SEMANTIC_SOURCE_CONTRACT_VERSION,
    REVIEWED_FORMAL_SOURCE_AUTHORITY_CONTRACT_VERSION,
)
from ltlf_parser import Formula, parse_formula


NATURAL_LANGUAGE = "natural_language"
REVIEWED_FORMAL_LTLF = "reviewed_formal_ltlf"


class SemanticSourceAuthorityError(ValueError):
    """Raised when source modality or authority cannot be proved."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code
        self.status = "awaiting_source_modality_choice"


@dataclass(frozen=True)
class ReviewedFormalSemanticOverride:
    authority_record_id: str
    requirement_id: str
    formula_text: str
    canonical_ast_hash: str
    parent_source_hash: str
    reviewer_id: str
    reviewer_role: str = "semantic_expert"
    approval_status: str = "approved"
    parser_contract: str = "ltlf_parser"
    ap_inventory_id: str = ""
    ap_inventory_hash: str = ""
    contract_version: str = REVIEWED_FORMAL_SOURCE_AUTHORITY_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if (
            not self.authority_record_id
            or not self.requirement_id
            or not self.parent_source_hash
        ):
            raise ValueError("reviewed formal authority has incomplete lineage")
        if self.reviewer_role != "semantic_expert":
            raise ValueError("reviewed formal authority requires semantic_expert")
        if self.approval_status != "approved":
            raise ValueError("reviewed formal authority is not approved")
        formula = parse_formula(self.formula_text)
        expected_hash = _hash_text(str(formula))
        if self.canonical_ast_hash != expected_hash:
            raise ValueError("reviewed formal authority AST hash mismatch")


@dataclass(frozen=True)
class EffectiveSemanticSource:
    source_id: str
    requirement_id: str
    source_kind: str
    source_text: str
    source_hash: str
    authority_kind: str
    authority_record_id: str
    parent_source_hash: str = ""
    origin: str = "original"
    ap_inventory_id: str = ""
    ap_inventory_hash: str = ""
    contract_version: str = EFFECTIVE_SEMANTIC_SOURCE_CONTRACT_VERSION
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.source_kind not in {NATURAL_LANGUAGE, REVIEWED_FORMAL_LTLF}:
            raise ValueError("unsupported effective semantic source kind")
        if not self.requirement_id or not self.source_text.strip():
            raise ValueError("effective semantic source requires id and text")
        if self.source_hash != _hash_text(self.source_text):
            raise ValueError("effective semantic source hash mismatch")
        if not self.authority_kind or not self.authority_record_id:
            raise ValueError("effective semantic source authority is incomplete")
        if self.contract_version != EFFECTIVE_SEMANTIC_SOURCE_CONTRACT_VERSION:
            raise ValueError("unsupported effective semantic source contract")
        expected = _fingerprint(self, exclude={"fingerprint"})
        if self.fingerprint != expected:
            object.__setattr__(self, "fingerprint", expected)
        expected_id = "ESRC-" + expected[:20]
        if self.source_id != expected_id:
            object.__setattr__(self, "source_id", expected_id)


def natural_language_source(
    *,
    requirement_id: str,
    source_text: str,
    authority_kind: str = "requirement_record",
    authority_record_id: str = "",
    parent_source_hash: str = "",
    origin: str = "original",
    ap_inventory_id: str = "",
    ap_inventory_hash: str = "",
) -> EffectiveSemanticSource:
    if is_complete_formula_like_source(source_text):
        raise SemanticSourceAuthorityError(
            "complete LTLf-like text requires an approved formal authority record",
            code="FORMULA_LIKE_SOURCE_REQUIRES_FORMAL_AUTHORITY",
        )
    authority_record_id = authority_record_id or f"RequirementRecord:{requirement_id}"
    return EffectiveSemanticSource(
        source_id="pending",
        requirement_id=requirement_id,
        source_kind=NATURAL_LANGUAGE,
        source_text=source_text,
        source_hash=_hash_text(source_text),
        authority_kind=authority_kind,
        authority_record_id=authority_record_id,
        parent_source_hash=parent_source_hash,
        origin=origin,
        ap_inventory_id=ap_inventory_id,
        ap_inventory_hash=ap_inventory_hash,
    )


def reviewed_formal_source(
    authority: ReviewedFormalSemanticOverride,
) -> EffectiveSemanticSource:
    return EffectiveSemanticSource(
        source_id="pending",
        requirement_id=authority.requirement_id,
        source_kind=REVIEWED_FORMAL_LTLF,
        source_text=authority.formula_text,
        source_hash=_hash_text(authority.formula_text),
        authority_kind="expert_formal_review",
        authority_record_id=authority.authority_record_id,
        parent_source_hash=authority.parent_source_hash,
        origin="human_override",
        ap_inventory_id=authority.ap_inventory_id,
        ap_inventory_hash=authority.ap_inventory_hash,
    )


def source_from_ordinary_override(override: object) -> EffectiveSemanticSource:
    if getattr(override, "status", "") != "active":
        raise ValueError("ordinary override must be active before source binding")
    return natural_language_source(
        requirement_id=str(getattr(override, "requirement_id", "")),
        source_text=str(getattr(override, "replacement_text", "")),
        authority_kind="ordinary_review",
        authority_record_id=str(getattr(override, "override_id", "")),
        parent_source_hash=str(
            getattr(override, "parent_effective_source_hash", "")
        ),
        origin="human_override",
        ap_inventory_id=str(
            getattr(override, "replacement_ap_inventory_id", "")
        ),
        ap_inventory_hash=str(
            getattr(override, "replacement_ap_inventory_hash", "")
        ),
    )


def bind_semantic_to_effective_source(
    semantic: object,
    source: EffectiveSemanticSource,
) -> object:
    ir = getattr(semantic, "faithful_ir", None)
    if ir is None:
        raise ValueError("effective source binding requires FaithfulSemanticIR")
    if (
        getattr(ir, "requirement_id", "") != source.requirement_id
        or getattr(ir, "source_hash", "") != source.source_hash
        or getattr(ir, "source_text", "") != source.source_text
    ):
        raise ValueError("effective source does not match semantic IR lineage")
    from faithful_ir import compute_ir_version

    def bind_ir(candidate):
        if (
            getattr(candidate, "requirement_id", "") != source.requirement_id
            or getattr(candidate, "source_hash", "") != source.source_hash
            or getattr(candidate, "source_text", "") != source.source_text
        ):
            raise ValueError("candidate IR effective source lineage mismatch")
        bound = replace(
            candidate,
            semantic_source_contract_version=source.contract_version,
            effective_source_fingerprint=source.fingerprint,
            effective_source_kind=source.source_kind,
            effective_source_authority_record_id=source.authority_record_id,
            ir_version="",
        )
        return replace(bound, ir_version=compute_ir_version(bound))

    bound_ir = bind_ir(ir)
    candidate_set = getattr(semantic, "candidate_ir_set", None)
    if candidate_set is not None:
        candidate_set = replace(
            candidate_set,
            branches=tuple(
                replace(branch, faithful_ir=bind_ir(branch.faithful_ir))
                for branch in candidate_set.branches
            ),
        )
    return replace(
        semantic,
        faithful_ir=bound_ir,
        candidate_ir_set=candidate_set,
        semantic_source_contract_version=source.contract_version,
        effective_source_fingerprint=source.fingerprint,
        effective_source_kind=source.source_kind,
        effective_source_authority_record_id=source.authority_record_id,
    )


def validate_semantic_source_binding(semantic: object) -> None:
    ir = getattr(semantic, "faithful_ir", None)
    if ir is None:
        raise ValueError("semantic source binding requires FaithfulSemanticIR")
    pairs = (
        ("semantic_source_contract_version", EFFECTIVE_SEMANTIC_SOURCE_CONTRACT_VERSION),
        ("effective_source_fingerprint", None),
        ("effective_source_kind", None),
        ("effective_source_authority_record_id", None),
    )
    for name, exact in pairs:
        semantic_value = str(getattr(semantic, name, ""))
        ir_value = str(getattr(ir, name, ""))
        if not semantic_value or semantic_value != ir_value:
            raise ValueError(f"semantic source binding mismatch: {name}")
        if exact is not None and semantic_value != exact:
            raise ValueError(f"unsupported semantic source binding: {name}")


def is_complete_formula_like_source(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    # Word aliases such as "until", "always", and "next" are also ordinary
    # language in this project.  Require at least one unambiguously formal
    # lexical marker before consulting the full parser.
    if not re.search(
        r"(?:<->|->|&&|\|\||\\\/|/\\|[!&|])|(?:\b[FGXU]\b)",
        value,
    ):
        return False
    try:
        formula = parse_formula(value)
    except (TypeError, ValueError):
        return False
    return any(node.op not in {"ap", "true", "false"} for node in _walk(formula))


def _walk(formula: Formula):
    yield formula
    for child in formula.args:
        yield from _walk(child)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fingerprint(value: Any, *, exclude: set[str] = set()) -> str:
    payload = asdict(value)
    for name in exclude:
        payload.pop(name, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "EffectiveSemanticSource",
    "NATURAL_LANGUAGE",
    "REVIEWED_FORMAL_LTLF",
    "ReviewedFormalSemanticOverride",
    "SemanticSourceAuthorityError",
    "bind_semantic_to_effective_source",
    "is_complete_formula_like_source",
    "natural_language_source",
    "reviewed_formal_source",
    "source_from_ordinary_override",
    "validate_semantic_source_binding",
]
