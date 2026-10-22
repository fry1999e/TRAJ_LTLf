"""Backend-authoritative source evidence location and canonicalization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from ap_binding_context import (
    APBindingContext,
    APBindingContextError,
    CATALOG_AUTHORITY_NONE,
    text_only_ap_binding_context,
)
from ap_grounding import (
    APGroundingContract,
    APGroundingIdentity,
    AUTHORITY_EXACT,
    AUTHORITY_REQUIRED_SUBSET,
    identity_for_evidence,
    proposal_identity,
)

from contract_versions import (
    FAITHFUL_IR_CANONICALIZER_VERSION,
    FAITHFUL_IR_CONTRACT_VERSION,
    FAITHFUL_IR_SCHEMA_VERSION,
    FAITHFUL_IR_VALIDATOR_VERSION,
)
from semantic_topology import (
    GraphIntegrityProblem,
    faithful_ir_proposal_mapping,
    raw_ir_graph_integrity_problems,
)
from semantic_node_contracts import node_contract_fingerprint
from contract_versions import TEMPORAL_LANGUAGE_CONTRACT_VERSION
from semantic_policy_binding import SemanticPolicyTargetRequest
from validated_generation import (
    CANONICALIZER_VERSION,
    CanonicalizedArtifact,
    RepairClass,
    Severity,
    ValidationIssue,
    json_pointer,
)


@dataclass(frozen=True)
class EvidenceLocation:
    source_span: tuple[int, int]
    evidence_text: str
    source_segment_ids: tuple[str, ...]
    occurrence_index: int


def locate_evidence_quote(
    source_text: str,
    segments: Sequence[object],
    quote: str,
    *,
    source_segment_ids: Sequence[str] = (),
    occurrence_index: int | None = None,
    proposed_span: object = None,
    whole_token: bool = False,
    stage: str = "generation",
    path: str = "/evidence_quote",
) -> EvidenceLocation | ValidationIssue:
    """Resolve an exact quote without trusting model-proposed numeric offsets.

    ``occurrence_index`` is relative to the candidate window selected by
    ``source_segment_ids``.  A valid ``proposed_span`` may disambiguate that
    same window, but the returned coordinates are always derived again from
    the source text.
    """

    if not isinstance(quote, str) or not quote:
        return ValidationIssue(
            code="EVIDENCE_QUOTE_REQUIRED",
            stage=stage,
            path=path,
            observed=quote,
            expected={"type": "non_empty_exact_source_substring"},
            repair_class=RepairClass.CONSTRAINED,
            allowed_paths=(path,),
            message="evidence_quote 必须是非空原文子串",
        )

    normalized_segments = tuple(_segment_view(segment) for segment in segments)
    requested = tuple(str(item) for item in source_segment_ids)
    by_id = {segment[0]: segment for segment in normalized_segments}
    known_requested = tuple(item for item in requested if item in by_id)
    requested_positions = [
        index for index, segment in enumerate(normalized_segments) if segment[0] in known_requested
    ]
    requested_is_contiguous = (
        len(known_requested) == len(requested)
        and len(set(requested)) == len(requested)
        and requested_positions == list(range(min(requested_positions), max(requested_positions) + 1))
        if requested_positions
        else not requested
    )

    candidates = _quote_candidates(source_text, quote, whole_token=whole_token)
    if not candidates:
        return ValidationIssue(
            code="EVIDENCE_QUOTE_NOT_FOUND",
            stage=stage,
            path=path,
            observed=quote,
            expected={"exact_substring_of_source": True},
            context={
                "source_length": len(source_text),
                "requested_segment_ids": list(requested),
                "match_mode": "whole_token" if whole_token else "substring",
            },
            repair_class=RepairClass.SEMANTIC_PATCH,
            allowed_paths=(path,),
            message="evidence_quote 无法在当前原文中定位",
        )

    constrained = candidates
    if requested and requested_is_contiguous:
        selected = [by_id[item] for item in requested]
        window = (min(item[2] for item in selected), max(item[3] for item in selected))
        within = tuple(
            candidate
            for candidate in candidates
            if window[0] <= candidate[0] and candidate[1] <= window[1]
        )
        if within:
            constrained = within

    if occurrence_index is None:
        occurrence_index = _validated_span_occurrence(
            source_text,
            quote,
            proposed_span,
            candidates=constrained,
            whole_token=whole_token,
        )
    # A valid segment window that contains exactly one candidate is already a
    # complete evidence selector.  Treat a stale/global occurrence index as
    # redundant and canonicalize it to the window-relative index instead of
    # spending a repair attempt on a choice that is no longer ambiguous.
    if len(constrained) == 1:
        occurrence_index = 0
    if occurrence_index is None and len(constrained) != 1:
        return ValidationIssue(
            code="EVIDENCE_QUOTE_OCCURRENCE_REQUIRED",
            stage=stage,
            path=path.rsplit("/", 1)[0] + "/occurrence_index",
            observed=None,
            expected={"integer": {"minimum": 0, "maximum": len(constrained) - 1}},
            context={
                "quote": quote,
                "candidate_spans": [list(item) for item in constrained],
                "offset_unit": "unicode_codepoint",
                "end_exclusive": True,
                "match_mode": "whole_token" if whole_token else "substring",
            },
            repair_class=RepairClass.CONSTRAINED,
            allowed_paths=(path.rsplit("/", 1)[0] + "/occurrence_index",),
            message="evidence_quote 在来源窗口内多次出现，需要 occurrence_index",
        )
    selected_index = occurrence_index if occurrence_index is not None else 0
    if not isinstance(selected_index, int) or not (0 <= selected_index < len(constrained)):
        return ValidationIssue(
            code="EVIDENCE_OCCURRENCE_OUT_OF_RANGE",
            stage=stage,
            path=path.rsplit("/", 1)[0] + "/occurrence_index",
            observed=selected_index,
            expected={"integer": {"minimum": 0, "maximum": len(constrained) - 1}},
            repair_class=RepairClass.CONSTRAINED,
            allowed_paths=(path.rsplit("/", 1)[0] + "/occurrence_index",),
            message="occurrence_index 超出候选范围",
        )

    start, end = constrained[selected_index]
    derived_segments = tuple(
        segment_id
        for segment_id, _index, segment_start, segment_end, _text in normalized_segments
        if segment_start < end and start < segment_end
    )
    return EvidenceLocation(
        source_span=(start, end),
        evidence_text=source_text[start:end],
        source_segment_ids=derived_segments,
        occurrence_index=selected_index,
    )


def canonicalize_plain_reading_proposal(
    proposal: Any,
    *,
    source_text: str,
    segments: Sequence[object],
    source_hash: str,
) -> CanonicalizedArtifact:
    """Derive evidence coordinates and stable ids for a plain-reading proposal."""

    if not isinstance(proposal, Mapping):
        return CanonicalizedArtifact(proposal)
    artifact = deepcopy(dict(proposal))
    ambiguities = artifact.get("ambiguities")
    if not isinstance(ambiguities, list):
        return CanonicalizedArtifact(artifact)
    issues: list[ValidationIssue] = []
    for index, ambiguity in enumerate(ambiguities):
        if not isinstance(ambiguity, Mapping):
            continue
        item = dict(ambiguity)
        base_path = json_pointer("ambiguities", index)
        quote = item.get("evidence_quote", item.get("evidence_text", ""))
        location = locate_evidence_quote(
            source_text,
            segments,
            str(quote) if quote is not None else "",
            source_segment_ids=item.get("source_segment_ids", ()),
            occurrence_index=item.get("occurrence_index"),
            stage="plain_reading",
            path=base_path + "/evidence_quote",
        )
        item["ambiguity_id"] = _stable_id(
            "amb",
            source_hash,
            str(index),
            str(quote),
        )
        raw_options = item.get("options")
        if isinstance(raw_options, list):
            item["options"] = [
                {**dict(option), "option_id": chr(ord("A") + option_index)}
                if isinstance(option, Mapping)
                else option
                for option_index, option in enumerate(raw_options)
            ]
        if isinstance(location, ValidationIssue):
            issues.append(location)
            # Numeric coordinates from the model are never retained as trusted data.
            item["source_span"] = [-1, -1]
            item["evidence_text"] = ""
        else:
            item["evidence_quote"] = location.evidence_text
            item["evidence_text"] = location.evidence_text
            item["source_span"] = list(location.source_span)
            item["source_segment_ids"] = list(location.source_segment_ids)
            if "occurrence_index" in item or location.occurrence_index:
                item["occurrence_index"] = location.occurrence_index
        ambiguities[index] = item
    artifact["ambiguities"] = ambiguities
    artifact["source_hash"] = source_hash
    artifact["contract_version"] = "plain-reading-contract-v5"
    artifact["canonicalizer_version"] = CANONICALIZER_VERSION
    return CanonicalizedArtifact(artifact, tuple(issues))


def canonicalize_faithful_ir_proposal(
    proposal: Any,
    *,
    source_text: str,
    segments: Sequence[object] = (),
    requirement_id: str = "",
    reviewed_ambiguities: Sequence[object] = (),
    ap_binding_context: APBindingContext | None = None,
    policy_target_request: SemanticPolicyTargetRequest | None = None,
) -> CanonicalizedArtifact:
    """Canonicalize backend-owned IR version, ids, source, and evidence spans.

    Active reviewed axes own their id, question, alternatives, and decision.
    A model may reference such an axis but cannot rewrite those fields.  All
    other model-proposed ids remain backend-owned positional ids.
    """

    context = ap_binding_context or text_only_ap_binding_context(
        source_text,
        requirement_id=requirement_id,
    )
    try:
        context.validate_source(source_text)
    except APBindingContextError as exc:
        return CanonicalizedArtifact(
            proposal,
            (
                ValidationIssue(
                    code=exc.code,
                    stage="semantic",
                    path="/faithful_ir/ap_context_fingerprint",
                    observed={
                        "source_hash": hashlib.sha256(
                            source_text.encode("utf-8")
                        ).hexdigest()
                    },
                    expected={
                        "effective_source_hash": context.effective_source_hash,
                        "ap_context_fingerprint": context.context_fingerprint,
                    },
                    context=dict(exc.diagnostics),
                    repair_class=RepairClass.HUMAN_REQUIRED,
                    allowed_paths=(),
                    message=str(exc),
                ),
            ),
        )
    if not isinstance(proposal, Mapping):
        return CanonicalizedArtifact(proposal)
    artifact = deepcopy(dict(proposal))
    container: Mapping[str, Any] | None = None
    if isinstance(artifact.get("faithful_ir"), Mapping):
        container = artifact
    elif isinstance(artifact.get("semantic"), Mapping) and isinstance(
        artifact["semantic"].get("faithful_ir"), Mapping
    ):
        container = artifact["semantic"]
    if container is None:
        return CanonicalizedArtifact(artifact)
    ir = dict(container["faithful_ir"])
    ir["schema_version"] = FAITHFUL_IR_SCHEMA_VERSION
    ir["source_text"] = source_text
    ir["source_hash"] = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    ir["contract_version"] = FAITHFUL_IR_CONTRACT_VERSION
    ir["canonicalizer_version"] = FAITHFUL_IR_CANONICALIZER_VERSION
    ir["validator_version"] = FAITHFUL_IR_VALIDATOR_VERSION
    ir["node_contract_fingerprint"] = node_contract_fingerprint()
    ir["temporal_language_contract_version"] = (
        TEMPORAL_LANGUAGE_CONTRACT_VERSION
    )
    ir["ap_context_fingerprint"] = context.context_fingerprint
    ir["ap_profile_mode"] = context.profile_mode
    ir["catalog_authority"] = context.catalog_authority
    grounding = context.grounding_contract
    if grounding is not None:
        ir["ap_grounding_fingerprint"] = grounding.fingerprint
        ir["ap_grounding_authority"] = grounding.authority
    if requirement_id:
        ir["requirement_id"] = requirement_id
    # Raw structural preflight must inspect the backend-selected current schema,
    # not a model-omitted or stale schema field in the detached input mapping.
    container["faithful_ir"] = ir
    graph_problems = raw_ir_graph_integrity_problems(artifact)
    issues: list[ValidationIssue] = list(
        _graph_integrity_validation_issues(graph_problems, artifact)
    )

    ap_name_map: dict[str, str] = {}
    aps = ir.get("atomic_propositions")
    if isinstance(aps, list):
        canonical_proposals: list[dict[str, Any]] = []
        for index, raw_ap in enumerate(aps):
            if not isinstance(raw_ap, Mapping):
                continue
            ap = dict(raw_ap)
            ap["ap_id"] = f"ap-{index + 1}"
            _canonicalize_ir_evidence(
                ap,
                source_text=source_text,
                segments=segments,
                stage="semantic",
                path=json_pointer("faithful_ir", "atomic_propositions", index),
                issues=issues,
                segment_field="supporting_segment_ids",
                ap_name=str(ap.get("name", "")),
            )
            old_name = str(ap.get("name", ""))
            grounded_identity = (
                identity_for_evidence(
                    grounding,
                    str(ap.get("evidence_text", "")),
                    _span(ap.get("source_span")),
                )
                if grounding is not None
                else None
            )
            if old_name and grounded_identity is not None:
                ap["name"] = grounded_identity.canonical_name
                ap["ap_identity_id"] = grounded_identity.identity_key
                ap["ap_mention_ids"] = [
                    item.mention_id for item in grounded_identity.mentions
                ]
                ap_name_map[old_name] = grounded_identity.canonical_name
            canonical_proposals.append(ap)
        if grounding is not None:
            canonical_proposals, materialized_map, ap_issues = (
                _materialize_grounded_ap_declarations(
                    canonical_proposals,
                    grounding,
                )
            )
            ap_name_map.update(materialized_map)
            issues.extend(ap_issues)
        aps = canonical_proposals
        ir["atomic_propositions"] = aps

    nodes = ir.get("nodes")
    if isinstance(nodes, list):
        id_map: dict[str, str] = {}
        can_rewrite_ids = not graph_problems
        if can_rewrite_ids:
            for index, raw_node in enumerate(nodes):
                if isinstance(raw_node, Mapping):
                    id_map[str(raw_node.get("node_id", ""))] = f"ir-n{index + 1}"
        for index, raw_node in enumerate(nodes):
            if not isinstance(raw_node, Mapping):
                continue
            node = dict(raw_node)
            if can_rewrite_ids:
                node["node_id"] = f"ir-n{index + 1}"
            if can_rewrite_ids and isinstance(node.get("children"), list):
                node["children"] = [id_map.get(str(item), str(item)) for item in node["children"]]
            _rewrite_ap_condition_name(node, ap_name_map)
            _canonicalize_ir_evidence(
                node,
                source_text=source_text,
                segments=segments,
                stage="semantic",
                path=json_pointer("faithful_ir", "nodes", index),
                issues=issues,
                segment_field="supporting_segment_ids",
                ap_name=_ap_condition_name(node),
            )
            if grounding is not None and node.get("node_type") == "ap_condition":
                if (
                    isinstance(aps, list)
                    and grounding.authority != AUTHORITY_EXACT
                ):
                    _materialize_proposal_ap_from_node(
                        node,
                        grounding=grounding,
                        declarations=aps,
                        issues=issues,
                        path=json_pointer("faithful_ir", "nodes", index),
                    )
                _bind_ap_condition_to_grounding(
                    node,
                    grounding=grounding,
                    declarations=aps if isinstance(aps, list) else (),
                    path=json_pointer("faithful_ir", "nodes", index),
                    issues=issues,
                )
            nodes[index] = node
        if isinstance(aps, list):
            for ap_index, declaration in enumerate(aps, start=1):
                declaration["ap_id"] = f"ap-{ap_index}"
            ir["atomic_propositions"] = aps
        if can_rewrite_ids:
            ir["root_node_id"] = id_map.get(
                str(ir.get("root_node_id", "")),
                "ir-n1" if nodes else "",
            )
        if policy_target_request is not None and can_rewrite_ids:
            target_path = (
                "/semantic/semantic_policy_target"
                if container is not artifact
                else "/semantic_policy_target"
            )
            raw_target = container.get("semantic_policy_target")
            if not isinstance(raw_target, Mapping):
                raw_target = artifact.get("semantic_policy_target")
            if not isinstance(raw_target, Mapping):
                issues.append(
                    ValidationIssue(
                        code="SEMANTIC_POLICY_TARGET_REFERENCE_REQUIRED",
                        stage="semantic",
                        path=target_path,
                        observed=None,
                        expected={
                            "request_id": policy_target_request.request_id,
                            "node_id": "temporary node id from this response",
                        },
                        repair_class=RepairClass.CONSTRAINED,
                        allowed_paths=(target_path,),
                        message="policy materialization requires same-response target reference",
                    )
                )
            else:
                raw_request_id = str(raw_target.get("request_id", ""))
                raw_node_id = str(raw_target.get("node_id", ""))
                if raw_request_id != policy_target_request.request_id:
                    issues.append(
                        ValidationIssue(
                            code="SEMANTIC_POLICY_TARGET_REQUEST_ID_MISMATCH",
                            stage="semantic",
                            path=target_path + "/request_id",
                            observed=raw_request_id,
                            expected=policy_target_request.request_id,
                            repair_class=RepairClass.CONSTRAINED,
                            allowed_paths=(target_path,),
                            message="policy target reference request_id is not backend-provided value",
                        )
                    )
                elif raw_node_id not in id_map:
                    issues.append(
                        ValidationIssue(
                            code="SEMANTIC_POLICY_TARGET_RAW_NODE_UNKNOWN",
                            stage="semantic",
                            path=target_path + "/node_id",
                            observed=raw_node_id,
                            expected={"same_response_node_ids": sorted(id_map)},
                            repair_class=RepairClass.CONSTRAINED,
                            allowed_paths=(target_path,),
                            message="policy target must reference a node in the same raw graph",
                        )
                    )
                else:
                    canonical_target = next(
                        (
                            item
                            for item in nodes
                            if isinstance(item, Mapping)
                            and str(item.get("node_id", ""))
                            == id_map[raw_node_id]
                        ),
                        None,
                    )
                    container["semantic_policy_target"] = {
                        "request_id": policy_target_request.request_id,
                        "node_id": id_map[raw_node_id],
                        "evidence_text": (
                            str(canonical_target.get("evidence_text", ""))
                            if isinstance(canonical_target, Mapping)
                            else ""
                        ),
                        "source_span": (
                            list(_span(canonical_target.get("source_span")))
                            if isinstance(canonical_target, Mapping)
                            else [-1, -1]
                        ),
                        "supporting_segment_ids": (
                            list(
                                canonical_target.get(
                                    "supporting_segment_ids",
                                    (),
                                )
                            )
                            if isinstance(canonical_target, Mapping)
                            else []
                        ),
                    }
        ambiguities = ir.get("ambiguities")
        if isinstance(ambiguities, list):
            reviewed_by_id = {
                _reviewed_axis_value(item, "ambiguity_id"): item
                for item in reviewed_ambiguities
                if _reviewed_axis_value(item, "ambiguity_id")
            }
            for index, raw_ambiguity in enumerate(ambiguities):
                if not isinstance(raw_ambiguity, Mapping):
                    continue
                ambiguity = dict(raw_ambiguity)
                proposed_id = str(ambiguity.get("ambiguity_id", ""))
                reviewed = reviewed_by_id.get(proposed_id)
                if reviewed is not None:
                    options = _reviewed_axis_value(reviewed, "options", ())
                    ambiguity["ambiguity_id"] = proposed_id
                    ambiguity["description"] = _reviewed_axis_value(
                        reviewed, "question"
                    )
                    ambiguity["alternatives"] = [
                        _reviewed_axis_value(option, "plain_interpretation")
                        for option in options
                    ]
                    ambiguity["resolved"] = (
                        _reviewed_axis_value(reviewed, "decision_status")
                        == "reviewer_selected"
                    )
                    ambiguity["origin_layer"] = "plain_reading"
                    ambiguity["related_axis_ids"] = [proposed_id]
                    ambiguity["relation_kind"] = "same_axis"
                else:
                    # New IR-layer axes own a semantic, order-independent id.
                    # The model-proposed id and array position are not identity.
                    ambiguity.pop("ambiguity_id", None)
                    ambiguity["origin_layer"] = "ir_extraction"
                    ambiguity["relation_kind"] = "independent"
                    ambiguity["related_axis_ids"] = []
                    ambiguity["resolved"] = False
                if can_rewrite_ids and isinstance(ambiguity.get("affected_node_ids"), list):
                    ambiguity["affected_node_ids"] = [
                        id_map.get(str(item), str(item))
                        for item in ambiguity["affected_node_ids"]
                    ]
                raw_patches = ambiguity.get("alternative_patches")
                if can_rewrite_ids and isinstance(raw_patches, list):
                    ambiguity["alternative_patches"] = [
                        {
                            **dict(raw_patch),
                            "node_id": id_map.get(
                                str(raw_patch.get("node_id", "")),
                                str(raw_patch.get("node_id", "")),
                            ),
                        }
                        if isinstance(raw_patch, Mapping)
                        else raw_patch
                        for raw_patch in raw_patches
                    ]
                if reviewed is None:
                    ambiguity["ambiguity_id"] = _stable_ir_ambiguity_id(
                        ambiguity,
                        nodes=nodes,
                        source_text=source_text,
                    )
                ambiguities[index] = ambiguity

    issues.extend(
        _ap_identity_integrity_issues(
            ir,
            context=context,
        )
    )

    container["faithful_ir"] = ir
    return CanonicalizedArtifact(artifact, tuple(issues))


def _unique_authorized_identity_in_evidence(
    evidence_text: str,
    catalog: Sequence[str],
    alias_groups: Sequence[Sequence[str]],
) -> str:
    exact = _unique_symbol_in_evidence(evidence_text, catalog)
    if exact:
        return exact
    normalized = _identity_tokens(evidence_text)
    matches: list[str] = []
    for index, group in enumerate(alias_groups):
        if index >= len(catalog):
            break
        if any(
            alias and _identity_tokens(alias) <= normalized
            for alias in group
        ):
            matches.append(catalog[index])
    return matches[0] if len(set(matches)) == 1 else ""


def _ap_identity_integrity_issues(
    ir: Mapping[str, Any],
    *,
    context: APBindingContext,
) -> tuple[ValidationIssue, ...]:
    """Reject unsupported many-evidence-to-one-identity collapses."""

    if context.catalog_authority != CATALOG_AUTHORITY_NONE:
        return ()
    raw_nodes = ir.get("nodes")
    raw_aps = ir.get("atomic_propositions")
    if not isinstance(raw_nodes, list) or not isinstance(raw_aps, list):
        return ()
    declaration_evidence = {
        str(item.get("name", "")): str(item.get("evidence_text", ""))
        for item in raw_aps
        if isinstance(item, Mapping)
    }
    uses: dict[str, list[dict[str, object]]] = {}
    for index, item in enumerate(raw_nodes):
        if not isinstance(item, Mapping) or str(item.get("node_type", "")) != "ap_condition":
            continue
        ap_name = _ap_condition_name(item)
        if not ap_name:
            continue
        evidence = str(item.get("evidence_text", ""))
        uses.setdefault(ap_name, []).append(
            {
                "node_index": index,
                "node_id": str(item.get("node_id", "")),
                "semantic_role": str(item.get("semantic_role", "")),
                "evidence_text": evidence,
                "tokens": _identity_tokens(evidence),
            }
        )

    findings: list[ValidationIssue] = []
    for ap_name, nodes in uses.items():
        if len(nodes) < 2:
            continue
        declaration_tokens = _identity_tokens(declaration_evidence.get(ap_name, ""))
        name_tokens = _identity_tokens(ap_name.replace("_", " "))
        incompatible: list[dict[str, object]] = []
        for node in nodes:
            tokens = set(node["tokens"])
            supported = bool(
                (declaration_tokens and _token_sets_related(tokens, declaration_tokens))
                or (name_tokens and _token_sets_related(tokens, name_tokens))
            )
            if not supported:
                incompatible.append(node)
        distinct_groups = {
            tuple(sorted(set(node["tokens"])))
            for node in nodes
            if node["tokens"]
        }
        roles = {
            str(node["semantic_role"]).strip().casefold()
            for node in nodes
            if str(node["semantic_role"]).strip()
        }
        pairwise_unrelated = any(
            not _token_sets_related(set(left), set(right))
            for left in distinct_groups
            for right in distinct_groups
            if left != right
        )
        if incompatible and len(distinct_groups) >= 2 and (len(roles) >= 2 or pairwise_unrelated):
            findings.append(
                ValidationIssue(
                    code="IR_AP_IDENTITY_COLLAPSE",
                    stage="semantic",
                    path="/faithful_ir/nodes",
                    observed={
                        "ap_name": ap_name,
                        "node_evidence_groups": [
                            {
                                "node_id": node["node_id"],
                                "semantic_role": node["semantic_role"],
                                "evidence_text": node["evidence_text"],
                            }
                            for node in nodes
                        ],
                    },
                    expected={
                        "identity_basis": (
                            "same grounded predicate, declared catalog, approved "
                            "inventory alias, or confirmed identity mapping"
                        )
                    },
                    context={
                        "catalog_authority": context.catalog_authority,
                        "ap_profile_mode": context.profile_mode,
                        "ap_context_fingerprint": context.context_fingerprint,
                        "identity_delta": {
                            "many_to_one_identity": ap_name,
                            "evidence_group_count": len(distinct_groups),
                            "node_reference_count": len(nodes),
                        },
                    },
                    repair_class=RepairClass.HUMAN_REQUIRED,
                    allowed_paths=(),
                    message=(
                        f"AP {ap_name!r} merges distinct source predicates without "
                        "catalog, approved alias, or human mapping authority"
                    ),
                )
            )
    return tuple(findings)


_IDENTITY_STOPWORDS = frozenset(
    {
        "an", "the", "is", "are", "be", "been", "being", "shall",
        "should", "must", "will", "when", "whenever", "if", "then", "there",
        "holds", "hold", "occur", "occurs", "occurred", "happen", "happens",
        "mode",  # roles/evidence still distinguish domain predicates around it
    }
)


def _identity_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", value.casefold())
        if token not in _IDENTITY_STOPWORDS and len(token) > 1
    }


def _token_sets_related(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    overlap = left & right
    return bool(overlap) and (
        len(overlap) / min(len(left), len(right)) >= 0.5
        or left <= right
        or right <= left
    )


def _stable_ir_ambiguity_id(
    ambiguity: Mapping[str, Any],
    *,
    nodes: Sequence[object],
    source_text: str,
) -> str:
    """Return a layer-qualified identity independent of proposal ordering."""

    node_by_id = {
        str(node.get("node_id", "")): node
        for node in nodes
        if isinstance(node, Mapping)
    }
    loci = []
    for node_id in ambiguity.get("affected_node_ids", ()):
        node = node_by_id.get(str(node_id), {})
        loci.append({
            "node_type": str(node.get("node_type", "")),
            "semantic_role": str(node.get("semantic_role", "")),
            "evidence_text": " ".join(
                str(node.get("evidence_text", "")).split()
            ).casefold(),
            "source_span": list(node.get("source_span", ())),
        })
    patches = []
    for raw_patch in ambiguity.get("alternative_patches", ()):
        if not isinstance(raw_patch, Mapping):
            continue
        patches.append({
            "alternative_id": str(raw_patch.get("alternative_id", "")),
            "node_type": str(raw_patch.get("node_type", "")),
            "parameter_updates": raw_patch.get("parameter_updates", {}),
        })
    payload = {
        "layer": "ir_extraction",
        "source_hash": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "description": " ".join(
            str(ambiguity.get("description", "")).split()
        ).casefold(),
        "alternatives": sorted(
            " ".join(str(item).split()).casefold()
            for item in ambiguity.get("alternatives", ())
        ),
        "loci": sorted(
            loci,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        ),
        "patches": sorted(
            patches,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        ),
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return "amb-ir-" + digest


def _reviewed_axis_value(
    value: object,
    field_name: str,
    default: Any = "",
) -> Any:
    if isinstance(value, Mapping):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _graph_integrity_validation_issues(
    problems: Sequence[GraphIntegrityProblem],
    proposal: object,
) -> tuple[ValidationIssue, ...]:
    _ir, prefix = faithful_ir_proposal_mapping(proposal)
    allowed_paths = (
        prefix + "/nodes",
        prefix + "/root_node_id",
    ) if prefix else ()
    return tuple(
        ValidationIssue(
            code=problem.code,
            stage="semantic",
            path=problem.path,
            observed=dict(problem.context),
            expected={"well_formed_temporary_ir_graph": True},
            context={**dict(problem.context), "evidence_independent": True},
            repair_class=(
                RepairClass.DETERMINISTIC
                if (
                    problem.code == "IR_NODE_ARITY_INVALID"
                    and problem.context.get("node_type")
                    in {"conjunction", "disjunction"}
                    and problem.context.get("child_count") == 1
                )
                else RepairClass.SEMANTIC_PATCH
            ),
            allowed_paths=allowed_paths,
            message=problem.message,
            severity=Severity.ERROR,
        )
        for problem in problems
    )


def _canonicalize_ir_evidence(
    item: dict[str, Any],
    *,
    source_text: str,
    segments: Sequence[object],
    stage: str,
    path: str,
    issues: list[ValidationIssue],
    segment_field: str,
    ap_name: str = "",
) -> None:
    quote = item.get("evidence_quote", item.get("evidence_text", ""))
    quote_text = str(quote) if quote is not None else ""
    whole_token = bool(
        ap_name
        and quote_text == ap_name
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ap_name)
    )
    location = locate_evidence_quote(
        source_text,
        segments,
        quote_text,
        source_segment_ids=item.get(segment_field, ()),
        occurrence_index=item.get("occurrence_index"),
        proposed_span=item.get("source_span"),
        whole_token=whole_token,
        stage=stage,
        path=path + "/evidence_quote",
    )
    if isinstance(location, ValidationIssue):
        issues.append(location)
        item["source_span"] = [-1, -1]
        item["evidence_text"] = ""
    else:
        item["evidence_quote"] = location.evidence_text
        item["evidence_text"] = location.evidence_text
        item["source_span"] = list(location.source_span)
        if segments:
            item[segment_field] = list(location.source_segment_ids)
        if "occurrence_index" in item or location.occurrence_index:
            item["occurrence_index"] = location.occurrence_index


def _validated_span_occurrence(
    source_text: str,
    quote: str,
    proposed_span: object,
    *,
    candidates: Sequence[tuple[int, int]] | None = None,
    whole_token: bool = False,
) -> int | None:
    """Use an exact span only as a hint within the active candidate window."""

    if (
        not quote
        or not isinstance(proposed_span, (list, tuple))
        or len(proposed_span) != 2
        or not all(isinstance(item, int) for item in proposed_span)
    ):
        return None
    start, end = proposed_span
    if not (0 <= start < end <= len(source_text)) or source_text[start:end] != quote:
        return None
    active_candidates = list(
        candidates
        if candidates is not None
        else _quote_candidates(source_text, quote, whole_token=whole_token)
    )
    try:
        return active_candidates.index((start, end))
    except ValueError:
        return None


def _quote_candidates(
    source_text: str,
    quote: str,
    *,
    whole_token: bool,
) -> tuple[tuple[int, int], ...]:
    pattern = re.escape(quote)
    if whole_token:
        pattern = rf"(?<![A-Za-z0-9_]){pattern}(?![A-Za-z0-9_])"
    return tuple(
        (match.start(), match.end())
        for match in re.finditer(pattern, source_text)
    )


def _ap_condition_name(item: Mapping[str, Any]) -> str:
    if str(item.get("node_type", "")).strip().lower() != "ap_condition":
        return ""
    parameters = item.get("parameters", {})
    if isinstance(parameters, Mapping):
        return str(parameters.get("ap_name", ""))
    if isinstance(parameters, (list, tuple)):
        for pair in parameters:
            if (
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and str(pair[0]) == "ap_name"
            ):
                return str(pair[1])
    return ""


def _unique_symbol_in_evidence(
    evidence_text: str,
    catalog: tuple[str, ...],
) -> str:
    matches = tuple(
        symbol
        for symbol in catalog
        if re.search(
            rf"(?<![A-Za-z0-9_']){re.escape(symbol)}(?![A-Za-z0-9_'])",
            evidence_text,
        )
    )
    return matches[0] if len(matches) == 1 else ""


def _rewrite_ap_condition_name(
    node: dict[str, Any],
    name_map: Mapping[str, str],
) -> None:
    if str(node.get("node_type", "")).strip().lower() != "ap_condition":
        return
    parameters = node.get("parameters")
    if isinstance(parameters, Mapping):
        updated = dict(parameters)
        old_name = str(updated.get("ap_name", ""))
        if old_name in name_map:
            updated["ap_name"] = name_map[old_name]
        node["parameters"] = updated
    elif isinstance(parameters, list):
        node["parameters"] = [
            [pair[0], name_map.get(str(pair[1]), pair[1])]
            if isinstance(pair, (list, tuple))
            and len(pair) == 2
            and str(pair[0]) == "ap_name"
            else pair
            for pair in parameters
        ]


def _materialize_grounded_ap_declarations(
    proposals: Sequence[Mapping[str, Any]],
    grounding: APGroundingContract,
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    tuple[ValidationIssue, ...],
]:
    """Materialize backend-owned AP declarations from one grounding contract."""

    remaining = [dict(item) for item in proposals]
    result: list[dict[str, Any]] = []
    name_map: dict[str, str] = {}
    issues: list[ValidationIssue] = []

    for identity in grounding.identities:
        matched: list[dict[str, Any]] = []
        for proposal in tuple(remaining):
            if (
                str(proposal.get("ap_identity_id", "")) == identity.identity_key
                or str(proposal.get("name", "")).casefold()
                == identity.canonical_name.casefold()
                or identity_for_evidence(
                    grounding,
                    str(proposal.get("evidence_text", "")),
                    _span(proposal.get("source_span")),
                )
                == identity
            ):
                matched.append(proposal)
                remaining.remove(proposal)
        for proposal in matched:
            old_name = str(proposal.get("name", ""))
            if old_name:
                name_map[old_name] = identity.canonical_name
        declarations = matched or [{}]
        for raw_declaration in declarations:
            declaration = dict(raw_declaration)
            mention = _best_grounding_mention(identity, declaration)
            if mention is not None:
                declaration["source_span"] = list(mention.source_span)
                declaration["evidence_text"] = mention.evidence_text
                declaration["supporting_segment_ids"] = list(
                    mention.supporting_segment_ids
                )
            elif not declaration.get("evidence_text"):
                issues.append(
                    ValidationIssue(
                        code="IR_AP_GROUNDING_EVIDENCE_REQUIRED",
                        stage="semantic",
                        path="/faithful_ir/atomic_propositions",
                        observed={"identity_key": identity.identity_key},
                        expected={"exact_source_evidence": True},
                        context={
                            "grounding_fingerprint": grounding.fingerprint,
                            "authority": grounding.authority,
                        },
                        repair_class=RepairClass.HUMAN_REQUIRED,
                        allowed_paths=(),
                        message=(
                            f"AP identity {identity.canonical_name!r} has no exact "
                            "source evidence binding"
                        ),
                    )
                )
            declaration.update(
                {
                    "name": identity.canonical_name,
                    "ap_identity_id": identity.identity_key,
                    "ap_mention_ids": [
                        item.mention_id for item in identity.mentions
                    ],
                    "aliases": list(
                        dict.fromkeys(
                            (
                                *declaration.get("aliases", ()),
                                *identity.aliases,
                            )
                        )
                    ),
                }
            )
            result.append(declaration)

    if grounding.authority != AUTHORITY_EXACT:
        for proposal in remaining:
            evidence_text = str(proposal.get("evidence_text", ""))
            span = _span(proposal.get("source_span"))
            try:
                identity = proposal_identity(
                    grounding,
                    canonical_name=str(proposal.get("name", "")),
                    evidence_text=evidence_text,
                    source_span=span,
                    supporting_segment_ids=tuple(
                        str(item)
                        for item in proposal.get(
                            "supporting_segment_ids",
                            (),
                        )
                    ),
                )
            except Exception as exc:
                if span == (-1, -1):
                    result.append(proposal)
                    continue
                issues.append(
                    ValidationIssue(
                        code=getattr(
                            exc,
                            "code",
                            "IR_AP_EVIDENCE_IDENTITY_MISMATCH",
                        ),
                        stage="semantic",
                        path="/faithful_ir/atomic_propositions",
                        observed={
                            "name": proposal.get("name"),
                            "evidence_text": evidence_text,
                            "source_span": list(span),
                        },
                        expected={"backend_bound_identity": True},
                        context={
                            "grounding_fingerprint": grounding.fingerprint,
                            "authority": grounding.authority,
                        },
                        repair_class=RepairClass.HUMAN_REQUIRED,
                        allowed_paths=(),
                        message=str(exc),
                    )
                )
                continue
            old_name = str(proposal.get("name", ""))
            proposal["name"] = identity.canonical_name
            proposal["ap_identity_id"] = identity.identity_key
            proposal["ap_mention_ids"] = [
                item.mention_id for item in identity.mentions
            ]
            if old_name:
                name_map[old_name] = identity.canonical_name
            result.append(proposal)

    for index, declaration in enumerate(result, start=1):
        declaration["ap_id"] = f"ap-{index}"

    return result, name_map, tuple(issues)


def _best_grounding_mention(
    identity: APGroundingIdentity,
    proposal: Mapping[str, Any],
):
    proposed_span = _span(proposal.get("source_span"))
    for mention in identity.mentions:
        if mention.source_span == proposed_span:
            return mention
    return identity.mentions[0] if identity.mentions else None


def _bind_ap_condition_to_grounding(
    node: dict[str, Any],
    *,
    grounding: APGroundingContract,
    declarations: Sequence[Mapping[str, Any]],
    path: str,
    issues: list[ValidationIssue],
) -> None:
    evidence_text = str(node.get("evidence_text", ""))
    span = _span(node.get("source_span"))
    if span == (-1, -1):
        return
    identity = identity_for_evidence(grounding, evidence_text, span)
    declaration_by_identity = {
        str(item.get("ap_identity_id", "")): item
        for item in declarations
        if str(item.get("ap_identity_id", ""))
    }
    declaration_by_name = {
        str(item.get("name", "")): item
        for item in declarations
        if str(item.get("name", ""))
    }
    if identity is not None and identity.identity_key in declaration_by_identity:
        _set_ap_condition_name(node, identity.canonical_name)
        return

    current_name = _ap_condition_name(node)
    declaration = declaration_by_name.get(current_name)
    if declaration is not None:
        declaration_evidence = str(declaration.get("evidence_text", ""))
        declaration_span = _span(declaration.get("source_span"))
        if declaration_span == (-1, -1):
            return
        if (
            declaration_span == span
            or (
                declaration_evidence
                and (
                    declaration_evidence.casefold() in evidence_text.casefold()
                    or evidence_text.casefold() in declaration_evidence.casefold()
                )
            )
        ):
            return

    issues.append(
        ValidationIssue(
            code="IR_AP_NODE_IDENTITY_MISMATCH",
            stage="semantic",
            path=path + "/parameters/ap_name",
            observed={
                "ap_name": current_name,
                "evidence_text": evidence_text,
                "source_span": list(span),
            },
            expected={
                "declaration_identity_ids": sorted(
                    declaration_by_identity
                ),
            },
            context={
                "grounding_fingerprint": grounding.fingerprint,
                "authority": grounding.authority,
            },
            repair_class=RepairClass.HUMAN_REQUIRED,
            allowed_paths=(),
            message=(
                "ap_condition evidence does not prove the referenced AP identity"
            ),
        )
    )


def _materialize_proposal_ap_from_node(
    node: dict[str, Any],
    *,
    grounding: APGroundingContract,
    declarations: list[dict[str, Any]],
    issues: list[ValidationIssue],
    path: str,
) -> None:
    """Create a backend-owned proposal declaration from exact node evidence.

    In ``proposal_only``/``required_subset`` mode the LLM owns neither stable
    AP identity nor the declaration array.  A node may therefore be the first
    place where a grounded phrase is proposed.  Materialize that declaration
    here so declaration—mention—node binding is still mechanical.
    """

    current_name = _ap_condition_name(node)
    if not current_name or any(
        str(item.get("name", "")) == current_name
        for item in declarations
    ):
        return
    evidence_text = str(node.get("evidence_text", ""))
    span = _span(node.get("source_span"))
    if span == (-1, -1):
        return
    try:
        identity = proposal_identity(
            grounding,
            canonical_name=current_name,
            evidence_text=evidence_text,
            source_span=span,
            supporting_segment_ids=tuple(
                str(item)
                for item in node.get("supporting_segment_ids", ())
            ),
        )
    except Exception as exc:
        issues.append(
            ValidationIssue(
                code=getattr(
                    exc,
                    "code",
                    "IR_AP_EVIDENCE_IDENTITY_MISMATCH",
                ),
                stage="semantic",
                path=path + "/parameters/ap_name",
                observed={
                    "ap_name": current_name,
                    "evidence_text": evidence_text,
                    "source_span": list(span),
                },
                expected={"backend_bound_identity": True},
                context={
                    "grounding_fingerprint": grounding.fingerprint,
                    "authority": grounding.authority,
                },
                repair_class=RepairClass.HUMAN_REQUIRED,
                allowed_paths=(),
                message=str(exc),
            )
        )
        return
    declaration = {
        "ap_id": f"ap-{len(declarations) + 1}",
        "name": identity.canonical_name,
        "ap_identity_id": identity.identity_key,
        "ap_mention_ids": [
            mention.mention_id for mention in identity.mentions
        ],
        "aliases": list(identity.aliases),
        "evidence_quote": evidence_text,
        "occurrence_index": int(node.get("occurrence_index", 0) or 0),
        "source_span": list(span),
        "evidence_text": evidence_text,
        "supporting_segment_ids": list(
            node.get("supporting_segment_ids", ())
        ),
    }
    declarations.append(declaration)
    _set_ap_condition_name(node, identity.canonical_name)


def _set_ap_condition_name(node: dict[str, Any], name: str) -> None:
    parameters = node.get("parameters")
    if isinstance(parameters, Mapping):
        updated = dict(parameters)
        updated["ap_name"] = name
        node["parameters"] = updated
        return
    if isinstance(parameters, list):
        found = False
        updated_items: list[Any] = []
        for item in parameters:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and str(item[0]) == "ap_name"
            ):
                updated_items.append([item[0], name])
                found = True
            else:
                updated_items.append(item)
        if not found:
            updated_items.append(["ap_name", name])
        node["parameters"] = updated_items


def _span(value: object) -> tuple[int, int]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        return int(value[0]), int(value[1])
    return (-1, -1)


def _segment_view(segment: object) -> tuple[str, int, int, int, str]:
    if isinstance(segment, Mapping):
        return (
            str(segment.get("segment_id", "")),
            int(segment.get("index", 0)),
            int(segment.get("start", 0)),
            int(segment.get("end", 0)),
            str(segment.get("text", "")),
        )
    return (
        str(getattr(segment, "segment_id")),
        int(getattr(segment, "index", 0)),
        int(getattr(segment, "start")),
        int(getattr(segment, "end")),
        str(getattr(segment, "text")),
    )


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


__all__ = [
    "EvidenceLocation",
    "canonicalize_faithful_ir_proposal",
    "canonicalize_plain_reading_proposal",
    "locate_evidence_quote",
]
