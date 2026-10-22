"""Authoritative policy-target request, reference, and binding contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from contract_versions import SEMANTIC_POLICY_BINDING_CONTRACT_VERSION
from semantic_node_contracts import NODE_CONTRACTS


class SemanticPolicyBindingError(ValueError):
    """A policy target could not be proved without a raw-ID guess."""


@dataclass(frozen=True)
class SemanticPolicyTargetRequest:
    request_id: str
    requirement_id: str
    source_hash: str
    segmentation_id: str
    reading_review_id: str
    axis_id: str
    expected_node_type: str
    unresolved_parameters: tuple[str, ...]
    controlled_parameter_names: tuple[str, ...]
    evidence_text: str
    source_span: tuple[int, int]
    supporting_segment_ids: tuple[str, ...]
    occurrence_index: int
    model_node_id_hint: str
    request_fingerprint: str
    parent_candidate_id: str = ""
    evidence_selector_fingerprint: str = ""
    parameter_contract_fingerprint: str = ""
    contract_version: str = SEMANTIC_POLICY_BINDING_CONTRACT_VERSION


@dataclass(frozen=True)
class SemanticPolicyTargetReference:
    request_id: str
    node_id: str
    evidence_text: str = ""
    source_span: tuple[int, int] = (-1, -1)
    supporting_segment_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticPolicyTargetBinding:
    binding_id: str
    request_id: str
    requirement_id: str
    source_hash: str
    segmentation_id: str
    reading_review_id: str
    axis_id: str
    node_id: str
    node_type: str
    evidence_text: str
    source_span: tuple[int, int]
    supporting_segment_ids: tuple[str, ...]
    controlled_parameter_names: tuple[str, ...]
    seed_ir_version: str
    structure_fingerprint: str
    binding_method: str
    binding_fingerprint: str
    parent_candidate_id: str = ""
    evidence_selector_fingerprint: str = ""
    parameter_contract_fingerprint: str = ""
    contract_version: str = SEMANTIC_POLICY_BINDING_CONTRACT_VERSION


def build_semantic_policy_target_request(
    raw_request: Mapping[str, object],
    *,
    requirement_id: str,
    source_hash: str,
    segmentation_id: str,
    reading_review_id: str,
    legal_options: Sequence[Mapping[str, object]],
    evidence_location: object,
) -> SemanticPolicyTargetRequest:
    if not legal_options:
        raise SemanticPolicyBindingError("policy target request has no legal options")
    axis_ids = {
        str(option.get("axis_id", "")).strip() for option in legal_options
    }
    node_types = {
        str(option.get("node_type", "")).strip() for option in legal_options
    }
    axis_ids.discard("")
    node_types.discard("")
    if len(axis_ids) != 1 or len(node_types) != 1:
        raise SemanticPolicyBindingError(
            "policy options do not establish one axis and node type"
        )
    axis_id = next(iter(axis_ids))
    node_type = next(iter(node_types))
    model_node_type = str(raw_request.get("node_type", "")).strip()
    if model_node_type and model_node_type != node_type:
        raise SemanticPolicyBindingError(
            "model policy node type conflicts with backend axis contract"
        )
    controlled_sets = [
        {
            str(value).strip()
            for value in option.get("controlled_parameters", ())
            if str(value).strip()
        }
        for option in legal_options
    ]
    controlled = set.intersection(*controlled_sets) if controlled_sets else set()
    unresolved = {
        str(value).strip()
        for value in raw_request.get("unresolved_parameters", ())
        if str(value).strip()
    }
    if not unresolved or not unresolved.issubset(controlled):
        raise SemanticPolicyBindingError(
            "policy unresolved parameters are outside one controlled axis"
        )
    evidence_text = str(getattr(evidence_location, "evidence_text", ""))
    source_span = tuple(getattr(evidence_location, "source_span", (-1, -1)))
    segments = tuple(getattr(evidence_location, "source_segment_ids", ()))
    occurrence = int(getattr(evidence_location, "occurrence_index", -1))
    if (
        not evidence_text
        or len(source_span) != 2
        or source_span[0] < 0
        or source_span[1] <= source_span[0]
        or occurrence < 0
    ):
        raise SemanticPolicyBindingError(
            "policy target request requires canonical source evidence"
        )
    authoritative = {
        "requirement_id": requirement_id,
        "source_hash": source_hash,
        "segmentation_id": segmentation_id,
        "reading_review_id": reading_review_id,
        "axis_id": axis_id,
        "expected_node_type": node_type,
        "unresolved_parameters": sorted(unresolved),
        "controlled_parameter_names": sorted(controlled),
        "evidence_text": evidence_text,
        "source_span": source_span,
        "supporting_segment_ids": segments,
        "occurrence_index": occurrence,
        "contract_version": SEMANTIC_POLICY_BINDING_CONTRACT_VERSION,
    }
    evidence_selector_fingerprint = _fingerprint(
        {
            "evidence_text": evidence_text,
            "source_span": source_span,
            "supporting_segment_ids": segments,
            "occurrence_index": occurrence,
        }
    )
    parameter_contract_fingerprint = _parameter_contract_fingerprint(
        node_type,
        tuple(sorted(controlled)),
    )
    parent_candidate_id = "SPSEED-" + _fingerprint(
        {
            "requirement_id": requirement_id,
            "source_hash": source_hash,
            "raw_request": dict(raw_request),
        }
    )[:20]
    authoritative.update(
        {
            "parent_candidate_id": parent_candidate_id,
            "evidence_selector_fingerprint": evidence_selector_fingerprint,
            "parameter_contract_fingerprint": parameter_contract_fingerprint,
        }
    )
    fingerprint = _fingerprint(authoritative)
    return SemanticPolicyTargetRequest(
        request_id="SPTR-" + fingerprint[:20],
        requirement_id=requirement_id,
        source_hash=source_hash,
        segmentation_id=segmentation_id,
        reading_review_id=reading_review_id,
        axis_id=axis_id,
        expected_node_type=node_type,
        unresolved_parameters=tuple(sorted(unresolved)),
        controlled_parameter_names=tuple(sorted(controlled)),
        evidence_text=evidence_text,
        source_span=(int(source_span[0]), int(source_span[1])),
        supporting_segment_ids=tuple(str(value) for value in segments),
        occurrence_index=occurrence,
        model_node_id_hint=str(raw_request.get("node_id", "")).strip(),
        request_fingerprint=fingerprint,
        parent_candidate_id=parent_candidate_id,
        evidence_selector_fingerprint=evidence_selector_fingerprint,
        parameter_contract_fingerprint=parameter_contract_fingerprint,
    )


def build_semantic_policy_target_binding(
    request: SemanticPolicyTargetRequest,
    reference: SemanticPolicyTargetReference,
    ir: object,
) -> SemanticPolicyTargetBinding:
    if reference.request_id != request.request_id:
        raise SemanticPolicyBindingError("policy target reference request_id mismatch")
    if str(getattr(ir, "requirement_id", "")) != request.requirement_id:
        raise SemanticPolicyBindingError("policy target requirement lineage mismatch")
    if str(getattr(ir, "source_hash", "")) != request.source_hash:
        raise SemanticPolicyBindingError("policy target source lineage mismatch")
    if request.segmentation_id and (
        str(getattr(ir, "segmentation_id", "")) != request.segmentation_id
    ):
        raise SemanticPolicyBindingError("policy target segmentation lineage mismatch")
    if request.reading_review_id and (
        str(getattr(ir, "reading_review_id", "")) != request.reading_review_id
    ):
        raise SemanticPolicyBindingError("policy target reading lineage mismatch")
    node_map = getattr(ir, "node_map", {})
    node = node_map.get(reference.node_id)
    if node is None:
        raise SemanticPolicyBindingError(
            "canonical policy target reference does not exist"
        )
    if str(getattr(node, "node_type", "")) != request.expected_node_type:
        raise SemanticPolicyBindingError("canonical policy target node type mismatch")
    request_segments = set(request.supporting_segment_ids)
    candidates = [
        candidate
        for candidate in node_map.values()
        if str(getattr(candidate, "node_type", ""))
        == request.expected_node_type
        and _spans_overlap(
            request.source_span,
            tuple(getattr(candidate, "source_span", (-1, -1))),
        )
        and (
            not request_segments
            or not set(
                getattr(candidate, "supporting_segment_ids", ())
            )
            or bool(
                request_segments.intersection(
                    getattr(candidate, "supporting_segment_ids", ())
                )
            )
        )
    ]
    if len(candidates) != 1:
        raise SemanticPolicyBindingError(
            "policy target evidence/type proof is not unique in the "
            "same canonical artifact"
        )
    if str(getattr(candidates[0], "node_id", "")) != reference.node_id:
        raise SemanticPolicyBindingError(
            "same-response policy target reference selects the wrong "
            "evidence-bound node"
        )
    contract = NODE_CONTRACTS.get(request.expected_node_type)
    if contract is None or not set(request.controlled_parameter_names).issubset(
        contract.parameters
    ):
        raise SemanticPolicyBindingError(
            "policy target node contract does not own controlled parameters"
        )
    node_span = tuple(getattr(node, "source_span", (-1, -1)))
    if not _spans_overlap(request.source_span, node_span):
        raise SemanticPolicyBindingError(
            "policy target evidence does not overlap canonical request evidence"
        )
    node_segments = set(getattr(node, "supporting_segment_ids", ()))
    if node_segments and request_segments and not node_segments.intersection(
        request_segments
    ):
        raise SemanticPolicyBindingError("policy target segment evidence mismatch")
    if (
        reference.evidence_text != str(getattr(node, "evidence_text", ""))
        or tuple(reference.source_span) != tuple(node_span)
        or tuple(reference.supporting_segment_ids)
        != tuple(getattr(node, "supporting_segment_ids", ()))
    ):
        raise SemanticPolicyBindingError(
            "policy target reference evidence is not from the same canonical artifact"
        )
    evidence_selector_fingerprint = _fingerprint(
        {
            "evidence_text": request.evidence_text,
            "source_span": request.source_span,
            "supporting_segment_ids": request.supporting_segment_ids,
            "occurrence_index": request.occurrence_index,
        }
    )
    if evidence_selector_fingerprint != request.evidence_selector_fingerprint:
        raise SemanticPolicyBindingError(
            "policy target request evidence selector is stale"
        )
    parameter_contract_fingerprint = _parameter_contract_fingerprint(
        request.expected_node_type,
        request.controlled_parameter_names,
    )
    if (
        parameter_contract_fingerprint
        != request.parameter_contract_fingerprint
    ):
        raise SemanticPolicyBindingError(
            "policy target parameter contract is stale"
        )
    structure = policy_target_structure_fingerprint(
        ir,
        node_id=reference.node_id,
        controlled=request.controlled_parameter_names,
    )
    payload = {
        "request_fingerprint": request.request_fingerprint,
        "node_id": reference.node_id,
        "node_type": request.expected_node_type,
        "seed_ir_version": str(getattr(ir, "ir_version", "")),
        "structure_fingerprint": structure,
        "binding_method": "same_artifact_reference",
        "parent_candidate_id": request.parent_candidate_id,
        "evidence_selector_fingerprint": evidence_selector_fingerprint,
        "parameter_contract_fingerprint": parameter_contract_fingerprint,
        "contract_version": SEMANTIC_POLICY_BINDING_CONTRACT_VERSION,
    }
    fingerprint = _fingerprint(payload)
    return SemanticPolicyTargetBinding(
        binding_id="SPTB-" + fingerprint[:20],
        request_id=request.request_id,
        requirement_id=request.requirement_id,
        source_hash=request.source_hash,
        segmentation_id=request.segmentation_id,
        reading_review_id=request.reading_review_id,
        axis_id=request.axis_id,
        node_id=reference.node_id,
        node_type=request.expected_node_type,
        evidence_text=str(getattr(node, "evidence_text", "")),
        source_span=(int(node_span[0]), int(node_span[1])),
        supporting_segment_ids=tuple(
            str(value) for value in getattr(node, "supporting_segment_ids", ())
        ),
        controlled_parameter_names=request.controlled_parameter_names,
        seed_ir_version=str(getattr(ir, "ir_version", "")),
        structure_fingerprint=structure,
        binding_method="same_artifact_reference",
        binding_fingerprint=fingerprint,
        parent_candidate_id=request.parent_candidate_id,
        evidence_selector_fingerprint=evidence_selector_fingerprint,
        parameter_contract_fingerprint=parameter_contract_fingerprint,
    )


def validate_semantic_policy_target_binding(
    binding: SemanticPolicyTargetBinding,
    ir: object,
    *,
    require_seed_version: bool = True,
) -> None:
    if binding.binding_method != "same_artifact_reference":
        raise SemanticPolicyBindingError("unsupported policy binding method")
    node = getattr(ir, "node_map", {}).get(binding.node_id)
    if node is None or str(getattr(node, "node_type", "")) != binding.node_type:
        raise SemanticPolicyBindingError("policy target binding node is stale")
    if (
        str(getattr(ir, "requirement_id", "")) != binding.requirement_id
        or str(getattr(ir, "source_hash", "")) != binding.source_hash
        or (
            binding.segmentation_id
            and str(getattr(ir, "segmentation_id", ""))
            != binding.segmentation_id
        )
        or (
            binding.reading_review_id
            and str(getattr(ir, "reading_review_id", ""))
            != binding.reading_review_id
        )
    ):
        raise SemanticPolicyBindingError("policy target binding lineage is stale")
    if (
        str(getattr(node, "evidence_text", "")) != binding.evidence_text
        or tuple(getattr(node, "source_span", (-1, -1)))
        != tuple(binding.source_span)
        or tuple(getattr(node, "supporting_segment_ids", ()))
        != tuple(binding.supporting_segment_ids)
    ):
        raise SemanticPolicyBindingError("policy target binding evidence is stale")
    expected_parameter_contract = _parameter_contract_fingerprint(
        binding.node_type,
        binding.controlled_parameter_names,
    )
    if expected_parameter_contract != binding.parameter_contract_fingerprint:
        raise SemanticPolicyBindingError(
            "policy target binding parameter contract is stale"
        )
    if (
        require_seed_version
        and str(getattr(ir, "ir_version", "")) != binding.seed_ir_version
    ):
        raise SemanticPolicyBindingError("policy target binding seed IR is stale")
    actual = policy_target_structure_fingerprint(
        ir,
        node_id=binding.node_id,
        controlled=binding.controlled_parameter_names,
    )
    if actual != binding.structure_fingerprint:
        raise SemanticPolicyBindingError("policy target structure fingerprint is stale")


def policy_target_structure_fingerprint(
    ir: object,
    *,
    node_id: str,
    controlled: Sequence[str],
) -> str:
    controlled_set = set(controlled)
    payload = {
        "requirement_id": getattr(ir, "requirement_id", ""),
        "source_hash": getattr(ir, "source_hash", ""),
        "root_node_id": getattr(ir, "root_node_id", ""),
        "atomic_propositions": [
            asdict(ap) for ap in getattr(ir, "atomic_propositions", ())
        ],
        "nodes": [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "children": list(node.children),
                "parameters": {
                    name: value
                    for name, value in node.parameters
                    if node.node_id != node_id or name not in controlled_set
                },
                "evidence_text": node.evidence_text,
                "source_span": list(node.source_span),
                "supporting_segment_ids": list(node.supporting_segment_ids),
            }
            for node in getattr(ir, "nodes", ())
        ],
    }
    return _fingerprint(payload)


def semantic_policy_target_request_payload(
    request: SemanticPolicyTargetRequest,
) -> dict[str, object]:
    payload = asdict(request)
    payload["model_node_id_hint"] = (
        request.model_node_id_hint + " (diagnostic only)"
        if request.model_node_id_hint
        else ""
    )
    return payload


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _parameter_contract_fingerprint(
    node_type: str,
    controlled: Sequence[str],
) -> str:
    contract = NODE_CONTRACTS.get(node_type)
    if contract is None:
        raise SemanticPolicyBindingError("policy target node contract is unknown")
    return _fingerprint(
        {
            "node_type": node_type,
            "controlled_parameters": sorted(str(item) for item in controlled),
            "parameter_contracts": {
                name: asdict(contract.parameters[name])
                for name in sorted(str(item) for item in controlled)
                if name in contract.parameters
            },
            "binding_contract_version": SEMANTIC_POLICY_BINDING_CONTRACT_VERSION,
        }
    )


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "SemanticPolicyBindingError",
    "SemanticPolicyTargetBinding",
    "SemanticPolicyTargetReference",
    "SemanticPolicyTargetRequest",
    "build_semantic_policy_target_binding",
    "build_semantic_policy_target_request",
    "policy_target_structure_fingerprint",
    "semantic_policy_target_request_payload",
    "validate_semantic_policy_target_binding",
]
