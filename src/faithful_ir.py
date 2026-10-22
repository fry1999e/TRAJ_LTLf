"""Faithful semantic IR parsing, validation, and rule-based construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Mapping

from ap_binding_context import APBindingContext

from contract_versions import (
    FAITHFUL_IR_CANONICALIZER_VERSION,
    FAITHFUL_IR_CONTRACT_VERSION,
    FAITHFUL_IR_SCHEMA_VERSION,
    FAITHFUL_IR_VALIDATOR_VERSION,
    TEMPORAL_LANGUAGE_CONTRACT_VERSION,
)
from relation_parameter_contracts import (
    PERIODICITY_PARAMETER_VALUES,
    periodicity_mode_contract,
)
from schemas import (
    AtomicProposition,
    FaithfulSemanticIR,
    IRProvenance,
    PlainReadingReview,
    SemanticAmbiguity,
    SemanticAlternativePatch,
    SemanticNode,
    SemanticStructure,
    SurfaceSegmentation,
)
from semantic_topology import (
    FUTURE_PROHIBITION_TOPOLOGY_CODE,
    FUTURE_PROHIBITION_TOPOLOGY_MESSAGE,
    FUTURE_PROHIBITION_TOPOLOGY_PATH,
    build_ir_semantic_view,
    faithful_ir_proposal_mapping,
    future_prohibition_binding_from_valid_graph,
    has_future_prohibition_topology,
    is_future_prohibition_source,
    scope_topology_problems,
)
from semantic_node_contracts import (
    NODE_CONTRACTS,
    node_contract_fingerprint,
    node_contract_findings,
    node_parameter_enum_values,
    required_node_parameter_names,
)
from temporal_resolution import temporal_range_errors
from temporal_language import temporal_node_grounding_errors


FORMULA_OUTPUT_KEYS = frozenset(
    {"formula", "formula_text", "candidate_formula", "final_formula"}
)

SUPPORTED_NODE_TYPES = frozenset(NODE_CONTRACTS)

TEMPORAL_NODE_TYPES = frozenset(
    {
        "global_constraint",
        "eventually",
        "eventually_strict",
        "next_step",
        "until",
        "release",
        "before",
        "after",
        "precedence",
        "ordered_sequence",
        "persist_to_end",
        "fixed_duration",
        "exact_run_length",
        "bounded_window",
        "cardinality",
        "recurrence",
        "periodicity",
        "start_boundary",
        "end_boundary",
        "short_trace_boundary",
    }
)

UNTIL_PARAMETER_VALUES = node_parameter_enum_values("until")
PRECEDENCE_PARAMETER_VALUES = node_parameter_enum_values("precedence")

PERIODICITY_REQUIRED_PARAMETERS = ("period", "mode", "origin", "terminal_policy")

_PERIODICITY_AMBIGUITY_PARAMETER_MARKERS = {
    "period": ("period", "interval", "frequency", "fifth"),
    "phase": (
        "phase",
        "index",
        "counting origin",
        "step origin",
        "first scheduled position",
        "fifth",
    ),
    "mode": ("mode", "spacing", "recurrence"),
    "off_schedule_policy": (
        "off_phase_policy",
        "off_schedule_policy",
        "off_phase",
        "off_schedule",
        "intermediate position",
        "between scheduled positions",
    ),
    "off_phase_policy": (
        "off_phase_policy",
        "off_schedule_policy",
        "off_phase",
        "off_schedule",
        "intermediate position",
        "between scheduled positions",
    ),
    "terminal_policy": (
        "terminal_policy",
        "terminal",
        "short trace",
        "finite prefix",
        "trace end",
        "end of trace",
    ),
}

BOUNDED_TEMPORAL_PARAMETER_VALUES = {
    node_type: node_parameter_enum_values(node_type)
    for node_type in (
        "fixed_duration", "exact_run_length", "bounded_window", "recurrence"
    )
}


@dataclass(frozen=True)
class FaithfulIRFinding:
    """A stable domain finding independent of generation orchestration."""

    code: str
    path: str
    message: str
    repairable: bool = True
    allowed_paths: tuple[str, ...] = ()


class FaithfulIRValidationError(ValueError):
    """Raised when an IR is lossy, disconnected, or not evidence-backed."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "FAITHFUL_IR_VALIDATION_FAILED",
        path: str = "/faithful_ir",
        repairable: bool = True,
        findings: tuple[FaithfulIRFinding, ...] = (),
    ):
        super().__init__(message)
        self.code = code
        self.path = path
        self.repairable = repairable
        self.findings = findings or (
            FaithfulIRFinding(
                code=code,
                path=path,
                message=message,
                repairable=repairable,
            ),
        )


class FaithfulIRTraceReadinessError(FaithfulIRValidationError):
    """Raised when a valid unresolved IR cannot yet construct trace queries."""

    suggested_rerun_stage = "semantic"

    def __init__(self, errors: tuple[str, ...], *, report=None):
        super().__init__(
            "; ".join(errors),
            code="IR_NOT_READY_FOR_TRACE_GENERATION",
            path="/faithful_ir/nodes",
        )
        self.diagnostics = {
            "suggested_rerun_stage": self.suggested_rerun_stage,
            "trace_readiness_errors": list(errors),
        }
        if report is not None:
            self.diagnostics.update(
                {
                    "readiness_contract_version": report.contract_version,
                    "readiness_fingerprint": report.fingerprint,
                    "missing_parameter_paths": list(report.missing_parameter_paths),
                    "unresolved_ambiguity_ids": list(
                        report.unresolved_ambiguity_ids
                    ),
                    "findings": [
                        {
                            "code": item.code,
                            "path": item.path,
                            "message": item.message,
                            "node_id": item.node_id,
                            "node_type": item.node_type,
                            "parameter_name": item.parameter_name,
                            "ambiguity_id": item.ambiguity_id,
                            "semantic_axis_ids": list(item.semantic_axis_ids),
                        }
                        for item in report.findings
                    ],
                }
            )


def reject_formula_fields(payload: Mapping[str, Any]) -> None:
    """Reject complete formula fields in online semantic-extraction output."""

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                if normalized in FORMULA_OUTPUT_KEYS:
                    raise FaithfulIRValidationError(
                        f"NLU output contains forbidden formula field: {path}{key}",
                        code="IR_FORBIDDEN_FORMULA_FIELD",
                        path="/" + (path + str(key)).replace(".", "/"),
                        repairable=False,
                    )
                walk(child, f"{path}{key}.")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}{index}.")

    walk(payload, "")


def faithful_ir_from_mapping(
    payload: Mapping[str, Any],
    *,
    source_text: str,
    requirement_id: str = "",
    extractor: str = "NLUAgent",
    trusted_provenance: bool = False,
    surface_segmentation: SurfaceSegmentation | None = None,
    plain_reading_review: PlainReadingReview | None = None,
    ap_binding_context: APBindingContext | None = None,
    working_interpretation: bool = False,
) -> FaithfulSemanticIR:
    reject_formula_fields(payload)
    ir_payload = payload.get("faithful_ir", payload)
    if not isinstance(ir_payload, Mapping):
        raise FaithfulIRValidationError("faithful_ir must be an object")
    declared_source = str(ir_payload.get("source_text", ""))
    if declared_source and declared_source != source_text:
        raise FaithfulIRValidationError(
            "faithful_ir.source_text must exactly match the current requirement"
        )

    aps = tuple(
        _repair_ap_evidence_span(
            _ap_from_mapping(item),
            source_text,
            surface_segmentation=surface_segmentation,
        )
        for item in _object_list(ir_payload, "atomic_propositions")
    )
    nodes = tuple(
        _repair_node_evidence_span(
            _node_from_mapping(item),
            source_text,
            surface_segmentation=surface_segmentation,
        )
        for item in _object_list(ir_payload, "nodes")
    )
    ambiguities = tuple(
        _ambiguity_from_mapping(item)
        for item in _object_list(ir_payload, "ambiguities", required=False)
    )
    provenance_payload = ir_payload.get("provenance", {})
    provenance = IRProvenance(
        extractor=str(
            provenance_payload.get("extractor", extractor)
            if isinstance(provenance_payload, Mapping)
            else extractor
        ),
        extraction_mode=str(
            provenance_payload.get("extraction_mode", "generated")
            if isinstance(provenance_payload, Mapping)
            else "generated"
        ),
        source=(
            str(provenance_payload.get("source", "natural_language"))
            if trusted_provenance and isinstance(provenance_payload, Mapping)
            else "natural_language"
        ),
        created_at=str(
            provenance_payload.get("created_at", "")
            if isinstance(provenance_payload, Mapping)
            else ""
        ),
        approved=(
            bool(provenance_payload.get("approved", False))
            if trusted_provenance and isinstance(provenance_payload, Mapping)
            else False
        ),
        approval_source=(
            str(provenance_payload.get("approval_source", ""))
            if trusted_provenance and isinstance(provenance_payload, Mapping)
            else ""
        ),
    )
    declared_segmentation_id = str(ir_payload.get("segmentation_id", ""))
    declared_reading_review_id = str(ir_payload.get("reading_review_id", ""))
    segmentation_id = _trusted_parent_id(
        declared_segmentation_id,
        (
            surface_segmentation.segmentation_id
            if surface_segmentation is not None
            else plain_reading_review.segmentation_id
            if plain_reading_review is not None
            else declared_segmentation_id
            if trusted_provenance
            else ""
        ),
        "segmentation_id",
    )
    reading_review_id = _trusted_parent_id(
        declared_reading_review_id,
        (
            plain_reading_review.reading_review_id
            if plain_reading_review
            else declared_reading_review_id
            if trusted_provenance
            else ""
        ),
        "reading_review_id",
    )
    reading_review_basis = (
        plain_reading_review.review_basis
        if plain_reading_review is not None
        else str(ir_payload.get("reading_review_basis", ""))
    )
    ir = FaithfulSemanticIR(
        requirement_id=str(ir_payload.get("requirement_id") or requirement_id or _requirement_id(source_text)),
        source_text=source_text,
        atomic_propositions=aps,
        root_node_id=str(ir_payload.get("root_node_id", "")),
        nodes=nodes,
        ambiguities=ambiguities,
        extraction_confidence=_float(ir_payload.get("extraction_confidence"), 0.0),
        provenance=provenance,
        schema_version=str(ir_payload.get("schema_version", FAITHFUL_IR_SCHEMA_VERSION)),
        ir_version=str(ir_payload.get("ir_version", "")),
        segmentation_id=segmentation_id,
        reading_review_id=reading_review_id,
        reading_review_basis=reading_review_basis,
        resolution_id=str(ir_payload.get("resolution_id", "")),
        resolution_fingerprint=str(ir_payload.get("resolution_fingerprint", "")),
        periodicity_profile_id=str(ir_payload.get("periodicity_profile_id", "periodicity-event-relative-v1")),
        periodicity_profile_version=str(ir_payload.get("periodicity_profile_version", "1")),
        contract_version=str(ir_payload.get("contract_version", FAITHFUL_IR_CONTRACT_VERSION)),
        canonicalizer_version=str(
            ir_payload.get(
                "canonicalizer_version",
                FAITHFUL_IR_CANONICALIZER_VERSION,
            )
        ),
        validator_version=str(
            ir_payload.get("validator_version", FAITHFUL_IR_VALIDATOR_VERSION)
        ),
        repair_policy_version=str(ir_payload.get("repair_policy_version", "validated-generation-repair-policy-v1")),
        issue_fingerprint=str(ir_payload.get("issue_fingerprint", "")),
        repair_history=tuple(
            dict(item) for item in ir_payload.get("repair_history", ()) if isinstance(item, Mapping)
        ),
        migration_history=tuple(
            dict(item) for item in ir_payload.get("migration_history", ()) if isinstance(item, Mapping)
        ),
        source_hash=str(ir_payload.get("source_hash", "")),
        ap_context_fingerprint=str(
            ir_payload.get("ap_context_fingerprint", "")
        ),
        ap_profile_mode=str(ir_payload.get("ap_profile_mode", "")),
        catalog_authority=str(ir_payload.get("catalog_authority", "none")),
        ap_grounding_fingerprint=str(
            ir_payload.get("ap_grounding_fingerprint", "")
        ),
        ap_grounding_authority=str(
            ir_payload.get("ap_grounding_authority", "proposal_only")
        ),
        node_contract_fingerprint=str(
            ir_payload.get("node_contract_fingerprint", "")
        ),
        temporal_language_contract_version=str(
            ir_payload.get(
                "temporal_language_contract_version",
                TEMPORAL_LANGUAGE_CONTRACT_VERSION,
            )
        ),
        semantic_source_contract_version=str(
            ir_payload.get("semantic_source_contract_version", "")
        ),
        effective_source_fingerprint=str(
            ir_payload.get("effective_source_fingerprint", "")
        ),
        effective_source_kind=str(
            ir_payload.get("effective_source_kind", "")
        ),
        effective_source_authority_record_id=str(
            ir_payload.get("effective_source_authority_record_id", "")
        ),
    )
    if ap_binding_context is not None:
        ir = bind_ir_to_ap_context(ir, ap_binding_context)
    if surface_segmentation is not None:
        known_segments = {
            segment.segment_id: segment
            for segment in surface_segmentation.segments
        }
        declared_segment_errors: list[str] = []
        for ap in ir.atomic_propositions:
            if ap.supporting_segment_ids:
                declared_segment_errors.extend(
                    _segment_reference_errors(
                        ap.supporting_segment_ids,
                        ap.source_span,
                        known_segments,
                        f"AP {ap.ap_id}",
                    )
                )
        for node in ir.nodes:
            if node.supporting_segment_ids:
                declared_segment_errors.extend(
                    _segment_reference_errors(
                        node.supporting_segment_ids,
                        node.source_span,
                        known_segments,
                        f"node {node.node_id}",
                    )
                )
        if declared_segment_errors:
            raise FaithfulIRValidationError("; ".join(declared_segment_errors))
        candidate_findings = (
            *collect_faithful_ir_findings(ir),
            *collect_review_axis_binding_findings(ir, plain_reading_review),
        )
        if candidate_findings:
            _raise_faithful_ir_findings(tuple(candidate_findings))
        ir = bind_ir_to_pre_ir_artifacts(
            ir,
            surface_segmentation=surface_segmentation,
            plain_reading_review=plain_reading_review,
        )
    if working_interpretation:
        ir = replace(ir, provenance=replace(ir.provenance,
            extraction_mode="working_interpretation_v1", approved=False, approval_source=""))
    elif ir.provenance.extraction_mode == "working_interpretation_v1":
        # Only a backend policy argument can enable working interpretation.
        ir = replace(ir, provenance=replace(ir.provenance, extraction_mode="generated"))
    canonical_version = compute_ir_version(ir)
    if ir.ir_version != canonical_version:
        ir = replace(ir, ir_version=canonical_version)
    validate_faithful_ir(
        ir,
        surface_segmentation=surface_segmentation,
        plain_reading_review=plain_reading_review,
    )
    return ir


def collect_faithful_ir_findings(
    ir: FaithfulSemanticIR,
    *,
    surface_segmentation: SurfaceSegmentation | None = None,
    plain_reading_review: PlainReadingReview | None = None,
) -> tuple[FaithfulIRFinding, ...]:
    """Collect every independent, mechanically checkable IR violation.

    Findings use the narrowest stable JSON Pointer available.  Cross-artifact
    invariants may legitimately point at the IR root, while node/AP violations
    must remain local to their owning item.
    """

    findings: list[FaithfulIRFinding] = []

    def add(
        code: str,
        path: str,
        message: str,
        *,
        repairable: bool = True,
        allowed_paths: tuple[str, ...] = (),
    ) -> None:
        findings.append(
            FaithfulIRFinding(
                code=code,
                path=path,
                message=message,
                repairable=repairable,
                allowed_paths=allowed_paths,
            )
        )

    def add_messages(
        messages: list[str] | tuple[str, ...],
        *,
        code: str,
        path: str,
        allowed_paths: tuple[str, ...] = (),
    ) -> None:
        for message in messages:
            add(code, path, message, allowed_paths=allowed_paths)

    if ir.schema_version != FAITHFUL_IR_SCHEMA_VERSION:
        add(
            "IR_SCHEMA_VERSION_UNSUPPORTED",
            "/faithful_ir/schema_version",
            (
                "online FaithfulSemanticIR requires "
                f"schema_version={FAITHFUL_IR_SCHEMA_VERSION}; got {ir.schema_version}"
            ),
            repairable=False,
        )
        return tuple(findings)
    if ir.contract_version != FAITHFUL_IR_CONTRACT_VERSION:
        add(
            "IR_CONTRACT_VERSION_UNSUPPORTED",
            "/faithful_ir/contract_version",
            (
                "online FaithfulSemanticIR requires "
                f"contract_version={FAITHFUL_IR_CONTRACT_VERSION}; "
                f"got {ir.contract_version}"
            ),
            repairable=False,
        )
    if ir.canonicalizer_version != FAITHFUL_IR_CANONICALIZER_VERSION:
        add(
            "IR_CANONICALIZER_VERSION_UNSUPPORTED",
            "/faithful_ir/canonicalizer_version",
            (
                "online FaithfulSemanticIR requires "
                f"canonicalizer_version={FAITHFUL_IR_CANONICALIZER_VERSION}; "
                f"got {ir.canonicalizer_version}"
            ),
            repairable=False,
        )
    if ir.validator_version != FAITHFUL_IR_VALIDATOR_VERSION:
        add(
            "IR_VALIDATOR_VERSION_UNSUPPORTED",
            "/faithful_ir/validator_version",
            (
                "online FaithfulSemanticIR requires "
                f"validator_version={FAITHFUL_IR_VALIDATOR_VERSION}; "
                f"got {ir.validator_version}"
            ),
            repairable=False,
        )
    if ir.node_contract_fingerprint != node_contract_fingerprint():
        add(
            "IR_NODE_CONTRACT_STALE",
            "/faithful_ir/node_contract_fingerprint",
            "FaithfulSemanticIR node contract fingerprint is stale",
            repairable=False,
        )
    if (
        ir.temporal_language_contract_version
        != TEMPORAL_LANGUAGE_CONTRACT_VERSION
    ):
        add(
            "IR_TEMPORAL_LANGUAGE_CONTRACT_STALE",
            "/faithful_ir/temporal_language_contract_version",
            "FaithfulSemanticIR temporal language contract is stale",
            repairable=False,
        )
    if not ir.source_text.strip():
        add("IR_SOURCE_TEXT_EMPTY", "/faithful_ir/source_text", "source_text is empty")
    if ir.ap_context_fingerprint:
        if ir.ap_profile_mode not in {
            "controlled_symbolic",
            "declared_catalog",
            "natural_language_grounded",
            "offline_formula_assisted",
            "undetermined",
        }:
            add(
                "IR_AP_CONTEXT_INVALID",
                "/faithful_ir/ap_profile_mode",
                "FaithfulSemanticIR has an invalid AP representation profile mode",
                repairable=False,
            )
        if ir.catalog_authority not in {
            "none",
            "controlled_symbolic_profile",
            "declared_catalog_metadata",
            "approved_human_inventory",
            "trusted_offline_formula_catalog",
        }:
            add(
                "IR_AP_CONTEXT_INVALID",
                "/faithful_ir/catalog_authority",
                "FaithfulSemanticIR has an invalid AP catalog authority",
                repairable=False,
            )
    ap_ids = [ap.ap_id for ap in ir.atomic_propositions]
    ap_names = [ap.name for ap in ir.atomic_propositions]
    ap_identity_ids = [
        ap.ap_identity_id for ap in ir.atomic_propositions if ap.ap_identity_id
    ]
    if ir.ap_grounding_fingerprint:
        if ir.ap_grounding_authority not in {
            "exact",
            "required_subset",
            "proposal_only",
            "conflict",
        }:
            add(
                "IR_AP_GROUNDING_AUTHORITY_INVALID",
                "/faithful_ir/ap_grounding_authority",
                "FaithfulSemanticIR has an invalid AP grounding authority",
                repairable=False,
            )
        if any(not ap.ap_identity_id for ap in ir.atomic_propositions):
            add(
                "IR_AP_IDENTITY_BINDING_MISSING",
                "/faithful_ir/atomic_propositions",
                "every AP declaration must bind one backend AP identity",
                repairable=False,
            )
        if len(ap_identity_ids) != len(set(ap_identity_ids)):
            add(
                "IR_AP_IDENTITY_COLLAPSE",
                "/faithful_ir/atomic_propositions",
                "multiple AP declarations bind the same AP identity",
                repairable=False,
            )
    if len(ap_ids) != len(set(ap_ids)) or len(ap_names) != len(set(ap_names)):
        add(
            "IR_AP_DUPLICATE",
            "/faithful_ir/atomic_propositions",
            "AP ids and names must be unique",
            allowed_paths=("/faithful_ir/atomic_propositions",),
        )
    for ap_index, ap in enumerate(ir.atomic_propositions):
        ap_path = f"/faithful_ir/atomic_propositions/{ap_index}"
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ap.name):
            add("IR_AP_NAME_INVALID", ap_path + "/name", f"invalid AP name: {ap.name}")
        add_messages(
            _evidence_errors(
                ir.source_text,
                ap.source_span,
                ap.evidence_text,
                f"AP {ap.ap_id}",
                supporting_segment_ids=ap.supporting_segment_ids,
            ),
            code="IR_AP_EVIDENCE_INVALID",
            path=ap_path,
            allowed_paths=(ap_path,),
        )
        if ap.formula_presence not in {"unspecified", "required", "optional"}:
            add(
                "IR_AP_FORMULA_PRESENCE_INVALID",
                ap_path + "/formula_presence",
                f"AP {ap.ap_id} has unsupported formula_presence={ap.formula_presence!r}"
            )
        if ap.required_in_formula and ap.optional_in_formula:
            add(
                "IR_AP_FORMULA_ROLE_CONFLICT",
                ap_path,
                f"AP {ap.ap_id} cannot be both required and optional in formulas"
            )
        add_messages(
            _supporting_segment_id_errors(
                ap.supporting_segment_ids,
                f"AP {ap.ap_id}",
            ),
            code="IR_AP_SEGMENT_REFERENCE_INVALID",
            path=ap_path + "/supporting_segment_ids",
            allowed_paths=(ap_path + "/supporting_segment_ids",),
        )

    if ir.ap_context_fingerprint and ir.catalog_authority == "none":
        findings.extend(_ap_identity_collapse_findings(ir))

    node_ids = [node.node_id for node in ir.nodes]
    node_map = ir.node_map
    if not node_ids or len(node_ids) != len(set(node_ids)):
        add(
            "IR_NODE_IDS_INVALID",
            "/faithful_ir/nodes",
            "node ids must be non-empty and unique",
            allowed_paths=("/faithful_ir/nodes",),
        )
    if ir.root_node_id not in node_map:
        add(
            "IR_ROOT_NODE_UNKNOWN",
            "/faithful_ir/root_node_id",
            "root_node_id does not reference a node",
            allowed_paths=("/faithful_ir/root_node_id", "/faithful_ir/nodes"),
        )
    declared_aps = set(ap_names)
    declaration_by_name = {
        ap.name: ap for ap in ir.atomic_propositions
    }
    unresolved_node_ids = {
        node_id
        for ambiguity in ir.ambiguities
        if not ambiguity.resolved
        for node_id in (
            ambiguity.affected_node_ids
            or tuple(node.node_id for node in ir.nodes)
        )
    }
    for node_index, node in enumerate(ir.nodes):
        node_path = f"/faithful_ir/nodes/{node_index}"
        if node.node_type not in SUPPORTED_NODE_TYPES:
            add(
                "IR_NODE_TYPE_UNSUPPORTED",
                node_path + "/node_type",
                f"unsupported node_type {node.node_type} at {node.node_id}",
            )
        for contract_finding in node_contract_findings(
            node,
            schema_version=ir.schema_version,
            allow_missing_required=node.node_id in unresolved_node_ids,
        ):
            finding_path = node_path + contract_finding.path_suffix
            add(
                contract_finding.code,
                finding_path,
                contract_finding.message,
                allowed_paths=(finding_path,),
            )
        if not 0.0 <= node.confidence <= 1.0:
            add("IR_NODE_CONFIDENCE_INVALID", node_path + "/confidence", f"invalid confidence at {node.node_id}")
        add_messages(
            temporal_range_errors(node),
            code="IR_TEMPORAL_RANGE_INVALID",
            path=node_path,
            allowed_paths=(node_path,),
        )
        # A node covered by an unresolved typed ambiguity deliberately
        # represents more than one source-supported reading.  Timing grounding
        # is enforced after that axis is resolved; rejecting the seed node here
        # would silently choose one reading before adjudication.
        if node.node_id not in unresolved_node_ids:
            add_messages(
                temporal_node_grounding_errors(node, ir=ir),
                code="IR_TEMPORAL_GROUNDING_MISMATCH",
                path=node_path + "/node_type",
                allowed_paths=(
                    node_path + "/node_type",
                    node_path + "/allowed_semantic_readings",
                    node_path + "/temporal_resolution",
                ),
            )
        for child_index, child in enumerate(node.children):
            if child not in node_map:
                add(
                    "IR_NODE_CHILD_UNKNOWN",
                    f"{node_path}/children/{child_index}",
                    f"missing child {child} referenced by {node.node_id}",
                    allowed_paths=(node_path + "/children", "/faithful_ir/nodes"),
                )
        if node.node_type == "ap_condition":
            ap_name = str(node.parameter("ap_name", ""))
            if ap_name not in declared_aps:
                add(
                    "IR_NODE_AP_UNDECLARED",
                    node_path + "/parameters/ap_name",
                    f"undeclared AP {ap_name!r} at {node.node_id}",
                    allowed_paths=(node_path + "/parameters", "/faithful_ir/atomic_propositions"),
                )
            elif ir.ap_grounding_fingerprint:
                declaration = declaration_by_name[ap_name]
                declaration_text = declaration.evidence_text.casefold()
                node_text = node.evidence_text.casefold()
                spans_overlap = (
                    declaration.source_span[0] < node.source_span[1]
                    and declaration.source_span[1] > node.source_span[0]
                )
                text_related = bool(
                    declaration_text
                    and node_text
                    and (
                        declaration_text in node_text
                        or node_text in declaration_text
                    )
                )
                if not spans_overlap and not text_related:
                    add(
                        "IR_AP_NODE_IDENTITY_MISMATCH",
                        node_path + "/parameters/ap_name",
                        (
                            f"node {node.node_id} evidence does not support "
                            f"AP identity {declaration.ap_identity_id}"
                        ),
                        repairable=False,
                    )
        if node.node_type == "next_step":
            add_messages(
                _next_step_contract_errors(ir, node),
                code="IR_NEXT_STEP_CONTRACT_INVALID",
                path=node_path + "/parameters",
                allowed_paths=(node_path + "/parameters", node_path + "/children"),
            )
        if node.node_type in {"until", "precedence", "before"}:
            add_messages(
                _relation_contract_errors(ir, node),
                code="IR_RELATION_CONTRACT_INVALID",
                path=node_path + "/parameters",
                allowed_paths=(node_path + "/parameters", node_path + "/children"),
            )
        if node.node_type == "periodicity":
            findings.extend(
                _periodicity_contract_findings(ir, node, node_index=node_index)
            )
        if node.node_type in BOUNDED_TEMPORAL_PARAMETER_VALUES:
            add_messages(
                _bounded_temporal_contract_errors(ir, node),
                code="IR_BOUNDED_TEMPORAL_CONTRACT_INVALID",
                path=node_path + "/parameters",
                allowed_paths=(node_path + "/parameters",),
            )
        if (
            node.node_type == "until"
            and ir.provenance.source == "natural_language"
            and re.search(
                r"\b(?:until|till)\s+(?:the\s+)?end(?:\s+of\s+(?:the\s+)?trace)?\b",
                ir.source_text,
                flags=re.IGNORECASE,
            )
        ):
            add(
                "IR_UNTIL_END_TOPOLOGY_INVALID",
                node_path,
                f"until node {node.node_id} must use persist_to_end for until-the-end wording"
            )
        add_messages(
            _evidence_errors(
                ir.source_text,
                node.source_span,
                node.evidence_text,
                f"node {node.node_id}",
                supporting_segment_ids=node.supporting_segment_ids,
            ),
            code="IR_NODE_EVIDENCE_INVALID",
            path=node_path,
            allowed_paths=(node_path,),
        )
        add_messages(
            _supporting_segment_id_errors(
                node.supporting_segment_ids,
                f"node {node.node_id}",
            ),
            code="IR_NODE_SEGMENT_REFERENCE_INVALID",
            path=node_path + "/supporting_segment_ids",
            allowed_paths=(node_path + "/supporting_segment_ids",),
        )
    for ambiguity_index, ambiguity in enumerate(ir.ambiguities):
        ambiguity_path = f"/faithful_ir/ambiguities/{ambiguity_index}"
        affected_ids = (
            set(ambiguity.affected_node_ids)
            if ambiguity.affected_node_ids
            else set(node_map)
        )
        if (
            not ambiguity.resolved
            and any(
                node_map[node_id].node_type == "periodicity"
                for node_id in affected_ids
                if node_id in node_map
            )
        ):
            add(
                "IR_UNRESOLVED_SEMANTICS",
                ambiguity_path + "/resolved",
                f"IR_UNRESOLVED_SEMANTICS: ambiguity {ambiguity.ambiguity_id}"
                ,
                repairable=False,
            )
        patch_ids: set[str] = set()
        for patch_index, patch in enumerate(ambiguity.alternative_patches):
            patch_path = f"{ambiguity_path}/alternative_patches/{patch_index}"
            if not patch.alternative_id:
                add(
                    "IR_AMBIGUITY_PATCH_ID_EMPTY",
                    patch_path + "/alternative_id",
                    f"ambiguity {ambiguity.ambiguity_id} has an empty alternative_id"
                )
            elif patch.alternative_id in patch_ids:
                add(
                    "IR_AMBIGUITY_PATCH_ID_DUPLICATE",
                    patch_path + "/alternative_id",
                    f"ambiguity {ambiguity.ambiguity_id} duplicates alternative "
                    f"{patch.alternative_id}"
                )
            patch_ids.add(patch.alternative_id)
            if patch.node_id not in node_map:
                add(
                    "IR_AMBIGUITY_PATCH_NODE_UNKNOWN",
                    patch_path + "/node_id",
                    f"ambiguity {ambiguity.ambiguity_id} patch references missing "
                    f"node {patch.node_id}"
                )
            if (
                ambiguity.affected_node_ids
                and patch.node_id not in ambiguity.affected_node_ids
            ):
                add(
                    "IR_AMBIGUITY_PATCH_NODE_OUTSIDE_AXIS",
                    patch_path + "/node_id",
                    f"ambiguity {ambiguity.ambiguity_id} patch node {patch.node_id} "
                    "is outside affected_node_ids"
                )

    for problem in scope_topology_problems(ir):
        add(
            problem.code,
            problem.path,
            problem.message,
            repairable=False,
            allowed_paths=(problem.path,),
        )

    if ir.root_node_id in node_map:
        visited: set[str] = set()
        active: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in active:
                add("IR_GRAPH_CYCLE", "/faithful_ir/nodes", f"cycle detected at {node_id}")
                return
            if node_id in visited:
                return
            active.add(node_id)
            for child_id in node_map[node_id].children:
                if child_id in node_map:
                    visit(child_id)
            active.remove(node_id)
            visited.add(node_id)

        visit(ir.root_node_id)
        disconnected = set(node_ids) - visited
        if disconnected:
            add(
                "IR_GRAPH_DISCONNECTED",
                "/faithful_ir/nodes",
                "disconnected nodes: " + ", ".join(sorted(disconnected)),
                allowed_paths=("/faithful_ir/nodes", "/faithful_ir/root_node_id"),
            )

    ambiguity_ids = [ambiguity.ambiguity_id for ambiguity in ir.ambiguities]
    if any(not ambiguity_id for ambiguity_id in ambiguity_ids):
        add("IR_AMBIGUITY_ID_EMPTY", "/faithful_ir/ambiguities", "ambiguity ids must be non-empty")
    if len(ambiguity_ids) != len(set(ambiguity_ids)):
        add("IR_AMBIGUITY_ID_DUPLICATE", "/faithful_ir/ambiguities", "ambiguity ids must be unique")
    for ambiguity_index, ambiguity in enumerate(ir.ambiguities):
        ambiguity_path = f"/faithful_ir/ambiguities/{ambiguity_index}"
        unknown = set(ambiguity.affected_node_ids) - set(node_ids)
        if unknown:
            add(
                "IR_AMBIGUITY_NODE_UNKNOWN",
                ambiguity_path + "/affected_node_ids",
                f"ambiguity {ambiguity.ambiguity_id} references unknown nodes: "
                + ", ".join(sorted(unknown))
            )
        if ambiguity.risk not in {"low", "medium", "high"}:
            add(
                "IR_AMBIGUITY_RISK_INVALID",
                ambiguity_path + "/risk",
                f"ambiguity {ambiguity.ambiguity_id} has unsupported risk {ambiguity.risk}"
            )
        if ambiguity.origin_layer not in {
            "plain_reading",
            "ir_extraction",
            "semantic_policy",
            "legacy",
        }:
            add(
                "IR_AMBIGUITY_ORIGIN_LAYER_INVALID",
                ambiguity_path + "/origin_layer",
                f"ambiguity {ambiguity.ambiguity_id} has unsupported origin layer "
                f"{ambiguity.origin_layer}",
            )
        if ambiguity.relation_kind not in {
            "same_axis",
            "derived_from",
            "independent",
        }:
            add(
                "IR_AMBIGUITY_RELATION_KIND_INVALID",
                ambiguity_path + "/relation_kind",
                f"ambiguity {ambiguity.ambiguity_id} has unsupported relation kind "
                f"{ambiguity.relation_kind}",
            )
        if ambiguity.relation_kind == "same_axis" and not ambiguity.related_axis_ids:
            add(
                "IR_AMBIGUITY_SAME_AXIS_PARENT_MISSING",
                ambiguity_path + "/related_axis_ids",
                f"ambiguity {ambiguity.ambiguity_id} same-axis binding lacks its review id",
            )
        normalized_alternatives = tuple(
            " ".join(alternative.split()).casefold()
            for alternative in ambiguity.alternatives
            if alternative.strip()
        )
        if ambiguity.alternatives and len(normalized_alternatives) < 2:
            add(
                "IR_AMBIGUITY_ALTERNATIVES_INCOMPLETE",
                ambiguity_path + "/alternatives",
                f"ambiguity {ambiguity.ambiguity_id} must contain at least two non-empty alternatives"
            )
        if len(normalized_alternatives) != len(set(normalized_alternatives)):
            add(
                "IR_AMBIGUITY_ALTERNATIVES_DUPLICATE",
                ambiguity_path + "/alternatives",
                f"ambiguity {ambiguity.ambiguity_id} alternatives must be distinct"
            )

    for message in _future_prohibition_contract_errors(ir):
        add(
            FUTURE_PROHIBITION_TOPOLOGY_CODE,
            FUTURE_PROHIBITION_TOPOLOGY_PATH,
            message,
            allowed_paths=(FUTURE_PROHIBITION_TOPOLOGY_PATH,),
        )
    add_messages(
        _parent_artifact_errors(ir, surface_segmentation, plain_reading_review),
        code="IR_PARENT_ARTIFACT_INVALID",
        path="/faithful_ir",
    )
    if surface_segmentation is not None:
        known_segments = {
            segment.segment_id: segment for segment in surface_segmentation.segments
        }
        for ap_index, ap in enumerate(ir.atomic_propositions):
            add_messages(
                _segment_reference_errors(
                    ap.supporting_segment_ids,
                    ap.source_span,
                    known_segments,
                    f"AP {ap.ap_id}",
                ),
                code="IR_AP_SEGMENT_BINDING_INVALID",
                path=f"/faithful_ir/atomic_propositions/{ap_index}/supporting_segment_ids",
            )
        for node_index, node in enumerate(ir.nodes):
            add_messages(
                _segment_reference_errors(
                    node.supporting_segment_ids,
                    node.source_span,
                    known_segments,
                    f"node {node.node_id}",
                ),
                code="IR_NODE_SEGMENT_BINDING_INVALID",
                path=f"/faithful_ir/nodes/{node_index}/supporting_segment_ids",
            )
    return tuple(findings)


def validate_faithful_ir(
    ir: FaithfulSemanticIR,
    *,
    surface_segmentation: SurfaceSegmentation | None = None,
    plain_reading_review: PlainReadingReview | None = None,
) -> None:
    findings = collect_faithful_ir_findings(
        ir,
        surface_segmentation=surface_segmentation,
        plain_reading_review=plain_reading_review,
    )
    if ir.provenance.extraction_mode == "working_interpretation_v1":
        # These findings concern source fidelity, not executability. Keep them
        # in collect_faithful_ir_findings for the output audit, never approve them.
        findings = tuple(f for f in findings if f.code not in WORKING_INTERPRETATION_WARNINGS)
    if not findings:
        return
    _raise_faithful_ir_findings(findings)


WORKING_INTERPRETATION_WARNINGS = frozenset({
    "IR_TEMPORAL_GROUNDING_MISMATCH", "IR_SCOPE_TOPOLOGY_MISMATCH",
    "IR_SCOPE_AMBIGUITY_UNRESOLVED", "FUTURE_PROHIBITION_TOPOLOGY_INVALID",
    "IR_UNTIL_END_TOPOLOGY_INVALID",
})


def recertify_legacy_faithful_ir_mapping(
    payload: Mapping[str, Any],
    *,
    source_text: str,
) -> tuple[FaithfulSemanticIR | None, dict[str, Any]]:
    """Explicitly audit an old IR contract without changing its semantics."""

    ir_payload = payload.get("faithful_ir", payload)
    if not isinstance(ir_payload, Mapping):
        return None, {
            "status": "stale_semantic_and_downstream",
            "semantic_behavior_preserved": False,
            "reason_code": "IR_MIGRATION_PAYLOAD_INVALID",
        }
    old_contract = str(ir_payload.get("contract_version", ""))
    old_validator = str(ir_payload.get("validator_version", ""))
    old_ir_version = str(ir_payload.get("ir_version", ""))
    old_artifact_hash = str(ir_payload.get("artifact_hash", ""))
    audit: dict[str, Any] = {
        "status": "stale_semantic_and_downstream",
        "semantic_behavior_preserved": False,
        "old_contract_version": old_contract,
        "old_validator_version": old_validator,
        "old_ir_version": old_ir_version,
        "old_artifact_hash": old_artifact_hash,
        "preserved_upstream_artifacts": (
            "surface_segmentation",
            "plain_reading_review",
            "human_semantic_decisions",
            "ap_inventory",
        ),
        "invalidated_artifacts": (
            "semantic_ir",
            "trace_obligation_plan",
            "trace_set",
            "constrained_skeleton_plan",
            "completion_hole_plan",
            "candidate_set",
            "evaluation",
            "approval",
        ),
    }
    if old_contract == FAITHFUL_IR_CONTRACT_VERSION and old_validator == FAITHFUL_IR_VALIDATOR_VERSION:
        audit.update(
            status="current_contract",
            reason_code="IR_MIGRATION_NOT_REQUIRED",
        )
        try:
            current = faithful_ir_from_mapping(
                ir_payload,
                source_text=source_text,
                trusted_provenance=True,
            )
        except FaithfulIRValidationError as exc:
            audit["reason_code"] = exc.code
            audit["validation_path"] = exc.path
            return None, audit
        audit["semantic_behavior_preserved"] = True
        audit["invalidated_artifacts"] = ()
        return current, audit
    if (
        str(ir_payload.get("schema_version", "")) != FAITHFUL_IR_SCHEMA_VERSION
        or str(ir_payload.get("canonicalizer_version", ""))
        != FAITHFUL_IR_CANONICALIZER_VERSION
    ):
        audit["reason_code"] = "IR_MIGRATION_SCHEMA_OR_CANONICALIZER_UNSUPPORTED"
        return None, audit

    migrated_payload = json.loads(json.dumps(ir_payload))
    migration_history = [
        dict(item)
        for item in ir_payload.get("migration_history", ())
        if isinstance(item, Mapping)
    ]
    migration_history.append(
        {
            "migration": "explicit-ir-recertification",
            "old_contract_version": old_contract,
            "old_validator_version": old_validator,
            "old_ir_version": old_ir_version,
            "old_artifact_hash": old_artifact_hash,
        }
    )
    migrated_payload.update(
        {
            "contract_version": FAITHFUL_IR_CONTRACT_VERSION,
            "validator_version": FAITHFUL_IR_VALIDATOR_VERSION,
            "ir_version": "",
            "artifact_hash": "",
            "migration_history": migration_history,
        }
    )
    try:
        recertified = faithful_ir_from_mapping(
            migrated_payload,
            source_text=source_text,
            trusted_provenance=True,
        )
        from ir_compiler import compile_ir_to_formula

        formula_fingerprint = hashlib.sha256(
            str(compile_ir_to_formula(recertified)).encode("utf-8")
        ).hexdigest()
    except FaithfulIRValidationError as exc:
        audit["reason_code"] = exc.code
        audit["validation_path"] = exc.path
        audit["findings"] = tuple(
            {"code": item.code, "path": item.path, "message": item.message}
            for item in exc.findings
        )
        return None, audit
    except (TypeError, ValueError) as exc:
        audit["reason_code"] = "IR_MIGRATION_COMPILATION_AUDIT_FAILED"
        audit["message"] = str(exc)
        return None, audit

    recertification_entry = {
        **migration_history[-1],
        "semantic_behavior_preserved": True,
        "compiled_formula_fingerprint_before": formula_fingerprint,
        "compiled_formula_fingerprint_after": formula_fingerprint,
        "new_ir_version": recertified.ir_version,
    }
    recertified = replace(
        recertified,
        migration_history=tuple((*migration_history[:-1], recertification_entry)),
        artifact_hash="",
    )
    recertified = replace(recertified, ir_version=compute_ir_version(recertified))
    validate_faithful_ir(recertified)
    audit.update(
        status="recertified_rederive_downstream",
        reason_code="IR_MIGRATION_RECERTIFIED",
        semantic_behavior_preserved=True,
        compiled_formula_fingerprint_before=formula_fingerprint,
        compiled_formula_fingerprint_after=formula_fingerprint,
        new_ir_version=recertified.ir_version,
        new_artifact_hash=recertified.artifact_hash,
    )
    return recertified, audit


def _raise_faithful_ir_findings(
    findings: tuple[FaithfulIRFinding, ...],
) -> None:
    primary = next(
        (
            finding
            for code in (FUTURE_PROHIBITION_TOPOLOGY_CODE, "IR_UNRESOLVED_SEMANTICS")
            for finding in findings
            if finding.code == code
        ),
        findings[0],
    )
    raise FaithfulIRValidationError(
        "; ".join(finding.message for finding in findings),
        code=primary.code,
        path=primary.path,
        repairable=primary.repairable,
        findings=findings,
    )


def _relation_contract_errors(
    ir: FaithfulSemanticIR,
    node: SemanticNode,
) -> list[str]:
    """Validate role order and high-risk parameters without inventing defaults."""

    errors: list[str] = []
    if len(node.children) != 2:
        errors.append(
            f"{node.node_type} node {node.node_id} must have exactly two ordered children"
        )
        return errors
    if (
        node.node_type in {"precedence", "before"}
        and node.children[0] == node.children[1]
    ):
        errors.append(
            f"{node.node_type} node {node.node_id} must use distinct prerequisite "
            "and constrained children"
        )

    allowed = (
        UNTIL_PARAMETER_VALUES
        if node.node_type == "until"
        else PRECEDENCE_PARAMETER_VALUES
    )
    parameters = dict(node.parameters)
    unresolved = _has_unresolved_relation_ambiguity(ir, node)
    for name, choices in allowed.items():
        value = str(parameters.get(name, "")).strip()
        if not value:
            if not unresolved:
                errors.append(
                    f"{node.node_type} node {node.node_id} missing parameter {name}; "
                    "declare an unresolved ambiguity instead of silently defaulting"
                )
            continue
        if value not in choices:
            errors.append(
                f"{node.node_type} node {node.node_id} has unsupported {name}={value!r}"
            )

    if node.node_type in {"precedence", "before"}:
        ordering = str(parameters.get("ordering", ""))
        same_step = str(parameters.get("same_step_policy", ""))
        if ordering == "strict" and same_step == "allowed":
            errors.append(
                f"{node.node_type} node {node.node_id} cannot combine strict ordering "
                "with allowed same-step"
            )
        if ordering == "non_strict" and same_step == "forbidden":
            errors.append(
                f"{node.node_type} node {node.node_id} cannot combine non_strict ordering "
                "with forbidden same-step"
            )
    if (
        node.node_type == "until"
        and parameters.get("witness_policy") == "optional_if_missing"
    ):
        for parent in ir.nodes:
            if parent.node_type != "disjunction" or node.node_id not in parent.children:
                continue
            for sibling_id in parent.children:
                if sibling_id == node.node_id:
                    continue
                sibling = ir.node_map.get(sibling_id)
                if (
                    sibling is not None
                    and sibling.node_type in {"global_constraint", "persist_to_end"}
                    and len(sibling.children) == 1
                    and _semantic_subtree_signature(ir, sibling.children[0])
                    == _semantic_subtree_signature(ir, node.children[0])
                ):
                    errors.append(
                        f"until node {node.node_id} uses optional_if_missing while "
                        f"disjunction sibling {sibling.node_id} already represents "
                        "the missing-witness persistence branch; use required witness "
                        "to avoid two encodings of the same semantic alternative"
                    )
    return errors


def _semantic_subtree_signature(
    ir: FaithfulSemanticIR,
    node_id: str,
    seen: frozenset[str] = frozenset(),
) -> tuple[object, ...]:
    """Compare denotationally identical IR branches without node-id identity."""

    if node_id in seen or node_id not in ir.node_map:
        return ("invalid_or_cyclic", node_id)
    node = ir.node_map[node_id]
    return (
        node.node_type,
        tuple(node.parameters),
        tuple(
            _semantic_subtree_signature(ir, child, seen | {node_id})
            for child in node.children
        ),
    )


def _next_step_contract_errors(
    ir: FaithfulSemanticIR,
    node: SemanticNode,
) -> list[str]:
    errors: list[str] = []
    if len(node.children) != 1:
        errors.append(
            f"next_step node {node.node_id} must have exactly one child"
        )
    steps = node.parameter("steps", 1)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        errors.append(
            f"next_step node {node.node_id} requires integer steps >= 1"
        )
    if node.children:
        child = ir.node_map.get(node.children[0])
        if child is not None and child.node_type == "eventually_strict":
            anchor = node.parameter("anchor_node_id")
            relative = node.parameter("relative_offset")
            if anchor in {None, ""} or relative is None:
                errors.append(
                    f"next_step node {node.node_id} directly wraps eventually_strict "
                    "without an independent anchor/offset provenance; this would count "
                    "one strict-future transition twice"
                )
    return errors


def _has_unresolved_relation_ambiguity(
    ir: FaithfulSemanticIR,
    node: SemanticNode,
) -> bool:
    relation_markers = {
        "until": ("until", "witness", "left", "strong", "weak"),
        "precedence": ("precedence", "prior", "strict", "witness", "reuse", "scope"),
        "before": ("before", "prior", "strict", "witness", "reuse", "scope"),
    }[node.node_type]
    for ambiguity in ir.ambiguities:
        if ambiguity.resolved:
            continue
        if ambiguity.affected_node_ids and node.node_id not in ambiguity.affected_node_ids:
            continue
        text = " ".join(
            (
                ambiguity.ambiguity_id,
                ambiguity.description,
                *ambiguity.alternatives,
            )
        ).lower()
        if any(marker in text for marker in relation_markers):
            return True
    return False


def _bounded_temporal_contract_errors(
    ir: FaithfulSemanticIR,
    node: SemanticNode,
) -> list[str]:
    """Check only cross-field temporal invariants.

    Required fields, primitive types, and enums are enforced by the shared node
    contract registry, which emits leaf-addressed findings for atomic repair.
    """

    errors: list[str] = []
    params = dict(node.parameters)

    if node.node_type == "fixed_duration" and all(
        name in params for name in ("duration", "minimum_horizon")
    ):
        try:
            duration = params["duration"]
            horizon = params["minimum_horizon"]
            if not isinstance(duration, int) or isinstance(duration, bool):
                return errors
            if not isinstance(horizon, int) or isinstance(horizon, bool):
                return errors
            short_policy = str(params.get("short_trace_policy", ""))
            terminal = str(params.get("terminal_obligation_policy", ""))
            if short_policy == "strong_next" and horizon < duration:
                errors.append(
                    f"fixed_duration node {node.node_id} minimum_horizon must cover duration"
                )
            if (short_policy, terminal) not in {
                ("strong_next", "require_full_duration"),
                ("available_prefix", "truncate_at_end"),
            }:
                errors.append(
                    f"fixed_duration node {node.node_id} has inconsistent short/terminal policy"
                )
        except (TypeError, ValueError):
            pass
    elif node.node_type == "exact_run_length" and all(
        name in params
        for name in ("start_offset", "duration", "minimum_horizon")
    ):
        try:
            offset = params["start_offset"]
            duration = params["duration"]
            horizon = params["minimum_horizon"]
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (offset, duration, horizon)
            ):
                return errors
            if horizon < offset + duration:
                errors.append(
                    f"exact_run_length node {node.node_id} minimum_horizon must "
                    "cover its start offset and duration"
                )
            expected = (
                str(params.get("origin", "")),
                str(params.get("start_position_policy", "")),
                str(params.get("terminal_obligation_policy", "")),
                str(params.get("post_window_policy", "")),
            )
            if expected != (
                "current_position",
                "required",
                "require_full_duration",
                "false_if_position_exists",
            ):
                errors.append(
                    f"exact_run_length node {node.node_id} has inconsistent "
                    "origin/start/terminal/post-window policy"
                )
        except (TypeError, ValueError):
            pass
    elif node.node_type == "bounded_window" and all(
        name in params
        for name in ("lower_bound", "upper_bound", "minimum_horizon")
    ):
        try:
            lower = params["lower_bound"]
            upper = params["upper_bound"]
            horizon = params["minimum_horizon"]
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (lower, upper, horizon)
            ):
                return errors
            if upper < lower:
                errors.append(
                    f"bounded_window node {node.node_id} upper_bound is below lower_bound"
                )
            if (
                str(params.get("short_trace_policy", ""))
                == "strong_upper_bound"
                and horizon < upper + 1
            ):
                errors.append(
                    f"bounded_window node {node.node_id} minimum_horizon must cover upper_bound"
                )
            if (
                str(params.get("short_trace_policy", ""))
                == "strong_upper_bound"
                and str(params.get("terminal_obligation_policy", ""))
                != "require_witness"
            ):
                errors.append(
                    f"bounded_window node {node.node_id} has inconsistent short/terminal policy"
                )
        except (TypeError, ValueError):
            pass
    return errors


def _periodicity_contract_errors(
    ir: FaithfulSemanticIR,
    node: SemanticNode,
) -> list[str]:
    """Compatibility view of the structured v4 periodicity findings."""

    node_index = next(
        (index for index, candidate in enumerate(ir.nodes) if candidate.node_id == node.node_id),
        0,
    )
    return [
        finding.message
        for finding in _periodicity_contract_findings(ir, node, node_index=node_index)
    ]


def _periodicity_contract_findings(
    ir: FaithfulSemanticIR,
    node: SemanticNode,
    *,
    node_index: int,
) -> tuple[FaithfulIRFinding, ...]:
    """Enforce the shared v4 mode contract with leaf-level issue paths."""

    findings: list[FaithfulIRFinding] = []
    params = dict(node.parameters)
    unresolved = _unresolved_periodicity_parameters(ir, node)
    mode = str(params.get("mode", "")).strip()
    parameters_path = f"/faithful_ir/nodes/{node_index}/parameters"

    def add(code: str, parameter: str, message: str) -> None:
        parameter_path = f"{parameters_path}/{parameter}"
        findings.append(
            FaithfulIRFinding(
                code=code,
                path=parameter_path,
                message=message,
                allowed_paths=(parameter_path,),
            )
        )

    contract = periodicity_mode_contract(mode)
    if contract is None:
        if "mode" not in unresolved:
            add(
                "PERIODICITY_REQUIRED_PARAMETER_MISSING"
                if not mode
                else "PERIODICITY_MODE_PARAMETER_MISMATCH",
                "mode",
                f"PERIODICITY_REQUIRED_PARAMETER_MISSING: node {node.node_id} parameter mode"
                if not mode
                else (
                    f"PERIODICITY_MODE_PARAMETER_MISMATCH: node {node.node_id} "
                    f"mode={mode!r}"
                ),
            )
        return tuple(findings)
    for name in sorted(contract.required):
        if params.get(name) in {None, ""} and name not in unresolved:
            add(
                "PERIODICITY_REQUIRED_PARAMETER_MISSING",
                name,
                f"PERIODICITY_REQUIRED_PARAMETER_MISSING: node {node.node_id} parameter {name}",
            )
    for name in sorted(contract.forbidden):
        if name in params:
            add(
                "PERIODICITY_FORBIDDEN_PARAMETER_PRESENT",
                name,
                f"PERIODICITY_FORBIDDEN_PARAMETER_PRESENT: node {node.node_id} parameter {name}",
            )
    for name in ("period", "phase"):
        if name not in params:
            continue
        try:
            value = int(params[name])
        except (TypeError, ValueError):
            add(
                "PERIODICITY_PARAMETER_NON_INTEGER",
                name,
                f"periodicity node {node.node_id} has non-integer {name}",
            )
            continue
        if value < (1 if name == "period" else 0):
            add(
                "PERIODICITY_PARAMETER_OUT_OF_RANGE",
                name,
                f"periodicity node {node.node_id} has invalid {name}={value}",
            )
    for name, choices in PERIODICITY_PARAMETER_VALUES.items():
        if name not in params:
            continue
        value = str(params.get(name, "")).strip()
        if value not in choices:
            add(
                "PERIODICITY_PARAMETER_VALUE_UNSUPPORTED",
                name,
                f"periodicity node {node.node_id} has unsupported {name}={value!r}",
            )
    for name, expected in contract.fixed_values.items():
        if name in unresolved or params.get(name) in {None, "", expected}:
            continue
        code = (
            "PERIODICITY_FIXED_PHASE_ALLOWED_UNSUPPORTED"
            if mode == "fixed_phase" and name == "off_schedule_policy"
            else "PERIODICITY_MODE_PARAMETER_MISMATCH"
        )
        add(
            code,
            name,
            f"{code}: node {node.node_id} {mode} requires {name}={expected!r}",
        )
    if mode == "fixed_phase":
        if "phase" in params and "period" in params:
            try:
                if int(params["phase"]) >= int(params["period"]):
                    add(
                        "PERIODICITY_MODE_PARAMETER_MISMATCH",
                        "phase",
                        f"PERIODICITY_MODE_PARAMETER_MISMATCH: node {node.node_id} phase must be below period",
                    )
            except (TypeError, ValueError):
                pass
    return tuple(findings)


def _unresolved_periodicity_parameters(
    ir: FaithfulSemanticIR,
    node: SemanticNode,
) -> frozenset[str]:
    """Return only parameters explicitly covered by live node ambiguities."""

    unresolved: set[str] = set()
    for ambiguity in ir.ambiguities:
        if ambiguity.resolved or (
            ambiguity.affected_node_ids
            and node.node_id not in ambiguity.affected_node_ids
        ):
            continue
        text = " ".join(
            (ambiguity.ambiguity_id, ambiguity.description, *ambiguity.alternatives)
        ).casefold()
        normalized = re.sub(r"[-\s]+", "_", text)
        spaced = normalized.replace("_", " ")
        for name, markers in _PERIODICITY_AMBIGUITY_PARAMETER_MARKERS.items():
            parameter_text = (
                re.sub(r"\boff phase(?: policy)?\b", "", spaced)
                if name == "phase"
                else spaced
            )
            if any(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(marker.replace('-', ' ').replace('_', ' '))}(?![a-z0-9])",
                    parameter_text,
                )
                for marker in markers
            ):
                unresolved.add(name)
    return frozenset(unresolved)


def trace_generation_readiness_errors(
    ir: FaithfulSemanticIR | None,
) -> tuple[str, ...]:
    """Report blockers for authoritative finite-trace construction.

    The implementation delegates to the shared capability audit so every
    execution entry point observes the same parameter, ambiguity, and runtime
    support contract.
    """

    if ir is None:
        return ()
    from semantic_readiness import (
        CAN_GENERATE_AUTHORITATIVE_TRACES,
        audit_semantic_readiness,
    )

    report = audit_semantic_readiness(ir)
    return report.messages_for(CAN_GENERATE_AUTHORITATIVE_TRACES)


def assert_ir_ready_for_trace_generation(ir: FaithfulSemanticIR) -> None:
    """Reject trace construction without converting unresolved policy into defaults."""

    from semantic_readiness import (
        CAN_GENERATE_AUTHORITATIVE_TRACES,
        audit_semantic_readiness,
    )

    report = audit_semantic_readiness(ir)
    errors = report.messages_for(CAN_GENERATE_AUTHORITATIVE_TRACES)
    if errors:
        raise FaithfulIRTraceReadinessError(errors, report=report)


def future_prohibition_topology_error_from_mapping(
    proposal: object,
    *,
    source_text: str,
) -> FaithfulIRValidationError | None:
    """Validate the domain topology without requiring grounded evidence spans."""

    if not is_future_prohibition_source(source_text):
        return None
    ir_payload, prefix = faithful_ir_proposal_mapping(proposal)
    if ir_payload is None:
        return None
    raw_nodes = ir_payload.get("nodes")
    if not isinstance(raw_nodes, list):
        return None
    node_map = {
        str(node.get("node_id", "")): node
        for node in raw_nodes
        if isinstance(node, Mapping) and str(node.get("node_id", ""))
    }
    if has_future_prohibition_topology(
        str(ir_payload.get("root_node_id", "")),
        node_map,
    ):
        return None
    return FaithfulIRValidationError(
        FUTURE_PROHIBITION_TOPOLOGY_MESSAGE,
        code=FUTURE_PROHIBITION_TOPOLOGY_CODE,
        path=(prefix + "/root_node_id" if prefix else FUTURE_PROHIBITION_TOPOLOGY_PATH),
    )


def _future_prohibition_contract_errors(ir: FaithfulSemanticIR) -> list[str]:
    """Reject prefix-precedence IR for post-trigger future prohibitions."""

    if not is_future_prohibition_source(ir.source_text):
        return []
    if has_future_prohibition_topology(ir.root_node_id, ir.node_map):
        return []
    return [FUTURE_PROHIBITION_TOPOLOGY_MESSAGE]


def compute_ir_version(ir: FaithfulSemanticIR) -> str:
    payload = {
        "schema_version": ir.schema_version,
        "requirement_id": ir.requirement_id,
        "source_text": ir.source_text,
        "aps": [
            (
                ap.ap_id,
                ap.name,
                ap.ap_identity_id,
                ap.ap_mention_ids,
                ap.aliases,
                ap.source_span,
                ap.evidence_text,
            )
            for ap in ir.atomic_propositions
        ],
        "root": ir.root_node_id,
        "nodes": [
            (
                node.node_id,
                node.node_type,
                node.children,
                node.semantic_role,
                node.scope,
                node.temporal_relation,
                node.boundary_policy,
                node.parameters,
                node.source_span,
                node.evidence_text,
            )
            for node in ir.nodes
        ],
        "ap_binding_context": (
            ir.ap_context_fingerprint,
            ir.ap_profile_mode,
            ir.catalog_authority,
            ir.ap_grounding_fingerprint,
            ir.ap_grounding_authority,
        ),
        "structure_contracts": (
            ir.node_contract_fingerprint,
            ir.temporal_language_contract_version,
        ),
    }
    if any(
        ap.formula_presence != "unspecified"
        or ap.required_in_formula
        or ap.optional_in_formula
        for ap in ir.atomic_propositions
    ):
        payload["formula_closure_ap_roles"] = [
            (
                ap.ap_id,
                ap.formula_presence,
                ap.required_in_formula,
                ap.optional_in_formula,
            )
            for ap in ir.atomic_propositions
        ]
    if any(
        node.allowed_semantic_readings or node.temporal_resolution != "fixed"
        for node in ir.nodes
    ):
        payload["temporal_semantic_ranges"] = [
            (
                node.node_id,
                node.allowed_semantic_readings,
                node.temporal_resolution,
            )
            for node in ir.nodes
        ]
    # Optional reviewed metadata participates in the hash whenever present, so
    # changing a reading or parent artifact invalidates every downstream plan.
    if ir.ambiguities:
        payload["ambiguities"] = [
            (
                ambiguity.ambiguity_id,
                ambiguity.description,
                ambiguity.affected_node_ids,
                ambiguity.alternatives,
                tuple(
                    (
                        patch.alternative_id,
                        patch.node_id,
                        patch.parameter_updates,
                        patch.node_type,
                        patch.description,
                    )
                    for patch in ambiguity.alternative_patches
                ),
                ambiguity.risk,
                ambiguity.resolved,
            )
            for ambiguity in ir.ambiguities
        ]
    if any(ap.supporting_segment_ids for ap in ir.atomic_propositions) or any(
        node.supporting_segment_ids for node in ir.nodes
    ):
        payload["segment_support"] = {
            "aps": [
                (ap.ap_id, ap.supporting_segment_ids)
                for ap in ir.atomic_propositions
            ],
            "nodes": [
                (node.node_id, node.supporting_segment_ids)
                for node in ir.nodes
            ],
        }
    if ir.segmentation_id or ir.reading_review_id or ir.reading_review_basis:
        payload["review_parents"] = (
            ir.segmentation_id,
            ir.reading_review_id,
            ir.reading_review_basis,
        )
    if ir.resolution_id or ir.resolution_fingerprint:
        payload["semantic_resolution_parent"] = (
            ir.resolution_id,
            ir.resolution_fingerprint,
            ir.periodicity_profile_id,
            ir.periodicity_profile_version,
        )
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return f"ir{ir.schema_version}-" + hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def bind_ir_to_ap_context(
    ir: FaithfulSemanticIR,
    ap_binding_context: APBindingContext,
) -> FaithfulSemanticIR:
    """Bind an IR to the exact AP authority used for its effective source."""

    ap_binding_context.validate_source(ir.source_text)
    if (
        ir.ap_context_fingerprint
        and ir.ap_context_fingerprint != ap_binding_context.context_fingerprint
    ):
        raise FaithfulIRValidationError(
            "FaithfulSemanticIR was produced under a different AP binding context",
            code="IR_AP_CONTEXT_STALE",
            path="/faithful_ir/ap_context_fingerprint",
            repairable=False,
        )
    grounded_aps = ir.atomic_propositions
    grounding = ap_binding_context.grounding_contract
    if grounding is not None:
        from ap_grounding import (
            AUTHORITY_EXACT,
            identity_for_evidence,
            proposal_identity,
        )

        rebound: list[AtomicProposition] = []
        by_name = grounding.identity_by_name
        for ap in ir.atomic_propositions:
            identity = by_name.get(ap.name)
            if identity is None:
                identity = identity_for_evidence(
                    grounding,
                    ap.evidence_text,
                    ap.source_span,
                )
            if identity is None and grounding.authority != AUTHORITY_EXACT:
                try:
                    identity = proposal_identity(
                        grounding,
                        canonical_name=ap.name,
                        evidence_text=ap.evidence_text,
                        source_span=ap.source_span,
                        supporting_segment_ids=ap.supporting_segment_ids,
                    )
                except Exception as exc:
                    raise FaithfulIRValidationError(
                        str(exc),
                        code=getattr(
                            exc,
                            "code",
                            "IR_AP_EVIDENCE_IDENTITY_MISMATCH",
                        ),
                        path="/faithful_ir/atomic_propositions",
                        repairable=False,
                    ) from exc
            if identity is None:
                raise FaithfulIRValidationError(
                    f"AP {ap.name!r} is not present in the exact grounding contract",
                    code="IR_AP_IDENTITY_BINDING_MISSING",
                    path="/faithful_ir/atomic_propositions",
                    repairable=False,
                )
            rebound.append(
                replace(
                    ap,
                    name=identity.canonical_name,
                    ap_identity_id=identity.identity_key,
                    ap_mention_ids=tuple(
                        item.mention_id for item in identity.mentions
                    ),
                )
            )
        grounded_aps = tuple(rebound)
    bound = replace(
        ir,
        atomic_propositions=grounded_aps,
        requirement_id=(ir.requirement_id or ap_binding_context.requirement_id),
        ap_context_fingerprint=ap_binding_context.context_fingerprint,
        ap_profile_mode=ap_binding_context.profile_mode,
        catalog_authority=ap_binding_context.catalog_authority,
        ap_grounding_fingerprint=(
            grounding.fingerprint
            if grounding is not None
            else ""
        ),
        ap_grounding_authority=ap_binding_context.grounding_authority,
    )
    return replace(bound, ir_version=compute_ir_version(bound))


def _ap_identity_collapse_findings(
    ir: FaithfulSemanticIR,
) -> tuple[FaithfulIRFinding, ...]:
    declaration_evidence = {
        item.name: _ap_identity_tokens(item.evidence_text)
        for item in ir.atomic_propositions
    }
    uses: dict[str, list[SemanticNode]] = {}
    for node in ir.nodes:
        if node.node_type == "ap_condition":
            uses.setdefault(str(node.parameter("ap_name", "")), []).append(node)
    findings: list[FaithfulIRFinding] = []
    for ap_name, nodes in uses.items():
        if len(nodes) < 2:
            continue
        name_tokens = _ap_identity_tokens(ap_name.replace("_", " "))
        evidence_groups = {
            tuple(sorted(_ap_identity_tokens(node.evidence_text)))
            for node in nodes
            if _ap_identity_tokens(node.evidence_text)
        }
        if len(evidence_groups) < 2:
            continue
        basis = declaration_evidence.get(ap_name, set()) or name_tokens
        unsupported = [
            node
            for node in nodes
            if not _ap_token_sets_related(
                _ap_identity_tokens(node.evidence_text),
                basis,
            )
        ]
        roles = {node.semantic_role.casefold() for node in nodes if node.semantic_role}
        if unsupported and (
            len(roles) >= 2
            or any(
                not _ap_token_sets_related(set(left), set(right))
                for left in evidence_groups
                for right in evidence_groups
                if left != right
            )
        ):
            findings.append(
                FaithfulIRFinding(
                    code="IR_AP_IDENTITY_COLLAPSE",
                    path="/faithful_ir/nodes",
                    message=(
                        f"AP {ap_name!r} merges distinct evidence predicates "
                        "without an approved identity basis"
                    ),
                    repairable=False,
                )
            )
    return tuple(findings)


_AP_IDENTITY_STOPWORDS = frozenset(
    {
        "an", "the", "is", "are", "be", "been", "being", "shall",
        "should", "must", "will", "when", "whenever", "if", "then", "there",
        "holds", "hold", "occur", "occurs", "occurred", "happen", "happens",
        "mode",
    }
)


def _ap_identity_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", value.casefold())
        if token not in _AP_IDENTITY_STOPWORDS and len(token) > 1
    }


def _ap_token_sets_related(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    overlap = left & right
    return bool(overlap) and (
        len(overlap) / min(len(left), len(right)) >= 0.5
        or left <= right
        or right <= left
    )


def attach_faithful_ir(
    semantic: SemanticStructure,
    requirement: str,
    *,
    requirement_id: str = "",
    ap_binding_context: APBindingContext | None = None,
) -> SemanticStructure:
    """Validate an already constructed native current-schema IR for production use."""

    if semantic.formula_text:
        raise FaithfulIRValidationError(
            "NLU semantic output contains forbidden formula_text"
        )
    semantic = _ensure_default_pre_ir_artifacts(
        semantic,
        requirement,
        requirement_id=requirement_id,
    )
    if (
        semantic.plain_reading_review is not None
        and semantic.plain_reading_review.downstream_status
        == "stale_due_to_revision"
        and semantic.plain_reading_review.overall_status
        in {"no_ambiguity", "plain_reading_reviewed", "ambiguous_unresolved"}
    ):
        semantic = replace(
            semantic,
            plain_reading_review=replace(
                semantic.plain_reading_review,
                downstream_status="current",
            ),
        )
    if semantic.faithful_ir is None:
        raise FaithfulIRValidationError(
            "FaithfulSemanticIR is required before attachment",
            code="NATIVE_FAITHFUL_IR_REQUIRED",
            path="/faithful_ir",
            repairable=False,
        )
    if ap_binding_context is not None:
        semantic = replace(
            semantic,
            faithful_ir=bind_ir_to_ap_context(
                semantic.faithful_ir,
                ap_binding_context,
            ),
        )
    ir = bind_ir_to_pre_ir_artifacts(
        semantic.faithful_ir,
        surface_segmentation=semantic.surface_segmentation,
        plain_reading_review=semantic.plain_reading_review,
    )
    validate_faithful_ir(
        ir,
        surface_segmentation=semantic.surface_segmentation,
        plain_reading_review=semantic.plain_reading_review,
    )
    # Attachment validates and binds structure; it is not a source-review
    # authority.  In particular, a previously rejected/pending semantic must
    # never become ready merely because the attached graph is well formed.
    # ``needs_semantic_review`` is therefore absorbing here.  A dedicated
    # review decision may promote it later through the review loop.
    attached_status = (
        "needs_semantic_review"
        if semantic.semantic_status == "needs_semantic_review"
        or ir.has_high_risk_ambiguity
        else "ready"
    )
    return replace(
        semantic,
        aps=ir.ap_names,
        faithful_ir=ir,
        semantic_status=attached_status,
    )


def bind_ir_to_pre_ir_artifacts(
    ir: FaithfulSemanticIR,
    *,
    surface_segmentation: SurfaceSegmentation,
    plain_reading_review: PlainReadingReview | None = None,
) -> FaithfulSemanticIR:
    """Bind every evidence span to backend-derived segments and review parents."""

    from plain_reading import assert_plain_reading_ready_for_semantic_ir
    from surface_segmentation import assert_surface_segmentation_gate

    assert_surface_segmentation_gate(
        surface_segmentation,
        current_source_text=ir.source_text,
    )
    if plain_reading_review is not None:
        assert_plain_reading_ready_for_semantic_ir(
            plain_reading_review,
            surface_segmentation,
        )
    if ir.segmentation_id and ir.segmentation_id != surface_segmentation.segmentation_id:
        raise FaithfulIRValidationError(
            "FaithfulSemanticIR references a stale surface segmentation"
        )
    expected_reading_id = (
        plain_reading_review.reading_review_id
        if plain_reading_review is not None
        else ""
    )
    if ir.reading_review_id and expected_reading_id and (
        ir.reading_review_id != expected_reading_id
    ):
        raise FaithfulIRValidationError(
            "FaithfulSemanticIR references a stale plain reading review"
        )

    aps = tuple(
        replace(
            ap,
            supporting_segment_ids=_segments_for_span(
                ap.source_span,
                surface_segmentation,
            ),
        )
        for ap in ir.atomic_propositions
    )
    nodes = tuple(
        replace(
            node,
            supporting_segment_ids=_segments_for_span(
                node.source_span,
                surface_segmentation,
            ),
        )
        for node in ir.nodes
    )
    ambiguities = list(ir.ambiguities)
    if plain_reading_review is not None:
        reviewed = tuple(plain_reading_review.ambiguities)
        rebound_ambiguities: list[SemanticAmbiguity] = []
        matched_review_ids: set[str] = set()
        for ambiguity_index, ambiguity in enumerate(ambiguities):
            review_match = _matching_plain_reading_ambiguity(
                ambiguity,
                reviewed,
                excluded_ids=matched_review_ids,
                ambiguity_index=ambiguity_index,
            )
            if review_match is None:
                rebound_ambiguities.append(ambiguity)
                continue
            _assert_reviewed_branch_matches_ir(
                ambiguity,
                review_match,
                {node.node_id: node for node in nodes},
            )
            matched_review_ids.add(review_match.ambiguity_id)
            rebound_ambiguities.append(
                replace(
                    ambiguity,
                    ambiguity_id=review_match.ambiguity_id,
                    description=review_match.question,
                    alternatives=tuple(
                        option.plain_interpretation
                        for option in review_match.options
                    ),
                    resolved=review_match.decision_status == "reviewer_selected",
                    origin_layer="plain_reading",
                    related_axis_ids=(review_match.ambiguity_id,),
                    relation_kind="same_axis",
                )
            )
        ambiguities = rebound_ambiguities
        for ambiguity in reviewed:
            if ambiguity.ambiguity_id not in matched_review_ids:
                ambiguities.append(
                    SemanticAmbiguity(
                        ambiguity_id=ambiguity.ambiguity_id,
                        description=ambiguity.question,
                        affected_node_ids=(ir.root_node_id,) if ir.root_node_id else (),
                        alternatives=tuple(
                            option.plain_interpretation for option in ambiguity.options
                        ),
                        risk="medium",
                        resolved=ambiguity.decision_status == "reviewer_selected",
                        origin_layer="plain_reading",
                        related_axis_ids=(ambiguity.ambiguity_id,),
                        relation_kind="same_axis",
                    )
                )
    bound = replace(
        ir,
        atomic_propositions=aps,
        nodes=nodes,
        ambiguities=tuple(ambiguities),
        segmentation_id=surface_segmentation.segmentation_id,
        reading_review_id=expected_reading_id,
        reading_review_basis=(
            plain_reading_review.review_basis
            if plain_reading_review is not None
            else ir.reading_review_basis
        ),
        ir_version="",
    )
    return replace(bound, ir_version=compute_ir_version(bound))


def collect_review_axis_binding_findings(
    ir: FaithfulSemanticIR,
    plain_reading_review: PlainReadingReview | None,
) -> tuple[FaithfulIRFinding, ...]:
    """Inspect review-axis identity/content without mutating or binding the IR."""

    if plain_reading_review is None:
        return ()
    reviewed = tuple(plain_reading_review.ambiguities)
    matched_review_ids: set[str] = set()
    findings: list[FaithfulIRFinding] = []
    node_map = ir.node_map
    for ambiguity_index, ambiguity in enumerate(ir.ambiguities):
        ambiguity_path = f"/faithful_ir/ambiguities/{ambiguity_index}"
        exact = next(
            (
                item
                for item in reviewed
                if item.ambiguity_id == ambiguity.ambiguity_id
            ),
            None,
        )
        match = exact
        if exact is not None:
            if ambiguity.ambiguity_id in matched_review_ids:
                findings.append(
                    FaithfulIRFinding(
                        code="IR_REVIEW_AXIS_DUPLICATE",
                        path=ambiguity_path + "/ambiguity_id",
                        message=(
                            "IR_REVIEW_AXIS_DUPLICATE: ambiguity "
                            f"{ambiguity.ambiguity_id} reuses an already-bound "
                            "plain-reading review id"
                        ),
                        allowed_paths=("/faithful_ir/ambiguities",),
                    )
                )
                continue
            if not _plain_reading_axis_alternatives_match(ambiguity, exact):
                findings.append(
                    FaithfulIRFinding(
                        code="IR_REVIEW_AXIS_CONTENT_MISMATCH",
                        path=ambiguity_path + "/alternatives",
                        message=(
                            "IR_REVIEW_AXIS_CONTENT_MISMATCH: ambiguity "
                            f"{ambiguity.ambiguity_id} reuses a plain-reading "
                            "review id but its alternatives do not match the "
                            "reviewed axis"
                        ),
                        allowed_paths=(ambiguity_path + "/alternatives",),
                    )
                )
        else:
            matches = tuple(
                item
                for item in reviewed
                if item.ambiguity_id not in matched_review_ids
                and _plain_reading_axis_text_matches(ambiguity, item)
            )
            match = matches[0] if len(matches) == 1 else None
        if match is None:
            continue
        matched_review_ids.add(match.ambiguity_id)
        try:
            _assert_reviewed_branch_matches_ir(ambiguity, match, node_map)
        except FaithfulIRValidationError as exc:
            findings.extend(exc.findings)
    return tuple(findings)


def _matching_plain_reading_ambiguity(
    ambiguity: SemanticAmbiguity,
    reviewed: tuple[object, ...],
    *,
    excluded_ids: set[str],
    ambiguity_index: int,
) -> object | None:
    exact = next(
        (
            item
            for item in reviewed
            if getattr(item, "ambiguity_id", "") == ambiguity.ambiguity_id
        ),
        None,
    )
    if exact is not None:
        if ambiguity.ambiguity_id in excluded_ids:
            raise FaithfulIRValidationError(
                "IR_REVIEW_AXIS_DUPLICATE: ambiguity "
                f"{ambiguity.ambiguity_id} reuses an already-bound plain-reading "
                "review id",
                code="IR_REVIEW_AXIS_DUPLICATE",
                path=f"/faithful_ir/ambiguities/{ambiguity_index}/ambiguity_id",
            )
        if not _plain_reading_axis_alternatives_match(ambiguity, exact):
            raise FaithfulIRValidationError(
                "IR_REVIEW_AXIS_CONTENT_MISMATCH: ambiguity "
                f"{ambiguity.ambiguity_id} reuses a plain-reading review id but "
                "its alternatives do not match the reviewed axis",
                code="IR_REVIEW_AXIS_CONTENT_MISMATCH",
                path=f"/faithful_ir/ambiguities/{ambiguity_index}/alternatives",
            )
        # The reviewed artifact owns the canonical question, options, and
        # decision.  A model may describe the selected branch as provenance
        # rather than repeat the review question verbatim; the exact trusted id
        # plus matching alternatives is sufficient to bind that same axis.
        return exact
    matches = []
    for item in reviewed:
        item_id = str(getattr(item, "ambiguity_id", ""))
        if not item_id or item_id in excluded_ids:
            continue
        if _plain_reading_axis_text_matches(ambiguity, item):
            matches.append(item)
    return matches[0] if len(matches) == 1 else None


def _plain_reading_axis_text_matches(
    ambiguity: SemanticAmbiguity,
    reviewed: object,
) -> bool:
    description = _normalized_ambiguity_text(ambiguity.description)
    question = _normalized_ambiguity_text(
        str(getattr(reviewed, "question", ""))
    )
    if not description or description != question:
        return False
    return _plain_reading_axis_alternatives_match(ambiguity, reviewed)


def _plain_reading_axis_alternatives_match(
    ambiguity: SemanticAmbiguity,
    reviewed: object,
) -> bool:
    alternatives = frozenset(
        _normalized_ambiguity_text(value)
        for value in ambiguity.alternatives
        if _normalized_ambiguity_text(value)
    )
    full_options = frozenset(
        _normalized_ambiguity_text(
            str(getattr(option, "plain_interpretation", ""))
        )
        for option in getattr(reviewed, "options", ())
        if _normalized_ambiguity_text(
            str(getattr(option, "plain_interpretation", ""))
        )
    )
    summaries = frozenset(
        _normalized_ambiguity_text(
            str(getattr(option, "difference_summary", ""))
        )
        for option in getattr(reviewed, "options", ())
        if _normalized_ambiguity_text(
            str(getattr(option, "difference_summary", ""))
        )
    )
    return bool(alternatives) and alternatives in {full_options, summaries}


def _assert_reviewed_branch_matches_ir(
    ambiguity: SemanticAmbiguity,
    reviewed: object,
    node_map: Mapping[str, SemanticNode],
) -> None:
    """Check a selected plain-reading branch when executable patches exist.

    Free-form readings have no general deterministic text-to-graph proof.  A
    complete alternative-patch set does provide such a proof, so do not mark
    that axis resolved unless the current node already reflects the selected
    patch.
    """

    if str(getattr(reviewed, "decision_status", "")) != "reviewer_selected":
        return
    patches = ambiguity.alternative_patches
    options = tuple(getattr(reviewed, "options", ()))
    selected_id = str(getattr(reviewed, "decision", ""))
    selected_index = next(
        (
            index
            for index, option in enumerate(options)
            if str(getattr(option, "option_id", "")) == selected_id
        ),
        None,
    )
    if not patches or selected_index is None:
        return
    selected_patch = next(
        (patch for patch in patches if patch.alternative_id == selected_id),
        None,
    )
    if selected_patch is None and len(patches) == len(options):
        selected_patch = patches[selected_index]
    if selected_patch is None:
        return
    node = node_map.get(selected_patch.node_id)
    matches = node is not None
    if matches and selected_patch.node_type:
        matches = node.node_type == selected_patch.node_type
    if matches:
        node_parameters = dict(node.parameters)
        matches = all(
            node_parameters.get(name) == value
            for name, value in selected_patch.parameter_updates
        )
    if not matches:
        raise FaithfulIRValidationError(
            "IR_REVIEWED_BRANCH_CONFLICT: selected plain-reading option "
            f"{getattr(reviewed, 'ambiguity_id', '')}={selected_id} is not "
            "materialized in the affected IR node",
            code="IR_REVIEWED_BRANCH_CONFLICT",
            path="/faithful_ir/nodes",
            repairable=False,
        )


def _normalized_ambiguity_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def rule_semantic_to_ir(
    semantic: SemanticStructure,
    requirement: str,
    *,
    requirement_id: str = "",
    ap_binding_context: APBindingContext | None = None,
) -> FaithfulSemanticIR:
    """Compile the deterministic rules backend directly into the current IR schema."""

    declared_names = semantic.aps or (semantic.trigger, semantic.response)
    names = tuple(
        dict.fromkeys(
            name
            for name in declared_names
            if name and name not in {"unknown", "unknown_ap"}
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        )
    )
    aps: list[AtomicProposition] = []
    nodes: list[SemanticNode] = []
    leaf_ids: dict[str, str] = {}
    for index, name in enumerate(names, start=1):
        span = _find_span(requirement, name)
        evidence = requirement[span[0]:span[1]] if span[0] >= 0 else name
        ap_id = f"ap-{index}"
        node_id = f"ir-ap-{index}"
        aps.append(AtomicProposition(ap_id, name, source_span=span, evidence_text=evidence))
        nodes.append(
            SemanticNode(
                node_id=node_id,
                node_type="ap_condition",
                semantic_role=(
                    "trigger" if name == semantic.trigger else
                    "response" if name == semantic.response else "participant"
                ),
                parameters=(("ap_name", name),),
                source_span=span,
                evidence_text=evidence,
                confidence=semantic.confidence,
            )
        )
        leaf_ids[name] = node_id

    if not nodes:
        span = (0, len(requirement))
        aps.append(AtomicProposition("ap-1", "unknown_ap", source_span=span, evidence_text=requirement))
        nodes.append(SemanticNode("ir-ap-1", "ap_condition", parameters=(("ap_name", "unknown_ap"),), source_span=span, evidence_text=requirement, confidence=0.0))
        leaf_ids["unknown_ap"] = "ir-ap-1"

    pattern = re.sub(r"[\s-]+", "_", semantic.pattern.lower().strip())
    root_id = nodes[0].node_id
    whole_span = (0, len(requirement))
    trigger_id = leaf_ids.get(semantic.trigger, root_id)
    response_id = leaf_ids.get(semantic.response, root_id)
    component_map = dict(semantic.components)
    if (
        pattern == "response"
        and component_map.get("response_relation")
        == "post_trigger_persistent_prohibition"
        and trigger_id != response_id
    ):
        negation_id = "ir-response-prohibited"
        suffix_id = "ir-response-suffix"
        next_id = "ir-response-next"
        implication_id = "ir-response-implication"
        root_id = "ir-root"
        nodes.extend(
            (
                _relation_node(
                    negation_id,
                    "negation",
                    (response_id,),
                    requirement,
                    semantic,
                    role="prohibited_event",
                ),
                _relation_node(
                    suffix_id,
                    "global_constraint",
                    (negation_id,),
                    requirement,
                    semantic,
                    role="persistent_future_prohibition",
                ),
                _relation_node(
                    next_id,
                    "next_step",
                    (suffix_id,),
                    requirement,
                    semantic,
                    role="strict_future_start",
                ),
                _relation_node(
                    implication_id,
                    "implication",
                    (trigger_id, next_id),
                    requirement,
                    semantic,
                    role="triggered_prohibition",
                ),
                _relation_node(
                    root_id,
                    "global_constraint",
                    (implication_id,),
                    requirement,
                    semantic,
                    role="requirement_scope",
                ),
            )
        )
    elif (
        pattern == "response"
        and component_map.get("response_relation")
        == "next_until_witness_or_persist_to_end"
        and len(names) >= 3
    ):
        witness_id = leaf_ids[names[2]]
        persist_id = "ir-response-persist"
        choice_id = "ir-response-choice"
        until_id = "ir-response-until"
        next_id = "ir-response-next"
        root_id = "ir-root"
        nodes.extend(
            (
                _relation_node(persist_id, "persist_to_end", (response_id,), requirement, semantic, role="suffix_persistence"),
                _relation_node(choice_id, "disjunction", (witness_id, persist_id), requirement, semantic, role="until_witness_choice"),
                _relation_node(
                    until_id,
                    "until",
                    (response_id, choice_id),
                    requirement,
                    semantic,
                    role="until_response",
                    parameters=(
                        ("witness_policy", "required"),
                        ("witness_timing", "current_or_future"),
                        ("left_at_witness", "not_required"),
                        ("scope_origin", "inherited"),
                    ),
                ),
                _relation_node(next_id, "next_step", (until_id,), requirement, semantic, role="next_response"),
                _relation_node(root_id, "implication", (trigger_id, next_id), requirement, semantic, role="triggered_requirement"),
            )
        )
    elif pattern == "until" and len(nodes) >= 2:
        root_id = "ir-root"
        left_id = nodes[0].node_id
        if str(component_map.get("until_left_negated", "")).lower() == "true":
            negated_left_id = "ir-until-left-negated"
            nodes.append(
                _relation_node(
                    negated_left_id,
                    "negation",
                    (left_id,),
                    requirement,
                    semantic,
                    role="until_left_condition",
                )
            )
            left_id = negated_left_id
        nodes.append(
            _relation_node(
                root_id,
                "until",
                (left_id, nodes[1].node_id),
                requirement,
                semantic,
                parameters=_rule_until_parameters(component_map),
            )
        )
    elif pattern in {"conjunction", "and"} and len(nodes) >= 2:
        root_id = "ir-root"
        nodes.append(_relation_node(root_id, "conjunction", tuple(node.node_id for node in nodes), requirement, semantic))
    elif pattern in {"disjunction", "or"} and len(nodes) >= 2:
        root_id = "ir-root"
        nodes.append(_relation_node(root_id, "disjunction", tuple(node.node_id for node in nodes), requirement, semantic))
    elif pattern in {"precedence", "before"} and len(nodes) >= 2:
        root_id = "ir-root"
        constrained_name, prerequisite_name = _rule_precedence_roles(semantic)
        constrained_id = leaf_ids.get(constrained_name, trigger_id)
        prerequisite_id = leaf_ids.get(prerequisite_name, response_id)
        nodes.append(
            _relation_node(
                root_id,
                "precedence",
                (prerequisite_id, constrained_id),
                requirement,
                semantic,
                role="strict_precedence_gate",
                parameters=_rule_precedence_parameters(component_map),
            )
        )
    elif pattern == "response" and trigger_id != response_id:
        temporal_type = _response_temporal_type(semantic)
        response_child = response_id
        if temporal_type:
            temporal_id = "ir-response-temporal"
            nodes.append(_relation_node(temporal_id, temporal_type, (response_id,), requirement, semantic, role="response_requirement"))
            response_child = temporal_id
        implication_id = "ir-implication"
        nodes.append(_relation_node(implication_id, "implication", (trigger_id, response_child), requirement, semantic, role="triggered_requirement"))
        root_id = implication_id
        if _has_global_scope(semantic):
            root_id = "ir-root"
            nodes.append(_relation_node(root_id, "global_constraint", (implication_id,), requirement, semantic, role="requirement_scope"))
    elif pattern in {"persistence", "global_state", "invariance"}:
        root_id = "ir-root"
        nodes.append(_relation_node(root_id, "global_constraint", (response_id,), requirement, semantic))

    reachable: set[str] = set()
    provisional_map = {node.node_id: node for node in nodes}

    def mark_reachable(node_id: str) -> None:
        if node_id in reachable or node_id not in provisional_map:
            return
        reachable.add(node_id)
        for child_id in provisional_map[node_id].children:
            mark_reachable(child_id)

    mark_reachable(root_id)
    disconnected = tuple(node.node_id for node in nodes if node.node_id not in reachable)
    nodes = [node for node in nodes if node.node_id in reachable]

    ambiguity_items: list[SemanticAmbiguity] = []
    if pattern == "until" and not _rule_until_policy_complete(component_map):
        ambiguity_items.append(
            SemanticAmbiguity(
                ambiguity_id="amb-until-policy",
                description=(
                    "Explicit until wording fixes the left/right relation but not the "
                    "strong/weak witness, immediate-witness, or boundary-inclusion policy."
                ),
                affected_node_ids=(root_id,),
                alternatives=(
                    "strong until with a required current-or-future witness",
                    "weak until where the left side may persist without a witness",
                ),
                risk="high",
            )
        )
    if (
        pattern in {"precedence", "before"}
        and not _rule_precedence_parameters(component_map)
    ):
        ambiguity_items.append(
            SemanticAmbiguity(
                ambiguity_id="amb-precedence-policy",
                description=(
                    "The text supports a prerequisite gate, but strictness, same-step, "
                    "reuse, or scope policy remains unresolved."
                ),
                affected_node_ids=(root_id,),
                alternatives=(
                    "strict reusable prerequisite from the trace start",
                    "non-strict or locally activated prerequisite gate",
                ),
                risk="high",
            )
        )
    if (
        semantic.confidence < 0.65
        or semantic.pattern_status == "unknown"
        or bool(disconnected)
        or semantic.trigger not in leaf_ids
        or semantic.response not in leaf_ids
    ):
        ambiguity_items.append(
            SemanticAmbiguity(
                ambiguity_id="amb-1",
                description=(
                    "Rule-based semantic summary cannot prove the complete nested structure"
                    + (f"; unmapped AP nodes: {', '.join(disconnected)}" if disconnected else "")
                    + "."
                ),
                affected_node_ids=(root_id,),
                alternatives=(
                    "regenerate a connected IR from the natural-language evidence",
                    "request manual semantic review",
                ),
                risk="high",
            )
        )
    ambiguities = tuple(ambiguity_items)
    ir = FaithfulSemanticIR(
        requirement_id=(
            requirement_id
            or (
                ap_binding_context.requirement_id
                if ap_binding_context is not None
                else _requirement_id(requirement)
            )
        ),
        source_text=requirement,
        atomic_propositions=tuple(aps),
        root_node_id=root_id,
        nodes=tuple(nodes),
        ambiguities=ambiguities,
        extraction_confidence=semantic.confidence,
        provenance=IRProvenance(
            extractor="deterministic_faithful_ir_rule",
            extraction_mode="rules_direct_v5",
        ),
        schema_version=FAITHFUL_IR_SCHEMA_VERSION,
        contract_version=FAITHFUL_IR_CONTRACT_VERSION,
        canonicalizer_version=FAITHFUL_IR_CANONICALIZER_VERSION,
        validator_version=FAITHFUL_IR_VALIDATOR_VERSION,
    )
    if ap_binding_context is not None:
        ir = bind_ir_to_ap_context(ir, ap_binding_context)
    ir = replace(ir, ir_version=compute_ir_version(ir))
    validate_faithful_ir(ir)
    return ir


def semantic_summary_from_ir(
    ir: FaithfulSemanticIR,
    *,
    rag_context: tuple[str, ...] = (),
) -> SemanticStructure:
    """Build a derived summary that cannot override the attached IR."""

    view = build_ir_semantic_view(ir)
    ap_names = tuple(name for _node_id, name in view.ap_names_by_node_id)
    trigger_names: tuple[str, ...] = ()
    response_names: tuple[str, ...] = ()
    relation = view.primary_relation
    if relation is not None and relation.antecedent_node_id:
        trigger_names = view.ap_names_under(ir.node_map, relation.antecedent_node_id)
        response_names = view.ap_names_under(ir.node_map, relation.consequent_node_id)
    trigger = trigger_names[0] if trigger_names else ap_names[0] if ap_names else "unknown_ap"
    response = (
        response_names[0]
        if response_names
        else ap_names[1]
        if len(ap_names) > 1
        else trigger
    )
    root = ir.node_map[ir.root_node_id]
    return SemanticStructure(
        trigger=trigger,
        response=response,
        temporal_keyword=root.temporal_relation,
        scope_keyword=root.scope,
        pattern=_pattern_summary(root.node_type),
        explanation="Compatibility summary derived from FaithfulSemanticIR.",
        confidence=ir.extraction_confidence,
        rag_context=rag_context,
        aps=ir.ap_names,
        faithful_ir=ir,
        semantic_status="needs_semantic_review" if ir.has_high_risk_ambiguity else "ready",
        semantic_source_contract_version=ir.semantic_source_contract_version,
        effective_source_fingerprint=ir.effective_source_fingerprint,
        effective_source_kind=ir.effective_source_kind,
        effective_source_authority_record_id=(
            ir.effective_source_authority_record_id
        ),
    )


def reference_formula_to_ir(
    formula: Any,
    *,
    requirement_id: str,
    source_text: str,
) -> FaithfulSemanticIR:
    """Offline-only adapter for producing a pending formula-migration IR.

    This helper must not be called by online NLU.  It records its oracle origin
    explicitly so stage visibility controls can keep the source formula hidden.
    A formula alone cannot establish reviewed segmentation, ordinary-language
    reading, or an ApprovedSemanticIR, so the returned provenance is deliberately
    unapproved.
    """

    atom_names: list[str] = []

    def collect(current: Any) -> None:
        if current.op == "ap" and current.name not in atom_names:
            atom_names.append(current.name)
        for child in current.args:
            collect(child)

    collect(formula)
    aps = tuple(
        AtomicProposition(
            ap_id=f"ap-{index}",
            name=name,
            source_span=_find_span(source_text, name),
            evidence_text=(
                source_text[slice(*_find_span(source_text, name))]
                if _find_span(source_text, name)[0] >= 0
                else source_text
            ),
        )
        for index, name in enumerate(atom_names, start=1)
    )
    ap_ids = {ap.name: ap.ap_id for ap in aps}
    nodes: list[SemanticNode] = []
    counter = 0
    op_types = {
        "not": "negation",
        "and": "conjunction",
        "or": "disjunction",
        "implies": "implication",
        "equiv": "equivalence",
        "iff": "equivalence",
        "G": "global_constraint",
        "F": "eventually",
        "X": "next_step",
        "U": "until",
        "R": "release",
        "true": "boolean_constant",
        "false": "boolean_constant",
    }

    def build(current: Any) -> str:
        nonlocal counter
        counter += 1
        node_id = f"ir-n{counter}"
        if current.op == "ap":
            span = _find_span(source_text, current.name)
            evidence = source_text[span[0]:span[1]] if span[0] >= 0 else source_text
            nodes.append(
                SemanticNode(
                    node_id=node_id,
                    node_type="ap_condition",
                    semantic_role="formula_migration_leaf",
                    parameters=(("ap_name", current.name),),
                    source_span=span,
                    evidence_text=evidence,
                    confidence=1.0,
                )
            )
            return node_id
        child_ids = tuple(build(child) for child in current.args)
        node_type = op_types.get(current.op)
        if node_type is None:
            raise FaithfulIRValidationError(
                f"unsupported reference operator during offline migration: {current.op}"
            )
        parameters: tuple[tuple[str, Any], ...]
        if current.op == "U":
            parameters = (
                ("witness_policy", "required"),
                ("witness_timing", "current_or_future"),
                ("left_at_witness", "not_required"),
                ("scope_origin", "inherited"),
            )
        elif current.op in {"true", "false"}:
            parameters = (("value", current.op == "true"),)
        elif current.op == "X":
            parameters = (("steps", 1),)
        else:
            parameters = ()
        nodes.append(
            SemanticNode(
                node_id=node_id,
                node_type=node_type,
                children=child_ids,
                semantic_role="offline_reference_migration",
                scope="all_positions" if current.op == "G" else "inherited",
                temporal_relation=node_type if node_type in TEMPORAL_NODE_TYPES else "none",
                boundary_policy="finite_trace_end_inclusive",
                parameters=parameters,
                source_span=(0, len(source_text)),
                evidence_text=source_text,
                confidence=1.0,
            )
        )
        return node_id

    root_id = build(formula)
    ir = FaithfulSemanticIR(
        requirement_id=requirement_id,
        source_text=source_text,
        atomic_propositions=aps,
        root_node_id=root_id,
        nodes=tuple(nodes),
        extraction_confidence=1.0,
        provenance=IRProvenance(
            extractor="offline_reference_formula_migrator",
            extraction_mode="offline_reference_migration_pending_review",
            source="approved_formula_migration",
            approved=False,
            approval_source="",
        ),
        schema_version=FAITHFUL_IR_SCHEMA_VERSION,
        contract_version=FAITHFUL_IR_CONTRACT_VERSION,
        canonicalizer_version=FAITHFUL_IR_CANONICALIZER_VERSION,
        validator_version=FAITHFUL_IR_VALIDATOR_VERSION,
    )
    ir = replace(ir, ir_version=compute_ir_version(ir))
    validate_faithful_ir(ir)
    return ir


def _ap_from_mapping(payload: Mapping[str, Any]) -> AtomicProposition:
    return AtomicProposition(
        ap_id=str(payload.get("ap_id", "")),
        name=str(payload.get("name", "")),
        ap_identity_id=str(payload.get("ap_identity_id", "")),
        ap_mention_ids=tuple(
            str(item) for item in payload.get("ap_mention_ids", ())
        ),
        aliases=tuple(str(item) for item in payload.get("aliases", ())),
        source_span=_span_tuple(payload.get("source_span")),
        evidence_text=str(payload.get("evidence_text", "")),
        supporting_segment_ids=tuple(
            str(item) for item in payload.get("supporting_segment_ids", ())
        ),
        formula_presence=str(payload.get("formula_presence", "unspecified")),
        required_in_formula=bool(payload.get("required_in_formula", False)),
        optional_in_formula=bool(payload.get("optional_in_formula", False)),
    )


def _node_from_mapping(payload: Mapping[str, Any]) -> SemanticNode:
    parameters = payload.get("parameters", {})
    if isinstance(parameters, Mapping):
        parameter_pairs = tuple((str(key), value) for key, value in parameters.items())
    else:
        parameter_pairs = tuple((str(item[0]), item[1]) for item in parameters if isinstance(item, (list, tuple)) and len(item) == 2)
    return SemanticNode(
        node_id=str(payload.get("node_id", "")),
        node_type=_normalize_node_type(payload.get("node_type", "")),
        children=tuple(str(item) for item in payload.get("children", ())),
        semantic_role=str(payload.get("semantic_role", "")),
        scope=str(payload.get("scope", "current_position")),
        temporal_relation=str(payload.get("temporal_relation", "none")),
        boundary_policy=str(payload.get("boundary_policy", "finite_trace_default")),
        allowed_semantic_readings=tuple(
            str(item)
            for item in payload.get("allowed_semantic_readings", ())
        ),
        temporal_resolution=str(payload.get("temporal_resolution", "fixed")),
        parameters=parameter_pairs,
        source_span=_span_tuple(payload.get("source_span")),
        evidence_text=str(payload.get("evidence_text", "")),
        confidence=_float(payload.get("confidence"), 0.0),
        supporting_segment_ids=tuple(
            str(item) for item in payload.get("supporting_segment_ids", ())
        ),
    )


def _normalize_node_type(value: object) -> str:
    raw = str(value).strip()
    normalized = re.sub(r"[\s-]+", "_", raw.lower())
    if normalized in SUPPORTED_NODE_TYPES:
        return normalized
    return raw


def _repair_ap_evidence_span(
    ap: AtomicProposition,
    source_text: str,
    *,
    surface_segmentation: SurfaceSegmentation | None = None,
) -> AtomicProposition:
    # Evidence produced by the authoritative proposal canonicalizer is already
    # complete when its span selects the exact evidence text.  In particular,
    # surrounding whitespace is part of an exact evidence quote and must not
    # trigger a second, token-oriented relocation here.
    if (
        _exact_evidence_span_matches(
            source_text,
            ap.source_span,
            ap.evidence_text,
        )
        and not _evidence_matches_ap_name(ap.evidence_text, ap.name)
    ):
        return ap
    span = _repairable_evidence_span(
        source_text,
        ap.source_span,
        ap.evidence_text,
        supporting_segment_ids=ap.supporting_segment_ids,
        surface_segmentation=surface_segmentation,
    )
    if span == ap.source_span and _evidence_matches_ap_name(ap.evidence_text, ap.name):
        span = _nearest_ap_token_span(source_text, ap.name, ap.source_span)
    if span == ap.source_span or not _exact_evidence_span_matches(
        source_text,
        span,
        ap.evidence_text,
    ):
        return ap
    return replace(ap, source_span=span)


def _repair_node_evidence_span(
    node: SemanticNode,
    source_text: str,
    *,
    surface_segmentation: SurfaceSegmentation | None = None,
) -> SemanticNode:
    ap_name = (
        str(node.parameter("ap_name", ""))
        if node.node_type == "ap_condition"
        else ""
    )
    if (
        _exact_evidence_span_matches(
            source_text,
            node.source_span,
            node.evidence_text,
        )
        and not _evidence_matches_ap_name(node.evidence_text, ap_name)
    ):
        return node
    span = _repairable_evidence_span(
        source_text,
        node.source_span,
        node.evidence_text,
        supporting_segment_ids=node.supporting_segment_ids,
        surface_segmentation=surface_segmentation,
    )
    if span == node.source_span and node.node_type == "ap_condition":
        if _evidence_matches_ap_name(node.evidence_text, ap_name):
            span = _nearest_ap_token_span(source_text, ap_name, node.source_span)
    if span == node.source_span or not _exact_evidence_span_matches(
        source_text,
        span,
        node.evidence_text,
    ):
        return node
    return replace(node, source_span=span)


def _evidence_matches_ap_name(evidence_text: str, ap_name: str) -> bool:
    return bool(ap_name) and evidence_text == ap_name


def _exact_evidence_span_matches(
    source_text: str,
    span: tuple[int, int],
    evidence_text: str,
) -> bool:
    """Return whether a backend-derived span exactly selects its evidence."""

    start, end = span
    return bool(
        evidence_text
        and 0 <= start < end <= len(source_text)
        and source_text[start:end] == evidence_text
    )


def _nearest_ap_token_span(
    source_text: str,
    ap_name: str,
    preferred_span: tuple[int, int],
) -> tuple[int, int]:
    candidates = _ap_token_spans(source_text, ap_name)
    if not candidates:
        return preferred_span

    start, end = preferred_span
    if 0 <= start < end <= len(source_text):
        center = (start + end) / 2
        return min(
            candidates,
            key=lambda candidate: (
                abs(((candidate[0] + candidate[1]) / 2) - center),
                abs(candidate[0] - start),
                candidate[0],
            ),
        )
    if len(candidates) == 1:
        return candidates[0]
    return preferred_span


def _ap_token_spans(source_text: str, ap_name: str) -> list[tuple[int, int]]:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ap_name):
        return []
    return [
        (match.start(), match.end())
        for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", source_text)
        if match.group(0) == ap_name
    ]


def _repairable_evidence_span(
    source_text: str,
    span: tuple[int, int],
    evidence_text: str,
    *,
    supporting_segment_ids: tuple[str, ...] = (),
    surface_segmentation: SurfaceSegmentation | None = None,
) -> tuple[int, int]:
    start, end = span
    if (
        0 <= start < end <= len(source_text)
        and evidence_text
        and source_text[start:end] == evidence_text
    ):
        return span
    if not evidence_text:
        return span

    trimmed_span = _trimmed_matching_span(source_text, span, evidence_text)
    if trimmed_span is not None:
        return trimmed_span

    candidates = _evidence_occurrences(source_text, evidence_text)
    if not candidates:
        return span
    if len(candidates) == 1:
        return candidates[0]

    constrained = _segment_constrained_candidates(
        candidates,
        supporting_segment_ids=supporting_segment_ids,
        surface_segmentation=surface_segmentation,
    )
    if len(constrained) == 1:
        return constrained[0]

    overlapping = [
        candidate
        for candidate in constrained or candidates
        if _spans_intersect(candidate, span)
    ]
    if len(overlapping) == 1:
        return overlapping[0]
    return span


def _trimmed_matching_span(
    source_text: str,
    span: tuple[int, int],
    evidence_text: str,
) -> tuple[int, int] | None:
    """Repair whitespace-only boundary drift without weakening exact evidence."""

    start, end = span
    if not (0 <= start < end <= len(source_text)):
        return None
    selected = source_text[start:end]
    left_trimmed = selected.lstrip()
    leading = len(selected) - len(left_trimmed)
    trimmed = left_trimmed.rstrip()
    if trimmed != evidence_text:
        return None
    return (start + leading, start + leading + len(evidence_text))


def _evidence_occurrences(
    source_text: str,
    evidence_text: str,
) -> list[tuple[int, int]]:
    if not evidence_text:
        return []
    return [
        (match.start(), match.end())
        for match in re.finditer(re.escape(evidence_text), source_text)
    ]


def _segment_constrained_candidates(
    candidates: list[tuple[int, int]],
    *,
    supporting_segment_ids: tuple[str, ...],
    surface_segmentation: SurfaceSegmentation | None,
) -> list[tuple[int, int]]:
    if surface_segmentation is None or not supporting_segment_ids:
        return []
    supported_spans = [
        (segment.start, segment.end)
        for segment in surface_segmentation.segments
        if segment.segment_id in supporting_segment_ids
    ]
    if not supported_spans:
        return []
    return [
        candidate
        for candidate in candidates
        if any(_spans_intersect(candidate, supported) for supported in supported_spans)
    ]


def _spans_intersect(
    left: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _ambiguity_from_mapping(payload: Mapping[str, Any]) -> SemanticAmbiguity:
    raw_patches = payload.get("alternative_patches", ())
    if raw_patches is None:
        raw_patches = ()
    if not isinstance(raw_patches, (list, tuple)):
        raise FaithfulIRValidationError("ambiguity.alternative_patches must be an array")
    patches: list[SemanticAlternativePatch] = []
    for index, raw_patch in enumerate(raw_patches):
        if not isinstance(raw_patch, Mapping):
            raise FaithfulIRValidationError(
                f"ambiguity.alternative_patches[{index}] must be an object"
            )
        raw_updates = raw_patch.get("parameter_updates", {})
        if isinstance(raw_updates, Mapping):
            updates = tuple((str(key), value) for key, value in raw_updates.items())
        elif isinstance(raw_updates, (list, tuple)):
            updates = tuple(
                (str(item[0]), item[1])
                for item in raw_updates
                if isinstance(item, (list, tuple)) and len(item) == 2
            )
            if len(updates) != len(raw_updates):
                raise FaithfulIRValidationError(
                    f"ambiguity.alternative_patches[{index}].parameter_updates "
                    "must contain key/value pairs"
                )
        else:
            raise FaithfulIRValidationError(
                f"ambiguity.alternative_patches[{index}].parameter_updates "
                "must be an object or pair array"
            )
        patches.append(
            SemanticAlternativePatch(
                alternative_id=str(raw_patch.get("alternative_id", "")),
                node_id=str(raw_patch.get("node_id", "")),
                parameter_updates=updates,
                node_type=str(raw_patch.get("node_type", "")),
                description=str(raw_patch.get("description", "")),
            )
        )
    return SemanticAmbiguity(
        ambiguity_id=str(payload.get("ambiguity_id", "")),
        description=str(payload.get("description", "")),
        affected_node_ids=tuple(str(item) for item in payload.get("affected_node_ids", ())),
        alternatives=tuple(str(item) for item in payload.get("alternatives", ())),
        alternative_patches=tuple(patches),
        risk=str(payload.get("risk", "medium")),
        resolved=bool(payload.get("resolved", False)),
        origin_layer=str(payload.get("origin_layer", "ir_extraction")),
        axis_kind=str(payload.get("axis_kind", "")),
        related_axis_ids=tuple(
            str(item) for item in payload.get("related_axis_ids", ())
        ),
        relation_kind=str(payload.get("relation_kind", "independent")),
    )


def _trusted_parent_id(declared: str, trusted: str, label: str) -> str:
    if declared and trusted and declared != trusted:
        raise FaithfulIRValidationError(
            f"faithful_ir.{label} does not match the reviewed parent artifact"
        )
    return trusted or declared


def _supporting_segment_id_errors(
    segment_ids: tuple[str, ...],
    label: str,
) -> list[str]:
    errors: list[str] = []
    if any(not segment_id for segment_id in segment_ids):
        errors.append(f"{label} has an empty supporting_segment_id")
    if len(segment_ids) != len(set(segment_ids)):
        errors.append(f"{label} supporting_segment_ids must be unique")
    return errors


def _parent_artifact_errors(
    ir: FaithfulSemanticIR,
    surface_segmentation: SurfaceSegmentation | None,
    plain_reading_review: PlainReadingReview | None,
) -> list[str]:
    errors: list[str] = []
    source_hash = hashlib.sha256(ir.source_text.encode("utf-8")).hexdigest()
    if ir.reading_review_id and not ir.segmentation_id:
        errors.append("reading_review_id requires segmentation_id")
    if ir.reading_review_basis and not ir.reading_review_id:
        errors.append("reading_review_basis requires reading_review_id")

    if surface_segmentation is not None:
        if surface_segmentation.source_text != ir.source_text:
            errors.append("surface segmentation source_text does not match IR source_text")
        if surface_segmentation.source_hash != source_hash:
            errors.append("surface segmentation source_hash does not match IR source_text")
        if ir.segmentation_id != surface_segmentation.segmentation_id:
            errors.append("IR segmentation_id does not match surface segmentation")

    if plain_reading_review is not None:
        if plain_reading_review.source_hash != source_hash:
            errors.append("plain reading source_hash does not match IR source_text")
        if ir.reading_review_id != plain_reading_review.reading_review_id:
            errors.append("IR reading_review_id does not match plain reading review")
        if ir.segmentation_id != plain_reading_review.segmentation_id:
            errors.append("IR segmentation_id does not match plain reading parent")
        if (
            surface_segmentation is not None
            and plain_reading_review.segmentation_id
            != surface_segmentation.segmentation_id
        ):
            errors.append("plain reading review references a different segmentation")
    return errors


def _segment_reference_errors(
    segment_ids: tuple[str, ...],
    span: tuple[int, int],
    known_segments: Mapping[str, Any],
    label: str,
) -> list[str]:
    if not segment_ids:
        return [f"{label} must reference at least one supporting segment"]
    unknown = tuple(
        segment_id for segment_id in segment_ids if segment_id not in known_segments
    )
    if unknown:
        return [
            f"{label} references unknown supporting segments: "
            + ", ".join(unknown)
        ]
    start, end = span
    if not any(
        known_segments[segment_id].start < end
        and start < known_segments[segment_id].end
        for segment_id in segment_ids
    ):
        return [f"{label} source_span does not overlap its supporting segments"]
    return []


def _object_list(payload: Mapping[str, Any], key: str, *, required: bool = True) -> list[Mapping[str, Any]]:
    value = payload.get(key)
    if value is None and not required:
        return []
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise FaithfulIRValidationError(f"{key} must be an array of objects")
    return value


def _evidence_errors(
    source: str,
    span: tuple[int, int],
    evidence: str,
    label: str,
    *,
    supporting_segment_ids: tuple[str, ...] = (),
) -> list[str]:
    start, end = span
    if start < 0 or end <= start or end > len(source):
        return [
            f"{label} has invalid source_span: span=[{start},{end}), "
            f"source_length={len(source)}"
        ]
    if not evidence:
        return [f"{label} has empty evidence_text: span=[{start},{end})"]
    if source[start:end] != evidence:
        occurrences = _evidence_occurrences(source, evidence)
        occurrence_text = "[" + ", ".join(
            f"[{candidate_start},{candidate_end})"
            for candidate_start, candidate_end in occurrences[:8]
        ) + ("]" if len(occurrences) <= 8 else ", ...]")
        return [
            f"{label} evidence_text does not match source_span: "
            f"span=[{start},{end}), evidence={_diagnostic_repr(evidence)}, "
            f"actual_slice={_diagnostic_repr(source[start:end])}, "
            f"occurrences={occurrence_text}, "
            f"supporting_segment_ids={list(supporting_segment_ids)!r}"
        ]
    return []


def _diagnostic_repr(value: str, limit: int = 120) -> str:
    if len(value) <= limit:
        return repr(value)
    return repr(value[: limit - 3] + "...")


def _trusted_parent_id(declared: str, actual: str, field_name: str) -> str:
    """Use backend parent ids and reject an LLM attempt to select another parent."""

    if actual:
        if declared and declared != actual:
            raise FaithfulIRValidationError(
                f"faithful_ir.{field_name} does not match the reviewed parent"
            )
        return actual
    # Without a supplied parent the caller cannot verify a declared identifier;
    # do not treat an LLM self-report as trusted provenance.
    return ""


def _supporting_segment_id_errors(
    segment_ids: tuple[str, ...],
    label: str,
) -> list[str]:
    if any(not segment_id for segment_id in segment_ids):
        return [f"{label} has an empty supporting_segment_id"]
    if len(segment_ids) != len(set(segment_ids)):
        return [f"{label} has duplicate supporting_segment_ids"]
    return []


def _parent_artifact_errors(
    ir: FaithfulSemanticIR,
    surface_segmentation: SurfaceSegmentation | None,
    plain_reading_review: PlainReadingReview | None,
) -> list[str]:
    errors: list[str] = []
    if surface_segmentation is not None:
        from surface_segmentation import source_hash, validate_surface_segmentation

        validation = validate_surface_segmentation(
            surface_segmentation,
            current_source_text=ir.source_text,
        )
        if validation.status != "passed":
            errors.append(
                "surface segmentation failed mechanical validation: "
                + validation.failure_type
            )
        if surface_segmentation.status not in {"surface_reviewed", "ambiguity_flagged"}:
            errors.append("surface segmentation has not passed review")
        if surface_segmentation.source_hash != source_hash(ir.source_text):
            errors.append("surface segmentation source_hash is stale")
        if ir.segmentation_id != surface_segmentation.segmentation_id:
            errors.append("IR segmentation_id does not match reviewed segmentation")

    if plain_reading_review is not None:
        if surface_segmentation is None:
            errors.append("plain reading review requires a surface segmentation parent")
        else:
            try:
                from plain_reading import assert_plain_reading_ready_for_semantic_ir

                assert_plain_reading_ready_for_semantic_ir(
                    plain_reading_review,
                    surface_segmentation,
                )
            except ValueError as exc:
                errors.append("plain reading gate failed: " + str(exc))
        if ir.reading_review_id != plain_reading_review.reading_review_id:
            errors.append("IR reading_review_id does not match reviewed reading")
    return errors


def _segment_reference_errors(
    segment_ids: tuple[str, ...],
    source_span: tuple[int, int],
    known_segments: Mapping[str, object],
    label: str,
) -> list[str]:
    if not segment_ids:
        return [f"{label} must cite supporting_segment_ids"]
    unknown = set(segment_ids) - set(known_segments)
    if unknown:
        return [
            f"{label} references unknown supporting segments: "
            + ", ".join(sorted(unknown))
        ]
    start, end = source_span
    covered = [known_segments[segment_id] for segment_id in segment_ids]
    if not any(
        getattr(segment, "start") < end and getattr(segment, "end") > start
        for segment in covered
    ):
        return [f"{label} supporting segments do not overlap source_span"]
    return []


def _segments_for_span(
    source_span: tuple[int, int],
    segmentation: SurfaceSegmentation,
) -> tuple[str, ...]:
    start, end = source_span
    return tuple(
        segment.segment_id
        for segment in segmentation.segments
        if segment.start < end and segment.end > start
    )


def _ensure_default_pre_ir_artifacts(
    semantic: SemanticStructure,
    requirement: str,
    *,
    requirement_id: str,
) -> SemanticStructure:
    """Backward-compatible rules gates for direct programmatic NLU callers."""

    segmentation = semantic.surface_segmentation
    reading = semantic.plain_reading_review
    if segmentation is None:
        from agents.segmentation_agent import approve_rule_segmentation
        from surface_segmentation import build_surface_segmentation, deterministic_marked_text

        effective_id = (
            requirement_id
            or (semantic.faithful_ir.requirement_id if semantic.faithful_ir else "")
            or _requirement_id(requirement)
        )
        marked_text, marker = deterministic_marked_text(requirement)
        segmentation = approve_rule_segmentation(
            build_surface_segmentation(
                requirement_id=effective_id,
                source_text=requirement,
                marked_text=marked_text,
                marker=marker,
                status="review_pending",
                experiment_mode="rules",
                data_split=semantic.data_split,
                label_source="rules",
                review_basis="mechanical_and_rules",
            )
        )
    if reading is None:
        from plain_reading import build_plain_reading_review

        reading = build_plain_reading_review(
            requirement_id=segmentation.requirement_id,
            segmentation=segmentation,
            ambiguities=(),
            overall_status="no_ambiguity",
            review_basis="plain_reading_disabled",
            reviewer_source="rules",
            experiment_mode="off",
            data_split=semantic.data_split,
        )
    return replace(
        semantic,
        surface_segmentation=segmentation,
        plain_reading_review=reading,
    )


def _relation_node(
    node_id: str,
    node_type: str,
    children: tuple[str, ...],
    requirement: str,
    semantic: SemanticStructure,
    role: str = "",
    parameters: tuple[tuple[str, Any], ...] = (),
) -> SemanticNode:
    if node_type == "next_step" and not parameters:
        parameters = (("steps", 1),)
    return SemanticNode(
        node_id=node_id,
        node_type=node_type,
        children=children,
        semantic_role=role or node_type,
        scope="all_positions" if node_type == "global_constraint" else "inherited",
        temporal_relation=node_type if node_type in TEMPORAL_NODE_TYPES else "none",
        boundary_policy="finite_trace_end_inclusive",
        parameters=parameters,
        source_span=(0, len(requirement)),
        evidence_text=requirement,
        confidence=semantic.confidence,
    )


def _rule_until_parameters(
    components: Mapping[str, str],
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (name, value)
        for name in UNTIL_PARAMETER_VALUES
        if (value := str(components.get(name, "")).strip())
    )


def _rule_until_policy_complete(components: Mapping[str, str]) -> bool:
    return all(str(components.get(name, "")).strip() for name in UNTIL_PARAMETER_VALUES)


def _rule_precedence_parameters(
    components: Mapping[str, str],
) -> tuple[tuple[str, Any], ...]:
    if str(components.get("precedence_policy_unresolved", "")).lower() == "true":
        return ()
    defaults = {
        "ordering": "strict",
        "witness_policy": "optional_if_no_constrained",
        "same_step_policy": "forbidden",
        "reuse_policy": "reusable",
        "scope_origin": "trace_start",
    }
    return tuple(
        (name, str(components.get(name, default)).strip() or default)
        for name, default in defaults.items()
    )


def precedence_roles_from_semantic(
    semantic: SemanticStructure,
) -> tuple[str, str]:
    components = dict(semantic.components)
    constrained = str(components.get("constrained_event", "")).strip()
    prerequisite = str(components.get("prerequisite_event", "")).strip()
    if constrained and prerequisite:
        return constrained, prerequisite
    relation = str(components.get("precedence_relation", ""))
    match = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s+only\s+after\s+prior\s+"
        r"\b([A-Za-z_][A-Za-z0-9_]*)\b",
        relation,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1), match.group(2)
    return semantic.trigger, semantic.response


_rule_precedence_roles = precedence_roles_from_semantic


def _response_temporal_type(semantic: SemanticStructure) -> str:
    text = semantic.temporal_keyword.lower()
    if "next" in text or "下一" in text:
        return "next_step"
    if "always" in text or "持续" in text:
        return "persist_to_end"
    if "event" in text or "final" in text or "最终" in text:
        return "eventually"
    return ""


def _has_global_scope(semantic: SemanticStructure) -> bool:
    text = semantic.scope_keyword.lower()
    return any(word in text for word in ("whenever", "every", "always", "每当", "始终"))


def _pattern_summary(node_type: str) -> str:
    return {
        "implication": "response",
        "global_constraint": "response",
        "until": "until",
        "precedence": "precedence",
        "conjunction": "conjunction",
        "disjunction": "disjunction",
    }.get(node_type, node_type)


def _find_span(text: str, evidence: str) -> tuple[int, int]:
    match = re.search(re.escape(evidence), text, flags=re.IGNORECASE)
    return (match.start(), match.end()) if match else (0, len(text))


def _span_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        return int(value.get("start", -1)), int(value.get("end", -1))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return -1, -1


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _requirement_id(source_text: str) -> str:
    digest = hashlib.sha1(source_text.encode("utf-8")).hexdigest()[:12]
    return f"REQ-{digest}"
