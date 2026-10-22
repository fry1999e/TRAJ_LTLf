"""Deterministic compilation from FaithfulSemanticIR into stage plans."""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools

from data_examples import LabeledTrace
from contract_versions import OBLIGATION_COMPILER_VERSION
from faithful_ir import (
    PERIODICITY_REQUIRED_PARAMETERS,
    TEMPORAL_NODE_TYPES,
    validate_faithful_ir,
)
from formula_closure import (
    build_strict_formula_closure_policy,
    validate_formula_closure,
)
from ir_trace_semantics import IRTraceSemanticsError, compare_ir_and_formula, evaluate_ir_trace
from ir_relation_feasibility import (
    IRRelationFeasibilityError,
    analyze_ir_relation_feasibility,
)
from ltlf_parser import And, Atom, F, FALSE, Formula, G, Iff, Implies, Not, Or, TRUE, U, X, parse_formula
from ltlf_syntax import validate_no_repeated_negation
from operator_confusions import OPERATOR_CONFUSION_CATALOG
from semantic_discriminators import (
    DISCRIMINATOR_WITNESS_FOUND,
    discriminator_values,
    synthesize_discriminator_trace,
)
from semantic_node_contracts import discriminator_horizon, structural_horizon
from semantic_readiness import (
    CAN_BUILD_DISCRIMINATION_PLAN,
    CAN_GENERATE_AUTHORITATIVE_TRACES,
    audit_semantic_readiness,
    require_semantic_capability,
)
from temporal_resolution import (
    allowed_operators_for_node,
    allowed_semantic_readings,
    resolved_node_kind,
)
from schemas import (
    CompletionHoleConstraint,
    CompletionHolePlan,
    ConstrainedSkeletonPlan,
    FaithfulSemanticIR,
    IRCoverageReport,
    OperatorResolutionReport,
    SkeletonPlanNode,
    TraceObligation,
    TraceObligationPlan,
)


@dataclass(frozen=True)
class NodeLoweringCertificate:
    """Auditable record of the compiler lowering exactly one IR node.

    This is deliberately produced by the compiler itself.  It is not another
    formula-shape recognizer and therefore cannot disagree with the lowering
    rules that created the formula.
    """

    node_id: str
    source_node_type: str
    resolved_kind: str
    child_node_ids: tuple[str, ...]
    child_formula_texts: tuple[str, ...]
    output_formula_text: str
    parameters: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class CompiledFormulaArtifact:
    """The only releasable formula representation of an approved IR."""

    ir_version: str
    formula: Formula
    lowering_certificate: tuple[NodeLoweringCertificate, ...]
    compiler_version: int = OBLIGATION_COMPILER_VERSION

    def validate_for(self, ir: FaithfulSemanticIR) -> None:
        if self.ir_version != ir.ir_version:
            raise ValueError(
                "compiled artifact ir_version mismatch: "
                f"expected {ir.ir_version}, got {self.ir_version}"
            )
        if self.compiler_version != OBLIGATION_COMPILER_VERSION:
            raise ValueError(
                "compiled artifact compiler version mismatch: "
                f"expected {OBLIGATION_COMPILER_VERSION}, got {self.compiler_version}"
            )
        records = {record.node_id: record for record in self.lowering_certificate}
        if len(records) != len(self.lowering_certificate):
            raise ValueError("lowering certificate contains duplicate node ids")
        if set(records) != set(ir.node_map):
            missing = sorted(set(ir.node_map) - set(records))
            extra = sorted(set(records) - set(ir.node_map))
            raise ValueError(
                "lowering certificate does not cover the IR exactly: "
                f"missing={missing}, extra={extra}"
            )
        for node in ir.nodes:
            record = records[node.node_id]
            if record.source_node_type != node.node_type:
                raise ValueError(
                    f"lowering certificate node type mismatch at {node.node_id}"
                )
            if record.child_node_ids != node.children:
                raise ValueError(
                    f"lowering certificate child order mismatch at {node.node_id}"
                )
            if record.parameters != node.parameters:
                raise ValueError(
                    f"lowering certificate parameters mismatch at {node.node_id}"
                )
        root_record = records[ir.root_node_id]
        if root_record.output_formula_text != str(self.formula):
            raise ValueError("lowering certificate root does not match compiled formula")


MAX_IR_SCOPE_DEPTH = 12
MAX_TRACE_OBLIGATIONS = 128
MAX_CONFUSION_TARGETS = 32
_COMPILED_ARTIFACT_CACHE: dict[
    tuple[int, str, tuple[tuple[str, str], ...]],
    CompiledFormulaArtifact,
] = {}
_TRACE_OBLIGATION_PLAN_CACHE: dict[
    tuple[int, str],
    TraceObligationPlan,
] = {}


def compile_trace_obligation_plan(ir: FaithfulSemanticIR) -> TraceObligationPlan:
    """Derive the complete structural obligation inventory.

    This compatibility API is intentionally not an execution authorization.
    Runtime callers must use :func:`compile_executable_trace_plan`; unresolved
    semantic review uses :func:`compile_semantic_discrimination_plan`.
    """

    validate_faithful_ir(ir)
    cache_key = (OBLIGATION_COMPILER_VERSION, ir.ir_version)
    # Persisted IRs have a content-derived version.  Hand-built/unit-test IRs
    # may deliberately leave it blank; those objects must never share a cache
    # entry merely because their version metadata is absent.
    cached = (
        _TRACE_OBLIGATION_PLAN_CACHE.get(cache_key)
        if ir.ir_version
        else None
    )
    if cached is not None:
        return cached
    _enforce_ir_budget(ir)
    feasibility = analyze_ir_relation_feasibility(ir)
    obligations: list[TraceObligation] = []
    confusions: list[str] = []
    root_horizon = structural_horizon(ir)
    for node in ir.nodes:
        if node.node_type == "ap_condition":
            continue
        # Local diagnostic obligations must not inherit unrelated exact-offset
        # depth from the root.  Only the root label is required to exercise the
        # complete IR horizon; non-root obligations use their own executable
        # subtree horizon.
        minimum_horizon = max(
            _minimum_horizon(node.node_type, node.parameters),
            structural_horizon(ir, node.node_id),
            root_horizon if node.node_id == ir.root_node_id else 1,
        )
        semantic_parameters = tuple(node.parameters)
        semantic_parameter_paths = _semantic_parameter_paths(node)
        typed_constraints = _typed_trace_constraints(node)
        local_evidence_mode = (
            "root_label" if node.node_id == ir.root_node_id else "diagnostic_only"
        )
        root_relation = node.node_id == ir.root_node_id
        if not root_relation or feasibility.satisfy_status == "witness_found":
            obligations.append(
                TraceObligation(
                obligation_id=f"obl-{node.node_id}-sat",
                ir_node_ids=(node.node_id,),
                required_trace_role="positive_witness",
                construction_constraints=(f"exercise {node.node_type}", *typed_constraints),
                expected_relation="satisfy",
                minimum_horizon=minimum_horizon,
                semantic_parameters=semantic_parameters,
                semantic_parameter_paths=semantic_parameter_paths,
                evidence_mode=local_evidence_mode,
                feasibility_report_hash=(
                    feasibility.artifact_hash if root_relation else ""
                ),
                gating=root_relation,
            )
            )
        if not root_relation or feasibility.violate_status == "witness_found":
            obligations.append(
                TraceObligation(
                obligation_id=f"obl-{node.node_id}-violate",
                ir_node_ids=(node.node_id,),
                required_trace_role="minimal_violation",
                construction_constraints=(
                    f"violate only {node.node_type} where possible",
                    *typed_constraints,
                ),
                expected_relation="violate",
                minimum_horizon=minimum_horizon,
                semantic_parameters=semantic_parameters,
                semantic_parameter_paths=semantic_parameter_paths,
                evidence_mode=local_evidence_mode,
                feasibility_report_hash=(
                    feasibility.artifact_hash if root_relation else ""
                ),
                gating=root_relation,
            )
            )
        allowed_operators = allowed_operators_for_node(node)
        if allowed_operators:
            obligations.append(
                TraceObligation(
                    obligation_id=f"obl-{node.node_id}-operator-resolution",
                    ir_node_ids=(node.node_id,),
                    required_trace_role="operator_resolution",
                    construction_constraints=(
                        "resolve the temporal operator using reviewed whole-trace labels",
                        "retain ambiguity when multiple joint assignments pass",
                        *typed_constraints,
                    ),
                    expected_relation="resolve",
                    minimum_horizon=minimum_horizon,
                    semantic_parameters=(
                        *semantic_parameters,
                        ("allowed_operators", allowed_operators),
                    ),
                    semantic_parameter_paths=semantic_parameter_paths,
                    evidence_mode="operator_resolution",
                )
            )
        boundary = _boundary_case(node.node_type)
        if boundary:
            boundary_horizon = (
                1
                if node.node_type == "periodicity"
                else (
                    max(minimum_horizon, discriminator_horizon(node))
                )
            )
            boundary_parameter_paths = _semantic_parameter_paths(
                node,
                names=_boundary_parameter_names(node.node_type),
            )
            obligations.append(
                TraceObligation(
                    obligation_id=f"obl-{node.node_id}-boundary",
                    ir_node_ids=(node.node_id,),
                    required_trace_role="boundary",
                    construction_constraints=(boundary,),
                    expected_relation="distinguish",
                    minimum_horizon=boundary_horizon,
                    boundary_case=boundary,
                    semantic_parameters=semantic_parameters,
                    semantic_parameter_paths=boundary_parameter_paths,
                    ambiguity_axis_ids=(boundary,),
                    evidence_mode="explicit_discriminator",
                )
            )
        catalog_entries = _catalog_confusions_for_unresolved_node(ir, node)
        obligations.extend(
            entry.to_trace_obligation(
                node.node_id,
                minimum_horizon=preflight.witness_length,
                semantic_parameters=semantic_parameters,
            )
            for entry, preflight in catalog_entries
            if preflight.witness_length is not None
        )
        confusions.extend(entry.confusion_id for entry, _preflight in catalog_entries)
        if len(obligations) > MAX_TRACE_OBLIGATIONS:
            raise ValueError(
                "IR trace-obligation budget exceeded: "
                f"{len(obligations)} > {MAX_TRACE_OBLIGATIONS}. "
                "Split the requirement or request semantic review."
            )
        if len(dict.fromkeys(confusions)) > MAX_CONFUSION_TARGETS:
            raise ValueError(
                "IR confusion-target budget exceeded: "
                f"{len(dict.fromkeys(confusions))} > {MAX_CONFUSION_TARGETS}. "
                "Split the requirement or request semantic review."
            )
    if not obligations:
        root = ir.node_map[ir.root_node_id]
        if feasibility.satisfy_status == "witness_found":
            obligations.append(
                TraceObligation(
                    f"obl-{root.node_id}-sat",
                    (root.node_id,),
                    "positive_witness",
                    expected_relation="satisfy",
                    feasibility_report_hash=feasibility.artifact_hash,
                    gating=True,
                )
            )
        if feasibility.violate_status == "witness_found":
            obligations.append(
                TraceObligation(
                    f"obl-{root.node_id}-violate",
                    (root.node_id,),
                    "minimal_violation",
                    expected_relation="violate",
                    feasibility_report_hash=feasibility.artifact_hash,
                    gating=True,
                )
            )
    readiness = audit_semantic_readiness(ir)
    plan = TraceObligationPlan(
        requirement_id=ir.requirement_id,
        ir_version=ir.ir_version,
        obligations=tuple(obligations),
        confusion_targets=tuple(dict.fromkeys(confusions)),
        coverage_requirements=(
            (
                "root obligations are exactly the realizable sides proven by "
                f"feasibility report {feasibility.artifact_hash}"
            ),
            "non-root satisfy/violate checks are diagnostic and non-gating",
            "every temporal IR node has a trajectory-resolved operator assignment",
            "all boundary obligations meet minimum horizon",
            "all declared confusion targets are distinguished",
        ),
        plan_purpose="structural_analysis",
        readiness_fingerprint=readiness.fingerprint,
        execution_blockers=readiness.messages_for(
            CAN_GENERATE_AUTHORITATIVE_TRACES
        ),
        ap_context_fingerprint=ir.ap_context_fingerprint,
        feasibility_report_hash=feasibility.artifact_hash,
        feasibility_classification=feasibility.classification,
        relation_feasibility=feasibility,
    )
    if ir.ir_version:
        if len(_TRACE_OBLIGATION_PLAN_CACHE) >= 256:
            _TRACE_OBLIGATION_PLAN_CACHE.clear()
        _TRACE_OBLIGATION_PLAN_CACHE[cache_key] = plan
    return plan


def compile_executable_trace_plan(ir: FaithfulSemanticIR) -> TraceObligationPlan:
    """Compile satisfy/violate witnesses only for resolved executable IR."""

    readiness = require_semantic_capability(
        ir,
        CAN_GENERATE_AUTHORITATIVE_TRACES,
    )
    structural = compile_trace_obligation_plan(ir)
    feasibility = structural.relation_feasibility
    if feasibility is not None and feasibility.classification in {
        "unknown_due_to_resource_limit",
        "unsupported_semantics",
    }:
        code = (
            "IR_RELATION_FEASIBILITY_RESOURCE_LIMIT"
            if feasibility.classification == "unknown_due_to_resource_limit"
            else "IR_RELATION_FEASIBILITY_UNSUPPORTED"
        )
        raise IRRelationFeasibilityError(
            code,
            "Root relation feasibility cannot authorize trace obligations.",
            report=feasibility,
            suggested_rerun_stage=(
                "trace"
                if feasibility.classification == "unknown_due_to_resource_limit"
                else "semantic"
            ),
        )
    return replace(
        structural,
        plan_purpose="executable_witness",
        readiness_fingerprint=readiness.fingerprint,
        execution_blockers=(),
    )


def compile_semantic_discrimination_plan(
    ir: FaithfulSemanticIR,
) -> TraceObligationPlan:
    """Compile only non-authoritative ambiguity-discrimination obligations."""

    readiness = audit_semantic_readiness(ir)
    if not readiness.structurally_valid:
        require_semantic_capability(ir, CAN_BUILD_DISCRIMINATION_PLAN)
    structural = compile_trace_obligation_plan(ir)
    obligations = tuple(
        obligation
        for obligation in structural.obligations
        if obligation.expected_relation in {"resolve", "distinguish"}
    )
    return replace(
        structural,
        obligations=obligations,
        coverage_requirements=(
            "all unresolved semantic axes are represented by discriminator obligations",
            "candidate semantics must not supply adjudicated truth labels",
            "all declared confusion targets are distinguished",
        ),
        plan_purpose="semantic_discrimination",
        readiness_fingerprint=readiness.fingerprint,
    )


def compile_review_trace_plan(ir: FaithfulSemanticIR) -> TraceObligationPlan:
    """Select a plan purpose for review orchestration, never for execution."""

    readiness = audit_semantic_readiness(ir)
    return (
        compile_executable_trace_plan(ir)
        if readiness.has_capability(CAN_GENERATE_AUTHORITATIVE_TRACES)
        else compile_semantic_discrimination_plan(ir)
    )


def _catalog_confusions_for_unresolved_node(ir, node):
    """Require neighbour discrimination for unresolved current-schema nodes."""

    affected = any(
        not ambiguity.resolved
        and (
            not ambiguity.affected_node_ids
            or node.node_id in ambiguity.affected_node_ids
        )
        for ambiguity in ir.ambiguities
    )
    parents = tuple(
        parent for parent in ir.nodes if node.node_id in parent.children
    )

    def has_global_ancestor(current_id: str, seen: frozenset[str] = frozenset()) -> bool:
        if current_id in seen:
            return False
        current_parents = tuple(
            parent for parent in ir.nodes if current_id in parent.children
        )
        return any(
            parent.node_type in {"global_constraint", "persist_to_end"}
            or has_global_ancestor(parent.node_id, seen | {current_id})
            for parent in current_parents
        )

    catalog = OPERATOR_CONFUSION_CATALOG
    entries = tuple(
        entry
        for entry in catalog.for_ir_node_kind(
            node.node_type
        )
        if entry.trace_distinguishable
    )
    entries = tuple(
        entry
        for entry in entries
        if entry.confusion_id != "next_depth_neighbor"
    )
    if node.node_type == "next_step" and int(node.parameter("steps", 1)) != 1:
        entries = tuple(
            entry for entry in entries if entry.confusion_id != "next_one_vs_next_two"
        )
    if node.node_type == "eventually":
        child = ir.node_map.get(node.children[0]) if node.children else None
        if child is None or child.node_type != "conjunction":
            entries = tuple(
                entry
                for entry in entries
                if entry.confusion_id != "simultaneous_vs_independent_eventuality"
            )
        direct_response = node.node_id == ir.root_node_id or any(
            parent.node_type == "implication"
            and len(parent.children) == 2
            and parent.children[1] == node.node_id
            for parent in parents
        )
        if not direct_response:
            entries = tuple(
                entry
                for entry in entries
                if entry.confusion_id != "eventually_vs_next"
            )
    if node.node_type == "conjunction":
        independent_eventuals = bool(node.children) and all(
            ir.node_map[child].node_type == "eventually"
            and len(ir.node_map[child].children) == 1
            for child in node.children
        )
        if not independent_eventuals:
            entries = tuple(
                entry
                for entry in entries
                if entry.confusion_id != "simultaneous_vs_independent_eventuality"
            )
    if node.node_type == "next_step":
        direct_response = node.node_id == ir.root_node_id or any(
            parent.node_type == "implication"
            and len(parent.children) == 2
            and parent.children[1] == node.node_id
            for parent in parents
        )
        if not direct_response:
            entries = tuple(
                entry
                for entry in entries
                if entry.confusion_id
                not in {
                    "eventually_vs_next",
                    "strict_eventually_vs_next",
                    "next_one_vs_next_two",
                    "current_vs_strict_future",
                }
            )
    if node.node_type in {"global_constraint", "persist_to_end"} and has_global_ancestor(
        node.node_id
    ):
        entries = tuple(
            entry
            for entry in entries
            if entry.confusion_id
            not in {"current_vs_global_scope", "local_implication_vs_global_response"}
        )
    if node.node_type == "until" and any(
        parent.node_type == "disjunction"
        and any(
            sibling.node_type in {"global_constraint", "persist_to_end"}
            and sibling.children == (node.children[0],)
            for sibling in (
                ir.node_map[sibling_id]
                for sibling_id in parent.children
                if sibling_id != node.node_id
            )
        )
        for parent in parents
    ):
        entries = tuple(
            entry
            for entry in entries
            if entry.confusion_id != "strong_until_vs_weak_until"
        )
    if node.node_type == "implication" and any(
        parent.node_type in {"global_constraint", "persist_to_end"}
        and node.node_id in parent.children
        for parent in ir.nodes
    ):
        entries = tuple(
            entry
            for entry in entries
            if entry.confusion_id != "local_implication_vs_global_response"
        )
    preflight_horizon = max(
        structural_horizon(ir),
        discriminator_horizon(node),
    )
    feasible_entries = []
    for entry in entries:
        synthesis = synthesize_discriminator_trace(
            ir,
            node.node_id,
            entry.confusion_id,
            minimum_horizon=max(preflight_horizon, entry.minimum_horizon),
            max_candidates=100_000,
        )
        if synthesis.status == DISCRIMINATOR_WITNESS_FOUND:
            feasible_entries.append((entry, synthesis))
    return tuple(feasible_entries)


def bind_traces_to_ir_plan(
    traces: list[LabeledTrace],
    plan: TraceObligationPlan,
    *,
    generation_source: str = "ir_plan_binding",
) -> list[LabeledTrace]:
    """Bind explicit formula-free trace claims to the current IR version.

    Trace kind alone is not evidence for every obligation with the same
    expected relation.  A current trace must carry an obligation or IR-node
    claim from its pending-query artifact; otherwise it remains unbound and
    cannot satisfy IR coverage.
    """

    bound: list[LabeledTrace] = []
    for index, trace in enumerate(traces, start=1):
        relation = (
            "distinguish" if trace.kind == "boundary"
            else "satisfy" if trace.expected
            else "violate"
        )
        claimed_ids = set(trace.obligation_ids) | set(trace.semantics.obligations)
        claimed_nodes = set(trace.covered_ir_node_ids)
        relation_candidates = tuple(
            item
            for item in plan.obligations
            if (
                item.evidence_mode == "root_label"
                and item.expected_relation == relation
                or item.evidence_mode == "explicit_discriminator"
                and relation == "distinguish"
                and item.expected_relation == "distinguish"
            )
            and len(trace.trace) >= item.minimum_horizon
        )
        obligations = tuple(
            item
            for item in plan.obligations
            if item.obligation_id in claimed_ids
            and item.expected_relation in {relation, "resolve"}
            and len(trace.trace) >= item.minimum_horizon
        )
        if not obligations and not claimed_ids and claimed_nodes:
            obligations = tuple(
                item
                for item in relation_candidates
                if claimed_nodes.intersection(item.ir_node_ids)
            )
        semantics = replace(
            trace.semantics,
            roles=tuple(
                dict.fromkeys(
                    item.required_trace_role for item in obligations
                )
            ),
            obligations=tuple(
                dict.fromkeys(
                    item.obligation_id for item in obligations
                )
            ),
            confusion_targets=tuple(
                dict.fromkeys(trace.semantics.confusion_targets)
            ),
        )
        bound.append(
            replace(
                trace,
                semantics=semantics,
                trace_id=trace.trace_id or trace.name or f"trace-{index}",
                covered_ir_node_ids=tuple(
                    dict.fromkeys(
                        node_id for item in obligations for node_id in item.ir_node_ids
                    )
                ),
                obligation_ids=tuple(
                    dict.fromkeys(
                        item.obligation_id for item in obligations
                    )
                ),
                trace_role=(
                    (obligations[0].required_trace_role if obligations else "")
                    or trace.role
                    or trace.kind
                ),
                boundary_role=trace.boundary_role or (
                    trace.target_ambiguity if trace.kind == "boundary" else ""
                ),
                generation_source=trace.generation_source or generation_source,
                ir_version=plan.ir_version,
            )
        )
    return bound


def compile_constrained_skeleton_plan(ir: FaithfulSemanticIR) -> ConstrainedSkeletonPlan:
    validate_faithful_ir(ir)
    parents: dict[str, str] = {}
    for node in ir.nodes:
        for child in node.children:
            parents[child] = node.node_id
    plan_nodes = tuple(
        SkeletonPlanNode(
            skeleton_node_id=f"sk-{node.node_id}",
            ir_node_id=node.node_id,
            parent_id=f"sk-{parents[node.node_id]}" if node.node_id in parents else "",
            children=tuple(f"sk-{child}" for child in node.children),
            node_type=node.node_type,
            fixed=node.node_type not in TEMPORAL_NODE_TYPES,
            allowed_operators=allowed_operators_for_node(node),
            scope_boundary=node.node_type in {"global_constraint", "bounded_window", "fixed_duration"},
        )
        for node in ir.nodes
    )
    return ConstrainedSkeletonPlan(
        requirement_id=ir.requirement_id,
        ir_version=ir.ir_version,
        root_skeleton_node_id=f"sk-{ir.root_node_id}",
        nodes=plan_nodes,
        fixed_ir_node_ids=tuple(
            node.node_id for node in ir.nodes if node.node_type not in TEMPORAL_NODE_TYPES
        ),
    )


def compile_completion_hole_plan(
    ir: FaithfulSemanticIR,
    skeleton_plan: ConstrainedSkeletonPlan,
) -> CompletionHolePlan:
    _require_version(ir.ir_version, skeleton_plan.ir_version, "SkeletonPlan")
    holes: list[CompletionHoleConstraint] = []
    temporal_nodes = {
        node.ir_node_id: node
        for node in skeleton_plan.nodes
        if node.allowed_operators
    }
    for ir_node_id, skeleton_node in temporal_nodes.items():
        parent_hole = ""
        parent = next(
            (item for item in skeleton_plan.nodes if item.skeleton_node_id == skeleton_node.parent_id),
            None,
        )
        if parent is not None and parent.ir_node_id in temporal_nodes:
            parent_hole = f"H-{parent.ir_node_id}"
        holes.append(
            CompletionHoleConstraint(
                hole_id=f"H-{ir_node_id}",
                ir_node_id=ir_node_id,
                skeleton_node_id=skeleton_node.skeleton_node_id,
                allowed_operators=skeleton_node.allowed_operators,
                parent_hole_id=parent_hole,
                scope_locked=True,
            )
        )
    return CompletionHolePlan(
        requirement_id=ir.requirement_id,
        ir_version=ir.ir_version,
        skeleton_plan_version=skeleton_plan.plan_version,
        holes=tuple(holes),
    )


def evaluate_ir_coverage(
    ir: FaithfulSemanticIR,
    trace_plan: TraceObligationPlan,
    traces: list[LabeledTrace],
    *,
    skeleton_plan: ConstrainedSkeletonPlan | None = None,
    hole_plan: CompletionHolePlan | None = None,
    operator_resolution: OperatorResolutionReport | None = None,
) -> IRCoverageReport:
    """Evaluate structural, reviewed-trace, and operator-resolution coverage."""

    reasons: list[str] = []
    version_consistent = trace_plan.ir_version == ir.ir_version
    if skeleton_plan is not None:
        version_consistent = version_consistent and skeleton_plan.ir_version == ir.ir_version
    if hole_plan is not None:
        version_consistent = version_consistent and hole_plan.ir_version == ir.ir_version
    if operator_resolution is not None and operator_resolution.status != "not_required":
        version_consistent = (
            version_consistent and operator_resolution.ir_version == ir.ir_version
        )
    if not version_consistent:
        reasons.append("derived artifact ir_version mismatch")

    covered_node_ids: set[str] = set()
    covered_obligation_ids: set[str] = set()
    covered_confusions: set[str] = set()
    obligation_map = {item.obligation_id: item for item in trace_plan.obligations}
    operator_obligations = {
        item.obligation_id: item
        for item in trace_plan.obligations
        if item.evidence_mode == "operator_resolution"
    }
    if operator_resolution is not None and operator_resolution.complete:
        resolved_nodes = set(operator_resolution.resolved_ir_node_ids)
        for obligation_id, obligation in operator_obligations.items():
            if set(obligation.ir_node_ids) <= resolved_nodes:
                covered_obligation_ids.add(obligation_id)
                covered_node_ids.update(obligation.ir_node_ids)
    for trace in traces:
        if trace.ir_version and trace.ir_version != ir.ir_version:
            reasons.append(f"trace {trace.name} uses stale ir_version")
            continue
        claimed_ids = set(_trace_obligation_ids(trace))
        target_nodes = {
            node_id
            for obligation_id in claimed_ids
            if obligation_id in obligation_map
            for node_id in obligation_map[obligation_id].ir_node_ids
        } or set(trace.covered_ir_node_ids) or {ir.root_node_id}
        try:
            node_values = {
                node_id: _possible_ir_node_values(ir, trace.trace, node_id)
                for node_id in target_nodes
            }
        except IRTraceSemanticsError as exc:
            reasons.append(f"trace {trace.name} cannot be checked against IR: {exc}")
            continue
        coherent_claims = set()
        for obligation_id in claimed_ids:
            obligation = obligation_map.get(obligation_id)
            if obligation is None:
                continue
            if ir.root_node_id in obligation.ir_node_ids:
                coherent = trace.expected in node_values.get(ir.root_node_id, frozenset())
            elif obligation.expected_relation == "satisfy":
                coherent = all(
                    node_values.get(node_id, frozenset()) == frozenset({True})
                    for node_id in obligation.ir_node_ids
                )
            elif obligation.expected_relation == "violate":
                coherent = all(
                    node_values.get(node_id, frozenset()) == frozenset({False})
                    for node_id in obligation.ir_node_ids
                )
            else:
                confusion_id = str(
                    dict(obligation.semantic_parameters).get("confusion_id", "")
                )
                if confusion_id and obligation.ir_node_ids:
                    target_node_id = obligation.ir_node_ids[0]
                    discriminator = discriminator_values(
                        ir,
                        target_node_id,
                        confusion_id,
                        trace.trace,
                    )
                    coherent = (
                        discriminator is not None
                        and discriminator[0] != discriminator[1]
                        and trace.expected == discriminator[0]
                    )
                else:
                    coherent = all(
                        bool(node_values.get(node_id, frozenset()))
                        for node_id in obligation.ir_node_ids
                    )
            if coherent:
                coherent_claims.add(obligation_id)
        if claimed_ids and not coherent_claims:
            reasons.append(
                f"trace {trace.name} label conflicts with independent IR semantics: "
                f"whole_label={trace.expected}, possible_node_values={node_values}"
            )
        trace_roles = set(_trace_roles(trace))
        valid_claims = {
            obligation_id
            for obligation_id in coherent_claims
            if obligation_id in obligation_map
            and (
                ir.root_node_id not in obligation_map[obligation_id].ir_node_ids
                or _trace_matches_expected_relation(
                    trace,
                    obligation_map[obligation_id].expected_relation,
                )
            )
            and obligation_map[obligation_id].required_trace_role
            in trace_roles
            and len(trace.trace) >= obligation_map[obligation_id].minimum_horizon
        }
        covered_obligation_ids.update(valid_claims)
        for obligation_id in valid_claims:
            covered_node_ids.update(obligation_map[obligation_id].ir_node_ids)
        valid_confusion_ids = {
            str(dict(obligation_map[obligation_id].semantic_parameters).get("confusion_id", ""))
            for obligation_id in valid_claims
            if obligation_id in obligation_map
        }
        valid_confusion_ids.discard("")
        # A discriminator axis is covered only after its claimed obligation
        # has passed the independent IR check above.  Do not require a second,
        # manually duplicated axis declaration on the trace: that metadata can
        # drift and would create a competing coverage rule.
        covered_confusions.update(valid_confusion_ids)

    required_plan_obligations = tuple(
        obligation
        for obligation in trace_plan.obligations
        if obligation.evidence_mode != "diagnostic_only"
    )
    required_nodes = {
        node_id
        for obligation in required_plan_obligations
        for node_id in obligation.ir_node_ids
    }
    required_obligations = {
        item.obligation_id for item in required_plan_obligations
    }
    missing_nodes = required_nodes - covered_node_ids
    missing_obligations = required_obligations - covered_obligation_ids
    missing_confusions = set(trace_plan.confusion_targets) - covered_confusions

    structure_mapping_complete = True
    if skeleton_plan is not None:
        mapped = {node.ir_node_id for node in skeleton_plan.nodes}
        structure_mapping_complete = set(ir.node_map) <= mapped
        if not structure_mapping_complete:
            reasons.append("skeleton plan does not map every IR node")
    if hole_plan is not None and skeleton_plan is not None:
        temporal_ids = {
            node.ir_node_id for node in skeleton_plan.nodes if node.allowed_operators
        }
        hole_ids = {hole.ir_node_id for hole in hole_plan.holes}
        structure_mapping_complete = structure_mapping_complete and temporal_ids <= hole_ids
        if not temporal_ids <= hole_ids:
            reasons.append("completion hole plan omits temporal IR nodes")

    if missing_nodes:
        reasons.append("IR nodes lack trace coverage")
    if missing_obligations:
        reasons.append("trace obligations are incomplete")
    missing_operator_obligations = set(operator_obligations) - covered_obligation_ids
    if missing_operator_obligations:
        reasons.append(
            "temporal operators are not uniquely resolved by reviewed trajectories"
        )
        if operator_resolution is not None:
            reasons.extend(operator_resolution.reasons)
    if missing_confusions:
        reasons.append("confusion targets lack distinguishing traces")
    complete = (
        version_consistent
        and structure_mapping_complete
        and not missing_nodes
        and not missing_obligations
        and not missing_confusions
    )
    return IRCoverageReport(
        requirement_id=ir.requirement_id,
        ir_version=ir.ir_version,
        status="complete" if complete else "incomplete",
        covered_ir_node_ids=tuple(sorted(covered_node_ids & required_nodes)),
        missing_ir_node_ids=tuple(sorted(missing_nodes)),
        covered_obligation_ids=tuple(sorted(covered_obligation_ids & required_obligations)),
        missing_obligation_ids=tuple(sorted(missing_obligations)),
        covered_confusions=tuple(sorted(covered_confusions & set(trace_plan.confusion_targets))),
        missing_confusions=tuple(sorted(missing_confusions)),
        structure_mapping_complete=structure_mapping_complete,
        version_consistent=version_consistent,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def assert_derived_plan_versions(
    ir: FaithfulSemanticIR,
    trace_plan: TraceObligationPlan,
    skeleton_plan: ConstrainedSkeletonPlan,
    hole_plan: CompletionHolePlan,
) -> None:
    for label, version in (
        ("TraceObligationPlan", trace_plan.ir_version),
        ("ConstrainedSkeletonPlan", skeleton_plan.ir_version),
        ("CompletionHolePlan", hole_plan.ir_version),
    ):
        _require_version(ir.ir_version, version, label)


def _enforce_ir_budget(ir: FaithfulSemanticIR) -> None:
    depth = _ir_depth(ir)
    if depth > MAX_IR_SCOPE_DEPTH:
        raise ValueError(
            "IR scope depth budget exceeded: "
            f"{depth} > {MAX_IR_SCOPE_DEPTH}. "
            "Split the requirement or request semantic review."
        )


def _ir_depth(ir: FaithfulSemanticIR) -> int:
    node_map = ir.node_map

    def depth(node_id: str, seen: set[str]) -> int:
        if node_id in seen or node_id not in node_map:
            return 0
        children = node_map[node_id].children
        if not children:
            return 1
        return 1 + max(depth(child, {*seen, node_id}) for child in children)

    return depth(ir.root_node_id, set())


def compile_ir_skeleton(ir: FaithfulSemanticIR) -> Formula:
    """Compile only fixed Boolean topology, erasing temporal wrappers."""

    validate_faithful_ir(ir)

    def build(node_id: str) -> Formula:
        node = ir.node_map[node_id]
        children = tuple(build(child) for child in node.children)
        if node.node_type == "ap_condition":
            return Atom(str(node.parameter("ap_name")))
        if node.node_type == "boolean_constant":
            return TRUE() if bool(node.parameter("value")) else FALSE()
        if node.node_type == "negation":
            return Not(children[0])
        if node.node_type == "conjunction":
            return _fold_binary(And, children)
        if node.node_type == "disjunction":
            return _fold_binary(Or, children)
        if node.node_type == "implication":
            return Implies(children[0], children[1])
        if node.node_type == "equivalence":
            return Iff(children[0], children[1])
        if node.node_type == "until":
            # Preserve the ordered binary relation. High-risk policy choices are
            # carried by the IR and expanded only by compile_ir_to_formula().
            return U(children[0], children[1])
        if node.node_type == "release":
            return Not(U(Not(children[0]), Not(children[1])))
        if node.node_type in {"before", "precedence"}:
            prerequisite, event = children[0], children[1]
            ordering = str(node.parameter("ordering", "strict"))
            witness = (
                And(prerequisite, Not(event))
                if ordering == "strict"
                else prerequisite
            )
            return U(Not(event), witness)
        if len(children) == 1:
            return children[0]
        if children:
            return _fold_binary(And, children)
        raise ValueError(f"IR node {node.node_id} cannot be represented as a skeleton")

    return build(ir.root_node_id)


def _lower_ir_node(
    node,
    children: tuple[Formula, ...],
    *,
    kind: str,
    ir: FaithfulSemanticIR,
) -> Formula:
    """Lower one typed node.  All compiler entry points use this authority."""

    if kind == "ap_condition":
        return Atom(str(_required_node_parameter(node, "ap_name")))
    if kind == "boolean_constant":
        return TRUE() if bool(_required_node_parameter(node, "value")) else FALSE()
    if kind == "negation":
        return Not(children[0])
    if kind == "conjunction":
        return _fold_binary(And, children)
    if kind == "disjunction":
        return _fold_binary(Or, children)
    if kind == "implication":
        return Implies(children[0], children[1])
    if kind == "equivalence":
        return Iff(children[0], children[1])
    if kind == "identity":
        return children[0]
    if kind in {"global_constraint", "persist_to_end"}:
        return G(children[0])
    if kind == "eventually":
        return F(children[0])
    if kind == "next_step":
        result = children[0]
        # A next_step reading on another unary node is the one-step X hole
        # choice. Only an actual next_step source node owns an explicit offset.
        steps = _required_node_parameter(node, "steps") if node.node_type == "next_step" else 1
        for _ in range(int(steps)):
            result = X(result)
        return result
    if kind == "eventually_strict":
        return X(F(children[0]))
    if kind == "until":
        return _compile_until_relation(node, children[0], children[1])
    if kind == "release":
        return Not(U(Not(children[0]), Not(children[1])))
    if kind == "fixed_duration":
        return _fixed_duration(
            children[0],
            int(_required_node_parameter(node, "duration")),
            short_policy=str(_required_node_parameter(node, "short_trace_policy")),
            terminal_policy=str(
                _required_node_parameter(node, "terminal_obligation_policy")
            ),
        )
    if kind == "exact_run_length":
        return _exact_run_length(
            children[0],
            start_offset=int(_required_node_parameter(node, "start_offset")),
            duration=int(_required_node_parameter(node, "duration")),
            origin=str(_required_node_parameter(node, "origin")),
            start_policy=str(
                _required_node_parameter(node, "start_position_policy")
            ),
            terminal_policy=str(
                _required_node_parameter(node, "terminal_obligation_policy")
            ),
            post_policy=str(
                _required_node_parameter(node, "post_window_policy")
            ),
        )
    if kind == "bounded_window":
        return _bounded_eventually(
            children[0],
            int(_required_node_parameter(node, "lower_bound")),
            int(_required_node_parameter(node, "upper_bound")),
            short_policy=str(_required_node_parameter(node, "short_trace_policy")),
            terminal_policy=str(
                _required_node_parameter(node, "terminal_obligation_policy")
            ),
        )
    if kind == "ordered_sequence":
        return _ordered_sequence(children)
    if kind in {"before", "precedence"}:
        return _compile_precedence_relation(node, children[0], children[1])
    if kind == "after":
        return F(And(children[0], X(F(children[1]))))
    if kind == "cardinality":
        return _cardinality(
            children[0],
            str(_required_node_parameter(node, "mode")),
            max(0, int(_required_node_parameter(node, "count"))),
        )
    if kind == "recurrence":
        _validate_recurrence_policy(node)
        return G(F(children[0]))
    if kind == "periodicity":
        return _compile_periodicity_relation(node, children[0], ir.ap_names)
    if kind == "start_boundary":
        return And(Atom(_initial_marker(ir)), children[0])
    if kind == "end_boundary":
        return F(G(children[0]))
    if kind == "short_trace_boundary":
        maximum = max(1, int(_required_node_parameter(node, "maximum_length")))
        return Not(_next_n(TRUE(), maximum))
    raise ValueError(f"IR node type {kind} needs an explicit completion compiler")


def _initial_marker(ir):
    name = "compiler_absolute_initial"
    while name in ir.ap_names:
        name += "_"
    return name


def _eliminate_initial_marker(formula, marker):
    """Evaluate the marker at absolute position zero, false on every suffix.

    Unroll each temporal operator once using its finite-trace recurrence.
    This preserves START nested inside arbitrary lowered temporal relations.
    """
    from functools import lru_cache
    @lru_cache(None)
    def transform(node, initial):
        if node.op == "ap":
            return (TRUE() if initial else FALSE()) if node.name == marker else node
        if not node.args:
            return node
        later = tuple(transform(c, False) for c in node.args)
        if not initial:
            return Formula(node.op, later, node.name)
        now = tuple(transform(c, True) for c in node.args)
        if node.op == "X":
            return X(later[0])
        if node.op == "F":
            return Or(now[0], X(F(later[0])))
        if node.op == "G":
            return And(now[0], Not(X(Not(G(later[0])))))
        if node.op == "U":
            return Or(now[1], And(now[0], X(U(*later))))
        return Formula(node.op, now, node.name)
    return transform(formula, True)


def compile_ir_artifact(
    ir: FaithfulSemanticIR,
    *,
    temporal_resolution_by_node: dict[str, str] | None = None,
) -> CompiledFormulaArtifact:
    """Compile one approved temporal assignment and issue its certificate.

    A node with more than one allowed semantic reading must be supplied in
    ``temporal_resolution_by_node``.  The independent IR evaluator receives the
    same assignment only as semantic input; it never calls this compiler.
    """

    validate_faithful_ir(ir)
    resolution_key = tuple(sorted((temporal_resolution_by_node or {}).items()))
    cache_key = (OBLIGATION_COMPILER_VERSION, ir.ir_version, resolution_key)
    cached = (
        _COMPILED_ARTIFACT_CACHE.get(cache_key)
        if ir.ir_version
        else None
    )
    if cached is not None:
        cached.validate_for(ir)
        return cached

    certificate_by_node: dict[str, NodeLoweringCertificate] = {}

    def build(node_id: str) -> Formula:
        node = ir.node_map[node_id]
        children = tuple(build(child) for child in node.children)
        try:
            kind = resolved_node_kind(node, temporal_resolution_by_node)
        except ValueError as exc:
            raise ValueError(f"invalid temporal resolution: {exc}") from exc
        result = _lower_ir_node(node, children, kind=kind, ir=ir)
        certificate_by_node[node_id] = NodeLoweringCertificate(
            node_id=node_id,
            source_node_type=node.node_type,
            resolved_kind=kind,
            child_node_ids=node.children,
            child_formula_texts=tuple(str(child) for child in children),
            output_formula_text=str(result),
            parameters=node.parameters,
        )
        return result

    formula = build(ir.root_node_id)
    if any(n.node_type == "start_boundary" for n in ir.nodes):
        formula = _eliminate_initial_marker(formula, _initial_marker(ir))
        certificate_by_node[ir.root_node_id] = replace(
            certificate_by_node[ir.root_node_id], output_formula_text=str(formula))
    closure_report = validate_formula_closure(
        formula,
        build_strict_formula_closure_policy(ir),
        ir_version=ir.ir_version,
    )
    if not closure_report.passed:
        raise ValueError(
            "compiler_contract_violation: "
            + "; ".join(closure_report.reasons)
        )
    validate_no_repeated_negation(formula, context="IR compiler output")
    try:
        reparsed = parse_formula(str(formula))
    except ValueError as exc:
        raise ValueError(
            f"compiler_contract_violation: round-trip parse failed: {exc}"
        ) from exc
    if reparsed != formula:
        raise ValueError(
            "compiler_contract_violation: round-trip AST differs from compiler output"
        )
    comparison_horizon = min(
        32,
        max(
            2,
            structural_horizon(ir),
            *(discriminator_horizon(node) for node in ir.nodes),
        ),
    )
    report = compare_ir_and_formula(
        ir,
        formula,
        max_length=comparison_horizon,
        max_traces=4096,
        temporal_resolution_by_node=temporal_resolution_by_node,
    )
    if not report.equivalent:
        raise ValueError(
            "compiler_semantics_mismatch: "
            f"{report.reason}; trace={report.mismatch_trace}; "
            f"ir={report.ir_value}; formula={report.formula_value}"
        )
    artifact = CompiledFormulaArtifact(
        ir_version=ir.ir_version,
        formula=formula,
        lowering_certificate=tuple(
            certificate_by_node[node.node_id] for node in ir.nodes
        ),
    )
    artifact.validate_for(ir)
    if ir.ir_version:
        if len(_COMPILED_ARTIFACT_CACHE) >= 256:
            _COMPILED_ARTIFACT_CACHE.clear()
        _COMPILED_ARTIFACT_CACHE[cache_key] = artifact
    return artifact


def compile_ir_to_formula(
    ir: FaithfulSemanticIR,
    *,
    temporal_resolution_by_node: dict[str, str] | None = None,
) -> Formula:
    """Compatibility API returning the formula from the certified compiler."""

    return compile_ir_artifact(
        ir,
        temporal_resolution_by_node=temporal_resolution_by_node,
    ).formula


def render_ir_hole_template(ir: FaithfulSemanticIR) -> str:
    """Render a diagnostic template whose holes retain exact IR nesting."""

    def render(node_id: str) -> str:
        node = ir.node_map[node_id]
        children = tuple(render(child) for child in node.children)
        if node.node_type == "ap_condition":
            return str(node.parameter("ap_name"))
        symbols = {
            "negation": "!",
            "conjunction": " & ",
            "disjunction": " | ",
            "implication": " -> ",
            "equivalence": " <-> ",
        }
        if node.node_type == "negation":
            return f"!({children[0]})"
        if node.node_type in symbols:
            return "(" + symbols[node.node_type].join(children) + ")"
        if node.node_type in {"until", "release"}:
            return f"({children[0]} H-{node.node_id} {children[1]})"
        if node.node_type in TEMPORAL_NODE_TYPES:
            return f"H-{node.node_id}({', '.join(children)})"
        return f"{node.node_type}({', '.join(children)})"

    return render(ir.root_node_id)


def _require_version(expected: str, actual: str, label: str) -> None:
    if not expected or expected != actual:
        raise ValueError(f"{label} ir_version mismatch: expected {expected}, got {actual}")


def _minimum_horizon(node_type: str, parameters: tuple[tuple[str, object], ...]) -> int:
    values = dict(parameters)
    declared = max(1, int(values.get("minimum_horizon", 1)))
    if node_type == "next_step":
        return max(2, declared, int(values.get("steps", 1)) + 1)
    if node_type in {"eventually_strict", "before", "after", "precedence", "until", "release"}:
        return max(2, declared)
    if node_type == "ordered_sequence":
        return max(2, declared, int(values.get("event_count", 2)))
    if node_type == "fixed_duration":
        return max(2, declared, int(values.get("duration", 1)))
    if node_type == "exact_run_length":
        return max(
            2,
            declared,
            int(values.get("start_offset", 0))
            + int(values.get("duration", 1))
            + 1,
        )
    if node_type == "bounded_window":
        return max(2, declared, int(values.get("upper_bound", 1)) + 1)
    if node_type in {"cardinality", "recurrence", "periodicity"}:
        return max(
            2,
            declared,
            int(values.get("count", values.get("period", 1))) + 1,
        )
    return declared


def _typed_trace_constraints(node) -> tuple[str, ...]:
    """Expose IR parameters to trace generation as mandatory typed constraints."""

    if node.node_type == "periodicity":
        params = dict(node.parameters)
        names = list(PERIODICITY_REQUIRED_PARAMETERS)
        if params.get("mode") == "fixed_phase":
            names.extend(("phase", "off_schedule_policy"))
        elif params.get("mode"):
            names.append("occurrence_policy")
        return tuple(
            f"periodicity.{name}={params.get(name, 'UNRESOLVED')}"
            for name in names
        )
    constraints = tuple(f"parameter.{key}={value}" for key, value in node.parameters)
    operators = allowed_operators_for_node(node)
    if len(operators) > 1:
        constraints = (
            *constraints,
            f"temporal_resolution={node.temporal_resolution}",
            "allowed_semantic_readings="
            + ",".join(allowed_semantic_readings(node)),
        )
    return constraints


def _semantic_parameter_paths(
    node,
    *,
    names: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Bind review questions to typed IR parameters instead of free-text axes."""

    parameter_names = names or tuple(name for name, _value in node.parameters)
    return tuple(
        f"/faithful_ir/nodes/{node.node_id}/parameters/{name}"
        for name in parameter_names
        if name
    )


def _boundary_parameter_names(node_type: str) -> tuple[str, ...]:
    return {
        "eventually": ("terminal_policy",),
        "next_step": ("steps",),
        "until": ("witness_timing", "witness_policy", "left_at_witness"),
        "release": ("terminal_policy",),
        "global_constraint": ("terminal_policy",),
        "persist_to_end": ("scope_origin",),
        "before": ("ordering", "same_step_policy"),
        "precedence": ("ordering", "same_step_policy"),
        "cardinality": ("count", "mode"),
        "exact_run_length": (
            "start_offset", "duration", "terminal_obligation_policy",
            "post_window_policy",
        ),
        "periodicity": ("terminal_policy",),
    }.get(node_type, ())


def _boundary_case(node_type: str) -> str:
    return {
        "eventually": "witness at final state versus missing witness",
        "eventually_strict": "same-step versus any strict-future witness",
        "next_step": "trigger at final state and shortest trace",
        "until": "immediate witness, final witness, and missing witness",
        "release": "finite-trace end release behavior",
        "global_constraint": "violation at final state",
        "persist_to_end": "suffix starts at final state",
        "before": "same-step versus strict ordering",
        "precedence": "first occurrence and same-step ordering",
        "cardinality": "one below and one above count threshold",
        "bounded_window": "witness at lower bound, upper bound, and just after upper bound",
        "exact_run_length": "full run, short trace, and the first post-window position",
        "periodicity": "short trace before first full period",
    }.get(node_type, "")


def _possible_ir_node_values(
    ir: FaithfulSemanticIR,
    trace,
    node_id: str,
) -> frozenset[bool]:
    """Execute a node under every approved temporal assignment (bounded)."""

    ranged_nodes = tuple(
        (node.node_id, allowed_operators_for_node(node))
        for node in ir.nodes
        if len(allowed_operators_for_node(node)) > 1
    )
    domains = tuple(domain for _, domain in ranged_nodes)
    combinations = itertools.product(*domains) if domains else ((),)
    values: set[bool] = set()
    for operators in combinations:
        resolution = {
            ranged_node_id: operator
            for (ranged_node_id, _), operator in zip(ranged_nodes, operators)
        }
        values.add(
            evaluate_ir_trace(
                ir,
                trace,
                node_id=node_id,
                temporal_resolution_by_node=resolution,
            )
        )
    return frozenset(values)


def _allowed_operators(node_type: str) -> tuple[str, ...]:
    return {
        "global_constraint": ("G",),
        "eventually": ("F",),
        "eventually_strict": ("XF",),
        "next_step": ("X",),
        "until": ("U",),
        "release": ("R",),
        "persist_to_end": ("G",),
        "before": ("BEFORE",),
        "after": ("AFTER",),
        "precedence": ("PRECEDENCE",),
        "ordered_sequence": ("ORDERED",),
        "fixed_duration": ("FIXED_DURATION",),
        "exact_run_length": ("EXACT_RUN_LENGTH",),
        "bounded_window": ("BOUNDED_WINDOW",),
        "cardinality": ("CARDINALITY",),
        "recurrence": ("GF",),
        "periodicity": ("PERIODIC",),
        "start_boundary": ("START",),
        "end_boundary": ("END",),
        "short_trace_boundary": ("SHORT_TRACE",),
    }.get(node_type, ())


def _trace_matches_expected_relation(trace: LabeledTrace, relation: str) -> bool:
    if relation == "satisfy":
        return trace.expected
    if relation == "violate":
        return not trace.expected
    if relation == "distinguish":
        return trace.kind == "boundary" or bool(trace.boundary_role)
    return False


def _trace_obligation_ids(trace: LabeledTrace) -> tuple[str, ...]:
    return trace.obligation_ids


def _trace_roles(trace: LabeledTrace) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            role
            for role in (trace.trace_role, trace.role, trace.query_role)
            if role
        )
    )


def _fold_binary(constructor, children: tuple[Formula, ...]) -> Formula:
    if not children:
        raise ValueError("Boolean IR node must have children")
    result = children[0]
    for child in children[1:]:
        result = constructor(result, child)
    return result


def _compile_until_relation(node, left: Formula, right: Formula) -> Formula:
    parameters = dict(node.parameters)
    witness_policy = str(parameters.get("witness_policy", ""))
    witness_timing = str(parameters.get("witness_timing", ""))
    left_at_witness = str(parameters.get("left_at_witness", ""))
    if not witness_policy or not witness_timing or not left_at_witness:
        raise ValueError(
            f"until node {node.node_id} has unresolved semantic policy; "
            "approve witness_policy, witness_timing, and left_at_witness first"
        )
    witness = And(left, right) if left_at_witness == "required" else right
    strong = U(left, witness)
    if witness_timing == "strict_future":
        strong = And(left, X(strong))
    if witness_policy == "required":
        return strong
    if witness_policy == "optional_if_missing":
        return Or(G(left), strong)
    raise ValueError(
        f"until node {node.node_id} has unsupported witness_policy={witness_policy!r}"
    )


def _compile_precedence_relation(
    node,
    prerequisite: Formula,
    event: Formula,
) -> Formula:
    parameters = dict(node.parameters)
    ordering = str(parameters.get("ordering", ""))
    witness_policy = str(parameters.get("witness_policy", ""))
    reuse_policy = str(parameters.get("reuse_policy", ""))
    scope_origin = str(parameters.get("scope_origin", ""))
    if not ordering or not witness_policy or not reuse_policy or not scope_origin:
        raise ValueError(
            f"precedence node {node.node_id} has unresolved semantic policy"
        )
    if reuse_policy != "reusable":
        raise ValueError(
            "one_to_one precedence is a prefix-counting constraint and is not "
            "representable by the current reusable-gate LTLf template"
        )
    if scope_origin != "trace_start":
        raise ValueError(
            "local_activation precedence requires an explicit activation child"
        )
    witness = (
        And(prerequisite, Not(event))
        if ordering == "strict"
        else prerequisite
    )
    strong = U(Not(event), witness)
    if witness_policy == "required":
        return strong
    if witness_policy == "optional_if_no_constrained":
        return Or(G(Not(event)), strong)
    raise ValueError(
        f"precedence node {node.node_id} has unsupported witness_policy={witness_policy!r}"
    )


def _compile_periodicity_relation(
    node,
    child: Formula,
    ap_names: tuple[str, ...],
) -> Formula:
    """Compile explicit finite-prefix periodicity without synthetic constants."""

    params = dict(node.parameters)
    missing = [
        name
        for name in PERIODICITY_REQUIRED_PARAMETERS
        if params.get(name) in {None, ""}
    ]
    if missing:
        raise ValueError(
            f"periodicity node {node.node_id} has unresolved semantic policy: "
            + ", ".join(missing)
        )
    period = int(params["period"])
    mode = str(params["mode"])
    terminal = str(params["terminal_policy"])
    if period < 1:
        raise ValueError("periodicity requires period >= 1")
    if terminal not in {"finite_prefix", "require_first_scheduled_witness"}:
        raise ValueError(f"unsupported periodicity terminal_policy={terminal!r}")
    if not ap_names:
        raise ValueError("periodicity compilation needs one declared AP for successor guards")

    witness_ap = Atom(ap_names[0])
    tautology = Or(witness_ap, Not(witness_ap))

    def guarded_at(offset: int, formula: Formula) -> Formula:
        if offset == 0:
            return formula
        exists = _next_n(tautology, offset)
        return Or(Not(exists), _next_n(formula, offset))

    gaps = tuple(guarded_at(offset, Not(child)) for offset in range(1, period))
    gap_formula = _fold_binary(And, gaps) if gaps else tautology
    occurrence = str(params.get("occurrence_policy", ""))

    if mode == "minimum_spacing":
        result = G(Implies(child, gap_formula))
        return And(F(child), result) if occurrence == "required" else result

    if mode in {"event_relative_exact_recurrence", "exact_inter_occurrence_spacing"}:
        exists_at_period = _next_n(tautology, period)
        if mode == "event_relative_exact_recurrence":
            continuation = guarded_at(period, child)
        else:
            continuation = Or(
                Not(exists_at_period),
                Or(_next_n(child, period), _next_n(G(Not(child)), period)),
            )
        body = And(gap_formula, continuation)
        result = G(Implies(child, body))
        return And(F(child), result) if occurrence == "required" else result

    if mode != "fixed_phase":
        raise ValueError(f"unsupported periodicity mode={mode!r}")
    phase = int(params["phase"])
    if phase < 0 or phase >= period:
        raise ValueError("fixed_phase requires 0 <= phase < period")
    if str(params.get("off_schedule_policy", "")) != "forbidden":
        raise ValueError(
            "fixed periodicity currently requires off_schedule_policy=forbidden"
        )

    first_phase = (
        _next_n(child, phase)
        if terminal == "require_first_scheduled_witness"
        else guarded_at(phase, child)
    )
    prefix: list[Formula] = [first_phase]
    prefix.extend(guarded_at(offset, Not(child)) for offset in range(phase))
    recurrence_parts = [
        guarded_at(offset, Not(child)) for offset in range(1, period)
    ]
    recurrence_parts.append(guarded_at(period, child))
    recurrence = _fold_binary(And, tuple(recurrence_parts))
    prefix.append(G(Implies(child, recurrence)))
    return _fold_binary(And, tuple(prefix))


def _required_node_parameter(node, name: str):
    params = dict(node.parameters)
    if name not in params or params[name] in {None, ""}:
        raise ValueError(
            f"compiler_contract_violation: {node.node_type} node "
            f"{node.node_id} is missing {name}"
        )
    return params[name]


def _validate_recurrence_policy(node) -> None:
    expected = {
        "origin": "current_position",
        "gap_policy": "unbounded_eventual",
        "short_trace_policy": "finite_suffix",
        "terminal_obligation_policy": "final_state_witness",
    }
    for name, expected_value in expected.items():
        actual = str(_required_node_parameter(node, name))
        if actual != expected_value:
            raise ValueError(
                f"compiler_contract_violation: recurrence {name}={actual!r} "
                f"is unsupported; expected {expected_value!r}"
            )
    if int(_required_node_parameter(node, "minimum_horizon")) < 1:
        raise ValueError("compiler_contract_violation: recurrence minimum_horizon < 1")


def _shift_formula(child: Formula, offset: int) -> Formula:
    shifted = child
    for _ in range(offset):
        shifted = X(shifted)
    return shifted


def _position_exists(child: Formula, offset: int) -> Formula:
    return _shift_formula(Or(child, Not(child)), offset)


def _fixed_duration(
    child: Formula,
    duration: int,
    *,
    short_policy: str,
    terminal_policy: str,
) -> Formula:
    if duration < 1:
        raise ValueError("fixed_duration requires duration >= 1")
    if (short_policy, terminal_policy) == (
        "strong_next",
        "require_full_duration",
    ):
        return _fold_binary(
            And,
            tuple(_shift_formula(child, offset) for offset in range(duration)),
        )
    if (short_policy, terminal_policy) == (
        "available_prefix",
        "truncate_at_end",
    ):
        return _fold_binary(
            And,
            tuple(
                Or(
                    Not(_position_exists(child, offset)),
                    _shift_formula(child, offset),
                )
                for offset in range(duration)
            ),
        )
    raise ValueError(
        "unsupported fixed_duration short/terminal policy: "
        f"{short_policy}/{terminal_policy}"
    )


def _exact_run_length(
    child: Formula,
    *,
    start_offset: int,
    duration: int,
    origin: str,
    start_policy: str,
    terminal_policy: str,
    post_policy: str,
) -> Formula:
    if start_offset < 0 or duration < 1:
        raise ValueError("exact_run_length requires offset >= 0 and duration >= 1")
    if (
        origin,
        start_policy,
        terminal_policy,
        post_policy,
    ) != (
        "current_position",
        "required",
        "require_full_duration",
        "false_if_position_exists",
    ):
        raise ValueError("unsupported exact_run_length policy")
    required = tuple(
        _shift_formula(child, offset)
        for offset in range(start_offset, start_offset + duration)
    )
    post_offset = start_offset + duration
    post = Or(
        Not(_position_exists(child, post_offset)),
        _shift_formula(Not(child), post_offset),
    )
    return _fold_binary(And, (*required, post))


def _bounded_eventually(
    child: Formula,
    lower: int,
    upper: int,
    *,
    short_policy: str,
    terminal_policy: str,
) -> Formula:
    if lower < 0 or upper < lower:
        raise ValueError("bounded_window requires 0 <= lower_bound <= upper_bound")
    witness = _fold_binary(
        Or,
        tuple(_shift_formula(child, offset) for offset in range(lower, upper + 1)),
    )
    if short_policy == "strong_upper_bound":
        if terminal_policy != "require_witness":
            raise ValueError(
                "strong_upper_bound requires terminal_obligation_policy=require_witness"
            )
        return And(_position_exists(child, upper), witness)
    if short_policy == "available_prefix":
        if terminal_policy == "require_witness":
            return witness
        if terminal_policy == "vacuous_if_window_unavailable":
            return Or(Not(_position_exists(child, lower)), witness)
    raise ValueError(
        "unsupported bounded_window short/terminal policy: "
        f"{short_policy}/{terminal_policy}"
    )


def _next_n(child: Formula, count: int) -> Formula:
    result = child
    for _ in range(count):
        result = X(result)
    return result


def _ordered_sequence(children: tuple[Formula, ...]) -> Formula:
    if not children:
        raise ValueError("ordered_sequence requires at least one event")
    result = F(children[-1])
    for child in reversed(children[:-1]):
        result = F(And(child, X(result)))
    return result


def _at_least(child: Formula, count: int) -> Formula:
    if count <= 0:
        # LTLf traces are non-empty.  This local tautology preserves the child
        # AP vocabulary without introducing a Boolean-constant escape hatch.
        return Or(child, Not(child))
    if count == 1:
        return F(child)
    return F(And(child, X(_at_least(child, count - 1))))


def _cardinality(child: Formula, mode: str, count: int) -> Formula:
    normalized = mode.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"at_least", "minimum", ">="}:
        return _at_least(child, count)
    if normalized in {"at_most", "maximum", "<="}:
        return Not(_at_least(child, count + 1))
    if normalized in {"exactly", "equal", "="}:
        return And(_at_least(child, count), Not(_at_least(child, count + 1)))
    raise ValueError(f"unsupported cardinality mode: {mode}")
