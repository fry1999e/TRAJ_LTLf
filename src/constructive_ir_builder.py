"""Construct complete IR candidates from source-bound relation graphs."""
from dataclasses import asdict, dataclass, replace
from itertools import product
import hashlib
import json

from faithful_ir import faithful_ir_from_mapping, assert_ir_ready_for_trace_generation
from semantic_node_contracts import NODE_CONTRACTS, node_contract_fingerprint
from typed_source_relation_graph import (
    SourceEdge,
    SourceAxisOption,
    SourceIntentAxis,
    SourceNode,
    SourceSemanticDebt,
    UnsupportedConstruct,
    SourceIntentGraphSet,
    build_source_coverage_ledger,
    parse_source_graph_set,
)
import re


MAX_CONSTRUCTIVE_CANDIDATES = 4


_UNTIL_CONVENTION_ALIASES = {
    # The frozen A72/C1026 source convention says that U is strong and its
    # right operand must occur.  These remaining fields are the standard
    # current-position LTLf U denotation, not runtime defaults.
    "strong_right_operand_required": {
        "witness_policy": "required",
        "witness_timing": "current_or_future",
        "left_at_witness": "not_required",
        "scope_origin": "current_position",
    },
}


@dataclass(frozen=True)
class BuiltIRCandidate:
    ir: object
    selections: tuple[tuple[str, str], ...] = ()
    source_graph_hash: str = ""


@dataclass(frozen=True)
class BuildResult:
    route: str
    ir: object = None
    graph: object = None
    reason_code: str = ""
    candidates: tuple[BuiltIRCandidate, ...] = ()
    axes: tuple[SourceIntentAxis, ...] = ()
    semantic_debts: tuple[SourceSemanticDebt, ...] = ()
    candidate_complete: bool = True


def build_ir(
    graph,
    ap_context,
    *,
    policies=None,
    surface_segmentation=None,
    plain_reading_review=None,
    policy_overrides=None,
    extractor="RCG-constructive-v1",
):
    """Policies must be a validated frozen DatasetSemanticsContract."""
    if graph.ap_binding_hash != ap_context.context_fingerprint:
        raise UnsupportedConstruct("AP_CONTEXT_STALE")
    fields = {}
    if policies is not None:
        from contracts.dataset_semantics_contract import validate_dataset_semantics_contract
        validate_dataset_semantics_contract(policies)
        if graph.contract_hash != policies["contract_hash"]:
            raise UnsupportedConstruct("CONTRACT_STALE")
        fields = policies["fields"]

    policy_overrides = dict(policy_overrides or {})

    def policy(key):
        if key in policy_overrides:
            return policy_overrides[key]
        item = fields.get(key, {})
        if item.get("unknown", True):
            raise UnsupportedConstruct("unresolved_axis:" + key)
        return item["value"]

    edges = {edge.from_ids[0]: edge for edge in graph.edges}
    aps, nodes = {}, []
    for source in graph.nodes:
        kind = source.kind
        params = dict(source.parameters)
        semantic_role = ""
        allowed_semantic_readings = ()
        if kind == "AP":
            kind = "ap_condition"
            params = {"ap_name": source.canonical_name}
            aps.setdefault(source.ap_identity_key, {
                "ap_id": source.ap_identity_key, "name": source.canonical_name,
                "source_span": source.source_span, "evidence_text": source.exact_text,
            })
        elif kind in {"true", "false"}:
            params = {"value": kind == "true"}; kind = "boolean_constant"
        elif kind == "next_step":
            # Finite endpoint interpretation must be explicitly contracted.
            if policy("finite_trace_semantics") != "nonempty_finite_strong_next":
                raise UnsupportedConstruct("unresolved_axis:finite_trace_semantics")
            params = {"steps": 1}
        elif kind == "eventually":
            value = policy("eventuality_policy")
            if value == "strict_future": kind = "eventually_strict"
            elif value not in {"current_or_future", "current_or_later"}: raise UnsupportedConstruct("unresolved_axis:eventuality_policy")
            if "eventuality_policy" in policy_overrides:
                semantic_role = "policy_axis_selection:eventuality_policy"
                allowed_semantic_readings = (kind,)
        elif kind == "until":
            value = policy("interval_boundary_policy")
            until = value.get("until") if isinstance(value, dict) else None
            if isinstance(until, str):
                until = _UNTIL_CONVENTION_ALIASES.get(until)
            required = NODE_CONTRACTS["until"].parameters
            if not isinstance(until, dict) or set(until) != set(required):
                raise UnsupportedConstruct("unresolved_axis:until_boundary")
            if any(not parameter.accepts(until[name]) for name, parameter in required.items()):
                raise UnsupportedConstruct("unresolved_axis:until_boundary")
            params = dict(until)
        contract = NODE_CONTRACTS[kind]
        children = edges[source.node_id].to_ids if source.node_id in edges else ()
        if len(children) < contract.min_children or (contract.max_children is not None and len(children) > contract.max_children):
            raise UnsupportedConstruct("IR_CHILD_ARITY")
        nodes.append({"node_id": source.node_id, "node_type": kind,
                      "children": children, "parameters": params,
                      "semantic_role": semantic_role,
                      "allowed_semantic_readings": allowed_semantic_readings,
                      "scope": "all_positions" if kind == "global_constraint" else "current_position",
                      "source_span": source.source_span, "evidence_text": source.exact_text})
    unresolved_review = bool(
        plain_reading_review is not None
        and plain_reading_review.overall_status == "ambiguous_unresolved"
    )
    ir = faithful_ir_from_mapping({"atomic_propositions": list(aps.values()), "nodes": nodes,
        "root_node_id": graph.root_node_id, "node_contract_fingerprint": node_contract_fingerprint()},
        source_text=graph.source_text, requirement_id=ap_context.requirement_id,
        extractor=extractor, ap_binding_context=ap_context,
        surface_segmentation=surface_segmentation,
        plain_reading_review=None if unresolved_review else plain_reading_review)
    if unresolved_review:
        from faithful_ir import compute_ir_version, validate_faithful_ir
        ir = replace(
            ir,
            reading_review_id=plain_reading_review.reading_review_id,
            reading_review_basis=plain_reading_review.review_basis,
            ir_version="",
        )
        ir = replace(ir, ir_version=compute_ir_version(ir))
        validate_faithful_ir(ir, surface_segmentation=surface_segmentation)
    assert_ir_ready_for_trace_generation(ir)
    return ir


def _required_policy_axes(graph, policies):
    fields = (policies or {}).get("fields", {})
    kinds = {node.kind for node in graph.nodes}
    result = []
    if "eventually" in kinds and fields.get("eventuality_policy", {}).get("unknown", True):
        result.append(("eventuality_policy", (
            ("current_or_future", "eventually may hold now or later"),
            ("strict_future", "eventually must hold strictly later"),
        )))
    if "next_step" in kinds and fields.get("finite_trace_semantics", {}).get("unknown", True):
        # The current executable schema has no weak-next node.  Returning debt
        # is safer than pretending the only implemented option is exhaustive.
        return (), (SourceSemanticDebt(
            "sdebt-" + hashlib.sha256((graph.source_hash + "finite_trace_semantics").encode()).hexdigest()[:16],
            "unsupported_semantics:finite_trace_semantics",
            (0, len(graph.source_text)), graph.source_text,
            "The schema implements strong next but cannot express the full unknown strong/weak-next axis.",
        ),)
    if "until" in kinds and fields.get("interval_boundary_policy", {}).get("unknown", True):
        return (), (SourceSemanticDebt(
            "sdebt-" + hashlib.sha256((graph.source_hash + "until_boundary").encode()).hexdigest()[:16],
            "ambiguity_debt:until_boundary",
            (0, len(graph.source_text)), graph.source_text,
            "The complete until boundary Cartesian product exceeds the first-batch frozen bound.",
        ),)
    return tuple(result), ()


def _policy_axis(graph, name, choices):
    span = (0, len(graph.source_text))
    target = next(
        (node.node_id for node in graph.nodes if node.kind == "eventually"),
        graph.root_node_id,
    )
    axis_id = "sax-" + hashlib.sha256(
        json.dumps((graph.source_hash, name), separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return SourceIntentAxis(
        axis_id,
        "sclaim-" + hashlib.sha256((graph.source_hash + name).encode()).hexdigest()[:16],
        name,
        span,
        graph.source_text,
        tuple(
            SourceAxisOption(value, label, "choose_policy_branch", target, value)
            for value, label in choices
        ),
        (name,),
    )


def _wrap_global(graph):
    """Wrap a complete graph in an explicit executable global obligation."""

    span = (0, len(graph.source_text))
    node_id = "sn-" + hashlib.sha256(
        json.dumps((graph.source_hash, "deontic-global", graph.root_node_id)).encode()
    ).hexdigest()[:16]
    node = SourceNode(
        node_id, "global_constraint", span, graph.source_text,
    )
    edge = SourceEdge(
        "se-" + hashlib.sha256((node_id + graph.root_node_id).encode()).hexdigest()[:16],
        "global_constraint", (node_id,), (graph.root_node_id,), span,
        authority="dataset_contract", parser_provenance="deontic-scope-policy-v1",
    )
    return replace(
        graph,
        nodes=(*graph.nodes, node),
        edges=(*graph.edges, edge),
        root_node_id=node_id,
        coverage_ledger=build_source_coverage_ledger(
            graph.source_text, (*graph.nodes, node),
            axes=graph.axes, debt_hints=graph.semantic_debts,
            ap_evidence_spans=tuple(entry.source_span for entry in graph.coverage_ledger.entries
                                    if entry.category == "AP_EVIDENCE") if graph.coverage_ledger is not None else (),
        ),
    )


def _deontic_scope_graph_set(graph_set, policies):
    """Apply an authoritative policy or materialize the exhaustive scope axis."""

    if not re.search(r"\b(?:shall|must)\b", graph_set.primary.source_text, re.I):
        return graph_set
    if all(any(node.kind == "global_constraint" for node in graph.nodes) for graph in graph_set.candidates):
        return graph_set
    fields = (policies or {}).get("fields", {})
    item = fields.get("deontic_scope_policy", {})
    policy = None if item.get("unknown", True) else item.get("value")
    # A v1 contract may still authoritatively declare that requirements are
    # evaluated only at position zero.  Preserve that source-bound meaning.
    if policy is None:
        legacy = fields.get("default_scope", {})
        if not legacy.get("unknown", True) and legacy.get("value") == "evaluate_at_position_0":
            policy = "initial_obligation"
    if policy in {"initial_obligation", "source_explicit_only"}:
        return graph_set
    if policy == "invariant_obligation":
        wrapped = tuple(_wrap_global(graph) for graph in graph_set.candidates)
        return replace(graph_set, primary=wrapped[0], candidates=wrapped)
    if policy not in {None, "unknown"}:
        raise UnsupportedConstruct("unresolved_axis:deontic_scope_policy")

    span = (0, len(graph_set.primary.source_text))
    axis_id = "sax-" + hashlib.sha256(
        (graph_set.primary.source_hash + "deontic_scope_policy").encode()
    ).hexdigest()[:16]
    axis = SourceIntentAxis(
        axis_id,
        "sclaim-" + hashlib.sha256((graph_set.primary.source_hash + "shall").encode()).hexdigest()[:16],
        "deontic_scope_policy",
        span,
        graph_set.primary.source_text,
        (
            SourceAxisOption("initial_obligation", "obligation at the current position", "choose_policy_branch", graph_set.primary.root_node_id, "initial_obligation"),
            SourceAxisOption("invariant_obligation", "obligation at every position", "wrap_global_constraint", graph_set.primary.root_node_id, "invariant_obligation"),
        ),
        ("deontic_scope_policy",),
    )
    prior = graph_set.candidate_selections or tuple(() for _ in graph_set.candidates)
    candidates = []
    selections = []
    for graph, selected in zip(graph_set.candidates, prior):
        candidates.extend((graph, _wrap_global(graph)))
        selections.extend((
            (*selected, (axis_id, "initial_obligation")),
            (*selected, (axis_id, "invariant_obligation")),
        ))
    return SourceIntentGraphSet(
        candidates[0], tuple(candidates), (*graph_set.axes, axis),
        graph_set.semantic_debts, graph_set.complete, tuple(selections),
    )
def _construct_graph_candidates(graph_set, ap_context, *, policies=None, max_candidates=MAX_CONSTRUCTIVE_CANDIDATES, **kwargs):
    graph_set = _deontic_scope_graph_set(graph_set, policies)
    policy_axes, policy_debts = _required_policy_axes(graph_set.primary, policies)
    if policy_debts:
        return (), tuple((*graph_set.semantic_debts, *policy_debts)), False, graph_set.axes
    axes = tuple(graph_set.axes) + tuple(
        _policy_axis(graph_set.primary, name, choices)
        for name, choices in policy_axes
    )
    dimensions = [
        tuple((axis.axis_id, option.option_id, option.value) for option in axis.options)
        for axis in axes
    ]
    # Graph topology variants are already one dimension.  Policy dimensions
    # are applied symmetrically to each graph variant.
    policy_dimensions = dimensions[len(graph_set.axes):]
    total = len(graph_set.candidates)
    for dimension in policy_dimensions:
        total *= len(dimension)
    if total > max_candidates:
        debt = SourceSemanticDebt(
            "sdebt-" + hashlib.sha256((graph_set.primary.source_hash + str(total)).encode()).hexdigest()[:16],
            "ambiguity_debt:candidate_bound_exceeded",
            (0, len(graph_set.primary.source_text)), graph_set.primary.source_text,
            f"Complete Cartesian product has {total} candidates; frozen bound is {max_candidates}.",
        )
        return (), tuple((*graph_set.semantic_debts, debt)), False, axes
    candidates = []
    source_selections = list(
        graph_set.candidate_selections
        or (() for _graph in graph_set.candidates)
    )
    if graph_set.axes and not graph_set.candidate_selections:
        if len(graph_set.axes) == 1 and len(graph_set.candidates) == len(graph_set.axes[0].options):
            source_selections = [
                ((graph_set.axes[0].axis_id, option.option_id),)
                for option in graph_set.axes[0].options
            ]
        else:
            debt = SourceSemanticDebt(
                "sdebt-" + hashlib.sha256((graph_set.primary.source_hash + "graph-axis-shape").encode()).hexdigest()[:16],
                "ambiguity_debt:graph_axis_shape",
                (0, len(graph_set.primary.source_text)), graph_set.primary.source_text,
                "Graph variants lack complete per-axis selection provenance.",
            )
            return (), tuple((*graph_set.semantic_debts, debt)), False, axes
    combinations = list(product(*policy_dimensions)) if policy_dimensions else [()]
    for graph, graph_selection in zip(graph_set.candidates, source_selections):
        for combination in combinations:
            overrides = {
                axis.axis_kind: value
                for axis, (_axis_id, _option_id, value) in zip(axes[len(graph_set.axes):], combination)
            }
            selections = tuple((*graph_selection, *((axis_id, option_id) for axis_id, option_id, _value in combination)))
            try:
                ir = build_ir(
                    graph,
                    ap_context,
                    policies=policies,
                    policy_overrides=overrides,
                    extractor=(
                        "RCG-source-candidate-v1"
                        if axes else "RCG-constructive-v1"
                    ),
                    **kwargs,
                )
            except UnsupportedConstruct as exc:
                debt = SourceSemanticDebt(
                    "sdebt-" + hashlib.sha256((graph.source_hash + exc.code).encode()).hexdigest()[:16],
                    "unsupported_semantics:" + exc.code,
                    exc.span if exc.span != (-1, -1) else (0, len(graph.source_text)),
                    graph.source_text[slice(*(exc.span if exc.span != (-1, -1) else (0, len(graph.source_text))))],
                    "Candidate could not be expressed by the executable IR schema.",
                )
                return (), tuple((*graph_set.semantic_debts, debt)), False, axes
            candidates.append(BuiltIRCandidate(ir, selections, graph.graph_hash))
    return tuple(candidates), tuple(graph_set.semantic_debts), graph_set.complete, axes


def try_construct_ir(text, ap_context, *, policies=None, max_candidates=MAX_CONSTRUCTIVE_CANDIDATES, **kwargs):
    graph = None
    try:
        graph_set = parse_source_graph_set(
            text, ap_context, contract_hash=(policies or {}).get("contract_hash", "")
        )
        graph = graph_set.primary
        candidates, debts, complete, axes = _construct_graph_candidates(
            graph_set, ap_context, policies=policies, max_candidates=max_candidates, **kwargs
        )
        if debts:
            return BuildResult(
                "U" if debts[0].reason_code.startswith(("unresolved_axis", "ambiguity_debt")) else "unsupported",
                graph=graph,
                reason_code=debts[0].reason_code,
                axes=axes, semantic_debts=debts, candidate_complete=complete,
            )
        if len(candidates) == 1 and not axes:
            return BuildResult("D", candidates[0].ir, graph, candidates=candidates)
        return BuildResult(
            "U", graph=graph,
            reason_code="SOURCE_SEMANTIC_CANDIDATES_REQUIRED",
            candidates=candidates, axes=axes, candidate_complete=complete,
        )
    except UnsupportedConstruct as exc:
        return BuildResult("U" if exc.code.startswith("unresolved_axis") else "G", graph=graph, reason_code=exc.code)


def try_construct_proposal_ir(text, ap_context, *, policies=None, **kwargs):
    """Construct an unapproved candidate from source-bound non-catalog identities."""

    graph = None
    if (
        ap_context.has_authoritative_catalog
        or ap_context.grounding_authority not in {"proposal_only", "required_subset"}
    ):
        return BuildResult("G", reason_code="AP_PROPOSAL_PATH_NOT_APPLICABLE")
    try:
        graph_set = parse_source_graph_set(
            text,
            ap_context,
            contract_hash=(policies or {}).get("contract_hash", ""),
            allow_proposal_identities=True,
        )
        graph = graph_set.primary
        candidates, debts, complete, axes = _construct_graph_candidates(
            graph_set, ap_context, policies=policies, **kwargs
        )
        if debts:
            return BuildResult(
                "U" if debts[0].reason_code.startswith(("unresolved_axis", "ambiguity_debt")) else "unsupported",
                graph=graph, reason_code=debts[0].reason_code,
                axes=axes, semantic_debts=debts, candidate_complete=complete,
            )
        if len(candidates) == 1 and not axes:
            reason = (
                "AP_REQUIRED_SUBSET_REVIEW_REQUIRED"
                if ap_context.grounding_authority == "required_subset"
                else "AP_PROPOSAL_REVIEW_REQUIRED"
            )
            return BuildResult(
                "U", candidates[0].ir, graph, reason,
                candidates=candidates,
            )
        return BuildResult(
            "U", graph=graph, reason_code="SOURCE_SEMANTIC_CANDIDATES_REQUIRED",
            candidates=candidates, axes=axes, candidate_complete=complete,
        )
    except UnsupportedConstruct as exc:
        return BuildResult("U" if exc.code.startswith("unresolved_axis") else "G", graph=graph, reason_code=exc.code)


def try_construct_from_proposed_graph(
    graph,
    ap_context,
    *,
    policies=None,
    max_candidates=MAX_CONSTRUCTIVE_CANDIDATES,
    **kwargs,
):
    """Validate/build a constrained G-route graph without requesting full IR."""

    candidates, debts, complete, axes = _construct_graph_candidates(
        SourceIntentGraphSet(
            graph,
            (graph,),
            graph.axes,
            graph.semantic_debts,
            not bool(graph.semantic_debts),
        ),
        ap_context,
        policies=policies,
        max_candidates=max_candidates,
        **kwargs,
    )
    if debts:
        return BuildResult(
            "U" if debts[0].reason_code.startswith(("unresolved_axis", "ambiguity_debt")) else "unsupported",
            graph=graph, reason_code=debts[0].reason_code,
            axes=axes, semantic_debts=debts, candidate_complete=complete,
        )
    if len(candidates) == 1 and not axes:
        return BuildResult(
            "G", candidates[0].ir, graph, "SOURCE_GRAPH_PROPOSAL_CONSTRUCTED",
            candidates=candidates,
        )
    return BuildResult(
        "G", graph=graph, reason_code="SOURCE_GRAPH_PROPOSAL_CANDIDATES_REQUIRED",
        candidates=candidates, axes=axes, candidate_complete=complete,
    )


def build_result_candidate_semantic(result, *, rag_context=(), plain_axis_bindings=None):
    """Convert a complete constructive candidate result into SemanticStructure."""

    if not result.candidates or not result.candidate_complete:
        raise UnsupportedConstruct(result.reason_code or "CANDIDATE_SET_INCOMPLETE")
    from faithful_ir import semantic_summary_from_ir
    from schemas import AmbiguityAxis, AmbiguityAxisSet, CandidateIRBranch, CandidateIRSet
    from semantic_progress import semantic_fingerprint

    plain_axis_bindings = plain_axis_bindings or {}
    branches = []
    for index, candidate in enumerate(result.candidates, start=1):
        assumptions = [
            f"source_axis_option:{axis_id}={option_id}"
            for axis_id, option_id in candidate.selections
        ]
        for ambiguity_id, option_by_source_option in plain_axis_bindings.items():
            for source_axis_id, source_option_id in candidate.selections:
                option_id = option_by_source_option.get((source_axis_id, source_option_id))
                if option_id:
                    assumptions.append(f"plain_reading_option:{ambiguity_id}={option_id}")
        branch_hash = hashlib.sha256(
            json.dumps((candidate.ir.ir_version, candidate.selections), sort_keys=True).encode()
        ).hexdigest()[:16]
        branches.append(CandidateIRBranch(
            ir_branch_id="source-branch-" + branch_hash,
            faithful_ir=candidate.ir,
            supported_text_spans=tuple(
                dict.fromkeys(axis.evidence_text for axis in result.axes)
            ),
            ambiguity_axis_ids=tuple(axis.axis_id for axis in result.axes),
            semantic_assumptions=tuple((*assumptions, "complete source-supported candidate; not yet approved")),
            plain_language_paraphrase="; ".join(
                option.label
                for axis in result.axes
                for option in axis.options
                if (axis.axis_id, option.option_id) in candidate.selections
            ),
            generation_reason="Materialized from a source-bound relation graph and complete typed axes.",
            applied_alternative_ids=tuple(option for _axis, option in candidate.selections),
            status="needs_review",
            structure_fingerprint=semantic_fingerprint(candidate.ir),
        ))
    axis_set = AmbiguityAxisSet(
        requirement_id=result.candidates[0].ir.requirement_id,
        axes=tuple(AmbiguityAxis(
            axis_id=axis.axis_id,
            nl_span=axis.evidence_text,
            possible_readings=tuple(option.label for option in axis.options),
            why_ambiguous="Source or authoritative contract supports multiple executable readings.",
            required_trace_roles=("boundary",),
            review_status="candidate_ready",
            review_basis="source_intent_graph",
        ) for axis in result.axes),
        source="source_intent_graph",
    )
    set_payload = {
        "source": result.graph.graph_hash if result.graph is not None else "",
        "branches": [(branch.ir_branch_id, branch.faithful_ir.ir_version) for branch in branches],
    }
    branch_set_id = "SIGSET-" + hashlib.sha256(
        json.dumps(set_payload, sort_keys=True).encode()
    ).hexdigest()[:20]
    candidate_set = CandidateIRSet(
        requirement_id=result.candidates[0].ir.requirement_id,
        nl_text=result.candidates[0].ir.source_text,
        branches=tuple(branches),
        ambiguity_axes=axis_set,
        max_branch_count=MAX_CONSTRUCTIVE_CANDIDATES,
        status="policy_candidates_ready",
        segmentation_id=result.candidates[0].ir.segmentation_id,
        reading_review_id=result.candidates[0].ir.reading_review_id,
        ambiguity_debt=tuple(f"unresolved_ir_choice:{axis.axis_id}" for axis in result.axes),
        branch_set_id=branch_set_id,
        policy_axis_ids=tuple(axis.axis_id for axis in result.axes),
        selection_status="unresolved",
        candidate_catalog_version="source-intent-graph-candidates-v1",
    )
    semantic = semantic_summary_from_ir(result.candidates[0].ir, rag_context=rag_context)
    return replace(
        semantic,
        candidate_ir_set=candidate_set,
        semantic_status="needs_semantic_review",
        source_axis_debt_resolutions=tuple(
            json.loads(json.dumps(asdict(binding), ensure_ascii=False))
            for binding in result.graph.axis_debt_resolutions
        ) if result.graph is not None else (),
        persistent_rag_allowed=False,
    )


def bind_plain_review_to_source_axes(review, result, *, mapping_client=None):
    """Bind reviewed option IDs to graph options only when wording is decisive."""

    if review is None:
        return {}, {}
    unresolved = tuple(
        axis for axis in review.ambiguities
        if axis.decision_status == "ambiguous_unresolved"
    )
    if not unresolved:
        return {}, {}

    def overlaps(left, right):
        return left[0] < right[1] and right[0] < left[1]

    def reading_class(value):
        text = str(value).casefold()
        classes = []
        vocab = {
            "wide_scope": ("whole", "entire", "together", "same future", "conjunction"),
            "distributed_scope": ("each", "independent", "separate", "different future"),
            "current_or_future": ("current", "including now", "now or later"),
            "strict_future": ("strict", "later", "after the current", "future position only"),
        }
        for name, words in vocab.items():
            if any(word in text for word in words):
                classes.append(name)
        return classes[0] if len(set(classes)) == 1 else ""

    gate_mapping = {}
    reverse_mapping = {}
    for ambiguity in unresolved:
        matching_axes = tuple(
            axis for axis in result.axes
            if overlaps(tuple(ambiguity.source_span), tuple(axis.source_span))
            and len(axis.options) == len(ambiguity.options)
        )
        if len(matching_axes) != 1:
            continue
        source_axis = matching_axes[0]
        by_class = {option.option_id: option for option in source_axis.options}
        option_map = {}
        reverse = {}
        for option in ambiguity.options:
            cls = reading_class(
                option.plain_interpretation + " " + option.difference_summary
            )
            source_option = by_class.get(cls)
            if source_option is None or source_option.option_id in reverse:
                option_map = {}
                break
            option_map[option.option_id] = {
                "operation": source_option.operation,
                "node_id": source_option.target_node_id,
                "source_axis_id": source_axis.axis_id,
                "source_option_id": source_option.option_id,
                "source_span": list(source_axis.source_span),
                "claim_id": source_axis.claim_id,
            }
            reverse[source_option.option_id] = option.option_id
        if len(option_map) == len(ambiguity.options):
            gate_mapping[ambiguity.ambiguity_id] = option_map
            reverse_mapping[ambiguity.ambiguity_id] = {
                (source_axis.axis_id, source_option_id): option_id
                for source_option_id, option_id in reverse.items()
            }
    if mapping_client is not None:
        # Ordinary-language meanings are handled by the model, rather than
        # treating English keyword matches as authority over the user's prose.
        gate_mapping, reverse_mapping = {}, {}
    unmapped = tuple(axis for axis in unresolved if axis.ambiguity_id not in gate_mapping)
    if mapping_client is not None and unmapped:
        extra, reverse = _model_plain_axis_bindings(unmapped, result, mapping_client)
        gate_mapping.update(extra)
        reverse_mapping.update(reverse)
    source_axis_ids = [next(iter(options.values()))["source_axis_id"] for options in gate_mapping.values()]
    if len(source_axis_ids) != len(set(source_axis_ids)):
        # Two purported independent axes cannot reuse one backend dimension;
        # that would present an incomplete Cartesian product as exhaustive.
        return {}, {}
    return gate_mapping, reverse_mapping


def _model_plain_axis_bindings(ambiguities, result, client):
    """Map wording to existing typed alternatives; never let a model add IRs."""
    from llm.errors import LLMResponseSchemaError
    from llm.schema_validation import validate_response_schema
    allowed = {
        ambiguity.ambiguity_id: tuple(
            axis for axis in result.axes
            if ambiguity.source_span[0] < axis.source_span[1]
            and axis.source_span[0] < ambiguity.source_span[1]
            and len(axis.options) == len(ambiguity.options)
        ) for ambiguity in ambiguities
    }
    eligible = tuple(axis for axis in ambiguities if allowed[axis.ambiguity_id])
    if not eligible:
        return {}, {}
    schema = {"type": "object", "properties": {"mappings": {
        "type": "array", "minItems": len(eligible), "maxItems": len(eligible),
        "items": {"type": "object", "properties": {
            "ambiguity_id": {"type": "string", "enum": [axis.ambiguity_id for axis in eligible]},
            "source_axis_id": {"type": "string"},
            "bindings": {"type": "array", "items": {
                "type": "object", "properties": {
                    "plain_option_id": {"type": "string"}, "source_option_id": {"type": "string"},
                }, "required": ["plain_option_id", "source_option_id"], "additionalProperties": False}},
            "reason": {"type": "string", "minLength": 1},
        }, "required": ["ambiguity_id", "source_axis_id", "bindings", "reason"], "additionalProperties": False},
    }}, "required": ["mappings"], "additionalProperties": False}
    payload = {
        "task": "map_plain_reading_options_to_source_axes",
        "source_text": result.graph.source_text,
        "axes": [{"ambiguity_id": ambiguity.ambiguity_id,
                  "evidence_text": ambiguity.evidence_text,
                  "plain_options": [asdict(option) for option in ambiguity.options],
                  "allowed_source_axes": [asdict(axis) for axis in allowed[ambiguity.ambiguity_id]]}
                 for ambiguity in eligible],
    }
    by_id = {axis.ambiguity_id: axis for axis in eligible}
    for _attempt in range(2):
        try:
            raw = client.complete_json(
                "Map ordinary-language option meanings, in any language, onto the supplied "
                "source-axis options. Use exact supplied IDs and a one-to-one mapping. "
                "Do not add nodes, axes, options, formulas, approval or truth labels. "
                "If no exact semantic correspondence exists, use an empty source_axis_id "
                "and an empty bindings list instead of guessing.",
                dict(payload), temperature=0.0, response_schema=schema,
            )
            validate_response_schema(raw, schema)
            if {item["ambiguity_id"] for item in raw["mappings"]} != set(by_id):
                raise ValueError("Missing or duplicate ambiguity mappings")
            gate, reverse = {}, {}
            for item in raw["mappings"]:
                ambiguity_id = item["ambiguity_id"]
                if not item["source_axis_id"] and not item["bindings"]:
                    continue
                axis = next((axis for axis in allowed[ambiguity_id] if axis.axis_id == item["source_axis_id"]), None)
                if axis is None:
                    raise ValueError("Mapping references an unavailable source axis")
                options = {option.option_id: option for option in axis.options}
                bindings = item["bindings"]
                if (len(bindings) != len(options)
                        or {entry["plain_option_id"] for entry in bindings} != {option.option_id for option in by_id[ambiguity_id].options}
                        or {entry["source_option_id"] for entry in bindings} != set(options)):
                    raise ValueError("Option mapping must be complete and one-to-one")
                gate[ambiguity_id] = {entry["plain_option_id"]: {
                    "operation": options[entry["source_option_id"]].operation,
                    "node_id": options[entry["source_option_id"]].target_node_id,
                    "source_axis_id": axis.axis_id, "source_option_id": entry["source_option_id"],
                    "source_span": list(axis.source_span), "claim_id": axis.claim_id,
                } for entry in bindings}
                reverse[ambiguity_id] = {
                    (axis.axis_id, entry["source_option_id"]): entry["plain_option_id"] for entry in bindings
                }
            return gate, reverse
        except (ValueError, TypeError, LLMResponseSchemaError) as exc:
            payload["validation_error"] = str(exc)
    return {}, {}
