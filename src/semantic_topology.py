"""Formula-free semantic topology contracts shared by NLU and IR validation.

The functions in this module operate on proposal mappings before they become
``FaithfulSemanticIR`` instances.  They deliberately do not compile formulas
and do not depend on a concrete agent.  This keeps domain topology detection,
participant binding, and raw graph integrity checks consistent across proposal
validation and deterministic repair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re
from types import SimpleNamespace
from typing import Any

from contract_versions import SEMANTIC_TOPOLOGY_CONTRACT_VERSION, FAITHFUL_IR_SCHEMA_VERSION
from semantic_node_contracts import node_contract_findings

FUTURE_PROHIBITION_TOPOLOGY_CODE = "FUTURE_PROHIBITION_TOPOLOGY_INVALID"
FUTURE_PROHIBITION_TOPOLOGY_PATH = "/faithful_ir/root_node_id"
FUTURE_PROHIBITION_TOPOLOGY_MESSAGE = (
    "post-trigger future prohibition cannot use precedence, after, or a flat "
    "response; model global(trigger -> next(global(negation(event))))"
)
IR_SCOPE_TOPOLOGY_MISMATCH = "IR_SCOPE_TOPOLOGY_MISMATCH"
IR_SCOPE_METADATA_INVALID = "IR_SCOPE_METADATA_INVALID"
IR_SCOPE_AMBIGUITY_UNRESOLVED = "IR_SCOPE_AMBIGUITY_UNRESOLVED"

_CANONICAL_SCOPE_METADATA = frozenset(
    {"current_position", "inherited", "all_positions"}
)
_GLOBAL_SCOPE_ALIASES = frozenset(
    {"global", "entire_trace", "entire trace", "whole_trace", "whole trace"}
)

# Versioned serialization aliases.  The node type proves the meaning of a
# global wrapper; aliases on other nodes still require semantic adjudication.
SCOPE_ALIAS_SCHEMA_VERSION = "scope-aliases-v1"
_SCOPE_ALIASES_BY_NODE = {
    "global_constraint": {
        "global": "all_positions", "entire_trace": "all_positions",
        "whole_trace": "all_positions", "entire trace": "all_positions",
        "whole trace": "all_positions",
    },
}


def normalize_mechanical_scope_aliases(proposal):
    """Return a copy with only contract-proven scope spelling normalized."""
    from copy import deepcopy

    normalized = deepcopy(proposal)
    ir, _ = faithful_ir_proposal_mapping(normalized)
    if ir is None and isinstance(normalized, dict) and "nodes" in normalized:
        ir = normalized
    if (not isinstance(ir, dict)
            or str(ir.get("schema_version", "")) != FAITHFUL_IR_SCHEMA_VERSION
            or not isinstance(ir.get("nodes"), list)):
        return normalized
    for node in ir["nodes"]:
        if not isinstance(node, dict):
            continue
        aliases = _SCOPE_ALIASES_BY_NODE.get(node.get("node_type"), {})
        value = node.get("scope")
        if isinstance(value, str) and value in aliases:
            node["scope"] = aliases[value]
    return normalized
_SCOPE_NODE_TYPES = frozenset(
    {"global_constraint", "persist_to_end", "bounded_window"}
)
_SCOPE_AMBIGUITY_AXES = frozenset(
    {"scope", "scope_topology", "global_scope", "evaluation_scope", "globality"}
)

_FUTURE_PROHIBITION_MARKERS = (
    "won't happen again",
    "will not happen again",
    "must not happen again",
    "won't occur again",
    "will not occur again",
    "must not occur again",
    "never happen again",
    "never happens again",
    "never occur again",
    "never occurs again",
    "之后不再",
    "以后不再",
)
_AP_TOKEN = r"[A-Za-z_][A-Za-z0-9_]*"
_FUTURE_PROHIBITION_PATTERNS = (
    re.compile(
        rf"\bonce\s+(?P<trigger>{_AP_TOKEN})\s+"
        rf"(?:happened|occurred|holds?),?\s+"
        rf"(?P<event>{_AP_TOKEN})\s+"
        r"(?:won't|will\s+not|must\s+not)\s+"
        r"(?:happen|occur)\s+again\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"\bafter\s+(?P<trigger>{_AP_TOKEN})\s+"
        rf"(?:happens|occurs|holds?),?\s+"
        rf"(?P<event>{_AP_TOKEN})\s+"
        r"(?:never|must\s+not)\s+(?:happens?|occurs?)\s+again\b",
        flags=re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class GraphIntegrityProblem:
    """One evidence-independent error in an uncanonicalized IR graph."""

    code: str
    path: str
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FutureProhibitionBinding:
    """Unambiguous formula-free participants for a suffix prohibition."""

    trigger_node_id: str
    prohibited_node_id: str
    basis: tuple[str, ...] = ()
    contract_version: str = SEMANTIC_TOPOLOGY_CONTRACT_VERSION


@dataclass(frozen=True)
class RelationBinding:
    """Ordered relation participants derived only from graph child order."""

    node_id: str
    relation_type: str
    participant_node_ids: tuple[str, ...]
    contract_version: str = SEMANTIC_TOPOLOGY_CONTRACT_VERSION

    @property
    def antecedent_node_id(self) -> str:
        return self.participant_node_ids[0] if self.participant_node_ids else ""

    @property
    def consequent_node_id(self) -> str:
        return self.participant_node_ids[1] if len(self.participant_node_ids) > 1 else ""


@dataclass(frozen=True)
class ScopeBinding:
    """Executable scope boundary represented by an explicit IR wrapper node."""

    node_id: str
    scope_type: str
    child_node_ids: tuple[str, ...]
    lower_bound: int | None = None
    upper_bound: int | None = None
    contract_version: str = SEMANTIC_TOPOLOGY_CONTRACT_VERSION


@dataclass(frozen=True)
class IRSemanticView:
    """Formula-free downstream view whose sole input is FaithfulSemanticIR."""

    ir_version: str
    root_node_id: str
    root_node_type: str
    relation_bindings: tuple[RelationBinding, ...]
    scope_bindings: tuple[ScopeBinding, ...]
    ap_names_by_node_id: tuple[tuple[str, str], ...]
    parent_node_ids: tuple[tuple[str, str], ...] = ()
    scope_ancestors_by_node_id: tuple[tuple[str, tuple[str, ...]], ...] = ()
    temporal_node_ids: tuple[str, ...] = ()
    unresolved_ambiguity_ids: tuple[str, ...] = ()
    contract_version: str = SEMANTIC_TOPOLOGY_CONTRACT_VERSION

    @property
    def primary_relation(self) -> RelationBinding | None:
        return self.relation_bindings[0] if self.relation_bindings else None

    def ap_names_under(
        self,
        node_map: Mapping[str, object],
        root_node_id: str,
    ) -> tuple[str, ...]:
        ap_names = dict(self.ap_names_by_node_id)
        return tuple(
            ap_names[node_id]
            for node_id in subtree_node_ids(node_map, (root_node_id,))
            if node_id in ap_names
        )

    def scope_ancestors(self, node_id: str) -> tuple[str, ...]:
        return dict(self.scope_ancestors_by_node_id).get(node_id, ())


def build_ir_semantic_view(ir: object) -> IRSemanticView:
    """Build the only semantic view allowed for online downstream decisions."""

    node_map = getattr(ir, "node_map")
    root_node_id = str(getattr(ir, "root_node_id"))
    relations: list[RelationBinding] = []
    scopes: list[ScopeBinding] = []
    ap_names: list[tuple[str, str]] = []
    parents: list[tuple[str, str]] = []
    scope_ancestors: list[tuple[str, tuple[str, ...]]] = []
    temporal_nodes: list[str] = []
    visited: set[str] = set()

    def visit(node_id: str, active_scopes: tuple[str, ...] = ()) -> None:
        if node_id in visited or node_id not in node_map:
            return
        visited.add(node_id)
        node = node_map[node_id]
        node_type = _node_type(node)
        children = _node_children(node)
        scope_ancestors.append((node_id, active_scopes))
        if node_type in {
            "implication",
            "equivalence",
            "before",
            "after",
            "precedence",
            "until",
            "release",
            "ordered_sequence",
        }:
            relations.append(RelationBinding(node_id, node_type, children))
        if node_type in _SCOPE_NODE_TYPES:
            parameters = _node_parameters(node)
            scopes.append(
                ScopeBinding(
                    node_id=node_id,
                    scope_type=node_type,
                    child_node_ids=children,
                    lower_bound=_optional_int(parameters.get("lower_bound")),
                    upper_bound=_optional_int(parameters.get("upper_bound")),
                )
            )
        if node_type not in {
            "ap_condition",
            "boolean_constant",
            "negation",
            "conjunction",
            "disjunction",
            "implication",
            "equivalence",
        }:
            temporal_nodes.append(node_id)
        if node_type == "ap_condition":
            ap_name = str(_node_parameters(node).get("ap_name", ""))
            if ap_name:
                ap_names.append((node_id, ap_name))
        for child_id in children:
            parents.append((child_id, node_id))
            visit(
                child_id,
                (*active_scopes, node_id)
                if node_type in _SCOPE_NODE_TYPES
                else active_scopes,
            )

    visit(root_node_id)
    ambiguities = getattr(ir, "ambiguities", ())
    return IRSemanticView(
        ir_version=str(getattr(ir, "ir_version", "")),
        root_node_id=root_node_id,
        root_node_type=_node_type(node_map.get(root_node_id)),
        relation_bindings=tuple(relations),
        scope_bindings=tuple(scopes),
        ap_names_by_node_id=tuple(ap_names),
        parent_node_ids=tuple(parents),
        scope_ancestors_by_node_id=tuple(scope_ancestors),
        temporal_node_ids=tuple(temporal_nodes),
        unresolved_ambiguity_ids=tuple(
            str(getattr(ambiguity, "ambiguity_id", ""))
            for ambiguity in ambiguities
            if not bool(getattr(ambiguity, "resolved", False))
        ),
    )


def scope_topology_problems(ir: object) -> tuple[GraphIntegrityProblem, ...]:
    """Validate that scope metadata cannot manufacture executable topology."""

    problems: list[GraphIntegrityProblem] = []
    nodes = tuple(getattr(ir, "nodes", ()))
    for index, node in enumerate(nodes):
        node_id = str(getattr(node, "node_id", ""))
        node_type = _node_type(node)
        scope = str(getattr(node, "scope", ""))
        path = f"/faithful_ir/nodes/{index}/scope"
        if (
            scope in _GLOBAL_SCOPE_ALIASES
            and node_type != "global_constraint"
        ):
            problems.append(
                GraphIntegrityProblem(
                    IR_SCOPE_TOPOLOGY_MISMATCH,
                    path,
                    f"node {node_id} claims global scope without global_constraint topology",
                    {"node_id": node_id, "node_type": node_type, "scope": scope},
                )
            )
            continue
        if scope not in _CANONICAL_SCOPE_METADATA:
            problems.append(
                GraphIntegrityProblem(
                    IR_SCOPE_METADATA_INVALID,
                    path,
                    f"node {node_id} has non-canonical scope metadata {scope!r}",
                    {"node_id": node_id, "scope": scope},
                )
            )
            continue
        if node_type != "global_constraint" and scope == "all_positions":
            problems.append(
                GraphIntegrityProblem(
                    IR_SCOPE_TOPOLOGY_MISMATCH,
                    path,
                    f"node {node_id} cannot claim global scope without global_constraint topology",
                    {"node_id": node_id, "node_type": node_type, "scope": scope},
                )
            )
        children = _node_children(node)
        if node_type in _SCOPE_NODE_TYPES and len(children) == 1:
            child = getattr(ir, "node_map", {}).get(children[0])
            if _node_type(child) == node_type:
                problems.append(
                    GraphIntegrityProblem(
                        IR_SCOPE_TOPOLOGY_MISMATCH,
                        f"/faithful_ir/nodes/{index}/children/0",
                        f"node {node_id} contains a redundant nested {node_type} wrapper",
                        {"node_id": node_id, "child_node_id": children[0]},
                    )
                )

    for index, ambiguity in enumerate(getattr(ir, "ambiguities", ())):
        if bool(getattr(ambiguity, "resolved", False)):
            continue
        axis = str(getattr(ambiguity, "axis_kind", "")).strip().casefold()
        is_scope_axis = axis in _SCOPE_AMBIGUITY_AXES
        if is_scope_axis:
            problems.append(
                GraphIntegrityProblem(
                    IR_SCOPE_AMBIGUITY_UNRESOLVED,
                    f"/faithful_ir/ambiguities/{index}/resolved",
                    "scope ambiguity must be resolved before IR approval or compilation",
                    {"ambiguity_id": str(getattr(ambiguity, "ambiguity_id", ""))},
                )
            )
    return tuple(problems)


def faithful_ir_proposal_mapping(
    proposal: object,
) -> tuple[Mapping[str, Any] | None, str]:
    """Return the proposal IR and its concrete JSON-pointer prefix."""

    if not isinstance(proposal, Mapping):
        return None, ""
    direct = proposal.get("faithful_ir")
    if isinstance(direct, Mapping):
        return direct, "/faithful_ir"
    semantic = proposal.get("semantic")
    if isinstance(semantic, Mapping):
        nested = semantic.get("faithful_ir")
        if isinstance(nested, Mapping):
            return nested, "/semantic/faithful_ir"
    return None, ""


def is_future_prohibition_source(source_text: str) -> bool:
    text = source_text.casefold()
    return any(marker in text for marker in _FUTURE_PROHIBITION_MARKERS)


def raw_ir_graph_integrity_problems(
    proposal: object,
) -> tuple[GraphIntegrityProblem, ...]:
    """Validate temporary IDs and references before positional ID rewriting.

    This deliberately checks only graph invariants shared by every IR pattern.
    Pattern-specific arity and temporal constraints remain in their existing
    validators so unrelated semantic patterns are not reinterpreted here.
    """

    ir, prefix = faithful_ir_proposal_mapping(proposal)
    if ir is None:
        return ()
    raw_nodes = ir.get("nodes")
    if not isinstance(raw_nodes, list):
        return ()

    problems: list[GraphIntegrityProblem] = []
    node_map: dict[str, Mapping[str, Any]] = {}
    id_indexes: dict[str, list[int]] = {}
    children_by_id: dict[str, tuple[str, ...]] = {}

    for index, raw_node in enumerate(raw_nodes):
        node_path = f"{prefix}/nodes/{index}"
        if not isinstance(raw_node, Mapping):
            problems.append(
                GraphIntegrityProblem(
                    "IR_GRAPH_NODE_OBJECT_REQUIRED",
                    node_path,
                    "faithful_ir node must be a JSON object",
                )
            )
            continue
        node_id = str(raw_node.get("node_id", "")).strip()
        if not node_id:
            problems.append(
                GraphIntegrityProblem(
                    "IR_GRAPH_NODE_ID_REQUIRED",
                    node_path + "/node_id",
                    "faithful_ir node requires a non-empty temporary node_id",
                )
            )
            continue
        id_indexes.setdefault(node_id, []).append(index)
        node_map[node_id] = raw_node
        raw_children = raw_node.get("children", ())
        if not isinstance(raw_children, (list, tuple)):
            problems.append(
                GraphIntegrityProblem(
                    "IR_GRAPH_CHILDREN_ARRAY_REQUIRED",
                    node_path + "/children",
                    f"node {node_id} children must be an array",
                )
            )
            continue
        children_by_id[node_id] = tuple(str(child).strip() for child in raw_children)

        node_type = raw_node.get("node_type")
        if not isinstance(node_type, str) or not node_type.strip():
            problems.append(
                GraphIntegrityProblem(
                    "IR_GRAPH_NODE_TYPE_REQUIRED",
                    node_path + "/node_type",
                    f"node {node_id} requires a non-empty node_type",
                    {"node_id": node_id},
                )
            )
            continue
        raw_parameters = raw_node.get("parameters", {})
        if not isinstance(raw_parameters, Mapping):
            problems.append(
                GraphIntegrityProblem(
                    "IR_GRAPH_PARAMETERS_OBJECT_REQUIRED",
                    node_path + "/parameters",
                    f"node {node_id} parameters must be an object",
                    {"node_id": node_id, "node_type": node_type},
                )
            )
            continue
        schema_version = str(ir.get("schema_version", ""))
        contract_findings = node_contract_findings(
            SimpleNamespace(
                node_type=node_type,
                children=tuple(str(child) for child in raw_children),
                parameters=tuple(raw_parameters.items()),
            ),
            schema_version=schema_version,
        )
        for finding in contract_findings:
            problems.append(
                GraphIntegrityProblem(
                    finding.code,
                    node_path + finding.path_suffix,
                    finding.message,
                    {
                        "node_id": node_id,
                        "node_type": node_type,
                        "child_count": len(raw_children),
                    },
                )
            )

    duplicate_ids = {
        node_id: indexes
        for node_id, indexes in id_indexes.items()
        if len(indexes) > 1
    }
    for node_id, indexes in sorted(duplicate_ids.items()):
        problems.append(
            GraphIntegrityProblem(
                "IR_GRAPH_DUPLICATE_NODE_ID",
                f"{prefix}/nodes/{indexes[1]}/node_id",
                f"temporary node_id {node_id!r} is duplicated",
                {"node_id": node_id, "indexes": tuple(indexes)},
            )
        )

    raw_aps = ir.get("atomic_propositions")
    declared_ap_names: set[str] = set()
    if isinstance(raw_aps, list):
        for raw_ap in raw_aps:
            if not isinstance(raw_ap, Mapping):
                continue
            for value in (raw_ap.get("name"), raw_ap.get("ap_id")):
                if isinstance(value, str) and value.strip():
                    declared_ap_names.add(value.strip())
    if declared_ap_names:
        for node_id, raw_node in node_map.items():
            if str(raw_node.get("node_type", "")) != "ap_condition":
                continue
            parameters = raw_node.get("parameters")
            ap_name = (
                str(parameters.get("ap_name", "")).strip()
                if isinstance(parameters, Mapping)
                else ""
            )
            if ap_name and ap_name not in declared_ap_names:
                node_index = id_indexes[node_id][0]
                problems.append(
                    GraphIntegrityProblem(
                        "IR_GRAPH_AP_REFERENCE_UNKNOWN",
                        f"{prefix}/nodes/{node_index}/parameters/ap_name",
                        f"ap_condition {node_id} references undeclared AP {ap_name!r}",
                        {"node_id": node_id, "ap_name": ap_name},
                    )
                )

    root_node_id = str(ir.get("root_node_id", "")).strip()
    if not root_node_id:
        problems.append(
            GraphIntegrityProblem(
                "IR_GRAPH_ROOT_ID_REQUIRED",
                prefix + "/root_node_id",
                "faithful_ir requires a non-empty root_node_id",
            )
        )
    elif root_node_id not in node_map:
        problems.append(
            GraphIntegrityProblem(
                "IR_GRAPH_ROOT_UNKNOWN",
                prefix + "/root_node_id",
                f"root_node_id references unknown node {root_node_id!r}",
                {"root_node_id": root_node_id},
            )
        )

    if duplicate_ids:
        # References through a duplicate ID are ambiguous.  Do not invent a
        # last-write-wins graph and do not report derivative reachability noise.
        return tuple(problems)

    for node_id, children in children_by_id.items():
        node_index = id_indexes[node_id][0]
        for child_index, child_id in enumerate(children):
            if child_id and child_id in node_map:
                continue
            problems.append(
                GraphIntegrityProblem(
                    "IR_GRAPH_CHILD_UNKNOWN",
                    f"{prefix}/nodes/{node_index}/children/{child_index}",
                    f"node {node_id} references unknown child {child_id!r}",
                    {"node_id": node_id, "child_id": child_id},
                )
            )

    if not root_node_id or root_node_id not in node_map:
        return tuple(problems)

    visited: set[str] = set()
    active: set[str] = set()
    cycle_nodes: list[str] = []

    def visit(node_id: str) -> None:
        if node_id in active:
            cycle_nodes.append(node_id)
            return
        if node_id in visited:
            return
        active.add(node_id)
        for child_id in children_by_id.get(node_id, ()):
            visit(child_id)
        active.remove(node_id)
        visited.add(node_id)

    visit(root_node_id)
    if cycle_nodes:
        problems.append(
            GraphIntegrityProblem(
                "IR_GRAPH_CYCLE",
                prefix + "/root_node_id",
                "faithful_ir graph contains a cycle",
                {"cycle_node_ids": tuple(dict.fromkeys(cycle_nodes))},
            )
        )
    disconnected = tuple(sorted(set(node_map) - visited))
    if disconnected:
        problems.append(
            GraphIntegrityProblem(
                "IR_GRAPH_DISCONNECTED_NODE",
                prefix + "/nodes",
                "faithful_ir graph contains nodes disconnected from root_node_id",
                {"node_ids": disconnected, "violation_count": len(disconnected)},
            )
        )
    return tuple(problems)


def has_future_prohibition_topology(
    root_node_id: str,
    node_map: Mapping[str, object],
) -> bool:
    """Require the exact formula-free wrapper for strict-future prohibition."""

    root = node_map.get(root_node_id)
    root_children = _node_children(root)
    if _node_type(root) != "global_constraint" or len(root_children) != 1:
        return False
    implication = node_map.get(root_children[0])
    implication_children = _node_children(implication)
    if _node_type(implication) != "implication" or len(implication_children) != 2:
        return False
    next_node = node_map.get(implication_children[1])
    next_children = _node_children(next_node)
    if _node_type(next_node) != "next_step" or len(next_children) != 1:
        return False
    suffix = node_map.get(next_children[0])
    suffix_children = _node_children(suffix)
    if (
        _node_type(suffix) not in {"global_constraint", "persist_to_end"}
        or len(suffix_children) != 1
    ):
        return False
    prohibition = node_map.get(suffix_children[0])
    return _node_type(prohibition) == "negation" and len(_node_children(prohibition)) == 1


def future_prohibition_binding_from_valid_graph(
    root_node_id: str,
    node_map: Mapping[str, object],
) -> FutureProhibitionBinding | None:
    if not has_future_prohibition_topology(root_node_id, node_map):
        return None
    root_children = _node_children(node_map[root_node_id])
    implication = node_map[root_children[0]]
    implication_children = _node_children(implication)
    next_node = node_map[implication_children[1]]
    suffix = node_map[_node_children(next_node)[0]]
    prohibition = node_map[_node_children(suffix)[0]]
    return FutureProhibitionBinding(
        trigger_node_id=implication_children[0],
        prohibited_node_id=_node_children(prohibition)[0],
        basis=("validated_topology",),
    )


def resolve_future_prohibition_binding(
    proposal: object,
    *,
    source_text: str,
) -> FutureProhibitionBinding | None:
    """Resolve trigger/prohibited subtrees without trusting free-text roles.

    Structure and source evidence are authoritative. Free-text
    ``semantic_role`` metadata is never consulted.
    """

    if not is_future_prohibition_source(source_text):
        return None
    integrity = raw_ir_graph_integrity_problems(proposal)
    if any(
        problem.code
        in {
            "IR_GRAPH_NODE_OBJECT_REQUIRED",
            "IR_GRAPH_NODE_ID_REQUIRED",
            "IR_GRAPH_CHILDREN_ARRAY_REQUIRED",
            "IR_GRAPH_DUPLICATE_NODE_ID",
            "IR_GRAPH_ROOT_ID_REQUIRED",
            "IR_GRAPH_ROOT_UNKNOWN",
            "IR_GRAPH_CHILD_UNKNOWN",
            "IR_GRAPH_CYCLE",
            "IR_GRAPH_DISCONNECTED_NODE",
        }
        for problem in integrity
    ):
        return None
    ir, _prefix = faithful_ir_proposal_mapping(proposal)
    if ir is None:
        return None
    raw_nodes = ir.get("nodes")
    if not isinstance(raw_nodes, list):
        return None
    node_map = {
        str(node.get("node_id", "")): node
        for node in raw_nodes
        if isinstance(node, Mapping) and str(node.get("node_id", ""))
    }
    root_id = str(ir.get("root_node_id", ""))
    if root_id not in node_map:
        return None

    validated_binding = future_prohibition_binding_from_valid_graph(
        root_id,
        node_map,
    )
    if validated_binding is not None:
        return validated_binding

    basis: list[str] = []
    structural_trigger: str | None = None
    structural_events: tuple[str, ...] = ()
    relation_id = root_id
    root = node_map[root_id]
    root_children = _node_children(root)
    if _node_type(root) == "global_constraint" and len(root_children) == 1:
        relation_id = root_children[0]
    relation = node_map.get(relation_id)
    relation_children = _node_children(relation)
    relation_type = _node_type(relation)
    if relation_type == "implication" and len(relation_children) == 2:
        structural_trigger = relation_children[0]
        structural_events = _negated_targets(
            relation_children[1],
            node_map,
        )
        basis.append("implication_structure")
    elif relation_type == "after" and len(relation_children) == 2:
        structural_trigger = relation_children[0]
        structural_events = _negated_targets(
            relation_children[1],
            node_map,
        )
        basis.append("after_structure")
    else:
        structural_events = _negated_targets(root_id, node_map)
        if structural_events:
            basis.append("negation_structure")

    lexical = future_prohibition_lexical_participants(source_text)
    ap_nodes = tuple(
        node_id
        for node_id, node in node_map.items()
        if _node_type(node) == "ap_condition"
    )
    lexical_triggers: tuple[str, ...] = ()
    lexical_events: tuple[str, ...] = ()
    if lexical is not None:
        lexical_triggers = _ap_nodes_named(node_map, lexical[0])
        lexical_events = _ap_nodes_named(node_map, lexical[1])
        if lexical_triggers and lexical_events:
            basis.append("source_lexical_evidence")

    trigger_id = _resolve_candidate_root(
        structural_trigger,
        lexical_triggers,
        node_map,
    )
    event_id = _resolve_structural_candidates(
        structural_events,
        lexical_events,
        node_map,
    )

    # Safe two-leaf completion is useful for flat LLM graphs, but only after the
    # source has already established this domain contract and one side is known.
    if len(ap_nodes) == 2:
        if trigger_id is not None and event_id is None:
            trigger_leaves = set(_ap_descendants(trigger_id, node_map))
            remaining = tuple(node_id for node_id in ap_nodes if node_id not in trigger_leaves)
            event_id = _unique(remaining)
            if event_id is not None:
                basis.append("two_leaf_completion")
        if event_id is not None and trigger_id is None:
            event_leaves = set(_ap_descendants(event_id, node_map))
            remaining = tuple(node_id for node_id in ap_nodes if node_id not in event_leaves)
            trigger_id = _unique(remaining)
            if trigger_id is not None:
                basis.append("two_leaf_completion")

    if not trigger_id or not event_id:
        return None
    if trigger_id == event_id:
        return None
    if set(_ap_descendants(trigger_id, node_map)) & set(
        _ap_descendants(event_id, node_map)
    ):
        return None
    return FutureProhibitionBinding(
        trigger_node_id=trigger_id,
        prohibited_node_id=event_id,
        basis=tuple(dict.fromkeys(basis)),
    )


def subtree_node_ids(
    node_map: Mapping[str, object],
    roots: Sequence[str],
) -> frozenset[str]:
    retained: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in retained or node_id not in node_map:
            return
        retained.add(node_id)
        for child_id in _node_children(node_map[node_id]):
            visit(child_id)

    for root in roots:
        visit(str(root))
    return frozenset(retained)


def future_prohibition_lexical_participants(
    source_text: str,
) -> tuple[str, str] | None:
    for pattern in _FUTURE_PROHIBITION_PATTERNS:
        match = pattern.search(source_text)
        if match is not None:
            return match.group("trigger"), match.group("event")
    return None


def _ap_nodes_named(
    node_map: Mapping[str, object],
    name: str,
) -> tuple[str, ...]:
    expected = name.casefold()
    token = re.compile(
        rf"(?<![A-Za-z0-9_']){re.escape(name)}(?![A-Za-z0-9_'])",
        flags=re.IGNORECASE,
    )
    return tuple(
        node_id
        for node_id, node in node_map.items()
        if _node_type(node) == "ap_condition"
        and (
            str(_node_parameters(node).get("ap_name", "")).casefold() == expected
            or bool(token.search(_node_evidence_text(node)))
        )
    )


def _node_evidence_text(node: object) -> str:
    if isinstance(node, Mapping):
        return str(node.get("evidence_text") or node.get("evidence_quote") or "")
    return str(
        getattr(node, "evidence_text", "")
        or getattr(node, "evidence_quote", "")
    )


def _resolve_candidate_root(
    structural: str | None,
    evidence_candidates: Sequence[str],
    node_map: Mapping[str, object],
) -> str | None:
    if structural is not None:
        if evidence_candidates and not any(
            _contains_node(structural, candidate, node_map)
            for candidate in evidence_candidates
        ):
            return None
        return structural
    return _unique(evidence_candidates)


def _resolve_structural_candidates(
    structural: Sequence[str],
    evidence_candidates: Sequence[str],
    node_map: Mapping[str, object],
) -> str | None:
    candidates = tuple(dict.fromkeys(str(item) for item in structural))
    if evidence_candidates:
        matching = tuple(
            candidate
            for candidate in candidates
            if any(
                _contains_node(candidate, evidence, node_map)
                for evidence in evidence_candidates
            )
        )
        if matching:
            candidates = matching
        elif candidates:
            return None
        else:
            return _unique(evidence_candidates)
    return _unique(candidates)


def _negated_targets(
    start_node_id: str,
    node_map: Mapping[str, object],
) -> tuple[str, ...]:
    targets: list[str] = []
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited or node_id not in node_map:
            return
        visited.add(node_id)
        node = node_map[node_id]
        children = _node_children(node)
        if _node_type(node) == "negation" and len(children) == 1:
            targets.append(children[0])
            return
        for child_id in children:
            visit(child_id)

    visit(start_node_id)
    return tuple(dict.fromkeys(targets))


def _contains_node(
    root_node_id: str,
    target_node_id: str,
    node_map: Mapping[str, object],
) -> bool:
    if root_node_id == target_node_id:
        return True
    visited: set[str] = set()
    pending = [root_node_id]
    while pending:
        node_id = pending.pop()
        if node_id in visited or node_id not in node_map:
            continue
        visited.add(node_id)
        children = _node_children(node_map[node_id])
        if target_node_id in children:
            return True
        pending.extend(children)
    return False


def _ap_descendants(
    root_node_id: str,
    node_map: Mapping[str, object],
) -> tuple[str, ...]:
    return tuple(
        node_id
        for node_id in subtree_node_ids(node_map, (root_node_id,))
        if _node_type(node_map[node_id]) == "ap_condition"
    )


def _node_type(node: object) -> str:
    if isinstance(node, Mapping):
        return str(node.get("node_type", ""))
    return str(getattr(node, "node_type", ""))


def _node_children(node: object) -> tuple[str, ...]:
    if isinstance(node, Mapping):
        children = node.get("children", ())
    else:
        children = getattr(node, "children", ())
    if not isinstance(children, (list, tuple)):
        return ()
    return tuple(str(child) for child in children)


def _node_parameters(node: object) -> Mapping[str, Any]:
    if isinstance(node, Mapping):
        parameters = node.get("parameters", {})
    else:
        parameters = getattr(node, "parameters", ())
    if isinstance(parameters, Mapping):
        return parameters
    if isinstance(parameters, (list, tuple)):
        return {
            str(item[0]): item[1]
            for item in parameters
            if isinstance(item, (list, tuple)) and len(item) == 2
        }
    return {}


def _unique(values: Sequence[str]) -> str | None:
    unique = tuple(dict.fromkeys(str(value) for value in values if str(value)))
    return unique[0] if len(unique) == 1 else None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "FUTURE_PROHIBITION_TOPOLOGY_CODE",
    "FUTURE_PROHIBITION_TOPOLOGY_MESSAGE",
    "FUTURE_PROHIBITION_TOPOLOGY_PATH",
    "FutureProhibitionBinding",
    "GraphIntegrityProblem",
    "IRSemanticView",
    "IR_SCOPE_AMBIGUITY_UNRESOLVED",
    "IR_SCOPE_METADATA_INVALID",
    "IR_SCOPE_TOPOLOGY_MISMATCH",
    "RelationBinding",
    "SEMANTIC_TOPOLOGY_CONTRACT_VERSION",
    "ScopeBinding",
    "build_ir_semantic_view",
    "faithful_ir_proposal_mapping",
    "future_prohibition_lexical_participants",
    "future_prohibition_binding_from_valid_graph",
    "has_future_prohibition_topology",
    "is_future_prohibition_source",
    "raw_ir_graph_integrity_problems",
    "resolve_future_prohibition_binding",
    "scope_topology_problems",
    "subtree_node_ids",
]
