"""Lexicographic semantic progress and representation-independent cycle guard."""
from dataclasses import dataclass
from typed_source_relation_graph import digest


class SemanticProgressStop(ValueError):
    code = "SEMANTIC_REVISION_NO_PROGRESS"


@dataclass(frozen=True, order=True)
class ConflictVector:
    verified_source_contradictions: int = 0
    unresolved_required_contract_axes: int = 0
    trace_label_conflicts: int = 0
    readiness_debt: int = 0
    structural_errors: int = 0

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in self.__dict__.values()):
            raise ValueError("conflict counts must be nonnegative integers")


def semantic_fingerprint(ir):
    """Ignore node ids, ordering and evidence selectors; retain all semantics."""
    identities = {ap.name: ap.ap_identity_id or ap.name for ap in ir.atomic_propositions}
    active = set()
    def visit(nid):
        if nid in active: raise ValueError("cyclic IR")
        active.add(nid)
        node = ir.node_map[nid]
        params = dict(node.parameters)
        if "ap_name" in params: params["ap_name"] = identities[params["ap_name"]]
        anchor = params.pop("anchor_node_id", None)
        if anchor is not None: params["anchor_semantics"] = visit(anchor)
        result = (node.node_type, node.scope, node.temporal_relation, node.boundary_policy,
                  node.allowed_semantic_readings, node.temporal_resolution, params,
                  tuple(visit(child) for child in node.children))
        active.remove(nid)
        return result
    return digest({"tree": visit(ir.root_node_id), "aps": sorted(identities.values()),
                   "axes": [(a.ambiguity_id, a.axis_kind, a.resolved, a.alternatives) for a in ir.ambiguities]})


class ProgressGuard:
    def __init__(self, ir, vector):
        self.last_legal_ir, self.vector = ir, vector
        self.seen = {semantic_fingerprint(ir)}

    def accept(self, ir, vector):
        fingerprint = semantic_fingerprint(ir)
        if fingerprint in self.seen or not vector < self.vector:
            raise SemanticProgressStop("semantic revision repeated or did not strictly reduce conflicts")
        self.seen.add(fingerprint)
        self.last_legal_ir, self.vector = ir, vector
        return ir


def assert_mechanical_repair_preserves_semantics(before, after):
    if semantic_fingerprint(before) != semantic_fingerprint(after):
        raise SemanticProgressStop("mechanical repair changed semantic invariants; source review required")
    return after
