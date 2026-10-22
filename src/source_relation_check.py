"""Formula-free source relation to FaithfulSemanticIR topology diagnostics.

This module reports only relations with explicit source anchors. It never
rewrites an IR and never turns a keyword into an unstated domain convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from typing import Iterable, Mapping, Any


@dataclass(frozen=True)
class SourceRelationClaim:
    claim_id: str
    relation: str
    antecedent_aps: tuple[str, ...]
    consequent_aps: tuple[str, ...]
    source_span: tuple[int, int]
    evidence_text: str
    authority: str = "source_text"
    confidence_type: str = "rule-assisted"
    conflict_reason: str = ""


@dataclass(frozen=True)
class RelationTopologyFinding:
    code: str
    severity: str
    claim_id: str
    node_ids: tuple[str, ...]
    message: str


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _aps_in_span(text: str, start: int, end: int, glossary: Mapping[str, Any]) -> tuple[str, ...]:
    found = []
    for ap, entry in glossary.items():
        if any(start <= a and b <= end for a, b in entry.get("spans", ())):
            found.append(ap)
            continue
        surface = str(entry.get("surface", ""))
        if surface and re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", text[start:end]):
            found.append(ap)
    return tuple(sorted(set(found)))


def extract_source_relation_claims(text: str, glossary: Mapping[str, Any]) -> tuple[SourceRelationClaim, ...]:
    """Extract explicit conditional direction; ambiguous parses remain absent."""
    candidates = []
    patterns = (
        ("if_then", re.compile(r"\bif\b(?P<a>.+?)\bthen\b(?P<c>.+?)(?:[.;]|$)", re.I | re.S)),
        ("only_if", re.compile(r"(?P<c>[^.;]+?)\bonly\s+if\b(?P<a>[^.;]+)", re.I)),
        ("postposed_if", re.compile(r"(?P<c>[^.;]+?)\bif\b(?P<a>[^.;]+)", re.I)),
        ("when", re.compile(r"\bwhen\b(?P<a>.+?)(?:,|\bthen\b)(?P<c>.+?)(?:[.;]|$)", re.I | re.S)),
    )
    occupied = []
    for relation, pattern in patterns:
        for match in pattern.finditer(text):
            if any(max(match.start(), a) < min(match.end(), b) for a, b in occupied):
                continue
            a_span, c_span = match.span("a"), match.span("c")
            if relation == "only_if":
                # "A only if B" states A -> B, unlike "A if B" (B -> A).
                antecedent = _aps_in_span(text, *c_span, glossary)
                consequent = _aps_in_span(text, *a_span, glossary)
            else:
                antecedent = _aps_in_span(text, *a_span, glossary)
                consequent = _aps_in_span(text, *c_span, glossary)
            if not antecedent or not consequent or set(antecedent) & set(consequent):
                continue
            payload = {"relation": relation, "a": antecedent, "c": consequent, "span": match.span()}
            candidates.append(SourceRelationClaim(
                claim_id="SRCREL-" + _digest(payload), relation=relation,
                antecedent_aps=antecedent, consequent_aps=consequent,
                source_span=match.span(), evidence_text=text[match.start():match.end()],
            ))
            occupied.append(match.span())
    # Coordination and ordered relations are recorded only when each side has
    # a distinct source-bound AP.  They remain reviewer-verifiable claims, not
    # inferred temporal operators.
    mentions = sorted(
        (start, end, ap)
        for ap, entry in glossary.items()
        for start, end in entry.get("spans", ())
    )
    for relation, marker_pattern in (
        ("coordination_or", r"\bor\b"),
        ("coordination_and", r"\band\b"),
    ):
        for marker in re.finditer(marker_pattern, text, re.I):
            sentence_start = max(text.rfind(".", 0, marker.start()), text.rfind(";", 0, marker.start())) + 1
            sentence_end_candidates = [position for token in (".", ";") if (position := text.find(token, marker.end())) >= 0]
            sentence_end = min(sentence_end_candidates) if sentence_end_candidates else len(text)
            left = [item for item in mentions if sentence_start <= item[0] and item[1] <= marker.start()]
            right = [item for item in mentions if marker.end() <= item[0] and item[1] <= sentence_end]
            if not left or not right:
                continue
            left_end = max(item[1] for item in left)
            right_start = min(item[0] for item in right)
            antecedent = tuple(sorted({ap for start, end, ap in left if end == left_end}))
            consequent = tuple(sorted({ap for start, end, ap in right if start == right_start}))
            payload = {"relation": relation, "a": antecedent, "c": consequent, "span": (left_end, right_start)}
            candidates.append(SourceRelationClaim(
                claim_id="SRCREL-" + _digest(payload),
                relation=relation,
                antecedent_aps=antecedent,
                consequent_aps=consequent,
                source_span=(left_end, right_start),
                evidence_text=text[left_end:right_start],
            ))
    binary_patterns = (
        ("before", re.compile(r"(?P<a>[^.;,]+?)\bbefore\b(?P<c>[^.;,]+)", re.I)),
        ("after", re.compile(r"(?P<a>[^.;,]+?)\bafter\b(?P<c>[^.;,]+)", re.I)),
        ("until", re.compile(r"(?P<a>[^.;,]+?)\buntil\b(?P<c>[^.;,]+)", re.I)),
        ("while", re.compile(r"\bwhile\b(?P<a>[^,.;]+?)(?:,|\bthen\b)(?P<c>[^.;]+)", re.I)),
    )
    for relation, pattern in binary_patterns:
        for match in pattern.finditer(text):
            antecedent = _aps_in_span(text, *match.span("a"), glossary)
            consequent = _aps_in_span(text, *match.span("c"), glossary)
            if not antecedent or not consequent or set(antecedent) & set(consequent):
                continue
            payload = {"relation": relation, "a": antecedent, "c": consequent, "span": match.span()}
            candidate = SourceRelationClaim(
                claim_id="SRCREL-" + _digest(payload),
                relation=relation,
                antecedent_aps=antecedent,
                consequent_aps=consequent,
                source_span=match.span(),
                evidence_text=text[match.start():match.end()],
            )
            if candidate.claim_id not in {item.claim_id for item in candidates}:
                candidates.append(candidate)
    return tuple(candidates)


def _ap_names(node_id: str, nodes: Mapping[str, Any], trail=()) -> set[str]:
    if node_id in trail or node_id not in nodes:
        return set()
    node = nodes[node_id]
    if node.node_type == "ap_condition":
        name = node.parameter("ap_name", "")
        return {name} if name else set()
    result = set()
    for child in node.children:
        result.update(_ap_names(child, nodes, trail + (node_id,)))
    return result


def _parents(nodes: Mapping[str, Any]) -> dict[str, set[str]]:
    result = {node_id: set() for node_id in nodes}
    for node in nodes.values():
        for child in node.children:
            result.setdefault(child, set()).add(node.node_id)
    return result


def _has_global_ancestor(node_id: str, nodes: Mapping[str, Any], parents: Mapping[str, set[str]]) -> bool:
    pending, seen = list(parents.get(node_id, ())), set()
    while pending:
        parent = pending.pop()
        if parent in seen:
            continue
        seen.add(parent)
        if nodes[parent].node_type == "global_constraint":
            return True
        pending.extend(parents.get(parent, ()))
    return False


def _matching_binary_node(claim, nodes, node_types):
    reverse = None
    for node in nodes.values():
        if node.node_type not in node_types or len(node.children) < 2:
            continue
        left = _ap_names(node.children[0], nodes)
        right = set().union(*(_ap_names(child, nodes) for child in node.children[1:]))
        if set(claim.antecedent_aps) <= left and set(claim.consequent_aps) <= right:
            return node, reverse
        if set(claim.antecedent_aps) <= right and set(claim.consequent_aps) <= left:
            reverse = node
    return None, reverse


def check_source_relation_topology(text: str, glossary: Mapping[str, Any], ir, ap_binding_context=None) -> dict[str, Any]:
    original_ir = ir
    canonicalized = False
    binding_findings = []
    if ap_binding_context is not None:
        ap_binding_context.validate_source(text)
        names = set(glossary) | {
            node.parameter("ap_name") for node in ir.nodes if node.node_type == "ap_condition"
        }
        # Proposal-only identities may canonicalize formula-free diagnostics,
        # but this lookup deliberately does not grant catalog authority.
        identities = {
            name: ap_binding_context.resolve_candidate_identity(name)
            for name in names
        }
        for name, identity in sorted(identities.items()):
            if identity is None:
                binding_findings.append(RelationTopologyFinding(
                    "AP_ALIAS_UNRESOLVED", "diagnostic", "", (),
                    "No unique authoritative AP identity for " + name,
                ))
        if not binding_findings:
            canonicalized = True
            normalized = {}
            for name, entry in glossary.items():
                key = identities[name].identity_key
                target = normalized.setdefault(key, {**entry, "spans": []})
                target["spans"].extend(entry.get("spans", ()))
            glossary = normalized
            ir = replace(ir, nodes=tuple(
                replace(node, parameters=tuple(
                    (key, identities[value].identity_key if key == "ap_name" else value)
                    for key, value in node.parameters
                )) if node.node_type == "ap_condition" else node
                for node in ir.nodes
            ))
    if binding_findings:
        report = {
            "version": "source-relation-topology-v2", "source_hash": _digest(text),
            "ir_version": ir.ir_version, "claims": [],
            "findings": [asdict(item) for item in binding_findings],
            "passed": False, "verdict": "inconclusive", "canonicalized": False,
            "diagnostic_only": True, "rewrite_performed": False,
        }
        report["report_hash"] = _digest(report)
        return report
    claims = extract_source_relation_claims(text, glossary)
    nodes = ir.node_map
    parents = _parents(nodes)
    implications = [node for node in ir.nodes if node.node_type == "implication" and len(node.children) == 2]
    findings: list[RelationTopologyFinding] = []
    explicit_global = bool(re.search(r"\b(globally|always|whenever|at\s+all\s+times)\b", text, re.I))
    for claim in claims:
        if claim.relation in {"coordination_or", "coordination_and"}:
            expected = "disjunction" if claim.relation.endswith("or") else "conjunction"
            matched = next(
                (
                    node for node in ir.nodes
                    if node.node_type == expected
                    and set(claim.antecedent_aps + claim.consequent_aps) <= _ap_names(node.node_id, nodes)
                ),
                None,
            )
            if matched is None:
                findings.append(RelationTopologyFinding(
                    "COORDINATION_TOPOLOGY_MISMATCH", "error", claim.claim_id, (),
                    f"Explicit {claim.relation.rsplit('_', 1)[-1]} members are not preserved by one {expected} node.",
                ))
            continue
        if claim.relation in {"before", "after", "until", "while"}:
            types = {
                "before": {"before", "precedence"},
                "after": {"after"},
                "until": {"until"},
                # 'while' may be represented as an implication whose guard is
                # the interval condition; boundary policy remains contractual.
                "while": {"implication", "until"},
            }[claim.relation]
            matched, reverse = _matching_binary_node(claim, nodes, types)
            if matched is None:
                findings.append(RelationTopologyFinding(
                    "ORDERED_ROLE_REVERSED" if reverse is not None else "ORDERED_RELATION_TOPOLOGY_MISSING",
                    "error", claim.claim_id,
                    (reverse.node_id,) if reverse is not None else (),
                    "IR does not preserve the explicitly anchored temporal/interval roles.",
                ))
            continue
        matched = None
        reverse = None
        for node in implications:
            left, right = (_ap_names(child, nodes) for child in node.children)
            if set(claim.antecedent_aps) <= left and set(claim.consequent_aps) <= right:
                matched = node
                break
            if set(claim.antecedent_aps) <= right and set(claim.consequent_aps) <= left:
                reverse = node
        if matched is None:
            if reverse is not None:
                findings.append(RelationTopologyFinding(
                    "CONDITIONAL_DIRECTION_REVERSED", "error", claim.claim_id,
                    (reverse.node_id,), "IR antecedent/consequent are reversed relative to the explicit source conditional.",
                ))
            else:
                findings.append(RelationTopologyFinding(
                    "CONDITIONAL_TOPOLOGY_MISSING", "error", claim.claim_id, (),
                    "No implication topology preserves the explicit source condition and result roles.",
                ))
        elif explicit_global and not _has_global_ancestor(matched.node_id, nodes, parents):
            findings.append(RelationTopologyFinding(
                "GLOBAL_SCOPE_DOES_NOT_WRAP_RELATION", "error", claim.claim_id,
                (matched.node_id,), "The source explicitly marks global/repeated scope, but that scope does not wrap the full implication.",
            ))
    node_types = {node.node_type for node in ir.nodes}
    duration_evidence = bool(re.search(r"\bfor\s+(?:only\s+)?(?:one|\d+)\s+(?:step|steps|time\s+units?)\b", text, re.I))
    total_count_evidence = bool(re.search(r"\b(?:in\s+total|altogether|total\s+of|exactly\s+\d+\s+times?)\b", text, re.I))
    if duration_evidence and "cardinality" in node_types and not total_count_evidence:
        findings.append(RelationTopologyFinding(
            "DURATION_MISREAD_AS_TOTAL_COUNT", "error", "", tuple(node.node_id for node in ir.nodes if node.node_type == "cardinality"),
            "The source explicitly supplies duration but no total-count scope; cardinality requires separate source support.",
        ))
    report = {
        "version": "source-relation-topology-v2",
        "source_hash": _digest(text),
        "ir_version": ir.ir_version,
        "claims": [asdict(item) for item in claims],
        "findings": [asdict(item) for item in findings],
        "passed": not any(item.severity == "error" for item in findings),
        "diagnostic_only": True,
        "rewrite_performed": False,
        # Regex claims are supported diagnostics, never mechanical proofs.
        "canonicalized": canonicalized,
        "verdict": "inconclusive",
    }
    report["report_hash"] = _digest(report)
    if canonicalized:
        from typed_source_relation_graph import parse_source_graph, UnsupportedConstruct
        try:
            graph = parse_source_graph(text, ap_binding_context)
        except UnsupportedConstruct:
            pass
        else:
            proven = _check_proven_graph(graph, original_ir, ap_binding_context)
            report.update(proven)
            report["report_hash"] = _digest({k: v for k, v in report.items() if k != "report_hash"})
    return report


def _check_proven_graph(graph, ir, context):
    """Certify exact Boolean topology; temporal boundary policies stay unknown."""
    source_nodes = {node.node_id: node for node in graph.nodes}
    edges = {edge.from_ids[0]: edge for edge in graph.edges}
    supported = {"AP", "negation", "conjunction", "disjunction", "implication", "global_constraint"}
    if any(node.kind not in supported for node in graph.nodes):
        return {}

    def source_signature(nid):
        node = source_nodes[nid]
        if node.kind == "AP": return ("ap_condition", node.ap_identity_key)
        children = tuple(source_signature(child) for child in edges[nid].to_ids)
        if node.kind in {"conjunction", "disjunction"}: children = tuple(sorted(children))
        return (node.kind, children)

    def ir_signature(nid):
        node = ir.node_map[nid]
        if node.node_type == "ap_condition":
            identity = context.resolve_identity(node.parameter("ap_name"))
            return ("ap_condition", identity.identity_key if identity else "?")
        children = tuple(ir_signature(child) for child in node.children)
        if node.node_type in {"conjunction", "disjunction"}: children = tuple(sorted(children))
        return (node.node_type, children)

    if source_signature(graph.root_node_id) == ir_signature(ir.root_node_id):
        return {"verdict": "verified_match", "passed": True, "findings": []}
    claims, findings = [], []
    for source in graph.nodes:
        if source.kind != "implication": continue
        signature = source_signature(source.node_id)
        # Only the same exact source region, with opposite complete subtrees,
        # supports a direction contradiction. Missing topology is inconclusive.
        for node in ir.nodes:
            if node.source_span != source.source_span or node.node_type != "implication": continue
            expected = signature[1]
            if expected[0] != expected[1] and ir_signature(node.node_id) == ("implication", expected[::-1]):
                cid = "SRCREL-" + _digest((graph.graph_hash, source.node_id))
                claims.append({"claim_id": cid, "confidence": "proven", "relation": "implication",
                               "source_span": source.source_span, "evidence_text": source.exact_text,
                               "source_edge_id": edges[source.node_id].edge_id})
                findings.append({"code": "CONDITIONAL_DIRECTION_REVERSED", "claim_id": cid,
                                 "node_ids": [node.node_id], "conflict_field": "children",
                                 "severity": "error", "verdict": "verified_contradiction"})
    return ({"verdict": "verified_contradiction", "claims": claims, "findings": findings,
             "passed": False} if findings else {})


def confirmed_contradictions(report, response, source_text):
    """Validate per-claim confirmation; unsupported/malformed claims abstain."""
    if not report.get("canonicalized") or report.get("source_hash") != _digest(source_text):
        return ()
    claims = {item["claim_id"]: item for item in report.get("claims", ())}
    decisions = response.get("claims", ())
    if not isinstance(decisions, list):
        return ()
    confirmed = []
    for finding in report.get("findings", ()):
        claim = claims.get(finding.get("claim_id"), {})
        if (claim.get("confidence") != "proven" or not finding.get("node_ids")
                or not finding.get("conflict_field")
                or finding.get("verdict") != "verified_contradiction"):
            continue
        matches = [item for item in decisions if isinstance(item, dict)
                   and item.get("claim_id") == claim.get("claim_id")]
        if len(matches) != 1:
            continue
        decision = matches[0]
        spans = decision.get("source_spans")
        if (decision.get("decision") != "confirm" or not isinstance(spans, list) or not spans
                or decision.get("affected_node_ids") != finding["node_ids"]):
            continue
        anchor = claim.get("source_span", ())
        if all(isinstance(span, (list, tuple)) and len(span) == 2
               and all(type(i) is int for i in span) and 0 <= span[0] < span[1] <= len(source_text)
               for span in spans) and list(anchor) in [list(span) for span in spans]:
            confirmed.append(finding)
    return tuple(confirmed)


__all__ = ["SourceRelationClaim", "RelationTopologyFinding", "extract_source_relation_claims", "check_source_relation_topology"]
