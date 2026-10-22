"""Structured, formula-free handoff from trace disputes to semantic revision."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def build_semantic_failure_packets(*, source_text, input_profile, semantic, reconciliation, resolution_reports=()):
    """Create immutable packet payloads without selecting a replacement IR."""
    ir = semantic.faithful_ir
    if ir is None:
        raise ValueError("semantic failure packets require FaithfulSemanticIR")
    reports = {report.dispute_id: report for report in resolution_reports}
    records = {record.evidence_id: record for record in reconciliation.records}
    from trace_conflict_localization import localize_trace_conflict
    packets = []
    for dispute in reconciliation.unresolved_disputes:
        report = reports.get(dispute.dispute_id)
        source_records = [records[item] for item in dispute.evidence_ids if item in records]
        location = localize_trace_conflict(ir, dispute, records, report)
        rationales = [record.trace.rationale for record in source_records if record.trace.rationale]
        conflict_type = (
            "ir_nl_mismatch" if report is not None and report.requires_semantic_version_change
            else "source_semantics_missing" if report is not None and report.conflict_cause in {"missing_policy", "semantic_ambiguity"}
            else "annotation_conflict" if dispute.reason_code == "SAME_TRACE_DIFFERENT_LABEL"
            else "ir_trace_disagreement"
        )
        packet = {
            "version": "semantic-failure-packet-v1",
            "packet_id": "pending",
            "source_text": source_text,
            "source_hash": ir.source_hash,
            "input_profile_id": input_profile.get("profile_id", ""),
            "input_profile_hash": input_profile.get("profile_hash", ""),
            "semantic_context_id": dispute.semantic_context_id,
            "evidence_session_id": dispute.semantic_context_id,
            "ir_version": ir.ir_version,
            "ir_artifact_hash": getattr(ir, "artifact_hash", ""),
            "dispute_id": dispute.dispute_id,
            "minimal_conflict_trace": [list(step) for step in dispute.trace_shape],
            "independent_labels": list(dispute.labels),
            "independent_rationales": rationales,
            "affected_node_ids": list(location.node_ids),
            "authorized_node_ids": list(location.node_ids),
            "authorized_ir_fields": ["nodes"],
            "authorized_operations": list(location.operations),
            "contract_fields": list(getattr(report, "contract_fields", ())) if report is not None else [],
            "conflict_type": conflict_type,
            "source_spans": [list(span) for span in location.source_spans],
            "localization_status": location.status,
            "requires_semantic_version_change": bool(report and report.requires_semantic_version_change),
            "claim_ids": list(location.claim_ids),
            "obligation_node_ids": list(location.obligation_node_ids),
            "resolved_trace_label": (
                str(getattr(report, "trace_label", "")) if report is not None else ""
            ),
            "before_conflict_vector": {
                "verified_source_contradictions": 0,
                "unresolved_required_contract_axes": 0,
                "trace_label_conflicts": len(reconciliation.unresolved_disputes),
                "readiness_debt": 0,
                "structural_errors": 0,
            },
            "after_conflict_vector": None,
            "allowed_routes": ["correct_isolated_annotation", "bounded_semantic_revision", "candidate_or_human_resolution", "terminate_no_progress"],
            "contains_formula": False,
        }
        packet["packet_id"] = "SFP-" + _hash(packet)[:20]
        packet["artifact_hash"] = _hash(packet)
        if location.status == "localized":
            from execution_observer import emit
            emit(
                "semantic_failure_packet_created",
                requirement_id=ir.requirement_id,
                ir_version=ir.ir_version,
                ir_artifact_hash=ir.artifact_hash,
                input_profile_hash=packet["input_profile_hash"],
                packet_id=packet["packet_id"],
                entrypoint="build_semantic_failure_packets",
            )
        packets.append(packet)
    return tuple(packets)


def semantic_revision_preflight_violations(semantic, packets, *, input_profile=None):
    """Validate revision authority before any provider request is sent.

    Scope validation after generation is still required, but it is too late to
    discover that a packet cannot authorize a change after an online request
    has already been made.  This check is intentionally strict and formula
    blind: malformed, stale, empty, or unlocatable authority stops locally.
    """

    ir = semantic.faithful_ir
    if ir is None:
        return ("current semantic has no FaithfulSemanticIR",)
    if not packets:
        return ("no semantic failure packet supplied",)
    expected_profile_hash = (
        str((input_profile or {}).get("profile_hash", ""))
        if input_profile is not None
        else ""
    )
    known_nodes = set(ir.node_map)
    violations = []
    for index, packet in enumerate(packets):
        prefix = f"packet[{index}]"
        if not isinstance(packet, dict):
            violations.append(prefix + " is not an object")
            continue
        if packet.get("contains_formula") is not False:
            violations.append(prefix + " is not formula-blind")
        if packet.get("ir_version") != ir.ir_version:
            violations.append(prefix + " references a stale IR version")
        packet_hash = str(packet.get("ir_artifact_hash", ""))
        if packet_hash and packet_hash != getattr(ir, "artifact_hash", ""):
            violations.append(prefix + " references a stale IR artifact")
        if expected_profile_hash and packet.get("input_profile_hash") != expected_profile_hash:
            violations.append(prefix + " references a stale input profile")
        if packet.get("source_hash", ir.source_hash) != ir.source_hash:
            violations.append(prefix + " references a stale source")
        if not packet.get("evidence_session_id", packet.get("semantic_context_id", "")):
            violations.append(prefix + " has no evidence session")
        if not packet.get("claim_ids"):
            violations.append(prefix + " has no source claim")
        allowed = tuple(
            str(value)
            for value in packet.get(
                "authorized_node_ids", packet.get("affected_node_ids", ())
            )
            if str(value)
        )
        if not allowed:
            violations.append(prefix + " authorizes no IR nodes")
        unknown = sorted(set(allowed) - known_nodes)
        if unknown:
            violations.append(prefix + " authorizes unknown IR nodes: " + ", ".join(unknown))
        fields = packet.get("authorized_ir_fields", ())
        if not isinstance(fields, list) or not fields or set(fields) - {"nodes"}:
            violations.append(prefix + " has invalid authorized IR fields")
        operations = packet.get("authorized_operations", ())
        if (not isinstance(operations, list) or not operations or
                set(operations) - {"replace_parameter", "swap_children"}):
            violations.append(prefix + " has invalid authorized operations")
        if packet.get("localization_status", "localized") != "localized":
            violations.append(prefix + " has no valid conflict localization")
        spans = packet.get("source_spans", ())
        if not isinstance(spans, list) or not spans:
            violations.append(prefix + " has no exact source span")
        else:
            for span in spans:
                if (
                    not isinstance(span, (list, tuple))
                    or len(span) != 2
                    or any(type(value) is not int for value in span)
                    or not 0 <= span[0] < span[1] <= len(ir.source_text)
                ):
                    violations.append(prefix + " has an invalid source span")
                    break
                start, end = span
                if not ir.source_text[start:end].strip():
                    violations.append(prefix + " source span has no evidence text")
                    break
    return tuple(dict.fromkeys(violations))


def semantic_revision_scope_violations(before, after, packets):
    """Return unauthorized semantic changes made by a bounded IR revision."""

    before_ir, after_ir = before.faithful_ir, after.faithful_ir
    if before_ir is None or after_ir is None:
        return ("revision must preserve a FaithfulSemanticIR",)
    allowed = {
        str(node_id)
        for packet in packets
        for node_id in packet.get("authorized_node_ids", packet.get("affected_node_ids", ()))
    }
    violations = []
    if before_ir.source_text != after_ir.source_text or before_ir.source_hash != after_ir.source_hash:
        violations.append("source binding changed")
    if before_ir.root_node_id != after_ir.root_node_id:
        violations.append("root_node_id changed outside packet authorization")
    if asdict(before_ir).get("atomic_propositions") != asdict(after_ir).get("atomic_propositions"):
        violations.append("atomic_propositions changed outside packet authorization")
    before_nodes = {node.node_id: asdict(node) for node in before_ir.nodes}
    after_nodes = {node.node_id: asdict(node) for node in after_ir.nodes}
    for node_id in sorted(set(before_nodes) | set(after_nodes)):
        if before_nodes.get(node_id) != after_nodes.get(node_id) and node_id not in allowed:
            violations.append("unauthorized node change: " + node_id)
    if not allowed:
        violations.append("failure packet authorizes no IR nodes")
    return tuple(violations)


__all__ = [
    "build_semantic_failure_packets",
    "semantic_revision_preflight_violations",
    "semantic_revision_scope_violations",
]
