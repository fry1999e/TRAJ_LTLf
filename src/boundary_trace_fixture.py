"""Versioned formula-blind boundary trace fixture loader and binder."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from schemas import (
    SemanticStructure,
    TraceAnnotation,
    TraceObligation,
    TraceObligationPlan,
    TraceQuery,
)
from trace_evidence_policy import is_boundary_obligation
from trace_formatter import format_trace_for_beginner


BOUNDARY_TRACE_FIXTURE_SCHEMA_VERSION = 1


class BoundaryTraceFixtureError(ValueError):
    """A stable, item-scoped fixture error that must never trigger fallback."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BoundaryFixtureRecord:
    requirement_id: str
    requirement_sha256: str
    atomic_propositions: tuple[str, ...]
    trace_id: str
    trace: tuple[tuple[str, ...], ...]
    boundary_role: str
    target_ambiguity_axis_ids: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    source_ir_version: str
    label: str
    confidence: str
    rationale: str
    temporal_simulation_summary: str
    annotator: str
    model: str
    prompt_version: str
    supported_text_spans: tuple[str, ...]


@dataclass(frozen=True)
class BoundaryTraceFixture:
    path: str
    sha256: str
    schema_version: int
    records: tuple[BoundaryFixtureRecord, ...]

    def records_for(self, requirement_id: str) -> tuple[BoundaryFixtureRecord, ...]:
        return tuple(
            record for record in self.records if record.requirement_id == requirement_id
        )


def requirement_sha256(requirement: str) -> str:
    return hashlib.sha256(requirement.strip().encode("utf-8")).hexdigest()


def load_boundary_trace_fixture(path: str | Path) -> BoundaryTraceFixture:
    fixture_path = Path(path)
    if not fixture_path.is_file():
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_NOT_FOUND",
            f"boundary trace fixture does not exist: {fixture_path}",
        )
    raw = fixture_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            "boundary trace fixture must be UTF-8",
        ) from exc
    schema_match = re.search(
        r"^schema_version\s*:\s*(\d+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    schema_version = int(schema_match.group(1)) if schema_match else 0
    if schema_version != BOUNDARY_TRACE_FIXTURE_SCHEMA_VERSION:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            "unsupported boundary fixture schema_version: "
            f"{schema_version}; expected {BOUNDARY_TRACE_FIXTURE_SCHEMA_VERSION}",
        )
    payloads = _json_code_blocks(text)
    if not payloads:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            "boundary fixture contains no JSON code block",
        )
    records: list[BoundaryFixtureRecord] = []
    for payload in payloads:
        for requirement_payload in _requirement_payloads(payload):
            records.extend(_records_from_payload(requirement_payload))
    if not records:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            "boundary fixture contains no records",
        )
    record_keys = [(record.requirement_id, record.trace_id) for record in records]
    duplicate_keys = {
        key for key in record_keys if record_keys.count(key) > 1
    }
    if duplicate_keys:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            "duplicate boundary trace ids: "
            + ", ".join(
                f"{requirement_id}/{trace_id}"
                for requirement_id, trace_id in sorted(duplicate_keys)
            ),
        )
    return BoundaryTraceFixture(
        path=str(fixture_path.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=schema_version,
        records=tuple(records),
    )


def bind_frozen_boundary_evidence(
    *,
    fixture: BoundaryTraceFixture,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure,
    trace_plan: TraceObligationPlan,
    data_split: str,
) -> tuple[list[TraceQuery], list[TraceAnnotation]]:
    """Bind frozen shapes to current boundary obligations without guessing."""

    records = fixture.records_for(requirement_id)
    if not records:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_REQUIREMENT_MISMATCH",
            f"fixture has no section for {requirement_id}",
        )
    expected_hash = requirement_sha256(requirement)
    aps = tuple(
        semantic.faithful_ir.ap_names
        if semantic.faithful_ir is not None
        else semantic.aps
    )
    ap_set = set(aps)
    obligations = tuple(
        item for item in trace_plan.obligations if is_boundary_obligation(item)
    )
    obligation_map = {item.obligation_id: item for item in obligations}
    queries: list[TraceQuery] = []
    annotations: list[TraceAnnotation] = []
    covered_ids: set[str] = set()

    for index, record in enumerate(records, start=1):
        if record.requirement_sha256 != expected_hash:
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_REQUIREMENT_MISMATCH",
                f"{requirement_id}/{record.trace_id} requirement hash mismatch",
            )
        if set(record.atomic_propositions) != ap_set:
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_REQUIREMENT_MISMATCH",
                f"{requirement_id}/{record.trace_id} AP set mismatch",
            )
        unknown = {
            ap for step in record.trace for ap in step if ap not in ap_set
        }
        if unknown:
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_SCHEMA_INVALID",
                f"{record.trace_id} contains unknown APs: {', '.join(sorted(unknown))}",
            )
        matched = _matching_obligations(record, obligations, obligation_map)
        if len(matched) != 1:
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_UNBOUND",
                f"{record.trace_id} matched {len(matched)} current boundary obligations",
            )
        obligation = matched[0]
        if obligation.obligation_id in covered_ids:
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_UNBOUND",
                f"{requirement_id} binds boundary obligation "
                f"{obligation.obligation_id} more than once",
            )
        if len(record.trace) < obligation.minimum_horizon:
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_UNBOUND",
                f"{record.trace_id} is shorter than obligation "
                f"{obligation.obligation_id} minimum horizon",
            )
        query_id = (
            record.trace_id
            if record.trace_id.startswith("QUERY-")
            else f"QUERY-FROZEN-BOUNDARY-{index:03d}"
        )
        trace = [set(step) for step in record.trace]
        axes = record.target_ambiguity_axis_ids or obligation.ambiguity_axis_ids
        target = record.boundary_role or obligation.boundary_case or "boundary"
        queries.append(
            TraceQuery(
                query_id=query_id,
                role="boundary",
                target_ambiguity=target,
                constraints=(
                    *obligation.construction_constraints,
                    "source=frozen_boundary_fixture",
                    f"fixture_sha256={fixture.sha256}",
                ),
                trace=trace,
                reason="Frozen formula-blind boundary evidence supplied by run policy.",
                beginner_view=format_trace_for_beginner(trace),
                data_split=data_split,
                label_source="frozen_independent_judge",
                formula_source="none",
                formula_visibility=(),
                ir_node_ids=obligation.ir_node_ids,
                obligation_ids=(obligation.obligation_id,),
                ir_version=trace_plan.ir_version,
                target_ambiguity_axis_ids=axes,
                semantic_parameter_paths=obligation.semantic_parameter_paths,
                generation_source="frozen_boundary_fixture",
            )
        )
        annotations.append(
            TraceAnnotation(
                query_id=query_id,
                label=record.label,
                confidence=record.confidence,
                rationale=record.rationale,
                label_source="frozen_independent_judge",
                annotator=record.annotator,
                model=record.model,
                prompt_version=record.prompt_version,
                temporal_simulation_summary=record.temporal_simulation_summary,
                supported_text_spans=record.supported_text_spans,
            )
        )
        covered_ids.add(obligation.obligation_id)

    required_ids = {
        item.obligation_id
        for item in obligations
        if item.evidence_mode not in {"diagnostic_only", "operator_resolution"}
    }
    missing = required_ids - covered_ids
    if missing:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_COVERAGE_INCOMPLETE",
            f"{requirement_id} fixture misses boundary obligations: "
            + ", ".join(sorted(missing)),
        )
    return queries, annotations


def _matching_obligations(
    record: BoundaryFixtureRecord,
    obligations: tuple[TraceObligation, ...],
    obligation_map: dict[str, TraceObligation],
) -> list[TraceObligation]:
    if record.obligation_ids:
        return [
            obligation_map[item]
            for item in record.obligation_ids
            if item in obligation_map
        ]
    selector = record.boundary_role.strip().lower()
    axes = {item.lower() for item in record.target_ambiguity_axis_ids}
    matches = []
    for obligation in obligations:
        confusion = str(
            dict(obligation.semantic_parameters).get("confusion_id", "")
        ).lower()
        candidates = {
            obligation.required_trace_role.lower(),
            obligation.boundary_case.lower(),
            confusion,
            *(item.lower() for item in obligation.ambiguity_axis_ids),
        }
        candidates.discard("")
        if selector and selector in candidates or axes.intersection(candidates):
            matches.append(obligation)
    if not matches and len(obligations) == 1 and selector == "boundary":
        return [obligations[0]]
    return matches


def _json_code_blocks(text: str) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = []
    for match in re.finditer(
        r"```json\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_SCHEMA_INVALID",
                f"invalid JSON block: {exc}",
            ) from exc
        if not isinstance(payload, Mapping):
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_SCHEMA_INVALID",
                "each boundary fixture JSON block must be an object",
            )
        payloads.append(payload)
    return payloads


def _requirement_payloads(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("requirements")
    if raw is None:
        return [payload]
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            "requirements must be a list of objects",
        )
    return list(raw)


def _records_from_payload(payload: Mapping[str, Any]) -> list[BoundaryFixtureRecord]:
    requirement_id = _required_text(payload, "requirement_id")
    requirement_hash = _required_text(payload, "requirement_sha256")
    aps = _required_string_tuple(payload, "atomic_propositions")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            f"{requirement_id} records must be a non-empty list",
        )
    records: list[BoundaryFixtureRecord] = []
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_SCHEMA_INVALID",
                f"{requirement_id} record must be an object",
            )
        kind = str(raw.get("kind", "boundary")).strip().lower()
        if kind != "boundary":
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_SCHEMA_INVALID",
                f"{requirement_id} fixture may contain boundary records only",
            )
        annotation = raw.get("annotation")
        if not isinstance(annotation, Mapping):
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_SCHEMA_INVALID",
                f"{requirement_id} record annotation must be an object",
            )
        if str(annotation.get("review_basis", "NL_only")) != "NL_only":
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_FORMULA_LEAKAGE",
                "frozen boundary annotations must use review_basis=NL_only",
            )
        label_source = str(
            annotation.get("label_source", "frozen_independent_judge")
        )
        if label_source not in {
            "independent_judge",
            "frozen_independent_judge",
        }:
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_SCHEMA_INVALID",
                "frozen boundary labels must come from an independent judge",
            )
        if str(annotation.get("formula_source", "none")) != "none" or annotation.get(
            "formula_visibility", []
        ):
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_FORMULA_LEAKAGE",
                "frozen boundary annotations must remain formula blind",
            )
        label = str(annotation.get("label", "")).upper()
        if label not in {"SAT", "UNSAT"}:
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_SCHEMA_INVALID",
                "frozen boundary annotation label must be SAT or UNSAT",
            )
        confidence = str(annotation.get("confidence", "medium")).lower()
        if confidence not in {"high", "medium", "low"}:
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_SCHEMA_INVALID",
                "unsupported frozen boundary annotation confidence",
            )
        rationale = str(annotation.get("rationale", "")).strip()
        if not rationale:
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_SCHEMA_INVALID",
                "frozen boundary annotation rationale must not be empty",
            )
        records.append(
            BoundaryFixtureRecord(
                requirement_id=requirement_id,
                requirement_sha256=requirement_hash,
                atomic_propositions=aps,
                trace_id=_required_text(raw, "trace_id"),
                trace=_trace_steps(raw.get("trace"), requirement_id),
                boundary_role=str(raw.get("boundary_role", "boundary")).strip(),
                target_ambiguity_axis_ids=_string_tuple(
                    raw.get("target_ambiguity_axis_ids", ())
                ),
                obligation_ids=_string_tuple(raw.get("obligation_ids", ())),
                source_ir_version=str(raw.get("ir_version", "")),
                label=label,
                confidence=confidence,
                rationale=rationale,
                temporal_simulation_summary=str(
                    annotation.get("temporal_simulation_summary", rationale)
                ).strip(),
                annotator=str(annotation.get("annotator", "frozen_fixture")),
                model=str(annotation.get("model", "")),
                prompt_version=str(annotation.get("prompt_version", "")),
                supported_text_spans=_string_tuple(
                    annotation.get("supported_text_spans", ())
                ),
            )
        )
    return records


def _trace_steps(value: Any, requirement_id: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list) or not value:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            f"{requirement_id} trace must be a non-empty list",
        )
    steps: list[tuple[str, ...]] = []
    for step in value:
        if not isinstance(step, list) or any(not isinstance(ap, str) for ap in step):
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_SCHEMA_INVALID",
                f"{requirement_id} trace steps must be string lists",
            )
        steps.append(tuple(dict.fromkeys(ap.strip() for ap in step if ap.strip())))
    return tuple(steps)


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            f"boundary fixture field {key} is required",
        )
    return value


def _required_string_tuple(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = _string_tuple(payload.get(key, ()))
    if not values:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            f"boundary fixture field {key} must be a non-empty string list",
        )
    return values


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def export_boundary_trace_fixture(
    records_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Export formula-blind boundary annotation records from audit JSON files."""

    root = Path(records_dir)
    if not root.is_dir():
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_NOT_FOUND",
            f"review records directory does not exist: {root}",
        )
    grouped: dict[str, dict[str, Any]] = {}
    seen_ids: set[tuple[str, str]] = set()
    seen_bindings: set[tuple[str, tuple[str, ...]]] = set()
    audit_paths = sorted(
        root.rglob("*.json"),
        key=lambda item: (item.stat().st_mtime_ns, str(item)),
        reverse=True,
    )
    for path in audit_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        requirement_id = _audit_requirement_id(payload)
        requirement = str(payload.get("requirement", "")).strip()
        annotations = payload.get("annotations", payload.get("queries"))
        brief = payload.get("adjudication_brief")
        if (
            not requirement_id
            or not requirement
            or not isinstance(annotations, list)
            or not isinstance(brief, Mapping)
        ):
            continue
        aps = tuple(
            str(item[0]).strip()
            for item in brief.get("ap_definitions", ())
            if isinstance(item, (list, tuple)) and item
        )
        if not aps:
            continue
        requirement_hash = requirement_sha256(requirement)
        target = grouped.get(requirement_id)
        if target is None:
            target = {
                "requirement_id": requirement_id,
                "requirement_sha256": requirement_hash,
                "atomic_propositions": list(aps),
                "records": [],
            }
            grouped[requirement_id] = target
        elif (
            target["requirement_sha256"] != requirement_hash
            or set(target["atomic_propositions"]) != set(aps)
        ):
            raise BoundaryTraceFixtureError(
                "BOUNDARY_FIXTURE_REQUIREMENT_MISMATCH",
                f"review records reuse {requirement_id} for different text or APs",
            )
        for item in annotations:
            if not isinstance(item, Mapping):
                continue
            query = item.get("query")
            annotation = item.get("annotation")
            if not isinstance(query, Mapping) or not isinstance(annotation, Mapping):
                continue
            role = str(query.get("role", "")).lower()
            ambiguity = str(query.get("target_ambiguity", "")).lower()
            if "boundary" not in role and "boundary" not in ambiguity:
                continue
            trace_id = str(query.get("query_id", "")).strip()
            trace_key = (requirement_id, trace_id)
            obligation_ids = tuple(
                sorted(_string_tuple(query.get("obligation_ids", ())))
            )
            selector_key = obligation_ids or (
                "selector:"
                + str(query.get("target_ambiguity") or role).strip().lower(),
                *_string_tuple(query.get("target_ambiguity_axis_ids", ())),
            )
            binding_key = (requirement_id, selector_key)
            label = str(annotation.get("label", "")).upper()
            label_source = str(annotation.get("label_source", ""))
            if (
                not trace_id
                or trace_key in seen_ids
                or binding_key in seen_bindings
                or label not in {"SAT", "UNSAT"}
                or label_source
                not in {"independent_judge", "frozen_independent_judge"}
                or str(query.get("formula_source", "none")) != "none"
                or bool(query.get("formula_visibility", []))
            ):
                continue
            rationale = str(annotation.get("rationale", "")).strip()
            if not rationale:
                continue
            seen_ids.add(trace_key)
            seen_bindings.add(binding_key)
            target["records"].append(
                {
                    "trace_id": trace_id,
                    "kind": "boundary",
                    "trace": query.get("trace", []),
                    "boundary_role": query.get("target_ambiguity") or role,
                    "target_ambiguity_axis_ids": query.get(
                        "target_ambiguity_axis_ids", []
                    ),
                    "obligation_ids": list(obligation_ids),
                    "ir_version": query.get("ir_version", ""),
                    "annotation": {
                        "label": label,
                        "confidence": annotation.get("confidence", "medium"),
                        "rationale": rationale,
                        "temporal_simulation_summary": annotation.get(
                            "temporal_simulation_summary", rationale
                        ),
                        "label_source": "frozen_independent_judge",
                        "review_basis": "NL_only",
                        "formula_source": "none",
                        "formula_visibility": [],
                        "annotator": annotation.get("annotator", ""),
                        "model": annotation.get("model", ""),
                        "prompt_version": annotation.get("prompt_version", ""),
                        "supported_text_spans": annotation.get(
                            "supported_text_spans", []
                        ),
                    },
                }
            )
    requirements = [
        payload for payload in grouped.values() if payload["records"]
    ]
    if not requirements:
        raise BoundaryTraceFixtureError(
            "BOUNDARY_FIXTURE_SCHEMA_INVALID",
            "no formula-blind boundary annotations were found in review records",
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = (
        "# Traj-LTLf Frozen Boundary Evidence\n"
        f"schema_version: {BOUNDARY_TRACE_FIXTURE_SCHEMA_VERSION}\n\n"
        "```json\n"
        + json.dumps(
            {"requirements": requirements},
            ensure_ascii=False,
            indent=2,
        )
        + "\n```\n"
    )
    output.write_text(document, encoding="utf-8")
    return output


def _audit_requirement_id(payload: Mapping[str, Any]) -> str:
    """Recover the stable batch id, including from older single-run records."""

    direct = str(payload.get("requirement_id", "")).strip()
    if direct:
        return direct
    semantic = payload.get("semantic")
    if not isinstance(semantic, Mapping):
        return ""
    for artifact_name in ("surface_segmentation", "plain_reading_review"):
        artifact = semantic.get(artifact_name)
        if isinstance(artifact, Mapping):
            recovered = str(artifact.get("requirement_id", "")).strip()
            if recovered:
                return recovered
    return ""


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or export Traj-LTLf frozen boundary evidence"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("path")
    export = subparsers.add_parser("export")
    export.add_argument("--records", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            fixture = load_boundary_trace_fixture(args.path)
            print(
                f"valid schema={fixture.schema_version} records={len(fixture.records)} "
                f"sha256={fixture.sha256}"
            )
        else:
            output = export_boundary_trace_fixture(args.records, args.output)
            fixture = load_boundary_trace_fixture(output)
            print(
                f"exported {output} records={len(fixture.records)} "
                f"sha256={fixture.sha256}"
            )
    except BoundaryTraceFixtureError as exc:
        raise SystemExit(f"{exc.code}: {exc}") from exc


__all__ = [
    "BOUNDARY_TRACE_FIXTURE_SCHEMA_VERSION",
    "BoundaryTraceFixture",
    "BoundaryTraceFixtureError",
    "bind_frozen_boundary_evidence",
    "export_boundary_trace_fixture",
    "load_boundary_trace_fixture",
    "requirement_sha256",
]


if __name__ == "__main__":
    _main()
