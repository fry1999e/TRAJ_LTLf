"""Blind per-trace annotation protocol for finite-trace queries."""

from __future__ import annotations
from execution_observer import observed_stage, emit

from dataclasses import asdict, replace
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from datetime import datetime

from agents.base_agent import JSONLLMClient
from schemas import (
    NLTemporalSimulationReview,
    SemanticStructure,
    TraceAdjudicationBrief,
    TraceAnnotation,
    TraceQuery,
)
from semantic_adjudication import build_trace_adjudication_brief


ANNOTATION_RATIONALE_START = "<!-- rationale:start -->"
ANNOTATION_RATIONALE_END = "<!-- rationale:end -->"


def trace_query_fingerprint(query: TraceQuery) -> str:
    """Return the stable identity used by frozen train-only trace gold."""

    payload = {
        "query_id": query.query_id,
        "trace": [sorted(step) for step in query.trace],
        "obligation_ids": list(query.obligation_ids),
        "ir_version": query.ir_version,
    }
    if query.purpose != "semantic_validation":
        payload.update({
            "purpose": query.purpose,
            "parent_candidate_set_id": query.parent_candidate_set_id,
            "parent_hole_plan_version": query.parent_hole_plan_version,
        })
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frozen_train_gold_annotations(
    queries: list[TraceQuery],
    reviews: tuple[NLTemporalSimulationReview, ...],
) -> list[TraceAnnotation]:
    """Apply frozen train-only NL reviews without formula or heuristic access.

    Gold is accepted only when the query id, trace valuation, obligation binding,
    IR version, and stable fingerprint all match.  A missing or stale review is a
    contract error; this function deliberately has no fallback judge.
    """

    by_trace_id: dict[str, NLTemporalSimulationReview] = {}
    duplicates: set[str] = set()
    for review in reviews:
        if review.trace_id in by_trace_id:
            duplicates.add(review.trace_id)
        by_trace_id[review.trace_id] = review
    if duplicates:
        raise ValueError(
            "Frozen NLTemporalSimulationReview contains duplicate trace ids: "
            + ", ".join(sorted(duplicates))
        )

    annotations: list[TraceAnnotation] = []
    errors: list[str] = []
    for query in queries:
        review = by_trace_id.get(query.query_id)
        if review is None:
            errors.append(f"{query.query_id}: missing frozen review")
            continue
        expected_steps = tuple(tuple(sorted(step)) for step in query.trace)
        if review.data_split != "train":
            errors.append(f"{query.query_id}: frozen review is not train-only")
        if review.label_source != "warmup_train_gold":
            errors.append(f"{query.query_id}: invalid frozen label source")
        if review.formula_source != "none" or review.formula_visibility:
            errors.append(f"{query.query_id}: formula visibility is not sealed")
        if review.trace_steps != expected_steps:
            errors.append(f"{query.query_id}: trace valuation drift")
        if review.obligation_ids != query.obligation_ids:
            errors.append(f"{query.query_id}: obligation binding drift")
        if review.ir_version != query.ir_version:
            errors.append(f"{query.query_id}: IR version drift")
        fingerprint = trace_query_fingerprint(query)
        if review.trace_fingerprint != fingerprint:
            errors.append(f"{query.query_id}: trace fingerprint drift")
        label = _label_from_nl_verdict(review.nl_verdict)
        if not label:
            errors.append(
                f"{query.query_id}: unsupported frozen verdict {review.nl_verdict!r}"
            )
            continue
        annotations.append(
            TraceAnnotation(
                query_id=query.query_id,
                label=label,
                confidence=review.confidence,
                rationale=review.reason or review.temporal_simulation_summary,
                label_source="warmup_train_gold",
                annotator=review.reviewer_id,
                model="frozen-train-only-nl-temporal-review",
                prompt_version="warmup-nl-temporal-review-v1",
                temporal_simulation_summary=review.temporal_simulation_summary,
                supported_text_spans=review.supported_text_spans,
                purpose=query.purpose,
            )
        )
    if errors:
        raise ValueError("Frozen warmup trace gold mismatch: " + "; ".join(errors))
    return annotations


def _label_from_nl_verdict(verdict: str) -> str:
    normalized = verdict.strip().lower()
    if normalized in {"positive_satisfying", "boundary_satisfying", "sat"}:
        return "SAT"
    if normalized in {"negative_violating", "boundary_violating", "unsat"}:
        return "UNSAT"
    if normalized == "unclear":
        return "UNCLEAR"
    if normalized == "infeasible":
        return "INFEASIBLE"
    return ""


@observed_stage("trace_annotation")
def independent_judge_annotations(
    requirement: str,
    semantic: SemanticStructure,
    queries: list[TraceQuery],
    judge_client: JSONLLMClient | None = None,
    adjudication_brief: TraceAdjudicationBrief | None = None,
) -> list[TraceAnnotation]:
    """
    Independent weak-supervision judge using only NL, APs, and finite traces.

    Only the deterministic branch remains for local tests. The released CLI
    obtains real annotations from the user with manual_trace_annotations.
    """

    if judge_client is not None:
        raise ValueError("Model trace reviewers are disabled; use manual_trace_annotations")
    if not queries:
        return []
    emit("before_trace_annotation", queries=queries)

    representative_queries, representative_by_query_id = (
        _deduplicate_queries_for_adjudication(queries)
    )

    brief = adjudication_brief or build_trace_adjudication_brief(
        requirement,
        semantic,
    )

    if judge_client is not None:
        representative_annotations = _llm_independent_judge_annotations(
            requirement=requirement,
            semantic=semantic,
            queries=representative_queries,
            judge_client=judge_client,
            adjudication_brief=brief,
        )
    else:
        representative_annotations = [
            TraceAnnotation(
                query_id=query.query_id,
                label=_judge_label(requirement, semantic, query),
                confidence=_judge_confidence(requirement, query),
                rationale=(
                    "Independent NL/trace judgment based on the reviewed semantic "
                    f"roles and query condition: {', '.join(query.constraints)}."
                ),
                label_source="independent_judge",
                model="deterministic-nl-judge",
                prompt_version="1",
                temporal_simulation_summary=(
                    "Deterministic NL judge simulated the finite trace against the "
                    "natural-language requirement and reviewed AP occurrences over time."
                ),
                purpose=query.purpose,
            )
            for query in representative_queries
        ]
    by_representative = {
        annotation.query_id: annotation
        for annotation in representative_annotations
    }
    return [
        replace(
            by_representative[representative_by_query_id[query.query_id]],
            query_id=query.query_id,
        )
        for query in queries
    ]


def _deduplicate_queries_for_adjudication(
    queries: list[TraceQuery],
) -> tuple[list[TraceQuery], dict[str, str]]:
    """Ask the Judge once for an identical finite trace in one semantic context."""

    representative_by_shape: dict[
        tuple[str, tuple[tuple[str, ...], ...]], TraceQuery
    ] = {}
    representative_by_query_id: dict[str, str] = {}
    representatives: list[TraceQuery] = []
    for query in queries:
        shape = tuple(tuple(sorted(step)) for step in query.trace)
        purpose_shape = (query.purpose, shape)
        representative = representative_by_shape.get(purpose_shape)
        if representative is None:
            representative = query
            representative_by_shape[purpose_shape] = query
            representatives.append(query)
        representative_by_query_id[query.query_id] = representative.query_id
    return representatives, representative_by_query_id


def _llm_independent_judge_annotations(
    *,
    requirement: str,
    semantic: SemanticStructure,
    queries: list[TraceQuery],
    judge_client: JSONLLMClient,
    adjudication_brief: TraceAdjudicationBrief,
) -> list[TraceAnnotation]:
    payload = {
        "task": "finite_trace_natural_language_adjudication",
        "adjudication_brief": {
            "brief_id": adjudication_brief.brief_id,
            "semantic_context_id": adjudication_brief.semantic_context_id,
            "natural_language_requirement": adjudication_brief.source_text,
            "trace_domain": adjudication_brief.trace_domain,
            "atomic_propositions": [
                {"name": name, "definition": definition}
                for name, definition in adjudication_brief.ap_definitions
            ],
            "reviewed_plain_language_decisions": list(
                adjudication_brief.reviewed_plain_language_decisions
            ),
            "open_plain_language_axes": list(
                adjudication_brief.open_plain_language_axes
            ),
            "global_trace_semantics": list(
                adjudication_brief.global_trace_semantics
            ),
        },
        "finite_trace_semantics": (
            "Each trace is finite; the last listed state is the end of the run."
        ),
        "queries": [
            {
                "query_id": query.query_id,
                "trace": [sorted(step) for step in query.trace],
                "beginner_view": query.beginner_view,
            }
            for query in queries
        ],
        "required_output": {
            "semantic_status": "pass or fail",
            "semantic_feedback": "short text",
            "trace_status": "pass or fail",
            "trace_issue_type": "content, length, or coverage",
            "annotations": [
                {
                    "query_id": "QUERY-...",
                    "label": "SAT, UNSAT, UNCLEAR, or INFEASIBLE",
                    "confidence": "high, medium, or low",
                    "rationale": "short natural-language reason",
                    "temporal_simulation_summary": "brief stepwise NL simulation",
                    "supported_text_spans": ["optional exact NL phrase"],
                    "uncertainty_kind": (
                        "none, annotation_uncertainty, query_infeasible, "
                        "reviewed_decision_conflict, or new_semantic_axis"
                    ),
                    "candidate_readings": [
                        "required only for a semantic conflict; give at least two"
                    ],
                    "challenged_axis_ids": ["optional reviewed/open axis ids"],
                    "affected_parameter_paths": [
                        "optional semantic parameter paths from the brief"
                    ],
                }
            ],
        },
    }
    result = judge_client.complete_json(
        _INDEPENDENT_JUDGE_SYSTEM_PROMPT,
        payload,
    )
    raw_annotations = _extract_judge_annotation_items(result)

    query_ids = {query.query_id for query in queries}
    query_by_id = {query.query_id: query for query in queries}
    annotations: list[TraceAnnotation] = []
    for item in raw_annotations:
        if not isinstance(item, dict):
            continue
        query_id = _read_annotation_query_id(item)
        if query_id not in query_ids:
            continue
        label = _normalize_annotation_label(item)
        confidence = _normalize_annotation_confidence(item.get("confidence", "medium"))
        rationale = _read_annotation_rationale(item)
        if not rationale:
            rationale = "Independent Judge annotated this finite trace from NL only."
        annotations.append(
            TraceAnnotation(
                query_id=query_id,
                label=label,
                confidence=confidence,
                rationale=rationale,
                label_source="independent_judge",
                model=str(getattr(judge_client, "model", "")),
                prompt_version=_INDEPENDENT_JUDGE_PROMPT_VERSION,
                temporal_simulation_summary=str(
                    _read_first_present(
                        item,
                        (
                            "temporal_simulation_summary",
                            "simulation_summary",
                            "trace_simulation",
                        ),
                    )
                    or ""
                ).strip(),
                supported_text_spans=tuple(
                    str(span)
                    for span in item.get("supported_text_spans", ())
                    if span
                )
                if isinstance(item.get("supported_text_spans", ()), list)
                else (),
                uncertainty_kind=_normalize_uncertainty_kind(
                    item.get("uncertainty_kind", "none"),
                    label,
                ),
                candidate_readings=_string_tuple(
                    item.get("candidate_readings", ())
                ),
                challenged_axis_ids=_string_tuple(
                    item.get("challenged_axis_ids", ())
                ),
                affected_parameter_paths=_string_tuple(
                    item.get("affected_parameter_paths", ())
                ),
                purpose=query_by_id[query_id].purpose,
            )
        )

    missing = sorted(query_ids - {annotation.query_id for annotation in annotations})
    if missing:
        raise ValueError("Judge response missed query ids: " + ", ".join(missing))
    return annotations


def _extract_judge_annotation_items(result: dict) -> list[dict]:
    if not isinstance(result, dict):
        raise ValueError(
            "Judge response must be a JSON object; got "
            + type(result).__name__
            + "."
        )

    direct = _coerce_annotation_container(result.get("annotations"))
    if direct is not None:
        return direct

    for key in (
        "trace_annotations",
        "trace_annotation",
        "query_annotations",
        "per_trace_annotations",
        "trace_judgments",
        "trace_judgements",
        "trace_results",
        "judgments",
        "judgements",
        "labels",
        "results",
    ):
        extracted = _coerce_annotation_container(result.get(key))
        if extracted is not None:
            return extracted

    for key in ("trace_review", "review", "adjudication", "answer", "output", "result"):
        nested = result.get(key)
        if isinstance(nested, dict):
            try:
                return _extract_judge_annotation_items(nested)
            except ValueError:
                pass

    mapping_items = _annotation_items_from_query_mapping(result)
    if mapping_items:
        return mapping_items

    if _looks_like_service_config_response(result):
        raise ValueError(
            "Judge response looks like a service/web configuration endpoint, "
            "not a chat-completions response. Check JUDGE_API_BASE_URL; "
            "OpenAI-compatible providers usually require a /v1 suffix. "
            "Top-level keys: " + ", ".join(sorted(str(key) for key in result.keys()))
        )

    raise ValueError(
        "Judge response must contain an annotations list. "
        "Top-level keys: " + ", ".join(sorted(str(key) for key in result.keys()))
    )


def _coerce_annotation_container(value: Any) -> list[dict] | None:
    if isinstance(value, list):
        if not value:
            return []
        if any(_looks_like_annotation_item(item) for item in value):
            return [item for item in value if isinstance(item, dict)]
        return None

    if isinstance(value, dict):
        for key in ("annotations", "items", "records", "results", "labels"):
            nested = _coerce_annotation_container(value.get(key))
            if nested is not None:
                return nested
        mapped = _annotation_items_from_query_mapping(value)
        if mapped:
            return mapped
        if _looks_like_annotation_item(value):
            return [dict(value)]

    return None


def _annotation_items_from_query_mapping(value: Mapping) -> list[dict]:
    items: list[dict] = []
    for raw_query_id, raw_annotation in value.items():
        query_id = str(raw_query_id).strip()
        if not query_id.startswith("QUERY-"):
            continue
        if isinstance(raw_annotation, dict):
            item = dict(raw_annotation)
            item.setdefault("query_id", query_id)
            items.append(item)
        elif isinstance(raw_annotation, str):
            items.append({"query_id": query_id, "label": raw_annotation})
    return items


def _looks_like_service_config_response(value: Mapping) -> bool:
    keys = {str(key) for key in value.keys()}
    service_config_markers = {
        "site_name",
        "site_subtitle",
        "registration_enabled",
        "api_base_url",
        "turnstile_enabled",
        "server_timezone",
        "payment_enabled",
        "custom_endpoints",
    }
    return len(keys & service_config_markers) >= 3


def _looks_like_annotation_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return bool(
        _read_annotation_query_id(item)
        and _read_first_present(
            item,
            (
                "label",
                "judgment",
                "judgement",
                "verdict",
                "truth",
                "satisfaction",
                "satisfies",
                "satisfied",
                "answer",
            ),
        )
    )


def _read_annotation_query_id(item: Mapping) -> str:
    return str(
        _read_first_present(
            item,
            (
                "query_id",
                "queryId",
                "queryID",
                "id",
                "trace_id",
                "traceId",
                "trace_query_id",
                "traceQueryId",
            ),
        )
        or ""
    ).strip()


def _normalize_annotation_label(item: Mapping) -> str:
    raw_label = _read_first_present(
        item,
        (
            "label",
            "judgment",
            "judgement",
            "verdict",
            "truth",
            "satisfaction",
            "satisfies",
            "satisfied",
            "answer",
        ),
    )
    if isinstance(raw_label, bool):
        return "SAT" if raw_label else "UNSAT"

    text = str(raw_label or "").strip().lower().replace("_", " ")
    normalized = re.sub(r"\s+", " ", text)
    if normalized in {"sat", "satisfied", "satisfies", "true", "yes"}:
        return "SAT"
    if normalized in {
        "unsat",
        "not satisfied",
        "violated",
        "violates",
        "violate",
        "false",
        "no",
    }:
        return "UNSAT"
    if normalized in {"unclear", "unknown", "ambiguous", "cannot determine"}:
        return "UNCLEAR"
    if normalized in {"infeasible", "impossible", "not feasible"}:
        return "INFEASIBLE"
    return str(raw_label or "").strip().upper()


def _normalize_annotation_confidence(value: Any) -> str:
    if isinstance(value, (int, float)):
        if value >= 0.75:
            return "high"
        if value >= 0.4:
            return "medium"
        return "low"

    text = str(value or "medium").strip().lower()
    if text in {"high", "medium", "low"}:
        return text
    try:
        return _normalize_annotation_confidence(float(text))
    except ValueError:
        return "medium"


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _normalize_uncertainty_kind(value: Any, label: str) -> str:
    normalized = str(value or "none").strip().lower().replace("-", "_")
    allowed = {
        "none",
        "annotation_uncertainty",
        "query_infeasible",
        "reviewed_decision_conflict",
        "new_semantic_axis",
    }
    if normalized in allowed:
        return normalized
    if label == "INFEASIBLE":
        return "query_infeasible"
    if label == "UNCLEAR":
        return "annotation_uncertainty"
    return "none"


def _read_annotation_rationale(item: Mapping) -> str:
    return str(
        _read_first_present(
            item,
            (
                "rationale",
                "reason",
                "explanation",
                "justification",
                "evidence",
            ),
        )
        or ""
    ).strip()


def _read_first_present(item: Mapping, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key not in item:
            continue
        value = item[key]
        if value is None or value == "":
            continue
        return value
    return None


_INDEPENDENT_JUDGE_PROMPT_VERSION = "independent-trace-judge-v3"
_INDEPENDENT_JUDGE_SYSTEM_PROMPT = """
You are an independent expert reviewer for finite-trace semantics.
Use only the supplied adjudication brief and the listed finite traces. The
brief contains binding reviewed decisions that must be treated as authoritative,
plus any explicitly open semantic axes. Decide whether each trace satisfies
the requirement under those decisions.
Return one JSON object with semantic_status, semantic_feedback, trace_status,
trace_issue_type, and a top-level annotations array. Do not nest annotations,
rename annotations to labels/results, or omit it when traces are present.
For each annotation, use query_id exactly as provided; label SAT, UNSAT,
UNCLEAR, or INFEASIBLE; confidence high, medium, or low; a short rationale;
and, when possible, temporal_simulation_summary explaining the trace step by
step against the natural-language requirement.
Ordinary uncertainty about one trace is annotation_uncertainty. An impossible
or unusable trace is query_infeasible. Report reviewed_decision_conflict or
new_semantic_axis only when the source text genuinely supports at least two
specific readings; then provide candidate_readings, relevant source spans, and
the affected axis ids or parameter paths. Never turn a generic low-confidence
label into a semantic conflict.
Do not introduce temporal-logic syntax or assume any hidden answer key.
Do not create, infer, optimize, compare, or use any LTL/LTLf formula,
candidate formula, reference formula, or expected final candidate.
Do not design a formula for the listed traces, and do not redesign traces for
any formula. Treat every listed trace as fixed evidence and adjudicate it
objectively against the natural-language requirement only.
""".strip()


def manual_trace_annotations(
    *, requirement: str, semantic: SemanticStructure, queries: list[TraceQuery],
    round_index: int, records_dir: str,
    adjudication_brief: TraceAdjudicationBrief | None = None,
    llm_client: JSONLLMClient | None = None,
) -> list[TraceAnnotation]:
    """Collect simple human verdicts and let the main model fill metadata."""
    from nonexpert_review import review_traces_with_simple_choices
    return review_traces_with_simple_choices(
        requirement=requirement, semantic=semantic, queries=queries,
        round_index=round_index, records_dir=records_dir,
        adjudication_brief=adjudication_brief, llm_client=llm_client,
    )


def write_trace_annotation_review_markdown(
    *,
    requirement: str,
    semantic: SemanticStructure,
    queries: list[TraceQuery],
    path: str | Path,
    adjudication_brief: TraceAdjudicationBrief | None = None,
) -> Path:
    """Write a blind review file that contains no formula or suggested truth."""

    review_path = Path(path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    brief = adjudication_brief or build_trace_adjudication_brief(
        requirement,
        semantic,
    )
    lines = [
        "# Traj-LTLf Finite Trace Annotation",
        "",
        "## Natural Language Requirement",
        "```text",
        brief.source_text,
        "```",
        "",
        "## Reviewed Semantic Brief",
        "```text",
        f"brief_id: {brief.brief_id}",
        "atomic_propositions: "
        + ", ".join(name for name, _definition in brief.ap_definitions),
        *[
            f"reviewed_plain_language_decision: {item}"
            for item in brief.reviewed_plain_language_decisions
        ],
        *[f"open_plain_language_axis: {item}" for item in brief.open_plain_language_axes],
        *[f"global_trace_semantics: {item}" for item in brief.global_trace_semantics],
        "finite_trace_semantics: the last listed state is the end of the trace",
        "```",
        "",
        "Label each feasible trace as SAT, UNSAT, UNCLEAR, or INFEASIBLE.",
        "",
    ]
    for query in queries:
        lines.extend(
            [
                f"### {query.query_id}",
                "",
                "```text",
                query.beginner_view,
                "```",
                "",
                f"purpose: {query.purpose}",
                "label: pending",
                "confidence: medium",
                "uncertainty_kind: none",
                "candidate_readings: []",
                "challenged_axis_ids: []",
                "affected_parameter_paths: []",
                ANNOTATION_RATIONALE_START,
                "",
                ANNOTATION_RATIONALE_END,
                "",
            ]
        )
    review_path.write_text("\n".join(lines), encoding="utf-8")
    return review_path


def parse_trace_annotation_review_markdown(
    path: str | Path,
) -> list[TraceAnnotation]:
    """Parse independent per-query labels from the review Markdown."""

    text = Path(path).read_text(encoding="utf-8")
    matches = list(
        re.finditer(
            r"^###\s+((?:QUERY|IRQ|OPQ)-[A-Za-z0-9-]+)\s*$",
            text,
            flags=re.MULTILINE,
        )
    )
    annotations: list[TraceAnnotation] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end]
        label = _read_field(section, "label").upper()
        confidence = _read_field(section, "confidence").lower()
        purpose = _read_field(section, "purpose") or "semantic_validation"
        uncertainty_kind = _normalize_uncertainty_kind(
            _read_field(section, "uncertainty_kind"),
            label,
        )
        rationale = _extract_between(
            section,
            ANNOTATION_RATIONALE_START,
            ANNOTATION_RATIONALE_END,
        )
        if label == "PENDING":
            continue
        annotations.append(
            TraceAnnotation(
                query_id=match.group(1),
                label=label,
                confidence=confidence,
                rationale=rationale,
                label_source="human",
                uncertainty_kind=uncertainty_kind,
                candidate_readings=_read_list_field(
                    section,
                    "candidate_readings",
                ),
                challenged_axis_ids=_read_list_field(
                    section,
                    "challenged_axis_ids",
                ),
                affected_parameter_paths=_read_list_field(
                    section,
                    "affected_parameter_paths",
                ),
                purpose=purpose,
            )
        )
    return annotations


def validate_trace_annotations(
    queries: list[TraceQuery],
    annotations: list[TraceAnnotation],
) -> list[str]:
    """Validate completeness without changing any adjudicated label."""

    errors: list[str] = []
    query_ids = {query.query_id for query in queries}
    annotation_ids = [annotation.query_id for annotation in annotations]
    duplicates = {
        query_id
        for query_id in annotation_ids
        if annotation_ids.count(query_id) > 1
    }
    if duplicates:
        errors.append("Duplicate annotations: " + ", ".join(sorted(duplicates)))
    unknown = set(annotation_ids) - query_ids
    if unknown:
        errors.append("Unknown query ids: " + ", ".join(sorted(unknown)))
    missing = query_ids - set(annotation_ids)
    if missing:
        errors.append("Pending queries: " + ", ".join(sorted(missing)))
    purpose_by_id = {query.query_id: query.purpose for query in queries}
    purpose_mismatches = tuple(
        annotation.query_id
        for annotation in annotations
        if annotation.query_id in purpose_by_id
        and annotation.purpose != purpose_by_id[annotation.query_id]
    )
    if purpose_mismatches:
        errors.append(
            "Annotation purpose mismatch: "
            + ", ".join(sorted(purpose_mismatches))
        )
    return errors


def annotation_record(
    query: TraceQuery,
    annotation: TraceAnnotation | None,
    *,
    requirement: str = "",
) -> dict:
    query_payload = trace_query_to_dict(query)
    nl_review = (
        _nl_temporal_review_record(requirement, query, annotation)
        if annotation is not None
        else None
    )
    return {
        "query": query_payload,
        "annotation": asdict(annotation) if annotation is not None else None,
        "nl_temporal_simulation_review": (
            asdict(nl_review) if nl_review is not None else None
        ),
    }


def trace_query_to_dict(query: TraceQuery) -> dict[str, Any]:
    payload = asdict(query)
    payload["trace"] = [sorted(step) for step in query.trace]
    return payload


def _nl_temporal_review_record(
    requirement: str,
    query: TraceQuery,
    annotation: TraceAnnotation,
) -> NLTemporalSimulationReview:
    is_boundary = (
        "boundary" in query.role.lower()
        or "boundary" in query.target_ambiguity.lower()
    )
    if annotation.label == "SAT":
        verdict = "boundary_satisfying" if is_boundary else "positive_satisfying"
    elif annotation.label == "UNSAT":
        verdict = "boundary_violating" if is_boundary else "negative_violating"
    else:
        verdict = "unclear"
    return NLTemporalSimulationReview(
        review_id=f"nl-review-{query.query_id}",
        trace_id=query.query_id,
        nl_text=requirement,
        trace_steps=tuple(tuple(sorted(step)) for step in query.trace),
        review_basis="NL_only",
        temporal_simulation_summary=(
            annotation.temporal_simulation_summary
            or annotation.rationale
            or "Trace reviewed against the natural-language requirement."
        ),
        nl_verdict=verdict,
        reason=annotation.rationale,
        supported_text_spans=annotation.supported_text_spans,
        reviewer_id=annotation.annotator or annotation.label_source,
        confidence=annotation.confidence,
        query_role=query.role,
        target_ambiguity=query.target_ambiguity,
        obligation_ids=query.obligation_ids,
        ir_version=query.ir_version,
        trace_fingerprint=trace_query_fingerprint(query),
        data_split=query.data_split,
        label_source=annotation.label_source,
        formula_source="none",
        formula_visibility=(),
    )


def _judge_label(
    requirement: str,
    semantic: SemanticStructure,
    query: TraceQuery,
) -> str:
    role = query.query_id.lower()
    pattern = semantic.pattern.lower()
    text = " ".join(
        (
            requirement,
            semantic.scope_keyword,
            semantic.temporal_keyword,
        )
    ).lower()

    if pattern == "response":
        if "missing-response" in role:
            return "UNSAT"
        if "response-before-trigger" in role:
            return "UNSAT"
        if "same-step" in role or "final-state" in role:
            strict_future = any(
                cue in text
                for cue in (
                    "next step",
                    "strict future",
                    "must occur later",
                    "之后",
                    "下一步",
                )
            )
            return "UNSAT" if strict_future else "SAT"
        if "witness" in role:
            return "SAT"
        trigger_positions = [
            index
            for index, step in enumerate(query.trace)
            if semantic.trigger in step
        ]
        if not trigger_positions:
            return "SAT"
        strict_future = any(
            cue in text
            for cue in ("next step", "strict future", "must occur later", "之后", "下一步")
        )
        global_scope = any(
            cue in text
            for cue in ("whenever", "always", "global", "every time", "每当", "始终")
        )
        positions = trigger_positions if global_scope else trigger_positions[:1]
        for position in positions:
            start = position + 1 if strict_future else position
            if not any(
                semantic.response in query.trace[index]
                for index in range(start, len(query.trace))
            ):
                return "UNSAT"
        return "SAT"

    if pattern in {"precedence", "before"}:
        if any(token in role for token in ("missing-predecessor", "late-predecessor")):
            return "UNSAT"
        if "same-step" in role:
            return "UNSAT"
        return "SAT"

    if pattern == "until":
        if "left-break" in role or "missing-until-witness" in role:
            return "UNSAT"
        return "SAT"

    if pattern in {"mutual_exclusion", "mutual_exclusion_with_coverage"}:
        if "cooccurrence" in role:
            return "UNSAT"
        if "neither" in role:
            return (
                "UNSAT"
                if pattern == "mutual_exclusion_with_coverage"
                else "SAT"
            )
        return "SAT"

    if pattern in {"persistence", "global_state", "invariance"}:
        return "SAT" if "global-witness" in role or "minimal-length" in role else "UNSAT"

    return "UNCLEAR"


def _judge_confidence(requirement: str, query: TraceQuery) -> str:
    if query.target_ambiguity in {
        "same_step_vs_strict_future",
        "final_state_behavior",
    } and not any(
        cue in requirement.lower()
        for cue in ("same", "simultaneous", "next", "later", "final", "eventually")
    ):
        return "low"
    return "medium"


def _read_field(section: str, name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}\s*:\s*(.*?)\s*$",
        section,
        flags=re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _read_list_field(section: str, name: str) -> tuple[str, ...]:
    raw = _read_field(section, name)
    if not raw or raw == "[]":
        return ()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [item.strip() for item in raw.split("|")]
    return _string_tuple(value)


def _extract_between(section: str, start_marker: str, end_marker: str) -> str:
    start = section.find(start_marker)
    end = section.find(end_marker)
    if start == -1 or end == -1 or end < start:
        return ""
    return section[start + len(start_marker) : end].strip()
