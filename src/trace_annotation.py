"""Trace-query adjudication and conversion to downstream LabeledTrace objects."""

from __future__ import annotations

import hashlib

from data_examples import LabeledTrace, TraceSemantics
from ltlf_checker import check_trace
from schemas import ExperimentConfig, TraceAnnotation, TraceQuery
from trace_evidence_policy import is_boundary_query


class TraceEvidenceConflictError(ValueError):
    """Raised when the same finite trace receives contradictory NL labels."""


def adjudicate_trace_queries(
    queries: list[TraceQuery],
    annotations: list[TraceAnnotation],
) -> tuple[list[LabeledTrace], list[TraceQuery]]:
    """
    Apply independent annotations without overwriting or inventing labels.

    UNCLEAR and INFEASIBLE queries are returned as unresolved evidence and are
    deliberately excluded from candidate scoring.
    """

    by_id = {annotation.query_id: annotation for annotation in annotations}
    if len(by_id) != len(annotations):
        raise ValueError("Duplicate TraceAnnotation.query_id values are not allowed.")

    labeled: list[LabeledTrace] = []
    unresolved: list[TraceQuery] = []
    for query in queries:
        annotation = by_id.get(query.query_id)
        if annotation is None:
            unresolved.append(query)
            continue
        if annotation.purpose != query.purpose:
            raise ValueError(
                f"TraceAnnotation purpose mismatch for {query.query_id}: "
                f"{annotation.purpose!r} != {query.purpose!r}"
            )
        if annotation.label in {"UNCLEAR", "INFEASIBLE"}:
            unresolved.append(query)
            continue
        if _annotation_requires_semantic_clarification(annotation):
            unresolved.append(query)
            continue

        expected = annotation.label == "SAT"
        trace_role = query.role
        is_boundary = is_boundary_query(query)
        kind = "boundary" if is_boundary else (
            "positive" if expected else "negative"
        )
        nl_verdict = _nl_verdict(annotation.label, is_boundary)
        review_id = _review_id(query.query_id, annotation)
        temporal_summary = (
            annotation.temporal_simulation_summary
            or annotation.rationale
            or "Trace reviewed against the natural-language requirement."
        )
        ambiguity_axes = _query_ambiguity_axes(query)
        semantics = TraceSemantics(
            semantic_label="SAT_WITNESS" if expected else "UNSAT",
            roles=(trace_role,) if trace_role else (),
            obligations=query.obligation_ids,
            confusion_targets=ambiguity_axes,
        )
        labeled.append(
            LabeledTrace(
                name=query.query_id,
                kind=kind,
                trace=[set(step) for step in query.trace],
                expected=expected,
                reason=query.reason,
                beginner_view=query.beginner_view,
                semantics=semantics,
                role=query.role,
                query_role=trace_role,
                target_ambiguity=query.target_ambiguity,
                query_constraints=query.constraints,
                adjudicated_label=annotation.label,
                confidence=annotation.confidence,
                rationale=annotation.rationale,
                label_source=annotation.label_source,
                data_split=query.data_split,
                formula_source=query.formula_source,
                formula_visibility=query.formula_visibility,
                annotation_locked=True,
                allowed_rag_stages=(
                    "skeleton",
                    "completion",
                    "selection",
                ),
                trace_id=query.query_id,
                covered_ir_node_ids=query.ir_node_ids,
                obligation_ids=query.obligation_ids,
                trace_role=trace_role,
                boundary_role=(
                    query.target_ambiguity
                    if is_boundary
                    else ""
                ),
                generation_source=query.generation_source or "independent_adjudication",
                ir_version=query.ir_version,
                target_ambiguity_axis_ids=ambiguity_axes,
                nl_verdict=nl_verdict,
                temporal_simulation_summary=temporal_summary,
                review_record_id=review_id,
                supported_text_spans=annotation.supported_text_spans,
                evidence_purpose=query.purpose,
                parent_candidate_set_id=query.parent_candidate_set_id,
                parent_hole_plan_version=query.parent_hole_plan_version,
            )
        )
    return labeled, unresolved


def merge_labeled_trace_evidence(
    existing: list[LabeledTrace],
    newly_adjudicated: list[LabeledTrace],
) -> list[LabeledTrace]:
    """Retain accepted evidence across rounds and reject label corruption.

    Exact repeated evidence is refreshed when it is rebound to a newer IR
    version.  Different obligation/role bindings are retained because one trace
    can legitimately provide more than one kind of coverage evidence.
    """

    merged = list(existing)
    shape_labels: dict[
        tuple[str, tuple[tuple[str, ...], ...]], bool
    ] = {}
    exact_index: dict[tuple[object, ...], int] = {}
    for index, trace in enumerate(merged):
        shape = (trace.evidence_purpose, _trace_shape(trace))
        previous = shape_labels.get(shape)
        if previous is not None and previous != trace.expected:
            raise TraceEvidenceConflictError(
                f"existing evidence already contains contradictory labels for {shape}"
            )
        shape_labels[shape] = trace.expected
        exact_index[_trace_evidence_key(trace)] = index

    for trace in newly_adjudicated:
        shape = (trace.evidence_purpose, _trace_shape(trace))
        previous = shape_labels.get(shape)
        if previous is not None and previous != trace.expected:
            raise TraceEvidenceConflictError(
                "contradictory adjudication for finite trace "
                f"{shape}: existing={previous}, new={trace.expected}"
            )
        shape_labels[shape] = trace.expected
        key = _trace_evidence_key(trace)
        duplicate = exact_index.get(key)
        if duplicate is None:
            exact_index[key] = len(merged)
            merged.append(trace)
        elif trace.ir_version and merged[duplicate].ir_version != trace.ir_version:
            # The human/Judge label is unchanged; only derived IR bindings are
            # refreshed so coverage does not remain attached to a stale version.
            merged[duplicate] = trace
    return merged


def _trace_shape(trace: LabeledTrace) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted(step)) for step in trace.trace)


def _trace_evidence_key(trace: LabeledTrace) -> tuple[object, ...]:
    return (
        _trace_shape(trace),
        trace.expected,
        trace.kind,
        tuple(trace.obligation_ids),
        trace.trace_role,
        tuple(trace.target_ambiguity_axis_ids),
        trace.evidence_purpose,
        trace.parent_candidate_set_id,
        trace.parent_hole_plan_version,
    )


def _annotation_requires_semantic_clarification(annotation: TraceAnnotation) -> bool:
    """Do not freeze a label when the reviewer explicitly depends on one reading."""

    if annotation.confidence == "low":
        return True
    if annotation.confidence != "medium":
        return False
    text = " ".join(
        (annotation.rationale, annotation.temporal_simulation_summary)
    ).lower()
    return any(
        marker in text
        for marker in (
            "under the interpretation",
            "depending on the interpretation",
            "counting convention",
            "natural language does not define",
            "ambiguous",
            "unclear whether",
            "cannot determine",
            "取决于解释",
            "计数约定",
            "存在歧义",
        )
    )


def _query_ambiguity_axes(query: TraceQuery) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            axis
            for axis in (*query.target_ambiguity_axis_ids, query.target_ambiguity)
            if axis and axis not in {"satisfy", "violate", "distinguish"}
        )
    )


def _nl_verdict(label: str, is_boundary: bool) -> str:
    if label == "SAT":
        return "boundary_satisfying" if is_boundary else "positive_satisfying"
    if label == "UNSAT":
        return "boundary_violating" if is_boundary else "negative_violating"
    return "unclear"


def _review_id(query_id: str, annotation: TraceAnnotation) -> str:
    digest = hashlib.sha1(
        "|".join(
            (
                query_id,
                annotation.label,
                annotation.confidence,
                annotation.rationale,
                annotation.label_source,
            )
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"nl-review-{digest}"


def oracle_annotations(
    queries: list[TraceQuery],
    config: ExperimentConfig,
) -> list[TraceAnnotation]:
    """Label queries only in the explicitly isolated formula-oracle mode."""

    if config.mode != "formula_oracle" or config.reference_formula is None:
        raise ValueError("Oracle annotation is only available in formula_oracle mode.")
    return [
        TraceAnnotation(
            query_id=query.query_id,
            label=(
                "SAT"
                if check_trace(query.trace, config.reference_formula)
                else "UNSAT"
            ),
            confidence="high",
            rationale="Formula-oracle upper-bound annotation.",
            label_source="formula_oracle",
            purpose=query.purpose,
        )
        for query in queries
    ]
