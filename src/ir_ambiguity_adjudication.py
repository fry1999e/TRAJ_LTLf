"""Two-round, candidate-blind NL trajectory adjudication for CandidateIRSet.

This stage chooses between complete IR branches.  It is deliberately separate
from operator-hole resolution, which runs only after one IR version has been
selected and consumes operator-purpose evidence.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from typing import Any

from faithful_ir import compute_ir_version, validate_faithful_ir
from ir_branch_discriminator import (
    PolicyBranchSelectionOutcome,
    select_policy_candidate_from_annotations,
    synthesize_policy_discriminating_queries,
)
from ir_trace_semantics import IRTraceSemanticsError, evaluate_ir_trace
from semantic_adjudication import (
    bind_trace_queries_to_brief,
    build_trace_adjudication_brief,
)
from schemas import (
    IRAmbiguityAdjudicationReport,
    IRAmbiguityAdjudicationRound,
    IRCandidateSemanticScore,
    SemanticStructure,
    TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION,
    TraceAnnotation,
    TraceQuery,
)
from trace_review_protocol import manual_trace_annotations


MAX_IR_ADJUDICATION_ROUNDS = 2


def requires_ir_candidate_adjudication(
    semantic: SemanticStructure | None,
) -> bool:
    candidate_set = semantic.candidate_ir_set if semantic is not None else None
    return bool(
        candidate_set is not None
        and len(candidate_set.branches) > 1
        and candidate_set.selection_status == "unresolved"
    )


def adjudicate_ir_candidate_branches(
    requirement: str,
    semantic: SemanticStructure,
    *,
    judge_client: Any = None,
    data_split: str = "test",
    max_rounds: int = MAX_IR_ADJUDICATION_ROUNDS,
    records_dir: str = "data/output/manual/review_records",
    annotation_provider=None,
    llm_client: Any = None,
) -> PolicyBranchSelectionOutcome:
    """Resolve a complete IR branch, with a non-authoritative fallback.

    The user sees only the confirmed ordinary-language meaning and raw finite
    traces. Candidate identities, IR graphs, formulas, and predictions are not
    included in the adjudication payload.
    """

    if judge_client is not None:
        raise ValueError("Model reviewers are disabled for IR branch adjudication")
    if annotation_provider is None:
        def annotation_provider(**kwargs):
            return manual_trace_annotations(**kwargs, llm_client=llm_client)
    candidate_set = semantic.candidate_ir_set
    if candidate_set is None or len(candidate_set.branches) < 2:
        report = _report(
            semantic,
            status="not_required",
            selected_branch_id=(
                candidate_set.branches[0].ir_branch_id
                if candidate_set is not None and candidate_set.branches
                else ""
            ),
            reasons=("fewer than two complete IR branches",),
        )
        return PolicyBranchSelectionOutcome(
            status="not_required",
            semantic=replace(semantic, ir_ambiguity_adjudication=report),
            selected_branch_id=report.selected_branch_id,
            reason=report.reasons[0],
        )
    _assert_plain_reading_is_closed(semantic)
    max_rounds = max(1, min(MAX_IR_ADJUDICATION_ROUNDS, int(max_rounds)))

    all_queries: list[TraceQuery] = []
    all_annotations: list[TraceAnnotation] = []
    rounds: list[IRAmbiguityAdjudicationRound] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    prior_failures: tuple[str, ...] = ()
    last_selection: PolicyBranchSelectionOutcome | None = None

    for round_index in range(1, max_rounds + 1):
        queries = synthesize_policy_discriminating_queries(
            semantic,
            max_length=5 + (round_index - 1) * 2,
            max_count=4,
            data_split=data_split,
            round_index=round_index,
            seen_traces=tuple(sorted(seen)),
            failure_reasons=prior_failures,
        )
        brief = build_trace_adjudication_brief(requirement, semantic)
        queries = bind_trace_queries_to_brief(queries, brief)
        annotations: list[TraceAnnotation] = []
        round_failures: list[str] = []
        if not queries:
            round_failures.append("no_new_discriminating_trace")
        else:
            try:
                annotations = annotation_provider(
                    requirement=requirement,
                    semantic=semantic,
                    queries=queries,
                    round_index=round_index,
                    records_dir=records_dir,
                    adjudication_brief=brief,
                )
            except Exception as exc:  # a failed Judge round must feed round two
                round_failures.append(
                    f"human_adjudication_error:{type(exc).__name__}:{exc}"
                )
                annotations = []

        usable = tuple(
            annotation
            for annotation in annotations
            if annotation.purpose == TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION
            and annotation.label in {"SAT", "UNSAT"}
            and annotation.confidence in {"high", "medium"}
            and annotation.uncertainty_kind == "none"
        )
        if len(usable) < len(queries):
            round_failures.append("unresolved_or_low_confidence_nl_labels")
        all_queries.extend(queries)
        all_annotations.extend(usable)
        answered_ids = {annotation.query_id for annotation in annotations}
        seen.update(_trace_key(query) for query in queries if query.query_id in answered_ids)

        scores = _candidate_scores(semantic, all_queries, all_annotations)
        if all_annotations:
            last_selection = select_policy_candidate_from_annotations(
                semantic,
                all_queries,
                all_annotations,
                accepted_label_sources=("human",),
                selection_basis="human_nl_trace_discrimination",
                decision_source="human_nl_trace_discrimination",
                mapping_rule_id="ir-branch-human-trace-discrimination-v1",
                reviewer_default="human-user",
            )
            if last_selection.status != "unique":
                round_failures.append(last_selection.status)
        else:
            round_failures.append("no_usable_nl_trace_labels")

        rounds.append(
            IRAmbiguityAdjudicationRound(
                round_index=round_index,
                query_ids=tuple(query.query_id for query in queries),
                accepted_query_ids=tuple(item.query_id for item in usable),
                unresolved_query_ids=tuple(
                    query.query_id
                    for query in queries
                    if query.query_id not in {item.query_id for item in usable}
                ),
                candidate_scores=scores,
                failure_reasons=tuple(dict.fromkeys(round_failures)),
            )
        )
        if last_selection is not None and last_selection.status == "unique":
            last_selection = _close_trace_selected_plain_reading(last_selection, all_annotations)
            report = _report(
                last_selection.semantic,
                status="unique_resolved",
                selected_branch_id=last_selection.selected_branch_id,
                rounds=tuple(rounds),
                supporting_query_ids=last_selection.supporting_query_ids,
                authoritative=True,
                selection_basis="human_nl_trace_discrimination",
                reasons=(last_selection.reason,),
            )
            return replace(
                last_selection,
                semantic=replace(
                    last_selection.semantic,
                    ir_ambiguity_adjudication=report,
                ),
            )
        prior_failures = tuple(dict.fromkeys(round_failures))
        if annotation_provider is not None and llm_client is not None:
            from human_review_session import review_budget_state, BUDGET_DEBT
            budget = review_budget_state(requirement, semantic, brief, records_dir)
            if budget and not budget["remaining"]:
                prior_failures = (*prior_failures, BUDGET_DEBT)
                break

    return _provisional_fallback(
        semantic,
        rounds=tuple(rounds),
        queries=all_queries,
        annotations=all_annotations,
        reasons=prior_failures or ("two rounds did not identify one IR branch",),
    )


def _close_trace_selected_plain_reading(selection, annotations):
    """Commit only the reading proved by the unique human trajectory choice."""
    semantic = selection.semantic
    review = semantic.plain_reading_review
    if review is None or review.overall_status != "ambiguous_unresolved":
        return selection
    segmentation = semantic.surface_segmentation
    candidate_set = semantic.candidate_ir_set
    if segmentation is None or candidate_set is None or candidate_set.selection_status != "unique":
        raise ValueError("trace-selected reading requires a unique bound candidate set")
    trusted = {annotation.query_id for annotation in annotations
               if annotation.label_source == "human" and annotation.label in {"SAT", "UNSAT"}
               and annotation.confidence in {"high", "medium"} and annotation.uncertainty_kind == "none"}
    if not selection.supporting_query_ids or not set(selection.supporting_query_ids).issubset(trusted):
        raise ValueError("trace-selected reading lacks locked human evidence")
    selected = next(branch for branch in candidate_set.branches
                    if branch.ir_branch_id == candidate_set.selected_branch_id)
    decisions = dict(review.selected_options)
    for ambiguity in review.ambiguities:
        if ambiguity.decision_status != "ambiguous_unresolved":
            continue
        prefix = f"plain_reading_option:{ambiguity.ambiguity_id}="
        options = [assumption[len(prefix):] for assumption in selected.semantic_assumptions
                   if assumption.startswith(prefix)]
        if len(options) != 1 or options[0] not in {option.option_id for option in ambiguity.options}:
            raise ValueError("selected branch lacks an exact plain-reading option binding")
        decisions[ambiguity.ambiguity_id] = options[0]
    from plain_reading import apply_plain_reading_decisions, build_plain_reading_review
    if review.no_ambiguity_confirmation.judge_evidence_json:
        # The earlier model's empty-axis inspection is an input, not the new
        # decision's evidence. Retain it in history, without claiming approval.
        review = build_plain_reading_review(
            requirement_id=review.requirement_id, segmentation=segmentation,
            ambiguities=review.ambiguities, overall_status="ambiguous_unresolved",
            review_basis="model_assisted_ordinary_language", reviewer_source=review.reviewer_source,
            generator=review.generator, experiment_mode=review.experiment_mode, data_split=review.data_split,
            review_history=(*review.review_history, {
                "operation": "model_reading_evidence_before_human_trajectory_selection",
                "old_reading_review_id": review.reading_review_id,
                "no_ambiguity_confirmation": asdict(review.no_ambiguity_confirmation),
            }), supersedes_reading_review_id=review.reading_review_id,
        )
    closed = apply_plain_reading_decisions(
        review, segmentation, decisions,
        reviewer_id="human-trajectory-selection", reviewer_role="ordinary_reviewer",
        reviewer_source="human_trace", review_basis="human_trajectory_selected_reading",
        comments={axis_id: "Selected by locked human trajectory judgments: "
                  + ", ".join(selection.supporting_query_ids) for axis_id in decisions},
    )
    from faithful_ir import bind_ir_to_pre_ir_artifacts
    from semantic_progress import semantic_fingerprint
    branches = []
    for branch in candidate_set.branches:
        rebound = replace(branch.faithful_ir, reading_review_id=closed.reading_review_id,
                          reading_review_basis=closed.review_basis, ir_version="")
        rebound = bind_ir_to_pre_ir_artifacts(rebound, surface_segmentation=segmentation,
                                            plain_reading_review=closed)
        validate_faithful_ir(rebound, surface_segmentation=segmentation, plain_reading_review=closed)
        branches.append(replace(branch, faithful_ir=rebound, structure_fingerprint=semantic_fingerprint(rebound)))
    rebound_set = replace(candidate_set, branches=tuple(branches), reading_review_id=closed.reading_review_id)
    selected_ir = next(branch.faithful_ir for branch in branches
                       if branch.ir_branch_id == candidate_set.selected_branch_id)
    return replace(selection, semantic=replace(
        semantic, faithful_ir=selected_ir, plain_reading_review=closed, candidate_ir_set=rebound_set,
        semantic_resolution=None, semantic_status="ready",
    ))


def _candidate_scores(
    semantic: SemanticStructure,
    queries: list[TraceQuery],
    annotations: list[TraceAnnotation],
) -> tuple[IRCandidateSemanticScore, ...]:
    candidate_set = semantic.candidate_ir_set
    if candidate_set is None:
        return ()
    query_by_id = {query.query_id: query for query in queries}
    labels = tuple(
        (query_by_id[item.query_id], item.label == "SAT")
        for item in annotations
        if item.query_id in query_by_id
    )
    scores: list[IRCandidateSemanticScore] = []
    for branch in candidate_set.branches:
        reasons: list[str] = []
        validation_passed = True
        try:
            validate_faithful_ir(branch.faithful_ir)
        except Exception as exc:
            validation_passed = False
            reasons.append(f"validation_error:{type(exc).__name__}:{exc}")
        matched = 0
        compared = 0
        if validation_passed:
            for query, expected in labels:
                try:
                    actual = evaluate_ir_trace(branch.faithful_ir, query.trace)
                except (IRTraceSemanticsError, ValueError, TypeError) as exc:
                    reasons.append(
                        f"trace_execution_error:{query.query_id}:{type(exc).__name__}"
                    )
                    continue
                compared += 1
                matched += actual == expected
        scores.append(
            IRCandidateSemanticScore(
                branch_id=branch.ir_branch_id,
                matched_count=matched,
                compared_count=compared,
                agreement=(matched / compared if compared else 0.0),
                validation_passed=validation_passed,
                reasons=tuple(reasons),
            )
        )
    return tuple(scores)


def _provisional_fallback(
    semantic: SemanticStructure,
    *,
    rounds: tuple[IRAmbiguityAdjudicationRound, ...],
    queries: list[TraceQuery],
    annotations: list[TraceAnnotation],
    reasons: tuple[str, ...],
) -> PolicyBranchSelectionOutcome:
    candidate_set = semantic.candidate_ir_set
    assert candidate_set is not None
    scores = _candidate_scores(semantic, queries, annotations)
    score_by_id = {score.branch_id: score for score in scores}
    valid = tuple(
        branch
        for branch in candidate_set.branches
        if score_by_id.get(branch.ir_branch_id) is not None
        and score_by_id[branch.ir_branch_id].validation_passed
    )
    if not valid:
        report = _report(
            semantic,
            status="no_valid_candidate",
            rounds=rounds,
            reasons=(*reasons, "no candidate IR passed hard validation"),
        )
        return PolicyBranchSelectionOutcome(
            status="no_valid_candidate",
            semantic=replace(semantic, ir_ambiguity_adjudication=report),
            reason="No candidate IR passed hard validation.",
        )

    selected = max(
        valid,
        key=lambda branch: (
            score_by_id[branch.ir_branch_id].agreement,
            score_by_id[branch.ir_branch_id].matched_count,
            -len(branch.semantic_assumptions),
            branch.structure_fingerprint,
            branch.ir_branch_id,
        ),
    )
    base_ir = semantic.faithful_ir
    base_resolution = {
        item.ambiguity_id: item.resolved
        for item in (base_ir.ambiguities if base_ir is not None else ())
    }
    selected_ir = replace(
        selected.faithful_ir,
        ambiguities=tuple(
            replace(
                item,
                resolved=base_resolution.get(item.ambiguity_id, item.resolved),
            )
            for item in selected.faithful_ir.ambiguities
        ),
        provenance=replace(
            selected.faithful_ir.provenance,
            extraction_mode="provisional_two_round_nl_trace_fallback",
            approved=False,
            approval_source="",
        ),
        ir_version="",
    )
    selected_ir = replace(selected_ir, ir_version=compute_ir_version(selected_ir))
    validate_faithful_ir(selected_ir)
    provisional_set = replace(
        candidate_set,
        branches=tuple(
            replace(
                branch,
                faithful_ir=(
                    selected_ir
                    if branch.ir_branch_id == selected.ir_branch_id
                    else branch.faithful_ir
                ),
                status=(
                    "provisional_selected"
                    if branch.ir_branch_id == selected.ir_branch_id
                    else "provisional_not_selected"
                ),
            )
            for branch in candidate_set.branches
        ),
        status="provisional_selected",
        selection_status="provisional",
        selection_basis="two_round_nl_trace_fallback",
        selected_branch_id=selected.ir_branch_id,
    )
    unresolved_ids = tuple(dict.fromkeys((
        *(
            item.ambiguity_id
            for item in selected_ir.ambiguities
            if not item.resolved
        ),
        *(
            axis.axis_id
            for axis in (
                candidate_set.ambiguity_axes.axes
                if candidate_set.ambiguity_axes is not None
                else ()
            )
            if axis.review_status not in {
                "resolved",
                "expert_trace_resolved",
                "llm_trace_resolved",
            }
        ),
    )))
    provisional_semantic = replace(
        semantic,
        faithful_ir=selected_ir,
        candidate_ir_set=provisional_set,
        approved_semantic_ir=None,
        labeled_trace_evidence=None,
        semantic_evidence_gate=None,
        semantic_status="needs_semantic_review",
        persistent_rag_allowed=False,
    )
    report = _report(
        provisional_semantic,
        status="provisional_fallback",
        selected_branch_id=selected.ir_branch_id,
        rounds=rounds,
        supporting_query_ids=tuple(item.query_id for item in annotations),
        unresolved_ambiguity_ids=unresolved_ids,
        provisional=True,
        selection_basis="two_round_nl_trace_fallback",
        reasons=reasons,
    )
    provisional_semantic = replace(
        provisional_semantic,
        ir_ambiguity_adjudication=report,
    )
    return PolicyBranchSelectionOutcome(
        status="provisional_fallback",
        semantic=provisional_semantic,
        consistent_branch_ids=tuple(branch.ir_branch_id for branch in valid),
        selected_branch_id=selected.ir_branch_id,
        supporting_query_ids=report.supporting_query_ids,
        reason=(
            "Two NL trajectory rounds were inconclusive; selected the stable "
            "highest-agreement valid IR branch provisionally."
        ),
    )


def _assert_plain_reading_is_closed(semantic: SemanticStructure) -> None:
    review = semantic.plain_reading_review
    if review is None:
        return
    segmentation = semantic.surface_segmentation
    if segmentation is None:
        raise ValueError("Plain Reading is missing its parent segmentation")
    from plain_reading import (
        assess_plain_reading_ir_readiness,
        assert_plain_reading_ready_for_candidate_ir,
        assert_plain_reading_ready_for_semantic_ir,
    )

    readiness = assess_plain_reading_ir_readiness(review, segmentation)
    if readiness.can_enter_semantic_ir:
        assert_plain_reading_ready_for_semantic_ir(review, segmentation)
        return
    candidate_set = semantic.candidate_ir_set
    if (
        readiness.disposition != "semantic_resolution_required"
        or candidate_set is None
        or candidate_set.candidate_catalog_version
        != "source-intent-graph-candidates-v1"
        or not candidate_set.branches
    ):
        assert_plain_reading_ready_for_semantic_ir(review, segmentation)
        return
    mappings = {}
    expected_combinations = 1
    for ambiguity in review.ambiguities:
        if ambiguity.decision_status != "ambiguous_unresolved":
            continue
        choices = {}
        for option in ambiguity.options:
            marker = f"plain_reading_option:{ambiguity.ambiguity_id}={option.option_id}"
            branch = next(
                (
                    item for item in candidate_set.branches
                    if marker in item.semantic_assumptions
                ),
                None,
            )
            if branch is None:
                break
            choices[option.option_id] = {
                "operation": "replace_node",
                "node_id": branch.faithful_ir.root_node_id,
            }
        if len(choices) != len(ambiguity.options):
            assert_plain_reading_ready_for_semantic_ir(review, segmentation)
            return
        mappings[ambiguity.ambiguity_id] = choices
        expected_combinations *= len(ambiguity.options)
    observed = {
        tuple(
            assumption
            for assumption in branch.semantic_assumptions
            if assumption.startswith("plain_reading_option:")
        )
        for branch in candidate_set.branches
    }
    if len(observed) != expected_combinations:
        raise ValueError("plain-reading candidate Cartesian product is incomplete")
    assert_plain_reading_ready_for_candidate_ir(
        review,
        segmentation,
        mapped_axes=mappings,
        max_candidates=candidate_set.max_branch_count,
    )


def _report(
    semantic: SemanticStructure,
    *,
    status: str,
    selected_branch_id: str = "",
    rounds: tuple[IRAmbiguityAdjudicationRound, ...] = (),
    supporting_query_ids: tuple[str, ...] = (),
    unresolved_ambiguity_ids: tuple[str, ...] = (),
    selection_basis: str = "",
    provisional: bool = False,
    authoritative: bool = False,
    reasons: tuple[str, ...] = (),
) -> IRAmbiguityAdjudicationReport:
    candidate_set = semantic.candidate_ir_set
    candidate_set_id = (
        candidate_set.branch_set_id
        if candidate_set is not None and candidate_set.branch_set_id
        else "candidate-set-"
        + _digest(
            [
                branch.ir_branch_id
                for branch in (candidate_set.branches if candidate_set else ())
            ],
            length=16,
        )
    )
    reading_review_id = (
        semantic.plain_reading_review.reading_review_id
        if semantic.plain_reading_review is not None
        else semantic.faithful_ir.reading_review_id
        if semantic.faithful_ir is not None
        else ""
    )
    payload = {
        "requirement_id": candidate_set.requirement_id if candidate_set else "",
        "candidate_set_id": candidate_set_id,
        "reading_review_id": reading_review_id,
        "status": status,
        "selected_branch_id": selected_branch_id,
        "rounds": [asdict(item) for item in rounds],
    }
    return IRAmbiguityAdjudicationReport(
        adjudication_id="IRADJ-" + _digest(payload, length=20),
        requirement_id=str(payload["requirement_id"]),
        candidate_set_id=candidate_set_id,
        reading_review_id=reading_review_id,
        status=status,
        selected_branch_id=selected_branch_id,
        rounds=rounds,
        supporting_query_ids=supporting_query_ids,
        unresolved_ambiguity_ids=unresolved_ambiguity_ids,
        selection_basis=selection_basis,
        provisional=provisional,
        authoritative=authoritative,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _trace_key(query: TraceQuery) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted(step)) for step in query.trace)


def _digest(payload: object, *, length: int) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:length]


__all__ = [
    "MAX_IR_ADJUDICATION_ROUNDS",
    "adjudicate_ir_candidate_branches",
    "requires_ir_candidate_adjudication",
]
