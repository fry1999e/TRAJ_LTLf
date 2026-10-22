"""
Load manually approved warmup examples from Markdown.

Warmup examples are separate from the main NL requirement file because they are
intended to teach the pipeline a complete, correct process before the main batch
runs. Each ``## WARM-001`` section must contain a ``requirement: ...`` field; the
whole section is preserved as the full-process reference text.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import inspect
import re
from pathlib import Path

from agents.llm_parsing import AP_PATTERN
from ap_binding_context import build_ap_binding_context
from ap_identity import offline_formula_assisted_profile
from contract_versions import SEMANTIC_EVIDENCE_CONTRACT_VERSION
from data_examples import LabeledTrace
from guided_learning import DEFAULT_ARTIFACT_VISIBILITY, GuidedLearningContext
from formula_closure import (
    FormulaClosurePolicy,
    build_strict_formula_closure_policy,
    validate_formula_closure,
)
from pattern_registry import (
    PatternSpec,
    collect_formula_atoms,
    normalize_formula_text,
    normalize_pattern,
    parse_learned_formula,
)
from requirement_loader import load_requirements_from_markdown
from schemas import (
    ApprovedSemanticIR,
    NLTemporalSimulationReview,
    PlainReadingReview,
    SemanticEvidenceGateReport,
    SemanticStructure,
    SurfaceSegmentation,
    TraceCoverageReport,
)
from faithful_ir import faithful_ir_from_mapping, reference_formula_to_ir
from ir_compiler import (
    compile_completion_hole_plan,
    compile_constrained_skeleton_plan,
    compile_review_trace_plan,
)
from trace_obligations import (
    analyze_obligations,
    enrich_trace_semantics,
    validate_trace_labels,
)
from semantic_freezing import freeze_approved_semantic_ir


@dataclass(frozen=True)
class WarmupExample:
    """One complete warmup example section from the warmup Markdown file."""

    requirement_id: str
    requirement: str
    process_text: str
    expected_traces: tuple[LabeledTrace, ...] = ()
    expected_semantic: SemanticStructure | None = None
    pattern_spec: PatternSpec | None = None
    guided_context: GuidedLearningContext | None = None
    gold_surface_segmentation: SurfaceSegmentation | None = None
    gold_plain_reading_review: PlainReadingReview | None = None
    gold_nl_temporal_reviews: tuple[NLTemporalSimulationReview, ...] = ()
    gold_labeled_trace_evidence: tuple[LabeledTrace, ...] = ()
    gold_trace_coverage_report: TraceCoverageReport | None = None
    gold_semantic_evidence_gate: SemanticEvidenceGateReport | None = None
    gold_approved_semantic_ir: ApprovedSemanticIR | None = None
    sample_kind: str = "new_unreviewed"
    gold_artifact_versions: tuple[tuple[str, str], ...] = ()


def load_warmup_examples_from_markdown(
    path: str | Path,
    *,
    allow_stale_semantic_gold: bool = False,
) -> list[WarmupExample]:
    """Return complete warmup examples from a Markdown file."""

    markdown_path = Path(path)
    if not markdown_path.exists():
        raise FileNotFoundError(
            f"未找到小样本热身文件: {markdown_path}. 请创建该文件，"
            "并使用 ## WARM-001 分节写入完整正确样例。"
        )

    text = markdown_path.read_text(encoding="utf-8")
    matches = list(
        re.finditer(
            r"^##\s+([A-Za-z]+-\d+)\s*$",
            text,
            flags=re.MULTILINE,
        )
    )
    if matches:
        return _parse_sectioned_warmup_examples(
            text,
            matches,
            markdown_path,
            allow_stale_semantic_gold=allow_stale_semantic_gold,
        )

    requirements = load_requirements_from_markdown(markdown_path)
    return [
        WarmupExample(
            requirement_id=f"WARM-{index:03d}",
            requirement=requirement,
            process_text=requirement,
            expected_traces=(),
            expected_semantic=None,
            pattern_spec=None,
            guided_context=None,
        )
        for index, requirement in enumerate(requirements, start=1)
    ]


def _parse_sectioned_warmup_examples(
    text: str,
    matches: list[re.Match[str]],
    path: Path,
    *,
    allow_stale_semantic_gold: bool = False,
) -> list[WarmupExample]:
    examples: list[WarmupExample] = []
    for index, match in enumerate(matches):
        requirement_id = match.group(1).upper()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        requirement = _read_requirement(section)
        if not requirement:
            raise ValueError(
                f"小样本文件 {path} 的 {requirement_id} 缺少 requirement 字段。"
            )
        (
            gold_surface_segmentation,
            surface_teacher_round,
            surface_gold_version,
        ) = _read_gold_surface_segmentation(
            section=section,
            requirement_id=requirement_id,
            requirement=requirement,
            path=path,
        )
        (
            gold_plain_reading_review,
            plain_teacher_round,
            plain_gold_version,
        ) = _read_gold_plain_reading_review(
            section=section,
            requirement_id=requirement_id,
            requirement=requirement,
            segmentation=gold_surface_segmentation,
            path=path,
        )
        _validate_semantic_response_contract(section, requirement_id, path)
        expected_semantic = _read_expected_semantic(
            section=section,
            requirement_id=requirement_id,
            requirement=requirement,
            path=path,
        )
        expected_traces = tuple(
            _read_expected_finite_traces(section, requirement_id, path)
        )
        gold_nl_temporal_reviews, review_gold_version = (
            _read_gold_nl_temporal_reviews(
                section=section,
                requirement_id=requirement_id,
                requirement=requirement,
                semantic=expected_semantic,
                path=path,
            )
        )
        gold_labeled_trace_evidence, evidence_gold_version = (
            _read_gold_labeled_trace_evidence(
                section=section,
                requirement_id=requirement_id,
                semantic=expected_semantic,
                path=path,
            )
        )
        gold_trace_coverage_report, coverage_gold_version = (
            _read_gold_trace_coverage_report(
                section=section,
                requirement_id=requirement_id,
                path=path,
            )
        )
        gold_semantic_evidence_gate, gate_gold_version = (
            _read_gold_semantic_evidence_gate(
                section=section,
                requirement_id=requirement_id,
                path=path,
            )
        )
        gold_approved_semantic_ir, approved_gold_version = (
            _read_gold_approved_semantic_ir(
                section=section,
                requirement_id=requirement_id,
                path=path,
            )
        )
        if not allow_stale_semantic_gold:
            _validate_frozen_trace_gold_contract(
                requirement_id=requirement_id,
                reviews=gold_nl_temporal_reviews,
                traces=gold_labeled_trace_evidence,
                path=path,
            )
            _validate_semantic_evidence_gate_contract(
                requirement_id=requirement_id,
                semantic=expected_semantic,
                traces=gold_labeled_trace_evidence,
                gate=gold_semantic_evidence_gate,
                approved_ir=gold_approved_semantic_ir,
                trace_coverage=gold_trace_coverage_report,
                contract_required="Gold TraceCoverageReport" in section,
                path=path,
            )
            _validate_expected_semantic_evidence_contract(
                requirement_id=requirement_id,
                semantic=expected_semantic,
                expected_traces=expected_traces,
                gold_gate=gold_semantic_evidence_gate,
                contract_required="Gold TraceCoverageReport" in section,
                path=path,
            )
        _validate_expected_trace_contract(
            requirement_id=requirement_id,
            requirement=requirement,
            semantic=expected_semantic,
            traces=expected_traces,
            path=path,
        )
        pattern_spec = _read_pattern_spec(
            section=section,
            requirement_id=requirement_id,
            requirement=requirement,
            semantic=expected_semantic,
            trace_count=len(expected_traces),
            path=path,
        )
        guided_context = _read_guided_learning_context(
            section=section,
            requirement_id=requirement_id,
            requirement=requirement,
            semantic=expected_semantic,
            traces=expected_traces,
            pattern_spec=pattern_spec,
            gold_surface_segmentation=gold_surface_segmentation,
            gold_plain_reading_review=gold_plain_reading_review,
            gold_nl_temporal_reviews=gold_nl_temporal_reviews,
            gold_labeled_trace_evidence=gold_labeled_trace_evidence,
            surface_teacher_round=surface_teacher_round,
            plain_teacher_round=plain_teacher_round,
        )
        sample_kind = _warmup_sample_kind(
            expected_semantic=expected_semantic,
            gold_surface_segmentation=gold_surface_segmentation,
            gold_plain_reading_review=gold_plain_reading_review,
            gold_nl_temporal_reviews=gold_nl_temporal_reviews,
            gold_labeled_trace_evidence=gold_labeled_trace_evidence,
        )
        gold_artifact_versions = tuple(
            (name, version)
            for name, version in (
                ("surface_segmentation", surface_gold_version),
                ("plain_reading_review", plain_gold_version),
                ("nl_temporal_simulation_review", review_gold_version),
                ("labeled_trace_evidence_set", evidence_gold_version),
                ("trace_coverage_report", coverage_gold_version),
                ("semantic_evidence_gate_report", gate_gold_version),
                ("approved_semantic_ir", approved_gold_version),
            )
            if version
        )
        if expected_semantic is not None:
            expected_semantic = replace(
                expected_semantic,
                formula_text="",
                formula_source="none",
                formula_visibility=(),
            )
        if guided_context is not None:
            guided_context = replace(
                guided_context,
                expected_semantic=expected_semantic,
            )
        examples.append(
            WarmupExample(
                requirement_id=requirement_id,
                requirement=requirement,
                process_text=section,
                expected_traces=expected_traces,
                expected_semantic=expected_semantic,
                pattern_spec=pattern_spec,
                guided_context=guided_context,
                gold_surface_segmentation=gold_surface_segmentation,
                gold_plain_reading_review=gold_plain_reading_review,
                gold_nl_temporal_reviews=gold_nl_temporal_reviews,
                gold_labeled_trace_evidence=gold_labeled_trace_evidence,
                gold_trace_coverage_report=gold_trace_coverage_report,
                gold_semantic_evidence_gate=gold_semantic_evidence_gate,
                gold_approved_semantic_ir=gold_approved_semantic_ir,
                sample_kind=sample_kind,
                gold_artifact_versions=gold_artifact_versions,
            )
        )

    if not examples:
        raise ValueError(f"小样本文件 {path} 中没有有效 warmup 样例。")
    return examples


def _validate_expected_trace_contract(
    *,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure | None,
    traces: tuple[LabeledTrace, ...],
    path: Path,
) -> None:
    if semantic is None or not traces:
        return
    profile = analyze_obligations(requirement, semantic)
    enriched = enrich_trace_semantics(list(traces), profile)
    _valid, conflicts = validate_trace_labels(enriched, profile)
    if conflicts:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 Expected Finite "
            "Traces 标签与 approved semantic obligation 不一致: "
            + "；".join(conflicts)
        )

    labels_by_shape: dict[tuple[tuple[str, ...], ...], bool] = {}
    for trace in enriched:
        shape = tuple(tuple(sorted(step)) for step in trace.trace)
        previous = labels_by_shape.get(shape)
        if previous is not None and previous != trace.expected:
            raise ValueError(
                f"小样本文件 {path} 的 {requirement_id} 中存在同形异标签轨迹: "
                f"{trace.name}"
            )
        labels_by_shape[shape] = trace.expected


def _read_requirement(section: str) -> str:
    field_match = re.search(
        r"^requirement\s*:\s*(.+?)\s*$",
        section,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if field_match:
        return field_match.group(1).strip()

    heading_match = re.search(
        r"^###\s+Requirement\s*$\s*```(?:text)?\s*(.*?)\s*```",
        section,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if heading_match:
        return " ".join(heading_match.group(1).split())

    return ""


_GOLD_SURFACE_HEADINGS = (
    "Gold Surface Segmentation",
    "Gold SurfaceSegmentation",
)
_GOLD_PLAIN_READING_HEADINGS = (
    "Gold Plain Reading Review",
    "Gold PlainReadingReview",
)
_GOLD_NL_TEMPORAL_REVIEW_HEADINGS = ("Gold NLTemporalSimulationReview",)
_GOLD_LABELED_TRACE_HEADINGS = ("Gold LabeledTraceEvidenceSet",)
_GOLD_TRACE_COVERAGE_HEADINGS = ("Gold TraceCoverageReport",)
_GOLD_SEMANTIC_EVIDENCE_GATE_HEADINGS = ("Gold SemanticEvidenceGateReport",)
_GOLD_APPROVED_SEMANTIC_IR_HEADINGS = ("Gold ApprovedSemanticIR",)


def _read_gold_nl_temporal_reviews(
    *,
    section: str,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure | None,
    path: Path,
) -> tuple[tuple[NLTemporalSimulationReview, ...], str]:
    artifact_payload, metadata = _read_gold_artifact_payload(
        section,
        _GOLD_NL_TEMPORAL_REVIEW_HEADINGS,
    )
    if artifact_payload is None:
        return (), ""
    _validated_teacher_signal_round(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold NLTemporalSimulationReview",
        path=path,
    )
    _validate_gold_visibility(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold NLTemporalSimulationReview",
        required_stage="trace_annotation",
        allowed_stages={"trace_annotation", "offline_validation"},
        path=path,
    )
    payload = _artifact_body(artifact_payload)
    contract_version = payload.get("semantic_evidence_contract_version")
    if contract_version is not None and (
        int(contract_version) != SEMANTIC_EVIDENCE_CONTRACT_VERSION
    ):
        raise ValueError(
            f"{requirement_id} Gold LabeledTraceEvidenceSet semantic evidence "
            "contract version drift。"
        )
    if str(payload.get("requirement_id", requirement_id)) != requirement_id:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 Gold "
            "NLTemporalSimulationReview requirement_id 不匹配。"
        )
    reviews_payload = payload.get("reviews")
    if not isinstance(reviews_payload, list) or not reviews_payload:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 Gold "
            "NLTemporalSimulationReview.reviews 必须是非空列表。"
        )
    ir_version = (
        semantic.faithful_ir.ir_version
        if semantic is not None and semantic.faithful_ir is not None
        else ""
    )
    reviews: list[NLTemporalSimulationReview] = []
    for raw in reviews_payload:
        if not isinstance(raw, dict):
            raise ValueError("NLTemporalSimulationReview item 必须是 object。")
        review = NLTemporalSimulationReview(
            review_id=str(raw.get("review_id", "")),
            trace_id=str(raw.get("trace_id", "")),
            nl_text=str(raw.get("nl_text", requirement)),
            trace_steps=tuple(
                tuple(str(ap) for ap in step)
                for step in raw.get("trace_steps", ())
            ),
            review_basis=str(raw.get("review_basis", "")),
            temporal_simulation_summary=str(
                raw.get("temporal_simulation_summary", "")
            ),
            nl_verdict=str(raw.get("nl_verdict", "unclear")),
            reason=str(raw.get("reason", "")),
            supported_text_spans=tuple(
                str(item) for item in raw.get("supported_text_spans", ())
            ),
            reviewer_id=str(raw.get("reviewer_id", "")),
            review_time=str(raw.get("review_time", "")),
            confidence=str(raw.get("confidence", "medium")),
            query_role=str(raw.get("query_role", "")),
            target_ambiguity=str(raw.get("target_ambiguity", "")),
            obligation_ids=tuple(
                str(item) for item in raw.get("obligation_ids", ())
            ),
            ir_version=str(raw.get("ir_version", "")),
            trace_fingerprint=str(raw.get("trace_fingerprint", "")),
            data_split=str(raw.get("data_split", "train")),
            label_source=str(raw.get("label_source", "warmup_train_gold")),
            formula_source=str(raw.get("formula_source", "none")),
            formula_visibility=tuple(
                str(item) for item in raw.get("formula_visibility", ())
            ),
        )
        if review.nl_text != requirement:
            raise ValueError(f"{requirement_id} frozen review NL text drift。")
        if review.ir_version != ir_version:
            raise ValueError(f"{requirement_id} frozen review IR version drift。")
        reviews.append(review)
    trace_ids = [review.trace_id for review in reviews]
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError(f"{requirement_id} frozen review trace_id 重复。")
    return tuple(reviews), str(payload.get("schema_version", ""))


def _read_gold_labeled_trace_evidence(
    *,
    section: str,
    requirement_id: str,
    semantic: SemanticStructure | None,
    path: Path,
) -> tuple[tuple[LabeledTrace, ...], str]:
    artifact_payload, metadata = _read_gold_artifact_payload(
        section,
        _GOLD_LABELED_TRACE_HEADINGS,
    )
    if artifact_payload is None:
        return (), ""
    _validated_teacher_signal_round(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold LabeledTraceEvidenceSet",
        path=path,
    )
    _validate_gold_visibility(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold LabeledTraceEvidenceSet",
        required_stage="trace_query_generation",
        allowed_stages={
            "trace_query_generation",
            "skeleton",
            "completion",
            "selection",
            "offline_validation",
        },
        path=path,
    )
    payload = _artifact_body(artifact_payload)
    if str(payload.get("requirement_id", requirement_id)) != requirement_id:
        raise ValueError(f"{requirement_id} gold trace evidence id drift。")
    raw_traces = payload.get("traces")
    if not isinstance(raw_traces, list) or not raw_traces:
        raise ValueError(f"{requirement_id} Gold LabeledTraceEvidenceSet.traces 为空。")
    ir_version = (
        semantic.faithful_ir.ir_version
        if semantic is not None and semantic.faithful_ir is not None
        else ""
    )
    traces: list[LabeledTrace] = []
    for raw in raw_traces:
        if not isinstance(raw, dict):
            raise ValueError("Gold labeled trace item 必须是 object。")
        raw_steps = raw.get("trace", ())
        trace = [set(str(ap) for ap in step) for step in raw_steps]
        label = str(raw.get("adjudicated_label", "")).upper()
        if label not in {"SAT", "UNSAT"}:
            raise ValueError(f"{requirement_id} gold trace label 非 Boolean。")
        item = LabeledTrace(
            name=str(raw.get("name", raw.get("trace_id", ""))),
            kind=str(raw.get("kind", "positive" if label == "SAT" else "negative")),
            trace=trace,
            expected=label == "SAT",
            reason=str(raw.get("reason", "Frozen train-only NL review.")),
            role=str(raw.get("role", "")),
            query_role=str(raw.get("query_role", "")),
            target_ambiguity=str(raw.get("target_ambiguity", "")),
            query_constraints=tuple(
                str(value) for value in raw.get("query_constraints", ())
            ),
            adjudicated_label=label,
            confidence=str(raw.get("confidence", "high")),
            rationale=str(raw.get("rationale", "")),
            label_source="warmup_train_gold",
            data_split="train",
            formula_source="none",
            formula_visibility=(),
            annotation_locked=True,
            trace_id=str(raw.get("trace_id", raw.get("name", ""))),
            covered_ir_node_ids=tuple(
                str(value) for value in raw.get("covered_ir_node_ids", ())
            ),
            obligation_ids=tuple(
                str(value) for value in raw.get("obligation_ids", ())
            ),
            trace_role=str(raw.get("trace_role", "")),
            boundary_role=str(raw.get("boundary_role", "")),
            generation_source=str(raw.get("generation_source", "frozen_train_gold")),
            ir_version=str(raw.get("ir_version", "")),
            target_ambiguity_axis_ids=tuple(
                str(value) for value in raw.get("target_ambiguity_axis_ids", ())
            ),
            nl_verdict=str(raw.get("nl_verdict", "")),
            temporal_simulation_summary=str(
                raw.get("temporal_simulation_summary", "")
            ),
            review_record_id=str(raw.get("review_record_id", "")),
            supported_text_spans=tuple(
                str(value) for value in raw.get("supported_text_spans", ())
            ),
        )
        if item.ir_version != ir_version:
            raise ValueError(f"{requirement_id} gold trace IR version drift。")
        traces.append(item)
    return tuple(traces), str(payload.get("schema_version", ""))


def _read_gold_trace_coverage_report(
    *,
    section: str,
    requirement_id: str,
    path: Path,
) -> tuple[TraceCoverageReport | None, str]:
    artifact_payload, metadata = _read_gold_artifact_payload(
        section,
        _GOLD_TRACE_COVERAGE_HEADINGS,
    )
    if artifact_payload is None:
        return None, ""
    _validated_teacher_signal_round(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold TraceCoverageReport",
        path=path,
    )
    _validate_gold_visibility(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold TraceCoverageReport",
        required_stage="semantic_approval",
        allowed_stages={"semantic_approval", "offline_validation"},
        path=path,
    )
    payload = _artifact_body(artifact_payload)
    schema_version = str(payload.pop("schema_version", ""))
    contract_version = int(payload.pop("semantic_evidence_contract_version", 0))
    if contract_version != SEMANTIC_EVIDENCE_CONTRACT_VERSION:
        raise ValueError(f"{requirement_id} Gold TraceCoverageReport version drift。")
    for field_name in (
        "missing",
        "coverage_debt",
        "covered_roles",
        "missing_roles",
        "covered_confusions",
        "missing_confusions",
    ):
        payload[field_name] = tuple(str(item) for item in payload.get(field_name, ()))
    report = TraceCoverageReport(**payload)
    if not report.complete:
        raise ValueError(f"{requirement_id} Gold TraceCoverageReport 未完成。")
    return report, schema_version


def _read_gold_semantic_evidence_gate(
    *,
    section: str,
    requirement_id: str,
    path: Path,
) -> tuple[SemanticEvidenceGateReport | None, str]:
    artifact_payload, metadata = _read_gold_artifact_payload(
        section,
        _GOLD_SEMANTIC_EVIDENCE_GATE_HEADINGS,
    )
    if artifact_payload is None:
        return None, ""
    _validated_teacher_signal_round(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold SemanticEvidenceGateReport",
        path=path,
    )
    _validate_gold_visibility(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold SemanticEvidenceGateReport",
        required_stage="semantic_approval",
        allowed_stages={"semantic_approval", "offline_validation"},
        path=path,
    )
    payload = _artifact_body(artifact_payload)
    schema_version = str(payload.pop("schema_version", ""))
    tuple_fields = {
        "required_obligation_ids",
        "covered_obligation_ids",
        "missing_obligation_ids",
        "required_boundary_obligation_ids",
        "covered_boundary_obligation_ids",
        "trace_ids",
        "conflicting_trace_ids",
        "remaining_ambiguity_debt",
        "reasons",
    }
    for field_name in tuple_fields:
        payload[field_name] = tuple(str(item) for item in payload.get(field_name, ()))
    gate = SemanticEvidenceGateReport(**payload)
    if gate.requirement_id != requirement_id:
        raise ValueError(f"{requirement_id} Gold SemanticEvidenceGateReport id drift。")
    if gate.evidence_contract_version != SEMANTIC_EVIDENCE_CONTRACT_VERSION:
        raise ValueError(f"{requirement_id} semantic evidence gate version drift。")
    return gate, schema_version


def _read_gold_approved_semantic_ir(
    *,
    section: str,
    requirement_id: str,
    path: Path,
) -> tuple[ApprovedSemanticIR | None, str]:
    artifact_payload, metadata = _read_gold_artifact_payload(
        section,
        _GOLD_APPROVED_SEMANTIC_IR_HEADINGS,
    )
    if artifact_payload is None:
        return None, ""
    _validated_teacher_signal_round(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold ApprovedSemanticIR",
        path=path,
    )
    _validate_gold_visibility(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold ApprovedSemanticIR",
        required_stage="skeleton",
        allowed_stages={"skeleton", "completion", "selection", "offline_validation"},
        path=path,
    )
    payload = _artifact_body(artifact_payload)
    schema_version = str(payload.pop("schema_version", ""))
    for field_name in (
        "approved_ap_mapping",
        "resolved_ambiguity_axes",
        "remaining_ambiguity_debt",
        "labeled_trace_evidence_ids",
    ):
        values = payload.get(field_name, ())
        payload[field_name] = (
            tuple(tuple(str(value) for value in item) for item in values)
            if field_name == "approved_ap_mapping"
            else tuple(str(item) for item in values)
        )
    approved = ApprovedSemanticIR(**payload)
    if approved.evidence_contract_version != SEMANTIC_EVIDENCE_CONTRACT_VERSION:
        raise ValueError(f"{requirement_id} Gold ApprovedSemanticIR version drift。")
    return approved, schema_version


def _validate_frozen_trace_gold_contract(
    *,
    requirement_id: str,
    reviews: tuple[NLTemporalSimulationReview, ...],
    traces: tuple[LabeledTrace, ...],
    path: Path,
) -> None:
    if not reviews and not traces:
        return
    if not reviews or not traces:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 必须同时提供 frozen "
            "NLTemporalSimulationReview 和 LabeledTraceEvidenceSet。"
        )
    by_review_id = {review.trace_id: review for review in reviews}
    by_trace_id = {trace.trace_id or trace.name: trace for trace in traces}
    if set(by_review_id) != set(by_trace_id):
        missing_review = sorted(set(by_trace_id) - set(by_review_id))
        missing_trace = sorted(set(by_review_id) - set(by_trace_id))
        raise ValueError(
            f"{requirement_id} frozen trace gold id set 不一致: "
            f"missing_review={missing_review}, missing_trace={missing_trace}"
        )
    errors: list[str] = []
    for trace_id, review in by_review_id.items():
        trace = by_trace_id[trace_id]
        shape = tuple(tuple(sorted(step)) for step in trace.trace)
        verdict_is_sat = review.nl_verdict in {
            "positive_satisfying",
            "boundary_satisfying",
            "SAT",
            "sat",
        }
        verdict_is_unsat = review.nl_verdict in {
            "negative_violating",
            "boundary_violating",
            "UNSAT",
            "unsat",
        }
        if review.trace_steps != shape:
            errors.append(f"{trace_id}: trace shape drift")
        if not (verdict_is_sat or verdict_is_unsat):
            errors.append(f"{trace_id}: non-Boolean frozen verdict")
        elif verdict_is_sat != trace.expected:
            errors.append(f"{trace_id}: review/evidence label drift")
        if review.obligation_ids != trace.obligation_ids:
            errors.append(f"{trace_id}: obligation binding drift")
        if review.ir_version != trace.ir_version:
            errors.append(f"{trace_id}: IR version drift")
    if errors:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} frozen trace gold 校验失败: "
            + "; ".join(errors)
        )


def _validate_semantic_evidence_gate_contract(
    *,
    requirement_id: str,
    semantic: SemanticStructure | None,
    traces: tuple[LabeledTrace, ...],
    gate: SemanticEvidenceGateReport | None,
    approved_ir: ApprovedSemanticIR | None,
    trace_coverage: TraceCoverageReport | None,
    contract_required: bool,
    path: Path,
) -> None:
    if not contract_required:
        # Allows the v3 migration tool to read the immediately preceding file
        # format once; every fully v3-written sample carries TraceCoverageReport.
        return
    if (
        semantic is None
        or not traces
        or gate is None
        or approved_ir is None
        or trace_coverage is None
    ):
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 缺少完整的 semantic "
            "evidence gate / ApprovedSemanticIR gold contract。"
        )
    frozen = freeze_approved_semantic_ir(
        replace(
            semantic,
            formula_text="",
            formula_source="none",
            formula_visibility=(),
        ),
        list(traces),
        approval_source="frozen_train_only_nl_temporal_review",
        approval_record=f"warmup-gold-{requirement_id}-semantic-approval-v1",
    )
    actual_gate = frozen.semantic_evidence_gate
    actual_approved = frozen.approved_semantic_ir
    if actual_gate is None or actual_approved is None:
        raise ValueError(f"{requirement_id} 无法重建 semantic evidence gate。")
    if actual_gate != gate:
        raise ValueError(
            f"{requirement_id} Gold SemanticEvidenceGateReport 与当前 obligation "
            "compiler 结果不一致。"
        )
    if actual_approved != approved_ir:
        raise ValueError(
            f"{requirement_id} Gold ApprovedSemanticIR 与当前 semantic evidence "
            "gate 结果不一致。"
        )
    if not gate.complete or not approved_ir.approved:
        raise ValueError(f"{requirement_id} frozen semantic evidence contract 未完成。")
    if not trace_coverage.complete or trace_coverage.actual_count != len(traces):
        raise ValueError(
            f"{requirement_id} Gold TraceCoverageReport 与 frozen trace set 不一致。"
        )


def _validate_expected_semantic_evidence_contract(
    *,
    requirement_id: str,
    semantic: SemanticStructure | None,
    expected_traces: tuple[LabeledTrace, ...],
    gold_gate: SemanticEvidenceGateReport | None,
    contract_required: bool,
    path: Path,
) -> None:
    if not contract_required:
        return
    if semantic is None or not expected_traces or gold_gate is None:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 缺少 Expected Finite "
            "Traces semantic evidence contract。"
        )
    frozen = freeze_approved_semantic_ir(
        replace(
            semantic,
            formula_text="",
            formula_source="none",
            formula_visibility=(),
        ),
        list(expected_traces),
        approval_source="frozen_train_only_expected_trace_review",
        approval_record=(
            f"warmup-gold-{requirement_id}-expected-trace-approval-v1"
        ),
    )
    gate = frozen.semantic_evidence_gate
    if gate is None or not gate.complete:
        reasons = gate.reasons if gate is not None else ("missing gate",)
        raise ValueError(
            f"{requirement_id} Expected Finite Traces 未满足统一完整性门禁: "
            + "; ".join(reasons)
        )
    if gate.required_obligation_ids != gold_gate.required_obligation_ids:
        raise ValueError(
            f"{requirement_id} Expected Finite Traces 与 Gold "
            "LabeledTraceEvidenceSet 使用了不同的 obligation contract。"
        )


def _read_gold_surface_segmentation(
    *,
    section: str,
    requirement_id: str,
    requirement: str,
    path: Path,
) -> tuple[object | None, int, str]:
    artifact_payload, metadata = _read_gold_artifact_payload(
        section,
        _GOLD_SURFACE_HEADINGS,
    )
    if artifact_payload is None:
        return None, 2, ""

    teacher_round = _validated_teacher_signal_round(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold Surface Segmentation",
        path=path,
    )
    _validate_gold_visibility(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold Surface Segmentation",
        required_stage="segmentation",
        allowed_stages={
            "segmentation",
            "plain_reading",
            "semantic_extraction",
            "offline_validation",
        },
        path=path,
    )
    payload = _artifact_body(artifact_payload)
    payload.setdefault("requirement_id", requirement_id)
    payload.setdefault("source_text", requirement)
    payload.setdefault("status", "surface_reviewed")
    payload.setdefault("experiment_mode", "warmup_fixture")
    payload.setdefault("data_split", "train")
    payload.setdefault("label_source", "warmup_fixture")
    payload.setdefault("review_basis", "curated_gold")
    payload.setdefault("persistent_rag_allowed", False)
    _validate_gold_source_hash(
        payload,
        requirement=requirement,
        requirement_id=requirement_id,
        artifact_name="Gold Surface Segmentation",
        path=path,
    )
    _prevalidate_surface_mapping(payload, requirement=requirement)

    try:
        segmentation = _surface_segmentation_from_payload(
            payload,
            requirement_id=requirement_id,
            requirement=requirement,
        )
        validation = _call_surface_validator(
            "validate_surface_segmentation",
            segmentation,
            source_text=requirement,
        )
        _assert_surface_validation_passed(validation, "surface segmentation")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 Gold Surface "
            f"Segmentation 校验失败: {exc}"
        ) from exc
    return segmentation, teacher_round, str(payload.get("schema_version", ""))


def _read_gold_plain_reading_review(
    *,
    section: str,
    requirement_id: str,
    requirement: str,
    segmentation: object | None,
    path: Path,
) -> tuple[object | None, int, str]:
    artifact_payload, metadata = _read_gold_artifact_payload(
        section,
        _GOLD_PLAIN_READING_HEADINGS,
    )
    if artifact_payload is None:
        return None, 2, ""
    if segmentation is None:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 Gold Plain Reading "
            "Review 必须引用同一样本已验证的 Gold Surface Segmentation。"
        )

    teacher_round = _validated_teacher_signal_round(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold Plain Reading Review",
        path=path,
    )
    _validate_gold_visibility(
        artifact_payload,
        metadata,
        requirement_id=requirement_id,
        artifact_name="Gold Plain Reading Review",
        required_stage="plain_reading",
        allowed_stages={
            "plain_reading",
            "semantic_extraction",
            "offline_validation",
        },
        path=path,
    )
    payload = _artifact_body(artifact_payload)
    payload.setdefault("requirement_id", requirement_id)
    payload.setdefault("segmentation_id", _artifact_value(segmentation, "segmentation_id", ""))
    payload.setdefault("review_basis", "curated_gold")
    payload.setdefault("reviewer_source", "warmup_fixture")
    payload.setdefault("experiment_mode", "warmup_fixture")
    payload.setdefault("data_split", "train")
    payload.setdefault("persistent_rag_allowed", False)
    payload = _expand_plain_reading_colon_payload(
        payload,
        segmentation=segmentation,
        requirement_id=requirement_id,
    )
    _validate_gold_source_hash(
        payload,
        requirement=requirement,
        requirement_id=requirement_id,
        artifact_name="Gold Plain Reading Review",
        path=path,
    )
    _prevalidate_plain_reading_mapping(payload, segmentation=segmentation)

    try:
        review = _call_surface_factory(
            "plain_reading_from_mapping",
            payload,
            source_text=requirement,
            segmentation=segmentation,
        )
        validation = _call_surface_validator(
            "validate_plain_reading_review",
            review,
            segmentation=segmentation,
            source_text=requirement,
        )
        _assert_surface_validation_passed(validation, "plain reading review")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 Gold Plain Reading "
            f"Review 校验失败: {exc}"
        ) from exc
    return review, teacher_round, str(payload.get("schema_version", ""))


def _read_gold_artifact_payload(
    section: str,
    headings: tuple[str, ...],
) -> tuple[dict | None, dict[str, str]]:
    body = _read_heading_body(section, headings)
    if body is None:
        return None, {}

    metadata = _read_colon_fields(body)
    fenced_blocks = re.findall(
        r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```",
        body,
        flags=re.DOTALL,
    )
    for block in fenced_blocks:
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            raise ValueError("warmup gold artifact JSON 顶层必须是 object。")
        return value, metadata

    raw_body = body.strip()
    if raw_body.startswith("{"):
        try:
            value = json.loads(raw_body)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value, metadata

    colon_source = fenced_blocks[0] if fenced_blocks else body
    colon_payload = _read_colon_fields(colon_source)
    if not colon_payload:
        raise ValueError("warmup gold artifact 必须使用 JSON 或 key: value 格式。")
    return {key: _parse_gold_scalar(value) for key, value in colon_payload.items()}, metadata


def _read_heading_body(
    section: str,
    headings: tuple[str, ...],
) -> str | None:
    names = "|".join(re.escape(heading) for heading in headings)
    match = re.search(
        rf"^(?:###)\s+(?:{names})\s*$\n?(.*?)(?=^###\s+|\Z)",
        section,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return match.group(1) if match else None


def _parse_gold_scalar(value: str) -> object:
    value = value.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value.strip("\"'")


def _artifact_body(payload: dict) -> dict:
    artifact = payload.get("artifact")
    result = dict(artifact) if isinstance(artifact, dict) else dict(payload)
    for key in ("teacher_signal_round", "visible_to", "artifact"):
        result.pop(key, None)
    return result


def _validated_teacher_signal_round(
    payload: dict,
    metadata: dict[str, str],
    *,
    requirement_id: str,
    artifact_name: str,
    path: Path,
) -> int:
    raw_value = payload.get("teacher_signal_round", metadata.get("teacher_signal_round"))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 {artifact_name} "
            "必须声明 teacher_signal_round: 2。"
        ) from exc
    if value != 2:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 {artifact_name} "
            f"teacher_signal_round 必须为 2，实际为 {value}。"
        )
    return value


def _validate_gold_visibility(
    payload: dict,
    metadata: dict[str, str],
    *,
    requirement_id: str,
    artifact_name: str,
    required_stage: str,
    allowed_stages: set[str],
    path: Path,
) -> None:
    raw_value = payload.get("visible_to", metadata.get("visible_to", ""))
    if isinstance(raw_value, (list, tuple)):
        values = {str(item).strip() for item in raw_value if str(item).strip()}
    else:
        values = {
            item.strip()
            for item in re.split(r"[, ]+", str(raw_value).strip())
            if item.strip()
        }
    if required_stage not in values:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 {artifact_name} "
            f"visible_to 必须包含 {required_stage}。"
        )
    unsupported = values - allowed_stages
    if unsupported:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 {artifact_name} "
            "visible_to 包含越阶段值: " + ", ".join(sorted(unsupported))
        )


def _validate_gold_source_hash(
    payload: dict,
    *,
    requirement: str,
    requirement_id: str,
    artifact_name: str,
    path: Path,
) -> None:
    expected = _surface_source_hash(requirement)
    actual = str(payload.get("source_hash", "")).strip()
    if not actual:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 {artifact_name} "
            "缺少 source_hash。"
        )
    if actual != expected:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 {artifact_name} "
            "source_hash 与当前 requirement 不一致。"
        )


def _surface_source_hash(source_text: str) -> str:
    from surface_review import source_hash

    return str(source_hash(source_text))


def _surface_segmentation_from_payload(
    payload: dict,
    *,
    requirement_id: str,
    requirement: str,
) -> object:
    if payload.get("marked_text") and not payload.get("segments"):
        payload = _materialize_marked_text_segments(
            payload,
            requirement_id=requirement_id,
            requirement=requirement,
        )
    return _call_surface_factory(
        "segmentation_from_mapping",
        payload,
        source_text=requirement,
        requirement_id=requirement_id,
    )


def _materialize_marked_text_segments(
    payload: dict,
    *,
    requirement_id: str,
    requirement: str,
) -> dict:
    result = dict(payload)
    marker = str(result.get("marker", "⟦CUT⟧"))
    marked_text = str(result.pop("marked_text"))
    parts = marked_text.split(marker)
    offset = 0
    segments: list[dict] = []
    for index, text in enumerate(parts, start=1):
        end = offset + len(text)
        segments.append(
            {
                "segment_id": f"seg-{index}",
                "index": index,
                "start": offset,
                "end": end,
                "text": text,
            }
        )
        offset = end
    result.setdefault("schema_version", "1")
    result.setdefault("segmentation_id", f"SEG-{requirement_id}-v1")
    result.setdefault("revision", 1)
    result.setdefault("requirement_id", requirement_id)
    result.setdefault("source_text", requirement)
    result.setdefault("source_hash_algorithm", "sha256")
    result.setdefault("offset_unit", "unicode_codepoint")
    result.setdefault("marker", marker)
    result["segments"] = segments
    result.setdefault(
        "mechanical_validation",
        {
            "status": "passed",
            "exact_reconstruction": True,
            "gap_count": 0,
            "overlap_count": 0,
            "reordered": False,
        },
    )
    return result


def _prevalidate_surface_mapping(payload: dict, *, requirement: str) -> None:
    source_text = str(payload.get("source_text", ""))
    if source_text != requirement:
        raise ValueError("Gold Surface Segmentation source_text 与 requirement 不一致。")
    marker = str(payload.get("marker", ""))
    marked_text = payload.get("marked_text")
    if marked_text is not None:
        if not marker:
            raise ValueError("Gold Surface Segmentation marker 不能为空。")
        if marker in requirement:
            raise ValueError("Gold Surface Segmentation marker 与原文冲突。")
        if str(marked_text).replace(marker, "") != requirement:
            raise ValueError("Gold Surface Segmentation 无法逐字符重构 requirement。")
        if str(marked_text).startswith(marker) or str(marked_text).endswith(marker):
            raise ValueError("Gold Surface Segmentation 不允许产生空片段。")
        if marker + marker in str(marked_text):
            raise ValueError("Gold Surface Segmentation 不允许连续 marker。")

    segments = payload.get("segments")
    if not isinstance(segments, list):
        return
    if not segments:
        raise ValueError("Gold Surface Segmentation segments 不能为空。")
    previous_end = 0
    reconstructed: list[str] = []
    ids: set[str] = set()
    for expected_index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError("Gold Surface Segmentation segment 必须是 object。")
        segment_id = str(segment.get("segment_id", ""))
        if not segment_id or segment_id in ids:
            raise ValueError("Gold Surface Segmentation segment_id 必须非空且唯一。")
        ids.add(segment_id)
        index = int(segment.get("index", 0))
        start = int(segment.get("start", -1))
        end = int(segment.get("end", -1))
        text = str(segment.get("text", ""))
        if index != expected_index:
            raise ValueError("Gold Surface Segmentation segment index 必须连续。")
        if start != previous_end or end <= start:
            raise ValueError("Gold Surface Segmentation span 必须连续且非空。")
        if end > len(requirement) or requirement[start:end] != text:
            raise ValueError("Gold Surface Segmentation span/text 与 requirement 不一致。")
        reconstructed.append(text)
        previous_end = end
    if previous_end != len(requirement) or "".join(reconstructed) != requirement:
        raise ValueError("Gold Surface Segmentation segments 未完整覆盖 requirement。")


def _prevalidate_plain_reading_mapping(
    payload: dict,
    *,
    segmentation: object,
) -> None:
    segmentation_id = str(_artifact_value(segmentation, "segmentation_id", ""))
    if str(payload.get("segmentation_id", "")) != segmentation_id:
        raise ValueError("Gold Plain Reading Review segmentation_id 与父产物不一致。")
    known_segments = {
        str(_artifact_value(segment, "segment_id", ""))
        for segment in _artifact_value(segmentation, "segments", ())
    }
    ambiguities = payload.get("ambiguities", [])
    if not isinstance(ambiguities, list):
        raise ValueError("Gold Plain Reading Review ambiguities 必须是 array。")
    for ambiguity in ambiguities:
        if not isinstance(ambiguity, dict):
            raise ValueError("Gold Plain Reading ambiguity 必须是 object。")
        source_segments = ambiguity.get("source_segment_ids", [])
        if not isinstance(source_segments, (list, tuple)) or not source_segments:
            raise ValueError("Gold Plain Reading ambiguity 必须引用 source_segment_ids。")
        unknown = {str(item) for item in source_segments} - known_segments
        if unknown:
            raise ValueError(
                "Gold Plain Reading Review 引用未知 segment_id: "
                + ", ".join(sorted(unknown))
            )
        options = ambiguity.get("options", [])
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("Gold Plain Reading ambiguity 至少需要两个 options。")
        option_ids = [str(option.get("option_id", "")) for option in options if isinstance(option, dict)]
        if len(option_ids) != len(options) or any(not value for value in option_ids):
            raise ValueError("Gold Plain Reading option_id 必须非空。")
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("Gold Plain Reading option_id 必须唯一。")
        normalized = [
            re.sub(r"\s+", " ", str(option.get("plain_interpretation", "")).strip()).lower()
            for option in options
        ]
        if any(not value for value in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("Gold Plain Reading options 必须非空且可区分。")
        decision = str(
            ambiguity.get("decision", ambiguity.get("selected_option", ""))
        ).strip()
        decision_status = str(ambiguity.get("decision_status", "")).strip()
        if decision_status == "reviewer_selected" and decision not in option_ids:
            raise ValueError("Gold Plain Reading selected option 不属于当前 options。")


def _call_surface_factory(function_name: str, primary: object, **values) -> object:
    import surface_review

    function = getattr(surface_review, function_name)
    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    kwargs = {
        key: value
        for key, value in values.items()
        if accepts_kwargs or key in signature.parameters
    }
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if positional:
        first_name = positional[0].name
        kwargs.pop(first_name, None)
        return function(primary, **kwargs)
    first = next(
        (
            parameter
            for parameter in parameters
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
            and parameter.name not in kwargs
        ),
        None,
    )
    if first is not None:
        kwargs[first.name] = primary
    return function(**kwargs)


def _call_surface_validator(function_name: str, artifact: object, **values) -> object:
    import surface_review

    if not hasattr(surface_review, function_name) and function_name == "validate_plain_reading_review":
        function_name = "validate_plain_reading"
    return _call_surface_factory(function_name, artifact, **values)


def _assert_surface_validation_passed(result: object, label: str) -> None:
    if result is False:
        raise ValueError(f"invalid {label}")
    if isinstance(result, (list, tuple)) and result:
        raise ValueError(f"invalid {label}: " + "; ".join(map(str, result)))


def _expand_plain_reading_colon_payload(
    payload: dict,
    *,
    segmentation: object,
    requirement_id: str,
) -> dict:
    if isinstance(payload.get("ambiguities"), list):
        return payload
    overall_status = str(payload.get("overall_status", "")).strip()
    if overall_status == "no_ambiguity":
        payload["ambiguities"] = []
        return payload
    option_a = str(payload.pop("option_a", "")).strip()
    option_b = str(payload.pop("option_b", "")).strip()
    ambiguity_id = str(payload.pop("ambiguity_id", "")).strip()
    if not (option_a or option_b or ambiguity_id):
        return payload
    if not option_a or not option_b or not ambiguity_id:
        raise ValueError(
            f"{requirement_id} 的 colon 格式 Plain Reading 必须同时提供 "
            "ambiguity_id、option_a 和 option_b。"
        )
    segment_ids = payload.pop("source_segment_ids", ())
    if isinstance(segment_ids, str):
        segment_ids = tuple(
            value.strip()
            for value in re.split(r"[, ]+", segment_ids)
            if value.strip()
        )
    if not segment_ids:
        segment_ids = tuple(
            str(_artifact_value(segment, "segment_id", ""))
            for segment in _artifact_value(segmentation, "segments", ())
            if _artifact_value(segment, "segment_id", "")
        )
    selected = str(
        payload.pop("selected_option", payload.pop("decision", ""))
    ).strip()
    payload["ambiguities"] = [
        {
            "ambiguity_id": ambiguity_id,
            "source_segment_ids": list(segment_ids),
            "question": str(payload.pop("question", ambiguity_id)),
            "options": [
                {
                    "option_id": "A",
                    "plain_interpretation": option_a,
                    "difference_summary": str(
                        payload.pop("option_a_difference", "option A")
                    ),
                },
                {
                    "option_id": "B",
                    "plain_interpretation": option_b,
                    "difference_summary": str(
                        payload.pop("option_b_difference", "option B")
                    ),
                },
            ],
            "decision": selected,
            "decision_status": str(
                payload.pop("decision_status", "reviewer_selected")
            ),
            "review_basis": str(payload.get("review_basis", "curated_gold")),
            "semantic_authority": str(
                payload.get("semantic_authority", "non_authoritative")
            ),
        }
    ]
    payload.setdefault("overall_status", "plain_reading_reviewed")
    return payload


def _artifact_value(artifact: object, field_name: str, default: object) -> object:
    if isinstance(artifact, dict):
        return artifact.get(field_name, default)
    return getattr(artifact, field_name, default)


def _warmup_sample_kind(
    *,
    expected_semantic: SemanticStructure | None,
    gold_surface_segmentation: object | None,
    gold_plain_reading_review: object | None,
    gold_nl_temporal_reviews: tuple[NLTemporalSimulationReview, ...] = (),
    gold_labeled_trace_evidence: tuple[LabeledTrace, ...] = (),
) -> str:
    if (
        gold_surface_segmentation is not None
        and gold_plain_reading_review is not None
    ):
        return "curated_complete"
    if expected_semantic is not None:
        provenance = (
            expected_semantic.faithful_ir.provenance
            if expected_semantic.faithful_ir is not None
            else None
        )
        if (
            bool(expected_semantic.formula_text)
            or (
                provenance is not None
                and (
                    "formula" in provenance.extractor.lower()
                    or "formula" in provenance.source.lower()
                )
            )
        ):
            return "legacy_formula_migration"
    return "new_unreviewed"


def _read_expected_semantic(
    *,
    section: str,
    requirement_id: str,
    requirement: str,
    path: Path,
) -> SemanticStructure | None:
    block = _read_heading_code_block(section, "Expected Semantic Structure")
    if not block:
        return None

    fields = _read_colon_fields(block)
    pattern = normalize_pattern(fields.get("pattern", "response") or "response")
    formula_text = normalize_formula_text(fields.get("approved_formula", ""))
    aps: tuple[str, ...] = ()
    approved_ir = _read_approved_faithful_ir(
        section=section,
        requirement_id=requirement_id,
        requirement=requirement,
        path=path,
    )
    if formula_text:
        try:
            parsed_formula = parse_learned_formula(formula_text)
            aps = collect_formula_atoms(parsed_formula)
            if approved_ir is None:
                approved_ir = reference_formula_to_ir(
                    parsed_formula,
                    requirement_id=requirement_id,
                    source_text=requirement,
                )
            elif set(aps) != set(approved_ir.ap_names):
                raise ValueError(
                    "approved_formula AP 与 Approved Faithful Semantic IR "
                    "atomic_propositions 不一致。"
                )
            closure_policy = (
                build_strict_formula_closure_policy(approved_ir)
                if approved_ir.ap_names
                else FormulaClosurePolicy(allowed_aps=())
            )
            closure_report = validate_formula_closure(
                parsed_formula,
                closure_policy,
                ir_version=(approved_ir.ir_version if approved_ir.ap_names else ""),
            )
            if not closure_report.passed:
                raise ValueError(
                    "approved_formula failed FormulaClosurePolicy: "
                    + "; ".join(closure_report.reasons)
                )
        except ValueError as exc:
            raise ValueError(
                f"小样本文件 {path} 的 {requirement_id} 中 approved_formula 无法解析: "
                f"{formula_text}。原因: {exc}"
            ) from exc
    elif approved_ir is not None:
        aps = approved_ir.ap_names

    trigger = _semantic_ap_or_fallback(
        fields.get("trigger", ""),
        fields.get("trigger_expression", ""),
        aps,
        0,
    )
    response = _semantic_ap_or_fallback(
        fields.get("response", ""),
        fields.get("response_expression", ""),
        aps,
        1 if len(aps) > 1 else 0,
    )
    components = tuple(
        (key, value)
        for key, value in fields.items()
        if key
        not in {
            "source_original_id",
            "evidence_status",
            "trigger",
            "trigger_expression",
            "response",
            "response_expression",
            "pattern",
            "scope_keyword",
            "temporal_keyword",
            "boolean_structure",
            "approved_formula",
            "explanation",
            "confidence",
            "derivation",
        }
    )

    return SemanticStructure(
        trigger=trigger,
        response=response,
        temporal_keyword=fields.get("temporal_keyword", ""),
        scope_keyword=fields.get("scope_keyword", ""),
        pattern=pattern,
        explanation=fields.get(
            "explanation",
            f"Loaded from curated warmup semantic structure {requirement_id}.",
        ),
        confidence=1.0,
        rag_context=(),
        aps=aps,
        formula_text=formula_text,
        components=components,
        faithful_ir=approved_ir,
        formula_source="human_approved",
        formula_visibility=("completion", "offline_validation"),
    )


def _read_approved_faithful_ir(
    *,
    section: str,
    requirement_id: str,
    requirement: str,
    path: Path,
):
    block = _read_heading_code_block(section, "Approved Faithful Semantic IR")
    if not block:
        return None
    try:
        payload = json.loads(block)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 "
            f"Approved Faithful Semantic IR 不是合法 JSON: {exc}"
        ) from exc
    try:
        context = build_ap_binding_context(
            requirement_id=requirement_id,
            source_text=requirement,
            representation_profile=offline_formula_assisted_profile(
                profile_id=(
                    "AP-PROFILE-warmup-offline-"
                    + requirement_id.casefold()
                ),
                provenance="train_only_warmup_offline_catalog",
            ),
            trusted_offline_catalog=tuple(
                str(item.get("name", ""))
                for item in payload.get("atomic_propositions", ())
                if isinstance(item, dict) and item.get("name")
            ),
            provenance=("train_only_warmup_offline_catalog",),
        )
        return faithful_ir_from_mapping(
            payload,
            source_text=requirement,
            requirement_id=requirement_id,
            extractor="warmup_loader",
            trusted_provenance=True,
            ap_binding_context=context,
        )
    except ValueError as exc:
        raise ValueError(
            f"小样本文件 {path} 的 {requirement_id} 中 "
            f"Approved Faithful Semantic IR 无法解析: {exc}"
        ) from exc


def _read_pattern_spec(
    *,
    section: str,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure | None,
    trace_count: int,
    path: Path,
) -> PatternSpec | None:
    if semantic is None or not semantic.formula_text:
        return None

    skeleton_block = _read_heading_code_block(section, "Expected Restricted Skeletons")
    hole_block = _read_heading_code_block(section, "Expected Hole Templates")
    final_candidate = _read_heading_code_block(section, "Expected Final Candidate")
    skeleton_template = normalize_formula_text(
        _read_field(skeleton_block, "nl_supported_formula_shape")
    )
    hole_template = normalize_formula_text(_read_field(hole_block, "template_text"))

    return PatternSpec(
        pattern=semantic.pattern,
        source_id=requirement_id,
        requirement=requirement,
        formula_text=semantic.formula_text,
        semantic_fields=semantic.components,
        derivation_rules=tuple(_read_derivation_rules(section)),
        trace_count=trace_count,
        skeleton_templates=(skeleton_template,) if skeleton_template else (),
        hole_template_text=hole_template,
        final_candidate_text=normalize_formula_text(final_candidate.strip())
        or semantic.formula_text,
    )


def _read_guided_learning_context(
    *,
    section: str,
    requirement_id: str,
    requirement: str,
    semantic: SemanticStructure | None,
    traces: tuple[LabeledTrace, ...],
    pattern_spec: PatternSpec | None,
    gold_surface_segmentation: object | None = None,
    gold_plain_reading_review: object | None = None,
    gold_nl_temporal_reviews: tuple[NLTemporalSimulationReview, ...] = (),
    gold_labeled_trace_evidence: tuple[LabeledTrace, ...] = (),
    surface_teacher_round: int = 2,
    plain_teacher_round: int = 2,
) -> GuidedLearningContext | None:
    skeleton_block = _read_heading_code_block(section, "Expected Restricted Skeletons")
    hole_block = _read_heading_code_block(section, "Expected Hole Templates")
    completion_block = _read_heading_code_block(section, "Expected Completion And Consistency")
    final_candidate = normalize_formula_text(
        _read_heading_code_block(section, "Expected Final Candidate").strip()
    )
    final_approval_record = _read_final_approval_record(section)
    artifact_visibility = _read_artifact_visibility(section)
    approved_formula = (
        semantic.formula_text
        if semantic is not None and semantic.formula_text
        else final_candidate
    )
    expected_skeletons = _read_expected_skeleton_texts(skeleton_block)
    expected_hole_template = normalize_formula_text(
        _read_field(hole_block, "template_text")
    )
    if not expected_hole_template and pattern_spec is not None:
        expected_hole_template = pattern_spec.hole_template_text
    approved_assignments = _read_approved_assignments(completion_block)
    approved_ir = semantic.faithful_ir if semantic is not None else None
    trace_plan = (
        compile_review_trace_plan(approved_ir) if approved_ir is not None else None
    )
    skeleton_plan = (
        compile_constrained_skeleton_plan(approved_ir) if approved_ir is not None else None
    )
    hole_plan = (
        compile_completion_hole_plan(approved_ir, skeleton_plan)
        if approved_ir is not None and skeleton_plan is not None
        else None
    )
    context = GuidedLearningContext(
        requirement_id=requirement_id,
        requirement=requirement,
        process_text=section,
        expected_semantic=semantic,
        expected_traces=traces,
        gold_nl_temporal_reviews=gold_nl_temporal_reviews,
        gold_labeled_trace_evidence=gold_labeled_trace_evidence,
        expected_restricted_skeletons=tuple(expected_skeletons),
        expected_hole_template=expected_hole_template,
        expected_completion=completion_block,
        expected_final_candidate=final_candidate,
        approved_formula=approved_formula,
        approved_assignments=tuple(approved_assignments),
        approved_ir=approved_ir,
        trace_obligation_plan=trace_plan,
        skeleton_plan=skeleton_plan,
        completion_hole_plan=hole_plan,
        final_approval_record=final_approval_record,
        artifact_visibility=artifact_visibility,
        gold_surface_segmentation=gold_surface_segmentation,
        gold_plain_reading_review=gold_plain_reading_review,
        surface_segmentation_teacher_signal_round=surface_teacher_round,
        plain_reading_teacher_signal_round=plain_teacher_round,
    )
    return context if context.enabled else None


def _read_expected_skeleton_texts(block: str) -> list[str]:
    if not block:
        return []
    skeletons: list[str] = []
    shape = normalize_formula_text(_read_field(block, "nl_supported_formula_shape"))
    if shape:
        skeletons.append(shape)
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("note:", "#")):
            continue
        if line.startswith("- "):
            value = normalize_formula_text(line[2:])
            if value and value not in skeletons:
                skeletons.append(value)
    return skeletons


def _read_approved_assignments(block: str) -> list[tuple[str, str]]:
    value = _read_field(block, "approved_assignments")
    if not value or value.lower() == "none_unresolved":
        return []
    assignments: list[tuple[str, str]] = []
    for item in re.split(r"[,;]", value):
        item = item.strip()
        if not item or "=" not in item:
            continue
        hole_id, operator = item.split("=", 1)
        hole_id = hole_id.strip()
        operator = operator.strip()
        if hole_id and operator:
            assignments.append((hole_id, operator))
    return assignments


def _semantic_ap_or_fallback(
    preferred: str,
    expression: str,
    aps: tuple[str, ...],
    fallback_index: int,
) -> str:
    preferred = preferred.strip()
    if AP_PATTERN.fullmatch(preferred):
        return preferred
    expression = expression.strip()
    if AP_PATTERN.fullmatch(expression):
        return expression
    if aps:
        return aps[min(fallback_index, len(aps) - 1)]
    return "unknown_ap"


def _read_colon_fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block.splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$", line.strip())
        if match:
            fields[match.group(1)] = match.group(2).strip()
    return fields


def _read_derivation_rules(section: str) -> list[str]:
    block = _read_heading_code_block(section, "Expected Semantic Structure")
    if not block:
        return []
    rules: list[str] = []
    in_derivation = False
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if line == "derivation:":
            in_derivation = True
            continue
        if in_derivation and line.startswith("- "):
            rules.append(line[2:].strip())
    return rules


def _read_expected_finite_traces(
    section: str,
    requirement_id: str,
    path: Path,
) -> list[LabeledTrace]:
    block = _read_heading_code_block(section, "Expected Finite Traces")
    if not block:
        return []

    traces: list[LabeledTrace] = []
    current_header: tuple[str, str] | None = None
    current_steps: dict[int, set[str]] = {}
    current_expected: bool | None = None
    current_trace_id = ""
    current_ir_version = ""
    current_obligation_ids: tuple[str, ...] = ()
    current_covered_nodes: tuple[str, ...] = ()
    current_trace_role = ""
    current_boundary_role = ""

    def flush_current() -> None:
        nonlocal current_header, current_steps, current_expected
        nonlocal current_trace_id, current_ir_version, current_obligation_ids
        nonlocal current_covered_nodes, current_trace_role, current_boundary_role
        if current_header is None:
            return
        kind, name = current_header
        if current_expected is None:
            raise ValueError(
                f"小样本文件 {path} 的 {requirement_id} 中轨迹 {name} 缺少 expected 字段。"
            )
        if not current_steps:
            raise ValueError(
                f"小样本文件 {path} 的 {requirement_id} 中轨迹 {name} 缺少 t0/t1 时间步。"
            )
        max_index = max(current_steps)
        trace = [current_steps.get(index, set()) for index in range(max_index + 1)]
        traces.append(
            LabeledTrace(
                name=name,
                kind=kind,
                trace=trace,
                expected=current_expected,
                reason=f"Loaded from curated warmup example {requirement_id}.",
                trace_id=current_trace_id or name,
                ir_version=current_ir_version,
                obligation_ids=current_obligation_ids,
                covered_ir_node_ids=current_covered_nodes,
                trace_role=current_trace_role,
                boundary_role=current_boundary_role,
                generation_source=(
                    "expected_finite_trace_migration"
                    if current_obligation_ids
                    else ""
                ),
            )
        )
        current_header = None
        current_steps = {}
        current_expected = None
        current_trace_id = ""
        current_ir_version = ""
        current_obligation_ids = ()
        current_covered_nodes = ()
        current_trace_role = ""
        current_boundary_role = ""

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header_match = re.match(r"^(positive|negative|boundary)\s+([A-Za-z_][A-Za-z0-9_-]*):$", line)
        if header_match:
            flush_current()
            current_header = (header_match.group(1), header_match.group(2))
            continue

        if current_header is None:
            continue

        step_match = re.match(r"^t(\d+)\s*:\s*(.*?)\s*$", line)
        if step_match:
            time_index = int(step_match.group(1))
            current_steps[time_index] = _parse_trace_step(
                step_match.group(2),
                requirement_id,
                path,
            )
            continue

        expected_match = re.match(r"^expected\s*:\s*(true|false)\s*$", line, flags=re.IGNORECASE)
        if expected_match:
            current_expected = expected_match.group(1).lower() == "true"
            continue

        metadata_match = re.match(
            r"^(trace_id|ir_version|obligation_ids|covered_ir_node_ids|"
            r"trace_role|boundary_role)\s*:\s*(.*?)\s*$",
            line,
        )
        if metadata_match:
            field_name, raw_value = metadata_match.groups()
            if field_name == "trace_id":
                current_trace_id = raw_value
            elif field_name == "ir_version":
                current_ir_version = raw_value
            elif field_name == "obligation_ids":
                current_obligation_ids = tuple(
                    item.strip() for item in raw_value.split(",") if item.strip()
                )
            elif field_name == "covered_ir_node_ids":
                current_covered_nodes = tuple(
                    item.strip() for item in raw_value.split(",") if item.strip()
                )
            elif field_name == "trace_role":
                current_trace_role = raw_value
            elif field_name == "boundary_role":
                current_boundary_role = raw_value
            continue

    flush_current()
    return traces


def _read_heading_code_block(section: str, heading: str) -> str:
    match = re.search(
        rf"^###\s+{re.escape(heading)}\s*$"
        r"\s*(?:visible_to\s*:\s*.*?\s*)?"
        r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```",
        section,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _read_final_approval_record(section: str) -> dict | None:
    block = _read_heading_code_block(section, "FinalApprovalRecord")
    if not block:
        return None
    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        return {
            "status": "final",
            "approval_source": "warmup_fixture",
            "raw_record": block,
        }
    return payload if isinstance(payload, dict) else None


def _read_artifact_visibility(
    section: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    visibility = {name: tuple(values) for name, values in DEFAULT_ARTIFACT_VISIBILITY}
    heading_to_artifact = {
        "Requirement": "natural_language",
        "Natural Language": "natural_language",
        "Gold Surface Segmentation": "surface_segmentation",
        "Gold SurfaceSegmentation": "surface_segmentation",
        "Gold Plain Reading Review": "plain_reading_review",
        "Gold PlainReadingReview": "plain_reading_review",
        "Gold AmbiguityAxisSet": "ambiguity_axis_set",
        "Gold CandidateIRSet": "candidate_ir_set",
        "Gold NLTemporalSimulationReview": "nl_temporal_simulation_review",
        "Gold LabeledTraceEvidenceSet": "labeled_trace_evidence_set",
        "Gold TraceCoverageReport": "labeled_trace_evidence_set",
        "Gold SemanticEvidenceGateReport": "approved_semantic_ir",
        "Expected Finite Traces": "labeled_trace_evidence_set",
        "Gold ApprovedSemanticIR": "approved_semantic_ir",
        "Approved Faithful Semantic IR": "approved_semantic_ir",
        "Expected Semantic Structure": "approved_semantic_ir",
        "Gold TraceObligationPlan": "trace_obligation_plan",
        "Gold ConstrainedSkeletonPlan": "constrained_skeleton_plan",
        "Expected Restricted Skeletons": "constrained_skeleton_plan",
        "Gold CompletionHolePlan": "completion_hole_plan",
        "Expected Hole Templates": "completion_hole_plan",
        "Gold Completion": "completion",
        "Expected Completion And Consistency": "completion",
        "Gold Final Formula": "final_formula",
        "Expected Final Candidate": "final_formula",
        "FinalApprovalRecord": "final_approval_record",
    }
    for match in re.finditer(
        r"^###\s+(.+?)\s*$\s*visible_to\s*:\s*(.+?)\s*$",
        section,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        heading = match.group(1).strip()
        artifact = heading_to_artifact.get(heading)
        if artifact is None:
            continue
        values = tuple(
            item.strip()
            for item in re.split(r"[, ]+", match.group(2).strip())
            if item.strip()
        )
        if values:
            visibility[artifact] = values
    return tuple((name, visibility[name]) for name in visibility)


def _parse_trace_step(
    raw_value: str,
    requirement_id: str,
    path: Path,
) -> set[str]:
    value = raw_value.strip()
    if not value:
        return set()

    aps: set[str] = set()
    for raw_ap in value.split(","):
        ap = raw_ap.strip()
        if not AP_PATTERN.fullmatch(ap):
            raise ValueError(
                f"小样本文件 {path} 的 {requirement_id} 中轨迹 AP 非法: {ap}。"
            )
        aps.add(ap)
    return aps


def _validate_semantic_response_contract(
    section: str,
    requirement_id: str,
    path: Path,
) -> None:
    response_matches = list(
        re.finditer(
            r"^response\s*:\s*(.+?)\s*$",
            section,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    if not response_matches:
        return

    response_expression = _read_field(section, "response_expression")
    response_definition = _read_field(section, "response_definition")
    for response_match in response_matches:
        response = response_match.group(1).strip()
        if not AP_PATTERN.fullmatch(response):
            raise ValueError(
                f"小样本文件 {path} 的 {requirement_id} 中 semantic.response "
                "必须是合法 AP 名称，只能使用英文字母、数字和下划线，且不能以数字开头。"
            )

        if not response_expression or _same_formula_text(response_expression, response):
            continue

        if not response_definition:
            raise ValueError(
                f"小样本文件 {path} 的 {requirement_id} 中 response_expression "
                "不是原子 AP 时，必须提供 response_definition 绑定合法 response 别名。"
            )

        definition_parts = response_definition.split(":=", 1)
        if len(definition_parts) != 2:
            raise ValueError(
                f"小样本文件 {path} 的 {requirement_id} 中 response_definition "
                "必须使用 '<response> := <expression>' 格式。"
            )
        definition_name, definition_expression = (
            definition_parts[0].strip(),
            definition_parts[1].strip(),
        )
        if definition_name != response:
            raise ValueError(
                f"小样本文件 {path} 的 {requirement_id} 中 response_definition "
                f"左侧必须等于 response: {response}。"
            )
        if not _same_formula_text(definition_expression, response_expression):
            raise ValueError(
                f"小样本文件 {path} 的 {requirement_id} 中 response_definition "
                "右侧必须等于 response_expression，避免别名和推导公式脱节。"
            )


def _read_field(section: str, field_name: str) -> str:
    field_match = re.search(
        rf"^{re.escape(field_name)}\s*:\s*(.+?)\s*$",
        section,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return field_match.group(1).strip() if field_match else ""


def _same_formula_text(left: str, right: str) -> bool:
    return re.sub(r"\s+", "", left) == re.sub(r"\s+", "", right)
