"""Guided warmup learning context.

Warmup examples carry expected artifacts for every stage. This module keeps
that teacher signal in one small structure so the generation path can receive it
without confusing it with ordinary RAG examples used by main requirements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from data_examples import LabeledTrace
from ltlf_parser import Formula, parse_formula
from ltlf_syntax import normalize_boolean_operator_text
from schemas import SemanticStructure
from schemas import (
    CompletionHolePlan,
    ConstrainedSkeletonPlan,
    FaithfulSemanticIR,
    PlainReadingReview,
    NLTemporalSimulationReview,
    SurfaceSegmentation,
    TraceObligationPlan,
)
from trace_formatter import trace_to_serializable


DEFAULT_ARTIFACT_VISIBILITY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("natural_language", ("all",)),
    (
        "surface_segmentation",
        ("plain_reading", "semantic_extraction", "offline_validation"),
    ),
    (
        "plain_reading_review",
        ("semantic_extraction", "offline_validation"),
    ),
    ("ambiguity_axis_set", ("semantic_extraction",)),
    ("candidate_ir_set", ("semantic_extraction",)),
    ("nl_temporal_simulation_review", ("trace_annotation", "offline_validation")),
    (
        "labeled_trace_evidence_set",
        ("trace_query_generation", "skeleton", "completion", "selection"),
    ),
    (
        "approved_semantic_ir",
        ("trace_query_generation", "skeleton", "completion", "selection"),
    ),
    ("trace_obligation_plan", ("trace_query_generation", "selection")),
    ("constrained_skeleton_plan", ("skeleton", "completion")),
    ("completion_hole_plan", ("completion",)),
    ("completion", ("completion", "offline_validation")),
    ("final_formula", ("offline_validation",)),
    ("final_approval_record", ("selection", "offline_validation")),
)


_STAGE_ALIASES = {
    "surface_segmentation": "segmentation",
    "segmentation_generation": "segmentation",
    "plain_reading_review": "plain_reading",
    "plain_reading_generation": "plain_reading",
    "trace_generation": "trace_query_generation",
    "formula_text_trace_generation": "trace_query_generation",
    "restricted_skeleton_generation": "skeleton",
    "hole_candidate_generation": "completion",
    "temporal_hole_completion": "completion",
    "semantic": "semantic_extraction",
    "traces": "trace_query_generation",
    "trace": "trace_query_generation",
    "template": "hole_candidate_generation",
    "consistency": "selection",
}


@dataclass(frozen=True)
class GuidedLearningContext:
    """Sealed teacher bundle used only by warmup stage-guided generation.

    The loader owns the complete bundle.  Agents must receive an ``attempt_view``
    rather than the complete object: attempt 1 contains no teacher answer, while
    attempt 2 exposes only the target artifact for the current stage.
    """

    requirement_id: str
    requirement: str
    process_text: str = ""
    expected_semantic: SemanticStructure | None = None
    expected_traces: tuple[LabeledTrace, ...] = ()
    gold_nl_temporal_reviews: tuple[NLTemporalSimulationReview, ...] = ()
    gold_labeled_trace_evidence: tuple[LabeledTrace, ...] = ()
    expected_restricted_skeletons: tuple[str, ...] = ()
    expected_hole_template: str = ""
    expected_completion: str = ""
    expected_final_candidate: str = ""
    approved_formula: str = ""
    approved_assignments: tuple[tuple[str, str], ...] = ()
    approved_ir: FaithfulSemanticIR | None = None
    trace_obligation_plan: TraceObligationPlan | None = None
    skeleton_plan: ConstrainedSkeletonPlan | None = None
    completion_hole_plan: CompletionHolePlan | None = None
    final_approval_record: dict | None = None
    artifact_visibility: tuple[tuple[str, tuple[str, ...]], ...] = (
        DEFAULT_ARTIFACT_VISIBILITY
    )
    gold_surface_segmentation: SurfaceSegmentation | None = None
    gold_plain_reading_review: PlainReadingReview | None = None
    surface_segmentation_teacher_signal_round: int = 2
    plain_reading_teacher_signal_round: int = 2
    learning_mode: str = "trajectory_guided_warmup_v2"
    attempt_index: int = 1
    teacher_stage: str = ""

    def attempt_view(self, stage: str, attempt: int) -> "GuidedLearningContext":
        """Return a stage-scoped view that enforces the two-attempt contract."""

        if attempt not in {1, 2}:
            raise ValueError("warmup guided attempt must be 1 or 2")
        visible_stage = _STAGE_ALIASES.get(stage, stage)
        sealed = replace(
            self,
            process_text="",
            expected_semantic=None,
            expected_traces=(),
            gold_nl_temporal_reviews=(),
            gold_labeled_trace_evidence=(),
            expected_restricted_skeletons=(),
            expected_hole_template="",
            expected_completion="",
            expected_final_candidate="",
            approved_formula="",
            approved_assignments=(),
            approved_ir=None,
            trace_obligation_plan=None,
            skeleton_plan=None,
            completion_hole_plan=None,
            final_approval_record=None,
            attempt_index=attempt,
            teacher_stage=visible_stage if attempt == 2 else "",
        )
        if attempt == 1:
            return sealed

        if visible_stage == "segmentation":
            return replace(
                sealed,
                gold_surface_segmentation=self.gold_surface_segmentation,
            )
        if visible_stage == "plain_reading":
            return replace(
                sealed,
                gold_surface_segmentation=self.gold_surface_segmentation,
                gold_plain_reading_review=self.gold_plain_reading_review,
            )
        if visible_stage == "semantic_extraction":
            return replace(
                sealed,
                expected_semantic=self.expected_semantic,
                approved_ir=self.approved_ir,
            )
        if visible_stage == "trace_query_generation":
            return replace(
                sealed,
                expected_traces=(
                    self.gold_labeled_trace_evidence or self.expected_traces
                ),
                approved_ir=self.approved_ir,
                trace_obligation_plan=self.trace_obligation_plan,
            )
        if visible_stage == "trace_annotation":
            return replace(
                sealed,
                gold_nl_temporal_reviews=self.gold_nl_temporal_reviews,
            )
        if visible_stage == "skeleton":
            return replace(
                sealed,
                expected_restricted_skeletons=self.expected_restricted_skeletons,
                approved_ir=self.approved_ir,
                skeleton_plan=self.skeleton_plan,
            )
        if visible_stage == "hole_candidate_generation":
            return replace(
                sealed,
                expected_hole_template=self.expected_hole_template,
                completion_hole_plan=self.completion_hole_plan,
            )
        if visible_stage == "completion":
            return replace(
                sealed,
                expected_completion=self.expected_completion,
                expected_final_candidate=self.expected_final_candidate,
                approved_formula=self.approved_formula,
                approved_assignments=self.approved_assignments,
                approved_ir=self.approved_ir,
                completion_hole_plan=self.completion_hole_plan,
            )
        if visible_stage == "selection":
            return replace(
                sealed,
                expected_final_candidate=self.expected_final_candidate,
                final_approval_record=self.final_approval_record,
            )
        if visible_stage == "offline_validation":
            return replace(
                sealed,
                expected_final_candidate=self.expected_final_candidate,
                approved_formula=self.approved_formula,
                final_approval_record=self.final_approval_record,
            )
        return sealed

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.expected_semantic is not None,
                self.gold_surface_segmentation is not None,
                self.gold_plain_reading_review is not None,
                bool(self.expected_traces),
                bool(self.gold_nl_temporal_reviews),
                bool(self.gold_labeled_trace_evidence),
                bool(self.expected_restricted_skeletons),
                bool(self.expected_hole_template),
                bool(self.expected_completion),
                bool(self.expected_final_candidate),
                bool(self.approved_formula),
                self.final_approval_record is not None,
            )
        )

    def expected_final_formula(self) -> Formula | None:
        """Return the expected final candidate formula if it is parseable."""

        text = self.expected_final_candidate or self.approved_formula
        if not text:
            return None
        return parse_formula(normalize_boolean_operator_text(text))

    def prompt_payload(self, stage: str, round_index: int | None = None) -> dict:
        """Compact JSON-serializable teacher signal for one agent prompt."""

        if round_index is None:
            round_index = self.attempt_index
        if round_index < 1:
            raise ValueError("guided round_index must be at least 1")
        visible_stage = _STAGE_ALIASES.get(stage, stage)
        payload = {
            "mode": self.learning_mode,
            "stage": stage,
            "visible_stage": visible_stage,
            "round_index": round_index,
            "attempt": round_index,
            "teacher_signal_exposed": bool(
                round_index >= 2
                and (not self.teacher_stage or self.teacher_stage == visible_stage)
            ),
            "requirement_id": self.requirement_id,
            "policy": (
                "Use only the artifacts explicitly exposed for this stage. "
                "Upstream semantic and trace-query stages must not receive a "
                "formula answer or formula-oracle truth labels."
            ),
            "visible_artifacts": self.visible_artifacts(stage, round_index),
        }
        if (
            visible_stage == "segmentation"
            and self.gold_surface_segmentation is not None
            and round_index >= self.surface_segmentation_teacher_signal_round
        ):
            payload["gold_surface_segmentation"] = _artifact_payload(
                self.gold_surface_segmentation
            )
        if visible_stage == "plain_reading":
            if self.gold_surface_segmentation is not None:
                # The segmentation is an already validated upstream artifact at
                # this point, not the current-stage teacher answer.
                payload["surface_segmentation"] = _artifact_payload(
                    self.gold_surface_segmentation
                )
            if (
                self.gold_plain_reading_review is not None
                and round_index >= self.plain_reading_teacher_signal_round
            ):
                payload["gold_plain_reading_review"] = _artifact_payload(
                    self.gold_plain_reading_review
                )
        if (
            visible_stage == "semantic_extraction"
            and round_index >= 2
            and self.expected_semantic is not None
        ):
            payload["expected_semantic"] = {
                "aps": list(self.expected_semantic.aps),
                "faithful_ir": (
                    asdict(self.approved_ir or self.expected_semantic.faithful_ir)
                    if (self.approved_ir or self.expected_semantic.faithful_ir)
                    else None
                ),
            }
        if visible_stage == "semantic_extraction":
            # Early gold is exposed downstream only as validated upstream input;
            # the gold-prefixed teacher keys never cross this stage boundary.
            if self.gold_surface_segmentation is not None:
                payload["surface_segmentation"] = _artifact_payload(
                    self.gold_surface_segmentation
                )
            if self.gold_plain_reading_review is not None:
                payload["plain_reading_review"] = _artifact_payload(
                    self.gold_plain_reading_review
                )
        if (
            visible_stage == "trace_query_generation"
            and round_index >= 2
            and self.expected_traces
        ):
            if self.approved_ir is not None:
                payload["faithful_ir"] = asdict(self.approved_ir)
            if self.trace_obligation_plan is not None:
                payload["trace_obligation_plan"] = asdict(self.trace_obligation_plan)
            # Trace-query generation may see query shapes, but never labels.
            payload["expected_traces"] = [
                {
                    "name": trace.name,
                    "trace": trace_to_serializable(trace.trace),
                    "query_role": trace.query_role or trace.role or trace.kind,
                    "target_ambiguity": trace.target_ambiguity,
                    "obligation_ids": list(trace.obligation_ids),
                }
                for trace in self.expected_traces
            ]
        if visible_stage == "skeleton" and round_index >= 2:
            if self.approved_ir is not None:
                payload["faithful_ir"] = asdict(self.approved_ir)
            if self.skeleton_plan is not None:
                payload["constrained_skeleton_plan"] = asdict(self.skeleton_plan)
        if visible_stage == "hole_candidate_generation" and round_index >= 2:
            if self.expected_hole_template:
                payload["expected_hole_template"] = self.expected_hole_template
            if self.completion_hole_plan is not None:
                payload["completion_hole_plan"] = asdict(self.completion_hole_plan)
        if visible_stage == "completion" and round_index >= 2:
            if self.approved_ir is not None:
                payload["faithful_ir"] = asdict(self.approved_ir)
            if self.completion_hole_plan is not None:
                payload["completion_hole_plan"] = asdict(self.completion_hole_plan)
            payload["approved_assignments"] = [
                {"hole_id": hole_id, "operator": operator}
                for hole_id, operator in self.approved_assignments
            ]
            if self.expected_completion:
                payload["expected_completion"] = self.expected_completion
        if (
            visible_stage == "selection"
            and round_index >= 2
            and self.final_approval_record is not None
        ):
            payload["final_approval_record"] = dict(self.final_approval_record)
        if visible_stage == "offline_validation":
            payload["approved_formula"] = self.approved_formula
            payload["expected_final_candidate"] = self.expected_final_candidate
            if self.final_approval_record is not None:
                payload["final_approval_record"] = dict(self.final_approval_record)
        return payload

    def visible_artifacts(self, stage: str, round_index: int = 1) -> list[str]:
        if round_index < 1:
            raise ValueError("guided round_index must be at least 1")
        visible_stage = _STAGE_ALIASES.get(stage, stage)
        artifacts: list[str] = []
        for name, visible_to in self.artifact_visibility:
            if "all" in visible_to or visible_stage in visible_to:
                artifacts.append(name)
        if (
            visible_stage == "segmentation"
            and self.gold_surface_segmentation is not None
            and round_index >= self.surface_segmentation_teacher_signal_round
        ):
            artifacts.append("gold_surface_segmentation")
        if (
            visible_stage == "plain_reading"
            and self.gold_plain_reading_review is not None
            and round_index >= self.plain_reading_teacher_signal_round
        ):
            artifacts.append("gold_plain_reading_review")
        return artifacts


def _artifact_payload(artifact: object) -> object:
    """Return a JSON-serializable copy without coupling to one artifact module."""

    if hasattr(artifact, "__dataclass_fields__"):
        return asdict(artifact)
    if isinstance(artifact, dict):
        return dict(artifact)
    return artifact


def guided_prompt_rule(stage: str) -> str:
    return (
        f"当前是 warmup 样本学习的 {stage} 阶段；只能使用 "
        "payload.guided_learning 中为本阶段显式暴露的字段。Semantic 和 "
        "Trace 查询阶段不得推断或请求 approved_formula、"
        "expected_final_candidate 或 formula-oracle 标签。"
    )
