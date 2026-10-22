"""
候选公式补全 Agent。

该阶段把受限骨架扩展为带孔候选公式，并生成 X/F/G 补全结果。
rules 模式使用可解释枚举；api 模式让 LLM 生成结构化孔位和补全公式，随后
仍由 TrajectoryConsistencyAgent 统一模拟轨迹并评分。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import re

from agents.base_agent import BaseAgent, JSONLLMClient
from agents.llm_parsing import (
    coerce_list_field,
    completion_operator_atoms,
    normalize_operator,
    normalize_operator_list,
    optional_string,
    require_list,
    require_mapping,
    require_string,
)
from agents.prompt_context import (
    FULL_PROCESS_SOURCE,
    KB_LTLF_SEMANTICS,
    KB_OPERATOR_COMPLETION,
    build_stage_prompt_context,
    structured_system_prompt,
)
from data_examples import LabeledTrace
from guided_learning import GuidedLearningContext, guided_prompt_rule
from formula_closure import (
    FormulaClosurePolicy,
    build_strict_formula_closure_policy,
    validate_formula_closure,
)
from ltlf_parser import F, G, Formula, Implies, X, parse_formula
from ltlf_syntax import (
    SUPPORTED_LTLF_FORMULA_GUIDANCE,
    has_repeated_negation,
    normalize_boolean_operator_text,
    validate_canonical_boolean_operators,
    validate_no_forbidden_temporal_operators,
    validate_no_repeated_negation,
)
from pattern_registry import (
    get_pattern_handler_for_semantic,
    learned_pattern_direct_execution_allowed,
)
from rag.example_memory import (
    GUIDED_COMPLETION_SOURCE,
    GUIDED_FULL_PROCESS_SOURCE,
    GUIDED_TEMPLATE_SOURCE,
    OPERATOR_COMPLETION_SOURCE,
    TEMPLATE_SOURCE,
)
from schemas import (
    HoleCandidate,
    SemanticStructure,
    TemporalCompletion,
    TemporalHole,
)
from temporal_domain import finite_requirement_for_pipeline
from temporal_language import temporal_readings_in_text
from faithful_ir import precedence_roles_from_semantic
from ir_compiler import (
    compile_completion_hole_plan,
    compile_constrained_skeleton_plan,
    compile_ir_skeleton,
    compile_ir_to_formula,
    render_ir_hole_template,
)
from ir_trace_semantics import IRTraceSemanticsError, evaluate_ir_trace
from utils import format_trace, format_truth


MIN_HOLE_CANDIDATE_TEMPLATES = 5
COMPLETION_MACRO_SCHEMAS = {
    "NEXT_CHAIN": {
        "assignment_shape": "X+",
        "expander": "deterministic repeated strong-next",
        "assignment_allowed": True,
    },
    "PERIODIC": {
        "required_ir_parameters": (
            "period",
            "mode",
            "origin",
            "terminal_policy",
        ),
        "expander": "ir_compiler._compile_periodicity_relation",
        "assignment_allowed": False,
    },
    "CARDINALITY": {
        "required_ir_parameters": ("mode", "count"),
        "expander": "ir_compiler._cardinality",
        "assignment_allowed": False,
    },
}


def _generation_ir_branches(semantic: SemanticStructure) -> list[tuple[str, object]]:
    candidate_set = semantic.candidate_ir_set
    if (
        candidate_set is not None
        and len(candidate_set.branches) > 1
        and not (
            semantic.approved_semantic_ir is not None
            and semantic.approved_semantic_ir.approved
        )
    ):
        return [
            (branch.ir_branch_id, branch.faithful_ir)
            for branch in candidate_set.branches
        ]
    if semantic.faithful_ir is None:
        return []
    return [("deterministic", semantic.faithful_ir)]


def _materialize_ir_temporal_assignments(
    candidate: HoleCandidate,
) -> list[HoleCandidate]:
    """Expand approved hole domains without asking an LLM to choose a field."""

    domains = tuple(hole.allowed_ops or ("epsilon",) for hole in candidate.holes)
    combinations = tuple(itertools.product(*domains)) if domains else ((),)
    materialized: list[HoleCandidate] = []
    for index, operators in enumerate(combinations, start=1):
        holes = tuple(
            replace(
                hole,
                allowed_ops=(operator,),
                reason=(
                    f"{hole.reason} Materialized as one trajectory-testable choice."
                ).strip(),
            )
            for hole, operator in zip(candidate.holes, operators)
        )
        suffix = f"-a{index:03d}" if len(combinations) > 1 else ""
        materialized.append(
            replace(
                candidate,
                holes=holes,
                candidate_id=f"{candidate.candidate_id}{suffix}",
                explanation=(
                    f"{candidate.explanation} Assignment {index}/{len(combinations)} "
                    "is retained only if complete-formula trajectory checks pass."
                ).strip(),
            )
        )
    return materialized


@dataclass(frozen=True)
class ReviewedTraceIRIssue:
    """Formula-free evidence that a reviewed trace and its approved IR disagree."""

    trace_id: str
    issue_type: str
    message: str
    expected: bool | None = None
    ir_actual: bool | None = None
    covered_ir_node_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "issue_type": self.issue_type,
            "message": self.message,
            "expected": self.expected,
            "ir_actual": self.ir_actual,
            "covered_ir_node_ids": list(self.covered_ir_node_ids),
            "obligation_ids": list(self.obligation_ids),
        }


class ReviewedTraceIRValidationError(ValueError):
    """Aggregate all deterministic IR/read-only trace contract failures."""

    pipeline_stage = "completion"

    def __init__(self, issues: list[ReviewedTraceIRIssue]):
        self.issues = tuple(issues)
        self.consistency_issues = [issue.to_dict() for issue in self.issues]
        self.affected_ir_node_ids = tuple(
            dict.fromkeys(
                node_id
                for issue in self.issues
                for node_id in issue.covered_ir_node_ids
            )
        )
        self.affected_obligation_ids = tuple(
            dict.fromkeys(
                obligation_id
                for issue in self.issues
                for obligation_id in issue.obligation_ids
            )
        )
        has_unexecutable = any(
            issue.issue_type == "unexecutable_ir" for issue in self.issues
        )
        self.diagnosis_error_class = (
            "ir_error" if has_unexecutable else "ir_ambiguity_state_mismatch"
        )
        prefix = (
            "blocked_by_unexecutable_ir"
            if has_unexecutable
            else "trace_label_ir_mismatch"
        )
        rendered = " | ".join(
            (
                f"trace={issue.trace_id}, type={issue.issue_type}, "
                f"label={issue.expected}, ir={issue.ir_actual}, "
                f"detail={issue.message}"
            )
            for issue in self.issues
        )
        super().__init__(
            f"{prefix}: responsible_stage="
            + (
                "semantic_ir"
                if has_unexecutable
                else "semantic_approval_or_trace_adjudication"
            )
            + f"; issue_count={len(self.issues)}; {rendered}"
        )


def _validate_reviewed_traces_against_deterministic_ir(ir, traces) -> None:
    issues: list[ReviewedTraceIRIssue] = []
    for trace in traces:
        if (
            not trace.annotation_locked
            or trace.label_source not in {"independent_judge", "human"}
            or trace.ir_version != ir.ir_version
            or ir.root_node_id not in trace.covered_ir_node_ids
        ):
            continue
        try:
            ir_expected = evaluate_ir_trace(ir, trace.trace)
        except IRTraceSemanticsError as exc:
            issues.append(
                ReviewedTraceIRIssue(
                    trace_id=trace.trace_id or trace.name,
                    issue_type="unexecutable_ir",
                    message=str(exc),
                    expected=trace.expected,
                    covered_ir_node_ids=tuple(trace.covered_ir_node_ids),
                    obligation_ids=tuple(trace.obligation_ids),
                )
            )
            continue
        if ir_expected != trace.expected:
            issues.append(
                ReviewedTraceIRIssue(
                    trace_id=trace.trace_id or trace.name,
                    issue_type="label_ir_mismatch",
                    message="reviewed label differs from deterministic IR evaluation",
                    expected=trace.expected,
                    ir_actual=ir_expected,
                    covered_ir_node_ids=tuple(trace.covered_ir_node_ids),
                    obligation_ids=tuple(trace.obligation_ids),
                )
            )
    if issues:
        raise ReviewedTraceIRValidationError(issues)


class AllCompletionsInvalidError(ValueError):
    """Raised when no authorized completion can be compiled and closed."""

    def __init__(self, message: str, invalid_completions: list[dict]):
        super().__init__(message)
        self.invalid_completions = list(invalid_completions)
        self.pipeline_stage = "completion"


class CandidateCompletionAgent(BaseAgent):
    """RAG + LLM 风格的受限骨架到完整 LTLf 候选补全 Agent。"""

    def __init__(self, retriever, llm_client: JSONLLMClient | None = None):
        super().__init__(retriever=retriever, name="CandidateCompletionAgent", llm_client=llm_client)
        self.last_invalid_completions: list[dict] = []

    def generate_hole_candidates(
        self,
        requirement: str,
        semantic: SemanticStructure,
        skeletons: list[Formula],
        traces: list[LabeledTrace],
        feedback_history: list[str] | None = None,
        guided_context: GuidedLearningContext | None = None,
        include_fallback: bool = True,
        batch_size: int = 5,
    ) -> list[HoleCandidate]:
        """
        把受限骨架扩展成带孔候选。

        response 模式下 rules 模式使用两个核心孔位：
            H_scope    in {epsilon, G}
            H_response in {epsilon, X, F, XF}
        """

        feedback_history = feedback_history or []
        if semantic.faithful_ir is not None:
            ir_branches = _generation_ir_branches(semantic)
            candidates: list[HoleCandidate] = []
            for branch_index, (branch_id, ir) in enumerate(ir_branches):
                skeleton_plan = compile_constrained_skeleton_plan(ir)
                hole_plan = compile_completion_hole_plan(ir, skeleton_plan)
                skeleton = (
                    skeletons[branch_index]
                    if branch_index < len(skeletons)
                    else compile_ir_skeleton(ir)
                )
                holes = tuple(
                    TemporalHole(
                        hole_id=constraint.hole_id,
                        target=constraint.skeleton_node_id,
                        allowed_ops=constraint.allowed_operators,
                        reason="Allowed operators compiled from FaithfulSemanticIR.",
                        role="ir_temporal_relation",
                        parent_hole_id=constraint.parent_hole_id,
                        ir_node_id=constraint.ir_node_id,
                        ir_version=ir.ir_version,
                    )
                    for constraint in hole_plan.holes
                )
                candidates.extend(
                    _materialize_ir_temporal_assignments(
                        HoleCandidate(
                            skeleton=skeleton,
                            template_text=render_ir_hole_template(ir),
                            holes=holes,
                            explanation=(
                                "IR-derived structure with a trajectory-resolvable temporal range."
                                if any(len(hole.allowed_ops) > 1 for hole in holes)
                                else "Deterministic IR-derived hole plan."
                            ),
                            candidate_id=f"{branch_id}-ir-{ir.ir_version}",
                        )
                    )
                )
            return candidates
        effective_requirement = finite_requirement_for_pipeline(requirement, semantic)
        retrieval_query = "\n".join(
            [
                "受限骨架到完整 LTLf 候选公式补全",
                effective_requirement,
                semantic.pattern,
                semantic.scope_keyword,
                semantic.temporal_keyword,
                "\n".join(str(skeleton) for skeleton in skeletons),
                _trace_summary_for_retrieval(traces),
                "\n".join(feedback_history),
                _guided_query_text(guided_context, "hole_template"),
            ]
        )
        prompt_context = build_stage_prompt_context(
            self.retriever,
            stage="hole_candidate_generation",
            query_parts=(retrieval_query,),
            domain_sources=(KB_OPERATOR_COMPLETION, KB_LTLF_SEMANTICS),
            example_sources=(
                TEMPLATE_SOURCE,
                GUIDED_TEMPLATE_SOURCE,
                FULL_PROCESS_SOURCE,
                GUIDED_FULL_PROCESS_SOURCE,
            ),
            agent_function_contract=_hole_candidate_contract(),
        )
        template_examples = self.retriever.retrieve_with_scores(
            retrieval_query,
            top_k=5,
            source_filter=TEMPLATE_SOURCE,
        )
        rag_context = _sanitize_rag_context(
            self.retrieve_context(
                retrieval_query,
                top_k=5,
                source_filter=(
                    "rag",
                    "builtin",
                    "runtime",
                    TEMPLATE_SOURCE,
                    FULL_PROCESS_SOURCE,
                ),
            )
        )
        template_context = tuple(
            _sanitize_context_text(
                f"{chunk.title} (similarity={score:.3f}): {chunk.text}"
            )
            for chunk, score in template_examples
        )
        combined_context = tuple(
            dict.fromkeys(
                (
                    *template_context,
                    *_sanitize_rag_context(prompt_context.as_rag_context()),
                    *rag_context,
                )
            )
        )

        learned_handler = get_pattern_handler_for_semantic(semantic)
        if (
            learned_handler is not None
            and semantic.pattern != "response"
            and learned_pattern_direct_execution_allowed(semantic)
        ):
            candidates = learned_handler.generate_hole_candidates(
                semantic=semantic,
                skeletons=skeletons,
                rag_context=combined_context,
            )
            return _ensure_min_hole_candidates(
                candidates,
                semantic=semantic,
                skeletons=skeletons,
                rag_context=combined_context,
            )

        if self.llm_enabled:
            candidates = self._generate_hole_candidates_with_llm(
                requirement=requirement,
                finite_requirement=effective_requirement,
                semantic=semantic,
                skeletons=skeletons,
                traces=traces,
                feedback_history=feedback_history,
                prompt_context=prompt_context,
                guided_context=guided_context,
                batch_size=batch_size,
            )
            return _ensure_min_hole_candidates(
                candidates,
                semantic=semantic,
                skeletons=skeletons,
                rag_context=tuple(prompt_context.as_rag_context()),
                include_fallback=include_fallback,
                maximum=(batch_size if not include_fallback else 12),
            )

        guided_candidates = _guided_hole_candidates(
            guided_context=guided_context,
            semantic=semantic,
            skeletons=skeletons,
            rag_context=combined_context,
        )
        if guided_candidates:
            return _ensure_min_hole_candidates(
                guided_candidates,
                semantic=semantic,
                skeletons=skeletons,
                rag_context=combined_context,
            )

        learned_candidates = _hole_candidates_from_template_examples(
            template_examples=template_examples,
            semantic=semantic,
            skeletons=skeletons,
            rag_context=combined_context,
        )
        if learned_candidates:
            return _ensure_min_hole_candidates(
                learned_candidates,
                semantic=semantic,
                skeletons=skeletons,
                rag_context=combined_context,
            )

        return _ensure_min_hole_candidates(
            [],
            semantic=semantic,
            skeletons=skeletons,
            rag_context=combined_context,
        )

    def complete_temporal_holes(
        self,
        semantic: SemanticStructure,
        hole_candidates: list[HoleCandidate],
        traces: list[LabeledTrace] | None = None,
        guided_context: GuidedLearningContext | None = None,
    ) -> list[TemporalCompletion]:
        """
        物化完整 LTLf 公式。

        当前 FaithfulSemanticIR 路径不根据 NL、RAG 或轨迹选择算子；上游已经
        把有限孔域完整展开为单值 HoleCandidate，本方法逐项确定性编译。轨迹
        只在后续整公式评估和 OperatorResolution 中决定哪些赋值存活。
        """

        self.last_invalid_completions = []
        traces = traces or []
        if semantic.faithful_ir is not None:
            ir_branches = _generation_ir_branches(semantic)
            completions: list[TemporalCompletion] = []
            for branch_index, (branch_id, ir) in enumerate(ir_branches, start=1):
                branch_candidates = [
                    candidate
                    for candidate in hole_candidates
                    if any(
                        hole.ir_version == ir.ir_version
                        for hole in candidate.holes
                    )
                    or f"-ir-{ir.ir_version}" in candidate.candidate_id
                ]
                if not branch_candidates and branch_index <= len(hole_candidates):
                    branch_candidates = [hole_candidates[branch_index - 1]]
                for candidate in branch_candidates:
                    assignments = tuple(
                        (
                            hole.hole_id,
                            hole.allowed_ops[0] if hole.allowed_ops else "epsilon",
                        )
                        for hole in candidate.holes
                    )
                    non_materialized = tuple(
                        hole.hole_id
                        for hole in candidate.holes
                        if len(hole.allowed_ops) != 1
                    )
                    resolution = {
                        hole.ir_node_id: operator
                        for hole, (_, operator) in zip(candidate.holes, assignments)
                        if hole.ir_node_id
                    }
                    completion_index = len(completions) + len(self.last_invalid_completions) + 1
                    try:
                        if non_materialized:
                            raise ValueError(
                                "IR temporal candidate was not materialized at holes "
                                + ", ".join(non_materialized)
                            )
                        formula = compile_ir_to_formula(
                            ir,
                            temporal_resolution_by_node=resolution,
                        )
                    except ValueError as exc:
                        self.last_invalid_completions.append(
                            _invalid_completion_record(
                                index=completion_index,
                                item={"branch_id": branch_id},
                                candidate_id=candidate.candidate_id,
                                fallback_candidate=candidate,
                                formula_text="",
                                assignments=assignments,
                                error=exc,
                                raw_completion={"source": "trajectory_guided_ir_compiler"},
                            )
                        )
                        continue
                    completions.append(
                        TemporalCompletion(
                            formula=formula,
                            assignments=assignments,
                            operator_derivation=tuple(
                                f"{hole_id} is an IR-approved candidate {operator}; "
                                "retention depends on whole-trace consistency"
                                for hole_id, operator in assignments
                            ),
                            candidate_id=candidate.candidate_id,
                            template_text=candidate.template_text,
                            ir_version=ir.ir_version,
                        )
                    )
            if not completions:
                raise AllCompletionsInvalidError(
                    "all executable temporal-choice IR branches failed compilation",
                    self.last_invalid_completions,
                )
            return _validate_completion_ap_whitelist(completions, semantic)
        hole_candidates = _attach_operator_completion_context(
            self,
            semantic=semantic,
            hole_candidates=hole_candidates,
            traces=traces,
        )

        learned_handler = get_pattern_handler_for_semantic(semantic)
        if (
            learned_handler is not None
            and semantic.pattern != "response"
            and learned_pattern_direct_execution_allowed(semantic)
        ):
            if guided_context is not None and guided_context.expected_final_candidate:
                return _validate_completion_ap_whitelist(
                    _guided_temporal_completions(
                        guided_context=guided_context,
                        hole_candidates=hole_candidates,
                    ),
                    semantic,
                )
            return _validate_completion_ap_whitelist(
                learned_handler.complete_temporal_holes(
                    semantic=semantic,
                    hole_candidates=hole_candidates,
                ),
                semantic,
            )

        if (
            guided_context is not None
            and guided_context.expected_final_candidate
            and not self.llm_enabled
        ):
            return _validate_completion_ap_whitelist(
                _guided_temporal_completions(
                    guided_context=guided_context,
                    hole_candidates=hole_candidates,
                ),
                semantic,
            )

        if self.llm_enabled:
            prompt_context = build_stage_prompt_context(
                self.retriever,
                stage="temporal_hole_completion",
                query_parts=(
                    "X/F/G 补全",
                    semantic.pattern,
                    semantic.trigger,
                    semantic.response,
                    semantic.scope_keyword,
                    semantic.temporal_keyword,
                    _trace_summary_for_retrieval(traces),
                    "\n".join(_hole_candidate_query_lines(hole_candidates)),
                    _guided_query_text(guided_context, "completion"),
                ),
                domain_sources=(KB_OPERATOR_COMPLETION, KB_LTLF_SEMANTICS),
                example_sources=(
                    OPERATOR_COMPLETION_SOURCE,
                    GUIDED_COMPLETION_SOURCE,
                    TEMPLATE_SOURCE,
                    GUIDED_TEMPLATE_SOURCE,
                    FULL_PROCESS_SOURCE,
                    GUIDED_FULL_PROCESS_SOURCE,
                ),
                agent_function_contract=_temporal_completion_contract(),
            )
            return _validate_completion_ap_whitelist(
                self._complete_temporal_holes_with_llm(
                    semantic=semantic,
                    hole_candidates=hole_candidates,
                    traces=traces,
                    prompt_context=prompt_context,
                    guided_context=guided_context,
                ),
                semantic,
            )

        return _validate_completion_ap_whitelist(
            [
                _complete_single_hole_candidate(
                    semantic,
                    hole_candidate,
                    traces=traces,
                )
                for hole_candidate in hole_candidates
            ],
            semantic,
        )

    def _generate_hole_candidates_with_llm(
        self,
        requirement: str,
        finite_requirement: str,
        semantic: SemanticStructure,
        skeletons: list[Formula],
        traces: list[LabeledTrace],
        feedback_history: list[str],
        prompt_context,
        guided_context: GuidedLearningContext | None = None,
        batch_size: int = 5,
    ) -> list[HoleCandidate]:
        trace_reference_rule = _trace_reference_rule(traces)
        extra_rules = (
            (
                guided_prompt_rule("hole_candidate_generation"),
                "expected_hole_template 只用于决定孔位位置和 allowed_ops，不要在本阶段选择最终 X/F/G 算子。",
            )
            if guided_context is not None
            else ()
        )
        system_prompt = structured_system_prompt(
            agent_name="候选公式孔位生成 Agent",
            stage_name="hole_candidate_generation",
            extra_rules=(
                "只把已给定的受限骨架扩展成可补全 X/F/G 的带孔模板。",
                f"只返回当前最有根据的 Top-{batch_size} 个 template_text 不同的带孔候选模板；不要超过 {batch_size} 个。",
                "如果 feedback_history 中列出上一批失败原因或已有模板，不要重复上一批；优先修复失败原因。",
                "template_text 本身禁止出现裸 X、F、G、next、eventually、finally、always、globally。",
                "template_text 只能用 H_scope/H_response 等孔位表示待补全时序算子。",
                "否定 ! 不得直接重复或仅隔括号嵌套重复；禁止 !!p 和 !(!p)。",
                "优先使用 NL 直接支持的模板；只加入少量等价布尔改写。",
                "布尔连接符必须统一使用 & 和 |，不要输出 && 或 ||。",
                "allowed_ops 只能包含 epsilon、X、F、G、XF。",
                trace_reference_rule
                + "、NL 和 RAG 示例只作为模板形状参考；不要选择最终算子，不要输出完整候选评分，不要重新生成骨架。",
                *extra_rules,
            ),
        )
        rag_context = _sanitize_rag_context(prompt_context.as_rag_context())
        payload = {
            "task": "hole_candidate_generation",
            "requirement": finite_requirement,
            "original_requirement": requirement,
            "trace_domain": semantic.trace_domain,
            "domain_warning": semantic.domain_warning,
            "semantic": {
                "trigger": semantic.trigger,
                "response": semantic.response,
                "pattern": semantic.pattern,
                "scope_keyword": semantic.scope_keyword,
                "temporal_keyword": semantic.temporal_keyword,
            },
            "skeletons": [str(skeleton) for skeleton in skeletons],
            "trace_summary": _trace_summary_for_retrieval(traces),
            "feedback_history": feedback_history,
            "guided_learning": (
                guided_context.prompt_payload("hole_candidate_generation")
                if guided_context is not None
                else None
            ),
            **prompt_context.payload_fields(),
            "rag_context": rag_context,
            "output_schema": {
                "hole_candidates": [
                    {
                        "skeleton_index": 0,
                        "template_text": "H_scope(request -> H_response(grant))",
                        "holes": [
                            {
                                "hole_id": "H_scope",
                                "target": "whole implication",
                                "allowed_ops": ["epsilon", "G"],
                                "reason": "why this hole is needed",
                                "role": "root_scope",
                                "parent_hole_id": "",
                                "source_hint": [],
                            }
                        ],
                        "explanation": "short explanation",
                    }
                ],
                "max_items": batch_size,
            },
        }
        raw_result = self.complete_json(system_prompt, payload, max_tokens=2500)
        items = coerce_list_field(
            raw_result,
            "hole_candidates",
            aliases=("candidates", "templates"),
            single_object_keys=("template_text", "holes"),
        )

        candidates: list[HoleCandidate] = []
        for index, item in enumerate(items, start=1):
            try:
                mapping = _hole_candidate_mapping_from_llm(item, index, skeletons)
                skeleton = _resolve_skeleton(mapping, skeletons, index)
                raw_holes = mapping.get("holes", [])
                if raw_holes is None:
                    raw_holes = []
                holes = tuple(
                    _temporal_hole_from_llm(
                        hole,
                        index,
                        hole_index,
                        semantic=semantic,
                    )
                    for hole_index, hole in enumerate(
                        require_list(raw_holes, f"hole_candidates[{index}].holes"),
                        start=1,
                    )
                )
                candidate = _sanitize_hole_candidate(
                    HoleCandidate(
                        skeleton=skeleton,
                        template_text=_hole_template_text_from_llm(
                            mapping,
                            default=str(skeleton),
                        ),
                        holes=holes,
                        rag_context=rag_context,
                        explanation=optional_string(
                            mapping,
                            "explanation",
                            "LLM 生成带孔候选。",
                        ),
                    ),
                    semantic=semantic,
                    strict=True,
                )
            except ValueError:
                continue
            candidates.append(candidate)

        return candidates

    def _complete_temporal_holes_with_llm(
        self,
        semantic: SemanticStructure,
        hole_candidates: list[HoleCandidate],
        traces: list[LabeledTrace],
        prompt_context,
        guided_context: GuidedLearningContext | None = None,
    ) -> list[TemporalCompletion]:
        trace_reference_rule = _trace_reference_rule(traces)
        extra_rules = (
            (
                guided_prompt_rule("temporal_hole_completion"),
                "expected_completion 和 approved_assignments 必须作为本轮 H_xxx operator 选择的 teacher signal。",
            )
            if guided_context is not None
            else ()
        )
        system_prompt = structured_system_prompt(
            agent_name="X/F/G 补全 Agent",
            stage_name="temporal_hole_completion",
            extra_rules=(
                "只基于孔位定义和 allowed_ops 生成多个完整 LTLf 候选公式。",
                "对每个输入 hole_candidate 返回且仅返回一个 completion，并保留 candidate_id。",
                "只要 template_text 中出现 H_xxx，就必须在 assignments 中记录该孔位选择的算子。",
                "assignments 可以是单个算子，也可以是由 X/F/G 原子组成的组合序列，"
                "例如 X X X；组合序列中的每个原子都必须是支持的未来时算子。",
                "补全算子时必须同时参考 NL 需求、RAG operator-completion 示例以及"
                + trace_reference_rule
                + "。",
                f"{SUPPORTED_LTLF_FORMULA_GUIDANCE}",
                "assignments 要记录每个 hole_id 选择的算子。",
                "operator_derivation 要按孔位说明为什么选择当前算子。",
                "不要修改孔位定义，不要新增骨架，不要计算 PosSat/NegRej/BoundMatch 或最终通过判断。",
                "epsilon 只能作为 assignments 中的 H_xxx 算子值；formula/final_candidate 等完整公式字段不能是 epsilon。",
                *extra_rules,
            ),
        )
        rag_context = _sanitize_rag_context(prompt_context.as_rag_context())
        payload = {
            "task": "temporal_hole_completion",
            "semantic": {
                "trigger": semantic.trigger,
                "response": semantic.response,
                "pattern": semantic.pattern,
                "scope_keyword": semantic.scope_keyword,
                "temporal_keyword": semantic.temporal_keyword,
            },
            "hole_candidates": [_hole_candidate_payload(candidate) for candidate in hole_candidates],
            "trace_context": _trace_summary_for_retrieval(traces),
            "trace_use_policy": trace_reference_rule,
            "guided_learning": (
                guided_context.prompt_payload("temporal_hole_completion")
                if guided_context is not None
                else None
            ),
            **prompt_context.payload_fields(),
            "rag_context": rag_context,
            "output_schema": {
                "completions": [
                    {
                        "candidate_id": "HC-001",
                        "formula": "G(request -> F grant)",
                        "assignments": {"H_scope": "G", "H_response": "F"},
                        "operator_derivation": [
                            "H_scope=G because NL requires whenever/global scope and traces include non-initial triggers.",
                            "H_response=F because positive/boundary traces allow response at a later finite position.",
                        ],
                    }
                ],
                "formula_constraints": SUPPORTED_LTLF_FORMULA_GUIDANCE,
            },
        }
        raw_result = self.complete_json(system_prompt, payload, max_tokens=1800)
        items = _completion_items_from_llm(raw_result)

        completions: list[TemporalCompletion] = []
        attempted_candidate_ids: set[str] = set()
        for index, item in enumerate(items, start=1):
            positional_candidate = (
                hole_candidates[index - 1]
                if index - 1 < len(hole_candidates)
                else None
            )
            fallback_candidate = positional_candidate
            mapping: dict = {}
            candidate_id = positional_candidate.candidate_id if positional_candidate else ""
            formula_text = ""
            assignments: tuple[tuple[str, str], ...] = ()
            try:
                mapping = require_mapping(item, f"completions[{index}]")
                candidate_id = optional_string(
                    mapping,
                    "candidate_id",
                    positional_candidate.candidate_id if positional_candidate else "",
                )
                fallback_candidate = _hole_candidate_for_completion(
                    hole_candidates,
                    candidate_id=candidate_id,
                    positional_candidate=positional_candidate,
                )
                if fallback_candidate is None:
                    raise ValueError(
                        f"completions[{index}] 无法匹配输入 hole_candidate: "
                        f"candidate_id={candidate_id or '<empty>'}。"
                    )
                candidate_id = fallback_candidate.candidate_id
                attempted_candidate_ids.add(candidate_id)
                assignments = _assignments_from_llm(
                    mapping.get("assignments"),
                    index,
                )
                _validate_completion_assignments(
                    hole_candidate=fallback_candidate,
                    assignments=assignments,
                    index=index,
                )
                formula_text = _completion_formula_text_from_llm(mapping)
                formula = _formula_from_llm_completion(
                    formula_text=formula_text,
                    semantic=semantic,
                    assignments=assignments,
                    hole_candidate=fallback_candidate,
                    index=index,
                )
                _validate_single_completion_ap_whitelist(
                    formula,
                    semantic,
                    context=f"completions[{index}].formula",
                )
                completions.append(
                    TemporalCompletion(
                        formula=formula,
                        assignments=assignments,
                        operator_derivation=_operator_derivation_from_llm(
                            mapping.get("operator_derivation"),
                            semantic=semantic,
                            hole_candidate=fallback_candidate,
                            traces=traces,
                            assignments=assignments,
                        ),
                        candidate_id=candidate_id,
                        template_text=optional_string(
                            mapping,
                            "template_text",
                            fallback_candidate.template_text,
                        ),
                    )
                )
            except ValueError as exc:
                self.last_invalid_completions.append(
                    _invalid_completion_record(
                        index=index,
                        item=item,
                        candidate_id=candidate_id,
                        fallback_candidate=fallback_candidate,
                        formula_text=formula_text,
                        assignments=assignments,
                        error=exc,
                        raw_completion=raw_result,
                    )
                )

        returned_candidate_ids = {
            completion.candidate_id
            for completion in completions
            if completion.candidate_id
        }
        for candidate in hole_candidates:
            if (
                candidate.candidate_id in returned_candidate_ids
                or candidate.candidate_id in attempted_candidate_ids
            ):
                continue
            self.last_invalid_completions.append(
                _invalid_completion_record(
                    index=len(self.last_invalid_completions) + 1,
                    item=None,
                    candidate_id=candidate.candidate_id,
                    fallback_candidate=candidate,
                    formula_text="",
                    assignments=(),
                    error=ValueError(
                        "LLM 未返回该 hole_candidate 对应的 completion。"
                    ),
                    raw_completion=raw_result,
                )
            )

        if not completions:
            raise AllCompletionsInvalidError(
                "completion 阶段全部候选均为 invalid；没有候选可进入轨迹评分。",
                self.last_invalid_completions,
            )
        return _align_completions_to_hole_candidates(
            completions=completions,
            hole_candidates=hole_candidates,
        )

def _validate_completion_ap_whitelist(
    completions: list[TemporalCompletion],
    semantic: SemanticStructure,
) -> list[TemporalCompletion]:
    validated: list[TemporalCompletion] = []
    for completion in completions:
        report = _validate_single_completion_ap_whitelist(
            completion.formula,
            semantic,
            context="完整候选公式",
            ir_version=completion.ir_version,
        )
        validated.append(
            replace(
                completion,
                ir_version=report.ir_version,
                closure_report=report,
            )
        )
    return validated


def _validate_single_completion_ap_whitelist(
    formula: Formula,
    semantic: SemanticStructure,
    *,
    context: str,
    ir_version: str = "",
):
    policy = _formula_closure_policy_for_semantic(semantic, ir_version=ir_version)
    bound_ir_version = ir_version or (
        semantic.faithful_ir.ir_version
        if semantic.faithful_ir is not None
        else policy.approved_ir_version
    )
    report = validate_formula_closure(
        formula,
        policy,
        ir_version=bound_ir_version,
    )
    if not report.passed:
        human_category = (
            "布尔常量；"
            if report.boolean_constants
            else "AP 白名单外元素；"
            if report.extra_aps
            else "公式闭包；"
        )
        raise ValueError(
            f"{context} failed FormulaClosurePolicy ({human_category}) "
            + "; ".join(report.reasons)
        )
    return report


def _formula_closure_policy_for_semantic(
    semantic: SemanticStructure,
    *,
    ir_version: str = "",
) -> FormulaClosurePolicy:
    if semantic.faithful_ir is not None:
        selected_ir = semantic.faithful_ir
        if ir_version and selected_ir.ir_version != ir_version:
            candidate_set = semantic.candidate_ir_set
            selected_ir = next(
                (
                    branch.faithful_ir
                    for branch in (candidate_set.branches if candidate_set else ())
                    if branch.faithful_ir.ir_version == ir_version
                ),
                selected_ir,
            )
        approved = (
            semantic.approved_semantic_ir
            if semantic.approved_semantic_ir is not None
            and semantic.approved_semantic_ir.ir_version == selected_ir.ir_version
            else None
        )
        return build_strict_formula_closure_policy(
            selected_ir,
            approved,
        )
    allowed_atoms = tuple(
        dict.fromkeys(
            (
                *semantic.aps,
                *(
                    atom
                    for atom in (semantic.trigger, semantic.response)
                    if atom and atom not in {"unknown", "unknown_ap"}
                ),
            )
        )
    )
    if not allowed_atoms:
        raise ValueError("FormulaClosurePolicy requires a non-empty AP alphabet")
    return FormulaClosurePolicy(
        allowed_aps=allowed_atoms,
        optional_aps=allowed_atoms,
    )


def _invalid_completion_record(
    *,
    index: int,
    item: object,
    candidate_id: str,
    fallback_candidate: HoleCandidate | None,
    formula_text: str,
    assignments: tuple[tuple[str, str], ...],
    error: ValueError,
    raw_completion: object,
) -> dict:
    invalid_reason = str(error)
    return {
        "status": "invalid",
        "index": index,
        "candidate_id": candidate_id,
        "template_text": (
            fallback_candidate.template_text if fallback_candidate is not None else ""
        ),
        "formula_text": formula_text,
        "assignments": [
            {"hole_id": hole_id, "selected_op": operator}
            for hole_id, operator in assignments
        ],
        "error_type": type(error).__name__,
        "error": invalid_reason,
        "invalid_reason": invalid_reason,
        "fallback_used": False,
        "raw_item": _safe_invalid_completion_item(item),
        "raw_completion": _safe_completion_payload(raw_completion),
    }


def _safe_completion_payload(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _safe_completion_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_completion_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def _safe_invalid_completion_item(item: object) -> object:
    if isinstance(item, dict):
        return {
            str(key): value
            for key, value in item.items()
            if key
            in {
                "candidate_id",
                "template_text",
                "formula",
                "formula_text",
                "ltlf_formula",
                "candidate_formula",
                "completed_formula",
                "complete_formula",
                "final_formula",
                "final_candidate",
                "temporal_formula",
                "assignments",
            }
        }
    if isinstance(item, (str, int, float, bool)) or item is None:
        return item
    return str(item)


def _hole_candidate_contract() -> dict:
    return {
        "agent": "CandidateCompletionAgent",
        "stage": "hole_candidate_generation",
        "responsibilities": [
            "Convert accepted restricted skeletons into temporal-hole templates.",
            "Define hole_id, target, allowed_ops, reason, role, parent_hole_id, and source_hint for each hole.",
            "Use unique H_scope_child_N or H_response_child_N ids for nested temporal subformulas.",
            "Use Top-5 examples only as template-shape references.",
        ],
        "forbidden": [
            "Do not choose final temporal operators.",
            "Do not generate new skeletons, traces, final rankings, or consistency scores.",
            "Do not change semantic trigger, response, pattern, or reviewed traces.",
        ],
    }


def _temporal_completion_contract() -> dict:
    return {
        "agent": "CandidateCompletionAgent",
        "stage": "temporal_hole_completion",
        "responsibilities": [
            "Generate full LTLf formulas by filling existing holes with allowed_ops only.",
            "Return assignments that record every selected hole operator.",
            "Allow assignment operator sequences such as X X X when every atom is a supported X/F/G operator.",
            "Return one operator_derivation item for each H_xxx assignment.",
            "Use NL, every supplied trace kind, and Top-5 examples only as operator-selection references.",
        ],
        "forbidden": [
            "Do not create new holes or skeletons.",
            "Do not simulate traces, compute scores, or choose the final winner.",
            "Do not introduce operators outside each hole's allowed_ops.",
        ],
    }


def _guided_hole_candidates(
    *,
    guided_context: GuidedLearningContext | None,
    semantic: SemanticStructure,
    skeletons: list[Formula],
    rag_context: tuple[str, ...],
) -> list[HoleCandidate]:
    if guided_context is None or not guided_context.expected_hole_template:
        return []
    skeleton = skeletons[0] if skeletons else _fallback_skeleton(semantic)
    return [
        _sanitize_hole_candidate(
            HoleCandidate(
                skeleton=skeleton,
                template_text=guided_context.expected_hole_template,
                holes=(),
                rag_context=rag_context,
                explanation=(
                    "warmup one-shot guided template: expected_hole_template "
                    "decides where temporal operators remain as H_xxx holes."
                ),
            ),
            semantic=semantic,
            strict=False,
        )
    ]


def _guided_temporal_completions(
    *,
    guided_context: GuidedLearningContext,
    hole_candidates: list[HoleCandidate],
) -> list[TemporalCompletion]:
    formula_text = guided_context.expected_final_candidate or guided_context.approved_formula
    formula = parse_formula(normalize_boolean_operator_text(formula_text))
    assignments = tuple(guided_context.approved_assignments)
    derivation = (
        "warmup one-shot guided completion selected the approved final formula "
        f"{formula_text}.",
    )
    return [
        TemporalCompletion(
            formula=formula,
            assignments=assignments,
            operator_derivation=derivation,
            candidate_id=candidate.candidate_id,
            template_text=candidate.template_text,
        )
        for candidate in hole_candidates
    ]


def _guided_query_text(
    guided_context: GuidedLearningContext | None,
    stage: str,
) -> str:
    if guided_context is None:
        return ""
    base = (
        f"warmup guided {stage} {guided_context.requirement_id} "
        "stage_scoped_teacher_signal"
    )
    if stage not in {"completion", "temporal_hole_completion"}:
        return base
    return (
        f"{base} completion_assignments_available "
        "final_formula_visible_only_to_offline_validation"
    )


def _hole_candidate_query_lines(hole_candidates: list[HoleCandidate]) -> list[str]:
    lines: list[str] = []
    for candidate in hole_candidates:
        lines.append(f"candidate_id: {candidate.candidate_id}")
        lines.append(f"skeleton: {candidate.skeleton}")
        lines.append(f"template_text: {candidate.template_text}")
        for hole in candidate.holes:
            lines.append(
                "hole: "
                f"{hole.hole_id}; target={hole.target}; "
                f"allowed_ops={','.join(hole.allowed_ops)}; "
                f"role={hole.role}; parent_hole_id={hole.parent_hole_id}; "
                f"source_hint={','.join(hole.source_hint)}; reason={hole.reason}"
            )
    return lines


COMPLETION_FORMULA_FIELDS = (
    "formula",
    "formula_text",
    "ltlf_formula",
    "ltlf",
    "candidate_formula",
    "completed_formula",
    "complete_formula",
    "final_formula",
    "final_candidate",
    "temporal_formula",
)

COMPLETION_ITEM_FIELDS = (
    "candidate_id",
    "template_text",
    "operator_derivation",
    *COMPLETION_FORMULA_FIELDS,
)


def _completion_items_from_llm(raw_result: dict) -> list:
    raw_completion_value = _first_present_value(
        raw_result,
        ("completions", "completion", "hole_completions", "candidates"),
    )
    assignment_items = _completion_assignment_items(raw_completion_value)
    if assignment_items is not None:
        return assignment_items

    try:
        items = coerce_list_field(
            raw_result,
            "completions",
            "completions",
            aliases=("completion", "hole_completions", "candidates"),
            single_object_keys=(
                *COMPLETION_FORMULA_FIELDS,
                "assignments",
            ),
        )
    except ValueError:
        assignment_items = _completion_assignment_items(raw_result)
        if assignment_items is not None:
            return assignment_items
        raise

    return [_normalize_completion_item(item) for item in items]


def _first_present_value(mapping: dict, keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _completion_assignment_items(value: object | None) -> list[dict] | None:
    if not isinstance(value, dict):
        return None
    if _looks_like_hole_assignment_map(value):
        return [_assignment_completion_item(value)]

    items: list[dict] = []
    for candidate_id, nested in value.items():
        if not isinstance(candidate_id, str) or not isinstance(nested, dict):
            return None
        if not _looks_like_hole_assignment_map(nested):
            return None
        item = _assignment_completion_item(nested)
        item.setdefault("candidate_id", candidate_id)
        items.append(item)
    return items or None


def _normalize_completion_item(item: object) -> object:
    if not isinstance(item, dict) or not _looks_like_hole_assignment_map(item):
        return item
    return _assignment_completion_item(item)


def _assignment_completion_item(mapping: dict) -> dict:
    item = {
        field: mapping[field]
        for field in COMPLETION_ITEM_FIELDS
        if field in mapping
    }
    item["assignments"] = {
        key: value
        for key, value in mapping.items()
        if isinstance(key, str) and key.startswith("H_")
    }
    return item


def _looks_like_hole_assignment_map(mapping: dict) -> bool:
    return any(isinstance(key, str) and key.startswith("H_") for key in mapping)


def _formula_from_llm_completion(
    formula_text: str,
    semantic: SemanticStructure,
    assignments: tuple[tuple[str, str], ...],
    hole_candidate: HoleCandidate | None,
    index: int,
) -> Formula:
    formula_text = formula_text.strip()
    formula_is_bare_epsilon = _is_bare_epsilon_formula_text(formula_text)
    if hole_candidate is not None:
        has_holes = bool(hole_candidate.holes) or bool(
            re.search(r"\bH_[A-Za-z0-9_]*\b", hole_candidate.template_text)
        )
        if has_holes:
            # Assignments are authoritative.  The optional complete formula is
            # only a redundant field and cannot bypass the reviewed template.
            if formula_is_bare_epsilon:
                raise ValueError(_bare_epsilon_formula_error(index))
            formula = _render_formula_from_template(hole_candidate, assignments)
            validate_no_repeated_negation(
                formula,
                context=f"completions[{index}].formula",
            )
            if formula_text:
                declared = _parse_completion_formula(formula_text, index)
                if declared != formula:
                    raise ValueError(
                        "completion_formula_assignment_mismatch: "
                        f"completions[{index}].formula={declared} does not match "
                        f"template+assignments={formula}"
                    )
            return formula

        rendered = _parse_completion_formula(hole_candidate.template_text, index)
        if formula_is_bare_epsilon:
            raise ValueError(_bare_epsilon_formula_error(index))
        if formula_text:
            declared = _parse_completion_formula(formula_text, index)
            if declared != rendered:
                raise ValueError(
                    "completion_formula_assignment_mismatch: "
                    f"completions[{index}].formula={declared} does not match "
                    f"holeless template={rendered}"
                )
        return rendered

    if assignments:
        raise ValueError(
            f"completions[{index}].assignments cannot be used without a "
            "matched hole_candidate"
        )
    if not formula_text:
        raise ValueError(
            f"completions[{index}] requires a matched hole_candidate and exact "
            "assignments, or a holeless formula input"
        )
    if formula_is_bare_epsilon:
        raise ValueError(_bare_epsilon_formula_error(index))
    try:
        return _parse_completion_formula(formula_text, index)
    except ValueError as exc:
        raise ValueError(
            f"completions[{index}].formula cannot be parsed: {formula_text}. "
            f"Reason: {exc}"
        ) from exc


def _parse_completion_formula(formula_text: str, index: int) -> Formula:
    if _is_bare_epsilon_formula_text(formula_text):
        raise ValueError(_bare_epsilon_formula_error(index))
    formula = parse_formula(formula_text)
    validate_no_repeated_negation(
        formula,
        context=f"completions[{index}].formula",
    )
    return formula


def _is_bare_epsilon_formula_text(formula_text: str) -> bool:
    return formula_text.strip().strip("`\"'“”‘’").lower() == "epsilon"


def _bare_epsilon_formula_error(index: int) -> str:
    return (
        f"completions[{index}].formula 中的 epsilon 只能作为 H_xxx "
        "assignment，不能作为完整 LTLf formula。"
    )


def _completion_formula_text_from_llm(mapping: dict) -> str:
    for field in (
        "formula",
        "formula_text",
        "ltlf_formula",
        "ltlf",
        "candidate_formula",
        "completed_formula",
        "complete_formula",
        "final_formula",
        "final_candidate",
        "temporal_formula",
    ):
        value = mapping.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_field in ("text", "value", "formula"):
                nested_value = value.get(nested_field)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    return ""


def _build_response_formula(
    semantic: SemanticStructure,
    assignments: tuple[tuple[str, str], ...],
) -> Formula:
    assignment_map = dict(assignments)
    trigger, response = _semantic_response_endpoint_formulas(semantic)

    response_formula = _apply_response_operator(
        response,
        assignment_map.get("H_response", "epsilon"),
    )
    implication = Implies(trigger, response_formula)

    if assignment_map.get("H_scope") == "G":
        return G(implication)
    return implication


def _semantic_response_endpoint_formulas(
    semantic: SemanticStructure,
) -> tuple[Formula, Formula]:
    trigger_text, response_text = _semantic_response_endpoint_texts(semantic)
    try:
        trigger = parse_formula(trigger_text)
    except ValueError:
        trigger = _atom(semantic.trigger)
    try:
        response = parse_formula(response_text)
    except ValueError:
        response = _atom(semantic.response)
    return trigger, response


def _semantic_response_endpoint_texts(
    semantic: SemanticStructure,
) -> tuple[str, str]:
    trigger_text = _component_formula_text(
        semantic,
        "trigger_definition",
        "trigger_formula",
    )
    response_text = _component_formula_text(
        semantic,
        "response_formula",
        "response_definition",
    )
    return (
        trigger_text or semantic.trigger or "trigger",
        response_text or semantic.response or "response",
    )


def _component_formula_text(
    semantic: SemanticStructure,
    *keys: str,
) -> str:
    components = dict(semantic.components)
    for key in keys:
        text = _component_rhs(components.get(key, ""))
        if not text:
            continue
        text = normalize_boolean_operator_text(text)
        try:
            return str(_strip_concrete_xfg(parse_formula(text)))
        except ValueError:
            return text
    return ""


def _component_rhs(value: str) -> str:
    text = value.strip().strip("`\"'“”‘’")
    if not text:
        return ""
    match = re.search(r"(?::=|=)", text)
    if match is not None:
        text = text[match.end() :].strip()
    return text.rstrip(".,;:，。；：")


def _complete_single_hole_candidate(
    semantic: SemanticStructure,
    hole_candidate: HoleCandidate,
    traces: list[LabeledTrace] | None = None,
) -> TemporalCompletion:
    traces = traces or []
    if not hole_candidate.holes:
        formula = _formula_for_holeless_candidate(hole_candidate)
        assignments = ()
        operator_derivation = (
            "模板不包含 H_xxx 孔位，按无孔候选直接解析。",
        )
    else:
        assignments = _select_assignments_for_hole_candidate(
            semantic=semantic,
            hole_candidate=hole_candidate,
            traces=traces,
        )
        operator_derivation = _operator_derivations_for_hole_candidate(
            semantic=semantic,
            hole_candidate=hole_candidate,
            traces=traces,
            assignments=assignments,
        )
        _validate_completion_assignments(
            hole_candidate=hole_candidate,
            assignments=assignments,
            index=1,
        )
        formula = _render_formula_from_template(hole_candidate, assignments)

    return TemporalCompletion(
        formula=formula,
        assignments=assignments,
        operator_derivation=operator_derivation,
        candidate_id=hole_candidate.candidate_id,
        template_text=hole_candidate.template_text,
    )


def _formula_for_holeless_candidate(hole_candidate: HoleCandidate) -> Formula:
    for text in (hole_candidate.template_text, str(hole_candidate.skeleton)):
        try:
            return parse_formula(text)
        except ValueError:
            continue
    return hole_candidate.skeleton


def _select_assignments_for_hole_candidate(
    *,
    semantic: SemanticStructure,
    hole_candidate: HoleCandidate,
    traces: list[LabeledTrace],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            hole.hole_id,
            _select_operator_for_hole(
                semantic=semantic,
                hole_candidate=hole_candidate,
                hole=hole,
                traces=traces,
            ),
        )
        for hole in hole_candidate.holes
    )


def _select_operator_for_hole(
    *,
    semantic: SemanticStructure,
    hole_candidate: HoleCandidate,
    hole: TemporalHole,
    traces: list[LabeledTrace],
) -> str:
    trace_operator = _operator_from_trace_context(
        semantic=semantic,
        traces=traces,
        hole=hole,
    )
    if trace_operator in hole.allowed_ops:
        return trace_operator

    rag_operator = _operator_from_rag_context(
        rag_context=hole_candidate.rag_context,
        hole_id=hole.hole_id,
    )
    if rag_operator in hole.allowed_ops:
        return rag_operator

    default_operator = _operator_from_semantic_keywords(semantic, hole.hole_id)
    if default_operator != "epsilon" and default_operator in hole.allowed_ops:
        return default_operator
    hinted_operator = _operator_from_source_hint(hole)
    if hinted_operator in hole.allowed_ops:
        return hinted_operator
    if default_operator in hole.allowed_ops:
        return default_operator
    if "epsilon" in hole.allowed_ops:
        return "epsilon"
    return hole.allowed_ops[0]


def _operator_derivations_for_hole_candidate(
    *,
    semantic: SemanticStructure,
    hole_candidate: HoleCandidate,
    traces: list[LabeledTrace],
    assignments: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    assignment_map = dict(assignments)
    trace_counts = _trace_kind_counts(traces)
    rag_hint_count = len(hole_candidate.rag_context)
    lines: list[str] = []
    for hole in hole_candidate.holes:
        selected = assignment_map.get(hole.hole_id, "epsilon")
        lines.append(
            f"{hole.hole_id}={selected}: target={hole.target}; "
            f"role={hole.role or 'unspecified'}; "
            f"parent={hole.parent_hole_id or 'none'}; "
            f"source_hint={','.join(hole.source_hint) or 'none'}; "
            f"allowed_ops={','.join(hole.allowed_ops)}; "
            f"NL(scope={semantic.scope_keyword or 'none'}, temporal={semantic.temporal_keyword or 'none'}); "
            f"traces(positive={trace_counts['positive']}, negative={trace_counts['negative']}, boundary={trace_counts['boundary']}); "
            f"RAG snippets={rag_hint_count}."
        )
    return tuple(lines)


def _trace_kind_counts(traces: list[LabeledTrace]) -> dict[str, int]:
    return {
        "positive": sum(1 for trace in traces if trace.kind == "positive"),
        "negative": sum(1 for trace in traces if trace.kind == "negative"),
        "boundary": sum(1 for trace in traces if trace.kind == "boundary"),
    }


def _trace_reference_rule(traces: list[LabeledTrace]) -> str:
    counts = _trace_kind_counts(traces)
    names = [
        label
        for kind, label in (
            ("positive", "正轨迹"),
            ("negative", "反轨迹"),
            ("boundary", "边界轨迹"),
        )
        if counts[kind]
    ]
    return "、".join(names) if names else "当前提供的有限轨迹"


def _operator_from_template_text(
    *,
    template_text: str,
    semantic: SemanticStructure,
    hole: TemporalHole,
) -> str:
    text = _normalize_template_text(template_text)
    if hole.hole_id in text:
        return ""

    if hole.hole_id == "H_scope":
        if re.match(r"^(G|globally|always)\s*(\(|\s)", text, flags=re.IGNORECASE):
            return "G"
        if re.match(r"^(F|finally|eventually)\s*(\(|\s)", text, flags=re.IGNORECASE):
            return "F"
        if re.match(r"^(X|next)\s*(\(|\s)", text, flags=re.IGNORECASE):
            return "X"
        return "epsilon"

    if hole.hole_id == "H_response":
        response = re.escape(semantic.response)
        if re.search(rf"\bX\s*\(\s*F\s+{response}\b", text):
            return "XF"
        if re.search(rf"\bX\s+F\s+{response}\b", text):
            return "XF"
        if re.search(rf"\bF\s*\(?\s*{response}\b", text):
            return "F"
        if re.search(rf"\b(finally|eventually)\s*\(?\s*{response}\b", text, flags=re.IGNORECASE):
            return "F"
        if re.search(rf"\bX\s*\(?\s*{response}\b", text):
            return "X"
        if re.search(rf"\bnext\s*\(?\s*{response}\b", text, flags=re.IGNORECASE):
            return "X"
        if re.search(rf"\bG\s*\(?\s*{response}\b", text):
            return "G"
        if re.search(rf"\b(always|globally)\s*\(?\s*{response}\b", text, flags=re.IGNORECASE):
            return "G"
    return ""


def _operator_from_trace_context(
    *,
    semantic: SemanticStructure,
    traces: list[LabeledTrace],
    hole: TemporalHole,
) -> str:
    """Choose an operator from trace shape only; final scoring remains separate."""

    if not traces:
        return ""

    joint_traces = [
        trace
        for trace in traces
        if trace.kind in {"positive", "negative", "boundary"}
    ]
    hole_id = hole.hole_id

    if _hole_is_root_scope(hole):
        if _has_late_trigger(joint_traces, semantic.trigger):
            return "G"
        if _keyword_has_global_scope(semantic):
            return "G"
        return "epsilon"

    if _hole_looks_like_response(hole):
        return _response_operator_from_labeled_traces(
            joint_traces,
            semantic=semantic,
        )

    return ""


def _hole_is_root_scope(hole: TemporalHole) -> bool:
    if hole.role:
        return hole.role == "root_scope"
    return hole.hole_id == "H_scope"


def _hole_looks_like_response(hole: TemporalHole) -> bool:
    if hole.role:
        return hole.role in {"response", "subresponse", "subformula"}
    return "response" in hole.hole_id.lower()


def _operator_from_source_hint(hole: TemporalHole) -> str:
    if not hole.source_hint:
        return ""
    if hole.source_hint == ("X", "F"):
        return "XF"
    if len(hole.source_hint) == 1:
        return hole.source_hint[0]
    return " ".join(hole.source_hint)


def _operator_from_rag_context(
    *,
    rag_context: tuple[str, ...],
    hole_id: str,
) -> str:
    pattern = re.compile(
        rf"\b{re.escape(hole_id)}\s*=\s*(epsilon|XF|X|F|G)\b",
        flags=re.IGNORECASE,
    )
    for context in rag_context:
        match = pattern.search(context)
        if match:
            return normalize_operator(match.group(1), f"rag_context.{hole_id}")
    return ""


def _has_late_trigger(traces: list[LabeledTrace], trigger: str) -> bool:
    if not trigger:
        return False
    for trace in traces:
        for position, step in enumerate(trace.trace):
            if position > 0 and trigger in step:
                return True
    return False


def _response_operator_from_labeled_traces(
    traces: list[LabeledTrace],
    *,
    semantic: SemanticStructure,
) -> str:
    trigger = semantic.trigger
    response = semantic.response
    saw_same_step = False
    saw_next_step = False
    saw_later_step = False
    for trace in traces:
        if not trace.expected:
            continue
        for position, step in enumerate(trace.trace):
            if trigger not in step:
                continue
            response_positions = [
                index
                for index in range(position, len(trace.trace))
                if response in trace.trace[index]
            ]
            if not response_positions:
                continue
            first_response = response_positions[0]
            if first_response == position:
                saw_same_step = True
            elif first_response == position + 1:
                saw_next_step = True
            else:
                saw_later_step = True

    keyword_operator = _operator_from_semantic_keywords(semantic, "H_response")
    if saw_later_step:
        return "F"
    if saw_next_step:
        return "XF" if keyword_operator == "XF" else "X"
    if saw_same_step:
        return keyword_operator if keyword_operator != "epsilon" else "epsilon"
    return keyword_operator if keyword_operator != "epsilon" else ""


def _keyword_has_global_scope(semantic: SemanticStructure) -> bool:
    keyword_text = " ".join(
        [semantic.scope_keyword, semantic.temporal_keyword, semantic.pattern]
    ).lower()
    return any(
        token in keyword_text
        for token in ("whenever", "always", "globally", "every", "each", "all")
    )


def _operator_from_semantic_keywords(
    semantic: SemanticStructure,
    hole_id: str,
) -> str:
    keyword_text = " ".join(
        [
            semantic.scope_keyword,
            semantic.temporal_keyword,
            semantic.pattern,
        ]
    ).lower()
    grounded_readings = temporal_readings_in_text(keyword_text)
    has_global = bool(
        grounded_readings & {"global_constraint", "persist_to_end"}
    ) or any(
        token in keyword_text
        for token in ("whenever", "every", "each", "all")
    )
    has_next = bool(
        grounded_readings & {"next_step", "eventually_strict"}
    )
    has_eventual = bool(
        grounded_readings & {"eventually", "eventually_strict"}
    )

    if hole_id == "H_scope":
        return "G" if has_global else "epsilon"
    if hole_id == "H_response" or hole_id.startswith("H_response_child_"):
        if has_next and has_eventual:
            return "XF"
        if has_next:
            return "X"
        if has_eventual:
            return "F"
        if has_global:
            return "G"
    return "epsilon"


def _hole_candidate_for_completion(
    hole_candidates: list[HoleCandidate],
    *,
    candidate_id: str,
    positional_candidate: HoleCandidate | None,
) -> HoleCandidate | None:
    if candidate_id:
        return next(
            (
                candidate
                for candidate in hole_candidates
                if candidate.candidate_id == candidate_id
            ),
            None,
        )
    return positional_candidate


def _align_completions_to_hole_candidates(
    *,
    completions: list[TemporalCompletion],
    hole_candidates: list[HoleCandidate],
) -> list[TemporalCompletion]:
    remaining = list(completions)
    aligned: list[TemporalCompletion] = []
    for hole_candidate in hole_candidates:
        match_index = next(
            (
                index
                for index, completion in enumerate(remaining)
                if completion.candidate_id
                and completion.candidate_id == hole_candidate.candidate_id
            ),
            None,
        )
        if match_index is None:
            match_index = next(
                (
                    index
                    for index, completion in enumerate(remaining)
                    if not completion.candidate_id
                ),
                None,
            )
        if match_index is None:
            continue
        completion = remaining.pop(match_index)
        aligned.append(
            replace(
                completion,
                candidate_id=hole_candidate.candidate_id,
                template_text=completion.template_text or hole_candidate.template_text,
            )
        )
    return aligned

def _atom(name: str) -> Formula:
    from ltlf_parser import Atom

    return Atom(name)


def _apply_response_operator(formula: Formula, operator: str) -> Formula:
    atoms = completion_operator_atoms(operator)
    if not atoms:
        return formula
    completed = formula
    for atom in reversed(atoms):
        if atom == "X":
            completed = X(completed)
        elif atom == "F":
            completed = F(completed)
        elif atom == "G":
            completed = G(completed)
        else:
            raise ValueError(f"未知补全算子: {operator}")
    return completed


def _trace_summary_for_retrieval(traces: list[LabeledTrace]) -> str:
    lines = []
    for trace in traces:
        lines.append(
            f"{trace.kind} {trace.name}: {format_trace(trace.trace)} "
            f"expected={format_truth(trace.expected)} reason={trace.reason}"
        )
    return "\n".join(lines)


def _attach_operator_completion_context(
    agent: CandidateCompletionAgent,
    *,
    semantic: SemanticStructure,
    hole_candidates: list[HoleCandidate],
    traces: list[LabeledTrace],
) -> list[HoleCandidate]:
    retrieval_query = "\n".join(
        [
            "RAG + 补全决策",
            semantic.pattern,
            semantic.trigger,
            semantic.response,
            semantic.scope_keyword,
            semantic.temporal_keyword,
            _trace_summary_for_retrieval(traces),
            "\n".join(candidate.template_text for candidate in hole_candidates),
        ]
    )
    rag_context = _sanitize_rag_context(
        agent.retrieve_context(
            retrieval_query,
            top_k=5,
            source_filter=(
                OPERATOR_COMPLETION_SOURCE,
                TEMPLATE_SOURCE,
                FULL_PROCESS_SOURCE,
                "rag",
                "builtin",
                "runtime",
            ),
        )
    )
    if not rag_context:
        return hole_candidates
    return [
        replace(
            candidate,
            rag_context=tuple(dict.fromkeys((*candidate.rag_context, *rag_context))),
        )
        for candidate in hole_candidates
    ]


def _render_formula_from_template(
    hole_candidate: HoleCandidate,
    assignments: tuple[tuple[str, str], ...],
) -> Formula:
    text = normalize_boolean_operator_text(hole_candidate.template_text)
    assignment_map = dict(assignments)
    text = _render_hole_calls(text, assignment_map)
    text = _render_hole_prefixes(text, assignment_map)
    if re.search(r"\bH_[A-Za-z0-9_]*\b", text):
        raise ValueError(f"模板仍包含未补全孔位: {text}")
    return parse_formula(text)


def _validate_completion_assignments(
    *,
    hole_candidate: HoleCandidate,
    assignments: tuple[tuple[str, str], ...],
    index: int,
) -> None:
    """Require exactly one allowed assignment for every reviewed hole."""

    expected_holes = tuple(hole.hole_id for hole in hole_candidate.holes)
    assigned_holes = tuple(hole_id for hole_id, _operator in assignments)
    duplicate_holes = sorted(
        {
            hole_id
            for hole_id in assigned_holes
            if assigned_holes.count(hole_id) > 1
        }
    )
    if duplicate_holes:
        raise ValueError(
            f"completions[{index}].assignments duplicate hole(s): "
            + ", ".join(duplicate_holes)
        )

    missing_holes = [
        hole_id for hole_id in expected_holes if hole_id not in assigned_holes
    ]
    extra_holes = [
        hole_id for hole_id in assigned_holes if hole_id not in expected_holes
    ]
    if missing_holes or extra_holes:
        details: list[str] = []
        if missing_holes:
            details.append("missing=" + ",".join(missing_holes))
        if extra_holes:
            details.append("extra=" + ",".join(extra_holes))
        raise ValueError(
            f"completions[{index}].assignments must bind every reviewed hole "
            f"exactly once ({'; '.join(details)})"
        )

    hole_map = {hole.hole_id: hole for hole in hole_candidate.holes}
    for hole_id, operator in assignments:
        allowed = hole_map[hole_id].allowed_ops
        # Repeated strong-next is the one supported macro assignment.  It has
        # a deterministic schema/expansion (NEXT_CHAIN := X+) and is accepted
        # only when the reviewed hole explicitly permits X.
        operator_atoms = completion_operator_atoms(operator)
        next_chain_allowed = (
            bool(operator_atoms)
            and set(operator_atoms) == {"X"}
            and "X" in allowed
        )
        if operator not in allowed and not next_chain_allowed:
            raise ValueError(
                f"completions[{index}].assignments.{hole_id} selected "
                f"{operator!r} outside allowed_ops={allowed!r}"
            )


def _render_hole_calls(text: str, assignment_map: dict[str, str]) -> str:
    while True:
        best_index = -1
        best_hole = ""
        for hole_id in assignment_map:
            index = text.rfind(f"{hole_id}(")
            if index > best_index:
                best_index = index
                best_hole = hole_id
        if best_index < 0:
            return text

        open_index = best_index + len(best_hole)
        close_index = _matching_paren_index(text, open_index)
        inner = text[open_index + 1 : close_index]
        replacement = _apply_operator_text(inner, assignment_map[best_hole])
        text = text[:best_index] + replacement + text[close_index + 1 :]


def _render_hole_prefixes(text: str, assignment_map: dict[str, str]) -> str:
    for hole_id, operator in assignment_map.items():
        pattern = re.compile(rf"\b{re.escape(hole_id)}([A-Za-z_][A-Za-z0-9_]*)\b")
        text = pattern.sub(
            lambda match: _apply_operator_text(match.group(1), operator),
            text,
        )
    return text


def _matching_paren_index(text: str, open_index: int) -> int:
    if open_index >= len(text) or text[open_index] != "(":
        raise ValueError(f"模板孔位缺少左括号: {text}")
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"模板孔位缺少右括号: {text}")


def _apply_operator_text(inner: str, operator: str) -> str:
    inner = inner.strip()
    atoms = completion_operator_atoms(operator)
    if not atoms:
        return inner
    completed = inner
    for atom in reversed(atoms):
        if atom not in {"X", "F", "G"}:
            raise ValueError(f"未知补全算子: {operator}")
        completed = f"{atom}({completed})"
    return completed


def _ensure_min_hole_candidates(
    candidates: list[HoleCandidate],
    *,
    semantic: SemanticStructure,
    skeletons: list[Formula],
    rag_context: tuple[str, ...],
    maximum: int = 12,
    include_fallback: bool = True,
) -> list[HoleCandidate]:
    unique_candidates: list[HoleCandidate] = []
    seen_templates: set[str] = set()

    def add_candidate(candidate: HoleCandidate) -> None:
        if len(unique_candidates) >= maximum:
            return
        raw_template = candidate.template_text or str(candidate.skeleton)
        if _template_has_repeated_negation(raw_template):
            return
        candidate = _sanitize_hole_candidate(candidate, semantic=semantic, strict=False)
        template_text = _normalize_template_text(candidate.template_text)
        if not template_text or template_text in seen_templates:
            return
        seen_templates.add(template_text)
        if template_text == candidate.template_text:
            unique_candidates.append(candidate)
        else:
            unique_candidates.append(
                HoleCandidate(
                    skeleton=candidate.skeleton,
                    template_text=template_text,
                    holes=candidate.holes,
                    rag_context=candidate.rag_context,
                    explanation=candidate.explanation,
                    candidate_id=candidate.candidate_id,
                )
            )

    for candidate in candidates:
        add_candidate(candidate)

    if include_fallback:
        for candidate in _fallback_hole_candidates(
            semantic=semantic,
            skeletons=skeletons,
            rag_context=rag_context,
        ):
            add_candidate(candidate)

    return _assign_candidate_ids(unique_candidates)


def _sanitize_hole_candidate(
    candidate: HoleCandidate,
    *,
    semantic: SemanticStructure,
    strict: bool,
) -> HoleCandidate:
    template_text = candidate.template_text or str(candidate.skeleton)
    if strict:
        validate_canonical_boolean_operators(template_text, context="template_text")
    template_text = normalize_boolean_operator_text(template_text)
    template_text, inferred_holes = _template_without_concrete_xfg(
        template_text,
        semantic,
        existing_holes=candidate.holes,
    )
    validate_no_forbidden_temporal_operators(
        template_text,
        forbidden=("X", "F", "G"),
        context="template_text",
    )
    template_text = _normalize_template_text(template_text)
    if _template_has_repeated_negation(template_text):
        raise ValueError(
            "template_text 包含重复否定；禁止 !!p 和 !(!p)，"
            "只有 X/F/G 时序算子允许形成一元算子链。"
        )
    return replace(
        candidate,
        template_text=template_text,
        holes=_ensure_template_holes(
            holes=_merge_temporal_holes(
                candidate.holes,
                inferred_holes,
                semantic=semantic,
            ),
            template_text=template_text,
            semantic=semantic,
        ),
    )


def _ensure_template_holes(
    *,
    holes: tuple[TemporalHole, ...],
    template_text: str,
    semantic: SemanticStructure,
) -> tuple[TemporalHole, ...]:
    completed = [
        _enrich_temporal_hole(hole, semantic=semantic)
        for hole in holes
    ]
    existing = {hole.hole_id for hole in completed}
    for hole_id in _hole_ids_in_template(template_text):
        if hole_id in existing:
            continue
        completed.append(_default_hole_for_id(hole_id, semantic))
        existing.add(hole_id)
    by_id = {hole.hole_id: hole for hole in completed}
    ordered = [
        by_id[hole_id]
        for hole_id in _hole_ids_in_template(template_text)
        if hole_id in by_id
    ]
    ordered_ids = {hole.hole_id for hole in ordered}
    ordered.extend(
        hole
        for hole in completed
        if hole.hole_id not in ordered_ids
    )
    return tuple(ordered)


def _merge_temporal_holes(
    existing_holes: tuple[TemporalHole, ...],
    inferred_holes: tuple[TemporalHole, ...],
    *,
    semantic: SemanticStructure,
) -> tuple[TemporalHole, ...]:
    merged: list[TemporalHole] = []
    by_id: dict[str, int] = {}
    for hole in (*existing_holes, *inferred_holes):
        enriched = _enrich_temporal_hole(hole, semantic=semantic)
        index = by_id.get(enriched.hole_id)
        if index is None:
            by_id[enriched.hole_id] = len(merged)
            merged.append(enriched)
            continue
        current = merged[index]
        merged[index] = replace(
            current,
            target=(
                current.target
                if current.target and current.target != "formula fragment"
                else enriched.target
            ),
            allowed_ops=tuple(
                dict.fromkeys((*current.allowed_ops, *enriched.allowed_ops))
            ),
            reason=current.reason or enriched.reason,
            role=current.role or enriched.role,
            parent_hole_id=current.parent_hole_id or enriched.parent_hole_id,
            source_hint=tuple(
                dict.fromkeys((*current.source_hint, *enriched.source_hint))
            ),
        )
    return tuple(merged)


def _enrich_temporal_hole(
    hole: TemporalHole,
    *,
    semantic: SemanticStructure,
) -> TemporalHole:
    default_hole = _default_hole_for_id(hole.hole_id, semantic)
    return replace(
        hole,
        target=hole.target or default_hole.target,
        allowed_ops=hole.allowed_ops or default_hole.allowed_ops,
        reason=hole.reason or default_hole.reason,
        role=hole.role or default_hole.role,
        parent_hole_id=hole.parent_hole_id or default_hole.parent_hole_id,
        source_hint=hole.source_hint or default_hole.source_hint,
    )


def _hole_ids_in_template(template_text: str) -> tuple[str, ...]:
    ids: list[str] = []
    for match in re.finditer(r"\bH_[A-Za-z0-9_]*\b", template_text):
        hole_id = match.group(0)
        if hole_id not in ids:
            ids.append(hole_id)
    return tuple(ids)


def _default_hole_for_id(
    hole_id: str,
    semantic: SemanticStructure,
) -> TemporalHole:
    if hole_id == "H_scope":
        return TemporalHole(
            hole_id=hole_id,
            target="whole formula scope",
            allowed_ops=("epsilon", "G"),
            reason="自动从 template_text 中发现的范围孔位。",
            role="root_scope",
        )
    if hole_id == "H_response":
        return TemporalHole(
            hole_id=hole_id,
            target=semantic.response or "response fragment",
            allowed_ops=("epsilon", "X", "F", "G", "XF"),
            reason="自动从 template_text 中发现的响应孔位。",
            role="response",
        )
    if hole_id.startswith("H_scope_child_"):
        return TemporalHole(
            hole_id=hole_id,
            target="nested non-response temporal fragment",
            allowed_ops=("epsilon", "X", "F", "G"),
            reason="从非 response 子公式中的具体 X/F/G 恢复出的子范围孔位。",
            role="subscope",
        )
    if hole_id.startswith("H_response_child_"):
        return TemporalHole(
            hole_id=hole_id,
            target="nested response temporal fragment",
            allowed_ops=("epsilon", "X", "F", "G"),
            reason="从 response 分支中的具体 X/F/G 恢复出的子响应孔位。",
            role="subresponse",
        )
    return TemporalHole(
        hole_id=hole_id,
        target="formula fragment",
        allowed_ops=("epsilon", "X", "F", "G", "XF"),
        reason="自动从 template_text 中发现的 H_xxx 孔位。",
        role="subformula",
    )


def _template_without_concrete_xfg(
    template_text: str,
    semantic: SemanticStructure,
    *,
    existing_holes: tuple[TemporalHole, ...] = (),
) -> tuple[str, tuple[TemporalHole, ...]]:
    normalized_text = normalize_boolean_operator_text(template_text)
    try:
        converted_text, inferred_holes = _convert_concrete_temporal_to_child_holes(
            normalized_text,
            semantic=semantic,
            existing_holes=existing_holes,
        )
        validate_no_forbidden_temporal_operators(
            converted_text,
            forbidden=("X", "F", "G"),
            context="template_text",
        )
        return converted_text, inferred_holes
    except ValueError:
        pass

    repaired_text = _repair_concrete_temporal_template_text(
        normalized_text,
        semantic,
    )
    try:
        validate_no_forbidden_temporal_operators(
            repaired_text,
            forbidden=("X", "F", "G"),
            context="template_text",
        )
        return repaired_text, ()
    except ValueError:
        pass

    try:
        validate_no_forbidden_temporal_operators(
            template_text,
            forbidden=("X", "F", "G"),
            context="template_text",
        )
        return template_text, ()
    except ValueError:
        pass

    try:
        formula = parse_formula(template_text)
    except ValueError:
        return template_text, ()
    return _formula_to_hole_template(formula, semantic), ()


@dataclass(frozen=True)
class _TemplateNode:
    op: str
    args: tuple["_TemplateNode", ...] = ()
    name: str = ""


@dataclass
class _ChildHoleMappingState:
    used_hole_ids: set[str]
    inferred_holes: list[TemporalHole]
    scope_child_index: int = 1
    response_child_index: int = 1


def _convert_concrete_temporal_to_child_holes(
    template_text: str,
    *,
    semantic: SemanticStructure,
    existing_holes: tuple[TemporalHole, ...],
) -> tuple[str, tuple[TemporalHole, ...]]:
    root = _parse_template_node(template_text)
    used_hole_ids = _collect_template_hole_ids(root)
    used_hole_ids.update(hole.hole_id for hole in existing_holes)
    state = _ChildHoleMappingState(
        used_hole_ids=used_hole_ids,
        inferred_holes=[],
    )
    converted = _map_temporal_template_node(
        root,
        semantic=semantic,
        state=state,
        context="scope",
        is_root=True,
        parent_hole_id="",
    )
    return (
        _normalize_template_text(_template_node_to_text(converted)),
        tuple(state.inferred_holes),
    )


def _parse_template_node(text: str) -> _TemplateNode:
    masked_text, masked_holes = _mask_template_hole_calls(text)
    formula = parse_formula(masked_text)
    return _template_node_from_formula(formula, masked_holes)


def _mask_template_hole_calls(
    text: str,
) -> tuple[str, dict[str, tuple[str, str]]]:
    output: list[str] = []
    holes: dict[str, tuple[str, str]] = {}
    index = 0
    placeholder_index = 1
    pattern = re.compile(r"H_[A-Za-z0-9_]*\s*\(")
    while index < len(text):
        match = pattern.match(text, index)
        if match is None:
            output.append(text[index])
            index += 1
            continue
        hole_id_match = re.match(r"H_[A-Za-z0-9_]*", match.group(0))
        if hole_id_match is None:
            output.append(text[index])
            index += 1
            continue
        hole_id = hole_id_match.group(0)
        open_index = text.find("(", match.start(), match.end())
        close_index = _matching_paren_index(text, open_index)
        placeholder = f"_template_hole_placeholder_{placeholder_index}"
        placeholder_index += 1
        holes[placeholder] = (
            hole_id,
            text[open_index + 1 : close_index],
        )
        output.append(placeholder)
        index = close_index + 1
    return "".join(output), holes


def _template_node_from_formula(
    formula: Formula,
    masked_holes: dict[str, tuple[str, str]],
) -> _TemplateNode:
    if formula.op == "ap" and formula.name in masked_holes:
        hole_id, inner_text = masked_holes[formula.name]
        return _TemplateNode(
            "hole",
            (_parse_template_node(inner_text),),
            hole_id,
        )
    return _TemplateNode(
        formula.op,
        tuple(
            _template_node_from_formula(child, masked_holes)
            for child in formula.args
        ),
        formula.name,
    )


def _collect_template_hole_ids(node: _TemplateNode) -> set[str]:
    ids = {node.name} if node.op == "hole" else set()
    for child in node.args:
        ids.update(_collect_template_hole_ids(child))
    return ids


def _map_temporal_template_node(
    node: _TemplateNode,
    *,
    semantic: SemanticStructure,
    state: _ChildHoleMappingState,
    context: str,
    is_root: bool,
    parent_hole_id: str,
) -> _TemplateNode:
    if node.op == "hole":
        default_hole = _default_hole_for_id(node.name, semantic)
        role = default_hole.role
        state.inferred_holes.append(
            replace(
                default_hole,
                parent_hole_id=parent_hole_id,
            )
        )
        child_context = (
            "response"
            if role in {"response", "subresponse"}
            else "scope"
            if role in {"root_scope", "subscope"}
            else context
        )
        child = _map_temporal_template_node(
            node.args[0],
            semantic=semantic,
            state=state,
            context=child_context,
            is_root=False,
            parent_hole_id=node.name,
        )
        return replace(node, args=(child,))

    if node.op in {"X", "F", "G"}:
        child = node.args[0]
        hole_id, role = _child_hole_identity(
            child,
            semantic=semantic,
            state=state,
            context=context,
            is_root=is_root,
        )
        state.used_hole_ids.add(hole_id)
        mapped_child = _map_temporal_template_node(
            child,
            semantic=semantic,
            state=state,
            context=context,
            is_root=False,
            parent_hole_id=hole_id,
        )
        allowed_ops = (
            ("epsilon", "X", "F", "G", "XF")
            if role == "response"
            else ("epsilon", "X", "F", "G")
        )
        state.inferred_holes.append(
            TemporalHole(
                hole_id=hole_id,
                target=_template_node_to_text(child),
                allowed_ops=allowed_ops,
                reason=(
                    f"由模板中提前具体化的 {node.op} 恢复为未决子孔位；"
                    "source_hint 仅作为补全参考。"
                ),
                role=role,
                parent_hole_id=parent_hole_id,
                source_hint=(node.op,),
            )
        )
        return _TemplateNode("hole", (mapped_child,), hole_id)

    if node.op == "implies":
        left = _map_temporal_template_node(
            node.args[0],
            semantic=semantic,
            state=state,
            context="scope",
            is_root=False,
            parent_hole_id=parent_hole_id,
        )
        right = _map_temporal_template_node(
            node.args[1],
            semantic=semantic,
            state=state,
            context="response",
            is_root=False,
            parent_hole_id=parent_hole_id,
        )
        return replace(node, args=(left, right))

    return replace(
        node,
        args=tuple(
            _map_temporal_template_node(
                child,
                semantic=semantic,
                state=state,
                context=context,
                is_root=False,
                parent_hole_id=parent_hole_id,
            )
            for child in node.args
        ),
    )


def _child_hole_identity(
    child: _TemplateNode,
    *,
    semantic: SemanticStructure,
    state: _ChildHoleMappingState,
    context: str,
    is_root: bool,
) -> tuple[str, str]:
    if is_root and "H_scope" not in state.used_hole_ids:
        return "H_scope", "root_scope"
    child_hole_ids = _collect_template_hole_ids(child)
    if (
        "H_response" not in state.used_hole_ids
        and _template_node_matches_response(child, semantic)
    ):
        return "H_response", "response"
    if (
        context == "response"
        or any(
            hole_id == "H_response"
            or hole_id.startswith("H_response_child_")
            for hole_id in child_hole_ids
        )
    ):
        return (
            _next_child_hole_id(
                prefix="H_response_child_",
                state=state,
            ),
            "subresponse",
        )
    return (
        _next_child_hole_id(
            prefix="H_scope_child_",
            state=state,
        ),
        "subscope",
    )


def _next_child_hole_id(
    *,
    prefix: str,
    state: _ChildHoleMappingState,
) -> str:
    attribute = (
        "response_child_index"
        if prefix.startswith("H_response")
        else "scope_child_index"
    )
    while True:
        index = getattr(state, attribute)
        setattr(state, attribute, index + 1)
        hole_id = f"{prefix}{index}"
        if hole_id not in state.used_hole_ids:
            return hole_id


def _template_node_matches_response(
    node: _TemplateNode,
    semantic: SemanticStructure,
) -> bool:
    if _collect_template_hole_ids(node):
        return False
    node_text = _normalize_template_text(_template_node_to_text(node))
    response_texts = [
        semantic.response,
        _semantic_response_endpoint_texts(semantic)[1],
    ]
    for response_text in response_texts:
        response_text = normalize_boolean_operator_text(response_text.strip())
        if not response_text:
            continue
        try:
            expected = str(_strip_concrete_xfg(parse_formula(response_text)))
        except ValueError:
            expected = response_text
        if node_text == _normalize_template_text(expected):
            return True
    return False


def _template_node_to_text(node: _TemplateNode) -> str:
    if node.op == "ap":
        return node.name
    if node.op in {"true", "false"}:
        return node.op
    if node.op == "hole":
        return f"{node.name}({_template_node_to_text(node.args[0])})"
    if node.op == "not":
        child = node.args[0]
        child_text = _template_node_to_text(child)
        if child.op in {"and", "or", "implies", "U"}:
            return f"!({child_text})"
        return f"!{child_text}"
    if node.op in {"X", "F", "G"}:
        child = node.args[0]
        child_text = _template_node_to_text(child)
        if child.op in {"and", "or", "implies", "U"}:
            return f"{node.op}({child_text})"
        return f"{node.op} {child_text}"
    if node.op in {"and", "or", "implies", "U"}:
        left, right = node.args
        symbol = {
            "and": " & ",
            "or": " | ",
            "implies": " -> ",
            "U": " U ",
        }[node.op]
        left_text = _template_node_to_text(left)
        right_text = _template_node_to_text(right)
        if left.op == "implies":
            left_text = f"({left_text})"
        if right.op in {"and", "or", "implies", "U"}:
            right_text = f"({right_text})"
        return f"{left_text}{symbol}{right_text}"
    raise ValueError(f"未知模板节点类型: {node.op}")


def _repair_concrete_temporal_template_text(
    template_text: str,
    semantic: SemanticStructure,
) -> str:
    text = _replace_outer_temporal_scope_text(template_text)
    text = _replace_temporal_wrappers_around_response_holes(text)
    text = _replace_temporal_response_wrappers(text, semantic)
    text = _replace_temporal_wrappers_around_response_holes(text)
    if "H_response(" in text and "H_scope(" not in text:
        text = f"H_scope({text})"
    return _normalize_template_text(text)


def _replace_outer_temporal_scope_text(text: str) -> str:
    stripped = text.strip()
    for operator in ("G", "F", "X"):
        match = re.match(rf"^{operator}\s*\(", stripped)
        if match is None:
            continue
        open_index = match.end() - 1
        try:
            close_index = _matching_paren_index(stripped, open_index)
        except ValueError:
            return text
        if stripped[close_index + 1 :].strip():
            return text
        inner = stripped[open_index + 1 : close_index].strip()
        return f"H_scope({inner})"
    return text


def _replace_temporal_wrappers_around_response_holes(text: str) -> str:
    previous = None
    current = text
    parenthesized_pattern = re.compile(
        r"\b[XFG]\s*\(\s*(H_response\s*\([^()]*\))\s*\)"
    )
    prefix_pattern = re.compile(
        r"\b[XFG]\s+(H_response\s*\([^()]*\))"
    )
    while previous != current:
        previous = current
        current = parenthesized_pattern.sub(r"\1", current)
        current = prefix_pattern.sub(r"\1", current)
    return current


def _replace_temporal_response_wrappers(
    text: str,
    semantic: SemanticStructure,
) -> str:
    response_texts = []
    for value in (semantic.response, _semantic_response_endpoint_texts(semantic)[1]):
        value = value.strip()
        if value and value not in response_texts:
            response_texts.append(value)

    for response_text in response_texts:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", response_text):
            continue
        escaped = re.escape(response_text)
        text = re.sub(
            rf"\b[XFG]\s*\(\s*{escaped}\s*\)",
            f"H_response({response_text})",
            text,
        )
        text = re.sub(
            rf"\b[XFG]\s+{escaped}\b",
            f"H_response({response_text})",
            text,
        )
    return text


def _assign_candidate_ids(candidates: list[HoleCandidate]) -> list[HoleCandidate]:
    assigned: list[HoleCandidate] = []
    used_ids: set[str] = set()
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = candidate.candidate_id or f"HC-{index:03d}"
        if candidate_id in used_ids:
            candidate_id = f"HC-{index:03d}"
        used_ids.add(candidate_id)
        assigned.append(replace(candidate, candidate_id=candidate_id))
    return assigned


def _fallback_hole_candidates(
    *,
    semantic: SemanticStructure,
    skeletons: list[Formula],
    rag_context: tuple[str, ...],
) -> list[HoleCandidate]:
    skeleton_list = list(skeletons)
    if not skeleton_list:
        skeleton_list = [_fallback_skeleton(semantic)]

    candidates: list[HoleCandidate] = []
    seen_templates: set[str] = set()
    holes = _default_response_holes(semantic) if semantic.pattern == "response" else ()
    explanation = (
        "确定性枚举所有可进入补全阶段的带孔候选模板；轨迹只在后续填充孔位算子时作为参考。"
    )

    def add_templates(skeleton: Formula, templates: list[str]) -> None:
        for template in templates:
            template_text = _normalize_template_text(template)
            if not template_text or template_text in seen_templates:
                continue
            seen_templates.add(template_text)
            candidates.append(
                HoleCandidate(
                    skeleton=skeleton,
                    template_text=template_text,
                    holes=holes,
                    rag_context=rag_context,
                    explanation=explanation,
                    candidate_id=f"FB-{len(candidates) + 1:03d}",
                )
            )

    for candidate in _semantic_pattern_hole_candidates(
        semantic=semantic,
        skeletons=skeleton_list,
        rag_context=rag_context,
        explanation=explanation,
    ):
        template_text = _normalize_template_text(candidate.template_text)
        if not template_text or template_text in seen_templates:
            continue
        seen_templates.add(template_text)
        candidates.append(
            replace(
                candidate,
                candidate_id=candidate.candidate_id or f"FB-{len(candidates) + 1:03d}",
                explanation=(
                    candidate.explanation
                    or explanation
                ),
            )
        )

    if semantic.pattern == "response":
        add_templates(skeleton_list[0], _response_template_variants(semantic, None))
        for skeleton in skeleton_list:
            add_templates(skeleton, _response_template_variants(semantic, skeleton))
    for skeleton in skeleton_list:
        add_templates(skeleton, _generic_template_variants(str(skeleton)))

    return candidates


def _semantic_pattern_hole_candidates(
    *,
    semantic: SemanticStructure,
    skeletons: list[Formula],
    rag_context: tuple[str, ...],
    explanation: str,
) -> list[HoleCandidate]:
    pattern = semantic.pattern.strip().lower().replace("-", "_").replace(" ", "_")
    skeleton = skeletons[0] if skeletons else _fallback_skeleton(semantic)
    if pattern == "precedence":
        constrained, prerequisite = precedence_roles_from_semantic(semantic)
        if not constrained or not prerequisite or constrained == prerequisite:
            return []
        return [
            HoleCandidate(
                skeleton=skeleton,
                template_text=(
                    f"H_scope(!{constrained}) | "
                    f"(!{constrained} U ({prerequisite} & !{constrained}))"
                ),
                holes=(
                    TemporalHole(
                        hole_id="H_scope",
                        target=f"no {constrained} anywhere",
                        allowed_ops=("G",),
                        reason=(
                            "strict precedence is encoded with a global no-event "
                            "vacuous branch plus an until branch where the prior "
                            "event occurs before the constrained event."
                        ),
                    ),
                ),
                rag_context=rag_context,
                explanation=explanation,
            )
        ]

    if pattern == "recurrence":
        ap = _primary_semantic_ap(semantic)
        if not ap:
            return []
        period = _period_from_text(_semantic_completion_evidence_text(semantic), default=5)
        template_text, holes = _recurrence_template(ap, period)
        return [
            HoleCandidate(
                skeleton=skeleton,
                template_text=template_text,
                holes=holes,
                rag_context=rag_context,
                explanation=explanation,
            )
        ]

    if pattern == "periodic":
        ap = _primary_semantic_ap(semantic)
        if not ap:
            return []
        period = _period_from_text(_semantic_completion_evidence_text(semantic), default=3)
        template_text, holes = _periodic_spacing_template(ap, period)
        return [
            HoleCandidate(
                skeleton=skeleton,
                template_text=template_text,
                holes=holes,
                rag_context=rag_context,
                explanation=explanation,
            )
        ]

    if pattern == "cardinality_constraint":
        ap = _primary_semantic_ap(semantic)
        if not ap:
            return []
        evidence = _semantic_completion_evidence_text(semantic)
        counted = f"!{ap}" if _cardinality_counts_absence(evidence, ap) else ap
        limit = _cardinality_limit_from_text(evidence, default=2)
        if limit != 2:
            return []
        return [
            HoleCandidate(
                skeleton=skeleton,
                template_text=(
                    f"!H_count_1({counted} & "
                    f"H_count_2({counted} & H_count_3({counted})))"
                ),
                holes=(
                    TemporalHole(
                        "H_count_1",
                        f"first occurrence of {counted}",
                        ("F",),
                        "at-most-two is encoded as absence of three ordered occurrences.",
                    ),
                    TemporalHole(
                        "H_count_2",
                        f"second strict-later occurrence of {counted}",
                        ("XF",),
                        "the second counted occurrence must be strictly after the first.",
                    ),
                    TemporalHole(
                        "H_count_3",
                        f"third strict-later occurrence of {counted}",
                        ("XF",),
                        "reject traces with a third counted occurrence.",
                    ),
                ),
                rag_context=rag_context,
                explanation=explanation,
            )
        ]

    return []


def _recurrence_template(
    ap: str,
    period: int,
) -> tuple[str, tuple[TemporalHole, ...]]:
    period = max(1, period)
    inner = ap
    holes: list[TemporalHole] = []
    for index in range(period, 0, -1):
        hole_id = f"H_step_{index}"
        inner = f"{hole_id}({inner})"
    for index in range(1, period + 1):
        holes.append(
            TemporalHole(
                hole_id=f"H_step_{index}",
                target=f"{ap} after {index} step(s)",
                allowed_ops=("X",),
                reason="recurrence step distance is represented by an explicit X-chain.",
            )
        )
    return f"{ap} & {inner}", tuple(holes)


def _periodic_spacing_template(
    ap: str,
    period: int,
) -> tuple[str, tuple[TemporalHole, ...]]:
    gap = max(1, period - 1)
    inner = f"!{ap}"
    for index in range(gap, 0, -1):
        hole_id = f"H_gap_{index}"
        if index == gap:
            inner = f"{hole_id}({inner})"
        else:
            inner = f"{hole_id}(!{ap} & {inner})"
    holes = tuple(
        TemporalHole(
            hole_id=f"H_gap_{index}",
            target=f"{ap} must be absent in gap step {index}",
            allowed_ops=("X",),
            reason="periodic spacing is represented by explicit next-step gap checks.",
        )
        for index in range(1, gap + 1)
    )
    return (
        f"H_scope({ap} -> {inner})",
        (
            TemporalHole(
                "H_scope",
                "periodic constraint over all finite-trace positions",
                ("G",),
                "periodic spacing applies globally to every occurrence.",
            ),
            *holes,
        ),
    )


def _primary_semantic_ap(semantic: SemanticStructure) -> str:
    for ap in (*semantic.aps, semantic.trigger, semantic.response):
        if ap and ap not in {"unknown", "unknown_ap"}:
            return ap
    return ""


def _semantic_completion_evidence_text(semantic: SemanticStructure) -> str:
    component_text = " ".join(
        f"{key} {value}" for key, value in semantic.components
    )
    return " ".join(
        [
            semantic.pattern,
            semantic.scope_keyword,
            semantic.temporal_keyword,
            semantic.explanation,
            component_text,
        ]
    ).lower()


def _period_from_text(text: str, *, default: int) -> int:
    ordinal_numbers = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    cardinal_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    patterns = (
        r"every[_\s-]+(\d+)",
        r"every[_\s-]+([A-Za-z]+)",
        r"(\d+)(?:st|nd|rd|th)?[_\s-]+step",
        r"([A-Za-z]+)[_\s-]+step",
        r"period[_\s-]*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1).lower()
        if raw.isdigit():
            return max(1, int(raw))
        if raw in ordinal_numbers:
            return ordinal_numbers[raw]
        if raw in cardinal_numbers:
            return cardinal_numbers[raw]
    return default


def _cardinality_limit_from_text(text: str, *, default: int) -> int:
    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    match = re.search(
        r"(?:at\s+most|no\s+more\s+than|<=)\s+(\d+|[A-Za-z]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return default
    raw = match.group(1).lower()
    if raw.isdigit():
        return max(0, int(raw))
    return number_words.get(raw, default)


def _cardinality_counts_absence(text: str, ap: str) -> bool:
    escaped = re.escape(ap)
    return bool(
        re.search(rf"\bnot\s+{escaped}\b", text)
        or re.search(rf"!\s*{escaped}\b", text)
    )


def _fallback_skeleton(semantic: SemanticStructure) -> Formula:
    trigger_text, response_text = _semantic_response_endpoint_texts(semantic)
    for text in (trigger_text, response_text, semantic.trigger, semantic.response):
        if not text:
            continue
        try:
            return parse_formula(text)
        except ValueError:
            continue
    return parse_formula("fallback_ap")


def _default_response_holes(semantic: SemanticStructure) -> tuple[TemporalHole, ...]:
    _trigger_text, response_text = _semantic_response_endpoint_texts(semantic)
    return (
        TemporalHole(
            hole_id="H_scope",
            target="whole implication",
            allowed_ops=("epsilon", "G"),
            reason="scope keyword 和非 t0 触发轨迹用于判断约束是否需要全局成立。",
        ),
        TemporalHole(
            hole_id="H_response",
            target=response_text,
            allowed_ops=("epsilon", "X", "F", "G", "XF"),
            reason="temporal keyword、正轨迹、反轨迹和边界轨迹共同用于区分当前、下一步、未来某步或持续约束。",
        ),
    )


def _formula_template_variants(formula: Formula) -> list[str]:
    variants: list[str] = []
    base_text = _formula_to_hole_template(formula, None)
    _extend_unique(variants, [base_text, _operator_alias_text(base_text)])

    parts = _response_implication_parts(formula)
    if parts is not None:
        _scope, trigger, response = parts
        trigger_aliases = _formula_aliases(_strip_concrete_xfg(trigger))
        for trigger_text in trigger_aliases:
            _extend_unique(
                variants,
                [f"H_scope({trigger_text} -> H_response({_unwrap_temporal_for_hole(response)}))"],
            )
        response_base = _unwrap_temporal_for_hole(response)
        _extend_unique(variants, _response_equivalent_templates(trigger_aliases[0], response_base))
        return variants

    _extend_unique(variants, _generic_template_variants(base_text))
    return variants


def _response_template_variants(
    semantic: SemanticStructure,
    formula: Formula | None,
) -> list[str]:
    variants: list[str] = []
    parts = _response_implication_parts(formula) if formula is not None else None
    if parts is not None:
        _scope, trigger, response = parts
        trigger_aliases = _formula_aliases(_strip_concrete_xfg(trigger))
        response_base = _unwrap_temporal_for_hole(response)
        for trigger_text in trigger_aliases:
            _extend_unique(
                variants,
                [f"H_scope({trigger_text} -> H_response({response_base}))"],
            )
        _extend_unique(variants, _response_equivalent_templates(trigger_aliases[0], response_base))
        return variants

    trigger = semantic.trigger or "trigger"
    response = semantic.response or "response"
    trigger, response = _semantic_response_endpoint_texts(semantic)
    try:
        trigger_aliases = _formula_aliases(
            _strip_concrete_xfg(parse_formula(trigger))
        )
    except ValueError:
        trigger_aliases = [trigger]
    variants: list[str] = []
    for trigger_text in trigger_aliases:
        _extend_unique(
            variants,
            [f"H_scope({trigger_text} -> H_response({response}))"],
        )
    _extend_unique(variants, _response_equivalent_templates(trigger_aliases[0], response))
    return variants


def _response_equivalent_templates(trigger: str, response: str) -> list[str]:
    trigger_p = _parenthesized_template_text(trigger)
    not_trigger = _negated_template_text(trigger)
    response_hole = f"H_response({response})"
    return [
        f"H_scope({trigger} -> {response_hole})",
        f"H_scope(({trigger}) -> {response_hole})",
        f"H_scope((({trigger})) -> {response_hole})",
        f"H_scope(({trigger} -> {response_hole}))",
        f"H_scope((({trigger}) -> {response_hole}))",
        f"H_scope(!({trigger} & !{response_hole}))",
        f"H_scope(!({trigger_p} & !{response_hole}))",
        f"H_scope(!(({trigger_p}) & (!{response_hole})))",
        f"H_scope({response_hole} | {not_trigger})",
        f"H_scope(({response_hole}) | ({not_trigger}))",
        f"H_scope(({not_trigger}) | {response_hole})",
        f"H_scope(!{trigger_p} | {response_hole})",
        f"H_scope(!(!{response_hole} & {trigger_p}))",
        f"H_scope(!((!{response_hole}) & ({trigger_p})))",
    ]


def _response_implication_parts(
    formula: Formula | None,
) -> tuple[str, Formula, Formula] | None:
    if formula is None:
        return None
    if formula.op == "G" and formula.args and formula.args[0].op == "implies":
        left, right = formula.args[0].args
        return "G", left, right
    if formula.op == "implies":
        left, right = formula.args
        return "", left, right
    return None


def _formula_to_hole_template(
    formula: Formula,
    semantic: SemanticStructure | None,
) -> str:
    parts = _response_implication_parts(formula)
    if parts is not None:
        _scope, trigger, response = parts
        trigger_text = _unwrap_temporal_for_hole(trigger)
        response_text = _unwrap_temporal_for_hole(response)
        return f"H_scope({trigger_text} -> H_response({response_text}))"

    stripped = _strip_concrete_xfg(formula)
    text = str(stripped)
    if semantic is not None and semantic.response:
        text = re.sub(
            rf"\b{re.escape(semantic.response)}\b",
            f"H_response({semantic.response})",
            text,
            count=1,
        )
    return f"H_scope({text})"


def _strip_concrete_xfg(formula: Formula) -> Formula:
    if formula.op in {"X", "F", "G"} and formula.args:
        return _strip_concrete_xfg(formula.args[0])
    if not formula.args:
        return formula
    return Formula(
        formula.op,
        tuple(_strip_concrete_xfg(child) for child in formula.args),
        formula.name,
    )


def _formula_aliases(formula: Formula) -> list[str]:
    aliases: list[str] = []
    base_text = str(formula)
    _extend_unique(aliases, [base_text])
    _extend_unique(aliases, _demorgan_aliases(formula))
    _extend_unique(aliases, [_operator_alias_text(base_text)])
    if formula.op in {"and", "or", "implies", "U"}:
        _extend_unique(aliases, [f"({base_text})"])
    return aliases


def _demorgan_aliases(formula: Formula) -> list[str]:
    if formula.op == "and":
        operands = _flatten_binary(formula, "and")
        if all(item.op == "not" and item.args for item in operands):
            positive_texts = [
                _nested_formula_text(item.args[0])
                for item in operands
            ]
            return [
                f"!({' & '.join(positive_texts)})",
            ]
    if formula.op == "not" and formula.args and formula.args[0].op == "and":
        operands = _flatten_binary(formula.args[0], "and")
        negative_texts = [
            f"!{_nested_formula_text(item)}"
            for item in operands
        ]
        return [
            " & ".join(negative_texts),
        ]
    return []


def _flatten_binary(formula: Formula, operator: str) -> tuple[Formula, ...]:
    if formula.op != operator:
        return (formula,)
    left, right = formula.args
    return (
        *_flatten_binary(left, operator),
        *_flatten_binary(right, operator),
    )


def _nested_formula_text(formula: Formula) -> str:
    text = str(formula)
    if formula.op in {"and", "or", "implies", "U"}:
        return f"({text})"
    return text


def _parenthesized_template_text(text: str) -> str:
    text = text.strip()
    if re.fullmatch(r"!?[A-Za-z_][A-Za-z0-9_]*", text):
        return text
    if text.startswith("(") and text.endswith(")"):
        return text
    return f"({text})"


def _negated_template_text(text: str) -> str:
    text = text.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return f"!{text}"
    return f"!{_parenthesized_template_text(text)}"


def _unwrap_temporal_for_hole(formula: Formula) -> str:
    current = formula
    while current.op in {"X", "F", "G"} and current.args:
        current = current.args[0]
    return str(_strip_concrete_xfg(current))


def _wrap_scope(scope: str, body: str) -> str:
    if scope:
        return f"{scope}({body})"
    return body


def _generic_template_variants(template_text: str) -> list[str]:
    base = _normalize_template_text(normalize_boolean_operator_text(template_text))
    if not base:
        return []
    variants = [base, _operator_alias_text(base)]
    variants.extend(
        [
            f"({base})",
            f"(({base}))",
            f"H_scope({base})",
            f"H_scope(({base}))",
            f"H_scope((({base})))",
        ]
    )
    return variants


def _template_has_repeated_negation(template_text: str) -> bool:
    parseable_text = re.sub(
        r"\bH_[A-Za-z0-9_]*\b",
        "X",
        normalize_boolean_operator_text(template_text),
    )
    try:
        return has_repeated_negation(parse_formula(parseable_text))
    except ValueError:
        return False


def _operator_alias_text(text: str) -> str:
    return normalize_boolean_operator_text(text)


def _normalize_template_text(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_boolean_operator_text(text).strip())


def _extend_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        normalized = _normalize_template_text(value)
        if normalized and normalized not in target:
            target.append(normalized)


def _hole_candidates_from_template_examples(
    *,
    template_examples: list[tuple[object, float]],
    semantic: SemanticStructure,
    skeletons: list[Formula],
    rag_context: tuple[str, ...],
) -> list[HoleCandidate]:
    candidates: list[HoleCandidate] = []
    seen: set[tuple[str, str]] = set()
    for chunk, _score in template_examples:
        parsed = _parse_template_example(chunk.text, semantic)
        if parsed is None:
            continue
        template_text, holes, explanation = parsed
        for skeleton in skeletons:
            key = (str(skeleton), template_text)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                HoleCandidate(
                    skeleton=skeleton,
                    template_text=template_text,
                    holes=holes,
                    rag_context=rag_context,
                    explanation=explanation,
                )
            )
    return candidates


def _parse_template_example(
    text: str,
    semantic: SemanticStructure,
) -> tuple[str, tuple[TemporalHole, ...], str] | None:
    template_text = _read_example_field(text, "template_text")
    if not template_text:
        return None

    example_trigger = _read_example_field(text, "trigger")
    example_response = _read_example_field(text, "response")
    adapted_template = _adapt_example_text(
        template_text,
        semantic=semantic,
        example_trigger=example_trigger,
        example_response=example_response,
    )
    adapted_template, inferred_holes = _template_without_concrete_xfg(
        normalize_boolean_operator_text(adapted_template),
        semantic,
    )

    holes: list[TemporalHole] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- hole_id="):
            continue
        mapping = _parse_semicolon_fields(stripped[2:])
        hole_id = mapping.get("hole_id")
        allowed_ops = mapping.get("allowed_ops")
        if not hole_id or not allowed_ops:
            continue
        target = _adapt_example_text(
            mapping.get("target", "formula fragment"),
            semantic=semantic,
            example_trigger=example_trigger,
            example_response=example_response,
        )
        reason = _adapt_example_text(
            mapping.get("reason", "来自相似正确模板样例。"),
            semantic=semantic,
            example_trigger=example_trigger,
            example_response=example_response,
        )
        holes.append(
            TemporalHole(
                hole_id=hole_id,
                target=target,
                allowed_ops=normalize_operator_list(
                    [item.strip() for item in allowed_ops.split(",") if item.strip()],
                    f"template_example.{hole_id}.allowed_ops",
                ),
                reason=reason,
                role=mapping.get("role", ""),
                parent_hole_id=mapping.get("parent_hole_id", ""),
                source_hint=tuple(
                    item.strip()
                    for item in mapping.get("source_hint", "").split(",")
                    if item.strip()
                ),
            )
        )

    merged_holes = _merge_temporal_holes(
        tuple(holes),
        inferred_holes,
        semantic=semantic,
    )
    if not merged_holes:
        return None

    return (
        adapted_template,
        _ensure_template_holes(
            holes=merged_holes,
            template_text=adapted_template,
            semantic=semantic,
        ),
        "从 Top-5 相似正确带孔模板 RAG 中迁移，并替换为当前 AP。",
    )


def _read_example_field(text: str, field_name: str) -> str:
    match = re.search(rf"^{re.escape(field_name)}\s*:\s*(.*?)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _parse_semicolon_fields(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in text.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _adapt_example_text(
    text: str,
    *,
    semantic: SemanticStructure,
    example_trigger: str,
    example_response: str,
) -> str:
    adapted = text
    if example_trigger and example_trigger != semantic.trigger:
        adapted = re.sub(
            rf"\b{re.escape(example_trigger)}\b",
            semantic.trigger,
            adapted,
        )
    if example_response and example_response != semantic.response:
        adapted = re.sub(
            rf"\b{re.escape(example_response)}\b",
            semantic.response,
            adapted,
        )
    return adapted


def _resolve_skeleton(mapping: dict, skeletons: list[Formula], index: int) -> Formula:
    raw_index = mapping.get("skeleton_index")
    if isinstance(raw_index, int):
        if 0 <= raw_index < len(skeletons):
            return skeletons[raw_index]
        if 1 <= raw_index <= len(skeletons):
            return skeletons[raw_index - 1]
        raise ValueError(f"hole_candidates[{index}].skeleton_index 越界。")

    raw_skeleton = mapping.get("skeleton")
    if isinstance(raw_skeleton, str) and raw_skeleton.strip():
        return parse_formula(raw_skeleton.strip())

    if len(skeletons) == 1:
        return skeletons[0]
    raise ValueError(f"hole_candidates[{index}] 需要 skeleton_index 或 skeleton。")


def _hole_candidate_mapping_from_llm(
    item: object,
    index: int,
    skeletons: list[Formula],
) -> dict:
    if isinstance(item, str) and item.strip():
        mapping: dict[str, object] = {
            "template_text": item.strip(),
            "holes": [],
        }
        if skeletons:
            mapping["skeleton_index"] = min(index - 1, len(skeletons) - 1)
        return mapping
    mapping = require_mapping(item, f"hole_candidates[{index}]")
    if "holes" not in mapping:
        mapping = {**mapping, "holes": []}
    return mapping


def _hole_template_text_from_llm(mapping: dict, *, default: str) -> str:
    for field in (
        "template_text",
        "template",
        "hole_template",
        "template_formula",
        "candidate_template",
    ):
        value = mapping.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _temporal_hole_from_llm(
    item: object,
    candidate_index: int,
    hole_index: int,
    *,
    semantic: SemanticStructure,
) -> TemporalHole:
    mapping = require_mapping(item, f"hole_candidates[{candidate_index}].holes[{hole_index}]")
    hole_id = require_string(mapping, "hole_id", f"hole_candidates[{candidate_index}].holes[{hole_index}]")
    default_hole = _default_hole_for_id(hole_id, semantic)
    return TemporalHole(
        hole_id=hole_id,
        target=optional_string(mapping, "target", default_hole.target),
        allowed_ops=_allowed_ops_from_llm(
            mapping,
            default_hole=default_hole,
            context=f"hole_candidates[{candidate_index}].holes[{hole_index}].allowed_ops",
        ),
        reason=optional_string(mapping, "reason", default_hole.reason),
        role=optional_string(mapping, "role", default_hole.role),
        parent_hole_id=optional_string(
            mapping,
            "parent_hole_id",
            default_hole.parent_hole_id,
        ),
        source_hint=_source_hint_from_llm(mapping, default_hole=default_hole),
    )


def _source_hint_from_llm(
    mapping: dict,
    *,
    default_hole: TemporalHole,
) -> tuple[str, ...]:
    raw_value = _first_present_value(
        mapping,
        ("source_hint", "source_operator", "operator_hint"),
    )
    if raw_value is None:
        return default_hole.source_hint
    if isinstance(raw_value, list):
        hints: list[str] = []
        for item in raw_value:
            if not isinstance(item, str):
                continue
            hints.extend(completion_operator_atoms(item))
        return tuple(hints)
    if isinstance(raw_value, str):
        return completion_operator_atoms(raw_value)
    return default_hole.source_hint


def _allowed_ops_from_llm(
    mapping: dict,
    *,
    default_hole: TemporalHole,
    context: str,
) -> tuple[str, ...]:
    raw_value = _first_present_value(
        mapping,
        ("allowed_ops", "allowed_operators", "operators", "ops"),
    )
    if raw_value is None:
        return default_hole.allowed_ops
    if isinstance(raw_value, list) and not raw_value:
        return default_hole.allowed_ops
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return default_hole.allowed_ops
        raw_value = [
            item.strip()
            for item in re.split(r"[,;/|、，]|\bor\b|\band\b", text)
            if item.strip()
        ]
        if not raw_value:
            return default_hole.allowed_ops
    return normalize_operator_list(raw_value, context)


def _hole_candidate_payload(candidate: HoleCandidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "skeleton": str(candidate.skeleton),
        "template_text": candidate.template_text,
        "holes": [
            {
                "hole_id": hole.hole_id,
                "target": hole.target,
                "allowed_ops": hole.allowed_ops,
                "reason": hole.reason,
                "role": hole.role,
                "parent_hole_id": hole.parent_hole_id,
                "source_hint": hole.source_hint,
            }
            for hole in candidate.holes
        ],
        "explanation": candidate.explanation,
    }


def _assignments_from_llm(value: object, index: int) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()

    assignments: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for hole_id, operator in value.items():
            if not isinstance(hole_id, str):
                raise ValueError(f"completions[{index}].assignments 的 hole_id 必须是字符串。")
            assignments.append((hole_id, normalize_operator(operator, f"completions[{index}].assignments.{hole_id}")))
        return tuple(assignments)

    if isinstance(value, list):
        for assignment_index, item in enumerate(value, start=1):
            if isinstance(item, dict):
                hole_id = require_string(item, "hole_id", f"completions[{index}].assignments[{assignment_index}]")
                raw_operator = item.get("operator", item.get("op"))
                assignments.append((hole_id, normalize_operator(raw_operator, f"completions[{index}].assignments[{assignment_index}].operator")))
            elif isinstance(item, list) and len(item) == 2 and isinstance(item[0], str):
                assignments.append((item[0], normalize_operator(item[1], f"completions[{index}].assignments[{assignment_index}]")))
            else:
                raise ValueError(f"completions[{index}].assignments[{assignment_index}] 格式不支持。")
        return tuple(assignments)

    raise ValueError(f"completions[{index}].assignments 必须是 object 或 array。")


def _operator_derivation_from_llm(
    value: object,
    *,
    semantic: SemanticStructure,
    hole_candidate: HoleCandidate | None,
    traces: list[LabeledTrace],
    assignments: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    if isinstance(value, list):
        derivations = tuple(str(item).strip() for item in value if str(item).strip())
        if derivations:
            return derivations
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if hole_candidate is None:
        assignment_text = ", ".join(
            f"{hole_id}={operator}" for hole_id, operator in assignments
        ) or "no holes"
        return (
            "LLM 未返回逐孔推导；保留 assignments="
            f"{assignment_text}，并由后续轨迹模拟统一核查。",
        )
    return _operator_derivations_for_hole_candidate(
        semantic=semantic,
        hole_candidate=hole_candidate,
        traces=traces,
        assignments=assignments,
    )


def _sanitize_rag_context(contexts: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(_sanitize_context_text(context) for context in contexts)


def _sanitize_context_text(text: str) -> str:
    old_policy = (
        "trace_use_policy: "
        + "boundary"
        + " traces are primary; "
        + "positive and negative traces are auxiliary"
    )
    sanitized = text.replace(
        old_policy,
        "trace_use_policy: positive, negative, and boundary traces are joint references",
    )
    sanitized = sanitized.replace("boundary" + "_evidence:", "trace_derivation:")
    sanitized = sanitized.replace(
        "positive" + "_negative" + "_evidence:",
        "trace_derivation:",
    )
    return sanitized






