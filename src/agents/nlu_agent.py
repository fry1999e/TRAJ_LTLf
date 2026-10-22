"""
自然语言理解 Agent。

职责：
    1. 使用 RAG 检索相关 LTLf 模式和语义知识。
    2. 分析 AP、完整布尔/时序嵌套、作用域、边界和歧义并生成保真 IR。
    3. 在人工评审不通过时，结合评审意见重新生成语义结构。
"""

from __future__ import annotations
from execution_observer import observed_stage

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re

from agents.base_agent import BaseAgent, JSONLLMClient
from agents.llm_parsing import AP_PATTERN, coerce_float, require_ap_name, require_mapping, optional_string
from guided_learning import GuidedLearningContext, guided_prompt_rule
from agents.prompt_context import (
    FULL_PROCESS_SOURCE,
    KB_LTLF_SEMANTICS,
    KB_PATTERNS,
    build_stage_prompt_context,
    structured_system_prompt,
)
from contract_versions import (
    FAITHFUL_IR_SCHEMA_VERSION,
    NLU_SEMANTIC_PROMPT_VERSION,
)
from pattern_registry import normalize_pattern
from plain_reading import (
    PlainReadingValidationError,
    PlainReadingSemanticResolutionRequired,
    assess_plain_reading_ir_readiness,
    assert_plain_reading_ready_for_candidate_ir,
    assert_plain_reading_ready_for_finalization,
)
from faithful_ir import (
    FUTURE_PROHIBITION_TOPOLOGY_CODE,
    FUTURE_PROHIBITION_TOPOLOGY_PATH,
    FaithfulIRFinding,
    FaithfulIRValidationError,
    PRECEDENCE_PARAMETER_VALUES,
    SUPPORTED_NODE_TYPES,
    UNTIL_PARAMETER_VALUES,
    attach_faithful_ir,
    faithful_ir_from_mapping,
    future_prohibition_topology_error_from_mapping,
    reject_formula_fields,
    rule_semantic_to_ir,
)
from relation_parameter_contracts import (
    PERIODICITY_PARAMETER_VALUES,
    periodicity_mode_constraints_payload,
)
from semantic_node_contracts import (
    LEGACY_NODE_TYPES,
    children_patch_json_schema,
    node_contract_fingerprint,
    node_contract_prompt_view,
    node_parameter_constraints_payload,
    node_proposal_json_schema,
)
from semantic_axes import semantic_axes_payload
from semantic_graph_repair import (
    apply_semantic_graph_transaction,
    normalize_unary_boolean_graph,
    semantic_graph_fingerprint,
    semantic_graph_transaction_from_mapping,
    semantic_graph_transaction_response_schema,
)
from semantic_topology import (
    faithful_ir_proposal_mapping,
    future_prohibition_lexical_participants,
    resolve_future_prohibition_binding,
    subtree_node_ids,
)
from rag.example_memory import (
    FAITHFUL_IR_SOURCE,
    GUIDED_FULL_PROCESS_SOURCE,
    GUIDED_SEMANTIC_SOURCE,
    PATTERN_SOURCE,
    SEMANTIC_SOURCE,
)
from schemas import PlainReadingReview, SemanticStructure, SurfaceSegmentation
from temporal_domain import apply_temporal_domain_assessment
from temporal_language import temporal_language_prompt_view
from temporal_policy_profiles import temporal_policy_profiles_payload
from semantic_policy_resolution import (
    candidate_seed_policy_decision,
    legal_policy_options_for_request,
    resolve_semantic_policy_request,
    semantic_resolution_request_payload,
)
from semantic_policy_branching import (
    apply_singleton_policy_option,
    materialize_policy_candidate_set,
)
from semantic_policy_binding import (
    SemanticPolicyBindingError,
    SemanticPolicyTargetBinding,
    SemanticPolicyTargetReference,
    SemanticPolicyTargetRequest,
    build_semantic_policy_target_binding,
    build_semantic_policy_target_request,
    semantic_policy_target_request_payload,
)
from semantic_source import (
    EffectiveSemanticSource,
    NATURAL_LANGUAGE,
    bind_semantic_to_effective_source,
    natural_language_source,
)
from generation_schemas import patch_authorization, patch_operations_schema
from evidence_grounding import (
    canonicalize_faithful_ir_proposal,
    locate_evidence_quote,
)
from ap_binding_context import (
    APBindingContext,
    text_only_ap_binding_context,
)
from validated_generation import (
    CandidateSubmission,
    PatchValidationError,
    PatchSubmission,
    RepairClass,
    RepairPolicy,
    Severity,
    ValidatedGenerationRunner,
    ValidationIssue,
    patch_path_states,
)


class NLUOutputShapeError(ValueError):
    """Raised when LLM-backed NLU returns a schema-v2/legacy output shape."""

    def __init__(self, message: str, *, failure_kind: str):
        super().__init__(message)
        self.failure_kind = failure_kind


class NLURepairExhaustedError(FaithfulIRValidationError):
    """Raised with structured diagnostics when repaired IR is still invalid."""

    suggested_rerun_stage = "semantic"

    def __init__(
        self,
        validation_error: str,
        *,
        first_failure_kind: str,
        first_failure: str,
        attempt: int = 2,
        reason_code: str = "",
        repair_error: str = "",
    ):
        llm_repair_calls = max(0, attempt - 1)
        if reason_code == "HUMAN_REVIEW_REQUIRED":
            message = (
                "语义生成已安全暂停：这句话还包含一个会改变判定结果的语义选择，"
                "系统没有擅自猜测。请按提示确认含义后重试。"
                " NLU initial proposal requires human semantic review before "
                f"schema-v{FAITHFUL_IR_SCHEMA_VERSION} materialization: "
                f"{validation_error}"
            )
        elif reason_code == "REPAIR_REJECTED":
            message = (
                f"NLU LLM generation call {attempt} (LLM repair call "
                f"{llm_repair_calls}) was "
                f"rejected before schema-v{FAITHFUL_IR_SCHEMA_VERSION} "
                f"validation: {repair_error or validation_error}"
            )
        elif reason_code == "REGRESSION_ROLLED_BACK":
            message = (
                f"NLU LLM generation call {attempt} produced a changed repair "
                "candidate with no validation progress; it was rolled back. "
                f"Current schema-v{FAITHFUL_IR_SCHEMA_VERSION} validation "
                f"failure: {validation_error}"
            )
        else:
            message = (
                f"NLU LLM generation call {attempt} (after {llm_repair_calls} "
                f"LLM repair call(s)) failed schema-v{FAITHFUL_IR_SCHEMA_VERSION} validation: "
                f"{validation_error}"
            )
        classified_code = (
            "IR_REPAIR_BUDGET_EXHAUSTED"
            if reason_code in {"REPAIR_BUDGET_EXHAUSTED", "CONSTRAINED_REPAIR_BUDGET_EXHAUSTED"}
            and llm_repair_calls > 0
            else "IR_SEMANTIC_AXIS_REQUIRED"
            if reason_code in {"HUMAN_REVIEW_REQUIRED", "IR_SEMANTIC_RESOLUTION_REQUIRED"}
            else "IR_GENERATION_SCHEMA_INVALID"
        )
        super().__init__(message, code=classified_code, repairable=False)
        self.diagnostics = {
            "failure_kind": "faithful_ir_validation_error",
            "attempt": attempt,
            "llm_generation_calls": attempt,
            "llm_repair_calls": llm_repair_calls,
            "first_failure_kind": first_failure_kind,
            "first_failure": first_failure,
            "validation_error": validation_error,
            "reason_code": reason_code,
            "repair_error": repair_error,
            "suggested_rerun_stage": self.suggested_rerun_stage,
        }


class NLUAgent(BaseAgent):
    """自然语言理解 Agent。"""

    def __init__(self, retriever, llm_client: JSONLLMClient | None = None):
        super().__init__(retriever=retriever, name="NLUAgent", llm_client=llm_client)
        self.last_generation_diagnostic: dict[str, object] = {}

    @observed_stage("ir_generation")
    def generate(
        self,
        requirement: str,
        feedback_history: list[str] | None = None,
        guided_context: GuidedLearningContext | None = None,
        surface_segmentation: SurfaceSegmentation | None = None,
        plain_reading_review: PlainReadingReview | None = None,
        ap_binding_context: APBindingContext | None = None,
        effective_semantic_source: EffectiveSemanticSource | None = None,
    ) -> SemanticStructure:
        """
        生成语义结构。

        feedback_history 保存人工评审意见。比如人工说“response 应该是 ack”，
        Agent 下一轮会优先尝试从意见中提取新的 response。
        """

        # Parent validation still precedes every call.  The finalization gate
        # is intentionally deferred until the source graph has had a chance to
        # provide a complete typed mapping for unresolved ordinary readings.
        if plain_reading_review is not None:
            if surface_segmentation is None:
                raise ValueError(
                    "plain_reading_review requires its parent surface_segmentation"
                )

        feedback_history = feedback_history or []
        ap_binding_context = ap_binding_context or text_only_ap_binding_context(
            requirement
        )
        ap_binding_context.validate_source(requirement)
        effective_semantic_source = (
            effective_semantic_source
            or natural_language_source(
                requirement_id=ap_binding_context.requirement_id,
                source_text=requirement,
            )
        )
        if (
            effective_semantic_source.source_kind != NATURAL_LANGUAGE
            or effective_semantic_source.requirement_id
            != ap_binding_context.requirement_id
            or effective_semantic_source.source_text != requirement
        ):
            raise ValueError(
                "NLU natural-language path received a mismatched effective source"
            )
        if self.llm_enabled:
            from ap_preparation import ensure_ap_candidates
            ap_binding_context = ensure_ap_candidates(
                ap_binding_context, self.llm_client, segmentation=surface_segmentation,
            ).context
        # The same reviewed parents and AP authority feed both frontends.
        # Constructive output still goes through the existing review/freeze gates.
        if self.llm_enabled:
            from ir_proposal_router import IRProposalRouter
            from faithful_ir import semantic_summary_from_ir
            built = IRProposalRouter().propose(
                requirement, ap_binding_context,
                policies=getattr(self.llm_client, "dataset_semantics_contract", None),
                surface_segmentation=surface_segmentation,
                plain_reading_review=plain_reading_review,
                proposal_client=self.llm_client,
                feedback_history=feedback_history,
            )
            if built.candidates and len(built.candidates) > 1:
                from constructive_ir_builder import (
                    bind_plain_review_to_source_axes,
                    build_result_candidate_semantic,
                )
                gate_mapping, reverse_mapping = bind_plain_review_to_source_axes(
                    plain_reading_review, built, mapping_client=self.llm_client,
                )
                if plain_reading_review is not None:
                    try:
                        assert_plain_reading_ready_for_candidate_ir(
                            plain_reading_review,
                            surface_segmentation,
                            mapped_axes=gate_mapping,
                            max_candidates=4,
                        )
                    except PlainReadingValidationError:
                        readiness = assess_plain_reading_ir_readiness(plain_reading_review, surface_segmentation)
                        if readiness.disposition != "semantic_resolution_required":
                            raise
                        raise PlainReadingSemanticResolutionRequired(
                            plain_reading_review, readiness, segmentation=surface_segmentation,
                        ) from None
                semantic = build_result_candidate_semantic(
                    built,
                    rag_context=(),
                    plain_axis_bindings=reverse_mapping,
                )
                semantic = replace(
                    semantic,
                    surface_segmentation=surface_segmentation,
                    plain_reading_review=plain_reading_review,
                )
                semantic = apply_temporal_domain_assessment(semantic, requirement)
                return bind_semantic_to_effective_source(
                    semantic,
                    effective_semantic_source,
                )
            if built.ir is not None:
                if plain_reading_review is not None:
                    assert_plain_reading_ready_for_finalization(
                        plain_reading_review,
                        surface_segmentation,
                    )
                semantic = semantic_summary_from_ir(built.ir, rag_context=())
                semantic = replace(semantic, surface_segmentation=surface_segmentation,
                                   plain_reading_review=plain_reading_review)
                semantic = apply_temporal_domain_assessment(semantic, requirement)
                return bind_semantic_to_effective_source(
                    attach_faithful_ir(semantic, requirement, ap_binding_context=ap_binding_context),
                    effective_semantic_source,
                )
            # Once a graph route was attempted, do not discard its reason/debt
            # and ask a model to hand-write a whole IR.  Unsupported semantics
            # remain explicit and auditable.
            if built.route in {"G", "U", "unsupported"}:
                if built.semantic_debts:
                    from source_graph_resolution import SourceGraphResolutionRequired
                    raise SourceGraphResolutionRequired(
                        requirement, ap_binding_context, built, segmentation=surface_segmentation,
                        plain_reading_review=plain_reading_review,
                    )
                raise FaithfulIRValidationError(
                    built.reason_code or "source graph did not yield executable IR",
                    code=(
                        "UNSUPPORTED_SEMANTICS"
                        if built.route == "unsupported"
                        else "SOURCE_GRAPH_CANDIDATE_REQUIRED"
                    ),
                    path="/source_intent_graph",
                    repairable=False,
                )
        if plain_reading_review is not None:
            assert_plain_reading_ready_for_finalization(
                plain_reading_review,
                surface_segmentation,
            )
        prompt_context = build_stage_prompt_context(
            self.retriever,
            stage="semantic_extraction",
            query_parts=(
                "自然语言理解",
                requirement,
                "\n".join(feedback_history),
                _guided_query_text(guided_context, "semantic"),
            ),
            domain_sources=(KB_PATTERNS, KB_LTLF_SEMANTICS),
            example_sources=(
                FAITHFUL_IR_SOURCE,
            ),
            agent_function_contract=_semantic_extraction_contract(),
        )
        rag_context = prompt_context.as_rag_context()

        if self.llm_enabled:
            semantic = self._generate_with_llm(
                requirement,
                feedback_history,
                prompt_context,
                guided_context=guided_context,
                surface_segmentation=surface_segmentation,
                plain_reading_review=plain_reading_review,
                ap_binding_context=ap_binding_context,
            )
            semantic = apply_temporal_domain_assessment(semantic, requirement)
            semantic = replace(
                semantic,
                surface_segmentation=surface_segmentation,
                plain_reading_review=plain_reading_review,
            )
            return bind_semantic_to_effective_source(
                attach_faithful_ir(
                    semantic,
                    requirement,
                    ap_binding_context=ap_binding_context,
                ),
                effective_semantic_source,
            )

        if (
            guided_context is not None
            and guided_context.expected_semantic is not None
        ):
            semantic = replace(
                guided_context.expected_semantic,
                formula_text="",
                formula_source="none",
                formula_visibility=(),
            )
            semantic = apply_temporal_domain_assessment(semantic, requirement)
            semantic = replace(
                semantic,
                surface_segmentation=surface_segmentation,
                plain_reading_review=plain_reading_review,
            )
            if semantic.faithful_ir is None:
                semantic = replace(
                    semantic,
                    faithful_ir=rule_semantic_to_ir(
                        semantic,
                        requirement,
                        ap_binding_context=ap_binding_context,
                    ),
                )
            return bind_semantic_to_effective_source(
                attach_faithful_ir(
                    semantic,
                    requirement,
                    ap_binding_context=ap_binding_context,
                ),
                effective_semantic_source,
            )

        semantic = self._infer_response_structure(requirement, rag_context)
        semantic = self._apply_feedback(semantic, feedback_history)
        semantic = apply_temporal_domain_assessment(semantic, requirement)
        semantic = replace(
            semantic,
            surface_segmentation=surface_segmentation,
            plain_reading_review=plain_reading_review,
        )
        semantic = replace(
            semantic,
            faithful_ir=rule_semantic_to_ir(
                semantic,
                requirement,
                ap_binding_context=ap_binding_context,
            ),
        )
        return bind_semantic_to_effective_source(
            attach_faithful_ir(
                semantic,
                requirement,
                ap_binding_context=ap_binding_context,
            ),
            effective_semantic_source,
        )

    def _generate_with_llm(
        self,
        requirement: str,
        feedback_history: list[str],
        prompt_context,
        guided_context: GuidedLearningContext | None = None,
        surface_segmentation: SurfaceSegmentation | None = None,
        plain_reading_review: PlainReadingReview | None = None,
        ap_binding_context: APBindingContext | None = None,
    ) -> SemanticStructure:
        if ap_binding_context is None:
            raise TypeError("formal NLU generation requires ap_binding_context")
        extra_rules = (
            (
                guided_prompt_rule("semantic_extraction"),
                "必须在 explanation 中说明哪些 expected semantic 字段引导了本次选择。",
            )
            if guided_context is not None
            else ()
        )
        system_prompt = structured_system_prompt(
            agent_name="自然语言理解 Agent",
            stage_name="semantic_extraction",
            extra_rules=(
                "只抽取当前需求中有文本证据支持的 AP 和 FaithfulSemanticIR。",
                "返回 JSON 顶层必须包含 faithful_ir，或 semantic.faithful_ir；"
                "只有 trigger/response/pattern 的旧输出会被拒绝。",
                "当前程序只接受英文 AP 名称：[A-Za-z_][A-Za-z0-9_]*。",
                "atomic_propositions 必须声明所有真实 AP；每个 ap_condition 节点的 "
                "parameters.ap_name 必须引用其中一个 atomic_propositions.name。",
                "同一 AP 即使在原文中出现多次，也只能在 atomic_propositions 中声明一次；"
                "每个文本出现位置应由不同 ap_condition 节点引用同一个 AP name，并各自"
                "携带证据选择器。不得按文本出现次数复制同名 AP 声明。",
                "faithful_ir.nodes[].node_type 必须严格取自 output_schema.supported_node_types；"
                "精确下一步使用 next_step，严格未来任意位置使用 eventually_strict，"
                "有界未来使用 bounded_window；不要写 next、strict_future 或 current_or_future。",
                "LLM 只确定公式无关的 IR 图结构与文本支持的时序语义范围：默认 "
                "temporal_resolution=fixed；只有同一一元结构存在可由有限轨迹区分的"
                "作用域/时序读法时，才使用 trace_resolvable 并填写 "
                "allowed_semantic_readings。该字段只能写语义名称，禁止写 G/F/X/U 等算子。",
                "每个 AP/节点只提出 exact evidence_quote 与 supporting_segment_ids；"
                "source_span、evidence_text、稳定 ID 和版本由后端派生。quote 在 "
                "supporting_segment_ids 限定的来源窗口内重复时需给 occurrence_index；"
                "该索引从 0 开始，并相对于同一窗口计数。",
                "surface_segmentation 只表示逐字无损的阅读边界，不得把 segment "
                "边界自动当成逻辑括号；每个 AP 和节点必须填写与 source_span "
                "相交的 supporting_segment_ids。",
                "plain_reading_review 中 reviewer_selected 是主分支生成约束，"
                "但仍是 non_authoritative；必须采用选中的 option，并沿用该 ambiguity_id。"
                "该已审核轴的 question、options 和 decision 由后端从 active review 恢复，"
                "不要改写它们；"
                "不得把已选轴重新输出为 resolved=false，也不得生成与选中读法直接冲突的主分支。",
                "plain_reading_review 中 ambiguous_unresolved 必须继续写入 "
                "faithful_ir.ambiguities，不得静默标记 resolved。",
                "完整布尔树、时序父子关系、作用域、边界和参数必须保存在 faithful_ir.nodes，"
                "不得压缩成单一 trigger/response 对。",
                "semantic_role 是供评审阅读的说明文本，不是机器枚举；trigger、response、"
                "否定对象和作用域必须由 children 拓扑、parameters.ap_name 与原文证据共同确定。",
                "不要主动输出 trigger、response 或 pattern 作为主语义字段；"
                "系统会从 IR 派生兼容摘要。",
                "不要用 trigger_001、response_001 这类未绑定占位符替代真实 AP。",
                "显式 until/till 只足以建立 until 候选；必须按 children=(left_condition, right_witness) "
                "保留方向，并用 parameters/ambiguities 表达 strong-vs-weak、immediate-vs-strict、"
                "left-at-witness 等未决语义。不得仅凭 until 默认强 U。",
                "until the end of the trace 应建模为 persist_to_end，不得发明 end AP。",
                "严格 before/precedence 使用 children=(prerequisite, constrained_event)，"
                "并记录 ordering、witness_policy、same_step_policy、reuse_policy、scope_origin；"
                "所有值必须严格取自 output_schema.relation_parameter_values.precedence，"
                "不要创造同义枚举，也不要在语义阶段改写成公式。当前执行器仅支持 "
                "reuse_policy=reusable 与 scope_origin=trace_start；如果原文确实要求 "
                "one_to_one 或 local_activation，必须请求语义裁决，不得悄悄改成可执行策略。",
                "until 的 parameters 必须严格取自 "
                "output_schema.relation_parameter_values.until；无法从原文确定时应声明"
                "关联该节点的未解决 ambiguity，不得猜测或创造同义枚举。",
                "Once/after A happened, B won't/must not happen again 不是 precedence；"
                "应保留全局触发、strict-future 起点、后缀持续和 B 否定的嵌套结构。",
                "periodicity 必须采用 mode-specific 完整参数。无显式绝对锚点时可由读法确定 "
                "origin=first_occurrence，但不得默认 terminal_policy；只有明确索引/"
                "分组/时钟锚点才使用 fixed_phase，并给出 phase 与 off_schedule_policy=forbidden。"
                "事件相对 mode 禁止 phase。严格遵守 "
                "output_schema.relation_parameter_values.periodicity 与 "
                "output_schema.parameter_constraints.periodicity；计数起点、固定相位/"
                "exact recurrence、可停止 exact spacing、minimum spacing 或是否至少出现一次"
                "无法由原文/人工读法确定时，返回 semantic resolution request，不得生成缺参 IR。不得使用 "
                "unknown、unspecified 或 unconstrained 等哨兵值；ambiguity 的 description "
                "必须逐项写出所有被省略的参数原名。",
                "在物化 IR 前先检查所有有限轨迹政策轴：有界窗口、固定持续、until 和周期结构"
                "是否在轨迹末端仍需要未来位置。普通读法已唯一但末端政策未明示时，这不是新的"
                "普通语言 ambiguity；必须返回 typed semantic_resolution_request，让后端展开全部"
                "合法 CandidateIR 分支，不得选择默认 profile。"
                "next_step 必须显式给出 steps；fixed_duration 必须给出 duration、origin、short_trace_policy、"
                "terminal_obligation_policy、minimum_horizon；bounded_window 必须给出 "
                "lower_bound、upper_bound 以及同样的有限轨迹策略。recurrence 必须显式给出 "
                "origin、gap_policy、short_trace_policy、terminal_obligation_policy 和 "
                "minimum_horizon。文本决定的 bounds/duration 不足时保留普通语义 ambiguity；"
                "仅短轨迹/末端政策不足时返回 project_policy 类型的 semantic resolution request，"
                "不得使用默认参数。",
                "凡是 whenever/every time/always each occurrence 这类重复触发读法，必须由 "
                "global_constraint 包住 implication；裸 implication 只约束有限轨迹初始位置，"
                "不得用 scope 文本字段代替拓扑中的全局节点。scope 只能使用 "
                "current_position、inherited、all_positions 三个规范元数据值；"
                "all_positions 只允许出现在 global_constraint 上。",
                "区分精确时点与存在窗口：at the next step 是 next_step(steps=1)；"
                "at some strict-future step 才是 eventually_strict；current or next 是 "
                "bounded_window(lower_bound=0, upper_bound=1)，并采用能在末状态保留当前时刻"
                "见证的 available_prefix/require_witness 策略。不得把有限窗口扩大为任意未来。",
                "多个相对时间短语必须统一换算到当前规则激活点：若 b 在激活后 1 步，c 又在 b 后 1 步，"
                "则 c 的总 offset 是 2；children 拓扑和 next_step.steps 必须表达累计后的精确位置，"
                "不得把两个相对偏移错误地并列成同一 offset。",
                "precedence/before 的 witness_policy 必须忠实表达条件性：若原文只限制 constrained_event"
                "发生时必须已有 prerequisite，则 constrained_event 从未发生时应使用 "
                "optional_if_no_constrained；只有原文另行要求事件必须发生时才使用 required。",
                "cardinality 的 mode/count 作用于 child 所表示的完整谓词。禁止为了套用阈值而移动否定、"
                "对被计数谓词取反，或把 at_most 与 at_least 互换；谓词极性必须由 negation children 保留。",
                "pattern 必须描述真实业务模式。",
                "禁止输出 formula、formula_text、candidate_formula 或 final_formula；"
                "FaithfulSemanticIR 是领域语义图，不是公式 AST。",
                *extra_rules,
            ),
        )
        rag_context = prompt_context.as_rag_context()
        payload = {
            "task": "semantic_extraction",
            "requirement": requirement,
            "feedback_history": feedback_history,
            "guided_learning": (
                guided_context.prompt_payload("semantic_extraction")
                if guided_context is not None
                else None
            ),
            "surface_segmentation": (
                _artifact_payload(surface_segmentation)
                if surface_segmentation is not None
                else None
            ),
            "plain_reading_review": (
                _artifact_payload(plain_reading_review)
                if plain_reading_review is not None
                else None
            ),
            **prompt_context.payload_fields(),
            "rag_context": rag_context,
            "ap_binding_context": ap_binding_context.prompt_payload(),
            "ap_grounding_contract": (
                ap_binding_context.grounding_contract.prompt_payload()
                if ap_binding_context.grounding_contract is not None
                else None
            ),
            "output_schema": {
                "required_top_level_one_of": [
                    "faithful_ir",
                    "semantic.faithful_ir",
                    "semantic_resolution_request",
                ],
                "supported_node_types": sorted(SUPPORTED_NODE_TYPES - LEGACY_NODE_TYPES),
                "node_type_aliases": {"next": "next_step"},
                "temporal_semantic_range": {
                    "resolution_modes": ["fixed", "trace_resolvable"],
                    "trace_resolvable_readings": [
                        "identity",
                        "immediate",
                        "global_constraint",
                        "persist_to_end",
                        "eventually",
                        "eventually_strict",
                        "next_step",
                    ],
                    "rule": (
                        "Use semantic names only. The backend expands the approved "
                        "range into operator candidates and trajectories eliminate them."
                    ),
                },
                "relation_parameter_values": _relation_parameter_schema(),
                "parameter_constraints": _nlu_parameter_constraints(),
                "node_contract": node_contract_prompt_view(),
                "node_contract_fingerprint": node_contract_fingerprint(),
                "node_proposal_schema": node_proposal_json_schema(),
                "children_repair_schema": children_patch_json_schema(),
                "temporal_policy_profiles": temporal_policy_profiles_payload(),
                "temporal_language_contract": temporal_language_prompt_view(),
                "semantic_axis_registry": semantic_axes_payload(),
                "format_rules": [
                    "Top-level JSON object should be {'faithful_ir': {...}, 'explanation': '...'} or {'semantic': {'faithful_ir': {...}}}.",
                    "Do not return legacy trigger/response-only JSON.",
                    "Keep explanation short; spend tokens on complete faithful_ir nodes.",
                    "Return exact evidence_quote text; backend derives and validates source_span/evidence_text.",
                    "Use node_type='next_step' for next-step semantics; do not use node_type='next'.",
                    "next_step.parameters.steps is a required JSON integer >= 1; steps=N means exactly N nested next positions.",
                    "Until children are ordered as [left_condition, right_witness]; precedence children are ordered as [prerequisite, constrained_event].",
                    "Until, precedence/before, and periodicity parameter values must exactly match output_schema.relation_parameter_values; do not invent aliases.",
                    "An executable precedence/before node currently requires reuse_policy='reusable' and scope_origin='trace_start'.",
                    "periodicity.period is an integer >=1; phase is fixed_phase-only and must satisfy 0 <= phase < period.",
                    "Never emit an unresolved or incomplete FaithfulSemanticIR; request semantic resolution when a business decision lacks source/human/policy basis.",
                    "Declare each AP name exactly once. Repeated textual mentions use multiple ap_condition nodes that reference the same declaration.",
                    "AP identity is backend-owned. Use only canonical names exposed by ap_grounding_contract, cite exact evidence, and never merge, split, rename, or invent AP identities.",
                    "Every node must have a unique non-empty temporary node_id. root_node_id and every children item must reference one of those temporary IDs; do not reuse IDs or leave disconnected nodes.",
                    "Do not default unresolved Until witness policy merely because the word until appears; declare a SemanticAmbiguity.",
                    "For a fixed temporal reading, omit allowed_semantic_readings or provide one semantic reading and set temporal_resolution='fixed'.",
                    "For a genuinely text-supported unary timing/scope range, set temporal_resolution='trace_resolvable' and provide at least two behaviorally distinct allowed_semantic_readings; never put LTLf operator tokens in this field.",
                    "faithful_ir.ambiguities must always be a JSON array of objects; use [] when there is no ambiguity.",
                ],
                "faithful_ir": {
                    "schema_version": FAITHFUL_IR_SCHEMA_VERSION,
                    "requirement_id": "stable requirement id",
                    "source_text": "exact requirement text",
                    "atomic_propositions": [{
                        "ap_id": "backend-derived",
                        "name": "AP name",
                        "aliases": [],
                        "evidence_quote": "exact source substring",
                        "occurrence_index": "zero-based; only when the quote repeats within the supporting-segment window",
                        "supporting_segment_ids": ["seg-1"],
                    }],
                    "root_node_id": "ir-n1",
                    "nodes": [{
                        "node_id": "unique temporary ID for children/root references; backend replaces it with a stable ID",
                        "node_type": "supported semantic node type",
                        "children": [],
                        "semantic_role": "role in requirement",
                        "scope": "scope interpretation",
                        "temporal_relation": "domain relation, not operator token",
                        "boundary_policy": "finite-trace interpretation",
                        "allowed_semantic_readings": [
                            "formula-free semantic reading; empty means the canonical node_type"
                        ],
                        "temporal_resolution": "fixed or trace_resolvable",
                        "parameters": {},
                        "evidence_quote": "exact source substring",
                        "occurrence_index": "zero-based; only when the quote repeats within the supporting-segment window",
                        "confidence": 0.0,
                        "supporting_segment_ids": ["seg-1"],
                    }],
                    "ambiguities": [],
                    "extraction_confidence": 0.0,
                    "provenance": {"extractor": "NLUAgent"},
                },
                "ambiguity_item_schema": {
                    "ambiguity_id": (
                        "copy the matching plain_reading_review ambiguity_id; "
                        "otherwise backend-derived"
                    ),
                    "description": (
                        "for a reviewed ambiguity_id this field is backend-owned; "
                        "otherwise describe only a genuinely new unresolved axis"
                    ),
                    "affected_node_ids": ["ir-n1"],
                    "alternatives": [
                        "backend-rehydrated for a reviewed ambiguity_id",
                        "otherwise a second source-supported interpretation",
                    ],
                    "risk": "medium",
                    "resolved": True,
                },
                "semantic_resolution_request": {
                    "node_id": (
                        "diagnostic temporary node hint from this first response; "
                        "never reused as canonical target identity"
                    ),
                    "node_type": "exact output_schema supported node type",
                    "axis_kind": "exact semantic_axis_registry axis id",
                    "evidence_quote": "exact source substring",
                    "unresolved_parameters": ["exact contract parameter name"],
                    "alternatives": [
                        "advisory plain-language interpretation only",
                        "backend will replace these with validated legal options",
                    ],
                    "question": "one reviewable semantic decision question",
                },
                "semantic_policy_target": {
                    "request_id": (
                        "exact backend-provided request_id; required only during "
                        "the second policy materialization response"
                    ),
                    "node_id": (
                        "temporary node_id from faithful_ir.nodes in that same "
                        "second response"
                    ),
                },
                "explanation": "short review-friendly explanation",
            },
        }
        generation_attempt = 0
        state: dict[str, object] = {
            "semantic": None,
            "first_failure_kind": "",
            "first_failure": "",
            "last_failure": "",
            "last_exception": None,
            "retry_instruction": "",
            "canonical_issues": (),
            "future_prohibition_binding": None,
            "semantic_policy_resolution": None,
            "semantic_resolution_request": None,
            "semantic_resolution_options": (),
            "semantic_policy_candidate_context": None,
            "semantic_policy_target_request": None,
            "semantic_policy_target_binding": None,
            "semantic_policy_target_issue": None,
            "last_raw_patch": None,
        }

        def canonical_policy_request(
            raw_request: Mapping[str, object],
            legal_options: tuple[dict[str, object], ...],
        ):
            quote = str(raw_request.get("evidence_quote", "")).strip()
            location = locate_evidence_quote(
                requirement,
                (
                    surface_segmentation.segments
                    if surface_segmentation is not None
                    else ()
                ),
                quote,
                source_segment_ids=tuple(
                    str(value)
                    for value in raw_request.get("supporting_segment_ids", ())
                ),
                occurrence_index=(
                    int(raw_request["occurrence_index"])
                    if isinstance(raw_request.get("occurrence_index"), int)
                    else None
                ),
                stage="semantic",
                path="/semantic_resolution_request/evidence_quote",
            )
            if isinstance(location, ValidationIssue):
                state["semantic_policy_target_issue"] = location
                return None
            try:
                request = build_semantic_policy_target_request(
                    raw_request,
                    requirement_id=ap_binding_context.requirement_id,
                    source_hash=hashlib.sha256(
                        requirement.encode("utf-8")
                    ).hexdigest(),
                    segmentation_id=(
                        surface_segmentation.segmentation_id
                        if surface_segmentation is not None
                        else ""
                    ),
                    reading_review_id=(
                        plain_reading_review.reading_review_id
                        if plain_reading_review is not None
                        else ""
                    ),
                    legal_options=legal_options,
                    evidence_location=location,
                )
            except SemanticPolicyBindingError as exc:
                state["semantic_policy_target_issue"] = ValidationIssue(
                    code="SEMANTIC_POLICY_TARGET_REQUEST_INVALID",
                    stage="semantic",
                    path="/semantic_resolution_request",
                    observed=dict(raw_request),
                    expected={"backend_canonical_policy_target_request": True},
                    repair_class=RepairClass.HUMAN_REQUIRED,
                    allowed_paths=(),
                    message=str(exc),
                )
                return None
            state["semantic_policy_target_request"] = request
            return request

        def propose() -> dict:
            nonlocal generation_attempt
            generation_attempt += 1
            raw = self.complete_json(
                system_prompt,
                payload,
                max_tokens=_nlu_max_tokens(),
            )
            resolution_request = semantic_resolution_request_payload(raw)
            legal_options = legal_policy_options_for_request(raw)
            if resolution_request is not None and len(legal_options) >= 2:
                target_request = canonical_policy_request(
                    resolution_request,
                    legal_options,
                )
                if target_request is None:
                    return raw
                seed_decision = candidate_seed_policy_decision(
                    raw,
                    legal_options[0],
                )
                seed_payload = seed_decision.to_dict()
                state["semantic_resolution_request"] = resolution_request
                state["semantic_resolution_options"] = legal_options
                state["semantic_policy_candidate_context"] = {
                    "request": deepcopy(resolution_request),
                    "options": deepcopy(list(legal_options)),
                    "seed_profile_id": seed_decision.profile_id,
                }
                materialization_payload = deepcopy(payload)
                materialization_payload.update({
                    "task": "semantic_ir_materialize_policy_candidate_seed",
                    "previous_semantic_resolution_request": raw,
                    "semantic_policy_resolution": seed_payload,
                    "semantic_policy_target_request": (
                        semantic_policy_target_request_payload(target_request)
                    ),
                    "legal_policy_candidates": list(legal_options),
                    "materialization_instruction": (
                        "Materialize the shared source-derived faithful_ir once using "
                        "the listed seed parameters. The seed is non-authoritative. "
                        "Preserve AP identity, graph topology, evidence, participants, "
                        "and text-supported bounds; the backend will deterministically "
                        "expand every legal policy candidate after validation."
                        " Return semantic_policy_target with the exact request_id "
                        "and a node_id from this same response."
                    ),
                })
                generation_attempt += 1
                return self.complete_json(
                    system_prompt
                    + "\nMaterialize a non-authoritative policy-candidate seed. "
                    "Do not choose or recommend a final finite-trace policy.",
                    materialization_payload,
                    max_tokens=_nlu_max_tokens(),
                )
            policy_decision = resolve_semantic_policy_request(
                raw,
                plain_reading_review=plain_reading_review,
            )
            if policy_decision is None:
                resolution_request = semantic_resolution_request_payload(raw)
                if resolution_request is not None:
                    state["semantic_resolution_request"] = resolution_request
                    state["semantic_resolution_options"] = (
                        legal_policy_options_for_request(raw)
                    )
                return raw
            target_request = canonical_policy_request(
                resolution_request or {},
                legal_policy_options_for_request(raw),
            )
            if target_request is None:
                return raw
            decision_payload = policy_decision.to_dict()
            state["semantic_policy_resolution"] = decision_payload
            state["semantic_resolution_options"] = (
                legal_policy_options_for_request(raw)
            )
            materialization_payload = deepcopy(payload)
            materialization_payload.update({
                "task": "semantic_ir_materialize_resolved_policy",
                "previous_semantic_resolution_request": raw,
                "semantic_policy_resolution": decision_payload,
                "semantic_policy_target_request": (
                    semantic_policy_target_request_payload(target_request)
                ),
                "legal_review_options": list(legal_policy_options_for_request(raw)),
                "materialization_instruction": (
                    "Materialize a complete faithful_ir for the current schema. Preserve the "
                    "human-selected ordinary-language reading, graph topology, AP "
                    "identity, evidence, and text-supported window bounds. Apply the "
                    "listed policy parameters exactly to the requested typed node. "
                    "The policy fills only controlled_parameters; it must not change "
                    "the selected reading or invent unrelated semantics. Do not return "
                    "another semantic_resolution_request for these resolved axes."
                    " Return semantic_policy_target with the exact request_id and "
                    "a node_id from this same response."
                ),
            })
            generation_attempt += 1
            return self.complete_json(
                system_prompt
                + "\nThe backend has resolved a policy-only finite-trace axis. "
                "Follow semantic_policy_resolution exactly and return the complete IR; "
                "do not reinterpret the reviewed reading.",
                materialization_payload,
                max_tokens=_nlu_max_tokens(),
            )

        def canonicalize(raw: object):
            result = canonicalize_faithful_ir_proposal(
                raw,
                source_text=requirement,
                segments=(
                    surface_segmentation.segments
                    if surface_segmentation is not None
                    else ()
                ),
                reviewed_ambiguities=(
                    tuple(plain_reading_review.ambiguities)
                    if plain_reading_review is not None
                    else ()
                ),
                requirement_id=ap_binding_context.requirement_id,
                ap_binding_context=ap_binding_context,
                policy_target_request=(
                    state.get("semantic_policy_target_request")
                    if isinstance(
                        state.get("semantic_policy_target_request"),
                        SemanticPolicyTargetRequest,
                    )
                    else None
                ),
            )
            canonical_issues = tuple(result.issues)
            state["canonical_issues"] = canonical_issues
            if canonical_issues:
                state["semantic"] = None
                summary = _nlu_issue_summary(canonical_issues)
                state["last_failure"] = summary
                state["last_exception"] = None
                state["retry_instruction"] = _nlu_canonical_repair_instruction(
                    canonical_issues
                )
                if not state["first_failure_kind"]:
                    state["first_failure_kind"] = (
                        "faithful_ir_validation_error"
                        if any(
                            issue.code.startswith("IR_GRAPH_")
                            for issue in canonical_issues
                        )
                        else "evidence_grounding_error"
                    )
                    state["first_failure"] = summary
            return result

        def validate(raw: object) -> tuple[ValidationIssue, ...]:
            target_issue = state.get("semantic_policy_target_issue")
            if isinstance(target_issue, ValidationIssue):
                return (target_issue,)
            topology_error = future_prohibition_topology_error_from_mapping(
                raw,
                source_text=requirement,
            )
            if topology_error is not None:
                binding = resolve_future_prohibition_binding(
                    raw,
                    source_text=requirement,
                )
                deterministic = binding is not None
                state["future_prohibition_binding"] = (
                    {
                        "trigger_node_id": binding.trigger_node_id,
                        "prohibited_node_id": binding.prohibited_node_id,
                        "basis": list(binding.basis),
                        "contract_version": binding.contract_version,
                    }
                    if binding is not None
                    else None
                )
                instruction = _nlu_ir_repair_instruction(str(topology_error))
                _record_nlu_failure(
                    state,
                    "faithful_ir_validation_error",
                    topology_error,
                    instruction,
                )
                return (
                    ValidationIssue(
                        code=FUTURE_PROHIBITION_TOPOLOGY_CODE,
                        stage="semantic",
                        path=topology_error.path,
                        observed=_bounded_nlu_artifact(raw),
                        expected={
                            "topology": (
                                "global(trigger -> "
                                "next(global(negation(event))))"
                            )
                        },
                        context={
                            "validation_error": str(topology_error),
                            "evidence_independent": True,
                        },
                        repair_class=(
                            RepairClass.DETERMINISTIC
                            if deterministic
                            else RepairClass.SEMANTIC_PATCH
                        ),
                        allowed_paths=_future_prohibition_allowed_paths(raw),
                        message=str(topology_error),
                        severity=Severity.ERROR,
                    ),
                )
            # Evidence canonicalization owns its unresolved placeholders. Running
            # Schema validation against unresolved [-1, -1] evidence spans would
            # only create cascading
            # segment/span errors that obscure the actionable grounding issue.
            # Evidence-independent domain topology was already checked above.
            if state.get("canonical_issues"):
                return ()
            duplicate_ap_issues = _nlu_duplicate_ap_name_issues(raw)
            if duplicate_ap_issues:
                summary = _nlu_issue_summary(duplicate_ap_issues)
                duplicate_error = FaithfulIRValidationError(
                    summary,
                    code="DUPLICATE_AP_NAME",
                    path=duplicate_ap_issues[0].path,
                )
                _record_nlu_failure(
                    state,
                    "faithful_ir_validation_error",
                    duplicate_error,
                    _nlu_duplicate_ap_repair_instruction(duplicate_ap_issues),
                )
                return duplicate_ap_issues
            try:
                semantic = _parse_llm_semantic_result(
                    raw,
                    requirement=requirement,
                    rag_context=rag_context,
                    attempt=generation_attempt,
                    surface_segmentation=surface_segmentation,
                    plain_reading_review=plain_reading_review,
                )
            except NLUOutputShapeError as exc:
                repairable = exc.failure_kind in {
                    "atomic_proposition_without_ir",
                    "missing_faithful_ir",
                    "legacy_trigger_response_only",
                }
                instruction = _nlu_retry_instruction(exc.failure_kind)
                _record_nlu_failure(state, exc.failure_kind, exc, instruction)
                return (
                    ValidationIssue(
                        code="NLU_OUTPUT_SHAPE_" + exc.failure_kind.upper(),
                        stage="semantic",
                        path="/faithful_ir",
                        observed=_bounded_nlu_artifact(raw),
                        expected={"required_top_level": ["faithful_ir"]},
                        context={"failure_kind": exc.failure_kind},
                        repair_class=(
                            RepairClass.CONSTRAINED
                            if repairable
                            else RepairClass.HUMAN_REQUIRED
                        ),
                        allowed_paths=(
                            "/faithful_ir",
                            "/semantic",
                            "/explanation",
                            "/trigger",
                            "/response",
                            "/pattern",
                            "/ap_id",
                            "/name",
                            "/aliases",
                            "/source_span",
                            "/evidence_text",
                        ),
                        message=str(exc),
                    ),
                )
            except FaithfulIRValidationError as exc:
                instruction = _nlu_ir_repair_instruction(str(exc))
                _record_nlu_failure(
                    state,
                    "faithful_ir_validation_error",
                    exc,
                    instruction,
                )
                domain_findings = tuple(exc.findings) or (
                    FaithfulIRFinding(
                        code=exc.code,
                        path=exc.path,
                        message=str(exc),
                        repairable=exc.repairable,
                    ),
                )
                return tuple(
                    ValidationIssue(
                        code=finding.code,
                        stage="semantic",
                        path=_nlu_rebase_finding_path(raw, finding.path),
                        observed=_bounded_nlu_artifact(raw),
                        expected={
                            "schema_version": FAITHFUL_IR_SCHEMA_VERSION,
                            "valid_faithful_ir": True,
                        },
                        context={
                            "validation_error": finding.message,
                            "finding_count": len(domain_findings),
                        },
                        repair_class=(
                            RepairClass.SEMANTIC_PATCH
                            if finding.repairable
                            else RepairClass.HUMAN_REQUIRED
                        ),
                        allowed_paths=_nlu_finding_allowed_paths(raw, finding),
                        message=finding.message,
                        severity=Severity.ERROR,
                    )
                    for finding in domain_findings
                )
            target_request = state.get("semantic_policy_target_request")
            if isinstance(target_request, SemanticPolicyTargetRequest):
                container, prefix = faithful_ir_proposal_mapping(raw)
                target_container = None
                if isinstance(raw, Mapping):
                    target_container = (
                        raw.get("semantic")
                        if isinstance(raw.get("semantic"), Mapping)
                        else raw
                    )
                target_mapping = (
                    target_container.get("semantic_policy_target")
                    if isinstance(target_container, Mapping)
                    else None
                )
                if not isinstance(target_mapping, Mapping):
                    return (
                        ValidationIssue(
                            code="SEMANTIC_POLICY_TARGET_REFERENCE_REQUIRED",
                            stage="semantic",
                            path=(
                                "/semantic/semantic_policy_target"
                                if prefix.startswith("/semantic")
                                else "/semantic_policy_target"
                            ),
                            observed=None,
                            expected={"request_id": target_request.request_id},
                            repair_class=RepairClass.CONSTRAINED,
                            allowed_paths=(
                                "/semantic/semantic_policy_target"
                                if prefix.startswith("/semantic")
                                else "/semantic_policy_target",
                            ),
                            message="canonical policy target reference is missing",
                        ),
                    )
                try:
                    binding = build_semantic_policy_target_binding(
                        target_request,
                        SemanticPolicyTargetReference(
                            request_id=str(target_mapping.get("request_id", "")),
                            node_id=str(target_mapping.get("node_id", "")),
                            evidence_text=str(
                                target_mapping.get("evidence_text", "")
                            ),
                            source_span=tuple(
                                target_mapping.get(
                                    "source_span",
                                    (-1, -1),
                                )
                            ),
                            supporting_segment_ids=tuple(
                                str(item)
                                for item in target_mapping.get(
                                    "supporting_segment_ids",
                                    (),
                                )
                            ),
                        ),
                        semantic.faithful_ir,
                    )
                except SemanticPolicyBindingError as exc:
                    target_path = (
                        "/semantic/semantic_policy_target"
                        if prefix.startswith("/semantic")
                        else "/semantic_policy_target"
                    )
                    return (
                        ValidationIssue(
                            code="SEMANTIC_POLICY_TARGET_BINDING_INVALID",
                            stage="semantic",
                            path=target_path,
                            observed=dict(target_mapping),
                            expected={
                                "request_id": target_request.request_id,
                                "node_type": target_request.expected_node_type,
                                "same_artifact_reference": True,
                            },
                            repair_class=RepairClass.CONSTRAINED,
                            allowed_paths=(target_path,),
                            message=str(exc),
                        ),
                    )
                state["semantic_policy_target_binding"] = binding
                semantic = replace(
                    semantic,
                    semantic_policy_target_bindings=(asdict(binding),),
                )
            state["semantic"] = semantic
            state["last_failure"] = ""
            state["last_exception"] = None
            return ()

        repair_policy = RepairPolicy(
            max_domain_repairs=2,
            max_constrained_repairs=1,
            max_semantic_patches=2,
            immutable_paths=(
                "/faithful_ir/source_text",
                "/faithful_ir/source_hash",
                "/faithful_ir/schema_version",
                "/faithful_ir/contract_version",
                "/faithful_ir/canonicalizer_version",
                "/faithful_ir/validator_version",
                "/semantic/faithful_ir/source_text",
                "/semantic/faithful_ir/source_hash",
                "/semantic/faithful_ir/schema_version",
                "/semantic/faithful_ir/contract_version",
                "/semantic/faithful_ir/canonicalizer_version",
                "/semantic/faithful_ir/validator_version",
            ),
        )

        def patch(
            current: object,
            issues: tuple[ValidationIssue, ...],
            repair_class: RepairClass,
        ) -> object:
            nonlocal generation_attempt
            if _requires_typed_graph_transaction(issues):
                parent_fingerprint = semantic_graph_fingerprint(current)
                generation_attempt += 1
                raw_transaction = self.complete_json(
                    (
                        "Repair only the reported Semantic IR graph structure. "
                        "Return stable-node-ID graph operations, not JSON Patch. "
                        "Do not change AP identity, source text, hashes, or versions."
                    ),
                    {
                        **payload,
                        "task": "semantic_ir_graph_transaction",
                        "previous_invalid_output": current,
                        "precondition_fingerprint": parent_fingerprint,
                        "issues": [issue.to_dict() for issue in issues],
                        "operation_contract": (
                            semantic_graph_transaction_response_schema()
                        ),
                        "node_contract_fingerprint": node_contract_fingerprint(),
                    },
                    max_tokens=_nlu_max_tokens(),
                    response_schema=semantic_graph_transaction_response_schema(),
                )
                if not isinstance(raw_transaction, Mapping):
                    raise PatchValidationError(
                        "semantic graph repair response must be an object"
                    )
                transaction = semantic_graph_transaction_from_mapping(
                    raw_transaction,
                    proposal=current,
                )
                candidate, receipt = apply_semantic_graph_transaction(
                    current,
                    transaction,
                )
                state["last_raw_patch"] = _bounded_nlu_artifact(raw_transaction)
                return CandidateSubmission(
                    candidate=candidate,
                    changed_targets=tuple(
                        sorted(
                            {
                                getattr(operation, "node_id", "")
                                or getattr(operation, "parent_node_id", "")
                                or "root_node_id"
                                for operation in transaction.operations
                            }
                        )
                    ),
                    contract_fingerprint=receipt.after_fingerprint,
                )
            allowed_paths = sorted(
                {
                    path
                    for issue in issues
                    for path in (issue.allowed_paths or (issue.path,))
                }
            )
            authorization = patch_authorization(
                allowed_paths,
                current_document=current,
                immutable_paths=repair_policy.immutable_paths,
            )
            allowed_path_states = list(patch_path_states(current, authorization))
            retry_payload = {
                **payload,
                "task": "semantic_ir_atomic_patch",
                "previous_invalid_output": current,
                "retry_instruction": state["retry_instruction"],
                "issues": [issue.to_dict() for issue in issues],
                "allowed_paths": list(authorization.paths),
                "allowed_path_states": allowed_path_states,
                "patch_contract_version": authorization.contract_version,
                "patch_contract_fingerprint": authorization.fingerprint,
                "patch_instruction": (
                    "Return patch operations only. Modify only paths authorized by "
                    "allowed_paths. Use add when an exact target path does not exist; "
                    "use replace or remove only when it exists, as reported by "
                    "allowed_path_states. source_span, evidence_text, stable IDs, hashes, "
                    "and versions are backend-derived; never patch them unless their "
                    "exact JSON Pointer is explicitly listed in allowed_paths."
                ),
                "patch_output_schema": {
                    "operations": [
                        {
                            "op": "add for a missing path; replace|remove for an existing path",
                            "path": "/allowed/path",
                            "value": "...",
                        }
                    ]
                },
                "repair_class": repair_class.value,
            }
            generation_attempt += 1
            raw_patch = self.complete_json(
                (
                    "Repair only the reported FaithfulSemanticIR paths. "
                    "Return JSON patch operations only."
                ),
                retry_payload,
                max_tokens=_nlu_max_tokens(),
                response_schema=patch_operations_schema(authorization),
            )
            state["last_raw_patch"] = _bounded_nlu_artifact(raw_patch)
            if _patch_changes_ap_identity(raw_patch, current=current):
                raise PatchValidationError(
                    "IR_AP_REPAIR_UNAUTHORIZED: semantic patch may not change "
                    "backend-owned AP identity declarations or references; "
                    "the requested operation is not an exact authorized target"
                )
            return PatchSubmission(
                _preserve_ap_evidence_selectors_in_patch(
                    raw_patch,
                    current=current,
                ),
                authorization,
            )

        def deterministic_repair(
            current: object,
            issues: tuple[ValidationIssue, ...],
        ) -> object:
            repaired = current
            handled = False
            duplicate_issues = tuple(
                issue for issue in issues if issue.code == "DUPLICATE_AP_NAME"
            )
            if duplicate_issues:
                repaired = _repair_mergeable_duplicate_ap_names(
                    repaired,
                    duplicate_issues,
                )
                handled = True
            if any(
                issue.code == FUTURE_PROHIBITION_TOPOLOGY_CODE for issue in issues
            ):
                topology_repaired = _repair_future_prohibition_topology(
                    repaired,
                    requirement=requirement,
                )
                if topology_repaired is None:
                    raise PatchValidationError(
                        "future-prohibition topology lacks unambiguous trigger/response leaves"
                    )
                repaired = topology_repaired
                handled = True
            unary_boolean_issues = tuple(
                issue
                for issue in issues
                if (
                    issue.code == "IR_NODE_ARITY_INVALID"
                    and issue.context.get("node_type")
                    in {"conjunction", "disjunction"}
                    and issue.context.get("child_count") == 1
                )
            )
            if unary_boolean_issues:
                normalized = normalize_unary_boolean_graph(repaired)
                if normalized is None:
                    raise PatchValidationError(
                        "unary Boolean repair no longer matches its precondition"
                    )
                repaired = normalized[0]
                handled = True
            if not handled:
                raise PatchValidationError(
                    "no deterministic NLU repair is registered for these issues"
                )
            return repaired

        result = ValidatedGenerationRunner(repair_policy).generate(
            proposal_fn=propose,
            canonicalize_fn=canonicalize,
            validate_fn=validate,
            deterministic_repair_fn=deterministic_repair,
            patch_fn=patch,
            prompt_version=NLU_SEMANTIC_PROMPT_VERSION,
        )
        generation_diagnostic = _enrich_nlu_generation_diagnostic(
            result.diagnostic(),
            state,
        )
        self.last_generation_diagnostic = generation_diagnostic
        semantic = state.get("semantic")
        if result.succeeded and isinstance(semantic, SemanticStructure):
            candidate_context = state.get("semantic_policy_candidate_context")
            if isinstance(candidate_context, Mapping):
                target_binding = state.get("semantic_policy_target_binding")
                if not isinstance(
                    target_binding,
                    SemanticPolicyTargetBinding,
                ):
                    raise SemanticPolicyBindingError(
                        "policy candidate materialization succeeded without target binding"
                    )
                return materialize_policy_candidate_set(
                    semantic,
                    target_binding=target_binding,
                    options=tuple(
                        dict(item)
                        for item in candidate_context.get("options", ())
                        if isinstance(item, Mapping)
                    ),
                )
            policy_resolution = state.get("semantic_policy_resolution")
            if isinstance(policy_resolution, Mapping):
                target_binding = state.get("semantic_policy_target_binding")
                options = tuple(
                    item
                    for item in state.get("semantic_resolution_options", ())
                    if isinstance(item, Mapping)
                )
                if (
                    not isinstance(target_binding, SemanticPolicyTargetBinding)
                    or len(options) != 1
                ):
                    raise SemanticPolicyBindingError(
                        "singleton policy lacks one validated target and option"
                    )
                semantic = apply_singleton_policy_option(
                    semantic,
                    target_binding=target_binding,
                    option=options[0],
                )
                resolution_record = deepcopy(dict(policy_resolution))
                resolution_record["target_binding_id"] = (
                    target_binding.binding_id
                )
                semantic = replace(
                    semantic,
                    semantic_policy_resolutions=(resolution_record,),
                )
            return semantic
        last_exception = state.get("last_exception")
        first_kind = str(state.get("first_failure_kind", ""))
        first_failure = str(state.get("first_failure", ""))
        last_failure = (
            _nlu_issue_summary(result.issues)
            or str(state.get("last_failure", ""))
            or result.reason_code
        )
        repair_error = _last_generation_repair_error(result)
        if isinstance(last_exception, NLUOutputShapeError):
            raise last_exception
        if isinstance(last_exception, FaithfulIRValidationError) or result.issues:
            error = NLURepairExhaustedError(
                last_failure,
                first_failure_kind=first_kind,
                first_failure=first_failure,
                attempt=generation_attempt,
                reason_code=result.reason_code,
                repair_error=repair_error,
            )
            error.diagnostics.update(generation_diagnostic)
            error.diagnostics["domain_repair_attempts"] = max(
                0,
                len(tuple(getattr(result, "attempts", ()))) - 1,
            )
            rejected_failure = _nlu_mapping_issue_summary(
                getattr(result, "rejected_candidate", {}).get("issues", ())
                if isinstance(getattr(result, "rejected_candidate", {}), Mapping)
                else ()
            )
            if rejected_failure:
                error.diagnostics["rejected_candidate_validation_error"] = (
                    rejected_failure
                )
            if repair_error:
                raise error from PatchValidationError(repair_error)
            if (
                isinstance(last_exception, FaithfulIRValidationError)
                and str(last_exception) in last_failure
            ):
                raise error from last_exception
            raise error
        raise FaithfulIRValidationError(last_failure)

    def _infer_response_structure(
        self,
        requirement: str,
        rag_context: tuple[str, ...],
    ) -> SemanticStructure:
        """用本地规则模拟 LLM 对 response 结构的抽取。"""

        english_pattern = re.search(
            r"whenever\s+([A-Za-z_][A-Za-z0-9_]*)\s+occurs?.*?"
            r"([A-Za-z_][A-Za-z0-9_]*)\s+should\s+eventually\s+occurs?",
            requirement,
            flags=re.IGNORECASE,
        )
        if english_pattern:
            trigger = english_pattern.group(1)
            response = english_pattern.group(2)
            return SemanticStructure(
                trigger=trigger,
                response=response,
                temporal_keyword="eventually",
                scope_keyword="whenever",
                pattern="response",
                explanation=(
                    "识别到 whenever ... should eventually ... 句式："
                    "whenever 表示全局约束，eventually 表示最终会发生。"
                ),
                confidence=0.94,
                rag_context=rag_context,
            )

        chinese_pattern = re.search(
            r"每当\s*([A-Za-z_][A-Za-z0-9_]*)\s*发生.*?"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(最终|最终必须|最终应该|最后).*?(发生|出现)",
            requirement,
            flags=re.IGNORECASE,
        )
        if chinese_pattern:
            trigger = chinese_pattern.group(1)
            response = chinese_pattern.group(2)
            return SemanticStructure(
                trigger=trigger,
                response=response,
                temporal_keyword="最终",
                scope_keyword="每当",
                pattern="response",
                explanation="识别到“每当 A 发生，B 最终出现”的 response 需求。",
                confidence=0.9,
                rag_context=rag_context,
            )

        complex_response = _known_next_until_always_response(requirement)
        if complex_response is not None:
            trigger = complex_response["trigger"]
            response = complex_response["response"]
            witness = complex_response["witness"]
            return SemanticStructure(
                trigger=trigger,
                response=response,
                temporal_keyword="next, until, always",
                scope_keyword="if",
                pattern="response",
                explanation=(
                    "识别到“If A holds then, in the next step, B holds until C "
                    "holds or always B holds”句式；右侧是复杂 LTLf 响应，"
                    "因此保留核心 response AP，并把复合响应结构写入 components。"
                ),
                confidence=0.88,
                rag_context=rag_context,
                aps=tuple(dict.fromkeys((trigger, response, witness))),
                components=(
                    ("response_relation", "next_until_witness_or_persist_to_end"),
                    ("until_left_ap", response),
                    ("until_witness_ap", witness),
                    ("persistence_ap", response),
                ),
            )

        future_prohibition = _infer_future_prohibition_semantic(
            requirement,
            rag_context,
        )
        if future_prohibition is not None:
            return future_prohibition

        explicit_until = _infer_explicit_until_semantic(requirement, rag_context)
        if explicit_until is not None:
            return explicit_until

        implicit_precedence = _infer_implicit_precedence_semantic(
            requirement,
            rag_context,
        )
        if implicit_precedence is not None:
            return implicit_precedence

        trigger, response = self._fallback_extract_aps(requirement)
        return SemanticStructure(
            trigger=trigger,
            response=response,
            temporal_keyword="eventually",
            scope_keyword="whenever",
            pattern="response",
            explanation=(
                "未匹配到完整模板，使用兜底 AP 抽取策略；"
                "正式场景建议由人工重点评审。"
            ),
            confidence=0.55,
            rag_context=rag_context,
        )

    def _fallback_extract_aps(self, requirement: str) -> tuple[str, str]:
        """从自然语言中取前两个看起来像 AP 的英文词作为兜底结果。"""

        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", requirement)
        stopwords = {
            "whenever",
            "when",
            "if",
            "occurs",
            "occur",
            "happens",
            "happen",
            "should",
            "must",
            "eventually",
            "always",
            "then",
            "the",
            "a",
            "an",
            "is",
            "be",
        }
        aps: list[str] = []
        for word in words:
            if word.lower() not in stopwords and word not in aps:
                aps.append(word)
        trigger = aps[0] if len(aps) >= 1 else "request"
        response = aps[1] if len(aps) >= 2 else "grant"
        return trigger, response

    def _apply_feedback(
        self,
        semantic: SemanticStructure,
        feedback_history: list[str],
    ) -> SemanticStructure:
        """
        根据人工评审意见修正语义结构。

        支持类似：
            trigger=alarm
            response=shutdown
        """

        feedback_text = "\n".join(feedback_history)
        if not feedback_text.strip():
            return semantic

        trigger = _feedback_field(feedback_text, "trigger") or semantic.trigger
        response = _feedback_field(feedback_text, "response") or semantic.response
        pattern = _feedback_field(feedback_text, "pattern") or semantic.pattern
        scope_keyword = (
            _feedback_field(feedback_text, "scope_keyword") or semantic.scope_keyword
        )
        temporal_keyword = (
            _feedback_field(feedback_text, "temporal_keyword")
            or semantic.temporal_keyword
        )
        feedback_aps = _feedback_aps(feedback_text)
        feedback_components = _feedback_components(feedback_text)
        components = _merge_components(semantic.components, feedback_components)

        changed = (
            trigger != semantic.trigger
            or response != semantic.response
            or normalize_pattern(pattern) != normalize_pattern(semantic.pattern)
            or scope_keyword != semantic.scope_keyword
            or temporal_keyword != semantic.temporal_keyword
            or bool(feedback_aps)
            or components != semantic.components
        )
        if not changed:
            return semantic

        explanation = semantic.explanation.strip()
        note = "已根据评审反馈修正语义字段和结构组件。"
        if note not in explanation:
            explanation = f"{explanation} {note}".strip()

        return replace(
            semantic,
            trigger=trigger,
            response=response,
            temporal_keyword=temporal_keyword,
            scope_keyword=scope_keyword,
            pattern=normalize_pattern(pattern),
            explanation=explanation,
            confidence=max(semantic.confidence, 0.8),
            aps=feedback_aps or semantic.aps,
            components=components,
        )


_NLU_RESULT_WRAPPER_KEYS = ("semantic", "result", "output", "data", "response")
_NLU_DEFAULT_MAX_TOKENS = 4000


def _parse_llm_semantic_result(
    raw_result: dict,
    *,
    requirement: str,
    rag_context: tuple[str, ...],
    attempt: int,
    surface_segmentation: SurfaceSegmentation | None = None,
    plain_reading_review: PlainReadingReview | None = None,
) -> SemanticStructure:
    result, result_path = _unpack_llm_semantic_result(raw_result)
    try:
        reject_formula_fields(result)
    except Exception:
        _write_nlu_debug_record(
            requirement=requirement,
            raw_result=raw_result,
            parsed_result=result,
            result_path=result_path,
            event_kind="forbidden_formula_field",
            attempt=attempt,
        )
        raise

    resolution_request = result.get("semantic_resolution_request")
    if resolution_request is not None:
        if not isinstance(resolution_request, Mapping):
            raise FaithfulIRValidationError(
                "semantic_resolution_request must be an object",
                code="IR_SEMANTIC_RESOLUTION_REQUEST_INVALID",
                path="/semantic_resolution_request",
                repairable=False,
            )
        required_fields = (
            "node_id",
            "evidence_quote",
            "unresolved_parameters",
            "alternatives",
            "question",
        )
        missing = tuple(
            name
            for name in required_fields
            if resolution_request.get(name) is None
            or resolution_request.get(name) == ""
            or resolution_request.get(name) == ()
        )
        parameters = resolution_request.get("unresolved_parameters")
        alternatives = resolution_request.get("alternatives")
        if missing or not isinstance(parameters, list) or not all(
            isinstance(item, str) and item.strip() for item in parameters
        ) or not isinstance(alternatives, list) or len(alternatives) < 2 or not all(
            isinstance(item, str) and item.strip() for item in alternatives
        ):
            detail = ", ".join(missing) or "invalid parameter/alternative arrays"
            raise FaithfulIRValidationError(
                "incomplete semantic_resolution_request: " + detail,
                code="IR_SEMANTIC_RESOLUTION_REQUEST_INVALID",
                path="/semantic_resolution_request",
                repairable=False,
            )
        raise FaithfulIRValidationError(
            "semantic decision required: " + str(resolution_request["question"]),
            code="IR_SEMANTIC_RESOLUTION_REQUIRED",
            path="/semantic_resolution_request",
            repairable=False,
        )

    if _contains_ir_payload(result):
        if result_path != "$":
            _write_nlu_debug_record(
                requirement=requirement,
                raw_result=raw_result,
                parsed_result=result,
                result_path=result_path,
                event_kind="wrapped_faithful_ir",
                attempt=attempt,
            )
        try:
            result = _normalize_v5_exact_next_parameters(result)
            from semantic_topology import normalize_mechanical_scope_aliases
            result = normalize_mechanical_scope_aliases(result)
            ir = faithful_ir_from_mapping(
                result,
                source_text=requirement,
                extractor="NLUAgent",
                surface_segmentation=surface_segmentation,
                plain_reading_review=plain_reading_review,
            )
        except FaithfulIRValidationError:
            _write_nlu_debug_record(
                requirement=requirement,
                raw_result=raw_result,
                parsed_result=result,
                result_path=result_path,
                event_kind="faithful_ir_validation_error",
                attempt=attempt,
            )
            raise
        from faithful_ir import semantic_summary_from_ir

        return semantic_summary_from_ir(ir, rag_context=rag_context)

    failure_kind = _classify_missing_ir_result(result)
    _write_nlu_debug_record(
        requirement=requirement,
        raw_result=raw_result,
        parsed_result=result,
        result_path=result_path,
        event_kind=failure_kind,
        attempt=attempt,
    )
    raise NLUOutputShapeError(
        (
            "NLU output must contain AP declarations and faithful_ir; "
            "legacy trigger/response-only output is rejected. "
            f"failure_kind={failure_kind}; result_path={result_path}; "
            f"keys={_result_keys(result)}"
        ),
        failure_kind=failure_kind,
    )


def _normalize_v5_exact_next_parameters(result: dict) -> dict:
    """Make the single unambiguous ``next_step`` shorthand explicit.

    The model-facing contract requires ``steps``.  This canonicalization only
    expands the node name's exact denotation to ``steps=1``; it never guesses a
    duration, window, scope, or relation policy.
    """

    normalized = deepcopy(result)
    container = _nlu_faithful_ir_container(normalized)
    if container is None:
        return normalized
    raw_ir = container.get("faithful_ir")
    if not isinstance(raw_ir, Mapping):
        raw_ir = container if "nodes" in container else None
    if not isinstance(raw_ir, dict):
        return normalized
    raw_ir.setdefault("schema_version", FAITHFUL_IR_SCHEMA_VERSION)
    nodes = raw_ir.get("nodes")
    if not isinstance(nodes, list):
        return normalized
    for node in nodes:
        if not isinstance(node, dict) or node.get("node_type") != "next_step":
            continue
        parameters = node.get("parameters")
        if parameters is None:
            node["parameters"] = {"steps": 1}
        elif isinstance(parameters, dict) and "steps" not in parameters:
            parameters["steps"] = 1
    return normalized


def _artifact_payload(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    return value


def _unpack_llm_semantic_result(raw_result: dict) -> tuple[dict, str]:
    root = require_mapping(raw_result, "semantic")
    fallback: tuple[dict, str] | None = None
    semantic = root.get("semantic")
    if isinstance(semantic, dict):
        fallback = (semantic, "$.semantic")

    queue: list[tuple[str, dict, int]] = [("$", root, 0)]
    seen: set[int] = set()
    while queue:
        path, current, depth = queue.pop(0)
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        if _contains_ir_payload(current):
            return current, path
        if depth >= 3:
            continue
        for key in _NLU_RESULT_WRAPPER_KEYS:
            nested = current.get(key)
            if isinstance(nested, dict):
                queue.append((f"{path}.{key}", nested, depth + 1))

    if fallback is not None:
        return fallback
    return root, "$"


def _contains_ir_payload(result: dict) -> bool:
    return "faithful_ir" in result or "nodes" in result


def _classify_missing_ir_result(result: dict) -> str:
    keys = set(result)
    if _looks_like_atomic_proposition_declaration(result):
        return "atomic_proposition_without_ir"
    if {"trigger", "response"}.issubset(keys):
        return "legacy_trigger_response_only"
    return "missing_faithful_ir"


def _looks_like_atomic_proposition_declaration(result: dict) -> bool:
    return {"ap_id", "name", "source_span", "evidence_text"}.issubset(result)


def _record_nlu_failure(
    state: dict[str, object],
    failure_kind: str,
    exc: Exception,
    instruction: str,
) -> None:
    if not state.get("first_failure_kind"):
        state["first_failure_kind"] = failure_kind
        state["first_failure"] = str(exc)
    state["last_failure"] = str(exc)
    state["last_exception"] = exc
    state["retry_instruction"] = instruction


def _bounded_nlu_artifact(value: object, *, limit: int = 12000) -> object:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return {"response_type": type(value).__name__}
    if len(text) <= limit:
        return value
    return {
        "response_type": type(value).__name__,
        "sha256": __import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
        "truncated": True,
    }


def _patch_changes_ap_identity(
    patch_payload: object,
    *,
    current: object,
) -> bool:
    if not isinstance(patch_payload, Mapping):
        return False
    operations = patch_payload.get("operations")
    if not isinstance(operations, list):
        return False
    current_ir, prefix = faithful_ir_proposal_mapping(current)
    current_refs = _mapping_ap_references(current_ir)
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        path = str(operation.get("path", ""))
        relative = path[len(prefix):] if prefix and path.startswith(prefix) else path
        if relative.startswith("/atomic_propositions"):
            evidence_selector_suffixes = (
                "/occurrence_index",
                "/evidence_quote",
                "/supporting_segment_ids",
            )
            if not relative.endswith(evidence_selector_suffixes):
                return True
        if "/parameters/ap_name" in relative:
            return True
        if relative == "/nodes" and isinstance(operation.get("value"), list):
            proposed_refs = _mapping_ap_references(
                {"nodes": operation.get("value", [])}
            )
            if proposed_refs != current_refs:
                return True
    return False


def _mapping_ap_references(ir: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(ir, Mapping):
        return ()
    counts: dict[str, int] = {}
    for node in ir.get("nodes", ()):
        if not isinstance(node, Mapping) or node.get("node_type") != "ap_condition":
            continue
        parameters = node.get("parameters", {})
        ap_name = ""
        if isinstance(parameters, Mapping):
            ap_name = str(parameters.get("ap_name", ""))
        elif isinstance(parameters, list):
            ap_name = next(
                (
                    str(pair[1])
                    for pair in parameters
                    if isinstance(pair, (list, tuple))
                    and len(pair) == 2
                    and str(pair[0]) == "ap_name"
                ),
                "",
            )
        if ap_name:
            counts[ap_name] = counts.get(ap_name, 0) + 1
    return tuple(sorted(counts.items()))


def _future_prohibition_allowed_paths(value: object) -> tuple[str, ...]:
    ir, prefix = faithful_ir_proposal_mapping(value)
    if ir is None or not prefix:
        return (FUTURE_PROHIBITION_TOPOLOGY_PATH,)
    paths = [prefix + "/nodes", prefix + "/root_node_id"]
    ambiguities = ir.get("ambiguities")
    if isinstance(ambiguities, list) and ambiguities:
        paths.append(prefix + "/ambiguities")
    return tuple(paths)


def _repair_future_prohibition_topology(
    value: object,
    *,
    requirement: str,
) -> dict[str, object] | None:
    binding = resolve_future_prohibition_binding(
        value,
        source_text=requirement,
    )
    if binding is None or not isinstance(value, Mapping):
        return None
    repaired = deepcopy(dict(value))
    container = _nlu_faithful_ir_container(repaired)
    if container is None:
        return None
    raw_ir = container.get("faithful_ir")
    if not isinstance(raw_ir, Mapping):
        return None
    ir = dict(raw_ir)
    raw_nodes = ir.get("nodes")
    if not isinstance(raw_nodes, list):
        return None
    node_map = {
        str(node.get("node_id", "")): node
        for node in raw_nodes
        if isinstance(node, Mapping) and str(node.get("node_id", ""))
    }
    trigger_id = binding.trigger_node_id
    response_id = binding.prohibited_node_id
    retained_ids = subtree_node_ids(
        node_map,
        (trigger_id, response_id),
    )
    retained_nodes = [
        dict(node)
        for node in raw_nodes
        if isinstance(node, Mapping)
        and str(node.get("node_id", "")) in retained_ids
    ]
    for node in retained_nodes:
        node["scope"] = (
            "current_position"
            if str(node.get("node_type", "")) == "ap_condition"
            else "inherited"
        )
    if trigger_id not in retained_ids or response_id not in retained_ids:
        return None
    confidence = max(
        (
            coerce_float(node.get("confidence"), 0.0)
            for node in retained_nodes
        ),
        default=0.0,
    )
    used_ids = set(node_map)

    def reserve_id(stem: str) -> str:
        candidate = f"repair-{stem}"
        suffix = 2
        while candidate in used_ids:
            candidate = f"repair-{stem}-{suffix}"
            suffix += 1
        used_ids.add(candidate)
        return candidate

    prohibition_id = reserve_id("prohibition")
    suffix_id = reserve_id("suffix")
    next_id = reserve_id("next")
    implication_id = reserve_id("implication")
    root_id = reserve_id("root")
    relation_specs = (
        (
            prohibition_id,
            "negation",
            [response_id],
            "prohibited_event",
            "inherited",
            "logical_negation",
        ),
        (
            suffix_id,
            "global_constraint",
            [prohibition_id],
            "persistent_future_prohibition",
            "all_positions",
            "persistent_to_trace_end",
        ),
        (
            next_id,
            "next_step",
            [suffix_id],
            "strict_future_start",
            "inherited",
            "immediate_next_step",
        ),
        (
            implication_id,
            "implication",
            [trigger_id, next_id],
            "triggered_prohibition",
            "inherited",
            "trigger_response",
        ),
        (
            root_id,
            "global_constraint",
            [implication_id],
            "requirement_scope",
            "all_positions",
            "all_positions",
        ),
    )
    relation_nodes = [
        {
            "node_id": node_id,
            "node_type": node_type,
            "children": children,
            "semantic_role": role,
            "scope": scope,
            "temporal_relation": relation,
            "boundary_policy": "finite_trace_strict_future",
            "parameters": ({"steps": 1} if node_type == "next_step" else {}),
            "evidence_quote": requirement,
            "confidence": confidence,
        }
        for node_id, node_type, children, role, scope, relation in relation_specs
    ]
    ambiguities = ir.get("ambiguities")
    if isinstance(ambiguities, list):
        rebound_ambiguities: list[object] = []
        for raw_ambiguity in ambiguities:
            if not isinstance(raw_ambiguity, Mapping):
                rebound_ambiguities.append(raw_ambiguity)
                continue
            ambiguity = dict(raw_ambiguity)
            affected = ambiguity.get("affected_node_ids")
            if isinstance(affected, list):
                mapped = [
                    str(node_id) if str(node_id) in retained_ids else root_id
                    for node_id in affected
                ]
                ambiguity["affected_node_ids"] = list(dict.fromkeys(mapped))
            rebound_ambiguities.append(ambiguity)
        ir["ambiguities"] = rebound_ambiguities
    ir["nodes"] = [*retained_nodes, *relation_nodes]
    ir["root_node_id"] = root_id
    container["faithful_ir"] = ir
    return repaired


_NLU_AP_EVIDENCE_AND_OWNED_FIELDS = frozenset(
    {
        "ap_id",
        "evidence_quote",
        "evidence_text",
        "source_span",
        "occurrence_index",
        "supporting_segment_ids",
    }
)
_NLU_AP_EVIDENCE_SELECTOR_FIELDS = (
    "evidence_quote",
    "occurrence_index",
    "supporting_segment_ids",
)


def _nlu_duplicate_ap_name_issues(
    value: object,
) -> tuple[ValidationIssue, ...]:
    """Report duplicate AP declarations before aggregate schema validation.

    Repeated source mentions are node-level evidence occurrences, not separate
    AP declarations.  Only declarations whose non-evidence fields are exactly
    compatible are eligible for deterministic collapse.
    """

    container = _nlu_faithful_ir_container(value)
    if container is None:
        return ()
    ir = container.get("faithful_ir")
    if not isinstance(ir, Mapping):
        return ()
    aps = ir.get("atomic_propositions")
    if not isinstance(aps, list):
        return ()
    prefix = (
        "/faithful_ir"
        if container is value
        else "/semantic/faithful_ir"
    )
    indexes_by_name: dict[str, list[int]] = {}
    for index, item in enumerate(aps):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            indexes_by_name.setdefault(name, []).append(index)

    issues: list[ValidationIssue] = []
    for name, indexes in indexes_by_name.items():
        if len(indexes) < 2:
            continue
        declarations = tuple(aps[index] for index in indexes)
        mergeable = _nlu_ap_declarations_are_mergeable(declarations)
        item_paths = tuple(
            f"{prefix}/atomic_propositions/{index}" for index in indexes
        )
        issues.append(
            ValidationIssue(
                code="DUPLICATE_AP_NAME",
                stage="semantic",
                path=item_paths[1],
                observed={"name": name, "indexes": list(indexes)},
                expected={
                    "unique_ap_name": True,
                    "repeated_mentions": (
                        "multiple ap_condition nodes reference one declaration"
                    ),
                },
                context={
                    "ap_name": name,
                    "indexes": list(indexes),
                    "deterministically_mergeable": mergeable,
                },
                repair_class=(
                    RepairClass.DETERMINISTIC
                    if mergeable
                    else RepairClass.SEMANTIC_PATCH
                ),
                allowed_paths=item_paths,
                message=(
                    f"duplicate AP name {name!r}; repeated textual mentions must "
                    "share one atomic_propositions declaration"
                ),
                severity=Severity.ERROR,
            )
        )
    return tuple(issues)


def _nlu_ap_declarations_are_mergeable(
    declarations: tuple[object, ...],
) -> bool:
    semantic_payloads: list[dict[str, object]] = []
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            return False
        aliases = declaration.get("aliases", ())
        normalized_aliases = (
            tuple(str(item) for item in aliases)
            if isinstance(aliases, (list, tuple))
            else aliases
        )
        known_semantic_fields = {
            "name",
            "aliases",
            "formula_presence",
            "required_in_formula",
            "optional_in_formula",
        }
        semantic_payload = {
            "name": str(declaration.get("name", "")),
            "aliases": normalized_aliases,
            "formula_presence": str(
                declaration.get("formula_presence", "unspecified")
            ),
            "required_in_formula": bool(
                declaration.get("required_in_formula", False)
            ),
            "optional_in_formula": bool(
                declaration.get("optional_in_formula", False)
            ),
        }
        semantic_payload.update(
            {
                str(key): value
                for key, value in declaration.items()
                if str(key) not in _NLU_AP_EVIDENCE_AND_OWNED_FIELDS
                and str(key) not in known_semantic_fields
            }
        )
        semantic_payloads.append(semantic_payload)
    return bool(semantic_payloads) and all(
        payload == semantic_payloads[0] for payload in semantic_payloads[1:]
    )


def _repair_mergeable_duplicate_ap_names(
    value: object,
    issues: tuple[ValidationIssue, ...],
) -> object:
    repaired = deepcopy(value)
    container = _nlu_faithful_ir_container(repaired)
    if container is None:
        raise PatchValidationError("duplicate-AP repair requires faithful_ir")
    ir = container.get("faithful_ir")
    if not isinstance(ir, dict):
        raise PatchValidationError("duplicate-AP repair requires a mutable IR object")
    aps = ir.get("atomic_propositions")
    if not isinstance(aps, list):
        raise PatchValidationError("duplicate-AP repair requires an AP array")
    names = {
        str(issue.context.get("ap_name", ""))
        for issue in issues
        if issue.code == "DUPLICATE_AP_NAME"
        and bool(issue.context.get("deterministically_mergeable"))
    }
    names.discard("")
    if not names:
        raise PatchValidationError("no mergeable duplicate AP name was selected")
    for name in names:
        declarations = tuple(
            item
            for item in aps
            if isinstance(item, Mapping) and str(item.get("name", "")) == name
        )
        if len(declarations) < 2 or not _nlu_ap_declarations_are_mergeable(
            declarations
        ):
            raise PatchValidationError(
                f"duplicate AP declarations for {name!r} are not safely mergeable"
            )
    seen: set[str] = set()
    retained: list[object] = []
    for item in aps:
        name = str(item.get("name", "")) if isinstance(item, Mapping) else ""
        if name in names:
            if name in seen:
                continue
            seen.add(name)
        retained.append(item)
    ir["atomic_propositions"] = retained
    return repaired


def _nlu_duplicate_ap_repair_instruction(
    issues: tuple[ValidationIssue, ...],
) -> str:
    names = sorted(
        {
            str(issue.context.get("ap_name", ""))
            for issue in issues
            if str(issue.context.get("ap_name", ""))
        }
    )
    return (
        "Each AP name must be declared exactly once. Consolidate duplicate AP "
        f"declarations for {', '.join(names)} while preserving one declaration's "
        "exact evidence selector. Keep every ap_condition occurrence and make it "
        "reference the retained AP name; do not replace the entire IR."
    )


def _preserve_ap_evidence_selectors_in_patch(
    patch_response: dict[str, object],
    *,
    current: object,
) -> dict[str, object]:
    """Prevent AP container replacement from implicitly dropping selectors."""

    raw_operations = patch_response.get("operations")
    container = _nlu_faithful_ir_container(current)
    if not isinstance(raw_operations, list) or container is None:
        return patch_response
    ir = container.get("faithful_ir")
    if not isinstance(ir, Mapping):
        return patch_response
    current_aps = ir.get("atomic_propositions")
    if not isinstance(current_aps, list):
        return patch_response
    prefix = (
        "/faithful_ir/atomic_propositions"
        if container is current
        else "/semantic/faithful_ir/atomic_propositions"
    )
    normalized: list[object] = []
    for raw_operation in raw_operations:
        if not isinstance(raw_operation, Mapping):
            normalized.append(raw_operation)
            continue
        operation = deepcopy(dict(raw_operation))
        if operation.get("op") not in {"add", "replace"}:
            normalized.append(operation)
            continue
        path = str(operation.get("path", ""))
        if path == prefix and isinstance(operation.get("value"), list):
            operation["value"] = _inherit_ap_selector_list(
                operation["value"],
                current_aps,
            )
        else:
            match = re.fullmatch(re.escape(prefix) + r"/(\d+)", path)
            if match and isinstance(operation.get("value"), Mapping):
                index = int(match.group(1))
                if 0 <= index < len(current_aps) and isinstance(
                    current_aps[index], Mapping
                ):
                    operation["value"] = _inherit_ap_selectors(
                        operation["value"],
                        current_aps[index],
                    )
        normalized.append(operation)
    return {"operations": normalized}


def _inherit_ap_selector_list(
    proposed: list[object],
    current: list[object],
) -> list[object]:
    current_by_name: dict[str, list[Mapping[str, object]]] = {}
    for item in current:
        if isinstance(item, Mapping):
            current_by_name.setdefault(str(item.get("name", "")), []).append(item)
    used_by_name: dict[str, int] = {}
    inherited: list[object] = []
    for item in proposed:
        if not isinstance(item, Mapping):
            inherited.append(item)
            continue
        name = str(item.get("name", ""))
        candidates = current_by_name.get(name, [])
        if not candidates:
            inherited.append(dict(item))
            continue
        used = used_by_name.get(name, 0)
        source = candidates[min(used, len(candidates) - 1)]
        used_by_name[name] = used + 1
        inherited.append(_inherit_ap_selectors(item, source))
    return inherited


def _inherit_ap_selectors(
    proposed: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, object]:
    inherited = dict(proposed)
    for field in _NLU_AP_EVIDENCE_SELECTOR_FIELDS:
        if field not in inherited and field in current:
            inherited[field] = deepcopy(current[field])
    return inherited


def _nlu_rebase_finding_path(value: object, path: str) -> str:
    container = _nlu_faithful_ir_container(value)
    if (
        container is not None
        and container is not value
        and path.startswith("/faithful_ir")
    ):
        return "/semantic" + path
    return path


def _requires_typed_graph_transaction(
    issues: tuple[ValidationIssue, ...],
) -> bool:
    graph_codes = {
        FUTURE_PROHIBITION_TOPOLOGY_CODE,
        "IR_NODE_ARITY_INVALID",
        "IR_SCOPE_TOPOLOGY_MISMATCH",
    }
    return any(
        issue.code in graph_codes
        or issue.code.startswith("IR_GRAPH_")
        for issue in issues
    )


def _nlu_finding_allowed_paths(
    value: object,
    finding: FaithfulIRFinding,
) -> tuple[str, ...]:
    """Map domain-owned repair scopes without parsing diagnostic prose."""

    container = _nlu_faithful_ir_container(value)
    if container is None:
        return (finding.path,)
    prefix = (
        "/faithful_ir"
        if container is value
        else "/semantic/faithful_ir"
    )
    raw_ir = container.get("faithful_ir")
    if not isinstance(raw_ir, Mapping):
        return (prefix,)

    if finding.allowed_paths:
        return tuple(
            _nlu_rebase_finding_path(value, path)
            for path in finding.allowed_paths
        )
    rebased_path = _nlu_rebase_finding_path(value, finding.path)
    if (
        finding.code == "IR_REVIEW_AXIS_CONTENT_MISMATCH"
        and not rebased_path.endswith("/alternatives")
    ):
        return (rebased_path.rstrip("/") + "/alternatives",)
    if rebased_path != prefix:
        return (rebased_path,)

    # Generic IR validation can require graph-level changes, but repair still
    # cannot replace the IR root that owns source/version metadata.
    return (
        f"{prefix}/atomic_propositions",
        f"{prefix}/root_node_id",
        f"{prefix}/nodes",
        f"{prefix}/ambiguities",
    )


def _nlu_faithful_ir_container(
    value: object,
) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("faithful_ir"), Mapping):
        return value
    semantic = value.get("semantic")
    if isinstance(semantic, dict) and isinstance(
        semantic.get("faithful_ir"), Mapping
    ):
        return semantic
    return None


def _nlu_retry_instruction(failure_kind: str) -> str:
    base = (
        "The previous NLU output was rejected because it did not contain "
        f"faithful_ir. Return only one schema-v{FAITHFUL_IR_SCHEMA_VERSION} root JSON object with "
        "faithful_ir. Do not return legacy trigger/response-only JSON, do "
        "not output formula fields, and keep explanation brief."
    )
    if failure_kind == "atomic_proposition_without_ir":
        return (
            base
            + " The previous output looked like a single "
            "faithful_ir.atomic_propositions[] entry. Wrap all AP declarations "
            "inside faithful_ir.atomic_propositions and include faithful_ir."
            "root_node_id plus a connected faithful_ir.nodes graph."
        )
    return base


def _relation_parameter_schema() -> dict[str, dict[str, list[str]]]:
    return {
        "until": {
            name: sorted(values) for name, values in UNTIL_PARAMETER_VALUES.items()
        },
        "precedence": {
            name: sorted(values)
            for name, values in PRECEDENCE_PARAMETER_VALUES.items()
        },
        "periodicity": {
            name: sorted(values)
            for name, values in PERIODICITY_PARAMETER_VALUES.items()
        },
    }


def _nlu_parameter_constraints() -> dict[str, object]:
    constraints = node_parameter_constraints_payload()
    periodicity = dict(constraints["periodicity"])
    periodicity["mode_matrix"] = periodicity_mode_constraints_payload()
    periodicity["unresolved_representation"] = (
        "emit semantic_resolution_request instead of incomplete FaithfulSemanticIR"
    )
    constraints["periodicity"] = periodicity
    for node_type in ("precedence", "before"):
        relation = dict(constraints[node_type])
        relation["currently_executable"] = {
            "reuse_policy": "reusable",
            "scope_origin": "trace_start",
        }
        relation["unsupported_policy_handling"] = (
            "request semantic resolution; do not coerce the policy"
        )
        constraints[node_type] = relation
    return constraints


def _relation_parameter_repair_rule() -> str:
    schema = _relation_parameter_schema()
    rendered = "; ".join(
        f"{node_type}: "
        + ", ".join(
            f"{name}={'|'.join(values)}" for name, values in parameters.items()
        )
        for node_type, parameters in schema.items()
    )
    periodicity_matrix = periodicity_mode_constraints_payload()
    rendered_matrix = "; ".join(
        f"{mode}: required={','.join(spec['required'])}; "
        f"forbidden={','.join(spec['forbidden'])}; "
        "fixed_values="
        + ",".join(
            f"{name}={value}"
            for name, value in dict(spec["fixed_values"]).items()
        )
        for mode, spec in periodicity_matrix.items()
    )
    return (
        " Use these exact relation parameter enums and do not invent aliases: "
        + rendered
        + ". Use this authoritative periodicity mode matrix: "
        + rendered_matrix
        + ". periodicity.period must be a JSON integer >= 1. phase is fixed_phase-only "
        "and must satisfy 0 <= phase < period. If a "
        "business decision is unresolved, return a semantic resolution request, not IR. Never use "
        "unknown, unspecified, or unconstrained as a parameter value."
    )


def _nlu_ir_repair_instruction(validation_error: str) -> str:
    return (
        f"The previous NLU output contained faithful_ir but failed schema-v{FAITHFUL_IR_SCHEMA_VERSION} "
        "validation. Repair only the faithful_ir structure and evidence fields; "
        "do not change the requirement meaning, do not output formulas, and "
        "return exactly one root JSON object with faithful_ir. Validation error: "
        f"{validation_error}. Use only supported node_type values from "
        "output_schema.supported_node_types; for next-step semantics use "
        "node_type='next_step', not 'next'. source_span, evidence_text, stable "
        "IDs, hashes, and versions are backend-derived fields; repair the "
        "evidence_quote or occurrence_index that determines them instead of "
        "patching numeric offsets."
        + _relation_parameter_repair_rule()
    )


def _nlu_canonical_repair_instruction(
    issues: tuple[ValidationIssue, ...],
) -> str:
    paths = sorted(
        {
            path
            for issue in issues
            for path in (issue.allowed_paths or (issue.path,))
        }
    )
    if any(issue.code.startswith("IR_GRAPH_") for issue in issues):
        return (
            "The previous FaithfulSemanticIR proposal has a malformed temporary "
            "node graph. Patch only the authorized nodes/root paths. Every node "
            "must have a unique non-empty temporary node_id; root_node_id and "
            "every children reference must resolve; the graph must be acyclic "
            "and every node must be reachable from the root. Do not patch "
            "backend-derived stable IDs, hashes, versions, source spans, or "
            "evidence text. Authorized paths: "
            + ", ".join(paths)
        )
    return (
        "The previous NLU proposal has unresolved backend evidence grounding. "
        "Patch only these exact evidence-selection paths: "
        + ", ".join(paths)
        + ". For EVIDENCE_QUOTE_OCCURRENCE_REQUIRED, choose the occurrence_index "
        "whose candidate span denotes the intended source evidence. Do not patch "
        "source_span, evidence_text, supporting segments, stable IDs, hashes, or "
        "versions; the backend will derive them after the evidence selector is valid."
    )


def _nlu_issue_summary(issues: object) -> str:
    if not isinstance(issues, (list, tuple)):
        return ""
    messages: list[str] = []
    for issue in issues:
        message = str(getattr(issue, "message", "")).strip()
        code = str(getattr(issue, "code", "")).strip()
        path = str(getattr(issue, "path", "")).strip()
        text = message or code
        if text and path:
            text = f"{text} ({path})"
        if text and text not in messages:
            messages.append(text)
    return "; ".join(messages)


def _nlu_mapping_issue_summary(issues: object) -> str:
    if not isinstance(issues, (list, tuple)):
        return ""
    messages: list[str] = []
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        message = str(issue.get("message", "")).strip()
        code = str(issue.get("code", "")).strip()
        path = str(issue.get("path", "")).strip()
        text = message or code
        if text and path:
            text = f"{text} ({path})"
        if text and text not in messages:
            messages.append(text)
    return "; ".join(messages)


def _enrich_nlu_generation_diagnostic(
    diagnostic: Mapping[str, object],
    state: Mapping[str, object],
) -> dict[str, object]:
    enriched = deepcopy(dict(diagnostic))
    binding = state.get("future_prohibition_binding")
    enriched["participant_resolution"] = (
        deepcopy(binding) if isinstance(binding, Mapping) else None
    )
    enriched["deterministic_repair_eligible"] = isinstance(binding, Mapping)
    policy_resolution = state.get("semantic_policy_resolution")
    enriched["semantic_policy_resolution"] = (
        deepcopy(policy_resolution)
        if isinstance(policy_resolution, Mapping)
        else None
    )
    enriched["policy_resolution_applied"] = isinstance(
        policy_resolution,
        Mapping,
    )
    semantic_request = state.get("semantic_resolution_request")
    enriched["semantic_resolution_request"] = (
        deepcopy(semantic_request)
        if isinstance(semantic_request, Mapping)
        else None
    )
    semantic_options = state.get("semantic_resolution_options")
    enriched["semantic_resolution_options"] = deepcopy(
        list(semantic_options)
        if isinstance(semantic_options, (tuple, list))
        else []
    )
    candidate_context = state.get("semantic_policy_candidate_context")
    enriched["policy_candidate_branching_applied"] = isinstance(
        candidate_context,
        Mapping,
    )
    enriched["semantic_policy_candidate_context"] = (
        deepcopy(dict(candidate_context))
        if isinstance(candidate_context, Mapping)
        else None
    )
    raw_patch = state.get("last_raw_patch")
    if raw_patch is not None:
        enriched["last_patch_response"] = deepcopy(raw_patch)

    current_graph_issues = [
        item
        for item in enriched.get("current_issues", ())
        if isinstance(item, Mapping)
        and str(item.get("code", "")).startswith("IR_GRAPH_")
    ]
    rejected = enriched.get("rejected_candidate")
    rejected_graph_issues: list[object] = []
    if isinstance(rejected, Mapping):
        rejected_graph_issues = [
            item
            for item in rejected.get("issues", ())
            if isinstance(item, Mapping)
            and str(item.get("code", "")).startswith("IR_GRAPH_")
        ]
    enriched["raw_graph_issues"] = deepcopy(current_graph_issues)
    enriched["candidate_graph_issues"] = deepcopy(rejected_graph_issues)

    reason_code = str(enriched.get("reason_code", ""))
    if rejected_graph_issues:
        termination_kind = "repair_introduced_graph_errors"
    elif reason_code == "REGRESSION_ROLLED_BACK":
        termination_kind = "repair_no_progress_rolled_back"
    elif reason_code == "NO_PROGRESS":
        termination_kind = "repair_no_progress"
    elif reason_code.endswith("BUDGET_EXHAUSTED"):
        termination_kind = "repair_budget_exhausted"
    elif reason_code == "HUMAN_REVIEW_REQUIRED":
        termination_kind = "semantic_adjudication_required"
    else:
        termination_kind = ""
    enriched["repair_termination_kind"] = termination_kind
    return enriched


def _last_generation_repair_error(result: object) -> str:
    attempts = getattr(result, "attempts", ())
    for attempt in reversed(tuple(attempts)):
        repair_error = str(getattr(attempt, "repair_error", "")).strip()
        if repair_error:
            return repair_error
    return ""


def _result_keys(result: dict) -> list[str]:
    return sorted(str(key) for key in result.keys())


def _nlu_max_tokens() -> int:
    raw_value = os.getenv("TRAJ_LTLF_NLU_MAX_TOKENS", "")
    if raw_value:
        try:
            value = int(raw_value)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return _NLU_DEFAULT_MAX_TOKENS


def _write_nlu_debug_record(
    *,
    requirement: str,
    raw_result: dict,
    parsed_result: dict,
    result_path: str,
    event_kind: str,
    attempt: int,
) -> None:
    if os.getenv("TRAJ_LTLF_NLU_DEBUG", "").lower() not in {"1", "true", "yes"}:
        return
    debug_dir = Path(
        os.getenv("TRAJ_LTLF_NLU_DEBUG_DIR", ".traj_ltlf_debug/nlu")
    )
    payload = {
        "event_kind": event_kind,
        "attempt": attempt,
        "result_path": result_path,
        "requirement": requirement,
        "raw_shape": _json_shape(raw_result),
        "parsed_shape": _json_shape(parsed_result),
        "raw_result": raw_result,
        "parsed_result": parsed_result,
    }
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        (debug_dir / f"nlu_{event_kind}_{timestamp}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError:
        return


def _json_shape(value: object) -> object:
    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": sorted(str(key) for key in value.keys()),
        }
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    return {"type": type(value).__name__}


def _feedback_field(feedback_text: str, field_name: str) -> str:
    match = re.search(
        rf"^\s*{re.escape(field_name)}\s*[:=：]\s*(.+?)\s*$",
        feedback_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _guided_query_text(
    guided_context: GuidedLearningContext | None,
    stage: str,
) -> str:
    if guided_context is None:
        return ""
    return (
        f"warmup guided {stage} {guided_context.requirement_id} "
        "stage_scoped_teacher_signal"
    )


def _feedback_aps(feedback_text: str) -> tuple[str, ...]:
    raw_value = _feedback_field(feedback_text, "aps")
    if not raw_value:
        return ()
    aps: list[str] = []
    for raw_ap in raw_value.split(","):
        ap = raw_ap.strip()
        if not ap:
            continue
        aps.append(require_ap_name(ap, "feedback.aps"))
    return tuple(dict.fromkeys(aps))


def _feedback_components(feedback_text: str) -> tuple[tuple[str, str], ...]:
    components: list[tuple[str, str]] = []
    for match in re.finditer(
        r"^\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$",
        feedback_text,
        flags=re.MULTILINE,
    ):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if value:
            components.append((key, value))
    return tuple(components)


def _merge_components(
    original: tuple[tuple[str, str], ...],
    updates: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if not updates:
        return original
    merged = dict(original)
    for key, value in updates:
        merged[key] = value
    return tuple(merged.items())


def _semantic_extraction_contract() -> dict:
    return {
        "agent": "NLUAgent",
        "stage": "semantic_extraction",
        "responsibilities": [
            "Extract AP declarations and a connected FaithfulSemanticIR from the current requirement.",
            "Preserve Boolean topology, temporal nesting, scope, boundary policy, parameters, ambiguities, and exact source evidence.",
            "Use only approved formula-free IR examples as format and reasoning references.",
        ],
        "forbidden": [
            "Do not generate formula_text or complete LTLf formulas in the semantic stage.",
            "Do not generate finite traces.",
            "Do not generate skeletons, holes, completions, or final candidate rankings.",
            "Do not compress complex semantics into trigger/response or pattern summaries.",
            "Do not invent APs or unsupported temporal meaning.",
        ],
    }


def _aps_from_llm_result(result: dict) -> tuple[str, ...]:
    raw_aps = result.get("aps")
    aps: list[str] = []
    if isinstance(raw_aps, list):
        for index, item in enumerate(raw_aps, start=1):
            if not isinstance(item, str):
                raise ValueError(f"semantic.aps[{index}] 必须是 AP 名称字符串。")
            candidate = _coerce_ap_value(
                item,
                known_aps=(),
                allow_formula_fragment=False,
            )
            aps.append(require_ap_name(candidate or item, f"semantic.aps[{index}]"))
    return tuple(dict.fromkeys(aps))


def _extend_aps_from_complex_fields(
    result: dict,
    aps: tuple[str, ...],
) -> tuple[str, ...]:
    completed = list(aps)
    for field_name in ("trigger", "response"):
        raw_value = optional_string(result, field_name, "")
        if not raw_value or not _looks_like_formula_fragment(raw_value):
            continue
        for token in _formula_fragment_ap_tokens(raw_value):
            ap = require_ap_name(token, f"semantic.{field_name}")
            if ap not in completed:
                completed.append(ap)
    return tuple(completed)


def _with_complex_field_components(
    components: tuple[tuple[str, str], ...],
    result: dict,
) -> tuple[tuple[str, str], ...]:
    completed = components
    field_components = (
        ("trigger", "trigger_definition"),
        ("response", "response_formula"),
    )
    for field_name, component_key in field_components:
        raw_value = optional_string(result, field_name, "")
        if not raw_value or not _looks_like_formula_fragment(raw_value):
            continue
        completed = _with_component(
            completed,
            component_key,
            _clean_formula_fragment(raw_value),
        )
    return completed


def _ap_field_or_fallback(
    result: dict,
    field_name: str,
    aps: tuple[str, ...],
    fallback_index: int,
    context: str,
    *,
    allow_formula_fragment: bool = False,
) -> str:
    raw_value = optional_string(result, field_name, "")
    if raw_value:
        candidate = _coerce_ap_value(
            raw_value,
            known_aps=aps,
            allow_formula_fragment=allow_formula_fragment,
        )
        if candidate:
            return require_ap_name(candidate, context)
        raise ValueError(
            f"{context} 必须是合法 AP 名称，只能使用英文字母、数字和下划线，"
            "且不能以数字开头。"
            f" 当前值为 {raw_value!r}。若该字段表达的是复杂响应公式，"
            "请在 components 中描述复合结构，并在 aps 中列出真实 AP。"
        )
    if aps:
        return aps[min(fallback_index, len(aps) - 1)]
    return "unknown_ap"


_AP_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_AP_FIELD_LABEL_PATTERN = re.compile(
    r"^(?:trigger|response|触发事件|响应事件)\s*[:=：]\s*",
    flags=re.IGNORECASE,
)
_AP_FIELD_NOISE_WORDS = {
    "ap",
    "and",
    "always",
    "be",
    "enabled",
    "eventually",
    "finally",
    "globally",
    "happen",
    "happens",
    "hold",
    "holds",
    "if",
    "in",
    "is",
    "must",
    "next",
    "not",
    "occur",
    "occurs",
    "or",
    "response",
    "should",
    "step",
    "temporal",
    "then",
    "the",
    "true",
    "false",
    "trigger",
    "until",
}


def _coerce_ap_value(
    value: str,
    *,
    known_aps: tuple[str, ...],
    allow_formula_fragment: bool,
) -> str:
    """Extract an AP name from slightly noisy LLM fields without accepting formulas."""

    cleaned = _strip_ap_wrappers(value)
    if AP_PATTERN.fullmatch(cleaned):
        return cleaned

    tokens = _AP_IDENTIFIER_PATTERN.findall(cleaned)
    known_tokens = [token for token in tokens if token in known_aps]
    if allow_formula_fragment and known_tokens:
        return known_tokens[0]

    meaningful = [
        token
        for token in tokens
        if token.lower() not in _AP_FIELD_NOISE_WORDS
        and token.upper() not in {"X", "F", "G", "U"}
    ]
    unique_meaningful = tuple(dict.fromkeys(meaningful))
    if len(unique_meaningful) == 1:
        return unique_meaningful[0]
    return ""


def _looks_like_formula_fragment(value: str) -> bool:
    cleaned = _clean_formula_fragment(value)
    if not cleaned or AP_PATTERN.fullmatch(cleaned):
        return False
    tokens = _formula_fragment_ap_tokens(cleaned)
    if len(tokens) > 1:
        return True
    return any(marker in cleaned for marker in ("|", "&", "!", "->", " U "))


def _formula_fragment_ap_tokens(value: str) -> tuple[str, ...]:
    cleaned = _clean_formula_fragment(value)
    tokens = _AP_IDENTIFIER_PATTERN.findall(cleaned)
    meaningful = [
        token
        for token in tokens
        if token.lower() not in _AP_FIELD_NOISE_WORDS
        and token.upper() not in {"X", "F", "G", "U"}
    ]
    return tuple(dict.fromkeys(meaningful))


def _clean_formula_fragment(value: str) -> str:
    cleaned = _AP_FIELD_LABEL_PATTERN.sub("", value.strip())
    cleaned = cleaned.strip("`\"'“”‘’")
    return cleaned.rstrip(".,;:，。；：")


_NL_AP = r"[A-Za-z_][A-Za-z0-9_]*"


def _infer_explicit_until_semantic(
    requirement: str,
    rag_context: tuple[str, ...],
) -> SemanticStructure | None:
    """Create an Until candidate without silently choosing strong/weak policy."""

    end_match = re.search(
        rf"\(?\b(?P<left>{_NL_AP})\b\)?\s+"
        r"(?:(?:must|should|shall)\s+)?(?:holds?\s+)?"
        r"(?:until|till)\s+(?:the\s+)?end(?:\s+of\s+(?:the\s+)?trace)?",
        requirement,
        flags=re.IGNORECASE,
    )
    if end_match:
        left = end_match.group("left")
        return SemanticStructure(
            trigger=left,
            response=left,
            temporal_keyword="persist to finite-trace end",
            scope_keyword="current suffix",
            pattern="persistence",
            explanation=(
                "Explicit until-the-end wording denotes suffix persistence; "
                "no synthetic end atomic proposition is introduced."
            ),
            confidence=0.96,
            rag_context=rag_context,
            aps=(left,),
            components=(("scope_origin", "current_position"),),
        )

    match = re.search(
        rf"\(?\b(?P<left>{_NL_AP})\b\)?\s+"
        r"(?:(?:must|should|shall)\s+)?(?:holds?\s+)?"
        rf"(?:until|till)\s+\(?\b(?P<right>{_NL_AP})\b\)?",
        requirement,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    left = match.group("left")
    right = match.group("right")
    if left.lower() in {"hold", "holds", "until", "till"} or left == right:
        return None
    prefix = requirement[: match.start("left")].lower()
    negated_left = bool(
        re.search(r"(?:do\s+not|don't|never|not)\s*\(?\s*$", prefix)
    )
    lower = requirement.lower()
    required_witness = any(
        marker in lower
        for marker in (
            "eventually occurs",
            "eventually happens",
            "must occur",
            "must happen",
            "is guaranteed to occur",
            "is reached",
        )
    )
    strict_future = any(
        marker in lower for marker in ("from the next step", "starting next step")
    )
    components: list[tuple[str, str]] = [
        ("until_left_ap", left),
        ("until_witness_ap", right),
        ("until_left_negated", "true" if negated_left else "false"),
        ("scope_origin", "current_position"),
    ]
    if strict_future:
        components.append(("witness_timing", "strict_future"))
    if required_witness:
        components.append(("witness_policy", "required"))
        if not strict_future:
            components.append(("witness_timing", "current_or_future"))
        components.append(("left_at_witness", "not_required"))
    return SemanticStructure(
        trigger=left,
        response=right,
        temporal_keyword="explicit until",
        scope_keyword="current position with finite-trace boundary review",
        pattern="until",
        explanation=(
            "Explicit until wording fixes the ordered left/right relation. "
            + (
                "The text explicitly supports a required witness."
                if required_witness
                else "Strong/weak and witness-boundary policy remain unresolved."
            )
        ),
        confidence=0.9 if required_witness else 0.82,
        rag_context=rag_context,
        aps=tuple(dict.fromkeys((left, right))),
        components=tuple(components),
    )


def _infer_implicit_precedence_semantic(
    requirement: str,
    rag_context: tuple[str, ...],
) -> SemanticStructure | None:
    """Recognize evidence-backed prefix gates without emitting an LTLf formula."""

    patterns = (
        rf"(?P<a>{_NL_AP})\s+can\s+only\s+happen\s+if\s+"
        rf"(?P<b>{_NL_AP})\s+(?:happened|happend|occurred)\s+before",
        rf"(?P<a>{_NL_AP})\s+requires?\s+an\s+earlier\s+(?P<b>{_NL_AP})",
        rf"no\s+(?P<a>{_NL_AP})\s+is\s+permitted\s+prior\s+to\s+(?P<b>{_NL_AP})",
        rf"(?P<a>{_NL_AP})\s+is\s+disabled\s+pending\s+(?P<b>{_NL_AP})",
        rf"(?P<b>{_NL_AP})\s+is\s+a\s+prerequisite\s+for\s+(?P<a>{_NL_AP})",
        rf"(?P<a>{_NL_AP})\s+becomes\s+enabled\s+once\s+"
        rf"(?P<b>{_NL_AP})\s+has\s+occurred",
    )
    match = next(
        (
            candidate
            for pattern in patterns
            if (candidate := re.search(pattern, requirement, flags=re.IGNORECASE))
        ),
        None,
    )
    if match is None:
        return None
    constrained = match.group("a")
    prerequisite = match.group("b")
    lower = match.group(0).lower()
    strict = any(
        marker in lower
        for marker in ("before", "earlier", "prior", "has occurred", "happened")
    )
    ordering = "strict" if strict else "non_strict"
    same_step = "forbidden" if strict else "allowed"
    components: tuple[tuple[str, str], ...] = (
        ("constrained_event", constrained),
        ("prerequisite_event", prerequisite),
        ("precedence_relation", f"{constrained} only after prior {prerequisite}"),
    )
    if strict:
        components = (
            *components,
            ("ordering", ordering),
            ("witness_policy", "optional_if_no_constrained"),
            ("same_step_policy", same_step),
            ("reuse_policy", "reusable"),
            ("scope_origin", "trace_start"),
        )
    else:
        components = (*components, ("precedence_policy_unresolved", "true"))
    return SemanticStructure(
        trigger=constrained,
        response=prerequisite,
        temporal_keyword=(
            "strict before/precedence" if strict else "precedence gate"
        ),
        scope_keyword="finite trace from start",
        pattern="precedence",
        explanation=(
            "The wording establishes a prefix gate: the constrained event is "
            "not permitted before the prerequisite event."
        ),
        confidence=0.92 if strict else 0.82,
        rag_context=rag_context,
        aps=tuple(dict.fromkeys((constrained, prerequisite))),
        components=components,
    )


def _infer_future_prohibition_semantic(
    requirement: str,
    rag_context: tuple[str, ...],
) -> SemanticStructure | None:
    """Recognize a trigger followed by a persistent strict-future prohibition."""

    participants = future_prohibition_lexical_participants(requirement)
    if participants is None:
        return None
    trigger, response = participants
    return SemanticStructure(
        trigger=trigger,
        response=response,
        temporal_keyword="strict-future persistent prohibition",
        scope_keyword="globally after each trigger",
        pattern="response",
        explanation=(
            "The wording establishes a global trigger whose strict-future suffix "
            "must continuously exclude the prohibited event."
        ),
        confidence=0.95,
        rag_context=rag_context,
        aps=tuple(dict.fromkeys((trigger, response))),
        components=(("response_relation", "post_trigger_persistent_prohibition"),),
    )


def _strip_ap_wrappers(value: str) -> str:
    cleaned = _clean_formula_fragment(value)
    while cleaned.startswith("(") and cleaned.endswith(")"):
        inner = cleaned[1:-1].strip()
        if not AP_PATTERN.fullmatch(inner):
            break
        cleaned = inner
    return cleaned


_NEXT_UNTIL_ALWAYS_RESPONSE_PATTERN = re.compile(
    r"\bif\s+(?P<trigger>[A-Za-z_][A-Za-z0-9_]*)\s+holds?\s+"
    r"then,?\s+in\s+the\s+next\s+step,?\s+"
    r"(?P<response>[A-Za-z_][A-Za-z0-9_]*)\s+holds?\s+until\s+"
    r"(?P<witness>[A-Za-z_][A-Za-z0-9_]*)\s+holds?\s+or\s+always\s+"
    r"(?P<always_response>[A-Za-z_][A-Za-z0-9_]*)\s+holds?\b",
    flags=re.IGNORECASE,
)


def _known_next_until_always_response(requirement: str) -> dict[str, str] | None:
    match = _NEXT_UNTIL_ALWAYS_RESPONSE_PATTERN.search(requirement)
    if match is None:
        return None
    trigger = match.group("trigger")
    response = match.group("response")
    witness = match.group("witness")
    always_response = match.group("always_response")
    if always_response.lower() != response.lower():
        return None
    return {
        "trigger": trigger,
        "response": response,
        "witness": witness,
    }


def _apply_known_next_until_always_response(
    semantic: SemanticStructure,
    requirement: str,
) -> SemanticStructure:
    known = _known_next_until_always_response(requirement)
    if known is None:
        return semantic

    trigger = known["trigger"]
    response = known["response"]
    witness = known["witness"]
    explanation = semantic.explanation.strip()
    note = (
        "该需求的响应侧包含 next/until/always 复合结构；"
        "trigger/response 字段只保存核心 AP，复合响应结构记录在 components 中。"
    )
    if note not in explanation:
        explanation = f"{explanation} {note}".strip()
    return replace(
        semantic,
        trigger=trigger,
        response=response,
        temporal_keyword="next, until, always",
        scope_keyword=semantic.scope_keyword or "if",
        pattern="response",
        explanation=explanation,
        confidence=max(semantic.confidence, 0.88),
        aps=tuple(dict.fromkeys((*semantic.aps, trigger, response, witness))),
        components=_merge_components(
            semantic.components,
            (
                ("response_relation", "next_until_witness_or_persist_to_end"),
                ("until_left_ap", response),
                ("until_witness_ap", witness),
                ("persistence_ap", response),
            ),
        ),
    )


def _components_from_llm_result(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        return ()
    components: list[tuple[str, str]] = []
    for key, item in value.items():
        if not isinstance(key, str) or not AP_PATTERN.fullmatch(key):
            continue
        if isinstance(item, str) and item.strip():
            components.append((key, item.strip()))
    return tuple(components)


def _rewrite_unsupported_strict_precedence(
    semantic: SemanticStructure,
    requirement: str,
) -> SemanticStructure:
    """Record common strict-before semantics without producing a full formula."""

    if not _looks_like_strict_precedence(semantic, requirement):
        return semantic

    trigger = semantic.trigger
    response = semantic.response
    if (
        trigger == response
        or trigger == "unknown_ap"
        or response == "unknown_ap"
    ):
        return semantic

    explanation = (
        semantic.explanation.strip()
        + " strict before/precedence 已在语义阶段标注为 precedence；"
        "完整有限轨迹公式留给后续骨架与孔位补全阶段生成。"
    ).strip()
    components = _with_component(
        semantic.components,
        "precedence_relation",
        f"{trigger} only after prior {response}",
    )
    return replace(
        semantic,
        temporal_keyword="strict before/precedence",
        scope_keyword=semantic.scope_keyword or "finite trace global",
        pattern="precedence",
        explanation=explanation,
        aps=tuple(dict.fromkeys((*semantic.aps, trigger, response))),
        components=components,
    )


def _looks_like_strict_precedence(
    semantic: SemanticStructure,
    requirement: str,
) -> bool:
    text = requirement.lower()
    has_before_wording = any(
        marker in text
        for marker in (
            "happened before",
            "happend before",
            "occurred before",
            "hold before",
            "holds before",
            "beforehand",
            "previously",
        )
    )
    has_only_if = "only" in text and "if" in text
    temporal_or_scope = f"{semantic.temporal_keyword} {semantic.scope_keyword}".lower()
    has_past_temporal_keyword = any(
        marker in temporal_or_scope
        for marker in ("before", "previous", "past once", "once")
    )
    has_precedence_pattern = "precedence" in semantic.pattern.lower()
    return (has_only_if and has_before_wording) or (
        has_precedence_pattern and (has_before_wording or has_past_temporal_keyword)
    )


def _with_component(
    components: tuple[tuple[str, str], ...],
    key: str,
    value: str,
) -> tuple[tuple[str, str], ...]:
    if any(existing_key == key for existing_key, _ in components):
        return components
    return (*components, (key, value))
