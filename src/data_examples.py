"""
Traj-LTLf 原型程序使用的示例数据。

第一版先围绕流程说明文档中的经典 response 需求展开：
    Whenever request occurs, grant should eventually occur.

也就是：
    每当 request 发生时，最终必须 grant。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ltlf_checker import Trace
from trace_formatter import format_trace_for_beginner


TRACE_PURPOSE_SEMANTIC_VALIDATION = "semantic_validation"
TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION = "ir_branch_discrimination"
TRACE_PURPOSE_OPERATOR_DISCRIMINATION = "operator_assignment_discrimination"
TRACE_EVIDENCE_PURPOSES = {
    TRACE_PURPOSE_SEMANTIC_VALIDATION,
    TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION,
    TRACE_PURPOSE_OPERATOR_DISCRIMINATION,
}


DEFAULT_REQUIREMENT = "Whenever request occurs, grant should eventually occur."
DEFAULT_RESPONSE_TRACE_COUNT = 6


@dataclass(frozen=True)
class TraceSemantics:
    """Structured semantic evidence carried by one finite trace."""

    semantic_label: str = ""
    roles: tuple[str, ...] = ()
    trigger_instances: tuple[str, ...] = ()
    obligations: tuple[str, ...] = ()
    required_horizon: int | None = None
    remaining_suffix: int | None = None
    confusion_targets: tuple[str, ...] = ()
    perturbation: str = ""


def trace_semantics_to_dict(semantics: TraceSemantics) -> dict[str, Any]:
    return {
        "semantic_label": semantics.semantic_label,
        "roles": list(semantics.roles),
        "trigger_instances": list(semantics.trigger_instances),
        "obligations": list(semantics.obligations),
        "required_horizon": semantics.required_horizon,
        "remaining_suffix": semantics.remaining_suffix,
        "confusion_targets": list(semantics.confusion_targets),
        "perturbation": semantics.perturbation,
    }


def trace_semantics_from_mapping(
    payload: Mapping[str, Any] | None,
) -> TraceSemantics:
    data = dict(payload or {})
    return TraceSemantics(
        semantic_label=str(data.get("semantic_label", "")),
        roles=tuple(str(item) for item in data.get("roles", ())),
        trigger_instances=tuple(
            str(item) for item in data.get("trigger_instances", ())
        ),
        obligations=tuple(str(item) for item in data.get("obligations", ())),
        required_horizon=data.get("required_horizon"),
        remaining_suffix=data.get("remaining_suffix"),
        confusion_targets=tuple(
            str(item) for item in data.get("confusion_targets", ())
        ),
        perturbation=str(data.get("perturbation", "")),
    )


@dataclass(frozen=True)
class LabeledTrace:
    """
    一条带人工标签的有限轨迹。

    kind:
      - "positive" 表示正轨迹，人工认为它应该满足需求
      - "negative" 表示反轨迹，人工认为它应该违反需求
      - "boundary" 表示边界轨迹，用来区分 F、XF、X 等细微语义

    expected:
      - True  表示人工标签为“满足”
      - False 表示人工标签为“不满足”
    """

    name: str
    kind: str
    trace: Trace
    expected: bool
    reason: str
    beginner_view: str = ""
    semantics: TraceSemantics = field(default_factory=TraceSemantics)
    role: str = ""
    query_role: str = ""
    target_ambiguity: str = ""
    query_constraints: tuple[str, ...] = ()
    adjudicated_label: str = ""
    confidence: str = ""
    rationale: str = ""
    label_source: str = "human"
    data_split: str = "test"
    formula_source: str = "none"
    formula_visibility: tuple[str, ...] = ()
    annotation_locked: bool = False
    allowed_rag_stages: tuple[str, ...] = (
        "skeleton",
        "completion",
        "selection",
    )
    trace_id: str = ""
    covered_ir_node_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    trace_role: str = ""
    boundary_role: str = ""
    generation_source: str = ""
    ir_version: str = ""
    target_ambiguity_axis_ids: tuple[str, ...] = ()
    nl_verdict: str = ""
    temporal_simulation_summary: str = ""
    review_record_id: str = ""
    supported_text_spans: tuple[str, ...] = ()
    evidence_purpose: str = TRACE_PURPOSE_SEMANTIC_VALIDATION
    parent_candidate_set_id: str = ""
    parent_hole_plan_version: str = ""

    def __post_init__(self) -> None:
        """
        自动补充零基础可读视图。

        例如内部轨迹 [{request}, {}, {grant}] 会显示为：
            t0: request

            t1:

            t2: grant
        """

        if self.evidence_purpose not in TRACE_EVIDENCE_PURPOSES:
            raise ValueError(
                "Unsupported labeled trace evidence purpose: "
                + self.evidence_purpose
            )
        if (
            self.evidence_purpose == TRACE_PURPOSE_IR_BRANCH_DISCRIMINATION
            and not self.parent_candidate_set_id
        ):
            raise ValueError(
                "IR branch evidence requires parent_candidate_set_id"
            )
        if (
            self.evidence_purpose == TRACE_PURPOSE_OPERATOR_DISCRIMINATION
            and not self.ir_version
        ):
            raise ValueError("operator evidence requires an IR version")
        if not self.beginner_view:
            object.__setattr__(
                self,
                "beginner_view",
                format_trace_for_beginner(self.trace),
            )
        label = self.adjudicated_label or ("SAT" if self.expected else "UNSAT")
        if label not in {"SAT", "UNSAT"}:
            raise ValueError(
                "LabeledTrace only accepts adjudicated SAT/UNSAT evidence."
            )
        if self.annotation_locked and (label == "SAT") != self.expected:
            raise ValueError(
                "LabeledTrace.expected conflicts with adjudicated_label."
            )
        if (label == "SAT") != self.expected:
            label = "SAT" if self.expected else "UNSAT"
        object.__setattr__(self, "adjudicated_label", label)


def build_response_traces(
    trigger: str,
    response: str,
    target_length: int | None = None,
    target_count: int | None = None,
) -> list[LabeledTrace]:
    """
    为 response 模式生成有限轨迹。

    response 模式指：
        whenever A occurs, B should eventually occur

    目标公式通常是：
        G(A -> F B)

    target_length 为 None 时生成默认 6 条 canonical 样例；否则每条轨迹
    都严格生成指定长度。默认集合不保留旧的 no-trigger/vacuous 样例，
    而是覆盖普通正/反、late-trigger 正/反和两条不同边界轨迹。
    """

    if target_length is not None:
        return _extend_response_trace_count(
            _build_fixed_length_response_traces(
                trigger=trigger,
                response=response,
                target_length=target_length,
            ),
            trigger=trigger,
            response=response,
            target_length=target_length,
            target_count=target_count,
        )

    traces = [
        LabeledTrace(
            name=f"pos_{trigger}_then_{response}",
            kind="positive",
            trace=[{trigger}, set(), {response}],
            expected=True,
            reason=f"{trigger} 出现后，后续有限轨迹中出现了 {response}。",
        ),
        LabeledTrace(
            name=f"pos_late_{trigger}_then_{response}",
            kind="positive",
            trace=[set(), {trigger}, {response}],
            expected=True,
            reason=f"{trigger} 不是在 t0 出现，仍然应该被全局约束检查到。",
        ),
        LabeledTrace(
            name=f"neg_{trigger}_without_{response}",
            kind="negative",
            trace=[{trigger}, set(), set()],
            expected=False,
            reason=f"{trigger} 出现后，直到有限轨迹结束都没有 {response}。",
        ),
        LabeledTrace(
            name=f"neg_late_{trigger}_without_{response}",
            kind="negative",
            trace=[set(), {trigger}, set()],
            expected=False,
            reason=(
                f"后半段出现 {trigger}，但之后仍然没有 {response}；"
                "用于区分是否需要外层 G。"
            ),
        ),
        LabeledTrace(
            name=f"bound_{trigger}_and_{response}_same_time",
            kind="boundary",
            trace=[{trigger, response}],
            expected=True,
            reason=f"F 允许当前时刻成立，因此 {trigger} 和 {response} 同时发生应视为满足。",
        ),
        LabeledTrace(
            name=f"bound_late_{trigger}_and_{response}_same_time",
            kind="boundary",
            trace=[set(), set(), {trigger, response}],
            expected=True,
            reason=(
                f"后段 {trigger} 与 {response} 同时发生；用于区分 F 与 X F "
                "以及验证边界轨迹不依赖单一位置。"
            ),
        ),
    ]
    return _extend_response_trace_count(
        traces,
        trigger=trigger,
        response=response,
        target_length=target_length,
        target_count=target_count,
    )


def _build_fixed_length_response_traces(
    trigger: str,
    response: str,
    target_length: int,
) -> list[LabeledTrace]:
    if target_length < 2:
        raise ValueError("response late-trigger 轨迹长度必须至少为 2。")

    first_response_index = target_length - 1
    late_trigger_index = 1
    late_negative_trigger_index = max(1, target_length - 2)

    return [
        LabeledTrace(
            name=f"pos_{trigger}_then_{response}",
            kind="positive",
            trace=_trace_with_events(
                target_length,
                ((0, trigger), (first_response_index, response)),
            ),
            expected=True,
            reason=(
                f"长度为 {target_length}；{trigger} 出现后，"
                f"有限轨迹中出现了 {response}。"
            ),
        ),
        LabeledTrace(
            name=f"pos_late_{trigger}_then_{response}",
            kind="positive",
            trace=_trace_with_events(
                target_length,
                ((late_trigger_index, trigger), (first_response_index, response)),
            ),
            expected=True,
            reason=(
                f"长度为 {target_length}；{trigger} 不只在首个时刻被检查，"
                f"其后仍出现了 {response}。"
            ),
        ),
        LabeledTrace(
            name=f"neg_{trigger}_without_{response}",
            kind="negative",
            trace=_trace_with_events(target_length, ((0, trigger),)),
            expected=False,
            reason=(
                f"长度为 {target_length}；{trigger} 出现后，"
                f"直到有限轨迹结束都没有 {response}。"
            ),
        ),
        LabeledTrace(
            name=f"neg_late_{trigger}_without_{response}",
            kind="negative",
            trace=_trace_with_events(
                target_length,
                ((late_negative_trigger_index, trigger),),
            ),
            expected=False,
            reason=(
                f"长度为 {target_length}；后段出现 {trigger}，"
                f"之后仍然没有 {response}；用于区分是否需要外层 G。"
            ),
        ),
        LabeledTrace(
            name=f"bound_{trigger}_and_{response}_same_time",
            kind="boundary",
            trace=_trace_with_events(
                target_length,
                ((0, trigger), (0, response)),
            ),
            expected=True,
            reason=(
                f"长度为 {target_length}；F 允许当前时刻成立，因此 "
                f"{trigger} 和 {response} 同时发生应视为满足。"
            ),
        ),
        LabeledTrace(
            name=f"bound_late_{trigger}_and_{response}_same_time",
            kind="boundary",
            trace=_trace_with_events(
                target_length,
                ((target_length - 1, trigger), (target_length - 1, response)),
            ),
            expected=True,
            reason=(
                f"长度为 {target_length}；在最后位置 {trigger} 与 {response} "
                "同时发生，用来区分 F 与 X F，并提供第二条不同边界轨迹。"
            ),
        ),
    ]


def _extend_response_trace_count(
    traces: list[LabeledTrace],
    *,
    trigger: str,
    response: str,
    target_length: int | None,
    target_count: int | None,
) -> list[LabeledTrace]:
    if target_count is None or target_count <= len(traces):
        return traces

    length = target_length or 3
    supplemental = [
        LabeledTrace(
            name=f"neg_{response}_before_late_{trigger}",
            kind="negative",
            trace=_trace_with_events(
                length,
                ((0, response), (min(1, length - 1), trigger)),
            ),
            expected=False,
            reason=(
                f"{response} 只在 {trigger} 之前出现，不能满足 "
                f"{trigger} 之后最终出现 {response} 的义务。"
            ),
        ),
        LabeledTrace(
            name=f"pos_repeated_{trigger}_then_{response}",
            kind="positive",
            trace=_trace_with_events(
                length,
                ((0, trigger), (min(1, length - 1), trigger), (length - 1, response)),
            ),
            expected=True,
            reason=(
                f"重复出现 {trigger} 时，后续 {response} 同时满足这些响应义务。"
            ),
        ),
        LabeledTrace(
            name=f"neg_repeated_{trigger}_without_{response}",
            kind="negative",
            trace=_trace_with_events(
                length,
                ((0, trigger), (min(1, length - 1), trigger)),
            ),
            expected=False,
            reason=(
                f"重复出现 {trigger} 后始终没有 {response}，应拒绝过弱候选。"
            ),
        ),
        LabeledTrace(
            name=f"bound_late_{trigger}_and_{response}_same_time",
            kind="boundary",
            trace=_trace_with_events(
                length,
                ((length - 1, trigger), (length - 1, response)),
            ),
            expected=True,
            reason=(
                f"在最后位置 {trigger} 与 {response} 同时发生，用来区分 F 与 X F。"
            ),
        ),
    ]

    extended = list(traces)
    seen_names = {trace.name for trace in extended}
    for trace in supplemental:
        if len(extended) >= target_count:
            break
        if trace.name in seen_names:
            continue
        seen_names.add(trace.name)
        extended.append(trace)
    return extended


def _empty_trace(target_length: int) -> Trace:
    return [set() for _ in range(target_length)]


def _trace_with_events(
    target_length: int,
    events: tuple[tuple[int, str], ...],
) -> Trace:
    trace = _empty_trace(target_length)
    for index, atom in events:
        trace[index].add(atom)
    return trace
