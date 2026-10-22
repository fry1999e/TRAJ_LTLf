"""
候选公式轨迹一致性 Agent。

该 Agent 不调度 ltlf_checker.check_trace。它直接按 LTLf 有限轨迹语义模拟
候选公式在人工审核轨迹上的求值过程，并由模拟结论聚合 PosSat、NegRej 和
BoundMatch。
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from data_examples import LabeledTrace
from ltlf_parser import Formula
from schemas import (
    CompletionDecision,
    FormulaConsistencyEvaluation,
    TraceSimulationResult,
    TraceSimulationStep,
)
from utils import format_trace, format_truth


class TrajectoryConsistencyAgent(BaseAgent):
    """通过有限轨迹语义模拟候选 LTLf 公式的一致性检查 Agent。"""

    def __init__(self, retriever):
        super().__init__(retriever=retriever, name="TrajectoryConsistencyAgent")

    def evaluate_candidates(
        self,
        candidates: list[Formula],
        traces: list[LabeledTrace],
        strict_boundary: bool = True,
    ) -> list[FormulaConsistencyEvaluation]:
        """模拟并评估一组候选公式。"""

        del strict_boundary
        return [
            self.evaluate_formula(
                candidate,
                traces,
                candidate_id=f"candidate-{index:03d}",
            )
            for index, candidate in enumerate(candidates, start=1)
        ]

    def evaluate_formula(
        self,
        formula: Formula,
        traces: list[LabeledTrace],
        w_pos: float = 0.3,
        w_neg: float = 0.3,
        w_bound: float = 0.4,
        size_penalty: float = 0.01,
        candidate_id: str = "",
    ) -> FormulaConsistencyEvaluation:
        """模拟一个公式在所有轨迹上的表现，并计算一致性指标。"""

        trace_results = tuple(
            self.simulate_trace(formula=formula, labeled_trace=labeled_trace)
            for labeled_trace in traces
        )

        positives = [item for item in trace_results if item.expected]
        negatives = [item for item in trace_results if not item.expected]
        boundaries = [item for item in trace_results if item.trace_kind == "boundary"]

        pos_sat = _safe_average(item.actual for item in positives)
        neg_rej = _safe_average(not item.actual for item in negatives)
        bound_match = _safe_average(item.matched for item in boundaries)
        score = (
            w_pos * pos_sat
            + w_neg * neg_rej
            + w_bound * bound_match
            - size_penalty * formula.size()
        )

        return FormulaConsistencyEvaluation(
            formula=formula,
            pos_sat=pos_sat,
            neg_rej=neg_rej,
            bound_match=bound_match,
            score=score,
            trace_results=trace_results,
            failure_summary=tuple(
                self._failure_summary(trace_results=trace_results)
            ),
            candidate_id=candidate_id,
            boundary_metric_available=bool(boundaries),
            boundary_trace_count=len(boundaries),
        )

    def evaluate_completion_decision(
        self,
        formula: Formula,
        assignments: tuple[tuple[str, str], ...],
        traces: list[LabeledTrace],
        candidate_id: str = "",
        template_text: str = "",
        operator_derivation: tuple[str, ...] = (),
    ) -> CompletionDecision:
        """
        为补全阶段生成统一的 CompletionDecision。

        CandidateCompletionAgent 只负责生成候选与孔位赋值，评分和证据统一由
        本 Agent 的有限轨迹模拟结果提供。
        """

        evaluation = self.evaluate_formula(formula=formula, traces=traces)
        return CompletionDecision(
            formula=formula,
            assignments=assignments,
            boundary_match=evaluation.bound_match,
            pos_sat=evaluation.pos_sat,
            neg_rej=evaluation.neg_rej,
            score=evaluation.score,
            operator_derivation=operator_derivation,
            trace_derivation=tuple(
                _format_simulation_derivation(list(evaluation.trace_results))
            ),
            reason=_completion_reason(
                assignments=assignments,
                boundary_match=evaluation.bound_match,
                pos_sat=evaluation.pos_sat,
                neg_rej=evaluation.neg_rej,
            ),
            candidate_id=candidate_id,
            template_text=template_text,
            boundary_metric_available=evaluation.boundary_metric_available,
            boundary_trace_count=evaluation.boundary_trace_count,
        )

    def simulate_trace(
        self,
        formula: Formula,
        labeled_trace: LabeledTrace,
    ) -> TraceSimulationResult:
        """模拟一个候选公式在一条有限轨迹上的求值过程。"""

        if not labeled_trace.trace:
            raise ValueError("本演示程序暂不处理空轨迹，请至少提供一个时间步。")

        steps: list[TraceSimulationStep] = []
        actual = self._simulate(
            trace=labeled_trace.trace,
            formula=formula,
            position=0,
            steps=steps,
        )
        matched = actual == labeled_trace.expected
        failure_reason = ""
        if not matched:
            failure_reason = (
                f"{labeled_trace.name}: {format_trace(labeled_trace.trace)} "
                f"模拟={format_truth(actual)}，人工标签="
                f"{format_truth(labeled_trace.expected)}。"
            )

        return TraceSimulationResult(
            formula=formula,
            trace_name=labeled_trace.name,
            trace_kind=labeled_trace.kind,
            expected=labeled_trace.expected,
            actual=actual,
            matched=matched,
            steps=tuple(steps),
            failure_reason=failure_reason,
        )

    def _simulate(
        self,
        trace: list[set[str]],
        formula: Formula,
        position: int,
        steps: list[TraceSimulationStep],
    ) -> bool:
        op = formula.op
        formula_text = str(formula)

        if op == "true":
            return self._record(steps, position, formula_text, True, "true 恒为满足。")
        if op == "false":
            return self._record(steps, position, formula_text, False, "false 恒为不满足。")

        if position >= len(trace):
            return self._record(
                steps,
                position,
                formula_text,
                False,
                "当前位置已经超过有限轨迹长度，公式不满足。",
            )

        if op == "ap":
            result = formula.name in trace[position]
            event_text = _event_text(trace[position])
            return self._record(
                steps,
                position,
                formula_text,
                result,
                f"t{position} 的事件集合为 {event_text}，"
                f"{formula.name} {'出现' if result else '未出现'}。",
            )

        if op == "not":
            child = formula.args[0]
            child_result = self._simulate(trace, child, position, steps)
            return self._record(
                steps,
                position,
                formula_text,
                not child_result,
                f"子公式 {child} 的模拟结果为 {format_truth(child_result)}，"
                "取反后得到当前结果。",
            )

        if op == "and":
            left, right = formula.args
            left_result = self._simulate(trace, left, position, steps)
            right_result = self._simulate(trace, right, position, steps)
            result = left_result and right_result
            return self._record(
                steps,
                position,
                formula_text,
                result,
                "合取要求左右子公式同时满足。",
            )

        if op == "or":
            left, right = formula.args
            left_result = self._simulate(trace, left, position, steps)
            right_result = self._simulate(trace, right, position, steps)
            result = left_result or right_result
            return self._record(
                steps,
                position,
                formula_text,
                result,
                "析取要求至少一个子公式满足。",
            )

        if op == "implies":
            left, right = formula.args
            left_result = self._simulate(trace, left, position, steps)
            right_result = self._simulate(trace, right, position, steps)
            result = (not left_result) or right_result
            return self._record(
                steps,
                position,
                formula_text,
                result,
                "蕴含 A -> B 在 A 不满足或 B 满足时成立。",
            )

        if op == "X":
            child = formula.args[0]
            next_position = position + 1
            if next_position >= len(trace):
                return self._record(
                    steps,
                    position,
                    formula_text,
                    False,
                    "强 next 需要真实存在下一步；当前已经没有下一步。",
                )
            child_result = self._simulate(trace, child, next_position, steps)
            return self._record(
                steps,
                position,
                formula_text,
                child_result,
                f"强 next 转到 t{next_position} 模拟子公式 {child}。",
            )

        if op == "F":
            child = formula.args[0]
            witnesses: list[int] = []
            for next_position in range(position, len(trace)):
                if self._simulate(trace, child, next_position, steps):
                    witnesses.append(next_position)
                    break
            result = bool(witnesses)
            detail = (
                f"F 从 t{position} 到轨迹结束寻找 {child}；"
                f"在 t{witnesses[0]} 找到满足位置。"
                if witnesses
                else f"F 从 t{position} 到轨迹结束没有找到满足 {child} 的位置。"
            )
            return self._record(steps, position, formula_text, result, detail)

        if op == "G":
            child = formula.args[0]
            failed_at: int | None = None
            for next_position in range(position, len(trace)):
                if not self._simulate(trace, child, next_position, steps):
                    failed_at = next_position
                    break
            result = failed_at is None
            detail = (
                f"G 要求从 t{position} 到轨迹结束每个位置都满足 {child}。"
                if result
                else f"G 在 t{failed_at} 发现子公式 {child} 不满足。"
            )
            return self._record(steps, position, formula_text, result, detail)

        if op == "U":
            left, right = formula.args
            witness: int | None = None
            for target_position in range(position, len(trace)):
                right_holds = self._simulate(trace, right, target_position, steps)
                left_holds_before = all(
                    self._simulate(trace, left, before_position, steps)
                    for before_position in range(position, target_position)
                )
                if right_holds and left_holds_before:
                    witness = target_position
                    break
            result = witness is not None
            detail = (
                f"U 在 t{witness} 找到右子公式 {right}，且此前左子公式 {left} 保持。"
                if witness is not None
                else f"U 没有找到满足右子公式 {right} 且此前 {left} 保持的位置。"
            )
            return self._record(steps, position, formula_text, result, detail)

        raise ValueError(f"未知公式节点类型: {op}")

    def _record(
        self,
        steps: list[TraceSimulationStep],
        position: int,
        formula_fragment: str,
        result: bool,
        explanation: str,
    ) -> bool:
        steps.append(
            TraceSimulationStep(
                time_index=position,
                formula_fragment=formula_fragment,
                result=result,
                explanation=explanation,
            )
        )
        return result

    def _failure_summary(
        self,
        trace_results: tuple[TraceSimulationResult, ...],
    ) -> list[str]:
        return [
            result.failure_reason
            for result in trace_results
            if result.failure_reason
        ]


def _safe_average(values: object) -> float:
    value_list = list(values)
    if not value_list:
        return 1.0
    return sum(1 for value in value_list if value) / len(value_list)


def _format_simulation_derivation(
    trace_results: list[TraceSimulationResult],
) -> list[str]:
    derivation: list[str] = []
    for result in trace_results:
        derivation.append(
            f"{result.trace_kind} {result.trace_name}: "
            f"模拟={format_truth(result.actual)} "
            f"标签={format_truth(result.expected)} "
            f"匹配={'是' if result.matched else '否'}"
        )
    return derivation


def _completion_reason(
    assignments: tuple[tuple[str, str], ...],
    boundary_match: float,
    pos_sat: float,
    neg_rej: float,
) -> str:
    assignment_text = ", ".join(f"{hole}={op}" for hole, op in assignments)
    return (
        f"补全选择 {assignment_text}；TrajectoryConsistencyAgent 模拟得到"
        f"边界轨迹匹配率 {boundary_match:.2f}，正轨迹满足率 {pos_sat:.2f}，"
        f"反轨迹拒绝率 {neg_rej:.2f}。"
    )


def _event_text(events: set[str]) -> str:
    if not events:
        return "{}"
    return "{" + ", ".join(sorted(events)) + "}"
