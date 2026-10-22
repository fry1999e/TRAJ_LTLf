"""Source-bound human answers shared by trajectory review stages."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from contracts.records import digest, read_json, write_json
from schemas import TraceAnnotation
from trace_formatter import trace_to_serializable


INITIAL_SCENARIO_COUNT = 3
MAX_SCENARIO_COUNT = 5
BUDGET_DEBT = "HUMAN_REVIEW_BUDGET_REACHED"
KEY_SCENARIOS_DEBT = "HUMAN_REVIEW_KEY_SCENARIOS_COMPLETE"
REVIEW_STOP_DEBTS = frozenset({BUDGET_DEBT, KEY_SCENARIOS_DEBT})
_DISPLAYED_CONTEXTS = set()
CHOICES = {
    "y": ("符合", "SAT"), "符合": ("符合", "SAT"),
    "n": ("不符合", "UNSAT"), "不符合": ("不符合", "UNSAT"),
    "u": ("无法判断", "UNCLEAR"), "无法判断": ("无法判断", "UNCLEAR"),
    "i": ("情景不可能", "INFEASIBLE"), "情景不可能": ("情景不可能", "INFEASIBLE"),
}


def trace_key(query):
    return digest(trace_to_serializable(query.trace))


def query_targets(query):
    return {*("obligation:" + item for item in query.obligation_ids),
            *("axis:" + item for item in query.target_ambiguity_axis_ids),
            "role:" + query.role, "purpose:" + query.purpose}


class HumanReviewSession:
    def __init__(self, requirement, semantic, brief, records_dir):
        ir = getattr(semantic, "faithful_ir", None)
        self.context = {
            "source_text": requirement,
            "requirement_id": getattr(ir, "requirement_id", getattr(brief, "requirement_id", "")),
            "ap_context_fingerprint": getattr(ir, "ap_context_fingerprint", ""),
            "ap_definitions": dict(brief.ap_definitions),
        }
        self.path = Path(records_dir) / f"human_trace_session_{digest(self.context)[:24]}.json"
        self.data = read_json(self.path) if self.path.exists() else {
            "schema_version": "human-trace-session-v1", "context": self.context, "answers": {},
        }
        if self.data.get("schema_version") != "human-trace-session-v1" or self.data.get("context") != self.context:
            raise ValueError("Human trace session has stale source/AP context")
        self.answers = self.data["answers"]
        for key, entry in self.answers.items():
            if digest(entry["trace"]) != key or CHOICES.get(entry["answer"]) != (entry["verdict"], entry["locked_label"]):
                raise ValueError("Stored human trace decision is inconsistent")
            annotation = entry.get("annotation")
            if annotation and (annotation.get("label_source") != "human" or annotation.get("label") != entry["locked_label"]):
                raise ValueError("Stored metadata changed a human trace decision")

    @property
    def remaining(self):
        return max(0, MAX_SCENARIO_COUNT - len(self.answers))

    @property
    def show_introduction(self):
        key = str(self.path.resolve())
        fresh = key not in _DISPLAYED_CONTEXTS
        _DISPLAYED_CONTEXTS.add(key)
        return fresh

    def save(self):
        write_json(self.path, self.data)

    def annotation(self, query):
        entry = self.answers.get(trace_key(query), {})
        raw = entry.get("annotation")
        if not raw:
            return None
        raw = dict(raw)
        for name in ("supported_text_spans", "candidate_readings", "challenged_axis_ids", "affected_parameter_paths"):
            raw[name] = tuple(raw.get(name, ()))
        return replace(TraceAnnotation(**raw), query_id=query.query_id, purpose=query.purpose)

    def annotations(self, queries):
        result = []
        for query in queries:
            annotation = self.annotation(query)
            if annotation is not None:
                result.append(annotation)
        return result

    def summary(self):
        return {"asked_count": len(self.answers), "maximum": MAX_SCENARIO_COUNT,
                "remaining": self.remaining, "session_path": str(self.path),
                **({"fallback_evidence": {"context": self.context, "answers": self.answers}}
                   if self.data.get("last_review", {}).get("metadata_status") == "deferred_for_candidate_fallback" else {}),
                **self.data.get("last_review", {})}


def review_budget_state(requirement, semantic, brief, records_dir):
    session = HumanReviewSession(requirement, semantic, brief, records_dir)
    return session.summary() if session.path.exists() else None


def semantic_review_budget(requirement, semantic, records_dir):
    from semantic_adjudication import build_trace_adjudication_brief
    return review_budget_state(requirement, semantic,
        build_trace_adjudication_brief(requirement, semantic), records_dir)


def review_should_stop(budget):
    return bool(budget and (not budget["remaining"] or
        budget.get("stop_reason") == "key_targets_covered" and budget.get("deferred_count", 0)))


def budget_limited_coverage(coverage, budget=None):
    at_limit = budget is None or not budget["remaining"]
    return replace(coverage, status="incomplete", semantic_complete=False, quota_complete=False,
        coverage_debt=tuple(dict.fromkeys((*coverage.coverage_debt, BUDGET_DEBT if at_limit else KEY_SCENARIOS_DEBT))),
        reason=("已达到人工情景审核上限" if at_limit else "关键情景已审核，停止重复检查") +
               "，未完成检查保留为覆盖债务，使用候选回退输出。")


def rank_groups(groups, session):
    """Greedily cover different review targets, preferring shorter tables."""
    remaining = list(groups)
    covered = set().union(*(set(entry.get("targets", ())) for entry in session.answers.values()
                            if entry["locked_label"] in {"SAT", "UNSAT"}))
    ordered = []
    while remaining:
        def priority(item):
            _key, queries = item
            targets = set().union(*(query_targets(query) for query in queries))
            return (len(targets - covered), -len(queries[0].trace),
                    -sum(len(step) for step in queries[0].trace))
        chosen = max(remaining, key=priority)
        remaining.remove(chosen)
        ordered.append(chosen)
        covered.update(set().union(*(query_targets(query) for query in chosen[1])))
    return ordered


def review_stop_reason(semantic, groups, session, *, branch_review):
    """Stop once an interpretation is unique or all planned targets are judged."""
    if branch_review:
        candidates = getattr(semantic, "candidate_ir_set", None)
        if candidates is not None and len(candidates.branches) > 1:
            from ir_trace_semantics import evaluate_ir_trace
            active_keys = {key for key, _queries in groups}
            answers = [entry for key, entry in session.answers.items()
                       if key in active_keys and entry["locked_label"] in {"SAT", "UNSAT"}]
            consistent = 0
            for branch in candidates.branches:
                try:
                    if all(evaluate_ir_trace(branch.faithful_ir, [set(step) for step in entry["trace"]])
                           == (entry["locked_label"] == "SAT") for entry in answers):
                        consistent += 1
                except ValueError:
                    continue
            if consistent == 1:
                return "unique_interpretation"
    elif any(query.obligation_ids for _key, queries in groups for query in queries):
        required = set().union(*(query_targets(query) for _key, queries in groups for query in queries))
        covered = set().union(*(query_targets(query) for key, queries in groups for query in queries
                                if session.answers.get(key, {}).get("locked_label") in {"SAT", "UNSAT"}))
        if required <= covered:
            return "key_targets_covered"
    return ""
