"""Simple human decisions with model-generated technical review metadata."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from llm.errors import LLMClientError
from llm.schema_validation import validate_response_schema
from schemas import TraceAnnotation
from trace_formatter import format_trace_truth_table, trace_to_serializable
from human_review_session import (
    CHOICES, INITIAL_SCENARIO_COUNT, MAX_SCENARIO_COUNT, HumanReviewSession,
    rank_groups, review_stop_reason, trace_key, query_targets,
)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(item) for item in value)
    return value


def review_semantic_with_model(requirement, semantic, client):
    """Check generated structure without asking a nonexpert to inspect an IR."""
    schema = {
        "type": "object", "properties": {
            "approved": {"type": "boolean"},
            "reason": {"type": "string", "minLength": 1},
        }, "required": ["approved", "reason"], "additionalProperties": False,
    }
    from semantic_review_view import semantic_review_payload
    raw = client.complete_json(
        "Check whether the proposed semantic structure faithfully represents the original "
        "requirement and the human-confirmed AP meanings. Check negation, direction, "
        "Boolean and temporal scope. Preserve genuinely unresolved alternatives for "
        "human trajectory judgments; do not invent domain assumptions or human approval. "
        "Read the complete graphs from each branch's graph reference, starting at its root. "
        "The primary graph and candidate graphs are the source of semantic structure. "
        "Pending branch selection is expected: assess whether the candidates preserve "
        "source-supported readings, not whether a human has already selected a branch. "
        "Do not reject solely because selection is unresolved; reject concrete source "
        "contradictions or missing meaning. A pending choice is never approval of a branch. "
        "Return approved and a concrete reason. This is model assistance, not an "
        "independent human review. Keep the reason to one to three sentences.",
        _json_value(semantic_review_payload(requirement, semantic)),
        temperature=0.0, response_schema=schema,
    )
    validate_response_schema(raw, schema)
    return raw


def review_traces_with_simple_choices(
    *, requirement, semantic, queries, round_index, records_dir,
    llm_client, adjudication_brief=None, input_fn=None,
):
    """Keep human verdicts authoritative; ask the model for technical fields."""
    if not queries:
        return []
    if llm_client is None:
        raise LLMClientError("简化轨迹审核需要主生成模型来整理标签、置信度和理由。")
    ids = [query.query_id for query in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate trace query ids")
    if adjudication_brief is None:
        from semantic_adjudication import build_trace_adjudication_brief
        adjudication_brief = build_trace_adjudication_brief(requirement, semantic)
    read = input_fn or input
    definitions = dict(adjudication_brief.ap_definitions)
    session = HumanReviewSession(requirement, semantic, adjudication_brief, records_dir)
    grouped = {}
    for query in queries:
        grouped.setdefault(trace_key(query), []).append(query)
    groups = list(grouped.items())
    fresh = rank_groups([(key, group) for key, group in groups if key not in session.answers], session)
    branch_review = any(query.purpose == "ir_branch_discrimination" for query in queries)
    phase = "需求含义确认" if branch_review else "关键情景核对"
    ap_columns = tuple(dict.fromkeys((*definitions, *sorted({name for query in queries for step in query.trace for name in step}))))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(records_dir) / f"nonexpert_trace_review_{round_index}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": "nonexpert_trace_review", "status": "awaiting_human",
        "source_text": requirement,
        "queries": [{**asdict(q), "trace": trace_to_serializable(q.trace)} for q in queries],
        "human_choices": [], "metadata_source": "model_assistant",
        "metadata_model": str(getattr(llm_client, "model", type(llm_client).__name__)),
        "ap_definitions": definitions,
        "trace_display": {"format": "truth_table_v1", "time_origin": 0, "ap_columns": list(ap_columns)},
        "review_phase": phase,
        "review_context": session.context,
        "query_groups": {key: [query.query_id for query in group] for key, group in groups},
        "reused_choices": [{"query_id": query.query_id, "source_record": session.answers[key]["source_record"],
                            "source_query_id": session.answers[key]["source_query_id"]}
                           for key, group in groups if key in session.answers for query in group],
    }

    def save():
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    if session.show_introduction:
        print("\n请判断表格中的情景是否符合原始需求。")
        print(f"原始需求：{requirement}")
        print("表格中 true 表示该时刻子命题成立，false 表示不成立；最后一行之后情景结束。")
        print(f"每组先看最多 {INITIAL_SCENARIO_COUNT} 个关键情景，仅在需要时补充；每条需求累计最多 {MAX_SCENARIO_COUNT} 个不同情景。")
        if definitions:
            print("子命题含义：")
            for name, meaning in definitions.items():
                print(f"  {name}：{meaning}")
    print(f"\n{phase}（已判断 {len(session.answers)} 个不同情景）：")
    if record["reused_choices"]:
        print("已复用相同情景的判断，无需重复回答。")

    def ask(key, group, position, count, *, supplemental=False):
        query = group[0]
        record["current_query_id"] = query.query_id
        save()
        heading = "补充情景" if supplemental else "关键情景"
        print(f"\n{heading} {position}/{count}（累计 {len(session.answers) + 1}/{MAX_SCENARIO_COUNT}）：")
        print(format_trace_truth_table(query.trace, ap_columns))
        while True:
            answer = read("请选择：y 符合 / n 不符合 / u 无法判断 / i 情景不可能：").strip().casefold()
            if answer in CHOICES:
                verdict, label = CHOICES[answer]
                choice = {
                    "query_id": query.query_id, "answer": answer,
                    "verdict": verdict, "locked_label": label,
                }
                record["human_choices"].append(choice)
                save()
                session.answers[key] = {**choice, "trace": trace_to_serializable(query.trace),
                    "targets": sorted(set().union(*(query_targets(item) for item in group))),
                    "source_record": str(path), "source_query_id": query.query_id}
                session.save()
                break
            print("请输入 y、n、u 或 i；空输入不会确认。")

    stop = review_stop_reason(semantic, groups, session, branch_review=branch_review)
    primary_count = min(INITIAL_SCENARIO_COUNT, session.remaining, len(fresh))
    for index in range(primary_count):
        if stop:
            break
        key, group = fresh.pop(0)
        ask(key, group, index + 1, primary_count)
        stop = review_stop_reason(semantic, groups, session, branch_review=branch_review)
    fresh = rank_groups(fresh, session)
    supplement_count = min(session.remaining, len(fresh)) if not stop else 0
    if supplement_count:
        print("现有判断尚未覆盖关键差异，再补充少量不同情景。")
    for index in range(supplement_count):
        if stop:
            break
        key, group = fresh.pop(0)
        ask(key, group, index + 1, supplement_count, supplemental=True)
        stop = review_stop_reason(semantic, groups, session, branch_review=branch_review)
    deferred = [query.query_id for key, group in groups if key not in session.answers for query in group]
    record.pop("current_query_id", None)
    record["deferred_query_ids"] = deferred
    session.data["last_review"] = {"deferred_count": len(deferred),
        "stop_reason": stop or ("budget_exhausted" if deferred and not session.remaining else "no_new_queries")}
    session.save()
    record["review_budget"] = {**session.summary(), "new_question_count": len(record["human_choices"])}
    if deferred and not session.remaining:
        print("已达到本条需求的情景审核上限；保留已完成判断，未完成检查交由候选回退处理。")
        if stop != "unique_interpretation":
            # Raw votes may rank a provisional formula, but cannot pass gates
            # without the still-missing model annotation fields.
            session.data["last_review"]["metadata_status"] = "deferred_for_candidate_fallback"
            session.save()
            annotations = session.annotations(queries)
            record.update(status="partial", metadata_status="deferred_for_candidate_fallback",
                          review_budget={**session.summary(), "new_question_count": len(record["human_choices"])},
                          annotations=[asdict(item) for item in annotations])
            save()
            return annotations
    representatives = [(key, group[0]) for key, group in groups
                       if key in session.answers and not session.answers[key].get("annotation")]
    ids = [query.query_id for _key, query in representatives]
    locked = {query.query_id: session.answers[key]["locked_label"] for key, query in representatives}
    record["status"] = "awaiting_model_metadata"
    save()
    if not representatives:
        annotations = session.annotations(queries)
        record.update(status="partial" if deferred else "complete", annotations=[asdict(item) for item in annotations])
        save()
        return annotations
    schema = {
        "type": "object", "properties": {"annotations": {
            "type": "array", "minItems": len(representatives), "maxItems": len(representatives),
            "items": {"type": "object", "properties": {
                "query_id": {"type": "string", "enum": ids},
                "label": {"type": "string", "enum": ["SAT", "UNSAT", "UNCLEAR", "INFEASIBLE"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "rationale": {"type": "string", "minLength": 1},
                "temporal_simulation_summary": {"type": "string", "minLength": 1},
            }, "required": ["query_id", "label", "confidence", "rationale", "temporal_simulation_summary"],
            "additionalProperties": False},
        }}, "required": ["annotations"], "additionalProperties": False,
    }
    try:
        raw = llm_client.complete_json(
            "Translate the user's simple trajectory decisions into internal labels and "
            "provide confidence, a short reason and temporal simulation in Chinese. "
            "The user's decisions are final: copy locked_label exactly for each query. "
            "Do not reverse a verdict, invent a user explanation or claim the user "
            "provided confidence or rationale. Reasons are your interpretation. "
            "Keep each rationale and simulation to one short sentence. "
            "Use only the original requirement, AP meanings and raw finite trace; "
            "no candidate formulas or hidden semantic structures are supplied.",
            {"task": "explain_human_trace_choices", "source_text": requirement,
             "ap_definitions": definitions,
             "human_choices": [{"query_id": query.query_id, "locked_label": locked[query.query_id],
                                "answer": session.answers[key]["answer"], "verdict": session.answers[key]["verdict"]}
                               for key, query in representatives],
             "queries": [{"query_id": q.query_id, "trace": trace_to_serializable(q.trace)} for _key, q in representatives]},
            temperature=0.0, response_schema=schema,
            max_tokens=max(1200, 512 * len(representatives) + 256),
        )
        validate_response_schema(raw, schema)
        items = raw["annotations"]
        if len({item["query_id"] for item in items}) != len(representatives):
            raise LLMClientError("模型返回了重复或缺失的轨迹编号；人工选择已保存。")
        if any(item["label"] != locked[item["query_id"]] for item in items):
            raise LLMClientError("模型试图改变人工轨迹判断，已拒绝；人工选择已保存。")
        by_id = {item["query_id"]: item for item in items}
        new_annotations = [TraceAnnotation(
            **by_id[q.query_id], label_source="human", annotator="human-user",
            model=record["metadata_model"], prompt_version="nonexpert-trace-assistance-v1",
            uncertainty_kind=("annotation_uncertainty" if locked[q.query_id] == "UNCLEAR"
                              else "query_infeasible" if locked[q.query_id] == "INFEASIBLE" else "none"),
            purpose=q.purpose,
        ) for _key, q in representatives]
        for (key, _query), annotation in zip(representatives, new_annotations):
            session.answers[key]["annotation"] = asdict(annotation)
        session.save()
        annotations = session.annotations(queries)
    except Exception as exc:
        record.update(status="model_metadata_failed", error_type=type(exc).__name__)
        save()
        raise
    record.update(status="partial" if deferred else "complete", model_response=raw,
                  annotations=[asdict(annotation) for annotation in annotations])
    save()
    return annotations
