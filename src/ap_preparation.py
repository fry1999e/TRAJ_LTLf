"""Source-bound AP proposals and explicit human approval before IR generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re

from ap_binding_context import (
    APBindingContext, APBindingContextError, bind_proposal_ap_mentions,
    build_ap_binding_context,
)
from ap_identity import (
    APIdentityValidationError, AP_INVENTORY_UNAPPROVED,
    approve_ap_inventory, propose_grounded_ap_inventory,
)
from ap_proposal_validation import (
    APProposalValidationError, ap_repair_feedback, deduplicate_identical_proposals,
    same_ap_proposal, validate_source_ap_proposals,
)
from execution_observer import request_stage
from llm.errors import LLMResponseSchemaError
from llm.schema_validation import validate_response_schema
from schemas import APInventory


AP_PROPOSAL_SCHEMA = {
    "type": "object", "required": ["aps"], "additionalProperties": False,
    "properties": {"aps": {
        "type": "array", "minItems": 1, "maxItems": 128,
        "items": {
            "type": "object", "required": ["canonical_name", "evidence_texts"],
            "additionalProperties": False,
            "properties": {
                "canonical_name": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
                "evidence_texts": {"type": "array", "minItems": 1,
                                   "items": {"type": "string", "minLength": 1}},
            },
        },
    }},
}
AP_PROPOSAL_PROMPT = (
    "Extract atomic event/state predicates from the original natural-language requirement. "
    "Return only {aps:[{canonical_name:string,evidence_texts:[string]}]}. "
    "Use parser-safe names and exact, case-sensitive source quotes. Keep full predicate "
    "meaning, negation and qualifiers. For a discontinuous predicate quote its actual "
    "fragments separately. Extract independently testable predicates joined by and/or "
    "as separate APs, leaving their Boolean connective in the semantic structure. "
    "For symbolic predicates, 'p holds' and 'p and q hold' use two APs named p and q, "
    "with evidence_texts ['p'] and ['q']. Do not create p_and_q_hold or assign p "
    "to a second AP. Evidence fragments of one AP must denote the same predicate, "
    "not different predicates joined by a Boolean connective. "
    "These rules apply to domain phrases and all languages, not only symbols. "
    "Different predicates sharing a subject need their own predicate-specific quotes. "
    "Do not split a compound entity name solely because its name contains and/or. "
    "Reuse one canonical name when the same predicate recurs in a requirement; "
    "never invent a contiguous paraphrase. Reading segments are "
    "only presentation boundaries: predicates may cross several segments. Connectives, "
    "modal words and temporal operators alone are not APs. Do not include modal/temporal "
    "scaffolding inside AP quotes: for 'grant should eventually occur', quote 'grant' "
    "and optionally 'occur' as separate fragments, never the whole temporal clause. "
    "Do not output formulas, "
    "semantic graphs, truth labels, approval decisions, or reference answers. "
    "All returned predicates are proposals awaiting human review."
)
AP_REPAIR_PROMPT = AP_PROPOSAL_PROMPT + (
    " This is a constrained repair of rejected AP proposals. Address every issue in "
    "ap_repair_feedback, including all overlapping source occurrences. Preserve the "
    "listed non-conflicting candidates and their qualifiers. A repeated entity word "
    "does not imply repeated predicate meaning; use context-specific quotes for "
    "distinct events. Never solve a conflict by deleting an event or weakening a "
    "condition. Return the complete repaired proposal list for human confirmation."
)


class APPreparationError(APBindingContextError):
    """Recoverable AP input/review failure, rather than an IR alias exception."""


@dataclass(frozen=True)
class PreparedAPContext:
    context: APBindingContext
    inventory: APInventory | None = None
    proposal_records: tuple[dict, ...] = ()


def source_symbol_ap_proposals(source):
    """Propose explicit symbol predicates only when no other content is left.

    This extracts an alphabet, never a Boolean/temporal reading or approval.
    Ordinary words, articles, negation and unrecognized clauses fall back to
    the model instead of being silently discarded.
    """
    symbol = r"(?:prop_\d+|[a-z])"
    predicate = re.compile(
        rf"(?<![A-Za-z0-9_'’])(?P<subjects>{symbol}(?:\s+(?:and|or)\s+{symbol})*)"
        r"\s+(?:holds?|occurs?)(?![A-Za-z0-9_])"
    )
    matches = list(predicate.finditer(source))
    if not matches:
        return None
    names, symbol_spans = [], set()
    for match in matches:
        for token in re.finditer(rf"\b{symbol}\b", match["subjects"]):
            name = token.group()
            if name not in names:
                names.append(name)
            start = match.start("subjects") + token.start()
            symbol_spans.add((start, start + len(name)))
    # A symbol elsewhere might be an article or an unrecognized predicate.
    for token in re.finditer(rf"(?<![A-Za-z0-9_'’]){symbol}(?![A-Za-z0-9_'’])", source):
        if token.span() not in symbol_spans:
            return None
    remaining = source
    for match in reversed(matches):
        remaining = remaining[:match.start()] + " " + remaining[match.end():]
    scaffolding = set("globally it is the case that if then eventually always whenever when while until next finally and or".split())
    if re.sub(r"[\w\s.,;:!?()]+", "", remaining, flags=re.UNICODE):
        return None
    if any(word.casefold() not in scaffolding for word in re.findall(r"\w+", remaining)):
        return None
    return {"aps": [{"canonical_name": name, "evidence_texts": [name]} for name in names]}


def _ascii_identifier_character(char):
    return char.isascii() and (char.isalnum() or char == "_")


def _pending_inventory(context, raw, segmentation, *, proposal_source="generation_model_ap_proposal"):
    validate_response_schema(raw, AP_PROPOSAL_SCHEMA)
    source = context.grounding_contract.source_text
    validate_source_ap_proposals(source, raw)
    inventory = propose_grounded_ap_inventory(
        requirement_id=context.requirement_id, source_text=source,
        profile=context.representation_profile, proposals=raw["aps"],
        segmentation=segmentation, proposal_source=proposal_source,
    )
    if set(inventory.issue_codes) - {AP_INVENTORY_UNAPPROVED}:
        raise ValueError("AP evidence is incomplete or invalid: " + ", ".join(inventory.issue_codes))
    # Validate every occurrence, including differently cased matches found by
    # the inventory loader. Actual evidence always comes from the source.
    proposals = []
    identities = {item.ap_identity_id: item for item in inventory.identities}
    owners = []
    for mention in inventory.mentions:
        name = identities[mention.ap_identity_id].canonical_name
        span = mention.source_span
        start, end = span
        if ((start and _ascii_identifier_character(source[start - 1]) and _ascii_identifier_character(source[start]))
                or (end < len(source) and _ascii_identifier_character(source[end - 1]) and _ascii_identifier_character(source[end]))):
            raise ValueError("AP evidence must not match part of another identifier")
        for previous, owner in owners:
            if span[0] < previous[1] and previous[0] < span[1] and owner != name:
                raise ValueError(
                    f"Overlapping source evidence belongs to different APs: {owner!r} and {name!r} "
                    f"at {source[max(span[0], previous[0]):min(span[1], previous[1])]!r}"
                )
        owners.append((span, name))
        proposals.append({"canonical_name": name, "evidence_text": mention.evidence_text,
                          "source_span": list(span),
                          "supporting_segment_ids": mention.supporting_segment_ids})
    candidate = bind_proposal_ap_mentions(context, proposals)
    candidate.validate_source(source)
    return PreparedAPContext(candidate, inventory)


def ensure_ap_candidates(context, client, *, segmentation=None, recovery_feedback=None):
    """Fill only an empty proposal context; never replace an existing authority."""
    if context.grounding_contract is None:
        raise APPreparationError("AP_GROUNDING_REQUIRED", "缺少 AP 来源上下文，请重新提交需求。")
    context.validate_source(context.grounding_contract.source_text)
    if segmentation is not None and (
        segmentation.requirement_id != context.requirement_id
        or segmentation.source_text != context.grounding_contract.source_text
    ):
        raise APPreparationError("AP_SOURCE_STALE", "切分与 AP 来源不一致，请使用同一条需求重新审核。")
    if context.grounding_contract.identities or context.has_authoritative_catalog:
        return PreparedAPContext(context)
    if context.grounding_authority != "proposal_only":
        raise APPreparationError("AP_GROUNDING_REVIEW_REQUIRED", "AP 定义存在冲突或缺失，需要人工补充。")
    symbolic = source_symbol_ap_proposals(context.grounding_contract.source_text)
    if symbolic is not None:
        return _pending_inventory(context, symbolic, segmentation,
                                  proposal_source="source_symbol_predicates_v1")
    if client is None:
        raise APPreparationError("AP_GROUNDING_REQUIRED", "尚无 AP 定义；请配置生成模型提取候选，或提供已审核的 AP 定义。")
    payload = {"task": "extract_source_ap_candidates", "source_text": context.grounding_contract.source_text}
    if recovery_feedback:
        payload["previous_failed_attempts"] = list(recovery_feedback)
        payload["instruction"] = (
            "Correct the previous failed AP proposals using the reported source-binding errors. "
            "Quote only exact event/state fragments, not modal or temporal scaffolding. "
            "Return proposals only, never approval or formulas."
        )
    # Deliberately omit cuts and reviewed alternative readings: the exact
    # effective source, rather than segment granularity, defines the evidence.
    failures = []
    for attempt in range(1, 3):
        raw = None
        try:
            with request_stage("ap_extraction"):
                prompt = AP_REPAIR_PROMPT if "ap_repair_feedback" in payload else AP_PROPOSAL_PROMPT
                raw = client.complete_json(prompt, dict(payload), temperature=0.0,
                                           response_schema=AP_PROPOSAL_SCHEMA)
            validate_response_schema(raw, AP_PROPOSAL_SCHEMA)
            original = raw
            raw = deduplicate_identical_proposals(raw)
            for item in payload.get("ap_repair_feedback", {}).get("preserve_candidates", []):
                if not any(same_ap_proposal(item, candidate) for candidate in raw["aps"]):
                    raise APProposalValidationError([{"code": "AP_REPAIR_DROPPED_VALID_PREDICATE",
                        "message": "AP repair changed or omitted a non-conflicting predicate: " + item["canonical_name"],
                        "proposal_indices": [], "protected_candidate": item}])
            prepared = _pending_inventory(context, raw, segmentation)
            record = {"attempt": attempt, "status": "valid", "proposal": original,
                      "normalization": "identical_proposals_deduplicated" if raw != original else "none"}
            if raw != original:
                record["effective_proposal"] = raw
            return replace(prepared, proposal_records=tuple([*failures, record]))
        except (ValueError, TypeError, APIdentityValidationError, LLMResponseSchemaError) as exc:
            feedback = ap_repair_feedback(exc, raw)
            failures.append({"attempt": attempt, "status": "invalid", "validation_error": str(exc), "proposal": raw,
                             "ap_repair_feedback": feedback})
            payload = {**payload, "validation_error": str(exc), "rejected_proposal": raw,
                       "ap_repair_feedback": feedback,
                       "instruction": (
                           "Regenerate source-bound AP proposals only. Fix the reported error: "
                           "split coordinated predicates into separate APs and reuse recurring predicates. "
                           "A source occurrence cannot belong to different APs; do not merely rename "
                           "an overlapping composite. Preserve all predicate meanings and qualifiers. "
                           "Do not add approval or formulas."
                       )}
    raise APPreparationError(
        "AP_PROPOSAL_INVALID",
        "AP 候选两次未通过原文绑定校验，尚未生成 IR。具体原因：" + failures[-1]["validation_error"],
        diagnostics={"attempts": 2, "validation_error": failures[-1]["validation_error"],
                     "failed_attempts": failures, "requirement_id": context.requirement_id,
                     "source_hash": context.effective_source_hash},
    )


def _edit_ap_proposals(command, current, read):
    """Edit one named proposal without changing its neighbours; accept legacy JSON."""
    argument = command[1:].strip()
    if argument.startswith("{"):
        return json.loads(argument)
    if not isinstance(current, dict) or not isinstance(current.get("aps"), list):
        raise ValueError("当前没有可修改的 AP，请输入 r 重新生成。")
    raw = json.loads(json.dumps(current))
    if argument:
        fields = argument.split(maxsplit=1)
        name = fields[0]
        evidence = fields[1] if len(fields) == 2 else None
    else:
        name = read("要修改的 AP 名称（canonical_name）：").strip()
        evidence = None
    matches = [item for item in raw["aps"]
               if isinstance(item, dict) and item.get("canonical_name") == name]
    if len(matches) != 1:
        raise ValueError(f"找不到唯一的 AP {name!r}，请使用上方显示的名称。")
    item = matches[0]
    if evidence is None:
        new_name = read(f"新名称（回车保留 {name}）：").strip()
        evidence = read("原文依据（evidence_texts；多个片段用 | 分隔，回车保留）：").strip()
        if new_name:
            item["canonical_name"] = new_name
        if not evidence:
            return raw
    # A plain phrase needs no quotes or JSON; a JSON list remains available
    # for exact fragments containing the separator itself.
    if evidence.startswith("["):
        item["evidence_texts"] = json.loads(evidence)
    else:
        quotes = [part.strip() for part in evidence.split("|")]
        item["evidence_texts"] = quotes
    return raw


def prepare_reviewed_ap_context(context, client, *, segmentation=None, interactive=False,
                               records_dir=None, input_fn=None, plain_reading_review=None):
    """Human approval is separate from extraction and is never inferred."""
    extraction_error = None
    recovery_errors = []
    resolution_source = ""
    try:
        prepared = ensure_ap_candidates(context, client, segmentation=segmentation)
    except APPreparationError as exc:
        if not interactive or exc.code != "AP_PROPOSAL_INVALID":
            raise
        prepared = None
        extraction_error = exc
        recovery_errors.append(dict(exc.diagnostics))
    if context.has_authoritative_catalog or not interactive:
        return prepared
    if prepared is not None and prepared.inventory is None:
        if context.grounding_authority != "proposal_only":
            raise APPreparationError("AP_GROUNDING_REVIEW_REQUIRED", "请先补全并审核当前 AP 定义。")
        raw = {"aps": [
            {"canonical_name": identity.canonical_name,
             "evidence_texts": list(dict.fromkeys(m.evidence_text for m in identity.mentions))}
            for identity in context.grounding_contract.identities
        ]}
        prepared = _pending_inventory(context, raw, segmentation)
    read = input_fn or input
    proposal_records = list(prepared.proposal_records) if prepared is not None else []
    from llm.overlap_client import OverlapJSONClient
    if isinstance(client, OverlapJSONClient) and prepared is not None:
        client.prefetch_graph(prepared.context, segmentation, plain_reading_review)
    raw = None
    while True:
        print("\n请审核原子命题（AP）及其原文依据：")
        print(context.grounding_contract.source_text)
        if prepared is None:
            print("模型候选未通过原文校验，需重新生成或修改，当前不能确认。")
            print(f"具体原因：{extraction_error.diagnostics['validation_error']}")
            if raw is None:
                raw = extraction_error.diagnostics.get("failed_attempts", [{}])[-1].get("proposal")
            if isinstance(raw, dict) and isinstance(raw.get("aps"), list):
                print("未通过校验的候选（仅供修改参考）：")
                print(json.dumps(raw, ensure_ascii=False))
        else:
            for identity in prepared.inventory.identities:
                mentions = [m for m in prepared.inventory.mentions if m.ap_identity_id == identity.ap_identity_id]
                print(f"  {identity.canonical_name}: " + "; ".join(f"{m.evidence_text!r} [{m.source_span[0]}:{m.source_span[1]}]" for m in mentions))
            raw = {"aps": [
                {"canonical_name": identity.canonical_name,
                 "evidence_texts": list(dict.fromkeys(
                     m.evidence_text for m in prepared.inventory.mentions
                     if m.ap_identity_id == identity.ap_identity_id))}
                for identity in prepared.inventory.identities
            ]}
        try:
            command = read('AP 审核：a=明确确认；r=重新生成；e=逐项修改；e 名称 原文片段1 | 原文片段2=仅改该 AP；c=暂不处理：').strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise APPreparationError("AP_GROUNDING_REVIEW_REQUIRED", "AP 审核尚未完成，未生成 IR。") from exc
        if command.casefold() == "a":
            if prepared is None:
                print("当前没有通过校验的 AP 候选，请输入 r 重新生成或 e 修改。")
                continue
            inventory = approve_ap_inventory(
                prepared.inventory, profile=context.representation_profile,
                reviewer_id="interactive-user", reviewer_source="human", segmentation=segmentation,
            )
            approved = build_ap_binding_context(
                requirement_id=context.requirement_id,
                source_text=context.grounding_contract.source_text,
                representation_profile=context.representation_profile, active_inventory=inventory,
                identity_mapping_id=context.identity_mapping_id,
                provenance=context.provenance,
            )
            if records_dir is not None:
                from contracts.records import write_json
                name = re.sub(r"[^A-Za-z0-9_-]", "_", context.requirement_id)
                write_json(Path(records_dir) / f"ap_review_{name}_{context.effective_source_hash[:12]}.json",
                           {"stage": "ap_grounding", "source_hash": context.effective_source_hash,
                            "ap_inventory": asdict(inventory),
                            "ap_context_fingerprint": approved.context_fingerprint,
                            **({"proposal_generation": proposal_records} if proposal_records else {}),
                            **({"proposal_recovery": {"failures": recovery_errors,
                                 "resolution_source": resolution_source}} if recovery_errors else {})})
            return PreparedAPContext(approved, inventory, tuple(proposal_records))
        if command.casefold() in {"c", "cancel", "defer"}:
            break
        if command.casefold() == "r":
            try:
                # An explicit regeneration may replace proposals, never authority.
                fresh_context = replace(context, grounding_contract=replace(
                    context.grounding_contract, identities=(), fingerprint="",
                ), context_fingerprint="")
                prepared = ensure_ap_candidates(
                    fresh_context, client, segmentation=segmentation,
                    recovery_feedback=extraction_error.diagnostics.get("failed_attempts") if extraction_error else None,
                )
                proposal_records.extend(prepared.proposal_records)
                resolution_source = "generation_model_ap_proposal"
            except APPreparationError as exc:
                if exc.code != "AP_PROPOSAL_INVALID":
                    raise
                extraction_error = exc
                recovery_errors.append(dict(exc.diagnostics))
                if prepared is not None:
                    print(f"重新生成未通过校验，保留当前有效候选。具体原因：{exc.diagnostics['validation_error']}")
                else:
                    raw = None
            continue
        if command.casefold() == "e" or (command[:1].casefold() == "e" and command[1:2].isspace()):
            try:
                edited = _edit_ap_proposals(command, raw, read)
                if prepared is None:
                    # Retain incremental repairs when several rejected APs
                    # need editing before the whole inventory can validate.
                    raw = edited
                prepared = _pending_inventory(context, edited, segmentation,
                                              proposal_source="human_ap_edit")
                resolution_source = "human_ap_edit"
            except (EOFError, KeyboardInterrupt) as exc:
                raise APPreparationError("AP_GROUNDING_REVIEW_REQUIRED", "AP 审核尚未完成，未生成 IR。") from exc
            except (ValueError, TypeError, APIdentityValidationError, LLMResponseSchemaError) as exc:
                print(f"AP 修改未通过校验：{exc}")
        else:
            print("请输入 a、r、e 或 c；也可输入 e 名称 原文依据，只修改该 AP；空输入不会批准 AP。")
    if prepared is None:
        raise extraction_error
    raise APPreparationError("AP_GROUNDING_REVIEW_REQUIRED", "AP 审核尚未完成，未生成 IR。")
