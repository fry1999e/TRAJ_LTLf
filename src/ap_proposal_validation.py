"""Source-only diagnostics for AP proposals and bounded model repair."""
import re

from ap_identity import exact_ap_evidence_spans


class APProposalValidationError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__(issues[0]["message"] + (f" ({len(issues)} issues in total)" if len(issues) > 1 else ""))


def validate_source_ap_proposals(source, raw):
    """Collect all conflicts so one repair can fix the whole candidate list."""
    issues, names, owners = [], {}, []

    def issue(code, message, indices, **details):
        issues.append({"code": code, "message": message, "proposal_indices": sorted(set(indices)), **details})

    scaffolding = set("should whenever eventually always shall must and or not if then globally when until before after next finally".split())
    scaffolding.update({"且", "或", "并且", "或者", "如果", "那么", "最终", "总是", "必须", "应当", "应该", "随后"})
    for index, item in enumerate(raw["aps"]):
        name = item["canonical_name"]
        if name.casefold() in names:
            issue("AP_DUPLICATE_NAME", "AP names must be unique ignoring case",
                  [names[name.casefold()], index], canonical_name=name)
        names[name.casefold()] = index
        for quote in item["evidence_texts"]:
            if quote != quote.strip() or quote not in source:
                issue("AP_QUOTE_NOT_EXACT", f"AP evidence must be an exact original quote: {quote!r}",
                      [index], evidence_text=quote)
                continue
            if quote.casefold() in scaffolding:
                issue("AP_SCAFFOLDING_ONLY", "A connective/modal/temporal word alone is not an AP",
                      [index], evidence_text=quote)
            if (re.search(r"\b(?:should|shall|must|will)\s+(?:(?:not|also)\s+)*(?:eventually|always|never)\b", quote, re.I)
                    or re.search(r"(?:应当|应该|必须|将)(?:会|要|不)?(?:最终|总是|始终|一直|永不)", quote)):
                issue("AP_TEMPORAL_CLAUSE", "AP evidence swallows a modal/temporal operator; quote only the event/state fragments: " + repr(quote),
                      [index], evidence_text=quote)
            if not exact_ap_evidence_spans(source, quote, case_sensitive=True):
                issue("AP_IDENTIFIER_SUBSTRING", f"AP evidence must not match part of another identifier: {quote!r}",
                      [index], evidence_text=quote)
                continue
            for span in exact_ap_evidence_spans(source, quote):
                for previous, owner, previous_quote in owners:
                    if owner == index or not (span[0] < previous[1] and previous[0] < span[1]):
                        continue
                    start, end = max(span[0], previous[0]), min(span[1], previous[1])
                    issue("AP_EVIDENCE_OVERLAP",
                          f"Overlapping source evidence belongs to different APs: {raw['aps'][owner]['canonical_name']!r} and {name!r} at {source[start:end]!r}",
                          [owner, index], ap_names=[raw["aps"][owner]["canonical_name"], name],
                          overlap_text=source[start:end], overlap_span=[start, end],
                          evidence_texts=[previous_quote, quote], source_spans=[list(previous), list(span)])
                if not any(owner == index and previous == span for previous, owner, _quote in owners):
                    owners.append((span, index, quote))
    if issues:
        raise APProposalValidationError(issues)


def ap_repair_feedback(error, raw):
    issues = getattr(error, "issues", None) or [{"code": "AP_SCHEMA_OR_BINDING_INVALID",
        "message": str(error), "path": getattr(error, "path", ""), "proposal_indices": []}]
    involved = {index for item in issues for index in item["proposal_indices"]}
    proposals = raw.get("aps", []) if isinstance(raw, dict) else []
    preserved = ([item for index, item in enumerate(proposals) if index not in involved]
                 if involved else [])
    return {"issues": issues, "preserve_candidates": preserved,
        "repair_operations": [
            "Split independently testable coordinated predicates into separate APs; retain every event/state.",
            "Reuse one AP for repeated occurrences of the same predicate, including condition and consequence.",
            "For distinct predicates sharing an entity word, quote predicate-specific source phrases instead of merging them.",
            "Use separate exact fragments for one discontinuous predicate; fragments cannot stand for different predicates.",
            "Keep negation, participants and qualifiers; leave Boolean and temporal operators outside APs.",
            "Preserve the listed non-conflicting candidates exactly. Return the complete repaired aps list, not a patch or approval.",
        ]}


def deduplicate_identical_proposals(raw):
    """Collapse exact repeats only, never merge differently named predicates."""
    unique = []
    for item in raw["aps"]:
        if not any(same_ap_proposal(item, previous) for previous in unique):
            unique.append(item)
    return {**raw, "aps": unique}


def same_ap_proposal(first, second):
    return (first["canonical_name"] == second["canonical_name"]
            and set(first["evidence_texts"]) == set(second["evidence_texts"]))
