"""Bounded shortest separation over IR semantics, with a blind public view."""
from dataclasses import dataclass
from itertools import product
from ir_trace_semantics import evaluate_ir_trace


@dataclass(frozen=True)
class DiscriminatorResult:
    status: str
    trace: tuple[tuple[str, ...], ...] = ()
    examined: int = 0
    # Internal only. Never serialize this object to an annotation provider.
    partition: tuple[bool, ...] = ()
    projected_aps: tuple[str, ...] = ()


def shortest_discriminator(irs, *, assignments=None, max_length=4, max_traces=4096, relevant_aps=None):
    if len(irs) < 2: return DiscriminatorResult("no_disagreement")
    if max_length < 1 or max_traces < 1: raise ValueError("positive frozen bounds required")
    source = (irs[0].source_hash, irs[0].ap_context_fingerprint)
    if any((ir.source_hash, ir.ap_context_fingerprint) != source for ir in irs):
        raise ValueError("discriminator inputs must share source and AP binding")
    assignments = assignments if assignments is not None else [{} for _ in irs]
    if len(assignments) != len(irs): raise ValueError("assignment cardinality mismatch")
    all_aps = set().union(*(set(ir.ap_names) for ir in irs))
    if relevant_aps is None:
        aps = sorted(all_aps)
    else:
        requested = set(relevant_aps)
        if not requested or not requested <= all_aps:
            raise ValueError("relevant_aps must be a nonempty subset of the IR AP domain")
        aps = sorted(requested)
    projected = tuple(aps) if set(aps) != all_aps else ()
    if 2 ** len(aps) > max_traces:
        return DiscriminatorResult("search_budget_exhausted", projected_aps=projected)
    states = [tuple(ap for ap, present in zip(aps, bits) if present)
              for bits in product((False, True), repeat=len(aps))]
    examined = 0
    for length in range(1, max_length + 1):
        best = None
        for trace in product(states, repeat=length):
            if examined >= max_traces:
                return best or DiscriminatorResult("search_budget_exhausted", examined=examined, projected_aps=projected)
            partition = tuple(evaluate_ir_trace(ir, [set(s) for s in trace], temporal_resolution_by_node=assignment)
                              for ir, assignment in zip(irs, assignments))
            examined += 1
            if len(set(partition)) > 1:
                balance = min(sum(partition), len(partition)-sum(partition))
                if best is None or balance > min(sum(best.partition), len(best.partition)-sum(best.partition)):
                    best = DiscriminatorResult("witness_found", trace, examined, partition, projected)
        if best: return best
    return DiscriminatorResult("no_witness_within_bound", examined=examined, projected_aps=projected)


def blind_annotation_payload(*, source_text, ap_context, contract, trace):
    from contracts.dataset_semantics_contract import contract_prompt_view, validate_dataset_semantics_contract
    ap_context.validate_source(source_text)
    validate_dataset_semantics_contract(contract)
    names = set(ap_context.declared_catalog)
    if not trace or any(not set(step) <= names for step in trace):
        raise ValueError("annotation trace uses unknown APs or is empty")
    glossary = []
    if ap_context.grounding_contract is None: raise ValueError("authoritative AP glossary required")
    for item in ap_context.grounding_contract.identities:
        glossary.append({"canonical_name": item.canonical_name, "mentions": [
            {"source_span": list(m.source_span), "text": m.evidence_text} for m in item.mentions]})
    return {"source_text": source_text, "ap_glossary": glossary,
            "non_quotable_contract": contract_prompt_view(contract),
            "trace": [sorted(step) for step in trace],
            "question": "Does this finite trace satisfy the source requirement under the stated contract? Return SAT, VIOLATE, or UNCLEAR with exact source evidence."}


def adjudicated_label(labels, risk, *, adjudication=None):
    required = 1 if risk == "low" else 2
    if risk not in {"low", "medium", "high"}: raise ValueError("unknown risk")
    if len(labels) != required or any(label not in {"SAT", "VIOLATE", "UNCLEAR"} for label in labels):
        return "UNCLEAR"
    if "UNCLEAR" in labels: return "UNCLEAR"
    if len(set(labels)) > 1 or risk == "high":
        return adjudication if adjudication in {"SAT", "VIOLATE"} else "UNCLEAR"
    return labels[0]


def validate_candidate_completeness(candidate_set, *, max_branches=4):
    """Do not call a truncated Cartesian product a complete CandidateIRSet."""
    from math import prod
    if candidate_set.ambiguity_debt or len(candidate_set.branches) > max_branches:
        return False
    axes = candidate_set.ambiguity_axes.axes if candidate_set.ambiguity_axes else ()
    if not axes:
        return len(candidate_set.branches) == 1
    if any(len(axis.possible_readings) < 2 for axis in axes): return False
    expected = prod(len(axis.possible_readings) for axis in axes)
    if expected > max_branches or len(candidate_set.branches) != expected: return False
    signatures = {tuple(sorted(branch.applied_alternative_ids)) for branch in candidate_set.branches}
    return len(signatures) == expected and all(signatures)
