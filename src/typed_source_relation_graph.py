"""Conservative, source-total parser for the explicit relation fragment.

Every non-punctuation character must belong to a catalog mention or grammar
token. Unsupported prose is never silently dropped. No dataset/template input.

The public graph is also the schema used by constrained model proposals.  A
model may propose spans, relation roles and unresolved axes, but stable IDs,
source/AP/contract hashes and exact evidence text are always derived locally.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from typing import Mapping

from semantic_node_contracts import NODE_CONTRACTS, node_contract_errors
from relation_parameter_contracts import periodicity_mode_contract
from schemas import FAITHFUL_IR_SCHEMA_VERSION, SemanticNode
from semantic_obligation_matrix import is_ap_denotation_only_debt, is_unbounded_eventuality_observation
from source_relation_lexicon import SOURCE_RELATION_LEXICON, relation_pattern, relation_tokens
from requirement_reading_contract import REQUIREMENT_READING_RULES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class UnsupportedConstruct(ValueError):
    def __init__(self, code, span=(-1, -1)):
        self.code, self.span = code, span
        super().__init__(code)


@dataclass(frozen=True)
class SourceNode:
    node_id: str
    kind: str
    source_span: tuple[int, int]
    exact_text: str
    ap_identity_key: str = ""
    canonical_name: str = ""
    parameters: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class SourceEdge:
    edge_id: str
    relation: str
    from_ids: tuple[str, ...]
    to_ids: tuple[str, ...]
    source_span: tuple[int, int]
    authority: str = "explicit_source"
    confidence: str = "proven"
    parser_provenance: str = "rcg-explicit-grammar-v1"


@dataclass(frozen=True)
class SourceAxisOption:
    option_id: str
    label: str
    operation: str
    target_node_id: str
    value: str


@dataclass(frozen=True)
class SourceIntentAxis:
    axis_id: str
    claim_id: str
    axis_kind: str
    source_span: tuple[int, int]
    evidence_text: str
    options: tuple[SourceAxisOption, ...]
    contract_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSemanticDebt:
    debt_id: str
    reason_code: str
    source_span: tuple[int, int]
    evidence_text: str
    detail: str = ""


@dataclass(frozen=True)
class SourceAxisDebtResolution:
    debt: SourceSemanticDebt
    axis_ids: tuple[str, ...]
    model: str
    reason: str


COVERAGE_CATEGORIES = frozenset({
    "AP_EVIDENCE",
    "RELATION_MARKER",
    "PREDICATE_SCAFFOLD",
    "CONTRACT_COVERED",
    "SEMANTIC_AXIS",
    "CAPABILITY_GAP",
    "SOURCE_UNCLEAR",
})


@dataclass(frozen=True)
class SourceCoverageEntry:
    category: str
    source_span: tuple[int, int]
    evidence_text: str
    owner: str = "backend"


@dataclass(frozen=True)
class SourceCoverageLedger:
    source_hash: str
    entries: tuple[SourceCoverageEntry, ...]
    blocking_debts: tuple[SourceSemanticDebt, ...] = ()
    ledger_version: str = "source-coverage-ledger-v1"

    @property
    def ledger_hash(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class TypedSourceRelationGraph:
    source_text: str
    source_hash: str
    ap_binding_hash: str
    contract_hash: str
    nodes: tuple[SourceNode, ...]
    edges: tuple[SourceEdge, ...]
    root_node_id: str
    unresolved_axes: tuple[str, ...] = ()
    axes: tuple[SourceIntentAxis, ...] = ()
    semantic_debts: tuple[SourceSemanticDebt, ...] = ()
    coverage_ledger: SourceCoverageLedger | None = None
    schema_version: str = "source-intent-graph-v2"
    proposal_mode: str = "explicit_grammar"
    axis_debt_resolutions: tuple[SourceAxisDebtResolution, ...] = ()

    @property
    def graph_hash(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class SourceIntentGraphSet:
    primary: TypedSourceRelationGraph
    candidates: tuple[TypedSourceRelationGraph, ...]
    axes: tuple[SourceIntentAxis, ...] = ()
    semantic_debts: tuple[SourceSemanticDebt, ...] = ()
    complete: bool = True
    candidate_selections: tuple[tuple[tuple[str, str], ...], ...] = ()


@dataclass(frozen=True)
class SourceNodeContract:
    """Formula-free source-node contract backed by one executable IR node."""

    ir_node_type: str
    min_children: int
    max_children: int | None
    parameters: Mapping[str, object]
    required_parameters: tuple[str, ...]
    root_allowed: bool = True


def _source_contract(ir_node_type: str) -> SourceNodeContract:
    contract = NODE_CONTRACTS[ir_node_type]
    return SourceNodeContract(
        ir_node_type=ir_node_type,
        min_children=contract.min_children,
        max_children=contract.max_children,
        parameters={
            name: parameter.json_schema()
            for name, parameter in sorted(contract.parameters.items())
        },
        required_parameters=tuple(sorted(
            name
            for name, parameter in contract.parameters.items()
            if parameter.required
        )),
    )


# This is the single registry used by the prompt view, response schema,
# proposal validator and IR builder.  A source kind is exposed only when the
# independent interpreter and compiler already implement its mapped IR node.
SOURCE_NODE_CONTRACTS: Mapping[str, SourceNodeContract] = {
    "AP": SourceNodeContract("ap_condition", 0, 0, {}, ()),
    "true": SourceNodeContract("boolean_constant", 0, 0, {}, ()),
    "false": SourceNodeContract("boolean_constant", 0, 0, {}, ()),
    **{
        kind: _source_contract(kind)
        for kind in (
            "negation",
            "conjunction",
            "disjunction",
            "implication",
            "global_constraint",
            "eventually",
            "next_step",
            "until",
            "fixed_duration",
            "exact_run_length",
            "bounded_window",
            "before",
            "precedence",
            "after",
            "cardinality",
            "recurrence",
            "periodicity",
        )
    },
}

SOURCE_RELATION_ARITY = {
    kind: (contract.min_children, contract.max_children)
    for kind, contract in SOURCE_NODE_CONTRACTS.items()
}

SOURCE_RELATION_PARAMETERS = {
    kind: dict(contract.parameters)
    for kind, contract in SOURCE_NODE_CONTRACTS.items()
    if contract.parameters
}

SOURCE_DEBT_REASON_CODES = (
    "semantic_axis",
    "capability_gap",
    "source_unclear",
    "unsupported_semantics",
)

_SCAFFOLD_TOKENS = frozenset({
    "a", "an", "the", "it", "is", "are", "be", "being", "been",
    "shall", "should", "must", "mode", "active", "holds", "hold",
    "satisfy", "satisfies", "true", "case", "that", "then", "of",
    "all", "following", "condition", "conditions", "at", "same", "time",
    "point", "instant", "in", "response", "right", "away", "promptly",
    "instantly", "without", "any", "delay",
})

_RELATION_MARKER_TOKENS = {
    kind: relation_tokens(kind) for kind in SOURCE_RELATION_LEXICON
}

_MATERIAL_DEBT_WORDS = frozenset({
    "boundary", "cardinality", "count", "counting", "duration", "edge",
    "exclusive", "inclusive", "persistence", "persist", "reset", "scope",
    "step", "steps", "strict future", "transition", "while", "weak", "strong",
    "deadline", "time bound", "current", "strict",
})


def source_graph_response_schema(ap_context=None) -> dict[str, object]:
    """Return the strict provider-independent G-route output contract."""

    node_alternatives = []
    for kind, contract in SOURCE_NODE_CONTRACTS.items():
        children = {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": contract.min_children,
        }
        if contract.max_children is not None:
            children["maxItems"] = contract.max_children
        properties = {
            "ref": {"type": "string", "minLength": 1},
            "kind": {"const": kind},
            "source_span": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
            "children": children,
        }
        required = ["ref", "kind", "source_span", "children"]
        if kind == "AP":
            # New proposals select a backend-owned identity. Legacy span-only
            # proposals remain readable; names remove model offset arithmetic.
            names = [identity.canonical_name for identity in ap_context.grounding_contract.identities] if (
                ap_context is not None and ap_context.grounding_contract is not None
            ) else []
            properties["ap_name"] = {"type": "string", **({"enum": names} if names else {"minLength": 1})}
            required = ["ref", "kind", "children"]
        if contract.parameters:
            properties["parameters"] = {
                "type": "object",
                "properties": dict(contract.parameters),
                "required": list(contract.required_parameters),
                "additionalProperties": False,
            }
            required.append("parameters")
        alternative = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        if kind == "AP":
            alternative["anyOf"] = [{"required": ["ap_name"]}, {"required": ["source_span"]}]
        node_alternatives.append(alternative)
    node = {"oneOf": node_alternatives}
    debt = {
        "type": "object",
        "properties": {
            "reason_code": {"type": "string", "enum": list(SOURCE_DEBT_REASON_CODES)},
            "source_span": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
            "detail": {"type": "string"},
        },
        "required": ["reason_code", "source_span", "detail"],
        "additionalProperties": False,
    }
    return {
        "$id": "traj_ltlf_source_intent_graph_v4",
        "type": "object",
        "properties": {
            "source_intent_graph": {
                "type": "object",
                "properties": {
                    "nodes": {"type": "array", "minItems": 1, "items": node},
                    "root_ref": {"type": "string", "minLength": 1},
                    "semantic_debts": {"type": "array", "items": debt},
                },
                "required": ["nodes", "root_ref", "semantic_debts"],
                "additionalProperties": False,
            }
        },
        "required": ["source_intent_graph"],
        "additionalProperties": False,
    }


def _proposal_debt_is_already_covered(
    debt: SourceSemanticDebt,
    text: str,
    nodes: tuple[SourceNode, ...],
    ap_evidence_spans: tuple[tuple[int, int], ...] = (),
) -> bool:
    """Conservatively suppress a model debt already encoded by the graph.

    This does not claim final semantic approval.  It only prevents grammar
    markers and AP predicate scaffolding from vetoing candidate construction
    after the graph already represents them.  Material boundary/count/state
    distinctions remain debts even when their marker overlaps a graph span.
    """

    detail = debt.detail.casefold()
    if not detail.strip():
        return False
    if is_unbounded_eventuality_observation(debt, text, nodes):
        return True
    if is_ap_denotation_only_debt(debt, nodes):
        return True
    if any(word in detail for word in _MATERIAL_DEBT_WORDS):
        return False
    start, end = debt.source_span
    fragment = list(text[start:end])
    spans = (*ap_evidence_spans, *(node.source_span for node in nodes if node.kind == "AP"))
    for span in spans:
        overlap_start = max(start, span[0])
        overlap_end = min(end, span[1])
        for index in range(max(0, overlap_start - start), max(0, overlap_end - start)):
            fragment[index] = " "
    tokens = set(re.findall(r"[A-Za-z0-9_]+", "".join(fragment).casefold()))
    if not tokens:
        return True
    covered = set(_SCAFFOLD_TOKENS)
    kinds = {node.kind for node in nodes}
    for kind in kinds:
        covered.update(_RELATION_MARKER_TOKENS.get(kind, ()))
    return tokens <= covered


def build_source_coverage_ledger(
    text: str,
    nodes: tuple[SourceNode, ...],
    *,
    axes: tuple[SourceIntentAxis, ...] = (),
    debt_hints: tuple[SourceSemanticDebt, ...] = (),
    contract_covered_spans: tuple[tuple[int, int], ...] = (),
    ap_evidence_spans: tuple[tuple[int, int], ...] = (),
) -> SourceCoverageLedger:
    """Build the backend-owned lexical coverage and blocking-debt ledger.

    Model debts are treated as hints: a hint that contains only AP predicate
    scaffolding or a relation already encoded by the graph is discarded.  In
    the other direction, explicit unsupported markers create capability debt
    even if the model omitted a debt record.
    """

    source_hash = hashlib.sha256(text.encode()).hexdigest()
    entries: list[SourceCoverageEntry] = []
    occupied: set[int] = set()

    def add(category: str, span: tuple[int, int], owner: str = "backend") -> None:
        if category not in COVERAGE_CATEGORIES:
            raise ValueError("unknown source coverage category: " + category)
        start, end = span
        if not 0 <= start < end <= len(text):
            return
        entries.append(SourceCoverageEntry(category, span, text[start:end], owner))
        occupied.update(range(start, end))

    for node in nodes:
        if node.kind == "AP":
            add("AP_EVIDENCE", node.source_span, node.node_id)
    for span in ap_evidence_spans:
        add("AP_EVIDENCE", span, "source_bound_ap_inventory")
    for span in contract_covered_spans:
        add("CONTRACT_COVERED", tuple(span), "dataset_contract")
    for axis in axes:
        add("SEMANTIC_AXIS", axis.source_span, axis.axis_id)

    kinds = {node.kind for node in nodes}
    marker_words = set()
    for kind in kinds:
        marker_words.update(_RELATION_MARKER_TOKENS.get(kind, ()))
    if {"global_constraint", "implication"} <= kinds:
        marker_words.add("while")

    tokens = tuple(re.finditer(r"[A-Za-z0-9_]+", text))
    for match in tokens:
        span = match.span()
        if all(index in occupied for index in range(*span)):
            continue
        token = match.group(0).casefold()
        if token in marker_words:
            add("RELATION_MARKER", span)
        elif token in _SCAFFOLD_TOKENS or token.isdigit() or token in _NUMBER_WORDS:
            add("PREDICATE_SCAFFOLD", span)
        else:
            add("SOURCE_UNCLEAR", span)

    debts = [
        debt for debt in debt_hints
        if not _proposal_debt_is_already_covered(debt, text, nodes, ap_evidence_spans)
    ]

    # Recognize only explicit, source-anchored missing capabilities.  General
    # uncovered prose stays visible as SOURCE_UNCLEAR coverage, but does not
    # become a blocker merely because a lexical whitelist is incomplete.
    missing_markers = []
    marker_contracts = (
        (r"\bwhile\b", {"global_constraint", "implication"}, "while", True),
        (r"\b(?:within|between)\b", {"bounded_window"}, "bounded_window", False),
        (relation_pattern("precedence"), {"before", "precedence"}, "precedence", False),
        (relation_pattern("after"), {"after", "periodicity"}, "after", False),
        (r"\bevery\s+(?:\d+|[a-z]+)\s+steps?\b", {"periodicity"}, "periodicity", False),
        (r"\bfor\s+(?:only\s+)?(?:\d+|[a-z]+)\s+steps?\b", {"fixed_duration", "exact_run_length"}, "duration", False),
        (r"\b(?:at\s+(?:least|most)|exactly)\s+(?:\d+|[a-z]+)\s+(?:times?|occurrences?)\b", {"cardinality"}, "cardinality", False),
    )
    for pattern, represented_by, label, require_all in marker_contracts:
        for match in re.finditer(pattern, text, re.I):
            represented = (
                represented_by <= kinds
                if require_all
                else not kinds.isdisjoint(represented_by)
            )
            if not represented:
                missing_markers.append((match.span(), label))
    existing_spans = {
        debt.source_span for debt in debts
        if debt.reason_code in {"capability_gap", "unsupported_semantics"}
    }
    for span, label in missing_markers:
        if span in existing_spans:
            continue
        debts.append(SourceSemanticDebt(
            _stable_id("sdebt-", source_hash, "capability_gap", label, span),
            "capability_gap",
            span,
            text[span[0]:span[1]],
            f"Source marker requires unsupported or omitted {label} semantics.",
        ))
        add("CAPABILITY_GAP", span, "backend_detector")

    entries.sort(key=lambda entry: (entry.source_span, entry.category, entry.owner))
    return SourceCoverageLedger(source_hash, tuple(entries), tuple(debts))


def _stable_id(prefix, *parts):
    return prefix + digest(parts)[:16]


_NUMBER_WORDS = {
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
    "eleven": 11,
    "twelve": 12,
}


def _source_numbers(fragment: str) -> frozenset[int]:
    values = set()
    for token in re.findall(r"\b(?:\d+|[A-Za-z]+)\b", fragment.casefold()):
        if token.isdigit():
            values.add(int(token))
        elif token in _NUMBER_WORDS:
            values.add(_NUMBER_WORDS[token])
    return frozenset(values)


def _validate_source_parameter_grounding(
    kind: str,
    parameters: Mapping[str, object],
    fragment: str,
) -> None:
    """Reject typed values that are not licensed by their exact source span.

    The structural registry validates types and closed-world fields.  This
    second check prevents a model from inventing numeric values or silently
    resolving an ordering mode that the cited words do not state.
    """

    folded = fragment.casefold()
    numbers = _source_numbers(fragment)
    if kind == "exact_run_length":
        if not re.search(r"\b(?:for\s+)?only\s+(?:\d+|[a-z]+)\s+steps?\b", folded):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
        if parameters["duration"] not in numbers:
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
        if parameters["minimum_horizon"] < (
            parameters["start_offset"] + parameters["duration"]
        ):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_INVALID")
    elif kind == "fixed_duration":
        if not re.search(r"\bfor\s+(?:\d+|[a-z]+)\s+steps?\b", folded):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
        if parameters["duration"] not in numbers:
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
    elif kind == "bounded_window":
        if not re.search(r"\b(?:within|between|from|no\s+later\s+than)\b", folded):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
        evidenced = {parameters["upper_bound"]}
        if parameters["lower_bound"] != 0:
            evidenced.add(parameters["lower_bound"])
        if not evidenced <= numbers:
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
    elif kind == "cardinality":
        marker = {
            "at_least": r"\bat\s+least\b",
            "at_most": r"\bat\s+most\b",
            "exactly": r"\bexactly\b",
        }[str(parameters["mode"])]
        if not re.search(marker, folded) or parameters["count"] not in numbers:
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
    elif kind == "periodicity":
        if not re.search(r"\bevery\b", folded) or parameters["period"] not in numbers:
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
        mode_contract = periodicity_mode_contract(parameters.get("mode"))
        if mode_contract is None:
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_INVALID")
        names = set(parameters)
        if not mode_contract.required <= names or names & mode_contract.forbidden:
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_INVALID")
        if any(parameters.get(name) != value for name, value in mode_contract.fixed_values.items()):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_INVALID")
        if parameters.get("mode") == "fixed_phase" and not re.search(
            r"\b(?:starting\s+now|from\s+the\s+(?:current|initial)\s+(?:position|step))\b",
            folded,
        ):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
    elif kind in {"before", "precedence"}:
        if not re.search(SOURCE_RELATION_LEXICON["precedence"].pattern, folded):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
        explicit_non_strict = bool(re.search(r"\b(?:at\s+or\s+before|no\s+later\s+than)\b", folded))
        expected = (
            ("non_strict", "allowed")
            if explicit_non_strict
            else ("strict", "forbidden")
        )
        if (parameters["ordering"], parameters["same_step_policy"]) != expected:
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
    elif kind == "after" and not re.search(r"\bafter\b", folded):
        raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")
    elif kind == "recurrence" and not re.search(
        r"\b(?:recur|recurs|repeatedly|again|always\s+eventually)\b", folded
    ):
        raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_UNGROUNDED")


def parse_source_graph(
    text,
    ap_context,
    *,
    contract_hash="",
    allow_proposal_identities=False,
    temporal_coordination_scope=None,
):
    ap_context.validate_source(text)
    source_bound_candidate_allowed = (
        allow_proposal_identities
        and ap_context.grounding_contract is not None
        and ap_context.grounding_authority in {"proposal_only", "required_subset"}
        and bool(ap_context.grounding_contract.identities)
    )
    if (
        not ap_context.has_authoritative_catalog
        and not source_bound_candidate_allowed
    ) or ap_context.grounding_contract is None:
        raise UnsupportedConstruct("AP_ALIAS_UNRESOLVED")
    mentions = {}
    for identity in ap_context.grounding_contract.identities:
        for mention in identity.mentions:
            start, end = mention.source_span
            if text[start:end] != mention.evidence_text:
                raise UnsupportedConstruct("AP_SPAN_INVALID", (start, end))
            key = (start, end)
            if key in mentions and mentions[key].identity_key != identity.identity_key:
                raise UnsupportedConstruct("AP_ALIAS_UNRESOLVED", key)
            mentions[key] = identity
    positions = sorted(mentions)
    if any(a[1] > b[0] for a, b in zip(positions, positions[1:])):
        raise UnsupportedConstruct("AP_OVERLAPPING_SPAN")
    by_start = {a: (b, mentions[a, b]) for a, b in positions}
    tokens = []
    pos = 0
    while pos < len(text):
        if pos in by_start:
            end, identity = by_start[pos]
            # Exact positive predication around an already authoritative AP.
            # Negated/modal/qualified prose outside these forms remains unsupported.
            suffix = re.match(
                r"\s+(?:mode\s+(?:shall\s+be|is)\s+active|(?:shall\s+)?holds?|mode\b)",
                text[end:],
                re.I,
            )
            if suffix and not any(end <= a < end + len(suffix[0]) for a in by_start):
                end += len(suffix[0])
            tokens.append(("AP", pos, end, identity)); pos = end; continue
        if text[pos].isspace():
            pos += 1; continue
        framing = re.match(
            r"(?:globally\s+it\s+is\s+the\s+case\s+that\b|globally\s*,?|"
            r"(?:it\s+is\s+)?always\s+the\s+case\s+that\b)",
            text[pos:],
            re.I,
        )
        if framing:
            end = pos + len(framing[0]); tokens.append(("globally", pos, end, None)); pos = end; continue
        recurring = re.match(r"(?:everytime|every\s+time)\s+when\b", text[pos:], re.I)
        if recurring:
            end = pos + len(recurring[0]); tokens.append(("whenever", pos, end, None)); pos = end; continue
        conditional_scaffold = re.match(
            r"(?:in\s+case\s+that|in\s+the\s+(?:case|event)(?:\s+that)?|"
            r"on\s+condition\s+that|"
            r"if\s+it\s+is\s+the\s+case\s+that|"
            r"it\s+is\s+the\s+case(?:\s+that)?|as\s+soon\s+as)\b",
            text[pos:],
            re.I,
        )
        if conditional_scaffold:
            end = pos + len(conditional_scaffold[0])
            tokens.append(("when", pos, end, None)); pos = end; continue
        global_response = re.match(
            r"(?!(?:in|if|when|whenever|always|globally|eventually|next|not)\b)"
            r"(?:[A-Za-z][A-Za-z0-9_-]*\s+){0,4}shall\s+always\s+satisfy\b",
            text[pos:],
            re.I,
        )
        if global_response:
            end = pos + len(global_response[0]); tokens.append(("global_response", pos, end, None)); pos = end; continue
        exact_duration = re.match(
            r"for\s+only\s+(one|two|three|four|five|\d+)\s+steps?\b",
            text[pos:],
            re.I,
        )
        if exact_duration:
            raw_count = exact_duration.group(1).casefold()
            count = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            }.get(raw_count, int(raw_count) if raw_count.isdigit() else 0)
            end = pos + len(exact_duration[0])
            tokens.append(("exact_duration", pos, end, count)); pos = end; continue
        then_scaffold = re.match(
            r"then(?:\s+(?:all\s+of\s+the\s+following\s+conditions\s+"
            r"(?:are\s+true|hold)|the\s+following\s+condition\s+"
            r"(?:is\s+true|holds)|in\s+response))?\s*:?",
            text[pos:],
            re.I,
        )
        if then_scaffold:
            end = pos + len(then_scaffold[0]); tokens.append(("then", pos, end, None)); pos = end; continue
        predicate_scaffold = re.match(
            r"(?:the\s+)?(?:transition\s+action|scenario|event)\s+that\b|"
            r"(?:gets?|is|was)\s+(?:detected|observed)\b|"
            r"(?:takes\s+place|happens|occurs)\b|"
            r"(?:all\s+of\s+the\s+following\s+conditions\s+(?:are\s+true|hold)|"
            r"the\s+following\s+condition\s+(?:is\s+true|holds))\s*:?",
            text[pos:],
            re.I,
        )
        if predicate_scaffold:
            # Closed whitelist of truth-conditional scaffolding. Event/state
            # modifiers and temporal adverbs are intentionally not included.
            pos += len(predicate_scaffold[0]); continue
        current_position_scaffold = re.match(
            r"(?:at\s+the\s+same\s+(?:time(?:\s+(?:instant|point))?|moment)|"
            r"at\s+once|instantly|right\s+away|(?:starting\s+)?without\s+any\s+delay|"
            r"immediately|simultaneously|in\s+no\s+time)\b",
            text[pos:],
            re.I,
        )
        if current_position_scaffold:
            pos += len(current_position_scaffold[0]); continue
        coordination_scaffold = re.match(r"(?:either|both)\b", text[pos:], re.I)
        if coordination_scaffold:
            pos += len(coordination_scaffold[0]); continue
        registered_relation = re.match(
            relation_pattern("precedence", "after"), text[pos:], re.I
        )
        if registered_relation:
            surface = re.sub(r"\s+", " ", registered_relation[0].casefold())
            if re.fullmatch(SOURCE_RELATION_LEXICON["after"].pattern, surface, re.I):
                normalized = "after"
            elif surface in {"at or before", "no later than"}:
                normalized = "before_non_strict"
            else:
                normalized = "before"
            end = pos + len(registered_relation[0])
            tokens.append((normalized, pos, end, None)); pos = end; continue
        match = re.match(r"only\s+(?:if|when)\b|if\b|then\b|when(?:ever)?\b|while\b|in\b|always\b|globally\b|eventually\b|next\b|never\b|not\b|and\b|or\b|until\b|true\b|false\b|[&|(),.]", text[pos:], re.I)
        if not match:
            raise UnsupportedConstruct("unsupported_construct", (pos, len(text)))
        end = pos + len(match[0])
        normalized = re.sub(r"\s+", " ", match[0].lower())
        if normalized == "&": normalized = "and"
        if normalized == "|": normalized = "or"
        tokens.append((normalized, pos, end, None)); pos = end
    if tokens and tokens[-1][0] == ".":
        tokens.pop()
    # Treat an Oxford comma immediately before a Boolean coordinator as part
    # of that coordinator, not as an empty list operand.
    tokens = [
        token for index, token in enumerate(tokens)
        if not (
            token[0] == ","
            and index + 1 < len(tokens)
            and tokens[index + 1][0] in {"and", "or", "then"}
        )
    ]
    nodes, edges = [], []

    def make(kind, items, children=(), identity=None, parameters=()):
        span = (items[0][1], items[-1][2])
        parameters = tuple(parameters)
        nid = _stable_id(
            "sn-", hashlib.sha256(text.encode()).hexdigest(), kind, span,
            tuple(children), identity.identity_key if identity else "", parameters,
        )
        if any(node.node_id == nid for node in nodes):
            nid = _stable_id("sn-", nid, len(nodes))
        nodes.append(SourceNode(nid, kind, span, text[span[0]:span[1]],
                                identity.identity_key if identity else "",
                                identity.canonical_name if identity else "",
                                parameters))
        if children:
            edges.append(SourceEdge(
                _stable_id("se-", nid, kind, tuple(children), span),
                kind, (nid,), tuple(children), span,
            ))
        return nid

    def parse(items):
        if not items:
            raise UnsupportedConstruct("unsupported_construct")
        depth, top = 0, []
        for i, token in enumerate(items):
            if token[0] == "(": depth += 1
            elif token[0] == ")": depth -= 1
            if depth < 0: raise UnsupportedConstruct("unsupported_construct")
            if depth == 0: top.append(i)
        if depth: raise UnsupportedConstruct("unsupported_construct")
        if items[0][0] == "(" and top == [len(items)-1]:
            return parse(items[1:-1])
        first = items[0][0]
        if len(items) == 2 and items[1][0] == "exact_duration":
            duration = int(items[1][3])
            return make(
                "exact_run_length",
                items,
                (parse(items[:1]),),
                parameters=(
                    ("start_offset", 0),
                    ("duration", duration),
                    ("origin", "current_position"),
                    ("start_position_policy", "required"),
                    ("terminal_obligation_policy", "require_full_duration"),
                    ("post_window_policy", "false_if_position_exists"),
                    ("minimum_horizon", duration),
                ),
            )
        if first in {"always", "globally", "whenever"}:
            if first == "whenever":
                comma = next((i for i in top if items[i][0] in {",", "then"}), None)
                if comma is None: raise UnsupportedConstruct("unresolved_axis")
                child = make("implication", items, (parse(items[1:comma]), parse(items[comma+1:])))
            else:
                nested = items[1:]
                if nested and nested[0][0] == "whenever":
                    nested = (("when", *nested[0][1:]), *nested[1:])
                if nested and nested[0][0] == "never":
                    nested = (("not", *nested[0][1:]), *nested[1:])
                child = parse(nested)
            return make("global_constraint", items, (child,))
        if first == "never":
            child = make("negation", items, (parse(items[1:]),))
            return make("global_constraint", items, (child,))
        if first == "in":
            marker = next((i for i in top if items[i][0] == "global_response"), None)
            if marker is None or marker <= 1 or marker >= len(items) - 1:
                raise UnsupportedConstruct("unsupported_construct")
            condition = parse(items[1:marker])
            body = parse(items[marker + 1:])
            implication = make("implication", items, (condition, body))
            return make("global_constraint", items, (implication,))
        if first in {"if", "when"}:
            delimiters = [i for i in top if items[i][0] in {"then", ","}]
            if len(delimiters) != 1: raise UnsupportedConstruct("unresolved_axis")
            i = delimiters[0]
            return make("implication", items, (parse(items[1:i]), parse(items[i+1:])))
        whiles = [i for i in top if items[i][0] == "while"]
        if whiles:
            if len(whiles) != 1 or whiles[0] in {0, len(items) - 1}:
                raise UnsupportedConstruct("unsupported_semantics:while_scope")
            i = whiles[0]
            body_text = text[items[0][1]:items[i - 1][2]]
            guard_text = text[items[i + 1][1]:items[-1][2]]
            if not (
                re.search(r"\b(?:shall|must)\s+be\s+active\b", body_text, re.I)
                and re.search(r"\b(?:exists?|is|are|holds?)\b", guard_text, re.I)
            ):
                raise UnsupportedConstruct("unsupported_semantics:while_scope")
            # A source form ``body while state_guard`` is a global guarded
            # state obligation. Interval/termination readings remain outside
            # this deterministic form and continue through U/G review.
            body, guard = parse(items[:i]), parse(items[i + 1:])
            child = make("implication", items, (guard, body))
            return make("global_constraint", items, (child,))
        conditions = [i for i in top if items[i][0] in {"if", "when", "only if", "only when"}]
        if conditions:
            if len(conditions) != 1: raise UnsupportedConstruct("unresolved_axis")
            i = conditions[0]; left, right = parse(items[:i]), parse(items[i+1:])
            return make("implication", items, (left, right) if items[i][0] in {"only if", "only when"} else (right, left))
        orderings = [i for i in top if items[i][0] in {"before", "before_non_strict", "after"}]
        if orderings:
            if len(orderings) != 1:
                raise UnsupportedConstruct("unresolved_axis:ordering")
            i = orderings[0]
            left, right = parse(items[:i]), parse(items[i + 1:])
            if items[i][0] == "after":
                return make("after", items, (right, left))
            strict = items[i][0] == "before"
            return make(
                "precedence", items, (left, right),
                parameters=(
                    ("ordering", "strict" if strict else "non_strict"),
                    ("witness_policy", "required"),
                    ("same_step_policy", "forbidden" if strict else "allowed"),
                    ("reuse_policy", "reusable"),
                    ("scope_origin", "trace_start"),
                ),
            )
        logical_ops = [i for i in top if items[i][0] in {"and", "or", "until"}]
        logical_kinds = {items[i][0] for i in logical_ops}
        list_commas = [
            i for i in top
            if items[i][0] == ","
            and i + 1 < len(items)
            and items[i + 1][0] not in {"and", "or"}
        ]
        ops = list(logical_ops)
        if logical_kinds <= {"and", "or"} and len(logical_kinds) == 1:
            ops = sorted((*ops, *list_commas))
        if ops:
            kinds = logical_kinds
            if len(kinds) > 1 or ("until" in kinds and len(ops) > 1):
                raise UnsupportedConstruct("unresolved_axis")
            # Prefix temporal scope over unparenthesized coordination is ambiguous.
            if first in {"eventually", "next", "not"}:
                if first != "eventually" or temporal_coordination_scope is None:
                    raise UnsupportedConstruct(
                        "unresolved_axis:temporal_coordination_scope",
                        (items[0][1], items[-1][2]),
                    )
                cuts = [0, *ops, len(items)]
                operands = tuple(
                    parse(items[a + 1:b])
                    for a, b in zip(cuts, cuts[1:])
                )
                logical_kind = next(iter(kinds))
                kind = {"and": "conjunction", "or": "disjunction"}[logical_kind]
                if temporal_coordination_scope == "wide":
                    child = make(kind, items[1:], operands)
                    return make("eventually", items, (child,))
                if temporal_coordination_scope == "distributed":
                    wrapped = tuple(
                        make("eventually", (items[0], *items[a + 1:b]), (operand,))
                        for operand, (a, b) in zip(operands, zip(cuts, cuts[1:]))
                    )
                    return make(kind, items, wrapped)
                raise UnsupportedConstruct("unresolved_axis:temporal_coordination_scope")
            cuts = [-1, *ops, len(items)]
            children = tuple(parse(items[a+1:b]) for a, b in zip(cuts, cuts[1:]))
            logical_kind = next(iter(kinds))
            kind = {"and": "conjunction", "or": "disjunction", "until": "until"}[logical_kind]
            return make(kind, items, children)
        if first in {"eventually", "next", "not"}:
            return make({"eventually": "eventually", "next": "next_step", "not": "negation"}[first], items, (parse(items[1:]),))
        if len(items) == 1 and first in {"AP", "true", "false"}:
            return make("AP" if first == "AP" else first, items, identity=items[0][3])
        raise UnsupportedConstruct("unsupported_construct", (items[0][1], items[-1][2]))

    root = parse(tokens)
    graph = TypedSourceRelationGraph(
        text,
        hashlib.sha256(text.encode()).hexdigest(),
        ap_context.context_fingerprint,
        contract_hash,
        tuple(nodes),
        tuple(edges),
        root,
    )
    ledger = build_source_coverage_ledger(text, graph.nodes)
    return replace(
        graph,
        semantic_debts=ledger.blocking_debts,
        coverage_ledger=ledger,
    )


def parse_source_graph_set(
    text,
    ap_context,
    *,
    contract_hash="",
    allow_proposal_identities=False,
):
    """Return every explicit-grammar topology for one bounded source axis."""

    try:
        graph = parse_source_graph(
            text,
            ap_context,
            contract_hash=contract_hash,
            allow_proposal_identities=allow_proposal_identities,
        )
        return SourceIntentGraphSet(graph, (graph,))
    except UnsupportedConstruct as exc:
        if exc.code != "unresolved_axis:temporal_coordination_scope":
            raise
        variants = tuple(
            parse_source_graph(
                text,
                ap_context,
                contract_hash=contract_hash,
                allow_proposal_identities=allow_proposal_identities,
                temporal_coordination_scope=mode,
            )
            for mode in ("wide", "distributed")
        )
        span = exc.span if exc.span != (-1, -1) else (0, len(text))
        target_ids = tuple(graph.root_node_id for graph in variants)
        axis_id = _stable_id("sax-", variants[0].source_hash, "temporal_coordination_scope", span)
        claim_id = _stable_id("sclaim-", variants[0].source_hash, span, "eventually")
        axis = SourceIntentAxis(
            axis_id=axis_id,
            claim_id=claim_id,
            axis_kind="temporal_coordination_scope",
            source_span=span,
            evidence_text=text[span[0]:span[1]],
            options=(
                SourceAxisOption(
                    "wide_scope",
                    "eventually applies to the whole coordination",
                    "replace_node",
                    target_ids[0],
                    "wide",
                ),
                SourceAxisOption(
                    "distributed_scope",
                    "eventually applies independently to each coordinated item",
                    "replace_node",
                    target_ids[1],
                    "distributed",
                ),
            ),
        )
        variants = tuple(
            replace(
                graph,
                unresolved_axes=(axis.axis_id,),
                axes=(axis,),
                coverage_ledger=build_source_coverage_ledger(
                    text,
                    graph.nodes,
                    axes=(axis,),
                    debt_hints=graph.semantic_debts,
                ),
            )
            for graph in variants
        )
        return SourceIntentGraphSet(
            variants[0], variants, (axis,), candidate_selections=(
                ((axis.axis_id, "wide_scope"),),
                ((axis.axis_id, "distributed_scope"),),
            )
        )


def source_graph_from_proposal(
    payload,
    text,
    ap_context,
    *,
    contract_hash="",
):
    """Canonicalize a formula-free model proposal into the local graph schema."""

    ap_context.validate_source(text)
    if not isinstance(payload, Mapping):
        raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_INVALID")
    raw = payload.get("source_intent_graph", payload)
    if not isinstance(raw, Mapping):
        raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_INVALID")

    def contains_formula_field(value):
        if isinstance(value, Mapping):
            if any(
                str(key).casefold()
                in {"formula", "formula_text", "ltl", "ltlf", "reference_formula"}
                for key in value
            ):
                return True
            return any(contains_formula_field(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_formula_field(item) for item in value)
        return False

    if contains_formula_field(raw):
        raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_CONTAINS_FORMULA")
    if set(raw) - {"nodes", "root_ref", "semantic_debts"}:
        raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_INVALID")
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_INVALID")
    identities = tuple(
        ap_context.grounding_contract.identities
        if ap_context.grounding_contract is not None else ()
    )
    by_span = {
        tuple(mention.source_span): identity
        for identity in identities
        for mention in identity.mentions
    }
    by_name = {identity.canonical_name: identity for identity in identities}
    allowed = set(SOURCE_RELATION_ARITY)
    staged = []
    refs = set()
    for index, item in enumerate(raw_nodes):
        if not isinstance(item, Mapping):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_INVALID")
        if (
            not {"ref", "kind", "children"} <= set(item)
            or set(item) - {"ref", "kind", "source_span", "children", "parameters", "ap_name"}
        ):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_INVALID")
        ref = str(item.get("ref", f"proposal-{index}")).strip()
        kind = str(item.get("kind", "")).strip()
        span = item.get("source_span")
        if "ap_name" in item:
            identity = by_name.get(item["ap_name"]) if isinstance(item["ap_name"], str) else None
            if kind != "AP" or identity is None or not identity.mentions:
                raise UnsupportedConstruct("AP_ALIAS_UNRESOLVED")
            legal_spans = [tuple(mention.source_span) for mention in identity.mentions]
            span = tuple(span) if isinstance(span, (tuple, list)) and tuple(span) in legal_spans else legal_spans[0]
        children = item.get("children", [])
        parameters = item.get("parameters", {})
        if (
            not ref or ref in refs or kind not in allowed
            or not isinstance(span, (list, tuple)) or len(span) != 2
            or any(type(value) is not int for value in span)
            or not 0 <= span[0] < span[1] <= len(text)
            or not isinstance(children, list)
            or not isinstance(parameters, Mapping)
        ):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_INVALID")
        source_contract = SOURCE_NODE_CONTRACTS[kind]
        if set(parameters) - set(source_contract.parameters):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_INVALID")
        if not set(source_contract.required_parameters) <= set(parameters):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_INVALID")
        # Reuse the executable node registry for primitive types, enums and
        # cross-field invariants instead of maintaining a second validator.
        if kind not in {"AP", "true", "false"}:
            findings = node_contract_errors(
                SemanticNode(
                    node_id=ref,
                    node_type=source_contract.ir_node_type,
                    children=tuple(str(child) for child in children),
                    parameters=tuple(sorted(parameters.items())),
                ),
                schema_version=FAITHFUL_IR_SCHEMA_VERSION,
            )
            if findings:
                raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_PARAMETER_INVALID")
        if parameters:
            _validate_source_parameter_grounding(
                kind,
                parameters,
                text[span[0]:span[1]],
            )
        refs.add(ref)
        identity = by_span.get(tuple(span)) if kind == "AP" else None
        if kind == "AP" and identity is None:
            raise UnsupportedConstruct("AP_ALIAS_UNRESOLVED", tuple(span))
        staged.append((
            ref, kind, tuple(span), tuple(str(child) for child in children),
            identity, tuple(sorted(parameters.items())),
        ))
    if any(child not in refs for _ref, _kind, _span, children, _identity, _params in staged for child in children):
        raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_CHILD_REF_INVALID")
    root_ref = str(raw.get("root_ref", "")).strip()
    if root_ref not in refs:
        raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_ROOT_INVALID")
    ref_to_id = {}
    for ref, kind, span, children, identity, parameters in staged:
        ref_to_id[ref] = _stable_id(
            "sn-", hashlib.sha256(text.encode()).hexdigest(), kind, span,
            identity.identity_key if identity else "", parameters, ref,
        )
    if len(set(ref_to_id.values())) != len(ref_to_id):
        raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_NODE_COLLISION")
    nodes = tuple(
        SourceNode(
            ref_to_id[ref], kind, span, text[span[0]:span[1]],
            identity.identity_key if identity else "",
            identity.canonical_name if identity else "",
            parameters,
        )
        for ref, kind, span, _children, identity, parameters in staged
    )
    edges = tuple(
        SourceEdge(
            _stable_id("se-", ref_to_id[ref], kind, tuple(ref_to_id[c] for c in children), span),
            kind,
            (ref_to_id[ref],),
            tuple(ref_to_id[child] for child in children),
            span,
            confidence="proposed",
            parser_provenance="constrained-source-graph-proposal-v1",
        )
        for ref, kind, span, children, _identity, _parameters in staged if children
    )
    # Every nontrivial proposed node must satisfy the same arity vocabulary as
    # the deterministic graph before it can reach IR construction.
    edge_by_parent = {edge.from_ids[0]: edge for edge in edges}
    for node in nodes:
        count = len(edge_by_parent[node.node_id].to_ids) if node.node_id in edge_by_parent else 0
        minimum, maximum = SOURCE_RELATION_ARITY[node.kind]
        if count < minimum or (maximum is not None and count > maximum):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_ARITY_INVALID", node.source_span)
    children_by_ref = {
        ref: children for ref, _kind, _span, children, _identity, _parameters in staged
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(ref: str) -> None:
        if ref in visiting:
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_CYCLE")
        if ref in visited:
            return
        visiting.add(ref)
        for child in children_by_ref[ref]:
            visit(child)
        visiting.remove(ref)
        visited.add(ref)

    visit(root_ref)
    if visited != refs:
        raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_DISCONNECTED")
    root_id = ref_to_id[root_ref]
    root_node = next(node for node in nodes if node.node_id == root_id)
    if (root_node.kind == "implication"
            and re.match(r"\s*(?:whenever|every\s+time(?:\s+that|\s+when)?)\b", root_node.exact_text, re.I)):
        # The explicit repeated guard has the same global meaning in the local
        # grammar and the model graph. Complete only this source-anchored wrapper.
        global_id = _stable_id("sn-", hashlib.sha256(text.encode()).hexdigest(), "explicit-repeated-guard", root_id)
        nodes = (*nodes, SourceNode(global_id, "global_constraint", root_node.source_span, root_node.exact_text))
        edges = (*edges, SourceEdge(_stable_id("se-", global_id, root_id), "global_constraint",
                                  (global_id,), (root_id,), root_node.source_span,
                                  parser_provenance="explicit-repeated-guard-v1"))
        root_id = global_id
    debts = []
    for item in raw.get("semantic_debts", []):
        if not isinstance(item, Mapping):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_INVALID")
        if set(item) != {"reason_code", "source_span", "detail"}:
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_INVALID")
        span = tuple(item.get("source_span", (-1, -1)))
        if len(span) != 2 or any(type(v) is not int for v in span) or not 0 <= span[0] < span[1] <= len(text):
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_INVALID")
        code = str(item.get("reason_code", "unsupported_semantics")).strip()
        if code not in SOURCE_DEBT_REASON_CODES:
            raise UnsupportedConstruct("SOURCE_GRAPH_PROPOSAL_DEBT_REASON_INVALID")
        debt = SourceSemanticDebt(
            _stable_id("sdebt-", hashlib.sha256(text.encode()).hexdigest(), code, span, str(item.get("detail", ""))),
            code, span, text[span[0]:span[1]], str(item.get("detail", "")),
        )
        debts.append(debt)
    ledger = build_source_coverage_ledger(
        text,
        nodes,
        debt_hints=tuple(debts),
        ap_evidence_spans=tuple(mention.source_span for identity in identities
                               if identity.identity_key in {node.ap_identity_key for node in nodes if node.kind == "AP"}
                               for mention in identity.mentions),
    )
    graph = TypedSourceRelationGraph(
        text,
        hashlib.sha256(text.encode()).hexdigest(),
        ap_context.context_fingerprint,
        contract_hash,
        nodes,
        edges,
        root_id,
        semantic_debts=ledger.blocking_debts,
        coverage_ledger=ledger,
        proposal_mode="constrained_model_proposal",
    )
    return graph


def source_graph_proposal_payload(text, ap_context, *, contract=None, plain_reading_review=None):
    """Formula-blind payload for a constrained SourceIntentGraph proposal."""

    identities = []
    if ap_context.grounding_contract is not None:
        for identity in ap_context.grounding_contract.identities:
            identities.append({
                "canonical_name": identity.canonical_name,
                "mentions": [
                    {"source_span": list(mention.source_span), "evidence_text": mention.evidence_text}
                    for mention in identity.mentions
                ],
            })
    unresolved = []
    if plain_reading_review is not None:
        unresolved = [
            {
                "ambiguity_id": axis.ambiguity_id,
                "source_span": list(axis.source_span),
                "evidence_text": axis.evidence_text,
                "options": [
                    {"option_id": option.option_id, "plain_interpretation": option.plain_interpretation}
                    for option in axis.options
                ],
            }
            for axis in plain_reading_review.ambiguities
            if axis.decision_status == "ambiguous_unresolved"
        ]
    return {
        "task": "propose_source_intent_graph",
        "source_text": text,
        "source_hash": hashlib.sha256(text.encode()).hexdigest(),
        "ap_mentions": identities,
        "non_quotable_contract": contract or {},
        "unresolved_plain_reading_axes": unresolved,
        "allowed_relation_kinds": [
            *SOURCE_RELATION_ARITY,
        ],
        "relation_arity": {
            kind: {"minimum": minimum, "maximum": maximum}
            for kind, (minimum, maximum) in SOURCE_RELATION_ARITY.items()
        },
        "relation_parameter_contract": SOURCE_RELATION_PARAMETERS,
        "representation_rules": [
            REQUIREMENT_READING_RULES,
            "nodes must be a flat array",
            "children must contain only local ref strings, never nested node objects",
            "every child ref and root_ref must name an existing node",
            "the graph must be acyclic and every node must be reachable from root_ref",
            "use only allowed_relation_kinds and satisfy relation_arity",
            "For every AP node set ap_name to an exact supplied canonical_name. The backend fills its source_span from that identity; omit AP offsets rather than calculating them. Non-AP nodes still require exact source_span.",
            "the backend enumerates current-or-future versus strict-future eventuality candidates; these choices alone are not an unsupported capability",
            "semantic_debts must describe any remaining distinction precisely with an exact source span; return an empty array when none remains",
        ],
        "required_output": {
            "source_intent_graph": {
                "nodes": [{
                    "ref": "local reference", "kind": "allowed kind",
                    "source_span": [0, 1], "children": [],
                    "parameters": {},
                }],
                "root_ref": "one node ref",
                "semantic_debts": [{"reason_code": "unsupported_semantics", "source_span": [0, 1], "detail": "why"}],
            }
        },
        "ap_node_example": {"ref": "predicate", "kind": "AP", "ap_name": "one supplied canonical_name", "children": []},
        "forbidden": ["formula", "formula_text", "LTL", "LTLf", "reference_formula"],
    }


def source_axis_review_fingerprint(graph):
    """Cache key for source-axis review, excluding requirement/run identity."""

    payload = {
        "schema_version": graph.schema_version,
        "source_hash": graph.source_hash,
        "contract_hash": graph.contract_hash,
        "nodes": [
            {
                "kind": node.kind,
                "source_span": node.source_span,
                "exact_text": node.exact_text,
                "canonical_name": node.canonical_name,
                "parameters": node.parameters,
            }
            for node in sorted(graph.nodes, key=lambda item: (item.source_span, item.kind))
        ],
        "axes": [
            {
                "axis_kind": axis.axis_kind,
                "source_span": axis.source_span,
                "evidence_text": axis.evidence_text,
                "options": [(option.option_id, option.label, option.operation, option.value) for option in axis.options],
                "contract_fields": axis.contract_fields,
            }
            for axis in graph.axes
        ],
        "debts": [
            (debt.reason_code, debt.source_span, debt.evidence_text, debt.detail)
            for debt in graph.semantic_debts
        ],
    }
    return digest(payload)


__all__ = [
    "COVERAGE_CATEGORIES",
    "SOURCE_DEBT_REASON_CODES",
    "SOURCE_NODE_CONTRACTS",
    "SOURCE_RELATION_ARITY",
    "SOURCE_RELATION_PARAMETERS",
    "SourceAxisOption",
    "SourceEdge",
    "SourceIntentAxis",
    "SourceIntentGraphSet",
    "SourceNode",
    "SourceNodeContract",
    "SourceCoverageEntry",
    "SourceCoverageLedger",
    "SourceSemanticDebt",
    "TypedSourceRelationGraph",
    "UnsupportedConstruct",
    "build_source_coverage_ledger",
    "parse_source_graph",
    "parse_source_graph_set",
    "source_axis_review_fingerprint",
    "source_graph_from_proposal",
    "source_graph_proposal_payload",
    "source_graph_response_schema",
]
