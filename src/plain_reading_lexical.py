"""Backend-owned lexical facts for Plain Reading validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re

from ap_grounding import APGroundingContract
from contract_versions import PLAIN_READING_LEXICAL_CONTRACT_VERSION


_DIGIT_QUANTITY_RE = re.compile(
    r"(?<!\w)(\d+(?:\.\d+)?)(?:st|nd|rd|th)?(?!\w)",
    re.IGNORECASE,
)
_QUANTITY_WORD_VALUES = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "eleventh": "11",
    "twelfth": "12",
    "thirteenth": "13",
    "fourteenth": "14",
    "fifteenth": "15",
    "sixteenth": "16",
    "seventeenth": "17",
    "eighteenth": "18",
    "nineteenth": "19",
    "twentieth": "20",
}
_QUANTITY_WORD_RE = re.compile(
    r"\b(?:"
    + "|".join(sorted(_QUANTITY_WORD_VALUES, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|without|cannot|can't|doesn't|isn't|aren't|"
    r"mustn't|shouldn't|won't)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlainReadingLexicalContract:
    source_hash: str
    grounding_fingerprint: str
    ap_identity_ids: tuple[str, ...]
    ap_identifiers: tuple[str, ...]
    quantity_evidence: tuple[tuple[str, str], ...]
    explicit_negation_terms: tuple[str, ...]
    contract_version: str = PLAIN_READING_LEXICAL_CONTRACT_VERSION
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.contract_version != PLAIN_READING_LEXICAL_CONTRACT_VERSION:
            raise ValueError("unsupported Plain Reading lexical contract")
        expected = _fingerprint(self)
        if self.fingerprint != expected:
            object.__setattr__(self, "fingerprint", expected)


def build_plain_reading_lexical_contract(
    source_text: str,
    grounding: APGroundingContract,
) -> PlainReadingLexicalContract:
    """Project the sole AP grounding authority into lexical validation facts."""

    grounding.validate_source(source_text)
    quantities = [
        (match.group(0), _canonical_number(match.group(1)))
        for match in _DIGIT_QUANTITY_RE.finditer(source_text)
    ]
    quantities.extend(
        (match.group(0), _QUANTITY_WORD_VALUES[match.group(0).casefold()])
        for match in _QUANTITY_WORD_RE.finditer(source_text)
    )
    negations = tuple(
        dict.fromkeys(match.group(0).casefold() for match in _NEGATION_RE.finditer(source_text))
    )
    return PlainReadingLexicalContract(
        source_hash=grounding.source_hash,
        grounding_fingerprint=grounding.fingerprint,
        ap_identity_ids=tuple(
            identity.identity_key for identity in grounding.identities
        ),
        ap_identifiers=tuple(
            identity.canonical_name for identity in grounding.identities
        ),
        quantity_evidence=tuple(quantities),
        explicit_negation_terms=negations,
    )


def _canonical_number(value: str) -> str:
    if "." not in value:
        return str(int(value))
    whole, fraction = value.split(".", 1)
    fraction = fraction.rstrip("0")
    return str(int(whole)) + (f".{fraction}" if fraction else "")


def _fingerprint(contract: PlainReadingLexicalContract) -> str:
    payload = asdict(contract)
    payload["fingerprint"] = ""
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "PlainReadingLexicalContract",
    "build_plain_reading_lexical_contract",
]
