"""Strict source-only boundary. This module never reads offline references."""
from __future__ import annotations

from dataclasses import dataclass
from .records import canonical, digest
import json
import unicodedata

FIELDS = frozenset({'qualified_id', 'requirement', 'ap_glossary', 'conventions', 'dataset_version', 'glossary_status'})
GLOSSARY_FIELDS = frozenset({'meaning', 'source', 'spans', 'surface'})
GLOSSARY_V2_FIELDS = GLOSSARY_FIELDS | frozenset({
    'inventory_schema_version', 'canonical_name', 'normalized_key',
    'identity_mentions', 'predicate_span', 'support_spans',
})
FORBIDDEN = frozenset({'reference', 'reference_formula', 'gold', 'gold_label', 'normalized_formula', 'template_signature', 'template_group', 'candidate_formula', 'candidates', 'survivors', 'outcome'})


def reject_hidden_fields(value):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.casefold()
            if lowered in FORBIDDEN or lowered.startswith(('reference_', 'gold_')):
                raise ValueError('forbidden online field: ' + key)
            reject_hidden_fields(child)
    elif isinstance(value, list):
        for child in value:
            reject_hidden_fields(child)


@dataclass(frozen=True)
class OnlineInput:
    # Store canonical bytes rather than a mutable nested dictionary.
    serialized: str

    @classmethod
    def from_mapping(cls, value):
        if type(value) is not dict or set(value) != FIELDS:
            raise ValueError('OnlineInput requires exactly the source whitelist; SyntheticCase is forbidden')
        reject_hidden_fields(value)
        if not isinstance(value['requirement'], str) or not value['requirement'].strip():
            raise ValueError('empty requirement')
        if not isinstance(value['qualified_id'], str) or ':' not in value['qualified_id']:
            raise ValueError('qualified ID required')
        if not isinstance(value['ap_glossary'], dict) or not isinstance(value['conventions'], dict):
            raise ValueError('invalid AP/policy mapping')
        normalized_names = [
            unicodedata.normalize('NFKC', str(ap)).strip().casefold()
            for ap in value['ap_glossary']
        ]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError('AP_CASEFOLD_COLLISION')
        for ap, entry in value['ap_glossary'].items():
            if not isinstance(entry, dict) or set(entry) not in {GLOSSARY_FIELDS, GLOSSARY_V2_FIELDS}:
                raise ValueError('invalid glossary entry: ' + ap)
            if set(entry) == GLOSSARY_V2_FIELDS:
                if entry['inventory_schema_version'] != 'ap-inventory-entry-v2':
                    raise ValueError('unsupported AP inventory entry version')
                if entry['canonical_name'] != ap:
                    raise ValueError('AP canonical_name must match its inventory key')
                expected_key = unicodedata.normalize('NFKC', ap).strip().casefold()
                if entry['normalized_key'] != expected_key:
                    raise ValueError('AP normalized_key mismatch')
                evidence_spans = [
                    entry['predicate_span'], *entry['identity_mentions'], *entry['support_spans']
                ]
                if tuple(entry['predicate_span']) not in {tuple(span) for span in entry['spans']}:
                    raise ValueError('AP predicate_span must be present in spans')
                for span in evidence_spans:
                    if len(span) != 2 or not all(type(n) is int for n in span) or not 0 <= span[0] < span[1] <= len(value['requirement']):
                        raise ValueError('invalid AP inventory evidence span')
            for span in entry['spans']:
                if len(span) != 2 or not all(type(n) is int for n in span) or not 0 <= span[0] < span[1] <= len(value['requirement']):
                    raise ValueError('invalid source span')
                if value['requirement'][span[0]:span[1]] != entry['surface']:
                    raise ValueError('glossary span does not match source')
        return cls(canonical(value))

    def payload(self):
        return json.loads(self.serialized)

    @property
    def input_hash(self):
        return digest(self.payload())


def commitment_payload(online, reviewed_structure, domains):
    # Caller must pass a dedicated structural projection, never a whole I1 record.
    reject_hidden_fields(reviewed_structure)
    reject_hidden_fields(domains)
    def check(value):
        if isinstance(value, dict):
            if any(k in value for k in ('traces', 'labels', 'reason', 'reasons', 'evidence', 'assignment')):
                raise ValueError('I2 commitment must precede evidence access')
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(reviewed_structure)
    check(domains)
    return {'input': online.payload(), 'reviewed_structure': reviewed_structure, 'domains': domains}
