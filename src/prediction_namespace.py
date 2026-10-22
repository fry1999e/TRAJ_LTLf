"""Input-only AP identity and source diagnostics; never inspect a reference."""
from ap_grounding import bind_source_quote, normalize_ap_key


def source_diagnostics(atoms, source, public_contract=None):
    rows = []
    definitions = {normalize_ap_key(k): normalize_ap_key(v) for k, v in (public_contract or {}).get('definitions', {}).items()}
    aliases = {normalize_ap_key(k): normalize_ap_key(v) for k,v in (public_contract or {}).get('aliases', {}).items()}
    conflicts = []
    for atom in atoms:
        fragments = atom.get('source_spans') or [
            {'text': atom.get('evidence_text', ''), 'span': atom.get('source_span')}]
        bindings, errors = [], []
        for fragment in fragments:
            try:
                recovered = bind_source_quote(source, fragment['text'], fragment.get('span'),
                                              occurrence=fragment.get('occurrence'))
                bindings.append({'original_span': fragment.get('span'), 'span': list(recovered),
                                 'text': source[slice(*recovered)]})
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
        rows.append({'name': atom['name'], 'bindings': bindings, 'errors': errors,
                     'identity_conflict': atom.get('identity_conflict')})
        key = aliases.get(normalize_ap_key(atom['name']), normalize_ap_key(atom['name']))
        if key in definitions and normalize_ap_key(atom.get('truth_definition', '')) != definitions[key]:
            conflicts.append({'name': atom['name'], 'reason': 'public_definition_not_identical_requires_semantic_verification'})
    return {'source_binding_status': 'bound' if rows and all(r['bindings'] and not r['errors'] for r in rows) else 'unverified',
            'source_bindings': rows, 'namespace_conflicts': conflicts}


def prediction_mapping(contract, source_input_hash):
    """Accept only an input-side contract frozen before any formula was built."""
    if not contract:
        return {}, 'unverified'
    if (contract.get('origin') != 'dataset_input'
            or contract.get('source_input_hash') != source_input_hash
            or contract.get('frozen_before_generation') is not True):
        raise ValueError('prediction namespace contract lacks input-side provenance')
    definitions = contract.get('definitions', {})
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError('public AP definitions missing')
    vocabulary = {normalize_ap_key(k): v for k, v in definitions.items()}
    if len(vocabulary) != len(definitions) or any(not isinstance(v, str) or not v.strip() for v in vocabulary.values()):
        raise ValueError('invalid public AP definitions')
    aliases = contract.get('aliases', {})
    if not isinstance(aliases, dict):
        raise ValueError('invalid prediction aliases')
    mapping = {k: k for k in vocabulary}
    for alias, target in aliases.items():
        key, value = normalize_ap_key(alias), normalize_ap_key(target)
        if value not in vocabulary or (key in mapping and mapping[key] != value):
            raise ValueError('conflicting prediction namespace alias')
        mapping[key] = value
    return mapping, 'confirmed'
