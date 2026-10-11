"""Declare the prospective v2 repair by copying fictional v1 sources unchanged.

No model, archive, inference, auditor or Git access. Existing v1 is read only.
This builder must not be run to repair material after blind firstpasses.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT / 'audits/grounded_factual_auditor_qualification_v1/draft'
OUT = Path(__file__).resolve().parent
REPLACEMENTS = {
    'Q07-C4': "The current l1 memory slot's stored e0 proves that the same enduring physical object is currently at l1 in the world.",
    'Q10-C3': 'The supplied tables view omits command fields and location-keyed delivery maps.',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    (OUT / 'sources.json').write_bytes((V1 / 'sources.json').read_bytes())
    cases = json.loads((V1 / 'cases.json').read_text())
    cases['schema'] = 'grounded_factual_auditor_qualification_cases_v2_draft'
    for row in cases['claims']:
        if row['claim_id'] in REPLACEMENTS:
            row['text'] = REPLACEMENTS[row['claim_id']]
    key = json.loads((V1 / 'author_proposed_key.json').read_text())
    key['schema'] = 'grounded_factual_auditor_qualification_author_key_v2_draft'
    for row in key['answers']:
        if row['claim_id'] == 'Q07-C4':
            row['basis'] = ('Current l1 stores represented e0; enduring object identity and current physical world location are excluded. '
                            'The explicit current assertion remains current_snapshot and crosses represented memory into world truth.')
        elif row['claim_id'] == 'Q10-C3':
            row['content_class'] = 'absence_scope'
            row['basis'] = ('The supplied tables contain command fields and location-keyed delivery maps, explicitly refuting their omission. '
                            'This is supplied-field presence/omission at record scope, not an asserted inter-domain relation.')
    write('cases.json', cases)
    write('author_proposed_key.json', key)
    roles = {
        'sources': str((OUT / 'sources.json').relative_to(ROOT)),
        'cases': str((OUT / 'cases.json').relative_to(ROOT)),
        'author_key': str((OUT / 'author_proposed_key.json').relative_to(ROOT)),
        'protocol': 'docs/GROUNDED_FACTUAL_AUDITOR_QUALIFICATION_V2_PROTOCOL.md',
        'scorer': 'scripts/assess_grounded_factual_auditor_qualification_v2.py',
        'builder': str(Path(__file__).relative_to(ROOT)),
        'material_provenance': str((OUT / 'material_provenance.json').relative_to(ROOT)),
    }
    write('material_paths.json', roles)
    originals = {name: {'path': str((V1 / name).relative_to(ROOT)), 'sha256': sha(V1 / name)}
                 for name in ('sources.json', 'cases.json', 'author_proposed_key.json')}
    changes = []
    old_cases = json.loads((V1 / 'cases.json').read_text())
    old_key = json.loads((V1 / 'author_proposed_key.json').read_text())
    for old_rows, new_rows in ((old_cases['claims'], cases['claims']), (old_key['answers'], key['answers'])):
        for old, new in zip(old_rows, new_rows):
            for field in old:
                if old[field] != new[field]:
                    changes.append({'claim_id': old['claim_id'], 'field': field, 'old': old[field], 'new': new[field]})
    write('material_provenance.json', {
        'schema': 'grounded_factual_auditor_qualification_material_provenance_v2',
        'material_version': 'grounded_factual_auditor_qualification_v2',
        'status': 'prospective_draft_not_registered_not_qualification',
        'source_kind': 'twelve_entirely_fictional_v1_packets_copied_byte_identically',
        'inherited_source_schema': 'grounded_factual_auditor_qualification_sources_v1_draft',
        'source_sha256': sha(OUT / 'sources.json'), 'v1_material': originals,
        'claim_count': 48, 'packet_count': 12, 'inventory_units_per_packet': 21,
        'changed_claim_ids': list(REPLACEMENTS), 'changes': changes,
        'unchanged': ['all_twelve_source_packet_bytes', 'other_46_assertions', 'all_12_inventory_queries',
                      'all_252_inventory_cells', 'all_supportedness_labels', 'all_temporal_scope_labels',
                      'all_evidence_references', 'other_46_key_rows'],
        'label_changes': 'Only Q10-C3 content_class changes from cross_level to absence_scope; Q07-C4 already had the proposed v2 labels.',
        'key_independently_reviewed': False, 'auditors_qualified': False, 'reporter_run': False,
        'plan': 'docs/GROUNDED_FACTUAL_AUDITOR_QUALIFICATION_V2_PLAN.md',
    })


if __name__ == '__main__':
    main()
