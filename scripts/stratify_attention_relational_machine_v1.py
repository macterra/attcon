"""Post-development descriptive origin counts for preserved machine facts only.

Written after narrator collection and its first machine assessment. This neither
assesses prose fidelity nor changes any registered endpoint or qualification rule.
It reconstructs covered slots from the frozen auditor's omission inventory; it
does not independently rerun that auditor against the raw response collection.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import attention_relational_report_v1 as exporter
import attention_relational_reporting_pipeline_v1 as pipeline
import inventory_attention_relational_origins_v1 as origins


MACHINE_SHA256 = '2355fae6b54317293d99c29c737c470c8ecd1b8bd012313dfa4f1cce5b5a3f97'
FIELDS = ('panel_source', 'underlying_source', 'origin_category', 'view', 'origin_kind', 'terminal_origin_kind')


def origin_key(prop, contexts, panel_sources):
    node = contexts[tuple(prop[k] for k in ('panel', 'view', 'command', 'node'))]
    origin = terminal = node['origin']
    category = origins.classify(origin)
    while terminal['kind'] == 'inherited_current':
        terminal = terminal['inherited_origin']
    source = panel_sources[prop['panel']]
    if category in ('source_current', 'source_conditional', 'inherited_source_current'):
        pipeline.require(source in ('A', 'ordinary', 'analytic'), 'estimate has invalid panel source')
        underlying = source
    else:
        underlying = 'sensor' if category == 'sensor' else 'W'
        pipeline.require((category == 'copied_W11' and source in ('A', 'ordinary', 'analytic'))
                         or (category == 'native_W' and source == 'W')
                         or (category == 'sensor' and source == 'sensor'), 'origin/provenance mismatch')
    return source, underlying, category, prop['view'], origin['kind'], terminal['kind']


def table(counts):
    return [dict(zip(FIELDS, key), **dict(values)) for key, values in sorted(counts.items())]


def stratify(exports, assessment):
    inventory = pipeline.validate_export(exports)
    pipeline.require(assessment['schema'] == 'attention_relational_report_v1_first_machine_factual_assessment'
                     and assessment['input_dependencies']['exports']['sha256'] == origins.EXPORT_SHA256,
                     'machine assessment has different export lineage')
    audits = pipeline._index(assessment['machine_audits'])
    pipeline.require(list(audits) == inventory['request_ids'] and len(audits) == 552, 'fixed552 IDs differ')
    aggregate, rows = defaultdict(Counter), []
    for identifier, row in audits.items():
        content = inventory['packets'][identifier]['content']
        provenance = inventory['provenance'][identifier]
        pipeline.require(row['case'] == provenance['case'] and type(row['transport_complete']) is bool,
                         'case or transport metadata differs')
        panel_sources = {p['panel']: p['source_key'] for p in provenance['panels']}
        pipeline.require(len(panel_sources) == len(provenance['panels'])
                         and set(panel_sources) == {p['panel'] for p in content['panels']}, 'panel alignment differs')
        required = {exporter._slot(p): p for p in exporter.required_propositions(content)}
        audit = row['audit']
        pipeline.require(audit['schema'] == 'attention_relational_report_v1_machine_factual_audit'
                         and audit['required_peak_count'] == len(required)
                         and audit['error_count'] == len(audit['errors'])
                         and audit['listed_machine_facts_pass'] == (not audit['errors']), 'primary audit inconsistent')
        omitted = set()
        for error in audit['errors']:
            if error['category'] == 'omitted_required_peak_set':
                expected = error['expected']
                slot = exporter._slot(expected)
                pipeline.require(slot in required and expected == required[slot] and slot not in omitted,
                                 'omitted expected atom is incorrect or duplicated')
                omitted.add(slot)
        pipeline.require(len(required) - len(omitted) == audit['covered_peak_count'], 'covered slot count differs')
        contexts = pipeline._public_contexts(content)
        counts = defaultdict(Counter)
        for slot, prop in required.items():
            covered, complete = slot not in omitted, row['transport_complete']
            values = {'required': 1, 'covered': int(covered), 'omitted': int(not covered),
                      'covered_in_complete_responses': int(covered and complete),
                      'unestablished_due_incomplete_transport': int(not complete)}
            key = origin_key(prop, contexts, panel_sources)
            counts[key].update(values)
            aggregate[key].update(values)
        rows.append({'id': identifier, 'case': deepcopy(row['case']),
                     'transport_complete': row['transport_complete'], 'primary_audit': deepcopy(audit),
                     'origin_counts': table(counts)})
    totals = Counter()
    for values in aggregate.values():
        totals.update(values)
    pipeline.require(totals['required'] == 25440 and totals['covered'] == 24022
                     and totals['required'] == totals['covered'] + totals['omitted'], 'complete totals differ')
    pipeline.require(assessment['totals']['required_peak_atoms'] == totals['required']
                     and assessment['totals']['covered_peak_atoms'] == totals['covered']
                     and assessment['totals']['covered_peak_atoms_in_complete_responses']
                     == totals['covered_in_complete_responses'], 'primary aggregate differs')
    pipeline.require(sum(v['required'] for k, v in aggregate.items() if k[1] == 'A') == 2100,
                     'attention-derived required denominator differs')
    return {'schema': 'attention_relational_report_v1_descriptive_machine_origin_counts',
            'chronology_and_limits': __doc__, 'unit': 'required role/distribution peak-set opportunity',
            'count_definitions': {'covered': 'Original machine-audit coverage, including incomplete responses',
                'covered_in_complete_responses': 'Covered atoms whose response transport completed',
                'unestablished_due_incomplete_transport': 'All required atoms of incomplete attempts; overlaps covered'},
            'opportunities': 552, 'totals': dict(totals), 'origin_counts': table(aggregate), 'reports': rows,
            'atom_coverage_is_not_report_fidelity': True,
            'incomplete_attempt_coverage_is_unestablished': True,
            'changes_primary_report_pass_or_registered_endpoints': False,
            'independent_observations_claimed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('exports', 'machine_audits', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing output replacement')
    exports = pipeline.bound_json(args.exports, origins.EXPORT_SHA256)
    assessment = pipeline.bound_json(args.machine_audits, MACHINE_SHA256)
    sources = {str(Path(m.__file__)): pipeline.digest(Path(m.__file__).read_bytes())
               for m in (pipeline, exporter, origins)}
    pipeline.require(all(sources.get(path) == sha for path, sha in assessment['source_sha256'].items()),
                     'primary assessment frozen source bytes differ')
    wrapper = Path(__file__).with_name('audit_attention_relational_machine_v1.py')
    pipeline.require(pipeline.digest(wrapper.read_bytes()) == assessment['wrapper_sha256'], 'primary wrapper differs')
    result = stratify(exports, assessment)
    result['input_dependencies'] = {
        'exports': {'path': str(args.exports), 'sha256': origins.EXPORT_SHA256},
        'machine_audits': {'path': str(args.machine_audits), 'sha256': MACHINE_SHA256}}
    sources[str(wrapper)] = assessment['wrapper_sha256']
    sources[str(Path(__file__))] = pipeline.digest(Path(__file__).read_bytes())
    result['source_sha256'] = sources
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
