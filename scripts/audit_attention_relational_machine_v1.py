"""Apply the frozen deterministic proposition auditor to every first response.

This is partial factual assessment only; it does not assess narrative fidelity,
reference correspondence, experience, or the source-of-qualia hypothesis.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

import attention_relational_reporting_pipeline_v1 as pipeline
import attention_relational_report_v1 as exporter
from inventory_attention_relational_origins_v1 import EXPORT_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('exports', 'reports'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing output replacement')
    pipeline.require(args.exports_sha256 == EXPORT_SHA256, 'not the preserved first export')
    exports = pipeline.bound_json(args.exports, args.exports_sha256)
    reports = pipeline.bound_json(args.reports, args.reports_sha256)
    inventory = pipeline.validate_export(exports)
    pipeline._validate_collection_lineage(reports, 'report', exports['requests'], exports)
    received = pipeline._records(reports)
    rows, totals, groups = [], Counter(), defaultdict(Counter)
    for identifier in inventory['request_ids']:
        record = received.get(identifier)
        audit = exporter.audit_report(inventory['packets'][identifier]['content'],
                                      record.get('parsed') if record else None)
        complete = bool(record and record.get('transport_complete'))
        case = inventory['provenance'][identifier]['case']
        counts = dict(opportunities=1, transport_complete=int(complete),
            literal_machine_facts_pass=int(audit['listed_machine_facts_pass']),
            transport_complete_and_machine_facts_pass=int(complete and audit['listed_machine_facts_pass']),
            required_peak_atoms=audit['required_peak_count'],
            covered_peak_atoms=audit['covered_peak_count'],
            covered_peak_atoms_in_complete_responses=audit['covered_peak_count'] if complete else 0)
        totals.update(counts)
        groups[(case['source'], case['view'])].update(counts)
        for error in audit['errors']:
            totals['error_' + error['category']] += 1
            groups[(case['source'], case['view'])]['error_' + error['category']] += 1
        rows.append(dict(id=identifier, case=case, transport_complete=complete, audit=audit))
    pipeline.require(totals['opportunities'] == 552 and totals['required_peak_atoms'] == 25440,
                     'fixed inventory differs')
    result = {'schema': 'attention_relational_report_v1_first_machine_factual_assessment',
        'partial_assessment_only': True, 'free_prose_fidelity_assessed': False,
        'reference_correspondence_assessed': False, 'experience_established': False,
        'input_dependencies': {k: {'path': str(getattr(args, k)), 'sha256': getattr(args, k + '_sha256')}
                               for k in ('exports', 'reports')},
        'source_sha256': {str(Path(m.__file__)): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                          for m in (pipeline, exporter)},
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'totals': dict(totals),
        'groups': [dict(source=k[0], view=k[1], **v) for k, v in sorted(groups.items())],
        'machine_audits': rows}
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
