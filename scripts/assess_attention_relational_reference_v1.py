"""Assess the already-frozen raw-reference endpoint on all original1104 positions.

Original rater interruption remains explicit. This reads no prose-auditor data
and emits no fidelity-conjoined results. The later full primary join remains
authoritative; its raw-reference projection must match these unchanged gates.
Import is inert: no API/model calls, credentials, or scientific output reads.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import attention_relational_origin_attribution_v1 as attribution
import attention_relational_report_v1 as exporter
import attention_relational_reporting_pipeline_v1 as pipeline
import inventory_attention_relational_origins_v1 as origins


HASHES = {'exports': '4f5f0bc2fa78b3b24b3580f52e306f20df0f3011d079b8139a2bc1f029050837',
          'reports': '80e07b465a4ac95832585be9594973ead12ba1b830f3f517cb250d0891ca6940',
          'ratings': '991cc2c5e07dbeb309e01501efc6397bea4d918e01935f64a8b5a7e540d796d0'}


def assess(exports, reports, ratings):
    inventory = pipeline.validate_export(exports)
    for collection in (reports, ratings):
        attribution.validate_collected_records(collection)
    pipeline._validate_collection_lineage(reports, 'report', exports['requests'], exports)
    prepared = pipeline.build_blind_rater_requests(reports, exports['templates'])
    pipeline._validate_collection_lineage(ratings, 'rate', prepared, exports, reports)
    report_records, rating_records = pipeline._records(reports), pipeline._records(ratings)
    pipeline.require(ratings.get('interrupted_batch') is True and ratings['request_count'] == 1104
                     and set(rating_records) == {r['id'] for r in prepared}, 'original interrupted inventory differs')
    pipeline.require(set(report_records) == set(inventory['request_ids']), 'fixed narrator inventory differs')
    rows, groups = [], defaultdict(Counter)
    for identifier in inventory['request_ids']:
        text = pipeline._report_text(report_records.get(identifier))
        case = inventory['provenance'][identifier]['case']
        for template in (False, True):
            text_id = identifier + ('_template' if template else '')
            exact_text = inventory['templates'][identifier]['report']['text'] if template else text
            unit_id = pipeline.blind_id(text_id)
            received = rating_records.get(unit_id)
            annotation = received.get('parsed') if received else None
            validation = pipeline.validate_blind_annotation(exact_text, unit_id, annotation)
            complete = bool(received and received.get('transport_complete'))
            annotation_valid = complete and validation['valid']
            text_available = template or bool(text.strip())
            rating_valid = annotation_valid and text_available
            arm = 'template' if template else 'narrator'
            group = '|'.join(str(case[k]) for k in ('encoder_seed', 'source', 'view', 'variant')) + '|' + arm
            count = groups[group]
            count.update(text_opportunities=1, rating_transport_complete=int(complete),
                reference_measurement_valid_before_actor_text_gate=int(annotation_valid), valid_rating=int(rating_valid),
                reference_outcome_unestablished_due_actor_text=int(not text_available),
                rating_invalid=int(complete and not validation['valid']), rating_missing_or_incomplete=int(not complete))
            for diagnostic in validation.get('evidence_diagnostics', []):
                count['rating_' + diagnostic['kind']] += 1
            features = {name: annotation['features'][name]['status'] if rating_valid else 'unestablished'
                        for name in ('C', 'O', 'D', 'A')}
            auxiliary = {name: annotation['auxiliary'][name]['status'] if rating_valid else 'unestablished'
                         for name in pipeline.AUXILIARY}
            for name, status in list(features.items()) + list(auxiliary.items()):
                count['feature_' + name + '_' + status] += 1
            endpoints = {}
            for endpoint in ('COD', 'CA'):
                obj = annotation['composites'][endpoint] if rating_valid else None
                status = obj['status'] if obj else 'unestablished'
                lower, upper = status == 'explicit', status in ('explicit', 'ambiguous')
                scopes = sorted({b['scope'] for b in obj['bindings']}) if obj else []
                times = sorted({b['time'] for b in obj['bindings']}) if obj else []
                count[endpoint + '_lower'] += int(lower)
                count[endpoint + '_upper'] += int(upper)
                count[endpoint + '_status_' + status] += 1
                for scope in pipeline.SCOPES:
                    count[endpoint + '_' + scope + '_lower'] += int(lower and scope in scopes)
                    count[endpoint + '_' + scope + '_upper'] += int(upper and scope in scopes)
                endpoints[endpoint] = dict(status=status, lower=lower, upper=upper, scopes=scopes, times=times)
            rows.append({'id': text_id, 'unit_id': unit_id, 'case': case, 'template': template,
                'text_sha256': pipeline.digest(exact_text.encode('utf-8')), 'actor_text_available': text_available,
                'rating_transport_complete': complete, 'rating_validation': validation,
                'reference_measurement_valid_before_actor_text_gate': annotation_valid,
                'rating_availability': received.get('availability') if received else 'missing',
                'rating_api_status': received.get('api_status') if received else None,
                'reported_composite_labels_before_validation': {name: annotation['composites'][name].get('status')
                    for name in ('COD', 'CA') if type(annotation) is dict
                    and type(annotation.get('composites')) is dict and type(annotation['composites'].get(name)) is dict},
                'reported_rater_limitations': annotation.get('limitations') if type(annotation) is dict else None,
                'features': features, 'auxiliary': auxiliary, 'endpoints': endpoints})
    totals, arms = Counter(), {'narrator': Counter(), 'template': Counter()}
    for key, count in groups.items():
        totals.update(count)
        arms[key.rsplit('|', 1)[1]].update(count)
    pipeline.require(len(rows) == totals['text_opportunities'] == 1104
                     and all(v['text_opportunities'] == 552 for v in arms.values())
                     and totals['rating_transport_complete'] == 83, 'fixed original reference accounting differs')
    rates = {key: dict(text_denominator=count['text_opportunities'],
                      valid_rating_rate=count['valid_rating'] / count['text_opportunities'],
                      **{endpoint + '_' + bound + '_rate': count[endpoint + '_' + bound] / count['text_opportunities']
                         for endpoint in ('COD', 'CA') for bound in ('lower', 'upper')})
             for key, count in groups.items()}
    return {'schema': 'attention_relational_report_v1_original_raw_reference_assessment',
        'scope_and_limits': __doc__, 'original_rater_interrupted': True,
        'opportunities': {'archive_cases': 12, 'narrator': 552, 'templates': 552, 'ratings': 1104},
        'totals': dict(totals), 'arm_totals': {k: dict(v) for k, v in arms.items()},
        'groups': {k: dict(v) for k, v in groups.items()}, 'group_rates': rates, 'all_text_results': rows,
        'reference_sha256': pipeline.REFERENCE_HASH, 'prose_auditor_data_read': False,
        'fidelity_conjoined_outputs_emitted': False, 'full_primary_join_pending': True,
        'criteria_changed': False, 'no_api_or_model_calls': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in HASHES:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing replacement of first raw-reference assessment')
    inputs = {name: pipeline.bound_json(getattr(args, name), sha) for name, sha in HASHES.items()}
    result = assess(**inputs)
    result['input_dependencies'] = {name: {'path': str(getattr(args, name)), 'sha256': sha}
                                    for name, sha in HASHES.items()}
    result['source_sha256'] = {str(Path(m.__file__)): pipeline.digest(Path(m.__file__).read_bytes())
                               for m in (pipeline, exporter, attribution, origins)}
    result['source_sha256'][str(Path(__file__))] = pipeline.digest(Path(__file__).read_bytes())
    result['criterion_sources_sha256'] = {path: pipeline.digest((pipeline.ROOT / path).read_bytes())
        for path in (pipeline.REFERENCE, pipeline.RATER_INSTRUCTIONS, pipeline.PROTOCOL)}
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
