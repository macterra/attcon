"""Prospective supplementary V3 preparation, raw collection, derivation and join.

No execution client is implemented. Generic once-only runner execution requires
a separately committed/reviewed runtime manifest and fictional acceptance.
Old raw ratings and original primary summaries remain separate and unchanged.
Integration follows the earlier published V3 auditor preselection unchanged,
although original endpoint outcomes were exposed during this implementation.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import attention_relational_origin_attribution_v1 as attribution
import attention_relational_quote_reference_v3 as quotes
import attention_relational_report_v1 as exporter
import attention_relational_reporting_pipeline_v1 as pipeline
import inventory_attention_relational_origins_v1 as origins
import run_attention_relational_reports_v1 as runner


BASE = 'audits/attention_relational_report_v1/quote_reference_v3'
INSTRUCTIONS = BASE + '/rater_instructions.txt'
RATE_ROOT = BASE + '/rate'
AUDITOR_MANIFEST = 'audits/attention_relational_report_v1/resource_measurement_v2/audit_preparation/runtime_manifest.json'
AUDITOR_MANIFEST_SHA256 = '973b6d64c76b6dc46468f0185a182a210176c6e756caa43a5b7800ea89ddb44a'
AUDITOR_ROOT = 'audits/attention_relational_report_v1/resource_measurement_v2/audit'
REPORTS_SHA256 = '80e07b465a4ac95832585be9594973ead12ba1b830f3f517cb250d0891ca6940'
OLD_RATINGS_SHA256 = '991cc2c5e07dbeb309e01501efc6397bea4d918e01935f64a8b5a7e540d796d0'
VERSION = 'attention_relational_quote_measurement_v3'


def digest(value):
    return pipeline.digest(pipeline.encoded(value))


def settings():
    return dict(model='gpt-5-mini-2025-08-07', reasoning_effort='medium', max_output_tokens=32768,
                timeout_seconds=600, concurrency=6, output_schema=quotes.response_schema())


def instructions():
    text = pipeline.frozen_instruction('rate')
    edits = [
        ('Record each evidence substring exactly as it appears in report_text, with start/end Unicode-codepoint offsets, end exclusive.',
         'Record each evidence substring exactly as it appears in report_text.'),
        (pipeline.RATING_SCHEMA, quotes.QUOTE_SCHEMA),
        ('"start":0,"end":1,', ''),
        ('The program must validate offsets, schema and evidence consistency;',
         'The program must ground literal quotations and validate schema and evidence consistency;')]
    for before, after in edits:
        pipeline.require(text.count(before) == 1, 'frozen instruction edit anchor differs')
        text = text.replace(before, after)
    pipeline.require((pipeline.ROOT / INSTRUCTIONS).read_text() == text + '\n',
                     'V3 instructions differ beyond the declared mechanical edits')
    return text


def text_units(inventory, reports):
    records, units = pipeline._records(reports), []
    for identifier in inventory['request_ids']:
        for template in (False, True):
            text_id = identifier + ('_template' if template else '')
            text = (inventory['templates'][identifier]['report']['text'] if template
                    else pipeline._report_text(records.get(identifier)))
            units.append(dict(id=text_id, unit_id=pipeline.blind_id(text_id), template=template,
                case=deepcopy(inventory['provenance'][identifier]['case']), text=text,
                text_sha256=pipeline.digest(text.encode('utf-8')),
                actor_text_available=template or bool(text.strip())))
    return units


def prepare_units(units, reference, instruction, selection, inputs, source_hashes):
    """Pure new-instrument assembly; production callers first validate old inputs."""
    pipeline.require(len(units) == 1104 and len({u['id'] for u in units}) == 1104
                     and len({u['unit_id'] for u in units}) == 1104
                     and sum(u['template'] for u in units) == 552, 'fixed1104 text inventory differs')
    for unit in units:
        pipeline.require(unit['unit_id'] == pipeline.blind_id(unit['id'])
                         and unit['text_sha256'] == pipeline.digest(unit['text'].encode('utf-8')),
                         'text/opaque-ID binding differs')
    requests = [dict(id=u['unit_id'], input=instruction + '\n' + pipeline.encoded(
        dict(reference=reference, unit_id=u['unit_id'], report_text=u['text'])).decode('ascii')) for u in units]
    requests.sort(key=lambda r: pipeline.digest((pipeline.ORDER_SALT + ':' + r['id']).encode('ascii')))
    lineage = dict(schema=VERSION + '_lineage', inputs=deepcopy(inputs), reference_sha256=digest(reference),
        instructions_content_sha256=pipeline.digest(instruction.encode('utf-8')),
        response_schema_content_sha256=digest(quotes.response_schema()), requests_content_sha256=digest(requests),
        units_content_sha256=digest(units), selected_factual_auditor=deepcopy(selection),
        source_sha256=deepcopy(source_hashes))
    return dict(schema=VERSION + '_preparation', instrument_scope_and_chronology=__doc__, units=deepcopy(units), requests=requests,
                lineage=lineage, settings=settings(), output_root=RATE_ROOT)


def source_hashes():
    files = [Path(m.__file__) for m in (pipeline, quotes, exporter, attribution, origins, runner)] + [Path(__file__)]
    files += [pipeline.ROOT / p for p in (INSTRUCTIONS, pipeline.REFERENCE, pipeline.RATER_INSTRUCTIONS, pipeline.PROTOCOL,
        'docs/ATTENTION_RELATIONAL_QUOTE_REFERENCE_V3_DESIGN.md',
        'tests/test_attention_relational_quote_measurement_v3.py', 'tests/test_attention_relational_quote_reference_v3.py')]
    return {str(p.resolve().relative_to(pipeline.ROOT)): pipeline.digest(p.read_bytes()) for p in files}


def prepare(exports, reports, inputs):
    pipeline.require(inputs['exports']['sha256'] == origins.EXPORT_SHA256
                     and inputs['reports']['sha256'] == REPORTS_SHA256, 'not the fixed source inputs')
    inventory = pipeline.validate_export(exports)
    attribution.validate_collected_records(reports)
    pipeline._validate_collection_lineage(reports, 'report', exports['requests'], exports)
    raw = (pipeline.ROOT / AUDITOR_MANIFEST).read_bytes()
    pipeline.require(pipeline.digest(raw) == AUDITOR_MANIFEST_SHA256, 'preselected auditor manifest bytes differ')
    selected = pipeline.decoded(raw)
    pipeline.require(selected['stage'] == 'audit' and selected['output_root'] == AUDITOR_ROOT
                     and selected['request_count'] == 537, 'whole preselected resource-v2 auditor differs')
    selection = dict(manifest_path=AUDITOR_MANIFEST, manifest_sha256=pipeline.digest(raw),
                     output_root=AUDITOR_ROOT, request_count=537)
    return prepare_units(text_units(inventory, reports), pipeline.frozen_reference(), instructions(),
                         selection, inputs, source_hashes())


def load_preparation(path, sha):
    pipeline.require(path is not None and sha is not None, 'bound V3 preparation required')
    prepared = pipeline.bound_json(path, sha)
    inputs = prepared['lineage']['inputs']
    original = {name: pipeline.bound_json(**dict(path=row['path'], expected_sha256=row['sha256']))
                for name, row in inputs.items()}
    pipeline.require(prepared == prepare(**original, inputs=inputs), 'prepared V3 bytes/content lineage differ')
    return prepared


def validate_prepared(prepared):
    lineage = prepared['lineage']
    pipeline.require(prepared['schema'] == VERSION + '_preparation' and lineage['schema'] == VERSION + '_lineage'
                     and prepared['settings'] == settings() and prepared['output_root'] == RATE_ROOT
                     and digest(prepared['units']) == lineage['units_content_sha256']
                     and digest(prepared['requests']) == lineage['requests_content_sha256'], 'V3 preparation content differs')
    requests = pipeline._index(prepared['requests'])
    pipeline.require(len(requests) == len(prepared['units']) == 1104
                     and set(requests) == {u['unit_id'] for u in prepared['units']}, 'V3 request/unit inventory differs')
    for unit in prepared['units']:
        prefix, payload = requests[unit['unit_id']]['input'].rsplit('\n', 1)
        payload = pipeline.decoded(payload)
        pipeline.require(set(payload) == {'reference', 'unit_id', 'report_text'}
                         and unit['unit_id'] == pipeline.blind_id(unit['id']) == payload['unit_id']
                         and unit['text'] == payload['report_text']
                         and unit['actor_text_available'] == (unit['template'] or bool(unit['text'].strip()))
                         and unit['text_sha256'] == pipeline.digest(unit['text'].encode('utf-8'))
                         and digest(payload['reference']) == lineage['reference_sha256']
                         and pipeline.digest(prefix.encode('utf-8')) == lineage['instructions_content_sha256'],
                         'actual V3 request/text/reference binding differs')


def build_binding(manifest, manifest_sha, requests, requests_sha, lineage, lineage_sha, prepared):
    validate_prepared(prepared)
    pipeline.require(manifest['stage'] == 'rate' and manifest['settings'] == prepared['settings']
                     and manifest['output_root'] == RATE_ROOT and requests == prepared['requests']
                     and manifest['request_ids'] == [r['id'] for r in requests]
                     and manifest['request_count'] == 1104 and lineage == prepared['lineage'], 'actual V3 runtime differs')
    pipeline.require(manifest['source_sha256'].get(manifest['requests_path']) == requests_sha
                     and manifest['source_sha256'].get(manifest['lineage_path']) == lineage_sha,
                     'actual V3 artifact byte hashes differ')
    # This is the generic raw-retention binding, with the actual V3 lineage and requests.
    return dict(schema='attention_relational_report_v1_collection_binding', stage='rate',
        measurement_version=VERSION, manifest_sha256=manifest_sha, requests_sha256=requests_sha,
        lineage_sha256=lineage_sha, request_ids=manifest['request_ids'], requests=deepcopy(requests),
        lineage=deepcopy(lineage), output_root=RATE_ROOT, preparation_content_sha256=digest(prepared))


def derive_collection(prepared, collection):
    validate_prepared(prepared)
    attribution.validate_collected_records(collection)
    binding = collection['stage_binding']
    pipeline.require(binding.get('schema') == 'attention_relational_report_v1_collection_binding'
                     and binding.get('measurement_version') == VERSION and binding['stage'] == 'rate'
                     and binding['requests'] == prepared['requests'] and binding['lineage'] == prepared['lineage']
                     and binding['output_root'] == RATE_ROOT
                     and binding['preparation_content_sha256'] == digest(prepared)
                     and binding['request_ids'] == [r['id'] for r in prepared['requests']]
                     and collection.get('manifest_sha256', binding['manifest_sha256']) == binding['manifest_sha256'],
                     'not an honestly bound actual V3 collection')
    records = pipeline._records(collection)
    pipeline.require(set(records) == set(binding['request_ids']) and len(records) == 1104, 'full raw V3 inventory differs')
    request_index = pipeline._index(prepared['requests'])
    rows = []
    for unit in prepared['units']:
        record = records[unit['unit_id']]
        raw = record.get('parsed')
        grounded = quotes.derive_annotation(unit['text'], unit['unit_id'], raw,
                                            expected_text_sha256=unit['text_sha256'])
        canonical = grounded.pop('canonical_annotation')
        established = bool(record['transport_complete'] and unit['actor_text_available'] and grounded['valid'])
        outcome = pipeline._unwrap_outcome(record['raw_outcome']) if record.get('raw_outcome') is not None else {}
        output_text = outcome.get('output_text')
        rows.append(dict(id=unit['id'], unit_id=unit['unit_id'], case=deepcopy(unit['case']), template=unit['template'],
            text_sha256=unit['text_sha256'], actor_text_available=unit['actor_text_available'],
            request_sha256=pipeline.runner_request_sha256(request_index[unit['unit_id']]),
            raw_response_content_sha256=digest(record['raw_outcome']) if record.get('raw_outcome') is not None else None,
            raw_output_text_sha256=pipeline.digest(output_text.encode('utf-8')) if type(output_text) is str else None,
            storage_provenance=deepcopy(record.get('storage_provenance')), transport_complete=record['transport_complete'],
            derivation=grounded, established=established, canonical_annotation=canonical if established else None))
    return dict(schema=VERSION + '_derived_collection', preparation_content_sha256=digest(prepared),
                raw_collection_content_sha256=digest(collection), lineage=deepcopy(prepared['lineage']), rows=rows)


def verify_factual_summary(summary, prepared):
    deps = summary['preparation_dependencies']
    pipeline.require(set(deps) == {'exports', 'reports', 'audits', 'ratings'}
                     and deps['exports']['sha256'] == origins.EXPORT_SHA256
                     and deps['reports']['sha256'] == REPORTS_SHA256
                     and deps['ratings']['sha256'] == OLD_RATINGS_SHA256, 'factual summary input versions differ')
    loaded = {name: pipeline.bound_json(row['path'], row['sha256']) for name, row in deps.items()}
    audit = loaded['audits']
    selected = prepared['lineage']['selected_factual_auditor']
    pipeline.require(selected['manifest_path'] == AUDITOR_MANIFEST and selected['output_root'] == AUDITOR_ROOT
                     and selected['manifest_sha256'] == AUDITOR_MANIFEST_SHA256
                     and audit['stage_binding']['manifest_sha256'] == selected['manifest_sha256']
                     and audit['stage_binding']['output_root'] == selected['output_root']
                     and audit['request_count'] == selected['request_count'] == 537
                     and len(pipeline._records(audit)) == 537
                     and set(pipeline._records(audit)) == set(audit['stage_binding']['request_ids'])
                     and ((audit.get('interrupted_batch') is True
                           and audit.get('all_available_partial_files_preserved') is True)
                          or type(audit.get('completion_sha256')) is str),
                     'factual join must retain the entire preselected resource-v2 auditor inventory')
    for name in ('reports', 'audits', 'ratings'):
        attribution.validate_collected_records(loaded[name])
    recomputed = pipeline.summarize_reporting_results(loaded['reports'], None, audit, loaded['ratings'],
        pipeline.build_pair_index(loaded['exports']['provenance']), loaded['exports'])
    attribution.verify_primary(summary, recomputed)
    return recomputed


def join_reference(derived, factual):
    """Reuse whole-version factual decisions; no itemwise fallback to other auditors."""
    truth = pipeline._index(factual['all_text_results'])
    pipeline.require(len(derived['rows']) == len(truth) == 1104
                     and {r['id'] for r in derived['rows']} == set(truth), 'factual/reference inventories differ')
    rows, groups = [], defaultdict(Counter)
    for row in derived['rows']:
        fact = truth[row['id']]
        pipeline.require(all(row[k] == fact[k] for k in ('unit_id', 'text_sha256', 'case', 'template', 'actor_text_available')),
                         'factual/reference exact ID/text/case binding differs')
        faithful = fact['fidelity_established']
        annotation = row['canonical_annotation'] if row['established'] else None
        arm = 'template' if row['template'] else 'narrator'
        key = '|'.join(str(row['case'][k]) for k in ('encoder_seed', 'source', 'view', 'variant')) + '|' + arm
        count = groups[key]
        count.update(text_opportunities=1, valid_reference=int(row['established']), fidelity_established=int(faithful),
                     rating_transport_complete=int(row['transport_complete']),
                     reference_unestablished=int(not row['established']))
        endpoints = {}
        for endpoint in ('COD', 'CA'):
            obj = annotation['composites'][endpoint] if annotation else None
            status = obj['status'] if obj else 'unestablished'
            lower, upper = status == 'explicit', status in ('explicit', 'ambiguous')
            scopes = sorted({b['scope'] for b in obj['bindings']}) if obj else []
            times = sorted({b['time'] for b in obj['bindings']}) if obj else []
            endpoints[endpoint] = dict(status=status, lower=lower, upper=upper, faithful_lower=lower and faithful,
                                      faithful_upper=upper and faithful, scopes=scopes, times=times)
            for metric in ('lower', 'upper', 'faithful_lower', 'faithful_upper'):
                count[endpoint + '_' + metric] += int(endpoints[endpoint][metric])
            count[endpoint + '_status_' + status] += 1
            for scope in pipeline.SCOPES:
                for metric in ('lower', 'upper', 'faithful_lower', 'faithful_upper'):
                    count[endpoint + '_' + scope + '_' + metric] += int(endpoints[endpoint][metric] and scope in scopes)
        rows.append(dict(id=row['id'], unit_id=row['unit_id'], case=deepcopy(row['case']), template=row['template'],
            text_sha256=row['text_sha256'], reference_established=row['established'], fidelity_established=faithful,
            actor_text_available=row['actor_text_available'], rating_transport_complete=row['transport_complete'],
            features={k: annotation['features'][k]['status'] if annotation else 'unestablished' for k in ('C', 'O', 'D', 'A')},
            auxiliary={k: annotation['auxiliary'][k]['status'] if annotation else 'unestablished' for k in pipeline.AUXILIARY},
            endpoints=endpoints))
    totals, arms = Counter(), {'narrator': Counter(), 'template': Counter()}
    for key, value in groups.items():
        totals.update(value)
        arms[key.rsplit('|', 1)[1]].update(value)
    pipeline.require(totals['text_opportunities'] == 1104 and all(a['text_opportunities'] == 552 for a in arms.values()),
                     'fixed supplementary denominators differ')
    def rates(count):
        denominator = count['text_opportunities']
        return dict(text_denominator=denominator, **{endpoint + '_' + metric + '_rate':
            count[endpoint + '_' + metric] / denominator for endpoint in ('COD', 'CA')
            for metric in ('lower', 'upper', 'faithful_lower', 'faithful_upper')})
    return dict(schema=VERSION + '_supplementary_assessment', scope_and_chronology=__doc__, supplementary_same_12_cases=True,
        original_primary_results_replaced=False, no_itemwise_auditor_selection=True, totals=dict(totals),
        arm_totals={k: dict(v) for k, v in arms.items()}, groups={k: dict(v) for k, v in groups.items()},
        arm_rates={k: rates(v) for k, v in arms.items()}, group_rates={k: rates(v) for k, v in groups.items()},
        all_text_results=rows, selected_factual_auditor=deepcopy(derived['lineage']['selected_factual_auditor']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'collect', 'derive', 'join'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in ('exports', 'reports', 'preparation', 'completion', 'collection', 'derived', 'factual_summary'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path)
        parser.add_argument('--' + name.replace('_', '-') + '-sha256')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--commit')
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing first artifact replacement')
    pipeline.require(args.output.resolve().is_relative_to((pipeline.ROOT / BASE).resolve()),
                     'V3 artifacts must remain in their separate V3 namespace')
    dependencies = {}

    def load(name):
        path, sha = getattr(args, name), getattr(args, name + '_sha256')
        pipeline.require(path is not None and sha is not None, 'bound input required: ' + name)
        dependencies[name] = dict(path=str(path), sha256=sha)
        return pipeline.bound_json(path, sha)

    if args.stage == 'prepare':
        for path in (args.exports, args.reports, args.output):
            runner.relative_path(str(path))
        exports, reports = load('exports'), load('reports')
        prepared = prepare(exports, reports, deepcopy(dependencies))
        args.output.mkdir(parents=True, exist_ok=False)
        artifacts = {'requests.json': prepared['requests'], 'lineage.json': prepared['lineage'],
                     'response_schema.json': quotes.response_schema(), 'preparation.json': prepared}
        for name, value in artifacts.items():
            pipeline.exclusive_json(args.output / name, value)
        sources = dict(prepared['lineage']['source_sha256'])
        sources.update({row['path']: row['sha256'] for row in dependencies.values()})
        sources[AUDITOR_MANIFEST] = prepared['lineage']['selected_factual_auditor']['manifest_sha256']
        sources.update({str(args.output / name): pipeline.digest((args.output / name).read_bytes()) for name in artifacts})
        draft = dict(schema=runner.SCHEMA, stage='rate', settings=prepared['settings'], request_count=1104,
            request_ids=[r['id'] for r in prepared['requests']], requests_path=str(args.output / 'requests.json'),
            lineage_path=str(args.output / 'lineage.json'), output_root=RATE_ROOT,
            review_path=str(args.output / 'runtime_review.json'), source_sha256=sources)
        pipeline.exclusive_json(args.output / 'runtime_manifest_draft.json', draft)
        return  # The draft deliberately lacks approval/review hashes and cannot execute.
    prepared = load_preparation(args.preparation, args.preparation_sha256)
    dependencies['preparation'] = dict(path=str(args.preparation), sha256=args.preparation_sha256)
    if args.stage == 'collect':
        manifest, requests, manifest_sha = runner.validate(str(args.manifest), args.commit)
        pipeline.require(manifest['source_sha256'].get(str(args.preparation)) == args.preparation_sha256,
                         'runtime did not bind this actual preparation artifact')
        lineage = pipeline.bound_json(manifest['lineage_path'], manifest['source_sha256'][manifest['lineage_path']])
        binding = build_binding(manifest, manifest_sha, requests, manifest['source_sha256'][manifest['requests_path']],
                                lineage, manifest['source_sha256'][manifest['lineage_path']], prepared)
        pipeline.require(args.completion is not None and args.completion_sha256 is not None, 'bound completion required')
        result = pipeline.collect_first_responses(args.completion, args.completion_sha256, binding['request_ids'],
                                                  RATE_ROOT, stage_binding=binding)
        dependencies.update(manifest=dict(path=str(args.manifest), sha256=manifest_sha, commit=args.commit),
                            completion=dict(path=str(args.completion), sha256=args.completion_sha256))
    elif args.stage == 'derive':
        result = derive_collection(prepared, load('collection'))
    else:
        derived = load('derived')
        raw_dep = derived['input_dependencies']['collection']
        raw = pipeline.bound_json(raw_dep['path'], raw_dep['sha256'])
        recomputed = derive_collection(prepared, raw)
        pipeline.require(all(derived.get(k) == v for k, v in recomputed.items()), 'derived data differ from raw V3 grounding')
        factual = verify_factual_summary(load('factual_summary'), prepared)
        result = join_reference(recomputed, factual)
    result['input_dependencies'] = dependencies
    result['source_sha256'] = source_hashes()
    result['no_api_or_model_calls'] = True
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
