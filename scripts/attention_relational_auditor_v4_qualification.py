"""Inert fictional V4 qualification preparation, retention and derived mechanics.

No API client, hidden-key reader, semantic grader, qualification verdict or target
release is implemented. Two stateless contexts share one model and are not an
empirical independence claim. Raw first responses and derived evidence stay apart.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import attention_relational_grounded_auditor_v4 as core
import attention_relational_origin_attribution_v1 as attribution
import attention_relational_report_v1 as exporter
import attention_relational_reporting_pipeline_v1 as pipeline
import inventory_attention_relational_origins_v1 as origins
import run_attention_relational_reports_v1 as runner


VERSION = 'attention_relational_auditor_v4_qualification'
PUBLIC_SCHEMA = 'attention_relational_auditor_v4_fictional_cases_v1'
BASE = 'audits/attention_relational_report_v1/grounded_auditor_v4/qualification'
RUN_ROOT = BASE + '/run'
AUDITORS = ('auditor_0', 'auditor_1')
ID_SALT = VERSION + '_id_20261019'
ORDER_SALT = VERSION + '_order_20261019'


def digest(value):
    return pipeline.digest(pipeline.encoded(value))


def settings():
    # Generic runner hardcodes low verbosity, store=False and zero SDK retries.
    return dict(model='gpt-5-mini-2025-08-07', reasoning_effort='medium', max_output_tokens=32768,
                timeout_seconds=600, concurrency=6, output_schema=core.response_schema())


def validate_cases(public):
    pipeline.require(type(public) is dict and set(public) == {'schema', 'cases'}
                     and public['schema'] == PUBLIC_SCHEMA, 'exact public-case wrapper required')
    cases = public['cases']
    pipeline.require(type(cases) is list and len(cases) == 24, 'exactly24 public cases required')
    identifiers = []
    for case in cases:
        pipeline.require(type(case) is dict and set(case) == {'case_id', 'packet', 'report_text'},
                         'public case has missing fields or spoiler/key/category fields')
        pipeline.require(type(case['case_id']) is str and bool(case['case_id'].strip())
                         and type(case['report_text']) is str and type(case['packet']) is dict,
                         'public case field types differ')
        identifiers.append(case['case_id'])
    pipeline.require(len(set(identifiers)) == 24, 'duplicate public case ID')
    return cases


def prepare(public, inputs, sources):
    """Pure assembly apart from the core's fixed instruction-file read; no key input."""
    cases = validate_cases(public)
    requests, mapping = [], []
    for auditor in AUDITORS:
        for number, case in enumerate(cases):
            identifier = 'q' + digest([ID_SALT, auditor, case['case_id']])[:24]
            request = core.build_request(identifier, case['report_text'], case['packet'])
            payload = pipeline.decoded(request['input'].rsplit('\n', 1)[1])
            requests.append(request)
            mapping.append(dict(request_id=identifier, unit_id=payload['unit_id'], auditor=auditor,
                case_id=case['case_id'], case_order=number, report_text_sha256=payload['report_text_sha256'],
                packet_content_sha256=payload['packet_content_sha256'],
                request_sha256=pipeline.runner_request_sha256(request)))
    requests.sort(key=lambda r: pipeline.digest((ORDER_SALT + ':' + r['id']).encode('ascii')))
    pipeline.require(len({r['id'] for r in requests}) == len({m['unit_id'] for m in mapping}) == 48,
                     'qualification request/unit ID collision')
    lineage = dict(schema=VERSION + '_lineage', inputs=deepcopy(inputs), source_sha256=deepcopy(sources),
        public_cases_content_sha256=digest(public), requests_content_sha256=digest(requests),
        mapping_content_sha256=digest(mapping), response_schema_content_sha256=digest(core.response_schema()),
        instructions_sha256=pipeline.digest(core.INSTRUCTIONS.read_bytes()), span_rule=core.SPAN_RULE,
        id_salt=ID_SALT, order_salt=ORDER_SALT)
    return dict(schema=VERSION + '_preparation', public_cases=deepcopy(public), requests=requests,
        private_grading_mapping=mapping, lineage=lineage, settings=settings(), output_root=RUN_ROOT,
        model_transport=dict(verbosity='low', store=False, sdk_max_retries=0, stateless=True),
        semantic_qualification_assessed=False, hidden_key_read=False)


def source_hashes():
    paths = [Path(m.__file__) for m in (core, attribution, exporter, pipeline, origins, runner)] + [Path(__file__)]
    paths += [core.INSTRUCTIONS, pipeline.ROOT / 'docs/ATTENTION_RELATIONAL_GROUNDED_AUDITOR_V4_DESIGN.md',
              pipeline.ROOT / 'tests/test_attention_relational_auditor_v4_qualification.py',
              pipeline.ROOT / 'tests/test_attention_relational_grounded_auditor_v4.py']
    return {str(p.resolve().relative_to(pipeline.ROOT)): pipeline.digest(p.read_bytes()) for p in paths}


def validate_preparation(prepared):
    pipeline.require(prepared == prepare(prepared['public_cases'], prepared['lineage']['inputs'],
                                          prepared['lineage']['source_sha256']), 'qualification preparation differs')


def load_preparation(path, sha):
    pipeline.require(path is not None and sha is not None, 'bound preparation required')
    prepared = pipeline.bound_json(path, sha)
    inputs = prepared['lineage']['inputs']
    pipeline.require(set(inputs) == {'cases'}, 'only public cases are qualification inputs')
    cases = pipeline.bound_json(inputs['cases']['path'], inputs['cases']['sha256'])
    pipeline.require(prepared == prepare(cases, inputs, source_hashes()), 'prepared source/input lineage differs')
    return prepared


def build_binding(manifest, manifest_sha, requests, requests_sha, lineage, lineage_sha, prepared):
    validate_preparation(prepared)
    pipeline.require(manifest['stage'] == 'audit' and manifest['settings'] == settings()
        and manifest['output_root'] == RUN_ROOT and manifest['request_count'] == 48
        and manifest['request_ids'] == [r['id'] for r in prepared['requests']]
        and requests == prepared['requests'] and lineage == prepared['lineage']
        and manifest['source_sha256'].get(manifest['requests_path']) == requests_sha
        and manifest['source_sha256'].get(manifest['lineage_path']) == lineage_sha, 'actual qualification runtime differs')
    return dict(schema='attention_relational_report_v1_collection_binding', stage='audit',
        measurement_version=VERSION, manifest_sha256=manifest_sha, requests_sha256=requests_sha,
        lineage_sha256=lineage_sha, request_ids=manifest['request_ids'], requests=deepcopy(requests),
        lineage=deepcopy(lineage), output_root=RUN_ROOT, preparation_content_sha256=digest(prepared))


def derive(prepared, collection):
    validate_preparation(prepared)
    attribution.validate_collected_records(collection)
    binding = collection['stage_binding']
    pipeline.require(binding.get('schema') == 'attention_relational_report_v1_collection_binding'
        and binding.get('measurement_version') == VERSION and binding['stage'] == 'audit'
        and binding['output_root'] == RUN_ROOT and binding['requests'] == prepared['requests']
        and binding['lineage'] == prepared['lineage'] and binding['preparation_content_sha256'] == digest(prepared)
        and binding['request_ids'] == [r['id'] for r in prepared['requests']]
        and collection.get('manifest_sha256', binding['manifest_sha256']) == binding['manifest_sha256'],
        'not the actual new qualification binding')
    records = pipeline._records(collection)
    pipeline.require(set(records) == set(binding['request_ids']) and len(records) == 48, 'full48 records required')
    requests = pipeline._index(prepared['requests'])
    rows, counts = [], defaultdict(Counter)
    for item in prepared['private_grading_mapping']:
        identifier = item['request_id']
        record = records[identifier]
        evidence = core.derive_annotation(requests[identifier], record.get('parsed'),
                                          expected_request_sha256=item['request_sha256'])
        complete = bool(record['transport_complete'])
        established = complete and evidence['mechanical_valid']
        if not established:
            evidence['derived_assertions'] = None
        raw = record.get('raw_outcome')
        outcome = pipeline._unwrap_outcome(raw) if raw is not None else {}
        output_text = outcome.get('output_text')
        rows.append(dict(**deepcopy(item), transport_complete=complete, availability=record.get('availability'),
            mechanical_contract_established=established, derived_evidence=evidence,
            raw_response_content_sha256=digest(raw) if raw is not None else None,
            raw_output_text_sha256=pipeline.digest(output_text.encode('utf-8')) if type(output_text) is str else None,
            storage_provenance=deepcopy(record.get('storage_provenance'))))
        counts[item['auditor']].update(opportunities=1, transport_complete=int(complete),
            mechanical_contract_established=int(established), contract_unestablished=int(not established))
    pipeline.require(set(counts) == set(AUDITORS) and all(c['opportunities'] == 24 for c in counts.values()),
                     'two-by24 opportunity accounting differs')
    return dict(schema=VERSION + '_derived_evidence', preparation_content_sha256=digest(prepared),
        raw_collection_content_sha256=digest(collection), lineage=deepcopy(prepared['lineage']),
        opportunities=48, by_auditor={a: dict(c) for a, c in counts.items()}, rows=rows,
        semantic_qualification_assessed=False, semantic_qualification_verdict_emitted=False,
        independent_keyed_reviews_required=True, hidden_key_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'collect', 'derive'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in ('cases', 'preparation', 'completion', 'collection'):
        parser.add_argument('--' + name, type=Path)
        parser.add_argument('--' + name + '-sha256')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--commit')
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing replacement of first artifact')
    destination = args.output.resolve()
    pipeline.require(destination.is_relative_to((pipeline.ROOT / BASE).resolve())
                     and not destination.is_relative_to((pipeline.ROOT / RUN_ROOT).resolve()),
                     'qualification preparation/collection/derivation must stay outside the separate raw run root')
    dependencies = {}

    def load(name):
        path, sha = getattr(args, name), getattr(args, name + '_sha256')
        pipeline.require(path is not None and sha is not None, 'bound input required: ' + name)
        dependencies[name] = dict(path=str(path), sha256=sha)
        return pipeline.bound_json(path, sha)

    if args.stage == 'prepare':
        runner.relative_path(str(args.output))
        cases = load('cases')
        runner.relative_path(str(args.cases))
        prepared = prepare(cases, deepcopy(dependencies), source_hashes())
        args.output.mkdir(parents=True, exist_ok=False)
        artifacts = {'requests.json': prepared['requests'], 'lineage.json': prepared['lineage'],
                     'response_schema.json': core.response_schema(), 'preparation.json': prepared,
                     'private_grading_mapping.json': prepared['private_grading_mapping']}
        for name, value in artifacts.items():
            pipeline.exclusive_json(args.output / name, value)
        sources = dict(prepared['lineage']['source_sha256'])
        sources[str(args.cases)] = args.cases_sha256
        sources.update({str(args.output / name): pipeline.digest((args.output / name).read_bytes()) for name in artifacts})
        draft = dict(schema=runner.SCHEMA, stage='audit', settings=settings(), request_count=48,
            request_ids=[r['id'] for r in prepared['requests']], output_root=RUN_ROOT,
            requests_path=str(args.output / 'requests.json'), lineage_path=str(args.output / 'lineage.json'),
            review_path=str(args.output / 'runtime_review.json'), source_sha256=sources)
        pipeline.exclusive_json(args.output / 'runtime_manifest_draft.json', draft)
        return  # Review, hidden-key freeze and release are external prerequisites, never invented here.
    prepared = load_preparation(args.preparation, args.preparation_sha256)
    dependencies['preparation'] = dict(path=str(args.preparation), sha256=args.preparation_sha256)
    if args.stage == 'collect':
        manifest, requests, manifest_sha = runner.validate(str(args.manifest), args.commit)
        pipeline.require(manifest['source_sha256'].get(str(args.preparation)) == args.preparation_sha256,
                         'actual runtime did not bind this preparation')
        sources = manifest['source_sha256']
        lineage = pipeline.bound_json(manifest['lineage_path'], sources[manifest['lineage_path']])
        binding = build_binding(manifest, manifest_sha, requests, sources[manifest['requests_path']],
                                lineage, sources[manifest['lineage_path']], prepared)
        pipeline.require(args.completion is not None and args.completion_sha256 is not None, 'bound first completion required')
        result = pipeline.collect_first_responses(args.completion, args.completion_sha256, binding['request_ids'],
                                                  RUN_ROOT, stage_binding=binding)
        dependencies.update(manifest=dict(path=str(args.manifest), sha256=manifest_sha, commit=args.commit),
                            completion=dict(path=str(args.completion), sha256=args.completion_sha256))
    else:
        result = derive(prepared, load('collection'))
    result['input_dependencies'] = dependencies
    result['source_sha256'] = source_hashes()
    result['no_api_or_model_calls'] = True
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
