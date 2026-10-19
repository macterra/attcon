"""Separate V5 qualification preparation/retention/mechanics, never semantic release.

No API, worker launcher, key reader, semantic grader, certificate or target runner.
Two phases by two fresh subagent contexts use V4 data-format compatibility only.
All first artifacts/unknowns remain; host/runtime identity is root-attested.
"""
import argparse
import base64
from copy import deepcopy
from pathlib import Path

import attention_relational_grounded_auditor_v4 as core
import attention_relational_reporting_pipeline_v1 as pipeline
import attention_relational_subagent_submit_v5 as submitter


VERSION = 'attention_relational_subagent_auditor_v5'
BASE = 'audits/attention_relational_report_v1/subagent_auditor_v5'
PLAN = 'docs/ATTENTION_RELATIONAL_SUBAGENT_AUDITOR_V5_PLAN.md'
WORKER = BASE + '/worker_instructions.txt'
WORKER_SHA = 'be0ebd1ce33f8698c7a8e39b6c2d4af06c92fb64a1bc614e8ecffcd6df91bd35'
SHORT_SHA = '473fab782d8a4b9f9ef83ed93c5b1ef9158df5b2e1ed2689a4c18a699157091f'
LONG_SHA = '00cbe49cea18170d41924c9eed0c93a13d832c867545794cee10c1dd69eae171'
LONG_KEY_SHA = '3aab8f32ed5101b33dcfc299fd9c4768079af8646e274f13bfec01656db204f5'
SHORT_SCHEMA = 'attention_relational_auditor_v4_fictional_cases_v1'
LONG_SCHEMA = 'attention_relational_auditor_v5_long_fictional_cases_v1'
APPROVAL_SCHEMA = VERSION + '_long_key_approval_v1'
PHASES, STREAMS = ('short', 'long'), ('stream_0', 'stream_1')
HELPER = 'scripts/attention_relational_subagent_submit_v5.py'
ID_SALT = VERSION + '_id_20261019:'
ORDER_SALT = 'attention_relational_subagent_auditor_v5_qualification_order_20261019:'


def digest(value):
    return pipeline.digest(pipeline.encoded(value))


def descriptor(path, sha):
    return dict(path=str(path), sha256=sha)


def valid_sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_public(public, phase):
    pipeline.require(type(public) is dict and set(public) == {'schema', 'cases'}
        and public['schema'] == (SHORT_SCHEMA if phase == 'short' else LONG_SCHEMA), 'exact public-case wrapper required')
    cases = public['cases']
    pipeline.require(type(cases) is list and len(cases) == (24 if phase == 'short' else 4), 'complete24/4 public cases required')
    pipeline.require(len(pipeline._index(cases, 'case_id')) == len(cases), 'duplicate public case')
    for case in cases:
        pipeline.require(set(case) == {'case_id', 'packet', 'report_text'} and type(case['case_id']) is str
            and bool(case['case_id']) and type(case['packet']) is dict and type(case['report_text']) is str,
            'public case fields differ or expose a key/category')
        if phase == 'long':
            pipeline.require(180 <= len(case['report_text'].split()) <= 220
                and len(case['packet']['panels']) == 2
                and sum(c['kind'] == 'role_distribution' for c in core.source_contexts(case['packet'])) == 64,
                'long cases require180-220 words and complete64-peak-set two-panel packets')
    return cases


def validate_approval(approval, long_sha):
    pipeline.require(type(approval) is dict and approval.get('schema') == APPROVAL_SCHEMA
        and approval.get('approved') is True and approval.get('independent_review') is True
        and approval.get('public_cases_sha256') == long_sha
        and approval.get('hidden_key_sha256') == LONG_KEY_SHA
        and all(valid_sha(approval.get(k)) for k in ('hidden_key_sha256', 'key_review_sha256'))
        and type(approval.get('reviewer_id')) is str and bool(approval['reviewer_id'].strip())
        and type(approval.get('mandatory_atom_count')) is int and approval['mandatory_atom_count'] > 0,
        'actual independently approved long-public/key metadata required; no synthetic default')


def prepare(short, long, approval, inputs, sources, worker_instructions):
    """Pure assembly apart from core's fixed instruction read; never read a key."""
    cases = {p: validate_public(v, p) for p, v in (('short', short), ('long', long))}
    validate_approval(approval, inputs['long_cases']['sha256'])
    pipeline.require(pipeline.digest(worker_instructions.encode('utf-8')) == WORKER_SHA, 'frozen worker instruction bytes differ')
    batches, mapping, schedule = {}, [], []
    bindings = dict(submission_helper_sha256=sources[HELPER], worker_instructions_sha256=WORKER_SHA,
        plan_sha256=sources[PLAN],
        core_source_sha256=sources['scripts/attention_relational_grounded_auditor_v4.py'],
        core_instructions_sha256=pipeline.digest(core.INSTRUCTIONS.read_bytes()), response_schema_content_sha256=digest(core.response_schema()))
    for phase in PHASES:
        ordered = sorted(cases[phase], key=lambda c: (pipeline.digest((ORDER_SALT + phase + ':' + c['case_id']).encode()), c['case_id']))
        for stream in STREAMS:
            batch_id = 'b' + digest([ID_SALT, phase, stream])[:24]
            output_root = BASE + '/qualification/outputs/' + batch_id
            requests = []
            for case in ordered:
                identifier = 'v' + digest([ID_SALT, phase, stream, case['case_id']])[:24]
                request = core.build_request(identifier, case['report_text'], case['packet'])
                value = pipeline.decoded(request['input'].rsplit('\n', 1)[1])
                requests.append(request)
                mapping.append(dict(request_id=identifier, unit_id=value['unit_id'], batch_id=batch_id, phase=phase, stream=stream,
                    case_id=case['case_id'], request_sha256=pipeline.runner_request_sha256(request),
                    report_text_sha256=value['report_text_sha256'], packet_content_sha256=value['packet_content_sha256']))
            batch = dict(schema=submitter.BATCH_SCHEMA, batch_id=batch_id, output_root=output_root,
                worker_instructions=worker_instructions, bindings=deepcopy(bindings), annotation_schema=core.response_schema(), requests=requests)
            batches[batch_id] = batch
            schedule.append(dict(batch_id=batch_id, phase=phase, stream=stream, public_file='public/' + batch_id + '.json',
                public_sha256=pipeline.digest(pipeline.encoded(batch) + b'\n'), output_root=output_root, request_count=len(requests)))
    pipeline.require(len(mapping) == len({m['request_id'] for m in mapping}) == len({m['unit_id'] for m in mapping}) == 56,
                     'all56 unique qualification opportunities required')
    return dict(schema=VERSION + '_preparation', public_batches=batches, private_mapping=mapping, schedule=schedule,
        public_cases=dict(short=deepcopy(short), long=deepcopy(long)), long_key_approval=deepcopy(approval),
        lineage=dict(schema=VERSION + '_lineage', inputs=deepcopy(inputs), source_sha256=deepcopy(sources),
            worker_instructions_sha256=WORKER_SHA, long_key_metadata=deepcopy(approval), public_batches_content_sha256=digest(batches),
            mapping_content_sha256=digest(mapping), plan_sha256=sources[PLAN]), hidden_key_content_read=False,
        method='fresh tool-assisted subagent contexts; inherited runtime is recorded by host receipts, not an API model claim',
        semantic_qualification_assessed=False, target_release_implemented=False)


def source_hashes():
    paths = [Path(m.__file__) for m in (core, pipeline, submitter)] + [Path(__file__), core.INSTRUCTIONS]
    paths += [pipeline.ROOT / PLAN, pipeline.ROOT / WORKER,
        pipeline.ROOT / 'tests/test_attention_relational_subagent_auditor_v5.py', pipeline.ROOT / 'tests/test_attention_relational_subagent_submit_v5.py',
        pipeline.ROOT / 'tests/test_attention_relational_template_index_v1.py',
        pipeline.ROOT / 'scripts/attention_relational_report_v1.py']
    return {str(p.resolve().relative_to(pipeline.ROOT)): pipeline.digest(p.read_bytes()) for p in paths}


def validate_prepared(prepared):
    pipeline.require(prepared == prepare(prepared['public_cases']['short'], prepared['public_cases']['long'],
        prepared['long_key_approval'], prepared['lineage']['inputs'], prepared['lineage']['source_sha256'],
        next(iter(prepared['public_batches'].values()))['worker_instructions']), 'V5 preparation differs')


def load_preparation(path, sha):
    prepared = pipeline.bound_json(path, sha)
    inputs = prepared['lineage']['inputs']
    pipeline.require(set(inputs) == {'short_cases', 'long_cases', 'long_key_review'}
        and inputs['short_cases']['sha256'] == SHORT_SHA and inputs['long_cases']['sha256'] == LONG_SHA,
                     'actual pinned short cases and bound long/key approval required')
    public = {k: pipeline.bound_json(v['path'], v['sha256']) for k, v in inputs.items()}
    pipeline.require(prepared == prepare(public['short_cases'], public['long_cases'], public['long_key_review'], inputs,
        source_hashes(), (pipeline.ROOT / WORKER).read_text()), 'actual prepared V5 source/input lineage differs')
    return prepared


def validate_attempts(prepared, preparation_sha, launch, launch_sha, outcome):
    """Root attestations bind actual host contexts; code cannot observe handles."""
    schedule = {b['batch_id']: b for b in prepared['schedule']}
    pipeline.require(launch.get('schema') == VERSION + '_launch_receipt' and launch.get('preparation_sha256') == preparation_sha
        and launch.get('all_inputs_frozen_before_first_tasks') is True, 'actual launch/input-freeze attestation required')
    launched = pipeline._index(launch['contexts'], 'batch_id')
    finished = pipeline._index(outcome['contexts'], 'batch_id')
    pipeline.require(outcome.get('schema') == VERSION + '_outcome_receipt' and outcome.get('preparation_sha256') == preparation_sha
        and outcome.get('launch_receipt_sha256') == launch_sha
        and outcome.get('no_feedback_before_all_first_contexts_saved_or_terminal') is True
        and set(launched) == set(finished) == set(schedule) and len({c['agent_id'] for c in launched.values()}) == 4,
        'all four actual first contexts must be bound before any derivation')
    for identifier, expected in schedule.items():
        context, terminal = launched[identifier], finished[identifier]
        pipeline.require(context.get('fork_turns') == 'none' and context.get('inherited_parent_model') is True
            and context.get('model_override') is None and type(context.get('agent_id')) is str and bool(context['agent_id'])
            and type(context.get('runtime_identity')) is dict and bool(context['runtime_identity'])
            and context.get('batch_sha256') == expected['public_sha256'] and context.get('output_root') == expected['output_root']
            and terminal.get('agent_id') == context['agent_id'], 'host context/source/runtime binding differs')
        first = terminal.get('first_batch')
        pipeline.require((type(first) is dict or first is None) and (type(first) is dict or terminal.get('actual_agent_terminal') is True
            and type(terminal.get('terminal_attestation')) is dict
            and terminal['terminal_attestation'].get('agent_id') == context['agent_id']
            and terminal['terminal_attestation'].get('terminal_observed') is True),
            'timeout or running context cannot stand for terminal first attempt')
        if first is not None:
            pipeline.require(Path(first['path']).resolve() == (Path(expected['output_root']) / 'first_batch.json').resolve()
                and valid_sha(first.get('sha256')), 'actual exclusive first-batch descriptor required')
    return launched, finished


def retained_file(path):
    raw = Path(path).read_bytes()
    try:
        parsed, error = pipeline.decoded(raw), None
    except (ValueError, UnicodeError) as exc:
        parsed, error = None, type(exc).__name__ + ': ' + str(exc)
    return dict(path=str(path), sha256=pipeline.digest(raw), raw_bytes_base64=base64.b64encode(raw).decode('ascii'), parsed=parsed, parse_error=error)


def collect(prepared, preparation_sha, launch, launch_sha, outcome):
    validate_prepared(prepared)
    launched, finished = validate_attempts(prepared, preparation_sha, launch, launch_sha, outcome)
    records, contexts = [], []
    for schedule in prepared['schedule']:
        bid = schedule['batch_id']
        batch, root = prepared['public_batches'][bid], Path(schedule['output_root'])
        requests = pipeline._index(batch['requests'])
        annotation_files = [retained_file(p) for p in sorted(root.glob('*.first.json'))]
        receipts = [retained_file(p) for p in sorted(root.glob('*.receipt.json'))]
        events = [dict(path=str(p), sha256=pipeline.digest(p.read_bytes()), raw_bytes_base64=base64.b64encode(p.read_bytes()).decode('ascii'))
                  for p in sorted(root.glob('*events.jsonl'))]
        context_issues, first, first_artifact, calculator = [], None, None, []
        first_ref = finished[bid].get('first_batch')
        if first_ref is None:
            context_issues.append('first_batch_manifest_unestablished_despite_terminal_retention')
        if first_ref is not None:
            if Path(first_ref['path']).is_file():
                first_artifact = retained_file(first_ref['path'])
                first = first_artifact['parsed'] if type(first_artifact['parsed']) is dict else None
                if first_artifact['sha256'] != first_ref['sha256']:
                    context_issues.append('first_batch_file_sha_differs')
            if first is None:
                context_issues.append('first_batch_missing_or_malformed')
            elif not (first.get('schema') == submitter.FIRST_SCHEMA and first.get('batch_sha256') == schedule['public_sha256']
                and first.get('batch_id') == bid and first.get('output_root') == schedule['output_root']
                and first.get('requested_count') == len(requests) and first.get('submission_helper_sha256') == batch['bindings']['submission_helper_sha256']):
                context_issues.append('first_batch_binding_differs')
            if first is not None and (type(first.get('calculator_artifacts')) is not list or type(first.get('positions')) is not list
                or any(type(ref) is not dict or type(ref.get('path')) is not str or not valid_sha(ref.get('sha256'))
                       for ref in first.get('calculator_artifacts', []))):
                context_issues.append('malformed_first_batch_structure')
                first = None
            for ref in (first.get('calculator_artifacts', []) if first else []):
                pipeline.require(Path(ref['path']).resolve().is_relative_to(root.resolve()), 'calculator artifact outside assigned context')
                if not Path(ref['path']).is_file():
                    calculator.append(dict(**ref, missing=True))
                    context_issues.append('calculator_artifact_missing')
                    continue
                retained = retained_file(ref['path'])
                # Calculator artifacts need not be JSON; raw bytes remain authoritative.
                if retained['sha256'] != ref['sha256']:
                    context_issues.append('calculator_artifact_sha_differs')
                calculator.append(retained)
            expected_events = first.get('submission_events') if first else None
            actual_events = next((e for e in events if Path(e['path']).name == 'submission_events.jsonl'), None)
            if expected_events != (descriptor(actual_events['path'], actual_events['sha256']) if actual_events else None):
                context_issues.append('first_event_log_binding_differs')
        try:
            declared = pipeline._index(first.get('positions', []), 'request_id') if first else {}
        except (ValueError, KeyError, TypeError):
            declared = {}
            context_issues.append('malformed_first_position_inventory')
        if first and set(declared) != set(requests):
            context_issues.append('first_batch_inventory_differs')
        if any(Path(f['path']).name.removesuffix('.first.json') not in requests for f in annotation_files):
            context_issues.append('unexpected_annotation_candidate_no_preferred_selection')
        for identifier, request in requests.items():
            files = [f for f in annotation_files if Path(f['path']).name == identifier + '.first.json']
            sources = [f for f in receipts if Path(f['path']).name == identifier + '.receipt.json']
            issues = list(context_issues)
            if len(files) != 1:
                issues.append('missing_or_ambiguous_first_annotation')
            if len(sources) != 1:
                issues.append('missing_or_ambiguous_submission_receipt')
            raw, receipt = (files[0] if len(files) == 1 else None), (sources[0] if len(sources) == 1 else None)
            actual_receipt = receipt['parsed'] if receipt else None
            if raw and (type(actual_receipt) is not dict or actual_receipt.get('schema') != 'attention_relational_subagent_auditor_v5_submission_receipt'
                or actual_receipt.get('request_id') != identifier or actual_receipt.get('batch_sha256') != schedule['public_sha256']
                or actual_receipt.get('request_sha256') != pipeline.runner_request_sha256(request)
                or actual_receipt.get('annotation_sha256') != raw['sha256']
                or actual_receipt.get('submission_helper_sha256') != batch['bindings']['submission_helper_sha256']):
                issues.append('first_submission_input_or_file_binding_differs')
            if first and declared.get(identifier) != dict(request_id=identifier, expected_request_sha256=pipeline.runner_request_sha256(request),
                annotation=descriptor(raw['path'], raw['sha256']) if raw else None,
                submission_receipt=descriptor(receipt['path'], receipt['sha256']) if receipt else None):
                issues.append('first_manifest_artifact_differs')
            if raw and raw['parse_error']:
                issues.append('first_annotation_not_JSON')
            records.append(dict(request_id=identifier, batch_id=bid, first_artifact_binding_established=not issues,
                issues=issues, annotation_file=deepcopy(raw), submission_receipt=deepcopy(receipt)))
        contexts.append(dict(schedule=deepcopy(schedule), host_context=deepcopy(launched[bid]), outcome=deepcopy(finished[bid]),
            first_batch=first, first_batch_artifact=first_artifact, annotation_files=annotation_files, receipt_files=receipts, event_logs=events,
            calculator_artifacts=calculator, issues=context_issues))
    pipeline.require(len(records) == 56, 'all56 first positions required')
    return dict(schema=VERSION + '_collection', preparation_sha256=preparation_sha, preparation_content_sha256=digest(prepared),
        launch_receipt_sha256=launch_sha, launch_receipt=deepcopy(launch), outcome_receipt=deepcopy(outcome),
        contexts=contexts, records=records, opportunities=56, semantic_qualification_assessed=False,
        live_agent_handles_observed_by_this_code=False, no_replacement_or_fallback_performed=True)


def derive(prepared, collection):
    validate_prepared(prepared)
    pipeline.require(collection.get('schema') == VERSION + '_collection' and collection.get('preparation_content_sha256') == digest(prepared),
                     'actual V5 collection/preparation differs')
    validate_attempts(prepared, collection['preparation_sha256'], collection['launch_receipt'],
                      collection['launch_receipt_sha256'], collection['outcome_receipt'])
    records = pipeline._index(collection['records'], 'request_id')
    requests = {r['id']: r for b in prepared['public_batches'].values() for r in b['requests']}
    pipeline.require(set(records) == set(requests) and len(records) == 56, 'full56 retained first positions required')
    rows = []
    for item in prepared['private_mapping']:
        record = records[item['request_id']]
        raw = record['annotation_file']
        if raw:
            data = base64.b64decode(raw['raw_bytes_base64'], validate=True)
            pipeline.require(pipeline.digest(data) == raw['sha256'], 'retained annotation raw SHA differs')
            try:
                parsed = pipeline.decoded(data)
            except (ValueError, UnicodeError):
                parsed = None
            pipeline.require(parsed == raw['parsed'], 'raw first annotation was repaired')
        evidence = core.derive_annotation(requests[item['request_id']], raw['parsed'] if raw else None,
                                          expected_request_sha256=item['request_sha256'])
        established = record['first_artifact_binding_established'] and evidence['mechanical_valid']
        if not established:
            evidence['derived_assertions'] = None
        rows.append(dict(**deepcopy(item), first_artifact_binding_established=record['first_artifact_binding_established'],
            artifact_issues=deepcopy(record['issues']), annotation_sha256=raw['sha256'] if raw else None,
            mechanical_contract_established=established, derived_evidence=evidence))
    return dict(schema=VERSION + '_derived_mechanics', preparation_content_sha256=digest(prepared),
        collection_content_sha256=digest(collection), opportunities=56, short_opportunities=48, long_opportunities=8,
        rows=rows, semantic_qualification_assessed=False, target_release_implemented=False,
        limitation='Legal source IDs and complete spans do not establish correct support, reasons or semantic extraction.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'collect', 'derive'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in ('short-cases', 'long-cases', 'long-key-review', 'preparation', 'launch-receipt', 'outcome-receipt', 'collection'):
        parser.add_argument('--' + name, type=Path)
        parser.add_argument('--' + name + '-sha256')
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing first artifact replacement')
    deps = {}

    def load(name):
        path, sha = getattr(args, name), getattr(args, name + '_sha256')
        pipeline.require(path is not None and valid_sha(sha), 'bound actual input required: ' + name)
        deps[name] = descriptor(path, sha)
        return pipeline.bound_json(path, sha)

    if args.stage == 'prepare':
        pipeline.require(args.short_cases_sha256 == SHORT_SHA and args.long_cases_sha256 == LONG_SHA,
                         'only the fixed original short and independently authored long public cases are allowed')
        short, long, approval = load('short_cases'), load('long_cases'), load('long_key_review')
        result = prepare(short, long, approval, deps, source_hashes(), (pipeline.ROOT / WORKER).read_text())
        args.output.mkdir(parents=True, exist_ok=False)
        for scheduled in result['schedule']:
            pipeline.exclusive_json(args.output / scheduled['public_file'], result['public_batches'][scheduled['batch_id']])
        pipeline.exclusive_json(args.output / 'preparation.json', result)
        return  # No worker launch, semantic gate or target authorization.
    prepared = load_preparation(args.preparation, args.preparation_sha256)
    deps['preparation'] = descriptor(args.preparation, args.preparation_sha256)
    if args.stage == 'collect':
        launch, outcome = load('launch_receipt'), load('outcome_receipt')
        result = collect(prepared, args.preparation_sha256, launch, args.launch_receipt_sha256, outcome)
    else:
        collection = load('collection')
        bound = collection['input_dependencies']
        launch = pipeline.bound_json(bound['launch_receipt']['path'], bound['launch_receipt']['sha256'])
        outcome = pipeline.bound_json(bound['outcome_receipt']['path'], bound['outcome_receipt']['sha256'])
        recomputed = collect(prepared, args.preparation_sha256, launch, bound['launch_receipt']['sha256'], outcome)
        pipeline.require(all(collection.get(k) == v for k, v in recomputed.items()), 'first collection differs from bound raw artifacts/contexts')
        result = derive(prepared, recomputed)
    result.update(input_dependencies=deps, source_sha256=source_hashes(), no_api_or_model_calls=True)
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
