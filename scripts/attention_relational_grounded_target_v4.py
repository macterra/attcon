"""Prospective V4 target preparation, first-response retention and factual assessment.

No API client, semantic qualification grader or reference join is implemented.
The CLI requires an actual hash-bound qualification certificate before preparing
target requests. Fictional pure assembly is not a target-release authorization.
Original/V2 factual measurements and the earlier V3 join remain unchanged.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import attention_relational_auditor_v4_qualification as qualification
import attention_relational_grounded_auditor_v4 as core
import attention_relational_origin_attribution_v1 as attribution
import attention_relational_report_v1 as exporter
import attention_relational_reporting_pipeline_v1 as pipeline
import run_attention_relational_reports_v1 as runner


VERSION = 'attention_relational_grounded_target_v4'
BASE = 'audits/attention_relational_report_v1/grounded_auditor_v4/target'
RUN_ROOT = BASE + '/run'
AUDITORS = qualification.AUDITORS
EXPORT_SHA = '4f5f0bc2fa78b3b24b3580f52e306f20df0f3011d079b8139a2bc1f029050837'
REPORT_SHA = '80e07b465a4ac95832585be9594973ead12ba1b830f3f517cb250d0891ca6940'
QUALIFICATION_SHA = '9a583a56ceac7e7eb2572effb431c47ac3fa04c4073e6ae3191921f73e626b6b'
KEY_SHA = 'c2dda44ac8b49bd0f076d12e22f2b7499ae714bfcc8da382e34c7648e8fab067'
POLICY_SHA = '5be72ec1090d6f367d914e73fa45c66a37e3fd4f8984db05484f03c1a438f8f1'
CERTIFICATE_SCHEMA = VERSION + '_semantic_qualification_certificate'
GRADE_SCHEMA = 'attention_relational_auditor_v4_independent_semantic_grade_v1'
ADJUDICATION_SCHEMA = VERSION + '_qualification_adjudication'
JOIN_NAME = 'supplementary_grounded_v4_factual_quote_v3_reference_v1'
ID_SALT, ORDER_SALT = VERSION + '_id_20261019', VERSION + '_order_20261019'
FACTUAL_RULE = dict(version=VERSION + '_factual_rule_v1', actor_transport_complete=True,
    actor_nonblank_text=True, listed_machine_facts_pass=True, auditors=list(AUDITORS),
    each_auditor=dict(transport_complete=True, mechanical_valid=True,
        complete_assertion_inventory=True, unresolved_ambiguity=False,
        nonempty_assertions=True, all_assertion_verdicts='supported'), combination='conjunction')
digest = qualification.digest
settings = qualification.settings


def descriptor(path, sha):
    return dict(path=str(path), sha256=sha)


def validate_resolution(first_results, final, first_hashes, bindings, resolution):
    """Mechanical release-attestation contract; semantic resolutions stay external."""
    pipeline.require(final == {a: True for a in AUDITORS} and all(type(v) is bool for v in final.values()),
                     'both final auditor qualifications required')
    if resolution is None:
        pipeline.require(len(first_results) == 2 and all(r == final for r in first_results),
                         'negative/uncertain first judgments require explicit bound adjudication')
    else:
        pipeline.require(resolution.get('schema') == ADJUDICATION_SCHEMA
            and all(resolution.get(k) == v for k, v in bindings.items())
            and resolution.get('independent_first_grade_sha256') == first_hashes
            and resolution.get('auditors_qualified') == final
            and resolution.get('no_unresolved_release_items') is True
            and resolution.get('covers_all_negative_and_uncertain_first_judgments') is True
            and all(type(resolution.get(k)) is list and bool(resolution[k]) for k in ('disagreements', 'resolutions')),
            'explicit first-grade disagreement/uncertainty resolution required')


def verify_certificate(path, sha):
    """Verify preserved provenance and first-grade attestations; never read a key.

    Independence/semantic correctness are reviewer attestations, not facts this
    file can infer. Exact qualification derivation and mechanical completeness
    are independently recomputed. No synthetic passing certificate is supplied.
    """
    cert = pipeline.bound_json(path, sha)
    refs = ('qualification_manifest', 'qualification_preparation', 'qualification_collection', 'qualification_derived')
    keys = {'schema', *refs, 'qualification_key_sha256', 'qualification_policy_sha256',
            'auditors_qualified', 'independent_first_grades', 'adjudication'}
    pipeline.require(type(cert) is dict and set(cert) == keys and cert['schema'] == CERTIFICATE_SCHEMA,
                     'exact actual semantic-qualification certificate required')
    pipeline.require(cert['qualification_manifest']['sha256'] == QUALIFICATION_SHA
        and cert['qualification_key_sha256'] == KEY_SHA and cert['qualification_policy_sha256'] == POLICY_SHA
        and cert['auditors_qualified'] == {a: True for a in AUDITORS}
        and all(type(v) is bool for v in cert['auditors_qualified'].values()), 'qualification release bindings differ')
    artifacts = {k: pipeline.bound_json(cert[k]['path'], cert[k]['sha256']) for k in refs}
    manifest = artifacts['qualification_manifest']
    prepared = qualification.load_preparation(**dict(path=cert['qualification_preparation']['path'],
                                                     sha=cert['qualification_preparation']['sha256']))
    pipeline.require(manifest['settings'] == settings() and manifest['output_root'] == qualification.RUN_ROOT
        and manifest['stage'] == 'audit' and manifest['request_count'] == 48
        and manifest['source_sha256'].get(cert['qualification_preparation']['path']) == cert['qualification_preparation']['sha256'],
        'actual qualification manifest/preparation differs')
    sources = manifest['source_sha256']
    expected_binding = qualification.build_binding(manifest, QUALIFICATION_SHA,
        pipeline.bound_json(manifest['requests_path'], sources[manifest['requests_path']]), sources[manifest['requests_path']],
        pipeline.bound_json(manifest['lineage_path'], sources[manifest['lineage_path']]), sources[manifest['lineage_path']], prepared)
    pipeline.require(artifacts['qualification_collection']['stage_binding'] == expected_binding,
                     'qualification collection does not bind the actual released manifest')
    recomputed = qualification.derive(prepared, artifacts['qualification_collection'])
    derived = artifacts['qualification_derived']
    pipeline.require(all(derived.get(k) == v for k, v in recomputed.items())
        and artifacts['qualification_collection']['stage_binding']['manifest_sha256'] == QUALIFICATION_SHA
        and all(r['mechanical_contract_established'] for r in recomputed['rows']),
        'qualification collection/derivation is not the full mechanically established first version')
    attested = {k: deepcopy(cert[k]) for k in (*refs, 'qualification_key_sha256', 'qualification_policy_sha256', 'auditors_qualified')}
    grade_bindings = dict(qualification_manifest_sha256=QUALIFICATION_SHA,
        qualification_preparation_sha256=cert['qualification_preparation']['sha256'],
        raw_collection_sha256=cert['qualification_collection']['sha256'], derived_sha256=cert['qualification_derived']['sha256'],
        hidden_key_sha256=KEY_SHA, grading_policy_sha256=POLICY_SHA)
    grades = cert['independent_first_grades']
    pipeline.require(type(grades) is list and len(grades) == 2, 'two independent first grades required')
    graders, contexts, paths, hashes, first_results = set(), set(), set(), set(), []
    expected_mapping = {r['request_id']: (r['case_id'], r['auditor']) for r in recomputed['rows']}
    for ref in grades:
        grade = pipeline.bound_json(ref['path'], ref['sha256'])
        pipeline.require(set(grade) == {*grade_bindings, 'schema', 'grader_id', 'independent_context_id',
            'first_pass', 'independent_context', 'auditors_qualified', 'request_grades', 'limitations'} and grade['schema'] == GRADE_SCHEMA
            and grade['first_pass'] is True and grade['independent_context'] is True
            and all(grade[k] == v for k, v in grade_bindings.items())
            and set(grade['auditors_qualified']) == set(AUDITORS)
            and all(type(v) is bool or v is None for v in grade['auditors_qualified'].values())
            and all(type(grade[k]) is str and grade[k].strip() for k in ('grader_id', 'independent_context_id')),
            'independent first-grade attestation differs')
        request_grades = pipeline._index(grade['request_grades'], 'request_id')
        pipeline.require(set(request_grades) == set(expected_mapping) and all(
            (g['case_id'], g['auditor']) == expected_mapping[i]
            and type(g['atom_grades']) is list and all(type(atom) is dict and bool(atom) for atom in g['atom_grades'])
            and type(g['extra_assertion_issues']) is list and 'full_clean_acceptance' in g and 'notes' in g
            for i, g in request_grades.items())
            and all(sum(len(g['atom_grades']) for g in request_grades.values() if g['auditor'] == a) == 65 for a in AUDITORS),
            'first grade lacks all48 mappings/130 detailed mandatory-atom assessments')
        graders.add(grade['grader_id']); contexts.add(grade['independent_context_id'])
        paths.add(str(Path(ref['path']).resolve())); hashes.add(ref['sha256'])
        first_results.append(grade['auditors_qualified'])
    pipeline.require(all(len(v) == 2 for v in (graders, contexts, paths, hashes)), 'first grades are not distinct')
    ref = cert['adjudication']
    resolution = None if ref is None else pipeline.bound_json(ref['path'], ref['sha256'])
    validate_resolution(first_results, cert['auditors_qualified'], [g['sha256'] for g in grades], grade_bindings, resolution)
    return dict(certificate=descriptor(path, sha), **attested, independent_first_grades=deepcopy(grades),
        qualification_instructions_sha256=prepared['lineage']['instructions_sha256'],
        qualification_schema_content_sha256=prepared['lineage']['response_schema_content_sha256'],
        qualification_core_sha256=prepared['lineage']['source_sha256'][str(Path(core.__file__).relative_to(pipeline.ROOT))],
        adjudication=deepcopy(cert['adjudication']), semantic_independence_verified_by_code=False)


def reference_plan(manifest, ref):
    pipeline.require(manifest.get('stage') == 'rate'
        and manifest.get('output_root') == 'audits/attention_relational_report_v1/quote_reference_v3/rate',
        'the exact preselected V3 reference runtime manifest is required')
    return dict(name=JOIN_NAME, reference_manifest=deepcopy(ref), rule=deepcopy(FACTUAL_RULE),
        pairing='Every original narrator ID and literal-template ID with exact UTF-8 text SHA256; frozen blind_id correspondence.',
        version_selection='Whole V4 factual version and whole bound V3 reference version; no per-item fallback.',
        earlier_v3_whole_resource_v2_preselection_unchanged=True, join_implemented=False)


def assemble(units, pairs, proof, plan, inputs, sources):
    """Pure fictional-fixture boundary; CLI authorization is verify_certificate.

    Actual input loading below additionally validates the pinned original export,
    first-report collection, template construction and all25440 required atoms.
    """
    pipeline.require(len(units) == len({u['id'] for u in units}) == 552 and len(pairs) == 684,
                     'full552 source and684 pair opportunities required')
    pipeline.require(pairs == pipeline.build_pair_index([dict(id=u['id'], case=u['case']) for u in units]),
                     'fixed pair inventory differs')
    pipeline.require(proof.get('qualification_manifest', {}).get('sha256') == QUALIFICATION_SHA
        and proof.get('qualification_instructions_sha256') == pipeline.digest(core.INSTRUCTIONS.read_bytes())
        and proof.get('qualification_schema_content_sha256') == digest(core.response_schema())
        and proof.get('qualification_core_sha256') == pipeline.digest(Path(core.__file__).read_bytes()),
        'target interface differs from qualified interface')
    pipeline.require(plan.get('name') == JOIN_NAME and plan.get('rule') == FACTUAL_RULE,
                     'prospectively bound distinct reference-join plan required')
    requests, mapping = [], []
    for auditor in AUDITORS:
        for unit in units:
            text = unit['text']
            available = unit['actor_transport_complete'] is True and bool(text.strip())
            item = dict(source_id=unit['id'], auditor=auditor, actor_available=available,
                report_text_sha256=pipeline.digest(text.encode('utf-8')), request_id=None, unit_id=None, request_sha256=None)
            if available:
                identifier = 't' + digest([ID_SALT, auditor, unit['id']])[:24]
                request = core.build_request(identifier, text, unit['packet'])
                payload = pipeline.decoded(request['input'].rsplit('\n', 1)[1])
                item.update(request_id=identifier, unit_id=payload['unit_id'], request_sha256=pipeline.runner_request_sha256(request))
                requests.append(request)
            mapping.append(item)
    requests.sort(key=lambda r: pipeline.digest((ORDER_SALT + ':' + r['id']).encode('ascii')))
    pipeline.require(len(requests) == len({r['id'] for r in requests}) == 1074 and len(mapping) == 1104,
                     'exact2x537 requests and2x552 opportunities required')
    lineage = dict(schema=VERSION + '_lineage', inputs=deepcopy(inputs), source_sha256=deepcopy(sources),
        qualification_proof=deepcopy(proof), reference_join_plan=deepcopy(plan), factual_rule=deepcopy(FACTUAL_RULE),
        requests_content_sha256=digest(requests), mapping_content_sha256=digest(mapping), units_content_sha256=digest(units),
        pair_index_content_sha256=digest(pairs), response_schema_content_sha256=digest(core.response_schema()),
        instructions_sha256=pipeline.digest(core.INSTRUCTIONS.read_bytes()), id_salt=ID_SALT, order_salt=ORDER_SALT)
    return dict(schema=VERSION + '_preparation', units=deepcopy(units), pair_index=deepcopy(pairs), requests=requests,
        private_mapping=mapping, lineage=lineage, settings=settings(), output_root=RUN_ROOT,
        model_transport=dict(verbosity='low', store=False, sdk_max_retries=0, stateless=True))


def source_hashes():
    sources = qualification.source_hashes()
    paths = [Path(qualification.__file__), Path(__file__), pipeline.ROOT / 'tests/test_attention_relational_grounded_target_v4.py']
    sources.update({str(p.resolve().relative_to(pipeline.ROOT)): pipeline.digest(p.read_bytes()) for p in paths})
    return sources


def prepare_target(inputs):
    pipeline.require(set(inputs) == {'exports', 'reports', 'certificate', 'reference_manifest'}
        and inputs['exports']['sha256'] == EXPORT_SHA and inputs['reports']['sha256'] == REPORT_SHA,
        'pinned original inputs and actual certificate/reference manifest required')
    proof = verify_certificate(**dict(path=inputs['certificate']['path'], sha=inputs['certificate']['sha256']))
    loaded = {k: pipeline.bound_json(v['path'], v['sha256']) for k, v in inputs.items() if k != 'certificate'}
    exports, reports = loaded['exports'], loaded['reports']
    inventory = pipeline.validate_export(exports)
    attribution.validate_collected_records(reports)
    pipeline._validate_collection_lineage(reports, 'report', exports['requests'], exports)
    records = pipeline._records(reports)
    pipeline.require(set(records) == set(inventory['request_ids']), 'all552 original first-report positions required')
    units = [dict(id=i, case=inventory['provenance'][i]['case'], packet=inventory['packets'][i]['content'],
        text=pipeline._report_text(records[i]), actor_transport_complete=records[i]['transport_complete'],
        actor_availability=records[i].get('availability'), report=deepcopy(records[i].get('parsed')),
        template=inventory['templates'][i]['report']) for i in inventory['request_ids']]
    pipeline.require(sum(len(exporter.required_propositions(u['packet'])) for u in units) == 25440,
                     'original25440 atom denominator differs')
    pipeline.require(all(exporter.audit_report(u['packet'], u['template'])['listed_machine_facts_pass'] for u in units),
                     'literal template construction differs')
    plan = reference_plan(loaded['reference_manifest'], inputs['reference_manifest'])
    reference = loaded['reference_manifest']
    pipeline.require(reference.get('schema') == runner.SCHEMA and reference.get('request_count') == 1104
        and set(reference.get('request_ids', [])) == {pipeline.blind_id(u['id'] + suffix) for u in units for suffix in ('', '_template')}
        and EXPORT_SHA in reference.get('source_sha256', {}).values() and REPORT_SHA in reference.get('source_sha256', {}).values()
        and reference.get('settings', {}).get('output_schema', {}).get('properties', {}).get('schema') ==
            pipeline._enum('attention_relational_reference_v3_quote_annotation'),
        'reference plan does not bind the complete actual V3 quote-reference version and original texts')
    plan['ordered_texts_content_sha256'] = digest([dict(id=u['id'], narrator=pipeline.digest(u['text'].encode('utf-8')),
        template=pipeline.digest(u['template']['text'].encode('utf-8'))) for u in units])
    return assemble(units, pipeline.build_pair_index(exports['provenance']), proof, plan, inputs, source_hashes())


def validate_preparation(prepared):
    lineage = prepared['lineage']
    pipeline.require(prepared == assemble(prepared['units'], prepared['pair_index'], lineage['qualification_proof'],
        lineage['reference_join_plan'], lineage['inputs'], lineage['source_sha256']), 'target preparation differs')


def load_preparation(path, sha):
    pipeline.require(path is not None and sha is not None, 'bound target preparation required')
    prepared = pipeline.bound_json(path, sha)
    pipeline.require(prepared == prepare_target(prepared['lineage']['inputs']), 'actual target source/input lineage differs')
    return prepared


def build_binding(manifest, manifest_sha, requests, requests_sha, lineage, lineage_sha, prepared):
    validate_preparation(prepared)
    pipeline.require(manifest['stage'] == 'audit' and manifest['settings'] == settings()
        and manifest['output_root'] == RUN_ROOT and manifest['request_count'] == 1074
        and manifest['request_ids'] == [r['id'] for r in prepared['requests']]
        and requests == prepared['requests'] and lineage == prepared['lineage']
        and manifest['source_sha256'].get(manifest['requests_path']) == requests_sha
        and manifest['source_sha256'].get(manifest['lineage_path']) == lineage_sha, 'actual V4 target runtime differs')
    return dict(schema='attention_relational_report_v1_collection_binding', stage='audit', measurement_version=VERSION,
        manifest_sha256=manifest_sha, requests_sha256=requests_sha, lineage_sha256=lineage_sha,
        request_ids=manifest['request_ids'], requests=deepcopy(requests), lineage=deepcopy(lineage),
        output_root=RUN_ROOT, preparation_content_sha256=digest(prepared))


def auditor_result(request, record, mapping):
    evidence = core.derive_annotation(request, record.get('parsed'), expected_request_sha256=mapping['request_sha256'])
    complete = record['transport_complete'] is True
    established = complete and evidence['mechanical_valid']
    assertions = evidence['derived_assertions'] if established else None
    criteria = dict(transport_complete=complete, mechanical_valid=evidence['mechanical_valid'],
        complete_assertion_inventory=evidence['complete_assertion_inventory'] is True,
        unresolved_ambiguity_absent=evidence['unresolved_ambiguity'] is False,
        nonempty_assertions=bool(assertions),
        all_assertions_supported=bool(assertions) and all(a['assertion']['verdict'] == 'supported' for a in assertions))
    if not established:
        evidence['derived_assertions'] = None
    raw = record.get('raw_outcome')
    outcome = pipeline._unwrap_outcome(raw) if raw is not None else {}
    output_text = outcome.get('output_text')
    return dict(**deepcopy(mapping), status='established' if all(criteria.values()) else
        ('not_established' if established else 'unestablished'), criteria=criteria, derived_evidence=evidence,
        availability=record.get('availability'), storage_provenance=deepcopy(record.get('storage_provenance')),
        raw_response_content_sha256=digest(raw) if raw is not None else None,
        raw_output_text_sha256=pipeline.digest(output_text.encode('utf-8')) if type(output_text) is str else None)


def assess(prepared, collection):
    validate_preparation(prepared)
    attribution.validate_collected_records(collection)
    binding = collection['stage_binding']
    pipeline.require(binding.get('schema') == 'attention_relational_report_v1_collection_binding'
        and binding.get('measurement_version') == VERSION and binding.get('stage') == 'audit'
        and binding.get('output_root') == RUN_ROOT and binding['requests'] == prepared['requests']
        and binding['lineage'] == prepared['lineage'] and binding['preparation_content_sha256'] == digest(prepared)
        and binding['request_ids'] == [r['id'] for r in prepared['requests']]
        and collection.get('manifest_sha256', binding['manifest_sha256']) == binding['manifest_sha256'],
        'not the actual V4 target collection binding')
    records, requests = pipeline._records(collection), pipeline._index(prepared['requests'])
    pipeline.require(set(records) == set(requests) and len(records) == 1074, 'all1074 first/missing audit records required')
    audits, by_auditor = {}, defaultdict(Counter)
    for item in prepared['private_mapping']:
        if item['request_id'] is None:
            result = dict(**deepcopy(item), status='unestablished', criteria=None, derived_evidence=None,
                          availability='actor_unavailable', raw_response_content_sha256=None, raw_output_text_sha256=None)
        else:
            result = auditor_result(requests[item['request_id']], records[item['request_id']], item)
        audits[(item['source_id'], item['auditor'])] = result
        by_auditor[item['auditor']].update(opportunities=1, actor_available=int(item['actor_available']),
            **{result['status']: 1})
    rows, templates, groups = [], [], defaultdict(Counter)
    for unit in prepared['units']:
        machine = exporter.audit_report(unit['packet'], unit['report'])
        per_auditor = {a: audits[(unit['id'], a)] for a in AUDITORS}
        actor_available = unit['actor_transport_complete'] is True and bool(unit['text'].strip())
        faithful = actor_available and machine['listed_machine_facts_pass'] and all(
            r['status'] == 'established' for r in per_auditor.values())
        rows.append(dict(id=unit['id'], case=deepcopy(unit['case']), report_text_sha256=pipeline.digest(unit['text'].encode('utf-8')),
            actor_transport_complete=unit['actor_transport_complete'], actor_available=actor_available,
            actor_availability=unit.get('actor_availability'), machine_audit=machine, auditors=per_auditor,
            v4_factual_fidelity_established=faithful))
        template = unit['template']
        template_audit = exporter.audit_report(unit['packet'], template)
        pipeline.require(template_audit['listed_machine_facts_pass'], 'template is not source-constructed literal control')
        templates.append(dict(id=unit['id'] + '_template', case=deepcopy(unit['case']),
            report_text_sha256=pipeline.digest(template['text'].encode('utf-8')),
            literal_template_fidelity_by_construction=True, machine_audit=template_audit))
        key = '|'.join(str(unit['case'][k]) for k in ('encoder_seed', 'source', 'view', 'variant'))
        groups[key].update(narrator_opportunities=1, actor_available=int(actor_available),
            v4_factual_fidelity_established=int(faithful), template_opportunities=1,
            required_peak_atoms=machine['required_peak_count'], covered_peak_atoms=machine['covered_peak_count'])
    indexed, units = pipeline._index(rows), pipeline._index(prepared['units'])
    pairs = []
    for pair in prepared['pair_index']:
        left, right = (units[pair[k]] for k in ('first_id', 'second_id'))
        pairs.append(dict(**deepcopy(pair), source_packet_content_identical=digest(left['packet']) == digest(right['packet']),
            source_required_peak_atoms_identical=exporter.required_propositions(left['packet']) == exporter.required_propositions(right['packet']),
            exact_narrator_text_identical=left['text'] == right['text'],
            first_v4_factual_fidelity_established=indexed[left['id']]['v4_factual_fidelity_established'],
            second_v4_factual_fidelity_established=indexed[right['id']]['v4_factual_fidelity_established'],
            auditor_inventory_row_ids={a: [audits[(left['id'], a)]['request_id'], audits[(right['id'], a)]['request_id']] for a in AUDITORS}))
    pipeline.require(len(rows) == len(templates) == 552 and len(audits) == 1104 and len(pairs) == 684
        and all(by_auditor[a]['opportunities'] == 552 for a in AUDITORS), 'complete fixed denominators differ')
    return dict(schema=VERSION + '_factual_assessment', preparation_content_sha256=digest(prepared),
        raw_collection_content_sha256=digest(collection), lineage=deepcopy(prepared['lineage']), factual_rule=deepcopy(FACTUAL_RULE),
        narrator_opportunities=552, audit_opportunities=1104, audit_requests=1074, literal_template_opportunities=552,
        pair_opportunities=684, required_peak_atoms=sum(r['machine_audit']['required_peak_count'] for r in rows),
        v4_factual_fidelity_established=sum(r['v4_factual_fidelity_established'] for r in rows),
        by_auditor={a: dict(by_auditor[a]) for a in AUDITORS}, groups={k: dict(v) for k, v in groups.items()},
        rows=rows, literal_templates=templates, pairs=pairs, reference_join_implemented=False,
        wording_difference_is_not_semantic_change_proof=True,
        limits='Shared-model corroboration; code verifies grounding mechanics, not semantic completeness or source-of-qualia success.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'collect', 'assess'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    for name in ('exports', 'reports', 'certificate', 'reference-manifest', 'preparation', 'completion', 'collection'):
        parser.add_argument('--' + name, type=Path)
        parser.add_argument('--' + name + '-sha256')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--commit')
    args = parser.parse_args()
    pipeline.require(not args.output.exists(), 'refusing replacement of first artifact')
    destination = args.output.resolve()
    pipeline.require(destination.is_relative_to((pipeline.ROOT / BASE).resolve())
        and not destination.is_relative_to((pipeline.ROOT / RUN_ROOT).resolve()), 'outputs must stay outside separate target raw-run root')
    dependencies = {}
    if args.stage == 'prepare':
        for name in ('exports', 'reports', 'certificate', 'reference_manifest'):
            path, sha = getattr(args, name), getattr(args, name + '_sha256')
            pipeline.require(path is not None and sha is not None, 'bound actual input required: ' + name)
            runner.relative_path(str(path))
            dependencies[name] = descriptor(path, sha)
        runner.relative_path(str(args.output))
        prepared = prepare_target(dependencies)
        args.output.mkdir(parents=True, exist_ok=False)
        artifacts = {'preparation.json': prepared, 'requests.json': prepared['requests'], 'lineage.json': prepared['lineage'],
                     'private_mapping.json': prepared['private_mapping'], 'response_schema.json': core.response_schema()}
        for name, value in artifacts.items():
            pipeline.exclusive_json(args.output / name, value)
        sources = dict(prepared['lineage']['source_sha256'])
        sources.update({v['path']: v['sha256'] for v in dependencies.values()})
        proof = prepared['lineage']['qualification_proof']
        sources.update({v['path']: v['sha256'] for k, v in proof.items() if k.startswith('qualification_') and type(v) is dict})
        sources.update({v['path']: v['sha256'] for v in proof['independent_first_grades']})
        if proof['adjudication'] is not None:
            sources[proof['adjudication']['path']] = proof['adjudication']['sha256']
        sources.update({str(args.output / name): pipeline.digest((args.output / name).read_bytes()) for name in artifacts})
        pipeline.exclusive_json(args.output / 'runtime_manifest_draft.json', dict(schema=runner.SCHEMA, stage='audit',
            settings=settings(), request_count=1074, request_ids=[r['id'] for r in prepared['requests']], output_root=RUN_ROOT,
            requests_path=str(args.output / 'requests.json'), lineage_path=str(args.output / 'lineage.json'),
            review_path=str(args.output / 'runtime_review.json'), source_sha256=sources))
        return  # Actual certificate and an independent runtime release remain prerequisites.
    prepared = load_preparation(args.preparation, args.preparation_sha256)
    dependencies['preparation'] = descriptor(args.preparation, args.preparation_sha256)
    if args.stage == 'collect':
        manifest, requests, manifest_sha = runner.validate(str(args.manifest), args.commit)
        sources = manifest['source_sha256']
        pipeline.require(sources.get(str(args.preparation)) == args.preparation_sha256, 'runtime did not bind target preparation')
        lineage = pipeline.bound_json(manifest['lineage_path'], sources[manifest['lineage_path']])
        binding = build_binding(manifest, manifest_sha, requests, sources[manifest['requests_path']],
                                lineage, sources[manifest['lineage_path']], prepared)
        pipeline.require(args.completion is not None and args.completion_sha256 is not None, 'bound actual first completion required')
        result = pipeline.collect_first_responses(args.completion, args.completion_sha256, binding['request_ids'], RUN_ROOT, stage_binding=binding)
        dependencies.update(manifest=descriptor(args.manifest, manifest_sha), completion=descriptor(args.completion, args.completion_sha256))
    else:
        pipeline.require(args.collection is not None and args.collection_sha256 is not None, 'bound actual target collection required')
        dependencies['collection'] = descriptor(args.collection, args.collection_sha256)
        result = assess(prepared, pipeline.bound_json(args.collection, args.collection_sha256))
    result.update(input_dependencies=dependencies, source_sha256=source_hashes(), no_api_or_model_calls=True)
    pipeline.exclusive_json(args.output, result)


if __name__ == '__main__':
    main()
