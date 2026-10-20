"""Finite fresh-context packet preparation and first-artifact collection.

No models, archives, API, Git, worker launch, truth grading or retries are provided.
All writes are exclusive. All planned contexts require actual root terminal
observations before any first response is opened. Root observation JSON:
 {"schema":"same_address_pilot_v1_root_terminal_observations",
  "observed_by":"root", "observations":[
   {"context_id":"R_ctx_01", "terminal":true,
    "state":"completed|failed|missing|skipped_unavailable",
    "actor_id":"actual agent ID or null", "observed_at_utc":"...",
    "evidence":"actual terminal observation, including missing context"}, ...]}
Requires jsonschema (available in the system Python environment). Use python3
for the CLI; toy pytest runs can expose system packages to the repository venv:
 PYTHONPATH=/usr/lib/python3/dist-packages .venv/bin/python -m pytest -q
 tests/test_same_address_model_report_workspace_v1.py
CLI stages: prepare-reporter, collect-reporter, prepare-reviews, collect-reviews.
Use --help on each stage for explicit input/output paths. Every output directory
is newly claimed; collection outputs and actor first.json use exclusive writes.
Do not alter the frozen protocol package or rerun preparation into an old path.

Root numerical release JSON:
 {"status":"released", "released_by":"root", "source_receipt_sha256":"..."}.
The release is root's source decision, not a new scientific qualification test.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

import jsonschema

VERSION = 'same_address_model_report_workspace_v1'
ENVELOPE = 'same_address_pilot_v1_first_context_responses'
STAGES = {'R': 'reporter', 'F1': 'factual', 'F2': 'factual', 'S1': 'semantic', 'S2': 'semantic'}
ASSETS = {
 'reporter': ['narrator_prompt.txt', 'narrator_response.schema.json'],
 'factual': ['factual_first_instructions.txt', 'factual_first.schema.json'],
 'semantic': ['semantic_first_wrapper.txt', 'rater_instructions.txt', 'reference.json', 'semantic_first.schema.json'],
}
SCHEMAS = {'reporter': 'narrator_response.schema.json', 'factual': 'factual_first.schema.json', 'semantic': 'semantic_first.schema.json'}


def require(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON object key: ' + key)
        result[key] = value
    return result


def decoded(raw):
    return json.loads(raw.decode('utf-8'), object_pairs_hook=reject_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))


def read_json(path):
    return decoded(Path(path).read_bytes())


def exclusive(path, raw):
    with Path(path).open('xb') as handle:
        handle.write(raw)


def exclusive_json(path, value):
    exclusive(path, encoded(value))


def descriptor(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path.read_bytes())}


def bound(desc):
    raw = Path(desc['path']).read_bytes()
    require(digest(raw) == desc['sha256'], 'bound input hash differs: ' + desc['path'])
    return decoded(raw)


def public_content(content):
    """Positive key allowlists prevent private labels entering actor packets."""
    require(type(content) is dict and set(content) == {'schema', 'decision_time', 'selected_node', 'queried_role',
        'panels', 'coordinate_system', 'field_semantics', 'conditional_semantics', 'processing_semantics'}, 'unexpected public packet fields')
    require(content['schema'] == 'same_address_model_report_source_v1_packet' and content['selected_node'] == 'n3'
        and content['decision_time'] == 12 and content['queried_role'] in ('red', 'blue'), 'public address/time/role differs')
    require(set(content['coordinate_system']) == {'rows', 'columns', 'row_direction', 'column_direction', 'separate_local_grid'}, 'coordinate metadata differs')
    require(len(content['panels']) in (1, 2), 'one or two declared panels required')
    for index, panel in enumerate(content['panels']):
        require(set(panel) == {'panel', 'panel_role', 'current', 'candidates', 'processing_scores'}, 'private/unexpected panel fields')
        require(panel['panel'] == f'p{index}' and panel['panel_role'] in ('predictor_estimate', 'native_reference'), 'panel name/role differs')
        require(set(panel['candidates']) == {'k0', 'k1', 'k2'}, 'three operations required')
        require(set(panel['processing_scores']) == {'rule', 'scores', 'tied_winners', 'first_index_choice', 'top_two_margin'}, 'score fields differ')
        require(set(panel['processing_scores']['scores']) == {'k0', 'k1', 'k2'}, 'score operation keys differ')
        for field in [panel['current'], *panel['candidates'].values()]:
            require(set(field) == {'node', 'role', 'origin', 'distribution', 'derived'}, 'private/unexpected field keys')
            require(field['node'] == 'n3' and field['role'] == content['queried_role'], 'field referent differs')
            require(set(field['origin']) <= {'kind', 'represented_time', 'feedback_through', 'executed_branch_claim', 'archived_origin_time', 'fixed_command', 'record_time'}, 'private/unexpected origin keys')
            require(set(field['derived']) == {'peaks', 'peak_probability', 'entropy_nats', 'expected_row', 'expected_column'}, 'derived fields differ')
            require(type(field['distribution']) is list and len(field['distribution']) == 4
                and all(type(row) is list and len(row) == 4 for row in field['distribution']), '4x4 distribution required')
    return deepcopy(content)


def load_protocol(protocol):
    protocol = Path(protocol).resolve()
    manifest = read_json(protocol / 'manifest.json')
    for path, sha in manifest['owned_files_sha256'].items():
        # Basenames are unique owned assets; do not let a manifest relocate reads.
        local = protocol / Path(path).name
        require(local.is_file() and digest(local.read_bytes()) == sha, 'protocol binding differs: ' + local.name)
    inventory = read_json(protocol / 'assignment_inventory.json')
    require(len(inventory['assignments']) == 72 and len(inventory['context_assignments']) == 60, 'fixed assignments required')
    require({r['unit_id'] for r in inventory['assignments']} == {f'unit_{i:03d}' for i in range(1,73)}, 'neutral unit set differs')
    for stream in STAGES:
        chunks = [c for c in inventory['context_assignments'] if c['stream'] == stream]
        require(len(chunks) == 12 and {c['chunk'] for c in chunks} == set(range(1,13)), 'fixed twelve contexts required')
        require(all(len(c['unit_ids']) == 6 for c in chunks), 'six assigned units/context required')
        require([u for c in chunks for u in c['unit_ids']] == [r['unit_id'] for r in inventory['assignments']], 'context order differs')
    return inventory


def outer_schema(stage, unit_ids, per_unit):
    response = deepcopy(per_unit)
    if stage != 'reporter':
        response = {'anyOf': [response, {'type':'null'}]}
    return {'$schema':'https://json-schema.org/draft/2020-12/schema', 'type':'object', 'additionalProperties':False,
        'required':['schema','stage','stream','chunk','input_sha256','instructions_sha256','context_id','responses'],
        'properties': {'schema': {'const':ENVELOPE}, 'stage':{'const':stage}, 'stream':{'enum':[s for s,v in STAGES.items() if v == stage]},
        'chunk':{'type':'integer','minimum':1,'maximum':12}, 'input_sha256':{'type':'string'}, 'instructions_sha256':{'type':'string'},
        'context_id':{'type':'string'}, 'responses':{'type':'array','minItems':6,'maxItems':6,'items':{'type':'object','additionalProperties':False,
        'required':['unit_id','response'],'properties':{'unit_id':{'enum':unit_ids},'response':response}}}}}


WORKER = '''You are a fresh AI actor with no inherited conversation. Read ONLY the files listed in allowed_files.json in this context directory; do not list directories or read any other file. Do not use Git, network, APIs, models, global metadata, assignment inventories or parent history. Do not contact other actors. Tools may only read the listed files and exclusively write first.json in this directory once. No other writes, retries, repairs or replacement output are permitted.
Apply the per-unit instructions unchanged to each of the six units separately. The outer JSON envelope is transport framing only, not an additional substantive report prompt or semantic rubric. Return exactly one envelope matching response.schema.json and expected.json: six responses in the supplied unit order, each {unit_id,response:<unchanged per-unit object>}. Where a per-unit object itself has unit_id, it must equal the outer unit_id. For review positions with report_text=null, do not invent text or an annotation; response must be null. Treat missing text as unavailable, not a negative feature. For factual review, source_json_pointer is relative to that assigned unit's complete input frame, including source_payload and source_ownership. No claims about support or semantic structure are inferred from formatting.
Preserve the first response even if incomplete, malformed, refused or failed. Do not reread peer responses or improve any first. Exclusively write the raw first envelope to first.json, then return its path and end the context. Do not run a validator or review the output to repair it. Root will observe all contexts terminal before opening first artifacts.
'''


def make_workspace(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    return out


def prepare_context(out, protocol, context, units):
    stream, chunk = context['stream'], context['chunk']
    stage = STAGES[stream]
    # Paths carry stream/chunk but never source/condition or design identifier.
    folder = out / f'{stream.lower()}_{chunk:02d}'
    folder.mkdir()
    input_value = {'schema':VERSION + '_input', 'stage':stage,'stream':stream,'chunk':chunk,
                   'context_id':context['context_id'],'units':units}
    exclusive_json(folder/'input.json', input_value)
    assets = {}
    for name in ASSETS[stage]:
        raw = (protocol/name).read_bytes()
        exclusive(folder/name, raw)
        assets[name] = digest(raw)
    exclusive(folder/'worker.txt', WORKER.encode())
    assets['worker.txt'] = digest(WORKER.encode())
    schema = outer_schema(stage, context['unit_ids'], read_json(protocol/SCHEMAS[stage]))
    exclusive_json(folder/'response.schema.json', schema)
    assets['response.schema.json'] = digest(encoded(schema))
    instructions_sha = digest(encoded(assets))
    expected = {'schema':ENVELOPE,'stage':stage,'stream':stream,'chunk':chunk,
        'context_id':context['context_id'],'input_sha256':digest(encoded(input_value)),
        'instructions_sha256':instructions_sha,'unit_ids':context['unit_ids']}
    exclusive_json(folder/'expected.json', expected)
    allowed = ['input.json', 'expected.json', 'response.schema.json', 'worker.txt', *ASSETS[stage]]
    exclusive_json(folder/'allowed_files.json', allowed)
    return {**expected,'directory':str(folder),'first_path':str(folder/'first.json'),
            'allowed_files':{name:descriptor(folder/name) for name in ['allowed_files.json',*allowed]},
            'instruction_asset_sha256':assets,'all_unavailable':stage != 'reporter' and all(u['report_text'] is None for u in units)}


def base_manifest(protocol, inventory, contexts, stage, backend_disclosure):
    return {'schema':VERSION+'_prepared','stage':stage,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'code':descriptor(__file__),'protocol_manifest':descriptor(protocol/'manifest.json'),
        'assignment_inventory':descriptor(protocol/'assignment_inventory.json'),
        'protocol_directory':str(protocol),'contexts':contexts,'planned_units':72,
        'actor_method':{'kind':'fresh_ai_subagent','fork_turns':'none','model':'inherited','model_override':None,
            'units_per_context':6,'provider_settings':{'temperature':None,'seed':None,'max_output_tokens':None},
            'unsupported_settings_are_null':True,'actual_backend_disclosure':backend_disclosure},
        'environment':{'python':sys.version,'platform':platform.platform(),
            'jsonschema_version':importlib.metadata.version('jsonschema'),'jsonschema_module':jsonschema.__file__},
        'runtime_release':'root must publish exact runtime/backend disclosure before first actor launch',
        'automatic_auditor_qualification':None,'semantic_or_factual_grading':False,'retry_allowed':False}


def prepare_reporter(source_dir, protocol_dir, output_dir, numerical_release_path, backend_disclosure=None):
    source_dir, protocol = Path(source_dir).resolve(), Path(protocol_dir).resolve()
    inventory = load_protocol(protocol)
    receipt_desc = descriptor(source_dir/'first_receipt.json')
    receipt = bound(receipt_desc)
    release_desc = descriptor(numerical_release_path)
    release = bound(release_desc)
    require(release.get('status') == 'released' and release.get('released_by') == 'root'
        and release.get('source_receipt_sha256') == receipt_desc['sha256'], 'root numerical release must bind completed source receipt')
    require(receipt.get('schema') == 'same_address_model_report_source_v1_receipt' and receipt.get('status') == 'completed'
        and receipt.get('opportunity_count') == 72 and receipt.get('pair_count') == 84, 'completed bounded source receipt required')
    packet_desc = descriptor(source_dir/'neutral_packets.json')
    require(receipt['artifact_sha256']['neutral_packets.json'] == packet_desc['sha256'], 'neutral packets differ from source receipt')
    packets = bound(packet_desc)
    require(type(packets) is list and len(packets) == 72, 'complete72 packet inventory required')
    indexed = {p['id']:p for p in packets}
    require(len(indexed) == 72 and set(indexed) == {r['design_opportunity_id'] for r in inventory['assignments']}, 'source assignment set differs')
    frames = {}
    for row in inventory['assignments']:
        packet = indexed[row['design_opportunity_id']]
        content = public_content(packet['content'])
        # Source builder's encoded() format is its own exact content binding.
        source_encoded = json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        require(packet['content_sha256'] == digest(source_encoded+b'\n'), 'source content hash differs')
        frames[row['unit_id']] = {'unit_id':row['unit_id'],'source_payload':content}
    out = make_workspace(output_dir)
    contexts = [prepare_context(out, protocol, c, [frames[u] for u in c['unit_ids']])
                for c in inventory['context_assignments'] if c['stream'] == 'R']
    manifest = base_manifest(protocol, inventory, contexts, 'reporter', backend_disclosure)
    manifest.update(source_packets=packet_desc, source_receipt=receipt_desc, numerical_release=release_desc,
        unit_input_sha256={unit_id:digest(encoded(frame)) for unit_id,frame in frames.items()},
        public_content_sha256={row['unit_id']:indexed[row['design_opportunity_id']]['content_sha256'] for row in inventory['assignments']})
    exclusive_json(out/'manifest.json', manifest)
    return manifest


def validate_prepared(prepared):
    require(digest(Path(prepared['code']['path']).read_bytes()) == prepared['code']['sha256'], 'workspace code binding differs')
    bound(prepared['protocol_manifest'])
    inventory = bound(prepared['assignment_inventory'])
    streams = {'R'} if prepared['stage'] == 'reporter' else {'F1','F2','S1','S2'}
    require(prepared['stage'] in ('reporter','reviews'), 'unknown preparation stage')
    planned = [c for c in inventory['context_assignments'] if c['stream'] in streams]
    require(len(prepared['contexts']) == len(planned) == (12 if prepared['stage'] == 'reporter' else 48), 'full planned context inventory required')
    require(len({c['context_id'] for c in prepared['contexts']}) == len(planned), 'duplicate prepared context ID')
    for context, assigned in zip(prepared['contexts'], planned):
        require(all(context[k] == assigned[k] for k in ['stream','chunk','context_id','unit_ids']), 'prepared context differs from frozen assignment')
        require(context['stage'] == STAGES[context['stream']], 'prepared stream/stage differs')
        for name, sha in context['instruction_asset_sha256'].items():
            require(digest(Path(context['directory'],name).read_bytes()) == sha, 'exact instruction asset binding differs')
        for desc in context['allowed_files'].values():
            raw = Path(desc['path']).read_bytes()
            require(digest(raw) == desc['sha256'], 'prepared actor file binding differs')
        require(digest(Path(context['directory'],'input.json').read_bytes()) == context['input_sha256'], 'prepared input binding differs')
        require(digest(encoded(context['instruction_asset_sha256'])) == context['instructions_sha256'], 'instruction bundle binding differs')


def terminal_gate(prepared, terminal_observations_path):
    observed = read_json(terminal_observations_path)
    require(observed.get('schema') == 'same_address_pilot_v1_root_terminal_observations' and observed.get('observed_by') == 'root', 'actual root terminal observations required')
    rows = observed.get('observations', [])
    require(type(rows) is list and len(rows) == len(prepared['contexts']), 'all planned terminal observations required')
    indexed = {r.get('context_id'):r for r in rows}
    require(len(indexed) == len(rows) and set(indexed) == {c['context_id'] for c in prepared['contexts']}, 'terminal context inventory differs')
    actor_ids = [r.get('actor_id') for r in rows if isinstance(r.get('actor_id'), str)]
    require(len(actor_ids) == len(set(actor_ids)), 'actual actor contexts must be distinct')
    for c in prepared['contexts']:
        row = indexed[c['context_id']]
        require(row.get('terminal') is True and row.get('state') in ('completed','failed','missing','skipped_unavailable')
            and isinstance(row.get('observed_at_utc'), str) and bool(row['observed_at_utc'])
            and isinstance(row.get('evidence'), str) and bool(row['evidence'].strip()), 'terminal record lacks actual observation')
        if row['state'] == 'completed':
            require(isinstance(row.get('actor_id'), str) and bool(row['actor_id']), 'completed context requires actual actor identifier')
        if row['state'] == 'skipped_unavailable':
            require(c['all_unavailable'], 'cannot skip a context with available report positions')
    return indexed


def resolve_pointer(frame, pointer):
    require(isinstance(pointer, str) and (pointer == '' or pointer.startswith('/')), 'JSON Pointer must be relative to assigned unit frame')
    current = frame
    for part in pointer.split('/')[1:]:
        part = part.replace('~1','/').replace('~0','~')
        if isinstance(current, list):
            require(part.isdigit() and (part == '0' or not part.startswith('0')), 'invalid array pointer')
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise ValueError('JSON Pointer traverses a scalar')
    return current


def annotation_diagnostics(protocol, stage, frame, annotation):
    validator_path = protocol/'structural_validate.py'
    namespace = {'__name__':'same_address_protocol_structural_diagnostics', '__file__':str(validator_path)}
    # Execute the bound diagnostic module without writing bytecode into protocol/.
    exec(compile(validator_path.read_text(), str(validator_path), 'exec'), namespace)
    result = namespace['inspect'](stage, frame['report_text'], annotation, read_json(protocol/SCHEMAS[stage]))
    if stage == 'factual':
        for claim in annotation.get('claims', []):
            for operand in claim.get('operands', []):
                try:
                    resolve_pointer(frame, operand['source_json_pointer'])
                except (ValueError, KeyError, IndexError, TypeError):
                    result['format_diagnostics'].append('source JSON Pointer does not resolve: '+str(operand.get('source_json_pointer')))
    return result


def collect(prepared_manifest_path, terminal_observations_path, output_path, expected_stage):
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    prepared_desc = descriptor(prepared_manifest_path)
    prepared = bound(prepared_desc)
    require(prepared['stage'] == expected_stage, 'wrong prepared stage')
    validate_prepared(prepared)
    # This gate intentionally precedes every first_path exists/open operation.
    terminals = terminal_gate(prepared, terminal_observations_path)
    protocol = Path(prepared['protocol_directory'])
    output = {'schema':VERSION+'_collection','stage':expected_stage,'prepared_manifest':prepared_desc,
        'terminal_observations':descriptor(terminal_observations_path),'all_contexts_observed_terminal':True,
        'contexts':[],'units':[],'semantic_or_factual_grading':False,'retry_performed':False}
    for context in prepared['contexts']:
        frames = read_json(Path(context['directory'])/'input.json')['units']
        frame_by_id = {f['unit_id']:f for f in frames}
        record = {'context_id':context['context_id'],'stream':context['stream'],'chunk':context['chunk'],
            'terminal_observation':terminals[context['context_id']], 'raw_base64':None,'raw_sha256':None,
            'parsed':None,'format_diagnostics':[],'status':'unestablished'}
        first_path = Path(context['first_path'])
        raw = None
        if first_path.is_file():
            raw = first_path.read_bytes()
            record.update(raw_base64=base64.b64encode(raw).decode(),raw_sha256=digest(raw))
        if raw is None:
            record['format_diagnostics'].append('first artifact unavailable')
        elif terminals[context['context_id']]['state'] in ('missing', 'skipped_unavailable'):
            record['format_diagnostics'].append('first artifact conflicts with root missing/skipped observation')
        else:
            try:
                parsed = decoded(raw)
                record['parsed'] = parsed
                jsonschema.Draft202012Validator(read_json(Path(context['directory'])/'response.schema.json')).validate(parsed)
                for key in ['schema','stage','stream','chunk','context_id','input_sha256','instructions_sha256']:
                    require(parsed[key] == context[key], 'outer binding differs: '+key)
                require([r['unit_id'] for r in parsed['responses']] == context['unit_ids'], 'response unit order/set differs')
                diagnostics = {}
                for response_row in parsed['responses']:
                    unit_id, response = response_row['unit_id'], response_row['response']
                    frame = frame_by_id[unit_id]
                    unavailable = expected_stage == 'reviews' and frame['report_text'] is None
                    require((response is None) == unavailable, 'null response only permitted for unavailable report position')
                    if response is not None and 'unit_id' in response:
                        require(response['unit_id'] == unit_id, 'inner/outer unit ID differs')
                    if response is not None and expected_stage == 'reviews':
                        diagnostics[unit_id] = annotation_diagnostics(protocol, STAGES[context['stream']], frame, response)
                record['unit_diagnostics'] = diagnostics
                require(not any(d['format_diagnostics'] for d in diagnostics.values()), 'one or more per-unit structural diagnostics; entire context unestablished')
                record['status'] = 'structurally_recorded'
            except (ValueError, KeyError, TypeError, AttributeError, IndexError, jsonschema.ValidationError) as exc:
                record['format_diagnostics'].append(str(exc))
        # Never salvage a row from a malformed whole context.
        for unit_id in context['unit_ids']:
            unit = {'unit_id':unit_id,'stream':context['stream'],'context_id':context['context_id'],
                'availability':'unestablished','response':None,'word_count':None,'feature_or_support_status':None}
            if record['status'] == 'structurally_recorded':
                response = next(r['response'] for r in record['parsed']['responses'] if r['unit_id'] == unit_id)
                unit['response'] = response
                if response is None:
                    unit['availability'] = 'report_unavailable'
                elif expected_stage == 'reporter':
                    unit['word_count'] = len(response['text'].split())
                    unit['availability'] = 'empty' if not response['text'].strip() else ('over_word_limit' if unit['word_count'] > 120 else 'available')
                else:
                    unit['availability'] = 'annotation_recorded'
            elif expected_stage == 'reviews' and frame_by_id[unit_id]['report_text'] is None:
                unit['availability'] = 'report_unavailable'
            output['units'].append(unit)
        output['contexts'].append(record)
    exclusive_json(output_path, output)
    return output


def collect_reporter(prepared_manifest_path, terminal_observations_path, output_path):
    return collect(prepared_manifest_path, terminal_observations_path, output_path, 'reporter')


def source_ownership(row, content):
    return {'panels':[{'panel':p['panel'], 'actual_source_owner': 'W' if p['panel_role'] == 'native_reference' else row['source'],
        'current_origin':p['current']['origin'], 'candidate_origins':{k:f['origin'] for k,f in p['candidates'].items()}}
        for p in content['panels']]}


def prepare_reviews(reporter_manifest_path, reporter_collection_path, output_dir, backend_disclosure=None):
    reporter = read_json(reporter_manifest_path)
    validate_prepared(reporter)
    collected = read_json(reporter_collection_path)
    require(collected['stage'] == 'reporter' and collected['all_contexts_observed_terminal'] is True
        and collected['prepared_manifest'] == descriptor(reporter_manifest_path), 'bound terminal Reporter collection required')
    # Recheck the retained root observation artifact; no peer review outputs exist here.
    observed = bound(collected['terminal_observations'])
    terminal_gate(reporter, collected['terminal_observations']['path'])
    require(len(collected['contexts']) == 12 and len(collected['units']) == 72, 'complete Reporter collection inventory required')
    reports = {u['unit_id']:u for u in collected['units']}
    require(len(reports) == 72 and all(u['stream'] == 'R' for u in reports.values()), 'Reporter unit inventory differs')
    protocol = Path(reporter['protocol_directory'])
    inventory = load_protocol(protocol)
    assignments = {r['unit_id']:r for r in inventory['assignments']}
    packets = {p['id']:p for p in bound(reporter['source_packets'])}
    frames = {}
    for unit_id, row in assignments.items():
        first = reports[unit_id]
        content = public_content(packets[row['design_opportunity_id']]['content'])
        # Invalid Reporter word bounds/empty/malformed artifacts do not invent a review text.
        text = first['response']['text'] if first['availability'] == 'available' else None
        frames[unit_id] = {'unit_id':unit_id,'report_text':text,'source_payload':content,
                          'source_ownership':source_ownership(row, content)}
    out = make_workspace(output_dir)
    contexts = []
    for c in inventory['context_assignments']:
        if c['stream'] == 'R':
            continue
        units = [frames[u] if STAGES[c['stream']] == 'factual' else
                 {'unit_id':u,'report_text':frames[u]['report_text']} for u in c['unit_ids']]
        contexts.append(prepare_context(out, protocol, c, units))
    manifest = base_manifest(protocol, inventory, contexts, 'reviews', backend_disclosure)
    manifest.update(reporter_manifest=descriptor(reporter_manifest_path), reporter_collection=descriptor(reporter_collection_path),
                    source_packets=reporter['source_packets'])
    exclusive_json(out/'manifest.json', manifest)
    return manifest


def collect_reviews(prepared_manifest_path, terminal_observations_path, output_path):
    return collect(prepared_manifest_path, terminal_observations_path, output_path, 'reviews')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare-reporter')
    for name in ['source-dir','protocol-dir','output-dir','numerical-release']:
        prep.add_argument('--'+name, type=Path, required=True)
    review = sub.add_parser('prepare-reviews')
    for name in ['reporter-manifest','reporter-collection','output-dir']:
        review.add_argument('--'+name, type=Path, required=True)
    for command in ['collect-reporter','collect-reviews']:
        child = sub.add_parser(command)
        for name in ['prepared-manifest','terminal-observations','output']:
            child.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'prepare-reporter':
        result = prepare_reporter(args.source_dir,args.protocol_dir,args.output_dir,args.numerical_release)
    elif args.command == 'prepare-reviews':
        result = prepare_reviews(args.reporter_manifest,args.reporter_collection,args.output_dir)
    else:
        fn = collect_reporter if args.command == 'collect-reporter' else collect_reviews
        result = fn(args.prepared_manifest,args.terminal_observations,args.output)
    print(json.dumps({'schema':VERSION+'_operation_receipt','operation':args.command,
        'planned_contexts':len(result['contexts']),'automatic_auditor_qualification':None},sort_keys=True))


if __name__ == '__main__':
    main()
