"""Freeze all first source-review assignments; never launch or supply judgments."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
assert Path.cwd().resolve() == ROOT
BASE = Path('audits/attention_relational_report_v1')
SOURCE_HELPERS = dict(
    template_workspace=dict(path='scripts/attention_relational_template_review_workspace_v1.py',
        sha256='20834fcb9c47ff0376088d5800196cbbd42763348a10f329f96a689fad89060b'),
    source_calculator=dict(path='scripts/attention_relational_source_calculation_v1.py',
        sha256='0132764497baefb497162abb123f34bff68c167e8f39b86fea03539a55c4c10b'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path, expected):
    assert sha(path) == expected, str(path)
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return dict(path=str(path), sha256=sha(path))


settings = [
    dict(kind='template', directory=BASE / 'template_attribution_v1/semantic_review',
        manifest_sha='0f444a55011dffb7b53e1cb18f45a8304b44282aba9b65216eda30a2e0b094fe',
        protocol='docs/ATTENTION_RELATIONAL_TEMPLATE_SEMANTIC_REVIEW_V1.md',
        protocol_sha='e55a74c579593711a3ef40e308b4ca0b2d7b220f1a491a52d7f2ab2e481bf7a7',
        schema_file=BASE / 'template_attribution_v1/semantic_review/first_review_output_schema.json',
        schema_sha='f7a70c0ef14b0f6279e9ae2db40f34c48442493c4b25b05e159fe915529e1a27',
        instruction_sha='695171b440c945aed1d6fe19859854d8cad8724ca239ef5c95c5a3b0dcd11d23',
        collector='attention_relational_template_review_collection_v1',
        output_schema='attention_relational_template_semantic_review_v1_first_chunk_grade',
        task_prefix='template_semantic_v1', digits=2, expected_chunks=23),
    dict(kind='narrator', directory=BASE / 'narrator_source_closure_v1',
        manifest_sha='4340dec9e349fb74950408f00ed7da5c19f6bc2ad820eff5b41d2cea7819a200',
        protocol='docs/ATTENTION_RELATIONAL_NARRATOR_SOURCE_CLOSURE_V1_PLAN.md',
        protocol_sha='0010bcc069864443ee4ae241eb06250f6900f8f3ce2a3f6cbd2c5429d2ff81db',
        schema_file=BASE / 'narrator_source_closure_v1/protocol/first_review_output_schema.json',
        schema_sha='52799b865e3affec9a0b2602d1fc968ec0c5cbcd1b5b6b782c518aef7000d4ba',
        instruction_sha='df4c8b3cb3330904497ae3a62c39ed71e693146860e377345cbcd34b17ec6907',
        collector='attention_relational_narrator_source_closure_collection_v1',
        output_schema='attention_relational_narrator_source_closure_v1_first_chunk_review',
        task_prefix='narrator_closure_v1', digits=3, expected_chunks=1),
]
all_assignments, releases = [], []
for helper in SOURCE_HELPERS.values():
    assert sha(helper['path']) == helper['sha256']
for config in settings:
    directory = config['directory']
    manifest_path = directory / 'preparation/manifest.json'
    manifest = bound(manifest_path, config['manifest_sha'])
    reference = manifest['frozen_reference']
    bound(reference['path'], reference['sha256'])
    preflight_path = directory / 'preparation_execution/first_collection_preflight.json'
    preflight = json.loads(preflight_path.read_text())
    assert preflight['actual_production_loader_succeeded'] is True
    assert preflight['manifest_sha256'] == config['manifest_sha']
    assert preflight['pairs'] == 684 and preflight['narrator_positions'] == 552
    assert len(manifest['chunks']) == config['expected_chunks'] == preflight['chunks']
    assert sha(config['protocol']) == config['protocol_sha']
    bound(config['schema_file'], config['schema_sha'])
    instructions = directory / ('worker_instructions.txt' if config['kind'] == 'template' else 'protocol/worker_instructions.txt')
    assert sha(instructions) == config['instruction_sha']
    contexts, assignments = [], []
    for stream in range(2):
        reviewer = f'reviewer_{stream}'
        for number, chunk in enumerate(manifest['chunks']):
            suffix = f'{number:0{config["digits"]}d}'
            task = f'{config["task_prefix"]}_s{stream}_c{suffix}'
            context = '/root/' + task
            chunk_path = manifest_path.parent / chunk['path']
            assert sha(chunk_path) == chunk['sha256']
            work_directory = directory / 'judgments' / reviewer / ('chunk_' + suffix)
            first = work_directory / 'first_review.json'
            assert not first.exists() and not work_directory.exists()
            envelope = dict(schema=config['output_schema'], reviewer_id=reviewer, independent_context_id=context,
                first_pass=True, independent_context=True, manifest_sha256=config['manifest_sha'],
                chunk_sha256=chunk['sha256'], protocol_sha256=config['protocol_sha'], output_schema_sha256=config['schema_sha'],
                limitations=['Source-aware AI review of repeated development cases; shared training limits independence.',
                    'Source geometry and exact values do not alone establish semantic meaning.',
                    'Consult per-record notes for reported uncertainties.'])
            if config['kind'] == 'narrator':
                envelope['peer_judgments_supplied'] = False
            envelope_ref = save(directory / 'first_envelopes' / f'{reviewer}_chunk_{suffix}.json', envelope)
            contexts.append(dict(reviewer_id=reviewer, independent_context_id=context, chunk_number=number,
                chunk_sha256=chunk['sha256'], first_artifact_path=str(first), fresh_context=True,
                first_pass=True, peer_judgments_supplied=False))
            assignments.append(dict(kind=config['kind'], task_name=task, independent_context_id=context,
                reviewer_id=reviewer, chunk_number=number, manifest=dict(path=str(manifest_path), sha256=config['manifest_sha']),
                chunk=dict(path=str(chunk_path), sha256=chunk['sha256']), envelope=envelope_ref,
                first_artifact_path=str(first), first_working_directory=str(work_directory / 'work'),
                instructions=dict(path=str(instructions), sha256=config['instruction_sha']),
                protocol=dict(path=config['protocol'], sha256=config['protocol_sha']),
                output_schema=dict(path=str(config['schema_file']), sha256=config['schema_sha']),
                frozen_reference=reference, source_helpers=SOURCE_HELPERS))
    launch = save(directory / 'first_launch_receipt.json', dict(schema=config['collector'] + '_launch',
        manifest_sha256=config['manifest_sha'], protocol_sha256=config['protocol_sha'],
        output_schema_sha256=config['schema_sha'], contexts=contexts))
    releases.append(dict(kind=config['kind'], launch_receipt=launch,
        actual_preflight=dict(path=str(preflight_path), sha256=sha(preflight_path)), context_count=len(contexts)))
    all_assignments.extend(assignments)
assert len(all_assignments) == len({a['independent_context_id'] for a in all_assignments}) == 48
assert len({a['first_artifact_path'] for a in all_assignments}) == 48
save(Path(__file__).parent / 'first_assignments.json', dict(schema='attention_relational_source_review_v1_first_assignments',
    source_sha256=sha(__file__), releases=releases, assignments=all_assignments,
    context_ids_are_preassigned_names_not_observed_launches=True, reviewers_launched=False,
    peer_judgments_or_desired_verdicts_supplied=False, semantic_judgments_generated=False))
print(json.dumps(dict(contexts=len(all_assignments), releases=releases), indent=2))
