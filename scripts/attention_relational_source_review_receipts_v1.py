"""Bind the fixed template review's root observations for final collection.

Operational accounting only: no report reading, semantic grading, replacement,
or liveness inference. Root observation records assert the observed history;
this program verifies their bindings, not whether their tool claims are true.
No outcome file is written until all46 actual launches and terminal observations
are retained. A terminal context's null first artifact remains null. A declared
first artifact is retained unchanged even if its bytes later fail collection.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path('audits/attention_relational_report_v1/source_review_release_v1')
SEMANTIC = Path('audits/attention_relational_report_v1/template_attribution_v1/semantic_review')
ASSIGNMENTS_SHA = '1bb582594d6c757deb82bb6058d55a51b1789ea56775f378fc53fc93fd787148'
PLANNED_SHA = '5bd0a398a3079ebd82dbcb72f0bc4727056c6c616cedb4d50b816c33420343c7'
MANIFEST_SHA = '0f444a55011dffb7b53e1cb18f45a8304b44282aba9b65216eda30a2e0b094fe'
COLLECTOR_VERSION = 'attention_relational_template_review_collection_v1'
LAUNCH_SCHEMA = 'attention_relational_source_review_v1_actual_launch_observation'
CONFIRMATION_SCHEMA = 'attention_relational_source_review_v1_launch_observation_confirmation'
TERMINAL_SCHEMA = 'attention_relational_source_review_v1_actual_terminal_observation'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def assemble(assignments, planned, planned_sha, launch_records, terminal_records):
    """Use complete supplied metadata; absent observations do not imply terminal."""
    all_assigned = {a['independent_context_id']: a for a in assignments}
    require(len(all_assigned) == len(assignments), 'duplicate assigned context')
    template = {k: v for k, v in all_assigned.items() if v['kind'] == 'template'}
    require(len(template) == 46, 'all46 template assignments required')
    require(planned['schema'] == COLLECTOR_VERSION + '_launch', 'wrong planned receipt schema')
    plans = {p['independent_context_id']: p for p in planned['contexts']}
    require(set(plans) == set(template) and len(planned['contexts']) == 46,
            'complete fixed planned contexts required')
    for context, item in template.items():
        plan = plans[context]
        require(all(plan[k] == item[k] for k in ('reviewer_id', 'chunk_number'))
                and plan['chunk_sha256'] == item['chunk']['sha256']
                and plan['first_artifact_path'] == item['first_artifact_path'],
                'plan/assignment identity differs')
        require(plan['fresh_context'] is True and plan['first_pass'] is True
                and plan['peer_judgments_supplied'] is False, 'fixed first-context conditions differ')
    launched, ordered = {}, {'reviewer_0': [], 'reviewer_1': []}
    for descriptor, receipt in launch_records:
        if receipt['schema'] == CONFIRMATION_SCHEMA:
            # A preserved chronology correction confirms an existing launch;
            # it is not an additional launch. Its exact bytes enter the ledger.
            require(receipt['original_bytes_preserved'] is True
                    and receipt['actual_observed_status'] == 'running',
                    'launch confirmation differs')
            continue
        require(receipt['schema'] == LAUNCH_SCHEMA, 'unknown launch observation schema')
        require(receipt['assignments_sha256'] == ASSIGNMENTS_SHA
                and receipt['semantic_judgments_or_peer_feedback_supplied'] is False,
                'actual launch dependency/feedback differs')
        for spawn in receipt['observed_spawn_calls']:
            context = spawn['independent_context_id']
            require(context in all_assigned, 'unassigned actual context')
            item = all_assigned[context]
            require(spawn['spawn_tool_result']['task_name'] == context
                    and spawn['observed_status'] == 'running'
                    and spawn['requested_fork_turns'] == 'none'
                    and spawn['requested_model_override'] is None
                    and spawn['requested_reasoning_effort_override'] is None,
                    'actual fresh-context launch conditions differ')
            require(spawn['assigned_chunk'] == item['chunk']
                    and spawn['assigned_first_artifact_path'] == item['first_artifact_path'],
                    'actual launch source/path differs')
            if item['kind'] != 'template':
                continue
            require(context not in launched, 'duplicate actual launch')
            launched[context] = deepcopy(descriptor)
            ordered[item['reviewer_id']].append(item['chunk_number'])
    for stream, chunks in ordered.items():
        require(chunks == list(range(len(chunks))), 'within-stream fixed launch order differs: ' + stream)
    outcomes, pending = [], []
    for plan in planned['contexts']:
        context = plan['independent_context_id']
        reasons = []
        if context not in launched:
            reasons.append('actual_launch_observation_missing')
        if context not in terminal_records:
            reasons.append('root_terminal_observation_missing')
        if reasons:
            pending.append(dict(independent_context_id=context, reasons=reasons))
            continue
        descriptor, observed = terminal_records[context]
        require(observed['schema'] == TERMINAL_SCHEMA
                and observed['independent_context_id'] == context
                and observed['chunk_sha256'] == plan['chunk_sha256']
                and observed['observed_terminal'] is True
                and observed['observation_kind'] in ('completed', 'failed', 'interrupted'),
                'terminal observation binding differs')
        require('first_artifact' in observed, 'terminal observation omits first-artifact disposition')
        artifact = observed['first_artifact']
        if artifact is not None:
            require(set(artifact) == {'path', 'sha256'}
                    and artifact['path'] == plan['first_artifact_path']
                    and isinstance(artifact['sha256'], str) and len(artifact['sha256']) == 64,
                    'first artifact descriptor differs')
        outcomes.append(dict(reviewer_id=plan['reviewer_id'],
            independent_context_id=context, chunk_number=plan['chunk_number'],
            chunk_sha256=plan['chunk_sha256'], first_artifact=deepcopy(artifact),
            terminal_attestation=dict(independent_context_id=context,
                chunk_sha256=plan['chunk_sha256'], observed_terminal=True,
                observation_kind=observed['observation_kind'],
                observation_artifact=deepcopy(descriptor))))
    if pending:
        return dict(ready=False, pending=pending, outcome_receipt=None)
    return dict(ready=True, pending=[], outcome_receipt=dict(
        schema=COLLECTOR_VERSION + '_outcomes', launch_receipt_sha256=planned_sha,
        manifest_sha256=planned['manifest_sha256'], contexts=outcomes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check-readiness', action='store_true')
    action.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'run from repository root')
    dependencies = {}

    def read(path, expected=None):
        raw = path.read_bytes()
        observed_sha = digest(raw)
        require(expected is None or observed_sha == expected, 'input bytes differ: ' + str(path))
        dependencies[str(path)] = observed_sha
        return dict(path=str(path), sha256=observed_sha), json.loads(raw)

    _, assignment_data = read(RELEASE / 'first_assignments.json', ASSIGNMENTS_SHA)
    assignments = assignment_data['assignments']
    _, planned = read(SEMANTIC / 'first_launch_receipt.json', PLANNED_SHA)
    launches = [read(p) for p in sorted((RELEASE / 'launch_observations').glob('*.json'))]
    # Preserve and verify the correction's reference to the unchanged original.
    for _, observation in launches:
        if observation['schema'] == CONFIRMATION_SCHEMA:
            original = observation['original_receipt']
            require(dependencies.get(original['path']) == original['sha256'],
                    'launch correction does not bind unchanged original receipt')
    terminals = {}
    for item in assignments:
        if item['kind'] != 'template':
            continue
        path = RELEASE / 'terminal_observations' / (item['task_name'] + '.json')
        if path.exists():
            terminals[item['independent_context_id']] = read(path)
    result = assemble(assignments, planned, PLANNED_SHA, launches, terminals)
    print(json.dumps(dict(ready=result['ready'], template_contexts=46,
        retained_terminal_observations=len(terminals), pending=result['pending'])))
    if args.check_readiness:
        return
    require(result['ready'], 'not ready; no outcome file written')
    # Reuse the unchanged frozen receipt contract before writing actual inputs.
    import attention_relational_template_review_collection_v1 as collector
    _, manifest = read(SEMANTIC / 'preparation/manifest.json', MANIFEST_SHA)
    outcome = result['outcome_receipt']
    collector.validate_receipts(manifest, MANIFEST_SHA, planned, PLANNED_SHA, outcome)
    output = args.output
    binding = output.with_name(output.name + '.binding.json')
    require(not output.exists() and not binding.exists(), 'exclusive first output required')
    ledger = dict(schema='attention_relational_source_review_outcome_binding_v1',
        source_sha256=digest(Path(__file__).read_bytes()), input_sha256=dependencies,
        root_observation_claims_independently_certified_by_code=False,
        first_artifact_descriptors_repaired=False, missing_judgments_filled=False,
        semantic_judgments_generated=False, liveness_inferred=False,
        full_collection_executed=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(outcome, handle, indent=2)
        handle.write('\n')
    ledger['outcome_receipt'] = dict(path=str(output), sha256=digest(output.read_bytes()))
    with binding.open('x') as handle:
        json.dump(ledger, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
