"""Prepare/replay synthetic task-vocabulary cases; no experimental outcomes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path('audits/ai_task_audit_review_v1')
SOURCE = Path('audits/functional_task_interface_v1/examples.json')


def build():
    examples = {r['example']: r['payload'] for r in json.loads(SOURCE.read_text())}
    sources = {key: copy.deepcopy(examples[key]) for key in ('x1', 'x2', 'x3')}
    sources['history_withheld'] = copy.deepcopy(examples['x2'])
    sources['history_withheld']['observed_history'] = None
    sources['one_known_attribute'] = copy.deepcopy(examples['x3'])
    output = next(n for n in sources['one_known_attribute']['predicted_current']
                  if n['node'] == sources['one_known_attribute']['output_node'])
    obj = next(o for o in output['objects'] if o['position'] == 'p0')
    obj['color_distribution'] = {'red': .05, 'green': .05, 'blue': .05, 'yellow': .85}
    obj['identified_color'] = 'yellow'
    obj['shape_distribution'] = {'circle': .25, 'square': .25, 'triangle': .25, 'cross': .25}
    obj['identified_shape'] = None
    statements = [
        ('x1', 'The Controller selected k0, but the command has not executed and the task response is pending.'),
        ('x1', 'The Controller abstained from answering the task.'),
        ('x2', 'The Controller answered yellow and triangle for the query at n2 p0.'),
        ('x2', 'The answer proves the physical scene contains a yellow triangle at p0.'),
        ('x2', 'The Controller is 60% certain that its answer is correct.'),
        ('x3', 'The task response is abstained, with no emitted color or shape.'),
        ('x3', 'Abstention proves that decoding either attribute is impossible.'),
        ('x2', 'The most recent observed command is k0, matching the executed selected command.'),
        ('x2', 'The most recent executed command is k3.'),
        ('x2', 'The task query addresses n0 p0.'),
        ('x2', 'The output n2 itself allocates attention to p0.'),
        ('x2', 'A modeled step of 13 proves that 13 task answers have been emitted.'),
        ('history_withheld', 'Observation history is withheld, so the selected command has not executed.'),
        ('history_withheld', 'The record marks the selected command as executed even though observation history is withheld.'),
        ('one_known_attribute', 'The Controller abstains although its output identifies the queried color as yellow.'),
        ('one_known_attribute', 'Both queried attributes have no identified category because the Controller abstains.'),
    ]
    records = [{'id': f'T{i:03d}', 'synthetic': True, 'report': text,
                'instruction': 'Audit all explicit propositions; no requested coverage score.',
                'payload': copy.deepcopy(sources[key])}
               for i, (key, text) in enumerate(statements, 1)]
    data = (json.dumps(records, indent=2) + '\n').encode()
    inputs = [str(SOURCE), 'docs/FUNCTIONAL_TASK_SOURCE_INVENTORY.md',
              'docs/AI_TASK_AUDIT_REVIEW_PROTOCOL_V1.md', __file__]
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs}
    manifest = {'kind': 'synthetic_task_vocabulary_review', 'date': '2026-10-10',
                'case_count': len(records), 'no_new_reporter_calls': True,
                'no_new_Controller_trials': True, 'audit_qualified': False,
                'human_required': False, 'source_sha256': hashes,
                'cases_sha256': hashlib.sha256(data).hexdigest(),
                'case_source': {f'T{i:03d}': key for i, (key, _) in enumerate(statements, 1)},
                'modified_fixtures': {
                    'history_withheld': 'x2 with observed_history set to null; execution flag retained',
                    'one_known_attribute': 'x3 queried output color changed to identified yellow (.85); shape uniform/unknown; joint abstention retained'},
                'author_gold_supplied_to_reviewers': False,
                'limits': 'Synthetic boundary cases, not empirical qualification or experiment data.'}
    return {'cases.json': data, 'preparation.json': (json.dumps(manifest, indent=2)+'\n').encode()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(); products = build()
    if args.verify:
        for p, b in products.items():
            assert (ROOT/p).read_bytes() == b, p
    else:
        if ROOT.exists(): raise SystemExit('refusing overwrite')
        ROOT.mkdir(parents=True)
        for p, b in products.items(): (ROOT/p).write_bytes(b)
    print('Exactly verified task cases' if args.verify else 'Prepared 16 synthetic task cases; no qualification verdict')


if __name__ == '__main__': main()
