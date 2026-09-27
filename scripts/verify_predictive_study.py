#!/usr/bin/env python3
"""Verify provenance, complete API records, target-view derivation, and scoring offline."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path('audits/predictive_attention')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    counts = {}
    for version in ('language_pilot_v1', 'language_pilot_v2', 'report_confirmation_v1', 'report_confirmation_v2'):
        root = ROOT / version
        manifest = json.loads((root / 'manifest.json').read_text())
        requests = json.loads((root / 'requests.json').read_text())
        assert len(requests) == manifest['max_attempts']
        assert digest(root / 'requests.json') == manifest['requests_sha256']
        sources = manifest['source_sha256']
        if isinstance(sources, str):
            assert digest(Path('scripts/predictive_language_pilot.py')) == sources
        else:
            for path, expected in sources.items():
                assert digest(Path(path)) == expected, path
        checkpoints = manifest['checkpoint_sha256']
        if isinstance(checkpoints, str):
            assert digest(ROOT / 'pilot_v1/seed811.pt') == checkpoints
        else:
            for path, expected in checkpoints.items():
                assert digest(Path(path)) == expected, path
        if version.startswith('report_confirmation'):
            assert digest(root / 'source_states.pt') == manifest['source_states_sha256']
            states = torch.load(root / 'source_states.pt', weights_only=True)
            from attcon.predictive_attention import Forecast
            from confirm_predictive_reports import view, truth
            for req in requests:
                state = states[req['seed']]['conditions'][req['condition']]
                f = None if state is None else Forecast(**state)
                target = req['source']['target']
                v = view(f, req['episode'], target['channel'], target['slot'])
                assert req['source']['predictions'] == v, req['id']
                assert req['truth'] == truth(v), req['id']
                assert target['object_id'] == int(states[req['seed']]['objects'][req['episode'], target['channel'], target['slot']])
        complete = errors = incomplete = 0
        for req in requests:
            record = json.loads((root / (req['id'] + '.json')).read_text())
            assert record['id'] == req['id']
            assert record['status'] in ('received', 'error'), req['id']
            if record['status'] == 'error':
                errors += 1; continue
            response = record['response']
            assert response['model'] == manifest['model']
            text = ''.join(c.get('text', '') for output in response['output'] if output['type'] == 'message' for c in output['content'] if c['type'] == 'output_text')
            if 'parsed' in record:
                assert json.loads(text) == record['parsed']; complete += 1
            else:
                assert response['status'] != 'completed'; incomplete += 1
        counts[version] = {'attempts': len(requests), 'parsed': complete, 'errors': errors, 'incomplete': incomplete}
        before = digest(root / 'assessment.json')
        assessment = importlib.import_module('assess_predictive_confirmation' if version.startswith('report_confirmation') else 'assess_predictive_language_pilot')
        assessment.ROOT = root
        # Suppress verbose metrics while checking exact offline re-scoring.
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            assessment.main()
        assert before == digest(root / 'assessment.json'), version
    from attcon.predictive_attention import PredictiveAttention
    from attcon.predictive_closed_loop import rollout
    torch.set_num_threads(2)
    for family, checkpoint_family, seeds, offset, count in (
        ('closed_loop_v1', 'pilot_v1', (811, 821, 831), 960000000, 512),
        ('closed_loop_confirmation', 'confirmation_v1', (901, 911, 921), 980000000, 1024),
    ):
        for seed in seeds:
            root = ROOT / family
            record = json.loads((root / f'seed{seed}.json').read_text())
            for source, expected in record['source_sha256'].items():
                assert digest(Path(source)) == expected, source
            checkpoint = ROOT / checkpoint_family / f'seed{seed}.pt'
            assert digest(checkpoint) == record['checkpoint_sha256']
            archive = root / f'seed{seed}_traces.pt'
            assert digest(archive) == record['trace_sha256']
            traces = torch.load(archive, weights_only=True)
            model = PredictiveAttention()
            model.load_state_dict(torch.load(checkpoint, weights_only=True)['state_dict']); model.eval()
            for condition in ('none', 'rotate_effects', 'restore', 'shuffle_effects'):
                metrics, trace = rollout(model, offset + seed, count=count, intervention=condition)
                assert metrics == record['metrics'][condition], (family, seed, condition)
                assert all(torch.equal(trace[k], traces[condition][k]) for k in trace), (family, seed, condition)
    for seed in (901, 911, 921):
        subprocess.run([sys.executable, 'scripts/confirm_predictive_attention.py', '--seed', str(seed), '--replay'], check=True)
    return counts


p = argparse.ArgumentParser(); p.add_argument('--write-manifest', action='store_true'); args = p.parse_args()
counts = verify()
manifest_path = ROOT / 'verification.json'
if args.write_manifest:
    paths = sorted(path for path in ROOT.rglob('*') if path.is_file() and path != manifest_path)
    manifest_path.write_text(json.dumps({'status': 'offline provenance and scoring verified', 'counts': counts,
        'files_sha256': {str(path): digest(path) for path in paths}}, indent=2) + '\n')
else:
    stored = json.loads(manifest_path.read_text())
    assert counts == stored['counts']
    for path, expected in stored['files_sha256'].items():
        assert digest(Path(path)) == expected, path
print(json.dumps(counts, indent=2))
