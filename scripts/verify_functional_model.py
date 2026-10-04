"""Recreate every archived physical and represented functional-model state."""
import hashlib
import json
from pathlib import Path
import torch
from attcon.bound_content import VisualEncoder, scenes
from attcon.predictive_attention import PredictiveAttention
from attcon.functional_model import paired_process, alternatives, execute_alternatives, neutral_record

ROOT = Path('audits/functional_model_v1')


def main():
    torch.set_num_threads(2)
    summary = json.loads((ROOT/'summary.json').read_text())
    for p, digest in {**summary['source_sha256'], **summary['checkpoint_sha256']}.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest, p
    assert hashlib.sha256((ROOT/'states.pt').read_bytes()).hexdigest() == summary['states_sha256']
    stored = torch.load(ROOT/'states.pt', weights_only=True)
    samples = json.loads((ROOT/'samples.json').read_text())
    with torch.no_grad():
        for visual_seed, attention_seed in ((1301, 901), (1311, 911), (1321, 921)):
            encoder, model = VisualEncoder(), PredictiveAttention()
            encoder.load_state_dict(torch.load(f'audits/bound_content/confirmation_v2/seed{visual_seed}.pt', weights_only=True)['state_dict'])
            model.load_state_dict(torch.load(f'audits/predictive_attention/confirmation_v1/seed{attention_seed}.pt', weights_only=True)['state_dict'])
            encoder.eval(); model.eval()
            patches, colors, shapes = scenes(710000000+visual_seed, summary['count_per_pair'])
            patches = patches[:, :1].expand(-1, 2, -1, -1, -1, -1).clone()
            visual = encoder(patches)
            for name, value in {'patches': patches, 'colors': colors[:, 0], 'shapes': shapes[:, 0], 'visual': visual}.items():
                assert torch.equal(stored[visual_seed][name], value), (visual_seed, name)
            for owner, p in enumerate(paired_process(720000000+attention_seed, summary['count_per_pair'])):
                current, _, predicted = alternatives(model, p.observations)
                _, _, changed = alternatives(model, p.observations, current.effects.flip(-2))
                expected = {'observations': p.observations, 'allocation': p.allocation,
                    'physical_recovery': p.recovery, 'physical_effects': p.next_effects,
                    'modeled_allocation': current.allocation, 'modeled_access': current.access,
                    'modeled_effects': current.effects, 'predicted_recovery': predicted,
                    'executed_recovery': execute_alternatives(p), 'model_only_recovery': changed}
                for name, value in expected.items():
                    assert torch.equal(stored[visual_seed][owner][name], value), (visual_seed, owner, name)
                for sample in samples:
                    if sample['visual_seed'] != visual_seed or sample['physical_owner'] != owner:
                        continue
                    row = sample['row']; order = ((0, 1, 2), (2, 0, 1), (1, 2, 0))[row % 3]
                    assert sample['payload'] == neutral_record(visual, current, predicted, p, row, order)
    print('All physical/represented tensors and 24 neutral samples exactly replayed')


if __name__ == '__main__':
    main()
