"""Learned-model integration development; archive before any prose pilot."""
import hashlib
import json
from pathlib import Path
import torch
from attcon.bound_content import VisualEncoder, scenes
from attcon.predictive_attention import PredictiveAttention
from attcon.functional_model import (paired_process, alternatives, execute_alternatives,
                                     buffer_contents, neutral_record)

ROOT = Path('audits/functional_model_v1')
COUNT = 128


def main():
    if ROOT.exists():
        raise SystemExit('refusing to overwrite development archive')
    torch.set_num_threads(2)
    results, samples, states, checkpoints = [], [], {}, {}
    with torch.no_grad():
        for visual_seed, attention_seed in ((1301, 901), (1311, 911), (1321, 921)):
            visual_path = Path(f'audits/bound_content/confirmation_v2/seed{visual_seed}.pt')
            attention_path = Path(f'audits/predictive_attention/confirmation_v1/seed{attention_seed}.pt')
            encoder, model = VisualEncoder(), PredictiveAttention()
            encoder.load_state_dict(torch.load(visual_path, weights_only=True)['state_dict'])
            model.load_state_dict(torch.load(attention_path, weights_only=True)['state_dict'])
            encoder.eval(); model.eval()
            for p in (visual_path, attention_path):
                checkpoints[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            patches, colors, shapes = scenes(710000000 + visual_seed, COUNT)
            # One shared scene for both channels, without channel-specific object clues.
            patches = patches[:, :1].expand(-1, 2, -1, -1, -1, -1).clone()
            visual = encoder(patches)
            process = paired_process(720000000 + attention_seed, COUNT)
            states[visual_seed] = {'patches': patches, 'colors': colors[:, 0], 'shapes': shapes[:, 0], 'visual': visual}
            for owner, p in enumerate(process):
                current, hidden, predicted = alternatives(model, p.observations)
                physical = execute_alternatives(p)
                errors = {'allocation_accuracy': float((current.allocation.argmax(-1) == p.allocation[:, -1].argmax(-1)).double().mean()),
                    'effect_accuracy': float((current.effects.argmax(-1) == p.next_effects.argmax(-1)).double().mean()),
                    'controlled_channel_accuracy': float((current.controllability().argmax(-1) == owner).double().mean()),
                    'current_recovery_mae': float((current.access[:, 0]-p.recovery[:, -1]).abs().double().mean()),
                    'counterfactual_recovery_mae': float((predicted-physical).abs().double().mean())}
                gates = {k: v >= .99 if k.endswith('accuracy') else v <= .04 for k, v in errors.items()}
                changed_effects = current.effects.flip(-2)
                _, _, changed = alternatives(model, p.observations, changed_effects)
                _, _, restored = alternatives(model, p.observations, current.effects)
                assert torch.equal(restored, predicted)
                flipped_physical = execute_alternatives(p, effects=process[1-owner].next_effects)
                a = process[1-owner].next_effects[:, 0]
                x = torch.cat((a.flatten(1), (.8*a).flatten(1), torch.nn.functional.one_hot(torch.zeros(COUNT, dtype=torch.long), 4).float()), -1)
                after, _ = model(x[:, None], hidden.clone())
                results.append({'visual_seed': visual_seed, 'attention_seed': attention_seed,
                    'physical_owner': owner, 'metrics': errors, 'gates': gates, 'all_gates_pass': all(gates.values()),
                    'model_only_physical_preservation': True, 'restoration_exact': True,
                    'model_only_access_change_mae': float((changed-predicted).abs().double().mean()),
                    'world_only_predicted_state_preservation': True,
                    'world_only_recovery_error_before_observation': float((predicted-flipped_physical).abs().double().mean()),
                    'world_only_recovery_error_after_one_observation': float((after.access[:, 0, 0]-flipped_physical[:, 0]).abs().double().mean())})
                states[visual_seed][owner] = {'observations': p.observations, 'allocation': p.allocation,
                    'physical_recovery': p.recovery, 'physical_effects': p.next_effects,
                    'modeled_allocation': current.allocation, 'modeled_access': current.access,
                    'modeled_effects': current.effects, 'predicted_recovery': predicted,
                    'executed_recovery': physical, 'model_only_recovery': changed}
                for row in range(4):
                    order = ((0, 1, 2), (2, 0, 1), (1, 2, 0))[row % 3]
                    samples.append({'visual_seed': visual_seed, 'row': row, 'physical_owner': owner,
                        'payload': neutral_record(visual, current, predicted, p, row, order)})
            # Both paired histories share commands and initial acquired-content state.
            assert torch.equal(process[0].observations[..., -4:], process[1].observations[..., -4:])
            assert torch.equal(buffer_contents(visual, torch.zeros_like(process[0].recovery[:, 0])),
                               buffer_contents(visual, torch.zeros_like(process[1].recovery[:, 0])))
    ROOT.mkdir(parents=True)
    torch.save(states, ROOT/'states.pt')
    paths = ['scripts/check_functional_model.py', 'src/attcon/functional_model.py',
             'src/attcon/predictive_attention.py', 'src/attcon/bound_content.py', 'docs/FUNCTIONAL_MODEL_PROTOCOL.md']
    summary = {'stage': 'engineering_development', 'count_per_pair': COUNT, 'results': results,
               'all_gates_pass': all(r['all_gates_pass'] for r in results),
               'checkpoint_sha256': checkpoints, 'torch_version': torch.__version__,
               'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
               'states_sha256': hashlib.sha256((ROOT/'states.pt').read_bytes()).hexdigest()}
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (ROOT/'samples.json').write_text(json.dumps(samples, indent=2)+'\n')
    print(json.dumps({'all_gates_pass': summary['all_gates_pass'], 'results': results}, indent=2))


if __name__ == '__main__':
    main()
