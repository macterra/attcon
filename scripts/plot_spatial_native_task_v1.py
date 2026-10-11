"""Plot fixed first-world native readouts; no new worlds, inference or fitting."""
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'audits/spatial_native_task_v1/run'
OUTPUT = ROOT / 'docs/assets/spatial_native_task_v1_fields.png'
MANIFEST = OUTPUT.with_suffix('.json')


def main():
    assert not OUTPUT.exists() and not MANIFEST.exists(), 'refusing overwrite'
    archive = RUN / 'states-1301.pt.gz'
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    completion = json.loads((RUN / 'completion.json').read_text())
    assert completion['status'] == 'completed'
    assert completion['archive_sha256'][str(archive.relative_to(ROOT))] == archive_sha
    with gzip.open(archive, 'rb') as stream:
        trace = torch.load(stream, map_location='cpu', weights_only=False)
    # Fixed illustration: first diagnostic world, first address, both roles.
    pixels = trace['diagnostic_worlds']['pixels'][0, 0].permute(1, 2, 0).numpy()
    cells = trace['diagnostic_worlds']['evaluator_cells'][0, 0]
    fig = plt.figure(figsize=(11, 5), layout='constrained')
    grid = fig.add_gridspec(2, 4, width_ratios=(1.2, 1, 1, 1))
    sensory = fig.add_subplot(grid[:, 0])
    sensory.imshow(pixels)
    sensory.set_title('Delivered patch\n(same saved input)')
    sensory.set_xticks([])
    sensory.set_yticks([])
    for role_index, role in enumerate(('red', 'blue')):
        for column, band in enumerate((1, 2, 4), 1):
            record = trace['native_execution']['primary'][f'T{band}']
            field = record['role_fields'][0, 0, role_index]
            action = record['selected_cells'][0, 0, role_index]
            truth = cells[role_index]
            ax = fig.add_subplot(grid[role_index, column])
            shown = ax.imshow(field.numpy(), cmap='viridis', vmin=0, vmax=1)
            ax.scatter(int(truth[1]), int(truth[0]), marker='o', s=100,
                       facecolors='none', edgecolors='white', linewidths=1.8)
            ax.scatter(int(action[1]), int(action[0]), marker='x', s=65,
                       color='orange', linewidths=1.8)
            ties = int(record['tie_counts'][0, 0, role_index])
            ax.set_title(f'{role}: T{band}\nmax={float(field.max()):.3f}; ties={ties}')
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xlim(-.5, 3.5)
            ax.set_ylim(3.5, -.5)
    fig.colorbar(shown, ax=fig.axes[1:], label='Conditional location probability', shrink=.85)
    fig.suptitle('Native H(W) readouts — alternatives from the same saved V\n'
                 'White circle: evaluator location; orange cross: emitted target. No learned A.', fontsize=11)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160)
    plt.close(fig)
    value = dict(encoder_seed=1301, diagnostic_world_index=0, address='l0',
                 roles=['red', 'blue'], bands=[1, 2, 4], selection='fixed first world/address',
                 archive_sha256=archive_sha, plot_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                 new_worlds=0, new_model_inference=0, new_fits=0,
                 attention_model_A_present=False, presentation_qualified=False)
    with MANIFEST.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
