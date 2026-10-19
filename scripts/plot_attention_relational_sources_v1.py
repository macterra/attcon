"""Illustrate the first scheduled source-control case from preserved JSON only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXPORT_SHA256 = '4f5f0bc2fa78b3b24b3580f52e306f20df0f3011d079b8139a2bc1f029050837'
PREFIX = 's1301_w00_t12_A_joined_'
VARIANTS = ('baseline', 'current_donor', 'current_restored', 'W_changed', 'W_restored')
TITLES = ('Baseline', 'Current A donor', 'A restored', 'W changed (synthetic)', 'W restored')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = args.exports.read_bytes()
    if digest(raw) != EXPORT_SHA256:
        raise ValueError('only the preserved first export is permitted')
    exports = json.loads(raw)
    packets = {r['id']: r for r in exports['packets']}
    chosen = [packets[PREFIX + variant] for variant in VARIANTS]
    for packet in chosen:
        content = packet['content']
        if digest(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()) != packet['content_sha256']:
            raise ValueError('packet content hash changed')
        if [p['panel'] for p in content['panels']] != ['p0', 'p1']:
            raise ValueError('fixed joined-panel inventory changed')
    current = [[p['current'] for p in packet['content']['panels']] for packet in chosen]
    if current[0] != current[2] or current[0] != current[4]:
        raise ValueError('restored current records differ from baseline')
    if current[0][1] != current[1][1] or current[0][0] != current[3][0]:
        raise ValueError('fixed-source control invariant failed')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    filenames = ('source_controls.png', 'source_controls.pdf', 'source_controls.json')
    if any((output / name).exists() for name in filenames):
        raise FileExistsError('illustration outputs are exclusive')
    fig, axes = plt.subplots(4, 5, figsize=(16, 11), layout='constrained')
    image = None
    for row in range(4):
        panel, role = row // 2, ('red', 'blue')[row % 2]
        for column in range(5):
            axis = axes[row, column]
            mosaic = np.full((9, 9), np.nan)
            for node_index, node in enumerate(current[column][panel]):
                if node['node'] != f'n{node_index}':
                    raise ValueError('node order differs')
                values = np.asarray(node['distributions'][role], dtype=float)
                if values.shape != (4, 4) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                    raise ValueError('invalid supplied localization distribution')
                y, x = (node_index // 2) * 5, (node_index % 2) * 5
                mosaic[y:y + 4, x:x + 4] = values
                axis.text(x - .35, y - .6, node['node'], fontsize=8, color='black')
            image = axis.imshow(mosaic, vmin=0, vmax=1, cmap='viridis', origin='upper')
            axis.set_facecolor('#e5e5e5')
            axis.set_xlim(-.7, 8.5)
            axis.set_ylim(8.5, -1.1)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(TITLES[column], fontsize=11)
            if column == 0:
                name = 'A-view current record' if panel == 0 else 'W comparison record'
                axis.set_ylabel(name + '\n' + role + ' localization', fontsize=10)
    fig.suptitle('First preselected reporting case: seed 1301, world 0, decision 12\n'
                 'All four addressed local grids; source and W interventions with exact restoration', fontsize=13)
    fig.supxlabel('Colors show supplied localization mass on the same 0–1 scale. Each labeled tile is a separate 4×4 grid.\n'
                  'The A-view combines copied W11 records with the A estimate at n3. W changed uses the saved synthetic time-13 branch.', fontsize=10)
    fig.colorbar(image, ax=axes, shrink=.65, pad=.015, label='Supplied localization mass')
    for name, format_name in zip(filenames[:2], ('png', 'pdf')):
        with (output / name).open('xb') as stream:
            fig.savefig(stream, format=format_name, dpi=180)
    plt.close(fig)
    record = {'schema': 'attention_relational_report_v1_first_case_source_control_illustration',
              'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'input_path': str(args.exports), 'input_sha256': EXPORT_SHA256,
              'source_sha256': digest(Path(__file__).read_bytes()),
              'selection': {'case': PREFIX, 'basis': 'first scheduled case and fixed control variants',
                            'based_on_report_or_reference_outcomes': False, 'all_nodes_and_roles_shown': True},
              'packets': chosen, 'normalization_or_values_changed': False,
              'new_model_or_api_calls': 0, 'report_fidelity_or_consciousness_result_asserted': False,
              'figure_sha256': {name: digest((output / name).read_bytes()) for name in filenames[:2]}}
    with (output / filenames[2]).open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    if args.exports.read_bytes() != raw:
        raise ValueError('source export changed while plotting')
    print(json.dumps({'output_dir': str(output), 'figure_sha256': record['figure_sha256']}, sort_keys=True))


if __name__ == '__main__':
    main()
