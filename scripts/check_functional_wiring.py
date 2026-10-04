"""Archive developmental identifiability evidence, not consciousness evidence."""
import hashlib
import json
from pathlib import Path
from attcon.functional_wiring import CONDITIONS, ORDERS, fixture, identify

ROOT = Path('audits/functional_wiring_v1')


def main():
    if ROOT.exists():
        raise SystemExit('refusing to overwrite developmental archive')
    rows, counts, initial_matches = [], {c: 0 for c in CONDITIONS}, 0
    for seed in range(610000, 610128):
        for order in ORDERS:
            initial = [fixture(seed, c, order, initial_only=True) for c in CONDITIONS]
            assert initial[0] == initial[1] == initial[2]
            initial_matches += 1
            for condition in CONDITIONS:
                payload = fixture(seed, condition, order)
                predicted = identify(payload)
                counts[condition] += predicted == condition
                rows.append({'seed': seed, 'order': order, 'condition': condition,
                             'prediction': predicted, 'payload': payload})
    ROOT.mkdir(parents=True)
    data = json.dumps(rows, separators=(',', ':')) + '\n'
    (ROOT / 'cases.json').write_text(data)
    paths = ['src/attcon/functional_wiring.py', 'scripts/check_functional_wiring.py',
             'docs/FUNCTIONAL_WIRING_DESIGN.md']
    summary = {'kind': 'engineering_development_only', 'paired_seeds': 128,
               'presentations_per_condition': 768, 'correct': counts,
               'identical_initial_triplets': initial_matches,
               'cases_sha256': hashlib.sha256(data.encode()).hexdigest(),
               'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
