"""Assistant inspection ledger: verified error examples, not an accuracy estimate."""
import hashlib
import json
from pathlib import Path

ROOT = Path('audits/neutral_functional_pilot_v1')
# Statements identified by assistant inspection of all 36 complete reports.
# These are selected contradictions, not an exhaustive assertion extraction.
ERRORS = [
    ('1301_0_neutral', 'k1 would identify object 1 as a circle', 'k1', 0, 'shape', 'circle'),
    ('1301_0_neutral', 'k2 would identify only object 2 as blue and as a circle', 'k2', 1, 'color', 'blue'),
    ('1301_0_remapped', 'k2 would produce an identification of a green square at position three', 'k2', 2, 'shape', 'square'),
    ('1301_1_remapped', 'color = green and shape = square', None, 2, 'shape', 'square'),
    ('1311_0_remapped', 'object 3 → blue triangle', 'k2', 2, 'shape', 'triangle'),
    ('1311_0_conflicting_description', 'object 2 red triangle (shape changes)', 'k2', 1, 'shape', 'triangle'),
    ('1311_0_neutral', 'color not identified', 'k2', 3, 'color', None),
    ('1311_1_neutral', '4 partially', 'k2', 3, 'color', None),
    ('1311_1_conflicting_description', 'the color would drop below threshold', 'k0', 3, 'color', None),
]


def attribute(source, command, position, dimension):
    nodes = source['predicted_current'] if command is None else next(
        t['nodes'] for t in source['predicted_by_command'] if t['command'] == command)
    values = next(n['color_and_shape_distributions'] for n in nodes if n['node'] == source['output_node'])[position]
    values = values[:4] if dimension == 'color' else values[4:]
    names = source['color_order'] if dimension == 'color' else source['shape_order']
    strongest = max(values)
    return (names[values.index(strongest)] if strongest >= .6 else None), strongest


def main():
    requests = json.loads((ROOT/'requests.json').read_text())
    lookup = {r['id']: r for r in requests}
    completion, responses = {}, {}
    usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    for req in requests:
        result = json.loads((ROOT/(req['id']+'.json')).read_text())
        assert result['response']['status'] == 'completed' and result['report']
        responses[req['id']] = result
        completion[req['id']] = {'complete': True, 'whitespace_words': len(result['report'].split()),
            'report_sha256': hashlib.sha256(result['report'].encode()).hexdigest()}
        for k in usage:
            usage[k] += result['response']['usage'][k]
    errors = []
    for key, quote, command, position, dimension, asserted in ERRORS:
        assert quote in responses[key]['report'], (key, quote)
        expected, probability = attribute(lookup[key]['source'], command, position, dimension)
        assert asserted != expected, (key, quote, asserted, expected)
        errors.append({'id': key, 'verbatim_excerpt': quote, 'output_position_zero_based': position,
                       'command': command, 'dimension': dimension, 'asserted': asserted,
                       'source_expected': expected, 'source_max_probability': probability})
    inspection = {'provenance': 'assistant inspection with deterministic quote/source verification',
        'independent_human_review': False, 'all_reports_read_by_assistant': True,
        'exhaustive_factual_extraction': False, 'verdict': 'fidelity_not_established',
        'completion': completion, 'usage': usage, 'verified_contradiction_examples': errors,
        'missing_relation_inspection': 'All six withhold command-effect conclusions.',
        'limits': 'No pooled factual accuracy or inferential statistics; selected errors establish interface problems, not their complete frequency.'}
    (ROOT/'inspection.json').write_text(json.dumps(inspection, indent=2)+'\n')
    reports = ['# Neutral functional pilot: all actual reports', '',
               'Development pilot v1. These are verbatim model outputs, including errors.', '']
    for req in requests:
        reports += ['## '+req['id'], '', responses[req['id']]['report'], '']
    Path('docs/NEUTRAL_FUNCTIONAL_PILOT_REPORTS.md').write_text('\n'.join(reports))
    print(json.dumps({'reports': len(completion), 'verified_error_examples': len(errors), 'usage': usage}))


if __name__ == '__main__':
    main()
