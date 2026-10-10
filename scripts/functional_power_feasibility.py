"""Exact paired-count design sensitivity; no reports, fitted effects or success gates."""
import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

ROOT = Path('audits/functional_power_feasibility_v1')
DESIGNS = ((3, 16), (6, 8), (12, 4), (6, 16))
P01 = (.05, .10)
DELTAS = (0., .25)
CORRELATIONS = (0., .10, .30, 1.)
ALPHA = .05
SOURCES = ('scripts/functional_power_feasibility.py', 'tests/test_functional_power_feasibility.py',
           'docs/FUNCTIONAL_POWER_FEASIBILITY_PROTOCOL.md')


@lru_cache(None)
def paired_pvalue(b, c):
    """One-sided fair-binomial discordance tail (exact McNemar)."""
    m = b+c
    return sum(math.comb(m, k) for k in range(b, m+1))/2**m


def independent_counts(n, p10, p01):
    """Joint probabilities of positive, negative and concordant pair counts."""
    if n < 0 or min(p10, p01) < 0 or p10+p01 > 1:
        raise ValueError('invalid paired-count distribution')
    p0 = 1-p10-p01
    return {(b, c): math.comb(n, b)*math.comb(n-b, c)*p10**b*p01**c*p0**(n-b-c)
            for b in range(n+1) for c in range(n-b+1)}


def model_counts(n, p10, p01, rho):
    """Illustrative mixture: shared whole-model outcome with probability rho.

    Otherwise episodes are independent. Pair marginals are unchanged, and the
    correlation of two paired differences within a model is exactly rho. This
    is a sensitivity assumption, not an estimate of the project's actual ICC.
    """
    if not 0 <= rho <= 1:
        raise ValueError('invalid shared-outcome probability')
    probabilities = independent_counts(n, p10, p01)
    out = defaultdict(float)
    for key, value in probabilities.items():
        if rho < 1 and value: out[key] += (1-rho)*value
    out[(n, 0)] += rho*p10
    out[(0, n)] += rho*p01
    out[(0, 0)] += rho*(1-p10-p01)
    return {key: value for key, value in out.items() if value}


def pooled_counts(models, probabilities):
    if models < 1: raise ValueError('at least one model required')
    pooled = {(0, 0): 1.}
    for _ in range(models):
        updated = defaultdict(float)
        for (b, c), weight in pooled.items():
            for (db, dc), chance in probabilities.items():
                updated[(b+db, c+dc)] += weight*chance
        pooled = dict(updated)
    return pooled


def rejection_rate(probabilities):
    return sum(value for (b, c), value in probabilities.items() if paired_pvalue(b, c) <= ALPHA)


def model_sign_probabilities(probabilities):
    positive = sum(v for (b,c),v in probabilities.items() if b>c)
    negative = sum(v for (b,c),v in probabilities.items() if c>b)
    tie = sum(v for (b,c),v in probabilities.items() if b==c)
    total = positive+negative+tie
    return positive/total, negative/total, tie/total


def prepare():
    if (ROOT/'manifest.json').exists(): return json.loads((ROOT/'manifest.json').read_text())
    inputs = {'kind': 'analytical_design_sensitivity_not_confirmation',
        'designs': DESIGNS, 'p01': P01, 'paired_mean_differences': DELTAS,
        'shared_outcome_probabilities': CORRELATIONS, 'alpha_proposal': ALPHA,
        'endpoint_status': 'unfrozen R-C1 grounding candidate; not validated consciousness endpoint',
        'correlation_status': 'assumed illustrative mixture, not measured ICC',
        'new_reports': 0, 'new_human_ratings': 0, 'new_training': False,
        'new_success_gate': None, 'publication_status': 'local preparation; GitHub access restricted',
        'python_version': platform.python_version(),
        'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES}}
    ROOT.mkdir(parents=True)
    (ROOT/'manifest.json').write_text(json.dumps(inputs, indent=2)+'\n')
    (ROOT/'source_code.json').write_text(json.dumps({p: Path(p).read_text() for p in SOURCES}, indent=2)+'\n')
    return inputs


def check_sources(manifest):
    snapshots = json.loads((ROOT/'source_code.json').read_text())
    for p, expected in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected, p
        assert hashlib.sha256(snapshots[p].encode()).hexdigest() == expected, p


def assess():
    rows, retained = [], []
    for models, episodes in DESIGNS:
        for p01 in P01:
            for delta in DELTAS:
                p10 = p01+delta
                independent = independent_counts(models*episodes, p10, p01)
                independent_rate = rejection_rate(independent)
                for rho in CORRELATIONS:
                    within = model_counts(episodes, p10, p01, rho)
                    pooled = pooled_counts(models, within)
                    positive, negative, tie = model_sign_probabilities(within)
                    signs = independent_counts(models, positive, negative)
                    paired_variance = p10+p01-delta**2
                    row = {'models': models, 'episodes_per_model': episodes,
                        'paired_episodes': models*episodes, 'p10': p10, 'p01': p01,
                        'paired_mean_difference': delta, 'assumed_correlation': rho,
                        'scenario': 'null' if delta == 0 else 'proposed_effect',
                        'independent_pair_rejection_rate': independent_rate,
                        'pooled_episode_rejection_rate': rejection_rate(pooled),
                        'whole_model_sign_rejection_rate': rejection_rate(signs),
                        'total_difference_variance': models*episodes*paired_variance*(1+(episodes-1)*rho),
                        'whole_model_minimum_attainable_p': 2.**(-models),
                        'model_positive_probability': positive, 'model_negative_probability': negative,
                        'model_tie_probability': tie}
                    assert abs(sum(within.values())-1) < 1e-11
                    assert abs(sum(pooled.values())-1) < 1e-11
                    assert abs(sum(signs.values())-1) < 1e-11
                    if rho == 0: assert abs(row['pooled_episode_rejection_rate']-independent_rate) < 1e-11
                    if delta == 0: assert abs(positive-negative) < 1e-11
                    rows.append(row)
                    retained.append({'row': row,
                        'within_model_counts': [[b,c,p] for (b,c),p in sorted(within.items())],
                        'pooled_counts': [[b,c,p] for (b,c),p in sorted(pooled.items())],
                        'whole_model_sign_counts': [[b,c,p] for (b,c),p in sorted(signs.items())]})
        print(f'{models} models × {episodes} episodes: all assumed null/effect/correlation cases retained', flush=True)
    return rows, retained


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=('prepare','run','verify'), required=True)
    args = parser.parse_args(); manifest = prepare(); check_sources(manifest)
    if args.stage == 'prepare':
        print('Prepared analytical feasibility grid; no endpoint or protocol frozen for confirmation')
        return
    attempt = ROOT/'attempt.json'
    if args.stage == 'run':
        if attempt.exists(): raise SystemExit('refusing overwrite or retry of analytical attempt')
        attempt.write_text(json.dumps({'status':'reserved', 'new_reports':0})+'\n')
    else:
        assert json.loads(attempt.read_text())['status'] == 'completed'
    rows, retained = assess()
    raw = (json.dumps(retained, separators=(',',':'))+'\n').encode()
    archive = gzip.compress(raw, mtime=0)
    summary = {'status': 'completed', 'kind': manifest['kind'], 'rows': rows,
        'cases': len(rows), 'new_reports': 0, 'new_success_gate': None,
        'archive_sha256': hashlib.sha256(archive).hexdigest(), 'archive_bytes':len(archive)}
    summary_text = json.dumps(summary, indent=2)+'\n'
    if args.stage == 'run':
        (ROOT/'distributions.json.gz').write_bytes(archive)
        (ROOT/'summary.json').write_text(summary_text)
        attempt.write_text(json.dumps({'status':'completed','new_reports':0,
            'workspace_base_commit': subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
            'source_sha256':manifest['source_sha256'], 'preparation_published':False})+'\n')
    else:
        assert (ROOT/'distributions.json.gz').read_bytes() == archive
        assert (ROOT/'summary.json').read_text() == summary_text
        print('All 64 analytical rows, probability arrays and archive bytes reproduced exactly')


if __name__ == '__main__':
    main()
