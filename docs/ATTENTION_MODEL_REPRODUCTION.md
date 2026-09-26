# Reproduce the attention-model reporting evaluation

Run from the repository root, using the existing virtual environment or the
Python 3.12 environment described in [the prior installation guide](FINAL_REPRODUCTION.md).
The tested dependencies remain PyTorch 2.5.1+cpu and NumPy 2.4.3. No model API,
API key, network service, or new controller training is required.

## Verify and replay all results

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/summarize_attention_model.py --verify --replay
```

This validates the archive and source hashes, safely restores missing checkpoints
under `outputs/attention_model/`, regenerates all test scenes, and checks every
metric and compressed case file against the committed results. A mismatched
existing checkpoint causes an error rather than being overwritten. Three seed
files contain three controllers and 15 report heads, including normalization
buffers and validation selections.

The archive is `artifacts/attention_model_checkpoints.tar.gz`. The authoritative
manifest is `audits/attention_model/summary.json`. The original local training
outputs are not needed for replay or reporter refitting.

## Refit the reporters on the archived controllers

Use a separate checkout when comparing with committed artifacts: these commands
rewrite the seed artifacts and reporter checkpoints. Verify first to restore the
archive. Each seed fits all five registered reporters with the fixed budget.

```bash
for seed in 107 207 307; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/attention_model_study.py --seed "$seed" --refit
done
.venv/bin/python scripts/summarize_attention_model.py --package --replay
.venv/bin/python scripts/analyze_attention_model_errors.py
```

An isolated temporary checkout reproduced all 15 reporter fits and matched the
committed checkpoints, artifacts, metrics, and case records exactly.

The controller itself remains frozen. This reproduces the new reporting experiment,
not the historical controller training. Exact original configs and controller
weights are archived; their original checkpoint hashes and seeds are recorded.

## Inspect individual cases

```bash
.venv/bin/python - <<'PY'
import gzip, json
from pathlib import Path
cases = json.loads(gzip.decompress(Path('audits/attention_model/seed107_cases.json.gz').read_bytes()))
print(cases.keys())
print(cases['ordinary']['truth']['m'][0])
print(cases['ordinary']['predictions']['state']['belief'][0])
PY
```

Case files contain ordinary states, physical traces, every reporter's predictions,
and donor/flip/erase/information-loss/outside-input cases. Context IDs identify
scene/schedule combinations. Seed JSON files include confidence intervals, every
validation selection, timestep and cue-regime breakdowns, deterministic examples,
and one complete episode trace.

The descriptive error audit is reproduced by `scripts/analyze_attention_model_errors.py`.
It adds no new gates or model selections. The numerical-correction folder retains
the initial evaluations, before the exact-95% float32 boundary correction.

## Tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

The full suite passed 160 tests, including 11 attention-model tests. New checks
cover unchanged legacy computation, intervention isolation, snapshot timing,
partition/switch separation, report-input isolation, sparse-map scoring, scene
bootstrap clustering, exact threshold arithmetic, and safe archive restoration.

Protocol and state-specification hashes are part of the recorded evidence. If
sources change, verification deliberately fails instead of silently accepting a
new experiment as the committed reference.
