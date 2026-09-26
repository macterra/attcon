# Earlier informational-state study reproduction

For the current attention-model study, use [ATTENTION_MODEL_REPRODUCTION.md](ATTENTION_MODEL_REPRODUCTION.md).

This earlier informational-state study is a local CPU evaluation. The tested environment is Python 3.12.3,
PyTorch 2.5.1+cpu, and NumPy 2.4.3. Primary dependency versions are pinned in
[requirements-final.txt](../requirements-final.txt). The repository's existing
virtual environment can be used directly, or create an environment with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-final.txt
.venv/bin/python -m pip install -e .
```

Run commands from the repository root. No API key, paid model call, or external
service is required for the experiments.

## Inspect and smoke-test

```bash
.venv/bin/python scripts/reproduce_final.py --plan
.venv/bin/python scripts/reproduce_final.py --smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

The smoke test runs tiny disjoint partitions through both tasks and architectures,
report fitting, causal interventions, and reward-only exploration. Its output is
`outputs/final_smoke.json`; it does not overwrite scientific audits or count as
evidence. The final full suite passed 149 tests.

## Verify the committed results

```bash
.venv/bin/python scripts/reproduce_final.py --verify
```

This verifies [the checkpoint archive](../artifacts/final_checkpoints.tar.gz),
restores its 42 allowlisted checkpoint files under `outputs/prospective/`, and
recomputes every controller, report, causal, and stress metric. It checks the
[completion manifest](../audits/project_completion.json), registered seed/cost
coverage, validation selection, source and dataset hashes, matching capacities,
and gate consistency. Archive files must match the manifest exactly; links,
extra members, unsafe paths, and checksum mismatches are rejected before restore.

For a checksum/provenance check without recomputing metrics:

```bash
.venv/bin/python scripts/final_project_audit.py --restore --verify
```

The archive contains selected trained controllers and readouts. Shuffled-null
scores and selections are in the JSON audits; full retraining regenerates those
fits. Failed scientific gates are expected results, not verification failures.

## Retrain the fixed matrix

```bash
.venv/bin/python scripts/reproduce_final.py --run --jobs 3
```

This runs all registered cells, the explicitly documented comparator correction,
reserved stress, summaries, archive packaging, and final metric replay. It
regenerates the versioned audit paths and checkpoint archive; use a separate
checkout when comparing a reproduction with the committed reference. Logs are
written under `outputs/reproduction_logs/`. `--jobs 1` lowers concurrent memory
use. Thresholds, seeds, and selection rules remain those in the protocol.

The [original protocol](COMPLETION_PROTOCOL.md) and
[correction record](COMPLETION_CORRECTIONS.md) state the claim boundaries.
The [final results](PROJECT_RESULTS.md) report bounded positive evidence for
accurate external state readouts and distinguish it from native reporting,
comparative advantage, and the separate Stage 8 hypotheses.
