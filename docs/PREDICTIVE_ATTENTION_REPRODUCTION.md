# Reproduce predictive attention reporting

From the repository root, with project dependencies installed:

```bash
.venv/bin/python scripts/verify_predictive_study.py
.venv/bin/python -m unittest discover -s tests
```

The first command is offline. It verifies every registered source and checkpoint
hash, all request/response records, full-state-to-target-view derivations, exact
mechanical re-scoring, and exact replay of all three confirmed attention models plus all six models'
closed-loop trajectories.
The verification manifest also covers retained pilots and closed-loop traces.
It uses local archived API responses; no credentials or new requests are needed.
The second command passed 168 tests in 64.909 seconds in the recorded run.

Artifacts live under `audits/predictive_attention/`:

- `pilot_v1/`: three development checkpoints and prediction/control metrics.
- `closed_loop_v1/`: all four conditions' full rollout tensors for three models.
- `closed_loop_confirmation/`: fresh full rollouts on the exact reported models.
- `confirmation_v1/`: three confirmed checkpoints and frozen-model metrics.
- `language_pilot_v1/`, `language_pilot_v2/`: exact prompts, errors, full responses,
  parsed commitments, usage, mechanical assessments, blank review packets.
- `report_confirmation_v1/`, `report_confirmation_v2/`: independent contexts,
  complete source tensors, same complete request records, and review packets.
- `verification.json`: hashes and attempt/completion counts.

Do not overwrite failed versions. The inference runners skip already attempted
requests, including failures. Running them against the intact archive makes no
new calls. Removing records and rerunning would spend API credits and would not
be an exact reproduction, since hosted model outputs need not be deterministic.

Fresh training is implemented by `predictive_attention_pilot.py` and
`confirm_predictive_attention.py`. They refuse to overwrite existing results.
Run new training in an isolated checkout or fresh named pilot destination; preserve
the registered archive. Confirmation training uses 1200 updates and fixed seeds,
not a selected checkpoint. The retained internal model's labels are simulator
predictions, not claims of awareness.

For independent review, open [the form](report-review.html). It saves only in
browser storage and exports a JSON file on request. It contains no condition
labels or source record. Avoid opening the publicly archived unblinding key until
the initial rating pass is complete. A second pass with source records is needed
for prose-fidelity assessment. Neither blank forms nor mechanical scores count
as human ratings.
