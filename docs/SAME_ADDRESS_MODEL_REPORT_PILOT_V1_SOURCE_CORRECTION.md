# Same-address pilot: command-symbol correction

2026-10-20. This operational addendum corrects one factual assumption in the
[frozen plan](SAME_ADDRESS_MODEL_REPORT_PILOT_V1_PLAN.md); its original bytes,
registration and first failed execution remain preserved. No report or new
predictor/native-H inference ran before that failure.

The first source run from commit `56f2c8679f504e4d4f18abaf214b44b3b12bc552`
stopped with `registered previous command must be c2`. The
[first failure receipt](../audits/attention_relational_report_v1/same_address_pilot_v1/source_run/first_receipt.json)
and [as-executed source](../audits/attention_relational_report_v1/same_address_pilot_v1/preexecution/first_executed_source.py)
are retained. A subsequent single metadata-only archive load established the
[actual command history](../audits/attention_relational_report_v1/same_address_pilot_v1/source_failure_metadata_v1/first_metadata.json).

At command index11, worlds0–3 have address3 and opaque command symbols
**[c0, c1, c1, c0]**, respectively. The schedule's third round indexes the third
entry of a per-world symbol permutation; it does not imply symbol value2. The
plan's references to last command11 being c2 were incorrect.

The corrected source keeps the same encoder1311, worlds0–3, decision12, n3,
red/blue/red/blue queries, checkpoints, native H, donor interventions, fixed k0
W counterfactual, original96-row arithmetic context,72 opportunities and84 pairs.
It verifies the actual four archived symbols and gathers each row's previous
symbol when checking that current model content equals the corresponding
same-address conditional content. Each representation retains its own time and
origin. No source identity or case is substituted and no effect is selected.

The k0 W counterfactual remains fixed even though it repeats the last operation
in worlds0 and3. Those controls may be exact nulls and must remain in the results.
The control is not changed to obtain a larger difference. Likewise, donor nulls
remain legitimate outcomes.

The corrected source receives a separate exact registration, independent review
and published release before execution. Its exclusive output directory is
`source_run_operational_correction_v1`; the original `source_run` failure is never
overwritten. This is a correction before the first source inference, with the
failed operational attempt retained, rather than a replacement scientific sample
or a new reporting prompt. Assessment instructions, first-response rules and the
published stop-or-advance decision remain unchanged. Successful reconstruction
would establish source readiness only; the qualia-source objective remains open.
