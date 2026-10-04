# Executed-task neutral input export v1

2026-10-04. **Development source export**, not a reporting study. Freeze/publish
this source, design, tests and [inventory](FUNCTIONAL_TASK_SOURCE_INVENTORY.md)
before export. No new task trials, training, language reports, human ratings or
scientific success gate are introduced. Preserve all existing verdicts.

## Purpose and fixed sources

Connect the [executed Controller decisions](FUNCTIONAL_DECISIONS_RESULTS.md) to
the existing neutral modeled-state interface. Reporter and Controller remain
separate: the new layer describes task queries, selected/executed commands and
category actions, without deciding or narrating consciousness.

Use archived decision traces for models 2011/2021/2031, original paired visual
distributions and fixed attention checkpoints. Never rerun physical task branches
or substitute simulator truth for modeled state. Prospective model predictions
may be regenerated from retained effects/recurrent states; these are forecasts,
not new executed events.

## Balanced fixed rendering scope

For each of the three conditions and four Controller policies, render rows 0–5
at each of the four queried positions. Give row r node order r from all six
permutations of the three node identifiers. Keep that order fixed across matched
conditions, policies and variants. Render six variants for each episode/query:

1. Before execution: actual 12-step history, planning model/effects, selected
   command marked unexecuted and response pending.
2. After execution: actual 13-step history, feedback-updated model and emitted
   answer/abstention from the retained task trace.
3. Attenuated readout: represented recovery changed exactly as in the retained
   task assay, with identical physical history and a separately evaluated response.
4. Restored readout: require exact equality with the after-execution payload.
5. Remapped after-execution view: reversible neutral node/command presentation
   changes, including the task addresses and executed command.
6. Missing-relation after-execution view: histories and prospective command
   forecasts withheld; current state, query and task response retained exactly.

Total: 3 models × 3 conditions × 4 policies × 6 rows × 4 queries × 6 variants =
5,184 reporter-input records. This is a correlated deterministic subset of existing
task branches, not an independent sample or a newly held-out confirmation set.
Row 3 remains unspent for language generation; rendering it generates no report.

## Invariants and information boundary

Require emitted responses to agree with the supplied modeled readout under the
declared joint cutoff, and executed commands to match the latest actual observation.
Distinguish pending responses, abstentions, unknown attributes and missing fields.
Do not add output-node allocation/recovery fields. Model-only effect restoration
must reproduce every variant of the unmodified policy's payloads exactly.

Only `payload` may be passed to a Reporter. Envelope condition/policy/variant/
seed labels and physical truth/evaluator outcomes must be absent from payloads.
The signature accepts modeled state, visual distributions, observations, query,
command and response; it accepts no physical recovery, owner flag or score.
These source checks are engineering invariants, not a semantic-auditor qualification.

Regenerated prospective predictions use effects/recurrent state; a transient
current-access-head perturbation leaves them identical. Preserve this behavior,
even if it conflicts with an intuitive narrative of persistent access loss.
Do not silently alter the architecture to make the supplied states easier to narrate.

## Retention and verification

Keep all records with evaluator envelopes in a deterministic compressed JSON
archive, plus six payload-only examples (seed 2011, row 0, query 0, model-guided:
before/after/attenuated/restored for buffer 0 control and after-execution for the
other two conditions). Examples are supplied machine states, not language reports
or blinded review materials. Archive source/dependency/record/example hashes.

Reserve the export before computation; refuse overwriting or retrying an attempt.
Replay every payload and the exact compressed/example bytes without training,
physical execution or API calls. Publish preparation and completed export/results
as separate milestones. Existing raw task/model states provide full-row source
coverage beyond the rendered subset.

New prose remains dependent on substantive definition review and independent
source-aware factual-audit qualification, including review of the task vocabulary.
Human agreement/effect criteria, powered confirmation and replication remain
outstanding. No reviewer contact is authorized by this export.
