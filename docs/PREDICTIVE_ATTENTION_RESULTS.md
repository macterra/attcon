# Predictive attention reporting: current results

2026-09-27. **The state-fidelity criteria now pass. The overall qualia-reporting
goal remains active: independent assessment of the prose is pending.**

We trained a model of allocation, access/retention, and command effects, used its
predictions to control attention, and tested reports of its state. The main
advance over the inspection-model study is reliable, selective reporting of this
richer model. First-person wording is separately manipulated, not counted as
consciousness evidence.

## What is identified and controlled

A GRU receives processing allocations, sampled signal strengths, and preceding
commands in a two-channel/four-slot simulator. One channel follows commands; the
other cycles independently. A predicts allocation, reconstruction probabilities
at three delays, and allocation under alternative commands. Its training labels
contain no phenomenological terms. V supplies independent object identifiers.

The policy selects commands using A's command-effect predictions. In closed-loop
runs, those commands determine subsequent allocation and reconstruction outcomes.
Allocation and access heads share the predictive model but are not independently
necessary for this policy. The current task is a simulated reconstruction process,
not natural vision or a language model's native attention mechanism.

## Prediction and causal use

Three fresh confirmation seeds (901/911/921), each evaluated on 1,024 new episodes:

| Measure | Across seeds |
| --- | --- |
| Allocation accuracy | 100% |
| Alternative-command effect accuracy | 99.988–99.998% |
| Controllable-channel identification | 99.984–100% |
| Access probability mean absolute error | 0.0232–0.0240 |
| Worst access probability error | 0.2143–0.2395 |
| Brier loss above simulator-probability oracle | 0.00116–0.00159 |
| Queried controlled-slot selection | 100% |
| Following synthetic command-effect rotation | 100% |
| Exact restoration and unchanged-field preservation | 100% |

Every frozen internal-model criterion passed. Exact checkpoint replay passed.
Separate development closed-loop runs on all three pilot models selected the
queried controlled slot 100% of the time; rotating the model's command table
reduced that to 0%, and restoration reproduced every trajectory tensor exactly.
A separately frozen [closed-loop confirmation](PREDICTIVE_CLOSED_LOOP_CONFIRMATION.md)
on the exact reported models (901/911/921), with 1,024 fresh episodes each,
also passes every selection, rotation, and exact-restoration gate.
The independently cycling channel was unchanged. These establish causal use,
not a requirement that conscious access be necessary for task performance.

## Reporting results and retained failures

The frozen reporter is `gpt-5-mini-2025-08-07`. All prompts, full API responses,
usage, forecasts, checkpoints, and assessments are archived.

| Version | Outcome |
| --- | --- |
| Numerical-array pilot v1 | 79.2% ordinary and 66.7% intervened complete commitments; index confusion, three incomplete outputs, one timeout |
| Style-only pilot v2 | 75% ordinary and 62.5% intervened; style did not repair fidelity |
| Explicit-target confirmation v1 | 96/96 primary reports and 72/72 intervention pairs correct, but constant-state tie handling failed (11/24 correct) |
| Explicit-target confirmation v2, fresh contexts | All 240 reports' structured commitments correct; all frozen mechanical gates pass |

V2 retains the same scoring criteria, adds explicit precedence for unknown/tied
allocation, and uses new simulator contexts. Earlier failures remain failures.

The passing confirmation covers 12 underlying episodes across three independently
trained models, ten conditions, and two prose styles. It includes **96/96** primary
reports, **72/72** selective-intervention pairs, and **24/24** exact restorations.
Conditions change allocation, access, or command effects while preserving object
identity and other model fields. The reporter follows the changed model state,
including synthetic disagreement with the physical process. Constant and absent
state return the stipulated uncertainty rather than invented positive facts.

All controls were reported accurately relative to their own supplied information.
Relative to the *actual controller's A*, physical forecasts and the independently
trained history predictor each match only **2/24** complete reports; shuffled and
constant states match **0/24**. The matched history predictor has the same training
budget and capacity, but is itself another attention-process model, not a
model-free alternative. Its ability to produce similar prose is not excluded.
**Both physical and history-predictor controls match A's focality and responsive
channel on 24/24 reports.** Their low complete-match rate comes from access
probabilities, not those categorical distinctions. This post-hoc breakdown is
archived separately and changes no gate. Differences at the 0.01 numeric
tolerance should not be interpreted as large phenomenological differences or
unique qualitative structure attributable to the controller's A.

## What this does and does not establish

The identified model supplies case-specific, counterfactually controllable report
content. A deterministic interface extracts continuous target values; a language
model translates them into prose and factual commitments. This is engineered
reporting, not spontaneous discovery of an introspective vocabulary. The neutral
and first-person conditions have identical state information. The first-person
instruction comes from the experimenter; lexical style comes partly from the
reporter's prior training.

**240/240 is structured-commitment fidelity, not independently verified prose
fidelity or consciousness-report correspondence.** Twelve underlying episodes are
a small sample; multiple prompts and conditions are correlated observations.
Independent assessment must judge the actual prose, compare control reports,
and address whether these distinctions amount to a manner of subjective access
rather than ordinary functional descriptions. Accurate functional reports alone
leave the source-of-qualia theory underdetermined. Proof of experience, emergence
without engineering, and superiority to a same-state decoder are not required.

The [blinded review form](report-review.html) is ready. No human ratings have been
received or fabricated. The [rubric](PREDICTIVE_REPORT_RUBRIC.md) predates generated
reports; it does not itself validate the proposed theoretical correspondence.
If ratings motivate a new aggregate criterion or interface revision, freeze it
before new confirmation reports. Do not declare the goal achieved from this
mechanical success alone.

## Reproduction and cost

See [offline reproduction](PREDICTIVE_ATTENTION_REPRODUCTION.md),
[internal confirmation](PREDICTIVE_CONFIRMATION_PROTOCOL.md),
[report confirmation v1](PREDICTIVE_REPORT_CONFIRMATION.md), and
[v2 correction](PREDICTIVE_REPORT_CONFIRMATION_V2.md).
All **168 tests pass**. The offline audit checks source/checkpoint hashes, all
672 attempted requests, 668 parsed reports, complete target-view derivation,
exact re-scoring, and three exact checkpoint replays. Archived API usage implies
approximately **$0.785** at the registered prices; an unreturned timed-out request
may have incurred additional usage unavailable in the response archive.

The [earlier inspection study](ATTENTION_MODEL_RESULTS.md) and its failed gates
remain unchanged. These new results do not retroactively pass that study.
