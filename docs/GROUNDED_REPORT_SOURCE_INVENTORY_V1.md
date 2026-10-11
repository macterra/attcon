# Grounded report source inventory v1: draft interface

2026-10-10. This source-only development component supplies inspectable factual
records. It has not generated experimental target views, run a Reporter, qualified
appearance correspondence or achieved the theory goal. The Controller remains
separate from the external Reporter. Registration and independent source-interface
review must precede target export and any reporting pilot.

The draft code is `scripts/grounded_report_interface_v1.py`, with generated
fixtures in `tests/test_grounded_report_interface_v1.py`. Importing it performs no
archive loading, source inference, probe fitting, record construction or writing.
There is no CLI/output writer. Future archive loading and record construction are
separate explicit calls.

## Source and fixed opportunity

Use the completed physical diagnostic's evaluation split 2, snapshot after eight
actual commands, for frozen visual seeds 1301/1311/1321. Those worlds differ from
the original grounded-outcome evaluation worlds. The outcome-model identity and
weights provenance point to the grounded study's frozen structured instruments;
we do not mix the two studies' per-world records.

A future exporter must retain worlds 0–15 by index for every model and all four
locations for every world, including unseen locations. No truth, correctness,
forecast extremum or reference similarity selects records. `selection_indices`
returns this fixed rule, and `record` rejects a row outside it. No selected
production world has been read during this draft's development.

The loader verifies completion, manifest/results hashes, the exact-replay receipt,
consumed archive hashes and the underlying structured instrument hash. It extracts
only declared memory/forecast/history/readout keys. It does not re-run a network
or external decoder. Mixed snapshot archives contain evaluator pixels/targets;
those remain outside the returned `SourceBundle` and reporting interface.

## Reporter-visible whitelist

| Field | Meaning and limits |
|---|---|
| Snapshot and frozen visual/structured/external-readout IDs | Provenance, with neutral command/location identifiers. No world index, condition ID, filename or intervention operator is supplied. |
| Native latest-per-command route-history rows | Each command's latest actually observed delivery, one-hot, or all-zero unknown. This is the model's history input, not a full chronological sequence. |
| Per-location observed flag and evidence item | Current stored-sample status. Item IDs denote stored evidence records, not independently tracked enduring objects. |
| Stored orientation logit, sign and fixed stored decision | Uncalibrated decision coordinate; positive selects vertical, otherwise horizontal. These are represented decisions, never certified world orientation. |
| Current/fresh-glimpse decision-success estimates | Learned represented forecasts, not observed correctness, sensory probabilities or felt confidence. |
| Delivery distribution and post-command decision-success per command/location | Forecasts. The exporter checks exact `delivery*refresh+(1-delivery)*current` composition against the archived matrix. It exposes no selected Controller policy or future result. |
| Risk-feature provenance | The local structured `risk_hidden`, dimensions 32 and frozen model identity. Hidden coordinates are not assigned qualitative meanings or dumped into a language prompt. |
| External image-value SD and nearest-neighbour RMS estimates | Archived shared linear readouts of risk features, explicitly external objective-statistic estimates in normalized image-value units. They are not native presentation or confidence fields. No clipping hides invalid physical estimates. |
| External readout aggregate MAE/RMSE and bootstrap intervals | Common all-seen evaluation errors; conditional fixed-model uncertainty, not per-item bounds or presentation qualification. Only these numeric summary fields are copied, not true-contrast/orientation strata. |

When unseen, stored logit/sign/decision/item and both physical estimates are null.
The separate fixed horizontal default guess is explicitly a default. Current,
refresh and post-command forecasts remain present; a chance decision forecast is
not observed content. No categorical sensory-confidence calibrator is used.

## Source-aware audit layer and draft variants

`record` returns `reporter_input` and `audit_provenance` separately. Only the first
is eligible for a future Reporter. Audit provenance retains source hashes/paths,
world index, source variant, synthetic-source flags, donor location/event indices,
and the genuine eight-event command/delivery chronology. That chronology is
external observed-history instrumentation: it was not consumed as a full sequence
by the outcome predictor and is not current Reporter context. Any later context
arm must declare that extra information and give every matched control the same
history.

Six source views are **draft**: natural, sham, native memory readdressing,
represented-delivery patch, inverse memory restoration and inverse delivery
restoration. Memory readdressing selects already archived frozen-network inference
and external readouts; route patching selects already archived output matrices
and recomposition without claiming new neural inference. Both keep the actual past
history unchanged. Shared source-semantics text states that current memory defines
represented contents, history does not certify their current placement, and
represented forecasts are not observed future events. Synthetic operator IDs stay
audit-only. Restoration must reproduce source/readout values and Reporter input
exactly; it has not tested prose restoration.

A same-information address-aware table helper separates evidence, risk,
external-readout, delivery and post-command tables. Deterministic joining recovers
the entire main input; addresses/history are retained. No ordinary apparatus
narrator has run. Factorial intervention conditions, identifier-remapping assay
conditions, matched narrators, prompts, annotation qualification and trial-level
acceptance rules remain prospective. This draft supplies no desired presentation
adjective, example consciousness report or reporting instruction.

Excluded from Reporter input and its whitelisted bundle: world orientation,
physical contrast/mask/phase/permutation, correctness labels, measured image targets,
pixels/noise, future images/branches/outcomes, human/reference ratings, arbitrary
appearance classes, calibrated sensory probabilities and scoring notes. An
estimated physical property does not automatically earn a qualitative presentation
feature. Natural-state and synthetic-source results must remain separate in any
later study.

## Generated-fixture validation receipt

Command: `PYTHONPATH=src:scripts .venv/bin/python -m unittest discover -s tests -p 'test_grounded_report_interface_v1.py' -v`.

| Development pass | Retained outcome |
|---|---|
| Initial import | Syntax error from an extra closing brace; corrected before any fixture ran. |
| First eight-fixture run | Seven passed; the negative-readout checker assumed its fixture was negative when it was +0.2. Corrected the toy offset, not production/source semantics. |
| Corrected eight fixtures | 8 passed in 0.280 seconds. |
| Added fake-archive boundary fixture | Eight passed; a string search for `999` incorrectly matched ordinary float decimal digits. Replaced it with exact equality of the whitelist record against the known toy source. |
| Final draft | **9 passed in 0.366 seconds.** |

Fixtures test fixed first-sixteen opportunities/all locations, unseen undefinedness,
sign/decision semantics, exact composition rejection, native versus external history,
readdressing/restoration, route-only patch provenance, table information equality,
unclipped estimates, forbidden-summary exclusion, malformed-source rejection and
fake-archive hash/replay/whitelist handling. All source tensors and fake archives
are generated fixtures; no real checkpoint inference, selected production record,
new fit, experimental target serialization, Reporter call or report rating occurs.
These tests establish interface checks, not consciousness-related correspondence.
