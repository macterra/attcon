# Executed Controller decisions: modeled effects guide acquisition and abstention

2026-10-04. The [frozen protocol](FUNCTIONAL_DECISIONS_PROTOCOL.md) implements
and executes a new deterministic Controller around the existing attention model.
This closes the missing category-action-trace component identified in the
[readout audit](FUNCTIONAL_READOUT_AUDIT_RESULTS.md). It adds no language reports,
human judgments, training or scientific success gate. Reporter and Controller
remain separate, and every earlier failed verdict remains unchanged.

## What actually executes

For a queried position in buffer 0, the Controller chooses the command maximizing
the minimum of its predicted dominant color/shape probabilities. It executes
one physical step at acquisition quality 0.8, observes actual allocation and
acquisition, updates the learned attention model and emits either both attribute
labels or an abstention. Both modeled probabilities must reach 0.6 after the
existing five-decimal rounding. The Controller never receives physical wiring,
automatic phase, physical recovery, scene labels or reporter text.

All 512 existing episodes per model are branched for all four queries, three
physical conditions and four policies: model-guided, random, effect-swap-guided
and exact restoration. This produces 36 contexts and 73,728 one-step task branches.
Branches reuse episodes and checkpoints; they are paired, correlated descriptive
observations, not 73,728 independent samples or a fresh confirmation study.

## Acquisition and category responses

The table gives ranges across models 2011/2021/2031. Each context has 2,048
episode/query branches. Answer fraction includes both correct and incorrect
answers; abstentions are not correct answers.

| Physical condition | Controller policy | Queried position acquired | Answer fraction |
|---|---|---|---|
| Commands control buffer 0 | Model-guided | 99.854–100% | 99.902–100% |
| Commands control buffer 0 | Random | 23.975–25.781% | 35.205–37.158% |
| Commands control buffer 0 | Effect-swap-guided | 9.814–19.287% | 25.244–33.740% |
| Commands control buffer 1 | Model-guided | 25% | 42.188–43.555% |
| Commands control buffer 1 | Random | 25% | 40.137–41.846% |
| Commands control buffer 1 | Effect-swap-guided | 25% | 40.479–41.455% |
| Commands control neither | Model-guided | 25% | 41.943–42.871% |
| Commands control neither | Random | 25% | 40.234–41.162% |
| Commands control neither | Effect-swap-guided | 25% | 40.771–41.211% |

Effect-swap planning changes 1,846 / 1,653 / 1,653 commands relative to model-guided
control in the buffer-0 condition, and changes exactly those physical buffer-0
allocations/recoveries. Responses change in 1,531 / 1,406 / 1,355 branches.
These numbers list seeds 2011/2021/2031 in order. Restoring modeled effects produces
exactly identical planning, commands, physical outcomes, updated model states and
responses for every branch.

When commands control buffer 1 or neither, changing policy leaves buffer-0 physical
allocation, recovery and physical thresholded responses exactly unchanged in every
paired branch. However, model-derived answers can differ: effect-swap planning
changes 34 / 35 / 45 responses in the buffer-1 condition and 22 / 29 / 34 in the
disconnected condition. These are estimator/cutoff effects in physically invariant
task-buffer outcomes, not physical access benefits. Model-versus-physical threshold
answer mismatches remain 3.027–3.516% for model-guided external control and
2.637–3.467% for model-guided disconnected control. Those comparisons use the same
engineered bridge and cutoff on physical recovery.

Model-guided own-buffer acquisition still misses 1 / 3 / 0 queried targets. Its
answers contain 2 / 0 / 0 wrong attribute pairs and 0 / 2 / 2 abstentions. Preserve
these errors; the study introduces no pass threshold to hide them. Raw-argmax
category accuracy is 99.9023% / 99.9512% / 100% across all conditions and policies,
and dominant categories never change between policies. Scene errors inherit the
fixed visual representation rather than demonstrating attention-created identities.

## Selective represented-recovery intervention

After each actual execution, reduce only the modeled buffer-0 recovery forecasts
to one quarter, keeping visual content, allocation, effects, recurrent state,
command and physical outcome fixed. Every emitted response becomes an abstention;
dominant category labels remain unchanged. Restoring recovery reproduces every
original response exactly.

This is an explicit response-rule dependency. It is also mathematically expected:
even a probability-one visual category has bridge probability at most 0.4375 after
this attenuation, below the declared 0.6 cutoff. It is not an independent discovery
of a phenomenal threshold or evidence that category decoding becomes impossible.
The intervention is transient at the readout, not a change to the estimator weights
or the physical world.

## What this adds to the project

There are now executed commands and category decisions causally linked to modeled
effect/recovery state. Actual-world feedback is distinct from self-generated
predictions and from the Reporter. This supplies a functional-use component that
the prior JSON export lacked. The rule itself is engineered, and the visual
representation is supplied unchanged rather than reacquired from noisy sensory
input. These observations do not establish that bridge probabilities are calibrated
to actual category errors or that abstention is a necessary information boundary.

All conditions contain attention models. This assay supplies neither an independently
justified consciousness-report contrast nor evidence of subjective experience.
Substantive author/independent definition review and factual-audit qualification
remain pending, followed by fresh reporting, powered confirmation and replication.
The prepared review packet and unspent language-development row are unchanged.

## Verification and archive

Seventeen relevant unit tests pass, including checks that execution uses actual
observations, planning follows changed forecasts and answering can faithfully follow
a physically wrong model. Exact replay covers every archived physical, modeled,
action and response array, metric, CSV row and archive hash. The separate paired
analysis is post hoc and descriptive; it verifies task-buffer physical invariance.

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest tests.test_functional_decisions tests.test_functional_interface tests.test_functional_controls tests.test_functional_model tests.test_predictive_attention
.venv/bin/python scripts/run_functional_decisions.py --stage verify
.venv/bin/python scripts/summarize_functional_decisions.py
```

The [manifest](https://github.com/macterra/attcon/blob/main/audits/functional_decisions_v1/manifest.json),
[summary](https://github.com/macterra/attcon/blob/main/audits/functional_decisions_v1/summary.json),
[all metrics](https://github.com/macterra/attcon/blob/main/audits/functional_decisions_v1/metrics.csv)
and [paired change counts](https://github.com/macterra/attcon/blob/main/audits/functional_decisions_v1/paired_policy_changes.csv)
are retained with exact source snapshots and three lossless decision archives.
Preparation was published at `f7dda50`, and full executed outcomes at `bc2da3e`.
