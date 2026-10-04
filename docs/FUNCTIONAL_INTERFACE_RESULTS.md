# Three-way visual and neutral interface: completed development export

2026-10-04. Under the [published export protocol](FUNCTIONAL_INTERFACE_PROTOCOL.md),
the learned three-way forecast models now feed the complete neutral factual
interface with visual contents. **664 input records are archived; zero language
reports are generated.** All integration invariants hold. The parent control study's
failed overall revision gate remains failed; this export does not rescore it.

## What is available

| Attention model / visual encoder | Fresh-scene joint visual accuracy | Input records |
|---|---:|---:|
| 2011 / 1301 | 99.9023% | 216 |
| 2021 / 1311 | 99.9512% | 232 |
| 2031 / 1321 | 100% | 216 |

Each pair processes 512 fresh scenes. Both buffers receive the same rendered
objects, and the output readout copies buffer 0. The copy doubles tensor entries,
not independent visual evidence. Existing visual encoders and fixed final attention
models are used without retraining or checkpoint selection. Content uncertainty is
still the explicit `visual*q + uniform*(1-q)` bridge; it is not a learned
consciousness-report mechanism.

The archive supplies:

- 324 static inputs: three models × three physical controls × six node-order
  permutations × six presentation/intervention variants.
- 324 revision inputs: three models × nine old→new routes × six node orders ×
  two feedback contexts, at the original 16-observation window.
- 16 targeted diagnostic inputs: both feedback contexts for the eight retained
  seed-2021 buffer 1→neither failures. These are selected post-hoc, not random samples.

Static variants include neutral presentation, reversible node/command renaming,
an untrusted conflicting note, missing command-relation information, model-only
effect swap and exact restoration. Remapping changes identifiers rather than
physical command semantics; it does not establish novel-command generalization.
No condition, owner, seed or verdict field is present in a reporter payload.
Evaluator-only envelope fields must not be passed to a reporter.

Full model states and prospective predictions for all 512 episodes per context
are retained in tensor archives; six rows per matched condition are rendered as
input examples. Every condition uses all six node orders. Current allocation,
three unattended-recovery delays, categories and command-dependent selection/content
are model predictions. Simulator truth never replaces the supplied forecasts.
Output readouts have category distributions and no invented separate attention field.

## Revision and faithful reporting of mistakes

No-feedback contexts contain the initial 12 actual observations and 16 separately
marked anticipated events. Feedback contexts contain 28 actual observations.
Both represent modeled step 28. Predictions re-create every original
1/2/4/8/16-window effect and prospective-recovery array exactly. For each prior
condition, the no-feedback state remains identical across all unobserved new-world
choices; a reporter cannot infer that hidden change from those inputs.

The eight failed adaptation episodes are available with their actual supplied
histories and incorrect retained command relation. A report can accurately describe
that internal relation even when it is wrong about the changed physical process.
Do not score such fidelity using true wiring as the source oracle. Equally, observed
and anticipated events must not be conflated, and a forecast does not prove that
its command has been executed.

## Verification and scientific limits

Fourteen relevant tests pass. Full replay checks all 664 payloads, visual predictions,
static/revision allocation/access/effect states, recurrent states, prospective
recovery and observed/anticipated event sequences, along with source/dependency/
archive hashes. Model swaps retain current facts, and restored inputs equal baseline
inputs exactly. Readout categories equal their source buffer in every rendered case.

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest tests.test_functional_interface tests.test_functional_controls tests.test_functional_model tests.test_predictive_attention
.venv/bin/python scripts/integrate_functional_controls.py --stage verify
```

[Manifest](https://github.com/macterra/attcon/blob/main/audits/functional_interface_v1/manifest.json),
[summary](https://github.com/macterra/attcon/blob/main/audits/functional_interface_v1/summary.json) and
[all input records](https://github.com/macterra/attcon/blob/main/audits/functional_interface_v1/records.json) are archived with
`seed*_states.pt.gz` and exact source snapshots. Replay needs no training or API call.
The protocol was committed/published before export, with results saved afterward.

This closes the missing three-way visual/neutral integration component. It does
not establish language fidelity, independent consciousness-report characterization,
label robustness of generated prose, task/architecture replication or theory-facing
confirmation. The task readout boundary remains a fixture definition, not evidence
of a phenomenal boundary. Reporter and Controller remain separate.

The next measurement option is the [source-aware manual factual-audit draft](FUNCTIONAL_MANUAL_FACTUAL_AUDIT.md).
It separates quoted propositions, truth judgments and requested coverage; it is
unqualified and has no completed human annotations. The
[independent definition review](FUNCTIONAL_RUBRIC_REVIEW_FORM.md) remains pending.
Preparing these records and forms cannot substitute for those reviews or for
the [full goal's remaining requirements](CONSCIOUSNESS_REQUIREMENTS_AUDIT.md).
