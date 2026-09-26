# Reporting-first research plan

Created 2026-09-26. This is the active execution plan; existing Stage 8 results and
their claim boundaries remain in force.

## Goal and decision rule

Develop a reproducible method for testing integrated, behaviorally accessible
internal content. Reporting is useful evidence only when it tracks internal
history, survives selective interventions, and connects to independent behavior.
No result in this toy program establishes subjective experience.

The next priority is to test reports on a task-trained representation without
training the agent on reports, source labels, confidence, reinspection, or explicit
access flags. A separately trained frozen-state reporter is still a supervised
measurement instrument: successful decoding alone is not spontaneous self-report
or a higher-order representation.

## Execution sequence

1. **Paired-history reporting pilot (implement and run now).** Train a generic GRU
   on delayed value choice. Counterbalance seen and unseen histories with identical
   final observations. Fit reporters only after freezing the agent, on disjoint
   context groups. Compare internal state with current observation, action logits,
   an untrained GRU, and shuffled report-label controls. Selectively transplant
   content directions fitted on reporter-training states; measure both report and
   action changes, access stability, and unrelated query-identity stability.
2. **Replication and access specificity.** Repeat the fixed pilot over three
   independent data/model seeds. Test longer delays, reordered events, and a second
   sequence architecture. Add selective access erasure/restoration and stale-memory
   cases. Distinguish reporting an encoded trace from reporting usable access;
   compare against an observer with the full history as an information upper bound.
3. **Endogenous regulation.** Let the agent choose whether and where to inspect
   under a cost. Train only on environmental returns. Test whether internal
   uncertainty/access representations predict and causally guide information
   seeking. Hold reporting supervision outside agent training. A failure here
   leaves Stage 4B and stronger access claims unsupported.
4. **Independent content convergence.** Add binding and delayed action tasks with
   optional communication between independently learned pathways. Require that
   interventions fitted on one function transfer to another for the same content,
   against viable split-pathway and shuffled-direction controls. Do not impose the
   state sharing whose emergence is being tested.
5. **Cross-system audit.** Repeat positive causal results across task structures,
   architectures, seeds, and comparator systems. Recompute the Stage 8 audit only
   when its existing evidence requirements are actually met.

Failure is an experimental outcome. A failed gate leads to a labeled diagnostic
or a new preregistered version, never a retrospective threshold change. Do not
spend external API credits until the internal-state assay works reliably.

## Pilot protocol frozen before the first run

- Eight keys, six values, four possible historical record positions, one blank
  delay step, and a final query. No exact-match routing or explicit memory slots.
- Each context contains all six possible queried values, each with a seen and an
  unseen history. In unseen histories the queried key never appears; its hidden
  answer is uniformly counterbalanced and cannot be inferred from the history.
- Agent receives event vectors only. Value-choice cross-entropy applies to both
  seen and unseen cases; unseen cases incentivize a uniform choice distribution.
  There are no unknown-action, confidence, source, or reporting targets for it.
- Disjoint context groups: 256 agent training, 128 reporter fitting, 64 validation,
  128 test. All variants of each context remain in the same split. This tests new
  contexts, not a claim of held-out key/value-conjunction generalization.
- Agent: 64-unit GRU, 40 epochs, AdamW at 0.003, batch size 256. Reporter: linear
  seven-class readout (six values plus unavailable), 250 full-batch updates. Inputs
  are padded to 64 dimensions so all report probes have identical parameter counts.
- Twenty independently permuted reporter-label fits estimate an empirical p95 null.
- Fit content directions only on seen reporter-fit states: the rank-five span of
  centered per-value state means. Transfer only that projection from a different
  value in the same held-out context. Compare with a fixed random rank-five
  subspace and a subspace fitted after permuting content labels. Report recipient/
  donor baseline coverage as well as unconditional intervention outcomes.
- Initial thresholds: seen choice accuracy >= 0.85; seen value-report accuracy and
  unseen unknown-report accuracy each >= 0.85; paired report accuracy >= 0.75;
  balanced report advantage over current observation >= 0.25 and empirical null
  p95 >= 0.20; joint action/report donor following >= 0.70; advantage over both
  causal nulls >= 0.25; report-access and query-identity stability >= 0.90, with
  baseline query-identity accuracy >= 0.85 and intervention eligibility >= 0.75.
- Action-logit and untrained-state probes are diagnostics, not gates: a positive
  action-logit probe bounds the need for richer state; an untrained-state success
  demonstrates reservoir decoding rather than learned introspection.
- Validation is reported separately. No choices or thresholds are tuned on test
  outcomes. The initial artifact is a single-seed pilot regardless of gate results.

## Deliverables and interpretation

The runner writes exact settings, split checks, losses, baseline and null scores,
intervention metrics, gates, and claim boundaries to a versioned JSON audit. Its
checkpoint is saved under ignored `outputs/` for reproducibility. Tests protect
counterbalancing, split isolation, reporter isolation, and intervention controls.

Even all gates passing means **bounded task-trained memory/report coupling**.
It does not establish learned regulatory self-modeling, spontaneous reporting,
independent multi-theory convergence, or satisfy Stage 8.

## Follow-up registered after the initial v1 pilot

The initial seed-1729 pilot failed task viability (seen choice accuracy 0.602),
reporting, and causal eligibility gates. Its artifact is retained unchanged.
Validation choice accuracy was also low (0.630). A dataset test found that unseen
histories could duplicate across splits for a different seed; the sampler now
rejects those actual duplicates. The original pilot passed the input-isolation
check and is unaffected by this correction.

Next diagnostic: use fresh seed 1741 and 160 training epochs, retaining the model,
reporter, split sizes, and original thresholds. This is a training-duration
diagnostic with new data, not a controlled estimate of the effect of duration.
Before running it, add random and permuted-subspace interventions with each
example's perturbation norm matched to the true content intervention. Require
joint-follow advantages >=0.25 against both additional controls. These v2 controls
address a limitation of v1, whose random subspace produced smaller state changes.
Do not promote a result merely because conditional intervention scores are high
when few recipient/donor pairs have correct baseline reports.

The longer run fit its training contexts perfectly but reached only 0.581 seen
choice accuracy on validation and 0.602 on test. Its reporting gates still failed.
This motivates a data-coverage diagnostic: seed 1753, 1,024 agent-training context
groups, 160 epochs, and the same reporter partitions and v2 gates. This changes
data coverage, not the architecture or report supervision. It is another fresh
pilot, not a paired causal attribution of the training-data effect. If viable,
repeat the identical configuration at seeds 1759 and 1777; retain every result.

## Results from the first execution

All five runs are retained: initial seed 1729, longer-training seed 1741, and
coverage seeds 1753, 1759, and 1777. Training diagnostics reproduce the original
validation/test scores from saved checkpoints. All dataset-isolation checks pass.

The coverage configuration is task-viable on all three seeds. Its held-out ranges
are below; ranges are seed minima/maxima, not confidence intervals.

| Measurement | Three-seed range | Frozen criterion |
| --- | --- | --- |
| Seen value choice | 94.8–95.7% | >=85%; passes all seeds |
| Seen value report | 81.9–87.0% | >=85%; passes one seed |
| Unseen unavailable report | 57.8–68.0% | >=85%; fails all seeds |
| Both reports correct within a history pair | 50.5–57.3% | >=75%; fails all seeds |
| Joint action/report following after content transplant | 74.0–79.6% | >=70%; passes all seeds |
| Joint-follow advantage over norm-matched random control | 72.8–78.8 percentage points | >=25; passes all seeds |
| Joint-follow advantage over norm-matched permuted control | 67.4–77.5 percentage points | >=25; passes all seeds |
| Access retained after content transplant | 84.5–88.7% | >=90%; fails all seeds |
| Baseline-correct recipient/donor eligibility | 66.7–75.5% | >=75%; passes one seed |

Full-state reports have balanced accuracy 72.4–75.1%, versus 50% for current
observation and 49.9–50.1% for untrained recurrent state. The action-logit reporter
reaches 77.2–82.0%, higher than the full-state reporter on each seed. Its inputs
are themselves a linear transformation of state, so this does not show that the
state lacks the information. It suggests a readout/generalization or calibration
limitation and bounds any claim that richer state access is necessary.

**Verdict: reporting gates not met.** Task-trained content has repeatable causal
coupling to choices and fitted reports in this assay. Accurate reporting of
unavailability and stability of access reports remain unresolved. Reporters are
supervised instruments; neither spontaneous self-report nor higher-order state
has been demonstrated. Stage 8 remains unchanged.

Evidence:

- `audits/paired_history_reporting_coverage_multiseed.json`: validated all-seed
  summary, frozen gates, per-metric ranges, and links to every coverage run.
- `audits/paired_history_training_diagnostic.json`: initial/long-training
  generalization gap, reproduced from saved checkpoints.
- `audits/paired_history_reporting_seed1729.json` and
  `audits/paired_history_reporting_v2_seed1741.json`: retained unsuccessful pilots.

The availability-readout diagnostic has now been executed on three fresh seeds
under a [fixed protocol](REPORTING_CALIBRATION.md). Its
[results](REPORTING_CALIBRATION_RESULTS.md) show a 13.3–14.1 percentage-point gain
in unavailable reports, but a 3.1–6.4 point loss in seen-content reports. Every
seed still fails the complete reporting criteria; simple action-score and entropy
reporters remain competitive. Calibration is a partial measurement limitation,
not a complete explanation of the remaining failures.

The [six-cycle campaign](REPORTING_CAMPAIGN.md) has now tested matched nonlinear
reporters, full-history observers, reporting-data coverage, an ungated RNN, delay
stress, and diagnostic content lesions. More fitting data improves decoding on
identical frozen agents, but unavailable reporting remains below threshold. The
RNN recipe fails task viability; extra delays weaken both choices and reports.
Lesion-induced unavailable reports provide bounded sensitivity, with residual
incorrect-value reports and no proof of inaccessible content.

The [delay and inspection campaign](REGULATION_CAMPAIGN.md) completes those
follow-ups. Variable-delay training restores choice robustness across three
seeds, while full reporting gates remain unmet. A nearly parameter-matched RNN
fails viability. Choice-null-space transplants isolate modest additional report
and policy sensitivity, but reward-trained action/confidence inspection policies
outperform full-state policies at every cost and seed. Keep all reporting gates
and the separation of reporter supervision from agent/control training. Next:
learn recurrent information acquisition in the environment with fresh, stale,
and missing information; require a replicated reward advantage beyond confidence
before claiming useful regulation. Reliable access reporting remains open.

## Reproduction and validation

Run a coverage pilot (repeat for seeds 1759 and 1777):

```bash
.venv/bin/python scripts/paired_history_reporting.py --seed 1753 --train-groups 1024 --epochs 160 --norm-matched-controls --out audits/paired_history_reporting_coverage_seed1753.json
```

Rebuild the replication summary:

```bash
.venv/bin/python scripts/summarize_history_reporting.py audits/paired_history_reporting_coverage_seed1753.json audits/paired_history_reporting_coverage_seed1759.json audits/paired_history_reporting_coverage_seed1777.json
```

The full unittest suite passed (64 tests at that point, including six new tests).
The final targeted suite passes all seven new tests, including the subsequently
added causal-scoring and matched-norm regression. Source compilation and whitespace
checks pass. No paid APIs were used.
