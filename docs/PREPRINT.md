# Accurate Reporting from Internal States in Small Recurrent Systems

Updated 2026-09-26. This manuscript describes the completed Attcon reporting
study. It replaces the earlier attention-control-centered draft; that benchmark
and the broader theory program remain documented in the [original specification](SPEC.md)
and [historical roadmap](ROADMAP.md).

## Abstract

Can a system accurately report its internal informational state? We study this
question using separately trained readouts of frozen recurrent agents in two
synthetic information-acquisition tasks: sequential verification and sensor
selection. The final evaluation comprises 36 fitted controllers across two
architectures, three policy families, and three seeds, plus 12 controllers trained
from experienced rewards. Reporters are fitted on 12 frozen full-state systems
using contexts disjoint from agent training and held-out evaluation. Across all
six GRU systems, verified-value and unverified-status reporting are 100% accurate,
while remembered sensor-quality reporting is 96.04–100% accurate. Serial RNN
quality reports reach 91.72–94.86%; routing RNN results are more variable at
78.76–99.96%. A corrected comparator supplied with answer logits and the observed
quality cue matches or exceeds the full-state report score in all 12 systems.
This limits claims of comparative advantage without negating reporting accuracy.
Answer-preserving quality interventions change 89.1–100% of routing GRU sensor
choices, compared with 0.3–4.4% under matched random interventions, linking reported
information to behavior. The findings support accurate reporting through trained
internal-state readouts in these tasks. They do not establish independently
learned native reports of the same distinctions, general introspection, or
subjective experience. Necessity for task performance is a separate question
from the reporting-accuracy goal.

## 1. Research question and scope

The primary project goal is accurate reporting of internal informational states.
We operationalize this through reports of remembered source quality and whether
initial information contains a verified current value. These are measurable
informational distinctions, rather than direct labels of conscious experience.

Three questions need separate answers:

1. **Accuracy:** can a trained reporter recover the specified information from
   the agent's internal state on held-out contexts?
2. **Causal involvement:** does manipulating the reported representation also
   affect behavior while preserving other measured outputs?
3. **Comparative advantage:** does access to the full state improve reporting or
   control over simpler information sources?

Accurate reporting does not require positive answers to the latter two questions.
A simpler readout reporting equally accurately can identify information sufficient
for the report. Likewise, useful task behavior does not require demonstrating
that conscious access was necessary to produce it.

The final experiment matrix was registered with additional comparative and causal
criteria. After evaluation, the project's reporting-first interpretation was
clarified. We retain the [original protocol](COMPLETION_PROTOCOL.md), its thresholds,
and all gate outcomes. This manuscript distinguishes absolute reporting accuracy
from those stronger criteria; it does not retrospectively relabel failed gates
as passes.

## 2. Background and earlier experiments

Attcon began with a cue-guided selective-search benchmark on a `5x5` grid. The
implemented controller uses a soft attention policy and a straight-through
single-cell glimpse. An earlier fully soft glimpse failed to learn the task;
its results were superseded after that implementation was repaired. On the
regenerated benchmark, recurrent and static baseline accuracy were 0.44 and 0.17,
respectively, with chance at 0.10. Those results concern the original attention
benchmark and are not the final reporting-study scores.

Subsequent experiments tested reporting from task-trained memory. The
[paired-history assay](REPORTING_PLAN.md) held final observations fixed while
varying prior access. Three viable agents reached 94.8–95.7% held-out choice
accuracy, but unavailable-content reports were only 57.8–68.0% accurate. A
[calibration diagnostic](REPORTING_CALIBRATION_RESULTS.md) and
[nonlinear reporting campaign](REPORTING_CAMPAIGN.md) improved some metrics
without satisfying all original reporting criteria. These failures remain part
of the evidence.

The [delay campaign](REGULATION_CAMPAIGN.md) improved temporal robustness, and
the [recurrent acquisition campaign](ACTIVE_INSPECTION_CAMPAIGN.md) learned
repeated inspection. Their results motivated the final study's separation of
current answer confidence from remembered prospective sensor quality. The final
positive reports use explicit environmental quality and validity cues; they do
not establish that the earlier paired-history reporting problem was solved.

## 3. Methods

### 3.1 Environments and data partitions

Both final tasks use six possible answer values and fresh, stale, or missing
initial information. Fresh information contains a valid current value; stale
information contains an independent old value and an invalidation cue; missing
information contains no initial value. A quality cue precedes the query. Four
distractor events, initial evidence, a validity cue, a delay, and a query form
the recurrent event stream. The observation representation has 14 channels.

In **serial verification**, an agent can inspect twice. The first sample has cued
reliability of either 0.55 or 0.90, and the second verifies the value exactly.
In **sensor routing**, the agent can choose sensor A or B before answering.
Sensor A's reliability is the cue value; sensor B's is 1.45 minus that value.
Both sensors have the same inspection cost.

Seeds 2503, 2521, and 2539 each use 128 agent-training contexts, 32 validation
contexts, 64 reporter-fitting contexts, 64 primary test contexts, and 64 reserved
stress contexts. Within each context, all six values, three initial conditions,
three costs (0.1, 0.25, 0.4), and two qualities are crossed. All 108 variants of
a context stay in one partition. Noise draws are shared across quality variants.
Training delays are zero or two steps; primary evaluation uses one step.

The quality and validity cues are environmental observations. They do not label
whether the agent has usable internal access after forgetting or perturbation.
Held-out contexts test generalization within this generator, not new domains.

### 3.2 Controllers

The fitted matrix crosses two tasks, GRU48 and RNN87 architectures, three policy
families, and three seeds, yielding 36 controllers. The architectures have 13,704
and 13,683 parameters, respectively. The inspection head receives either the
full recurrent state, confidence statistics, or confidence statistics plus the
observed quality cue. All families receive cost and budget; all encoders receive
the full event stream. The cue comparator has exact cue retention as a memory
aid. Its answer decisions retain the answer logits.

Fitted training combines answer cross-entropy with inspection values learned
from fitted Bellman targets over replayed branches. Training uses Adam at 0.003,
batches of 512, and 80 epochs, with target-network updates each epoch. Epochs
40, 60, and 80 are candidates for validation-based selection. Initialization,
minibatch schedules, and inspection-head capacity are matched within comparisons.
No report labels enter controller training.

A separate matrix trains 12 GRU controllers from scratch using only experienced
chosen-action rewards: two tasks, state or cue-confidence policies, and three
seeds. Epsilon-greedy exploration decreases from 0.40 to 0.10 over 600 minibatches
of 512 episodes. The agent may answer, inspect, or decline for a payoff of 0.30.
Chosen-answer correctness is learned from binary reward; inspection values use
cost and target-network continuation. Validation selects among updates 200, 400,
and 600. This is a different objective and training budget from fitted control,
so cross-matrix differences are not controlled estimates of algorithm superiority.

### 3.3 Reporters and accuracy measures

Each of the 12 fitted full-state controllers is frozen before reporter training.
Reporters read the root state before acquisition. The target has 14 classes:
two quality levels crossed with six verified values or an unverified label.
Fresh initial information is labeled verified; stale and missing information
are labeled unverified.

The registered reporter inputs are recurrent state, answer logits, or maximum
answer probability and normalized entropy plus the quality cue. Inputs are
padded to 128 dimensions, with a 32-unit tanh hidden layer and 14 outputs,
yielding equal-capacity readouts. Normalization is fitted only on reporter-fitting
data. Reporters train for 200 Adam steps at 0.01, with L2 candidates of zero and
0.001 selected using validation accuracy. Five shuffled-label fits provide a
coarse descriptive null rather than a formal significance test.

Quality accuracy measures recovery of the remembered low/high cue. Verified-value
accuracy measures the correct value within verified cases; unverified accuracy
measures the unverified label within stale/missing cases. Balanced source accuracy
averages the latter two metrics. The primary comparison score averages quality
and balanced source accuracy. Joint accuracy requires both quality and the
value/source label to be correct.

The registered individual accuracy thresholds are 90%. A separate registered
comparative threshold requires a 0.02 advantage over the cue reporter. We report
these separately because absolute accuracy and comparative advantage answer
different questions.

### 3.4 Comparator correction

The original confidence-plus-cue reporter omitted answer identity even though
its target required naming a verified value. Its disadvantage therefore could
not establish fair full-state reporting superiority. After inspecting the three
serial-GRU reporting artifacts, we documented this flaw and added a diagnostic
comparator to all 12 systems: all six answer logits plus the same quality cue,
with matched reporter capacity, partitions, optimization, and selection.

The [correction record](COMPLETION_CORRECTIONS.md) preserves that chronology.
The diagnostic reused the existing test contexts and is not independent
confirmation on new data. Original artifacts and gate outcomes remain intact.
The control-policy comparison did not have this omitted-answer-identity flaw.

### 3.5 Causal interventions and stress evaluation

We fit a high-minus-low quality direction using paired reporter-fitting states,
remove its component in the answer-weight row space, and normalize it. A low-
quality donor's projection is transplanted into a high-quality recipient matched
on context, value, condition, and cost. Interventions are evaluated on initially
unverified held-out histories. Random directions in the same answer null space
are matched to each case's perturbation norm.

We check initial answer-logit invariance within `1e-5`, changes in reports and
inspection behavior, environmental returns, and restoration to the baseline
state. Actual sensor reliability remains unchanged under the transplant.
Consequently, reward changes measure consequences of manipulating the internal
representation, not the truth of an introspective report about a changed world.

Reserved-context control evaluation uses the primary delay, a five-step delay,
and a reliability reduction of 0.15 while the original quality cue remains
visible. Serial verification's exact second sample remains exact. These stress
evaluations measure control performance; they do not establish reporter accuracy
under every stress condition. Analytic comparison policies know actual degraded
reliability, whereas learned policies see the old cue.

## 4. Results

### 4.1 Accurate reports from frozen internal states

| Task and architecture | Verified value | Unverified status | Quality | Joint quality and source/value |
| --- | --- | --- | --- | --- |
| Serial GRU | 100% | 100% | 96.99–100% | 96.99–100% |
| Routing GRU | 100% | 100% | 96.04–99.99% | 96.04–99.99% |
| Serial RNN | 100% | 100% | 91.72–94.86% | 91.72–94.86% |
| Routing RNN | 100% | 100% | 78.76–99.96% | 78.76–99.96% |

Ranges are minima and maxima across three trained seeds, not confidence
intervals. All six GRU systems and all three serial RNN systems clear all three
individual accuracy thresholds. Two of three routing RNN systems clear the
quality threshold. Every final system reports verified values and unverified
status correctly on its primary held-out evaluation.

These are positive results for the reporting-accuracy question. Their scope is
specific: a supervised readout can recover these distinctions from task-trained
internal states. The cue-rich final environments and report targets differ from
the earlier paired-history assay, whose unavailable-report failures remain.

The source summaries are linked from the [final evidence map](PROJECT_RESULTS.md)
and recorded in the [completion audit](https://github.com/macterra/attcon/blob/main/audits/project_completion.json).

### 4.2 Simpler information suffices for equally accurate reports

The corrected answer-logit-plus-quality comparator matches or exceeds the
full-state reporter's primary score in all 12 systems. No corrected comparison
clears the 0.02 full-state advantage threshold. The initial apparent advantage
was confounded by omitted answer identity.

This result does not invalidate the state reporters' accuracy. It demonstrates
that the corrected comparator's information is also sufficient for those reports.
Because that comparator receives the quality cue directly, it is not itself a
demonstration of learned quality memory.

### 4.3 Reported quality is causally connected to control

In routing GRUs, the quality transplant changes 86.7–100% of quality reports and
89.1–100% of sensor choices. Matched random perturbations change 0.3–4.4% of sensor
choices. Initial answer logits remain invariant within tolerance, and restoration
recovers baseline reports and decisions. Returns decrease by 0.310–0.362 when the
internal cue is changed while the actual sensor remains unchanged.

In serial GRUs, initial inspection decisions remain unchanged, but 36.1–58.6%
of later verification trajectories change, compared with 3.5–27.3% for matched
random perturbations. RNN effects are weaker or less selective; random changes
sometimes improve their weaker baseline policies. All 12 systems pass the
intervention invariance and restoration checks.

The GRU findings link reported prospective information to behavior. They do not
establish that such information is a higher-order representation, that all
perturbation responses are beneficial, or that conscious access is necessary.

### 4.4 Acquisition and native answer/decline

Fitted GRU state policies beat every fixed inspection policy by at least the
registered 0.02 margin at every tested cost and seed in both tasks. No task and
architecture combination consistently clears the fair cue-confidence advantage
across all required costs and seeds. RNN performance is more variable.

Reward-only controllers learn acquisition and answer decisions, and some use the
native decline action. They do not achieve a replicated advantage over the fair
comparator across all costs and seeds. Answering or declining is a native output
learned through reward, but decline can reflect a payoff-sensitive confidence
threshold. These experiments did not train native reports naming remembered
quality, verified content, or source status.

### 4.5 Generalization and broader criteria

Longer delays and misspecified sensors often reduce control reward. Five of 108
individual stress seed/cost cells pass the complete registered control gates;
no task satisfies the full promotion requirement across costs, seeds, and
architectures. These exceptions are retained without changing the criteria.
Context-bootstrap intervals use 2,000 resamples and are conditional on trained
systems, not population-level uncertainty over arbitrary models.

The prerequisite for promoting independent-content convergence was not met.
The separate Stage 8 audit remains three passing and five partial gates, with
no overall support. This outcome concerns the broader historical convergence
program, not whether the measured reports were accurate.

## 5. Interpretation and limitations

The central result is accurate reporting through trained internal-state readouts
on held-out contexts in small synthetic tasks. A controller plus a fitted
reporter can produce correct reports of specified informational distinctions.
The reporter remains a supervised measurement component: we have not shown that
the controller independently learns to express those same distinctions.

Several boundaries matter when interpreting the result:

- **Report target:** remembered environmental quality and source verification are
  narrower than knowledge of one's own usable memory. Verification labels may
  cease to describe internal access after memory damage.
- **Supervision:** agent and reporter training are separated, but the reporters
  receive explicit report labels. Successful decoding is not spontaneous self-report.
- **Generalization:** both tasks share a six-value generator, and only two small
  recurrent architectures were tested. Primary report accuracy does not establish
  robustness to arbitrary delays, lesions, or distribution shifts.
- **Comparisons:** the omitted-identity correction reused test data. It repairs
  interpretation of superiority but is not a fresh confirmatory experiment.
- **Uncertainty:** three seeds and a five-fit shuffled-label null provide limited
  evidence about variability beyond the measured systems.
- **Consciousness:** neither accurate reports nor causal report/control coupling
  establishes subjective experience. Conversely, lack of a necessity or superiority
  result is not failure on the narrower reporting-accuracy goal.

The next reporting questions are whether agents can learn native reports of these
informational distinctions, whether unavailable-content reports can be made
reliable in the earlier paired-history setting, and whether report accuracy
survives controlled memory damage and broader generalization. These are optional
extensions of the completed evaluation.

## 6. Reproducibility

The [reproduction guide](FINAL_REPRODUCTION.md) provides installation and the full
command matrix. The tested environment uses Python 3.12.3, PyTorch 2.5.1+cpu, and
NumPy 2.4.3. No paid model APIs are required for the final study.

From the repository root, after installing the documented environment:

```bash
# Restore the archived checkpoints and replay the final metrics.
.venv/bin/python scripts/reproduce_final.py --verify

# Inspect the registered reproduction commands.
.venv/bin/python scripts/reproduce_final.py --plan

# Retrain the full matrix; use a separate checkout to compare with the reference.
.venv/bin/python scripts/reproduce_final.py --run --jobs 3
```

The committed archive contains 42 checkpoint files. The completion manifest
records 55 primary artifacts, checksums, gate outcomes, and limitations. All
saved final controller, reporter, intervention, and stress metrics were replayed
from checkpoints at study closeout, and the full suite passed 149 tests. Those
are recorded experiment-validation results; this manuscript revision changes
only documentation and does not rerun or alter the experiments.

The [progress record](COMPLETION_PROGRESS.md) and [final results](PROJECT_RESULTS.md)
provide the evidence trail. Earlier protocols, unsuccessful experiments, the
comparator correction, and the original Stage 8 verdict remain preserved.

## 7. Conclusion

Small task-trained recurrent systems support accurate reports of verified content,
unverified status, and remembered sensor quality through separately trained
readouts. GRU reporting is consistently accurate across the tested tasks and
seeds; routing RNN quality reporting is less consistent. A simpler comparator
matches or exceeds the full-state reports, and selective interventions show
that remembered quality can influence both reports and behavior.

The reporting goal therefore has bounded positive support. The remaining boundary
is between accurate external readouts and an agent that independently learns to
report its own informational state reliably across conditions. Necessity for task
performance and broader consciousness claims are separate questions.
