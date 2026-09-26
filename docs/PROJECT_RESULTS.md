# Final research results

**The bounded research prototype and final evaluation are complete.** The study
supports learned information acquisition and causal prospective representations
in small recurrent systems. It does **not** establish comparator-resistant
conscious-access monitoring, subjective experience, or Stage 8 convergence.

Completion follows the [registered stopping rule](COMPLETION_PROTOCOL.md): run
the finite matrix, retain failures, verify the artifacts, and close unsupported
promotion paths. It does not mean that every broader scientific hypothesis in
the historical roadmap has been demonstrated.

## What was completed

- Two prospective tasks: sequential verification and choosing between sensors.
- Two recurrent architectures: GRU48 and nearly parameter-matched RNN87.
- Three fresh seeds and matched state, confidence, and confidence-with-quality-cue
  controls: 36 fitted controllers.
- From-scratch, reward-only exploration with native answer/decline decisions:
  12 additional GRU controllers across both tasks and all seeds.
- Independent reports and answer-preserving interventions on 12 fitted state
  systems, plus a corrected identity-preserving reporting comparator.
- Disjoint reserved-context evaluations with longer delays and unannounced sensor
  degradation, without refitting.
- A validated manifest of 55 primary artifacts and 42 checkpoint files. All saved
  controller, report, causal, and stress metrics were recomputed from checkpoints.
- A reproduction entry point, tested dependency versions, checkpoint archive,
  smoke pipeline, and 149 passing unit tests.

The [machine-readable completion audit](../audits/project_completion.json)
contains every gate, source/data/checkpoint checksum, correction, and limitation.
The [progress record](COMPLETION_PROGRESS.md) links the individual milestones.

## Claim and evidence map

| Question | Result | Meaning |
| --- | --- | --- |
| Can recurrent agents acquire information adaptively? | Supported in both fitted-GRU tasks across all seeds/costs against fixed policies. | Learned task-level control, not merely a passive reporter. |
| Can acquisition be learned from experienced rewards? | Yes, in the finite exploration experiments; performance and decline behavior vary. | No full answer-label vector or optimal-inspection target enters that learner's loss. |
| Does prospective quality affect behavior independently of initial answer logits? | Yes, clearly in the routing GRUs and in later serial verification. | The remembered environmental cue causally influences control. |
| Do independent reports read a representation used by control? | Bounded positive evidence from quality transplants and restoration. | Shared prospective task information; not proof of a higher-order or conscious-access representation. |
| Does full-state control consistently beat a fair cue-informed confidence policy? | No registered task/architecture group passes all costs and seeds. | Stronger full-state advantage remains unsupported. |
| Does full-state reporting beat a fair content-and-quality comparator? | No corrected comparison reaches the required margin; the corrected comparator matches or exceeds the state readout's primary score in all 12 systems. | Original apparent superiority was confounded by omitted answer identity. |
| Are results robust to architecture and distribution shifts? | Mixed. RNN policies are weaker/variable; delays and misspecified sensors can reduce reward. | No broad architecture- or domain-independent mechanism claim. |
| Does this justify independent-content or Stage 8 promotion? | No. The preregistered prerequisite fails. | Earlier unforced-convergence failures and Stage 8's3 pass/5 partial verdict remain unchanged. |

## Main results

Fitted GRUs beat every fixed inspection policy by the registered margin at every
cost and seed in both tasks. In serial verification, state returns span
0.8562–0.8675,0.6846–0.7107,and0.5291–0.5667 at costs 0.1,0.25,and0.4. In routing,
they span0.8509–0.8643,0.7509–0.7643,and0.6509–0.6643. None of those GRU comparisons
reaches the 0.02 fair-cue advantage across the required cells. The RNN yields
occasional favorable comparisons but no replicated complete support gate.

Reward-only exploration learns from chosen-action rewards and costs. It can
acquire, answer, and sometimes decline for a fixed payoff. This tests finite
exploration and a native communicative action. It does not establish natural
language introspection, and a decline can reflect ordinary payoff-sensitive
confidence. Objectives and budgets differ from fitted control, so their numbers
are not a controlled algorithm-superiority comparison.

The strongest causal result is in routing GRUs. A fitted quality transplant in
the answer head's null space changes89.1–100% of sensor choices, versus0.3–4.4%
for matched random directions. Quality reports change 86.7–100%. Initial answer
logits remain unchanged within1e-5, and restoration returns decisions and reports
to baseline. The actual sensor remains unchanged, so the wrong remembered cue
reduces reward. This establishes a causal prospective representation inside this
task; the cue-informed comparator still prevents a stronger introspection claim.

Serial GRU transplants leave initial inspection decisions unchanged but alter
36.1–58.6% of later verification trajectories. RNN effects are less selective,
and some random perturbations improve their weak baseline policies. Sensitivity
alone must therefore not be interpreted as a beneficial access-monitoring system.

## Reporting correction

The initial registered confidence-plus-quality reporter omitted answer identity,
although the target included the verified value. Its apparent disadvantage was
therefore not a fair test of reporting superiority. This was identified during
analysis and [documented explicitly](COMPLETION_CORRECTIONS.md).

The original artifacts and gates are preserved. An additional reporter receives
all six answer logits plus the same quality cue, with identical capacity,
partitions, optimization, and validation selection. It matches or exceeds the
state reporter's primary score in all 12 systems. This correction reuses test
data and is a diagnostic, not independent confirmation of a new positive claim.
The control-policy comparison did not omit the answer logits used for answering.

## Robustness and limits

All reserved contexts are disjoint from training, validation, report fitting, and
primary test contexts. Longer delays and unannounced sensor degradation often
reduce reward. Five of 108 individual stress seed/cost cells pass all gates, but
no task satisfies the full promotion requirement across costs, seeds, and
architectures. These exceptions are retained, not used to select a new headline.
Analytic policies know the degraded reliability; learned policies see the old cue.

Quality is explicitly supplied in an environmental cue; remembering it does not
by itself establish a representation of the model's own knowledge state. Both
tasks share a six-value synthetic generator. The architecture comparison
covers two small recurrent families, not pretrained language models or broad
domain generalization. Report labels concern environmental quality and source
verification, not necessarily an agent's usable internal access after arbitrary
memory damage. Context-bootstrap intervals are pointwise and conditional on a
trained model. The five-fit shuffled-report null is descriptive, not a formal
significance test.

The earlier paired-history reporting failures remain failures. Stage 8 is still
not met. The final study closes with these limitations rather than retuning
thresholds until a positive claim appears. Broader scientific extensions are
new research, not unfinished deliverables from this bounded evaluation.

## Verify or reproduce

See [FINAL_REPRODUCTION.md](FINAL_REPRODUCTION.md). Fast verification restores
the committed checkpoint archive and replays every final metric; full reproduction
retrains all registered controllers and reporters. No paid model APIs are needed.
