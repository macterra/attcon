# Reporting an Attention-Control Model: Faithful Counterfactual Reports and the Qualia Hypothesis

Working preprint, updated 2026-09-27. The preceding inspection-model manuscript is
[archived unchanged](INSPECTION_MODEL_PREPRINT.md). All prior failed results remain
part of the project record.

> Update in progress: the active [object-linked study](BOUND_CONTENT_PROGRESS.md)
> adds represented colored shapes and a separate free-prose audit. Its first two
> confirmations failed; fresh confirmation v3 is running. The manuscript below
> describes the earlier numerical study, [archived here](PREDICTIVE_NUMERICAL_PREPRINT.md).

## Abstract

We test an observable consequence proposed for the hypothesis that an attention-
control model is the source of qualia: accurate reports of that model should show
specified structure of consciousness reports. We distinguish identifying the
model, reporting its state faithfully, and independently assessing report
character. An initial inspection-model study achieved only 82.36–88.80% exact
learned-report accuracy and failed intervention criteria. We therefore constructed
a predictive attention model trained on allocation and reconstruction outcomes,
without phenomenological labels. In three fresh confirmations, allocation accuracy
is 100%, command-effect prediction accuracy 99.988–99.998%, and access-probability
mean absolute error 0.0232–0.0240. Its command-effect predictions guide attention;
interventions change choices and exact restoration recovers them. After retained
reporting failures and an explicitly registered interface correction, a fixed
language reporter produces correct structured commitments on 240/240 reports,
including 72/72 selective-intervention pairs and 24/24 restorations. These reports
cover twelve underlying episodes, three independently trained models, ten
conditions, and neutral versus first-person styles. Physical forecasts and a
matched history predictor each match the actual model's complete commitments on
2/24 cases. Independent prose-fidelity and consciousness-report assessment remains
pending. The experiment establishes faithful, counterfactually controlled reporting
of an identified attention model within this engineered setting; it does not yet
establish the proposed phenomenological correspondence or settle the theory.

## 1. Question and evidence standard

The question is whether reports accurately describe the attention-control model's
state in ways that correspond to consciousness reports. It is not whether
conscious access is necessary for task performance, whether the model beats a
simpler same-state decoder, or whether experience can be proven experimentally.
Engineering the architecture and report interface is permitted, with its role
made explicit.

The attention-schema account motivates a simplified representation of the
agent–attention–object relation and dissociations between the process and its
modeled state. [Webb and Graziano (2015)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00500/full).
Human perceptual-awareness research motivates distinguishing graded reported
clarity from mere task correctness. [Early Local Activity in Temporal Areas
Reflects Graded Content of Visual Perception (2016)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.00572/full).
Neither source identifies simulated reconstruction probability with qualia.
Our report distinctions are proposed operational correspondences, not established
equivalences to human experience.

## 2. Earlier inspection-model study

We identified a learned inspection-history model that contributes directly to
attention selection in three previously trained recurrent controllers. Fifteen
reporters partly recovered its state, including mistaken model beliefs, but exact
complete reports and robustness to isolated interventions failed registered
criteria. Direct telemetry was exact, while authored report expressions supplied
no independent consciousness-report evidence. The [archived manuscript](INSPECTION_MODEL_PREPRINT.md),
[results](ATTENTION_MODEL_RESULTS.md), and [protocol](ATTENTION_MODEL_PHENOMENOLOGY_PROTOCOL.md)
retain the methods, failures, and numerical correction. The following study is a
new architecture and assay, not a retrospective reclassification of those results.

## 3. Predictive model, task, and control

A two-channel simulator has four object slots per channel. One channel responds
to the agent's commands; the other follows an independent cyclic allocation.
Sampled signal strength determines an object's reconstruction probability;
unattended information decays by 0.75 per step. Object identifiers V are separate
from the recurrent attention model A. This is a toy reconstruction process, not
natural perceptual experience or a probe of the language reporter's own attention.

A GRU64 receives allocations, sampled strengths, and commands. It predicts current
allocation, recoverability now and after two unattended delays, and next allocation
under each alternative command. A policy selects commands from those predicted
effects. Only simulator process targets train A; no consciousness-report targets
enter that training. The policy consumes the command-effect component, while
access and retention share the model without independent causal necessity for
this policy. Full state, history, physical truth, and V are archived separately.

Three development models (811/821/831) precede frozen confirmation models
901/911/921. Each trains for 1200 updates with the same architecture and optimizer.
The [internal confirmation protocol](PREDICTIVE_CONFIRMATION_PROTOCOL.md) fixes
fresh contexts and acceptance criteria. All three models pass every gate and
replay exactly. In separate development closed-loop assays, ordinary commands
select queried controlled slots at 100%; rotating the modeled command table
reduces this to 0%. Restoring the table reproduces the entire trajectory exactly.
A subsequent [frozen closed-loop confirmation](PREDICTIVE_CLOSED_LOOP_CONFIRMATION.md)
on the three exact reported models also passes all selection, rotation, and
exact-restoration gates on fresh episodes. These tests establish functional
identity; task advantage is not a qualia gate.

## 4. Report interface and retained corrections

A frozen `gpt-5-mini-2025-08-07` reporter receives a target descriptor and model
predictions. The initial numerical-array pilot fails through indexing errors and
incomplete responses. A style-only pilot does not repair fidelity. A deterministic
target-specific interface then exposes allocation probability, the channel maximum,
recoverability, and command-dependent variation under explicit labels. It does
not supply experiential labels or sample consciousness reports.

The first explicit-interface confirmation passes all ordinary/intervention
commitments but fails constant-state uncertainty handling. The correction makes
the unknown/tie branch explicit and is tested on new contexts under unchanged
scoring criteria. The [v1 protocol](PREDICTIVE_REPORT_CONFIRMATION.md) and
[v2 correction](PREDICTIVE_REPORT_CONFIRMATION_V2.md) preserve this sequence.
All failed versions, API errors, and responses remain archived.

Ten conditions include ordinary A, three isolated state interventions, restoration,
physical forecasts, an independently trained history predictor, shuffled A,
constant A, and V alone. V and uninvolved fields remain fixed under intervention.
The history predictor has the same capacity and training budget and receives the
same observations; it is another process model, not a model-free alternative.
Two styles share identical facts: neutral prose and explicitly requested ordinary
first-person prose. That instruction manipulates style and cannot itself provide
evidence for consciousness-report correspondence.

## 5. Results

| Registered internal-model measure | Range over three confirmation seeds |
| --- | --- |
| Allocation accuracy | 100% |
| Alternative-command effect accuracy | 99.988–99.998% |
| Controllable-channel identification | 99.984–100% |
| Recoverability MAE | 0.0232–0.0240 |
| Maximum recoverability error | 0.2143–0.2395 |
| Brier loss above simulator-probability oracle | 0.00116–0.00159 |

In report confirmation v2, all 240 structured commitments are correct relative to
their supplied state, with 96/96 primary reports, 72/72 correct intervention pairs,
and 24/24 exact restorations. Constant and missing inputs produce the stipulated
uncertainty. The physical and history-predictor controls each match actual A's
complete commitments on 2/24 cases; shuffled and constant controls match 0/24.
Both controls match A's focality and responsive-channel categories on 24/24
reports; the low complete-match rate comes from continuous access estimates
under the strict 0.01 tolerance. It is not evidence of unique qualitative
structure or equally large phenomenological differences. The controls'
reports remain accurate about their own supplied forecasts.

The report sample contains twelve underlying episodes, not 240 independent
situations. Prompt and intervention observations are correlated; replication over
three seeds does not establish broad task or architecture generalization. All
168 code tests pass. Offline verification checks complete provenance, target-view
derivation, exact re-scoring, and all three confirmed checkpoint replays.

## 6. Interpretation and remaining crux

The identified model determines report content in a way that survives isolated
counterfactual changes. The reporter can describe A even when its state disagrees
with the physical process. This is stronger evidence of actual model-state
reporting than the previous inspection-model decoder supplied.

However, structured-commitment accuracy does not establish free-text fidelity or
consciousness-report character. The vocabulary comes partly from the language
model's prior and, in the styled condition, an explicit first-person instruction.
A functional reconstruction model plus an ordinary decoder may explain all
observed results. The same-state simpler decoder is not a disqualifying rival;
the unresolved issue is whether the faithfully expressed relations have the
independently motivated structure at issue in the theory.

The [blinded rubric](PREDICTIVE_REPORT_RUBRIC.md) and [review form](report-review.html)
separate manner of access, graded presentation, agent–object relation, and prose
fidelity from first-person wording and generic experience claims. Independent
ratings have not been received. We therefore do not claim the overall research
goal is achieved. A new criterion selected after seeing ratings would need fresh
confirmation, and any subsequent failure must remain in the record.

## 7. Availability

[Full results](PREDICTIVE_ATTENTION_RESULTS.md), [offline reproduction](PREDICTIVE_ATTENTION_REPRODUCTION.md),
and [current progress](PREDICTIVE_ATTENTION_PROGRESS.md) link the complete study.
All models, simulator code, interventions, API requests/responses, failed versions,
and review materials are in the repository. Reproducing the recorded mechanical
results requires no new API calls. Hosted-model reruns are new observations and
are not promised to be byte-identical.
